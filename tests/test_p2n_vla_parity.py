"""M1 exit check: full-size fp32 parity of the ``oat.model.vla`` PI0.5 port against LeRobot's PI0.5.

The reference tensors come from LeRobot ``PI05Pytorch`` (lerobot 2577da0e, transformers 5.5.4, torch 2.11),
dumped in true IEEE fp32 by ``scripts/p2n_vla_parity/dump_pi05_reference.py``:

- ``output/parity/pi05_reference_fp32.pt``, from CUDA on an RTX 4090. The ``[cuda]`` variants compare against it.
- ``output/parity/pi05_reference_fp32_cpu.pt``, the CPU twin. The ``[cpu]`` variants compare against it.

The stack is built exactly as the ``pi05_ki_flow`` baseline builds it, but in fp32 throughout:
``SiglipEncoder(SIGLIP_SO400M, 2048, fp32)``, ``GemmaJoint(GEMMA_2B, GEMMA_300M, 'adarms', frozen_dtype=fp32)``
and ``FlowHeads(1024)``, loaded with ``load_pi05(mode='flow')``. The tests feed it the dumped inputs and check,
on each device:

- End to end, from raw images and token ids: image tokens, text embeddings (bitwise), positions and masks, the
  final prefix hidden states, post-RoPE K/V of all 18 layers, tied-head logits, the time conditioning, v_t and
  the 10-step ``sample_actions``.
- Stage by stage, where each stage gets the reference's own upstream tensors so that it is isolated:
  - the VLM, on the dumped prefix embeddings, with SDPA forced to its math backend (LeRobot's eager arithmetic);
  - the expert and flow head, on the dumped K/V: v_t, the velocity at every Euler step of the dumped trajectory,
    and the actions.
- p2n mode: ``GemmaJoint('segment')`` loaded with ``mode='p2n', t0=0.6`` reproduces the adaRMS expert driven by
  ``cond = c(0.6)`` on the same prefix. It also reproduces LeRobot's own velocity at t = 0.6, which is Euler step 4.

The dump's conventions, reproduced here:
- Camera slots are [base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb]. The third slot is all -1 and masked.
- SigLIP runs once over all B*3 images in slot-major order, as LeRobot's eval-mode ``_embed_images`` does.
  Calling it per slot changes the CUDA GEMM shapes and moves the image tokens by about 2e-4 abs.
- The prefix is [3x256 image tokens | 64 right-padded prompt ids], positions are cumsum(valid)-1 and
  pos0 = number of valid tokens. Logits are taken at the last 4 valid prompt positions.
- The suffix is 50 bidirectional action tokens at pos0 + k. Sampling is 10 Euler steps from t = 1 with dt = -0.1.

Known differences, all quantified by the stage checks and far below the tolerances:
1. LeRobot's pad query rows (prompt pads and the empty camera) attend uniformly to every key. Ours attend to the
   valid keys. Hidden rows and the K/V of layers >= 1 are therefore compared on valid positions only. Layer 0 is
   compared at every position.
2. Our VLM uses SDPA, the plan's memory-efficient path; LeRobot uses eager attention. With SDPA forced to the math
   backend, the VLM is bitwise equal on the reference machine. The default kernels differ by about 4e-6 relative.
3. On CUDA, /venv/oat's cuBLAS 12.9 picks different kernels from the dump's cuBLAS 12.8 for the expert's small
   GEMMs, about 3e-7 relative on v_t. The same code under the dump's torch and cuBLAS is bitwise equal.

The port had one genuine discrepancy, now fixed in ``gemma_joint.rope_cos_sin``: RoPE inverse frequencies were
built on the positions' device, and CUDA's ``pow`` is 1 ulp off in 4 of 128 entries. That cost up to 2e-6 rad at
position 560 and ruled out bitwise post-RoPE keys on GPU. The two fast tests at the top guard the fix and need no
weights.

Run from the repository root. The weight-backed tests take about 2 minutes, about 30 GB of host RAM and about
15 GiB of GPU memory:

    CUDA_VISIBLE_DEVICES=1 /venv/oat/bin/python -m pytest tests/test_p2n_vla_parity.py -q -s

``-m "not gpu"`` keeps only the CPU variants, and ``-m "not requires_pi05"`` keeps only the fast RoPE tests.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
import gc
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest
import torch
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel

from oat.model.vla.flow_head import FlowHeads, flow_condition, flow_velocity, posemb_sincos, sample_actions
from oat.model.vla.gemma_joint import AdaRMSNorm, GemmaJoint, SegmentRMSNorm, rope_cos_sin, rope_inv_freq
from oat.model.vla.layout import (N_RAWDIFF_SLOTS, N_SEGMENTS, NEG, build_flow_suffix_layout, build_prefix_layout,
                                  build_suffix_layout)
from oat.model.vla.pi05_checkpoint import (EXPERT_LM_HEAD, FLOW_HEAD_NAMES, PI05_SHA256, LoadReport,
                                           default_pi05_path, load_pi05)
from oat.model.vla.siglip import SiglipEncoder
from oat.model.vla.specs import GEMMA_2B, GEMMA_300M, SIGLIP_SO400M, TINY_VLM

REPO_ROOT = Path(__file__).resolve().parents[1]
PARITY_DIR = REPO_ROOT / "scripts" / "p2n_vla_parity"
REFERENCES = {
    "cuda": REPO_ROOT / "output" / "parity" / "pi05_reference_fp32.pt",
    "cpu": REPO_ROOT / "output" / "parity" / "pi05_reference_fp32_cpu.pt",
}
DUMP_COMMANDS = {
    "cuda": "CUDA_VISIBLE_DEVICES=1 HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1 /venv/lerobot_ref/bin/python "
            "scripts/p2n_vla_parity/dump_pi05_reference.py",
    "cpu": "HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1 /venv/lerobot_ref/bin/python "
           "scripts/p2n_vla_parity/dump_pi05_reference.py --device cpu "
           "--output output/parity/pi05_reference_fp32_cpu.pt",
}
GPU_FREE_BYTES_NEEDED = 17 * 2**30      # fp32 flow stack (13 GiB) + p2n expert (1.3 GiB) + activations
P2N_T0 = 0.6


def pi05_slow(test):
    """Marks for the tests that load the 14.5 GB pi05_base weights."""
    return pytest.mark.requires_pi05(pytest.mark.slow(test))


# Tolerance ``tol`` means |actual - ref| <= tol * max|ref| + tol * |ref| elementwise (rtol = tol and
# atol = tol * max|ref|). fp32 rounding errors scale with the magnitude of the summands, so the absolute
# tolerance follows each tensor's scale. Each value is a few times the larger of the CUDA and CPU errors
# measured on the reference machine (the values are printed by ``test_zz_parity_report``). fp32 epsilon is 1.2e-7.
TOL = {
    # end to end: raw images + prompt ids -> ... (our default SDPA kernels)
    "image_embeddings": 2e-6,          # measured 0 (bitwise) on CUDA and CPU
    "image_embeddings_per_slot": 1e-5,  # measured 1.6e-6 on CUDA (different GEMM shapes), 0 on CPU
    "prefix_hidden": 2e-5,             # measured 3.7e-6 on CUDA, 1.6e-6 on CPU
    "prefix_kv": 5e-5,                 # per layer, relative to that layer's max; measured up to 8.9e-6
    "vlm_logits": 2e-5,                # measured 5.2e-6
    "time_embedding": 1e-6,            # measured 0
    "adarms_cond": 1e-6,               # measured 0
    "v_t": 5e-6,                       # measured 4.0e-7
    "sample_actions": 2e-5,            # measured 4.1e-6
    # stage by stage: each stage fed the reference's own inputs
    "stage_vlm_math": 1e-6,            # measured 0 (bitwise) for hidden, every layer's K/V and the logits
    "stage_expert": 2e-6,              # measured 3.4e-7 on CUDA (cuBLAS 12.9 vs 12.8), 0 on CPU
    # p2n fold
    "p2n_c0": 1e-6,                    # measured 0 on CPU, 9e-8 on CUDA (c0 is folded on the CPU)
    "p2n_fold": 2e-6,                  # fp32 Dense(c0) against float64; measured 3.4e-7
    "p2n_expert": 5e-6,                # measured 0 on CPU, 1.3e-6 on CUDA
}


# ----------------------------------------------------------------------------- helpers
def _diff_stats(actual: Tensor, expected: Tensor) -> Dict[str, float]:
    a, e = actual.detach().to("cpu", torch.float64), expected.detach().to("cpu", torch.float64)
    if a.shape != e.shape:
        raise AssertionError(f"shape mismatch {tuple(a.shape)} vs {tuple(e.shape)}")
    diff = (a - e).abs()
    scale = float(e.abs().max()) if e.numel() else 0.0
    max_abs = float(diff.max()) if diff.numel() else 0.0
    return {"max_abs": max_abs, "ref_absmax": scale, "max_rel": max_abs / scale if scale else max_abs,
            "rel_fro": float((a - e).norm() / e.norm()) if float(e.norm()) else float((a - e).norm()),
            "bitwise": bool(torch.equal(actual.detach().cpu(), expected.detach().cpu()))}


def _assert_parity(name: str, actual: Tensor, expected: Tensor, tol: float,
                   report: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, float]:
    """Record the error statistics of ``actual`` against ``expected`` and assert scale-aware fp32 closeness."""
    stats = _diff_stats(actual, expected)
    if report is not None:
        report[name] = dict(stats, tol=tol)
    actual = actual.detach().to("cpu", torch.float32)
    expected = expected.detach().to("cpu", torch.float32)
    torch.testing.assert_close(
        actual, expected, rtol=tol, atol=tol * stats["ref_absmax"],
        msg=lambda message: (f"{name}: fp32 parity failed (tol {tol:.1e}): max_abs {stats['max_abs']:.3e}, "
                             f"max_rel {stats['max_rel']:.3e}, rel_fro {stats['rel_fro']:.3e}\n{message}"))
    return stats


@contextlib.contextmanager
def _ieee_fp32():
    """Disable TF32 for CUDA matmuls and cuDNN convolutions, then restore the previous settings exactly.

    torch leaves cuDNN convolutions on TF32 by default, which moves SigLIP's patch embedding by about 1e-3
    relative. The reference was dumped with every fp32 path in IEEE mode.
    """
    backends = torch.backends
    if hasattr(backends, "fp32_precision"):          # torch >= 2.9 API
        # Only the new API is used, and it is restored leaf by leaf. Inside the context, torch's legacy
        # ``cudnn.allow_tf32`` getter refuses to answer for a mix of legacy and new flags. Nothing here reads it.
        leaves = [(backends, "fp32_precision"), (backends.cuda.matmul, "fp32_precision"),
                  (backends.cudnn, "fp32_precision"), (backends.cudnn.conv, "fp32_precision"),
                  (backends.cudnn.rnn, "fp32_precision")]
        saved = [(owner, name, getattr(owner, name)) for owner, name in leaves]
        try:
            for owner, name in leaves:
                setattr(owner, name, "ieee")
            yield
        finally:
            for owner, name, value in reversed(saved):
                setattr(owner, name, value)
    else:                                           # legacy flags
        saved = (backends.cuda.matmul.allow_tf32, backends.cudnn.allow_tf32)
        try:
            backends.cuda.matmul.allow_tf32 = False
            backends.cudnn.allow_tf32 = False
            yield
        finally:
            backends.cuda.matmul.allow_tf32, backends.cudnn.allow_tf32 = saved


def _load_reference_spec():
    """The dump's schema validator (``scripts/p2n_vla_parity/pi05_reference_spec.py``; torch-only, no lerobot)."""
    path = PARITY_DIR / "pi05_reference_spec.py"
    if not path.is_file():
        pytest.skip(f"SKIPPED LOUDLY: parity reference helpers missing at {path}")
    module_spec = importlib.util.spec_from_file_location("p2n_vla_pi05_reference_spec", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def _require_weights() -> Path:
    path = default_pi05_path()
    if not path.is_file():
        pytest.skip(f"SKIPPED LOUDLY: pi05_base weights missing at {path} (run scripts/fetch_p2n_vla_assets.py)")
    return path


def _load_reference(device: str) -> dict:
    path = REFERENCES[device]
    if not path.is_file():
        pytest.skip(f"SKIPPED LOUDLY: LeRobot fp32 reference for {device} missing at {path}; "
                    f"produce it with: {DUMP_COMMANDS[device]}")
    reference = torch.load(path, map_location="cpu", weights_only=True)
    _load_reference_spec().validate_reference(reference)
    return reference


def _camera_layout(reference: dict) -> Tuple[int, int, int]:
    """(batch, camera slots, tokens per image) of the dump."""
    batch, slots = reference["inputs"]["img_masks"].shape
    return batch, slots, int(reference["metadata"]["n_tokens_per_image"])


def _siglip_like_lerobot(siglip: SiglipEncoder, images: Tensor) -> Tensor:
    """LeRobot's eval-mode ``_embed_images``: one SigLIP call over every slot (slot-major), then
    ``chunk`` per slot and concatenate along tokens. images [B, n_slots, 3, S, S] -> [B, n_slots*T, W]."""
    batch, slots = images.shape[:2]
    tokens = siglip(images.transpose(0, 1).reshape(slots * batch, *images.shape[2:]))
    per_image, width = tokens.shape[1:]
    return tokens.reshape(slots, batch, per_image, width).transpose(0, 1).reshape(batch, slots * per_image, width)


# ----------------------------------------------------------------------------- fast: the RoPE fix
@pytest.mark.parametrize("head_dim", [GEMMA_2B.head_dim, TINY_VLM.head_dim])
def test_rope_inv_freq_is_the_transformers_cpu_table(head_dim):
    """transformers (hence LeRobot) builds RoPE's inverse frequencies on the CPU, and so do we, on every device."""
    from transformers import GemmaConfig
    from transformers.models.gemma.modeling_gemma import GemmaRotaryEmbedding

    config = GemmaConfig(vocab_size=128, hidden_size=8 * head_dim, intermediate_size=64, num_hidden_layers=1,
                         num_attention_heads=8, num_key_value_heads=1, head_dim=head_dim, rope_theta=10000.0)
    reference = GemmaRotaryEmbedding(config).inv_freq
    ours = rope_inv_freq(head_dim, 10000.0)
    assert ours.dtype == torch.float32 and ours.device.type == "cpu"
    assert torch.equal(ours, reference)


@pytest.mark.gpu
def test_rope_tables_agree_across_devices():
    """With CPU-built inverse frequencies, CUDA tables differ from CPU tables only by cos/sin rounding (1 ulp).
    Building them on CUDA cost up to 3.8e-6 at positions below 1024 for head_dim 256."""
    if not torch.cuda.is_available():
        pytest.skip("SKIPPED LOUDLY: CUDA is not available")
    positions = torch.arange(1024)[None].expand(2, -1)
    cos_cpu, sin_cpu = rope_cos_sin(positions, GEMMA_2B.head_dim, GEMMA_2B.rope_theta)
    cos_gpu, sin_gpu = rope_cos_sin(positions.cuda(), GEMMA_2B.head_dim, GEMMA_2B.rope_theta)
    assert cos_gpu.device.type == "cuda" and cos_gpu.dtype == torch.float32
    worst = max(float((cos_gpu.cpu() - cos_cpu).abs().max()), float((sin_gpu.cpu() - sin_cpu).abs().max()))
    assert worst <= 2.5e-7, worst      # measured 6e-8


# ----------------------------------------------------------------------------- fixtures
@dataclass
class FlowStack:
    siglip: SiglipEncoder
    joint: GemmaJoint
    heads: FlowHeads
    report: LoadReport

    def to(self, device) -> "FlowStack":
        for module in (self.siglip, self.joint, self.heads):
            module.to(device).eval()
        return self


@dataclass
class P2NStack:
    joint: GemmaJoint
    report: LoadReport


@dataclass
class ParityRun:
    device: torch.device
    reference: dict
    outputs: Dict[str, object]                       # CPU fp32 tensors in the reference layout
    prefix_kv: Tuple[Tuple[Tensor, Tensor], ...]     # our end-to-end prefix K/V, on ``device``
    pos0: Tensor                                     # [B] on ``device``
    prefix_valid: Tensor                             # [B, P] on ``device``
    report: Dict[str, Dict[str, float]] = field(default_factory=dict)


@pytest.fixture(scope="module")
def flow_stack():
    weights = _require_weights()
    siglip = SiglipEncoder(SIGLIP_SO400M, GEMMA_2B.width, frozen_dtype=torch.float32)
    joint = GemmaJoint(GEMMA_2B, GEMMA_300M, "adarms", frozen_dtype=torch.float32, activation_checkpointing=False)
    heads = FlowHeads(GEMMA_300M.width)
    report = load_pi05(weights, siglip=siglip, joint=joint, mode="flow", flow_heads=heads)
    stack = FlowStack(siglip, joint, heads, report).to("cpu")
    yield stack
    del stack
    gc.collect()


@pytest.fixture(scope="module")
def p2n_stack():
    weights = _require_weights()
    siglip = SiglipEncoder(SIGLIP_SO400M, GEMMA_2B.width, frozen_dtype=torch.float32)
    joint = GemmaJoint(GEMMA_2B, GEMMA_300M, "segment", frozen_dtype=torch.float32, activation_checkpointing=False)
    report = load_pi05(weights, siglip=siglip, joint=joint, mode="p2n", t0=P2N_T0)
    del siglip          # only needed to account for every checkpoint tensor
    joint.eval()
    yield P2NStack(joint, report)
    del joint
    gc.collect()


def _gpu_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("SKIPPED LOUDLY: CUDA is not available (pin a free GPU with CUDA_VISIBLE_DEVICES)")
    device = torch.device("cuda", torch.cuda.current_device())
    free, _ = torch.cuda.mem_get_info(device)
    if free < GPU_FREE_BYTES_NEEDED:
        pytest.skip(f"SKIPPED LOUDLY: {torch.cuda.get_device_name(device)} has {free / 2**30:.1f} GiB free, "
                    f"the fp32 parity stack needs {GPU_FREE_BYTES_NEEDED / 2**30:.0f} GiB "
                    "(pin a free GPU with CUDA_VISIBLE_DEVICES)")
    return device


@torch.no_grad()
def _run_flow_stack(stack: FlowStack, reference: dict, device: torch.device) -> ParityRun:
    siglip, joint, heads = stack.siglip, stack.joint, stack.heads
    meta = reference["metadata"]
    inputs = {key: value.to(device) for key, value in reference["inputs"].items()}
    batch, slots, per_image = _camera_layout(reference)
    steps = int(meta["num_inference_steps"])
    cpu = lambda tensor: tensor.detach().to("cpu", torch.float32)  # noqa: E731

    # ---- end to end, the policy's path (default SDPA kernels)
    images = inputs["images"]
    image_tokens = _siglip_like_lerobot(siglip, images)
    image_tokens_per_slot = torch.cat([siglip(images[:, slot]) for slot in range(slots)], dim=1)
    image_valid = inputs["img_masks"].repeat_interleave(per_image, dim=1)
    text = joint.embed_text(inputs["lang_tokens"])
    embeds = torch.cat((image_tokens, text), dim=1)
    layout = build_prefix_layout(image_valid, inputs["lang_masks"])
    prefix = joint.prefix_forward(embeds, layout)
    rows = torch.arange(batch, device=device)[:, None]
    logit_index = reference["vlm_logits_prefix_index"].to(device)
    logits = joint.vlm_logits(prefix.hidden[rows, logit_index])
    t = inputs["t"]
    v_t = flow_velocity(joint, heads, prefix.kv, prefix.pos0, layout.nonki_valid, inputs["x_t"], t)
    actions = sample_actions(joint, heads, prefix.kv, prefix.pos0, layout.nonki_valid, inputs["noise"],
                             num_steps=steps)

    # ---- stage: the VLM alone, on the reference prefix embeddings, with LeRobot's eager arithmetic
    with sdpa_kernel([SDPBackend.MATH]):
        prefix_math = joint.prefix_forward(reference["prefix_embeddings"].to(device), layout)
    logits_math = joint.vlm_logits(prefix_math.hidden[rows, logit_index])

    # ---- stage: the expert and flow head alone, on the reference K/V
    kv_ref = tuple((k.to(device), v.to(device)) for k, v in zip(reference["prefix_keys"], reference["prefix_values"]))
    valid_ref = reference["prefix_pad_masks"].to(device)
    pos0_ref = valid_ref.sum(dim=1)
    v_t_ref_kv = flow_velocity(joint, heads, kv_ref, pos0_ref, valid_ref, inputs["x_t"], t)
    trajectory, times = reference["sample_trajectory"].to(device), reference["sample_times"].to(device)
    step_velocities = torch.stack([
        flow_velocity(joint, heads, kv_ref, pos0_ref, valid_ref, trajectory[step], times[step].expand(batch))
        for step in range(steps)])
    actions_ref_kv = sample_actions(joint, heads, kv_ref, pos0_ref, valid_ref, inputs["noise"], num_steps=steps)

    outputs = {
        "image_embeddings": cpu(image_tokens), "image_embeddings_per_slot": cpu(image_tokens_per_slot),
        "text_embeddings": cpu(text), "prefix_positions": layout.positions.cpu(), "pos0": prefix.pos0.cpu(),
        "prefix_valid": layout.valid.cpu(), "prefix_bias": layout.bias.cpu(),
        "prefix_hidden": cpu(prefix.hidden), "prefix_keys": [cpu(k) for k, _ in prefix.kv],
        "prefix_values": [cpu(v) for _, v in prefix.kv], "vlm_logits": cpu(logits),
        "time_embedding": cpu(posemb_sincos(t, heads.width)), "adarms_cond": cpu(flow_condition(heads, t)),
        "v_t": cpu(v_t), "sample_actions": cpu(actions),
        "stage_prefix_hidden": cpu(prefix_math.hidden), "stage_prefix_keys": [cpu(k) for k, _ in prefix_math.kv],
        "stage_prefix_values": [cpu(v) for _, v in prefix_math.kv], "stage_vlm_logits": cpu(logits_math),
        "stage_v_t": cpu(v_t_ref_kv), "stage_step_velocities": cpu(step_velocities),
        "stage_sample_actions": cpu(actions_ref_kv),
    }
    return ParityRun(device, reference, outputs, prefix.kv, prefix.pos0, layout.nonki_valid)


@pytest.fixture(scope="module", params=["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def parity(request):
    """The flow stack's outputs on one device against that device's LeRobot dump."""
    device = _gpu_or_skip() if request.param == "cuda" else torch.device("cpu")
    reference = _load_reference(request.param)
    flow_stack = request.getfixturevalue("flow_stack")   # load the weights only once nothing else can skip
    with _ieee_fp32():
        try:
            flow_stack.to(device)
            yield _run_flow_stack(flow_stack, reference, device)
        finally:
            flow_stack.to("cpu")
            if device.type == "cuda":
                torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- loading
@pi05_slow
def test_flow_load_accounts_for_every_tensor(flow_stack):
    report = flow_stack.report
    assert report.mode == "flow" and report.sha256 == PI05_SHA256
    assert report.dropped == [EXPERT_LM_HEAD]
    assert len(report.consumed) == len(set(report.consumed)) == 811
    for module in (flow_stack.siglip, flow_stack.joint, flow_stack.heads):
        assert all(p.dtype == torch.float32 for p in module.parameters())


# ----------------------------------------------------------------------------- conventions
@pi05_slow
def test_prefix_and_suffix_conventions_match_the_dump(parity):
    ref, out = parity.reference, parity.outputs
    batch, slots, per_image = _camera_layout(ref)
    pad = ref["prefix_pad_masks"]
    assert torch.equal(out["prefix_valid"], pad)
    assert torch.equal(out["prefix_positions"], ref["prefix_position_ids"])
    assert out["pos0"].tolist() == [int(v) for v in ref["metadata"]["prefix_valid_counts"]] == pad.sum(1).tolist()
    # Valid query rows see exactly LeRobot's keys. Pad rows (prompt pads and the empty camera) see the valid
    # keys plus themselves, where LeRobot's see nothing and so attend uniformly. That is a deliberate difference.
    visible = out["prefix_bias"][:, 0] == 0
    assert torch.equal(visible[pad], ref["prefix_att_2d_masks"][pad])
    expected_pad_rows = pad[:, None, :].expand_as(visible) | torch.eye(pad.shape[1], dtype=torch.bool)[None]
    assert torch.equal(visible[~pad], expected_pad_rows[~pad]) and not ref["prefix_att_2d_masks"][~pad].any()
    assert bool((out["prefix_bias"][:, 0][~visible] == NEG).all())
    # Flow suffix: positions pos0 + k and bidirectional action rows that see the valid prefix keys.
    suffix = build_flow_suffix_layout(out["pos0"], pad, ref["v_t"].shape[1])
    assert torch.equal(suffix.positions, ref["suffix_position_ids"])
    assert torch.equal(suffix.bias[:, 0] == 0, ref["suffix_att_2d_masks"])
    # RoPE: the CPU-built inverse frequencies are the reference's table bit for bit.
    assert torch.equal(rope_inv_freq(GEMMA_2B.head_dim, GEMMA_2B.rope_theta), ref["rope_inv_freq"])
    assert out["image_embeddings"].shape == (batch, slots * per_image, GEMMA_2B.width)


# ----------------------------------------------------------------------------- end to end
@pi05_slow
def test_image_token_embeddings(parity):
    out, ref = parity.outputs, parity.reference
    _assert_parity("image_embeddings", out["image_embeddings"], ref["image_embeddings"],
                   TOL["image_embeddings"], parity.report)
    # Per-slot calls are a valid way to run SigLIP; on CUDA they differ only through GEMM shapes.
    _assert_parity("image_embeddings_per_slot", out["image_embeddings_per_slot"], ref["image_embeddings"],
                   TOL["image_embeddings_per_slot"], parity.report)


@pi05_slow
def test_text_embeddings_are_bitwise(parity):
    n_image_tokens = parity.reference["image_embeddings"].shape[1]
    expected = parity.reference["prefix_embeddings"][:, n_image_tokens:]
    parity.report["text_embeddings"] = dict(_diff_stats(parity.outputs["text_embeddings"], expected), tol=0.0)
    assert torch.equal(parity.outputs["text_embeddings"], expected), "embed_tokens(ids) * fp32 sqrt(2048) must be exact"


@pi05_slow
def test_prefix_final_hidden(parity):
    valid = parity.reference["prefix_pad_masks"]
    _assert_parity("prefix_hidden", parity.outputs["prefix_hidden"][valid], parity.reference["prefix_hidden"][valid],
                   TOL["prefix_hidden"], parity.report)


def _check_kv_layers(parity: ParityRun, prefix: str, keys: List[Tensor], values: List[Tensor], tol: float):
    """Post-RoPE K/V of every layer, each against its own scale. Layer 0 is compared at all positions, later
    layers at valid positions. The report keeps the worst layer per kind, and a failure lists every bad layer."""
    ref = parity.reference
    valid = ref["prefix_pad_masks"]
    failures = []
    for kind, ours, theirs in (("keys", keys, ref["prefix_keys"]), ("values", values, ref["prefix_values"])):
        assert len(ours) == len(theirs) == GEMMA_2B.depth
        worst, worst_layer, all_bitwise = None, -1, True
        for layer, (a, b) in enumerate(zip(ours, theirs)):
            if layer > 0:
                a, b = a[:, 0][valid], b[:, 0][valid]
            try:
                stats = _assert_parity(f"{prefix}{kind}[{layer:02d}]", a, b, tol)
            except AssertionError as error:
                failures.append(str(error).splitlines()[0])
                stats = _diff_stats(a, b)
            all_bitwise &= stats["bitwise"]
            if worst is None or stats["max_rel"] > worst["max_rel"]:
                worst, worst_layer = stats, layer
        parity.report[f"{prefix}{kind} [{len(ours)} layers; worst L{worst_layer:02d}]"] = dict(
            worst, tol=tol, bitwise=all_bitwise)
    assert not failures, "per-layer K/V parity failed:\n" + "\n".join(failures)


@pi05_slow
def test_prefix_kv_every_layer(parity):
    _check_kv_layers(parity, "prefix_", parity.outputs["prefix_keys"], parity.outputs["prefix_values"],
                     TOL["prefix_kv"])


@pi05_slow
def test_vlm_logits_at_dumped_positions(parity):
    ours, theirs = parity.outputs["vlm_logits"], parity.reference["vlm_logits"]
    _assert_parity("vlm_logits", ours, theirs, TOL["vlm_logits"], parity.report)
    # Top-1/top-2 margins in the dump are >= 0.27 and the closest top-5 pair is 0.0055 apart, against logit errors
    # of <= 2.5e-4, so both rankings are stable in fp32.
    assert torch.equal(ours.argmax(dim=-1), theirs.argmax(dim=-1))
    # The token after "\nAction: " (pi05_base's prior is openpi FAST ids), ranked exactly as LeRobot ranks it.
    top5 = torch.topk(ours[:, -1], k=5, dim=-1).indices.tolist()
    assert top5 == parity.reference["metadata"]["top5_next_token_ids"]


@pi05_slow
def test_time_conditioning(parity):
    _assert_parity("time_embedding", parity.outputs["time_embedding"], parity.reference["time_embedding"],
                   TOL["time_embedding"], parity.report)
    _assert_parity("adarms_cond", parity.outputs["adarms_cond"], parity.reference["adarms_cond"],
                   TOL["adarms_cond"], parity.report)


@pi05_slow
def test_flow_velocity(parity):
    _assert_parity("v_t", parity.outputs["v_t"], parity.reference["v_t"], TOL["v_t"], parity.report)


@pi05_slow
def test_sample_actions(parity):
    _assert_parity("sample_actions", parity.outputs["sample_actions"], parity.reference["sample_actions"],
                   TOL["sample_actions"], parity.report)


# ----------------------------------------------------------------------------- stage by stage
@pi05_slow
def test_stage_vlm_with_eager_arithmetic(parity):
    """Our VLM on LeRobot's prefix embeddings, with SDPA forced to the math backend, which is LeRobot's eager
    attention arithmetic. This isolates the Gemma-2B port (norms, RoPE, projections, GeGLU, residuals, final norm,
    tied head) from SigLIP and from the attention-kernel choice."""
    out, ref = parity.outputs, parity.reference
    valid = ref["prefix_pad_masks"]
    tol = TOL["stage_vlm_math"]
    _assert_parity("stage_prefix_hidden", out["stage_prefix_hidden"][valid], ref["prefix_hidden"][valid], tol,
                   parity.report)
    _check_kv_layers(parity, "stage_prefix_", out["stage_prefix_keys"], out["stage_prefix_values"], tol)
    _assert_parity("stage_vlm_logits", out["stage_vlm_logits"], ref["vlm_logits"], tol, parity.report)


@pi05_slow
def test_stage_expert_on_reference_kv(parity):
    """The adaRMS expert and flow head on LeRobot's own prefix K/V, isolated from every VLM difference."""
    out, ref = parity.outputs, parity.reference
    tol = TOL["stage_expert"]
    _assert_parity("stage_v_t", out["stage_v_t"], ref["v_t"], tol, parity.report)
    for step in range(out["stage_step_velocities"].shape[0]):
        _assert_parity(f"stage_velocity[t={float(ref['sample_times'][step]):.1f}]", out["stage_step_velocities"][step],
                       ref["sample_velocities"][step], tol, parity.report)
    _assert_parity("stage_sample_actions", out["stage_sample_actions"], ref["sample_actions"], tol, parity.report)


# ----------------------------------------------------------------------------- p2n mode
def _expert_norm_pairs(folded: GemmaJoint, unfolded: GemmaJoint):
    for index, (layer_s, layer_a) in enumerate(zip(folded.expert.layers, unfolded.expert.layers)):
        for name in ("input_layernorm", "post_attention_layernorm"):
            yield f"layers.{index}.{name}", getattr(layer_s, name), getattr(layer_a, name)
    yield "norm", folded.expert.norm, unfolded.expert.norm


def _shared_weights(joint: GemmaJoint) -> Dict[str, Tensor]:
    """Every tensor the flow and p2n loads must map identically (all but the expert norms)."""
    shared = {f"vlm.{key}": value for key, value in joint.vlm.state_dict().items()}
    shared.update({f"expert.{key}": value for key, value in joint.expert.state_dict().items()
                   if "layernorm" not in key and not key.startswith("norm.")})
    return shared


@pi05_slow
def test_p2n_fold_reproduces_adarms_expert(parity, flow_stack, p2n_stack):
    """``mode='p2n'`` folds every adaRMS Dense at c0 = c(0.6) into per-segment modulations. At step 0 the folded
    expert must reproduce the adaRMS expert driven by cond = c(0.6) on the same prefix, for every segment id, and
    it must reproduce LeRobot's own velocity at t = 0.6."""
    device, ref, report = parity.device, parity.reference, p2n_stack.report
    joint_a, heads, joint_s = flow_stack.joint, flow_stack.heads, p2n_stack.joint
    batch = ref["v_t"].shape[0]
    expected_drops = [EXPERT_LM_HEAD] + [f"{name}.{suffix}" for name in FLOW_HEAD_NAMES if name.startswith("action_")
                                         for suffix in ("weight", "bias")]
    assert report.mode == "p2n" and report.t0 == P2N_T0 and report.sha256 == PI05_SHA256
    assert len(report.folded) == 2 * GEMMA_300M.depth + 1
    assert sorted(report.dropped) == sorted(expected_drops)
    assert len(report.consumed) + len(report.dropped) == 812

    # The folded condition is the flow head's condition at t0. compute_c0 runs on the CPU, the flow head on the device.
    with torch.no_grad():
        cond = flow_condition(heads, torch.full((batch,), P2N_T0, dtype=torch.float32, device=device))
    _assert_parity("p2n_c0", report.c0[None].expand(batch, -1), cond, TOL["p2n_c0"], parity.report)

    # Every folded modulation row equals Dense(c0), checked in float64. All segments start identical.
    c0 = report.c0.double()
    worst: Dict[str, float] = {}
    for name, norm_s, norm_a in _expert_norm_pairs(joint_s, joint_a):
        assert isinstance(norm_s, SegmentRMSNorm) and isinstance(norm_a, AdaRMSNorm)
        rows = norm_s.modulation.detach()
        assert rows.shape == (N_SEGMENTS, 3 * GEMMA_300M.width) and rows.device.type == "cpu"
        assert all(torch.equal(rows[0], rows[i]) for i in range(1, N_SEGMENTS)), name
        dense = norm_a.dense
        expected = dense.weight.detach().cpu().double() @ c0 + dense.bias.detach().cpu().double()
        stats = _assert_parity(f"p2n_fold.{name}", rows[0], expected, TOL["p2n_fold"])
        worst = max(worst, stats, key=lambda s: s.get("max_rel", -1.0))
    parity.report["p2n_fold (worst of 37 norms)"] = dict(worst, tol=TOL["p2n_fold"])
    final = joint_s.expert.norm.modulation.detach()[0, 2 * GEMMA_300M.width:]
    assert bool((final == 0).all()), "the final-norm gate rows are zero in pi05_base"

    # Both loads map the shared weights identically.
    shared_s, shared_a = _shared_weights(joint_s), _shared_weights(joint_a)
    assert sorted(shared_s) == sorted(shared_a)
    mismatched = [key for key in shared_s if not torch.equal(shared_s[key], shared_a[key].cpu())]
    assert not mismatched, f"p2n and flow loads disagree on {mismatched[:5]}"

    # Expert outputs on the same prefix: our end-to-end prefix K/V, and then LeRobot's K/V.
    joint_s.expert.to(device)
    try:
        with torch.no_grad():
            n_act = ref["v_t"].shape[1]
            tokens = heads.action_in_proj(ref["inputs"]["x_t"].to(device))
            flow_layout = build_flow_suffix_layout(parity.pos0, parity.prefix_valid, n_act)
            mixed_segments = torch.arange(n_act, device=device) % N_SEGMENTS
            hidden_a = joint_a.expert_forward(tokens, parity.prefix_kv, flow_layout.bias, flow_layout.positions,
                                              cond=cond).hidden
            hidden_s = joint_s.expert_forward(tokens, parity.prefix_kv, flow_layout.bias, flow_layout.positions,
                                              seg_ids=mixed_segments).hidden
            _assert_parity("p2n_expert_hidden (flow layout)", hidden_s, hidden_a, TOL["p2n_expert"], parity.report)
            _assert_parity("p2n_velocity (flow layout)", heads.action_out_proj(hidden_s),
                           heads.action_out_proj(hidden_a), TOL["p2n_expert"], parity.report)

            # A P2N suffix [RAW/DIFF 9 | HIST 4 | AR 8]: segment ids 0/1/2, a per-sample log-gate and invalid slots.
            generator = torch.Generator().manual_seed(7)
            rawdiff_valid = torch.ones(batch, N_RAWDIFF_SLOTS, dtype=torch.bool)
            rawdiff_valid[0, :4] = False
            log_gate = torch.linspace(-1.5, -0.05, batch)[:, None]
            suffix = build_suffix_layout(parity.pos0, parity.prefix_valid, rawdiff_valid.to(device), 4, 8,
                                         log_gate=log_gate.to(device))
            x = torch.randn(batch, suffix.positions.shape[1], GEMMA_300M.width, generator=generator).to(device)
            hidden_a = joint_a.expert_forward(x, parity.prefix_kv, suffix.bias, suffix.positions, cond=cond).hidden
            hidden_s = joint_s.expert_forward(x, parity.prefix_kv, suffix.bias, suffix.positions,
                                              seg_ids=suffix.seg_ids).hidden
            _assert_parity("p2n_expert_hidden (P2N layout)", hidden_s, hidden_a, TOL["p2n_expert"], parity.report)

            # LeRobot's velocity at t = 0.6: Euler step 4 of the dumped trajectory, on LeRobot's K/V.
            step = int(torch.nonzero(ref["sample_times"] == torch.tensor(P2N_T0, dtype=torch.float32))[0, 0])
            kv_ref = tuple((k.to(device), v.to(device)) for k, v in zip(ref["prefix_keys"], ref["prefix_values"]))
            valid_ref = ref["prefix_pad_masks"].to(device)
            layout_ref = build_flow_suffix_layout(valid_ref.sum(dim=1), valid_ref, n_act)
            tokens = heads.action_in_proj(ref["sample_trajectory"][step].to(device))
            hidden_s = joint_s.expert_forward(tokens, kv_ref, layout_ref.bias, layout_ref.positions,
                                              seg_ids=mixed_segments).hidden
            _assert_parity("p2n_velocity vs LeRobot (t=0.6)", heads.action_out_proj(hidden_s),
                           ref["sample_velocities"][step], TOL["stage_expert"], parity.report)
    finally:
        joint_s.expert.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- report
@pi05_slow
def test_zz_parity_report(parity, capsys):
    """Print every recorded max abs / max rel error. It runs last for each device."""
    assert parity.report, "no parity statistics were recorded"
    lines = [f"\n[p2n-vla parity | {parity.device.type} vs {REFERENCES[parity.device.type].name}]",
             f"  {'quantity':<44}{'max_abs':>11}{'max_rel':>11}{'tol':>9}  bitwise"]
    for name, row in parity.report.items():
        lines.append(f"  {name:<44}{row['max_abs']:>11.2e}{row['max_rel']:>11.2e}{row['tol']:>9.0e}  "
                     f"{'yes' if row['bitwise'] else 'no'}")
    with capsys.disabled():
        print("\n".join(lines))
