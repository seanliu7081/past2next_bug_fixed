"""pi05_base loader tests: synthetic checkpoints (fast) and the real 14.5 GB file (marked)."""
import time

import pytest
import torch

from oat.model.vla.flow_head import FlowHeads
from oat.model.vla.gemma_joint import GemmaJoint
from oat.model.vla.layout import SEG_AR, build_prefix_layout, build_suffix_layout, decode_view, prefill_view
from oat.model.vla.lora import inject_lora
from oat.model.vla.pi05_checkpoint import (EXPERT_LM_HEAD, EXPERT_PREFIX, LM_HEAD, PROJECTOR_PREFIX,
                                           SIGLIP_PREFIX, VLM_PREFIX, compute_c0, default_pi05_path,
                                           load_pi05, read_header)
from oat.model.vla.siglip import SiglipEncoder
from oat.model.vla.specs import GEMMA_2B, GEMMA_300M, SIGLIP_SO400M, TINY_EXPERT, TINY_SIGLIP, GemmaSpec

SMALL_VLM = GemmaSpec(64, 2, 128, 8, 1, 16, vocab_size=300)


def _modules(norm_mode):
    siglip = SiglipEncoder(TINY_SIGLIP, SMALL_VLM.width, frozen_dtype=torch.float32)
    joint = GemmaJoint(SMALL_VLM, TINY_EXPERT, norm_mode, frozen_dtype=torch.float32, activation_checkpointing=False)
    heads = FlowHeads(TINY_EXPERT.width) if norm_mode == "adarms" else None
    return siglip, joint, heads


def _synthetic_state(seed=0):
    """Tensors under the real pi05 key names, shaped for the small specs."""
    generator = torch.Generator().manual_seed(seed)
    rand = lambda *shape: torch.randn(*shape, generator=generator) * 0.05  # noqa: E731
    siglip, joint, heads = _modules("adarms")
    state = {SIGLIP_PREFIX + k: rand(*v.shape) for k, v in siglip.vision.state_dict().items()}
    state[PROJECTOR_PREFIX + "weight"] = rand(SMALL_VLM.width, TINY_SIGLIP.hidden)
    state[PROJECTOR_PREFIX + "bias"] = rand(SMALL_VLM.width)
    state[LM_HEAD] = rand(SMALL_VLM.vocab_size, SMALL_VLM.width)
    for i in range(SMALL_VLM.depth):
        for name, value in joint.vlm.layers[i].state_dict().items():
            state[f"{VLM_PREFIX}layers.{i}.{name}"] = rand(*value.shape)
    state[VLM_PREFIX + "norm.weight"] = rand(SMALL_VLM.width)
    width = TINY_EXPERT.width
    for i in range(TINY_EXPERT.depth):
        for name, value in joint.expert.layers[i].state_dict().items():
            state[f"{EXPERT_PREFIX}layers.{i}.{name}"] = rand(*value.shape)
    norm_weight, norm_bias = rand(3 * width, width), rand(3 * width)
    norm_weight[2 * width:] = 0
    norm_bias[2 * width:] = 0
    state[EXPERT_PREFIX + "norm.dense.weight"] = norm_weight
    state[EXPERT_PREFIX + "norm.dense.bias"] = norm_bias
    state[EXPERT_LM_HEAD] = rand(SMALL_VLM.vocab_size, width)
    for name, module in heads.named_children():
        state[f"{name}.weight"] = rand(*module.weight.shape)
        state[f"{name}.bias"] = rand(*module.bias.shape)
    return state


def _save(state, tmp_path, name="pi05.safetensors"):
    from safetensors.torch import save_file
    path = tmp_path / name
    save_file({k: v.contiguous() for k, v in state.items()}, str(path))
    return path


def test_flow_mode_loads_every_tensor(tmp_path):
    state = _synthetic_state()
    path = _save(state, tmp_path)
    siglip, joint, heads = _modules("adarms")
    report = load_pi05(path, siglip=siglip, joint=joint, mode="flow", flow_heads=heads)
    assert report.dropped == [EXPERT_LM_HEAD]
    assert set(report.consumed) | set(report.dropped) == set(state)
    assert torch.equal(joint.vlm.embed_tokens.weight, state[LM_HEAD])
    assert torch.equal(joint.expert.layers[1].input_layernorm.dense.weight,
                       state[EXPERT_PREFIX + "layers.1.input_layernorm.dense.weight"])
    assert torch.equal(heads.time_mlp_in.weight, state["time_mlp_in.weight"])
    assert torch.equal(siglip.projector.bias, state[PROJECTOR_PREFIX + "bias"])


def test_p2n_mode_folds_adarms_and_matches_flow_expert(tmp_path):
    state = _synthetic_state(1)
    path = _save(state, tmp_path)
    siglip_f, joint_f, heads = _modules("adarms")
    load_pi05(path, siglip=siglip_f, joint=joint_f, mode="flow", flow_heads=heads)
    siglip_s, joint_s, _ = _modules("segment")
    report = load_pi05(path, siglip=siglip_s, joint=joint_s, mode="p2n", t0=0.6)
    assert sorted(report.dropped) == sorted([EXPERT_LM_HEAD, "action_in_proj.weight", "action_in_proj.bias",
                                             "action_out_proj.weight", "action_out_proj.bias"])
    c0 = compute_c0({k: state[k] for k in state if k.startswith("time_mlp")}, 0.6, TINY_EXPERT.width)
    assert torch.allclose(report.c0, c0)
    dense_w = state[EXPERT_PREFIX + "layers.0.input_layernorm.dense.weight"]
    dense_b = state[EXPERT_PREFIX + "layers.0.input_layernorm.dense.bias"]
    expected = dense_w @ c0 + dense_b
    for row in joint_s.expert.layers[0].input_layernorm.modulation:
        assert torch.allclose(row, expected, atol=1e-6)

    joint_f.eval(), joint_s.eval()
    img_valid = torch.ones(2, 4, dtype=torch.bool)
    prompt_valid = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)
    layout = build_prefix_layout(img_valid, prompt_valid)
    embeds = torch.randn(2, 8, SMALL_VLM.width)
    out_f, out_s = joint_f.prefix_forward(embeds, layout), joint_s.prefix_forward(embeds, layout)
    suffix = build_suffix_layout(out_f.pos0, layout.nonki_valid, None, 0, 5)
    x = torch.randn(2, 5, TINY_EXPERT.width)
    folded = joint_s.expert_forward(x, out_s.kv, suffix.bias, suffix.positions, seg_ids=suffix.seg_ids).hidden
    unfolded = joint_f.expert_forward(x, out_f.kv, suffix.bias, suffix.positions, cond=c0[None].expand(2, -1)).hidden
    assert torch.allclose(folded, unfolded, atol=1e-5)


def test_loader_rejects_bad_checkpoints(tmp_path):
    state = _synthetic_state(2)
    extra = dict(state, unexpected_tensor=torch.zeros(1))
    siglip, joint, heads = _modules("adarms")
    with pytest.raises(KeyError, match="Unexpected"):
        load_pi05(_save(extra, tmp_path, "extra.safetensors"), siglip=siglip, joint=joint, mode="flow",
                  flow_heads=heads)
    missing = {k: v for k, v in state.items() if not k.endswith("layers.1.mlp.up_proj.weight")}
    with pytest.raises(KeyError):
        load_pi05(_save(missing, tmp_path, "missing.safetensors"), siglip=siglip, joint=joint, mode="flow",
                  flow_heads=heads)
    gated = dict(state)
    gated[EXPERT_PREFIX + "norm.dense.bias"] = gated[EXPERT_PREFIX + "norm.dense.bias"].clone() + 1
    with pytest.raises(ValueError, match="final-norm gate"):
        load_pi05(_save(gated, tmp_path, "gated.safetensors"), siglip=siglip, joint=joint, mode="flow",
                  flow_heads=heads)
    siglip_s, joint_s, _ = _modules("segment")
    with pytest.raises(ValueError):
        load_pi05(_save(state, tmp_path), siglip=siglip_s, joint=joint_s, mode="flow", flow_heads=heads)
    inject_lora(joint_s.vlm.layers, 2, 2.0)
    with pytest.raises(RuntimeError, match="before injecting LoRA"):
        load_pi05(_save(state, tmp_path), siglip=siglip_s, joint=joint_s, mode="p2n")


def test_compute_c0_matches_reference_formula():
    torch.manual_seed(0)
    state = {"time_mlp_in.weight": torch.randn(8, 8), "time_mlp_in.bias": torch.randn(8),
             "time_mlp_out.weight": torch.randn(8, 8), "time_mlp_out.bias": torch.randn(8)}
    import math
    fraction = torch.linspace(0, 1, 4, dtype=torch.float64)
    period = 4e-3 * (4.0 / 4e-3) ** fraction
    t0 = torch.tensor(0.6, dtype=torch.float32).double()  # PI0.5 timesteps are fp32 values
    angle = t0 * 2 * math.pi / period
    emb = torch.cat((angle.sin(), angle.cos())).float()
    hidden = torch.nn.functional.silu(state["time_mlp_in.weight"] @ emb + state["time_mlp_in.bias"])
    expected = torch.nn.functional.silu(state["time_mlp_out.weight"] @ hidden + state["time_mlp_out.bias"])
    assert torch.allclose(compute_c0(state, 0.6, 8), expected, atol=1e-6)


# ---------------------------------------------------------------- real pi05_base weights
PI05 = default_pi05_path()
needs_pi05 = pytest.mark.skipif(not PI05.is_file(), reason=f"pi05_base weights missing at {PI05}")


@pytest.mark.requires_pi05
@needs_pi05
def test_real_header_layout():
    header = read_header(PI05)
    assert len(header) == 812 and all(dtype == "F32" for dtype, _ in header.values())
    assert header[LM_HEAD][1] == [257152, 2048]
    assert header[EXPERT_PREFIX + "layers.0.input_layernorm.dense.weight"][1] == [3072, 1024]
    assert header[SIGLIP_PREFIX + "vision_model.encoder.layers.26.mlp.fc1.weight"][1] == [4304, 1152]


def _full_modules(dtype=torch.bfloat16):
    siglip = SiglipEncoder(SIGLIP_SO400M, GEMMA_2B.width, frozen_dtype=dtype)
    joint = GemmaJoint(GEMMA_2B, GEMMA_300M, "segment", frozen_dtype=dtype, activation_checkpointing=False)
    return siglip, joint


@pytest.mark.requires_pi05
@pytest.mark.slow
@needs_pi05
def test_full_size_p2n_load_on_cpu():
    siglip, joint = _full_modules()
    report = load_pi05(PI05, siglip=siglip, joint=joint, mode="p2n", t0=0.6)
    assert report.sha256 == "0eb11ca9587678c1d2ef8cf32807c29f8ce53a2bfdfc1aa4a4c96f16fca59b0f"
    assert torch.isfinite(report.c0).all() and report.c0.shape == (1024,)
    assert len(report.folded) == 2 * 18 + 1
    frozen = sum(p.numel() for p in list(siglip.parameters()) + list(joint.vlm.parameters()))
    expert = sum(p.numel() for p in joint.expert.parameters())
    assert 2.90e9 < frozen < 2.95e9 and 3.0e8 < expert < 3.2e8, (frozen, expert)


@pytest.mark.requires_pi05
@pytest.mark.gpu
@pytest.mark.slow
@needs_pi05
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_full_size_gpu_prefix_and_cached_decode(capsys):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    siglip, joint = _full_modules()
    load_pi05(PI05, siglip=siglip, joint=joint, mode="p2n", t0=0.6)
    inject_lora(joint.vlm.layers, 16, 16.0)
    device = torch.device("cuda:0")
    siglip.to(device), joint.to(device).eval()
    torch.cuda.reset_peak_memory_stats(device)
    batch, n_img, n_prompt = 2, 512, 64
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        images = siglip(torch.rand(batch * 2, 3, 224, 224, device=device) * 2 - 1).reshape(batch, n_img, -1)
        ids = torch.randint(3, 250000, (batch, n_prompt), device=device)
        prompt_valid = torch.ones(batch, n_prompt, dtype=torch.bool, device=device)
        prompt_valid[0, 40:] = False
        layout = build_prefix_layout(torch.ones(batch, n_img, dtype=torch.bool, device=device), prompt_valid)
        embeds = torch.cat((images, joint.embed_text(ids)), dim=1)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            prefix = joint.prefix_forward(embeds, layout)
        torch.cuda.synchronize(device)
        prefix_ms = (time.perf_counter() - start) * 1e3
        suffix = build_suffix_layout(prefix.pos0, layout.nonki_valid, torch.ones(batch, 9, dtype=torch.bool,
                                                                                 device=device), 4, 8,
                                     log_gate=torch.full((batch, 1), -0.1, device=device))
        cond = torch.randn(batch, 14, 1024, device=device)
        bias, positions, seg = prefill_view(suffix, 14)
        start = time.perf_counter()
        step = joint.expert_forward(cond, prefix.kv, bias, positions, seg_ids=seg, use_cache=True)
        cache = step.cache
        for k in range(1, 8):
            bias, position = decode_view(suffix, k)
            step = joint.expert_forward(torch.randn(batch, 1, 1024, device=device), prefix.kv, bias, position,
                                        seg_ids=torch.tensor([SEG_AR], device=device), cache=cache, use_cache=True)
            cache = step.cache
        torch.cuda.synchronize(device)
        decode_ms = (time.perf_counter() - start) * 1e3
    assert torch.isfinite(step.hidden).all()
    peak = torch.cuda.max_memory_allocated(device) / 2**30
    with capsys.disabled():
        print(f"\n[full-size gpu] prefix {prefix_ms:.1f} ms, 8-step decode {decode_ms:.1f} ms, peak {peak:.2f} GiB")
    siglip.cpu(), joint.cpu()
    torch.cuda.empty_cache()
