"""Fixed inputs and file schema for the LeRobot PI0.5 fp32 parity reference.

This module depends only on torch and sentencepiece. Two environments import it:

- ``dump_pi05_reference.py``, under ``/venv/lerobot_ref`` (Python 3.12, lerobot, transformers 5.5);
- the parity checks under ``/venv/oat`` (Python 3.10, transformers 5.2).

It must therefore stay Python 3.10 compatible and must never import lerobot or oat.

The reference file is ``output/parity/pi05_reference_fp32.pt``. ``validate_reference`` documents its
layout and checks it. Every tensor is on CPU. Float tensors are fp32; ids, positions and indices are
int64; masks are bool.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_FORMAT = "p2n_vla_pi05_reference_fp32_v1"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "parity" / "pi05_reference_fp32.pt"
DEFAULT_SPM = REPO_ROOT / "data" / "pretrained" / "p2n_vla" / "paligemma_tokenizer.model"
SPM_SHA256 = "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"

PI05_REPO = "lerobot/pi05_base"
PI05_REVISION = "b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba"
PI05_SHA256 = "0eb11ca9587678c1d2ef8cf32807c29f8ce53a2bfdfc1aa4a4c96f16fca59b0f"
HF_HOME = Path("/workspace/.hf_home")
PI05_SNAPSHOT = HF_HOME / "hub" / "models--lerobot--pi05_base" / "snapshots" / PI05_REVISION
PI05_WEIGHTS = PI05_SNAPSHOT / "model.safetensors"

# huggingface/lerobot main at the time of the dump. It needs Python >= 3.12 and transformers >= 5.4, < 5.6.
LEROBOT_COMMIT = "2577da0ef39b47f870592d62d81edc8bce922cc4"
LEROBOT_VENV = Path("/venv/lerobot_ref")

# ----------------------------------------------------------------------------- fixed inputs
PROMPTS: Tuple[str, ...] = (
    "Task: pick up the book and place it in the back compartment of the caddy, "
    "State: 12 200 37 128 255 0 64 99;\nAction: ",
    "Task: put both moka pots on the stove, State: 3 3 3 3 3 3 3 3;\nAction: ",
)
# The first prompt is 51 SentencePiece tokens including BOS, so the requested 48 cannot hold it.
# 64 is the smallest round length that leaves pads in both rows (13 and 29).
LANG_MAX_LEN = 64
BATCH = len(PROMPTS)
N_REAL_CAMERAS = 2              # base_0_rgb and left_wrist_0_rgb; pi05_base's third slot stays empty
IMAGE_SIZE = 224
CHUNK_SIZE = 50                 # lerobot/pi05_base config.json chunk_size
ACTION_DIM = 32                 # max_action_dim
T_VALUES: Tuple[float, ...] = (0.3, 0.8)
SEED = 20261006
N_LOGIT_POSITIONS = 4
PAD_ID, EOS_ID, BOS_ID = 0, 1, 2

# Geometry of the pi05_base backbone. These values are checked against the live model at dump time.
VLM_WIDTH, EXPERT_WIDTH, DEPTH, HEAD_DIM, VOCAB_SIZE = 2048, 1024, 18, 256, 257152
TOKENS_PER_IMAGE = (IMAGE_SIZE // 14) ** 2   # SigLIP So400m/14 at 224 px gives 256

# Regression guard: the public big_vision SentencePiece model (sha256 above) with add_bos=True.
EXPECTED_PROMPT_IDS: Tuple[Tuple[int, ...], ...] = (
    (2, 7071, 235292, 4788, 908, 573, 2870, 578, 2040, 665, 575, 573, 1355, 46416, 576, 573, 132588,
     235269, 3040, 235292, 235248, 235274, 235284, 235248, 235284, 235276, 235276, 235248, 235304, 235324,
     235248, 235274, 235284, 235321, 235248, 235284, 235308, 235308, 235248, 235276, 235248, 235318, 235310,
     235248, 235315, 235315, 235289, 108, 4022, 235292, 235248),
    (2, 7071, 235292, 2507, 2145, 705, 1161, 37801, 611, 573, 37932, 235269, 3040, 235292, 235248, 235304,
     235248, 235304, 235248, 235304, 235248, 235304, 235248, 235304, 235248, 235304, 235248, 235304, 235248,
     235304, 235289, 108, 4022, 235292, 235248),
)

INPUT_RECIPE = (
    "One CPU torch.Generator seeded with SEED draws, in this order: "
    "(1) camera images in [0,1] [B, 2, 3, 224, 224], built from a coarse U(0,1) 8x8 field upsampled bilinearly "
    "(align_corners=False), plus 0.08*N(0,1) grain, clamped to [0,1]; "
    "(2) noise ~ N(0,1) [B, H, 32]; "
    "(3) eps ~ N(0,1) [B, H, 32]; "
    "(4) target ~ U(-1,1) [B, H, 32], with x_t = t*eps + (1-t)*target and t = T_VALUES. "
    "Model images = images01*2-1 (LeRobot _preprocess_images), and the empty third slot is -1 with mask False. "
    "Prompts are SentencePiece ids with BOS, right-padded with 0 to LANG_MAX_LEN, with mask True on real tokens."
)


def configure_ieee_fp32(deterministic: bool = True) -> Dict[str, object]:
    """Turn TF32 off for matmul and cuDNN; with ``deterministic``, also make kernels deterministic.

    By default torch leaves cuDNN conv on TF32, which affects SigLIP's patch embedding. Call this
    before any CUDA work, because CUBLAS_WORKSPACE_CONFIG must be set before cuBLAS initializes.
    Returns the resulting state.
    """
    import os

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if hasattr(torch.backends, "fp32_precision"):          # torch >= 2.9 API
        torch.backends.fp32_precision = "ieee"
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.fp32_precision = "ieee"
        torch.backends.cudnn.conv.fp32_precision = "ieee"
        torch.backends.cudnn.rnn.fp32_precision = "ieee"
        state: Dict[str, object] = {
            "api": "torch.backends.*.fp32_precision",
            "global": torch.backends.fp32_precision,
            "cuda_matmul": torch.backends.cuda.matmul.fp32_precision,
            "cudnn_conv": torch.backends.cudnn.conv.fp32_precision,
            "cudnn_rnn": torch.backends.cudnn.rnn.fp32_precision,
        }
        bad = {name: value for name, value in state.items() if name != "api" and value != "ieee"}
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        state = {"api": "legacy allow_tf32", "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                 "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}
        bad = {name: value for name, value in state.items() if value is True}
    if bad:
        raise RuntimeError(f"Could not disable TF32: {bad}")
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        state.update({"cudnn_benchmark": False, "cudnn_deterministic": True, "deterministic_algorithms": True,
                      "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")})
    return state


def ieee_fp32_active() -> bool:
    """True while matmul and cuDNN conv run in IEEE fp32 (no TF32)."""
    if hasattr(torch.backends, "fp32_precision"):
        return (torch.backends.cuda.matmul.fp32_precision == "ieee"
                and torch.backends.cudnn.conv.fp32_precision == "ieee")
    return not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32


def sha256_file(path, chunk_bytes: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def tokenize_prompts(spm_path=DEFAULT_SPM, prompts: Sequence[str] = PROMPTS,
                     max_len: int = LANG_MAX_LEN) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return SentencePiece ids with BOS, right-padded with 0: ``tokens`` [B, max_len] int64 and ``masks`` [B, max_len] bool."""
    import sentencepiece

    if isinstance(max_len, bool) or not isinstance(max_len, int) or max_len < 1:
        raise ValueError("max_len must be a positive integer")
    if not prompts:
        raise ValueError("prompts must not be empty")
    path = Path(spm_path)
    if not path.is_file():
        raise FileNotFoundError(f"SentencePiece model not found: {path}")
    processor = sentencepiece.SentencePieceProcessor(model_file=str(path))
    if (processor.pad_id(), processor.eos_id(), processor.bos_id()) != (PAD_ID, EOS_ID, BOS_ID):
        raise ValueError("Unexpected special ids in the SentencePiece model; expected pad=0, eos=1, bos=2")
    tokens = torch.full((len(prompts), max_len), PAD_ID, dtype=torch.long)
    masks = torch.zeros((len(prompts), max_len), dtype=torch.bool)
    for row, text in enumerate(prompts):
        ids = processor.encode(text, add_bos=True)
        if not ids or ids[0] != BOS_ID:
            raise ValueError(f"Prompt {row} does not start with BOS")
        if len(ids) > max_len:
            raise ValueError(f"Prompt {row} needs {len(ids)} tokens, which exceeds max_len={max_len}")
        tokens[row, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        masks[row, :len(ids)] = True
    return tokens, masks


def make_camera_images01(batch: int, n_cameras: int, size: int, generator: torch.Generator) -> torch.Tensor:
    """Smooth-plus-grain test images in [0, 1], shape [B, n_cameras, 3, size, size] fp32."""
    coarse = torch.rand((batch * n_cameras, 3, 8, 8), generator=generator, dtype=torch.float32)
    smooth = F.interpolate(coarse, size=(size, size), mode="bilinear", align_corners=False)
    grain = 0.08 * torch.randn((batch * n_cameras, 3, size, size), generator=generator, dtype=torch.float32)
    images = (smooth + grain).clamp(0.0, 1.0)
    return images.reshape(batch, n_cameras, 3, size, size).contiguous()


def make_flow_inputs(batch: int, chunk_size: int, action_dim: int, t_values: Sequence[float],
                     generator: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``noise`` and ``x_t`` [B, H, A] fp32 and ``t`` [B] fp32, with x_t on the flow path at time t."""
    if len(t_values) != batch:
        raise ValueError("t_values needs one entry per sample")
    shape = (batch, chunk_size, action_dim)
    noise = torch.randn(shape, generator=generator, dtype=torch.float32)
    eps = torch.randn(shape, generator=generator, dtype=torch.float32)
    target = torch.rand(shape, generator=generator, dtype=torch.float32) * 2.0 - 1.0
    t = torch.tensor(list(t_values), dtype=torch.float32)
    x_t = t[:, None, None] * eps + (1.0 - t[:, None, None]) * target
    return noise, x_t, t


def build_reference_inputs(spm_path=DEFAULT_SPM, *, max_len: int = LANG_MAX_LEN, chunk_size: int = CHUNK_SIZE,
                           action_dim: int = ACTION_DIM, seed: int = SEED) -> Dict[str, torch.Tensor]:
    """Return the deterministic CPU inputs. ``images01`` holds only the real cameras, in [0, 1]."""
    generator = torch.Generator().manual_seed(int(seed))
    images01 = make_camera_images01(BATCH, N_REAL_CAMERAS, IMAGE_SIZE, generator)
    noise, x_t, t = make_flow_inputs(BATCH, chunk_size, action_dim, T_VALUES, generator)
    tokens, masks = tokenize_prompts(spm_path, PROMPTS, max_len)
    return {"images01": images01, "lang_tokens": tokens, "lang_masks": masks, "noise": noise, "x_t": x_t, "t": t}


def to_model_images(images01: torch.Tensor, n_slots: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """LeRobot's camera convention: real cameras become ``x*2-1``, and missing slots become -1 with mask False.

    Returns ``images`` [B, n_slots, 3, S, S] and ``img_masks`` [B, n_slots].
    """
    batch, n_real = images01.shape[:2]
    if n_slots < n_real:
        raise ValueError("n_slots must be at least the number of real cameras")
    real = images01 * 2.0 - 1.0
    empty = torch.full((batch, n_slots - n_real) + tuple(images01.shape[2:]), -1.0, dtype=images01.dtype)
    masks = torch.zeros((batch, n_slots), dtype=torch.bool)
    masks[:, :n_real] = True
    return torch.cat((real, empty), dim=1), masks


# ----------------------------------------------------------------------------- file schema
REQUIRED_METADATA_KEYS = (
    "lerobot_commit", "lerobot_version", "transformers_version", "torch_version", "python_version",
    "pi05_repo", "pi05_revision", "pi05_weights", "spm_sha256", "device", "dtype", "autocast", "fp32_precision",
    "attn_implementation", "chunk_size", "max_action_dim", "num_inference_steps", "min_period", "max_period",
    "camera_slots", "camera_slot_valid", "empty_camera_fill", "n_tokens_per_image", "n_image_tokens",
    "lang_max_len", "prefix_len", "lang_valid_lengths", "prefix_valid_counts", "keys_post_rope",
    "position_construction", "mask_construction", "scaling", "seed", "prompts", "t_values", "input_recipe",
    "logit_positions", "weights_check", "diagnostics", "deviations", "geometry",
)


def _shape_of(value) -> str:
    return str(tuple(value.shape)) if isinstance(value, torch.Tensor) else type(value).__name__


_PLAIN_TYPES = (str, int, float, bool, type(None))


def _non_plain_paths(value, path: str = "metadata") -> List[str]:
    """Paths of metadata leaves that torch.load(weights_only=True) may reject (exact plain types only)."""
    if isinstance(value, dict) and type(value) is dict:
        return [p for key, item in value.items() for p in
                ([f"{path} key {key!r}"] if type(key) is not str else []) + _non_plain_paths(item, f"{path}.{key}")]
    if type(value) in (list, tuple):
        return [p for index, item in enumerate(value) for p in _non_plain_paths(item, f"{path}[{index}]")]
    return [] if type(value) in _PLAIN_TYPES else [f"{path} ({type(value).__module__}.{type(value).__name__})"]


def validate_reference(ref: Mapping) -> Dict[str, Tuple[int, ...]]:
    """Check keys, shapes, dtypes, finiteness and basic internal consistency of a reference payload.

    Raises ValueError that lists every problem found. Returns ``{key: shape}`` for the tensors.
    """
    problems: List[str] = []
    if not isinstance(ref, Mapping):
        raise ValueError("Reference payload must be a mapping")
    if ref.get("format") != REFERENCE_FORMAT:
        problems.append(f"format must be {REFERENCE_FORMAT!r}, got {ref.get('format')!r}")
    meta = ref.get("metadata")
    if not isinstance(meta, Mapping):
        raise ValueError("Reference payload has no metadata mapping: " + "; ".join(problems))
    missing_meta = [key for key in REQUIRED_METADATA_KEYS if key not in meta]
    if missing_meta:
        raise ValueError(f"metadata is missing keys {missing_meta}")
    non_plain = _non_plain_paths(dict(meta))
    if non_plain:
        problems.append(f"metadata must hold only plain str/int/float/bool/None/list/dict values: {non_plain[:5]}")

    batch = len(meta["prompts"])
    slots = len(meta["camera_slots"])
    lang = int(meta["lang_max_len"])
    chunk = int(meta["chunk_size"])
    action = int(meta["max_action_dim"])
    per_image = int(meta["n_tokens_per_image"])
    n_img = slots * per_image
    prefix = n_img + lang
    steps = int(meta["num_inference_steps"])
    geometry = meta["geometry"]
    width, depth, head_dim = int(geometry["vlm_width"]), int(geometry["depth"]), int(geometry["head_dim"])
    vocab, expert_width = int(geometry["vocab_size"]), int(geometry["expert_width"])
    if int(meta["n_image_tokens"]) != n_img or int(meta["prefix_len"]) != prefix:
        problems.append("metadata n_image_tokens/prefix_len disagree with the camera slots and lang_max_len")

    f32, i64, boolean = torch.float32, torch.int64, torch.bool
    expected = {
        ("inputs", "images"): ((batch, slots, 3, IMAGE_SIZE, IMAGE_SIZE), f32),
        ("inputs", "img_masks"): ((batch, slots), boolean),
        ("inputs", "lang_tokens"): ((batch, lang), i64),
        ("inputs", "lang_masks"): ((batch, lang), boolean),
        ("inputs", "noise"): ((batch, chunk, action), f32),
        ("inputs", "x_t"): ((batch, chunk, action), f32),
        ("inputs", "t"): ((batch,), f32),
        ("image_embeddings",): ((batch, n_img, width), f32),
        ("image_token_valid",): ((batch, n_img), boolean),
        ("prefix_embeddings",): ((batch, prefix, width), f32),
        ("prefix_pad_masks",): ((batch, prefix), boolean),
        ("prefix_position_ids",): ((batch, prefix), i64),
        ("prefix_att_2d_masks",): ((batch, prefix, prefix), boolean),
        ("prefix_hidden",): ((batch, prefix, width), f32),
        ("vlm_logits",): ((batch, N_LOGIT_POSITIONS, vocab), f32),
        ("vlm_logits_prompt_index",): ((batch, N_LOGIT_POSITIONS), i64),
        ("vlm_logits_prefix_index",): ((batch, N_LOGIT_POSITIONS), i64),
        ("vlm_logits_input_ids",): ((batch, N_LOGIT_POSITIONS), i64),
        ("time_embedding",): ((batch, expert_width), f32),
        ("adarms_cond",): ((batch, expert_width), f32),
        ("suffix_position_ids",): ((batch, chunk), i64),
        ("suffix_att_2d_masks",): ((batch, chunk, prefix + chunk), boolean),
        ("v_t",): ((batch, chunk, action), f32),
        ("v_t_joint",): ((batch, chunk, action), f32),
        ("sample_actions",): ((batch, chunk, action), f32),
        ("sample_trajectory",): ((steps + 1, batch, chunk, action), f32),
        ("sample_velocities",): ((steps, batch, chunk, action), f32),
        ("sample_times",): ((steps,), f32),
        ("rope_inv_freq",): ((head_dim // 2,), f32),
    }
    shapes: Dict[str, Tuple[int, ...]] = {}
    for path, (shape, dtype) in expected.items():
        node = ref
        for part in path:
            node = node.get(part) if isinstance(node, Mapping) else None
        name = ".".join(path)
        if not isinstance(node, torch.Tensor):
            problems.append(f"{name} is missing or not a tensor ({_shape_of(node)})")
            continue
        shapes[name] = tuple(node.shape)
        if tuple(node.shape) != shape:
            problems.append(f"{name} has shape {tuple(node.shape)}, expected {shape}")
        if node.dtype != dtype:
            problems.append(f"{name} has dtype {node.dtype}, expected {dtype}")
        if node.device.type != "cpu":
            problems.append(f"{name} must be stored on CPU")
        if node.is_floating_point() and not bool(torch.isfinite(node).all()):
            problems.append(f"{name} contains non-finite values")

    for kind in ("prefix_keys", "prefix_values"):
        layers = ref.get(kind)
        if not isinstance(layers, (list, tuple)) or len(layers) != depth:
            problems.append(f"{kind} must be a list of {depth} tensors")
            continue
        for index, tensor in enumerate(layers):
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != (batch, 1, prefix, head_dim):
                problems.append(f"{kind}[{index}] must be [{batch}, 1, {prefix}, {head_dim}], got {_shape_of(tensor)}")
            elif tensor.dtype != f32 or not bool(torch.isfinite(tensor).all()):
                problems.append(f"{kind}[{index}] must be finite fp32")
        shapes[kind] = (len(layers),) + (tuple(layers[0].shape) if isinstance(layers[0], torch.Tensor) else ())

    if problems:
        raise ValueError("Invalid PI0.5 reference:\n - " + "\n - ".join(problems))

    # Internal consistency (cheap; the expensive checks live in the parity tests).
    inputs = ref["inputs"]
    images = inputs["images"]
    if float(images.min()) < -1.0 or float(images.max()) > 1.0:
        problems.append("images must lie in [-1, 1]")
    slot_valid = torch.tensor([bool(v) for v in meta["camera_slot_valid"]])
    if not torch.equal(inputs["img_masks"], slot_valid[None].expand(batch, slots)):
        problems.append("img_masks disagree with metadata camera_slot_valid")
    lang_masks, tokens = inputs["lang_masks"], inputs["lang_tokens"]
    lengths = lang_masks.sum(dim=1)
    if not torch.equal(lang_masks, torch.arange(lang)[None] < lengths[:, None]):
        problems.append("lang_masks must be a right-padded prefix pattern")
    if bool((tokens[~lang_masks] != PAD_ID).any()) or bool((tokens[:, 0] != BOS_ID).any()):
        problems.append("lang_tokens must start with BOS and use pad id 0 on padding")
    if [int(v) for v in lengths] != [int(v) for v in meta["lang_valid_lengths"]]:
        problems.append("metadata lang_valid_lengths disagree with lang_masks")
    token_valid = inputs["img_masks"].repeat_interleave(per_image, dim=1)
    if not torch.equal(ref["image_token_valid"], token_valid):
        problems.append("image_token_valid must repeat img_masks over each image's tokens")
    pad = ref["prefix_pad_masks"]
    if not torch.equal(pad, torch.cat((token_valid, lang_masks), dim=1)):
        problems.append("prefix_pad_masks must be [image token validity | lang_masks]")
    if not torch.equal(ref["prefix_position_ids"], torch.cumsum(pad.long(), dim=1) - 1):
        problems.append("prefix_position_ids must be cumsum(prefix_pad_masks) - 1")
    if not torch.equal(ref["prefix_att_2d_masks"], pad[:, None, :] & pad[:, :, None]):
        problems.append("prefix_att_2d_masks must be the bidirectional pad mask (pad rows see nothing)")
    pos0 = pad.sum(dim=1)
    if [int(v) for v in pos0] != [int(v) for v in meta["prefix_valid_counts"]]:
        problems.append("metadata prefix_valid_counts disagree with prefix_pad_masks")
    if not torch.equal(ref["suffix_position_ids"], pos0[:, None] + torch.arange(chunk)[None]):
        problems.append("suffix_position_ids must be pos0 + arange(chunk_size)")
    expected_suffix = torch.cat((pad[:, None, :].expand(batch, chunk, prefix),
                                 torch.ones((batch, chunk, chunk), dtype=torch.bool)), dim=2)
    if not torch.equal(ref["suffix_att_2d_masks"], expected_suffix):
        problems.append("suffix_att_2d_masks must be [prefix pad mask | all suffix keys]")
    if not torch.equal(ref["image_embeddings"], ref["prefix_embeddings"][:, :n_img]):
        problems.append("image_embeddings must equal the image part of prefix_embeddings (no extra scaling)")
    prompt_index = lengths[:, None] - N_LOGIT_POSITIONS + torch.arange(N_LOGIT_POSITIONS)[None]
    if not torch.equal(ref["vlm_logits_prompt_index"], prompt_index):
        problems.append("vlm_logits_prompt_index must be the last 4 valid prompt positions")
    if not torch.equal(ref["vlm_logits_prefix_index"], prompt_index + n_img):
        problems.append("vlm_logits_prefix_index must be n_image_tokens + vlm_logits_prompt_index")
    if not torch.equal(ref["vlm_logits_input_ids"], torch.gather(tokens, 1, prompt_index)):
        problems.append("vlm_logits_input_ids must be the prompt ids at the logit positions")
    if not torch.equal(inputs["t"], torch.tensor([float(v) for v in meta["t_values"]], dtype=torch.float32)):
        problems.append("inputs.t disagrees with metadata t_values")
    times = torch.tensor([1.0 + step * (-1.0 / steps) for step in range(steps)], dtype=torch.float32)
    if not torch.equal(ref["sample_times"], times):
        problems.append("sample_times must follow euler_integrate: 1 + k*(-1/steps)")
    trajectory = ref["sample_trajectory"]
    if not torch.equal(trajectory[0], inputs["noise"]):
        problems.append("sample_trajectory[0] must be the input noise")
    dt = -1.0 / steps
    for step in range(steps):
        if not torch.equal(trajectory[step + 1], trajectory[step] + dt * ref["sample_velocities"][step]):
            problems.append(f"sample_trajectory step {step} is not x + dt*v")
            break
    if not torch.allclose(trajectory[-1], ref["sample_actions"], rtol=0.0, atol=1e-5):
        problems.append("sample_trajectory[-1] must match sample_actions")
    if not math.isclose(float(meta["scaling"]["text_embed_scale"]), math.sqrt(width), rel_tol=1e-6):
        problems.append("text_embed_scale must be sqrt(vlm width)")
    if meta["keys_post_rope"] is not True:
        problems.append("metadata keys_post_rope must be True for the LeRobot/transformers cache")
    if problems:
        raise ValueError("Inconsistent PI0.5 reference:\n - " + "\n - ".join(problems))
    return shapes


# ----------------------------------------------------------------------------- comparison
ROW_TENSORS = ("prefix_hidden",)                       # [B, P, W]: rows indexed by prefix position
KEY_TENSORS = ("prefix_keys", "prefix_values")         # per layer [B, 1, P, hd]: keys indexed by prefix position
DIRECT_TENSORS = ("image_embeddings", "prefix_embeddings", "vlm_logits", "time_embedding", "adarms_cond", "v_t",
                  "v_t_joint", "sample_actions", "sample_trajectory", "sample_velocities")


def _diff_stats(candidate: torch.Tensor, reference: torch.Tensor) -> Dict[str, float]:
    """``max_abs``; ``max_rel_to_absmax`` (max |diff| / max |ref|); ``rel_fro`` (||diff||_F / ||ref||_F); ``mean_abs``."""
    if candidate.shape != reference.shape:
        raise ValueError(f"shape mismatch {tuple(candidate.shape)} vs {tuple(reference.shape)}")
    delta = candidate.double() - reference.double()
    diff = delta.abs()
    scale = float(reference.double().abs().max()) if reference.numel() else 0.0
    norm = float(reference.double().norm()) if reference.numel() else 0.0
    max_abs = float(diff.max()) if diff.numel() else 0.0
    fro = float(delta.norm()) if delta.numel() else 0.0
    return {"max_abs": max_abs, "ref_abs_max": scale, "max_rel_to_absmax": max_abs / scale if scale else max_abs,
            "rel_fro": fro / norm if norm else fro, "mean_abs": float(diff.mean()) if diff.numel() else 0.0}


def compare_references(candidate: Mapping, reference: Mapping, valid_only: bool = True) -> Dict[str, Dict[str, float]]:
    """Per-tensor differences ``candidate - reference`` (both in the reference layout).

    With ``valid_only``, ``prefix_hidden`` rows and ``prefix_keys``/``prefix_values`` positions are restricted
    to ``reference['prefix_pad_masks']``. LeRobot's pad rows, which cover prompt pads and the empty camera,
    attend uniformly to every key, so they are not comparable with layouts in which pad rows see the valid keys.
    Layer 0 K/V depend only on the embeddings and are always compared at every position.
    Tensors missing from ``candidate`` are skipped.
    """
    valid = reference["prefix_pad_masks"]
    out: Dict[str, Dict[str, float]] = {}
    for name in DIRECT_TENSORS:
        if name in candidate:
            out[name] = _diff_stats(candidate[name], reference[name])
    for name in ROW_TENSORS:
        if name in candidate:
            a, b = candidate[name], reference[name]
            out[name] = _diff_stats(a[valid], b[valid]) if valid_only else _diff_stats(a, b)
    for name in KEY_TENSORS:
        if name not in candidate:
            continue
        for layer, (a, b) in enumerate(zip(candidate[name], reference[name])):
            if valid_only and layer > 0:
                a, b = a[:, 0][valid], b[:, 0][valid]          # [n_valid, hd]
            out[f"{name}[{layer}]"] = _diff_stats(a, b)
    return out
