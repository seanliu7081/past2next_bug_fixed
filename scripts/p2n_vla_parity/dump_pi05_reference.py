#!/usr/bin/env python
"""Dump fp32 reference tensors from LeRobot's PyTorch PI0.5 (``PI05Policy`` / ``PI05Pytorch``).

This runs ONLY under the throwaway venv ``/venv/lerobot_ref``:
- Python 3.12 and torch 2.11 (cu128);
- lerobot at commit 2577da0ef39b47f870592d62d81edc8bce922cc4;
- transformers 5.5.x (lerobot pins >=5.4,<5.6).

Under ``/venv/oat`` (transformers 5.2), LeRobot's PI0.5 code silently mis-scales the text and image
embeddings by about 45x, so the script refuses to run there.

Exact command (run from the repository root /workspace/past2next_bug_fixed):

    CUDA_VISIBLE_DEVICES=1 HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1 \\
        /venv/lerobot_ref/bin/python scripts/p2n_vla_parity/dump_pi05_reference.py

- Measured on an RTX 4090: 16.2 GiB peak CUDA allocation, 20 GiB peak host RSS, and about 2 minutes,
  mostly the sha256 of the 14.5 GB file plus loading and verifying the weights.
- CPU fallback: add ``--device cpu`` and ``--output output/parity/pi05_reference_fp32_cpu.pt``. It
  measured 32 GiB peak RSS and about 2 minutes on this machine.
- GPU 0 is reserved. The script refuses CUDA unless ``CUDA_VISIBLE_DEVICES`` is set explicitly.
- Compare two dumps (e.g. CPU vs CUDA) with ``compare_pi05_references.py``.

What it does:
1. Builds ``PI05Policy`` from the local ``lerobot/pi05_base`` snapshot (revision b211f3d4) in fp32,
   in eval mode. There is no autocast, TF32 is off for matmul and cuDNN, and algorithms are deterministic.
   It then checks every stored tensor against ``model.safetensors``. LeRobot's ``from_pretrained``
   swallows load errors and would otherwise silently return random weights.
2. Bypasses LeRobot's preprocessor, which needs the gated HF PaliGemma tokenizer. Prompt ids instead
   come from the public big_vision SentencePiece model. The fixed seeded inputs are defined in
   ``pi05_reference_spec.py``:
   - B=2;
   - two 224x224 cameras, with the third pi05 slot filled at -1 and mask=False, as LeRobot does for
     missing cameras;
   - two prompts with BOS, right-padded with 0 to 64;
   - noise and x_t [2, 50, 32], and t = [0.3, 0.8].
3. Runs LeRobot's own entry points:
   - ``_preprocess_images``, ``_embed_images``, ``embed_prefix``;
   - the prefix-only ``PaliGemmaWithExpertModel.forward(use_cache=True)``, exactly as ``sample_actions``
     does it;
   - ``denoise_step``, ``embed_suffix``, ``sample_actions``.
   It also runs a training-style joint prefix+suffix pass as a cross-check.
4. Saves ``output/parity/pi05_reference_fp32.pt`` atomically, validates it with
   ``pi05_reference_spec.validate_reference`` and reloads it with ``weights_only=True``.

Keys in the cache are post-RoPE. transformers' ``GemmaAttention`` applies RoPE before
``DynamicCache.update``, and the script verifies this numerically on layer 0.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pi05_reference_spec as spec  # noqa: E402

PRESENT_CAMERA_KEYS = ("observation.images.base_0_rgb", "observation.images.left_wrist_0_rgb")


# ----------------------------------------------------------------------------- environment
def _version_tuple(text: str):
    parts = []
    for piece in text.split(".")[:2]:
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def environment_problems() -> list:
    """Reasons this interpreter cannot produce the reference (empty when it can)."""
    import importlib.util

    problems = []
    if sys.version_info < (3, 12):
        problems.append(f"Python {platform.python_version()} < 3.12 (lerobot main requires >= 3.12)")
    try:
        import transformers

        version = _version_tuple(transformers.__version__)
        if not ((5, 4) <= version < (5, 6)):
            problems.append(f"transformers {transformers.__version__} is outside lerobot's pin >=5.4,<5.6 "
                            "(5.2 mis-scales PI0.5 text and image embeddings by ~45x)")
    except ImportError:
        problems.append("transformers is not installed")
    if importlib.util.find_spec("lerobot") is None:
        problems.append("lerobot is not installed")
    if importlib.util.find_spec("sentencepiece") is None:
        problems.append("sentencepiece is not installed")
    return problems


def require_environment() -> None:
    problems = environment_problems()
    if problems:
        sys.stderr.write(
            "dump_pi05_reference.py must run under the throwaway venv /venv/lerobot_ref "
            f"(lerobot {spec.LEROBOT_COMMIT[:8]}, transformers 5.5, Python 3.12). Problems:\n - "
            + "\n - ".join(problems) + "\n")
        raise SystemExit(2)


def lerobot_commit() -> str:
    import importlib.metadata as metadata

    try:
        direct = metadata.distribution("lerobot").read_text("direct_url.json")
        return json.loads(direct)["vcs_info"]["commit_id"] if direct else "unknown"
    except Exception:  # noqa: BLE001 - provenance only
        return "unknown"



# ----------------------------------------------------------------------------- model
def load_policy(snapshot: Path, device: str):
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    if not (snapshot / "config.json").is_file() or not (snapshot / "model.safetensors").is_file():
        raise FileNotFoundError(f"pi05_base snapshot incomplete: {snapshot}")
    # Load from the local snapshot directory. With a repo id, LeRobot passes kwargs.get('revision')
    # (always None) to cached_file for model.safetensors, so offline loading would resolve 'main'.
    # This cache has no refs/main, and LeRobot would silently fall back to random weights.
    config = PreTrainedConfig.from_pretrained(str(snapshot))
    if config.type != "pi05":
        raise ValueError(f"Expected a pi05 config, got {config.type!r}")
    config.device = device                       # config.json says 'mps'
    if config.dtype != torch.float32:
        raise ValueError(f"pi05_base config dtype must be float32, got {config.dtype}")
    if config.compile_model or config.use_amp or config.gradient_checkpointing:
        raise ValueError("compile_model, use_amp and gradient_checkpointing must all be off for the reference")
    if config.rtc_config is not None or config.use_visual_memory or config.use_proprioceptive_memory:
        raise ValueError("RTC and MEM must be off for the reference")
    policy = PI05Policy.from_pretrained(str(snapshot), config=config)
    policy.eval()
    bad_dtypes = sorted({str(p.dtype) for p in policy.parameters()} - {"torch.float32"})
    if bad_dtypes:
        raise RuntimeError(f"All parameters must be fp32, found {bad_dtypes}")
    if policy.model.paligemma_with_expert.precision != torch.float32:
        raise RuntimeError("PaliGemmaWithExpertModel.precision must be float32 (vision autocast off)")
    return policy


@torch.no_grad()
def verify_weights(policy, weights: Path) -> dict:
    """Compare every model parameter with model.safetensors bit for bit (LeRobot swallows load errors)."""
    from safetensors import safe_open

    state = policy.state_dict()
    embed_key = "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    covered, missing, mismatched = set(), [], []
    compared = 0
    with safe_open(str(weights), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            targets = ["model." + key]
            if key == "paligemma_with_expert.paligemma.lm_head.weight":
                targets.append(embed_key)            # LeRobot copies the stored lm_head into embed_tokens
            for target in targets:
                if target not in state:
                    missing.append(target)
                    continue
                compared += 1
                covered.add(target)
                if not torch.equal(state[target].detach().to("cpu"), tensor):
                    mismatched.append(target)
    uncovered = sorted(name for name, _ in policy.named_parameters() if name not in covered)
    if missing or mismatched or uncovered:
        raise RuntimeError(f"pi05_base weights did not load faithfully: missing={missing[:5]} "
                           f"mismatched={mismatched[:5]} parameters_not_in_checkpoint={uncovered[:5]}")
    return {"tensors_compared": compared, "bitwise_equal": True, "parameters_covered": len(covered),
            "embed_tokens_equals_lm_head": True}


def check_geometry(policy) -> None:
    """The spec's pi05 geometry constants must describe the live model."""
    pwe = policy.model.paligemma_with_expert
    text, expert = pwe.paligemma.model.language_model.config, pwe.gemma_expert.model.config
    vision = pwe.paligemma.model.vision_tower.config
    got = {
        "vlm_width": text.hidden_size, "expert_width": expert.hidden_size,
        "depth": text.num_hidden_layers, "expert_depth": expert.num_hidden_layers,
        "head_dim": text.head_dim, "expert_head_dim": expert.head_dim,
        "num_heads": text.num_attention_heads, "num_kv_heads": text.num_key_value_heads,
        "vocab_size": pwe.paligemma.lm_head.out_features,
        "tokens_per_image": (vision.image_size // vision.patch_size) ** 2,
    }
    want = {
        "vlm_width": spec.VLM_WIDTH, "expert_width": spec.EXPERT_WIDTH, "depth": spec.DEPTH,
        "expert_depth": spec.DEPTH, "head_dim": spec.HEAD_DIM, "expert_head_dim": spec.HEAD_DIM,
        "num_heads": 8, "num_kv_heads": 1, "vocab_size": spec.VOCAB_SIZE, "tokens_per_image": spec.TOKENS_PER_IMAGE,
    }
    if got != want:
        raise RuntimeError(f"pi05 geometry differs from pi05_reference_spec: got {got}, expected {want}")


def describe_attention(policy) -> dict:
    pwe = policy.model.paligemma_with_expert
    return {
        "vlm_prefix": pwe.paligemma.model.language_model.config._attn_implementation,
        "expert": pwe.gemma_expert.model.config._attn_implementation,
        "vision": pwe.paligemma.model.vision_tower.config._attn_implementation,
        "joint_layers": "eager (modeling_gemma.eager_attention_forward, fp32_joint_attention=False)",
    }


# ----------------------------------------------------------------------------- reference computation
@torch.no_grad()
def compute_reference(policy, inputs: dict, device: str) -> dict:
    from lerobot.policies.common.vla_utils import (
        create_sinusoidal_pos_embedding, make_att_2d_masks)
    from transformers.models.gemma.modeling_gemma import apply_rotary_pos_emb

    model = policy.model
    pwe = model.paligemma_with_expert
    config = policy.config
    language_model = pwe.paligemma.model.language_model
    if model.training or pwe.training:
        raise RuntimeError("Model must be in eval mode")
    if any(torch.is_autocast_enabled(kind) for kind in ("cuda", "cpu")):
        raise RuntimeError("Autocast must be off")

    batch = inputs["lang_tokens"].shape[0]
    chunk = config.chunk_size
    slots = list(config.image_features)
    if tuple(slots[:len(PRESENT_CAMERA_KEYS)]) != PRESENT_CAMERA_KEYS:
        raise RuntimeError(f"Unexpected camera slots {slots}")
    camera_batch = {key: inputs["images01"][:, i].to(device) for i, key in enumerate(PRESENT_CAMERA_KEYS)}
    images, img_masks = policy._preprocess_images(camera_batch)        # LeRobot fills missing slots
    tokens = inputs["lang_tokens"].to(device)
    masks = inputs["lang_masks"].to(device)
    noise = inputs["noise"].to(device)
    x_t = inputs["x_t"].to(device)
    t = inputs["t"].to(device)

    # --- image tokens: SigLIP + projector (transformers >= 5.4 get_image_features: no scaling)
    image_embeds = torch.cat(model._embed_images(images, img_masks), dim=1)     # [B, n_slots*256, 2048]
    single_camera = pwe.embed_image(images[0])                                   # unbatched cross-check

    # --- prefix, exactly as PI05Pytorch.sample_actions builds it
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, tokens, masks)
    prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    mask_dtype = prefix_embs.dtype if model.use_typed_attention_masks else None
    prefix_att_4d = model._prepare_attention_masks_4d(prefix_att_2d, dtype=mask_dtype)
    language_model.config._attn_implementation = "eager"
    (prefix_hidden, _), past_key_values = pwe.forward(
        attention_mask=prefix_att_4d, position_ids=prefix_position_ids, past_key_values=None,
        inputs_embeds=[prefix_embs, None], use_cache=True)
    keys = [k.detach().clone() for k, _, _ in past_key_values]
    values = [v.detach().clone() for _, v, _ in past_key_values]

    # --- checks of the embedding conventions and of post-RoPE keys
    n_img = image_embeds.shape[1]
    embed_tokens = language_model.embed_tokens
    text_scale = embed_tokens.embed_scale.to(embed_tokens.weight.dtype)
    lang_ok = torch.equal(prefix_embs[:, n_img:], embed_tokens.weight[tokens] * text_scale)
    image_ok = torch.equal(prefix_embs[:, :n_img], image_embeds)
    layer0 = language_model.layers[0]
    normed, _ = layer0.input_layernorm(prefix_embs)
    k_pre = layer0.self_attn.k_proj(normed).view(batch, -1, 1, spec.HEAD_DIM).transpose(1, 2)
    v_pre = layer0.self_attn.v_proj(normed).view(batch, -1, 1, spec.HEAD_DIM).transpose(1, 2)
    cos, sin = language_model.rotary_emb(prefix_embs, prefix_position_ids)
    _, k_post = apply_rotary_pos_emb(k_pre, k_pre, cos, sin)
    post_rope_check = {
        "layer0_recomputed_post_rope_max_abs_diff": float((k_post - keys[0]).abs().max()),
        "layer0_recomputed_pre_rope_max_abs_diff": float((k_pre - keys[0]).abs().max()),
        "layer0_values_max_abs_diff": float((v_pre - values[0]).abs().max()),
        "key_abs_max": float(keys[0].abs().max()),
    }
    keys_post_rope = (post_rope_check["layer0_recomputed_post_rope_max_abs_diff"]
                      <= 1e-5 * max(1.0, post_rope_check["key_abs_max"])
                      < post_rope_check["layer0_recomputed_pre_rope_max_abs_diff"])
    if not (lang_ok and image_ok and keys_post_rope):
        raise RuntimeError(f"Convention check failed: lang_scaled={lang_ok} image_unscaled={image_ok} "
                           f"post_rope={post_rope_check}")

    # --- VLM logits for the last 4 valid prompt positions
    valid_len = masks.sum(dim=1)
    prompt_index = valid_len[:, None] - spec.N_LOGIT_POSITIONS + torch.arange(spec.N_LOGIT_POSITIONS, device=device)
    prefix_index = prompt_index + n_img
    rows = torch.arange(batch, device=device)[:, None]
    logits = pwe.paligemma.lm_head(prefix_hidden[rows, prefix_index]).float()
    input_ids = torch.gather(tokens, 1, prompt_index)

    # --- expert velocity on the cached prefix (PI05Pytorch.denoise_step)
    keys_before = [k.clone() for k in keys]
    v_t = model.denoise_step(prefix_pad_masks=prefix_pad_masks, past_key_values=past_key_values,
                             x_t=x_t, timestep=t)
    cache_unchanged = all(torch.equal(a, k) for a, (k, _, _) in zip(keys_before, past_key_values))
    suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = model.embed_suffix(x_t, t)
    time_embedding = create_sinusoidal_pos_embedding(
        t, model.action_in_proj.out_features, min_period=config.min_period, max_period=config.max_period,
        device=t.device).type(dtype=t.dtype)
    suffix_len = suffix_pad_masks.shape[1]
    prefix_len = prefix_pad_masks.shape[1]
    suffix_att_2d = torch.cat((prefix_pad_masks[:, None, :].expand(batch, suffix_len, prefix_len),
                               make_att_2d_masks(suffix_pad_masks, suffix_att_masks)), dim=2)
    suffix_position_ids = torch.sum(prefix_pad_masks, dim=-1)[:, None] + torch.cumsum(suffix_pad_masks, dim=1) - 1

    # --- training-style joint pass (PI05Pytorch.forward without the loss) as a cross-check
    pad_all = torch.cat((prefix_pad_masks, suffix_pad_masks), dim=1)
    att_all = torch.cat((prefix_att_masks, suffix_att_masks), dim=1)
    joint_4d = model._prepare_attention_masks_4d(make_att_2d_masks(pad_all, att_all))
    (_, suffix_out), _ = pwe.forward(
        attention_mask=joint_4d, position_ids=torch.cumsum(pad_all, dim=1) - 1, past_key_values=None,
        inputs_embeds=[prefix_embs, suffix_embs], use_cache=False, adarms_cond=[None, adarms_cond])
    v_t_joint = model.action_out_proj(suffix_out[:, -chunk:].to(torch.float32))

    # --- sample_actions: the official call, plus the same Euler loop with every step recorded
    actions = model.sample_actions(images, img_masks, tokens, masks, noise=noise)
    steps = config.num_inference_steps
    dt = -1.0 / steps
    trajectory, velocities, times = [noise], [], []
    x = noise
    for step in range(steps):
        time_value = 1.0 + step * dt                             # lerobot euler_integrate
        timestep = torch.tensor(time_value, dtype=torch.float32, device=device).expand(batch)
        velocity = model.denoise_step(prefix_pad_masks=prefix_pad_masks, past_key_values=past_key_values,
                                      x_t=x, timestep=timestep)
        x = x + dt * velocity
        trajectory.append(x)
        velocities.append(velocity)
        times.append(timestep[0])

    def cpu(tensor):
        return tensor.detach().to("cpu").contiguous()

    model_images = torch.stack(images, dim=1)                          # [B, n_slots, 3, S, S]
    model_masks = torch.stack(img_masks, dim=1)                        # [B, n_slots]
    top5 = torch.topk(logits[:, -1], k=5, dim=-1).indices
    return {
        "inputs": {
            "images": cpu(model_images), "img_masks": cpu(model_masks),
            "lang_tokens": cpu(tokens), "lang_masks": cpu(masks),
            "noise": cpu(noise), "x_t": cpu(x_t), "t": cpu(t),
        },
        "image_embeddings": cpu(image_embeds),
        "image_token_valid": cpu(prefix_pad_masks[:, :n_img]),
        "prefix_embeddings": cpu(prefix_embs),
        "prefix_pad_masks": cpu(prefix_pad_masks),
        "prefix_position_ids": cpu(prefix_position_ids),
        "prefix_att_2d_masks": cpu(prefix_att_2d),
        "prefix_hidden": cpu(prefix_hidden),
        "prefix_keys": [cpu(k) for k in keys],
        "prefix_values": [cpu(v) for v in values],
        "vlm_logits": cpu(logits),
        "vlm_logits_prompt_index": cpu(prompt_index),
        "vlm_logits_prefix_index": cpu(prefix_index),
        "vlm_logits_input_ids": cpu(input_ids),
        "time_embedding": cpu(time_embedding),
        "adarms_cond": cpu(adarms_cond),
        "suffix_position_ids": cpu(suffix_position_ids),
        "suffix_att_2d_masks": cpu(suffix_att_2d.bool()),
        "v_t": cpu(v_t),
        "v_t_joint": cpu(v_t_joint),
        "sample_actions": cpu(actions),
        "sample_trajectory": cpu(torch.stack(trajectory)),
        "sample_velocities": cpu(torch.stack(velocities)),
        "sample_times": cpu(torch.stack(times)),
        "rope_inv_freq": cpu(language_model.rotary_emb.inv_freq.float()),
        "_internal": {
            "text_embed_scale": float(text_scale),
            "post_rope_check": post_rope_check,
            "cache_unchanged_after_denoise": bool(cache_unchanged),
            "image_embed_unbatched_max_abs_diff": float((single_camera - image_embeds[:, :single_camera.shape[1]])
                                                        .abs().max()),
            "v_t_joint_vs_cached_max_abs_diff": float((v_t_joint - v_t).abs().max()),
            "v_t_abs_max": float(v_t.abs().max()),
            "sample_actions_loop_vs_official_max_abs_diff": float((trajectory[-1] - actions).abs().max()),
            "sample_actions_loop_bitwise_equal_official": bool(torch.equal(trajectory[-1], actions)),
            "top5_next_token_ids": cpu(top5).tolist(),
            "camera_slots": slots,
            "n_slots": len(images),
            "mlp_activation": type(language_model.layers[0].mlp.act_fn).__name__,
            "expert_mlp_activation": type(pwe.gemma_expert.model.layers[0].mlp.act_fn).__name__,
            "attention_scaling": float(language_model.layers[0].self_attn.scaling),
            "rms_norm_eps": float(language_model.norm.eps),
            "rope_theta": float(language_model.config.rope_parameters["rope_theta"]),
        },
    }


def tensor_tree_diff(first, second, prefix="") -> dict:
    """Max abs difference per tensor between two payloads; only shared tensor leaves are compared."""
    out = {}
    if isinstance(first, dict):
        for key in first:
            if key != "_internal" and key in second:
                out.update(tensor_tree_diff(first[key], second[key], f"{prefix}{key}."))
    elif isinstance(first, (list, tuple)):
        for index, (a, b) in enumerate(zip(first, second)):
            out.update(tensor_tree_diff(a, b, f"{prefix}{index}."))
    elif isinstance(first, torch.Tensor):
        name = prefix.rstrip(".")
        if first.shape != second.shape:
            out[name] = float("inf")
        elif first.is_floating_point():
            out[name] = float((first - second).abs().max()) if first.numel() else 0.0
        else:
            out[name] = 0.0 if torch.equal(first, second) else float("inf")
    return out


# ----------------------------------------------------------------------------- main
def build_metadata(args, policy, result, environment, fp32_state, weights_check, sha_info, reproducibility,
                   spm_sha) -> dict:
    import importlib.metadata as metadata

    internal = result["_internal"]
    config = policy.config
    inputs = result["inputs"]
    pad = result["prefix_pad_masks"]
    n_img = result["image_embeddings"].shape[1]
    per_image = n_img // internal["n_slots"]
    spm_pieces = []
    try:
        import sentencepiece

        processor = sentencepiece.SentencePieceProcessor(model_file=str(args.spm))
        spm_pieces = [[processor.id_to_piece(int(i)) for i in row] for row in internal["top5_next_token_ids"]]
    except Exception:  # noqa: BLE001 - cosmetic only
        pass

    def version(name):
        try:
            return metadata.version(name)
        except Exception:  # noqa: BLE001
            return "unknown"

    return {
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/p2n_vla_parity/dump_pi05_reference.py",
        "script_sha256": spec.sha256_file(Path(__file__).resolve()),
        "spec_sha256": spec.sha256_file(Path(spec.__file__).resolve()),
        "command": " ".join([sys.executable] + sys.argv),
        "python_version": platform.python_version(),
        "sys_prefix": sys.prefix,
        "lerobot_commit": environment["lerobot_commit"],
        "lerobot_commit_expected": spec.LEROBOT_COMMIT,
        "lerobot_version": version("lerobot"),
        "transformers_version": version("transformers"),
        "torch_version": str(torch.__version__),     # TorchVersion is not weights_only-loadable
        "torch_cuda": str(torch.version.cuda) if torch.version.cuda else None,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "numpy_version": version("numpy"),
        "safetensors_version": version("safetensors"),
        "sentencepiece_version": version("sentencepiece"),
        "device": args.device,
        "gpu_name": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "hf_home": os.environ.get("HF_HOME"),
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
        "pi05_repo": spec.PI05_REPO,
        "pi05_revision": spec.PI05_REVISION,
        "pi05_snapshot": str(args.snapshot),
        "pi05_weights": str((args.snapshot / "model.safetensors").resolve()),
        **sha_info,
        "spm_path": str(args.spm),
        "spm_sha256": spm_sha,
        "weights_check": weights_check,
        "dtype": "float32",
        "autocast": False,
        "fp32_precision": fp32_state,
        "attn_implementation": describe_attention(policy),
        "geometry": {"vlm_width": spec.VLM_WIDTH, "expert_width": spec.EXPERT_WIDTH, "depth": spec.DEPTH,
                     "head_dim": spec.HEAD_DIM, "vocab_size": spec.VOCAB_SIZE, "num_heads": 8, "num_kv_heads": 1,
                     "mlp_activation": internal["mlp_activation"],
                     "expert_mlp_activation": internal["expert_mlp_activation"]},
        "chunk_size": int(config.chunk_size),
        "max_action_dim": int(config.max_action_dim),
        "num_inference_steps": int(config.num_inference_steps),
        "min_period": float(config.min_period),
        "max_period": float(config.max_period),
        "camera_slots": internal["camera_slots"],
        "camera_slot_valid": [bool(v) for v in inputs["img_masks"][0].tolist()],
        "empty_camera_fill": -1.0,
        "n_tokens_per_image": per_image,
        "n_image_tokens": n_img,
        "lang_max_len": int(inputs["lang_tokens"].shape[1]),
        "prefix_len": int(pad.shape[1]),
        "lang_valid_lengths": [int(v) for v in inputs["lang_masks"].sum(dim=1)],
        "prefix_valid_counts": [int(v) for v in pad.sum(dim=1)],
        "keys_post_rope": True,
        "post_rope_check": internal["post_rope_check"],
        "position_construction": (
            "prefix_position_ids = cumsum(prefix_pad_masks) - 1 over [cam0 256 | cam1 256 | cam2 256 (masked) | "
            "prompt 64]. Masked tokens repeat the previous position, so the empty camera's tokens sit at 511, prompt "
            "token j at 512+j, and prompt pads repeat the last valid position. Suffix (denoise_step) positions are "
            "sum(prefix_pad_masks) + arange(chunk_size), i.e. pos0 + k with pos0 = number of valid prefix tokens."),
        "mask_construction": (
            "Prefix: att_masks all 0 -> one bidirectional block; make_att_2d_masks ANDs with pad[:,None,:]&pad[:,:,None], "
            "so valid rows see all valid keys and pad rows (prompt pads AND all 256 tokens of the empty camera) see "
            "NO key: their additive row is all -2.3819763e38, which gives uniform attention over all P keys. Their hidden "
            "states and their K/V at layers >= 1 are therefore not comparable with the repo layout, where pad rows see "
            "valid keys + own diagonal. Compare valid rows/keys only. Additive 4D masks are torch.where(mask, 0.0, "
            "-2.3819763e38) in fp32. Suffix: rows see valid prefix keys + all 50 action keys (bidirectional), "
            "suffix_att_2d_masks [B, 50, P+50]."),
        "scaling": {
            "text_embed_scale": internal["text_embed_scale"],
            "text_embed_scale_note": "GemmaTextScaledWordEmbedding: weight[ids] * fp32(sqrt(2048)) buffer",
            "image_embed_scale": 1.0,
            "image_embed_note": "projector(SigLIP last_hidden_state) with no division by sqrt(2048) (transformers >= 5.4)",
            "attention_scaling": internal["attention_scaling"],
            "attention_note": "eager: (q @ k^T) * scaling + mask; softmax in fp32; q,k post-RoPE; k/v repeat_kv to 8 heads",
            "rms_norm_eps": internal["rms_norm_eps"],
            "rms_norm_note": "VLM: x*rsqrt(mean(x^2)+eps)*(1+w); expert adaRMS: x_hat*(1+scale)+shift, gate on residual",
            "rope_theta": internal["rope_theta"],
            "rope_note": ("inv_freq = 1/(theta**(arange(0,256,2)/256)) computed on CPU at init (saved as rope_inv_freq); "
                          "freqs = inv_freq @ positions (fp32), emb=cat(freqs,freqs); rotate_half"),
            "mask_value": -2.3819763e38,
            "time_embedding_note": ("create_sinusoidal_pos_embedding in float64 on CUDA, then cast to fp32: "
                                    "fraction=linspace(0,1,512); period=4e-3*(4/4e-3)**fraction; "
                                    "cat(sin(t*2pi/period), cos(...)); adarms_cond = silu(time_mlp_out(silu(time_mlp_in(.))))"),
        },
        "seed": int(args.seed),
        "prompts": list(spec.PROMPTS),
        "t_values": [float(v) for v in spec.T_VALUES],
        "input_recipe": spec.INPUT_RECIPE,
        "logit_positions": ("vlm_logits[b, j] = lm_head(prefix_hidden[b, vlm_logits_prefix_index[b, j]]) for the last 4 "
                            "valid prompt tokens (prompt index valid_len-4 .. valid_len-1); prefix index = 768 + prompt "
                            "index. The last row predicts the token after 'Action: '."),
        "top5_next_token_ids": internal["top5_next_token_ids"],
        "top5_next_token_pieces": spm_pieces,
        "top5_next_token_note": ("pi05_base's VLM predicts ids in openpi's FAST action range 257023-t "
                                 "(254976..257023) after 'Action: ', i.e. its FAST pre-training prior."),
        "diagnostics": {
            "cache_unchanged_after_denoise": internal["cache_unchanged_after_denoise"],
            "image_embed_unbatched_max_abs_diff": internal["image_embed_unbatched_max_abs_diff"],
            "image_embed_abs_max": float(result["image_embeddings"].abs().max()),
            "v_t_joint_vs_cached_max_abs_diff": internal["v_t_joint_vs_cached_max_abs_diff"],
            "v_t_abs_max": internal["v_t_abs_max"],
            "sample_actions_loop_vs_official_max_abs_diff": internal["sample_actions_loop_vs_official_max_abs_diff"],
            "sample_actions_loop_bitwise_equal_official": internal["sample_actions_loop_bitwise_equal_official"],
            "cuda_max_memory_allocated_gib": (torch.cuda.max_memory_allocated() / 2**30
                                              if args.device == "cuda" else None),
            **reproducibility,
        },
        "deviations": [
            "lang_max_len=64 instead of 48: prompt 0 is 51 SentencePiece tokens with BOS, so it cannot be "
            "right-padded to 48 without truncation. 64 leaves 13 and 29 pads.",
            "TF32 is disabled for matmul and cuDNN conv (LeRobot leaves cuDNN TF32 on by default), so this is a true fp32 "
            "reference.",
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=spec.DEFAULT_OUTPUT)
    parser.add_argument("--snapshot", type=Path, default=spec.PI05_SNAPSHOT)
    parser.add_argument("--spm", type=Path, default=spec.DEFAULT_SPM)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=spec.SEED)
    parser.add_argument("--max-len", type=int, default=spec.LANG_MAX_LEN)
    parser.add_argument("--repeat", type=int, default=2,
                        help="compute the reference this many times and require bitwise-identical tensors")
    parser.add_argument("--skip-sha256", action="store_true", help="skip hashing the 14.5 GB safetensors file")
    parser.add_argument("--skip-weight-check", action="store_true",
                        help="skip the tensor-by-tensor comparison against model.safetensors")
    parser.add_argument("--check-env", action="store_true", help="only check the interpreter environment")
    args = parser.parse_args(argv)
    started = time.perf_counter()

    require_environment()
    if args.check_env:
        print(f"environment OK: {sys.executable}")
        return 0
    if args.repeat < 1:
        parser.error("--repeat must be >= 1")
    if args.device == "cuda":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if not visible:
            parser.error("--device cuda needs an explicit CUDA_VISIBLE_DEVICES (GPU 0 is reserved; use "
                         "CUDA_VISIBLE_DEVICES=1)")
    os.environ.setdefault("HF_HOME", str(spec.HF_HOME))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    fp32_state = spec.configure_ieee_fp32(deterministic=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available")

    spm_sha = spec.sha256_file(args.spm)
    if spm_sha != spec.SPM_SHA256:
        raise RuntimeError(f"SentencePiece sha256 mismatch: {spm_sha}")
    weights = args.snapshot / "model.safetensors"
    sha_info = {"pi05_sha256_expected": spec.PI05_SHA256, "pi05_sha256_verified": False}
    if not args.skip_sha256:
        print(f"hashing {weights} ...", flush=True)
        digest = spec.sha256_file(weights)
        if digest != spec.PI05_SHA256:
            raise RuntimeError(f"pi05_base sha256 mismatch: {digest}")
        sha_info.update({"pi05_sha256": digest, "pi05_sha256_verified": True})

    environment = {"lerobot_commit": lerobot_commit()}
    if environment["lerobot_commit"] != spec.LEROBOT_COMMIT:
        print(f"WARNING: lerobot commit {environment['lerobot_commit']} != pinned {spec.LEROBOT_COMMIT}",
              file=sys.stderr)

    inputs = spec.build_reference_inputs(args.spm, max_len=args.max_len, seed=args.seed)
    for row, expected in enumerate(spec.EXPECTED_PROMPT_IDS):
        got = inputs["lang_tokens"][row][inputs["lang_masks"][row]].tolist()
        if tuple(got) != expected:
            raise RuntimeError(f"Prompt {row} tokenizes differently than the recorded ids")

    print("building PI05Policy from the local pi05_base snapshot (fp32) ...", flush=True)
    policy = load_policy(args.snapshot, args.device)
    if policy.config.chunk_size != spec.CHUNK_SIZE or policy.config.max_action_dim != spec.ACTION_DIM:
        raise RuntimeError("pi05_base chunk_size/max_action_dim differ from the spec constants")
    check_geometry(policy)
    weights_check = {"skipped": True}
    if not args.skip_weight_check:
        print("verifying every tensor against model.safetensors ...", flush=True)
        weights_check = verify_weights(policy, weights)

    results = []
    for _ in range(args.repeat):
        results.append(compute_reference(policy, inputs, args.device))
    if not spec.ieee_fp32_active():
        raise RuntimeError("TF32 was re-enabled during the computation (by LeRobot?); the reference is not pure fp32")
    result = results[0]
    reproducibility = {"repeats": args.repeat}
    if args.repeat > 1:
        diffs = {}
        for other in results[1:]:
            for name, value in tensor_tree_diff(result, other).items():
                diffs[name] = max(diffs.get(name, 0.0), value)
        worst = max(diffs.values()) if diffs else 0.0
        reproducibility.update({"reproducible_bitwise": worst == 0.0, "reproducibility_max_abs_diff": worst})
        if worst != 0.0:
            print(f"WARNING: repeated computation differs (max abs {worst:.3e})", file=sys.stderr)
    reproducibility["wall_seconds_to_results"] = round(time.perf_counter() - started, 1)
    reproducibility["host_max_rss_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)

    metadata = build_metadata(args, policy, result, environment, fp32_state, weights_check, sha_info,
                              reproducibility, spm_sha)
    payload = {key: value for key, value in result.items() if key != "_internal"}
    payload["format"] = spec.REFERENCE_FORMAT
    payload["metadata"] = metadata
    regenerated_images, regenerated_masks = spec.to_model_images(inputs["images01"], len(metadata["camera_slots"]))
    if not (torch.equal(payload["inputs"]["images"], regenerated_images)
            and torch.equal(payload["inputs"]["img_masks"], regenerated_masks)):
        raise RuntimeError("LeRobot's image preprocessing differs from pi05_reference_spec.to_model_images")
    spec.validate_reference(payload)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(args.output.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, args.output)
    reloaded = torch.load(args.output, map_location="cpu", weights_only=True)
    shapes = spec.validate_reference(reloaded)
    print(f"wrote {args.output} ({args.output.stat().st_size / 2**20:.1f} MiB)")
    for name, shape in sorted(shapes.items()):
        print(f"  {name}: {shape}")
    print(json.dumps(metadata["diagnostics"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
