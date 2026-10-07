"""Strict loader for ``lerobot/pi05_base`` weights into the P2N-VLA backbone.

Every checkpoint tensor is either copied into a destination parameter or dropped
by an explicit, mode-specific allow-list; every destination parameter must be
filled. ``mode='flow'`` keeps PI0.5's adaRMS Dense layers and flow heads (parity
and the pi05_ki_flow baseline). ``mode='p2n'`` folds each expert adaRMS Dense at a
constant condition ``c0 = time_mlp(sincos(t0))`` into per-segment modulations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
import struct
from typing import Callable, Dict, List, Literal, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from oat.model.vla.flow_head import FlowHeads, posemb_sincos
from oat.model.vla.gemma_joint import AdaRMSNorm, GemmaJoint, SegmentRMSNorm
from oat.model.vla.siglip import SiglipEncoder

PI05_REPO = "lerobot/pi05_base"
PI05_REVISION = "b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba"
PI05_SHA256 = "0eb11ca9587678c1d2ef8cf32807c29f8ce53a2bfdfc1aa4a4c96f16fca59b0f"

SIGLIP_PREFIX = "paligemma_with_expert.paligemma.model.vision_tower."
PROJECTOR_PREFIX = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear."
VLM_PREFIX = "paligemma_with_expert.paligemma.model.language_model."
LM_HEAD = "paligemma_with_expert.paligemma.lm_head.weight"
EXPERT_PREFIX = "paligemma_with_expert.gemma_expert.model."
EXPERT_LM_HEAD = "paligemma_with_expert.gemma_expert.lm_head.weight"
FLOW_HEAD_NAMES = ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out")


def default_pi05_path(hf_home: Optional[str] = None) -> Path:
    import os
    root = Path(hf_home or os.environ.get("HF_HOME", "/workspace/.hf_home"))
    return root / "hub/models--lerobot--pi05_base/snapshots" / PI05_REVISION / "model.safetensors"


def read_header(path) -> Dict[str, Tuple[str, List[int]]]:
    """Parse the safetensors JSON header without touching tensor data."""
    with open(path, "rb") as stream:
        (length,) = struct.unpack("<Q", stream.read(8))
        header = json.loads(stream.read(length))
    header.pop("__metadata__", None)
    return {name: (info["dtype"], list(info["shape"])) for name, info in header.items()}


def file_sha256(path, chunk: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def resolved_blob_sha256(path) -> Optional[str]:
    """HF cache blobs are named by their sha256; use that instead of rehashing 14.5 GB."""
    name = Path(path).resolve().name
    return name if re.fullmatch(r"[0-9a-f]{64}", name) else None


@torch.no_grad()
def compute_c0(state: Mapping[str, Tensor], t0: float, width: int) -> Tensor:
    """``silu(time_mlp_out(silu(time_mlp_in(sincos(t0)))))`` in fp32, exactly as PI0.5."""
    if not 0.0 <= t0 <= 1.0:
        raise ValueError("t0 must lie in [0, 1]")
    embedding = posemb_sincos(torch.tensor([float(t0)]), width)
    hidden = F.silu(F.linear(embedding, state["time_mlp_in.weight"].float(), state["time_mlp_in.bias"].float()))
    return F.silu(F.linear(hidden, state["time_mlp_out.weight"].float(), state["time_mlp_out.bias"].float()))[0]


@dataclass
class LoadReport:
    mode: str
    consumed: List[str] = field(default_factory=list)
    dropped: List[str] = field(default_factory=list)
    folded: List[str] = field(default_factory=list)
    c0: Optional[Tensor] = None
    t0: Optional[float] = None
    sha256: Optional[str] = None


def _plain_targets(siglip: SiglipEncoder, joint: GemmaJoint, flow_heads: Optional[FlowHeads]) -> Dict[str, Tensor]:
    """Checkpoint key -> destination tensor for every directly copied parameter."""
    targets: Dict[str, Tensor] = {}
    for name, tensor in siglip.vision.state_dict(keep_vars=True).items():
        targets[SIGLIP_PREFIX + name] = tensor
    for name in ("weight", "bias"):
        targets[PROJECTOR_PREFIX + name] = getattr(siglip.projector, name)
    targets[LM_HEAD] = joint.vlm.embed_tokens.weight
    for index, layer in enumerate(joint.vlm.layers):
        base = f"{VLM_PREFIX}layers.{index}."
        targets[base + "input_layernorm.weight"] = layer.input_layernorm.weight
        targets[base + "post_attention_layernorm.weight"] = layer.post_attention_layernorm.weight
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            module = getattr(layer.self_attn, proj)
            if not isinstance(module, nn.Linear):
                raise RuntimeError("Load pi05 weights before injecting LoRA")
            targets[f"{base}self_attn.{proj}.weight"] = module.weight
        for proj in ("gate_proj", "up_proj", "down_proj"):
            module = getattr(layer.mlp, proj)
            if not isinstance(module, nn.Linear):
                raise RuntimeError("Load pi05 weights before injecting LoRA")
            targets[f"{base}mlp.{proj}.weight"] = module.weight
    targets[VLM_PREFIX + "norm.weight"] = joint.vlm.norm.weight
    for index, layer in enumerate(joint.expert.layers):
        base = f"{EXPERT_PREFIX}layers.{index}."
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            targets[f"{base}self_attn.{proj}.weight"] = getattr(layer.self_attn, proj).weight
        for proj in ("gate_proj", "up_proj", "down_proj"):
            targets[f"{base}mlp.{proj}.weight"] = getattr(layer.mlp, proj).weight
    if flow_heads is not None:
        for name in FLOW_HEAD_NAMES:
            module = getattr(flow_heads, name)
            targets[f"{name}.weight"] = module.weight
            targets[f"{name}.bias"] = module.bias
    return targets


def _expert_norms(joint: GemmaJoint) -> Dict[str, nn.Module]:
    norms = {}
    for index, layer in enumerate(joint.expert.layers):
        norms[f"{EXPERT_PREFIX}layers.{index}.input_layernorm.dense."] = layer.input_layernorm
        norms[f"{EXPERT_PREFIX}layers.{index}.post_attention_layernorm.dense."] = layer.post_attention_layernorm
    norms[f"{EXPERT_PREFIX}norm.dense."] = joint.expert.norm
    return norms


@torch.no_grad()
def load_pi05(path, *, siglip: SiglipEncoder, joint: GemmaJoint, mode: Literal["flow", "p2n"],
              t0: float = 0.6, flow_heads: Optional[FlowHeads] = None, strict: bool = True,
              expected_sha256: Optional[str] = None, verify_sha256: bool = False) -> LoadReport:
    from safetensors import safe_open

    if mode not in ("flow", "p2n"):
        raise ValueError("mode must be 'flow' or 'p2n'")
    expected_norm = AdaRMSNorm if mode == "flow" else SegmentRMSNorm
    if joint.norm_mode != ("adarms" if mode == "flow" else "segment"):
        raise ValueError(f"mode={mode!r} needs a GemmaJoint with norm_mode={'adarms' if mode == 'flow' else 'segment'}")
    if mode == "flow" and flow_heads is None:
        raise ValueError("mode='flow' needs flow_heads")
    if mode == "p2n" and flow_heads is not None:
        raise ValueError("mode='p2n' folds the time conditioning; flow_heads must be None")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"pi05 weights not found: {path}")
    sha = resolved_blob_sha256(path)
    if verify_sha256 or (expected_sha256 and sha is None):
        sha = file_sha256(path)
    if expected_sha256 is not None and sha != expected_sha256:
        raise ValueError(f"pi05 weights sha256 {sha} != expected {expected_sha256}")

    report = LoadReport(mode=mode, sha256=sha, t0=t0 if mode == "p2n" else None)
    targets = _plain_targets(siglip, joint, flow_heads)
    norms = _expert_norms(joint)
    filled = set()
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        state_for_c0 = {}
        if mode == "p2n":
            for name in ("time_mlp_in.weight", "time_mlp_in.bias", "time_mlp_out.weight", "time_mlp_out.bias"):
                if name not in keys:
                    raise KeyError(f"Checkpoint lacks {name}, needed to fold adaRMS")
                state_for_c0[name] = handle.get_tensor(name)
            report.c0 = compute_c0(state_for_c0, t0, joint.expert_spec.width)

        allowed_drops = {EXPERT_LM_HEAD}
        if mode == "p2n":
            allowed_drops |= {f"{name}.{suffix}" for name in FLOW_HEAD_NAMES for suffix in ("weight", "bias")}

        for key in sorted(keys):
            if key in targets:
                destination = targets[key]
                tensor = handle.get_tensor(key)
                if tuple(tensor.shape) != tuple(destination.shape):
                    raise ValueError(f"{key}: checkpoint shape {tuple(tensor.shape)} != {tuple(destination.shape)}")
                destination.data.copy_(tensor.to(device=destination.device, dtype=destination.dtype))
                filled.add(key)
                report.consumed.append(key)
                continue
            matched = next((prefix for prefix in norms if key.startswith(prefix)), None)
            if matched is not None:
                continue  # handled per norm below (weight + bias together)
            if key in allowed_drops:
                report.dropped.append(key)
                continue
            if strict:
                raise KeyError(f"Unexpected pi05 checkpoint tensor: {key}")
            report.dropped.append(key)

        for prefix, norm in norms.items():
            if not isinstance(norm, expected_norm):
                raise TypeError(f"{prefix} expected {expected_norm.__name__}")
            weight_key, bias_key = prefix + "weight", prefix + "bias"
            if weight_key not in keys or bias_key not in keys:
                raise KeyError(f"Checkpoint lacks {prefix}weight/bias")
            weight, bias = handle.get_tensor(weight_key).float(), handle.get_tensor(bias_key).float()
            width = norm.dim
            if weight.shape != (3 * width, joint.expert_spec.width) or bias.shape != (3 * width,):
                raise ValueError(f"{prefix} has unexpected shapes {tuple(weight.shape)}, {tuple(bias.shape)}")
            if prefix.endswith("model.norm.dense."):
                if weight[2 * width:].abs().max() != 0 or bias[2 * width:].abs().max() != 0:
                    raise ValueError("The expert final-norm gate rows must be zero (they are discarded)")
            if mode == "flow":
                norm.dense.weight.data.copy_(weight.to(norm.dense.weight.dtype))
                norm.dense.bias.data.copy_(bias.to(norm.dense.bias.dtype))
            else:
                modulation = weight @ report.c0 + bias
                norm.modulation.data.copy_(modulation[None].expand_as(norm.modulation).to(norm.modulation.dtype))
                report.folded.append(prefix)
            report.consumed.extend([weight_key, bias_key])
        if mode == "p2n":
            report.consumed.extend(sorted(state_for_c0))
            report.dropped = [key for key in report.dropped if key not in state_for_c0]

    missing = sorted(set(targets) - filled)
    if missing:
        raise KeyError(f"{len(missing)} destination tensors were not filled, e.g. {missing[:5]}")
    leftovers = sorted(keys - set(report.consumed) - set(report.dropped))
    if strict and leftovers:
        raise KeyError(f"Unaccounted pi05 tensors: {leftovers[:5]}")
    return report
