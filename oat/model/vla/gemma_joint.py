"""PaliGemma (Gemma-2B) prefix stack and PI0.5 action-expert stack with shared per-layer attention.

Training runs two passes that are exactly PI0.5's joint pass: the VLM prefix
first (prefix rows never read the suffix), then the expert over the cached,
caller-detached prefix K/V. Conventions follow openpi/LeRobot PI0.5: Gemma
RMSNorm ``x̂·(1+w)``, rotate-half RoPE (theta 1e4), GeGLU, MQA, text embeddings
scaled by sqrt(width), adaRMS ``x̂·(1+scale)+shift`` with a raw-gate residual.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from oat.model.vla.layout import N_SEGMENTS, PrefixLayout
from oat.model.vla.specs import GemmaSpec, check_joint_compatible

KeyValue = Tuple[Tensor, Tensor]


def _rms(x: Tensor, eps: float) -> Tensor:
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        return (_rms(x, self.eps) * (1.0 + self.weight.float())).to(x.dtype)


class AdaRMSNorm(nn.Module):
    """PI0.5 adaptive RMSNorm: ``[scale|shift|gate] = Dense(cond)``."""

    def __init__(self, dim: int, cond_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim, self.eps = dim, eps
        self.dense = nn.Linear(cond_dim, 3 * dim)
        nn.init.zeros_(self.dense.weight)
        with torch.no_grad():  # random-init default: identity norm with an open residual gate
            self.dense.bias.zero_()
            self.dense.bias[2 * dim:] = 1.0

    def modulation(self, cond: Tensor) -> Tensor:
        with torch.autocast(device_type=cond.device.type, enabled=False):
            return F.linear(cond.float(), self.dense.weight.float(), self.dense.bias.float())

    def forward(self, x: Tensor, cond: Tensor) -> Tuple[Tensor, Tensor]:
        mod = self.modulation(cond)
        if mod.ndim == 2:
            mod = mod[:, None, :]
        scale, shift, gate = mod.chunk(3, dim=-1)
        y = _rms(x, self.eps) * (1.0 + scale) + shift
        return y.to(x.dtype), gate.to(x.dtype)


class SegmentRMSNorm(nn.Module):
    """adaRMS with the Dense folded at a constant condition, one modulation per segment."""

    def __init__(self, dim: int, n_segments: int = N_SEGMENTS, eps: float = 1e-6):
        super().__init__()
        self.dim, self.eps = dim, eps
        modulation = torch.zeros(n_segments, 3 * dim)
        modulation[:, 2 * dim:] = 1.0  # random-init default; the pi05 loader overwrites it
        self.modulation = nn.Parameter(modulation)

    def forward(self, x: Tensor, seg_ids: Tensor) -> Tuple[Tensor, Tensor]:
        if seg_ids.ndim != 1 or seg_ids.shape[0] != x.shape[-2]:
            raise ValueError("seg_ids must be [S] and match the token axis")
        mod = self.modulation.float()[seg_ids][None]  # [1, S, 3*dim]
        scale, shift, gate = mod.chunk(3, dim=-1)
        y = _rms(x, self.eps) * (1.0 + scale) + shift
        return y.to(x.dtype), gate.to(x.dtype)


class GemmaAttentionProjections(nn.Module):
    def __init__(self, width: int, num_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.q_proj = nn.Linear(width, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(width, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(width, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, width, bias=False)


class GemmaMLP(nn.Module):
    def __init__(self, width: int, mlp_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(width, mlp_dim, bias=False)
        self.up_proj = nn.Linear(width, mlp_dim, bias=False)
        self.down_proj = nn.Linear(mlp_dim, width, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class GemmaLayer(nn.Module):
    def __init__(self, spec: GemmaSpec):
        super().__init__()
        self.input_layernorm = GemmaRMSNorm(spec.width, spec.eps)
        self.self_attn = GemmaAttentionProjections(spec.width, spec.num_heads, spec.num_kv_heads, spec.head_dim)
        self.post_attention_layernorm = GemmaRMSNorm(spec.width, spec.eps)
        self.mlp = GemmaMLP(spec.width, spec.mlp_dim)


class ExpertLayer(nn.Module):
    def __init__(self, spec: GemmaSpec, norm_mode: str, n_segments: int):
        super().__init__()
        norm = (lambda: AdaRMSNorm(spec.width, spec.width, spec.eps)) if norm_mode == "adarms" else \
            (lambda: SegmentRMSNorm(spec.width, n_segments, spec.eps))
        self.input_layernorm = norm()
        self.self_attn = GemmaAttentionProjections(spec.width, spec.num_heads, spec.num_kv_heads, spec.head_dim)
        self.post_attention_layernorm = norm()
        self.mlp = GemmaMLP(spec.width, spec.mlp_dim)


class GemmaStack(nn.Module):
    def __init__(self, spec: GemmaSpec):
        super().__init__()
        self.embed_tokens = nn.Embedding(spec.vocab_size, spec.width)
        self.layers = nn.ModuleList([GemmaLayer(spec) for _ in range(spec.depth)])
        self.norm = GemmaRMSNorm(spec.width, spec.eps)


class ExpertStack(nn.Module):
    def __init__(self, spec: GemmaSpec, norm_mode: str, n_segments: int):
        super().__init__()
        self.layers = nn.ModuleList([ExpertLayer(spec, norm_mode, n_segments) for _ in range(spec.depth)])
        self.norm = (AdaRMSNorm(spec.width, spec.width, spec.eps) if norm_mode == "adarms"
                     else SegmentRMSNorm(spec.width, n_segments, spec.eps))


def rope_inv_freq(head_dim: int, theta: float) -> Tensor:
    """fp32 RoPE inverse frequencies ``theta^(-arange(0, hd, 2)/hd)``, always built on the CPU.

    transformers (and therefore LeRobot PI0.5) builds this table on the CPU and moves it to the model's
    device. CUDA's ``pow`` differs from the CPU kernel by 1 ulp in a few entries (4 of 128 at head_dim 256,
    up to 2e-6 rad at position ~560), so building it on the device made RoPE device-dependent and broke
    bitwise parity of the post-RoPE keys with the reference.
    """
    exponent = torch.arange(0, head_dim, 2, dtype=torch.int64).to(torch.float32) / head_dim
    return 1.0 / (theta ** exponent)


def rope_cos_sin(positions: Tensor, head_dim: int, theta: float) -> Tuple[Tensor, Tensor]:
    """Rotary tables [B, 1, L, head_dim] in fp32, rebuilt per call (never memoized)."""
    # The pageable host-to-device copy is staged by the driver, so non_blocking never stalls the host.
    inv_freq = rope_inv_freq(head_dim, theta).to(positions.device, non_blocking=True)
    freqs = positions.to(torch.float32)[..., None] * inv_freq
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos()[:, None], emb.sin()[:, None]


def _rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    xf = x.float()
    return (xf * cos + _rotate_half(xf) * sin).to(x.dtype)


@dataclass(frozen=True)
class PrefixOutput:
    hidden: Tensor                      # [B, P, Wv] after the final norm
    kv: Tuple[KeyValue, ...]            # per layer, [B, kv_heads, nonki_len, hd] post-RoPE
    pos0: Tensor                        # [B] long


@dataclass(frozen=True)
class ExpertCache:
    kv: Tuple[KeyValue, ...]            # per layer, suffix keys/values so far
    length: int


@dataclass(frozen=True)
class ExpertOutput:
    hidden: Tensor                      # [B, S, We] after the final norm
    cache: Optional[ExpertCache]
    probe_mass: Optional[Tensor]        # [n_layers]


class GemmaJoint(nn.Module):
    def __init__(self, vlm: GemmaSpec, expert: GemmaSpec, norm_mode: Literal["adarms", "segment"],
                 n_segments: int = N_SEGMENTS, frozen_dtype: torch.dtype = torch.bfloat16,
                 activation_checkpointing: bool = True):
        super().__init__()
        check_joint_compatible(vlm, expert)
        if norm_mode not in ("adarms", "segment"):
            raise ValueError("norm_mode must be 'adarms' or 'segment'")
        self.vlm_spec, self.expert_spec = vlm, expert
        self.norm_mode = norm_mode
        self.activation_checkpointing = bool(activation_checkpointing)
        self.vlm = GemmaStack(vlm)
        self.expert = ExpertStack(expert, norm_mode, n_segments)
        # Frozen PaliGemma linear/embedding weights in frozen_dtype; norms stay fp32.
        for name, parameter in self.vlm.named_parameters():
            parameter.requires_grad_(False)
            if not name.endswith("layernorm.weight") and name != "norm.weight":
                parameter.data = parameter.data.to(frozen_dtype)

    # ------------------------------------------------------------------ prefix
    def embed_text(self, ids: Tensor) -> Tensor:
        if ids.ndim != 2 or ids.dtype != torch.long:
            raise ValueError("ids must be a long tensor [B, L]")
        embeds = self.vlm.embed_tokens(ids)
        return embeds * torch.tensor(math.sqrt(self.vlm_spec.width), dtype=embeds.dtype, device=embeds.device)

    def vlm_logits(self, hidden: Tensor) -> Tensor:
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            return hidden.float() @ self.vlm.embed_tokens.weight.float().T

    def _vlm_layer(self, layer: GemmaLayer, x: Tensor, cos: Tensor, sin: Tensor, bias: Tensor):
        spec = self.vlm_spec
        batch, length, _ = x.shape
        h = layer.input_layernorm(x)
        q = layer.self_attn.q_proj(h).view(batch, length, spec.num_heads, spec.head_dim).transpose(1, 2)
        k = layer.self_attn.k_proj(h).view(batch, length, spec.num_kv_heads, spec.head_dim).transpose(1, 2)
        v = layer.self_attn.v_proj(h).view(batch, length, spec.num_kv_heads, spec.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if spec.num_kv_heads == 1:
            # Stride-0 expansion keeps the mem-efficient SDPA kernel (GQA + mask -> math only).
            k_full = k.expand(batch, spec.num_heads, length, spec.head_dim)
            v_full = v.expand(batch, spec.num_heads, length, spec.head_dim)
        else:
            repeat = spec.num_heads // spec.num_kv_heads
            k_full, v_full = k.repeat_interleave(repeat, dim=1), v.repeat_interleave(repeat, dim=1)
        out = F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=bias.to(q.dtype))
        out = out.transpose(1, 2).reshape(batch, length, spec.num_heads * spec.head_dim)
        x = x + layer.self_attn.o_proj(out)
        x = x + layer.mlp(layer.post_attention_layernorm(x))
        return x, k, v

    def prefix_forward(self, embeds: Tensor, layout: PrefixLayout) -> PrefixOutput:
        batch, length, width = embeds.shape
        if width != self.vlm_spec.width:
            raise ValueError(f"Prefix embeddings must have width {self.vlm_spec.width}")
        if layout.bias.shape != (batch, 1, length, length) or layout.positions.shape != (batch, length):
            raise ValueError("Prefix layout does not match the embeddings")
        cos, sin = rope_cos_sin(layout.positions, self.vlm_spec.head_dim, self.vlm_spec.rope_theta)
        use_checkpoint = self.activation_checkpointing and self.training and torch.is_grad_enabled()
        x, kv = embeds, []
        for layer in self.vlm.layers:
            if use_checkpoint:
                def run(value, current=layer):
                    return self._vlm_layer(current, value, cos, sin, layout.bias)
                x, k, v = checkpoint(run, x, use_reentrant=False)
            else:
                x, k, v = self._vlm_layer(layer, x, cos, sin, layout.bias)
            kv.append((k[:, :, :layout.nonki_len], v[:, :, :layout.nonki_len]))
        return PrefixOutput(self.vlm.norm(x), tuple(kv), layout.pos0)

    # ------------------------------------------------------------------ expert
    def _norm(self, norm: nn.Module, x: Tensor, seg_ids: Optional[Tensor], cond: Optional[Tensor]):
        return norm(x, seg_ids) if self.norm_mode == "segment" else norm(x, cond)

    def _expert_layer(self, layer: ExpertLayer, x: Tensor, prefix_kv: KeyValue, cached: Optional[KeyValue],
                      bias: Tensor, cos: Tensor, sin: Tensor, seg_ids, cond, probe_cols):
        spec = self.expert_spec
        batch, length, _ = x.shape
        h, attn_gate = self._norm(layer.input_layernorm, x, seg_ids, cond)
        q = layer.self_attn.q_proj(h).view(batch, length, spec.num_heads, spec.head_dim).transpose(1, 2)
        k = layer.self_attn.k_proj(h).view(batch, length, spec.num_kv_heads, spec.head_dim).transpose(1, 2)
        v = layer.self_attn.v_proj(h).view(batch, length, spec.num_kv_heads, spec.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        keys = [prefix_kv[0].to(k.dtype)] + ([cached[0]] if cached is not None else []) + [k]
        values = [prefix_kv[1].to(v.dtype)] + ([cached[1]] if cached is not None else []) + [v]
        keys, values = torch.cat(keys, dim=2), torch.cat(values, dim=2)
        if keys.shape[2] != bias.shape[-1]:
            raise ValueError(f"Expert bias covers {bias.shape[-1]} keys, attention has {keys.shape[2]}")
        with torch.autocast(device_type=x.device.type, enabled=False):
            qf, kf, vf = q.float(), keys.float(), values.float()
            if spec.num_kv_heads not in (1, spec.num_heads):
                repeat = spec.num_heads // spec.num_kv_heads
                kf, vf = kf.repeat_interleave(repeat, dim=1), vf.repeat_interleave(repeat, dim=1)
            logits = (qf * spec.head_dim ** -0.5) @ kf.transpose(-1, -2) + bias.float()
            probs = logits.softmax(dim=-1)
            out = probs @ vf
            mass = None
            if probe_cols is not None:
                start, stop = probe_cols
                mass = probs[:, :, -1, start:stop].sum(dim=-1).mean().detach()
        out = out.to(h.dtype).transpose(1, 2).reshape(batch, length, spec.num_heads * spec.head_dim)
        x = x + layer.self_attn.o_proj(out) * attn_gate
        h, mlp_gate = self._norm(layer.post_attention_layernorm, x, seg_ids, cond)
        x = x + layer.mlp(h) * mlp_gate
        return x, k, v, mass

    def expert_forward(self, x: Tensor, prefix_kv: Sequence[KeyValue], layout_bias: Tensor, positions: Tensor, *,
                       seg_ids: Optional[Tensor] = None, cond: Optional[Tensor] = None,
                       cache: Optional[ExpertCache] = None, use_cache: bool = False,
                       probe_cols: Optional[Tuple[int, int]] = None) -> ExpertOutput:
        spec = self.expert_spec
        batch, length, width = x.shape
        if width != spec.width:
            raise ValueError(f"Expert inputs must have width {spec.width}")
        if len(prefix_kv) != spec.depth:
            raise ValueError("prefix_kv must provide one (key, value) pair per layer")
        if self.norm_mode == "segment":
            if seg_ids is None or seg_ids.shape != (length,):
                raise ValueError("Segment-mode expert needs seg_ids of shape [S]")
            seg_ids = seg_ids.to(x.device)
        else:
            if cond is None or cond.shape != (batch, spec.width):
                raise ValueError("adaRMS-mode expert needs cond of shape [B, width]")
        if positions.shape != (batch, length):
            raise ValueError("positions must be [B, S]")
        if layout_bias.ndim != 4 or layout_bias.shape[:3] != (batch, 1, length):
            raise ValueError("layout_bias must be [B, 1, S, K]")
        if cache is not None and len(cache.kv) != spec.depth:
            raise ValueError("Cache layer count does not match the expert")
        cos, sin = rope_cos_sin(positions, spec.head_dim, spec.rope_theta)
        new_kv, masses = [], []
        for index, layer in enumerate(self.expert.layers):
            cached = cache.kv[index] if cache is not None else None
            x, k, v, mass = self._expert_layer(layer, x, prefix_kv[index], cached, layout_bias, cos, sin,
                                               seg_ids, cond, probe_cols)
            if use_cache:
                if cached is not None:
                    k, v = torch.cat((cached[0], k), dim=2), torch.cat((cached[1], v), dim=2)
                new_kv.append((k, v))
            masses.append(mass)
        hidden, _ = self._norm(self.expert.norm, x, seg_ids, cond)
        new_cache = None
        if use_cache:
            new_cache = ExpertCache(tuple(new_kv), (cache.length if cache is not None else 0) + length)
        probe = torch.stack(masses) if probe_cols is not None else None
        return ExpertOutput(hidden, new_cache, probe)
