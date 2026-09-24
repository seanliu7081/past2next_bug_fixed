"""Bidirectional DiT-X velocity network for the eight OAT scalar-code slots.

Only action queries receive adaptive modulation. Memory keys/values therefore
remain fixed during an Euler solve and can be cached in eval/no_grad mode.
"""
from dataclasses import dataclass
import math
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class HeadRMSNorm(nn.Module):
    """RMS normalization along the last feature axis, with FP32 reduction."""
    def __init__(self, width: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = float(eps)

    def forward(self, value: Tensor) -> Tensor:
        with torch.autocast(device_type=value.device.type, enabled=False):
            normalized = value.float() * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + self.eps)
            normalized = normalized * self.weight.float()
        return normalized.to(value.dtype)


def sinusoidal_embedding(time: Tensor, width: int = 256, max_period: float = 10000.0) -> Tensor:
    if width < 2 or width % 2:
        raise ValueError("Sinusoidal embedding width must be a positive even number.")
    with torch.autocast(device_type=time.device.type, enabled=False):
        frequencies = torch.exp(-math.log(max_period) * torch.arange(width // 2, device=time.device, dtype=torch.float32) / (width // 2))
        phase = time.float()[:, None] * frequencies[None]
        return torch.cat((torch.cos(phase), torch.sin(phase)), dim=-1)


class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim: int, sinusoidal_dim: int = 256, hidden_dim: int = 1024):
        super().__init__()
        self.sinusoidal_dim = sinusoidal_dim
        self.mlp = nn.Sequential(nn.Linear(sinusoidal_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, embed_dim))

    def forward(self, time: Tensor) -> Tensor:
        return self.mlp(sinusoidal_embedding(time, self.sinusoidal_dim))


class SwiGLU(nn.Module):
    def __init__(self, width: int, hidden_dim: int):
        super().__init__()
        self.gate = nn.Linear(width, hidden_dim, bias=False)
        self.up = nn.Linear(width, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, width, bias=False)

    def forward(self, value: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(value)) * self.up(value))


class QKNormAttention(nn.Module):
    def __init__(self, width: int, heads: int, *, cross: bool):
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.cross = cross
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.out = nn.Linear(width, width)
        self.q_norm = HeadRMSNorm(self.head_dim)
        self.k_norm = HeadRMSNorm(self.head_dim)

    def _heads(self, value: Tensor) -> Tensor:
        return value.reshape(value.shape[0], value.shape[1], self.heads, self.head_dim).transpose(1, 2)

    def project_kv(self, memory: Tensor) -> tuple[Tensor, Tensor]:
        return self.k_norm(self._heads(self.k(memory))), self._heads(self.v(memory))

    def forward(self, query: Tensor, memory: Optional[Tensor] = None, bias: Optional[Tensor] = None,
                kv: Optional[tuple[Tensor, Tensor]] = None) -> Tensor:
        q = self.q_norm(self._heads(self.q(query)))
        if kv is None:
            k, v = self.project_kv(query if memory is None else memory)
        else:
            k, v = kv
        if k.dtype != q.dtype or v.dtype != q.dtype:
            raise ValueError("Cached KV precision differs from the velocity network; use one autocast scope per solve.")
        attention = F.scaled_dot_product_attention(q, k, v, attn_mask=None if bias is None else bias.to(q.dtype),
                                                  dropout_p=0.0, is_causal=False, scale=self.head_dim ** -0.5)
        return self.out(attention.transpose(1, 2).reshape(query.shape))


class DiTXBlock(nn.Module):
    def __init__(self, width: int, heads: int, ffn_dim: int):
        super().__init__()
        self.norm_sa = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.norm_ca = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.norm_ff = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.self_attention = QKNormAttention(width, heads, cross=False)
        self.cross_attention = QKNormAttention(width, heads, cross=True)
        self.ffn = SwiGLU(width, ffn_dim)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 9 * width))

    @staticmethod
    def _modulate(value: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        return value * (1.0 + scale[:, None]) + shift[:, None]

    def forward(self, x: Tensor, global_condition: Tensor, memory: Tensor, bias: Tensor,
                kv: Optional[tuple[Tensor, Tensor]] = None) -> Tensor:
        shift_sa, scale_sa, gate_sa, shift_ca, scale_ca, gate_ca, shift_ff, scale_ff, gate_ff = self.modulation(global_condition).chunk(9, dim=-1)
        x = x + gate_sa[:, None] * self.self_attention(self._modulate(self.norm_sa(x), shift_sa, scale_sa))
        x = x + gate_ca[:, None] * self.cross_attention(self._modulate(self.norm_ca(x), shift_ca, scale_ca), memory, bias, kv)
        x = x + gate_ff[:, None] * self.ffn(self._modulate(self.norm_ff(x), shift_ff, scale_ff))
        return x


@dataclass(frozen=True)
class StaticCrossKV:
    """An ephemeral cache owned by one model and one immutable chunk context."""
    model_identity: int
    context_identity: int
    memory_identity: int
    layers: tuple[tuple[Tensor, Tensor], ...]
    bias: Tensor


class DiTXLatentFlow(nn.Module):
    def __init__(self, *, current_state_dim: int, code_dim: int = 5, num_slots: int = 8,
                 embed_dim: int = 768, n_layers: int = 16, n_heads: int = 12, ffn_dim: int = 2048,
                 dropout: float = 0.0, activation_checkpointing: bool = True,
                 time_embed_dim: int = 256, time_hidden_dim: int = 1024,
                 initialization: str = "xavier_uniform_slot_normal_0.02"):
        super().__init__()
        if min(current_state_dim, code_dim, num_slots, embed_dim, n_layers, n_heads, ffn_dim) < 1:
            raise ValueError("All DiT-X dimensions must be positive.")
        if embed_dim % n_heads:
            raise ValueError("embed_dim must be divisible by n_heads.")
        if dropout != 0.0:
            raise ValueError("This joint FM/CT recipe requires dropout=0.0.")
        if initialization != "xavier_uniform_slot_normal_0.02":
            raise ValueError("Unsupported DiT-X initialization.")
        self.current_state_dim = int(current_state_dim)
        self.code_dim, self.num_slots, self.embed_dim = int(code_dim), int(num_slots), int(embed_dim)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.initialization = initialization
        self.input_projection = nn.Linear(code_dim, embed_dim)
        self.slot_embedding = nn.Parameter(torch.empty(1, num_slots, embed_dim))
        self.time_embedding = TimeEmbedding(embed_dim, time_embed_dim, time_hidden_dim)
        self.step_embedding = TimeEmbedding(embed_dim, time_embed_dim, time_hidden_dim)
        self.current_state_embedding = nn.Sequential(nn.Linear(current_state_dim, time_hidden_dim), nn.SiLU(), nn.Linear(time_hidden_dim, embed_dim))
        self.global_projection = nn.Linear(3 * embed_dim, embed_dim)
        self.blocks = nn.ModuleList(DiTXBlock(embed_dim, n_heads, ffn_dim) for _ in range(n_layers))
        self.output = nn.Sequential(HeadRMSNorm(embed_dim), nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, code_dim))
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.slot_embedding, std=0.02)
        for block in self.blocks:
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def _context(self, context) -> tuple[Tensor, Tensor]:
        memory = context.memory
        if memory.ndim != 3 or memory.shape[-1] != self.embed_dim:
            raise ValueError(f"Context must have shape [B, tokens, {self.embed_dim}].")
        bias = context.attention_bias(dtype=torch.float32)
        if bias.shape != (memory.shape[0], 1, 1, memory.shape[1]):
            raise ValueError("Context attention bias must have shape [B, 1, 1, tokens].")
        if torch.isnan(bias).any() or torch.isposinf(bias).any():
            raise ValueError("Context attention bias cannot contain NaN or +Inf.")
        visible = ~torch.isneginf(bias[:, 0, 0])
        if not visible.any(-1).all():
            raise ValueError("Every row needs at least one readable context token.")
        # Sanitize before projections, so NaN/Inf padding and closed summaries
        # cannot enter a matmul, its backward, or cached K/V.
        memory = torch.where(visible[..., None], memory, torch.zeros_like(memory))
        if not torch.isfinite(memory).all():
            raise ValueError("Visible context memory contains NaN or Inf.")
        return memory, bias

    def build_kv_cache(self, context) -> StaticCrossKV:
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("Static cross-KV is available only in eval mode with gradients disabled.")
        memory, bias = self._context(context)
        return StaticCrossKV(id(self), id(context), id(context.memory),
                             tuple(block.cross_attention.project_kv(memory) for block in self.blocks), bias)

    def forward(self, z_t: Tensor, *, time: Tensor, step_size: Tensor, context,
                current_state: Tensor, kv_cache: Optional[StaticCrossKV] = None) -> Tensor:
        if z_t.ndim != 3 or z_t.shape[1:] != (self.num_slots, self.code_dim):
            raise ValueError(f"Expected noisy latents [B, {self.num_slots}, {self.code_dim}].")
        batch = z_t.shape[0]
        if current_state.shape != (batch, self.current_state_dim):
            raise ValueError(f"current_state must be [B, {self.current_state_dim}], excluding any history summaries.")
        for name, value in (("time", time), ("step_size", step_size)):
            if value.shape != (batch,) or not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError(f"{name} must be a finite floating point [B] tensor.")
            if (value < 0).any() or (value > 1).any():
                raise ValueError(f"{name} must be in [0, 1].")
        if not torch.isfinite(z_t).all() or not torch.isfinite(current_state).all():
            raise ValueError("Latents and current state must be finite.")
        if kv_cache is None:
            memory, bias = self._context(context)
        else:
            if self.training or torch.is_grad_enabled():
                raise RuntimeError("Do not use a cross-KV cache during gradient-enabled or training forwards.")
            if (kv_cache.model_identity, kv_cache.context_identity, kv_cache.memory_identity) != (id(self), id(context), id(context.memory)):
                raise ValueError("Cross-KV cache belongs to another model or context chunk.")
            memory, bias = context.memory, kv_cache.bias
        if memory.shape[0] != batch:
            raise ValueError("Latent/context batch sizes differ.")
        global_condition = self.global_projection(torch.cat((self.time_embedding(time), self.step_embedding(step_size),
                                                               self.current_state_embedding(current_state.float())), dim=-1))
        x = self.input_projection(z_t) + self.slot_embedding
        for index, block in enumerate(self.blocks):
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                # Explicit tensor arguments bind each recomputation to this
                # forward's memory and bias, rather than a mutable cache.
                x = checkpoint(block, x, global_condition, memory, bias, use_reentrant=False)
            else:
                x = block(x, global_condition, memory, bias, None if kv_cache is None else kv_cache.layers[index])
        return self.output(x)
