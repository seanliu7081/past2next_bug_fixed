"""LN/GELU DiT-X velocity field over normalized continuous actions.

AdaLN sees only time and relative step size. Observation and history features
remain in cross-attention memory, whose K/V are static throughout one solve.
No tokenizer, quantizer, action normalizer, or execution state is owned here.
"""

from dataclasses import dataclass
import math
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from oat.model.common.action_flow_context import ACTION_FLOW_CONTEXT_LAYOUTS, ActionFlowContextBatch


class FP32LayerNorm(nn.LayerNorm):
    """LayerNorm with FP32 accumulation, retaining the input activation dtype."""

    def forward(self, value: Tensor) -> Tensor:
        with torch.autocast(device_type=value.device.type, enabled=False):
            output = F.layer_norm(
                value.float(), self.normalized_shape,
                None if self.weight is None else self.weight.float(),
                None if self.bias is None else self.bias.float(), self.eps,
            )
        return output.to(value.dtype)


class ActionRMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = float(eps)

    def forward(self, value: Tensor) -> Tensor:
        with torch.autocast(device_type=value.device.type, enabled=False):
            normalized = value.float() * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + self.eps)
            normalized = normalized * self.weight.float()
        return normalized.to(value.dtype)


def sinusoidal_embedding(time: Tensor, width: int = 128, max_period: float = 10000.0) -> Tensor:
    if width < 2 or width % 2:
        raise ValueError("Sinusoidal embedding width must be a positive even number.")
    with torch.autocast(device_type=time.device.type, enabled=False):
        frequencies = torch.exp(-math.log(max_period) * torch.arange(width // 2, device=time.device, dtype=torch.float32) / (width // 2))
        phase = time.float()[:, None] * frequencies[None]
        return torch.cat((torch.cos(phase), torch.sin(phase)), dim=-1)


class ActionTimeEmbedding(nn.Module):
    def __init__(self, embed_dim: int, sinusoidal_dim: int = 128, hidden_dim: int = 512):
        super().__init__()
        self.sinusoidal_dim = sinusoidal_dim
        self.mlp = nn.Sequential(nn.Linear(sinusoidal_dim, hidden_dim), nn.Mish(), nn.Linear(hidden_dim, embed_dim))

    def forward(self, time: Tensor) -> Tensor:
        return self.mlp(sinusoidal_embedding(time, self.sinusoidal_dim))


class ActionAttention(nn.Module):
    """Bidirectional self-attention or cross-attention with per-head Q/K LN."""

    def __init__(self, width: int, heads: int, *, cross: bool):
        super().__init__()
        self.heads, self.head_dim, self.cross = heads, width // heads, cross
        self.q = nn.Linear(width, width, bias=True)
        self.k = nn.Linear(width, width, bias=True)
        self.v = nn.Linear(width, width, bias=True)
        self.out = nn.Linear(width, width, bias=True)
        # Self-attention intentionally has no QK normalization parameters.
        self.q_norm = FP32LayerNorm(self.head_dim, eps=1e-6) if cross else nn.Identity()
        self.k_norm = FP32LayerNorm(self.head_dim, eps=1e-6) if cross else nn.Identity()

    def _heads(self, value: Tensor) -> Tensor:
        return value.reshape(value.shape[0], value.shape[1], self.heads, self.head_dim).transpose(1, 2)

    def project_kv(self, memory: Tensor) -> tuple[Tensor, Tensor]:
        return self.k_norm(self._heads(self.k(memory))), self._heads(self.v(memory))

    def forward(self, query: Tensor, memory: Optional[Tensor] = None, bias: Optional[Tensor] = None,
                kv: Optional[tuple[Tensor, Tensor]] = None) -> Tensor:
        q = self.q_norm(self._heads(self.q(query)))
        k, v = self.project_kv(query if memory is None else memory) if kv is None else kv
        if k.dtype != q.dtype or v.dtype != q.dtype:
            raise ValueError("Cached KV precision differs; use one autocast scope for the entire solve.")
        output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None if bias is None else bias.to(q.dtype),
            dropout_p=0.0, is_causal=False, scale=self.head_dim ** -0.5,
        )
        return self.out(output.transpose(1, 2).reshape(query.shape))


class DiTXActionBlock(nn.Module):
    def __init__(self, width: int, heads: int, ffn_hidden_dim: int):
        super().__init__()
        self.norm_sa = FP32LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.norm_ca = FP32LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.norm_ff = FP32LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.self_attention = ActionAttention(width, heads, cross=False)
        self.cross_attention = ActionAttention(width, heads, cross=True)
        self.ffn = nn.Sequential(nn.Linear(width, ffn_hidden_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_hidden_dim, width))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 9 * width))

    @staticmethod
    def _modulate(value: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        return value * (1.0 + scale[:, None]) + shift[:, None]

    def forward(self, x: Tensor, global_condition: Tensor, memory: Tensor, bias: Tensor,
                kv: Optional[tuple[Tensor, Tensor]] = None) -> Tensor:
        shift_sa, scale_sa, gate_sa, shift_ca, scale_ca, gate_ca, shift_ff, scale_ff, gate_ff = self.modulation(global_condition).chunk(9, dim=-1)
        x = x + gate_sa[:, None] * self.self_attention(self._modulate(self.norm_sa(x), shift_sa, scale_sa))
        x = x + gate_ca[:, None] * self.cross_attention(self._modulate(self.norm_ca(x), shift_ca, scale_ca), memory, bias, kv)
        return x + gate_ff[:, None] * self.ffn(self._modulate(self.norm_ff(x), shift_ff, scale_ff))


def _tensor_stamp(tensor):
    if tensor is None:
        return None
    # Inference tensors do not expose a version counter.
    return (id(tensor), None if torch.is_inference(tensor) else tensor._version)


def _context_stamp(context):
    return tuple(_tensor_stamp(getattr(context, name)) for name in ("memory", "valid_mask", "segment_ids", "history_log_gate"))


@dataclass(frozen=True)
class ActionStaticCrossKV:
    """Ephemeral cache for one immutable context and one eval-mode model."""

    model_identity: int
    context_identity: int
    context_stamp: tuple
    layers: tuple[tuple[Tensor, Tensor], ...]
    bias: Tensor


class DiTXActionVectorField(nn.Module):
    def __init__(self, *, action_dim: int = 7, horizon: int = 16, embed_dim: int = 768,
                 n_layers: int = 16, n_heads: int = 12, ffn_hidden_dim: int = 3072,
                 time_embed_dim: int = 128, time_hidden_dim: int = 512, dropout: float = 0.0,
                 activation_checkpointing: bool = True, variant: str = "p2n_action_flow",
                 initialization: str = "xavier_uniform_position_normal_0.02"):
        super().__init__()
        dimensions = (action_dim, horizon, embed_dim, n_layers, n_heads, ffn_hidden_dim, time_embed_dim, time_hidden_dim)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in dimensions):
            raise ValueError("All action DiT-X dimensions must be positive integers.")
        if embed_dim % n_heads or time_embed_dim % 2:
            raise ValueError("embed_dim must be divisible by n_heads and time_embed_dim must be even.")
        if dropout != 0.0:
            raise ValueError("This shared-condition FM/CT recipe requires dropout=0.0.")
        if initialization != "xavier_uniform_position_normal_0.02":
            raise ValueError("Unsupported action DiT-X initialization.")
        if variant not in ACTION_FLOW_CONTEXT_LAYOUTS:
            raise ValueError(f"Unknown continuous_action_flow variant: {variant!r}.")
        self.action_dim, self.horizon, self.embed_dim = action_dim, horizon, embed_dim
        self.variant, self.initialization = variant, initialization
        self.activation_checkpointing = bool(activation_checkpointing)
        self.input_projection = nn.Linear(action_dim, embed_dim)
        self.position_embedding = nn.Parameter(torch.empty(1, horizon, embed_dim))
        self.time_embedding = ActionTimeEmbedding(embed_dim, time_embed_dim, time_hidden_dim)
        self.step_embedding = ActionTimeEmbedding(embed_dim, time_embed_dim, time_hidden_dim)
        self.global_projection = nn.Linear(2 * embed_dim, embed_dim)
        self.blocks = nn.ModuleList(DiTXActionBlock(embed_dim, n_heads, ffn_hidden_dim) for _ in range(n_layers))
        self.output = nn.Sequential(ActionRMSNorm(embed_dim), nn.Linear(embed_dim, embed_dim), nn.GELU(approximate="tanh"), nn.Linear(embed_dim, action_dim))
        self._initialize()

    def _initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.position_embedding, std=0.02)
        for block in self.blocks:
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def _context(self, context: ActionFlowContextBatch):
        if not isinstance(context, ActionFlowContextBatch):
            raise TypeError("Direct-action DiT-X requires an ActionFlowContextBatch.")
        context.validate_variant(self.variant)
        memory = context.memory
        if memory.shape[-1] != self.embed_dim:
            raise ValueError(f"Context width must be {self.embed_dim}.")
        bias = context.attention_bias(dtype=torch.float32)
        if torch.isnan(bias).any() or torch.isposinf(bias).any():
            raise ValueError("Context attention bias cannot contain NaN or +Inf.")
        visible = ~torch.isneginf(bias[:, 0, 0])
        if not visible.any(-1).all():
            raise ValueError("Every row needs at least one readable context token.")
        # Clean masked/closed contents before K/V projection and its backward.
        memory = torch.where(visible[..., None], memory, torch.zeros_like(memory))
        if not torch.isfinite(memory).all():
            raise ValueError("Visible context memory contains NaN or Inf.")
        return memory, bias

    def build_kv_cache(self, context: ActionFlowContextBatch) -> ActionStaticCrossKV:
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("Static cross-KV requires eval mode with gradients disabled.")
        memory, bias = self._context(context)
        return ActionStaticCrossKV(id(self), id(context), _context_stamp(context),
                                   tuple(block.cross_attention.project_kv(memory) for block in self.blocks), bias)

    def forward(self, noisy_actions: Tensor, *, time: Tensor, step_size: Tensor,
                context: ActionFlowContextBatch, kv_cache: Optional[ActionStaticCrossKV] = None) -> Tensor:
        if noisy_actions.ndim != 3 or noisy_actions.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError(f"Expected noisy actions [B,{self.horizon},{self.action_dim}].")
        batch = noisy_actions.shape[0]
        if not noisy_actions.is_floating_point() or not torch.isfinite(noisy_actions).all():
            raise ValueError("Noisy actions must contain finite floating point values.")
        for name, value in (("time", time), ("step_size", step_size)):
            if value.shape != (batch,) or value.dtype != torch.float32 or value.device != noisy_actions.device or not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite FP32 [B] on the action device.")
            if ((value < 0) | (value > 1)).any():
                raise ValueError(f"{name} must be in [0,1].")
        if kv_cache is None:
            memory, bias = self._context(context)
        else:
            if self.training or torch.is_grad_enabled():
                raise RuntimeError("Do not use cross-KV caches during training or gradient-enabled forwards.")
            if (kv_cache.model_identity, kv_cache.context_identity, kv_cache.context_stamp) != (id(self), id(context), _context_stamp(context)):
                raise ValueError("Cross-KV cache belongs to another model or context chunk, or the context was mutated.")
            if len(kv_cache.layers) != len(self.blocks):
                raise ValueError("Cross-KV cache does not match the model depth.")
            memory, bias = context.memory, kv_cache.bias
        if memory.shape[0] != batch or memory.device != noisy_actions.device:
            raise ValueError("Action and context batch sizes/devices differ.")
        global_condition = self.global_projection(torch.cat((self.time_embedding(time), self.step_embedding(step_size)), dim=-1))
        x = self.input_projection(noisy_actions) + self.position_embedding
        for index, block in enumerate(self.blocks):
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                x = checkpoint(block, x, global_condition, memory, bias, use_reentrant=False)
            else:
                x = block(x, global_condition, memory, bias, None if kv_cache is None else kv_cache.layers[index])
        return self.output(x)


DiTXActionFlow = DiTXActionVectorField
