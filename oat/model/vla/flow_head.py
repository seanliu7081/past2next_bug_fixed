"""PI0.5 flow-matching head on the joint stack (parity reference and the pi05_ki_flow baseline)."""
from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from oat.model.vla.layout import build_flow_suffix_layout

FLOW_ACTION_DIM = 32
MIN_PERIOD, MAX_PERIOD = 4e-3, 4.0


def posemb_sincos(t: Tensor, dim: int, min_period: float = MIN_PERIOD, max_period: float = MAX_PERIOD) -> Tensor:
    """PI0 sine-cosine time embedding (sin first), computed in float64 then cast to fp32."""
    if dim % 2:
        raise ValueError("Time-embedding width must be even")
    if t.ndim != 1:
        raise ValueError("t must be a 1-D tensor")
    compute = torch.float64 if t.device.type != "mps" else torch.float32
    fraction = torch.linspace(0.0, 1.0, dim // 2, dtype=compute, device=t.device)
    period = min_period * (max_period / min_period) ** fraction
    angle = t.to(compute)[:, None] * (2.0 * math.pi / period)[None, :]
    return torch.cat((torch.sin(angle), torch.cos(angle)), dim=-1).to(torch.float32)


class FlowHeads(nn.Module):
    """``action_in_proj``, ``action_out_proj`` and the adaRMS time MLP of PI0.5 (fp32)."""

    def __init__(self, width: int, action_dim: int = FLOW_ACTION_DIM):
        super().__init__()
        self.width, self.action_dim = width, action_dim
        self.action_in_proj = nn.Linear(action_dim, width)
        self.action_out_proj = nn.Linear(width, action_dim)
        self.time_mlp_in = nn.Linear(width, width)
        self.time_mlp_out = nn.Linear(width, width)


def flow_condition(heads: FlowHeads, t: Tensor) -> Tensor:
    embedding = posemb_sincos(t, heads.width)
    with torch.autocast(device_type=t.device.type, enabled=False):
        hidden = F.silu(F.linear(embedding, heads.time_mlp_in.weight.float(), heads.time_mlp_in.bias.float()))
        return F.silu(F.linear(hidden, heads.time_mlp_out.weight.float(), heads.time_mlp_out.bias.float()))


def flow_velocity(joint, heads: FlowHeads, prefix_kv: Sequence, pos0: Tensor, prefix_valid: Tensor,
                  x_t: Tensor, t: Tensor) -> Tensor:
    """Velocity ``v(x_t, t)`` for noisy action chunks ``x_t`` [B, H, 32] over a cached prefix."""
    if x_t.ndim != 3 or x_t.shape[-1] != heads.action_dim:
        raise ValueError(f"x_t must be [B, H, {heads.action_dim}]")
    layout = build_flow_suffix_layout(pos0, prefix_valid, x_t.shape[1])
    tokens = heads.action_in_proj(x_t.to(heads.action_in_proj.weight.dtype))
    out = joint.expert_forward(tokens, prefix_kv, layout.bias, layout.positions, cond=flow_condition(heads, t))
    return heads.action_out_proj(out.hidden)


@torch.no_grad()
def sample_actions(joint, heads: FlowHeads, prefix_kv: Sequence, pos0: Tensor, prefix_valid: Tensor,
                   noise: Tensor, num_steps: int = 10) -> Tensor:
    """Euler integration from t=1 (noise) to t=0, exactly like openpi ``Pi0.sample_actions``."""
    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 1:
        raise ValueError("num_steps must be a positive integer")
    dt = -1.0 / num_steps
    x_t, time = noise, 1.0
    while time >= -dt / 2:
        t = torch.full((noise.shape[0],), time, dtype=torch.float32, device=noise.device)
        velocity = flow_velocity(joint, heads, prefix_kv, pos0, prefix_valid, x_t, t)
        x_t = x_t + dt * velocity.to(x_t.dtype)
        time += dt
    return x_t
