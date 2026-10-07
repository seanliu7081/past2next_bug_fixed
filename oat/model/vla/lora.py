"""Minimal LoRA for the frozen Gemma-2B projections (kept unmerged)."""
from __future__ import annotations

import math
from typing import Iterable, List

import torch
from torch import Tensor, nn
import torch.nn.functional as F

DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    """``base(x) + B(A(x)) * alpha / rank`` with a frozen base and fp32 adapters (B starts at zero)."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear wraps an nn.Linear")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ValueError("LoRA rank must be a positive integer")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        self.base = base.requires_grad_(False)
        self.rank, self.alpha = rank, float(alpha)
        self.scaling = self.alpha / rank
        device = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: Tensor) -> Tensor:
        lora_a, lora_b = self.lora_A, self.lora_B
        if lora_a.dtype != x.dtype and not torch.is_autocast_enabled(x.device.type):
            lora_a, lora_b = lora_a.to(x.dtype), lora_b.to(x.dtype)
        return self.base(x) + F.linear(F.linear(x, lora_a), lora_b) * self.scaling

    @torch.no_grad()
    def merged_weight(self) -> Tensor:
        """Base plus adapter in fp32 (merging into a bf16 base would round the delta away)."""
        return self.base.weight.float() + (self.lora_B.float() @ self.lora_A.float()) * self.scaling


def inject_lora(module: nn.Module, rank: int, alpha: float, targets: Iterable[str] = DEFAULT_TARGETS) -> List[str]:
    """Replace every nn.Linear child named in ``targets`` under ``module``; return replaced paths."""
    targets = tuple(targets)
    replaced = []
    for parent_name, parent in list(module.named_modules()):
        for child_name, child in list(parent.named_children()):
            if child_name in targets and isinstance(child, nn.Linear):
                setattr(parent, child_name, LoRALinear(child, rank, alpha))
                replaced.append(f"{parent_name}.{child_name}" if parent_name else child_name)
    if not replaced:
        raise ValueError("inject_lora found no target linear layers")
    return replaced


def lora_parameters(module: nn.Module) -> List[nn.Parameter]:
    return [p for m in module.modules() if isinstance(m, LoRALinear) for p in (m.lora_A, m.lora_B)]
