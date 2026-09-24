"""Typed, explicit conditioning layout shared by the new Past2Next variants.

A ``True`` validity entry always means a readable token.  Additive SDPA masks
are built here so padding and the summary gate have exactly the same semantics
in teacher forcing, cached inference, and self-past generation.
"""
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple

import torch
from torch import Tensor


class Segment(IntEnum):
    VISUAL = 0
    PROPRIO = 1
    RAW_ACTION = 2
    ACTION_DIFF = 3
    HISTORY_SUMMARY = 4


ContextSegment = Segment
CONTEXT_SCHEMA_VERSION = 1


@dataclass
class ContextBatch:
    memory: Tensor
    valid_mask: Tensor
    segment_ids: Tensor
    observation_summary: Optional[Tensor] = None
    history_summary_pool: Optional[Tensor] = None
    history_valid_fraction: Optional[Tensor] = None
    history_log_gate: Optional[Tensor] = None

    def validate(self) -> "ContextBatch":
        if self.memory.ndim != 3 or not self.memory.is_floating_point():
            raise ValueError("Context memory must be floating point [B, N, D].")
        batch, count, width = self.memory.shape
        if min(batch, count, width) < 1:
            raise ValueError("Context memory dimensions must be nonempty.")
        if self.valid_mask.shape != (batch, count) or self.valid_mask.dtype != torch.bool:
            raise ValueError("Context valid_mask must be bool [B, N] (True means visible).")
        if self.segment_ids.shape != (count,) or self.segment_ids.dtype != torch.long:
            raise ValueError("Context segment_ids must be int64 [N].")
        if self.valid_mask.device != self.memory.device or self.segment_ids.device != self.memory.device:
            raise ValueError("Context tensors must use the same device.")
        if ((self.segment_ids < 0) | (self.segment_ids >= len(Segment))).any():
            raise ValueError("Context contains an unknown segment ID.")
        observation = self.segment_mask(Segment.VISUAL) | self.segment_mask(Segment.PROPRIO)
        if not (self.valid_mask & observation[None]).any(dim=1).all():
            raise ValueError("Every sample needs at least one visible observation token.")
        for name, shape in (
            ("observation_summary", (batch, width)),
            ("history_summary_pool", (batch, width)),
            ("history_valid_fraction", (batch, 1)),
            ("history_log_gate", (batch, 1)),
        ):
            value = getattr(self, name)
            if value is not None:
                if value.shape != shape or value.device != self.memory.device or not value.is_floating_point():
                    raise ValueError(f"Context {name} must be floating point {shape} on the memory device.")
                if not self.segment_mask(Segment.HISTORY_SUMMARY).any():
                    raise ValueError(f"Context {name} requires HISTORY_SUMMARY tokens.")
        if self.history_log_gate is not None:
            if torch.isnan(self.history_log_gate).any() or (self.history_log_gate > 0).any():
                raise ValueError("history_log_gate must be <= 0; -inf denotes a closed gate.")
        if self.history_valid_fraction is not None:
            fraction = self.history_valid_fraction
            if not torch.isfinite(fraction).all() or ((fraction < 0) | (fraction > 1)).any():
                raise ValueError("history_valid_fraction must be finite and in [0, 1].")
        return self

    def validate_variant(self, variant: str, num_summary_tokens: Optional[int] = None) -> "ContextBatch":
        """Check the variant contract without relying on token positions."""
        self.validate()
        count = int(self.segment_mask(Segment.HISTORY_SUMMARY).sum())
        metadata = (self.observation_summary, self.history_summary_pool,
                    self.history_valid_fraction, self.history_log_gate)
        if variant == "p2n_new":
            if count or any(value is not None for value in metadata):
                raise ValueError("p2n_new must not contain summary tokens or gate metadata.")
        elif variant == "p2n_state_gate_new":
            if count == 0 or any(value is None for value in metadata):
                raise ValueError("p2n_state_gate_new requires history summaries and all gate metadata.")
        else:
            raise ValueError(f"Unknown new Past2Next variant: {variant!r}.")
        if num_summary_tokens is not None and count != num_summary_tokens:
            raise ValueError(f"Expected {num_summary_tokens} history summaries, received {count}.")
        return self

    def segment_mask(self, segment: Segment) -> Tensor:
        return self.segment_ids == int(segment)

    def segment_memory(self, segment: Segment) -> Tuple[Tensor, Tensor]:
        selected = self.segment_mask(segment)
        return self.memory[:, selected], self.valid_mask[:, selected]

    def attention_bias(self, dtype: Optional[torch.dtype] = None) -> Tensor:
        """Return additive cross-attention bias ``[B, 1, 1, N]``.

        ``where`` avoids the undefined ``-inf * 0`` in closed-gate mode and
        preserves the learned log-gate's gradient on summary positions only.
        """
        dtype = self.memory.dtype if dtype is None else dtype
        bias = torch.zeros(self.valid_mask.shape, device=self.memory.device, dtype=dtype)
        if self.history_log_gate is not None:
            bias = torch.where(
                self.segment_mask(Segment.HISTORY_SUMMARY)[None],
                self.history_log_gate.to(dtype=dtype), bias,
            )
        bias = bias.masked_fill(~self.valid_mask, float("-inf"))
        return bias[:, None, None, :]

    def sanitized_memory(self) -> Tensor:
        """Prevent masked padding (including NaNs) from entering K/V matmuls."""
        visible = self.valid_mask
        if self.history_log_gate is not None:
            closed_summary = self.segment_mask(Segment.HISTORY_SUMMARY)[None] & torch.isneginf(self.history_log_gate)
            visible = visible & ~closed_summary
        return torch.where(visible[..., None], self.memory, torch.zeros_like(self.memory))


def build_attention_bias(context: ContextBatch, dtype: Optional[torch.dtype] = None) -> Tensor:
    return context.attention_bias(dtype=dtype)
