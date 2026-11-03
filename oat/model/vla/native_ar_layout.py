"""Observation/history/action masks for autoregression inside the VLM itself.

History occupies nine fixed positions, even when some or all history is missing.
The attention mask, rather than position compaction, controls history validity.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor

from oat.model.vla.layout import NEG, _bool_matrix


NATIVE_HISTORY_SLOTS = 9


@dataclass(frozen=True)
class NativeARLayout:
    positions: Tensor       # [B, Q], RoPE positions of the query rows
    bias: Tensor            # [B, 1, Q, K], additive attention mask
    valid: Tensor           # [B, K], includes cached keys for a decode view
    pos0: Tensor            # [B], number of valid observation tokens
    observation_len: int    # physical observation length, including padding
    n_history: int
    n_ar: int               # action inputs present among the keys
    query_start: int = 0    # physical offset of the first query in the full sequence

    @property
    def context_len(self) -> int:
        return self.observation_len + self.n_history

    @property
    def action_start(self) -> int:
        return self.context_len


def build_native_ar_layout(observation_valid: Tensor, history_valid: Tensor,
                           n_ar: int) -> NativeARLayout:
    """Build [observation | history9 | shifted action inputs].

    Observation rows read only valid observations. History rows read valid
    observations and all valid history. Action row i also reads action rows
    j <= i: its input is BOS (i=0) or the previous action code, never its target.
    An invalid query retains its diagonal to avoid empty softmax rows, but is
    never exposed as a key to another valid query.
    """
    observation_valid = _bool_matrix(observation_valid, "observation_valid")
    batch, observation_len = observation_valid.shape
    history_valid = _bool_matrix(history_valid, "history_valid", batch).to(observation_valid.device)
    if history_valid.shape[1] != NATIVE_HISTORY_SLOTS:
        raise ValueError(f"history_valid must have {NATIVE_HISTORY_SLOTS} columns")
    if not observation_valid.any(dim=1).all():
        raise ValueError("Every sample needs at least one valid observation token")
    if isinstance(n_ar, bool) or not isinstance(n_ar, int) or n_ar < 0:
        raise ValueError("n_ar must be a nonnegative integer")
    device = observation_valid.device
    context_len = observation_len + NATIVE_HISTORY_SLOTS
    total = context_len + n_ar
    valid = torch.cat((observation_valid, history_valid,
                       torch.ones(batch, n_ar, dtype=torch.bool, device=device)), dim=1)
    allowed = torch.zeros(batch, total, total, dtype=torch.bool, device=device)
    allowed[:, :observation_len, :observation_len] = observation_valid[:, None]
    allowed[:, observation_len:, :context_len] = valid[:, None, :context_len]
    if n_ar:
        allowed[:, context_len:, context_len:] = torch.ones(n_ar, n_ar, dtype=torch.bool,
                                                            device=device).tril()
    diagonal = torch.arange(total, device=device)
    allowed[:, diagonal, diagonal] = True
    bias = torch.zeros(batch, 1, total, total, dtype=torch.float32, device=device)
    bias.masked_fill_(~allowed[:, None], NEG)
    observation_positions = (observation_valid.long().cumsum(dim=1) - 1).clamp_min(0)
    pos0 = observation_valid.long().sum(dim=1)
    suffix_positions = pos0[:, None] + torch.arange(NATIVE_HISTORY_SLOTS + n_ar, device=device)[None]
    positions = torch.cat((observation_positions, suffix_positions), dim=1)
    return NativeARLayout(positions, bias, valid, pos0, observation_len, NATIVE_HISTORY_SLOTS, n_ar)


def native_prefill_view(layout: NativeARLayout, n_ar: int = 1) -> NativeARLayout:
    """Full context plus the first n_ar action inputs (normally one BOS)."""
    if layout.query_start != 0:
        raise ValueError("Prefill needs a complete layout, not a decode view")
    if isinstance(n_ar, bool) or not isinstance(n_ar, int) or not 0 <= n_ar <= layout.n_ar:
        raise ValueError("n_ar is outside the action block")
    stop = layout.context_len + n_ar
    return replace(layout, positions=layout.positions[:, :stop], bias=layout.bias[:, :, :stop, :stop],
                   valid=layout.valid[:, :stop], n_ar=n_ar)


def native_decode_view(layout: NativeARLayout, ar_index: int) -> NativeARLayout:
    """One action input; all physical rows before it must already be cached."""
    if layout.query_start != 0:
        raise ValueError("Decode slicing needs a complete layout")
    if (isinstance(ar_index, bool) or not isinstance(ar_index, int)
            or not 0 <= ar_index < layout.n_ar):
        raise ValueError("ar_index is outside the action block")
    row = layout.context_len + ar_index
    stop = row + 1
    return replace(layout, positions=layout.positions[:, row:stop], bias=layout.bias[:, :, row:stop, :stop],
                   valid=layout.valid[:, :stop], n_ar=ar_index + 1, query_start=row)
