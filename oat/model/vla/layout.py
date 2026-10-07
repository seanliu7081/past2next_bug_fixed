"""Attention masks and RoPE positions for the P2N-VLA prefix (VLM) and suffix (expert).

All masks are additive fp32 biases with value 0 (visible) or ``NEG`` (hidden).
Every query row keeps its own diagonal key, so no softmax row is ever empty.
Positions never depend on history validity or on the variant, which keeps
teacher forcing, cached decoding and self-past generation numerically aligned.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor

NEG = -2.3819763e38

SEG_RAWDIFF, SEG_HIST, SEG_AR = 0, 1, 2
N_SEGMENTS = 3
N_RAWDIFF_SLOTS = 9          # 7 past commands + first and second command differences
HIST_POSITION_OFFSET = 9     # every HIST summary shares pos0 + 9 (an unordered set)
AR_POSITION_OFFSET = 10      # AR input k (BOS, z0, ...) sits at pos0 + 10 + k


def _bool_matrix(value: Tensor, name: str, batch: Optional[int] = None) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != 2:
        raise ValueError(f"{name} must be a [B, N] tensor")
    if value.dtype != torch.bool:
        if value.is_floating_point() and not ((value == 0) | (value == 1)).all():
            raise ValueError(f"{name} must contain only 0/1 flags")
        value = value.to(torch.bool)
    if batch is not None and value.shape[0] != batch:
        raise ValueError(f"{name} batch size {value.shape[0]} != {batch}")
    return value


@dataclass(frozen=True)
class PrefixLayout:
    positions: Tensor     # [B, P] long
    bias: Tensor          # [B, 1, P, P] fp32
    valid: Tensor         # [B, P] bool (KI rows are always valid)
    pos0: Tensor          # [B] long: number of valid non-KI tokens
    nonki_len: int
    ki_len: int

    @property
    def nonki_valid(self) -> Tensor:
        return self.valid[:, :self.nonki_len]


def build_prefix_layout(img_valid: Tensor, prompt_valid: Tensor, ki_len: int = 0) -> PrefixLayout:
    """Physical order is [images | right-padded prompt | KI block].

    Non-KI rows see every valid non-KI key plus themselves (pad rows therefore
    still read valid keys), never KI keys. KI rows see every valid non-KI key and
    earlier KI keys. Non-KI positions follow ``cumsum(valid) - 1`` so pads do not
    advance RoPE; KI positions continue from ``pos0``.
    """
    img_valid = _bool_matrix(img_valid, "img_valid")
    batch = img_valid.shape[0]
    prompt_valid = _bool_matrix(prompt_valid, "prompt_valid", batch)
    if isinstance(ki_len, bool) or not isinstance(ki_len, int) or ki_len < 0:
        raise ValueError("ki_len must be a nonnegative integer")
    device = img_valid.device
    nonki_valid = torch.cat((img_valid, prompt_valid.to(device)), dim=1)
    if not nonki_valid.any(dim=1).all():
        raise ValueError("Every sample needs at least one valid prefix token")
    nonki_len = nonki_valid.shape[1]
    total = nonki_len + ki_len

    allowed = torch.zeros(batch, total, total, dtype=torch.bool, device=device)
    allowed[:, :nonki_len, :nonki_len] = nonki_valid[:, None, :]
    if ki_len:
        allowed[:, nonki_len:, :nonki_len] = nonki_valid[:, None, :]
        allowed[:, nonki_len:, nonki_len:] = torch.ones(ki_len, ki_len, dtype=torch.bool,
                                                        device=device).tril()
    diagonal = torch.arange(total, device=device)
    allowed[:, diagonal, diagonal] = True
    bias = torch.zeros(batch, 1, total, total, dtype=torch.float32, device=device)
    bias.masked_fill_(~allowed[:, None], NEG)

    nonki_positions = (nonki_valid.long().cumsum(dim=1) - 1).clamp_min(0)
    pos0 = nonki_valid.long().sum(dim=1)
    ki_positions = pos0[:, None] + torch.arange(ki_len, device=device)[None]
    positions = torch.cat((nonki_positions, ki_positions), dim=1)
    valid = torch.cat((nonki_valid, torch.ones(batch, ki_len, dtype=torch.bool, device=device)), dim=1)
    return PrefixLayout(positions, bias, valid, pos0, nonki_len, ki_len)


@dataclass(frozen=True)
class SuffixLayout:
    positions: Tensor     # [B, S] long
    bias: Tensor          # [B, 1, S, nonki_len + S] fp32
    seg_ids: Tensor       # [S] long
    n_rawdiff: int
    n_hist: int
    n_ar: int
    nonki_len: int

    @property
    def n_cond(self) -> int:
        return self.n_rawdiff + self.n_hist

    @property
    def hist_columns(self) -> Tuple[int, int]:
        """Key-column range [a, b) of the HIST tokens inside ``bias``."""
        start = self.nonki_len + self.n_rawdiff
        return start, start + self.n_hist


def _validated_pos0(pos0: Tensor, batch: int) -> Tensor:
    if not isinstance(pos0, Tensor) or pos0.shape != (batch,) or pos0.dtype != torch.long:
        raise ValueError("pos0 must be a long tensor of shape [B]")
    return pos0


def build_suffix_layout(pos0: Tensor, prefix_valid: Tensor, rawdiff_valid: Optional[Tensor],
                        n_hist: int, n_ar: int, log_gate: Optional[Tensor] = None,
                        hist_closed: bool = False) -> SuffixLayout:
    """Suffix order is [RAWDIFF (0 or 9) | HIST (0 or n_hist) | AR (n_ar)].

    RAWDIFF and HIST rows see only themselves (static condition memory). AR row i
    sees valid prefix keys, valid RAWDIFF keys, HIST keys through the per-sample
    ``log_gate`` bias (0 when None, hidden when closed) and AR keys j <= i.
    """
    prefix_valid = _bool_matrix(prefix_valid, "prefix_valid")
    batch, nonki_len = prefix_valid.shape
    pos0 = _validated_pos0(pos0, batch)
    device = prefix_valid.device
    for name, value in (("n_hist", n_hist), ("n_ar", n_ar)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if rawdiff_valid is None:
        n_rawdiff = 0
    else:
        rawdiff_valid = _bool_matrix(rawdiff_valid, "rawdiff_valid", batch).to(device)
        n_rawdiff = rawdiff_valid.shape[1]
        if n_rawdiff != N_RAWDIFF_SLOTS:
            raise ValueError(f"rawdiff_valid must have {N_RAWDIFF_SLOTS} columns")
    if n_ar > AR_POSITION_OFFSET + 64:
        raise ValueError("Unexpectedly long AR suffix")
    if log_gate is not None:
        if hist_closed:
            raise ValueError("A closed gate cannot also carry a learned log-gate")
        if (not isinstance(log_gate, Tensor) or not log_gate.is_floating_point()
                or log_gate.shape != (batch, 1)):
            raise ValueError("log_gate must be a floating tensor of shape [B, 1]")
        if not torch.isfinite(log_gate).all() or (log_gate > 0).any():
            raise ValueError("log_gate must be finite and <= 0")
    total = n_rawdiff + n_hist + n_ar

    arange = lambda n: torch.arange(n, device=device)  # noqa: E731
    positions = torch.cat((
        pos0[:, None] + arange(n_rawdiff)[None],
        (pos0[:, None] + HIST_POSITION_OFFSET).expand(batch, n_hist),
        pos0[:, None] + AR_POSITION_OFFSET + arange(n_ar)[None],
    ), dim=1)
    seg_ids = torch.cat((
        torch.full((n_rawdiff,), SEG_RAWDIFF, dtype=torch.long, device=device),
        torch.full((n_hist,), SEG_HIST, dtype=torch.long, device=device),
        torch.full((n_ar,), SEG_AR, dtype=torch.long, device=device),
    ))

    bias = torch.full((batch, total, nonki_len + total), NEG, dtype=torch.float32, device=device)
    rows = arange(total)
    bias[:, rows, nonki_len + rows] = 0.0
    ar0 = n_rawdiff + n_hist
    if n_ar:
        zero = torch.zeros((), dtype=torch.float32, device=device)
        neg = torch.full((), NEG, dtype=torch.float32, device=device)
        bias[:, ar0:, :nonki_len] = torch.where(prefix_valid[:, None, :], zero, neg)
        if n_rawdiff:
            bias[:, ar0:, nonki_len:nonki_len + n_rawdiff] = torch.where(
                rawdiff_valid[:, None, :], zero, neg)
        if n_hist:
            hist = slice(nonki_len + n_rawdiff, nonki_len + ar0)
            if hist_closed:
                bias[:, ar0:, hist] = NEG
            elif log_gate is None:
                bias[:, ar0:, hist] = 0.0
            else:
                bias[:, ar0:, hist] = log_gate.float().view(batch, 1, 1).expand(batch, n_ar, n_hist)
        causal = torch.ones(n_ar, n_ar, dtype=torch.bool, device=device).tril()
        bias[:, ar0:, nonki_len + ar0:] = torch.where(causal, zero, neg)
    return SuffixLayout(positions, bias[:, None], seg_ids, n_rawdiff, n_hist, n_ar, nonki_len)


def prefill_view(layout: SuffixLayout, n_rows: int) -> Tuple[Tensor, Tensor, Tensor]:
    """Bias, positions and segment ids for the first ``n_rows`` suffix tokens (no cache)."""
    if not 1 <= n_rows <= layout.positions.shape[1]:
        raise ValueError("n_rows outside the suffix")
    keys = layout.nonki_len + n_rows
    return (layout.bias[:, :, :n_rows, :keys], layout.positions[:, :n_rows],
            layout.seg_ids[:n_rows])


def decode_view(layout: SuffixLayout, ar_index: int) -> Tuple[Tensor, Tensor]:
    """Bias row and position of AR input ``ar_index`` given everything before it is cached."""
    if not 0 <= ar_index < layout.n_ar:
        raise ValueError("ar_index outside the AR block")
    row = layout.n_cond + ar_index
    keys = layout.nonki_len + row + 1
    return layout.bias[:, :, row:row + 1, :keys], layout.positions[:, row:row + 1]


def build_flow_suffix_layout(pos0: Tensor, prefix_valid: Tensor, n_act: int) -> SuffixLayout:
    """PI0.5 flow suffix: action tokens see valid prefix keys and all action tokens."""
    prefix_valid = _bool_matrix(prefix_valid, "prefix_valid")
    batch, nonki_len = prefix_valid.shape
    pos0 = _validated_pos0(pos0, batch)
    if isinstance(n_act, bool) or not isinstance(n_act, int) or n_act < 1:
        raise ValueError("n_act must be a positive integer")
    device = prefix_valid.device
    positions = pos0[:, None] + torch.arange(n_act, device=device)[None]
    bias = torch.zeros(batch, n_act, nonki_len + n_act, dtype=torch.float32, device=device)
    bias[:, :, :nonki_len].masked_fill_(~prefix_valid[:, None, :], NEG)
    seg_ids = torch.full((n_act,), SEG_AR, dtype=torch.long, device=device)
    return SuffixLayout(positions, bias[:, None], seg_ids, 0, 0, n_act, nonki_len)
