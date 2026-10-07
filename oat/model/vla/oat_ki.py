"""Knowledge-insulation (KI) vocabulary: the VLM side of P2N-VLA predicts the 8 OAT tokens.

A trainable fp32 table ``rows`` of shape ``[n_codes + 1, Wv]``:
- rows ``0..n_codes-1`` are OAT codes, initialized from the RAW PaliGemma embedding rows
  ``E[V - 1 - skip - t]`` (pi052's mapping with ``fast_skip_tokens=1152``, which keeps every ``<loc>``/``<seg>``
  row intact; for V=257152 that is ids 255999 down to 251000);
- row ``n_codes`` is ``KI_BOS``, initialized from the raw ``<bos>`` row ``E[2]``.

The KI block appended after the (right-padded) prompt is ``[KI_BOS, z0, ..., z6]``; it is causal and
self-contained, so the logits at block position k predict ``z_k`` (k = 0..7) without ever reading a pad row.
Inputs are scaled by ``sqrt(Wv)`` like Gemma text embeddings; the logits are tied to the unscaled rows
(``hidden @ rows[:n_codes].T``) and computed in fp32 with autocast disabled.
"""
from __future__ import annotations

import math
from numbers import Integral

import torch
from torch import Tensor, nn
from torch.nn import functional as F


PALIGEMMA_BOS_ID = 2


def _positive_int(value, name, minimum=1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(value)


class OATKITable(nn.Module):
    """``rows: Parameter[n_codes + 1, width]`` fp32; ``rows[n_codes]`` is ``KI_BOS``."""

    def __init__(self, width: int, n_codes: int = 5000, skip: int = 1152):
        super().__init__()
        self.width = _positive_int(width, "width")
        self.n_codes = _positive_int(n_codes, "n_codes")
        self.skip = _positive_int(skip, "skip", minimum=0)
        self.bos_index = self.n_codes
        self.embed_scale = math.sqrt(self.width)
        self.rows = nn.Parameter(torch.empty(self.n_codes + 1, self.width, dtype=torch.float32))
        # Placeholder at the scale of Gemma rows; the policy always calls init_from_embedding.
        nn.init.normal_(self.rows, std=self.width ** -0.5)

    def extra_repr(self) -> str:
        return f"width={self.width}, n_codes={self.n_codes}, skip={self.skip}"

    def source_indices(self, vocab_size: int) -> Tensor:
        """Embedding ids copied into rows ``0..n_codes-1``: ``vocab_size - 1 - skip - t``."""
        vocab_size = _positive_int(vocab_size, "vocab_size")
        first = vocab_size - 1 - self.skip
        if first - (self.n_codes - 1) <= PALIGEMMA_BOS_ID:
            raise ValueError(f"Vocabulary of {vocab_size} rows cannot hold {self.n_codes} KI rows below "
                             f"skip={self.skip} without reaching the special tokens")
        return first - torch.arange(self.n_codes, dtype=torch.long)

    @torch.no_grad()
    def init_from_embedding(self, embed_weight: Tensor) -> None:
        """Copy RAW (unscaled) rows: ``rows[t] = E[V-1-skip-t]`` and ``rows[n_codes] = E[2]`` (``<bos>``)."""
        if not isinstance(embed_weight, torch.Tensor) or not embed_weight.is_floating_point():
            raise TypeError("embed_weight must be a floating torch.Tensor")
        if embed_weight.ndim != 2 or embed_weight.shape[1] != self.width:
            raise ValueError(f"embed_weight must have shape [V, {self.width}], got {tuple(embed_weight.shape)}")
        source = embed_weight.detach()
        index = self.source_indices(source.shape[0]).to(source.device)
        codes = source.index_select(0, index).float()
        bos = source[PALIGEMMA_BOS_ID].float()
        if not (torch.isfinite(codes).all() and torch.isfinite(bos).all()):
            raise ValueError("Embedding rows used for the KI table must be finite")
        self.rows[:self.n_codes].copy_(codes.to(self.rows.device))
        self.rows[self.n_codes].copy_(bos.to(self.rows.device))

    def _targets(self, targets: Tensor) -> Tensor:
        if not isinstance(targets, torch.Tensor):
            raise TypeError("KI targets must be a torch.Tensor")
        if targets.dtype == torch.bool or targets.is_floating_point() or targets.is_complex():
            raise TypeError(f"KI targets must be integer token ids, got {targets.dtype}")
        if targets.ndim != 2 or targets.shape[1] < 1:
            raise ValueError(f"KI targets must have shape [B, L], got {tuple(targets.shape)}")
        if targets.device != self.rows.device:
            raise ValueError(f"KI targets are on {targets.device}, table on {self.rows.device}")
        if targets.numel() and ((targets < 0).any() or (targets >= self.n_codes).any()):
            raise ValueError(f"KI targets must lie in [0, {self.n_codes})")
        return targets.long()

    def block_ids(self, targets: Tensor) -> Tensor:
        """Table ids of the KI input block: ``[KI_BOS, t_0, ..., t_{L-2}]`` for targets ``[B, L]``."""
        targets = self._targets(targets)
        bos = torch.full_like(targets[:, :1], self.bos_index)
        return torch.cat((bos, targets[:, :-1]), dim=1)

    def embed_block(self, targets: Tensor) -> Tensor:
        """``[B, L, width]`` fp32 KI inputs: ``rows[[KI_BOS, t_0..t_{L-2}]] * sqrt(width)``."""
        return F.embedding(self.block_ids(targets), self.rows) * self.embed_scale

    def logits(self, hidden: Tensor) -> Tensor:
        """``[..., n_codes]`` fp32 logits from KI-row hidden states ``[..., width]`` (tied, unscaled rows)."""
        if not isinstance(hidden, torch.Tensor) or not hidden.is_floating_point():
            raise TypeError("hidden must be a floating torch.Tensor")
        if hidden.ndim < 1 or hidden.shape[-1] != self.width:
            raise ValueError(f"hidden must have shape [..., {self.width}], got {tuple(hidden.shape)}")
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            return hidden.float() @ self.rows[:self.n_codes].float().t()
