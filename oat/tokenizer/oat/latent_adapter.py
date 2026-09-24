"""Frozen OAT access in its native, normalized FSQ scalar-code space.

Projection is deliberately independent of ``FSQ.forward``: the latter applies
an encoder-side tanh bound and would change already legal grid points.
"""
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class LatentTargets:
    codes: Tensor
    indices: Tensor


def _ordinary_detached(value: Tensor) -> Tensor:
    # An enclosing inference_mode must never leak inference tensors into a
    # student Linear that saves its inputs for backward.
    with torch.inference_mode(False), torch.no_grad():
        return value.detach().clone()


class FrozenOATLatentAdapter(nn.Module):
    projection_version = "fsq_nearest_grid_v1"
    rounding = "torch.round_ties_to_even"

    def __init__(self, tokenizer: nn.Module, *, levels: Sequence[int] = (8, 5, 5, 5, 5),
                 num_slots: int = 8, action_horizon: int = 16, action_dim: int = 7):
        super().__init__()
        actual_levels = tuple(int(x) for x in tokenizer.quantizer._levels.tolist())
        if actual_levels != tuple(levels):
            raise ValueError(f"OAT FSQ levels {actual_levels} differ from required {tuple(levels)}.")
        checks = {
            "latent_horizon": (int(tokenizer.latent_horizon), int(num_slots)),
            "decoder.latent_horizon": (int(tokenizer.decoder.latent_horizon), int(num_slots)),
            "decoder.sample_horizon": (int(tokenizer.decoder.sample_horizon), int(action_horizon)),
            "decoder.sample_dim": (int(tokenizer.decoder.sample_dim), int(action_dim)),
            "quantizer.dim": (int(tokenizer.quantizer.dim), len(actual_levels)),
        }
        for field, (actual, expected) in checks.items():
            if actual != expected:
                raise ValueError(f"OAT {field}={actual}, expected {expected}.")
        if any(x < 2 for x in actual_levels):
            raise ValueError("Every FSQ level must be at least two.")
        self.tokenizer = tokenizer.requires_grad_(False).eval()
        self.levels = actual_levels
        self.num_slots = int(num_slots)
        self.code_dim = len(actual_levels)
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.register_buffer("_levels", torch.tensor(actual_levels, dtype=torch.long), persistent=False)
        self.train(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.tokenizer.requires_grad_(False).eval()
        return self

    def metadata(self) -> dict:
        return {"latent_space": "fsq_normalized_codes", "levels": list(self.levels),
                "num_slots": self.num_slots, "code_dim": self.code_dim,
                "action_horizon": self.action_horizon, "action_dim": self.action_dim,
                "projection_version": self.projection_version, "rounding": self.rounding}

    def _validate_codes(self, codes: Tensor, *, full_shape: bool = False) -> None:
        if not codes.is_floating_point() or codes.ndim < 1 or codes.shape[-1] != self.code_dim:
            raise ValueError(f"Expected floating point FSQ codes ending in {self.code_dim} coordinates.")
        if full_shape and (codes.ndim != 3 or codes.shape[1:] != (self.num_slots, self.code_dim)):
            raise ValueError(f"Expected codes [B, {self.num_slots}, {self.code_dim}].")
        if not torch.isfinite(codes).all():
            raise ValueError("FSQ codes contain NaN or Inf.")

    def _validate_grid(self, codes: Tensor) -> None:
        with torch.autocast(device_type=codes.device.type, enabled=False):
            levels = self._levels.to(device=codes.device)
            half_width = (levels // 2).float()
            integer_codes = codes.float() * half_width + half_width
            legal = ((integer_codes >= 0) & (integer_codes <= levels - 1)
                     & (integer_codes == torch.round(integer_codes)))
        if not legal.all():
            raise ValueError("Expected legal quantized FSQ grid points.")

    @torch.no_grad()
    def encode_actions(self, raw_actions: Tensor) -> LatentTargets:
        if raw_actions.ndim != 3 or raw_actions.shape[1:] != (self.action_horizon, self.action_dim):
            raise ValueError(f"Expected raw actions [B, {self.action_horizon}, {self.action_dim}].")
        if not raw_actions.is_floating_point() or not torch.isfinite(raw_actions).all():
            raise ValueError("Raw actions must be finite floating point values.")
        self.tokenizer.eval()
        with torch.autocast(device_type=raw_actions.device.type, enabled=False):
            # OAT.encode performs its own saved action normalization exactly once.
            codes, indices = self.tokenizer.encode(raw_actions.float())
        codes = _ordinary_detached(codes.float())
        indices = _ordinary_detached(indices.long())
        self._validate_codes(codes, full_shape=True)
        if indices.shape != codes.shape[:2]:
            raise ValueError("OAT returned malformed token indices.")
        self._validate_grid(codes)
        return LatentTargets(codes, indices)

    @torch.no_grad()
    def snap_codes(self, continuous_codes: Tensor) -> Tensor:
        self._validate_codes(continuous_codes)
        with torch.autocast(device_type=continuous_codes.device.type, enabled=False):
            levels = self._levels.to(device=continuous_codes.device)
            half_width = (levels // 2).float()
            q = torch.round(continuous_codes.float() * half_width + half_width)
            q = torch.minimum(torch.maximum(q, torch.zeros_like(q)), (levels - 1).float())
            return (q - half_width) / half_width

    @torch.no_grad()
    def decode_grid_codes(self, grid_codes: Tensor) -> Tensor:
        self._validate_codes(grid_codes, full_shape=True)
        self._validate_grid(grid_codes)
        self.tokenizer.eval()
        with torch.autocast(device_type=grid_codes.device.type, enabled=False):
            actions = self.tokenizer.decode(grid_codes.float())
        if actions.shape != (grid_codes.shape[0], self.action_horizon, self.action_dim):
            raise ValueError("OAT decoder returned an unexpected action shape.")
        if not torch.isfinite(actions).all():
            raise ValueError("OAT decoder produced nonfinite actions.")
        return _ordinary_detached(actions.float())
