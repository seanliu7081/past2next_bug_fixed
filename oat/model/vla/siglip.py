"""Frozen SigLIP So400m/14 vision tower plus the PaliGemma projector."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from oat.model.vla.specs import SiglipSpec


class SiglipEncoder(nn.Module):
    """Images in [-1, 1] -> unscaled image tokens of the VLM width (PI0.5 convention)."""

    def __init__(self, spec: SiglipSpec, out_dim: int, frozen_dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        from transformers import SiglipVisionConfig, SiglipVisionModel

        config = SiglipVisionConfig(
            hidden_size=spec.hidden, intermediate_size=spec.intermediate,
            num_hidden_layers=spec.layers, num_attention_heads=spec.heads,
            patch_size=spec.patch, image_size=spec.image_size, num_channels=3,
            layer_norm_eps=spec.eps, hidden_act="gelu_pytorch_tanh",
        )
        # transformers builds a randomly initialized MAP head unless this attribute exists.
        config.vision_use_head = False
        self.spec = spec
        self.vision = SiglipVisionModel(config)
        if getattr(self.vision.vision_model, "head", None) is not None:
            raise RuntimeError("SigLIP must not contain a pooling head")
        self.projector = nn.Linear(spec.hidden, out_dim)
        self.to(frozen_dtype)
        self.requires_grad_(False)
        super().train(False)

    def train(self, mode: bool = True):
        # Frozen and deterministic in every mode.
        return super().train(False)

    @property
    def num_tokens(self) -> int:
        return self.spec.num_tokens

    @torch.no_grad()
    def forward(self, pixels: Tensor) -> Tensor:
        expected = (3, self.spec.image_size, self.spec.image_size)
        if pixels.ndim != 4 or tuple(pixels.shape[1:]) != expected:
            raise ValueError(f"pixels must be [N, {expected[0]}, {expected[1]}, {expected[2]}]")
        dtype = self.projector.weight.dtype
        hidden = self.vision(pixel_values=pixels.to(dtype)).last_hidden_state
        return self.projector(hidden)
