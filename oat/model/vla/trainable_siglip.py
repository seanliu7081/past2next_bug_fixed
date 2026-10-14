"""Trainable SigLIP wrapper that retains the pretrained encoder's state keys."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from oat.model.vla.siglip import SiglipEncoder


class TrainableSiglipEncoder(nn.Module):
    """Reuse an already loaded vision tower and projector without allocating a copy.

    Master parameters are FP32; the policy's CUDA autocast provides BF16 compute.
    The original frozen encoder remains unchanged for the existing LoRA policies.
    """

    def __init__(self, encoder: SiglipEncoder, *, activation_checkpointing: bool = True):
        super().__init__()
        self.spec = encoder.spec
        self.vision = encoder.vision
        self.projector = encoder.projector
        self.activation_checkpointing = bool(activation_checkpointing)
        self.float().requires_grad_(True)
        if self.activation_checkpointing:
            self.vision.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            self.vision.gradient_checkpointing_disable()

    @property
    def num_tokens(self) -> int:
        return self.spec.num_tokens

    def forward(self, pixels: Tensor) -> Tensor:
        expected = (3, self.spec.image_size, self.spec.image_size)
        if pixels.ndim != 4 or tuple(pixels.shape[1:]) != expected:
            raise ValueError(f"pixels must be [N, {expected[0]}, {expected[1]}, {expected[2]}]")
        hidden = self.vision(pixel_values=pixels.to(self.projector.weight.dtype)).last_hidden_state
        return self.projector(hidden)
