"""Trainable ResNet-18 observation tokens for direct action flow.

This adapter reuses the original Robomimic camera architecture, normalization,
GroupNorm, SpatialSoftmax, and crop behavior. A prepared conditioning tensor
contains normalized image crops, never cached backbone features: the student
and EMA teacher each encode the same crops with their own camera networks.
"""
from __future__ import annotations

from oat.perception.latent_flow_resnet_obs_encoder import FlowResNetObservationEncoder


class ActionFlowResNetObservationEncoder(FlowResNetObservationEncoder):
    """One trainable visual token per camera/frame plus current-state tokens."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The repository normalizer reconstructs its parameter dictionary from
        # checkpoint tensors. A post-hook also runs during a containing policy's
        # load_state_dict, after those tensors have actually been reconstructed.
        self.register_load_state_dict_post_hook(self._restore_normalizer_after_load)

    def _restore_pixel_normalizer(self):
        reference = self.visual_projection.weight
        self.vision_encoder.normalizer.to(device=reference.device, dtype=reference.dtype)
        self.vision_encoder.normalizer.requires_grad_(False)

    def _restore_normalizer_after_load(self, module, incompatible_keys):
        self._restore_pixel_normalizer()

    def set_normalizer(self, normalizer):
        super().set_normalizer(normalizer)
        self._restore_pixel_normalizer()

    def requires_grad_(self, requires_grad=True):
        super().requires_grad_(requires_grad)
        self.vision_encoder.normalizer.requires_grad_(False)
        return self
