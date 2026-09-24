"""Detached action-flow inputs with shared image crops for trainable ResNet-18."""
from dataclasses import dataclass
from typing import Mapping

import torch

from oat.model.flow.consistency_flow import validate_training_microbatch


@dataclass
class PreparedActionFlowResNetBatch:
    noisy_actions: torch.Tensor
    time: torch.Tensor
    step_size: torch.Tensor
    velocity_targets: torch.Tensor
    fm_indices: torch.Tensor
    ct_indices: torch.Tensor
    obs: Mapping[str, torch.Tensor]
    past_actions: torch.Tensor
    past_action_valid: torch.Tensor
    prepared_visual: torch.Tensor

    def validate(self):
        if self.noisy_actions.ndim != 3 or self.velocity_targets.shape != self.noisy_actions.shape:
            raise ValueError("Noisy actions and velocity targets must be matching [B,horizon,action_dim].")
        batch, horizon, action_dim = self.noisy_actions.shape
        validate_training_microbatch(batch)
        if min(horizon, action_dim) < 1:
            raise ValueError("Action horizon and dimension must be positive.")
        device = self.noisy_actions.device
        for name in ("noisy_actions", "time", "step_size", "velocity_targets"):
            value = getattr(self, name)
            if value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError(f"{name} must contain finite FP32 values.")
        if self.time.shape != (batch,) or self.step_size.shape != (batch,):
            raise ValueError("Time and step_size must preserve the [B] axis.")
        if ((self.time < 0) | (self.time >= 1)).any() or ((self.step_size < 0) | (self.step_size >= 1)).any():
            raise ValueError("Training time and relative step_size must be in [0,1).")
        for name in ("fm_indices", "ct_indices"):
            value = getattr(self, name)
            if value.ndim != 1 or value.dtype != torch.int64 or value.device != device:
                raise ValueError(f"{name} must be an int64 vector on the action device.")
        indices = torch.cat((self.fm_indices, self.ct_indices))
        if not torch.equal(indices.sort().values, torch.arange(batch, device=device)):
            raise ValueError("FM/CT rows must partition every sample exactly once.")
        if self.fm_indices.numel() != 3 * batch // 4:
            raise ValueError("Training requires a 3:1 FM/CT split.")
        if (self.step_size[self.fm_indices] != 0).any():
            raise ValueError("FM rows must use zero relative step_size.")
        if self.past_actions.ndim != 3 or self.past_actions.shape[0] != batch or self.past_actions.shape[-1] != action_dim:
            raise ValueError("Historical commands must have shape [B,past_n,action_dim].")
        if self.past_action_valid.dtype != torch.bool or self.past_action_valid.shape != self.past_actions.shape[:2]:
            raise ValueError("Historical command validity must be bool [B,past_n].")
        if self.past_actions.dtype != torch.float32 or not torch.isfinite(self.past_actions[self.past_action_valid]).all():
            raise ValueError("Valid historical commands must contain finite FP32 values.")
        if not isinstance(self.obs, Mapping) or not self.obs:
            raise ValueError("Prepared observations must be a nonempty tensor mapping.")
        visual = self.prepared_visual
        if (visual.ndim != 6 or visual.shape[0] != batch or visual.shape[3] != 3
                or min(visual.shape[1:]) < 1):
            raise ValueError("Prepared ResNet crops must have shape [B,frames,cameras,3,height,width].")
        if not visual.is_floating_point() or not torch.isfinite(visual).all():
            raise ValueError("Prepared ResNet crops must contain finite floating-point values.")
        tensors = {name: value for name, value in vars(self).items() if isinstance(value, torch.Tensor)}
        for name, value in self.obs.items():
            if not isinstance(value, torch.Tensor) or value.ndim == 0 or value.shape[0] != batch:
                raise ValueError(f"Observation {name!r} must be a tensor preserving the batch axis.")
            tensors[f"obs.{name}"] = value
        for name, value in tensors.items():
            if value.device != device or value.requires_grad or torch.is_inference(value):
                raise ValueError(f"{name} must be detached, ordinary tensors on the action device.")
        return self


ActionFlowResNetTrainingBatch = PreparedActionFlowResNetBatch
