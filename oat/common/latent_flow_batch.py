"""Detached data crossing the single DDP student forward boundary."""
from dataclasses import dataclass
from typing import Mapping
import torch

@dataclass
class PreparedFlowBatch:
    noisy_latents: torch.Tensor
    time: torch.Tensor
    step_size: torch.Tensor
    velocity_targets: torch.Tensor
    fm_indices: torch.Tensor
    ct_indices: torch.Tensor
    obs: Mapping[str, torch.Tensor]
    past_actions: torch.Tensor
    past_action_valid: torch.Tensor
    frozen_patches: torch.Tensor

    def validate(self):
        batch = self.noisy_latents.shape[0]
        if self.noisy_latents.ndim != 3 or self.velocity_targets.shape != self.noisy_latents.shape:
            raise ValueError('Noisy latents and velocity targets must be matching [B,slots,code_dim]')
        for name in ('noisy_latents', 'time', 'step_size', 'velocity_targets'):
            value = getattr(self, name)
            if value.dtype != torch.float32 or value.requires_grad or not torch.isfinite(value).all():
                raise ValueError(f'{name} must be detached, finite FP32')
        if self.time.shape != (batch,) or self.step_size.shape != (batch,):
            raise ValueError('Time and step_size must preserve the batch axis')
        indices = torch.cat((self.fm_indices, self.ct_indices))
        if indices.dtype != torch.int64 or not torch.equal(indices.sort().values, torch.arange(batch, device=indices.device)):
            raise ValueError('FM/CT rows must partition every sample exactly once')
        if batch < 4 or batch % 4 or len(self.fm_indices) != 3 * batch // 4:
            raise ValueError('Training requires a 3:1 split and microbatch divisible by four')
        if self.past_action_valid.dtype != torch.bool or self.past_action_valid.shape != self.past_actions.shape[:2]:
            raise ValueError('Historical command validity must be bool [B,past_n]')
        if self.frozen_patches.requires_grad or self.past_actions.requires_grad:
            raise ValueError('Prepared conditioning must be detached')
        return self

FlowTrainingBatch = PreparedFlowBatch
