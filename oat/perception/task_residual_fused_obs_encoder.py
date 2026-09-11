"""Optional categorical task residual for the existing LIBERO-10 fused features."""
from typing import Dict, Optional

import torch
from torch import nn
from omegaconf import DictConfig

from oat.perception.fused_obs_encoder import FusedObservationEncoder


class TaskResidualFusedObservationEncoder(FusedObservationEncoder):
    """Add a zero-initialized task table while retaining all original features.

    The original normalized scalar UID remains in the 138-dimensional fused
    feature. Raw UID 30..39 selects a learned residual of the same dimension,
    so tokenization, feature shapes, and the policy's history/dynamics builder
    remain unchanged. Only `task_residual.weight` is new checkpoint state.

    Legacy state dictionaries may omit this one table; it is then initialized
    to zero. All other strict loading checks remain in place, and a table
    present in a new checkpoint is loaded normally. A metadata-free state dict
    cannot distinguish a legacy file from a later file with this key deleted.
    """

    FIRST_TASK_UID = 30
    NUM_TASKS = 10
    FEATURE_DIM = 138

    def __init__(self, shape_meta: Dict,
                 vision_encoder: Optional[DictConfig] = None,
                 text_encoder: Optional[DictConfig] = None,
                 state_encoder: Optional[DictConfig] = None):
        uid_meta = shape_meta.get("obs", {}).get("task_uid", {})
        if uid_meta.get("type") != "state" or list(uid_meta.get("shape", [])) != [1]:
            raise ValueError("Task residual requires the original scalar task_uid state port")
        super().__init__(shape_meta=shape_meta, vision_encoder=vision_encoder,
                         text_encoder=text_encoder, state_encoder=state_encoder)
        if self.output_feature_dim() != self.FEATURE_DIM:
            raise ValueError("Task residual requires the original 138-dimensional fused features")
        # Avoid nn.Embedding's random initialization so even constructor RNG
        # consumption matches the unchanged fused encoder.
        self.task_residual = nn.Embedding.from_pretrained(
            torch.zeros(self.NUM_TASKS, self.FEATURE_DIM), freeze=False,
        )
        self.legacy_task_residual_initialized = False

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        key = prefix + "task_residual.weight"
        self.legacy_task_residual_initialized = key not in state_dict
        if self.legacy_task_residual_initialized:
            # Explicitly synthesize only this new parameter. In particular,
            # never use strict=False or replace any existing learned tensor.
            state_dict[key] = torch.zeros_like(self.task_residual.weight)
            print(f"Initialized missing legacy task residual {key} to zero")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def forward(self, obs_dict: Dict) -> torch.Tensor:
        uid = obs_dict.get("task_uid")
        if not isinstance(uid, torch.Tensor) or uid.ndim != 3 or uid.shape[-1] != 1:
            raise ValueError("task_uid must be a tensor with shape [batch, frames, 1]")
        if uid.dtype == torch.bool or uid.is_complex():
            raise ValueError("task_uid must contain raw integer IDs 30..39")
        if not torch.isfinite(uid).all():
            raise ValueError("task_uid must contain finite raw integer IDs 30..39")
        if uid.is_floating_point() and not torch.equal(uid, uid.round()):
            raise ValueError("task_uid must contain raw integer IDs 30..39")
        if not ((uid >= self.FIRST_TASK_UID) &
                (uid < self.FIRST_TASK_UID + self.NUM_TASKS)).all():
            raise ValueError("task_uid must contain raw integer IDs 30..39")
        indices = uid[..., 0].long() - self.FIRST_TASK_UID
        features = super().forward(obs_dict)
        if tuple(features.shape[:2]) != tuple(indices.shape):
            raise ValueError("Task IDs must match the fused batch and frame dimensions")
        residual = self.task_residual(indices).to(dtype=features.dtype)
        return features + residual
