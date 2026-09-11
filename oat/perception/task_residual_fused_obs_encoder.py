"""Optional categorical task residual for fused observation features."""
from numbers import Integral
from typing import Dict, Optional, Sequence

import torch
from torch import nn
from omegaconf import DictConfig

from oat.perception.fused_obs_encoder import FusedObservationEncoder


class TaskResidualFusedObservationEncoder(FusedObservationEncoder):
    """Add a zero-initialized task table while retaining all original features.

    The normalized scalar UID remains in the fused feature. A raw UID selects
    a learned residual of the same dimension, preserving feature shapes and
    the policy's history/dynamics builder. By default the task IDs are
    LIBERO-10's 30..39; other task sets can supply their raw IDs explicitly.
    Only `task_residual.weight` is new checkpoint state.

    Legacy state dictionaries may omit this one table; it is then initialized
    to zero. All other strict loading checks remain in place, and a table
    present in a new checkpoint is loaded normally. A metadata-free state dict
    cannot distinguish a legacy file from a later file with this key deleted.
    """

    FIRST_TASK_UID = 30
    NUM_TASKS = 10

    def __init__(self, shape_meta: Dict,
                 vision_encoder: Optional[DictConfig] = None,
                 text_encoder: Optional[DictConfig] = None,
                 state_encoder: Optional[DictConfig] = None,
                 task_uids: Optional[Sequence[int]] = None):
        if task_uids is None:
            task_uids = range(self.FIRST_TASK_UID, self.FIRST_TASK_UID + self.NUM_TASKS)
        task_uids = tuple(task_uids)
        if not task_uids or any(isinstance(uid, bool) or not isinstance(uid, Integral)
                                for uid in task_uids):
            raise ValueError("task_uids must be a nonempty sequence of integer IDs")
        if len(set(task_uids)) != len(task_uids):
            raise ValueError("task_uids must contain unique integer IDs")
        uid_meta = shape_meta.get("obs", {}).get("task_uid", {})
        if uid_meta.get("type") != "state" or list(uid_meta.get("shape", [])) != [1]:
            raise ValueError("Task residual requires the original scalar task_uid state port")
        super().__init__(shape_meta=shape_meta, vision_encoder=vision_encoder,
                         text_encoder=text_encoder, state_encoder=state_encoder)
        self.task_uids = tuple(int(uid) for uid in task_uids)
        self.num_tasks = len(self.task_uids)
        self.feature_dim = self.output_feature_dim()
        # The task configuration supplies IDs when a checkpoint is restored.
        # Keep the lookup on the module's device without adding checkpoint keys.
        self.register_buffer("_task_uids", torch.tensor(self.task_uids), persistent=False)
        # Avoid nn.Embedding's random initialization so even constructor RNG
        # consumption matches the unchanged fused encoder.
        self.task_residual = nn.Embedding.from_pretrained(
            torch.zeros(self.num_tasks, self.feature_dim), freeze=False,
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
        valid_ids = ", ".join(str(task_uid) for task_uid in self.task_uids)
        if uid.dtype == torch.bool or uid.is_complex():
            raise ValueError(f"task_uid must contain raw integer IDs from [{valid_ids}]")
        if not torch.isfinite(uid).all():
            raise ValueError(f"task_uid must contain finite raw integer IDs from [{valid_ids}]")
        if uid.is_floating_point() and not torch.equal(uid, uid.round()):
            raise ValueError(f"task_uid must contain raw integer IDs from [{valid_ids}]")
        matches = uid == self._task_uids
        if not matches.any(dim=-1).all():
            raise ValueError(f"task_uid must contain raw integer IDs from [{valid_ids}]")
        indices = matches.long().argmax(dim=-1)
        features = super().forward(obs_dict)
        if tuple(features.shape[:2]) != tuple(indices.shape):
            raise ValueError("Task IDs must match the fused batch and frame dimensions")
        residual = self.task_residual(indices).to(dtype=features.dtype)
        return features + residual
