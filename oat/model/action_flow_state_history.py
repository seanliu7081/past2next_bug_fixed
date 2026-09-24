"""Continuous-action rotation-6D history geometry without policy/codec imports.

The geometry and validity rules match the existing real-robot history adapter.
Keeping this class here makes direct-flow imports independent of OAT and FSQ.
"""

import torch

from oat.model.state_action_history import StateActionHistoryEncoder


class Rotation6DStateActionHistoryEncoder(StateActionHistoryEncoder):
    """Decode raw 6D orientations before computing R[t] @ R[t-1].T.

    Input layout must be specified as two contiguous matrix rows or columns.
    Output geometry always uses the original history encoder's two-column 6D
    representation. Rotation features bypass scalar normalization; other state
    fields and action commands retain their original normalization and units.
    """

    def __init__(self, *args, rotation_6d_layout="rows", **kwargs):
        if rotation_6d_layout not in ("rows", "columns"):
            raise ValueError("rotation_6d_layout must be rows or columns")
        super().__init__(*args, **kwargs)
        self.rotation_6d_layout = rotation_6d_layout
        self.rotation_6d_keys = tuple(key for key in self.state_shapes if key.endswith("_rot6d"))
        if any(self.state_shapes[key] != (6,) for key in self.rotation_6d_keys):
            raise ValueError("Rotation-6D history fields must have shape (6,)")

    def _matrix_from_6d(self, value):
        first, second = value[..., :3], value[..., 3:]
        first_norm = torch.linalg.vector_norm(first, dim=-1, keepdim=True)
        if (first_norm <= 1e-8).any():
            raise ValueError("Valid rotation-6D history has a degenerate first axis")
        first = first / first_norm
        second = second - (first * second).sum(dim=-1, keepdim=True) * first
        second_norm = torch.linalg.vector_norm(second, dim=-1, keepdim=True)
        if (second_norm <= 1e-8).any():
            raise ValueError("Valid rotation-6D history has collinear axes")
        second = second / second_norm
        third = torch.linalg.cross(first, second, dim=-1)
        dim = -2 if self.rotation_6d_layout == "rows" else -1
        return torch.stack((first, second, third), dim=dim)

    def build_features(self, state_history, state_valid, past_actions, normalizer):
        # Reuse the established checks, temporal alignment, boundary masks,
        # non-rotation features, and action validity handling unchanged.
        features, padding_mask = super().build_features(
            state_history, state_valid, past_actions, normalizer,
        )
        features = features.clone()
        transition_valid = torch.cat((torch.zeros_like(state_valid[:, :1]),
                                     state_valid[:, :-1] & state_valid[:, 1:]), dim=1)
        for key in self.rotation_6d_keys:
            raw = state_history[key]
            geometry = raw.float() if raw.dtype in (torch.float16, torch.bfloat16) else raw
            identity = geometry.new_tensor([1, 0, 0, 0, 1, 0])
            geometry = torch.where(state_valid[..., None], geometry, identity)
            with torch.autocast(device_type=geometry.device.type, enabled=False):
                rotation = self._matrix_from_6d(geometry)
                absolute = self._rotation_6d(rotation)
                relative = rotation[:, 1:] @ rotation[:, :-1].transpose(-1, -2)
                delta = self._rotation_6d(relative)
            delta = torch.cat((delta.new_zeros(delta.shape[0], 1, 6), delta), dim=1)
            absolute = torch.where(state_valid[..., None], absolute, torch.zeros_like(absolute))
            delta = torch.where(transition_valid[..., None], delta, torch.zeros_like(delta))
            features[..., self.feature_slices[f"absolute/{key}"]] = absolute.to(features.dtype)
            features[..., self.feature_slices[f"delta/{key}"]] = delta.to(features.dtype)
        return features, padding_mask

