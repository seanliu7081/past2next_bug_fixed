"""Real-robot rotation-6D adapter for the independent state-history gate policy.

Current observations keep their original position/rotation-6D/gripper schema.
Only the history encoder converts raw orientations to geometric absolute and
world-relative rotations, matching the original quaternion history semantics.
"""

import torch

from oat.model.state_action_history import StateActionHistoryEncoder
from oat.policy.past2next_state_history_gate import Past2NextStateHistoryGatePolicy


REAL_ROBOT_STATE_KEYS = (
    "robot0_eef_pos", "robot0_eef_rot6d", "robot0_gripper_qpos",
)


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


class Past2NextRealRobotStateHistoryGatePolicy(Past2NextStateHistoryGatePolicy):
    """Gate policy with native real-robot observations and geometric history.

    No robot or simulator is connected by this class. Deployed callers must
    supply every control-step history and acknowledge actual command execution,
    following the same interfaces as the parent state-history gate policy.
    """

    def __init__(self, *args, state_history_keys=REAL_ROBOT_STATE_KEYS,
                 rotation_6d_layout="rows", **kwargs):
        super().__init__(*args, state_history_keys=state_history_keys, **kwargs)
        old = self.history_encoder
        # Retain all learned tensor shapes and initial weights; introducing a
        # geometry adapter should not change the initialized network or RNG.
        with torch.random.fork_rng(devices=[]):
            encoder = Rotation6DStateActionHistoryEncoder(
                state_shapes=old.state_shapes, action_dim=old.action_dim,
                history_steps=old.history_steps, output_dim=old.output_dim,
                embed_dim=old.input_projection[0].out_features,
                n_heads=old.summary_attention.num_heads,
                n_layers=len(old.temporal_encoder.layers),
                n_summary_tokens=old.n_summary_tokens,
                dropout=old.summary_attention.dropout,
                rotation_6d_layout=rotation_6d_layout,
            )
        encoder.to(device=old.position_embedding.device, dtype=old.position_embedding.dtype)
        encoder.load_state_dict(old.state_dict())
        encoder.train(old.training)
        self.history_encoder = encoder
        self.rotation_6d_layout = rotation_6d_layout

    def get_policy_name(self):
        return "past2next_state_history_gate_real_robot_" + "|".join(
            modality for modality in self.modalities if modality != "state"
        )

    def create_dummy_observation(self, batch_size=1, device=None):
        obs = super().create_dummy_observation(batch_size=batch_size, device=device)
        for key in self.history_encoder.rotation_6d_keys:
            value = obs["state_history__" + key]
            value[:, -1] = value.new_tensor([1, 0, 0, 0, 1, 0])
        return obs
