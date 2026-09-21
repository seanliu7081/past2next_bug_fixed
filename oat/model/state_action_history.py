"""Temporal fusion of observed state histories and aligned command histories."""

from math import prod
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn


class StateActionHistoryEncoder(nn.Module):
    """Summarize H observed states and H-1 commands into fixed-size memory.

    Quaternion ports are identified by a ``_quat`` suffix and must have shape
    (4,), in xyzw order. Absolute orientations and world-frame relative
    rotations ``R_t @ R_(t-1).T`` use the first two matrix columns (6D).
    Ordinary fields use the supplied, already-fitted normalizer. Their deltas
    are differences in that same normalized space, without division by dt.

    Commands must already be normalized. At state index i, its action feature
    is command i-1; only transitions between two valid states have action and
    delta features. These are aligned conditions, not an assertion that a
    synthetic self-past command caused the observed transition. No dynamics
    supervision or environment execution is performed by this module.

    All observations are at or before the current policy decision. Attention
    can therefore mix the entire valid history. Padded states are masked in
    both temporal attention and learned-query summarization.
    """

    def __init__(
        self,
        state_shapes: Optional[Mapping[str, Sequence[int]]] = None,
        action_dim: int = 7,
        history_steps: int = 8,
        output_dim: int = 256,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        n_summary_tokens: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if state_shapes is None:
            state_shapes = {
                "robot0_eef_pos": (3,),
                "robot0_eef_quat": (4,),
                "robot0_gripper_qpos": (2,),
            }
        self.state_shapes = {key: tuple(shape) for key, shape in state_shapes.items()}
        if not self.state_shapes or any(
            not shape or any(not isinstance(size, int) or size < 1 for size in shape)
            for shape in self.state_shapes.values()
        ):
            raise ValueError("state_shapes must contain nonempty positive tensor shapes")
        if history_steps < 2 or min(action_dim, output_dim, embed_dim, n_heads,
                                    n_layers, n_summary_tokens) < 1:
            raise ValueError("history_steps must be at least 2 and model dimensions positive")
        if embed_dim % n_heads:
            raise ValueError("embed_dim must be divisible by n_heads")
        self.quaternion_keys = tuple(key for key in self.state_shapes if key.endswith("_quat"))
        if any(self.state_shapes[key] != (4,) for key in self.quaternion_keys):
            raise ValueError("Quaternion state ports must have shape (4,) in xyzw order")
        self.action_dim = action_dim
        self.history_steps = history_steps
        self.output_dim = output_dim
        self.n_summary_tokens = n_summary_tokens

        # Public feature slices make state/action timing explicit and inspectable.
        self.feature_slices = {}
        offset = 0
        for group in ("absolute", "delta"):
            for key, shape in self.state_shapes.items():
                size = 6 if key in self.quaternion_keys else prod(shape)
                self.feature_slices[f"{group}/{key}"] = slice(offset, offset + size)
                offset += size
        for key, size in (("action", action_dim), ("state_valid", 1),
                          ("transition_valid", 1), ("action_valid", 1)):
            self.feature_slices[key] = slice(offset, offset + size)
            offset += size
        self.feature_dim = offset
        self.input_projection = nn.Sequential(nn.Linear(offset, embed_dim), nn.GELU())
        self.position_embedding = nn.Parameter(torch.empty(1, history_steps, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=4 * embed_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            layer, num_layers=n_layers, norm=nn.LayerNorm(embed_dim),
            enable_nested_tensor=False,
        )
        self.summary_queries = nn.Parameter(torch.empty(1, n_summary_tokens, embed_dim))
        self.summary_attention = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True,
        )
        self.output_projection = nn.Sequential(nn.LayerNorm(embed_dim),
                                               nn.Linear(embed_dim, output_dim))
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.summary_queries, std=0.02)

    @staticmethod
    def _rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
        """Convert normalized xyzw quaternions without choosing an antipodal sign."""
        x, y, z, w = quaternion.unbind(-1)
        return torch.stack((
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ), dim=-1).reshape(*quaternion.shape[:-1], 3, 3)

    @staticmethod
    def _rotation_6d(rotation: torch.Tensor) -> torch.Tensor:
        # Column 0 followed by column 1, rather than flattened interleaved rows.
        return rotation[..., :, :2].transpose(-1, -2).flatten(-2)

    def build_features(
        self,
        state_history: Dict[str, torch.Tensor],
        state_valid: torch.Tensor,
        past_actions: torch.Tensor,
        normalizer,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return features [B,H,F] and the True-means-padding mask [B,H].

        Layout is all absolute state fields, all delta fields, aligned action,
        then state/transition/action validity flags; see ``feature_slices``.
        A static valid rotation has an identity relative rotation, not six zeros.
        Invalid entries have zero features and cannot contaminate valid entries.
        """
        if (not isinstance(state_valid, torch.Tensor) or state_valid.dtype != torch.bool
                or state_valid.ndim != 2 or state_valid.shape[1] != self.history_steps
                or state_valid.shape[0] < 1):
            raise ValueError("state_valid must be a nonempty bool tensor of shape (B,H)")
        if (not state_valid[:, -1].all()
                or (state_valid[:, :-1] & ~state_valid[:, 1:]).any()):
            raise ValueError("Valid states must form a contiguous suffix ending at the current state")
        batch_size, steps = state_valid.shape
        if (not isinstance(past_actions, torch.Tensor) or not past_actions.is_floating_point()
                or past_actions.shape != (batch_size, steps - 1, self.action_dim)
                or past_actions.device != state_valid.device):
            raise ValueError("past_actions must be floating (B,H-1,action_dim) on the mask device")
        transition_valid = torch.cat((torch.zeros_like(state_valid[:, :1]),
                                      state_valid[:, :-1] & state_valid[:, 1:]), dim=1)
        if not torch.isfinite(past_actions[transition_valid[:, 1:]]).all():
            raise ValueError("Valid past actions must be finite")
        safe_actions = torch.where(transition_valid[:, 1:, None], past_actions,
                                   torch.zeros_like(past_actions))
        aligned_actions = torch.cat((safe_actions.new_zeros(batch_size, 1, self.action_dim),
                                     safe_actions), dim=1)

        absolute_features, delta_features = [], []
        for key, shape in self.state_shapes.items():
            value = state_history.get(key)
            if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
                    or value.shape != (batch_size, steps, *shape)
                    or value.device != state_valid.device):
                raise ValueError(f"State {key!r} must be floating (B,H,{shape}) on the mask device")
            if not torch.isfinite(value[state_valid]).all():
                raise ValueError(f"Valid state {key!r} must be finite")
            value = value.flatten(2)
            # Sanitize before normalization / quaternion conversion. Masking NaN
            # after arithmetic alone can still poison gradients.
            value = torch.where(state_valid[..., None], value, torch.zeros_like(value))
            if key in self.quaternion_keys:
                quaternion = value.float() if value.dtype in (torch.float16, torch.bfloat16) else value
                identity = quaternion.new_tensor([0, 0, 0, 1])
                quaternion = torch.where(state_valid[..., None], quaternion, identity)
                norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
                if (not torch.isfinite(norm).all() or (norm <= 1e-8).any()):
                    raise ValueError(f"Valid quaternion {key!r} must have finite nonzero norm")
                # Relative geometry should not lose precision to the policy's
                # mixed-precision matmul autocast before feature projection.
                with torch.autocast(device_type=quaternion.device.type, enabled=False):
                    rotation = self._rotation_matrix(quaternion / norm)
                    absolute = self._rotation_6d(rotation)
                    relative = rotation[:, 1:] @ rotation[:, :-1].transpose(-1, -2)
                    delta = self._rotation_6d(relative)
            else:
                absolute = normalizer[key].normalize(value)
                delta = absolute[:, 1:] - absolute[:, :-1]
            absolute = torch.where(state_valid[..., None], absolute, torch.zeros_like(absolute))
            delta = torch.cat((delta.new_zeros(batch_size, 1, delta.shape[-1]), delta), dim=1)
            delta = torch.where(transition_valid[..., None], delta, torch.zeros_like(delta))
            absolute_features.append(absolute.to(device=past_actions.device, dtype=past_actions.dtype))
            delta_features.append(delta.to(device=past_actions.device, dtype=past_actions.dtype))
        flags = [mask[..., None].to(dtype=past_actions.dtype)
                 for mask in (state_valid, transition_valid, transition_valid)]
        features = torch.cat((*absolute_features, *delta_features, aligned_actions, *flags), dim=-1)
        return features, ~state_valid

    def forward(self, state_history, state_valid, past_actions, normalizer):
        features, padding_mask = self.build_features(
            state_history, state_valid, past_actions, normalizer,
        )
        memory = self.input_projection(features) + self.position_embedding
        memory = self.temporal_encoder(memory, src_key_padding_mask=padding_mask)
        queries = self.summary_queries.expand(memory.shape[0], -1, -1)
        summary, _ = self.summary_attention(
            queries, memory, memory, key_padding_mask=padding_mask, need_weights=False,
        )
        return self.output_projection(queries + summary)
