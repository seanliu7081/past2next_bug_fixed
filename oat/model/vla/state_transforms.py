"""Prompt-state geometry for P2N-VLA (pi05-style discrete state in the language prompt).

The prompt state is a short vector built from the CURRENT observation. Each key is mapped by a named
transform and the parts are concatenated in key order:

- ``identity``: the raw vector (positions, gripper finger joints or widths). No unit conversion is applied;
  the train-split q01/q99 normalizer (``oat/dataset/vla_dataset.py``) handles the scale.
- ``quat_to_axis_angle`` (alias ``quat_xyzw_to_axis_angle``): exactly
  ``robosuite.utils.transform_utils.quat2axisangle``, batched. This is also openpi's LIBERO state
  (``examples/libero/main.py`` ``_quat2axisangle``). Quaternions are NOT canonicalized, so the result keeps
  robosuite's sign convention (angle ``2*acos(w)`` in ``[0, 2*pi]``).
- ``rot6d_rows_to_axis_angle`` / ``rot6d_columns_to_axis_angle``: the Gram-Schmidt reconstruction of
  ``Rotation6DStateActionHistoryEncoder`` (``oat/policy/past2next_state_history_gate_real_robot.py``)
  followed by the SO(3) log map (canonical angle in ``[0, pi]``).

All geometry runs in float64 with autocast disabled, so training and rollout produce identical values.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import torch
from omegaconf import OmegaConf
from torch import Tensor


ROT6D_LAYOUTS = ("rows", "columns")
_ROT6D_EPS = 1e-8

# name -> (input dim or None for any, output dim or None for "same as input")
PROMPT_STATE_TRANSFORMS: Dict[str, Tuple[Optional[int], Optional[int]]] = {
    "identity": (None, None),
    "quat_to_axis_angle": (4, 3),
    "quat_xyzw_to_axis_angle": (4, 3),
    "rot6d_rows_to_axis_angle": (6, 3),
    "rot6d_columns_to_axis_angle": (6, 3),
}

# openpi LIBERO state: eef_pos(3) + axis-angle(eef_quat)(3) + gripper_qpos(2) = 8 dims.
LIBERO_PROMPT_STATE = dict(
    keys=["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
    transforms={
        "robot0_eef_pos": "identity",
        "robot0_eef_quat": "quat_to_axis_angle",
        "robot0_gripper_qpos": "identity",
    },
)


def _plain(value):
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return copy.deepcopy(value)


def _float_vectors(value: Tensor, name: str, last_dim: int) -> Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be a floating tensor, got {value.dtype}")
    if value.ndim < 1 or value.shape[-1] != last_dim:
        raise ValueError(f"{name} must have shape [..., {last_dim}], got {tuple(value.shape)}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value


def _result_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.promote_types(dtype, torch.float32)


def quat_xyzw_to_axis_angle(q: Tensor) -> Tensor:
    """Batched ``robosuite.utils.transform_utils.quat2axisangle`` for xyzw quaternions ``[..., 4] -> [..., 3]``.

    Same arithmetic as robosuite: ``w`` is clipped to [-1, 1], ``den = sqrt(1 - w*w)``, and the result is
    ``xyz * 2 * acos(w) / den``. Robosuite returns zeros iff ``math.isclose(den, 0.0)``, which with its
    default tolerances (abs_tol=0) means ``den == 0`` exactly; the same rule is used here. Computed in
    float64; returned as float32 (float64 for float64 input).
    """
    q = _float_vectors(q, "quaternion", 4)
    with torch.autocast(device_type=q.device.type, enabled=False):
        q64 = q.double()
        w = q64[..., 3].clamp(-1.0, 1.0)
        den = torch.sqrt(1.0 - w * w)
        zero = den == 0
        angle = torch.acos(w)
        out = (q64[..., :3] * 2.0 * angle[..., None]) / torch.where(zero, torch.ones_like(den), den)[..., None]
        out = torch.where(zero[..., None], torch.zeros_like(out), out)
    return out.to(_result_dtype(q.dtype))


def rot6d_to_matrix(r: Tensor, layout: Literal["rows", "columns"]) -> Tensor:
    """Gram-Schmidt 6D -> rotation matrix ``[..., 3, 3]`` (float64), as ``Rotation6DStateActionHistoryEncoder``.

    ``r[..., :3]`` and ``r[..., 3:]`` are the first two rows (``layout='rows'``) or columns
    (``layout='columns'``) of the rotation matrix; the third is their cross product.
    """
    if layout not in ROT6D_LAYOUTS:
        raise ValueError(f"rotation-6D layout must be one of {ROT6D_LAYOUTS}, got {layout!r}")
    r = _float_vectors(r, "rotation-6D", 6)
    with torch.autocast(device_type=r.device.type, enabled=False):
        r64 = r.double()
        first, second = r64[..., :3], r64[..., 3:]
        first_norm = torch.linalg.vector_norm(first, dim=-1, keepdim=True)
        if (first_norm <= _ROT6D_EPS).any():
            raise ValueError("Rotation-6D input has a degenerate first axis")
        first = first / first_norm
        second = second - (first * second).sum(dim=-1, keepdim=True) * first
        second_norm = torch.linalg.vector_norm(second, dim=-1, keepdim=True)
        if (second_norm <= _ROT6D_EPS).any():
            raise ValueError("Rotation-6D input has collinear axes")
        second = second / second_norm
        third = torch.linalg.cross(first, second, dim=-1)
        return torch.stack((first, second, third), dim=-2 if layout == "rows" else -1)


def matrix_to_axis_angle(rotation: Tensor) -> Tensor:
    """SO(3) log map ``[..., 3, 3] -> [..., 3]`` with the canonical angle in ``[0, pi]`` (float64 internally).

    Uses a Shepperd-style quaternion extraction (each row picks its best-conditioned candidate), then
    ``angle = 2*atan2(|xyz|, w)`` with ``w >= 0``; stable near both 0 and pi.
    """
    if not isinstance(rotation, torch.Tensor) or not rotation.is_floating_point():
        raise TypeError("rotation must be a floating torch.Tensor")
    if rotation.ndim < 2 or tuple(rotation.shape[-2:]) != (3, 3):
        raise ValueError(f"rotation must have shape [..., 3, 3], got {tuple(rotation.shape)}")
    if not torch.isfinite(rotation).all():
        raise ValueError("rotation must be finite")
    with torch.autocast(device_type=rotation.device.type, enabled=False):
        m = rotation.double()
        m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
        m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
        m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
        # 4*q_i^2 for i in (w, x, y, z)
        four_sq = torch.stack((1 + m00 + m11 + m22, 1 + m00 - m11 - m22,
                               1 - m00 + m11 - m22, 1 - m00 - m11 + m22), dim=-1)
        q_abs = torch.sqrt(four_sq.clamp_min(0.0))  # 2*|q_i|
        # candidate i holds (w, x, y, z) * 4*q_i, i.e. q scaled by 2*q_abs[i]
        candidates = torch.stack((
            torch.stack((q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01), dim=-1),
            torch.stack((m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20), dim=-1),
            torch.stack((m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21), dim=-1),
            torch.stack((m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2), dim=-1),
        ), dim=-2)
        best = q_abs.argmax(dim=-1, keepdim=True)
        chosen = torch.gather(candidates, -2, best[..., None].expand(*best.shape[:-1], 1, 4)).squeeze(-2)
        denom = 2.0 * torch.gather(q_abs, -1, best)  # = 4*|q_best| >= 2 for a valid rotation
        quat = chosen / denom  # (w, x, y, z)
        quat = torch.where(quat[..., :1] < 0, -quat, quat)
        w, xyz = quat[..., 0], quat[..., 1:]
        s = torch.linalg.vector_norm(xyz, dim=-1)
        angle = 2.0 * torch.atan2(s, w)
        small = s < 1e-12
        scale = torch.where(small, 2.0 / w.clamp_min(1e-12), angle / torch.where(small, torch.ones_like(s), s))
        out = xyz * scale[..., None]
    return out.to(_result_dtype(rotation.dtype))


def rot6d_to_axis_angle(r: Tensor, layout: Literal["rows", "columns"]) -> Tensor:
    """Rotation-6D ``[..., 6]`` -> axis-angle ``[..., 3]`` (Gram-Schmidt + log map; angle in ``[0, pi]``).

    Returned as float32 (float64 for float64 input); the geometry runs in float64.
    """
    dtype = r.dtype if isinstance(r, torch.Tensor) else torch.float32
    return matrix_to_axis_angle(rot6d_to_matrix(r, layout)).to(_result_dtype(dtype))


def apply_prompt_state_transform(name: str, value: Tensor) -> Tensor:
    """Apply one named prompt-state transform to ``[..., D]`` (float64 result for identity on float64)."""
    if name not in PROMPT_STATE_TRANSFORMS:
        raise ValueError(f"Unknown prompt-state transform {name!r}; known: {sorted(PROMPT_STATE_TRANSFORMS)}")
    if name == "identity":
        return value
    if name in ("quat_to_axis_angle", "quat_xyzw_to_axis_angle"):
        return quat_xyzw_to_axis_angle(value)
    layout = "rows" if name == "rot6d_rows_to_axis_angle" else "columns"
    return rot6d_to_axis_angle(value, layout)


@dataclass
class PromptStateSpec:
    """Which observation keys form the prompt state, and how each is transformed.

    ``transforms`` maps a key to a name in ``PROMPT_STATE_TRANSFORMS``; keys without an entry use
    ``identity``. After construction ``transforms`` lists every key explicitly.
    """

    keys: List[str]
    transforms: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        keys, transforms = _plain(self.keys), _plain(self.transforms)
        if isinstance(keys, (str, bytes)) or not isinstance(keys, Sequence):
            raise TypeError("Prompt-state keys must be a sequence of observation keys")
        keys = list(keys)
        if not keys or any(not isinstance(key, str) or not key for key in keys):
            raise ValueError("Prompt-state keys must be a nonempty list of nonempty strings")
        if len(set(keys)) != len(keys):
            raise ValueError(f"Prompt-state keys must be distinct, got {keys}")
        if transforms is None:
            transforms = {}
        if not isinstance(transforms, Mapping):
            raise TypeError("Prompt-state transforms must be a mapping {key: transform name}")
        unknown = sorted(set(transforms) - set(keys))
        if unknown:
            raise ValueError(f"Prompt-state transforms name keys that are not prompt-state keys: {unknown}")
        resolved = {}
        for key in keys:
            name = transforms.get(key, "identity")
            if not isinstance(name, str) or name not in PROMPT_STATE_TRANSFORMS:
                raise ValueError(f"Unknown prompt-state transform {name!r} for {key!r}; "
                                 f"known: {sorted(PROMPT_STATE_TRANSFORMS)}")
            resolved[key] = name
        self.keys, self.transforms = keys, resolved

    @classmethod
    def from_config(cls, config) -> "PromptStateSpec":
        """Build from ``{'keys': [...], 'transforms': {...}}`` (dict or OmegaConf) or pass a spec through."""
        if isinstance(config, PromptStateSpec):
            return cls(list(config.keys), dict(config.transforms))
        config = _plain(config)
        if not isinstance(config, Mapping):
            raise TypeError("prompt_state must be a mapping with 'keys' and optional 'transforms'")
        extra = sorted(set(config) - {"keys", "transforms"})
        if extra:
            raise ValueError(f"Unexpected prompt_state fields: {extra}")
        if "keys" not in config:
            raise ValueError("prompt_state must define 'keys'")
        return cls(config["keys"], config.get("transforms") or {})

    def to_dict(self) -> dict:
        return {"keys": list(self.keys), "transforms": dict(self.transforms)}

    def part_dim(self, key: str, input_dim: int) -> int:
        expected, output = PROMPT_STATE_TRANSFORMS[self.transforms[key]]
        if expected is not None and int(input_dim) != expected:
            raise ValueError(f"Prompt-state key {key!r} ({self.transforms[key]}) needs {expected} values, "
                             f"got {input_dim}")
        return int(input_dim) if output is None else output

    def output_dim(self, key_shapes: Mapping[str, Sequence[int]]) -> int:
        """Prompt-state width for per-frame shapes ``{key: (D,)}`` (e.g. from ``shape_meta['obs']``)."""
        total = 0
        for key in self.keys:
            if key not in key_shapes:
                raise KeyError(f"Prompt-state key {key!r} has no shape")
            shape = tuple(key_shapes[key])
            if len(shape) != 1:
                raise ValueError(f"Prompt-state key {key!r} must be a vector per frame, got shape {shape}")
            total += self.part_dim(key, shape[0])
        return total

    def extract(self, obs: Mapping[str, Tensor]) -> Tensor:
        """``[B, D]`` fp32 prompt state from the LAST frame of each key (``[B, To, Dk]`` or ``[B, Dk]``)."""
        if not isinstance(obs, Mapping):
            raise TypeError("Observations must be a mapping of tensors")
        frames, batch, device = [], None, None
        for key in self.keys:
            if key not in obs:
                raise KeyError(f"Prompt-state key {key!r} missing from observations")
            value = obs[key]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Prompt-state observation {key!r} must be a torch.Tensor")
            if value.dtype == torch.bool or value.is_complex():
                raise TypeError(f"Prompt-state observation {key!r} must be numeric, got {value.dtype}")
            if value.ndim == 3:
                if value.shape[1] < 1:
                    raise ValueError(f"Prompt-state observation {key!r} has an empty time axis")
                value = value[:, -1]
            elif value.ndim != 2:
                raise ValueError(f"Prompt-state observation {key!r} must have shape [B,To,D] or [B,D], "
                                 f"got {tuple(value.shape)}")
            if batch is None:
                batch, device = value.shape[0], value.device
            elif value.shape[0] != batch:
                raise ValueError(f"Prompt-state observation {key!r} has batch {value.shape[0]}, expected {batch}")
            elif value.device != device:
                raise ValueError(f"Prompt-state observation {key!r} is on {value.device}, expected {device}")
            self.part_dim(key, value.shape[-1])
            frames.append(value)
        with torch.autocast(device_type=device.type, enabled=False):
            frames = [value.double() for value in frames]
            if not torch.stack([torch.isfinite(value).all() for value in frames]).all():
                raise ValueError("Prompt-state observations must be finite")
            parts = [apply_prompt_state_transform(self.transforms[key], value)
                     for key, value in zip(self.keys, frames)]
            return torch.cat([part.double() for part in parts], dim=-1).float()
