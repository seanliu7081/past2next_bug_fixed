"""P2N-VLA datasets: existing previous-window / state-history sampling plus prompt-state statistics.

Sampling, alignment, the episode split and the action/state/RGB normalizer fields are inherited unchanged
from ``RealRobotZarrDatasetWithPrevWindow`` / ``RealRobotZarrDatasetWithStateHistory`` (which keep RGB as
uint8 and give it a fixed byte-range normalizer instead of fitting ~27 GB of float frames).

The only addition is the ``'prompt_state'`` normalizer field for the pi05 discrete state in the prompt: the
prompt-state vector (``PromptStateSpec``: per-key transforms, e.g. quaternion -> axis-angle) of every TRAINING
frame (``normalization_train_mask``; never validation frames, also when called on a validation view) gives
per-dimension q01/q99 (``numpy.quantile``, linear interpolation), and the openpi quantile map
``x -> 2 (x - q01) / (q99 - q01 + 1e-6) - 1`` is stored as a linear normalizer with
``scale = 2 / (q99 - q01 + 1e-6)`` and ``offset = -1 - q01 * scale``. The field is unclipped; the policy
clips to [-1, 1] when it discretizes the state (``paligemma_prompt.discretize_state``).
"""
from __future__ import annotations

import warnings
from typing import Dict, Mapping

import numpy as np
import torch

from oat.dataset.real_robot_dataset import RealRobotZarrDatasetWithPrevWindow
from oat.dataset.real_robot_state_history import RealRobotZarrDatasetWithStateHistory
from oat.model.common.normalizer import SingleFieldLinearNormalizer
from oat.model.vla.state_transforms import PromptStateSpec


PROMPT_STATE_FIELD = "prompt_state"
PROMPT_STATE_QUANTILES = (0.01, 0.99)
PROMPT_STATE_RANGE_EPS = 1e-6


class PromptStateStatsMixin:
    """Adds the frozen ``'prompt_state'`` q01/q99 field to the inherited training-split normalizer."""

    def __init__(self, *args, prompt_state: Mapping, **kwargs):
        if args:
            raise TypeError(f"{type(self).__name__} takes keyword arguments only")
        spec = PromptStateSpec.from_config(prompt_state)
        obs_keys = list(kwargs.get("obs_keys", ()))
        missing = [key for key in spec.keys if key not in obs_keys]
        if missing:
            raise ValueError(f"Prompt-state keys must be included in obs_keys: {missing}")
        if PROMPT_STATE_FIELD in obs_keys:
            raise ValueError(f"obs_keys cannot use the reserved normalizer field {PROMPT_STATE_FIELD!r}")
        super().__init__(**kwargs)
        shapes = {}
        for key in spec.keys:
            values = self.replay_buffer[key]
            if values.dtype.kind not in "iuf" or values.ndim != 2 or values.shape[1] < 1:
                raise ValueError(f"Prompt-state key {key!r} must contain numeric state vectors, got "
                                 f"{values.dtype} {tuple(values.shape)}")
            shapes[key] = (int(values.shape[1]),)
        self.prompt_state_spec = spec
        self.prompt_state_dim = spec.output_dim(shapes)

    def training_frame_mask(self) -> np.ndarray:
        """Replay-frame mask of the fitting split (training episodes), independent of the current view."""
        episode_ends = np.asarray(self.replay_buffer.episode_ends)
        lengths = np.diff(np.concatenate(([0], episode_ends)))
        frame_mask = np.repeat(np.asarray(self.normalization_train_mask, dtype=bool), lengths)
        if not frame_mask.any():
            raise ValueError("Prompt-state statistics need at least one training frame")
        return frame_mask

    def training_prompt_states(self) -> torch.Tensor:
        """``[N_train_frames, D]`` fp32 prompt states, computed exactly as the policy computes them online.

        Float replay values are cast to float32 first, as ``ZarrDatasetWithPrevWindow`` emits them.
        """
        frame_mask = self.training_frame_mask()
        values = {key: torch.from_numpy(np.ascontiguousarray(self.replay_buffer[key][frame_mask],
                                                             dtype=np.float32))
                  for key in self.prompt_state_spec.keys}
        return self.prompt_state_spec.extract(values)

    def prompt_state_stats(self) -> Dict[str, np.ndarray]:
        """Per-dimension statistics of the training-frame prompt states (float64 arrays)."""
        states = self.training_prompt_states().double().numpy()
        q01, q99 = np.quantile(states, PROMPT_STATE_QUANTILES, axis=0)
        return {
            "q01": q01, "q99": q99, "min": states.min(axis=0), "max": states.max(axis=0),
            "mean": states.mean(axis=0), "std": states.std(axis=0),
            "n_frames": np.asarray(states.shape[0]),
        }

    def prompt_state_provenance(self) -> dict:
        """JSON-friendly record of how the ``'prompt_state'`` field is fitted (for preflight/metadata)."""
        return {
            "field": PROMPT_STATE_FIELD, "spec": self.prompt_state_spec.to_dict(), "dim": self.prompt_state_dim,
            "quantiles": list(PROMPT_STATE_QUANTILES), "quantile_method": "numpy.quantile(linear)",
            "range_eps": PROMPT_STATE_RANGE_EPS, "source": "training_replay_frames",
            "train_episodes": int(np.count_nonzero(self.normalization_train_mask)),
            "train_frames": int(self.training_frame_mask().sum()),
        }

    def get_normalizer(self, mode="limits", **kwargs):
        normalizer = super().get_normalizer(mode=mode, **kwargs)
        if PROMPT_STATE_FIELD in normalizer.params_dict:
            raise ValueError(f"Inherited normalizer already defines {PROMPT_STATE_FIELD!r}")
        stats = self.prompt_state_stats()
        span = stats["q99"] - stats["q01"]
        degenerate = np.flatnonzero(span < 1e-4)
        if degenerate.size:
            warnings.warn(f"Prompt-state dims {degenerate.tolist()} have q99 - q01 < 1e-4; their bins "
                          "saturate at 0/255", RuntimeWarning)
        scale = 2.0 / (span + PROMPT_STATE_RANGE_EPS)
        offset = -1.0 - stats["q01"] * scale

        def as_tensor(value):
            return torch.from_numpy(np.asarray(value, dtype=np.float32).copy())

        field = SingleFieldLinearNormalizer.create_manual(
            scale=as_tensor(scale), offset=as_tensor(offset),
            input_stats_dict={name: as_tensor(stats[name]) for name in ("min", "max", "mean", "std")})
        field.requires_grad_(False)
        normalizer[PROMPT_STATE_FIELD] = field
        normalizer.requires_grad_(False)
        return normalizer


class VLAZarrDatasetWithPrevWindow(PromptStateStatsMixin, RealRobotZarrDatasetWithPrevWindow):
    """``p2n_vla`` / ``pi05_ki_flow`` data: previous-window samples plus prompt-state statistics."""


class VLAZarrDatasetWithStateHistory(PromptStateStatsMixin, RealRobotZarrDatasetWithStateHistory):
    """``p2n_vla_state_gate`` data: measured state history plus prompt-state statistics."""
