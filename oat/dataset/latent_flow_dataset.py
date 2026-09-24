"""Add flow evaluation metadata without changing any sampled training values.

The sample ID is the absolute action anchor in the replay buffer, independent
of episode splits, sampler order, batching, or distributed rank. Dataset identity
hashes source location, array schema, episode boundaries and raw action bytes.
This identifies the data source without copying/hashing multi-gigabyte RGB arrays.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from oat.dataset.zarr_dataset_with_prev_window import ZarrDatasetWithPrevWindow
from oat.dataset.zarr_dataset_with_state_history import ZarrDatasetWithStateHistory
from oat.dataset.real_robot_dataset import RealRobotZarrDatasetWithPrevWindow
from oat.dataset.real_robot_state_history import RealRobotZarrDatasetWithStateHistory


def dataset_fingerprint(source, episode_ends, arrays, action_key="action"):
    """Stable SHA256 of source/schema/episodes/actions; independent of view split."""
    schema = {key: {"shape": list(arrays[key].shape), "dtype": str(arrays[key].dtype)}
              for key in sorted(arrays.keys())}
    header = {"version": 1, "source": str(Path(source).expanduser().resolve()),
              "arrays": schema, "action_key": action_key}
    digest = hashlib.sha256(json.dumps(header, sort_keys=True).encode("utf-8"))
    digest.update(np.asarray(episode_ends, dtype="<i8").tobytes())
    action = arrays[action_key]
    for start in range(0, len(action), 65536):
        digest.update(np.ascontiguousarray(action[start:start + 65536]).tobytes())
    return "sha256:" + digest.hexdigest()


class FutureActionValidityMixin:
    """Metadata-only extension to the existing previous-window action layout."""

    def __init__(self, *args, **kwargs):
        source = kwargs.get("zarr_path")
        if source is None:
            raise ValueError("Flow datasets require an explicit zarr_path keyword")
        super().__init__(*args, **kwargs)
        expected_anchor = self.n_exec_steps + self.past_n + max(self.n_obs_steps - 1, 0)
        if self.pad_before != expected_anchor or self.seq_len != self.pad_before + self.n_action_steps:
            raise ValueError("Future validity requires the documented previous-window action layout")
        self.dataset_identity = dataset_fingerprint(
            source, self.replay_buffer.episode_ends,
            self.replay_buffer.root["data"], self.action_key)
        self.sample_id_definition = "absolute_action_anchor_v1"

    def __getitem__(self, idx):
        result = super().__getitem__(idx)
        buffer_start, _, sample_start, sample_end = self.seq_sampler.indices[idx]
        positions = self.pad_before + np.arange(self.n_action_steps)
        result["future_action_valid"] = torch.as_tensor(
            (positions >= sample_start) & (positions < sample_end), dtype=torch.bool)
        result["sample_id"] = torch.as_tensor(
            int(buffer_start + self.pad_before - sample_start), dtype=torch.int64)
        return result


class LatentFlowZarrDatasetWithPrevWindow(FutureActionValidityMixin, ZarrDatasetWithPrevWindow):
    """LIBERO flow dataset, retaining the original normalization and split."""


class LatentFlowZarrDatasetWithStateHistory(FutureActionValidityMixin, ZarrDatasetWithStateHistory):
    """LIBERO flow dataset with measured quaternion state history."""


class LatentFlowRealRobotZarrDatasetWithPrevWindow(FutureActionValidityMixin, RealRobotZarrDatasetWithPrevWindow):
    """Real-robot flow dataset with training-only state normalization."""


class LatentFlowRealRobotZarrDatasetWithStateHistory(FutureActionValidityMixin, RealRobotZarrDatasetWithStateHistory):
    """Real-robot flow dataset with measured rotation-6D state history."""
