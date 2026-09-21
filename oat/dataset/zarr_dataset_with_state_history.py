"""Causal state windows aligned with demonstrated past commands."""

from numbers import Integral

import numpy as np

from oat.dataset.zarr_dataset_with_prev_window import ZarrDatasetWithPrevWindow


class ZarrDatasetWithStateHistory(ZarrDatasetWithPrevWindow):
    """Add raw state history to both current and previous observations.

    At action start ``t``, ``past_n`` commands occupy ``[t-past_n, t)``
    and ``state_history_steps = past_n + 1`` states occupy ``[t-past_n, t]``.
    Each selected key adds ``state_history__<key>`` to ``obs`` and
    ``prev_obs``, accompanied by a boolean ``state_history_valid`` mask.
    Unavailable states are zero; the first real state is valid even when no
    action has executed. Existing observation windows, images and targets keep
    the parent's layout. Quaternion conversion belongs to the model.

    Normalization uses the parent's original replay fields and training split;
    overlapping/padded history windows never introduce additional fitting rows.
    """

    def __init__(
        self,
        state_history_steps=8,
        state_history_keys=("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"),
        past_n=7,
        history_padding="zero",
        return_history_validity=True,
        **kwargs,
    ):
        for name, value in (("state_history_steps", state_history_steps), ("past_n", past_n)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if state_history_steps != past_n + 1:
            raise ValueError("state_history_steps must equal past_n + 1")
        if history_padding != "zero" or return_history_validity is not True:
            raise ValueError("State history requires history_padding='zero' and return_history_validity=True")
        for name, default in (("n_obs_steps", 2), ("n_action_steps", 16), ("n_exec_steps", 8)):
            value = kwargs.get(name, default)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(state_history_keys, (str, bytes)):
            raise ValueError("state_history_keys must be a sequence of distinct observation keys")
        try:
            keys = tuple(state_history_keys)
        except TypeError as exc:
            raise ValueError("state_history_keys must be a sequence of observation keys") from exc
        if not keys or any(not isinstance(key, str) or not key for key in keys):
            raise ValueError("state_history_keys must contain nonempty strings")
        if len(set(keys)) != len(keys):
            raise ValueError("state_history_keys must be distinct")
        obs_keys = list(kwargs.get("obs_keys", ()))
        if any(not isinstance(key, str) or not key for key in obs_keys):
            raise ValueError("obs_keys must contain nonempty strings")
        if any(key == "state_history_valid" or key.startswith("state_history__") for key in obs_keys):
            raise ValueError("obs_keys cannot use reserved state-history output names")
        missing = set(keys) - set(obs_keys)
        if missing:
            raise ValueError(f"state_history_keys must be included in obs_keys: {sorted(missing)}")

        self.state_history_steps = int(state_history_steps)
        self.state_history_keys = keys
        super().__init__(
            past_n=int(past_n), history_padding=history_padding,
            return_history_validity=return_history_validity, **kwargs,
        )
        for key in keys:
            values = self.replay_buffer[key]
            if values.dtype.kind not in "iuf" or values.ndim != 2 or values.shape[1] < 1:
                raise ValueError(f"State history key {key!r} must contain nonempty numeric state vectors")
            expected = 4 if key.endswith("_quat") else 3 if key.endswith("_eef_pos") else None
            if expected is not None and values.shape[1] != expected:
                raise ValueError(f"State history key {key!r} must have shape ({expected},)")

    def _sample_to_data(self, sample):
        data = super()._sample_to_data(sample)
        for name, end in (("obs", self.pad_before),
                          ("prev_obs", self.pad_before - self.n_exec_steps)):
            start = end - self.state_history_steps + 1
            for key in self.state_history_keys:
                # Independent storage prevents masking a view of ordinary obs/replay.
                data[name]["state_history__" + key] = sample[key][start:end + 1].astype(np.float32)
        return data

    def _apply_history_metadata(self, data, idx, prev_stride=None):
        super()._apply_history_metadata(data, idx, prev_stride=prev_stride)
        step = int(data["episode_step"])
        for name, end in (("obs", step), ("prev_obs", step - self.n_exec_steps)):
            positions = np.arange(end - self.state_history_steps + 1, end + 1)
            valid = positions >= 0
            data[name]["state_history_valid"] = valid
            for key in self.state_history_keys:
                data[name]["state_history__" + key][~valid] = 0
