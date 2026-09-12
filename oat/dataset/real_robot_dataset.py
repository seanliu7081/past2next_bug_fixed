"""Real-robot sampling with training-only state and action normalization.

Both stages retain the original episode split and sample alignment. Images stay
uint8 in the replay buffer and batches; their normalizer uses fixed byte-range
endpoints instead of materializing the full image dataset as float32.
"""

import numpy as np

from oat.dataset.zarr_dataset import ZarrDataset
from oat.dataset.zarr_dataset_with_prev_window import ZarrDatasetWithPrevWindow
from oat.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer


class _RealRobotNormalization:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # get_validation_dataset() shallow-copies the dataset and replaces its
        # train_mask with the validation mask. Keep the fitting split separately
        # so calling get_normalizer() on either view never fits the holdout.
        self.normalization_train_mask = self.train_mask.copy()

    def get_normalizer(self, mode="limits", **kwargs):
        """Fit actions/states on training frames and map RGB bytes to [-1, 1].

        No physical-unit conversion is performed. In particular, gripper state
        remains measured width in millimeters while the action's gripper channel
        remains an absolute command (0=open, 1=closed).
        """
        episode_ends = np.asarray(self.replay_buffer.episode_ends)
        lengths = np.diff(np.concatenate(([0], episode_ends)))
        frame_mask = np.repeat(self.normalization_train_mask, lengths)
        if not frame_mask.any():
            raise ValueError("Real-robot normalization needs at least one training frame")

        data = {"action": self.replay_buffer[self.action_key][frame_mask]}
        rgb_keys = []
        for key in self.numeric_obs_keys:
            array = self.replay_buffer[key]
            if len(array.shape) == 4 and array.shape[-1] == 3:
                if array.dtype != np.uint8:
                    raise ValueError(f"RGB observation {key!r} must contain uint8 pixels")
                rgb_keys.append(key)
            else:
                data[key] = array[frame_mask]

        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        for key in rgb_keys:
            # These are reference byte-range endpoints, not image statistics.
            # Match the encoder's expected [-1, 1] inputs without fitting RGB.
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                np.array([[0, 0, 0], [255, 255, 255]], dtype=np.float32),
                mode="limits",
            )
        return normalizer


class RealRobotZarrDataset(_RealRobotNormalization, ZarrDataset):
    """Tokenizer dataset with the existing Zarr sampler and a clean holdout."""


class RealRobotZarrDatasetWithPrevWindow(
    _RealRobotNormalization, ZarrDatasetWithPrevWindow
):
    """Past2Next dataset with unchanged current/previous-window alignment."""
