import numpy as np
import torch
from typing import Any, Dict, List, Optional

from oat.dataset.zarr_dataset import is_numeric_dtype
from oat.dataset.zarr_dataset_with_past import ZarrDatasetWithPastAction


class ZarrDatasetWithPrevWindow(ZarrDatasetWithPastAction):
    """
    ZarrDatasetWithPastAction plus the PREVIOUS rollout window, so that a
    policy can regenerate its own `past_action` during training instead of
    reading the ground truth.

    Why the previous window is needed
    ---------------------------------
    At rollout, `past_action` for chunk k is not "some actions" — it is
    literally what the policy produced when it ran on chunk k-1:

        _past_buffer = action_pred[:, n_exec - past_n : n_exec]
                                                (past2next.py:337-343)

    Reproducing that offline means re-running the policy on the observation
    window one execution-stride back.  This dataset returns that window
    (`prev_obs`) together with the inputs that call itself needs
    (`prev_past_action`), which is exactly one level of unrolling.

    Emitted keys
    ------------
        obs, action, past_action   — identical real-world frames to the parent
        prev_obs                   — obs window shifted back by n_exec_steps
        prev_past_action           — past_action for that shifted window

    Time layout (To=n_obs_steps, Ta=n_action_steps, S=n_exec_steps, t = chunk start)

        prev_past_action : [t-S-past_n, ..., t-S-1]
        prev_obs         : [t-S-To+1, ..., t-S]
        past_action      : [t-past_n, ..., t-1]
        obs              : [t-To+1, ..., t]
        action           : [t, ..., t+Ta-1]

    Index arithmetic, mirroring the parent but shifted right by S:

        action_start = S + past_n + max(To - 1, 0)      ( == pad_before )
        obs               = sample[S + past_n : S + past_n + To]
        action            = sample[action_start : action_start + Ta]
        past_action       = sample[action_start - past_n : action_start]
        prev_obs          = sample[past_n : past_n + To]
        prev_past_action  = sample[action_start - S - past_n : action_start - S]

    With To=2, Ta=16, past_n=7, S=8 that is pad_before=16, pad_after=15,
    seq_len=32; obs=sample[15:17], action=sample[16:32],
    past_action=sample[9:16], prev_obs=sample[7:9],
    prev_past_action=sample[1:8].

    Notes
    -----
    * `n_exec_steps` is the policy's `n_action_steps` (actions actually
      executed per chunk, 8 in the past2next config) — NOT the dataset's
      `n_action_steps`, which is the chunk/prediction length `horizon` (16).
      They are different quantities that share a name in the configs.
    * The wider window (pad_before grows by S) costs one extra pair of RGB
      frames per sample.  Boundary padding is unchanged — still the
      SequenceSampler's edge repetition — so this dataset is a pure superset
      of the parent's output.
    """

    def __init__(
        self,
        past_n: int = 7,
        n_exec_steps: int = 8,
        # all ZarrDatasetWithPastAction / ZarrDataset args
        zarr_path: str = "",
        obs_keys: List[str] = [],
        action_key: str = "action",
        n_obs_steps: int = 2,
        n_action_steps: int = 16,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes: Optional[int] = None,
        history_padding: str = "edge",
        return_history_validity: bool = False,
    ):
        # Builds the parent's sampler; widened again below.
        super().__init__(
            past_n=past_n,
            zarr_path=zarr_path,
            obs_keys=obs_keys,
            action_key=action_key,
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            seed=seed,
            val_ratio=val_ratio,
            max_train_episodes=max_train_episodes,
            history_padding=history_padding,
            return_history_validity=return_history_validity,
        )

        assert n_exec_steps >= 1, "n_exec_steps must be >= 1"
        self.n_exec_steps = n_exec_steps

        # ── Widen the window by one execution stride ──────────────────────
        from oat.common.seq_sampler import SequenceSampler

        self.pad_before = max(n_obs_steps - 1, 0) + past_n + n_exec_steps
        self.pad_after = max(n_action_steps - 1, 0)
        self.seq_len = self.pad_before + 1 + self.pad_after

        self.seq_sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.seq_len,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.train_mask,
        )

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _slice_obs(self, sample, start: int) -> Dict[str, Any]:
        """Obs window of length n_obs_steps beginning at sample index `start`."""
        To = self.n_obs_steps
        obs = {}
        for k in self.numeric_obs_keys:
            raw = sample[k][start: start + To]
            obs[k] = raw.astype(np.float32) if raw.dtype.kind == "f" else raw
        for k in self.text_obs_keys:
            obs[k] = sample[k][start]
        return obs

    def _sample_to_data(self, sample):
        To = self.n_obs_steps
        Ta = self.n_action_steps
        past_n = self.past_n
        S = self.n_exec_steps

        action_start = S + past_n + max(To - 1, 0)
        prev_action_start = action_start - S

        acts = sample[self.action_key]

        return {
            "obs": self._slice_obs(sample, S + past_n),
            "action": acts[action_start: action_start + Ta].astype(np.float32),
            "past_action": acts[action_start - past_n: action_start].astype(np.float32),
            "prev_obs": self._slice_obs(sample, past_n),
            "prev_past_action": acts[
                prev_action_start - past_n: prev_action_start
            ].astype(np.float32),
        }

    @staticmethod
    def _obs_to_torch(obs: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for k, v in obs.items():
            if isinstance(v, np.ndarray) and is_numeric_dtype(v):
                out[k] = torch.from_numpy(v)
            elif isinstance(v, bytes):
                out[k] = v.decode("utf-8")
            else:
                out[k] = v
        return out

    # ── Dataset interface ───────────────────────────────────────────────────

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.seq_sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        self._apply_history_metadata(data, idx, prev_stride=self.n_exec_steps)

        result = {
            "obs": self._obs_to_torch(data["obs"]),
            "action": torch.from_numpy(data["action"]),
            "past_action": torch.from_numpy(data["past_action"]),
            "prev_obs": self._obs_to_torch(data["prev_obs"]),
            "prev_past_action": torch.from_numpy(data["prev_past_action"]),
        }
        for key in ("past_action_valid", "prev_past_action_valid",
                    "episode_step", "prev_window_valid"):
            if key in data:
                result[key] = torch.as_tensor(data[key])
        return result
