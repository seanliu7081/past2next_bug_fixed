"""Causal action windows and training-only predictive residual statistics."""
import copy
import operator

import numpy as np
import torch
import zarr

from oat.common.seq_sampler import downsample_mask, get_val_mask
from oat.dataset.base_dataset import BaseDataset


class PRTDataset(BaseDataset):
    """One sample starts at t and contains commands [t-K,t) and [t,t+H).

    History before the episode is zero. Every future command is recorded, so a
    short episode or a suffix with fewer than H actions contributes no windows.
    Only commands and episode boundaries are read from the demonstration store.
    """

    def __init__(
        self, zarr_path, past_n=7, horizon=16, action_key="action",
        seed=42, val_ratio=0.1, max_train_episodes=None,
    ):
        self.past_n = operator.index(past_n)
        self.horizon = operator.index(horizon)
        if self.past_n < 1 or self.horizon < 1:
            raise ValueError("past_n and horizon must be positive")
        if not 0 <= val_ratio < 1:
            raise ValueError("val_ratio must be in [0, 1)")
        if max_train_episodes is not None and operator.index(max_train_episodes) < 1:
            raise ValueError("max_train_episodes must be positive")

        root = zarr.open_group(str(zarr_path), mode="r")
        self.episode_ends = root["meta/episode_ends"][:]
        self.actions = root[f"data/{action_key}"][:].astype(np.float32, copy=False)
        if self.actions.ndim != 2 or self.actions.shape[1] != 7:
            raise ValueError("PRT requires seven command channels (xyz, rotation, gripper)")
        if not np.isfinite(self.actions).all():
            raise ValueError("PRT commands must be finite")
        if (self.episode_ends.ndim != 1 or self.episode_ends.size == 0
                or not np.issubdtype(self.episode_ends.dtype, np.integer)):
            raise ValueError("episode_ends must be a nonempty vector of integer boundaries")
        self.episode_starts = np.r_[0, self.episode_ends[:-1]].astype(np.int64)
        lengths = self.episode_ends.astype(np.int64) - self.episode_starts
        if np.any(lengths <= 0) or self.episode_ends[-1] != len(self.actions):
            raise ValueError("episode_ends must increase and cover the action array")

        self.val_mask = get_val_mask(len(lengths), val_ratio, seed)
        self.train_mask = downsample_mask(~self.val_mask, max_train_episodes, seed)
        counts = np.maximum(lengths - self.horizon + 1, 0)
        self._train_offsets = np.r_[0, np.cumsum(counts * self.train_mask)]
        self._val_offsets = np.r_[0, np.cumsum(counts * self.val_mask)]
        self._offsets = self._train_offsets

    def get_validation_dataset(self):
        view = copy.copy(self)
        view._offsets = self._val_offsets
        return view

    def __len__(self):
        return int(self._offsets[-1])

    def __getitem__(self, idx):
        idx = range(len(self))[operator.index(idx)]
        ep = np.searchsorted(self._offsets[1:], idx, side="right")
        start = self.episode_starts[ep]
        t = start + idx - self._offsets[ep]
        n = min(t - start, self.past_n)
        past = np.zeros((self.past_n, 7), dtype=np.float32)
        past[self.past_n - n:] = self.actions[t - n:t]
        return {
            "action": torch.from_numpy(self.actions[t:t + self.horizon].copy()),
            "past_action": torch.from_numpy(past),
            "past_action_valid": torch.from_numpy(np.arange(self.past_n) >= self.past_n - n),
        }

    def _training_windows(self, batch_size):
        """Stream batches of continuous-arm windows, even on a validation view."""
        past_steps = np.arange(-self.past_n, 0)
        future_steps = np.arange(self.horizon)
        for lo in range(0, self._train_offsets[-1], batch_size):
            ids = np.arange(lo, min(lo + batch_size, self._train_offsets[-1]))
            ep = np.searchsorted(self._train_offsets[1:], ids, side="right")
            starts = self.episode_starts[ep]
            t = starts + ids - self._train_offsets[ep]
            past_idx = t[:, None] + past_steps
            past = self.actions[np.maximum(past_idx, starts[:, None]), :6].astype(np.float64)
            past[past_idx < starts[:, None]] = 0
            future = self.actions[t[:, None] + future_steps, :6].astype(np.float64)
            yield past.reshape(-1, self.past_n, 2, 3), future.reshape(-1, self.horizon, 2, 3)

    def fit_prior(self, ridge=1e-3, sigma_floor=1e-6, batch_size=1024):
        """Return W[2,H,K] and sigma[H,2] in raw command units as CPU float32.

        Each triad shares temporal coefficients. Ridge is relative to the
        average Gram diagonal within that triad. Sigma is the square root of
        the average of its three residual coordinate variances, with a floor.
        """
        batch_size = operator.index(batch_size)
        if batch_size < 1 or not np.isfinite(ridge) or ridge < 0:
            raise ValueError("batch_size must be positive and ridge finite and nonnegative")
        if not np.isfinite(sigma_floor) or sigma_floor <= 0:
            raise ValueError("sigma_floor must be finite and positive")
        if self._train_offsets[-1] == 0:
            raise ValueError("No training episode has a complete future action window")

        gram = np.zeros((2, self.past_n, self.past_n), dtype=np.float64)
        cross = np.zeros((2, self.past_n, self.horizon), dtype=np.float64)
        for past, future in self._training_windows(batch_size):
            for group in range(2):
                x = past[:, :, group, :].transpose(0, 2, 1).reshape(-1, self.past_n)
                y = future[:, :, group, :].transpose(0, 2, 1).reshape(-1, self.horizon)
                gram[group] += x.T @ x
                cross[group] += x.T @ y
        weights = np.empty((2, self.horizon, self.past_n), dtype=np.float64)
        for group in range(2):
            penalty = ridge * np.trace(gram[group]) / self.past_n
            system = gram[group] + penalty * np.eye(self.past_n)
            weights[group] = np.linalg.lstsq(system, cross[group], rcond=None)[0].T

        # Merge batch moments around their own means to avoid subtractive
        # cancellation, e.g. when a command or a coordinate is almost constant.
        count = 0
        mean = np.zeros((self.horizon, 2, 3), dtype=np.float64)
        m2 = np.zeros_like(mean)
        for past, future in self._training_windows(batch_size):
            residual = future - np.einsum("bkgc,ghk->bhgc", past, weights)
            n = len(residual)
            batch_mean = residual.mean(axis=0)
            delta = batch_mean - mean
            m2 += np.square(residual - batch_mean).sum(axis=0) + delta**2 * (count * n / (count + n))
            mean += delta * (n / (count + n))
            count += n
        sigma = np.maximum(np.sqrt(m2.mean(axis=-1) / count), sigma_floor)
        return torch.from_numpy(weights.astype(np.float32)), torch.from_numpy(sigma.astype(np.float32))
