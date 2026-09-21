"""Acknowledge executed command prefixes without changing existing rollout loops.

The simulator's ``cur_step`` counts calls to its action-step API, not actuator
motion. Automatic resets are unsupported because they can replace an episode
without executing the submitted chunk. This proxy serves synchronous runners.
"""

import numpy as np
import torch

from oat.env_runner.libero_runner import LiberoRunner
from oat.env_runner.robocasa_multitask_runner import RoboCasaMultiTaskRunner


class ExecutedActionVectorEnv:
    """Forward vector operations and report confirmed prefixes after each step."""

    def __init__(self, env, policy):
        if getattr(env, "autoreset", None) is not False:
            raise ValueError("Executed-action history requires autoreset=False")
        if not callable(getattr(policy, "record_executed_actions", None)):
            raise TypeError("Policy must implement record_executed_actions")
        self._env = env
        self._policy = policy
        self._counts = None

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _read_counts(self):
        counts = np.asarray(self._env.get_attr("cur_step"))
        if counts.shape != (self._env.num_envs,) or counts.dtype.kind not in "iu":
            raise ValueError("cur_step must contain one integer per environment")
        if np.any(counts < 0) or np.any(counts > np.iinfo(np.int64).max):
            raise ValueError("cur_step counts must be nonnegative int64 values")
        return counts.astype(np.int64, copy=True)

    def reset(self, *args, **kwargs):
        self._counts = None
        result = self._env.reset(*args, **kwargs)
        self._counts = self._read_counts()
        return result

    def step(self, actions):
        if self._counts is None:
            raise RuntimeError("Reset the execution-history adapter before stepping")
        if isinstance(actions, torch.Tensor):
            submitted = actions.detach().clone()
        elif isinstance(actions, np.ndarray):
            submitted = actions.copy()
        else:
            raise TypeError("Actions must be a numpy array or torch tensor")
        if (submitted.ndim != 3 or submitted.shape[0] != self._env.num_envs
                or submitted.shape[1] < 1 or submitted.shape[2] < 1):
            raise ValueError("Actions must have shape (num_envs, chunk_length, action_dim)")

        previous = self._counts
        # If stepping or acknowledgement fails, execution is uncertain. Require
        # an explicit reset rather than attributing those steps to a later chunk.
        self._counts = None
        result = self._env.step(actions)
        current = self._read_counts()
        executed = current - previous
        if np.any(executed < 0):
            raise ValueError("Unexpected environment reset during action execution")
        if np.any(executed > submitted.shape[1]):
            raise ValueError("Executed count exceeds the submitted action prefix")
        self._policy.record_executed_actions(submitted, executed_lengths=executed)
        self._counts = current
        return result


class _ExecutedPastRunnerMixin:
    def run(self, policy, **kwargs):
        original = self.env
        adapter = ExecutedActionVectorEnv(original, policy)
        self.env = adapter
        try:
            return super().run(policy, **kwargs)
        finally:
            self.env = original


class LiberoExecutedPastRunner(_ExecutedPastRunnerMixin, LiberoRunner):
    """LIBERO runner with confirmed action history and corrected/official resets."""

    def __init__(self, *args, protocol="corrected", **kwargs):
        if protocol not in ("corrected", "official"):
            raise ValueError("Executed-action LIBERO requires corrected or official protocol")
        super().__init__(*args, protocol=protocol, **kwargs)


class RoboCasaExecutedPastRunner(_ExecutedPastRunnerMixin, RoboCasaMultiTaskRunner):
    """Reuse the corrected RoboCasa loop with executed-action acknowledgement."""
