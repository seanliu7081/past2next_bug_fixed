"""Collect low-dimensional states at every executed simulator control step.

RGB observations and their vector transport retain the original two-frame
contract. A separate RPC retrieves only the small state-history snapshots.
"""

from collections import deque
from numbers import Integral

import dill
import gymnasium
import numpy as np

from oat.env_runner.executed_action_runner import LiberoExecutedPastRunner


DEFAULT_STATE_HISTORY_KEYS = (
    "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
)
STATE_HISTORY_PREFIX = "state_history__"
STATE_HISTORY_VALID = "state_history_valid"


def _validate_history_settings(state_history_steps, state_history_keys):
    if (isinstance(state_history_steps, bool)
            or not isinstance(state_history_steps, Integral)
            or state_history_steps < 1):
        raise ValueError("state_history_steps must be a positive integer")
    if isinstance(state_history_keys, (str, bytes)):
        raise ValueError("state_history_keys must be a sequence of distinct keys")
    try:
        keys = tuple(state_history_keys)
    except TypeError as exc:
        raise ValueError("state_history_keys must be a sequence of distinct keys") from exc
    if (not keys or any(not isinstance(key, str) or not key for key in keys)
            or len(set(keys)) != len(keys)):
        raise ValueError("state_history_keys must contain distinct nonempty strings")
    return int(state_history_steps), keys


class LowDimStateHistoryWrapper(gymnasium.Wrapper):
    """Keep reset state and each scalar-step state, without retaining RGB."""

    def __init__(self, env, state_history_steps=8,
                 state_history_keys=DEFAULT_STATE_HISTORY_KEYS):
        steps, keys = _validate_history_settings(state_history_steps, state_history_keys)
        super().__init__(env)
        if not isinstance(env.observation_space, gymnasium.spaces.Dict):
            raise ValueError("State history requires dictionary observations")
        for key in keys:
            space = env.observation_space.spaces.get(key)
            if (not isinstance(space, gymnasium.spaces.Box)
                    or len(space.shape) != 1 or space.shape[0] < 1
                    or np.dtype(space.dtype).kind not in "fiu"):
                raise ValueError(f"State history key {key!r} must be a numeric vector")
        self.state_history_steps = steps
        self.state_history_keys = keys
        self._state_history = {key: deque(maxlen=steps) for key in keys}

    def __getattr__(self, name):
        # Original LIBERO initializers configure or replace outer.env.env.
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)

    def _record(self, observation):
        values = {}
        for key in self.state_history_keys:
            value = np.asarray(observation[key])
            expected_shape = self.observation_space[key].shape
            if value.shape != expected_shape or value.dtype.kind not in "fiu":
                raise ValueError(f"State history key {key!r} must have shape {expected_shape}")
            values[key] = value.copy()
        for key, value in values.items():
            self._state_history[key].append(value)

    def reset(self, **kwargs):
        for history in self._state_history.values():
            history.clear()
        result = self.env.reset(**kwargs)
        self._record(result[0])
        return result

    def step(self, action):
        result = self.env.step(action)
        self._record(result[0])
        return result

    def snapshot(self):
        count = len(self._state_history[self.state_history_keys[0]])
        if count == 0:
            raise RuntimeError("Reset the state-history collector before taking a snapshot")
        result = {}
        for key, history in self._state_history.items():
            values = np.stack(tuple(history))
            padded = np.zeros((self.state_history_steps,) + values.shape[1:], dtype=values.dtype)
            padded[-count:] = values
            result[STATE_HISTORY_PREFIX + key] = padded
        valid = np.zeros(self.state_history_steps, dtype=np.bool_)
        valid[-count:] = True
        result[STATE_HISTORY_VALID] = valid
        return result


class StateHistoryInitializer:
    """Run the saved initializer, then restore the collector after task changes."""

    def __init__(self, initializer_dill, state_history_steps=8,
                 state_history_keys=DEFAULT_STATE_HISTORY_KEYS):
        self.state_history_steps, self.state_history_keys = _validate_history_settings(
            state_history_steps, state_history_keys,
        )
        self.initializer_dill = initializer_dill

    def __call__(self, outer):
        result = dill.loads(self.initializer_dill)(outer)
        base = outer.env.env
        if isinstance(base, LowDimStateHistoryWrapper):
            if (base.state_history_steps == self.state_history_steps
                    and base.state_history_keys == self.state_history_keys):
                return result
            base = base.env
        outer.env.env = LowDimStateHistoryWrapper(
            base, state_history_steps=self.state_history_steps,
            state_history_keys=self.state_history_keys,
        )
        return result


def _snapshot_state_history(outer):
    collector = outer.env.env
    if not isinstance(collector, LowDimStateHistoryWrapper):
        raise RuntimeError("Install the state-history collector before resetting the runner")
    return collector.snapshot()


class StateHistoryVectorEnv:
    """Enrich completed reset/step results without changing vector spaces."""

    def __init__(self, env):
        if getattr(env, "autoreset", None) is not False:
            raise ValueError("Continuous state history requires autoreset=False")
        self._env = env
        self._snapshot_dill = dill.dumps(_snapshot_state_history)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _enrich(self, observation):
        snapshots = self._env.call("run_dill_function", self._snapshot_dill)
        if len(snapshots) != self._env.num_envs or not snapshots:
            raise ValueError("Expected one state-history snapshot per environment")
        keys = set(snapshots[0])
        if any(set(snapshot) != keys for snapshot in snapshots):
            raise ValueError("State-history keys differ between environments")
        if keys.intersection(observation):
            raise ValueError("State-history metadata collides with an observation key")
        result = dict(observation)
        for key in snapshots[0]:
            result[key] = np.stack([snapshot[key] for snapshot in snapshots])
        return result

    def reset(self, *args, **kwargs):
        observation, info = self._env.reset(*args, **kwargs)
        return self._enrich(observation), info

    def step(self, actions):
        observation, reward, terminated, truncated, info = self._env.step(actions)
        return self._enrich(observation), reward, terminated, truncated, info


class LiberoStateHistoryRunner(LiberoExecutedPastRunner):
    """Use continuous measured-state history and confirmed action acknowledgments."""

    def __init__(self, *args, state_history_steps=8,
                 state_history_keys=DEFAULT_STATE_HISTORY_KEYS,
                 protocol="corrected", **kwargs):
        self.state_history_steps, self.state_history_keys = _validate_history_settings(
            state_history_steps, state_history_keys,
        )
        super().__init__(*args, protocol=protocol, **kwargs)
        self.env_init_fn_dills = [
            dill.dumps(StateHistoryInitializer(
                initializer, self.state_history_steps, self.state_history_keys,
            ))
            for initializer in self.env_init_fn_dills
        ]

    def run(self, policy, **kwargs):
        original = self.env
        self.env = StateHistoryVectorEnv(original)
        try:
            return super().run(policy, **kwargs)
        finally:
            self.env = original
