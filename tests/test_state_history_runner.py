"""Continuous states are collected inside action chunks without retaining RGB."""

from functools import partial

import dill
import gymnasium
import numpy as np
import pytest
import torch

from oat.env_runner.executed_action_runner import ExecutedActionVectorEnv
from oat.env_runner.libero_runner import LiberoRunner
from oat.env_runner.state_history_runner import (
    DEFAULT_STATE_HISTORY_KEYS, LiberoStateHistoryRunner,
    LowDimStateHistoryWrapper, StateHistoryInitializer, StateHistoryVectorEnv,
)
from oat.gymnasium_util.async_vector_env import AsyncVectorEnv
from oat.gymnasium_util.multistep_wrapper import MultiStepWrapper
from oat.gymnasium_util.video_recording_wrapper import VideoRecordingWrapper


class StateCounter(gymnasium.Env):
    metadata = {}
    render_mode = None

    def __init__(self, terminal_step=100, offset=0, task_name="task_a"):
        self.terminal_step = terminal_step
        self.offset = offset
        self.task_name = task_name
        self.cur_step = 0
        self.closed = False
        self.configured = None
        self.reset_calls = 0
        self.observation_space = gymnasium.spaces.Dict({
            key: gymnasium.spaces.Box(-np.inf, np.inf, shape=(width,), dtype=np.float32)
            for key, width in zip(DEFAULT_STATE_HISTORY_KEYS, (3, 4, 2))
        } | {"agentview_rgb": gymnasium.spaces.Box(0, 255, (3, 3, 3), np.uint8)})
        self.action_space = gymnasium.spaces.Box(-1, 1, (1,), np.float32)
        self._shared = {key: np.zeros(width, dtype=np.float32)
                        for key, width in zip(DEFAULT_STATE_HISTORY_KEYS, (3, 4, 2))}

    def observe(self):
        for value in self._shared.values():
            value[:] = self.offset + self.cur_step
        self.last_obs = dict(self._shared)
        self.last_obs["agentview_rgb"] = np.full((3, 3, 3), self.cur_step, dtype=np.uint8)
        return self.last_obs

    def reset(self, *, seed=None, options=None):
        self.cur_step = 0
        self.reset_calls += 1
        return self.observe(), {"seed": seed}

    def step(self, action):
        self.cur_step += 1
        done = self.cur_step >= self.terminal_step
        return self.observe(), float(done), done, False, {"counter": self.cur_step}

    def configure_episode(self, seed, init_state_id):
        self.configured = (seed, init_state_id)

    def close(self):
        self.closed = True


class InactiveRecorder:
    def stop(self):
        pass

    def is_ready(self):
        return False


def state_environment(terminal_step=100, offset=0):
    return MultiStepWrapper(
        VideoRecordingWrapper(StateCounter(terminal_step, offset), InactiveRecorder()),
        n_obs_steps=2, n_action_steps=8, max_episode_steps=100,
    )


def initialize_task(outer, task_name="task_a", seed=42, init_state_id=None):
    if outer.env.env.task_name != task_name:
        outer.env.env.close()
        outer.env.env = StateCounter(offset=200, task_name=task_name)
    outer.env.env.configure_episode(seed, init_state_id)
    return "initialized"


def wrapped_initializer(**kwargs):
    return dill.dumps(StateHistoryInitializer(dill.dumps(partial(initialize_task, **kwargs))))


def install_collector(outer, **kwargs):
    return outer.run_dill_function(wrapped_initializer(**kwargs))


def test_collector_preserves_observations_and_copies_only_lowdim_states():
    core = StateCounter(offset=10)
    wrapper = LowDimStateHistoryWrapper(core)
    assert wrapper.observation_space is core.observation_space
    with pytest.raises(RuntimeError, match="Reset"):
        wrapper.snapshot()
    obs, info = wrapper.reset(seed=37)
    assert obs is core.last_obs and info == {"seed": 37}
    assert set(obs) == set(core.observation_space)
    initial = wrapper.snapshot()
    np.testing.assert_array_equal(initial["state_history_valid"], [False] * 7 + [True])
    np.testing.assert_array_equal(initial["state_history__robot0_eef_pos"][:, 0], [0] * 7 + [10])
    for _ in range(8):
        result = wrapper.step(np.zeros(1))
        assert result[0] is core.last_obs
    snapshot = wrapper.snapshot()
    np.testing.assert_array_equal(snapshot["state_history__robot0_eef_pos"][:, 0], np.arange(11, 19))
    assert snapshot["state_history_valid"].all()
    assert set(wrapper._state_history) == set(DEFAULT_STATE_HISTORY_KEYS)
    assert all(value.ndim == 1 for history in wrapper._state_history.values() for value in history)
    snapshot["state_history__robot0_eef_pos"][:] = -999
    core.last_obs["robot0_eef_pos"][:] = -888
    np.testing.assert_array_equal(wrapper.snapshot()["state_history__robot0_eef_pos"][:, 0], np.arange(11, 19))
    assert initial["state_history__robot0_eef_pos"][-1, 0] == 10
    wrapper.reset()
    np.testing.assert_array_equal(wrapper.snapshot()["state_history_valid"], [False] * 7 + [True])


@pytest.mark.parametrize("steps", [1, 16])
def test_configurable_length_and_state_keys(steps):
    wrapper = LowDimStateHistoryWrapper(StateCounter(), state_history_steps=steps,
                                       state_history_keys=("robot0_eef_pos",))
    wrapper.reset()
    for _ in range(20):
        wrapper.step(np.zeros(1))
    snapshot = wrapper.snapshot()
    assert set(snapshot) == {"state_history__robot0_eef_pos", "state_history_valid"}
    np.testing.assert_array_equal(snapshot["state_history__robot0_eef_pos"][:, 0], np.arange(21 - steps, 21))


def test_initializer_survives_task_replacement_and_does_not_reset_early():
    outer = state_environment()
    assert install_collector(outer, seed=7, init_state_id=3) == "initialized"
    collector = outer.env.env
    assert isinstance(collector, LowDimStateHistoryWrapper)
    assert collector.env.reset_calls == 0
    assert collector.configured == (7, 3)
    outer.reset()
    obs, *_ = outer.step(np.zeros((8, 1)))
    assert obs["agentview_rgb"].shape == (2, 3, 3, 3)
    assert collector.snapshot()["state_history_valid"].all()
    install_collector(outer, seed=8)
    assert outer.env.env is collector
    assert collector.configured == (8, None)
    old_core = collector.env
    install_collector(outer, task_name="task_b", seed=19, init_state_id=5)
    replacement = outer.env.env
    assert old_core.closed and replacement is not collector
    assert replacement.task_name == "task_b"
    assert replacement.configured == (19, 5)
    assert replacement.reset_calls == 0
    with pytest.raises(RuntimeError, match="Reset"):
        replacement.snapshot()
    outer.reset()
    values = replacement.snapshot()
    np.testing.assert_array_equal(values["state_history__robot0_eef_pos"][:, 0], [0] * 7 + [200])
    np.testing.assert_array_equal(values["state_history_valid"], [False] * 7 + [True])


class RecordingPolicy:
    device = torch.device("cpu")
    dtype = torch.float32

    def __init__(self, fail_prediction=False):
        self.fail_prediction = fail_prediction
        self.histories = []
        self.acknowledgments = []
        self.resets = 0

    def reset(self):
        self.resets += 1

    def get_policy_name(self):
        return "continuous_state_test"

    def get_observation_ports(self):
        return ["agentview_rgb", "state_history__robot0_eef_pos", "state_history_valid"]

    def predict_action(self, obs, **kwargs):
        assert obs["agentview_rgb"].shape[1] == 2
        self.histories.append({key: value.clone() for key, value in obs.items()})
        if self.fail_prediction and len(self.histories) == 2:
            raise RuntimeError("prediction failed")
        return {"action": torch.zeros(obs["agentview_rgb"].shape[0], 8, 1)}

    def record_executed_actions(self, actions, executed_lengths=None):
        self.acknowledgments.append(executed_lengths.copy())


def test_async_workers_collect_each_control_step_partial_prefix_and_reset():
    vector = AsyncVectorEnv(
        [partial(state_environment, limit, offset) for limit, offset in ((2, 0), (10, 100), (100, 200))],
        shared_memory=False, context="fork", autoreset=False,
    )
    policy = RecordingPolicy()
    state_proxy = StateHistoryVectorEnv(vector)
    proxy = ExecutedActionVectorEnv(state_proxy, policy)
    try:
        vector.call_each("run_dill_function", args_list=[(wrapped_initializer(),)] * 3)
        obs, _ = proxy.reset()
        assert obs["agentview_rgb"].shape == (3, 2, 3, 3, 3)
        assert "state_history_valid" not in vector.single_observation_space.spaces
        np.testing.assert_array_equal(obs["state_history_valid"], [[False] * 7 + [True]] * 3)
        actions = np.zeros((3, 8, 1), dtype=np.float32)
        obs, *_ = proxy.step(actions)
        np.testing.assert_array_equal(obs["state_history__robot0_eef_pos"][:, :, 0], [
            [0, 0, 0, 0, 0, 0, 1, 2], np.arange(101, 109), np.arange(201, 209),
        ])
        np.testing.assert_array_equal(obs["state_history_valid"][0], [False] * 5 + [True] * 3)
        assert obs["state_history_valid"][1:].all()
        obs, *_ = proxy.step(actions)
        np.testing.assert_array_equal(obs["state_history__robot0_eef_pos"][:, :, 0], [
            [0, 0, 0, 0, 0, 0, 1, 2], np.arange(103, 111), np.arange(209, 217),
        ])
        np.testing.assert_array_equal(policy.acknowledgments, [[2, 8, 8], [0, 2, 8]])
        assert vector.get_attr("cur_step") == (2, 10, 16)
        obs, _ = proxy.reset()
        np.testing.assert_array_equal(obs["state_history_valid"], [[False] * 7 + [True]] * 3)
        proxy.step(actions)
        np.testing.assert_array_equal(policy.acknowledgments[-1], [2, 8, 8])
    finally:
        proxy.close()


@pytest.mark.parametrize("fail_prediction", [False, True])
def test_original_runner_loop_sees_history_and_restores_wrappers(fail_prediction):
    vector = AsyncVectorEnv([partial(state_environment, limit) for limit in (2, 10)],
                            shared_memory=False, context="fork", autoreset=False)
    runner = LiberoStateHistoryRunner.__new__(LiberoStateHistoryRunner)
    runner.env = vector
    runner.env_fns = [None, None]
    runner.env_init_fn_dills = [wrapped_initializer(), wrapped_initializer()]
    runner.env_seeds = [1000, 1001]
    runner.env_task_names = ["task_a", "task_a"]
    runner.task_name = "task_a"
    runner.max_episode_steps = 16
    runner.n_action_steps = 8
    runner.tqdm_interval_sec = 1000
    runner.protocol = "corrected"
    runner.episode_schedule = [{"episode_index": 0}, {"episode_index": 1}]
    runner.episode_records_path = None
    policy = RecordingPolicy(fail_prediction=fail_prediction)
    try:
        if fail_prediction:
            with pytest.raises(RuntimeError, match="prediction failed"):
                runner.run(policy, temperature=0)
        else:
            assert runner.run(policy, temperature=0)["mean_success_rate"] == 1
        assert runner.env is vector
        assert len(policy.histories) == 2
        np.testing.assert_array_equal(policy.histories[1]["state_history__robot0_eef_pos"][1, :, 0], np.arange(1, 9))
        np.testing.assert_array_equal(policy.acknowledgments[0], [2, 8])
    finally:
        vector.close()


@pytest.mark.parametrize("kwargs", [
    {"state_history_steps": 0}, {"state_history_steps": True},
    {"state_history_steps": 2.5}, {"state_history_keys": "robot0_eef_pos"},
    {"state_history_keys": []}, {"state_history_keys": ["x", "x"]},
    {"state_history_keys": [""]}, {"state_history_keys": [None]},
    {"protocol": "legacy"},
])
def test_runner_rejects_invalid_settings_before_allocating_environments(monkeypatch, kwargs):
    def fail_initialization(*args, **kw):
        raise AssertionError("Invalid settings reached environment initialization")
    monkeypatch.setattr(LiberoRunner, "__init__", fail_initialization)
    with pytest.raises(ValueError):
        LiberoStateHistoryRunner("unused", **kwargs)


@pytest.mark.parametrize("key", ["missing", "agentview_rgb"])
def test_collector_rejects_missing_or_image_keys(key):
    with pytest.raises(ValueError, match="numeric vector"):
        LowDimStateHistoryWrapper(StateCounter(), state_history_keys=[key])


@pytest.mark.parametrize("protocol", ["corrected", "official"])
def test_runner_constructor_preserves_protocol_and_wraps_saved_initializers(monkeypatch, protocol):
    original = dill.dumps(initialize_task)
    captured = {}
    def initialize(self, *args, **kwargs):
        captured.update(kwargs)
        self.env_init_fn_dills = [original]
    monkeypatch.setattr(LiberoRunner, "__init__", initialize)
    runner = LiberoStateHistoryRunner("unused", protocol=protocol, state_history_steps=16)
    assert captured == {"protocol": protocol}
    outer = state_environment()
    outer.run_dill_function(runner.env_init_fn_dills[0])
    assert outer.env.env.state_history_steps == 16
    assert outer.env.env.state_history_keys == DEFAULT_STATE_HISTORY_KEYS


@pytest.mark.parametrize("autoreset", [True, None, 0])
def test_proxy_requires_disabled_autoreset(autoreset):
    class Vector:
        pass
    env = Vector()
    env.autoreset = autoreset
    with pytest.raises(ValueError, match="autoreset=False"):
        StateHistoryVectorEnv(env)
