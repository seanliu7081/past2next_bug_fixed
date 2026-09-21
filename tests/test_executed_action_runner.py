"""Execution acknowledgements use completed simulator steps, including early stops."""

from functools import partial

import gymnasium
import numpy as np
import pytest
import torch

from oat.env_runner.executed_action_runner import (
    ExecutedActionVectorEnv, LiberoExecutedPastRunner, RoboCasaExecutedPastRunner,
)
from oat.env_runner.libero_runner import LiberoRunner
from oat.gymnasium_util.async_vector_env import AsyncVectorEnv
from oat.gymnasium_util.multistep_wrapper import MultiStepWrapper


class RecordingPolicy:
    device = torch.device("cpu")
    dtype = torch.float32

    def __init__(self, events=None, fail_prediction=False):
        self.records = []
        self.events = [] if events is None else events
        self.predictions = 0
        self.fail_prediction = fail_prediction

    def record_executed_actions(self, actions, executed_lengths=None):
        self.events.append("acknowledge")
        self.records.append((actions, executed_lengths.copy()))

    def reset(self):
        self.events.append("policy_reset")

    def get_policy_name(self):
        return "test_executed_history"

    def get_observation_ports(self):
        return ["state"]

    def predict_action(self, obs, **kwargs):
        assert set(obs) == {"state"}
        assert kwargs == {"temperature": 0}
        self.events.append("predict")
        self.predictions += 1
        if self.fail_prediction and self.predictions == 2:
            raise RuntimeError("prediction failed")
        return {"action": torch.full((obs["state"].shape[0], 8, 1), float(self.predictions))}


class FakeVectorEnv:
    autoreset = False

    def __init__(self, terminal_steps=(2, 10, 100), baseline=None):
        self.terminal_steps = np.asarray(terminal_steps, dtype=np.int64)
        self.num_envs = len(terminal_steps)
        self.baseline = np.zeros(self.num_envs, dtype=np.int64) if baseline is None else np.array(baseline)
        self.counts = self.baseline.copy()
        self.forced_counts = None
        self.fail_step = False
        self.mutate_actions = False
        self.events = []
        self.step_calls = 0

    def observation(self):
        state = np.repeat(self.counts[:, None, None], 2, axis=1).astype(np.float32)
        return {"state": state, "unused": state.copy()}

    def reset(self, **kwargs):
        self.events.append("env_reset")
        self.counts[:] = self.baseline
        return self.observation(), {"reset_kwargs": kwargs}

    def get_attr(self, name):
        assert name == "cur_step"
        self.events.append("read_counts")
        return self.counts if self.forced_counts is None else self.forced_counts

    def step(self, actions):
        self.events.append("env_step")
        self.step_calls += 1
        self.counts += np.minimum(actions.shape[1], np.maximum(self.terminal_steps - self.counts, 0))
        if self.mutate_actions:
            actions.fill_(99) if isinstance(actions, torch.Tensor) else actions.fill(99)
        if self.fail_step:
            raise RuntimeError("step failed after partial execution")
        done = self.counts >= self.terminal_steps
        return self.observation(), done.astype(float), done, np.zeros(self.num_envs, dtype=bool), {}

    def call_each(self, name, **kwargs):
        assert name == "run_dill_function"
        self.events.append("initialize_episodes")

    def call(self, name):
        assert name == "get_rewards"
        return [[0] * int(count) for count in self.counts]

    def render(self):
        self.events.append("render")
        return [None] * self.num_envs

    def close(self):
        self.events.append("close")


@pytest.mark.parametrize("tensor", [False, True], ids=["numpy", "torch"])
def test_partial_execution_and_snapshot_before_mutating_step(tensor):
    env = FakeVectorEnv()
    policy = RecordingPolicy(env.events)
    proxy = ExecutedActionVectorEnv(env, policy)
    result = proxy.reset(seed=31)
    assert result[1] == {"reset_kwargs": {"seed": 31}}
    expected = np.arange(24, dtype=np.float32).reshape(3, 8, 1)
    actions = torch.from_numpy(expected.copy()) if tensor else expected.copy()
    env.mutate_actions = True
    first = proxy.step(actions)
    assert first[0]["state"].shape == (3, 2, 1)
    np.testing.assert_array_equal(policy.records[0][0], expected)
    np.testing.assert_array_equal(policy.records[0][1], [2, 8, 8])
    proxy.step(actions)
    np.testing.assert_array_equal(policy.records[1][1], [0, 2, 8])
    assert env.events.index("env_step") < env.events.index("acknowledge")
    proxy.reset()
    proxy.step(actions)
    np.testing.assert_array_equal(policy.records[-1][1], [2, 8, 8])
    assert proxy.num_envs == 3
    proxy.close()
    assert env.events[-1] == "close"


def test_reset_baseline_is_read_instead_of_assumed_zero():
    env = FakeVectorEnv(terminal_steps=(100, 100, 100), baseline=(3, 4, 5))
    policy = RecordingPolicy()
    proxy = ExecutedActionVectorEnv(env, policy)
    actions = np.zeros((3, 8, 1))
    with pytest.raises(RuntimeError, match="Reset"):
        proxy.step(actions)
    proxy.reset()
    proxy.step(actions)
    np.testing.assert_array_equal(policy.records[0][1], [8, 8, 8])


@pytest.mark.parametrize("autoreset", [True, None, 0])
def test_requires_explicit_no_autoreset(autoreset):
    env = FakeVectorEnv()
    env.autoreset = autoreset
    with pytest.raises(ValueError, match="autoreset=False"):
        ExecutedActionVectorEnv(env, RecordingPolicy())


def test_missing_acknowledgement_hook_is_rejected():
    with pytest.raises(TypeError, match="record_executed_actions"):
        ExecutedActionVectorEnv(FakeVectorEnv(), object())


def test_failed_step_does_not_acknowledge_or_allow_ambiguous_retry():
    env = FakeVectorEnv()
    policy = RecordingPolicy()
    proxy = ExecutedActionVectorEnv(env, policy)
    proxy.reset()
    env.fail_step = True
    with pytest.raises(RuntimeError, match="step failed"):
        proxy.step(np.zeros((3, 8, 1)))
    assert not policy.records
    with pytest.raises(RuntimeError, match="Reset"):
        proxy.step(np.zeros((3, 8, 1)))
    assert env.step_calls == 1
    env.fail_step = False
    proxy.reset()
    proxy.step(np.zeros((3, 8, 1)))
    np.testing.assert_array_equal(policy.records[0][1], [2, 8, 8])


@pytest.mark.parametrize("counts,match", [
    ([1, 2], "one integer"),
    ([1.0, 2.0, 8.0], "one integer"),
    ([True, False, True], "one integer"),
    ([-1, 2, 8], "nonnegative"),
    ([2, 8, 9], "exceeds"),
])
def test_invalid_counts_never_reach_policy(counts, match):
    env = FakeVectorEnv()
    policy = RecordingPolicy()
    proxy = ExecutedActionVectorEnv(env, policy)
    proxy.reset()
    env.forced_counts = counts
    with pytest.raises(ValueError, match=match):
        proxy.step(np.zeros((3, 8, 1)))
    assert not policy.records
    with pytest.raises(RuntimeError, match="Reset"):
        proxy.step(np.zeros((3, 8, 1)))


def test_unexpected_reset_cannot_be_mistaken_for_execution():
    env = FakeVectorEnv(baseline=(5, 5, 5))
    policy = RecordingPolicy()
    proxy = ExecutedActionVectorEnv(env, policy)
    proxy.reset()
    env.forced_counts = [4, 7, 13]
    with pytest.raises(ValueError, match="Unexpected environment reset"):
        proxy.step(np.zeros((3, 8, 1)))
    assert not policy.records


@pytest.mark.parametrize("runner_type", [LiberoExecutedPastRunner, RoboCasaExecutedPastRunner])
@pytest.mark.parametrize("fail_prediction", [False, True])
def test_existing_runner_flow_and_finally_restore(runner_type, fail_prediction):
    runner = runner_type.__new__(runner_type)
    original = FakeVectorEnv(terminal_steps=(2, 10))
    runner.env = original
    runner.env_fns = [None, None]
    runner.env_init_fn_dills = [b"first", b"second"]
    runner.env_seeds = [1000, 1001]
    runner.env_task_names = ["test_task", "test_task"]
    runner.task_name = "test_task"
    runner.max_episode_steps = 16
    runner.n_action_steps = 8
    runner.tqdm_interval_sec = 1000
    runner.protocol = "corrected"
    runner.episode_schedule = [{"episode_index": 0}, {"episode_index": 1}]
    runner.episode_records_path = None
    policy = RecordingPolicy(original.events, fail_prediction=fail_prediction)
    if fail_prediction:
        with pytest.raises(RuntimeError, match="prediction failed"):
            runner.run(policy, temperature=0)
        assert len(policy.records) == 1
    else:
        result = runner.run(policy, temperature=0)
        assert result["mean_success_rate"] == 1.0
        assert result["test_task/mean_success_rate"] == 1.0
        np.testing.assert_array_equal([record[1] for record in policy.records], [[2, 8], [0, 2]])
        if runner_type is LiberoExecutedPastRunner:
            assert [record["policy_steps"] for record in runner.last_episode_records] == [2, 10]
    assert runner.env is original
    assert original.events[:4] == ["initialize_episodes", "env_reset", "read_counts", "policy_reset"]
    assert original.events.index("acknowledge") < original.events.index("predict", original.events.index("predict") + 1)


def test_libero_protocol_is_checked_before_base_initialization(monkeypatch):
    protocols = []

    def initialize(self, *args, **kwargs):
        protocols.append(kwargs["protocol"])

    monkeypatch.setattr(LiberoRunner, "__init__", initialize)
    with pytest.raises(ValueError, match="corrected or official"):
        LiberoExecutedPastRunner("unused", protocol="legacy")
    assert protocols == []
    LiberoExecutedPastRunner("unused")
    LiberoExecutedPastRunner("unused", protocol="official")
    assert protocols == ["corrected", "official"]


class CounterSimulator(gymnasium.Env):
    metadata = {}
    render_mode = None

    def __init__(self, terminal_step):
        self.terminal_step = terminal_step
        self.cur_step = 0
        self.observation_space = gymnasium.spaces.Box(0, 100, shape=(1,), dtype=np.float32)
        self.action_space = gymnasium.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self.cur_step = 0
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        self.cur_step += 1
        done = self.cur_step >= self.terminal_step
        return np.array([self.cur_step], dtype=np.float32), float(done), done, False, {}


def counter_environment(terminal_step):
    return MultiStepWrapper(CounterSimulator(terminal_step), n_obs_steps=2,
                            n_action_steps=8, max_episode_steps=100)


def test_actual_async_workers_report_each_executed_prefix_and_reset():
    vector = AsyncVectorEnv([partial(counter_environment, limit) for limit in (2, 10, 100)],
                            shared_memory=False, context="fork", autoreset=False)
    policy = RecordingPolicy()
    proxy = ExecutedActionVectorEnv(vector, policy)
    try:
        actions = np.zeros((3, 8, 1), dtype=np.float32)
        proxy.reset()
        proxy.step(actions)
        proxy.step(actions)
        np.testing.assert_array_equal([record[1] for record in policy.records], [[2, 8, 8], [0, 2, 8]])
        assert vector.get_attr("cur_step") == (2, 10, 16)
        assert [len(rewards) for rewards in vector.call("get_rewards")] == [2, 10, 16]
        proxy.reset()
        proxy.step(actions)
        np.testing.assert_array_equal(policy.records[-1][1], [2, 8, 8])
    finally:
        proxy.close()
