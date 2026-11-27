from pathlib import Path

import gymnasium as gym
import hydra
import numpy as np
import torch

from oat.env_runner import mimicgen_runner as rollout
from scripts.prepare_mimicgen_prt import TASKS


def test_mimicgen_recipes_enable_evaluation_and_use_training_task_ids(tmp_path):
    config_dir = Path(__file__).resolve().parents[1] / "oat/config"
    with hydra.initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        tokenizer = hydra.compose(config_name="PRT_Tok_MimicGen6")
        policy = hydra.compose(config_name="PRT_Sparse_Policy_MimicGen6")
    assert tokenizer.logging.mode == policy.logging.mode == "online"
    assert tokenizer.task.tokenizer.lazy_eval is False
    assert policy.task.policy.lazy_eval is False
    runner = hydra.utils.instantiate(policy.task.policy.env_runner, output_dir=str(tmp_path))
    assert isinstance(runner, rollout.MimicGenRunner)
    assert runner.tasks == [(name, uid) for uid, name in enumerate(TASKS)]


class CounterEnv(gym.Env):
    action_space = gym.spaces.Box(-1, 1, (7,), np.float32)
    observation_space = gym.spaces.Dict({"state": gym.spaces.Box(0, 100, (1,), np.float32)})

    def reset(self, *, seed=None, options=None):
        self.stop = 1 if seed % 3 else 5
        self.steps = 0
        return {"state": np.zeros(1, np.float32)}, {}

    def step(self, action):
        assert self.steps < self.stop, "a finished slot was stepped again"
        self.steps += 1
        success = self.stop == 1
        return ({"state": np.array([self.steps], np.float32)}, float(success),
                success, self.steps == 5, {"success": success})


def make_counter(task_name, uid, image_size, egl_device, max_episode_steps, n_obs_steps, n_action_steps):
    assert task_name == "threading_d2" and uid == 5
    return rollout._EpisodeChunks(CounterEnv(), n_obs_steps, n_action_steps)


class RecordingPolicy:
    device = torch.device("cpu")
    dtype = torch.float32

    def __init__(self):
        self.lengths = []
        self.states = []
        self.resets = 0

    def reset(self):
        self.resets += 1
        self.pending = False

    def get_observation_ports(self):
        return ["state"]

    def predict_action(self, obs):
        assert not self.pending
        self.pending = True
        self.states.append(obs["state"].tolist())
        return {"action": torch.ones((len(obs["state"]), 3, 7))}

    def record_executed_actions(self, actions, executed_lengths):
        assert self.pending and np.all(actions == 1)
        self.lengths.append(executed_lengths.tolist())
        self.pending = False


def test_real_worker_pool_reports_only_executed_prefixes(monkeypatch, tmp_path):
    monkeypatch.setattr(rollout, "_make_env", make_counter)
    runner = rollout.MimicGenRunner(
        tmp_path, ["threading_d2"], n_per_task=3, n_parallel_envs=2, n_action_steps=3,
    )
    policy = RecordingPolicy()
    metrics = runner.run(policy)
    assert metrics == {"n_episodes": 3, "mean_success_rate": 2/3, "threading_d2/mean_success_rate": 2/3}
    assert policy.resets == 2
    assert policy.lengths == [[1, 3], [0, 2], [1]]
    assert policy.states[0] == [[[0], [0]], [[0], [0]]]
    assert policy.states[1] == [[[0], [1]], [[2], [3]]]
    assert policy.states[2] == [[[0], [0]]]
    assert runner.env is None
