"""Protocol regressions use a fake simulator; no GPU or rendering is needed."""
import importlib.util
import json
import pathlib
import tempfile
import unittest

import numpy as np
import gymnasium

from oat.gymnasium_util.async_vector_env import AsyncVectorEnv
from oat.gymnasium_util.multistep_wrapper import MultiStepWrapper

from oat.env.libero.env import LiberoEnv
from oat.env_runner.libero_runner import atomic_write_records, build_episode_schedule

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts/evaluate_candidate.py"
spec = importlib.util.spec_from_file_location("evaluate_candidate", SCRIPT)
evaluate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate)


class FakeSimulator:
    def __init__(self):
        self.clock = 0
        self.resets = 0
        self.seeds = []
        self.actions = []
        self.initial_states = []

    def seed(self, value):
        self.seeds.append(value)

    def reset(self):
        self.resets += 1
        self.clock = 0
        return {"clock": self.clock}

    def step(self, action):
        self.actions.append(np.array(action))
        self.clock += 1
        return {"clock": self.clock}, 0, False, {}

    def set_init_state(self, value):
        self.initial_states.append(value)
        self.clock = int(value)
        return {"clock": self.clock}


def fake_env(protocol):
    env = LiberoEnv.__new__(LiberoEnv)
    env.env = FakeSimulator()
    env.protocol = protocol
    env.episode_seed = 101
    env.init_state_id = None
    env.init_states = [10, 20]
    env.task_prompt = "test task"
    env._extract_obs = lambda value: value.copy()
    return env


class CounterEnv(gymnasium.Env):
    metadata = {}
    render_mode = None

    def __init__(self, terminal_step):
        self.terminal_step = terminal_step
        self.observation_space = gymnasium.spaces.Box(0, 100, (1,), dtype=np.float32)
        self.action_space = gymnasium.spaces.Box(-1, 1, (1,), dtype=np.float32)
        self.steps = 0
        self.reset_count = 0

    def reset(self, seed=None, options=None):
        self.steps = 0
        self.reset_count += 1
        return np.array([0], dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        done = self.steps >= self.terminal_step
        return np.array([self.steps], dtype=np.float32), float(done), done, False, {}


def counter_factory(terminal_step):
    return lambda: MultiStepWrapper(CounterEnv(terminal_step), n_obs_steps=1,
                                   n_action_steps=1, max_episode_steps=10)


class ProtocolTests(unittest.TestCase):
    def test_corrected_returns_post_settle_observation_and_applies_seed(self):
        env = fake_env("corrected")
        env.configure_episode(1234)
        obs, info = env.reset()
        self.assertEqual(obs["clock"], 10)
        self.assertEqual(env.env.resets, 1)
        self.assertEqual(env.env.seeds, [1234])
        self.assertEqual(info["episode_seed"], 1234)
        self.assertEqual(env.cur_step, 0)
        np.testing.assert_array_equal(env.env.actions, [[0] * 6 + [-1]] * 10)

    def test_legacy_preserves_stale_observation_and_ignored_reset_seed(self):
        env = fake_env("legacy")
        obs, _ = env.reset(seed=987)
        self.assertEqual(obs["clock"], 0)
        self.assertEqual(env.env.clock, 10)
        self.assertEqual(env.env.seeds, [])

    def test_official_initial_state_and_five_zero_actions(self):
        env = fake_env("official")
        env.configure_episode(4321, 1)
        obs, info = env.reset()
        self.assertEqual(obs["clock"], 25)
        self.assertEqual(env.env.resets, 1)
        self.assertEqual(env.env.initial_states, [20])
        self.assertEqual(info["init_state_id"], 1)
        np.testing.assert_array_equal(env.env.actions, np.zeros((5, 7)))
        self.assertEqual(env.cur_step, 0)

    def test_official_indices_never_wrap(self):
        env = fake_env("official")
        for index in (-1, 2, None):
            with self.assertRaises(ValueError):
                env.configure_episode(123, index)

    def test_schedule_stable_across_worker_counts(self):
        expected = build_episode_schedule(["a", "b", "c"], 12, 1, 100, "corrected")
        for workers in (2, 3, 7, 12):
            self.assertEqual(expected, build_episode_schedule(["a", "b", "c"], 12, workers, 100, "corrected"))
        self.assertEqual([r["episode_seed"] for r in expected], list(range(100, 112)))

    def test_official_state_ids_advance_per_task(self):
        records = build_episode_schedule(["a", "b"], 6, 2, 500, "official", 10)
        self.assertEqual([r["init_state_id"] for r in records], [10, 10, 11, 11, 12, 12])
        self.assertTrue(all(r["episode_seed_applied"] for r in records))

    def test_legacy_schedule_preserves_original_batch_order(self):
        records = build_episode_schedule(["a", "b", "c"], 9, 2, 1000, "legacy")
        self.assertEqual([r["task_name"] for r in records], ["a", "b"] * 3 + ["c"] * 3)
        self.assertTrue(all(not r["episode_seed_applied"] for r in records))

    def test_no_autoreset_keeps_completed_episode_records(self):
        env = AsyncVectorEnv([counter_factory(1), counter_factory(3)],
                             shared_memory=False, autoreset=False)
        try:
            env.reset()
            actions = np.zeros((2, 1, 1), dtype=np.float32)
            env.step(actions)
            env.step(actions)
            self.assertEqual([len(r) for r in env.call("get_rewards")], [1, 2])
            self.assertEqual(env.get_attr("reset_count"), (1, 1))
            env.step(actions)
            self.assertEqual([len(r) for r in env.call("get_rewards")], [1, 3])
        finally:
            env.close()

    def test_no_autoreset_executes_first_action_after_explicit_reset(self):
        env = AsyncVectorEnv([counter_factory(1)], shared_memory=False, autoreset=False)
        try:
            actions = np.zeros((1, 1, 1), dtype=np.float32)
            env.reset()
            env.step(actions)
            env.reset()
            obs, _, done, _, _ = env.step(actions)
            self.assertTrue(done[0])
            self.assertEqual(float(obs[0, 0, 0]), 1)
            self.assertEqual(env.get_attr("reset_count"), (2,))
            self.assertEqual(len(env.call("get_rewards")[0]), 1)
        finally:
            env.close()

    def test_legacy_worker_autoreset_is_preserved(self):
        env = AsyncVectorEnv([counter_factory(1)], shared_memory=False)
        try:
            actions = np.zeros((1, 1, 1), dtype=np.float32)
            env.reset()
            env.step(actions)
            env.reset()
            _, _, done, _, _ = env.step(actions)
            self.assertFalse(done[0])
            self.assertEqual(env.get_attr("reset_count"), (3,))
            self.assertEqual(len(env.call("get_rewards")[0]), 0)
        finally:
            env.close()

    def test_atomic_episode_output_and_summary(self):
        records = [
            {"task_name": "a", "success": True, "policy_steps": 12, "video_path": "/tmp/a.mp4"},
            {"task_name": "a", "success": False, "policy_steps": 550, "video_path": None},
            {"task_name": "b", "success": True, "policy_steps": 33, "video_path": None},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "episodes.jsonl"
            atomic_write_records(path, records[:1])
            atomic_write_records(path, records)
            self.assertEqual([json.loads(line) for line in path.read_text().splitlines()], records)
            summary = evaluate.summarize_records(records)
            self.assertEqual(summary["successes"], 2)
            self.assertEqual(summary["success_rate"], 2 / 3)
            self.assertEqual(summary["macro_task_success_rate"], 0.75)
            self.assertEqual(summary["per_task"]["a"]["trials"], 2)
            evaluate.atomic_json(pathlib.Path(directory) / "summary.json", summary)
            self.assertEqual(len(list(pathlib.Path(directory).iterdir())), 2)


if __name__ == "__main__":
    unittest.main()
