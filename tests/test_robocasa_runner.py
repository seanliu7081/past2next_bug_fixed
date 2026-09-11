"""RoboCasa evaluation contracts, without requiring simulator assets."""

from types import SimpleNamespace

import dill
import numpy as np
import pytest

from oat.env.robocasa.env import RoboCasaEnv
from oat.env.robocasa.factory import get_subtasks, get_task_uid
from oat.env.robocasa.multitask_env import MultiTaskRoboCasaEnv
from oat.env_runner import robocasa_multitask_runner as runner_module


class FakeVectorEnv:
    def __init__(self, env_fns, **kwargs):
        self.kwargs = kwargs


def test_sink3_preserves_dataset_task_ids():
    assert [get_task_uid(name) for name in get_subtasks("sink3")] == [4, 2, 5]
    with pytest.raises(KeyError):
        get_task_uid("unknown")


@pytest.mark.parametrize("workers", [1, 2, 3, 6])
def test_sink3_schedule_and_episode_seed(monkeypatch, tmp_path, workers):
    monkeypatch.setattr(runner_module, "AsyncVectorEnv", FakeVectorEnv)
    monkeypatch.setattr(runner_module.torch.cuda, "device_count", lambda: 1)
    runner = runner_module.RoboCasaMultiTaskRunner(
        tmp_path, n_test=6, n_test_vis=0, n_parallel_envs=workers,
    )
    assert runner.env_task_names == get_subtasks("sink3") * 2
    assert runner.env_seeds == list(range(1000, 1006))
    assert runner.env.kwargs["autoreset"] is False
    assert runner.env.kwargs["context"] == "forkserver"
    configured_seeds = []
    reset_calls = []
    inner = SimpleNamespace(
        env_name=runner.env_task_names[4],
        configure_episode=configured_seeds.append,
    )
    wrapped = SimpleNamespace(
        env=SimpleNamespace(
            env=inner,
            video_recoder=SimpleNamespace(stop=lambda: None),
            file_path="old.mp4",
        ),
        reset=lambda: reset_calls.append(True),
    )
    dill.loads(runner.env_init_fn_dills[4])(wrapped)
    assert configured_seeds == [1004]
    assert reset_calls == []  # The vector runner performs exactly one reset.
    assert wrapped.env.file_path is None


def test_robocasa_reset_applies_and_consumes_scheduled_seed():
    env = RoboCasaEnv.__new__(RoboCasaEnv)
    simulator = SimpleNamespace(seed=0, rng=np.random.default_rng(0))
    simulator.reset = lambda: {"value": simulator.rng.uniform(size=4)}
    env.env = simulator
    env._episode_seed = None
    env._extract_obs = lambda observation: observation
    env.configure_episode(1000)
    first, _ = env.reset()
    second, _ = env.reset()
    env.configure_episode(1000)
    repeated, _ = env.reset()
    assert simulator.seed == 1000
    np.testing.assert_array_equal(first["value"], repeated["value"])
    assert not np.array_equal(first["value"], second["value"])
    assert env._episode_seed is None
    assert env.cur_step == 0
    assert not env.done


def test_robocasa_observations_match_training_image_origin_and_uid():
    env = MultiTaskRoboCasaEnv.__new__(MultiTaskRoboCasaEnv)
    env.state_ports = ["robot0_eef_pos"]
    env.camera_names = ["robot0_eye_in_hand"]
    env.task_uid = 2
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    observation = env._extract_obs({
        "robot0_eef_pos": np.array([1, 2, 3], dtype=np.float64),
        "robot0_eye_in_hand_image": image,
    })
    np.testing.assert_array_equal(observation["robot0_eye_in_hand_image"], image[::-1])
    assert observation["robot0_eef_pos"].dtype == np.float32
    np.testing.assert_array_equal(observation["task_uid"], np.array([2], dtype=np.uint8))
