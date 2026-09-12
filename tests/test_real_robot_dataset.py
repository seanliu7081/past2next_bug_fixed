"""Exercise split isolation and Past2Next episode boundaries on small Zarr data."""

import numpy as np
import pytest
import torch

from oat.common.replay_buffer import ReplayBuffer
from oat.common.seq_sampler import get_val_mask
from oat.dataset.real_robot_dataset import (
    RealRobotZarrDataset,
    RealRobotZarrDatasetWithPrevWindow,
)


OBS_KEYS = [
    "agentview_rgb", "robot0_eye_in_hand_rgb", "robot0_eef_pos",
    "robot0_eef_rot6d", "robot0_gripper_qpos", "task_uid",
]


@pytest.fixture
def robot_store(tmp_path):
    replay = ReplayBuffer.create_empty_numpy()
    val_mask = get_val_mask(10, val_ratio=0.1, seed=42)
    lengths = np.arange(17, 27)
    for episode, length in enumerate(lengths):
        # A distinct per-episode ramp makes accidental boundary crossing visible.
        values = (episode * 100 + np.arange(length)).astype(np.float32)
        if val_mask[episode]:
            values += 1_000_000
        action = np.repeat(values[:, None], 7, axis=1)
        action[:, 6] = np.arange(length) % 2
        replay.add_episode({
            "action": action,
            "robot0_eef_pos": np.repeat(values[:, None], 3, axis=1),
            "robot0_eef_rot6d": np.tile([1., 0., 0., 0., 1., 0.], (length, 1)).astype(np.float32),
            "robot0_gripper_qpos": (80 * (1 - action[:, 6:7])).astype(np.float32),
            "task_uid": np.zeros((length, 1), dtype=np.int64),
            "agentview_rgb": np.full((length, 4, 5, 3), 37, dtype=np.uint8),
            "robot0_eye_in_hand_rgb": np.full((length, 4, 5, 3), 64, dtype=np.uint8),
        })
    path = tmp_path / "real_robot.zarr"
    replay.save_to_path(str(path))
    return str(path), replay, val_mask, lengths


def make_datasets(path):
    shared = dict(zarr_path=path, n_action_steps=16, seed=42, val_ratio=0.1)
    tokenizer = RealRobotZarrDataset(**shared, n_obs_steps=0, obs_keys=[])
    policy = RealRobotZarrDatasetWithPrevWindow(
        **shared, n_obs_steps=2, obs_keys=OBS_KEYS, past_n=7, n_exec_steps=8,
    )
    return tokenizer, policy


def test_stages_share_split_and_fit_only_training_episodes(robot_store):
    path, replay, val_mask, lengths = robot_store
    tokenizer, policy = make_datasets(path)
    training_frames = np.repeat(~val_mask, lengths)
    expected = torch.from_numpy(replay["action"][training_frames])
    for dataset in (tokenizer, policy):
        np.testing.assert_array_equal(dataset.train_mask, ~val_mask)
        assert len(dataset) == lengths[~val_mask].sum()
        validation = dataset.get_validation_dataset()
        np.testing.assert_array_equal(validation.train_mask, val_mask)
        assert len(validation) == lengths[val_mask].sum()
        # Even a normalizer obtained from the validation view must use training.
        for view in (dataset, validation):
            stats = view.get_normalizer()["action"].get_input_stats()
            torch.testing.assert_close(stats["min"], expected.min(dim=0).values)
            torch.testing.assert_close(stats["max"], expected.max(dim=0).values)
            torch.testing.assert_close(stats["mean"], expected.mean(dim=0))
    stats = policy.get_normalizer()["robot0_eef_pos"].get_input_stats()
    torch.testing.assert_close(stats["max"], expected[:, :3].max(dim=0).values)
    assert policy.get_validation_dataset()[0]["action"][0, 0] >= 1_000_000


def test_previous_and_current_windows_stay_inside_each_episode(robot_store):
    path, replay, val_mask, lengths = robot_store
    _, policy = make_datasets(path)
    episode_starts = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    for dataset, mask in ((policy, ~val_mask), (policy.get_validation_dataset(), val_mask)):
        dataset_offset = 0
        for episode in np.flatnonzero(mask):
            length, start = lengths[episode], episode_starts[episode]
            for step in (0, 1, 8, length - 1):
                sample = dataset[dataset_offset + step]
                def expected_action(relative_steps):
                    indices = start + np.clip(relative_steps, 0, length - 1)
                    return torch.from_numpy(replay["action"][indices])
                torch.testing.assert_close(sample["action"], expected_action(np.arange(step, step + 16)))
                torch.testing.assert_close(sample["past_action"], expected_action(np.arange(step - 7, step)))
                torch.testing.assert_close(sample["prev_past_action"], expected_action(np.arange(step - 15, step - 8)))
                for key, shift in (("obs", 0), ("prev_obs", 8)):
                    expected = expected_action(np.arange(step - shift - 1, step - shift + 1))
                    torch.testing.assert_close(sample[key]["robot0_eef_pos"], expected[:, :3])
                    assert sample[key]["agentview_rgb"].dtype == torch.uint8
                    assert sample[key]["robot0_eef_rot6d"].shape == (2, 6)
                    assert sample[key]["robot0_gripper_qpos"].shape == (2, 1)
                    assert sample[key]["task_uid"].eq(0).all()
            dataset_offset += length


def test_rgb_normalizer_never_reads_pixels_and_preserves_units(robot_store, monkeypatch):
    path, _, _, _ = robot_store
    _, policy = make_datasets(path)
    sample = policy[0]

    class UnreadableImage:
        dtype = np.dtype("uint8")
        shape = policy.replay_buffer["agentview_rgb"].shape

        def __getitem__(self, index):
            raise AssertionError("Normalizer must not read image pixels")

        def __array__(self, *args, **kwargs):
            raise AssertionError("Normalizer must not materialize image pixels")

    for key in OBS_KEYS[:2]:
        monkeypatch.setitem(policy.replay_buffer.root["data"], key, UnreadableImage())
    normalizer = policy.get_normalizer()
    for key in OBS_KEYS[:2]:
        endpoints = torch.tensor([[0, 0, 0], [255, 255, 255]], dtype=torch.uint8)
        torch.testing.assert_close(normalizer[key].normalize(endpoints), torch.tensor([[-1.] * 3, [1.] * 3]))
    # Observation width and command level use separate ranges and directions.
    width_stats = normalizer["robot0_gripper_qpos"].get_input_stats()
    assert width_stats["min"].item() == 0
    assert width_stats["max"].item() == 80
    action_stats = normalizer["action"].get_input_stats()
    assert action_stats["min"][-1].item() == 0
    assert action_stats["max"][-1].item() == 1
    torch.testing.assert_close(sample["obs"]["robot0_gripper_qpos"][-1], 80 * (1 - sample["action"][0, 6:7]))
    torch.testing.assert_close(
        normalizer["action"].unnormalize(normalizer["action"].normalize(sample["action"])),
        sample["action"], atol=1e-4, rtol=1e-5,
    )
