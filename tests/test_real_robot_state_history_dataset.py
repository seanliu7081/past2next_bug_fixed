"""Check the additive real-robot history dataset using tiny synthetic images."""

import numpy as np
import pytest
import torch

from oat.common.replay_buffer import ReplayBuffer
from oat.common.seq_sampler import get_val_mask
from oat.dataset.real_robot_dataset import RealRobotZarrDatasetWithPrevWindow
from oat.dataset.real_robot_state_history import RealRobotZarrDatasetWithStateHistory


STATE_KEYS = ("robot0_eef_pos", "robot0_eef_rot6d", "robot0_gripper_qpos")
RGB_KEYS = ("agentview_rgb", "robot0_eye_in_hand_rgb")
OBS_KEYS = [*RGB_KEYS, *STATE_KEYS, "task_uid"]
LENGTHS = (3, 9, 17, 23, 6)


@pytest.fixture
def robot_store(tmp_path):
    replay = ReplayBuffer.create_empty_numpy()
    holdout = get_val_mask(len(LENGTHS), val_ratio=0.2, seed=42)
    episodes = []
    for episode, length in enumerate(LENGTHS):
        steps = np.arange(length, dtype=np.float32)
        base = 100 * (episode + 1) + (1_000_000 if holdout[episode] else 0)
        values = base + steps
        theta = steps / 10
        # First two rotation-matrix rows, kept raw by the dataset.
        rotation = np.stack((np.cos(theta), -np.sin(theta), np.zeros(length),
                             np.sin(theta), np.cos(theta), np.zeros(length)), axis=-1).astype(np.float32)
        action = np.repeat(values[:, None], 7, axis=1)
        action[:, 6] = steps % 2
        fields = {
            "action": action,
            "robot0_eef_pos": np.repeat(values[:, None], 3, axis=1),
            "robot0_eef_rot6d": rotation,
            "robot0_gripper_qpos": (80 * (1 - action[:, 6:7])).astype(np.float32),
            "task_uid": np.zeros((length, 1), dtype=np.int64),
            "agentview_rgb": np.full((length, 3, 4, 3), 37, dtype=np.uint8),
            "robot0_eye_in_hand_rgb": np.full((length, 3, 4, 3), 64, dtype=np.uint8),
        }
        replay.add_episode(fields)
        episodes.append(fields)
    path = tmp_path / "tiny_real_robot_history.zarr"
    replay.save_to_path(str(path))
    return str(path), episodes, holdout


def options(path, **overrides):
    args = dict(zarr_path=path, obs_keys=OBS_KEYS, n_obs_steps=2,
                n_action_steps=16, seed=42, val_ratio=0.2, past_n=7,
                n_exec_steps=8, history_padding="zero", return_history_validity=True)
    args.update(overrides)
    return args


@pytest.mark.parametrize("history_steps", [8, 16])
@pytest.mark.parametrize("stride", [4, 8])
def test_causal_current_previous_histories_and_episode_boundaries(robot_store, history_steps, stride):
    path, episodes, holdout = robot_store
    kwargs = options(path, past_n=history_steps - 1, n_exec_steps=stride)
    dataset = RealRobotZarrDatasetWithStateHistory(
        state_history_steps=history_steps, state_history_keys=STATE_KEYS, **kwargs)
    parent = RealRobotZarrDatasetWithPrevWindow(**kwargs)
    views = ((dataset, parent, ~holdout),
             (dataset.get_validation_dataset(), parent.get_validation_dataset(), holdout))
    for view, old_view, selected in views:
        assert isinstance(view, RealRobotZarrDatasetWithStateHistory)
        assert len(view) == sum(length for length, chosen in zip(LENGTHS, selected) if chosen)
        offset = 0
        for episode_id in np.flatnonzero(selected):
            fields = episodes[episode_id]
            length = LENGTHS[episode_id]
            for step in range(length):
                sample = view[offset + step]
                old = old_view[offset + step]
                assert sample["episode_step"].item() == step
                for name in set(old) - {"obs", "prev_obs"}:
                    torch.testing.assert_close(sample[name], old[name], rtol=0, atol=0)
                for name, end, action_mask in (("obs", step, "past_action_valid"),
                                               ("prev_obs", step - stride, "prev_past_action_valid")):
                    for key, previous in old[name].items():
                        torch.testing.assert_close(sample[name][key], previous, rtol=0, atol=0)
                    positions = np.arange(end - history_steps + 1, end + 1)
                    valid = positions >= 0
                    torch.testing.assert_close(sample[name]["state_history_valid"], torch.from_numpy(valid))
                    assert sample[name]["state_history_valid"].dtype == torch.bool
                    torch.testing.assert_close(sample[action_mask], torch.from_numpy(valid[:-1] & valid[1:]))
                    for key in STATE_KEYS:
                        expected = np.zeros((history_steps, fields[key].shape[1]), dtype=np.float32)
                        expected[valid] = fields[key][positions[valid]]
                        actual = sample[name]["state_history__" + key]
                        assert actual.dtype == torch.float32
                        torch.testing.assert_close(actual, torch.from_numpy(expected), rtol=0, atol=0)
                    for key in RGB_KEYS:
                        assert sample[name][key].dtype == torch.uint8
                        assert sample[name][key].shape == (2, 3, 4, 3)
                    assert sample[name]["robot0_eef_rot6d"].shape == (2, 6)
                    assert sample[name]["robot0_gripper_qpos"].shape == (2, 1)
                if step == 0:
                    assert sample["obs"]["state_history_valid"].sum().item() == 1
                    assert not sample["prev_obs"]["state_history_valid"].any()
                    assert not sample["past_action"].any()
                    assert not sample["prev_past_action"].any()
            offset += length


@pytest.mark.parametrize("mode", ["limits", "gaussian"])
def test_training_and_validation_normalizers_use_only_training_frames(robot_store, mode):
    path, episodes, holdout = robot_store
    dataset = RealRobotZarrDatasetWithStateHistory(
        state_history_keys=STATE_KEYS, **options(path))
    validation = dataset.get_validation_dataset()
    np.testing.assert_array_equal(dataset.train_mask, ~holdout)
    np.testing.assert_array_equal(validation.train_mask, holdout)
    fields = ("action", *STATE_KEYS, "task_uid")
    for view in (dataset, validation):
        normalizer = view.get_normalizer(mode=mode)
        assert not any(key.startswith("state_history") for key in normalizer.get_input_stats())
        for key in fields:
            training_values = np.concatenate([episode[key] for episode, chosen
                                               in zip(episodes, ~holdout) if chosen])
            expected = torch.from_numpy(training_values).float()
            stats = normalizer[key].get_input_stats()
            torch.testing.assert_close(stats["min"], expected.amin(dim=0))
            torch.testing.assert_close(stats["max"], expected.amax(dim=0))
            torch.testing.assert_close(stats["mean"], expected.mean(dim=0))
            torch.testing.assert_close(stats["std"], expected.std(dim=0))
        assert normalizer["robot0_eef_pos"].get_input_stats()["max"].max() < 1_000_000
    assert validation[0]["obs"]["state_history__robot0_eef_pos"][-1].min() > 1_000_000


def test_rgb_statistics_are_fixed_without_reading_pixels_and_units_are_preserved(robot_store, monkeypatch):
    path, _, _ = robot_store
    dataset = RealRobotZarrDatasetWithStateHistory(
        state_history_keys=STATE_KEYS, **options(path))
    sample = dataset[0]

    class UnreadableImage:
        dtype = np.dtype("uint8")
        shape = dataset.replay_buffer[RGB_KEYS[0]].shape

        def __getitem__(self, index):
            raise AssertionError("RGB normalization must not read pixels")

        def __array__(self, *args, **kwargs):
            raise AssertionError("RGB normalization must not materialize images")

    for key in RGB_KEYS:
        monkeypatch.setitem(dataset.replay_buffer.root["data"], key, UnreadableImage())
    for view in (dataset, dataset.get_validation_dataset()):
        normalizer = view.get_normalizer()
        for key in RGB_KEYS:
            endpoints = torch.tensor([[0, 0, 0], [255, 255, 255]], dtype=torch.uint8)
            torch.testing.assert_close(normalizer[key].normalize(endpoints),
                                       torch.tensor([[-1.] * 3, [1.] * 3]))
            assert sample["obs"][key].dtype == torch.uint8
        widths = normalizer["robot0_gripper_qpos"].get_input_stats()
        assert widths["min"].item() == 0 and widths["max"].item() == 80
        commands = normalizer["action"].get_input_stats()
        assert commands["min"][-1].item() == 0 and commands["max"][-1].item() == 1
    torch.testing.assert_close(sample["obs"]["state_history__robot0_gripper_qpos"][-1],
                               80 * (1 - sample["action"][0, 6:7]))
