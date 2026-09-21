"""State/action transition alignment and inherited train-only normalization."""

import numpy as np
import pytest
import torch

from oat.common.replay_buffer import ReplayBuffer
from oat.common.seq_sampler import get_val_mask
from oat.dataset.zarr_dataset_with_prev_window import ZarrDatasetWithPrevWindow
from oat.dataset.zarr_dataset_with_state_history import ZarrDatasetWithStateHistory


STATE_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
OBS_KEYS = [*STATE_KEYS, "rgb", "description"]
LENGTHS = (27, 19, 31, 23)
SEED = 42
VAL_RATIO = 0.25


@pytest.fixture
def make_store(tmp_path):
    paths = []

    def create(holdout_shift=0.0, replace=None):
        replay = ReplayBuffer.create_empty_numpy()
        holdout = get_val_mask(len(LENGTHS), VAL_RATIO, SEED)
        episodes = []
        for index, length in enumerate(LENGTHS):
            steps = np.arange(length, dtype=np.float64)
            base = 100 * (index + 1)
            shift = holdout_shift if holdout[index] else 0
            theta = steps / 10
            quat = np.stack([np.sin(theta / 2), np.zeros(length),
                             np.zeros(length), np.cos(theta / 2)], axis=-1)
            quat[1::2] *= -1  # Dataset must preserve raw quaternion sign and xyzw order.
            episode = {
                "commands": (base + shift + steps[:, None] + np.arange(7)[None, :] / 10).astype(np.float32),
                STATE_KEYS[0]: base + shift + steps[:, None] + np.array([0.1, 0.2, 0.3]),
                STATE_KEYS[1]: quat,
                STATE_KEYS[2]: (steps[:, None] * np.array([0.01, -0.02]) + base + shift).astype(np.float32),
                "rgb": np.broadcast_to(((base + steps) % 251)[:, None, None, None], (length, 3, 2, 2)).astype(np.uint8),
                "description": np.array([f"task_{index}"] * length),
            }
            if replace is not None:
                key, factory = replace
                episode[key] = factory(length)
            replay.add_episode(episode)
            episodes.append(episode)
        path = tmp_path / f"states_{len(paths)}.zarr"
        replay.save_to_path(str(path))
        paths.append(path)
        return str(path), episodes

    return create


def dataset_kwargs(path, **overrides):
    kwargs = dict(zarr_path=path, obs_keys=OBS_KEYS, action_key="commands",
                  n_obs_steps=2, n_action_steps=16, seed=SEED)
    kwargs.update(overrides)
    return kwargs


def expected_states(episode, key, end, history_steps):
    positions = np.arange(end - history_steps + 1, end + 1)
    valid = positions >= 0
    values = np.zeros((history_steps, episode[key].shape[1]), dtype=np.float32)
    values[valid] = episode[key][positions[valid]]
    return torch.from_numpy(values), torch.from_numpy(valid)


@pytest.mark.parametrize("history_steps", [8, 16])
@pytest.mark.parametrize("stride", [4, 8])
def test_current_and_previous_states_align_without_changing_parent_outputs(make_store, history_steps, stride):
    path, episodes = make_store()
    kwargs = dataset_kwargs(path, past_n=history_steps - 1, n_exec_steps=stride,
                            history_padding="zero", return_history_validity=True)
    dataset = ZarrDatasetWithStateHistory(state_history_steps=history_steps, **kwargs)
    parent = ZarrDatasetWithPrevWindow(**kwargs)
    offset = 0
    for episode in episodes:
        for step in range(len(episode["commands"])):
            index = offset + step
            sample, old = dataset[index], parent[index]
            assert set(sample) == set(old)
            assert sample["episode_step"].item() == step
            for name in ("obs", "prev_obs"):
                assert set(sample[name]) == set(old[name]) | {
                    *("state_history__" + key for key in STATE_KEYS), "state_history_valid",
                }
                for key, value in old[name].items():
                    if isinstance(value, torch.Tensor):
                        torch.testing.assert_close(sample[name][key], value, rtol=0, atol=0)
                    else:
                        assert sample[name][key] == value
                assert sample[name]["rgb"].shape == (2, 3, 2, 2)
            for key in set(sample) - {"obs", "prev_obs"}:
                torch.testing.assert_close(sample[key], old[key], rtol=0, atol=0)

            for name, end, action_mask in (
                ("obs", step, "past_action_valid"),
                ("prev_obs", step - stride, "prev_past_action_valid"),
            ):
                for key in STATE_KEYS:
                    expected, valid = expected_states(episode, key, end, history_steps)
                    actual = sample[name]["state_history__" + key]
                    assert actual.dtype == torch.float32
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                assert sample[name]["state_history_valid"].dtype == torch.bool
                torch.testing.assert_close(sample[name]["state_history_valid"], valid)
                # Each past command joins the state at its start to the next state.
                torch.testing.assert_close(sample[action_mask], valid[:-1] & valid[1:])
            if step == 0:
                assert sample["obs"]["state_history_valid"].sum().item() == 1
                assert not sample["prev_obs"]["state_history_valid"].any()
                assert not sample["past_action_valid"].any()
        offset += len(episode["commands"])


def test_history_masking_and_caller_mutation_do_not_change_replay_or_regular_observations(make_store):
    path, episodes = make_store()
    dataset = ZarrDatasetWithStateHistory(**dataset_kwargs(path))
    sample = dataset[0]
    key = STATE_KEYS[0]
    sample["obs"]["state_history__" + key].fill_(999)
    sample["prev_obs"]["state_history__" + key].fill_(-999)
    expected, valid = expected_states(episodes[0], key, 0, 8)
    again = dataset[0]
    torch.testing.assert_close(again["obs"]["state_history__" + key], expected)
    torch.testing.assert_close(again["obs"]["state_history_valid"], valid)
    torch.testing.assert_close(again["obs"][key][-1], torch.from_numpy(episodes[0][key][0]).float())
    np.testing.assert_array_equal(dataset.replay_buffer[key][:LENGTHS[0]], episodes[0][key])


@pytest.mark.parametrize("mode", ["limits", "gaussian"])
@pytest.mark.parametrize("max_train_episodes", [None, 1])
def test_state_history_validation_views_keep_original_training_statistics(make_store, mode, max_train_episodes):
    path, episodes = make_store(holdout_shift=100_000)
    kwargs = dataset_kwargs(path, val_ratio=VAL_RATIO, max_train_episodes=max_train_episodes)
    dataset = ZarrDatasetWithStateHistory(**kwargs)
    validation = dataset.get_validation_dataset()
    assert isinstance(validation, ZarrDatasetWithStateHistory)
    assert validation.state_history_steps == 8
    assert validation.state_history_keys == STATE_KEYS
    assert validation[0]["obs"]["state_history_valid"].tolist() == [False] * 7 + [True]
    fields = {"action": "commands", **{key: key for key in [*STATE_KEYS, "rgb"]}}
    train_normalizer = dataset.get_normalizer(mode=mode)
    for view in (dataset, validation):
        normalizer = view.get_normalizer(mode=mode)
        assert set(normalizer.get_input_stats()) == set(fields)
        for output_key, key in fields.items():
            values = np.concatenate([episode[key] for episode, selected
                                     in zip(episodes, dataset.train_mask) if selected])
            expected = torch.from_numpy(values).float().reshape(-1, values.shape[-1])
            stats = normalizer[output_key].get_input_stats()
            torch.testing.assert_close(stats["min"], expected.amin(dim=0))
            torch.testing.assert_close(stats["max"], expected.amax(dim=0))
            torch.testing.assert_close(stats["mean"], expected.mean(dim=0))
            torch.testing.assert_close(stats["std"], expected.std(dim=0))
            torch.testing.assert_close(normalizer[output_key].normalize(expected),
                                       train_normalizer[output_key].normalize(expected))
    # Changing held-out extremes cannot affect raw-state/action normalization.
    changed_path, _ = make_store(holdout_shift=-200_000)
    changed = ZarrDatasetWithStateHistory(**{**kwargs, "zarr_path": changed_path})
    before = train_normalizer.state_dict()
    after = changed.get_validation_dataset().get_normalizer(mode=mode).state_dict()
    assert before.keys() == after.keys()
    for key in before:
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)


@pytest.mark.parametrize("options,match", [
    ({"state_history_steps": 7}, "past_n"),
    ({"state_history_steps": 8.0}, "integer"),
    ({"past_n": True}, "integer"),
    ({"n_obs_steps": 0}, "n_obs_steps"),
    ({"n_action_steps": 0}, "n_action_steps"),
    ({"n_exec_steps": 0}, "n_exec_steps"),
    ({"history_padding": "edge"}, "zero"),
    ({"return_history_validity": False}, "validity"),
    ({"state_history_keys": ()}, "nonempty"),
    ({"state_history_keys": STATE_KEYS[0]}, "sequence"),
    ({"state_history_keys": [STATE_KEYS[0], STATE_KEYS[0]]}, "distinct"),
    ({"state_history_keys": [None]}, "nonempty"),
    ({"state_history_keys": ["missing"]}, "obs_keys"),
    ({"obs_keys": [*OBS_KEYS, "state_history_valid"]}, "reserved"),
    ({"obs_keys": [*OBS_KEYS, "state_history__custom"]}, "reserved"),
])
def test_invalid_options_fail_before_loading_replay(options, match):
    with pytest.raises(ValueError, match=match):
        ZarrDatasetWithStateHistory(**dataset_kwargs("unused.zarr", **options))


@pytest.mark.parametrize("key,factory,match", [
    (STATE_KEYS[0], lambda length: np.zeros((length, 2)), "shape"),
    (STATE_KEYS[1], lambda length: np.zeros((length, 3)), "shape"),
    (STATE_KEYS[2], lambda length: np.zeros(length), "vectors"),
    (STATE_KEYS[2], lambda length: np.zeros((length, 2, 2)), "vectors"),
    (STATE_KEYS[2], lambda length: np.zeros((length, 2), dtype=bool), "numeric"),
    (STATE_KEYS[2], lambda length: np.full((length, 2), "text"), "numeric"),
])
def test_invalid_state_shapes_and_dtypes_fail_clearly(make_store, key, factory, match):
    path, _ = make_store(replace=(key, factory))
    with pytest.raises(ValueError, match=match):
        ZarrDatasetWithStateHistory(**dataset_kwargs(path))
