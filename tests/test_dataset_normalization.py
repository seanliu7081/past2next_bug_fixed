"""Normalization must depend only on the episodes selected for training."""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from oat.common.replay_buffer import ReplayBuffer
from oat.common.seq_sampler import get_val_mask
from oat.dataset.base_dataset import make_train_validation_datasets
from oat.dataset.prt_dataset import PRTDataset
from oat.dataset.zarr_dataset import ZarrDataset
from oat.dataset.zarr_dataset_with_past import ZarrDatasetWithPastAction
from oat.dataset.zarr_dataset_with_prev_window import ZarrDatasetWithPrevWindow


DATASETS = [ZarrDataset, ZarrDatasetWithPastAction, ZarrDatasetWithPrevWindow]
LENGTHS = (4, 7, 5, 9, 6, 8)
SEED = 42
VAL_RATIO = 1 / 3
ACTION_KEY = "commands"


@pytest.fixture
def make_store(tmp_path):
    stores = []

    def create(holdout_value=10_000.0):
        validation = get_val_mask(len(LENGTHS), VAL_RATIO, SEED)
        replay = ReplayBuffer.create_empty_numpy()
        episodes = []
        for index, length in enumerate(LENGTHS):
            values = index * 7 + np.arange(length, dtype=np.float32) ** 2
            actions = values[:, None] + np.arange(7, dtype=np.float32)[None, :] / 4
            state = values[:, None] * np.array([1, -2, 3], dtype=np.float32)
            rgb = (np.arange(length * 18).reshape(length, 2, 3, 3) % 61 + index).astype(np.uint8)
            if validation[index]:
                actions = actions + holdout_value
                state = state - holdout_value
                rgb.fill(255 if holdout_value > 0 else 0)
            episode = {ACTION_KEY: actions, "state": state, "rgb": rgb}
            episodes.append(episode)
            replay.add_episode(episode)
        path = tmp_path / f"normalization_{len(stores)}.zarr"
        replay.save_to_path(str(path))
        stores.append(path)
        return str(path), episodes

    return create


def make_dataset(dataset_type, path, **kwargs):
    settings = dict(zarr_path=path, action_key=ACTION_KEY, n_action_steps=4,
                    seed=SEED, val_ratio=VAL_RATIO)
    if dataset_type is ZarrDataset:
        settings.update(n_obs_steps=0, obs_keys=[])
    else:
        settings.update(n_obs_steps=2, obs_keys=["state", "rgb"], past_n=3,
                        history_padding="zero", return_history_validity=True)
    if dataset_type is ZarrDatasetWithPrevWindow:
        settings["n_exec_steps"] = 2
    settings.update(kwargs)
    return dataset_type(**settings)


def assert_training_statistics(normalizer, episodes, episode_mask, obs_keys, mode):
    fields = {"action": ACTION_KEY, **{key: key for key in obs_keys}}
    assert set(normalizer.get_input_stats()) == set(fields)
    for output_key, source_key in fields.items():
        expected = np.concatenate([
            episode[source_key] for episode, selected in zip(episodes, episode_mask)
            if selected
        ])
        expected = torch.as_tensor(expected, dtype=torch.float32).reshape(-1, expected.shape[-1])
        stats = normalizer[output_key].get_input_stats()
        torch.testing.assert_close(stats["min"], expected.amin(dim=0))
        torch.testing.assert_close(stats["max"], expected.amax(dim=0))
        torch.testing.assert_close(stats["mean"], expected.mean(dim=0))
        torch.testing.assert_close(stats["std"], expected.std(dim=0))
        normalized = normalizer[output_key].normalize(expected)
        if mode == "limits":
            torch.testing.assert_close(normalized.amin(dim=0), -torch.ones(expected.shape[-1]))
            torch.testing.assert_close(normalized.amax(dim=0), torch.ones(expected.shape[-1]))
        else:
            torch.testing.assert_close(normalized.mean(dim=0), torch.zeros(expected.shape[-1]),
                                       atol=1e-6, rtol=0)
            torch.testing.assert_close(normalized.std(dim=0), torch.ones(expected.shape[-1]))


@pytest.mark.parametrize("dataset_type", DATASETS, ids=["tokenizer", "past", "previous_window"])
@pytest.mark.parametrize("mode", ["limits", "gaussian"])
@pytest.mark.parametrize("max_train_episodes", [None, 2], ids=["full_train_split", "capped_train_split"])
def test_training_statistics_are_shared_with_validation_view(
        make_store, dataset_type, mode, max_train_episodes):
    path, episodes = make_store()
    dataset = make_dataset(dataset_type, path, max_train_episodes=max_train_episodes)
    holdout = get_val_mask(len(LENGTHS), VAL_RATIO, SEED)
    assert not dataset.train_mask[holdout].any()
    assert int(dataset.train_mask.sum()) == (max_train_episodes or int((~holdout).sum()))
    validation = dataset.get_validation_dataset()
    np.testing.assert_array_equal(validation.train_mask, ~dataset.train_mask)
    # Unequal lengths require weighting each real training frame once, without
    # padded history/future windows, held-out rows, or episodes excluded by the cap.
    assert len(dataset) == np.array(LENGTHS)[dataset.train_mask].sum()
    for view in (dataset, validation):
        assert_training_statistics(view.get_normalizer(mode=mode), episodes,
                                   dataset.train_mask, dataset.obs_keys, mode)


@pytest.mark.parametrize("dataset_type", DATASETS, ids=["tokenizer", "past", "previous_window"])
@pytest.mark.parametrize("mode", ["limits", "gaussian"])
def test_changing_holdout_values_does_not_change_normalization(make_store, dataset_type, mode):
    original_path, _ = make_store(holdout_value=10_000.0)
    changed_path, _ = make_store(holdout_value=-20_000.0)
    original = make_dataset(dataset_type, original_path)
    changed = make_dataset(dataset_type, changed_path)
    original_state = original.get_normalizer(mode=mode).state_dict()
    for view in (changed, changed.get_validation_dataset()):
        changed_state = view.get_normalizer(mode=mode).state_dict()
        assert original_state.keys() == changed_state.keys()
        for key in original_state:
            torch.testing.assert_close(original_state[key], changed_state[key], rtol=0, atol=0)


@pytest.mark.parametrize("dataset_type", DATASETS, ids=["tokenizer", "past", "previous_window"])
@pytest.mark.parametrize("mode", ["limits", "gaussian"])
def test_no_holdout_preserves_all_data_fit_even_from_empty_validation_view(
        make_store, dataset_type, mode):
    path, episodes = make_store()
    dataset = make_dataset(dataset_type, path, val_ratio=0.0)
    validation = dataset.get_validation_dataset()
    assert dataset.train_mask.all()
    assert len(validation) == 0
    for view in (dataset, validation):
        assert_training_statistics(view.get_normalizer(mode=mode), episodes,
                                   dataset.train_mask, dataset.obs_keys, mode)


@pytest.mark.parametrize("dataset_type", [PRTDataset, ZarrDatasetWithPrevWindow])
def test_separate_validation_store_uses_all_its_episodes(make_store, dataset_type):
    train_path, train_episodes = make_store(holdout_value=0.0)
    validation_path, validation_episodes = make_store(holdout_value=5_000.0)
    settings = dict(zarr_path=train_path, action_key=ACTION_KEY, past_n=2,
                    seed=SEED, val_ratio=VAL_RATIO, max_train_episodes=2)
    if dataset_type is PRTDataset:
        settings["horizon"] = 1
    else:
        settings.update(n_action_steps=1, n_exec_steps=1, n_obs_steps=1, obs_keys=["state"])
    task = OmegaConf.create({
        "dataset": {"_target_": f"{dataset_type.__module__}.{dataset_type.__name__}", **settings},
        "validation_zarr_path": validation_path,
    })
    train, validation = make_train_validation_datasets(task)
    assert train.train_mask.sum() == 2
    assert validation.train_mask.all()
    expected_actions = np.concatenate([ep[ACTION_KEY] for ep in validation_episodes])
    actual_actions = np.stack([validation[i]["action"][0].numpy() for i in range(len(validation))])
    np.testing.assert_array_equal(actual_actions, expected_actions)
    if dataset_type is PRTDataset:
        reference = PRTDataset(**{**settings, "val_ratio": 0.0})
        for actual, expected in zip(train.fit_prior(), reference.fit_prior()):
            torch.testing.assert_close(actual, expected)
    else:
        assert_training_statistics(train.get_normalizer(), train_episodes,
                                   train.train_mask, ["state"], mode="limits")

    task.validation_zarr_path = train_path
    with pytest.raises(ValueError, match="separate validation store"):
        make_train_validation_datasets(task)
