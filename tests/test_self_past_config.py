"""The standalone self-past recipe must activate the corrected history path."""

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import get_class, instantiate
import numpy as np
import pytest
import torch
from torch.utils.data import default_collate

from oat.common.replay_buffer import ReplayBuffer
from oat.model.common.normalizer import LinearNormalizer
from test_history_training import TinyObservationEncoder, TinyTokenizer, make_batch


def configuration(overrides=()):
    directory = str(Path(__file__).resolve().parents[1] / "oat/config")
    with initialize_config_dir(config_dir=directory, version_base=None):
        return compose(config_name="train_past2next_self_past", overrides=list(overrides))


def tiny_policy(cfg):
    fields = ("n_action_steps", "n_obs_steps", "past_n", "temperature", "topk",
              "self_past_p", "self_past_warmup_steps", "self_past_temperature",
              "self_past_topk", "self_past_schedule", "self_past_ramp_steps")
    policy = get_class(cfg.policy._target_)(
        **{key: cfg.policy[key] for key in fields},
        shape_meta={"action": {"shape": [7]},
                    "obs": {"state": {"shape": [7], "type": "state"}}},
        obs_encoder=TinyObservationEncoder(), action_tokenizer=TinyTokenizer(),
        embed_dim=8, n_layers=1, n_heads=2, dropout=0,
    )
    normalizer = LinearNormalizer()
    normalizer.fit({"action": torch.tensor([-1., 1.])[:, None].expand(2, 7)})
    policy.set_normalizer(normalizer)
    return policy


@pytest.mark.parametrize("task,action_dim", [
    ("libero/libero10_with_prev_window", 7),
    ("robocasa/sink3_with_prev_window", 12),
])
def test_corrected_recipe_keeps_hyperparameters_and_task_overrides(task, action_dim):
    cfg = configuration([f"task/policy={task}"])
    assert cfg.task.policy.dataset.history_padding == "zero"
    assert cfg.task.policy.dataset.return_history_validity is True
    assert cfg.task.policy.env_runner.protocol == "corrected"
    assert cfg.shape_meta.action.shape[0] == action_dim
    assert cfg.policy.self_past_schedule == "optimizer_step"
    assert cfg.policy.self_past_ramp_steps == 0
    assert cfg.policy.self_past_warmup_steps == 1000
    assert cfg.policy.self_past_p == 0.5
    assert cfg.policy.temperature == 1.0 and cfg.policy.topk == 10
    assert cfg.policy.self_past_temperature is None and cfg.policy.self_past_topk is None
    assert cfg.policy.embed_dim == 256 and cfg.policy.n_layers == cfg.policy.n_heads == 4
    assert cfg.optimizer.policy_lr == 5e-5 and cfg.optimizer.obs_enc_lr == 1e-5


def test_composed_dataset_zeroes_cold_start_and_skips_missing_previous_windows(tmp_path, monkeypatch):
    cfg = configuration()
    replay = ReplayBuffer.create_empty_numpy()
    for episode in range(10):
        values = (100 * (episode + 1) + np.arange(12)).astype(np.float32)
        actions = np.repeat(values[:, None], 7, axis=1)
        replay.add_episode({"action": actions, "state": actions.copy()})
    path = tmp_path / "history.zarr"
    replay.save_to_path(str(path))
    dataset = instantiate(
        cfg.task.policy.dataset, zarr_path=str(path), obs_keys=["state"], val_ratio=0.1,
    )
    assert dataset.train_mask.sum() == 9
    for start in (0, 12):
        first = dataset[start]
        assert first["episode_step"] == 0
        assert torch.equal(first["past_action"], torch.zeros(7, 7))
        assert not first["past_action_valid"].any()
        assert not first["prev_past_action_valid"].any()
        assert not first["prev_window_valid"]
    assert not dataset[7]["prev_window_valid"]
    assert dataset[8]["prev_window_valid"]

    policy = tiny_policy(cfg)
    generated_batch_sizes = []

    def generate(previous):
        generated_batch_sizes.append(previous["prev_past_action"].shape[0])
        return torch.full_like(previous["prev_past_action"], 9)

    monkeypatch.setattr(policy, "_generate_prev_past", generate)
    batch = default_collate([dataset[0], dataset[8]])
    history = policy._maybe_self_past(batch, batch["past_action"], probability=1)
    assert generated_batch_sizes == [1]
    assert torch.equal(history[0], torch.zeros(7, 7))
    assert torch.equal(history[1], torch.full((7, 7), 9.0))
    unavailable = default_collate([dataset[0], dataset[7]])
    history = policy._maybe_self_past(unavailable, unavailable["past_action"], probability=1)
    assert generated_batch_sizes == [1]
    torch.testing.assert_close(history, unavailable["past_action"])


def test_composed_schedule_survives_save_load_and_validation(tmp_path):
    cfg = configuration()
    policy = tiny_policy(cfg).train()
    batch = make_batch()
    policy(batch, history_mode="expert")
    assert policy.self_past_step == 0
    policy.set_self_past_step(999)
    assert policy.self_past_probability() == 0
    policy.on_optimizer_step()
    assert policy.self_past_step == 1000
    assert policy.self_past_probability() == 0.5

    checkpoint = tmp_path / "schedule.pt"
    torch.save(policy.state_dict(), checkpoint)
    restored = tiny_policy(cfg)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    assert restored.self_past_step == 1000
    assert restored.self_past_probability() == 0.5
    restored.eval()
    # Default validation must use expert history, even after the warmup ends.
    restored(batch)
    restored.on_optimizer_step()
    assert restored.self_past_step == 1000
    restored.train()
    restored.on_optimizer_step()
    assert restored.self_past_step == 1001
