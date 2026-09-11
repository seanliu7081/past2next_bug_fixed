"""Small CPU checks for episode alignment, curriculum state, and offline history."""
import io
from types import SimpleNamespace

import dill
import numpy as np
import pytest
import torch
from torch import nn

from oat.common.replay_buffer import ReplayBuffer
from oat.dataset.zarr_dataset_with_past import ZarrDatasetWithPastAction
from oat.dataset.zarr_dataset_with_prev_window import ZarrDatasetWithPrevWindow
from oat.model.common.normalizer import LinearNormalizer
from oat.policy.past2next_self_past import Past2NextSelfPastPolicy
from oat.workspace.train_policy import TrainPolicyWorkspace


@pytest.fixture
def ramp_path(tmp_path):
    replay = ReplayBuffer.create_empty_numpy()
    for offset in (100, 200):
        actions = np.repeat((offset + np.arange(24))[:, None], 7, axis=1).astype(np.float32)
        replay.add_episode({"action": actions, "state": actions.copy()})
    path = tmp_path / "ramp.zarr"
    replay.save_to_path(str(path))
    return str(path)


@pytest.mark.parametrize("dataset_type,stride", [
    (ZarrDatasetWithPastAction, None),
    (ZarrDatasetWithPrevWindow, 4),
    (ZarrDatasetWithPrevWindow, 8),
])
def test_history_alignment_and_legacy_padding(ramp_path, dataset_type, stride):
    kwargs = dict(zarr_path=ramp_path, obs_keys=["state"], n_obs_steps=2,
                  n_action_steps=16, past_n=7)
    if stride is not None:
        kwargs["n_exec_steps"] = stride
    aligned = dataset_type(**kwargs, history_padding="zero", return_history_validity=True)
    legacy = dataset_type(**kwargs)
    for index in (0, 3, 8, 23, 24, 27, 32, 47):
        sample, old = aligned[index], legacy[index]
        step = index % 24
        base = 100 if index < 24 else 200
        assert sample["episode_step"].item() == step
        expected = [base + t if t >= 0 else 0 for t in range(step - 7, step)]
        torch.testing.assert_close(sample["past_action"][:, 0], torch.tensor(expected).float())
        assert sample["past_action_valid"].tolist() == [t >= 0 for t in range(step - 7, step)]
        # Observations and supervised future chunks must be unchanged, even at boundaries.
        torch.testing.assert_close(sample["obs"]["state"], old["obs"]["state"])
        torch.testing.assert_close(sample["action"], old["action"])
        assert sample["action"][0, 0].item() == base + step
        assert "past_action_valid" not in old
        assert "episode_step" not in old
        if step == 0:
            assert torch.all(old["past_action"] == base)
            assert torch.all(sample["past_action"] == 0)
            assert torch.all(sample["obs"]["state"] == base)
        if stride is not None:
            assert sample["prev_window_valid"].item() == (step >= stride)
            previous = [base + t if t >= 0 else 0
                        for t in range(step - stride - 7, step - stride)]
            torch.testing.assert_close(sample["prev_past_action"][:, 0], torch.tensor(previous).float())


class TinyObservationEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(7, 8)

    def modalities(self):
        return ["state"]

    def output_feature_dim(self):
        return 8

    def set_normalizer(self, normalizer):
        pass

    def forward(self, obs):
        return self.project(obs["state"])


class TinyTokenizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.placeholder = nn.Parameter(torch.zeros(1))
        self.register_buffer("normalizer_sentinel", torch.tensor(3.0))
        self.quantizer = SimpleNamespace(codebook_size=8)
        self.latent_horizon = 2

    def tokenize(self, actions):
        return torch.zeros(actions.shape[0], 2, dtype=torch.long, device=actions.device)

    def detokenize(self, tokens):
        return torch.arange(16, device=tokens.device).float()[None, :, None].expand(tokens.shape[0], 16, 7)


def make_policy(schedule="optimizer_step", stride=8):
    policy = Past2NextSelfPastPolicy(
        shape_meta={"action": {"shape": [7]}, "obs": {"state": {"shape": [7], "type": "state"}}},
        obs_encoder=TinyObservationEncoder(), action_tokenizer=TinyTokenizer(),
        n_action_steps=stride, n_obs_steps=2, past_n=7,
        embed_dim=8, n_layers=1, n_heads=2, dropout=0,
        self_past_p=0.5, self_past_warmup_steps=2, self_past_ramp_steps=4,
        self_past_schedule=schedule,
    )
    normalizer = LinearNormalizer()
    normalizer.fit({"action": torch.tensor([-1., 1.])[:, None].expand(2, 7)})
    policy.set_normalizer(normalizer)
    return policy


def make_batch(batch_size=2):
    return {"action": torch.zeros(batch_size, 16, 7),
            "obs": {"state": torch.zeros(batch_size, 2, 7)},
            "past_action": torch.zeros(batch_size, 7, 7)}


def test_scheduler_persistence_and_validation_does_not_advance():
    policy = make_policy()
    batch = make_batch()
    policy.train()
    policy(batch, history_mode="expert")
    policy(batch, history_mode="expert")
    assert policy.self_past_step == 0  # accumulation / forward passes are not updates
    for _ in range(4):
        policy.on_optimizer_step()
    assert policy.self_past_step == 4
    assert policy.self_past_probability() == 0.25
    policy.eval()
    policy(batch)
    policy.on_optimizer_step()
    assert policy.self_past_step == 4
    state = io.BytesIO()
    torch.save(policy.state_dict(), state)
    state.seek(0)
    restored = make_policy()
    restored.load_state_dict(torch.load(state, weights_only=True))
    assert restored.self_past_step == 4
    assert restored.self_past_probability() == 0.25
    restored.on_optimizer_step()
    assert restored.self_past_step == 5
    assert restored.self_past_probability() == 0.375


def test_historical_state_dict_stays_compatible():
    legacy = make_policy("legacy")
    assert not any("optimizer_step" in key for key in legacy.state_dict())
    make_policy("legacy").load_state_dict(legacy.state_dict(), strict=True)
    corrected = make_policy()
    corrected.load_state_dict(legacy.state_dict(), strict=True)
    assert corrected.self_past_step == 0
    legacy.eval()
    legacy(make_batch(), history_mode="expert")
    assert legacy.self_past_step == 0


def test_invalid_previous_windows_are_not_generated(monkeypatch):
    policy = make_policy()
    seen = []

    def generate(batch):
        seen.append(batch["prev_past_action"].shape[0])
        return torch.full_like(batch["prev_past_action"], 9)

    monkeypatch.setattr(policy, "_generate_prev_past", generate)
    batch = make_batch()
    batch.update(prev_obs=batch["obs"], prev_past_action=batch["past_action"],
                 prev_window_valid=torch.tensor([False, True]),
                 past_action_valid=torch.tensor([[False] * 7, [True] * 7]))
    mixed = policy._maybe_self_past(batch, batch["past_action"], probability=1)
    assert seen == [1]
    assert torch.all(mixed[0] == 0)
    assert torch.all(mixed[1] == 9)
    batch["prev_window_valid"].zero_()
    policy._maybe_self_past(batch, batch["past_action"], probability=1)
    assert seen == [1]


@pytest.mark.parametrize("stride", [4, 8])
def test_explicit_validation_history_does_not_leak_and_rollout_buffer_updates(stride):
    policy = make_policy(stride=stride).eval()
    batch = make_batch()
    policy.predict_action(batch["obs"], temperature=0, past_actions=batch["past_action"])
    assert policy._past_buffer is None
    policy.predict_action(batch["obs"], temperature=0)
    expected = torch.cat([torch.zeros(max(7 - stride, 0)), torch.arange(max(stride - 7, 0), stride).float()])
    torch.testing.assert_close(policy._past_buffer[0, :, 0], expected)
    saved = policy._past_buffer
    TrainPolicyWorkspace._predict_validation_action(policy, batch)
    assert policy._past_buffer is saved
    torch.testing.assert_close(saved[0, :, 0], expected)


def test_dynamic_features_use_same_normalized_history():
    policy = make_policy()
    policy.acc_proj = policy.jerk_proj = policy.raw_proj = nn.Identity()
    history = torch.arange(49).reshape(1, 7, 7).float()
    history[:, :3] = 0
    normalized = policy.action_normalizer["action"].normalize(history)
    condition = policy._build_condition(torch.zeros(1, 2, 7), history)
    torch.testing.assert_close(condition[:, 2], normalized[:, -1] - normalized[:, -2])
    torch.testing.assert_close(condition[:, 3], normalized[:, -1] - 2 * normalized[:, -2] + normalized[:, -3])
    torch.testing.assert_close(condition[:, 4:], normalized)


def test_finetune_initialization_resets_progress_and_preserves_tokenizer(tmp_path):
    source = make_policy()
    source.set_self_past_step(12)
    source.action_tokenizer.normalizer_sentinel.fill_(17)
    path = tmp_path / "source.ckpt"
    torch.save({"state_dicts": {"ema_model": source.state_dict()}}, path, pickle_module=dill)
    workspace = object.__new__(TrainPolicyWorkspace)
    workspace.model = make_policy()
    workspace.ema_model = make_policy()
    workspace._initialize_policy_weights(path, weights="ema")
    for policy in (workspace.model, workspace.ema_model):
        assert policy.self_past_step == 0
        assert policy.action_tokenizer.normalizer_sentinel.item() == 17


def test_spatial_resize_only_regenerates_explicit_coordinate_buffers(tmp_path, capsys):
    from robomimic.models.base_nets import SpatialSoftmax

    source = make_policy()
    source.obs_encoder.spatial_pool = SpatialSoftmax([4, 3, 3], num_kp=2)
    source.obs_encoder.spatial_alias = source.obs_encoder.spatial_pool
    source.action_tokenizer.normalizer_sentinel.fill_(29)
    source_state = source.state_dict()
    path = tmp_path / "spatial.ckpt"
    torch.save({"state_dicts": {"ema_model": source_state}}, path, pickle_module=dill)

    workspace = object.__new__(TrainPolicyWorkspace)
    workspace.model = make_policy()
    workspace.model.obs_encoder.spatial_pool = SpatialSoftmax([4, 4, 4], num_kp=2)
    workspace.model.obs_encoder.spatial_alias = workspace.model.obs_encoder.spatial_pool
    workspace.ema_model = make_policy()
    workspace.ema_model.obs_encoder.spatial_pool = SpatialSoftmax([4, 4, 4], num_kp=2)
    workspace.ema_model.obs_encoder.spatial_alias = workspace.ema_model.obs_encoder.spatial_pool
    original_grid = workspace.model.obs_encoder.spatial_pool.pos_x.clone()
    with pytest.raises(RuntimeError, match="size mismatch"):
        workspace._initialize_policy_weights(path)
    regenerated = workspace._initialize_policy_weights(path, allow_spatial_resize=True)
    assert regenerated == [f"obs_encoder.{name}.{axis}"
                           for name in ("spatial_alias", "spatial_pool")
                           for axis in ("pos_x", "pos_y")]
    for policy in (workspace.model, workspace.ema_model):
        torch.testing.assert_close(policy.obs_encoder.spatial_pool.pos_x, original_grid)
        assert policy.action_tokenizer.normalizer_sentinel.item() == 29
        for name, param in policy.named_parameters():
            torch.testing.assert_close(param, source_state[name])
    assert "Regenerated SpatialSoftmax buffer obs_encoder.spatial_pool.pos_x" in capsys.readouterr().out

    # An unrelated learned-weight mismatch remains fatal, even with opt-in.
    bad_state = source_state.copy()
    bad_state["obs_encoder.project.weight"] = torch.zeros(1, 1)
    torch.save({"state_dicts": {"ema_model": bad_state}}, path, pickle_module=dill)
    with pytest.raises(ValueError, match="obs_encoder.project.weight"):
        workspace._initialize_policy_weights(path, allow_spatial_resize=True)
