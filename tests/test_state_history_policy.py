"""CPU integration tests for the independent continuous-history policy."""

import contextlib
import copy
from pathlib import Path

import dill
import hydra
import pytest
import torch
from hydra import compose, initialize_config_dir
from torch import nn

from oat.common.hydra_util import register_new_resolvers
from oat.model.common.normalizer import LinearNormalizer
from oat.model.diffusion.ema_model import EMAModel
from oat.policy.base_policy import BasePolicy
from oat.policy.past2next_state_history import Past2NextStateHistoryPolicy
from oat.workspace.train_policy import TrainPolicyWorkspace
from test_history_training import TinyTokenizer


SHAPES = {"robot0_eef_pos": [3], "robot0_eef_quat": [4], "robot0_gripper_qpos": [2]}
META = {"action": {"shape": [7]}, "obs": {
    key: {"shape": shape, "type": "state"} for key, shape in SHAPES.items()}}
ROOT = Path(__file__).resolve().parents[1]


class PhysicalObservationEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(9, 16)

    def modalities(self):
        return ["state"]

    def output_feature_dim(self):
        return 16

    def set_normalizer(self, normalizer):
        pass

    def forward(self, obs):
        return self.project(torch.cat([obs[key] for key in SHAPES], dim=-1))


def policy_config(history_steps=8, stride=8):
    return {
        "_target_": "oat.policy.past2next_state_history.Past2NextStateHistoryPolicy",
        "shape_meta": META,
        "obs_encoder": {"_target_": "test_state_history_policy.PhysicalObservationEncoder"},
        "action_tokenizer": {"_target_": "test_history_training.TinyTokenizer"},
        "n_obs_steps": 2, "n_action_steps": stride, "past_n": history_steps - 1,
        "state_history_steps": history_steps, "embed_dim": 16, "n_layers": 1,
        "n_heads": 2, "dropout": 0.1, "history_embed_dim": 16,
        "history_n_heads": 2, "history_n_layers": 1, "history_dropout": 0.1,
        "temperature": 0, "topk": 3, "self_past_p": 1,
        "self_past_warmup_steps": 0, "self_past_schedule": "optimizer_step",
    }


def make_policy(history_steps=8, stride=8):
    policy = hydra.utils.instantiate(policy_config(history_steps, stride))
    normalizer = LinearNormalizer()
    normalizer.fit({key: torch.tensor([-1., 1.])[:, None].expand(2, shape[0])
                    for key, shape in {**SHAPES, "action": [7]}.items()})
    policy.set_normalizer(normalizer)
    return policy


def make_obs(history_steps=8, batch_size=2, valid_steps=None):
    obs = {key: torch.randn(batch_size, 2, *shape) for key, shape in SHAPES.items()}
    for key, shape in SHAPES.items():
        value = torch.randn(batch_size, history_steps, *shape)
        if key.endswith("_quat"):
            value /= value.norm(dim=-1, keepdim=True)
        obs["state_history__" + key] = value
    mask = torch.ones(batch_size, history_steps, dtype=torch.bool)
    if valid_steps is not None:
        mask[:, :-valid_steps] = False
        for key in SHAPES:
            obs["state_history__" + key][~mask] = 0
    obs["state_history_valid"] = mask
    return obs


def make_batch(history_steps=8):
    return {"obs": make_obs(history_steps), "prev_obs": make_obs(history_steps),
            "action": torch.randn(2, 16, 7),
            "past_action": torch.randn(2, history_steps - 1, 7),
            "prev_past_action": torch.randn(2, history_steps - 1, 7),
            "past_action_valid": torch.ones(2, history_steps - 1, dtype=torch.bool),
            "prev_window_valid": torch.ones(2, dtype=torch.bool)}


@pytest.mark.parametrize("history_steps", [8, 16])
@pytest.mark.parametrize("mixed_precision", [False, True])
def test_active_self_past_backward_optimizer_and_ema(history_steps, mixed_precision):
    policy = make_policy(history_steps).train()
    batch = make_batch(history_steps)
    optimizer = policy.get_optimizer(1e-3, 1e-3, 1e-4, (0.9, 0.95))
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(parameters) == len({id(p) for p in parameters})
    assert {id(p) for p in parameters} == {id(p) for p in policy.parameters() if p.requires_grad}
    ema_policy = copy.deepcopy(policy)
    ema = EMAModel(ema_policy)
    before = policy.history_encoder.input_projection[0].weight.detach().clone()
    autocast = torch.autocast("cpu", dtype=torch.bfloat16) if mixed_precision else contextlib.nullcontext()
    with autocast:
        loss = policy(batch)
    loss.backward()
    assert torch.isfinite(loss)
    missing = [name for name, p in policy.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing
    assert all(torch.isfinite(p.grad).all() for p in parameters)
    assert policy.history_encoder.training and policy.model.training
    assert not policy.action_tokenizer.training
    assert all(p.grad is None for p in policy.action_tokenizer.parameters())
    optimizer.step()
    policy.on_optimizer_step()
    ema.step(policy)
    assert policy.self_past_step == 1
    assert not torch.equal(before, policy.history_encoder.input_projection[0].weight)
    assert torch.isfinite(ema_policy.history_encoder.input_projection[0].weight).all()


@pytest.mark.parametrize("history_steps,stride", [(8, 8), (16, 8), (16, 4)])
def test_previous_generation_uses_temporal_history_and_exec_prefix(history_steps, stride, monkeypatch):
    policy = make_policy(history_steps, stride).train()
    batch = make_batch(history_steps)
    seen = []
    original = policy.history_encoder.forward

    def capture(history, mask, past, normalizer):
        seen.append((policy.history_encoder.training, history["robot0_eef_pos"].clone()))
        return original(history, mask, past, normalizer)

    monkeypatch.setattr(policy.history_encoder, "forward", capture)
    generated = policy._generate_prev_past(batch)
    predicted = torch.arange(stride).float()[None, :, None].expand(2, stride, 7)
    expected = torch.cat((batch["prev_past_action"], predicted), dim=1)[:, -(history_steps - 1):]
    torch.testing.assert_close(generated, expected)
    assert not generated.is_inference() and not generated.requires_grad
    assert seen[0][0] is False and policy.history_encoder.training
    torch.testing.assert_close(seen[0][1], batch["prev_obs"]["state_history__robot0_eef_pos"])


def test_generation_and_training_use_same_selected_commands(monkeypatch):
    policy = make_policy().train()
    batch = make_batch()
    expected = torch.full_like(batch["past_action"], 0.37)
    monkeypatch.setattr(policy, "_maybe_self_past", lambda *args, **kwargs: expected)
    seen = []
    handle = policy.raw_proj.register_forward_pre_hook(lambda module, args: seen.append(args[0].clone()))
    original = policy.history_encoder.forward

    def capture(history, mask, past, normalizer):
        seen.append(past.clone())
        return original(history, mask, past, normalizer)

    monkeypatch.setattr(policy.history_encoder, "forward", capture)
    policy(batch)
    handle.remove()
    assert len(seen) == 2
    for value in seen:
        torch.testing.assert_close(value, policy.action_normalizer["action"].normalize(expected))


def test_stateful_predictions_require_ack_and_offline_validation_is_stateless():
    policy = make_policy().eval()
    batch = make_batch()
    with torch.inference_mode():
        prediction = policy.predict_action(batch["obs"])
    assert prediction["action"].shape == (2, 8, 7)
    assert prediction["action_pred"].shape == (2, 16, 7)
    assert torch.count_nonzero(policy._past_buffer) == 0
    with pytest.raises(RuntimeError, match="pending"):
        policy.predict_action(batch["obs"])
    snapshot = policy._past_buffer.clone()
    with torch.inference_mode():
        TrainPolicyWorkspace._predict_validation_action(policy, batch)
    torch.testing.assert_close(snapshot, policy._past_buffer)
    assert policy._pending_execution_steps == 8
    # Execution layer changed commands; record only its confirmed prefix.
    commands = torch.full((2, 8, 7), float("nan"))
    commands[0, :3] = 0.25
    policy.record_executed_actions(commands, [3, 0])
    torch.testing.assert_close(policy._past_buffer[0, -3:], commands[0, :3])
    assert torch.count_nonzero(policy._past_buffer[1]) == 0
    assert policy._pending_execution_steps is None
    policy.reset()
    assert policy._past_buffer is None


def test_new_conditions_respond_to_earlier_state_but_ignore_invalid_prefix():
    policy = make_policy().eval()
    obs = make_obs(valid_steps=3)
    past = torch.zeros(2, 7, 7)
    with torch.no_grad():
        condition = policy._condition(obs, past)
        assert condition.shape == (2, 2 + 2 + 7 + 4, 16)
        assert policy.model.cond_pos_emb.shape[1] == condition.shape[1]
        poisoned = copy.deepcopy(obs)
        for key in SHAPES:
            poisoned["state_history__" + key][:, :-3] = float("nan")
        torch.testing.assert_close(condition, policy._condition(poisoned, past))
        changed = copy.deepcopy(obs)
        changed["state_history__robot0_eef_pos"][:, -3] += 0.5
        new_condition = policy._condition(changed, past)
        torch.testing.assert_close(condition[:, :-4], new_condition[:, :-4])
        assert not torch.equal(condition[:, -4:], new_condition[:, -4:])


def test_runner_float_masks_missing_metadata_and_dummy_observations():
    policy = make_policy().eval()
    obs = make_obs(valid_steps=1)
    past = torch.zeros(2, 7, 7)
    with torch.no_grad():
        expected = policy._condition(obs, past)
        obs["state_history_valid"] = obs["state_history_valid"].float()
        torch.testing.assert_close(expected, policy._condition(obs, past))
    obs["state_history_valid"][0, 0] = 0.2
    with pytest.raises(ValueError, match="zero/one"):
        policy._condition(obs, past)
    with pytest.raises(KeyError, match="Missing state history"):
        policy.predict_action({key: value for key, value in obs.items() if key in SHAPES},
                              past_actions=past)
    with torch.inference_mode():
        result = policy.predict_action(policy.create_dummy_observation(2), past_actions=past)
    assert result["action"].shape == (2, 8, 7)


@pytest.mark.parametrize("history_steps", [8, 16])
def test_saved_config_and_weights_reload_without_runtime_history(tmp_path, history_steps):
    from omegaconf import OmegaConf
    policy = make_policy(history_steps).eval()
    policy.set_self_past_step(123)
    batch = make_batch(history_steps)
    with torch.inference_mode():
        expected = policy.predict_action(batch["obs"])
    checkpoint = tmp_path / "history.ckpt"
    cfg = OmegaConf.create({"policy": policy_config(history_steps), "training": {"use_ema": True}})
    torch.save({"cfg": cfg, "state_dicts": {"ema_model": policy.state_dict()}},
               checkpoint, pickle_module=dill)
    loaded = BasePolicy.from_checkpoint(str(checkpoint))
    assert isinstance(loaded, Past2NextStateHistoryPolicy)
    assert loaded.self_past_step == 123
    assert loaded._past_buffer is None and loaded._pending_execution_steps is None
    with torch.inference_mode():
        actual = loaded.predict_action(batch["obs"])
    torch.testing.assert_close(actual["action_pred"], expected["action_pred"])
    torch.testing.assert_close(loaded.history_encoder.summary_queries,
                               policy.history_encoder.summary_queries)


@pytest.mark.parametrize("history_steps", [8, 16])
def test_config_resolves_consistent_dataset_policy_runner_histories(history_steps):
    register_new_resolvers()
    with initialize_config_dir(config_dir=str(ROOT / "oat/config"), version_base=None):
        cfg = compose(config_name="experimental/train_past2next_state_history", overrides=[
            "policy.action_tokenizer.checkpoint=/tmp/example.ckpt",
            f"state_history_steps={history_steps}",
        ])
    assert cfg.past_n == history_steps - 1
    assert cfg.policy.state_history_steps == cfg.task.policy.dataset.state_history_steps == cfg.task.policy.env_runner.state_history_steps == history_steps
    assert cfg.policy.past_n == cfg.task.policy.dataset.past_n == history_steps - 1
    assert cfg.n_obs_steps == 2 and cfg.task.policy.dataset.n_obs_steps == 2
    assert cfg.task.policy.dataset.history_padding == "zero"
    assert cfg.task.policy.env_runner.protocol == "corrected"
    assert cfg.training.use_ema and cfg.training.validate_generated_history
    assert cfg.training.num_epochs == 2001 and cfg.training.rollout_every == 100
    assert cfg.training.val_every == 1
    assert cfg.task.policy.dataset.val_ratio == 0.1
