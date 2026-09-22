"""Bounded CPU checks for native real-robot rotation-6D history and gate policy."""

import contextlib
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from torch import nn

from oat.common.hydra_util import register_new_resolvers
from oat.model.common.normalizer import LinearNormalizer
from oat.policy.past2next_state_history_gate_real_robot import (
    Past2NextRealRobotStateHistoryGatePolicy,
    Rotation6DStateActionHistoryEncoder,
)


ROOT = Path(__file__).resolve().parents[1]
SHAPES = {"robot0_eef_pos": [3], "robot0_eef_rot6d": [6], "robot0_gripper_qpos": [1]}
ROT_KEY = "robot0_eef_rot6d"
META = {"action": {"shape": [7]}, "obs": {
    key: {"shape": shape, "type": "state"} for key, shape in SHAPES.items()}}


def rotation(axis, angle):
    c, s = math.cos(angle), math.sin(angle)
    matrices = {
        "x": [[1, 0, 0], [0, c, -s], [0, s, c]],
        "y": [[c, 0, s], [0, 1, 0], [-s, 0, c]],
        "z": [[c, -s, 0], [s, c, 0], [0, 0, 1]],
    }
    return torch.tensor(matrices[axis], dtype=torch.float32)


def rotation_sequence():
    # These orientations do not commute, distinguishing world from local deltas.
    return torch.stack((
        rotation("x", .5),
        rotation("y", .7) @ rotation("x", -.2),
        rotation("z", -.4) @ rotation("y", .3),
        rotation("x", .6) @ rotation("z", .4),
    ))


def encode_rotation(matrix, layout):
    if layout == "rows":
        return torch.cat((matrix[..., 0, :], matrix[..., 1, :]), dim=-1)
    return torch.cat((matrix[..., :, 0], matrix[..., :, 1]), dim=-1)


def make_normalizer(low=-1., high=1.):
    normalizer = LinearNormalizer()
    normalizer.fit({key: torch.tensor([low, high])[:, None].expand(2, shape[0])
                    for key, shape in {**SHAPES, "action": [7]}.items()})
    return normalizer


def make_history(layout, batch_size=1):
    return {
        "robot0_eef_pos": torch.arange(12).float().reshape(1, 4, 3).expand(batch_size, -1, -1).clone(),
        ROT_KEY: encode_rotation(rotation_sequence(), layout)[None].expand(batch_size, -1, -1).clone(),
        "robot0_gripper_qpos": torch.linspace(0, 1, 4).reshape(1, 4, 1).expand(batch_size, -1, -1).clone(),
    }


def make_encoder(layout):
    return Rotation6DStateActionHistoryEncoder(
        state_shapes=SHAPES, action_dim=7, history_steps=4,
        output_dim=16, embed_dim=16, n_heads=2, n_layers=1,
        n_summary_tokens=2, dropout=0, rotation_6d_layout=layout,
    ).eval()


@pytest.mark.parametrize("layout", ["rows", "columns"])
def test_absolute_and_world_relative_rotation_geometry_bypasses_scalar_normalization(layout):
    encoder = make_encoder(layout)
    states = make_history(layout)
    raw = states[ROT_KEY].clone()
    # Gram-Schmidt must recover the same rotations from scaled, nonorthogonal axes.
    states[ROT_KEY][..., :3] = 2.7 * raw[..., :3]
    states[ROT_KEY][..., 3:] = .4 * raw[..., 3:] + .2 * raw[..., :3]
    features, padding = encoder.build_features(
        states, torch.ones(1, 4, dtype=torch.bool), torch.zeros(1, 3, 7),
        make_normalizer(1., 5.),
    )
    matrices = rotation_sequence()
    expected_absolute = encode_rotation(matrices, "columns")[None]
    expected_relative = matrices[1:] @ matrices[:-1].transpose(-1, -2)
    wrong_local = matrices[:-1].transpose(-1, -2) @ matrices[1:]
    assert not torch.allclose(expected_relative, wrong_local)
    expected_delta = torch.cat((torch.zeros(1, 6), encode_rotation(expected_relative, "columns")))[None]
    torch.testing.assert_close(features[..., encoder.feature_slices[f"absolute/{ROT_KEY}"]],
                               expected_absolute, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(features[..., encoder.feature_slices[f"delta/{ROT_KEY}"]],
                               expected_delta, atol=1e-6, rtol=1e-5)
    assert not padding.any()


@pytest.mark.parametrize("layout", ["rows", "columns"])
def test_invalid_nan_padding_cannot_contaminate_features_summary_or_gradients(layout):
    encoder = make_encoder(layout)
    clean = make_history(layout)
    valid = torch.tensor([[False, False, True, True]])
    poisoned = {}
    for key, value in clean.items():
        clean[key] = value.clone()
        clean[key][:, :2] = 0
        poisoned[key] = value.clone()
        poisoned[key][:, :2] = torch.nan
        poisoned[key].requires_grad_()
    commands = torch.zeros(1, 3, 7)
    poisoned_commands = commands.clone()
    poisoned_commands[:, :2] = torch.nan
    poisoned_commands.requires_grad_()
    normalizer = make_normalizer()
    expected, _ = encoder.build_features(clean, valid, commands, normalizer)
    actual, padding = encoder.build_features(poisoned, valid, poisoned_commands, normalizer)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(padding, ~valid)
    assert torch.isfinite(actual).all() and torch.count_nonzero(actual[:, :2]) == 0
    assert torch.count_nonzero(actual[:, 2, encoder.feature_slices[f"delta/{ROT_KEY}"]]) == 0
    summary = encoder(poisoned, valid, poisoned_commands, normalizer)
    torch.testing.assert_close(summary, encoder(clean, valid, commands, normalizer))
    summary.square().mean().backward()
    for value in [*poisoned.values(), poisoned_commands]:
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert torch.count_nonzero(value.grad[:, :2]) == 0


@pytest.mark.parametrize("layout", ["rows", "columns"])
@pytest.mark.parametrize("bad_rotation", [
    [0, 0, 0, 0, 1, 0],
    [1, 0, 0, 2, 0, 0],
    [float("nan"), 0, 0, 0, 1, 0],
])
def test_valid_degenerate_or_nonfinite_rotations_are_rejected(layout, bad_rotation):
    encoder = make_encoder(layout)
    states = make_history(layout)
    states[ROT_KEY][:, -1] = torch.tensor(bad_rotation)
    with pytest.raises(ValueError, match="degenerate|collinear|finite"):
        encoder.build_features(states, torch.ones(1, 4, dtype=torch.bool),
                               torch.zeros(1, 3, 7), make_normalizer())


class TinyRealObservationEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(10, 16)

    def modalities(self):
        return ["state"]

    def output_feature_dim(self):
        return 16

    def set_normalizer(self, normalizer):
        pass

    def forward(self, obs):
        return self.project(torch.cat([obs[key] for key in SHAPES], dim=-1))


class TinyRealTokenizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.placeholder = nn.Parameter(torch.zeros(1))
        self.quantizer = SimpleNamespace(codebook_size=8)
        self.latent_horizon = 2

    def tokenize(self, actions):
        return torch.zeros(actions.shape[0], 2, dtype=torch.long, device=actions.device)

    def detokenize(self, tokens):
        return tokens.float().sum(-1)[:, None, None].expand(-1, 16, 7)


def make_policy(layout):
    policy = Past2NextRealRobotStateHistoryGatePolicy(
        shape_meta=META, obs_encoder=TinyRealObservationEncoder(),
        action_tokenizer=TinyRealTokenizer(), n_obs_steps=2, n_action_steps=2,
        past_n=3, state_history_steps=4, embed_dim=16, n_layers=1, n_heads=2,
        dropout=0, history_embed_dim=16, history_n_heads=2, history_n_layers=1,
        history_summary_tokens=2, history_dropout=0, history_gate_hidden_dim=16,
        history_gate_mode="learned", rotation_6d_layout=layout,
        temperature=0, self_past_p=1, self_past_warmup_steps=0,
        self_past_ramp_steps=0, self_past_schedule="optimizer_step",
    )
    policy.set_normalizer(make_normalizer())
    return policy


def make_observation(layout):
    states = make_history(layout, batch_size=2)
    obs = {key: value[:, -2:].clone() for key, value in states.items()}
    obs.update({"state_history__" + key: value for key, value in states.items()})
    obs["state_history_valid"] = torch.ones(2, 4, dtype=torch.bool)
    return obs


@pytest.mark.parametrize("layout", ["rows", "columns"])
@pytest.mark.parametrize("mixed_precision", [False, True])
def test_real_policy_self_past_backward_is_finite_and_tokenizer_stays_frozen(layout, mixed_precision):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71)
        policy = make_policy(layout).train()
        batch = {
            "obs": make_observation(layout), "prev_obs": make_observation(layout),
            "action": torch.randn(2, 16, 7), "past_action": torch.randn(2, 3, 7),
            "prev_past_action": torch.randn(2, 3, 7),
            "past_action_valid": torch.ones(2, 3, dtype=torch.bool),
            "prev_window_valid": torch.ones(2, dtype=torch.bool),
        }
    seen = []
    handle = policy.history_gate.register_forward_pre_hook(
        lambda module, args: seen.append((module.training, torch.is_inference_mode_enabled())))
    amp = torch.autocast("cpu", dtype=torch.bfloat16) if mixed_precision else contextlib.nullcontext()
    try:
        with amp:
            loss = policy(batch, history_mode="generated")
    finally:
        handle.remove()
    assert seen == [(False, True), (True, False)]
    assert torch.isfinite(loss)
    loss.backward()
    for name, parameter in policy.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
    assert policy.history_gate[-1].weight.grad.abs().sum() > 0
    assert policy.history_encoder.input_projection[0].weight.grad.abs().sum() > 0
    assert policy.history_encoder.training and policy.history_gate.training
    assert not policy.action_tokenizer.training
    assert all(parameter.grad is None for parameter in policy.action_tokenizer.parameters())
    assert policy._past_buffer is None and policy._pending_execution_steps is None


@pytest.mark.parametrize("layout", ["rows", "columns"])
def test_real_policy_cached_prediction_preserves_execution_feedback_and_dummy_geometry(layout):
    policy = make_policy(layout).eval()
    assert isinstance(policy.history_encoder, Rotation6DStateActionHistoryEncoder)
    assert policy.state_history_keys == tuple(SHAPES)
    assert not any("quat" in key for key in policy.get_observation_ports())
    dummy = policy.create_dummy_observation(batch_size=2)
    torch.testing.assert_close(dummy["state_history__" + ROT_KEY][:, -1],
                               torch.tensor([[1., 0, 0, 0, 1, 0]]).expand(2, -1))
    cached_inputs = []
    handle = policy.model.tok_emb.register_forward_pre_hook(
        lambda module, args: cached_inputs.append(args[0].detach().clone()))
    try:
        with torch.inference_mode():
            result = policy.predict_action(dummy, use_k_tokens=2)
    finally:
        handle.remove()
    assert result["action"].shape == (2, 2, 7)
    assert result["action_pred"].shape == (2, 16, 7)
    assert torch.isfinite(result["action"]).all()
    assert len(cached_inputs) == 2
    assert torch.all(cached_inputs[0] == policy.bos_id)
    assert torch.all(cached_inputs[1] != policy.bos_id)
    with pytest.raises(RuntimeError, match="pending"):
        policy.predict_action(dummy)
    executed = torch.full((2, 2, 7), .25)
    policy.record_executed_actions(executed, [1, 2])
    torch.testing.assert_close(policy._past_buffer[0, -1:], executed[0, :1])
    torch.testing.assert_close(policy._past_buffer[1, -2:], executed[1])
    assert policy._pending_execution_steps is None
    policy.reset()
    assert policy._past_buffer is None


@pytest.mark.parametrize("history_steps", [8, 16])
def test_real_robot_config_uses_native_shapes_and_held_out_offline_validation(history_steps):
    register_new_resolvers()
    with initialize_config_dir(config_dir=str(ROOT / "oat/config"), version_base=None):
        cfg = compose(config_name="experimental/train_past2next_state_history_gate_real_robot", overrides=[
            f"state_history_steps={history_steps}",
            "policy.action_tokenizer.checkpoint=/tmp/test_real_robot_tokenizer.ckpt",
        ])
    assert cfg.policy._target_.endswith("Past2NextRealRobotStateHistoryGatePolicy")
    assert list(cfg.state_history_keys) == list(SHAPES)
    assert list(cfg.policy.state_history_keys) == list(SHAPES)
    assert not any("quat" in key for key in cfg.task.policy.shape_meta.obs)
    for key, shape in SHAPES.items():
        assert list(cfg.task.policy.shape_meta.obs[key].shape) == shape
    assert cfg.task.policy.lazy_eval is True and cfg.task.policy.env_runner is None
    assert cfg.task.policy.dataset._target_.endswith("RealRobotZarrDatasetWithStateHistory")
    assert cfg.task.policy.dataset.state_history_steps == history_steps
    assert cfg.policy.past_n == cfg.task.policy.dataset.past_n == history_steps - 1
    assert cfg.task.policy.dataset.val_ratio > 0
    assert cfg.training.offline_validation_enabled is True
    assert cfg.training.offline_validation_reason == "held_out_real_robot_episodes"
    assert cfg.checkpoint.topk.monitor_key == "val_loss" and cfg.checkpoint.topk.mode == "min"
