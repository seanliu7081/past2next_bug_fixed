"""History timing, rotation geometry, padding and differentiable temporal fusion."""

import math

import pytest
import torch

from oat.model.common.normalizer import LinearNormalizer
from oat.model.state_action_history import StateActionHistoryEncoder


STATE_SHAPES = {"robot0_eef_pos": (3,), "robot0_eef_quat": (4,),
                "robot0_gripper_qpos": (2,)}


def make_inputs(steps=8, batch_size=2, state_shapes=None):
    shapes = STATE_SHAPES if state_shapes is None else state_shapes
    states = {}
    normalizer = LinearNormalizer()
    for key, shape in shapes.items():
        width = math.prod(shape)
        values = torch.arange(batch_size * steps * width).float().reshape(batch_size, steps, *shape) / 20
        if key.endswith("_quat"):
            values.zero_()
            values[..., -1] = 1
        else:
            low = -torch.arange(1, width + 1).float()
            high = torch.arange(1, width + 1).float() * 3
            normalizer.fit({key: torch.stack((low, high))})
        states[key] = values
    valid = torch.ones(batch_size, steps, dtype=torch.bool)
    actions = torch.arange(batch_size * (steps - 1) * 7).reshape(batch_size, steps - 1, 7).float() / 100
    return states, valid, actions, normalizer


def make_encoder(steps=8, state_shapes=None):
    return StateActionHistoryEncoder(
        state_shapes=STATE_SHAPES if state_shapes is None else state_shapes,
        action_dim=7, history_steps=steps, output_dim=13,
        embed_dim=16, n_heads=4, n_layers=2, n_summary_tokens=4, dropout=0,
    )


@pytest.mark.parametrize("steps", [8, 16])
def test_temporal_fusion_shape_and_gradients(steps):
    encoder = make_encoder(steps).train()
    states, valid, actions, normalizer = make_inputs(steps)
    actions.requires_grad_()
    states["robot0_eef_pos"].requires_grad_()
    output = encoder(states, valid, actions, normalizer)
    assert output.shape == (2, 4, 13)
    assert torch.isfinite(output).all()
    weights = torch.linspace(-1, 2, output.numel()).reshape_as(output)
    (output * weights).sum().backward()
    assert torch.isfinite(actions.grad).all() and actions.grad.abs().sum() > 0
    assert states["robot0_eef_pos"].grad.abs().sum() > 0
    for name, parameter in encoder.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert encoder.temporal_encoder.layers[0].self_attn.in_proj_weight.grad.abs().sum() > 0
    assert encoder.summary_queries.grad.abs().sum() > 0
    assert all("normalizer" not in name for name in encoder.state_dict())


def test_actions_align_with_transition_endpoints_and_use_existing_normalizer():
    encoder = make_encoder()
    states, valid, actions, normalizer = make_inputs()
    valid[0, :4] = False
    features, padding = encoder.build_features(states, valid, actions, normalizer)
    torch.testing.assert_close(padding, ~valid)
    aligned = features[..., encoder.feature_slices["action"]]
    torch.testing.assert_close(aligned[1, 1:], actions[1])
    torch.testing.assert_close(aligned[0, 5:], actions[0, 4:])
    assert torch.count_nonzero(aligned[0, :5]) == 0
    assert torch.count_nonzero(aligned[1, 0]) == 0
    for key in ("robot0_eef_pos", "robot0_gripper_qpos"):
        absolute = features[..., encoder.feature_slices[f"absolute/{key}"]]
        normalized = normalizer[key].normalize(states[key])
        torch.testing.assert_close(absolute[valid], normalized[valid])
        delta = features[..., encoder.feature_slices[f"delta/{key}"]]
        torch.testing.assert_close(delta[1, 1:], normalized[1, 1:] - normalized[1, :-1])
        torch.testing.assert_close(delta[0, 5:], normalized[0, 5:] - normalized[0, 4:-1])
        assert torch.count_nonzero(delta[0, :5]) == 0
    torch.testing.assert_close(features[..., encoder.feature_slices["state_valid"]].squeeze(-1), valid.float())
    transition = torch.cat((torch.zeros_like(valid[:, :1]), valid[:, 1:] & valid[:, :-1]), dim=1)
    for key in ("transition_valid", "action_valid"):
        torch.testing.assert_close(features[..., encoder.feature_slices[key]].squeeze(-1), transition.float())


def test_relative_rotation_is_world_frame_and_quaternion_sign_invariant():
    encoder = make_encoder().eval()
    states, valid, actions, normalizer = make_inputs()
    # Start at Rx(90deg), then rotate about WORLD z by 90deg: Rz @ Rx.
    root_half = math.sqrt(0.5)
    states["robot0_eef_quat"][:] = torch.tensor([root_half, 0, 0, root_half])
    states["robot0_eef_quat"][:, -1] = torch.tensor([0.5, 0.5, 0.5, 0.5])
    features, _ = encoder.build_features(states, valid, actions, normalizer)
    relative = features[..., encoder.feature_slices["delta/robot0_eef_quat"]]
    expected_world_z = torch.tensor([0, 1, 0, -1, 0, 0]).float()
    torch.testing.assert_close(relative[:, -1], expected_world_z.expand(2, -1), atol=1e-6, rtol=0)
    identity = torch.tensor([1, 0, 0, 0, 1, 0]).float()
    torch.testing.assert_close(relative[:, 1:-1], identity.expand(2, 6, 6), atol=1e-6, rtol=0)
    assert torch.count_nonzero(relative[:, 0]) == 0
    original_output = encoder(states, valid, actions, normalizer)
    states["robot0_eef_quat"][:, ::2] *= -3  # Antipodal signs / quaternion scale carry no pose information.
    signed_features, _ = encoder.build_features(states, valid, actions, normalizer)
    torch.testing.assert_close(signed_features, features, atol=1e-6, rtol=0)
    torch.testing.assert_close(encoder(states, valid, actions, normalizer), original_output, atol=1e-6, rtol=0)


def test_reset_padding_garbage_does_not_affect_features_outputs_or_gradients():
    encoder = make_encoder().eval()
    states, valid, actions, normalizer = make_inputs()
    valid[0, :-1] = False  # Reset: only the current state exists, no valid transition.
    valid[1, :-3] = False
    original_features, _ = encoder.build_features(states, valid, actions, normalizer)
    original_output = encoder(states, valid, actions, normalizer)
    for key in states:
        states[key][~valid] = float("nan")
        states[key].requires_grad_()
    transition_valid = valid[:, :-1] & valid[:, 1:]
    actions[~transition_valid] = float("inf")
    actions.requires_grad_()
    features, _ = encoder.build_features(states, valid, actions, normalizer)
    output = encoder(states, valid, actions, normalizer)
    torch.testing.assert_close(features, original_features, atol=0, rtol=0)
    torch.testing.assert_close(output, original_output, atol=0, rtol=0)
    assert torch.count_nonzero(features[~valid]) == 0
    output.square().sum().backward()
    assert torch.isfinite(actions.grad).all()
    assert torch.count_nonzero(actions.grad[~transition_valid]) == 0
    for value in states.values():
        assert torch.isfinite(value.grad).all()
        assert torch.count_nonzero(value.grad[~valid]) == 0


def test_padding_masks_reach_both_attention_stages(monkeypatch):
    encoder = make_encoder().eval()
    states, valid, actions, normalizer = make_inputs()
    valid[:, :-2] = False
    captured = {}
    temporal = encoder.temporal_encoder.forward
    summary = encoder.summary_attention.forward

    def run_temporal(src, **kwargs):
        captured["temporal"] = kwargs["src_key_padding_mask"]
        return temporal(src, **kwargs)

    def run_summary(*args, **kwargs):
        captured["summary"] = kwargs["key_padding_mask"]
        return summary(*args, **kwargs)

    monkeypatch.setattr(encoder.temporal_encoder, "forward", run_temporal)
    monkeypatch.setattr(encoder.summary_attention, "forward", run_summary)
    encoder(states, valid, actions, normalizer)
    torch.testing.assert_close(captured["temporal"], ~valid)
    torch.testing.assert_close(captured["summary"], ~valid)


@pytest.mark.parametrize("problem", ["empty", "gap", "missing_current", "float_mask",
                                    "zero_quaternion", "nan_quaternion", "nan_position",
                                    "infinite_action", "wrong_action_shape", "missing_port"])
def test_invalid_observed_histories_are_rejected(problem):
    encoder = make_encoder()
    states, valid, actions, normalizer = make_inputs()
    if problem == "empty":
        valid.zero_()
    elif problem == "gap":
        valid[:, 3] = False
    elif problem == "missing_current":
        valid[:, -1] = False
    elif problem == "float_mask":
        valid = valid.float()
    elif problem == "zero_quaternion":
        states["robot0_eef_quat"][:, -1] = 0
    elif problem == "nan_quaternion":
        states["robot0_eef_quat"][:, -1, 0] = float("nan")
    elif problem == "nan_position":
        states["robot0_eef_pos"][:, -1, 0] = float("nan")
    elif problem == "infinite_action":
        actions[:, -1, 0] = float("inf")
    elif problem == "wrong_action_shape":
        actions = actions[:, :-1]
    elif problem == "missing_port":
        del states["robot0_eef_pos"]
    with pytest.raises(ValueError):
        encoder(states, valid, actions, normalizer)


def test_configurable_joint_history_without_quaternions():
    shapes = {"robot0_joint_pos": (7,), "robot0_gripper_qpos": (2,)}
    encoder = make_encoder(16, shapes)
    states, valid, actions, normalizer = make_inputs(16, state_shapes=shapes)
    features, _ = encoder.build_features(states, valid, actions, normalizer)
    normalized_joints = normalizer["robot0_joint_pos"].normalize(states["robot0_joint_pos"])
    torch.testing.assert_close(features[:, 1:, encoder.feature_slices["delta/robot0_joint_pos"]],
                               normalized_joints[:, 1:] - normalized_joints[:, :-1])
    assert encoder(states, valid, actions, normalizer).shape == (2, 4, 13)


def test_mixed_precision_preserves_rotation_features_and_backpropagates():
    encoder = make_encoder().train()
    states, valid, actions, normalizer = make_inputs()
    angles = torch.linspace(0.02, 0.19, 8)
    states["robot0_eef_quat"][..., 2] = torch.sin(angles / 2)
    states["robot0_eef_quat"][..., 3] = torch.cos(angles / 2)
    features, _ = encoder.build_features(states, valid, actions, normalizer)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        mixed_features, _ = encoder.build_features(states, valid, actions, normalizer)
        output = encoder(states, valid, actions, normalizer)
        loss = output.float().square().mean()
    torch.testing.assert_close(mixed_features, features, atol=0, rtol=0)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(encoder.input_projection[0].weight.grad).all()
