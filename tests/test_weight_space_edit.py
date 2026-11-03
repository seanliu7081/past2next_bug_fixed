"""CPU checks for locality, flow sign, exact restore, and diagnostic capture."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from oat.model.vla.weight_space_edit import (
    OutputHeadCapture, apply_output_head_patch, compare_action_chunks,
    load_patch_spec, tensor_sha256,
)


def policy(dtype=torch.float32):
    head = nn.Linear(5, 8, dtype=dtype)
    return SimpleNamespace(flow_heads=SimpleNamespace(action_out_proj=head), action_dim=7, flow_num_steps=2)


def spec(**kwargs):
    value = dict(schema_version=1, kind="action_output_rank_one", rows=[2],
                 coefficients=[-0.2], direction=[3., 0., 4., 0., 0.])
    value.update(kwargs)
    return value


def test_patch_is_local_reversible_and_audited():
    p = policy()
    head = p.flow_heads.action_out_proj
    before_w, before_b = head.weight.detach().clone(), head.bias.detach().clone()
    handle = apply_output_head_patch(p, spec(bias_delta=[-0.1]))
    untouched = [0, 1, 3, 4, 5, 6, 7]
    assert torch.equal(head.weight[untouched], before_w[untouched])
    assert torch.equal(head.bias[untouched], before_b[untouched])
    assert handle.provenance["changed_values"] == 3
    assert handle.provenance["gradient_updates"] == 0
    assert head.weight.grad is None
    assert handle.provenance["before_sha256"]["weight"] == tensor_sha256(before_w)
    handle.restore()
    handle.restore()
    assert torch.equal(head.weight, before_w)
    assert torch.equal(head.bias, before_b)


def test_velocity_edit_has_negative_integration_sign():
    p = policy()
    head = p.flow_heads.action_out_proj
    hidden = torch.tensor([[[0.6, 0., 0.8, 0., 0.]]])
    before_velocity = head(hidden).detach()
    handle = apply_output_head_patch(p, spec())
    after_velocity = head(hidden).detach()
    # With fixed features, ten Euler steps of dt=-0.1 yield a positive z change.
    original_action = torch.zeros_like(before_velocity)
    edited_action = torch.zeros_like(after_velocity)
    for _ in range(10):
        original_action += -0.1 * before_velocity
        edited_action += -0.1 * after_velocity
    assert (edited_action - original_action)[0, 0, 2].item() == pytest.approx(0.2, abs=1e-6)
    assert torch.equal(edited_action[..., :2], original_action[..., :2])
    handle.restore()


@pytest.mark.parametrize("bad", [dict(rows=[7]), dict(rows=[2, 2], coefficients=[1, 1]),
                                 dict(direction=[0.] * 5), dict(coefficients=[float("nan")]),
                                 dict(expected_weight_sha256="wrong"), dict(direction=[1., 2.])])
def test_invalid_edit_never_mutates(bad):
    p = policy()
    head = p.flow_heads.action_out_proj
    original = tensor_sha256(head.weight), tensor_sha256(head.bias)
    with pytest.raises((ValueError, TypeError)):
        apply_output_head_patch(p, spec(**bad))
    assert original == (tensor_sha256(head.weight), tensor_sha256(head.bias))


def test_restore_refuses_to_overwrite_later_edit():
    p = policy()
    handle = apply_output_head_patch(p, spec())
    with torch.no_grad():
        p.flow_heads.action_out_proj.bias[0] += 1
    with pytest.raises(RuntimeError, match="changed after"):
        handle.restore()


def test_capture_preserves_outputs_and_tracks_steps():
    p = policy()
    head = p.flow_heads.action_out_proj
    hidden = torch.randn(2, 3, 5)
    expected = head(hidden).detach()
    with OutputHeadCapture(p) as capture:
        actual = head(hidden)
        head(hidden + 1)
        data = capture.pop()
        assert torch.equal(actual.detach(), expected)
        assert data["hidden"].shape == (2, 2, 3, 5)
        assert data["velocity"].shape == (2, 2, 3, 8)
        np.testing.assert_array_equal(data["hidden"][0], hidden.numpy())
        assert data["flow_steps"] == 2
        assert not capture.records
        head(hidden)
        with pytest.raises(RuntimeError, match="Expected 2"):
            capture.pop()
    assert not head._forward_hooks


def test_relative_npz_spec_and_bfloat16_fingerprint(tmp_path):
    import json
    np.savez(tmp_path / "activation.npz", direction=np.array([3., 0., 4., 0., 0.]))
    value = spec(direction_file="activation.npz")
    del value["direction"]
    (tmp_path / "patch.json").write_text(json.dumps(value))
    loaded = load_patch_spec(tmp_path / "patch.json")
    assert loaded["direction_file"] == str(tmp_path / "activation.npz")
    p = policy(torch.bfloat16)
    original = tensor_sha256(p.flow_heads.action_out_proj.weight)
    handle = apply_output_head_patch(p, loaded)
    assert handle.provenance["changed_values"] > 0
    handle.restore()
    assert tensor_sha256(p.flow_heads.action_out_proj.weight) == original


def test_raw_and_normalized_comparisons():
    normalizer = SimpleNamespace(normalize=lambda value: value * torch.tensor([2., 3.]) + 4.)
    original = torch.zeros(1, 2, 2)
    edited = torch.tensor([[[0.1, 0.2], [0.3, 0.4]]])
    result = compare_action_chunks(original, edited, normalizer)
    np.testing.assert_allclose(result["raw_delta"], edited.numpy())
    np.testing.assert_allclose(result["normalized_delta"], (edited * torch.tensor([2., 3.])).numpy(), atol=3e-7)
    assert result["normalized_max_abs"] == pytest.approx(1.2, abs=1e-6)
