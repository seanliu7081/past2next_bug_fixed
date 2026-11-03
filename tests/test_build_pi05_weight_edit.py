"""Behavioral tests for manually selected contrasts and standalone snapshot export."""
import json
from pathlib import Path

import dill
import numpy as np
import pytest
import torch

from oat.model.vla.weight_space_edit import (
    TARGET, apply_output_head_patch, checkpoint_output_head, export_edited_snapshot,
    file_sha256, load_patch_spec, tensor_sha256,
)
from scripts.build_pi05_weight_edit import centered_contrast, load_capture_trace, main, select_indices


def make_snapshot(path: Path):
    payload = {
        "format": "p2n_vla_checkpoint_v1", "cfg": {"preserve": "original training config"},
        "policy_config": {
            "_target_": "oat.policy.pi05_past.Pi05PastPolicy", "shape_meta": {"action": {"shape": [7]}},
            "n_action_steps": 2, "flow_num_steps": 2,
        },
        "metadata": {"training_run": {"optimizer_step": 50000}},
        "state_dicts": {"model": {
            f"{TARGET}.weight": torch.arange(40).reshape(8, 5).to(torch.bfloat16) / 100,
            f"{TARGET}.bias": torch.arange(8).to(torch.bfloat16) / 100,
            "action_normalizer.params_dict.action.scale": torch.tensor([2.] * 7),
            "unrelated.weight": torch.arange(19, dtype=torch.float64),
        }},
    }
    torch.save(payload, path, pickle_module=dill)
    return payload


def features():
    value = np.zeros((3, 2, 1, 4, 5), dtype=np.float32)
    value[0, ...] = [2, 1, 0, 0, 0]
    value[1, ...] = [5, 1, 4, 0, 0]
    value[2, ...] = [-1, 1, -4, 0, 0]
    return value


def test_centered_contrast_anchors_and_collateral_response():
    built = centered_contrast(features(), [1], [0], [0, 1], 0.3)
    delta = built["first_order_normalized_action_delta"]
    np.testing.assert_allclose(delta[1], 0.3, atol=1e-7)
    np.testing.assert_allclose(delta[0], 0, atol=1e-7)
    # A context on the other side of the reference receives the opposite edit.
    np.testing.assert_allclose(delta[2], -0.3, atol=1e-7)
    np.testing.assert_allclose(built["signal"][1], 1, atol=1e-7)
    with pytest.raises(ValueError, match="disjoint"):
        centered_contrast(features(), [0, 1], [1], [0, 1], 0.3)
    with pytest.raises(ValueError, match="contrast"):
        centered_contrast(np.zeros_like(features()), [1], [0], [0, 1], 0.3)


def test_selectors_are_explicit_bounded_and_half_open():
    assert select_indices("0:3,5,2", 6) == [0, 1, 2, 5]
    for invalid in ("-1", "6", "0:7", "2:2", "0:2:1", ""):
        with pytest.raises(ValueError):
            select_indices(invalid, 6)


def test_reference_variation_nulls_each_reference_chunk_mean():
    # References vary along x; risk also has a distinct y component. A plain
    # contrast responds to x and disturbs references; protection retains only y.
    value = np.zeros((4, 2, 1, 4, 5), dtype=np.float32)
    for index, point in enumerate(([0., 1., 0., 0., 0.], [2., 1., 0., 0., 0.],
                                   [4., 1., 0., 0., 0.], [5., 3., 0., 0., 0.])):
        value[index] = point
    # Within-chunk variation has zero mean and must not be mistaken for a
    # stronger per-step protection guarantee.
    value[0, 0, :, :, 1] += 0.5
    value[0, 1, :, :, 1] -= 0.5
    plain = centered_contrast(value, [3], [0, 1, 2], [0, 1], 0.3)
    protected = centered_contrast(value, [3], [0, 1, 2], [0, 1], 0.3, True)
    plain_delta = plain["first_order_normalized_action_delta"][:3].mean(axis=(1, 2))
    assert np.abs(plain_delta).max() > 0.05
    reference_delta = protected["first_order_normalized_action_delta"][:3].mean(axis=(1, 2))
    np.testing.assert_allclose(reference_delta, 0, atol=1e-7)
    np.testing.assert_allclose(protected["first_order_normalized_action_delta"][3], 0.3, atol=1e-7)
    report = protected["reference_protection"]
    assert report["reference_variation_rank"] == 1
    assert 0 < report["retained_contrast_fraction"] < 1
    assert report["reference_chunk_mean_signal_max_abs"] < 1e-6
    assert protected["signal"][0, 0].mean() != pytest.approx(0)


def test_reference_protection_rejects_contrast_entirely_in_reference_span():
    value = np.zeros((3, 2, 1, 4, 5), dtype=np.float32)
    value[1, ..., 0] = 1
    value[2, ..., 0] = 2
    with pytest.raises(ValueError, match="after reference protection"):
        centered_contrast(value, [2], [0, 1], [0, 1], 0.3, True)


def test_builder_and_export_preserve_source_and_runtime_patch(tmp_path):
    source = tmp_path / "source.ckpt"
    original = make_snapshot(source)
    original_hash = file_sha256(source)
    trace = tmp_path / "captures"
    trace.mkdir()
    for i, hidden in enumerate(features()):
        np.savez(trace / f"chunk_{i:04d}.npz", hidden=hidden, policy_step=i * 8, eef_pos=[0, 0, i])
    output = tmp_path / "candidate"
    destination = tmp_path / "edited.ckpt"
    summary = main([
        "--trace-dir", str(trace), "--snapshot", str(source), "--risk-chunks", "1",
        "--reference-chunks", "0", "--raw-action-change", "0.15", "--output-dir", str(output),
        "--export-snapshot", str(destination),
    ])
    assert file_sha256(source) == original_hash
    assert summary["risk_mean_first_order_raw_change"] == pytest.approx(0.15, abs=1e-7)
    assert summary["reference_mean_first_order_raw_change"] == pytest.approx(0, abs=1e-7)
    patch = load_patch_spec(output / "patch.json")
    runtime, _ = checkpoint_output_head(original)
    assert patch["expected_weight_sha256"] == tensor_sha256(runtime.flow_heads.action_out_proj.weight)
    applied = apply_output_head_patch(runtime, patch)
    exported = torch.load(destination, pickle_module=dill, weights_only=False)
    restored, _ = checkpoint_output_head(exported)
    assert torch.equal(restored.flow_heads.action_out_proj.weight, applied.head.weight)
    assert torch.equal(restored.flow_heads.action_out_proj.bias, applied.head.bias)
    assert exported["cfg"] == original["cfg"]
    assert exported["policy_config"] == original["policy_config"]
    for key, value in original["state_dicts"]["model"].items():
        if not key.startswith(TARGET):
            assert torch.equal(value, exported["state_dicts"]["model"][key])
    provenance = exported["metadata"]["weight_space_edit"]
    assert provenance["source_snapshot_sha256"] == original_hash
    assert provenance["source_stored_head_dtype"] == "torch.bfloat16"
    assert provenance["exported_head_dtype"] == "torch.float32"
    diagnostics = json.loads((output / "responses.json").read_text())
    assert len(diagnostics["per_chunk"][1]["centered_signal_mean_by_denoising_step"]) == 2
    assert diagnostics["per_chunk"][2]["first_order_raw_mean"] < 0
    with pytest.raises(FileExistsError):
        export_edited_snapshot(source, destination, patch)
    assert file_sha256(source) == original_hash


def test_incomplete_capture_trace_rejected(tmp_path):
    np.savez(tmp_path / "chunk_0001.npz", hidden=features()[0])
    with pytest.raises(ValueError, match="contiguous"):
        load_capture_trace(tmp_path)
