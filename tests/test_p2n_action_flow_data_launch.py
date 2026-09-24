"""Boundary, data-compatibility and explicit launcher contracts on CPU."""
from pathlib import Path

import numpy as np
import pytest
import torch

from oat.common.replay_buffer import ReplayBuffer
from oat.dataset import action_flow_dataset as datasets
from oat.dataset.zarr_dataset_with_prev_window import ZarrDatasetWithPrevWindow
from oat.dataset.zarr_dataset_with_state_history import ZarrDatasetWithStateHistory
from oat.dataset.real_robot_dataset import RealRobotZarrDatasetWithPrevWindow
from oat.dataset.real_robot_state_history import RealRobotZarrDatasetWithStateHistory
from scripts import train_p2n_action_flow as launch


@pytest.fixture
def store(tmp_path):
    replay = ReplayBuffer.create_empty_numpy()
    lengths = [1, 3, 9, 18, 2]
    for episode, length in enumerate(lengths):
        values = 100 * episode + np.arange(length, dtype=np.float32)
        replay.add_episode({
            "action": np.repeat(values[:, None], 7, axis=1),
            "robot0_eef_pos": np.repeat(values[:, None], 3, axis=1),
            "robot0_eef_rot6d": np.tile(np.array([1, 0, 0, 0, 1, 0], dtype=np.float32), (length, 1)),
            "robot0_eef_quat": np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (length, 1)),
            "robot0_gripper_qpos": values[:, None],
            "image": np.full((length, 2, 3, 3), episode, dtype=np.uint8),
        })
    path = tmp_path / "data.zarr"
    replay.save_to_path(str(path))
    return path, lengths


@pytest.mark.parametrize("new,old,gate", [
    (datasets.ActionFlowZarrDatasetWithPrevWindow, ZarrDatasetWithPrevWindow, False),
    (datasets.ActionFlowZarrDatasetWithStateHistory, ZarrDatasetWithStateHistory, True),
    (datasets.ActionFlowRealRobotZarrDatasetWithPrevWindow, RealRobotZarrDatasetWithPrevWindow, False),
    (datasets.ActionFlowRealRobotZarrDatasetWithStateHistory, RealRobotZarrDatasetWithStateHistory, True),
])
def test_all_dataset_adapters_preserve_actions_splits_and_boundary_masks(store, new, old, gate):
    path, lengths = store
    kwargs = dict(zarr_path=str(path), obs_keys=["robot0_eef_pos", "robot0_eef_rot6d", "robot0_eef_quat", "robot0_gripper_qpos", "image"],
                  n_obs_steps=2, n_action_steps=16, past_n=7, n_exec_steps=8,
                  seed=42, val_ratio=0.2, history_padding="zero", return_history_validity=True)
    if gate:
        kwargs.update(state_history_steps=8, state_history_keys=["robot0_eef_pos", "robot0_eef_rot6d", "robot0_gripper_qpos"])
    current, previous = new(**kwargs), old(**kwargs)
    offsets = np.r_[0, np.cumsum(lengths)]
    all_ids = []
    for view, parent in ((current, previous), (current.get_validation_dataset(), previous.get_validation_dataset())):
        assert view.dataset_identity == current.dataset_identity
        assert len(view) == len(parent)
        np.testing.assert_array_equal(view.train_mask, parent.train_mask)
        np.testing.assert_array_equal(view.seq_sampler.indices, parent.seq_sampler.indices)
        for index in range(len(view)):
            item, original = view[index], parent[index]
            sample_id = item["sample_id"].item()
            all_ids.append(sample_id)
            episode = np.searchsorted(offsets[1:], sample_id, side="right")
            remaining = offsets[episode + 1] - sample_id
            expected_mask = torch.arange(16) < remaining
            assert item["sample_id"].dtype == torch.int64
            assert item["future_action_valid"].dtype == torch.bool
            torch.testing.assert_close(item["future_action_valid"], expected_mask)
            for key, value in original.items():
                if isinstance(value, dict):
                    for field, tensor in value.items():
                        assert item[key][field].numpy().tobytes() == tensor.numpy().tobytes()
                else:
                    assert item[key].numpy().tobytes() == value.numpy().tobytes()
            if remaining == 1:
                assert item["future_action_valid"].sum() == 1
                torch.testing.assert_close(item["action"], item["action"][:1].expand(16, 7))
            if gate:
                state = item["obs"]["state_history_valid"]
                torch.testing.assert_close(item["past_action_valid"], state[:-1] & state[1:])
        old_normalizer, new_normalizer = parent.get_normalizer(), view.get_normalizer()
        for key, value in old_normalizer.state_dict().items():
            torch.testing.assert_close(new_normalizer.state_dict()[key], value, rtol=0, atol=0)
    assert sorted(all_ids) == list(range(sum(lengths)))
    assert len(all_ids) == len(set(all_ids))
    # Same physical windows have identical IDs with a different split.
    kwargs["val_ratio"] = 0.0
    unsplit = new(**kwargs)
    assert unsplit.dataset_identity == current.dataset_identity
    assert [unsplit[i]["sample_id"].item() for i in range(len(unsplit))] == list(range(sum(lengths)))


@pytest.mark.parametrize("variant", ["p2n_action_flow", "p2n_state_gate_action_flow"])
@pytest.mark.parametrize("task", ["libero", "real_robot"])
def test_four_configs_and_overrides(variant, task):
    cfg = launch.compose_config(variant, task, ["dataloader.batch_size=8", "training.gradient_accumulate_every=4", "training.num_epochs=17"])
    assert cfg.policy_family == "continuous_action_flow"
    assert cfg.dataloader.batch_size == 8 and cfg.training.gradient_accumulate_every == 4
    assert cfg.training.num_epochs == 17
    assert cfg.policy.dropout == 0.0 and cfg.policy.activation_checkpointing
    assert cfg.policy.flow.inference_steps == cfg.policy.flow.self_past_steps == 8
    assert cfg.policy.self_past_chunk_size == 2
    assert cfg.policy.embed_dim == 768 and cfg.policy.n_layers == 16
    assert cfg.policy.resampler_ffn_dim == 2048
    assert cfg.policy.flow.ffn_type == "gelu" and cfg.policy.flow.ffn_hidden_dim == 3072
    assert cfg.policy.flow.self_qk_norm is False and cfg.policy.flow.cross_qk_norm == "layernorm"
    assert cfg.policy.action_dim == 7 and cfg.normalization.refit_on_resume is False
    assert cfg.action_schema.execution_anchor == "current_observation_t"
    assert "tokenizer_checkpoint" not in cfg.policy
    if task == "real_robot":
        assert cfg.task.policy.env_runner is None
        assert cfg.task.policy.dataset.val_ratio == 0.05
        assert cfg.training.num_demo == 77
    if variant == "p2n_action_flow":
        assert not any(key.startswith("history_") for key in cfg.policy)
    else:
        assert cfg.policy.history_dropout == 0.0


@pytest.mark.parametrize("override,match", [
    ("training.use_ema=false", "requires use_ema"),
    ("dataloader.batch_size=2", "multiple of four"),
    ("dataloader.batch_size=5", "multiple of four"),
    ("dataloader.drop_last=false", "drop_last"),
    ("val_dataloader.drop_last=true", "drop_last"),
    ("policy.flow.ct_weight=0.25", "ct_weight"),
    ("policy.flow.solver=heun", "solver"),
    ("policy.dropout=0.1", "zero dropout"),
    ("training.max_val_steps=1", "Full held-out validation"),
    ("training.max_reconst_steps=10", "Full held-out validation"),
])
def test_invalid_recipe_rejected_before_any_training(override, match):
    with pytest.raises(ValueError, match=match):
        launch.compose_config("p2n_action_flow", "real_robot", [override])


def test_explicit_gpu_required_and_no_duplicate_selection():
    for extra in ([], ["--gpus", "1,1"], ["--gpus", "0,1", "--num-processes", "1"]):
        with pytest.raises(SystemExit):
            launch.main(["--variant", "p2n_action_flow", "--task", "real_robot", *extra, "--dry-run"])


def test_dry_run_checks_sources_but_never_probes_gpu_or_launches(monkeypatch, tmp_path):
    calls = []
    def preflight(cfg, output, world_size):
        calls.append((cfg.dataloader.batch_size, cfg.training.gradient_accumulate_every, world_size))
        return {"checked": True}
    monkeypatch.setattr(launch, "preflight", preflight)
    def forbidden(*args, **kwargs):
        raise AssertionError("Dry-run must not launch anything or inspect GPUs")
    monkeypatch.setattr(launch, "check_gpu_selection", forbidden)
    monkeypatch.setattr(launch.subprocess, "run", forbidden)
    out = tmp_path / "uncreated"
    result = launch.main(["--variant", "p2n_action_flow", "--task", "real_robot", "--gpus", "2,3", "--output", str(out),
                          "--dry-run", "--", "dataloader.batch_size=8", "training.gradient_accumulate_every=4"])
    assert result == {"checked": True}
    assert calls == [(8, 4, 2)]
    assert not out.exists()


def test_fingerprint_changes_with_action_source_not_view(store):
    path, _ = store
    import zarr
    root = zarr.open(str(path), mode="r")
    arrays = {key: root["data"][key] for key in root["data"].keys()}
    ends = root["meta"]["episode_ends"][:]
    original = datasets.dataset_fingerprint(path, ends, arrays)
    arrays = {**arrays, "action": arrays["action"][:].copy()}
    arrays["action"][0, 0] += 1
    assert datasets.dataset_fingerprint(path, ends, arrays) != original


def test_resume_preserves_saved_overrides_and_rejects_cross_variant():
    cfg = launch.compose_config("p2n_action_flow", "real_robot", ["dataloader.batch_size=8", "training.gradient_accumulate_every=4", "training.num_epochs=37"])
    resumed = launch.compose_resume_config({"cfg": cfg}, "p2n_action_flow", "real_robot",
        ["training.resume=true", "training.resume_checkpoint=/tmp/saved.ckpt"])
    assert resumed.dataloader.batch_size == 8
    assert resumed.training.gradient_accumulate_every == 4
    assert resumed.training.num_epochs == 37
    with pytest.raises(ValueError, match="family, variant or task"):
        launch.compose_resume_config({"cfg": cfg}, "p2n_state_gate_action_flow", "real_robot")


def test_resume_preflight_is_offline_and_checks_data_and_rank_rng(monkeypatch, tmp_path):
    import copy
    import dill
    from oat.workspace.train_p2n_action_flow import TrainP2NActionFlowWorkspace
    cfg = launch.compose_config("p2n_action_flow", "real_robot", ["training.resume=true", "training.resume_checkpoint=/missing/original.ckpt"])
    split = {"identity": "sha256:test", "train_episode_ids": [0, 1], "validation_episode_ids": [2],
             "train": {"windows": 32}, "validation": {"windows": 7}}
    payload = {"cfg": cfg, "metadata": {"policy_family": "continuous_action_flow", "artifact_schema_version": 1,
               "variant": cfg.variant, "task_type": cfg.task_type, "dataset_split": copy.deepcopy(split)},
               "policy_config": {"construction_mode": "restore"},
               "state_dicts": {"model": {}, "ema_model": {}, "optimizer": {}},
               "pickles": {key: dill.dumps(None) for key in TrainP2NActionFlowWorkspace.include_keys},
               "training_state": {"rng_states": [{}, {}]}}
    monkeypatch.setattr(launch, "inspect_dataset", lambda cfg: split)
    def forbidden(*args, **kwargs):
        raise AssertionError("Resume cannot refit normalization or access external DINO sources")
    monkeypatch.setattr(launch, "validate_dino_source", forbidden)
    monkeypatch.setattr(launch, "inspect_training_normalizer", forbidden)
    report = launch.preflight(cfg, tmp_path / "out", 2, resume_payload=payload)
    assert report["sources"]["external_frozen_sources_required"] is False
    with pytest.raises(ValueError, match="world size"):
        launch.preflight(cfg, tmp_path / "out", 1, resume_payload=payload)
    payload["metadata"]["dataset_split"]["identity"] = "sha256:different"
    with pytest.raises(ValueError, match="dataset identity"):
        launch.preflight(cfg, tmp_path / "out", 2, resume_payload=payload)


@pytest.mark.parametrize("mode", ["open", "closed"])
def test_formal_gate_training_requires_learned_gate(mode):
    with pytest.raises(ValueError, match="history_gate_mode=learned"):
        launch.compose_config("p2n_state_gate_action_flow", "real_robot", [f"policy.history_gate_mode={mode}"])


def test_cross_family_fields_are_rejected():
    for override in ("+policy.tokenizer_checkpoint=/absent/weights", "+policy.flow.levels=[8,5,5,5,5]",
                     "+policy.flow.code_dim=5", "policy_family=oat_latent_flow", "+policy.use_k_tokens=8"):
        with pytest.raises(ValueError, match="family|schema"):
            launch.compose_config("p2n_action_flow", "real_robot", [override])


def test_training_normalizer_provenance_and_roundtrip_exclude_validation_frames(store):
    path, _ = store
    kwargs = dict(zarr_path=str(path), obs_keys=["robot0_eef_pos", "robot0_eef_rot6d", "robot0_gripper_qpos"],
                  n_obs_steps=2, n_action_steps=16, past_n=7, n_exec_steps=8,
                  seed=42, val_ratio=0.2, history_padding="zero", return_history_validity=True)
    ds = datasets.ActionFlowRealRobotZarrDatasetWithPrevWindow(**kwargs)
    train = ds.get_normalizer()
    valid = ds.get_validation_dataset().get_normalizer()
    assert all(not param.requires_grad for param in train.parameters())
    expected_ids = np.flatnonzero(ds.normalization_train_mask).tolist()
    assert ds.normalizer_provenance["train_episode_ids"] == expected_ids
    assert ds.normalizer_provenance["source"] == "training_replay_frames"
    for name, value in train.state_dict().items():
        torch.testing.assert_close(valid.state_dict()[name], value, atol=0, rtol=0)
    for field in ("action", "robot0_eef_rot6d", "robot0_gripper_qpos"):
        raw = torch.from_numpy(ds.replay_buffer[field][:])
        restored = train[field].unnormalize(train[field].normalize(raw))
        assert torch.isfinite(restored).all()
        torch.testing.assert_close(restored, raw, atol=1e-4, rtol=1e-5)
    for idx in range(len(ds)):
        item = ds[idx]
        torch.testing.assert_close(item["action"][0, :3], item["obs"]["robot0_eef_pos"][-1])


def test_python_flags_preserve_later_hydra_overrides(monkeypatch, tmp_path):
    calls = []
    def preflight(cfg, output, world_size):
        calls.append((cfg.dataloader.batch_size, cfg.val_dataloader.batch_size,
                      cfg.training.gradient_accumulate_every, cfg.training.num_epochs))
        return {"checked": True}
    monkeypatch.setattr(launch, "preflight", preflight)
    launch.main(["--variant", "p2n_action_flow", "--task", "real_robot", "--gpus", "2,3",
                 "--dry-run", "--batch-size", "4", "--val-batch-size", "3", "--grad-accum", "8",
                 "--output", str(tmp_path / "absent"), "--", "dataloader.batch_size=8", "training.num_epochs=19"])
    assert calls == [(8, 3, 8, 19)]


def test_shell_flags_sequential_variants_and_override_order(tmp_path):
    import json
    import os
    import subprocess
    recorder = tmp_path / "capture.py"
    calls_file = tmp_path / "calls.jsonl"
    recorder.write_text("#!/usr/bin/env python3\nimport json, os, sys\nwith open(os.environ['P2N_CAPTURE'], 'a') as f:\n    f.write(json.dumps(sys.argv[1:]) + '\\n')\n")
    recorder.chmod(0o755)
    script = Path(__file__).resolve().parents[1] / "train_p2n_action_flow.sh"
    subprocess.run(["bash", str(script), "--python", str(recorder), "--gpus", "2,3",
                    "--batch-size", "4", "--val-batch-size", "4", "--grad-accum", "8", "--dry-run",
                    "--", "dataloader.batch_size=8", "training.num_epochs=9", "logging.mode=online"],
                   env={**os.environ, "P2N_CAPTURE": str(calls_file)}, check=True)
    calls = [json.loads(line) for line in calls_file.read_text().splitlines()]
    assert len(calls) == 2
    assert [args[args.index("--variant") + 1] for args in calls] == ["p2n_action_flow", "p2n_state_gate_action_flow"]
    for args in calls:
        assert args[args.index("--gpus") + 1] == "2,3"
        assert args[-3:] == ["dataloader.batch_size=8", "training.num_epochs=9", "logging.mode=online"]
        assert "dataloader.batch_size=4" in args and "val_dataloader.batch_size=4" in args
        assert "training.gradient_accumulate_every=8" in args and "--tokenizer" not in args
