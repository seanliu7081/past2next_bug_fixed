"""CPU contracts for the separate ResNet-18 direct-action-flow launch path."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import dill
import numpy as np
from omegaconf import OmegaConf
import pytest

from scripts import train_p2n_action_flow as original
from scripts import train_p2n_action_flow_resnet18 as launch

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("p2n_action_flow", "p2n_state_gate_action_flow")


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("task", ("libero", "real_robot"))
def test_all_recipes_preserve_flow_and_dataset_but_replace_encoder(variant, task):
    cfg = launch.compose_config(variant, task)
    old = original.compose_config(variant, task)
    assert cfg.policy.obs_encoder_type == "resnet18"
    assert cfg.policy._target_.endswith("ResNet18Policy")
    assert cfg._target_.endswith("ResNet18Workspace")
    assert OmegaConf.to_container(cfg.policy.resnet_config) == {
        "crop_shape": [112, 112], "use_group_norm": True,
        "share_rgb_model": False, "eval_fixed_crop": True,
    }
    assert cfg.policy_family == old.policy_family == "continuous_action_flow"
    assert cfg.variant == old.variant == variant
    assert cfg.task_type == old.task_type == task
    for first, second in ((cfg.policy.flow, old.policy.flow), (cfg.action_schema, old.action_schema),
                          (cfg.task.policy.dataset, old.task.policy.dataset)):
        assert OmegaConf.to_container(first, resolve=True) == OmegaConf.to_container(second, resolve=True)
    assert not any("dino" in key or "resampler" in key or "tokenizer" in key for key in cfg.policy)
    assert cfg.policy.embed_dim == 768 and cfg.policy.n_layers == 16
    assert cfg.horizon == 16 and cfg.policy.action_dim == 7
    assert cfg.logging.project.endswith("resnet18") and "resnet18" in cfg.logging.tags


@pytest.mark.parametrize("override,match", [
    ("policy.obs_encoder_type=dinov3", "observation encoder"),
    ("+policy.dino_path=/missing", "DINO/Resampler"),
    ("+policy.resampler_depth=2", "DINO/Resampler"),
    ("+policy.tokenizer_checkpoint=/missing", "codec/AR"),
    ("policy.resnet_config.crop_shape=[128,128]", "smaller"),
    ("policy.resnet_config.crop_shape=[0,100]", "positive integers"),
    ("policy.resnet_config.use_group_norm=1", "boolean"),
    ("dataloader.batch_size=2", "multiple of four"),
    ("training.use_ema=false", "requires use_ema"),
])
def test_preflight_rejects_incompatible_configuration(override, match):
    with pytest.raises(ValueError, match=match):
        launch.compose_config(VARIANTS[0], "real_robot", [override])


@pytest.fixture
def store(tmp_path):
    from oat.common.replay_buffer import ReplayBuffer
    replay = ReplayBuffer.create_empty_numpy()
    for episode in range(5):
        length = 20
        values = episode * 10 + np.arange(length, dtype=np.float32)
        replay.add_episode({
            "action": np.repeat(values[:, None], 7, axis=1),
            "robot0_eef_pos": np.repeat(values[:, None], 3, axis=1),
            "robot0_eef_rot6d": np.tile(np.array([1, 0, 0, 0, 1, 0], np.float32), (length, 1)),
            "robot0_gripper_qpos": values[:, None],
            "task_uid": np.zeros((length, 1), np.float32),
            "agentview_rgb": np.full((length, 128, 128, 3), episode, np.uint8),
            "robot0_eye_in_hand_rgb": np.full((length, 128, 128, 3), episode, np.uint8),
        })
    path = tmp_path / "data.zarr"
    replay.save_to_path(str(path))
    return path


@pytest.mark.parametrize("variant", VARIANTS)
def test_complete_cpu_preflight_checks_actual_data_and_train_only_normalization(variant, store, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU preflight may not inspect GPUs or DINO sources")
    monkeypatch.setattr(launch, "check_gpu_selection", forbidden)
    monkeypatch.setattr(original, "validate_dino_source", forbidden)
    cfg = launch.compose_config(variant, "real_robot", [f"task.policy.dataset.zarr_path={store}", "training.num_demo=5"])
    output = tmp_path / "not_created"
    report = launch.preflight(cfg, output, world_size=2)
    assert report["obs_encoder_type"] == "resnet18" and report["effective_batch"] == 64
    assert set(report["sources"]) == {"resnet18", "normalizer"}
    assert report["sources"]["resnet18"]["pretrained"] is False
    assert report["sources"]["resnet18"]["trainable"] is True
    assert report["sources"]["resnet18"]["external_vision_checkpoint_required"] is False
    norm = report["sources"]["normalizer"]
    assert norm["round_trip"] == "passed" and norm["train_frames"] == 80
    assert norm["train_episode_ids"] == report["dataset_split"]["train_episode_ids"]
    assert not norm["refit_on_resume"] and not norm["padded_frames"] and not norm["overlapping_windows"]
    assert not output.exists()


def test_resume_keeps_saved_options_and_rejects_dino_before_sources():
    cfg = launch.compose_config(VARIANTS[0], "real_robot", ["training.num_epochs=17"])
    overrides = ["training.resume=true", "training.resume_checkpoint=/tmp/saved.ckpt"]
    resumed = launch.compose_resume_config({"cfg": cfg}, VARIANTS[0], "real_robot", overrides)
    assert resumed.training.num_epochs == 17 and resumed.policy.obs_encoder_type == "resnet18"
    with pytest.raises(ValueError, match="observation encoder"):
        launch.compose_resume_config({"cfg": original.compose_config(VARIANTS[0], "real_robot")},
                                     VARIANTS[0], "real_robot", overrides)
    with pytest.raises(ValueError, match="observation encoder"):
        launch.compose_resume_config({"cfg": cfg}, VARIANTS[0], "real_robot",
                                     [*overrides, "policy.obs_encoder_type=dinov3"])


def _resume_payload(cfg):
    from oat.workspace.train_p2n_action_flow_resnet18 import TrainP2NActionFlowResNet18Workspace
    policy = OmegaConf.to_container(cfg.policy, resolve=True)
    policy["construction_mode"] = "restore"
    return {"cfg": cfg, "metadata": {"policy_family": "continuous_action_flow", "artifact_schema_version": 1,
            "variant": cfg.variant, "task_type": cfg.task_type, "obs_encoder_type": "resnet18"},
            "policy_config": policy,
            "state_dicts": {"model": {}, "ema_model": {}, "optimizer": {}},
            "pickles": {key: dill.dumps(None) for key in TrainP2NActionFlowResNet18Workspace.include_keys}}


@pytest.mark.parametrize("location", ("metadata", "policy_config", "cfg"))
def test_workspace_rejects_encoder_mismatch_in_each_artifact_contract(location):
    from oat.workspace.train_p2n_action_flow_resnet18 import TrainP2NActionFlowResNet18Workspace as Workspace
    cfg = launch.compose_config(VARIANTS[0], "real_robot")
    payload = _resume_payload(cfg)
    assert Workspace.validate_resume_payload(payload, cfg) is payload
    payload = copy.deepcopy(payload)
    target = payload[location].policy if location == "cfg" else payload[location]
    target["obs_encoder_type"] = "dinov3"
    with pytest.raises(ValueError, match="observation encoder"):
        Workspace.validate_resume_payload(payload, cfg)


def test_resume_preflight_checks_split_and_world_size_without_refitting(monkeypatch, tmp_path):
    cfg = launch.compose_config(VARIANTS[0], "real_robot", ["training.resume=true", "training.resume_checkpoint=/missing/saved.ckpt"])
    payload = _resume_payload(cfg)
    split = {"identity": "test", "train_episode_ids": [0, 1], "validation_episode_ids": [2],
             "train": {"windows": 32}, "validation": {"windows": 8}}
    payload["metadata"]["dataset_split"] = copy.deepcopy(split)
    payload["training_state"] = {"rng_states": [{}, {}]}
    monkeypatch.setattr(launch, "inspect_dataset", lambda cfg: split)
    def forbidden(*args, **kwargs):
        raise AssertionError("Resume must preserve its normalization")
    monkeypatch.setattr(launch, "inspect_training_normalizer", forbidden)
    report = launch.preflight(cfg, tmp_path / "out", 2, resume_payload=payload)
    assert report["sources"]["external_frozen_sources_required"] is False
    with pytest.raises(ValueError, match="world size"):
        launch.preflight(cfg, tmp_path / "out", 1, resume_payload=payload)
    payload["metadata"]["dataset_split"]["identity"] = "changed"
    with pytest.raises(ValueError, match="dataset identity"):
        launch.preflight(cfg, tmp_path / "out", 2, resume_payload=payload)


def test_python_dry_run_preserves_flags_and_hydra_precedence_without_launch(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("Dry-run cannot probe GPU or launch workers")
    monkeypatch.setattr(launch, "check_gpu_selection", forbidden)
    monkeypatch.setattr(launch.subprocess, "run", forbidden)
    monkeypatch.setattr(launch, "preflight", lambda cfg, output, world_size: {
        "batch": cfg.dataloader.batch_size, "validation": cfg.val_dataloader.batch_size,
        "accum": cfg.training.gradient_accumulate_every, "world_size": world_size})
    output = tmp_path / "uncreated"
    report = launch.main(["--variant", VARIANTS[0], "--task", "real_robot", "--gpus", "2,3",
        "--batch-size", "4", "--val-batch-size", "4", "--grad-accum", "8", "--dry-run",
        "--output", str(output), "--", "dataloader.batch_size=8"])
    assert report == {"batch": 8, "validation": 4, "accum": 8, "world_size": 2}
    assert not output.exists()


def _bash_capture(tmp_path, *args):
    recorder, calls_file = tmp_path / "capture.py", tmp_path / "calls.jsonl"
    recorder.write_text(f"#!{sys.executable}\nimport json, os, sys\nwith open(os.environ['P2N_CAPTURE'], 'a') as f:\n    f.write(json.dumps(sys.argv[1:]) + '\\n')\n")
    recorder.chmod(0o755)
    env = {key: value for key, value in os.environ.items() if not key.startswith("FLOW_")}
    env["P2N_CAPTURE"] = str(calls_file)
    result = subprocess.run(["bash", str(ROOT / "train_p2n_action_flow_resnet18.sh"),
                             "--python", str(recorder), *args], env=env, text=True, capture_output=True)
    calls = [json.loads(line) for line in calls_file.read_text().splitlines()] if calls_file.exists() else []
    return result, calls


def test_shell_runs_both_sequentially_with_distinct_output_and_flag_overrides(tmp_path):
    result, calls = _bash_capture(tmp_path, "--gpus=2,3", "--batch-size", "4", "--val-batch-size", "4",
        "--grad-accum", "8", "--dry-run", "--", "dataloader.batch_size=8", "training.num_epochs=9")
    assert result.returncode == 0, result.stderr
    assert len(calls) == 2
    outputs = []
    for args, variant in zip(calls, VARIANTS):
        assert args[0].endswith("scripts/train_p2n_action_flow_resnet18.py")
        assert args[args.index("--variant") + 1] == variant
        assert args[args.index("--gpus") + 1] == "2,3"
        assert "--dino" not in args and "--tokenizer" not in args
        assert {"dataloader.batch_size=4", "val_dataloader.batch_size=4", "training.gradient_accumulate_every=8"} <= set(args)
        assert args[-2:] == ["dataloader.batch_size=8", "training.num_epochs=9"]
        outputs.append(args[args.index("--output") + 1])
    assert outputs[0] != outputs[1] and all("resnet18" in path for path in outputs)


@pytest.mark.parametrize("args", [("--dino", "/missing"), ("--batch-size", "2"), ("--resume", "/missing")])
def test_shell_rejects_unsupported_source_invalid_batch_or_ambiguous_resume(tmp_path, args):
    result, calls = _bash_capture(tmp_path, *args)
    assert result.returncode == 2 and not calls
