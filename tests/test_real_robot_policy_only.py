"""Verify fresh single-GPU policies reuse the correct trained tokenizers."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest

from oat.common.hydra_util import register_new_resolvers

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "train_real_robot_policy", ROOT / "scripts/train_real_robot_policy.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def compose_command(command):
    config_arg = next(arg for arg in command if arg.startswith("--config-name="))
    register_new_resolvers()
    with initialize_config_dir(config_dir=str(ROOT / "oat/config"), version_base=None):
        cfg = compose(config_name=config_arg.split("=", 1)[1],
                      overrides=command[command.index(config_arg) + 1:])
    OmegaConf.resolve(cfg)
    return cfg


@pytest.mark.parametrize("variant", ["current", "left_noise"])
@pytest.mark.parametrize("smoke", [False, True])
def test_policy_only_command_is_fresh_single_gpu_and_logs_natively(tmp_path, variant, smoke):
    checkpoint = tmp_path / "frozen_tokenizer.ckpt"
    command = launcher.policy_command("fruits", variant, tmp_path, checkpoint,
                                      "test-entity", "real_robot", "unique-run-id",
                                      "fresh-policy", smoke=smoke)
    cfg = compose_command(command)
    assert "--nproc_per_node=1" in command
    assert "--config-name=train_past2next_scratch_all500" in command
    assert all("train_oattok" not in arg for arg in command)
    assert cfg.dataloader.batch_size == cfg.val_dataloader.batch_size == 64
    assert cfg.training.resume is False
    assert cfg.training.init_checkpoint is None
    assert cfg.policy.action_tokenizer.checkpoint == str(checkpoint)
    assert cfg.task.policy.dataset.seed == 42
    assert cfg.task.policy.dataset.val_ratio == 0.1
    assert cfg.logging.entity == "test-entity"
    assert cfg.logging.project == "real_robot"
    assert cfg.logging.group == f"fruits_{variant}_single_gpu"
    assert cfg.logging.id == "unique-run-id"
    assert cfg.logging.name == "fresh-policy"
    assert cfg.logging.resume is False
    assert cfg.logging.mode == ("offline" if smoke else "online")
    assert cfg.training.num_epochs == (1 if smoke else 251)
    if smoke:
        assert cfg.training.max_train_steps == 2
        assert cfg.training.max_val_steps == cfg.training.max_reconst_steps == 1
        assert cfg.policy.self_past_p == 1.0
        assert cfg.policy.self_past_warmup_steps == cfg.policy.self_past_ramp_steps == 0


def test_tokenizer_pairing_rejects_other_variant_task_and_shape(tmp_path):
    cfg = compose_command(launcher.stage_command(
        "tokenizer", "fruits", "current", tmp_path, 2))
    assert launcher.verify_tokenizer_config(cfg, "fruits", "current")["action_dim"] == 7
    with pytest.raises(ValueError, match="action_aug.mode"):
        launcher.verify_tokenizer_config(cfg, "fruits", "left_noise")
    with pytest.raises(ValueError, match="task_name"):
        launcher.verify_tokenizer_config(cfg, "nut_washer", "current")
    cfg.action_dim = 12
    with pytest.raises(ValueError, match="action_dim"):
        launcher.verify_tokenizer_config(cfg, "fruits", "current")


def test_single_gpu_rejects_lists_and_invalid_indices():
    assert launcher.single_gpu("0") == 0
    assert launcher.single_gpu("7") == 7
    for value in ("0,1", "-1", "0 1", "", "cuda:0"):
        with pytest.raises(argparse.ArgumentTypeError):
            launcher.single_gpu(value)


def test_policy_only_run_copies_tokenizer_checks_final_checkpoint_and_cleans_wandb(
        tmp_path, monkeypatch):
    source = tmp_path / "original.ckpt"
    source.write_bytes(b"existing trained tokenizer")
    output = tmp_path / "new_policy"
    monkeypatch.setattr(launcher, "verify_tokenizer", lambda *args: {"weights_finite": True})
    monkeypatch.setattr(launcher, "dataset_manifest", lambda task: {"task": task})
    monkeypatch.setattr(sys, "argv", [
        "train_real_robot_policy.py", "--task", "fruits", "--variant", "current",
        "--tokenizer", str(source), "--gpu", "3", "--output-dir", str(output),
        "--entity", "test-entity",
    ])
    stale_keys = ("WANDB_RUN_ID", "WANDB_RESUME", "WANDB_SERVICE", "_WANDB_SERVICE")
    for key in stale_keys:
        monkeypatch.setenv(key, "old-run")
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "3"
        assert kwargs["env"]["WANDB_MODE"] == "online"
        assert not any(key in kwargs["env"] for key in stale_keys)
        if any(arg.endswith("check_real_robot_checkpoint.py") for arg in command):
            Path(command[command.index("--output") + 1]).write_text('{"ok": true}')
            return
        assert "--config-name=train_past2next_scratch_all500" in command
        assert (output / "frozen_tokenizer.ckpt").read_bytes() == source.read_bytes()
        stage_dir = output / "policy"
        (stage_dir / "checkpoints").mkdir(parents=True)
        (stage_dir / "checkpoints/ep-0000_mse-1.250000.ckpt").write_bytes(b"policy")
        (stage_dir / "logs.json").write_text('{"epoch": 0, "test_reconst_mse": 1.25}\n')

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    launcher.main()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["tokenizer"]["source_sha256"] == manifest["tokenizer"]["copied_sha256"]
    assert manifest["tokenizer"]["retrained"] is False
    assert list(manifest["commands"]) == ["policy"]
    assert len(calls) == 2
    assert not (output / "tokenizer").exists()
    status = json.loads((output / "status.json").read_text())
    assert status["state"] == "completed"
    assert Path(status["checkpoint_check"]).is_file()
    with pytest.raises(FileExistsError, match="existing output"):
        launcher.main()


def test_resume_preserves_manifest_run_identity_and_appends_log(tmp_path, monkeypatch):
    source = tmp_path / "tokenizer.ckpt"
    source.write_bytes(b"trained tokenizer")
    output = tmp_path / "run"
    monkeypatch.setattr(launcher, "verify_tokenizer", lambda *args: {"verified": True})
    monkeypatch.setattr(launcher, "dataset_manifest", lambda task: {"task": task})
    args = argparse.Namespace(task="fruits", variant="current", tokenizer=source, gpu=0,
                              output_dir=output, entity="test-entity", project="real_robot",
                              policy_epochs=251, smoke=False)
    manifest = launcher.run_manifest(args)
    output.mkdir()
    frozen = output / "frozen_tokenizer.ckpt"
    frozen.write_bytes(source.read_bytes())
    manifest["tokenizer"]["copied_sha256"] = launcher.sha256_file(frozen)
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    original_bytes = manifest_path.read_bytes()
    (output / "policy.log").write_text("previous training log\n")
    checkpoints = output / "policy/checkpoints"
    checkpoints.mkdir(parents=True)
    latest = checkpoints / "latest.ckpt"
    latest.write_bytes(b"complete v2 checkpoint")
    metadata = {"path": str(latest), "sha256": launcher.sha256_file(latest),
                "next_epoch": 48, "completed_optimizer_steps": 11616, "checkpoint_version": 2}
    monkeypatch.setattr(launcher, "verify_resume_checkpoint", lambda path, cfg: metadata)
    monkeypatch.setattr(sys, "argv", [
        "train_real_robot_policy.py", "--task", "fruits", "--variant", "current",
        "--tokenizer", str(source), "--gpu", "0", "--output-dir", str(output),
        "--entity", "test-entity", "--resume",
    ])

    def fake_resume(command, **kwargs):
        if any(arg.endswith("check_real_robot_checkpoint.py") for arg in command):
            Path(command[command.index("--output") + 1]).write_text('{"ok": true}')
            return
        cfg = compose_command(command)
        assert cfg.training.resume is True
        assert cfg.training.resume_checkpoint == str(latest)
        assert cfg.training.num_epochs == 251
        assert cfg.training.init_checkpoint is None
        assert cfg.logging.resume == "must"
        assert cfg.logging.id == manifest["wandb"]["run_id"]
        assert cfg.logging.name == manifest["wandb"]["run_name"]
        assert cfg.logging.entity == manifest["wandb"]["entity"]
        assert cfg.checkpoint.topk.monitor_key == "test_reconst_mse"
        assert cfg.checkpoint.topk.k == 0
        assert cfg.checkpoint.save_all is True
        assert cfg.training.checkpoint_every == 20
        assert cfg.training.snapshot_every == 0
        assert cfg.checkpoint.topk.format_str == "ep-{epoch:04d}_mse-{test_reconst_mse:.6f}.ckpt"
        assert kwargs["stdout"].mode == "a"
        kwargs["stdout"].write("resumed training log\n")
        (checkpoints / "ep-0048_mse-0.000100.ckpt").write_bytes(b"resumed policy")
        (output / "policy/logs.json").write_text('{"epoch": 48, "test_reconst_mse": 0.0001}\n')

    monkeypatch.setattr(launcher.subprocess, "run", fake_resume)
    launcher.main()
    assert manifest_path.read_bytes() == original_bytes
    assert (output / "policy.log").read_text() == "previous training log\nresumed training log\n"
    status = json.loads((output / "status.json").read_text())
    assert status["state"] == "completed"
    assert status["policy_mse"] == 0.0001
    assert status["resume_checkpoint"]["completed_optimizer_steps"] == 11616
    assert status["resume_count"] == 1
    assert Path(status["resume_manifest"]).is_file()
    args.gpu = 1
    with pytest.raises(ValueError, match="Resume gpu"):
        launcher.resume_manifest(args)


def test_resume_rejects_incomplete_version_two_checkpoint(tmp_path):
    import dill
    import torch

    saved = {"checkpoint_version": 2, "epoch": 1, "global_step": 242,
             "completed_optimizer_steps": 242, "ema_state": {"optimization_step": 242, "decay": 0.9},
             "lr_scheduler_state": None}
    checkpoint = tmp_path / "incomplete.ckpt"
    torch.save({"pickles": {key: dill.dumps(value) for key, value in saved.items()}},
               checkpoint, pickle_module=dill)
    cfg = OmegaConf.create({"training": {"num_epochs": 251}})
    with pytest.raises(ValueError, match="scheduler state"):
        launcher.verify_resume_checkpoint(checkpoint, cfg)
