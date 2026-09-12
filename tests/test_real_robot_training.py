"""Verify the single-task experiment matrix and tokenizer-to-policy handoff."""
import importlib.util
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf
import pytest
import zarr

from oat.common.hydra_util import register_new_resolvers
from oat.common.seq_sampler import get_val_mask

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "train_real_robot", ROOT / "scripts/train_real_robot.py")
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


@pytest.mark.parametrize("task,episodes", [("fruits", 51), ("nut_washer", 62)])
@pytest.mark.parametrize("variant", ["current", "left_noise"])
@pytest.mark.parametrize("stage,global_batch", [("tokenizer", 256), ("policy", 64)])
def test_experiment_matrix_preserves_requested_policy_and_isolates_tasks(
        tmp_path, task, episodes, variant, stage, global_batch):
    frozen = tmp_path / "comma, and space" / "frozen.ckpt"
    cfg = compose_command(launcher.stage_command(
        stage, task, variant, tmp_path / "run, with space", 2, frozen))
    task_cfg = cfg.task[stage]
    assert cfg.seed == task_cfg.dataset.seed == 42
    assert cfg.training.num_demo == episodes
    assert cfg.dataloader.batch_size * 2 == global_batch
    assert cfg.val_dataloader.batch_size * 2 == global_batch
    assert cfg.val_dataloader.drop_last is False
    assert cfg.training.resume is cfg.logging.resume is False
    assert task_cfg.fps == 30
    assert task_cfg.dataset.zarr_path.endswith(f"/{task}_N{episodes}.zarr")
    assert task_cfg.dataset.val_ratio == 0.1
    assert task_cfg.dataset.max_train_episodes is None
    assert task_cfg.shape_meta.action.shape == [7]
    assert task_cfg.dataset._target_.startswith("oat.dataset.real_robot_dataset.")
    if stage == "tokenizer":
        aug = launcher.VARIANTS[variant]
        assert cfg.tokenizer.action_aug.mode == aug["mode"]
        assert cfg.tokenizer.action_aug.augment_position == aug["augment_position"]
        assert cfg.tokenizer.action_aug.max_angle_deg == 30.0
        assert cfg.tokenizer.action_aug.p == 0.6
        assert cfg.training.num_epochs == 5001
    else:
        assert cfg.name == "train_past2next_scratch_all500"
        assert cfg.policy._target_ == "oat.policy.past2next_self_past.Past2NextSelfPastPolicy"
        assert cfg.policy.action_tokenizer.checkpoint == str(frozen)
        assert cfg.training.init_checkpoint is None
        assert cfg.training.num_epochs == 251
        assert cfg.optimizer.policy_lr == cfg.optimizer.obs_enc_lr == 1e-5
        assert cfg.policy.obs_encoder.vision_encoder.eval_fixed_crop is True
        assert cfg.policy.obs_encoder.vision_encoder.crop_shape == [112, 112]
        assert (cfg.n_obs_steps, cfg.past_n, cfg.horizon, cfg.n_action_steps) == (2, 7, 16, 8)
        assert task_cfg.dataset.n_exec_steps == 8
        assert task_cfg.task_uids == [0]
        assert task_cfg.lazy_eval is True
        assert "_target_" not in task_cfg.env_runner
        assert cfg.training.offline_validation_enabled is True
        assert cfg.training.validate_generated_history is True
        assert cfg.checkpoint.topk.monitor_key == "test_reconst_mse"
        assert cfg.checkpoint.topk.format_str == "ep-{epoch:04d}_mse-{test_reconst_mse:.6f}.ckpt"
        assert cfg.checkpoint.topk.mode == "min"
        assert cfg.checkpoint.topk.k == 0
        assert cfg.checkpoint.save_all is True
        assert cfg.training.checkpoint_every == 20
        assert cfg.training.snapshot_every == 0
        assert task_cfg.shape_meta.obs.robot0_eef_rot6d.shape == [6]
        assert task_cfg.shape_meta.obs.robot0_gripper_qpos.shape == [1]
        assert "robot0_eef_quat" not in task_cfg.shape_meta.obs


@pytest.mark.parametrize("stage", ["tokenizer", "policy"])
def test_smoke_limits_both_stages_and_exercises_generated_history(tmp_path, stage):
    cfg = compose_command(launcher.stage_command(
        stage, "fruits", "left_noise", tmp_path, 1, tmp_path / "tok.ckpt", smoke=True))
    assert cfg.training.num_epochs == 1
    assert cfg.training.max_train_steps == 2
    assert cfg.training.max_val_steps == cfg.training.max_reconst_steps == 1
    assert cfg.dataloader.num_workers == cfg.val_dataloader.num_workers == 0
    assert not cfg.dataloader.persistent_workers
    assert not cfg.val_dataloader.persistent_workers
    if stage == "policy":
        assert cfg.policy.self_past_p == 1.0
        assert cfg.policy.self_past_warmup_steps == cfg.policy.self_past_ramp_steps == 0


def test_checkpoint_selection_uses_finite_full_precision_retained_metrics(tmp_path):
    (tmp_path / "checkpoints").mkdir()
    records = [
        {"epoch": 0, "test_reconst_mse": float("nan")},
        {"epoch": 10, "test_reconst_mse": 0.02010},
        {"epoch": 20, "test_reconst_mse": 0.02040},
        {"epoch": 30, "test_reconst_mse": 0.00001},  # pruned
    ]
    (tmp_path / "logs.json").write_text("".join(json.dumps(r) + "\n" for r in records))
    for name in ("ep-0000_mse-nan.ckpt", "ep-0010_mse-0.020.ckpt", "ep-0020_mse-0.020.ckpt"):
        (tmp_path / "checkpoints" / name).write_bytes(b"checkpoint")
    chosen, metric = launcher.best_tokenizer_checkpoint(tmp_path)
    assert chosen.name == "ep-0010_mse-0.020.ckpt"
    assert metric == 0.02010


def test_manifest_records_sampler_split_and_rejects_mixed_task_data(tmp_path, monkeypatch):
    path = tmp_path / "fruits.zarr"
    group = zarr.open_group(str(path), mode="w")
    ends = np.arange(1, 12, dtype=np.int64) * 7
    group.create_dataset("meta/episode_ends", data=ends)
    for key, shape in {"action": (7,), **launcher.OBS_SHAPES}.items():
        group.create_dataset(f"data/{key}", shape=(77, *shape),
                             dtype="uint8" if key.endswith("_rgb") else "float32")
    monkeypatch.setitem(launcher.DATASETS, "fruits", (path, 11))
    manifest = launcher.dataset_manifest("fruits")
    mask = get_val_mask(11, 0.1, 42)
    assert manifest["split"]["validation_episode_ids"] == np.flatnonzero(mask).tolist()
    assert manifest["split"]["train_episode_ids"] == np.flatnonzero(~mask).tolist()
    assert manifest["split"]["train_frames"] + manifest["split"]["validation_frames"] == 77
    assert manifest["split"]["normalizer_fit"] == "training_episodes_only"
    group["data/task_uid"][0] = 1
    with pytest.raises(ValueError, match="one task_uid=0"):
        launcher.dataset_manifest("fruits")


def test_two_stage_run_passes_selected_frozen_checkpoint_and_records_policy(
        tmp_path, monkeypatch):
    output = tmp_path / "run"
    commands = {stage: [stage] for stage in ("tokenizer", "policy")}
    monkeypatch.setattr(launcher, "run_manifest", lambda _: {"commands": commands})
    monkeypatch.setattr("sys.argv", ["train_real_robot.py", "--task", "fruits",
                                    "--variant", "current", "--output-dir", str(output)])

    def fake_train(command, **kwargs):
        if any(arg.endswith("check_real_robot_checkpoint.py") for arg in command):
            assert command[command.index("--checkpoint") + 1].endswith(
                "ep-0000_mse-0.012346.ckpt")
            Path(command[command.index("--output") + 1]).write_text('{"ok": true}')
            return
        stage = command[0]
        checkpoint_dir = output / stage / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        if stage == "tokenizer":
            record = {"epoch": 0, "test_reconst_mse": 0.0123}
            name = "ep-0000_mse-0.012.ckpt"
        else:
            assert (output / "frozen_tokenizer.ckpt").read_bytes() == b"selected checkpoint"
            record = {"epoch": 0, "test_reconst_mse": 0.01234567, "val_loss": 2.1234567}
            name = "ep-0000_mse-0.012346.ckpt"
        (checkpoint_dir / name).write_bytes(b"selected checkpoint")
        (output / stage / "logs.json").write_text(json.dumps(record) + "\n")

    monkeypatch.setattr(launcher.subprocess, "run", fake_train)
    launcher.main()
    status = json.loads((output / "status.json").read_text())
    assert status["state"] == "completed"
    assert status["tokenizer_mse"] == 0.0123
    assert status["policy_mse"] == 0.01234567
    assert Path(status["best_policy"]).is_file()
    assert Path(status["checkpoint_check"]).is_file()


def test_policy_checkpoint_selection_uses_mse_not_validation_loss(tmp_path):
    (tmp_path / "checkpoints").mkdir()
    records = [
        {"epoch": 1, "test_reconst_mse": 0.01000012, "val_loss": 9.0},
        {"epoch": 2, "test_reconst_mse": 0.01000034, "val_loss": 1.0},
        {"epoch": 3, "test_reconst_mse": 0.000001, "val_loss": 0.5},
    ]
    (tmp_path / "logs.json").write_text("".join(json.dumps(r) + "\n" for r in records))
    for name in ("ep-0001_mse-0.010000.ckpt", "ep-0002_mse-0.010000.ckpt"):
        (tmp_path / "checkpoints" / name).write_bytes(b"checkpoint")
    chosen, metric = launcher.best_policy_checkpoint(tmp_path)
    assert chosen.name == "ep-0001_mse-0.010000.ckpt"
    assert metric == 0.01000012
