"""Keep gate recipes intact through the shared tokenizer-to-policy launcher."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pytest
import yaml
import zarr

from scripts.train_real_robot import best_policy_checkpoint


ROOT = Path(__file__).resolve().parents[1]
GATE_CONFIG = "experimental/train_past2next_state_history_gate_real_robot"
VAL_FILENAME = "ep-{epoch:04d}_val-{val_loss:.6f}.ckpt"
MSE_FILENAME = "ep-{epoch:04d}_mse-{test_reconst_mse:.6f}.ckpt"


@pytest.fixture
def robot_data(tmp_path):
    """Small valid episode metadata and lazily filled camera arrays."""
    path = tmp_path / "robot data, fixture.zarr"
    root = zarr.open_group(str(path), mode="w")
    root.create_dataset("meta/episode_ends", data=np.array([2, 4, 6, 8], dtype=np.int64))
    shapes = {
        "action": (7,), "agentview_rgb": (128, 128, 3),
        "robot0_eye_in_hand_rgb": (128, 128, 3), "robot0_eef_pos": (3,),
        "robot0_eef_rot6d": (6,), "robot0_gripper_qpos": (1,), "task_uid": (1,),
    }
    for key, shape in shapes.items():
        root.create_dataset(f"data/{key}", shape=(8, *shape),
                            dtype="uint8" if key.endswith("_rgb") else "float32")
    root["data/robot0_eef_rot6d"][:] = np.tile(
        np.array([1, 0, 0, 0, 1, 0], dtype=np.float32), (8, 1))
    return path


def dry_run(tmp_path, dataset, policy_config, *, policy_epochs=None):
    output = tmp_path / "new run, no aug"
    env = os.environ.copy()
    for key in ("TOKENIZER_EPOCHS", "POLICY_EPOCHS", "RUN_DIR"):
        env.pop(key, None)
    env.update({"TRAIN_PY": sys.executable, "WANDB_MODE": "offline"})
    if policy_epochs is not None:
        env["POLICY_EPOCHS"] = str(policy_epochs)
    # Make accidental GPU initialization fail, even on hosts with idle GPUs.
    guard_dir = tmp_path / "cpu_guard"
    guard_dir.mkdir()
    (guard_dir / "sitecustomize.py").write_text(
        "import torch\n"
        "def reject_cuda(*args, **kwargs):\n"
        "    raise AssertionError('dry-run must not initialize CUDA')\n"
        "torch.cuda._lazy_init = reject_cuda\n"
    )
    env["PYTHONPATH"] = os.pathsep.join(
        str(value) for value in (guard_dir, ROOT, env.get("PYTHONPATH", "")) if value)
    result = subprocess.run(
        ["bash", str(ROOT / "train_pen_cabinet.sh"),
         "--tokenizer-config", "oattok", "--dataset", str(dataset),
         "--policy-config", policy_config, "--gpus", "4,5",
         "--output-dir", str(output), "--dry-run"],
        cwd=tmp_path, env=env, text=True, capture_output=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    # Hydra emits the two YAML mappings back-to-back without a separator.
    policy_start = re.search(r"(?m)^task:\n  policy:", result.stdout)
    assert policy_start is not None, result.stdout
    blocks = (result.stdout[:policy_start.start()], result.stdout[policy_start.start():])
    tokenizer, policy = [yaml.safe_load(block) for block in blocks]
    assert not output.exists()
    assert "${" not in result.stdout
    assert "4 episodes, 8 frames; 3 train / 1 validation" in result.stderr
    for stage, cfg in (("tokenizer", tokenizer), ("policy", policy)):
        assert cfg["task"][stage]["dataset"]["zarr_path"] == str(dataset)
        assert cfg["training"]["num_demo"] == 4
        assert cfg["training"]["resume"] is False
        assert cfg["val_dataloader"]["drop_last"] is False
        assert cfg["logging"]["mode"] == "offline"
    assert tokenizer["tokenizer"]["_target_"] == "oat.tokenizer.oat.tokenizer.OATTok"
    assert "action_aug" not in tokenizer["tokenizer"]
    assert tokenizer["training"]["num_epochs"] == 3001
    assert tokenizer["dataloader"]["batch_size"] == 256
    assert tokenizer["val_dataloader"]["batch_size"] == 128
    assert policy["policy"]["action_tokenizer"]["checkpoint"] == str(output / "frozen_tokenizer.ckpt")
    assert policy["training"]["init_checkpoint"] is None
    return tokenizer, policy


@pytest.mark.parametrize("policy_epochs", [None, 17])
def test_gate_dry_run_preserves_recipe_and_uses_new_frozen_tokenizer(
        tmp_path, robot_data, policy_epochs):
    _, policy = dry_run(tmp_path, robot_data, GATE_CONFIG + ".yaml",
                        policy_epochs=policy_epochs)
    assert policy["policy"]["_target_"] == (
        "oat.policy.past2next_state_history_gate_real_robot."
        "Past2NextRealRobotStateHistoryGatePolicy")
    assert policy["policy"]["history_gate_mode"] == "learned"
    assert policy["policy"]["rotation_6d_layout"] == "rows"
    assert policy["training"]["num_epochs"] == (2001 if policy_epochs is None else policy_epochs)
    assert policy["training"]["checkpoint_every"] == 100
    assert policy["dataloader"]["batch_size"] == policy["val_dataloader"]["batch_size"] == 32
    assert policy["checkpoint"]["topk"] == {
        "monitor_key": "val_loss", "mode": "min", "k": 0, "format_str": VAL_FILENAME,
    }
    assert policy["checkpoint"]["save_all"] is True
    dataset = policy["task"]["policy"]["dataset"]
    assert dataset["_target_"] == "oat.dataset.real_robot_state_history.RealRobotZarrDatasetWithStateHistory"
    assert dataset["state_history_steps"] == policy["policy"]["state_history_steps"] == 8
    assert dataset["state_history_keys"] == ["robot0_eef_pos", "robot0_eef_rot6d", "robot0_gripper_qpos"]
    assert dataset["history_padding"] == "zero"
    assert dataset["return_history_validity"] is True
    assert policy["past_n"] == 7
    assert policy["task"]["policy"]["env_runner"] is None


def test_baseline_dry_run_keeps_existing_training_recipe(tmp_path, robot_data):
    _, policy = dry_run(tmp_path, robot_data, "train_past2next_scratch_all500")
    assert policy["policy"]["_target_"] == "oat.policy.past2next_self_past.Past2NextSelfPastPolicy"
    assert policy["training"]["num_epochs"] == 1001
    assert policy["training"]["checkpoint_every"] == 50
    assert policy["dataloader"]["batch_size"] == 64
    assert policy["val_dataloader"]["batch_size"] == 32
    assert policy["checkpoint"]["topk"] == {
        "monitor_key": "test_reconst_mse", "mode": "min", "k": 0, "format_str": MSE_FILENAME,
    }
    assert policy["checkpoint"]["save_all"] is True


def test_policy_checkpoint_selection_follows_gate_loss_instead_of_action_mse(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    records = [
        {"epoch": 0, "val_loss": 2.0000004, "test_reconst_mse": .01},
        {"epoch": 100, "val_loss": 2.0000001, "test_reconst_mse": .02},
        {"epoch": 200, "val_loss": float("nan"), "test_reconst_mse": .03},
        {"epoch": 300, "val_loss": .1, "test_reconst_mse": .001},  # Pruned.
    ]
    (tmp_path / "logs.json").write_text("".join(json.dumps(record) + "\n" for record in records))
    for record in records[:-1]:
        for filename in (VAL_FILENAME, MSE_FILENAME):
            (checkpoint_dir / filename.format(**record)).touch()
    gate_path, gate_metric = best_policy_checkpoint(
        tmp_path, metric_name="val_loss", filename=VAL_FILENAME)
    assert gate_path.name == "ep-0100_val-2.000000.ckpt"
    assert gate_metric == 2.0000001
    baseline_path, baseline_metric = best_policy_checkpoint(tmp_path)
    assert baseline_path.name == "ep-0000_mse-0.010000.ckpt"
    assert baseline_metric == .01
