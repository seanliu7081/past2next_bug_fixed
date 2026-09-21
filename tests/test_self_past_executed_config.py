"""Preserve offline self-past training while opting into acknowledged inference."""
import inspect
import os
from pathlib import Path
import subprocess
import sys

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
import numpy as np
from omegaconf import OmegaConf
import torch

from oat.common.replay_buffer import ReplayBuffer
from oat.dataset.zarr_dataset_with_prev_window import ZarrDatasetWithPrevWindow
from oat.policy.past2next_self_past import Past2NextSelfPastPolicy


ROOT = Path(__file__).resolve().parents[1]
CONFIG_NAME = "experimental/train_past2next_self_past_executed"
RECIPE_NAME = "train_past2next_self_past_executed"


def configuration(name=CONFIG_NAME):
    with initialize_config_dir(config_dir=str(ROOT / "oat/config"), version_base=None):
        return compose(config_name=name, overrides=[
            "policy.action_tokenizer.checkpoint=/tmp/config-only-tokenizer.ckpt",
        ])


def test_standalone_recipe_preserves_every_scratch_training_setting():
    cfg = configuration()
    baseline = configuration("train_past2next_scratch")
    assert cfg.name == RECIPE_NAME
    assert cfg.policy._target_ == "oat.policy.past2next_self_past_executed.Past2NextSelfPastExecutedPolicy"
    assert cfg.task.policy.env_runner._target_ == "oat.env_runner.executed_action_runner.LiberoExecutedPastRunner"
    assert cfg.task.policy.dataset._target_ == "oat.dataset.zarr_dataset_with_prev_window.ZarrDatasetWithPrevWindow"
    assert cfg.seed == cfg.training.seed == cfg.task.policy.dataset.seed == 42
    assert cfg.task.policy.dataset.val_ratio == 0.1
    assert (cfg.horizon, cfg.n_action_steps, cfg.n_obs_steps, cfg.past_n) == (16, 8, 2, 7)
    assert cfg.task.policy.dataset.n_exec_steps == 8
    assert cfg.task.policy.dataset.history_padding == "zero"
    assert cfg.task.policy.dataset.return_history_validity is True
    assert cfg.task.policy.env_runner.protocol == "corrected"
    assert cfg.training.validate_generated_history
    assert not cfg.training.resume and cfg.training.init_checkpoint is None
    assert cfg.policy.self_past_p == 0.5
    assert cfg.policy.self_past_warmup_steps == 1000
    assert cfg.policy.self_past_ramp_steps == 4000
    assert cfg.policy.self_past_schedule == "optimizer_step"
    assert cfg.policy.self_past_temperature == 0.0
    assert cfg.policy.self_past_topk == 10

    policy = OmegaConf.to_container(cfg.policy, resolve=True)
    old_policy = OmegaConf.to_container(baseline.policy, resolve=True)
    assert {k: v for k, v in policy.items() if k != "_target_"} == {
        k: v for k, v in old_policy.items() if k != "_target_"
    }
    # The new policy preserves the existing self-past constructor contract.
    # Binding real composed kwargs requires no checkpoint loading or GPU.
    inspect.signature(Past2NextSelfPastPolicy.__init__).bind(
        None, **{key: value for key, value in policy.items() if key != "_target_"})
    for section in ("training", "optimizer", "dataloader", "val_dataloader", "ema", "checkpoint"):
        assert OmegaConf.to_container(cfg[section], resolve=True) == OmegaConf.to_container(baseline[section], resolve=True)
    task = OmegaConf.to_container(cfg.task.policy, resolve=True)
    original_task = OmegaConf.to_container(baseline.task.policy, resolve=True)
    task["env_runner"]["_target_"] = original_task["env_runner"]["_target_"]
    assert task == original_task
    source = OmegaConf.load(ROOT / "oat/config" / f"{CONFIG_NAME}.yaml")
    assert list(source.defaults) == [
        {"/task/policy": "libero/libero10_with_prev_window"}, "_self_",
    ]
    assert not (ROOT / "oat/config" / f"{RECIPE_NAME}.yaml").exists()


def test_composed_dataset_keeps_previous_windows_for_offline_self_past(tmp_path):
    cfg = configuration()
    replay = ReplayBuffer.create_empty_numpy()
    for episode in range(10):
        values = (100 * (episode + 1) + np.arange(24)).astype(np.float32)
        actions = np.repeat(values[:, None], 7, axis=1)
        replay.add_episode({"action": actions, "state": actions.copy()})
    path = tmp_path / "demonstrations.zarr"
    replay.save_to_path(str(path))
    dataset = instantiate(cfg.task.policy.dataset, zarr_path=str(path), obs_keys=["state"])
    assert type(dataset) is ZarrDatasetWithPrevWindow
    assert dataset.train_mask.sum() == 9
    assert dataset.get_validation_dataset().train_mask.sum() == 1
    first, aligned, later = dataset[0], dataset[8], dataset[16]
    assert first["episode_step"] == 0
    assert torch.equal(first["past_action"], torch.zeros(7, 7))
    assert not first["past_action_valid"].any()
    assert not first["prev_window_valid"]
    assert not dataset[7]["prev_window_valid"]
    assert aligned["prev_window_valid"]
    assert torch.equal(aligned["prev_past_action"], torch.zeros(7, 7))
    assert not aligned["prev_past_action_valid"].any()
    current = later["action"][0, 0].item()
    torch.testing.assert_close(later["past_action"][:, 0], torch.arange(current - 7, current))
    torch.testing.assert_close(later["prev_past_action"][:, 0], torch.arange(current - 15, current - 8))
    assert later["prev_obs"]["state"][-1, 0].item() == current - 8
    assert later["past_action_valid"].all() and later["prev_past_action_valid"].all()
    assert first["action"].shape == later["action"].shape == (16, 7)


def run_dry_launcher(tmp_path, checkpoint, *extra):
    return subprocess.run(
        ["bash", str(ROOT / "train_self_past_executed.sh"), "--dry-run", str(checkpoint), *extra],
        cwd=tmp_path, env={**os.environ, "TRAIN_PY": sys.executable,
                           "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True, text=True, timeout=60,
    )


def test_launcher_dry_run_handles_literal_paths_without_loading_checkpoint(tmp_path):
    checkpoint = tmp_path / "tokenizer directory ; literal" / 'not a "real", checkpoint.ckpt'
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"config dry-run must never load these bytes")
    output = tmp_path / "new output ; literal directory"
    result = run_dry_launcher(tmp_path, checkpoint.relative_to(tmp_path), output.name)
    assert result.returncode == 0, result.stdout + result.stderr
    header, output_line, config_text = result.stdout.split("\n", 2)
    assert header == f"Tokenizer: {checkpoint}"
    assert output_line == f"Output directory: {output}"
    cfg = OmegaConf.create(config_text)
    assert cfg.name == RECIPE_NAME
    assert cfg.policy.action_tokenizer.checkpoint == str(checkpoint)
    assert cfg.seed == cfg.task.policy.dataset.seed == 42
    assert cfg.training.validate_generated_history
    assert cfg.policy.self_past_p == 0.5
    assert cfg.task.policy.env_runner._target_ == "oat.env_runner.executed_action_runner.LiberoExecutedPastRunner"
    assert not output.exists()


def test_launcher_default_output_is_fresh_and_missing_checkpoint_fails(tmp_path):
    checkpoint = tmp_path / "tokenizer.ckpt"
    checkpoint.write_bytes(b"not loaded in dry-run")
    outputs = []
    for _ in range(2):
        result = run_dry_launcher(tmp_path, checkpoint)
        assert result.returncode == 0, result.stdout + result.stderr
        output = Path(result.stdout.splitlines()[1].removeprefix("Output directory: "))
        assert output.parent == ROOT / "output/training"
        assert output.name.startswith("self_past_executed_seed42_")
        assert not output.exists()
        outputs.append(output)
    assert outputs[0] != outputs[1]
    missing = run_dry_launcher(tmp_path, tmp_path / "missing.ckpt")
    assert missing.returncode == 2
    assert "Tokenizer checkpoint not found" in missing.stderr
