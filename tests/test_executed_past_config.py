"""The new recipe uses real demonstration history and preserves baseline settings."""
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
from oat.dataset.zarr_dataset_with_past import ZarrDatasetWithPastAction
from oat.policy.past2next import Past2NextPolicy


ROOT = Path(__file__).resolve().parents[1]
CONFIG_NAME = "experimental/train_past2next_executed_past"
RECIPE_NAME = "train_past2next_executed_past"


def configuration(name=CONFIG_NAME):
    with initialize_config_dir(config_dir=str(ROOT / "oat/config"), version_base=None):
        return compose(config_name=name, overrides=[
            "policy.action_tokenizer.checkpoint=/tmp/config-only-tokenizer.ckpt",
        ])


def test_standalone_recipe_preserves_baseline_and_removes_synthetic_history():
    cfg = configuration()
    baseline = configuration("train_past2next_scratch")
    assert cfg.policy._target_ == "oat.policy.past2next_executed_past.Past2NextExecutedPastPolicy"
    assert cfg.task.policy.env_runner._target_ == "oat.env_runner.executed_action_runner.LiberoExecutedPastRunner"
    assert cfg.task.policy.dataset._target_ == "oat.dataset.zarr_dataset_with_past.ZarrDatasetWithPastAction"
    assert cfg.seed == cfg.training.seed == cfg.task.policy.dataset.seed == 42
    assert cfg.task.policy.dataset.val_ratio == 0.1
    assert (cfg.horizon, cfg.n_action_steps, cfg.n_obs_steps, cfg.past_n) == (16, 8, 2, 7)
    assert cfg.task.policy.dataset.history_padding == "zero"
    assert cfg.task.policy.dataset.return_history_validity is True
    assert "n_exec_steps" not in cfg.task.policy.dataset
    assert cfg.task.policy.env_runner.protocol == "corrected"
    assert not cfg.training.validate_generated_history
    assert not cfg.training.resume and cfg.training.init_checkpoint is None

    policy = OmegaConf.to_container(cfg.policy, resolve=True)
    old_policy = OmegaConf.to_container(baseline.policy, resolve=True)
    assert not any(key.startswith("self_past_") for key in policy)
    assert {k: v for k, v in policy.items() if k != "_target_"} == {
        k: v for k, v in old_policy.items() if k != "_target_" and not k.startswith("self_past_")
    }
    # The executed-history class inherits this exact constructor. Binding the
    # composed kwargs catches stale self-past fields without loading a tokenizer.
    inspect.signature(Past2NextPolicy.__init__).bind(
        None, **{key: value for key, value in policy.items() if key != "_target_"})
    dataset = OmegaConf.to_container(cfg.task.policy.dataset, resolve=True)
    inspect.signature(ZarrDatasetWithPastAction.__init__).bind(
        None, **{key: value for key, value in dataset.items() if key != "_target_"})
    for section in ("optimizer", "dataloader", "val_dataloader", "ema", "checkpoint"):
        assert OmegaConf.to_container(cfg[section], resolve=True) == OmegaConf.to_container(baseline[section], resolve=True)
    assert {k: v for k, v in cfg.training.items() if k != "validate_generated_history"} == {
        k: v for k, v in baseline.training.items() if k != "validate_generated_history"
    }
    source = OmegaConf.load(ROOT / "oat/config" / f"{CONFIG_NAME}.yaml")
    assert list(source.defaults) == [
        {"/task/policy": "libero/libero10_executed_past"}, "_self_",
    ]


def test_composed_dataset_returns_only_executed_demo_history(tmp_path):
    cfg = configuration()
    replay = ReplayBuffer.create_empty_numpy()
    for episode in range(10):
        values = (100 * (episode + 1) + np.arange(24)).astype(np.float32)
        actions = np.repeat(values[:, None], 7, axis=1)
        replay.add_episode({"action": actions, "state": actions.copy()})
    path = tmp_path / "demonstrations.zarr"
    replay.save_to_path(str(path))
    dataset = instantiate(cfg.task.policy.dataset, zarr_path=str(path), obs_keys=["state"])
    assert type(dataset) is ZarrDatasetWithPastAction
    assert dataset.train_mask.sum() == 9
    assert dataset.get_validation_dataset().train_mask.sum() == 1
    first, later = dataset[0], dataset[8]
    assert first["episode_step"] == 0
    assert torch.equal(first["past_action"], torch.zeros(7, 7))
    assert not first["past_action_valid"].any()
    current = later["action"][0, 0].item()
    torch.testing.assert_close(later["past_action"][:, 0], torch.arange(current - 7, current))
    assert later["past_action_valid"].all()
    assert first["action"].shape == later["action"].shape == (16, 7)
    assert not any(key.startswith("prev_") for sample in (first, later) for key in sample)


def run_dry_launcher(tmp_path, checkpoint, *extra):
    return subprocess.run(
        ["bash", str(ROOT / "train_executed_past.sh"), "--dry-run", str(checkpoint), *extra],
        cwd=tmp_path, env={**os.environ, "TRAIN_PY": sys.executable,
                           "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True, text=True, timeout=60,
    )


def test_launcher_dry_run_resolves_caller_paths_without_loading_checkpoint(tmp_path):
    checkpoint = tmp_path / "tokenizer directory" / "not a real checkpoint.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"config dry-run must never load these bytes")
    output = tmp_path / "new output directory"
    result = run_dry_launcher(tmp_path, checkpoint.relative_to(tmp_path), output.name)
    assert result.returncode == 0, result.stdout + result.stderr
    header, output_line, config_text = result.stdout.split("\n", 2)
    assert header == f"Tokenizer: {checkpoint}"
    assert output_line == f"Output directory: {output}"
    cfg = OmegaConf.create(config_text)
    assert cfg.name == RECIPE_NAME
    assert cfg.policy.action_tokenizer.checkpoint == str(checkpoint)
    assert cfg.seed == cfg.task.policy.dataset.seed == 42
    assert not cfg.training.validate_generated_history
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
        assert output.name.startswith("executed_past_seed42_")
        assert not output.exists()
        outputs.append(output)
    assert outputs[0] != outputs[1]
    missing = run_dry_launcher(tmp_path, tmp_path / "missing.ckpt")
    assert missing.returncode == 2
    assert "Tokenizer checkpoint not found" in missing.stderr
