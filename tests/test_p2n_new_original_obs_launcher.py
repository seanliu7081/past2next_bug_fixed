"""Additive launcher contracts, exercised without data, weights, CUDA or robots."""
from types import SimpleNamespace
import sys

import pytest
from omegaconf import OmegaConf

from scripts import train_p2n_new_original_obs as launcher


@pytest.mark.parametrize("variant", ["p2n_new", "p2n_state_gate_new"])
@pytest.mark.parametrize("task", ["real_robot", "libero"])
def test_clean_configs_match_task_and_modern_training_recipe(variant, task):
    cfg = launcher.compose_config(variant, task)
    old = launcher.legacy.compose_config(variant, task)
    assert cfg.shape_meta == old.shape_meta
    assert cfg.task.policy.dataset == old.task.policy.dataset
    assert cfg.policy.tokenizer_checkpoint
    assert cfg.policy.obs_encoder_type == "original_fused"
    assert cfg.policy.context_schema_version == 2
    assert cfg.policy.context_layout == "original_fused_v1"
    assert cfg.policy.original_obs_config.crop_shape == [112, 112]
    assert cfg.optimizer.obs_enc_lr == 1e-5
    assert cfg.optimizer.policy_lr == old.optimizer.policy_lr
    assert cfg.training.validation_max_samples is None
    assert cfg.dataloader == old.dataloader
    assert cfg.val_dataloader == old.val_dataloader
    for field in ("num_epochs", "num_demo", "gradient_accumulate_every", "checkpoint_every", "seed"):
        assert cfg.training[field] == old.training[field]
    for field in ("embed_dim", "n_layers", "n_heads", "ffn_dim", "horizon", "past_n", "self_past_p",
                  "self_past_warmup_steps", "self_past_ramp_steps", "self_past_chunk_size"):
        assert cfg.policy[field] == old.policy[field]
    for field in cfg.policy:
        assert not field.startswith(("dino", "convnext", "resampler", "num_visual", "visual_resampler"))
    if task == "real_robot":
        assert cfg.policy.tokenizer_checkpoint == old.policy.tokenizer_checkpoint
        assert cfg.task.policy.env_runner is None
        assert cfg.task.policy.dataset.val_ratio == 0.05
        assert cfg.training.num_demo == 77
    else:
        assert cfg.task.policy.lazy_eval is False
        assert "libero10" in cfg.policy.tokenizer_checkpoint
    if variant == "p2n_new":
        assert not any(key.startswith(("history_", "state_history")) for key in cfg.policy)


def _no_runtime(*args, **kwargs):
    pytest.fail("dry-run touched a model, dataset, simulator, GPU or subprocess")


@pytest.mark.parametrize("task", ["libero", "real_robot"])
@pytest.mark.parametrize("equals", [False, True])
def test_user_flags_and_hydra_overrides_resolve_without_runtime(monkeypatch, task, equals):
    monkeypatch.setattr(launcher, "preflight", _no_runtime)
    monkeypatch.setattr(launcher, "check_gpu_idle", _no_runtime)
    monkeypatch.setattr(launcher, "inspect_simulator", _no_runtime)
    monkeypatch.setattr(launcher, "configure_training_devices", _no_runtime)
    monkeypatch.setattr(launcher.subprocess, "run", _no_runtime)
    lazy = ["--lazy-eval=false"] if equals else ["--lazy-eval", "false"]
    cfg = launcher.main(["--variant", "p2n_state_gate_new", "--task", task,
        *lazy, "--num-train-epochs", "31", "--batch-size", "3", "--test-num", "17",
        "--val-batch-size", "2", "--num-processes", "2", "--gradient-accumulation", "2",
        "--num-workers", "0", "--val-num-workers", "0", "--eval-every", "7",
        "--dry-run", "--", "dataloader.batch_size=5"])
    assert cfg.task.policy.lazy_eval is False
    assert cfg.training.num_epochs == 31
    assert cfg.dataloader.batch_size == 5  # Explicit Hydra overrides take precedence.
    assert cfg.val_dataloader.batch_size == 2
    assert cfg.training.gradient_accumulate_every == 2
    assert cfg.training.rollout_every == 7
    assert cfg.dataloader.persistent_workers is False
    assert cfg.val_dataloader.persistent_workers is False
    if task == "libero":
        assert cfg.task.policy.env_runner.n_test == 17
        assert cfg.training.validation_max_samples is None
    else:
        assert cfg.training.validation_max_samples == 17
        assert cfg.task.policy.env_runner is None
        assert launcher.evaluation_summary(cfg)["physical_robot_rollout"] is False


@pytest.mark.parametrize("override,match", [
    ("policy.embed_dim=256", "embed_dim"),
    ("policy.context_schema_version=1", "context_schema_version"),
    ("policy.context_layout=legacy", "context_layout"),
    ("+policy.num_visual_queries=64", "num_visual_queries"),
    ("+policy.dino_path=null", "dino_path"),
    ("policy.original_obs_config.pretrained=true", "pretrained"),
    ("policy.original_obs_config.crop_shape=[129,112]", "crop_shape"),
    ("policy.original_obs_config.crop_shape=[128,112]", "crop_shape"),
    ("task.policy.lazy_eval=invalid", "lazy_eval"),
    ("training.validation_max_samples=0", "validation_max_samples"),
    ("task.policy.env_runner.n_test=0", "n_test"),
    ("dataloader.num_workers=0", "persistent_workers"),
])
def test_invalid_schema_rejected_before_construction(override, match):
    with pytest.raises(ValueError, match=match):
        launcher.compose_config("p2n_new", "libero", [override])


def test_explicit_original_crop_override_is_recorded():
    cfg = launcher.compose_config("p2n_new", "real_robot", ["policy.original_obs_config.crop_shape=[76,76]"])
    assert cfg.policy.original_obs_config.crop_shape == [76, 76]


def test_underscore_aliases_and_resume_flag(tmp_path):
    cfg = launcher.main(["--task", "real_robot", "--lazy_eval=false", "--num_train_epochs", "4",
        "--batch_size", "2", "--test_num", "9", "--resume", str(tmp_path / "missing.ckpt"), "--dry-run"])
    assert cfg.training.resume is True
    assert cfg.training.resume_checkpoint == str(tmp_path / "missing.ckpt")
    assert cfg.training.num_epochs == 4
    assert cfg.training.validation_max_samples == 9


@pytest.mark.parametrize("flags", [
    ["--num-processes", "0"], ["--lazy-eval", "perhaps"], ["--parallel-envs", "2"],
])
def test_invalid_cli_options_have_clear_errors(flags):
    with pytest.raises(SystemExit):
        launcher.main(["--task", "real_robot", "--dry-run", *flags])


@pytest.mark.parametrize("mismatched_split", [False, True])
def test_resume_preflight_uses_embedded_policy_and_never_tokenizer_source_or_fit(monkeypatch, tmp_path, mismatched_split):
    import hydra
    import torch
    cfg = launcher.compose_config("p2n_new", "real_robot", [
        "training.resume=true", "training.resume_checkpoint=/missing/external.ckpt"])
    split = {"train": {"windows": 128}, "identity": {"episode_and_action_sha256": "fake", "train_episode_ids": [0, 1], "validation_episode_ids": [2]}}
    restore_cfg = {"_target_": "embedded.Restore", "construction_mode": "restore"}
    payload = {"policy_config": restore_cfg, "state_dicts": {"model": {"embedded": 1}}, "metadata": {"dataset_split": split}}
    if mismatched_split:
        payload["metadata"] = {"dataset_split": {"identity": {"episode_and_action_sha256": "changed"}}}
    calls = []
    class Model:
        max_seq_len = 8
        def load_state_dict(self, state, strict):
            assert strict and state == {"embedded": 1}
            calls.append("strict_load")
        def get_optimizer(self, **kwargs):
            return object()
    class Workspace:
        def __init__(self, config, output_dir):
            pass
        @staticmethod
        def validate_resume_payload(saved, config):
            assert saved is payload
            calls.append("validate_resume")
        def training_report(self, model):
            return {}
    monkeypatch.setitem(sys.modules, "oat.workspace.train_p2n_new_original_obs",
                        SimpleNamespace(TrainP2NNewOriginalObsWorkspace=Workspace))
    monkeypatch.setattr(launcher, "inspect_dataset", lambda config: split)
    monkeypatch.setattr(launcher.legacy, "validate_tokenizer_source", _no_runtime)
    monkeypatch.setattr(torch, "load", lambda *a, **kw: payload)
    def instantiate(config):
        assert config is restore_cfg
        calls.append("restore_construct")
        return Model()
    monkeypatch.setattr(hydra.utils, "instantiate", instantiate)
    if mismatched_split:
        with pytest.raises(ValueError, match="fingerprint"):
            launcher.preflight(cfg, tmp_path, world_size=2)
        assert calls == ["validate_resume"]
        return
    report = launcher.preflight(cfg, tmp_path, world_size=2)
    assert calls == ["validate_resume", "restore_construct", "strict_load"]
    assert report["world_size"] == 2
    assert report["evaluation"]["physical_robot_rollout"] is False


def test_fresh_preflight_rejects_reusing_nonempty_output_before_dataset(monkeypatch, tmp_path):
    cfg = launcher.compose_config("p2n_new", "real_robot")
    (tmp_path / "existing").write_text("keep")
    monkeypatch.setattr(launcher, "inspect_dataset", _no_runtime)
    with pytest.raises(ValueError, match="Fresh output directory must be empty"):
        launcher.preflight(cfg, tmp_path)


def test_preflight_dataset_reads_raw_sample_and_fingerprints_actions(tmp_path):
    import numpy as np
    import zarr
    dataset_path = tmp_path / "tiny.zarr"
    cfg = launcher.compose_config("p2n_new", "real_robot", [
        f"task.policy.dataset.zarr_path={dataset_path}", "training.num_demo=2",
        "task.policy.dataset.val_ratio=0.5"])
    store = zarr.open(str(dataset_path), mode="w")
    store.create_dataset("meta/episode_ends", data=np.array([4, 8], dtype=np.int64))
    store.create_dataset("data/action", data=np.zeros((8, 7), dtype=np.float32))
    for key, spec in cfg.shape_meta.obs.items():
        dtype = np.uint8 if spec.type == "rgb" else np.float32
        store.create_dataset(f"data/{key}", data=np.zeros((8, *spec.shape), dtype=dtype))
    first = launcher.inspect_dataset(cfg)
    assert first["raw_training_observation_sample"]["agentview_rgb"] == {
        "shape": [2, 128, 128, 3], "dtype": "uint8"}
    assert first["train"]["episodes"] == first["validation"]["episodes"] == 1
    store["data/action"][0, 0] = 1.0
    second = launcher.inspect_dataset(cfg)
    assert first["identity"]["episode_and_action_sha256"] != second["identity"]["episode_and_action_sha256"]
    assert first["identity"]["train_episode_ids"] == second["identity"]["train_episode_ids"]
