"""Additive vision/action launch contract: schedule, gradient route and live logging."""
from pathlib import Path

from omegaconf import OmegaConf
import pytest

from scripts import train_p2n_vla as legacy
from scripts import train_p2n_vla_vision_action as launcher


@pytest.mark.parametrize("variant", ["p2n_vla", "p2n_vla_state_gate"])
def test_vision_action_config_composes_and_preserves_gate_schema(variant):
    cfg = launcher.compose_config(variant)
    assert cfg.policy._target_ == launcher.VARIANTS[variant]["target"]
    assert cfg.policy.training_scope == "vision_action"
    assert cfg.policy.vision_activation_checkpointing is True
    assert cfg.policy.lambda_ki > 0
    assert cfg.logging.mode == "online"
    assert cfg.task.policy.lazy_eval is False
    assert cfg.dataloader.batch_size == cfg.val_dataloader.batch_size == 1
    assert cfg.training.gradient_accumulate_every == 16
    assert cfg.training.max_train_steps == 1600
    assert cfg.training.max_optimizer_steps is None
    gate = variant == "p2n_vla_state_gate"
    assert cfg.task.policy.dataset._target_ == legacy.DATASET_TARGETS[gate]
    assert cfg.task.policy.env_runner._target_ == legacy.RUNNER_TARGETS[gate]
    if gate:
        assert cfg.policy.state_history_steps == 8
        assert cfg.policy.history_summary_tokens == 4
    assert cfg.policy.spm_path == str(launcher.ROOT / "data/pretrained/p2n_vla/paligemma_tokenizer.model")
    assert cfg.policy.tokenizer_checkpoint == str(launcher.ROOT / "data/pretrained/p2n_vla/oat_from_upd030000.ckpt")
    # Reusing legacy validation must not alter its process-global targets or default recipe.
    assert legacy.VARIANTS[variant]["target"] != cfg.policy._target_
    old = legacy.compose_config(variant)
    assert old.policy._target_ == legacy.VARIANTS[variant]["target"]
    assert old.dataloader.batch_size == 8


def test_requested_command_has_exact_gpu_eval_batch_and_online_schedule(capsys):
    summary = launcher.main([
        "--finetune", "vision_action", "--variant", "p2n_vla", "--task", "libero",
        "--devices", "0,1", "--num-processes", "2", "--num-train-epochs", "300",
        "--batch-size", "1", "--gradient-accumulation", "16", "--lazy-eval=false",
        "--eval-every", "100", "--test-num", "500", "--wandb-mode", "online", "--dry-run",
    ])
    assert summary["world_size"] == 2
    assert summary["logging_mode"] == "online"
    assert summary["training_scope"] == "vision_action"
    assert summary["vision_trainable"] is True
    assert summary["language_trainable"] is False
    assert summary["language_lora_trainable"] is False
    assert summary["schedule"]["effective_batch"] == 32
    assert summary["schedule"]["expected_optimizer_updates"] == 30000
    assert summary["schedule"]["updates_per_epoch"] == 100
    assert [(point["epoch"], point["optimizer_step"]) for point in summary["rollout"]["rollouts"]] == [
        (100, 10000), (200, 20000), (300, 30000)]
    assert summary["rollout"]["n_test"] == 500
    assert "Configuration validated only" in capsys.readouterr().out


def test_epoch_and_accumulation_flags_do_not_leave_a_hidden_30000_update_cap():
    args = launcher.build_parser().parse_args([
        "--num-train-epochs", "400", "--batch-size", "2", "--gradient-accumulation", "8",
        "--eval-every", "80", "--val-every", "70", "--lazy_eval=false",
        "--checkpoint-every", "50", "--snapshot-every", "1000",
    ])
    cfg = launcher.config_from_args(args)
    schedule = legacy.expected_schedule(cfg, 124600, world_size=2)
    assert cfg.training.max_train_steps == 800
    assert cfg.training.rollout_every == 80
    assert cfg.training.val_every == 70
    assert cfg.training.checkpoint_every == 50
    assert cfg.training.snapshot_every == 1000
    assert schedule["effective_batch"] == 32
    assert schedule["expected_optimizer_updates"] == schedule["cosine_horizon"] == 40000
    assert schedule["updates_per_epoch"] == 100


@pytest.mark.parametrize("override,message", [
    ("policy.lambda_ki=0", "lambda_ki > 0"),
    ("policy.training_scope=lora", "training_scope"),
    ("policy.vision_activation_checkpointing=1", "must be a boolean"),
    ("policy._target_=oat.policy.p2n_vla.P2NVLAPolicy", "must target"),
    ("variant=p2n_vla_state_gate", "must agree"),
    ("dataloader.batch_size=0", "batch_size"),
    ("training.gradient_accumulate_every=0", "gradient_accumulate_every"),
    ("task.policy.env_runner.n_test=501", "balanced"),
])
def test_invalid_finetuning_and_recipe_overrides_are_rejected(override, message):
    with pytest.raises(ValueError, match=message):
        launcher.compose_config(overrides=[override])


def test_probe_keeps_wandb_online_and_respects_explicit_hydra_overrides():
    args = launcher.build_parser().parse_args(["--probe", "--probe-steps", "2"])
    cfg = launcher.config_from_args(args)
    assert cfg.training.probe.enabled is True
    assert cfg.training.probe.optimizer_steps == 2
    assert cfg.logging.mode == "online"
    override = launcher.config_from_args(args, ["logging.mode=disabled"])
    assert override.logging.mode == "disabled"


def test_lower_memory_flags_disable_ema_and_reduce_parallel_eval_without_changing_cadence():
    args = launcher.build_parser().parse_args([
        "--devices", "0,1", "--num-processes", "2", "--lazy-eval=false", "--eval-every", "100",
        "--test-num", "500", "--use-ema=false", "--eval-parallel-envs", "5", "--wandb-mode", "online",
    ])
    cfg = launcher.config_from_args(args)
    assert cfg.training.use_ema is False
    assert cfg.task.policy.env_runner.n_parallel_envs == 5
    assert cfg.logging.mode == "online"
    plan = legacy.rollout_plan(cfg)
    assert plan["weights"] == "model"
    assert plan["n_test"] == 500
    assert plan["n_parallel_envs"] == 5
    assert plan["every_epochs"] == 100
    assert plan["expected_minutes_per_rollout"] < 0.8 * plan["timeout_minutes"]
    defaults = launcher.compose_config()
    assert defaults.training.use_ema is True
    assert defaults.task.policy.env_runner.n_parallel_envs == 10


def test_launch_passes_two_gpu_torchrun_and_online_environment(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DISABLED", "true")
    seen = {}

    def preflight(cfg, output, world_size, **kwargs):
        assert cfg.logging.mode == "online"
        assert world_size == 2
        return {"status": "mock preflight"}

    def check_gpus(devices, allow_busy):
        seen["devices"] = devices
        assert allow_busy is False

    def launch(command, **kwargs):
        seen["command"] = command
        seen["env"] = kwargs["env"]
        assert kwargs["check"] is True

    monkeypatch.setattr(legacy, "preflight", preflight)
    monkeypatch.setattr(legacy, "check_gpu_selection", check_gpus)
    monkeypatch.setattr(launcher.subprocess, "run", launch)
    output = tmp_path / "run"
    launcher.main(["--devices", "0,1", "--num-processes", "2", "--output", str(output),
                   "--wandb-mode", "online", "--eval-every", "100", "--lazy-eval=false"])
    assert seen["devices"] == ["0", "1"]
    assert seen["env"]["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert seen["env"]["WANDB_MODE"] == "online"
    assert "WANDB_DISABLED" not in seen["env"]
    assert seen["command"][1:6] == ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "2"]
    assert Path(seen["command"][6]).name == "train_p2n_vla_vision_action.py"
    resolved = OmegaConf.load(output / "p2n_vla_vision_action_resolved.yaml")
    assert resolved.policy.training_scope == "vision_action"
    assert resolved.logging.mode == "online"
    assert resolved.training.rollout_every == 100
    capsys.readouterr()


@pytest.mark.parametrize("flags", [
    ["--devices", "0,0"], ["--devices", "0,1", "--num-processes", "1"],
    ["--num-processes", "0"], ["--probe", "--resume", "latest.ckpt"],
    ["--probe-steps", "0"], ["--finetune", "language"],
])
def test_invalid_launch_flags_fail_before_preflight(flags):
    with pytest.raises(SystemExit):
        launcher.main([*flags, "--dry-run"])
