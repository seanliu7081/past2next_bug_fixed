"""CPU launch contracts for the pen_cabinet latent-flow task selection."""
from pathlib import Path
import json
import os
import subprocess
import sys

import pytest
from omegaconf import OmegaConf

from scripts import train_p2n_latent_flow as launch

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("p2n_latent_flow", "p2n_state_gate_latent_flow")
ENCODERS = ("dinov3", "resnet18")
PEN_DATA = "/workspace/ysk/zarr/pen_cabinet_N67.zarr"


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("encoder", ENCODERS)
def test_pen_selection_has_matched_sources_split_and_flow_contract(variant, encoder):
    cfg = launch.compose_config(variant, "pen_cabinet", [
        "dataloader.batch_size=8", "val_dataloader.batch_size=2",
        "training.gradient_accumulate_every=4",
    ], obs_encoder=encoder)
    assert cfg.task_type == cfg.policy.task == "real_robot"
    assert cfg.task.policy.task_name == "pen_cabinet_N67"
    assert cfg.task.policy.name == "real_robot_pen_cabinet_N67"
    assert cfg.policy_family == "oat_latent_flow" and cfg.variant == variant
    assert launch.observation_encoder(cfg) == encoder
    assert cfg.task.policy.lazy_eval and cfg.task.policy.env_runner is None
    ds = cfg.task.policy.dataset
    assert ds.zarr_path == PEN_DATA
    assert ds.val_ratio == 0.1 and ds.seed == 42
    assert cfg.training.num_demo == 67
    assert "pen_cabinet" in cfg.policy.tokenizer_checkpoint
    assert cfg.training.num_epochs == 2001
    assert cfg.dataloader.batch_size == 8 and cfg.val_dataloader.batch_size == 2
    assert cfg.training.gradient_accumulate_every == 4
    # Prevent inherited nut-washer data, OAT, validation labels or run names.
    assert "nut_washer" not in OmegaConf.to_yaml(cfg, resolve=True)
    assert ds.history_padding == "zero" and ds.return_history_validity
    assert ds.n_obs_steps == 2 and ds.n_action_steps == 16 and ds.past_n == 7
    if variant == "p2n_state_gate_latent_flow":
        assert ds._target_ == "oat.dataset.latent_flow_dataset.LatentFlowRealRobotZarrDatasetWithStateHistory"
        assert ds.state_history_steps == cfg.policy.state_history_steps == 8
        assert "robot0_eef_rot6d" in ds.state_history_keys
        assert cfg.policy.rotation_6d_layout == "rows"
    else:
        assert ds._target_ == "oat.dataset.latent_flow_dataset.LatentFlowRealRobotZarrDatasetWithPrevWindow"
        assert not any(key.startswith(("history_", "state_history")) for key in cfg.policy)
    if encoder == "resnet18":
        assert cfg.policy.dino_path is None and cfg.policy.dino_revision is None
        assert list(cfg.policy.resnet_config.crop_shape) == [112, 112]


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("encoder", ENCODERS)
def test_pen_resume_preserves_task_sources_and_encoder(variant, encoder):
    cfg = launch.compose_config(variant, "pen_cabinet", [
        "dataloader.batch_size=8", "training.gradient_accumulate_every=4",
        "training.num_epochs=37",
    ], obs_encoder=encoder)
    resumed = launch.compose_resume_config({"cfg": cfg}, variant, "pen_cabinet", [
        "training.resume=true", "training.resume_checkpoint=/tmp/pen_saved.ckpt",
    ])
    assert resumed.task_type == resumed.policy.task == "real_robot"
    assert resumed.task.policy.dataset.zarr_path == PEN_DATA
    assert resumed.policy.tokenizer_checkpoint == cfg.policy.tokenizer_checkpoint
    assert launch.observation_encoder(resumed) == encoder
    assert resumed.dataloader.batch_size == 8
    assert resumed.training.gradient_accumulate_every == 4
    assert resumed.training.num_epochs == 37 and resumed.training.resume


@pytest.mark.parametrize("encoder", ENCODERS)
def test_pen_command_rejects_nut_resume_despite_shared_robot_family(encoder):
    cfg = launch.compose_config("p2n_latent_flow", "real_robot", obs_encoder=encoder)
    with pytest.raises(ValueError, match="[Tt]ask|pen_cabinet"):
        launch.compose_resume_config({"cfg": cfg}, "p2n_latent_flow", "pen_cabinet")


def test_single_bash_plain_pen_resnet_forwards_task_batch_and_online_wandb(tmp_path):
    # Capture arguments rather than importing a policy or starting a W&B run.
    fake = tmp_path / "fake_python"
    log = tmp_path / "call.json"
    fake.write_text(f"#!{sys.executable}\nimport json, os, sys\n"
                    "with open(os.environ['FLOW_TEST_CALL'], 'w') as stream:\n"
                    "    json.dump({'args': sys.argv[1:], 'wandb': os.environ.get('WANDB_MODE')}, stream)\n")
    fake.chmod(0o755)
    env = {key: value for key, value in os.environ.items() if not key.startswith("FLOW_")}
    env["FLOW_TEST_CALL"] = str(log)
    result = subprocess.run([
        "bash", str(ROOT / "train_p2n_latent_flow.sh"), "--python", str(fake),
        "--variant", "p2n_latent_flow", "--task", "pen_cabinet",
        "--obs-encoder", "resnet18", "--gpus", "2,3", "--batch-size", "8",
        "--val-batch-size", "2", "--grad-accum", "4", "--dry-run",
    ], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    call = json.loads(log.read_text())
    argv = call["args"]
    assert argv[argv.index("--variant") + 1] == "p2n_latent_flow"
    assert argv[argv.index("--task") + 1] == "pen_cabinet"
    assert argv[argv.index("--obs-encoder") + 1] == "resnet18"
    assert argv[argv.index("--gpus") + 1] == "2,3"
    assert "--dino" not in argv and "--dino-revision" not in argv
    assert "pen_cabinet" in argv[argv.index("--output") + 1]
    assert {"dataloader.batch_size=8", "val_dataloader.batch_size=2",
            "training.gradient_accumulate_every=4", "logging.mode=online", "--dry-run"} <= set(argv)
    assert call["wandb"] == "online"


def test_python_pen_dry_run_routes_without_gpu_worker_or_wandb(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("Dry-run must not launch a worker or inspect GPUs")
    monkeypatch.setattr(launch, "check_gpu_selection", forbidden)
    monkeypatch.setattr(launch.subprocess, "run", forbidden)
    monkeypatch.setattr(launch, "preflight", lambda cfg, output, world_size: {
        "task": cfg.task.policy.task_name, "encoder": launch.observation_encoder(cfg),
        "data": cfg.task.policy.dataset.zarr_path, "world_size": world_size,
    })
    out = tmp_path / "uncreated"
    result = launch.main([
        "--variant", "p2n_latent_flow", "--task", "pen_cabinet",
        "--obs-encoder", "resnet18", "--gpus", "2,3",
        "--output", str(out), "--dry-run",
    ])
    assert result == {"task": "pen_cabinet_N67", "encoder": "resnet18",
                      "data": PEN_DATA, "world_size": 2}
    assert not out.exists()
