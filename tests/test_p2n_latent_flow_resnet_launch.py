"""CPU contracts for the original-ResNet observation recipes and launcher."""
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("p2n_latent_flow", "p2n_state_gate_latent_flow")


def _compose(variant, task, *, resnet, overrides=()):
    name = f"train_{variant}" + ("_resnet18" if resnet else "")
    if task == "real_robot":
        name = f"experimental/{name}_real_robot"
    with initialize_config_dir(config_dir=str(ROOT / "oat/config"), version_base=None):
        return compose(config_name=name, overrides=list(overrides))


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("task", ("libero", "real_robot"))
def test_resnet_recipes_retain_latent_flow_and_data_contract(variant, task):
    cfg = _compose(variant, task, resnet=True,
                   overrides=("dataloader.batch_size=8", "training.gradient_accumulate_every=4"))
    old = _compose(variant, task, resnet=False)
    assert cfg.policy.obs_encoder_type == "resnet18"
    assert cfg.policy.dino_path is None and cfg.policy.dino_revision is None
    assert cfg.policy.image_brightness == cfg.policy.image_contrast == 0.0
    assert cfg.policy.num_visual_queries is None and cfg.policy.resampler_depth is None
    assert OmegaConf.to_container(cfg.policy.resnet_config) == {
        "crop_shape": [112, 112], "use_group_norm": True,
        "share_rgb_model": False, "eval_fixed_crop": True,
    }
    assert cfg.policy_family == old.policy_family == "oat_latent_flow"
    assert cfg.variant == old.variant == variant
    assert cfg.task_type == old.task_type == task
    assert OmegaConf.to_container(cfg.policy.flow, resolve=True) == OmegaConf.to_container(old.policy.flow, resolve=True)
    assert OmegaConf.to_container(cfg.task.policy.dataset, resolve=True) == OmegaConf.to_container(old.task.policy.dataset, resolve=True)
    assert cfg.policy.tokenizer_checkpoint == old.policy.tokenizer_checkpoint
    assert cfg.dataloader.batch_size == 8 and cfg.training.gradient_accumulate_every == 4
    assert cfg.training.use_ema and cfg.dataloader.drop_last
    assert "resnet18" in cfg.logging.group and "resnet18" in cfg.logging.tags
    if task == "real_robot":
        assert cfg.training.num_epochs == 2001 and cfg.task.policy.env_runner is None
        assert cfg.logging.project == "real_robot_p2n_latent_flow"
    if variant == "p2n_state_gate_latent_flow":
        assert cfg.policy.state_history_steps == 8 and cfg.policy.history_summary_tokens == 4
    else:
        assert not any(key.startswith("history_") for key in cfg.policy)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("task", ("libero", "real_robot"))
def test_launcher_selects_resnet_and_preserves_legacy_positional_api(variant, task):
    from scripts import train_p2n_latent_flow as launch
    legacy = launch.compose_config(variant, task, ["dataloader.batch_size=8"])
    cfg = launch.compose_config(variant, task, ["dataloader.batch_size=8"], obs_encoder="resnet18")
    assert launch.observation_encoder(legacy) == "dinov3"
    assert launch.observation_encoder(cfg) == "resnet18"
    assert legacy.dataloader.batch_size == cfg.dataloader.batch_size == 8


@pytest.mark.parametrize("override", [
    "policy.dino_path=/does/not/exist", "policy.dino_revision=abcdef",
    "+policy.dino_config={model_type:dinov3}", "+policy.processor_config={image_mean:[0.5,0.5,0.5]}",
    "policy.obs_encoder_type=dinov3",
])
def test_resnet_rejects_mixed_encoder_configuration(override):
    from scripts import train_p2n_latent_flow as launch
    with pytest.raises(ValueError, match="DINO|observation encoder"):
        launch.compose_config("p2n_latent_flow", "real_robot", [override], obs_encoder="resnet18")


def test_preflight_requires_oat_but_never_dino_for_resnet(monkeypatch, tmp_path):
    from scripts import train_p2n_latent_flow as launch
    cfg = launch.compose_config("p2n_latent_flow", "real_robot", obs_encoder="resnet18")
    calls = []
    def forbidden(*args, **kwargs):
        raise AssertionError("ResNet CPU preflight must not inspect DINO or GPUs")
    monkeypatch.setattr(launch, "validate_dino_source", forbidden)
    monkeypatch.setattr(launch, "check_gpu_selection", forbidden)
    monkeypatch.setattr(launch, "validate_tokenizer_source", lambda cfg: calls.append("oat") or {"weights": "ema_model"})
    monkeypatch.setattr(launch, "inspect_dataset", lambda cfg: {"train": {"windows": 64}})
    out = tmp_path / "uncreated"
    report = launch.preflight(cfg, out, 2)
    assert calls == ["oat"]
    assert report["obs_encoder"] == "resnet18" and "dino" not in report["sources"]
    assert report["sources"]["resnet18"]["pretrained"] is False
    assert report["sources"]["resnet18"]["external_vision_checkpoint_required"] is False
    assert report["effective_batch"] == 64
    assert not out.exists()


@pytest.mark.parametrize("encoder", ("dinov3", "resnet18"))
def test_resume_infers_encoder_and_rejects_cross_encoder(encoder):
    from scripts import train_p2n_latent_flow as launch
    cfg = launch.compose_config("p2n_latent_flow", "real_robot", obs_encoder=encoder)
    overrides = ["training.resume=true", "training.resume_checkpoint=/tmp/saved.ckpt"]
    resumed = launch.compose_resume_config({"cfg": cfg}, "p2n_latent_flow", "real_robot", overrides)
    assert launch.observation_encoder(resumed) == encoder
    other = "resnet18" if encoder == "dinov3" else "dinov3"
    with pytest.raises(ValueError, match="Resume observation encoder"):
        launch.compose_resume_config({"cfg": cfg}, "p2n_latent_flow", "real_robot", overrides, obs_encoder=other)
    with pytest.raises(ValueError, match="Selected observation encoder"):
        launch.compose_resume_config({"cfg": cfg}, "p2n_latent_flow", "real_robot",
                                     [*overrides, f"++policy.obs_encoder_type={other}"])


def _bash_capture(tmp_path, *args):
    import json
    import os
    import subprocess
    import sys
    fake = tmp_path / "fake_python"
    log = tmp_path / "calls.jsonl"
    fake.write_text(f"#!{sys.executable}\nimport json, os, sys\n"
                    "with open(os.environ['FLOW_TEST_CALLS'], 'a') as stream:\n"
                    "    stream.write(json.dumps({'args': sys.argv[1:], 'wandb': os.environ.get('WANDB_MODE')}) + '\\n')\n")
    fake.chmod(0o755)
    env = {key: value for key, value in os.environ.items() if not key.startswith("FLOW_")}
    env["FLOW_TEST_CALLS"] = str(log)
    result = subprocess.run(["bash", str(ROOT / "train_p2n_latent_flow.sh"), "--python", str(fake), *args],
                            env=env, text=True, capture_output=True)
    calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return result, calls


def test_single_bash_resnet_routes_both_variants_and_batch_flags(tmp_path):
    result, calls = _bash_capture(tmp_path, "--obs-encoder", "resnet18", "--gpus", "4,5", "--batch-size", "8",
                                 "--val-batch-size", "2", "--grad-accum", "4", "--dry-run")
    assert result.returncode == 0, result.stderr
    assert len(calls) == 2
    for call, variant in zip(calls, VARIANTS):
        argv = call["args"]
        assert argv[argv.index("--variant") + 1] == variant
        assert argv[argv.index("--obs-encoder") + 1] == "resnet18"
        assert argv[argv.index("--gpus") + 1] == "4,5"
        assert "--dino" not in argv and "--dino-revision" not in argv
        assert "resnet18" in argv[argv.index("--output") + 1]
        assert {"dataloader.batch_size=8", "val_dataloader.batch_size=2", "training.gradient_accumulate_every=4", "logging.mode=online"} <= set(argv)
        assert call["wandb"] == "online"


def test_single_bash_default_dino_and_resume_encoder_inference(tmp_path):
    result, calls = _bash_capture(tmp_path, "--variant", VARIANTS[0], "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "--dino" in calls[0]["args"]
    assert calls[0]["args"][calls[0]["args"].index("--obs-encoder") + 1] == "dinov3"
    result, calls = _bash_capture(tmp_path, "--variant", VARIANTS[0], "--resume", "/tmp/saved.ckpt", "--dry-run")
    assert result.returncode == 0, result.stderr
    argv = calls[-1]["args"]
    assert "--resume" in argv and "--obs-encoder" not in argv and "--dino" not in argv


@pytest.mark.parametrize("extra", [("--dino", "/tmp/source"), ("--dino-revision", "abcdef")])
def test_single_bash_rejects_dino_flags_for_resnet(tmp_path, extra):
    result, calls = _bash_capture(tmp_path, "--obs-encoder=resnet18", *extra, "--dry-run")
    assert result.returncode == 2 and "does not accept" in result.stderr
    assert not calls


def test_python_dry_run_selects_resnet_without_launching(monkeypatch, tmp_path):
    from scripts import train_p2n_latent_flow as launch
    def forbidden(*args, **kwargs):
        raise AssertionError("Dry-run may not launch a worker or probe GPU")
    monkeypatch.setattr(launch, "check_gpu_selection", forbidden)
    monkeypatch.setattr(launch.subprocess, "run", forbidden)
    monkeypatch.setattr(launch, "preflight", lambda cfg, output, world_size: {
        "encoder": launch.observation_encoder(cfg), "world_size": world_size, "output": str(output)})
    out = tmp_path / "uncreated"
    report = launch.main(["--variant", VARIANTS[0], "--task", "real_robot", "--gpus", "2,3",
                          "--obs-encoder", "resnet18", "--output", str(out), "--dry-run"])
    assert report["encoder"] == "resnet18" and report["world_size"] == 2
    assert not out.exists()
