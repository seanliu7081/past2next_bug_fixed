"""Native VLM v1 integration: recipe, explicit unsupported knobs, and launch batching."""
from pathlib import Path
import subprocess

import pytest

import scripts.train_p2n_vla as launcher


VARIANT = "pi05_p2n_vlm_v1"
REPO = Path(__file__).resolve().parents[1]


def compose(*overrides):
    return launcher.compose_config(VARIANT, overrides=list(overrides))


def test_native_recipe_matches_expert_ar_training_conditions():
    cfg = compose()
    baseline = launcher.compose_config("pi05_p2n")
    assert cfg.policy._target_ == "oat.policy.pi05_p2n_vlm_v1.Pi05P2NVLMV1Policy"
    assert cfg.training == baseline.training
    assert cfg.optimizer == baseline.optimizer
    assert cfg.dataloader == baseline.dataloader
    assert cfg.val_dataloader == baseline.val_dataloader
    assert cfg.task == baseline.task
    for key in ("train_image_aug", "lora_rank", "lora_alpha", "train_lora", "train_vision",
                "vision_activation_checkpointing", "cond_dropout", "use_past", "temperature", "topk",
                "self_past_p", "self_past_warmup_steps", "self_past_ramp_steps", "self_past_chunk_size",
                "self_past_temperature", "self_past_topk", "self_past_schedule", "tokenizer_checkpoint"):
        assert cfg.policy[key] == baseline.policy[key], key
    assert all(cfg.policy[key] is None for key in ("lambda_ki", "ki_skip", "adarms_t0"))
    assert "cond_t0" not in cfg.policy and "kv_grad_scale" not in cfg.policy


@pytest.mark.parametrize("override,match", [
    ("policy.lambda_ki=1", "lambda_ki"),
    ("policy.ki_skip=1152", "ki_skip"),
    ("policy.adarms_t0=0.6", "adarms_t0"),
    ("+policy.cond_t0=0.6", "cond_t0"),
    ("+policy.kv_grad_scale=1", "kv_grad_scale"),
    ("+policy.flow_num_steps=10", "flow_num_steps"),
    ("policy.temperature=-1", "temperature"),
    ("policy.self_past_topk=0", "self_past_topk"),
    ("policy.cond_dropout=1", "cond_dropout"),
    ("policy.train_vision=1", "train_vision"),
    ("policy.use_past=false", "no generated history"),
])
def test_native_validation_rejects_expert_settings_and_invalid_values(override, match):
    with pytest.raises(ValueError, match=match):
        compose(override)


def test_native_no_past_and_inherited_self_past_sampler():
    compose("policy.use_past=false", "training.validate_generated_history=false")
    compose("policy.self_past_temperature=null", "policy.self_past_topk=null")


@pytest.mark.parametrize("devices,accumulation", [("0", 8), ("0,1", 4), ("0,1,2,3", 2)])
def test_native_launcher_preserves_effective_batch(devices, accumulation, capsys, tmp_path):
    result = launcher.main(["--variant", VARIANT, "--devices", devices, "--dry-run", "--output", str(tmp_path)])
    capsys.readouterr()
    assert result["schedule"]["effective_batch"] == 32
    assert result["schedule"]["batches_per_rank"] == 100 * accumulation
    assert result["schedule"]["updates_per_epoch"] == 100
    assert not list(tmp_path.iterdir())


def test_explicit_accumulation_override_wins(capsys, tmp_path):
    result = launcher.main(["--variant", VARIANT, "--devices", "0,1,2", "--dry-run", "--output", str(tmp_path),
                            "--", "training.gradient_accumulate_every=2"])
    capsys.readouterr()
    assert result["schedule"]["effective_batch"] == 24


def test_native_shell_wrapper_routes_variant_without_outputs(tmp_path):
    script = REPO / "train_pi05_p2n_vlm_v1.sh"
    result = subprocess.run(["bash", str(script), "--devices", "0,1,2,3", "--dry-run", "--output", str(tmp_path)],
                            cwd=REPO, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-2000:]
    assert f'"variant": "{VARIANT}"' in result.stdout
    assert '"effective_batch": 32' in result.stdout
    assert not list(tmp_path.iterdir())
    refused = subprocess.run(["bash", str(script), "--variant", "pi05_past"],
                             cwd=REPO, capture_output=True, text=True, timeout=30)
    assert refused.returncode == 2 and "fixes --variant" in refused.stderr
