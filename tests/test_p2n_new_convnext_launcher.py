"""Launcher acceptance without opening a dataset or pretrained checkpoint."""
from types import SimpleNamespace

import pytest

from scripts import train_p2n_new_convnext as launcher


@pytest.mark.parametrize("variant", ["p2n_new", "p2n_state_gate_new"])
def test_additive_configs_preserve_respective_real_robot_contract(variant):
    original = launcher.legacy.compose_config(variant, "real_robot")
    cfg = launcher.compose_config(variant, "real_robot")
    assert cfg.variant == original.variant
    assert cfg.training == original.training
    assert cfg.task == original.task
    assert cfg.optimizer == original.optimizer
    assert cfg.dataloader == original.dataloader
    assert cfg.val_dataloader == original.val_dataloader
    assert cfg.training.num_demo == 77
    assert cfg.training.num_epochs == 2001
    assert cfg.seed == 42
    assert cfg.task.policy.dataset.val_ratio == .05
    assert cfg.task.policy.env_runner is None
    assert cfg.policy.tokenizer_checkpoint == original.policy.tokenizer_checkpoint
    for key in ("embed_dim", "n_layers", "n_heads", "ffn_dim", "dropout", "self_past_p",
                "self_past_warmup_steps", "self_past_ramp_steps", "self_past_chunk_size",
                "num_visual_queries", "resampler_depth", "horizon", "n_action_steps", "past_n"):
        assert cfg.policy[key] == original.policy[key]
    assert cfg.policy.expected_action_tokens == 8
    assert cfg.policy.visual_resampler_dim == 256
    assert cfg.policy.visual_resampler_heads == 4
    assert cfg.policy.visual_resampler_ffn_dim == 768
    assert "dinov3" not in str(cfg.logging.tags)


@pytest.mark.parametrize("override,field", [
    ("policy.embed_dim=256", "embed_dim"),
    ("policy.n_layers=8", "n_layers"),
    ("policy.n_heads=4", "n_heads"),
    ("policy.ffn_dim=768", "ffn_dim"),
    ("policy.expected_action_tokens=16", "expected_action_tokens"),
    ("policy.num_visual_queries=16", "num_visual_queries"),
    ("policy.vision_feature_stages=[1,3]", "vision_feature_stages"),
    ("policy.convnext_frozen=false", "convnext_frozen"),
    ("policy.visual_resampler_dim=768", "visual_resampler_dim"),
    ("policy.dino_path=/local/dino", "DINO"),
])
def test_invalid_backend_contract_rejected_before_loading(override, field):
    with pytest.raises(ValueError, match=field):
        launcher.compose_config("p2n_new", "real_robot", [override])


def test_dry_run_never_preflights_or_checks_gpus(monkeypatch, capsys):
    def unexpected(*args, **kwargs):
        pytest.fail("Dry-run attempted model/data/GPU preflight")
    monkeypatch.setattr(launcher, "preflight", unexpected)
    monkeypatch.setattr(launcher, "check_gpu_idle", unexpected)
    launcher.main(["--vision", "convnext_nano", "--variant", "p2n_new", "--task", "real_robot",
                   "--convnext", "/nonexistent/local/weights", "--convnext-revision", "unused-for-dry-run",
                   "--dry-run", "--", "dataloader.batch_size=3", "training.gradient_accumulate_every=2"])
    output = capsys.readouterr().out
    assert '"effective_batch": 12' in output
    assert "p2n_new_convnext_nano_real_robot_seed42" in output
    assert "Configuration resolved only" in output


@pytest.mark.parametrize("prefix", [[], ["--vision", "dinov3"]])
def test_default_dino_preserves_legacy_cli(monkeypatch, prefix):
    calls = []
    monkeypatch.setattr(launcher.legacy, "main", lambda argv: calls.append(argv))
    arguments = ["--variant", "p2n_new", "--task", "libero", "--dino", "/local/dino", "--dry-run", "--", "seed=71"]
    launcher.main([*prefix, *arguments])
    assert calls == [arguments]


def test_nano_libero_is_not_silently_remapped():
    with pytest.raises(ValueError, match="real_robot only"):
        launcher.compose_config("p2n_new", "libero")


def test_cli_conflicting_weight_sources_rejected():
    with pytest.raises(SystemExit):
        launcher.main(["--vision", "convnext_nano", "--variant", "p2n_new", "--task", "real_robot",
                       "--convnext", "/nano", "--dino", "/dino", "--dry-run"])


def test_gpu_idle_check_only_inspects_selected_devices(monkeypatch):
    def run(command, **kwargs):
        if "--query-gpu=index,uuid,name" in command:
            return SimpleNamespace(stdout="0, GPU-a, First\n1, GPU-b, Second\n")
        return SimpleNamespace(stdout="GPU-b, 987, unrelated_training\n")
    monkeypatch.setattr(launcher.subprocess, "run", run)
    assert launcher.check_gpu_idle("0", 1)[0]["uuid"] == "GPU-a"
    with pytest.raises(RuntimeError, match="987"):
        launcher.check_gpu_idle("1", 1)
    with pytest.raises(ValueError, match="only 1 GPUs"):
        launcher.check_gpu_idle("0", 2)
    with pytest.raises(ValueError, match="duplicate"):
        launcher.check_gpu_idle("0,GPU-a", 1)
