"""P2N-VLA launcher (M4): config composition and validation, preflight checks, worker entry, eval script."""
import json
import os
import subprocess
from pathlib import Path

from omegaconf import OmegaConf
import pytest
import torch

import scripts.evaluate_p2n_vla as evaluate
import scripts.train_p2n_vla as launcher
from test_p2n_vla_workspace import StubGatePolicy, StubVLAPolicy, cpu_accelerate, run_workspace, stub_config  # noqa: F401

REPO = Path(__file__).resolve().parents[1]
ASSETS = json.loads((REPO / "data/pretrained/p2n_vla/assets.json").read_text())
LIBERO_ZARR = Path("/workspace/past_action/data/libero/libero10_N500.zarr")
OAT = Path("/workspace/hf_upload/tokenizer_oattok_so3aug_ep4960_mse0.001.ckpt")
VARIANTS = ("p2n_vla", "p2n_vla_state_gate", "pi05_ki_flow")


def compose(variant, *overrides):
    return launcher.compose_config(variant, "libero", list(overrides))


@pytest.mark.parametrize("variant", VARIANTS)
def test_all_three_configs_compose_with_the_full_recipe(variant):
    cfg = compose(variant)
    policy, training = cfg.policy, cfg.training
    assert cfg.variant == policy.variant == variant
    assert policy._target_ == launcher.VARIANTS[variant]["target"]
    assert cfg._target_ == "oat.workspace.train_p2n_vla.TrainP2NVLAWorkspace"
    # batch and length: 300 epochs x 200 micro-batches / accumulation 2 = 30000 updates at batch 8 x 2 x 2
    # (an epoch is 100 updates; in-training rollouts every 50 epochs land on the 5k-update snapshots)
    assert cfg.dataloader.batch_size == 8 and training.gradient_accumulate_every == 2
    assert training.num_epochs == 300 and training.max_train_steps == 200 and training.max_optimizer_steps == 30000
    assert cfg.optimizer == {"policy_lr": 5e-5, "new_module_lr": 1e-4, "weight_decay": 1e-10,
                             "betas": [0.9, 0.95], "eps": 1e-8, "fused": None}
    assert training.lr_warmup_steps == 1000 and training.min_lr_ratio == 0.1 and training.max_grad_norm == 1.0
    assert cfg.ema.decay == 0.999 and training.use_ema is True
    assert training.snapshot_every == 5000 and training.checkpoint_every == 10
    assert training.val_every == 50 and training.max_val_steps == 200
    assert training.rollout_every == 50 and training.rollout_at_final is True and training.rollout_seed == 44
    assert cfg.task.policy.lazy_eval is True  # rollouts are opt-in: task.policy.lazy_eval=false
    runner = cfg.task.policy.env_runner
    assert (runner.protocol, runner.n_test, runner.test_start_seed, runner.init_state_offset) == ("official", 500, 3000, 0)
    assert runner.n_parallel_envs == 10 and runner.max_episode_steps == 550 and runner.n_test_vis == 0
    assert (policy.lora_rank, policy.lora_alpha, policy.adarms_t0, policy.lambda_ki) == (16, 16.0, 0.6, 1.0)
    assert policy.temperature == 0.0 and policy.model_size == "full"
    assert policy.prompt.max_len == 96 and cfg.max_prompt_len == 96
    assert policy.prompt.instruction_source == "libero10"
    if variant == "pi05_ki_flow":
        assert policy.use_past is False and policy.self_past_p == 0.0 and policy.flow_num_steps == 10
        assert training.validate_generated_history is False
    else:
        assert policy.use_past is True
        assert (policy.self_past_p, policy.self_past_warmup_steps, policy.self_past_ramp_steps) == (0.5, 1000, 4000)
        assert (policy.self_past_chunk_size, policy.self_past_temperature, policy.self_past_topk) == (4, 1.0, 10)
    # assets pinned to the M0 manifest and the LIBERO SO(3)-aug OAT
    assert policy.pi05_sha256 == ASSETS["files"]["pi05_base/model.safetensors"]["sha256"]
    assert policy.spm_path == ASSETS["files"]["paligemma_tokenizer.model"]["path"]
    assert policy.spm_sha256 == ASSETS["files"]["paligemma_tokenizer.model"]["sha256"]
    assert ASSETS["pi05_revision"] in policy.pi05_weights
    assert policy.tokenizer_checkpoint == str(OAT)
    assert cfg.task.policy.dataset.zarr_path == str(LIBERO_ZARR)
    assert cfg.logging.mode == "offline"
    assert compose(variant, "logging.mode=online").logging.mode == "online"
    gate = variant == "p2n_vla_state_gate"
    assert cfg.task.policy.dataset._target_ == launcher.DATASET_TARGETS[gate]
    assert cfg.task.policy.env_runner._target_ == launcher.RUNNER_TARGETS[gate]
    if gate:
        assert policy.state_history_steps == 8 and policy.history_summary_tokens == 4
        assert list(policy.state_history_keys) == list(cfg.task.policy.dataset.state_history_keys)
        assert policy.history_gate_mode == "learned" and policy.history_gate_init == 0.9
    schedule = launcher.expected_schedule(cfg, 124600, 2)
    assert schedule["batches_per_rank"] == 200 and schedule["updates_per_epoch"] == 100
    assert schedule["planned_optimizer_updates"] == schedule["expected_optimizer_updates"] == 30000
    assert schedule["effective_batch"] == 32 and schedule["cosine_horizon"] == 30000


@pytest.mark.parametrize("variant,override,message", [
    ("p2n_vla_state_gate", "policy._target_=oat.policy.p2n_vla.P2NVLAPolicy", "must agree"),
    ("p2n_vla", "variant=p2n_vla_state_gate", "must agree"),
    ("p2n_vla", "policy.variant=pi05_ki_flow", "must agree"),
    ("pi05_ki_flow", "policy._target_=oat.policy.p2n_vla.P2NVLAPolicy", "must agree"),
    ("p2n_vla", "+policy.history_dropout=0.1", "history/gate"),
    ("p2n_vla", "task.policy.dataset._target_=oat.dataset.vla_dataset.VLAZarrDatasetWithStateHistory", "dataset"),
    ("p2n_vla_state_gate", "task.policy.env_runner._target_=oat.env_runner.p2n_new_runner.P2NNewLiberoRunner",
     "evaluated by"),
    ("p2n_vla_state_gate", "policy.state_history_steps=7", "past_n \\+ 1"),
    ("pi05_ki_flow", "policy.use_past=true", "use_past"),
    ("p2n_vla", "task.policy.dataset.zarr_path=data/libero/libero10_N500.zarr", "absolute"),
    ("p2n_vla", "n_obs_steps=2", "n_obs_steps"),
    ("p2n_vla", "task.policy.dataset.prompt_state={keys:[robot0_eef_pos],transforms:{}}", "prompt"),
    ("p2n_vla", "training.lr_warmup_steps=40000", "lr_warmup_steps"),
    ("p2n_vla", "dataloader.drop_last=false", "drop_last"),
    ("p2n_vla", "logging.mode=verbose", "logging.mode"),
    ("p2n_vla", "task.policy.lazy_eval=1", "lazy_eval"),
    ("p2n_vla", "training.init_checkpoint=/x.ckpt", "init_checkpoint"),
    ("p2n_vla", "training.resume=true", "resume"),
    ("p2n_vla", "policy.model_size=tiny", "pi05_weights=null"),
    ("p2n_vla", "policy.construction_mode=restore", "fresh"),
    ("p2n_vla", "+optimizer.momentum=0.9", "optimizer"),
    ("p2n_vla", "ema.decay=1.0", "ema.decay"),
])
def test_config_validation_rejects_disagreements(variant, override, message):
    with pytest.raises(ValueError, match=message):
        compose(variant, override)


def test_in_training_rollouts_validate_and_plan_the_official_schedule():
    for variant in ("p2n_vla", "p2n_vla_state_gate"):
        cfg = compose(variant, "task.policy.lazy_eval=false", "logging.mode=online")
        plan = launcher.rollout_plan(cfg)
        assert plan["enabled"] and plan["protocol"] == "official" and plan["n_test"] == 500
        assert plan["episodes_per_task"] == 50 and plan["episode_seeds"] == [3000, 3499]
        assert [(point["epoch"], point["optimizer_step"]) for point in plan["rollouts"]] == [
            (50, 5000), (100, 10000), (150, 15000), (200, 20000), (250, 25000), (300, 30000)]
        assert launcher.describe(cfg, 2, "out", "dry_run")["rollout"] == plan
    assert launcher.rollout_plan(compose("p2n_vla")) == {"enabled": False}
    # a final rollout is appended when num_epochs is not a multiple of rollout_every
    short = compose("p2n_vla", "task.policy.lazy_eval=false", "training.num_epochs=120")
    assert [point["epoch"] for point in launcher.rollout_plan(short)["rollouts"]] == [50, 100, 120]
    for overrides, message in (
            (("training.rollout_every=0",), "rollout_every"),
            (("task.policy.env_runner.n_test=505",), "balanced"),
            (("task.policy.env_runner.init_state_offset=10",), "initial states per task"),
            (("task.policy.env_runner.n_test_vis=501",), "n_test_vis"),
            (("task.policy.env_runner.protocol=corrected", "task.policy.env_runner.init_state_offset=1"),
             "official protocol only"),
            (("training.rollout_at_final=1",), "rollout_at_final")):
        with pytest.raises(ValueError, match=message):
            compose("p2n_vla", "task.policy.lazy_eval=false", *overrides)
    # corrected-protocol rollouts need no balance (and 40 per task + offset 10 still fits the 50 official states)
    compose("p2n_vla", "task.policy.lazy_eval=false", "task.policy.env_runner.protocol=corrected",
            "task.policy.env_runner.n_test=105")
    compose("p2n_vla", "task.policy.lazy_eval=false", "task.policy.env_runner.n_test=400",
            "task.policy.env_runner.init_state_offset=10")


def test_preflight_checks_wandb_credentials_and_egl_rendering(monkeypatch, tmp_path):
    online = compose("p2n_vla", "logging.mode=online")
    monkeypatch.setenv("WANDB_API_KEY", "x" * 40)
    assert launcher.check_wandb(online)["credentials"] == "WANDB_API_KEY"
    monkeypatch.delenv("WANDB_API_KEY")
    monkeypatch.setenv("HOME", str(tmp_path))  # no ~/.netrc
    monkeypatch.delenv("NETRC", raising=False)
    with pytest.raises(ValueError, match="wandb login"):
        launcher.check_wandb(online)
    (tmp_path / ".netrc").write_text("machine api.wandb.ai\n  login user\n  password " + "y" * 40 + "\n")
    (tmp_path / ".netrc").chmod(0o644)  # wandb accepts it; the stdlib's default-file permission check must not apply
    assert launcher.check_wandb(online)["credentials"] == "wandb netrc lookup"
    # $NETRC and WANDB_BASE_URL are honoured the way wandb itself resolves them
    other = tmp_path / "other.netrc"
    other.write_text("machine wandb.example.org\n  login user\n  password " + "z" * 40 + "\n")
    monkeypatch.setenv("NETRC", str(other))
    with pytest.raises(ValueError, match="api.wandb.ai"):
        launcher.check_wandb(online)
    monkeypatch.setenv("WANDB_BASE_URL", "https://wandb.example.org")
    assert launcher.check_wandb(online)["host"] == "https://wandb.example.org"
    # wandb names netrc machines by host:port and expands ~ in $NETRC
    (tmp_path / "port.netrc").write_text("machine localhost:8080\n  login user\n  password " + "w" * 40 + "\n")
    monkeypatch.setenv("NETRC", "~/port.netrc")
    monkeypatch.setenv("WANDB_BASE_URL", "http://localhost:8080")
    assert launcher.check_wandb(online)["host"] == "http://localhost:8080"
    monkeypatch.delenv("WANDB_BASE_URL")
    monkeypatch.setenv("WANDB_IDENTITY_TOKEN_FILE", str(tmp_path / "token"))
    assert launcher.check_wandb(online)["credentials"] == "WANDB_IDENTITY_TOKEN_FILE"
    monkeypatch.delenv("WANDB_IDENTITY_TOKEN_FILE")
    assert launcher.check_wandb(compose("p2n_vla")) == {"mode": "offline"}

    import oat.env_runner.p2n_vla_rollout as rollout
    assert launcher.check_rollout(compose("p2n_vla")) == {"enabled": False}
    cfg = compose("p2n_vla", "task.policy.lazy_eval=false")
    monkeypatch.setenv("MUJOCO_GL", "egl")
    devices = [{"uuid": "GPU-a", "cuda_ordinal": 0, "egl_device_id": 1}]
    monkeypatch.setattr(rollout, "probe_egl_renderers", lambda: devices)
    report = launcher.check_rollout(cfg)
    assert report["egl_renderers"] == devices and report["plan"]["n_test"] == 500

    def broken():
        raise subprocess.CalledProcessError(1, ["probe"], stderr="libEGL.so.1: cannot open shared object file")
    monkeypatch.setattr(rollout, "probe_egl_renderers", broken)
    with pytest.raises(RuntimeError, match="install-display-drivers"):
        launcher.check_rollout(cfg)


def test_rollout_plan_stops_where_training_stops_and_rejects_empty_plans():
    # the pilot stops at 4,000 updates = epoch 40, so only the final rollout happens
    pilot = compose("p2n_vla", "task.policy.lazy_eval=false", "training.max_optimizer_steps=4000")
    assert [(p["epoch"], p["optimizer_step"]) for p in launcher.rollout_plan(pilot)["rollouts"]] == [(40, 4000)]
    with pytest.raises(ValueError, match="No in-training rollout"):
        compose("p2n_vla", "task.policy.lazy_eval=false", "training.rollout_every=400", "training.rollout_at_final=false")
    with pytest.raises(ValueError, match="max_train_steps"):  # epoch length must be known to plan rollouts
        compose("p2n_vla", "task.policy.lazy_eval=false", "training.max_train_steps=null")
    with pytest.raises(ValueError, match="too close"):  # 500 one-episode chunks would outlast the watchdog
        compose("p2n_vla", "task.policy.lazy_eval=false", "task.policy.env_runner.n_parallel_envs=1")
    with pytest.raises(ValueError, match="rollout_timeout_minutes"):
        compose("p2n_vla", "task.policy.lazy_eval=false", "training.rollout_timeout_minutes=130")
    # a probe-only setting never blocks a training run (the probe rounds its episodes up to a full chunk)
    compose("p2n_vla", "task.policy.lazy_eval=false", "task.policy.env_runner.n_parallel_envs=20")
    compose("p2n_vla", "task.policy.lazy_eval=false", "training.probe.rollout_episodes=0")
    plan = launcher.rollout_plan(compose("p2n_vla", "task.policy.lazy_eval=false"))
    assert 35 < plan["expected_minutes_per_rollout"] < 45 and plan["timeout_minutes"] == 100


def test_resume_warns_when_evaluation_or_logging_settings_change(monkeypatch, capsys):
    from oat.workspace.train_p2n_vla import TrainP2NVLAWorkspace
    original = compose("p2n_vla", "task.policy.lazy_eval=false", "logging.mode=online")
    payload = {"cfg": OmegaConf.to_container(original, resolve=True),
               "training": {"dataset_split": {"identity": {"train_episode_ids": [1], "validation_episode_ids": [2]}},
                            "counters": {"epoch": 50}, "world_size": 2}}
    monkeypatch.setattr(TrainP2NVLAWorkspace, "read_payload", staticmethod(lambda path: payload))
    monkeypatch.setattr(TrainP2NVLAWorkspace, "validate_resume_payload", staticmethod(lambda *a, **k: None))
    split = {"train_episode_ids": [1], "validation_episode_ids": [2]}
    plain = compose("p2n_vla", "training.resume=true", "training.resume_checkpoint=/x/latest.ckpt")
    warnings = launcher.check_resume(plain, 2, split)["warnings"]
    assert any("task.policy.lazy_eval=False" in item for item in warnings)
    assert any("logging.mode=online" in item for item in warnings)
    assert "WARNING (resume)" in capsys.readouterr().err
    same = compose("p2n_vla", "training.resume=true", "training.resume_checkpoint=/x/latest.ckpt",
                   "task.policy.lazy_eval=false", "logging.mode=online")
    assert launcher.check_resume(same, 2, split)["warnings"] == []


def test_tiny_override_is_valid_when_weights_are_unset():
    cfg = compose("p2n_vla", "policy.model_size=tiny", "policy.pi05_weights=null")
    assert cfg.policy.model_size == "tiny" and cfg.policy.pi05_weights is None


def test_cli_dry_run_applies_paths_and_overrides(tmp_path, capsys):
    oat = tmp_path / "oat.ckpt"
    summary = launcher.main(["--variant", "p2n_vla", "--task", "libero", "--dry-run", "--tokenizer", str(oat),
                             "--pi05", str(tmp_path / "w.safetensors"), "--spm", str(tmp_path / "t.model"),
                             "--devices", "0,1", "--output", str(tmp_path / "run"),
                             "--", "training.num_epochs=1", "logging.mode=online"])
    assert summary["mode"] == "dry_run" and summary["world_size"] == 2
    assert summary["assets"]["tokenizer_checkpoint"] == str(oat)
    assert summary["assets"]["pi05_weights"] == str(tmp_path / "w.safetensors")
    assert summary["assets"]["spm_path"] == str(tmp_path / "t.model")
    assert summary["logging_mode"] == "online" and summary["output"] == str(tmp_path / "run")
    assert summary["schedule"]["planned_optimizer_updates"] == 100  # one 100-update epoch
    assert "Configuration resolved and validated only" in capsys.readouterr().out
    resumed = launcher.main(["--variant", "p2n_vla_state_gate", "--dry-run",
                             "--resume", str(tmp_path / "run7/checkpoints/latest.ckpt")])
    assert resumed["variant"] == "p2n_vla_state_gate"
    assert resumed["output"] == str(tmp_path / "run7")  # resumes into the checkpoint's run directory
    probe = launcher.compose_config("p2n_vla", "libero", ["training.probe.enabled=true"])
    assert probe.training.probe.enabled and probe.training.probe.optimizer_steps == 3


@pytest.mark.parametrize("argv", [
    ["--variant", "p2n_vla", "--dry-run", "--devices", "0,1", "--num-processes", "3"],
    ["--variant", "p2n_vla", "--dry-run", "--devices", "0,0"],
    ["--variant", "p2n_vla", "--probe", "--resume", "x.ckpt"],
    ["--variant", "p2n_vla", "--probe", "--probe-steps", "0"],
    ["--variant", "p2n_vla", "--dry-run", "--preflight"],
    ["--variant", "p2n_new", "--dry-run"],
])
def test_cli_rejects_inconsistent_arguments(argv):
    with pytest.raises(SystemExit):
        launcher.main(argv)


def test_output_directory_and_asset_checks(tmp_path):
    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "file").write_text("x")
    with pytest.raises(ValueError, match="empty"):
        launcher.check_output_dir(tmp_path / "busy", resume=False)
    assert launcher.check_output_dir(tmp_path / "busy", resume=True)["resume"]
    assert not launcher.check_output_dir(tmp_path / "new/run", resume=False)["exists"]
    spm = tmp_path / "fake.model"
    spm.write_bytes(b"not the pinned tokenizer")
    cfg = compose("p2n_vla", "policy.model_size=tiny", "policy.pi05_weights=null", f"policy.spm_path={spm}")
    with pytest.raises(ValueError, match="SentencePiece sha256"):
        launcher.check_assets(cfg)
    missing = compose("p2n_vla", f"policy.spm_path={tmp_path / 'missing.model'}")
    with pytest.raises(FileNotFoundError):
        launcher.check_assets(missing)


def test_worker_entry_runs_the_workspace_in_process(tmp_path):
    cfg = stub_config(**{"training.num_epochs": 1, "training.max_train_steps": 2, "training.snapshot_every": 0})
    path = tmp_path / "worker.yaml"
    OmegaConf.save(cfg, path)
    launcher.main(["--worker-config", str(path), "--output", str(tmp_path / "run")])
    assert sorted(p.name for p in (tmp_path / "run/snapshots").iterdir()) == ["upd-000001_ema.ckpt"]
    assert (tmp_path / "run/checkpoints/latest.ckpt").is_file()


@pytest.mark.requires_data
@pytest.mark.requires_pi05
def test_light_preflight_on_the_real_dataset_oat_and_assets(tmp_path):
    for path in (LIBERO_ZARR, OAT, Path(ASSETS["files"]["pi05_base/model.safetensors"]["path"])):
        if not path.exists():
            pytest.skip(f"MISSING RESOURCE: {path}")
    for variant in VARIANTS:
        cfg = compose(variant)
        report = launcher.preflight(cfg, tmp_path / f"{variant}_out", 2)
        split = report["dataset_split"]
        assert split["train"] == {"episodes": 450, "windows": 124600}
        assert split["validation"] == {"episodes": 50, "windows": 13490}
        sources = report["sources"]
        assert sources["oat"]["levels"] == [8, 5, 5, 5, 5] and sources["oat"]["weights"] == "ema_model"
        assert sources["assets"]["pi05"]["tensors"] == 812
        assert sources["assets"]["pi05"]["sha256"] == ASSETS["files"]["pi05_base/model.safetensors"]["sha256"]
        prompts = sources["prompts"]
        assert prompts["state_dims"] == 8 and set(prompts["per_uid"]) == {str(uid) for uid in range(30, 40)}
        assert prompts["worst_case_tokens"] <= 96
        assert report["schedule"]["planned_optimizer_updates"] == 30000
        assert report["schedule"]["windows_seen_per_epoch"] == 3200  # 200 micro-batches x 8 x 2 ranks
        assert sources["rollout"] == {"enabled": False} and sources["wandb"] == {"mode": "offline"}


@pytest.mark.requires_data
def test_preflight_rejects_a_mismatched_tokenizer_split(tmp_path):
    if not LIBERO_ZARR.exists() or not OAT.exists():
        pytest.skip(f"MISSING RESOURCE: {LIBERO_ZARR} or {OAT}")
    cfg = compose("p2n_vla", "task.policy.dataset.val_ratio=0.05")
    with pytest.raises(ValueError, match="val_ratio"):
        launcher.validate_tokenizer_source(cfg)


# ------------------------------------------------------------------------------ eval script
def eval_args(snapshot, *extra):
    return evaluate.parser().parse_args(["--snapshot", str(snapshot), *extra])


def test_eval_argument_validation(tmp_path):
    snapshot = tmp_path / "upd-000001_ema.ckpt"
    snapshot.write_bytes(b"x")
    evaluate.validate_args(eval_args(snapshot))
    for extra, message in ((["--n-test", "0"], "n-test"), (["--use-k-tokens", "9"], "use-k-tokens"),
                           (["--temperature", "-1"], "temperature"), (["--init-state-offset", "3"], "official"),
                           (["--n-test-vis", "5", "--n-test", "2"], "n-test-vis"),
                           (["--pi05", str(tmp_path / "none")], "pi05")):
        with pytest.raises((ValueError, FileNotFoundError), match=message):
            evaluate.validate_args(eval_args(snapshot, *extra))
    with pytest.raises(SystemExit):
        eval_args(snapshot, "--protocol", "legacy")
    run = tmp_path / "run"
    (run / "snapshots").mkdir(parents=True)
    args = eval_args(run / "snapshots/upd-030000_ema.ckpt", "--protocol", "official", "--n-test", "500",
                     "--seed", "44", "--episode-start-seed", "3000", "--force-gate", "closed")
    assert evaluate.default_output_dir(args) == \
        run / "eval/upd-030000_ema_ema_official_n500_seed44_ep3000_gate-closed"


def test_eval_default_output_dir_is_unique_per_result_affecting_option(tmp_path):
    """Regression: the never-overwrite rule made paired evaluations of one snapshot collide.

    docs/P2N_VLA.md runs the official protocol and then ``--use-k-tokens 4`` on the same
    snapshot with the same protocol, n-test and seed; both used to map to one folder.
    """
    snapshot = tmp_path / "run/snapshots/upd-030000_ema.ckpt"
    official = ["--protocol", "official", "--n-test", "500", "--seed", "44", "--episode-start-seed", "3000"]
    variants = [
        [], ["--use-k-tokens", "4"], ["--use-k-tokens", "2"], ["--temperature", "0.5"],
        ["--temperature", "0.5", "--topk", "5"], ["--force-gate", "closed"], ["--force-gate", "open"],
        ["--episode-start-seed", "4000"], ["--tasks", "0,1"], ["--tasks", "0,2"],
        ["--tasks", "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"],
        ["--init-state-offset", "50"], ["--max-episode-steps", "600"], ["--weights", "model"],
    ]
    names = [evaluate.default_output_dir(eval_args(snapshot, *official, *extra)) for extra in variants]
    assert len(set(names)) == len(names), names
    assert all(path.parent == tmp_path / "run/eval" for path in names)
    assert names[1].name == "upd-030000_ema_ema_official_n500_seed44_ep3000_k4"
    assert names[4].name.endswith("_T0.5_topk5")
    corrected = eval_args(snapshot, "--protocol", "corrected", "--n-test", "100", "--seed", "45",
                          "--episode-start-seed", "4000")
    assert evaluate.default_output_dir(corrected).name == "upd-030000_ema_ema_corrected_n100_seed45_ep4000"


class StubFlowPolicy(StubVLAPolicy):
    """Stand-in for PI05KIFlowPolicy: inherits the token keywords but samples by flow matching."""

    VARIANT = "pi05_ki_flow"
    VARIANT_CODE = 12
    HAS_AR_HEAD = False
    supports_generated_history_validation = False

    def __init__(self, *args, flow_num_steps=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.flow_num_steps = flow_num_steps


def test_eval_rejects_token_overrides_the_flow_head_would_silently_ignore():
    """Regression: PI05KIFlowPolicy.predict_action accepts use_k_tokens/temperature/topk and ignores them."""
    flow = StubFlowPolicy()
    assert not evaluate.decodes_tokens(flow) and not evaluate.decodes_tokens(StubFlowPolicy)
    assert evaluate.decodes_tokens(StubVLAPolicy()) and evaluate.decodes_tokens(StubGatePolicy)
    for extra in (["--use-k-tokens", "4"], ["--temperature", "0.5"], ["--temperature", "0.5", "--topk", "3"]):
        args = evaluate.parser().parse_args(["--snapshot", "x", *extra])
        for target in (flow, StubFlowPolicy):  # instance, and the class pre-check before the weights load
            with pytest.raises(ValueError, match="no autoregressive token head"):
                evaluate.inference_kwargs(target, args)
    plain_args = evaluate.parser().parse_args(["--snapshot", "x"])
    assert evaluate.inference_kwargs(flow, plain_args) == {}
    inference = evaluate.effective_inference(flow, plain_args, None)
    assert inference["decoder"] == "flow_matching" and inference["flow_num_steps"] == 10
    assert inference["use_k_tokens"] is None and inference["temperature"] is None and inference["topk"] is None
    ar = evaluate.effective_inference(StubVLAPolicy(), plain_args, None)
    assert ar["decoder"] == "autoregressive_oat_tokens" and ar["temperature"] == 0.0


def test_eval_rejects_topk_without_sampling_and_k_beyond_the_latent_horizon():
    policy = StubVLAPolicy()  # temperature 0.0: greedy decoding never reads top-k
    topk_only = evaluate.parser().parse_args(["--snapshot", "x", "--topk", "5"])
    with pytest.raises(ValueError, match="no effect at temperature 0"):
        evaluate.inference_kwargs(policy, topk_only)
    sampled = evaluate.parser().parse_args(["--snapshot", "x", "--topk", "5", "--temperature", "0.7"])
    assert evaluate.inference_kwargs(policy, sampled) == {"temperature": 0.7, "topk": 5}
    policy.max_seq_len = 4
    too_many = evaluate.parser().parse_args(["--snapshot", "x", "--use-k-tokens", "6"])
    with pytest.raises(ValueError, match="exceeds the 4 OAT tokens"):
        evaluate.inference_kwargs(policy, too_many)


def test_force_gate_inference_kwargs_and_runner_selection():
    gate = StubGatePolicy()
    assert evaluate.apply_force_gate(gate, "closed") == {"mode": "closed", "mechanism": "set_history_gate_mode"}
    assert gate.history_gate_mode == "closed" and gate.get_history_gate_metrics()["gate_mean"] == 0.0

    class AttributeOnly(StubGatePolicy):
        set_history_gate_mode = None
    attribute = AttributeOnly()
    assert evaluate.apply_force_gate(attribute, "open")["mechanism"].startswith("history_gate_mode")
    assert attribute.history_gate_mode == "open"
    with pytest.raises(ValueError, match="state_gate"):
        evaluate.apply_force_gate(StubVLAPolicy(), "open")
    assert evaluate.apply_force_gate(StubVLAPolicy(), None) is None

    plain = StubVLAPolicy()
    args = evaluate.parser().parse_args(["--snapshot", "x", "--temperature", "0.5", "--use-k-tokens", "4"])
    assert evaluate.inference_kwargs(plain, args) == {"temperature": 0.5, "use_k_tokens": 4}

    class NoTokens(StubVLAPolicy):  # e.g. the flow baseline has no token budget
        def predict_action(self, obs_dict, temperature=None, past_actions=None, past_action_valid=None):
            raise AssertionError("not called")
    with pytest.raises(ValueError, match="use_k_tokens"):
        evaluate.inference_kwargs(NoTokens(), args)

    cfg = {"task": {"policy": {"env_runner": {"_target_": "oat.env_runner.p2n_new_runner.P2NNewLiberoRunner",
                                              "task_name": "libero10", "protocol": "corrected"}}}}
    runner_args = evaluate.parser().parse_args(["--snapshot", "x", "--protocol", "official", "--n-test", "20"])
    config = evaluate.build_runner_config(cfg, plain, runner_args, None, Path("/tmp/out"))
    assert config["protocol"] == "official" and config["n_test"] == 20 and config["n_obs_steps"] == 1
    assert config["n_action_steps"] == 8 and config["episode_records_path"] == "/tmp/out/episodes.jsonl"
    with pytest.raises(ValueError, match="P2NStateGateNewLiberoRunner"):
        evaluate.build_runner_config(cfg, gate, runner_args, None, Path("/tmp/out"))


def test_eval_renders_on_the_policy_gpu_even_when_egl_indices_are_swapped(monkeypatch):
    # The 2x4090 host: CUDA 0 is EGL 1 and CUDA 1 is EGL 0, so MUJOCO_EGL_DEVICE_ID=<cuda index> is wrong.
    records = [{"uuid": "GPU-b1e32c9e-dd90-c8da-d410-1520cbfcf924", "cuda_ordinal": 1, "egl_device_id": 0},
               {"uuid": "GPU-443020d4-2ff8-c61c-997a-3ed6d0343fa3", "cuda_ordinal": 0, "egl_device_id": 1}]
    uuids = {0: "443020d4-2ff8-c61c-997a-3ed6d0343fa3", 1: "B1E32C9E-DD90-C8DA-D410-1520CBFCF924"}

    class Properties:
        def __init__(self, index):
            self.uuid = uuids[index]

    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda device: Properties(torch.device(device).index or 0))
    monkeypatch.setenv("MUJOCO_GL", "egl")
    assert evaluate.resolve_renderer("cuda:0", probe=lambda: records)["egl_device_id"] == 1
    assert evaluate.resolve_renderer("cuda:1", probe=lambda: records)["egl_device_id"] == 0
    assert evaluate.resolve_renderer("cpu", probe=lambda: records) is None
    with pytest.raises(RuntimeError, match="Cannot identify one EGL renderer"):
        evaluate.resolve_renderer("cuda:0", probe=lambda: records[:1])
    monkeypatch.setenv("MUJOCO_GL", "osmesa")
    assert evaluate.resolve_renderer("cuda:0", probe=lambda: records) is None

    monkeypatch.delenv("P2N_LIBERO_EGL_DEVICE_ID", raising=False)
    legacy = {"_target_": "oat.env_runner.p2n_new_runner.P2NStateGateNewLiberoRunner", "n_test": 10}
    assert evaluate.scope_runner_to_renderer(legacy, None) is legacy and "P2N_LIBERO_EGL_DEVICE_ID" not in os.environ
    scoped = evaluate.scope_runner_to_renderer(legacy, records[1])
    assert scoped["_target_"] == "oat.env_runner.p2n_new_convnext_libero10_runner.P2NStateGateNewLiberoRunner"
    assert scoped["n_test"] == 10 and legacy["_target_"].startswith("oat.env_runner.p2n_new_runner.")
    assert os.environ["P2N_LIBERO_EGL_DEVICE_ID"] == "1"
    import oat.env_runner.p2n_new_convnext_libero10_runner as scoped_module
    for name in evaluate.RUNNERS.values():  # the wrappers exist for both runner classes
        assert getattr(scoped_module, name)._legacy_name == name


@pytest.mark.slow
def test_eval_dry_run_restores_a_snapshot_and_writes_provenance(tmp_path, monkeypatch):
    # main() chdirs to the repo and sets rendering/thread variables; keep them test-local.
    monkeypatch.chdir(REPO)
    for key, default in (("MUJOCO_GL", "egl"), ("OMP_NUM_THREADS", "4"), ("MKL_NUM_THREADS", "4"),
                         ("HF_HOME", "/workspace/.hf_home")):
        monkeypatch.setenv(key, os.environ.get(key, default))
    threads = str(torch.get_num_threads())
    cfg = stub_config(**{"training.num_epochs": 1, "training.max_train_steps": 2, "training.snapshot_every": 0})
    OmegaConf.update(cfg, "task.policy.env_runner", {
        "_target_": "oat.env_runner.p2n_new_runner.P2NNewLiberoRunner", "task_name": "libero10",
        "protocol": "corrected", "n_test": 100, "n_test_vis": 0, "n_parallel_envs": 10}, merge=False)
    run_workspace(cfg, tmp_path / "run")
    snapshot = tmp_path / "run/snapshots/upd-000001_ema.ckpt"
    output = tmp_path / "eval"
    assert evaluate.main(["--snapshot", str(snapshot), "--output-dir", str(output), "--device", "cpu",
                          "--n-test", "20", "--threads", threads, "--dry-run"]) == 0
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["status"] == "ready" and metadata["variant"] == "p2n_vla"
    assert metadata["optimizer_step"] == 1 and metadata["checkpoint_kind"] == "snapshot"
    assert metadata["effective_inference"]["temperature"] == 0.0
    assert metadata["all_tasks_covered"] and metadata["balanced_schedule"]
    schedule = json.loads((output / "schedule.json").read_text())
    assert len(schedule) == 20 and schedule[0]["episode_seed"] == 1000
    runner = json.loads((output / "runner_config.json").read_text())
    assert runner["_target_"].endswith("P2NNewLiberoRunner") and runner["n_test"] == 20
    hashes = json.loads((output / "source_hashes.json").read_text())
    assert "scripts/evaluate_p2n_vla.py" in hashes and "oat/workspace/train_p2n_vla.py" in hashes
    with pytest.raises(FileExistsError):
        evaluate.main(["--snapshot", str(snapshot), "--output-dir", str(output), "--device", "cpu",
                       "--threads", threads, "--dry-run"])
    # Snapshots hold only EMA weights.
    with pytest.raises(ValueError, match="no 'model' weights"):
        evaluate.main(["--snapshot", str(snapshot), "--output-dir", str(tmp_path / "eval_model"),
                       "--device", "cpu", "--weights", "model", "--threads", threads, "--dry-run"])
    failed = json.loads((tmp_path / "eval_model/metadata.json").read_text())
    assert failed["status"] == "failed" and "no 'model' weights" in failed["error"]


@pytest.mark.slow
def test_eval_dry_run_with_a_real_tiny_policy_snapshot(tmp_path, monkeypatch):
    from test_p2n_vla_workspace import _require_real_policy_stack, make_libero_like_zarr, real_tiny_config
    _require_real_policy_stack()
    monkeypatch.chdir(REPO)
    for key, default in (("MUJOCO_GL", "egl"), ("OMP_NUM_THREADS", "4"), ("MKL_NUM_THREADS", "4"),
                         ("HF_HOME", "/workspace/.hf_home")):
        monkeypatch.setenv(key, os.environ.get(key, default))
    zarr_path = make_libero_like_zarr(tmp_path / "libero_like.zarr")
    run_workspace(real_tiny_config("p2n_vla_state_gate", zarr_path, "training.num_epochs=1"), tmp_path / "run")
    snapshot = sorted((tmp_path / "run/snapshots").iterdir())[-1]
    output = tmp_path / "eval"
    assert evaluate.main(["--snapshot", str(snapshot), "--output-dir", str(output), "--device", "cpu",
                          "--n-test", "10", "--protocol", "official", "--seed", "44", "--episode-start-seed", "3000",
                          "--force-gate", "closed", "--threads", str(torch.get_num_threads()), "--dry-run"]) == 0
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["status"] == "ready" and metadata["variant"] == "p2n_vla_state_gate"
    assert metadata["policy_class"] == "oat.policy.p2n_vla_state_gate.P2NVLAStateGatePolicy"
    assert metadata["runner_target"].endswith("P2NStateGateNewLiberoRunner")
    inference = metadata["effective_inference"]
    assert inference["temperature"] == 0.0 and inference["use_k_tokens"] == 8 and inference["n_action_steps"] == 8
    assert inference["force_gate"]["mode"] == "closed"
    assert metadata["assets"]["tokenizer"]["sha256"] and metadata["assets"]["sentencepiece"]["sha256"]
    assert len(metadata["official_initial_state_files"]) == 10
    schedule = json.loads((output / "schedule.json").read_text())
    assert [record["init_state_id"] for record in schedule] == [0] * 10 and schedule[0]["episode_seed"] == 3000


@pytest.mark.slow
def test_eval_dry_run_of_the_real_flow_baseline_rejects_token_overrides(tmp_path, monkeypatch):
    """The real PI05KIFlowPolicy: token keywords fail fast (before the restore) instead of being ignored."""
    from test_p2n_vla_workspace import _require_real_policy_stack, make_libero_like_zarr, real_tiny_config
    _require_real_policy_stack()
    monkeypatch.chdir(REPO)
    for key, default in (("MUJOCO_GL", "egl"), ("OMP_NUM_THREADS", "4"), ("MKL_NUM_THREADS", "4"),
                         ("HF_HOME", "/workspace/.hf_home")):
        monkeypatch.setenv(key, os.environ.get(key, default))
    zarr_path = make_libero_like_zarr(tmp_path / "libero_like.zarr")
    run_workspace(real_tiny_config("pi05_ki_flow", zarr_path, "training.num_epochs=1"), tmp_path / "run")
    snapshot = sorted((tmp_path / "run/snapshots").iterdir())[-1]
    common = ["--snapshot", str(snapshot), "--device", "cpu", "--n-test", "10",
              "--threads", str(torch.get_num_threads()), "--dry-run"]
    assert evaluate.main([*common, "--output-dir", str(tmp_path / "eval")]) == 0
    inference = json.loads((tmp_path / "eval/metadata.json").read_text())["effective_inference"]
    assert inference["decoder"] == "flow_matching" and inference["flow_num_steps"] == 10
    assert inference["use_k_tokens"] is None and inference["temperature"] is None

    from oat.policy.pi05_ki_flow import PI05KIFlowPolicy
    loads = []
    monkeypatch.setattr(PI05KIFlowPolicy, "from_checkpoint",
                        classmethod(lambda cls, *args, **kwargs: loads.append(1)))
    for extra, name in ((["--use-k-tokens", "4"], "k4"), (["--temperature", "0.5"], "T"),
                        (["--force-gate", "closed"], "gate")):
        output = tmp_path / f"eval_{name}"
        with pytest.raises(ValueError, match="no autoregressive token head|state_gate"):
            evaluate.main([*common, *extra, "--output-dir", str(output)])
        assert json.loads((output / "metadata.json").read_text())["status"] == "failed"
    assert not loads  # rejected before the multi-GB restore
