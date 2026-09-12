"""Train or explicitly resume one single-GPU real-robot policy with a frozen tokenizer."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_real_robot import (
    ACTION_SEMANTICS, DATASETS, SEED, VAL_RATIO, VARIANTS,
    best_policy_checkpoint, dataset_manifest, stage_command,
)


def single_gpu(value):
    if not value.isascii() or not value.isdigit():
        raise argparse.ArgumentTypeError("--gpu requires one nonnegative GPU index, not a list")
    return int(value)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_tokenizer_config(cfg, task, variant):
    """Reject a tokenizer from another task, split, shape, or augmentation run."""
    from omegaconf import OmegaConf
    from oat.common.hydra_util import register_new_resolvers

    register_new_resolvers()
    OmegaConf.resolve(cfg)
    expected = {
        "_target_": "oat.workspace.train_oattok.TrainOATTokWorkspace",
        "tokenizer._target_": "oat.tokenizer.oat.tokenizer_so3_aug.OATTokSO3Aug",
        "task.tokenizer.task_name": task,
        "task.tokenizer.name": f"real_robot_{task}",
        "task.tokenizer.fps": 30,
        "task.tokenizer.dataset._target_": "oat.dataset.real_robot_dataset.RealRobotZarrDataset",
        "task.tokenizer.dataset.seed": SEED,
        "task.tokenizer.dataset.val_ratio": VAL_RATIO,
        "task.tokenizer.dataset.max_train_episodes": None,
        "training.num_demo": DATASETS[task][1],
        "seed": SEED,
        "action_dim": 7,
        "horizon": 16,
        "tokenizer.encoder.sample_dim": 7,
        "tokenizer.encoder.sample_horizon": 16,
        "tokenizer.decoder.sample_dim": 7,
        "tokenizer.decoder.sample_horizon": 16,
        "tokenizer.action_aug.mode": VARIANTS[variant]["mode"],
        "tokenizer.action_aug.augment_position": VARIANTS[variant]["augment_position"],
        "tokenizer.action_aug.max_angle_deg": 30.0,
        "tokenizer.action_aug.p": 0.6,
        "tokenizer.action_aug.pos_start": 0,
        "tokenizer.action_aug.pos_end": 3,
        "tokenizer.action_aug.rot_start": 3,
        "tokenizer.action_aug.rot_end": 6,
    }
    for key, value in expected.items():
        actual = OmegaConf.select(cfg, key, default="<missing>")
        if actual != value:
            raise ValueError(f"Tokenizer mismatch for {key}: {actual!r}; expected {value!r}")
    if list(cfg.task.tokenizer.shape_meta.action.shape) != [7]:
        raise ValueError("Tokenizer task action shape must be [7]")
    dataset_path = Path(cfg.task.tokenizer.dataset.zarr_path).resolve()
    if dataset_path != DATASETS[task][0].resolve():
        raise ValueError(f"Tokenizer dataset {dataset_path} does not match task {task}")
    return {
        "task": task, "variant": variant, "dataset": str(dataset_path),
        "action_dim": 7, "horizon": 16, "seed": SEED, "val_ratio": VAL_RATIO,
        "augmentation": {**VARIANTS[variant], "max_angle_deg": 30.0, "p": 0.6},
    }


def verify_tokenizer(path, task, variant):
    """Load trusted local training weights on CPU before policy initialization."""
    import dill
    import hydra
    import torch

    torch.set_num_threads(4)
    with path.open("rb") as stream:
        payload = torch.load(stream, map_location="cpu", pickle_module=dill, weights_only=False)
    cfg = payload["cfg"]
    metadata = verify_tokenizer_config(cfg, task, variant)
    model = hydra.utils.instantiate(cfg.tokenizer)
    state_key = "ema_model" if cfg.training.use_ema else "model"
    model.load_state_dict(payload["state_dicts"][state_key], strict=True)
    if model.decoder.sample_dim != 7 or model.decoder.sample_horizon != 16:
        raise ValueError("Loaded tokenizer has incompatible action dimensions")
    if tuple(model.normalizer["action"].params_dict["scale"].shape) != (7,):
        raise ValueError("Tokenizer does not contain a seven-channel action normalizer")
    for key, value in model.state_dict().items():
        if torch.is_floating_point(value) and not torch.isfinite(value).all():
            raise ValueError(f"Tokenizer contains nonfinite weights: {key}")
    metadata.update(state_key=state_key, weights_loaded_on="cpu", weights_finite=True)
    return metadata


def policy_command(task, variant, output_dir, tokenizer, entity, project,
                   run_id, run_name, smoke=False, policy_epochs=251):
    command = stage_command(
        "policy", task, variant, output_dir / "policy", 1,
        tokenizer=tokenizer, smoke=smoke, policy_epochs=policy_epochs)
    logging = {
        "logging.mode": "offline" if smoke else "online",
        "logging.project": project,
        "logging.group": f"{task}_{variant}_single_gpu",
        "logging.id": run_id,
        "logging.name": run_name,
    }
    command = [arg for arg in command if arg.split("=", 1)[0] not in logging]
    command += [f"{key}={json.dumps(value)}" for key, value in logging.items()]
    command += [f"+logging.entity={json.dumps(entity)}"]
    return command


def run_manifest(args):
    output = args.output_dir.resolve()
    source = args.tokenizer.resolve(strict=True)
    metadata = verify_tokenizer(source, args.task, args.variant)
    run_id = uuid.uuid4().hex[:12]
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"{args.task}_{args.variant}_single_gpu_{timestamp}_{run_id}"
    frozen = output / "frozen_tokenizer.ckpt"
    command = policy_command(args.task, args.variant, output, frozen,
                             args.entity, args.project, run_id, run_name,
                             args.smoke, args.policy_epochs)
    paths = [
        "scripts/train_real_robot_policy.py", "scripts/train_real_robot.py",
        "scripts/run_workspace.py", "scripts/check_real_robot_checkpoint.py",
        "oat/config/train_past2next_scratch_all500.yaml",
        f"oat/config/task/policy/real_robot/{args.task}_with_prev_window.yaml",
        "oat/dataset/real_robot_dataset.py", "oat/workspace/train_policy.py",
        "oat/policy/past2next.py", "oat/policy/past2next_self_past.py",
    ]
    return {
        "schema_version": 1, "run_type": "policy_only_single_gpu_from_scratch",
        "task": args.task, "variant": args.variant, "gpu": args.gpu,
        "output_dir": str(output), "smoke": args.smoke,
        "policy_config": "train_past2next_scratch_all500",
        "policy_epochs": 1 if args.smoke else args.policy_epochs,
        "policy_resume": False, "policy_init_checkpoint": None,
        "global_batch_size": 64, "per_gpu_batch_size": 64,
        "tokenizer": {"source": str(source), "source_sha256": sha256_file(source),
                      "copied_path": str(frozen), "metadata": metadata,
                      "retrained": False},
        "action_semantics": ACTION_SEMANTICS,
        "dataset": dataset_manifest(args.task),
        "wandb": {"entity": args.entity, "project": args.project, "run_id": run_id,
                  "run_name": run_name, "group": f"{args.task}_{args.variant}_single_gpu",
                  "mode": "offline" if args.smoke else "online",
                  "url": None if args.smoke else
                      f"https://wandb.ai/{args.entity}/{args.project}/runs/{run_id}"},
        "python": sys.executable,
        "source_sha256": {path: sha256_file(ROOT / path) for path in paths},
        "commands": {"policy": command},
    }


def _command_config(command):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from oat.common.hydra_util import register_new_resolvers

    register_new_resolvers()
    config_arg = next(arg for arg in command if arg.startswith("--config-name="))
    with initialize_config_dir(config_dir=str(ROOT / "oat/config"), version_base=None):
        cfg = compose(config_name=config_arg.split("=", 1)[1],
                      overrides=command[command.index(config_arg) + 1:])
    OmegaConf.resolve(cfg)
    return cfg


def verify_resume_checkpoint(path, expected_cfg):
    """Strictly load complete v2 model, optimizer, EMA, and scheduler state on CPU."""
    import copy
    import math
    import dill
    import hydra
    import torch
    from omegaconf import OmegaConf
    from oat.workspace.train_policy import TrainPolicyWorkspace
    from oat.model.common.lr_scheduler import get_scheduler

    torch.set_num_threads(4)
    with path.open("rb") as stream:
        payload = torch.load(stream, map_location="cpu", pickle_module=dill, weights_only=False)
    saved = {key: dill.loads(value) for key, value in payload["pickles"].items()}
    required = ("checkpoint_version", "epoch", "global_step", "completed_optimizer_steps",
                "ema_state", "lr_scheduler_state")
    if any(key not in saved for key in required) or saved["checkpoint_version"] != 2:
        raise ValueError("Resume requires a complete version-2 training checkpoint")
    epoch, batches, updates = (saved[key] for key in
                              ("epoch", "global_step", "completed_optimizer_steps"))
    if not all(isinstance(value, int) and value > 0 for value in (epoch, batches, updates)):
        raise ValueError("Resume checkpoint must contain completed epochs and optimizer updates")
    if epoch > expected_cfg.training.num_epochs or updates > batches:
        raise ValueError("Resume checkpoint has inconsistent training counters")
    scheduler_state, ema_state = saved["lr_scheduler_state"], saved["ema_state"]
    if (not isinstance(scheduler_state, dict) or scheduler_state.get("last_epoch") != updates
            or not scheduler_state.get("_last_lr")):
        raise ValueError("Resume checkpoint is missing coherent scheduler state")
    if (not isinstance(ema_state, dict) or ema_state.get("optimization_step") != updates
            or not math.isfinite(float(ema_state.get("decay", float("nan"))))):
        raise ValueError("Resume checkpoint is missing coherent EMA state")
    cfg = payload["cfg"]
    OmegaConf.resolve(cfg)
    for key in ("policy", "shape_meta", "task", "optimizer", "ema", "dataloader",
                "val_dataloader", "seed", "horizon", "n_action_steps", "n_obs_steps", "past_n"):
        actual, expected = OmegaConf.select(cfg, key), OmegaConf.select(expected_cfg, key)
        if OmegaConf.is_config(actual):
            actual = OmegaConf.to_container(actual, resolve=True)
            expected = OmegaConf.to_container(expected, resolve=True)
        if actual != expected:
            raise ValueError(f"Resume checkpoint differs from the original run's {key}")
    for key, value in OmegaConf.to_container(expected_cfg.training, resolve=True).items():
        if key not in ("resume", "resume_checkpoint", "checkpoint_every", "snapshot_every") and cfg.training.get(key) != value:
            raise ValueError(f"Resume checkpoint training.{key} differs from the original run")
    for key in ("id", "name", "entity", "project", "group", "mode"):
        if cfg.logging.get(key) != expected_cfg.logging.get(key):
            raise ValueError(f"Resume checkpoint W&B {key} differs from the original run")
    states = payload["state_dicts"]
    if (not all(key in states for key in ("model", "ema_model", "optimizer"))
            or not states["optimizer"].get("state") or not states["optimizer"].get("param_groups")):
        raise ValueError("Resume checkpoint is missing model, EMA model, or optimizer state")
    workspace = TrainPolicyWorkspace(copy.deepcopy(cfg), lazy_instantiation=False)
    workspace.load_payload(payload)
    for name in ("model", "ema_model"):
        model = getattr(workspace, name)
        for key, value in model.state_dict().items():
            if torch.is_floating_point(value) and not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite {name} tensor in resume checkpoint: {key}")
        if int(model._self_past_optimizer_step.item()) != updates:
            raise ValueError(f"Resume {name} history counter differs from optimizer progress")
    for parameter, state in workspace.optimizer.state.items():
        if not all(key in state for key in ("step", "exp_avg", "exp_avg_sq")):
            raise ValueError("Resume checkpoint has incomplete Adam optimizer state")
        if not 0 < float(state["step"]) <= updates:
            raise ValueError("Resume optimizer step exceeds completed updates")
        for key in ("exp_avg", "exp_avg_sq"):
            if state[key].shape != parameter.shape or not torch.isfinite(state[key]).all():
                raise ValueError(f"Resume optimizer has invalid {key} tensors")
    ema = hydra.utils.instantiate(cfg.ema, model=workspace.ema_model)
    scheduler = get_scheduler(cfg.training.lr_scheduler, workspace.optimizer,
                              num_warmup_steps=cfg.training.lr_warmup_steps,
                              num_training_steps=1, last_epoch=updates - 1)
    workspace._restore_training_helpers(ema, scheduler)
    if scheduler.state_dict() != scheduler_state or ema.optimization_step != updates:
        raise ValueError("Resume helpers did not restore the exact scheduler and EMA progress")
    return {"path": str(path), "sha256": sha256_file(path), "checkpoint_version": 2,
            "next_epoch": epoch, "completed_batches": batches, "completed_optimizer_steps": updates,
            "model_optimizer_ema_scheduler_loaded_on": "cpu"}


def resume_manifest(args):
    import copy

    output = args.output_dir.resolve()
    original_path = output / "manifest.json"
    original = json.loads(original_path.read_text())
    for key, value in (("task", args.task), ("variant", args.variant), ("gpu", args.gpu),
                       ("output_dir", str(output)), ("policy_epochs", args.policy_epochs),
                       ("smoke", False), ("policy_config", "train_past2next_scratch_all500")):
        if original.get(key) != value:
            raise ValueError(f"Resume {key} must match the original manifest")
    if args.smoke:
        raise ValueError("--resume cannot change a full run into a smoke run")
    for key, value in (("entity", args.entity), ("project", args.project), ("mode", "online")):
        if original["wandb"].get(key) != value:
            raise ValueError(f"Resume W&B {key} must match the original manifest")
    source = args.tokenizer.resolve(strict=True)
    tokenizer = original["tokenizer"]
    frozen = output / "frozen_tokenizer.ckpt"
    if source != Path(tokenizer["source"]).resolve() or Path(tokenizer["copied_path"]).resolve() != frozen:
        raise ValueError("Resume tokenizer paths must match the original manifest")
    expected_hash = tokenizer["source_sha256"]
    if (tokenizer.get("copied_sha256") != expected_hash or sha256_file(source) != expected_hash
            or sha256_file(frozen) != expected_hash):
        raise ValueError("Resume tokenizer hashes differ from the original trained tokenizer")
    verify_tokenizer(frozen, args.task, args.variant)
    if dataset_manifest(args.task) != original["dataset"]:
        raise ValueError("Resume dataset or episode split differs from the original manifest")
    command = original["commands"]["policy"]
    if "--nproc_per_node=1" not in command or "--config-name=train_past2next_scratch_all500" not in command:
        raise ValueError("Resume requires the original single-GPU all500 policy command")
    cfg = _command_config(command)
    for key, value in (("id", original["wandb"]["run_id"]),
                       ("name", original["wandb"]["run_name"]),
                       ("entity", args.entity), ("project", args.project),
                       ("group", original["wandb"]["group"]), ("mode", "online")):
        if cfg.logging.get(key) != value:
            raise ValueError(f"Original command's W&B {key} differs from its manifest")
    if cfg.training.num_epochs != args.policy_epochs or cfg.training.init_checkpoint is not None:
        raise ValueError("Resume must preserve the original training budget and initialization")
    checkpoint = output / "policy/checkpoints/latest.ckpt"
    checkpoint_metadata = verify_resume_checkpoint(checkpoint, cfg)
    replacements = {
        "training.resume": "true", "logging.resume": '"must"',
        "checkpoint.topk.monitor_key": "test_reconst_mse",
        "checkpoint.topk.k": "0", "training.checkpoint_every": "20",
        "training.snapshot_every": "0",
        "checkpoint.topk.format_str": '"ep-{epoch:04d}_mse-{test_reconst_mse:.6f}.ckpt"',
    }
    resumed_command = [arg for arg in command
                       if arg.split("=", 1)[0].lstrip("+") not in (*replacements, "training.resume_checkpoint", "checkpoint.save_all")]
    resumed_command += [f"{key}={value}" for key, value in replacements.items()]
    resumed_command += [f"++training.resume_checkpoint={json.dumps(str(checkpoint))}",
                        "++checkpoint.save_all=true"]
    manifest = copy.deepcopy(original)
    manifest.update(policy_resume=True, resumed_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                    original_manifest_sha256=sha256_file(original_path),
                    resume_checkpoint=checkpoint_metadata, commands={"policy": resumed_command})
    manifest["source_sha256"] = {path: sha256_file(ROOT / path) for path in original["source_sha256"]}
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=DATASETS, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--gpu", type=single_gpu, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", default="real_robot")
    parser.add_argument("--policy-epochs", type=int, default=251)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Resume the existing run and W&B identity from its complete latest checkpoint")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate checkpoint and print commands without writing or training")
    args = parser.parse_args()
    if args.policy_epochs < 1:
        parser.error("--policy-epochs must be positive")
    if not args.entity.strip() or not args.project.strip():
        parser.error("W&B entity and project must be nonempty")
    output = args.output_dir.resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(f"Refusing existing output directory: {output}")
    if args.resume and not output.is_dir():
        raise FileNotFoundError(f"Resume output directory does not exist: {output}")
    manifest = resume_manifest(args) if args.resume else run_manifest(args)
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    if not args.resume:
        output.mkdir(parents=True, exist_ok=False)
    status = {"started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "state": "preparing", "task": args.task, "variant": args.variant,
              "gpu": args.gpu, "wandb": manifest["wandb"]}
    if args.resume:
        status_path = output / "status.json"
        if status_path.is_file():
            status = json.loads(status_path.read_text())
        status.update(state="preparing_resume", resumed_at=manifest["resumed_at"],
                      resume_count=int(status.get("resume_count", 0)) + 1,
                      resume_checkpoint=manifest["resume_checkpoint"])

    def save_status():
        status["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        temporary = output / "status.tmp.json"
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(output / "status.json")

    env = os.environ.copy()
    for key in ("WANDB_RUN_ID", "WANDB_NAME", "WANDB_RUN_GROUP", "WANDB_RESUME",
                "WANDB_SERVICE", "_WANDB_SERVICE"):
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONUNBUFFERED="1",
               WANDB_MODE="offline" if args.smoke else "online",
               WANDB_ENTITY=args.entity, WANDB_PROJECT=args.project,
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1",
               HYDRA_FULL_ERROR="1")
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"]
                                   if env.get("PYTHONPATH") else "")
    try:
        save_status()
        source = Path(manifest["tokenizer"]["source"])
        frozen = Path(manifest["tokenizer"]["copied_path"])
        if args.resume:
            checkpoint = Path(manifest["resume_checkpoint"]["path"])
            if sha256_file(checkpoint) != manifest["resume_checkpoint"]["sha256"]:
                raise ValueError("Resume checkpoint changed after validation; stop the previous job first")
            resume_dir = output / "resumes"
            resume_dir.mkdir(exist_ok=True)
            timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            manifest_path = resume_dir / f"{timestamp}.json"
            status["resume_manifest"] = str(manifest_path)
        else:
            shutil.copy2(source, frozen)
            copied_hash = sha256_file(frozen)
            manifest["tokenizer"]["copied_sha256"] = copied_hash
            if copied_hash != manifest["tokenizer"]["source_sha256"]:
                raise ValueError("Copied tokenizer differs from the validated source checkpoint")
            manifest_path = output / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        status.update(state="training_policy", frozen_tokenizer=str(frozen))
        save_status()
        command = manifest["commands"]["policy"]
        print(f"Starting {args.task}/{args.variant} on GPU {args.gpu}: {json.dumps(command)}", flush=True)
        with (output / "policy.log").open("a" if args.resume else "w") as logfile:
            subprocess.run(command, cwd=ROOT, env=env, stdout=logfile,
                           stderr=subprocess.STDOUT, check=True)
        selected, metric = best_policy_checkpoint(output / "policy")
        status.update(state="verifying_checkpoint", best_policy=str(selected), policy_mse=metric)
        save_status()
        check_path = output / "checkpoint_check.json"
        check_command = [sys.executable, str(ROOT / "scripts/check_real_robot_checkpoint.py"),
                         "--checkpoint", str(selected), "--device", "cuda:0",
                         "--output", str(check_path)]
        with (output / "checkpoint_check.log").open("w") as logfile:
            subprocess.run(check_command, cwd=ROOT, env=env, stdout=logfile,
                           stderr=subprocess.STDOUT, check=True)
        status.update(state="completed", checkpoint_check=str(check_path))
    except BaseException as exc:
        status.update(state="failed", error=str(exc))
        raise
    finally:
        save_status()


if __name__ == "__main__":
    main()
