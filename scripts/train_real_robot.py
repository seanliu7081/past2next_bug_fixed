"""Train a per-dataset tokenizer, then train_past2next_scratch_all500 from scratch.

Each invocation owns one dataset and one augmentation variant. Run the four
task/variant combinations in separate output directories. Use --dry-run to
inspect the dataset split and exact commands without creating files or training.
"""
import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEED = 42
VAL_RATIO = 0.1
DATASETS = {
    "fruits": (Path("/workspace/ysk/zarr/fruits_N51.zarr"), 51),
    "nut_washer": (Path("/workspace/ysk/zarr/nut_washer_N62.zarr"), 62),
}
VARIANTS = {
    "current": {"mode": "conjugate", "augment_position": True},
    "left_noise": {"mode": "left_noise", "augment_position": False},
}
OBS_SHAPES = {
    "agentview_rgb": (128, 128, 3),
    "robot0_eye_in_hand_rgb": (128, 128, 3),
    "robot0_eef_pos": (3,),
    "robot0_eef_rot6d": (6,),
    "robot0_gripper_qpos": (1,),
    "task_uid": (1,),
}
ACTION_SEMANTICS = {
    "fps": 30,
    "translation": {"slice": [0, 3], "representation": "delta_position",
                    "frame": "base", "units": "metres",
                    "integration": "p_next = p_previous + dp"},
    "rotation": {"slice": [3, 6], "representation": "rotation_vector",
                 "frame": "end_effector", "units": "radians",
                 "integration": "R_next = R_previous @ Exp(drotvec)"},
    "gripper": {"index": 6, "representation": "absolute_command",
                "open": 0.0, "closed": 1.0},
    "gripper_observation": {"key": "robot0_gripper_qpos",
                            "representation": "measured_width",
                            "units": "millimetres", "range": [0, 80]},
}


def _best_checkpoint(stage_dir, metric_name, filename):
    """Use full precision metrics and only checkpoints retained on disk."""
    candidates = []
    for line in (stage_dir / "logs.json").read_text().splitlines():
        record = json.loads(line)
        metric = record.get(metric_name)
        if metric is None or not math.isfinite(metric):
            continue
        path = stage_dir / "checkpoints" / filename.format(**record)
        if path.is_file():
            candidates.append((metric, -record["epoch"], path))
    if not candidates:
        raise RuntimeError(
            f"{stage_dir.name} produced no checkpoint with a finite {metric_name}")
    metric, _, path = min(candidates)
    return path, metric


def best_tokenizer_checkpoint(stage_dir):
    return _best_checkpoint(stage_dir, "test_reconst_mse",
                            "ep-{epoch:04d}_mse-{test_reconst_mse:.3f}.ckpt")


def best_policy_checkpoint(stage_dir):
    return _best_checkpoint(stage_dir, "test_reconst_mse",
                            "ep-{epoch:04d}_mse-{test_reconst_mse:.6f}.ckpt")


def stage_command(stage, task, variant, output_dir, num_gpus, tokenizer=None,
                  smoke=False, tokenizer_epochs=5001, policy_epochs=251):
    if stage not in ("tokenizer", "policy"):
        raise ValueError(f"Unknown stage: {stage}")
    if num_gpus not in (1, 2, 4, 8):
        raise ValueError("GPU count must be 1, 2, 4, or 8")
    common = [
        f"seed={SEED}", f"training.num_demo={DATASETS[task][1]}",
        "training.resume=false", "logging.mode=offline", "logging.resume=false",
        "logging.project=real_robot", f"logging.group={task}_{variant}",
        "training.tqdm_interval_sec=30", "val_dataloader.drop_last=false",
        f"hydra.run.dir={json.dumps(str(output_dir))}",
    ]
    if stage == "tokenizer":
        aug = VARIANTS[variant]
        config = "train_oattok_so3aug"
        overrides = [
            f"task/tokenizer=real_robot/{task}",
            f"tokenizer.action_aug.mode={aug['mode']}",
            f"tokenizer.action_aug.augment_position={str(aug['augment_position']).lower()}",
            "tokenizer.action_aug.max_angle_deg=30.0", "tokenizer.action_aug.p=0.6",
            f"training.num_epochs={tokenizer_epochs}",
            f"dataloader.batch_size={256 // num_gpus}",
            f"val_dataloader.batch_size={256 // num_gpus}",
        ]
    else:
        if tokenizer is None:
            raise ValueError("Policy training requires a frozen tokenizer checkpoint")
        config = "train_past2next_scratch_all500"
        overrides = [
            f"task/policy=real_robot/{task}_with_prev_window",
            f"policy.action_tokenizer.checkpoint={json.dumps(str(tokenizer))}",
            "training.init_checkpoint=null", "task.policy.lazy_eval=true",
            "training.offline_validation_enabled=true",
            "training.offline_validation_reason=held_out_real_robot_episodes",
            f"training.num_epochs={policy_epochs}",
            f"dataloader.batch_size={64 // num_gpus}",
            f"val_dataloader.batch_size={64 // num_gpus}",
            "checkpoint.topk.monitor_key=test_reconst_mse", "checkpoint.topk.mode=min",
            "checkpoint.topk.k=0", "training.checkpoint_every=20",
            "training.snapshot_every=0", "++checkpoint.save_all=true",
            "checkpoint.topk.format_str='ep-{epoch:04d}_mse-{test_reconst_mse:.6f}.ckpt'",
        ]
    if smoke:
        overrides += [
            "training.num_epochs=1", "training.max_train_steps=2",
            "training.max_val_steps=1", "training.max_reconst_steps=1",
            "dataloader.num_workers=0", "dataloader.persistent_workers=false",
            "val_dataloader.num_workers=0", "val_dataloader.persistent_workers=false",
        ]
        if stage == "policy":
            overrides += ["policy.self_past_warmup_steps=0",
                          "policy.self_past_ramp_steps=0", "policy.self_past_p=1.0"]
    return [sys.executable, "-m", "torch.distributed.run", "--standalone",
            f"--nproc_per_node={num_gpus}", str(ROOT / "scripts/run_workspace.py"),
            f"--config-name={config}", *common, *overrides]


def dataset_manifest(task):
    """Validate metadata and record the exact shared split without loading RGB."""
    import numpy as np
    import zarr
    from oat.common.seq_sampler import get_val_mask

    path, expected_episodes = DATASETS[task]
    group = zarr.open_group(str(path), mode="r")
    episode_ends = np.asarray(group["meta/episode_ends"][:], dtype=np.int64)
    if (episode_ends.ndim != 1 or len(episode_ends) != expected_episodes
            or np.any(np.diff(np.r_[0, episode_ends]) <= 0)):
        raise ValueError(f"Unexpected episode boundaries in {path}")
    frames = int(episode_ends[-1])
    arrays = {}
    for key, shape in {"action": (7,), **OBS_SHAPES}.items():
        array = group[f"data/{key}"]
        if tuple(array.shape) != (frames, *shape):
            raise ValueError(f"{key} shape {array.shape} != {(frames, *shape)}")
        if key.endswith("_rgb") and array.dtype != np.uint8:
            raise ValueError(f"{key} must contain uint8 RGB pixels")
        arrays[key] = {"shape": list(array.shape), "dtype": str(array.dtype)}
    uids = np.unique(group["data/task_uid"][:]).tolist()
    if uids != [0]:
        raise ValueError(f"Expected one task_uid=0 for {task}, found {uids}")
    val_mask = get_val_mask(expected_episodes, VAL_RATIO, SEED)
    lengths = np.diff(np.r_[0, episode_ends])
    return {
        "path": str(path), "episodes": expected_episodes, "frames": frames,
        "arrays": arrays, "task_uids": uids,
        "split": {
            "seed": SEED, "val_ratio": VAL_RATIO, "episode_id_base": 0,
            "train_episode_ids": np.flatnonzero(~val_mask).tolist(),
            "validation_episode_ids": np.flatnonzero(val_mask).tolist(),
            "train_frames": int(lengths[~val_mask].sum()),
            "validation_frames": int(lengths[val_mask].sum()),
            "episode_ends": episode_ends.tolist(),
            "normalizer_fit": "training_episodes_only",
            "shared_between": "tokenizer_and_policy_and_both_augmentation_variants",
        },
    }


def run_manifest(args):
    output = args.output_dir.resolve()
    num_gpus = len(args.gpus.split(","))
    commands = {
        stage: stage_command(
            stage, args.task, args.variant, output / stage, num_gpus,
            output / "frozen_tokenizer.ckpt", args.smoke,
            args.tokenizer_epochs, args.policy_epochs)
        for stage in ("tokenizer", "policy")
    }
    source_paths = [
        "scripts/train_real_robot.py", "scripts/run_workspace.py",
        "scripts/check_real_robot_checkpoint.py",
        "oat/config/train_oattok_so3aug.yaml",
        "oat/config/train_past2next_scratch_all500.yaml",
        f"oat/config/task/tokenizer/real_robot/{args.task}.yaml",
        f"oat/config/task/policy/real_robot/{args.task}_with_prev_window.yaml",
        "oat/dataset/real_robot_dataset.py",
        "oat/tokenizer/oat/augment/so3_action_chunk_aug.py",
        "oat/tokenizer/oat/tokenizer_so3_aug.py",
        "oat/workspace/train_policy.py", "oat/workspace/train_oattok.py",
        "oat/policy/past2next.py", "oat/policy/past2next_self_past.py",
    ]
    versions = {}
    for name in ("torch", "torchvision", "zarr", "numpy", "hydra-core",
                 "accelerate", "robomimic", "wandb"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "schema_version": 1, "task": args.task, "variant": args.variant,
        "policy_config": "train_past2next_scratch_all500",
        "output_dir": str(output), "gpus": args.gpus, "smoke": args.smoke,
        "tokenizer_epochs": 1 if args.smoke else args.tokenizer_epochs,
        "policy_epochs": 1 if args.smoke else args.policy_epochs,
        "augmentation": {**VARIANTS[args.variant], "max_angle_deg": 30.0, "p": 0.6},
        "action_semantics": ACTION_SEMANTICS,
        "dataset": dataset_manifest(args.task),
        "global_batch_sizes": {"tokenizer": 256, "policy": 64},
        "python": sys.executable, "package_versions": versions,
        "source_sha256": {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in source_paths
        },
        "commands": commands,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=DATASETS, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--tokenizer-epochs", type=int, default=5001)
    parser.add_argument("--policy-epochs", type=int, default=251)
    parser.add_argument("--smoke", action="store_true",
                        help="One epoch, two training/one validation batch per stage")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the verified dataset split and commands; do not train")
    args = parser.parse_args()
    gpu_ids = args.gpus.split(",")
    if (len(gpu_ids) not in (1, 2, 4, 8) or len(set(gpu_ids)) != len(gpu_ids)
            or any(not value.isdigit() for value in gpu_ids)):
        parser.error("--gpus must list 1, 2, 4, or 8 distinct nonnegative GPU indices")
    if args.tokenizer_epochs < 1 or args.policy_epochs < 1:
        parser.error("Epoch counts must be positive")
    manifest = run_manifest(args)
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return

    output = args.output_dir.resolve()
    # Never silently overwrite or resume another task/variant's training.
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    status = {"started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "state": "starting", "task": args.task, "variant": args.variant}

    def save_status():
        status["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        temporary = output / "status.tmp.json"
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(output / "status.json")

    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpus, PYTHONUNBUFFERED="1", WANDB_MODE="offline",
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1",
               HYDRA_FULL_ERROR="1")
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"]
                                   if env.get("PYTHONPATH") else "")
    try:
        for stage in ("tokenizer", "policy"):
            stage_dir = output / stage
            command = manifest["commands"][stage]
            status.update(state=f"training_{stage}")
            save_status()
            print(f"Starting {args.task}/{args.variant} {stage}: {json.dumps(command)}", flush=True)
            with (output / f"{stage}.log").open("w") as logfile:
                subprocess.run(command, cwd=ROOT, env=env, stdout=logfile,
                               stderr=subprocess.STDOUT, check=True)
            if stage == "tokenizer":
                selected, metric = best_tokenizer_checkpoint(stage_dir)
                frozen = output / "frozen_tokenizer.ckpt"
                shutil.copy2(selected, frozen)
                status.update(tokenizer_source=str(selected), tokenizer_mse=metric,
                              frozen_tokenizer=str(frozen))
                save_status()
                print(f"Freezing tokenizer {selected} (validation MSE {metric:.8f})", flush=True)
            else:
                selected, metric = best_policy_checkpoint(stage_dir)
                status.update(state="verifying_checkpoint", best_policy=str(selected),
                              policy_mse=metric)
                save_status()
                check_path = output / "checkpoint_check.json"
                command = [sys.executable, str(ROOT / "scripts/check_real_robot_checkpoint.py"),
                           "--checkpoint", str(selected), "--device", "cuda:0",
                           "--output", str(check_path)]
                with (output / "checkpoint_check.log").open("w") as logfile:
                    subprocess.run(command, cwd=ROOT, env=env, stdout=logfile,
                                   stderr=subprocess.STDOUT, check=True)
                status["checkpoint_check"] = str(check_path)
        status["state"] = "completed"
    except BaseException as exc:
        status.update(state="failed", error=str(exc))
        raise
    finally:
        save_status()


if __name__ == "__main__":
    main()
