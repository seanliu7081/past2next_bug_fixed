#!/usr/bin/env python
"""Explicit new-variant launcher. Dry-run never constructs a model or starts training."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CONFIG_NAMES = {
    ("p2n_new", "libero"): "train_p2n_new",
    ("p2n_state_gate_new", "libero"): "train_p2n_state_gate_new",
    ("p2n_new", "real_robot"): "experimental/train_p2n_new_real_robot",
    ("p2n_state_gate_new", "real_robot"): "experimental/train_p2n_state_gate_new_real_robot",
}


def compose_config(variant, task, overrides=()):
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=str(ROOT / "oat/config"), version_base=None):
        cfg = compose(config_name=CONFIG_NAMES[variant, task], overrides=list(overrides))
    validate_config(cfg, variant, task)
    return cfg


def validate_config(cfg, variant, task):
    gate = variant == "p2n_state_gate_new"
    target = f"oat.policy.{variant}.{'P2NStateGateNewPolicy' if gate else 'P2NNewPolicy'}"
    if cfg.variant != variant or cfg.policy.variant != variant or cfg.policy._target_ != target:
        raise ValueError("Selected variant, policy target and explicit variant must agree")
    if cfg.task_type != task or cfg.policy.task != task:
        raise ValueError("Selected task and policy task must agree")
    if cfg.training.get("init_checkpoint"):
        raise ValueError("Old policy initialization is incompatible; use fresh initialization or explicit resume")
    if cfg.training.resume and not cfg.training.get("resume_checkpoint"):
        raise ValueError("Resume requires --resume CHECKPOINT or training.resume_checkpoint")
    if cfg.task.policy.dataset.history_padding != "zero" or cfg.task.policy.dataset.return_history_validity is not True:
        raise ValueError("Both variants require zero history padding and explicit validity")
    dataset_target = cfg.task.policy.dataset._target_
    expected_dataset = {
        (False, "libero"): "oat.dataset.zarr_dataset_with_prev_window.ZarrDatasetWithPrevWindow",
        (True, "libero"): "oat.dataset.zarr_dataset_with_state_history.ZarrDatasetWithStateHistory",
        (False, "real_robot"): "oat.dataset.real_robot_dataset.RealRobotZarrDatasetWithPrevWindow",
        (True, "real_robot"): "oat.dataset.real_robot_state_history.RealRobotZarrDatasetWithStateHistory",
    }[gate, task]
    if dataset_target != expected_dataset:
        raise ValueError(f"Variant/task requires dataset {expected_dataset}")
    if not gate and any(key.startswith("history_") or key.startswith("state_history") for key in cfg.policy):
        raise ValueError("p2n_new must not construct state history or gate modules")
    if gate and (cfg.policy.state_history_steps != cfg.past_n + 1 or cfg.policy.history_summary_tokens != 4):
        raise ValueError("Gate recipe requires past_n + 1 measured states and four summaries")
    if task == "real_robot":
        if cfg.task.policy.env_runner is not None or not cfg.task.policy.lazy_eval:
            raise ValueError("Real robot uses offline validation and no simulator runner")
    else:
        expected_runner = "P2NStateGateNewLiberoRunner" if gate else "P2NNewLiberoRunner"
        if not cfg.task.policy.env_runner._target_.endswith("." + expected_runner):
            raise ValueError("LIBERO runner does not match execution/state-history capabilities")
    for key in ("num_epochs", "gradient_accumulate_every", "checkpoint_every", "val_every", "sample_every"):
        if int(cfg.training[key]) < 1:
            raise ValueError(f"training.{key} must be positive")


def inspect_dataset(cfg):
    """Read Zarr schema and episode metadata without copying image arrays into RAM."""
    import numpy as np
    import zarr
    from oat.common.seq_sampler import get_val_mask, downsample_mask
    ds = cfg.task.policy.dataset
    path = Path(ds.zarr_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Dataset is missing: {path}")
    root = zarr.open(str(path), mode="r")
    arrays = root["data"]
    ends = np.asarray(root["meta"]["episode_ends"][:])
    if ends.ndim != 1 or not len(ends) or (np.diff(np.r_[0, ends]) <= 0).any():
        raise ValueError("Dataset episode boundaries must be increasing")
    action = arrays[ds.action_key]
    if tuple(action.shape[1:]) != tuple(cfg.shape_meta.action.shape) or action.shape[0] != ends[-1]:
        raise ValueError("Dataset action shape/episode count does not match policy schema")
    for key, spec in cfg.shape_meta.obs.items():
        if key not in ds.obs_keys or key not in arrays:
            raise ValueError(f"Dataset is missing required observation {key}")
        value = arrays[key]
        if tuple(value.shape[1:]) != tuple(spec.shape) or value.shape[0] != ends[-1]:
            raise ValueError(f"Dataset observation schema mismatch: {key}")
        if spec.type == "rgb" and value.dtype != np.dtype("uint8"):
            raise ValueError(f"RGB port {key} must be uint8 byte-range RGB")
    if int(cfg.training.num_demo) != len(ends):
        raise ValueError(f"training.num_demo={cfg.training.num_demo} differs from {len(ends)} actual episodes")
    held_out = get_val_mask(len(ends), float(ds.val_ratio), int(ds.seed))
    train = downsample_mask(~held_out, ds.get("max_train_episodes"), int(ds.seed))
    # Existing validation views use the complement of the actual training mask.
    validation = ~train
    if cfg.training.offline_validation_enabled and not validation.any():
        raise ValueError("Offline validation is enabled but the actual held-out split is empty")
    lengths = np.diff(np.r_[0, ends])
    return {"schema_version": 1, "data_path": str(path), "val_ratio": float(ds.val_ratio),
            "seed": int(ds.seed), "train_episode_ids": np.flatnonzero(train).tolist(),
            "validation_episode_ids": np.flatnonzero(validation).tolist(),
            "train": {"episodes": int(train.sum()), "windows": int(lengths[train].sum())},
            "validation": {"episodes": int(validation.sum()), "windows": int(lengths[validation].sum())},
            "offline_validation_enabled": bool(cfg.training.offline_validation_enabled)}


def validate_tokenizer_source(cfg):
    import dill
    import torch
    path = cfg.policy.tokenizer_checkpoint
    if not path or not Path(path).is_file():
        raise FileNotFoundError(f"Fresh training requires a task-matched tokenizer checkpoint: {path}")
    payload = torch.load(path, map_location="cpu", pickle_module=dill, weights_only=False)
    saved = payload.get("cfg")
    if saved is None or "tokenizer" not in saved.get("task", {}):
        raise ValueError("Tokenizer checkpoint lacks task/normalizer provenance metadata")
    source = saved.task.tokenizer.dataset
    target = cfg.task.policy.dataset
    for key in ("zarr_path", "action_key", "n_action_steps", "seed", "val_ratio", "max_train_episodes"):
        if source.get(key) != target.get(key):
            raise ValueError(f"Tokenizer action/normalizer source mismatch: dataset.{key}")
    if "ema_model" not in payload.get("state_dicts", {}):
        raise ValueError("Fresh training requires the specified tokenizer EMA weights")


def preflight(cfg, output, world_size=2):
    import copy
    import dill
    import hydra
    import torch
    from accelerate.data_loader import BatchSamplerShard
    from oat.common.p2n_new_capabilities import resolve_update_schedule
    from oat.workspace.train_p2n_new import TrainP2NNewWorkspace
    if output.exists() and any(output.iterdir()) and not cfg.training.resume:
        raise ValueError(f"Fresh output directory must be empty: {output}")
    split = inspect_dataset(cfg)
    if cfg.training.resume:
        payload = torch.load(cfg.training.resume_checkpoint, map_location="cpu", pickle_module=dill, weights_only=False)
        TrainP2NNewWorkspace.validate_resume_payload(payload, cfg)
        policy_cfg = payload["policy_config"]
        model = hydra.utils.instantiate(policy_cfg)
        model.load_state_dict(payload["state_dicts"]["model"], strict=True)
    else:
        path = cfg.policy.dino_path
        if not path or not Path(path).is_dir():
            raise FileNotFoundError(f"Fresh training requires a local pinned DINOv3-S/16 snapshot: {path}")
        validate_tokenizer_source(cfg)
        model = hydra.utils.instantiate(cfg.policy)
    optimizer = model.get_optimizer(**cfg.optimizer)
    batch_sampler = torch.utils.data.BatchSampler(
        torch.utils.data.SequentialSampler(range(split["train"]["windows"])),
        batch_size=int(cfg.dataloader.batch_size), drop_last=bool(cfg.dataloader.drop_last))
    batches = len(BatchSamplerShard(batch_sampler, num_processes=world_size, process_index=0))
    schedule = resolve_update_schedule(batches, cfg.training.num_epochs, cfg.training.gradient_accumulate_every,
        max_train_steps=cfg.training.max_train_steps, warmup_steps=cfg.training.lr_warmup_steps,
        warmup_ratio=cfg.training.lr_warmup_ratio)
    worker = TrainP2NNewWorkspace(cfg, output_dir=str(output))
    worker.model, worker.optimizer = model, optimizer
    worker.dataset_split, worker.update_schedule = split, schedule
    report = worker.training_report(model)
    report["effective_batch"] = int(cfg.dataloader.batch_size) * world_size * int(cfg.training.gradient_accumulate_every)
    report["world_size"] = world_size
    report["validation_batch_per_rank"] = int(cfg.val_dataloader.batch_size)
    report["self_past_chunk_size"] = int(cfg.policy.self_past_chunk_size)
    report["status"] = "CPU construction/schema checks complete; GPU peak and DDP smoke still required"
    print(json.dumps(report, indent=2))
    return report


def main(argv=None):
    from omegaconf import OmegaConf
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--worker-config":
        parser = argparse.ArgumentParser()
        parser.add_argument("--worker-config", required=True)
        parser.add_argument("--output", required=True)
        args = parser.parse_args(argv)
        from oat.workspace.train_p2n_new import TrainP2NNewWorkspace
        cfg = OmegaConf.load(args.worker_config)
        TrainP2NNewWorkspace(cfg, output_dir=args.output).run()
        return
    overrides = []
    if "--" in argv:
        index = argv.index("--")
        overrides, argv = argv[index + 1:], argv[:index]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("p2n_new", "p2n_state_gate_new"), required=True)
    parser.add_argument("--task", choices=("libero", "real_robot"), required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--dino")
    parser.add_argument("--dino-revision")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--devices", help="CUDA_VISIBLE_DEVICES, e.g. 2,3; existing jobs are untouched")
    parser.add_argument("--num-processes", type=int, default=2)
    parser.add_argument("--cpu", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Resolve all config entries without data/model loading")
    mode.add_argument("--preflight", action="store_true", help="Validate local weights/schema and count instantiated CPU parameters; do not train")
    args = parser.parse_args(argv)
    if args.num_processes < 1:
        parser.error("--num-processes must be positive")
    generated = []
    for value, key in ((args.tokenizer, "policy.tokenizer_checkpoint"), (args.dino, "policy.dino_path"),
                       (args.dino_revision, "policy.dino_revision")):
        if value is not None:
            generated.append(f"{key}={json.dumps(str(value))}")
    if args.resume:
        generated.extend(["training.resume=true", f"training.resume_checkpoint={json.dumps(str(args.resume.resolve()))}"])
    cfg = compose_config(args.variant, args.task, [*generated, *overrides])
    output = (args.output or ROOT / "output/training" / f"{args.variant}_{args.task}_seed{cfg.seed}").resolve()
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print(json.dumps({"output": str(output), "world_size": args.num_processes,
                      "effective_batch": int(cfg.dataloader.batch_size) * args.num_processes * int(cfg.training.gradient_accumulate_every),
                      "mode": "dry_run" if args.dry_run else "preflight" if args.preflight else "training"}, indent=2))
    if args.dry_run:
        print("Configuration resolved only: weights, parameter counts, dataset split and GPU memory are unverified.")
        return
    if args.devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    if args.cpu:
        os.environ["ACCELERATE_USE_CPU"] = "true"
    report = preflight(cfg, output, args.num_processes)
    if args.preflight:
        return
    output.mkdir(parents=True, exist_ok=True)
    resolved = output / "p2n_new_resolved.yaml"
    resolved.write_text(OmegaConf.to_yaml(cfg, resolve=True))
    (output / "p2n_new_preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    command = [sys.executable]
    if args.num_processes > 1:
        command += ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node", str(args.num_processes)]
    command += [str(Path(__file__).resolve()), "--worker-config", str(resolved), "--output", str(output)]
    subprocess.run(command, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
