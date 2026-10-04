#!/usr/bin/env python3
"""Train modern Past2Next with the original fused RGB/state observation encoder.

Flags accept either '--name value' or '--name=value'. Additional Hydra overrides
follow '--' and take precedence over flags. Dry-run only resolves configuration.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import train_p2n_new as legacy
from scripts.train_p2n_new_convnext import check_gpu_idle
from scripts.train_p2n_new_convnext_libero10 import configure_training_devices, inspect_simulator

CONFIG_NAMES = {
    (variant, task): f"experimental/train_{variant}_original_obs_{task}"
    for variant in ("p2n_new", "p2n_state_gate_new")
    for task in ("libero", "real_robot")
}


def compose_config(variant, task, overrides=()):
    from hydra import compose, initialize_config_dir
    from oat.common.hydra_util import register_new_resolvers
    if (variant, task) not in CONFIG_NAMES:
        raise ValueError("Supported variants are p2n_new/p2n_state_gate_new; tasks are libero/real_robot")
    register_new_resolvers()
    with initialize_config_dir(config_dir=str(ROOT / "oat/config"), version_base=None):
        cfg = compose(config_name=CONFIG_NAMES[variant, task], overrides=list(overrides))
    validate_config(cfg, variant, task)
    return cfg


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def validate_config(cfg, variant, task):
    gate = variant == "p2n_state_gate_new"
    class_name = "P2NStateGateNewOriginalObsPolicy" if gate else "P2NNewOriginalObsPolicy"
    if cfg.policy._target_ != f"oat.policy.p2n_new_original_obs.{class_name}":
        raise ValueError("Selected variant must use its original-fused policy target")
    if cfg._target_ != "oat.workspace.train_p2n_new_original_obs.TrainP2NNewOriginalObsWorkspace":
        raise ValueError("Original observations require the normalizer-aware original-fused workspace")
    # Keep the original dataset/history/runner contract without importing its
    # DINO model. Real-robot evaluation here is offline, regardless of lazy_eval.
    original_schema = copy.deepcopy(cfg)
    original_class = "P2NStateGateNewPolicy" if gate else "P2NNewPolicy"
    original_schema.policy._target_ = f"oat.policy.{variant}.{original_class}"
    if task == "real_robot":
        original_schema.task.policy.lazy_eval = True
    legacy.validate_config(original_schema, variant, task)
    forbidden = ("dino_path", "dino_revision", "convnext_path", "convnext_model_name", "convnext_frozen",
                 "num_visual_queries", "num_visual_tokens", "resampler_depth", "visual_resampler_dim",
                 "visual_resampler_heads", "visual_resampler_ffn_dim", "vision_feature_stages",
                 "vision_image_size", "image_brightness", "image_contrast", "rgb_range")
    for key in forbidden:
        if key in cfg.policy:
            raise ValueError(f"Original-fused schema must not include policy.{key}")
    required = dict(obs_encoder_type="original_fused", context_layout="original_fused_v1",
                    context_schema_version=2, embed_dim=768, n_layers=16, n_heads=12,
                    ffn_dim=2048, dropout=0.1, expected_action_tokens=8,
                    n_obs_steps=2, n_action_steps=8, horizon=16, past_n=7)
    for key, expected in required.items():
        if cfg.policy.get(key) != expected:
            raise ValueError(f"Original-fused production contract requires policy.{key}={expected!r}")
    original_obs = cfg.policy.original_obs_config
    fixed = dict(eval_fixed_crop=True, use_group_norm=True, share_rgb_model=False,
                 pretrained=False, state_out_dim=None, feature_dimension=64,
                 spatial_softmax_num_kp=32, spatial_softmax_temperature=1.0, spatial_softmax_noise=0.0)
    for key, expected in fixed.items():
        if key not in original_obs or original_obs[key] != expected:
            raise ValueError(f"Original encoder requires original_obs_config.{key}={expected!r}")
    crop = original_obs.get("crop_shape")
    if crop is None or len(crop) != 2 or any(not isinstance(v, int) or isinstance(v, bool) or v < 1 or v >= 128 for v in crop):
        raise ValueError("original_obs_config.crop_shape must contain two integer sizes in [1,127]")
    rgb = [spec for spec in cfg.shape_meta.obs.values() if spec.type == "rgb"]
    if len(rgb) != 2 or any(list(spec.shape) != [128, 128, 3] for spec in rgb):
        raise ValueError("Current task recipes require two 128x128 RGB cameras")
    if any(spec.type not in ("rgb", "state") for spec in cfg.shape_meta.obs.values()):
        raise ValueError("Original-fused observations support only RGB and state ports")
    if "task_uid" not in cfg.shape_meta.obs or list(cfg.shape_meta.action.shape) != [7]:
        raise ValueError("Current task recipes require task_uid and seven-dimensional actions")
    if not isinstance(cfg.task.policy.lazy_eval, bool):
        raise ValueError("task.policy.lazy_eval must be true or false")
    for name, value in (("dataloader.batch_size", cfg.dataloader.batch_size),
                        ("val_dataloader.batch_size", cfg.val_dataloader.batch_size),
                        ("training.rollout_every", cfg.training.rollout_every)):
        _positive_int(value, name)
    limit = cfg.training.validation_max_samples
    if limit is not None:
        _positive_int(limit, "training.validation_max_samples")
    for key in ("dataloader", "val_dataloader"):
        loader = cfg[key]
        if isinstance(loader.num_workers, bool) or not isinstance(loader.num_workers, int) or loader.num_workers < 0:
            raise ValueError(f"{key}.num_workers must be a nonnegative integer")
        if loader.num_workers == 0 and loader.persistent_workers:
            raise ValueError(f"{key}.persistent_workers must be false when num_workers=0")
    if task == "libero":
        runner = cfg.task.policy.env_runner
        if cfg.task.policy.name != "libero10" or runner.task_name != "libero10":
            raise ValueError("The supplied LIBERO recipe supports the LIBERO-10 suite")
        for key in ("n_test", "n_parallel_envs"):
            _positive_int(runner[key], f"task.policy.env_runner.{key}")
        if not isinstance(runner.n_test_vis, int) or not 0 <= runner.n_test_vis <= runner.n_test:
            raise ValueError("Evaluation videos must be between zero and n_test")


def evaluation_summary(cfg):
    if cfg.task_type == "real_robot":
        return {"kind": "offline", "lazy_eval": cfg.task.policy.lazy_eval,
                "validation_max_samples": cfg.training.validation_max_samples,
                "physical_robot_rollout": False,
                "description": "Held-out recorded robot data only; no automatic physical robot rollout"}
    return {"kind": "libero", "lazy_eval": cfg.task.policy.lazy_eval,
            "episodes_per_evaluation": int(cfg.task.policy.env_runner.n_test),
            "episodes_scope": "total across all 10 tasks, not per task",
            "rollout_every": int(cfg.training.rollout_every)}


def inspect_dataset(cfg):
    """Metadata, action fingerprint and a raw training observation; no image copy."""
    import hashlib
    import numpy as np
    import zarr
    split = legacy.inspect_dataset(cfg)
    root = zarr.open(str(Path(cfg.task.policy.dataset.zarr_path).expanduser()), mode="r")
    ends = np.asarray(root["meta"]["episode_ends"][:])
    digest = hashlib.sha256()
    digest.update(ends.tobytes())
    action = root["data"][cfg.task.policy.dataset.action_key]
    for start in range(0, len(action), 65536):
        digest.update(np.asarray(action[start:start + 65536]).tobytes())
    split["identity"] = {
        "episode_and_action_sha256": digest.hexdigest(),
        "train_episode_ids": split["train_episode_ids"],
        "validation_episode_ids": split["validation_episode_ids"],
    }
    if not split["train_episode_ids"]:
        raise ValueError("Training episode split must be nonempty")
    episode = split["train_episode_ids"][0]
    start = int(ends[episode - 1]) if episode else 0
    stop = min(start + int(cfg.n_obs_steps), int(ends[episode]))
    observed = {}
    for key in cfg.shape_meta.obs:
        sample = np.asarray(root["data"][key][start:stop])
        if not np.isfinite(sample).all():
            raise ValueError(f"Dataset training sample has nonfinite observation values: {key}")
        observed[key] = {"shape": list(sample.shape), "dtype": str(sample.dtype)}
    split["raw_training_observation_sample"] = observed
    return split


def preflight(cfg, output, world_size=2):
    """Read metadata and construct on CPU; resume never fits normalizers."""
    import dill
    import hydra
    import torch
    from accelerate.data_loader import BatchSamplerShard
    from oat.common.p2n_new_capabilities import resolve_update_schedule
    from oat.workspace.train_p2n_new_original_obs import TrainP2NNewOriginalObsWorkspace

    output = Path(output)
    if output.exists() and any(output.iterdir()) and not cfg.training.resume:
        raise ValueError(f"Fresh output directory must be empty: {output}")
    split = inspect_dataset(cfg)
    if cfg.training.resume:
        payload = torch.load(cfg.training.resume_checkpoint, map_location="cpu", pickle_module=dill, weights_only=False)
        TrainP2NNewOriginalObsWorkspace.validate_resume_payload(payload, cfg)
        saved_split = payload.get("metadata", {}).get("dataset_split")
        if saved_split is None or saved_split.get("identity") != split["identity"]:
            raise ValueError("Resume dataset split or episode/action fingerprint differs")
        model = hydra.utils.instantiate(payload["policy_config"])
        model.load_state_dict(payload["state_dicts"]["model"], strict=True)
    else:
        legacy.validate_tokenizer_source(cfg)
        model = hydra.utils.instantiate(cfg.policy)
    if model.max_seq_len != 8:
        raise ValueError(f"Selected OAT must have exactly eight action tokens, got {model.max_seq_len}")
    optimizer = model.get_optimizer(**cfg.optimizer)
    sampler = torch.utils.data.BatchSampler(
        torch.utils.data.SequentialSampler(range(split["train"]["windows"])),
        batch_size=int(cfg.dataloader.batch_size), drop_last=bool(cfg.dataloader.drop_last))
    batches = len(BatchSamplerShard(sampler, num_processes=world_size, process_index=0))
    schedule = resolve_update_schedule(
        batches, cfg.training.num_epochs, cfg.training.gradient_accumulate_every,
        max_train_steps=cfg.training.max_train_steps, warmup_steps=cfg.training.lr_warmup_steps,
        warmup_ratio=cfg.training.lr_warmup_ratio)
    worker = TrainP2NNewOriginalObsWorkspace(cfg, output_dir=str(output))
    worker.model, worker.optimizer = model, optimizer
    worker.dataset_split, worker.update_schedule = split, schedule
    report = worker.training_report(model)
    report.update({"world_size": world_size, "vision": "original_fused",
                   "effective_batch": int(cfg.dataloader.batch_size) * world_size * int(cfg.training.gradient_accumulate_every),
                   "validation_batch_per_rank": int(cfg.val_dataloader.batch_size),
                   "self_past_chunk_size": int(cfg.policy.self_past_chunk_size),
                   "evaluation": evaluation_summary(cfg),
                   "normalizer_initialization": ("embedded artifact restored; no fitting" if cfg.training.resume
                                                 else "fit from training episodes only when workspace starts"),
                   "status": "CPU source/schema/construction checks complete; GPU/DDP performance unmeasured"})
    if cfg.task_type == "libero":
        report["simulator"] = inspect_simulator(cfg)
    print(json.dumps(report, indent=2))
    return report


def boolean(value):
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in ("true", "1", "yes", "on"):
        return True
    if normalized in ("false", "0", "no", "off"):
        return False
    raise argparse.ArgumentTypeError("Expected true or false")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("p2n_new", "p2n_state_gate_new"), default="p2n_new")
    parser.add_argument("--task", choices=("libero", "real_robot"), required=True,
                        help="libero selects LIBERO-10; real_robot selects Nut Washer N77")
    parser.add_argument("--lazy-eval", "--lazy_eval", type=boolean,
                        help="false enables LIBERO rollouts; real_robot always evaluates recorded held-out data")
    parser.add_argument("--num-train-epochs", "--num_train_epochs", "--num-epochs", "--num_epochs", type=int)
    parser.add_argument("--batch-size", "--batch_size", type=int, help="Training batch per GPU/process")
    parser.add_argument("--val-batch-size", "--val_batch_size", type=int)
    parser.add_argument("--test-num", "--test_num", type=int,
                        help="LIBERO total rollout episodes; real_robot maximum global held-out validation windows")
    parser.add_argument("--eval-every", "--eval_every", type=int, help="LIBERO rollout interval in epochs")
    parser.add_argument("--val-every", "--val_every", type=int, help="Offline validation interval in epochs")
    parser.add_argument("--parallel-envs", "--parallel_envs", type=int, help="LIBERO simulator workers")
    parser.add_argument("--gradient-accumulation", "--gradient_accumulation", type=int)
    parser.add_argument("--num-workers", "--num_workers", type=int, help="Training DataLoader workers")
    parser.add_argument("--val-num-workers", "--val_num_workers", type=int)
    parser.add_argument("--num-demo", "--num_demo", type=int, help="Expected total dataset episodes")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--tokenizer", help="Frozen OAT checkpoint, matched to dataset/split")
    parser.add_argument("--dataset", help="Zarr path")
    parser.add_argument("--devices", help="GPU indices or UUIDs from nvidia-smi, e.g. 0,1")
    parser.add_argument("--num-processes", "--num_processes", type=int, default=2)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--cpu", action="store_true", help="Explicit CPU execution (not for full-size training)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Configuration only; no weights/data/GPU access")
    mode.add_argument("--preflight", action="store_true", help="CPU source/schema/encoder checks; no training")
    return parser


def flag_overrides(args, parser):
    if args.num_processes < 1:
        parser.error("--num-processes must be positive")
    if args.task != "libero" and args.parallel_envs is not None:
        parser.error("--parallel-envs is only meaningful for LIBERO")
    mapping = ((args.lazy_eval, "task.policy.lazy_eval"),
               (args.num_train_epochs, "training.num_epochs"),
               (args.batch_size, "dataloader.batch_size"),
               (args.val_batch_size, "val_dataloader.batch_size"),
               (args.eval_every, "training.rollout_every"),
               (args.val_every, "training.val_every"),
               (args.parallel_envs, "task.policy.env_runner.n_parallel_envs"),
               (args.gradient_accumulation, "training.gradient_accumulate_every"),
               (args.num_workers, "dataloader.num_workers"),
               (args.val_num_workers, "val_dataloader.num_workers"),
               (args.num_demo, "training.num_demo"),
               (args.seed, "seed"),
               (args.tokenizer, "policy.tokenizer_checkpoint"),
               (args.dataset, "task.policy.dataset.zarr_path"))
    generated = [f"{key}={json.dumps(value)}" for value, key in mapping if value is not None]
    if args.num_workers == 0:
        generated.append("dataloader.persistent_workers=false")
    if args.val_num_workers == 0:
        generated.append("val_dataloader.persistent_workers=false")
    if args.test_num is not None:
        key = "task.policy.env_runner.n_test" if args.task == "libero" else "training.validation_max_samples"
        generated.append(f"{key}={args.test_num}")
    if args.resume:
        generated.extend(["training.resume=true",
                          f"training.resume_checkpoint={json.dumps(str(args.resume.expanduser().resolve()))}"])
    return generated


def main(argv=None):
    from omegaconf import OmegaConf
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--worker-config":
        from oat.workspace.train_p2n_new_original_obs import TrainP2NNewOriginalObsWorkspace
        parser = argparse.ArgumentParser()
        parser.add_argument("--worker-config", required=True)
        parser.add_argument("--output", required=True)
        args = parser.parse_args(argv)
        cfg = OmegaConf.load(args.worker_config)
        validate_config(cfg, cfg.variant, cfg.task_type)
        TrainP2NNewOriginalObsWorkspace(cfg, output_dir=args.output).run()
        return
    split = argv.index("--") if "--" in argv else len(argv)
    flags, overrides = argv[:split], argv[split + 1:]
    parser = build_parser()
    args = parser.parse_args(flags)
    cfg = compose_config(args.variant, args.task, [*flag_overrides(args, parser), *overrides])
    output = (args.output or ROOT / "output/training" /
              f"{args.variant}_original_obs_{args.task}_seed{cfg.seed}").expanduser().resolve()
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print(json.dumps({"variant": cfg.variant, "task": cfg.task_type, "vision": "original_fused",
                      "output": str(output), "world_size": args.num_processes,
                      "effective_batch": int(cfg.dataloader.batch_size) * args.num_processes * int(cfg.training.gradient_accumulate_every),
                      "evaluation": evaluation_summary(cfg),
                      "mode": "dry_run" if args.dry_run else "preflight" if args.preflight else "training"}, indent=2))
    if args.dry_run:
        print("Configuration resolved only: no weights, dataset, simulator, GPU or logging initialization.")
        return cfg
    if args.task == "libero":
        os.environ.setdefault("MUJOCO_GL", "osmesa" if args.cpu else "egl")
        if os.environ["MUJOCO_GL"] in ("egl", "osmesa"):
            os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
    if args.cpu:
        os.environ["ACCELERATE_USE_CPU"] = "true"
    elif not args.preflight:
        selected = check_gpu_idle(args.devices, args.num_processes)
        configure_training_devices(selected, render=args.task == "libero" and not cfg.task.policy.lazy_eval)
    report = preflight(cfg, output, args.num_processes)
    if args.preflight:
        return report
    if not args.cpu:
        check_gpu_idle(os.environ["CUDA_VISIBLE_DEVICES"], args.num_processes)
    output.mkdir(parents=True, exist_ok=True)
    resolved = output / "p2n_new_original_obs_resolved.yaml"
    resolved.write_text(OmegaConf.to_yaml(cfg, resolve=True))
    (output / "p2n_new_original_obs_preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    command = [sys.executable]
    if args.num_processes > 1:
        command += ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node", str(args.num_processes)]
    command += [str(Path(__file__).resolve()), "--worker-config", str(resolved), "--output", str(output)]
    subprocess.run(command, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
