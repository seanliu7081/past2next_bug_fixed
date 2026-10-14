#!/usr/bin/env python
"""Train SigLIP/projector and the P2N action expert while keeping Gemma frozen.

The KI head stays trainable and supplies the vision gradient through frozen Gemma.
All original P2N-VLA files and launchers are preserved. Hydra overrides follow ``--``.
Example (micro-batch 1 per GPU, effective batch 32, live W&B):
  bash train_p2n_vla_vision_action.sh --finetune vision_action --devices 0,1 \
      --num-processes 2 --lazy-eval=false --eval-every 100 --wandb-mode online \
      --batch-size 1 --gradient-accumulation 16 --num-train-epochs 300
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import train_p2n_vla as legacy  # noqa: E402

VARIANTS = {
    "p2n_vla": {
        "config": "train_p2n_vla_vision_action",
        "target": "oat.policy.p2n_vla_vision_action.P2NVLAVisionActionPolicy",
    },
    "p2n_vla_state_gate": {
        "config": "train_p2n_vla_state_gate_vision_action",
        "target": "oat.policy.p2n_vla_vision_action.P2NVLAStateGateVisionActionPolicy",
    },
}


def validate_config(cfg, variant, task):
    """Apply the established recipe checks and the stricter vision/action contract."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; choose from {sorted(VARIANTS)}")
    expected = VARIANTS[variant]["target"]
    if cfg.policy.get("_target_") != expected:
        raise ValueError(f"Vision/action variant {variant!r} must target {expected}")
    if cfg.policy.get("training_scope") != "vision_action":
        raise ValueError("policy.training_scope must be vision_action (Gemma and its LoRA are frozen)")
    if not isinstance(cfg.policy.get("vision_activation_checkpointing"), bool):
        raise ValueError("policy.vision_activation_checkpointing must be a boolean")
    lambda_ki = float(cfg.policy.get("lambda_ki", 0))
    if not math.isfinite(lambda_ki) or lambda_ki <= 0:
        raise ValueError("Vision training requires policy.lambda_ki > 0: action KV insulation blocks AR gradients")
    # Rollout planning divides by accumulation before the legacy training checks.
    for key in ("num_epochs", "gradient_accumulate_every"):
        legacy._int(cfg.training.get(key), f"training.{key}")
    legacy._int(cfg.training.get("max_train_steps"), "training.max_train_steps", allow_none=True)
    # The existing validator intentionally pins its policy targets. Validate a private
    # translated copy so its dataset, gate, optimizer and rollout checks remain authoritative.
    translated = deepcopy(cfg)
    translated.policy._target_ = legacy.VARIANTS[variant]["target"]
    legacy.validate_config(translated, variant, task)


def compose_config(variant="p2n_vla", task="libero", overrides=(), *, config_dir=None):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; choose from {sorted(VARIANTS)}")
    if task not in legacy.TASKS:
        raise ValueError(f"Unknown task {task!r}; choose from {legacy.TASKS}")
    from hydra import compose, initialize_config_dir
    from oat.common.hydra_util import register_new_resolvers
    register_new_resolvers()
    with initialize_config_dir(config_dir=str(Path(config_dir or legacy.CONFIG_DIR).resolve()), version_base=None):
        cfg = compose(config_name=VARIANTS[variant]["config"], overrides=list(overrides))
    validate_config(cfg, variant, task)
    return cfg


def boolean(value):
    if value.lower() not in ("true", "false"):
        raise argparse.ArgumentTypeError("use true or false")
    return value.lower() == "true"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--finetune", choices=("vision_action",), default="vision_action",
                        help="Train full vision/projector + action expert; freeze Gemma and language LoRA")
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="p2n_vla")
    parser.add_argument("--task", choices=legacy.TASKS, default="libero")
    parser.add_argument("--devices", help="Distinct CUDA GPU indices, e.g. 0,1")
    parser.add_argument("--num-processes", type=int, help="Defaults to the number of --devices, otherwise 2")
    parser.add_argument("--num-train-epochs", type=int)
    parser.add_argument("--batch-size", type=int, help="Training micro-batch per GPU (default 1)")
    parser.add_argument("--val-batch-size", type=int, help="Validation micro-batch per GPU (default 1)")
    parser.add_argument("--gradient-accumulation", type=int, help="Micro-batches per optimizer update (default 16)")
    parser.add_argument("--lazy-eval", "--lazy_eval", dest="lazy_eval", type=boolean,
                        help="false enables in-training LIBERO evaluation (default false)")
    parser.add_argument("--eval-every", type=int, help="Simulator evaluation cadence in completed epochs")
    parser.add_argument("--val-every", type=int, help="Offline validation cadence (zero-based epochs, plus final epoch)")
    parser.add_argument("--eval-parallel-envs", type=int, help="Parallel LIBERO environments per evaluation (default 10)")
    parser.add_argument("--test-num", type=int, help="Total LIBERO episodes per evaluation (default 500)")
    parser.add_argument("--use-ema", type=boolean, help="Maintain EMA weights (default true); false saves GPU memory")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"),
                        help="W&B mode (default online, including probes)")
    parser.add_argument("--checkpoint-every", type=int, help="Resume checkpoint cadence in epochs")
    parser.add_argument("--snapshot-every", type=int, help="EMA snapshot cadence in optimizer steps; 0 disables periodic saves")
    parser.add_argument("--tokenizer", help="Frozen OAT checkpoint")
    parser.add_argument("--pi05", help="pi05_base model.safetensors")
    parser.add_argument("--spm", help="PaliGemma SentencePiece model")
    parser.add_argument("--output", type=Path, help="Fresh runs require an empty directory")
    parser.add_argument("--resume", type=Path, help="Resume a vision/action run with matching world size")
    parser.add_argument("--cpu", action="store_true", help="CPU/gloo smoke runs")
    parser.add_argument("--allow-busy-gpus", action="store_true")
    parser.add_argument("--full-hash", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--probe-steps", type=int, default=3)
    parser.add_argument("--probe-self-past-step", type=int)
    parser.add_argument("--config-dir", type=Path, help=argparse.SUPPRESS)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Validate configuration without reading assets or GPUs")
    mode.add_argument("--preflight", action="store_true", help="Full CPU preflight including model build")
    mode.add_argument("--probe", action="store_true", help="Bounded optimizer/memory/rollout probe")
    return parser


def config_from_args(args, overrides=()):
    generated = []
    mapping = (
        (args.num_train_epochs, "training.num_epochs"),
        (args.batch_size, "dataloader.batch_size"),
        (args.val_batch_size, "val_dataloader.batch_size"),
        (args.gradient_accumulation, "training.gradient_accumulate_every"),
        (args.lazy_eval, "task.policy.lazy_eval"),
        (args.eval_every, "training.rollout_every"),
        (args.val_every, "training.val_every"),
        (args.eval_parallel_envs, "task.policy.env_runner.n_parallel_envs"),
        (args.test_num, "task.policy.env_runner.n_test"),
        (args.use_ema, "training.use_ema"),
        (args.wandb_mode, "logging.mode"),
        (args.checkpoint_every, "training.checkpoint_every"),
        (args.snapshot_every, "training.snapshot_every"),
    )
    for value, key in mapping:
        if value is not None:
            generated.append(f"{key}={json.dumps(value)}")
    for value, key in ((args.tokenizer, "policy.tokenizer_checkpoint"), (args.pi05, "policy.pi05_weights"),
                       (args.spm, "policy.spm_path")):
        if value is not None:
            generated.append(f"{key}={json.dumps(str(Path(value).expanduser().resolve()))}")
    if args.resume:
        generated += ["training.resume=true",
                      f"training.resume_checkpoint={json.dumps(str(args.resume.expanduser().resolve()))}"]
    if args.probe:
        generated += ["training.probe.enabled=true", f"training.probe.optimizer_steps={args.probe_steps}"]
        if args.probe_self_past_step is not None:
            generated.append(f"training.probe.self_past_step={args.probe_self_past_step}")
    return compose_config(args.variant, args.task, [*generated, *overrides], config_dir=args.config_dir)


def worker_environment(cfg):
    """Make the resolved logging flag authoritative over stale shell W&B settings."""
    env = dict(os.environ)
    env["WANDB_MODE"] = str(cfg.logging.mode)
    if cfg.logging.mode == "online":
        env.pop("WANDB_DISABLED", None)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("HF_HOME", "/workspace/.hf_home")
    return env


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--worker-config":
        return legacy._worker(argv)
    overrides = []
    if "--" in argv:
        index = argv.index("--")
        argv, overrides = argv[:index], argv[index + 1:]
    parser = build_parser()
    args = parser.parse_args(argv)
    devices = None
    if args.devices:
        devices = [item.strip() for item in args.devices.split(",")]
        if any(not item.isdigit() for item in devices) or len(set(devices)) != len(devices):
            parser.error("--devices must list distinct integer GPU indices, e.g. 0,1")
    world_size = args.num_processes if args.num_processes is not None else (len(devices) if devices else 2)
    if world_size < 1:
        parser.error("--num-processes must be positive")
    if devices and world_size != len(devices):
        parser.error("--num-processes must equal the number of --devices")
    if args.probe and args.resume:
        parser.error("--probe always starts fresh")
    if args.probe_steps < 1:
        parser.error("--probe-steps must be positive")
    cfg = config_from_args(args, overrides)
    from omegaconf import OmegaConf
    if args.output is not None:
        output = args.output.expanduser().resolve()
    elif args.resume:
        checkpoint = args.resume.expanduser().resolve()
        output = checkpoint.parent.parent if checkpoint.parent.name == "checkpoints" else checkpoint.parent
    elif args.probe:
        output = ROOT / "output/probe" / f"{args.variant}_vision_action_{args.task}_{datetime.now():%Y%m%d_%H%M%S}"
    else:
        output = ROOT / "output/training" / f"{args.variant}_vision_action_{args.task}_seed{cfg.seed}"
    mode = "dry_run" if args.dry_run else "preflight" if args.preflight else "probe" if args.probe else "train"
    summary = legacy.describe(cfg, world_size, output, mode)
    summary.update(training_scope="vision_action", language_trainable=False, language_lora_trainable=False,
                   vision_trainable=True, vision_activation_checkpointing=cfg.policy.vision_activation_checkpointing)
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print(json.dumps(summary, indent=2, default=str))
    if args.dry_run:
        print("Configuration validated only: data, weights and GPU memory are unverified.")
        return summary
    if devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    if args.cpu:
        os.environ["ACCELERATE_USE_CPU"] = "true"
    report = legacy.preflight(cfg, output, world_size, build_model=args.preflight, full_hash=args.full_hash)
    report["training_scope"] = "vision_action"
    print(json.dumps(report, indent=2, default=str))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, default=str) + "\n")
    if args.preflight:
        return report
    if not args.cpu and devices:
        legacy.check_gpu_selection(devices, args.allow_busy_gpus)
    output.mkdir(parents=True, exist_ok=True)
    suffix = f"_resume_{datetime.now():%Y%m%d_%H%M%S}" if args.resume else ""
    resolved = output / f"p2n_vla_vision_action_resolved{suffix}.yaml"
    resolved.write_text(OmegaConf.to_yaml(cfg, resolve=True))
    (output / f"p2n_vla_vision_action_preflight{suffix}.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    command = [sys.executable]
    if world_size > 1:
        command += ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node", str(world_size)]
    command += [str(Path(__file__).resolve()), "--worker-config", str(resolved), "--output", str(output)]
    print("Launching:", " ".join(command), flush=True)
    subprocess.run(command, cwd=str(ROOT), check=True, env=worker_environment(cfg))
    if args.probe and (output / "probe.json").is_file():
        print((output / "probe.json").read_text())
    return report


if __name__ == "__main__":
    main()
