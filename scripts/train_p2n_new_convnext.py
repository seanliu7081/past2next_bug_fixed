#!/usr/bin/env python3
"""Additive P2N launcher: default DINO delegates to the unchanged original script.

Nano dry-run is configuration-only. Fresh preflight loads only local, explicitly
selected pretrained weights; training checks GPU availability before launching.
"""
from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import train_p2n_new as legacy

CONFIG_NAMES = {
    ("p2n_new", "real_robot"): "experimental/train_p2n_new_convnext_nano_real_robot",
    ("p2n_state_gate_new", "real_robot"): "experimental/train_p2n_state_gate_new_convnext_nano_real_robot",
}


def compose_config(variant, task, overrides=()):
    from hydra import compose, initialize_config_dir
    from oat.common.hydra_util import register_new_resolvers
    if (variant, task) not in CONFIG_NAMES:
        raise ValueError("convnext_nano supports p2n_new and p2n_state_gate_new on real_robot only")
    register_new_resolvers()
    with initialize_config_dir(config_dir=str(ROOT / "oat/config"), version_base=None):
        cfg = compose(config_name=CONFIG_NAMES[variant, task], overrides=list(overrides))
    validate_config(cfg, variant, task)
    return cfg


def validate_config(cfg, variant, task):
    gate = variant == "p2n_state_gate_new"
    class_name = "P2NStateGateNewPolicy" if gate else "P2NNewPolicy"
    if task != "real_robot":
        raise ValueError("convnext_nano delivery supports real_robot only")
    if cfg.policy._target_ != f"oat.policy.p2n_new_convnext.{class_name}":
        raise ValueError("Selected variant must use its additive ConvNeXt policy class")
    if cfg._target_ != "oat.workspace.train_p2n_new_convnext.TrainP2NNewWorkspace":
        raise ValueError("ConvNeXt requires the additive workspace with encoder resume protection")
    # Reuse every existing task, dataset, gate and training check without altering
    # the original validator or its exact policy-target requirement.
    original_schema = copy.deepcopy(cfg)
    original_schema.policy._target_ = f"oat.policy.{variant}.{class_name}"
    legacy.validate_config(original_schema, variant, task)
    if cfg.policy.get("dino_path") or cfg.policy.get("dino_revision"):
        raise ValueError("ConvNeXt cannot also specify DINO weights or a DINO revision")
    required = {
        "obs_encoder_type": "convnextv2_tokens",
        "convnext_model_name": "convnextv2_nano.fcmae_ft_in22k_in1k",
        "convnext_frozen": True, "vision_image_size": 224,
        "vision_feature_stages": [2, 3], "visual_resampler_dim": 256,
        "visual_resampler_heads": 4, "visual_resampler_ffn_dim": 768,
        "resampler_depth": 2, "num_visual_queries": 64,
        "embed_dim": 768, "n_layers": 16, "n_heads": 12, "ffn_dim": 2048,
        "dropout": 0.1, "expected_action_tokens": 8,
        "rgb_range": "uint8", "image_brightness": 0.1, "image_contrast": 0.1,
        "n_obs_steps": 2, "n_action_steps": 8, "horizon": 16, "past_n": 7,
    }
    for key, expected in required.items():
        if cfg.policy.get(key) != expected:
            raise ValueError(f"ConvNeXt production contract requires policy.{key}={expected!r}")
    rgb_ports = [spec for spec in cfg.shape_meta.obs.values() if spec.type == "rgb"]
    if len(rgb_ports) != 2 or list(cfg.shape_meta.action.shape) != [7]:
        raise ValueError("ConvNeXt real-robot contract requires two cameras and seven-dimensional actions")


def check_gpu_idle(devices=None, required_devices=1, *, allowed_pids=()):
    """Read nvidia-smi only; never terminate or otherwise change existing jobs.

    Numeric selections are nvidia-smi indices. Callers use the returned UUIDs
    to remove ambiguity with CUDA's default device ordering.
    """
    def query(fields, kind):
        result = subprocess.run(
            ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True,
        )
        return [[item.strip() for item in row] for row in csv.reader(io.StringIO(result.stdout)) if row]

    try:
        rows = query("index,uuid,name", "gpu")
        processes = query("gpu_uuid,pid,process_name", "compute-apps")
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("Cannot verify idle GPUs with nvidia-smi before execution") from error
    gpu_by_index = {row[0]: {"index": row[0], "uuid": row[1], "name": row[2]} for row in rows}
    mask = devices if devices is not None else os.environ.get("CUDA_VISIBLE_DEVICES")
    requested = list(gpu_by_index) if mask is None else [item.strip() for item in mask.split(",") if item.strip()]
    selected = []
    for value in requested:
        if value in gpu_by_index:
            selected.append(gpu_by_index[value])
        else:
            matches = [gpu for gpu in gpu_by_index.values() if gpu["uuid"].startswith(value)]
            if len(matches) != 1:
                raise ValueError(f"Cannot identify selected GPU {value!r}; use an index or unique GPU UUID")
            selected.append(matches[0])
    if len({gpu["uuid"] for gpu in selected}) != len(selected):
        raise ValueError("Selected GPU mask contains duplicate devices")
    if len(selected) < required_devices:
        raise ValueError(f"Requested {required_devices} processes but only {len(selected)} GPUs are visible")
    selected = selected[:required_devices]
    selected_uuids = {gpu["uuid"] for gpu in selected}
    allowed = {str(pid) for pid in allowed_pids}
    busy = [{"gpu_uuid": uuid, "pid": pid, "process": name}
            for uuid, pid, name in processes if uuid in selected_uuids and pid not in allowed]
    if busy:
        raise RuntimeError(f"Selected GPUs already have compute jobs: {json.dumps(busy)}")
    return selected


def preflight(cfg, output, world_size=2):
    import dill
    import hydra
    import torch
    from accelerate.data_loader import BatchSamplerShard
    from oat.common.p2n_new_capabilities import resolve_update_schedule
    from oat.workspace.train_p2n_new_convnext import TrainP2NNewWorkspace

    if output.exists() and any(output.iterdir()) and not cfg.training.resume:
        raise ValueError(f"Fresh output directory must be empty: {output}")
    split = legacy.inspect_dataset(cfg)
    if cfg.training.resume:
        payload = torch.load(cfg.training.resume_checkpoint, map_location="cpu", pickle_module=dill, weights_only=False)
        TrainP2NNewWorkspace.validate_resume_payload(payload, cfg)
        model = hydra.utils.instantiate(payload["policy_config"])
        model.load_state_dict(payload["state_dicts"]["model"], strict=True)
    else:
        path = cfg.policy.convnext_path
        if not path or not Path(path).expanduser().exists():
            raise FileNotFoundError(f"Fresh training requires explicit local pinned ConvNeXt weights: {path}")
        legacy.validate_tokenizer_source(cfg)
        model = hydra.utils.instantiate(cfg.policy)
    if model.max_seq_len != 8:
        raise ValueError(f"Selected OAT must have exactly eight action tokens, got {model.max_seq_len}")
    optimizer = model.get_optimizer(**cfg.optimizer)
    batch_sampler = torch.utils.data.BatchSampler(
        torch.utils.data.SequentialSampler(range(split["train"]["windows"])),
        batch_size=int(cfg.dataloader.batch_size), drop_last=bool(cfg.dataloader.drop_last))
    batches = len(BatchSamplerShard(batch_sampler, num_processes=world_size, process_index=0))
    schedule = resolve_update_schedule(
        batches, cfg.training.num_epochs, cfg.training.gradient_accumulate_every,
        max_train_steps=cfg.training.max_train_steps, warmup_steps=cfg.training.lr_warmup_steps,
        warmup_ratio=cfg.training.lr_warmup_ratio)
    worker = TrainP2NNewWorkspace(cfg, output_dir=str(output))
    worker.model, worker.optimizer = model, optimizer
    worker.dataset_split, worker.update_schedule = split, schedule
    report = worker.training_report(model)
    report.update({
        "vision": "convnext_nano", "encoder_type": "convnextv2_tokens",
        "effective_batch": int(cfg.dataloader.batch_size) * world_size * int(cfg.training.gradient_accumulate_every),
        "world_size": world_size, "validation_batch_per_rank": int(cfg.val_dataloader.batch_size),
        "self_past_chunk_size": int(cfg.policy.self_past_chunk_size),
        "status": "CPU construction/schema checks complete; real GPU/DDP performance remains unmeasured",
    })
    print(json.dumps(report, indent=2))
    return report


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--worker-config":
        from omegaconf import OmegaConf
        from oat.workspace.train_p2n_new_convnext import TrainP2NNewWorkspace
        parser = argparse.ArgumentParser()
        parser.add_argument("--worker-config", required=True)
        parser.add_argument("--output", required=True)
        args = parser.parse_args(argv)
        cfg = OmegaConf.load(args.worker_config)
        validate_config(cfg, cfg.variant, cfg.task_type)
        TrainP2NNewWorkspace(cfg, output_dir=args.output).run()
        return
    split = argv.index("--") if "--" in argv else len(argv)
    flags, overrides = argv[:split], argv[split + 1:]
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--vision", choices=("dinov3", "convnext_nano"), default="dinov3")
    selection, remaining = selector.parse_known_args(flags)
    if selection.vision == "dinov3" and not any(value in ("--help", "-h") for value in remaining):
        return legacy.main([*remaining, *(["--", *overrides] if overrides else [])])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vision", choices=("dinov3", "convnext_nano"), default="dinov3")
    parser.add_argument("--variant", choices=("p2n_new", "p2n_state_gate_new"), required=True)
    parser.add_argument("--task", choices=("libero", "real_robot"), required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--convnext")
    parser.add_argument("--convnext-revision")
    parser.add_argument("--dino")
    parser.add_argument("--dino-revision")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--devices", help="Idle GPU indices or UUIDs, e.g. 0,1; existing jobs are untouched")
    parser.add_argument("--num-processes", type=int, default=2)
    parser.add_argument("--cpu", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Configuration only: no weights, datasets or GPU inspection")
    mode.add_argument("--preflight", action="store_true", help="CPU local-weight/schema/parameter checks; no training")
    args = parser.parse_args(flags)
    if args.num_processes < 1:
        parser.error("--num-processes must be positive")
    if args.dino or args.dino_revision:
        parser.error("--vision convnext_nano cannot also select DINO weights or revision")
    generated = []
    for value, key in ((args.tokenizer, "policy.tokenizer_checkpoint"), (args.convnext, "policy.convnext_path"),
                       (args.convnext_revision, "policy.convnext_revision")):
        if value is not None:
            generated.append(f"{key}={json.dumps(str(value))}")
    if args.resume:
        generated.extend(["training.resume=true", f"training.resume_checkpoint={json.dumps(str(args.resume.resolve()))}"])
    cfg = compose_config(args.variant, args.task, [*generated, *overrides])
    from omegaconf import OmegaConf
    output = (args.output or ROOT / "output/training" / f"{args.variant}_convnext_nano_{args.task}_seed{cfg.seed}").resolve()
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print(json.dumps({"vision": args.vision, "output": str(output), "world_size": args.num_processes,
                      "effective_batch": int(cfg.dataloader.batch_size) * args.num_processes * int(cfg.training.gradient_accumulate_every),
                      "mode": "dry_run" if args.dry_run else "preflight" if args.preflight else "training"}, indent=2))
    if args.dry_run:
        print("Configuration resolved only: weights, dataset split, parameters and GPU performance are unverified.")
        return
    if args.devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    if args.cpu:
        os.environ["ACCELERATE_USE_CPU"] = "true"
    elif not args.preflight:
        selected = check_gpu_idle(args.devices, args.num_processes)
        # UUIDs remove ambiguity between CUDA's default ordering and nvidia-smi.
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu["uuid"] for gpu in selected)
    report = preflight(cfg, output, args.num_processes)
    if args.preflight:
        return report
    if not args.cpu:
        check_gpu_idle(os.environ["CUDA_VISIBLE_DEVICES"], args.num_processes)
    output.mkdir(parents=True, exist_ok=True)
    resolved = output / "p2n_new_convnext_resolved.yaml"
    resolved.write_text(OmegaConf.to_yaml(cfg, resolve=True))
    (output / "p2n_new_convnext_preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    command = [sys.executable]
    if args.num_processes > 1:
        command += ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node", str(args.num_processes)]
    command += [str(Path(__file__).resolve()), "--worker-config", str(resolved), "--output", str(output)]
    subprocess.run(command, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
