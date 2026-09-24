#!/usr/bin/env python
"""Explicit latent-flow launcher; dry-run performs CPU preflight and never trains."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CONFIG_NAMES = {
    ("p2n_latent_flow", "libero"): "train_p2n_latent_flow",
    ("p2n_state_gate_latent_flow", "libero"): "train_p2n_state_gate_latent_flow",
    ("p2n_latent_flow", "real_robot"): "experimental/train_p2n_latent_flow_real_robot",
    ("p2n_state_gate_latent_flow", "real_robot"): "experimental/train_p2n_state_gate_latent_flow_real_robot",
}


def compose_config(variant, task, overrides=()):
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=str(ROOT / "oat/config"), version_base=None):
        cfg = compose(config_name=CONFIG_NAMES[variant, task], overrides=list(overrides))
    validate_config(cfg, variant, task)
    return cfg


def compose_resume_config(payload, variant, task, overrides=()):
    """Resume from saved resolved configuration, then apply explicit user overrides."""
    from hydra import compose, initialize_config_dir
    from hydra.core.config_store import ConfigStore
    from omegaconf import OmegaConf
    saved = payload.get("cfg")
    if saved is None:
        raise ValueError("Resume artifact lacks its resolved training configuration")
    saved = OmegaConf.create(OmegaConf.to_container(saved, resolve=True) if OmegaConf.is_config(saved) else saved)
    if saved.get("policy_family") != "oat_latent_flow" or saved.get("variant") != variant or saved.get("task_type") != task:
        raise ValueError("Resume artifact family, variant or task differs from the selected command")
    # Register in memory only; no configuration file or external source is changed.
    name = "_latent_flow_saved_resume"
    ConfigStore.instance().store(name=name, node=saved)
    with initialize_config_dir(config_dir=str(ROOT / "oat/config"), version_base=None):
        cfg = compose(config_name=name, overrides=list(overrides))
    validate_config(cfg, variant, task)
    return cfg


def validate_config(cfg, variant, task):
    gate = variant == "p2n_state_gate_latent_flow"
    cls = "P2NStateGateLatentFlowPolicy" if gate else "P2NLatentFlowPolicy"
    expected = f"oat.policy.{variant}.{cls}"
    if cfg.policy_family != "oat_latent_flow" or cfg.variant != variant or cfg.policy.variant != variant or cfg.policy._target_ != expected:
        raise ValueError("Selected flow family, variant and policy target must agree")
    if cfg.task_type != task or cfg.policy.task != task:
        raise ValueError("Selected task and policy task must agree")
    if cfg.training.get("init_checkpoint"):
        raise ValueError("AR warm-start is unsupported; select fresh flow training or flow resume")
    if cfg.training.resume and not cfg.training.get("resume_checkpoint"):
        raise ValueError("Resume requires a full --resume CHECKPOINT")
    if cfg.training.use_ema is not True:
        raise ValueError("Flow consistency training requires use_ema=true")
    batch = cfg.dataloader.batch_size
    if isinstance(batch, bool) or int(batch) != batch or batch < 4 or batch % 4:
        raise ValueError("Training microbatch must be a multiple of four and at least four for the 3:1 FM/CT split")
    if cfg.dataloader.drop_last is not True or cfg.val_dataloader.drop_last is not False:
        raise ValueError("Training requires drop_last=true and validation requires drop_last=false")
    ds = cfg.task.policy.dataset
    if ds.history_padding != "zero" or ds.return_history_validity is not True:
        raise ValueError("Both flow variants require zero history padding and explicit validity")
    dataset_name = ("LatentFlowRealRobot" if task == "real_robot" else "LatentFlow") + ("ZarrDatasetWithStateHistory" if gate else "ZarrDatasetWithPrevWindow")
    if ds._target_ != "oat.dataset.latent_flow_dataset." + dataset_name:
        raise ValueError("Selected variant/task does not match the flow dataset adapter")
    if not gate and any(k.startswith("history_") or k.startswith("state_history") for k in cfg.policy):
        raise ValueError("The plain flow variant must not create gate/state-history modules")
    if gate and (cfg.policy.state_history_steps != 8 or cfg.policy.history_summary_tokens != 4 or cfg.policy.history_dropout != 0):
        raise ValueError("Gate flow recipe requires eight measured states, four summaries and zero dropout")
    if gate and cfg.policy.history_gate_mode != "learned":
        raise ValueError("Formal flow training requires history_gate_mode=learned; open/closed are test-only")
    if task == "real_robot":
        if cfg.task.policy.env_runner is not None or cfg.task.policy.lazy_eval is not True:
            raise ValueError("Real-robot flow uses offline evaluation and no simulator runner")
        if gate and ("robot0_eef_rot6d" not in cfg.policy.state_history_keys or cfg.policy.rotation_6d_layout != "rows"):
            raise ValueError("Real-robot history requires rotation-6D rows")
    else:
        runner = "P2NStateGateLatentFlowLiberoRunner" if gate else "P2NLatentFlowLiberoRunner"
        if cfg.task.policy.env_runner._target_ != "oat.env_runner.p2n_latent_flow_runner." + runner:
            raise ValueError("LIBERO runner must implement the selected flow execution/history protocol")
    fixed = {"horizon": 16, "n_obs_steps": 2, "past_n": 7}
    if any(cfg[key] != value for key, value in fixed.items()) or cfg.policy.dropout != 0:
        raise ValueError("Flow recipe requires horizon16, two observations, seven past actions and zero dropout")
    required = {"latent_space": "fsq_normalized_codes", "num_slots": 8, "code_dim": 5,
                "levels": [8, 5, 5, 5, 5], "fm_fraction": 0.75, "ct_weight": 1.0,
                "fm_beta": [1.0, 1.5], "fm_time_scale": 0.999, "ct_time_bins": 10,
                "teacher_dt_mode": "same_relative_dt", "solver": "euler",
                "endpoint_projection": "fsq_nearest_grid"}
    for key, value in required.items():
        if cfg.policy.flow.get(key) != value:
            raise ValueError(f"Unsupported flow recipe: policy.flow.{key} must equal {value}")
    for key in ("inference_steps", "self_past_steps"):
        if int(cfg.policy.flow[key]) < 1:
            raise ValueError(f"policy.flow.{key} must be positive")
    for key in ("num_epochs", "gradient_accumulate_every", "checkpoint_every", "val_every", "sample_every"):
        if int(cfg.training[key]) < 1:
            raise ValueError(f"training.{key} must be positive")
    if cfg.training.get("max_val_steps") is not None or cfg.training.get("max_reconst_steps") is not None:
        raise ValueError("Full held-out validation requires max_val_steps=null and max_reconst_steps=null")
    if int(cfg.val_dataloader.batch_size) < 1 or int(cfg.policy.self_past_chunk_size) < 1:
        raise ValueError("Validation batch and self-past chunk size must be positive")


def inspect_dataset(cfg):
    # Reuse the existing read-only schema/split inspector; it does not construct a policy.
    from scripts.train_p2n_new import inspect_dataset as inspect_existing_dataset
    import zarr
    from oat.dataset.latent_flow_dataset import dataset_fingerprint
    split = inspect_existing_dataset(cfg)
    ds = cfg.task.policy.dataset
    root = zarr.open(str(Path(ds.zarr_path).expanduser()), mode="r")
    arrays = {key: root["data"][key] for key in sorted(set([ds.action_key, *ds.obs_keys]))}
    split["identity"] = dataset_fingerprint(ds.zarr_path, root["meta"]["episode_ends"][:], arrays, ds.action_key)
    split["sample_id_definition"] = "absolute_action_anchor_v1"
    split["future_action_valid_definition"] = "pad_before + arange(horizon) in [sample_start,sample_end)"
    return split


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_tokenizer_source(cfg):
    import dill
    import torch
    from omegaconf import OmegaConf
    path = cfg.policy.tokenizer_checkpoint
    if not path or not Path(path).is_file():
        raise FileNotFoundError(f"Fresh training needs a task-matched local OAT checkpoint: {path}")
    payload = torch.load(path, map_location="cpu", pickle_module=dill, weights_only=False)
    saved = payload.get("cfg")
    if saved is None or "tokenizer" not in saved.get("task", {}):
        raise ValueError("OAT checkpoint lacks task/normalizer provenance metadata")
    source, target = saved.task.tokenizer.dataset, cfg.task.policy.dataset
    for key in ("zarr_path", "action_key", "n_action_steps", "seed", "val_ratio", "max_train_episodes"):
        a, b = source.get(key), target.get(key)
        if key == "zarr_path":
            a, b = str(Path(a).expanduser().resolve()), str(Path(b).expanduser().resolve())
        if a != b:
            raise ValueError(f"OAT action/normalizer provenance mismatch: dataset.{key}")
    tok = saved.tokenizer
    expected = {"sample_dim": 7, "sample_horizon": 16, "latent_dim": 5}
    for component in ("encoder", "decoder"):
        if any(tok[component].get(key) != value for key, value in expected.items()):
            raise ValueError(f"OAT {component} schema must encode 16x7 actions into 8x5 scalar codes")
    if tok.encoder.num_registers != 8 or tok.decoder.latent_horizon != 8 or list(tok.quantizer.levels) != [8, 5, 5, 5, 5]:
        raise ValueError("OAT latent horizon or FSQ levels do not match the flow recipe")
    weights = payload.get("state_dicts", {}).get("ema_model")
    if not weights:
        raise ValueError("Fresh flow training requires tokenizer EMA weights")
    for key in ("normalizer.params_dict.action.scale", "normalizer.params_dict.action.offset"):
        if key not in weights or tuple(weights[key].shape) != (7,) or not torch.isfinite(weights[key]).all():
            raise ValueError("OAT EMA must include its finite seven-dimensional action normalizer")
    if (weights["normalizer.params_dict.action.scale"] == 0).any():
        raise ValueError("OAT action normalization scale must be nonzero")
    return {"checkpoint": str(Path(path).resolve()), "sha256": _file_sha256(path), "weights": "ema_model",
            "config": OmegaConf.to_container(tok, resolve=True), "latent_shape": [8, 5],
            "action_shape": [16, 7], "levels": [8, 5, 5, 5, 5]}


def validate_dino_source(cfg):
    from oat.perception.dinov3_patch_encoder import DINOv3PatchEncoder, _weight_digest, DINO_MODEL_ID
    value = cfg.policy.dino_path
    if not value or not Path(value).is_dir():
        raise FileNotFoundError(f"Fresh training requires a local DINOv3-S/16 snapshot: {value}")
    path = Path(value).expanduser()
    config = json.loads((path / "config.json").read_text())
    processor = json.loads((path / "preprocessor_config.json").read_text())
    DINOv3PatchEncoder._validate_production_config(config)
    revision = cfg.policy.dino_revision or config.get("_commit_hash") or path.name
    if not re.fullmatch(r"[0-9a-fA-F]{40}", str(revision)):
        raise ValueError("DINO requires a pinned 40-character commit revision")
    if not processor.get("image_mean") or not processor.get("image_std"):
        raise ValueError("DINO processor must preserve its original image normalization")
    return {"model_id": DINO_MODEL_ID, "path": str(path.resolve()), "revision": revision,
            "weight_sha256": _weight_digest(path)}


def preflight(cfg, output, world_size=2, resume_payload=None):
    import torch
    from accelerate.data_loader import BatchSamplerShard
    from oat.common.p2n_new_capabilities import resolve_update_schedule
    output = Path(output)
    if output.exists() and (not output.is_dir() or (any(output.iterdir()) and not cfg.training.resume)):
        raise ValueError(f"Fresh output directory must be empty: {output}")
    ancestor = output
    while not ancestor.exists():
        ancestor = ancestor.parent
    if not os.access(ancestor, os.W_OK):
        raise PermissionError(f"Output parent is not writable: {ancestor}")
    split = inspect_dataset(cfg)
    if cfg.training.resume:
        import dill
        from oat.workspace.train_p2n_latent_flow import TrainP2NLatentFlowWorkspace
        path = Path(cfg.training.resume_checkpoint)
        payload = resume_payload if resume_payload is not None else torch.load(path, map_location="cpu", pickle_module=dill, weights_only=False)
        TrainP2NLatentFlowWorkspace.validate_resume_payload(payload, cfg)
        saved_split = payload["metadata"].get("dataset_split", {})
        if saved_split.get("identity") != split["identity"] or saved_split.get("train_episode_ids") != split["train_episode_ids"] or saved_split.get("validation_episode_ids") != split["validation_episode_ids"]:
            raise ValueError("Resume dataset identity or actual episode masks differ from the artifact")
        states = payload.get("training_state", {}).get("rng_states")
        if states is None and "rng_states" in payload.get("pickles", {}):
            states = dill.loads(payload["pickles"]["rng_states"])
        if states is None or len(states) != world_size:
            raise ValueError("Exact resume requires the saved distributed world size and per-rank RNG states")
        # Resume constructs both frozen architectures from this complete artifact.
        sources = {"resume": str(path.resolve()), "external_frozen_sources_required": False}
    else:
        sources = {"dino": validate_dino_source(cfg), "oat": validate_tokenizer_source(cfg)}
        cfg.policy.dino_revision = sources["dino"]["revision"]
    sampler = torch.utils.data.BatchSampler(torch.utils.data.SequentialSampler(range(split["train"]["windows"])),
        batch_size=int(cfg.dataloader.batch_size), drop_last=True)
    batches = len(BatchSamplerShard(sampler, num_processes=world_size, process_index=0))
    schedule = resolve_update_schedule(batches, int(cfg.training.num_epochs), int(cfg.training.gradient_accumulate_every),
        max_train_steps=cfg.training.max_train_steps, warmup_steps=cfg.training.lr_warmup_steps,
        warmup_ratio=float(cfg.training.lr_warmup_ratio))
    return {"policy_family": cfg.policy_family, "variant": cfg.variant, "task": cfg.task_type,
            "world_size": world_size, "effective_batch": int(cfg.dataloader.batch_size) * world_size * int(cfg.training.gradient_accumulate_every),
            "dataset_split": split, "sources": sources, "update_schedule": schedule,
            "output": str(output), "status": "CPU source/schema preflight passed; no model, GPU work or training started",
            "unmeasured": ["full-model GPU peak memory", "real two-rank training", "robot closed-loop quality"]}


def check_gpu_selection(gpus, allow_busy=False):
    """Read NVIDIA inventory immediately before launch, without a CUDA context."""
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                            text=True, capture_output=True, check=True)
    inventory = {}
    for row in result.stdout.splitlines():
        index, memory, utilization = [part.strip() for part in row.split(",")]
        inventory[index] = (int(memory), int(utilization))
    missing = set(gpus) - set(inventory)
    if missing:
        raise ValueError(f"Selected GPU indices do not exist: {sorted(missing)}")
    busy = {gpu: inventory[gpu] for gpu in gpus if inventory[gpu][0] > 1024 or inventory[gpu][1] > 5}
    if busy and not allow_busy:
        raise RuntimeError(f"Selected GPUs appear busy (memory MiB, utilization %): {busy}; choose free GPUs or explicitly use --allow-busy-gpus")


def main(argv=None):
    from omegaconf import OmegaConf
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--worker-config":
        parser = argparse.ArgumentParser()
        parser.add_argument("--worker-config", required=True)
        parser.add_argument("--output", required=True)
        args = parser.parse_args(argv)
        from oat.workspace.train_p2n_latent_flow import TrainP2NLatentFlowWorkspace
        TrainP2NLatentFlowWorkspace(OmegaConf.load(args.worker_config), output_dir=args.output).run()
        return
    overrides = []
    if "--" in argv:
        index = argv.index("--")
        overrides, argv = argv[index + 1:], argv[:index]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("p2n_latent_flow", "p2n_state_gate_latent_flow"), required=True)
    parser.add_argument("--task", choices=("libero", "real_robot"), required=True)
    parser.add_argument("--gpus", required=True, help="Explicit physical GPU indices, e.g. 2,3; dry-run does not inspect/use them")
    parser.add_argument("--num-processes", type=int, help="Defaults to the number of selected GPUs; must match")
    parser.add_argument("--tokenizer")
    parser.add_argument("--dino")
    parser.add_argument("--dino-revision")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--allow-busy-gpus", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Resolve and validate local data/weights on CPU; never construct/train a policy")
    mode.add_argument("--preflight", action="store_true", help="Alias of --dry-run with the same complete CPU checks")
    args = parser.parse_args(argv)
    gpus = args.gpus.split(",")
    if not gpus or any(not item.isdigit() for item in gpus) or len(set(gpus)) != len(gpus):
        parser.error("--gpus must contain distinct comma-separated integer GPU indices")
    world_size = args.num_processes if args.num_processes is not None else len(gpus)
    if world_size != len(gpus):
        parser.error("--num-processes must equal the number of explicitly selected GPUs")
    generated = []
    for value, key in ((args.tokenizer, "policy.tokenizer_checkpoint"), (args.dino, "policy.dino_path"), (args.dino_revision, "policy.dino_revision")):
        if value is not None:
            generated.append(f"{key}={json.dumps(str(value))}")
    if args.resume:
        generated.extend(["training.resume=true", f"training.resume_checkpoint={json.dumps(str(args.resume.resolve()))}"])
    payload = None
    if args.resume:
        import dill
        import torch
        payload = torch.load(args.resume, map_location="cpu", pickle_module=dill, weights_only=False)
        cfg = compose_resume_config(payload, args.variant, args.task, [*generated, *overrides])
    else:
        cfg = compose_config(args.variant, args.task, [*generated, *overrides])
    output = (args.output or ROOT / "output/training" / f"{args.variant}_{args.task}_seed{cfg.seed}").resolve()
    print(OmegaConf.to_yaml(cfg, resolve=True))
    report = preflight(cfg, output, world_size, resume_payload=payload) if payload is not None else preflight(cfg, output, world_size)
    print(json.dumps(report, indent=2))
    if args.dry_run or args.preflight:
        return report
    check_gpu_selection(gpus, args.allow_busy_gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    output.mkdir(parents=True, exist_ok=True)
    resolved = output / "p2n_latent_flow_resolved.yaml"
    resolved.write_text(OmegaConf.to_yaml(cfg, resolve=True))
    (output / "p2n_latent_flow_preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    command = [sys.executable]
    if world_size > 1:
        command += ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node", str(world_size)]
    command += [str(Path(__file__).resolve()), "--worker-config", str(resolved), "--output", str(output)]
    subprocess.run(command, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
