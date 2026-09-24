#!/usr/bin/env python
"""Explicit direct action-flow launcher; dry-run performs CPU preflight and never trains."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CONFIG_NAMES = {
    ("p2n_action_flow", "libero"): "train_p2n_action_flow",
    ("p2n_state_gate_action_flow", "libero"): "train_p2n_state_gate_action_flow",
    ("p2n_action_flow", "real_robot"): "experimental/train_p2n_action_flow_real_robot",
    ("p2n_state_gate_action_flow", "real_robot"): "experimental/train_p2n_state_gate_action_flow_real_robot",
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
    if saved.get("policy_family") != "continuous_action_flow" or saved.get("variant") != variant or saved.get("task_type") != task:
        raise ValueError("Resume artifact family, variant or task differs from the selected command")
    # Register in memory only; no configuration file or external source is changed.
    name = "_action_flow_saved_resume"
    ConfigStore.instance().store(name=name, node=saved)
    with initialize_config_dir(config_dir=str(ROOT / "oat/config"), version_base=None):
        cfg = compose(config_name=name, overrides=list(overrides))
    validate_config(cfg, variant, task)
    return cfg


def validate_config(cfg, variant, task):
    from omegaconf import OmegaConf
    forbidden = {"tokenizer", "tokenizer_checkpoint", "latent_space", "code_dim", "num_slots",
                 "levels", "latent_horizon", "codebook", "bos", "use_k_tokens", "top_k", "endpoint_projection"}
    def reject_codec_fields(node, prefix=""):
        if isinstance(node, dict):
            for key, value in node.items():
                if key.lower() in forbidden:
                    raise ValueError(f"continuous_action_flow family/schema rejects codec/AR field {prefix}{key}")
                reject_codec_fields(value, prefix + key + ".")
        elif isinstance(node, list):
            for value in node:
                reject_codec_fields(value, prefix)
    reject_codec_fields(OmegaConf.to_container(cfg, resolve=True))
    gate = variant == "p2n_state_gate_action_flow"
    cls = "P2NStateGateActionFlowPolicy" if gate else "P2NActionFlowPolicy"
    expected = f"oat.policy.{variant}.{cls}"
    if cfg.policy_family != "continuous_action_flow" or cfg.variant != variant or cfg.policy.variant != variant or cfg.policy._target_ != expected:
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
    dataset_name = ("ActionFlowRealRobot" if task == "real_robot" else "ActionFlow") + ("ZarrDatasetWithStateHistory" if gate else "ZarrDatasetWithPrevWindow")
    if ds._target_ != "oat.dataset.action_flow_dataset." + dataset_name:
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
        runner = "P2NStateGateActionFlowLiberoRunner" if gate else "P2NActionFlowLiberoRunner"
        if cfg.task.policy.env_runner._target_ != "oat.env_runner.p2n_action_flow_runner." + runner:
            raise ValueError("LIBERO runner must implement the selected flow execution/history protocol")
    fixed = {"horizon": 16, "n_obs_steps": 2, "past_n": 7}
    if any(cfg[key] != value for key, value in fixed.items()) or cfg.policy.dropout != 0:
        raise ValueError("Flow recipe requires horizon16, two observations, seven past actions and zero dropout")
    if cfg.policy.action_dim != 7 or list(cfg.shape_meta.action.shape) != [7]:
        raise ValueError("Direct action flow requires a seven-dimensional action schema")
    if cfg.policy.resampler_ffn_type != "swiglu" or int(cfg.policy.resampler_ffn_dim) != 2048:
        raise ValueError("Visual Resampler requires SwiGLU intermediate=2048")
    if (cfg.normalization.source != "training_replay_frames" or cfg.normalization.mode != "limits"
            or cfg.normalization.refit_on_resume is not False):
        raise ValueError("Action normalization requires training replay frames, limits and no refit on resume")
    schema = cfg.action_schema
    if (schema.action_dim != 7 or schema.horizon != cfg.horizon or schema.execution_steps != cfg.n_action_steps
            or schema.execution_anchor != "current_observation_t" or int(schema.control_frequency_hz) < 1):
        raise ValueError("Action schema must record dimensions, current-t execution anchor and control frequency")
    for component in ("translation", "rotation", "gripper", "gripper_observation"):
        if not schema[component].get("units") or not schema[component].get("representation"):
            raise ValueError(f"Action schema requires explicit units/representation for {component}")
    required = {"space": "normalized_actions", "ffn_type": "gelu", "ffn_hidden_dim": 3072,
                "self_qk_norm": False, "cross_qk_norm": "layernorm", "qkv_bias": True,
                "time_embed_dim": 128, "global_condition": "time_and_step_only", "pre_norm_modality": False,
                "fm_fraction": 0.75, "ct_weight": 1.0,
                "fm_beta": [1.0, 1.5], "fm_time_scale": 0.999, "ct_time_bins": 10,
                "teacher_dt_mode": "same_relative_dt", "solver": "euler",
                "loss_padding": "supervise_edge_repeated_targets", "output_transform": "action_unnormalize"}
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
    from oat.dataset.action_flow_dataset import dataset_fingerprint
    split = inspect_existing_dataset(cfg)
    ds = cfg.task.policy.dataset
    root = zarr.open(str(Path(ds.zarr_path).expanduser()), mode="r")
    arrays = {key: root["data"][key] for key in sorted(set([ds.action_key, *ds.obs_keys]))}
    split["identity"] = dataset_fingerprint(ds.zarr_path, root["meta"]["episode_ends"][:], arrays, ds.action_key)
    split["sample_id_definition"] = "absolute_action_anchor_v1"
    split["future_action_valid_definition"] = "pad_before + arange(horizon) in [sample_start,sample_end)"
    return split


def inspect_training_normalizer(cfg, split):
    """Fit only the small numeric training replay arrays, never sampled windows.

    This mirrors the existing dataset normalizer math without loading cameras or
    instantiating a policy. Its result is a CPU preflight report, not a substitute
    for the dataset-owned normalizer used by fresh training.
    """
    import numpy as np
    import torch
    import zarr
    from oat.model.common.normalizer import LinearNormalizer
    ds = cfg.task.policy.dataset
    root = zarr.open(str(Path(ds.zarr_path).expanduser()), mode="r")
    ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    lengths = np.diff(np.r_[0, ends])
    train_mask = np.zeros(len(ends), dtype=bool)
    train_mask[split["train_episode_ids"]] = True
    frame_mask = np.repeat(train_mask, lengths)
    if not frame_mask.any():
        raise ValueError("Normalizer requires nonempty training replay frames")
    fields = {"action": ds.action_key, **{key: key for key, spec in cfg.shape_meta.obs.items() if spec.type != "rgb"}}
    normalizer = LinearNormalizer()
    for name, key in fields.items():
        data = np.asarray(root["data"][key][:])[frame_mask]
        if not np.isfinite(data).all():
            raise ValueError(f"Training replay field {key} contains non-finite values")
        normalizer.fit({name: data}, last_n_dims=1, mode=cfg.normalization.mode)
        field = normalizer[name]
        scale, offset = field.params_dict["scale"], field.params_dict["offset"]
        if not torch.isfinite(scale).all() or not torch.isfinite(offset).all() or (scale == 0).any():
            raise ValueError(f"Training normalizer for {name} must be finite and invertible")
        original = torch.as_tensor(data, dtype=torch.float32)
        recovered = field.unnormalize(field.normalize(original))
        if not torch.isfinite(recovered).all() or not torch.allclose(recovered, original, atol=2e-5, rtol=2e-5):
            raise ValueError(f"Training normalizer round trip failed for {name}")
    action = normalizer["action"]
    return {"source": "training_replay_frames", "mode": cfg.normalization.mode,
            "data_path": str(Path(ds.zarr_path).expanduser().resolve()),
            "dataset_identity": split["identity"], "action_key": ds.action_key,
            "train_episode_ids": split["train_episode_ids"], "train_frames": int(frame_mask.sum()),
            "fields": list(fields), "overlapping_windows": False, "padded_frames": False,
            "refit_on_resume": False,
            "action_scale": action.params_dict["scale"].tolist(),
            "action_offset": action.params_dict["offset"].tolist(),
            "round_trip": "passed"}


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
        from oat.workspace.train_p2n_action_flow import TrainP2NActionFlowWorkspace
        path = Path(cfg.training.resume_checkpoint)
        payload = resume_payload if resume_payload is not None else torch.load(path, map_location="cpu", pickle_module=dill, weights_only=False)
        TrainP2NActionFlowWorkspace.validate_resume_payload(payload, cfg)
        saved_split = payload["metadata"].get("dataset_split", {})
        if saved_split.get("identity") != split["identity"] or saved_split.get("train_episode_ids") != split["train_episode_ids"] or saved_split.get("validation_episode_ids") != split["validation_episode_ids"]:
            raise ValueError("Resume dataset identity or actual episode masks differ from the artifact")
        states = payload.get("training_state", {}).get("rng_states")
        if states is None and "rng_states" in payload.get("pickles", {}):
            states = dill.loads(payload["pickles"]["rng_states"])
        if states is None or len(states) != world_size:
            raise ValueError("Exact resume requires the saved distributed world size and per-rank RNG states")
        # Resume reconstructs frozen DINO from this self-contained artifact.
        sources = {"resume": str(path.resolve()), "external_frozen_sources_required": False}
    else:
        sources = {"dino": validate_dino_source(cfg), "normalizer": inspect_training_normalizer(cfg, split)}
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
        from oat.workspace.train_p2n_action_flow import TrainP2NActionFlowWorkspace
        TrainP2NActionFlowWorkspace(OmegaConf.load(args.worker_config), output_dir=args.output).run()
        return
    overrides = []
    if "--" in argv:
        index = argv.index("--")
        overrides, argv = argv[index + 1:], argv[:index]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("p2n_action_flow", "p2n_state_gate_action_flow"), required=True)
    parser.add_argument("--task", choices=("libero", "real_robot"), required=True)
    parser.add_argument("--gpus", required=True, help="Explicit physical GPU indices, e.g. 2,3; dry-run does not inspect/use them")
    parser.add_argument("--num-processes", type=int, help="Defaults to the number of selected GPUs; must match")
    parser.add_argument("--batch-size", type=int, help="Training microbatch per rank; >=4 and divisible by four")
    parser.add_argument("--val-batch-size", type=int, help="Validation batch per rank")
    parser.add_argument("--grad-accum", type=int, help="Microbatches per successful optimizer update")
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
    for value, key in ((args.dino, "policy.dino_path"), (args.dino_revision, "policy.dino_revision")):
        if value is not None:
            generated.append(f"{key}={json.dumps(str(value))}")
    for value, key in ((args.batch_size, "dataloader.batch_size"), (args.val_batch_size, "val_dataloader.batch_size"),
                       (args.grad_accum, "training.gradient_accumulate_every")):
        if value is not None:
            generated.append(f"{key}={value}")
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
    resume_output = args.resume.resolve().parent if args.resume else None
    if args.resume and resume_output.name == "checkpoints":
        resume_output = resume_output.parent
    output = (args.output or (resume_output if args.resume else ROOT / "output/training" / f"{args.variant}_{args.task}_seed{cfg.seed}")).resolve()
    print(OmegaConf.to_yaml(cfg, resolve=True))
    report = preflight(cfg, output, world_size, resume_payload=payload) if payload is not None else preflight(cfg, output, world_size)
    print(json.dumps(report, indent=2))
    if args.dry_run or args.preflight:
        return report
    check_gpu_selection(gpus, args.allow_busy_gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    output.mkdir(parents=True, exist_ok=True)
    resolved = output / "p2n_action_flow_resolved.yaml"
    resolved.write_text(OmegaConf.to_yaml(cfg, resolve=True))
    (output / "p2n_action_flow_preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    command = [sys.executable]
    if world_size > 1:
        command += ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node", str(world_size)]
    command += [str(Path(__file__).resolve()), "--worker-config", str(resolved), "--output", str(output)]
    subprocess.run(command, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
