#!/usr/bin/env python
"""P2N-VLA launcher (LIBERO-10). See docs/P2N_VLA.md for literal commands.

Modes (mutually exclusive):
  --dry-run    compose and validate the configuration only; reads no data, weights or GPUs
  --preflight  dataset schema/split, OAT provenance, asset sha256s, prompt lengths, CPU model build
               and parameter counts; trains nothing
  --probe      short torchrun probe: N optimizer steps with self-past at its full probability, one
               worst-case history_mode='generated' pass and a validation pass; writes probe.json
               (memory, samples/s, go/no-go)
  (none)       light preflight (no model build), then torchrun training

Everything after ``--`` is passed to Hydra as overrides, e.g. ``-- training.num_epochs=1``.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
# This checkout must win over sibling editable installs of the oat package (e.g. /workspace/past_action).
sys.path.insert(0, str(ROOT))

CONFIG_DIR = ROOT / "oat/config"
ASSETS_MANIFEST = ROOT / "data/pretrained/p2n_vla/assets.json"
POLICY_FAMILY = "p2n_vla"
VARIANTS = {
    "p2n_vla": {"config": "train_p2n_vla", "target": "oat.policy.p2n_vla.P2NVLAPolicy",
                "gate": False, "flow": False},
    "p2n_vla_state_gate": {"config": "train_p2n_vla_state_gate",
                           "target": "oat.policy.p2n_vla_state_gate.P2NVLAStateGatePolicy",
                           "gate": True, "flow": False},
    "pi05_ki_flow": {"config": "train_pi05_ki_flow", "target": "oat.policy.pi05_ki_flow.PI05KIFlowPolicy",
                     "gate": False, "flow": True},
}
TASKS = ("libero",)
DATASET_TARGETS = {False: "oat.dataset.vla_dataset.VLAZarrDatasetWithPrevWindow",
                   True: "oat.dataset.vla_dataset.VLAZarrDatasetWithStateHistory"}
RUNNER_TARGETS = {False: "oat.env_runner.p2n_new_runner.P2NNewLiberoRunner",
                  True: "oat.env_runner.p2n_new_runner.P2NStateGateNewLiberoRunner"}
OPTIMIZER_KEYS = {"policy_lr", "new_module_lr", "weight_decay", "betas", "eps", "fused"}
EXPECTED_PI05_TENSORS = 812


def _plain(value):
    from omegaconf import OmegaConf
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


# --------------------------------------------------------------------------- configuration
def compose_config(variant, task="libero", overrides=(), *, config_dir=None):
    """Compose ``oat/config/<variant config>`` with Hydra overrides and validate it."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; choose from {sorted(VARIANTS)}")
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}; choose from {TASKS}")
    from hydra import compose, initialize_config_dir
    from oat.common.hydra_util import register_new_resolvers
    register_new_resolvers()
    with initialize_config_dir(config_dir=str(Path(config_dir or CONFIG_DIR).resolve()), version_base=None):
        cfg = compose(config_name=VARIANTS[variant]["config"], overrides=list(overrides))
    validate_config(cfg, variant, task)
    return cfg


def _int(value, name, minimum=1, allow_none=False):
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}" + (" or null" if allow_none else ""))
    return value


def validate_config(cfg, variant, task):
    """Variant/target/dataset/runner agreement plus the recipe invariants the workspace relies on."""
    spec = VARIANTS[variant]
    gate, flow = spec["gate"], spec["flow"]
    policy, training = cfg.policy, cfg.training
    if cfg.get("policy_family") != POLICY_FAMILY:
        raise ValueError(f"policy_family must be {POLICY_FAMILY!r}")
    if cfg.get("variant") != variant or policy.get("variant") != variant or policy.get("_target_") != spec["target"]:
        raise ValueError(f"Selected variant {variant!r}, the config variant {cfg.get('variant')!r}, "
                         f"policy.variant {policy.get('variant')!r} and policy._target_ {policy.get('_target_')!r} "
                         f"must agree (expected {spec['target']})")
    if cfg.get("task_type") != task or policy.get("task") != task:
        raise ValueError("Selected task, task_type and policy.task must agree")
    if cfg.get("_target_") != "oat.workspace.train_p2n_vla.TrainP2NVLAWorkspace":
        raise ValueError("The config must target oat.workspace.train_p2n_vla.TrainP2NVLAWorkspace")
    if training.get("init_checkpoint"):
        raise ValueError("P2N-VLA starts from pi05_base; init_checkpoint is unsupported (use --resume)")
    if training.get("resume") and not training.get("resume_checkpoint"):
        raise ValueError("Resume requires --resume CHECKPOINT (training.resume_checkpoint)")
    if policy.get("construction_mode") != "fresh":
        raise ValueError("Training constructs the policy fresh; resume restores it from the checkpoint")
    if policy.get("model_size") not in ("full", "tiny"):
        raise ValueError("policy.model_size must be full or tiny")
    if policy.model_size == "tiny" and policy.get("pi05_weights"):
        raise ValueError("Tiny models are randomly initialized; set policy.pi05_weights=null")
    # schema alignment (OAT: 16x7 actions <-> 8 tokens; execute 8; 7 past commands; current frame only)
    expected = {"horizon": 16, "n_action_steps": 8, "past_n": 7, "n_obs_steps": 1}
    for key, value in expected.items():
        if cfg.get(key) != value or policy.get(key) != value:
            raise ValueError(f"P2N-VLA recipe requires {key}={value} at the top level and in policy")
    ds = cfg.task.policy.dataset
    if ds.get("_target_") != DATASET_TARGETS[gate]:
        raise ValueError(f"{variant} requires dataset {DATASET_TARGETS[gate]}")
    if ds.get("history_padding") != "zero" or ds.get("return_history_validity") is not True:
        raise ValueError("The dataset needs zero history padding and explicit validity")
    if not Path(str(ds.get("zarr_path"))).is_absolute():
        raise ValueError("task.policy.dataset.zarr_path must be absolute (the OAT records an absolute path)")
    if (ds.get("n_action_steps") != cfg.horizon or ds.get("n_exec_steps") != cfg.n_action_steps
            or ds.get("past_n") != cfg.past_n or ds.get("n_obs_steps") != cfg.n_obs_steps):
        raise ValueError("Dataset windows must match horizon/n_action_steps/past_n/n_obs_steps")
    obs_keys = list(ds.get("obs_keys") or [])
    if "task_uid" not in obs_keys:
        raise ValueError("task_uid must be an observation key (the prompt language comes from it)")
    prompt = _plain(policy.get("prompt")) or {}
    prompt_state = _plain(ds.get("prompt_state")) or {}
    if list(prompt.get("state_keys") or []) != list(prompt_state.get("keys") or []) or \
            dict(prompt.get("state_transforms") or {}) != dict(prompt_state.get("transforms") or {}):
        raise ValueError("policy.prompt state keys/transforms must equal task.policy.dataset.prompt_state")
    for key in prompt_state.get("keys") or []:
        if key not in obs_keys or key not in cfg.shape_meta.obs:
            raise ValueError(f"Prompt-state key {key!r} must be a dataset observation and in shape_meta")
    _int(prompt.get("max_len"), "policy.prompt.max_len", minimum=2)
    if list(policy.get("rgb_ports") or []) and not set(policy.wrist_ports) <= set(policy.rgb_ports):
        raise ValueError("policy.wrist_ports must be a subset of policy.rgb_ports")
    for port in policy.get("rgb_ports") or []:
        if port not in obs_keys or cfg.shape_meta.obs[port].get("type") != "rgb":
            raise ValueError(f"RGB port {port!r} must be an rgb dataset observation")
    runner = cfg.task.policy.get("env_runner")
    if runner is None or runner.get("_target_") != RUNNER_TARGETS[gate]:
        raise ValueError(f"{variant} must be evaluated by {RUNNER_TARGETS[gate]}")
    if runner.get("protocol") not in ("corrected", "official"):
        raise ValueError("The P2N runners support the corrected and official protocols only")
    if cfg.task.policy.get("lazy_eval") is not True:
        raise ValueError("No in-training rollouts: set task.policy.lazy_eval=true and use scripts/evaluate_p2n_vla.py")
    history = sorted(key for key in policy if str(key).startswith(("history_", "state_history")))
    if not gate and history:
        raise ValueError(f"{variant} must not construct history/gate modules: {history}")
    if gate:
        if policy.get("state_history_steps") != cfg.past_n + 1 or policy.get("history_summary_tokens") != 4:
            raise ValueError("The gate recipe needs past_n + 1 measured states and four summaries")
        if policy.get("history_gate_mode") not in ("learned", "open", "closed"):
            raise ValueError("history_gate_mode must be learned, open or closed")
        for name, node in (("dataset", ds), ("env_runner", runner)):
            if (list(node.get("state_history_keys") or []) != list(policy.state_history_keys)
                    or node.get("state_history_steps") != policy.state_history_steps):
                raise ValueError(f"policy and {name} state-history keys/steps must agree")
        if policy.get("use_past") is not True:
            raise ValueError("The gate variant needs past conditioning (use_past=true)")
    if flow:
        if policy.get("use_past") is not False:
            raise ValueError("pi05_ki_flow takes no past input: set policy.use_past=false")
        if training.get("validate_generated_history"):
            raise ValueError("pi05_ki_flow has no generated-history validation")
    lambda_ki = float(policy.get("lambda_ki", 1.0))
    if not math.isfinite(lambda_ki) or lambda_ki < 0:
        raise ValueError("policy.lambda_ki must be finite and nonnegative")
    _int(policy.get("lora_rank"), "policy.lora_rank")
    if not 0.0 <= float(policy.get("self_past_p", 0.0)) <= 1.0:
        raise ValueError("policy.self_past_p must lie in [0, 1]")
    if float(policy.get("temperature", 0.0)) < 0:
        raise ValueError("policy.temperature must be nonnegative (0 = greedy)")
    # training recipe
    for key in ("num_epochs", "gradient_accumulate_every", "checkpoint_every", "val_every", "log_every", "num_demo"):
        _int(training.get(key), f"training.{key}")
    _int(training.get("snapshot_every"), "training.snapshot_every", minimum=0)
    for key in ("max_train_steps", "max_optimizer_steps", "max_val_steps"):
        _int(training.get(key), f"training.{key}", allow_none=True)
    _int(training.get("max_reconst_steps"), "training.max_reconst_steps", minimum=0, allow_none=True)
    warmup = _int(training.get("lr_warmup_steps"), "training.lr_warmup_steps", minimum=0)
    if training.get("max_optimizer_steps") is not None and warmup > training.max_optimizer_steps:
        raise ValueError("training.lr_warmup_steps cannot exceed training.max_optimizer_steps")
    if not 0.0 <= float(training.get("min_lr_ratio")) <= 1.0:
        raise ValueError("training.min_lr_ratio must lie in [0, 1]")
    if training.get("max_grad_norm") is not None and not float(training.max_grad_norm) > 0:
        raise ValueError("training.max_grad_norm must be positive or null")
    if not isinstance(training.get("use_ema"), bool):
        raise ValueError("training.use_ema must be a boolean")
    decay = float(cfg.ema.get("decay"))
    if not 0.0 <= decay < 1.0:
        raise ValueError("ema.decay must lie in [0, 1)")
    if set(cfg.optimizer) != OPTIMIZER_KEYS:
        raise ValueError(f"optimizer must define exactly {sorted(OPTIMIZER_KEYS)}")
    _int(cfg.dataloader.get("batch_size"), "dataloader.batch_size")
    _int(cfg.val_dataloader.get("batch_size"), "val_dataloader.batch_size")
    if cfg.dataloader.get("drop_last") is not True or cfg.val_dataloader.get("drop_last") is not False:
        raise ValueError("Training needs drop_last=true (full micro-batches); validation needs drop_last=false")
    if cfg.logging.get("mode") not in ("online", "offline", "disabled"):
        raise ValueError("logging.mode must be online, offline or disabled")


def expected_schedule(cfg, train_windows=None, world_size=2):
    """Update arithmetic; exact when the number of training windows is known."""
    from oat.common.p2n_new_capabilities import resolve_update_schedule
    training = cfg.training
    micro = int(cfg.dataloader.batch_size)
    accumulation = int(training.gradient_accumulate_every)
    cap = training.get("max_train_steps")
    if train_windows is None:
        if cap is None:
            return {"note": "needs the dataset size (run --preflight)"}
        batches = int(cap)
    else:
        import torch
        from accelerate.data_loader import BatchSamplerShard
        from oat.workspace.train_p2n_new import _LimitedBatchSampler
        sampler = torch.utils.data.BatchSampler(range(int(train_windows)), batch_size=micro, drop_last=True)
        if cap is not None:
            sampler = _LimitedBatchSampler(sampler, int(cap) * world_size)
        batches = len(BatchSamplerShard(sampler, num_processes=world_size, process_index=0))
    schedule = resolve_update_schedule(batches, int(training.num_epochs), accumulation, max_train_steps=cap,
                                       warmup_steps=0)
    schedule["lr_warmup_steps"] = int(training.lr_warmup_steps)
    limit = training.get("max_optimizer_steps")
    schedule.update({
        "micro_batch_per_rank": micro, "world_size": int(world_size),
        "effective_batch": micro * int(world_size) * accumulation,
        "max_optimizer_steps": limit,
        "cosine_horizon": int(limit) if limit is not None else schedule["planned_optimizer_updates"],
        "expected_optimizer_updates": (schedule["planned_optimizer_updates"] if limit is None
                                       else min(int(limit), schedule["planned_optimizer_updates"])),
        "min_lr_ratio": float(training.min_lr_ratio)})
    if train_windows is not None:
        schedule["train_windows"] = int(train_windows)
        schedule["windows_seen_per_epoch"] = batches * micro * int(world_size)
    return schedule


def describe(cfg, world_size, output, mode):
    policy = cfg.policy
    return {"mode": mode, "variant": cfg.variant, "task": cfg.task_type, "policy": policy._target_,
            "model_size": policy.model_size, "output": str(output), "world_size": world_size,
            "lambda_ki": policy.lambda_ki, "lora": [policy.lora_rank, policy.lora_alpha],
            "self_past": {key: policy.get(key) for key in ("self_past_p", "self_past_warmup_steps",
                                                            "self_past_ramp_steps", "self_past_chunk_size",
                                                            "self_past_temperature", "self_past_topk")},
            "optimizer": _plain(cfg.optimizer), "ema": _plain(cfg.ema),
            "schedule": expected_schedule(cfg, None, world_size),
            "assets": {key: policy.get(key) for key in ("pi05_weights", "pi05_sha256", "spm_path", "spm_sha256",
                                                        "tokenizer_checkpoint")},
            "dataset": cfg.task.policy.dataset.zarr_path, "logging_mode": cfg.logging.mode}


# ------------------------------------------------------------------------------- preflight
def check_output_dir(output, resume):
    output = Path(output)
    if output.exists() and not output.is_dir():
        raise ValueError(f"Output path is not a directory: {output}")
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError(f"Fresh output directory must be empty: {output}")
    ancestor = output
    while not ancestor.exists():
        ancestor = ancestor.parent
    if not os.access(ancestor, os.W_OK):
        raise PermissionError(f"Output location is not writable: {ancestor}")
    return {"output": str(output), "exists": output.exists(), "resume": bool(resume)}


def inspect_dataset(cfg):
    """Zarr schema and the exact train/validation episode split (no image arrays are read)."""
    from scripts.train_p2n_new import inspect_dataset as inspect_existing
    return inspect_existing(cfg)


def validate_tokenizer_source(cfg):
    """OAT provenance: same zarr/split/horizon, FSQ [8,5,5,5,5], 16x7 <-> 8x5, EMA weights + normalizer."""
    from scripts.train_p2n_latent_flow import validate_tokenizer_source as validate_oat
    return validate_oat(cfg)


def check_assets(cfg, *, full_hash=False):
    from oat.model.vla.pi05_checkpoint import (PI05_SHA256, default_pi05_path, file_sha256, read_header,
                                               resolved_blob_sha256)
    policy = cfg.policy
    manifest = json.loads(ASSETS_MANIFEST.read_text()) if ASSETS_MANIFEST.is_file() else None
    files = (manifest or {}).get("files", {})
    report = {"manifest": str(ASSETS_MANIFEST) if manifest else None}
    if policy.model_size == "full":
        path = Path(policy.get("pi05_weights") or default_pi05_path())
        if not path.is_file():
            raise FileNotFoundError(f"pi05_base weights not found: {path} (run scripts/fetch_p2n_vla_assets.py)")
        digest, method = (None, None) if full_hash else (resolved_blob_sha256(path), "hf_blob_name")
        if digest is None:
            digest, method = file_sha256(path), "full_file"
        pinned = policy.get("pi05_sha256") or PI05_SHA256
        recorded = files.get("pi05_base/model.safetensors", {}).get("sha256")
        if digest != pinned or digest != PI05_SHA256 or (recorded and recorded != digest):
            raise ValueError(f"pi05 weights sha256 {digest} does not match the pinned {PI05_SHA256}")
        tensors = len(read_header(path))
        if tensors != EXPECTED_PI05_TENSORS:
            raise ValueError(f"pi05 header lists {tensors} tensors, expected {EXPECTED_PI05_TENSORS}")
        report["pi05"] = {"path": str(path), "sha256": digest, "method": method, "tensors": tensors,
                          "bytes": path.stat().st_size}
    spm = Path(str(policy.get("spm_path") or ""))
    if not spm.is_file():
        raise FileNotFoundError(f"PaliGemma SentencePiece model not found: {spm}")
    digest = file_sha256(spm)
    recorded = files.get("paligemma_tokenizer.model", {}).get("sha256")
    if (policy.get("spm_sha256") and digest != policy.spm_sha256) or (recorded and recorded != digest):
        raise ValueError(f"SentencePiece sha256 {digest} does not match the pinned value")
    report["sentencepiece"] = {"path": str(spm), "sha256": digest}
    return report


def check_prompts(cfg):
    """Every instruction must fit max_len with every state bin at its longest (3 digits)."""
    from omegaconf import OmegaConf
    from oat.model.vla.paligemma_prompt import PaliGemmaTokenizer, PromptBuilder
    from oat.model.vla.state_transforms import PromptStateSpec
    prompt = _plain(cfg.policy.prompt)
    builder = PromptBuilder(PaliGemmaTokenizer(str(cfg.policy.spm_path)), prompt["instruction_source"],
                            int(prompt["max_len"]))
    spec = PromptStateSpec(list(prompt["state_keys"]), dict(prompt.get("state_transforms") or {}))
    shapes = {key: tuple(value["shape"]) for key, value in OmegaConf.to_container(cfg.shape_meta.obs).items()}
    dims = spec.output_dim(shapes)
    lengths = builder.worst_case_lengths(dims)
    worst = max(lengths.values())
    if worst > builder.max_len:
        raise ValueError(f"Worst-case prompt needs {worst} tokens > max_len {builder.max_len}: {lengths}")
    uids = sorted(uid for uid in lengths if uid is not None)
    if uids:
        import torch
        ids, valid = builder.build(torch.tensor(uids), torch.full((len(uids), dims), 255, dtype=torch.long))
        if valid.sum(1).tolist() != [lengths[uid] for uid in uids]:
            raise ValueError("PromptBuilder.build disagrees with the worst-case token counts")
    return {"max_len": builder.max_len, "state_dims": dims, "worst_case_tokens": worst,
            "headroom": builder.max_len - worst,
            "per_uid": {str(uid): length for uid, length in lengths.items()}}


def check_resume(cfg, world_size, split):
    from oat.workspace.train_p2n_vla import TrainP2NVLAWorkspace
    path = Path(cfg.training.resume_checkpoint)
    payload = TrainP2NVLAWorkspace.read_payload(path)
    TrainP2NVLAWorkspace.validate_resume_payload(payload, cfg, world_size=world_size)
    saved = payload["training"]["dataset_split"]["identity"]
    for key in ("train_episode_ids", "validation_episode_ids"):
        if saved.get(key) != split.get(key):
            raise ValueError(f"Resume {key} differ from the current dataset split")
    counters = payload["training"]["counters"]
    return {"checkpoint": str(path.resolve()), "counters": counters,
            "world_size": payload["training"]["world_size"]}


def model_report(cfg):
    """Build the policy on CPU and account for parameters, groups, artifacts and dtypes."""
    import hydra
    import torch
    from oat.workspace.train_p2n_vla import validate_parameter_partition
    started = time.monotonic()
    policy = hydra.utils.instantiate(cfg.policy)
    seconds = time.monotonic() - started
    dtype = getattr(policy, "dtype", None)
    if dtype is not None and dtype != torch.float32:
        raise ValueError(f"policy.dtype must be float32 (runner casts observations to it), got {dtype}")
    optimizer = policy.get_optimizer(**_plain(cfg.optimizer))
    named, clip = validate_parameter_partition(policy, optimizer)

    def count(parameters):
        unique = {id(p): p for p in parameters}
        return sum(p.numel() for p in unique.values())

    parameters = list(policy.parameters())
    trainable = [p for _, p in named]
    frozen_keys = policy.frozen_base_keys()
    state = policy.state_dict()
    artifact = policy.artifact_state_dict()
    trainable_names = {name for name, _ in named}
    trainable_dtypes = sorted({str(p.dtype) for p in trainable})
    if trainable_dtypes != ["torch.float32"]:
        raise ValueError(f"Trainable parameters must be fp32 masters, got {trainable_dtypes}")

    def nbytes(tensors):
        return sum(t.numel() * t.element_size() for t in tensors)

    artifact_bytes = nbytes(artifact.values())
    trainable_bytes = nbytes(trainable)
    report = {
        "build_seconds": seconds,
        "parameters": {"total": count(parameters), "trainable": count(trainable),
                       "frozen": count(parameters) - count(trainable)},
        "optimizer_groups": [{"name": group.get("name"), "lr": group["lr"],
                              "weight_decay": group.get("weight_decay"), "tensors": len(group["params"]),
                              "elements": count(group["params"])} for group in optimizer.param_groups],
        "clip_groups": {name: {"tensors": len(params), "elements": count(params)} for name, params in clip.items()},
        "frozen_base_keys": {"count": len(frozen_keys),
                             "elements": sum(state[key].numel() for key in frozen_keys if key in state)},
        "artifact": {"keys": len(artifact), "bytes": artifact_bytes,
                     "trainable_keys": len(trainable_names & set(artifact))},
        "estimated_bytes": {
            "snapshot": artifact_bytes - trainable_bytes // 2,       # trainable tensors stored in bf16
            "resume_checkpoint": 2 * artifact_bytes + 2 * trainable_bytes,  # live + EMA + Adam moments
        },
        "dtype": str(dtype),
        "metadata": {key: value for key, value in policy.artifact_metadata().items()
                     if key in ("variant", "pi05", "tokenizer", "sentencepiece", "prompt", "parameters")},
    }
    del policy, optimizer
    return report


def preflight(cfg, output, world_size=2, *, build_model=False, full_hash=False):
    resume = bool(cfg.training.get("resume"))
    report = {"policy_family": POLICY_FAMILY, "variant": cfg.variant, "task": cfg.task_type,
              "output": check_output_dir(output, resume), "world_size": world_size}
    split = inspect_dataset(cfg)
    report["dataset_split"] = split
    sources = {"assets": check_assets(cfg, full_hash=full_hash)}
    if resume:
        sources["resume"] = check_resume(cfg, world_size, split)
    else:
        sources["oat"] = validate_tokenizer_source(cfg)
    sources["prompts"] = check_prompts(cfg)
    report["sources"] = sources
    report["schedule"] = expected_schedule(cfg, split["train"]["windows"], world_size)
    if build_model:
        report["model"] = model_report(cfg)
        report["status"] = "CPU preflight passed (schema, split, provenance, assets, prompts, model build)"
    else:
        report["status"] = "Light preflight passed (schema, split, provenance, assets, prompts)"
    return report


# ---------------------------------------------------------------------------------- launch
def check_gpu_selection(devices, allow_busy=False):
    from scripts.train_p2n_latent_flow import check_gpu_selection as check
    return check(devices, allow_busy)


def _worker(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    from omegaconf import OmegaConf
    from oat.workspace.train_p2n_vla import TrainP2NVLAWorkspace
    TrainP2NVLAWorkspace(OmegaConf.load(args.worker_config), output_dir=args.output).run()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--task", choices=TASKS, default="libero")
    parser.add_argument("--tokenizer", help="Frozen OAT checkpoint (policy.tokenizer_checkpoint)")
    parser.add_argument("--pi05", help="pi05_base model.safetensors (policy.pi05_weights)")
    parser.add_argument("--spm", help="PaliGemma SentencePiece model (policy.spm_path)")
    parser.add_argument("--output", type=Path, help="Run directory (fresh runs need it empty)")
    parser.add_argument("--devices", help="CUDA_VISIBLE_DEVICES for the run, e.g. 0,1")
    parser.add_argument("--num-processes", type=int, help="torchrun processes; defaults to the number of --devices")
    parser.add_argument("--resume", type=Path, help="checkpoints/latest.ckpt of an earlier run (same world size)")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU (gloo); for tiny-model smoke runs")
    parser.add_argument("--allow-busy-gpus", action="store_true")
    parser.add_argument("--full-hash", action="store_true", help="Re-hash pi05 weights instead of trusting the HF blob name")
    parser.add_argument("--report", type=Path, help="Also write the preflight report JSON here")
    parser.add_argument("--probe-steps", type=int, default=3, help="--probe: optimizer steps to time")
    parser.add_argument("--probe-self-past-step", type=int,
                        help="--probe: self-past counter (default warmup+ramp, i.e. p=self_past_p; 0 gives p=0)")
    parser.add_argument("--config-dir", type=Path, help=argparse.SUPPRESS)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Compose and validate the configuration only")
    mode.add_argument("--preflight", action="store_true", help="Full CPU preflight including a model build")
    mode.add_argument("--probe", action="store_true", help="torchrun memory/throughput probe")
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--worker-config":
        return _worker(argv)
    overrides = []
    if "--" in argv:
        index = argv.index("--")
        argv, overrides = argv[:index], argv[index + 1:]
    parser = build_parser()
    args = parser.parse_args(argv)
    devices = None
    if args.devices:
        devices = [item.strip() for item in args.devices.split(",")]
        if not devices or any(not item.isdigit() for item in devices) or len(set(devices)) != len(devices):
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
    generated = []
    for value, key in ((args.tokenizer, "policy.tokenizer_checkpoint"), (args.pi05, "policy.pi05_weights"),
                       (args.spm, "policy.spm_path")):
        if value is not None:
            generated.append(f"{key}={json.dumps(str(Path(value).expanduser().resolve()))}")
    if args.resume:
        generated += ["training.resume=true",
                      f"training.resume_checkpoint={json.dumps(str(args.resume.expanduser().resolve()))}"]
    if args.probe:
        generated += ["training.probe.enabled=true", f"training.probe.optimizer_steps={args.probe_steps}",
                      "logging.mode=disabled"]
        if args.probe_self_past_step is not None:
            generated.append(f"training.probe.self_past_step={args.probe_self_past_step}")
    cfg = compose_config(args.variant, args.task, [*generated, *overrides], config_dir=args.config_dir)
    from omegaconf import OmegaConf
    if args.output is not None:
        output = args.output.expanduser().resolve()
    elif args.resume:
        # Resume into the run that wrote <run>/checkpoints/latest.ckpt.
        checkpoint = args.resume.expanduser().resolve()
        output = checkpoint.parent.parent if checkpoint.parent.name == "checkpoints" else checkpoint.parent
    elif args.probe:
        output = ROOT / "output/probe" / f"{args.variant}_{args.task}_{datetime.now():%Y%m%d_%H%M%S}"
    else:
        output = ROOT / "output/training" / f"{args.variant}_{args.task}_seed{cfg.seed}"
    mode = "dry_run" if args.dry_run else "preflight" if args.preflight else "probe" if args.probe else "train"
    summary = describe(cfg, world_size, output, mode)
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print(json.dumps(summary, indent=2, default=str))
    if args.dry_run:
        print("Configuration resolved and validated only: data, weights, prompts and GPU memory are unverified.")
        return summary
    if devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    if args.cpu:
        os.environ["ACCELERATE_USE_CPU"] = "true"
    report = preflight(cfg, output, world_size, build_model=args.preflight, full_hash=args.full_hash)
    print(json.dumps(report, indent=2, default=str))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, default=str) + "\n")
    if args.preflight:
        return report
    if not args.cpu and devices:
        check_gpu_selection(devices, args.allow_busy_gpus)
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_resume_{stamp}" if args.resume else ""
    resolved = output / f"p2n_vla_resolved{suffix}.yaml"
    resolved.write_text(OmegaConf.to_yaml(cfg, resolve=True))
    (output / f"p2n_vla_preflight{suffix}.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("HF_HOME", "/workspace/.hf_home")
    command = [sys.executable]
    if world_size > 1:
        command += ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node", str(world_size)]
    command += [str(Path(__file__).resolve()), "--worker-config", str(resolved), "--output", str(output)]
    print("Launching:", " ".join(command), flush=True)
    subprocess.run(command, cwd=str(ROOT), check=True, env=env)
    if args.probe:
        probe = output / "probe.json"
        if probe.is_file():
            print(probe.read_text())
    return report


if __name__ == "__main__":
    main()
