#!/usr/bin/env python
"""Evaluate a P2N-VLA EMA snapshot (or resume checkpoint) on LIBERO-10 into an immutable folder.

The policy is restored with ``<PolicyClass>.from_checkpoint`` (frozen pi05 backbone from the
base weights, everything else from the artifact) and the runner is rebuilt from the run's
embedded task config (``P2NNewLiberoRunner`` or ``P2NStateGateNewLiberoRunner``). No crop
override exists for these policies; decoding is greedy (T=0) unless ``--temperature`` is set.

With EGL on CUDA, the simulators render on the policy's own GPU: its EGL index is matched by CUDA
UUID (EGL indices need not equal CUDA ordinals) and any MUJOCO_EGL_DEVICE_ID is ignored.

Example (one GPU; run one process per GPU to evaluate several snapshots):
    CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl /venv/oat/bin/python scripts/evaluate_p2n_vla.py \\
        --snapshot output/training/p2n_vla_s42/snapshots/upd-030000_ema.ckpt \\
        --protocol official --n-test 500 --seed 44 --episode-start-seed 3000

Writes summary.json (per-task success with Wilson 95% intervals), episodes.jsonl,
metadata.json, source_hashes.json, schedule.json, runner_config.json and resolved_config.json.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

ROOT_DIR = Path(__file__).resolve().parents[1]
# This checkout must win over sibling editable installs of the oat package (e.g. /workspace/past_action).
sys.path.insert(0, str(ROOT_DIR))

# Stdlib-only helpers shared with the legacy evaluator (schedule, Wilson summaries, hashing).
from scripts.evaluate_candidate import (  # noqa: E402
    atomic_json, git_text, official_initial_state_files, sha256, summarize_records)

CHECKPOINT_FORMAT = "p2n_vla_checkpoint_v1"
POLICY_TARGETS = {
    "oat.policy.p2n_vla.P2NVLAPolicy": "p2n_vla",
    "oat.policy.p2n_vla_state_gate.P2NVLAStateGatePolicy": "p2n_vla_state_gate",
    "oat.policy.pi05_ki_flow.PI05KIFlowPolicy": "pi05_ki_flow",
}
RUNNERS = {False: "P2NNewLiberoRunner", True: "P2NStateGateNewLiberoRunner"}
# Same runner classes, constructed with the simulators bound to one EGL device (P2N_LIBERO_EGL_DEVICE_ID).
SCOPED_RUNNER_MODULE = "oat.env_runner.p2n_new_convnext_libero10_runner"
INFERENCE_KEYS = ("temperature", "topk", "use_k_tokens")
DEFAULT_MAX_EPISODE_STEPS = 550


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapshot", "--checkpoint", dest="snapshot", required=True, type=Path,
                   help="snapshots/upd-NNNNNN_ema.ckpt or checkpoints/latest.ckpt")
    p.add_argument("--output-dir", type=Path,
                   help="Must not exist; default <run>/eval/<snapshot>_<weights>_<protocol>_n<N>_seed<S>_ep<E> "
                        "plus a suffix for every other result-affecting option (see default_output_dir)")
    p.add_argument("--protocol", choices=("corrected", "official"), default="corrected")
    p.add_argument("--n-test", type=int, default=100, help="Total episodes across the selected tasks")
    p.add_argument("--n-parallel-envs", type=int, default=10)
    p.add_argument("--n-test-vis", type=int, default=0)
    p.add_argument("--tasks", help="Comma-separated LIBERO-10 task indices or exact names; default all ten")
    p.add_argument("--seed", type=int, default=42, help="Policy sampling seed")
    p.add_argument("--episode-start-seed", type=int, default=1000)
    p.add_argument("--init-state-offset", type=int, default=0,
                   help="First official initial-state index per task (official protocol only)")
    p.add_argument("--force-gate", choices=("open", "closed"),
                   help="Gate variant only: evaluate with the HIST gate forced open/closed (mask-only change)")
    p.add_argument("--use-k-tokens", type=int, help="Decode only the first k OAT tokens (1..8)")
    p.add_argument("--temperature", type=float, help="Sampling temperature; 0 is greedy (the default policy setting)")
    p.add_argument("--topk", type=int)
    p.add_argument("--weights", choices=("ema", "model"), default="ema")
    p.add_argument("--pi05", type=Path, help="pi05_base model.safetensors (default: the pinned HF cache path)")
    p.add_argument("--spm", type=Path, help="PaliGemma SentencePiece model (default: the path in the artifact)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-episode-steps", type=int, default=DEFAULT_MAX_EPISODE_STEPS,
                   help="Policy steps per episode, settling excluded")
    p.add_argument("--dry-run", action="store_true",
                   help="Load the policy and write provenance and the schedule without creating simulators")
    return p


def validate_args(args):
    for name in ("n_test", "n_parallel_envs", "threads", "max_episode_steps"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not 0 <= args.n_test_vis <= args.n_test:
        raise ValueError("--n-test-vis must be between 0 and --n-test")
    if args.init_state_offset < 0 or (args.protocol != "official" and args.init_state_offset):
        raise ValueError("A nonnegative --init-state-offset is only supported with --protocol official")
    if args.temperature is not None and (not math.isfinite(args.temperature) or args.temperature < 0):
        raise ValueError("--temperature must be finite and nonnegative; 0 means greedy")
    if args.topk is not None and args.topk < 1:
        raise ValueError("--topk must be positive")
    if args.use_k_tokens is not None and not 1 <= args.use_k_tokens <= 8:
        raise ValueError("--use-k-tokens must lie in [1, 8] (8 OAT tokens per chunk)")
    if args.seed < 0 or args.seed >= 2 ** 32 or args.episode_start_seed < 0 \
            or args.episode_start_seed + args.n_test >= 2 ** 32:
        raise ValueError("Seeds must fit NumPy's unsigned 32-bit seed range")
    if not args.snapshot.is_file():
        raise FileNotFoundError(args.snapshot)
    for name in ("pi05", "spm"):
        value = getattr(args, name)
        if value is not None and not value.is_file():
            raise FileNotFoundError(f"--{name}: {value}")


def _task_selector_label(selector):
    items = [item.strip() for item in selector.split(",")]
    if all(item.isdigit() for item in items):
        return "-".join(items)
    return hashlib.sha256(",".join(items).encode()).hexdigest()[:10]


def default_output_dir(args):
    """``<run>/eval/<name>``; the name encodes every option that changes the results.

    Two evaluations that differ in any of them (episode seeds, task subset, initial-state
    offset, step budget, token budget, sampling, forced gate) get different folders, so
    paired runs on one snapshot (e.g. the official run and a ``--use-k-tokens 4`` run) never
    collide on the never-overwrite rule.
    """
    snapshot = args.snapshot
    run = snapshot.parent.parent if snapshot.parent.name in ("snapshots", "checkpoints") else snapshot.parent
    parts = [snapshot.stem, args.weights, args.protocol, f"n{args.n_test}", f"seed{args.seed}",
             f"ep{args.episode_start_seed}"]
    if args.init_state_offset:
        parts.append(f"init{args.init_state_offset}")
    if args.tasks:
        parts.append(f"tasks-{_task_selector_label(args.tasks)}")
    if args.max_episode_steps != DEFAULT_MAX_EPISODE_STEPS:
        parts.append(f"steps{args.max_episode_steps}")
    if args.use_k_tokens is not None:
        parts.append(f"k{args.use_k_tokens}")
    if args.temperature is not None:
        parts.append(f"T{args.temperature:g}")
    if args.topk is not None:
        parts.append(f"topk{args.topk}")
    if args.force_gate:
        parts.append(f"gate-{args.force_gate}")
    return run / "eval" / "_".join(parts)


def read_payload(path):
    import dill
    import torch
    payload = torch.load(path, map_location="cpu", pickle_module=dill, weights_only=False, mmap=True)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} snapshot/checkpoint")
    for key in ("cfg", "policy_config", "metadata", "state_dicts"):
        if key not in payload:
            raise ValueError(f"Checkpoint lacks {key!r}")
    return payload


def policy_class(payload):
    """Resolve the policy class lazily (P2N-VLA classes are imported only when needed)."""
    import hydra
    target = payload["policy_config"].get("_target_")
    if not target:
        raise ValueError("Checkpoint policy_config has no _target_")
    cls = hydra.utils.get_class(target)
    if target not in POLICY_TARGETS and getattr(cls, "policy_family", None) != "p2n_vla":
        raise ValueError(f"{target} is not a P2N-VLA policy")
    if not callable(getattr(cls, "from_checkpoint", None)):
        raise TypeError(f"{target} lacks from_checkpoint")
    return cls


def load_policy(args, payload):
    key = {"ema": "ema_model", "model": "model"}[args.weights]
    if key not in payload["state_dicts"]:
        raise ValueError(f"Checkpoint has no {key!r} weights (available: {sorted(payload['state_dicts'])}); "
                         "use --weights accordingly")
    cls = policy_class(payload)
    policy = cls.from_checkpoint(str(args.snapshot), base_weights=None if args.pi05 is None else str(args.pi05),
                                 weights=args.weights, device=args.device,
                                 spm_path=None if args.spm is None else str(args.spm))
    policy.eval()
    return policy


def apply_force_gate(policy, mode):
    """Eval-time gate diagnostic: only the HIST attention bias changes (open = no bias, closed = removed)."""
    if mode is None:
        return None
    from oat.common.p2n_new_capabilities import policy_capability
    if not policy_capability(policy, "supports_history_summary_gate"):
        raise ValueError("--force-gate needs the p2n_vla_state_gate variant")
    setter = getattr(policy, "set_history_gate_mode", None)
    if callable(setter):
        setter(mode)
        mechanism = "set_history_gate_mode"
    elif hasattr(policy, "history_gate_mode"):
        policy.history_gate_mode = mode
        mechanism = "history_gate_mode attribute (read per call by compute_log_gate)"
    else:
        raise AttributeError("The gate policy exposes neither set_history_gate_mode(mode) nor history_gate_mode")
    if getattr(policy, "history_gate_mode", mode) != mode:
        raise RuntimeError("The gate mode did not change")
    return {"mode": mode, "mechanism": mechanism}


def decodes_tokens(policy):
    """True for the autoregressive OAT-token heads; False for the PI0.5 flow-matching baseline.

    The flow head integrates a velocity field (``flow_num_steps`` Euler steps): its
    ``predict_action`` inherits the token keywords but never reads them.
    """
    flag = getattr(policy, "HAS_AR_HEAD", None)
    if flag is None:
        return getattr(policy, "flow_num_steps", None) is None
    return bool(flag)


def inference_kwargs(policy, args):
    """Only overrides the policy actually uses; nothing silently ignored.

    ``policy`` may also be the policy class (a pre-check before the weights are loaded);
    instance-only settings (``max_seq_len``, ``temperature``) are then checked later.
    """
    label = policy.__name__ if isinstance(policy, type) else type(policy).__name__
    requested = {name: getattr(args, name) for name in INFERENCE_KEYS if getattr(args, name) is not None}
    if requested and not decodes_tokens(policy):
        flags = ", ".join("--" + name.replace("_", "-") for name in requested)
        raise ValueError(f"{label} has no autoregressive token head (it samples actions by flow matching): "
                         f"{flags} would be silently ignored")
    accepted = inspect.signature(policy.predict_action).parameters
    for name in requested:
        if name not in accepted:
            raise ValueError(f"{label}.predict_action does not accept {name}")
    if "use_k_tokens" in requested:
        limit = getattr(policy, "max_seq_len", None)
        if limit is not None and requested["use_k_tokens"] > int(limit):
            raise ValueError(f"--use-k-tokens {requested['use_k_tokens']} exceeds the {int(limit)} OAT tokens per chunk")
    if "topk" in requested:
        temperature = requested.get("temperature", getattr(policy, "temperature", None))
        if temperature is not None and float(temperature) == 0:
            raise ValueError("--topk only affects sampling and has no effect at temperature 0 (greedy decoding); "
                             "pass --temperature > 0 as well")
    return requested


def effective_inference(policy, args, gate):
    tokens = decodes_tokens(policy)

    def pick(name, attribute):
        if not tokens:
            return None  # not applicable to the flow head (and rejected as an override)
        value = getattr(args, name)
        return getattr(policy, attribute, None) if value is None else value
    return {"decoder": "autoregressive_oat_tokens" if tokens else "flow_matching",
            "temperature": pick("temperature", "temperature"), "topk": pick("topk", "topk"),
            "use_k_tokens": pick("use_k_tokens", "max_seq_len"),
            "n_action_steps": int(policy.n_action_steps), "n_obs_steps": int(policy.n_obs_steps),
            "history_init": "zeros (acknowledged executed commands thereafter)",
            "flow_num_steps": getattr(policy, "flow_num_steps", None), "force_gate": gate,
            "crop_override": None}


def select_tasks(runner_config, selector):
    from oat.env.libero.factory import get_subtasks
    all_tasks = get_subtasks(runner_config["task_name"])
    if not selector:
        return all_tasks, None
    names = []
    for item in selector.split(","):
        item = item.strip()
        if item.isdigit():
            index = int(item)
            if index >= len(all_tasks):
                raise ValueError(f"Task index out of range: {index}")
            names.append(all_tasks[index])
        elif item in all_tasks:
            names.append(item)
        else:
            raise ValueError(f"Unknown task: {item}")
    if len(set(names)) != len(names):
        raise ValueError("--tasks must not repeat a task")
    return all_tasks, names


def build_runner_config(cfg, policy, args, task_names, output_dir):
    from oat.common.p2n_new_capabilities import policy_capability
    runner = copy.deepcopy(((cfg.get("task") or {}).get("policy") or {}).get("env_runner"))
    if not runner:
        raise ValueError("The checkpoint's task config has no env_runner")
    expected = RUNNERS[policy_capability(policy, "requires_state_history")]
    if not str(runner.get("_target_", "")).endswith("." + expected):
        raise ValueError(f"{type(policy).__name__} must be evaluated by {expected}, "
                         f"but the embedded config names {runner.get('_target_')}")
    runner.update({
        "n_test": args.n_test, "n_test_vis": args.n_test_vis, "n_parallel_envs": args.n_parallel_envs,
        "test_start_seed": args.episode_start_seed, "n_action_steps": int(policy.n_action_steps),
        "n_obs_steps": int(policy.n_obs_steps), "max_episode_steps": args.max_episode_steps,
        "protocol": args.protocol, "task_names": task_names, "init_state_offset": args.init_state_offset,
        "episode_records_path": str(Path(output_dir) / "episodes.jsonl"), "output_dir": str(output_dir),
    })
    return runner


def probe_egl_renderers():
    """EGL devices with their CUDA identity, enumerated in an isolated process without rendering.

    EGL indices need not match CUDA ordinals (on this 2x4090 host they are swapped), so a numeric
    ``MUJOCO_EGL_DEVICE_ID`` equal to the CUDA index renders on the other GPU.
    """
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID")
    env.pop("CUDA_VISIBLE_DEVICES", None)
    result = subprocess.run([sys.executable, "-m", "oat.common.libero_egl_devices"], cwd=str(ROOT_DIR), env=env,
                            text=True, capture_output=True, check=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


def resolve_renderer(device, probe=probe_egl_renderers):
    """The EGL device on the policy's physical GPU, matched by CUDA UUID; None unless rendering with EGL on CUDA."""
    import torch
    device = torch.device(device)
    if os.environ.get("MUJOCO_GL", "egl").lower().strip() != "egl" or device.type != "cuda":
        return None
    uuid = str(torch.cuda.get_device_properties(device).uuid).removeprefix("GPU-").lower()
    matches = [record for record in probe() if str(record["uuid"]).removeprefix("GPU-").lower() == uuid]
    if len(matches) != 1:
        raise RuntimeError(f"Cannot identify one EGL renderer for CUDA device {device} (uuid {uuid})")
    return dict(matches[0])


def scope_runner_to_renderer(runner_config, renderer):
    """Build the simulators on ``renderer``'s EGL device.

    robosuite reads ``MUJOCO_EGL_DEVICE_ID`` per rendering context and asserts at import that it appears in
    ``CUDA_VISIBLE_DEVICES``; the scoped wrappers set both to the EGL index only while the runner forks its
    simulators (the policy's CUDA context already exists by then).
    """
    if renderer is None:
        return runner_config
    os.environ["P2N_LIBERO_EGL_DEVICE_ID"] = str(int(renderer["egl_device_id"]))
    scoped = dict(runner_config)
    scoped["_target_"] = f"{SCOPED_RUNNER_MODULE}.{str(runner_config['_target_']).rsplit('.', 1)[1]}"
    return scoped


def source_hashes():
    return {str(path.relative_to(ROOT_DIR)): sha256(path)
            for directory in (ROOT_DIR / "oat", ROOT_DIR / "scripts")
            for path in sorted(directory.rglob("*")) if path.suffix in {".py", ".yaml"}}


def asset_metadata(policy, args):
    from oat.model.vla.pi05_checkpoint import default_pi05_path, resolved_blob_sha256
    pi05 = Path(args.pi05) if args.pi05 is not None else default_pi05_path()
    return {"pi05": {"path": str(pi05), "sha256_hf_blob": resolved_blob_sha256(pi05) if pi05.exists() else None,
                     "policy_recorded_sha256": getattr(policy, "pi05_sha256", None)},
            "sentencepiece": {"path": getattr(policy, "spm_path", None),
                              "sha256": getattr(policy, "spm_sha256", None)},
            "tokenizer": copy.deepcopy(getattr(policy, "_tokenizer_metadata", None))}


def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(argv=None):
    args = parser().parse_args(argv)
    args.snapshot = args.snapshot.expanduser().resolve()
    validate_args(args)
    args.output_dir = (args.output_dir or default_output_dir(args)).expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    # Rendering and thread settings must exist before torch/robosuite are imported.
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
    os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
    os.chdir(ROOT_DIR)
    started = time.monotonic()
    metadata = {
        "schema_version": 1, "status": "initializing", "created_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "python": sys.executable, "repository": str(ROOT_DIR), "checkpoint": str(args.snapshot),
        "checkpoint_sha256": sha256(args.snapshot), "git_head": git_text("rev-parse", "HEAD"),
        "git_status": git_text("status", "--porcelain"),
        "environment": {key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "MUJOCO_GL",
                                                               "MUJOCO_EGL_DEVICE_ID", "HF_HOME")},
    }
    atomic_json(args.output_dir / "metadata.json", metadata)
    if os.environ["MUJOCO_GL"].lower().strip() == "egl" and str(args.device).startswith("cuda"):
        # The renderer is resolved from the policy's GPU below; a stale numeric id would only trip
        # robosuite's import-time CUDA_VISIBLE_DEVICES check (the user's value stays in metadata).
        os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
    runner = None
    try:
        import hydra
        import torch
        from oat.env_runner.libero_runner import build_episode_schedule

        torch.set_num_threads(args.threads)
        seed_everything(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        payload = read_payload(args.snapshot)
        cfg = payload["cfg"]
        run_metadata = payload["metadata"].get("training_run") or {}
        # Fail fast on overrides the policy class cannot honour, before the multi-GB restore.
        cls = policy_class(payload)
        inference_kwargs(cls, args)
        if args.force_gate is not None and not getattr(cls, "supports_history_summary_gate", False):
            raise ValueError("--force-gate needs the p2n_vla_state_gate variant")
        policy = load_policy(args, payload)
        gate = apply_force_gate(policy, args.force_gate)
        kwargs = inference_kwargs(policy, args)
        runner_template = ((cfg.get("task") or {}).get("policy") or {}).get("env_runner") or {}
        all_tasks, task_names = select_tasks(runner_template, args.tasks)
        selected = all_tasks if task_names is None else task_names
        runner_config = build_runner_config(cfg, policy, args, task_names, args.output_dir)
        embedded_runner_target = runner_config["_target_"]
        renderer = resolve_renderer(policy.device)
        runner_config = scope_runner_to_renderer(runner_config, renderer)
        schedule = build_episode_schedule(selected, args.n_test, min(args.n_parallel_envs, args.n_test),
                                          args.episode_start_seed, args.protocol, args.init_state_offset)
        counts = Counter(record["task_name"] for record in schedule)
        atomic_json(args.output_dir / "schedule.json", schedule)
        atomic_json(args.output_dir / "resolved_config.json", cfg)
        atomic_json(args.output_dir / "runner_config.json", runner_config)
        atomic_json(args.output_dir / "source_hashes.json", source_hashes())
        modules = {}
        for name in ("oat", type(policy).__module__, "oat.policy.p2n_vla_common", "oat.env_runner.p2n_new_runner",
                     SCOPED_RUNNER_MODULE, "oat.env_runner.libero_runner", "oat.env.libero.env", "libero.libero"):
            module = importlib.import_module(name)
            path = getattr(module, "__file__", None)
            modules[name] = {"path": path, "sha256": sha256(path) if path and Path(path).is_file() else None}
        if not Path(modules["oat"]["path"]).resolve().is_relative_to(ROOT_DIR):
            raise RuntimeError("oat was imported from the wrong checkout")
        versions = {}
        for package in ("torch", "numpy", "transformers", "sentencepiece", "robosuite", "mujoco", "hydra-core"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        inference = effective_inference(policy, args, gate)
        metadata.update({
            "status": "ready" if args.dry_run else "running",
            "variant": getattr(policy, "variant", None), "policy_class": f"{type(policy).__module__}.{type(policy).__name__}",
            "weights": args.weights, "optimizer_step": run_metadata.get("optimizer_step"),
            "ema": run_metadata.get("ema"), "checkpoint_kind": run_metadata.get("kind"),
            "policy_metadata": {key: value for key, value in payload["metadata"].items()
                                if key in ("variant", "variant_code", "model_size", "pi05", "tokenizer",
                                           "sentencepiece", "prompt", "parameters", "artifact_schema")},
            "assets": asset_metadata(policy, args), "module_paths": modules, "package_versions": versions,
            "effective_inference": inference, "runner_target": runner_config["_target_"],
            "embedded_runner_target": embedded_runner_target, "renderer": renderer,
            "official_initial_state_files": official_initial_state_files(selected) if args.protocol == "official" else {},
            "task_episode_counts": dict(counts), "all_tasks_covered": set(counts) == set(all_tasks),
            "balanced_schedule": len(set(counts.values())) == 1,
            "settling_steps": 5 if args.protocol == "official" else 10,
            "device": args.device,
            "cuda_device": torch.cuda.get_device_name(policy.device) if policy.device.type == "cuda" else None,
            "load_seconds": time.monotonic() - started,
        })
        atomic_json(args.output_dir / "metadata.json", metadata)
        if args.dry_run:
            print(json.dumps({"status": "ready", "output_dir": str(args.output_dir),
                              "effective_inference": inference}, sort_keys=True))
            return 0
        runner = hydra.utils.instantiate(runner_config)
        # Simulator construction must not shift the policy RNG between paired evaluations.
        seed_everything(args.seed)
        rollout_started = time.monotonic()
        runner.run(policy, **kwargs)
        rollout_seconds = time.monotonic() - rollout_started
        records = runner.last_episode_records
        if len(records) != args.n_test:
            raise RuntimeError(f"Expected {args.n_test} episode records, received {len(records)}")
        summary = summarize_records(records)
        summary.update({
            "variant": metadata["variant"], "protocol": args.protocol, "seed": args.seed,
            "episode_start_seed": args.episode_start_seed, "checkpoint": str(args.snapshot),
            "checkpoint_sha256": metadata["checkpoint_sha256"], "weights": args.weights,
            "optimizer_step": metadata["optimizer_step"], "force_gate": args.force_gate,
            "rollout_seconds": rollout_seconds, "episodes_per_second": len(records) / rollout_seconds,
            "max_episode_steps": args.max_episode_steps, "effective_inference": inference,
            "balanced_schedule": metadata["balanced_schedule"], "all_tasks_covered": metadata["all_tasks_covered"],
        })
        atomic_json(args.output_dir / "summary.json", summary)
        metadata.update({"status": "complete", "total_seconds": time.monotonic() - started})
        atomic_json(args.output_dir / "metadata.json", metadata)
        print(json.dumps(summary, sort_keys=True))
        return 0
    except BaseException as error:
        metadata.update({"status": "failed", "error": f"{type(error).__name__}: {error}",
                         "traceback": traceback.format_exc(), "total_seconds": time.monotonic() - started})
        atomic_json(args.output_dir / "metadata.json", metadata)
        raise
    finally:
        if runner is not None:
            runner.close()


if __name__ == "__main__":
    raise SystemExit(main())
