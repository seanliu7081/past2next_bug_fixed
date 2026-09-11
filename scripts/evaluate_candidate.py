"""Noninteractive, reproducible candidate evaluation with immutable result folders.

Example:
    MUJOCO_GL=egl /venv/oat/bin/python scripts/evaluate_candidate.py \
        --checkpoint /path/to/policy.ckpt --output-dir output/eval/baseline \
        --protocol corrected --n-test 100 --seed 42 --episode-start-seed 1000
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
import traceback

ROOT_DIR = Path(__file__).resolve().parents[1]
# This checkout must win over sibling editable installs of the oat package.
sys.path.insert(0, str(ROOT_DIR))


def atomic_json(path, value):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_source_metadata(cfg):
    """Record the external action-tokenizer checkpoint used by the policy."""
    from omegaconf import OmegaConf
    checkpoint = OmegaConf.select(cfg, "policy.action_tokenizer.checkpoint")
    if checkpoint is None:
        raise ValueError("Policy configuration is missing its action-tokenizer checkpoint")
    path = Path(str(checkpoint)).expanduser().resolve()
    return {"tokenizer_checkpoint": str(path), "tokenizer_sha256": sha256(path)}


def load_primary_policy(args, overrides):
    from oat.policy.base_policy import BasePolicy
    return BasePolicy.from_checkpoint(
        str(args.checkpoint), return_configuration=True,
        weights=args.weights, policy_overrides=overrides,
    )


def wilson_interval(successes, trials):
    if trials == 0:
        return [None, None]
    z = 1.959963984540054
    rate = successes / trials
    denominator = 1 + z * z / trials
    center = (rate + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials * trials)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def summarize_records(records):
    per_task = {}
    for name in sorted({r["task_name"] for r in records}):
        task_records = [r for r in records if r["task_name"] == name]
        successes = sum(int(r["success"]) for r in task_records)
        trials = len(task_records)
        per_task[name] = {
            "successes": successes, "trials": trials,
            "success_rate": successes / trials,
            "wilson_95_interval": wilson_interval(successes, trials),
            "mean_policy_steps": sum(r["policy_steps"] for r in task_records) / trials,
        }
    successes = sum(int(r["success"]) for r in records)
    trials = len(records)
    return {
        "successes": successes,
        "trials": trials,
        "success_rate": successes / trials if trials else None,
        "macro_task_success_rate": sum(t["success_rate"] for t in per_task.values()) / len(per_task) if per_task else None,
        "wilson_95_interval": wilson_interval(successes, trials),
        "interval_note": "Episode-level Wilson interval; repeated evaluations of the same initial states are not independent new environments.",
        "per_task": per_task,
    }


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path, help="Must not already exist; results are never overwritten")
    p.add_argument("--protocol", choices=("legacy", "corrected", "official"), default="corrected")
    p.add_argument("--n-test", type=int, default=500, help="Total episodes across selected tasks")
    p.add_argument("--n-parallel-envs", type=int, default=10)
    p.add_argument("--n-test-vis", type=int, default=0)
    p.add_argument("--tasks", help="Comma-separated LIBERO task indices or exact names; default all tasks")
    p.add_argument("--seed", type=int, default=42, help="Policy sampling and crop RNG seed")
    p.add_argument("--episode-start-seed", type=int, default=1000)
    p.add_argument("--init-state-offset", type=int, default=0, help="First per-task official initial-state index; indices never wrap")
    p.add_argument("--temperature", type=float)
    p.add_argument("--topk", type=int)
    p.add_argument("--use-k-tokens", type=int)
    p.add_argument("--n-action-steps", type=int)
    p.add_argument("--crop-mode", choices=("checkpoint", "center"), default="center",
                   help="Center crops by default; checkpoint uses its saved crop flag with current encoder defaults")
    p.add_argument("--weights", choices=("ema", "model"), default="ema")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-episode-steps", type=int, default=550,
                   help="Policy steps, excludes settling; retain 550 for baseline comparisons")
    p.add_argument("--dry-run", action="store_true", help="Load policy and save full provenance/schedule without creating simulators")
    return p


def validate_args(args):
    for name in ("n_test", "n_parallel_envs", "threads", "max_episode_steps"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not 0 <= args.n_test_vis <= args.n_test:
        raise ValueError("--n-test-vis must be between 0 and --n-test")
    if args.init_state_offset < 0 or (args.protocol != "official" and args.init_state_offset):
        raise ValueError("A nonnegative initial-state offset is only supported with --protocol official")
    if args.temperature is not None and (not math.isfinite(args.temperature) or args.temperature < 0):
        raise ValueError("Temperature must be finite and nonnegative; zero means greedy")
    for name in ("topk", "use_k_tokens", "n_action_steps"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.seed < 0 or args.episode_start_seed < 0 or args.episode_start_seed + args.n_test >= 2 ** 32:
        raise ValueError("Seeds must fit NumPy's unsigned 32-bit seed range")
    if args.seed >= 2 ** 32:
        raise ValueError("Policy seed must fit NumPy's unsigned 32-bit seed range")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)


def checkpoint_policy_overrides(args):
    """Override evaluation behavior while preserving the native checkpoint crop size."""
    overrides = {}
    if args.crop_mode == "center":
        overrides["obs_encoder"] = {"vision_encoder": {"eval_fixed_crop": True}}
    if args.n_action_steps is not None:
        overrides["n_action_steps"] = args.n_action_steps
    return overrides


def crop_inference_settings(policy, args):
    randomizers = [
        {"module": name, "class": type(module).__module__ + "." + type(module).__name__,
         "num_crops": module.num_crops, "crop_size": [module.crop_height, module.crop_width]}
        for name, module in policy.named_modules()
        if all(hasattr(module, attr) for attr in ("crop_height", "crop_width", "num_crops"))
    ]
    return {"crop_mode": args.crop_mode, "crop_randomizers": randomizers}


def git_text(*args):
    result = subprocess.run(["git", "-C", str(ROOT_DIR), *args], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def official_initial_state_files(task_names):
    """Hash exactly the task state files resolved by LiberoEnv, without loading a simulator."""
    from libero.libero import benchmark, get_libero_path
    from oat.env.libero.env import task_name_to_suite_and_ids

    suites = {}
    files = {}
    for task_name in task_names:
        suite_name, task_index, _ = task_name_to_suite_and_ids[task_name]
        if suite_name not in suites:
            suites[suite_name] = benchmark.get_benchmark_dict()[suite_name]()
        task = suites[suite_name].get_task(task_index)
        if task.name != task_name:
            raise RuntimeError(f"Initial-state task mapping mismatch: {task_name} != {task.name}")
        path = Path(get_libero_path("init_states"), task.problem_folder, task.init_states_file).resolve()
        files[task_name] = {
            "path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size,
            "suite_name": suite_name, "suite_task_index": task_index,
        }
    return files


def main(argv=None):
    args = parser().parse_args(argv)
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    # Establish rendering/thread settings before importing torch, robosuite, etc.
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
    os.chdir(ROOT_DIR)
    started = time.monotonic()
    metadata = {
        "schema_version": 1,
        "status": "initializing",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {k: str(v) if isinstance(v, Path) else
                      [str(item) if isinstance(item, Path) else item for item in v] if isinstance(v, list) else v
                      for k, v in vars(args).items()},
        "python": sys.executable,
        "repository": str(ROOT_DIR),
        "checkpoint_sha256": sha256(args.checkpoint),
        "git_head": git_text("rev-parse", "HEAD"),
        "git_status": git_text("status", "--porcelain"),
    }
    atomic_json(args.output_dir / "metadata.json", metadata)
    runner = None
    try:
        import numpy as np
        import torch
        import hydra
        from omegaconf import OmegaConf
        from oat.env.libero.factory import get_subtasks
        from oat.env_runner.libero_runner import build_episode_schedule

        torch.set_num_threads(args.threads)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        overrides = checkpoint_policy_overrides(args)
        policy, cfg = load_primary_policy(args, overrides)
        base_policy_module = type(policy).__module__
        if policy.n_action_steps > int(cfg.horizon):
            raise ValueError("Execution stride exceeds the checkpoint prediction horizon")
        if args.use_k_tokens is not None and args.use_k_tokens > policy.max_seq_len:
            raise ValueError("Requested token count exceeds the checkpoint latent horizon")
        policy.to(torch.device(args.device))
        policy.eval()

        runner_config = OmegaConf.to_container(cfg.task.policy.env_runner, resolve=True)
        all_tasks = get_subtasks(runner_config["task_name"])
        task_names = None
        if args.tasks:
            task_names = []
            for selector in args.tasks.split(","):
                selector = selector.strip()
                if selector.isdigit():
                    index = int(selector)
                    if index >= len(all_tasks):
                        raise ValueError(f"Task index out of range: {index}")
                    task_names.append(all_tasks[index])
                elif selector in all_tasks:
                    task_names.append(selector)
                else:
                    raise ValueError(f"Unknown task: {selector}")
        selected_tasks = all_tasks if task_names is None else task_names
        runner_config.update({
            "n_test": args.n_test,
            "n_test_vis": args.n_test_vis,
            "n_parallel_envs": args.n_parallel_envs,
            "test_start_seed": args.episode_start_seed,
            "n_action_steps": policy.n_action_steps,
            "n_obs_steps": policy.n_obs_steps,
            "max_episode_steps": args.max_episode_steps,
            "protocol": args.protocol,
            "task_names": task_names,
            "init_state_offset": args.init_state_offset,
            "episode_records_path": str(args.output_dir / "episodes.jsonl"),
            "output_dir": str(args.output_dir),
        })
        schedule = build_episode_schedule(
            selected_tasks, args.n_test, min(args.n_parallel_envs, args.n_test),
            args.episode_start_seed, args.protocol, args.init_state_offset,
        )
        schedule_counts = Counter(r["task_name"] for r in schedule)
        atomic_json(args.output_dir / "schedule.json", schedule)
        atomic_json(args.output_dir / "resolved_config.json", OmegaConf.to_container(cfg, resolve=True))
        atomic_json(args.output_dir / "runner_config.json", runner_config)

        module_names = [
            "oat", type(policy).__module__, base_policy_module, "oat.policy.base_policy",
            "oat.policy.past2next", "oat.env.libero.env", "oat.env_runner.libero_runner",
            "oat.gymnasium_util.async_vector_env",
            "oat.perception.robomimic_vision_encoder", "oat.perception.crop_randomizer",
            "oat.tokenizer.oat.tokenizer",
            "robomimic.models.base_nets", "libero.libero", "libero.libero.envs.env_wrapper",
            "libero.libero.benchmark",
        ]
        module_paths = {}
        for name in module_names:
            module = importlib.import_module(name)
            path = getattr(module, "__file__", None)
            module_paths[name] = {"path": path, "sha256": sha256(path) if path and Path(path).is_file() else None}
        if not Path(module_paths["oat"]["path"]).resolve().is_relative_to(ROOT_DIR):
            raise RuntimeError("oat imported from the wrong checkout")
        versions = {}
        for package in ("torch", "numpy", "robomimic", "robosuite", "mujoco", "hydra-core", "gymnasium"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        tokenizer_metadata = tokenizer_source_metadata(cfg)
        source_hashes = {str(path.relative_to(ROOT_DIR)): sha256(path)
                         for directory in (ROOT_DIR / "oat", ROOT_DIR / "scripts")
                         for path in sorted(directory.rglob("*"))
                         if path.suffix in {".py", ".yaml"}}
        atomic_json(args.output_dir / "source_hashes.json", source_hashes)
        diff = git_text("diff", "--no-ext-diff", "HEAD")
        if diff is not None:
            (args.output_dir / "source_diff.patch").write_text(diff + "\n")
        effective_inference = {
            "temperature": policy.temperature if args.temperature is None else args.temperature,
            "topk": policy.topk if args.topk is None else args.topk,
            "use_k_tokens": policy.max_seq_len if args.use_k_tokens is None else args.use_k_tokens,
            "n_action_steps": policy.n_action_steps,
            "history_init": "zeros",
            **crop_inference_settings(policy, args),
        }
        metadata.update({
            "status": "ready" if args.dry_run else "running",
            "module_paths": module_paths,
            "package_versions": versions,
            **tokenizer_metadata,
            "effective_inference": effective_inference,
            "official_initial_state_files": official_initial_state_files(selected_tasks) if args.protocol == "official" else {},
            "task_episode_counts": dict(schedule_counts),
            "all_tasks_covered": set(schedule_counts) == set(all_tasks),
            "balanced_schedule": len(set(schedule_counts.values())) == 1,
            "settling_steps": 5 if args.protocol == "official" else 10,
            "settling_action": [0.0] * 7 if args.protocol == "official" else [0.0] * 6 + [-1.0],
            "episode_seed_applied": args.protocol != "legacy",
            "vector_autoreset": args.protocol == "legacy",
            "policy_steps_reliable": args.protocol != "legacy",
            "policy_steps_note": (
                "Legacy automatic resets can replace completed episode histories; policy_steps and mean_policy_steps are unreliable. Success-rate accumulation retains historical behavior."
                if args.protocol == "legacy" else
                "Automatic resets are disabled; step counts describe the retained evaluated episode."
            ),
            "legacy_note": "Legacy retains stale first observations, double reset, and historical seed-label behavior; it is not comparable to corrected/official scores." if args.protocol == "legacy" else None,
            "official_note": "Uses official saved initial states and five zero settling actions; policy-step budget is recorded independently." if args.protocol == "official" else None,
            "cuda_device": torch.cuda.get_device_name(policy.device) if policy.device.type == "cuda" else None,
            "torch_threads": torch.get_num_threads(),
            "load_seconds": time.monotonic() - started,
        })
        atomic_json(args.output_dir / "metadata.json", metadata)
        if args.dry_run:
            print(json.dumps({"status": "ready", "output_dir": str(args.output_dir), "effective_inference": effective_inference}, sort_keys=True))
            return 0

        runner = hydra.utils.instantiate(runner_config)
        inference_kwargs = {key: getattr(args, key) for key in ("temperature", "topk", "use_k_tokens") if getattr(args, key) is not None}
        # Construction and optional state-file loading must not consume the
        # policy RNG stream differently between paired evaluations.
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        rollout_started = time.monotonic()
        runner.run(policy, **inference_kwargs)
        rollout_seconds = time.monotonic() - rollout_started
        records = runner.last_episode_records
        if len(records) != args.n_test:
            raise RuntimeError(f"Expected {args.n_test} episode records, received {len(records)}")
        summary = summarize_records(records)
        summary.update({
            "protocol": args.protocol, "seed": args.seed,
            "checkpoint": str(args.checkpoint), "checkpoint_sha256": metadata["checkpoint_sha256"],
            "rollout_seconds": rollout_seconds,
            "episodes_per_second": len(records) / rollout_seconds,
            "max_episode_steps": args.max_episode_steps,
            "effective_inference": effective_inference,
            "balanced_schedule": metadata["balanced_schedule"],
            "all_tasks_covered": metadata["all_tasks_covered"],
            "policy_steps_reliable": metadata["policy_steps_reliable"],
            "policy_steps_note": metadata["policy_steps_note"],
        })
        atomic_json(args.output_dir / "summary.json", summary)
        metadata.update({"status": "complete", "total_seconds": time.monotonic() - started})
        atomic_json(args.output_dir / "metadata.json", metadata)
        print(json.dumps(summary, sort_keys=True))
        return 0
    except BaseException as error:
        metadata.update({"status": "failed", "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc(), "total_seconds": time.monotonic() - started})
        atomic_json(args.output_dir / "metadata.json", metadata)
        raise
    finally:
        if runner is not None:
            runner.close()


if __name__ == "__main__":
    raise SystemExit(main())
