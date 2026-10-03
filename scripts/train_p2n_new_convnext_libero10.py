#!/usr/bin/env python3
"""Separate frozen ConvNeXt Nano launcher for LIBERO-10 and simulator evaluation."""
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

from scripts import train_p2n_new_convnext as nano

CONFIG_NAMES = {
    "p2n_new": "experimental/train_p2n_new_convnext_nano_libero10",
    "p2n_state_gate_new": "experimental/train_p2n_state_gate_new_convnext_nano_libero10",
}


def compose_config(variant, overrides=()):
    from hydra import compose, initialize_config_dir
    from oat.common.hydra_util import register_new_resolvers

    register_new_resolvers()
    with initialize_config_dir(config_dir=str(ROOT / "oat/config"), version_base=None):
        cfg = compose(config_name=CONFIG_NAMES[variant], overrides=list(overrides))
    validate_config(cfg, variant)
    return cfg


def validate_config(cfg, variant):
    gate = variant == "p2n_state_gate_new"
    class_name = "P2NStateGateNewPolicy" if gate else "P2NNewPolicy"
    if cfg.policy._target_ != f"oat.policy.p2n_new_convnext.{class_name}":
        raise ValueError("LIBERO Nano requires the corresponding additive ConvNeXt policy")
    if cfg._target_ != "oat.workspace.train_p2n_new_convnext.TrainP2NNewWorkspace":
        raise ValueError("LIBERO Nano requires the encoder-aware workspace")
    original_schema = copy.deepcopy(cfg)
    original_schema.policy._target_ = f"oat.policy.{variant}.{class_name}"
    nano.legacy.validate_config(original_schema, variant, "libero")
    if cfg.task.policy.name != "libero10" or cfg.task.policy.env_runner.task_name != "libero10":
        raise ValueError("This launcher requires the LIBERO-10 suite")
    if cfg.policy.get("dino_path") or cfg.policy.get("dino_revision"):
        raise ValueError("LIBERO Nano cannot also select DINO weights")
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
            raise ValueError(f"LIBERO Nano requires policy.{key}={expected!r}")
    expected_obs = {
        "agentview_rgb": [128, 128, 3], "robot0_eye_in_hand_rgb": [128, 128, 3],
        "robot0_eef_pos": [3], "robot0_eef_quat": [4],
        "robot0_gripper_qpos": [2], "task_uid": [1],
    }
    if {key: list(spec.shape) for key, spec in cfg.shape_meta.obs.items()} != expected_obs:
        raise ValueError("LIBERO-10 requires its two-camera, quaternion and two-finger observation schema")
    if list(cfg.shape_meta.action.shape) != [7]:
        raise ValueError("LIBERO-10 requires seven-dimensional actions")
    if not isinstance(cfg.task.policy.lazy_eval, bool):
        raise ValueError("task.policy.lazy_eval must be true or false")
    for name, value in (
        ("training.rollout_every", cfg.training.rollout_every),
        ("task.policy.env_runner.n_test", cfg.task.policy.env_runner.n_test),
        ("task.policy.env_runner.n_parallel_envs", cfg.task.policy.env_runner.n_parallel_envs),
        ("dataloader.batch_size", cfg.dataloader.batch_size),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if cfg.task.policy.env_runner.n_test_vis > cfg.task.policy.env_runner.n_test:
        raise ValueError("Evaluation videos cannot exceed evaluation episodes")


def inspect_simulator(cfg):
    """Check imports and assets without constructing an environment or renderer."""
    if cfg.task.policy.lazy_eval:
        return {"enabled": False}
    config_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", str(Path.home() / ".libero")))
    if not (config_dir / "config.yaml").is_file():
        raise FileNotFoundError(f"LIBERO configuration is missing: {config_dir / 'config.yaml'}")
    import hydra
    import importlib.util
    for package in ("libero", "robosuite", "mujoco", "OpenGL", "gymnasium"):
        if importlib.util.find_spec(package) is None:
            raise ImportError(f"LIBERO evaluation dependency is missing: {package}")
    from libero.libero import get_libero_path, benchmark

    hydra.utils.get_class(cfg.task.policy.env_runner._target_)
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    bddl_root = Path(get_libero_path("bddl_files"))
    assets = Path(get_libero_path("assets"))
    if not assets.is_dir():
        raise FileNotFoundError(f"LIBERO assets are missing: {assets}")
    for index in range(suite.n_tasks):
        task = suite.get_task(index)
        path = bddl_root / task.problem_folder / task.bddl_file
        if not path.is_file():
            raise FileNotFoundError(f"LIBERO task definition is missing: {path}")
        if cfg.task.policy.env_runner.protocol == "official":
            path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
            if not path.is_file():
                raise FileNotFoundError(f"LIBERO initial states are missing: {path}")
    return {
        "enabled": True, "tasks": suite.n_tasks,
        "rollout_every": int(cfg.training.rollout_every),
        "epoch_labels": "0, interval, 2*interval, ... (after each labeled epoch)",
        "episodes_per_evaluation": int(cfg.task.policy.env_runner.n_test),
        "parallel_envs": min(int(cfg.task.policy.env_runner.n_parallel_envs),
                             int(cfg.task.policy.env_runner.n_test)),
        "protocol": cfg.task.policy.env_runner.protocol,
        "assets": str(assets), "status": "Dependencies/assets checked; simulator rollout not run",
    }


def configure_training_devices(selected, *, render=True):
    """Keep precise CUDA UUID selection and map the renderer independently.

    EGL indices can differ from CUDA ordinals. The isolated probe enumerates
    device metadata without opening rendering contexts. Simulator wrappers use
    the selected EGL index only while constructing/forking their environments.
    """
    renderer = None
    if render and os.environ.get("MUJOCO_GL", "egl") == "egl":
        env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID")
        env.pop("CUDA_VISIBLE_DEVICES", None)
        result = subprocess.run(
            [sys.executable, "-m", "oat.common.libero_egl_devices"],
            cwd=str(ROOT), env=env, text=True, capture_output=True, check=True,
        )
        devices = json.loads(result.stdout.strip())
        uuid = selected[0]["uuid"].removeprefix("GPU-").lower()
        matches = [device for device in devices
                   if device["uuid"].removeprefix("GPU-").lower() == uuid]
        if len(matches) != 1:
            raise RuntimeError(f"Cannot identify one EGL renderer for GPU {selected[0]['index']}")
        renderer = matches[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu["uuid"] for gpu in selected)
    # Installed robosuite compares CUDA and EGL indices at import time. The
    # scoped simulator wrapper supplies matching renderer-only values instead.
    os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
    os.environ.pop("P2N_LIBERO_EGL_DEVICE_ID", None)
    if renderer is not None:
        os.environ["P2N_LIBERO_EGL_DEVICE_ID"] = str(renderer["egl_device_id"])
    return {"training_gpus": selected, "renderer": renderer}


def main(argv=None):
    from omegaconf import OmegaConf

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--worker-config":
        from oat.workspace.train_p2n_new_convnext import TrainP2NNewWorkspace
        parser = argparse.ArgumentParser()
        parser.add_argument("--worker-config", required=True)
        parser.add_argument("--output", required=True)
        args = parser.parse_args(argv)
        cfg = OmegaConf.load(args.worker_config)
        validate_config(cfg, cfg.variant)
        TrainP2NNewWorkspace(cfg, output_dir=args.output).run()
        return

    split = argv.index("--") if "--" in argv else len(argv)
    flags, overrides = argv[:split], argv[split + 1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(CONFIG_NAMES), required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--dataset")
    parser.add_argument("--convnext")
    parser.add_argument("--convnext-revision")
    parser.add_argument("--devices", help="GPU indices or UUIDs from nvidia-smi")
    parser.add_argument("--num-processes", type=int, default=2)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Resolve configuration only")
    mode.add_argument("--preflight", action="store_true", help="CPU model, dataset and simulator asset checks")
    args = parser.parse_args(flags)
    if args.num_processes < 1:
        parser.error("--num-processes must be positive")
    generated = []
    for value, key in (
        (args.tokenizer, "policy.tokenizer_checkpoint"),
        (args.dataset, "task.policy.dataset.zarr_path"),
        (args.convnext, "policy.convnext_path"),
        (args.convnext_revision, "policy.convnext_revision"),
    ):
        if value is not None:
            generated.append(f"{key}={json.dumps(str(value))}")
    if args.resume:
        generated.extend(["training.resume=true",
                          f"training.resume_checkpoint={json.dumps(str(args.resume.resolve()))}"])
    cfg = compose_config(args.variant, [*generated, *overrides])
    output = (args.output or ROOT / "output/training" /
              f"{args.variant}_convnext_nano_libero10_seed{cfg.seed}").resolve()
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print(json.dumps({
        "variant": cfg.variant, "task": "libero10", "vision": "convnext_nano",
        "output": str(output), "world_size": args.num_processes,
        "effective_batch": int(cfg.dataloader.batch_size) * args.num_processes *
                           int(cfg.training.gradient_accumulate_every),
        "lazy_eval": cfg.task.policy.lazy_eval, "eval_every": cfg.training.rollout_every,
        "mode": "dry_run" if args.dry_run else "preflight" if args.preflight else "training",
    }, indent=2))
    if args.dry_run:
        print("Configuration resolved only: no weights, dataset, simulator, GPU or W&B initialization.")
        return
    os.environ.setdefault("MUJOCO_GL", "egl")
    if os.environ["MUJOCO_GL"] in ("egl", "osmesa"):
        os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
    selected = None
    if not args.preflight:
        selected = nano.check_gpu_idle(args.devices, args.num_processes)
        configure_training_devices(selected, render=not cfg.task.policy.lazy_eval)
    evaluation = inspect_simulator(cfg)
    report = nano.preflight(cfg, output, args.num_processes)
    report["evaluation"] = evaluation
    print(json.dumps({"evaluation": evaluation}, indent=2))
    if args.preflight:
        return report
    nano.check_gpu_idle(",".join(gpu["uuid"] for gpu in selected), args.num_processes)
    output.mkdir(parents=True, exist_ok=True)
    resolved = output / "p2n_new_convnext_libero10_resolved.yaml"
    resolved.write_text(OmegaConf.to_yaml(cfg, resolve=True))
    (output / "p2n_new_convnext_libero10_preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    command = [sys.executable]
    if args.num_processes > 1:
        command += ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node", str(args.num_processes)]
    command += [str(Path(__file__).resolve()), "--worker-config", str(resolved), "--output", str(output)]
    subprocess.run(command, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
