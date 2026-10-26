#!/usr/bin/env python3
"""Simulator gates for the DexMimicGen port, run against a converted zarr.

  orientation  (V3) reset_to(states[0]) renders vs zarr frame 0, unflipped and flipped
  parity       (V4) live replay from states[0]: obs after each stored action vs the next zarr row
                    (state ports incl. rot6d <=1e-5, images); obs are force-updated like the data
  state-replay (V4) reset_to(each of the last N recorded states) -> any _check_success
  open-loop    (V4) reset_to(states[0]) then the stored actions -> success rate (replay ceiling)
  seeding      (V6) same seed -> identical initial sim state across processes; different seeds differ
  arm-order    (V4) +x on the first arm's delta slice moves robot0 (= right arm)
  tokenizer    (V7) per --tokenizer checkpoint, on the validation split (seed 42, val 0.05):
                    per-part normalized reconstruction MSE, constant-dim error, and open-loop
                    replay with tokenized 16-step chunks (first 8 executed) vs native replay

Environment: PYTHONPATH=<repo>:/workspace/past_action_robocasa/robosuite:/workspace/third_party/dexmimicgen
MUJOCO_GL=egl, and MUJOCO_EGL_DEVICE_ID / --egl-device set to an allowed EGL index (GPUs 0,1,4,5 ->
EGL 3,2,7,6). Results are appended as JSON lines to --report.
"""
import argparse
import gzip
import json
from multiprocessing import get_context
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import zarr  # noqa: E402


def load_models(zarr_path):
    root = zarr.open(str(zarr_path), "r")
    sidecar = Path(zarr_path).with_name(root.attrs["model_file_sidecar"])
    with gzip.open(sidecar, "rt") as stream:
        return [json.loads(line)["model_file"] for line in stream]


def episode_bounds(root):
    ends = root["meta/episode_ends"][:]
    return np.concatenate(([0], ends[:-1])), ends


def make_env(args, render, **kwargs):
    from oat.env.dexmg.env import DexMGEnv
    root = zarr.open(str(args.zarr), "r")
    return DexMGEnv(task_name=root.attrs["dexmg_task"], zarr_path=args.zarr, enable_render=render,
                    render_gpu_device_id=args.egl_device if render else -1, **kwargs)


def pick_episodes(n_episodes, count, seed=0):
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(n_episodes, size=min(count, n_episodes), replace=False).tolist())


def gate_orientation(args):
    root = zarr.open(str(args.zarr), "r")
    starts, _ = episode_bounds(root)
    models = load_models(args.zarr)
    env = make_env(args, render=True, image_flip=False)
    results = {}
    for episode in pick_episodes(len(starts), args.episodes):
        row = int(starts[episode])
        obs = env.reset_to(models[episode], root["data/sim_state"][row])
        for camera in env.camera_names:
            key = f"{camera}_image"
            stored = root["data"][key][row].astype(np.float32)
            rendered = obs[key].astype(np.float32)
            entry = results.setdefault(key, {"unflipped": [], "flipped": []})
            entry["unflipped"].append(float(np.mean((rendered - stored) ** 2)))
            entry["flipped"].append(float(np.mean((rendered[::-1] - stored) ** 2)))
    summary = {key: {name: float(np.mean(values)) for name, values in entry.items()} for key, entry in results.items()}
    flips = {key: value["flipped"] < value["unflipped"] for key, value in summary.items()}
    consistent = len(set(flips.values())) == 1
    best = {key: min(value.values()) for key, value in summary.items()}
    return {"gate": "V3", "mse": summary, "image_flip": next(iter(flips.values())) if consistent else None,
            "pass": consistent and max(best.values()) < args.image_mse_tol}


def gate_parity(args):
    """Live replay from states[0]: observations after each stored action vs the next zarr row."""
    root = zarr.open(str(args.zarr), "r")
    starts, ends = episode_bounds(root)
    models = load_models(args.zarr)
    env = make_env(args, render=True)
    env.state_ports = list(dict.fromkeys(env.state_ports + sorted(
        key for key in root["data"].keys() if key.endswith("_joint_pos"))))
    worst, image_mse = {}, {}
    for episode in pick_episodes(len(starts), args.episodes, seed=1):
        start = int(starts[episode])
        steps = min(args.parity_steps, int(ends[episode]) - start - 1)
        obs = env.reset_to(models[episode], root["data/sim_state"][start])
        for t in range(steps + 1):
            row = start + t
            for port in env.state_ports:
                worst[port] = max(worst.get(port, 0.0), float(np.abs(obs[port] - root["data"][port][row]).max()))
            for camera in env.camera_names:
                key = f"{camera}_image"
                error = float(np.mean((obs[key].astype(np.float32) - root["data"][key][row]) ** 2))
                image_mse[key] = max(image_mse.get(key, 0.0), error)
            if t < steps:
                obs, *_ = env.step(root["data/action"][row])
    return {"gate": "V4-parity", "steps_per_episode": args.parity_steps, "max_abs_error": worst,
            "max_image_mse": image_mse,
            "pass": max(worst.values()) <= args.parity_tol and max(image_mse.values()) < args.image_mse_tol}


def _state_replay_worker(payload):
    args, episodes = payload
    root = zarr.open(str(args.zarr), "r")
    _, ends = episode_bounds(root)
    models = load_models(args.zarr)
    env = make_env(args, render=False)
    out = []
    for episode in episodes:
        # Demos can run a few steps past success (Box: success held ~14 states, then lost in the
        # final ~7); the runner's success is sticky, so any of the last N recorded states counts.
        success = False
        for row in range(int(ends[episode]) - 1, int(ends[episode]) - 1 - args.state_replay_tail, -1):
            env.reset_to(models[episode], root["data/sim_state"][row])
            if env.env._check_success():
                success = True
                break
        out.append(success)
    env.close()
    return out


def _open_loop_worker(payload):
    args, episodes = payload
    root = zarr.open(str(args.zarr), "r")
    starts, ends = episode_bounds(root)
    models = load_models(args.zarr)
    env = make_env(args, render=False)
    out = []
    for episode in episodes:
        start, end = int(starts[episode]), int(ends[episode])
        env.reset_to(models[episode], root["data/sim_state"][start])
        actions = root["data/action"][start:end].astype(np.float64)
        success, first = False, None
        for t, action in enumerate(actions):
            env.env.step(action)
            if env.env._check_success():
                success, first = True, t
                break
        out.append({"episode": episode, "success": success, "first_success_step": first, "length": end - start})
    env.close()
    return out


def _parallel(worker, args, episodes):
    shards = [episodes[index::args.workers] for index in range(args.workers)]
    shards = [shard for shard in shards if shard]
    with get_context("forkserver").Pool(len(shards)) as pool:
        return [item for part in pool.map(worker, [(args, shard) for shard in shards]) for item in part]


def gate_state_replay(args):
    root = zarr.open(str(args.zarr), "r")
    episodes = pick_episodes(len(root["meta/episode_ends"]), args.episodes)
    results = _parallel(_state_replay_worker, args, episodes)
    rate = float(np.mean(results))
    return {"gate": "V4-state-replay", "episodes": len(results), "success_rate": rate, "pass": rate >= 0.98}


def gate_open_loop(args):
    root = zarr.open(str(args.zarr), "r")
    episodes = pick_episodes(len(root["meta/episode_ends"]), args.episodes)
    started = time.time()
    results = _parallel(_open_loop_worker, args, episodes)
    rate = float(np.mean([item["success"] for item in results]))
    return {"gate": "V4-open-loop", "episodes": len(results), "success_rate": rate,
            "seconds": round(time.time() - started, 1), "failures": [item["episode"] for item in results if not item["success"]],
            "pass": None, "note": "replay ceiling; recorded for the comparison, not a pass/fail gate"}


def _seed_worker(payload):
    args, seeds = payload
    env = make_env(args, render=False)
    states = []
    for seed in seeds:
        env.configure_episode(seed)
        env.reset()
        states.append(env.env.sim.get_state().flatten().tolist())
    env.close()
    return states


def gate_seeding(args):
    seeds = [100000, 100001, 100002]
    with get_context("forkserver").Pool(2) as pool:
        first, second = pool.map(_seed_worker, [(args, seeds), (args, list(reversed(seeds)))])
    second = list(reversed(second))
    same = all(np.array_equal(a, b) for a, b in zip(first, second))
    distinct = not np.array_equal(first[0], first[1])
    return {"gate": "V6", "same_seed_identical_across_processes_and_order": same,
            "different_seeds_differ": distinct, "pass": same and distinct}


def gate_arm_order(args):
    env = make_env(args, render=False)
    env.configure_episode(7)
    env.reset()
    raw0 = env.raw_observation()
    action = np.zeros(env.action_space.shape)
    action[:12] = env.env.action_spec[0][:12] * 0  # hands stay at their zero command
    action[0] = 1.0  # +x on the first delta slice
    for _ in range(10):
        env.env.step(action)
    raw1 = env.raw_observation()
    prefixes = env.arm_prefixes
    moved = {prefix: float(raw1[f"{prefix}_eef_pos"][0] - raw0[f"{prefix}_eef_pos"][0]) for prefix in prefixes}
    return {"gate": "V4-arm-order", "dx_after_first_slice_+x": moved,
            "pass": abs(moved[prefixes[0]]) > 5 * abs(moved[prefixes[1]])}


# Action layout per embodiment for per-part reconstruction error (see the converter docstring).
PARTS = {
    "panda_dex": {"R_pos": (0, 3), "R_rot": (3, 6), "R_hand": (6, 12),
                  "L_pos": (12, 15), "L_rot": (15, 18), "L_hand": (18, 24)},
    "gr1": {"R_pos": (0, 3), "R_rot": (3, 6), "L_pos": (6, 9), "L_rot": (9, 12),
            "R_hand": (12, 18), "L_hand": (18, 24)},
}


def _replay_sequences_worker(payload):
    """Replay given per-episode action sequences from states[0]; return success per sequence."""
    args, jobs = payload
    root = zarr.open(str(args.zarr), "r")
    starts, _ = episode_bounds(root)
    models = load_models(args.zarr)
    env = make_env(args, render=False)
    out = []
    for episode, name, actions in jobs:
        env.reset_to(models[episode], root["data/sim_state"][int(starts[episode])])
        success = False
        for action in actions:
            env.env.step(np.asarray(action, dtype=np.float64))
            if env.env._check_success():
                success = True
                break
        out.append((episode, name, success))
    env.close()
    return out


def chunk_displacement(tokenizer, actions_all, starts, ends, episodes, horizon, exec_steps, device, embodiment):
    """Per executed chunk: commanded end-effector displacement error of the reconstruction.

    For OSC-delta arms (Panda-dex) the controller clips inputs to [-1, 1] and scales them by
    0.05 m / 0.5 rad per step; the error of the summed first ``exec_steps`` deltas is what an
    8-step closed-loop policy cannot correct until its next replan. Reported next to the expert's
    own per-chunk motion. GR1 actions are absolute, so this is Panda-only.
    """
    import torch
    if embodiment != "panda_dex":
        return None
    slices = {"R": (0, 3, 3, 6), "L": (12, 15, 15, 18)}
    pos_err, rot_err, pos_motion = [], [], []
    for episode in episodes:
        actions = np.asarray(actions_all[int(starts[episode]):int(ends[episode])], dtype=np.float32)
        windows = np.stack([np.concatenate((actions[t:t + horizon], np.repeat(actions[-1:], max(0, t + horizon - len(actions)), 0)))[:horizon]
                            for t in range(0, len(actions), exec_steps)])
        with torch.inference_mode():
            recon = tokenizer.autoencode(torch.from_numpy(windows).to(device)).cpu().numpy()
        expert = np.clip(windows[:, :exec_steps], -1, 1)
        recon = np.clip(recon[:, :exec_steps], -1, 1)
        for p0, p1, r0, r1 in slices.values():
            pos_err.append(0.05 * np.linalg.norm((recon[..., p0:p1] - expert[..., p0:p1]).sum(1), axis=-1))
            rot_err.append(0.5 * np.linalg.norm((recon[..., r0:r1] - expert[..., r0:r1]).sum(1), axis=-1))
            pos_motion.append(0.05 * np.linalg.norm(expert[..., p0:p1].sum(1), axis=-1))
    pos_err, rot_err, pos_motion = (np.concatenate(x) for x in (pos_err, rot_err, pos_motion))
    return {"pos_err_mm_median": float(1000 * np.median(pos_err)), "pos_err_mm_p90": float(1000 * np.quantile(pos_err, 0.9)),
            "rot_err_deg_median": float(np.degrees(np.median(rot_err))), "rot_err_deg_p90": float(np.degrees(np.quantile(rot_err, 0.9))),
            "expert_pos_motion_mm_median": float(1000 * np.median(pos_motion))}


def gate_tokenizer(args):
    import torch
    from oat.common.seq_sampler import get_val_mask
    from oat.tokenizer.oat.tokenizer import OATTok
    root = zarr.open(str(args.zarr), "r")
    starts, ends = episode_bounds(root)
    val = np.flatnonzero(get_val_mask(len(ends), val_ratio=args.val_ratio, seed=args.split_seed))
    replay = val[: args.episodes]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parts = PARTS[root.attrs["embodiment"]]
    actions_all = root["data/action"]
    results = []
    jobs = [(int(e), "native", actions_all[int(starts[e]):int(ends[e])]) for e in replay]
    for path in args.tokenizer:
        tokenizer = OATTok.from_checkpoint(str(path)).to(device).eval()
        horizon = int(tokenizer.decoder.sample_horizon)
        normalizer = tokenizer.normalizer["action"]
        squared = {name: [] for name in parts}
        constant_error = 0.0
        span = np.ptp(actions_all[: int(ends[-1])], axis=0) if args.constant_check else None
        for episode in val:
            actions = np.asarray(actions_all[int(starts[episode]):int(ends[episode])], dtype=np.float32)
            windows = np.stack([np.concatenate((actions[t:t + horizon], np.repeat(actions[-1:], max(0, t + horizon - len(actions)), 0)))[:horizon]
                                for t in range(0, len(actions), 4)])
            with torch.inference_mode():
                batch = torch.from_numpy(windows).to(device)
                recon = tokenizer.autoencode(batch)
                error = (normalizer.normalize(recon) - normalizer.normalize(batch)) ** 2
            error = error.cpu().numpy()
            for name, (lo, hi) in parts.items():
                squared[name].append(error[..., lo:hi].reshape(-1))
            if span is not None:
                constant = np.flatnonzero(span < 1e-4)
                if len(constant):
                    constant_error = max(constant_error, float(np.abs(recon.cpu().numpy()[..., constant] - windows[..., constant]).max()))
        mse = {name: float(np.mean(np.concatenate(values))) for name, values in squared.items()}
        chunk = chunk_displacement(tokenizer, actions_all, starts, ends, val, horizon, args.exec_steps, device,
                                   root.attrs["embodiment"])
        for episode in replay:
            actions = np.asarray(actions_all[int(starts[episode]):int(ends[episode])], dtype=np.float32)
            chunks = []
            for t in range(0, len(actions), args.exec_steps):
                window = np.concatenate((actions[t:t + horizon], np.repeat(actions[-1:], max(0, t + horizon - len(actions)), 0)))[:horizon]
                with torch.inference_mode():
                    recon = tokenizer.autoencode(torch.from_numpy(window[None]).to(device))[0].cpu().numpy()
                chunks.append(recon[: min(args.exec_steps, len(actions) - t)])
            jobs.append((int(episode), str(path), np.concatenate(chunks)))
        results.append({"tokenizer": str(path), "latent_horizon": int(tokenizer.latent_horizon),
                        "codebook": int(tokenizer.quantizer.codebook_size), "val_part_mse": mse,
                        "chunk_displacement": chunk, "constant_dim_max_abs_error": constant_error})
    shards = [jobs[i::args.workers] for i in range(args.workers) if jobs[i::args.workers]]
    with get_context("forkserver").Pool(len(shards)) as pool:
        replayed = [item for part in pool.map(_replay_sequences_worker, [(args, shard) for shard in shards]) for item in part]
    rate = {}
    for _, name, success in replayed:
        rate.setdefault(name, []).append(success)
    native = float(np.mean(rate["native"]))
    for entry in results:
        # Informational only: open-loop replay of delta actions is brittle (Box: +N(0, 0.02) noise on
        # the arm deltas already drops native replay 90% -> 25%), so it does not gate closed-loop use.
        entry["tokenized_replay_success"] = float(np.mean(rate[entry["tokenizer"]]))
    best = {name: min(entry["val_part_mse"][name] for entry in results) for name in parts}
    for entry in results:
        entry["within_1p25x_best_per_part"] = all(entry["val_part_mse"][n] <= 1.25 * best[n] for n in parts)
    return {"gate": "V7", "replay_episodes": len(replay), "native_replay_success": native,
            "exec_steps": args.exec_steps, "tokenizers": results,
            "selection": "smallest config within 1.25x of the best per-part MSE; chunk displacement and "
                         "open-loop replay are reported for judgement, not pass/fail",
            "pass": any(e["within_1p25x_best_per_part"] and e["constant_dim_max_abs_error"] < 1e-4 for e in results)}


GATES = {"orientation": gate_orientation, "parity": gate_parity, "state-replay": gate_state_replay,
         "open-loop": gate_open_loop, "seeding": gate_seeding, "arm-order": gate_arm_order, "tokenizer": gate_tokenizer}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("gates", nargs="+", choices=sorted(GATES))
    parser.add_argument("--zarr", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--egl-device", type=int, default=int(os.environ.get("MUJOCO_EGL_DEVICE_ID", -1)))
    parser.add_argument("--image-mse-tol", type=float, default=1.0)
    parser.add_argument("--parity-tol", type=float, default=1e-5)
    parser.add_argument("--state-replay-tail", type=int, default=30, help="recorded states checked from the end")
    parser.add_argument("--parity-steps", type=int, default=40, help="live replay steps checked per episode")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--tokenizer", type=Path, nargs="+", default=[], help="stage-1 checkpoints for the tokenizer gate")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--exec-steps", type=int, default=8)
    parser.add_argument("--constant-check", action="store_true", default=True)
    args = parser.parse_args()
    for name in args.gates:
        result = {"zarr": str(args.zarr), **GATES[name](args)}
        print(json.dumps(result), flush=True)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            with open(args.report, "a") as stream:
                stream.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
