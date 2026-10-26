#!/usr/bin/env python3
"""Evaluate a trained policy checkpoint in its DexMimicGen task (paired, seeded episodes).

The runner class follows the policy's capabilities: plain DexMGRunner (R1, DP), executed-past
(policies with ``record_executed_actions``), measured-state history (R1g), and the P2N
original-obs wrappers (R2/R3, which also restore through their own loader). Episode seeds are
``test_start_seed + i`` for i in [0, n_test), identical for every arm; ``--num-shards`` /
``--shard-index`` split them across processes (e.g. one per GPU) without changing the set.

Run with PYTHONPATH=<repo>:/workspace/past_action_robocasa/robosuite:/workspace/third_party/dexmimicgen,
MUJOCO_GL=egl, CUDA_VISIBLE_DEVICES=<policy GPU> and --egl-device-ids <probed EGL index>.
Dex work may only use smi GPUs 0,1,4,5 (EGL 3,2,7,6).
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dill  # noqa: E402
import hydra  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
import torch  # noqa: E402

from oat.common.hydra_util import register_new_resolvers  # noqa: E402
from oat.common.p2n_new_capabilities import policy_capability  # noqa: E402

RUNNERS = "oat.env_runner.dexmg_runner."


def load_policy(checkpoint, weights):
    with open(checkpoint, "rb") as stream:
        payload = torch.load(stream, map_location="cpu", pickle_module=dill, weights_only=False)
    policy_config = payload.get("policy_config") or payload["cfg"].get("policy", {})
    if policy_config.get("obs_encoder_type") == "original_fused":
        from oat.policy.p2n_new_original_obs import P2NNewOriginalObsPolicy, P2NStateGateNewOriginalObsPolicy
        cls = (P2NStateGateNewOriginalObsPolicy if policy_config.get("variant") == "p2n_state_gate_new"
               else P2NNewOriginalObsPolicy)
        policy, cfg = cls.from_checkpoint(checkpoint, return_configuration=True, weights=weights)
        return policy, cfg, True
    from oat.policy.base_policy import BasePolicy
    policy, cfg = BasePolicy.from_checkpoint(checkpoint, return_configuration=True, weights=weights)
    return policy, cfg, False


def runner_class(policy, is_p2n):
    executed = policy_capability(policy, "requires_execution_acknowledgement")
    state_history = policy_capability(policy, "requires_state_history")
    if is_p2n:
        return "P2NStateGateNewDexMGRunner" if state_history else "P2NNewDexMGRunner"
    if state_history:
        return "DexMGStateHistoryRunner"
    return "DexMGExecutedPastRunner" if executed else "DexMGRunner"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", choices=("ema", "model"), help="default: EMA when the run used EMA")
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--n-test-vis", type=int, default=4)
    parser.add_argument("--n-parallel-envs", type=int, default=25)
    parser.add_argument("--test-start-seed", type=int, default=100000)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--egl-device-ids", type=int, nargs="+", required=True)
    parser.add_argument("--max-episode-steps", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--topk", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset-init", choices=("val", "train"),
                        help="start from recorded initial states of the seed-42/val-0.05 split (diagnostic)")
    args = parser.parse_args()

    register_new_resolvers()
    policy, cfg, is_p2n = load_policy(args.checkpoint, args.weights)
    policy.to(torch.device(args.device)).eval()
    runner_cfg = OmegaConf.to_container(cfg.task.policy.env_runner, resolve=False)
    name = runner_class(policy, is_p2n)
    runner_cfg["_target_"] = RUNNERS + name
    runner_cfg.update(n_test=args.n_test, n_test_vis=args.n_test_vis, n_parallel_envs=args.n_parallel_envs,
                      test_start_seed=args.test_start_seed, shard_index=args.shard_index,
                      num_shards=args.num_shards, egl_device_ids=list(args.egl_device_ids),
                      n_obs_steps=int(cfg.n_obs_steps), n_action_steps=int(cfg.n_action_steps))
    if args.dataset_init:
        import zarr
        from oat.common.seq_sampler import get_val_mask
        dataset = cfg.task.policy.dataset
        n_episodes = len(zarr.open(str(runner_cfg["zarr_path"]), "r")["meta/episode_ends"])
        val = get_val_mask(n_episodes, val_ratio=float(dataset.val_ratio), seed=int(dataset.seed))
        chosen = np.flatnonzero(val if args.dataset_init == "val" else ~val)[: args.n_test]
        runner_cfg["dataset_init_episodes"] = [int(x) for x in chosen]
    if args.max_episode_steps:
        runner_cfg["max_episode_steps"] = args.max_episode_steps
    if "StateHistory" in name or "StateGate" in name:
        runner_cfg.update(state_history_steps=int(policy.state_history_steps),
                          state_history_keys=list(policy.state_history_keys))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runner = hydra.utils.instantiate(OmegaConf.create(runner_cfg), output_dir=str(args.output_dir))
    kwargs = {key: value for key, value in (("temperature", args.temperature), ("topk", args.topk))
              if value is not None}
    try:
        log = runner.run(policy, **kwargs)
    finally:
        runner.close()
    summary = {"checkpoint": str(args.checkpoint.resolve()), "runner": name,
               "task": runner_cfg["task_name"], "weights": args.weights or "default",
               "decoding_overrides": kwargs, "n_episodes": int(log["n_episodes"]),
               "success_rate": float(log["mean_success_rate"]), "shard": [args.shard_index, args.num_shards],
               "test_start_seed": args.test_start_seed, "n_test": args.n_test,
               "dataset_init": args.dataset_init,
               "episode_log": str(runner.episode_log)}
    out = args.output_dir / f"summary_shard{args.shard_index}of{args.num_shards}.json"
    out.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
