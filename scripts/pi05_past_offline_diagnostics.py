#!/usr/bin/env python
"""Offline copycat diagnostics of a P2N-VLA flow snapshot (Pi05_past or its control) on held-out windows.

Run at every development snapshot of every arm (docs/PI05_PAST.md, experiment plan). On the run's own
held-out split, with the same sampling noise in every condition:

* reliance ratio: mean change of the predicted executed chunk when the history is swapped with another
  window's, over the change when the observation is swapped instead (normalized action units).
* gripper toggles: on windows whose demonstrated executed chunk toggles the gripper (relative to the last
  past command), the recall of a predicted toggle within --toggle-tolerance steps and its timing error;
  elsewhere, the rate of predicted toggles.
* innovation R^2 per chunk position: prediction of a[t+i] - a[t-1]. The demonstrations nearly copy the last
  command, so plain R^2 of the actions says little about what the head adds.

Example (one GPU):
    CUDA_VISIBLE_DEVICES=6 /venv/oat/bin/python scripts/pi05_past_offline_diagnostics.py \\
        --snapshot output/training/pi05_past_libero10_seed42/snapshots/upd-010000_model.ckpt --weights model

Writes <run>/eval/<snapshot>_<weights>_offline_diagnostics.json unless --output is given.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

N_EXECUTED = 8
GRIPPER = -1


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapshot", required=True, type=Path, help="snapshots/upd-NNNNNN_<weights>.ckpt")
    p.add_argument("--weights", choices=("ema", "model"), default="model")
    p.add_argument("--output", type=Path, help="JSON path (must not exist)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-windows", type=int, default=2000, help="Evenly spaced held-out windows")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--toggle-tolerance", type=int, default=2, help="Steps a predicted toggle may be off by")
    p.add_argument("--pi05", type=Path, help="pi05_base model.safetensors (default: the pinned HF cache path)")
    p.add_argument("--spm", type=Path, help="PaliGemma SentencePiece model (default: the path in the artifact)")
    return p


def first_toggles(gripper):
    """[B, 1 + T] gripper commands (last past command first) -> index in [0, T) of the first sign change, or -1."""
    import torch
    sign = torch.sign(gripper)
    change = (sign[:, 1:] != sign[:, :-1]) & (sign[:, 1:] != 0) & (sign[:, :-1] != 0)
    index = change.float().argmax(dim=1)
    return torch.where(change.any(dim=1), index, torch.full_like(index, -1))


class Accumulator:
    def __init__(self, horizon, action_dim):
        import torch
        self.change = {"history": 0.0, "observation": 0.0}
        self.windows = 0
        self.toggle = {"windows": 0, "recalled": 0, "timing_error": 0.0, "other_windows": 0, "false_toggles": 0}
        self.innovation = {name: torch.zeros(horizon, action_dim, dtype=torch.float64)
                           for name in ("sum", "square", "error")}
        self.innovation_windows = 0

    def add(self, normalize, past, valid, target, plain, history_swap, observation_swap, tolerance):
        import torch
        executed = slice(0, N_EXECUTED)
        base = normalize(plain[:, executed])
        for name, other in (("history", history_swap), ("observation", observation_swap)):
            self.change[name] += float((normalize(other[:, executed]) - base).flatten(1).norm(dim=1).sum())
        self.windows += plain.shape[0]
        last = valid[:, -1]
        if not last.any():
            return
        past, target, plain = past[last], target[last], plain[last]
        demonstrated = first_toggles(torch.cat((past[:, -1:, GRIPPER], target[:, executed, GRIPPER]), dim=1))
        predicted = first_toggles(torch.cat((past[:, -1:, GRIPPER], plain[:, executed, GRIPPER]), dim=1))
        has = demonstrated >= 0
        hit = has & (predicted >= 0) & ((predicted - demonstrated).abs() <= tolerance)
        self.toggle["windows"] += int(has.sum())
        self.toggle["recalled"] += int(hit.sum())
        self.toggle["timing_error"] += float((predicted - demonstrated).abs()[hit].sum())
        self.toggle["other_windows"] += int((~has).sum())
        self.toggle["false_toggles"] += int(((predicted >= 0) & ~has).sum())
        reference = normalize(past[:, -1:])
        truth, guess = normalize(target) - reference, normalize(plain) - reference
        self.innovation["sum"] += truth.sum(0).double().cpu()
        self.innovation["square"] += truth.square().sum(0).double().cpu()
        self.innovation["error"] += (guess - truth).square().sum(0).double().cpu()
        self.innovation_windows += int(last.sum())

    def summary(self):
        n = self.innovation_windows
        r2 = None
        if n > 1:
            spread = (self.innovation["square"] - self.innovation["sum"].square() / n).sum(dim=1)
            r2 = [float(1 - e / s) if s > 0 else None for e, s in zip(self.innovation["error"].sum(dim=1), spread)]
        toggle = self.toggle
        history = self.change["history"] / max(self.windows, 1)
        observation = self.change["observation"] / max(self.windows, 1)
        return {
            "windows": self.windows,
            "reliance": {"history_swap_change": history, "observation_swap_change": observation,
                         "ratio": history / observation if observation > 0 else None},
            "gripper_toggles": {
                "toggle_windows": toggle["windows"],
                "recall": toggle["recalled"] / toggle["windows"] if toggle["windows"] else None,
                "timing_error": toggle["timing_error"] / toggle["recalled"] if toggle["recalled"] else None,
                "false_toggle_rate": (toggle["false_toggles"] / toggle["other_windows"]
                                      if toggle["other_windows"] else None)},
            "innovation_r2": r2, "innovation_windows": n,
        }


def _roll(value):
    if isinstance(value, dict):
        return {key: _roll(item) for key, item in value.items()}
    return value.roll(1, dims=0)


def main(argv=None):
    args = parser().parse_args(argv)
    if args.max_windows < 2 or args.batch_size < 2 or args.toggle_tolerance < 0:
        raise ValueError("--max-windows and --batch-size need >= 2 (swaps pair windows); --toggle-tolerance >= 0")
    snapshot = args.snapshot.expanduser().resolve()
    run = snapshot.parent.parent if snapshot.parent.name in ("snapshots", "checkpoints") else snapshot.parent
    output = (args.output or run / "eval" / f"{snapshot.stem}_{args.weights}_offline_diagnostics.json")
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
    import hydra
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from scripts.evaluate_p2n_vla import read_payload, policy_class

    payload = read_payload(snapshot)
    key = {"ema": "ema_model", "model": "model"}[args.weights]
    if key not in payload["state_dicts"]:
        raise ValueError(f"Snapshot has no {key!r} weights; use --weights accordingly")
    policy = policy_class(payload).from_checkpoint(
        str(snapshot), base_weights=None if args.pi05 is None else str(args.pi05), weights=args.weights,
        device=args.device, spm_path=None if args.spm is None else str(args.spm))
    policy.eval()
    dataset = hydra.utils.instantiate(payload["cfg"]["task"]["policy"]["dataset"]).get_validation_dataset()
    if len(dataset) < 2:
        raise ValueError("The held-out split has fewer than 2 windows")
    # Evenly spaced windows in a fixed shuffled order, so a swap partner usually comes from another episode.
    indices = np.unique(np.linspace(0, len(dataset) - 1, min(args.max_windows, len(dataset))).round().astype(int))
    indices = np.random.default_rng(args.seed).permutation(indices)
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=args.batch_size, shuffle=False)
    normalizer = policy.action_normalizer["action"]

    def normalize(value):
        return normalizer.normalize(value.float()).float()

    def to_device(value):
        if isinstance(value, dict):
            return {name: to_device(item) for name, item in value.items()}
        return value.to(policy.device) if isinstance(value, torch.Tensor) else value

    accumulator = Accumulator(int(policy.horizon), int(policy.action_dim))
    for batch_index, batch in enumerate(loader):
        batch = to_device(batch)
        if batch["action"].shape[0] < 2:
            continue
        obs, past, valid = batch["obs"], batch["past_action"].float(), batch["past_action_valid"].bool()

        def predict(observation, history, history_valid):
            torch.manual_seed(args.seed * 1_000_003 + batch_index)  # the same sampling noise in every condition
            with torch.no_grad():
                return policy.predict_action(observation, past_actions=history,
                                             past_action_valid=history_valid)["action_pred"].float()

        plain = predict(obs, past, valid)
        accumulator.add(normalize, past, valid, batch["action"].float(), plain,
                        predict(obs, _roll(past), _roll(valid)), predict(_roll(obs), past, valid),
                        args.toggle_tolerance)
    report = accumulator.summary()
    report.update({"snapshot": str(snapshot), "weights": args.weights, "variant": getattr(policy, "variant", None),
                   "use_past": bool(getattr(policy, "use_past", False)), "seed": args.seed,
                   "toggle_tolerance": args.toggle_tolerance, "held_out_windows": len(dataset)})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps(report, sort_keys=True))
    return report


if __name__ == "__main__":
    main()
