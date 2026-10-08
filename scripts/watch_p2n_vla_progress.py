#!/usr/bin/env python
"""Live whole-run progress bar for a P2N-VLA run, read from <run>/logs.jsonl.

  /venv/oat/bin/python scripts/watch_p2n_vla_progress.py output/training/p2n_vla_libero10_s42

Read-only: it never touches the training processes, so it is safe to start, stop (Ctrl-C) and
restart at any time. One bar from the run's current optimizer update to training.max_optimizer_steps
(from the newest p2n_vla_resolved*.yaml), with epoch, loss, samples/s and the latest in-training
LIBERO-10 success rate. The bar pauses during the ~40 min rollouts (every rollout_every epochs) and
says so; it flags a run that has written nothing for --stale-minutes outside a rollout.

A resume restarts at the last checkpoint, so when the resumed run writes its first record the bar
may step back (by up to ~1k updates) and restart its rate/ETA estimate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import tqdm


def read_config(run):
    """(max_optimizer_steps, rollout_every, in-training rollouts enabled) from the newest resolved config."""
    configs = sorted(run.glob("p2n_vla_resolved*.yaml"), key=lambda path: path.stat().st_mtime)
    if not configs:
        return 30000, 50, True
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(configs[-1])
    return (int(cfg.training.max_optimizer_steps), int(cfg.training.get("rollout_every") or 0),
            not bool(cfg.task.policy.get("lazy_eval", True)))


class Progress:
    def __init__(self, total, rollout_every, rollouts, stale_minutes):
        self.total, self.rollout_every, self.rollouts = total, rollout_every, rollouts
        self.stale_seconds = 60.0 * stale_minutes
        self.step = 0
        self.epoch = self.loss = self.samples_per_sec = None
        self.success_rate = self.success_step = None
        self.in_rollout = self.done = False
        self.last_record = time.monotonic()
        self.bar = None

    def apply(self, record):
        """Fold one logs.jsonl record into the state; returns the update step it reports, if any."""
        self.last_record = time.monotonic()
        event = record.get("event")
        if event == "train_step":
            self.in_rollout = False
            self.epoch = record.get("epoch", self.epoch)
            self.loss = record.get("train/loss", self.loss)
            self.samples_per_sec = record.get("samples_per_sec", self.samples_per_sec)
            return int(record["optimizer_step"])
        if event == "epoch":
            completed = int(record["epoch"]) + 1  # records are 0-indexed
            step = int(record["optimizer_step"])
            self.in_rollout = self.rollouts and (
                (self.rollout_every and completed % self.rollout_every == 0) or step >= self.total)
            self.done = step >= self.total and not self.in_rollout
            return step
        if event == "rollout":
            self.in_rollout = False
            self.success_rate = record.get("rollout/success_rate", record.get("mean_success_rate"))
            self.success_step = int(record["optimizer_step"])
            self.done = self.success_step >= self.total
        return None

    def postfix(self):
        parts = []
        if self.epoch is not None:
            parts.append(f"ep {self.epoch}")
        if self.loss is not None:
            parts.append(f"loss {self.loss:.3f}")
        if self.samples_per_sec is not None:
            parts.append(f"{self.samples_per_sec:.1f} samp/s")
        if self.success_rate is not None:
            parts.append(f"SR@{self.success_step / 1000:g}k {self.success_rate:.1%}")
        idle = time.monotonic() - self.last_record
        if self.in_rollout:
            parts.append(f"LIBERO eval running ({idle / 60:.0f} of ~40 min)")
        elif idle > self.stale_seconds:
            parts.append(f"NO NEW RECORDS FOR {idle / 60:.0f} min (still loading, or stopped?)")
        return " | ".join(parts)

    def open_bar(self, initial):
        if self.bar is not None:
            self.bar.close()
        self.bar = tqdm.tqdm(total=self.total, initial=initial, unit="upd", desc="P2N-VLA", dynamic_ncols=True)
        self.step = initial

    def advance(self, step):
        if step < self.step:  # a resume replaying from its checkpoint
            self.open_bar(step)
        elif step > self.step:
            self.bar.update(step - self.step)
            self.step = step

    def refresh(self):
        self.bar.set_postfix_str(self.postfix(), refresh=False)
        self.bar.refresh()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="Run directory (contains logs.jsonl)")
    parser.add_argument("--poll", type=float, default=1.0, help="Seconds between checks for new records")
    parser.add_argument("--stale-minutes", type=float, default=10.0)
    args = parser.parse_args()
    log = args.run / "logs.jsonl"
    while not log.is_file():
        print(f"waiting for {log} ...", flush=True)
        time.sleep(5)
    progress = Progress(*read_config(args.run), args.stale_minutes)
    with open(log) as stream:
        pending, initial = "", 0
        for line in stream:  # catch up on what is already there
            if not line.endswith("\n"):  # the trainer is mid-write; finish it in the follow loop
                pending = line
                break
            try:
                step = progress.apply(json.loads(line))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
            initial = step if step is not None else initial
        progress.open_bar(initial)
        progress.last_record = time.monotonic()
        try:
            while not progress.done:
                chunk = stream.readline()
                if not chunk:
                    progress.refresh()
                    time.sleep(args.poll)
                    continue
                pending += chunk
                if not pending.endswith("\n"):  # the trainer is mid-write
                    continue
                line, pending = pending, ""
                try:
                    step = progress.apply(json.loads(line))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
                if step is not None:
                    progress.advance(step)
            progress.refresh()
            print("\nTraining finished.", flush=True)
        except KeyboardInterrupt:
            pass
        finally:
            progress.bar.close()


if __name__ == "__main__":
    main()
