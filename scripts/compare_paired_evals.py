#!/usr/bin/env python
"""Paired comparison of LIBERO evaluations on the same episodes (exact McNemar). Stdlib only.

Every episode of A is matched to the episode of B with the same (task_name, init_state_id, episode_seed). The two
episode sets must be identical, and so must the schedule: protocol, initial-state offset, parallel-environment count
(KITCHEN_SCENE4/6/8 initial states depend on the worker schedule) and evaluator seed. Reports both success
rates, the discordant counts (b: only A succeeded, c: only B succeeded), the paired difference B - A with a 95%
Wald interval and the exact two-sided McNemar p-value, overall and with --per-task per task.

  # per step: every evaluation step both runs scored (in-training official 500, or its offline re-score)
  /venv/oat/bin/python scripts/compare_paired_evals.py --runs output/training/pi05_past_libero10_seed42 \
      output/training/pi05_p2n_libero10_seed42
  # pooled: several blocks per arm, e.g. the final test's three corrected blocks of each chosen snapshot
  /venv/oat/bin/python scripts/compare_paired_evals.py --a A_ep5000 A_ep5500 A_ep6000 --b B_ep5000 B_ep5500 B_ep6000
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import sys

ROLLOUT = re.compile(r"rollout_epoch-\d+_upd-(\d+)(?:_r(\d+))?$")
RESCORE = re.compile(r"upd-(\d+)_model_model_official_n500_seed44_ep3000$")  # the in-training schedule, offline


def load(directories):
    """Episode outcomes keyed by (task_name, init_state_id, episode_seed), and the schedule of every block."""
    outcomes, schedules = {}, set()
    for directory in map(Path, directories):
        summary = json.loads((directory / "summary.json").read_text())
        if summary.get("failed"):
            raise ValueError(f"{directory}: the evaluation failed")
        runner_path = directory / "runner_config.json"
        runner = json.loads(runner_path.read_text()) if runner_path.is_file() else {}
        records = [json.loads(line) for line in (directory / "episodes.jsonl").read_text().splitlines()]
        if summary.get("trials") is not None and len(records) != summary["trials"]:
            raise ValueError(f"{directory}: {len(records)} episode records for {summary['trials']} trials (incomplete?)")
        schedules.add((summary.get("protocol"), summary.get("init_state_offset"),
                       summary.get("n_parallel_envs", runner.get("n_parallel_envs")), summary.get("seed")))
        for record in records:
            # The corrected protocol draws initial states from the episode seed (init_state_id is null).
            key = (record["task_name"], record.get("init_state_id"), int(record["episode_seed"]))
            if key in outcomes:
                raise ValueError(f"{directory}: duplicate episode {key}")
            outcomes[key] = bool(record["success"])
    return outcomes, schedules


def mcnemar_p(b, c):
    """Exact two-sided McNemar p-value: the binomial test of b against n = b + c at 1/2."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def compare(a, b):
    n = len(a)
    only_a = sum(a[k] and not b[k] for k in a)
    only_b = sum(b[k] and not a[k] for k in a)
    diff = (only_b - only_a) / n
    se = math.sqrt(max(only_a + only_b - (only_b - only_a) ** 2 / n, 0.0)) / n
    return {"n": n, "a": sum(a.values()) / n, "b": sum(b.values()) / n, "only_a": only_a, "only_b": only_b,
            "diff": diff, "ci95": [diff - 1.96 * se, diff + 1.96 * se], "p": mcnemar_p(only_a, only_b)}


def paired(dirs_a, dirs_b, per_task=False):
    (a, schedule_a), (b, schedule_b) = load(dirs_a), load(dirs_b)
    if set(a) != set(b):
        raise ValueError(f"Episode sets differ: {len(set(a) - set(b))} only in A, {len(set(b) - set(a))} only in B")
    if schedule_a != schedule_b:
        raise ValueError(f"Schedules differ (protocol, init offset, parallel envs, evaluator seed): "
                         f"{schedule_a} vs {schedule_b}")
    result = compare(a, b)
    if per_task:
        tasks = defaultdict(list)
        for key in a:
            tasks[key[0]].append(key)
        result["per_task"] = {task: compare({k: a[k] for k in keys}, {k: b[k] for k in keys})
                              for task, keys in sorted(tasks.items())}
    return result


def scored_steps(run):
    """Optimizer step -> evaluation directory: the newest successful in-training attempt, else the offline re-score."""
    steps, attempts = {}, {}
    for directory in sorted((Path(run) / "eval").iterdir()):
        rollout, rescore = ROLLOUT.search(directory.name), RESCORE.search(directory.name)
        if not (rollout or rescore) or not (directory / "summary.json").is_file():
            continue
        if json.loads((directory / "summary.json").read_text()).get("failed"):
            continue
        step = int((rollout or rescore).group(1))
        rank = (1, int(rollout.group(2) or 0)) if rollout else (0, 0)  # in-training attempts first, newest wins
        if step not in steps or rank > attempts[step]:
            steps[step], attempts[step] = directory, rank
    return steps


def line(label, r):
    return (f"{label:>14}  A {r['a']:.3f}  B {r['b']:.3f}  only A {r['only_a']:3d}  only B {r['only_b']:3d}  "
            f"B-A {100 * r['diff']:+5.1f} pts [{100 * r['ci95'][0]:+5.1f}, {100 * r['ci95'][1]:+5.1f}]  "
            f"p {r['p']:.3g}  (n {r['n']})")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs=2, metavar=("RUN_A", "RUN_B"), help="pair every evaluation step of two runs")
    parser.add_argument("--a", nargs="+", help="evaluation directories of arm A (pooled)")
    parser.add_argument("--b", nargs="+", help="evaluation directories of arm B (pooled), the same schedules as A")
    parser.add_argument("--per-task", action="store_true")
    parser.add_argument("--json", type=Path, help="also write the results here")
    args = parser.parse_args(argv)
    if bool(args.runs) == bool(args.a or args.b) or bool(args.a) != bool(args.b):
        parser.error("pass either --runs RUN_A RUN_B or both --a and --b")
    results = {}
    if args.runs:
        steps_a, steps_b = scored_steps(args.runs[0]), scored_steps(args.runs[1])
        print(f"A = {args.runs[0]}\nB = {args.runs[1]}")
        for step in sorted(set(steps_a) & set(steps_b)):
            results[step] = paired([steps_a[step]], [steps_b[step]], args.per_task)
            print(line(f"{step // 1000}k", results[step]))
        unpaired = sorted(set(steps_a) ^ set(steps_b))
        if unpaired:
            print(f"unpaired steps (only one run scored them): {unpaired}")
    else:
        results["pooled"] = paired(args.a, args.b, args.per_task)
        print(line("pooled", results["pooled"]))
    if args.per_task:
        for key, result in results.items():
            print(f"-- per task ({key})")
            for task, r in result["per_task"].items():
                print(line(task[:14], r), task)
    if args.json:
        args.json.write_text(json.dumps({str(k): v for k, v in results.items()}, indent=2) + "\n")
    return results


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
