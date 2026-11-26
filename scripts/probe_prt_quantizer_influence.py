"""CPU probe of one-token residual quantizer label assignment on real LIBERO windows.

Run with /venv/oat/bin/python scripts/probe_prt_quantizer_influence.py --data PATH --out NEW_DIR.
All errors are mean squared arm commands in dataset units. No simulator or image policy is used.
"""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import sklearn
import zarr
from scipy.special import softmax
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits


HISTORY, HORIZON, VOCAB = 7, 8, 32
SEEDS = (11, 23, 47)
SPLIT_SEED, CODEBOOK_SEED, TEACHER_SEED = 20261126, 71, 101
BATCH, EPOCHS, LR = 256, 20, 0.002
STRENGTHS = (0., 0.5, 2.)
COST_MIXES = (0., 0.5, 1.)


def digest(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def windows(path):
    root = zarr.open_group(str(path), mode="r")
    keys = ["action", "task_uid", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
    raw = {key: root["data"][key][:] for key in keys}
    ends = root["meta/episode_ends"][:]
    starts = np.r_[0, ends[:-1]]
    tasks = raw["task_uid"].reshape(-1)
    episode_task = tasks[starts]
    rows, episodes = [], []
    for ep, (start, end) in enumerate(zip(starts, ends)):
        if not np.all(tasks[start:end] == episode_task[ep]):
            raise ValueError(f"Episode {ep} has inconsistent task labels")
        times = np.arange(start, end - HORIZON + 1, HORIZON)
        if not len(times):
            raise ValueError(f"Episode {ep} is shorter than the executed prefix")
        rows.extend(times)
        episodes.extend([ep] * len(times))
    t, episode = np.asarray(rows), np.asarray(episodes)
    previous = t[:, None] + np.arange(-HISTORY, 0)
    valid = previous >= starts[episode, None]
    history = raw["action"][np.maximum(previous, starts[episode, None])] * valid[..., None]
    future = raw["action"][t[:, None] + np.arange(HORIZON), :6]
    robot_state = np.concatenate([raw[key][t] for key in keys[2:]], axis=1)
    task_flags = (tasks[t, None] == np.unique(episode_task)).astype(np.float32)
    features = np.column_stack((history.reshape(len(t), -1), robot_state, task_flags))
    hashes = {k: digest(v) for k, v in raw.items()}
    hashes["episode_ends"] = digest(ends)
    return history, future, features, episode, episode_task, hashes


def split_episodes(episode_task):
    rng = np.random.default_rng(SPLIT_SEED)
    splits = {k: [] for k in ("fit", "design", "dev", "test")}
    for task in np.unique(episode_task):
        ids = rng.permutation(np.flatnonzero(episode_task == task))
        n = len(ids) // 10
        if n < 1:
            raise ValueError("Need ten complete episodes per task")
        for name, subset in zip(splits, (ids[3*n:], ids[:n], ids[n:2*n], ids[2*n:3*n])):
            splits[name].extend(subset.tolist())
    return {k: sorted(v) for k, v in splits.items()}


def prior_residuals(history, future, fit):
    estimate = np.empty_like(future)
    for start in (0, 3):
        channels = slice(start, start + 3)
        x = history[fit, :, channels].transpose(0, 2, 1).reshape(-1, HISTORY).astype("float64")
        y = future[fit, :, channels].transpose(0, 2, 1).reshape(-1, HORIZON).astype("float64")
        gram = x.T @ x
        w = np.linalg.solve(gram + 1e-3 * gram.trace()/HISTORY * np.eye(HISTORY), x.T @ y)
        estimate[:, :, channels] = np.einsum("ntc,th->nhc", history[:, :, channels], w)
    return (future - estimate).reshape(len(history), -1)


def squared_cost(residual, centers):
    norms = np.square(centers).sum(axis=1)
    sse = np.square(residual).sum(axis=1, keepdims=True) + norms - 2 * residual @ centers.T
    return np.maximum(sse / residual.shape[1], 0.)


def head(x, labels, cost, train, seed, steps, mix=0.):
    """Fresh same-seed Adam head; pooled arms also get exactly 'steps' updates."""
    rng = np.random.default_rng(seed)
    w = (.01 * rng.standard_normal((x.shape[1], VOCAB))).astype(np.float32)
    m, v = np.zeros_like(w), np.zeros_like(w)
    for step in range(1, steps + 1):
        ids = rng.choice(train, size=BATCH, replace=True)
        p = softmax(x[ids] @ w, axis=1)
        ce = p.copy()
        ce[np.arange(BATCH), labels[ids]] -= 1
        d = cost[ids]
        risk = p * (d - (p * d).sum(axis=1, keepdims=True))
        grad = x[ids].T @ ((1 - mix) * ce + mix * risk) / BATCH
        m = .9*m + .1*grad
        v = .999*v + .001*grad**2
        w -= LR * (m / (1-.9**step)) / (np.sqrt(v / (1-.999**step)) + 1e-8)
    return w


def assignment_scores(x, raw_cost, fit, design, teacher, episode):
    p = softmax(x @ teacher, axis=1)
    nearest = raw_cost.argmin(axis=1)
    first_two = np.argsort(raw_cost[fit], axis=1)[:, :2]
    rows = np.arange(len(fit))
    d_gap = np.median(raw_cost[fit][rows, first_two[:, 1]] - raw_cost[fit][rows, first_two[:, 0]])
    design_counts = np.bincount(episode[design])
    weights = 1 / design_counts[episode[design]]
    weights = (weights / weights.sum()).astype("float32")
    dp, dc = p[design], raw_cost[design]
    g = x[design].T @ (weights[:, None] * dp * (dc - (dp*dc).sum(axis=1, keepdims=True)))
    scores = {"surprise": -np.log(p.clip(min=1e-20)), "influence": x @ g}
    scales = {}
    for name, s in scores.items():
        gap = abs(s[fit][rows, first_two[:, 1]] - s[fit][rows, first_two[:, 0]])
        scales[name] = float(d_gap / max(np.median(gap), 1e-12))
    best = raw_cost[np.arange(len(x)), nearest]
    # The additive floor protects the candidate set against numerical ties at zero.
    mask = raw_cost <= 1.25 * best[:, None] + 1e-7
    return scores, scales, mask


def evaluate(w, x, cost, labels, ids, episode):
    pred = (x[ids] @ w).argmax(axis=1)
    unique = np.unique(episode[ids])
    pv, ov = cost[ids, pred], cost[ids, labels[ids]]
    return {int(e): (float(pv[episode[ids] == e].mean()), float(ov[episode[ids] == e].mean()))
            for e in unique}


def bootstrap(delta, tasks):
    rng = np.random.default_rng(8043)
    draws = np.zeros(2000)
    for t in np.unique(tasks):
        d = delta[tasks == t]
        draws += rng.choice(d, size=(len(draws), len(d))).mean(axis=1)
    return (np.quantile(draws / len(np.unique(tasks)), [.025, .975])).tolist()


def run(args):
    history, future, x, ep, task, hashes = windows(args.data)
    split = split_episodes(task)
    idx = {k: np.flatnonzero(np.isin(ep, v)) for k, v in split.items()}
    fit, design, dev, test = (idx[k] for k in split)
    x = np.clip((x - x[fit].mean(axis=0)) / x[fit].std(axis=0).clip(min=1e-4), -8, 8)
    x = np.column_stack((x, np.ones(len(x), dtype=np.float32)))
    residual = prior_residuals(history, future, fit)
    km = MiniBatchKMeans(n_clusters=VOCAB, random_state=CODEBOOK_SEED, n_init=5,
                         batch_size=1024, max_iter=100)
    centers = km.fit(residual[fit]).cluster_centers_
    d = squared_cost(residual, centers)
    base = d.argmin(axis=1)
    # Cost scaling is a fixed fit-only scalar; argmin and raw reporting stay in command units.
    scale = max(float(d[fit].mean()), 1e-8)
    cost = d / scale
    steps = EPOCHS * int(np.ceil(len(fit) / BATCH))
    teacher = head(x, base, cost, fit, TEACHER_SEED, steps)
    scores, scales, mask = assignment_scores(x, d, fit, design, teacher, ep)
    rows, models, chosen, all_dev = [], {}, {}, {}
    pooled = np.r_[fit, design]
    for method in ("nearest", "surprise", "influence", "nearest_pooled", "direct_cost", "direct_cost_pooled"):
        grid = (COST_MIXES if method.startswith("direct") else
                STRENGTHS if method in scores else (0.,))
        train_ids = pooled if method.endswith("pooled") else fit
        all_dev[method] = {}
        for strength in grid:
            labels = base
            if method in scores and strength:
                penalty = strength * scales[method] * scores[method]
                labels = np.where(mask, d + penalty, np.inf).argmin(axis=1)
            mix = strength if method.startswith("direct") else 0.
            key = (digest(labels[train_ids]), digest(train_ids), mix)
            if key not in models:
                models[key] = [head(x, labels, cost, train_ids, s, steps, mix) for s in SEEDS]
            val = [evaluate(w, x, d, labels, dev, ep) for w in models[key]]
            metric = float(np.mean([v[0] for each in val for v in each.values()]))
            all_dev[method][str(strength)] = metric
            if method not in chosen or metric < chosen[method][0]:
                chosen[method] = (metric, strength, key, labels)

    # All choices are locked before scoring the blind test episodes.
    test_eps = np.array(split["test"])
    per_method = {}
    summary = {}
    for name, (dev_mse, strength, key, labels) in chosen.items():
        errors, oracle = [], []
        for seed, w in zip(SEEDS, models[key]):
            metrics = evaluate(w, x, d, labels, test, ep)
            errors.append([metrics[e][0] for e in test_eps])
            oracle.append([metrics[e][1] for e in test_eps])
            for e, (greedy, assigned) in metrics.items():
                rows.append([name, strength, seed, e, task[e], np.count_nonzero(ep[test] == e), greedy, assigned])
        a = np.asarray(errors)
        per_method[name] = a.mean(axis=0)
        summary[name] = dict(strength=strength, dev_mse=dev_mse, test_greedy_mse=float(a.mean()),
                             test_by_seed=a.mean(axis=1).tolist(), test_encoder_mse=float(np.mean(oracle)),
                             changed_fraction_fit=float(np.mean(labels[fit] != base[fit])),
                             changed_fraction_test=float(np.mean(labels[test] != base[test])))
    for name, err in per_method.items():
        delta = err - per_method["nearest"]
        summary[name].update(paired_delta_vs_nearest=float(delta.mean()), ci95=bootstrap(delta, task[test_eps]),
                             episodes_won=int(np.sum(delta < 0)),
                             per_task_greedy_mse={str(t):float(err[task[test_eps] == t].mean()) for t in np.unique(task)})
    report = dict(data=str(args.data.resolve()), raw_field_sha256=hashes, splits=split,
                  split_sha256={k:digest(np.asarray(v, dtype="<i8")) for k, v in split.items()},
                  split_window_counts={k:len(v) for k, v in idx.items()}, history=HISTORY, future=HORIZON,
                  k=VOCAB, stride=HORIZON, feature_count=x.shape[1], split_seed=SPLIT_SEED,
                  codebook_seed=CODEBOOK_SEED, teacher_seed=TEACHER_SEED, student_seeds=SEEDS,
                  batch=BATCH, updates_per_head=steps, lr=LR, cost_mix_grid=COST_MIXES, label_strength_grid=STRENGTHS,
                  score_scales=scales, candidate_rel_slack=.25, candidate_abs_slack=1e-7,
                  centers_sha256=digest(centers), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  versions=dict(python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__,
                                zarr=zarr.__version__), dev_grid=all_dev, test=summary)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.out / "test_episodes.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "strength", "seed", "episode", "task", "windows", "greedy_mse", "assigned_mse"])
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out must be a new directory")
    with threadpool_limits(limits=1):
        run(args)
