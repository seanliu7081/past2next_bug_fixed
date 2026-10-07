"""M5 overfit check: score a P2N-VLA snapshot on its own TRAINING windows.

The workspace validates on the held-out complement of the training episodes, so an overfit run
(``task.policy.dataset.max_train_episodes=2``) needs this offline pass. It re-instantiates the dataset
config embedded in the snapshot and reports, over the training (or validation) windows:

- teacher-forced ``loss_ar`` / ``ar_token_acc`` (+ ``loss_ki`` / ``ki_token_acc``) with the dataset history;
- greedy generation (AR heads): generated-token accuracy vs. the OAT ids of the GT chunk;
- stateless ``predict_action`` reconstruction MSE vs. the GT 16x7 chunk (raw action space, like ``val/``);
- the OAT round-trip floor on the same windows, ``detokenize(tokenize(action))`` vs. ``action``.

The overfit passes when the reconstruction MSE approaches the round-trip floor and the token
accuracies approach 1.

    CUDA_VISIBLE_DEVICES=0 /venv/oat/bin/python scripts/p2n_vla_bringup/overfit_recon.py \
        --snapshot output/training/overfit_2ep_p2n_vla/snapshots/upd-003000_ema.ckpt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--pi05", type=Path, default=None)
    parser.add_argument("--spm", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None,
                        help="JSON report (default: <run>/overfit_recon_<snapshot>_<weights>_<split>.json)")
    return parser.parse_args(argv)


def _move(value, device):
    import torch
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    return value


def _generate_tokens(policy, obs, past, valid):
    """Greedy tokens through the same calls as ``P2NVLACommonPolicy._generate_actions``."""
    import torch
    with policy._rollout_mode(), torch.no_grad(), policy._autocast():
        prefix = policy.build_prefix(obs, train_aug=False)
        prefix_out, layout, _ = policy.vlm_pass(prefix, None)
        cond = policy.build_conditions(obs, past, valid)
        log_gate, hist_closed = policy.compute_log_gate(obs, prefix, prefix_out, layout, cond)
        return policy.generate_tokens(prefix_out, layout, cond, log_gate, hist_closed, policy.max_seq_len,
                                      0.0, policy.topk)


def main(argv=None):
    args = parse_args(argv)
    import hydra
    import torch
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    from oat.common.p2n_new_capabilities import predict_validation_action, validate_history_batch
    from scripts.evaluate_p2n_vla import decodes_tokens, policy_class, read_payload

    started = time.monotonic()
    payload = read_payload(args.snapshot)
    cls = policy_class(payload)
    policy = cls.from_checkpoint(str(args.snapshot), base_weights=None if args.pi05 is None else str(args.pi05),
                                 weights=args.weights, device=args.device,
                                 spm_path=None if args.spm is None else str(args.spm))
    policy.eval()
    device = torch.device(args.device)
    cfg = OmegaConf.create(payload["cfg"])
    dataset = hydra.utils.instantiate(cfg.task.policy.dataset)
    if args.split == "val":
        dataset = dataset.get_validation_dataset()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        drop_last=False, pin_memory=device.type == "cuda")
    ar_head = decodes_tokens(policy)
    normalizer = policy.action_normalizer["action"]
    sums, weight_total, batches = {}, 0.0, 0
    max_abs_path_diff = 0.0

    def add(name, value, weight):
        if value is None:
            return
        total, count = sums.get(name, (0.0, 0.0))
        sums[name] = (total + float(value) * weight, count + weight)

    with torch.no_grad():
        for index, batch in enumerate(loader):
            if args.max_batches is not None and index >= args.max_batches:
                break
            batch = _move(batch, device)
            validate_history_batch(policy, batch)
            weight = float(batch["action"].shape[0])
            policy(batch, history_mode="expert")
            for name, value in (policy.last_loss_components or {}).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    add(f"teacher_forced/{name}", value, weight)
            target = batch["action"].float()
            prediction = predict_validation_action(policy, batch)["action_pred"].float()
            error = (prediction - target).square()
            add("reconstruction_mse", error.mean(), weight)
            add("reconstruction_mse_executed", error[:, :policy.n_action_steps].mean(), weight)
            normalized_error = (normalizer.normalize(prediction) - normalizer.normalize(target)).square()
            add("reconstruction_mse_normalized", normalized_error.mean(), weight)
            if ar_head:
                targets = policy.encode_targets(batch["action"])
                floor = policy._detokenize(targets)
                add("oat_floor_mse", (floor - target).square().mean(), weight)
                add("oat_floor_mse_normalized",
                    (normalizer.normalize(floor) - normalizer.normalize(target)).square().mean(), weight)
                tokens = _generate_tokens(policy, batch["obs"], batch["past_action"], batch["past_action_valid"])
                matches = (tokens == targets).float()
                add("generated_token_acc", matches.mean(), weight)
                add("generated_chunk_exact", matches.all(dim=1).float().mean(), weight)
                for k in range(matches.shape[1]):
                    add(f"generated_token_acc_k{k}", matches[:, k].mean(), weight)
                # predict_action must decode exactly the greedy tokens scored above.
                max_abs_path_diff = max(max_abs_path_diff,
                                        float((policy._detokenize(tokens) - prediction).abs().max()))
            weight_total += weight
            batches += 1
    policy.reset()
    report = {name: total / count for name, (total, count) in sorted(sums.items())}
    report.update({
        "snapshot": str(args.snapshot.resolve()), "weights": args.weights, "split": args.split,
        "policy_class": f"{cls.__module__}.{cls.__qualname__}", "windows": int(weight_total), "batches": batches,
        "dataset_windows": len(dataset),
        "max_train_episodes": OmegaConf.select(cfg, "task.policy.dataset.max_train_episodes"),
        "predict_action_vs_greedy_tokens_max_abs_diff": max_abs_path_diff if ar_head else None,
        "seconds": time.monotonic() - started,
    })
    output = args.output or (args.snapshot.resolve().parent.parent
                             / f"overfit_recon_{args.snapshot.stem}_{args.weights}_{args.split}.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"wrote {output}")
    return report


if __name__ == "__main__":
    main()
