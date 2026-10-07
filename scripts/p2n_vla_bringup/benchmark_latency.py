#!/usr/bin/env python
"""Batch-1 P2N-VLA inference latency, split into prefix (SigLIP + VLM), expert decode and OAT detokenize.

Example:
  CUDA_VISIBLE_DEVICES=0 /venv/oat/bin/python scripts/p2n_vla_bringup/benchmark_latency.py \
      --variant p2n_vla_state_gate --repeats 30

Uses synthetic LIBERO-shaped observations and a synthetic normalizer (latency does not
depend on the statistics). Reports p50/p95 per stage and end to end in milliseconds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DEFAULT_TOKENIZER = "/workspace/hf_upload/tokenizer_oattok_so3aug_ep4960_mse0.001.ckpt"


def _percentiles(values):
    values = sorted(values)
    pick = lambda q: values[min(len(values) - 1, int(round(q * (len(values) - 1))))]  # noqa: E731
    return {"p50": pick(0.5), "p95": pick(0.95), "mean": statistics.fmean(values)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("p2n_vla", "p2n_vla_state_gate", "pi05_ki_flow"),
                        default="p2n_vla_state_gate")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "tests"))
    from test_p2n_vla_policy import SHAPE_META, _normalizer, make_batch  # synthetic fixtures
    from oat.policy.p2n_vla import P2NVLAPolicy
    from oat.policy.p2n_vla_state_gate import P2NVLAStateGatePolicy
    from oat.policy.pi05_ki_flow import PI05KIFlowPolicy

    cls = {"p2n_vla": P2NVLAPolicy, "p2n_vla_state_gate": P2NVLAStateGatePolicy,
           "pi05_ki_flow": PI05KIFlowPolicy}[args.variant]
    gate = cls is P2NVLAStateGatePolicy
    device = torch.device(args.device)
    start = time.perf_counter()
    policy = cls(shape_meta=SHAPE_META, model_size="full", tokenizer_checkpoint=args.tokenizer)
    policy.set_normalizer(_normalizer())
    policy.to(device).eval()
    build_s = time.perf_counter() - start
    batch = make_batch(batch=args.batch, gate=gate)
    obs = {k: v.to(device) for k, v in batch["obs"].items()}
    past = batch["past_action"].to(device)
    valid = batch["past_action_valid"].to(device)
    if gate:
        obs["state_history_valid"] = obs["state_history_valid"].bool()

    def sync():
        torch.cuda.synchronize(device)

    stages = {"prefix": [], "decode": [], "detokenize": [], "end_to_end": []}
    for index in range(args.warmup + args.repeats):
        with torch.no_grad(), policy._autocast():
            sync()
            t0 = time.perf_counter()
            prefix = policy.build_prefix(obs, train_aug=False)
            prefix_out, layout, _ = policy.vlm_pass(prefix, None)
            sync()
            t1 = time.perf_counter()
            if cls is PI05KIFlowPolicy:
                from oat.model.vla.flow_head import sample_actions
                noise = torch.randn(args.batch, policy.horizon, 32, device=device)
                sampled = sample_actions(policy.joint, policy.flow_heads, policy._detached_kv(prefix_out),
                                         prefix_out.pos0, layout.nonki_valid, noise, policy.flow_num_steps)
                sync()
                t2 = t3 = time.perf_counter()
            else:
                cond = policy.build_conditions(obs, past, valid)
                log_gate, closed = policy.compute_log_gate(obs, prefix, prefix_out, layout, cond)
                tokens = policy.generate_tokens(prefix_out, layout, cond, log_gate, closed, policy.max_seq_len,
                                                0.0, None)
                sync()
                t2 = time.perf_counter()
                policy._detokenize(tokens)
                sync()
                t3 = time.perf_counter()
        if index >= args.warmup:
            stages["prefix"].append((t1 - t0) * 1e3)
            stages["decode"].append((t2 - t1) * 1e3)
            stages["detokenize"].append((t3 - t2) * 1e3)
            stages["end_to_end"].append((t3 - t0) * 1e3)
    report = {"variant": args.variant, "batch": args.batch, "device": torch.cuda.get_device_name(device),
              "build_seconds": build_s, "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
              "milliseconds": {stage: _percentiles(values) for stage, values in stages.items()},
              "budget_ms_30hz_8_steps": 8 / 30 * 1e3}
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
