#!/usr/bin/env python
"""M1 exit check: the repo's PI0.5 port (``oat.model.vla``) against the LeRobot fp32 reference.

Runs under /venv/oat, from the repository root:

    CUDA_VISIBLE_DEVICES=1 /venv/oat/bin/python scripts/p2n_vla_parity/check_m1_parity.py --precision fp32
    CUDA_VISIBLE_DEVICES=1 /venv/oat/bin/python scripts/p2n_vla_parity/check_m1_parity.py --precision bf16
    /venv/oat/bin/python scripts/p2n_vla_parity/check_m1_parity.py --device cpu --precision fp32

The check builds the ``flow``-mode stack, exactly as the ``pi05_ki_flow`` baseline does:
- ``SiglipEncoder``, ``GemmaJoint(norm_mode='adarms')`` and ``FlowHeads``;
- ``load_pi05(mode='flow')`` from the local pi05_base snapshot.

It then feeds it the reference inputs:
- the three camera slots, with the third one masked;
- prompt ids padded to 64;
- noise, x_t and t.

And it compares the outputs with ``output/parity/pi05_reference_fp32.pt``:
- image tokens, prefix embeddings and positions;
- final prefix hidden states and per-layer K/V, on valid rows and keys;
- tied-head logits;
- sincos and adaRMS conditioning;
- the flow velocity and the 10-step ``sample_actions``.

Precision modes:
- ``fp32``: frozen weights fp32, TF32 off, no autocast. The metric is max |diff| / max |ref|. Thresholds
  sit at about 4-5x the measured CPU-vs-CUDA spread of the LeRobot reference itself.
- ``bf16``: the policy's precision. Frozen backbone weights are bf16 and the stack runs under
  ``torch.autocast(bf16, cache_enabled=False)``. The metric is the relative Frobenius error. Thresholds
  sit at about 2x the values measured on an RTX 4090.

The exit code is 0 when every check passes and 1 otherwise.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pi05_reference_spec as spec  # noqa: E402

if str(spec.REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(spec.REPO_ROOT))

GROUPS = ("image_embeddings", "prefix_embeddings", "prefix_hidden", "prefix_keys", "prefix_values", "vlm_logits",
          "time_embedding", "adarms_cond", "v_t", "sample_actions")
# metric, {group: threshold}. Measured values (RTX 4090, repo vs reference) are in the comments.
THRESHOLDS: Dict[str, Tuple[str, Dict[str, float]]] = {
    "fp32": ("max_rel_to_absmax", {
        "image_embeddings": 1e-5,     # 1.6e-6
        "prefix_embeddings": 1e-5,    # 3.9e-7 (text part bitwise)
        "prefix_hidden": 1e-4,        # 1.3e-5 (LeRobot CPU vs CUDA: 1.4e-5)
        "prefix_keys": 1e-4,          # 1.3e-5 (2.2e-5)
        "prefix_values": 1.5e-4,      # 2.0e-5 (2.7e-5)
        "vlm_logits": 5e-5,           # 4.0e-6
        "time_embedding": 1e-6,       # 0
        "adarms_cond": 1e-5,          # 0
        "v_t": 1e-5,                  # 5.7e-7
        "sample_actions": 5e-5,       # 5.0e-6
    }),
    "bf16": ("rel_fro", {
        "image_embeddings": 2e-2,     # 9.8e-3
        "prefix_embeddings": 2e-2,
        "prefix_hidden": 1e-1,        # 5.0e-2
        "prefix_keys": 1e-1,          # 4.6e-2 at layer 17
        "prefix_values": 1.5e-1,      # 6.9e-2 at layer 17
        "vlm_logits": 5e-2,           # 2.3e-2
        "time_embedding": 1e-6,       # 0 (computed in fp64/fp32 outside autocast)
        "adarms_cond": 1e-6,          # 0 (fp32, autocast disabled)
        "v_t": 1.5e-2,                # 6.2e-3
        "sample_actions": 5e-2,       # 2.2e-2
    }),
}


def group_metric(stats: Mapping[str, Mapping[str, float]], metric: str) -> Dict[str, float]:
    """Max of ``metric`` per group; per-layer entries such as ``prefix_keys[3]`` fold into ``prefix_keys``."""
    grouped: Dict[str, float] = {}
    for name, row in stats.items():
        group = name.split("[")[0]
        grouped[group] = max(grouped.get(group, 0.0), float(row[metric]))
    return grouped


def evaluate(stats: Mapping[str, Mapping[str, float]], precision: str,
             thresholds: Optional[Mapping[str, float]] = None) -> List[Tuple[str, float, float, bool]]:
    """Return ``(group, value, threshold, ok)`` rows. Raises if a thresholded group was not compared."""
    if precision not in THRESHOLDS:
        raise ValueError(f"precision must be one of {sorted(THRESHOLDS)}")
    metric, defaults = THRESHOLDS[precision]
    limits = dict(defaults if thresholds is None else thresholds)
    grouped = group_metric(stats, metric)
    missing = sorted(set(limits) - set(grouped))
    if missing:
        raise ValueError(f"stats are missing groups {missing}")
    return [(group, grouped[group], limits[group], bool(grouped[group] <= limits[group])) for group in limits]


def default_reference(device: str) -> Path:
    cpu_twin = spec.DEFAULT_OUTPUT.with_name("pi05_reference_fp32_cpu.pt")
    return cpu_twin if device == "cpu" and cpu_twin.is_file() else spec.DEFAULT_OUTPUT


def build_m1_stack(precision: str, weights: Optional[Path] = None):
    from oat.model.vla.flow_head import FlowHeads
    from oat.model.vla.gemma_joint import GemmaJoint
    from oat.model.vla.pi05_checkpoint import default_pi05_path, load_pi05
    from oat.model.vla.siglip import SiglipEncoder
    from oat.model.vla.specs import GEMMA_2B, GEMMA_300M, SIGLIP_SO400M

    frozen = torch.float32 if precision == "fp32" else torch.bfloat16
    siglip = SiglipEncoder(SIGLIP_SO400M, GEMMA_2B.width, frozen_dtype=frozen)
    joint = GemmaJoint(GEMMA_2B, GEMMA_300M, "adarms", frozen_dtype=frozen, activation_checkpointing=False)
    heads = FlowHeads(GEMMA_300M.width)
    report = load_pi05(weights or default_pi05_path(), siglip=siglip, joint=joint, mode="flow", flow_heads=heads)
    return siglip, joint, heads, report


@torch.no_grad()
def run_m1_stack(siglip, joint, heads, reference: Mapping, device: str, precision: str) -> Dict[str, object]:
    """Outputs of the repo stack in the reference layout (CPU fp32 tensors) plus exact-equality checks."""
    from oat.model.vla.flow_head import flow_condition, flow_velocity, posemb_sincos, sample_actions
    from oat.model.vla.layout import build_prefix_layout

    meta = reference["metadata"]
    inputs = {key: value.to(device) for key, value in reference["inputs"].items()}
    autocast = (torch.autocast(device_type=device, dtype=torch.bfloat16, cache_enabled=False)
                if precision == "bf16" else contextlib.nullcontext())
    with autocast:
        images = inputs["images"]
        image_tokens = torch.cat([siglip(images[:, slot]) for slot in range(images.shape[1])], dim=1)
        text = joint.embed_text(inputs["lang_tokens"])
        embeds = torch.cat((image_tokens.to(text.dtype), text), dim=1)
        layout = build_prefix_layout(reference["image_token_valid"].to(device), inputs["lang_masks"])
        prefix = joint.prefix_forward(embeds, layout)
        rows = torch.arange(embeds.shape[0], device=device)[:, None]
        logits = joint.vlm_logits(prefix.hidden[rows, reference["vlm_logits_prefix_index"].to(device)])
        t = inputs["t"]
        velocity = flow_velocity(joint, heads, prefix.kv, prefix.pos0, layout.nonki_valid, inputs["x_t"], t)
        actions = sample_actions(joint, heads, prefix.kv, prefix.pos0, layout.nonki_valid, inputs["noise"],
                                 num_steps=int(meta["num_inference_steps"]))
        time_embedding = posemb_sincos(t, joint.expert_spec.width)
        cond = flow_condition(heads, t)

    def cpu(tensor):
        return tensor.detach().float().cpu()

    n_img = int(meta["n_image_tokens"])
    candidate = {
        "image_embeddings": cpu(image_tokens), "prefix_embeddings": cpu(embeds), "prefix_hidden": cpu(prefix.hidden),
        "prefix_keys": [cpu(k) for k, _ in prefix.kv], "prefix_values": [cpu(v) for _, v in prefix.kv],
        "vlm_logits": cpu(logits), "time_embedding": cpu(time_embedding), "adarms_cond": cpu(cond),
        "v_t": cpu(velocity), "sample_actions": cpu(actions),
    }
    exact = {
        "positions_equal": bool(torch.equal(layout.positions.cpu(), reference["prefix_position_ids"])),
        "pos0_equal": prefix.pos0.cpu().tolist() == [int(v) for v in meta["prefix_valid_counts"]],
        "text_embeddings_bitwise": bool(torch.equal(candidate["prefix_embeddings"][:, n_img:],
                                                    reference["prefix_embeddings"][:, n_img:])),
        "top5_after_action": torch.topk(candidate["vlm_logits"][:, -1], k=5, dim=-1).indices.tolist(),
        "top5_after_action_reference": meta["top5_next_token_ids"],
    }
    return {"candidate": candidate, "exact": exact}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--precision", choices=tuple(THRESHOLDS), default="fp32")
    parser.add_argument("--reference", type=Path, default=None,
                        help="default: the CUDA reference, or its CPU twin with --device cpu when present")
    parser.add_argument("--weights", type=Path, default=None, help="pi05_base model.safetensors (default: HF cache)")
    parser.add_argument("--json", type=Path, default=None, help="write the full statistics as JSON")
    args = parser.parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available (set CUDA_VISIBLE_DEVICES=1; GPU 0 is reserved)")
    started = time.perf_counter()
    spec.configure_ieee_fp32(deterministic=True)

    reference_path = args.reference or default_reference(args.device)
    if not reference_path.is_file():
        parser.error(f"reference missing at {reference_path}; produce it with dump_pi05_reference.py")
    reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    spec.validate_reference(reference)
    siglip, joint, heads, report = build_m1_stack(args.precision, args.weights)
    for module in (siglip, joint, heads):
        module.to(args.device).eval()
    result = run_m1_stack(siglip, joint, heads, reference, args.device, args.precision)
    stats = spec.compare_references(result["candidate"], reference, valid_only=True)
    rows = evaluate(stats, args.precision)
    exact = result["exact"]
    exact_ok = exact["positions_equal"] and exact["pos0_equal"] and (
        exact["text_embeddings_bitwise"] or args.precision != "fp32")

    metric = THRESHOLDS[args.precision][0]
    print(f"reference: {reference_path} | device: {args.device} | precision: {args.precision} | metric: {metric}")
    print(f"loader: mode={report.mode} consumed={len(report.consumed)} dropped={report.dropped}")
    for key, value in exact.items():
        print(f"  {key}: {value}")
    for group, value, limit, ok in rows:
        print(f"  {group:<18} {value:10.3e}  <= {limit:8.1e}  {'PASS' if ok else 'FAIL'}")
    passed = exact_ok and all(ok for *_, ok in rows)
    elapsed = time.perf_counter() - started
    memory = torch.cuda.max_memory_allocated() / 2**30 if args.device == "cuda" else None
    print(f"{'PASSED' if passed else 'FAILED'} in {elapsed:.0f}s"
          + (f", cuda max allocated {memory:.1f} GiB" if memory is not None else ""))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "reference": str(reference_path), "device": args.device, "precision": args.precision,
            "metric": metric, "passed": passed, "exact": exact,
            "groups": {group: {"value": value, "threshold": limit, "ok": ok} for group, value, limit, ok in rows},
            "stats": stats, "seconds": elapsed, "cuda_max_memory_allocated_gib": memory,
        }, indent=1))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
