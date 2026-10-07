#!/usr/bin/env python
"""Compare two PI0.5 reference payloads tensor by tensor (e.g. the CPU and CUDA dumps).

Runs under either venv (torch only):

    /venv/oat/bin/python scripts/p2n_vla_parity/compare_pi05_references.py \\
        output/parity/pi05_reference_fp32_cpu.pt output/parity/pi05_reference_fp32.pt

By default, prefix rows and K/V positions are restricted to the valid (non-pad) prefix tokens;
see ``pi05_reference_spec.compare_references``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pi05_reference_spec as spec  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("reference", type=Path, nargs="?", default=spec.DEFAULT_OUTPUT)
    parser.add_argument("--all-rows", action="store_true", help="also compare pad rows and pad K/V positions")
    parser.add_argument("--json", type=Path, default=None, help="optionally write the statistics as JSON")
    args = parser.parse_args(argv)

    candidate = torch.load(args.candidate, map_location="cpu", weights_only=True)
    reference = torch.load(args.reference, map_location="cpu", weights_only=True)
    spec.validate_reference(candidate)
    spec.validate_reference(reference)
    for key in ("lang_tokens", "lang_masks", "images", "img_masks", "noise", "x_t", "t"):
        if not torch.equal(candidate["inputs"][key], reference["inputs"][key]):
            raise SystemExit(f"inputs.{key} differ: the two payloads were not produced from the same inputs")
    stats = spec.compare_references(candidate, reference, valid_only=not args.all_rows)
    width = max(len(name) for name in stats)
    print(f"{'tensor':<{width}}  {'max_abs':>11}  {'ref_absmax':>11}  {'max_rel':>11}  {'rel_fro':>11}  {'mean_abs':>11}")
    for name, row in stats.items():
        print(f"{name:<{width}}  {row['max_abs']:11.3e}  {row['ref_abs_max']:11.3e}  "
              f"{row['max_rel_to_absmax']:11.3e}  {row['rel_fro']:11.3e}  {row['mean_abs']:11.3e}")
    if args.json is not None:
        args.json.write_text(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
