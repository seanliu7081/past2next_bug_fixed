#!/usr/bin/env python
"""Construct an explicit weight edit from manually selected activation contexts.

This is a contrast-of-means construction, not fitting or reward optimization.
Choose risk/reference chunks after inspecting a baseline, specify the desired
first-order action change, and evaluate the resulting fixed patch separately.
All range selectors use Python's half-open convention, e.g. ``0:3,7``.

Example::

    /venv/oat/bin/python scripts/build_pi05_weight_edit.py \
      --trace /path/to/baseline/head_captures --snapshot /path/to/50k.ckpt \
      --risk-chunks 10:15 --reference-chunks 0:5 --raw-action-change 0.15 \
      --output-dir /path/to/new-candidate

The action-change magnitude is in LIBERO command units, NOT metres. The estimate
holds baseline hidden features fixed; actual closed-loop and denoising effects
must be measured. The static linear edit may also affect unselected stages.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from oat.model.vla.weight_space_edit import (  # noqa: E402
    TARGET, apply_output_head_patch, checkpoint_output_head, export_edited_snapshot,
    file_sha256, tensor_sha256,
)


def select_indices(selector: str, size: int) -> list[int]:
    """Parse a bounded, nonnegative list of indices and half-open ranges."""
    if selector == "all":
        return list(range(size))
    chosen = set()
    for item in selector.split(","):
        if not item.strip():
            raise ValueError("Empty range selector")
        if ":" in item:
            pieces = item.split(":")
            if len(pieces) != 2:
                raise ValueError("Ranges must be START:STOP (half-open)")
            start, stop = (int(piece) for piece in pieces)
            if not 0 <= start < stop <= size:
                raise ValueError(f"Range {item!r} is outside [0,{size})")
            chosen.update(range(start, stop))
        else:
            value = int(item)
            if not 0 <= value < size:
                raise ValueError(f"Index {value} is outside [0,{size})")
            chosen.add(value)
    if not chosen:
        raise ValueError("Selector chose no indices")
    return sorted(chosen)


def load_capture_trace(path: Path):
    """Read evaluator chunk_NNNN.npz files (or one aggregate hidden archive)."""
    path = path.expanduser().resolve()
    files = sorted(path.glob("chunk_*.npz"), key=lambda file: int(file.stem.split("_")[-1])) \
        if path.is_dir() else [path]
    if not files:
        raise ValueError(f"No chunk captures found in {path}")
    hidden, steps, eef, ids, sources = [], [], [], [], []
    for index, file in enumerate(files):
        with np.load(file, allow_pickle=False) as data:
            feature = np.asarray(data["hidden"], dtype=np.float32)
            if not path.is_dir() and feature.ndim == 5:
                count = feature.shape[0]
                hidden.extend(feature)
                steps.extend(np.asarray(data.get("chunk_start_steps", np.arange(count))).tolist())
                eef.extend(np.full((count, 3), np.nan))
                ids.extend(range(count))
            else:
                if feature.ndim != 4:
                    raise ValueError(f"{file}: expected [steps,batch,horizon,width] hidden array")
                hidden.append(feature)
                steps.append(int(np.asarray(data.get("policy_step", index))))
                eef.append(np.asarray(data.get("eef_pos", [np.nan] * 3), dtype=np.float64))
                ids.append(int(file.stem.split("_")[-1]) if path.is_dir() else 0)
        sources.append({"path": str(file), "sha256": file_sha256(file)})
    if ids != list(range(len(ids))):
        raise ValueError("Captures must be contiguous from chunk_0000; incomplete/gapped traces are ambiguous")
    features = np.stack(hidden)
    if not np.isfinite(features).all() or features.ndim != 5:
        raise ValueError("Features must be finite [chunks,steps,batch,horizon,width]")
    return features, np.asarray(steps, dtype=np.int64), np.asarray(eef), sources


def centered_contrast(features: np.ndarray, risk: list[int], reference: list[int],
                      horizon_positions: list[int], normalized_action_change: float,
                      protect_reference_variation: bool = False):
    """Construct a rank-one velocity edit with risk/reference mean responses 1/0.

    All denoising steps contribute equally because the deployed Euler integrator
    uses uniform dt=-1/S. Only the caller-specified horizon positions determine
    the contrast. All chunks/horizon positions receive diagnostic predictions.
    Optional reference protection analytically projects the contrast into the
    orthogonal complement of reference chunk-mean variation. It protects those
    means to first order, not every individual token or denoising trajectory.
    """
    if set(risk) & set(reference):
        raise ValueError("Risk and reference selections must be disjoint")
    if not risk or not reference or not np.isfinite(normalized_action_change) or normalized_action_change == 0:
        raise ValueError("Need nonempty selections and a finite nonzero requested change")
    if features.ndim != 5 or not np.isfinite(features).all():
        raise ValueError("Expected finite features [chunks,steps,batch,horizon,width]")
    selected = features[:, :, :, horizon_positions, :]
    risk_mean = selected[risk].mean(axis=(0, 1, 2, 3), dtype=np.float64)
    reference_mean = selected[reference].mean(axis=(0, 1, 2, 3), dtype=np.float64)
    original_contrast = risk_mean - reference_mean
    contrast = original_contrast.copy()
    original_norm = float(np.linalg.norm(original_contrast))
    reference_chunk_means = selected[reference].mean(axis=(1, 2, 3), dtype=np.float64)
    variations = reference_chunk_means - reference_mean
    protection = {
        "enabled": bool(protect_reference_variation), "reference_variation_rank": 0,
        "svd_tolerance": 0.0, "singular_values": [], "original_contrast_norm": original_norm,
    }
    if protect_reference_variation:
        # One SVD determines a numerical span; no target fitting or search loop.
        _, singular, right = np.linalg.svd(variations, full_matrices=False)
        tolerance = float(max(variations.shape) * np.finfo(np.float64).eps * singular[0]) \
            if singular.size and singular[0] > 0 else 0.0
        rank = int(np.count_nonzero(singular > tolerance))
        basis = right[:rank]
        contrast -= basis.T @ (basis @ contrast)
        protection.update(reference_variation_rank=rank, svd_tolerance=tolerance,
                          singular_values=singular.tolist())
    norm = float(np.linalg.norm(contrast))
    if norm <= 1e-8 * max(1.0, float(np.linalg.norm(risk_mean)), float(np.linalg.norm(reference_mean))):
        raise ValueError("Selected contexts have no numerically resolvable contrast after reference protection"
                         if protect_reference_variation else "Selected contexts have no numerically resolvable contrast")
    protection.update(retained_contrast_norm=norm,
                      retained_contrast_fraction=norm / original_norm if original_norm else 0.0)
    direction = np.asarray(contrast / norm, dtype=np.float32)
    center = float(reference_mean @ direction.astype(np.float64))
    separation = float(original_contrast @ direction.astype(np.float64))
    if not np.isfinite(separation) or separation <= 0:
        raise ValueError("Reference-protected direction has no positive risk separation")
    # dt sums to -1. The minus sign translates desired action change to velocity.
    coefficient = float(np.float32(-normalized_action_change / separation))
    bias_delta = float(np.float32(-coefficient * center))
    projections = np.einsum("csbhd,d->csbh", features.astype(np.float64), direction.astype(np.float64))
    signal = (projections - center) / separation
    velocity_delta = coefficient * projections + bias_delta
    first_order_action_delta = -velocity_delta.mean(axis=1)
    reference_signals = signal[reference][:, :, :, horizon_positions].mean(axis=(1, 2, 3))
    reference_deltas = first_order_action_delta[reference][:, :, horizon_positions].mean(axis=(1, 2))
    protection.update(reference_chunk_mean_signal_max_abs=float(np.abs(reference_signals).max()),
                      reference_chunk_mean_normalized_action_delta_max_abs=float(np.abs(reference_deltas).max()))
    return {
        "direction": direction, "risk_mean": risk_mean, "reference_mean": reference_mean,
        "center": center, "separation": separation, "coefficient": coefficient,
        "bias_delta": bias_delta, "signal": signal,
        "first_order_normalized_action_delta": first_order_action_delta,
        "reference_protection": protection,
    }


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace", "--trace-dir", dest="trace", required=True, type=Path,
                   help="head_captures directory or aggregate NPZ")
    p.add_argument("--snapshot", required=True, type=Path)
    p.add_argument("--weights", choices=("model", "ema"), default="model")
    risk = p.add_mutually_exclusive_group(required=True)
    risk.add_argument("--risk-chunks", help="Chunk IDs; e.g. 10:15,20 (half-open ranges)")
    risk.add_argument("--risk-steps", help="Select chunks whose starting policy step matches this range")
    reference = p.add_mutually_exclusive_group(required=True)
    reference.add_argument("--reference-chunks")
    reference.add_argument("--reference-steps")
    p.add_argument("--horizon-positions", help="Positions defining each context; default deployed first 8")
    p.add_argument("--protect-reference-variation", action="store_true",
                   help="Project out reference chunk-mean variation using one SVD; preserves those means to first order")
    p.add_argument("--row", type=int, default=2, help="Action row: xyz=0,1,2; default z")
    change = p.add_mutually_exclusive_group(required=True)
    change.add_argument("--raw-action-change", type=float, help="Intended risk-mean change in raw command units")
    change.add_argument("--normalized-action-change", type=float)
    p.add_argument("--output-dir", required=True, type=Path, help="New directory; never overwritten")
    p.add_argument("--export-snapshot", type=Path, help="Optional new compatible .ckpt path; never overwritten")
    p.add_argument("--description", default="Manually selected centered risk/reference activation contrast")
    return p


def main(argv=None):
    import dill

    args = parser().parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    if args.export_snapshot is not None and args.export_snapshot.expanduser().resolve().exists():
        raise FileExistsError(args.export_snapshot)
    features, steps, eef, source_files = load_capture_trace(args.trace)
    payload = torch.load(args.snapshot, map_location="cpu", pickle_module=dill, weights_only=False, mmap=True)
    policy, key = checkpoint_output_head(payload, args.weights)
    if features.shape[-1] != policy.flow_heads.action_out_proj.in_features:
        raise ValueError("Captured feature width does not match the source checkpoint")
    configured_steps = int(payload["policy_config"].get("flow_num_steps", 10))
    if features.shape[1] != configured_steps:
        raise ValueError("Capture must include every deployed denoising step")
    if not 0 <= args.row < policy.action_dim:
        raise ValueError("Requested row is outside the real action dimensions")
    horizon = features.shape[3]
    n_exec = int(payload["policy_config"].get("n_action_steps", min(8, horizon)))
    positions = select_indices(args.horizon_positions or f"0:{n_exec}", horizon)

    def choose(chunks, policy_steps):
        if chunks is not None:
            return select_indices(chunks, features.shape[0])
        wanted = set(select_indices(policy_steps, int(steps.max()) + 1))
        result = [i for i, step in enumerate(steps) if int(step) in wanted]
        if not result:
            raise ValueError("Policy-step selector matches no captured chunk start")
        return result

    risk, reference = choose(args.risk_chunks, args.risk_steps), choose(args.reference_chunks, args.reference_steps)
    state = payload["state_dicts"][key]
    scale = float(state["action_normalizer.params_dict.action.scale"][args.row])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Expected a positive finite action normalization scale")
    normalized_change = args.normalized_action_change if args.normalized_action_change is not None \
        else args.raw_action_change * scale
    raw_change = normalized_change / scale
    construction = centered_contrast(features, risk, reference, positions, normalized_change,
                                    protect_reference_variation=args.protect_reference_variation)
    calibration = {
        "method": ("Manually selected contrast projected into reference chunk-variation nullspace by one SVD; "
                   "no fit loop, gradients, or reward search") if args.protect_reference_variation else
                  "Manually selected contrast of means; no fit loop, gradients, or reward search",
        "trace": str(args.trace.expanduser().resolve()), "trace_files": source_files,
        "source_snapshot": str(args.snapshot.expanduser().resolve()),
        "source_snapshot_sha256": file_sha256(args.snapshot), "weights": args.weights,
        "risk_chunks": risk, "reference_chunks": reference, "horizon_positions": positions,
        "flow_steps": configured_steps, "action_row": args.row, "action_normalizer_scale": scale,
        "intended_risk_mean_raw_action_change": raw_change,
        "intended_risk_mean_normalized_action_change": normalized_change,
        "reference_projection_center": construction["center"], "contrast_separation": construction["separation"],
        "reference_protection": construction["reference_protection"],
        "limitation": "First-order fixed-feature estimate; no guarantee under denoising or closed-loop feedback",
    }
    patch = {
        "schema_version": 1, "kind": "action_output_rank_one", "target": TARGET,
        "description": args.description, "rows": [args.row],
        "coefficients": [construction["coefficient"]], "bias_delta": [construction["bias_delta"]],
        "direction_file": "direction.npz", "direction_key": "direction", "normalize_direction": False,
        "expected_weight_sha256": tensor_sha256(policy.flow_heads.action_out_proj.weight),
        "expected_bias_sha256": tensor_sha256(policy.flow_heads.action_out_proj.bias),
        "calibration": calibration,
    }
    normalized_delta = construction["first_order_normalized_action_delta"]
    raw_delta = normalized_delta / scale
    rows = []
    for i in range(features.shape[0]):
        selected_delta = raw_delta[i, :, positions]
        label = "risk" if i in risk else "reference" if i in reference else "unselected"
        row = {
            "chunk": i, "policy_step": int(steps[i]), "selection": label,
            "first_order_raw_mean": float(selected_delta.mean()),
            "first_order_raw_min": float(selected_delta.min()), "first_order_raw_max": float(selected_delta.max()),
            "first_order_normalized_mean": float(normalized_delta[i, :, positions].mean()),
            "centered_signal_mean": float(construction["signal"][i, :, :, positions].mean()),
            "centered_signal_mean_by_denoising_step": construction["signal"][i][:, :, positions].mean(axis=(1, 2)).tolist(),
            "raw_action_change_contribution_by_denoising_step": (
                raw_change * construction["signal"][i][:, :, positions].mean(axis=(1, 2)) / configured_steps
            ).tolist(),
        }
        if np.isfinite(eef[i]).all():
            row["eef_pos"] = eef[i].tolist()
        rows.append(row)
    diagnostic = {"calibration": calibration, "per_chunk": rows,
                  "denoising_times": [1.0 - i / configured_steps for i in range(configured_steps)],
                  "interpretation": "Same static weights act at every stage; unselected effects are collateral estimates"}
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "direction.npz", direction=construction["direction"],
                        risk_mean=construction["risk_mean"], reference_mean=construction["reference_mean"])
    (output / "patch.json").write_text(json.dumps(patch, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(output / "responses.npz", signal=construction["signal"],
                        first_order_normalized_action_delta=normalized_delta,
                        first_order_raw_action_delta=raw_delta, chunk_start_steps=steps)
    (output / "responses.json").write_text(json.dumps(diagnostic, indent=2, allow_nan=False) + "\n")
    application_spec = dict(patch, direction_file=str(output / "direction.npz"))
    applied = apply_output_head_patch(policy, application_spec)
    (output / "patch_provenance.json").write_text(json.dumps(applied.provenance, indent=2, allow_nan=False) + "\n")
    summary = {"patch": str(output / "patch.json"), "responses": str(output / "responses.json"),
               "changed_values": applied.provenance["changed_values"],
               "risk_mean_first_order_raw_change": float(raw_delta[risk][:, :, positions].mean()),
               "reference_mean_first_order_raw_change": float(raw_delta[reference][:, :, positions].mean()),
               "reference_protection": construction["reference_protection"]}
    if args.export_snapshot is not None:
        exported = export_edited_snapshot(args.snapshot, args.export_snapshot, application_spec, args.weights)
        (output / "export.json").write_text(json.dumps(exported, indent=2, allow_nan=False) + "\n")
        summary["exported_snapshot"] = exported["destination"]
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
