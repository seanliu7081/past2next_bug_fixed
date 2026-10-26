#!/usr/bin/env python3
"""Read-only report on a DexMimicGen robomimic HDF5 (gate V1 of the DexMimicGen port).

Prints env_args, demo counts/lengths, observation shapes, per-dim action statistics
(constant / near-constant dims, +-1 saturation of OSC delta dims), checks that the flat
``actions`` equal the concatenation of ``action_dict`` in a stated layout, and reports the
gripper-mount body quaternions recorded in the demos' ``model_file`` XML.

Standalone (h5py + numpy only) so it can run isolated:  python -I scripts/inspect_dexmg_hdf5.py FILE
"""
import argparse
import json
import re
import sys

import h5py
import numpy as np

# Flat action layouts from the DexMimicGen sources (see the port plan). Each entry is
# (action_dict key, width); the concatenation must reproduce data/demo_k/actions.
LAYOUTS = {
    "panda_dex": [("right_rel_pos", 3), ("right_rel_rot_axis_angle", 3), ("right_gripper", 6),
                  ("left_rel_pos", 3), ("left_rel_rot_axis_angle", 3), ("left_gripper", 6)],
    "gr1": [("right_abs_pos", 3), ("right_abs_rot_axis_angle", 3), ("left_abs_pos", 3),
            ("left_abs_rot_axis_angle", 3), ("right_gripper", 6), ("left_gripper", 6)],
}
SATURATION_SLICES = {"panda_dex": [slice(0, 6), slice(12, 18)], "gr1": []}


def demo_keys(data):
    return sorted(data.keys(), key=lambda name: int(name.split("_")[1]))


def detect_embodiment(env_kwargs):
    robots = env_kwargs.get("robots")
    robots = robots if isinstance(robots, (list, tuple)) else [robots]
    if any("GR1" in str(robot) for robot in robots):
        return "gr1"
    if any("PandaDex" in str(robot) for robot in robots):
        return "panda_dex"
    return "unknown"


def check_layout(demo, layout):
    if "action_dict" not in demo:
        return None, "no action_dict group"
    available = set(demo["action_dict"].keys())
    missing = [key for key, _ in layout if key not in available]
    if missing:
        return None, f"action_dict lacks {missing}; available {sorted(available)}"
    parts = [np.asarray(demo["action_dict"][key][:], dtype=np.float64) for key, _ in layout]
    widths = [part.shape[1] for part in parts]
    if widths != [width for _, width in layout]:
        return None, f"widths {widths} differ from layout {[w for _, w in layout]}"
    error = float(np.abs(np.concatenate(parts, axis=1) - demo["actions"][:]).max())
    return error, sorted(available)


def mount_quats(xml):
    """Quaternions of gripper root bodies (robosuite names them gripper<k>_<arm>_<...>)."""
    found = {}
    for match in re.finditer(r"<body\b[^>]*>", xml):
        tag = match.group(0)
        name = re.search(r'name="([^"]+)"', tag)
        quat = re.search(r'quat="([^"]+)"', tag)
        if name and quat and re.match(r"gripper\d+_\w+?_(right|left)_hand$|gripper\d+_right_\w*hand$",
                                      name.group(1)):
            found.setdefault(name.group(1), quat.group(1))
    if not found:  # fall back to the first gripper body of each prefix
        for match in re.finditer(r'<body\b[^>]*name="(gripper\d+_[^"]+)"[^>]*quat="([^"]+)"', xml):
            prefix = match.group(1).split("_")[0]
            found.setdefault(prefix, f"{match.group(1)}: {match.group(2)}")
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--layout-demos", type=int, default=20, help="demos used for the action_dict check")
    parser.add_argument("--json", help="also write the report as JSON here")
    args = parser.parse_args()

    report = {"path": args.path}
    with h5py.File(args.path, "r") as root:
        data = root["data"]
        env_args = json.loads(data.attrs["env_args"])
        env_kwargs = env_args.get("env_kwargs", {})
        report["env_name"] = env_args.get("env_name")
        report["env_version"] = env_args.get("env_version")
        report["env_kwargs"] = env_kwargs
        embodiment = detect_embodiment(env_kwargs)
        report["embodiment"] = embodiment

        names = demo_keys(data)
        lengths = np.array([data[name]["actions"].shape[0] for name in names])
        report["num_demos"] = len(names)
        report["total_attr"] = int(data.attrs.get("total", -1))
        report["total_steps"] = int(lengths.sum())
        report["length_min_mean_max"] = [int(lengths.min()), float(lengths.mean()), int(lengths.max())]
        report["mask_groups"] = list(root["mask"].keys()) if "mask" in root else []

        first = data[names[0]]
        report["demo_keys"] = sorted(first.keys())
        report["obs"] = {key: [list(value.shape[1:]), str(value.dtype)] for key, value in first["obs"].items()}
        report["action_dim"] = int(first["actions"].shape[1])
        report["datagen_info"] = sorted(first["datagen_info"].keys()) if "datagen_info" in first else None
        report["action_dict_keys"] = sorted(first["action_dict"].keys()) if "action_dict" in first else None

        actions = np.concatenate([data[name]["actions"][:] for name in names]).astype(np.float64)
        span = actions.max(0) - actions.min(0)
        std = actions.std(0)
        report["action_stats"] = {
            "min": actions.min(0).round(5).tolist(), "max": actions.max(0).round(5).tolist(),
            "mean": actions.mean(0).round(5).tolist(), "std": std.round(5).tolist(),
        }
        report["constant_dims"] = np.flatnonzero(span < 1e-4).tolist()
        report["near_constant_dims"] = np.flatnonzero((span >= 1e-4) & (std < 1e-2)).tolist()
        report["saturation_fraction"] = {
            f"{s.start}:{s.stop}": float((np.abs(actions[:, s]) >= 1.0 - 1e-6).mean())
            for s in SATURATION_SLICES.get(embodiment, [])
        }

        layout = LAYOUTS.get(embodiment)
        if layout is not None:
            errors, detail = [], None
            for name in names[:: max(1, len(names) // args.layout_demos)][: args.layout_demos]:
                error, detail = check_layout(data[name], layout)
                if error is None:
                    break
                errors.append(error)
            report["layout_check"] = {"layout": layout, "max_abs_error": max(errors) if errors else None,
                                      "demos_checked": len(errors), "detail": detail}

        if "model_file" in first.attrs:
            report["mount_quats_demo0"] = mount_quats(first.attrs["model_file"])

    if args.json:
        with open(args.json, "w") as stream:
            json.dump(report, stream, indent=1)
    for key, value in report.items():
        print(f"{key}: {json.dumps(value) if not isinstance(value, str) else value}")
    ok = (report["total_attr"] in (-1, report["total_steps"]) and report.get("layout_check", {}).get(
        "max_abs_error", 0.0) is not None and report.get("layout_check", {}).get("max_abs_error", 0.0) < 1e-5)
    print("V1:", "PASS" if ok else "CHECK")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
