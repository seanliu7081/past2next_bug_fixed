#!/usr/bin/env python3
"""Convert one DexMimicGen generated HDF5 into this repo's zarr layout (DexMimicGen port, P1).

Output ``data/<key>`` arrays (one row per control step) plus ``meta/episode_ends``:
  * ``<camera>_image``                uint8 (H, W, 3), exactly as stored (orientation checked by gate V3)
  * ``<arm>_eef_pos``                 float32 (3)
  * ``<arm>_eef_rot6d``               float32 (6), first two ROWS of R from ``<arm>_eef_quat_site`` (xyzw)
  * ``<arm>_eef_quat_site``           float32 (4), raw, kept for checks
  * ``<arm>_gripper_qpos``            float32 (hand joints)
  * ``*_joint_pos``                   float32
  * ``action``                        float32, the flat stored ``actions`` (native action space)
  * ``task_uid``                      int64 (1), constant per task
  * ``sim_state``                     float64, the stored MuJoCo ``states`` (for reset_to-based checks)
  * ``subtask__<name>``               int8, DexMimicGen subtask termination signals (diagnostics)
Root attrs record env_args (with the task's env name), the source sha256, the demo order and the
conventions. Each episode's ``model_file`` XML goes to ``<out>.model_files.jsonl.gz``.

The zarr is written to ``<out>.tmp`` and moved into place only after the parity check (gate V2:
random (episode, t) samples equal the HDF5 after casting) passes.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import h5py
import numpy as np
import zarr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from oat.common.replay_buffer import ReplayBuffer  # noqa: E402
from oat.env.dexmg.factory import ARM_PREFIXES, get_task  # noqa: E402
from oat.env.dexmg.geometry import quat_xyzw_to_rot6d_rows  # noqa: E402

CONVERTER_VERSION = 1
DROP_ENV_KWARGS = ("env_lang",)  # generation metadata that robosuite environments reject


def demo_names(data):
    return sorted((name for name in data.keys() if name.startswith("demo_")),
                  key=lambda name: int(name.split("_")[1]))


def file_sha256(path, block=1 << 24):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(block), b""):
            digest.update(chunk)
    return digest.hexdigest()


def episode_arrays(demo, task, state_keys, image_keys):
    obs = demo["obs"]
    episode = {key: obs[key][:] for key in image_keys}
    for key in state_keys:
        episode[key] = obs[key][:].astype(np.float32)
    for prefix in ARM_PREFIXES[task["embodiment"]]:
        episode[f"{prefix}_eef_rot6d"] = quat_xyzw_to_rot6d_rows(
            obs[f"{prefix}_eef_quat_site"][:]).astype(np.float32)
    actions = demo["actions"][:]
    episode["action"] = actions.astype(np.float32)
    episode["task_uid"] = np.full((len(actions), 1), task["uid"], dtype=np.int64)
    episode["sim_state"] = demo["states"][:].astype(np.float64)
    if "datagen_info" in demo and "subtask_term_signals" in demo["datagen_info"]:
        for name, signal in demo["datagen_info/subtask_term_signals"].items():
            episode[f"subtask__{name}"] = signal[:].astype(np.int8)
    lengths = {key: len(value) for key, value in episode.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Inconsistent step counts in {demo.name}: {lengths}")
    return episode


def resolve_keys(first_obs, task):
    image_keys = [f"{camera}_image" for camera in task["cameras"]]
    missing = [key for key in image_keys if key not in first_obs]
    if missing:
        raise KeyError(f"HDF5 lacks camera observations {missing}; has {sorted(first_obs.keys())}")
    state_keys = []
    for prefix in ARM_PREFIXES[task["embodiment"]]:
        for suffix in ("eef_pos", "eef_quat_site", "gripper_qpos"):
            key = f"{prefix}_{suffix}"
            if key not in first_obs:
                raise KeyError(f"HDF5 lacks {key}; has {sorted(first_obs.keys())}")
            state_keys.append(key)
    state_keys += sorted(key for key in first_obs.keys() if key.endswith("_joint_pos"))
    return state_keys, image_keys


def verify_parity(zarr_path, hdf5_path, names, state_keys, image_keys, task, n_samples, seed):
    """Gate V2: random (episode, t) samples equal the source after the converter's casts."""
    root = zarr.open(str(zarr_path), "r")
    ends = root["meta/episode_ends"][:]
    starts = np.concatenate(([0], ends[:-1]))
    rng = np.random.default_rng(seed)
    checked = 0
    with h5py.File(hdf5_path, "r") as source:
        lengths = np.array([source[f"data/{name}/actions"].shape[0] for name in names])
        if not np.array_equal(ends, np.cumsum(lengths)):
            raise AssertionError("meta/episode_ends != cumsum(demo lengths)")
        for _ in range(n_samples):
            episode = int(rng.integers(len(names)))
            t = int(rng.integers(lengths[episode]))
            row = int(starts[episode] + t)
            demo = source[f"data/{names[episode]}"]
            for key in image_keys:
                assert np.array_equal(root["data"][key][row], demo["obs"][key][t]), (key, episode, t)
            for key in state_keys:
                assert np.array_equal(root["data"][key][row], demo["obs"][key][t].astype(np.float32)), (key, episode, t)
            for prefix in ARM_PREFIXES[task["embodiment"]]:
                expected = quat_xyzw_to_rot6d_rows(demo["obs"][f"{prefix}_eef_quat_site"][t:t + 1])[0]
                assert np.allclose(root["data"][f"{prefix}_eef_rot6d"][row], expected, atol=1e-6), (prefix, episode, t)
            assert np.array_equal(root["data"]["action"][row], demo["actions"][t].astype(np.float32)), ("action", episode, t)
            assert np.array_equal(root["data"]["sim_state"][row], demo["states"][t]), ("sim_state", episode, t)
            assert int(root["data"]["task_uid"][row, 0]) == task["uid"]
            checked += 1
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, help="key of oat.env.dexmg.factory.DEXMG_TASKS")
    parser.add_argument("--hdf5", type=Path, help="defaults to data/dexmg/raw/<factory hdf5 name>")
    parser.add_argument("--out", type=Path, help="defaults to data/dexmg/<task>_N<demos>.zarr")
    parser.add_argument("--max-demos", type=int, help="convert only the first N demos (smoke tests)")
    parser.add_argument("--parity-samples", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    task = get_task(args.task)
    hdf5_path = args.hdf5 or ROOT / "data/dexmg/raw" / task["hdf5"]
    with h5py.File(hdf5_path, "r") as source:
        data = source["data"]
        names = demo_names(data)[: args.max_demos]
        env_args = json.loads(data.attrs["env_args"])
        state_keys, image_keys = resolve_keys(data[names[0]]["obs"], task)
    stored_cameras = tuple(env_args["env_kwargs"].get("camera_names", ()))
    if stored_cameras != tuple(task["cameras"]):
        raise ValueError(f"env_args cameras {stored_cameras} differ from the task table {task['cameras']}")

    out = args.out or ROOT / "data/dexmg" / f"{args.task}_N{len(names)}.zarr"
    if out.exists() and not args.overwrite:
        raise FileExistsError(f"{out} exists; pass --overwrite")
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    sidecar = out.with_name(out.name + ".model_files.jsonl.gz")
    sidecar_tmp = sidecar.with_name(sidecar.name + ".tmp")

    buffer = ReplayBuffer.create_empty_zarr(storage=zarr.DirectoryStore(str(tmp)))
    with h5py.File(hdf5_path, "r") as source, gzip.open(sidecar_tmp, "wt") as models:
        for index, name in enumerate(names):
            demo = source[f"data/{name}"]
            buffer.add_episode(episode_arrays(demo, task, state_keys, image_keys), compressors="disk")
            models.write(json.dumps({"episode": index, "demo": name,
                                     "model_file": demo.attrs["model_file"]}) + "\n")
            if (index + 1) % 100 == 0 or index + 1 == len(names):
                print(f"  {index + 1}/{len(names)} episodes, {buffer.n_steps} steps", flush=True)
    total_steps = int(buffer.n_steps)
    del buffer

    stored_env_args = json.loads(json.dumps(env_args))
    stored_env_args["source_env_name"] = env_args["env_name"]
    stored_env_args["env_name"] = task["env_name"]
    for key in DROP_ENV_KWARGS:
        stored_env_args["env_kwargs"].pop(key, None)
    root = zarr.open(str(tmp), "r+")
    root.attrs.update({
        "dexmg_task": args.task, "task_uid": task["uid"], "embodiment": task["embodiment"],
        "env_args": stored_env_args, "eval_horizon": task["horizon"],
        "source_hdf5": str(Path(hdf5_path).resolve()), "source_sha256": file_sha256(hdf5_path),
        "demo_order": names, "converter_version": CONVERTER_VERSION,
        "rot6d_layout": "rows", "quat_source": "eef_quat_site_xyzw",
        "image_convention": "as_stored_hdf5", "camera_names": list(task["cameras"]),
        "model_file_sidecar": sidecar.name,
    })
    checked = verify_parity(tmp, hdf5_path, names, state_keys, image_keys, task,
                            args.parity_samples, seed=0)

    if out.exists():
        shutil.rmtree(out)
    os.replace(tmp, out)
    os.replace(sidecar_tmp, sidecar)
    summary = {key: [list(value.shape), str(value.dtype)] for key, value in zarr.open(str(out), "r")["data"].items()}
    print(json.dumps({"out": str(out), "episodes": len(names), "steps": total_steps,
                      "parity_samples": checked, "arrays": summary}, indent=1))
    print("V2: PASS")


if __name__ == "__main__":
    main()
