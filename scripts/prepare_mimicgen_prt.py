"""Build LIBERO-style MimicGen Zarr stores for PRT (50 train + 5 validation per task)."""
import argparse
import errno
from pathlib import Path
import sys
import tempfile

import cv2
import h5py
import numpy as np
from numcodecs import Blosc
import zarr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from oat.common.replay_buffer import ReplayBuffer


TASKS = {
    "stack_three_d1": "Stack the three blocks.",
    "hammer_cleanup_d1": "Put the hammer in the drawer and close the drawer.",
    "mug_cleanup_d1": "Put the mug in the drawer and close the drawer.",
    "coffee_d2": "Insert the coffee pod and close the coffee machine.",
    "square_d2": "Place the square nut on the peg.",
    "threading_d2": "Thread the needle through the ring.",
}
SPLITS = {
    "train": ("mimicgen6_N50.zarr", slice(0, 50)),
    "validation": ("mimicgen6_val.zarr", slice(50, 55)),
}
REPOSITORY = "amandlek/mimicgen_datasets"
REVISION = "33016f8a62c02334f929f2913af8fdd2a8a129e1"
DEFAULT_DIR = ROOT / "data/mimicgen"


def read_episode(demo, task, uid):
    actions = np.asarray(demo["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7 or not len(actions) or not np.isfinite(actions).all():
        raise ValueError(f"{task}{demo.name}: expected finite nonempty [T, 7] OSC actions")
    if not np.isin(actions[:, 6], [-1, 1]).all():
        raise ValueError(f"{task}{demo.name}: PRT requires gripper commands of -1 or +1")
    episode = {"action": actions, "task_uid": np.full((len(actions), 1), uid, dtype=np.int64),
               "prompt": np.full(len(actions), TASKS[task], dtype="<U128")}
    obs = demo["obs"]
    for key, dim in (("robot0_eef_pos", 3), ("robot0_eef_quat", 4), ("robot0_gripper_qpos", 2),
                     ("robot0_joint_pos", 7)):
        state = np.asarray(obs[key], dtype=np.float32)
        if state.shape != (len(actions), dim) or not np.isfinite(state).all():
            raise ValueError(f"{task}{demo.name}: invalid {key} shape or values")
        episode[key] = state
    for camera in ("agentview", "robot0_eye_in_hand"):
        frames = np.asarray(obs[f"{camera}_image"])
        if frames.ndim != 4 or frames.shape[0] != len(actions) or frames.shape[-1] != 3 or frames.dtype != np.uint8:
            raise ValueError(f"{task}{demo.name}: expected uint8 [T, H, W, 3] for {camera}")
        if frames.shape[1:3] != (128, 128):
            frames = np.stack([cv2.resize(frame, (128, 128), interpolation=cv2.INTER_LINEAR)
                               for frame in frames])
        episode[f"{camera}_rgb"] = frames
    return episode


def check_existing(path, split, count):
    root = zarr.open_group(str(path), mode="r")
    if (root.attrs.get("task_names") != list(TASKS) or root.attrs.get("split") != split
            or root.attrs.get("num_demos_per_task") != count
            or len(root["meta/episode_ends"]) != len(TASKS) * count):
        raise ValueError(f"Existing {path} does not match this {split} split. Choose a new --output-dir.")


def prepare_split(raw_dir, output_dir, split):
    name, indices = SPLITS[split]
    destination = output_dir / name
    count = indices.stop - indices.start
    if destination.exists():
        check_existing(destination, split, count)
        print(f"Using {destination}", flush=True)
        return
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)
    with tempfile.TemporaryDirectory(dir=output_dir, prefix=f".{name}-") as tmp:
        stage = Path(tmp) / name
        replay = ReplayBuffer.create_empty_zarr(storage=zarr.DirectoryStore(str(stage)))
        demos_by_task = {}
        for uid, task in enumerate(TASKS):
            with h5py.File(raw_dir / "core" / f"{task}.hdf5", "r") as source:
                data = source["data"]
                names = sorted((key for key in data if key.startswith("demo_")),
                               key=lambda key: int(key.removeprefix("demo_")))
                if len(names) < 55:
                    raise ValueError(f"{task}: expected at least 55 demos, found {len(names)}")
                demos_by_task[task] = names[indices]
                for demo_name in demos_by_task[task]:
                    replay.add_episode(read_episode(data[demo_name], task, uid), compressors=compressor)
            print(f"{split}: {task}, {count} demos", flush=True)
        replay.root.attrs.update({"split": split, "task_names": list(TASKS), "num_demos_per_task": count,
                                  "demo_names_by_task": demos_by_task})
        try:
            stage.rename(destination)
        except OSError as error:
            if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            check_existing(destination, split, count)
    print(f"Ready: {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="fetch the six official HDF5 files first")
    parser.add_argument("--hdf5-dir", type=Path, default=DEFAULT_DIR / "hdf5",
                        help="parent of core/<task>.hdf5")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.download:
        from huggingface_hub import hf_hub_download
        for task in TASKS:
            hf_hub_download(repo_id=REPOSITORY, repo_type="dataset", filename=f"core/{task}.hdf5",
                            revision=REVISION, local_dir=str(args.hdf5_dir))
    for split in SPLITS:
        prepare_split(args.hdf5_dir, args.output_dir, split)


if __name__ == "__main__":
    main()
