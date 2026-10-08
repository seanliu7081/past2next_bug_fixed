"""LIBERO rollouts for P2N-VLA: renderer placement, runner configs, episode summaries.

Shared by ``scripts/evaluate_p2n_vla.py`` (standalone snapshot evaluation) and
``oat/workspace/train_p2n_vla.py`` (official evaluation every ``training.rollout_every`` epochs).

EGL device indices need not equal CUDA ordinals: on the 2x4090 development host they are swapped, so a
numeric ``MUJOCO_EGL_DEVICE_ID`` equal to the CUDA index renders on the other GPU. ``resolve_renderer``
matches the policy GPU's CUDA UUID against ``oat.common.libero_egl_devices``, which runs in an isolated
process. ``scope_runner_to_renderer`` then builds the simulators through the scoped wrappers in
``oat.env_runner.p2n_new_convnext_libero10_runner``, which bind robosuite's two variables to that EGL index
only while the runner forks its simulators.
"""
from __future__ import annotations

import copy
import json
import math
import os
import pathlib
import subprocess
import sys
import threading
from typing import Callable, Dict, List, Mapping, Optional

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
# Same runner classes, constructed with the simulators bound to one EGL device (P2N_LIBERO_EGL_DEVICE_ID).
SCOPED_RUNNER_MODULE = "oat.env_runner.p2n_new_convnext_libero10_runner"
WILSON_Z = 1.959963984540054


# ------------------------------------------------------------------------------ renderer
def probe_egl_renderers() -> List[Dict]:
    """EGL devices with their CUDA identity, enumerated in an isolated process without rendering."""
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID")
    env.pop("CUDA_VISIBLE_DEVICES", None)
    result = subprocess.run([sys.executable, "-m", "oat.common.libero_egl_devices"], cwd=str(REPO_ROOT),
                            env=env, text=True, capture_output=True, check=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


def resolve_renderer(device, probe: Callable[[], List[Dict]] = probe_egl_renderers) -> Optional[Dict]:
    """The EGL device on ``device``'s physical GPU, matched by CUDA UUID; None unless rendering with EGL on CUDA."""
    import torch
    device = torch.device(device)
    if os.environ.get("MUJOCO_GL", "egl").lower().strip() != "egl" or device.type != "cuda":
        return None
    uuid = str(torch.cuda.get_device_properties(device).uuid).removeprefix("GPU-").lower()
    matches = [record for record in probe() if str(record["uuid"]).removeprefix("GPU-").lower() == uuid]
    if len(matches) != 1:
        raise RuntimeError(f"Cannot identify one EGL renderer for CUDA device {device} (uuid {uuid})")
    return dict(matches[0])


def scope_runner_to_renderer(runner_config: Mapping, renderer: Optional[Mapping]):
    """Build the simulators on ``renderer``'s EGL device.

    robosuite reads ``MUJOCO_EGL_DEVICE_ID`` for each rendering context, and at import it asserts that the
    value appears in ``CUDA_VISIBLE_DEVICES``. The scoped wrappers set both to the EGL index only while the
    runner forks its simulators. The policy's CUDA context already exists by then, so it is unaffected.
    """
    if renderer is None:
        return runner_config
    os.environ["P2N_LIBERO_EGL_DEVICE_ID"] = str(int(renderer["egl_device_id"]))
    scoped = dict(runner_config)
    scoped["_target_"] = f"{SCOPED_RUNNER_MODULE}.{str(runner_config['_target_']).rsplit('.', 1)[1]}"
    return scoped


def rollout_runner_config(template: Mapping, *, n_action_steps: int, n_obs_steps: int, output_dir,
                          n_test: Optional[int] = None) -> Dict:
    """Runner kwargs from a task config's ``env_runner`` block (protocol, n_test, seeds, ... kept)."""
    runner = copy.deepcopy(dict(template))
    runner.update({"n_action_steps": int(n_action_steps), "n_obs_steps": int(n_obs_steps),
                   "output_dir": str(output_dir),
                   "episode_records_path": str(pathlib.Path(output_dir) / "episodes.jsonl")})
    if n_test is not None:
        runner["n_test"] = int(n_test)
    # LiberoRunner asserts n_test_vis <= n_test (a shortened probe rollout must not trip it).
    runner["n_test_vis"] = min(int(runner.get("n_test_vis", 0) or 0), int(runner["n_test"]))
    return runner


def close_runner(runner, *, force: bool, timeout: float = 300.0) -> None:
    """Close a LIBERO runner without ever blocking on a dead or stuck simulator worker.

    ``AsyncVectorEnv.close()`` waits indefinitely for its workers. After a worker error it re-enters the pending
    wait, which then blocks forever or raises before terminating anything, and even ``close(terminate=True)``
    can raise first. So after a failure (``force``) the workers are killed outright. A normal close gets
    ``timeout`` seconds, and any worker still alive afterwards is killed. The environment is then marked
    closed, so ``VectorEnv.__del__`` cannot re-enter a blocking ``close()`` at garbage collection or exit.
    """
    if not force:
        closer = threading.Thread(target=runner.close, name="p2n_rollout_close", daemon=True)
        closer.start()
        closer.join(timeout)
    env = getattr(runner, "env", None)
    processes = list(getattr(env, "processes", None) or [])
    if any(_alive(process) for process in processes):
        for process in processes:
            try:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=10)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=10)
            except Exception:  # noqa: BLE001 - best-effort teardown; the original error (if any) propagates
                pass
        for pipe in list(getattr(env, "parent_pipes", None) or []):
            try:
                if pipe is not None:
                    pipe.close()
            except Exception:  # noqa: BLE001
                pass
        try:
            env.closed = True
        except Exception:  # noqa: BLE001
            pass


def _alive(process) -> bool:
    try:
        return bool(process.is_alive())
    except Exception:  # noqa: BLE001
        return False


# ------------------------------------------------------------------------------- summary
def wilson_interval(successes: int, trials: int):
    """95% Wilson interval; the same formula as ``scripts/evaluate_candidate.py``."""
    if trials == 0:
        return [None, None]
    z = WILSON_Z
    rate = successes / trials
    denominator = 1 + z * z / trials
    center = (rate + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials * trials)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def summarize_records(records: List[Mapping]) -> Dict:
    """Per-task and overall success with Wilson intervals; the same schema as ``evaluate_candidate.py``."""
    per_task = {}
    for name in sorted({r["task_name"] for r in records}):
        task_records = [r for r in records if r["task_name"] == name]
        successes = sum(int(r["success"]) for r in task_records)
        trials = len(task_records)
        per_task[name] = {
            "successes": successes, "trials": trials,
            "success_rate": successes / trials,
            "wilson_95_interval": wilson_interval(successes, trials),
            "mean_policy_steps": sum(r["policy_steps"] for r in task_records) / trials,
        }
    successes = sum(int(r["success"]) for r in records)
    trials = len(records)
    return {
        "successes": successes,
        "trials": trials,
        "success_rate": successes / trials if trials else None,
        "macro_task_success_rate": (sum(t["success_rate"] for t in per_task.values()) / len(per_task)
                                    if per_task else None),
        "wilson_95_interval": wilson_interval(successes, trials),
        "interval_note": "Episode-level Wilson interval; repeated evaluations of the same initial states "
                         "are not independent new environments.",
        "per_task": per_task,
    }


# ------------------------------------------------------------------------------- memory
class GpuMemorySampler:
    """Peak ``nvidia-smi`` memory.used (MiB) of one GPU, polled by a daemon thread.

    EGL renderer memory lives outside PyTorch's allocator, so ``torch.cuda.max_memory_reserved`` cannot
    see it. Without ``uuid`` or ``nvidia-smi`` the sampler records nothing.
    """

    def __init__(self, uuid: Optional[str], interval: float = 2.0):
        self.uuid = None if uuid is None else str(uuid).removeprefix("GPU-").lower()
        self.interval = float(interval)
        self.peak_mib: Optional[float] = None
        self.total_mib: Optional[float] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _sample(self) -> None:
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=uuid,memory.used,memory.total",
                                     "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.SubprocessError):
            return
        for line in result.stdout.splitlines():
            fields = [item.strip() for item in line.split(",")]
            if len(fields) != 3 or fields[0].removeprefix("GPU-").lower() != self.uuid:
                continue
            try:
                used, total = float(fields[1]), float(fields[2])
            except ValueError:
                return
            self.peak_mib = used if self.peak_mib is None else max(self.peak_mib, used)
            self.total_mib = total

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval)

    def __enter__(self):
        if self.uuid is not None:
            self._thread = threading.Thread(target=self._loop, name="p2n_gpu_memory", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
            self._sample()
        return False

    def report(self) -> Dict[str, Optional[float]]:
        gb = (lambda mib: None if mib is None else mib * 2 ** 20 / 1e9)
        headroom = (None if self.peak_mib is None or self.total_mib is None
                    else gb(self.total_mib - self.peak_mib))
        return {"peak_used_gb": gb(self.peak_mib), "total_gb": gb(self.total_mib), "headroom_gb": headroom}
