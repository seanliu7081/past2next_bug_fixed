"""Keep simulator EGL selection separate from the training process CUDA mask.

The installed robosuite renderer reads EGL indices from CUDA_VISIBLE_DEVICES.
Only simulator construction and its child processes receive this temporary
mask. The parent policy keeps the launcher's UUID mask for all training and
inference. Existing executed-action and measured-state runners are reused.
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib
import os


@contextmanager
def renderer_env():
    """Scope the selected physical EGL device to simulator import/construction."""
    backend = os.environ.get("MUJOCO_GL", "egl").lower().strip()
    if backend != "egl":
        yield
        return

    device = os.environ.get("P2N_LIBERO_EGL_DEVICE_ID")
    if device is None or not device.isdigit():
        raise RuntimeError(
            "LIBERO EGL evaluation requires P2N_LIBERO_EGL_DEVICE_ID from the "
            "LIBERO-10 launcher GPU-to-EGL identity probe"
        )
    keys = ("CUDA_VISIBLE_DEVICES", "MUJOCO_EGL_DEVICE_ID")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        # robosuite both asserts these identifiers agree and parses them as EGL
        # indices. Simulators fork here; their numeric rendering mask persists.
        for key in keys:
            os.environ[key] = device
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class _ScopedLiberoRunner:
    _legacy_name = None

    def __init__(self, *args, **kwargs):
        with renderer_env():
            module = importlib.import_module("oat.env_runner.p2n_new_runner")
            self._runner = getattr(module, self._legacy_name)(*args, **kwargs)

    def __getattr__(self, name):
        runner = self.__dict__.get("_runner")
        if runner is None:
            raise AttributeError(name)
        return getattr(runner, name)

    def run(self, policy, **kwargs):
        return self._runner.run(policy, **kwargs)

    def close(self):
        return self._runner.close()


class P2NNewLiberoRunner(_ScopedLiberoRunner):
    _legacy_name = "P2NNewLiberoRunner"


class P2NStateGateNewLiberoRunner(_ScopedLiberoRunner):
    _legacy_name = "P2NStateGateNewLiberoRunner"
