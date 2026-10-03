"""Enumerate CUDA identities for EGL devices without creating a renderer.

Run this module in an isolated process with CUDA_VISIBLE_DEVICES unset and
CUDA_DEVICE_ORDER=PCI_BUS_ID. EGL indices need not match CUDA ordinals.
"""
from __future__ import annotations

import ctypes
import json
import os


def probe_egl_devices():
    """Return CUDA UUID/ordinal and EGL index records; no contexts are rendered."""
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        raise RuntimeError("EGL identity probing requires CUDA_VISIBLE_DEVICES to be unset")
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("EGL identity probing requires CUDA_DEVICE_ORDER=PCI_BUS_ID")

    import torch

    properties = [torch.cuda.get_device_properties(index)
                  for index in range(torch.cuda.device_count())]
    if not properties:
        raise RuntimeError("No CUDA devices are available for LIBERO EGL evaluation")

    egl = ctypes.CDLL("libEGL.so.1")
    egl.eglGetProcAddress.argtypes = [ctypes.c_char_p]
    egl.eglGetProcAddress.restype = ctypes.c_void_p

    def extension(name, result_type, *argument_types):
        address = egl.eglGetProcAddress(name.encode("ascii"))
        if not address:
            raise RuntimeError(f"Required EGL extension is unavailable: {name}")
        return ctypes.CFUNCTYPE(result_type, *argument_types)(address)

    query_devices = extension(
        "eglQueryDevicesEXT", ctypes.c_uint, ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int),
    )
    query_attribute = extension(
        "eglQueryDeviceAttribEXT", ctypes.c_uint, ctypes.c_void_p,
        ctypes.c_int, ctypes.POINTER(ctypes.c_ssize_t),
    )
    query_string = extension(
        "eglQueryDeviceStringEXT", ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int,
    )
    device_count = ctypes.c_int()
    if not query_devices(0, None, ctypes.byref(device_count)) or device_count.value < 1:
        raise RuntimeError("EGL could not enumerate any devices")
    devices = (ctypes.c_void_p * device_count.value)()
    if not query_devices(len(devices), devices, ctypes.byref(device_count)):
        raise RuntimeError("EGL device enumeration failed")

    records = []
    for egl_index in range(device_count.value):
        extensions = query_string(devices[egl_index], 0x3055) or b""  # EGL_EXTENSIONS
        if b"EGL_NV_device_cuda" not in extensions.split():
            continue
        cuda_ordinal = ctypes.c_ssize_t(-1)
        if not query_attribute(devices[egl_index], 0x323A, ctypes.byref(cuda_ordinal)):
            raise RuntimeError(f"EGL device {egl_index} did not provide EGL_CUDA_DEVICE_NV")
        index = cuda_ordinal.value
        if not 0 <= index < len(properties):
            raise RuntimeError(f"EGL device {egl_index} reported unavailable CUDA ordinal {index}")
        prop = properties[index]
        records.append({
            "uuid": "GPU-" + str(prop.uuid).removeprefix("GPU-").lower(),
            "cuda_ordinal": index,
            "egl_device_id": egl_index,
            "name": prop.name,
        })
    if not records:
        raise RuntimeError("No EGL devices expose NVIDIA CUDA identity for LIBERO evaluation")
    return records


if __name__ == "__main__":
    print(json.dumps(probe_egl_devices()))
