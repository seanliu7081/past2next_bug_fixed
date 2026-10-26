"""Orientation features shared by the DexMimicGen converter and environment."""
import numpy as np


def quat_xyzw_to_rot6d_rows(quat):
    """(..., 4) xyzw quaternions -> (..., 6) = [R[0, :], R[1, :]] (rows layout).

    Matches robosuite's ``transform_utils.quat2mat`` and the repo's Rotation6D decoder
    with ``rotation_6d_layout: rows``.
    """
    quat = np.asarray(quat, dtype=np.float64)
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    x, y, z, w = np.moveaxis(quat, -1, 0)
    row0 = np.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), axis=-1)
    row1 = np.stack((2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)), axis=-1)
    return np.concatenate((row0, row1), axis=-1)
