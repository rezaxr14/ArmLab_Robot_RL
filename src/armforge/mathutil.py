"""Small, dependency-free (numpy only) rotation helpers. Quaternions are MuJoCo ``(w, x, y, z)``."""

from __future__ import annotations

import numpy as np


def quat_normalize(q) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("zero-length quaternion")
    return q / n


def quat_to_mat(q) -> np.ndarray:
    w, x, y, z = quat_normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def mat_to_quat(R) -> np.ndarray:
    R = np.asarray(R, dtype=float)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.asarray(q)
    return q / np.linalg.norm(q)


def axis_angle_to_mat(axis, angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def rot_z(angle: float) -> np.ndarray:
    return axis_angle_to_mat([0, 0, 1], angle)


def yaw_to_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def rotation_error(R_current, R_target) -> np.ndarray:
    """Small-angle world-frame rotation vector that rotates ``R_current`` towards ``R_target``."""
    Rc = np.asarray(R_current).reshape(3, 3)
    Rt = np.asarray(R_target).reshape(3, 3)
    return 0.5 * sum(np.cross(Rc[:, i], Rt[:, i]) for i in range(3))


def look_at_xyaxes(position, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """MuJoCo ``xyaxes`` (6 numbers) for a camera at ``position`` looking at ``target``.

    A MuJoCo camera looks along its local -z axis, with +x to the right and +y up.
    """
    pos = np.asarray(position, float)
    fwd = np.asarray(target, float) - pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.asarray(up, float))
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, fwd)
    return np.concatenate([right, cam_up])


def fmt(values) -> str:
    """Format numbers for an MJCF attribute with enough precision and no scientific noise."""
    return " ".join(f"{float(v):.6g}" for v in np.atleast_1d(values))
