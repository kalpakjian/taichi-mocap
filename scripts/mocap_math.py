"""Shared rotation math for the mocap scripts (no mediapipe / cv2 dependency)."""

import numpy as np


def align_rotation(a, b):
    """Minimal rotation taking unit vector a onto unit vector b."""
    v, c = np.cross(a, b), float(np.dot(a, b))
    n = np.linalg.norm(v)
    if n < 1e-9:
        if c > 0:
            return np.eye(3)
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis = axis / np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def euler_zxy(R):
    """Degrees for BVH channel order Zrotation Xrotation Yrotation (R = Rz @ Rx @ Ry)."""
    a = float(np.arcsin(np.clip(R[2, 1], -1.0, 1.0)))
    if abs(abs(R[2, 1]) - 1.0) < 1e-6:  # gimbal lock: only c +/- b is defined
        return np.degrees([0.0, a, np.arctan2(R[0, 2], R[0, 0])])
    return np.degrees([np.arctan2(-R[0, 1], R[1, 1]), a, np.arctan2(-R[2, 0], R[2, 2])])


def euler_zxy_matrix(c, a, b):
    """Inverse of euler_zxy: degrees -> rotation matrix (for FK checks)."""
    cz, sz = np.cos(np.radians(c)), np.sin(np.radians(c))
    cx, sx = np.cos(np.radians(a)), np.sin(np.radians(a))
    cy, sy = np.cos(np.radians(b)), np.sin(np.radians(b))
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return rz @ rx @ ry
