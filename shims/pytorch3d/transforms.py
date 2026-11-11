"""Rotation helpers with pytorch3d's shapes and conventions (leading dims are batched).

Signatures mirror pytorch3d.transforms so GVHMR code runs unchanged.
"""

import functools

import torch

__all__ = [
    "axis_angle_to_matrix",
    "matrix_to_axis_angle",
    "axis_angle_to_quaternion",
    "quaternion_to_axis_angle",
    "matrix_to_quaternion",
    "quaternion_to_matrix",
    "matrix_to_rotation_6d",
    "rotation_6d_to_matrix",
    "so3_exp_map",
    "so3_log_map",
    "euler_angles_to_matrix",
]


def _sqrt_positive_part(x):
    return torch.sqrt(torch.clamp(x, min=1e-24))


def axis_angle_to_matrix(aa):
    """(..., 3) axis-angle -> (..., 3, 3) rotation. Rodrigues, zero vector -> identity."""
    theta = torch.linalg.norm(aa, dim=-1, keepdim=True)
    axis = aa / torch.clamp(theta, min=1e-12)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = torch.zeros_like(x)
    k = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], dim=-1).reshape(*aa.shape[:-1], 3, 3)
    sin, cos = torch.sin(theta)[..., None], torch.cos(theta)[..., None]
    eye = torch.eye(3, device=aa.device, dtype=aa.dtype).expand_as(k)
    R = eye + sin * k + (1 - cos) * (k @ k)
    return torch.where((theta[..., None] < 1e-10).expand_as(R), eye, R)


def quaternion_to_matrix(q):
    """(..., 4) quaternion in (w, x, y, z) order -> (..., 3, 3) rotation."""
    q = q / torch.clamp(torch.linalg.norm(q, dim=-1, keepdim=True), min=1e-12)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    rows = torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    )
    return rows.reshape(*q.shape[:-1], 3, 3)


def matrix_to_quaternion(R):
    """(..., 3, 3) rotation -> (..., 4) quaternion, canonicalised to w >= 0."""
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    trace = m00 + m11 + m22

    s0 = _sqrt_positive_part(trace + 1.0) * 2  # 4w
    s1 = _sqrt_positive_part(1.0 + m00 - m11 - m22) * 2  # 4x
    s2 = _sqrt_positive_part(1.0 - m00 + m11 - m22) * 2  # 4y
    s3 = _sqrt_positive_part(1.0 - m00 - m11 + m22) * 2  # 4z

    q0 = torch.stack([0.25 * s0, (m21 - m12) / s0, (m02 - m20) / s0, (m10 - m01) / s0], -1)
    q1 = torch.stack([(m21 - m12) / s1, 0.25 * s1, (m01 + m10) / s1, (m02 + m20) / s1], -1)
    q2 = torch.stack([(m02 - m20) / s2, (m01 + m10) / s2, 0.25 * s2, (m12 + m21) / s2], -1)
    q3 = torch.stack([(m10 - m01) / s3, (m02 + m20) / s3, (m12 + m21) / s3, 0.25 * s3], -1)

    use0 = trace > 0
    use1 = (~use0) & (m00 >= m11) & (m00 >= m22)
    use2 = (~use0) & (~use1) & (m11 >= m22)
    q = torch.where(use0[..., None], q0,
                    torch.where(use1[..., None], q1,
                                torch.where(use2[..., None], q2, q3)))
    q = q / torch.clamp(torch.linalg.norm(q, dim=-1, keepdim=True), min=1e-12)
    return torch.where(q[..., :1] < 0, -q, q)


def axis_angle_to_quaternion(aa):
    return matrix_to_quaternion(axis_angle_to_matrix(aa))


def quaternion_to_axis_angle(q):
    return matrix_to_axis_angle(quaternion_to_matrix(q))


def matrix_to_axis_angle(R):
    """(..., 3, 3) rotation -> (..., 3) axis-angle with angle in [0, pi]."""
    q = matrix_to_quaternion(R)
    w, v = q[..., 0], q[..., 1:]
    sin_half = torch.linalg.norm(v, dim=-1, keepdim=True)
    angle = 2 * torch.atan2(sin_half, w[..., None])
    axis = v / torch.clamp(sin_half, min=1e-12)
    aa = axis * angle
    # angle ~ 0: pick a deterministic axis (x) so downstream code gets a finite number
    fallback = torch.zeros_like(aa)
    fallback[..., 0] = angle[..., 0]
    return torch.where(sin_half < 1e-10, fallback, aa)


def matrix_to_rotation_6d(R):
    """(..., 3, 3) -> (..., 6), dropping the last row (pytorch3d also uses first two rows)."""
    return R[..., :2, :].reshape(*R.shape[:-2], 6).clone()


def rotation_6d_to_matrix(d6):
    """(..., 6) -> (..., 3, 3) via Gram-Schmidt; rows of R, matching pytorch3d."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / torch.clamp(torch.linalg.norm(a1, dim=-1, keepdim=True), min=1e-12)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = b2 / torch.clamp(torch.linalg.norm(b2, dim=-1, keepdim=True), min=1e-12)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)


def so3_exp_map(log_rot):
    return axis_angle_to_matrix(log_rot)


def so3_log_map(R):
    return matrix_to_axis_angle(R)


def euler_angles_to_matrix(euler_angles, convention="XYZ"):
    """Compose per-axis rotations in the given order: R = R_c0 @ R_c1 @ R_c2.

    GVHMR only calls this with a single non-zero angle (random y-rotation augmentation),
    so the composition order cannot bite; flag it if you start using multi-axis angles.
    """
    axes = {"X": 0, "Y": 1, "Z": 2}
    mats = []
    for name, angle in zip(convention, torch.unbind(euler_angles, dim=-1)):
        onehot = torch.zeros_like(euler_angles)
        onehot[..., axes[name.upper()]] = angle
        mats.append(axis_angle_to_matrix(onehot))
    return functools.reduce(torch.matmul, mats)
