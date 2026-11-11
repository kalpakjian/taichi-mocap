"""Self-check for the pytorch3d shim: run with PYTHONPATH=C:\\Users\\princ\\shims."""

import numpy as np
import torch

from pytorch3d.ops.knn import knn_points
from pytorch3d.renderer import look_at_rotation
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    euler_angles_to_matrix,
    matrix_to_axis_angle,
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    quaternion_to_matrix,
    rotation_6d_to_matrix,
)

torch.manual_seed(0)
gen = torch.Generator().manual_seed(0)
aa = torch.randn(64, 3, generator=gen) * 1.5
R = axis_angle_to_matrix(aa)

# 1. rotation matrices are valid and keep their axis fixed
ortho = (R @ R.transpose(-1, -2) - torch.eye(3)).abs().max().item()
det = (torch.linalg.det(R) - 1).abs().max().item()
axis = aa / aa.norm(dim=-1, keepdim=True)
fixed = (R @ axis.unsqueeze(-1)).squeeze(-1) - axis
axis_err = fixed.norm(dim=-1).max().item()

# 2. round trips
rt_aa = (axis_angle_to_matrix(matrix_to_axis_angle(R)) - R).abs().max().item()
rt_q = (quaternion_to_matrix(matrix_to_quaternion(R)) - R).abs().max().item()
rt_6d = (rotation_6d_to_matrix(matrix_to_rotation_6d(R)) - R).abs().max().item()

# 3. single-axis euler must match axis-angle (the only way GVHMR uses it)
theta = torch.rand(8, generator=gen) * 3.0
euler_y = torch.stack([theta, torch.zeros_like(theta), torch.zeros_like(theta)], -1)  # YXZ -> Y first
aa_y = torch.stack([torch.zeros_like(theta), theta, torch.zeros_like(theta)], -1)
euler_err = (euler_angles_to_matrix(euler_y, "YXZ") - axis_angle_to_matrix(aa_y)).abs().max().item()

# 4. knn_points against a brute-force search
p1 = torch.randn(2, 100, 3, generator=gen)
p2 = torch.randn(2, 20, 3, generator=gen)
d2, idx, nn = knn_points(p1, p2, K=3, return_nn=True)
brute = (p2.unsqueeze(2) - p1.unsqueeze(1)).pow(2).sum(-1)
brute_d, _ = torch.sort(brute, dim=-1)
knn_err = (d2 - brute_d[..., :3]).abs().max().item()
nn_ok = (nn - p1.gather(1, idx.reshape(2, -1, 1).expand(-1, -1, 3)).reshape(2, 20, 3, 3)).abs().max().item()

# 5. look_at_rotation: the target must sit at depth +d on the camera z axis
cams = torch.tensor([[0.0, 0.0, 5.0]])
target = torch.zeros(1, 3)
Rl = look_at_rotation(cams, target)
depth = (Rl @ (target - cams).unsqueeze(-1)).squeeze(-1)

checks = {
    "orthonormal": ortho,
    "det=1": det,
    "axis fixed": axis_err,
    "aa roundtrip": rt_aa,
    "quat roundtrip": rt_q,
    "6d roundtrip": rt_6d,
    "euler vs aa": euler_err,
    "knn dists": knn_err,
    "knn neighbours": nn_ok,
    "look_at depth": (depth - torch.tensor([[0.0, 0.0, 5.0]])).abs().max().item(),
}
for name, val in checks.items():
    print(f"{name:<16} {val:.2e}")
assert max(checks.values()) < 1e-5, "shim check failed"

# 6. batched, higher-dimensional leading dims behave like pytorch3d
aa4 = torch.randn(2, 5, 7, 3, generator=gen)
assert axis_angle_to_matrix(aa4).shape == (2, 5, 7, 3, 3)
assert matrix_to_quaternion(axis_angle_to_matrix(aa4)).shape == (2, 5, 7, 4)
print("shapes ok")
print("ALL OK")
