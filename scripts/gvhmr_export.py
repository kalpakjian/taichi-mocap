"""Export GVHMR SMPL-X results to a 22-joint BVH (+npz) for Blender.

Rotations come straight from the SMPL-X parameters (axis-angle per joint), so the BVH keeps
twist and reproduces the pose exactly - unlike deriving rotations from joint positions.

  python gvhmr_export.py <hmr4d_results.pt> <outdir> --tag gvhmr60 --fix-root
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mocap_math import euler_zxy  # noqa: E402

NAMES = ["Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2", "L_Ankle", "R_Ankle",
         "Spine3", "L_Foot", "R_Foot", "Neck", "L_Collar", "R_Collar", "Head",
         "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist"]
PARENT = {"Pelvis": None, "L_Hip": "Pelvis", "R_Hip": "Pelvis", "Spine1": "Pelvis",
          "L_Knee": "L_Hip", "R_Knee": "R_Hip", "Spine2": "Spine1", "L_Ankle": "L_Knee",
          "R_Ankle": "R_Knee", "Spine3": "Spine2", "L_Foot": "L_Ankle", "R_Foot": "R_Ankle",
          "Neck": "Spine3", "L_Collar": "Spine3", "R_Collar": "Spine3", "Head": "Neck",
          "L_Shoulder": "L_Collar", "R_Shoulder": "R_Collar", "L_Elbow": "L_Shoulder",
          "R_Elbow": "R_Shoulder", "L_Wrist": "L_Elbow", "R_Wrist": "R_Elbow"}


def write_bvh(path, rest_joints, rot_root_deg, rot_local_deg, root_pos_m, fps, to_cm=100.0):
    """rest_joints: {name: (3,)} metres; rot_*_deg: per-frame Euler triples."""
    rest = {n: rest_joints[j] for j, n in enumerate(NAMES)}
    lift = -min(rest["L_Foot"][1], rest["R_Foot"][1])  # put the feet on the floor
    offsets = {n: (rest[n] - rest[PARENT[n]]) * to_cm for n in NAMES if PARENT[n]}
    children = {n: [c for c in NAMES if PARENT[c] == n] for n in NAMES}
    order = ["Pelvis"]
    lines = ["HIERARCHY", "ROOT Pelvis", "{", "\tOFFSET 0.00 0.00 0.00",
             '\tCHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation']

    def emit(name, depth):
        pad = "\t" * depth
        for child in children[name]:
            order.append(child)
            x, y, z = offsets[child]
            lines.append(f"{pad}JOINT {child}")
            lines.append(f"{pad}{{")
            lines.append(f"{pad}\tOFFSET {x:.4f} {y:.4f} {z:.4f}")
            lines.append(f"{pad}\tCHANNELS 3 Zrotation Xrotation Yrotation")
            if children[child]:
                emit(child, depth + 1)
            else:
                tip = offsets[child] / max(1e-9, np.linalg.norm(offsets[child])) * 2.0
                lines.extend([f"{pad}\tEnd Site", f"{pad}\t{{",
                              f"{pad}\t\tOFFSET {tip[0]:.4f} {tip[1]:.4f} {tip[2]:.4f}", f"{pad}\t}}"])
            lines.append(f"{pad}}}")

    emit("Pelvis", 1)
    lines.extend(["}", "MOTION", f"Frames: {len(root_pos_m)}", f"Frame Time: {1.0 / fps:.8f}"])

    for f, pos in enumerate(root_pos_m):
        row = [pos[0] * to_cm, (pos[1] + lift) * to_cm, pos[2] * to_cm, *rot_root_deg[f]]
        for name in order[1:]:
            row.extend(rot_local_deg[f][name])
        lines.append(" ".join(f"{v:.4f}" for v in row))

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return len(NAMES) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("outdir")
    ap.add_argument("--tag", default="gvhmr60")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--fix-root", action="store_true",
                    help="remove sideways drift: static-camera in-place footage makes GVHMR wander")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    pred = torch.load(a.results, map_location="cpu", weights_only=False)
    par = {k: v.float() for k, v in pred["smpl_params_global"].items()}

    from hmr4d.utils.smplx_utils import make_smplx
    from pytorch3d.transforms import axis_angle_to_matrix  # shim: pure torch

    model = make_smplx("neu_fullpose").eval()
    n = par["betas"].shape[0]
    zeros = lambda d: torch.zeros(n, d)  # noqa: E731
    # smplx defaults these to a single frame, which breaks the batch concat
    extra = {"jaw_pose": zeros(3), "leye_pose": zeros(3), "reye_pose": zeros(3),
             "left_hand_pose": zeros(45), "right_hand_pose": zeros(45),
             "expression": zeros(model.num_expression_coeffs)}
    with torch.no_grad():
        out = model(betas=par["betas"], body_pose=par["body_pose"],
                    global_orient=par["global_orient"], transl=par["transl"], **extra)
        rest_out = model(betas=par["betas"][:1], body_pose=torch.zeros(1, 63),
                         global_orient=torch.zeros(1, 3), transl=torch.zeros(1, 3),
                         **{k: v[:1] for k, v in extra.items()})
    joints = out.joints[:, :22].cpu().numpy()
    rest = rest_out.joints[0, :22].cpu().numpy()

    root_pos = par["transl"].cpu().numpy().copy()  # metres, gravity-aligned (Y up)
    if a.fix_root:
        # taichi happens in place, so horizontal travel is depth-estimation drift
        drift = root_pos[:, [0, 2]] - np.median(root_pos[:, [0, 2]], axis=0)
        root_pos[:, 0] -= drift[:, 0]
        root_pos[:, 2] -= drift[:, 1]
        joints[..., 0] -= drift[:, 0:1]
        joints[..., 2] -= drift[:, 1:2]
        print(f"root drift removed: x {drift[:, 0].ptp():.2f} m, z {drift[:, 1].ptp():.2f} m")

    R_root = axis_angle_to_matrix(par["global_orient"]).cpu().numpy()
    rot_root = [euler_zxy(R_root[i]) for i in range(n)]
    R_body = axis_angle_to_matrix(par["body_pose"].reshape(n, 21, 3)).cpu().numpy()
    rot_local = [{name: euler_zxy(R_body[f, j]) for j, name in enumerate(NAMES[1:])} for f in range(n)]

    npz = os.path.join(a.outdir, f"{a.tag}.npz")
    np.savez_compressed(npz, joints3d=joints, fps=a.fps, joint_names=np.array(NAMES))
    bvh = os.path.join(a.outdir, f"{a.tag}.bvh")
    n_bones = write_bvh(bvh, rest.astype(np.float64), rot_root, rot_local, root_pos, a.fps)

    with open(bvh, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    m = lines.index("MOTION")
    rows = lines[m + 3:]
    assert int(lines[m + 1].split(":")[1]) == len(rows) == n
    assert all(len(r.split()) == 6 + 3 * n_bones for r in rows), "channel count mismatch"
    print(f"ok {n} frames, {n_bones} bones -> {bvh}")


if __name__ == "__main__":
    main()
