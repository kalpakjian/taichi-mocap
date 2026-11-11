"""Forward-kinematics check: BVH channels must reproduce the source joint positions.

  python verify_bvh.py motion.bvh motion.npz
"""

import sys

import numpy as np


def euler_zxy(c, a, b):
    """Rotation matrix for BVH channel order Zrotation Xrotation Yrotation."""
    cz, sz = np.cos(np.radians(c)), np.sin(np.radians(c))
    cx, sx = np.cos(np.radians(a)), np.sin(np.radians(a))
    cy, sy = np.cos(np.radians(b)), np.sin(np.radians(b))
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return rz @ rx @ ry


def parse(path):
    lines = open(path, encoding="utf-8").read().splitlines()
    i = lines.index("MOTION")
    hier = lines[1:i]

    joints, tokens = [], [t.split() for t in hier]
    parent_of = {}
    stack, brace_owner, pending = [], [], None
    for tk in tokens:
        key = tk[0]
        if key in ("ROOT", "JOINT"):
            joints.append({"name": tk[1], "offset": None, "channels": 0, "parent": stack[-1] if stack else None})
            stack.append(tk[1])
            pending = "joint"
        elif key == "End Site":
            pending = "end"
        elif key == "{":
            brace_owner.append(pending == "joint")
            pending = None
        elif key == "OFFSET" and joints and joints[-1]["offset"] is None:
            joints[-1]["offset"] = np.array([float(v) for v in tk[1:4]])
        elif key == "CHANNELS":
            joints[-1]["channels"] = int(tk[1])
        elif key == "}":
            if brace_owner.pop():
                stack.pop()

    names = [j["name"] for j in joints]
    for j in joints:
        parent_of[j["name"]] = j["parent"]
    n_frames = int(lines[i + 1].split(":")[1])
    data = np.array([[float(v) for v in ln.split()] for ln in lines[i + 3:i + 3 + n_frames]], np.float64)
    return joints, names, parent_of, data


def fk(joints, names, data):
    out = np.zeros((len(data), len(names), 3))
    first = {j["name"]: idx for idx, j in enumerate(joints)}
    idx = 0
    per_frame = []
    for f in range(len(data)):
        pos, rot = {}, {}
        off = 0
        for j in joints:
            ch = j["channels"]
            vals = data[f, off:off + ch]
            off += ch
            if j["parent"] is None:
                pos[j["name"]] = vals[:3]
                rot[j["name"]] = euler_zxy(vals[3], vals[4], vals[5])
            else:
                p = j["parent"]
                pos[j["name"]] = pos[p] + rot[p] @ j["offset"]
                rot[j["name"]] = rot[p] @ euler_zxy(vals[0], vals[1], vals[2])
        per_frame.append([pos[n] for n in names])
        idx += 1
    return np.array(per_frame)


if __name__ == "__main__":
    bvh, npz = sys.argv[1], sys.argv[2]
    joints, names, parent_of, data = parse(bvh)
    est = fk(joints, names, data)

    z = np.load(npz)
    ref_names = [str(n) for n in z["joint_names"]]
    assert set(ref_names) == set(names), (ref_names, names)
    src = z["joints3d"][:, [ref_names.index(n) for n in names]] * 100.0  # metres -> cm
    assert src.shape == est.shape, (src.shape, est.shape)

    hips = names.index("Hips") if "Hips" in names else next(
        i for i, n in enumerate(names) if parent_of[n] is None)
    est_rel, src_rel = est - est[:, [hips]], src - src[:, [hips]]  # root may be lifted to the floor
    err = np.linalg.norm(est_rel - src_rel, axis=-1)
    # BVH offsets keep the rest bone length, so positions can only match if the captured
    # bone lengths are stable; the meaningful check is bone *direction*.
    primary = [n for n in names if n in ("Spine", "Head", "L_Leg", "L_Foot", "L_Toe",
                                        "R_Leg", "R_Foot", "R_Toe",
                                        "L_ForeArm", "L_Hand", "R_ForeArm", "R_Hand")]
    worst_cos, worst_bone = 1.0, None
    for n in primary:
        i, p = names.index(n), names.index(parent_of[n])
        a, b = est[:, i] - est[:, p], src[:, i] - src[:, p]
        cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
        if cos.min() < worst_cos:
            worst_cos, worst_bone = float(cos.min()), n
    print(f"frames {len(data)}  joints {len(names)}  worst bone direction cos {worst_cos:.6f} ({worst_bone})")
    print(f"max position error {err.max():.2f} cm (bone-length drift, expected)")
    assert worst_cos > 0.9999, "BVH bone directions do not match the capture"

    lengths = [np.linalg.norm(src[:, i] - src[:, [names.index(parent_of[n])], :], axis=-1).mean()
               for i, n in enumerate(names) if parent_of[n]]
    for n, l in zip([n for n in names if parent_of[n]], lengths):
        print(f"  {n:<12} mean length {l:6.2f} cm")
    print("OK")
