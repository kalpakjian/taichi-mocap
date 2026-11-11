"""Facing test: with a fitted camera, is the nose nearer than the ears?

The performer faces the camera for most of this clip, so the correct 3D capture must put the
nose in front of the ears. A mirrored capture flips this depth ordering.
"""
import os
import cv2
import numpy as np

NPZ = os.environ.get("NPZ", "mocap.npz")
z = np.load(NPZ)
world, img2d, vis = z["world"].astype(np.float64), z["image2d"].astype(np.float64), z["visibility"]
W, H = int(z["width"]), int(z["height"])
K = np.array([[1.5 * max(W, H), 0, W / 2], [0, 1.5 * max(W, H), H / 2], [0, 0, 1.0]])
NOSE, EAR_L, EAR_R = 0, 7, 8


def test(pts, label):
    near, errs = [], []
    for i in range(0, len(pts), 10):
        if min(vis[i][NOSE], vis[i][EAR_L], vis[i][EAR_R]) < 0.6:
            continue
        good = vis[i] > 0.6
        obj = pts[i][good].reshape(-1, 1, 3)
        im = np.stack([img2d[i][good, 0] * W, img2d[i][good, 1] * H], 1).reshape(-1, 1, 2)
        ok, rv, tv = cv2.solvePnP(obj, im, K, None, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            continue
        R = cv2.Rodrigues(rv)[0]
        cam = (R @ pts[i].T).T + tv.ravel()  # camera-space coordinates
        depth = lambda j: float(np.linalg.norm(cam[j]))  # noqa: E731
        near.append(depth(NOSE) < 0.5 * (depth(EAR_L) + depth(EAR_R)))
        p, _ = cv2.projectPoints(obj, rv, tv, K, None)
        errs.append(float(np.linalg.norm(p.reshape(-1, 2) - im.reshape(-1, 2), axis=1).mean()))
    frac = float(np.mean(near)) if near else float("nan")
    print(f"{label:<24} frames={len(near):3d}  nose-in-front {frac:5.2f}  "
          f"median reproj {np.median(errs):5.1f} px")
    return frac


mirror = world.copy()
mirror[..., 2] *= -1
f_bad = test(mirror, "mirrored (old bug)")
f_ok = test(world, "fixed (x, -y, -z)")
print()
print(f"facing agreement: fixed {f_ok:.2f} vs mirrored {f_bad:.2f} "
      f"(1.0 = nose always nearer, expected for the correct capture)")
