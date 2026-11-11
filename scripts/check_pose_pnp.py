"""Independent check: does the captured 3D skeleton explain the 2D landmarks?

Runs solvePnP with a focal-length search on two variants - the mirrored one (y flipped only,
the old bug) and the handedness-preserving one - and reports reprojection error in pixels.
"""

import os
import sys

import cv2
import numpy as np

NPZ = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("NPZ", "mocap.npz")
z = np.load(NPZ)
world, img, vis = z["world"].astype(np.float64), z["image2d"].astype(np.float64), z["visibility"]
W, H = int(z["width"]), int(z["height"])
print("npz keys:", sorted(z.keys()), "frames:", len(world), "image:", W, "x", H)


def evaluate(pts3d, label):
    errs, cams = [], []
    for f in range(0, len(world), max(1, len(world) // 12)):
        good = vis[f] > 0.6
        if good.sum() < 8:
            continue
        obj = pts3d[f][good].reshape(-1, 1, 3)
        imgp = np.stack([img[f][good, 0] * W, img[f][good, 1] * H], 1).reshape(-1, 1, 2)
        best = (1e9, None)
        for k in (0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.2):
            f_len = k * max(W, H)
            K = np.array([[f_len, 0, W / 2], [0, f_len, H / 2], [0, 0, 1.0]])
            ok, rvec, tvec = cv2.solvePnP(obj, imgp, K, None, flags=cv2.SOLVEPNP_SQPNP)
            if not ok:
                continue
            proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
            err = float(np.linalg.norm(proj.reshape(-1, 2) - imgp.reshape(-1, 2), axis=1).mean())
            if err < best[0]:
                best = (err, (rvec, tvec, K))
        if best[1] is not None:
            errs.append(best[0])
            cams.append(best[1])
    errs = np.array(errs)
    print(f"{label:<28} frames={len(errs):3d}  median err {np.median(errs):6.1f} px  "
          f"min {errs.min():5.1f}  max {errs.max():6.1f}")
    if cams:
        rvec, tvec, K = sorted(cams, key=lambda c: 0)[len(cams) // 2]
        cam_pos = (-cv2.Rodrigues(rvec)[0].T @ tvec).ravel()
        print(f"{'':28} camera at x={cam_pos[0]:6.2f} y={cam_pos[1]:6.2f} z={cam_pos[2]:6.2f} m"
              f"  f={K[0, 0] / max(W, H):.2f}xmax(W,H)")
    return float(np.median(errs)) if len(errs) else 1e9


mirrored = world.copy()
mirrored[..., 2] *= -1  # what the old code produced: det = -1, i.e. a mirror image
e_bad = evaluate(mirrored, "mirrored (old, y only)")
e_ok = evaluate(world, "fixed (y and z flipped)")
print()
print("VERDICT:", "fixed geometry fits the image" if e_ok < e_bad * 0.5 else
      "inconclusive - check focal search / visibility")
