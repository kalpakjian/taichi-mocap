"""Project the captured 3D skeleton back onto the source video (reprojection QA).

If the 3D capture is correct, the projected joints sit on the performer. Solves one global
focal length, then a camera per frame, and writes an overlay video with the source audio.

  python reproject_check.py in.mp4 mocap/taichi60.npz out.mp4
"""

import subprocess
import sys

import cv2
import numpy as np

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]

SRC, NPZ, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
FPS = 15.0

z = np.load(NPZ)
world, vis = z["world"].astype(np.float64), z["visibility"]
W, H = int(z["width"]), int(z["height"])
img2d = z["image2d"].astype(np.float64)


def solve(idx, K, init=None):
    good = vis[idx] > 0.6
    if good.sum() < 8:
        return None
    obj = world[idx][good].reshape(-1, 1, 3)
    im = np.stack([img2d[idx][good, 0] * W, img2d[idx][good, 1] * H], 1).reshape(-1, 1, 2)
    if init is None:
        ok, rvec, tvec = cv2.solvePnP(obj, im, K, None, flags=cv2.SOLVEPNP_SQPNP)
    else:
        ok, rvec, tvec = cv2.solvePnP(obj, im, K, None, rvec=init[0], tvec=init[1], useExtrinsicGuess=True)
    if not ok:
        return None
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
    err = float(np.linalg.norm(proj.reshape(-1, 2) - im.reshape(-1, 2), axis=1).mean())
    return rvec, tvec, err


# one global focal length for the whole clip
best = (1e9, None)
for k in (0.8, 1.0, 1.2, 1.5, 1.8, 2.2):
    f = k * max(W, H)
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    errs = [r[2] for i in range(0, len(world), max(1, len(world) // 20))
            if (r := solve(i, K)) is not None]
    if errs and np.median(errs) < best[0]:
        best = (float(np.median(errs)), K, k)
med, K, k = best
print(f"global focal: {k}xmax(W,H) = f{K[0,0]:.0f}px, median reprojection error {med:.1f} px")

decode = ["ffmpeg", "-v", "error", "-ss", str(z["start"]), "-i", SRC, "-map", "0:v:0",
          "-frames:v", str(len(world)), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
encode = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
          "-r", str(FPS), "-i", "pipe:0",
          "-ss", str(z["start"]), "-t", f"{len(world) / FPS:.3f}", "-i", SRC,
          "-map", "0:v:0", "-map", "1:a:0?",
          "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
          "-c:a", "aac", "-b:a", "128k", OUT]
dec = subprocess.Popen(decode, stdout=subprocess.PIPE)
enc = subprocess.Popen(encode, stdin=subprocess.PIPE)

errs = []
prev = None
i = 0
try:
    while True:
        raw = dec.stdout.read(W * H * 3)
        if not raw or len(raw) < W * H * 3:
            break
        bgr = np.frombuffer(raw, np.uint8).reshape(H, W, 3).copy()
        if i < len(world):
            r = solve(i, K, prev)
            if r is not None:
                prev = (r[0], r[1])
                errs.append(r[2])
                proj, _ = cv2.projectPoints(world[i], r[0], r[1], K, None)
                pts = proj.reshape(-1, 2)
                for a, b in CONNECTIONS:
                    if min(vis[i][a], vis[i][b]) < 0.3:
                        continue
                    cv2.line(bgr, tuple(int(v) for v in pts[a]), tuple(int(v) for v in pts[b]),
                             (255, 200, 0), 2, cv2.LINE_AA)
                for j in range(33):
                    if vis[i][j] > 0.3:
                        cv2.circle(bgr, tuple(int(v) for v in pts[j]), 3, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.putText(bgr, f"reproj err {r[2]:.1f}px", (12, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        enc.stdin.write(bgr.tobytes())
        i += 1
finally:
    dec.kill()
    enc.stdin.close()
    enc.wait()
    cv2.destroyAllWindows()

print(f"frames={i}  median reprojection {np.median(errs):.1f} px  p90 {np.percentile(errs, 90):.1f} px")
print("wrote", OUT)
