"""Extract a 3D body skeleton from video (MediaPipe Tasks pose), no SMPL license needed.

Outputs: skeleton overlay mp4, world-landmark .npz, and a swing-only .bvh for Blender.

  python pose_extract.py in.mp4 outdir --start 100 --seconds 60

shortcut: rotations carry no twist (direction-only) and per-frame jitter is smoothed;
upgrade to GVHMR/SMPL once the SMPL + SMPL-X body models are licensed and installed.
"""

import argparse
import os
import subprocess
import sys

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from mocap_math import align_rotation, euler_zxy  # shared with gvhmr_export.py

NOSE, L_SH, R_SH = 0, 11, 12
L_EL, R_EL = 13, 14
L_WR, R_WR = 15, 16
L_HIP, R_HIP = 23, 24
L_KN, R_KN = 25, 26
L_AN, R_AN = 27, 28
L_FTI, R_FTI = 31, 32

# (joint name, parent, source) - source is a landmark index or an averaged pair
SKELETON = [
    ("Hips", None, ("avg", L_HIP, R_HIP)),
    ("Spine", "Hips", ("avg", L_SH, R_SH)),
    ("Head", "Spine", ("pt", NOSE)),
    ("L_UpLeg", "Hips", ("pt", L_HIP)),
    ("L_Leg", "L_UpLeg", ("pt", L_KN)),
    ("L_Foot", "L_Leg", ("pt", L_AN)),
    ("L_Toe", "L_Foot", ("pt", L_FTI)),
    ("R_UpLeg", "Hips", ("pt", R_HIP)),
    ("R_Leg", "R_UpLeg", ("pt", R_KN)),
    ("R_Foot", "R_Leg", ("pt", R_AN)),
    ("R_Toe", "R_Foot", ("pt", R_FTI)),
    ("L_Shoulder", "Spine", ("pt", L_SH)),
    ("L_ForeArm", "L_Shoulder", ("pt", L_EL)),
    ("L_Hand", "L_ForeArm", ("pt", L_WR)),
    ("R_Shoulder", "Spine", ("pt", R_SH)),
    ("R_ForeArm", "R_Shoulder", ("pt", R_EL)),
    ("R_Hand", "R_ForeArm", ("pt", R_WR)),
]
NAMES = [s[0] for s in SKELETON]
PARENT = {s[0]: s[1] for s in SKELETON}
# Each joint's own bone direction is defined by its primary child (BVH offsets live in the parent)
PRIMARY = {
    "Hips": "Spine", "Spine": "Head",
    "L_UpLeg": "L_Leg", "L_Leg": "L_Foot", "L_Foot": "L_Toe",
    "R_UpLeg": "R_Leg", "R_Leg": "R_Foot", "R_Foot": "R_Toe",
    "L_Shoulder": "L_ForeArm", "L_ForeArm": "L_Hand",
    "R_Shoulder": "R_ForeArm", "R_ForeArm": "R_Hand",
}

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]


def probe(src):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate", "-of", "csv=p=0", src],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(",")
    num, den = out[2].split("/")
    return int(out[0]), int(out[1]), float(num) / float(den)


def read_exact(stream, size):
    data = stream.read(size)
    if not data:
        return None
    while len(data) < size:
        more = stream.read(size - len(data))
        if not more:
            return None
        data += more
    return data


def joint_positions(world):
    """(33,3) world landmarks -> {joint name: (3,)} for the 17 tracked joints."""
    pos = {}
    for name, _, src in SKELETON:
        pos[name] = 0.5 * (world[src[1]] + world[src[2]]) if src[0] == "avg" else world[src[1]]
    return pos


def smooth(arr, k):
    """Moving average over time; arr is (frames, joints, 3)."""
    if k <= 1:
        return arr
    pad = k // 2
    padded = np.pad(arr, ((pad, pad), (0, 0), (0, 0)), mode="edge")
    kernel = np.ones(k) / k
    return np.stack([np.stack([np.convolve(padded[:, j, c], kernel, mode="valid")
                               for c in range(3)], axis=-1) for j in range(arr.shape[1])], axis=1)


def pick_rest_frame(pos):
    """Frame closest to standing with straight limbs - best rest pose for retargeting."""
    best, best_score = 0, -1e9
    for i in range(0, len(pos), max(1, len(pos) // 200)):
        p = {n: pos[i, j] for j, n in enumerate(NAMES)}
        arms = np.linalg.norm(p["L_ForeArm"] - p["L_Shoulder"]) + np.linalg.norm(p["R_ForeArm"] - p["R_Shoulder"])
        reach = np.linalg.norm(p["L_Hand"] - p["L_Shoulder"]) + np.linalg.norm(p["R_Hand"] - p["R_Shoulder"])
        legs = np.linalg.norm(p["L_Foot"] - p["Hips"]) + np.linalg.norm(p["R_Foot"] - p["Hips"])
        score = 0.5 * (reach - arms) + legs
        if score > best_score:
            best, best_score = i, score
    return best


def write_bvh(path, pos, fps, to_cm=100.0):
    """pos: (frames, joints, 3) in metres, Y-up. Writes a 17-joint swing-only BVH."""
    rest = {n: pos[pick_rest_frame(pos), j] for j, n in enumerate(NAMES)}
    offsets, rest_dir = {}, {}
    for name in NAMES:
        if PARENT[name] is not None:
            offsets[name] = (rest[name] - rest[PARENT[name]]) * to_cm
        if name in PRIMARY:
            d = rest[PRIMARY[name]] - rest[name]
            rest_dir[name] = d / max(1e-9, np.linalg.norm(d))
    lift = -min(rest[n][1] for n in ("L_Foot", "R_Foot", "L_Toe", "R_Toe"))  # put feet on the floor

    children = {n: [c for c in NAMES if PARENT[c] == n] for n in NAMES}
    order = ["Hips"]  # channel rows must follow declaration (depth-first) order
    lines = ["HIERARCHY", "ROOT Hips", "{", "\tOFFSET 0.00 0.00 0.00",
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
                tip = offsets[child] / max(1e-9, np.linalg.norm(offsets[child])) * 2.0  # 2 cm tip, else Blender warns
                lines.extend([f"{pad}\tEnd Site", f"{pad}\t{{",
                              f"{pad}\t\tOFFSET {tip[0]:.4f} {tip[1]:.4f} {tip[2]:.4f}", f"{pad}\t}}"])
            lines.append(f"{pad}}}")

    emit("Hips", 1)
    lines.extend(["}", "MOTION", f"Frames: {len(pos)}", f"Frame Time: {1.0 / fps:.8f}"])

    for f in range(len(pos)):
        cur = {n: pos[f, j] for j, n in enumerate(NAMES)}
        rot_world = {}
        for name in NAMES:
            if name in PRIMARY:
                d = cur[PRIMARY[name]] - cur[name]
                rot_world[name] = align_rotation(rest_dir[name], d / max(1e-9, np.linalg.norm(d)))
        for name in NAMES:  # leaves carry no bone, they inherit their parent's frame
            if name not in rot_world:
                rot_world[name] = rot_world[PARENT[name]] if PARENT[name] else np.eye(3)
        row = [cur["Hips"][0] * to_cm, (cur["Hips"][1] + lift) * to_cm, cur["Hips"][2] * to_cm,
               *euler_zxy(rot_world["Hips"])]
        for name in order[1:]:
            row.extend(euler_zxy(rot_world[PARENT[name]].T @ rot_world[name]))
        lines.append(" ".join(f"{v:.4f}" for v in row))

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return len(NAMES) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("outdir")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--model", default=r"C:\Users\princ\models\mediapipe\pose_landmarker_heavy.task")
    ap.add_argument("--smooth", type=int, default=5, help="moving-average window in frames")
    ap.add_argument("--tag", default="pose")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    w, h, fps = probe(a.src)
    n_target = int(round(a.seconds * fps))

    opts = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=a.model),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.4,
        output_segmentation_masks=False,
    )
    landmarker = vision.PoseLandmarker.create_from_options(opts)

    decode = ["ffmpeg", "-v", "error", "-ss", str(a.start), "-i", a.src, "-map", "0:v:0",
              "-frames:v", str(n_target), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    overlay_path = os.path.join(a.outdir, f"overlay_{a.tag}.mp4")
    encode = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
              "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:0",
              "-i", a.src, "-map", "0:v:0", "-map", "1:a:0?",
              "-c:v", "libx264", "-preset", "medium", "-crf", "20",
              "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest", overlay_path]

    dec = subprocess.Popen(decode, stdout=subprocess.PIPE)
    enc = subprocess.Popen(encode, stdin=subprocess.PIPE)

    world_seq, img_seq, vis_seq, missing = [], [], [], 0
    i = 0
    try:
        while True:
            raw = read_exact(dec.stdout, w * h * 3)
            if raw is None:
                break
            bgr = np.frombuffer(raw, np.uint8).reshape(h, w, 3).copy()
            res = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)),
                int(i * 1000 / fps))

            if res.pose_world_landmarks:
                world_seq.append(np.array([[p.x, p.y, p.z] for p in res.pose_world_landmarks[0]], np.float32))
                lms = res.pose_landmarks[0]
                vis_seq.append(np.array([p.visibility for p in lms], np.float32))
                img_seq.append(np.array([[p.x, p.y, p.z] for p in lms], np.float32))
                for p1, p2 in CONNECTIONS:
                    if min(lms[p1].visibility, lms[p2].visibility) < 0.3:
                        continue
                    cv2.line(bgr, (int(lms[p1].x * w), int(lms[p1].y * h)),
                             (int(lms[p2].x * w), int(lms[p2].y * h)), (0, 255, 0), 2, cv2.LINE_AA)
                for lm in lms:
                    if lm.visibility > 0.3:
                        cv2.circle(bgr, (int(lm.x * w), int(lm.y * h)), 3, (0, 0, 255), -1, cv2.LINE_AA)
                label = f"f{i} vis={float(vis_seq[-1].mean()):.2f}"
                colour = (0, 255, 255)
            else:
                missing += 1
                world_seq.append(np.zeros((33, 3), np.float32))
                vis_seq.append(np.zeros(33, np.float32))
                img_seq.append(np.zeros((33, 3), np.float32))
                label, colour = f"f{i} NO POSE", (0, 0, 255)
            cv2.putText(bgr, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2, cv2.LINE_AA)

            enc.stdin.write(bgr.tobytes())
            i += 1
            if i % 100 == 0:
                print(f"{i}/{n_target} frames, missed={missing}", file=sys.stderr, flush=True)
    finally:
        dec.kill()
        enc.stdin.close()
        enc.wait()
        landmarker.close()

    if enc.returncode:
        sys.exit(f"encoder failed ({enc.returncode})")
    world = np.stack(world_seq)
    if missing:
        print(f"warning: {missing}/{len(world)} frames had no pose", file=sys.stderr)

    if float(world[:, L_AN, 1].mean()) > float(world[:, NOSE, 1].mean()):
        # MediaPipe world landmarks are (x right, y DOWN, z away from camera). Flipping only
        # y would leave a left-handed frame, which mirrors the whole motion; flip y and z.
        world[..., 1] *= -1
        world[..., 2] *= -1
        print("converted to y-up / z-toward-camera (handedness preserved)", file=sys.stderr)

    pos = smooth(np.stack([[joint_positions(w)[n] for n in NAMES] for w in world]), a.smooth)

    npz_path = os.path.join(a.outdir, f"{a.tag}.npz")
    np.savez_compressed(npz_path, world=world, image2d=np.stack(img_seq), joints3d=pos,
                        visibility=np.stack(vis_seq), fps=fps, width=w, height=h,
                        start=a.start, joint_names=np.array(NAMES))

    bvh_path = os.path.join(a.outdir, f"{a.tag}.bvh")
    n_bones = write_bvh(bvh_path, pos, fps)

    # self-check: header frame count and channel count must match what we wrote
    with open(bvh_path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    m = lines.index("MOTION")
    header_frames = int(lines[m + 1].split(":")[1])
    rows = lines[m + 3:]
    expect = 6 + 3 * n_bones
    assert header_frames == len(rows) == len(pos), (header_frames, len(rows), len(pos))
    assert all(len(r.split()) == expect for r in rows), f"channel count != {expect}"
    assert np.isfinite(pos).all(), "non-finite joint positions"

    print(f"ok: {len(pos)} frames, {n_bones} bones, missed={missing}", file=sys.stderr)
    print(f"overlay: {overlay_path}\nnpz: {npz_path}\nbvh: {bvh_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

