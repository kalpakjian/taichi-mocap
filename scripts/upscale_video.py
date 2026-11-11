"""Stream a video through an ESRGAN/SwinIR-style model (spandrel) and re-encode.

Decode -> GPU upscale -> encode, all through pipes so no intermediate frames hit disk.

  python upscale_video.py in.flv out.mp4 --model C:\\Users\\princ\\models\\esrgan\\RealESRGAN_x2plus.pth --denoise
"""

import argparse
import subprocess
import sys
import time

import numpy as np
import torch
from spandrel import ModelLoader


def probe(src):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", src],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(",")
    return int(out[0]), int(out[1])


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--model", required=True)
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--denoise", action="store_true", help="light hqdn3d before upscaling (helps low-bitrate sources)")
    p.add_argument("--crf", type=int, default=18)
    p.add_argument("--half", action="store_true", help="fp16 inference (RRDBNet/ESRGAN only)")
    p.add_argument("--limit", type=int, default=0, help="stop after N frames (smoke test)")
    a = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        desc = ModelLoader().load_from_file(a.model)
        model, scale = desc.model.eval().to(dev), desc.scale
    except Exception as e:
        sys.exit(f"cannot load model {a.model}: {e}")
    if a.half:
        model.half()

    w, h = probe(a.src)
    vf = ["hqdn3d=1.5:1.5:6:6"] if a.denoise else []

    decode = ["ffmpeg", "-v", "error", "-i", a.src, "-map", "0:v:0",
              *(["-vf", ",".join(vf)] if vf else []),
              "-fps_mode", "cfr", "-r", str(a.fps), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    encode = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
              "-s", f"{w * scale}x{h * scale}", "-r", str(a.fps), "-i", "pipe:0",
              "-i", a.src, "-map", "0:v:0", "-map", "1:a:0?",
              "-c:v", "libx264", "-preset", "medium", "-crf", str(a.crf),
              "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", a.dst]

    dec = subprocess.Popen(decode, stdout=subprocess.PIPE)
    enc = subprocess.Popen(encode, stdin=subprocess.PIPE)
    frame_bytes = w * h * 3
    n, t0 = 0, time.time()
    try:
        while a.limit == 0 or n < a.limit:
            raw = read_exact(dec.stdout, frame_bytes)
            if raw is None:
                break
            x = torch.from_numpy(np.frombuffer(raw, np.uint8).reshape(h, w, 3).copy()).to(dev)
            x = x.permute(2, 0, 1).unsqueeze(0).float().div_(255)
            if a.half:
                x = x.half()
            with torch.inference_mode():
                y = model(x).clamp_(0, 1)
            out = (y[0].permute(1, 2, 0).float().cpu().numpy() * 255 + 0.5).astype(np.uint8)
            enc.stdin.write(out.tobytes())
            n += 1
            if n % 100 == 0:
                print(f"{n} frames, {n / (time.time() - t0):.1f} fps", file=sys.stderr, flush=True)
    except BrokenPipeError:
        sys.exit("encoder died early - check ffmpeg output above")
    finally:
        dec.kill()
        enc.stdin.close()
        enc.wait()

    if enc.returncode:
        sys.exit(f"encoder failed with code {enc.returncode}")
    print(f"done: {n} frames -> {a.dst} in {time.time() - t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
