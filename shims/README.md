# GVHMR on Windows without pytorch3d

GVHMR upstream pins `pytorch3d` to a Linux wheel and Windows has no official build.
`shims/pytorch3d` is a pure-torch replacement for the handful of pytorch3d functions GVHMR
imports, so the pipeline runs natively on Windows.

## What is shimmed

| upstream import | shim | status |
|---|---|---|
| `pytorch3d.transforms.*` (10 functions) | re-implemented in torch | matches pytorch3d (< 1e-6) |
| `pytorch3d.ops.knn.knn_points` | brute-force top-k | exact for the K=1 use in `geo_transform.py` |
| `pytorch3d.renderer.cameras.look_at_rotation` | re-implemented | exact |
| `pytorch3d.renderer.*` / `pytorch3d.structures` | stubs that raise on use | preview rendering is skipped (`--no_render`) |

Checks:

```powershell
$env:PYTHONPATH = "<repo>\shims"
python test_pytorch3d_shim.py      # round-trips, kNN, look_at -> ALL OK
python check_gvhmr_imports.py      # all 12 pytorch3d-touching modules import
```

## No hard-coded paths

Every script takes parameters, with env-var equivalents:

| Thing | Parameter | Env var | Default |
|---|---|---|---|
| GVHMR clone | `-GvhmrRepo` | `GVHMR_HOME` | `$HOME\GVHMR` |
| Python env | `-Venv` | `GVHMR_VENV` | `$HOME\venvs\gvhmr312` |
| Working dir (video, BVH, renders) | `-Work` | `MOCAP_WORK` | `$HOME\mocap-work` |
| Blender | — | `BLENDER_EXE` | newest `blender.exe` under `%ProgramFiles%\Blender Foundation` |
| uv | — | `UV_EXE` | `uv` on PATH, else `$HOME\.local\bin\uv.exe` |
| MediaPipe model | `--model` | `POSE_MODEL` | `$HOME\models\mediapipe\pose_landmarker_heavy.task` |

`$HOME` is `$env:USERPROFILE` on Windows, so nothing depends on the account name.

## Install

```powershell
git clone https://github.com/kalpakjian/taichi-mocap; cd taichi-mocap
git clone https://github.com/zju3dv/GVHMR "$HOME\GVHMR"      # or set GVHMR_HOME

.\shims\install_gvhmr.ps1        # venv + torch 2.3.0+cu121 + deps + hmr4d (editable)
.\shims\download_weights.ps1     # 5.3 GB of checkpoints from a public HF mirror
```

Then apply the `--no_render` patch to the GVHMR clone (it skips the mesh preview, which needs the
real pytorch3d renderer; the motion output is identical):

```powershell
cd "$HOME\GVHMR"
git apply <repo>\shims\gvhmr-demo-no-render.patch
```

## Body models (licence, you must download them)

1. **SMPL-X neutral** — register at https://smpl-x.is.tue.mpg.de/ , download
   `SMPLX_NEUTRAL.npz` (SMPL-X v1.1, **not** the "locked head" variant), then let the helper
   place it in both paths upstream uses:

   ```powershell
   .\shims\install_body_models.ps1                      # scans Downloads + Desktop
   .\shims\install_body_models.ps1 -SearchDir D:\models
   ```

2. **SMPL neutral** (`SMPL_NEUTRAL.pkl`) is only needed for the mesh preview, which is disabled.

Verify everything before a long run:

```powershell
python shims\verify_gvhmr_ready.py
# expects: smplx loaded: 10475 verts, 55 joints ... GVHMR instance ok ... READY
```

## Run

```powershell
# whole clip: needs a constant-frame-rate 30 fps copy in the work dir
ffmpeg -i IN.mp4 -vf fps=30 -c:v libx264 -crf 20 -an "$HOME\mocap-work\source_30fps.mp4"
.\shims\run_full_pipeline.ps1 -Work "$HOME\mocap-work" -CamAngle 90 -CamSide left

# re-render the same BVH from another angle (no GVHMR re-run)
.\shims\render_variant.ps1 -Work "$HOME\mocap-work" -CamAngle 270
```

`-CamAngle` is measured from the performer's facing direction in the first frames; the camera then
stays fixed for the whole clip.

## Known limits of this setup

- No preview mesh videos (software renderer missing). Motion output is unaffected.
- `euler_angles_to_matrix` composes axes in the given order; GVHMR only feeds it single-axis
  rotations, so this cannot differ in practice. Re-check if that changes.
- The ankle→toe axis used by `bvh_preview_blender.py` to find "front" pointed backwards on the
  test footage, which is why both `CAM_ANGLE=90` and `CAM_ANGLE=270` are worth rendering.
