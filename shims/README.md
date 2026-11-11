# GVHMR on Windows without pytorch3d

GVHMR upstream pins `pytorch3d` to a Linux wheel, and Windows has no official build.
`C:\Users\princ\shims\pytorch3d` is a pure-torch replacement for the handful of
pytorch3d functions GVHMR imports, so the pipeline runs natively.

## What is shimmed

| upstream import | shim | status |
|---|---|---|
| `pytorch3d.transforms.*` (10 functions) | re-implemented in torch | matches pytorch3d (verified < 1e-6) |
| `pytorch3d.ops.knn.knn_points` | brute-force top-k | exact for the K=1 use in `geo_transform.py` |
| `pytorch3d.renderer.cameras.look_at_rotation` | re-implemented | exact |
| `pytorch3d.renderer.*` / `pytorch3d.structures` | stubs that raise on use | preview videos disabled (`--no_render`) |

Tests: `python test_pytorch3d_shim.py` (needs `PYTHONPATH=C:\Users\princ\shims`).
Wiring check: `python check_gvhmr_imports.py` — all 12 pytorch3d-touching modules import.

## Environment

- venv: `C:\Users\princ\venvs\gvhmr312` (Python 3.12, torch 2.3.0+cu121, numpy 1.26.4)
- repo: `C:\Users\princ\GVHMR` (installed editable; `--no_render` flag added to `tools/demo/demo.py`)
- deps added beyond upstream `requirements.txt`: `yacs` (missing upstream), `pycolmap` (Windows wheel)

```powershell
cd C:\Users\princ\GVHMR
$env:PYTHONPATH = 'C:\Users\princ\shims'
& 'C:\Users\princ\venvs\gvhmr312\Scripts\python.exe' tools/demo/demo.py `
    --video 'C:\Users\princ\Downloads\陈式太极拳老架一路74式马畅示范_增强720p.mp4' `
    -s --no_render
```

`-s` = static camera (this footage is a fixed tripod shot, so the VO solver is skipped).

## Still missing (licence, must be downloaded by you)

1. **SMPL-X neutral** — register at https://smpl-x.is.tue.mpg.de/ , download
   `SMPLX_NEUTRAL.npz` (SMPL-X v1.0), place it BOTH in:
   - `C:\Users\princ\GVHMR\inputs\checkpoints\body_models\smplx\SMPLX_NEUTRAL.npz`
   - `C:\Users\princ\GVHMR\inputs\models\smplx\SMPLX_NEUTRAL.npz` (upstream uses this path in `make_smplx("neu_fullpose")`)
2. **SMPL neutral** (only needed for the mesh preview videos, which are disabled) —
   https://smpl.is.tue.mpg.de/ -> `SMPL_NEUTRAL.pkl` in `inputs\checkpoints\body_models\smpl\`

Checkpoints are already in place (~5.3 GB, from the `camenduru/GVHMR` HF mirror):
`gvhmr/gvhmr_siga24_release.ckpt`, `hmr2/epoch=10-step=25000.ckpt`,
`vitpose/vitpose-h-multi-coco.pth`, `yolo/yolov8x.pt`.

## Known limits of this setup

- No preview mesh videos (software renderer missing). Motion output is unaffected.
- `euler_angles_to_matrix` composes axes in the given order; GVHMR only feeds it
  single-axis rotations, so this cannot differ in practice. Re-check if that changes.
- Untested end-to-end: the run needs the SMPL-X file above.
