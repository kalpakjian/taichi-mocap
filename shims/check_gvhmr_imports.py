"""Import smoke test: every GVHMR module that touches pytorch3d, plus CUDA."""
import importlib
import sys
import traceback

import torch

print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
import pytorch3d

print("pytorch3d shim", pytorch3d.__version__, pytorch3d.__file__)

targets = [
    "hmr4d.utils.geo_transform",
    "hmr4d.utils.geo.hmr_global",
    "hmr4d.utils.geo.flip_utils",
    "hmr4d.utils.body_model.smpl_lite",
    "hmr4d.utils.body_model.smplx_lite",
    "hmr4d.utils.vis.renderer",
    "hmr4d.utils.wis3d_utils",
    "hmr4d.model.gvhmr.utils.endecoder",
    "hmr4d.model.gvhmr.utils.postprocess",
    "hmr4d.model.gvhmr.pipeline.gvhmr_pipeline",
    "hmr4d.utils.preproc",
    "hmr4d.utils.smplx_utils",
]
bad = 0
for name in targets:
    try:
        importlib.import_module(name)
        print("OK  ", name)
    except Exception as exc:  # noqa: BLE001 - report every failure, keep going
        bad += 1
        print("FAIL", name, "->", type(exc).__name__, exc)
print("failures:", bad)
sys.exit(1 if bad else 0)
