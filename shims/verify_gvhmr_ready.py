"""Is GVHMR ready to run? Body models present, smplx loads, checkpoint instantiates."""
import os
import sys
from pathlib import Path

GV = Path(os.environ.get("GVHMR_HOME", Path.home() / "GVHMR"))
paths = {
    "SMPLX (needed)": GV / "inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz",
    "SMPLX (2nd copy)": GV / "inputs/models/smplx/SMPLX_NEUTRAL.npz",
    "SMPL (render only)": GV / "inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl",
}
missing = 0
for label, p in paths.items():
    ok = p.exists()
    missing += 0 if ok else (1 if "render" not in label else 0)
    size = f"{p.stat().st_size / 1e6:.1f} MB" if ok else "-"
    print(f"{'OK  ' if ok else 'MISS'} {label:<20} {size:>10}  {p}")

if missing:
    print("\n=> still blocked: download SMPLX_NEUTRAL.npz (any name) into Downloads, then run:")
    print("   powershell -ExecutionPolicy Bypass -File C:\\Users\\princ\\shims\\install_body_models.ps1")
    sys.exit(1)

import torch  # noqa: E402
from smplx import SMPLX  # noqa: E402

m = SMPLX(str(paths["SMPLX (needed)"]), use_pca=False, flat_hand_mean=True)
print(f"smplx loaded: {m.v_template.shape[0]} verts, {len(m.parents)} joints, "
      f"faces {getattr(m, 'faces', None) is not None}")

import hydra  # noqa: E402
from hydra import compose, initialize_config_module  # noqa: E402

from hmr4d.configs import register_store_gvhmr  # noqa: E402
from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL  # noqa: E402,F401  (registers the hydra config)

with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
    register_store_gvhmr()
    cfg = compose(config_name="demo", overrides=["video_name=smoke", "static_cam=true"])
model = hydra.utils.instantiate(cfg.model, _recursive_=False)
model.load_pretrained_model(cfg.ckpt_path)
print(f"GVHMR instance ok: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, ckpt loaded")
print("\nREADY")
