"""Rendering stand-in.

GVHMR only uses pytorch3d.renderer for preview videos. The classes below import fine
but refuse to run, so visualisation has to be skipped (renderer-free runs give the same
motion output). Replace this package with a real pytorch3d build if you want the videos.
"""

import torch
import torch.nn.functional as F


class _Unavailable:
    _what = "pytorch3d renderer"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            f"{self._what} is not available in the pytorch3d shim. "
            "Run GVHMR without the visualisation step (keep the motion output), "
            "or install a real pytorch3d."
        )


class PerspectiveCameras(_Unavailable):
    _what = "PerspectiveCameras"


class TexturesVertex(_Unavailable):
    _what = "TexturesVertex"


class PointLights(_Unavailable):
    _what = "PointLights"


class Materials(_Unavailable):
    _what = "Materials"


class RasterizationSettings(_Unavailable):
    _what = "RasterizationSettings"


class MeshRenderer(_Unavailable):
    _what = "MeshRenderer"


class MeshRasterizer(_Unavailable):
    _what = "MeshRasterizer"


class SoftPhongShader(_Unavailable):
    _what = "SoftPhongShader"


def look_at_rotation(camera_position, at=None, up=None, eps=1e-5, **kwargs):
    """World -> camera rotation matrix, same convention as pytorch3d (z axis looks at `at`)."""
    if at is None:
        at = torch.zeros_like(camera_position)
    if up is None:
        up = torch.tensor([0.0, 1.0, 0.0], device=camera_position.device).expand_as(camera_position)
    z_axis = F.normalize(at - camera_position, eps=eps)
    x_axis = F.normalize(torch.cross(up, z_axis, dim=1), eps=eps)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=1), eps=eps)
    return torch.stack([x_axis, y_axis, z_axis], dim=2)
