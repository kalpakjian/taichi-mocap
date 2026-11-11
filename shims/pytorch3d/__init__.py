"""Pure-torch stand-in for the parts of pytorch3d that GVHMR imports.

Only what GVHMR touches is implemented: rotation conversions (pytorch3d.transforms),
knn_points (pytorch3d.ops.knn) and look_at_rotation. The rendering classes raise a
clear error instead of pretending to work - GVHMR only uses them for preview videos.

Install by putting this folder on PYTHONPATH (see install note in renderer/__init__.py).
"""

__version__ = "0.7.6+shim"
