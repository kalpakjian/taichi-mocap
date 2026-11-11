class _Unavailable:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "pytorch3d.structures is not available in the shim (visualisation only). "
            "Run GVHMR without the rendering step."
        )


class Meshes(_Unavailable):
    pass


def join_meshes_as_scene(*args, **kwargs):
    raise NotImplementedError("pytorch3d.structures.join_meshes_as_scene is a shim stub (rendering only)")
