"""comfyui-uv-decimate: keep the textures when a mesh is simplified.

Adds one node, "Decimate Mesh (Keep UV)" (3d/decimate), which reduces a mesh's face count while
keeping its UV set and its PBR material:

* above the UV seam floor a 5D texture-metric QEM collapse keeps the existing atlas,
* below it the mesh is retopologised UV-free and every channel is baked onto a fresh xatlas
  atlas by a GPU BVH PBR bake,
* smooth welded vertex normals and tangents are always rebuilt, and a tangent-space normal map
  is re-projected into the new low-poly TBN frames.

Modules
-------
``uv_pipeline``        end-to-end pipeline used by the node (``run``).
``nodes``              the ComfyUI node class and its mappings.
``contracts``          payload/config/metric types shared by the strategies.
``mesh_convert``       ComfyUI <-> payload conversion (``to_payload``, ``from_payload``).
``mesh_normals``       welded vertex normals and tangents.
``strategies``         decimation strategies (``auto_router.AutoRouterStrategy``).
``baking``             tangent-space normal rebake and PBR map baking.

``mesh_convert`` / ``mesh_normals`` are the implementation modules themselves (external scripts keep
importing them by their bare name) and ``uv_pipeline`` adds the payload-based ``unpack_mesh`` /
``pack_mesh`` aliases around the public ``run`` entry point.
"""
from __future__ import annotations

try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:  # loaded as a plain file rather than a package
    import importlib.util as _ilu
    from pathlib import Path as _Path

    _spec = _ilu.spec_from_file_location(
        "comfyui_uv_decimate_nodes", _Path(__file__).resolve().with_name("nodes.py")
    )
    _mod = _ilu.module_from_spec(_spec)
    assert _spec is not None and _spec.loader is not None
    _spec.loader.exec_module(_mod)
    NODE_CLASS_MAPPINGS = _mod.NODE_CLASS_MAPPINGS
    NODE_DISPLAY_NAME_MAPPINGS = _mod.NODE_DISPLAY_NAME_MAPPINGS

VERSION = "0.1.0"
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "VERSION", "WEB_DIRECTORY"]
