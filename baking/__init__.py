"""Texture baking strategies for the UV-decimate pipeline.

* :func:`~baking.tbn_rebaker.rebake_normal_map_uv_space` — Strategy A (``keep_uv``): the
  decimated low-poly keeps the high-poly's UV parameterization, so the existing
  tangent-space normal map is re-expressed in the low-poly's per-texel TBN frames.
* :func:`~baking.tbn_rebaker.bake_pbr_maps_3d_transfer` — Strategy B (``retopo_bake``):
  the low-poly carries a brand-new atlas, so every channel is transferred through a 3D
  tight-cage ray cast against the high-poly.

Both functions take and return plain numpy arrays; ComfyUI's torch helpers are imported
lazily so importing this package stays cheap.
"""
from __future__ import annotations

from .tbn_rebaker import bake_pbr_maps_3d_transfer, rebake_normal_map_uv_space

__all__ = ["rebake_normal_map_uv_space", "bake_pbr_maps_3d_transfer"]
