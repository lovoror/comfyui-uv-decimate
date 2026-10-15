"""Routing between the two decimation strategies.

The router exists because the two mechanisms fail in *opposite* places, and picking wrong is
expensive in both directions:

* 5D texture-metric QEM is fast, preserves the albedo/metallic-roughness/emissive texels bit for
  bit and only has to rebake the normal map -- but ``preserveboundary=True`` turns every UV-seam
  vertex into a lock.  A vertex with two or more distinct UVs has to stay (dropping it would
  fold the charts together), and every lock keeps the triangles around it alive: roughly
  ``1.35 * seam_lock_floor`` triangles are immovable.  Below that budget the collapser stalls
  far above the target and hands back a mesh that is not low-poly at all.
* UV-free simplification plus ``xatlas`` plus a PBR bake has no such floor -- it is the only way
  to reach 5k-35k faces on a mesh with tens of thousands of chart borders -- but it resamples
  every colour texel and rebuilds the atlas, so it is slower and slightly softer.

So: keep the atlas whenever the budget clears the seam lock floor, and rebuild the atlas when it
does not.  The decision is made from the mesh's own measurements, and a 5D pass that stalls
anyway is caught afterwards and re-run through the UV-free path.
"""

from __future__ import annotations

import logging

try:  # normal case: imported as part of the comfyui-uv-decimate package
    from .base import DecimationStrategy
    from .retopo_bake import RetopoBakeStrategy
    from .tex_qem import TextureQEMStrategy
except ImportError:  # loaded as a plain directory rather than a package
    from base import DecimationStrategy
    from retopo_bake import RetopoBakeStrategy
    from tex_qem import TextureQEMStrategy

try:
    from ..contracts import DecimateConfig, MeshPayload
except ImportError:  # loaded as a plain directory rather than a package
    from contracts import DecimateConfig, MeshPayload

LOGGER = logging.getLogger(__name__)

TEXTURE_QEM = "tex_qem"
RETOPO_BAKE = "retopo_bake"
#: ``keep_uv`` was the original name of the atlas-preserving strategy; keep it working.
TEXTURE_QEM_ALIASES = (TEXTURE_QEM, "keep_uv")
#: Headroom over the 1.35 triangles-per-locked-seam-vertex estimate, so a mesh whose charts are
#: worse than average still clears its floor instead of stalling in the collapser.
SEAM_LOCK_HEADROOM = 1.45
#: ... and a floor on that floor: a budget under ~5.5% of the input has to rebuild the atlas
#: whatever the seams say, because a collapse that large cannot stay faithful to any atlas.
FACE_SHARE_FLOOR = 0.055
#: A 5D result above this multiple of the target means it stalled on its seam locks.
TEXTURE_QEM_FACE_TOLERANCE = 1.25


class AutoRouterStrategy(DecimationStrategy):
    """Pick the cheaper strategy that can still reach the requested face count."""

    name = "auto"

    def __init__(self) -> None:
        self.texture_qem = TextureQEMStrategy()
        self.retopo_bake = RetopoBakeStrategy()
        #: Which strategy actually ran last, for the node layer's report.
        self.last_route = self.name

    def select(self, mesh_data: MeshPayload, config: DecimateConfig) -> str:
        """Name the strategy that should run, without running it."""
        requested = str(config.strategy or "auto").lower()
        if requested in TEXTURE_QEM_ALIASES:
            return TEXTURE_QEM
        if requested == RETOPO_BAKE:
            return RETOPO_BAKE
        if requested != self.name:
            raise ValueError("unknown strategy %r; expected one of %s"
                             % (config.strategy, sorted(TEXTURE_QEM_ALIASES
                                                        + (self.name, RETOPO_BAKE))))
        if mesh_data.uvs is None:
            # Nothing to preserve and nothing to tear: the texture-coordinate quadrics would be
            # dead weight, so the plain 3D collapse in TextureQEMStrategy is the right one.
            return TEXTURE_QEM
        metrics = self.analyze_uv_topology(mesh_data.vertices, mesh_data.faces, mesh_data.uvs)
        lock_floor = int(metrics["seam_lock_floor"] * SEAM_LOCK_HEADROOM)
        share_floor = int(mesh_data.face_count * FACE_SHARE_FLOOR)
        return TEXTURE_QEM if int(config.target_faces) >= max(lock_floor, share_floor) \
            else RETOPO_BAKE

    def execute(self, mesh_data: MeshPayload, config: DecimateConfig) -> MeshPayload:
        """Run the selected strategy, falling back when 5D QEM proves unable to reach budget."""
        route = self.select(mesh_data, config)
        self.last_route = route
        if route == RETOPO_BAKE:
            return self.retopo_bake.execute(mesh_data, config)
        result = self.texture_qem.execute(mesh_data, config)
        if self._stalled(result, config):
            LOGGER.warning("%s: 5D QEM stalled at %d faces for a target of %d (its UV seams "
                           "are locked); re-running through %s", self.name, result.face_count,
                           int(config.target_faces), RETOPO_BAKE)
            self.last_route = RETOPO_BAKE
            return self.retopo_bake.execute(mesh_data, config)
        return result

    @staticmethod
    def _stalled(result: MeshPayload, config: DecimateConfig) -> bool:
        """True when the 5D collapse could not honour its budget because of its seam locks."""
        face_count = int(result.face_count)
        target = max(int(config.target_faces), 1)
        return face_count == 0 or face_count > target * TEXTURE_QEM_FACE_TOLERANCE


__all__ = ["AutoRouterStrategy", "FACE_SHARE_FLOOR", "SEAM_LOCK_HEADROOM", "RETOPO_BAKE",
           "TEXTURE_QEM", "TEXTURE_QEM_FACE_TOLERANCE", "TEXTURE_QEM_ALIASES"]
