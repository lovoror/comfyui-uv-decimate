"""5D texture-metric quadric error collapse that keeps the input atlas.

The idea is Garland & Heckbert's: measure how much a collapse changes the surface and always
take the cheapest one.  Here the metric lives in five dimensions -- three of position plus the
two texture coordinates -- so a collapse that drags a chart border across itself (and would
smear a whole texture island) costs far more than one inside a flat chart.  The atlas therefore
survives the collapse, and only the *tangent-space normal map* has to be re-expressed, because
its texels encode directions in the old high-poly TBN frames.

That last part is what makes this strategy cheap: albedo, metallic-roughness and emissive texels
still address the same surface points through the same UVs, so they are passed through
untouched -- no re-sampling, no blur -- while the normal map is recomposed through the low-poly
frames by :func:`baking.tbn_rebaker.rebake_normal_map_uv_space`.
"""

from __future__ import annotations

import logging

import numpy as np

try:  # the pipeline module is the sibling of this package
    from .. import uv_pipeline
except ImportError:  # loaded as a plain directory: put the node directory on the path
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import uv_pipeline

try:  # normal case: imported as part of the comfyui-uv-decimate package
    from .base import (
        PLANAR_QUADRIC,
        DecimationStrategy,
        drop_degenerate_faces,
        ensure_tbn,
        from_pymeshlab,
        rebuild_payload,
        to_pymeshlab,
    )
except ImportError:  # loaded as a plain directory rather than a package
    from base import (
        PLANAR_QUADRIC,
        DecimationStrategy,
        drop_degenerate_faces,
        ensure_tbn,
        from_pymeshlab,
        rebuild_payload,
        to_pymeshlab,
    )

try:
    from ..baking.tbn_rebaker import NORMAL_CHANNEL, rebake_normal_map_uv_space
    from ..contracts import DecimateConfig, MeshPayload
except ImportError:  # loaded as a plain directory rather than a package
    from baking.tbn_rebaker import NORMAL_CHANNEL, rebake_normal_map_uv_space
    from contracts import DecimateConfig, MeshPayload

LOGGER = logging.getLogger(__name__)


class TextureQEMStrategy(DecimationStrategy):
    """Decimate with 5D (position + UV) quadrics, then rebake only the normal map."""

    name = "tex_qem"

    def execute(self, mesh_data: MeshPayload, config: DecimateConfig) -> MeshPayload:
        """Collapse ``mesh_data`` to the target face count, keeping its atlas."""
        mesh_set = to_pymeshlab(mesh_data.vertices, mesh_data.faces, mesh_data.uvs)
        self._clean(mesh_set, config)
        self._collapse(mesh_set, mesh_data.uvs is not None, config)
        vertices, faces, uvs = from_pymeshlab(mesh_set)
        if mesh_data.uvs is not None and uvs is None:
            LOGGER.warning("%s: the 5D collapser returned no texture coordinates at all; the "
                           "input atlas was lost and the normal map cannot be rebaked",
                           self.name)
        vertices, faces, uvs = drop_degenerate_faces(vertices, faces, uvs)
        self.require_faces(self.name, faces.shape[0])
        normals, tangents = ensure_tbn(vertices, faces, uvs, config)
        textures = self._rebake(mesh_data, vertices, faces, uvs, normals, tangents, config)
        return rebuild_payload(mesh_data, vertices=vertices, faces=faces, uvs=uvs,
                               normals=normals, tangents=tangents, textures=textures)

    @staticmethod
    def _clean(mesh_set, config: DecimateConfig) -> None:
        """Weld the 3D vertices a UV split duplicated, *before* any quadric is built.

        A textured GLB stores one vertex per (position, uv) pair, so every chart border also
        looks like an open 3D boundary: the input of ``_raw_ship_00016.glb`` carries 476,355
        vertices for 345,301 distinct positions (exact equality and rounding at 5, 6, 7 and 8
        decimals all agree, and ``meshing_remove_duplicate_vertices`` welds it to the same
        345,301). Left unwelded, the quadric builder measures each of those borders as a crack
        and the collapser cannot cross it: measured on that asset, a 5D collapse to 50,000 faces
        stalls at 252,089 faces without this pass and reaches exactly 50,000 faces with it.
        ``uv_pipeline.clean`` welds the positions while preserving the per-corner UVs, so
        running it *after* the collapse is useless.
        """
        if getattr(config, "clean", True):
            uv_pipeline.clean(mesh_set)

    def _collapse(self, mesh_set, has_uvs: bool, config: DecimateConfig):
        """Collapse with the strongest collapser this pymeshlab build offers.

        The 5D texture-metric collapser needs the per-corner texture coordinates to be
        *complete*: a wedge table whose texture indices were never set makes it refuse the whole
        mesh ("Mesh has some inconsistent tex coordinates"), which is why
        ``uv_pipeline.build_mesh_set`` fills the table through
        ``compute_texcoord_transfer_vertex_to_wedge`` instead of the ``w_tex_coords_matrix``
        constructor argument. Measured on ``_raw_ship_00016.glb``: 698,337 -> 50,000 faces in
        8.6 s with no warning. No texture *image* is required for the metric -- the quadrics come
        from the wedge coordinates alone.
        """
        if mesh_set.current_mesh().face_number() <= int(config.target_faces):
            return mesh_set
        if not (has_uvs and self._collapse_with_texture(mesh_set, config)):
            self._collapse_plain(mesh_set, config, has_uvs)
        return mesh_set

    def _collapse_with_texture(self, mesh_set, config: DecimateConfig) -> bool:
        """5D collapser: position quadrics plus texture-coordinate quadrics.

        ``extratcoordw`` is the weight of the two texture-coordinate quadrics -- at 1.0 a UV
        fold costs as much as a positional one -- and ``preserveboundary`` forbids collapsing
        the boundary vertices themselves, so the chart borders the metric just paid for cannot
        be flattened away afterwards.  ``planarweight`` and ``autoclean`` are not parameters of
        this filter (``planarquadric`` is), which is why the two paths build their own kwargs.

        ``preservenormal`` comes from ``config.preserve_normal`` (``True`` by default) rather
        than :data:`~strategies.base.PRESERVE_NORMAL`: with it off, the 5D metric is free to
        invert a triangle's normal on a thin region -- measured on ``_raw_ship_00016.glb``, the
        green cloth sail comes back from a 50,000-face collapse with black spots where triangles
        flipped through the surface.  Locking the dense mesh's face normals costs nothing here:
        the same asset still reaches exactly 50,000 faces cleanly with ``preservenormal=True``.
        """
        try:
            mesh_set.apply_filter(
                "meshing_decimation_quadric_edge_collapse_with_texture",
                targetfacenum=int(config.target_faces),
                qualitythr=float(config.quality_thr),
                extratcoordw=float(config.texture_weight),
                preserveboundary=bool(config.preserve_boundary),
                optimalplacement=True,
                preservenormal=bool(config.preserve_normal),
                planarquadric=PLANAR_QUADRIC,
            )
        except Exception as error:
            LOGGER.warning("%s: the 5D texture-metric collapser is unavailable (%s); using the "
                           "3D collapser, which still carries the per-corner UVs", self.name,
                           error)
            return False
        return True

    @staticmethod
    def _collapse_plain(mesh_set, config: DecimateConfig, has_uvs: bool) -> None:
        """3D quadric collapse that keeps the atlas.

        ``autoclean`` runs only on a mesh without an atlas.  Measured on a UV-split sphere it
        keeps the atlas readable but drops ~15% of the faces the collapser was asked to produce,
        while the split vertex count stays the same -- it deletes unreferenced vertices, and the
        split is driven by genuinely distinct texture coordinates, not by orphans.  So on a mesh
        with an atlas it costs face budget and buys nothing.  The collapser interpolates the
        wedge table either way, so a low-poly's UVs are still the input's UVs.

        Like the 5D path, this one honours ``config.preserve_normal`` so a forced
        ``strategy="tex_qem"`` cannot silently reintroduce the inverted triangles the 5D
        collapser is protected against.
        """
        mesh_set.apply_filter(
            "meshing_decimation_quadric_edge_collapse",
            targetfacenum=int(config.target_faces),
            qualitythr=float(config.quality_thr),
            preserveboundary=bool(config.preserve_boundary),
            planarquadric=PLANAR_QUADRIC,
            planarweight=float(config.planar_weight),
            optimalplacement=True,
            preservenormal=bool(config.preserve_normal),
            autoclean=not has_uvs,
        )

    @staticmethod
    def _rebake(mesh_data: MeshPayload, vertices: np.ndarray, faces: np.ndarray,
                uvs: np.ndarray, normals: np.ndarray, tangents: np.ndarray,
                config: DecimateConfig) -> dict:
        """Re-express the input's tangent-space normal map in the low-poly's TBN frames.

        Everything else is handed back verbatim: the UVs did not change, so those texels still
        address the same surface points and re-sampling them would only add blur.
        """
        textures = dict(mesh_data.textures or {})
        normal_map = textures.get(NORMAL_CHANNEL)
        if not (config.rebake_normal_map and normal_map is not None and uvs is not None
                and mesh_data.uvs is not None):
            return textures
        high_normals, high_tangents = ensure_tbn(
            mesh_data.vertices, mesh_data.faces, mesh_data.uvs, config,
            normals=mesh_data.normals, tangents=mesh_data.tangents)
        textures[NORMAL_CHANNEL] = rebake_normal_map_uv_space(
            mesh_data.vertices, mesh_data.faces, mesh_data.uvs, high_normals, high_tangents,
            normal_map,
            vertices, faces, uvs, normals, tangents,
            resolution=int(config.bake_resolution),
        )
        return textures


__all__ = ["TextureQEMStrategy"]
