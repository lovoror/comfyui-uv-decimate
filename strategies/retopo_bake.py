"""UV-free simplification, ``xatlas`` re-parameterization and GPU PBR surface transfer.

The input atlas is treated as disposable.  The surface is welded, simplified in three
dimensions, given a brand new ``xatlas`` atlas, and every PBR channel is then projected onto
that atlas through a tight-cage GPU BVH bake.  This is the only way to reach very low face
budgets: once a mesh's chart borders lock more triangles than the budget allows, a
texture-metric collapser physically cannot move -- it would have to tear the atlas apart -- so
the atlas has to be rebuilt from scratch instead.

Two mechanisms come from ``huxingyi/autoremesher``:

* ``AutoRemesher::decimateIfTooDense`` (``autoremesher.cpp:146-261``) never collapses a dense
  mesh in one go.  It first targets a cheap intermediate triangle count so the expensive stage
  -- here the optional isotropic remesh, which is quadratic in triangle count -- works on a
  quiet mesh instead of fighting dense input noise.
* ``AutoRemesher::initializeVoxelSize`` (``autoremesher.cpp:128-135``) derives that stage's edge
  length from the target triangle count, so the remesh lands on budget by construction rather
  than by iteration.
"""

from __future__ import annotations

import logging

import numpy as np
import pymeshlab as pml
import pyfqmr
import xatlas

try:  # normal case: imported as part of the comfyui-uv-decimate package
    from .base import (
        PLANAR_QUADRIC,
        PRESERVE_NORMAL,
        DecimationStrategy,
        compute_equilateral_target_edge_length,
        drop_degenerate_faces,
        ensure_tbn,
        from_pymeshlab,
        rebuild_payload,
        split_to_welded_3d,
        to_pymeshlab,
    )
except ImportError:  # loaded as a plain directory rather than a package
    from base import (
        PLANAR_QUADRIC,
        PRESERVE_NORMAL,
        DecimationStrategy,
        compute_equilateral_target_edge_length,
        drop_degenerate_faces,
        ensure_tbn,
        from_pymeshlab,
        rebuild_payload,
        split_to_welded_3d,
        to_pymeshlab,
    )

try:
    from ..baking.tbn_rebaker import bake_pbr_maps_3d_transfer
    from ..contracts import DecimateConfig, MeshPayload
except ImportError:  # loaded as a plain directory rather than a package
    from baking.tbn_rebaker import bake_pbr_maps_3d_transfer
    from contracts import DecimateConfig, MeshPayload

LOGGER = logging.getLogger(__name__)

#: Texels of empty gutter xatlas leaves between charts.  The baker dilates every map outwards,
#: so without a gutter a chart would bleed into its neighbour along the seam.
ATLAS_PADDING = 4
#: Crease angle used for the remesher's feature edges when the config does not set one.
DEFAULT_FEATURE_ANGLE = 45.0
#: Isotropic remesh iterations when ``config.remesh_iterations`` is left at its default of 0.
DEFAULT_REMESH_ITERATIONS = 3
#: A mesh more than this many times over budget is pre-decimated first (``decimateIfTooDense``).
PRE_DECIMATE_TRIGGER = 4
#: ... down to this multiple of the target (``decimateIfTooDense`` uses 3x).
PRE_DECIMATE_FACTOR = 3
#: pyfqmr has no quality threshold; its aggressiveness is the equivalent speed/quality knob.
PYFQMR_AGGRESSIVENESS = 7.0


def simplify_3d(vertices: np.ndarray, faces: np.ndarray, target_faces: int,
                config: DecimateConfig):
    """Collapse the UV-free surface to ``target_faces`` triangles.

    pymeshlab runs MeshLab's quadric collapser (with the rank-3 planar quadric and the optional
    boundary lock); when that collapser is unavailable the Same-Surface-Error-metric
    implementation in ``pyfqmr`` takes over, so the face budget is honoured either way.
    """
    target = max(int(target_faces), 1)
    if int(faces.shape[0]) <= target:
        return vertices, faces
    try:
        mesh_set = to_pymeshlab(vertices, faces)
        mesh_set.apply_filter(
            "meshing_decimation_quadric_edge_collapse",
            targetfacenum=target,
            qualitythr=float(config.quality_thr),
            preserveboundary=bool(config.preserve_boundary),
            planarquadric=PLANAR_QUADRIC,
            planarweight=float(config.planar_weight),
            optimalplacement=True,
            preservenormal=PRESERVE_NORMAL,
            autoclean=True,
        )
        out_vertices, out_faces, _ = from_pymeshlab(mesh_set)
        if int(out_faces.shape[0]) > 0:
            return out_vertices, out_faces
        LOGGER.warning("pymeshlab collapser returned an empty mesh; falling back to pyfqmr")
    except Exception as error:  # pymeshlab missing, or the filter refused this mesh
        LOGGER.warning("pymeshlab collapser failed (%s); falling back to pyfqmr", error)
    return simplify_3d_pyfqmr(vertices, faces, target, config)


def simplify_3d_pyfqmr(vertices: np.ndarray, faces: np.ndarray, target_faces: int,
                       config: DecimateConfig):
    """``pyfqmr`` fallback: a plain 3D quadric collapse to the face budget.

    ``preserve_border`` is pyfqmr's equivalent of MeshLab's ``preserveboundary``; it keeps the
    open boundaries of a non-watertight mesh from eroding away.
    """
    simplifier = pyfqmr.Simplify()
    simplifier.setMesh(
        np.ascontiguousarray(vertices, dtype=np.float64),
        np.ascontiguousarray(faces, dtype=np.int32),
    )
    simplifier.simplify_mesh(
        target_count=max(int(target_faces), 1),
        aggressiveness=PYFQMR_AGGRESSIVENESS,
        preserve_border=bool(config.preserve_boundary),
        verbose=False,
    )
    out_vertices, out_faces, _ = simplifier.getMesh()
    return (np.asarray(out_vertices, dtype=np.float32).reshape(-1, 3),
            np.asarray(out_faces, dtype=np.int64).reshape(-1, 3))


def isotropic_remesh(vertices: np.ndarray, faces: np.ndarray, target_faces: int,
                     config: DecimateConfig):
    """Isotropic explicit remeshing at the equilateral edge length for ``target_faces``.

    A remesh redistributes triangles uniformly instead of collapsing the existing ones, which is
    what keeps very low budgets clean (regular triangles) rather than stretched.  ``targetlen``
    is the edge length whose equilateral triangles would fill the surface with ``target_faces``
    of them (``AutoRemesher::initializeVoxelSize``), so the remesh itself lands close to budget
    and the collapser afterwards only has to trim.

    Returns ``(vertices, faces)``; raises on a mesh without area, since no edge length exists.
    """
    edge_length = compute_equilateral_target_edge_length(vertices, faces, target_faces)
    if not (edge_length > 0.0):
        raise ValueError("cannot remesh a mesh with no surface area")
    mesh_set = to_pymeshlab(vertices, faces)
    mesh_set.apply_filter(
        "meshing_isotropic_explicit_remeshing",
        iterations=int(config.remesh_iterations) or DEFAULT_REMESH_ITERATIONS,
        adaptive=True,
        targetlen=pml.PureValue(float(edge_length)),
        featuredeg=float(config.crease_angle) if config.crease_angle else DEFAULT_FEATURE_ANGLE,
        checksurfdist=True,
    )
    out_vertices, out_faces, _ = from_pymeshlab(mesh_set)
    return out_vertices, out_faces


def repack_uv_xatlas(vertices: np.ndarray, faces: np.ndarray, resolution: int = 2048,
                     padding: int = ATLAS_PADDING):
    """Build a fresh atlas with ``xatlas``; returns ``(vertices, faces, uvs)``.

    xatlas splits a vertex wherever two charts need different texture coordinates, so the mesh
    that comes back has *more* vertices than the one that went in: the returned faces index the
    returned vertices and the returned UVs live in ``[0, 1]`` with ``padding`` texels of gutter.
    The input atlas is ignored -- wielding ``pack_options`` here is what makes the bake
    resolution authoritative instead of a leftover from the old atlas.
    """
    positions = np.ascontiguousarray(vertices, dtype=np.float32)
    indices = np.ascontiguousarray(faces, dtype=np.uint32)
    chart_options = xatlas.ChartOptions()
    pack_options = xatlas.PackOptions()
    pack_options.resolution = int(resolution)
    pack_options.padding = int(padding)
    pack_options.bilinear = True
    pack_options.rotate_charts = True
    pack_options.create_image = False
    atlas = xatlas.Atlas()
    atlas.add_mesh(positions, indices)
    try:
        atlas.generate(chart_options, pack_options)
        mapping, new_indices, uvs = atlas.get_mesh(0)
    except Exception as error:
        raise RuntimeError("xatlas could not parameterize this mesh (%s); a UV-free "
                           "simplification needs a manifold, mostly-closed surface" % error)
    return (positions[np.asarray(mapping, dtype=np.int64).reshape(-1)].reshape(-1, 3),
            np.asarray(new_indices, dtype=np.int64).reshape(-1, 3),
            np.asarray(uvs, dtype=np.float32).reshape(-1, 2))


class RetopoBakeStrategy(DecimationStrategy):
    """Rebuild the atlas from scratch and bake every PBR channel onto it."""

    name = "retopo_bake"

    def execute(self, mesh_data: MeshPayload, config: DecimateConfig) -> MeshPayload:
        """Simplify the welded surface, re-parameterize it and transfer the input's maps."""
        vertices, faces = split_to_welded_3d(mesh_data.vertices, mesh_data.faces)
        vertices, faces = self._pre_decimate(vertices, faces, config)
        vertices, faces = self._simplify(vertices, faces, config)
        low_vertices, low_faces, low_uvs = repack_uv_xatlas(
            vertices, faces, resolution=int(config.bake_resolution), padding=ATLAS_PADDING)
        low_vertices, low_faces, low_uvs = drop_degenerate_faces(
            low_vertices, low_faces, low_uvs)
        self.require_faces(self.name, low_faces.shape[0])
        if low_uvs is None:
            raise RuntimeError("%s: xatlas returned no texture coordinates" % self.name)
        normals, tangents = ensure_tbn(low_vertices, low_faces, low_uvs, config)
        low_mesh = MeshPayload(vertices=low_vertices, faces=low_faces, uvs=low_uvs,
                               normals=normals, tangents=tangents)
        LOGGER.info("%s: %d vertices / %d faces -> %d vertices / %d faces", self.name,
                    mesh_data.vertex_count, mesh_data.face_count,
                    low_mesh.vertex_count, low_mesh.face_count)
        textures = self._bake(mesh_data, low_mesh, config)
        return rebuild_payload(mesh_data, vertices=low_mesh.vertices, faces=low_mesh.faces,
                               uvs=low_mesh.uvs, normals=low_mesh.normals,
                               tangents=low_mesh.tangents, textures=textures)

    def _pre_decimate(self, vertices: np.ndarray, faces: np.ndarray, config: DecimateConfig):
        """Cheaply cut a very dense mesh down before the expensive stage.

        ``AutoRemesher::decimateIfTooDense`` does exactly this: a mesh more than four times over
        budget is first reduced to roughly three times the target, so that whatever runs next
        (the quadratic isotropic remesh, or the final collapser) is not dominated by input noise.
        The intermediate pass uses pyfqmr, which is the fastest collapser available here.
        """
        target = max(int(config.target_faces), 1)
        face_count = int(faces.shape[0])
        if face_count <= target * PRE_DECIMATE_TRIGGER:
            return vertices, faces
        budget = max(target * PRE_DECIMATE_FACTOR, target)
        return simplify_3d_pyfqmr(vertices, faces, min(face_count, budget), config)

    def _simplify(self, vertices: np.ndarray, faces: np.ndarray, config: DecimateConfig):
        """Reach the face budget: optional isotropic remesh, then the UV-free collapser."""
        target = max(int(config.target_faces), 1)
        if not config.isotropic_remesh:
            # Default path: a direct UV-free quadric collapse.  It keeps the model's own
            # silhouette, so 5k-35k faces stay crisp game-ready low-poly instead of a
            # re-sampled approximation of themselves.
            return simplify_3d(vertices, faces, target, config)
        try:
            vertices, faces = isotropic_remesh(vertices, faces, target, config)
        except Exception as error:  # pymeshlab without the remesher, or a mesh it rejects
            LOGGER.warning("%s: isotropic remesh unavailable (%s); using the plain collapser",
                           self.name, error)
        return simplify_3d(vertices, faces, target, config)

    def _bake(self, mesh_data: MeshPayload, low_mesh: MeshPayload,
              config: DecimateConfig) -> dict:
        """Project every input image from the dense surface onto the new atlas.

        The index maps are not re-used: their texels were authored against the old atlas, so
        they are re-derived -- ``normal_map`` by composing the high-poly normals in the low-poly
        TBN frames, the colour channels by sampling the high-poly UVs at the cage hit.
        """
        if not mesh_data.textures:
            LOGGER.info("%s: the input carries no images; only the geometry was rebuilt",
                        self.name)
            return {}
        if mesh_data.uvs is None:
            raise RuntimeError("%s: the input has images but no UVs to sample them from"
                               % self.name)
        high_normals, high_tangents = ensure_tbn(
            mesh_data.vertices, mesh_data.faces, mesh_data.uvs, config,
            normals=mesh_data.normals, tangents=mesh_data.tangents)
        return bake_pbr_maps_3d_transfer(
            mesh_data.vertices, mesh_data.faces, mesh_data.uvs, high_normals, high_tangents,
            mesh_data.textures,
            low_mesh.vertices, low_mesh.faces, low_mesh.uvs, low_mesh.normals, low_mesh.tangents,
            resolution=int(config.bake_resolution), cage_ratio=float(config.cage_ratio),
        )


__all__ = ["RetopoBakeStrategy", "isotropic_remesh", "repack_uv_xatlas", "simplify_3d",
           "simplify_3d_pyfqmr"]
