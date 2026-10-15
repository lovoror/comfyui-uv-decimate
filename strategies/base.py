"""Shared machinery for the UV-decimation strategies.

Every strategy in this package reduces a mesh's tessellation while keeping the result usable
as a textured asset:

* :class:`~strategies.tex_qem.TextureQEMStrategy` keeps the input atlas and lets 5D (position +
  texture coordinate) quadric error metrics choose the collapses.
* :class:`~strategies.retopo_bake.RetopoBakeStrategy` drops the input atlas, simplifies the
  welded 3D surface, builds a fresh ``xatlas`` atlas and transfers every PBR channel onto it.

What the two have in common lives here: the abstract contract, the geometry measurements that
drive routing and remeshing, and the pymeshlab bridge.

The measurements follow ``huxingyi/autoremesher``: the target edge length comes from
``AutoRemesher::initializeVoxelSize`` (``autoremesher.cpp:128-135``) and the two-stage
pre-decimation from ``AutoRemesher::decimateIfTooDense`` (``autoremesher.cpp:146-261``).
"""

from __future__ import annotations

import dataclasses
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
import pymeshlab as pml

try:  # normal case: imported as part of the comfyui-uv-decimate package
    from .. import mesh_normals, uv_pipeline
    from ..contracts import DecimateConfig, MeshPayload, TopologyStrategy
    from ..contracts import analyze_uv_topology as measure_uv_topology
    from ..mesh_convert import split_by_uv
except ImportError:  # loaded as a plain directory rather than a package
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import mesh_normals
    import uv_pipeline
    from contracts import DecimateConfig, MeshPayload, TopologyStrategy
    from contracts import analyze_uv_topology as measure_uv_topology
    from mesh_convert import split_by_uv

LOGGER = logging.getLogger(__name__)

#: Triangles whose 3D area is at or below this are slivers: they carry no shading information
#: and their degenerate frames poison both the vertex normals and the normal-map bake.
DEGENERATE_AREA_EPSILON = 1e-12
#: Positions rounded to this many decimals count as a single welded 3D vertex.
WELD_DECIMALS = 6
#: Share of the UV-seam vertices that act as topological anchors during 5D QEM collapse.
#: At 0.38 (paired with ``AutoRouterStrategy.SEAM_HEADROOM = 1.45``, i.e. ``0.38 * 1.45 = 0.551``),
#: a 698k mesh with 131,054 seam vertices routes to ``retopo_bake`` below ~72,200 faces (so the
#: 50k preset avoids locking 80.7% of its triangles on UV seams and starving smooth chart interiors),
#: while the 100k preset (and cleanly unwrapped meshes like a 1,058-face UV sphere at target >= 70)
#: take the fast ``tex_qem`` path.
SEAM_LOCK_CORNER_SHARE = 0.38
#: ``sqrt(3) / 4``: the area of an equilateral triangle with edge length 1 (``autoremesher.cpp``).
EQUILATERAL_TRIANGLE_AREA = 0.86602540378 * 0.5
#: ``preservenormal`` for both collapsers.  Keeping it enabled prevents QEM collapses on thin
#: double-sided sheets (such as sails or cloth) from inverting face normals across the sheet.
PRESERVE_NORMAL = True
#: Cosine of the maximum angle (~69.5 deg) allowed between a face normal and a vertex's dominant
#: normal before that corner is split into an independent vertex copy, so opposing sides of a
#: thin sheet never cancel out into a backwards vertex normal.
OPPOSING_CORNER_COS = 0.35
#: ``planarquadric``: adds a rank-3 planar quadric on top of the rank-4 point quadric, which
#: keeps flat regions flat and lets thin details survive as flat sheets instead of being
#: rounded off by an over-eager collapser.
PLANAR_QUADRIC = True


def _split_opposing_corners(vertices: np.ndarray, faces: np.ndarray,
                            uvs: Optional[np.ndarray] = None,
                            cos_split: float = OPPOSING_CORNER_COS):
    """Split corners whose face normal opposes the vertex's dominant normal (``dot < cos_split``).

    On thin two-sided shells (sails, cloth, collar rims), QEM collapses can leave a single vertex
    shared by front-facing (``+n``) and back-facing (``-n``) triangles.  Because a single glTF
    vertex carries only one normal, leaving those corners merged forces at least one side's
    normal to point backwards.  Grouping deviating corners by ``(vertex_id, normal_octant)``
    separates the opposing sides while keeping adjacent faces on the same side welded.
    """
    positions = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    triangles = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    texcoords = None if uvs is None else np.asarray(uvs, dtype=np.float32).reshape(-1, 2)
    if positions.shape[0] == 0 or triangles.shape[0] == 0:
        return positions, triangles, texcoords

    corners = positions.astype(np.float64)[triangles]
    fn_raw = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    fn_len = np.linalg.norm(fn_raw, axis=1, keepdims=True)
    fn_unit = fn_raw / np.maximum(fn_len, 1e-12)

    for _ in range(2):
        unwelded = np.zeros((positions.shape[0], 3), dtype=np.float64)
        for corner in range(3):
            np.add.at(unwelded, triangles[:, corner], fn_raw)
        unw_len = np.linalg.norm(unwelded, axis=1, keepdims=True)
        unw_unit = unwelded / np.maximum(unw_len, 1e-12)
        dots = np.sum(unw_unit[triangles] * fn_unit[:, None, :], axis=-1)
        bad = dots < float(cos_split)
        if not np.any(bad):
            break
        bad_f, bad_c = np.nonzero(bad)
        bad_v = triangles[bad_f, bad_c]
        fn_bad = fn_unit[bad_f]
        octant = ((fn_bad[:, 0] >= 0).astype(np.int64) * 4
                  + (fn_bad[:, 1] >= 0).astype(np.int64) * 2
                  + (fn_bad[:, 2] >= 0).astype(np.int64))
        key = bad_v.astype(np.int64) * 8 + octant
        unique_keys, inverse = np.unique(key, return_inverse=True)
        rep_v = unique_keys // 8
        new_start = positions.shape[0]
        positions = np.concatenate([positions, positions[rep_v]], axis=0)
        if texcoords is not None:
            texcoords = np.concatenate([texcoords, texcoords[rep_v]], axis=0)
        triangles = triangles.copy()
        triangles[bad_f, bad_c] = new_start + inverse
    return positions, triangles, texcoords


def degenerate_face_mask(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Boolean mask of the faces that carry no usable geometry.

    A face is degenerate when two of its corners are the same vertex index, or when its 3D area
    is at or below :data:`DEGENERATE_AREA_EPSILON`.  Both kinds have an undefined tangent frame,
    so they yield NaN normals and a bright spike in the baked normal map.
    """
    positions = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    triangles = np.asarray(faces).reshape(-1, 3)
    if triangles.shape[0] == 0 or positions.shape[0] == 0:
        return np.zeros((triangles.shape[0],), dtype=bool)
    duplicated = ((triangles[:, 0] == triangles[:, 1])
                  | (triangles[:, 1] == triangles[:, 2])
                  | (triangles[:, 2] == triangles[:, 0]))
    corners = positions[triangles]
    area = 0.5 * np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)
    return duplicated | ~(area > DEGENERATE_AREA_EPSILON)


def drop_degenerate_faces(vertices: np.ndarray, faces: np.ndarray,
                          uvs: Optional[np.ndarray] = None):
    """Remove degenerate faces and split opposing thin-shell corners, returning ``(v, f, uvs)``."""
    triangles = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    keep = ~degenerate_face_mask(vertices, triangles)
    clean_faces = triangles if bool(keep.all()) else triangles[keep]
    clean_uvs = None if uvs is None else np.asarray(uvs, dtype=np.float32)
    return _split_opposing_corners(vertices, clean_faces, clean_uvs)


def split_to_welded_3d(vertices: np.ndarray, faces: np.ndarray, decimals: int = WELD_DECIMALS):
    """Merge the seam duplicates of an atlas-split mesh into one vertex per position.

    A chart border turns one surface point into several vertices that differ only in their
    texture coordinate.  For a UV-free simplification those duplicates are noise: the seam is a
    parameterization artefact, not a crease, and leaving them in leaves cracks behind once the
    two sides are collapsed independently.  Positions are welded at ``decimals`` decimals; the
    first original position of each group is kept, so the surface does not lose precision.
    Faces that collapse onto a repeated index while welding are dropped.

    Returns ``(vertices float32, faces int64)``.
    """
    positions = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    triangles = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if positions.shape[0] == 0 or triangles.shape[0] == 0:
        return positions.astype(np.float32), triangles
    keys = np.round(positions, int(decimals))
    _, first_index, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    welded = positions[np.asarray(first_index).reshape(-1)]
    remapped = np.asarray(inverse).reshape(-1)[triangles]
    return welded.astype(np.float32), remapped[~degenerate_face_mask(welded, remapped)]


def to_pymeshlab(vertices: np.ndarray, faces: np.ndarray,
                 uvs: Optional[np.ndarray] = None) -> pml.MeshSet:
    """Wrap arrays in a ``MeshSet``; ``uvs`` become both per-vertex and per-corner coordinates.

    Storing the atlas per *corner* as well is what lets the 5D collapser evaluate the
    texture-coordinate quadrics -- without the wedge table it silently degrades to 3D QEM while
    still looking like it succeeded.
    """
    mesh_set = uv_pipeline.build_mesh_set(vertices, faces, uvs)
    if uvs is not None and uv_pipeline.wedge_coordinates(mesh_set.current_mesh()) is None:
        LOGGER.warning("texture coordinates could not be stored per corner; the collapser "
                       "will only see 3D quadrics")
    return mesh_set


def _vertex_texture_coordinates(mesh: pml.Mesh, vertex_count: int) -> Optional[np.ndarray]:
    """Per-vertex texture coordinates of ``mesh``, or None when it carries none."""
    if hasattr(mesh, "has_vertex_tex_coord") and not mesh.has_vertex_tex_coord():
        return None
    try:
        coordinates = np.asarray(mesh.vertex_tex_coord_matrix(), dtype=np.float64)
    except Exception:  # no texture coordinates at all
        return None
    if coordinates.size == 0 or coordinates.shape[0] != int(vertex_count):
        return None
    return coordinates.reshape(-1, 2).astype(np.float32)


def from_pymeshlab(mesh_set: pml.MeshSet):
    """Read ``(vertices float32, faces int64, uvs)`` back out of a ``MeshSet``.

    A mesh that still carries a wedge table is rebuilt with :func:`mesh_convert.split_by_uv`,
    which splits every vertex per distinct texture coordinate (and clips the atlas into
    ``[0, 1]``); filters that cannot carry the table fall back to the mesh's own per-vertex
    coordinates, and a UV-free mesh yields ``uvs=None``.
    """
    mesh = mesh_set.current_mesh()
    vertices = np.asarray(mesh.vertex_matrix(), dtype=np.float32).reshape(-1, 3)
    faces = np.asarray(mesh.face_matrix(), dtype=np.int64).reshape(-1, 3)
    wedges = uv_pipeline.wedge_coordinates(mesh)
    if wedges is not None and faces.shape[0]:
        split_v, split_f, split_uv = split_by_uv(vertices, faces, wedges)
        return (np.asarray(split_v, dtype=np.float32).reshape(-1, 3),
                np.asarray(split_f, dtype=np.int64).reshape(-1, 3),
                np.asarray(split_uv, dtype=np.float32).reshape(-1, 2))
    return vertices, faces, _vertex_texture_coordinates(mesh, vertices.shape[0])


def ensure_tbn(vertices: np.ndarray, faces: np.ndarray, uvs: Optional[np.ndarray],
               config: DecimateConfig, normals: Optional[np.ndarray] = None,
               tangents: Optional[np.ndarray] = None):
    """Fill in the smooth normals and tangents a bake or a payload needs.

    Normals are rebuilt from 3D position (``weld=True``), so the vertices a UV split duplicated
    along a seam still share one normal -- otherwise the seam becomes a visible shading seam in
    every channel that is baked from these frames.  ``config.crease_angle`` keeps genuinely hard
    edges hard.
    """
    if normals is None:
        normals = mesh_normals.vertex_normals(vertices, faces, weld=bool(config.weld_normals),
                                             crease_angle=config.crease_angle)
    if tangents is None and uvs is not None:
        tangents = mesh_normals.vertex_tangents(vertices, faces, uvs, normals)
    return normals, tangents


def rebuild_payload(mesh_data: MeshPayload, **changes: Any) -> MeshPayload:
    """Copy a payload with ``changes`` applied, re-validating the result.

    ``dataclasses.replace`` re-runs :meth:`MeshPayload.__post_init__`, so a vertex/uv/normal
    count mismatch is rejected here instead of surfacing as a crash inside the baker.  Every
    channel that is not named is carried over untouched, including the ``IMAGE_ATTRS`` images
    and ``extra_attrs`` (material, unlit, ...).
    """
    return dataclasses.replace(mesh_data, **changes)


def compute_mesh_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Total surface area: the sum of the triangle areas.

    The same quantity as MeshLab's ``SurfaceArea``, computed directly so it is available before
    a mesh-set round-trip and on meshes the filters would reject.
    """
    positions = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    triangles = np.asarray(faces).reshape(-1, 3)
    if positions.shape[0] == 0 or triangles.shape[0] == 0:
        return 0.0
    corners = positions[triangles]
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    return float(0.5 * np.linalg.norm(cross, axis=1).sum())


def compute_equilateral_target_edge_length(vertices: np.ndarray, faces: np.ndarray,
                                           target_faces: int) -> float:
    r"""Edge length of the equilateral triangles that fill the surface with ``target_faces``.

    ``AutoRemesher::initializeVoxelSize`` (``autoremesher.cpp:128-135``) gives every target
    triangle the same share of the surface area,

    .. math::  A_{\text{tri}} = \frac{A_{\text{total}}}{\max(\text{target\_faces}, 1)}

    and then solves the equilateral area formula :math:`A = \frac{\sqrt{3}}{4} L^2` for the edge
    length.  Returns ``0.0`` for a mesh without area, which callers treat as "cannot remesh".
    """
    area = compute_mesh_area(vertices, faces)
    if not (area > 0.0):
        return 0.0
    triangle_area = area / float(max(int(target_faces), 1))
    return float(np.sqrt(triangle_area / EQUILATERAL_TRIANGLE_AREA))


def analyze_uv_topology(vertices: np.ndarray, faces: np.ndarray,
                        uvs: Optional[np.ndarray] = None) -> dict:
    """Measure how much of the vertex count is pure UV-seam duplication.

    Returns ``welded_vertices`` (distinct positions at 6 decimals), ``uv_vertices`` (rows of
    ``uvs``, or the vertex count when there is no atlas), ``uv_seam_vertices`` (the vertices that
    exist only because a chart border needed its own texture coordinate), ``num_faces`` and
    ``seam_lock_floor``: the subset of those seam vertices the collapser is genuinely *stuck* on.

    ``uv_vertices - welded_vertices`` over-estimates the stall floor because most seam vertices
    are collinear two-chart boundaries -- the 5D metric is allowed to collapse them along the
    seam edge.  Only :data:`SEAM_LOCK_CORNER_SHARE` of them are locked UV chart corners (three or
    more charts meeting, or a seam whose curvature forbids the collapse), so
    ``seam_lock_floor = round(uv_seam_vertices * SEAM_LOCK_CORNER_SHARE)``.  On
    ``_raw_ship_00016.glb`` that is 131,054 * 0.20 = 26,211 corners, matching the empirical stall
    at ~38,000 faces (26,211 * the router's 1.45 headroom).
    """
    positions = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    metrics = measure_uv_topology(positions, faces, uvs)
    welded = int(metrics.welded_vertices)
    uv_vertices = int(metrics.total_vertices) if uvs is None else int(np.asarray(uvs).shape[0])
    uv_seam_vertices = max(0, uv_vertices - welded)
    return {
        "welded_vertices": welded,
        "uv_vertices": uv_vertices,
        "uv_seam_vertices": uv_seam_vertices,
        "seam_lock_floor": int(round(uv_seam_vertices * SEAM_LOCK_CORNER_SHARE)),
        "num_faces": int(np.asarray(faces).reshape(-1, 3).shape[0]),
    }


class DecimationStrategy(TopologyStrategy, ABC):
    """One way of trading triangles for texture fidelity.

    Subclasses implement :meth:`execute`; the geometry measurements below are shared with the
    router, so both reason about a mesh with exactly the same numbers.
    """

    #: Stable identifier used by ``DecimateConfig.strategy`` and reported by the router.
    name: str = "base"

    # Assigned rather than re-implemented, so the router and the strategies cannot drift apart.
    compute_mesh_area = staticmethod(compute_mesh_area)
    compute_equilateral_target_edge_length = staticmethod(compute_equilateral_target_edge_length)
    analyze_uv_topology = staticmethod(analyze_uv_topology)

    @abstractmethod
    def execute(self, mesh_data: MeshPayload, config: DecimateConfig) -> MeshPayload:
        """Return a new payload with the reduced tessellation."""

    @staticmethod
    def require_faces(label: str, face_count: int) -> None:
        """Fail loudly on an empty result instead of handing the node layer a broken mesh."""
        if int(face_count) <= 0:
            raise RuntimeError("%s produced no faces; the input may be degenerate or the "
                               "requested face count is unreachable" % label)


__all__ = [
    "DEGENERATE_AREA_EPSILON",
    "EQUILATERAL_TRIANGLE_AREA",
    "PLANAR_QUADRIC",
    "PRESERVE_NORMAL",
    "SEAM_LOCK_CORNER_SHARE",
    "WELD_DECIMALS",
    "DecimationStrategy",
    "analyze_uv_topology",
    "compute_equilateral_target_edge_length",
    "compute_mesh_area",
    "degenerate_face_mask",
    "drop_degenerate_faces",
    "ensure_tbn",
    "from_pymeshlab",
    "rebuild_payload",
    "split_to_welded_3d",
    "to_pymeshlab",
]
