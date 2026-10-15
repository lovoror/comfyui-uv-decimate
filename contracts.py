"""Shared contracts for the UV-safe decimation pipeline.

Every stage -- conversion, decimation strategy, normal rebuild, texture bake -- speaks
:class:`MeshPayload`: a plain-numpy snapshot of one mesh batch item whose shapes, dtypes
and ranges are fixed, so a strategy never has to re-derive them from ComfyUI's batched
``(B, N, 3)`` tensors.

Conventions
-----------
* Positions are right-handed, Y-up (what ComfyUI's ``SaveGLB`` writes).
* ``uvs`` follow glTF: origin bottom-left, values in ``[0, 1]``.
* ``tangents[:, :3]`` is the unit tangent, orthogonal to ``normals``;
  ``tangents[:, 3]`` is the handedness ``w`` in ``{-1.0, 1.0}``.
* ``textures`` maps each present name in :data:`IMAGE_ATTRS` to linear RGB ``(H, W, 3)``
  float32 in ``[0.0, 1.0]`` (ComfyUI's MESH image attributes carry no alpha).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

#: Image attributes that live in the same UV space as the base colour texture and can
#: therefore be carried over unchanged when only the tessellation changes.
IMAGE_ATTRS: tuple[str, ...] = ("texture", "metallic_roughness", "normal_map", "emissive")

#: Share of the seam vertices that quadric decimation can still be asked to keep before
#: it starts collapsing across chart borders (measured lower bound, see analyze_uv_topology).
#:
#: ``total_vertices - welded_vertices`` counts *every* seam vertex, and roughly 80% of those are
#: collinear two-chart boundary vertices that the 5D metric may still collapse along the seam
#: edge.  Only the remaining ~20% are locked UV chart corners -- places where three or more
#: Share of UV-seam vertices below which keeping the original atlas starves smooth chart interiors.
#: On heavily fragmented atlases (e.g. ``_raw_ship_00016.glb`` with 131,054 seam vertices across
#: 1,467 charts), collapsing below ``0.55 * seam_vertices`` (~72,080 faces) locks >80% of all
#: triangles on UV seams and starves chart interiors down to ~2% of faces.
SEAM_FACE_SHARE = 0.55


def _as_array(value: Any, dtype: Any, ndim: int, name: str) -> np.ndarray:
    """Cast ``value`` to ``dtype`` and assert it is exactly ``ndim``-dimensional."""
    array = np.asarray(value, dtype=dtype)
    if array.ndim != ndim:
        raise ValueError("%s must be %d-dimensional, got shape %s" % (name, ndim, array.shape))
    return array


@dataclass
class MeshPayload:
    """One mesh batch item as contiguous numpy arrays.

    ``vertices`` ``(V, 3)`` float32 positions, ``faces`` ``(F, 3)`` int64 triangle indices.
    Optional per-vertex attributes are ``uvs`` ``(V, 2)`` float32, ``normals`` ``(V, 3)``
    float32 unit vectors and ``tangents`` ``(V, 4)`` float32. ``textures`` holds ``(H, W, 3)``
    float32 RGB images for the names in :data:`IMAGE_ATTRS`; ``extra_attrs`` keeps every
    mesh attribute that is neither geometry nor an image (material, unlit, ...) so a
    round-trip through the pipeline does not silently drop data.
    """

    vertices: np.ndarray
    faces: np.ndarray
    uvs: Optional[np.ndarray] = None
    normals: Optional[np.ndarray] = None
    tangents: Optional[np.ndarray] = None
    textures: dict[str, np.ndarray] = field(default_factory=dict)
    extra_attrs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.vertices = _as_array(self.vertices, np.float32, 2, "vertices")
        self.faces = _as_array(self.faces, np.int64, 2, "faces")
        if self.vertices.shape[1] != 3:
            raise ValueError("vertices must be (V, 3), got %s" % (self.vertices.shape,))
        if self.faces.shape[1] != 3:
            raise ValueError("faces must be (F, 3), got %s" % (self.faces.shape,))
        for name, width in (("uvs", 2), ("normals", 3), ("tangents", 4)):
            value = getattr(self, name)
            if value is None:
                continue
            array = _as_array(value, np.float32, 2, name)
            if array.shape[1] != width:
                raise ValueError("%s must be (V, %d), got %s" % (name, width, array.shape))
            if array.shape[0] != self.vertices.shape[0]:
                raise ValueError("%s must have one row per vertex (%d), got %d"
                                 % (name, self.vertices.shape[0], array.shape[0]))
            setattr(self, name, array)
        self.textures = dict(self.textures or {})
        self.extra_attrs = dict(self.extra_attrs or {})

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def face_count(self) -> int:
        return int(self.faces.shape[0])


@dataclass
class DecimateConfig:
    """Tunables of one decimation run.

    ``strategy`` selects the :class:`~strategies.base.DecimationStrategy`; ``"auto"`` picks one
    from the mesh metrics, ``"tex_qem"`` (alias ``"keep_uv"``) and ``"retopo_bake"`` force one.
    The remaining fields are passed straight to the backend filters.
    """

    target_faces: int = 50000
    strategy: str = "auto"
    preserve_boundary: bool = True
    preserve_normal: bool = True
    planar_quadric: bool = True
    #: MeshLab ``qualitythr``: lower lets the collapser move further in flat regions.
    quality_thr: float = 0.5
    #: Weight of the texture-coordinate quadrics (``extratcoordw``) in the 5D metric.
    #: Because VCG already sums one 5D quadric per distinct UV wedge at seam vertices,
    #: a light weight (0.05) protects UV boundaries without starving 3D surface curvature.
    texture_weight: float = 0.05
    #: Weight of the rank-3 planar quadric that keeps flat regions flat.
    planar_weight: float = 0.001
    weld_normals: bool = True
    crease_angle: Optional[float] = None
    rebake_normal_map: bool = True
    bake_resolution: int = 2048
    cage_ratio: float = 0.012
    #: Iterations of the isotropic explicit remesher (0 falls back to the collapser default).
    remesh_iterations: int = 0
    #: Run ``meshing_isotropic_explicit_remeshing`` before the final collapse (retopo_bake).
    isotropic_remesh: bool = False
    #: Run the weld/dedup pass (:func:`~uv_pipeline.clean`) before collapsing. It keeps the
    #: atlas; it exists so the 3D vertices a UV split duplicated are welded before the quadrics
    #: are built, otherwise every chart border is measured as an open boundary crack.
    clean: bool = True

    STRATEGIES = ("auto", "tex_qem", "keep_uv", "retopo_bake")

    def __post_init__(self) -> None:
        if self.strategy not in self.STRATEGIES:
            raise ValueError("strategy must be one of %s, got %r" % (self.STRATEGIES, self.strategy))
        self.target_faces = int(self.target_faces)
        self.bake_resolution = int(self.bake_resolution)
        self.remesh_iterations = int(self.remesh_iterations)
        self.isotropic_remesh = bool(self.isotropic_remesh)
        self.quality_thr = float(self.quality_thr)
        self.texture_weight = float(self.texture_weight)
        self.planar_weight = float(self.planar_weight)
        self.clean = bool(self.clean)
        if self.crease_angle is not None:
            self.crease_angle = float(self.crease_angle)

    # -- spelling aliases -------------------------------------------------------------------
    # The node widgets and the pipeline's keyword arguments use the names on the left, the
    # dataclass fields are the names on the right. Both spellings address the same slot, so a
    # caller never has to know which one the contract picked.
    @property
    def uv_weight(self) -> float:
        """``texture_weight`` under the node's spelling."""
        return self.texture_weight

    @uv_weight.setter
    def uv_weight(self, value) -> None:
        self.texture_weight = float(value)

    @property
    def rebake_normals(self) -> bool:
        """``rebake_normal_map`` under the pipeline's spelling."""
        return bool(self.rebake_normal_map)

    @rebake_normals.setter
    def rebake_normals(self, value) -> None:
        self.rebake_normal_map = bool(value)

    @property
    def crease_angle_deg(self) -> Optional[float]:
        """``crease_angle`` under the node's spelling; ``None`` means "no crease split"."""
        return self.crease_angle

    @crease_angle_deg.setter
    def crease_angle_deg(self, value) -> None:
        self.crease_angle = None if value is None else float(value)


@dataclass
class UVTopologyMetrics:
    """How much of the input's vertex count is pure UV-seam duplication."""

    total_vertices: int
    welded_vertices: int
    total_faces: int
    uv_seam_vertices: int
    min_keep_uv_faces: int


def analyze_uv_topology(vertices: np.ndarray, faces: np.ndarray,
                        uvs: Optional[np.ndarray] = None) -> UVTopologyMetrics:
    """Measure seam duplication of a mesh.

    ``welded_vertices`` counts distinct positions at 6 decimals; everything above that is a
    vertex duplicated by a UV seam (or a hard edge). ``min_keep_uv_faces`` is the face budget
    below which quadric decimation necessarily collapses across chart borders.

    ``uvs`` is accepted for call-site symmetry and future seam-localisation work; the counts
    above are derived from positions and faces only, exactly as specified.
    """
    positions = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    triangles = np.asarray(faces).reshape(-1, 3)
    total_vertices = int(positions.shape[0])
    total_faces = int(triangles.shape[0])
    if total_vertices:
        welded = int(np.unique(np.round(positions, 6), axis=0).shape[0])
    else:
        welded = 0
    uv_seam_vertices = max(0, total_vertices - welded)
    return UVTopologyMetrics(
        total_vertices=total_vertices,
        welded_vertices=welded,
        total_faces=total_faces,
        uv_seam_vertices=uv_seam_vertices,
        min_keep_uv_faces=int(round(uv_seam_vertices * SEAM_FACE_SHARE)),
    )


class TopologyStrategy(ABC):
    """One way of reducing a mesh while keeping its texture coordinates usable."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier, e.g. ``"keep_uv"``."""

    @abstractmethod
    def execute(self, payload: MeshPayload, cfg: DecimateConfig) -> MeshPayload:
        """Return a new payload with the reduced tessellation."""


__all__ = [
    "IMAGE_ATTRS",
    "SEAM_FACE_SHARE",
    "MeshPayload",
    "DecimateConfig",
    "UVTopologyMetrics",
    "analyze_uv_topology",
    "TopologyStrategy",
]
