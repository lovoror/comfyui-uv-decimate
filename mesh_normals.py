"""Normals and tangents for a decimated mesh.

A topology-changing op invalidates per-vertex normals, and glTF viewers fall back to flat
(per-face) shading when the NORMAL attribute is missing, which reads as a faceted, blocky
surface. These helpers rebuild smooth normals (and tangents for the normal map) from the
new tessellation.

Root cause #2 this module fixes: after the wedge/UV re-split, every vertex on a UV seam is
duplicated, so each copy sees only its own half of the faces. Averaging per vertex index
therefore turns every UV seam into a shading seam. ``vertex_normals(weld=True)`` averages
by 3D position instead (UV splits share the exact same ``(x, y, z)``), so the seam is
invisible again while an explicit ``crease_angle`` can still keep hard edges hard.
"""
from __future__ import annotations

import numpy as np

EPSILON = 1e-12
#: Positions closer than ``scale * WELD_EPSILON`` (scale = bbox diagonal) weld together.
WELD_EPSILON = 1e-6
_FALLBACK_NORMAL = np.array((0.0, 0.0, 1.0), dtype=np.float64)
_UP = np.array((0.0, 0.0, 1.0), dtype=np.float64)
_ALT_UP = np.array((0.0, 1.0, 0.0), dtype=np.float64)


def _normalise(vectors, fallback=None):
    """Unit-length copy of ``vectors``; zero-length rows become ``fallback`` if given."""
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    unit = vectors / np.maximum(lengths, EPSILON)
    if fallback is not None:
        degenerate = lengths[..., 0] <= EPSILON
        if np.any(degenerate):
            unit[degenerate] = fallback
    return unit


def _face_frames(vertices, faces):
    """Raw cross product (magnitude = 2 * area) plus the two edge vectors per triangle."""
    tri = vertices[faces]
    edge1 = tri[:, 1] - tri[:, 0]
    edge2 = tri[:, 2] - tri[:, 0]
    return np.cross(edge1, edge2), edge1, edge2


def _welded_inverse(vertices):
    """Bucket index of every vertex's 3D position; UV-seam copies land in one bucket."""
    vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    if vertices.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    scale = max(float(np.ptp(vertices, axis=0).max()), 1e-8)
    quantised = np.round(vertices / (scale * WELD_EPSILON)).astype(np.int64)
    _, inverse = np.unique(quantised, axis=0, return_inverse=True)
    return np.asarray(inverse).reshape(-1)


def _fallback_tangent(normals):
    """Any unit vector orthogonal to ``normals``: ``cross(up, n)``, ``cross(up2, n)`` if parallel."""
    fallback = np.cross(_UP, normals)
    lengths = np.linalg.norm(fallback, axis=1)
    weak = lengths <= 1e-6
    if np.any(weak):
        fallback[weak] = np.cross(_ALT_UP, normals[weak])
        lengths = np.linalg.norm(fallback, axis=1)
    return fallback / np.maximum(lengths, EPSILON)[:, None]


#: Default maximum angle (degrees) across which 3D-coincident UV-seam vertices share normals
#: when ``crease_angle`` is not explicitly specified; prevents thin two-sided sheets (front/back
#: at ~180 deg) and sharp >60 deg creases from cancelling each other's normals.
DEFAULT_WELD_CREASE_DEG = 60.0


def vertex_normals(v, f, weld=True, crease_angle=None):
    """Area-weighted smooth normals, one per vertex, normalised to unit length.

    ``weld=True`` (default) accumulates face normals across UV-seam duplicates that share the
    same 3D position, gated by ``crease_angle`` (or :data:`DEFAULT_WELD_CREASE_DEG` when
    ``crease_angle`` is ``None``) so smooth UV seams disappear while opposing sides of thin
    two-sided sheets and genuine hard edges never cancel or flip each other's normals.
    """
    vertices = np.asarray(v, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(f, dtype=np.int64).reshape(-1, 3)
    if vertices.shape[0] == 0 or faces.shape[0] == 0:
        return np.zeros((vertices.shape[0], 3), dtype=np.float32)
    face_normals, _, _ = _face_frames(vertices, faces)

    unwelded = np.zeros((vertices.shape[0], 3), dtype=np.float64)
    for corner in range(3):
        np.add.at(unwelded, faces[:, corner], face_normals)
    unwelded_unit = _normalise(unwelded, _FALLBACK_NORMAL)

    if crease_angle is not None:
        limit = float(np.cos(np.deg2rad(float(crease_angle))))
        unit_fn = _normalise(face_normals, _FALLBACK_NORMAL)
        filtered = np.zeros((vertices.shape[0], 3), dtype=np.float64)
        for corner in range(3):
            corners = faces[:, corner]
            cosine = np.sum(unit_fn * unwelded_unit[corners], axis=1)
            keep = cosine >= limit
            if np.any(keep):
                np.add.at(filtered, corners[keep], face_normals[keep])
        has_filtered = np.linalg.norm(filtered, axis=1) > EPSILON
        unwelded[has_filtered] = filtered[has_filtered]
        unwelded_unit = _normalise(unwelded, _FALLBACK_NORMAL)

    if not weld:
        return unwelded_unit.astype(np.float32)

    weld_deg = float(crease_angle) if crease_angle is not None else DEFAULT_WELD_CREASE_DEG
    cos_thr = float(np.cos(np.deg2rad(weld_deg)))
    inverse = _welded_inverse(vertices)
    order = np.argsort(inverse, kind="stable")
    inv_sorted = inverse[order]
    starts = np.flatnonzero(np.r_[True, inv_sorted[1:] != inv_sorted[:-1]])
    counts = np.diff(np.r_[starts, len(inv_sorted)])
    multi = np.flatnonzero(counts > 1)
    max_k = int(counts[multi].max()) if multi.size else 1

    out = unwelded.copy()
    for d in range(1, max_k):
        valid = multi[counts[multi] > d]
        s_idx = starts[valid]
        lengths = counts[valid] - d
        total = int(lengths.sum())
        if total == 0:
            continue
        grp = np.repeat(np.arange(len(lengths)), lengths)
        within = np.arange(total) - np.repeat(np.cumsum(np.r_[0, lengths[:-1]]), lengths)
        pos_a = s_idx[grp] + within
        pos_b = pos_a + d
        va = order[pos_a]
        vb = order[pos_b]
        dots = np.einsum("ij,ij->i", unwelded_unit[va], unwelded_unit[vb])
        keep = dots >= cos_thr
        if np.any(keep):
            va_k, vb_k = va[keep], vb[keep]
            np.add.at(out, va_k, unwelded[vb_k])
            np.add.at(out, vb_k, unwelded[va_k])

    return _normalise(out, _FALLBACK_NORMAL).astype(np.float32)


def vertex_tangents(v, f, uv, vn):
    """Per-vertex tangents ``(V, 4)``: unit ``xyz`` orthogonal to ``vn`` plus glTF ``w``.

    Triangles with a degenerate UV area (``|det| <= 1e-12``) contribute an orthogonal
    fallback tangent instead of an infinite one, so a fully degenerate UV layout still
    yields a usable ``(V, 4)`` block rather than ``None``.
    """
    if uv is None or vn is None:
        return None
    vertices = np.asarray(v, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(f, dtype=np.int64).reshape(-1, 3)
    uvs = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    if uvs.shape[0] != vertices.shape[0]:
        return None
    normals = _normalise(np.asarray(vn, dtype=np.float64).reshape(-1, 3), _FALLBACK_NORMAL)

    _, edge1, edge2 = _face_frames(vertices, faces)
    tri_uv = uvs[faces]
    duv1 = tri_uv[:, 1] - tri_uv[:, 0]
    duv2 = tri_uv[:, 2] - tri_uv[:, 0]
    determinant = duv1[:, 0] * duv2[:, 1] - duv2[:, 0] * duv1[:, 1]
    usable = np.abs(determinant) > EPSILON
    inverse = np.zeros_like(determinant)
    inverse[usable] = 1.0 / determinant[usable]
    face_tangent = (edge1 * duv2[:, 1:2] - edge2 * duv1[:, 1:2]) * inverse[:, None]
    face_bitangent = (edge2 * duv1[:, 0:1] - edge1 * duv2[:, 0:1]) * inverse[:, None]

    tangent = np.zeros((vertices.shape[0], 3), dtype=np.float64)
    bitangent = np.zeros((vertices.shape[0], 3), dtype=np.float64)
    for corner in range(3):
        corners = faces[:, corner]
        if np.any(usable):
            np.add.at(tangent, corners[usable], face_tangent[usable])
            np.add.at(bitangent, corners[usable], face_bitangent[usable])
        if np.any(~usable):
            degenerate = corners[~usable]
            np.add.at(tangent, degenerate, _fallback_tangent(normals[degenerate]))

    projected = tangent - normals * np.sum(normals * tangent, axis=1, keepdims=True)
    lengths = np.linalg.norm(projected, axis=1, keepdims=True)
    fallback = lengths[:, 0] <= EPSILON
    if np.any(fallback):
        projected[fallback] = _fallback_tangent(normals[fallback])
    projected = _normalise(projected)

    handedness = np.where(np.sum(np.cross(normals, projected) * bitangent, axis=1) < 0.0,
                          -1.0, 1.0)
    return np.concatenate([projected, handedness[:, None]], axis=1).astype(np.float32)


def compute(v, f, uv=None, weld_normals=True, crease_angle=None):
    """Return ``(vertex_normals, vertex_tangents)``; tangents are ``None`` without UVs."""
    normals = vertex_normals(v, f, weld=weld_normals, crease_angle=crease_angle)
    tangents = vertex_tangents(v, f, uv, normals) if uv is not None else None
    return normals, tangents
