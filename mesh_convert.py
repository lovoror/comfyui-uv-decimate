"""ComfyUI MESH <-> numpy conversions used by the decimation pipeline.

Only batch item 0 is supported, as in the core Save/Preview 3D nodes. Two layers live here:
the legacy array helpers (``extract_inputs``/``first_item``, ``split_by_uv``, ``pack_mesh``)
and the :class:`~contracts.MeshPayload` bridge (``to_payload``/``from_payload``), which also
carries normals, tangents and the ``IMAGE_ATTRS`` images.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import torch

try:
    from .contracts import IMAGE_ATTRS, MeshPayload
except ImportError:  # loaded as a plain directory rather than a package
    from contracts import IMAGE_ATTRS, MeshPayload

#: Material attributes; a topology change does not invalidate them.
SCALAR_ATTRS = ("material", "unlit", "occlusion_in_mr")
#: Tessellation attributes: stale after a topology change, so never copied to ``extra_attrs``.
GEOMETRY_ATTRS = ("vertices", "faces", "uvs", "normals", "tangents", "vertex_colors", "colors",
                  "vertex_counts", "face_counts")


def to_numpy(x) -> Optional[np.ndarray]:
    """Convert a ``torch.Tensor`` or array-like to numpy; ``None`` stays ``None``."""
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def image_to_numpy_float(x) -> Optional[np.ndarray]:
    """Normalise a ComfyUI image to a contiguous ``(H, W, 3)`` float32 array in ``[0, 1]``.

    Accepts ``(1, H, W, C)``, ``(H, W, C)`` or ``(C, H, W)``, float ``[0, 1]`` or integer
    ``[0, 255]``; channels are truncated to RGB, or repeated when there are fewer than three.
    """
    array = to_numpy(x)
    if array is None or array.size == 0:
        return None
    array = np.asarray(array)
    if array.ndim == 4:
        array = array[0]
    if array.ndim == 2:
        array = array[:, :, None]
    if array.ndim != 3:
        return None
    first, last = array.shape[0], array.shape[-1]
    if last not in (1, 3, 4) and first in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] < 3:
        array = np.repeat(array[:, :, :1], 3, axis=2)
    array = array[:, :, :3]
    if np.issubdtype(array.dtype, np.integer) or array.dtype == np.bool_:
        array = array.astype(np.float32) / 255.0
    else:
        array = np.clip(array.astype(np.float32), 0.0, 1.0)
    return np.ascontiguousarray(array, dtype=np.float32)


def numpy_float_to_image_tensor(arr: np.ndarray, like_ref: Any = None) -> Any:
    """Turn a ``(H, W, 3)`` float image in ``[0, 1]`` into a ComfyUI image tensor.

    Returns ``(1, H, W, 3)`` (or ``(H, W, 3)`` when ``like_ref`` is an unbatched 3-D tensor),
    which is what ``SaveGLB`` expects. ``H``/``W`` come from ``arr`` -- a rebake changes the
    resolution -- while ``like_ref`` only supplies the torch dtype, device and batch rank.
    """
    image = np.asarray(arr, dtype=np.float32)
    if image.ndim == 4:
        image = image[0]
    if image.ndim != 3:
        return None
    image = np.ascontiguousarray(image[:, :, :3], dtype=np.float32)

    dtype, device, batched = torch.float32, torch.device("cpu"), True
    if torch.is_tensor(like_ref) and like_ref.is_floating_point():
        dtype, device = like_ref.dtype, like_ref.device
        batched = like_ref.ndim != 3
    tensor = torch.from_numpy(image).to(dtype=dtype, device=device)
    return tensor.unsqueeze(0) if batched else tensor


def extract_inputs(mesh) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Item 0 as float64 ``(V, 3)`` vertices, int32 ``(F, 3)`` faces, float64 ``(V, 2)`` uvs."""
    vertices, faces = mesh.vertices, mesh.faces
    uvs = getattr(mesh, "uvs", None)
    vertex_counts = getattr(mesh, "vertex_counts", None)
    face_counts = getattr(mesh, "face_counts", None)

    if isinstance(vertices, (list, tuple)):
        v0, f0 = vertices[0], faces[0]
        uv0 = None if uvs is None else uvs[0]
    else:
        n_vertices = int(vertex_counts[0]) if vertex_counts is not None else vertices.shape[1]
        n_faces = int(face_counts[0]) if face_counts is not None else faces.shape[1]
        v0, f0 = vertices[0, :n_vertices], faces[0, :n_faces]
        uv0 = None if uvs is None else uvs[0, :n_vertices]

    v = to_numpy(v0).astype(np.float64).reshape(-1, 3)
    f = to_numpy(f0).astype(np.int32).reshape(-1, 3)
    uv = None if uv0 is None else to_numpy(uv0).astype(np.float64).reshape(-1, 2)
    return v, f, uv


#: Backwards-compatible alias: older callers import ``first_item``.
first_item = extract_inputs


def _split(v, f, wedge_uv, decimals):
    """``(sv, sf, suv, source_index)``: one vertex per distinct ``(position, uv)`` pair.

    pymeshlab keeps one texture coordinate per face corner; glTF needs one per vertex. Keyed on
    ``(corner index, rounded u, rounded v)`` and resolved with ``np.unique`` (``O(F log F)`` in C
    rather than a Python dict loop). ``source_index`` maps each new vertex back to its original
    index, for callers re-indexing attributes computed on the pre-split mesh.
    """
    flat_v = np.asarray(f, dtype=np.int64).reshape(-1)
    wuv = np.asarray(wedge_uv, dtype=np.float64).reshape(-1, 2)
    if wuv.shape[0] != flat_v.shape[0]:
        raise ValueError("wedge_uv has %d rows, expected one per face corner (%d)"
                         % (wuv.shape[0], flat_v.shape[0]))
    keys = np.stack([flat_v.astype(np.float64), np.round(wuv[:, 0], decimals),
                     np.round(wuv[:, 1], decimals)], axis=1)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    source_index = unique[:, 0].astype(np.int64)
    return (np.asarray(v, dtype=np.float32)[source_index],
            inverse.reshape(-1, 3).astype(np.int64),
            np.clip(unique[:, 1:3], 0.0, 1.0).astype(np.float32),
            source_index)


def split_by_uv(v, f, wedge_uv, decimals: int = 6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split by UV seam: float32 ``(V_new, 3)`` positions, int64 ``(F, 3)`` faces,
    float32 ``(V_new, 2)`` uvs clipped to ``[0, 1]``."""
    sv, sf, suv, _ = _split(v, f, wedge_uv, decimals)
    return sv, sf, suv


def split_by_uv_with_source(v, f, wedge_uv, decimals: int = 6):
    """Same split as :func:`split_by_uv`, plus the ``source_index`` remap of every new vertex."""
    return _split(v, f, wedge_uv, decimals)


def pack_mesh(mesh, *args, **kwargs):
    """Wrap arrays back into a MESH, carrying the source textures and material.

    Two call conventions are accepted, told apart by the first argument:

    * ``pack_mesh(mesh, sv, sf, suv, vn=None, vt=None, texture_overrides=None)`` -- payload form.
    * ``pack_mesh(vertices, faces, uvs, source_mesh, normals=None, tangents=None,
      texture_overrides=None)`` -- original array form, still used by ``nodes.py``.

    ``texture_overrides`` maps a name in :data:`IMAGE_ATTRS` (e.g. a rebaked ``"normal_map"``)
    to a numpy ``(H, W, 3)`` float image in ``[0, 1]``, so a rebake replaces the source map.
    """
    if isinstance(mesh, np.ndarray) or torch.is_tensor(mesh):
        vertices, faces = mesh, args[0] if args else kwargs.pop("faces", None)
        rest = list(args[1:])
        uvs = rest.pop(0) if rest else kwargs.pop("uvs", None)
        source_mesh = rest.pop(0) if rest else kwargs.pop("source_mesh", None)
        normals = rest.pop(0) if rest else kwargs.pop("normals", None)
        tangents = rest.pop(0) if rest else kwargs.pop("tangents", None)
        return _finish(source_mesh, vertices, faces, uvs, normals, tangents,
                       kwargs.pop("texture_overrides", None))
    vertices = args[0] if args else None
    faces = args[1] if len(args) > 1 else None
    rest = list(args[2:])
    uvs = rest.pop(0) if rest else None
    normals = rest.pop(0) if rest else None
    tangents = rest.pop(0) if rest else None
    return _finish(mesh, vertices, faces, uvs, normals, tangents,
                   kwargs.pop("texture_overrides", None))


def _tensor(array, dtype):
    return torch.from_numpy(np.ascontiguousarray(np.asarray(array, dtype=dtype)))


def _finish(source_mesh, vertices, faces, uvs, normals, tangents, texture_overrides=None):
    kwargs = {}
    for attr in SCALAR_ATTRS + IMAGE_ATTRS:
        value = getattr(source_mesh, attr, None)
        if value is not None:
            kwargs[attr] = value
    for attr, image in (texture_overrides or {}).items():
        if attr in IMAGE_ATTRS and image is not None:
            kwargs[attr] = numpy_float_to_image_tensor(image, like_ref=kwargs.get(attr))

    v = _tensor(vertices, np.float32).unsqueeze(0)
    f = _tensor(faces, np.int64).unsqueeze(0)
    u = None if uvs is None else _tensor(uvs, np.float32).unsqueeze(0)
    n = None if normals is None else [_tensor(normals, np.float32)]
    t = None if tangents is None else [_tensor(tangents, np.float32)]
    normals_batched = None if n is None else n[0].unsqueeze(0)
    tangents_batched = None if t is None else t[0].unsqueeze(0)

    try:  # the constructor the core Save 3D nodes use
        from comfy_extras.nodes_save_3d import pack_variable_mesh_batch
        packed = pack_variable_mesh_batch([v[0]], [f[0]], uvs=None if u is None else [u[0]],
                                          normals=n, tangents=t, **kwargs)
        for attr in IMAGE_ATTRS + SCALAR_ATTRS:  # dropped by pack_variable_mesh_batch
            if attr in kwargs:
                try:
                    setattr(packed, attr, kwargs[attr])
                except Exception:
                    pass
        return packed
    except Exception:
        pass

    try:
        try:
            from comfy_api.latest import Types
        except Exception:  # pragma: no cover - older ComfyUI
            from comfy_api.v0_0_2 import Types
        return Types.MESH(vertices=v, faces=f, uvs=u, normals=normals_batched,
                          tangents=tangents_batched, **kwargs)
    except Exception:  # standalone: a namespace carrying the same attributes
        return SimpleNamespace(vertices=v, faces=f, uvs=u, normals=normals_batched,
                               tangents=tangents_batched, **kwargs)


def _per_vertex_attribute(mesh, name, n_vertices) -> Optional[np.ndarray]:
    """Read a per-vertex ``(V, D)`` attribute of item 0, or ``None`` when absent/short."""
    value = getattr(mesh, name, None)
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    array = None if value is None else to_numpy(value)
    if array is None:
        return None
    array = np.asarray(array)
    if array.ndim == 3:
        array = array[0]
    if array.ndim != 2 or array.shape[0] < n_vertices:
        return None
    return array[:n_vertices]


def _extra_attrs(mesh) -> dict:
    """Every non-geometry, non-image attribute of the source mesh (material, unlit, ...)."""
    extra = {}
    for name in dir(mesh):
        if name.startswith("_") or name in GEOMETRY_ATTRS or name in IMAGE_ATTRS:
            continue
        try:
            value = getattr(mesh, name)
        except Exception:
            continue
        if callable(value):
            continue
        extra[name] = value
    return extra


def to_payload(mesh) -> MeshPayload:
    """Snapshot item 0 of a ComfyUI mesh as a :class:`MeshPayload`.

    Positions/faces/uvs come from :func:`extract_inputs`; normals, tangents and the
    :data:`IMAGE_ATTRS` images are read when the mesh carries them, and ``extra_attrs`` keeps
    every remaining attribute (material, unlit, occlusion_in_mr, ...).
    """
    vertices, faces, uvs = extract_inputs(mesh)
    n_vertices = vertices.shape[0]
    return MeshPayload(
        vertices=vertices.astype(np.float32),
        faces=faces.astype(np.int64),
        uvs=None if uvs is None else np.clip(uvs, 0.0, 1.0).astype(np.float32),
        normals=_per_vertex_attribute(mesh, "normals", n_vertices),
        tangents=_per_vertex_attribute(mesh, "tangents", n_vertices),
        textures={attr: image for attr, image in
                  ((attr, image_to_numpy_float(getattr(mesh, attr, None))) for attr in IMAGE_ATTRS)
                  if image is not None},
        extra_attrs=_extra_attrs(mesh),
    )


def from_payload(mesh, payload: MeshPayload):
    """Rebuild a mesh object from ``payload``, carrying the source material and textures.

    The payload's ``textures`` are pushed as overrides (so a rebaked map wins) and its
    ``extra_attrs`` are re-attached; the geometry always comes from the payload.
    """
    packed = pack_mesh(mesh, payload.vertices, payload.faces, payload.uvs,
                       payload.normals, payload.tangents,
                       texture_overrides=dict(payload.textures))
    for name, value in payload.extra_attrs.items():
        if name in GEOMETRY_ATTRS or name in IMAGE_ATTRS:
            continue
        try:
            setattr(packed, name, value)
        except Exception:
            pass
    return packed
