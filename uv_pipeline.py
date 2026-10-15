"""The UV-safe decimation pipeline: one MESH in, one MESH out.

``run()`` is the Slice 4 entry point used by :mod:`nodes`:

1. the ComfyUI ``MESH`` is snapshotted into a :class:`~contracts.MeshPayload`
   (:func:`unpack_mesh`, i.e. ``mesh_convert.to_payload``),
2. the arguments build a :class:`~contracts.DecimateConfig` (:func:`build_config`),
3. the payload is handed to the router strategy, ``strategies.auto_router.AutoRouterStrategy``,
4. the resulting payload is packed back with :func:`pack_mesh`
   (``mesh_convert.from_payload``), which re-attaches the material, every image attribute
   (including a rebaked normal map) and all remaining mesh attributes.

Why the payload indirection exists: every core ComfyUI 3D op runs through a
``(vertices, faces, colours)`` helper and drops ``uvs``/``normals``/``tangents``, so a textured
GLB comes out untextured. pymeshlab instead keeps one texture coordinate per face corner
("wedge"), which survives welding and the texture-metric quadric collapse, so the
base-colour atlas stays usable.

The array-level API of the previous slice (``run_arrays``, ``build_mesh_set``, ``clean``,
``decimate``, ``wedge_coordinates``) is kept unchanged for external callers; ``run`` forwards
to :func:`run_arrays` when its first argument is not a mesh object.

Deliberately NOT used: ``meshing_close_holes``. It leaves faces whose wedge coordinates are
invalid; quadric decimation then refuses to collapse anything at all (measured: 696,942 faces
in, 696,942 faces out) and the wedge table can no longer be trusted.
"""
from __future__ import annotations

import importlib
import logging
import time
from typing import Any, Optional

import numpy as np
import pymeshlab as pml

try:
    from . import mesh_normals
    from .contracts import DecimateConfig, MeshPayload
    from .mesh_convert import from_payload, to_payload
except ImportError:  # loaded as a plain directory rather than a package
    import mesh_normals
    from contracts import DecimateConfig, MeshPayload
    from mesh_convert import from_payload, to_payload

LOGGER = logging.getLogger("comfyui_uv_decimate")

#: Repairs that keep the per-corner texture coordinates intact.
#:
#: ``meshing_repair_non_manifold_edges`` / ``_vertices`` are deliberately absent: on a
#: Pixal3D-style hull they delete every triangle at a T-junction, which punched 992 open
#: boundary holes into the rigging of ``_raw_ship_00016.glb`` (measured ``3d_bdy`` 3259 -> 4251)
#: while the collapser only needs duplicate vertices welded and duplicate faces dropped.
CLEAN_STEPS = (
    ("meshing_remove_duplicate_vertices", {}),
    ("meshing_remove_duplicate_faces", {}),
    ("meshing_remove_unreferenced_vertices", {}),
)

#: Iterations of the isotropic explicit remesher when ``isotropic_remesh`` is requested.
ISOTROPIC_REMESH_ITERATIONS = 3

#: ``mesh_convert`` primitives, under the names the pipeline contract uses.
unpack_mesh = to_payload
pack_mesh = from_payload

#: The normal/tangent rebuilders, under the names the pipeline contract uses.
compute_smooth_vertex_normals = mesh_normals.vertex_normals
compute_vertex_tangents = mesh_normals.vertex_tangents

#: Router strategies are looked up in this order (first hit wins).
ROUTER_CANDIDATES = (("strategies.auto_router", "AutoRouterStrategy"),
                     ("strategies", "AutoRouterStrategy"))

_ROUTER_CACHE: list = []
_IMPORT_ERRORS: list = []


def _append(log, message: str) -> None:
    """Append to a mutable report list; a ``None``/tuple ``log`` is tolerated."""
    try:
        log.append(message)
    except AttributeError:
        pass
    LOGGER.debug("%s", message)


def _has_wedges(mesh) -> bool:
    try:
        return bool(mesh.has_wedge_tex_coord())
    except Exception:
        return False


# --------------------------------------------------------------------------------------
# MeshSet level helpers (unchanged public API)
# --------------------------------------------------------------------------------------
def build_mesh_set(vertices, faces, uvs):
    """Create a MeshSet carrying both per-vertex and per-corner texture coordinates.

    The per-corner ("wedge") table is built by
    ``compute_texcoord_transfer_vertex_to_wedge``, never by a ``w_tex_coords_matrix``
    constructor argument: the constructor writes the wedge coordinates ``WT(f, i).P()`` but
    leaves each wedge's texture *index* ``WT(f, i).N()`` at ``-1``, and every texture-aware
    filter then refuses the mesh with ``Mesh has some inconsistent tex coordinates (some faces
    without texture)``. The transfer filter fills in coordinates and index together.
    """
    kwargs = {"vertex_matrix": np.asarray(vertices, dtype=np.float64),
              "face_matrix": np.asarray(faces, dtype=np.int32)}
    if uvs is not None:
        kwargs["v_tex_coords_matrix"] = np.asarray(uvs, dtype=np.float64)
    mesh_set = pml.MeshSet()
    mesh_set.add_mesh(pml.Mesh(**kwargs), "input")
    if uvs is not None:
        mesh_set.apply_filter("compute_texcoord_transfer_vertex_to_wedge")
    return mesh_set


def clean(mesh_set, log=()):
    """Weld duplicates and repair non-manifold edges/vertices, keeping the wedge table."""
    for name, params in CLEAN_STEPS:
        start = time.time()
        mesh_set.apply_filter(name, **params)
        current = mesh_set.current_mesh()
        _append(log, "%s -> %d vertices, %d faces (%.2fs)"
                % (name, current.vertex_number(), current.face_number(), time.time() - start))
    return mesh_set


def _report(mesh_set, label, started, log):
    """Log one collapse step and hand the mesh set back."""
    out = mesh_set.current_mesh()
    _append(log, "%s -> %d vertices, %d faces (%.2fs)"
            % (label, out.vertex_number(), out.face_number(), time.time() - started))
    return mesh_set


def decimate(mesh_set, target_faces, preserve_normal=True, planar_quadric=True,
             optimal_placement=True, quality_threshold=1.0, log=(), preserve_boundary=False,
             texture_weight=None, planar_weight=None):
    """Quadric edge collapse, carrying the wedge texture coordinates.

    ``texture_weight`` (``extratcoordw``) switches to MeshLab's *texture-metric* collapser,
    which lifts the quadric to 5D by adding the UV distance term, so collapses that would
    smear the atlas lose against collapses that only move geometry. ``planar_weight``
    (``planarweight``) tunes the rank-3 planar quadric of the plain collapser; ``None`` leaves
    the filter's own default in place.
    """
    current = mesh_set.current_mesh()
    if current.face_number() <= int(target_faces):
        _append(log, "decimation skipped: %d faces are already at or below the target"
                % current.face_number())
        return mesh_set
    start = time.time()
    if texture_weight is not None and _has_wedges(current):
        try:
            mesh_set.apply_filter(
                "meshing_decimation_quadric_edge_collapse_with_texture",
                targetfacenum=int(target_faces),
                qualitythr=float(quality_threshold),
                extratcoordw=float(texture_weight),
                preserveboundary=bool(preserve_boundary),
                optimalplacement=bool(optimal_placement),
                preservenormal=bool(preserve_normal),
                planarquadric=bool(planar_quadric),
            )
            return _report(mesh_set,
                           "meshing_decimation_quadric_edge_collapse_with_texture"
                           "(extratcoordw=%.3f)" % float(texture_weight), start, log)
        except pml.PyMeshLabException as error:
            # The filter refuses a per-corner table it finds inconsistent (degenerate pole fans
            # and the like). The plain collapser still interpolates those corners, it merely
            # cannot weigh them, so the collapse continues instead of failing the whole run.
            _append(log, "the 5D texture-metric collapser refused the wedge table (%s); "
                         "using the 3D collapser" % error)
    params = {"targetfacenum": int(target_faces),
              "autoclean": False,           # True merges UV seams and scrambles the atlas
              "preserveboundary": bool(preserve_boundary),
              "preservenormal": bool(preserve_normal),
              "planarquadric": bool(planar_quadric),
              "optimalplacement": bool(optimal_placement),
              "qualitythr": float(quality_threshold)}
    if planar_weight is not None:
        params["planarweight"] = float(planar_weight)
    mesh_set.apply_filter("meshing_decimation_quadric_edge_collapse", **params)
    return _report(mesh_set, "meshing_decimation_quadric_edge_collapse", start, log)


def wedge_coordinates(mesh):
    """Per-corner texture coordinates of ``mesh`` as an (3 * faces, 2) array, or None."""
    if hasattr(mesh, "has_wedge_tex_coord") and not mesh.has_wedge_tex_coord():
        return None
    try:
        wedges = np.asarray(mesh.wedge_tex_coord_matrix())
    except Exception:
        return None
    if wedges.size == 0 or wedges.shape[0] != mesh.face_number() * 3:
        return None
    return wedges.astype(np.float64)


# --------------------------------------------------------------------------------------
# Legacy array-level pipeline
# --------------------------------------------------------------------------------------
def run_arrays(vertices, faces, uvs, target_faces, repair_topology=True, preserve_normal=True,
               planar_quadric=True, optimal_placement=True, quality_threshold=1.0,
               smooth_normals=True, want_tangents=True):
    """Return ``dict(vertices, faces, uvs, normals, tangents, report)``.

    ``uvs`` may be None; the result then carries no texture coordinates, matching the
    input. Wedge coordinates collapse away when the atlas cannot be trusted, in which
    case the caller falls back to vertex texture coordinates.
    """
    log = []
    mesh_set = build_mesh_set(vertices, faces, uvs)
    if repair_topology:
        clean(mesh_set, log)
    decimate(mesh_set, target_faces, preserve_normal, planar_quadric, optimal_placement,
             quality_threshold, log)

    out = mesh_set.current_mesh()
    out_vertices = out.vertex_matrix().astype(np.float64)
    out_faces = out.face_matrix().astype(np.int64)

    result = {"vertices": out_vertices, "faces": out_faces, "uvs": None,
              "normals": None, "tangents": None, "report": log}
    if uvs is None:
        return result

    wedges = wedge_coordinates(out)
    if wedges is None:
        log.append("wedge texture coordinates unavailable; keeping per-vertex uvs")
        fallback = _vertex_uvs(out, out_vertices.shape[0])
        if fallback is None:
            log.append("per-vertex uvs unavailable too; output carries no texture coordinates")
            return result
        result["uvs"] = fallback
        if smooth_normals:
            result["normals"] = compute_smooth_vertex_normals(out_vertices, out_faces)
        return result

    split_vertices, split_faces, split_uvs, source_index = split_by_uv_with_source(
        out_vertices, out_faces, wedges)
    log.append("re-split at UV seams: %d vertices, %d faces" % (len(split_vertices), len(split_faces)))
    result["vertices"] = split_vertices
    result["faces"] = split_faces
    result["uvs"] = split_uvs

    if smooth_normals:
        # Computed before the seam split so vertices that share a position share a normal,
        # then indexed back: without this the UV seams would also become shading seams.
        base = compute_smooth_vertex_normals(out_vertices, out_faces)
        result["normals"] = base[source_index]
        log.append("rebuilt smooth normals for %d vertices" % len(split_vertices))
        if want_tangents and split_uvs is not None:
            tangents = compute_vertex_tangents(split_vertices, split_faces, split_uvs,
                                               result["normals"])
            if tangents is None:
                log.append("tangents unavailable (degenerate uv triangles)")
            else:
                result["tangents"] = tangents
                log.append("rebuilt tangents for the normal map")
    return result


def _vertex_uvs(mesh, vertex_count: int) -> Optional[np.ndarray]:
    """Per-vertex texture coordinates of ``mesh`` when they match ``vertex_count``."""
    if hasattr(mesh, "has_vertex_tex_coord") and not mesh.has_vertex_tex_coord():
        return None
    try:
        array = np.asarray(mesh.vertex_tex_coord_matrix(), dtype=np.float64)
    except Exception:
        return None
    if array.ndim != 2 or array.shape[0] != vertex_count or array.shape[1] < 2:
        return None
    return array[:, :2]


def split_by_uv_with_source(vertices, faces, wedges):
    """``mesh_convert.split_by_uv_with_source`` with a graceful fallback to ``split_by_uv``."""
    try:
        from .mesh_convert import split_by_uv_with_source as split
    except ImportError:
        from mesh_convert import split_by_uv_with_source as split
    return split(vertices, faces, wedges)


# --------------------------------------------------------------------------------------
# Config / strategy plumbing
# --------------------------------------------------------------------------------------
def canonical_strategy(value: Any) -> str:
    """Map a user-facing strategy name onto ``DecimateConfig.STRATEGIES``."""
    text = str(value or "auto").strip().lower().replace("-", "_").replace(" ", "_")
    text = {"texqem": "tex_qem", "texture_qem": "tex_qem", "texture": "tex_qem",
            "qem": "tex_qem", "keepuv": "keep_uv", "retopo": "retopo_bake",
            "bake": "retopo_bake", "retopobake": "retopo_bake"}.get(text, text)
    supported = tuple(getattr(DecimateConfig, "STRATEGIES", ("auto", "keep_uv", "retopo_bake")))
    if text in supported:
        return text
    if text in ("tex_qem", "keep_uv") and "keep_uv" in supported:
        return "keep_uv"
    return "auto"


def build_config(target_faces, clean=True, strategy="auto", uv_weight=0.05, quality_thr=0.5,
                 planar_weight=0.001, preserve_boundary=True, rebake_normals=True,
                 bake_resolution=2048, crease_angle_deg=None, isotropic_remesh=False):
    """Build the :class:`~contracts.DecimateConfig` of one run from the node's widgets.

    The node's spellings (``uv_weight``, ``rebake_normals``, ``crease_angle_deg``) are mapped
    onto the dataclass fields (``texture_weight``, ``rebake_normal_map``, ``crease_angle``);
    :class:`~contracts.DecimateConfig` exposes both spellings, so a caller may read the config
    back either way. Fields this dataclass does not declare are attached as instance attributes,
    so a config object from an older contract still carries every value the strategies read.
    """
    crease = float(crease_angle_deg) if crease_angle_deg and float(crease_angle_deg) > 0.0 else None
    values = {
        "target_faces": max(int(target_faces or 0), 0),
        "strategy": canonical_strategy(strategy),
        "preserve_boundary": bool(preserve_boundary),
        "quality_thr": float(quality_thr),
        "texture_weight": float(uv_weight),
        "planar_weight": float(planar_weight),
        "crease_angle": crease,
        "rebake_normal_map": bool(rebake_normals),
        "bake_resolution": max(int(bake_resolution or 0), 1),
        "isotropic_remesh": bool(isotropic_remesh),
        "remesh_iterations": ISOTROPIC_REMESH_ITERATIONS if isotropic_remesh else 0,
        "clean": bool(clean),
    }
    fields = set(getattr(DecimateConfig, "__dataclass_fields__", {}) or {})
    declared = {key: value for key, value in values.items() if key in fields}
    config = DecimateConfig(**declared)
    for key, value in values.items():
        if key not in declared:  # keep the value readable even on an older dataclass
            setattr(config, key, value)
    return config


def _import_candidate(name: str):
    """Import ``name`` relatively when this module is a package, absolutely otherwise."""
    try:
        if __package__:
            return importlib.import_module("." + name, __package__)
        return importlib.import_module(name)
    except ImportError as error:
        _IMPORT_ERRORS.append("%s (%s)" % (name, error))
        return None


def resolve_router_class():
    """The :class:`~strategies.auto_router.AutoRouterStrategy` class to run.

    ``strategies`` ships with the node, so there is no built-in second implementation to fall
    back on: an unresolvable router is a broken installation and is reported as such.
    """
    if _ROUTER_CACHE:
        return _ROUTER_CACHE[0]
    for module_name, attribute in ROUTER_CANDIDATES:
        module = _import_candidate(module_name)
        candidate = None if module is None else getattr(module, attribute, None)
        if isinstance(candidate, type):
            _ROUTER_CACHE[:] = [candidate, module_name]
            return candidate
    raise RuntimeError("AutoRouterStrategy could not be imported from %s: %s"
                       % (", ".join(name for name, _ in ROUTER_CANDIDATES),
                          "; ".join(_IMPORT_ERRORS) or "no import error recorded"))


def router_source() -> str:
    """Module the router came from (for the node's log line)."""
    resolve_router_class()
    return str(_ROUTER_CACHE[1])


def ensure_smooth_attributes(payload: MeshPayload, weld_normals: bool = True,
                             crease_angle: Optional[float] = None,
                             want_tangents: bool = True) -> MeshPayload:
    """Fill in missing smooth welded normals and tangents (a no-op when both are present)."""
    normals = payload.normals
    if normals is None:
        normals = compute_smooth_vertex_normals(payload.vertices, payload.faces,
                                                weld=weld_normals, crease_angle=crease_angle)
    tangents = payload.tangents
    if tangents is None and want_tangents and payload.uvs is not None:
        tangents = compute_vertex_tangents(payload.vertices, payload.faces, payload.uvs, normals)
    if normals is payload.normals and tangents is payload.tangents:
        return payload
    return MeshPayload(vertices=payload.vertices, faces=payload.faces, uvs=payload.uvs,
                       normals=normals, tangents=tangents, textures=dict(payload.textures),
                       extra_attrs=payload.extra_attrs)


# --------------------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------------------
#: Old array-call option -> the new parameter slot a shifted positional call lands in.
_LEGACY_POSITIONAL = (("repair_topology", "uv_weight"), ("preserve_normal", "quality_thr"),
                      ("planar_quadric", "planar_weight"), ("optimal_placement", "preserve_boundary"),
                      ("quality_threshold", "rebake_normals"), ("smooth_normals", "bake_resolution"),
                      ("want_tangents", "crease_angle_deg"))


def _run_arrays_call(mesh, target_faces, clean, strategy, legacy, *new_slots):
    """Forward a pre-Slice-4 array call (positional or keyword) to :func:`run_arrays`."""
    legacy = dict(legacy)
    vertices = legacy.pop("vertices", mesh)
    faces = legacy.pop("faces", target_faces)
    uvs = legacy.pop("uvs", None if isinstance(clean, bool) else clean)
    budget = legacy.pop("target_faces", None)
    if budget is None:
        for candidate in (target_faces, strategy):
            if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
                budget = candidate
                break
        else:
            budget = 0
    options = {name: legacy[name] for name in
               ("repair_topology", "preserve_normal", "planar_quadric", "optimal_placement",
                "quality_threshold", "smooth_normals", "want_tangents") if name in legacy}
    slots = dict(zip(("uv_weight", "quality_thr", "planar_weight", "preserve_boundary",
                      "rebake_normals", "bake_resolution", "crease_angle_deg"), new_slots))
    for old_name, slot in _LEGACY_POSITIONAL:
        if old_name not in options and isinstance(slots[slot], bool):
            options[old_name] = slots[slot]
    LOGGER.info("array-level run: target %s faces, wedge uvs %s", budget, uvs is not None)
    return run_arrays(vertices, faces, uvs, int(budget), **options)


def run(mesh=None, target_faces: Optional[int] = None, clean: bool = True, strategy: str = "auto",
        uv_weight: float = 0.05, quality_thr: float = 0.5, planar_weight: float = 0.001,
        preserve_boundary: bool = True, rebake_normals: bool = True, bake_resolution: int = 2048,
        crease_angle_deg: Optional[float] = None, isotropic_remesh: bool = False, **legacy):
    """Decimate ``mesh`` to ``target_faces`` and return a new ``MESH``.

    ``mesh`` is a ComfyUI ``MESH`` (item 0 is used); passing raw arrays instead forwards to
    :func:`run_arrays` with the original ``(vertices, faces, uvs, target_faces, ...)``
    convention -- positionally or through the old ``vertices``/``faces``/``uvs`` keywords -- so
    the previous slice's call sites keep working.

    ``target_faces <= 0``, or a mesh that is already small enough, is a pass-through that only
    fills in the missing smooth normals/tangents before packing the mesh back.
    """
    if legacy.get("vertices") is not None or not hasattr(mesh, "vertices"):
        return _run_arrays_call(mesh, target_faces, clean, strategy, legacy, uv_weight, quality_thr,
                                planar_weight, preserve_boundary, rebake_normals, bake_resolution,
                                crease_angle_deg)

    started = time.perf_counter()
    config = build_config(target_faces=target_faces, clean=clean, strategy=strategy,
                          uv_weight=uv_weight, quality_thr=quality_thr,
                          planar_weight=planar_weight, preserve_boundary=preserve_boundary,
                          rebake_normals=rebake_normals, bake_resolution=bake_resolution,
                          crease_angle_deg=crease_angle_deg, isotropic_remesh=isotropic_remesh)
    payload = unpack_mesh(mesh)
    budget = int(config.target_faces)
    if budget <= 0 or payload.face_count <= budget:
        LOGGER.info("no decimation needed: %d faces, %d vertices (target %s); "
                    "only normals/tangents are checked",
                    payload.face_count, payload.vertex_count,
                    "unset" if budget <= 0 else budget)
        enriched = ensure_smooth_attributes(payload, getattr(config, "weld_normals", True),
                                           config.crease_angle)
        return pack_mesh(mesh, enriched)

    payload = ensure_smooth_attributes(payload, getattr(config, "weld_normals", True),
                                       config.crease_angle)
    router = resolve_router_class()()
    out = router.execute(payload, config)
    elapsed = time.perf_counter() - started
    LOGGER.info("%s: %d -> %d faces, %d -> %d vertices in %.2fs (target %d, clean %s)",
                router_source(), payload.face_count, out.face_count, payload.vertex_count,
                out.vertex_count, elapsed, budget, bool(clean))
    return pack_mesh(mesh, out)


__all__ = [
    "CLEAN_STEPS",
    "ISOTROPIC_REMESH_ITERATIONS",
    "build_config",
    "build_mesh_set",
    "canonical_strategy",
    "clean",
    "compute_smooth_vertex_normals",
    "compute_vertex_tangents",
    "decimate",
    "ensure_smooth_attributes",
    "pack_mesh",
    "resolve_router_class",
    "router_source",
    "run",
    "run_arrays",
    "split_by_uv_with_source",
    "unpack_mesh",
    "wedge_coordinates",
]
