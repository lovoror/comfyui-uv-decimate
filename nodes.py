"""ComfyUI node: SOTA UV-safe decimation ("Decimate Mesh (Keep UV)", 3d/decimate).

The node is a thin adapter: it resolves the preset into a face budget, coerces the widgets
(a saved workflow may carry the previous slice's values or a shifted widget row) and calls
:func:`uv_pipeline.run`, which owns the whole pipeline -- payload snapshot, strategy routing
(``auto`` / ``tex_qem`` / ``retopo_bake``), texture-metric collapse, atlas-preserving re-split,
normal-map rebake and the pack back into a ``MESH``.

Backward compatibility
----------------------
* ``NODE_NAME``/``CLASS`` and the display name are unchanged, so existing workflow JSONs keep
  resolving the node, and the widget row still starts with ``mesh``, ``preset``,
  ``target_faces``, ``clean`` in that order.
* Older spellings are still accepted: ``NODE_ID``, ``FACE_PRESETS``, the ``"50k"`` / ``"200k"``
  presets, ``first_item``-style array input and the legacy keyword names
  (``face_preset``, ``target_face_count``, ``repair_topology``, ``quality_threshold``).
  Widgets a legacy workflow fills with booleans where the new node expects a number fall back
  to the new default instead of being clamped, so such a workflow runs with SOTA defaults.
"""
from __future__ import annotations

import logging
import time
from types import SimpleNamespace

try:
    from . import mesh_convert, uv_pipeline
except ImportError:  # loaded as a plain directory rather than a package
    import mesh_convert
    import uv_pipeline

LOGGER = logging.getLogger("comfyui_uv_decimate")

NODE_NAME = "DecimateMeshKeepUV"
DISPLAY_NAME = "Decimate Mesh (Keep UV)"
CATEGORY = "3d/decimate"

PRESET_CUSTOM = "custom"
#: Canonical presets. ``custom`` (0) means "use the target_faces slider".
PRESET_FACES = {
    "1k": 1000,
    "2k": 2000,
    "5k": 5000,
    "10k": 10000,
    "20k": 20000,
    "30k": 30000,
    "50k (recommended)": 50000,
    "100k": 100000,
    PRESET_CUSTOM: 0,
}
PRESET_CHOICES = [
    "1k",
    "2k",
    "5k",
    "10k",
    "20k",
    "30k",
    "50k (recommended)",
    "100k",
    PRESET_CUSTOM,
]
DEFAULT_PRESET = "50k (recommended)"
#: Preset spellings of the previous slice: still resolved, so old workflows keep their budget.
LEGACY_PRESET_FACES = {"50k": 50000, "200k": 200000, "500k": 500000}
#: The previous slice's ``FACE_PRESETS`` dict, kept importable for external callers.
FACE_PRESETS = {
    "1k": 1000,
    "2k": 2000,
    "5k": 5000,
    "10k": 10000,
    "20k": 20000,
    "30k": 30000,
    "50k": 50000,
    "100k": 100000,
    "200k": 200000,
}

DEFAULT_TARGET_FACES = 50000
MIN_TARGET_FACES = 100
MAX_TARGET_FACES = 50000000
STRATEGY_CHOICES = ["auto", "tex_qem", "retopo_bake"]
STRATEGY_TOOLTIP = ("auto: 5D Texture-Metric QEM above UV seam floor, UV-free Retopo + xatlas "
                    "+ GPU BVH PBR Bake below seam floor.")
#: Below this share of the input faces, thin parts (ropes, straps) start to collapse.
THIN_DETAIL_SHARE = 0.03
#: Legacy node inputs still accepted as keywords; only the first four have a new equivalent.
LEGACY_ARGUMENTS = ("face_preset", "target_face_count", "repair_topology", "quality_threshold",
                    "preserve_normal", "planar_quadric", "optimal_placement", "smooth_normals")
LEGACY_MAPPED = ("face_preset", "target_face_count", "repair_topology", "quality_threshold")


def _emit(node_id, message):
    """Progress text to the ComfyUI frontend when available, plus stdout."""
    try:
        from server import PromptServer
        PromptServer.instance.send_progress_text(message, node_id)
    except Exception:
        pass
    print("[%s %s] %s" % (NODE_NAME, time.strftime("%H:%M:%S"), message), flush=True)


# --------------------------------------------------------------------------------------
# Widget coercion: a saved workflow may hand us an older or shifted value.
# --------------------------------------------------------------------------------------
def _coerce_bool(value, default):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "1", "yes", "on"):
            return True
        if text in ("false", "0", "no", "off"):
            return False
    return default


def _coerce_number(value, default):
    """``float(value)`` or ``None``; booleans mean "the old widget was a checkbox"."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _coerce_float(value, default, low, high):
    number = _coerce_number(value, default)
    if number is None:
        return float(default)
    return float(min(max(number, low), high))


def _coerce_int(value, default, low, high):
    number = _coerce_number(value, default)
    if number is None:
        return int(default)
    return int(min(max(round(number), low), high))


def _coerce_choice(value, choices, default):
    return value if isinstance(value, str) and value in choices else default


def _preset_faces(preset):
    """Face count of ``preset`` (current or legacy spelling), or ``None`` for custom."""
    key = str(preset).strip()
    if key == PRESET_CUSTOM or not key:
        return None
    if key in PRESET_FACES:
        return PRESET_FACES[key]
    if key in LEGACY_PRESET_FACES:
        return LEGACY_PRESET_FACES[key]
    if key in FACE_PRESETS:
        return FACE_PRESETS[key]
    number = _coerce_number(key, None)
    return int(number) if number and number > 0 else None


def _resolve_faces(preset, target_faces):
    """``target_faces`` when the preset is custom, else the preset's own face count."""
    resolved = _preset_faces(preset)
    if resolved is not None:
        return max(int(resolved), 1)
    return max(_coerce_int(target_faces, DEFAULT_TARGET_FACES, 1, MAX_TARGET_FACES), 1)


# --------------------------------------------------------------------------------------
# Batched MESH handling
# --------------------------------------------------------------------------------------
def _torch():
    import torch
    return torch


def _batch_helpers():
    """``(get_mesh_batch_item, pack_variable_mesh_batch)`` from ComfyUI, or ``None``."""
    try:
        from comfy_extras.nodes_save_3d import get_mesh_batch_item, pack_variable_mesh_batch
    except Exception:
        return None
    return get_mesh_batch_item, pack_variable_mesh_batch


def _mesh_batch_size(mesh) -> int:
    vertices = getattr(mesh, "vertices", None)
    if isinstance(vertices, (list, tuple)):
        return max(len(vertices), 1)
    shape = getattr(vertices, "shape", None)
    return int(shape[0]) if shape else 1


def _slice_attribute(mesh, name, index, count, batched=True):
    value = getattr(mesh, name, None)
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return value[index] if index < len(value) else None
    if not batched:
        return value
    try:
        return value[index, :count]
    except Exception:
        return None


def _slice_image(value, index):
    """Item ``index`` of an image attribute; an unbatched image is shared by every item."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return value[index] if index < len(value) else None
    if getattr(value, "ndim", 0) == 4:
        return value[index]
    return value


def _item_mesh(source, index, get_mesh_batch_item):
    """One batch item of ``source`` as a standalone single-item mesh for the pipeline.

    ComfyUI's ``get_mesh_batch_item`` slices to unbatched per-item tensors, so every one of them
    is unsqueezed back into the single-item batch shape the payload conversion expects.
    """
    vertices, faces, colors, uvs, normals = get_mesh_batch_item(source, index)
    tangents = _slice_attribute(source, "tangents", index, int(vertices.shape[0]))
    item = SimpleNamespace(vertices=vertices.unsqueeze(0), faces=faces.unsqueeze(0),
                           uvs=None if uvs is None else uvs.unsqueeze(0),
                           normals=None if normals is None else normals.unsqueeze(0),
                           tangents=None if tangents is None else tangents.unsqueeze(0))
    if colors is not None:
        item.vertex_colors = colors.unsqueeze(0)
    for name in mesh_convert.IMAGE_ATTRS + mesh_convert.SCALAR_ATTRS:
        value = (_slice_image(getattr(source, name, None), index) if name in mesh_convert.IMAGE_ATTRS
                 else _slice_attribute(source, name, index, 0))
        if value is not None:
            setattr(item, name, value)
    return item


def _pack_batch(source, outputs, node_id, pack_variable_mesh_batch):
    """Rebuild one MESH from the per-item pipeline results."""
    vertices, faces, uvs, normals, tangents = [], [], [], [], []
    keep_uvs = all(getattr(out, "uvs", None) is not None for out in outputs)
    keep_normals = all(getattr(out, "normals", None) is not None for out in outputs)
    keep_tangents = all(getattr(out, "tangents", None) is not None for out in outputs)
    for out in outputs:
        vertices.append(out.vertices[0])
        faces.append(out.faces[0])
        if keep_uvs:
            uvs.append(out.uvs[0])
        if keep_normals:
            normals.append(out.normals[0])
        if keep_tangents:
            tangents.append(out.tangents[0])
    images = _batch_images(source, outputs, node_id)
    scalars = {name: getattr(source, name) for name in mesh_convert.SCALAR_ATTRS
               if getattr(source, name, None) is not None}
    return pack_variable_mesh_batch(vertices, faces, uvs=uvs or None, normals=normals or None,
                                    tangents=tangents or None, **images, **scalars)


def _batch_images(source, outputs, node_id):
    """Stack the per-item images; a mismatch falls back to the source's own stack."""
    torch = _torch()
    stacked = {}
    for name in mesh_convert.IMAGE_ATTRS:
        per_item = [getattr(out, name, None) for out in outputs]
        fallback = getattr(source, name, None)
        if any(item is None for item in per_item):
            if fallback is not None:
                stacked[name] = fallback
            continue
        shapes = {tuple(item.shape[1:]) for item in per_item}
        if len(shapes) != 1:
            _emit(node_id, "%s differs per batch item; keeping the source texture" % name)
            stacked[name] = fallback
            continue
        try:
            stacked[name] = torch.cat([item if item.ndim == 4 else item.unsqueeze(0)
                                       for item in per_item], dim=0)
        except Exception:
            stacked[name] = fallback
    return {name: value for name, value in stacked.items() if value is not None}


def _face_count(mesh) -> int:
    faces = getattr(mesh, "faces", None)
    if faces is None:
        return 0
    if isinstance(faces, (list, tuple)):
        return sum(int(getattr(f, "shape", [0])[0]) for f in faces)
    counts = getattr(mesh, "face_counts", None)
    if counts is not None:
        try:
            return int(counts.sum().item()) if hasattr(counts, "sum") else int(sum(counts))
        except Exception:
            pass
    shape = getattr(faces, "shape", None)
    return int(shape[1]) if shape and len(shape) >= 2 else 0


# --------------------------------------------------------------------------------------
# Node
# --------------------------------------------------------------------------------------
class DecimateMeshKeepUV:
    """Reduce a mesh's face count while keeping its UV set, textures and shading."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "preset": (PRESET_CHOICES, {
                    "default": DEFAULT_PRESET,
                    "tooltip": "Ready-made face counts. Anything other than custom overrides "
                               "the slider below. Fewer faces means thinner parts are more "
                               "likely to collapse."}),
                "target_faces": ("INT", {
                    "default": DEFAULT_TARGET_FACES, "min": MIN_TARGET_FACES,
                    "max": MAX_TARGET_FACES, "step": 1000,
                    "tooltip": "Face count to aim for; used when preset is custom. The strategies "
                               "may stop slightly above or below it."}),
                "clean": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Weld duplicate vertices, drop duplicate faces and remove "
                               "unreferenced vertices before collapsing. This is what welds the "
                               "positions a UV split duplicated, so the collapser stops reading "
                               "chart borders as open edges. Keeps the UV set."}),
            },
            "optional": {
                "strategy": (STRATEGY_CHOICES, {"default": "auto", "tooltip": STRATEGY_TOOLTIP}),
                "rebake_normals": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Rebake composed tangent-space normal map into the new low-poly "
                               "TBN frame to eliminate triangular faceting."}),
                "uv_weight": ("FLOAT", {
                    "default": 0.05, "min": 0.01, "max": 20.0, "step": 0.05,
                    "tooltip": "5D QEM UV attribute weight (extratcoordw)."}),
                "quality_thr": ("FLOAT", {
                    "default": 0.5, "min": 0.05, "max": 1.0, "step": 0.05,
                    "tooltip": "Triangle quality threshold (0.5 prevents needle/sliver "
                               "triangles)."}),
                "bake_resolution": ("INT", {
                    "default": 2048, "min": 512, "max": 4096, "step": 256,
                    "tooltip": "Texture baking resolution when retopo_bake strategy is used."}),
                "crease_angle_deg": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 180.0, "step": 1.0,
                    "tooltip": "Hard-edge crease angle in degrees (0 = full smooth welded vertex "
                               "normals, recommended when normal map is present)."}),
                "isotropic_remesh": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Run AutoRemesher-style equilateral isotropic remeshing before "
                               "xatlas in retopo_bake mode."}),
            },
            "hidden": {"node_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "execute"
    CATEGORY = CATEGORY
    DESCRIPTION = ("SOTA UV-safe decimation. Above the UV seam floor a 5D texture-metric QEM "
                   "collapse keeps the existing atlas; below it the mesh is retopologised "
                   "UV-free and every PBR channel is baked onto a fresh xatlas atlas. Smooth "
                   "welded normals and tangents are always rebuilt, and a normal map is "
                   "re-projected into the new TBN frames.")

    def execute(self, mesh, preset=DEFAULT_PRESET, target_faces=DEFAULT_TARGET_FACES, clean=True,
                strategy="auto", rebake_normals=True,
                uv_weight=0.05, quality_thr=0.5, bake_resolution=2048, crease_angle_deg=0.0,
                isotropic_remesh=False, node_id=None, **legacy):
        started = time.time()
        preset = legacy.get("face_preset", preset)
        target_faces = legacy.get("target_face_count", target_faces)
        clean = legacy.get("repair_topology", clean)
        quality_thr = legacy.get("quality_threshold", quality_thr)
        ignored = sorted(name for name in legacy if name not in LEGACY_MAPPED)
        preset = _coerce_choice(preset, PRESET_CHOICES, DEFAULT_PRESET) \
            if _preset_faces(preset) is None and str(preset) not in LEGACY_PRESET_FACES \
            else preset
        clean = _coerce_bool(clean, True)
        strategy = _coerce_choice(strategy, STRATEGY_CHOICES, "auto")
        rebake_normals = _coerce_bool(rebake_normals, True)
        uv_weight = _coerce_float(uv_weight, 0.05, 0.01, 20.0)
        quality_thr = _coerce_float(quality_thr, 0.5, 0.05, 1.0)
        bake_resolution = _coerce_int(bake_resolution, 2048, 512, 4096)
        crease_angle_deg = _coerce_float(crease_angle_deg, 0.0, 0.0, 180.0)
        isotropic_remesh = _coerce_bool(isotropic_remesh, False)
        faces = _resolve_faces(preset, target_faces)
        crease = float(crease_angle_deg) if crease_angle_deg and crease_angle_deg > 0.0 else None

        in_faces = _face_count(mesh)
        _emit(node_id, "input: %d faces, target %d (preset %s), strategy %s, clean %s"
              % (in_faces, faces, preset, strategy, clean))
        if ignored:
            _emit(node_id, "ignoring legacy options %s (not exposed by this node version)"
                  % ", ".join(ignored))

        call = dict(target_faces=faces, clean=clean, strategy=strategy, uv_weight=uv_weight,
                    quality_thr=quality_thr, rebake_normals=rebake_normals,
                    bake_resolution=bake_resolution, crease_angle_deg=crease,
                    isotropic_remesh=isotropic_remesh)
        packed = self._run_items(mesh, call, node_id)

        out_faces = _face_count(packed)
        _emit(node_id, "%d -> %d faces in %.1fs (uv_weight %.2f, quality %.2f, crease %s)"
              % (in_faces, out_faces, time.time() - started, uv_weight, quality_thr,
                 "smooth" if crease is None else "%.0f deg" % crease))
        if in_faces and out_faces < in_faces * THIN_DETAIL_SHARE:
            _emit(node_id, "only %.1f%% of the input faces are left: thin parts such as ropes or "
                           "straps may have collapsed. Check the result before using it."
                  % (100.0 * out_faces / in_faces))
        return (packed,)

    #: The previous slice's entry point, kept so direct callers keep working.
    def run(self, mesh, face_preset, target_face_count, **kwargs):
        return self.execute(mesh, face_preset, target_face_count, **kwargs)

    @staticmethod
    def _run_items(mesh, call, node_id):
        """Run the pipeline on every batch item, or on the mesh itself for a single item."""
        try:
            count = _mesh_batch_size(mesh)
        except Exception:
            count = 1
        helpers = _batch_helpers() if count > 1 else None
        if helpers is None:
            if count > 1:
                _emit(node_id, "batch of %d items: nodes_save_3d is unavailable, decimating item 0"
                      % count)
            return uv_pipeline.run(mesh, **call)
        get_item, pack_batch = helpers
        try:
            outputs = [uv_pipeline.run(_item_mesh(mesh, index, get_item), **call)
                       for index in range(count)]
            return _pack_batch(mesh, outputs, node_id, pack_batch)
        except Exception as error:  # a MESH missing the attributes the helper reads
            _emit(node_id, "batch handling unavailable (%s: %s); decimating item 0"
                  % (type(error).__name__, error))
            return uv_pipeline.run(mesh, **call)


NODE_CLASS_MAPPINGS = {NODE_NAME: DecimateMeshKeepUV}
NODE_DISPLAY_NAME_MAPPINGS = {NODE_NAME: DISPLAY_NAME}
#: Previous slice's name for the node id.
NODE_ID = NODE_NAME
