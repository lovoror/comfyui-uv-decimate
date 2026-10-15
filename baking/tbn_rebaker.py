"""Tangent-space normal rebaking and high-to-low PBR transfer for UV-decimated meshes.

* `rebake_normal_map_uv_space` -- Strategy A (`keep_uv`): the low-poly keeps the high-poly's UV
  parameterization, so the existing tangent-space normal map is re-expressed in the low-poly's
  per-texel TBN frames.
* `bake_pbr_maps_3d_transfer` -- Strategy B (`retopo_bake`): the low-poly has a brand new atlas,
  so every channel is transferred through a 3D tight-cage ray cast.

UV <-> texel convention (identical to ComfyUI's rasterizer -- never flip!):
`_rasterize_uv_barycentric` uses `tri_uv = (uvs * resolution)[faces]` with coverage at texel
centres (`nodes_mesh_postprocess.py:160-161`), i.e. `row y = v * resolution`, `column x = u *
resolution` (verified: a triangle at `v = 0.05` covers row 3 of a 64x64 grid).  Direct texel
reads index `y * resolution + x`; `grid_sample` gets `gx = 2u - 1`, `gy = 2v - 1` with
`align_corners=False`.  Never add a `1 - v` flip here.

Dtypes/shapes: vertices `(V,3) f32`, faces `(F,3) int`, uvs `(V,2) f32`, normals `(V,3) f32`,
tangents `(V,4) f32` (xyz + glTF handedness `w`); textures `(H,W,C) f32` and every returned map
are `(resolution, resolution, 3) f32` in `[0,1]` (glTF/OpenGL +Y tangent space).
"""
from __future__ import annotations

import functools
import logging
from types import SimpleNamespace
from typing import Optional

import numpy as np

LOGGER = logging.getLogger(__name__)

FLAT_TANGENT_NORMAL = np.array([0.5, 0.5, 1.0], dtype=np.float32)
DIRECT_CONE_DOT = 0.15          # Strategy A: direct UV-space high/low texel pair
RAY_CONE_DOT = 0.15             # Strategy A: tight-cage ray hit
TRANSFER_CONE_DOT = 0.10        # Strategy B: 3D transfer (else fall back to the low normal)
SOFT_CONE_WIDTH = 0.35          # width of the smooth fade just above each threshold
DIRECT_POSITION_TOLERANCE = 0.015   # fraction of the bbox diagonal
DEFAULT_RESOLUTION = 2048
DEFAULT_CAGE_RATIO = 0.012
CLOSEST_POINT_CHUNK = 200_000
NORMAL_CHANNEL = "normal_map"
# glTF neutral values, used only where the high-poly has no sampleable surface.
CHANNEL_NEUTRAL = {"texture": 1.0, "metallic_roughness": 1.0, "emissive": 0.0}


@functools.lru_cache(maxsize=1)
def _comfy() -> SimpleNamespace:
    """Lazily import ComfyUI's GPU helpers once (keeps this module import-light).

    Contracts (`comfy_extras.nodes_mesh_postprocess`): `_rasterize_uv_barycentric(faces, uvs,
    res)` -> `(face_idx [H,W], bary [H,W,3], coverage [H,W] bool)`; `_build_triangle_bvh(tri)`
    -> BVH; `_closest_hit_rays_bvh(orig, dirs, tri, bvh, tmin, tmax, cull_backface)` -> `(t,
    face with -1 on miss, hit)`; `_closest_points_on_mesh_bvh(Q, tri, bvh, return_face=True)`
    -> `(points, face)`; `_barycentric(p, tri)` -> `[N,3]`; `_jfa_fill_gpu(img, mask)` -> dilated.
    """
    import torch  # ComfyUI is a runtime dependency only

    from comfy_extras.nodes_mesh_postprocess import (
        _barycentric, _build_triangle_bvh, _closest_hit_rays_bvh, _closest_points_on_mesh_bvh,
        _jfa_fill_gpu, _rasterize_uv_barycentric,
    )

    return SimpleNamespace(
        torch=torch, rasterize=_rasterize_uv_barycentric, build_bvh=_build_triangle_bvh,
        closest_hits=_closest_hit_rays_bvh, closest_points=_closest_points_on_mesh_bvh,
        barycentric=_barycentric, jfa_fill=_jfa_fill_gpu,
    )


def _device():
    import comfy.model_management

    return comfy.model_management.get_torch_device()


def _tensor(array, dtype, device):
    torch = _comfy().torch
    return torch.from_numpy(np.ascontiguousarray(array, dtype=dtype)).to(device)


def _check_mesh(label, verts, faces, uv, normals, tangents):
    """Fail fast on a malformed mesh instead of rasterizing garbage."""
    verts, faces = np.asarray(verts), np.asarray(faces)
    shape_ok = verts.ndim == 2 and verts.shape[1] == 3 and faces.ndim == 2 and faces.shape[1] == 3
    index_ok = not faces.size or (int(faces.min()) >= 0 and int(faces.max()) < verts.shape[0])
    if not (shape_ok and index_ok):
        raise ValueError(f"{label}: bad topology {verts.shape} / {faces.shape}")
    if uv is not None and np.asarray(uv).shape != (verts.shape[0], 2):
        raise ValueError(f"{label}: uvs must be (V,2), got {np.asarray(uv).shape}")
    if np.asarray(normals).shape != verts.shape or np.asarray(tangents).shape != (verts.shape[0], 4):
        raise ValueError(f"{label}: normals must be (V,3) and tangents (V,4)")


def _bbox_diagonal(*vertex_sets):
    """Diagonal of the union bounding box (identical when low is derived from high)."""
    usable = [np.asarray(v, dtype=np.float32) for v in vertex_sets if np.asarray(v).size]
    if not usable:
        return 0.0
    lo = np.min([v.min(axis=0) for v in usable], axis=0)
    hi = np.max([v.max(axis=0) for v in usable], axis=0)
    return float(np.linalg.norm(hi - lo))


def _mesh_tensors(verts_np, faces_np, normals_np, tangents_np, device):
    """Pack one mesh's attributes into device tensors: (faces, verts, normals, tangents)."""
    return (
        _tensor(faces_np, np.int64, device), _tensor(verts_np, np.float32, device),
        _tensor(normals_np, np.float32, device), _tensor(tangents_np, np.float32, device),
    )


class _SurfaceFrame:
    """Per-texel surface frame sampled on one mesh's own UV chart grid."""

    __slots__ = ("mask", "position", "normal", "tangent", "bitangent")

    def __init__(self, mask, position, normal, tangent, bitangent):
        self.mask, self.position = mask, position
        self.normal, self.tangent, self.bitangent = normal, tangent, bitangent


def _orthonormal_frame(normal_raw, tangent_raw, handedness):
    """Gram-Schmidt tangent against the interpolated normal; bitangent = w * (N x T)."""
    torch = _comfy().torch
    normal = torch.nn.functional.normalize(normal_raw, dim=-1, eps=1e-6)
    orthogonal = tangent_raw - normal * (normal * tangent_raw).sum(-1, keepdim=True)
    tangent = torch.nn.functional.normalize(orthogonal, dim=-1, eps=1e-6)
    return normal, tangent, handedness[:, None] * torch.cross(normal, tangent, dim=-1)


def _interp_at(attr, faces, face_ids, bary):
    """Barycentric interpolation of a per-vertex attribute at (face, bary) pairs."""
    return (bary[:, :, None] * attr[faces[face_ids]]).sum(1)


def _frame_at(normals, tangents, faces, face_ids, bary):
    """Interpolated, orthonormalized (N, T, B) at (face, bary) pairs."""
    normal_raw = _interp_at(normals, faces, face_ids, bary)
    tangent_raw = _interp_at(tangents[:, :3], faces, face_ids, bary)
    handedness = _interp_at(tangents[:, 3:4], faces, face_ids, bary)[:, 0]
    return _orthonormal_frame(normal_raw, tangent_raw, handedness)


def _rasterize_frame(mesh, faces_np, uv_np, resolution):
    """Rasterize a mesh in its own UV space into a per-covered-texel surface frame.

    Returns `(frame | None, coverage [H,W] bool, face_idx, bary)`; the frame's rows follow
    `nonzero(coverage)` row-major order.
    """
    comfy = _comfy()
    face_idx, bary, coverage = comfy.rasterize(faces_np, uv_np, resolution)
    if not bool(coverage.any()):
        return None, coverage, face_idx, bary
    faces, verts, normals, tangents = mesh
    face_ids, bary_sel = face_idx[coverage], bary[coverage]
    position = _interp_at(verts, faces, face_ids, bary_sel)
    normal, tangent, bitangent = _frame_at(normals, tangents, faces, face_ids, bary_sel)
    return _SurfaceFrame(coverage, position, normal, tangent, bitangent), coverage, face_idx, bary


def _compose_world_normal(normal, tangent, bitangent, map_rgb):
    """Rotate a tangent-space `[K,3]` map sample (`None` = no map) into world space via a TBN."""
    if map_rgb is None:
        return normal
    torch = _comfy().torch
    ts = torch.nn.functional.normalize(map_rgb * 2.0 - 1.0, dim=-1, eps=1e-6)
    world = ts[:, 0:1] * tangent + ts[:, 1:2] * bitangent + ts[:, 2:3] * normal
    return torch.nn.functional.normalize(world, dim=-1, eps=1e-6)


def _encode_tangent_normal(nh, normal, tangent, bitangent):
    """Project a world normal into a low-poly TBN frame and encode into `[0,1]`."""
    torch = _comfy().torch
    ts = torch.stack([(nh * tangent).sum(-1), (nh * bitangent).sum(-1),
                      (nh * normal).sum(-1).clamp_min(0.15)], dim=-1)
    ts = torch.nn.functional.normalize(ts, dim=-1, eps=1e-6)
    return ts.mul_(0.5).add_(0.5).clamp_(0.0, 1.0)


def _cone_alpha(nh, normal, threshold):
    """Smooth 0..1 acceptance of `nh` against a texel normal (0 -> use the texel normal)."""
    return ((nh * normal).sum(-1) - threshold).div_(SOFT_CONE_WIDTH).clamp_(0.0, 1.0)


def _uv_to_grid(uv):
    """UV -> `grid_sample` coords `gx = 2u - 1`, `gy = 2v - 1` (no `v` flip; see module doc)."""
    torch = _comfy().torch
    return torch.stack([2.0 * uv[:, 0] - 1.0, 2.0 * uv[:, 1] - 1.0], dim=-1)


def _sample_texture(image_np, uv):
    """Bilinear sample of an `(H,W,C)` map at UVs; edge-clamped; returns `[N,C]`."""
    torch = _comfy().torch
    image = _tensor(image_np, np.float32, uv.device)
    if image.ndim != 3:
        raise ValueError(f"texture must be (H,W,C), got {tuple(image.shape)}")
    grid = _uv_to_grid(uv)[None, :, None, :]
    sampled = torch.nn.functional.grid_sample(
        image.permute(2, 0, 1)[None], grid, mode="bilinear", padding_mode="border",
        align_corners=False,
    )
    return sampled[0, :, :, 0].transpose(0, 1)


def _texel_values(image_np, texel_yx, resolution):
    """Nearest read of the texel under each `(y,x)` index -> `[K,C]` (resized if needed)."""
    torch = _comfy().torch
    image = np.asarray(image_np, dtype=np.float32)
    if image.ndim != 3:
        raise ValueError(f"texture must be (H,W,C), got {image.shape}")
    if image.shape[0] != resolution or image.shape[1] != resolution:
        tex = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)[None]
        resized = torch.nn.functional.interpolate(
            tex, size=(resolution, resolution), mode="bilinear", align_corners=False
        )
        image = resized[0].permute(1, 2, 0).contiguous().cpu().numpy()
    flat = _tensor(image, np.float32, texel_yx.device).reshape(-1, image.shape[2])
    return flat[texel_yx[:, 0].long() * resolution + texel_yx[:, 1].long()]


def _fit_rgb(values, fill):
    """Force `[N,C]` to `[N,3]`; missing channels take the channel's neutral value."""
    torch = _comfy().torch
    if values.shape[1] == 3:
        return values
    if values.shape[1] > 3:
        return values[:, :3]
    pad = values[:, -1:] * 0.0 + float(fill)
    return torch.cat([values, pad.expand(-1, 3 - values.shape[1])], dim=1)


def _finish_map(texel_yx, values, resolution, default):
    """Scatter `[K,3]` texel values onto an image seeded with `default`, then JFA-dilate."""
    default = np.asarray(default, dtype=np.float32).reshape(1, 1, 3)
    out = np.tile(default, (resolution, resolution, 1))
    coverage = np.zeros((resolution, resolution), dtype=bool)
    yx = texel_yx.detach().cpu().numpy().astype(np.int64, copy=False)
    if yx.size:
        coverage[yx[:, 0], yx[:, 1]] = True
        out[yx[:, 0], yx[:, 1]] = values.detach().float().cpu().numpy()
    # Dilate into the UV gutter so bilinear/mip sampling at chart edges cannot bleed.
    return _comfy().jfa_fill(out, coverage)


def _closest_hits(points, normals, tri, bvh, cage):
    """Fast-path primary ray + closest-point anchored normal-parallel outer-shell transfer:
    1. Cast primary ray from `P + cage*N` along `-N`. Accept immediately when the hit lies
       within `0.28 * cage` of `P` and its face normal is parallel (`dot > 0.72`).
    2. For the remaining ~18% of texels (misses, grazing hits, or far overhanging hits),
       snap to the nearest high-poly point `cp_pt` and lift along `fn_cp` by `0.55 * cage`
       to resolve double-walled shells without grazing adjacent 3D relief walls.
    """
    comfy, torch = _comfy(), _comfy().torch
    orig_p = points + normals * cage
    dist_p, face_p, hit_p = comfy.closest_hits(
        orig_p, -normals, tri, bvh, tmin=0.0, tmax=2.0 * cage, cull_backface=True
    )
    hp_p = orig_p - dist_p[:, None] * normals
    sf_p = face_p.clamp_min(0)
    fn_p = torch.nn.functional.normalize(
        torch.cross(
            tri[sf_p, 1] - tri[sf_p, 0],
            tri[sf_p, 2] - tri[sf_p, 0],
            dim=-1,
        ),
        dim=-1,
        eps=1e-6,
    )
    fast_ok = hit_p & (torch.abs(cage - dist_p) <= 0.28 * cage) & ((fn_p * normals).sum(-1) > 0.72)
    need_refine = ~fast_ok
    if not bool(need_refine.any()):
        return hp_p, face_p

    r_idx = torch.nonzero(need_refine, as_tuple=True)[0]
    pts_r = points[r_idx]
    n_r = normals[r_idx]

    cp_pt = torch.empty_like(pts_r)
    cp_face = torch.full((pts_r.shape[0],), -1, dtype=torch.long, device=points.device)
    for chunk in torch.arange(pts_r.shape[0], device=points.device).split(CLOSEST_POINT_CHUNK):
        snapped, snapped_face = comfy.closest_points(
            pts_r[chunk], tri, bvh, return_face=True
        )
        cp_pt[chunk] = snapped
        cp_face[chunk] = snapped_face

    safe_cp = cp_face.clamp_min(0)
    fn_cp = torch.nn.functional.normalize(
        torch.cross(
            tri[safe_cp, 1] - tri[safe_cp, 0],
            tri[safe_cp, 2] - tri[safe_cp, 0],
            dim=-1,
        ),
        dim=-1,
        eps=1e-6,
    )
    cp_front = (fn_cp * n_r).sum(-1)
    ray_n = torch.where(cp_front[:, None] > 0.35, fn_cp, n_r)

    lift = 0.55 * cage
    origin = cp_pt + ray_n * lift
    dist, face, hit = comfy.closest_hits(
        origin, -ray_n, tri, bvh, tmin=0.0, tmax=1.10 * lift, cull_backface=True
    )
    hp_cand = origin - dist[:, None] * ray_n
    safe_face = face.clamp_min(0)
    fn_hit = torch.nn.functional.normalize(
        torch.cross(
            tri[safe_face, 1] - tri[safe_face, 0],
            tri[safe_face, 2] - tri[safe_face, 0],
            dim=-1,
        ),
        dim=-1,
        eps=1e-6,
    )
    use_hit = hit & ((fn_hit * ray_n).sum(-1) > 0.72)
    hp_r = torch.where(use_hit[:, None], hp_cand, cp_pt)
    hf_r = torch.where(use_hit, face, cp_face)

    need_wide = (~use_hit) & (cp_front <= 0.25) & hit_p[r_idx]
    if bool(need_wide.any()):
        hp_r[need_wide] = hp_p[r_idx][need_wide]
        hf_r[need_wide] = face_p[r_idx][need_wide]

    hit_point = hp_p.clone()
    hit_face = face_p.clone()
    hit_point[r_idx] = hp_r
    hit_face[r_idx] = hf_r
    return hit_point, hit_face


def rebake_normal_map_uv_space(
    high_v: np.ndarray, high_f: np.ndarray, high_uv: np.ndarray, high_n: np.ndarray,
    high_t: np.ndarray, high_normal_map: Optional[np.ndarray],
    low_v: np.ndarray, low_f: np.ndarray, low_uv: np.ndarray, low_n: np.ndarray,
    low_t: np.ndarray, resolution: Optional[int] = None,
) -> np.ndarray:
    """Re-project a tangent-space normal map from `high`'s TBN frames into `low`'s.

    `low` keeps `high`'s UVs, so most texels resolve by a direct UV lookup; texels where the
    decimator nudged a UV edge fall back to a tight-cage 3D lookup.  Returns `(res, res, 3) f32`.
    """
    comfy, torch = _comfy(), _comfy().torch
    if resolution is None:
        resolution = high_normal_map.shape[0] if high_normal_map is not None else DEFAULT_RESOLUTION
    resolution = int(resolution)
    if resolution <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}")
    _check_mesh("high", high_v, high_f, high_uv, high_n, high_t)
    _check_mesh("low", low_v, low_f, low_uv, low_n, low_t)

    device = _device()
    low_mesh = _mesh_tensors(low_v, low_f, low_n, low_t, device)
    high_mesh = _mesh_tensors(high_v, high_f, high_n, high_t, device)
    low_frame, _, _, _ = _rasterize_frame(low_mesh, low_f, low_uv, resolution)
    if low_frame is None:
        LOGGER.warning("rebake_normal_map_uv_space: low mesh covers no texel at %d", resolution)
        return np.tile(FLAT_TANGENT_NORMAL, (resolution, resolution, 1))

    texel_yx = torch.nonzero(low_frame.mask, as_tuple=False)
    n_l, t_l, b_l = low_frame.normal, low_frame.tangent, low_frame.bitangent
    diag = max(_bbox_diagonal(high_v, low_v), 1e-6)

    # High-poly frame and world normal at the *same* texel as each low texel.
    hi_face_idx, hi_bary, hi_cov = comfy.rasterize(high_f, high_uv, resolution)
    hi_faces_sel, hi_bary_sel = hi_face_idx[low_frame.mask], hi_bary[low_frame.mask]
    hfaces, hverts, hnormals, htangents = high_mesh
    p_h = _interp_at(hverts, hfaces, hi_faces_sel, hi_bary_sel)
    n_h, t_h, b_h = _frame_at(hnormals, htangents, hfaces, hi_faces_sel, hi_bary_sel)
    if high_normal_map is not None:
        nh_uv = _compose_world_normal(
            n_h, t_h, b_h, _texel_values(high_normal_map, texel_yx, resolution)
        )
    else:
        nh_uv = n_h

    # Only trust the direct texel when coverage, position and normal cone all agree.
    close_enough = torch.linalg.norm(p_h - low_frame.position, dim=-1) <= DIRECT_POSITION_TOLERANCE * diag
    direct_ok = hi_cov[low_frame.mask] & close_enough & ((nh_uv * n_l).sum(-1) >= DIRECT_CONE_DOT)
    nh = torch.where(direct_ok[:, None], nh_uv, n_l)

    # Boundary texels: query the 3D high-poly with a tight cage; otherwise stay flat (N = N_l).
    fallback = torch.nonzero(~direct_ok, as_tuple=True)[0]
    if fallback.numel() and int(high_f.shape[0]) > 0:
        tri_h = hverts[hfaces]
        cage = DEFAULT_CAGE_RATIO * diag
        hit_point, hit_face = _closest_hits(
            low_frame.position[fallback], n_l[fallback], tri_h, comfy.build_bvh(tri_h), cage
        )
        valid = hit_face >= 0
        if bool(valid.any()):
            slot, face_ids = fallback[valid], hit_face[valid]
            bary = comfy.barycentric(hit_point[valid], tri_h[face_ids])
            n_hit, t_hit, b_hit = _frame_at(hnormals, htangents, hfaces, face_ids, bary)
            if high_normal_map is not None:
                uv_hit = _interp_at(_tensor(high_uv, np.float32, device), hfaces, face_ids, bary)
                n_hit = _compose_world_normal(
                    n_hit, t_hit, b_hit, _sample_texture(high_normal_map, uv_hit)
                )
            alpha = _cone_alpha(n_hit, n_l[slot], RAY_CONE_DOT)
            blended = alpha[:, None] * n_hit + (1.0 - alpha)[:, None] * n_l[slot]
            nh[slot] = torch.nn.functional.normalize(blended, dim=-1, eps=1e-6)

    return _finish_map(texel_yx, _encode_tangent_normal(nh, n_l, t_l, b_l), resolution,
                       FLAT_TANGENT_NORMAL)


def bake_pbr_maps_3d_transfer(
    high_v: np.ndarray, high_f: np.ndarray, high_uv: Optional[np.ndarray], high_n: np.ndarray,
    high_t: Optional[np.ndarray], high_textures: dict,
    low_v: np.ndarray, low_f: np.ndarray, low_uv: np.ndarray, low_n: np.ndarray,
    low_t: np.ndarray, resolution: int = DEFAULT_RESOLUTION,
    cage_ratio: float = DEFAULT_CAGE_RATIO,
) -> dict:
    """Transfer every high-poly channel onto a low-poly that carries a new UV atlas.

    Texels are matched with a tight cage (`cage_ratio * bbox diagonal`): primary ray along
    `-N_l`, then a reverse ray along `+N_l`, then a closest-point snap.  `"normal_map"` is
    re-composed in the low-poly's TBN frames, every other entry of `high_textures` is bilinearly
    sampled at the interpolated high-poly UV, and all maps are JFA-dilated.  Returns
    `{channel: (resolution, resolution, 3) f32 in [0,1]}`.
    """
    comfy, torch = _comfy(), _comfy().torch
    resolution = int(resolution)
    if resolution <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}")
    if cage_ratio <= 0:
        raise ValueError(f"cage_ratio must be positive, got {cage_ratio}")
    _check_mesh("high", high_v, high_f, high_uv, high_n, high_t)
    _check_mesh("low", low_v, low_f, low_uv, low_n, low_t)

    device = _device()
    channels = [name for name in high_textures if name != NORMAL_CHANNEL]

    def neutral_tiles():
        tiles = {name: np.tile(np.full(3, CHANNEL_NEUTRAL.get(name, 0.0), np.float32),
                               (resolution, resolution, 1)) for name in channels}
        tiles[NORMAL_CHANNEL] = np.tile(FLAT_TANGENT_NORMAL, (resolution, resolution, 1))
        return tiles

    low_mesh = _mesh_tensors(low_v, low_f, low_n, low_t, device)
    high_mesh = _mesh_tensors(high_v, high_f, high_n, high_t, device)
    low_frame, _, _, _ = _rasterize_frame(low_mesh, low_f, low_uv, resolution)
    if low_frame is None or int(high_f.shape[0]) == 0:
        LOGGER.warning("bake_pbr_maps_3d_transfer: no low coverage (%s) or empty high mesh (%d)",
                       low_frame is None, int(high_f.shape[0]))
        return neutral_tiles()

    hfaces, hverts, hnormals, htangents = high_mesh
    tri_h = hverts[hfaces]
    diag = max(_bbox_diagonal(high_v, low_v), 1e-6)
    hit_point, hit_face = _closest_hits(
        low_frame.position, low_frame.normal, tri_h, comfy.build_bvh(tri_h),
        float(cage_ratio) * diag,
    )
    valid = hit_face >= 0
    if not bool(valid.any()):
        LOGGER.warning("bake_pbr_maps_3d_transfer: no texel matched the high mesh")
        return neutral_tiles()

    write_yx = torch.nonzero(low_frame.mask, as_tuple=False)[valid]
    face_ids = hit_face[valid]
    bary = comfy.barycentric(hit_point[valid], tri_h[face_ids])
    n_l, t_l, b_l = low_frame.normal[valid], low_frame.tangent[valid], low_frame.bitangent[valid]

    uv_h = None
    if high_uv is not None:
        uv_h = _interp_at(_tensor(high_uv, np.float32, device), hfaces, face_ids, bary)

    result: dict = {}
    for name in channels:
        default = float(CHANNEL_NEUTRAL.get(name, 0.0))
        if uv_h is None:
            LOGGER.warning("bake_pbr_maps_3d_transfer: high_uv is None, %r keeps its neutral value",
                           name)
            result[name] = np.tile(np.full(3, default, np.float32), (resolution, resolution, 1))
            continue
        sampled = _fit_rgb(_sample_texture(high_textures[name], uv_h), default)
        result[name] = _finish_map(write_yx, sampled, resolution, np.full(3, default, np.float32))

    n_h, t_h, b_h = _frame_at(hnormals, htangents, hfaces, face_ids, bary)
    normal_source = high_textures.get(NORMAL_CHANNEL)
    if normal_source is not None and high_t is not None and uv_h is not None:
        nh = _compose_world_normal(n_h, t_h, b_h, _sample_texture(normal_source, uv_h))
    else:
        if normal_source is not None and high_t is None:
            LOGGER.warning("bake_pbr_maps_3d_transfer: no high tangents, normal map falls back to "
                           "the smooth high-poly normal")
        nh = n_h
    # Thin-shell guard: never let a back-facing hit invert front-surface shading.
    alpha = _cone_alpha(nh, n_l, TRANSFER_CONE_DOT)
    nh = torch.nn.functional.normalize(
        alpha[:, None] * nh + (1.0 - alpha)[:, None] * n_l, dim=-1, eps=1e-6
    )
    result[NORMAL_CHANNEL] = _finish_map(
        write_yx, _encode_tangent_normal(nh, n_l, t_l, b_l), resolution, FLAT_TANGENT_NORMAL
    )
    return result
