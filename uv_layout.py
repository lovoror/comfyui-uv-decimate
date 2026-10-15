"""2D UV Layout & UV Utilization Inspector for ComfyUI 3D meshes.

Computes exact 2D UV space utilization (coverage %), UV overlap %, connected UV
island count, UV seam edge ratio, and 3D-to-2D texel stretch distortion, and
renders a high-contrast 2D UV layout diagnostic image with optional HUD banner.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np

from .contracts import MeshPayload, UVTopologyMetrics, analyze_uv_topology
from . import mesh_convert


BACKGROUND_MODES = (
    "texture_and_islands",
    "uv_islands",
    "texture",
    "distortion_stretch",
    "dark_grid",
)


@dataclass(frozen=True)
class UVLayoutReport:
    """Quantitative 2D UV layout and packing utilization metrics."""

    total_faces: int
    welded_3d_verts: int
    uv_split_verts: int
    n_islands: int
    seam_vertices: int
    seam_edge_ratio: float
    seam_lock_floor: int
    utilization_pct: float
    analytic_area_pct: float
    overlap_pct: float
    out_of_bounds_pct: float
    stretch_p50: float
    stretch_p95: float

    def format_text(self) -> str:
        uv_ratio = self.uv_split_verts / max(self.welded_3d_verts, 1)
        lines = [
            f"UV Utilization (Coverage): {self.utilization_pct:.2f}% (Analytic Sum: {self.analytic_area_pct:.2f}%)",
            f"UV Overlap (Multi-Hit):    {self.overlap_pct:.2f}%",
            f"Out-of-Bounds [0,1]^2:     {self.out_of_bounds_pct:.2f}%",
            f"Connected UV Islands:      {self.n_islands:,}",
            f"Seam Edges / Lock Floor:   {self.seam_edge_ratio * 100.0:.1f}% ({self.seam_lock_floor:,} 3D verts locked)",
            f"Mesh Topology:             {self.total_faces:,} faces | {self.welded_3d_verts:,} 3D verts | {self.uv_split_verts:,} UV verts ({uv_ratio:.2f}x)",
            f"Texel Density Deviation:   p50={self.stretch_p50:.2f}x | p95={self.stretch_p95:.2f}x",
        ]
        return "\n".join(lines)


def _compute_uv_island_labels(faces: np.ndarray, uvs: np.ndarray) -> Tuple[np.ndarray, int, np.ndarray]:
    """Return ``(face_island_id, n_islands, is_seam_edge_mask)`` for each face's 3 edges."""
    num_faces = int(len(faces))
    if num_faces == 0:
        return np.zeros(0, dtype=np.int32), 0, np.zeros((0, 3), dtype=bool)

    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
    except Exception:
        return np.zeros(num_faces, dtype=np.int32), 1, np.zeros((num_faces, 3), dtype=bool)

    uv_q = np.round(np.asarray(uvs, dtype=np.float64) * 1e6).astype(np.int64)
    _, uv_inv = np.unique(uv_q, axis=0, return_inverse=True)
    uv_inv = uv_inv.astype(np.int64)
    f_uv = uv_inv[faces]
    n_uv = int(uv_inv.max()) + 1 if uv_inv.size else 1

    e0 = f_uv[:, [0, 1]]
    e1 = f_uv[:, [1, 2]]
    e2 = f_uv[:, [2, 0]]
    rows = np.concatenate([e0[:, 0], e1[:, 0], e2[:, 0], e0[:, 1], e1[:, 1], e2[:, 1]])
    cols = np.concatenate([e0[:, 1], e1[:, 1], e2[:, 1], e0[:, 0], e1[:, 0], e2[:, 0]])
    graph = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n_uv, n_uv))
    n_islands, vert_labels = connected_components(graph, directed=False)
    face_island = vert_labels[f_uv[:, 0]].astype(np.int32)

    # Identify UV boundary edges (edges in UV space that are not shared by 2 UV triangles)
    all_uv_edges = np.concatenate([e0, e1, e2], axis=0)
    uv_edges_sorted = np.sort(all_uv_edges, axis=1)
    _, uv_edge_inv, uv_edge_counts = np.unique(
        uv_edges_sorted, axis=0, return_inverse=True, return_counts=True
    )
    is_boundary_flat = uv_edge_counts[uv_edge_inv] == 1
    is_seam_edge = is_boundary_flat.reshape(3, num_faces).T
    return face_island, int(n_islands), is_seam_edge


def _rasterize_uv_coverage_and_overlap(
    faces: np.ndarray,
    uvs: np.ndarray,
    resolution: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rasterize UV triangles into ``(R, R)`` returning ``(face_id_map, covered_mask, overlap_mask)``.

    Uses both forward (0..F-1) and reverse (F-1..0) face-index rasterization so any texel
    covered by two or more distinct UV triangles is detected with 100% precision.
    """
    R = int(resolution)
    num_faces = int(len(faces))
    if num_faces == 0 or uvs is None or len(uvs) == 0:
        empty_i = np.full((R, R), -1, dtype=np.int32)
        empty_b = np.zeros((R, R), dtype=bool)
        return empty_i, empty_b, empty_b

    try:
        import torch
        from comfy_extras.nodes_mesh_postprocess import _rasterize_uv_barycentric

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        uv_t = torch.from_numpy(np.ascontiguousarray(uvs, dtype=np.float32)).to(device)
        f_fwd = torch.from_numpy(np.ascontiguousarray(faces, dtype=np.int64)).to(device)
        valid_fwd, fidx_fwd, _ = _rasterize_uv_barycentric(uv_t, f_fwd, R, R)

        f_rev = torch.flip(f_fwd, dims=[0])
        valid_rev, fidx_rev_raw, _ = _rasterize_uv_barycentric(uv_t, f_rev, R, R)
        fidx_rev = (num_faces - 1) - fidx_rev_raw

        covered = valid_fwd.cpu().numpy()
        face_map = torch.where(valid_fwd, fidx_fwd, torch.full_like(fidx_fwd, -1)).cpu().numpy().astype(np.int32)
        overlap = (valid_fwd & valid_rev & (fidx_fwd != fidx_rev)).cpu().numpy()
        return face_map, covered, overlap
    except Exception:
        return _rasterize_uv_cpu_fallback(faces, uvs, R)


def _rasterize_uv_cpu_fallback(
    faces: np.ndarray,
    uvs: np.ndarray,
    R: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized CPU scanline/bounding-box UV triangle rasterizer fallback."""
    face_fwd = np.full((R, R), -1, dtype=np.int32)
    face_rev = np.full((R, R), -1, dtype=np.int32)
    tri_uv = uvs[faces]  # (F, 3, 2)
    px = tri_uv[:, :, 0] * R - 0.5
    py = tri_uv[:, :, 1] * R - 0.5

    x0, y0 = px[:, 0], py[:, 0]
    x1, y1 = px[:, 1], py[:, 1]
    x2, y2 = px[:, 2], py[:, 2]

    xmin = np.clip(np.floor(np.minimum(np.minimum(x0, x1), x2)).astype(np.int32), 0, R - 1)
    xmax = np.clip(np.ceil(np.maximum(np.maximum(x0, x1), x2)).astype(np.int32), 0, R - 1)
    ymin = np.clip(np.floor(np.minimum(np.minimum(y0, y1), y2)).astype(np.int32), 0, R - 1)
    ymax = np.clip(np.ceil(np.maximum(np.maximum(y0, y1), y2)).astype(np.int32), 0, R - 1)

    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    valid_tri = np.abs(denom) > 1e-12

    for f in np.flatnonzero(valid_tri):
        x_lo, x_hi = xmin[f], xmax[f]
        y_lo, y_hi = ymin[f], ymax[f]
        if x_lo > x_hi or y_lo > y_hi:
            continue
        gx, gy = np.meshgrid(np.arange(x_lo, x_hi + 1), np.arange(y_lo, y_hi + 1))
        inv_d = 1.0 / denom[f]
        w0 = ((y1[f] - y2[f]) * (gx - x2[f]) + (x2[f] - x1[f]) * (gy - y2[f])) * inv_d
        w1 = ((y2[f] - y0[f]) * (gx - x2[f]) + (x0[f] - x2[f]) * (gy - y2[f])) * inv_d
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-5) & (w1 >= -1e-5) & (w2 >= -1e-5)
        if not np.any(inside):
            continue
        sub_fwd = face_fwd[y_lo : y_hi + 1, x_lo : x_hi + 1]
        sub_rev = face_rev[y_lo : y_hi + 1, x_lo : x_hi + 1]
        first_hit = inside & (sub_fwd < 0)
        sub_fwd[first_hit] = f
        sub_rev[inside] = f

    covered = face_fwd >= 0
    overlap = covered & (face_rev >= 0) & (face_fwd != face_rev)
    return face_fwd, covered, overlap


def _draw_edges_into_canvas(
    canvas: np.ndarray,
    uv_a: np.ndarray,
    uv_b: np.ndarray,
    color: np.ndarray,
    alpha: float,
    thickness: int = 1,
) -> None:
    """Vectorized multi-sample line drawer for UV wireframe and seam edges on ``(R, R, 3)``."""
    if len(uv_a) == 0 or alpha <= 0.0:
        return
    R = canvas.shape[0]
    x0 = uv_a[:, 0] * (R - 1)
    y0 = uv_a[:, 1] * (R - 1)
    x1 = uv_b[:, 0] * (R - 1)
    y1 = uv_b[:, 1] * (R - 1)

    lengths = np.hypot(x1 - x0, y1 - y0)
    max_len = float(lengths.max()) if lengths.size else 0.0
    if max_len < 0.5:
        return

    # Adaptive steps per segment batch to avoid oversampling tiny triangles
    short_mask = lengths <= 6.0
    med_mask = (lengths > 6.0) & (lengths <= 32.0)
    long_mask = lengths > 32.0

    px_list = []
    py_list = []
    for mask, n_steps in ((short_mask, 6), (med_mask, 28), (long_mask, 96)):
        if not np.any(mask):
            continue
        t = np.linspace(0.0, 1.0, n_steps, dtype=np.float32)[None, :]
        xs = x0[mask, None] + (x1[mask, None] - x0[mask, None]) * t
        ys = y0[mask, None] + (y1[mask, None] - y0[mask, None]) * t
        px_list.append(np.rint(xs).astype(np.int32).ravel())
        py_list.append(np.rint(ys).astype(np.int32).ravel())

    if not px_list:
        return
    px = np.concatenate(px_list)
    py = np.concatenate(py_list)
    in_bounds = (px >= 0) & (px < R) & (py >= 0) & (py < R)
    px = px[in_bounds]
    py = py[in_bounds]

    c = np.asarray(color, dtype=np.float32).reshape(1, 3)
    a = float(np.clip(alpha, 0.0, 1.0))

    offsets = [(0, 0)]
    if thickness >= 2:
        offsets.extend([(1, 0), (-1, 0), (0, 1), (0, -1)])
    for dx, dy in offsets:
        qx = np.clip(px + dx, 0, R - 1)
        qy = np.clip(py + dy, 0, R - 1)
        flat_idx = np.unique(qy * R + qx)
        ry, rx = np.divmod(flat_idx, R)
        canvas[ry, rx] = canvas[ry, rx] * (1.0 - a) + c * a


def _island_palette(n_islands: int) -> np.ndarray:
    """Golden-ratio pastel RGB palette ``(max(n_islands, 1), 3)`` in ``[0, 1]``."""
    count = max(int(n_islands), 1)
    idx = np.arange(count, dtype=np.float32)
    hue = np.mod(idx * 0.61803398875 + 0.11, 1.0)
    sat = 0.58 + 0.18 * np.sin(idx * 1.73)
    val = 0.86 + 0.10 * np.cos(idx * 2.31)

    h6 = hue * 6.0
    i = np.floor(h6).astype(np.int32) % 6
    f = h6 - np.floor(h6)
    p = val * (1.0 - sat)
    q = val * (1.0 - f * sat)
    t = val * (1.0 - (1.0 - f) * sat)

    rgb = np.zeros((count, 3), dtype=np.float32)
    for k, (r, g, b) in enumerate(
        [(val, t, p), (q, val, p), (p, val, t), (p, q, val), (t, p, val), (val, p, q)]
    ):
        m = i == k
        rgb[m, 0] = r[m]
        rgb[m, 1] = g[m]
        rgb[m, 2] = b[m]
    return rgb


def _stretch_colormap(log2_stretch: np.ndarray) -> np.ndarray:
    """Map log2(UV_area_share / 3D_area_share) in [-2, +2] to Blue -> Green -> Red RGB."""
    x = np.clip(log2_stretch / 2.0, -1.0, 1.0)
    rgb = np.zeros((len(x), 3), dtype=np.float32)
    neg = x < 0.0
    pos = ~neg
    # -1 (blue: compressed UVs) -> 0 (emerald green: ideal 1:1 isometric density)
    t_neg = x[neg] + 1.0
    rgb[neg, 0] = 0.12 * (1.0 - t_neg) + 0.18 * t_neg
    rgb[neg, 1] = 0.45 * (1.0 - t_neg) + 0.85 * t_neg
    rgb[neg, 2] = 0.95 * (1.0 - t_neg) + 0.35 * t_neg
    # 0 (emerald green) -> +1 (crimson red: stretched UVs)
    t_pos = x[pos]
    rgb[pos, 0] = 0.18 * (1.0 - t_pos) + 0.96 * t_pos
    rgb[pos, 1] = 0.85 * (1.0 - t_pos) + 0.22 * t_pos
    rgb[pos, 2] = 0.35 * (1.0 - t_pos) + 0.18 * t_pos
    return rgb


def _build_dark_uv_grid(R: int) -> np.ndarray:
    """Create a dark technical UV checkerboard + coordinate grid ``(R, R, 3)``."""
    y, x = np.mgrid[0:R, 0:R]
    tile = ((x * 16 // R) + (y * 16 // R)) % 2
    bg = np.where(tile[:, :, None] == 0, 0.095, 0.125).astype(np.float32)
    bg = np.repeat(bg, 3, axis=2)

    # Major 1/4 grid lines and minor 1/16 grid lines
    for div, col in ((16, 0.16), (4, 0.24)):
        step = max(R // div, 1)
        mask = (x % step == 0) | (y % step == 0)
        bg[mask] = col
    # Outer [0,1] border
    bg[0, :] = bg[-1, :] = bg[:, 0] = bg[:, -1] = (0.35, 0.40, 0.48)
    return bg


def _append_hud_banner(canvas_rgb: np.ndarray, report: UVLayoutReport) -> np.ndarray:
    """Render a clean diagnostic HUD header above ``canvas_rgb`` using PIL."""
    from PIL import Image, ImageDraw, ImageFont

    R = canvas_rgb.shape[0]
    hud_h = max(92, int(R * 0.10))
    body_u8 = np.clip(np.rint(canvas_rgb * 255.0), 0, 255).astype(np.uint8)
    img = Image.new("RGB", (R, R + hud_h), (18, 21, 28))
    img.paste(Image.fromarray(body_u8), (0, hud_h))

    draw = ImageDraw.Draw(img)
    try:
        font_lg = ImageFont.truetype("consola.ttf", max(12, int(R * 0.0175)))
        font_sm = ImageFont.truetype("consola.ttf", max(11, int(R * 0.0135)))
    except Exception:
        font_lg = ImageFont.load_default()
        font_sm = font_lg

    pad = max(12, int(R * 0.016))
    util = report.utilization_pct
    bar_color = (46, 213, 115) if util >= 55.0 else ((255, 177, 66) if util >= 30.0 else (255, 82, 82))

    title = (
        f"UV UTILIZATION: {util:.1f}%  |  ISLANDS: {report.n_islands:,}  |  "
        f"OVERLAP: {report.overlap_pct:.2f}%  |  SEAM EDGES: {report.seam_edge_ratio * 100.0:.1f}%"
    )
    uv_ratio = report.uv_split_verts / max(report.welded_3d_verts, 1)
    sub = (
        f"Faces: {report.total_faces:,} | 3D Verts: {report.welded_3d_verts:,} | "
        f"UV Verts: {report.uv_split_verts:,} ({uv_ratio:.2f}x) | "
        f"Seam Floor: {report.seam_lock_floor:,} | "
        f"Stretch p50/p95: {report.stretch_p50:.2f}x/{report.stretch_p95:.2f}x"
    )
    draw.text((pad, int(hud_h * 0.14)), title, fill=(240, 246, 252), font=font_lg)
    draw.text((pad, int(hud_h * 0.50)), sub, fill=(165, 178, 198), font=font_sm)

    # Utilization progress bar at bottom of HUD
    bar_y0 = int(hud_h * 0.80)
    bar_y1 = hud_h - max(6, int(hud_h * 0.08))
    bar_w = R - 2 * pad
    draw.rectangle([pad, bar_y0, pad + bar_w, bar_y1], fill=(38, 44, 58), outline=(70, 80, 100))
    fill_w = int(round(bar_w * np.clip(util / 100.0, 0.0, 1.0)))
    if fill_w > 0:
        draw.rectangle([pad, bar_y0, pad + fill_w, bar_y1], fill=bar_color)

    return np.asarray(img, dtype=np.float32) / 255.0


def render_uv_layout(
    payload: MeshPayload,
    resolution: int = 1024,
    background_mode: str = "texture_and_islands",
    show_wireframe: bool = True,
    highlight_seams: bool = True,
    wireframe_opacity: float = 0.65,
    show_hud: bool = True,
) -> Tuple[np.ndarray, UVLayoutReport]:
    """Render a 2D UV layout diagnostic image ``(H, W, 3)`` float32 and compute ``UVLayoutReport``."""
    R = int(np.clip(resolution, 256, 4096))
    verts = np.asarray(payload.vertices, dtype=np.float64)
    faces = np.asarray(payload.faces, dtype=np.int64)
    uvs = np.asarray(payload.uvs, dtype=np.float64) if payload.uvs is not None else None

    topo: UVTopologyMetrics = analyze_uv_topology(verts, faces, uvs)
    if uvs is None or len(uvs) == 0 or len(faces) == 0:
        canvas = _build_dark_uv_grid(R)
        empty_report = UVLayoutReport(
            total_faces=int(len(faces)),
            welded_3d_verts=int(topo.welded_vertices),
            uv_split_verts=0,
            n_islands=0,
            seam_vertices=0,
            seam_edge_ratio=0.0,
            seam_lock_floor=0,
            utilization_pct=0.0,
            analytic_area_pct=0.0,
            overlap_pct=0.0,
            out_of_bounds_pct=0.0,
            stretch_p50=1.0,
            stretch_p95=1.0,
        )
        if show_hud:
            canvas = _append_hud_banner(canvas, empty_report)
        return canvas, empty_report

    face_island, n_islands, is_seam_edge = _compute_uv_island_labels(faces, uvs)
    face_map, covered_mask, overlap_mask = _rasterize_uv_coverage_and_overlap(faces, uvs, R)

    # Analytical 2D UV area & 3D-to-2D stretch distortion
    tri_uv = uvs[faces]  # (F, 3, 2)
    du1 = tri_uv[:, 1] - tri_uv[:, 0]
    du2 = tri_uv[:, 2] - tri_uv[:, 0]
    uv_areas = 0.5 * np.abs(du1[:, 0] * du2[:, 1] - du1[:, 1] * du2[:, 0])
    analytic_area_pct = float(np.sum(uv_areas) * 100.0)

    tri_3d = verts[faces]
    e1 = tri_3d[:, 1] - tri_3d[:, 0]
    e2 = tri_3d[:, 2] - tri_3d[:, 0]
    area_3d = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)

    uv_share = uv_areas / max(float(np.sum(uv_areas)), 1e-12)
    area3d_share = area_3d / max(float(np.sum(area_3d)), 1e-12)
    ratio = (uv_share + 1e-12) / (area3d_share + 1e-12)
    log2_stretch = np.log2(np.clip(ratio, 0.0625, 16.0))
    abs_dev = np.maximum(ratio, 1.0 / np.maximum(ratio, 1e-6))
    stretch_p50 = float(np.percentile(abs_dev, 50)) if abs_dev.size else 1.0
    stretch_p95 = float(np.percentile(abs_dev, 95)) if abs_dev.size else 1.0

    oob_mask = (uvs[:, 0] < 0.0) | (uvs[:, 0] > 1.0) | (uvs[:, 1] < 0.0) | (uvs[:, 1] > 1.0)
    oob_pct = float(np.mean(oob_mask) * 100.0) if len(uvs) else 0.0
    utilization_pct = float(np.mean(covered_mask) * 100.0)
    overlap_pct = float(np.mean(overlap_mask) * 100.0)
    seam_edge_ratio = float(np.mean(is_seam_edge)) if is_seam_edge.size else 0.0

    report = UVLayoutReport(
        total_faces=int(len(faces)),
        welded_3d_verts=int(topo.welded_vertices),
        uv_split_verts=int(topo.total_vertices),
        n_islands=int(n_islands),
        seam_vertices=int(topo.uv_seam_vertices),
        seam_edge_ratio=seam_edge_ratio,
        seam_lock_floor=int(topo.min_keep_uv_faces),
        utilization_pct=utilization_pct,
        analytic_area_pct=analytic_area_pct,
        overlap_pct=overlap_pct,
        out_of_bounds_pct=oob_pct,
        stretch_p50=stretch_p50,
        stretch_p95=stretch_p95,
    )

    # Base canvas: dark grid in uncovered UV space
    canvas = _build_dark_uv_grid(R)
    tex_rgb = _extract_resized_texture(payload.textures.get("texture"), R)
    palette = _island_palette(max(n_islands, 1))
    island_rgb = palette[np.clip(face_island[np.maximum(face_map, 0)], 0, len(palette) - 1)]

    if background_mode == "texture" and tex_rgb is not None:
        canvas = np.where(covered_mask[:, :, None], tex_rgb, canvas * 0.75 + tex_rgb * 0.25)
    elif background_mode == "texture_and_islands" and tex_rgb is not None:
        blended = tex_rgb * 0.62 + island_rgb * 0.38
        canvas = np.where(covered_mask[:, :, None], blended, canvas)
    elif background_mode == "distortion_stretch":
        tri_colors = _stretch_colormap(log2_stretch)
        stretch_rgb = tri_colors[np.maximum(face_map, 0)]
        canvas = np.where(covered_mask[:, :, None], stretch_rgb, canvas)
    elif background_mode in ("uv_islands", "texture_and_islands", "texture"):
        canvas = np.where(covered_mask[:, :, None], island_rgb, canvas)

    # Highlight overlapping UV texels in vivid magenta-red
    if np.any(overlap_mask):
        overlap_col = np.array([1.0, 0.18, 0.42], dtype=np.float32)
        canvas[overlap_mask] = canvas[overlap_mask] * 0.35 + overlap_col * 0.65

    # Draw interior UV wireframe edges
    uv_clipped = np.clip(uvs, 0.0, 1.0)
    e_pairs = [(faces[:, 0], faces[:, 1], 0), (faces[:, 1], faces[:, 2], 1), (faces[:, 2], faces[:, 0], 2)]
    if show_wireframe and wireframe_opacity > 0.0:
        all_a = np.concatenate([uv_clipped[a] for a, _, _ in e_pairs], axis=0)
        all_b = np.concatenate([uv_clipped[b] for _, b, _ in e_pairs], axis=0)
        _draw_edges_into_canvas(
            canvas,
            all_a,
            all_b,
            color=np.array([0.92, 0.96, 1.0], dtype=np.float32),
            alpha=float(wireframe_opacity) * 0.48,
            thickness=1,
        )

    # Draw UV island seam / boundary edges on top in high-contrast cyan
    if highlight_seams:
        seam_a = []
        seam_b = []
        for a, b, k in e_pairs:
            m = is_seam_edge[:, k]
            if np.any(m):
                seam_a.append(uv_clipped[a[m]])
                seam_b.append(uv_clipped[b[m]])
        if seam_a:
            _draw_edges_into_canvas(
                canvas,
                np.concatenate(seam_a, axis=0),
                np.concatenate(seam_b, axis=0),
                color=np.array([0.0, 0.92, 1.0], dtype=np.float32),
                alpha=min(1.0, float(wireframe_opacity) * 0.95 + 0.25),
                thickness=max(1, R // 1024),
            )

    if show_hud:
        canvas = _append_hud_banner(canvas, report)
    return np.clip(canvas, 0.0, 1.0).astype(np.float32), report


def _extract_resized_texture(tex_obj, R: int) -> Optional[np.ndarray]:
    """Convert a ComfyUI texture tensor/array into ``(R, R, 3)`` float32 in ``[0, 1]``."""
    if tex_obj is None:
        return None
    try:
        from PIL import Image

        arr = tex_obj.detach().cpu().numpy() if hasattr(tex_obj, "detach") else np.asarray(tex_obj)
        while arr.ndim > 3:
            arr = arr[0]
        if arr.ndim != 3 or arr.shape[2] < 3:
            return None
        rgb = arr[:, :, :3].astype(np.float32)
        if rgb.max() > 1.05:
            rgb = rgb / 255.0
        rgb = np.clip(rgb, 0.0, 1.0)
        if rgb.shape[0] == R and rgb.shape[1] == R:
            return rgb
        pil = Image.fromarray(np.rint(rgb * 255.0).astype(np.uint8)).resize((R, R), Image.BILINEAR)
        return np.asarray(pil, dtype=np.float32) / 255.0
    except Exception:
        return None
