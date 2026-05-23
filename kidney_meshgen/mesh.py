from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter
from skimage import measure

from .config import GeneratorConfig
from .graph import AnatomyGraph, Primitive


@dataclass
class MeshBuildResult:
    mesh_outer: trimesh.Trimesh
    mesh_inner: trimesh.Trimesh
    vertex_labels: np.ndarray
    face_labels: np.ndarray
    bounds_min_mm: Tuple[float, float, float]
    bounds_max_mm: Tuple[float, float, float]
    grid_resolution: int
    grid_shape: Tuple[int, int, int]
    spacing_mm: Tuple[float, float, float]

    def stats(self) -> Dict:
        return {
            "vertices": int(len(self.mesh_outer.vertices)),
            "faces": int(len(self.mesh_outer.faces)),
            "bounds_min_mm": [float(v) for v in self.bounds_min_mm],
            "bounds_max_mm": [float(v) for v in self.bounds_max_mm],
            "grid_resolution": int(self.grid_resolution),
            "grid_shape": [int(v) for v in self.grid_shape],
            "spacing_mm": [float(v) for v in self.spacing_mm],
            "surface_area_mm2": float(self.mesh_outer.area),
            "volume_mm3": float(abs(self.mesh_outer.volume)) if self.mesh_outer.is_watertight else None,
            "is_watertight": bool(self.mesh_outer.is_watertight),
        }


def _primitive_bounds(primitive: Primitive) -> Tuple[np.ndarray, np.ndarray]:
    if primitive.kind == "ellipsoid":
        c = np.asarray(primitive.center, dtype=float)
        r = np.asarray(primitive.radii, dtype=float)
        return c - r, c + r
    if primitive.kind == "tapered_capsule":
        p0 = np.asarray(primitive.p0, dtype=float)
        p1 = np.asarray(primitive.p1, dtype=float)
        r = max(float(primitive.r0), float(primitive.r1))
        mn = np.minimum(p0, p1) - r
        mx = np.maximum(p0, p1) + r
        return mn, mx
    raise ValueError(f"Unsupported primitive kind: {primitive.kind}")


def compute_bounds(graph: AnatomyGraph, padding_mm: float) -> Tuple[np.ndarray, np.ndarray]:
    mins: List[np.ndarray] = []
    maxs: List[np.ndarray] = []
    for p in graph.primitives:
        mn, mx = _primitive_bounds(p)
        mins.append(mn)
        maxs.append(mx)
    bounds_min = np.min(np.vstack(mins), axis=0) - padding_mm
    bounds_max = np.max(np.vstack(maxs), axis=0) + padding_mm
    return bounds_min, bounds_max


def primitive_sdf(points: np.ndarray, primitive: Primitive) -> np.ndarray:
    """Approximate signed distance to a primitive. Negative means inside the lumen volume."""
    if primitive.kind == "ellipsoid":
        c = np.asarray(primitive.center, dtype=float)
        r = np.asarray(primitive.radii, dtype=float)
        q = (points - c) / np.maximum(r, 1e-6)
        return (np.linalg.norm(q, axis=1) - 1.0) * float(np.min(r))

    if primitive.kind == "tapered_capsule":
        p0 = np.asarray(primitive.p0, dtype=float)
        p1 = np.asarray(primitive.p1, dtype=float)
        r0 = float(primitive.r0)
        r1 = float(primitive.r1)
        v = p1 - p0
        vv = float(np.dot(v, v))
        if vv < 1e-8:
            return np.linalg.norm(points - p0[None, :], axis=1) - max(r0, r1)
        t = np.clip(((points - p0[None, :]) @ v) / vv, 0.0, 1.0)
        closest = p0[None, :] + t[:, None] * v[None, :]
        radius = r0 + t * (r1 - r0)
        return np.linalg.norm(points - closest, axis=1) - radius

    raise ValueError(f"Unsupported primitive kind: {primitive.kind}")


def label_vertices(vertices: np.ndarray, graph: AnatomyGraph, batch_size: int = 200_000) -> np.ndarray:
    labels = np.zeros(len(vertices), dtype=np.int32)
    for start in range(0, len(vertices), batch_size):
        stop = min(start + batch_size, len(vertices))
        pts = vertices[start:stop]
        best = np.full(len(pts), np.inf, dtype=np.float32)
        best_label = np.zeros(len(pts), dtype=np.int32)
        for prim in graph.primitives:
            d = primitive_sdf(pts, prim).astype(np.float32)
            mask = d < best
            if np.any(mask):
                best[mask] = d[mask]
                best_label[mask] = int(prim.label_id)
        labels[start:stop] = best_label
    return labels


def labels_for_faces(faces: np.ndarray, vertex_labels: np.ndarray) -> np.ndarray:
    face_labels = np.empty(len(faces), dtype=np.int32)
    for i, tri in enumerate(faces):
        labs = vertex_labels[tri]
        vals, counts = np.unique(labs, return_counts=True)
        face_labels[i] = int(vals[np.argmax(counts)])
    return face_labels


def _add_surface_noise(field: np.ndarray, rng: np.random.Generator, amplitude_mm: float) -> np.ndarray:
    if amplitude_mm <= 0:
        return field
    noise = rng.normal(0.0, 1.0, size=field.shape).astype(np.float32)
    sigma = tuple(max(1.0, s / 56.0) for s in field.shape)
    noise = gaussian_filter(noise, sigma=sigma)
    std = float(np.std(noise))
    if std > 1e-6:
        noise = noise / std
    return field + amplitude_mm * noise.astype(np.float32)


def _grid_shape_from_bounds(bounds_min: np.ndarray, bounds_max: np.ndarray, config: GeneratorConfig) -> Tuple[int, int, int]:
    extents = np.maximum(bounds_max - bounds_min, 1e-6)
    longest = float(np.max(extents))
    target = max(int(config.grid_resolution), int(config.min_grid_axis), 48)
    spacing = longest / max(target - 1, 1)
    dims = np.ceil(extents / max(spacing, 1e-6)).astype(int) + 1
    dims = np.maximum(dims, int(config.min_grid_axis))
    return int(dims[0]), int(dims[1]), int(dims[2])


def _primitive_crop_indices(values: np.ndarray, lo: float, hi: float, margin: float) -> Tuple[int, int]:
    start = int(np.searchsorted(values, lo - margin, side="left"))
    stop = int(np.searchsorted(values, hi + margin, side="right"))
    start = max(0, start)
    stop = min(len(values), stop)
    return start, stop



def _open_ureter_start_if_requested(mesh: trimesh.Trimesh, graph: AnatomyGraph, config: GeneratorConfig) -> trimesh.Trimesh:
    """Remove the proximal ureter cap so the case has an actual entry lumen.

    The implicit capsule representation naturally creates a rounded cap at the
    first ureter node. For endoscope navigation, an open tube is more useful:
    scope trajectories can begin slightly outside or inside the ureter without
    seeing a closed wall behind them.
    """
    if not bool(getattr(config, "open_ureter_start", False)):
        return mesh
    nodes = graph.node_map()
    if "ureter_start" not in nodes:
        return mesh
    start = np.asarray(nodes["ureter_start"].position_mm, dtype=float)
    look_node = "ureter_mid_01" if "ureter_mid_01" in nodes else "upj" if "upj" in nodes else None
    if look_node is None:
        return mesh
    target = np.asarray(nodes[look_node].position_mm, dtype=float)
    forward = target - start
    n = float(np.linalg.norm(forward))
    if n < 1e-8:
        return mesh
    forward = forward / n
    cut_offset = float(getattr(config, "open_ureter_start_offset_mm", 1.2))
    centroids = mesh.triangles_center
    signed = (centroids - start[None, :]) @ forward
    keep = signed >= cut_offset
    removed = int(np.count_nonzero(~keep))
    if removed <= 0:
        return mesh
    # Safety: the entry cap is a small part of the mesh. If the plane would
    # remove too much, keep the mesh unchanged rather than destroying a case.
    if removed > max(50, int(0.25 * len(mesh.faces))):
        return mesh
    opened = mesh.copy()
    opened.update_faces(keep)
    opened.remove_unreferenced_vertices()
    opened.fix_normals()
    return opened


def build_lumen_mesh(graph: AnatomyGraph, config: GeneratorConfig) -> MeshBuildResult:
    rng = np.random.default_rng(config.seed + 991)

    bounds_min, bounds_max = compute_bounds(graph, config.padding_mm)
    nx, ny, nz = _grid_shape_from_bounds(bounds_min, bounds_max, config)

    xs = np.linspace(bounds_min[0], bounds_max[0], nx, dtype=np.float32)
    ys = np.linspace(bounds_min[1], bounds_max[1], ny, dtype=np.float32)
    zs = np.linspace(bounds_min[2], bounds_max[2], nz, dtype=np.float32)
    spacing = (
        float((bounds_max[0] - bounds_min[0]) / max(nx - 1, 1)),
        float((bounds_max[1] - bounds_min[1]) / max(ny - 1, 1)),
        float((bounds_max[2] - bounds_min[2]) / max(nz - 1, 1)),
    )

    extents = bounds_max - bounds_min
    far_value = float(np.linalg.norm(extents) + 10.0)
    field = np.full((nx, ny, nz), far_value, dtype=np.float32)

    max_spacing = max(spacing)
    for primitive in graph.primitives:
        mn, mx = _primitive_bounds(primitive)
        if primitive.kind == "ellipsoid":
            margin = float(np.max(np.asarray(primitive.radii, dtype=float))) + 3.5 * max_spacing
        else:
            margin = max(float(primitive.r0), float(primitive.r1)) + 3.5 * max_spacing

        ix0, ix1 = _primitive_crop_indices(xs, mn[0], mx[0], margin)
        iy0, iy1 = _primitive_crop_indices(ys, mn[1], mx[1], margin)
        iz0, iz1 = _primitive_crop_indices(zs, mn[2], mx[2], margin)
        if ix1 <= ix0 or iy1 <= iy0 or iz1 <= iz0:
            continue

        X, Y, Z = np.meshgrid(xs[ix0:ix1], ys[iy0:iy1], zs[iz0:iz1], indexing="ij")
        pts = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=1)
        sdf = primitive_sdf(pts, primitive).astype(np.float32).reshape((ix1 - ix0, iy1 - iy0, iz1 - iz0))
        field[ix0:ix1, iy0:iy1, iz0:iz1] = np.minimum(field[ix0:ix1, iy0:iy1, iz0:iz1], sdf)

    field = _add_surface_noise(field, rng, float(config.surface_noise_mm))

    if not (np.nanmin(field) <= 0.0 <= np.nanmax(field)):
        raise RuntimeError("SDF did not cross zero; try increasing padding or checking anatomy parameters.")

    verts, faces, normals, values = measure.marching_cubes(
        field,
        level=0.0,
        spacing=spacing,
        gradient_direction="descent",
        allow_degenerate=False,
    )
    verts = verts + bounds_min[None, :]

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)

    # Remove tiny fragments if noise creates them.
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        areas = np.array([c.area for c in components])
        keep = areas >= max(areas.max() * 0.015, 4.0)
        if np.any(keep):
            mesh = trimesh.util.concatenate([c for c, k in zip(components, keep) if k])

    if config.mesh_smoothing_iterations and config.mesh_smoothing_iterations > 0:
        try:
            trimesh.smoothing.filter_taubin(
                mesh,
                lamb=0.42,
                nu=-0.52,
                iterations=int(config.mesh_smoothing_iterations),
            )
        except Exception:
            pass

    if config.mesh_decimation_faces:
        target = int(config.mesh_decimation_faces)
        if 0 < target < len(mesh.faces):
            try:
                mesh = mesh.simplify_quadric_decimation(target)
            except Exception:
                pass

    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    mesh = _open_ureter_start_if_requested(mesh, graph, config)

    vlabels = label_vertices(np.asarray(mesh.vertices), graph)
    flabels = labels_for_faces(np.asarray(mesh.faces), vlabels)

    inner_faces = np.asarray(mesh.faces)[:, ::-1].copy()
    inner = trimesh.Trimesh(vertices=np.asarray(mesh.vertices).copy(), faces=inner_faces, process=False)
    inner.fix_normals()

    return MeshBuildResult(
        mesh_outer=mesh,
        mesh_inner=inner,
        vertex_labels=vlabels,
        face_labels=flabels,
        bounds_min_mm=tuple(float(v) for v in bounds_min),
        bounds_max_mm=tuple(float(v) for v in bounds_max),
        grid_resolution=max(nx, ny, nz),
        grid_shape=(nx, ny, nz),
        spacing_mm=spacing,
    )
