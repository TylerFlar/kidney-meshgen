from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import trimesh
from skimage import measure

from .config import GeneratorConfig
from .graph import AnatomyGraph
from .sdf import primitive_bounds, primitive_sdf


@dataclass
class MeshBuildResult:
    mesh_outer: trimesh.Trimesh
    mesh_inner: trimesh.Trimesh
    mesh_collision_outer: trimesh.Trimesh
    vertex_labels: np.ndarray
    face_labels: np.ndarray
    bounds_min_mm: Tuple[float, float, float]
    bounds_max_mm: Tuple[float, float, float]
    grid_resolution: int
    grid_shape: Tuple[int, int, int]
    spacing_mm: Tuple[float, float, float]
    visual_displacement_stats: Dict[str, float]

    def stats(self) -> Dict:
        return {
            "vertices": int(len(self.mesh_outer.vertices)),
            "faces": int(len(self.mesh_outer.faces)),
            "collision_vertices": int(len(self.mesh_collision_outer.vertices)),
            "collision_faces": int(len(self.mesh_collision_outer.faces)),
            "bounds_min_mm": [float(v) for v in self.bounds_min_mm],
            "bounds_max_mm": [float(v) for v in self.bounds_max_mm],
            "grid_resolution": int(self.grid_resolution),
            "grid_shape": [int(v) for v in self.grid_shape],
            "spacing_mm": [float(v) for v in self.spacing_mm],
            "surface_area_mm2": float(self.mesh_outer.area),
            "volume_mm3": float(abs(self.mesh_outer.volume)) if self.mesh_outer.is_watertight else None,
            "is_watertight": bool(self.mesh_outer.is_watertight),
            "collision_surface_area_mm2": float(self.mesh_collision_outer.area),
            "collision_volume_mm3": (
                float(abs(self.mesh_collision_outer.volume)) if self.mesh_collision_outer.is_watertight else None
            ),
            "collision_is_watertight": bool(self.mesh_collision_outer.is_watertight),
            "visual_displacement": {k: float(v) for k, v in self.visual_displacement_stats.items()},
        }


def compute_bounds(graph: AnatomyGraph, padding_mm: float) -> Tuple[np.ndarray, np.ndarray]:
    mins: List[np.ndarray] = []
    maxs: List[np.ndarray] = []
    for p in graph.primitives:
        mn, mx = primitive_bounds(p)
        mins.append(mn)
        maxs.append(mx)
    bounds_min = np.min(np.vstack(mins), axis=0) - padding_mm
    bounds_max = np.max(np.vstack(maxs), axis=0) + padding_mm
    return bounds_min, bounds_max


def _normalize(v: np.ndarray, fallback: Tuple[float, float, float] = (1.0, 0.0, 0.0)) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return np.asarray(fallback, dtype=float)
    return v / n


def label_vertices(vertices: np.ndarray, graph: AnatomyGraph, batch_size: int = 200_000) -> np.ndarray:
    labels = np.zeros(len(vertices), dtype=np.int32)
    for start in range(0, len(vertices), batch_size):
        stop = min(start + batch_size, len(vertices))
        pts = vertices[start:stop]
        best = np.full(len(pts), np.inf, dtype=np.float32)
        best_label = np.zeros(len(pts), dtype=np.int32)
        for prim in graph.primitives:
            if getattr(prim, "operation", "union") == "subtract":
                continue
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


def _harmonic_noise(points: np.ndarray, rng: np.random.Generator, wavelengths: Tuple[float, float], octaves: int = 7) -> np.ndarray:
    if len(points) == 0:
        return np.asarray([], dtype=float)
    lo = max(float(wavelengths[0]), 0.25)
    hi = max(float(wavelengths[1]), lo)
    value = np.zeros(len(points), dtype=float)
    amp = 1.0
    amp_sum = 0.0
    for _ in range(max(int(octaves), 1)):
        direction = _normalize(rng.normal(0.0, 1.0, size=3), (1.0, 0.0, 0.0))
        wavelength = float(rng.uniform(lo, hi))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        value += amp * np.sin((points @ direction) * (2.0 * np.pi / wavelength) + phase)
        amp_sum += amp
        amp *= 0.58
    if amp_sum > 1e-8:
        value /= amp_sum
    std = float(np.std(value))
    if std > 1e-8:
        value /= std
    return np.clip(value, -2.5, 2.5) / 2.5


def _min_distance_to_points(vertices: np.ndarray, points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.full(len(vertices), np.inf, dtype=float)
    diff = vertices[:, None, :] - points[None, :, :]
    return np.sqrt(np.min(np.sum(diff * diff, axis=2), axis=1))


def _pelvis_fold_weight(vertices: np.ndarray, graph: AnatomyGraph, band_mm: float) -> Tuple[np.ndarray, np.ndarray]:
    pelvis = next((p for p in graph.primitives if p.id == "pelvis_ellipsoid" and p.kind == "ellipsoid"), None)
    if pelvis is None:
        return np.zeros(len(vertices), dtype=float), np.zeros(len(vertices), dtype=float)
    sdf = np.abs(primitive_sdf(vertices, pelvis))
    band = max(float(band_mm), 1e-3)
    weight = np.exp(-0.5 * (sdf / band) ** 2)
    center = np.asarray(pelvis.center, dtype=float)
    radii = np.asarray(pelvis.radii, dtype=float)
    radial_mm = np.linalg.norm((vertices - center[None, :]) / np.maximum(radii[None, :], 1e-6), axis=1) * float(np.mean(radii))
    return weight, radial_mm


def _apply_visual_surface_displacement(
    mesh: trimesh.Trimesh,
    graph: AnatomyGraph,
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> Tuple[trimesh.Trimesh, Dict[str, float]]:
    noise_amp = max(float(config.visual_surface_noise_mm), 0.0)
    fold_amp = max(float(config.visual_fold_amplitude_mm), 0.0)
    max_disp = max(float(config.visual_max_displacement_mm), 0.0)
    if (noise_amp <= 0.0 and fold_amp <= 0.0) or max_disp <= 0.0 or len(mesh.vertices) == 0:
        return mesh, {
            "max_abs_mm": 0.0,
            "rms_mm": 0.0,
            "visual_surface_noise_mm": noise_amp,
            "fold_amplitude_mm": fold_amp,
        }

    visual = mesh.copy()
    vertices = np.asarray(visual.vertices, dtype=float)
    normals = np.asarray(visual.vertex_normals, dtype=float)
    norm = np.linalg.norm(normals, axis=1)
    bad = norm < 1e-8
    if np.any(bad):
        normals[bad] = np.array([1.0, 0.0, 0.0], dtype=float)
        norm[bad] = 1.0
    normals = normals / norm[:, None]

    fine = noise_amp * _harmonic_noise(vertices, rng, (1.1, 3.1), octaves=7)
    fold = np.zeros(len(vertices), dtype=float)
    band = max(float(config.visual_fold_band_mm), 1e-3)
    wavelength_span = tuple(config.visual_fold_wavelength_mm)
    wavelength = float(rng.uniform(max(0.5, wavelength_span[0]), max(0.5, wavelength_span[1])))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))

    if fold_amp > 0.0:
        pelvis_weight, pelvis_radial = _pelvis_fold_weight(vertices, graph, band)
        pelvis_signal = np.sin((pelvis_radial / wavelength) * 2.0 * np.pi + phase)
        nodes = graph.node_map()
        neck_points = np.asarray(
            [node.position_mm for node in nodes.values() if str(node.id).endswith("_neck")],
            dtype=float,
        )
        if neck_points.ndim != 2:
            neck_points = np.empty((0, 3), dtype=float)
        neck_distance = _min_distance_to_points(vertices, neck_points)
        neck_weight = np.exp(-0.5 * (neck_distance / band) ** 2)
        neck_signal = np.sin((neck_distance / wavelength) * 2.0 * np.pi + phase * 0.73)
        fold_weight = np.maximum(0.55 * pelvis_weight, neck_weight)
        fold_signal = 0.55 * pelvis_weight * pelvis_signal + neck_weight * neck_signal
        # Bias folds slightly inward from the lumen surface while keeping a
        # signed component so they read as low ridges rather than dents.
        inward_bias = -0.45 * fold_weight * (0.5 + 0.5 * np.sin(fold_signal * np.pi))
        fold = fold_amp * (0.55 * fold_signal + inward_bias)

    displacement = np.clip(fine + fold, -max_disp, max_disp)
    visual.vertices = vertices + normals * displacement[:, None]
    visual.remove_unreferenced_vertices()
    visual.fix_normals()
    return visual, {
        "max_abs_mm": float(np.max(np.abs(displacement))) if len(displacement) else 0.0,
        "rms_mm": float(np.sqrt(np.mean(displacement**2))) if len(displacement) else 0.0,
        "visual_surface_noise_mm": float(noise_amp),
        "fold_amplitude_mm": float(fold_amp),
        "fold_band_mm": float(band),
        "max_allowed_mm": float(max_disp),
    }


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
    union_primitives = [p for p in graph.primitives if getattr(p, "operation", "union") != "subtract"]
    subtract_primitives = [p for p in graph.primitives if getattr(p, "operation", "union") == "subtract"]

    for primitive in union_primitives:
        mn, mx = primitive_bounds(primitive)
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

    for primitive in subtract_primitives:
        mn, mx = primitive_bounds(primitive)
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
        field[ix0:ix1, iy0:iy1, iz0:iz1] = np.maximum(field[ix0:ix1, iy0:iy1, iz0:iz1], -sdf)

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

    # Remove tiny disconnected fragments from implicit blending edge cases.
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

    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    collision_outer = _open_ureter_start_if_requested(mesh, graph, config)
    collision_outer.remove_unreferenced_vertices()
    collision_outer.fix_normals()

    visual_outer = collision_outer.copy()
    if config.mesh_decimation_faces:
        target = int(config.mesh_decimation_faces)
        if 0 < target < len(visual_outer.faces):
            try:
                visual_outer = visual_outer.simplify_quadric_decimation(target)
            except Exception:
                pass

    visual_outer, displacement_stats = _apply_visual_surface_displacement(visual_outer, graph, config, rng)

    vlabels = label_vertices(np.asarray(visual_outer.vertices), graph)
    flabels = labels_for_faces(np.asarray(visual_outer.faces), vlabels)

    inner_faces = np.asarray(visual_outer.faces)[:, ::-1].copy()
    inner = trimesh.Trimesh(vertices=np.asarray(visual_outer.vertices).copy(), faces=inner_faces, process=False)
    inner.fix_normals()
    return MeshBuildResult(
        mesh_outer=visual_outer,
        mesh_inner=inner,
        mesh_collision_outer=collision_outer,
        vertex_labels=vlabels,
        face_labels=flabels,
        bounds_min_mm=tuple(float(v) for v in bounds_min),
        bounds_max_mm=tuple(float(v) for v in bounds_max),
        grid_resolution=max(nx, ny, nz),
        grid_shape=(nx, ny, nz),
        spacing_mm=spacing,
        visual_displacement_stats=displacement_stats,
    )
