from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import trimesh

from .config import GeneratorConfig
from .exporters import scope_start_pose
from .graph import AnatomyGraph, Edge, edge_length
from .mesh import MeshBuildResult, compute_bounds
from .sdf import primitive_bounds, primitive_sdf
from .stones import StoneInfo


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _build_adjacency(graph: AnatomyGraph) -> Dict[str, List[Tuple[str, Edge]]]:
    adj: Dict[str, List[Tuple[str, Edge]]] = {}
    for e in graph.edges:
        adj.setdefault(e.source, []).append((e.target, e))
        adj.setdefault(e.target, []).append((e.source, e))
    return adj


def _shortest_node_path(graph: AnatomyGraph, start: str, goal: str) -> Tuple[List[str], List[Edge]]:
    nodes = graph.node_map()
    adj = _build_adjacency(graph)
    dist = {start: 0.0}
    prev: Dict[str, Tuple[str, Edge]] = {}
    visited = set()
    while True:
        current = None
        best = float("inf")
        for nid, val in dist.items():
            if nid not in visited and val < best:
                current = nid
                best = val
        if current is None:
            break
        if current == goal:
            break
        visited.add(current)
        for nxt, edge in adj.get(current, []):
            if nxt in visited:
                continue
            cand = dist[current] + edge_length(edge, nodes)
            if cand < dist.get(nxt, float("inf")):
                dist[nxt] = cand
                prev[nxt] = (current, edge)
    if goal not in dist:
        return [start], []
    node_path = [goal]
    edge_path: List[Edge] = []
    cur = goal
    while cur != start:
        p, edge = prev[cur]
        node_path.append(p)
        edge_path.append(edge)
        cur = p
    node_path.reverse()
    edge_path.reverse()
    return node_path, edge_path


def _edge_between(graph: AnatomyGraph, a: str, b: str) -> Edge | None:
    for e in graph.edges:
        if (e.source == a and e.target == b) or (e.source == b and e.target == a):
            return e
    return None


def _interpolate_route(graph: AnatomyGraph, node_path: List[str], samples_per_edge: int, scope_radius_mm: float) -> List[Dict]:
    nodes = graph.node_map()
    frames: List[Dict] = []
    samples_per_edge = max(int(samples_per_edge), 1)
    if len(node_path) == 1:
        p = np.asarray(nodes[node_path[0]].position_mm, dtype=float)
        return [{"position_mm": [float(v) for v in p], "radius_mm": None, "estimated_clearance_mm": None, "edge_id": None, "edge_t": 0.0}]

    for a, b in zip(node_path[:-1], node_path[1:]):
        edge = _edge_between(graph, a, b)
        if edge is None:
            continue
        p0 = np.asarray(nodes[a].position_mm, dtype=float)
        p1 = np.asarray(nodes[b].position_mm, dtype=float)
        if edge.source == a:
            r0, r1 = float(edge.radius0_mm), float(edge.radius1_mm)
        else:
            r0, r1 = float(edge.radius1_mm), float(edge.radius0_mm)
        for i in range(samples_per_edge):
            if frames and i == 0:
                continue
            t = i / float(samples_per_edge)
            pos = p0 * (1 - t) + p1 * t
            radius = r0 * (1 - t) + r1 * t
            frames.append(
                {
                    "position_mm": [float(v) for v in pos],
                    "radius_mm": float(radius),
                    "estimated_clearance_mm": float(radius - scope_radius_mm),
                    "edge_id": edge.id,
                    "edge_t": float(t if edge.source == a else 1.0 - t),
                    "region": edge.region,
                    "kind": edge.kind,
                }
            )
    # Add exact final node.
    last = node_path[-1]
    if frames:
        prev = np.asarray(frames[-1]["position_mm"], dtype=float)
        p = np.asarray(nodes[last].position_mm, dtype=float)
        if np.linalg.norm(prev - p) > 1e-6:
            edge = _edge_between(graph, node_path[-2], last)
            radius = float(edge.radius1_mm if edge and edge.target == last else edge.radius0_mm if edge else 0.0)
            frames.append(
                {
                    "position_mm": [float(v) for v in p],
                    "radius_mm": radius,
                    "estimated_clearance_mm": float(radius - scope_radius_mm) if radius else None,
                    "edge_id": edge.id if edge else None,
                    "edge_t": 1.0,
                    "region": edge.region if edge else None,
                    "kind": edge.kind if edge else None,
                }
            )
    return frames


def _attach_look_vectors(frames: List[Dict], fov_degrees: float) -> None:
    for i, fr in enumerate(frames):
        pos = np.asarray(fr["position_mm"], dtype=float)
        if i < len(frames) - 1:
            look = np.asarray(frames[i + 1]["position_mm"], dtype=float)
        elif i > 0:
            prev = np.asarray(frames[i - 1]["position_mm"], dtype=float)
            look = pos + (pos - prev)
        else:
            look = pos + np.array([1.0, 0.0, 0.0])
        fwd = look - pos
        norm = float(np.linalg.norm(fwd))
        if norm > 1e-8:
            fwd = fwd / norm
        fr["look_at_mm"] = [float(v) for v in look]
        fr["forward"] = [float(v) for v in fwd]
        fr["up"] = [0.0, 1.0, 0.0]
        fr["fov_degrees"] = float(fov_degrees)


def write_waypoint_files(out_dir: Path, graph: AnatomyGraph, config: GeneratorConfig) -> Dict[str, str]:
    files: Dict[str, str] = {}
    way_dir = out_dir / "waypoints"
    _ensure_dir(way_dir)

    nodes = graph.node_map()
    start = "ureter_start" if "ureter_start" in nodes else ("upj" if "upj" in nodes else "pelvis_center")
    paths: List[Dict] = []
    frames: List[Dict] = []
    scope_radius_mm = float(config.scope_outer_diameter_mm) / 2.0

    def add_path(path_id: str, task: str, target_node: str, target_info: Dict | None = None) -> None:
        node_path, edge_path = _shortest_node_path(graph, start, target_node)
        pts = _interpolate_route(graph, node_path, int(config.waypoint_samples_per_edge), scope_radius_mm)
        if not pts:
            return
        _attach_look_vectors(pts, float(config.scope_fov_degrees))
        start_idx = len(frames)
        for i, fr in enumerate(pts):
            fr["path_id"] = path_id
            fr["frame_index"] = len(frames)
            fr["path_local_index"] = i
            frames.append(fr)
        length = sum(edge_length(e, nodes) for e in edge_path)
        min_clearance = min((fr.get("estimated_clearance_mm") for fr in pts if fr.get("estimated_clearance_mm") is not None), default=None)
        paths.append(
            {
                "id": path_id,
                "task": task,
                "target_node": target_node,
                "target": target_info or {},
                "node_path": node_path,
                "edge_path": [e.id for e in edge_path],
                "length_mm": float(length),
                "minimum_centerline_radius_mm": min((fr.get("radius_mm") for fr in pts if fr.get("radius_mm") is not None), default=None),
                "minimum_estimated_scope_clearance_mm": min_clearance,
                "frame_start": start_idx,
                "frame_count": len(pts),
            }
        )

    add_path("entry_to_pelvis", "entry_to_pelvis", "pelvis_center", {"region": "renal_pelvis"})
    lower_access_nodes = sorted([node_id for node_id in nodes if node_id.startswith("lower_access_attach_")])
    if lower_access_nodes:
        add_path(
            "entry_to_lower_access_trunk",
            "lower_pole_access",
            lower_access_nodes[-1],
            {"region": "lower_pole_access", "attachment_node_count": len(lower_access_nodes)},
        )
    for target in graph.calyx_targets:
        add_path(
            f"entry_to_{target['id']}",
            "calyx_inspection",
            str(target["cup_node"]),
            {"calyx_id": target["id"], "region": target["region"], "center_mm": target["center_mm"]},
        )

    waypoint_data = {
        "schema": "kidney_meshgen_navigation_waypoints_v0.7",
        "units": "millimeters",
        "start_node": start,
        "scope_outer_diameter_mm": float(config.scope_outer_diameter_mm),
        "paths": paths,
        "frames": frames,
    }
    path = way_dir / "navigation_waypoints.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(waypoint_data, f, indent=2)
    files["navigation_waypoints_json"] = "waypoints/navigation_waypoints.json"

    csv_path = way_dir / "navigation_waypoints.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_index",
            "path_id",
            "path_local_index",
            "x_mm",
            "y_mm",
            "z_mm",
            "look_x_mm",
            "look_y_mm",
            "look_z_mm",
            "edge_id",
            "edge_t",
            "radius_mm",
            "estimated_clearance_mm",
        ])
        for fr in frames:
            p = fr["position_mm"]
            look = fr["look_at_mm"]
            writer.writerow([
                fr["frame_index"], fr["path_id"], fr["path_local_index"], *p, *look,
                fr.get("edge_id"), fr.get("edge_t"), fr.get("radius_mm"), fr.get("estimated_clearance_mm"),
            ])
    files["navigation_waypoints_csv"] = "waypoints/navigation_waypoints.csv"
    return files


def write_coverage_points(out_dir: Path, mesh_result: MeshBuildResult, graph: AnatomyGraph, config: GeneratorConfig) -> Dict[str, str]:
    files: Dict[str, str] = {}
    n = int(config.coverage_sample_count)
    if n <= 0:
        return files
    cov_dir = out_dir / "coverage"
    _ensure_dir(cov_dir)
    rng = np.random.default_rng(config.seed + 5050)
    try:
        points, face_indices = trimesh.sample.sample_surface(mesh_result.mesh_outer, n, seed=rng)
    except TypeError:
        points, face_indices = trimesh.sample.sample_surface(mesh_result.mesh_outer, n)
    labels = mesh_result.face_labels[np.asarray(face_indices, dtype=int)]
    path = cov_dir / "coverage_points.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["point_index", "x_mm", "y_mm", "z_mm", "label_id", "label"])
        for i, (pt, label_id) in enumerate(zip(points, labels)):
            writer.writerow([i, float(pt[0]), float(pt[1]), float(pt[2]), int(label_id), graph.labels.get(int(label_id), "unknown")])
    files["coverage_points_csv"] = "coverage/coverage_points.csv"
    return files


def write_collision_proxy(out_dir: Path, mesh_result: MeshBuildResult, config: GeneratorConfig) -> Dict[str, str]:
    files: Dict[str, str] = {}
    if not bool(config.export_collision_proxy):
        return files
    coll_dir = out_dir / "collision"
    _ensure_dir(coll_dir)
    mesh = mesh_result.mesh_collision_outer.copy()
    target = config.collision_proxy_faces
    if target and 0 < int(target) < len(mesh.faces):
        try:
            mesh = mesh.simplify_quadric_decimation(int(target))
        except Exception:
            pass
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    obj_path = coll_dir / "lumen_collision_proxy.obj"
    mesh.export(obj_path)
    files["lumen_collision_proxy_obj"] = "collision/lumen_collision_proxy.obj"
    return files


def _grid_shape_from_bounds(bounds_min: np.ndarray, bounds_max: np.ndarray, target_long_axis: int, min_axis: int) -> Tuple[int, int, int]:
    extents = np.maximum(bounds_max - bounds_min, 1e-6)
    longest = float(np.max(extents))
    target = max(int(target_long_axis), int(min_axis), 24)
    spacing = longest / max(target - 1, 1)
    dims = np.ceil(extents / max(spacing, 1e-6)).astype(int) + 1
    dims = np.maximum(dims, int(min_axis))
    return int(dims[0]), int(dims[1]), int(dims[2])


def _crop_indices(values: np.ndarray, lo: float, hi: float, margin: float) -> Tuple[int, int]:
    start = int(np.searchsorted(values, lo - margin, side="left"))
    stop = int(np.searchsorted(values, hi + margin, side="right"))
    return max(0, start), min(len(values), stop)


def write_sdf_grid(out_dir: Path, graph: AnatomyGraph, config: GeneratorConfig) -> Dict[str, str]:
    files: Dict[str, str] = {}
    if not bool(config.export_sdf_grid):
        return files
    coll_dir = out_dir / "collision"
    _ensure_dir(coll_dir)
    bounds_min, bounds_max = compute_bounds(graph, float(config.padding_mm))
    nx, ny, nz = _grid_shape_from_bounds(bounds_min, bounds_max, int(config.sdf_grid_resolution), int(config.sdf_min_grid_axis))
    xs = np.linspace(bounds_min[0], bounds_max[0], nx, dtype=np.float32)
    ys = np.linspace(bounds_min[1], bounds_max[1], ny, dtype=np.float32)
    zs = np.linspace(bounds_min[2], bounds_max[2], nz, dtype=np.float32)
    spacing = np.array([
        float((bounds_max[0] - bounds_min[0]) / max(nx - 1, 1)),
        float((bounds_max[1] - bounds_min[1]) / max(ny - 1, 1)),
        float((bounds_max[2] - bounds_min[2]) / max(nz - 1, 1)),
    ], dtype=np.float32)
    far_value = float(np.linalg.norm(bounds_max - bounds_min) + 10.0)
    field = np.full((nx, ny, nz), far_value, dtype=np.float32)
    max_spacing = float(np.max(spacing))
    union_primitives = [p for p in graph.primitives if getattr(p, "operation", "union") != "subtract"]
    subtract_primitives = [p for p in graph.primitives if getattr(p, "operation", "union") == "subtract"]
    for primitive in union_primitives:
        mn, mx = primitive_bounds(primitive)
        if primitive.kind == "ellipsoid":
            margin = float(np.max(np.asarray(primitive.radii, dtype=float))) + 3.5 * max_spacing
        else:
            margin = max(float(primitive.r0), float(primitive.r1)) + 3.5 * max_spacing
        ix0, ix1 = _crop_indices(xs, mn[0], mx[0], margin)
        iy0, iy1 = _crop_indices(ys, mn[1], mx[1], margin)
        iz0, iz1 = _crop_indices(zs, mn[2], mx[2], margin)
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
        ix0, ix1 = _crop_indices(xs, mn[0], mx[0], margin)
        iy0, iy1 = _crop_indices(ys, mn[1], mx[1], margin)
        iz0, iz1 = _crop_indices(zs, mn[2], mx[2], margin)
        if ix1 <= ix0 or iy1 <= iy0 or iz1 <= iz0:
            continue
        X, Y, Z = np.meshgrid(xs[ix0:ix1], ys[iy0:iy1], zs[iz0:iz1], indexing="ij")
        pts = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=1)
        sdf = primitive_sdf(pts, primitive).astype(np.float32).reshape((ix1 - ix0, iy1 - iy0, iz1 - iz0))
        field[ix0:ix1, iy0:iy1, iz0:iz1] = np.maximum(field[ix0:ix1, iy0:iy1, iz0:iz1], -sdf)
    # Store float16 to keep the package small. Values are approximate and intended
    # for fast clearance/collision prechecks, not clinical measurements.
    npz_path = coll_dir / "lumen_sdf_grid.npz"
    np.savez_compressed(
        npz_path,
        sdf_mm=field.astype(np.float16),
        bounds_min_mm=bounds_min.astype(np.float32),
        bounds_max_mm=bounds_max.astype(np.float32),
        spacing_mm=spacing,
        grid_shape=np.asarray([nx, ny, nz], dtype=np.int32),
    )
    files["lumen_sdf_grid_npz"] = "collision/lumen_sdf_grid.npz"
    meta = {
        "schema": "kidney_meshgen_sdf_grid_v0.7",
        "units": "millimeters",
        "file": "collision/lumen_sdf_grid.npz",
        "field_name": "sdf_mm",
        "convention": "negative_inside_lumen_positive_outside",
        "bounds_min_mm": [float(v) for v in bounds_min],
        "bounds_max_mm": [float(v) for v in bounds_max],
        "spacing_mm": [float(v) for v in spacing],
        "grid_shape": [int(nx), int(ny), int(nz)],
        "approximation": "smooth analytic union with subtractive papilla solids; visual mucosal displacement is not included",
    }
    meta_path = coll_dir / "lumen_sdf_grid.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    files["lumen_sdf_grid_json"] = "collision/lumen_sdf_grid.json"
    return files


def _runtime_tasks(graph: AnatomyGraph, stone_infos: List[StoneInfo]) -> List[Dict]:
    tasks = [
        {"id": "entry_to_pelvis", "goal": "navigate from ureter entry tube to renal pelvis", "target_node": "pelvis_center"},
        {"id": "inspect_all_calyces", "goal": "visit all generated calyx targets", "target_calyx_ids": [t["id"] for t in graph.calyx_targets]},
        {"id": "find_all_stones", "goal": "visually localize all stones", "stone_ids": [s.id for s in stone_infos]},
    ]
    lower = [t for t in graph.calyx_targets if t.get("region") == "lower"]
    if lower:
        tasks.append({"id": "lower_pole_access", "goal": "reach and inspect lower-pole calyces", "target_calyx_ids": [t["id"] for t in lower]})
    return tasks


def write_runtime_scene(out_dir: Path, config: GeneratorConfig, graph: AnatomyGraph, stone_infos: List[StoneInfo], file_map: Dict[str, str]) -> Dict[str, str]:
    files: Dict[str, str] = {}
    scene = {
        "schema": "kidney_meshgen_runtime_scene_v0.7",
        "anatomy_id": config.anatomy_id,
        "units": "millimeters",
        "unity_scale_to_meters": 0.001,
        "coordinate_system": config.coordinate_system,
        "start_pose": scope_start_pose(graph),
        "assets": {
            "visual_lumen": file_map.get("lumen_inner_glb") or file_map.get("lumen_inner_obj"),
            "collision_lumen": file_map.get("lumen_collision_proxy_obj") or file_map.get("lumen_outer_obj"),
            "sdf_grid": file_map.get("lumen_sdf_grid_npz"),
            "stones": file_map.get("stones_glb") or file_map.get("stones_obj"),
            "centerline_graph": file_map.get("centerline_graph_json"),
            "waypoints": file_map.get("navigation_waypoints_json"),
            "coverage_points": file_map.get("coverage_points_csv"),
            "labels": file_map.get("labels_json"),
        },
        "scope_model": {
            "model": "kinematic_piecewise_constant_curvature",
            "outer_diameter_mm": float(config.scope_outer_diameter_mm),
            "tip_length_mm": float(config.scope_tip_length_mm),
            "fov_degrees": float(config.scope_fov_degrees),
            "max_deflection_deg": float(config.scope_max_deflection_deg),
            "camera_rate_hz": int(config.default_camera_rate_hz),
            "control_rate_hz": int(config.default_control_rate_hz),
        },
        "action_space": {
            "advance_retract_mm_s": [-30.0, 30.0],
            "roll_deg_s": [-120.0, 120.0],
            "primary_deflect_deg_s": [-90.0, 90.0],
            "secondary_deflect_deg_s": [-60.0, 60.0],
            "laser_fiber_mm_s": [-20.0, 20.0],
            "basket_mm_s": [-20.0, 20.0],
            "basket_open_close": [0.0, 1.0],
        },
        "tasks": _runtime_tasks(graph, stone_infos),
        "future_extension_hooks": [
            "replace kinematic scope with deformation/contact backend",
            "add detailed laser fiber and basket mesh animation",
            "swap Unity materials/shaders for higher-fidelity wet tissue rendering",
        ],
    }
    path = out_dir / "runtime_scene.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(scene, f, indent=2)
    files["runtime_scene_json"] = "runtime_scene.json"

    if bool(config.write_unity_support):
        unity_dir = out_dir / "unity"
        _ensure_dir(unity_dir)
        unity_cfg = {
            "schema": "kidney_meshgen_unity_scene_v0.7",
            "units": "millimeters",
            "unityScaleToMeters": 0.001,
            "visualMesh": scene["assets"]["visual_lumen"],
            "collisionMesh": scene["assets"]["collision_lumen"],
            "stoneMesh": scene["assets"]["stones"],
            "manifest": "scene_manifest.json",
            "runtimeScene": "runtime_scene.json",
            "centerlineGraph": scene["assets"]["centerline_graph"],
            "waypoints": scene["assets"]["waypoints"],
            "coveragePoints": scene["assets"]["coverage_points"],
            "sdfGrid": scene["assets"].get("sdf_grid"),
            "notes": [
                "Use lumen_inner.* or a two-sided material for inside-lumen rendering.",
                "Use the collision proxy or approximate SDF grid for clearance/collision checks.",
                "Use runtime_scene.json as the single runtime descriptor for Unity/Gym bridge setup.",
            ],
        }
        cfg_path = unity_dir / "kidney_meshgen_unity_scene.json"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(unity_cfg, f, indent=2)
        files["unity_scene_json"] = "unity/kidney_meshgen_unity_scene.json"
        helper_dir = Path(__file__).resolve().parent.parent / "unity"
        for helper_name in ["KidneyMeshgenManifest.cs", "KidneyMeshgenSceneLoader.cs"]:
            src = helper_dir / helper_name
            if src.exists():
                dst = unity_dir / helper_name
                dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
                files[f"unity_{helper_name.replace('.cs', '').lower()}_cs"] = f"unity/{helper_name}"
    return files


def write_simulator_exports(
    out_dir: Path,
    config: GeneratorConfig,
    graph: AnatomyGraph,
    mesh_result: MeshBuildResult,
    stone_infos: List[StoneInfo],
    stone_meshes: List[trimesh.Trimesh],
    file_map: Dict[str, str],
) -> Dict[str, str]:
    files: Dict[str, str] = {}
    files.update(write_waypoint_files(out_dir, graph, config))
    file_map.update(files)
    files.update(write_coverage_points(out_dir, mesh_result, graph, config))
    file_map.update(files)
    files.update(write_collision_proxy(out_dir, mesh_result, config))
    file_map.update(files)
    files.update(write_sdf_grid(out_dir, graph, config))
    file_map.update(files)
    files.update(write_runtime_scene(out_dir, config, graph, stone_infos, file_map))
    return files
