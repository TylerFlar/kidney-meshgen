from __future__ import annotations

import csv
import json
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import trimesh

from .config import GeneratorConfig
from .graph import AnatomyGraph, Edge, edge_length
from .mesh import MeshBuildResult
from .stones import StoneInfo


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def label_color(label_id: int) -> Tuple[int, int, int, int]:
    if label_id == 1:
        return (232, 129, 160, 255)  # renal pelvis
    if label_id == 2:
        return (120, 180, 245, 255)  # ureter
    if label_id == 3:
        return (180, 140, 245, 255)  # UPJ
    rng = np.random.default_rng(label_id * 1009 + 17)
    return (int(rng.integers(80, 245)), int(rng.integers(80, 245)), int(rng.integers(80, 245)), 255)


def _safe_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_").replace(":", "_")


def _normalize(v: np.ndarray, fallback: Tuple[float, float, float] = (0.0, 0.0, 1.0)) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return np.asarray(fallback, dtype=float)
    return v / n


def _quat_xyzw_from_forward_up(forward: np.ndarray, up: np.ndarray) -> List[float]:
    """Quaternion from forward/up vectors, returned as [x, y, z, w]."""
    f = _normalize(forward, (0.0, 0.0, 1.0))
    u = _normalize(up, (0.0, 1.0, 0.0))
    r = np.cross(u, f)
    if np.linalg.norm(r) < 1e-8:
        r = np.cross(np.array([1.0, 0.0, 0.0]), f)
    r = _normalize(r, (1.0, 0.0, 0.0))
    u2 = _normalize(np.cross(f, r), (0.0, 1.0, 0.0))
    m = np.array(
        [[r[0], u2[0], f[0]], [r[1], u2[1], f[1]], [r[2], u2[2], f[2]]],
        dtype=float,
    )
    tr = float(np.trace(m))
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qx, qy, qz, qw], dtype=float)
    q = _normalize(q, (0.0, 0.0, 0.0, 1.0))
    return [float(v) for v in q]


def _source_mm_to_unity_m(v: np.ndarray) -> List[float]:
    # Source: x=lateral, y=AP, z=cranio-caudal. Unity y-up mapping: x=x, y=z, z=y.
    return [float(v[0] * 0.001), float(v[2] * 0.001), float(v[1] * 0.001)]


def _source_vec_to_unity(v: np.ndarray) -> np.ndarray:
    return np.asarray([v[0], v[2], v[1]], dtype=float)


def scope_start_pose(graph: AnatomyGraph) -> Dict:
    nodes = graph.node_map()
    if "ureter_start" in nodes:
        start_node = "ureter_start"
        look_node = "ureter_mid_01" if "ureter_mid_01" in nodes else "upj"
    elif "upj" in nodes:
        start_node = "upj"
        look_node = "pelvis_center"
    else:
        start_node = graph.nodes[0].id
        look_node = graph.nodes[min(1, len(graph.nodes) - 1)].id

    p = np.asarray(nodes[start_node].position_mm, dtype=float)
    target = np.asarray(nodes[look_node].position_mm, dtype=float)
    forward = _normalize(target - p, (0.0, 0.0, 1.0))
    up_source = np.array([0.0, 1.0, 0.0])
    q_source = _quat_xyzw_from_forward_up(forward, up_source)
    f_unity = _source_vec_to_unity(forward)
    up_unity = _source_vec_to_unity(up_source)
    q_unity = _quat_xyzw_from_forward_up(f_unity, up_unity)

    return {
        "node": start_node,
        "position_mm": [float(v) for v in p],
        "look_at_node": look_node,
        "look_at_position_mm": [float(v) for v in target],
        "forward_vector": [float(v) for v in forward],
        "up_vector": [0.0, 1.0, 0.0],
        "rotation_quaternion_xyzw": q_source,
        "unity_y_up": {
            "position_m": _source_mm_to_unity_m(p),
            "look_at_position_m": _source_mm_to_unity_m(target),
            "forward_vector": [float(v) for v in _normalize(f_unity)],
            "up_vector": [float(v) for v in _normalize(up_unity, (0.0, 1.0, 0.0))],
            "rotation_quaternion_xyzw": q_unity,
        },
    }


# Backward compatibility for earlier internal callers.
_scope_start_pose = scope_start_pose


def apply_vertex_label_colors(mesh: trimesh.Trimesh, vertex_labels: np.ndarray) -> trimesh.Trimesh:
    mesh = mesh.copy()
    colors = np.array([label_color(int(label)) for label in vertex_labels], dtype=np.uint8)
    mesh.visual.vertex_colors = colors
    return mesh


def export_meshes(
    out_dir: Path,
    mesh_result: MeshBuildResult,
    graph: AnatomyGraph,
    config: GeneratorConfig,
    stone_meshes: List[trimesh.Trimesh],
) -> Dict[str, str]:
    _ensure_dir(out_dir)
    files: Dict[str, str] = {}

    outer_colored = apply_vertex_label_colors(mesh_result.mesh_outer, mesh_result.vertex_labels)
    inner_colored = apply_vertex_label_colors(mesh_result.mesh_inner, mesh_result.vertex_labels)

    if config.export_obj:
        outer_path = out_dir / "lumen_outer.obj"
        inner_path = out_dir / "lumen_inner.obj"
        outer_colored.export(outer_path)
        inner_colored.export(inner_path)
        files["lumen_outer_obj"] = str(outer_path.name)
        files["lumen_inner_obj"] = str(inner_path.name)

    if config.export_ply:
        outer_ply = out_dir / "lumen_outer.ply"
        inner_ply = out_dir / "lumen_inner.ply"
        outer_colored.export(outer_ply)
        inner_colored.export(inner_ply)
        files["lumen_outer_ply"] = str(outer_ply.name)
        files["lumen_inner_ply"] = str(inner_ply.name)

    if config.export_glb:
        outer_glb = out_dir / "lumen_outer.glb"
        inner_glb = out_dir / "lumen_inner.glb"
        outer_colored.export(outer_glb)
        inner_colored.export(inner_glb)
        files["lumen_outer_glb"] = str(outer_glb.name)
        files["lumen_inner_glb"] = str(inner_glb.name)

    # Per-region submeshes for semantic materials and calyx coverage scoring.
    regions_dir = out_dir / "regions"
    _ensure_dir(regions_dir)
    labels = mesh_result.face_labels
    for label_id in sorted(np.unique(labels)):
        if int(label_id) == 0:
            continue
        face_indices = np.where(labels == label_id)[0]
        if len(face_indices) == 0:
            continue
        region_name = graph.labels.get(int(label_id), f"label_{int(label_id)}")
        safe = _safe_name(region_name)
        submesh = mesh_result.mesh_outer.submesh([face_indices], append=True, repair=False)
        if submesh is not None and len(submesh.faces) > 0:
            rel = f"regions/{int(label_id):03d}_{safe}.obj"
            submesh.export(out_dir / rel)

    # Stones remain separate to support segmentation, visibility, and tool/laser tasks.
    stones_dir = out_dir / "stones"
    _ensure_dir(stones_dir)
    if stone_meshes:
        for idx, mesh in enumerate(stone_meshes):
            mesh.export(stones_dir / f"stone_{idx:03d}.obj")
        combined = trimesh.util.concatenate(stone_meshes)
        combined.visual.vertex_colors = np.tile(np.array([[212, 192, 145, 255]], dtype=np.uint8), (len(combined.vertices), 1))
        combined.export(stones_dir / "stones.obj")
        files["stones_obj"] = "stones/stones.obj"
        if config.export_glb:
            combined.export(stones_dir / "stones.glb")
            files["stones_glb"] = "stones/stones.glb"

    if config.export_glb:
        scene = trimesh.Scene()
        scene.add_geometry(inner_colored, geom_name="lumen_inner")
        if stone_meshes:
            for idx, mesh in enumerate(stone_meshes):
                scene.add_geometry(mesh, geom_name=f"stone_{idx:03d}")
        scene_path = out_dir / "scene_inside_with_stones.glb"
        scene.export(scene_path)
        files["scene_inside_with_stones_glb"] = scene_path.name

    return files


def write_label_tables(out_dir: Path, mesh_result: MeshBuildResult, graph: AnatomyGraph) -> Dict[str, str]:
    files: Dict[str, str] = {}
    label_map_path = out_dir / "labels.json"
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in graph.labels.items()}, f, indent=2)
    files["labels_json"] = label_map_path.name

    face_path = out_dir / "face_labels.csv"
    with open(face_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["face_index", "label_id", "label"])
        for i, label_id in enumerate(mesh_result.face_labels):
            writer.writerow([i, int(label_id), graph.labels.get(int(label_id), "unknown")])
    files["face_labels_csv"] = face_path.name

    vertex_path = out_dir / "vertex_labels.csv"
    with open(vertex_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["vertex_index", "label_id", "label"])
        for i, label_id in enumerate(mesh_result.vertex_labels):
            writer.writerow([i, int(label_id), graph.labels.get(int(label_id), "unknown")])
    files["vertex_labels_csv"] = vertex_path.name
    return files


def write_centerline(out_dir: Path, graph: AnatomyGraph) -> Dict[str, str]:
    nodes = graph.node_map()
    data = graph.to_dict()
    for edge in data["edges"]:
        e = next(x for x in graph.edges if x.id == edge["id"])
        edge["length_mm"] = edge_length(e, nodes)
    path = out_dir / "centerline_graph.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return {"centerline_graph_json": path.name}


def _sample_edge(edge: Edge, graph: AnatomyGraph, spacing_mm: float, reverse: bool = False) -> List[Dict]:
    nodes = graph.node_map()
    p0 = np.asarray(nodes[edge.source].position_mm, dtype=float)
    p1 = np.asarray(nodes[edge.target].position_mm, dtype=float)
    if reverse:
        p0, p1 = p1, p0
        r0, r1 = float(edge.radius1_mm), float(edge.radius0_mm)
        source, target = edge.target, edge.source
    else:
        r0, r1 = float(edge.radius0_mm), float(edge.radius1_mm)
        source, target = edge.source, edge.target
    length = float(np.linalg.norm(p1 - p0))
    steps = max(2, int(np.ceil(length / max(spacing_mm, 1e-3))) + 1)
    forward = _normalize(p1 - p0, (0.0, 0.0, 1.0))
    points = []
    for t in np.linspace(0.0, 1.0, steps):
        pos = p0 * (1 - t) + p1 * t
        radius = r0 * (1 - t) + r1 * t
        points.append(
            {
                "edge_id": edge.id,
                "source": source,
                "target": target,
                "region": edge.region,
                "kind": edge.kind,
                "t": float(t),
                "position_mm": [float(v) for v in pos],
                "forward_vector": [float(v) for v in forward],
                "radius_mm": float(radius),
            }
        )
    return points


def _find_node_route(graph: AnatomyGraph, start: str, goal: str) -> List[Tuple[Edge, bool]]:
    adj: Dict[str, List[Tuple[str, Edge, bool]]] = {}
    for e in graph.edges:
        adj.setdefault(e.source, []).append((e.target, e, False))
        adj.setdefault(e.target, []).append((e.source, e, True))
    q = deque([start])
    parent: Dict[str, Tuple[str, Edge, bool] | None] = {start: None}
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        for nxt, edge, rev in adj.get(cur, []):
            if nxt in parent:
                continue
            parent[nxt] = (cur, edge, rev)
            q.append(nxt)
    if goal not in parent:
        return []
    route: List[Tuple[Edge, bool]] = []
    cur = goal
    while parent[cur] is not None:
        prev, edge, rev = parent[cur]
        route.append((edge, rev))
        cur = prev
    route.reverse()
    return route


def write_camera_paths(out_dir: Path, graph: AnatomyGraph, config: GeneratorConfig) -> Dict[str, str]:
    spacing = float(config.camera_path_spacing_mm)
    all_waypoints: List[Dict] = []
    for e in graph.edges:
        samples = _sample_edge(e, graph, spacing_mm=spacing, reverse=False)
        if all_waypoints and samples:
            samples = samples[1:]
        all_waypoints.extend(samples)

    nodes = graph.node_map()
    start = "ureter_start" if "ureter_start" in nodes else "upj" if "upj" in nodes else graph.nodes[0].id
    routes = []
    for target in graph.calyx_targets:
        goal = target.get("cup_node")
        if goal not in nodes:
            continue
        route_edges = _find_node_route(graph, start, goal)
        route_points: List[Dict] = []
        length = 0.0
        for edge, rev in route_edges:
            length += edge_length(edge, nodes)
            samples = _sample_edge(edge, graph, spacing_mm=spacing, reverse=rev)
            if route_points and samples:
                samples = samples[1:]
            route_points.extend(samples)
        routes.append(
            {
                "route_id": f"route_to_{target['id']}",
                "target_calyx_id": target["id"],
                "target_region": target["region"],
                "target_cup_node": goal,
                "length_mm": float(length),
                "waypoints": route_points,
            }
        )

    data = {
        "schema": "kidney_meshgen_camera_paths_v0.7",
        "units": "millimeters",
        "spacing_mm": spacing,
        "start_node": start,
        "scope_start_pose": scope_start_pose(graph),
        "all_centerline_waypoints": all_waypoints,
        "routes": routes,
    }
    path = out_dir / "camera_paths.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return {"camera_paths_json": path.name}


def write_manifest(
    out_dir: Path,
    config: GeneratorConfig,
    graph: AnatomyGraph,
    mesh_result: MeshBuildResult,
    stone_infos: List[StoneInfo],
    file_map: Dict[str, str],
) -> Dict[str, str]:
    manifest = {
        "schema": "kidney_meshgen_mesh_manifest_v0.7",
        "anatomy_id": config.anatomy_id,
        "seed": config.seed,
        "units": "millimeters",
        "unity_scale_to_meters": 0.001,
        "coordinate_system": config.coordinate_system,
        "pelvis_type": graph.pelvis_type,
        "anatomy_metadata": graph.metadata,
        "files": file_map,
        "mesh_stats": mesh_result.stats(),
        "labels": {str(k): v for k, v in graph.labels.items()},
        "calyx_targets": graph.calyx_targets,
        "stones": [s.to_dict() for s in stone_infos],
        "scope_start_pose": scope_start_pose(graph),
        "simulator": {
            "primary_engine": "Unity real-time simulator",
            "visual_mesh": file_map.get("lumen_inner_glb") or file_map.get("lumen_inner_obj"),
            "collision_mesh": file_map.get("lumen_collision_proxy_obj") or file_map.get("lumen_outer_obj"),
            "sdf_grid": file_map.get("lumen_sdf_grid_npz"),
            "task_scene": file_map.get("runtime_scene_json"),
            "waypoints": file_map.get("navigation_waypoints_json"),
            "coverage_points": file_map.get("coverage_points_csv"),
        },
        "recommended_observation_space": {
            "rgb": "uint8 HxWx3 endoscopic render",
            "depth_mm": "float32 HxW renderer depth, optional in Unity",
            "semantic_mask": "uint16 HxW label IDs, optional",
            "scope_pose_mm": "6-DoF camera/scope-tip pose",
            "scope_state": "insertion, roll, deflection, tool state",
            "centerline_progress": "nearest centerline edge and edge t",
            "stone_visibility": "per-stone visible pixel count",
            "clearance_mm": "from collision proxy or approximate SDF grid",
        },
        "recommended_action_space": {
            "advance_retract_mm_s": [-30.0, 30.0],
            "roll_deg_s": [-120.0, 120.0],
            "primary_deflect_deg_s": [-90.0, 90.0],
            "secondary_deflect_deg_s": [-60.0, 60.0],
            "laser_fiber_mm_s": [-20.0, 20.0],
            "basket_mm_s": [-20.0, 20.0],
            "basket_open_close": [0.0, 1.0],
        },
        "recommended_metrics": [
            "entry_to_pelvis_success",
            "stone_finding_success",
            "time_to_first_stone_s",
            "time_to_all_stones_s",
            "calyx_coverage_fraction",
            "surface_coverage_fraction",
            "wall_contacts_per_minute",
            "lost_view_events",
            "trajectory_length_mm",
            "minimum_wall_clearance_mm",
            "control_oscillation_score",
            "lower_pole_access_success",
            "lower_pole_revisit_success",
        ],
        "renderer_notes": {
            "inside_mesh": "Use lumen_inner.* for an endoscopic camera inside the collecting system, or use a two-sided material.",
            "entry_tube": "The proximal ureter start is open by default so the simulator starts in a true tube rather than against a rounded cap.",
            "anatomy_profile": "Takazawa profile names calyces as T/U/M/L/B with anterior/posterior pairs where applicable.",
            "papilla_fornix": "Minor calyx cups include subtractive papilla solids so the fornix is cup-like rather than a smooth bulb.",
            "lower_pole": "The lower pole includes an explicit major access trunk plus minor-calyx fan-out for a cleaner lower-pole navigation task.",
            "stone_meshes": "Stones are separate meshes to simplify visibility, detection, segmentation, and task scoring.",
            "region_meshes": "Per-label OBJ files are provided under regions/ for semantic materials or calyx coverage.",
            "camera_paths": "camera_paths.json gives centerline samples and routes from the ureter start to every calyx target.",
            "camera_rate_hz": config.default_camera_rate_hz,
            "control_rate_hz": config.default_control_rate_hz,
            "scope_fov_degrees": config.scope_fov_degrees,
        },
        "config": config.to_dict(),
    }
    path = out_dir / "scene_manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return {"scene_manifest_json": path.name}
