from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .sdf import primitive_sdf


@dataclass
class RenderPathOptions:
    traversal: str = "dfs"
    target_node: Optional[str] = None
    fps: float = 30.0
    speed_mm_s: float = 18.0
    sample_spacing_mm: float = 1.0
    smooth_window_mm: float = 3.0
    max_smooth_offset_mm: float = 0.75
    lookahead_mm: float = 6.0
    wall_clearance_mm: float = 0.45
    fov_degrees: float = 85.0
    max_frames: Optional[int] = None


def _normalize(v: np.ndarray, fallback: Sequence[float] = (0.0, 0.0, 1.0)) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return np.asarray(fallback, dtype=float)
    return v / n


def _load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _node_map(graph: Dict) -> Dict[str, Dict]:
    return {str(node["id"]): node for node in graph.get("nodes", [])}


def _edge_key(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _edge_lookup(graph: Dict) -> Dict[Tuple[str, str], Dict]:
    edges: Dict[Tuple[str, str], Dict] = {}
    for edge in graph.get("edges", []):
        edges[_edge_key(str(edge["source"]), str(edge["target"]))] = edge
    return edges


def _edge_length(edge: Dict, nodes: Dict[str, Dict]) -> float:
    p0 = np.asarray(nodes[str(edge["source"])]["position_mm"], dtype=float)
    p1 = np.asarray(nodes[str(edge["target"])]["position_mm"], dtype=float)
    return float(np.linalg.norm(p1 - p0))


def _adjacency(graph: Dict, nodes: Dict[str, Dict]) -> Dict[str, List[Tuple[str, Dict]]]:
    adj: Dict[str, List[Tuple[str, Dict]]] = {}
    for edge in graph.get("edges", []):
        src = str(edge["source"])
        dst = str(edge["target"])
        adj.setdefault(src, []).append((dst, edge))
        adj.setdefault(dst, []).append((src, edge))
    for node_id, items in adj.items():
        p = np.asarray(nodes[node_id]["position_mm"], dtype=float)

        def sort_key(item: Tuple[str, Dict]) -> Tuple[str, str, float, str]:
            nxt, edge = item
            q = np.asarray(nodes[nxt]["position_mm"], dtype=float)
            direction = q - p
            return (
                str(edge.get("region", "")),
                str(edge.get("kind", "")),
                -float(direction[2]),
                str(nxt),
            )

        items.sort(key=sort_key)
    return adj


def _default_start_node(graph: Dict, camera_paths: Optional[Dict]) -> str:
    nodes = _node_map(graph)
    if camera_paths and camera_paths.get("start_node") in nodes:
        return str(camera_paths["start_node"])
    if "ureter_start" in nodes:
        return "ureter_start"
    if "upj" in nodes:
        return "upj"
    if graph.get("nodes"):
        return str(graph["nodes"][0]["id"])
    raise ValueError("centerline graph does not contain any nodes")


def _dfs_node_walk(graph: Dict, start: str) -> List[str]:
    nodes = _node_map(graph)
    adj = _adjacency(graph, nodes)
    visited = {start}
    walk = [start]

    def visit(node_id: str) -> None:
        for nxt, _edge in adj.get(node_id, []):
            if nxt in visited:
                continue
            visited.add(nxt)
            walk.append(nxt)
            visit(nxt)
            walk.append(node_id)

    visit(start)
    return walk


def _shortest_node_route(graph: Dict, start: str, goal: str) -> List[str]:
    nodes = _node_map(graph)
    if goal not in nodes:
        raise ValueError(f"target node {goal!r} is not in centerline_graph.json")
    adj = _adjacency(graph, nodes)
    queue = [start]
    parent: Dict[str, Optional[str]] = {start: None}
    for cur in queue:
        if cur == goal:
            break
        for nxt, _edge in adj.get(cur, []):
            if nxt in parent:
                continue
            parent[nxt] = cur
            queue.append(nxt)
    if goal not in parent:
        raise ValueError(f"no route from {start!r} to {goal!r}")
    route = [goal]
    while route[-1] != start:
        prev = parent[route[-1]]
        if prev is None:
            break
        route.append(prev)
    route.reverse()
    return route


def _node_sequence(graph: Dict, start: str, options: RenderPathOptions) -> Tuple[List[str], str]:
    traversal = str(options.traversal).lower()
    if options.target_node:
        route = _shortest_node_route(graph, start, options.target_node)
        return route + route[-2::-1], "route"
    if traversal == "dfs":
        return _dfs_node_walk(graph, start), "dfs"
    if traversal == "pelvis":
        route = _shortest_node_route(graph, start, "pelvis_center")
        return route + route[-2::-1], "pelvis"
    raise ValueError("traversal must be 'dfs' or 'pelvis', or provide target_node")


def _sample_node_sequence(graph: Dict, node_ids: Sequence[str], spacing_mm: float) -> List[Dict]:
    nodes = _node_map(graph)
    edges = _edge_lookup(graph)
    spacing = max(float(spacing_mm), 0.1)
    samples: List[Dict] = []
    for a, b in zip(node_ids[:-1], node_ids[1:]):
        edge = edges.get(_edge_key(a, b))
        if edge is None:
            raise ValueError(f"no edge between {a!r} and {b!r}")
        p0 = np.asarray(nodes[a]["position_mm"], dtype=float)
        p1 = np.asarray(nodes[b]["position_mm"], dtype=float)
        length = float(np.linalg.norm(p1 - p0))
        steps = max(1, int(np.ceil(length / spacing)))
        reverse = str(edge["source"]) != a
        if reverse:
            r0 = float(edge["radius1_mm"])
            r1 = float(edge["radius0_mm"])
        else:
            r0 = float(edge["radius0_mm"])
            r1 = float(edge["radius1_mm"])
        edge_source = str(edge["source"])
        edge_target = str(edge["target"])
        edge_forward = np.asarray(nodes[edge_target]["position_mm"], dtype=float) - np.asarray(
            nodes[edge_source]["position_mm"],
            dtype=float,
        )
        for i in range(steps + 1):
            if samples and i == 0:
                continue
            t = i / float(steps)
            pos = p0 * (1.0 - t) + p1 * t
            radius = r0 * (1.0 - t) + r1 * t
            edge_t = 1.0 - t if reverse else t
            samples.append(
                {
                    "position": pos,
                    "radius_mm": float(radius),
                    "edge_id": edge.get("id"),
                    "edge_t": float(edge_t),
                    "edge_forward": edge_forward,
                    "region": edge.get("region"),
                    "kind": edge.get("kind"),
                }
            )
    return samples


def _cumulative_lengths(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.asarray([], dtype=float)
    if len(points) == 1:
        return np.asarray([0.0], dtype=float)
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(seg)))


def _resample_samples(samples: List[Dict], options: RenderPathOptions) -> Tuple[List[Dict], float]:
    if not samples:
        raise ValueError("camera path has no samples")
    points = np.vstack([sample["position"] for sample in samples])
    lengths = _cumulative_lengths(points)
    total = float(lengths[-1])
    if total < 1e-6:
        raise ValueError("camera path length is zero")
    fps = max(float(options.fps), 1.0)
    speed = max(float(options.speed_mm_s), 0.1)
    desired_step = speed / fps
    frame_distances = np.arange(0.0, total, desired_step, dtype=float)
    if len(frame_distances) == 0 or abs(float(frame_distances[-1]) - total) > 1e-6:
        frame_distances = np.concatenate((frame_distances, [total]))
    effective_step = total / max(len(frame_distances) - 1, 1)

    coords = np.column_stack(
        [
            np.interp(frame_distances, lengths, points[:, axis])
            for axis in range(3)
        ]
    )
    radii = np.interp(frame_distances, lengths, np.asarray([float(s["radius_mm"]) for s in samples], dtype=float))
    frame_samples: List[Dict] = []
    indices = np.searchsorted(lengths, frame_distances, side="right") - 1
    indices = np.clip(indices, 0, len(samples) - 1)
    for idx, distance, pos, radius in zip(indices, frame_distances, coords, radii):
        base = samples[int(idx)]
        frame_samples.append(
            {
                "distance_mm": float(distance),
                "position": pos,
                "original_position": pos.copy(),
                "radius_mm": float(radius),
                "edge_id": base.get("edge_id"),
                "edge_t": base.get("edge_t"),
                "edge_forward": base.get("edge_forward"),
                "region": base.get("region"),
                "kind": base.get("kind"),
            }
        )
    return frame_samples, effective_step


def _output_frame_indices(native_frame_count: int, max_frames: Optional[int]) -> List[int]:
    if native_frame_count <= 0:
        return []
    if max_frames is None or int(max_frames) <= 0 or int(max_frames) >= native_frame_count:
        return list(range(native_frame_count))
    if int(max_frames) == 1:
        return [0]
    return [int(v) for v in np.linspace(0, native_frame_count - 1, int(max_frames), dtype=int)]


def _union_sdf(points: np.ndarray, primitives: Iterable[Dict]) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    sdf = np.full(len(pts), np.inf, dtype=float)
    for primitive in primitives:
        sdf = np.minimum(sdf, primitive_sdf(pts, primitive))
    return sdf


def _smooth_positions(samples: List[Dict], graph: Dict, options: RenderPathOptions, effective_step_mm: float) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    positions = np.vstack([sample["position"] for sample in samples])
    radii = np.asarray([float(sample["radius_mm"]) for sample in samples], dtype=float)
    warnings: List[str] = []
    if len(positions) < 3 or options.smooth_window_mm <= 0:
        sdf = _union_sdf(positions, graph.get("primitives", []))
        return positions, sdf, warnings

    sigma_frames = max(float(options.smooth_window_mm) / max(float(effective_step_mm), 1e-6), 0.0)
    smoothed = gaussian_filter1d(positions, sigma=sigma_frames, axis=0, mode="nearest")
    max_global_offset = max(float(options.max_smooth_offset_mm), 0.0)
    safe_offsets = np.maximum(0.0, (radii - float(options.wall_clearance_mm)) * 0.35)
    offset_limits = np.minimum(max_global_offset, safe_offsets)
    deltas = smoothed - positions
    norms = np.linalg.norm(deltas, axis=1)
    mask = norms > np.maximum(offset_limits, 1e-8)
    if np.any(mask):
        smoothed[mask] = positions[mask] + deltas[mask] * (offset_limits[mask] / norms[mask])[:, None]

    primitives = graph.get("primitives", [])
    target_sdf = -float(options.wall_clearance_mm)
    smooth_sdf = _union_sdf(smoothed, primitives)
    original_sdf = _union_sdf(positions, primitives)
    unsafe = np.where(smooth_sdf > target_sdf)[0]
    for idx in unsafe:
        if original_sdf[idx] > target_sdf:
            smoothed[idx] = positions[idx]
            warnings.append(
                f"frame {idx} original centerline clearance is below requested wall clearance "
                f"({original_sdf[idx]:.3f} mm sdf)"
            )
            continue
        lo = 0.0
        hi = 1.0
        best = positions[idx]
        for _ in range(18):
            mid = (lo + hi) * 0.5
            candidate = positions[idx] * (1.0 - mid) + smoothed[idx] * mid
            sdf = float(_union_sdf(candidate[None, :], primitives)[0])
            if sdf <= target_sdf:
                lo = mid
                best = candidate
            else:
                hi = mid
        smoothed[idx] = best
    smoothed[0] = positions[0]
    smoothed[-1] = positions[-1]
    final_sdf = _union_sdf(smoothed, primitives)
    clamped = int(np.count_nonzero(unsafe))
    if clamped:
        warnings.append(f"clamped {clamped} smoothed frames back toward the centerline to preserve lumen clearance")
    return smoothed, final_sdf, warnings


def _parallel_transport_up(forwards: np.ndarray, initial_up: Sequence[float] = (0.0, 1.0, 0.0)) -> np.ndarray:
    ups = np.zeros_like(forwards)
    up = np.asarray(initial_up, dtype=float)
    up = up - forwards[0] * float(np.dot(up, forwards[0]))
    up = _normalize(up, (1.0, 0.0, 0.0))
    ups[0] = up
    for idx in range(1, len(forwards)):
        f = forwards[idx]
        up = ups[idx - 1] - f * float(np.dot(ups[idx - 1], f))
        if np.linalg.norm(up) < 1e-8:
            fallback = np.array([0.0, 1.0, 0.0])
            if abs(float(np.dot(fallback, f))) > 0.95:
                fallback = np.array([1.0, 0.0, 0.0])
            up = fallback - f * float(np.dot(fallback, f))
        ups[idx] = _normalize(up, (0.0, 1.0, 0.0))
    return ups


def _camera_matrix(position: np.ndarray, forward: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = _normalize(forward, (0.0, 0.0, 1.0))
    u = up - f * float(np.dot(up, f))
    u = _normalize(u, (0.0, 1.0, 0.0))
    right = _normalize(np.cross(f, u), (1.0, 0.0, 0.0))
    backward = -f
    u = _normalize(np.cross(backward, right), (0.0, 1.0, 0.0))
    mat = np.eye(4, dtype=float)
    mat[:3, 0] = right
    mat[:3, 1] = u
    mat[:3, 2] = backward
    mat[:3, 3] = position
    return mat


def _attach_orientation(samples: List[Dict], positions: np.ndarray, options: RenderPathOptions) -> None:
    distances = np.asarray([float(sample["distance_mm"]) for sample in samples], dtype=float)
    forwards = np.zeros_like(positions)
    for idx, (distance, pos) in enumerate(zip(distances, positions)):
        raw_forward = samples[idx].get("edge_forward")
        forward = np.asarray(raw_forward, dtype=float) if raw_forward is not None else np.zeros(3, dtype=float)
        if forward.shape != (3,) or np.linalg.norm(forward) < 1e-8:
            lookahead = max(float(options.lookahead_mm), 0.1)
            target_distance = min(float(distances[-1]), distance + lookahead)
            j = int(np.searchsorted(distances, target_distance, side="left"))
            if j <= idx and idx < len(positions) - 1:
                j = idx + 1
            if j < len(positions) and np.linalg.norm(positions[j] - pos) > 1e-6:
                forward = positions[j] - pos
            elif idx > 0:
                forward = pos - positions[idx - 1]
            else:
                forward = np.array([0.0, 0.0, 1.0], dtype=float)
        forwards[idx] = _normalize(forward, forwards[idx - 1] if idx else (0.0, 0.0, 1.0))
    if len(forwards) > 2:
        sigma = max(0.5, min(3.0, float(options.lookahead_mm) / max(float(options.speed_mm_s / options.fps), 1e-6) * 0.25))
        forwards = gaussian_filter1d(forwards, sigma=sigma, axis=0, mode="nearest")
        forwards = np.vstack([_normalize(f, (0.0, 0.0, 1.0)) for f in forwards])
    ups = _parallel_transport_up(forwards)
    for idx, sample in enumerate(samples):
        mat = _camera_matrix(positions[idx], forwards[idx], ups[idx])
        sample["position"] = positions[idx]
        sample["forward"] = forwards[idx]
        sample["up"] = ups[idx]
        sample["cam2world"] = mat


def build_blenderproc_camera_plan(case_dir: str | Path, options: Optional[RenderPathOptions] = None) -> Dict:
    case_dir = Path(case_dir)
    options = options or RenderPathOptions()
    graph = _load_json(case_dir / "centerline_graph.json")
    camera_paths_path = case_dir / "camera_paths.json"
    camera_paths = _load_json(camera_paths_path) if camera_paths_path.exists() else None
    start = _default_start_node(graph, camera_paths)
    nodes, traversal = _node_sequence(graph, start, options)
    dense = _sample_node_sequence(graph, nodes, float(options.sample_spacing_mm))
    samples, effective_step = _resample_samples(dense, options)
    smoothed, sdf, warnings = _smooth_positions(samples, graph, options, effective_step)
    _attach_orientation(samples, smoothed, options)

    fps = max(float(options.fps), 1.0)
    native_frame_count = len(samples)
    output_indices = _output_frame_indices(native_frame_count, options.max_frames)
    frames = []
    for output_idx, source_idx in enumerate(output_indices):
        sample = samples[source_idx]
        sdf_value = sdf[source_idx]
        mat = np.asarray(sample["cam2world"], dtype=float)
        pos = np.asarray(sample["position"], dtype=float)
        frames.append(
            {
                "frame_index": output_idx,
                "source_frame_index": int(source_idx),
                "time_s": float(source_idx / fps),
                "distance_mm": float(sample["distance_mm"]),
                "position_mm": [float(v) for v in pos],
                "forward": [float(v) for v in np.asarray(sample["forward"], dtype=float)],
                "up": [float(v) for v in np.asarray(sample["up"], dtype=float)],
                "cam2world_opengl": [[float(v) for v in row] for row in mat],
                "edge_id": sample.get("edge_id"),
                "edge_t": sample.get("edge_t"),
                "region": sample.get("region"),
                "kind": sample.get("kind"),
                "radius_mm": float(sample["radius_mm"]),
                "sdf_mm": float(sdf_value),
            }
        )

    return {
        "schema": "kidney_meshgen_blenderproc_camera_plan_v0.1",
        "units": "millimeters",
        "case_dir": str(case_dir),
        "start_node": start,
        "traversal": traversal,
        "target_node": options.target_node,
        "node_walk": list(nodes),
        "fps": float(fps),
        "speed_mm_s": float(options.speed_mm_s),
        "effective_step_mm": float(effective_step),
        "fov_degrees": float(options.fov_degrees),
        "wall_clearance_mm": float(options.wall_clearance_mm),
        "smoothing": {
            "sample_spacing_mm": float(options.sample_spacing_mm),
            "smooth_window_mm": float(options.smooth_window_mm),
            "max_smooth_offset_mm": float(options.max_smooth_offset_mm),
            "lookahead_mm": float(options.lookahead_mm),
        },
        "native_frame_count": int(native_frame_count),
        "subsampled": bool(len(output_indices) != native_frame_count),
        "output_source_frame_indices": [int(v) for v in output_indices],
        "frame_count": len(frames),
        "warnings": warnings,
        "frames": frames,
    }


def write_blenderproc_camera_plan(case_dir: str | Path, out_dir: str | Path, options: Optional[RenderPathOptions] = None) -> Dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = build_blenderproc_camera_plan(case_dir, options)
    json_path = out_dir / "camera_poses.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)

    csv_path = out_dir / "camera_poses.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frame_index",
                "time_s",
                "x_mm",
                "y_mm",
                "z_mm",
                "forward_x",
                "forward_y",
                "forward_z",
                "up_x",
                "up_y",
                "up_z",
                "edge_id",
                "edge_t",
                "radius_mm",
                "sdf_mm",
            ]
        )
        for frame in plan["frames"]:
            writer.writerow(
                [
                    frame["frame_index"],
                    frame["time_s"],
                    *frame["position_mm"],
                    *frame["forward"],
                    *frame["up"],
                    frame.get("edge_id"),
                    frame.get("edge_t"),
                    frame.get("radius_mm"),
                    frame.get("sdf_mm"),
                ]
            )
    return {
        "camera_poses_json": str(json_path),
        "camera_poses_csv": str(csv_path),
        "frame_count": str(plan["frame_count"]),
    }
