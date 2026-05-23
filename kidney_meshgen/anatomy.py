from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .config import GeneratorConfig
from .graph import AnatomyGraph, Edge, Node, Primitive


def _rand_range(rng: np.random.Generator, span: Tuple[float, float]) -> float:
    return float(rng.uniform(float(span[0]), float(span[1])))


def _split_calyces(rng: np.random.Generator, n: int) -> Dict[str, int]:
    """Allocate minor calyces across upper/middle/lower poles."""
    raw = rng.multinomial(n - 3, [0.34, 0.33, 0.33]) + 1
    return {"upper": int(raw[0]), "middle": int(raw[1]), "lower": int(raw[2])}


@dataclass
class _SegmentGeom:
    p0: np.ndarray
    p1: np.ndarray
    radius_mm: float
    segment_type: str
    source_node: str
    target_node: str


@dataclass
class _CupGeom:
    center: np.ndarray
    radius_mm: float
    label: str


def _point_to_segment_distance(pt: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-8:
        return float(np.linalg.norm(pt - a))
    t = float(np.clip(np.dot(pt - a, ab) / denom, 0.0, 1.0))
    proj = a + t * ab
    return float(np.linalg.norm(pt - proj))


def _segment_segment_distance(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> float:
    """Shortest distance between 3D segments."""
    u = q1 - p1
    v = q2 - p2
    w0 = p1 - p2
    a = float(np.dot(u, u))
    b = float(np.dot(u, v))
    c = float(np.dot(v, v))
    d = float(np.dot(u, w0))
    e = float(np.dot(v, w0))
    eps = 1e-8
    D = a * c - b * b
    sN = D
    sD = D
    tN = D
    tD = D

    if D < eps:
        sN = 0.0
        sD = 1.0
        tN = e
        tD = c
    else:
        sN = (b * e - c * d)
        tN = (a * e - b * d)
        if sN < 0.0:
            sN = 0.0
            tN = e
            tD = c
        elif sN > sD:
            sN = sD
            tN = e + b
            tD = c

    if tN < 0.0:
        tN = 0.0
        if -d < 0.0:
            sN = 0.0
        elif -d > a:
            sN = sD
        else:
            sN = -d
            sD = a
    elif tN > tD:
        tN = tD
        if (-d + b) < 0.0:
            sN = 0.0
        elif (-d + b) > a:
            sN = sD
        else:
            sN = (-d + b)
            sD = a

    sc = 0.0 if abs(sN) < eps else sN / sD
    tc = 0.0 if abs(tN) < eps else tN / tD
    dP = w0 + sc * u - tc * v
    return float(np.linalg.norm(dP))


def _ellipsoid_scaled_norm(pt: np.ndarray, center: np.ndarray, radii: np.ndarray) -> float:
    return float(np.linalg.norm((pt - center) / np.maximum(radii, 1e-8)))


def build_anatomy_graph(config: GeneratorConfig) -> AnatomyGraph:
    rng = np.random.default_rng(config.seed)

    side_sign = 1.0 if config.side.lower() == "right" else -1.0
    pelvis_type = config.pelvis_type
    if pelvis_type == "random":
        pelvis_type = str(rng.choice(["single", "divided"], p=[0.62, 0.38]))

    lower_pole_mode = str(getattr(config, "lower_pole_access", "intermediate")).lower()
    if lower_pole_mode == "random":
        lower_pole_mode = str(rng.choice(["easy", "intermediate", "hard"], p=[0.32, 0.44, 0.24]))
    if lower_pole_mode == "easy":
        lower_angle_range = (25.0, 46.0)
        lower_trunk_range = (14.0, 25.0)
        lower_radius_range = (3.0, 4.6)
        lower_branch_range = (7.0, 13.0)
        lower_splay_range = (-24.0, 24.0)
        lower_y_span = 5.5
    elif lower_pole_mode == "hard":
        lower_angle_range = (56.0, 82.0)
        lower_trunk_range = (28.0, 46.0)
        lower_radius_range = (1.7, 2.9)
        lower_branch_range = (12.0, 24.0)
        lower_splay_range = (-48.0, 48.0)
        lower_y_span = 10.5
    else:
        lower_pole_mode = "intermediate"
        lower_angle_range = tuple(config.lower_pole_angle_degrees)
        lower_trunk_range = tuple(config.lower_access_trunk_length_mm)
        lower_radius_range = tuple(config.lower_access_radius_mm)
        lower_branch_range = tuple(config.lower_calyx_branch_length_mm)
        lower_splay_range = tuple(config.lower_calyx_splay_degrees)
        lower_y_span = 7.0

    labels: Dict[int, str] = {
        0: "background",
        1: "renal_pelvis",
        2: "ureter_stub",
        3: "upj",
    }
    next_label_id = 4

    nodes: List[Node] = []
    edges: List[Edge] = []
    primitives: List[Primitive] = []
    calyx_targets: List[Dict] = []
    segment_geoms: List[_SegmentGeom] = []
    cup_geoms: List[_CupGeom] = []

    def add_node(node_id: str, pos, kind: str, region: str) -> None:
        nodes.append(Node(node_id, tuple(float(v) for v in pos), kind, region))

    def label_to_id(region: str) -> int:
        nonlocal next_label_id
        for k, v in labels.items():
            if v == region:
                return k
        labels[next_label_id] = region
        next_label_id += 1
        return next_label_id - 1

    def node_lookup() -> Dict[str, Node]:
        return {n.id: n for n in nodes}

    def add_edge(edge_id: str, source: str, target: str, r0: float, r1: float, kind: str, region: str, parent: str = "") -> None:
        edges.append(Edge(edge_id, source, target, float(r0), float(r1), kind, region, parent))
        nmap = node_lookup()
        p0 = np.asarray(nmap[source].position_mm, dtype=float)
        p1 = np.asarray(nmap[target].position_mm, dtype=float)
        primitives.append(
            Primitive(
                id=edge_id,
                kind="tapered_capsule",
                label=region,
                label_id=label_to_id(region),
                p0=tuple(p0),
                p1=tuple(p1),
                r0=float(r0),
                r1=float(r1),
            )
        )
        segment_geoms.append(
            _SegmentGeom(
                p0=p0,
                p1=p1,
                radius_mm=max(float(r0), float(r1)),
                segment_type=kind,
                source_node=source,
                target_node=target,
            )
        )

    # Core pelvis and ureter nodes.
    pelvis_center = np.array([0.0, 0.0, 0.0], dtype=float)
    upj = np.array([-4.0 * side_sign, 0.0, -10.5], dtype=float)
    add_node("pelvis_center", pelvis_center, "pelvis", "renal_pelvis")
    add_node("upj", upj, "junction", "upj")

    pelvis_radii = np.array(
        [
            _rand_range(rng, config.pelvis_radius_x_mm),
            _rand_range(rng, config.pelvis_radius_y_mm),
            _rand_range(rng, config.pelvis_radius_z_mm),
        ],
        dtype=float,
    )
    primitives.append(
        Primitive(
            id="pelvis_ellipsoid",
            kind="ellipsoid",
            label="renal_pelvis",
            label_id=1,
            center=tuple(pelvis_center),
            radii=tuple(float(v) for v in pelvis_radii),
        )
    )

    upj_radius = _rand_range(rng, config.upj_radius_mm)
    add_edge("edge_upj_pelvis", "upj", "pelvis_center", upj_radius, max(upj_radius * 1.22, 4.4), "upj", "upj")

    if config.include_ureter_stub:
        ureter_len = _rand_range(rng, config.ureter_stub_length_mm)
        ur = _rand_range(rng, config.ureter_radius_mm)
        seg_count = max(int(config.ureter_segment_count), 2)
        curve_mag = _rand_range(rng, config.ureter_curve_mm)

        ureter_points: List[np.ndarray] = []
        start = upj + np.array([-0.5 * side_sign, 0.0, -ureter_len], dtype=float)
        for i in range(seg_count + 1):
            t = i / seg_count
            pos = start * (1 - t) + upj * t
            sway_y = curve_mag * np.sin(np.pi * t) * rng.uniform(-1.0, 1.0)
            sway_x = 0.35 * curve_mag * np.sin(2.0 * np.pi * t + rng.uniform(-0.35, 0.35)) * side_sign
            pos = pos + np.array([sway_x, sway_y, 0.0])
            ureter_points.append(pos)

        add_node("ureter_start", ureter_points[0], "scope_start", "ureter_stub")
        last_id = "ureter_start"
        for i, pos in enumerate(ureter_points[1:-1], start=1):
            node_id = f"ureter_mid_{i:02d}"
            add_node(node_id, pos, "ureter", "ureter_stub")
            r_next = ur * rng.uniform(0.96, 1.08)
            add_edge(f"edge_ureter_{i-1:02d}", last_id, node_id, ur, r_next, "ureter", "ureter_stub")
            ur = r_next
            last_id = node_id
        add_edge(f"edge_ureter_{seg_count-1:02d}", last_id, "upj", ur, upj_radius, "ureter", "ureter_stub")

    # Major pelvis outlets / regional roots.
    region_roots: Dict[str, str] = {}
    if pelvis_type == "divided":
        outlet_specs = {
            "upper": np.array([5.8 * side_sign, -1.0, 8.6]),
            "middle": np.array([6.8 * side_sign, 0.0, 0.6]),
            "lower": np.array([5.8 * side_sign, 1.2, -9.6]),
        }
        outlet_r_range = (3.0, 4.8)
        label_prefix = "major"
    else:
        outlet_specs = {
            "upper": np.array([3.8 * side_sign, -1.2, 6.8]),
            "middle": np.array([4.6 * side_sign, 0.0, 0.0]),
            "lower": np.array([3.8 * side_sign, 1.2, -7.2]),
        }
        outlet_r_range = (2.5, 4.2)
        label_prefix = "outlet"

    for region, base_offset in outlet_specs.items():
        region_name = f"{label_prefix}_{region}"
        label_to_id(region_name)
        node_id = f"{label_prefix}_{region}"
        pos = pelvis_center + base_offset + np.array([0.0, rng.uniform(-1.2, 1.2), rng.uniform(-1.5, 1.5)])
        add_node(node_id, pos, "major_calyx", region_name)
        r1 = _rand_range(rng, outlet_r_range)
        add_edge(f"edge_pelvis_{node_id}", "pelvis_center", node_id, max(pelvis_radii[0] * 0.38, 3.2), r1, "major_calyx", region_name)
        region_roots[region] = node_id

    n_calyces = int(rng.integers(config.calyx_count_min, config.calyx_count_max + 1))
    groups = _split_calyces(rng, n_calyces)

    # Explicit lower-pole access trunk: lower major branch -> separate lower
    # attachment nodes -> minor calyces. This keeps the lower pole navigable and
    # avoids many cups merging near the pelvis.
    lower_attachment_ids: List[str] = []
    if config.lower_pole_clean_access and groups.get("lower", 0) > 0:
        lower_root_id = region_roots["lower"]
        lower_root_pos = np.asarray(node_lookup()[lower_root_id].position_mm, dtype=float)
        trunk_len = _rand_range(rng, lower_trunk_range)
        trunk_angle = -np.deg2rad(_rand_range(rng, lower_angle_range))
        trunk_y = rng.uniform(-1.0, 1.0) * _rand_range(rng, config.lower_access_curve_mm)
        tip = lower_root_pos + np.array(
            [side_sign * trunk_len * np.cos(trunk_angle), trunk_y, trunk_len * np.sin(trunk_angle)],
            dtype=float,
        )
        control = lower_root_pos * 0.55 + tip * 0.45 + np.array(
            [side_sign * rng.uniform(-1.5, 1.5), trunk_y * 0.6 + rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)],
            dtype=float,
        )
        attach_count = max(groups["lower"], 2)
        r0 = _rand_range(rng, lower_radius_range)
        r1 = max(1.6, r0 * rng.uniform(0.62, 0.82))
        prev_id = lower_root_id
        prev_r = r0
        for j in range(1, attach_count + 1):
            t = j / attach_count
            pos = (1 - t) ** 2 * lower_root_pos + 2 * (1 - t) * t * control + t**2 * tip
            node_id = f"lower_access_attach_{j:02d}"
            add_node(node_id, pos, "lower_access_attachment", "lower_pole_access")
            this_r = r0 + t * (r1 - r0)
            add_edge(
                f"edge_lower_access_{j:02d}",
                prev_id,
                node_id,
                prev_r,
                this_r,
                "lower_access",
                "lower_pole_access",
                parent="lower",
            )
            lower_attachment_ids.append(node_id)
            prev_id = node_id
            prev_r = this_r

    region_specs = {
        "upper": {"angle_deg": (22.0, 62.0), "length_mm": (14.0, 30.0), "y_span_mm": 9.0},
        "middle": {"angle_deg": (-16.0, 18.0), "length_mm": (13.0, 28.0), "y_span_mm": 10.5},
        "lower": {
            "angle_deg": lower_splay_range if bool(config.lower_pole_clean_access) else tuple(-v for v in config.lower_pole_angle_degrees[::-1]),
            "length_mm": lower_branch_range if bool(config.lower_pole_clean_access) else (18.0, 34.0),
            "y_span_mm": lower_y_span if bool(config.lower_pole_clean_access) else 8.5,
        },
    }

    def candidate_ok(
        cup_center: np.ndarray,
        cup_radius: float,
        prox_seg: Tuple[np.ndarray, np.ndarray, float, str],
        distal_seg: Tuple[np.ndarray, np.ndarray, float, str],
    ) -> bool:
        if _ellipsoid_scaled_norm(
            cup_center,
            pelvis_center,
            pelvis_radii + np.array([config.cup_to_pelvis_clearance_mm] * 3),
        ) < 1.0:
            return False

        for cup in cup_geoms:
            min_sep = cup.radius_mm + cup_radius + config.cup_center_clearance_mm
            if float(np.linalg.norm(cup.center - cup_center)) < min_sep:
                return False

        root_id = prox_seg[3]
        for seg in segment_geoms:
            shares_root = seg.source_node == root_id or seg.target_node == root_id

            # Proximal branch segments are expected to share geometry at the
            # major-calyx outlet. Do not reject a candidate simply because two
            # branches meet at their shared root; downstream checks still guard
            # against fused cups or distal chambers.
            if not shares_root:
                d1 = _segment_segment_distance(prox_seg[0], prox_seg[1], seg.p0, seg.p1)
                allow1 = prox_seg[2] + seg.radius_mm + config.tube_clearance_mm * 0.75
                if d1 < allow1:
                    return False

            d2 = _segment_segment_distance(distal_seg[0], distal_seg[1], seg.p0, seg.p1)
            allow2 = distal_seg[2] + seg.radius_mm + config.tube_clearance_mm
            if shares_root:
                allow2 *= 0.62
            if d2 < allow2:
                return False

            d3 = _point_to_segment_distance(cup_center, seg.p0, seg.p1)
            cup_allow = cup_radius + seg.radius_mm + config.tube_clearance_mm
            if shares_root:
                cup_allow *= 0.55
            if d3 < cup_allow:
                return False

        return True

    calyx_index = 0
    for region in ["upper", "middle", "lower"]:
        count = groups[region]
        spec = region_specs[region]
        base_root_id = region_roots[region]
        base_angles = np.linspace(spec["angle_deg"][0], spec["angle_deg"][1], count) if count > 1 else np.array([np.mean(spec["angle_deg"])])
        rng.shuffle(base_angles)
        # Keep cups within a pole from collapsing into the same AP location.
        # This works with the clearance checks below to reduce accidental
        # chamber fusion without making the anatomy look too regular.
        y_slots = np.linspace(-spec["y_span_mm"], spec["y_span_mm"], count) if count > 1 else np.array([0.0])
        rng.shuffle(y_slots)

        for i in range(count):
            if region == "lower" and lower_attachment_ids:
                root_id = lower_attachment_ids[min(i, len(lower_attachment_ids) - 1)]
            else:
                root_id = base_root_id
            root_pos = np.array(node_lookup()[root_id].position_mm, dtype=float)

            calyx_index += 1
            label = f"calyx_{region}_{i+1:02d}"
            label_to_id(label)

            best_candidate = None
            best_score = -np.inf
            for _attempt in range(int(config.branch_sample_attempts)):
                reach = rng.uniform(*spec["length_mm"])
                angle = np.deg2rad(base_angles[i] + rng.normal(0.0, 5.0))
                xoff = side_sign * (reach * np.cos(angle) + rng.uniform(-1.0, 1.0))
                zoff = reach * np.sin(angle) + rng.uniform(-1.2, 1.2)
                yoff = float(np.clip(y_slots[i] + rng.normal(0.0, spec["y_span_mm"] * 0.18), -spec["y_span_mm"], spec["y_span_mm"]))
                endpoint = root_pos + np.array([xoff, yoff, zoff], dtype=float)

                if region == "lower":
                    # Lower minor cups should fan from the access trunk, not rejoin the pelvis.
                    endpoint[2] -= rng.uniform(0.25, 2.0)
                    endpoint[1] += rng.uniform(-1.2, 1.2)

                neck_frac = rng.uniform(0.50, 0.74)
                neck = root_pos * (1 - neck_frac) + endpoint * neck_frac
                neck += np.array(
                    [side_sign * rng.uniform(-0.8, 0.8), rng.uniform(-1.8, 1.8), rng.uniform(-1.5, 1.5)],
                    dtype=float,
                )

                if region == "lower" and lower_attachment_ids:
                    r_start = _rand_range(rng, (1.35, 2.45))
                    r_neck = min(_rand_range(rng, config.infundibulum_radius_mm), r_start * rng.uniform(0.82, 1.05))
                    r_cup_tube = max(1.15, min(_rand_range(rng, (1.3, 2.4)), r_neck * 1.10))
                else:
                    r_start = _rand_range(rng, (2.3, 4.6))
                    r_neck = _rand_range(rng, config.infundibulum_radius_mm)
                    r_cup_tube = max(1.3, _rand_range(rng, (1.5, 3.0)))
                cup_r = _rand_range(rng, config.calyx_cup_radius_mm)
                prox = (root_pos, neck, max(r_start, r_neck), root_id)
                dist = (neck, endpoint, max(r_neck, r_cup_tube), root_id)

                if candidate_ok(endpoint, cup_r, prox, dist):
                    clearance = [float(np.linalg.norm(cup.center - endpoint)) - (cup.radius_mm + cup_r) for cup in cup_geoms]
                    score = (min(clearance) if clearance else 100.0)
                    score += abs(endpoint[2] - root_pos[2]) * 0.25
                    score += abs(endpoint[1] - root_pos[1]) * 0.12
                    if score > best_score:
                        best_candidate = (neck, endpoint, r_start, r_neck, r_cup_tube, cup_r)
                        best_score = score

            if best_candidate is None:
                fallback_reach = float(np.mean(spec["length_mm"]))
                fallback_angle = np.deg2rad(float(np.mean(spec["angle_deg"])))
                endpoint = root_pos + np.array(
                    [side_sign * fallback_reach * np.cos(fallback_angle), float(y_slots[i]), fallback_reach * np.sin(fallback_angle)],
                    dtype=float,
                )
                # If a rare fallback is needed, push it away from previous cups
                # and use a modest cup radius to avoid obviously fused chambers.
                for cup in cup_geoms:
                    sep_vec = endpoint - cup.center
                    sep = float(np.linalg.norm(sep_vec))
                    min_sep = cup.radius_mm + 3.5 + max(config.cup_center_clearance_mm * 0.5, 2.0)
                    if sep < min_sep and sep > 1e-6:
                        endpoint = cup.center + sep_vec / sep * min_sep
                neck = root_pos * 0.42 + endpoint * 0.58
                best_candidate = (neck, endpoint, 3.0, 2.1, 1.8, 3.2)

            neck, endpoint, r_start, r_neck, r_cup_tube, cup_r = best_candidate
            neck_id = f"{label}_neck"
            cup_id = f"{label}_cup"
            add_node(neck_id, neck, "infundibulum", label)
            add_node(cup_id, endpoint, "calyx_cup", label)

            add_edge(f"edge_{label}_proximal", root_id, neck_id, r_start, r_neck, "infundibulum", label, parent=region)
            add_edge(f"edge_{label}_distal", neck_id, cup_id, r_neck, r_cup_tube, "infundibulum", label, parent=region)

            cup_radii = (
                cup_r * rng.uniform(0.92, 1.08),
                cup_r * rng.uniform(0.82, 1.02),
                cup_r * rng.uniform(0.82, 1.06),
            )
            primitives.append(
                Primitive(
                    id=f"{label}_cup_ellipsoid",
                    kind="ellipsoid",
                    label=label,
                    label_id=label_to_id(label),
                    center=tuple(endpoint),
                    radii=tuple(float(v) for v in cup_radii),
                )
            )
            cup_geoms.append(_CupGeom(center=np.asarray(endpoint, dtype=float), radius_mm=float(max(cup_radii)), label=label))
            calyx_targets.append(
                {
                    "id": label,
                    "region": region,
                    "cup_node": cup_id,
                    "center_mm": [float(v) for v in endpoint],
                    "approx_radius_mm": float(cup_r),
                }
            )

    metadata = {
        "calyx_distribution": groups,
        "lower_pole_access_mode": lower_pole_mode if lower_attachment_ids else "direct_branch",
        "lower_pole_attachment_nodes": lower_attachment_ids,
        "coordinate_notes": "x=lateral, y=anterior/posterior-like, z=cranio-caudal; values are millimeters",
    }
    return AnatomyGraph(
        nodes=nodes,
        edges=edges,
        primitives=primitives,
        labels=labels,
        calyx_targets=calyx_targets,
        pelvis_type=pelvis_type,
        metadata=metadata,
    )
