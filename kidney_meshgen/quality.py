from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .config import GeneratorConfig
from .graph import AnatomyGraph, edge_length


def _segment_distance(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> float:
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
        sN = b * e - c * d
        tN = a * e - b * d
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
        if -d + b < 0.0:
            sN = 0.0
        elif -d + b > a:
            sN = sD
        else:
            sN = -d + b
            sD = a
    sc = 0.0 if abs(sN) < eps else sN / sD
    tc = 0.0 if abs(tN) < eps else tN / tD
    return float(np.linalg.norm(w0 + sc * u - tc * v))


def analyze_geometry_quality(graph: AnatomyGraph, config: GeneratorConfig) -> Dict:
    nodes = graph.node_map()
    warnings: List[str] = []

    lower_edges = [e for e in graph.edges if e.region == "lower_pole_access" or e.kind == "lower_access"]
    lower_access_length = float(sum(edge_length(e, nodes) for e in lower_edges)) if lower_edges else 0.0
    lower_attachment_nodes = sorted([n.id for n in graph.nodes if n.id.startswith("lower_access_attach_")])

    # Segment clearances: distance between centerline segments minus the two max radii.
    min_seg_clearance = float("inf")
    min_seg_pair: Tuple[str, str] | None = None
    for i, a in enumerate(graph.edges):
        a0 = np.asarray(nodes[a.source].position_mm, dtype=float)
        a1 = np.asarray(nodes[a.target].position_mm, dtype=float)
        ar = max(float(a.radius0_mm), float(a.radius1_mm))
        for b in graph.edges[i + 1 :]:
            # Edges that directly enter the renal pelvis are expected to blend
            # into a shared chamber, so they should not be treated as accidental
            # distal-calyx/corridor merges.
            if "pelvis_center" in {a.source, a.target, b.source, b.target}:
                continue
            if {a.source, a.target}.intersection({b.source, b.target}):
                continue
            same_branch_family = bool(a.parent_region and a.parent_region == b.parent_region)
            proximal_family_blend = same_branch_family and (
                "proximal" in a.id or "proximal" in b.id or a.kind == "major_calyx" or b.kind == "major_calyx"
            )
            if proximal_family_blend:
                continue
            b0 = np.asarray(nodes[b.source].position_mm, dtype=float)
            b1 = np.asarray(nodes[b.target].position_mm, dtype=float)
            br = max(float(b.radius0_mm), float(b.radius1_mm))
            clearance = _segment_distance(a0, a1, b0, b1) - (ar + br)
            if clearance < min_seg_clearance:
                min_seg_clearance = float(clearance)
                min_seg_pair = (a.id, b.id)

    if not np.isfinite(min_seg_clearance):
        min_seg_clearance = None

    # Cup center clearance using approximate cup radius metadata.
    targets = graph.calyx_targets
    subtractive_papillae = [
        p for p in graph.primitives if getattr(p, "operation", "union") == "subtract" and "papilla" in p.id
    ]
    takazawa_pairs: Dict[str, List[str]] = {}
    for target in targets:
        level = str(target.get("level") or target.get("region") or "")
        ap = target.get("anterior_posterior")
        if level in {"upper", "middle", "lower"} and ap:
            takazawa_pairs.setdefault(level, []).append(str(ap))

    min_cup_clearance = float("inf")
    min_cup_pair: Tuple[str, str] | None = None
    for i, a in enumerate(targets):
        ac = np.asarray(a["center_mm"], dtype=float)
        ar = float(a.get("approx_radius_mm", 0.0))
        for b in targets[i + 1 :]:
            bc = np.asarray(b["center_mm"], dtype=float)
            br = float(b.get("approx_radius_mm", 0.0))
            clearance = float(np.linalg.norm(ac - bc) - (ar + br))
            if clearance < min_cup_clearance:
                min_cup_clearance = clearance
                min_cup_pair = (str(a["id"]), str(b["id"]))
    if not np.isfinite(min_cup_clearance):
        min_cup_clearance = None

    if lower_edges and len(lower_attachment_nodes) == 0:
        warnings.append("Lower-pole access edges exist but no lower_access_attach_* nodes were generated.")
    if config.lower_pole_clean_access and lower_access_length < 8.0:
        warnings.append("Lower-pole access trunk is shorter than expected.")
    if graph.metadata.get("anatomy_realism_profile") == "takazawa":
        for level in ["upper", "lower"]:
            present = set(takazawa_pairs.get(level, []))
            if not {"anterior", "posterior"}.issubset(present):
                warnings.append(f"Takazawa {level} calyx pair is incomplete: {sorted(present)}.")
    if graph.metadata.get("papilla_fornix_enabled") and len(subtractive_papillae) < len(targets):
        warnings.append("Papilla/fornix profile is enabled but not every calyx has a subtractive papilla primitive.")
    if min_seg_clearance is not None and min_seg_clearance < float(config.geometry_qa_warn_clearance_mm):
        warnings.append(f"Low non-connected segment clearance: {min_seg_clearance:.2f} mm between {min_seg_pair}.")
    if min_cup_clearance is not None and min_cup_clearance < 0.0:
        warnings.append(f"Approximate cup overlap: {min_cup_clearance:.2f} mm between {min_cup_pair}.")

    return {
        "schema": "kidney_meshgen_geometry_quality_v0.7",
        "units": "millimeters",
        "anatomy": {
            "profile": graph.metadata.get("anatomy_realism_profile"),
            "pelvicalyceal_class": graph.metadata.get("pelvicalyceal_class"),
            "takazawa_type": graph.metadata.get("takazawa_type"),
            "calyx_distribution": graph.metadata.get("calyx_distribution"),
            "anterior_posterior_pairs": {k: sorted(v) for k, v in takazawa_pairs.items()},
            "subtractive_papilla_count": len(subtractive_papillae),
        },
        "lower_pole": {
            "clean_access_enabled": bool(config.lower_pole_clean_access),
            "access_edge_count": len(lower_edges),
            "access_length_mm": lower_access_length,
            "attachment_nodes": lower_attachment_nodes,
            "lower_calyx_count": len([t for t in targets if t.get("region") == "lower"]),
        },
        "clearance": {
            "minimum_non_connected_segment_clearance_mm": min_seg_clearance,
            "minimum_non_connected_segment_pair": min_seg_pair,
            "minimum_cup_center_clearance_mm": min_cup_clearance,
            "minimum_cup_center_pair": min_cup_pair,
        },
        "warnings": warnings,
    }


def write_quality_report(out_dir: str | Path, graph: AnatomyGraph, config: GeneratorConfig) -> Dict[str, str]:
    out_dir = Path(out_dir)
    quality_dir = out_dir / "quality"
    quality_dir.mkdir(parents=True, exist_ok=True)
    report = analyze_geometry_quality(graph, config)
    path = quality_dir / "geometry_quality.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return {"geometry_quality_json": "quality/geometry_quality.json"}
