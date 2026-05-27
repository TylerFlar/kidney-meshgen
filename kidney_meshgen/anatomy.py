from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .config import GeneratorConfig
from .graph import AnatomyGraph, Edge, Node, Primitive
from .sdf import primitive_profile_max_scale


def _rand_range(rng: np.random.Generator, span: Tuple[float, float]) -> float:
    return float(rng.uniform(float(span[0]), float(span[1])))


def _split_calyces(rng: np.random.Generator, n: int) -> Dict[str, int]:
    """Allocate minor calyces across upper/middle/lower poles for the basic profile."""
    raw = rng.multinomial(n - 3, [0.34, 0.33, 0.33]) + 1
    return {"upper": int(raw[0]), "middle": int(raw[1]), "lower": int(raw[2])}


def _normalize(v: np.ndarray, fallback: Tuple[float, float, float] = (1.0, 0.0, 0.0)) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return np.asarray(fallback, dtype=float)
    return v / n


def _cross_section_frame(axis: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    axis = _normalize(axis, (1.0, 0.0, 0.0))
    reference = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(axis, reference))) > 0.88:
        reference = np.array([0.0, 1.0, 0.0], dtype=float)
    u = _normalize(np.cross(axis, reference), (0.0, 1.0, 0.0))
    v = _normalize(np.cross(axis, u), (0.0, 0.0, 1.0))
    theta = float(rng.uniform(0.0, 2.0 * np.pi))
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return _normalize(u * c + v * s), _normalize(-u * s + v * c)


def _tube_profile_kwargs(
    p0: np.ndarray,
    p1: np.ndarray,
    kind: str,
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> Dict:
    """Return optional smooth anatomical tube-profile modifiers.

    These affect the analytic lumen SDF, so they are intentionally low
    amplitude and represented in clearance/collision outputs rather than as
    high-frequency render detail.
    """
    if kind != "infundibulum":
        return {}

    oval_lo, oval_hi = tuple(config.infundibulum_cross_section_ovality)
    narrow_lo, narrow_hi = tuple(config.infundibulum_narrowing_fraction)
    width_lo, width_hi = tuple(config.infundibulum_narrowing_width)
    oval0 = float(rng.uniform(max(0.0, oval_lo), max(0.0, oval_hi)))
    oval1 = float(np.clip(oval0 * rng.uniform(0.72, 1.28), 0.0, max(0.0, oval_hi) * 1.15))
    u, v = _cross_section_frame(p1 - p0, rng)

    # Preserve approximately similar area while making the minimum axis only
    # mildly smaller than the nominal edge radius.
    scale0 = (1.0 + oval0, max(0.72, 1.0 - 0.48 * oval0))
    scale1 = (1.0 + oval1, max(0.72, 1.0 - 0.48 * oval1))
    return {
        "cross_section_u": tuple(float(x) for x in u),
        "cross_section_v": tuple(float(x) for x in v),
        "cross_section_scale0": tuple(float(x) for x in scale0),
        "cross_section_scale1": tuple(float(x) for x in scale1),
        "narrowing_t": float(rng.uniform(0.34, 0.72)),
        "narrowing_width": float(rng.uniform(max(0.05, width_lo), max(0.05, width_hi))),
        "narrowing_fraction": float(rng.uniform(max(0.0, narrow_lo), max(0.0, narrow_hi))),
    }


def _intersect_span(preferred: Tuple[float, float], allowed: Tuple[float, float]) -> Tuple[float, float]:
    lo = max(float(preferred[0]), float(allowed[0]))
    hi = min(float(preferred[1]), float(allowed[1]))
    if hi <= lo:
        return float(allowed[0]), float(allowed[1])
    return lo, hi


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


@dataclass
class _CalyxSpec:
    label: str
    level: str
    ap: str
    takazawa_short: str
    angle_deg: Tuple[float, float]
    length_mm: Tuple[float, float]
    y_center_mm: float
    y_jitter_mm: float
    infundibulum_radius_mm: Tuple[float, float]
    cup_radius_mm: Tuple[float, float]
    accessory_index: int = 0


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
        if (-d + b) < 0.0:
            sN = 0.0
        elif (-d + b) > a:
            sN = sD
        else:
            sN = -d + b
            sD = a

    sc = 0.0 if abs(sN) < eps else sN / sD
    tc = 0.0 if abs(tN) < eps else tN / tD
    dP = w0 + sc * u - tc * v
    return float(np.linalg.norm(dP))


def _ellipsoid_scaled_norm(pt: np.ndarray, center: np.ndarray, radii: np.ndarray) -> float:
    return float(np.linalg.norm((pt - center) / np.maximum(radii, 1e-8)))


def _pelvis_alias(value: str) -> str:
    token = value.strip().lower().replace("-", "_")
    aliases = {
        "i": "type_i",
        "1": "type_i",
        "type1": "type_i",
        "type_1": "type_i",
        "type_i": "type_i",
        "single": "type_i",
        "true": "type_i",
        "ii": "type_ii",
        "2": "type_ii",
        "type2": "type_ii",
        "type_2": "type_ii",
        "type_ii": "type_ii",
        "divided": "type_ii",
        "bifurcated": "type_ii",
    }
    return aliases.get(token, token)


def _resolve_pelvicalyceal_class(
    config: GeneratorConfig,
    rng: np.random.Generator,
    profile: str,
) -> Tuple[str, str, str]:
    requested_pelvis = _pelvis_alias(str(getattr(config, "pelvis_type", "random")))
    requested_class = _pelvis_alias(str(getattr(config, "pelvicalyceal_class", "random")))

    if requested_pelvis in {"type_i", "type_ii"}:
        pelvicalyceal_class = requested_pelvis
    elif requested_class in {"type_i", "type_ii"}:
        pelvicalyceal_class = requested_class
    else:
        probs = [0.58, 0.42] if profile == "takazawa" else [0.62, 0.38]
        pelvicalyceal_class = str(rng.choice(["type_i", "type_ii"], p=probs))

    pelvis_type = "single" if pelvicalyceal_class == "type_i" else "divided"
    if pelvicalyceal_class == "type_ii":
        return pelvis_type, pelvicalyceal_class, "type_ii"

    subtype = str(getattr(config, "type_i_subtype", "random")).strip().lower().replace("-", "_")
    subtype = {"a": "ia", "b": "ib", "c": "ic", "type_ia": "ia", "type_ib": "ib", "type_ic": "ic"}.get(subtype, subtype)
    if subtype not in {"ia", "ib", "ic"}:
        subtype = str(rng.choice(["ia", "ib", "ic"], p=[43.0 / 58.0, 4.0 / 58.0, 11.0 / 58.0]))
    return pelvis_type, pelvicalyceal_class, f"type_I{subtype[-1]}"


def _sample_pelvis_radii(
    config: GeneratorConfig,
    rng: np.random.Generator,
    profile: str,
    pelvicalyceal_class: str,
    takazawa_subtype: str,
) -> np.ndarray:
    radii = np.array(
        [
            _rand_range(rng, config.pelvis_radius_x_mm),
            _rand_range(rng, config.pelvis_radius_y_mm),
            _rand_range(rng, config.pelvis_radius_z_mm),
        ],
        dtype=float,
    )
    if profile != "takazawa":
        return radii
    if pelvicalyceal_class == "type_ii":
        return radii * np.array([0.86, 0.9, 0.88], dtype=float)
    if takazawa_subtype == "type_Ib":
        return radii * np.array([1.18, 1.24, 1.08], dtype=float)
    if takazawa_subtype == "type_Ic":
        return radii * np.array([0.74, 0.68, 0.88], dtype=float)
    return radii


def _sample_takazawa_calyx_count(config: GeneratorConfig, rng: np.random.Generator) -> int:
    lo = max(7, int(config.calyx_count_min))
    hi = max(lo, int(config.calyx_count_max))
    choices = np.array([7, 8, 9, 10, 11, 12, 13], dtype=int)
    probs = np.array([0.30, 0.51, 0.09, 0.05, 0.025, 0.015, 0.01], dtype=float)
    mask = (choices >= lo) & (choices <= hi)
    if not np.any(mask):
        return int(rng.integers(lo, hi + 1))
    choices = choices[mask]
    probs = probs[mask]
    probs = probs / probs.sum()
    return int(rng.choice(choices, p=probs))


def _level_distribution(
    level: str,
    config: GeneratorConfig,
    lower_splay_range: Tuple[float, float],
    lower_branch_range: Tuple[float, float],
    lower_y_span: float,
    lower_access: bool,
) -> Dict[str, Tuple[float, float] | float]:
    cup_span = tuple(config.calyx_cup_radius_mm)
    radius_span = tuple(config.infundibulum_radius_mm)
    table = {
        "top": {
            "angle": (58.0, 82.0),
            "length": (9.0, 19.0),
            "radius": _intersect_span((1.25, 2.25), radius_span),
            "cup": _intersect_span((2.7, 4.8), cup_span),
            "y": 0.0,
            "jitter": 1.8,
        },
        "upper": {
            "angle": (26.0, 58.0),
            "length": (12.0, 25.0),
            "radius": _intersect_span((1.35, 2.55), radius_span),
            "cup": _intersect_span((3.0, 5.2), cup_span),
            "y": 6.6,
            "jitter": 1.8,
        },
        "middle": {
            "angle": (-10.0, 16.0),
            "length": (11.0, 24.0),
            "radius": _intersect_span((1.45, 2.7), radius_span),
            "cup": _intersect_span((3.0, 5.4), cup_span),
            "y": 7.5,
            "jitter": 2.0,
        },
        "lower": {
            "angle": lower_splay_range if lower_access else (-52.0, -18.0),
            "length": lower_branch_range if lower_access else (16.0, 31.0),
            "radius": _intersect_span((1.25, 2.55), radius_span),
            "cup": _intersect_span((3.1, 5.6), cup_span),
            "y": lower_y_span * 0.72 if lower_access else 7.1,
            "jitter": 2.0,
        },
        "bottom": {
            "angle": (-34.0, -8.0) if lower_access else (-78.0, -48.0),
            "length": (8.0, 18.0),
            "radius": _intersect_span((1.2, 2.35), radius_span),
            "cup": _intersect_span((2.7, 5.0), cup_span),
            "y": 0.0,
            "jitter": 1.8,
        },
    }
    return table[level]


def _make_calyx_spec(
    label: str,
    level: str,
    ap: str,
    short_name: str,
    accessory_index: int,
    config: GeneratorConfig,
    lower_splay_range: Tuple[float, float],
    lower_branch_range: Tuple[float, float],
    lower_y_span: float,
    lower_access: bool,
) -> _CalyxSpec:
    dist = _level_distribution(level, config, lower_splay_range, lower_branch_range, lower_y_span, lower_access)
    y_abs = float(dist["y"])
    y_center = 0.0
    if ap == "anterior":
        y_center = y_abs
    elif ap == "posterior":
        y_center = -y_abs
    if accessory_index:
        y_center += (1.4 + 0.45 * accessory_index) * (1.0 if y_center >= 0.0 else -1.0)
    return _CalyxSpec(
        label=label,
        level=level,
        ap=ap,
        takazawa_short=short_name,
        angle_deg=dist["angle"],  # type: ignore[arg-type]
        length_mm=dist["length"],  # type: ignore[arg-type]
        y_center_mm=float(y_center),
        y_jitter_mm=float(dist["jitter"]),
        infundibulum_radius_mm=dist["radius"],  # type: ignore[arg-type]
        cup_radius_mm=dist["cup"],  # type: ignore[arg-type]
        accessory_index=accessory_index,
    )


def _make_takazawa_calyx_specs(
    config: GeneratorConfig,
    rng: np.random.Generator,
    lower_splay_range: Tuple[float, float],
    lower_branch_range: Tuple[float, float],
    lower_y_span: float,
    lower_access: bool,
) -> List[_CalyxSpec]:
    n = _sample_takazawa_calyx_count(config, rng)
    base = [
        ("calyx_top", "top", "", "T", 0),
        ("calyx_upper_anterior", "upper", "anterior", "UA", 0),
        ("calyx_upper_posterior", "upper", "posterior", "UP", 0),
        ("calyx_middle_anterior", "middle", "anterior", "MA", 0),
        ("calyx_middle_posterior", "middle", "posterior", "MP", 0),
        ("calyx_lower_anterior", "lower", "anterior", "LA", 0),
        ("calyx_lower_posterior", "lower", "posterior", "LP", 0),
        ("calyx_bottom", "bottom", "", "B", 0),
    ]
    if n == 7:
        drop = "calyx_middle_posterior" if rng.random() < 0.5 else "calyx_middle_anterior"
        base = [item for item in base if item[0] != drop]

    accessory_counts = {"upper": 0, "middle": 0, "lower": 0}
    for _ in range(max(0, n - 8)):
        level = str(rng.choice(["upper", "middle", "lower"], p=[0.3, 0.35, 0.35]))
        ap = str(rng.choice(["anterior", "posterior"]))
        accessory_counts[level] += 1
        short = f"{level[0].upper()}{ap[0].upper()}{accessory_counts[level] + 1}"
        label = f"calyx_{level}_{ap}_accessory_{accessory_counts[level]:02d}"
        base.append((label, level, ap, short, accessory_counts[level]))

    level_order = {"top": 0, "upper": 1, "middle": 2, "lower": 3, "bottom": 4}
    ap_order = {"": 0, "anterior": 1, "posterior": 2}
    base.sort(key=lambda item: (level_order[item[1]], ap_order[item[2]], item[4], item[0]))
    return [
        _make_calyx_spec(
            label,
            level,
            ap,
            short,
            accessory_index,
            config,
            lower_splay_range,
            lower_branch_range,
            lower_y_span,
            lower_access,
        )
        for label, level, ap, short, accessory_index in base
    ]


def _make_basic_calyx_specs(
    config: GeneratorConfig,
    rng: np.random.Generator,
    lower_splay_range: Tuple[float, float],
    lower_branch_range: Tuple[float, float],
    lower_y_span: float,
    lower_access: bool,
) -> List[_CalyxSpec]:
    n = int(rng.integers(int(config.calyx_count_min), int(config.calyx_count_max) + 1))
    groups = _split_calyces(rng, n)
    specs: List[_CalyxSpec] = []
    for level in ["upper", "middle", "lower"]:
        count = groups[level]
        for i in range(count):
            ap = "anterior" if i % 2 == 0 else "posterior"
            label = f"calyx_{level}_{i + 1:02d}"
            specs.append(
                _make_calyx_spec(
                    label,
                    level,
                    ap,
                    f"{level[0].upper()}{i + 1}",
                    0,
                    config,
                    lower_splay_range,
                    lower_branch_range,
                    lower_y_span,
                    lower_access,
                )
            )
    return specs


def _calyx_distribution(calyx_specs: List[_CalyxSpec]) -> Dict[str, int]:
    distribution = {"top": 0, "upper": 0, "middle": 0, "lower": 0, "bottom": 0}
    for spec in calyx_specs:
        distribution[spec.level] = distribution.get(spec.level, 0) + 1
    return {k: v for k, v in distribution.items() if v > 0}


def _branch_family(level: str, pelvicalyceal_class: str = "type_i") -> str:
    if level in {"top", "upper"}:
        return "upper_pole"
    if pelvicalyceal_class == "type_ii" and level == "middle":
        return "lower_pole"
    if level in {"lower", "bottom"}:
        return "lower_pole"
    return "middle_pole"


def build_anatomy_graph(config: GeneratorConfig) -> AnatomyGraph:
    rng = np.random.default_rng(config.seed)

    side_sign = 1.0 if config.side.lower() == "right" else -1.0
    profile = str(getattr(config, "anatomy_realism_profile", "takazawa")).strip().lower()
    if profile not in {"takazawa", "legacy", "basic"}:
        profile = "takazawa"
    pelvis_type, pelvicalyceal_class, takazawa_subtype = _resolve_pelvicalyceal_class(config, rng, profile)

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

    lower_access_enabled = bool(config.lower_pole_clean_access)
    if profile == "takazawa":
        calyx_specs = _make_takazawa_calyx_specs(
            config,
            rng,
            lower_splay_range,
            lower_branch_range,
            lower_y_span,
            lower_access_enabled,
        )
    else:
        calyx_specs = _make_basic_calyx_specs(
            config,
            rng,
            lower_splay_range,
            lower_branch_range,
            lower_y_span,
            lower_access_enabled,
        )
    if pelvicalyceal_class == "type_ii":
        for spec in calyx_specs:
            if spec.level == "middle":
                spec.angle_deg = (-24.0, 2.0)

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

    def add_edge(
        edge_id: str,
        source: str,
        target: str,
        r0: float,
        r1: float,
        kind: str,
        region: str,
        parent: str = "",
    ) -> None:
        edges.append(Edge(edge_id, source, target, float(r0), float(r1), kind, region, parent))
        nmap = node_lookup()
        p0 = np.asarray(nmap[source].position_mm, dtype=float)
        p1 = np.asarray(nmap[target].position_mm, dtype=float)
        profile_kwargs = _tube_profile_kwargs(p0, p1, kind, config, rng)
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
                **profile_kwargs,
            )
        )
        segment_geoms.append(
            _SegmentGeom(
                p0=p0,
                p1=p1,
                radius_mm=max(float(r0), float(r1)) * primitive_profile_max_scale(profile_kwargs),
                segment_type=kind,
                source_node=source,
                target_node=target,
            )
        )

    pelvis_center = np.array([0.0, 0.0, 0.0], dtype=float)
    upj = np.array([-4.0 * side_sign, 0.0, -10.5], dtype=float)
    add_node("pelvis_center", pelvis_center, "pelvis", "renal_pelvis")
    add_node("upj", upj, "junction", "upj")

    pelvis_radii = _sample_pelvis_radii(config, rng, profile, pelvicalyceal_class, takazawa_subtype)
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
            add_edge(f"edge_ureter_{i - 1:02d}", last_id, node_id, ur, r_next, "ureter", "ureter_stub")
            ur = r_next
            last_id = node_id
        add_edge(f"edge_ureter_{seg_count - 1:02d}", last_id, "upj", ur, upj_radius, "ureter", "ureter_stub")

    # Major branches and regional roots. In Type II, top/upper calyces drain
    # through the upper branch, while middle/lower/bottom drain through a lower branch.
    region_roots: Dict[str, str] = {}
    roots_needed = {spec.level for spec in calyx_specs}
    lower_access_specs = [spec for spec in calyx_specs if spec.level in {"lower", "bottom"}]
    if lower_access_enabled and lower_access_specs:
        roots_needed.discard("bottom")
        roots_needed.add("lower")

    def add_level_root(level: str, parent_id: str, offset: np.ndarray, radius_range: Tuple[float, float]) -> None:
        region_name = f"major_{level}" if pelvis_type == "divided" else f"outlet_{level}"
        node_id = f"{region_name}"
        pos = pelvis_center + offset + np.array([0.0, rng.uniform(-0.8, 0.8), rng.uniform(-0.9, 0.9)])
        add_node(node_id, pos, "major_calyx", region_name)
        parent_radius = max(pelvis_radii[0] * (0.34 if pelvis_type == "single" else 0.28), 2.8)
        if parent_id != "pelvis_center":
            parent_radius = _rand_range(rng, (2.1, 3.4))
        r1 = _rand_range(rng, radius_range)
        add_edge(
            f"edge_{parent_id}_{node_id}",
            parent_id,
            node_id,
            parent_radius,
            r1,
            "major_calyx",
            region_name,
            parent=_branch_family(level, pelvicalyceal_class),
        )
        region_roots[level] = node_id

    if pelvis_type == "divided":
        branch_radius = _rand_range(rng, (3.2, 4.9))
        upper_branch = pelvis_center + np.array([4.7 * side_sign, -0.6, 6.8], dtype=float)
        lower_branch = pelvis_center + np.array([4.9 * side_sign, 0.7, -2.2], dtype=float)
        add_node("typeII_upper_branch", upper_branch, "pelvic_branch", "major_upper_branch")
        add_node("typeII_lower_branch", lower_branch, "pelvic_branch", "major_lower_branch")
        add_edge(
            "edge_pelvis_typeII_upper_branch",
            "pelvis_center",
            "typeII_upper_branch",
            max(branch_radius, 3.7),
            branch_radius * rng.uniform(0.74, 0.92),
            "major_calyx",
            "major_upper_branch",
            parent="upper_pole",
        )
        add_edge(
            "edge_pelvis_typeII_lower_branch",
            "pelvis_center",
            "typeII_lower_branch",
            max(branch_radius, 3.7),
            branch_radius * rng.uniform(0.78, 0.98),
            "major_calyx",
            "major_lower_branch",
            parent="lower_pole",
        )
        offsets = {
            "top": np.array([7.0 * side_sign, -0.3, 12.0]),
            "upper": np.array([8.8 * side_sign, -0.2, 8.4]),
            "middle": np.array([8.7 * side_sign, 0.2, -1.6]),
            "lower": np.array([8.3 * side_sign, 0.6, -6.2]),
            "bottom": np.array([6.6 * side_sign, 0.4, -10.2]),
        }
        for level in ["top", "upper", "middle", "lower", "bottom"]:
            if level not in roots_needed:
                continue
            parent = "typeII_upper_branch" if level in {"top", "upper"} else "typeII_lower_branch"
            add_level_root(level, parent, offsets[level], (2.4, 4.2))
    else:
        offsets = {
            "top": np.array([2.5 * side_sign, -0.3, 10.2]),
            "upper": np.array([4.0 * side_sign, -0.4, 6.4]),
            "middle": np.array([4.9 * side_sign, 0.0, 0.2]),
            "lower": np.array([4.0 * side_sign, 0.5, -6.8]),
            "bottom": np.array([2.5 * side_sign, 0.3, -10.6]),
        }
        for level in ["top", "upper", "middle", "lower", "bottom"]:
            if level not in roots_needed:
                continue
            add_level_root(level, "pelvis_center", offsets[level], (2.3, 4.0))

    lower_attachment_ids: List[str] = []
    lower_spec_root: Dict[str, str] = {}
    if lower_access_enabled and lower_access_specs:
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
        attach_count = max(len(lower_access_specs), 2)
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
                parent="lower_pole",
            )
            lower_attachment_ids.append(node_id)
            prev_id = node_id
            prev_r = this_r
        for idx, spec in enumerate(lower_access_specs):
            lower_spec_root[spec.label] = lower_attachment_ids[min(idx, len(lower_attachment_ids) - 1)]

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

    for calyx_index, spec in enumerate(calyx_specs, start=1):
        root_id = lower_spec_root.get(spec.label) or region_roots[spec.level]
        root_pos = np.array(node_lookup()[root_id].position_mm, dtype=float)
        label_to_id(spec.label)

        best_candidate = None
        best_score = -np.inf
        for _attempt in range(int(config.branch_sample_attempts)):
            reach = rng.uniform(*spec.length_mm)
            angle_deg = float(rng.uniform(*spec.angle_deg) + rng.normal(0.0, 4.0))
            angle = np.deg2rad(angle_deg)
            xoff = side_sign * (reach * np.cos(angle) + rng.uniform(-0.85, 0.85))
            zoff = reach * np.sin(angle) + rng.uniform(-1.0, 1.0)
            yoff = spec.y_center_mm + rng.normal(0.0, spec.y_jitter_mm)
            if spec.ap:
                yoff = float(np.clip(yoff, spec.y_center_mm - spec.y_jitter_mm * 1.35, spec.y_center_mm + spec.y_jitter_mm * 1.35))
            endpoint = root_pos + np.array([xoff, yoff, zoff], dtype=float)

            if spec.level in {"lower", "bottom"} and lower_attachment_ids:
                endpoint[2] -= rng.uniform(0.25, 2.0)
                endpoint[1] += rng.uniform(-1.0, 1.0)

            neck_frac = rng.uniform(0.50, 0.74)
            neck = root_pos * (1 - neck_frac) + endpoint * neck_frac
            neck += np.array(
                [side_sign * rng.uniform(-0.75, 0.75), rng.uniform(-1.5, 1.5), rng.uniform(-1.3, 1.3)],
                dtype=float,
            )

            if spec.level in {"lower", "bottom"} and lower_attachment_ids:
                r_start = _rand_range(rng, _intersect_span((1.25, 2.45), spec.infundibulum_radius_mm))
                r_neck = min(_rand_range(rng, spec.infundibulum_radius_mm), r_start * rng.uniform(0.82, 1.05))
                r_cup_tube = max(1.1, min(_rand_range(rng, (1.25, 2.35)), r_neck * 1.1))
            else:
                r_start_span = (1.6, 3.0) if pelvicalyceal_class == "type_ii" and spec.level == "middle" else (1.9, 3.5)
                r_start = _rand_range(rng, r_start_span)
                r_neck = _rand_range(rng, spec.infundibulum_radius_mm)
                r_cup_tube = max(1.25, min(_rand_range(rng, (1.4, 2.8)), r_neck * 1.18))
            cup_r = _rand_range(rng, spec.cup_radius_mm)
            profile_clearance_scale = 1.0 + max(0.0, float(config.infundibulum_cross_section_ovality[1]))
            prox = (root_pos, neck, max(r_start, r_neck) * profile_clearance_scale, root_id)
            dist = (neck, endpoint, max(r_neck, r_cup_tube) * profile_clearance_scale, root_id)

            if candidate_ok(endpoint, cup_r, prox, dist):
                clearance = [float(np.linalg.norm(cup.center - endpoint)) - (cup.radius_mm + cup_r) for cup in cup_geoms]
                score = min(clearance) if clearance else 100.0
                score += abs(endpoint[2] - root_pos[2]) * 0.22
                score += abs(endpoint[1] - root_pos[1]) * 0.12
                if score > best_score:
                    best_candidate = (neck, endpoint, r_start, r_neck, r_cup_tube, cup_r, angle_deg)
                    best_score = score

        if best_candidate is None:
            fallback_reach = float(np.mean(spec.length_mm))
            fallback_angle_deg = float(np.mean(spec.angle_deg))
            fallback_angle = np.deg2rad(fallback_angle_deg)
            endpoint = root_pos + np.array(
                [
                    side_sign * fallback_reach * np.cos(fallback_angle),
                    float(spec.y_center_mm),
                    fallback_reach * np.sin(fallback_angle),
                ],
                dtype=float,
            )
            for cup in cup_geoms:
                sep_vec = endpoint - cup.center
                sep = float(np.linalg.norm(sep_vec))
                min_sep = cup.radius_mm + 3.5 + max(config.cup_center_clearance_mm * 0.5, 2.0)
                if sep < min_sep and sep > 1e-6:
                    endpoint = cup.center + sep_vec / sep * min_sep
            neck = root_pos * 0.42 + endpoint * 0.58
            best_candidate = (neck, endpoint, 2.8, 2.0, 1.7, 3.2, fallback_angle_deg)

        neck, endpoint, r_start, r_neck, r_cup_tube, cup_r, angle_deg = best_candidate
        neck_id = f"{spec.label}_neck"
        cup_id = f"{spec.label}_cup"
        add_node(neck_id, neck, "infundibulum", spec.label)
        add_node(cup_id, endpoint, "calyx_cup", spec.label)

        add_edge(
            f"edge_{spec.label}_proximal",
            root_id,
            neck_id,
            r_start,
            r_neck,
            "infundibulum",
            spec.label,
            parent=_branch_family(spec.level, pelvicalyceal_class),
        )
        add_edge(
            f"edge_{spec.label}_distal",
            neck_id,
            cup_id,
            r_neck,
            r_cup_tube,
            "infundibulum",
            spec.label,
            parent=_branch_family(spec.level, pelvicalyceal_class),
        )

        axis = _normalize(endpoint - neck, (side_sign, 0.0, 0.0))
        cup_radii = (
            cup_r * rng.uniform(0.92, 1.08),
            cup_r * rng.uniform(0.82, 1.02),
            cup_r * rng.uniform(0.82, 1.06),
        )
        primitives.append(
            Primitive(
                id=f"{spec.label}_fornix_cup_ellipsoid",
                kind="ellipsoid",
                label=spec.label,
                label_id=label_to_id(spec.label),
                center=tuple(endpoint),
                radii=tuple(float(v) for v in cup_radii),
            )
        )

        papilla_info = None
        if bool(getattr(config, "papilla_fornix_enabled", True)):
            cup_max = float(max(cup_radii))
            base_offset = cup_max * _rand_range(rng, tuple(config.fornix_depth_fraction))
            papilla_len = min(_rand_range(rng, config.papilla_length_mm), cup_max * 0.84)
            papilla_radius = min(_rand_range(rng, config.papilla_radius_mm), cup_max * 0.36, max(r_cup_tube * 0.92, 0.65))
            base = endpoint + axis * base_offset
            tip = endpoint + axis * (base_offset - papilla_len)
            add_node(f"{spec.label}_papilla_base", base, "papilla_base", spec.label)
            add_node(f"{spec.label}_papilla_tip", tip, "papilla_tip", spec.label)
            primitives.append(
                Primitive(
                    id=f"{spec.label}_papilla_solid",
                    kind="tapered_capsule",
                    label=spec.label,
                    label_id=label_to_id(spec.label),
                    operation="subtract",
                    p0=tuple(base),
                    p1=tuple(tip),
                    r0=float(papilla_radius * 1.12),
                    r1=float(max(papilla_radius * 0.42, 0.28)),
                )
            )
            papilla_info = {
                "base_mm": [float(v) for v in base],
                "tip_mm": [float(v) for v in tip],
                "radius_mm": float(papilla_radius),
                "length_mm": float(np.linalg.norm(base - tip)),
                "fornix_depth_mm": float(base_offset),
            }

        cup_geoms.append(_CupGeom(center=np.asarray(endpoint, dtype=float), radius_mm=float(max(cup_radii)), label=spec.label))
        calyx_targets.append(
            {
                "id": spec.label,
                "region": spec.level,
                "level": spec.level,
                "anterior_posterior": spec.ap or None,
                "takazawa_name": spec.takazawa_short,
                "accessory_index": int(spec.accessory_index),
                "cup_node": cup_id,
                "center_mm": [float(v) for v in endpoint],
                "approx_radius_mm": float(cup_r),
                "cup_radii_mm": [float(v) for v in cup_radii],
                "infundibulum_length_mm": float(np.linalg.norm(root_pos - neck) + np.linalg.norm(neck - endpoint)),
                "infundibulum_radius_mm": float(min(r_neck, r_cup_tube)),
                "branch_angle_degrees": float(angle_deg),
                "root_node": root_id,
                "papilla": papilla_info,
                "ordinal": calyx_index,
            }
        )

    metadata = {
        "anatomy_realism_profile": profile,
        "pelvicalyceal_class": pelvicalyceal_class,
        "pelvicalyceal_class_description": "single pelvis" if pelvicalyceal_class == "type_i" else "divided pelvis",
        "takazawa_type": takazawa_subtype,
        "calyx_distribution": _calyx_distribution(calyx_specs),
        "calyx_naming": "top/upper/middle/lower/bottom; upper/middle/lower preferentially anterior/posterior pairs",
        "anterior_posterior_axis": "positive_y_anterior_negative_y_posterior",
        "papilla_fornix_enabled": bool(getattr(config, "papilla_fornix_enabled", True)),
        "lower_pole_access_mode": lower_pole_mode if lower_attachment_ids else "direct_branch",
        "lower_pole_attachment_nodes": lower_attachment_ids,
        "lower_pole_distribution_basis": {
            "easy": "shorter, wider, flatter lower infundibulum",
            "intermediate": "configured lower_pole_angle_degrees and lower_access_* spans",
            "hard": "longer, narrower, steeper lower infundibulum",
        },
        "research_basis": [
            "Takazawa et al. Type I single pelvis / Type II divided pelvis and T/U/M/L/B calyx names.",
            "Takazawa CT-urography data: 8 calyces most common, 7 second most common.",
            "Lower-pole access spans use published IL/IW/IUA concepts and unfavorable narrow/long/steep thresholds.",
            "Papilla-fornix cups model cup-shaped minor calyces surrounding renal papillae.",
            "Infundibular profiles include mild non-circular cross sections and local constrictions to reflect reported width/asymmetry variation.",
            "Visual-only mucosal folds/noise are separated from the smooth collision/SDF layer.",
        ],
        "coordinate_notes": "x=lateral, y=anterior/posterior, z=cranio-caudal; values are millimeters",
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
