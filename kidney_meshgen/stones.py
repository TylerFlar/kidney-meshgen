from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import trimesh

from .config import GeneratorConfig
from .graph import AnatomyGraph

STONE_LABEL_ID = 1001

STONE_RESEARCH_BASIS = [
    {
        "topic": "stone_composition_and_gross_morphology",
        "note": (
            "Visual classes follow clinical descriptions of calcium oxalate monohydrate/dihydrate, "
            "calcium phosphate, uric acid, struvite, and cystine stones: COM tends to be dark, hard, "
            "and smoother; COD is lighter, brittle, and jagged; uric acid is yellow/orange/red-brown; "
            "struvite/phosphate stones are pale/chalky and brittle; cystine is amber-yellow and waxy."
        ),
        "sources": [
            "https://www.ncbi.nlm.nih.gov/sites/books/NBK442014/",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC9818792/",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC5418413/",
        ],
    },
    {
        "topic": "crystal_forms_and_surface_texture",
        "note": (
            "The material classes encode the clinically useful crystal distinctions: whewellite/COM, "
            "weddellite/COD, uric acid phases, struvite with apatite/carbonate apatite mixtures, and "
            "cystine hexagonal crystallites."
        ),
        "sources": [
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC5685519/",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC11122214/",
            "https://www.mdpi.com/2073-4352/14/3/238",
        ],
    },
    {
        "topic": "laser_fragmentation_and_dusting",
        "note": (
            "The laser-fragmented gravel state models URS lithotripsy products ranging from extractable "
            "fragments to submillimeter dust/gravel after fragmentation, dusting, and pop-dusting."
        ),
        "sources": [
            "https://www.frontiersin.org/journals/surgery/articles/10.3389/fsurg.2017.00057/full",
            "https://pubmed.ncbi.nlm.nih.gov/30315636/",
            "https://pubmed.ncbi.nlm.nih.gov/30648761/",
        ],
    },
]


STONE_MATERIAL_CLASSES: Dict[str, Dict[str, Any]] = {
    "COM": {
        "display_name": "calcium oxalate monohydrate",
        "mineral": "whewellite",
        "visual_traits": "dark brown, very hard, smoother rounded surface with subtle laminated relief",
        "palette": [
            [0.16, 0.095, 0.055, 1.0],
            [0.24, 0.135, 0.075, 1.0],
            [0.34, 0.205, 0.115, 1.0],
            [0.11, 0.075, 0.055, 1.0],
        ],
        "roughness": [0.48, 0.72],
        "specular": [0.16, 0.32],
        "crystal_bump_strength": [0.035, 0.09],
        "crystal_bump_distance_mm": [0.07, 0.18],
        "crystal_bump_scale": [70.0, 125.0],
        "crystal_bump_detail": [8.0, 13.0],
        "color_jitter": 0.10,
        "color_variation_strength": [0.10, 0.22],
        "color_variation_scale": [18.0, 44.0],
        "shape_irregularity_scale": 0.55,
        "surface_granularity": 0.025,
        "spike_strength": 0.012,
        "fracture_planes": [0, 3],
        "fracture_plane_strength": [0.40, 0.72],
        "fragment_angularity": 0.55,
        "fragment_volume_retention": [0.72, 0.94],
    },
    "COD": {
        "display_name": "calcium oxalate dihydrate",
        "mineral": "weddellite",
        "visual_traits": "yellow to light brown, brittle, jagged, sharper crystalline surface",
        "palette": [
            [0.70, 0.55, 0.29, 1.0],
            [0.84, 0.68, 0.38, 1.0],
            [0.58, 0.42, 0.22, 1.0],
            [0.88, 0.78, 0.52, 1.0],
        ],
        "roughness": [0.62, 0.86],
        "specular": [0.16, 0.30],
        "crystal_bump_strength": [0.10, 0.21],
        "crystal_bump_distance_mm": [0.12, 0.34],
        "crystal_bump_scale": [95.0, 190.0],
        "crystal_bump_detail": [10.0, 15.0],
        "color_jitter": 0.13,
        "color_variation_strength": [0.16, 0.34],
        "color_variation_scale": [26.0, 62.0],
        "shape_irregularity_scale": 1.10,
        "surface_granularity": 0.070,
        "spike_strength": 0.055,
        "fracture_planes": [3, 8],
        "fracture_plane_strength": [0.70, 1.0],
        "fragment_angularity": 0.95,
        "fragment_volume_retention": [0.62, 0.86],
    },
    "uric_acid": {
        "display_name": "uric acid",
        "mineral": "uricite/uric acid phases",
        "visual_traits": "yellow, orange, reddish, or brown; often smoother and waxy-to-glossy",
        "palette": [
            [0.86, 0.52, 0.16, 1.0],
            [0.96, 0.72, 0.24, 1.0],
            [0.62, 0.28, 0.09, 1.0],
            [0.48, 0.17, 0.07, 1.0],
        ],
        "roughness": [0.34, 0.58],
        "specular": [0.24, 0.44],
        "crystal_bump_strength": [0.030, 0.080],
        "crystal_bump_distance_mm": [0.05, 0.15],
        "crystal_bump_scale": [48.0, 105.0],
        "crystal_bump_detail": [6.0, 11.0],
        "color_jitter": 0.14,
        "color_variation_strength": [0.12, 0.30],
        "color_variation_scale": [14.0, 38.0],
        "shape_irregularity_scale": 0.62,
        "surface_granularity": 0.030,
        "spike_strength": 0.010,
        "fracture_planes": [1, 4],
        "fracture_plane_strength": [0.45, 0.82],
        "fragment_angularity": 0.62,
        "fragment_volume_retention": [0.70, 0.92],
    },
    "struvite_apatite": {
        "display_name": "struvite / carbonate apatite",
        "mineral": "magnesium ammonium phosphate with calcium phosphate apatite",
        "visual_traits": "off-white, gray-white, yellowish, chalky, granular, brittle",
        "palette": [
            [0.88, 0.86, 0.72, 1.0],
            [0.74, 0.72, 0.62, 1.0],
            [0.58, 0.55, 0.46, 1.0],
            [0.93, 0.91, 0.82, 1.0],
        ],
        "roughness": [0.78, 0.96],
        "specular": [0.06, 0.20],
        "crystal_bump_strength": [0.12, 0.25],
        "crystal_bump_distance_mm": [0.12, 0.38],
        "crystal_bump_scale": [75.0, 175.0],
        "crystal_bump_detail": [8.0, 14.0],
        "color_jitter": 0.09,
        "color_variation_strength": [0.18, 0.40],
        "color_variation_scale": [32.0, 88.0],
        "shape_irregularity_scale": 0.95,
        "surface_granularity": 0.105,
        "spike_strength": 0.028,
        "fracture_planes": [2, 7],
        "fracture_plane_strength": [0.66, 0.96],
        "fragment_angularity": 0.86,
        "fragment_volume_retention": [0.55, 0.82],
    },
    "cystine": {
        "display_name": "cystine",
        "mineral": "cystine",
        "visual_traits": "amber, tan, or yellow with waxy surface and hexagonal crystallite relief",
        "palette": [
            [0.86, 0.67, 0.26, 1.0],
            [0.72, 0.53, 0.18, 1.0],
            [0.93, 0.78, 0.36, 1.0],
            [0.58, 0.56, 0.28, 1.0],
        ],
        "roughness": [0.42, 0.68],
        "specular": [0.28, 0.50],
        "crystal_bump_strength": [0.065, 0.145],
        "crystal_bump_distance_mm": [0.08, 0.24],
        "crystal_bump_scale": [55.0, 140.0],
        "crystal_bump_detail": [8.0, 12.0],
        "color_jitter": 0.10,
        "color_variation_strength": [0.10, 0.26],
        "color_variation_scale": [18.0, 50.0],
        "shape_irregularity_scale": 0.72,
        "surface_granularity": 0.050,
        "spike_strength": 0.022,
        "fracture_planes": [2, 6],
        "fracture_plane_strength": [0.52, 0.88],
        "fragment_angularity": 0.72,
        "fragment_volume_retention": [0.66, 0.90],
    },
}

STONE_MATERIAL_ALIASES = {
    "com": "COM",
    "calcium_oxalate_monohydrate": "COM",
    "whewellite": "COM",
    "cod": "COD",
    "calcium_oxalate_dihydrate": "COD",
    "weddellite": "COD",
    "uric": "uric_acid",
    "uric_acid": "uric_acid",
    "urate": "uric_acid",
    "struvite": "struvite_apatite",
    "apatite": "struvite_apatite",
    "struvite_apatite": "struvite_apatite",
    "struvite/apatite": "struvite_apatite",
    "infection": "struvite_apatite",
    "phosphate": "struvite_apatite",
    "cystine": "cystine",
}


@dataclass
class StoneInfo:
    id: str
    calyx_id: str
    region: str
    center_mm: Tuple[float, float, float]
    radius_mm: float
    mesh_file: str
    material_class: str
    state: str
    label_id: int = STONE_LABEL_ID
    color_rgba: Tuple[float, float, float, float] = (0.5, 0.4, 0.25, 1.0)
    roughness: float = 0.7
    specular: float = 0.2
    crystal_bump_strength: float = 0.08
    crystal_bump_distance_mm: float = 0.15
    crystal_bump_scale: float = 90.0
    fracture_plane_count: int = 0
    fragment_count: int = 1
    fragment_radius_mm: Tuple[float, float] = (0.0, 0.0)
    gravel_spread_mm: float = 0.0
    render_profile: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


def normalize_stone_material_class(name: str) -> str:
    key = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    if key in STONE_MATERIAL_CLASSES:
        return key
    if key in STONE_MATERIAL_ALIASES:
        return STONE_MATERIAL_ALIASES[key]
    valid = ", ".join(STONE_MATERIAL_CLASSES)
    raise ValueError(f"Unknown stone material class {name!r}. Expected one of: {valid}")


def stone_material_class_summary() -> Dict[str, Dict[str, Any]]:
    return {
        name: {
            "display_name": str(profile["display_name"]),
            "mineral": str(profile["mineral"]),
            "visual_traits": str(profile["visual_traits"]),
        }
        for name, profile in STONE_MATERIAL_CLASSES.items()
    }


def _span(values: Sequence[Any], cast=float) -> Tuple[Any, Any]:
    if len(values) != 2:
        raise ValueError(f"Expected a 2-value range, got {values!r}")
    a, b = cast(values[0]), cast(values[1])
    return (a, b) if a <= b else (b, a)


def _rand_range(rng: np.random.Generator, values: Sequence[float]) -> float:
    lo, hi = _span(values, float)
    return float(rng.uniform(lo, hi))


def _random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    vec = rng.normal(0.0, 1.0, size=3)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return vec / norm


def _configured_material_classes(config: GeneratorConfig) -> List[str]:
    raw = getattr(config, "stone_material_classes", tuple(STONE_MATERIAL_CLASSES))
    if isinstance(raw, str):
        tokens = [part.strip() for part in raw.replace(";", ",").split(",") if part.strip()]
    else:
        tokens = [str(part).strip() for part in raw if str(part).strip()]
    if not tokens or any(token.lower() in {"all", "random"} for token in tokens):
        return list(STONE_MATERIAL_CLASSES)
    return [normalize_stone_material_class(token) for token in tokens]


def _choose_stone_state(rng: np.random.Generator, config: GeneratorConfig) -> str:
    mode = str(getattr(config, "stone_fragmentation", "mixed")).strip().lower().replace("-", "_")
    if mode in {"intact", "whole"}:
        return "intact"
    if mode in {"gravel", "fragmented", "laser_fragmented", "laser_fragmented_gravel"}:
        return "laser_fragmented_gravel"
    if mode != "mixed":
        raise ValueError("stone_fragmentation must be one of: intact, gravel, laser_fragmented_gravel, mixed")
    probability = float(np.clip(getattr(config, "stone_gravel_probability", 0.35), 0.0, 1.0))
    return "laser_fragmented_gravel" if float(rng.uniform(0.0, 1.0)) < probability else "intact"


def _jitter_color(rng: np.random.Generator, color: Sequence[float], sigma: float) -> Tuple[float, float, float, float]:
    rgba = np.asarray(color, dtype=float).copy()
    rgba[:3] = np.clip(rgba[:3] * rng.normal(1.0, sigma, size=3), 0.025, 1.0)
    rgba[3] = float(rgba[3]) if len(rgba) > 3 else 1.0
    return tuple(float(v) for v in rgba)


def _blend_rgb(a: Sequence[float], b: Sequence[float], t: float) -> List[float]:
    av = np.asarray(a[:3], dtype=float)
    bv = np.asarray(b[:3], dtype=float)
    rgb = np.clip(av * (1.0 - float(t)) + bv * float(t), 0.0, 1.0)
    return [float(v) for v in rgb]


def _sample_fracture_plane_count(
    rng: np.random.Generator,
    config: GeneratorConfig,
    profile: Dict[str, Any],
) -> int:
    class_lo, class_hi = _span(profile["fracture_planes"], int)
    cfg_lo, cfg_hi = _span(getattr(config, "stone_fracture_planes", (0, 8)), int)
    lo = max(class_lo, cfg_lo)
    hi = min(class_hi, cfg_hi)
    if lo > hi:
        lo, hi = cfg_lo, cfg_hi
    return int(rng.integers(max(lo, 0), max(hi, lo) + 1))


def _sample_render_profile(
    rng: np.random.Generator,
    config: GeneratorConfig,
    material_class: str,
) -> Dict[str, Any]:
    profile = STONE_MATERIAL_CLASSES[material_class]
    palette = profile["palette"]
    base = _jitter_color(
        rng,
        palette[int(rng.integers(0, len(palette)))],
        sigma=float(profile.get("color_jitter", 0.10)),
    )
    roughness = _rand_range(rng, profile["roughness"])
    specular = _rand_range(rng, profile["specular"])
    bump_strength = _rand_range(rng, profile["crystal_bump_strength"])
    bump_distance = _rand_range(rng, profile["crystal_bump_distance_mm"])
    bump_scale = _rand_range(rng, profile["crystal_bump_scale"])
    bump_detail = _rand_range(rng, profile["crystal_bump_detail"])
    fracture_planes = _sample_fracture_plane_count(rng, config, profile)
    highlight_bias = 0.30 if material_class == "struvite_apatite" else 0.18
    shadow_bias = 0.28 if material_class in {"COM", "uric_acid"} else 0.20
    return {
        "material_class": material_class,
        "display_name": profile["display_name"],
        "mineral": profile["mineral"],
        "base_color": [float(v) for v in base],
        "secondary_color": [*_blend_rgb(base, (1.0, 0.96, 0.82), highlight_bias), 1.0],
        "shadow_color": [*_blend_rgb(base, (0.03, 0.025, 0.02), shadow_bias), 1.0],
        "roughness": float(roughness),
        "specular": float(specular),
        "crystal_bump_strength": float(bump_strength),
        "crystal_bump_distance_mm": float(bump_distance),
        "crystal_bump_scale": float(bump_scale),
        "crystal_bump_detail": float(bump_detail),
        "color_variation_strength": _rand_range(rng, profile["color_variation_strength"]),
        "color_variation_scale": _rand_range(rng, profile["color_variation_scale"]),
        "fracture_plane_count": int(fracture_planes),
        "fracture_darkening": float(rng.uniform(0.18, 0.42)),
        "waxy": bool(material_class in {"uric_acid", "cystine"}),
        "chalky": bool(material_class == "struvite_apatite"),
    }


def _apply_fracture_planes(
    rng: np.random.Generator,
    vertices: np.ndarray,
    radius: float,
    plane_count: int,
    strength_range: Sequence[float],
) -> np.ndarray:
    out = vertices.copy()
    if plane_count <= 0:
        return out
    strength_lo, strength_hi = _span(strength_range, float)
    for _ in range(int(plane_count)):
        normal = _random_unit_vector(rng)
        offset = float(radius) * float(rng.uniform(0.36, 0.82))
        side = out @ normal - offset
        mask = side > 0.0
        if not np.any(mask):
            continue
        strength = float(rng.uniform(strength_lo, strength_hi))
        out[mask] -= normal[None, :] * side[mask, None] * strength
        scrape = rng.normal(0.0, float(radius) * 0.006, size=(int(np.sum(mask)), 3))
        out[mask] += scrape - normal[None, :] * (scrape @ normal)[:, None]
    return out


def _faceted_stone_mesh(
    rng: np.random.Generator,
    radius: float,
    irregularity: float,
    material_class: str,
    render_profile: Dict[str, Any],
    subdivisions: int,
    fragment: bool = False,
) -> trimesh.Trimesh:
    profile = STONE_MATERIAL_CLASSES[material_class]
    mesh = trimesh.creation.icosphere(subdivisions=int(subdivisions), radius=float(radius))
    verts = np.asarray(mesh.vertices, dtype=float).copy()
    norms = np.linalg.norm(verts, axis=1)
    dirs = verts / np.maximum(norms[:, None], 1e-8)

    shape_scale = float(profile["shape_irregularity_scale"])
    irregular = max(float(irregularity), 0.0) * (shape_scale + (0.50 if fragment else 0.0))
    radial = 1.0 + rng.normal(0.0, irregular, size=len(verts))
    lobe_count = int(rng.integers(4, 10 if fragment else 8))
    for _ in range(lobe_count):
        direction = _random_unit_vector(rng)
        power = float(rng.uniform(1.4, 4.5))
        amplitude = float(rng.uniform(-0.10, 0.16)) * (1.0 + irregular)
        radial += amplitude * np.maximum(0.0, dirs @ direction) ** power

    granularity = float(profile["surface_granularity"]) * (1.35 if fragment else 1.0)
    if granularity > 0.0:
        radial += rng.normal(0.0, granularity, size=len(verts))

    spike_strength = float(profile["spike_strength"]) * (0.6 if fragment else 1.0)
    spike_count = int(rng.integers(6, 18)) if spike_strength > 0 else 0
    for _ in range(spike_count):
        direction = _random_unit_vector(rng)
        width = float(rng.uniform(0.035, 0.13))
        dot = np.clip(dirs @ direction, -1.0, 1.0)
        radial += spike_strength * np.exp((dot - 1.0) / max(width, 1e-4))

    radial = np.clip(radial, 0.42 if fragment else 0.55, 1.78)
    verts = dirs * (float(radius) * radial[:, None])
    stretch_width = 0.26 + float(profile["fragment_angularity"]) * (0.16 if fragment else 0.08)
    stretch = rng.uniform(1.0 - stretch_width, 1.0 + stretch_width, size=3)
    verts *= stretch[None, :]

    plane_count = int(render_profile["fracture_plane_count"])
    if fragment:
        plane_count = max(3, plane_count + int(rng.integers(1, 5)))
    verts = _apply_fracture_planes(
        rng,
        verts,
        float(radius),
        plane_count,
        profile["fracture_plane_strength"],
    )

    mesh.vertices = verts
    mesh.fix_normals()
    color = np.asarray(render_profile["base_color"], dtype=float)
    mesh.visual.vertex_colors = np.tile(np.clip(np.rint(color * 255), 0, 255).astype(np.uint8), (len(mesh.vertices), 1))
    return mesh


def _sample_fragment_radii(
    rng: np.random.Generator,
    radius: float,
    count: int,
    config: GeneratorConfig,
    profile: Dict[str, Any],
) -> np.ndarray:
    frac_lo, frac_hi = _span(getattr(config, "stone_fragment_radius_fraction", (0.08, 0.24)), float)
    raw = float(radius) * rng.uniform(frac_lo, frac_hi, size=int(count))
    raw *= rng.lognormal(mean=0.0, sigma=0.30, size=int(count))
    raw = np.clip(raw, 0.045, float(radius) * 0.38)
    retention = _rand_range(rng, profile["fragment_volume_retention"])
    target_volume = retention * float(radius) ** 3
    current_volume = max(float(np.sum(raw**3)), 1e-8)
    scale = min(1.0, (target_volume / current_volume) ** (1.0 / 3.0))
    return np.clip(raw * scale, 0.04, float(radius) * 0.40)


def _sample_gravel_offsets(
    rng: np.random.Generator,
    radii: np.ndarray,
    radius: float,
    cup_radius: float,
    config: GeneratorConfig,
) -> Tuple[np.ndarray, float]:
    spread_lo, spread_hi = _span(getattr(config, "stone_gravel_spread_fraction", (0.42, 0.92)), float)
    max_spread = min(max(float(cup_radius) - float(np.max(radii)) * 1.4, float(radius) * 0.25), float(radius) * spread_hi)
    spread = max(float(radius) * spread_lo, max_spread * float(rng.uniform(0.72, 1.0)))
    spread = max(spread, float(np.max(radii)) * 1.5)
    settle_normal = _random_unit_vector(rng)
    offsets: List[np.ndarray] = []
    for rad in radii:
        best = None
        best_score = -float("inf")
        for _ in range(24):
            direction = _random_unit_vector(rng)
            candidate = direction * spread * float(rng.uniform(0.0, 1.0)) ** (1.0 / 3.0)
            candidate -= settle_normal * abs(float(rng.normal(0.0, spread * 0.14)))
            if not offsets:
                best = candidate
                break
            distances = [float(np.linalg.norm(candidate - other)) - 0.62 * float(rad + radii[j]) for j, other in enumerate(offsets)]
            score = min(distances)
            if score > best_score:
                best = candidate
                best_score = score
            if score > 0.0:
                break
        offsets.append(np.asarray(best, dtype=float))
    return np.asarray(offsets, dtype=float), float(spread)


def _make_gravel_field(
    rng: np.random.Generator,
    center: np.ndarray,
    radius: float,
    cup_radius: float,
    material_class: str,
    render_profile: Dict[str, Any],
    config: GeneratorConfig,
) -> Tuple[trimesh.Trimesh, Dict[str, Any]]:
    profile = STONE_MATERIAL_CLASSES[material_class]
    count_lo, count_hi = _span(getattr(config, "stone_fragment_count", (18, 64)), int)
    count = int(rng.integers(max(count_lo, 1), max(count_hi, count_lo) + 1))
    radii = _sample_fragment_radii(rng, radius, count, config, profile)
    offsets, spread = _sample_gravel_offsets(rng, radii, radius, cup_radius, config)
    fragments: List[trimesh.Trimesh] = []
    for frag_idx, (frag_radius, offset) in enumerate(zip(radii, offsets)):
        frag = _faceted_stone_mesh(
            rng,
            float(frag_radius),
            float(config.stone_irregularity),
            material_class,
            render_profile,
            subdivisions=1,
            fragment=True,
        )
        frag.apply_translation(center + offset)
        frag.metadata["name"] = f"fragment_{frag_idx:03d}"
        fragments.append(frag)
    field_mesh = trimesh.util.concatenate(fragments)
    field_mesh.fix_normals()
    return field_mesh, {
        "fragment_count": int(count),
        "fragment_radius_mm": (float(np.min(radii)), float(np.max(radii))),
        "gravel_spread_mm": float(spread),
    }


def generate_stones(graph: AnatomyGraph, config: GeneratorConfig) -> Tuple[List[trimesh.Trimesh], List[StoneInfo]]:
    rng = np.random.default_rng(config.seed + 12345)
    if config.stone_count <= 0 or len(graph.calyx_targets) == 0:
        return [], []

    material_classes = _configured_material_classes(config)
    calyx_targets = list(graph.calyx_targets)
    rng.shuffle(calyx_targets)
    chosen = [calyx_targets[i % len(calyx_targets)] for i in range(config.stone_count)]

    meshes: List[trimesh.Trimesh] = []
    infos: List[StoneInfo] = []
    for idx, target in enumerate(chosen):
        radius = float(rng.uniform(config.stone_radius_mm[0], config.stone_radius_mm[1]))
        cup_radius = float(target.get("approx_radius_mm", radius + 1.0))
        center = np.asarray(target["center_mm"], dtype=float)
        offset_mag = max(0.0, cup_radius - radius * 0.8)
        offset = rng.normal(0.0, 1.0, size=3)
        offset = offset / max(np.linalg.norm(offset), 1e-8) * rng.uniform(0.0, offset_mag * 0.55)
        stone_center = center + offset

        material_class = str(rng.choice(material_classes))
        render_profile = _sample_render_profile(rng, config, material_class)
        state = _choose_stone_state(rng, config)
        if state == "intact":
            mesh = _faceted_stone_mesh(
                rng,
                radius,
                config.stone_irregularity,
                material_class,
                render_profile,
                subdivisions=int(getattr(config, "stone_surface_subdivisions", 3)),
                fragment=False,
            )
            mesh.apply_translation(stone_center)
            fragment_count = 1
            fragment_radius_mm = (radius, radius)
            gravel_spread_mm = 0.0
        else:
            mesh, gravel_stats = _make_gravel_field(
                rng,
                stone_center,
                radius,
                cup_radius,
                material_class,
                render_profile,
                config,
            )
            fragment_count = int(gravel_stats["fragment_count"])
            fragment_radius_mm = tuple(float(v) for v in gravel_stats["fragment_radius_mm"])
            gravel_spread_mm = float(gravel_stats["gravel_spread_mm"])

        mesh.metadata["name"] = f"stone_{idx:03d}"
        mesh.metadata["material_class"] = material_class
        mesh.metadata["state"] = state
        meshes.append(mesh)
        infos.append(
            StoneInfo(
                id=f"stone_{idx:03d}",
                calyx_id=str(target["id"]),
                region=str(target["region"]),
                center_mm=tuple(float(v) for v in stone_center),
                radius_mm=radius,
                mesh_file=f"stones/stone_{idx:03d}.obj",
                material_class=material_class,
                state=state,
                color_rgba=tuple(float(v) for v in render_profile["base_color"]),
                roughness=float(render_profile["roughness"]),
                specular=float(render_profile["specular"]),
                crystal_bump_strength=float(render_profile["crystal_bump_strength"]),
                crystal_bump_distance_mm=float(render_profile["crystal_bump_distance_mm"]),
                crystal_bump_scale=float(render_profile["crystal_bump_scale"]),
                fracture_plane_count=int(render_profile["fracture_plane_count"]),
                fragment_count=fragment_count,
                fragment_radius_mm=fragment_radius_mm,
                gravel_spread_mm=gravel_spread_mm,
                render_profile=render_profile,
            )
        )
    return meshes, infos
