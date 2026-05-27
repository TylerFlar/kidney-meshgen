from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from kidney_meshgen.dataset import MODALITY_FORMATS, SEMANTIC_LABELS, write_dataset_convention_files
from kidney_meshgen.stones import STONE_MATERIAL_CLASSES

SENSOR_PROFILES: Dict[str, Dict[str, Any]] = {
    "none": {
        "description": "Pinhole camera with no image-domain endoscope effects.",
        "horizontal_fov_degrees": None,
        "distortion": {"k1": 0.0, "k2": 0.0, "k3": 0.0, "p1": 0.0, "p2": 0.0},
        "vignette": {"enabled": False},
        "sensor_noise": {"enabled": False},
        "exposure_ev": 0.0,
        "white_balance_rgb": [1.0, 1.0, 1.0],
        "motion_blur_length": 0.0,
        "rolling_shutter_type": "NONE",
        "rolling_shutter_length": 1.0,
    },
    "flexible_ureteroscope_hd": {
        "description": "Research-informed synthetic HD flexible ureteroscope sensor.",
        "horizontal_fov_degrees": 87.0,
        "principal_point_fraction": [0.502, 0.498],
        "distortion": {"k1": -0.18, "k2": 0.045, "k3": -0.006, "p1": 0.00035, "p2": -0.00025},
        "vignette": {"enabled": True, "radius_fraction": 0.49, "softness_fraction": 0.035, "strength": 0.42},
        "sensor_noise": {
            "enabled": True,
            "full_well_electrons": 8500.0,
            "read_noise_electrons": 4.8,
            "prnu_std": 0.012,
            "dark_offset": 0.0015,
        },
        "exposure_ev": -0.12,
        "white_balance_rgb": [1.07, 1.0, 0.92],
        "motion_blur_length": 0.28,
        "rolling_shutter_type": "TOP",
        "rolling_shutter_length": 0.08,
    },
    "rigid_ureteroscope_hd": {
        "description": "Research-informed synthetic rigid ureteroscope sensor.",
        "horizontal_fov_degrees": 72.0,
        "principal_point_fraction": [0.500, 0.501],
        "distortion": {"k1": -0.075, "k2": 0.012, "k3": 0.0, "p1": 0.00015, "p2": 0.0001},
        "vignette": {"enabled": True, "radius_fraction": 0.50, "softness_fraction": 0.028, "strength": 0.30},
        "sensor_noise": {
            "enabled": True,
            "full_well_electrons": 11000.0,
            "read_noise_electrons": 3.8,
            "prnu_std": 0.008,
            "dark_offset": 0.001,
        },
        "exposure_ev": -0.05,
        "white_balance_rgb": [1.04, 1.0, 0.95],
        "motion_blur_length": 0.18,
        "rolling_shutter_type": "TOP",
        "rolling_shutter_length": 0.10,
    },
    "wide_fov_micro": {
        "description": "Research-informed synthetic wide-FOV miniature endoscope sensor.",
        "horizontal_fov_degrees": 108.0,
        "principal_point_fraction": [0.506, 0.494],
        "distortion": {"k1": -0.32, "k2": 0.095, "k3": -0.018, "p1": 0.0006, "p2": -0.00045},
        "vignette": {"enabled": True, "radius_fraction": 0.475, "softness_fraction": 0.045, "strength": 0.55},
        "sensor_noise": {
            "enabled": True,
            "full_well_electrons": 6200.0,
            "read_noise_electrons": 6.2,
            "prnu_std": 0.016,
            "dark_offset": 0.002,
        },
        "exposure_ev": -0.18,
        "white_balance_rgb": [1.10, 1.0, 0.88],
        "motion_blur_length": 0.32,
        "rolling_shutter_type": "TOP",
        "rolling_shutter_length": 0.06,
    },
}


MATERIAL_PRESETS: Dict[str, Dict[str, Any]] = {
    "baseline": {
        "tissue_base_color": [0.78, 0.27, 0.34, 1.0],
        "tissue_roughness": 0.28,
        "tissue_specular": 0.78,
        "tissue_coat_weight": 0.35,
        "tissue_coat_roughness": 0.12,
        "bump_strength_scale": 1.0,
        "bump_distance_scale": 1.0,
        "bump_scale_scale": 1.0,
    },
    "pale_wet_mucosa": {
        "tissue_base_color": [0.82, 0.34, 0.36, 1.0],
        "tissue_roughness": 0.24,
        "tissue_specular": 0.82,
        "tissue_coat_weight": 0.42,
        "tissue_coat_roughness": 0.10,
        "bump_strength_scale": 0.85,
        "bump_distance_scale": 0.9,
        "bump_scale_scale": 0.9,
    },
    "erythematous_gloss": {
        "tissue_base_color": [0.70, 0.18, 0.25, 1.0],
        "tissue_roughness": 0.20,
        "tissue_specular": 0.88,
        "tissue_coat_weight": 0.48,
        "tissue_coat_roughness": 0.09,
        "bump_strength_scale": 1.15,
        "bump_distance_scale": 1.05,
        "bump_scale_scale": 1.15,
    },
    "irrigated_low_contrast": {
        "tissue_base_color": [0.76, 0.31, 0.35, 1.0],
        "tissue_roughness": 0.33,
        "tissue_specular": 0.70,
        "tissue_coat_weight": 0.28,
        "tissue_coat_roughness": 0.16,
        "bump_strength_scale": 0.7,
        "bump_distance_scale": 0.85,
        "bump_scale_scale": 0.75,
    },
}


LIGHT_PRESETS: Dict[str, Dict[str, Any]] = {
    "baseline": {
        "spot_energy_scale": 1.0,
        "fill_energy_scale": 1.0,
        "spot_angle_delta_degrees": 0.0,
        "spot_color": [1.0, 0.97, 0.92],
        "fill_color": [0.84, 0.92, 1.0],
    },
    "cool_led": {
        "spot_energy_scale": 1.08,
        "fill_energy_scale": 0.85,
        "spot_angle_delta_degrees": -4.0,
        "spot_color": [0.92, 0.97, 1.0],
        "fill_color": [0.70, 0.86, 1.0],
    },
    "warm_halogen": {
        "spot_energy_scale": 0.94,
        "fill_energy_scale": 1.15,
        "spot_angle_delta_degrees": 3.0,
        "spot_color": [1.0, 0.90, 0.78],
        "fill_color": [1.0, 0.82, 0.70],
    },
    "narrow_hotspot": {
        "spot_energy_scale": 1.20,
        "fill_energy_scale": 0.65,
        "spot_angle_delta_degrees": -12.0,
        "spot_color": [1.0, 0.96, 0.88],
        "fill_color": [0.78, 0.88, 1.0],
    },
    "diffuse_irrigated": {
        "spot_energy_scale": 0.82,
        "fill_energy_scale": 1.55,
        "spot_angle_delta_degrees": 8.0,
        "spot_color": [0.94, 0.98, 1.0],
        "fill_color": [0.78, 0.94, 1.0],
    },
}


FLUID_PRESETS: Dict[str, Dict[str, Any]] = {
    "low": {
        "description": "Lightly cloudy irrigation with sparse floating stone dust and mild lens contamination.",
        "volume_density_scale": 1.25,
        "volume_absorption_density_scale": 0.12,
        "volume_noise_strength": 0.22,
        "volume_noise_scale": 5.0,
        "debris_per_mm": 0.18,
        "debris_max": 180,
        "debris_radius_mm": [0.035, 0.13],
        "bubble_per_mm": 0.025,
        "bubble_max": 32,
        "bubble_radius_mm": [0.10, 0.38],
        "lens_droplets": [1, 4],
        "lens_film_strength": 0.025,
        "lens_occlusion_events_per_100_frames": 0.45,
        "lens_occlusion_max": 1,
    },
    "medium": {
        "description": "Cloudy irrigated field with visible floating stone dust, bubbles, droplets, and occasional blur.",
        "volume_density_scale": 1.75,
        "volume_absorption_density_scale": 0.18,
        "volume_noise_strength": 0.34,
        "volume_noise_scale": 7.0,
        "debris_per_mm": 0.45,
        "debris_max": 380,
        "debris_radius_mm": [0.04, 0.18],
        "bubble_per_mm": 0.055,
        "bubble_max": 64,
        "bubble_radius_mm": [0.12, 0.52],
        "lens_droplets": [3, 8],
        "lens_film_strength": 0.045,
        "lens_occlusion_events_per_100_frames": 1.2,
        "lens_occlusion_max": 3,
    },
    "high": {
        "description": "Poorer visibility with dense dust, more bubbles, stronger film, and repeated near-lens occlusion.",
        "volume_density_scale": 2.6,
        "volume_absorption_density_scale": 0.26,
        "volume_noise_strength": 0.46,
        "volume_noise_scale": 9.0,
        "debris_per_mm": 0.90,
        "debris_max": 700,
        "debris_radius_mm": [0.05, 0.24],
        "bubble_per_mm": 0.11,
        "bubble_max": 110,
        "bubble_radius_mm": [0.14, 0.72],
        "lens_droplets": [6, 14],
        "lens_film_strength": 0.075,
        "lens_occlusion_events_per_100_frames": 2.6,
        "lens_occlusion_max": 5,
    },
}


RESEARCH_BASIS = [
    {
        "topic": "camera_intrinsics_and_brown_conrady_distortion",
        "note": "K-matrix pinhole intrinsics with Brown-Conrady radial/decentering distortion.",
        "sources": [
            "https://dlr-rm.github.io/BlenderProc/blenderproc.api.camera.html",
            "https://dlr-rm.github.io/BlenderProc/examples/advanced/lens_distortion/README.html",
            "Brown 1966/1971 photogrammetric camera calibration; Conrady 1919 decentering distortion",
        ],
    },
    {
        "topic": "renderer_passes_and_motion_artifacts",
        "note": "BlenderProc normals, semantic IDs, motion blur, and rolling-shutter hooks.",
        "sources": [
            "https://dlr-rm.github.io/BlenderProc/blenderproc.api.renderer.html",
            "https://dlr-rm.github.io/BlenderProc/examples/advanced/motion_blur_rolling_shutter/README.html",
        ],
    },
    {
        "topic": "sensor_image_formation",
        "note": "Shot/read noise, PRNU, exposure, white balance, and vignetting are camera sensor/image-pipeline approximations.",
        "sources": [
            "EMVA 1288 camera characterization model",
            "Hasinoff 2014 photon/read noise image formation model",
        ],
    },
    {
        "topic": "irrigation_dust_and_lens_soiling",
        "note": "RIRS visibility depends on irrigation; stone dust floats under irrigation, while endoscopic views are degraded by lens condensation and retained contaminants.",
        "sources": [
            "https://icurology.org/DOIx.php?id=10.4111%2Ficu.20200526",
            "https://journals.sagepub.com/doi/10.1089/end.2009.0594",
            "https://pubmed.ncbi.nlm.nih.gov/30020986/",
        ],
    },
    {
        "topic": "cloudy_irrigation_volume_rendering",
        "note": "Cloudy irrigation uses volumetric scattering/absorption with sparse explicit bubble and debris geometry.",
        "sources": [
            "https://docs.blender.org/manual/en/latest/render/shader_nodes/shader/volume_scatter.html",
            "https://docs.blender.org/manual/en/latest/render/materials/components/volume.html",
        ],
    },
]


def _load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _first_existing(case_dir: Path, candidates: Iterable[Optional[str]]) -> Optional[Path]:
    for rel in candidates:
        if not rel:
            continue
        path = case_dir / rel
        if path.exists():
            return path
    return None


def _load_json_arg(value: str) -> Any:
    stripped = value.strip()
    if stripped.startswith("[") or stripped.startswith("{"):
        return json.loads(stripped)
    path = Path(value)
    if path.exists():
        return _load_json(path)
    return json.loads(stripped)


def _parse_float_tuple(value: Optional[str], expected: int, name: str) -> Optional[Tuple[float, ...]]:
    if value is None:
        return None
    raw = value.strip()
    if not raw:
        return None
    if raw.startswith("[") or raw.startswith("{"):
        parsed = _load_json_arg(raw)
        if isinstance(parsed, dict):
            parsed = parsed.get(name) or parsed.get("values")
    else:
        parsed = [part.strip() for part in raw.replace(";", ",").split(",") if part.strip()]
    if not isinstance(parsed, (list, tuple)) or len(parsed) != expected:
        raise ValueError(f"{name} must contain exactly {expected} numeric values")
    return tuple(float(v) for v in parsed)


def _intrinsics_from_fov(width: int, height: int, horizontal_fov_degrees: float, principal_point: Sequence[float]) -> np.ndarray:
    fx = float(width) / (2.0 * math.tan(math.radians(float(horizontal_fov_degrees)) * 0.5))
    fy = fx
    cx = float(principal_point[0]) * float(width)
    cy = float(principal_point[1]) * float(height)
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)


def _scale_k_matrix(k: np.ndarray, width: int, height: int, reference_resolution: Optional[Sequence[float]]) -> np.ndarray:
    scaled = np.asarray(k, dtype=float).reshape(3, 3).copy()
    if reference_resolution:
        ref_w = max(float(reference_resolution[0]), 1.0)
        ref_h = max(float(reference_resolution[1]), 1.0)
        sx = float(width) / ref_w
        sy = float(height) / ref_h
        scaled[0, :] *= sx
        scaled[1, :] *= sy
        scaled[2, :] = [0.0, 0.0, 1.0]
    return scaled


def _parse_k_matrix(value: Optional[str], width: int, height: int) -> Optional[np.ndarray]:
    if value is None:
        return None
    parsed = _load_json_arg(value)
    reference_resolution = None
    if isinstance(parsed, dict):
        reference_resolution = parsed.get("resolution") or parsed.get("image_size")
        parsed = parsed.get("K") or parsed.get("intrinsics")
    k = np.asarray(parsed, dtype=float)
    if k.size != 9:
        raise ValueError("--camera-k must be a 3x3 matrix or a JSON object with a K field")
    return _scale_k_matrix(k.reshape(3, 3), width, height, reference_resolution)


def _distortion_dict(values: Optional[Sequence[float]], profile: Dict[str, Any]) -> Dict[str, float]:
    base = dict(profile.get("distortion", {}))
    if values is not None:
        base.update({name: float(value) for name, value in zip(("k1", "k2", "k3", "p1", "p2"), values)})
    for name in ("k1", "k2", "k3", "p1", "p2"):
        base[name] = float(base.get(name, 0.0))
    return base


def _has_distortion(distortion: Dict[str, float]) -> bool:
    return any(abs(float(distortion.get(name, 0.0))) > 1e-12 for name in ("k1", "k2", "k3", "p1", "p2"))


def _resolve_sensor_model(args: argparse.Namespace, width: int, height: int, plan_fov_degrees: float) -> Dict[str, Any]:
    profile_name = str(args.sensor_profile)
    profile = SENSOR_PROFILES[profile_name]
    k = _parse_k_matrix(args.camera_k, width, height)
    fov = float(plan_fov_degrees)
    if k is None and profile_name != "none":
        fov = float(profile.get("horizontal_fov_degrees") or plan_fov_degrees)
        principal = profile.get("principal_point_fraction", [0.5, 0.5])
        k = _intrinsics_from_fov(width, height, fov, principal)
    distortion_values = _parse_float_tuple(args.distortion_coeffs, 5, "distortion_coeffs")
    distortion = _distortion_dict(distortion_values, profile)
    if args.no_lens_distortion:
        distortion = {"k1": 0.0, "k2": 0.0, "k3": 0.0, "p1": 0.0, "p2": 0.0}

    white_balance = _parse_float_tuple(args.white_balance, 3, "white_balance")
    exposure_ev = float(args.exposure_ev) if args.exposure_ev is not None else float(profile.get("exposure_ev", 0.0))
    motion_blur_length = (
        float(args.motion_blur_length)
        if args.motion_blur_length is not None
        else float(profile.get("motion_blur_length", 0.0))
    )
    rolling_shutter_type = (
        str(args.rolling_shutter_type).upper()
        if args.rolling_shutter_type is not None
        else str(profile.get("rolling_shutter_type", "NONE")).upper()
    )
    rolling_shutter_length = (
        float(args.rolling_shutter_length)
        if args.rolling_shutter_length is not None
        else float(profile.get("rolling_shutter_length", 1.0))
    )
    effects_enabled = bool(profile_name != "none" and not args.no_sensor_effects)
    return {
        "profile": profile_name,
        "description": profile.get("description"),
        "K": None if k is None else [[float(v) for v in row] for row in k],
        "horizontal_fov_degrees": fov,
        "distortion_model": "brown_conrady_opencv",
        "distortion": distortion,
        "lens_distortion_enabled": bool(_has_distortion(distortion)),
        "image_effects_enabled": effects_enabled,
        "vignette": dict(profile.get("vignette", {"enabled": False})),
        "sensor_noise": dict(profile.get("sensor_noise", {"enabled": False})),
        "exposure_ev": exposure_ev,
        "white_balance_rgb": [float(v) for v in (white_balance or profile.get("white_balance_rgb", [1.0, 1.0, 1.0]))],
        "motion_blur_length": max(float(motion_blur_length), 0.0),
        "rolling_shutter_type": rolling_shutter_type,
        "rolling_shutter_length": max(float(rolling_shutter_length), 0.0),
    }


def _random_choice_name(rng: np.random.Generator, requested: str, choices: Dict[str, Dict[str, Any]]) -> str:
    if requested != "random":
        return requested
    names = [name for name in choices if name != "baseline"]
    return str(rng.choice(names))


def _jitter_color(rng: np.random.Generator, color: Sequence[float], sigma: float = 0.035) -> List[float]:
    rgba = np.asarray(color, dtype=float).copy()
    rgba[:3] = np.clip(rgba[:3] * rng.normal(1.0, sigma, size=3), 0.02, 1.0)
    rgba[3] = float(rgba[3]) if len(rgba) > 3 else 1.0
    return [float(v) for v in rgba]


def _normalize(v: np.ndarray, fallback: Sequence[float] = (0.0, 0.0, 1.0)) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-8:
        return np.asarray(fallback, dtype=float)
    return arr / norm


def _sample_realism_preset(args: argparse.Namespace, rng: np.random.Generator) -> Dict[str, Any]:
    material_name = _random_choice_name(rng, str(args.material_preset), MATERIAL_PRESETS)
    light_name = _random_choice_name(rng, str(args.light_preset), LIGHT_PRESETS)
    material = dict(MATERIAL_PRESETS[material_name])
    light = dict(LIGHT_PRESETS[light_name])
    if bool(args.randomize_realism):
        material["tissue_base_color"] = _jitter_color(rng, material["tissue_base_color"], sigma=0.045)
        for key, lo, hi in (
            ("tissue_roughness", 0.86, 1.14),
            ("tissue_specular", 0.90, 1.10),
            ("tissue_coat_weight", 0.85, 1.20),
            ("bump_strength_scale", 0.85, 1.18),
            ("bump_distance_scale", 0.90, 1.15),
            ("bump_scale_scale", 0.85, 1.15),
        ):
            material[key] = float(material[key]) * float(rng.uniform(lo, hi))
        for key, lo, hi in (
            ("spot_energy_scale", 0.86, 1.18),
            ("fill_energy_scale", 0.75, 1.25),
            ("spot_angle_delta_degrees", -3.5, 3.5),
        ):
            if key.endswith("delta_degrees"):
                light[key] = float(light[key]) + float(rng.uniform(lo, hi))
            else:
                light[key] = float(light[key]) * float(rng.uniform(lo, hi))
    return {
        "material_preset": material_name,
        "light_preset": light_name,
        "material": material,
        "light": light,
    }


def _set_principled_input(material, names: Sequence[str], value) -> None:
    nodes = material.node_tree.nodes
    bsdf = next((node for node in nodes if node.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        return
    for name in names:
        if name in bsdf.inputs:
            bsdf.inputs[name].default_value = value
            return


def _add_noise_bump(material, strength: float, distance: float, scale: float, detail: float) -> None:
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    bsdf = next((node for node in nodes if node.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None or "Normal" not in bsdf.inputs:
        return
    noise = nodes.new(type="ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = scale
    noise.inputs["Detail"].default_value = detail
    noise.inputs["Roughness"].default_value = 0.58
    bump = nodes.new(type="ShaderNodeBump")
    bump.inputs["Strength"].default_value = strength
    bump.inputs["Distance"].default_value = distance
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])


def _mix_color(a: Sequence[float], b: Sequence[float], t: float) -> Tuple[float, float, float, float]:
    av = np.asarray(a, dtype=float)
    bv = np.asarray(b, dtype=float)
    if av.size == 3:
        av = np.append(av, 1.0)
    if bv.size == 3:
        bv = np.append(bv, 1.0)
    out = np.clip(av * (1.0 - float(t)) + bv * float(t), 0.0, 1.0)
    out[3] = 1.0
    return tuple(float(v) for v in out)


def _add_noise_color_variation(
    material,
    base_color: Sequence[float],
    secondary_color: Sequence[float],
    shadow_color: Sequence[float],
    scale: float,
    detail: float,
    strength: float,
) -> None:
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    bsdf = next((node for node in nodes if node.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None or "Base Color" not in bsdf.inputs:
        return
    noise = nodes.new(type="ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = float(scale)
    noise.inputs["Detail"].default_value = float(detail)
    noise.inputs["Roughness"].default_value = 0.62
    ramp = nodes.new(type="ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.18
    ramp.color_ramp.elements[0].color = _mix_color(base_color, shadow_color, strength)
    ramp.color_ramp.elements[1].position = 1.0
    ramp.color_ramp.elements[1].color = _mix_color(base_color, secondary_color, strength)
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])


def _make_tissue_material(bpy, config: Dict, preset: Dict[str, Any]):
    mat = bpy.data.materials.new("wet_mucosa_procedural")
    mat.use_nodes = True
    base_color = tuple(float(v) for v in preset["tissue_base_color"])
    mat.diffuse_color = base_color
    try:
        mat.use_screen_refraction = True
    except AttributeError:
        pass
    _set_principled_input(mat, ["Base Color"], base_color)
    _set_principled_input(mat, ["Roughness"], min(float(preset["tissue_roughness"]), 0.68))
    _set_principled_input(mat, ["Metallic"], 0.0)
    _set_principled_input(mat, ["Alpha"], 1.0)
    _set_principled_input(
        mat,
        ["Specular IOR Level", "Specular"],
        float(preset["tissue_specular"]),
    )
    _set_principled_input(
        mat,
        ["Coat Weight", "Clearcoat"],
        float(preset["tissue_coat_weight"]),
    )
    _set_principled_input(mat, ["Coat Roughness", "Clearcoat Roughness"], float(preset["tissue_coat_roughness"]))
    _add_noise_bump(
        mat,
        strength=float(config["render_mucosal_bump_strength"]) * float(preset["bump_strength_scale"]),
        distance=float(config["render_mucosal_bump_distance_mm"]) * float(preset["bump_distance_scale"]),
        scale=float(config["render_mucosal_bump_scale"]) * float(preset["bump_scale_scale"]),
        detail=12.0,
    )
    return mat


def _make_stone_material(bpy, stone: Dict[str, Any]):
    material_class = str(stone["material_class"])
    if material_class not in STONE_MATERIAL_CLASSES:
        valid = ", ".join(STONE_MATERIAL_CLASSES)
        raise ValueError(f"Unknown stone material class {material_class!r}. Expected one of: {valid}")
    profile = dict(stone["render_profile"])
    mat = bpy.data.materials.new(f"{stone['id']}_{material_class}_procedural")
    mat.use_nodes = True
    base_color = tuple(float(v) for v in profile["base_color"])
    mat.diffuse_color = base_color
    _set_principled_input(mat, ["Base Color"], base_color)
    _set_principled_input(mat, ["Roughness"], float(profile["roughness"]))
    _set_principled_input(mat, ["Specular IOR Level", "Specular"], float(profile["specular"]))
    if bool(profile.get("waxy", False)):
        _set_principled_input(mat, ["Coat Weight", "Clearcoat"], 0.10)
        _set_principled_input(mat, ["Coat Roughness", "Clearcoat Roughness"], 0.18)
    if bool(profile.get("chalky", False)):
        _set_principled_input(mat, ["Coat Weight", "Clearcoat"], 0.0)
    _add_noise_color_variation(
        mat,
        base_color=base_color,
        secondary_color=profile["secondary_color"],
        shadow_color=profile["shadow_color"],
        scale=float(profile["color_variation_scale"]),
        detail=max(float(profile["crystal_bump_detail"]) - 2.0, 2.0),
        strength=float(profile["color_variation_strength"]),
    )
    _add_noise_bump(
        mat,
        strength=float(profile["crystal_bump_strength"]),
        distance=float(profile["crystal_bump_distance_mm"]),
        scale=float(profile["crystal_bump_scale"]),
        detail=float(profile["crystal_bump_detail"]),
    )
    return mat


def _assign_material(objects, material) -> None:
    for obj in objects:
        blender_obj = getattr(obj, "blender_obj", obj)
        if not hasattr(blender_obj.data, "materials"):
            continue
        blender_obj.data.materials.clear()
        blender_obj.data.materials.append(material)
        try:
            obj.set_shading_mode("SMOOTH")
        except Exception:
            pass


def _load_obj_assets(bproc, path: Path):
    return bproc.loader.load_obj(str(path))


def _set_category(objects, category_id: int) -> None:
    for obj in objects:
        blender_obj = getattr(obj, "blender_obj", obj)
        blender_obj["category_id"] = int(category_id)


def _scene_bounds_from_plan(plan: Dict) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray([frame["position_mm"] for frame in plan["frames"]], dtype=float)
    margin = 25.0
    return np.min(pts, axis=0) - margin, np.max(pts, axis=0) + margin


def _plan_path_length_mm(plan: Dict) -> float:
    frames = list(plan.get("frames", []))
    if len(frames) < 2:
        return 0.0
    distances = [frame.get("distance_mm") for frame in frames]
    if all(value is not None for value in distances):
        values = [float(value) for value in distances]
        return max(0.0, max(values) - min(values))
    points = np.asarray([frame["position_mm"] for frame in frames], dtype=float)
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def _count_from_path(path_length_mm: float, per_mm: float, maximum: int) -> int:
    if float(path_length_mm) <= 0.0 or float(per_mm) <= 0.0 or int(maximum) <= 0:
        return 0
    return int(min(int(maximum), max(0, round(float(path_length_mm) * float(per_mm)))))


def _lens_interval(rng: np.random.Generator, frame_count: int, persistent_probability: float = 0.7) -> Tuple[int, int]:
    if frame_count <= 1 or float(rng.random()) < float(persistent_probability):
        return 0, max(int(frame_count), 1)
    min_duration = max(2, int(frame_count * 0.12))
    max_duration = max(min_duration + 1, int(frame_count * 0.55))
    duration = int(rng.integers(min_duration, max_duration + 1))
    start = int(rng.integers(0, max(1, frame_count - duration + 1)))
    return start, min(frame_count, start + duration)


def _make_lens_droplets(
    rng: np.random.Generator,
    preset: Dict[str, Any],
    frame_count: int,
) -> List[Dict[str, Any]]:
    low, high = [int(v) for v in preset.get("lens_droplets", [0, 0])]
    count = int(rng.integers(min(low, high), max(low, high) + 1)) if max(low, high) > 0 else 0
    droplets: List[Dict[str, Any]] = []
    for _ in range(count):
        radius = float(rng.uniform(0.014, 0.058))
        rx = radius * float(rng.uniform(0.75, 1.55))
        ry = radius * float(rng.uniform(0.80, 1.65))
        start, end = _lens_interval(rng, frame_count)
        droplets.append(
            {
                "cx": float(rng.uniform(0.10, 0.90)),
                "cy": float(rng.uniform(0.10, 0.90)),
                "rx": rx,
                "ry": ry,
                "angle_rad": float(rng.uniform(0.0, math.tau)),
                "alpha": float(rng.uniform(0.20, 0.44)),
                "rim_gain": float(rng.uniform(0.20, 0.52)),
                "highlight_gain": float(rng.uniform(0.32, 0.80)),
                "highlight_dx": float(rng.uniform(-0.35, -0.12)),
                "highlight_dy": float(rng.uniform(-0.38, -0.10)),
                "start_frame": int(start),
                "end_frame": int(end),
            }
        )
    return droplets


def _make_lens_occlusions(
    rng: np.random.Generator,
    preset: Dict[str, Any],
    frame_count: int,
) -> List[Dict[str, Any]]:
    max_events = int(preset.get("lens_occlusion_max", 0))
    expected = float(preset.get("lens_occlusion_events_per_100_frames", 0.0)) * max(int(frame_count), 1) / 100.0
    event_count = int(min(max_events, rng.poisson(expected))) if max_events > 0 and expected > 0.0 else 0
    if event_count == 0 and max_events > 0 and expected >= 0.7 and float(rng.random()) < min(expected, 0.9):
        event_count = 1
    events: List[Dict[str, Any]] = []
    for _ in range(event_count):
        duration = int(rng.integers(max(2, int(frame_count * 0.04)), max(3, int(frame_count * 0.18)) + 1))
        start = int(rng.integers(0, max(1, frame_count - duration + 1)))
        side = str(rng.choice(["left", "right", "top", "bottom", "center"]))
        if side == "left":
            cx, cy = float(rng.uniform(-0.10, 0.18)), float(rng.uniform(0.15, 0.85))
        elif side == "right":
            cx, cy = float(rng.uniform(0.82, 1.10)), float(rng.uniform(0.15, 0.85))
        elif side == "top":
            cx, cy = float(rng.uniform(0.15, 0.85)), float(rng.uniform(-0.08, 0.20))
        elif side == "bottom":
            cx, cy = float(rng.uniform(0.15, 0.85)), float(rng.uniform(0.80, 1.08))
        else:
            cx, cy = float(rng.uniform(0.30, 0.70)), float(rng.uniform(0.30, 0.70))
        events.append(
            {
                "cx": cx,
                "cy": cy,
                "rx": float(rng.uniform(0.14, 0.36)),
                "ry": float(rng.uniform(0.12, 0.32)),
                "angle_rad": float(rng.uniform(0.0, math.tau)),
                "alpha": float(rng.uniform(0.22, 0.54)),
                "color": [float(v) for v in rng.uniform([0.70, 0.78, 0.78], [0.90, 0.96, 0.98])],
                "start_frame": int(start),
                "end_frame": int(min(frame_count, start + duration)),
            }
        )
    return events


def _resolve_fluid_model(args: argparse.Namespace, plan: Dict, rng: np.random.Generator) -> Dict[str, Any]:
    requested = str(getattr(args, "fluid_preset", "auto"))
    if requested == "auto":
        preset_name = "medium"
    elif requested in FLUID_PRESETS:
        preset_name = requested
    else:
        raise ValueError(f"unknown fluid preset {requested!r}")

    preset = dict(FLUID_PRESETS[preset_name])
    liquid = str(getattr(args, "liquid", "film"))
    volume_enabled = liquid == "volume"
    enhancements_enabled = bool(volume_enabled)
    frame_count = int(plan.get("frame_count", len(plan.get("frames", []))))
    path_length = _plan_path_length_mm(plan)
    base_density = max(float(getattr(args, "volume_density", 0.002)), 0.0)
    volume_density = base_density * float(preset.get("volume_density_scale", 1.0))
    debris_count = _count_from_path(path_length, float(preset.get("debris_per_mm", 0.0)), int(preset.get("debris_max", 0)))
    bubble_count = _count_from_path(path_length, float(preset.get("bubble_per_mm", 0.0)), int(preset.get("bubble_max", 0)))

    if not enhancements_enabled:
        debris_count = 0
        bubble_count = 0

    lens = {
        "enabled": bool(enhancements_enabled),
        "film_strength": float(preset.get("lens_film_strength", 0.0)) if enhancements_enabled else 0.0,
        "film_seed": int(rng.integers(0, 2**31 - 1)),
        "film_color": [0.78, 0.91, 0.96],
        "droplets": _make_lens_droplets(rng, preset, frame_count) if enhancements_enabled else [],
        "occlusions": _make_lens_occlusions(rng, preset, frame_count) if enhancements_enabled else [],
    }
    return {
        "enabled": bool(enhancements_enabled),
        "volume_enabled": bool(volume_enabled),
        "requested_preset": requested,
        "preset": preset_name,
        "description": preset.get("description"),
        "path_length_mm": float(path_length),
        "frame_count": int(frame_count),
        "volume_density": float(volume_density),
        "volume_color": [0.78, 0.93, 1.0, 1.0],
        "volume_absorption_density": float(volume_density) * float(preset.get("volume_absorption_density_scale", 0.0)),
        "volume_noise_strength": float(preset.get("volume_noise_strength", 0.0)) if enhancements_enabled else 0.0,
        "volume_noise_scale": float(preset.get("volume_noise_scale", 4.0)),
        "debris_count": int(debris_count),
        "debris_radius_mm": list(preset.get("debris_radius_mm", [0.04, 0.16])),
        "bubble_count": int(bubble_count),
        "bubble_radius_mm": list(preset.get("bubble_radius_mm", [0.12, 0.45])),
        "lens": lens,
        "rgb_only_lens_effects": bool(enhancements_enabled),
    }


def _set_volume_density_from_noise(nodes, links, volume_node, density: float, noise_strength: float, noise_scale: float) -> None:
    if float(noise_strength) <= 0.0:
        volume_node.inputs["Density"].default_value = float(density)
        return
    noise = nodes.new(type="ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = float(noise_scale)
    noise.inputs["Detail"].default_value = 6.0
    noise.inputs["Roughness"].default_value = 0.62
    noise_weight = nodes.new(type="ShaderNodeMath")
    noise_weight.operation = "MULTIPLY"
    noise_weight.inputs[1].default_value = float(noise_strength)
    base = nodes.new(type="ShaderNodeMath")
    base.operation = "ADD"
    base.inputs[0].default_value = max(0.05, 1.0 - float(noise_strength) * 0.5)
    density_scale = nodes.new(type="ShaderNodeMath")
    density_scale.operation = "MULTIPLY"
    density_scale.inputs[1].default_value = float(density)
    links.new(noise.outputs["Fac"], noise_weight.inputs[0])
    links.new(noise_weight.outputs["Value"], base.inputs[1])
    links.new(base.outputs["Value"], density_scale.inputs[0])
    links.new(density_scale.outputs["Value"], volume_node.inputs["Density"])


def _add_liquid_volume(bpy, bounds_min: np.ndarray, bounds_max: np.ndarray, fluid_model: Dict[str, Any]) -> None:
    center = (bounds_min + bounds_max) * 0.5
    extents = np.maximum(bounds_max - bounds_min, 1.0)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=center)
    cube = bpy.context.object
    cube.name = "cloudy_irrigation_volume"
    cube.dimensions = extents
    mat = bpy.data.materials.new("cloudy_irrigation_volume")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    transparent = nodes.new(type="ShaderNodeBsdfTransparent")
    volume = nodes.new(type="ShaderNodeVolumeScatter")
    volume.inputs["Color"].default_value = tuple(float(v) for v in fluid_model.get("volume_color", [0.78, 0.93, 1.0, 1.0]))
    for input_socket in volume.inputs:
        if "Anisotropy" in input_socket.name:
            input_socket.default_value = 0.18
            break
    _set_volume_density_from_noise(
        nodes,
        links,
        volume,
        float(fluid_model.get("volume_density", 0.002)),
        float(fluid_model.get("volume_noise_strength", 0.0)),
        float(fluid_model.get("volume_noise_scale", 4.0)),
    )
    links.new(transparent.outputs["BSDF"], output.inputs["Surface"])
    absorption_density = float(fluid_model.get("volume_absorption_density", 0.0))
    if absorption_density > 0.0:
        absorption = nodes.new(type="ShaderNodeVolumeAbsorption")
        absorption.inputs["Color"].default_value = (0.68, 0.88, 1.0, 1.0)
        absorption.inputs["Density"].default_value = absorption_density
        add_volume = nodes.new(type="ShaderNodeAddShader")
        links.new(volume.outputs["Volume"], add_volume.inputs[0])
        links.new(absorption.outputs["Volume"], add_volume.inputs[1])
        links.new(add_volume.outputs["Shader"], output.inputs["Volume"])
    else:
        links.new(volume.outputs["Volume"], output.inputs["Volume"])
    mat.blend_method = "BLEND"
    cube.data.materials.append(mat)
    cube.hide_select = True


def _icosahedron_geometry() -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
    phi = (1.0 + math.sqrt(5.0)) * 0.5
    vertices = np.asarray(
        [
            (-1, phi, 0),
            (1, phi, 0),
            (-1, -phi, 0),
            (1, -phi, 0),
            (0, -1, phi),
            (0, 1, phi),
            (0, -1, -phi),
            (0, 1, -phi),
            (phi, 0, -1),
            (phi, 0, 1),
            (-phi, 0, -1),
            (-phi, 0, 1),
        ],
        dtype=float,
    )
    vertices = vertices / np.linalg.norm(vertices, axis=1)[:, None]
    faces = [
        (0, 11, 5),
        (0, 5, 1),
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        (1, 5, 9),
        (5, 11, 4),
        (11, 10, 2),
        (10, 7, 6),
        (7, 1, 8),
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
        (4, 9, 5),
        (2, 4, 11),
        (6, 2, 10),
        (8, 6, 7),
        (9, 8, 1),
    ]
    return vertices, faces


def _random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    import mathutils

    euler = mathutils.Euler(
        (
            float(rng.uniform(0.0, math.tau)),
            float(rng.uniform(0.0, math.tau)),
            float(rng.uniform(0.0, math.tau)),
        ),
        "XYZ",
    )
    return np.asarray(euler.to_matrix(), dtype=float)


def _sample_fluid_centers(
    plan: Dict,
    count: int,
    rng: np.random.Generator,
    min_ahead_mm: float,
    max_ahead_mm: float,
    radius_fraction: float,
) -> np.ndarray:
    frames = list(plan.get("frames", []))
    if int(count) <= 0 or not frames:
        return np.empty((0, 3), dtype=float)
    centers = np.zeros((int(count), 3), dtype=float)
    for idx in range(int(count)):
        frame = frames[int(rng.integers(0, len(frames)))]
        pos = np.asarray(frame["position_mm"], dtype=float)
        forward = _normalize(np.asarray(frame.get("forward", [0.0, 0.0, 1.0]), dtype=float))
        up = _normalize(np.asarray(frame.get("up", [0.0, 1.0, 0.0]), dtype=float))
        right = _normalize(np.cross(forward, up), (1.0, 0.0, 0.0))
        up = _normalize(np.cross(right, forward), (0.0, 1.0, 0.0))
        radius = max(float(frame.get("radius_mm", 1.5)), 0.25)
        max_lateral = max(0.06, min(radius * float(radius_fraction), 4.0))
        theta = float(rng.uniform(0.0, math.tau))
        radial = max_lateral * math.sqrt(float(rng.uniform(0.0, 1.0)))
        ahead = float(rng.uniform(float(min_ahead_mm), float(max_ahead_mm)))
        centers[idx] = pos + forward * ahead + right * (math.cos(theta) * radial) + up * (math.sin(theta) * radial)
    return centers


def _create_icosphere_batch(
    bpy,
    collection,
    name: str,
    centers: np.ndarray,
    radii: np.ndarray,
    material,
    rng: np.random.Generator,
    category_id: int,
    smooth: bool,
    irregularity: float,
) -> Optional[Any]:
    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float)
    if len(centers) == 0:
        return None
    base_vertices, base_faces = _icosahedron_geometry()
    vertices: List[Tuple[float, float, float]] = []
    faces: List[Tuple[int, int, int]] = []
    for center, radius in zip(centers, radii):
        local = base_vertices.copy()
        if float(irregularity) > 0.0:
            local *= rng.uniform(1.0 - float(irregularity), 1.0 + float(irregularity), size=(len(local), 1))
            stretch = rng.uniform(0.70, 1.35, size=3)
        else:
            stretch = rng.uniform(0.88, 1.12, size=3)
        rotation = _random_rotation_matrix(rng)
        transformed = ((local * stretch) @ rotation.T) * float(radius) + center
        offset = len(vertices)
        vertices.extend((float(v[0]), float(v[1]), float(v[2])) for v in transformed)
        faces.extend((a + offset, b + offset, c + offset) for a, b, c in base_faces)

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    if bool(smooth):
        for polygon in mesh.polygons:
            polygon.use_smooth = True
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    obj.data.materials.append(material)
    obj["category_id"] = int(category_id)
    obj.hide_select = True
    return obj


def _make_debris_material(bpy):
    mat = bpy.data.materials.new("suspended_stone_dust")
    mat.use_nodes = True
    _set_principled_input(mat, ["Base Color"], (0.74, 0.66, 0.46, 1.0))
    _set_principled_input(mat, ["Roughness"], 0.78)
    _set_principled_input(mat, ["Specular IOR Level", "Specular"], 0.18)
    _add_noise_bump(mat, strength=0.12, distance=0.05, scale=95.0, detail=5.0)
    return mat


def _make_bubble_material(bpy):
    mat = bpy.data.materials.new("air_bubble_in_irrigation")
    mat.use_nodes = True
    mat.diffuse_color = (0.84, 0.96, 1.0, 0.24)
    try:
        mat.use_screen_refraction = True
    except AttributeError:
        pass
    mat.blend_method = "BLEND"
    mat.use_nodes = True
    _set_principled_input(mat, ["Base Color"], (0.86, 0.97, 1.0, 0.24))
    _set_principled_input(mat, ["Alpha"], 0.24)
    _set_principled_input(mat, ["Roughness"], 0.015)
    _set_principled_input(mat, ["Specular IOR Level", "Specular"], 1.0)
    _set_principled_input(mat, ["Transmission Weight", "Transmission"], 0.70)
    _set_principled_input(mat, ["IOR"], 1.0)
    return mat


def _add_suspended_debris_and_bubbles(
    bpy,
    plan: Dict,
    fluid_model: Dict[str, Any],
    rng: np.random.Generator,
) -> None:
    if not bool(fluid_model.get("enabled", False)):
        return
    collection = bpy.data.collections.new("irrigation_suspended_debris")
    bpy.context.scene.collection.children.link(collection)

    debris_count = int(fluid_model.get("debris_count", 0))
    if debris_count > 0:
        debris_centers = _sample_fluid_centers(
            plan,
            debris_count,
            rng,
            min_ahead_mm=1.5,
            max_ahead_mm=16.0,
            radius_fraction=0.58,
        )
        lo, hi = [float(v) for v in fluid_model.get("debris_radius_mm", [0.04, 0.16])]
        debris_radii = rng.uniform(min(lo, hi), max(lo, hi), size=debris_count)
        _create_icosphere_batch(
            bpy,
            collection,
            "suspended_stone_dust",
            debris_centers,
            debris_radii,
            _make_debris_material(bpy),
            rng,
            category_id=1002,
            smooth=False,
            irregularity=0.28,
        )

    bubble_count = int(fluid_model.get("bubble_count", 0))
    if bubble_count > 0:
        bubble_centers = _sample_fluid_centers(
            plan,
            bubble_count,
            rng,
            min_ahead_mm=2.2,
            max_ahead_mm=20.0,
            radius_fraction=0.50,
        )
        lo, hi = [float(v) for v in fluid_model.get("bubble_radius_mm", [0.12, 0.45])]
        bubble_radii = rng.uniform(min(lo, hi), max(lo, hi), size=bubble_count)
        _create_icosphere_batch(
            bpy,
            collection,
            "irrigation_air_bubbles",
            bubble_centers,
            bubble_radii,
            _make_bubble_material(bpy),
            rng,
            category_id=1003,
            smooth=True,
            irregularity=0.0,
        )


def _configure_color_management(bpy) -> None:
    settings = bpy.context.scene.view_settings
    for transform in ("AgX", "Filmic", "Standard"):
        try:
            settings.view_transform = transform
            break
        except TypeError:
            continue
    for look in ("Medium High Contrast", "Medium Contrast", "None"):
        try:
            settings.look = look
            break
        except TypeError:
            continue
    settings.exposure = -0.25
    settings.gamma = 1.0


def _configure_camera(
    bproc,
    bpy,
    width: int,
    height: int,
    fov_degrees: float,
    clip_start: float,
    clip_end: float,
    dof: bool,
    focus_distance: float,
    fstop: float,
    sensor_model: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    k = np.asarray(sensor_model["K"], dtype=float) if sensor_model.get("K") is not None else None
    if k is not None:
        bproc.camera.set_intrinsics_from_K_matrix(
            k,
            image_width=int(width),
            image_height=int(height),
            clip_start=float(clip_start),
            clip_end=float(clip_end),
        )
    else:
        bproc.camera.set_intrinsics_from_blender_params(
            lens=math.radians(float(fov_degrees)),
            image_width=int(width),
            image_height=int(height),
            clip_start=float(clip_start),
            clip_end=float(clip_end),
            lens_unit="FOV",
        )
    camera = bpy.context.scene.camera
    if camera is not None and dof:
        camera.data.dof.use_dof = True
        camera.data.dof.focus_distance = float(focus_distance)
        camera.data.dof.aperture_fstop = float(fstop)
    distortion = sensor_model.get("distortion", {})
    if not bool(sensor_model.get("lens_distortion_enabled")):
        return None
    result = bproc.camera.set_lens_distortion(
        float(distortion.get("k1", 0.0)),
        float(distortion.get("k2", 0.0)),
        float(distortion.get("k3", 0.0)),
        float(distortion.get("p1", 0.0)),
        float(distortion.get("p2", 0.0)),
    )
    mapping_coords = result
    orig_res_x = int(width)
    orig_res_y = int(height)
    if isinstance(result, tuple):
        if len(result) == 2:
            mapping_coords, orig_res = result
            orig_res_x, orig_res_y = int(orig_res[0]), int(orig_res[1])
        elif len(result) == 3:
            mapping_coords, orig_res_x, orig_res_y = result
            orig_res_x, orig_res_y = int(orig_res_x), int(orig_res_y)
    return {
        "mapping_coords": mapping_coords,
        "orig_res_x": orig_res_x,
        "orig_res_y": orig_res_y,
    }


def _matrix_from_frame(frame: Dict) -> np.ndarray:
    return np.asarray(frame["cam2world_opengl"], dtype=float)


def _add_camera_poses(bproc, plan: Dict) -> None:
    for frame in plan["frames"]:
        bproc.camera.add_camera_pose(_matrix_from_frame(frame))


def _create_camera_lights(
    bpy,
    energy: float,
    fill_energy: float,
    spot_angle_degrees: float,
    light_preset: Dict[str, Any],
):
    spot_data = bpy.data.lights.new("endoscope_spot", type="SPOT")
    spot_data.energy = float(energy) * float(light_preset["spot_energy_scale"])
    spot_data.spot_size = math.radians(max(8.0, float(spot_angle_degrees) + float(light_preset["spot_angle_delta_degrees"])))
    spot_data.spot_blend = 0.72
    spot_data.shadow_soft_size = 1.0
    spot_data.color = tuple(float(v) for v in light_preset["spot_color"])
    spot = bpy.data.objects.new("endoscope_spot", spot_data)
    bpy.context.collection.objects.link(spot)

    point_data = bpy.data.lights.new("endoscope_fill", type="POINT")
    point_data.energy = float(fill_energy) * float(light_preset["fill_energy_scale"])
    point_data.shadow_soft_size = 3.0
    point_data.color = tuple(float(v) for v in light_preset["fill_color"])
    point = bpy.data.objects.new("endoscope_fill", point_data)
    bpy.context.collection.objects.link(point)
    return spot, point


def _keyframe_lights(bpy, plan: Dict, spot, point) -> None:
    import mathutils

    for frame in plan["frames"]:
        frame_idx = int(frame["frame_index"])
        mat = _matrix_from_frame(frame)
        rot = mathutils.Matrix(mat[:3, :3]).to_euler()
        spot.location = mat[:3, 3]
        spot.rotation_euler = rot
        spot.keyframe_insert(data_path="location", frame=frame_idx)
        spot.keyframe_insert(data_path="rotation_euler", frame=frame_idx)

        offset = mat[:3, :3] @ np.asarray([0.9, -0.65, -0.25], dtype=float)
        point.location = mat[:3, 3] + offset
        point.keyframe_insert(data_path="location", frame=frame_idx)


def _configure_renderer(bproc, bpy, args: argparse.Namespace) -> None:
    bproc.renderer.set_output_format(file_format="PNG", color_depth=int(args.color_depth), enable_transparency=False)
    bproc.renderer.set_max_amount_of_samples(int(args.samples))
    bproc.renderer.set_noise_threshold(float(args.noise_threshold))
    bproc.renderer.set_denoiser(None if str(args.denoiser).lower() == "none" else str(args.denoiser).upper())
    bproc.renderer.set_light_bounces(
        diffuse_bounces=4,
        glossy_bounces=4,
        transmission_bounces=6,
        transparent_max_bounces=8,
        volume_bounces=4,
        max_bounces=8,
    )
    try:
        bproc.renderer.set_render_devices(use_only_cpu=bool(args.cpu))
    except Exception:
        pass
    if int(args.cpu_threads) >= 0:
        bproc.renderer.set_cpu_threads(int(args.cpu_threads))
    bproc.renderer.set_world_background([0.0, 0.0, 0.0], strength=0.0)
    try:
        bpy.context.scene.cycles.use_fast_gi = False
    except Exception:
        pass


def _configure_motion_artifacts(bproc, sensor_model: Dict[str, Any]) -> None:
    motion_blur_length = float(sensor_model.get("motion_blur_length", 0.0))
    rolling_shutter_type = str(sensor_model.get("rolling_shutter_type", "NONE")).upper()
    rolling_shutter_length = float(sensor_model.get("rolling_shutter_length", 1.0))
    if motion_blur_length <= 0.0 and rolling_shutter_type == "NONE":
        return
    bproc.renderer.enable_motion_blur(
        motion_blur_length=max(motion_blur_length, 1e-6),
        rolling_shutter_type=rolling_shutter_type,
        rolling_shutter_length=rolling_shutter_length,
    )


def _frame_path(out_dir: Path, prefix: str, index: int, suffix: str) -> Path:
    return out_dir / f"{prefix}{index:06d}.{suffix}"


def _as_float_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[-1] > 3:
        arr = arr[..., :3]
    arr = arr.astype(np.float32, copy=False)
    if np.issubdtype(np.asarray(image).dtype, np.integer):
        max_value = float(np.iinfo(np.asarray(image).dtype).max)
        arr = arr / max(max_value, 1.0)
    elif float(np.nanmax(arr)) > 2.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _temporal_fade(frame_index: int, start_frame: int, end_frame: int) -> float:
    start = int(start_frame)
    end = int(end_frame)
    if frame_index < start or frame_index >= end:
        return 0.0
    duration = max(end - start, 1)
    ramp = max(1, int(duration * 0.18))
    fade_in = min(1.0, (frame_index - start + 1) / float(ramp))
    fade_out = min(1.0, (end - frame_index) / float(ramp))
    return float(min(fade_in, fade_out))


def _rotated_ellipse_distance(
    xx: np.ndarray,
    yy: np.ndarray,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    angle_rad: float,
) -> np.ndarray:
    dx = xx - float(cx)
    dy = yy - float(cy)
    c = math.cos(float(angle_rad))
    s = math.sin(float(angle_rad))
    xr = c * dx + s * dy
    yr = -s * dx + c * dy
    return np.sqrt((xr / max(float(rx), 1e-6)) ** 2 + (yr / max(float(ry), 1e-6)) ** 2)


def _apply_lens_fluid_effects(image: np.ndarray, fluid_model: Dict[str, Any], frame_index: int) -> np.ndarray:
    out = _as_float_image(image).copy()
    if not bool(fluid_model.get("enabled", False)):
        return out
    lens = dict(fluid_model.get("lens", {}))
    if not bool(lens.get("enabled", False)):
        return out
    height, width = out.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    xx = xx / max(float(width - 1), 1.0)
    yy = yy / max(float(height - 1), 1.0)

    film_strength = float(lens.get("film_strength", 0.0))
    if film_strength > 0.0:
        seed_phase = (int(lens.get("film_seed", 0)) % 10007) / 10007.0
        phase = seed_phase + float(frame_index) * 0.013
        veil = (
            0.48
            + 0.18 * np.sin((xx * 5.0 + yy * 1.7 + phase) * math.tau)
            + 0.12 * np.sin((xx * -1.8 + yy * 4.1 + phase * 1.7) * math.tau)
        )
        edge_bias = np.clip(np.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2) * 1.8, 0.0, 1.0)
        veil = np.clip(veil * (0.55 + 0.45 * edge_bias), 0.0, 1.0) * film_strength
        color = np.asarray(lens.get("film_color", [0.78, 0.91, 0.96]), dtype=np.float32)
        out = out * (1.0 - veil[..., None]) + color * veil[..., None]

    for droplet in lens.get("droplets", []):
        fade = _temporal_fade(frame_index, int(droplet.get("start_frame", 0)), int(droplet.get("end_frame", 1)))
        if fade <= 0.0:
            continue
        distance = _rotated_ellipse_distance(
            xx,
            yy,
            float(droplet.get("cx", 0.5)),
            float(droplet.get("cy", 0.5)),
            float(droplet.get("rx", 0.03)),
            float(droplet.get("ry", 0.03)),
            float(droplet.get("angle_rad", 0.0)),
        )
        alpha = float(droplet.get("alpha", 0.3)) * fade
        body = np.clip(1.0 - distance, 0.0, 1.0) ** 0.65
        ring = np.exp(-((distance - 1.0) ** 2) / (2.0 * 0.045**2))
        cx = float(droplet.get("cx", 0.5)) + float(droplet.get("highlight_dx", -0.2)) * float(droplet.get("rx", 0.03))
        cy = float(droplet.get("cy", 0.5)) + float(droplet.get("highlight_dy", -0.2)) * float(droplet.get("ry", 0.03))
        highlight = np.exp(
            -0.5
            * (
                ((xx - cx) / max(float(droplet.get("rx", 0.03)) * 0.20, 1e-5)) ** 2
                + ((yy - cy) / max(float(droplet.get("ry", 0.03)) * 0.20, 1e-5)) ** 2
            )
        )
        tint = np.asarray([0.82, 0.94, 1.0], dtype=np.float32)
        out = out * (1.0 - body[..., None] * alpha * 0.12) + tint * (body[..., None] * alpha * 0.12)
        specular = ring * alpha * float(droplet.get("rim_gain", 0.35))
        specular += highlight * alpha * float(droplet.get("highlight_gain", 0.55))
        out = out + specular[..., None]

    for occlusion in lens.get("occlusions", []):
        fade = _temporal_fade(frame_index, int(occlusion.get("start_frame", 0)), int(occlusion.get("end_frame", 1)))
        if fade <= 0.0:
            continue
        distance = _rotated_ellipse_distance(
            xx,
            yy,
            float(occlusion.get("cx", 0.5)),
            float(occlusion.get("cy", 0.5)),
            float(occlusion.get("rx", 0.22)),
            float(occlusion.get("ry", 0.18)),
            float(occlusion.get("angle_rad", 0.0)),
        )
        mask = np.exp(-0.5 * distance**2)
        alpha = float(occlusion.get("alpha", 0.35)) * fade
        color = np.asarray(occlusion.get("color", [0.80, 0.90, 0.92]), dtype=np.float32)
        out = out * (1.0 - mask[..., None] * alpha) + color * (mask[..., None] * alpha)

    return np.clip(out, 0.0, 1.0)


def _to_uint_image(image: np.ndarray, color_depth: int) -> np.ndarray:
    max_value = 65535 if int(color_depth) == 16 else 255
    dtype = np.uint16 if int(color_depth) == 16 else np.uint8
    return np.clip(np.rint(np.clip(image, 0.0, 1.0) * max_value), 0, max_value).astype(dtype)


def _circular_mask_and_vignette(
    height: int,
    width: int,
    radius_fraction: float,
    softness_fraction: float,
    strength: float,
) -> Tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    cx = (float(width) - 1.0) * 0.5
    cy = (float(height) - 1.0) * 0.5
    radius = max(min(float(width), float(height)) * float(radius_fraction), 1.0)
    softness = max(min(float(width), float(height)) * float(softness_fraction), 1.0)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    norm = np.clip(rr / radius, 0.0, 1.5)
    vignette = 1.0 - float(strength) * norm**2
    vignette = np.clip(vignette, 0.0, 1.0)
    soft_mask = np.clip((radius + softness - rr) / softness, 0.0, 1.0)
    return soft_mask.astype(np.float32), (soft_mask * vignette).astype(np.float32)


def _apply_rgb_sensor_effects(image: np.ndarray, sensor_model: Dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    out = _as_float_image(image)
    if not bool(sensor_model.get("image_effects_enabled", False)):
        return out
    out = out * float(2.0 ** float(sensor_model.get("exposure_ev", 0.0)))
    out = out * np.asarray(sensor_model.get("white_balance_rgb", [1.0, 1.0, 1.0]), dtype=np.float32)
    noise = dict(sensor_model.get("sensor_noise", {}))
    if bool(noise.get("enabled", False)):
        full_well = max(float(noise.get("full_well_electrons", 9000.0)), 1.0)
        read_noise = max(float(noise.get("read_noise_electrons", 0.0)), 0.0)
        prnu_std = max(float(noise.get("prnu_std", 0.0)), 0.0)
        dark_offset = max(float(noise.get("dark_offset", 0.0)), 0.0)
        electrons = np.clip(out, 0.0, 1.0) * full_well
        shot = rng.normal(0.0, np.sqrt(np.maximum(electrons, 1.0))).astype(np.float32)
        read = rng.normal(0.0, read_noise, size=out.shape).astype(np.float32)
        out = (electrons + shot + read) / full_well
        if prnu_std > 0.0:
            out = out * rng.normal(1.0, prnu_std, size=out.shape).astype(np.float32)
        out = out + dark_offset
    vignette = dict(sensor_model.get("vignette", {}))
    if bool(vignette.get("enabled", False)):
        _, gain = _circular_mask_and_vignette(
            out.shape[0],
            out.shape[1],
            float(vignette.get("radius_fraction", 0.49)),
            float(vignette.get("softness_fraction", 0.035)),
            float(vignette.get("strength", 0.4)),
        )
        out = out * gain[..., None]
    return np.clip(out, 0.0, 1.0)


def _apply_output_mask(values: np.ndarray, sensor_model: Dict[str, Any], fill_value: float = 0.0) -> np.ndarray:
    if not bool(sensor_model.get("image_effects_enabled", False)):
        return values
    vignette = dict(sensor_model.get("vignette", {}))
    if not bool(vignette.get("enabled", False)):
        return values
    arr = np.asarray(values).copy()
    mask, _ = _circular_mask_and_vignette(
        arr.shape[0],
        arr.shape[1],
        float(vignette.get("radius_fraction", 0.49)),
        float(vignette.get("softness_fraction", 0.035)),
        float(vignette.get("strength", 0.4)),
    )
    outside = mask <= 0.0
    if arr.ndim == 3:
        arr[outside, :] = fill_value
    else:
        arr[outside] = fill_value
    return arr


def _write_png(path: Path, image: np.ndarray) -> None:
    import imageio.v3 as iio

    iio.imwrite(path, image)


def _write_float_frames(frames: Sequence[np.ndarray], out_dir: Path, prefix: str, sensor_model: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames):
        arr = _apply_output_mask(np.asarray(frame), sensor_model, fill_value=np.nan)
        np.save(_frame_path(out_dir, prefix, idx, "npy"), arr.astype(np.float32, copy=False))


def _write_semantic_frames(frames: Sequence[np.ndarray], out_dir: Path, prefix: str, sensor_model: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames):
        arr = _apply_output_mask(np.asarray(frame), sensor_model, fill_value=0.0)
        if arr.ndim == 3:
            arr = arr[..., 0]
        _write_png(_frame_path(out_dir, prefix, idx, "png"), np.asarray(np.rint(arr), dtype=np.uint16))


def _write_rgb_frames(
    frames: Sequence[np.ndarray],
    out_dir: Path,
    sensor_model: Dict[str, Any],
    fluid_model: Dict[str, Any],
    color_depth: int,
    rng: np.random.Generator,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames):
        rgb = _apply_lens_fluid_effects(np.asarray(frame), fluid_model, idx)
        rgb = _apply_rgb_sensor_effects(rgb, sensor_model, rng)
        _write_png(_frame_path(out_dir, "frame_", idx, "png"), _to_uint_image(rgb, color_depth))


def _write_sensor_mask(out_dir: Path, sensor_model: Dict[str, Any], width: int, height: int) -> Optional[str]:
    if not bool(sensor_model.get("image_effects_enabled", False)):
        return None
    vignette = dict(sensor_model.get("vignette", {}))
    if not bool(vignette.get("enabled", False)):
        return None
    sensor_dir = out_dir / "sensor"
    sensor_dir.mkdir(parents=True, exist_ok=True)
    mask, _ = _circular_mask_and_vignette(
        height,
        width,
        float(vignette.get("radius_fraction", 0.49)),
        float(vignette.get("softness_fraction", 0.035)),
        float(vignette.get("strength", 0.4)),
    )
    path = sensor_dir / "circular_mask.png"
    _write_png(path, (mask > 0.0).astype(np.uint8) * 255)
    return str(path)


def _apply_lens_distortion_to_data(bproc, data: Dict[str, Any], lens_mapping: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if lens_mapping is None:
        return data
    mapping_coords = lens_mapping["mapping_coords"]
    orig_res_x = int(lens_mapping["orig_res_x"])
    orig_res_y = int(lens_mapping["orig_res_y"])
    for key in ("colors", "depth", "distance", "normals", "segmap"):
        if key not in data:
            continue
        data[key] = bproc.postprocessing.apply_lens_distortion(
            data[key],
            mapping_coords,
            orig_res_x,
            orig_res_y,
            use_interpolation=(key == "colors"),
        )
    return data


def _render_and_write_outputs(
    bproc,
    args: argparse.Namespace,
    out_dir: Path,
    rgb_dir: Path,
    sensor_model: Dict[str, Any],
    fluid_model: Dict[str, Any],
    lens_mapping: Optional[Dict[str, Any]],
    rng: np.random.Generator,
) -> Dict[str, str]:
    output_paths: Dict[str, str] = {"rgb_dir": str(rgb_dir)}
    if args.enable_depth:
        depth_dir = out_dir / "depth"
        depth_dir.mkdir(parents=True, exist_ok=True)
        bproc.renderer.enable_depth_output(
            activate_antialiasing=False,
            output_dir=None,
            file_prefix="depth_",
            output_key="depth",
        )
        output_paths["depth_dir"] = str(depth_dir)
    if args.enable_normals:
        normals_dir = out_dir / "normals"
        normals_dir.mkdir(parents=True, exist_ok=True)
        bproc.renderer.enable_normals_output(
            output_dir=None,
            file_prefix="normals_",
            output_key="normals",
        )
        output_paths["normals_dir"] = str(normals_dir)
    if args.enable_semantic:
        semantic_dir = out_dir / "semantic"
        semantic_dir.mkdir(parents=True, exist_ok=True)
        bproc.renderer.enable_segmentation_output(
            map_by="category_id",
            default_values={"category_id": 0},
            output_dir=None,
            file_prefix="semantic_",
            output_key="segmap",
        )
        output_paths["semantic_dir"] = str(semantic_dir)

    data = bproc.renderer.render(output_dir=None, file_prefix="frame_", return_data=True)
    data = _apply_lens_distortion_to_data(bproc, data, lens_mapping)
    if "colors" not in data:
        raise RuntimeError("BlenderProc render did not return RGB color frames.")
    _write_rgb_frames(data["colors"], rgb_dir, sensor_model, fluid_model, int(args.color_depth), rng)
    if args.enable_depth:
        if "depth" not in data:
            raise RuntimeError("Depth output was requested, but BlenderProc did not return depth frames.")
        _write_float_frames(data["depth"], out_dir / "depth", "depth_", sensor_model)
    if args.enable_normals:
        if "normals" not in data:
            raise RuntimeError("Normals output was requested, but BlenderProc did not return normal frames.")
        _write_float_frames(data["normals"], out_dir / "normals", "normals_", sensor_model)
    if args.enable_semantic:
        if "segmap" not in data:
            raise RuntimeError("Semantic output was requested, but BlenderProc did not return segmentation frames.")
        _write_semantic_frames(data["segmap"], out_dir / "semantic", "semantic_", sensor_model)
    return output_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a kidney-meshgen case with BlenderProc.")
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--pose-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--include-stones", action="store_true")
    parser.add_argument("--liquid", choices=["film", "volume"], default="film")
    parser.add_argument("--fluid-preset", choices=["auto", *sorted(FLUID_PRESETS)], default="auto")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--noise-threshold", type=float, default=0.01)
    parser.add_argument("--denoiser", choices=["OPTIX", "INTEL", "none"], default="OPTIX")
    parser.add_argument("--color-depth", type=int, choices=[8, 16], default=8)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=-1)
    parser.add_argument("--light-energy", type=float, default=850.0)
    parser.add_argument("--fill-light-energy", type=float, default=55.0)
    parser.add_argument("--spot-angle-degrees", type=float, default=92.0)
    parser.add_argument("--clip-start-mm", type=float, default=0.08)
    parser.add_argument("--clip-end-mm", type=float, default=260.0)
    parser.add_argument("--enable-depth", action="store_true")
    parser.add_argument("--enable-normals", action="store_true")
    parser.add_argument("--enable-semantic", action="store_true")
    parser.add_argument("--depth-of-field", action="store_true")
    parser.add_argument("--focus-distance-mm", type=float, default=18.0)
    parser.add_argument("--fstop", type=float, default=7.0)
    parser.add_argument("--volume-density", type=float, default=0.002)
    parser.add_argument("--render-seed", type=int)
    parser.add_argument("--sensor-profile", choices=sorted(SENSOR_PROFILES), default="flexible_ureteroscope_hd")
    parser.add_argument("--camera-k")
    parser.add_argument("--distortion-coeffs")
    parser.add_argument("--no-lens-distortion", action="store_true")
    parser.add_argument("--no-sensor-effects", action="store_true")
    parser.add_argument("--exposure-ev", type=float)
    parser.add_argument("--white-balance")
    parser.add_argument("--motion-blur-length", type=float)
    parser.add_argument("--rolling-shutter-type", choices=["NONE", "TOP", "BOTTOM", "LEFT", "RIGHT"])
    parser.add_argument("--rolling-shutter-length", type=float)
    parser.add_argument("--material-preset", choices=["random", *sorted(MATERIAL_PRESETS)], default="random")
    parser.add_argument("--light-preset", choices=["random", *sorted(LIGHT_PRESETS)], default="random")
    parser.add_argument("--randomize-realism", dest="randomize_realism", action="store_true", default=True)
    parser.add_argument("--no-randomize-realism", dest="randomize_realism", action="store_false")
    parser.add_argument("--split-ratios", default="0.8,0.1,0.1")
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--no-splits", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    import blenderproc as bproc
    import bpy

    args = build_parser().parse_args(argv)
    case_dir = Path(args.case_dir)
    out_dir = Path(args.out)
    rgb_dir = out_dir / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_json(case_dir / "scene_manifest.json")
    plan = _load_json(Path(args.pose_file))
    files = manifest.get("files", {})
    render_seed = int(args.render_seed) if args.render_seed is not None else int(np.random.SeedSequence().entropy)
    rng = np.random.default_rng(render_seed)
    sensor_model = _resolve_sensor_model(
        args,
        int(args.width),
        int(args.height),
        float(plan.get("fov_degrees", 85.0)),
    )
    realism = _sample_realism_preset(args, rng)
    fluid_model = _resolve_fluid_model(args, plan, rng)

    bproc.init()
    _configure_color_management(bpy)

    lumen_path = _first_existing(case_dir, [files.get("lumen_inner_obj"), files.get("lumen_inner_glb")])
    if lumen_path is None or lumen_path.suffix.lower() != ".obj":
        raise RuntimeError("BlenderProc renderer currently expects lumen_inner.obj in the generated case.")
    lumen = _load_obj_assets(bproc, lumen_path)
    _assign_material(lumen, _make_tissue_material(bpy, manifest["config"], realism["material"]))
    _set_category(lumen, 1)

    rendered_stones: List[Dict[str, Any]] = []
    if args.include_stones:
        for stone in manifest.get("stones", []):
            stone_path = case_dir / str(stone.get("mesh_file", ""))
            if stone_path.exists():
                loaded = _load_obj_assets(bproc, stone_path)
                _assign_material(loaded, _make_stone_material(bpy, stone))
                _set_category(loaded, int(stone["label_id"]))
                rendered_stones.append(
                    {
                        "id": stone["id"],
                        "material_class": stone["material_class"],
                        "state": stone["state"],
                        "fragment_count": int(stone["fragment_count"]),
                    }
                )

    if args.liquid == "volume":
        bounds_min, bounds_max = _scene_bounds_from_plan(plan)
        _add_liquid_volume(bpy, bounds_min, bounds_max, fluid_model)
        _add_suspended_debris_and_bubbles(bpy, plan, fluid_model, rng)

    _configure_renderer(bproc, bpy, args)
    _configure_motion_artifacts(bproc, sensor_model)
    lens_mapping = _configure_camera(
        bproc,
        bpy,
        int(args.width),
        int(args.height),
        float(sensor_model.get("horizontal_fov_degrees") or plan.get("fov_degrees", 85.0)),
        float(args.clip_start_mm),
        float(args.clip_end_mm),
        bool(args.depth_of_field),
        float(args.focus_distance_mm),
        float(args.fstop),
        sensor_model,
    )
    _add_camera_poses(bproc, plan)
    spot, point = _create_camera_lights(
        bpy,
        float(args.light_energy),
        float(args.fill_light_energy),
        float(args.spot_angle_degrees),
        realism["light"],
    )
    _keyframe_lights(bpy, plan, spot, point)

    output_paths = _render_and_write_outputs(bproc, args, out_dir, rgb_dir, sensor_model, fluid_model, lens_mapping, rng)
    mask_path = _write_sensor_mask(out_dir, sensor_model, int(args.width), int(args.height))
    if mask_path is not None:
        output_paths["sensor_mask"] = mask_path

    split_seed = int(args.split_seed) if args.split_seed is not None else int(render_seed)
    manifest_for_dataset = dict(manifest)
    manifest_for_dataset["case_dir"] = str(case_dir)
    dataset_paths = write_dataset_convention_files(
        out_dir=out_dir,
        manifest=manifest_for_dataset,
        plan=plan,
        output_paths=output_paths,
        sensor_model=sensor_model,
        width=int(args.width),
        height=int(args.height),
        clip_start_mm=float(args.clip_start_mm),
        clip_end_mm=float(args.clip_end_mm),
        render_seed=int(render_seed),
        split_ratios=args.split_ratios,
        split_seed=split_seed,
        write_splits=not bool(args.no_splits),
        realism=realism,
        fluid_model=fluid_model,
        material_preset_arg=args.material_preset,
        light_preset_arg=args.light_preset,
        randomize_realism=bool(args.randomize_realism),
        lens_mapping=lens_mapping,
        pose_file=Path(args.pose_file),
    )
    output_paths.update(dataset_paths)

    metadata = {
        "schema": "kidney_meshgen_blenderproc_render_v0.2",
        "case_dir": str(case_dir),
        "pose_file": str(Path(args.pose_file)),
        "outputs": output_paths,
        "frame_count": int(plan.get("frame_count", len(plan.get("frames", [])))),
        "include_stones": bool(args.include_stones),
        "liquid": args.liquid,
        "fluid": fluid_model,
        "resolution": [int(args.width), int(args.height)],
        "samples": int(args.samples),
        "noise_threshold": float(args.noise_threshold),
        "denoiser": args.denoiser,
        "render_seed": render_seed,
        "sensor": {
            key: value
            for key, value in sensor_model.items()
            if key not in {"image_effects_enabled"} or bool(value)
        },
        "lens_distortion_postprocess": {
            "enabled": lens_mapping is not None,
            "original_resolution": None
            if lens_mapping is None
            else [int(lens_mapping["orig_res_x"]), int(lens_mapping["orig_res_y"])],
        },
        "realism_randomization": realism,
        "rendered_stones": rendered_stones,
        "stone_material_classes": manifest.get("stone_material_model", {}).get("classes"),
        "depth_enabled": bool(args.enable_depth),
        "normals_enabled": bool(args.enable_normals),
        "semantic_enabled": bool(args.enable_semantic),
        "output_formats": MODALITY_FORMATS,
        "semantic_labels": SEMANTIC_LABELS,
        "research_basis": RESEARCH_BASIS,
    }
    with open(out_dir / "render_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
