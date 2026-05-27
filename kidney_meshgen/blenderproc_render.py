from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

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
        "stone_base_color": [0.72, 0.63, 0.43, 1.0],
        "stone_roughness": 0.68,
        "stone_specular": 0.28,
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
        "stone_base_color": [0.76, 0.68, 0.48, 1.0],
        "stone_roughness": 0.64,
        "stone_specular": 0.26,
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
        "stone_base_color": [0.66, 0.57, 0.39, 1.0],
        "stone_roughness": 0.72,
        "stone_specular": 0.24,
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
        "stone_base_color": [0.74, 0.65, 0.46, 1.0],
        "stone_roughness": 0.70,
        "stone_specular": 0.22,
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


def _sample_realism_preset(args: argparse.Namespace, rng: np.random.Generator) -> Dict[str, Any]:
    material_name = _random_choice_name(rng, str(args.material_preset), MATERIAL_PRESETS)
    light_name = _random_choice_name(rng, str(args.light_preset), LIGHT_PRESETS)
    material = dict(MATERIAL_PRESETS[material_name])
    light = dict(LIGHT_PRESETS[light_name])
    if bool(args.randomize_realism):
        material["tissue_base_color"] = _jitter_color(rng, material["tissue_base_color"], sigma=0.045)
        material["stone_base_color"] = _jitter_color(rng, material["stone_base_color"], sigma=0.05)
        for key, lo, hi in (
            ("tissue_roughness", 0.86, 1.14),
            ("tissue_specular", 0.90, 1.10),
            ("tissue_coat_weight", 0.85, 1.20),
            ("stone_roughness", 0.90, 1.12),
            ("stone_specular", 0.85, 1.20),
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


def _make_tissue_material(bpy, liquid: str, config: Dict, preset: Dict[str, Any]):
    mat = bpy.data.materials.new("wet_mucosa_procedural")
    mat.use_nodes = True
    base_color = tuple(float(v) for v in preset["tissue_base_color"])
    mat.diffuse_color = base_color
    try:
        mat.use_screen_refraction = True
    except AttributeError:
        pass
    wet_scale = 1.0 if liquid != "off" else 1.45
    _set_principled_input(mat, ["Base Color"], base_color)
    _set_principled_input(mat, ["Roughness"], min(float(preset["tissue_roughness"]) * wet_scale, 0.68))
    _set_principled_input(mat, ["Metallic"], 0.0)
    _set_principled_input(mat, ["Alpha"], 1.0)
    _set_principled_input(
        mat,
        ["Specular IOR Level", "Specular"],
        float(preset["tissue_specular"]) if liquid != "off" else min(float(preset["tissue_specular"]) * 0.62, 0.55),
    )
    _set_principled_input(
        mat,
        ["Coat Weight", "Clearcoat"],
        float(preset["tissue_coat_weight"]) if liquid != "off" else min(float(preset["tissue_coat_weight"]) * 0.18, 0.08),
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


def _make_stone_material(bpy, preset: Dict[str, Any]):
    mat = bpy.data.materials.new("calcium_oxalate_stone_procedural")
    mat.use_nodes = True
    base_color = tuple(float(v) for v in preset["stone_base_color"])
    mat.diffuse_color = base_color
    _set_principled_input(mat, ["Base Color"], base_color)
    _set_principled_input(mat, ["Roughness"], float(preset["stone_roughness"]))
    _set_principled_input(mat, ["Specular IOR Level", "Specular"], float(preset["stone_specular"]))
    _add_noise_bump(mat, strength=0.075, distance=0.38, scale=72.0, detail=9.0)
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


def _add_liquid_volume(bpy, bounds_min: np.ndarray, bounds_max: np.ndarray, density: float) -> None:
    center = (bounds_min + bounds_max) * 0.5
    extents = np.maximum(bounds_max - bounds_min, 1.0)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=center)
    cube = bpy.context.object
    cube.name = "subtle_irrigation_volume"
    cube.dimensions = extents
    mat = bpy.data.materials.new("clear_irrigation_volume")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    transparent = nodes.new(type="ShaderNodeBsdfTransparent")
    volume = nodes.new(type="ShaderNodeVolumeScatter")
    volume.inputs["Color"].default_value = (0.78, 0.93, 1.0, 1.0)
    volume.inputs["Density"].default_value = density
    links.new(transparent.outputs["BSDF"], output.inputs["Surface"])
    links.new(volume.outputs["Volume"], output.inputs["Volume"])
    mat.blend_method = "BLEND"
    cube.data.materials.append(mat)
    cube.hide_select = True


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
    color_depth: int,
    rng: np.random.Generator,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames):
        rgb = _apply_rgb_sensor_effects(np.asarray(frame), sensor_model, rng)
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
    _write_rgb_frames(data["colors"], rgb_dir, sensor_model, int(args.color_depth), rng)
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
    parser.add_argument("--liquid", choices=["off", "film", "volume"], default="film")
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

    bproc.init()
    _configure_color_management(bpy)

    lumen_path = _first_existing(case_dir, [files.get("lumen_inner_obj"), files.get("lumen_inner_glb")])
    if lumen_path is None or lumen_path.suffix.lower() != ".obj":
        raise RuntimeError("BlenderProc renderer currently expects lumen_inner.obj in the generated case.")
    lumen = _load_obj_assets(bproc, lumen_path)
    _assign_material(lumen, _make_tissue_material(bpy, args.liquid, manifest["config"], realism["material"]))
    _set_category(lumen, 1)

    if args.include_stones:
        stone_material = _make_stone_material(bpy, realism["material"])
        stone_objects = []
        for stone in manifest.get("stones", []):
            stone_path = case_dir / str(stone.get("mesh_file", ""))
            if stone_path.exists():
                loaded = _load_obj_assets(bproc, stone_path)
                stone_objects.extend(loaded)
        _assign_material(stone_objects, stone_material)
        _set_category(stone_objects, 1001)

    if args.liquid == "volume":
        bounds_min, bounds_max = _scene_bounds_from_plan(plan)
        _add_liquid_volume(bpy, bounds_min, bounds_max, float(args.volume_density))

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

    output_paths = _render_and_write_outputs(bproc, args, out_dir, rgb_dir, sensor_model, lens_mapping, rng)
    mask_path = _write_sensor_mask(out_dir, sensor_model, int(args.width), int(args.height))
    if mask_path is not None:
        output_paths["sensor_mask"] = mask_path

    metadata = {
        "schema": "kidney_meshgen_blenderproc_render_v0.2",
        "case_dir": str(case_dir),
        "pose_file": str(Path(args.pose_file)),
        "outputs": output_paths,
        "frame_count": int(plan.get("frame_count", len(plan.get("frames", [])))),
        "include_stones": bool(args.include_stones),
        "liquid": args.liquid,
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
        "depth_enabled": bool(args.enable_depth),
        "normals_enabled": bool(args.enable_normals),
        "semantic_enabled": bool(args.enable_semantic),
        "output_formats": {
            "rgb": "png",
            "depth": "float32 npy",
            "normals": "float32 npy",
            "semantic": "uint16 png",
        },
        "semantic_labels": {
            "background": 0,
            "lumen_mucosa": 1,
            "stone": 1001,
        },
        "research_basis": RESEARCH_BASIS,
    }
    with open(out_dir / "render_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
