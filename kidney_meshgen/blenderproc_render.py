from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


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


def _make_tissue_material(bpy, liquid: str):
    mat = bpy.data.materials.new("wet_mucosa_procedural")
    mat.use_nodes = True
    mat.diffuse_color = (0.76, 0.25, 0.32, 1.0)
    try:
        mat.use_screen_refraction = True
    except AttributeError:
        pass
    _set_principled_input(mat, ["Base Color"], (0.78, 0.27, 0.34, 1.0))
    _set_principled_input(mat, ["Roughness"], 0.28 if liquid != "off" else 0.42)
    _set_principled_input(mat, ["Metallic"], 0.0)
    _set_principled_input(mat, ["Alpha"], 1.0)
    _set_principled_input(mat, ["Specular IOR Level", "Specular"], 0.78 if liquid != "off" else 0.48)
    _set_principled_input(mat, ["Coat Weight", "Clearcoat"], 0.35 if liquid != "off" else 0.05)
    _set_principled_input(mat, ["Coat Roughness", "Clearcoat Roughness"], 0.12)
    _add_noise_bump(mat, strength=0.045, distance=0.55, scale=38.0, detail=10.0)
    return mat


def _make_stone_material(bpy):
    mat = bpy.data.materials.new("calcium_oxalate_stone_procedural")
    mat.use_nodes = True
    mat.diffuse_color = (0.72, 0.63, 0.43, 1.0)
    _set_principled_input(mat, ["Base Color"], (0.72, 0.63, 0.43, 1.0))
    _set_principled_input(mat, ["Roughness"], 0.68)
    _set_principled_input(mat, ["Specular IOR Level", "Specular"], 0.28)
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


def _configure_camera(bproc, bpy, width: int, height: int, fov_degrees: float, clip_start: float, clip_end: float, dof: bool, focus_distance: float, fstop: float) -> None:
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


def _matrix_from_frame(frame: Dict) -> np.ndarray:
    return np.asarray(frame["cam2world_opengl"], dtype=float)


def _add_camera_poses(bproc, plan: Dict) -> None:
    for frame in plan["frames"]:
        bproc.camera.add_camera_pose(_matrix_from_frame(frame))


def _create_camera_lights(bpy, energy: float, fill_energy: float, spot_angle_degrees: float):
    spot_data = bpy.data.lights.new("endoscope_spot", type="SPOT")
    spot_data.energy = float(energy)
    spot_data.spot_size = math.radians(float(spot_angle_degrees))
    spot_data.spot_blend = 0.72
    spot_data.shadow_soft_size = 1.0
    spot = bpy.data.objects.new("endoscope_spot", spot_data)
    bpy.context.collection.objects.link(spot)

    point_data = bpy.data.lights.new("endoscope_fill", type="POINT")
    point_data.energy = float(fill_energy)
    point_data.shadow_soft_size = 3.0
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
    parser.add_argument("--depth-of-field", action="store_true")
    parser.add_argument("--focus-distance-mm", type=float, default=18.0)
    parser.add_argument("--fstop", type=float, default=7.0)
    parser.add_argument("--volume-density", type=float, default=0.002)
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

    bproc.init()
    _configure_color_management(bpy)

    lumen_path = _first_existing(case_dir, [files.get("lumen_inner_obj"), files.get("lumen_inner_glb")])
    if lumen_path is None or lumen_path.suffix.lower() != ".obj":
        raise RuntimeError("BlenderProc renderer currently expects lumen_inner.obj in the generated case.")
    lumen = _load_obj_assets(bproc, lumen_path)
    _assign_material(lumen, _make_tissue_material(bpy, args.liquid))
    _set_category(lumen, 1)

    if args.include_stones:
        stone_material = _make_stone_material(bpy)
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
    _configure_camera(
        bproc,
        bpy,
        int(args.width),
        int(args.height),
        float(plan.get("fov_degrees", 85.0)),
        float(args.clip_start_mm),
        float(args.clip_end_mm),
        bool(args.depth_of_field),
        float(args.focus_distance_mm),
        float(args.fstop),
    )
    _add_camera_poses(bproc, plan)
    spot, point = _create_camera_lights(bpy, float(args.light_energy), float(args.fill_light_energy), float(args.spot_angle_degrees))
    _keyframe_lights(bpy, plan, spot, point)

    if args.enable_depth:
        depth_dir = out_dir / "depth"
        depth_dir.mkdir(parents=True, exist_ok=True)
        bproc.renderer.enable_depth_output(activate_antialiasing=False, output_dir=str(depth_dir), file_prefix="depth_")

    bproc.renderer.render(output_dir=str(rgb_dir), file_prefix="frame_", return_data=False)

    metadata = {
        "schema": "kidney_meshgen_blenderproc_render_v0.1",
        "case_dir": str(case_dir),
        "pose_file": str(Path(args.pose_file)),
        "rgb_dir": str(rgb_dir),
        "frame_count": int(plan.get("frame_count", len(plan.get("frames", [])))),
        "include_stones": bool(args.include_stones),
        "liquid": args.liquid,
        "resolution": [int(args.width), int(args.height)],
        "samples": int(args.samples),
        "noise_threshold": float(args.noise_threshold),
        "denoiser": args.denoiser,
    }
    with open(out_dir / "render_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
