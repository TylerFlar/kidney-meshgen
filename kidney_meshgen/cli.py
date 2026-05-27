from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from .blenderproc_render import FLUID_PRESETS, LIGHT_PRESETS, MATERIAL_PRESETS, SENSOR_PROFILES, _resolve_sensor_model
from .config import GeneratorConfig
from .dataset import write_camera_intrinsics_file, write_split_files
from .generator import generate_case
from .render_path import RenderPathOptions, write_blenderproc_camera_plan
from .stones import STONE_MATERIAL_CLASSES


def _load_config(path: Optional[str]) -> GeneratorConfig:
    if path:
        return GeneratorConfig.from_yaml(path)
    return GeneratorConfig()


def _csv_values(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.replace(";", ",").split(",") if part.strip())


def _apply_common_overrides(cfg: GeneratorConfig, args: argparse.Namespace) -> GeneratorConfig:
    if args.seed is not None:
        cfg.seed = int(args.seed)
    if args.anatomy_id is not None:
        cfg.anatomy_id = args.anatomy_id
    if args.side is not None:
        cfg.side = args.side
    if args.grid is not None:
        cfg.grid_resolution = int(args.grid)
    if args.min_grid_axis is not None:
        cfg.min_grid_axis = int(args.min_grid_axis)
    if args.stones is not None:
        cfg.stone_count = int(args.stones)
    if args.stone_materials is not None:
        cfg.stone_material_classes = _csv_values(args.stone_materials)
    if args.stone_fragmentation is not None:
        cfg.stone_fragmentation = args.stone_fragmentation
    if args.stone_gravel_probability is not None:
        cfg.stone_gravel_probability = float(args.stone_gravel_probability)
    if args.anatomy_profile is not None:
        cfg.anatomy_realism_profile = args.anatomy_profile
    if args.pelvis_type is not None:
        cfg.pelvis_type = args.pelvis_type
    if args.pelvicalyceal_class is not None:
        cfg.pelvicalyceal_class = args.pelvicalyceal_class
    if args.type_i_subtype is not None:
        cfg.type_i_subtype = args.type_i_subtype
    if args.lower_pole_access is not None:
        cfg.lower_pole_access = args.lower_pole_access
    if args.ureter_segments is not None:
        cfg.ureter_segment_count = int(args.ureter_segments)
    if args.graph_retries is not None:
        cfg.graph_clean_retry_attempts = int(args.graph_retries)
    if args.scope_diameter is not None:
        cfg.scope_outer_diameter_mm = float(args.scope_diameter)
    if args.no_glb:
        cfg.export_glb = False
    if args.ply:
        cfg.export_ply = True
    if args.decimate_faces is not None:
        cfg.mesh_decimation_faces = int(args.decimate_faces)
    if args.closed_ureter_start:
        cfg.open_ureter_start = False
    if args.no_papilla_fornix:
        cfg.papilla_fornix_enabled = False
    if args.no_collision_proxy:
        cfg.export_collision_proxy = False
    if args.no_sdf_grid:
        cfg.export_sdf_grid = False
    if args.no_unity_support:
        cfg.write_unity_support = False
    return cfg


def cmd_generate(args: argparse.Namespace) -> None:
    cfg = _load_config(args.config)
    cfg = _apply_common_overrides(cfg, args)
    manifest = generate_case(cfg, args.out, make_preview=not args.no_preview)
    print(json.dumps({
        "anatomy_id": manifest["anatomy_id"],
        "seed": manifest["seed"],
        "manifest": str(Path(args.out) / "scene_manifest.json"),
        "runtime_scene": str(Path(args.out) / "runtime_scene.json"),
        "vertices": manifest["mesh_stats"]["vertices"],
        "faces": manifest["mesh_stats"]["faces"],
        "watertight": manifest["mesh_stats"].get("is_watertight"),
        "stones": len(manifest["stones"]),
        "anatomy_profile": manifest.get("anatomy_metadata", {}).get("anatomy_realism_profile"),
        "pelvicalyceal_class": manifest.get("anatomy_metadata", {}).get("pelvicalyceal_class"),
        "takazawa_type": manifest.get("anatomy_metadata", {}).get("takazawa_type"),
        "lower_pole_access": manifest.get("anatomy_metadata", {}).get("lower_pole_access_mode"),
        "open_ureter_start": manifest.get("config", {}).get("open_ureter_start"),
    }, indent=2))


def _quality_defaults(quality: str) -> dict:
    presets = {
        "preview": {"width": 960, "height": 540, "samples": 32, "noise_threshold": 0.04},
        "balanced": {"width": 1920, "height": 1080, "samples": 128, "noise_threshold": 0.015},
        "high": {"width": 2560, "height": 1440, "samples": 256, "noise_threshold": 0.008},
        "cinematic": {"width": 3840, "height": 2160, "samples": 512, "noise_threshold": 0.004},
    }
    return dict(presets[quality])


def cmd_render_blenderproc(args: argparse.Namespace) -> None:
    case_dir = Path(args.case_dir)
    if not (case_dir / "scene_manifest.json").exists():
        raise SystemExit(f"{case_dir} does not look like a generated kidney-meshgen case; scene_manifest.json is missing.")
    out_dir = Path(args.out) if args.out else case_dir / "blenderproc_render"
    out_dir.mkdir(parents=True, exist_ok=True)

    defaults = _quality_defaults(args.quality)
    width = int(args.width or defaults["width"])
    height = int(args.height or defaults["height"])
    samples = int(args.samples or defaults["samples"])
    noise_threshold = float(args.noise_threshold if args.noise_threshold is not None else defaults["noise_threshold"])

    path_options = RenderPathOptions(
        traversal=args.traversal,
        target_node=args.target_node,
        fps=float(args.fps),
        speed_mm_s=float(args.speed),
        sample_spacing_mm=float(args.sample_spacing),
        smooth_window_mm=float(args.smooth_window),
        max_smooth_offset_mm=float(args.max_smooth_offset),
        lookahead_mm=float(args.lookahead),
        wall_clearance_mm=float(args.wall_clearance),
        fov_degrees=float(args.fov),
        max_frames=args.max_frames,
    )
    plan_files = write_blenderproc_camera_plan(case_dir, out_dir, path_options)
    pose_file = Path(plan_files["camera_poses_json"])
    if args.plan_only:
        with open(pose_file, "r", encoding="utf-8") as f:
            plan = json.load(f)
        sensor_model = _resolve_sensor_model(args, width, height, float(plan.get("fov_degrees", path_options.fov_degrees)))
        intrinsics_path = write_camera_intrinsics_file(
            out_dir,
            sensor_model,
            width,
            height,
            float(args.clip_start),
            float(args.clip_end),
            float(plan.get("fov_degrees", path_options.fov_degrees)),
        )
        split_seed = int(args.split_seed) if args.split_seed is not None else int(args.render_seed or 0)
        split_paths = {}
        if not args.no_splits:
            split_paths = write_split_files(out_dir, int(plan_files["frame_count"]), args.split_ratios, seed=split_seed)
        print(json.dumps({
            "camera_poses": str(pose_file),
            "camera_poses_csv": plan_files["camera_poses_csv"],
            "camera_intrinsics": str(intrinsics_path),
            "splits": None if args.no_splits else str(out_dir / split_paths["splits_json"]),
            "frame_count": int(plan_files["frame_count"]),
        }, indent=2))
        return

    blenderproc = shutil.which(args.blenderproc)
    if blenderproc is None:
        raise SystemExit(
            "Could not find the blenderproc executable. Install it with `uv sync --extra render` "
            "or `uv pip install blenderproc`, then rerun this command."
        )

    script = Path(__file__).resolve().with_name("blenderproc_render.py")
    cmd = [
        blenderproc,
        "run",
        str(script),
        "--case-dir",
        str(case_dir),
        "--pose-file",
        str(pose_file),
        "--out",
        str(out_dir),
        "--liquid",
        args.liquid,
        "--fluid-preset",
        args.fluid_preset,
        "--width",
        str(width),
        "--height",
        str(height),
        "--samples",
        str(samples),
        "--noise-threshold",
        str(noise_threshold),
        "--denoiser",
        args.denoiser,
        "--color-depth",
        str(args.color_depth),
        "--light-energy",
        str(args.light_energy),
        "--fill-light-energy",
        str(args.fill_light_energy),
        "--spot-angle-degrees",
        str(args.spot_angle),
        "--clip-start-mm",
        str(args.clip_start),
        "--clip-end-mm",
        str(args.clip_end),
        "--focus-distance-mm",
        str(args.focus_distance),
        "--fstop",
        str(args.fstop),
        "--sensor-profile",
        args.sensor_profile,
        "--material-preset",
        args.material_preset,
        "--light-preset",
        args.light_preset,
    ]
    if args.render_seed is not None:
        cmd.extend(["--render-seed", str(args.render_seed)])
    if args.camera_k is not None:
        cmd.extend(["--camera-k", str(args.camera_k)])
    if args.distortion_coeffs is not None:
        cmd.extend(["--distortion-coeffs", str(args.distortion_coeffs)])
    if args.no_lens_distortion:
        cmd.append("--no-lens-distortion")
    if args.no_sensor_effects:
        cmd.append("--no-sensor-effects")
    if args.exposure_ev is not None:
        cmd.extend(["--exposure-ev", str(args.exposure_ev)])
    if args.white_balance is not None:
        cmd.extend(["--white-balance", str(args.white_balance)])
    if args.motion_blur_length is not None:
        cmd.extend(["--motion-blur-length", str(args.motion_blur_length)])
    if args.rolling_shutter_type is not None:
        cmd.extend(["--rolling-shutter-type", args.rolling_shutter_type])
    if args.rolling_shutter_length is not None:
        cmd.extend(["--rolling-shutter-length", str(args.rolling_shutter_length)])
    if not args.randomize_realism:
        cmd.append("--no-randomize-realism")
    if args.include_stones:
        cmd.append("--include-stones")
    if args.cpu:
        cmd.append("--cpu")
    if args.cpu_threads is not None:
        cmd.extend(["--cpu-threads", str(args.cpu_threads)])
    if args.depth:
        cmd.append("--enable-depth")
    if args.normals:
        cmd.append("--enable-normals")
    if args.semantic:
        cmd.append("--enable-semantic")
    if args.depth_of_field:
        cmd.append("--depth-of-field")
    cmd.extend(["--split-ratios", args.split_ratios])
    if args.split_seed is not None:
        cmd.extend(["--split-seed", str(args.split_seed)])
    if args.no_splits:
        cmd.append("--no-splits")

    subprocess.run(cmd, check=True)
    print(json.dumps({
        "render_dir": str(out_dir),
        "rgb_dir": str(out_dir / "rgb"),
        "camera_intrinsics": str(out_dir / "camera_intrinsics.json"),
        "frames": str(out_dir / "frames.json"),
        "camera_poses": str(pose_file),
        "camera_poses_csv": plan_files["camera_poses_csv"],
        "randomization": str(out_dir / "randomization.json"),
        "dataset_manifest": str(out_dir / "dataset_manifest.json"),
        "splits": None if args.no_splits else str(out_dir / "splits" / "splits.json"),
        "frame_count": int(plan_files["frame_count"]),
        "include_stones": bool(args.include_stones),
        "resolution": [width, height],
        "quality": args.quality,
        "sensor_profile": args.sensor_profile,
        "normals": bool(args.normals),
        "semantic": bool(args.semantic),
    }, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kidney-meshgen",
        description="Generate renal collecting-system meshes for real-time ureteroscopy navigation/control simulation.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="Generate one kidney case.")
    gen.add_argument("--out", type=str, required=True, help="Output directory.")
    gen.add_argument("--config", type=str, help="YAML config file. Defaults to built-in config.")
    gen.add_argument("--seed", type=int, help="Random seed.")
    gen.add_argument("--anatomy-id", type=str, help="Anatomy/case ID.")
    gen.add_argument("--side", choices=["right", "left"], help="Generate right or left collecting system orientation.")
    gen.add_argument("--grid", type=int, help="Target sample count along the longest marching-cubes axis. 192-240 is a good high-resolution range.")
    gen.add_argument("--min-grid-axis", type=int, help="Minimum sample count along shorter axes.")
    gen.add_argument("--stones", type=int, help="Number of stones to place.")
    gen.add_argument(
        "--stone-materials",
        help=f"Comma-separated stone material classes to sample from: {', '.join(STONE_MATERIAL_CLASSES)}.",
    )
    gen.add_argument(
        "--stone-fragmentation",
        choices=["intact", "gravel", "laser_fragmented_gravel", "mixed"],
        help="Generate intact stones, laser-fragmented gravel fields, or a mixed cohort.",
    )
    gen.add_argument(
        "--stone-gravel-probability",
        type=float,
        help="Probability that each stone becomes a laser-fragmented gravel field when --stone-fragmentation mixed.",
    )
    gen.add_argument("--anatomy-profile", choices=["takazawa", "basic"], help="Anatomy realism profile.")
    gen.add_argument("--pelvis-type", choices=["random", "single", "divided", "type_i", "type_ii"], help="Pelvis morphology family.")
    gen.add_argument("--pelvicalyceal-class", choices=["random", "type_i", "type_ii"], help="Takazawa Type I/II selector.")
    gen.add_argument("--type-i-subtype", choices=["random", "ia", "ib", "ic"], help="Type I pelvis width subtype.")
    gen.add_argument("--lower-pole-access", choices=["random", "easy", "intermediate", "hard"], help="Lower-pole access difficulty model.")
    gen.add_argument("--ureter-segments", type=int, help="Number of polyline segments in the entry ureter tube.")
    gen.add_argument("--graph-retries", type=int, help="How many graph candidates to try before meshing; higher values reduce accidental chamber merges.")
    gen.add_argument("--scope-diameter", type=float, help="Scope outer diameter in mm for clearance estimates.")
    gen.add_argument("--closed-ureter-start", action="store_true", help="Keep the proximal ureter cap closed instead of cutting an open entry.")
    gen.add_argument("--no-papilla-fornix", action="store_true", help="Disable subtractive papilla/fornix cup carving.")
    gen.add_argument("--no-collision-proxy", action="store_true", help="Skip collision/lumen_collision_proxy.obj.")
    gen.add_argument("--no-sdf-grid", action="store_true", help="Skip collision/lumen_sdf_grid.npz.")
    gen.add_argument("--no-unity-support", action="store_true", help="Skip copying Unity helper scripts/config into each generated case.")
    gen.add_argument("--no-glb", action="store_true", help="Skip GLB export.")
    gen.add_argument("--ply", action="store_true", help="Also export PLY.")
    gen.add_argument("--decimate-faces", type=int, help="Optional target face count for visual mesh simplification, if trimesh backend supports it.")
    gen.add_argument("--no-preview", action="store_true", help="Do not create preview_centerline.png.")
    gen.set_defaults(func=cmd_generate)

    render = sub.add_parser("render-blenderproc", help="Render a generated case with BlenderProc.")
    render.add_argument("--case-dir", type=str, required=True, help="Generated case directory containing scene_manifest.json.")
    render.add_argument("--out", type=str, help="Render output directory. Defaults to CASE_DIR/blenderproc_render.")
    render.add_argument("--blenderproc", type=str, default="blenderproc", help="Name or path of the blenderproc executable.")
    render.add_argument("--plan-only", action="store_true", help="Only write camera_poses.json/csv; do not invoke BlenderProc.")
    render.add_argument("--include-stones", action="store_true", help="Render stone meshes. Off by default.")
    render.add_argument("--liquid", choices=["film", "volume"], default="film", help="Wet material/liquid treatment.")
    render.add_argument(
        "--fluid-preset",
        choices=["auto", *sorted(FLUID_PRESETS)],
        default="auto",
        help="Fluid/debris realism for --liquid volume; auto resolves to medium.",
    )
    render.add_argument("--traversal", choices=["dfs", "pelvis"], default="dfs", help="Camera path traversal when --target-node is not set.")
    render.add_argument("--target-node", type=str, help="Fly from entry to a specific graph node, then reverse back to entry.")
    render.add_argument("--fps", type=float, default=30.0, help="Camera frame rate for pose timestamps.")
    render.add_argument("--speed", type=float, default=18.0, help="Nominal camera speed in millimeters per second.")
    render.add_argument("--max-frames", type=int, help="Cap frame count by resampling the full path.")
    render.add_argument("--sample-spacing", type=float, default=1.0, help="Dense centerline sampling spacing in millimeters.")
    render.add_argument("--smooth-window", type=float, default=3.0, help="Gaussian smoothing window in millimeters.")
    render.add_argument("--max-smooth-offset", type=float, default=0.75, help="Maximum smoothing displacement from centerline in millimeters.")
    render.add_argument("--lookahead", type=float, default=6.0, help="Look-ahead distance for camera orientation in millimeters.")
    render.add_argument("--wall-clearance", type=float, default=0.45, help="Required analytic SDF clearance from the lumen wall in millimeters.")
    render.add_argument("--fov", type=float, default=85.0, help="Endoscope field of view in degrees.")
    render.add_argument("--quality", choices=["preview", "balanced", "high", "cinematic"], default="balanced", help="Resolution/sampling preset.")
    render.add_argument("--width", type=int, help="Override render width.")
    render.add_argument("--height", type=int, help="Override render height.")
    render.add_argument("--samples", type=int, help="Override Cycles max samples per pixel.")
    render.add_argument("--noise-threshold", type=float, help="Override Cycles adaptive sampling noise threshold.")
    render.add_argument("--denoiser", choices=["OPTIX", "INTEL", "none"], default="OPTIX", help="Cycles denoiser.")
    render.add_argument("--color-depth", type=int, choices=[8, 16], default=8, help="PNG color depth.")
    render.add_argument("--depth", action="store_true", help="Also render depth frames.")
    render.add_argument("--normals", action="store_true", help="Also render surface normal frames.")
    render.add_argument("--semantic", action="store_true", help="Also render semantic category-ID segmentation frames.")
    render.add_argument("--depth-of-field", action="store_true", help="Enable camera depth of field.")
    render.add_argument("--focus-distance", type=float, default=18.0, help="Depth-of-field focus distance in millimeters.")
    render.add_argument("--fstop", type=float, default=7.0, help="Depth-of-field f-stop.")
    render.add_argument("--light-energy", type=float, default=850.0, help="Camera-mounted spot light energy.")
    render.add_argument("--fill-light-energy", type=float, default=55.0, help="Camera-mounted fill point light energy.")
    render.add_argument("--spot-angle", type=float, default=92.0, help="Camera-mounted spot angle in degrees.")
    render.add_argument("--clip-start", type=float, default=0.08, help="Camera near clip in millimeters.")
    render.add_argument("--clip-end", type=float, default=260.0, help="Camera far clip in millimeters.")
    render.add_argument("--render-seed", type=int, help="Seed for per-run material, light, and sensor randomization.")
    render.add_argument(
        "--sensor-profile",
        choices=sorted(SENSOR_PROFILES),
        default="flexible_ureteroscope_hd",
        help="Endoscope sensor profile for K, distortion, vignette, exposure/WB, noise, and shutter artifacts.",
    )
    render.add_argument(
        "--camera-k",
        help="Custom intrinsics K as JSON/list or path to JSON with K and optional resolution fields.",
    )
    render.add_argument(
        "--distortion-coeffs",
        help="Brown-Conrady coefficients as k1,k2,k3,p1,p2 or equivalent JSON list.",
    )
    render.add_argument("--no-lens-distortion", action="store_true", help="Disable Brown-Conrady distortion.")
    render.add_argument("--no-sensor-effects", action="store_true", help="Disable vignette, exposure/WB, and sensor noise.")
    render.add_argument("--exposure-ev", type=float, help="Override profile exposure compensation in EV.")
    render.add_argument("--white-balance", help="Override profile white balance as R,G,B multipliers.")
    render.add_argument("--motion-blur-length", type=float, help="Override shutter-open fraction between frames; 0 disables.")
    render.add_argument(
        "--rolling-shutter-type",
        choices=["NONE", "TOP", "BOTTOM", "LEFT", "RIGHT"],
        help="Rolling shutter scan direction.",
    )
    render.add_argument("--rolling-shutter-length", type=float, help="Rolling-shutter scanline exposure fraction.")
    render.add_argument(
        "--material-preset",
        choices=["random", *sorted(MATERIAL_PRESETS)],
        default="random",
        help="Material preset to use, or random per run.",
    )
    render.add_argument(
        "--light-preset",
        choices=["random", *sorted(LIGHT_PRESETS)],
        default="random",
        help="Light preset to use, or random per run.",
    )
    render.add_argument("--randomize-realism", dest="randomize_realism", action="store_true", default=True)
    render.add_argument("--no-randomize-realism", dest="randomize_realism", action="store_false")
    render.add_argument(
        "--split-ratios",
        default="0.8,0.1,0.1",
        help="Frame split ratios as train,val,test or train=0.8,val=0.1,test=0.1.",
    )
    render.add_argument("--split-seed", type=int, help="Seed used to shuffle frame IDs before train/val/test split assignment.")
    render.add_argument("--no-splits", action="store_true", help="Skip writing splits/train.txt, val.txt, test.txt, and splits.json.")
    render.add_argument("--cpu", action="store_true", help="Force CPU rendering.")
    render.add_argument("--cpu-threads", type=int, help="CPU thread count to pass to BlenderProc.")
    render.set_defaults(func=cmd_render_blenderproc)
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
