from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from .config import GeneratorConfig
from .generator import generate_case


def _load_config(path: Optional[str]) -> GeneratorConfig:
    if path:
        return GeneratorConfig.from_yaml(path)
    return GeneratorConfig()


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
    if args.pelvis_type is not None:
        cfg.pelvis_type = args.pelvis_type
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
        "lower_pole_access": manifest.get("anatomy_metadata", {}).get("lower_pole_access_mode"),
        "open_ureter_start": manifest.get("config", {}).get("open_ureter_start"),
    }, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kidney-meshgen",
        description="Generate renal collecting-system meshes for real-time ureteroscopy navigation/control simulation.",
    )
    gen = parser.add_subparsers(dest="command", required=True).add_parser("generate", help="Generate one kidney case.")
    gen.add_argument("--out", type=str, required=True, help="Output directory.")
    gen.add_argument("--config", type=str, help="YAML config file. Defaults to built-in config.")
    gen.add_argument("--seed", type=int, help="Random seed.")
    gen.add_argument("--anatomy-id", type=str, help="Anatomy/case ID.")
    gen.add_argument("--side", choices=["right", "left"], help="Generate right or left collecting system orientation.")
    gen.add_argument("--grid", type=int, help="Target sample count along the longest marching-cubes axis. 192-240 is a good high-resolution range.")
    gen.add_argument("--min-grid-axis", type=int, help="Minimum sample count along shorter axes.")
    gen.add_argument("--stones", type=int, help="Number of stones to place.")
    gen.add_argument("--pelvis-type", choices=["random", "single", "divided"], help="Pelvis morphology family.")
    gen.add_argument("--lower-pole-access", choices=["random", "easy", "intermediate", "hard"], help="Lower-pole access difficulty model.")
    gen.add_argument("--ureter-segments", type=int, help="Number of polyline segments in the entry ureter tube.")
    gen.add_argument("--graph-retries", type=int, help="How many graph candidates to try before meshing; higher values reduce accidental chamber merges.")
    gen.add_argument("--scope-diameter", type=float, help="Scope outer diameter in mm for clearance estimates.")
    gen.add_argument("--closed-ureter-start", action="store_true", help="Keep the proximal ureter cap closed instead of cutting an open entry.")
    gen.add_argument("--no-collision-proxy", action="store_true", help="Skip collision/lumen_collision_proxy.obj.")
    gen.add_argument("--no-sdf-grid", action="store_true", help="Skip collision/lumen_sdf_grid.npz.")
    gen.add_argument("--no-unity-support", action="store_true", help="Skip copying Unity helper scripts/config into each generated case.")
    gen.add_argument("--no-glb", action="store_true", help="Skip GLB export.")
    gen.add_argument("--ply", action="store_true", help="Also export PLY.")
    gen.add_argument("--decimate-faces", type=int, help="Optional target face count for visual mesh simplification, if trimesh backend supports it.")
    gen.add_argument("--no-preview", action="store_true", help="Do not create preview_centerline.png.")
    gen.set_defaults(func=cmd_generate)
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
