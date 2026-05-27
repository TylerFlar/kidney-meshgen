import argparse
import json
from pathlib import Path

import numpy as np

from kidney_meshgen import GeneratorConfig, generate_case
from kidney_meshgen.blenderproc_render import (
    _apply_rgb_sensor_effects,
    _circular_mask_and_vignette,
    _parse_k_matrix,
    _resolve_sensor_model,
    _sample_realism_preset,
)
from kidney_meshgen.cli import build_parser
from kidney_meshgen.render_path import RenderPathOptions, build_blenderproc_camera_plan, write_blenderproc_camera_plan


def _make_case(tmp_path: Path) -> Path:
    cfg = GeneratorConfig(
        seed=9,
        grid_resolution=72,
        min_grid_axis=48,
        anatomy_id="render_path_case",
        export_glb=False,
        export_sdf_grid=False,
        stone_count=1,
        coverage_sample_count=16,
    )
    case_dir = tmp_path / "case"
    generate_case(cfg, case_dir, make_preview=False)
    return case_dir


def _sensor_args(**overrides):
    values = {
        "sensor_profile": "flexible_ureteroscope_hd",
        "camera_k": None,
        "distortion_coeffs": None,
        "no_lens_distortion": False,
        "no_sensor_effects": False,
        "exposure_ev": None,
        "white_balance": None,
        "motion_blur_length": None,
        "rolling_shutter_type": None,
        "rolling_shutter_length": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_blenderproc_plan_routes_to_target_and_back(tmp_path: Path):
    case_dir = _make_case(tmp_path)
    options = RenderPathOptions(
        target_node="pelvis_center",
        fps=10.0,
        speed_mm_s=35.0,
        max_frames=80,
        wall_clearance_mm=0.1,
    )
    plan = build_blenderproc_camera_plan(case_dir, options)
    assert plan["schema"] == "kidney_meshgen_blenderproc_camera_plan_v0.1"
    assert plan["target_node"] == "pelvis_center"
    assert plan["frame_count"] <= 80
    start = np.asarray(plan["frames"][0]["position_mm"], dtype=float)
    end = np.asarray(plan["frames"][-1]["position_mm"], dtype=float)
    assert np.linalg.norm(start - end) < 1e-6
    assert len(plan["frames"][0]["cam2world_opengl"]) == 4
    assert len(plan["frames"][0]["cam2world_opengl"][0]) == 4
    assert max(frame["sdf_mm"] for frame in plan["frames"]) <= -0.1


def test_blenderproc_pose_files_are_written(tmp_path: Path):
    case_dir = _make_case(tmp_path)
    out_dir = tmp_path / "render"
    files = write_blenderproc_camera_plan(
        case_dir,
        out_dir,
        RenderPathOptions(traversal="pelvis", fps=12.0, speed_mm_s=40.0, max_frames=40, wall_clearance_mm=0.1),
    )
    assert Path(files["camera_poses_json"]).exists()
    assert Path(files["camera_poses_csv"]).exists()
    assert int(files["frame_count"]) <= 40


def test_sensor_profile_resolves_intrinsics_and_distortion():
    sensor = _resolve_sensor_model(_sensor_args(), width=1920, height=1080, plan_fov_degrees=85.0)
    k = np.asarray(sensor["K"], dtype=float)
    assert k.shape == (3, 3)
    assert k[0, 0] > 0.0
    assert k[1, 1] == k[0, 0]
    assert 950.0 < k[0, 2] < 970.0
    assert sensor["lens_distortion_enabled"] is True
    assert sensor["image_effects_enabled"] is True
    assert sensor["rolling_shutter_type"] == "TOP"


def test_custom_k_matrix_json_scales_from_reference_resolution(tmp_path):
    k_path = tmp_path / "camera.json"
    k_path.write_text(
        json.dumps({"resolution": [1920, 1080], "K": [[900, 0, 960], [0, 900, 540], [0, 0, 1]]}),
        encoding="utf-8",
    )
    k = _parse_k_matrix(str(k_path), width=960, height=540)
    assert np.allclose(k, [[450, 0, 480], [0, 450, 270], [0, 0, 1]])


def test_circular_vignette_darkens_edges_and_masks_corners():
    mask, gain = _circular_mask_and_vignette(100, 120, radius_fraction=0.46, softness_fraction=0.04, strength=0.5)
    assert mask[50, 60] == 1.0
    assert gain[50, 60] > gain[5, 5]
    assert mask[0, 0] == 0.0


def test_rgb_sensor_effects_are_deterministic_with_seed():
    sensor = _resolve_sensor_model(_sensor_args(no_lens_distortion=True), width=64, height=64, plan_fov_degrees=85.0)
    image = np.full((64, 64, 3), 128, dtype=np.uint8)
    a = _apply_rgb_sensor_effects(image, sensor, np.random.default_rng(123))
    b = _apply_rgb_sensor_effects(image, sensor, np.random.default_rng(123))
    assert np.allclose(a, b)
    assert a.shape == image.shape
    assert float(a[0, 0, 0]) < float(a[32, 32, 0])


def test_no_sensor_effects_leaves_rgb_unchanged():
    sensor = _resolve_sensor_model(
        _sensor_args(no_lens_distortion=True, no_sensor_effects=True),
        width=16,
        height=16,
        plan_fov_degrees=85.0,
    )
    image = np.full((16, 16, 3), 128, dtype=np.uint8)
    out = _apply_rgb_sensor_effects(image, sensor, np.random.default_rng(123))
    assert sensor["image_effects_enabled"] is False
    assert np.allclose(out, image.astype(np.float32) / 255.0)


def test_realism_preset_sampling_is_seeded():
    args = argparse.Namespace(material_preset="random", light_preset="random", randomize_realism=True)
    first = _sample_realism_preset(args, np.random.default_rng(42))
    second = _sample_realism_preset(args, np.random.default_rng(42))
    assert first == second
    assert first["material_preset"] != "baseline"
    assert first["light_preset"] != "baseline"


def test_cli_exposes_realism_outputs_and_sensor_flags():
    parser = build_parser()
    args = parser.parse_args(
        [
            "render-blenderproc",
            "--case-dir",
            "case",
            "--plan-only",
            "--normals",
            "--semantic",
            "--sensor-profile",
            "wide_fov_micro",
            "--distortion-coeffs",
            "-0.2,0.04,0,0,0",
            "--render-seed",
            "77",
        ]
    )
    assert args.normals is True
    assert args.semantic is True
    assert args.sensor_profile == "wide_fov_micro"
    assert args.render_seed == 77
