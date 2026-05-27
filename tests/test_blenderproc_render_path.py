import argparse
import json
from pathlib import Path

import numpy as np

from kidney_meshgen import GeneratorConfig, generate_case
from kidney_meshgen.blenderproc_render import (
    _apply_lens_fluid_effects,
    _apply_plan_temporal_constraints,
    _apply_rgb_sensor_effects,
    _circular_mask_and_vignette,
    _parse_k_matrix,
    _resolve_fluid_model,
    _resolve_sensor_model,
    _sample_realism_preset,
)
from kidney_meshgen.cli import build_parser
from kidney_meshgen.dataset import build_camera_intrinsics, split_frame_indices, write_dataset_convention_files
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


def _fluid_args(**overrides):
    values = {
        "liquid": "volume",
        "fluid_preset": "auto",
        "volume_density": 0.002,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _short_plan():
    return {
        "frame_count": 3,
        "frames": [
            {
                "distance_mm": 0.0,
                "position_mm": [0.0, 0.0, 0.0],
                "forward": [0.0, 0.0, 1.0],
                "up": [0.0, 1.0, 0.0],
                "radius_mm": 2.0,
            },
            {
                "distance_mm": 50.0,
                "position_mm": [0.0, 0.0, 50.0],
                "forward": [0.0, 0.0, 1.0],
                "up": [0.0, 1.0, 0.0],
                "radius_mm": 2.3,
            },
            {
                "distance_mm": 100.0,
                "position_mm": [0.0, 0.0, 100.0],
                "forward": [0.0, 0.0, 1.0],
                "up": [0.0, 1.0, 0.0],
                "radius_mm": 2.0,
            },
        ],
    }


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
    start_forward = np.asarray(plan["frames"][0]["forward"], dtype=float)
    end_forward = np.asarray(plan["frames"][-1]["forward"], dtype=float)
    assert float(np.dot(start_forward, end_forward)) > 0.95


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


def test_max_frames_subsamples_finished_native_trajectory(tmp_path: Path):
    case_dir = _make_case(tmp_path)
    options = RenderPathOptions(traversal="pelvis", fps=30.0, speed_mm_s=18.0, wall_clearance_mm=0.1)
    full = build_blenderproc_camera_plan(case_dir, options)
    capped = build_blenderproc_camera_plan(
        case_dir,
        RenderPathOptions(traversal="pelvis", fps=30.0, speed_mm_s=18.0, max_frames=8, wall_clearance_mm=0.1),
    )
    assert capped["subsampled"] is True
    assert capped["native_frame_count"] == full["frame_count"]
    assert capped["frame_count"] == 8
    assert capped["output_source_frame_indices"][0] == 0
    assert capped["output_source_frame_indices"][-1] == full["frame_count"] - 1
    assert capped["frames"][1]["time_s"] > 1.0 / 30.0


def test_subsampled_plan_disables_motion_interpolation_artifacts():
    sensor = {"motion_blur_length": 0.28, "rolling_shutter_type": "TOP", "rolling_shutter_length": 0.08}
    constrained = _apply_plan_temporal_constraints(sensor, {"subsampled": True})
    assert constrained["motion_blur_length"] == 0.0
    assert constrained["rolling_shutter_type"] == "NONE"
    assert "subsampled" in constrained["temporal_effects_disabled_reason"]


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


def test_camera_intrinsics_sidecar_resolves_fov_when_k_is_absent():
    sensor = _resolve_sensor_model(_sensor_args(sensor_profile="none"), width=640, height=480, plan_fov_degrees=80.0)
    payload = build_camera_intrinsics(sensor, 640, 480, clip_start_mm=0.08, clip_end_mm=260.0, plan_fov_degrees=80.0)
    assert payload["schema"] == "kidney_meshgen_camera_intrinsics_v0.1"
    assert payload["resolution"] == [640, 480]
    assert payload["K"][0][0] > 0.0
    assert payload["cx"] == 320.0
    assert payload["camera_model"] == "pinhole"


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


def test_fluid_model_auto_enhances_liquid_volume_deterministically():
    plan = _short_plan()
    first = _resolve_fluid_model(_fluid_args(), plan, np.random.default_rng(12))
    second = _resolve_fluid_model(_fluid_args(), plan, np.random.default_rng(12))
    assert first == second
    assert first["volume_enabled"] is True
    assert first["enabled"] is True
    assert first["preset"] == "medium"
    assert first["debris_count"] > 0
    assert first["bubble_count"] > 0
    assert first["volume_density"] > 0.002


def test_fluid_model_requires_liquid_volume():
    model = _resolve_fluid_model(_fluid_args(liquid="film"), _short_plan(), np.random.default_rng(12))
    assert model["volume_enabled"] is False
    assert model["enabled"] is False
    assert model["preset"] == "medium"
    assert model["debris_count"] == 0


def test_dataset_sidecars_link_modalities_and_splits(tmp_path):
    plan = {
        "schema": "kidney_meshgen_blenderproc_camera_plan_v0.1",
        "frame_count": 3,
        "fov_degrees": 85.0,
        "frames": [
            {
                "frame_index": 0,
                "time_s": 0.0,
                "cam2world_opengl": np.eye(4).tolist(),
                "position_mm": [0.0, 0.0, 0.0],
                "forward": [0.0, 0.0, 1.0],
                "up": [0.0, 1.0, 0.0],
            },
            {
                "frame_index": 1,
                "time_s": 0.1,
                "cam2world_opengl": np.eye(4).tolist(),
                "position_mm": [0.0, 0.0, 1.0],
                "forward": [0.0, 0.0, 1.0],
                "up": [0.0, 1.0, 0.0],
            },
            {
                "frame_index": 2,
                "time_s": 0.2,
                "cam2world_opengl": np.eye(4).tolist(),
                "position_mm": [0.0, 0.0, 2.0],
                "forward": [0.0, 0.0, 1.0],
                "up": [0.0, 1.0, 0.0],
            },
        ],
    }
    manifest = {
        "anatomy_id": "dataset_case",
        "seed": 4,
        "case_dir": "case",
        "config": {"side": "right", "anatomy_realism_profile": "takazawa"},
        "anatomy_metadata": {"pelvicalyceal_class": "type_i"},
        "stones": [{"material_class": "COM", "state": "intact"}],
    }
    sensor = _resolve_sensor_model(_sensor_args(no_lens_distortion=True), width=320, height=240, plan_fov_degrees=85.0)
    paths = write_dataset_convention_files(
        out_dir=tmp_path,
        manifest=manifest,
        plan=plan,
        output_paths={
            "rgb_dir": str(tmp_path / "rgb"),
            "depth_dir": str(tmp_path / "depth"),
            "normals_dir": str(tmp_path / "normals"),
            "semantic_dir": str(tmp_path / "semantic"),
        },
        sensor_model=sensor,
        width=320,
        height=240,
        clip_start_mm=0.08,
        clip_end_mm=260.0,
        render_seed=99,
        split_ratios="1,1,1",
        split_seed=12,
        write_splits=True,
        realism={"material_preset": "baseline", "light_preset": "baseline"},
        fluid_model={"enabled": False},
        material_preset_arg="baseline",
        light_preset_arg="baseline",
        randomize_realism=False,
        pose_file="camera_poses.json",
    )
    assert paths["dataset_manifest_json"] == "dataset_manifest.json"
    frames = json.loads((tmp_path / "frames.json").read_text(encoding="utf-8"))
    assert frames["modalities"] == ["rgb", "depth", "normals", "semantic"]
    assert frames["frames"][0]["files"]["rgb"] == "rgb/frame_000000.png"
    assert frames["frames"][0]["files"]["semantic"] == "semantic/semantic_000000.png"
    randomization = json.loads((tmp_path / "randomization.json").read_text(encoding="utf-8"))
    assert randomization["render_seed"] == 99
    splits = json.loads((tmp_path / "splits" / "splits.json").read_text(encoding="utf-8"))
    assert sum(splits["counts"].values()) == 3
    assert (tmp_path / "splits" / "train.txt").exists()


def test_split_frame_indices_are_seeded_and_cover_all_frames():
    first = split_frame_indices(10, "0.6,0.2,0.2", seed=5)
    second = split_frame_indices(10, "0.6,0.2,0.2", seed=5)
    assert first == second
    assigned = sorted(idx for indices in first.values() for idx in indices)
    assert assigned == list(range(10))
    assert {name: len(indices) for name, indices in first.items()} == {"train": 6, "val": 2, "test": 2}


def test_lens_fluid_effects_modify_rgb_frame():
    image = np.full((80, 120, 3), 0.35, dtype=np.float32)
    model = {
        "enabled": True,
        "lens": {
            "enabled": True,
            "film_strength": 0.04,
            "film_seed": 5,
            "film_color": [0.78, 0.91, 0.96],
            "droplets": [
                {
                    "cx": 0.5,
                    "cy": 0.5,
                    "rx": 0.12,
                    "ry": 0.10,
                    "angle_rad": 0.0,
                    "alpha": 0.35,
                    "rim_gain": 0.4,
                    "highlight_gain": 0.7,
                    "highlight_dx": -0.2,
                    "highlight_dy": -0.2,
                    "start_frame": 0,
                    "end_frame": 3,
                }
            ],
            "occlusions": [],
        },
    }
    out = _apply_lens_fluid_effects(image, model, frame_index=0)
    assert out.shape == image.shape
    assert not np.allclose(out, image)
    assert float(np.max(out)) > float(np.max(image))


def test_cli_exposes_realism_outputs_and_sensor_flags():
    parser = build_parser()
    gen_args = parser.parse_args(
        [
            "generate",
            "--out",
            "case",
            "--stone-materials",
            "COM,cystine",
            "--stone-fragmentation",
            "gravel",
            "--stone-gravel-probability",
            "1.0",
        ]
    )
    assert gen_args.stone_materials == "COM,cystine"
    assert gen_args.stone_fragmentation == "gravel"
    assert gen_args.stone_gravel_probability == 1.0

    args = parser.parse_args(
        [
            "render-blenderproc",
            "--case-dir",
            "case",
            "--plan-only",
            "--liquid",
            "volume",
            "--fluid-preset",
            "high",
            "--normals",
            "--semantic",
            "--sensor-profile",
            "wide_fov_micro",
            "--distortion-coeffs",
            "-0.2,0.04,0,0,0",
            "--render-seed",
            "77",
            "--split-ratios",
            "0.7,0.2,0.1",
            "--split-seed",
            "5",
        ]
    )
    assert args.normals is True
    assert args.semantic is True
    assert args.sensor_profile == "wide_fov_micro"
    assert args.fluid_preset == "high"
    assert args.render_seed == 77
    assert args.split_ratios == "0.7,0.2,0.1"
    assert args.split_seed == 5
