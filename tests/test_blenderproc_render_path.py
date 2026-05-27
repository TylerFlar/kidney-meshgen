from pathlib import Path

import numpy as np

from kidney_meshgen import GeneratorConfig, generate_case
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
