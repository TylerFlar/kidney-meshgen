from pathlib import Path

from kidney_meshgen import GeneratorConfig, generate_case


def test_generate_small_case(tmp_path: Path):
    cfg = GeneratorConfig(
        seed=1,
        grid_resolution=72,
        min_grid_axis=48,
        stone_count=2,
        anatomy_id="test_case",
        export_glb=False,
        export_sdf_grid=False,
        coverage_sample_count=64,
    )
    manifest = generate_case(cfg, tmp_path / "case", make_preview=False)
    assert manifest["anatomy_id"] == "test_case"
    assert manifest["schema"] == "kidney_meshgen_mesh_manifest_v0.6"
    assert manifest["mesh_stats"]["vertices"] > 100
    assert len(manifest["calyx_targets"]) >= cfg.calyx_count_min
    assert len(manifest["stones"]) == 2
    assert manifest["simulator"]["task_scene"] == "runtime_scene.json"
    assert (tmp_path / "case" / "lumen_inner.obj").exists()
    assert (tmp_path / "case" / "centerline_graph.json").exists()
    assert (tmp_path / "case" / "runtime_scene.json").exists()
    assert (tmp_path / "case" / "waypoints" / "navigation_waypoints.json").exists()
    assert (tmp_path / "case" / "collision" / "lumen_collision_proxy.obj").exists()


def test_entry_tube_open_start_and_grid_metadata(tmp_path: Path):
    cfg = GeneratorConfig(
        seed=3,
        grid_resolution=96,
        min_grid_axis=56,
        anatomy_id="tube_case",
        export_glb=False,
        export_sdf_grid=True,
        sdf_grid_resolution=48,
        sdf_min_grid_axis=28,
        stone_count=0,
        coverage_sample_count=32,
    )
    manifest = generate_case(cfg, tmp_path / "case2", make_preview=False)
    start = manifest["scope_start_pose"]["position_mm"]
    assert start[2] < -50.0
    assert manifest["scope_start_pose"]["look_at_node"].startswith("ureter_mid")
    assert manifest["mesh_stats"]["grid_shape"][2] >= cfg.min_grid_axis
    assert manifest["config"]["open_ureter_start"] is True
    assert (tmp_path / "case2" / "collision" / "lumen_sdf_grid.npz").exists()
    assert (tmp_path / "case2" / "collision" / "lumen_sdf_grid.json").exists()
