from pathlib import Path

from kidney_meshgen import GeneratorConfig, generate_case
from kidney_meshgen.anatomy import build_anatomy_graph
from kidney_meshgen.quality import analyze_geometry_quality


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
    assert manifest["schema"] == "kidney_meshgen_mesh_manifest_v0.7"
    assert manifest["mesh_stats"]["vertices"] > 100
    assert manifest["mesh_stats"]["collision_vertices"] > 100
    assert manifest["mesh_stats"]["visual_displacement"]["max_abs_mm"] > 0.0
    assert len(manifest["calyx_targets"]) >= cfg.calyx_count_min
    assert manifest["anatomy_metadata"]["anatomy_realism_profile"] == "takazawa"
    assert manifest["anatomy_metadata"]["pelvicalyceal_class"] in {"type_i", "type_ii"}
    assert any(target["takazawa_name"] == "T" for target in manifest["calyx_targets"])
    assert any(target["anterior_posterior"] == "anterior" for target in manifest["calyx_targets"])
    assert all(target["papilla"] is not None for target in manifest["calyx_targets"])
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


def test_takazawa_type_ii_profile_names_pairs_and_papillae():
    cfg = GeneratorConfig(seed=13, pelvicalyceal_class="type_ii", stone_count=0)
    graph = build_anatomy_graph(cfg)
    quality = analyze_geometry_quality(graph, cfg)
    assert graph.metadata["pelvicalyceal_class"] == "type_ii"
    assert graph.pelvis_type == "divided"
    infundibula = [
        p for p in graph.primitives
        if p.kind == "tapered_capsule" and "calyx_" in p.id and p.operation != "subtract"
    ]
    assert any(p.cross_section_scale0 is not None and p.cross_section_scale1 is not None for p in infundibula)
    assert any(p.narrowing_fraction is not None and p.narrowing_fraction > 0.0 for p in infundibula)
    assert graph.metadata["calyx_distribution"]["top"] == 1
    assert graph.metadata["calyx_distribution"]["bottom"] == 1
    assert {"calyx_upper_anterior", "calyx_upper_posterior"}.issubset({target["id"] for target in graph.calyx_targets})
    assert all(target["papilla"] is not None for target in graph.calyx_targets)
    assert quality["warnings"] == []
