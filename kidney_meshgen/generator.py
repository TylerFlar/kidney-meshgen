from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from .anatomy import build_anatomy_graph
from .config import GeneratorConfig
from .exporters import (
    export_meshes,
    write_camera_paths,
    write_centerline,
    write_label_tables,
    write_manifest,
)
from .mesh import build_lumen_mesh
from .preview import write_preview_png
from .quality import analyze_geometry_quality, write_quality_report
from .sim_export import write_simulator_exports
from .stones import generate_stones


def generate_case(config: GeneratorConfig, out_dir: str | Path, make_preview: bool = True) -> Dict:
    """Generate one procedural kidney collecting-system case.

    Returns the manifest dict loaded from disk.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    requested_seed = int(config.seed)

    # Build one or more graph candidates before meshing. This is cheap relative
    # to marching cubes and lets the generator avoid obvious chamber/tube merges.
    attempts = max(1, int(getattr(config, "graph_clean_retry_attempts", 1)))
    best_graph = None
    best_config = None
    best_score = float("inf")
    for attempt in range(attempts):
        cfg_try = GeneratorConfig(**config.to_dict())
        cfg_try.seed = int(config.seed) + attempt * 9973
        graph_try = build_anatomy_graph(cfg_try)
        quality = analyze_geometry_quality(graph_try, cfg_try)
        seg_clear = quality.get("clearance", {}).get("minimum_non_connected_segment_clearance_mm")
        cup_clear = quality.get("clearance", {}).get("minimum_cup_center_clearance_mm")
        score = 1000.0 * len(quality.get("warnings", []))
        if seg_clear is not None and seg_clear < 0:
            score += abs(float(seg_clear)) * 10.0
        if cup_clear is not None and cup_clear < 0:
            score += abs(float(cup_clear)) * 20.0
        if score < best_score:
            best_graph, best_config, best_score = graph_try, cfg_try, score
        if not quality.get("warnings"):
            break

    graph = best_graph
    config = best_config or config
    if graph is not None:
        graph.metadata["requested_seed"] = requested_seed
        graph.metadata["selected_seed"] = int(config.seed)
        graph.metadata["graph_retry_attempts_configured"] = attempts
        graph.metadata["graph_retry_offset"] = int((int(config.seed) - requested_seed) // 9973) if int(config.seed) >= requested_seed else 0
    mesh_result = build_lumen_mesh(graph, config)
    stone_meshes, stone_infos = generate_stones(graph, config)

    files: Dict[str, str] = {}
    files.update(export_meshes(out_dir, mesh_result, graph, config, stone_meshes))
    files.update(write_label_tables(out_dir, mesh_result, graph))
    files.update(write_centerline(out_dir, graph))
    files.update(write_camera_paths(out_dir, graph, config))
    files.update(write_quality_report(out_dir, graph, config))

    if make_preview:
        preview_file = write_preview_png(out_dir, graph, stone_infos)
        files["preview_png"] = preview_file

    # Save resolved config for exact reproduction before writing descriptors.
    config.to_yaml(out_dir / "resolved_config.yaml")
    files["resolved_config_yaml"] = "resolved_config.yaml"

    files.update(write_simulator_exports(out_dir, config, graph, mesh_result, stone_infos, stone_meshes, files))
    manifest_file = write_manifest(out_dir, config, graph, mesh_result, stone_infos, files)
    files.update(manifest_file)

    with open(out_dir / "scene_manifest.json", "r", encoding="utf-8") as f:
        return json.load(f)
