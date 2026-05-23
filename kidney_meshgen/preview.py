from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from .graph import AnatomyGraph
from .stones import StoneInfo


def write_preview_png(out_dir: Path, graph: AnatomyGraph, stone_infos: List[StoneInfo]) -> str:
    """Write a simple centerline/topology preview image.

    This is deliberately not a diagnostic anatomical rendering; it is a fast QA view
    for generated geometry and stone placement.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ""

    nodes = graph.node_map()
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    region_colors = {"upper": "tab:red", "middle": "tab:green", "lower": "tab:blue"}
    for edge in graph.edges:
        p0 = np.asarray(nodes[edge.source].position_mm)
        p1 = np.asarray(nodes[edge.target].position_mm)
        color = "0.35"
        for region, c in region_colors.items():
            if region in edge.region or region in edge.parent_region:
                color = c
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], linewidth=max(0.8, edge.radius1_mm / 2.2), color=color)

    targets = graph.calyx_targets
    if targets:
        pts = np.array([t["center_mm"] for t in targets], dtype=float)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=30, marker="o", label="calyx cups")

    if stone_infos:
        spts = np.array([s.center_mm for s in stone_infos], dtype=float)
        sizes = np.array([max(25, s.radius_mm * 18) for s in stone_infos])
        ax.scatter(spts[:, 0], spts[:, 1], spts[:, 2], s=sizes, marker="*", label="stones")

    ax.set_title(f"Procedural collecting system: {graph.pelvis_type}")
    ax.set_xlabel("x lateral (mm)")
    ax.set_ylabel("y AP (mm)")
    ax.set_zlabel("z cranio-caudal (mm)")
    try:
        ax.set_box_aspect([1.2, 0.7, 1.6])
    except Exception:
        pass
    ax.view_init(elev=18, azim=-62)
    ax.legend(loc="upper left")
    fig.tight_layout()
    path = out_dir / "preview_centerline.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.name
