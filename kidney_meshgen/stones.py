from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
import trimesh

from .config import GeneratorConfig
from .graph import AnatomyGraph


@dataclass
class StoneInfo:
    id: str
    calyx_id: str
    region: str
    center_mm: Tuple[float, float, float]
    radius_mm: float
    mesh_file: str
    label_id: int = 1001
    material: str = "stone_generic"

    def to_dict(self) -> Dict:
        return asdict(self)


def _irregular_icosphere(rng: np.random.Generator, radius: float, irregularity: float) -> trimesh.Trimesh:
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    verts = np.asarray(mesh.vertices).copy()
    norms = np.linalg.norm(verts, axis=1)
    dirs = verts / np.maximum(norms[:, None], 1e-8)
    # Low-frequency-ish perturbation by mixing random radial noise and mild ellipsoid scaling.
    radial = 1.0 + rng.normal(0.0, irregularity, size=len(verts))
    radial = np.clip(radial, 0.55, 1.65)
    scale = np.array([
        rng.uniform(0.75, 1.25),
        rng.uniform(0.75, 1.25),
        rng.uniform(0.75, 1.25),
    ])
    verts = dirs * (radius * radial[:, None])
    verts = verts * scale[None, :]
    mesh.vertices = verts
    mesh.fix_normals()
    return mesh


def generate_stones(graph: AnatomyGraph, config: GeneratorConfig) -> Tuple[List[trimesh.Trimesh], List[StoneInfo]]:
    rng = np.random.default_rng(config.seed + 12345)
    if config.stone_count <= 0 or len(graph.calyx_targets) == 0:
        return [], []

    calyx_targets = list(graph.calyx_targets)
    rng.shuffle(calyx_targets)
    chosen = [calyx_targets[i % len(calyx_targets)] for i in range(config.stone_count)]

    meshes: List[trimesh.Trimesh] = []
    infos: List[StoneInfo] = []
    for idx, target in enumerate(chosen):
        radius = float(rng.uniform(config.stone_radius_mm[0], config.stone_radius_mm[1]))
        cup_radius = float(target.get("approx_radius_mm", radius + 1.0))
        center = np.asarray(target["center_mm"], dtype=float)
        offset_mag = max(0.0, cup_radius - radius * 0.8)
        offset = rng.normal(0.0, 1.0, size=3)
        offset = offset / max(np.linalg.norm(offset), 1e-8) * rng.uniform(0.0, offset_mag * 0.55)
        # Keep stones toward the distal calyx cup; avoid placing them too far into the neck.
        stone_center = center + offset
        mesh = _irregular_icosphere(rng, radius, config.stone_irregularity)
        mesh.apply_translation(stone_center)
        mesh.metadata["name"] = f"stone_{idx:03d}"
        meshes.append(mesh)
        infos.append(StoneInfo(
            id=f"stone_{idx:03d}",
            calyx_id=str(target["id"]),
            region=str(target["region"]),
            center_mm=tuple(float(v) for v in stone_center),
            radius_mm=radius,
            mesh_file=f"stones/stone_{idx:03d}.obj",
        ))
    return meshes, infos


def combine_stones(meshes: List[trimesh.Trimesh]) -> trimesh.Trimesh | None:
    if not meshes:
        return None
    return trimesh.util.concatenate(meshes)
