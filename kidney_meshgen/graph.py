from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class Node:
    id: str
    position_mm: Tuple[float, float, float]
    kind: str
    region: str


@dataclass
class Edge:
    id: str
    source: str
    target: str
    radius0_mm: float
    radius1_mm: float
    kind: str
    region: str
    parent_region: str = ""


@dataclass
class Primitive:
    id: str
    kind: str  # tapered_capsule or ellipsoid
    label: str
    label_id: int
    operation: str = "union"  # union or subtract; subtract carves solid papillae from the lumen
    # For tapered capsule
    p0: Tuple[float, float, float] | None = None
    p1: Tuple[float, float, float] | None = None
    r0: float | None = None
    r1: float | None = None
    # Optional non-circular tube profile for tapered capsules. The u/v vectors
    # span the cross section; scales are interpolated from p0 to p1.
    cross_section_u: Tuple[float, float, float] | None = None
    cross_section_v: Tuple[float, float, float] | None = None
    cross_section_scale0: Tuple[float, float] | None = None
    cross_section_scale1: Tuple[float, float] | None = None
    narrowing_t: float | None = None
    narrowing_width: float | None = None
    narrowing_fraction: float | None = None
    # For ellipsoid
    center: Tuple[float, float, float] | None = None
    radii: Tuple[float, float, float] | None = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class AnatomyGraph:
    nodes: List[Node]
    edges: List[Edge]
    primitives: List[Primitive]
    labels: Dict[int, str]
    calyx_targets: List[Dict]
    pelvis_type: str
    metadata: Dict = field(default_factory=dict)

    def node_map(self) -> Dict[str, Node]:
        return {n.id: n for n in self.nodes}

    def edge_map(self) -> Dict[str, Edge]:
        return {e.id: e for e in self.edges}

    def to_dict(self) -> Dict:
        return {
            "pelvis_type": self.pelvis_type,
            "metadata": self.metadata,
            "labels": {str(k): v for k, v in self.labels.items()},
            "nodes": [asdict(n) for n in self.nodes],
            "edges": [asdict(e) for e in self.edges],
            "primitives": [p.to_dict() for p in self.primitives],
            "calyx_targets": self.calyx_targets,
        }


def edge_length(edge: Edge, nodes: Dict[str, Node]) -> float:
    p0 = np.asarray(nodes[edge.source].position_mm, dtype=float)
    p1 = np.asarray(nodes[edge.target].position_mm, dtype=float)
    return float(np.linalg.norm(p1 - p0))
