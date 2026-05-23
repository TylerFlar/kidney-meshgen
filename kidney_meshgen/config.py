from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml


@dataclass
class GeneratorConfig:
    """Configuration for a ureteroscopy mesh/simulation case.

    All geometric values are in millimeters unless otherwise noted.
    `grid_resolution` is interpreted as the target sample count along the
    longest bounding-box axis during marching-cubes extraction.
    """

    seed: int = 7
    anatomy_id: str = "kidney_case"
    side: str = "right"  # right or left; left mirrors x coordinates

    # Mesh extraction and core outputs
    grid_resolution: int = 192
    min_grid_axis: int = 88
    padding_mm: float = 10.0
    mesh_smoothing_iterations: int = 10
    mesh_decimation_faces: Optional[int] = None
    export_glb: bool = True
    export_obj: bool = True
    export_ply: bool = False

    # Simulator assets
    export_collision_proxy: bool = True
    collision_proxy_faces: Optional[int] = 8000
    export_sdf_grid: bool = True
    sdf_grid_resolution: int = 96
    sdf_min_grid_axis: int = 44
    coverage_sample_count: int = 2048
    waypoint_samples_per_edge: int = 8
    write_unity_support: bool = True

    # Ureter entry. Default is open because the scope should begin in a tube,
    # not behind a closed hemispherical cap.
    open_ureter_start: bool = True
    open_ureter_start_offset_mm: float = 1.2

    # Overall collecting-system morphology
    calyx_count_min: int = 7
    calyx_count_max: int = 13
    pelvis_type: str = "random"  # random, single, divided
    pelvis_radius_x_mm: Tuple[float, float] = (7.0, 11.5)
    pelvis_radius_y_mm: Tuple[float, float] = (4.5, 8.0)
    pelvis_radius_z_mm: Tuple[float, float] = (8.5, 13.5)
    pelvis_noise_mm: float = 0.25
    surface_noise_mm: float = 0.08

    # Tube dimensions
    ureter_radius_mm: Tuple[float, float] = (2.1, 3.3)
    upj_radius_mm: Tuple[float, float] = (2.9, 4.1)
    infundibulum_radius_mm: Tuple[float, float] = (1.45, 2.9)
    calyx_cup_radius_mm: Tuple[float, float] = (2.9, 5.4)
    lower_pole_angle_degrees: Tuple[float, float] = (35.0, 75.0)

    # Clean lower-pole access model for navigation/control.
    lower_pole_clean_access: bool = True
    lower_access_trunk_length_mm: Tuple[float, float] = (28.0, 46.0)
    lower_access_curve_mm: Tuple[float, float] = (1.5, 5.5)
    lower_access_radius_mm: Tuple[float, float] = (2.3, 3.7)
    lower_calyx_splay_degrees: Tuple[float, float] = (-38.0, 18.0)
    lower_calyx_branch_length_mm: Tuple[float, float] = (9.0, 20.0)
    lower_pole_access: str = "intermediate"  # easy, intermediate, hard, random

    # Compatibility aliases retained for older config files.
    lower_pole_major_length_mm: Tuple[float, float] = (20.0, 38.0)
    lower_pole_iua_degrees: Tuple[float, float] = (25.0, 62.0)
    lower_pole_hub_radius_mm: Tuple[float, float] = (2.4, 4.2)
    lower_pole_minor_branch_length_mm: Tuple[float, float] = (8.0, 20.0)
    lower_pole_fan_y_mm: Tuple[float, float] = (5.5, 12.0)
    lower_pole_fan_angle_degrees: Tuple[float, float] = (-32.0, 28.0)
    lower_pole_access_length_mm: Tuple[float, float] = (11.0, 20.0)
    lower_pole_minor_length_mm: Tuple[float, float] = (11.0, 25.0)
    lower_pole_gate_radius_mm: Tuple[float, float] = (2.4, 4.2)

    # Ureter entry tube
    include_ureter_stub: bool = True
    ureter_stub_length_mm: Tuple[float, float] = (90.0, 135.0)
    ureter_segment_count: int = 5
    ureter_curve_mm: Tuple[float, float] = (2.0, 6.5)

    # Branch-placement / collision-avoidance settings
    branch_sample_attempts: int = 140
    graph_clean_retry_attempts: int = 12
    cup_center_clearance_mm: float = 8.0
    tube_clearance_mm: float = 1.65
    cup_to_pelvis_clearance_mm: float = 2.25
    geometry_qa_warn_clearance_mm: float = 0.0

    # Scene content
    stone_count: int = 3
    stone_radius_mm: Tuple[float, float] = (1.5, 5.0)
    stone_irregularity: float = 0.22

    # Scope/control metadata. These are not physics yet; they are
    # limits and defaults a real-time simulator can consume.
    scope_outer_diameter_mm: float = 3.0
    scope_tip_length_mm: float = 12.0
    scope_fov_degrees: float = 85.0
    scope_start_region: str = "ureter_stub"
    scope_max_deflection_deg: float = 275.0
    default_camera_rate_hz: int = 30
    default_control_rate_hz: int = 15
    camera_path_spacing_mm: float = 3.5

    # Coordinate system note: x = lateral branch direction, y = anterior/posterior, z = cranio-caudal.
    coordinate_system: str = "x_lateral_y_ap_z_craniocaudal_mm"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GeneratorConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        valid = set(cls.__dataclass_fields__.keys())
        unknown = sorted(set(data.keys()) - valid)
        if unknown:
            raise ValueError(f"Unknown config keys in {path}: {unknown}")
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


DEFAULT_CONFIG = GeneratorConfig()
