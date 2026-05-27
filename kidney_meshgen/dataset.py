from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SEMANTIC_LABELS: Dict[str, int] = {
    "background": 0,
    "lumen_mucosa": 1,
    "stone": 1001,
    "suspended_stone_dust": 1002,
    "irrigation_air_bubble": 1003,
}

MODALITY_FORMATS: Dict[str, str] = {
    "rgb": "png",
    "depth": "float32 npy",
    "normals": "float32 npy",
    "semantic": "uint16 png",
}

_FRAME_FILE_TEMPLATES: Dict[str, Tuple[str, str, str]] = {
    "rgb": ("rgb", "frame_", "png"),
    "depth": ("depth", "depth_", "npy"),
    "normals": ("normals", "normals_", "npy"),
    "semantic": ("semantic", "semantic_", "png"),
}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _path_for_frame(modality: str, frame_index: int) -> str:
    directory, prefix, suffix = _FRAME_FILE_TEMPLATES[modality]
    return f"{directory}/{prefix}{int(frame_index):06d}.{suffix}"


def _as_float_matrix(values: Sequence[Sequence[float]]) -> List[List[float]]:
    return [[float(v) for v in row] for row in values]


def _intrinsics_from_horizontal_fov(
    width: int,
    height: int,
    horizontal_fov_degrees: float,
    principal_point_fraction: Sequence[float] = (0.5, 0.5),
) -> List[List[float]]:
    fx = float(width) / (2.0 * math.tan(math.radians(float(horizontal_fov_degrees)) * 0.5))
    fy = fx
    cx = float(principal_point_fraction[0]) * float(width)
    cy = float(principal_point_fraction[1]) * float(height)
    return [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]


def build_camera_intrinsics(
    sensor_model: Mapping[str, Any],
    width: int,
    height: int,
    clip_start_mm: float,
    clip_end_mm: float,
    plan_fov_degrees: float,
    lens_mapping: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the dataset-side intrinsics payload.

    BlenderProc can be configured from either a K matrix or a FOV. Downstream
    dataset loaders usually want an explicit K, so the FOV path is resolved here.
    """

    if sensor_model.get("K") is None:
        k_matrix = _intrinsics_from_horizontal_fov(
            int(width),
            int(height),
            float(sensor_model.get("horizontal_fov_degrees") or plan_fov_degrees),
        )
    else:
        k_matrix = _as_float_matrix(sensor_model["K"])

    fx = float(k_matrix[0][0])
    fy = float(k_matrix[1][1])
    cx = float(k_matrix[0][2])
    cy = float(k_matrix[1][2])
    horizontal_fov = math.degrees(2.0 * math.atan(float(width) / max(2.0 * fx, 1e-12)))
    vertical_fov = math.degrees(2.0 * math.atan(float(height) / max(2.0 * fy, 1e-12)))
    lens_enabled = bool(sensor_model.get("lens_distortion_enabled", False))

    return {
        "schema": "kidney_meshgen_camera_intrinsics_v0.1",
        "camera_model": "pinhole_brown_conrady_opencv" if lens_enabled else "pinhole",
        "units": {"length": "millimeters", "image": "pixels"},
        "resolution": [int(width), int(height)],
        "K": k_matrix,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "horizontal_fov_degrees": float(horizontal_fov),
        "vertical_fov_degrees": float(vertical_fov),
        "clip_start_mm": float(clip_start_mm),
        "clip_end_mm": float(clip_end_mm),
        "distortion_model": sensor_model.get("distortion_model", "brown_conrady_opencv"),
        "distortion": {
            name: float(dict(sensor_model.get("distortion", {})).get(name, 0.0))
            for name in ("k1", "k2", "k3", "p1", "p2")
        },
        "lens_distortion_enabled": lens_enabled,
        "lens_distortion_postprocess": {
            "enabled": lens_mapping is not None,
            "original_resolution": None
            if lens_mapping is None
            else [int(lens_mapping["orig_res_x"]), int(lens_mapping["orig_res_y"])],
        },
        "sensor_profile": sensor_model.get("profile"),
        "coordinate_conventions": {
            "image_origin": "top_left",
            "pixel_coordinates": "pixels",
            "pose_matrix": "cam2world_opengl",
            "camera_frame": "OpenGL camera frame; camera looks along local -Z and +Y is up.",
            "world_units": "millimeters",
        },
    }


def write_camera_intrinsics_file(
    out_dir: str | Path,
    sensor_model: Mapping[str, Any],
    width: int,
    height: int,
    clip_start_mm: float,
    clip_end_mm: float,
    plan_fov_degrees: float,
    lens_mapping: Optional[Mapping[str, Any]] = None,
) -> Path:
    out_dir = Path(out_dir)
    path = out_dir / "camera_intrinsics.json"
    _write_json(
        path,
        build_camera_intrinsics(
            sensor_model,
            width,
            height,
            clip_start_mm,
            clip_end_mm,
            plan_fov_degrees,
            lens_mapping=lens_mapping,
        ),
    )
    return path


def _enabled_modalities(output_paths: Mapping[str, str]) -> List[str]:
    modalities = ["rgb"]
    if output_paths.get("depth_dir"):
        modalities.append("depth")
    if output_paths.get("normals_dir"):
        modalities.append("normals")
    if output_paths.get("semantic_dir"):
        modalities.append("semantic")
    return modalities


def build_frame_index(plan: Mapping[str, Any], output_paths: Mapping[str, str]) -> Dict[str, Any]:
    modalities = _enabled_modalities(output_paths)
    frames = []
    for frame in plan.get("frames", []):
        frame_index = int(frame["frame_index"])
        files = {modality: _path_for_frame(modality, frame_index) for modality in modalities}
        frames.append(
            {
                "frame_index": frame_index,
                "frame_id": f"{frame_index:06d}",
                "time_s": float(frame.get("time_s", 0.0)),
                "files": files,
                "pose": {
                    "cam2world_opengl": frame.get("cam2world_opengl"),
                    "position_mm": frame.get("position_mm"),
                    "forward": frame.get("forward"),
                    "up": frame.get("up"),
                    "edge_id": frame.get("edge_id"),
                    "edge_t": frame.get("edge_t"),
                    "region": frame.get("region"),
                    "kind": frame.get("kind"),
                    "radius_mm": frame.get("radius_mm"),
                    "sdf_mm": frame.get("sdf_mm"),
                },
            }
        )

    return {
        "schema": "kidney_meshgen_dataset_frames_v0.1",
        "units": {"length": "millimeters", "time": "seconds"},
        "source_pose_schema": plan.get("schema"),
        "frame_count": int(plan.get("frame_count", len(frames))),
        "modalities": modalities,
        "modality_formats": {name: MODALITY_FORMATS[name] for name in modalities},
        "frames": frames,
    }


def parse_split_ratios(value: str | Sequence[float] | Mapping[str, float]) -> Dict[str, float]:
    if isinstance(value, Mapping):
        raw = [float(value.get(name, 0.0)) for name in ("train", "val", "test")]
    elif isinstance(value, str):
        text = value.strip()
        if "=" in text:
            parts = {}
            for item in text.replace(";", ",").split(","):
                if not item.strip():
                    continue
                key, ratio = item.split("=", 1)
                parts[key.strip().lower()] = float(ratio)
            raw = [float(parts.get(name, 0.0)) for name in ("train", "val", "test")]
        else:
            raw = [float(part.strip()) for part in text.replace(";", ",").split(",") if part.strip()]
    else:
        raw = [float(v) for v in value]

    if len(raw) != 3:
        raise ValueError("split ratios must contain train,val,test values")
    if any(v < 0.0 for v in raw):
        raise ValueError("split ratios must be non-negative")
    total = sum(raw)
    if total <= 0.0:
        raise ValueError("at least one split ratio must be positive")
    return {name: float(v / total) for name, v in zip(("train", "val", "test"), raw)}


def _split_counts(frame_count: int, ratios: Mapping[str, float]) -> Dict[str, int]:
    count = max(int(frame_count), 0)
    names = ("train", "val", "test")
    exact = {name: float(ratios[name]) * count for name in names}
    counts = {name: int(math.floor(exact[name])) for name in names}
    remaining = count - sum(counts.values())
    remainders = sorted(names, key=lambda name: (exact[name] - counts[name], ratios[name]), reverse=True)
    for name in remainders[:remaining]:
        counts[name] += 1
    return counts


def split_frame_indices(
    frame_count: int,
    ratios: str | Sequence[float] | Mapping[str, float] = (0.8, 0.1, 0.1),
    seed: Optional[int] = 0,
    shuffle: bool = True,
) -> Dict[str, List[int]]:
    normalized = parse_split_ratios(ratios)
    frame_indices = list(range(max(int(frame_count), 0)))
    if shuffle:
        random.Random(seed).shuffle(frame_indices)
    counts = _split_counts(len(frame_indices), normalized)
    train_end = counts["train"]
    val_end = train_end + counts["val"]
    return {
        "train": sorted(frame_indices[:train_end]),
        "val": sorted(frame_indices[train_end:val_end]),
        "test": sorted(frame_indices[val_end:]),
    }


def write_split_files(
    out_dir: str | Path,
    frame_count: int,
    ratios: str | Sequence[float] | Mapping[str, float] = (0.8, 0.1, 0.1),
    seed: Optional[int] = 0,
) -> Dict[str, str]:
    out_dir = Path(out_dir)
    split_dir = out_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    normalized = parse_split_ratios(ratios)
    splits = split_frame_indices(frame_count, normalized, seed=seed)
    paths: Dict[str, str] = {}
    for split_name, indices in splits.items():
        path = split_dir / f"{split_name}.txt"
        path.write_text("".join(f"{idx:06d}\n" for idx in indices), encoding="utf-8")
        paths[f"split_{split_name}_txt"] = f"splits/{split_name}.txt"

    payload = {
        "schema": "kidney_meshgen_dataset_splits_v0.1",
        "split_unit": "frame",
        "seed": None if seed is None else int(seed),
        "ratios": normalized,
        "counts": {name: len(indices) for name, indices in splits.items()},
        "frame_ids": {name: [f"{idx:06d}" for idx in indices] for name, indices in splits.items()},
        "frame_indices": splits,
    }
    _write_json(split_dir / "splits.json", payload)
    paths["splits_json"] = "splits/splits.json"
    return paths


def _count_active_events(events: Iterable[Mapping[str, Any]], frame_index: int) -> int:
    count = 0
    for event in events:
        if int(event.get("start_frame", 0)) <= frame_index < int(event.get("end_frame", 0)):
            count += 1
    return count


def build_per_frame_randomization(plan: Mapping[str, Any], fluid_model: Mapping[str, Any]) -> List[Dict[str, Any]]:
    lens = dict(fluid_model.get("lens", {}))
    if not bool(fluid_model.get("enabled", False)) or not bool(lens.get("enabled", False)):
        return []
    droplets = list(lens.get("droplets", []))
    occlusions = list(lens.get("occlusions", []))
    return [
        {
            "frame_index": int(frame["frame_index"]),
            "lens_film_strength": float(lens.get("film_strength", 0.0)),
            "active_lens_droplets": _count_active_events(droplets, int(frame["frame_index"])),
            "active_lens_occlusions": _count_active_events(occlusions, int(frame["frame_index"])),
        }
        for frame in plan.get("frames", [])
    ]


def build_randomization_metadata(
    manifest: Mapping[str, Any],
    plan: Mapping[str, Any],
    render_seed: int,
    split_seed: Optional[int],
    sensor_model: Mapping[str, Any],
    realism: Mapping[str, Any],
    fluid_model: Mapping[str, Any],
    material_preset_arg: str,
    light_preset_arg: str,
    randomize_realism: bool,
) -> Dict[str, Any]:
    return {
        "schema": "kidney_meshgen_dataset_randomization_v0.1",
        "scope": "sequence",
        "generator_seed": manifest.get("seed"),
        "render_seed": int(render_seed),
        "split_seed": None if split_seed is None else int(split_seed),
        "anatomy_id": manifest.get("anatomy_id"),
        "case_randomization": {
            "side": manifest.get("config", {}).get("side"),
            "anatomy_realism_profile": manifest.get("config", {}).get("anatomy_realism_profile"),
            "pelvicalyceal_class": manifest.get("anatomy_metadata", {}).get("pelvicalyceal_class"),
            "takazawa_type": manifest.get("anatomy_metadata", {}).get("takazawa_type"),
            "lower_pole_access": manifest.get("anatomy_metadata", {}).get("lower_pole_access_mode"),
            "stone_count": len(manifest.get("stones", [])),
            "stone_materials": [stone.get("material_class") for stone in manifest.get("stones", [])],
            "stone_states": [stone.get("state") for stone in manifest.get("stones", [])],
        },
        "render_randomization": {
            "sensor_profile": sensor_model.get("profile"),
            "material_preset_requested": material_preset_arg,
            "light_preset_requested": light_preset_arg,
            "randomize_realism": bool(randomize_realism),
            "realism": realism,
            "fluid": fluid_model,
        },
        "per_frame": build_per_frame_randomization(plan, fluid_model),
    }


def write_dataset_convention_files(
    out_dir: str | Path,
    manifest: Mapping[str, Any],
    plan: Mapping[str, Any],
    output_paths: Mapping[str, str],
    sensor_model: Mapping[str, Any],
    width: int,
    height: int,
    clip_start_mm: float,
    clip_end_mm: float,
    render_seed: int,
    split_ratios: str | Sequence[float] | Mapping[str, float],
    split_seed: Optional[int],
    write_splits: bool,
    realism: Mapping[str, Any],
    fluid_model: Mapping[str, Any],
    material_preset_arg: str,
    light_preset_arg: str,
    randomize_realism: bool,
    lens_mapping: Optional[Mapping[str, Any]] = None,
    pose_file: Optional[str | Path] = None,
) -> Dict[str, str]:
    out_dir = Path(out_dir)
    paths: Dict[str, str] = {}

    write_camera_intrinsics_file(
        out_dir,
        sensor_model,
        int(width),
        int(height),
        float(clip_start_mm),
        float(clip_end_mm),
        float(plan.get("fov_degrees", 85.0)),
        lens_mapping=lens_mapping,
    )
    paths["camera_intrinsics_json"] = "camera_intrinsics.json"

    frame_index = build_frame_index(plan, output_paths)
    _write_json(out_dir / "frames.json", frame_index)
    paths["frames_json"] = "frames.json"

    randomization = build_randomization_metadata(
        manifest,
        plan,
        int(render_seed),
        split_seed,
        sensor_model,
        realism,
        fluid_model,
        material_preset_arg,
        light_preset_arg,
        bool(randomize_realism),
    )
    _write_json(out_dir / "randomization.json", randomization)
    paths["randomization_json"] = "randomization.json"

    if write_splits:
        paths.update(write_split_files(out_dir, int(frame_index["frame_count"]), split_ratios, seed=split_seed))

    manifest_payload = {
        "schema": "kidney_meshgen_dataset_manifest_v0.1",
        "case_anatomy_id": manifest.get("anatomy_id"),
        "case_dir": str(manifest.get("case_dir", "")),
        "pose_file": None if pose_file is None else str(pose_file),
        "frame_count": int(frame_index["frame_count"]),
        "modalities": frame_index["modalities"],
        "files": {
            "camera_intrinsics": "camera_intrinsics.json",
            "frames": "frames.json",
            "randomization": "randomization.json",
            "splits": "splits/splits.json" if write_splits else None,
            "render_metadata": "render_metadata.json",
        },
        "semantic_labels": SEMANTIC_LABELS,
        "future_exports": {
            "hdf5": "not_implemented",
            "bop": "not_implemented",
        },
    }
    _write_json(out_dir / "dataset_manifest.json", manifest_payload)
    paths["dataset_manifest_json"] = "dataset_manifest.json"
    return paths
