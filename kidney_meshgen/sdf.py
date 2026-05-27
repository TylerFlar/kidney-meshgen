from __future__ import annotations

from typing import Any, Tuple

import numpy as np


def _value(primitive: Any, key: str, default=None):
    if isinstance(primitive, dict):
        return primitive.get(key, default)
    return getattr(primitive, key, default)


def _normalize(v: np.ndarray, fallback: Tuple[float, float, float] = (1.0, 0.0, 0.0)) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return np.asarray(fallback, dtype=float)
    return v / n


def primitive_profile_max_scale(primitive: Any) -> float:
    scale = 1.0
    for key in ("cross_section_scale0", "cross_section_scale1"):
        values = _value(primitive, key)
        if values is not None:
            scale = max(scale, max(float(v) for v in values))
    return scale


def primitive_bounds(primitive: Any) -> Tuple[np.ndarray, np.ndarray]:
    kind = _value(primitive, "kind")
    if kind == "ellipsoid":
        center = np.asarray(_value(primitive, "center"), dtype=float)
        radii = np.asarray(_value(primitive, "radii"), dtype=float)
        return center - radii, center + radii
    if kind == "tapered_capsule":
        p0 = np.asarray(_value(primitive, "p0"), dtype=float)
        p1 = np.asarray(_value(primitive, "p1"), dtype=float)
        radius = max(float(_value(primitive, "r0")), float(_value(primitive, "r1")))
        radius *= primitive_profile_max_scale(primitive)
        return np.minimum(p0, p1) - radius, np.maximum(p0, p1) + radius
    raise ValueError(f"unsupported primitive kind: {kind!r}")


def _tube_profile_arrays(primitive: Any, axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    axis = _normalize(axis)
    cross_u = _value(primitive, "cross_section_u")
    cross_v = _value(primitive, "cross_section_v")
    if cross_u is not None and cross_v is not None:
        u = _normalize(np.asarray(cross_u, dtype=float), (0.0, 1.0, 0.0))
        v = _normalize(np.asarray(cross_v, dtype=float), (0.0, 0.0, 1.0))
        u = _normalize(u - axis * float(np.dot(u, axis)), (0.0, 1.0, 0.0))
        v = _normalize(v - axis * float(np.dot(v, axis)) - u * float(np.dot(v, u)), (0.0, 0.0, 1.0))
    else:
        reference = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(float(np.dot(axis, reference))) > 0.88:
            reference = np.array([0.0, 1.0, 0.0], dtype=float)
        u = _normalize(np.cross(axis, reference), (0.0, 1.0, 0.0))
        v = _normalize(np.cross(axis, u), (0.0, 0.0, 1.0))
    scale0 = np.asarray(_value(primitive, "cross_section_scale0", (1.0, 1.0)) or (1.0, 1.0), dtype=float)
    scale1 = np.asarray(_value(primitive, "cross_section_scale1", (1.0, 1.0)) or (1.0, 1.0), dtype=float)
    return u, v, np.maximum(scale0, 1e-4), np.maximum(scale1, 1e-4)


def _narrowed_radius(radius: np.ndarray, t: np.ndarray, primitive: Any) -> np.ndarray:
    fraction = float(_value(primitive, "narrowing_fraction", 0.0) or 0.0)
    width = float(_value(primitive, "narrowing_width", 0.0) or 0.0)
    if fraction <= 0.0 or width <= 1e-5:
        return radius
    center = _value(primitive, "narrowing_t", 0.5)
    center = 0.5 if center is None else float(center)
    profile = np.exp(-0.5 * ((t - center) / width) ** 2)
    return np.maximum(radius * (1.0 - fraction * profile), radius * 0.35)


def primitive_sdf(points: np.ndarray, primitive: Any) -> np.ndarray:
    """Approximate signed distance. Negative values are inside the lumen volume."""
    pts = np.asarray(points, dtype=float)
    kind = _value(primitive, "kind")
    if kind == "ellipsoid":
        center = np.asarray(_value(primitive, "center"), dtype=float)
        radii = np.asarray(_value(primitive, "radii"), dtype=float)
        q = (pts - center[None, :]) / np.maximum(radii[None, :], 1e-6)
        return (np.linalg.norm(q, axis=1) - 1.0) * float(np.min(radii))

    if kind == "tapered_capsule":
        p0 = np.asarray(_value(primitive, "p0"), dtype=float)
        p1 = np.asarray(_value(primitive, "p1"), dtype=float)
        r0 = float(_value(primitive, "r0"))
        r1 = float(_value(primitive, "r1"))
        axis_vec = p1 - p0
        axis_len2 = float(np.dot(axis_vec, axis_vec))
        if axis_len2 < 1e-8:
            return np.linalg.norm(pts - p0[None, :], axis=1) - max(r0, r1)
        t = np.clip(((pts - p0[None, :]) @ axis_vec) / axis_len2, 0.0, 1.0)
        closest = p0[None, :] + t[:, None] * axis_vec[None, :]
        radius = _narrowed_radius(r0 + t * (r1 - r0), t, primitive)
        axis = axis_vec / np.sqrt(axis_len2)
        u, v, scale0, scale1 = _tube_profile_arrays(primitive, axis)
        scales = scale0[None, :] * (1.0 - t[:, None]) + scale1[None, :] * t[:, None]
        delta = pts - closest
        du = delta @ u
        dv = delta @ v
        da = delta @ axis
        su = np.maximum(radius * scales[:, 0], 1e-6)
        sv = np.maximum(radius * scales[:, 1], 1e-6)
        sr = np.maximum(radius, 1e-6)
        q = np.sqrt((du / su) ** 2 + (dv / sv) ** 2 + (da / sr) ** 2)
        return (q - 1.0) * np.minimum.reduce([su, sv, sr])

    raise ValueError(f"unsupported primitive kind: {kind!r}")
