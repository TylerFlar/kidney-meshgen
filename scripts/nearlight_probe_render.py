import blenderproc as bproc  # noqa: F401,E402  (MUST be the first import for BlenderProc)

# NEARLIGHT-GATE-001 controlled linear-radiance probe render.
#
# Standalone BlenderProc scene (NOT the slam_eval generator path) that renders a
# co-located near-light onto a fronto-parallel Lambertian plane and a Lambertian
# sphere across a camera-surface distance ladder, emitting **linear** scene
# radiance (.npy + OPEN_EXR, view_transform=Raw, no look/exposure/sensor effects).
#
# Physics go/no-go for the near-light inverse-square cue: with a co-located point
# light and a black world, a Lambertian surface obeys
#     L(u,v) = (albedo/pi) * I * cos(theta) / r^2
# so log(L / cos theta) vs log(r) must regress to slope = -2.
#
# Run (from kidney-meshgen, with the render venv's blenderproc):
#   blenderproc run scripts/nearlight_probe_render.py -- --out output/nearlight_probe --target plane
#   blenderproc run scripts/nearlight_probe_render.py -- --out output/nearlight_probe --target sphere

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _make_lambertian(name: str, albedo: float):
    mat = bproc.material.create(name)
    a = float(albedo)
    mat.set_principled_shader_value("Base Color", [a, a, a, 1.0])
    mat.set_principled_shader_value("Roughness", 1.0)
    mat.set_principled_shader_value("Metallic", 0.0)
    # Kill the specular lobe: with a co-located light the mirror highlight lands
    # exactly on the on-axis sample we read, so a non-zero specular would corrupt
    # the slope. Input was renamed across Blender versions -> try both names.
    for key in ("Specular IOR Level", "Specular"):
        try:
            mat.set_principled_shader_value(key, 0.0)
            break
        except Exception:
            continue
    for key in ("Coat Weight", "Sheen Weight"):
        try:
            mat.set_principled_shader_value(key, 0.0)
        except Exception:
            pass
    return mat


def _configure_linear_color_management():
    import bpy

    vs = bpy.context.scene.view_settings
    for transform in ("Raw", "Standard"):
        try:
            vs.view_transform = transform
            break
        except TypeError:
            continue
    try:
        vs.look = "None"
    except TypeError:
        pass
    vs.exposure = 0.0
    vs.gamma = 1.0
    return {
        "view_transform": vs.view_transform,
        "look": vs.look,
        "exposure": float(vs.exposure),
        "gamma": float(vs.gamma),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", choices=["plane", "sphere"], required=True)
    ap.add_argument("--distances-mm", default="3,4,5,6,8,10,12,15,20")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--fov-deg", type=float, default=60.0)
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--light-energy", type=float, default=10.0)
    ap.add_argument("--light-baseline-mm", type=float, default=0.0,
                    help="point-light offset behind the camera along +Z (b); 0 = co-located")
    ap.add_argument("--albedo", type=float, default=0.5)
    ap.add_argument("--sphere-radius-mm", type=float, default=2.0)
    ap.add_argument("--plane-halfsize-mm", type=float, default=60.0)
    args = ap.parse_args()

    distances = [float(x) for x in args.distances_mm.split(",")]
    out_dir = Path(args.out) / args.target
    out_dir.mkdir(parents=True, exist_ok=True)

    bproc.init()
    cm = _configure_linear_color_management()

    # Black world: only the co-located point light illuminates the surface, so
    # the recorded radiance is pure near-light direct shading.
    bproc.renderer.set_world_background([0.0, 0.0, 0.0], strength=0.0)

    mat = _make_lambertian(f"lam_{args.target}", args.albedo)

    sphere_center = [0.0, 0.0, 0.0]
    sphere_radius = float(args.sphere_radius_mm)
    if args.target == "plane":
        obj = bproc.object.create_primitive("PLANE")
        obj.set_location([0.0, 0.0, 0.0])  # lies in z=0, normal +Z toward camera
        obj.set_scale([float(args.plane_halfsize_mm), float(args.plane_halfsize_mm), 1.0])
    else:
        obj = bproc.object.create_primitive("SPHERE")
        obj.set_location(sphere_center)
        obj.set_scale([sphere_radius, sphere_radius, sphere_radius])
    obj.add_material(mat)
    try:
        obj.set_shading_mode("SMOOTH")
    except Exception:
        pass

    res = int(args.res)
    bproc.camera.set_intrinsics_from_blender_params(
        lens=math.radians(float(args.fov_deg)),
        image_width=res,
        image_height=res,
        clip_start=0.02,
        clip_end=5000.0,
        lens_unit="FOV",
    )
    fx = (res / 2.0) / math.tan(math.radians(float(args.fov_deg)) / 2.0)
    fy = fx
    cx = res / 2.0
    cy = res / 2.0

    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_energy(float(args.light_energy))
    light.set_color([1.0, 1.0, 1.0])

    b = float(args.light_baseline_mm)
    for i, d in enumerate(distances):
        cam2world = np.eye(4)
        cam2world[:3, 3] = [0.0, 0.0, d]  # camera on +Z axis looking -Z (identity rot)
        bproc.camera.add_camera_pose(cam2world, frame=i)
        light.set_location([0.0, 0.0, d + b], frame=i)
        light.set_energy(float(args.light_energy), frame=i)

    bproc.renderer.set_max_amount_of_samples(int(args.samples))
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    try:
        bproc.renderer.enable_normals_output()
        normals_enabled = True
    except Exception:
        normals_enabled = False
    try:
        bproc.renderer.set_output_format(file_format="OPEN_EXR", color_depth="32")
    except Exception:
        pass

    data = bproc.renderer.render()

    colors = data.get("colors")
    depth = data.get("depth")
    normals = data.get("normals") if normals_enabled else None

    for i, d in enumerate(distances):
        np.save(out_dir / f"colors_{i:06d}.npy", np.asarray(colors[i], dtype=np.float32))
        np.save(out_dir / f"depth_{i:06d}.npy", np.asarray(depth[i], dtype=np.float32))
        if normals is not None:
            np.save(out_dir / f"normals_{i:06d}.npy", np.asarray(normals[i], dtype=np.float32))

    meta = {
        "experiment": "NEARLIGHT-GATE-001",
        "target": args.target,
        "distances_mm": distances,
        "resolution": [res, res],
        "fov_deg": float(args.fov_deg),
        "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        "samples": int(args.samples),
        "albedo": float(args.albedo),
        "light": {
            "type": "POINT",
            "energy_w": float(args.light_energy),
            "baseline_mm_b": b,
            "co_located": b == 0.0,
        },
        "sphere": {"center": sphere_center, "radius_mm": sphere_radius},
        "world_background": {"color": [0, 0, 0], "strength": 0.0},
        "color_management": cm,
        "depth_convention": "planar_z (Blender unit == mm); background = inf/nan",
        "normals_enabled": normals is not None,
        "units_note": "1 Blender unit treated as 1 mm; inverse-square slope is scale-invariant",
    }
    with open(out_dir / "probe_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps({"target": args.target, "frames": len(distances), "out": str(out_dir),
                      "color_management": cm}))


if __name__ == "__main__":
    main()
