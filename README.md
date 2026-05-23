# kidney-meshgen

Procedural renal collecting-system mesh generator for **real-time flexible ureteroscopy / RIRS navigation and direct-control simulation**.

This package is intentionally lean. It focuses on the assets needed to build and test a real-time simulator without pulling in BlenderProc, Isaac/Omniverse, USD, or dataset-generation workflows that are not needed yet.

Generated anatomy and runtime assets include:

```text
ureter entry tube
renal pelvis / UPJ
upper, middle, and lower calyces
cleaner lower-pole access trunk
stones
visual lumen mesh
collision proxy
approximate SDF grid
centerline graph
navigation waypoints
coverage points
runtime_scene.json descriptor
Unity metadata helpers
```

This is for AI simulation and research prototyping, not clinical diagnosis, patient-specific surgical planning, or treatment guidance.

## Version 0.6

This release completes the project rename to `kidney-meshgen`:

- Distribution and CLI name: `kidney-meshgen`
- Python package import: `kidney_meshgen`
- Generated schemas and Unity helpers use `kidney_meshgen` / `KidneyMeshgen`
- Project setup uses `uv`, with `uv.lock` committed for reproducible installs
- Development tooling includes Ruff and pytest through the `dev` dependency group

## Install

```bash
uv sync
```

Optional test/preview tools:

```bash
uv sync --extra preview --dev
```

## Generate one case

```bash
uv run kidney-meshgen generate \
  --out output/kidney_case \
  --seed 7 \
  --anatomy-id kidney_case \
  --stones 3
```

Higher-resolution preset:

```bash
uv run kidney-meshgen generate \
  --config configs/high_res.yaml \
  --out output/high_res_case \
  --seed 21
```

Fast smoke-test case:

```bash
uv run kidney-meshgen generate \
  --out output/smoke \
  --seed 1 \
  --grid 96 \
  --min-grid-axis 56 \
  --stones 2 \
  --no-glb \
  --no-sdf-grid
```

## Important outputs

Each generated case contains:

```text
scene_manifest.json                  Full machine-readable case manifest
runtime_scene.json                    Runtime descriptor for a Unity / Gym bridge
centerline_graph.json                 Nodes, edges, radii, regions, calyx targets
camera_paths.json                     Dense centerline samples and routes to calyces
lumen_inner.obj/.glb                  Inward-facing visual mesh for endoscopic rendering
lumen_outer.obj/.glb                  Outward-facing mesh for inspection/debugging
stones/stone_###.obj                  Individual stone meshes
stones/stones.obj/.glb                Combined stones
regions/*.obj                         Per-region semantic submeshes
labels.json                           Label ID map
face_labels.csv                       Mesh face labels
vertex_labels.csv                     Mesh vertex labels
waypoints/navigation_waypoints.json   Navigation/control waypoints
waypoints/navigation_waypoints.csv    Compact waypoint table
coverage/coverage_points.csv          Surface samples for coverage scoring
collision/lumen_collision_proxy.obj   Collision mesh
collision/lumen_sdf_grid.npz          Approximate analytic SDF grid for clearance checks
collision/lumen_sdf_grid.json         SDF grid metadata
quality/geometry_quality.json         Geometry QA report
unity/kidney_meshgen_unity_scene.json Unity-oriented descriptor
unity/*.cs                            Lightweight Unity metadata loader scripts
preview_centerline.png                Quick QA preview
resolved_config.yaml                  Exact reproducibility config
```

Use `lumen_inner.*` for the endoscopic camera. The proximal ureter start is **open by default**, so the scope can begin in a tube instead of looking at a rounded cap.

## Runtime descriptor

`runtime_scene.json` is the file I would load first in a Unity simulator. It points to the visual mesh, collision mesh, approximate SDF grid, stones, waypoints, labels, and task definitions.

Recommended first simulator loop:

```text
Observation:
  RGB frame
  optional depth / semantic mask
  scope pose and scope state
  nearest centerline edge/t
  visible stone pixels
  coverage state
  clearance/contact state

Action:
  advance/retract
  roll
  primary deflection
  optional secondary deflection
  laser fiber advance/retract
  basket advance/retract
  basket open/close

Metrics:
  entry-to-pelvis success
  calyx coverage
  lower-pole access success
  stone finding success
  time to first/all stones
  wall contacts per minute
  minimum clearance
  lost-view events
  trajectory length
  oscillation score
```

## Generator Capabilities

The generator keeps optional export paths narrow and focuses on simulator-ready assets:

- **Open ureter entry**: the proximal ureter cap is cut open by default with `open_ureter_start: true`.
- **Longer entry tube**: default ureter length is increased so the scope starts in a realistic approach tube.
- **Cleaner lower pole**: the lower pole uses an explicit access trunk and minor-calyx fan-out.
- **Stronger branch clearance**: retry sampling and increased clearances reduce accidental chamber fusion.
- **Collision assets**: outputs both a collision proxy mesh and an approximate SDF grid.
- **Navigation waypoints**: route waypoints include radius and estimated scope clearance.
- **Lean runtime descriptor**: `runtime_scene.json` replaces BlenderProc/Isaac descriptors.

## Useful config knobs

```yaml
open_ureter_start: true
open_ureter_start_offset_mm: 1.2
lower_pole_access: intermediate   # easy, intermediate, hard, random
graph_clean_retry_attempts: 12
branch_sample_attempts: 140
cup_center_clearance_mm: 8.0
tube_clearance_mm: 1.65
scope_outer_diameter_mm: 3.0
export_sdf_grid: true
export_collision_proxy: true
```

## Python usage

```python
from kidney_meshgen import GeneratorConfig, generate_case

cfg = GeneratorConfig(
    seed=42,
    anatomy_id="kidney_042",
    stone_count=4,
    lower_pole_access="intermediate",
)
manifest = generate_case(cfg, "output/kidney_042")
print(manifest["simulator"])
```

## Unity usage

1. Generate a case.
2. Import `lumen_inner.glb` or `lumen_inner.obj` as the visual lumen.
3. Import `collision/lumen_collision_proxy.obj` as a static collision mesh.
4. Import `stones/stones.glb` or individual `stones/stone_###.obj` meshes.
5. Add `scene_manifest.json` and `runtime_scene.json` as TextAssets.
6. Use `unity/KidneyMeshgenSceneLoader.cs` to parse metadata and place the scope at the generated start pose.
7. Use `waypoints/navigation_waypoints.json` for scripted QA paths, imitation-learning rollouts, or centerline progress rewards.

## Tests

```bash
uv run pytest tests
uv run ruff check .
```

## Limitations

The generator does not model tissue deformation, real ureteroscope shaft mechanics, irrigation fluid physics, blood/dust/bubbles, or clinically validated population distributions. It is meant to give you a clean procedural environment that can later be connected to better deformation, instrument-control, and rendering modules.
