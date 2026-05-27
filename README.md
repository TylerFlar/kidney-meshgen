# kidney-meshgen

Procedural renal collecting-system mesh generator for **real-time flexible ureteroscopy / RIRS navigation and direct-control simulation**.

The core generator is intentionally lean. It focuses on the assets needed to build and test a real-time simulator, while higher-cost offline rendering tools such as BlenderProc stay behind optional extras.

Generated anatomy and runtime assets include:

```text
ureter entry tube
renal pelvis / UPJ
Takazawa-style top, upper, middle, lower, and bottom calyces
anterior/posterior calyx pairs where applicable
papilla/fornix cup surface detail
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

## Anatomy model

The current anatomy realism profile includes:

- Default `takazawa` anatomy profile with Type I single-pelvis and Type II divided-pelvis classes.
- Top/upper/middle/lower/bottom calyx naming, with upper/middle/lower anterior/posterior pairs.
- Subtractive papilla solids inside each minor calyx, producing cup-like fornices instead of smooth bulbs.
- Mild local infundibular narrowing plus non-circular, asymmetric tube cross sections.
- Visual-only mucosal folds/noise near the pelvis and calyx necks, with a smooth collision proxy/SDF.
- Region-specific sampled branch length, radius, and angle metadata in each `calyx_targets` entry.
- Geometry QA distinguishes intended proximal branch-family blending from distal/cup overlap risks.

## Preview images

These previews were sampled from a 12-case no-stone RGB-D evaluation set rendered with the realistic flexible-ureteroscope sensor profile.

![Realistic RGB endoscope montage](docs/rgb_montage.png)

![Colorized depth montage](docs/depth_montage.png)

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

Choose a material cohort or force a laser-fragmented gravel field:

```bash
uv run kidney-meshgen generate \
  --out output/gravel_case \
  --stone-materials COM,COD,uric_acid,struvite_apatite,cystine \
  --stone-fragmentation gravel \
  --stones 2
```

Force a Takazawa class when you want a controlled cohort:

```bash
uv run kidney-meshgen generate \
  --out output/type_ii_case \
  --pelvicalyceal-class type_ii \
  --stones 0
```

## BlenderProc rendering

Install the optional renderer dependency:

```bash
uv sync --extra render --dev
```

Render a generated case with the default DFS/backtracking camera path. Stones are **off by default**:

```bash
uv run kidney-meshgen render-blenderproc \
  --case-dir output/kidney_case \
  --out output/kidney_case/blenderproc_render
```

Turn stones on explicitly:

```bash
uv run kidney-meshgen render-blenderproc \
  --case-dir output/kidney_case \
  --include-stones
```

Write only the camera plan, without invoking BlenderProc:

```bash
uv run kidney-meshgen render-blenderproc \
  --case-dir output/kidney_case \
  --plan-only
```

Important render outputs:

```text
blenderproc_render/rgb/frame_*.png     Rendered RGB image sequence
blenderproc_render/depth/*             Optional depth frames
blenderproc_render/normals/*           Optional normal frames
blenderproc_render/semantic/*          Optional semantic category-ID frames
blenderproc_render/sensor/circular_mask.png Optional circular endoscope mask
blenderproc_render/camera_intrinsics.json Explicit K, distortion, resolution, and camera conventions
blenderproc_render/frames.json         Per-frame pose plus RGB/depth/normals/semantic file paths
blenderproc_render/camera_poses.json   Per-frame camera-to-world matrices and pose metadata
blenderproc_render/camera_poses.csv    Compact per-frame pose table
blenderproc_render/randomization.json  Sequence/per-frame randomization seed and sampled parameters
blenderproc_render/splits/*.txt        Train/val/test frame IDs
blenderproc_render/splits/splits.json  Split metadata and frame-index assignments
blenderproc_render/dataset_manifest.json Dataset sidecar index; HDF5/BOP exports are reserved for later
blenderproc_render/render_metadata.json Renderer settings used for the run
```

The camera path starts at the entry node, performs a deterministic DFS walk over the centerline graph, and backtracks over the same edges. During backtracking the camera does not turn around; it reverses through the same anatomical view direction, matching a scope being withdrawn. You can instead fly to one node and reverse back:

```bash
uv run kidney-meshgen render-blenderproc \
  --case-dir output/kidney_case \
  --target-node pelvis_center
```

The path planner first builds the native trajectory at the requested `--fps` and `--speed`, smooths the sampled centerline motion, then validates every smoothed camera center against the analytic primitive SDF. If a smoothed point would violate `--wall-clearance`, it is blended back toward the original centerline point. If `--max-frames` is provided, frames are selected from the completed native trajectory rather than changing the trajectory spacing. Motion blur and rolling shutter are disabled automatically for subsampled plans because those effects require consecutive native frames. Useful path knobs:

```text
--speed 18                 Nominal camera speed in mm/s
--fps 30                   Pose timestamp rate
--smooth-window 3.0        Motion smoothing window in mm
--max-smooth-offset 0.75   Maximum displacement away from centerline
--wall-clearance 0.45      Required lumen SDF clearance in mm
--max-frames 300           Select frames from the completed native path
```

Renderer quality presets:

```text
preview    960x540,   32 samples
balanced   1920x1080, 128 samples
high       2560x1440, 256 samples
cinematic  3840x2160, 512 samples
```

Realism-oriented knobs:

```text
--quality high             Higher resolution and Cycles samples
--liquid film              Wet glossy tissue material, default
--liquid volume            Adds irrigated volume scattering
--fluid-preset auto        Fluid/debris realism: low, medium, high, or auto
--depth-of-field           Optional short-range camera DOF
--depth                    Also write depth frames
--normals                  Also write surface-normal frames
--semantic                 Also write semantic category-ID frames
--denoiser OPTIX|INTEL     Cycles denoising backend
--split-ratios 0.8,0.1,0.1 Train/val/test frame split ratios
--split-seed 123           Optional split shuffle seed; defaults to render seed
--no-splits                Skip train/val/test split files
```

With `--liquid volume`, `--fluid-preset auto` resolves to `medium`. That mode adds cloudy irrigation, sparse suspended stone-dust particles, air bubbles, RGB-only lens film/droplets, and occasional partial near-lens occlusion. Use `low` or `high` to tune visibility degradation. Lens contamination is applied to RGB frames only; depth, normals, and semantic outputs stay tied to scene geometry.

Endoscope sensor realism is enabled by default with `--sensor-profile flexible_ureteroscope_hd`. Profiles provide a synthetic calibrated intrinsics matrix `K`, Brown-Conrady radial/decentering distortion, circular vignette/mask, exposure and white-balance gains, rolling-shutter/motion-blur settings, and a simple shot/read/PRNU sensor-noise model. Use `--sensor-profile none` for a plain pinhole camera model.

Custom calibration and sensor overrides:

```text
--camera-k camera.json                 JSON matrix or file with {"resolution": [w,h], "K": [[fx,0,cx], ...]}
--distortion-coeffs k1,k2,k3,p1,p2     Brown-Conrady/OpenCV-style coefficients
--no-lens-distortion                   Keep calibrated K but skip distortion warping
--no-sensor-effects                    Skip vignette, mask, exposure/WB, and sensor noise
--exposure-ev -0.2                     Override profile exposure compensation
--white-balance 1.05,1.0,0.94          Override RGB white-balance multipliers
--motion-blur-length 0.25              Shutter-open fraction between frames; 0 disables
--rolling-shutter-type TOP             NONE, TOP, BOTTOM, LEFT, or RIGHT
--rolling-shutter-length 0.08          Scanline exposure fraction for rolling shutter
```

Each run also randomizes a tissue material preset and a camera-light preset unless you pass `--no-randomize-realism`. The selected preset, jittered tissue/light values, resolved sensor model, semantic label IDs, and render seed are saved in `render_metadata.json` and `randomization.json`. Use `--render-seed N` to reproduce the same per-run realism choices. `frames.json` is the dataset-loader entry point for per-frame pose and modality paths; `camera_intrinsics.json` stores the explicit K matrix and Brown-Conrady coefficients used for the run.

### Realistic RGB-D evaluation set

For visual odometry, reconstruction, and camera-tracking evaluation, use no-stone cases with native frame spacing rather than a sparse `--max-frames` preview. The following pattern renders realistic endoscope RGB-D at 10 fps with calibrated intrinsics, Brown-Conrady lens distortion, vignette/circular mask, exposure/white-balance variation, shot/read/PRNU sensor noise, motion blur, and rolling shutter:

```bash
uv sync --extra render --dev

uv run kidney-meshgen generate \
  --out output/rgbd_eval_10fps_nostones/eval_001 \
  --seed 2001 \
  --anatomy-id eval_001 \
  --side right \
  --pelvicalyceal-class type_i \
  --lower-pole-access easy \
  --stones 0 \
  --no-preview

uv run kidney-meshgen render-blenderproc \
  --case-dir output/rgbd_eval_10fps_nostones/eval_001 \
  --out output/rgbd_eval_10fps_nostones/eval_001/blenderproc_rgbd_10fps_sensor128 \
  --quality preview \
  --width 640 \
  --height 360 \
  --samples 128 \
  --noise-threshold 0.015 \
  --fps 10 \
  --speed 18 \
  --depth \
  --denoiser OPTIX \
  --render-seed 7101 \
  --split-seed 7101
```

A useful 12-case evaluation cohort mixes right/left sides, Type I/Type II/random anatomy, and easy/intermediate/hard lower-pole access. One generated cohort used 4,532 native 10 fps RGB-D frames across 12 no-stone cases, with no low/black RGB frames, no empty depth frames, and per-case `frames.json`, `camera_poses.json`, `camera_intrinsics.json`, `randomization.json`, and train/val/test split files. A local dataset index can be written at `output/rgbd_eval_10fps_nostones/dataset_index.json`; `output/` is intentionally ignored by git.

Stone appearance is sampled during case generation and preserved in `scene_manifest.json`, so BlenderProc renders each stone with its own composition-aware material rather than a single generic shader. Supported material classes are:

```text
COM                calcium oxalate monohydrate / whewellite: dark, hard, smoother
COD                calcium oxalate dihydrate / weddellite: lighter, brittle, jagged
uric_acid          yellow-orange to red-brown, smoother/waxy
struvite_apatite   pale chalky infection/phosphate mix, granular and brittle
cystine            amber-yellow waxy stones with hexagonal crystal relief
```

`stone_fragmentation: intact` generates single faceted stones. `stone_fragmentation: gravel` generates laser-fragmented gravel fields with many angular fragments. `mixed` samples both using `stone_gravel_probability`.

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
scene_manifest.json                  Full manifest, including stone material/state metadata
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

`lumen_inner.*` and `lumen_outer.*` are the visual meshes. They include low-amplitude mucosal folds/noise for rendering. `collision/lumen_collision_proxy.obj` and `collision/lumen_sdf_grid.*` are built from the smooth analytic/collision surface, so visual roughness does not create contact artifacts.

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
- **Anatomy realism profile**: default `takazawa` profile emits Type I/II pelvis classes and named T/U/M/L/B calyces.
- **Anterior/posterior pairs**: upper, middle, and lower levels are generated as A/P pairs when the sampled calyx count allows it.
- **Papilla/fornix cups**: each minor calyx has a subtractive papilla primitive so cup surfaces are not just smooth bulbs.
- **Infundibular variation**: minor calyx necks can be locally narrowed, asymmetric, and mildly non-circular.
- **Render/collision split**: visual meshes get subtle mucosal folds and roughness while collision/SDF assets stay smooth.
- **Region-specific metadata**: every calyx target records sampled infundibular length, radius, branch angle, cup radii, and papilla geometry.
- **Longer entry tube**: default ureter length is increased so the scope starts in a realistic approach tube.
- **Cleaner lower pole**: the lower pole uses an explicit access trunk and minor-calyx fan-out.
- **Stronger branch clearance**: retry sampling and increased clearances reduce accidental chamber fusion.
- **Collision assets**: outputs both a collision proxy mesh and an approximate SDF grid.
- **Composition-aware stones**: COM, COD, uric acid, struvite/apatite, and cystine stones carry randomized color, crystal bump, roughness, and fracture-plane metadata.
- **Laser gravel fields**: stones can be generated intact or as clustered angular fragments for dusting/fragmentation simulation.
- **Navigation waypoints**: route waypoints include radius and estimated scope clearance.
- **Lean runtime descriptor**: `runtime_scene.json` replaces BlenderProc/Isaac descriptors.

## Research basis

The default profile is research-informed, not patient-specific. It uses Takazawa-style pelvicalyceal Type I/II branching and calyx naming from CT-urography/3D reconstruction work, lower-pole access parameters based on infundibular length/width/angle literature, and urothelial anatomy for a folded, wet mucosal visual layer.

- Takazawa-style pelvicalyceal classification: <https://www.jstage.jst.go.jp/article/jsejje/28/2/28_331/_article/-char/en>
- Modified Takazawa / 3D virtual reconstruction context: <https://pmc.ncbi.nlm.nih.gov/articles/PMC8350222/>
- Lower-pole infundibular angle, length, and width relevance: <https://pmc.ncbi.nlm.nih.gov/articles/PMC11252207/>
- Renal pelvis/calyx urothelial lining and lamina propria context: <https://basicmedicalkey.com/renal-pelvis-and-ureter-2/>
- BlenderProc calibrated `K` intrinsics and Brown-Conrady lens distortion workflow: <https://dlr-rm.github.io/BlenderProc/examples/advanced/lens_distortion/README.html>
- BlenderProc depth, normals, segmentation, motion blur, and rolling-shutter renderer hooks: <https://dlr-rm.github.io/BlenderProc/blenderproc.api.renderer.html>
- RIRS irrigation/stone-dust visibility background: <https://icurology.org/DOIx.php?id=10.4111%2Ficu.20200526>
- Endoscopic lens fogging and retained contaminant background: <https://journals.sagepub.com/doi/10.1089/end.2009.0594>
- Lens washing/contaminant accumulation background: <https://pubmed.ncbi.nlm.nih.gov/30020986/>
- Stone composition, color, hardness, and gross morphology summaries: <https://www.ncbi.nlm.nih.gov/sites/books/NBK442014/>
- Morphoconstitutional stone description context: <https://pmc.ncbi.nlm.nih.gov/articles/PMC9818792/>
- Kidney stone crystal/surface morphology context: <https://pmc.ncbi.nlm.nih.gov/articles/PMC5685519/>
- Holmium laser dusting and fragmentation background: <https://www.frontiersin.org/journals/surgery/articles/10.3389/fsurg.2017.00057/full>
- Blender volume scatter model: <https://docs.blender.org/manual/en/latest/render/shader_nodes/shader/volume_scatter.html>
- Brown-Conrady radial/tangential lens model background: <https://pmc.ncbi.nlm.nih.gov/articles/PMC4934233/>
- Sensor-noise model inspiration: photon shot noise, read noise, PRNU, exposure and white-balance terms following common EMVA 1288 / computational photography image-formation approximations.

## Useful config knobs

```yaml
anatomy_realism_profile: takazawa
pelvicalyceal_class: random      # random, type_i, type_ii
type_i_subtype: random           # random, ia, ib, ic
papilla_fornix_enabled: true
open_ureter_start: true
open_ureter_start_offset_mm: 1.2
lower_pole_access: intermediate   # easy, intermediate, hard, random
graph_clean_retry_attempts: 12
branch_sample_attempts: 140
cup_center_clearance_mm: 8.0
tube_clearance_mm: 1.65
scope_outer_diameter_mm: 3.0
stone_count: 3
stone_material_classes: [COM, COD, uric_acid, struvite_apatite, cystine]
stone_fragmentation: mixed          # intact, gravel, laser_fragmented_gravel, mixed
stone_gravel_probability: 0.35
stone_fragment_count: [18, 64]
stone_fragment_radius_fraction: [0.08, 0.24]
stone_gravel_spread_fraction: [0.42, 0.92]
stone_fracture_planes: [0, 8]
stone_surface_subdivisions: 3
visual_surface_noise_mm: 0.06
visual_fold_amplitude_mm: 0.18
visual_fold_band_mm: 7.0
infundibulum_cross_section_ovality: [0.04, 0.16]
infundibulum_narrowing_fraction: [0.04, 0.14]
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
    pelvicalyceal_class="type_i",
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

The generator does not model tissue deformation, real ureteroscope shaft mechanics, Navier-Stokes irrigation physics, blood physiology, or clinically validated patient-specific population distributions. The fluid/debris renderer is a visual approximation for simulation data, not a surgical fluid simulation. The realism profile is research-informed anatomy for simulation, not a diagnostic or surgical-planning model.
