# Project Media

This folder contains compact visual documentation for generated kidney-meshgen data. Large generated datasets stay under `output/`, which is intentionally ignored by git.

- `rgb_montage.png`: realistic flexible-ureteroscope RGB frames rendered from a no-stone RGB-D evaluation cohort.
- `depth_montage.png`: matching depth frames colorized for visual inspection. Dataset depth frames are stored as float32 `.npy` arrays.

Use the rendered `frames.json`, `camera_poses.json`, and `camera_intrinsics.json` files in `output/` for experiments; these images are documentation previews only.
