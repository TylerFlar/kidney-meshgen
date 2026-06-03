#!/usr/bin/env bash
# NEARLIGHT-BASELINE-OBSERVABILITY-001 -- render the 110 eval windows in linear
# radiance for the baseline ladder b in {0, 1.16, 3, 5} mm. One render per case
# per baseline; only the window frames are rendered (subset, renumbered) for speed.
# Poses/intrinsics are byte-identical to the PNG eval (same camera_poses.json +
# sensor profile). Idempotent: skips a (case,b) whose frames.json already exists.
set -u
cd "$(dirname "$0")/.." || exit 1

BP=.venv/Scripts/blenderproc.exe
ENTRY=kidney_meshgen/blenderproc_entry.py
SRC=output/slam_eval_10fps_nostones
SUBSETS="G:/01_Active/Code/arclab/endonav-slam/runs/nearlight-baseline-001/subsets"
CASES="001 002 003 004 005 006 007 008 009 010 011 012"
# baseline value -> output tag
declare -A TAG=( ["0"]="0" ["1.16"]="1p16" ["3"]="3" ["5"]="5" )

for B in 0 1.16 3 5; do
  T=${TAG[$B]}
  for C in $CASES; do
    CASE=slam_eval_${C}
    OUT=output/slam_eval_linear_exr_b${T}/${CASE}
    if [ -f "${OUT}/frames.json" ]; then
      echo "[skip] ${CASE} b=${B} (exists)"
      continue
    fi
    echo "=== render ${CASE} b=${B} -> ${OUT} ==="
    "$BP" run "$ENTRY" -- \
      --case-dir "${SRC}/${CASE}" \
      --pose-file "${SRC}/${CASE}/blenderproc_rgbd_10fps_sensor128/camera_poses.json" \
      --out "${OUT}" \
      --width 640 --height 360 --samples 128 --noise-threshold 0.015 --denoiser OPTIX \
      --sensor-profile flexible_ureteroscope_hd \
      --material-preset baseline --light-preset baseline --no-randomize-realism \
      --linear-exr --light-baseline-mm "${B}" \
      --frame-indices-file "${SUBSETS}/${CASE}.json" \
      --render-seed 7101 --no-splits 2>&1 | grep -E "Finished rendering|render |Error|error|Traceback" | tail -3
  done
done
echo "=== ALL BASELINE RENDERS DONE ==="
