#!/usr/bin/env bash
# INSTRUMENT-LENGTH-ANCHOR-001 -- render the 110 eval windows of the 12 slam_eval
# cases with a known-width instrument shaft (category_id=2) rigidly fixed in the
# camera frame, crossing the FOV. Linear radiance + GT depth + GT semantic, b=3 mm.
# Mirrors render_nearlight_baseline_ladder.sh; only the eval-window frames are
# rendered (subset, renumbered). Idempotent: skips a case whose frames.json exists.
set -u
cd "$(dirname "$0")/.." || exit 1

BP=.venv/Scripts/blenderproc.exe
ENTRY=kidney_meshgen/blenderproc_entry.py
SRC=output/slam_eval_10fps_nostones
SUBSETS="G:/01_Active/Code/arclab/endonav-slam/runs/nearlight-baseline-001/subsets"
OUT_ROOT=output/slam_eval_instrument_b3
INSTR_FR_MM="${1:-3.0}"          # diameter mm: 3.0 (~9 Fr wide channel) | 0.365 (thin fiber)
CASES="001 002 003 004 005 006 007 008 009 010 011 012"

for C in $CASES; do
  CASE=slam_eval_${C}
  OUT=${OUT_ROOT}/${CASE}
  if [ -f "${OUT}/frames.json" ]; then
    echo "[skip] ${CASE} (exists)"
    continue
  fi
  echo "=== render ${CASE} instrument=${INSTR_FR_MM}mm -> ${OUT} ==="
  "$BP" run "$ENTRY" -- \
    --case-dir "${SRC}/${CASE}" \
    --pose-file "${SRC}/${CASE}/blenderproc_rgbd_10fps_sensor128/camera_poses.json" \
    --out "${OUT}" \
    --width 640 --height 360 --samples 64 --noise-threshold 0.015 --denoiser OPTIX \
    --sensor-profile flexible_ureteroscope_hd \
    --material-preset baseline --light-preset baseline --no-randomize-realism \
    --linear-exr --light-baseline-mm 3 \
    --enable-depth --enable-semantic \
    --instrument-fr-mm "${INSTR_FR_MM}" \
    --frame-indices-file "${SUBSETS}/${CASE}.json" \
    --render-seed 7101 --no-splits 2>&1 | grep -E "Finished rendering|render |Error|error|Traceback" | tail -3
done
echo "=== INSTRUMENT COHORT RENDER DONE (${INSTR_FR_MM}mm) ==="
