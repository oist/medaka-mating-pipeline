#!/bin/bash -l
#SBATCH --job-name=medaka-d2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

# ==============================================================================
# Single-chunk Detectron2 instance-segmentation inference on saion (Line B).
#
# Submitted as an array task by submit_detectron2_experiment.sh. Each task runs the
# self-contained chunk runner (d2_infer_chunk.py) over one frame range of one view's
# video and writes one masks .h5 chunk (<view>.masks_<S>-<E>.h5) to OUTPUT_DIR.
# Chunks stay separate, consumed per-window at analysis time by 33_build_segid_dataset.
#
# The runner is deliberately standalone (only the detectron.2 venv + a co-located
# mask_io.py) so it needs neither the medaka repo nor py3.11 (tomllib). The env comes
# from the detectron.2 module via D2_SETUP / D2_PYTHON.
#
# Environment variables (set by the submitter via --export):
#   VIDEO_SRC, VIDEO_STAGED, D2_MODEL, D2_CONFIG, OUTPUT_DIR, CHUNK_FILE, VIEW
#   D2_RUNNER      — path to d2_infer_chunk.py (mask_io.py must sit beside it)
#   D2_PYTHON      — python with detectron2 (default: python, resolved after D2_SETUP)
#   D2_SETUP       — env snippet eval'd first (e.g. module load detectron.2/0.6)
#   SCORE_THRESH   — ROI score threshold (default 0.5)
#   FRAME_STRIDE   — every Nth frame (default 1)
# ==============================================================================

set -euo pipefail

: "${VIDEO_SRC:?VIDEO_SRC is required}"
: "${VIDEO_STAGED:?VIDEO_STAGED is required}"
: "${D2_MODEL:?D2_MODEL is required}"
: "${D2_CONFIG:?D2_CONFIG is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"
: "${CHUNK_FILE:?CHUNK_FILE is required}"
: "${VIEW:?VIEW is required}"
: "${D2_RUNNER:?D2_RUNNER is required}"

SCORE_THRESH="${SCORE_THRESH:-0.5}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
D2_PYTHON="${D2_PYTHON:-python}"
D2_SETUP="${D2_SETUP:-}"

echo "=== medaka Detectron2 chunk ==="
echo "Job: $SLURM_JOB_ID  Task: $SLURM_ARRAY_TASK_ID  Node: $(hostname)"
echo "Partition: ${SLURM_JOB_PARTITION:-?}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1 || true
echo "View: ${VIEW}  Score thr: ${SCORE_THRESH}  Frame stride: ${FRAME_STRIDE}"
echo "Started: $(date)"

# ── Detectron2 environment (no module auto-loaded; D2_SETUP brings it in) ──
module purge 2>/dev/null || true
if [ -n "$D2_SETUP" ]; then
    echo "D2_SETUP: $D2_SETUP"
    eval "$D2_SETUP"
fi

# ── Stage video (task 0 copies, others wait on a sentinel file) ──
STAGING_DONE="${OUTPUT_DIR}/.staging_${VIEW}.done"
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    if [ ! -f "$STAGING_DONE" ]; then
        echo "Task 0: staging ${VIDEO_SRC} -> ${VIDEO_STAGED}"
        mkdir -p "$(dirname "$VIDEO_STAGED")"
        cp "$VIDEO_SRC" "$VIDEO_STAGED"
        touch "$STAGING_DONE"
        echo "Staged $(du -h "$VIDEO_STAGED" | cut -f1)"
    else
        echo "Task 0: staged video already present."
    fi
else
    echo "Task $SLURM_ARRAY_TASK_ID: waiting for staging..."
    WAITED=0
    while [ ! -f "$STAGING_DONE" ]; do
        sleep 5; WAITED=$((WAITED + 5))
        [ "$WAITED" -gt 1800 ] && { echo "ERROR: timed out waiting for staging (>30 min)"; exit 1; }
    done
fi

# ── Read this task's frame range ──
CHUNK_LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CHUNK_FILE")
START_FRAME=$(echo "$CHUNK_LINE" | awk '{print $1}')
END_FRAME=$(echo "$CHUNK_LINE" | awk '{print $2}')
OUTPUT_H5="${OUTPUT_DIR}/${VIEW}.masks_${START_FRAME}-${END_FRAME}.h5"
echo "Frames: ${START_FRAME}-${END_FRAME}"
echo "Output: ${OUTPUT_H5}"

if [ -s "$OUTPUT_H5" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "Skip: ${OUTPUT_H5} already exists (set FORCE=1 to re-run)."
    exit 0
fi

"$D2_PYTHON" "$D2_RUNNER" \
    --video "$VIDEO_STAGED" \
    --model "$D2_MODEL" \
    --config "$D2_CONFIG" \
    --start "$START_FRAME" \
    --end "$END_FRAME" \
    --stride "$FRAME_STRIDE" \
    --score-thresh "$SCORE_THRESH" \
    --output "$OUTPUT_H5" \
    --device cuda

echo "Produced: $(du -h "$OUTPUT_H5" | cut -f1)"
echo "Finished: $(date)"
