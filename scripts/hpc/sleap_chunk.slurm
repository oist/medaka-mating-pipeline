#!/bin/bash -l
#SBATCH --job-name=medaka-sleap
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

# ==============================================================================
# Single-chunk SLEAP-NN inference on saion.
#
# Submitted as an array task by submit_experiment.sh. Each task processes one
# frame range of one video (top_cam / side1_cam / side2_cam) and writes one
# .slp chunk to OUTPUT_DIR. Chunks are merged later by merge_chunks.sh.
#
# Partition (largegpu / gpu / test-gpu) is selected by the submitter via
# `sbatch --partition=...`; this file sets everything else.
#
# Environment variables (set by the submitter):
#   VIDEO_SRC      — source video on /bucket (read-only from compute node)
#   VIDEO_STAGED   — staged video on /work (fast scratch)
#   CENTROID_MODEL — SLEAP centroid model directory
#   INSTANCE_MODEL — SLEAP centered-instance model directory
#   OUTPUT_DIR     — /work dir where chunk .slp files are written
#   CHUNK_FILE     — text file: one "START END" pair per array task
#   VIEW           — one of top_cam / side1_cam / side2_cam (used in output name)
#   MAX_INSTANCES  — default 3 for medaka (1 F + 2 M)
#   BATCH_SIZE     — default 16
#   SLEAP_PYTHON     — optional absolute path to a Python environment. When set,
#                      modules are skipped and inference runs as:
#                      $SLEAP_PYTHON -m sleap_nn.cli track ...
#   SLEAP_MODULE     — module to load when SLEAP_PYTHON is unset; default
#                      sleap-nn/0.1.2 (also available: sleap-nn/0.1.0,
#                      sleap/1.5.2, sleap/1.6.2)
#   MODULEPATH_EXTRA — extra modulefile path; default /apps/unit/ReiterU/.modulefiles
# ==============================================================================

set -euo pipefail

: "${VIDEO_SRC:?VIDEO_SRC is required}"
: "${VIDEO_STAGED:?VIDEO_STAGED is required}"
: "${CENTROID_MODEL:?CENTROID_MODEL is required}"
: "${INSTANCE_MODEL:?INSTANCE_MODEL is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"
: "${CHUNK_FILE:?CHUNK_FILE is required}"
: "${VIEW:?VIEW is required}"

MAX_INSTANCES="${MAX_INSTANCES:-3}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SLEAP_PYTHON="${SLEAP_PYTHON:-}"
SLEAP_MODULE="${SLEAP_MODULE:-sleap-nn/0.1.2}"
MODULEPATH_EXTRA="${MODULEPATH_EXTRA:-/apps/unit/ReiterU/.modulefiles}"

echo "=== medaka SLEAP chunk ==="
echo "Job: $SLURM_JOB_ID  Task: $SLURM_ARRAY_TASK_ID  Node: $(hostname)"
echo "Partition: ${SLURM_JOB_PARTITION:-?}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || true
echo "View: ${VIEW}  Max instances: ${MAX_INSTANCES}  Batch size: ${BATCH_SIZE}"
echo "Started: $(date)"

if [ -n "$SLEAP_PYTHON" ]; then
    if [ ! -x "$SLEAP_PYTHON" ]; then
        echo "ERROR: SLEAP_PYTHON is not executable: $SLEAP_PYTHON" >&2
        exit 1
    fi
    echo "SLEAP runtime: explicit Python ($SLEAP_PYTHON); module loading skipped"
    "$SLEAP_PYTHON" - <<'PY'
import platform

import sleap_io
import sleap_nn
import torch

print(f"Python: {platform.python_version()}")
print(f"sleap-nn: {sleap_nn.__version__}")
print(f"sleap-io: {sleap_io.__version__}")
print(f"torch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}")
    print(f"GPU memory bytes: {props.total_memory}")
PY
else
    echo "SLEAP runtime: module $SLEAP_MODULE"
    module purge
    if [ -n "$MODULEPATH_EXTRA" ]; then
        module use "$MODULEPATH_EXTRA"
    fi
    module load "$SLEAP_MODULE"
    if command -v sleap-nn >/dev/null 2>&1; then
        sleap-nn system
    fi
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
        sleep 5
        WAITED=$((WAITED + 5))
        if [ "$WAITED" -gt 1800 ]; then
            echo "ERROR: timed out waiting for staging (>30 min)"
            exit 1
        fi
    done
fi

# ── Read this task's frame range ──
CHUNK_LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CHUNK_FILE")
START_FRAME=$(echo "$CHUNK_LINE" | awk '{print $1}')
END_FRAME=$(echo "$CHUNK_LINE" | awk '{print $2}')
FRAME_RANGE="${START_FRAME}-${END_FRAME}"

OUTPUT_SLP="${OUTPUT_DIR}/${VIEW}.frames_${FRAME_RANGE}.slp"
echo "Frames: ${FRAME_RANGE}"
echo "Output: ${OUTPUT_SLP}"

if [ -s "$OUTPUT_SLP" ] && [ "${FORCE:-0}" != "1" ]; then
    CHECK_PYTHON="$SLEAP_PYTHON"
    if [ -z "$CHECK_PYTHON" ]; then
        SLEAP_CLI="$(command -v sleap-nn || true)"
        if [ -n "$SLEAP_CLI" ]; then
            CHECK_PYTHON="$(head -n 1 "$SLEAP_CLI")"
            CHECK_PYTHON="${CHECK_PYTHON#\#!}"
        fi
    fi
    if [ -n "$CHECK_PYTHON" ] && "$CHECK_PYTHON" -c \
        'import sleap_io as sio, sys; labels = sio.load_slp(sys.argv[1]); assert labels.videos and labels.skeletons' \
        "$OUTPUT_SLP"; then
        echo "Skip: ${OUTPUT_SLP} already exists and is loadable (set FORCE=1 to re-run)."
        exit 0
    fi
    echo "ERROR: existing output is not a loadable SLP: $OUTPUT_SLP" >&2
    echo "Inspect it, then resubmit this task with FORCE=1 to replace it." >&2
    exit 1
fi

# ── Explicit Python for reproducible installs; module fallbacks for back-compat ──
if [ -n "$SLEAP_PYTHON" ]; then
    "$SLEAP_PYTHON" -m sleap_nn.cli track \
        --data_path "$VIDEO_STAGED" \
        --frames "$FRAME_RANGE" \
        --no_empty_frames \
        --model_paths "$CENTROID_MODEL" \
        --model_paths "$INSTANCE_MODEL" \
        --max_instances "$MAX_INSTANCES" \
        --batch_size "$BATCH_SIZE" \
        --output_path "$OUTPUT_SLP" \
        --device cuda
elif command -v sleap-nn >/dev/null 2>&1; then
    sleap-nn track \
        --data_path "$VIDEO_STAGED" \
        --frames "$FRAME_RANGE" \
        --no_empty_frames \
        --model_paths "$CENTROID_MODEL" \
        --model_paths "$INSTANCE_MODEL" \
        --max_instances "$MAX_INSTANCES" \
        --batch_size "$BATCH_SIZE" \
        --output_path "$OUTPUT_SLP" \
        --device cuda
elif command -v sleap-nn-track >/dev/null 2>&1; then
    sleap-nn-track \
        -i "$VIDEO_STAGED" \
        --frames "$FRAME_RANGE" \
        --no_empty_frames \
        -m "$CENTROID_MODEL" \
        -m "$INSTANCE_MODEL" \
        -n "$MAX_INSTANCES" \
        -b "$BATCH_SIZE" \
        -o "$OUTPUT_SLP"
else
    sleap-track \
        "$VIDEO_STAGED" \
        --frames "$FRAME_RANGE" \
        --no-empty-frames \
        -m "$CENTROID_MODEL" \
        -m "$INSTANCE_MODEL" \
        --max_instances "$MAX_INSTANCES" \
        --batch_size "$BATCH_SIZE" \
        -o "$OUTPUT_SLP"
fi

echo "Produced: $(du -h "$OUTPUT_SLP" | cut -f1)"
echo "Finished: $(date)"
