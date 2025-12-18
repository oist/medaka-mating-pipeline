#!/usr/bin/env bash
set -euo pipefail

# Template for running SLEAP 1.5.x inference on a cluster.
#
# This assumes SLEAP is installed as a uv tool:
#   uv tool install "sleap[nn]" ...
#
# Usage (edit paths as needed):
#   sbatch scripts/sleap/run_inference_slurm.sh /path/to/video.mp4 /path/to/models/model_dir
#
# Notes:
# - `sleap-track` output defaults to: <input>.predictions.slp
# - For top-down pipelines, pass two model dirs (centroid then instance).

INPUT_PATH="${1:?First arg must be input video (.mp4) or labels (.slp)}"
MODEL_1="${2:?Second arg must be model directory (or centroid model dir)}"
MODEL_2="${3:-}"

BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_INSTANCES="${MAX_INSTANCES:-3}"
PEAK_THRESHOLD="${PEAK_THRESHOLD:-0.2}"

TRACKER="${TRACKER:-simple}"
OUTPUT_PATH="${OUTPUT_PATH:-}"  # optional; if empty uses SLEAP default

CMD=(uv tool run sleap-track -m "${MODEL_1}")
if [[ -n "${MODEL_2}" ]]; then
	CMD+=( -m "${MODEL_2}" )
fi

CMD+=( --batch_size "${BATCH_SIZE}" --max_instances "${MAX_INSTANCES}" --peak_threshold "${PEAK_THRESHOLD}" )

if [[ -n "${TRACKER}" && "${TRACKER}" != "none" ]]; then
	CMD+=( --tracking.tracker "${TRACKER}" )
fi

if [[ -n "${OUTPUT_PATH}" ]]; then
	CMD+=( -o "${OUTPUT_PATH}" )
fi

CMD+=( "${INPUT_PATH}" )

echo "Running: ${CMD[*]}"
"${CMD[@]}"
