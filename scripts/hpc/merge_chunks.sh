#!/bin/bash
# ==============================================================================
# Copy per-view SLEAP chunk .slp files from /work to /bucket.
#
# No merging happens here — chunks stay as separate files. They are combined
# at analysis import time (see scripts/sleap/export_tracks.py and the downstream
# tracking code), which writes a single HDF5/merged .slp for reuse.
#
# Expected /work layout (written by submit_experiment.sh):
#   <WORK_EXP>/
#     top_cam/<view>.frames_<S>-<E>.slp ...
#     side1_cam/...
#     side2_cam/...
#
# Produces:
#   <BUCKET_EXP_OUT>/
#     top_cam.frames_*.slp
#     side1_cam.frames_*.slp
#     side2_cam.frames_*.slp
#     logs/                  (slurm per-task .out/.err)
#
# Usage:
#   bash merge_chunks.sh <WORK_EXP> <BUCKET_EXP_OUT> [NO_MERGE_UNUSED]
# ==============================================================================

set -euo pipefail

WORK_EXP="${1:?WORK_EXP (experiment scratch dir) required}"
BUCKET_EXP_OUT="${2:?BUCKET_EXP_OUT (bucket destination) required}"
# third arg kept for back-compat with existing watcher invocations; unused.

echo "=== merge_chunks.sh (copy-only) ==="
echo "Work source : $WORK_EXP"
echo "Bucket dest : $BUCKET_EXP_OUT"

[ -d "$WORK_EXP" ] || { echo "ERROR: work dir missing: $WORK_EXP" >&2; exit 1; }
mkdir -p "$BUCKET_EXP_OUT" "$BUCKET_EXP_OUT/logs"

for view in top_cam side1_cam side2_cam; do
    view_dir="${WORK_EXP}/${view}"
    [ -d "$view_dir" ] || { echo "[${view}] no dir, skip"; continue; }

    mapfile -t chunks < <(ls "${view_dir}"/${view}.frames_*.slp 2>/dev/null | sort)
    if [ "${#chunks[@]}" -eq 0 ]; then
        echo "[${view}] no chunks found, skip"
        continue
    fi
    echo "[${view}] ${#chunks[@]} chunks -> ${BUCKET_EXP_OUT}/"
    rsync -av --partial "${chunks[@]}" "${BUCKET_EXP_OUT}/"
done

# Slurm per-task logs are useful for debugging — tuck them under logs/.
find "$WORK_EXP" -maxdepth 2 \( -name 'slurm_*.out' -o -name 'slurm_*.err' \) -print0 2>/dev/null \
    | xargs -0 -r -I{} rsync -a {} "${BUCKET_EXP_OUT}/logs/" || true

echo ""
echo "Done. Bucket output: $BUCKET_EXP_OUT"
ls -lh "$BUCKET_EXP_OUT"
