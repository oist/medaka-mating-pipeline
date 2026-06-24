#!/bin/bash
# ==============================================================================
# Copy per-view Detectron2 mask chunk .h5 files from /work to /bucket.
#
# No merging happens here — chunks stay as separate per-frame-range files (mirror
# of merge_chunks.sh for SLEAP). They are consumed per-window at analysis time by
# sam2_id_experiment/scripts/33_build_segid_dataset.py, which reads one masks .h5
# via iter_mask_instances_h5 and filters to the requested --window frame range.
#
# Expected /work layout (written by submit_detectron2_experiment.sh):
#   <WORK_EXP>/
#     top_cam/<view>.masks_<S>-<E>.h5 ...
#     side1_cam/...
#     side2_cam/...
#
# Produces:
#   <BUCKET_EXP_OUT>/
#     top_cam.masks_*.h5
#     side1_cam.masks_*.h5
#     side2_cam.masks_*.h5
#     logs/                  (slurm per-task .out/.err)
#
# Usage:
#   bash merge_detectron2_chunks.sh <WORK_EXP> <BUCKET_EXP_OUT>
# ==============================================================================

set -euo pipefail

WORK_EXP="${1:?WORK_EXP (experiment scratch dir) required}"
BUCKET_EXP_OUT="${2:?BUCKET_EXP_OUT (bucket destination) required}"

echo "=== merge_detectron2_chunks.sh (copy-only) ==="
echo "Work source : $WORK_EXP"
echo "Bucket dest : $BUCKET_EXP_OUT"

[ -d "$WORK_EXP" ] || { echo "ERROR: work dir missing: $WORK_EXP" >&2; exit 1; }
mkdir -p "$BUCKET_EXP_OUT" "$BUCKET_EXP_OUT/logs"

for view in top_cam side1_cam side2_cam; do
    view_dir="${WORK_EXP}/${view}"
    [ -d "$view_dir" ] || { echo "[${view}] no dir, skip"; continue; }

    mapfile -t chunks < <(ls "${view_dir}"/${view}.masks_*.h5 2>/dev/null | sort)
    if [ "${#chunks[@]}" -eq 0 ]; then
        echo "[${view}] no chunks found, skip"
        continue
    fi
    echo "[${view}] ${#chunks[@]} chunks -> ${BUCKET_EXP_OUT}/"
    rsync -av --partial "${chunks[@]}" "${BUCKET_EXP_OUT}/"
done

find "$WORK_EXP" -maxdepth 2 \( -name 'slurm_*.out' -o -name 'slurm_*.err' \) -print0 2>/dev/null \
    | xargs -0 -r -I{} rsync -a {} "${BUCKET_EXP_OUT}/logs/" || true

echo ""
echo "Done. Bucket output: $BUCKET_EXP_OUT"
ls -lh "$BUCKET_EXP_OUT"
