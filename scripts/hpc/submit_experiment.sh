#!/bin/bash
# ==============================================================================
# Submit SLEAP inference for ONE medaka experiment on saion.
#
# For the given experiment folder, submits three Slurm array jobs — one per
# view (top_cam / side1_cam / side2_cam) — so the three videos run in parallel.
# Each view can itself be split into NUM_CHUNKS chunks so the array parallelises
# frames across GPUs (useful when a single video would exceed the 12 h wall).
#
# After the arrays finish, a single dependent copy job rsyncs outputs from
# /work back to the bucket destination and (optionally) merges chunks.
#
# Usage:
#   bash scripts/hpc/submit_experiment.sh <EXPERIMENT> [options]
#
# Positional:
#   EXPERIMENT   Experiment folder name under MEDAKA_BUCKET_ROOT
#                (e.g. 20251220_D_1_1_FCMC).
#
# Environment variables (with defaults):
#   MEDAKA_BUCKET_ROOT   /bucket/StephensU/medaka/medaka_202512
#   MEDAKA_WORK_ROOT     /work/StephensU/makoto/medaka_sleap
#   MEDAKA_BUCKET_OUT    /bucket/StephensU/medaka/sleap_output
#   TOP_CENTROID         /bucket/StephensU/medaka/medaka_202512/sleap_models/sleap_top_mouth/models/260313_152145.centroid.n=330
#   TOP_INSTANCE         /bucket/StephensU/medaka/medaka_202512/sleap_models/sleap_top_mouth/models/260313_175737.centered_instance.n=330
#   SIDE_CENTROID        /bucket/StephensU/medaka/medaka_202512/sleap_models/sleap_side_mouth/models/260111_170857.centroid.n=593
#   SIDE_INSTANCE        /bucket/StephensU/medaka/medaka_202512/sleap_models/sleap_side_mouth/models/260112_115936.centered_instance.n=593
#   NUM_CHUNKS           4     (array tasks per view; 1 = no chunking)
#   MAX_INSTANCES        3     (1 female + 2 males)
#   BATCH_SIZE           16
#   PARTITION            largegpu   (or gpu / test-gpu)
#   TIME_LIMIT           12:00:00
#   SLEAP_MODULE         sleap-nn/0.1.2   (also: sleap-nn/0.1.0, sleap/1.5.2, sleap/1.6.2)
#   MODULEPATH_EXTRA     /apps/unit/ReiterU/.modulefiles
#   SKIP_VIEWS           (optional) space-separated views to skip, e.g. "side2_cam"
#   DRY_RUN              1 = print commands only, 2 = also skip ffprobe
#   NO_MERGE             1 = skip final merge after arrays complete
# ==============================================================================

set -euo pipefail

EXPERIMENT="${1:-}"
if [ -z "$EXPERIMENT" ]; then
    echo "usage: $0 <EXPERIMENT> [e.g. 20251220_D_1_1_FCMC]" >&2
    exit 2
fi

# ── Configuration (env-overridable) ──
MEDAKA_BUCKET_ROOT="${MEDAKA_BUCKET_ROOT:-/bucket/StephensU/medaka/medaka_202512}"
MEDAKA_WORK_ROOT="${MEDAKA_WORK_ROOT:-/work/StephensU/makoto/medaka_sleap}"
MEDAKA_BUCKET_OUT="${MEDAKA_BUCKET_OUT:-/bucket/StephensU/medaka/sleap_output}"

TOP_CENTROID="${TOP_CENTROID:-/bucket/StephensU/medaka/medaka_202512/sleap_models/sleap_top_mouth/models/260313_152145.centroid.n=330}"
TOP_INSTANCE="${TOP_INSTANCE:-/bucket/StephensU/medaka/medaka_202512/sleap_models/sleap_top_mouth/models/260313_175737.centered_instance.n=330}"
SIDE_CENTROID="${SIDE_CENTROID:-/bucket/StephensU/medaka/medaka_202512/sleap_models/sleap_side_mouth/models/260111_170857.centroid.n=593}"
SIDE_INSTANCE="${SIDE_INSTANCE:-/bucket/StephensU/medaka/medaka_202512/sleap_models/sleap_side_mouth/models/260112_115936.centered_instance.n=593}"

NUM_CHUNKS="${NUM_CHUNKS:-4}"
MAX_INSTANCES="${MAX_INSTANCES:-3}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PARTITION="${PARTITION:-largegpu}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
SLEAP_MODULE="${SLEAP_MODULE:-sleap-nn/0.1.2}"
MODULEPATH_EXTRA="${MODULEPATH_EXTRA:-/apps/unit/ReiterU/.modulefiles}"
SKIP_VIEWS="${SKIP_VIEWS:-}"
DRY_RUN="${DRY_RUN:-0}"
NO_MERGE="${NO_MERGE:-0}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKER="${SCRIPT_DIR}/sleap_chunk.slurm"
[ -f "$WORKER" ] || { echo "ERROR: missing $WORKER" >&2; exit 1; }

EXP_DIR="${MEDAKA_BUCKET_ROOT}/${EXPERIMENT}"
[ -d "$EXP_DIR" ] || { echo "ERROR: experiment dir not found: $EXP_DIR" >&2; exit 1; }

WORK_EXP="${MEDAKA_WORK_ROOT}/${EXPERIMENT}"
BUCKET_EXP_OUT="${MEDAKA_BUCKET_OUT}/${EXPERIMENT}"
mkdir -p "$WORK_EXP"

echo "============================================"
echo "  Medaka SLEAP submission: ${EXPERIMENT}"
echo "============================================"
echo "Bucket in     : $EXP_DIR"
echo "Work scratch  : $WORK_EXP"
echo "Bucket out    : $BUCKET_EXP_OUT"
echo "Partition     : $PARTITION    Time limit: $TIME_LIMIT"
echo "Chunks/view   : $NUM_CHUNKS   Max instances: $MAX_INSTANCES"
echo "Top models    : $TOP_CENTROID"
echo "              : $TOP_INSTANCE"
echo "Side models   : $SIDE_CENTROID"
echo "              : $SIDE_INSTANCE"
echo ""

probe_frames() {
    # Read the authoritative frame count from the camera folder's metadata.csv
    # (written by campy when recording closed). Falls back to frametimes.npy
    # length if the csv is missing.
    local video="$1"
    local cam_dir meta n
    cam_dir="$(dirname "$video")"
    meta="${cam_dir}/metadata.csv"
    if [ -f "$meta" ]; then
        n=$(awk -F'"' '$2 == "totalFrames" { print $4; exit }' "$meta")
        if [[ "$n" =~ ^[0-9]+$ ]] && [ "$n" -gt 0 ]; then
            echo "$n"; return
        fi
    fi
    local ft="${cam_dir}/frametimes.npy"
    if [ -f "$ft" ] && command -v python3 >/dev/null 2>&1; then
        n=$(python3 -c "import numpy as np,sys; a=np.load(sys.argv[1]); print(a.shape[-1])" "$ft" 2>/dev/null || true)
        if [[ "$n" =~ ^[0-9]+$ ]] && [ "$n" -gt 0 ]; then
            echo "$n"; return
        fi
    fi
    echo 0
}

write_chunks() {
    local total="$1" n_chunks="$2" chunk_file="$3"
    local per=$(( (total + n_chunks - 1) / n_chunks ))
    > "$chunk_file"
    for i in $(seq 0 $((n_chunks - 1))); do
        local start=$((i * per))
        local end=$(( (i + 1) * per - 1 ))
        [ "$end" -ge "$total" ] && end=$((total - 1))
        [ "$start" -gt "$end" ] && continue
        echo "$start $end" >> "$chunk_file"
    done
}

should_skip() {
    local view="$1"
    for s in $SKIP_VIEWS; do
        [ "$s" = "$view" ] && return 0
    done
    return 1
}

submit_view() {
    # $1=view $2=video_src $3=centroid $4=instance
    # Prints the job id on stdout; status messages go to stderr so the caller
    # can capture the id cleanly with $(...).
    local view="$1" video_src="$2" centroid="$3" instance="$4"

    if should_skip "$view"; then
        echo "[${view}] skipped (SKIP_VIEWS)" >&2
        return 0
    fi
    if [ ! -f "$video_src" ]; then
        echo "[${view}] MISSING video: $video_src" >&2
        return 1
    fi

    local frames=0
    if [ "$DRY_RUN" != "2" ]; then
        frames=$(probe_frames "$video_src")
    fi
    if [ "$frames" -eq 0 ]; then
        if [ "$DRY_RUN" != "0" ]; then
            frames=1000000  # placeholder for dry-run
            echo "[${view}] DRY RUN: assuming $frames frames" >&2
        else
            echo "[${view}] ERROR: could not probe frames for $video_src" >&2
            return 1
        fi
    fi

    local view_out="${WORK_EXP}/${view}"
    mkdir -p "$view_out"
    local chunk_file="${view_out}/chunks.txt"
    local n_chunks="$NUM_CHUNKS"
    if [ "$frames" -lt "$n_chunks" ]; then
        n_chunks=1
    fi
    write_chunks "$frames" "$n_chunks" "$chunk_file"
    local actual_chunks
    actual_chunks=$(wc -l < "$chunk_file")

    local video_staged="${view_out}/$(basename "$video_src")"

    echo "[${view}] frames=$frames  chunks=$actual_chunks  out=$view_out" >&2

    local sbatch_cmd=(
        sbatch --parsable
        --partition="$PARTITION"
        --time="$TIME_LIMIT"
        --array="0-$((actual_chunks - 1))"
        --job-name="medaka_${EXPERIMENT}_${view}"
        --output="${view_out}/slurm_%A_%a.out"
        --error="${view_out}/slurm_%A_%a.err"
        --export=ALL,VIDEO_SRC="$video_src",VIDEO_STAGED="$video_staged",CENTROID_MODEL="$centroid",INSTANCE_MODEL="$instance",OUTPUT_DIR="$view_out",CHUNK_FILE="$chunk_file",VIEW="$view",MAX_INSTANCES="$MAX_INSTANCES",BATCH_SIZE="$BATCH_SIZE",SLEAP_MODULE="$SLEAP_MODULE",MODULEPATH_EXTRA="${MODULEPATH_EXTRA:-}"
        "$WORKER"
    )

    if [ "$DRY_RUN" != "0" ]; then
        printf '   DRY: %s\n' "${sbatch_cmd[*]}" >&2
        return 0
    fi

    local jid
    jid=$("${sbatch_cmd[@]}")
    echo "[${view}] submitted array $jid (tasks=$actual_chunks)" >&2
    echo "$jid"
}

# ── Submit all three views ──
TOP_JOB=$(submit_view "top_cam"   "${EXP_DIR}/top_cam/cam0_top.mp4"    "$TOP_CENTROID"  "$TOP_INSTANCE"  || true)
S1_JOB=$(submit_view  "side1_cam" "${EXP_DIR}/side1_cam/cam1_side.mp4" "$SIDE_CENTROID" "$SIDE_INSTANCE" || true)
S2_JOB=$(submit_view  "side2_cam" "${EXP_DIR}/side2_cam/cam2_side.mp4" "$SIDE_CENTROID" "$SIDE_INSTANCE" || true)

DEP_LIST=""
for jid in "$TOP_JOB" "$S1_JOB" "$S2_JOB"; do
    [[ "$jid" =~ ^[0-9]+$ ]] || continue
    if [ -z "$DEP_LIST" ]; then DEP_LIST="$jid"; else DEP_LIST="${DEP_LIST}:${jid}"; fi
done

if [ "$DRY_RUN" != "0" ]; then
    echo ""
    echo "DRY RUN complete. Nothing was submitted."
    exit 0
fi

if [ -z "$DEP_LIST" ]; then
    echo ""
    echo "No jobs were submitted — skipping merge/copy step." >&2
    exit 1
fi

# ── Background merge + copy-back (runs on the login node under nohup) ──
# /bucket is read-only from compute nodes, so merging chunks and rsyncing back
# must happen from the login node. The background process polls sacct until
# all arrays are terminal, then runs merge_chunks.sh.
WATCHER="${WORK_EXP}/merge_and_copy.sh"
DEPS_COLON_SEPARATED="${DEP_LIST}"
cat > "$WATCHER" << EOF
#!/bin/bash
set -euo pipefail
JIDS="${DEPS_COLON_SEPARATED//:/ }"
echo "\$(date): waiting on jobs: \$JIDS"

${MODULEPATH_EXTRA:+module use ${MODULEPATH_EXTRA} >/dev/null 2>&1 || true}
module load ${SLEAP_MODULE} >/dev/null 2>&1 || true

MAX_WAIT=\$(( 60 * 60 * 24 ))   # 24 h
ELAPSED=0
while [ "\$ELAPSED" -lt "\$MAX_WAIT" ]; do
    REMAINING=0
    for jid in \$JIDS; do
        if squeue -h -j "\$jid" 2>/dev/null | grep -q .; then
            REMAINING=\$((REMAINING + 1))
        fi
    done
    [ "\$REMAINING" -eq 0 ] && break
    sleep 60
    ELAPSED=\$((ELAPSED + 60))
done

echo "\$(date): arrays done, merging + copying back"
bash "${SCRIPT_DIR}/merge_chunks.sh" "${WORK_EXP}" "${BUCKET_EXP_OUT}" "${NO_MERGE}"
echo "\$(date): finished"
EOF
chmod +x "$WATCHER"

LOG="${WORK_EXP}/merge_and_copy.log"
nohup bash "$WATCHER" >> "$LOG" 2>&1 &
WATCHER_PID=$!

echo ""
echo "Background watcher : PID $WATCHER_PID  (log: $LOG)"
echo "Monitor arrays     : squeue -u \$USER"
echo "Work dir           : $WORK_EXP"
echo "Final destination  : $BUCKET_EXP_OUT"
echo "Safe to disconnect — watcher runs under nohup."
