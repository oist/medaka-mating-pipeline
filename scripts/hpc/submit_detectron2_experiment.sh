#!/bin/bash
# ==============================================================================
# Submit Detectron2 instance-segmentation inference (Line B) for ONE medaka
# experiment on saion. Direct mirror of submit_experiment.sh (SLEAP), producing
# per-view masks .h5 chunks instead of .slp chunks.
#
# For the given experiment folder, submits up to three Slurm array jobs — one per
# view (top_cam / side1_cam / side2_cam) — so the videos run in parallel. Each
# view splits into NUM_CHUNKS array tasks across frames. A background watcher
# rsyncs the per-view mask chunks from /work to /bucket once the arrays finish.
#
# Usage:
#   bash scripts/hpc/submit_detectron2_experiment.sh <EXPERIMENT> [options via env]
#
# Detectron2 has NO module on saion (unlike SLEAP). Provide the env via:
#   D2_PYTHON  python with detectron2+torch(+CUDA) (e.g. .conda/envs/medaka_d2/bin/python)
#   D2_SETUP   optional shell snippet eval'd before running (conda activate / source)
# Build that env ONCE on a GPU node (see scripts/hpc/README.md, Detectron2 section).
#
# Environment variables (with defaults):
#   MEDAKA_BUCKET_ROOT   /bucket/StephensU/medaka/medaka_202512
#   MEDAKA_WORK_ROOT     /work/StephensU/makoto/medaka_detectron2
#   MEDAKA_BUCKET_OUT    /bucket/StephensU/medaka/detectron2_output
#   TOP_D2_MODEL         (none — set to enable top_cam; else top is skipped)
#   TOP_D2_CONFIG        (none)
#   SIDE_D2_MODEL        /bucket/StephensU/medaka/medaka_202512/detectron2_models/side_detectron2_model_firstprompt/model_final.pth
#   SIDE_D2_CONFIG       /bucket/StephensU/medaka/medaka_202512/detectron2_models/side_detectron2_model_firstprompt/detectron2_config.yaml
#   NUM_CHUNKS           4
#   SCORE_THRESH         0.5
#   FRAME_STRIDE         1
#   PARTITION            largegpu   (or gpu / test-gpu)
#   GRES                 gpu:1      (use gpu:v100:1 on the gpu partition: V100 has a new
#                                    driver for torch cu121; the partition's P100s do not)
#   TIME_LIMIT           12:00:00
#   D2_RUNNER            <script_dir>/d2_infer_chunk.py  (mask_io.py must sit beside it)
#   D2_PYTHON            python
#   D2_SETUP             (empty)
#   SKIP_VIEWS           (optional) space-separated views to skip, e.g. "top_cam"
#   DRY_RUN              1 = print commands only, 2 = also skip frame probe
#   NO_MERGE             1 = skip final copy-back after arrays complete
# ==============================================================================

set -euo pipefail

EXPERIMENT="${1:-}"
if [ -z "$EXPERIMENT" ]; then
    echo "usage: $0 <EXPERIMENT> [e.g. 20251225_K_1_2_FCMC]" >&2
    exit 2
fi

MEDAKA_BUCKET_ROOT="${MEDAKA_BUCKET_ROOT:-/bucket/StephensU/medaka/medaka_202512}"
MEDAKA_WORK_ROOT="${MEDAKA_WORK_ROOT:-/work/StephensU/makoto/medaka_detectron2}"
MEDAKA_BUCKET_OUT="${MEDAKA_BUCKET_OUT:-/bucket/StephensU/medaka/detectron2_output}"

TOP_D2_MODEL="${TOP_D2_MODEL:-}"
TOP_D2_CONFIG="${TOP_D2_CONFIG:-}"
SIDE_D2_MODEL="${SIDE_D2_MODEL:-/bucket/StephensU/medaka/medaka_202512/detectron2_models/side_detectron2_model_firstprompt/model_final.pth}"
SIDE_D2_CONFIG="${SIDE_D2_CONFIG:-/bucket/StephensU/medaka/medaka_202512/detectron2_models/side_detectron2_model_firstprompt/detectron2_config.yaml}"

NUM_CHUNKS="${NUM_CHUNKS:-4}"
SCORE_THRESH="${SCORE_THRESH:-0.5}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
PARTITION="${PARTITION:-largegpu}"
GRES="${GRES:-gpu:1}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
D2_PYTHON="${D2_PYTHON:-python}"
D2_SETUP="${D2_SETUP:-}"
SKIP_VIEWS="${SKIP_VIEWS:-}"
DRY_RUN="${DRY_RUN:-0}"
NO_MERGE="${NO_MERGE:-0}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
D2_RUNNER="${D2_RUNNER:-${SCRIPT_DIR}/d2_infer_chunk.py}"
WORKER="${SCRIPT_DIR}/detectron2_chunk.slurm"
[ -f "$WORKER" ] || { echo "ERROR: missing $WORKER" >&2; exit 1; }
[ -f "$D2_RUNNER" ] || { echo "ERROR: missing runner $D2_RUNNER (mask_io.py must sit beside it)" >&2; exit 1; }

EXP_DIR="${MEDAKA_BUCKET_ROOT}/${EXPERIMENT}"
[ -d "$EXP_DIR" ] || { echo "ERROR: experiment dir not found: $EXP_DIR" >&2; exit 1; }

WORK_EXP="${MEDAKA_WORK_ROOT}/${EXPERIMENT}"
BUCKET_EXP_OUT="${MEDAKA_BUCKET_OUT}/${EXPERIMENT}"
mkdir -p "$WORK_EXP"

echo "============================================"
echo "  Medaka Detectron2 submission: ${EXPERIMENT}"
echo "============================================"
echo "Bucket in     : $EXP_DIR"
echo "Work scratch  : $WORK_EXP"
echo "Bucket out    : $BUCKET_EXP_OUT"
echo "Partition     : $PARTITION    GRES: $GRES    Time limit: $TIME_LIMIT"
echo "Chunks/view   : $NUM_CHUNKS   Score thr: $SCORE_THRESH   Frame stride: $FRAME_STRIDE"
echo "Side model    : $SIDE_D2_MODEL"
echo "Top model     : ${TOP_D2_MODEL:-<none — top_cam skipped unless TOP_D2_MODEL set>}"
echo "D2_PYTHON     : $D2_PYTHON"
echo ""

probe_frames() {
    local video="$1"
    local cam_dir meta n
    cam_dir="$(dirname "$video")"
    meta="${cam_dir}/metadata.csv"
    if [ -f "$meta" ]; then
        n=$(awk -F'"' '$2 == "totalFrames" { print $4; exit }' "$meta")
        if [[ "$n" =~ ^[0-9]+$ ]] && [ "$n" -gt 0 ]; then echo "$n"; return; fi
    fi
    local ft="${cam_dir}/frametimes.npy"
    if [ -f "$ft" ] && command -v python3 >/dev/null 2>&1; then
        n=$(python3 -c "import numpy as np,sys; a=np.load(sys.argv[1]); print(a.shape[-1])" "$ft" 2>/dev/null || true)
        if [[ "$n" =~ ^[0-9]+$ ]] && [ "$n" -gt 0 ]; then echo "$n"; return; fi
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
    for s in $SKIP_VIEWS; do [ "$s" = "$view" ] && return 0; done
    return 1
}

submit_view() {
    # $1=view $2=video_src $3=d2_model $4=d2_config
    local view="$1" video_src="$2" d2_model="$3" d2_config="$4"

    if should_skip "$view"; then echo "[${view}] skipped (SKIP_VIEWS)" >&2; return 0; fi
    if [ -z "$d2_model" ] || [ -z "$d2_config" ]; then
        echo "[${view}] no Detectron2 model/config configured — skipped" >&2; return 0
    fi
    if [ ! -f "$video_src" ]; then echo "[${view}] MISSING video: $video_src" >&2; return 1; fi

    local frames=0
    if [ "$DRY_RUN" != "2" ]; then frames=$(probe_frames "$video_src"); fi
    if [ "$frames" -eq 0 ]; then
        if [ "$DRY_RUN" != "0" ]; then
            frames=1000000
            echo "[${view}] DRY RUN: assuming $frames frames" >&2
        else
            echo "[${view}] ERROR: could not probe frames for $video_src" >&2; return 1
        fi
    fi

    local view_out="${WORK_EXP}/${view}"
    mkdir -p "$view_out"
    local chunk_file="${view_out}/chunks.txt"
    local n_chunks="$NUM_CHUNKS"
    [ "$frames" -lt "$n_chunks" ] && n_chunks=1
    write_chunks "$frames" "$n_chunks" "$chunk_file"
    local actual_chunks
    actual_chunks=$(wc -l < "$chunk_file")

    local video_staged="${view_out}/$(basename "$video_src")"
    echo "[${view}] frames=$frames  chunks=$actual_chunks  out=$view_out" >&2

    local sbatch_cmd=(
        sbatch --parsable
        --partition="$PARTITION"
        --gres="$GRES"
        --time="$TIME_LIMIT"
        --array="0-$((actual_chunks - 1))"
        --job-name="medakaD2_${EXPERIMENT}_${view}"
        --output="${view_out}/slurm_%A_%a.out"
        --error="${view_out}/slurm_%A_%a.err"
        --export=ALL,VIDEO_SRC="$video_src",VIDEO_STAGED="$video_staged",D2_MODEL="$d2_model",D2_CONFIG="$d2_config",OUTPUT_DIR="$view_out",CHUNK_FILE="$chunk_file",VIEW="$view",D2_RUNNER="$D2_RUNNER",SCORE_THRESH="$SCORE_THRESH",FRAME_STRIDE="$FRAME_STRIDE",D2_PYTHON="$D2_PYTHON",D2_SETUP="$D2_SETUP"
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

TOP_JOB=$(submit_view "top_cam"   "${EXP_DIR}/top_cam/cam0_top.mp4"    "$TOP_D2_MODEL"  "$TOP_D2_CONFIG"  || true)
S1_JOB=$(submit_view  "side1_cam" "${EXP_DIR}/side1_cam/cam1_side.mp4" "$SIDE_D2_MODEL" "$SIDE_D2_CONFIG" || true)
S2_JOB=$(submit_view  "side2_cam" "${EXP_DIR}/side2_cam/cam2_side.mp4" "$SIDE_D2_MODEL" "$SIDE_D2_CONFIG" || true)

DEP_LIST=""
for jid in "$TOP_JOB" "$S1_JOB" "$S2_JOB"; do
    [[ "$jid" =~ ^[0-9]+$ ]] || continue
    if [ -z "$DEP_LIST" ]; then DEP_LIST="$jid"; else DEP_LIST="${DEP_LIST}:${jid}"; fi
done

if [ "$DRY_RUN" != "0" ]; then
    echo ""; echo "DRY RUN complete. Nothing was submitted."; exit 0
fi
if [ -z "$DEP_LIST" ]; then
    echo ""; echo "No jobs were submitted — skipping copy-back step." >&2; exit 1
fi

# ── Background copy-back (login node; /bucket is read-only from compute nodes) ──
WATCHER="${WORK_EXP}/copy_back.sh"
DEPS_COLON_SEPARATED="${DEP_LIST}"
cat > "$WATCHER" << EOF
#!/bin/bash
set -euo pipefail
JIDS="${DEPS_COLON_SEPARATED//:/ }"
echo "\$(date): waiting on jobs: \$JIDS"
MAX_WAIT=\$(( 60 * 60 * 24 ))
ELAPSED=0
while [ "\$ELAPSED" -lt "\$MAX_WAIT" ]; do
    REMAINING=0
    for jid in \$JIDS; do
        if squeue -h -j "\$jid" 2>/dev/null | grep -q .; then REMAINING=\$((REMAINING + 1)); fi
    done
    [ "\$REMAINING" -eq 0 ] && break
    sleep 60; ELAPSED=\$((ELAPSED + 60))
done
echo "\$(date): arrays done, copying mask chunks back"
bash "${SCRIPT_DIR}/merge_detectron2_chunks.sh" "${WORK_EXP}" "${BUCKET_EXP_OUT}"
echo "\$(date): finished"
EOF
chmod +x "$WATCHER"

LOG="${WORK_EXP}/copy_back.log"
nohup bash "$WATCHER" >> "$LOG" 2>&1 &
WATCHER_PID=$!

echo ""
echo "Background watcher : PID $WATCHER_PID  (log: $LOG)"
echo "Monitor arrays     : squeue -u \$USER"
echo "Work dir           : $WORK_EXP"
echo "Final destination  : $BUCKET_EXP_OUT"
echo "Safe to disconnect — watcher runs under nohup."
