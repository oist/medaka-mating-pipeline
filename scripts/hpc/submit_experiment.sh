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
#   ROUND_TAG            optional single path component placed after <experiment>
#                        in both work and bucket output paths
#   TOP_CENTROID         /bucket/StephensU/medaka/medaka_202512/sleap_models/sleap_top_mouth/models/260313_152145.centroid.n=330
#   TOP_INSTANCE         /bucket/StephensU/medaka/medaka_202512/sleap_models/sleap_top_mouth/models/260313_175737.centered_instance.n=330
#   SIDE_CENTROID        /bucket/StephensU/medaka/medaka_202512/sleap_models/sleap_side_mouth/models/260111_170857.centroid.n=593
#   SIDE_INSTANCE        /bucket/StephensU/medaka/medaka_202512/sleap_models/sleap_side_mouth/models/260112_115936.centered_instance.n=593
#   NUM_CHUNKS           4     (array tasks per view; 1 = no chunking)
#   MAX_INSTANCES        3     (1 female + 2 males)
#   BATCH_SIZE           16
#   PARTITION            largegpu   (or gpu / test-gpu)
#   TIME_LIMIT           12:00:00
#   SLEAP_PYTHON         optional absolute Python path; skips module loading and
#                        runs `python -m sleap_nn.cli track`
#   SLEAP_MODULE         sleap-nn/0.1.2   (also: sleap-nn/0.1.0, sleap/1.5.2, sleap/1.6.2)
#   MODULEPATH_EXTRA     /apps/unit/ReiterU/.modulefiles
#   SKIP_VIEWS           (optional) space-separated views to skip, e.g. "side2_cam"
#   REQUIRE_ALL_VIEWS    1 = reject SKIP_VIEWS and refuse finalization unless
#                        all three view arrays were submitted
#   DRY_RUN              1 = print commands only, 2 = also skip ffprobe
#   NO_MERGE             1 = skip final merge after arrays complete
#   WATCHER_MAX_WAIT_SECONDS  maximum queue+run wait before finalizer aborts
#                             (default 604800 = 7 days)
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
ROUND_TAG="${ROUND_TAG:-}"

TOP_CENTROID="${TOP_CENTROID:-/bucket/StephensU/medaka/medaka_202512/sleap_models/sleap_top_mouth/models/260313_152145.centroid.n=330}"
TOP_INSTANCE="${TOP_INSTANCE:-/bucket/StephensU/medaka/medaka_202512/sleap_models/sleap_top_mouth/models/260313_175737.centered_instance.n=330}"
SIDE_CENTROID="${SIDE_CENTROID:-/bucket/StephensU/medaka/medaka_202512/sleap_models/sleap_side_mouth/models/260111_170857.centroid.n=593}"
SIDE_INSTANCE="${SIDE_INSTANCE:-/bucket/StephensU/medaka/medaka_202512/sleap_models/sleap_side_mouth/models/260112_115936.centered_instance.n=593}"

NUM_CHUNKS="${NUM_CHUNKS:-4}"
MAX_INSTANCES="${MAX_INSTANCES:-3}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PARTITION="${PARTITION:-largegpu}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
SLEAP_PYTHON="${SLEAP_PYTHON:-}"
SLEAP_MODULE="${SLEAP_MODULE:-sleap-nn/0.1.2}"
MODULEPATH_EXTRA="${MODULEPATH_EXTRA:-/apps/unit/ReiterU/.modulefiles}"
SKIP_VIEWS="${SKIP_VIEWS:-}"
REQUIRE_ALL_VIEWS="${REQUIRE_ALL_VIEWS:-0}"
DRY_RUN="${DRY_RUN:-0}"
NO_MERGE="${NO_MERGE:-0}"
WATCHER_MAX_WAIT_SECONDS="${WATCHER_MAX_WAIT_SECONDS:-604800}"

if [ -n "$ROUND_TAG" ] && [[ ! "$ROUND_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "ERROR: ROUND_TAG must be one safe path component: $ROUND_TAG" >&2
    exit 2
fi
if ! [[ "$WATCHER_MAX_WAIT_SECONDS" =~ ^[0-9]+$ ]] || [ "$WATCHER_MAX_WAIT_SECONDS" -le 0 ]; then
    echo "ERROR: WATCHER_MAX_WAIT_SECONDS must be a positive integer" >&2
    exit 2
fi
if [ "$REQUIRE_ALL_VIEWS" = "1" ] && [ -n "$SKIP_VIEWS" ]; then
    echo "ERROR: REQUIRE_ALL_VIEWS=1 is incompatible with SKIP_VIEWS" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKER="${SCRIPT_DIR}/sleap_chunk.slurm"
[ -f "$WORKER" ] || { echo "ERROR: missing $WORKER" >&2; exit 1; }

EXP_DIR="${MEDAKA_BUCKET_ROOT}/${EXPERIMENT}"
[ -d "$EXP_DIR" ] || { echo "ERROR: experiment dir not found: $EXP_DIR" >&2; exit 1; }

WORK_EXP="${MEDAKA_WORK_ROOT}/${EXPERIMENT}"
BUCKET_EXP_OUT="${MEDAKA_BUCKET_OUT}/${EXPERIMENT}"
if [ -n "$ROUND_TAG" ]; then
    WORK_EXP="${WORK_EXP}/${ROUND_TAG}"
    BUCKET_EXP_OUT="${BUCKET_EXP_OUT}/${ROUND_TAG}"
fi
mkdir -p "$WORK_EXP"

verify_model_dir() {
    local label="$1" model_dir="$2"
    [ -d "$model_dir" ] || { echo "ERROR: ${label} model dir missing: $model_dir" >&2; exit 1; }
    [ -s "${model_dir}/best.ckpt" ] || {
        echo "ERROR: ${label} best.ckpt missing/empty: ${model_dir}/best.ckpt" >&2
        exit 1
    }
    [ -s "${model_dir}/training_config.yaml" ] || {
        echo "ERROR: ${label} training_config.yaml missing/empty: ${model_dir}/training_config.yaml" >&2
        exit 1
    }
}

verify_model_dir "top centroid" "$TOP_CENTROID"
verify_model_dir "top instance" "$TOP_INSTANCE"
verify_model_dir "side centroid" "$SIDE_CENTROID"
verify_model_dir "side instance" "$SIDE_INSTANCE"
if [ -n "$SLEAP_PYTHON" ] && [ ! -x "$SLEAP_PYTHON" ]; then
    echo "ERROR: SLEAP_PYTHON is not executable: $SLEAP_PYTHON" >&2
    exit 1
fi
if [ "$REQUIRE_ALL_VIEWS" = "1" ]; then
    for required_video in \
        "${EXP_DIR}/top_cam/cam0_top.mp4" \
        "${EXP_DIR}/side1_cam/cam1_side.mp4" \
        "${EXP_DIR}/side2_cam/cam2_side.mp4"; do
        [ -f "$required_video" ] || {
            echo "ERROR: required production video missing: $required_video" >&2
            exit 1
        }
    done
fi

echo "============================================"
echo "  Medaka SLEAP submission: ${EXPERIMENT}"
echo "============================================"
echo "Bucket in     : $EXP_DIR"
echo "Work scratch  : $WORK_EXP"
echo "Bucket out    : $BUCKET_EXP_OUT"
echo "Round tag     : ${ROUND_TAG:-<none>}"
echo "Partition     : $PARTITION    Time limit: $TIME_LIMIT"
echo "Chunks/view   : $NUM_CHUNKS   Max instances: $MAX_INSTANCES"
echo "SLEAP Python  : ${SLEAP_PYTHON:-<module mode>}"
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
    # Prints "<job_id>:<task_count>" on stdout; status messages go to stderr so
    # the caller can capture the accounting specification cleanly with $(...).
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
        --export=ALL,VIDEO_SRC="$video_src",VIDEO_STAGED="$video_staged",CENTROID_MODEL="$centroid",INSTANCE_MODEL="$instance",OUTPUT_DIR="$view_out",CHUNK_FILE="$chunk_file",VIEW="$view",MAX_INSTANCES="$MAX_INSTANCES",BATCH_SIZE="$BATCH_SIZE",SLEAP_PYTHON="$SLEAP_PYTHON",SLEAP_MODULE="$SLEAP_MODULE",MODULEPATH_EXTRA="${MODULEPATH_EXTRA:-}"
        "$WORKER"
    )

    if [ "$DRY_RUN" != "0" ]; then
        printf '   DRY: %s\n' "${sbatch_cmd[*]}" >&2
        return 0
    fi

    local jid
    jid=$("${sbatch_cmd[@]}")
    echo "[${view}] submitted array $jid (tasks=$actual_chunks)" >&2
    echo "${jid}:${actual_chunks}"
}

# ── Submit all three views ──
TOP_JOB=$(submit_view "top_cam"   "${EXP_DIR}/top_cam/cam0_top.mp4"    "$TOP_CENTROID"  "$TOP_INSTANCE"  || true)
S1_JOB=$(submit_view  "side1_cam" "${EXP_DIR}/side1_cam/cam1_side.mp4" "$SIDE_CENTROID" "$SIDE_INSTANCE" || true)
S2_JOB=$(submit_view  "side2_cam" "${EXP_DIR}/side2_cam/cam2_side.mp4" "$SIDE_CENTROID" "$SIDE_INSTANCE" || true)

DEP_LIST=""
JOB_SPECS=""
for spec in "$TOP_JOB" "$S1_JOB" "$S2_JOB"; do
    [[ "$spec" =~ ^[0-9]+:[0-9]+$ ]] || continue
    jid="${spec%%:*}"
    if [ -z "$DEP_LIST" ]; then DEP_LIST="$jid"; else DEP_LIST="${DEP_LIST}:${jid}"; fi
    if [ -z "$JOB_SPECS" ]; then JOB_SPECS="$spec"; else JOB_SPECS="${JOB_SPECS} ${spec}"; fi
done

if [ "$DRY_RUN" != "0" ]; then
    echo ""
    echo "DRY RUN complete. Nothing was submitted."
    exit 0
fi

if [ "$REQUIRE_ALL_VIEWS" = "1" ]; then
    read -r -a submitted_specs <<< "$JOB_SPECS"
    if [ "${#submitted_specs[@]}" -ne 3 ]; then
        echo "ERROR: required 3 view arrays, but submitted ${#submitted_specs[@]}." >&2
        echo "Submitted specs: ${JOB_SPECS:-<none>}" >&2
        echo "No copy/finalization watcher will be started." >&2
        exit 1
    fi
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
cat > "$WATCHER" << EOF
#!/bin/bash
set -euo pipefail
JOB_SPECS="${JOB_SPECS}"
echo "\$(date): waiting on array specs: \$JOB_SPECS"

MAX_WAIT="${WATCHER_MAX_WAIT_SECONDS}"
ELAPSED=0
while [ "\$ELAPSED" -lt "\$MAX_WAIT" ]; do
    ALL_COMPLETE=1
    FAILED_ROWS=()
    for spec in \$JOB_SPECS; do
        jid="\${spec%%:*}"
        expected="\${spec##*:}"
        mapfile -t rows < <(
            sacct -n -X -P -j "\$jid" --format=JobID,State,ExitCode 2>/dev/null |
                awk -F'|' -v prefix="\${jid}_" '\$1 ~ ("^" prefix "[0-9]+\$") {print \$1 "|" \$2 "|" \$3}'
        )
        if [ "\${#rows[@]}" -ne "\$expected" ]; then
            ALL_COMPLETE=0
            echo "\$(date): array \$jid accounting rows \${#rows[@]}/\$expected"
            continue
        fi
        for row in "\${rows[@]}"; do
            IFS='|' read -r task_id state exit_code <<< "\$row"
            state="\${state%% *}"
            state="\${state%+}"
            if [ "\$state" = "COMPLETED" ] && [ "\$exit_code" = "0:0" ]; then
                continue
            fi
            case "\$state" in
                FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED)
                    FAILED_ROWS+=("\${task_id}|\${state}|\${exit_code}")
                    ;;
                COMPLETED)
                    FAILED_ROWS+=("\${task_id}|\${state}|\${exit_code}")
                    ;;
                *)
                    ALL_COMPLETE=0
                    ;;
            esac
        done
    done
    if [ "\${#FAILED_ROWS[@]}" -gt 0 ]; then
        echo "\$(date): refusing copy; terminal task failures:" >&2
        printf '  %s\n' "\${FAILED_ROWS[@]}" >&2
        exit 1
    fi
    [ "\$ALL_COMPLETE" -eq 1 ] && break
    sleep 60
    ELAPSED=\$((ELAPSED + 60))
done

if [ "\$ELAPSED" -ge "\$MAX_WAIT" ]; then
    echo "\$(date): refusing copy; watcher timed out after \$MAX_WAIT seconds" >&2
    exit 1
fi

echo "\$(date): every expected array task is COMPLETED with ExitCode 0:0"
echo "\$(date): copying chunks and logs"
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
