#!/bin/bash
# ==============================================================================
# Submit SLEAP inference for MANY medaka experiments on saion.
#
# Walks MEDAKA_BUCKET_ROOT, filters folders by regex, and calls
# submit_experiment.sh for each. Respects Slurm's queue limits automatically
# (largegpu = 8 nodes for this account); extra experiments wait in queue.
#
# Usage:
#   bash scripts/hpc/submit_batch.sh [--experiments <regex>]
#
# All environment variables documented in submit_experiment.sh apply here too
# (MEDAKA_BUCKET_ROOT, NUM_CHUNKS, PARTITION, etc).
#
# Extra options:
#   --experiments <regex>   only submit experiments whose folder name matches
#                           (default: ^[0-9]{8}_[A-Z]_[0-9]+_[0-9]+_FCMC$ —
#                           recording folders, skipping cali_*/*_test/*_TEST)
#   --skip-existing         skip experiments already present in $MEDAKA_BUCKET_OUT
#                           with all three merged view .slp files
#   --dry-run               forward DRY_RUN=1 to submit_experiment.sh
#   --list                  print the matching experiment list and exit
# ==============================================================================

set -euo pipefail

REGEX='^[0-9]{8}_[A-Z]_[0-9]+_[0-9]+_FCMC$'
SKIP_EXISTING=0
DRY_RUN=0
LIST_ONLY=0

while [ $# -gt 0 ]; do
    case "$1" in
        --experiments) REGEX="$2"; shift 2;;
        --skip-existing) SKIP_EXISTING=1; shift;;
        --dry-run) DRY_RUN=1; shift;;
        --list) LIST_ONLY=1; shift;;
        -h|--help) sed -n '2,26p' "$0"; exit 0;;
        *) echo "Unknown arg: $1" >&2; exit 2;;
    esac
done

MEDAKA_BUCKET_ROOT="${MEDAKA_BUCKET_ROOT:-/bucket/StephensU/medaka/medaka_202512}"
MEDAKA_BUCKET_OUT="${MEDAKA_BUCKET_OUT:-/bucket/StephensU/medaka/sleap_output}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SUBMIT="${SCRIPT_DIR}/submit_experiment.sh"
[ -x "$SUBMIT" ] || chmod +x "$SUBMIT" || true

[ -d "$MEDAKA_BUCKET_ROOT" ] || { echo "ERROR: $MEDAKA_BUCKET_ROOT not found" >&2; exit 1; }

mapfile -t CANDIDATES < <(ls -1 "$MEDAKA_BUCKET_ROOT" 2>/dev/null | sort)
SELECTED=()
for name in "${CANDIDATES[@]}"; do
    [ -d "${MEDAKA_BUCKET_ROOT}/${name}/top_cam" ] || continue
    [[ "$name" =~ $REGEX ]] || continue
    if [ "$SKIP_EXISTING" -eq 1 ]; then
        out="${MEDAKA_BUCKET_OUT}/${name}"
        if [ -s "${out}/top_cam.slp" ] && [ -s "${out}/side1_cam.slp" ] && [ -s "${out}/side2_cam.slp" ]; then
            echo "skip (already done): $name"
            continue
        fi
    fi
    SELECTED+=("$name")
done

echo "Matched ${#SELECTED[@]} experiments under ${MEDAKA_BUCKET_ROOT}"
echo "Regex: $REGEX"

if [ "$LIST_ONLY" -eq 1 ]; then
    printf '  %s\n' "${SELECTED[@]}"
    exit 0
fi

if [ "${#SELECTED[@]}" -eq 0 ]; then
    echo "Nothing to submit."
    exit 0
fi

for name in "${SELECTED[@]}"; do
    echo ""
    echo "############################################"
    echo "  $name"
    echo "############################################"
    if [ "$DRY_RUN" -eq 1 ]; then
        DRY_RUN=1 bash "$SUBMIT" "$name" || true
    else
        bash "$SUBMIT" "$name" || echo "(submission failed for $name — continuing)"
    fi
done

echo ""
echo "All done. Monitor: squeue -u \$USER"
