#!/usr/bin/env bash
set -euo pipefail

# Summarize common SLEAP inference outputs from a directory.
#
# Usage:
#   ./scripts/sleap/summarize_inference_logs.sh runs/

ROOT_DIR="${1:-runs}"

if [[ ! -d "${ROOT_DIR}" ]]; then
	echo "Not a directory: ${ROOT_DIR}" >&2
	exit 2
fi

echo "== SLEAP prediction files (.slp) =="
find "${ROOT_DIR}" -type f -name "*.slp" -print | sort

echo
echo "== SLEAP analysis exports (analysis.h5 / analysis.csv) =="
find "${ROOT_DIR}" -type f \( -name "*.analysis.h5" -o -name "*.analysis.csv" \) -print | sort
