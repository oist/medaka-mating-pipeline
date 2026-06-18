# Launch the crossing-review GUI for one round5 experiment.
#
# Usage (from anywhere):
#   .\scripts\analysis\review_crossings.ps1 20251225_K_1_2_FCMC
#
# Reviews the 40 tightest crossings; records female ID / swap / severity / notes
# to  reconstruction_3d\round5_260611\<exp>\round5_260611_crossing_review.csv
# (resumable — re-run to continue where you left off).
param([Parameter(Mandatory = $true)][string]$exp)
$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)   # scripts\analysis -> repo root
$py = Join-Path $repo ".venv\Scripts\python.exe"
$R = "Z:\StephensU\medaka"
$model = Join-Path $repo "calibration_results\cali_20251217_manual\merged_rule_reassigned_beads_only\models"
Set-Location $repo

& $py -m scripts.analysis.crossing_review_gui `
  --csv-3d "$R\reconstruction_3d\round5_260611\$exp\round5_260611_3d_identified.csv" `
  --slp-dir "$R\sleap_output\round5_260611\$exp" `
  --model-folder "$model" `
  --video-top "$R\medaka_202512\$exp\top_cam\cam0_top.mp4" `
  --video-side1 "$R\medaka_202512\$exp\side1_cam\cam1_side.mp4" `
  --video-side2 "$R\medaka_202512\$exp\side2_cam\cam2_side.mp4" `
  --out-csv "$R\reconstruction_3d\round5_260611\$exp\round5_260611_crossing_review.csv"
