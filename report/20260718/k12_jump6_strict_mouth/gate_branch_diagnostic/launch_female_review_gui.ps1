param(
    [Parameter(Mandatory = $true)]
    [string]$Annotator
)

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot '..\..\..\..')
Push-Location $RepoRoot
try {
    & .venv\Scripts\python.exe -m scripts.analysis.female_sex_gt_annotator `
        20251225_K_1_2_FCMC `
        --candidate-csv report\20260718\k12_jump6_strict_mouth\gate_branch_diagnostic\female_annotator_candidates.csv `
        --slp-dir '\\bucket.oist.jp\bucket\StephensU\medaka\sleap_output\20251225_K_1_2_FCMC\round7_260704' `
        --video-dir '\\bucket.oist.jp\bucket\StephensU\medaka\medaka_202512\20251225_K_1_2_FCMC' `
        --geometry-csv report\20260718\k12_jump6_strict_mouth\gate_branch_diagnostic\20251225_K_1_2_FCMC\jump6_gate_review_geometry.csv `
        --geometry-id-column fish_id `
        --calibration-model-dir calibration_results\fixed_refractive_192x192x186 `
        --calibration-backend refractive `
        --annotations-csv annotations\fish_sex_gt\20251225_K_1_2_FCMC\jump6_gate_branch_female_review.csv `
        --audit-jsonl annotations\fish_sex_gt\20251225_K_1_2_FCMC\jump6_gate_branch_female_review.audit.jsonl `
        --annotator $Annotator `
        --context-frames 166
}
finally {
    Pop-Location
}
