[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$Annotator,

    [switch]$Preflight
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Workspace Python was not found: $python"
}

$experiment = "20251225_K_1_2_FCMC"
$jump6 = Join-Path $repoRoot "report\20260718\k12_jump6_strict_mouth\mouth_only_jump6_3d_aligned_to_jump8.csv"
$jump8 = Join-Path $repoRoot "report\20260718\k12_jump8_strict_mouth_rerun\mouth_only_jump8_rerun_3d_aligned_to_historical_frame.csv"
$videoDir = "\\bucket.oist.jp\bucket\StephensU\medaka\medaka_202512\20251225_K_1_2_FCMC"
$slpDir = "\\bucket.oist.jp\bucket\StephensU\medaka\sleap_output\20251225_K_1_2_FCMC\round7_260704"
$calibrationDir = Join-Path $repoRoot "calibration_results\fixed_refractive_192x192x186"
$annotationDir = Join-Path $repoRoot "annotations\contact_episode_gt\20251225_K_1_2_FCMC"

$arguments = @(
    "-m", "scripts.analysis.contact_episode_annotator",
    $experiment,
    "--paths-config", (Join-Path $repoRoot "configs\paths.local.toml"),
    "--tracking-csv", "jump6=$jump6",
    "--tracking-csv", "jump8=$jump8",
    "--detector-method", "jump8",
    "--video-dir", $videoDir,
    "--slp-dir", $slpDir,
    "--calibration-model-dir", $calibrationDir,
    "--calibration-backend", "auto",
    "--anchor-mode", "mouth",
    "--enter-mm", "12",
    "--exit-mm", "16",
    "--merge-gap-frames", "5",
    "--context-frames", "166",
    "--scene-cache-radius", "166",
    "--scene-cache-mb", "768",
    "--annotations-csv", (Join-Path $annotationDir "contact_episode_gt.csv"),
    "--audit-jsonl", (Join-Path $annotationDir "contact_episode_gt.audit.jsonl"),
    "--candidates-csv", (Join-Path $annotationDir "contact_candidates_jump8.csv"),
    "--detector-summary-json", (Join-Path $annotationDir "contact_candidates_jump8_summary.json"),
    "--ifd-cache-dir", (Join-Path $annotationDir "ifd_cache"),
    "--annotator", $Annotator
)

if ($Preflight) {
    $arguments += "--preflight"
}

Push-Location $repoRoot
try {
    & $python @arguments
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
