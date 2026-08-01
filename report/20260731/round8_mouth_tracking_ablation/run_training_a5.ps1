param(
    [string]$Python = ".venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$reportRoot = "report\20260731\round8_mouth_tracking_ablation"
$outputRoot = Join-Path $reportRoot "training_a5"
if (Test-Path -LiteralPath $outputRoot) {
    throw "Refusing to mix A5 training generations: $outputRoot exists"
}
New-Item -ItemType Directory -Path $outputRoot | Out-Null
$windows = @(
    @{ Name = "reviewed_event_1"; Start = 72371; End = 73027 },
    @{ Name = "reviewed_event_3"; Start = 174316; End = 180515 },
    @{ Name = "control_for_event_1"; Start = 32264; End = 32920 },
    @{ Name = "control_for_event_3"; Start = 21840; End = 28039 }
)

foreach ($window in $windows) {
    $output = Join-Path $outputRoot $window.Name
    New-Item -ItemType Directory -Path $output | Out-Null
    $a4 = Join-Path (
        Join-Path (
            Join-Path $reportRoot "training_a4_cutoff_sweep"
        ) $window.Name
    ) "s035_m005_g0\a4_selected_observations.parquet"
    $a2 = Join-Path (
        Join-Path (
            Join-Path $reportRoot "training_relink"
        ) $window.Name
    ) "A2_v025_h5\a2_direct.csv"
    $geometryDir = Join-Path (
        Join-Path $reportRoot "training_geometry"
    ) $window.Name
    $geometry = Join-Path $geometryDir "strict_mouth_selected_geometry.parquet"
    $geometrySummary = Join-Path $geometryDir "geometry_summary.json"

    $hintArguments = @(
        "-m", "scripts.reconstruction_3d.mouth_tracklet_ablation", "hints",
        "--a4-selected", $a4,
        "--a2-direct", $a2,
        "--output-dir", $output,
        "--minimum-mouth-confidence", "0.5",
        "--minimum-group-support", "3",
        "--maximum-group-median-residual", "1.0",
        "--maximum-group-mad-residual", "0.25",
        "--minimum-group-median-mouth-confidence", "0.5",
        "--maximum-group-assignment-cost", "0",
        "--minimum-mapping-votes", "3",
        "--minimum-mapping-purity", "0.9",
        "--minimum-vote-margin", "2",
        "--xyz-tolerance-mm", "1e-9"
    )
    $hintProcess = Start-Process -FilePath $Python `
        -ArgumentList $hintArguments `
        -RedirectStandardOutput (Join-Path $output "hints.stdout.log") `
        -RedirectStandardError (Join-Path $output "hints.stderr.log") `
        -WindowStyle Hidden -PassThru
    $hintProcess.WaitForExit()
    $hintProcess.Refresh()
    if ($hintProcess.ExitCode -ne 0) {
        throw "A5 hints failed for $($window.Name)"
    }
    if ((Get-Item -LiteralPath (Join-Path $output "hints.stderr.log")).Length -gt 0) {
        throw "A5 hints wrote stderr for $($window.Name)"
    }

    $a5Arguments = @(
        "-m", "scripts.reconstruction_3d.mouth_tracklet_ablation", "a5",
        "--geometry", $geometry,
        "--geometry-summary", $geometrySummary,
        "--hints", (Join-Path $output "a5_hints.csv"),
        "--forced-reset-frames", (Join-Path $output "a5_forced_reset_frames.json"),
        "--hints-summary", (Join-Path $output "a5_hints_summary.json"),
        "--output-dir", $output,
        "--frame-start", [string]$window.Start,
        "--frame-end", [string]$window.End,
        "--velocity-weight", "0.25",
        "--acceleration-weight", "0.05",
        "--uncertainty-growth", "0.25",
        "--max-motion-gap-frames", "20",
        "--motion-history-frames", "5",
        "--max-jump-mm", "8",
        "--max-jump-hard-cap-mm", "80",
        "--max-gap-frames", "100",
        "--beam-width", "16",
        "--entry-margin", "0.5",
        "--retention-margin", "2",
        "--resolution-margin", "1",
        "--resolution-stability-frames", "3",
        "--minimum-lookahead-frames", "3",
        "--maximum-defer-horizon", "100",
        "--xyz-tolerance-mm", "1e-9"
    )
    $a5Process = Start-Process -FilePath $Python `
        -ArgumentList $a5Arguments `
        -RedirectStandardOutput (Join-Path $output "a5.stdout.log") `
        -RedirectStandardError (Join-Path $output "a5.stderr.log") `
        -WindowStyle Hidden -PassThru
    $a5Process.WaitForExit()
    $a5Process.Refresh()
    if ($a5Process.ExitCode -ne 0) {
        throw "A5 run failed for $($window.Name)"
    }
    if ((Get-Item -LiteralPath (Join-Path $output "a5.stderr.log")).Length -gt 0) {
        throw "A5 run wrote stderr for $($window.Name)"
    }
}
