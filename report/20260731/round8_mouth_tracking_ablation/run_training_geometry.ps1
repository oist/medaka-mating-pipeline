param(
    [string]$Python = ".venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$reportRoot = "report\20260731\round8_mouth_tracking_ablation"
$cache = Join-Path $reportRoot "candidate_cache_strict_mouth_unfiltered_nodescores"
$outputRoot = Join-Path $reportRoot "training_geometry"
$windows = @(
    @{ Name = "reviewed_event_1"; Start = 72371; End = 73027 },
    @{ Name = "reviewed_event_3"; Start = 174316; End = 180515 },
    @{ Name = "control_for_event_1"; Start = 32264; End = 32920 },
    @{ Name = "control_for_event_3"; Start = 21840; End = 28039 }
)

foreach ($window in $windows) {
    $output = Join-Path $outputRoot $window.Name
    New-Item -ItemType Directory -Force -Path $output | Out-Null
    $arguments = @(
        "-m", "scripts.reconstruction_3d.mouth_ablation", "geometry",
        "--cache-dir", $cache,
        "--paths-config", "configs\paths.local.toml",
        "--output-dir", $output,
        "--frame-start", [string]$window.Start,
        "--frame-end", [string]$window.End,
        "--block-frames", "500",
        "--evaluation-batch-frames", "128",
        "--xyz-bounds", "-20", "220"
    )
    $stdout = Join-Path $output "geometry.stdout.log"
    $stderr = Join-Path $output "geometry.stderr.log"
    $process = Start-Process -FilePath $Python `
        -ArgumentList $arguments `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -WindowStyle Hidden -PassThru
    Wait-Process -Id $process.Id
    if ($process.ExitCode -ne 0) {
        throw "geometry exited $($process.ExitCode); inspect $output"
    }
    if ((Get-Item -LiteralPath $stderr).Length -gt 0) {
        throw "geometry wrote stderr; inspect $output"
    }
}
