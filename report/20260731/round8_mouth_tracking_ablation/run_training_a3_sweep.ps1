param(
    [string]$Python = ".venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$reportRoot = "report\20260731\round8_mouth_tracking_ablation"
$geometryRoot = Join-Path $reportRoot "training_geometry"
$outputRoot = Join-Path $reportRoot "training_relink_a3"

$windows = @(
    @{ Name = "reviewed_event_1"; Start = 72371; End = 73027 },
    @{ Name = "reviewed_event_3"; Start = 174316; End = 180515 },
    @{ Name = "control_for_event_1"; Start = 32264; End = 32920 },
    @{ Name = "control_for_event_3"; Start = 21840; End = 28039 }
)
$accelerations = @(
    @{ Name = "A3_a000"; Weight = "0" },
    @{ Name = "A3_a005"; Weight = "0.05" },
    @{ Name = "A3_a010"; Weight = "0.1" }
)

foreach ($window in $windows) {
    $geometry = Join-Path (
        Join-Path $geometryRoot $window.Name
    ) "strict_mouth_selected_geometry.parquet"
    foreach ($acceleration in $accelerations) {
        $output = Join-Path (
            Join-Path $outputRoot $window.Name
        ) $acceleration.Name
        New-Item -ItemType Directory -Force -Path $output | Out-Null
        $arguments = @(
            "-m", "scripts.reconstruction_3d.mouth_ablation", "relink",
            "--geometry", $geometry,
            "--output-dir", $output,
            "--method", "A3",
            "--frame-start", [string]$window.Start,
            "--frame-end", [string]$window.End,
            "--velocity-weight", "0.25",
            "--tangent-weight", "0",
            "--acceleration-weight", $acceleration.Weight,
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
            "--maximum-defer-horizon", "100"
        )
        $stdout = Join-Path $output "relink.stdout.log"
        $stderr = Join-Path $output "relink.stderr.log"
        $process = Start-Process -FilePath $Python `
            -ArgumentList $arguments `
            -RedirectStandardOutput $stdout `
            -RedirectStandardError $stderr `
            -WindowStyle Hidden -PassThru
        Wait-Process -Id $process.Id
        if ($process.ExitCode -ne 0) {
            throw "A3 relink exited $($process.ExitCode); inspect $output"
        }
        if ((Get-Item -LiteralPath $stderr).Length -gt 0) {
            throw "A3 relink wrote stderr; inspect $output"
        }
    }
}
