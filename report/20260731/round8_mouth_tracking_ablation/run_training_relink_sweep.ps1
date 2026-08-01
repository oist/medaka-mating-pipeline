param(
    [string]$Python = ".venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$reportRoot = "report\20260731\round8_mouth_tracking_ablation"
$geometryRoot = Join-Path $reportRoot "training_geometry"
$outputRoot = Join-Path $reportRoot "training_relink"

$windows = @(
    @{ Name = "reviewed_event_1"; Start = 72371; End = 73027 },
    @{ Name = "reviewed_event_3"; Start = 174316; End = 180515 },
    @{ Name = "control_for_event_1"; Start = 32264; End = 32920 },
    @{ Name = "control_for_event_3"; Start = 21840; End = 28039 }
)
$methods = @(
    @{
        Name = "A1"
        Arguments = @("--method", "A1")
    },
    @{
        Name = "A2_v025_h5"
        Arguments = @(
            "--method", "A2", "--velocity-weight", "0.25",
            "--motion-history-frames", "5"
        )
    },
    @{
        Name = "A2_v050_h5"
        Arguments = @(
            "--method", "A2", "--velocity-weight", "0.5",
            "--motion-history-frames", "5"
        )
    },
    @{
        Name = "A2_v100_h5"
        Arguments = @(
            "--method", "A2", "--velocity-weight", "1.0",
            "--motion-history-frames", "5"
        )
    },
    @{
        Name = "A2_v050_h9"
        Arguments = @(
            "--method", "A2", "--velocity-weight", "0.5",
            "--motion-history-frames", "9"
        )
    },
    @{
        Name = "A2_v100_h9"
        Arguments = @(
            "--method", "A2", "--velocity-weight", "1.0",
            "--motion-history-frames", "9"
        )
    }
)

foreach ($window in $windows) {
    $geometry = Join-Path (
        Join-Path $geometryRoot $window.Name
    ) "strict_mouth_selected_geometry.parquet"
    foreach ($method in $methods) {
        $output = Join-Path (
            Join-Path $outputRoot $window.Name
        ) $method.Name
        New-Item -ItemType Directory -Force -Path $output | Out-Null
        foreach ($staleName in @(
            "relink.stdout.log",
            "relink.stderr.log",
            "relink_resource_samples.csv",
            "relink_resource_summary.json"
        )) {
            $stalePath = Join-Path $output $staleName
            if (Test-Path -LiteralPath $stalePath) {
                Remove-Item -LiteralPath $stalePath -Force
            }
        }
        $arguments = @(
            "-m", "scripts.reconstruction_3d.mouth_ablation", "relink",
            "--geometry", $geometry,
            "--output-dir", $output,
            "--frame-start", [string]$window.Start,
            "--frame-end", [string]$window.End,
            "--tangent-weight", "0",
            "--acceleration-weight", "0",
            "--uncertainty-growth", "0.25",
            "--max-motion-gap-frames", "20",
            "--max-jump-mm", "8",
            "--max-jump-hard-cap-mm", "80",
            "--max-gap-frames", "100"
        ) + $method.Arguments
        $process = Start-Process -FilePath $Python `
            -ArgumentList $arguments `
            -RedirectStandardOutput (Join-Path $output "relink.stdout.log") `
            -RedirectStandardError (Join-Path $output "relink.stderr.log") `
            -WindowStyle Hidden -PassThru
        Wait-Process -Id $process.Id
        if ((Get-Item (Join-Path $output "relink.stderr.log")).Length -gt 0) {
            throw "relink failed or warned; inspect $output"
        }
    }
}
