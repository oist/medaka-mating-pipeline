param(
    [string]$Python = ".venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$reportRoot = "report\20260731\round8_mouth_tracking_ablation"
$cache = Join-Path $reportRoot "candidate_cache_strict_mouth_unfiltered_nodescores"
$scales = Join-Path $reportRoot "frozen_body_length_scales.json"
$outputRoot = Join-Path $reportRoot "training_a4_cutoff_sweep"
if (Test-Path -LiteralPath $outputRoot) {
    throw "Refusing to mix generations: output root already exists: $outputRoot"
}
New-Item -ItemType Directory -Path $outputRoot | Out-Null
$expectedScalesHash = "63bf6de26bb85e76da97976ffa0158c8d4e27d7ea9e3208b627fe8ade2eb9209"
$expectedRunnerHash = "1e75e10654b5d6c233ec3ea5ff2f8c969dac3dd41027031cd332d3c88096d4e0"
$actualScalesHash = (Get-FileHash -LiteralPath $scales -Algorithm SHA256).Hash.ToLower()
if ($actualScalesHash -ne $expectedScalesHash) {
    throw "Frozen scales hash changed: $actualScalesHash"
}

$windows = @(
    @{ Name = "reviewed_event_1"; Start = 72371; End = 73027 },
    @{ Name = "reviewed_event_3"; Start = 174316; End = 180515 },
    @{ Name = "control_for_event_1"; Start = 32264; End = 32920 },
    @{ Name = "control_for_event_3"; Start = 21840; End = 28039 }
)
$separations = @(
    @{ Label = "s035"; Value = "0.35" },
    @{ Label = "s050"; Value = "0.5" },
    @{ Label = "s065"; Value = "0.65" }
)
$margins = @(
    @{ Label = "m005"; Value = "0.05" },
    @{ Label = "m010"; Value = "0.1" }
)
$gaps = @(
    @{ Label = "g0"; Value = "0" },
    @{ Label = "g2"; Value = "2" },
    @{ Label = "g5"; Value = "5" }
)
$validatedRuns = 0
$statusCounts = @{}

foreach ($separation in $separations) {
    foreach ($margin in $margins) {
        foreach ($gap in $gaps) {
            $configuration = "$($separation.Label)_$($margin.Label)_$($gap.Label)"
            $processes = @()
            foreach ($window in $windows) {
                $output = Join-Path (
                    Join-Path $outputRoot $window.Name
                ) $configuration
                New-Item -ItemType Directory -Force -Path $output | Out-Null
                $arguments = @(
                    "-m", "scripts.reconstruction_3d.mouth_tracklet_ablation",
                    "run",
                    "--cache-dir", $cache,
                    "--paths-config", "configs\paths.local.toml",
                    "--frozen-scales", $scales,
                    "--output-dir", $output,
                    "--frame-start", [string]$window.Start,
                    "--frame-end", [string]$window.End,
                    "--hypothesis-block-frames", "250",
                    "--evaluation-batch-frames", "128",
                    "--xyz-bounds", "-20", "220",
                    "--separation-body-lengths", $separation.Value,
                    "--ambiguity-margin-body-lengths", $margin.Value,
                    "--maximum-missing-gap-frames", $gap.Value,
                    "--minimum-mouth-score", "0.3",
                    "--minimum-fragment-observations", "2",
                    "--minimum-group-support", "3",
                    "--beam-width", "256"
                )
                $process = Start-Process -FilePath $Python `
                    -ArgumentList $arguments `
                    -RedirectStandardOutput (Join-Path $output "a4.stdout.log") `
                    -RedirectStandardError (Join-Path $output "a4.stderr.log") `
                    -WindowStyle Hidden -PassThru
                $processes += [pscustomobject]@{
                    Process = $process
                    Output = $output
                    Window = $window.Name
                }
            }
            foreach ($record in $processes) {
                $record.Process.WaitForExit()
                $record.Process.Refresh()
                if ($record.Process.ExitCode -ne 0) {
                    throw "$configuration/$($record.Window) exited $($record.Process.ExitCode)"
                }
                $stderr = Join-Path $record.Output "a4.stderr.log"
                if ((Get-Item -LiteralPath $stderr).Length -gt 0) {
                    throw "$configuration/$($record.Window) wrote stderr"
                }
                $summaryPath = Join-Path $record.Output "a4_summary.json"
                if (-not (Test-Path -LiteralPath $summaryPath)) {
                    throw "$configuration/$($record.Window) lacks summary"
                }
                $summary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
                $config = $summary.configuration
                if ($config.frozen_body_scales_sha256 -ne $expectedScalesHash) {
                    throw "$configuration/$($record.Window) binds wrong scales"
                }
                $runnerHash = $summary.code_sha256.'scripts\reconstruction_3d\mouth_tracklet_ablation.py'
                if ($runnerHash -ne $expectedRunnerHash) {
                    throw "$configuration/$($record.Window) binds wrong runner"
                }
                if (
                    [double]$config.tracklet_config.separation_body_lengths -ne
                    [double]$separation.Value
                ) {
                    throw "$configuration/$($record.Window) separation mismatch"
                }
                if (
                    [double]$config.tracklet_config.assignment_ambiguity_margin_body_lengths -ne
                    [double]$margin.Value
                ) {
                    throw "$configuration/$($record.Window) ambiguity mismatch"
                }
                if (
                    [int]$config.tracklet_config.maximum_missing_gap_frames -ne
                    [int]$gap.Value
                ) {
                    throw "$configuration/$($record.Window) gap mismatch"
                }
                foreach ($outputProperty in $summary.outputs.psobject.Properties) {
                    $artifact = $outputProperty.Value
                    if (-not (Test-Path -LiteralPath $artifact.path)) {
                        throw "$configuration/$($record.Window) missing $($outputProperty.Name)"
                    }
                    $artifactHash = (
                        Get-FileHash -LiteralPath $artifact.path -Algorithm SHA256
                    ).Hash.ToLower()
                    if ($artifactHash -ne $artifact.sha256) {
                        throw "$configuration/$($record.Window) hash mismatch $($outputProperty.Name)"
                    }
                }
                if (-not $statusCounts.ContainsKey([string]$summary.status)) {
                    $statusCounts[[string]$summary.status] = 0
                }
                $statusCounts[[string]$summary.status] += 1
                $validatedRuns += 1
            }
        }
    }
}

$summaries = Get-ChildItem -Path $outputRoot -Recurse -Filter "a4_summary.json"
$summaryHashes = @(
    $summaries |
        Sort-Object FullName |
        ForEach-Object {
            [ordered]@{
                relative_path = [IO.Path]::GetRelativePath(
                    (Resolve-Path -LiteralPath $outputRoot).Path,
                    $_.FullName
                )
                sha256 = (
                    Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256
                ).Hash.ToLower()
            }
        }
)
$completion = [ordered]@{
    schema_version = 1
    status = "completed"
    execution_completion_status = "completed"
    completed_at = [DateTimeOffset]::Now.ToString("o")
    expected_configurations = 18
    expected_windows = 4
    expected_summaries = 72
    observed_summaries = $summaries.Count
    validated_runs = $validatedRuns
    scientific_solution_status_counts = $statusCounts
    runner_sha256 = $expectedRunnerHash
    frozen_scales_sha256 = $actualScalesHash
    all_output_artifacts_authenticated = ($validatedRuns -eq 72)
    run_summary_hashes = $summaryHashes
}
$completion | ConvertTo-Json | Set-Content -LiteralPath (
    Join-Path $outputRoot "SWEEP_COMPLETED.json"
) -Encoding utf8
if ($summaries.Count -ne 72) {
    throw "Expected 72 A4 summaries, found $($summaries.Count)"
}
if ($validatedRuns -ne 72) {
    throw "Expected 72 authenticated A4 runs, validated $validatedRuns"
}
