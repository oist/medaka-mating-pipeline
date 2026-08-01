$ErrorActionPreference = "Stop"
$reportRoot = "report\20260731\round8_mouth_tracking_ablation"
$outputRoot = Join-Path $reportRoot "training_a4_cutoff_sweep"
$launcher = Join-Path $reportRoot "run_training_a4_cutoff_sweep.ps1"
$expectedRunnerHash = "1e75e10654b5d6c233ec3ea5ff2f8c969dac3dd41027031cd332d3c88096d4e0"
$expectedScalesHash = "63bf6de26bb85e76da97976ffa0158c8d4e27d7ea9e3208b627fe8ade2eb9209"
$executionLauncherHash = "0d87de5894af9335721c1f1b672ca1921160a7128694c2d526a2562da3a89b4f"
$postvalidationLauncherHash = (
    Get-FileHash -LiteralPath $launcher -Algorithm SHA256
).Hash.ToLower()
if (
    $postvalidationLauncherHash -ne
    "bc534144fa28557bec99b9b581714accdcd2de8317faf65f366d6d30dac1c9ee"
) {
    throw "Unexpected post-validation launcher hash: $postvalidationLauncherHash"
}

$windowFrames = [ordered]@{
    reviewed_event_1 = @(72371, 73027)
    reviewed_event_3 = @(174316, 180515)
    control_for_event_1 = @(32264, 32920)
    control_for_event_3 = @(21840, 28039)
}
$configurations = @(
    foreach ($separation in @("s035", "s050", "s065")) {
        foreach ($margin in @("m005", "m010")) {
            foreach ($gap in @("g0", "g2", "g5")) {
                "${separation}_${margin}_${gap}"
            }
        }
    }
)
$requiredOutputs = @(
    "admissible_hypotheses", "group_observations", "groups", "paths",
    "links", "dispositions", "selected", "direct", "filled",
    "top_cam_tracklets", "top_cam_dispositions", "top_cam_bridges",
    "top_cam_summary", "side1_cam_tracklets", "side1_cam_dispositions",
    "side1_cam_bridges", "side1_cam_summary", "side2_cam_tracklets",
    "side2_cam_dispositions", "side2_cam_bridges", "side2_cam_summary"
)

$resolvedRoot = (Resolve-Path -LiteralPath $outputRoot).Path
$summaryRecords = @()
$statusCounts = @{}
$validatedArtifacts = 0
foreach ($window in $windowFrames.Keys) {
    $start = [int]$windowFrames[$window][0]
    $end = [int]$windowFrames[$window][1]
    $frames = $end - $start + 1
    foreach ($configuration in $configurations) {
        $runDir = Join-Path (Join-Path $outputRoot $window) $configuration
        $summaryPath = Join-Path $runDir "a4_summary.json"
        if (-not (Test-Path -LiteralPath $summaryPath)) {
            throw "Missing summary: $summaryPath"
        }
        $summary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
        if ($summary.configuration.frame_start -ne $start) {
            throw "Frame start mismatch: $summaryPath"
        }
        if ($summary.configuration.frame_end -ne $end) {
            throw "Frame end mismatch: $summaryPath"
        }
        if (
            $summary.configuration.frozen_body_scales_sha256 -ne
            $expectedScalesHash
        ) {
            throw "Scale mismatch: $summaryPath"
        }
        if (
            $summary.code_sha256.'scripts\reconstruction_3d\mouth_tracklet_ablation.py' -ne
            $expectedRunnerHash
        ) {
            throw "Runner mismatch: $summaryPath"
        }
        if ($configuration -notmatch '^s(\d{3})_m(\d{3})_g(\d+)$') {
            throw "Malformed configuration name: $configuration"
        }
        $expectedSeparation = [int]$Matches[1] / 100.0
        $expectedMargin = [int]$Matches[2] / 100.0
        $expectedGap = [int]$Matches[3]
        $trackletConfig = $summary.configuration.tracklet_config
        if (
            [double]$trackletConfig.separation_body_lengths -ne
            $expectedSeparation
        ) {
            throw "Separation mismatch: $summaryPath"
        }
        if (
            [double]$trackletConfig.assignment_ambiguity_margin_body_lengths -ne
            $expectedMargin
        ) {
            throw "Ambiguity mismatch: $summaryPath"
        }
        if (
            [int]$trackletConfig.maximum_missing_gap_frames -ne
            $expectedGap
        ) {
            throw "Gap mismatch: $summaryPath"
        }
        foreach ($name in $requiredOutputs) {
            if ($name -notin $summary.outputs.psobject.Properties.Name) {
                throw "Missing manifest entry $name in $summaryPath"
            }
        }
        $expectedBlocks = [math]::Ceiling($frames / 250.0)
        $blockEntries = @(
            $summary.outputs.psobject.Properties |
                Where-Object Name -Like "admissible_f*"
        )
        if ($blockEntries.Count -ne 2 * $expectedBlocks) {
            throw "Hypothesis block manifest mismatch: $summaryPath"
        }
        if (
            @($summary.outputs.psobject.Properties).Count -ne
            $requiredOutputs.Count + 2 * $expectedBlocks
        ) {
            throw "Unexpected manifest entry count: $summaryPath"
        }
        foreach ($property in $summary.outputs.psobject.Properties) {
            $artifact = $property.Value
            $resolvedArtifact = (Resolve-Path -LiteralPath $artifact.path).Path
            if (
                -not $resolvedArtifact.StartsWith(
                    $resolvedRoot,
                    [System.StringComparison]::OrdinalIgnoreCase
                )
            ) {
                throw "Artifact escapes sweep root: $resolvedArtifact"
            }
            $actualHash = (
                Get-FileHash -LiteralPath $resolvedArtifact -Algorithm SHA256
            ).Hash.ToLower()
            if ($actualHash -ne $artifact.sha256) {
                throw "Artifact hash mismatch $($property.Name): $summaryPath"
            }
            $validatedArtifacts += 1
        }
        $direct = Import-Csv -LiteralPath $summary.outputs.direct.path
        $filled = Import-Csv -LiteralPath $summary.outputs.filled.path
        $expectedRows = 3 * $frames
        if ($direct.Count -ne $expectedRows -or $filled.Count -ne $expectedRows) {
            throw "Incomplete 3xframe grid: $summaryPath"
        }
        $directKeys = @(
            $direct | ForEach-Object { "$($_.frame):$($_.fish_id)" }
        )
        if (($directKeys | Sort-Object -Unique).Count -ne $expectedRows) {
            throw "Duplicate direct frame/identity keys: $summaryPath"
        }
        if ($summary.status -eq "insufficient_three_path_solution") {
            if (@($direct | Where-Object { $_.X -ne "" }).Count -ne 0) {
                throw "Scientific abstention exposes finite direct rows: $summaryPath"
            }
            if (
                @($direct.identity_status | Sort-Object -Unique) -ne
                "unassigned_tracklet_path"
            ) {
                throw "Scientific abstention status mismatch: $summaryPath"
            }
        } elseif ($summary.status -ne "completed") {
            throw "Unknown scientific solution status: $($summary.status)"
        }
        if (-not $statusCounts.ContainsKey([string]$summary.status)) {
            $statusCounts[[string]$summary.status] = 0
        }
        $statusCounts[[string]$summary.status] += 1
        $relativePath = [IO.Path]::GetRelativePath(
            $resolvedRoot,
            (Resolve-Path -LiteralPath $summaryPath).Path
        )
        $summaryRecords += [ordered]@{
            relative_path = $relativePath
            sha256 = (
                Get-FileHash -LiteralPath $summaryPath -Algorithm SHA256
            ).Hash.ToLower()
            execution_status = "completed_and_authenticated"
            scientific_solution_status = [string]$summary.status
            configuration_hash = [string]$summary.configuration_hash
        }
    }
}
if ($summaryRecords.Count -ne 72) {
    throw "Expected 72 validated summaries, got $($summaryRecords.Count)"
}
$completion = [ordered]@{
    schema_version = 1
    status = "completed"
    execution_completion_status = "completed"
    completion_validated_at = [DateTimeOffset]::Now.ToString("o")
    expected_configurations = 18
    expected_windows = 4
    expected_summaries = 72
    observed_summaries = 72
    validated_runs = 72
    validated_output_artifacts = $validatedArtifacts
    scientific_solution_status_counts = $statusCounts
    runner_sha256 = $expectedRunnerHash
    frozen_scales_sha256 = $expectedScalesHash
    all_output_artifacts_authenticated = $true
    execution_launcher_sha256 = $executionLauncherHash
    postvalidation_launcher_sha256 = $postvalidationLauncherHash
    launcher_drift_reason = "Only the completion-footer schema and post-validation fields changed during execution; scientific runner, frozen scales, grid, and run commands did not drift."
    postvalidation_script_sha256 = (
        Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256
    ).Hash.ToLower()
    run_summary_hashes = @(
        $summaryRecords | Sort-Object { $_["relative_path"] }
    )
}
$marker = Join-Path $outputRoot "SWEEP_COMPLETED.json"
$temporaryMarker = Join-Path $outputRoot ".SWEEP_COMPLETED.postvalidation.tmp"
$completion |
    ConvertTo-Json -Depth 8 |
    Set-Content -LiteralPath $temporaryMarker -Encoding utf8
Move-Item -LiteralPath $temporaryMarker -Destination $marker -Force
$completion | ConvertTo-Json -Depth 4
