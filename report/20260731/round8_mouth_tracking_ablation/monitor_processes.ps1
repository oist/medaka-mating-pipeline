param(
    [Parameter(Mandatory = $true)]
    [int[]]$ProcessId,
    [Parameter(Mandatory = $true)]
    [string]$CsvPath,
    [Parameter(Mandatory = $true)]
    [string]$SummaryPath,
    [double]$StartedUnixSeconds,
    [double]$IntervalSeconds = 1.0,
    [switch]$DiscoverDescendants
)

$ErrorActionPreference = "Stop"
$samples = [System.Collections.Generic.List[object]]::new()

do {
    $trackedIds = [System.Collections.Generic.HashSet[int]]::new()
    foreach ($rootId in $ProcessId) {
        [void]$trackedIds.Add([int]$rootId)
    }
    if ($DiscoverDescendants) {
        $allProcessRows = @(
            Get-CimInstance Win32_Process |
                Select-Object ProcessId, ParentProcessId
        )
        do {
            $added = 0
            foreach ($row in $allProcessRows) {
                if (
                    $trackedIds.Contains([int]$row.ParentProcessId) -and
                    -not $trackedIds.Contains([int]$row.ProcessId)
                ) {
                    [void]$trackedIds.Add([int]$row.ProcessId)
                    $added += 1
                }
            }
        } while ($added -gt 0)
    }
    $processes = @(
        Get-Process -Id @($trackedIds) -ErrorAction SilentlyContinue
    )
    if ($processes.Count -gt 0) {
        $sample = [pscustomobject]@{
            sampled_at = [DateTimeOffset]::Now.ToString("o")
            process_ids = (($processes.Id | Sort-Object) -join "|")
            working_set_bytes = [int64](($processes | Measure-Object WorkingSet64 -Sum).Sum)
            peak_working_set_bytes = [int64](($processes | Measure-Object PeakWorkingSet64 -Sum).Sum)
            cpu_seconds = [double](($processes | Measure-Object CPU -Sum).Sum)
        }
        $samples.Add($sample)
        $sample | Export-Csv -NoTypeInformation -Append -Path $CsvPath
        Start-Sleep -Milliseconds ([math]::Max(100, [int]($IntervalSeconds * 1000)))
    }
} while ($processes.Count -gt 0)

$finished = [DateTimeOffset]::Now
$started = if ($StartedUnixSeconds -gt 0) {
    [DateTimeOffset]::FromUnixTimeMilliseconds(
        [int64]([math]::Round($StartedUnixSeconds * 1000))
    )
} elseif ($samples.Count -gt 0) {
    [DateTimeOffset]::Parse($samples[0].sampled_at)
} else {
    $finished
}

$summary = [ordered]@{
    schema_version = 1
    process_ids = $ProcessId
    sampling_interval_seconds = $IntervalSeconds
    monitor_started_at = if ($samples.Count) { $samples[0].sampled_at } else { $null }
    process_started_at = $started.ToString("o")
    finished_at = $finished.ToString("o")
    wall_seconds = ($finished - $started).TotalSeconds
    samples = $samples.Count
    max_sampled_working_set_bytes = if ($samples.Count) {
        [int64](($samples | Measure-Object working_set_bytes -Maximum).Maximum)
    } else {
        $null
    }
    max_reported_peak_working_set_bytes = if ($samples.Count) {
        [int64](($samples | Measure-Object peak_working_set_bytes -Maximum).Maximum)
    } else {
        $null
    }
    peak_memory_method = if ($DiscoverDescendants) {
        "$IntervalSeconds s Get-Process sampling; sums root process IDs and recursively discovered descendants"
    } else {
        "$IntervalSeconds s Get-Process sampling; sums all explicitly supplied wrapper and worker process IDs"
    }
}
$summary | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 -Path $SummaryPath
