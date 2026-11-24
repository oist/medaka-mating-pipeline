# Start / stop / inspect the SLEAP-tracking + medaka-3d queues for configs/batch_queue.txt.
#   powershell -ExecutionPolicy Bypass -File .\scripts\sleap\queue_ctl.ps1 -Action start     # runs as Windows scheduled tasks: survives closing VSCode
#   .\scripts\sleap\queue_ctl.ps1 -Action stop      # stops both queues and their child processes
#   .\scripts\sleap\queue_ctl.ps1 -Action status
# Two independent tasks (see queue_run.ps1): medaka_sleap_queue (local GPU, single queue) and
# medaka_m3d_queue (medaka-3d, waits for the three .slp of each experiment).
param([ValidateSet('start', 'stop', 'status')][string]$Action = 'status')

$ld    = '\\bucket.oist.jp\bucket\StephensU\medaka\runs\batch_pipeline'
$tasks = [ordered]@{ sleap = 'medaka_sleap_queue'; m3d = 'medaka_m3d_queue' }
$match = @{ sleap = 'batch_bottomup_track|sleap_nn'; m3d = 'medaka3d' }

function Stop-Queue([string]$Stage) {
    Stop-ScheduledTask -TaskName $tasks[$Stage] -ErrorAction SilentlyContinue
    Get-CimInstance Win32_Process |
        Where-Object { $_.Name -match 'python|pwsh' -and $_.CommandLine -match $match[$Stage] -and $_.CommandLine -notmatch 'queue_ctl' } |
        ForEach-Object { taskkill /PID $_.ProcessId /T /F 2>$null | Out-Null }
}

foreach ($stage in $tasks.Keys) {
    switch ($Action) {
        'start' {
            $act = New-ScheduledTaskAction -Execute "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" `
                -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$PSScriptRoot\queue_run.ps1`" -Stage $stage"
            $set = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries `
                -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew
            $pri = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive
            Register-ScheduledTask -TaskName $tasks[$stage] -Action $act -Settings $set -Principal $pri -Force | Out-Null
            Start-ScheduledTask -TaskName $tasks[$stage]
            "$($tasks[$stage]) started"
        }
        'stop' {
            Stop-Queue $stage
            Unregister-ScheduledTask -TaskName $tasks[$stage] -Confirm:$false -ErrorAction SilentlyContinue
            "$($tasks[$stage]) stopped"
        }
        'status' {
            $t = Get-ScheduledTask -TaskName $tasks[$stage] -ErrorAction SilentlyContinue
            "{0}: {1}" -f $tasks[$stage], $(if ($t) { $t.State } else { 'not registered' })
            Get-Content "$ld\queue_$stage.out.log" -Tail 3 -ErrorAction SilentlyContinue
        }
    }
}
