# Run one stage of the queue over configs/batch_queue.txt (launched by queue_ctl.ps1 as a scheduled task).
param([Parameter(Mandatory)][ValidateSet('sleap', 'm3d')][string]$Stage)

$repo = (Resolve-Path "$PSScriptRoot\..\..").Path
$py   = "$repo\.venv\Scripts\python.exe"
$cfg  = "$repo\configs\paths.ReiterU.toml"
$ld   = '\\bucket.oist.jp\bucket\StephensU\medaka\runs\batch_pipeline'
$tag  = 's14_bottomup_3node_reviewed'
$top  = 'retrain_20260810_3node_reviewed/top/s14.bottomup_3node.reviewed_top_f24_ms64_paf8'
$side = 'retrain_20260810_3node_reviewed/side/s14.bottomup_3node.reviewed_side_f24_ms64_paf8'
$exps = Get-Content "$repo\configs\batch_queue.txt" | Where-Object { $_ -and $_ -notmatch '^\s*#' } | ForEach-Object { $_.Trim() }

$cmd = switch ($Stage) {
    'sleap' { @('-u', '-m', 'scripts.sleap.batch_bottomup_track') + $exps +
              @('--paths-config', $cfg, '--round-tag', $tag, '--top-model', $top, '--side-model', $side, '--batch-size', '4') }
    'm3d'   { @('-u', '-m', 'scripts.reconstruction_3d.medaka3d_batch') + $exps +
              @('--round-tag', $tag, '--jobs', '1', '--watch', '10') }
}
Set-Location $repo
"# $(Get-Date) start $Stage ($($exps.Count) experiments)" | Add-Content "$ld\queue_$Stage.log"
& $py @cmd 1>> "$ld\queue_$Stage.out.log" 2>> "$ld\queue_$Stage.err.log"
"# $(Get-Date) exit $LASTEXITCODE" | Add-Content "$ld\queue_$Stage.log"
