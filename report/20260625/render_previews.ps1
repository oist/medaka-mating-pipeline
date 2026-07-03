# Render each slide of the 2026-06-25 decks to PNG via PowerPoint COM (visual QC).
# python-pptx cannot render; PowerPoint COM can. Re-runnable.
#   pwsh -File report\20260625\render_previews.ps1
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

$jobs = @(
    @{ pptx = "medaka_progress_report_20260625.pptx";    out = "slide_previews" },
    @{ pptx = "medaka_progress_report_20260625_JA.pptx"; out = "slide_previews_ja" }
)

$pp = New-Object -ComObject PowerPoint.Application
try {
    foreach ($j in $jobs) {
        $pptxPath = Join-Path $here $j.pptx
        $outDir   = Join-Path $here $j.out
        if (-not (Test-Path $pptxPath)) { Write-Host "[MISS] $($j.pptx)"; continue }
        New-Item -ItemType Directory -Force -Path $outDir | Out-Null
        Get-ChildItem -Path $outDir -Filter *.png -ErrorAction SilentlyContinue | Remove-Item -Force
        $pres = $pp.Presentations.Open($pptxPath, $true, $false, $false)  # ReadOnly, !Untitled, !WithWindow
        $n = $pres.Slides.Count
        for ($i = 1; $i -le $n; $i++) {
            $png = Join-Path $outDir ("slide{0:D2}.png" -f $i)
            $pres.Slides.Item($i).Export($png, "PNG", 1280, 720)
        }
        $pres.Close()
        Write-Host "[ok ] $($j.pptx) -> $($j.out)\  ($n slides)"
    }
}
finally {
    $pp.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($pp) | Out-Null
}
