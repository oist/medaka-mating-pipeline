# Render the catalogue deck + draft poster to PNG via PowerPoint COM (QC + browsing).
# Usage:  powershell -ExecutionPolicy Bypass -File report\20260630\render_previews.ps1
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

$jobs = @(
    @{ pptx = Join-Path $here "medaka_poster_figures_20260630.pptx"; out = Join-Path $here "slide_previews";        w = 1280; h = 720 },
    @{ pptx = Join-Path $here "draft_poster_A0_landscape_20260630.pptx"; out = Join-Path $here "poster_preview"; w = 1900; h = 1344 }
)

$pp = New-Object -ComObject PowerPoint.Application
try {
    foreach ($j in $jobs) {
        if (-not (Test-Path $j.pptx)) { Write-Host "MISS $($j.pptx)"; continue }
        if (-not (Test-Path $j.out)) { New-Item -ItemType Directory -Path $j.out | Out-Null }
        $pres = $pp.Presentations.Open($j.pptx, $true, $false, $false)  # ReadOnly, untitled, no window
        $n = $pres.Slides.Count
        for ($i = 1; $i -le $n; $i++) {
            $png = Join-Path $j.out ("slide_{0:D2}.png" -f $i)
            $pres.Slides.Item($i).Export($png, "PNG", $j.w, $j.h)
        }
        $pres.Close()
        Write-Host "Rendered $n slides -> $($j.out)"
    }
}
finally {
    $pp.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($pp) | Out-Null
}
