# 批量渲染两个 Courtship clip 的轨迹叠加视频
# 用法：在项目根目录下执行
#   .\scripts\analysis\render_two_courtship_videos.ps1
# 或指定路径后：
#   $env:INPUT_DIR="C:\...\data\sleap_output"; $env:PROCESSED_DIR="D:\sleap_output"; .\scripts\analysis\render_two_courtship_videos.ps1

$ErrorActionPreference = "Stop"
$ProjectRoot = if ($PSScriptRoot) { Split-Path (Split-Path $PSScriptRoot -Parent) -Parent } else { $PWD }
$Script = Join-Path $ProjectRoot "scripts\analysis\render_tracking_video.py"

$InputDir   = if ($env:INPUT_DIR)   { $env:INPUT_DIR }   else { Join-Path $ProjectRoot "data\sleap_output" }
$ProcessedDir = if ($env:PROCESSED_DIR) { $env:PROCESSED_DIR } else { "D:\sleap_output" }
$VideoDir   = if ($env:VIDEO_DIR)   { $env:VIDEO_DIR }   else { "D:\behavior_clips\20251220_E_1_2_FCMC_clips" }
$OutputDir  = if ($env:OUTPUT_DIR)  { $env:OUTPUT_DIR }  else { "D:\sleap_output" }
$Fps        = if ($env:FPS)        { $env:FPS }         else { "166" }

$Clips = @(
    "20251220_E_1_2_FCMC_male1_Courtship_1464.337-1472.705",
    "20251220_E_1_2_FCMC_male1_Courtship_2082.090-2090.072"
)

foreach ($ExpId in $Clips) {
    $OutFile = Join-Path $OutputDir "${ExpId}_tracking.mp4"
    # 视频名：20251220_E_1_2_FCMC_PLAYER1_male1_Courtship_1464.337-1472.705.mp4
    $ClipSuffix = $ExpId -replace "^20251220_E_1_2_FCMC_", ""
    $VTop   = Join-Path $VideoDir "20251220_E_1_2_FCMC_PLAYER1_$ClipSuffix.mp4"
    $VSide1 = Join-Path $VideoDir "20251220_E_1_2_FCMC_PLAYER2_$ClipSuffix.mp4"
    $VSide2 = Join-Path $VideoDir "20251220_E_1_2_FCMC_PLAYER3_$ClipSuffix.mp4"

    Write-Host "========== $ExpId ==========" -ForegroundColor Cyan
    python $Script `
        --input-dir $InputDir `
        --processed-dir $ProcessedDir `
        --experiment-id $ExpId `
        --video-top $VTop `
        --video-side1 $VSide1 `
        --video-side2 $VSide2 `
        --fps $Fps `
        --output-dir $OutputDir `
        --trail-duration-s 1 `
        --output $OutFile
}

Write-Host "Done." -ForegroundColor Green
