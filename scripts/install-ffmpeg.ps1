# FFmpeg 自动下载、解压并配置 PATH（仅当前用户）
# 在 PowerShell 中运行: .\scripts\install-ffmpeg.ps1
# 若执行策略限制，可先运行: Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

$ErrorActionPreference = "Stop"
$FFMPEG_VERSION = "8.0.1"
$URL = "https://github.com/GyanD/codexffmpeg/releases/download/$FFMPEG_VERSION/ffmpeg-$FFMPEG_VERSION-full_build.zip"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ToolsDir = Join-Path $ProjectRoot "tools"
$ZipPath = Join-Path $ToolsDir "ffmpeg.zip"
$ExtractDir = Join-Path $ToolsDir "ffmpeg"
$BinName = "ffmpeg-$FFMPEG_VERSION-full_build"

Write-Host "FFmpeg 安装脚本 (Gyan build $FFMPEG_VERSION)" -ForegroundColor Cyan
Write-Host "项目目录: $ProjectRoot"
Write-Host ""

# 1. 创建目录
New-Item -ItemType Directory -Force -Path $ToolsDir | Out-Null

# 2. 下载
if (-not (Test-Path $ZipPath) -or (Get-Item $ZipPath).Length -lt 10MB) {
    Write-Host "正在下载 FFmpeg ..." -ForegroundColor Yellow
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    try {
        Invoke-WebRequest -Uri $URL -OutFile $ZipPath -UseBasicParsing
    } catch {
        Write-Host "下载失败。若网络受限，请手动下载:" -ForegroundColor Red
        Write-Host "  $URL"
        Write-Host "  将 zip 保存为: $ZipPath"
        exit 1
    }
    Write-Host "下载完成." -ForegroundColor Green
} else {
    Write-Host "已存在 zip，跳过下载." -ForegroundColor Green
}

# 3. 解压
$ExpectedBin = Join-Path $ExtractDir "$BinName\bin\ffmpeg.exe"
if (-not (Test-Path $ExpectedBin)) {
    Write-Host "正在解压 ..." -ForegroundColor Yellow
    Remove-Item -Path $ExtractDir -Recurse -Force -ErrorAction SilentlyContinue
    Expand-Archive -Path $ZipPath -DestinationPath $ExtractDir -Force
    # zip 内是一层目录 ffmpeg-8.0.1-full_build，保留该结构
    Write-Host "解压完成." -ForegroundColor Green
} else {
    Write-Host "已存在解压目录，跳过解压." -ForegroundColor Green
}

$BinDir = (Resolve-Path (Join-Path $ExtractDir "$BinName\bin")).Path
if (-not (Test-Path (Join-Path $BinDir "ffmpeg.exe"))) {
    Write-Host "错误: 未找到 ffmpeg.exe，请检查 zip 结构。" -ForegroundColor Red
    exit 1
}

# 4. 写入当前用户 PATH
$UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($UserPath -notlike "*$BinDir*") {
    [Environment]::SetEnvironmentVariable("Path", "$UserPath;$BinDir", "User")
    Write-Host "已将该目录加入用户 PATH: $BinDir" -ForegroundColor Green
} else {
    Write-Host "PATH 中已包含 FFmpeg 目录，未修改。" -ForegroundColor Green
}

# 5. 当前会话 PATH 也加上，便于立即使用
$env:Path = "$env:Path;$BinDir"

# 6. 验证
$ver = & (Join-Path $BinDir "ffmpeg.exe") -version 2>&1 | Select-Object -First 1
Write-Host ""
Write-Host "安装完成。版本信息: $ver" -ForegroundColor Green
Write-Host "请重新打开终端，或在本终端直接运行: ffmpeg -version" -ForegroundColor Cyan
