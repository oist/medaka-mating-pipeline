# Download, extract, and add FFmpeg to PATH for the current user.
# Run in PowerShell: .\scripts\install-ffmpeg.ps1
# If execution policy blocks the script, run:
#   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

$ErrorActionPreference = "Stop"
$FFMPEG_VERSION = "8.0.1"
$URL = "https://github.com/GyanD/codexffmpeg/releases/download/$FFMPEG_VERSION/ffmpeg-$FFMPEG_VERSION-full_build.zip"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ToolsDir = Join-Path $ProjectRoot "tools"
$ZipPath = Join-Path $ToolsDir "ffmpeg.zip"
$ExtractDir = Join-Path $ToolsDir "ffmpeg"
$BinName = "ffmpeg-$FFMPEG_VERSION-full_build"

Write-Host "FFmpeg installer (Gyan build $FFMPEG_VERSION)" -ForegroundColor Cyan
Write-Host "Project root: $ProjectRoot"
Write-Host ""

# 1. Create target directories.
New-Item -ItemType Directory -Force -Path $ToolsDir | Out-Null

# 2. Download the zip if needed.
if (-not (Test-Path $ZipPath) -or (Get-Item $ZipPath).Length -lt 10MB) {
    Write-Host "Downloading FFmpeg ..." -ForegroundColor Yellow
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    try {
        Invoke-WebRequest -Uri $URL -OutFile $ZipPath -UseBasicParsing
    } catch {
        Write-Host "Download failed. If network access is restricted, download it manually:" -ForegroundColor Red
        Write-Host "  $URL"
        Write-Host "  Save the zip as: $ZipPath"
        exit 1
    }
    Write-Host "Download complete." -ForegroundColor Green
} else {
    Write-Host "Zip file already exists; skipping download." -ForegroundColor Green
}

# 3. Extract the archive.
$ExpectedBin = Join-Path $ExtractDir "$BinName\bin\ffmpeg.exe"
if (-not (Test-Path $ExpectedBin)) {
    Write-Host "Extracting archive ..." -ForegroundColor Yellow
    Remove-Item -Path $ExtractDir -Recurse -Force -ErrorAction SilentlyContinue
    Expand-Archive -Path $ZipPath -DestinationPath $ExtractDir -Force
    Write-Host "Extraction complete." -ForegroundColor Green
} else {
    Write-Host "Extraction directory already exists; skipping extraction." -ForegroundColor Green
}

$BinDir = (Resolve-Path (Join-Path $ExtractDir "$BinName\bin")).Path
if (-not (Test-Path (Join-Path $BinDir "ffmpeg.exe"))) {
    Write-Host "Error: ffmpeg.exe was not found. Check the zip contents." -ForegroundColor Red
    exit 1
}

# 4. Add the binary directory to the current user's PATH.
$UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($UserPath -notlike "*$BinDir*") {
    [Environment]::SetEnvironmentVariable("Path", "$UserPath;$BinDir", "User")
    Write-Host "Added this directory to the user PATH: $BinDir" -ForegroundColor Green
} else {
    Write-Host "The user PATH already contains the FFmpeg directory." -ForegroundColor Green
}

# 5. Add it to the current shell session as well.
$env:Path = "$env:Path;$BinDir"

# 6. Verify the installation.
$ver = & (Join-Path $BinDir "ffmpeg.exe") -version 2>&1 | Select-Object -First 1
Write-Host ""
Write-Host "Installation complete. Version: $ver" -ForegroundColor Green
Write-Host "Reopen the terminal, or run 'ffmpeg -version' in this session now." -ForegroundColor Cyan
