# TOMMY2 — Windows 11 deployment setup
# Run in PowerShell as Administrator
# Requires: Python 3.11+ installed, CUDA 11.8 installed (for GTX 1080)
#
# Install CUDA 11.8 first from:
# https://developer.nvidia.com/cuda-11-8-0-download-archive
# Select: Windows > x86_64 > 11 > exe (local)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "=== TOMMY2 — Windows deployment setup ===" -ForegroundColor Cyan

$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

# Create and activate virtualenv
python -m venv venv
& ".\venv\Scripts\Activate.ps1"

pip install --upgrade pip wheel

# llama-cpp-python with CUDA 11.8 support (GTX 1080 = Pascal = CUDA 6.1, supported by 11.8)
# Pre-built CUDA 11.8 wheel — no Visual Studio / CMake required
pip install llama-cpp-python `
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu118

# faster-whisper: install CTranslate2 with CUDA support first
pip install ctranslate2 --extra-index-url https://pypi.ctranslate2.eu.org
pip install faster-whisper

# Remaining dependencies
pip install flask flask-sock TTS sounddevice soundfile numpy huggingface_hub

Write-Host ""
Write-Host "Dependencies installed." -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Copy the 'models/' folder from the Linux build machine, OR:"
Write-Host "     python scripts\download_models.py   (needs internet once)"
Write-Host "  2. Set LLM_N_GPU_LAYERS = 35 in backend\config.py"
Write-Host "  3. Run: .\run.bat"
