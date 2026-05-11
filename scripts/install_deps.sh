#!/usr/bin/env bash
# Linux development machine setup (Ubuntu 22.04 / 24.04)
set -e

echo "=== TOMMY2 — Linux dev setup ==="

# System packages
sudo apt update
sudo apt install -y \
    portaudio19-dev \
    libsndfile1 \
    ffmpeg \
    build-essential \
    python3-venv \
    python3-dev

# Create virtualenv
cd "$(dirname "$0")/.."
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip wheel

# CPU-only llama-cpp-python (works immediately, no CUDA driver needed)
pip install llama-cpp-python

# All other deps
pip install -r requirements.txt

echo ""
echo "Done. To enable GPU acceleration after installing the NVIDIA driver:"
echo "  CMAKE_ARGS=\"-DLLAMA_CUDA=on\" pip install llama-cpp-python --force-reinstall"
echo "  Then set LLM_N_GPU_LAYERS = 35 in backend/config.py"
echo ""
echo "Next step: source venv/bin/activate && python scripts/download_models.py"
