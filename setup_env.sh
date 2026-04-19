#!/usr/bin/env bash
# Create and fully configure the diffpose-video conda environment.
# Run once: bash setup_env.sh
set -e

ENV_NAME="diffpose-video"

echo "==> Creating conda environment: $ENV_NAME (Python 3.11)"
conda create -n "$ENV_NAME" python=3.11 pip -y 2>/dev/null || true

echo "==> Installing PyTorch (CUDA 12.8) ..."
conda run -n "$ENV_NAME" pip install \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

echo "==> Installing onnxruntime-gpu ..."
conda run -n "$ENV_NAME" pip install "onnxruntime-gpu==1.20.1"

echo "==> Installing diffpose-video (editable) ..."
conda run -n "$ENV_NAME" pip install -e "$(dirname "$0")"

# rtmlib pulls in the CPU-only onnxruntime which overwrites onnxruntime-gpu.
# Reinstall the GPU version last to ensure CUDAExecutionProvider is available.
echo "==> Reinstalling onnxruntime-gpu (override CPU version pulled by rtmlib) ..."
conda run -n "$ENV_NAME" pip install --force-reinstall "onnxruntime-gpu==1.20.1"

echo "==> Downloading pretrained checkpoints ..."
conda run -n "$ENV_NAME" diffpose-download

echo ""
echo "Done!  Activate with:  conda activate $ENV_NAME"
