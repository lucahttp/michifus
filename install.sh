#!/bin/bash
# Install deps for adapter training on Lightning Studio (H100/A100/T4)
set -e
echo "[install] starting $(date)"
pip install --quiet --upgrade pip
pip install --quiet torch torchvision torchaudio
pip install --quiet transformers accelerate datasets sentencepiece
pip install --quiet huggingface_hub
echo "[install] done $(date)"
