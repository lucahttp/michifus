#!/bin/bash
# Install on H100 - PyTorch already there, just need transformers/accelerate/huggingface
set -e
PY=/system/conda/miniconda3/bin/python

$PY -m pip install --quiet --no-cache-dir \
  "transformers>=4.55,<5" \
  "accelerate>=1.0" \
  "huggingface_hub>=0.25" \
  "safetensors" \
  "pyyaml"

echo "=== versions ==="
$PY -c "
import transformers, accelerate, huggingface_hub, safetensors
print('transformers', transformers.__version__)
print('accelerate', accelerate.__version__)
print('huggingface_hub', huggingface_hub.__version__)
print('safetensors', safetensors.__version__)
import torch
print('torch', torch.__version__)
print('cuda', torch.cuda.is_available())
print('device', torch.cuda.get_device_name(0))
print('vram_total_gb', torch.cuda.get_device_properties(0).total_memory/1e9)
"
