#!/bin/bash
# Install Phase 2 dependencies on Lightning AI Studio (H100 / A100)
set -e

PY=/system/conda/miniconda3/bin/python

echo "=== Installing Phase 2 dependencies ==="
$PY -m pip install --quiet --no-cache-dir \
  "transformers>=4.55,<5" \
  "datasets>=2.20" \
  "accelerate>=1.0" \
  "huggingface_hub>=0.25" \
  "safetensors" \
  "pyyaml" \
  "torchaudio" \
  "peft"

echo "=== Phase 2 environment sanity check ==="
$PY -c "
import torch, transformers, datasets, accelerate, huggingface_hub
print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0), 'VRAM:', torch.cuda.mem_get_info()[1]/1e9, 'GB')
print('transformers:', transformers.__version__)
print('datasets:', datasets.__version__)
print('accelerate:', accelerate.__version__)
"
