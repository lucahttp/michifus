# Michi-Adapter — Gemma 4 E2B + Mimi Streaming Voice Adapter

## Architecture

Full-duplex speech-to-speech pipeline:

```
Microphone → mel spectrogram (128 bins, 16kHz)
    → trained in_adapter (1024→2048) → Gemma 4 E2B QAT (LLM reasoning)
    → trained out_adapter (2048→4096) → Supertonic-3 TTS → Speaker
```

**Components:**
- `app_michi_real.py` — Working pipeline (half-duplex). Gemma + Supertonic-3.
- `app_michi_streaming.py` — Attempt at full-duplex with token streaming (incomplete).
- `adapter.py` — Model definition for Mimi↔Gemma projection layers.
- `train.py` / `train_phase2.py` / `train_phase3_sft.py` — Training scripts.
- `checkpoints/` — Trained adapter weights (Phases 1, 2, 3).

## Requirements

- Windows with Python 3.12+ (no WSL required)
- ~15GB VRAM (tested on AMD RX 6800 16GB + NVIDIA T4)
- Models downloaded from HuggingFace:
  - `unsloth/gemma-4-E2B-it-qat-mobile-GGUF` (GGUF, ~2GB)
  - `Supertonic/supertonic-3` (TTS, auto-downloaded)

## Setup

```bash
# 1. Install dependencies
pip install torch numpy sounddevice soundfile llama-cpp-python supertonic huggingface_hub

# 2. Download Gemma model (if not present)
python -c "from huggingface_hub import snapshot_download; snapshot_download('unsloth/gemma-4-E2B-it-qat-mobile-GGUF')"

# 3. Run demo
python app_michi_real.py --mode demo

# 4. Live microphone mode
python app_michi_real.py --mode mic
```

## Checkpoints

| File | Phase | Description |
|------|-------|-------------|
| `checkpoints/adapter_local_phase1.pt` | 1 | Synthetic alignment |
| `checkpoints/adapter_local_phase2.pt` | 2 | Real embedding alignment |
| `checkpoints/adapter_phase3_sft.pt` | 3 | Full SFT conversational |

## ## Full-Duplex Status

`app_michi_streaming.py` is **full-duplex**:
- Mic input flows via `sd.InputStream` callback (non-blocking)
- Audio output plays via `sd.OutputStream` callback draining a queue
- No `sd.wait()` blocks the input loop — mic and speaker work concurrently

Run:
```bash
python app_michi_streaming.py --mode demo   # synthetic test
python app_michi_streaming.py --mode mic     # live full-duplex
```

## License

MIT — use freely.
