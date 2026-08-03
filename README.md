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

## ## Michi-Lite (Apple Silicon, lightweight)

`app_michi_lite.py` is the **working** full-duplex path on M1/M2/M3 — no
trained Gemma-adapter checkpoints, no GGUF, no llama.cpp. Runs entirely on
MLX + MPS.

Pipeline:
```
Mic 16kHz → Silero VAD → mlx-whisper-tiny (STT)
  → mlx-lm Qwen2.5-0.5B-Instruct-4bit (LLM, ES)
  → mlx-audio Kokoro-82M-bf16 (TTS) → Speaker 24kHz
```

Full-duplex design:
- `sd.InputStream` callback always non-blocking
- `sd.OutputStream` callback drains a play queue
- Speaker-active flag raises VAD threshold ×3 to suppress echo bleed
- Barge-in: new speech while TTS playing flushes the play queue
- All config via env vars (no hardcoded `D:/` or `E:/` paths)

Setup + run:
```bash
pip install sounddevice soundfile mlx-whisper mlx-lm mlx-audio silero-vad scipy
python app_michi_lite.py --mode demo   # one-shot synth, no mic
python app_michi_lite.py --mode mic    # live full-duplex mic+speaker
```

Env vars (override defaults):
| Var | Default |
|---|---|
| `MICHI_STT` | `mlx-community/whisper-tiny-mlx` |
| `MICHI_LLM` | `mlx-community/Qwen2.5-0.5B-Instruct-4bit` |
| `MICHI_TTS` | `mlx-community/Kokoro-82M-bf16` |
| `MICHI_VOICE` | `af_heart` |
| `MICHI_TTS_LANG` | `es` |
| `MICHI_SYSTEM` | short Spanish persona |
| `MICHI_MAX_TOKENS` | `80` |
| `MICHI_FULL_DUPLEX` | `1` |
| `MICHI_ECHO_GAIN` | `3.0` |

This is the path that runs on a stock MacBook Air (M1, 8GB). The original
Gemma-4-E2B + Mimi adapter pipeline (`app_michi_streaming.py`) stays as the
high-end / heavy-compute path — train the adapter checkpoints first, then
deploy on a 16GB-VRAM box.

## License

MIT — use freely.
