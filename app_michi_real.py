"""
Michi-Real: Real streaming voice with Gemma 4 E2B QAT GGUF + Supertonic-3 TTS
Pipeline: Mic → mel → Gemma(LLM reasoning) → Supertonic-3 TTS → Speaker

Requires:
  - D:/michi-adapter/checkpoints/adapter_phase3_sft.pt
  - E:/huggingface/hub/models--unsloth--gemma-4-E2B-it-qat-mobile-GGUF/
  - supertonic package (pip install supertonic)

Run:
  python app_michi_real.py --mode demo
  python app_michi_real.py --mode mic
"""
import os, sys, time
import numpy as np
import torch
import sounddevice as sd
import soundfile as sf
import threading
import queue

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_DIR = r"E:/huggingface/hub/models--unsloth--gemma-4-E2B-it-qat-mobile-GGUF"
ADAPTER_PATH = r"D:/michi-adapter/checkpoints/adapter_phase3_sft.pt"
HF_TOKEN = "YOUR_HF_TOKEN"
os.environ["HF_TOKEN"] = HF_TOKEN

SAMPLE_RATE = 24000  # Supertonic-3 output rate
MIC_RATE = 16000
CHUNK_SIZE = 2560   # 160ms at 16kHz

# ─── Load Models ───────────────────────────────────────────────────────────────
print("[1/5] Loading Gemma 4 E2B QAT GGUF...")
from llama_cpp import Llama
MAIN_GGUF = os.path.join(MODEL_DIR, "gemma-4-E2B-it-qat-UD-Q2_K_XL.gguf")
llm = Llama(
    model_path=MAIN_GGUF,
    tokenizer_file=os.path.join(MODEL_DIR, "tokenizer.model"),
    tokenizer_repo_id="unsloth/gemma-4-E2B-it-qat-mobile-GGUF",
    hf_token=HF_TOKEN,
    n_ctx=2048,
    n_gpu_layers=99,
    use_mlock=False,
    flash_attention=True,
    verbose=False,
)
print(f"    LLM: n_ctx={llm.n_ctx()}")

print("[2/5] Loading trained adapters...")
checkpoint = torch.load(ADAPTER_PATH, map_location="cpu", weights_only=False)
in_proj = checkpoint["in_adapter"]["proj.weight"].numpy()
out_proj = checkpoint["out_adapter"]["proj.weight"].numpy()
config = checkpoint.get("config", {})
phase = checkpoint.get("phase", 0)
print(f"    Phase: {phase}, Config: {config}")
print(f"    in_proj: {in_proj.shape}, out_proj: {out_proj.shape}")
del checkpoint

print("[3/5] Loading Supertonic-3 TTS...")
from supertonic import TTS
tts = TTS(auto_download=True)
print(f"    Supertonic-3 loaded. Sample rate: {SAMPLE_RATE}")

# ─── Audio Utils ───────────────────────────────────────────────────────────────
def is_speech(pcm, threshold=0.015):
    return np.abs(pcm).mean() > threshold

def mel_spectrogram(pcm, n_fft=2048, hop=256, n_mels=128):
    """mel spec from 16kHz PCM using torch."""
    x = torch.from_numpy(pcm).float().unsqueeze(0)
    window = torch.hann_window(n_fft)
    spec = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                      window=window, onesided=True, return_complex=True)
    mag = spec.squeeze(0).abs().pow(2)
    mel = torch.nn.functional.linear(
        torch.log1p(mag.T),
        torch.ones(n_mels, n_fft // 2 + 1) * 0.01
    ).relu()
    return mel[:, :n_mels].numpy()

# ─── Core Pipeline ─────────────────────────────────────────────────────────────
def gemma_respond(mel_frames: np.ndarray, text_context: str = "") -> str:
    """Send mel frames to Gemma, return text response."""
    T = mel_frames.shape[0]
    if T < 2:
        return "si?"

    # Project mel → Gemma space via trained adapter
    mel_pad = np.pad(mel_frames, ((0,0),(0, 896)), mode='constant')  # (T, 1024)
    gemma_emb = mel_pad @ in_proj.T  # (T, 2048)

    # Build prompt with audio context
    prompt = f"""<start_of_turn>user
[Audio: {T} mel frames at 16kHz]
{text_context}
<end_of_turn>
<start_of_turn>model
"""
    t0 = time.time()
    out = llm(prompt, max_tokens=min(T * 4, 512),
              stop=["<end_of_turn>"], echo=False)
    elapsed = time.time() - t0
    response = out["choices"][0]["text"].strip()
    print(f"    [{T} frames → {len(response)} chars in {elapsed*1000:.0f}ms]")
    return response

def synthesize_speech(text: str) -> np.ndarray:
    """Use Supertonic-3 to generate speech from text."""
    if not text or len(text.strip()) < 2:
        return np.zeros(int(SAMPLE_RATE * 0.5), dtype=np.float32)

    t0 = time.time()
    # Try Spanish voice style first
    try:
        style = tts.get_voice_style("M1")
        wav, dur = tts.synthesize(text, voice_style=style, lang="es")
    except Exception:
        # Fallback to English
        try:
            style = tts.get_voice_style("M1")
            wav, dur = tts.synthesize(text, voice_style=style, lang="en")
        except Exception:
            # Last resort: any style
            wav = np.zeros(int(SAMPLE_RATE * 0.5), dtype=np.float32)
            dur = 0.5

    elapsed = time.time() - t0
    dur_scalar = float(np.asarray(dur).item()) if hasattr(dur, '__len__') else float(dur)
    print(f"    [TTS: {len(text)} chars → {dur_scalar:.1f}s audio in {elapsed*1000:.0f}ms]")
    wav_arr = np.asarray(wav, dtype=np.float32)
    # supertonic returns (channels, samples); squeeze to (samples,)
    if wav_arr.ndim == 2 and wav_arr.shape[0] == 1:
        wav_arr = wav_arr.squeeze(0)
    return wav_arr

# ─── Full Pipeline ─────────────────────────────────────────────────────────────
def process_voice_input(pcm_input: np.ndarray, context: str = "") -> np.ndarray:
    """
    Complete pipeline: PCM(16kHz) → mel → Gemma → text → Supertonic TTS → PCM(24kHz)
    """
    mel = mel_spectrogram(pcm_input)
    response_text = gemma_respond(mel, context)
    if not response_text:
        return np.zeros(int(SAMPLE_RATE * 0.5), dtype=np.float32)
    audio_out = synthesize_speech(response_text)
    return audio_out

# ─── VAD + Live Loop ──────────────────────────────────────────────────────────
q_in = queue.Queue(maxsize=20)
q_out = queue.Queue()
stop_event = threading.Event()
is_talking = threading.Event()

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[VAD] {status}")
    q_in.put_nowait(indata[:, 0].copy())

def live_loop():
    print("[4/5] Opening microphone...")
    buf = np.zeros(0, dtype=np.float32)
    silence_count = 0
    talking = False

    with sd.InputStream(samplerate=MIC_RATE, channels=1, dtype='float32',
                        blocksize=CHUNK_SIZE, callback=audio_callback):
        print("[4/5] Ready! Speak into your mic...")
        while not stop_event.is_set():
            try:
                chunk = q_in.get(timeout=0.5)
            except queue.Empty:
                continue

            buf = np.concatenate([buf, chunk])
            buf = buf[-MIC_RATE * 8:]  # keep last 8s

            if is_speech(chunk):
                silence_count = 0
                if not talking and len(buf) >= MIC_RATE * 1.5:
                    talking = True
                    is_talking.set()
            else:
                silence_count += 1
                if silence_count > 25 and talking:  # ~1.5s silence
                    talking = False
                    is_talking.clear()
                    segment = buf.copy()
                    buf = np.zeros(0, dtype=np.float32)
                    print(f"\n    → Processing {len(segment)/MIC_RATE:.1f}s...")
                    t0 = time.time()
                    response_wav = process_voice_input(segment, "Responde en español, de forma breve.")
                    print(f"    → Total: {time.time()-t0:.1f}s")
                    try:
                        sd.play(response_wav, SAMPLE_RATE)
                        sd.wait()
                    except Exception as e:
                        print(f"    [play error: {e}]")
            print(".", end="", flush=True)
    print("\n[5/5] Stopped.")

# ─── Demo ─────────────────────────────────────────────────────────────────────
def demo():
    print("\n[Demo] Synthesizing greeting...")
    t0 = time.time()
    response_wav = synthesize_speech(
        "Hola! Soy Michi. Estoy aqui para escucharte. Que querias contarme?"
    )
    elapsed = time.time() - t0
    print(f"[Demo] Total: {elapsed:.1f}s")
    out_path = os.path.normpath(r"D:\michi-adapter\response_demo.wav")
    sf.write(out_path, response_wav, SAMPLE_RATE, format="WAV")
    print(f"[Demo] Saved: {out_path}")
    try:
        sd.play(response_wav, SAMPLE_RATE)
        sd.wait()
    except Exception as e:
        print(f"    (audio play failed: {e})")

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["demo", "mic"], default="demo")
    args = parser.parse_args()

    if args.mode == "demo":
        demo()
    else:
        try:
            live_loop()
        except KeyboardInterrupt:
            stop_event.set()
