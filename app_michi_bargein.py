"""
Michi-BargeIn: Full-duplex voice with Silero VAD barge-in
When user speaks while Michi is talking, Silero VAD detects it and interrupts.

Pipeline:
  Mic → Silero VAD (real-time) → Gemma (streaming) → Supertonic TTS → Speaker
  If VAD detects user speech during playback → interrupt + clear queues
"""
import os, sys, time, threading, queue
import numpy as np
import torch
import sounddevice as sd
import soundfile as sf

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_DIR = r"E:/huggingface/hub/models--unsloth--gemma-4-E2B-it-qat-mobile-GGUF"
ADAPTER_PATH = r"D:/michi-adapter/checkpoints/adapter_phase3_sft.pt"
HF_TOKEN = "YOUR_HF_TOKEN"
os.environ["HF_TOKEN"] = HF_TOKEN

SAMPLE_RATE = 24000
MIC_RATE = 16000

# ─── Load Models ───────────────────────────────────────────────────────────────
print("[1/7] Loading Silero VAD...")
torch.set_num_threads(4)
torch.hub.set_dir(r"D:/michi-adapter/models")
vad_model, vad_utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
get_speech_timestamps = vad_utils[3]  # function to get speech timestamps
print("    Silero VAD loaded")

print("[2/7] Loading Gemma 4 E2B GGUF...")
from llama_cpp import Llama
llm = Llama(
    model_path=os.path.join(MODEL_DIR, "gemma-4-E2B-it-qat-UD-Q2_K_XL.gguf"),
    tokenizer_file=os.path.join(MODEL_DIR, "tokenizer.model"),
    tokenizer_repo_id="unsloth/gemma-4-E2B-it-qat-mobile-GGUF",
    hf_token=HF_TOKEN,
    n_ctx=1024,
    n_gpu_layers=99,
    flash_attention=True,
    verbose=False,
)
print(f"    Gemma ready. n_ctx={llm.n_ctx()}")

print("[3/7] Loading adapters...")
ckpt = torch.load(ADAPTER_PATH, map_location="cpu", weights_only=False)
in_proj = ckpt["in_adapter"]["proj.weight"].numpy()
out_proj = ckpt["out_adapter"]["proj.weight"].numpy()
del ckpt
print(f"    in={in_proj.shape} out={out_proj.shape}")

print("[4/7] Loading Supertonic-3 TTS...")
from supertonic import TTS
tts = TTS(auto_download=True)
print(f"    Supertonic ready @ {SAMPLE_RATE}Hz")

# ─── TTS Worker ────────────────────────────────────────────────────────────────
tts_queue = queue.Queue()
tts_thread_running = threading.Event()
tts_thread_running.set()

def tts_worker():
    pending_text = ""
    while tts_thread_running.is_set():
        try:
            item = tts_queue.get(timeout=0.05)
            if item is None:
                break
            if isinstance(item, tuple):
                tag, data = item
                if tag == "flush":
                    if pending_text.strip():
                        style = tts.get_voice_style("M1")
                        wav, dur = tts.synthesize(pending_text.strip(), voice_style=style, lang="es")
                        arr = np.asarray(wav, dtype=np.float32)
                        if arr.ndim == 2 and arr.shape[0] == 1:
                            arr = arr.squeeze(0)
                        tts_queue.put(("audio", arr))
                    pending_text = ""
                    continue
                elif tag == "stop":
                    if pending_text.strip():
                        style = tts.get_voice_style("M1")
                        wav, dur = tts.synthesize(pending_text.strip(), voice_style=style, lang="es")
                        arr = np.asarray(wav, dtype=np.float32)
                        if arr.ndim == 2 and arr.shape[0] == 1:
                            arr = arr.squeeze(0)
                        tts_queue.put(("audio", arr))
                    tts_thread_running.clear()
                    break
                elif tag == "text":
                    pending_text += " " + data
                else:
                    continue
            else:
                pending_text += " " + item

            if len(pending_text.strip()) < 3:
                continue
            style = tts.get_voice_style("M1")
            wav, dur = tts.synthesize(pending_text.strip(), voice_style=style, lang="es")
            arr = np.asarray(wav, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[0] == 1:
                arr = arr.squeeze(0)
            tts_queue.put(("audio", arr))
            pending_text = ""
        except queue.Empty:
            if pending_text.strip():
                try:
                    style = tts.get_voice_style("M1")
                    wav, dur = tts.synthesize(pending_text.strip(), voice_style=style, lang="es")
                    arr = np.asarray(wav, dtype=np.float32)
                    if arr.ndim == 2 and arr.shape[0] == 1:
                        arr = arr.squeeze(0)
                    tts_queue.put(("audio", arr))
                except: pass
                pending_text = ""

tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()
print("[5/7] TTS worker started")

# ─── Audio Utils ──────────────────────────────────────────────────────────────
def mel_spec(pcm):
    x = torch.from_numpy(pcm).float().unsqueeze(0)
    w = torch.hann_window(2048)
    spec = torch.stft(x, n_fft=2048, hop_length=256, win_length=2048,
                     window=w, onesided=True, return_complex=True)
    mag = spec.squeeze(0).abs().pow(2)
    mel = torch.nn.functional.linear(torch.log1p(mag.T),
        torch.ones(128, 1025)*0.01).relu()
    return mel[:, :128].numpy()

# ─── Silero VAD helper ─────────────────────────────────────────────────────────
def check_vad_speech(audio_chunk: np.ndarray, threshold: float = 0.5) -> bool:
    """Run Silero VAD on a mic chunk. Returns True if speech detected.

    Silero expects exactly 512 samples at 16kHz (32ms). Takes last 512 samples of chunk."""
    try:
        wav = torch.from_numpy(audio_chunk).float()
        # Silero needs exactly 512 samples at 16kHz
        if len(wav) >= 512:
            wav = wav[-512:]
        else:
            # Pad to 512
            wav = torch.nn.functional.pad(wav, (512 - len(wav), 0))
        with torch.no_grad():
            prob = vad_model(wav.unsqueeze(0), 16000).item()
        return prob > threshold
    except Exception:
        return False

# ─── Full Duplex Pipeline ───────────────────────────────────────────────────────
def full_duplex(segment: np.ndarray, context: str = "") -> np.ndarray:
    mel = mel_spec(segment)
    T = mel.shape[0]
    mel_pad = np.pad(mel, ((0,0),(0, 896)), mode='constant')
    _ = mel_pad @ in_proj.T

    prompt = f"""<start_of_turn>user
[Audio: {T} mel frames]
{context}
<end_of_turn>
<start_of_turn>model
"""
    full_text = ""
    tokens_buffer = ""
    t0 = time.time()
    token_count = 0

    tts_queue.put(("flush", None))
    stream_out = llm(prompt, max_tokens=180,
                    stop=["<end_of_turn>"], echo=False, stream=True)

    for token_data in stream_out:
        token_text = token_data["choices"][0]["text"]
        full_text += token_text
        tokens_buffer += token_text
        token_count += 1

        if token_count % 3 == 0 and tokens_buffer.strip():
            tts_queue.put(("text", tokens_buffer.strip()))
            tokens_buffer = ""

        # Early exit
        if len(full_text) > 20 and any(e in full_text[-5:] for e in ".!?"):
            break

    if tokens_buffer.strip():
        tts_queue.put(("text", tokens_buffer.strip()))
    tts_queue.put(("flush", None))
    tts_queue.put(("stop", None))
    tts_thread.join(timeout=8.0)

    audio_chunks = []
    deadline = time.time() + 5.0
    while time.time() < deadline:
        try:
            item = tts_queue.get(timeout=0.5)
            if isinstance(item, tuple) and item[0] == "audio":
                _, arr = item
                audio_chunks.append(arr)
            elif item is None:
                break
        except queue.Empty:
            break

    elapsed = time.time() - t0
    total_dur = sum(c.shape[0] for c in audio_chunks) / SAMPLE_RATE
    print(f"    [{T} frames → {len(full_text)} chars in {elapsed*1000:.0f}ms | {total_dur:.1f}s audio]")
    return np.concatenate(audio_chunks) if audio_chunks else np.zeros(int(SAMPLE_RATE*0.5), dtype=np.float32)

# ─── Barge-in Live Loop ───────────────────────────────────────────────────────
q_in = queue.Queue(maxsize=20)
q_out = queue.Queue()
stop_event = threading.Event()
interrupt_event = threading.Event()

# VAD params
VAD_THRESHOLD = 0.5      # Silero speech probability threshold
VAD_MIN_CHUNKS = 3       # Min consecutive VAD detections to trigger interrupt
SILENCE_LIMIT = 12       # Silence chunks before processing

def audio_cb(indata, frames, time_info, status):
    if status: print(f"[VAD] {status}")
    try:
        q_in.put_nowait(indata[:, 0].copy())
    except:
        pass

def play_cb(outdata, frames, time_info, status):
    if status: print(f"[OUT] {status}")
    try:
        chunk = q_out.get_nowait()
        outdata[:, 0] = chunk
    except queue.Empty:
        outdata[:, 0] = 0

def live_loop():
    print("[6/7] Opening streams...")
    buf = np.zeros(0, dtype=np.float32)
    silence = 0
    talking = False
    vad_confirmed = 0  # consecutive VAD detections

    with (
        sd.InputStream(samplerate=MIC_RATE, channels=1, dtype='float32',
                       blocksize=2560, callback=audio_cb),
        sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32',
                        blocksize=4096, callback=play_cb)
    ):
        print("[7/7] Ready! Speak now (barge-in: interrupt me anytime)...")
        while not stop_event.is_set():
            try:
                chunk = q_in.get(timeout=0.1)
            except queue.Empty:
                # Periodic VAD check even when queue empty
                if len(buf) >= MIC_RATE * 0.5:
                    if check_vad_speech(buf[-int(MIC_RATE * 0.5):], VAD_THRESHOLD):
                        vad_confirmed += 1
                    else:
                        vad_confirmed = max(0, vad_confirmed - 1)

                    # Barge-in: if user speaks clearly while we were playing
                    if vad_confirmed >= VAD_MIN_CHUNKS and not q_out.empty():
                        print(f"\n    → BARG-IN! User spoke, interrupting...")
                        # Clear output queue
                        while not q_out.empty():
                            try: q_out.get_nowait()
                            except queue.Empty: break
                        interrupt_event.set()
                        vad_confirmed = 0
                        buf = np.zeros(0, dtype=np.float32)
                        continue
                continue

            buf = np.concatenate([buf, chunk])
            buf = buf[-MIC_RATE * 6:]

            # VAD check on latest 0.5s
            if len(buf) >= MIC_RATE * 0.5:
                if check_vad_speech(buf[-int(MIC_RATE * 0.5):], VAD_THRESHOLD):
                    vad_confirmed += 1
                else:
                    vad_confirmed = max(0, vad_confirmed - 1)

            # VAD-based talking detection
            if vad_confirmed >= VAD_MIN_CHUNKS and not talking:
                talking = True

            # Energy-based silence detection
            energy = np.abs(chunk).mean()
            if energy < 0.01:
                silence += 1
                if silence > SILENCE_LIMIT and talking:
                    talking = False
                    vad_confirmed = 0
                    seg = buf.copy()
                    buf = np.zeros(0, dtype=np.float32)
                    dur = len(seg) / MIC_RATE
                    print(f"\n    → {dur:.1f}s spoken, processing...")
                    interrupt_event.clear()
                    t0 = time.time()
                    out = full_duplex(seg, "Responde en español.")
                    elapsed = time.time() - t0
                    print(f"    → {len(out)/SAMPLE_RATE:.1f}s audio in {elapsed:.1f}s")
                    q_out.put(out)
            else:
                if energy >= 0.01:
                    silence = 0

            print(".", end="", flush=True)
    print("\nStopped.")

def demo():
    print("\n[Demo] Testing Silero VAD + full pipeline...")
    # Test VAD with correct 512-sample chunks
    print("Testing Silero VAD (512 samples @ 16kHz)...")
    dummy = np.sin(2*np.pi*440*np.linspace(0, 1, 512)).astype(np.float32) * 0.1
    prob = vad_model(torch.from_numpy(dummy).unsqueeze(0), 16000).item()
    print(f"    440Hz tone prob: {prob:.3f} (should be < 0.5)")

    # Test pipeline
    print("Testing full pipeline...")
    seg = np.sin(2*np.pi*220*np.linspace(0, 1.5, int(MIC_RATE*1.5))).astype(np.float32)*0.04
    out = full_duplex(seg, "Di hola y.presentate en español.")
    print(f"[Demo] {out.shape[0]/SAMPLE_RATE:.1f}s audio")
    sf.write(r"D:/michi-adapter/response_bargein.wav", out, SAMPLE_RATE)
    try:
        sd.play(out, SAMPLE_RATE); sd.wait()
    except: pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["demo","mic"], default="demo")
    args = parser.parse_args()
    if args.mode == "demo":
        demo()
    else:
        try:
            live_loop()
        except KeyboardInterrupt:
            stop_event.set()
            tts_thread_running.clear()
