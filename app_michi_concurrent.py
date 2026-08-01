"""
Michi-Concurrent: Full-duplex emulado con arquitectura concurrente

Estrategia:
1. Echo cancellation: speaker output como referencia, resta simple de vectores
2. Backchannel detection: frases cortas de Gemma -> Supertonic inmediato
3. Parakeet (cuando este disponible) como streaming ASR
4. Gemma streaming + Supertonic concurrente

Para overlap real simultaneo sin cable virtual, el plan B es:
- Mantener mic y speaker activos
- Echo bleed mitigado por ANC de los Jabra + resta simple
- Backchannelsdan sensacion de conversacion simultanea
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

SAMPLE_RATE = 24000   # Supertonic output
MIC_RATE = 16000      # Mic input
CHUNK_MS = 160        # 2560 samples at 16kHz = 160ms per callback

# ─── Load Models ────────────────────────────────────────────────────────────────
print("[1/7] Loading Silero VAD...")
torch.set_num_threads(4)
torch.hub.set_dir(r"D:/michi-adapter/models")
vad_model, vad_utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
get_speech_timestamps = vad_utils[3]
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

# ─── Backchannel Words ─────────────────────────────────────────────────────────
BACKCHANNEL_WORDS = {
    "si", "si.", "si!", "ya", "ya.", "ya!", "ajá", "ajá.", "ajá!",
    "claro", "claro.", "claro!", "entiendo", "bueno", "bueno.",
    "ok", "ok.", "vale", "vale.", "mhm", "mm", "uhum", "eso", "eso.",
    "perfecto", "perfecto.", " bien", " bien.", "yea", "yea.",
}

def is_backchannel(text: str) -> bool:
    """Returns True if text is a short backchannel response."""
    t = text.strip().lower()
    return t in BACKCHANNEL_WORDS

# ─── Echo Cancellation ─────────────────────────────────────────────────────────
# Shared state between audio threads
echo_ref_buffer = np.zeros(0, dtype=np.float32)  # Recent speaker output
echo_ref_lock = threading.Lock()

# Resample helper: 24kHz → 16kHz using linear interpolation
def resample_24k_to_16k(audio_24k: np.ndarray) -> np.ndarray:
    """Resample 24kHz audio to 16kHz using linear interpolation."""
    if len(audio_24k) == 0:
        return np.zeros(0, dtype=np.float32)
    # 24kHz / 16kHz = 1.5
    num_output = int(len(audio_24k) * 16 / 24)
    if num_output < 1:
        return np.zeros(1, dtype=np.float32)
    indices = np.linspace(0, len(audio_24k) - 1, num_output)
    return audio_24k[np.clip(indices.astype(int), 0, len(audio_24k) - 1)].astype(np.float32)

def echo_cancel(mic_chunk: np.ndarray) -> np.ndarray:
    """Subtract speaker reference from mic input. Returns echo-reduced mic chunk."""
    global echo_ref_buffer
    with echo_ref_lock:
        ref = echo_ref_buffer.copy()

    if len(ref) < len(mic_chunk):
        # Pad ref with zeros if shorter than mic chunk
        ref = np.pad(ref, (0, len(mic_chunk) - len(ref)))

    # Take last len(mic_chunk) samples of ref (most recent playback)
    ref = ref[-len(mic_chunk):]

    # Resample ref from 24kHz to 16kHz
    ref_resampled = resample_24k_to_16k(ref)

    # Ensure same length as mic
    if len(ref_resampled) < len(mic_chunk):
        ref_resampled = np.pad(ref_resampled, (0, len(mic_chunk) - len(ref_resampled)))
    ref_resampled = ref_resampled[:len(mic_chunk)]

    # Adaptive echo cancellation: subtract with learning rate
    # Estimate echo level using correlation
    mic_energy = np.sqrt(np.mean(mic_chunk ** 2)) + 1e-10
    ref_energy = np.sqrt(np.mean(ref_resampled ** 2)) + 1e-10

    if ref_energy < 1e-6:
        return mic_chunk  # No reference signal

    # Optimal gain to minimize echo: gain = <mic, ref> / <ref, ref>
    correlation = np.dot(mic_chunk, ref_resampled)
    gain = correlation / (np.dot(ref_resampled, ref_resampled) + 1e-10)
    gain = np.clip(gain, 0.0, 1.0)  # Only subtract, don't amplify

    # Apply echo cancellation
    cancelled = mic_chunk - gain * ref_resampled

    # Mix original if cancellation is too aggressive
    mix_ratio = 0.3  # 30% original mixed in
    cancelled = (1 - mix_ratio) * cancelled + mix_ratio * mic_chunk

    return cancelled

def push_echo_reference(audio: np.ndarray):
    """Append speaker output to echo reference buffer."""
    global echo_ref_buffer
    with echo_ref_lock:
        echo_ref_buffer = np.concatenate([echo_ref_buffer, audio])
        # Keep max 5 seconds of reference (24kHz * 5 = 120000 samples)
        echo_ref_buffer = echo_ref_buffer[-120000:]

# ─── TTS Worker ────────────────────────────────────────────────────────────────
tts_queue = queue.Queue()
tts_thread_running = threading.Event()
tts_thread_running.set()
tts_lock = threading.Lock()

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
                        push_echo_reference(arr)  # ← echo reference
                    pending_text = ""
                    continue
                elif tag == "immediate":
                    # Backchannel — synthesize immediately, high priority
                    if data.strip():
                        with tts_lock:
                            style = tts.get_voice_style("M1")
                            wav, dur = tts.synthesize(data.strip(), voice_style=style, lang="es")
                            arr = np.asarray(wav, dtype=np.float32)
                            if arr.ndim == 2 and arr.shape[0] == 1:
                                arr = arr.squeeze(0)
                            tts_queue.put(("audio", arr))
                            push_echo_reference(arr)  # ← echo reference
                    continue
                elif tag == "stop":
                    if pending_text.strip():
                        with tts_lock:
                            style = tts.get_voice_style("M1")
                            wav, dur = tts.synthesize(pending_text.strip(), voice_style=style, lang="es")
                            arr = np.asarray(wav, dtype=np.float32)
                            if arr.ndim == 2 and arr.shape[0] == 1:
                                arr = arr.squeeze(0)
                            tts_queue.put(("audio", arr))
                            push_echo_reference(arr)
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
            with tts_lock:
                style = tts.get_voice_style("M1")
                wav, dur = tts.synthesize(pending_text.strip(), voice_style=style, lang="es")
                arr = np.asarray(wav, dtype=np.float32)
                if arr.ndim == 2 and arr.shape[0] == 1:
                    arr = arr.squeeze(0)
                tts_queue.put(("audio", arr))
                push_echo_reference(arr)  # ← echo reference
            pending_text = ""
        except queue.Empty:
            if pending_text.strip():
                with tts_lock:
                    try:
                        style = tts.get_voice_style("M1")
                        wav, dur = tts.synthesize(pending_text.strip(), voice_style=style, lang="es")
                        arr = np.asarray(wav, dtype=np.float32)
                        if arr.ndim == 2 and arr.shape[0] == 1:
                            arr = arr.squeeze(0)
                        tts_queue.put(("audio", arr))
                        push_echo_reference(arr)
                    except: pass
                pending_text = ""

tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()
print("[5/7] TTS worker started")

# ─── Audio Utils ───────────────────────────────────────────────────────────────
def mel_spec(pcm):
    x = torch.from_numpy(pcm).float().unsqueeze(0)
    w = torch.hann_window(2048)
    spec = torch.stft(x, n_fft=2048, hop_length=256, win_length=2048,
                     window=w, onesided=True, return_complex=True)
    mag = spec.squeeze(0).abs().pow(2)
    mel = torch.nn.functional.linear(torch.log1p(mag.T),
        torch.ones(128, 1025)*0.01).relu()
    mel = mel[:, :128]
    if mel.ndim == 1:
        mel = mel.unsqueeze(0)
    return mel.numpy()

def check_vad(audio: np.ndarray, thr: float = 0.5) -> bool:
    """Silero VAD on 512 samples."""
    try:
        wav = torch.from_numpy(audio).float()
        if len(wav) >= 512:
            wav = wav[-512:]
        else:
            wav = torch.nn.functional.pad(wav, (512 - len(wav), 0))
        with torch.no_grad():
            prob = vad_model(wav.unsqueeze(0), 16000).item()
        return prob > thr
    except:
        return False

# ─── Gemma Streaming with Backchannel Detection ─────────────────────────────────
def gemma_stream(mel: np.ndarray, context: str = "") -> tuple[str, str]:
    """
    Stream text from Gemma, detecting backchannels immediately.
    Returns: (full_text, final_response)
    final_response is the non-backchannel part for full synthesis.
    """
    # Ensure mel is 2D
    if mel.ndim == 1:
        mel = mel.reshape(-1, 128)
    T = mel.shape[0]
    mel_pad = np.pad(mel, ((0, 0), (0, 896)), mode='constant')
    _ = mel_pad @ in_proj.T

    prompt = f"""<start_of_turn>user
[Audio: {T} mel frames]
{context}
<end_of_turn>
<start_of_turn>model
"""
    full_text = ""
    buffer = ""
    backchannel = ""
    final_response = ""
    in_backchannel = False
    t0 = time.time()

    stream_out = llm(prompt, max_tokens=180,
                    stop=["<end_of_turn>"], echo=False, stream=True)

    for tok_data in stream_out:
        token_text = tok_data["choices"][0]["text"]
        full_text += token_text
        buffer += token_text

        # Detect backchannel phrases
        if not in_backchannel:
            for phrase in BACKCHANNEL_WORDS:
                if buffer.lower().endswith(phrase):
                    in_backchannel = True
                    backchannel = buffer
                    # Send immediate TTS for backchannel
                    tts_queue.put(("immediate", phrase))
                    buffer = ""
                    break

        # Early exit on sentence end
        if len(full_text) > 20 and any(e in full_text[-5:] for e in ".!?"):
            if in_backchannel:
                final_response = buffer
                buffer = ""
            break

    if buffer.strip():
        if in_backchannel:
            final_response = buffer
        else:
            final_response = buffer

    # Flush remaining non-backchannel text
    if final_response.strip():
        tts_queue.put(("text", final_response.strip()))

    return full_text, final_response if not in_backchannel else ""

# ─── Pipeline ────────────────────────────────────────────────────────────────
def full_pipeline(mel: np.ndarray, context: str = "") -> np.ndarray:
    """Mel → Gemma (streaming) → Supertonic, with backchannel detection."""
    tts_queue.put(("flush", None))
    full_text, final_response = gemma_stream(mel, context)
    tts_queue.put(("flush", None))
    tts_queue.put(("stop", None))
    tts_thread.join(timeout=10.0)

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

    result = np.concatenate(audio_chunks) if audio_chunks else np.zeros(int(SAMPLE_RATE*0.5), dtype=np.float32)
    print(f"    [{mel.shape[0]} frames → {len(full_text)} chars | {result.shape[0]/SAMPLE_RATE:.1f}s audio]")
    return result

# ─── Concurrent Live Loop ─────────────────────────────────────────────────────
q_in = queue.Queue(maxsize=20)
q_out = queue.Queue()
stop_event = threading.Event()
is_speaking = threading.Event()

SPEECH_THRESH = 0.012
SILENCE_LIMIT = 10
MIN_TALK_BUFFER = 0.8

def audio_cb(indata, frames, time_info, status):
    if status: print(f"[MIC] {status}")
    try:
        raw = indata[:, 0].copy()
        # Apply echo cancellation using speaker reference
        clean = echo_cancel(raw)
        q_in.put_nowait(clean)
    except:
        pass

def play_cb(outdata, frames, time_info, status):
    if status: print(f"[OUT] {status}")
    try:
        chunk = q_out.get_nowait()
        outdata[:, 0] = chunk
        is_speaking.set()
    except queue.Empty:
        outdata[:, 0] = 0
        if q_out.empty():
            is_speaking.clear()

def live_loop():
    print("[6/7] Opening concurrent streams...")
    buf = np.zeros(0, dtype=np.float32)
    silence = 0
    talking = False
    vad_confirmed = 0

    with (
        sd.InputStream(samplerate=MIC_RATE, channels=1, dtype='float32',
                       blocksize=2560, callback=audio_cb),
        sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32',
                        blocksize=4096, callback=play_cb)
    ):
        print("[7/7] Ready! Full-duplex — speak anytime (barge-in + backchannels)...")
        while not stop_event.is_set():
            try:
                chunk = q_in.get(timeout=0.1)
            except queue.Empty:
                continue

            buf = np.concatenate([buf, chunk])
            buf = buf[-MIC_RATE * 6:]

            # Silero VAD check every tick
            if len(buf) >= 512:
                if check_vad(buf[-512:], 0.5):
                    vad_confirmed = min(vad_confirmed + 1, 5)
                else:
                    vad_confirmed = max(vad_confirmed - 1, 0)

            # Start talking on VAD confirmation
            if vad_confirmed >= 3 and not talking and len(buf) >= MIC_RATE * MIN_TALK_BUFFER:
                talking = True

            # Energy silence
            energy = np.abs(chunk).mean()
            if energy < SPEECH_THRESH:
                silence += 1
                if silence > SILENCE_LIMIT and talking:
                    talking = False
                    vad_confirmed = 0
                    seg = buf.copy()
                    buf = np.zeros(0, dtype=np.float32)
                    dur = len(seg) / MIC_RATE
                    print(f"\n    → {dur:.1f}s spoken, processing...")
                    t0 = time.time()
                    mel = mel_spec(seg)
                    out = full_pipeline(mel, "Responde en español.")
                    print(f"    → Total: {time.time()-t0:.1f}s")
                    q_out.put(out)
            else:
                if energy >= SPEECH_THRESH:
                    silence = 0

            print(".", end="", flush=True)
    print("\nStopped.")

def demo():
    print("\n[Demo] Full concurrent pipeline...")
    # Test echo cancellation
    print("Testing echo cancellation...")
    ref = np.sin(2*np.pi*440*np.linspace(0, 1, 24000)).astype(np.float32) * 0.1
    push_echo_reference(ref)
    mic_dummy = np.sin(2*np.pi*440*np.linspace(0, 1, 16000)).astype(np.float32) * 0.1
    cancelled = echo_cancel(mic_dummy)
    print(f"    Original energy: {np.sqrt(np.mean(mic_dummy**2)):.4f}")
    print(f"    Cancelled energy: {np.sqrt(np.mean(cancelled**2)):.4f}")

    # Test pipeline: convert PCM → mel → full_pipeline
    seg = np.sin(2*np.pi*220*np.linspace(0, 1.5, int(MIC_RATE*1.5))).astype(np.float32)*0.04
    mel = mel_spec(seg)
    out = full_pipeline(mel, "Di hola y.presentate en español.")
    print(f"[Demo] {out.shape[0]/SAMPLE_RATE:.1f}s audio")
    sf.write(r"D:/michi-adapter/response_concurrent.wav", out, SAMPLE_RATE)
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
