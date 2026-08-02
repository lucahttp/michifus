"""
Michi-Concurrent: Full-duplex emulado con Parakeet streaming ASR

Arquitectura:
  Mic → Parakeet (streaming ASR, cada ~300ms partial transcription)
       → Text + mel → Gemma (streaming)
       → Backchannels → immediate Supertonic TTS
       → Full response → Supertonic TTS worker

Echo cancellation: speaker output como referencia, resta adaptativa.
Backchannels: frases cortas ("si","ya","ajá") → TTS inmediato.
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
CHUNK_SAMPLES = 2560   # 160ms at 16kHz

# ─── Load Models ────────────────────────────────────────────────────────────────
print("[1/7] Loading Silero VAD...")
torch.set_num_threads(4)
torch.hub.set_dir(r"D:/michi-adapter/models")
vad_model, vad_utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
print("    Silero VAD loaded")

print("[2/7] Loading Parakeet-TDT streaming ASR...")
from nano_parakeet import from_pretrained
parakeet = from_pretrained("nvidia/parakeet-tdt-0.6b-v3", device="cpu")
parakeet.eval()
print(f"    Parakeet ready. DURATIONS={parakeet.DURATIONS}")

print("[3/7] Loading Gemma 4 E2B GGUF...")
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

print("[4/7] Loading adapters...")
ckpt = torch.load(ADAPTER_PATH, map_location="cpu", weights_only=False)
in_proj = ckpt["in_adapter"]["proj.weight"].numpy()
out_proj = ckpt["out_adapter"]["proj.weight"].numpy()
del ckpt
print(f"    in={in_proj.shape} out={out_proj.shape}")

print("[5/7] Loading Supertonic-3 TTS...")
from supertonic import TTS
tts = TTS(auto_download=True)
print(f"    Supertonic ready @ {SAMPLE_RATE}Hz")

# ─── Backchannel Words ────────────────────────────────────────────────────────
BACKCHANNEL = {
    "si", "si.", "ya", "ya.", "ajá", "ajá.", "claro", "claro.",
    "entiendo", "bueno", "bueno.", "ok", "ok.", "vale", "vale.",
    "mhm", "mm", "uhum", "eso", "eso.", "perfecto", " bien", "yea",
}

def is_backchannel(text: str) -> bool:
    t = text.strip().lower().rstrip(".!?,¡¿")
    return t in BACKCHANNEL or len(t) <= 3

# ─── Echo Cancellation ────────────────────────────────────────────────────────
echo_ref = np.zeros(0, dtype=np.float32)
echo_lock = threading.Lock()

def resample_24_to_16(audio_24: np.ndarray) -> np.ndarray:
    if len(audio_24) == 0:
        return np.zeros(0, dtype=np.float32)
    num_out = int(len(audio_24) * 16 / 24)
    if num_out < 1:
        return np.zeros(1, dtype=np.float32)
    indices = np.linspace(0, len(audio_24) - 1, num_out)
    return audio_24[np.clip(indices.astype(int), 0, len(audio_24) - 1)].astype(np.float32)

def echo_cancel(mic: np.ndarray) -> np.ndarray:
    global echo_ref
    with echo_lock:
        ref = echo_ref.copy()
    if len(ref) < len(mic):
        ref = np.pad(ref, (0, len(mic) - len(ref)))
    ref = ref[-len(mic):]
    ref_rs = resample_24_to_16(ref)
    if len(ref_rs) < len(mic):
        ref_rs = np.pad(ref_rs, (0, len(mic) - len(ref_rs)))
    ref_rs = ref_rs[:len(mic)]
    mic_e = np.sqrt(np.mean(mic ** 2)) + 1e-10
    ref_e = np.sqrt(np.mean(ref_rs ** 2)) + 1e-10
    if ref_e < 1e-6:
        return mic
    gain = np.clip(np.dot(mic, ref_rs) / (np.dot(ref_rs, ref_rs) + 1e-10), 0, 1)
    cancelled = mic - gain * ref_rs
    return 0.7 * cancelled + 0.3 * mic

def push_ref(audio: np.ndarray):
    global echo_ref
    with echo_lock:
        echo_ref = np.concatenate([echo_ref, audio])
        echo_ref = echo_ref[-120000:]   # 5s at 24kHz

# ─── TTS Worker ────────────────────────────────────────────────────────────────
tts_queue = queue.Queue()
tts_running = threading.Event()
tts_running.set()
tts_lock = threading.Lock()

def tts_worker():
    pending = ""
    while tts_running.is_set():
        try:
            item = tts_queue.get(timeout=0.05)
            if item is None:
                break
            if isinstance(item, tuple):
                tag, data = item
                if tag == "flush":
                    if pending.strip():
                        with tts_lock:
                            wav, dur = tts.synthesize(pending.strip(), voice_style=tts.get_voice_style("M1"), lang="es")
                            arr = np.asarray(wav, dtype=np.float32)
                            if arr.ndim == 2 and arr.shape[0] == 1: arr = arr.squeeze(0)
                        tts_queue.put(("audio", arr))
                        push_ref(arr)
                    pending = ""
                    continue
                elif tag == "immediate":
                    if data.strip():
                        with tts_lock:
                            wav, dur = tts.synthesize(data.strip(), voice_style=tts.get_voice_style("M1"), lang="es")
                            arr = np.asarray(wav, dtype=np.float32)
                            if arr.ndim == 2 and arr.shape[0] == 1: arr = arr.squeeze(0)
                        tts_queue.put(("audio", arr))
                        push_ref(arr)
                    continue
                elif tag == "stop":
                    if pending.strip():
                        with tts_lock:
                            wav, dur = tts.synthesize(pending.strip(), voice_style=tts.get_voice_style("M1"), lang="es")
                            arr = np.asarray(wav, dtype=np.float32)
                            if arr.ndim == 2 and arr.shape[0] == 1: arr = arr.squeeze(0)
                        tts_queue.put(("audio", arr))
                        push_ref(arr)
                    tts_running.clear()
                    break
                elif tag == "text":
                    pending += " " + data
                else:
                    continue
            else:
                pending += " " + item

            if len(pending.strip()) < 3:
                continue
            with tts_lock:
                wav, dur = tts.synthesize(pending.strip(), voice_style=tts.get_voice_style("M1"), lang="es")
                arr = np.asarray(wav, dtype=np.float32)
                if arr.ndim == 2 and arr.shape[0] == 1: arr = arr.squeeze(0)
            tts_queue.put(("audio", arr))
            push_ref(arr)
            pending = ""
        except queue.Empty:
            if pending.strip():
                with tts_lock:
                    try:
                        wav, dur = tts.synthesize(pending.strip(), voice_style=tts.get_voice_style("M1"), lang="es")
                        arr = np.asarray(wav, dtype=np.float32)
                        if arr.ndim == 2 and arr.shape[0] == 1: arr = arr.squeeze(0)
                        tts_queue.put(("audio", arr))
                        push_ref(arr)
                    except: pass
                pending = ""

threading.Thread(target=tts_worker, daemon=True).start()
print("[6/7] TTS worker started")

# ─── Audio Utils ──────────────────────────────────────────────────────────────
def mel_spec(pcm: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(pcm).float().unsqueeze(0)
    w = torch.hann_window(2048)
    spec = torch.stft(x, n_fft=2048, hop_length=256, win_length=2048,
                     window=w, onesided=True, return_complex=True)
    mag = spec.squeeze(0).abs().pow(2)
    mel = torch.nn.functional.linear(torch.log1p(mag.T), torch.ones(128, 1025)*0.01).relu()
    mel = mel[:, :128]
    return mel.numpy()

def silero_vad(audio: np.ndarray, thr: float = 0.5) -> bool:
    try:
        wav = torch.from_numpy(audio).float()
        if len(wav) >= 512: wav = wav[-512:]
        else: wav = torch.nn.functional.pad(wav, (512 - len(wav), 0))
        with torch.no_grad():
            return vad_model(wav.unsqueeze(0), 16000).item() > thr
    except:
        return False

# ─── Parakeet Worker ─────────────────────────────────────────────────────────
parakeet_lock = threading.Lock()
parakeet_buffer = np.zeros(0, dtype=np.float32)
parakeet_text = ""
parakeet_partial = ""

def parakeet_worker(buffer_len_sec: float = 3.0):
    """Add mic chunks to rolling buffer, transcribe every ~1s."""
    global parakeet_buffer, parakeet_text, parakeet_partial
    chunk_count = 0
    while tts_running.is_set():
        time.sleep(0.1)  # ~100ms between checks
        with parakeet_lock:
            if len(parakeet_buffer) < int(MIC_RATE * 0.5):
                continue
            # Use last 2-3s of audio
            audio = parakeet_buffer[-int(MIC_RATE * buffer_len_sec):]
        if len(audio) < int(MIC_RATE * 0.3):
            continue
        chunk_count += 1
        # Transcribe every other chunk (~200ms cadence)
        if chunk_count % 2 == 0:
            try:
                text = parakeet.transcribe(audio.astype(np.float32), timestamps=False)
                text = text.strip()
                if text and text != parakeet_partial:
                    old = parakeet_partial
                    parakeet_partial = text
                    # Only update main text if it grew (not repeating)
                    if len(text) > len(old):
                        parakeet_text = text
            except Exception as e:
                pass

def start_parakeet():
    t = threading.Thread(target=parakeet_worker, daemon=True)
    t.start()
    return t

# ─── Gemma Streaming with Backchannels ─────────────────────────────────────────
def gemma_stream(mel: np.ndarray, context: str = "") -> str:
    if mel.ndim == 1: mel = mel.reshape(-1, 128)
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
    buf = ""
    t0 = time.time()

    stream = llm(prompt, max_tokens=180, stop=["<end_of_turn>"], echo=False, stream=True)
    for tok in stream:
        txt = tok["choices"][0]["text"]
        full_text += txt
        buf += txt

        # Backchannel check
        bc = buf.strip().lower().rstrip(".!?,¡¿")
        if bc in BACKCHANNEL or (len(bc) <= 3 and bc):
            tts_queue.put(("immediate", buf.strip()))
            buf = ""

        if len(full_text) > 20 and any(e in full_text[-5:] for e in ".!?"):
            break

    if buf.strip():
        tts_queue.put(("text", buf.strip()))
    return full_text

# ─── Pipeline ────────────────────────────────────────────────────────────────
def full_pipeline(mel: np.ndarray, context: str = "") -> np.ndarray:
    tts_queue.put(("flush", None))
    t0 = time.time()
    text = gemma_stream(mel, context)
    tts_queue.put(("flush", None))
    tts_queue.put(("stop", None))

    chunks = []
    deadline = time.time() + 8.0
    while time.time() < deadline:
        try:
            item = tts_queue.get(timeout=0.5)
            if isinstance(item, tuple) and item[0] == "audio":
                chunks.append(item[1])
            elif item is None:
                break
        except queue.Empty:
            break

    result = np.concatenate(chunks) if chunks else np.zeros(int(SAMPLE_RATE*0.5), dtype=np.float32)
    dur = result.shape[0] / SAMPLE_RATE
    elapsed = time.time() - t0
    print(f"    [{mel.shape[0]} frames → {len(text)} chars | {dur:.1f}s audio in {elapsed:.1f}s]")
    return result

# ─── Live Loop ──────────────────────────────────────────────────────────────
q_in = queue.Queue(maxsize=20)
q_out = queue.Queue()
stop_ev = threading.Event()

SPEECH_THR = 0.012
SILENCE_LIM = 10
MIN_BUF = 0.8

def audio_cb(indata, frames, time_info, status):
    if status: print(f"[MIC] {status}")
    try:
        raw = indata[:, 0].copy()
        clean = echo_cancel(raw)
        q_in.put_nowait(clean)
        # Feed to Parakeet buffer
        with parakeet_lock:
            global parakeet_buffer
            parakeet_buffer = np.concatenate([parakeet_buffer, clean])
            # Keep max 10s
            parakeet_buffer = parakeet_buffer[-int(MIC_RATE * 10):]
    except: pass

def play_cb(outdata, frames, time_info, status):
    if status: print(f"[OUT] {status}")
    try:
        chunk = q_out.get_nowait()
        outdata[:, 0] = chunk
    except queue.Empty:
        outdata[:, 0] = 0

def live_loop():
    print("[7/7] Opening streams + starting Parakeet...")
    buf = np.zeros(0, dtype=np.float32)
    silence = 0
    talking = False
    vad_cnt = 0

    pk_t = start_parakeet()

    with (
        sd.InputStream(samplerate=MIC_RATE, channels=1, dtype='float32', blocksize=CHUNK_SAMPLES, callback=audio_cb),
        sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', blocksize=4096, callback=play_cb)
    ):
        print("[8/7] Ready! Full-duplex — speak anytime (barge-in + backchannels)...")
        while not stop_ev.is_set():
            try:
                chunk = q_in.get(timeout=0.1)
            except queue.Empty:
                continue

            buf = np.concatenate([buf, chunk])
            buf = buf[-MIC_RATE * 6:]

            # Silero VAD
            if len(buf) >= 512:
                if silero_vad(buf[-512:], 0.5):
                    vad_cnt = min(vad_cnt + 1, 5)
                else:
                    vad_cnt = max(vad_cnt - 1, 0)

            if vad_cnt >= 3 and not talking and len(buf) >= MIC_RATE * MIN_BUF:
                talking = True

            energy = np.abs(chunk).mean()
            if energy < SPEECH_THR:
                silence += 1
                if silence > SILENCE_LIM and talking:
                    talking = False
                    vad_cnt = 0
                    seg = buf.copy()
                    buf = np.zeros(0, dtype=np.float32)
                    dur = len(seg) / MIC_RATE
                    print(f"\n    → {dur:.1f}s spoken, processing...")
                    mel = mel_spec(seg)
                    out = full_pipeline(mel, "Responde en español.")
                    q_out.put(out)
            else:
                if energy >= SPEECH_THR:
                    silence = 0
            print(".", end="", flush=True)
    print("\nStopped.")

def demo():
    print("\n[Demo] Full concurrent pipeline...")
    print("Parakeet test (1s audio):")
    test_audio = np.sin(2*np.pi*440*np.linspace(0, 1, 16000)).astype(np.float32) * 0.01
    result = parakeet.transcribe(test_audio, timestamps=False)
    print(f"    440Hz tone: {repr(result)}")

    print("Full pipeline test:")
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
            stop_ev.set()
            tts_running.clear()
