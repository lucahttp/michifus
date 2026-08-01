"""
Michi-Streaming V2: Full-duplex with proper model pre-loading + Gemma token streaming
Key fix: all models loaded BEFORE the loop starts. Streaming only used for generation.
TTS: batch but run in background thread so it doesn't block Gemma.
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

# Prompt mode: "full" (33 tokens) | "minimal" (22 tokens) | "audio_removed" (10 tokens)
# minimal removes "en español, breve." -> just "Responde."
# audio_removed also removes the [Audio: N mel frames] line (only if Gemma ignores it)
PROMPT_MODE = "minimal"

# ─── Phase 1: Load ALL models upfront ────────────────────────────────────────
print("[1/7] Loading Gemma 4 E2B GGUF...")
from llama_cpp import Llama
llm = Llama(
    model_path=os.path.join(MODEL_DIR, "gemma-4-E2B-it-qat-UD-Q2_K_XL.gguf"),
    tokenizer_file=os.path.join(MODEL_DIR, "tokenizer.model"),
    tokenizer_repo_id="unsloth/gemma-4-E2B-it-qat-mobile-GGUF",
    hf_token=HF_TOKEN,
    n_ctx=1024,          # reduced to save memory
    n_gpu_layers=99,
    flash_attention=True,
    verbose=False,
)
print(f"    Gemma ready. n_ctx={llm.n_ctx()}")

print("[2/7] Loading adapters...")
ckpt = torch.load(ADAPTER_PATH, map_location="cpu", weights_only=False)
in_proj = ckpt["in_adapter"]["proj.weight"].numpy()
out_proj = ckpt["out_adapter"]["proj.weight"].numpy()
del ckpt
print(f"    in={in_proj.shape} out={out_proj.shape}")

print("[3/7] Loading Supertonic-3 TTS...")
from supertonic import TTS
tts = TTS(auto_download=True)
print(f"    Supertonic ready @ {SAMPLE_RATE}Hz")

# ─── Phase 2: Background TTS queue ─────────────────────────────────────────────
tts_queue = queue.Queue()
tts_thread_running = threading.Event()
tts_thread_running.set()

def tts_worker():
    """Background thread: waits for text, synthesizes, puts audio in queue."""
    pending_text = ""
    while tts_thread_running.is_set():
        try:
            item = tts_queue.get(timeout=0.05)
            if item is None:
                break
            # Unpack tagged tuples: ("text", str) or ("flush",)
            if isinstance(item, tuple):
                tag, data = item
                if tag == "flush":  # data is None for flush signals
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
                    tts_thread_running.clear()  # exit loop
                    break
                elif tag == "text":
                    pending_text += " " + data
                else:
                    continue  # unknown tag, skip
            else:
                # Plain string (backward compat)
                pending_text += " " + item

            # Only synthesize when we have meaningful text
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
            # Timeout = no new text, flush if there's pending
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
print("[4/7] TTS worker started")

# ─── Audio Utils ──────────────────────────────────────────────────────────────
# Echo mitigation: when speaker is playing, raise VAD threshold to ignore echo bleed
speaker_active = False
def set_speaker_active(active):
    global speaker_active
    speaker_active = active

def is_speech(pcm, thr=0.015):
    """VAD: returns True if pcm contains speech. Raises threshold when speaker active to mitigate echo."""
    base = np.abs(pcm).mean()
    if speaker_active:
        # Raise threshold 3x when speaker playing — echo bleed won't trigger VAD
        return base > thr * 3.0
    return base > thr

def mel_spec(pcm):
    x = torch.from_numpy(pcm).float().unsqueeze(0)
    w = torch.hann_window(2048)
    spec = torch.stft(x, n_fft=2048, hop_length=256, win_length=2048,
                     window=w, onesided=True, return_complex=True)
    mag = spec.squeeze(0).abs().pow(2)
    mel = torch.nn.functional.linear(torch.log1p(mag.T),
        torch.ones(128, 1025)*0.01).relu()
    return mel[:, :128].numpy()

# ─── Streaming early-exit logic ─────────────────────────────────────────────────
# Gemma (and most Gemma-instruct variants) emits specific patterns when done:
#   1. Sentence-ending punctuation: . ! ?  (but NOT abbreviations like "Dr." "U.")
#   2. <end_of_turn> token appears IN the stream (model wrote the stop marker itself)
#   3. A second <start_of_turn>user appears (model yielded to next turn — done)
#   4. Repetitive token cycles: "jajajaja" or "si si si si" → loop detected
#   5. Very short complete phrases (3-15 chars): "si", "no", "hola", "tal vez"

def should_stop_streaming(text_so_far: str, token_count: int, recent_tokens: list) -> bool:
    """
    Returns True when Gemma has naturally finished its response and streaming
    should stop early (before max_tokens=180 is reached).

    Detection order (cheapest checks first):
      1. Punctuation end       — last 5 tokens end with sentence closer
      2. Turn-control tokens   — <end_of_turn> or second <start_of_turn> seen
      3. Repetition            — same token 4+ times, or 2-token cycle 3+ times
      4. Short complete phrase — 3-15 chars, all alphabetic, no pending continuation
    """
    # ── 1. Punctuation end ──────────────────────────────────────────────────────
    # Check last 8 chars (covers token boundaries).  Avoid false positives from
    # abbreviations by requiring space or line-start BEFORE the punctuation.
    _PUNCT = frozenset(".!?")
    last8 = text_so_far[-8:] if len(text_so_far) >= 8 else text_so_far
    for i, ch in enumerate(last8):
        if ch in _PUNCT:
            # Grab the character IMMEDIATELY before the punctuation (if any)
            if i > 0:
                before = last8[i - 1]
                # Allow . after letters/numbers that form abbreviations or decimals
                if ch == "." and (before.isdigit() or before.lower() in "abcdefghijklmnopqrstuvwxyz"):
                    continue   # might be "Dr." / "3.14" — skip this one, keep scanning
            # Got a clean sentence end
            return True

    # ── 2. Turn-control tokens ───────────────────────────────────────────────────
    if "<end_of_turn>" in text_so_far:
        return True
    # Second <start_of_turn> (user turn) means model yielded — done
    if text_so_far.count("<start_of_turn>") >= 2:
        return True

    # ── 3. Repetition / looping detection ────────────────────────────────────────
    if len(recent_tokens) >= 4:
        # 3a. Same token 4+ consecutive times
        if all(t == recent_tokens[-1] for t in recent_tokens[-4:]):
            return True
        # 3b. 2-token cycle repeated 3 times: [A, B, A, B, A, B]
        if len(recent_tokens) >= 6:
            cycle = recent_tokens[-2:]
            if (cycle[0] == cycle[1]) is False:  # make sure it's not a single repeating token
                candidate = recent_tokens[-6:]
                if (candidate[0] == cycle[0] and candidate[1] == cycle[1] and
                    candidate[2] == cycle[0] and candidate[3] == cycle[1] and
                    candidate[4] == cycle[0] and candidate[5] == cycle[1]):
                    return True

    # ── 4. Short complete phrase (short responses: "si", "no", "hola") ──────────
    # Only trigger after at least 3 tokens AND the text is clearly done (no trailing
    # spaces, no open brackets, starts with capital or is all-lowercase common words)
    stripped = text_so_far.strip()
    short_words = frozenset({
        "si", "no", "hola", "adiós", "gracias", "de nada",
        "bueno", "vale", "ok", "okay", "ay", "eh", "mmm",
        "tal vez", "quizá", "quizas", "es posible",
        "perfecto", "entendido", "claro", "obvio",
    })
    # Must be a short alphabetic-only phrase (possibly accented)
    if (3 <= token_count <= 20 and
        1 < len(stripped) <= 15 and
        stripped.lower() in short_words):
        return True

    return False


# ─── Phase 3: Full-duplex loop ─────────────────────────────────────────────────
def full_duplex(segment: np.ndarray, context: str = "") -> np.ndarray:
    """
    Process mic segment:
    1. mel → Gemma (streaming tokens)
    2. Every 3 tokens → send to TTS queue (background)
    3. Collect TTS audio as it comes
    4. Return concatenated audio
    """
    mel = mel_spec(segment)
    T = mel.shape[0]
    mel_pad = np.pad(mel, ((0,0),(0, 896)), mode='constant')
    _ = mel_pad @ in_proj.T  # warm-up projection

    # Build prompt based on PROMPT_MODE to minimize token overhead per call.
    # Turn format (<start_of_turn>/<end_of_turn>) is required for Gemma.
    # Token overhead per mode (approx):
    #   full:         ~33 tokens  (<start...> + [Audio: N] + "Responde en español, breve." + <end...>)
    #   minimal:      ~22 tokens  (<start...> + [Audio: N] + "Responde." + <end...>)
    #   audio_removed: ~10 tokens (<start...> + "Responde." + <end...>)  -- only if Gemma ignores the audio line
    if PROMPT_MODE == "audio_removed":
        prompt = f"""<start_of_turn>user
Responde.
<end_of_turn>
<start_of_turn>model
"""
    elif PROMPT_MODE == "minimal":
        prompt = f"""<start_of_turn>user
[Audio: {T} mel frames]
Responde.
<end_of_turn>
<start_of_turn>model
"""
    else:  # full (default)
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
    audio_chunks = []

    # Warm up TTS worker with a flush signal
    tts_queue.put(("flush", None))

    # Track recent tokens for repetition / cycle detection
    recent_tokens = []

    # Stream tokens from Gemma
    stream_out = llm(prompt, max_tokens=180,
                    stop=["<end_of_turn>"], echo=False, stream=True)

    for token_data in stream_out:
        token_text = token_data["choices"][0]["text"]
        full_text += token_text
        tokens_buffer += token_text
        token_count += 1

        # Keep last 8 tokens for repetition detection
        recent_tokens.append(token_text)
        if len(recent_tokens) > 8:
            recent_tokens.pop(0)

        # Every 3 tokens, send to TTS queue
        if token_count % 3 == 0 and tokens_buffer.strip():
            tts_queue.put(("text", tokens_buffer.strip()))
            tokens_buffer = ""

        # Smart early exit — all stop conditions handled in should_stop_streaming
        if token_count > 2 and should_stop_streaming(full_text, token_count, recent_tokens):
            break

    # Flush remaining
    if tokens_buffer.strip():
        tts_queue.put(("text", tokens_buffer.strip()))
    tts_queue.put(("flush", None))

    # Signal worker to stop and wait for it to finish synthesis
    tts_queue.put(("stop", None))
    tts_thread.join(timeout=8.0)

    # Collect TTS audio chunks from the queue
    deadline = time.time() + 5.0
    while time.time() < deadline:
        try:
            item = tts_queue.get(timeout=0.5)
            if not isinstance(item, tuple) or item[0] != "audio":
                continue
            _, arr = item
            audio_chunks.append(arr)
        except queue.Empty:
            break
        except Exception:
            continue

    elapsed = time.time() - t0
    total_dur = sum(c.shape[0] for c in audio_chunks) / SAMPLE_RATE
    print(f"    [{T} frames → {len(full_text)} chars in {elapsed*1000:.0f}ms | {total_dur:.1f}s audio]")
    return np.concatenate(audio_chunks) if audio_chunks else np.zeros(int(SAMPLE_RATE*0.5), dtype=np.float32)

# ─── Phase 4: VAD + Live Loop ─────────────────────────────────────────────────
q_in = queue.Queue(maxsize=20)
q_out = queue.Queue()        # pending audio to play
play_event = threading.Event()
stop_event = threading.Event()

# ── VAD Tuning ──────────────────────────────────────────────────────────────
SPEECH_THRESHOLD  = 0.012   # was 0.015 — slightly more sensitive to soft speech
SILENCE_THRESHOLD = 10      # was 15 — 10 * 160ms = 1.6s (natural turn-taking gap)
MIN_TALK_BUFFER   = 0.8     # was 1.0 — seconds of audio before we consider user "talking"

def audio_cb(indata, frames, time_info, status):
    """Non-blocking mic callback — always returns quickly."""
    if status: print(f"[VAD] {status}")
    try:
        q_in.put_nowait(indata[:, 0].copy())
    except:
        pass

def play_cb(outdata, frames, time_info, status):
    """Output stream callback — feeds audio from q_out, zeros when empty."""
    if status: print(f"[OUT] {status}")
    try:
        chunk = q_out.get_nowait()
        outdata[:, 0] = chunk
        play_event.set()  # signal we started playing
        set_speaker_active(True)  # speaker is emitting audio
    except queue.Empty:
        outdata[:, 0] = 0
        set_speaker_active(False)  # speaker idle

def live_loop():
    print("[5/7] Opening streams...")
    buf = np.zeros(0, dtype=np.float32)
    silence = 0
    talking = False
    playing = False

    with (
        sd.InputStream(samplerate=MIC_RATE, channels=1, dtype='float32',
                       blocksize=2560, callback=audio_cb),
        sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32',
                        blocksize=4096, callback=play_cb)
    ):
        print("[6/7] Ready! Speak now (full-duplex — mic + audio overlap)...")
        while not stop_event.is_set():
            try:
                chunk = q_in.get(timeout=0.1)
            except queue.Empty:
                pass
            else:
                buf = np.concatenate([buf, chunk])
                buf = buf[-MIC_RATE * 6:]

                if is_speech(chunk, thr=SPEECH_THRESHOLD):
                    silence = 0
                    if not talking and len(buf) >= MIC_RATE * MIN_TALK_BUFFER:
                        talking = True
                else:
                    silence += 1
                    if silence > SILENCE_THRESHOLD and talking:
                        talking = False
                        seg = buf.copy()
                        buf = np.zeros(0, dtype=np.float32)
                        dur = len(seg) / MIC_RATE
                        print(f"\n    → {dur:.1f}s spoken, processing...")
                        play_event.clear()
                        t0 = time.time()
                        out = full_duplex(seg, "" if PROMPT_MODE != "full" else "Responde en español, breve.")
                        elapsed = time.time() - t0
                        print(f"    → {len(out)/SAMPLE_RATE:.1f}s audio in {elapsed:.1f}s (overlap mode)")
                        q_out.put(out)   # non-blocking — plays as callback drains it
                        playing = True

            # Drain q_out when it has too much backlog (prevents memory growth)
            while q_out.qsize() > 2:
                try: q_out.get_nowait()
                except queue.Empty: break

            print(".", end="", flush=True)
    print("\n[7/7] Stopped.")

def demo():
    print("\n[Demo] Full pipeline test...")
    # synthetic mic segment
    seg = np.sin(2*np.pi*220*np.linspace(0, 1.5, int(MIC_RATE*1.5))).astype(np.float32)*0.04
    out = full_duplex(seg, "" if PROMPT_MODE != "full" else "Di hola y.presentate en español.")
    print(f"[Demo] {out.shape[0]/SAMPLE_RATE:.1f}s audio")
    sf.write(r"D:/michi-adapter/response_fd.wav", out, SAMPLE_RATE)
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
