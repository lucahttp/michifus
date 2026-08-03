"""
Michi-Lite — full-duplex voice agent on Apple Silicon (lightweight).

Pipeline (no trained checkpoints, no GGUF, no llama.cpp):
  Mic 16kHz → Silero VAD → mlx-whisper-tiny (STT)
         → mlx-lm Qwen2.5-0.5B-Instruct 4-bit (LLM, ES)
         → mlx-audio Kokoro-82M-bf16 (TTS, 24kHz) → Speaker

Full-duplex:
  - sounddevice InputStream callback (non-blocking) feeds q_mic
  - sounddevice OutputStream callback drains q_play
  - processing runs on main thread; mic keeps capturing (barge-in ready)
  - speaker_active() raises VAD threshold during TTS to suppress echo

Cross-platform config via env vars (no hardcoded D:/ or E:/ paths):
  MICHI_STT    default mlx-community/whisper-tiny-mlx
  MICHI_LLM    default mlx-community/Qwen2.5-0.5B-Instruct-4bit
  MICHI_TTS    default mlx-community/Kokoro-82M-bf16
  MICHI_VOICE  default af_heart
  MICHI_SYSTEM default short Spanish persona prompt
  MICHI_MAX_TOKENS default 80
  MICHI_FULL_DUPLEX default 1 (set 0 for half-duplex)
"""
import os
import sys
import time
import queue
import signal
import threading
import tempfile
import numpy as np
import sounddevice as sd
import soundfile as sf

# Audio defaults
MIC_RATE = 16000
TTS_RATE = 24000
CHUNK_MS = 32
CHUNK = int(MIC_RATE * CHUNK_MS / 1000)  # 512 samples @ 16kHz

# Config from env (no hardcoded paths)
WHISPER_REPO = os.environ.get("MICHI_STT", "mlx-community/whisper-tiny-mlx")
LLM_REPO = os.environ.get("MICHI_LLM", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")
TTS_REPO = os.environ.get("MICHI_TTS", "mlx-community/Kokoro-82M-bf16")
TTS_VOICE = os.environ.get("MICHI_VOICE", "af_heart")
TTS_LANG = os.environ.get("MICHI_TTS_LANG", "es")
SYSTEM_PROMPT = os.environ.get(
    "MICHI_SYSTEM",
    "Sos Michi, un asistente de voz. Respondé en español, breve y "
    "conversacional. No inventes datos; si no sabés, decilo.",
)
LLM_MAX_TOKENS = int(os.environ.get("MICHI_MAX_TOKENS", "80"))
FULL_DUPLEX = os.environ.get("MICHI_FULL_DUPLEX", "1") == "1"
ECHO_GAIN = float(os.environ.get("MICHI_ECHO_GAIN", "3.0"))  # VAD x-mult when speaker active

_state = {"whisper": None, "llm": None, "tok": None, "tts_warmed": False}


def _banner():
    print("=" * 60)
    print("Michi-Lite — full-duplex voice agent (lightweight)")
    print("=" * 60)
    print(f"  STT: {WHISPER_REPO}")
    print(f"  LLM: {LLM_REPO}")
    print(f"  TTS: {TTS_REPO}  voice={TTS_VOICE}  lang={TTS_LANG}")
    print(f"  full-duplex: {FULL_DUPLEX}  echo_mult: {ECHO_GAIN}x")
    print("=" * 60)


# ─── Silero VAD ────────────────────────────────────────────────────────────
_vad_model = None
def _get_vad():
    global _vad_model
    if _vad_model is None:
        import torch
        m, _ = torch.hub.load("snakers4/silero-vad", "silero_vad",
                              trust_repo=True, verbose=False)
        _vad_model = m
    return _vad_model


def vad_speech(pcm: np.ndarray, threshold: float = 0.5) -> bool:
    """Silero VAD on a 16kHz mono float32 chunk (>=512 samples)."""
    if pcm.size < 512:
        return False
    import torch
    chunk = pcm[:512] if pcm.size >= 512 else np.pad(pcm, (0, 512 - pcm.size))
    t = torch.from_numpy(chunk.astype(np.float32))
    prob = _get_vad()(t, MIC_RATE).item()
    return prob > threshold


def is_speech_rms(pcm: np.ndarray, threshold: float = 0.01) -> bool:
    """Cheap RMS gate layered on top of VAD for fast path."""
    return float(np.abs(pcm).mean()) > threshold


# ─── STT (mlx-whisper) ─────────────────────────────────────────────────────
def _load_whisper():
    if _state["whisper"] is None:
        import mlx_whisper
        _state["whisper"] = mlx_whisper
    return _state["whisper"]


def transcribe(pcm: np.ndarray) -> str:
    mw = _load_whisper()
    # mlx_whisper expects float32 mono at 16kHz; normalize peaks lightly
    audio = pcm.astype(np.float32, copy=False)
    peak = float(np.max(np.abs(audio))) or 1.0
    if peak > 1.0:
        audio = audio / peak
    res = mw.transcribe(audio, path_or_hf_repo=WHISPER_REPO, verbose=False)
    return (res.get("text") or "").strip()


# ─── LLM (mlx-lm) ──────────────────────────────────────────────────────────
def _load_llm():
    if _state["llm"] is None:
        from mlx_lm import load
        print(f"  [llm] loading {LLM_REPO}...")
        t0 = time.time()
        model, tok = load(LLM_REPO)
        print(f"  [llm] loaded in {time.time()-t0:.1f}s")
        _state["llm"] = model
        _state["tok"] = tok
    return _state["llm"], _state["tok"]


_chat_history: list[dict] = []
_HISTORY_TURNS = 6


def llm_respond(user_text: str) -> str:
    if not user_text.strip():
        return ""
    from mlx_lm import generate
    model, tok = _load_llm()

    _chat_history.append({"role": "user", "content": user_text})
    history = _chat_history[-(_HISTORY_TURNS * 2):]
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + history
    prompt = tok.apply_chat_template(msgs, tokenize=False,
                                     add_generation_prompt=True)
    t0 = time.time()
    out = generate(model, tok, prompt=prompt, max_tokens=LLM_MAX_TOKENS,
                   verbose=False)
    elapsed = time.time() - t0
    reply = (out or "").strip()
    # Some chat templates echo the prompt; strip leading whitespace/newlines
    if reply.startswith(_chat_history[-1]["content"][:20]):
        # Defensive: avoid double reply if model parrots user
        pass
    _chat_history.append({"role": "assistant", "content": reply})
    print(f"  [llm] {elapsed:.1f}s -> {reply[:120]!r}")
    return reply


# ─── TTS (mlx-audio Kokoro) ────────────────────────────────────────────────
def _ensure_tts():
    if _state["tts_warmed"]:
        return
    # Eagerly import so first synthesis isn't blocked by module load.
    import mlx_audio.tts.generate  # noqa: F401
    _state["tts_warmed"] = True


def tts_synthesize(text: str) -> np.ndarray:
    text = (text or "").strip()
    if not text:
        return np.zeros(0, dtype=np.float32)
    from mlx_audio.tts.generate import generate_audio
    # mlx-audio quirk: it treats output_path as a *directory* and writes
    # <output_path>/audio_000.wav inside it.
    tmpdir = tempfile.mkdtemp(prefix="michi_tts_")
    generate_audio(
        text=text,
        model=TTS_REPO,
        voice=TTS_VOICE,
        lang_code=TTS_LANG,
        output_path=tmpdir,
        play=False,
        stream=False,
        save=False,
        verbose=False,
    )
    inner = os.path.join(tmpdir, "audio_000.wav")
    if not os.path.exists(inner):
        for f in os.listdir(tmpdir):
            if f.endswith(".wav"):
                inner = os.path.join(tmpdir, f)
                break
    if not os.path.exists(inner):
        return np.zeros(0, dtype=np.float32)
    audio, sr = sf.read(inner, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TTS_RATE:
        from scipy.signal import resample
        n = int(round(len(audio) * TTS_RATE / sr))
        audio = resample(audio, n).astype(np.float32)
    return audio.astype(np.float32)


# ─── Full-duplex plumbing ──────────────────────────────────────────────────
q_mic = queue.Queue(maxsize=400)
q_play = queue.Queue()
stop_evt = threading.Event()
tts_evt = threading.Event()  # set while TTS audio is in speaker queue
vad_gate = threading.Event()  # gate to discard echo (set when speaker active)


def mic_cb(indata, frames, t, status):
    if status:
        return
    try:
        q_mic.put_nowait(indata[:, 0].copy())
    except queue.Full:
        # Drop oldest by draining one
        try:
            q_mic.get_nowait()
            q_mic.put_nowait(indata[:, 0].copy())
        except Exception:
            pass


def play_cb(outdata, frames, t, status):
    try:
        chunk = q_play.get_nowait()
        n = min(len(chunk), frames)
        outdata[:n, 0] = chunk[:n]
        if n < frames:
            outdata[n:, 0] = 0
        tts_evt.set()
        vad_gate.set()
    except queue.Empty:
        outdata[:, 0] = 0
        # Only clear after a small idle so transient gaps don't flap
        if not q_play._qsize_check() if hasattr(q_play, "_qsize_check") else True:
            tts_evt.clear()
            vad_gate.clear()


# Patch queue to expose _qsize_check behind a clean attribute
def _qsize(self):
    return self.qsize()
queue.Queue._qsize_check = _qsize  # type: ignore[attr-defined]


def _barge_in():
    """Stop current TTS playback — flush the play queue."""
    while True:
        try:
            q_play.get_nowait()
        except queue.Empty:
            break
    tts_evt.clear()
    vad_gate.clear()


# ─── Pipeline ─────────────────────────────────────────────────────────────
def process_segment(seg: np.ndarray) -> None:
    dur = len(seg) / MIC_RATE
    print(f"\n[seg {dur:.1f}s] tts_active={tts_evt.is_set()}", flush=True)

    # Barge-in: if TTS is playing, drop it before doing new work
    if tts_evt.is_set():
        _barge_in()

    # STT
    t0 = time.time()
    try:
        text = transcribe(seg)
    except Exception as e:
        print(f"  [stt error] {e}")
        return
    print(f"  [stt {time.time()-t0:.1f}s] {text!r}", flush=True)
    if not text:
        return

    # LLM
    t0 = time.time()
    try:
        reply = llm_respond(text)
    except Exception as e:
        print(f"  [llm error] {e}")
        return
    if not reply:
        return

    # TTS
    t0 = time.time()
    try:
        audio = tts_synthesize(reply)
    except Exception as e:
        print(f"  [tts error] {e}")
        return
    print(f"  [tts {time.time()-t0:.1f}s] {len(audio)/TTS_RATE:.1f}s audio",
          flush=True)
    if audio.size > 0:
        q_play.put(audio)
        tts_evt.set()
        vad_gate.set()


def live_loop() -> None:
    # Preload models on main thread so the live loop doesn't stall
    print("[boot] warming up...")
    _load_whisper()
    _load_llm()
    _ensure_tts()
    print("[boot] ready — speak. Ctrl+C to stop.")

    buf = np.zeros(0, dtype=np.float32)
    silence_runs = 0
    talking = False
    SILENCE_RUNS_END = int(0.7 * 1000 / CHUNK_MS)  # 700ms of silence to end
    MIN_TALK_SEC = 0.5

    with sd.InputStream(samplerate=MIC_RATE, channels=1, dtype="float32",
                        blocksize=CHUNK, callback=mic_cb), \
         sd.OutputStream(samplerate=TTS_RATE, channels=1, dtype="float32",
                         blocksize=2048, callback=play_cb):
        try:
            while not stop_evt.is_set():
                try:
                    chunk = q_mic.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Echo gate: if speaker playing, raise VAD threshold
                base_thr = 0.01
                thr = base_thr * (ECHO_GAIN if vad_gate.is_set() else 1.0)
                speech = is_speech_rms(chunk, threshold=thr)

                buf = np.concatenate([buf, chunk])[-MIC_RATE * 8:]

                if speech:
                    silence_runs = 0
                    if not talking and len(buf) >= int(MIC_RATE * MIN_TALK_SEC):
                        talking = True
                else:
                    if talking:
                        silence_runs += 1
                        if silence_runs >= SILENCE_RUNS_END:
                            seg = buf.copy()
                            buf = np.zeros(0, dtype=np.float32)
                            silence_runs = 0
                            talking = False
                            process_segment(seg)
        except KeyboardInterrupt:
            stop_evt.set()


def demo() -> None:
    print("[demo] warmup + single turn")
    _load_whisper()
    _load_llm()
    _ensure_tts()
    print("[demo] generating sample turn...")
    prompt = "Hola, ¿cómo estás? Respondé muy breve en español."
    reply = llm_respond(prompt)
    print(f"[demo] reply: {reply!r}")
    if not reply:
        return
    audio = tts_synthesize(reply)
    if audio.size:
        sf.write("/tmp/michi_demo.wav", audio, TTS_RATE)
        print(f"[demo] saved /tmp/michi_demo.wav ({len(audio)/TTS_RATE:.1f}s)")
        try:
            sd.play(audio, TTS_RATE)
            sd.wait()
        except Exception as e:
            print(f"[demo] playback failed: {e}")


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Michi-Lite voice agent")
    p.add_argument("--mode", choices=["demo", "mic"], default="mic")
    args = p.parse_args()

    _banner()
    if args.mode == "demo":
        demo()
    else:
        signal.signal(signal.SIGINT, lambda *_: stop_evt.set())
        try:
            live_loop()
        except KeyboardInterrupt:
            stop_evt.set()
    print("\n[exit] bye")


if __name__ == "__main__":
    main()
