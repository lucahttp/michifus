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
import ctypes
import numpy as np
import sounddevice as sd
import soundfile as sf

# Audio defaults
MIC_RATE = 16000
TTS_RATE = 24000
CHUNK_MS = 32
CHUNK = int(MIC_RATE * CHUNK_MS / 1000)  # 512 samples @ 16kHz
PLAY_BLOCK = 2048  # output stream blocksize for full-duplex chunking

# Wire supertonic-3 MLX package shipped via HF Space (no pip build yet).
import sys as _sys
_SUPERTONIC_DIR = os.path.expanduser(
    "~/.cache/michi/spaces--mlx-community--supertonic-3/snapshots"
)
if os.path.isdir(_SUPERTONIC_DIR):
    for d in os.listdir(_SUPERTONIC_DIR):
        sub = os.path.join(_SUPERTONIC_DIR, d, "supertonic_mlx")
        if os.path.isdir(sub) and sub not in _sys.path:
            _sys.path.insert(0, os.path.dirname(sub))
            break
del _sys

# Config from env (no hardcoded paths)
ENCODER_TYPE = os.environ.get("MICHI_ENCODER", "parakeet").lower()  # parakeet|whisper
PARAKEET_MLX_REPO = os.environ.get(
    "MICHI_STT", "mlx-community/parakeet-tdt-0.6b-v3"
)
WHISPER_REPO = os.environ.get(
    "MICHI_WHISPER", "openai/whisper-small"
)
DUPLEX_CKPT = os.environ.get(
    "MICHI_DUPLEX_CKPT",  # trained projectors from colab_train_duplex_lite
    os.path.expanduser("~/michi-adapter/checkpoints/duplex_phase3.pt")
)
LLM_REPO = os.environ.get("MICHI_LLM", "openbmb/MiniCPM5-1B-MLX")
SUPERTONIC_DIR = os.environ.get(
    "MICHI_TTS",
    os.path.expanduser(
        "~/.cache/michi/models--mlx-community--supertonic-3/snapshots"
    ),
)
TTS_VOICE = os.environ.get("MICHI_TTS_VOICE", "M1")
TTS_LANG = os.environ.get("MICHI_TTS_LANG", "en")
SYSTEM_PROMPT = os.environ.get(
    "MICHI_SYSTEM",
    "You are Michi, a voice assistant. Respond in English, brief and "
    "conversational. Don't make up data; if unsure, say so.",
)
LLM_MAX_TOKENS = int(os.environ.get("MICHI_MAX_TOKENS", "80"))
FULL_DUPLEX = os.environ.get("MICHI_FULL_DUPLEX", "1") == "1"
ECHO_GAIN = float(os.environ.get("MICHI_ECHO_GAIN", "3.0"))  # VAD x-mult when speaker active

_state = {"encoder": None, "llm": None, "tok": None, "tts": None, "tts_style": None,
          "proj": None, "proc": None, "encoder_kind": None}


def _banner():
    print("=" * 60)
    print("Michi-Lite — full-duplex voice agent (lightweight)")
    print("=" * 60)
    if ENCODER_TYPE == "whisper":
        print(f"  STT: Whisper-small HF @ {WHISPER_REPO} (+ optional projectors)")
    else:
        print(f"  STT: parakeet-tdt-0.6b MLX @ {PARAKEET_MLX_REPO}")
    print(f"  LLM: {LLM_REPO}")
    print(f"  TTS: supertonic-3 MLX  voice={TTS_VOICE}  lang={TTS_LANG}")
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


# ─── STT (switchable: parakeet-mlx or Whisper-small HF) ───────────────────
def _load_parakeet():
    """MLX-native Parakeet-TDT 0.6B (v3 by default)."""
    if _state.get("encoder") is None or _state.get("encoder_kind") != "parakeet":
        from parakeet_mlx import from_pretrained as parakeet_from_pretrained
        print(f"  [stt] loading parakeet-tdt MLX: {PARAKEET_MLX_REPO}")
        t0 = time.time()
        mdl = parakeet_from_pretrained(PARAKEET_MLX_REPO)
        print(f"  [stt] loaded {type(mdl).__name__} in {time.time()-t0:.1f}s")
        _state["encoder"] = mdl
        _state["encoder_kind"] = "parakeet"
    return _state["encoder"]


def _load_whisper():
    """HF Whisper-small + optional trained projectors from duplex_lite checkpoint."""
    if _state.get("encoder") is None or _state.get("encoder_kind") != "whisper":
        import torch
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        print(f"  [stt] loading Whisper HF: {WHISPER_REPO}")
        t0 = time.time()
        proc = WhisperProcessor.from_pretrained(WHISPER_REPO)
        mdl = WhisperForConditionalGeneration.from_pretrained(WHISPER_REPO)
        # Force Spanish + no timestamps
        mdl.config.forced_decoder_ids = proc.get_decoder_prompt_ids(
            language="es", task="transcribe")
        mdl.eval()
        # MPS if available, else CPU (M1 unified memory works)
        try:
            if torch.backends.mps.is_available():
                mdl = mdl.to("mps")
                print(f"  [stt] Whisper on MPS", flush=True)
        except Exception:
            pass
        print(f"  [stt] loaded in {time.time()-t0:.1f}s")
        _state["encoder"] = mdl
        _state["proc"] = proc
        _state["encoder_kind"] = "whisper"

        # Optional: load trained projectors (Whisper → LLM hidden space)
        if os.path.exists(DUPLEX_CKPT):
            try:
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                from duplex_lite import WhisperToLLM, LLMToTTS, LLM_CONFIGS
                ck = torch.load(DUPLEX_CKPT, map_location="cpu", weights_only=False)
                proj_in = WhisperToLLM(ck["whisper_dim"], ck["llm_dim"])
                proj_in.load_state_dict(ck["audio_proj"])
                proj_in.eval()
                _state["proj"] = proj_in
                print(f"  [stt] loaded trained projectors from {DUPLEX_CKPT}",
                      flush=True)
            except Exception as e:
                print(f"  [stt] projector load failed (using plain Whisper): {e}",
                      flush=True)
                _state["proj"] = None
        else:
            _state["proj"] = None
    return _state["encoder"]


def _load_stt():
    """Factory: returns whichever encoder MICHI_ENCODER selects."""
    if ENCODER_TYPE == "whisper":
        return _load_whisper()
    return _load_parakeet()


def transcribe(pcm: np.ndarray) -> str:
    audio = pcm.astype(np.float32, copy=False)
    peak = float(np.max(np.abs(audio))) or 1.0
    if peak > 1.0:
        audio = audio / peak

    if ENCODER_TYPE == "whisper":
        return _transcribe_whisper(audio)
    return _transcribe_parakeet(audio)


def _transcribe_parakeet(audio: np.ndarray) -> str:
    mdl = _load_parakeet()
    fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="michi_in_")
    try:
        sf.write(tmp, audio, MIC_RATE)
        result = mdl.transcribe(tmp, dtype="float32")
    finally:
        try: os.close(fd); os.unlink(tmp)
        except OSError: pass
    if isinstance(result, str):
        return result.strip()
    chunks = []
    if hasattr(result, "sentences"):
        for s in result.sentences:
            t = getattr(s, "text", "") or ""
            if t: chunks.append(t)
    if chunks:
        return " ".join(chunks).strip()
    return (getattr(result, "text", "") or "").strip()


def _transcribe_whisper(audio: np.ndarray) -> str:
    import torch
    mdl = _load_whisper()
    proc = _state["proc"]
    device = next(mdl.parameters()).device
    inputs = proc(audio, sampling_rate=MIC_RATE, return_tensors="pt")
    input_features = inputs.input_features.to(device)
    with torch.no_grad():
        gen = mdl.generate(input_features, max_new_tokens=200,
                            num_beams=1, do_sample=False)
    text = proc.batch_decode(gen, skip_special_tokens=True)[0]
    return text.strip()


# ─── LLM (mlx-lm) ──────────────────────────────────────────────────────────
def _load_llm():
    if _state["llm"] is None:
        from mlx_lm import load as lm_load
        print(f"  [llm] loading {LLM_REPO} (mlx-lm)...")
        t0 = time.time()
        model, tok = lm_load(LLM_REPO)
        print(f"  [llm] loaded in {time.time()-t0:.1f}s")
        _state["llm"] = model
        _state["tok"] = tok
    return _state["llm"], _state["tok"]


_chat_history: list[dict] = []
_HISTORY_TURNS = 2  # keep prompt short for fast inference


def llm_respond_stream(user_text: str) -> tuple[str, float]:
    """Returns (reply_text, first_token_ms). Returns ("", 0) on empty input."""
    if not user_text.strip():
        return "", 0.0
    from mlx_lm import stream_generate
    model, tok = _load_llm()

    _chat_history.append({"role": "user", "content": user_text})
    history = _chat_history[-(_HISTORY_TURNS * 2):]
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + history
    prompt = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    t0 = time.time()
    first_ms = 0.0
    out = ""
    for chunk in stream_generate(model, tok, prompt, max_tokens=LLM_MAX_TOKENS):
        s = chunk.text
        if first_ms == 0.0:
            first_ms = (time.time() - t0) * 1000
        out += s
        if out.endswith(("<end_of_turn>", "<eos>")):
            break
    reply = (out
             .replace("<end_of_turn>", "").replace("<eos>", "")
             .replace("<|message|>", "").replace("<|channel|>", "")
             .strip())
    _chat_history.append({"role": "assistant", "content": reply})
    return reply, first_ms


# Backwards-compat alias used by demo()
def llm_respond(user_text: str) -> str:
    reply, _ = llm_respond_stream(user_text)
    return reply


# ─── TTS (supertonic-3 MLX) ───────────────────────────────────────────────
def _resolve_supertonic_dir():
    base = SUPERTONIC_DIR
    if os.path.basename(base) == "snapshots" and os.path.isdir(base):
        for d in os.listdir(base):
            sub = os.path.join(base, d)
            if os.path.isdir(sub) and os.path.exists(
                os.path.join(sub, "tts.json")
            ):
                base = sub
                break
    if not os.path.exists(os.path.join(base, "tts.json")):
        raise FileNotFoundError(
            f"supertonic-3 model dir missing tts.json: {base}\n"
            "Set MICHI_TTS to the snapshot dir.")
    return base


def _load_tts():
    if _state.get("tts") is None:
        from supertonic_mlx import SupertonicMLX
        mdir = _resolve_supertonic_dir()
        print(f"  [tts] loading supertonic-3 MLX from {mdir}")
        t0 = time.time()
        tts = SupertonicMLX.from_pretrained(mdir)
        print(f"  [tts] loaded {tts.sample_rate}Hz in {time.time()-t0:.1f}s")
        _state["tts"] = tts
        _state["tts_style"] = tts.get_voice_style(TTS_VOICE)
    return _state["tts"], _state["tts_style"]


def tts_synthesize(text: str) -> np.ndarray:
    text = (text or "").strip()
    if not text:
        return np.zeros(0, dtype=np.float32)
    tts, style = _load_tts()
    wav, dur = tts.synthesize(text, TTS_LANG, style)
    # wav: (1, total_samples); dur: array of per-utterance seconds
    if wav.ndim == 2:
        wav = wav[0]
    sample_rate = tts.sample_rate
    n = int(sample_rate * float(dur[0]))
    audio = wav[:n].astype(np.float32)
    if sample_rate != TTS_RATE:
        from scipy.signal import resample
        ratio = TTS_RATE / sample_rate
        audio = resample(audio, int(round(len(audio) * ratio))).astype(np.float32)
    return audio


# ─── AEC (libspeexdsp via ctypes — wheels no compilan para M1) ────────────
_AEC_FRAME = 160       # 10ms @ 16kHz
_AEC_TAPS  = 1024      # ~64ms tail
_AEC_LIBM = None
_AEC_AVAILABLE = False
try:
    _speex_lib = "/opt/homebrew/opt/speexdsp/lib/libspeexdsp.dylib"
    _AEC_LIBM = ctypes.CDLL(_speex_lib)
    _AEC_LIBM.speex_echo_state_init.argtypes = [ctypes.c_int, ctypes.c_int]
    _AEC_LIBM.speex_echo_state_init.restype  = ctypes.c_void_p
    _AEC_LIBM.speex_echo_state_destroy.argtypes = [ctypes.c_void_p]
    _AEC_LIBM.speex_echo_state_destroy.restype  = None
    _AEC_LIBM.speex_echo_cancellation.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int16),
        ctypes.POINTER(ctypes.c_int16),
        ctypes.POINTER(ctypes.c_int16),
    ]
    _AEC_LIBM.speex_echo_cancellation.restype  = None
    _AEC_AVAILABLE = True
except Exception as _e:
    print(f"  [aec] libspeexdsp not loaded: {_e}", flush=True)


def _f2i(f32: np.ndarray) -> np.ndarray:
    """float32 [-1,1] → int16 (clipping)."""
    return np.clip(f32, -1.0, 1.0).astype(np.float32).view(np.int16) if False \
        else (f32 * 32767.0).clip(-32768, 32767).astype(np.int16)


def _i2f(i16: np.ndarray) -> np.ndarray:
    return i16.astype(np.float32) / 32767.0


class AECProcessor:
    """Speex AEC int16 wrapper. Threads: mic_cb (writer) + TTSWorker (writer).
    Reads on demand from process() — caller is the live_loop."""

    def __init__(self):
        self._st = None
        if _AEC_AVAILABLE:
            try:
                self._st = _AEC_LIBM.speex_echo_state_init(_AEC_FRAME, _AEC_TAPS)
            except Exception as e:
                print(f"  [aec] init failed: {e}", flush=True)
        self._ref_buf  = np.zeros(0, dtype=np.int16)   # TTS reference (speaker)
        self._fwd_buf  = np.zeros(0, dtype=np.int16)   # mic forward
        self._out_buf  = np.zeros(0, dtype=np.float32) # processed leftovers
        self._lock = threading.Lock()

    def push_ref(self, f32_1d: np.ndarray) -> None:
        """Called from TTSWorker before each audio block is queued for play."""
        if not _AEC_AVAILABLE or self._st is None or f32_1d.size == 0:
            return
        i16 = _f2i(f32_1d)
        with self._lock:
            self._ref_buf = np.concatenate([self._ref_buf, i16])

    def process(self, f32_1d: np.ndarray) -> np.ndarray:
        """Forward mic samples. Returns any processed output that's ready."""
        if not _AEC_AVAILABLE or self._st is None or f32_1d.size == 0:
            return f32_1d
        i16 = _f2i(f32_1d)
        with self._lock:
            self._fwd_buf = np.concatenate([self._fwd_buf, i16])
            n_full = (len(self._fwd_buf) // _AEC_FRAME) * _AEC_FRAME
            if n_full == 0:
                return np.zeros(0, dtype=np.float32)
            mic_full = self._fwd_buf[:n_full]
            self._fwd_buf = self._fwd_buf[n_full:]
            # Pad reference if too short
            if len(self._ref_buf) < n_full:
                pad = np.zeros(n_full - len(self._ref_buf), dtype=np.int16)
                ref_full = np.concatenate([self._ref_buf, pad])
            else:
                ref_full = self._ref_buf[:n_full]
                self._ref_buf = self._ref_buf[n_full:]
            out_i16 = np.zeros(n_full, dtype=np.int16)
        # Speex cancels N×160 in-place. Loop frame-by-frame (cheap, ~64 frames/sec).
        for off in range(0, n_full, _AEC_FRAME):
            _AEC_LIBM.speex_echo_cancellation(
                self._st,
                mic_full[off:off+_AEC_FRAME].ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                ref_full[off:off+_AEC_FRAME].ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                out_i16[off:off+_AEC_FRAME].ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            )
        out_f = _i2f(out_i16)
        if self._out_buf.size:
            out_f = np.concatenate([self._out_buf, out_f])
            self._out_buf = np.zeros(0, dtype=np.float32)
        return out_f

    def close(self):
        if self._st is not None:
            _AEC_LIBM.speex_echo_state_destroy(self._st)
            self._st = None


# Resample helper (TTS 24k → mic 16k) for the AEC reference stream.
try:
    from scipy.signal import resample_poly
    def _resample_to_mic(audio: np.ndarray, src_rate: int, dst_rate: int = MIC_RATE) -> np.ndarray:
        from math import gcd
        g = gcd(src_rate, dst_rate)
        return resample_poly(audio, dst_rate // g, src_rate // g).astype(np.float32)
except Exception:
    def _resample_to_mic(audio: np.ndarray, src_rate: int, dst_rate: int = MIC_RATE) -> np.ndarray:
        # Linear fallback
        ratio = dst_rate / src_rate
        n_new = int(round(len(audio) * ratio))
        x_old = np.linspace(0, 1, len(audio))
        x_new = np.linspace(0, 1, n_new)
        return np.interp(x_new, x_old, audio).astype(np.float32)


# ─── Full-duplex plumbing ──────────────────────────────────────────────────
q_mic = queue.Queue(maxsize=400)
q_play = queue.Queue()
stop_evt = threading.Event()
tts_evt = threading.Event()  # set while TTS audio is in speaker queue
vad_gate = threading.Event()  # gate to discard echo (set when speaker active)
aec = AECProcessor()


def mic_cb(indata, frames, t, status):
    if status:
        return
    raw = indata[:, 0].copy()
    # AEC forward stream — always run (cheap; speex only adapts when ref present)
    cleaned = aec.process(raw)
    if cleaned.size == 0:
        # Not enough samples yet for a full frame; queue raw
        cleaned = raw
    try:
        q_mic.put_nowait(cleaned)
    except queue.Full:
        try:
            q_mic.get_nowait()
            q_mic.put_nowait(cleaned)
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
        # Only clear when queue fully drained — avoids flapping on gaps.
        if q_play.qsize() == 0:
            tts_evt.clear()
            vad_gate.clear()


def _barge_in():
    """Stop current TTS playback — flush the play queue."""
    while True:
        try:
            q_play.get_nowait()
        except queue.Empty:
            break
    tts_evt.clear()
    vad_gate.clear()


# ─── TTS worker (so synth never stalls the live loop) ──────────────────────
class TTSWorker(threading.Thread):
    """Off-main-thread synthesis. Streams fixed-size play blocks into q_play
    so the OutputStream callback can keep draining without blocking."""

    def __init__(self):
        super().__init__(daemon=True, name="michi-tts")
        self.q = queue.Queue()
        self._stop = threading.Event()

    def push(self, text: str, meta: dict | None = None):
        self.q.put(("synth", (text, meta or {})))

    def stop(self):
        self.q.put(("stop", None))

    def run(self):
        # MLX requires an explicit new_stream on each device for every thread.
        try:
            import mlx.core as _mx
            for _dev in ("cpu", "gpu"):
                try:
                    _stream = _mx.new_stream(getattr(_mx, _dev))
                    _mx.set_default_stream(_stream)
                except Exception:
                    pass
        except Exception:
            pass

        while not self._stop.is_set():
            try:
                tag, data = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            if tag == "stop":
                break
            text, meta = data if isinstance(data, tuple) else (data, {})
            # Lazy-load TTS on this worker thread
            if _state["tts"] is None:
                print(f"  [boot-tts] load...", flush=True)
                _load_tts()
            t0 = time.time()
            try:
                audio = tts_synthesize(text)
            except Exception as e:
                print(f"  [tts error] {e}", flush=True)
                continue
            synth_ms = (time.time() - t0) * 1000
            if audio.size == 0:
                continue
            # E2E: from "user stopped speaking" to "audio queued for play"
            speech_end = meta.get("speech_end")
            if speech_end is not None:
                e2e_ms = (time.time() - speech_end) * 1000
                print(f"  [e2e] user→audio {e2e_ms:.0f}ms "
                      f"(synth {synth_ms:.0f}ms {len(audio)/TTS_RATE:.1f}s)",
                      flush=True)
            print(f"  [tts {synth_ms:.0f}ms] "
                  f"{len(audio)/TTS_RATE:.1f}s queued", flush=True)
            if audio.size == 0:
                continue
            print(f"  [tts {time.time()-t0:.1f}s] "
                  f"{len(audio)/TTS_RATE:.1f}s audio", flush=True)
            # Chunk into PLAY_BLOCK so each play_cb gets exactly one frame;
            # rest of audio actually plays instead of being truncated.
            for off in range(0, len(audio), PLAY_BLOCK):
                block = audio[off:off + PLAY_BLOCK]
                if len(block) < PLAY_BLOCK:
                    pad = np.zeros(PLAY_BLOCK - len(block), dtype=np.float32)
                    block = np.concatenate([block, pad])
                # Feed AEC reference (TTS audio in 24kHz → resample to 16k for speex)
                try:
                    ref16 = _resample_to_mic(block, TTS_RATE, MIC_RATE)
                    aec.push_ref(ref16)
                except Exception:
                    pass
                q_play.put(block)


class PipelineWorker(threading.Thread):
    """Off-main-thread STT+LLM so the live loop never stalls on inference.
    Each segment runs: parakeet transcribe -> llm respond -> tts push."""

    def __init__(self, tts_worker: "TTSWorker"):
        super().__init__(daemon=True, name="michi-pipe")
        self.q = queue.Queue()
        self.tts_worker = tts_worker
        self._stop = threading.Event()

    def push_audio(self, seg: np.ndarray, meta: dict | None = None):
        self.q.put(("seg", (seg, meta or {})))

    def stop(self):
        self.q.put(("stop", None))

    def run(self):
        # MLX requires an explicit new_stream on each device for every
        # thread. parakeet uses CPU stream; gemma LLM uses gpu stream.
        try:
            import mlx.core as _mx
            for _dev in ("cpu", "gpu"):
                try:
                    _stream = _mx.new_stream(getattr(_mx, _dev))
                    _mx.set_default_stream(_stream)
                except Exception:
                    pass
        except Exception:
            pass

        while not self._stop.is_set():
            try:
                tag, data = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            if tag == "stop":
                break
            try:
                seg, meta = data if isinstance(data, tuple) else (data, {})
                self._process(seg, meta)
            except Exception as e:
                print(f"  [pipe error] {e}", flush=True)

    def _process(self, seg: np.ndarray, meta: dict | None = None):
        meta = meta or {}
        t_speech_end = meta.get("t_speech_end", time.time())

        # Lazy-load models on this worker thread (MLX requires the model
        # + its streams live on the same thread that will run inference).
        if _state["encoder"] is None:
            print(f"  [boot-stt] load...", flush=True)
            _load_parakeet()
        if _state["llm"] is None:
            print(f"  [boot-llm] load...", flush=True)
            _load_llm()
        print(f"  [e2e] pickup={time.time()-t_speech_end:.2f}s", flush=True)

        # Barge-in: drop pending TTS if user started a new turn
        if tts_evt.is_set():
            _barge_in()

        # STT
        t0 = time.time()
        text = transcribe(seg)
        stt_ms = (time.time() - t0) * 1000
        if not text:
            return
        print(f"  [stt {stt_ms:.0f}ms] {text!r}", flush=True)

        # LLM (stream so we measure first-token latency)
        t0 = time.time()
        reply, first_ms = llm_respond_stream(text)
        llm_total_ms = (time.time() - t0) * 1000
        if not reply:
            return
        print(f"  [llm] first {first_ms:.0f}ms / total {llm_total_ms:.0f}ms "
              f"len={len(reply)} -> tts", flush=True)

        # Hand off to TTS worker
        self.tts_worker.push(reply, {"t_llm_done": time.time(),
                                     "speech_end": t_speech_end,
                                     "reply_len": len(reply)})
        tts_evt.set()
        vad_gate.set()


tts_worker: TTSWorker | None = None
pipeline_worker: PipelineWorker | None = None


# ─── Pipeline ─────────────────────────────────────────────────────────────
def process_segment(seg: np.ndarray) -> None:
    """Thin shim — pushes audio to PipelineWorker (off-thread STT+LLM)."""
    dur = len(seg) / MIC_RATE
    print(f"\n[seg {dur:.1f}s] tts_active={tts_evt.is_set()}", flush=True)
    if pipeline_worker is not None:
        pipeline_worker.push_audio(seg,
            meta={"t_speech_end": time.time()})


def live_loop() -> None:
    print("[boot] starting workers (models load on first use)...")

    global tts_worker, pipeline_worker
    tts_worker = TTSWorker()
    tts_worker.start()
    pipeline_worker = PipelineWorker(tts_worker)
    pipeline_worker.start()
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
    if pipeline_worker is not None:
        pipeline_worker.stop()
        pipeline_worker.join(timeout=3.0)
    if tts_worker is not None:
        tts_worker.stop()
        tts_worker.join(timeout=3.0)


def demo() -> None:
    print("[demo] warmup + single turn")
    _load_parakeet()
    _load_llm()
    _load_tts()
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
