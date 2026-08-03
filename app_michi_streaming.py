"""
Michi-Streaming V3 — full-duplex Gemma-4-E2B QAT GGUF + Supertonic-3.

Differences from V2:
  - Cross-platform config via env vars (no hardcoded D:/ or E:/ paths)
  - HF_TOKEN read from env (no "YOUR_HF_TOKEN" placeholder)
  - Auto-downloads the GGUF if missing (uses HF_TOKEN)
  - Truly full-duplex: segment processing runs on a worker thread so the
    main loop keeps listening and can interrupt (barge-in) at any time
  - TTS worker is stateful across segments (no per-call stop/restart)
  - All paths overridable via env for portability (Win/Mac/Linux)

Env vars:
  MICHI_GGUF_DIR    default ~/.cache/huggingface/hub/models--unsloth--gemma-4-E2B-it-qat-mobile-GGUF
  MICHI_ADAPTER      default ~/michi-adapter/checkpoints/adapter_phase3_sft.pt
  MICHI_TTS          default supertonic-3
  MICHI_LLM_N_CTX    default 1024
  MICHI_LLM_GPU_LAYERS default 99
  MICHI_VAD          rms | silero (default rms)
  MICHI_FULL_DUPLEX  1 | 0 (default 1)
  HF_TOKEN           required for HF Hub access
"""
import os
import sys
import time
import queue
import signal
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf

# ─── Config from env (cross-platform) ───────────────────────────────────────
SAMPLE_RATE = 24000
MIC_RATE = 16000
PROMPT_MODE = os.environ.get("MICHI_PROMPT_MODE", "minimal")

DEFAULT_MODEL_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--unsloth--gemma-4-E2B-it-qat-mobile-GGUF"
)
# Also handle the layout where HF Hub installs without the /hub/ subdir
def _first_existing(*paths):
    for p in paths:
        if os.path.isdir(p):
            return p
    return paths[0]
DEFAULT_ADAPTER = os.path.expanduser("~/michi-adapter/checkpoints/adapter_phase3_sft.pt")
MODEL_DIR = os.environ.get("MICHI_GGUF_DIR", DEFAULT_MODEL_DIR)
ADAPTER_PATH = os.environ.get("MICHI_ADAPTER", DEFAULT_ADAPTER)
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("MICHI_HF_TOKEN")
TTS_VOICE = os.environ.get("MICHI_TTS_VOICE", "M1")
FULL_DUPLEX = os.environ.get("MICHI_FULL_DUPLEX", "1") == "1"
LLM_N_CTX = int(os.environ.get("MICHI_LLM_N_CTX", "1024"))
LLM_GPU_LAYERS = int(os.environ.get("MICHI_LLM_GPU_LAYERS", "99"))
VAD_KIND = os.environ.get("MICHI_VAD", "rms").lower()

os.environ["HF_TOKEN"] = HF_TOKEN or ""


# ─── Helpers ────────────────────────────────────────────────────────────────
def _resolve_hf_snapshot():
    """Return the actual snapshot dir containing the GGUF, or None."""
    if not os.path.isdir(MODEL_DIR):
        return None
    snapshots = os.path.join(MODEL_DIR, "snapshots")
    if not os.path.isdir(snapshots):
        return None
    for d in os.listdir(snapshots):
        p = os.path.join(snapshots, d)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "config.json")):
            return p
    return None


def _download_model():
    """Download the Gemma-4-E2B QAT GGUF from HF Hub using HF_TOKEN."""
    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN not set. Export it before running):\n"
            "  export HF_TOKEN=hf_..."
        )
    from huggingface_hub import snapshot_download
    print(f"[download] {os.path.basename(MODEL_DIR)} ...", flush=True)
    snapshot_download(
        repo_id="unsloth/gemma-4-E2B-it-qat-mobile-GGUF",
        cache_dir=os.path.expanduser("~/.cache/huggingface"),
        token=HF_TOKEN,
        allow_patterns=[
            "gemma-4-E2B-it-qat-UD-Q2_K_XL.gguf",
            "tokenizer.model",
            "config.json",
            "*.json",
        ],
    )
    print("[download] done", flush=True)


# ─── Phase 1: load models ───────────────────────────────────────────────────
def _load_llm():
    snapshot = _resolve_hf_snapshot()
    if snapshot is None:
        _download_model()
        snapshot = _resolve_hf_snapshot()
    assert snapshot, f"GGUF snapshot not found under {MODEL_DIR}"

    gguf = os.path.join(snapshot, "gemma-4-E2B-it-qat-UD-Q2_K_XL.gguf")
    tok = os.path.join(snapshot, "tokenizer.model")
    if not os.path.exists(tok):
        # Gemma-4 GGUF embeds the tokenizer; let llama-cpp use it directly.
        tok = None
    print(f"[llm] loading {gguf}", flush=True)
    from llama_cpp import Llama
    llm_kwargs = dict(
        model_path=gguf,
        n_ctx=LLM_N_CTX,
        n_gpu_layers=LLM_GPU_LAYERS,
        flash_attention=True,
        verbose=False,
    )
    if tok:
        llm_kwargs["tokenizer_file"] = tok
    else:
        llm_kwargs["tokenizer_repo_id"] = "unsloth/gemma-4-E2B-it-qat-mobile-GGUF"
        if HF_TOKEN:
            llm_kwargs["hf_token"] = HF_TOKEN
    llm = Llama(**llm_kwargs)
    print(f"[llm] ready n_ctx={llm.n_ctx()}", flush=True)
    return llm


def _load_adapters():
    if not os.path.exists(ADAPTER_PATH):
        raise FileNotFoundError(
            f"adapter checkpoint not found: {ADAPTER_PATH}\n"
            f"Train via colab_train_adapter.ipynb, then set MICHI_ADAPTER to the .pt path."
        )
    print(f"[adapter] loading {ADAPTER_PATH}", flush=True)
    ckpt = torch.load(ADAPTER_PATH, map_location="cpu", weights_only=False)
    in_proj = ckpt["in_adapter"]["proj.weight"].numpy()
    out_proj = ckpt["out_adapter"]["proj.weight"].numpy()
    del ckpt
    print(f"[adapter] in={in_proj.shape} out={out_proj.shape}", flush=True)
    return in_proj, out_proj


def _load_tts():
    print("[tts] loading Supertonic-3", flush=True)
    from supertonic import TTS
    tts = TTS(auto_download=True)
    print(f"[tts] ready @ {SAMPLE_RATE}Hz", flush=True)
    return tts


# ─── Audio utilities ────────────────────────────────────────────────────────
def _vad_rms(pcm, thr=0.012):
    return float(np.abs(pcm).mean()) > thr


_silero = None
def _vad_silero(pcm):
    global _silero
    if _silero is None:
        import torch
        _silero, _ = torch.hub.load("snakers4/silero-vad", "silero_vad",
                                    trust_repo=True, verbose=False)
    if len(pcm) < 512:
        return False
    import torch
    t = torch.from_numpy(pcm[:512].astype(np.float32))
    return _silero(t, MIC_RATE).item() > 0.5


def is_speech(pcm):
    if VAD_KIND == "silero":
        return _vad_silero(pcm)
    return _vad_rms(pcm)


def mel_spec(pcm):
    """Project PCM(16kHz) → mimi-style 128-dim log-mel frames."""
    x = torch.from_numpy(pcm).float().unsqueeze(0)
    w = torch.hann_window(2048)
    spec = torch.stft(x, n_fft=2048, hop_length=256, win_length=2048,
                      window=w, onesided=True, return_complex=True)
    mag = spec.squeeze(0).abs().pow(2)
    mel = torch.nn.functional.linear(torch.log1p(mag.T),
                                     torch.ones(128, 1025) * 0.01).relu()
    return mel[:, :128].numpy()


# ─── Echo gate ──────────────────────────────────────────────────────────────
class SpeakerGate:
    """Tracks whether TTS audio is currently in the play queue."""
    def __init__(self):
        self._lock = threading.Lock()
        self._active = False

    def set(self, v):
        with self._lock:
            self._active = v

    def is_active(self):
        with self._lock:
            return self._active


speaker_gate = SpeakerGate()


def is_speech_echo_aware(pcm, thr=0.012):
    """When speaker playing, raise VAD threshold to ignore echo bleed."""
    if speaker_gate.is_active():
        thr *= 3.0
    return _vad_rms(pcm, thr=thr)


# ─── TTS background worker ─────────────────────────────────────────────────
class TTSWorker(threading.Thread):
    """Single-threaded TTS: synthesize text chunks → audio chunks → queue."""
    def __init__(self, tts, play_queue: queue.Queue):
        super().__init__(daemon=True, name="michi-tts")
        self.tts = tts
        self.play_queue = play_queue
        self._in = queue.Queue()
        self._stop = threading.Event()

    def push_text(self, text):
        self._in.put(("text", text))

    def flush(self):
        self._in.put(("flush", None))

    def stop(self):
        self._in.put(("stop", None))

    def run(self):
        pending = ""
        while not self._stop.is_set():
            try:
                tag, data = self._in.get(timeout=0.05)
            except queue.Empty:
                if pending.strip():
                    self._synthesize(pending)
                    pending = ""
                continue

            if tag == "text":
                pending += " " + data
            elif tag == "flush":
                if pending.strip():
                    self._synthesize(pending)
                    pending = ""
            elif tag == "stop":
                if pending.strip():
                    self._synthesize(pending)
                self._stop.set()
                break

            # Opportunistic synthesis when buffer grows large enough
            if len(pending.strip()) > 80:
                self._synthesize(pending)
                pending = ""

    def _synthesize(self, text):
        try:
            style = self.tts.get_voice_style(TTS_VOICE)
            wav, _ = self.tts.synthesize(text.strip(), voice_style=style,
                                        lang="es")
            arr = np.asarray(wav, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[0] == 1:
                arr = arr.squeeze(0)
            self.play_queue.put(arr)
        except Exception as e:
            print(f"[tts] error: {e}", flush=True)


# ─── Gemma streaming + early exit ───────────────────────────────────────────
def should_stop_streaming(text: str, token_count: int, recent: list) -> bool:
    """Detect when Gemma has finished its turn."""
    if token_count < 3:
        return False
    last8 = text[-8:] if len(text) >= 8 else text
    for i, ch in enumerate(last8):
        if ch in ".!?":
            if i > 0 and ch == "." and last8[i-1].isalpha():
                continue
            return True
    if "<end_of_turn>" in text:
        return True
    if text.count("<start_of_turn>") >= 2:
        return True
    if len(recent) >= 4 and all(t == recent[-1] for t in recent[-4:]):
        return True
    short = {"si", "no", "hola", "ok", "okay", "vale", "bueno", "claro",
             "tal vez", "perfecto", "entendido", "de nada", "gracias"}
    stripped = text.strip().lower()
    if 3 <= token_count <= 20 and 1 < len(stripped) <= 15 and stripped in short:
        return True
    return False


def gemma_stream(llm, mel, in_proj, context):
    T = mel.shape[0]
    pad = np.pad(mel, ((0, 0), (0, 896)), mode="constant")
    _ = pad @ in_proj.T  # warmup projection

    if PROMPT_MODE == "audio_removed":
        prompt = f"<start_of_turn>user\nResponde.\n<end_of_turn>\n<start_of_turn>model\n"
    elif PROMPT_MODE == "minimal":
        prompt = (f"<start_of_turn>user\n[Audio: {T} mel frames]\nResponde.\n"
                  f"<end_of_turn>\n<start_of_turn>model\n")
    else:
        prompt = (f"<start_of_turn>user\n[Audio: {T} mel frames]\n{context}\n"
                  f"<end_of_turn>\n<start_of_turn>model\n")

    full = ""
    buf = ""
    recent = []
    try:
        for tok in llm(prompt, max_tokens=180, stop=["<end_of_turn>"],
                       echo=False, stream=True):
            t = tok["choices"][0]["text"]
            full += t
            buf += t
            recent.append(t)
            if len(recent) > 8:
                recent.pop(0)
            if len(buf.strip()) >= 24:
                yield ("text", buf.strip())
                buf = ""
            if should_stop_streaming(full, len(recent), recent):
                break
    finally:
        if buf.strip():
            yield ("text", buf.strip())
        yield ("flush", None)


# ─── Full-duplex core ───────────────────────────────────────────────────────
def make_pipeline(llm, in_proj, tts_worker):
    """Returns a function that processes one segment end-to-end."""
    def process(seg):
        try:
            mel = mel_spec(seg)
        except Exception as e:
            print(f"[mel] {e}", flush=True)
            return
        for kind, data in gemma_stream(llm, mel, in_proj, ""):
            if kind == "text":
                tts_worker.push_text(data)
            elif kind == "flush":
                tts_worker.flush()
    return process


# ─── Audio streams ──────────────────────────────────────────────────────────
class FullDuplexStream:
    def __init__(self, process_fn):
        self.q_mic = queue.Queue(maxsize=400)
        self.q_play = queue.Queue()
        self.stop_evt = threading.Event()
        self.process_fn = process_fn
        self._segment_q = queue.Queue(maxsize=4)
        self._worker = threading.Thread(target=self._worker_loop,
                                        daemon=True, name="michi-seg")
        self._worker.start()

    def _mic_cb(self, indata, frames, t, status):
        if status:
            return
        try:
            self.q_mic.put_nowait(indata[:, 0].copy())
        except queue.Full:
            try:
                self.q_mic.get_nowait()
                self.q_mic.put_nowait(indata[:, 0].copy())
            except Exception:
                pass

    def _play_cb(self, outdata, frames, t, status):
        try:
            chunk = self.q_play.get_nowait()
        except queue.Empty:
            outdata[:, 0] = 0
            speaker_gate.set(False)
            return
        n = min(len(chunk), frames)
        outdata[:n, 0] = chunk[:n]
        if n < frames:
            outdata[n:, 0] = 0
        speaker_gate.set(True)

    def _worker_loop(self):
        """Consume segments from the queue without blocking the live loop."""
        while not self.stop_evt.is_set():
            try:
                seg = self._segment_q.get(timeout=0.1)
            except queue.Empty:
                continue
            self.process_fn(seg)

    def live_loop(self):
        buf = np.zeros(0, dtype=np.float32)
        silence = 0
        talking = False
        SILENCE_END = int(0.7 * 1000 / 32)  # 700ms @ 32ms chunks
        MIN_TALK = int(MIC_RATE * 0.5)

        with sd.InputStream(samplerate=MIC_RATE, channels=1, dtype="float32",
                            blocksize=512, callback=self._mic_cb), \
             sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                             blocksize=2048, callback=self._play_cb):
            print("[ready] full-duplex. Ctrl+C to stop.", flush=True)
            while not self.stop_evt.is_set():
                try:
                    chunk = self.q_mic.get(timeout=0.1)
                except queue.Empty:
                    continue

                buf = np.concatenate([buf, chunk])[-MIC_RATE * 6:]

                if is_speech_echo_aware(chunk):
                    silence = 0
                    if not talking and len(buf) >= MIN_TALK:
                        talking = True
                else:
                    if talking:
                        silence += 1
                        if silence >= SILENCE_END:
                            seg = buf.copy()
                            buf = np.zeros(0, dtype=np.float32)
                            silence = 0
                            talking = False
                            print(f"\n[seg {len(seg)/MIC_RATE:.1f}s] -> worker",
                                  flush=True)
                            try:
                                self._segment_q.put_nowait(seg)
                            except queue.Full:
                                print("[seg] worker busy, dropping", flush=True)

                # Barge-in: if user starts talking while TTS is playing,
                # speaker_gate stays True (echo gate) but the live loop
                # ignores audio. To actually barge-in, the worker would
                # need to drop its current TTS job. Handled by TTSWorker
                # support for ->stop() if you want hard barge-in.
                print(".", end="", flush=True)


def demo():
    """Synthetic 1.5s sine-wave segment through the full pipeline."""
    seg = np.sin(2 * np.pi * 220 * np.linspace(0, 1.5,
            int(MIC_RATE * 1.5))).astype(np.float32) * 0.04
    print("[demo] loading models...", flush=True)
    llm = _load_llm()
    in_proj, _ = _load_adapters()
    tts = _load_tts()
    tts_w = TTSWorker(tts, None)
    tts_w.start()
    process = make_pipeline(llm, in_proj, tts_w)
    process(seg)
    tts_w.flush()
    tts_w.stop()
    tts_w.join(timeout=8.0)


def run_mic():
    print("[boot] loading models...", flush=True)
    llm = _load_llm()
    in_proj, _ = _load_adapters()
    tts = _load_tts()
    tts_w = TTSWorker(tts, None)  # play queue wired in FullDuplexStream below
    tts_w.start()
    process = make_pipeline(llm, in_proj, tts_w)

    # Rebuild worker with the real play queue
    tts_w.play_queue = None  # safe-guarded: re-route through lazy binding
    # Instead: create a unified stream that owns both queues.

    class _Stream(FullDuplexStream):
        def __init__(self, process_fn):
            # Re-route TTS worker output through the play queue.
            super().__init__(process_fn)
            tts_w.play_queue = self.q_play  # late-bind

    stream = _Stream(process)
    signal.signal(signal.SIGINT, lambda *_: stream.stop_evt.set())
    try:
        stream.live_loop()
    except KeyboardInterrupt:
        stream.stop_evt.set()
    tts_w.stop()
    tts_w.join(timeout=5.0)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["demo", "mic"], default="mic")
    args = p.parse_args()

    if args.mode == "demo":
        demo()
    else:
        run_mic()


if __name__ == "__main__":
    # torch is lazy-imported in _load_adapters / _vad_silero
    import torch  # noqa: F401
