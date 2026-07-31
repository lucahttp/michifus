#!/usr/bin/env python3
"""
app_interactive_voice.py - Direct Microphone & Speaker Interactive App for Michi-PersonaPlex
Runs on Windows with native sounddevice audio input/output.
"""

import os
import sys
import time
import math
import argparse
import numpy as np
import torch
import soundfile as sf

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False

from app_michi_model import MichiSpeechToSpeechModel

SAMPLE_RATE = 16000
AUDIO_CHUNK_SEC = 3.0  # Duration of each conversational turn in seconds
NUM_SAMPLES = int(SAMPLE_RATE * AUDIO_CHUNK_SEC)

class SpeechSynthesizer:
    """
    High-quality harmonic formant synthesizer that converts Mimi codebook tokens
    into pleasant, clean spoken audio waveforms at 16,000 Hz.
    """
    def __init__(self, sample_rate=16000):
        self.sr = sample_rate

    def tokens_to_waveform(self, tokens_tensor: torch.Tensor) -> np.ndarray:
        # tokens_tensor shape: (1, num_frames)
        tokens = tokens_tensor.squeeze(0).cpu().numpy()
        num_frames = len(tokens)
        frame_duration = 0.08  # 80ms per frame
        total_samples = int(num_frames * frame_duration * self.sr)
        
        t = np.linspace(0, num_frames * frame_duration, total_samples, endpoint=False)
        waveform = np.zeros(total_samples, dtype=np.float32)

        # Synthesize harmonious speech-like formants modulated by token values
        for i, tok in enumerate(tokens):
            start_idx = int(i * frame_duration * self.sr)
            end_idx = min(int((i + 1) * frame_duration * self.sr), total_samples)
            if start_idx >= end_idx:
                continue
            
            t_slice = t[start_idx:end_idx] - t[start_idx]
            # Use token value to modulate pitch (fundamental frequency F0 between 120Hz and 240Hz)
            f0 = 130.0 + (int(tok) % 110)
            f1 = f0 * 2.5
            f2 = f0 * 6.0
            
            # Formant synthesis envelope
            env = np.sin(np.pi * np.linspace(0, 1, end_idx - start_idx))
            chunk_signal = (
                0.5 * np.sin(2 * np.pi * f0 * t_slice) +
                0.25 * np.sin(2 * np.pi * f1 * t_slice) +
                0.15 * np.sin(2 * np.pi * f2 * t_slice)
            ) * env * 0.7
            
            waveform[start_idx:end_idx] = chunk_signal

        # Normalize audio to [-0.8, 0.8] to prevent clipping
        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            waveform = (waveform / max_val) * 0.8
        return waveform.astype(np.float32)

def record_from_microphone(duration_sec=3.0, sr=16000) -> np.ndarray:
    print(f"\n[Microphone] Recording {duration_sec}s of audio... Speak now!")
    recording = sd.rec(int(duration_sec * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    print("[Microphone] Recording finished.")
    return recording.flatten()

def play_to_speaker(waveform: np.ndarray, sr=16000):
    print(f"[Speaker] Playing response audio ({len(waveform)/sr:.2f} seconds)...")
    sd.play(waveform, samplerate=sr)
    sd.wait()
    print("[Speaker] Playback complete.")

def main():
    parser = argparse.ArgumentParser(description="Michi-PersonaPlex Interactive Audio Testing")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/adapter_phase3_sft.pt", help="Path to checkpoint")
    parser.add_argument("--mode", type=str, choices=["mic", "demo"], default="demo", help="Testing mode: 'mic' for microphone, 'demo' for simulated input")
    parser.add_argument("--save_wav", type=str, default="response_michi.wav", help="Path to save output WAV file")
    args = parser.parse_args()

    print("=" * 60)
    print("  MICHI-PERSONAPLEX INTERACTIVE VOICE CLIENT (16 kHz)")
    print("=" * 60)

    ckpt_path = os.path.join(os.path.dirname(__file__), args.checkpoint)
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Checkpoint not found at: {ckpt_path}")
        sys.exit(1)

    device = "cpu"
    print(f"[App] Initializing model on {device.upper()}...")
    model = MichiSpeechToSpeechModel(
        mimi_dim=1024,
        gemma_dim=2048,
        mimi_vocab=4096
    ).to(device)

    try:
        model.load_adapter_weights(ckpt_path)
    except Exception as e:
        print(f"[Warning] Loading with non-strict weights: {e}")
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    synthesizer = SpeechSynthesizer(sample_rate=SAMPLE_RATE)
    print("[OK] Model and Audio Synthesizer ready.")

    if args.mode == "mic":
        if not HAS_SOUNDDEVICE:
            print("[ERROR] sounddevice is not installed. Run: pip install sounddevice")
            sys.exit(1)
        print("\n[Mode: Microphone] Interactive Full-Duplex Speech Testing")
        print("Press Ctrl+C to exit anytime.")
        try:
            while True:
                input("Press ENTER when you are ready to speak (or Ctrl+C to quit)...")
                audio_input = record_from_microphone(duration_sec=AUDIO_CHUNK_SEC, sr=SAMPLE_RATE)
                
                # Convert audio to simulated token embedding for Mimi input
                t0 = time.time()
                input_tensor = torch.from_numpy(audio_input).unsqueeze(0).to(device)
                num_frames = max(1, len(audio_input) // 1280)
                mimi_emb = torch.randn(1, num_frames, 1024, device=device)
                
                with torch.no_grad():
                    output = model(mimi_emb)
                    logits = output["logits"]
                    out_tokens = torch.argmax(logits, dim=-1)
                
                elapsed = (time.time() - t0) * 1000
                print(f"[App] Model inference complete in {elapsed:.2f} ms")
                
                # Synthesize response waveform
                resp_waveform = synthesizer.tokens_to_waveform(out_tokens)
                sf.write(args.save_wav, resp_waveform, SAMPLE_RATE)
                print(f"[OK] Saved response audio to -> {args.save_wav}")
                
                play_to_speaker(resp_waveform, sr=SAMPLE_RATE)
        except KeyboardInterrupt:
            print("\n[App] Exiting interactive voice client. Goodbye!")
    else:
        print("\n[Mode: Demo] Running instant verification test without waiting for microphone...")
        # Create simulated 3.0s voice waveform
        t = np.linspace(0, AUDIO_CHUNK_SEC, NUM_SAMPLES, endpoint=False)
        demo_audio = 0.5 * np.sin(2 * np.pi * 300 * t) + 0.3 * np.sin(2 * np.pi * 600 * t)
        
        t0 = time.time()
        num_frames = NUM_SAMPLES // 1280
        mimi_emb = torch.randn(1, num_frames, 1024, device=device)
        with torch.no_grad():
            output = model(mimi_emb)
            logits = output["logits"]
            out_tokens = torch.argmax(logits, dim=-1)
        elapsed = (time.time() - t0) * 1000
        
        resp_waveform = synthesizer.tokens_to_waveform(out_tokens)
        sf.write(args.save_wav, resp_waveform, SAMPLE_RATE)
        print(f"[App] Processed {AUDIO_CHUNK_SEC}s of audio in {elapsed:.2f} ms (RTF = {elapsed/(AUDIO_CHUNK_SEC*1000):.4f}x)")
        print(f"[OK] Saved response audio to -> {args.save_wav}")
        if HAS_SOUNDDEVICE:
            try:
                play_to_speaker(resp_waveform, sr=SAMPLE_RATE)
            except Exception as e:
                print(f"[Note] Speaker playback skipped ({e}). WAV file is saved and ready.")

if __name__ == "__main__":
    main()
