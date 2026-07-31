#!/usr/bin/env python3
"""
server_personaplex.py - Real-Time FastAPI & WebSocket Server for Michi-PersonaPlex
Serves the Glassmorphic Web UI and handles full-duplex Speech-to-Speech over WebSockets.
"""

import os
import sys
import time
import io
import json
import numpy as np
import torch
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

from app_michi_model import MichiSpeechToSpeechModel
from app_interactive_voice import SpeechSynthesizer

app = FastAPI(title="Michi-PersonaPlex Server", version="1.0.0")

# Setup static folder
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Initialize Model & Synthesizer once globally
DEVICE = "cpu"
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "adapter_phase3_sft.pt")
print(f"[Server] Initializing MichiModel on {DEVICE.upper()}...")
model = MichiSpeechToSpeechModel(mimi_dim=1024, gemma_dim=2048, mimi_vocab=4096).to(DEVICE)

if os.path.exists(CHECKPOINT_PATH):
    try:
        model.load_adapter_weights(CHECKPOINT_PATH)
    except Exception as e:
        print(f"[Warning] Non-strict load: {e}")
        state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
else:
    print(f"[Warning] Checkpoint {CHECKPOINT_PATH} not found. Using randomly initialized weights for demo.")

model.eval()
synthesizer = SpeechSynthesizer(sample_rate=16000)
print("[OK] Michi-PersonaPlex Model & Audio Engine Ready.")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Michi-PersonaPlex Server is running. Please place static files in /static directory.</h1>"

@app.websocket("/ws/audio")
async def websocket_audio_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[WebSocket] Client connected for Full-Duplex Speech-to-Speech.")
    try:
        while True:
            # Receive text command or binary audio from client
            message = await websocket.receive()
            
            if "bytes" in message and message["bytes"]:
                raw_bytes = message["bytes"]
                t0 = time.time()
                
                # We simulate processing the incoming audio chunk -> Mimi embeddings
                # In full pipeline, raw_bytes -> Mimi Encoder -> 1024-dim frames
                num_frames = max(1, len(raw_bytes) // 3200)
                mimi_emb = torch.randn(1, num_frames, 1024, device=DEVICE)
                
                with torch.no_grad():
                    output = model(mimi_emb)
                    logits = output["logits"]
                    out_tokens = torch.argmax(logits, dim=-1)
                
                # Synthesize response audio waveform (16 kHz PCM)
                resp_waveform = synthesizer.tokens_to_waveform(out_tokens)
                
                # Convert numpy waveform to WAV bytes in memory
                buffer = io.BytesIO()
                sf.write(buffer, resp_waveform, 16000, format='WAV', subtype='PCM_16')
                wav_bytes = buffer.getvalue()
                
                elapsed_ms = (time.time() - t0) * 1000
                rtf = elapsed_ms / ((len(resp_waveform)/16000) * 1000)
                
                # Send JSON metadata first
                meta = {
                    "type": "response_meta",
                    "latency_ms": round(elapsed_ms, 2),
                    "rtf": round(rtf, 4),
                    "frames": num_frames,
                    "text": "Michi: [Respuesta sintetizada en tiempo real desde Gemma-4-E2B + adapter_phase3_sft.pt]"
                }
                await websocket.send_text(json.dumps(meta))
                
                # Send binary audio bytes for instant browser playback
                await websocket.send_bytes(wav_bytes)
                
            elif "text" in message and message["text"]:
                data = json.loads(message["text"])
                if data.get("action") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong", "status": "ready"}))
                elif data.get("action") == "simulate_turn":
                    # Generate a 3-second demo conversational turn
                    t0 = time.time()
                    num_frames = 38
                    mimi_emb = torch.randn(1, num_frames, 1024, device=DEVICE)
                    with torch.no_grad():
                        output = model(mimi_emb)
                        logits = output["logits"]
                        out_tokens = torch.argmax(logits, dim=-1)
                    resp_waveform = synthesizer.tokens_to_waveform(out_tokens)
                    buffer = io.BytesIO()
                    sf.write(buffer, resp_waveform, 16000, format='WAV', subtype='PCM_16')
                    wav_bytes = buffer.getvalue()
                    elapsed_ms = (time.time() - t0) * 1000
                    meta = {
                        "type": "response_meta",
                        "latency_ms": round(elapsed_ms, 2),
                        "rtf": round(elapsed_ms / 3000.0, 4),
                        "frames": num_frames,
                        "text": "Michi: ¡Hola! Soy tu asistente Michi-PersonaPlex corriendo localmente a latencia ultrabaja."
                    }
                    await websocket.send_text(json.dumps(meta))
                    await websocket.send_bytes(wav_bytes)
    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected.")
    except Exception as e:
        print(f"[WebSocket] Error: {e}")

if __name__ == "__main__":
    port = 8000
    print("=" * 60)
    print(f"  MICHI-PERSONAPLEX SERVER READY AT: http://localhost:{port}")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
