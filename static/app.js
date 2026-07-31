/**
 * app.js - Client-Side Engine for Michi-PersonaPlex Web UI
 * Handles WebAudio capture, WebSockets full-duplex communication, and 60FPS glowing waveform animation.
 */

document.addEventListener("DOMContentLoaded", () => {
    const wsUrl = `ws://${window.location.host}/ws/audio`;
    let ws = null;
    let audioCtx = null;
    let micStream = null;
    let mediaRecorder = null;
    let isRecording = false;

    // UI Elements
    const statusBadge = document.getElementById("server-status");
    const statusText = statusBadge.querySelector(".status-text");
    const canvas = document.getElementById("waveform-canvas");
    const ctx = canvas.getContext("2d");
    const orbStatusText = document.getElementById("orb-status-text");
    const btnMic = document.getElementById("btn-mic");
    const btnMicText = document.getElementById("btn-mic-text");
    const btnDemo = document.getElementById("btn-demo");
    const statLatency = document.getElementById("stat-latency");
    const statRtf = document.getElementById("stat-rtf");
    const transcriptBox = document.getElementById("transcript-box");

    // Animation variables
    let animationId = null;
    let visualizerData = new Uint8Array(64).fill(10);
    let isSpeaking = false;

    // --- WebSocket Connection ---
    function connectWebSocket() {
        ws = new WebSocket(wsUrl);
        ws.binaryType = "arraybuffer";

        ws.onopen = () => {
            statusBadge.classList.add("connected");
            statusText.textContent = "Servidor Conectado (Local Windows D:\\)";
            addTranscriptMsg("sistema", "WebSocket activo. Conectado al motor local Michi-PersonaPlex.");
            startWaveformAnimation();
        };

        ws.onmessage = async (event) => {
            if (typeof event.data === "string") {
                const meta = JSON.parse(event.data);
                if (meta.type === "response_meta") {
                    statLatency.textContent = `${meta.latency_ms} ms`;
                    statRtf.textContent = `${meta.rtf} x`;
                    addTranscriptMsg("michi", meta.text);
                }
            } else if (event.data instanceof ArrayBuffer) {
                // Incoming WAV / PCM Audio from Server -> Play in browser!
                playAudioResponse(event.data);
            }
        };

        ws.onclose = () => {
            statusBadge.classList.remove("connected");
            statusText.textContent = "Desconectado. Reintentando...";
            setTimeout(connectWebSocket, 3000);
        };
    }

    // --- Audio Playback ---
    async function playAudioResponse(arrayBuffer) {
        if (!audioCtx) {
            audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        }
        try {
            const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
            const source = audioCtx.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(audioCtx.destination);
            
            isSpeaking = true;
            orbStatusText.textContent = "MICHI ESTÁ HABLANDO...";
            orbStatusText.style.color = "#b388ff";
            orbStatusText.style.borderColor = "#b388ff";

            source.onended = () => {
                isSpeaking = false;
                orbStatusText.textContent = "LISTO PARA CONVERSAR";
                orbStatusText.style.color = "#00f2fe";
                orbStatusText.style.borderColor = "rgba(0, 242, 254, 0.3)";
            };

            source.start(0);
        } catch (err) {
            console.error("Error decoding response audio:", err);
        }
    }

    // --- Microphone Capture ---
    async function toggleMicrophone() {
        if (!isRecording) {
            try {
                micStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
                mediaRecorder = new MediaRecorder(micStream);
                const chunks = [];

                mediaRecorder.ondataavailable = (e) => {
                    if (e.data.size > 0) chunks.push(e.data);
                };

                mediaRecorder.onstop = async () => {
                    const blob = new Blob(chunks, { type: "audio/webm" });
                    const arrayBuffer = await blob.arrayBuffer();
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        addTranscriptMsg("usuario", "🎤 [Audio de micrófono enviado -> procesando por Gemma 4 E2B + Mimi adapter]");
                        orbStatusText.textContent = "PROCESANDO CONVERSACIÓN...";
                        ws.send(arrayBuffer);
                    }
                    // Stop tracks
                    micStream.getTracks().forEach(t => t.stop());
                };

                mediaRecorder.start();
                isRecording = true;
                btnMic.classList.add("recording");
                btnMicText.textContent = "Detener y Enviar (3s auto...)";
                orbStatusText.textContent = "ESCUCHANDO TU VOZ...";
                orbStatusText.style.color = "#ff1744";
                orbStatusText.style.borderColor = "#ff1744";

                // Automatically stop recording after 3 seconds for turn-taking
                setTimeout(() => {
                    if (isRecording) {
                        toggleMicrophone();
                    }
                }, 3000);

            } catch (err) {
                alert("No se pudo acceder al micrófono. Verificá los permisos del navegador.");
                console.error(err);
            }
        } else {
            isRecording = false;
            if (mediaRecorder && mediaRecorder.state !== "inactive") {
                mediaRecorder.stop();
            }
            btnMic.classList.remove("recording");
            btnMicText.textContent = "Hablar por Micrófono";
        }
    }

    // --- Instant Demo Trigger ---
    btnDemo.addEventListener("click", () => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            addTranscriptMsg("usuario", "⚡ [Prueba instantánea solicitada al servidor local]");
            orbStatusText.textContent = "GENERANDO RESPUESTA...";
            ws.send(JSON.stringify({ action: "simulate_turn" }));
        }
    });

    btnMic.addEventListener("click", toggleMicrophone);

    // --- Transcript Helper ---
    function addTranscriptMsg(sender, text) {
        const div = document.createElement("div");
        div.className = `msg ${sender}-msg`;
        
        const senderSpan = document.createElement("span");
        senderSpan.className = "sender";
        senderSpan.textContent = sender === "michi" ? "🐱 Michi:" : (sender === "usuario" ? "🎤 Vos:" : "⚙️ Sistema:");

        const textSpan = document.createElement("span");
        textSpan.className = "text";
        textSpan.textContent = text;

        div.appendChild(senderSpan);
        div.appendChild(textSpan);
        transcriptBox.appendChild(div);
        transcriptBox.scrollTop = transcriptBox.scrollHeight;
    }

    // --- 60FPS Glowing Waveform Visualizer ---
    function startWaveformAnimation() {
        const w = canvas.width;
        const h = canvas.height;
        let phase = 0;

        function draw() {
            ctx.clearRect(0, 0, w, h);
            phase += 0.05;

            // Draw center glow line
            ctx.beginPath();
            ctx.lineWidth = 3;
            
            let strokeColor = "#00f2fe";
            if (isRecording) strokeColor = "#ff1744";
            else if (isSpeaking) strokeColor = "#b388ff";

            ctx.strokeStyle = strokeColor;
            ctx.shadowBlur = 15;
            ctx.shadowColor = strokeColor;

            for (let x = 0; x < w; x++) {
                const nx = x / w;
                // Harmonic frequency modulated waves
                let amp = 15;
                if (isRecording) amp = 55 + Math.sin(phase * 3) * 20;
                else if (isSpeaking) amp = 45 + Math.cos(phase * 4) * 25;

                const y = h / 2 + Math.sin(nx * Math.PI * 6 + phase) * amp * Math.sin(nx * Math.PI);
                if (x === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();

            // Draw secondary reflection line
            ctx.beginPath();
            ctx.lineWidth = 1.5;
            ctx.strokeStyle = "rgba(255, 255, 255, 0.25)";
            ctx.shadowBlur = 0;
            for (let x = 0; x < w; x++) {
                const nx = x / w;
                const y = h / 2 + Math.cos(nx * Math.PI * 8 - phase * 0.7) * (isSpeaking || isRecording ? 25 : 8) * Math.sin(nx * Math.PI);
                if (x === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();

            animationId = requestAnimationFrame(draw);
        }

        draw();
    }

    // Connect on boot
    connectWebSocket();
});
