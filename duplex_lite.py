"""Duplex-Lite S2S model — Nemotron-style architecture, 1/10 the size.

Pipeline:
    audio 16kHz ─► Encoder (frozen, switchable)
                    - Whisper-small (HF, 768-d, 80ms frames)
                    - Parakeet-tdt-0.6b-v3 (HF, 1024-d, NeMo origin)
                  ─► EncoderToLLM (trainable, enc_dim → llm_dim)
                  ─► LLM (frozen, 4-bit) ─► text tokens
                  ─► LLM hidden states ─► LLMToTTS (trainable, llm_dim → 256)
                  ─► Supertonic-3 (frozen) ─► audio 24kHz

Trainable params: ~3-5M (projectors + optional LoRA on LLM).
Frozen: encoder (~70M-600M), LLM (0.5-3B in 4-bit), Supertonic-3.
"""
import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig


# ─── Projectors ──────────────────────────────────────────────────────────────
class EncoderToLLM(nn.Module):
    """Project any encoder embeddings (B, T, enc_dim) into LLM hidden space."""
    def __init__(self, enc_dim: int = 768, llm_dim: int = 896):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(enc_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# Backwards-compat alias
WhisperToLLM = EncoderToLLM


class LLMToTTS(nn.Module):
    """Project LLM hidden states → small conditioning vector for Supertonic-3."""
    def __init__(self, llm_dim: int = 896, tts_cond_dim: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(llm_dim, tts_cond_dim),
            nn.LayerNorm(tts_cond_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ─── Model assembly ──────────────────────────────────────────────────────────
LLM_CONFIGS = {
    "qwen2.5-0.5b": {
        "repo": "Qwen/Qwen2.5-0.5B-Instruct",
        "dim": 896,
        "context": 4096,
    },
    "minicpm5-1b": {
        "repo": "openbmb/MiniCPM5-1B-MLX",
        "dim": 1024,
        "context": 4096,
    },
    "gemma-4-e2b": {
        "repo": "google/gemma-4-E2B-it",
        "dim": 2304,  # E2B hidden size (estimate)
        "context": 8192,
    },
}

ENCODER_CONFIGS = {
    "whisper-small": {
        "repo": "openai/whisper-small",
        "dim": 768,
        "kind": "whisper",
        "sample_rate": 16000,
        "frame_ms": 80,
    },
    "parakeet-tdt-0.6b-v3": {
        "repo": "nvidia/parakeet-tdt-0.6b-v3",
        "dim": 1024,
        "kind": "parakeet",
        "sample_rate": 16000,
        "frame_ms": 80,  # ~80ms stride, varies
    },
}


class DuplexLite(nn.Module):
    """Encoder (frozen, switchable) + projectors (trainable) + LLM (frozen 4-bit)."""

    def __init__(self, llm_key: str = "gemma-4-e2b",
                 encoder_key: str = "whisper-small",
                 quantize_llm: bool = True,
                 device: str = "cuda"):
        super().__init__()
        llm_cfg = LLM_CONFIGS[llm_key]
        enc_cfg = ENCODER_CONFIGS[encoder_key]
        self.llm_key = llm_key
        self.encoder_key = encoder_key
        self.llm_dim = llm_cfg["dim"]
        self.encoder_dim = enc_cfg["dim"]
        self.encoder_kind = enc_cfg["kind"]

        # 1) Encoder (frozen) — Whisper or Parakeet
        print(f"[duplex] loading {encoder_key} encoder {enc_cfg['repo']}...", flush=True)
        if self.encoder_kind == "whisper":
            from transformers import WhisperModel, WhisperProcessor
            self.encoder = WhisperModel.from_pretrained(enc_cfg["repo"]).encoder
            # WhisperProcessor has feature_extractor (log-mel) + tokenizer
            self.feature_extractor = WhisperProcessor.from_pretrained(enc_cfg["repo"]).feature_extractor
        elif self.encoder_kind == "parakeet":
            from transformers import ParakeetForCTC
            full = ParakeetForCTC.from_pretrained(enc_cfg["repo"])
            self.encoder = full.encoder  # FastConformer
            # Parakeet accepts raw audio directly, no feature extractor needed
            self.feature_extractor = None
        else:
            raise ValueError(f"Unknown encoder kind: {self.encoder_kind}")
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        # 2) Projectors (trainable) — encoder_dim → llm_dim
        # Keep projectors in fp32 for stable gradients. fp16 grads NaN instantly.
        self.audio_proj = EncoderToLLM(self.encoder_dim, self.llm_dim).to(torch.float32)
        self.tts_proj = LLMToTTS(self.llm_dim, 256).to(torch.float32)

        # 3) LLM (frozen, 4-bit)
        print(f"[duplex] loading LLM {llm_cfg['repo']} (4-bit={quantize_llm})...", flush=True)
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=quantize_llm,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        ) if quantize_llm else None
        self.tokenizer = AutoTokenizer.from_pretrained(llm_cfg["repo"])
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_cfg["repo"],
            quantization_config=bnb_cfg,
            torch_dtype=torch.float16 if not quantize_llm else None,
            device_map="auto",
        )
        self.llm.eval()
        for p in self.llm.parameters():
            p.requires_grad = False

        # Move frozen encoder + LLM to device; projectors stay fp32 on CPU-side
        self.encoder.to(device)
        self.llm.to(device)
        self.audio_proj.to(device)
        self.tts_proj.to(device)

    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self):
        return sum(p.numel() for p in self.parameters())

    def encode_audio(self, audio_16k: torch.Tensor) -> torch.Tensor:
        """Encoder forward. Input: [B, samples] @ 16kHz, returns [B, T, enc_dim]."""
        with torch.no_grad():
            if self.encoder_kind == "whisper":
                # Whisper needs log-mel features, not raw audio
                # audio_16k: [B, samples] -> list of 1D arrays for feature_extractor
                import numpy as np
                audios_np = [a.cpu().numpy() for a in audio_16k]
                feats = self.feature_extractor(
                    audios_np, sampling_rate=16000, return_tensors="pt"
                ).input_features.to(audio_16k.device, dtype=torch.float16)
                out = self.encoder(feats)
            else:
                # Parakeet accepts raw audio
                out = self.encoder(audio_16k)
            return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]

    def forward_train(self, audio_16k: torch.Tensor, target_text):
        """Phase 1: train EncoderToLLM projector.

        audio_16k: [B, samples] @ 16kHz
        target_text: list of strings (length B)
        Returns CE loss on next-token prediction conditioned on audio embeddings.
        """
        device = audio_16k.device
        audio_emb = self.encode_audio(audio_16k)  # [B, T_a, enc_dim]
        proj = self.audio_proj(audio_emb.float())  # [B, T_a, llm_dim]

        tok = self.tokenizer(target_text, return_tensors="pt", padding=True,
                              truncation=True, max_length=128).to(device)
        input_ids = tok.input_ids
        embed_layer = self.llm.get_input_embeddings()
        text_emb = embed_layer(input_ids).to(proj.dtype)  # [B, T_t, llm_dim]

        # Cat audio + text embeds as LLM input
        inputs_embeds = torch.cat([proj, text_emb], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], device=device, dtype=torch.long)
        labels = torch.full(inputs_embeds.shape[:2], -100, device=device, dtype=torch.long)
        labels[:, proj.shape[1]:] = input_ids

        out = self.llm(inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        labels=labels)
        return out.loss

    def save_projectors(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "audio_proj": self.audio_proj.state_dict(),
            "tts_proj": self.tts_proj.state_dict(),
            "llm_key": self.llm_key,
            "encoder_key": self.encoder_key,
            "encoder_dim": self.encoder_dim,
            "llm_dim": self.llm_dim,
        }, path)

    def load_projectors(self, path: str):
        ck = torch.load(path, map_location="cpu")
        self.audio_proj.load_state_dict(ck["audio_proj"])
        self.tts_proj.load_state_dict(ck["tts_proj"])

    @staticmethod
    def supported_llm_keys():
        return list(LLM_CONFIGS.keys())

    @staticmethod
    def supported_encoder_keys():
        return list(ENCODER_CONFIGS.keys())


# ─── Parakeet-v3 wrapper (inference only, MLX-based, bypasses projectors) ───
class ParakeetDirectSTT:
    """Parakeet-tdt-0.6b-v3 via parakeet-mlx. Returns plain text. Used in
    app_michi_lite.py when MICHI_ENCODER=parakeet — projector is bypassed.

    This is a STT-only path: text → LLM → TTS. No audio embeddings, no projector.
    """

    def __init__(self):
        import os
        from parakeet_mlx import from_pretrained
        repo = os.environ.get("MICHI_PARAKEET_REPO",
                              "mlx-community/parakeet-tdt-0.6b-v3")
        print(f"[parakeet] loading {repo}...", flush=True)
        self.model = from_pretrained(repo)
        print(f"[parakeet] ready", flush=True)

    def transcribe(self, audio_16k_1d) -> str:
        """audio_16k_1d: numpy 1D float32 @ 16kHz. Returns text."""
        from parakeet_mlx.audio import Audio
        audio = Audio.from_array(audio_16k_1d, sample_rate=16000)
        result = self.model.transcribe(audio)
        # result has .text or .segments; pick whichever exists
        if hasattr(result, "text"):
            return result.text.strip()
        return str(result).strip()


# ─── Standalone test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    llm_key = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5-0.5b"
    enc_key = sys.argv[2] if len(sys.argv) > 2 else "whisper-small"
    print(f"LLM={llm_key}  ENCODER={enc_key}")
    model = DuplexLite(llm_key=llm_key, encoder_key=enc_key, quantize_llm=True,
                        device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"total params:     {model.total_params()/1e6:.1f}M")
    print(f"trainable params: {model.trainable_params()/1e6:.3f}M")
    print(f"encoder_dim={model.encoder_dim} ({model.encoder_key}), "
          f"llm_dim={model.llm_dim} ({model.llm_key})")
    print(f"supported LLMs:     {model.supported_llm_keys()}")
    print(f"supported encoders: {model.supported_encoder_keys()}")