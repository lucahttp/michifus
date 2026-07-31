"""Integrated Michi-PersonaPlex Backbone Swap Model (Escenario B).

Connects:
  - Mimi Audio Codec Tokens (1024-dim / 4096-vocab)
  - MichiAdapter (MimiToGemma [1024->2048] & GemmaToMimi [2048->4096])
  - Gemma 4 E2B LLM Hidden Space (2048-dim)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from adapter import MichiAdapter, MimiToGemma, GemmaToMimi


class MichiSpeechToSpeechModel(nn.Module):
    """Full Duplex Audio-to-Audio Speech Model.
    
    Translates incoming Mimi audio embeddings into Gemma hidden space,
    processes context with Gemma, and projects output hidden states
    back into Mimi audio codebook logits.
    """
    def __init__(
        self,
        mimi_dim: int = 1024,
        gemma_dim: int = 2048,
        mimi_vocab: int = 4096,
        checkpoint_path: str = None
    ):
        super().__init__()
        self.mimi_dim = mimi_dim
        self.gemma_dim = gemma_dim
        self.mimi_vocab = mimi_vocab

        self.adapter = MichiAdapter(mimi_dim=mimi_dim, gemma_dim=gemma_dim, mimi_vocab=mimi_vocab)
        
        # Linear simulated Gemma backbone projection for fast SFT / local deployment
        self.gemma_backbone_sim = nn.Linear(gemma_dim, gemma_dim)

        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_adapter_weights(checkpoint_path)

    def load_adapter_weights(self, checkpoint_path: str):
        print(f"[MichiModel] Loading adapter weights from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if "in_adapter" in ckpt:
            self.adapter.in_adapter.load_state_dict(ckpt["in_adapter"])
        if "out_adapter" in ckpt:
            self.adapter.out_adapter.load_state_dict(ckpt["out_adapter"])
        print("[MichiModel] Weights loaded successfully.")

    def forward(self, input_mimi_emb: torch.Tensor, target_mimi_tokens: torch.Tensor = None):
        return self.forward_speech_to_speech(input_mimi_emb, target_mimi_tokens)

    def forward_speech_to_speech(self, input_mimi_emb: torch.Tensor, target_mimi_tokens: torch.Tensor = None):
        """Full forward pass:
          input_mimi_emb [B, T, 1024] 
            -> in_adapter -> [B, T, 2048] (Gemma hidden space)
            -> gemma_backbone -> [B, T, 2048]
            -> out_adapter -> [B, T, 4096] (Mimi Codebook Logits)
        """
        # Step 1: Mimi -> Gemma
        gemma_in = self.adapter.in_adapter(input_mimi_emb)

        # Step 2: Gemma LLM processing
        gemma_out = self.gemma_backbone_sim(gemma_in) + gemma_in  # residual link

        # Step 3: Gemma -> Mimi Logits
        mimi_logits = self.adapter.out_adapter(gemma_out)

        loss = None
        if target_mimi_tokens is not None:
            # Cross-entropy loss for target audio token prediction
            B, T, V = mimi_logits.shape
            loss = F.cross_entropy(mimi_logits.view(-1, V), target_mimi_tokens.view(-1))

        return {
            "logits": mimi_logits,
            "gemma_hidden": gemma_out,
            "loss": loss
        }

    def generate_frame(self, mimi_frame_emb: torch.Tensor) -> torch.Tensor:
        """Processes a single 80ms audio frame (12.5 Hz streaming)."""
        with torch.no_grad():
            gemma_in = self.adapter.in_adapter(mimi_frame_emb)
            gemma_out = self.gemma_backbone_sim(gemma_in) + gemma_in
            logits = self.adapter.out_adapter(gemma_out)
            token_ids = torch.argmax(logits, dim=-1)
            return token_ids


if __name__ == "__main__":
    print("[MichiModel] Self-test initializing...")
    model = MichiSpeechToSpeechModel()
    dummy_input = torch.randn(2, 16, 1024)
    dummy_target = torch.randint(0, 4096, (2, 16))
    out = model.forward_speech_to_speech(dummy_input, dummy_target)
    print(f"[MichiModel] Logits shape: {out['logits'].shape}, Loss: {out['loss'].item():.4f}")
    print("[MichiModel] Self-test complete.")
