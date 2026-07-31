"""Mimi <-> Gemma adapter for Escenario B (backbone swap)."""
import torch
import torch.nn as nn


class MimiToGemma(nn.Module):
    """Mimi token embeddings (1024) -> Gemma hidden space (2048)."""
    def __init__(self, mimi_dim: int = 1024, gemma_dim: int = 2048):
        super().__init__()
        self.proj = nn.Linear(mimi_dim, gemma_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, mimi_dim]
        return self.proj(x)



class GemmaToMimi(nn.Module):
    """Gemma hidden states (2048) -> Mimi codebook logits (4096)."""
    def __init__(self, gemma_dim: int = 2048, mimi_vocab: int = 4096):
        super().__init__()
        self.proj = nn.Linear(gemma_dim, mimi_vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class MichiAdapter(nn.Module):
    """Wraps both adapters. ~6M params."""
    def __init__(self, mimi_dim: int = 1024, gemma_dim: int = 2048, mimi_vocab: int = 4096):
        super().__init__()
        self.in_adapter = MimiToGemma(mimi_dim, gemma_dim)
        self.out_adapter = GemmaToMimi(gemma_dim, mimi_vocab)

    def alignment_loss(self, mimi_tokens: torch.Tensor, gemma_emb: torch.Tensor) -> torch.Tensor:
        """Phase 1 loss: MSE + Cosine similarity between adapter(Mimi) and Gemma."""
        projected = self.in_adapter(mimi_tokens)
        mse = torch.nn.functional.mse_loss(projected, gemma_emb)
        cos = 1.0 - torch.nn.functional.cosine_similarity(projected, gemma_emb, dim=-1).mean()
        return mse + cos



