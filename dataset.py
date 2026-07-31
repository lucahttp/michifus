"""Dataset loaders. Phase 1 uses synthetic data with a FIXED projection W so the
alignment task is actually learnable; Phase 2 swaps in DailyTalkContiguous.

Why fixed W: if W is random per sample the model can trivially succeed. With one
shared W the adapter must learn a *function* that approximates W over a manifold
of input tokens. Loss should converge toward the irreducible noise floor."""

import math

import torch
from torch.utils.data import Dataset


class SyntheticAudioText(Dataset):
    """Learnable synthetic alignment data.

    Mimics the Mimi->Gemma relationship:
      - mimi tokens: float vectors in R^{mimi_dim}, drawn from K fixed "prototypes"
        with small Gaussian noise so the dataset is dense on the manifold.
      - gemma emb: same prototype passed through a FIXED W_projection.
    The adapter should learn to invert-and-reproject, i.e. find W_tilde ~= W.
    """
    def __init__(self, n_samples: int = 2000, mimi_dim: int = 1024,
                 gemma_dim: int = 2048, seq_len: int = 50,
                 n_prototypes: int = 256, noise_std: float = 0.05,
                 seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.seq_len = seq_len
        self.mimi_dim = mimi_dim
        self.gemma_dim = gemma_dim
        self.noise_std = noise_std
        P = torch.randn(n_prototypes, mimi_dim, generator=g)
        self.P = torch.nn.functional.normalize(P, dim=-1)
        bound = math.sqrt(6.0 / (mimi_dim + gemma_dim))
        self.W = (torch.rand(mimi_dim, gemma_dim, generator=g) * 2 - 1) * bound
        idx = torch.randint(0, n_prototypes, (n_samples, seq_len), generator=g)
        self.idx = idx
        self.n = n_samples

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        ids = self.idx[i]
        mimi = self.P[ids] + self.noise_std * torch.randn_like(self.P[ids])
        gemma = mimi @ self.W
        return {"mimi": mimi, "gemma": gemma}


class DailyTalkStreaming(Dataset):
    """Real audio via Mimi codec. Phase 2 hook — not loaded by Phase 1."""
    def __init__(self, seq_len: int = 50, split: str = "train"):
        from datasets import load_dataset
        self.ds = load_dataset("kyutai/DailyTalkContiguous", split=split, streaming=True)
        self.seq_len = seq_len

    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        sample = next(iter(self.ds))
        audio = torch.tensor(sample["audio"]["array"][:16000 * 5])
        return {"audio_raw": audio}
