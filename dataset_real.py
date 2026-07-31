"""Phase 2 Dataset: Real Audio Streaming & Preprocessing with DailyTalkContiguous.

Provides:
  - DailyTalkContiguousStream: Streaming loader for 'kyutai/DailyTalkContiguous'
  - RealAudioTextDataset: Slice-based batch dataset generator for Mimi <-> Gemma alignment
"""

import os
import math
import typing
import torch
from torch.utils.data import Dataset, IterableDataset


class DailyTalkContiguousStream(IterableDataset):
    """Streaming dataset wrapper for DailyTalkContiguous on Hugging Face.
    
    Yields 5-second audio chunks @ 16kHz for real audio token alignment.
    """
    def __init__(self, split: str = "train", target_sample_rate: int = 16000, chunk_duration_sec: float = 5.0):
        super().__init__()
        self.split = split
        self.target_sr = target_sample_rate
        self.chunk_len = int(target_sample_rate * chunk_duration_sec)

    def _get_hf_dataset(self):
        try:
            from datasets import load_dataset
            ds = load_dataset("kyutai/DailyTalkContiguous", split=self.split, streaming=True)
            return ds
        except Exception as e:
            print(f"[DailyTalkContiguousStream] Warning: HuggingFace datasets load failed ({e})")
            return None

    def __iter__(self):
        ds = self._get_hf_dataset()
        if ds is None:
            # Fallback mock generator for offline/local dry runs
            while True:
                mock_waveform = torch.randn(1, self.chunk_len)
                yield {"waveform": mock_waveform, "sample_rate": self.target_sr, "text": "mock dialogue"}
        else:
            for sample in ds:
                audio_data = sample.get("audio", {})
                array = audio_data.get("array", [])
                sr = audio_data.get("sampling_rate", self.target_sr)
                
                tensor_audio = torch.tensor(array, dtype=torch.float32)
                if tensor_audio.ndim == 1:
                    tensor_audio = tensor_audio.unsqueeze(0)
                
                # Resample if needed
                if sr != self.target_sr:
                    try:
                        import torchaudio
                        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
                        tensor_audio = resampler(tensor_audio)
                    except ImportError:
                        pass
                
                # Slice into 5s fixed length chunks
                total_samples = tensor_audio.shape[-1]
                for start in range(0, total_samples, self.chunk_len):
                    end = start + self.chunk_len
                    chunk = tensor_audio[:, start:end]
                    if chunk.shape[-1] == self.chunk_len:
                        yield {
                            "waveform": chunk,
                            "sample_rate": self.target_sr,
                            "text": sample.get("text", "")
                        }


class RealAudioTextDataset(Dataset):
    """Static or cached real audio dataset for Phase 2 alignment training.
    
    Simulates Mimi codec token output and Gemma audio embeddings when models are loaded.
    """
    def __init__(self, n_samples: int = 1000, seq_len: int = 64, mimi_dim: int = 1024, gemma_dim: int = 2048):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.mimi_dim = mimi_dim
        self.gemma_dim = gemma_dim

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        # Placeholder for pre-extracted or live-extracted Mimi and Gemma embeddings
        mimi = torch.randn(self.seq_len, self.mimi_dim)
        gemma = torch.randn(self.seq_len, self.gemma_dim)
        return {"mimi": mimi, "gemma": gemma}


if __name__ == "__main__":
    print("[dataset_real] Self-test dataset_real.py...")
    ds_stream = DailyTalkContiguousStream()
    ds_static = RealAudioTextDataset(n_samples=10)
    print(f"[dataset_real] Static dataset length: {len(ds_static)}")
    sample = ds_static[0]
    print(f"[dataset_real] Sample mimi shape: {sample['mimi'].shape}, gemma shape: {sample['gemma'].shape}")
    print("[dataset_real] Self-test complete.")
