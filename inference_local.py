"""Real-Time Local Streaming Inference Benchmark.

Loads D:\michi-adapter\checkpoints\adapter_local_phase2.pt and measures per-frame processing latency.
Target: < 80ms per 12.5 Hz audio frame.
"""

import time
import torch
from adapter import MichiAdapter


def benchmark_inference(ckpt_path: str = "checkpoints/adapter_local_phase2.pt", n_frames: int = 100):
    print("=" * 60)
    print("  MICHI-ADAPTER REAL-TIME STREAMING INFERENCE BENCHMARK")
    print("=" * 60)

    device = "cpu"
    adapter = MichiAdapter().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    adapter.in_adapter.load_state_dict(ckpt["in_adapter"])
    adapter.out_adapter.load_state_dict(ckpt["out_adapter"])
    adapter.eval()

    print(f"[OK] Loaded weights from: {ckpt_path}")
    print(f"[OK] Running {n_frames} simulated 12.5 Hz streaming frames (1 frame = 80ms audio window)...")

    latencies = []

    # Warmup
    for _ in range(10):
        mimi_frame = torch.randn(1, 1, 1024)
        _ = adapter.in_adapter(mimi_frame)

    with torch.no_grad():
        for i in range(n_frames):
            mimi_frame = torch.randn(1, 1, 1024)
            gemma_sim_state = torch.randn(1, 1, 2048)

            t0 = time.perf_counter()
            
            # Step A: Mimi -> Gemma (in_adapter)
            gemma_emb = adapter.in_adapter(mimi_frame)
            
            # Step B: Gemma -> Mimi logits (out_adapter)
            mimi_logits = adapter.out_adapter(gemma_sim_state)
            
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(dt_ms)

    avg_ms = sum(latencies) / len(latencies)
    p95_ms = sorted(latencies)[int(0.95 * len(latencies))]
    min_ms = min(latencies)
    max_ms = max(latencies)

    print("-" * 60)
    print(f"  Avg Latency per Frame:  {avg_ms:.3f} ms")
    print(f"  P95 Latency per Frame:  {p95_ms:.3f} ms")
    print(f"  Min / Max Latency:     {min_ms:.3f} ms / {max_ms:.3f} ms")
    print(f"  Real-Time Budget (80ms): {'PASS (Ultra-Fast)' if avg_ms < 80 else 'FAIL'}")
    print("=" * 60)


if __name__ == "__main__":
    benchmark_inference()
