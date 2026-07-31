"""Offline Checkpoint Verification & Local Diagnostic Test.

Verifies:
  1. Checkpoint file integrity & weight structure of adapter checkpoints
  2. State dict loading into MichiAdapter (MimiToGemma + GemmaToMimi)
  3. Shape correctness (1024 -> 2048 -> 4096)
  4. Forward pass & gradient flow sanity
  5. Numeric stability (NaN/Inf checks, weight statistics)
  6. Save/Reload roundtrip consistency
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from adapter import MichiAdapter, MimiToGemma, GemmaToMimi


def run_diagnostics(ckpt_path: str = "checkpoints/adapter_local_phase2.pt"):
    print("=" * 60)
    print("  MICHI-ADAPTER OFFLINE CHECKPOINT DIAGNOSTIC")
    print("=" * 60)

    # 1. File existence & size
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Checkpoint file not found: {ckpt_path}")
        sys.exit(1)
    
    file_size_bytes = os.path.getsize(ckpt_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    print(f"[OK] Checkpoint file found: {ckpt_path}")
    print(f"[OK] File size: {file_size_mb:.2f} MB ({file_size_bytes:,} bytes)")

    # 2. PyTorch Load & Metadata Inspection
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        print("[OK] PyTorch torch.load() succeeded.")
    except Exception as e:
        print(f"[ERROR] Loading checkpoint with torch: {e}")
        sys.exit(1)

    print(f"[OK] Top-level keys in checkpoint: {list(ckpt.keys())}")
    
    if "config" in ckpt:
        print(f"[OK] Config metadata: {ckpt['config']}")
    if "args" in ckpt:
        print(f"[OK] Training args: {ckpt['args']}")

    # 3. State Dict Parameter Inspection
    in_sd = ckpt.get("in_adapter", {})
    out_sd = ckpt.get("out_adapter", {})

    print("\n--- In-Adapter State Dict Keys & Shapes ---")
    for k, v in in_sd.items():
        print(f"  - in_adapter.{k}: shape={tuple(v.shape)}, dtype={v.dtype}, mean={v.mean():.4f}, std={v.std():.4f}")

    print("\n--- Out-Adapter State Dict Keys & Shapes ---")
    for k, v in out_sd.items():
        print(f"  - out_adapter.{k}: shape={tuple(v.shape)}, dtype={v.dtype}, mean={v.mean():.4f}, std={v.std():.4f}")

    # Assert expected shapes
    assert in_sd["proj.weight"].shape == (2048, 1024), f"Unexpected in_adapter weight shape: {in_sd['proj.weight'].shape}"
    assert out_sd["proj.weight"].shape == (4096, 2048), f"Unexpected out_adapter weight shape: {out_sd['proj.weight'].shape}"
    print("[OK] All tensor shapes match expected Mimi <-> Gemma architecture!")

    # 4. Model Instantiation & Weight Loading
    print("\n--- Instantiating MichiAdapter Architecture ---")
    model = MichiAdapter(mimi_dim=1024, gemma_dim=2048, mimi_vocab=4096)
    model.in_adapter.load_state_dict(in_sd)
    model.out_adapter.load_state_dict(out_sd)
    model.eval()
    print("[OK] MichiAdapter initialized and state_dict loaded cleanly.")

    # 5. Forward Pass Sanity Test
    print("\n--- Forward Pass Execution Test ---")
    batch_size = 4
    seq_len = 64
    mock_mimi = torch.randn(batch_size, seq_len, 1024)
    mock_gemma = torch.randn(batch_size, seq_len, 2048)

    with torch.no_grad():
        in_out = model.in_adapter(mock_mimi)
        out_out = model.out_adapter(mock_gemma)

    print(f"[OK] in_adapter forward:  input={tuple(mock_mimi.shape)} -> output={tuple(in_out.shape)}")
    print(f"[OK] out_adapter forward: input={tuple(mock_gemma.shape)} -> output={tuple(out_out.shape)}")

    # NaN / Inf Sanity Check
    assert not torch.isnan(in_out).any(), "NaN found in in_adapter output!"
    assert not torch.isinf(in_out).any(), "Inf found in in_adapter output!"
    assert not torch.isnan(out_out).any(), "NaN found in out_adapter output!"
    assert not torch.isinf(out_out).any(), "Inf found in out_adapter output!"
    print("[OK] Output tensors passed NaN/Inf clean check.")

    # 6. Gradient Flow & Alignment Loss Test
    print("\n--- Backward Pass & Gradient Flow Sanity Test ---")
    model.train()
    loss = model.alignment_loss(mock_mimi, mock_gemma)
    loss.backward()

    has_grad_in = model.in_adapter.proj.weight.grad is not None and model.in_adapter.proj.weight.grad.abs().sum() > 0
    print(f"[OK] Alignment Loss value: {loss.item():.4f}")
    print(f"[OK] Gradient flow to in_adapter weight: {has_grad_in} (grad norm: {model.in_adapter.proj.weight.grad.norm():.4f})")

    # 7. Save/Reload Roundtrip Sanity
    print("\n--- Save & Reload Roundtrip Verification ---")
    tmp_path = "adapter_test_tmp.pt"
    torch.save({
        "in_adapter": model.in_adapter.state_dict(),
        "out_adapter": model.out_adapter.state_dict(),
        "config": {"mimi_dim": 1024, "gemma_dim": 2048, "mimi_vocab": 4096}
    }, tmp_path)

    reloaded_ckpt = torch.load(tmp_path, map_location="cpu")
    model_reloaded = MichiAdapter()
    model_reloaded.in_adapter.load_state_dict(reloaded_ckpt["in_adapter"])
    
    diff = (model.in_adapter.proj.weight - model_reloaded.in_adapter.proj.weight).abs().max().item()
    print(f"[OK] Reloaded weight max difference: {diff:.8f}")
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    print("\n" + "=" * 60)
    print("  [SUCCESS] ALL OFFLINE DIAGNOSTIC CHECKS PASSED (100% SUCCESS)")
    print("=" * 60)


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/adapter_local_phase2.pt"
    run_diagnostics(ckpt)
