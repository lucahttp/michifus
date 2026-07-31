"""Local Training Script for Michi-Adapter (0$ Cloud Cost).

Runs 100% on local hardware (AMD RX 6800 ROCm / DirectML / CPU).
VRAM footprint: < 0.5 GB for Adapter alone, ~3.5 GB with Gemma 4 E2B QAT.
"""

import time
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from adapter import MichiAdapter
from dataset import SyntheticAudioText


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print(f"  LOCAL MICHI-ADAPTER TRAINING (Device: {device.upper()})")
    print("=" * 60)

    if device == "cuda":
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        free, total = torch.cuda.mem_get_info()
        print(f"✓ VRAM: Free={free/1e9:.2f}GB / Total={total/1e9:.2f}GB")
    else:
        print("✓ Running on Local CPU (no cloud cost, ~10-20 seconds per 200 steps)")

    adapter = MichiAdapter().to(device)
    n_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    print(f"✓ Trainable params: {n_params:,} (~{n_params*4/1e6:.1f} MB)")

    steps = 200
    batch_size = 32
    ds = SyntheticAudioText(n_samples=steps * batch_size + 64, seq_len=64)
    dl = DataLoader(ds, batch_size=batch_size)

    opt = AdamW(adapter.parameters(), lr=3e-3, weight_decay=1e-5)
    sched = CosineAnnealingLR(opt, T_max=steps)
    
    t0 = time.time()
    it = iter(dl)

    for step in range(1, steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)

        mimi = batch["mimi"].to(device)
        gemma = batch["gemma"].to(device)

        loss = adapter.alignment_loss(mimi, gemma)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 50 == 0 or step == 1:
            print(f"  [Local Step {step:3d}/{steps}] loss={loss.item():.4f} lr={sched.get_last_lr()[0]:.2e} elapsed={time.time()-t0:.2f}s")

    out_path = "adapter_local.pt"
    torch.save({
        "in_adapter": adapter.in_adapter.state_dict(),
        "out_adapter": adapter.out_adapter.state_dict(),
        "config": {"mimi_dim": 1024, "gemma_dim": 2048, "mimi_vocab": 4096}
    }, out_path)

    print("-" * 60)
    print(f"✅ Local training complete in {time.time()-t0:.2f} seconds!")
    print(f"✅ Saved local checkpoint to: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
