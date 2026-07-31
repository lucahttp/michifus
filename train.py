"""Phase 1: synthetic alignment training for the Mimi->Gemma adapter.

Run on the H100 studio:
    /system/conda/miniconda3/bin/python train.py --steps 200 --batch_size 32

Goal: prove the adapter converges to align a fixed synth projection W. Loss
should drop from ~1.0 -> ~0.05 (noise floor). VRAM target: well under 1 GB."""

import argparse
import time

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from adapter import MichiAdapter
from dataset import SyntheticAudioText


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--save_path", type=str, default="adapter.pt")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}", flush=True)
    if device == "cuda":
        print(f"[train] GPU={torch.cuda.get_device_name(0)}", flush=True)
        free, total = torch.cuda.mem_get_info()
        print(f"[train] VRAM free={free/1e9:.1f}GB total={total/1e9:.1f}GB", flush=True)

    adapter = MichiAdapter().to(device)
    n_params = sum(t.numel() for t in adapter.parameters() if t.requires_grad)
    print(f"[train] adapter trainable params={n_params:,}", flush=True)

    ds = SyntheticAudioText(
        n_samples=args.steps * args.batch_size + 64,
        seq_len=args.seq_len,
        seed=args.seed,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=0)

    opt = AdamW(adapter.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = CosineAnnealingLR(opt, T_max=args.steps)
    adapter.train()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    it = iter(dl)
    t0 = time.time()
    window = []

    for step in range(1, args.steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)

        mimi = batch["mimi"].to(device, non_blocking=True)
        gemma = batch["gemma"].to(device, non_blocking=True)

        loss = adapter.alignment_loss(mimi, gemma)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        opt.step()
        sched.step()
        window.append(loss.item())

        if step % args.log_every == 0 or step == 1:
            avg = sum(window) / len(window)
            window = []
            vram = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0
            print(
                f"[train] step={step}/{args.steps} loss={avg:.4f} "
                f"lr={sched.get_last_lr()[0]:.2e} vram={vram:.2f}GB "
                f"elapsed={time.time()-t0:.1f}s",
                flush=True,
            )

    out = args.save_path
    torch.save(
        {
            "in_adapter": adapter.in_adapter.state_dict(),
            "out_adapter": adapter.out_adapter.state_dict(),
            "config": {"mimi_dim": 1024, "gemma_dim": 2048, "mimi_vocab": 4096},
            "args": vars(args),
        },
        out,
    )
    print(f"[train] saved -> {out}", flush=True)
    print(f"[train] done in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
