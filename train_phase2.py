"""Phase 2 Training Script: Real Audio & Gemma 4 E2B Alignment.

Run on H100 / A100 Studio:
    /system/conda/miniconda3/bin/python train_phase2.py --steps 1000 --batch_size 32 --checkpoint_in adapter.pt

Phase 2 warm-starts from Phase 1 adapter.pt, loads real audio (or streaming DailyTalkContiguous),
and aligns Mimi embeddings to Gemma 4 E2B hidden space.
"""

import argparse
import time
import os
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from adapter import MichiAdapter
from dataset_real import RealAudioTextDataset, DailyTalkContiguousStream


def main():
    p = argparse.ArgumentParser(description="Michi-Adapter Phase 2 Alignment Training")
    p.add_argument("--steps", type=int, default=1000, help="Number of training steps")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4 for warm-start)")
    p.add_argument("--seq_len", type=int, default=64, help="Sequence length of audio frames")
    p.add_argument("--checkpoint_in", type=str, default="adapter.pt", help="Phase 1 checkpoint to warm-start from")
    p.add_argument("--checkpoint_out", type=str, default="adapter_phase2.pt", help="Output Phase 2 checkpoint path")
    p.add_argument("--log_every", type=int, default=10, help="Logging frequency in steps")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--use_gemma_hf", action="store_true", help="Load real HF google/gemma-4-E2B if available")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train_phase2] device={device}", flush=True)
    if device == "cuda":
        print(f"[train_phase2] GPU={torch.cuda.get_device_name(0)}", flush=True)
        free, total = torch.cuda.mem_get_info()
        print(f"[train_phase2] VRAM free={free/1e9:.1f}GB total={total/1e9:.1f}GB", flush=True)

    adapter = MichiAdapter().to(device)
    
    # Warm-start from Phase 1 checkpoint if available
    if os.path.exists(args.checkpoint_in):
        print(f"[train_phase2] Warm-starting from checkpoint: {args.checkpoint_in}", flush=True)
        ckpt = torch.load(args.checkpoint_in, map_location=device)
        if "in_adapter" in ckpt:
            adapter.in_adapter.load_state_dict(ckpt["in_adapter"])
            print("[train_phase2] Loaded in_adapter state_dict successfully.", flush=True)
        if "out_adapter" in ckpt:
            adapter.out_adapter.load_state_dict(ckpt["out_adapter"])
            print("[train_phase2] Loaded out_adapter state_dict successfully.", flush=True)
    else:
        print(f"[train_phase2] Warning: Checkpoint {args.checkpoint_in} not found. Starting from scratch.", flush=True)

    n_params = sum(t.numel() for t in adapter.parameters() if t.requires_grad)
    print(f"[train_phase2] Adapter trainable parameters: {n_params:,}", flush=True)

    # Optional Gemma HF model loader for real embedding extraction
    gemma_model = None
    if args.use_gemma_hf:
        try:
            from transformers import AutoModel
            print("[train_phase2] Loading google/gemma-4-E2B from HuggingFace...", flush=True)
            gemma_model = AutoModel.from_pretrained("google/gemma-4-E2B", torch_dtype=torch.bfloat16).to(device)
            gemma_model.eval()
            for p_elem in gemma_model.parameters():
                p_elem.requires_grad = False
            print("[train_phase2] Gemma 4 E2B loaded and frozen.", flush=True)
        except Exception as e:
            print(f"[train_phase2] Could not load HF Gemma ({e}). Using audio-text projection alignment.", flush=True)

    ds = RealAudioTextDataset(
        n_samples=args.steps * args.batch_size + 128,
        seq_len=args.seq_len,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

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
                f"[train_phase2] step={step}/{args.steps} loss={avg:.4f} "
                f"lr={sched.get_last_lr()[0]:.2e} vram={vram:.2f}GB "
                f"elapsed={time.time()-t0:.1f}s",
                flush=True,
            )

    out = args.checkpoint_out
    torch.save(
        {
            "in_adapter": adapter.in_adapter.state_dict(),
            "out_adapter": adapter.out_adapter.state_dict(),
            "config": {"mimi_dim": 1024, "gemma_dim": 2048, "mimi_vocab": 4096},
            "phase": 2,
            "args": vars(args),
        },
        out,
    )
    print(f"[train_phase2] Saved Phase 2 checkpoint -> {out}", flush=True)
    print(f"[train_phase2] Done in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
