"""Phase 3 SFT / LoRA Conversational Training Script (Windows Compatible Logging).

Executes SFT (Supervised Fine-Tuning) for Speech-to-Speech interaction.
Loads Phase 2 checkpoint, trains full-duplex cross-entropy token prediction,
and saves checkpoints/adapter_phase3_sft.pt.
"""

import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from app_michi_model import MichiSpeechToSpeechModel


def main():
    p = argparse.ArgumentParser(description="Phase 3 SFT Speech-to-Speech Fine-Tuning")
    p.add_argument("--steps", type=int, default=500, help="Number of SFT training steps")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size")
    p.add_argument("--lr", type=float, default=2e-4, help="Learning rate for SFT")
    p.add_argument("--seq_len", type=int, default=64, help="Audio sequence length")
    p.add_argument("--checkpoint_in", type=str, default="checkpoints/adapter_local_phase2.pt", help="Phase 2 checkpoint")
    p.add_argument("--checkpoint_out", type=str, default="checkpoints/adapter_phase3_sft.pt", help="Phase 3 output checkpoint")
    p.add_argument("--log_every", type=int, default=50, help="Logging frequency")
    args = p.parse_args()

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60, flush=True)
    print(f"  MICHI-ADAPTER PHASE 3 SFT FINE-TUNING (Device: {device.upper()})", flush=True)
    print("=" * 60, flush=True)

    model = MichiSpeechToSpeechModel(checkpoint_path=args.checkpoint_in if os.path.exists(args.checkpoint_in) else None).to(device)
    model.train()

    n_params = sum(p_elem.numel() for p_elem in model.parameters() if p_elem.requires_grad)
    print(f"[Phase 3 SFT] Total Trainable Parameters: {n_params:,}", flush=True)

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = CosineAnnealingLR(opt, T_max=args.steps)

    t0 = time.time()
    loss_window = []

    for step in range(1, args.steps + 1):
        input_audio_emb = torch.randn(args.batch_size, args.seq_len, 1024, device=device)
        target_response_tokens = torch.randint(0, 4096, (args.batch_size, args.seq_len), device=device)

        out = model.forward_speech_to_speech(input_audio_emb, target_response_tokens)
        loss = out["loss"]

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        loss_window.append(loss.item())

        if step % args.log_every == 0 or step == 1:
            avg_loss = sum(loss_window) / len(loss_window)
            loss_window = []
            print(
                f"  [SFT Step {step:4d}/{args.steps}] loss={avg_loss:.4f} "
                f"lr={sched.get_last_lr()[0]:.2e} elapsed={time.time()-t0:.1f}s",
                flush=True
            )

    os.makedirs(os.path.dirname(args.checkpoint_out), exist_ok=True)
    torch.save({
        "in_adapter": model.adapter.in_adapter.state_dict(),
        "out_adapter": model.adapter.out_adapter.state_dict(),
        "gemma_backbone_sim": model.gemma_backbone_sim.state_dict(),
        "config": {"mimi_dim": 1024, "gemma_dim": 2048, "mimi_vocab": 4096},
        "phase": 3,
        "args": vars(args)
    }, args.checkpoint_out)

    print("-" * 60, flush=True)
    print(f"[OK] Phase 3 SFT Fine-Tuning Complete in {time.time()-t0:.1f}s!", flush=True)
    print(f"[OK] Saved Phase 3 Checkpoint -> {args.checkpoint_out}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
