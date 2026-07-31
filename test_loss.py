import torch
import torch.nn as nn
import torch.nn.functional as F

mimi = torch.randn(32, 64, 1024)
W = torch.randn(1024, 2048) * 0.05
gemma = mimi @ W

# Test 1: pure Cosine loss
proj1 = nn.Linear(1024, 2048)
opt1 = torch.optim.AdamW(proj1.parameters(), lr=3e-3)
for _ in range(100):
    opt1.zero_grad()
    loss1 = 1.0 - F.cosine_similarity(proj1(mimi), gemma, dim=-1).mean()
    loss1.backward()
    opt1.step()
print("Pure Cosine Loss after 100 steps:", (1.0 - F.cosine_similarity(proj1(mimi), gemma, dim=-1).mean()).item())

# Test 2: MSE loss
proj2 = nn.Linear(1024, 2048)
opt2 = torch.optim.AdamW(proj2.parameters(), lr=3e-3)
for _ in range(100):
    opt2.zero_grad()
    loss2 = F.mse_loss(proj2(mimi), gemma)
    loss2.backward()
    opt2.step()
print("MSE Loss Cosine sim after 100 steps:", (1.0 - F.cosine_similarity(proj2(mimi), gemma, dim=-1).mean()).item())
