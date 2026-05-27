import torch
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
p1 = torch.randn(256, 500, 3, device=device)
p2 = torch.randn(256, 500, 3, device=device)

start = time.time()
for _ in range(100):
    dist = torch.cdist(p1, p2)
    loss = dist.min(dim=2)[0].mean() + dist.min(dim=1)[0].mean()
torch.cuda.synchronize() if torch.cuda.is_available() else None
print(f"100 iterations time: {time.time() - start:.4f} s")
