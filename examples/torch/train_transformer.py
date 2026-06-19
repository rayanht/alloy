"""Transformer-block training loop on the Alloy backend, checked against eager."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import alloy_torch  # noqa: F401  imports register the "alloy" backend
from alloy_torch.training import set_training_mode

set_training_mode(True)  # before torch.compile


class Classifier(nn.Module):
    def __init__(self, d_model=128, n_heads=4, d_ff=512, vocab=64):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.head = nn.Linear(d_model, vocab)

    def forward(self, x):
        B, T, C = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, T, C)
        x = x + self.proj(attn)
        x = x + self.ff(self.norm2(x))
        return self.head(x)


def make_model() -> nn.Module:
    torch.manual_seed(0)
    return Classifier()


def train(backend: str, steps: int = 15, lr: float = 0.05) -> list[float]:
    torch._dynamo.reset()
    model = make_model()
    torch.manual_seed(1)
    x = torch.randn(4, 16, 128)
    labels = torch.randint(0, 64, (4, 16))
    step = torch.compile(model, backend="alloy") if backend == "alloy" else model
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(step(x).reshape(-1, 64), labels.reshape(-1))
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return losses


def main() -> None:
    alloy = train("alloy")
    eager = train("eager")
    err = max(abs(a - e) for a, e in zip(alloy, eager))
    print(f"SGD + cross-entropy, 15 steps: loss {alloy[0]:.4f} -> {alloy[-1]:.4f}; max_abs_err vs eager={err:.2e}")
    assert err < 1e-3
    print("PASSED")


if __name__ == "__main__":
    main()
