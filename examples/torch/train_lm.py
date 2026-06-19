"""Tiny language-model training loop on the Alloy backend, checked against eager."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import alloy_torch  # noqa: F401  imports register the "alloy" backend
from alloy_torch.training import set_training_mode

set_training_mode(True)  # before torch.compile


class TinyLM(nn.Module):
    def __init__(self, vocab=64, d=64, heads=4):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.n_heads = heads
        self.d_head = d // heads
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.head = nn.Linear(d, vocab)

    def forward(self, idx):
        x = self.emb(idx)
        B, T, C = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(B, T, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, C))
        x = x + self.ff(self.norm2(x))
        return self.head(x)


def make_model() -> nn.Module:
    torch.manual_seed(0)
    return TinyLM()


def train(backend: str, steps: int = 12, lr: float = 0.02) -> list[float]:
    torch._dynamo.reset()
    model = make_model()
    torch.manual_seed(1)
    x = torch.randint(0, 64, (4, 16))
    y = torch.randint(0, 64, (4, 16))
    step = torch.compile(model, backend="alloy") if backend == "alloy" else model
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(step(x).reshape(-1, 64), y.reshape(-1))
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return losses


def main() -> None:
    alloy = train("alloy")
    eager = train("eager")
    err = max(abs(a - e) for a, e in zip(alloy, eager))
    print(f"SGD + cross-entropy, 12 steps: loss {alloy[0]:.4f} -> {alloy[-1]:.4f}; max_abs_err vs eager={err:.2e}")
    # The embedding backward scatter-adds atomically (sum order non-deterministic
    # at f32-ULP), so the trajectory tracks eager within a loose band.
    assert err < 1e-2
    print("PASSED")


if __name__ == "__main__":
    main()
