"""MLP training loop on the Alloy backend, checked against eager."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import alloy_torch  # noqa: F401  imports register the "alloy" backend
from alloy_torch.training import set_training_mode

set_training_mode(True)  # before torch.compile


def make_model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(64, 128), nn.LayerNorm(128), nn.GELU(), nn.Linear(128, 1))


def train(backend: str, steps: int = 20, lr: float = 0.05) -> list[float]:
    torch._dynamo.reset()
    model = make_model()
    torch.manual_seed(1)
    x, y = torch.randn(32, 64), torch.randn(32, 1)
    step = torch.compile(model, backend="alloy") if backend == "alloy" else model
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = F.mse_loss(step(x), y)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return losses


def main() -> None:
    alloy = train("alloy")
    eager = train("eager")
    err = max(abs(a - e) for a, e in zip(alloy, eager))
    print(f"AdamW, 20 steps: loss {alloy[0]:.4f} -> {alloy[-1]:.4f}; max_abs_err vs eager={err:.2e}")
    assert err < 1e-3
    print("PASSED")


if __name__ == "__main__":
    main()
