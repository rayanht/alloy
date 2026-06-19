"""Multi-step training loop checked against CPU eager. The single forward+backward
grad tests only reach the run-0 handler path; a loop covers the plan-replay path
(run 1+) the optimizer drives."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

import alloy_torch  # noqa: F401 — registers the "alloy" backend
from alloy_torch.training import set_training_mode

set_training_mode(True)


def _train(
    make_model: Callable[[], nn.Module],
    make_data: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    *,
    backend: str,
    steps: int,
    lr: float,
    optim: str,
) -> list[float]:
    torch._dynamo.reset()
    model = make_model()
    x, y = make_data()
    fn = torch.compile(model, backend="alloy", dynamic=False) if backend == "alloy" else model
    opt_cls = torch.optim.AdamW if optim == "adamw" else torch.optim.SGD
    opt = opt_cls(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = F.mse_loss(fn(x), y)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().to("cpu")))
    return losses


def _check_training(
    make_model: Callable[[], nn.Module],
    make_data: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    *,
    steps: int = 8,
    lr: float = 0.05,
    optim: str = "sgd",
    atol: float = 1e-3,
) -> None:
    ref = _train(make_model, make_data, backend="cpu", steps=steps, lr=lr, optim=optim)
    got = _train(make_model, make_data, backend="alloy", steps=steps, lr=lr, optim=optim)
    assert all(v == v for v in got), f"alloy training produced NaN: {got}"
    for i, (a, r) in enumerate(zip(got, ref)):
        assert abs(a - r) <= atol + 1e-2 * abs(r), (
            f"step {i}: alloy loss {a:.6f} != cpu-eager {r:.6f} (diff {abs(a - r):.2e})\n"
            f"  alloy={['%.4f' % v for v in got]}\n  ref  ={['%.4f' % v for v in ref]}"
        )


def _mlp() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))


def _mlp_layernorm() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(32, 64), nn.LayerNorm(64), nn.ReLU(), nn.Linear(64, 1))


def _regression_data() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)
    x = torch.randn(16, 32)
    w = torch.randn(32, 1)
    return x, x @ w + 0.1 * torch.randn(16, 1)


class TestTrainingLoop:
    def test_mlp_sgd_trajectory(self):
        _check_training(_mlp, _regression_data, steps=10, lr=0.02)

    def test_mlp_layernorm_sgd_trajectory(self):
        _check_training(_mlp_layernorm, _regression_data, steps=10, lr=0.02)

    def test_loss_moves_every_step(self):
        # A skipped optimizer update freezes the forward for a step (loss[i] ==
        # loss[i-1]); the loss must strictly move on every step.
        got = _train(_mlp, _regression_data, backend="alloy", steps=6, lr=0.05, optim="sgd")
        for i in range(1, len(got)):
            assert got[i] != got[i - 1], f"loss frozen at step {i}: {got}"

    def test_plain_mm_covers_full_output(self):
        # Non-transposed mm (aten.mm with a contiguous rhs) launches plain `dot`,
        # which has no dispatch_spec — an auto-grid would compute only one tile.
        # Exercise an N that maps to a single N-tile so a one-tile launch would
        # silently pass on N but drop the M rows past the first tile.
        torch.manual_seed(0)
        for m, k, n in [(64, 64, 32), (64, 128, 32), (128, 128, 48)]:
            a, b = torch.randn(m, k), torch.randn(k, n)
            torch._dynamo.reset()
            got = torch.compile(lambda x, y: x @ y, backend="alloy", dynamic=False)(a, b)
            torch.testing.assert_close(got, a @ b, atol=1e-4, rtol=1e-4)
