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


def _lm() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Embedding(50, 32), nn.Linear(32, 50))


def _lm_data() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)
    return torch.randint(0, 50, (4, 16)), torch.randint(0, 50, (4, 16))


def _train_ce(make_model, make_data, *, backend, steps, lr) -> list[float]:
    torch._dynamo.reset()
    model = make_model()
    x, y = make_data()
    fn = torch.compile(model, backend="alloy", dynamic=False) if backend == "alloy" else model
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(fn(x).reshape(-1, 50), y.reshape(-1))
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().to("cpu")))
    return losses


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

    def test_embedding_lm_trajectory(self):
        # The embedding backward scatter-adds via an atomic store (sum order
        # non-deterministic at f32-ULP), so the loss tracks eager within a
        # loose band rather than bit-exactly.
        ref = _train_ce(_lm, _lm_data, backend="cpu", steps=10, lr=0.1)
        got = _train_ce(_lm, _lm_data, backend="alloy", steps=10, lr=0.1)
        assert all(v == v for v in got), f"alloy LM training produced NaN: {got}"
        for i, (a, r) in enumerate(zip(got, ref)):
            assert abs(a - r) <= 0.02, f"step {i}: alloy {a:.4f} vs eager {r:.4f}"

    def test_dropout_statistics(self):
        torch._dynamo.reset()
        torch.manual_seed(0)
        fn = torch.compile(lambda x: F.dropout(x, 0.4, training=True), backend="alloy", dynamic=False)
        y = fn(torch.ones(4096)).clone()
        zero_frac = (y == 0).float().mean().item()
        assert 0.35 < zero_frac < 0.45, zero_frac  # ~p zeroed
        surv = y[y != 0]
        assert torch.allclose(surv, torch.full_like(surv, 1 / 0.6), atol=1e-4)  # 1/(1-p)

    def test_dropout_reproducible_and_varies(self):
        torch._dynamo.reset()
        fn = torch.compile(lambda x: F.dropout(x, 0.3, training=True), backend="alloy", dynamic=False)
        x = torch.ones(2048)
        torch.manual_seed(7)
        a = [fn(x).clone() for _ in range(3)]
        torch.manual_seed(7)
        b = [fn(x).clone() for _ in range(3)]
        assert all(torch.equal(a[i], b[i]) for i in range(3))  # reproducible under manual_seed
        assert not torch.equal(a[0], a[1])  # different mask each forward

    def test_dropout_eval_passthrough(self):
        torch._dynamo.reset()
        fn = torch.compile(lambda x: F.dropout(x, 0.5, training=False), backend="alloy", dynamic=False)
        x = torch.randn(64)
        assert torch.equal(fn(x).clone(), x)

    def test_dropout_mlp_trains_reproducibly(self):
        # Trains through native_dropout + native_dropout_backward; bit-exact
        # reproducible under the same seed.
        def run():
            torch._dynamo.reset()
            torch.manual_seed(0)
            m = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 10)).train()
            fn = torch.compile(m, backend="alloy")
            opt = torch.optim.SGD(m.parameters(), lr=0.05)
            torch.manual_seed(1)
            x, y = torch.randn(16, 32), torch.randint(0, 10, (16,))
            out = []
            for _ in range(8):
                opt.zero_grad()
                loss = F.cross_entropy(fn(x), y)
                loss.backward()
                opt.step()
                out.append(float(loss.detach()))
            return out

        a, b = run(), run()
        assert max(abs(x - y) for x, y in zip(a, b)) < 1e-9, "dropout training not reproducible"
        assert a[-1] < a[0], "loss did not decrease"

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
