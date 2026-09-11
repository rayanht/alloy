"""Mixed-dtype training: bf16 params, and bf16 compute against f32 master weights.

Both paths were broken in ways fp32 could never surface — every buffer in an fp32
graph has the same byte size, and every cast is a no-op.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import alloy_torch  # noqa: F401 — registers the "alloy" backend
from alloy_torch.training import set_training_mode

set_training_mode(True)


@pytest.mark.parametrize("rows,cols", [(192, 64), (256, 64), (1024, 64)])
def test_compiled_adamw_over_bf16_params(rows, cols):
    """A compiled AdamW step whose update is computed in f32 must not be routed
    onto the bf16 parameter's storage — the value is twice the buffer's size."""
    torch._dynamo.reset()
    torch.manual_seed(0)
    model = nn.Embedding(rows, cols).to(torch.bfloat16)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    forward = torch.compile(model, backend="alloy", dynamic=False)
    step = torch.compile(opt.step, backend="alloy", dynamic=False)
    index = torch.randint(0, rows, (8,))

    for _ in range(3):
        opt.zero_grad(set_to_none=False)
        forward(index).float().pow(2).mean().backward()
        step()

    assert torch.isfinite(model.weight).all()
    assert model.weight.abs().max() > 0


def test_layernorm_bias_grad_under_autocast():
    """The bias grad reduces a widening cast that a sibling op also consumes.
    Duplicating that cast per consumer must not leave this one reading a buffer
    nothing writes."""

    class NormLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(32)
            self.linear = nn.Linear(32, 32, bias=False)

        def forward(self, x):
            return self.linear(self.norm(x))

    def grads(backend: str):
        torch._dynamo.reset()
        torch.manual_seed(0)
        model = NormLinear()

        def call(x):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                return model(x).float().pow(2).sum()

        fn = call if backend == "eager" else torch.compile(call, backend="alloy", dynamic=False)
        torch.manual_seed(1)
        fn(torch.randn(8, 32)).backward()
        return model.norm.bias.grad.clone(), model.norm.weight.grad.clone()

    ref_bias, ref_weight = grads("eager")
    got_bias, got_weight = grads("alloy")
    assert not (got_bias == 0).all(), "bias grad is all zero — the reduction never ran"
    torch.testing.assert_close(got_bias, ref_bias, rtol=5e-2, atol=1e-2)
    torch.testing.assert_close(got_weight, ref_weight, rtol=5e-2, atol=1e-2)


def test_bf16_autocast_training_tracks_eager():
    """Trained trajectory under bf16 autocast, f32 master weights."""

    class TinyLM(nn.Module):
        def __init__(self, vocab=64, dim=32):
            super().__init__()
            self.vocab = vocab
            self.embed = nn.Embedding(vocab, dim)
            self.norm = nn.LayerNorm(dim)
            self.head = nn.Linear(dim, vocab, bias=False)

        def forward(self, idx, targets):
            logits = self.head(self.norm(self.embed(idx)))
            return F.cross_entropy(logits.float(), targets)

    def losses(backend: str):
        torch._dynamo.reset()
        torch.manual_seed(0)
        model = TinyLM()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2)

        def call(idx, targets):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                return model(idx, targets)

        fn = call if backend == "eager" else torch.compile(call, backend="alloy", dynamic=False)
        torch.manual_seed(1)
        idx = torch.randint(0, 64, (16,))
        targets = torch.randint(0, 64, (16,))
        out = []
        for _ in range(6):
            opt.zero_grad(set_to_none=False)
            loss = fn(idx, targets)
            loss.backward()
            opt.step()
            out.append(float(loss.detach()))
        return out

    ref, got = losses("eager"), losses("alloy")
    assert all(v == v for v in got), f"alloy produced NaN: {got}"
    for i, (a, r) in enumerate(zip(got, ref)):
        assert abs(a - r) <= 5e-2, f"step {i}: alloy {a:.5f} vs eager {r:.5f}\n{got}\n{ref}"
