"""Tied embedding/LM-head gradients, and compile isolation across models.

The tied case sums two contributions into one parameter — an atomic scatter-add
over the indexed rows plus the head's weight grad — so the scatter must not
absorb the add as a store epilogue (rows no token touched would keep neither).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import alloy_torch  # noqa: F401 — registers the "alloy" backend
from alloy_torch.training import set_training_mode

set_training_mode(True)

VOCAB, DIM, TOKENS = 32, 8, 16


class TiedLM(nn.Module):
    def __init__(self, tie: bool = True):
        super().__init__()
        self.wte = nn.Embedding(VOCAB, DIM)
        self.head = nn.Linear(DIM, VOCAB, bias=False)
        self.head.weight = self.wte.weight if tie else nn.Parameter(self.wte.weight.detach().clone())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(self.head(self.wte(idx)), targets)


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)
    return torch.randint(0, VOCAB, (TOKENS,)), torch.randint(0, VOCAB, (TOKENS,))


def _run(backend: str, tie: bool) -> dict[str, torch.Tensor]:
    torch._dynamo.reset()
    torch.manual_seed(0)
    model = TiedLM(tie)
    fn = model if backend == "eager" else torch.compile(model, backend="alloy", dynamic=False)
    fn(*_inputs()).backward()
    return {name: p.grad.clone() for name, p in model.named_parameters()}


def test_tied_embedding_grad():
    ref, got = _run("eager", True), _run("alloy", True)
    torch.testing.assert_close(got["wte.weight"], ref["wte.weight"], rtol=1e-4, atol=1e-5)


def test_index_put_accumulate_then_add():
    """Rows outside the index keep the added tensor."""
    torch.manual_seed(0)
    index = torch.randint(0, VOCAB, (TOKENS,))
    values = torch.randn(TOKENS, DIM)
    other = torch.randn(VOCAB, DIM)

    def fn(index, values, other):
        return other + torch.zeros(VOCAB, DIM).index_put([index], values, accumulate=True)

    torch._dynamo.reset()
    got = torch.compile(fn, backend="alloy", dynamic=False)(index, values, other)
    torch.testing.assert_close(got, fn(index, values, other), rtol=1e-5, atol=1e-6)


def test_second_compile_of_same_model():
    """Fusion retypes the fused input param — on a shared param object that
    leaks into the next compile through the trace cache."""
    first = _run("alloy", False)
    second = _run("alloy", False)
    for name, grad in first.items():
        torch.testing.assert_close(second[name], grad, rtol=1e-4, atol=1e-5)
