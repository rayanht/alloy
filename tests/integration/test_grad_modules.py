"""Module-level gradient correctness — small compositions of ops.

These sit between op-level tests (``test_grad_ops.py``) and end-to-end
model tests (``test_grad_models.py``). They exercise backward paths that
op-level tests don't cover because each op runs in isolation:

  * parameter grads for an ``nn.Module`` with multiple layers,
  * residual connections (grads flow through the add + through the
    subgraph and must sum correctly),
  * prenorm vs postnorm layouts,
  * GELU / SiLU fused into the MLP block.

Shapes are tiny so the whole file stays under ~5s wall time.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tests.grad_helpers import check_grads, inputs_factory, module_factory

_XFAIL_LN_RESIDUAL_BWD = pytest.mark.xfail(
    reason=(
        "LayerNorm weight grad is wrong when LN consumes a residual add; "
        "likely gemm_residual_layernorm fused rewrite dropping a backward path"
    ),
    strict=False,
)
_XFAIL_SDPA_MHA = pytest.mark.xfail(
    reason=(
        "Multi-head attention forward+backward via QKV projection + SDPA "
        "diverges — same family as the op-level SDPA-backward xfail"
    ),
    strict=False,
)


# ---------------------------------------------------------------------------
# Linear stacks — matmul backward with param grads threaded through bias
# ---------------------------------------------------------------------------


class TestLinearStacks:
    def test_single_linear(self):
        check_grads(
            module_factory(lambda: nn.Linear(64, 32)),
            inputs_factory((8, 64)),
        )

    def test_linear_no_bias(self):
        check_grads(
            module_factory(lambda: nn.Linear(32, 16, bias=False)),
            inputs_factory((4, 32)),
        )

    def test_two_linears_with_relu(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(32, 64)
                self.fc2 = nn.Linear(64, 32)

            def forward(self, x):
                return self.fc2(F.relu(self.fc1(x)))

        check_grads(module_factory(lambda: M()), inputs_factory((4, 32)))

    def test_linear_gelu_linear(self):
        """GPT-2 / BERT MLP block."""

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(32, 128)
                self.fc2 = nn.Linear(128, 32)

            def forward(self, x):
                return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))

        check_grads(module_factory(lambda: M()), inputs_factory((4, 32)), atol=1e-4)


# ---------------------------------------------------------------------------
# Residual + norm — the pattern prenorm-style transformers use everywhere
# ---------------------------------------------------------------------------


class TestResidual:
    def test_residual_add(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(32, 32)

            def forward(self, x):
                return x + self.fc(x)

        check_grads(module_factory(lambda: M()), inputs_factory((4, 32)))

    def test_residual_through_layernorm(self):
        """Postnorm block: x = LN(x + sublayer(x))."""

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(32, 32)
                self.ln = nn.LayerNorm(32)

            def forward(self, x):
                return self.ln(x + self.fc(x))

        check_grads(module_factory(lambda: M()), inputs_factory((4, 32)), atol=1e-4)

    def test_prenorm_residual(self):
        """Prenorm block: x = x + sublayer(LN(x))."""

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(32, 32)
                self.ln = nn.LayerNorm(32)

            def forward(self, x):
                return x + self.fc(self.ln(x))

        check_grads(module_factory(lambda: M()), inputs_factory((4, 32)), atol=1e-4)


# ---------------------------------------------------------------------------
# Multi-head attention — the fused SDPA path
# ---------------------------------------------------------------------------


class TestAttention:
    def test_mha_self_attn(self):
        """Single-block self-attention with QKV projections and out projection."""

        class Attn(nn.Module):
            def __init__(self, d=32, nh=4):
                super().__init__()
                self.nh = nh
                self.qkv = nn.Linear(d, 3 * d, bias=False)
                self.out = nn.Linear(d, d, bias=False)

            def forward(self, x):
                b, t, d = x.shape
                qkv = self.qkv(x).reshape(b, t, 3, self.nh, d // self.nh)
                q, k, v = qkv.permute(2, 0, 3, 1, 4)
                y = F.scaled_dot_product_attention(q, k, v)
                y = y.transpose(1, 2).reshape(b, t, d)
                return self.out(y)

        check_grads(
            module_factory(lambda: Attn()),
            inputs_factory((1, 16, 32)),
            atol=1e-3,
        )


# ---------------------------------------------------------------------------
# Mini transformer block — attention + MLP + prenorm + residual
# ---------------------------------------------------------------------------


class TestTransformerBlock:
    def test_gpt2_style_block(self):
        class Block(nn.Module):
            def __init__(self, d=32, nh=4, ff=64):
                super().__init__()
                self.ln1 = nn.LayerNorm(d)
                self.qkv = nn.Linear(d, 3 * d, bias=True)
                self.o = nn.Linear(d, d, bias=True)
                self.ln2 = nn.LayerNorm(d)
                self.fc1 = nn.Linear(d, ff, bias=True)
                self.fc2 = nn.Linear(ff, d, bias=True)
                self.nh = nh

            def forward(self, x):
                b, t, d = x.shape
                h = self.ln1(x)
                qkv = self.qkv(h).reshape(b, t, 3, self.nh, d // self.nh)
                q, k, v = qkv.permute(2, 0, 3, 1, 4)
                y = F.scaled_dot_product_attention(q, k, v)
                y = y.transpose(1, 2).reshape(b, t, d)
                x = x + self.o(y)
                x = x + self.fc2(F.gelu(self.fc1(self.ln2(x)), approximate="tanh"))
                return x

        check_grads(
            module_factory(lambda: Block()),
            inputs_factory((1, 16, 32)),
            atol=1e-3,
        )


# ---------------------------------------------------------------------------
# LM head + cross-entropy — the tail of a causal LM, including the sliced-
# label shift pattern that broke Llama training.
# ---------------------------------------------------------------------------


class TestLMHead:
    def test_lm_head_ce(self):
        """Tied LM head (linear) + cross-entropy, no label shift."""

        class Head(nn.Module):
            def __init__(self, d=32, v=64):
                super().__init__()
                self.head = nn.Linear(d, v, bias=False)

            def forward(self, x, labels):
                logits = self.head(x)
                b, t, vv = logits.shape
                return F.cross_entropy(
                    logits.reshape(b * t, vv),
                    labels.reshape(b * t),
                    ignore_index=-100,
                )

        def make_inputs():
            torch.manual_seed(0)
            x = torch.randn(2, 8, 32, requires_grad=True)
            labels = torch.randint(0, 64, (2, 8))
            return x, labels

        check_grads(module_factory(lambda: Head()), make_inputs, atol=1e-4)

    def test_lm_head_ce_shifted(self):
        """HF ForCausalLMLoss tail: shift labels, then CE with ignore_index."""

        class Head(nn.Module):
            def __init__(self, d=32, v=64):
                super().__init__()
                self.head = nn.Linear(d, v, bias=False)

            def forward(self, x, labels):
                logits = self.head(x)
                shift = F.pad(labels, (0, 1), value=-100)[:, 1:]
                b, t, vv = logits.shape
                return F.cross_entropy(
                    logits.reshape(b * t, vv),
                    shift.reshape(b * t),
                    ignore_index=-100,
                )

        def make_inputs():
            torch.manual_seed(0)
            x = torch.randn(2, 8, 32, requires_grad=True)
            labels = torch.randint(0, 64, (2, 8))
            return x, labels

        check_grads(module_factory(lambda: Head()), make_inputs, atol=1e-4)
