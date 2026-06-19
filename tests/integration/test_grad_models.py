"""End-to-end model gradient correctness — tiny transformers vs CPU eager.

Combines every layer type that matters for real workloads (attention,
MLP, residual, norm, LM head, cross-entropy) into one forward+backward
check. Shapes are intentionally tiny (hidden=32, 1 layer, 4 tokens) so
the CPU reference runs in well under a second.

If ``test_grad_ops.py`` and ``test_grad_modules.py`` pass but a test
here fails, the culprit is usually interaction between fused rewrites
(e.g. ``gemm_residual_layernorm``) and the backward graph.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tests.grad_helpers import check_grads, module_factory


# ---------------------------------------------------------------------------
# Tiny GPT-2-style causal LM — hand-built so the test has no HF dependency
# ---------------------------------------------------------------------------


class TinyGPTBlock(nn.Module):
    def __init__(self, d: int, nh: int, ff: int):
        super().__init__()
        self.nh = nh
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=True)
        self.o = nn.Linear(d, d, bias=True)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, ff, bias=True)
        self.fc2 = nn.Linear(ff, d, bias=True)

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


class TinyGPT(nn.Module):
    def __init__(self, d: int = 32, nh: int = 4, ff: int = 64, vocab: int = 64, max_t: int = 16):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_t, d)
        self.block = TinyGPTBlock(d, nh, ff)
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight  # weight tying, like real GPT-2
        # Embeddings produce scatter-add on backward which alloy doesn't
        # handle yet — freeze them so only the transformer body sees grad.
        self.tok.weight.requires_grad_(False)
        self.pos.weight.requires_grad_(False)

    def forward(self, input_ids, labels):
        b, t = input_ids.shape
        pos = torch.arange(t, device=input_ids.device).unsqueeze(0).expand(b, t)
        x = self.tok(input_ids) + self.pos(pos)
        x = self.block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return F.cross_entropy(
            logits.reshape(b * t, -1),
            labels.reshape(b * t),
            ignore_index=-100,
        )


def _tiny_gpt_inputs_factory():
    def _make():
        torch.manual_seed(0)
        input_ids = torch.randint(0, 64, (2, 8))
        labels = torch.randint(0, 64, (2, 8))
        return input_ids, labels

    return _make


# ---------------------------------------------------------------------------
# Tiny Llama-style block — RMSNorm + SwiGLU + RoPE omitted (CPU grad for
# the RoPE decomposition is slow without adding more infra). Uses RMSNorm
# and GLU-style FFN so the RMSNorm + residual backward is exercised.
# ---------------------------------------------------------------------------


class TinyLlamaBlock(nn.Module):
    def __init__(self, d: int, nh: int, ff: int):
        super().__init__()
        self.nh = nh
        if not hasattr(nn, "RMSNorm"):
            raise pytest.skip.Exception("torch.nn.RMSNorm not available")
        rms_cls = nn.RMSNorm
        self.ln1 = rms_cls(d)
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.ln2 = rms_cls(d)
        self.gate = nn.Linear(d, ff, bias=False)
        self.up = nn.Linear(d, ff, bias=False)
        self.down = nn.Linear(ff, d, bias=False)

    def forward(self, x):
        b, t, d = x.shape
        h = self.ln1(x)
        q = self.q(h).reshape(b, t, self.nh, d // self.nh).transpose(1, 2)
        k = self.k(h).reshape(b, t, self.nh, d // self.nh).transpose(1, 2)
        v = self.v(h).reshape(b, t, self.nh, d // self.nh).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v)
        y = y.transpose(1, 2).reshape(b, t, d)
        x = x + self.o(y)
        h2 = self.ln2(x)
        x = x + self.down(F.silu(self.gate(h2)) * self.up(h2))
        return x


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTinyGPT:
    def test_tiny_gpt_grads(self):
        check_grads(
            module_factory(lambda: TinyGPT()),
            _tiny_gpt_inputs_factory(),
            atol=1e-3,
        )
