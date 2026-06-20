"""Op-level gradient correctness — alloy vs CPU eager on individual ATen ops.

Each test wraps one or two ops in a tiny ``nn.Module``/function, runs
backward, and asserts parameter+input gradients match the CPU-eager
reference within tight tolerance. Broad coverage of the backward paths
the torch.compile frontend decomposes into.

Failures here point at a broken backward kernel, a fusion bug that
corrupts an epilogue, or a view-handling bug.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tests.grad_helpers import (
    check_grads,
    fn_factory,
    inputs_factory,
    module_factory,
)


# ---------------------------------------------------------------------------
# Matmul family
# ---------------------------------------------------------------------------


class TestMatmul:
    def test_mm_small(self):
        check_grads(
            fn_factory(lambda a, b: a @ b),
            inputs_factory((8, 16), (16, 32)),
        )

    def test_mm_medium(self):
        check_grads(
            fn_factory(lambda a, b: a @ b),
            inputs_factory((64, 128), (128, 96)),
            atol=1e-4,
        )

    def test_mm_transpose_rhs(self):
        # b.T exercises the transpose-RHS GEMM path
        check_grads(
            fn_factory(lambda a, b: a @ b.T),
            inputs_factory((8, 16), (32, 16)),
        )

    def test_addmm(self):
        check_grads(
            fn_factory(lambda x, w, b: F.linear(x, w, b)),
            inputs_factory((8, 16), (32, 16), (32,)),
        )

    def test_bmm(self):
        check_grads(
            fn_factory(lambda a, b: torch.bmm(a, b)),
            inputs_factory((4, 8, 16), (4, 16, 32)),
        )


# ---------------------------------------------------------------------------
# LayerNorm / RMSNorm
# ---------------------------------------------------------------------------


class TestLayerNorm:
    def test_layernorm(self):
        check_grads(
            module_factory(lambda: nn.LayerNorm(64)),
            inputs_factory((4, 64)),
            atol=1e-5,
        )

    def test_layernorm_3d(self):
        check_grads(
            module_factory(lambda: nn.LayerNorm(128)),
            inputs_factory((2, 16, 128)),
            atol=1e-4,
        )

    def test_layernorm_no_affine(self):
        check_grads(
            module_factory(lambda: nn.LayerNorm(32, elementwise_affine=False)),
            inputs_factory((8, 32)),
        )

    def test_rms_norm(self):
        if not hasattr(nn, "RMSNorm"):
            pytest.skip("torch.nn.RMSNorm not available")
        rms = nn.RMSNorm
        check_grads(
            module_factory(lambda: rms(64)),
            inputs_factory((4, 64)),
        )


# ---------------------------------------------------------------------------
# Softmax / log_softmax
# ---------------------------------------------------------------------------


class TestSoftmax:
    def test_softmax_last_dim(self):
        check_grads(
            fn_factory(lambda x: F.softmax(x, dim=-1)),
            inputs_factory((8, 64)),
        )

    def test_softmax_inner_dim(self):
        check_grads(
            fn_factory(lambda x: F.softmax(x, dim=1)),
            inputs_factory((4, 16, 32)),
        )

    def test_log_softmax(self):
        check_grads(
            fn_factory(lambda x: F.log_softmax(x, dim=-1)),
            inputs_factory((8, 64)),
        )


# ---------------------------------------------------------------------------
# Cross-entropy — including the sliced-label shift pattern
# ---------------------------------------------------------------------------


class TestCrossEntropy:
    def test_ce_basic(self):
        def make_fn():
            def fn(logits, labels):
                return F.cross_entropy(logits, labels)

            return fn

        def make_inputs():
            torch.manual_seed(0)
            logits = torch.randn(16, 128, requires_grad=True)
            labels = torch.randint(0, 128, (16,))
            return logits, labels

        check_grads(make_fn, make_inputs)

    def test_ce_ignore_index(self):
        def make_fn():
            def fn(logits, labels):
                return F.cross_entropy(logits, labels, ignore_index=-100)

            return fn

        def make_inputs():
            torch.manual_seed(0)
            logits = torch.randn(16, 128, requires_grad=True)
            labels = torch.randint(0, 128, (16,))
            labels[-1] = -100
            return logits, labels

        check_grads(make_fn, make_inputs)

    def test_ce_sliced_labels(self):
        """HF ForCausalLMLoss pattern: F.pad(labels, (0,1), -100)[1:]."""

        def make_fn():
            def fn(logits, labels):
                shift = F.pad(labels, (0, 1), value=-100)[1:]
                return F.cross_entropy(logits, shift, ignore_index=-100)

            return fn

        def make_inputs():
            torch.manual_seed(0)
            logits = torch.randn(16, 128, requires_grad=True)
            labels = torch.randint(0, 128, (16,))
            return logits, labels

        check_grads(make_fn, make_inputs)

    def test_ce_3d_logits(self):
        def make_fn():
            def fn(logits, labels):
                b, t, v = logits.shape
                return F.cross_entropy(
                    logits.reshape(b * t, v),
                    labels.reshape(b * t),
                    ignore_index=-100,
                )

            return fn

        def make_inputs():
            torch.manual_seed(0)
            logits = torch.randn(2, 8, 64, requires_grad=True)
            labels = torch.randint(0, 64, (2, 8))
            return logits, labels

        check_grads(make_fn, make_inputs)


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------


class TestActivations:
    def test_gelu_tanh(self):
        check_grads(
            fn_factory(lambda x: F.gelu(x, approximate="tanh")),
            inputs_factory((8, 64)),
        )

    def test_gelu_none(self):
        check_grads(
            fn_factory(lambda x: F.gelu(x)),
            inputs_factory((8, 64)),
        )

    def test_silu(self):
        check_grads(
            fn_factory(lambda x: F.silu(x)),
            inputs_factory((8, 64)),
        )

    def test_relu(self):
        check_grads(
            fn_factory(lambda x: F.relu(x)),
            inputs_factory((8, 64)),
        )

    def test_sigmoid(self):
        check_grads(
            fn_factory(lambda x: torch.sigmoid(x)),
            inputs_factory((8, 64)),
        )

    def test_tanh(self):
        check_grads(
            fn_factory(lambda x: torch.tanh(x)),
            inputs_factory((8, 64)),
        )


# ---------------------------------------------------------------------------
# SDPA — masked and unmasked
# ---------------------------------------------------------------------------


class TestSDPA:
    def test_sdpa_unmasked(self):
        check_grads(
            fn_factory(lambda q, k, v: F.scaled_dot_product_attention(q, k, v)),
            inputs_factory((1, 4, 16, 32), (1, 4, 16, 32), (1, 4, 16, 32)),
            atol=1e-4,
        )

    def test_sdpa_causal(self):
        check_grads(
            fn_factory(lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True)),
            inputs_factory((1, 4, 16, 32), (1, 4, 16, 32), (1, 4, 16, 32)),
            atol=1e-4,
        )


# ---------------------------------------------------------------------------
# Indexing — gather, index_select, slice, view ops
# ---------------------------------------------------------------------------


class TestEmbedding:
    def test_embedding_backward(self):
        # embedding backward decomposes to index_put(accumulate=True): the
        # weight grad scatter-adds grad_output rows at the token indices.
        def make_inputs():
            torch.manual_seed(0)
            return (torch.randint(0, 50, (4, 8)),)

        check_grads(
            module_factory(lambda: nn.Embedding(50, 16)),
            make_inputs,
            check_input_grads=False,
            atol=1e-4,
        )


class TestIndexing:
    def test_gather_dim_last(self):
        def make_fn():
            def fn(x, idx):
                return x.gather(1, idx)

            return fn

        def make_inputs():
            torch.manual_seed(0)
            x = torch.randn(8, 32, requires_grad=True)
            idx = torch.randint(0, 32, (8, 4))
            return x, idx

        check_grads(make_fn, make_inputs)

    def test_slice_last_dim(self):
        check_grads(
            fn_factory(lambda x: x[..., :32]),
            inputs_factory((4, 64)),
        )

    def test_slice_middle(self):
        check_grads(
            fn_factory(lambda x: x[:, 4:12, :]),
            inputs_factory((2, 16, 32)),
        )


# ---------------------------------------------------------------------------
# View-shape ops
# ---------------------------------------------------------------------------


class TestViews:
    def test_reshape(self):
        check_grads(
            fn_factory(lambda x: x.reshape(8, 32)),
            inputs_factory((4, 2, 32)),
        )

    def test_transpose(self):
        check_grads(
            fn_factory(lambda x: x.transpose(-1, -2)),
            inputs_factory((4, 8, 16)),
        )

    def test_permute(self):
        check_grads(
            fn_factory(lambda x: x.permute(0, 2, 1, 3)),
            inputs_factory((2, 4, 8, 16)),
        )

    def test_cat_last_dim(self):
        check_grads(
            fn_factory(lambda a, b: torch.cat([a, b], dim=-1)),
            inputs_factory((4, 16), (4, 32)),
        )

    def test_unsqueeze(self):
        check_grads(
            fn_factory(lambda x: x.unsqueeze(1)),
            inputs_factory((4, 16)),
        )


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------


class TestReductions:
    def test_sum_last_dim(self):
        check_grads(
            fn_factory(lambda x: x.sum(dim=-1)),
            inputs_factory((4, 32)),
        )

    def test_sum_all(self):
        check_grads(
            fn_factory(lambda x: x.sum()),
            inputs_factory((4, 32)),
        )

    def test_mean_last_dim(self):
        check_grads(
            fn_factory(lambda x: x.mean(dim=-1)),
            inputs_factory((4, 32)),
        )

    def test_var_last_dim(self):
        check_grads(
            fn_factory(lambda x: x.var(dim=-1)),
            inputs_factory((4, 32)),
            atol=1e-4,
        )


# ---------------------------------------------------------------------------
# Binary elementwise with broadcast
# ---------------------------------------------------------------------------


class TestBinaryBroadcast:
    def test_add_broadcast_rowwise(self):
        check_grads(
            fn_factory(lambda a, b: a + b),
            inputs_factory((8, 32), (32,)),
        )

    def test_mul_broadcast_colwise(self):
        check_grads(
            fn_factory(lambda a, b: a * b),
            inputs_factory((8, 32), (8, 1)),
        )

    def test_sub_same_shape(self):
        check_grads(
            fn_factory(lambda a, b: a - b),
            inputs_factory((4, 32), (4, 32)),
        )

    def test_div_by_scalar(self):
        check_grads(
            fn_factory(lambda x: x / 3.0),
            inputs_factory((4, 32)),
        )
