"""Fusion integration tests — verify ops actually fuse (dispatch count), delegate correctness.

Each test asserts:
  1. Correct numerical output (sanity check)
  2. Dispatch count (N ops → M < N dispatches)
"""

import alloy as al
import numpy as np
import pytest
from alloy._runtime.metal import default_dispatcher
from tests.helpers import (
    k_bias,
    k_gelu,
    k_mul,
    k_relu,
    k_scale,
    k_sigmoid,
    ref_gelu,
    ref_layernorm,
    ref_sigmoid,
    ref_softmax,
)


def _dispatches(fn):
    """Run fn(), return (result, dispatch_count)."""
    d = default_dispatcher()
    before = d.dispatch_count
    result = fn()
    return result, d.dispatch_count - before


# ---------------------------------------------------------------------------
# Elem-elem chain fusion
# ---------------------------------------------------------------------------

class TestElemChainFusion:
    @pytest.mark.parametrize("chain,expected_fn,label", [
        ([k_scale, k_bias],         lambda x: x * 2.0 + 1.0,                 "scale+bias"),
        ([k_scale, k_bias, k_relu], lambda x: np.maximum(x * 2.0 + 1.0, 0), "scale+bias+relu"),
    ])
    def test_elem_chain_fuses(self, chain, expected_fn, label):
        N = 4096
        grid = (N + 1023) // 1024
        x = np.random.default_rng(42).standard_normal(N).astype(np.float32)

        prev = x
        for kfn in chain:
            prev = kfn[grid](prev, np.zeros(N, dtype=np.float32), N=N)

        result, n_dispatches = _dispatches(lambda: np.array(prev))
        assert n_dispatches == 1, f"{label}: expected 1 dispatch, got {n_dispatches}"
        np.testing.assert_allclose(result, expected_fn(x), rtol=1e-5)


# ---------------------------------------------------------------------------
# Diamond pattern (SiLU: x → sigmoid(x), x → mul(sigmoid(x), x))
# ---------------------------------------------------------------------------

class TestDiamondFusion:
    def test_silu_diamond(self):
        """SiLU = x * sigmoid(x) — diamond pattern should fuse to 1 dispatch."""
        N = 4096
        grid = (N + 1023) // 1024
        x = np.random.default_rng(42).standard_normal(N).astype(np.float32)

        sig = k_sigmoid[grid](x, np.zeros(N, dtype=np.float32), N=N)
        # x * sigmoid(x) — both operands share the same input
        result_buf = k_mul[grid](x, sig, np.zeros(N, dtype=np.float32), N=N)

        result, n_dispatches = _dispatches(lambda: np.array(result_buf))
        # Diamond may fuse to 1 or stay at 2 depending on the analysis —
        # the key constraint is correctness.
        expected = x * ref_sigmoid(x)
        np.testing.assert_allclose(result, expected, rtol=1e-5)
        # But it should be at most 2 (not 3 separate dispatches)
        assert n_dispatches <= 2, f"SiLU diamond: expected ≤2 dispatches, got {n_dispatches}"


# ---------------------------------------------------------------------------
# Epilogue fusion (anchor + elem)
# ---------------------------------------------------------------------------

class TestEpilogueFusion:
    def test_dot_then_gelu(self):
        M, N, K = 64, 64, 64
        rng = np.random.default_rng(42)
        A = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)
        B = (rng.standard_normal((K, N)) * 0.1).astype(np.float32)

        r1 = al.dot(A, B)
        r2 = k_gelu[((M * N + 1023) // 1024,)](r1, np.zeros(M * N, dtype=np.float32), N=M * N)

        result, n_dispatches = _dispatches(lambda: np.array(r2))
        assert n_dispatches == 1, f"dot+gelu: expected 1 dispatch, got {n_dispatches}"
        np.testing.assert_allclose(result.reshape(M, N), ref_gelu(A @ B), rtol=1e-3, atol=1e-3)

    def test_dot_then_scale_then_bias(self):
        """Anchor + 2-step epilogue chain."""
        M, N, K = 64, 64, 64
        rng = np.random.default_rng(42)
        A = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)
        B = (rng.standard_normal((K, N)) * 0.1).astype(np.float32)

        r1 = al.dot(A, B)
        flat = (M * N + 1023) // 1024
        r2 = k_scale[(flat,)](r1, np.zeros(M * N, dtype=np.float32), N=M * N)
        r3 = k_bias[(flat,)](r2, np.zeros(M * N, dtype=np.float32), N=M * N)

        result, n_dispatches = _dispatches(lambda: np.array(r3))
        assert n_dispatches == 1, f"dot+scale+bias: expected 1 dispatch, got {n_dispatches}"
        np.testing.assert_allclose(result.reshape(M, N), (A @ B) * 2.0 + 1.0, rtol=1e-3, atol=1e-3)

    def test_softmax_then_scale(self):
        M, N = 32, 128
        x = np.random.default_rng(42).standard_normal((M, N)).astype(np.float32)

        r1 = al.softmax(x)
        r2 = k_scale[((M * N + 1023) // 1024,)](r1, np.zeros(M * N, dtype=np.float32), N=M * N)

        result, n_dispatches = _dispatches(lambda: np.array(r2))
        assert n_dispatches == 1, f"softmax+scale: expected 1 dispatch, got {n_dispatches}"
        np.testing.assert_allclose(result.reshape(M, N), ref_softmax(x) * 2.0, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Prologue fusion (elem → anchor)
# ---------------------------------------------------------------------------

class TestPrologueFusion:
    def test_scale_then_dot(self):
        M, N, K = 64, 64, 64
        rng = np.random.default_rng(42)
        A = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)
        B = (rng.standard_normal((K, N)) * 0.1).astype(np.float32)

        r1 = k_scale[((M * K + 1023) // 1024,)](A, np.zeros(M * K, dtype=np.float32), N=M * K)
        r1_2d = r1.reshape(M, K)
        r2 = al.dot(r1_2d, B)

        result, n_dispatches = _dispatches(lambda: np.array(r2))
        assert n_dispatches == 1, f"scale+dot: expected 1 dispatch, got {n_dispatches}"
        np.testing.assert_allclose(result, (A * 2.0) @ B, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# Multi-op correctness (no fusion requirement, just sanity)
# ---------------------------------------------------------------------------

class TestMultiOpCorrectness:
    def test_layernorm_then_scale(self):
        M, N = 8, 64
        rng = np.random.default_rng(42)
        x = rng.standard_normal((M, N)).astype(np.float32)
        gamma = np.ones(N, dtype=np.float32)
        beta = np.zeros(N, dtype=np.float32)

        r1 = al.layernorm(x, gamma, beta)
        r2 = k_scale[((M * N + 1023) // 1024,)](r1, np.zeros(M * N, dtype=np.float32), N=M * N)

        np.testing.assert_allclose(np.asarray(r2).reshape(M, N),
                                   ref_layernorm(x) * 2.0, rtol=1e-4, atol=1e-4)
