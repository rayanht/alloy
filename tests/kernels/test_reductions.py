"""GPU correctness tests for reductions, softmax, layernorm, rms_norm, cross_entropy, rope."""

import alloy as al
import numpy as np
import pytest
from tests.helpers import ref_softmax, ref_layernorm, ref_rms_norm, ref_cross_entropy, ref_rope
from alloy._compiler.dtypes import float32
from alloy._dispatch.buf_utils import _alloc_aligned
from alloy._runtime.convert import to_alloy_buffer


# ---------------------------------------------------------------------------
# Flat reductions
# ---------------------------------------------------------------------------

_REDUCE_OPS = {
    "reduce_sum": al.reduce_sum,
    "reduce_max": al.reduce_max,
    "reduce_min": al.reduce_min,
    "mean": al.mean,
}


@pytest.mark.parametrize("op,N,np_fn,rtol", [
    ("reduce_sum", 256,       lambda x: x.sum(),  1e-4),
    ("reduce_sum", 100_000,   lambda x: x.sum(),  1e-3),
    ("reduce_sum", 1_000_000, lambda x: x.sum(),  1e-3),
    ("reduce_max", 4096,      lambda x: x.max(),  1e-5),
    ("reduce_max", 500_000,   lambda x: x.max(),  1e-5),
    ("reduce_min", 4096,      lambda x: x.min(),  1e-5),
    ("reduce_min", 500_000,   lambda x: x.min(),  1e-5),
    ("mean",       8192,      lambda x: x.mean(), 1e-4),
    ("mean",       1_000_000, lambda x: x.mean(), 1e-3),
], ids=lambda x: f"{x}" if isinstance(x, int) else "")
def test_flat_reduce(op, N, np_fn, rtol):
    x = np.random.default_rng(42).standard_normal(N).astype(np.float32)
    result = _REDUCE_OPS[op](x)
    np.testing.assert_allclose(np.asarray(result).flat[0], np_fn(x), rtol=rtol)


# ---------------------------------------------------------------------------
# Row reductions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("op,np_fn", [
    ("reduce_sum", lambda x: x.sum(axis=1)),
    ("mean",       lambda x: x.mean(axis=1)),
    ("reduce_max", lambda x: x.max(axis=1)),
])
def test_row_reduce(op, np_fn):
    M, N = 8, 64
    x = np.random.default_rng(42).standard_normal((M, N)).astype(np.float32)
    np.testing.assert_allclose(_REDUCE_OPS[op](x, axis=1), np_fn(x), rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Softmax
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M,N", [(4, 64), (32, 512), (8, 4096)])
def test_softmax(M, N):
    x = np.random.default_rng(42).standard_normal((M, N)).astype(np.float32)
    np.testing.assert_allclose(al.softmax(x), ref_softmax(x), rtol=1e-5, atol=1e-5)


def test_softmax_rows_sum_to_one():
    M, N = 16, 256
    x = np.random.default_rng(42).standard_normal((M, N)).astype(np.float32)
    np.testing.assert_allclose(np.asarray(al.softmax(x)).sum(axis=1), np.ones(M), rtol=1e-5)


# ---------------------------------------------------------------------------
# LayerNorm
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M,N,use_affine", [
    (4, 64, False),
    (8, 128, True),
    (32, 1024, False),
])
def test_layernorm(M, N, use_affine):
    rng = np.random.default_rng(42)
    x = rng.standard_normal((M, N)).astype(np.float32)
    gamma = rng.standard_normal(N).astype(np.float32) if use_affine else np.ones(N, dtype=np.float32)
    beta = rng.standard_normal(N).astype(np.float32) if use_affine else np.zeros(N, dtype=np.float32)
    np.testing.assert_allclose(al.layernorm(x, gamma, beta),
                               ref_layernorm(x, gamma, beta), rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M,N", [(4, 64), (16, 256), (32, 1024)])
def test_rms_norm(M, N):
    rng = np.random.default_rng(42)
    x = rng.standard_normal((M, N)).astype(np.float32)
    w = np.ones(N, dtype=np.float32)
    out_buf = _alloc_aligned((M, N), float32)
    rrms_buf = _alloc_aligned((M,), float32)
    al.rms_norm(to_alloy_buffer(x), to_alloy_buffer(w), out_buf, rrms_buf)
    out_buf.sync()
    np.testing.assert_allclose(np.asarray(out_buf.numpy), ref_rms_norm(x, w), rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Cross-entropy
# ---------------------------------------------------------------------------

def test_cross_entropy():
    M, N = 8, 64
    rng = np.random.default_rng(42)
    logits = rng.standard_normal((M, N)).astype(np.float32)
    labels = rng.integers(0, N, size=M).astype(np.int32)
    np.testing.assert_allclose(al.cross_entropy(logits, labels),
                               ref_cross_entropy(logits, labels), rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M,D", [(4, 16), (8, 64)])
def test_rope(M, D):
    x = np.random.default_rng(42).standard_normal((M, D)).astype(np.float32)
    np.testing.assert_allclose(al.rope(x), ref_rope(x), rtol=1e-4, atol=1e-4)
