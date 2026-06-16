"""GPU correctness tests for f16, bf16, and cast operations."""

import alloy as al
import numpy as np
import pytest

try:
    import ml_dtypes
    HAS_ML_DTYPES = True
except ImportError:
    HAS_ML_DTYPES = False


# ---------------------------------------------------------------------------
# f16
# ---------------------------------------------------------------------------

def test_f16_vector_add():
    @al.kernel
    def k(x, y, out: al.output, N: al.constexpr):
        pid = al.program_id(0)
        offs = pid * 1024 + al.arange(0, 1024)
        mask = offs < N
        al.store(out + offs, al.load(x + offs, mask=mask) + al.load(y + offs, mask=mask), mask=mask)

    N = 4096
    rng = np.random.default_rng(42)
    x = rng.standard_normal(N).astype(np.float16)
    y = rng.standard_normal(N).astype(np.float16)
    out = np.zeros(N, dtype=np.float16)
    expected = (x.astype(np.float32) + y.astype(np.float32)).astype(np.float16)
    np.testing.assert_allclose(k[(N + 1023) // 1024](x, y, out, N=N), expected, atol=0.01)


def test_f16_exp():
    @al.kernel
    def k(x, out: al.output, N: al.constexpr):
        pid = al.program_id(0)
        offs = pid * 1024 + al.arange(0, 1024)
        mask = offs < N
        al.store(out + offs, al.exp(al.load(x + offs, mask=mask)), mask=mask)

    N = 4096
    x = (np.random.default_rng(42).standard_normal(N) * 0.5).astype(np.float16)
    out = np.zeros(N, dtype=np.float16)
    expected = np.exp(x.astype(np.float32)).astype(np.float16)
    np.testing.assert_allclose(k[(N + 1023) // 1024](x, out, N=N), expected, atol=0.05, rtol=0.05)


# ---------------------------------------------------------------------------
# bf16
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_ML_DTYPES, reason="ml_dtypes not installed")
def test_bf16_reduce_sum():
    x = np.random.default_rng(42).standard_normal(256).astype(ml_dtypes.bfloat16)
    result = al.reduce_sum(x)
    expected = float(x.astype(np.float32).sum())
    np.testing.assert_allclose(float(np.asarray(result).flat[0]), expected, rtol=0.05)


@pytest.mark.skipif(not HAS_ML_DTYPES, reason="ml_dtypes not installed")
def test_bf16_softmax():
    x = np.random.default_rng(42).standard_normal((4, 64)).astype(ml_dtypes.bfloat16)
    np.testing.assert_allclose(np.asarray(al.softmax(x)).sum(axis=1), np.ones(4), rtol=1e-4)


# ---------------------------------------------------------------------------
# al.cast()
# ---------------------------------------------------------------------------

def test_cast_f32_to_f16():
    @al.kernel
    def k(x, out: al.output, N: al.constexpr):
        pid = al.program_id(0)
        offs = pid * 1024 + al.arange(0, 1024)
        mask = offs < N
        al.store(out + offs, al.cast(al.load(x + offs, mask=mask), "f16"), mask=mask)

    N = 1024
    x = np.random.default_rng(42).standard_normal(N).astype(np.float32)
    out = np.zeros(N, dtype=np.float16)
    np.testing.assert_allclose(k[1](x, out, N=N), x.astype(np.float16), atol=0.01)
