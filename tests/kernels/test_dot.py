"""GPU correctness tests for GEMM kernels — dot, dot_transpose_rhs."""

import alloy as al
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# al.dot (A @ B)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M,N,K,BM,BN,BK", [
    (8, 8, 8, 8, 8, 8),
    (32, 32, 32, 16, 16, 8),
    (64, 64, 64, 32, 32, 8),
    (100, 50, 73, 16, 16, 8),      # rectangular, non-aligned
    (256, 256, 256, 32, 32, 16),    # large
])
def test_dot(M, N, K, BM, BN, BK):
    rng = np.random.default_rng(42)
    A = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)
    B = (rng.standard_normal((K, N)) * 0.1).astype(np.float32)
    result = al.dot(A, B, BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK)
    np.testing.assert_allclose(result, A @ B, rtol=1e-3, atol=1e-4)


def test_dot_identity():
    rng = np.random.default_rng(42)
    A = (rng.standard_normal((32, 32)) * 0.1).astype(np.float32)
    I = np.eye(32, dtype=np.float32)
    np.testing.assert_allclose(al.dot(A, I, BLOCK_M=16, BLOCK_N=16, BLOCK_K=8), A, rtol=1e-5)


def test_dot_tuned():
    rng = np.random.default_rng(42)
    A = (rng.standard_normal((128, 128)) * 0.1).astype(np.float32)
    B = (rng.standard_normal((128, 128)) * 0.1).astype(np.float32)
    np.testing.assert_allclose(al.dot(A, B), A @ B, rtol=1e-3, atol=1e-4)


# ---------------------------------------------------------------------------
# al.dot_transpose_rhs (A @ B.T)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M,N,K", [
    (8, 8, 8),
    (32, 32, 32),
    (64, 128, 64),
    (16, 2048, 2048),   # typical Llama shape
])
def test_dot_transpose_rhs(M, N, K):
    rng = np.random.default_rng(42)
    A = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)
    B_T = (rng.standard_normal((N, K)) * 0.1).astype(np.float32)
    result = al.dot_transpose_rhs(A, B_T, BLOCK_M=min(M, 32), BLOCK_N=min(N, 32), BLOCK_K=min(K, 16))
    np.testing.assert_allclose(result, A @ B_T.T, rtol=1e-3, atol=1e-4)


# ---------------------------------------------------------------------------
# f16 GEMM
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M,N,K", [
    (64, 64, 64),
    (256, 256, 256),
])
def test_dot_f16(M, N, K):
    rng = np.random.default_rng(42)
    A = (rng.standard_normal((M, K)) * 0.1).astype(np.float16)
    B = (rng.standard_normal((K, N)) * 0.1).astype(np.float16)
    result = al.dot(A, B, BLOCK_M=32, BLOCK_N=32, BLOCK_K=32)
    expected = (A.astype(np.float32) @ B.astype(np.float32)).astype(np.float16)
    np.testing.assert_allclose(
        np.asarray(result).astype(np.float32),
        expected.astype(np.float32),
        rtol=0.05, atol=1e-2,
    )
