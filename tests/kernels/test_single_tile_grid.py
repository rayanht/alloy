"""Auto-grid correctness when a block tile exactly covers a dimension.

A blocked pid's grid stride can coincide with an unrelated dim's stride —
BLOCK_N == N makes `B + rk * N + rn` look like an index into B's dim 0 — which
silently sized the N grid from K.
"""

import numpy as np
import pytest
from alloy.std.gemm import dot, dot_transpose_lhs


@pytest.mark.parametrize("M,K,N,BN", [
    (32, 32, 8, 8),
    (128, 256, 64, 64),
    (128, 256, 128, 128),
    (256, 256, 64, 64),
])
def test_dot_block_n_covers_n(M, K, N, BN):
    rng = np.random.default_rng(0)
    A = rng.standard_normal((M, K), dtype=np.float32)
    B = rng.standard_normal((K, N), dtype=np.float32)
    C = np.zeros((M, N), dtype=np.float32)
    got = np.array(dot(A, B, C, BLOCK_M=64, BLOCK_N=BN, BLOCK_K=16)).reshape(M, N)
    np.testing.assert_allclose(got, A @ B, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("K,M,N,BN", [(256, 128, 64, 64), (1024, 64, 384, 64)])
def test_dot_transpose_lhs_block_n_covers_n(K, M, N, BN):
    rng = np.random.default_rng(0)
    A_T = rng.standard_normal((K, M), dtype=np.float32)
    B = rng.standard_normal((K, N), dtype=np.float32)
    C = np.zeros((M, N), dtype=np.float32)
    got = np.array(dot_transpose_lhs(A_T, B, C, BLOCK_M=64, BLOCK_N=BN, BLOCK_K=16)).reshape(M, N)
    np.testing.assert_allclose(got, A_T.T @ B, rtol=1e-3, atol=1e-2)
