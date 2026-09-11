"""Batched GEMM — batch as a grid axis, with and without a transposed rhs.

Shape-congruent independent matmuls cannot be merged by the same-LHS batching
rewrite, and routing them through a per-batch loop costs a dispatch each.
"""

import numpy as np
import pytest
from alloy.std.gemm import dot_batched


@pytest.mark.parametrize("batch,M,K,N", [
    (1, 64, 64, 192),   # BLOCK_M == M makes the m-tile pid's stride equal the batch's
    (2, 64, 64, 192),
    (6, 64, 64, 192),
    (1, 64, 64, 128),
    (2, 256, 64, 256),
    (6, 384, 384, 384),
    (3, 100, 73, 50),   # non-aligned
])
def test_dot_batched(batch, M, K, N):
    rng = np.random.default_rng(0)
    a = rng.standard_normal((batch, M, K), dtype=np.float32)
    b = rng.standard_normal((batch, K, N), dtype=np.float32)
    out = np.zeros((batch, M, N), dtype=np.float32)
    got = np.array(dot_batched(a, b, out)).reshape(batch, M, N)
    np.testing.assert_allclose(got, np.matmul(a, b), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch,M,K,N", [(2, 384, 1152, 384), (6, 384, 384, 384), (1, 64, 64, 192)])
def test_dot_batched_transposed_rhs(batch, M, K, N):
    rng = np.random.default_rng(0)
    a = rng.standard_normal((batch, M, K), dtype=np.float32)
    b = rng.standard_normal((batch, N, K), dtype=np.float32)
    out = np.zeros((batch, M, N), dtype=np.float32)
    got = np.array(dot_batched(a, b, out, _TRANS_RHS=1)).reshape(batch, M, N)
    np.testing.assert_allclose(got, np.matmul(a, b.transpose(0, 2, 1)), rtol=1e-3, atol=1e-3)
