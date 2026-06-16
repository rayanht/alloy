"""GPU integration tests for fused extra buffer indexing.

Each test triggers a specific IndexTransform variant through the full
fusion → codegen → dispatch path. Correctness + dispatch count.

Variants:
  Identity:         same-shape contiguous extra (e.g., dot + add(same_shape))
  RowBroadcast:     bias (N,) fused into (M, N) GEMM epilogue (dot + add_bias)
  ColumnBroadcast:  per-row scale (M,) fused into (M, N) epilogue
  ScalarBroadcast:  stride-0 scalar fused into epilogue
  Broadcast:        smaller-than-output extra with modular wrap
"""

import alloy as al
import numpy as np
from alloy._runtime.metal import default_dispatcher
from tests.helpers import k_scale


def _dispatches(fn):
    d = default_dispatcher()
    before = d.dispatch_count
    result = fn()
    return result, d.dispatch_count - before


# ---------------------------------------------------------------------------
# Identity: same-shape contiguous extra
# ---------------------------------------------------------------------------

def test_epilogue_add_same_shape():
    """dot(A, B) + C where C is (M, N) — identity indexing for extra."""
    M, N, K = 64, 64, 64
    rng = np.random.default_rng(42)
    A = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)
    B = (rng.standard_normal((K, N)) * 0.1).astype(np.float32)
    C = (rng.standard_normal((M, N)) * 0.1).astype(np.float32)

    r1 = al.dot(A, B)
    r2 = al.std.add(r1, C, N=M * N)

    result, n_disp = _dispatches(lambda: np.array(r2))
    assert n_disp == 1
    np.testing.assert_allclose(result.reshape(M, N), A @ B + C, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# RowBroadcast: bias (N,) added to (M, N) output
# ---------------------------------------------------------------------------

def test_epilogue_add_row_bias():
    """dot(A, B) + bias where bias is (N,) — row broadcast."""
    M, N, K = 64, 128, 64
    rng = np.random.default_rng(42)
    A = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)
    B = (rng.standard_normal((K, N)) * 0.1).astype(np.float32)
    bias = (rng.standard_normal(N) * 0.1).astype(np.float32)

    r1 = al.dot(A, B)
    r2 = al.std.add(r1, bias, N=M * N)

    result, n_disp = _dispatches(lambda: np.array(r2))
    assert n_disp == 1
    np.testing.assert_allclose(result.reshape(M, N), A @ B + bias, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# Elem chain with two inputs (binary op fused)
# ---------------------------------------------------------------------------

def test_elem_chain_binary_add():
    """scale(x) → add(scaled, y) where y is a separate buffer."""
    N = 4096
    grid = (N + 1023) // 1024
    rng = np.random.default_rng(42)
    x = rng.standard_normal(N).astype(np.float32)
    y = rng.standard_normal(N).astype(np.float32)

    scaled = k_scale[grid](x, np.zeros(N, dtype=np.float32), N=N)
    added = al.std.add(scaled, y, N=N)

    result, n_disp = _dispatches(lambda: np.array(added))
    assert n_disp == 1
    np.testing.assert_allclose(result, x * 2.0 + y, rtol=1e-5)


# ---------------------------------------------------------------------------
# Epilogue chain: dot → scale → bias (multi-step with extra)
# ---------------------------------------------------------------------------

def test_epilogue_chain_scale_then_add_bias():
    """dot(A, B) → scale(2.0) → add(bias) — chain with extra buffer."""
    M, N, K = 64, 64, 64
    rng = np.random.default_rng(42)
    A = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)
    B = (rng.standard_normal((K, N)) * 0.1).astype(np.float32)
    bias = (rng.standard_normal(N) * 0.1).astype(np.float32)

    r1 = al.dot(A, B)
    flat = (M * N + 1023) // 1024
    r2 = k_scale[(flat,)](r1, np.zeros(M * N, dtype=np.float32), N=M * N)
    r3 = al.std.add(r2, bias, N=M * N)

    result, n_disp = _dispatches(lambda: np.array(r3))
    assert n_disp == 1
    np.testing.assert_allclose(result.reshape(M, N), (A @ B) * 2.0 + bias,
                               rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# Prologue with extra: scale(x) fused into dot load
# ---------------------------------------------------------------------------

def test_epilogue_add_2d_extra_rectangular():
    """dot(A, B) + C where output is (M, N) with M != N.

    The extra C is (M, N) — same shape, but the row stride (N) is not equal
    to M. This exercises StridedTransform: the transform must carry the
    actual row stride, not assume it equals some ambient constant.
    """
    M, N, K = 32, 128, 64
    rng = np.random.default_rng(42)
    A = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)
    B = (rng.standard_normal((K, N)) * 0.1).astype(np.float32)
    C = (rng.standard_normal((M, N)) * 0.1).astype(np.float32)

    r1 = al.dot(A, B)
    r2 = al.std.add(r1, C, N=M * N)

    result, n_disp = _dispatches(lambda: np.array(r2))
    assert n_disp == 1
    np.testing.assert_allclose(result.reshape(M, N), A @ B + C, rtol=1e-3, atol=1e-3)


def test_epilogue_add_2d_extra_wide():
    """dot(A, B) + C where C is (M, N) with N much larger than M.

    Specifically tests that the row stride in the IndexTransform is N (128),
    not M (16) — a mismatch here would read garbage from the wrong rows.
    """
    M, N, K = 16, 128, 64
    rng = np.random.default_rng(42)
    A = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)
    B = (rng.standard_normal((K, N)) * 0.1).astype(np.float32)
    C = (rng.standard_normal((M, N)) * 0.1).astype(np.float32)

    r1 = al.dot(A, B)
    r2 = al.std.add(r1, C, N=M * N)

    result, n_disp = _dispatches(lambda: np.array(r2))
    assert n_disp == 1
    np.testing.assert_allclose(result.reshape(M, N), A @ B + C, rtol=1e-3, atol=1e-3)


def test_prologue_scale_into_dot():
    """scale(A) fused as prologue into dot — extra is the A buffer."""
    M, N, K = 64, 64, 64
    rng = np.random.default_rng(42)
    A = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)
    B = (rng.standard_normal((K, N)) * 0.1).astype(np.float32)

    scaled = k_scale[((M * K + 1023) // 1024,)](A, np.zeros(M * K, dtype=np.float32), N=M * K)
    r = al.dot(scaled.reshape(M, K), B)

    result, n_disp = _dispatches(lambda: np.array(r))
    assert n_disp == 1
    np.testing.assert_allclose(result, (A * 2.0) @ B, rtol=1e-3, atol=1e-3)
