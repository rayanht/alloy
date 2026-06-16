"""Regression tests for elementwise kernels that previously missed fusion."""

import alloy as al
import numpy as np
from alloy._runtime.buffer_ops import k_bitwise_and_nd, k_le_nd
from alloy._runtime.metal import default_dispatcher
from alloy_torch.ops.elementwise import k_where_nd
from tests.helpers import get_unary_kernel, k_scale


def _dispatches(fn):
    d = default_dispatcher()
    before = d.dispatch_count
    result = fn()
    return result, d.dispatch_count - before


def test_log_fuses_with_scale():
    """scale(x) → log(scaled) should fuse to 1 dispatch.

    k_log is a standard load→al.log→store pattern.
    """
    N = 4096
    grid = (N + 1023) // 1024
    rng = np.random.default_rng(42)
    x = np.abs(rng.standard_normal(N).astype(np.float32)) + 0.01

    scaled = k_scale[grid](x, np.zeros(N, dtype=np.float32), N=N)

    # k_log from the torch registry uses _make_elementwise_unary("k_log", lambda v: al.log(v)).
    # Replicate with our own kernel to avoid importing the registry here.
    log_k = get_unary_kernel("log")
    result_buf = log_k[grid](scaled, np.zeros(N, dtype=np.float32), N=N)

    result, n_disp = _dispatches(lambda: np.array(result_buf))
    expected = np.log(x * 2.0)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
    assert n_disp == 1, f"scale→log should fuse to 1 dispatch, got {n_disp}"


def test_le_nd_fuses_with_scale():
    """scale(x) → le_nd(scaled, threshold) should fuse to 1 dispatch.

    k_le_nd uses 4D strided indexing but in the 1D contiguous case
    it's a standard load→compare→where→store pattern.
    """
    N = 4096
    grid = (N + 1023) // 1024
    rng = np.random.default_rng(42)
    x = rng.standard_normal(N).astype(np.float32)
    threshold = np.full(N, 0.5, dtype=np.float32)

    scaled = k_scale[grid](x, np.zeros(N, dtype=np.float32), N=N)
    le_result = k_le_nd[grid](
        scaled, threshold, np.zeros(N, dtype=np.float32),
        N=N, OUT0=1, OUT1=1, OUT2=1, OUT3=N,
        X_STR0=0, X_STR1=0, X_STR2=0, X_STR3=1,
        Y_STR0=0, Y_STR1=0, Y_STR2=0, Y_STR3=1,
    )

    result, n_disp = _dispatches(lambda: np.array(le_result))
    expected = np.where(x * 2.0 <= threshold, 1.0, 0.0)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
    assert n_disp == 1, f"scale→le_nd should fuse to 1 dispatch, got {n_disp}"


def test_bitwise_and_nd_fuses_with_scale():
    """scale(x) → bitwise_and(scaled, mask) should fuse to 1 dispatch.

    k_bitwise_and_nd uses cast→bitwise→cast to handle float buffers.
    """
    N = 4096
    grid = (N + 1023) // 1024
    rng = np.random.default_rng(42)
    x = rng.integers(0, 256, size=N).astype(np.float32)

    scaled = k_scale[grid](x, np.zeros(N, dtype=np.float32), N=N)
    mask = np.full(N, 0x0F, dtype=np.float32)
    result_buf = k_bitwise_and_nd[grid](
        scaled, mask, np.zeros(N, dtype=np.float32),
        N=N, OUT0=1, OUT1=1, OUT2=1, OUT3=N,
        X_STR0=0, X_STR1=0, X_STR2=0, X_STR3=1,
        Y_STR0=0, Y_STR1=0, Y_STR2=0, Y_STR3=1,
    )

    result, n_disp = _dispatches(lambda: np.array(result_buf))
    expected = ((x * 2.0).astype(np.int32) & 0x0F).astype(np.float32)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
    assert n_disp == 1, f"scale→bitwise_and_nd should fuse to 1 dispatch, got {n_disp}"


def test_where_nd_fuses_with_scale():
    """scale(x) → where(cond, scaled, y) should fuse to 1 dispatch.

    k_where_nd uses al.where with a single store.
    """
    N = 4096
    grid = (N + 1023) // 1024
    rng = np.random.default_rng(42)
    x = rng.standard_normal(N).astype(np.float32)
    cond = (rng.standard_normal(N) > 0).astype(np.float32)
    y_val = np.full(N, -1.0, dtype=np.float32)

    scaled = k_scale[grid](x, np.zeros(N, dtype=np.float32), N=N)
    result_buf = k_where_nd[grid](
        cond, scaled, y_val, np.zeros(N, dtype=np.float32),
        N=N, OUT0=1, OUT1=1, OUT2=1, OUT3=N,
        C_STR0=0, C_STR1=0, C_STR2=0, C_STR3=1,
        X_STR0=0, X_STR1=0, X_STR2=0, X_STR3=1,
        Y_STR0=0, Y_STR1=0, Y_STR2=0, Y_STR3=1,
    )

    result, n_disp = _dispatches(lambda: np.array(result_buf))
    expected = np.where(cond != 0, x * 2.0, -1.0)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
    assert n_disp == 1, f"scale→where_nd should fuse to 1 dispatch, got {n_disp}"


def test_logical_not_fuses_with_compare():
    """compare(x, 0) → logical_not should fuse to 1 dispatch.

    k_logical_not uses al.where(v == 0, 1, 0) — standard elem pattern.
    """
    N = 4096
    grid = (N + 1023) // 1024
    rng = np.random.default_rng(42)
    x = rng.standard_normal(N).astype(np.float32)

    scaled = k_scale[grid](x, np.zeros(N, dtype=np.float32), N=N)

    @al.kernel
    def k_logical_not(x, out: al.output, N: al.constexpr):
        pid = al.program_id(0)
        offs = pid * 1024 + al.arange(0, 1024)
        mask = offs < N
        v = al.load(x + offs, mask=mask)
        al.store(out + offs, al.where(v == 0, 1, 0), mask=mask)

    result_buf = k_logical_not[grid](scaled, np.zeros(N, dtype=np.float32), N=N)

    result, n_disp = _dispatches(lambda: np.array(result_buf))
    # x * 2.0 is never exactly 0 for random floats, so logical_not should be all 0
    expected = np.where(x * 2.0 == 0, 1, 0).astype(np.float32)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
    assert n_disp == 1, f"scale→logical_not should fuse to 1 dispatch, got {n_disp}"
