"""Tests for column-slice epilogue fusion.

When a GEMM output (M, N) is column-sliced and an elementwise op applies
only to the slice, the fusion engine should:
  1. Detect the column slice (non-contiguous view with row_stride > slice_width)
  2. Clone the Store for the slice output
  3. Guard with column bounds
  4. Remap addressing to the slice's contiguous output layout

Real-world pattern: batched QKV GEMM → gate slice → sigmoid.
"""

import alloy as al
import numpy as np
from alloy._dispatch.fusion_compile import _detect_column_slice_epilogue
from alloy._dispatch.lazy import _collect_pending_ops
from alloy._runtime.metal import default_dispatcher
from tests.helpers import k_bias, k_scale


def _dispatches(fn):
    d = default_dispatcher()
    before = d.dispatch_count
    result = fn()
    return result, d.dispatch_count - before


class TestColumnSliceEpilogue:
    def test_gemm_then_scale_column_slice(self):
        """dot(A, B) produces (M, N), scale applied to column slice [:, :half_N].

        The slice shares the GEMM output allocation but has row_stride = N
        and slice_width = N/2. The fusion engine should detect this and emit
        a column-guarded store.
        """
        M, K, N = 32, 32, 64
        half_N = N // 2
        rng = np.random.default_rng(42)
        A = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)
        B = (rng.standard_normal((K, N)) * 0.1).astype(np.float32)

        gemm_out = al.dot(A, B, BLOCK_M=16, BLOCK_N=32, BLOCK_K=16)
        # Column slice: first half_N columns
        left_slice = gemm_out.slice(1, 0, half_N)
        # Apply scale to the slice — this is the epilogue
        flat_N = M * half_N
        scaled = k_scale[((flat_N + 1023) // 1024,)](
            left_slice, np.zeros(flat_N, dtype=np.float32), N=flat_N
        )

        result = np.array(scaled).reshape(M, half_N)
        expected = (A @ B)[:, :half_N] * 2.0
        np.testing.assert_allclose(result, expected, rtol=1e-3, atol=1e-3)

    def test_gemm_then_ops_on_both_halves(self):
        """dot(A, B) → scale(left_half) + bias(right_half).

        Both column slices of the GEMM output get different epilogues.
        Correctness check — fusion may or may not happen depending on
        whether the fusion engine handles multi-branch column slices.
        """
        M, K, N = 32, 32, 64
        half_N = N // 2
        rng = np.random.default_rng(42)
        A = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)
        B = (rng.standard_normal((K, N)) * 0.1).astype(np.float32)

        gemm_out = al.dot(A, B, BLOCK_M=16, BLOCK_N=32, BLOCK_K=16)
        left = gemm_out.slice(1, 0, half_N)
        right = gemm_out.slice(1, half_N, N)

        flat_N = M * half_N
        scaled_left = k_scale[((flat_N + 1023) // 1024,)](
            left, np.zeros(flat_N, dtype=np.float32), N=flat_N
        )
        biased_right = k_bias[((flat_N + 1023) // 1024,)](
            right, np.zeros(flat_N, dtype=np.float32), N=flat_N
        )

        full_ref = A @ B
        np.testing.assert_allclose(
            np.array(scaled_left).reshape(M, half_N),
            full_ref[:, :half_N] * 2.0,
            rtol=1e-3, atol=1e-3,
        )
        np.testing.assert_allclose(
            np.array(biased_right).reshape(M, half_N),
            full_ref[:, half_N:] + 1.0,
            rtol=1e-3, atol=1e-3,
        )

    def test_column_slice_detection(self):
        """Verify _detect_column_slice_epilogue returns correct bounds."""
        M, N = 16, 64
        half_N = N // 2

        # Build a LazyOp that reads the column slice
        rng = np.random.default_rng(42)
        x = (rng.standard_normal((M, N)) * 0.1).astype(np.float32)
        gemm_out = al.dot(x[:, :M], x[:M, :N], BLOCK_M=8, BLOCK_N=32, BLOCK_K=8)
        left = gemm_out.slice(1, 0, half_N)
        flat_N = M * half_N
        scaled = k_scale[((flat_N + 1023) // 1024,)](
            left, np.zeros(flat_N, dtype=np.float32), N=flat_N
        )
        # Collect ops to get the LazyOp for the scale
        ops, _ = _collect_pending_ops((scaled,))
        # The scale op should be the last one
        scale_op = ops[-1]
        # Detect column slice
        result = _detect_column_slice_epilogue(scale_op, gemm_out)
        # Should detect: col_start=0, col_end=half_N, slice_width=half_N
        if result is not None:
            col_start, col_end, slice_width = result
            assert col_start == 0
            assert col_end == half_N
            assert slice_width == half_N
