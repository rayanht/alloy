"""GPU correctness tests for im2col kernels (conv1d/conv2d unfolding)."""

import alloy as al
import numpy as np
import pytest


def _ref_im2col_1d(x_3d, IN_C, IN_LEN, OUT_LEN, K, STRIDE, PADDING):
    """Reference im2col for conv1d: x_3d is (batch, IN_C, IN_LEN)."""
    batch = x_3d.shape[0]
    CK = IN_C * K
    col = np.zeros((batch * OUT_LEN, CK), dtype=np.float32)
    for b in range(batch):
        for t in range(OUT_LEN):
            for ic in range(IN_C):
                for ki in range(K):
                    in_pos = t * STRIDE + ki - PADDING
                    if 0 <= in_pos < IN_LEN:
                        col[b * OUT_LEN + t, ic * K + ki] = x_3d[b, ic, in_pos]
    return col


def _ref_im2col_2d(x_4d, IN_C, IN_H, IN_W, OUT_H, OUT_W, KH, KW, STRIDE_H, STRIDE_W, PAD_H, PAD_W):
    """Reference im2col for conv2d: x_4d is (batch, IN_C, IN_H, IN_W)."""
    batch = x_4d.shape[0]
    CKK = IN_C * KH * KW
    col = np.zeros((batch * OUT_H * OUT_W, CKK), dtype=np.float32)
    for b in range(batch):
        for oh in range(OUT_H):
            for ow in range(OUT_W):
                idx = b * OUT_H * OUT_W + oh * OUT_W + ow
                for ic in range(IN_C):
                    for kh in range(KH):
                        for kw in range(KW):
                            ih = oh * STRIDE_H + kh - PAD_H
                            iw = ow * STRIDE_W + kw - PAD_W
                            col_off = ic * KH * KW + kh * KW + kw
                            if 0 <= ih < IN_H and 0 <= iw < IN_W:
                                col[idx, col_off] = x_4d[b, ic, ih, iw]
    return col


class TestIm2col1D:
    @pytest.mark.parametrize("IN_C,IN_LEN,K,STRIDE,PADDING", [
        (1, 8, 3, 1, 1),
        (3, 16, 3, 1, 0),
        (2, 10, 5, 2, 2),
    ])
    def test_correctness(self, IN_C, IN_LEN, K, STRIDE, PADDING):
        OUT_LEN = (IN_LEN + 2 * PADDING - K) // STRIDE + 1
        CK = IN_C * K
        batch = 1
        rng = np.random.default_rng(42)
        x_3d = rng.standard_normal((batch, IN_C, IN_LEN)).astype(np.float32)
        x_flat = x_3d.ravel()
        col = np.zeros((batch * OUT_LEN, CK), dtype=np.float32)

        result = al.std.im2col_1d(x_flat, col,
                                   IN_C=IN_C, IN_LEN=IN_LEN, OUT_LEN=OUT_LEN,
                                   CK=CK, K=K, STRIDE=STRIDE, PADDING=PADDING)
        expected = _ref_im2col_1d(x_3d, IN_C, IN_LEN, OUT_LEN, K, STRIDE, PADDING)
        np.testing.assert_allclose(result, expected, atol=1e-5)


class TestIm2col2D:
    @pytest.mark.parametrize("IN_C,IN_H,IN_W,KH,KW,STRIDE_H,STRIDE_W,PAD_H,PAD_W", [
        (1, 4, 4, 3, 3, 1, 1, 1, 1),
        (3, 8, 8, 3, 3, 1, 1, 0, 0),
        (1, 6, 6, 3, 3, 2, 2, 1, 1),
    ])
    def test_correctness(self, IN_C, IN_H, IN_W, KH, KW, STRIDE_H, STRIDE_W, PAD_H, PAD_W):
        OUT_H = (IN_H + 2 * PAD_H - KH) // STRIDE_H + 1
        OUT_W = (IN_W + 2 * PAD_W - KW) // STRIDE_W + 1
        CKK = IN_C * KH * KW
        batch = 1
        rng = np.random.default_rng(42)
        x_4d = rng.standard_normal((batch, IN_C, IN_H, IN_W)).astype(np.float32)
        x_flat = x_4d.ravel()
        col = np.zeros((batch * OUT_H * OUT_W, CKK), dtype=np.float32)

        result = al.std.im2col_2d(x_flat, col,
                                   IN_C=IN_C, IN_H=IN_H, IN_W=IN_W,
                                   OUT_H=OUT_H, OUT_W=OUT_W,
                                   KH=KH, KW=KW, CKK=CKK,
                                   STRIDE_H=STRIDE_H, STRIDE_W=STRIDE_W,
                                   PAD_H=PAD_H, PAD_W=PAD_W)
        expected = _ref_im2col_2d(x_4d, IN_C, IN_H, IN_W, OUT_H, OUT_W,
                                   KH, KW, STRIDE_H, STRIDE_W, PAD_H, PAD_W)
        np.testing.assert_allclose(result, expected, atol=1e-5)
