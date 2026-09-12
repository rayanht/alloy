"""The numpy K-quantizers round-trip through gguf-py's dequantizer with the expected error."""

import gguf
import numpy as np

from alloy_server.gguf import kquant


def rel_err(x, y):
    return np.abs(x - y).mean() / np.abs(x).mean()


def test_q4_k_roundtrip():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((16, 1024)).astype(np.float32)
    packed = kquant.quantize_q4_k(x)
    assert packed.shape == (16, 4 * 144)
    back = gguf.quants.dequantize(packed, gguf.GGMLQuantizationType.Q4_K)
    assert back.shape == x.shape
    assert rel_err(x, back) < 0.09
    # zero blocks stay zero
    z = kquant.quantize_q4_k(np.zeros((1, 256), dtype=np.float32))
    assert np.all(gguf.quants.dequantize(z, gguf.GGMLQuantizationType.Q4_K) == 0)


def test_q6_k_roundtrip():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((8, 2, 512)).astype(np.float32)
    packed = kquant.quantize_q6_k(x)
    assert packed.shape == (8, 2, 2 * 210)
    back = gguf.quants.dequantize(packed.reshape(16, -1), gguf.GGMLQuantizationType.Q6_K)
    assert rel_err(x.reshape(16, -1), back) < 0.025
    z = kquant.quantize_q6_k(np.zeros((1, 256), dtype=np.float32))
    assert np.all(gguf.quants.dequantize(z, gguf.GGMLQuantizationType.Q6_K) == 0)
