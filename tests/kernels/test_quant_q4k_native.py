"""GPU correctness tests for the GGUF-native Q4_K kernels (144-byte superblocks).

Reference: a numpy decode of the exact GGUF block_q4_K layout (d/dmin f16,
12B packed 6-bit scales/mins via get_scale_min_k4, 128B interleaved nibbles;
weight = d*scale*nibble - dmin*min). Every migrated kernel decodes the same
random blocks and must match this reference. The matvec (dot_q4_k_v2) is the
kernel_bench/llama-validated body; this additionally pins the new pieces — the
tiled cooperative-load dequant, the amortized rows kernels, and embedding.
"""

import numpy as np
import torch
import torch.nn.functional as F

from alloy.std.quant import (
    dot_q4_k,
    dot_q4_k_gelu_v2,
    dot_q4_k_silu_v2,
    dot_q4_k_silu_v2_rows,
    dot_q4_k_v2,
    dot_q4_k_v2_rows,
    embedding_q4_k,
)
from alloy._compiler.dtypes import float32, int32, uint8
from alloy._dispatch.buf_utils import _alloc_aligned
from alloy._runtime._metal_ext import gpu_sync
from alloy._runtime.alloy_buffer import materialize_many


def _buf(arr, dtype):
    b = _alloc_aligned(tuple(arr.shape), dtype)
    b.numpy[:] = arr
    return b


def _make_blocks(N, NB, rng):
    """Random valid Q4_K blocks (N, NB*144) uint8 with sane f16 d/dmin."""
    blk = rng.integers(0, 256, size=(N, NB, 144), dtype=np.uint8)
    d = (rng.standard_normal((N, NB)) * 0.02).astype(np.float16)
    dmin = (rng.standard_normal((N, NB)) * 0.01).astype(np.float16)
    blk[:, :, 0:2] = d.reshape(N, NB).view(np.uint8).reshape(N, NB, 2)
    blk[:, :, 2:4] = dmin.reshape(N, NB).view(np.uint8).reshape(N, NB, 2)
    return blk.reshape(N, NB * 144)


def _get_scale_min_k4(j, sc):
    if j < 4:
        return sc[j] & 63, sc[j + 4] & 63
    return ((sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4),
            (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4))


def _dequant(blk, N, NB):
    """blk: (N, NB*144) uint8 -> W (N, NB*256) float32."""
    W = np.empty((N, NB * 256), dtype=np.float32)
    b = blk.reshape(N, NB, 144)
    for n in range(N):
        for ib in range(NB):
            block = b[n, ib]
            d = block[0:2].view(np.float16)[0].astype(np.float32)
            dmin = block[2:4].view(np.float16)[0].astype(np.float32)
            sc = block[4:16].astype(np.int32)
            qs = block[16:144].astype(np.int32)
            for g in range(8):
                s, m = _get_scale_min_k4(g, sc)
                base = ib * 256 + g * 32
                for l in range(32):
                    byte = qs[(g // 2) * 32 + l]
                    nib = (byte >> 4) if (g & 1) else (byte & 0xF)
                    W[n, base + l] = d * float(s) * float(nib) - dmin * float(m)
    return W


def _ceil(a, b):
    return (a + b - 1) // b


def test_q4k_matvec_decode_matches_reference():
    N, K = 64, 2560  # NB=10 -> R=2 (exercises the masked tail)
    NB = K // 256
    rng = np.random.default_rng(0)
    blk = _make_blocks(N, NB, rng)
    W = _dequant(blk, N, NB)
    A = (rng.standard_normal((1, K)) * 0.5).astype(np.float32)
    ref = A @ W.T  # (1, N)

    for nsg, nr0 in [(1, 1), (2, 2), (2, 1), (1, 4)]:
        a_buf = _buf(A, float32)
        blk_buf = _buf(blk, uint8)
        c = _alloc_aligned((N,), float32)
        dot_q4_k_v2[(N // (nr0 * nsg),)](a_buf, blk_buf, c, NSG=nsg, NR0=nr0)
        materialize_many([c])
        gpu_sync()
        got = c.numpy.copy()
        assert np.allclose(got, ref.reshape(-1), atol=3e-3, rtol=3e-3), (
            f"nsg={nsg} nr0={nr0} max abs {np.abs(got - ref.reshape(-1)).max():.3e}"
        )


def test_q4k_tiled_coop_load_matches_reference():
    N, K, M = 64, 2560, 48
    NB = K // 256
    rng = np.random.default_rng(1)
    blk = _make_blocks(N, NB, rng)
    W = _dequant(blk, N, NB)
    A = (rng.standard_normal((M, K)) * 0.5).astype(np.float32)
    # The tiled path downcasts both the activation and the dequantized weight to
    # f16 in shared memory (f32 MMA accumulator), so the reference is the f16
    # matmul, not the f32 one.
    Af16 = A.astype(np.float16).astype(np.float32)
    Wf16 = W.astype(np.float16).astype(np.float32)
    ref = Af16 @ Wf16.T  # (M, N)

    BM, BN, BK = 16, 64, 64
    a_buf = _buf(A, float32)
    blk_buf = _buf(blk, uint8)
    c = _alloc_aligned((M * N,), float32)
    dot_q4_k[(_ceil(M, BM), _ceil(N, BN))](
        a_buf, blk_buf, c, BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK
    )
    materialize_many([c])
    gpu_sync()
    got = c.numpy.copy().reshape(M, N)
    assert np.allclose(got, ref, atol=2e-1, rtol=5e-3), (
        f"max abs {np.abs(got - ref).max():.3e}"
    )


def test_q4k_rows_matches_reference():
    N, K, M = 64, 2560, 5
    NB = K // 256
    rng = np.random.default_rng(2)
    blk = _make_blocks(N, NB, rng)
    W = _dequant(blk, N, NB)
    A = (rng.standard_normal((M, K)) * 0.5).astype(np.float32)
    ref = A @ W.T  # (M, N)

    a_buf = _buf(A, float32)
    blk_buf = _buf(blk, uint8)
    c = _alloc_aligned((M * N,), float32)
    dot_q4_k_v2_rows[(N,)](a_buf, blk_buf, c)
    materialize_many([c])
    gpu_sync()
    got = c.numpy.copy().reshape(M, N)
    assert np.allclose(got, ref, atol=3e-3, rtol=3e-3), (
        f"max abs {np.abs(got - ref).max():.3e}"
    )


def test_q4k_embedding_matches_reference():
    N, K, T = 100, 2560, 8
    NB = K // 256
    rng = np.random.default_rng(3)
    blk = _make_blocks(N, NB, rng)
    W = _dequant(blk, N, NB)
    ids = rng.integers(0, N, size=T).astype(np.int32)
    ref = W[ids]  # (T, K)

    ids_buf = _buf(ids, int32)
    blk_buf = _buf(blk.reshape(-1), uint8)
    out = _alloc_aligned((T * K,), float32)
    embedding_q4_k[(_ceil(T * K, 1024),)](
        ids_buf, blk_buf, out, NUM_INDICES=T, WIDTH=K
    )
    materialize_many([out])
    gpu_sync()
    got = out.numpy.copy().reshape(T, K)
    assert np.allclose(got, ref, atol=1e-3, rtol=1e-3), (
        f"max abs {np.abs(got - ref).max():.3e}"
    )


def test_q4k_silu_matvec_matches_reference():
    N, K = 64, 2560
    NB = K // 256
    rng = np.random.default_rng(4)
    gblk = _make_blocks(N, NB, rng)
    ublk = _make_blocks(N, NB, rng)
    G = _dequant(gblk, N, NB)
    U = _dequant(ublk, N, NB)
    A = (rng.standard_normal((1, K)) * 0.5).astype(np.float32)
    g = torch.from_numpy(A @ G.T)
    u = torch.from_numpy(A @ U.T)
    ref_silu = (g * torch.sigmoid(g) * u).numpy().reshape(-1)
    ref_gelu = (F.gelu(g, approximate="tanh") * u).numpy().reshape(-1)

    a_buf = _buf(A, float32)
    g_buf = _buf(gblk, uint8)
    u_buf = _buf(ublk, uint8)
    cs = _alloc_aligned((N,), float32)
    cg = _alloc_aligned((N,), float32)
    dot_q4_k_silu_v2[(N // 2,)](a_buf, g_buf, u_buf, cs, NSG=2, NR0=1)
    dot_q4_k_gelu_v2[(N // 2,)](a_buf, g_buf, u_buf, cg, NSG=2, NR0=1)
    materialize_many([cs, cg])
    gpu_sync()
    assert np.allclose(cs.numpy.copy(), ref_silu, atol=3e-3, rtol=3e-3), (
        f"silu max abs {np.abs(cs.numpy.copy() - ref_silu).max():.3e}"
    )
    assert np.allclose(cg.numpy.copy(), ref_gelu, atol=3e-3, rtol=3e-3), (
        f"gelu max abs {np.abs(cg.numpy.copy() - ref_gelu).max():.3e}"
    )


def test_q4k_silu_rows_matches_reference():
    N, K, M = 64, 2560, 5
    NB = K // 256
    rng = np.random.default_rng(5)
    gblk = _make_blocks(N, NB, rng)
    ublk = _make_blocks(N, NB, rng)
    G = _dequant(gblk, N, NB)
    U = _dequant(ublk, N, NB)
    A = (rng.standard_normal((M, K)) * 0.5).astype(np.float32)
    g = torch.from_numpy(A @ G.T)
    u = torch.from_numpy(A @ U.T)
    ref = (g * torch.sigmoid(g) * u).numpy()

    a_buf = _buf(A, float32)
    g_buf = _buf(gblk, uint8)
    u_buf = _buf(ublk, uint8)
    c = _alloc_aligned((M * N,), float32)
    dot_q4_k_silu_v2_rows[(N,)](a_buf, g_buf, u_buf, c)
    materialize_many([c])
    gpu_sync()
    got = c.numpy.copy().reshape(M, N)
    assert np.allclose(got, ref, atol=3e-3, rtol=3e-3), (
        f"max abs {np.abs(got - ref).max():.3e}"
    )
