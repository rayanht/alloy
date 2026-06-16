"""GPU correctness tests for attention kernels."""

import alloy as al
import numpy as np
import pytest
from alloy._compiler.dtypes import float32
from alloy._dispatch.buf_utils import _alloc_aligned
from alloy.std.attention import (
    attention_decode_combine_vector,
    attention_decode_vector_split,
)
from tests.helpers import ref_attention, ref_attention_batched


# ---------------------------------------------------------------------------
# Basic attention (flattened BH*N, D layout)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N,D", [(16, 16), (32, 32), (64, 32)])
def test_attention_basic(N, D):
    rng = np.random.default_rng(42)
    Q = (rng.standard_normal((N, D)) * 0.1).astype(np.float32)
    K = (rng.standard_normal((N, D)) * 0.1).astype(np.float32)
    V = (rng.standard_normal((N, D)) * 0.1).astype(np.float32)
    np.testing.assert_allclose(al.attention(Q, K, V), ref_attention(Q, K, V),
                               rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("N,D", [(16, 16), (32, 32)])
def test_attention_causal(N, D):
    rng = np.random.default_rng(42)
    Q = (rng.standard_normal((N, D)) * 0.1).astype(np.float32)
    K = (rng.standard_normal((N, D)) * 0.1).astype(np.float32)
    V = (rng.standard_normal((N, D)) * 0.1).astype(np.float32)
    np.testing.assert_allclose(al.attention(Q, K, V, causal=1),
                               ref_attention(Q, K, V, causal=True),
                               rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# Batched attention
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("BH,N,D", [(2, 16, 16), (4, 16, 16)])
def test_attention_batched(BH, N, D):
    rng = np.random.default_rng(42)
    Q = (rng.standard_normal((BH, N, D)) * 0.1).astype(np.float32)
    K = (rng.standard_normal((BH, N, D)) * 0.1).astype(np.float32)
    V = (rng.standard_normal((BH, N, D)) * 0.1).astype(np.float32)
    result = al.attention(Q.reshape(BH * N, D), K.reshape(BH * N, D),
                          V.reshape(BH * N, D), BH=BH)
    result_arr = np.asarray(result).reshape(BH, N, D)
    np.testing.assert_allclose(result_arr, ref_attention_batched(Q, K, V),
                               rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# Masked attention
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N,D", [(16, 16), (32, 32)])
def test_attention_masked(N, D):
    rng = np.random.default_rng(42)
    Q = (rng.standard_normal((N, D)) * 0.1).astype(np.float32)
    K = (rng.standard_normal((N, D)) * 0.1).astype(np.float32)
    V = (rng.standard_normal((N, D)) * 0.1).astype(np.float32)
    # Additive mask: zeros → no masking effect
    mask = np.zeros((N, N), dtype=np.float32)
    result = al.std.attention_masked_by_batch(Q, K, V, mask, BH=1,
                                               BLOCK_M=min(N, 16), BLOCK_N=min(N, 16))
    np.testing.assert_allclose(result, ref_attention(Q, K, V), rtol=5e-3, atol=5e-3)


def test_attention_masked_with_causal_mask():
    N, D = 16, 16
    rng = np.random.default_rng(42)
    Q = (rng.standard_normal((N, D)) * 0.1).astype(np.float32)
    K = (rng.standard_normal((N, D)) * 0.1).astype(np.float32)
    V = (rng.standard_normal((N, D)) * 0.1).astype(np.float32)
    # Build causal additive mask: upper triangle = -1e30
    mask = np.triu(np.full((N, N), -1e30, dtype=np.float32), k=1)
    result = al.std.attention_masked_by_batch(Q, K, V, mask, BH=1,
                                               BLOCK_M=16, BLOCK_N=16)
    np.testing.assert_allclose(result, ref_attention(Q, K, V, causal=True),
                               rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# Strided attention (4D tensor layout with explicit strides)
# ---------------------------------------------------------------------------

def test_attention_strided_single_head():
    B, H, N, D = 1, 1, 16, 16
    rng = np.random.default_rng(42)
    Q = (rng.standard_normal((B, H, N, D)) * 0.1).astype(np.float32)
    K = (rng.standard_normal((B, H, N, D)) * 0.1).astype(np.float32)
    V = (rng.standard_normal((B, H, N, D)) * 0.1).astype(np.float32)
    BH = B * H
    result = al.std.attention_strided(
        Q.ravel(), K.ravel(), V.ravel(),
        BH=BH, HEADS_PER_BATCH=H, SEQ_LEN=N, HEAD_DIM=D,
        Q_BATCH_STRIDE=H * N * D, Q_HEAD_STRIDE=N * D, Q_SEQ_STRIDE=D,
        K_BATCH_STRIDE=H * N * D, K_HEAD_STRIDE=N * D, K_SEQ_STRIDE=D,
        V_BATCH_STRIDE=H * N * D, V_HEAD_STRIDE=N * D, V_SEQ_STRIDE=D,
        BLOCK_M=16, BLOCK_N=16,
    )
    expected = ref_attention(Q[0, 0], K[0, 0], V[0, 0])
    np.testing.assert_allclose(np.asarray(result).reshape(N, D), expected,
                               rtol=1e-3, atol=1e-3)


def test_attention_strided_multi_head():
    B, H, N, D = 1, 2, 16, 16
    rng = np.random.default_rng(42)
    Q = (rng.standard_normal((B, H, N, D)) * 0.1).astype(np.float32)
    K = (rng.standard_normal((B, H, N, D)) * 0.1).astype(np.float32)
    V = (rng.standard_normal((B, H, N, D)) * 0.1).astype(np.float32)
    BH = B * H
    result = al.std.attention_strided(
        Q.ravel(), K.ravel(), V.ravel(),
        BH=BH, HEADS_PER_BATCH=H, SEQ_LEN=N, HEAD_DIM=D,
        Q_BATCH_STRIDE=H * N * D, Q_HEAD_STRIDE=N * D, Q_SEQ_STRIDE=D,
        K_BATCH_STRIDE=H * N * D, K_HEAD_STRIDE=N * D, K_SEQ_STRIDE=D,
        V_BATCH_STRIDE=H * N * D, V_HEAD_STRIDE=N * D, V_SEQ_STRIDE=D,
        BLOCK_M=16, BLOCK_N=16,
    )
    # Output layout: O[batch*N*O_STRIDE + head*D + seq*O_STRIDE + d]
    # where O_STRIDE = H*D. So output is (allocated_rows, H*D), take first B*N rows.
    result_flat = np.asarray(result).ravel()
    O_STRIDE = H * D
    result_arr = np.zeros((B, N, H, D), dtype=np.float32)
    for b in range(B):
        for n in range(N):
            for h in range(H):
                off = b * N * O_STRIDE + n * O_STRIDE + h * D
                result_arr[b, n, h, :] = result_flat[off:off + D]
    for h in range(H):
        expected = ref_attention(Q[0, h], K[0, h], V[0, h])
        np.testing.assert_allclose(result_arr[0, :, h, :], expected, rtol=1e-3, atol=1e-3)


def test_attention_kv_update_causal_decode_matches_reference():
    """Fused cache update + causal decode attention for grouped-query heads."""
    rng = np.random.default_rng(123)
    q_heads, kv_heads, kv_group = 4, 2, 2
    n, d, kv_len, cache_pos = 1, 32, 33, 16

    q = (rng.standard_normal((1, q_heads, n, d)) * 0.1).astype(np.float32)
    new_k = (rng.standard_normal((kv_heads, d)) * 0.1).astype(np.float32)
    new_v = (rng.standard_normal((kv_heads, d)) * 0.1).astype(np.float32)
    k_cache = (rng.standard_normal((kv_heads, kv_len, d)) * 0.1).astype(np.float32)
    v_cache = (rng.standard_normal((kv_heads, kv_len, d)) * 0.1).astype(np.float32)
    cache_pos_buf = np.array([cache_pos], dtype=np.int32)

    result = al.std.attention_kv_update(
        q.ravel(),
        new_k.ravel(),
        new_v.ravel(),
        cache_pos_buf,
        k_cache.ravel(),
        v_cache.ravel(),
        BH=q_heads,
        HEADS_PER_BATCH=q_heads,
        SEQ_LEN=n,
        HEAD_DIM=d,
        Q_BATCH_STRIDE=q_heads * n * d,
        Q_HEAD_STRIDE=n * d,
        Q_SEQ_STRIDE=d,
        NK_HEAD_STRIDE=d,
        NV_HEAD_STRIDE=d,
        KC_HEAD_STRIDE=kv_len * d,
        KC_SEQ_STRIDE=d,
        VC_HEAD_STRIDE=kv_len * d,
        VC_SEQ_STRIDE=d,
        BLOCK_M=8,
        BLOCK_N=16,
        causal=1,
        KV_GROUP=kv_group,
        KV_LEN=kv_len,
    )

    got = np.asarray(result).reshape(-1)[: q_heads * d].reshape(q_heads, d)
    k_updated = k_cache.copy()
    v_updated = v_cache.copy()
    k_updated[:, cache_pos, :] = new_k
    v_updated[:, cache_pos, :] = new_v

    expected = np.zeros((q_heads, d), dtype=np.float32)
    scale = 1.0 / np.sqrt(float(d))
    for head in range(q_heads):
        kv_head = head // kv_group
        scores = k_updated[kv_head, : cache_pos + 1] @ q[0, head, 0] * scale
        scores = scores - scores.max()
        probs = np.exp(scores)
        probs = probs / probs.sum()
        expected[head] = probs @ v_updated[kv_head, : cache_pos + 1]

    np.testing.assert_allclose(got, expected, rtol=1e-3, atol=1e-3)


def test_decode_vector_split_sliding_window_wrap():
    """Vector flash-decode past a circular sliding window.

    The sliding K/V cache is a circular buffer of physical size W; once
    cache_pos >= W the write wraps (slot = pos % W) and the live window is the
    LAST W logical positions [cache_pos+1-W, cache_pos]. The kernel must walk
    those logical positions (reading slot = j % W), not stop at the physical
    buffer end W. It previously bounded the K-loop at N_KV (== W), so after the
    wrap it dropped the newest positions (the self token first) — on gemma4 a
    45-token prompt then degenerated hard ~6 tokens past position 512. Here
    cache_pos=600 wraps by 89 positions; the reference attends the full window.
    """
    D, W, splits = 128, 512, 4
    cache_pos = 600                       # wrapped: window lo = cache_pos+1-W = 89
    lo = cache_pos + 1 - W
    rng = np.random.default_rng(0)
    kp = {p: rng.standard_normal(D).astype(np.float32) for p in range(lo, cache_pos + 1)}
    vp = {p: rng.standard_normal(D).astype(np.float32) for p in range(lo, cache_pos + 1)}
    q = rng.standard_normal(D).astype(np.float32)
    k_cache = np.zeros((W, D), np.float32)
    v_cache = np.zeros((W, D), np.float32)
    for p in range(lo, cache_pos):        # pre-write every window pos but the current one
        k_cache[p % W] = kp[p]
        v_cache[p % W] = vp[p]
    pos_buf = np.array([cache_pos], np.int32)
    scale = float(1.0 / np.sqrt(D))

    partial_o = _alloc_aligned((1, splits, D), float32)
    partial_lse = _alloc_aligned((1, splits), float32)
    out = _alloc_aligned((1, D), float32)
    # kp[cache_pos]/vp[cache_pos] are the current token; the kernel writes them
    # into the cache at slot cache_pos % W before attending (WRITE_KV=1).
    attention_decode_vector_split[(1, splits)](
        q.ravel(), kp[cache_pos].ravel(), vp[cache_pos].ravel(), pos_buf,
        k_cache.ravel(), v_cache.ravel(), partial_o, partial_lse,
        BH=1, HEADS_PER_BATCH=1, HEAD_DIM=D,
        KC_SEQ_STRIDE=D, VC_SEQ_STRIDE=D,
        KV_GROUP=1, KV_LEN=W, SPLITS=splits, SLIDING_WINDOW=W,
        CUSTOM_SCALE=scale, WRITE_KV=1,
    )
    attention_decode_combine_vector[(1,)](
        partial_o, partial_lse, out, BH=1, HEADS_PER_BATCH=1, HEAD_DIM=D, SPLITS=splits,
    )
    got = np.asarray(out).reshape(D)

    # Reference: full softmax attention over the logical window [lo, cache_pos].
    kw = np.stack([kp[p] for p in range(lo, cache_pos + 1)])
    vw = np.stack([vp[p] for p in range(lo, cache_pos + 1)])
    s = (kw @ q) * scale
    s = s - s.max()
    probs = np.exp(s)
    probs = probs / probs.sum()
    expected = probs @ vw
    np.testing.assert_allclose(got, expected, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# GQA (grouped-query attention)
# ---------------------------------------------------------------------------

def test_attention_gqa():
    """GQA: 4 Q heads share 2 KV heads (KV_GROUP=2)."""
    N, D = 16, 16
    Q_HEADS, KV_HEADS = 4, 2
    KV_GROUP = Q_HEADS // KV_HEADS
    rng = np.random.default_rng(42)
    Q = (rng.standard_normal((Q_HEADS, N, D)) * 0.1).astype(np.float32)
    K = (rng.standard_normal((KV_HEADS, N, D)) * 0.1).astype(np.float32)
    V = (rng.standard_normal((KV_HEADS, N, D)) * 0.1).astype(np.float32)

    # Expand K/V to match Q heads for reference
    K_expanded = np.repeat(K, KV_GROUP, axis=0)
    V_expanded = np.repeat(V, KV_GROUP, axis=0)

    result = al.attention(Q.reshape(Q_HEADS * N, D),
                          K_expanded.reshape(Q_HEADS * N, D),
                          V_expanded.reshape(Q_HEADS * N, D),
                          BH=Q_HEADS)
    expected = ref_attention_batched(Q, K_expanded, V_expanded)
    np.testing.assert_allclose(np.asarray(result).reshape(Q_HEADS, N, D),
                               expected, rtol=1e-3, atol=1e-3)
