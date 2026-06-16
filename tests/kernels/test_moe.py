"""GPU correctness tests for the Qwen3.5-MoE routed-expert kernels.

Strategy: the gathered MoE kernels must produce, for a token routed to expert
`e`, exactly what the proven non-MoE quant kernels produce on expert `e`'s
weight slice. So we generate random Q4_K/Q6_K bytes, route one token to a chosen
expert, and compare the gathered kernel against the existing kernel applied to
the Python-sliced expert. Identical bytes + identical math ⇒ bit-exact match,
which pins the runtime gather + fused-layout addressing.
"""

import numpy as np
import torch

import torch.nn.functional as F

from alloy.std.moe import moe_gate_up_silu, moe_down_combine, moe_router_topk
from alloy.std.quant import dot_q4_k_silu_v2, dot_q6_k_v2
from alloy_torch.ops.moe import _gguf_moe_routed_handler
from alloy._compiler.dtypes import float32, uint8, int32
from alloy._dispatch.buf_utils import _alloc_aligned
from alloy._runtime._metal_ext import gpu_sync
from alloy._runtime.alloy_buffer import materialize_many


def _buf(arr, dtype):
    b = _alloc_aligned(tuple(arr.shape), dtype)
    b.numpy[:] = arr
    return b


def _q4k_blocks(rows, nb, rng):
    """Random valid GGUF-native Q4_K blocks (rows..., nb*144) uint8 with sane f16
    d/dmin. d is small because the native weight is d*scale_raw*nibble with the
    6-bit scale_raw up to 63 — d~0.002 gives an effective per-weight scale ~0.05
    (the magnitude the OLD normalized test used), keeping the two-stage MoE GEMM
    within f16 range (random d~0.02 overflows f16 after gate_up→down compounding)."""
    blk = rng.integers(0, 256, size=rows + (nb, 144), dtype=np.uint8)
    n = nb
    for r in rows:
        n *= r
    d = (rng.standard_normal(n) * 0.002).astype(np.float16).view(np.uint8).reshape(rows + (nb, 2))
    dm = (rng.standard_normal(n) * 0.001).astype(np.float16).view(np.uint8).reshape(rows + (nb, 2))
    blk[..., 0:2] = d
    blk[..., 2:4] = dm
    return blk.reshape(rows + (nb * 144,))


def test_moe_gate_up_silu_matches_sliced_expert():
    K, I, E = 256, 64, 4
    NB = K // 256
    rng = np.random.default_rng(0)

    # Fused gate_up: (E, 2I, NB*144) GGUF-native Q4_K superblocks.
    gu = _q4k_blocks((E, 2 * I), NB, rng)
    A = rng.standard_normal((1, K)).astype(np.float32)

    e = 2  # route the single token to expert 2
    routing = np.array([e], dtype=np.int32)

    # --- gathered MoE kernel ---
    a_buf = _buf(A.reshape(-1), float32)
    gu_buf = _buf(gu.reshape(-1), uint8)
    routing_buf = _buf(routing, int32)
    h_out = _alloc_aligned((1 * I,), float32)
    moe_gate_up_silu[(1, I)](
        a_buf, gu_buf, routing_buf, h_out,
        K=K, MOE_INTER=I, TOP_K=1,
    )

    # --- reference: dot_q4_k_silu_v2 on expert e's gate / up slices ---
    gate_blk = np.ascontiguousarray(gu[e, 0:I])
    up_blk = np.ascontiguousarray(gu[e, I:2 * I])
    a_buf2 = _buf(A, float32)                       # (1, K) — ref reads A.shape
    gate_buf = _buf(gate_blk, uint8)               # (I, NB*144) — ref reads .shape[0]=N
    up_buf = _buf(up_blk, uint8)
    h_ref = _alloc_aligned((1, I), float32)
    dot_q4_k_silu_v2[(I,)](a_buf2, gate_buf, up_buf, h_ref, NSG=1, NR0=1)

    materialize_many([h_out, h_ref])
    gpu_sync()

    got = torch.from_numpy(h_out.numpy.copy()).reshape(I)
    ref = torch.from_numpy(h_ref.numpy.copy()).reshape(I)
    assert torch.allclose(got, ref, atol=1e-4, rtol=1e-4), (
        f"max abs diff {(got - ref).abs().max().item():.3e}"
    )


def test_gguf_moe_routed_handler_matches_per_expert_reference():
    """Full routed op (router topk → gate_up silu → down combine) vs an
    independent reference: torch router + the *existing* per-expert quant
    kernels, weighted-summed. Exercises the handler + 3-kernel chaining for
    BOTH dispatch paths: T==1 decode (gathered f32 GEMV — tight) and T>1 prefill
    (grouped f16-MMA GEMM — f16 tolerance, like the dense GEMMs)."""
    H, I, E, TOP_K = 256, 512, 4, 2
    NB = H // 256
    ROW = (I // 256) * 210
    rng = np.random.default_rng(7)

    gu = _q4k_blocks((E, 2 * I), NB, rng)
    down = rng.integers(0, 256, (E, H, ROW), dtype=np.uint8)
    dview = down.reshape(E, H, I // 256, 210)
    db = np.array([0.02], dtype=np.float16).view(np.uint8)
    dview[..., 208] = db[0]
    dview[..., 209] = db[1]
    # T==1 -> decode/GEMV path (f32, tight); T>1 -> prefill/grouped path (f16 MMA, f16 tol).
    # Compared by max-relative error (max|diff|/max|ref|) — the standard f16-GEMM metric;
    # per-element rtol is meaningless on a GEMM output's wide dynamic range, but max-relative
    # still catches catastrophic bugs (e.g. the auto-grid mis-size gave rel~1.0).
    for T, rtol in [(1, 1e-3), (2, 5e-3)]:
        x = rng.standard_normal((T, H)).astype(np.float32)
        logits = rng.standard_normal((T, E)).astype(np.float32)

        # --- the op under test (handler + kernel chain) ---
        y = _gguf_moe_routed_handler(
            _buf(x, float32), _buf(logits, float32),
            _buf(gu.reshape(-1), uint8), _buf(down.reshape(-1), uint8),
            E, TOP_K, I,
        )
        materialize_many([y])
        gpu_sync()
        got = torch.from_numpy(y.numpy.copy()).reshape(T, H)

        # --- reference: torch router + existing per-expert kernels, weighted sum ---
        lt = torch.from_numpy(logits)
        topval, topidx = torch.topk(lt, TOP_K, dim=-1)
        w = torch.softmax(topval, dim=-1)
        ref = torch.zeros(T, H)
        for t in range(T):
            for s in range(TOP_K):
                e = int(topidx[t, s])
                h_e = _alloc_aligned((1, I), float32)
                dot_q4_k_silu_v2[(I,)](
                    _buf(x[t:t + 1], float32),
                    _buf(np.ascontiguousarray(gu[e, 0:I]), uint8),
                    _buf(np.ascontiguousarray(gu[e, I:2 * I]), uint8),
                    h_e, NSG=1, NR0=1,
                )
                y_e = _alloc_aligned((1, H), float32)
                dot_q6_k_v2[(H,)](h_e, _buf(np.ascontiguousarray(down[e]), uint8), y_e, GROUP_SIZE=256)
                materialize_many([y_e])
                gpu_sync()
                ref[t] += float(w[t, s]) * torch.from_numpy(y_e.numpy.copy()).reshape(H)

        max_rel = (got - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
        assert max_rel < rtol, f"T={T} max-relative error {max_rel:.3e} (>= {rtol})"


def test_moe_router_topk_matches_hf():
    T, E, TOP_K = 3, 256, 8
    rng = np.random.default_rng(2)
    logits = rng.standard_normal((T, E)).astype(np.float32)

    logits_buf = _buf(logits, float32)
    idx_out = _alloc_aligned((T * TOP_K,), int32)
    w_out = _alloc_aligned((T * TOP_K,), float32)
    active_out = _alloc_aligned((1,), int32)
    moe_router_topk[(T,)](
        logits_buf, idx_out, w_out, active_out, NUM_EXPERTS=E, TOP_K=TOP_K, BLOCK=256,
    )
    materialize_many([idx_out, w_out, active_out])
    gpu_sync()
    assert int(active_out.numpy[0]) == T
    got_idx = torch.from_numpy(idx_out.numpy.copy()).reshape(T, TOP_K)
    got_w = torch.from_numpy(w_out.numpy.copy()).reshape(T, TOP_K)

    # HF reference: softmax over all experts, top-k, renormalize.
    lt = torch.from_numpy(logits)
    probs = F.softmax(lt.float(), dim=-1)
    ref_val, ref_idx = torch.topk(probs, TOP_K, dim=-1)
    ref_w = ref_val / ref_val.sum(dim=-1, keepdim=True)

    assert torch.equal(got_idx, ref_idx.to(torch.int32)), f"idx mismatch:\n{got_idx}\n{ref_idx}"
    assert torch.allclose(got_w, ref_w, atol=1e-6, rtol=1e-5), (
        f"weight max abs diff {(got_w - ref_w).abs().max().item():.3e}"
    )


def test_moe_down_combine_matches_sliced_expert():
    H, I, E, GROUP = 128, 512, 4, 256
    ROW_BYTES = (I // 256) * 210
    rng = np.random.default_rng(1)

    down_q6 = rng.integers(0, 256, size=(E, H, ROW_BYTES), dtype=np.uint8)
    # The per-256-block f16 scale `d` lives in bytes [208:210] of each 210-byte
    # block; random bytes there can decode to inf/nan. Pin it to a sane value so
    # the dequant stays finite (we're testing the gather/reduce, not numerics).
    n_groups = I // 256
    dview = down_q6.reshape(E, H, n_groups, 210)
    d_bytes = np.array([0.02], dtype=np.float16).view(np.uint8)
    dview[..., 208] = d_bytes[0]
    dview[..., 209] = d_bytes[1]
    h = rng.standard_normal((1, I)).astype(np.float32)

    e = 3  # route the single token to expert 3, weight 1.0
    routing = np.array([e], dtype=np.int32)
    weights = np.array([1.0], dtype=np.float32)

    # --- gathered MoE down kernel ---
    h_buf = _buf(h.reshape(-1), float32)
    down_buf = _buf(down_q6.reshape(-1), uint8)
    routing_buf = _buf(routing, int32)
    weights_buf = _buf(weights, float32)
    y_out = _alloc_aligned((1 * H,), float32)
    moe_down_combine[(1, H)](
        h_buf, down_buf, routing_buf, weights_buf, y_out,
        HID=H, MOE_INTER=I, TOP_K=1,
    )

    # --- reference: dot_q6_k_v2 on expert e's down slice ---
    down_e = np.ascontiguousarray(down_q6[e])       # (H, ROW_BYTES)
    h_buf2 = _buf(h, float32)                        # (1, I)
    down_e_buf = _buf(down_e, uint8)                 # (H, ROW_BYTES)
    y_ref = _alloc_aligned((1, H), float32)
    dot_q6_k_v2[(H,)](h_buf2, down_e_buf, y_ref, GROUP_SIZE=GROUP)

    materialize_many([y_out, y_ref])
    gpu_sync()

    got = torch.from_numpy(y_out.numpy.copy()).reshape(H)
    ref = torch.from_numpy(y_ref.numpy.copy()).reshape(H)
    assert torch.allclose(got, ref, atol=1e-5, rtol=1e-4), (
        f"max abs diff {(got - ref).abs().max().item():.3e}"
    )
