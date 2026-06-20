"""Bit-exact tests for the KV-cache q8_0 quantize/dequant kernels vs a numpy
ggml-q8_0 reference (d = max|x|/127, q = round-half-away(x/d), d==0 -> q=0),
plus the q8 flash-decoding attention kernel vs reference attention over the
dequantized cache."""
import numpy as np

from alloy.std.attention import attention_decode_combine_vector
from alloy.std.kv_quant import (
    attention_decode_vector_split_q8,
    kv_dequant_q8_range,
    kv_quantize_q8_range,
)
from alloy._compiler.dtypes import float16, float32, int8, int32
from alloy._dispatch.buf_utils import _alloc_aligned
from alloy._runtime._metal_ext import gpu_sync
from alloy._runtime.alloy_buffer import materialize_many


def _quantize_ref(kv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ggml q8_0 over the last axis in 32-elem blocks. kv: (H, S, D) f16."""
    h, s, d = kv.shape
    x = kv.astype(np.float32).reshape(h, s, d // 32, 32)
    m = np.abs(x).max(axis=-1)
    scales = (m / 127.0).astype(np.float16)
    with np.errstate(divide="ignore"):
        inv = np.where(m > 0, 127.0 / m, 0.0)
    q = x * inv[..., None]
    # np.round is half-even; ggml/MSL round() is half away from zero.
    codes = (np.sign(q) * np.floor(np.abs(q) + 0.5)).astype(np.int8)
    return codes.reshape(h, s, d), scales.reshape(h, s, d // 32)



def _assert_codes(got, ref, kv):
    """Codes must match the numpy reference exactly EXCEPT at exact half-step
    ties (q_exact = n + 0.5), where Metal fast-math (rcp/fma contraction) may
    legitimately round to the other side."""
    h, s, d = kv.shape
    x = kv.astype(np.float32).reshape(h, s, d // 32, 32)
    m = np.abs(x).max(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        qx = np.where(m > 0, x * (127.0 / m), 0.0).reshape(h, s, d)
    diff = got.astype(np.int32) - ref.astype(np.int32)
    assert (np.abs(diff) <= 1).all(), f"code diff > 1: max {np.abs(diff).max()}"
    near_tie = np.abs(np.abs(qx) % 1.0 - 0.5) < 1e-3
    bad = (diff != 0) & ~near_tie
    assert not bad.any(), f"{bad.sum()} non-tie code mismatches"


def _ring0():
    """Sentinel last-real buffer (negative = unbounded) for non-ring / legacy calls."""
    b = _alloc_aligned((1,), int32)
    b.numpy[:] = -1
    return b

def _bufs(kv: np.ndarray, s_max: int):
    h, _, d = kv.shape
    kv_buf = _alloc_aligned((h * s_max * d,), float16)
    kv_buf.numpy[:] = 0
    kv_buf.numpy.reshape(h, s_max, d)[:, : kv.shape[1]] = kv
    codes_buf = _alloc_aligned((h * s_max * d,), int8)
    scales_buf = _alloc_aligned((h * s_max * (d // 32),), float16)
    start_buf = _alloc_aligned((1,), int32)
    return kv_buf, codes_buf, scales_buf, start_buf


def test_kv_quantize_q8_matches_numpy():
    H, S, D, S_MAX = 4, 17, 128, 64
    rng = np.random.default_rng(0)
    kv = (rng.standard_normal((H, S, D)) * 3).astype(np.float16)
    kv[0, 0, :32] = 0.0  # an all-zero block: d=0 -> q=0 path

    codes_ref, scales_ref = _quantize_ref(kv)

    kv_buf, codes_buf, scales_buf, start_buf = _bufs(kv, S_MAX)
    start_buf.numpy[:] = 0
    kv_quantize_q8_range[(H, S)](kv_buf, start_buf, _ring0(), codes_buf, scales_buf,
                                 S_MAX=S_MAX, HEAD_DIM=D,
                                 SRC_HEAD_STRIDE=S_MAX * D, SRC_SEQ_STRIDE=D)
    materialize_many([codes_buf, scales_buf])
    gpu_sync()

    codes_got = codes_buf.numpy.copy().reshape(H, S_MAX, D)[:, :S]
    scales_got = scales_buf.numpy.copy().reshape(H, S_MAX, D // 32)[:, :S]
    assert np.array_equal(scales_got, scales_ref), (
        f"{np.count_nonzero(scales_got != scales_ref)}/{scales_ref.size} scales differ"
    )
    _assert_codes(codes_got, codes_ref, kv)


def test_kv_quantize_q8_runtime_start_offset():
    """START is a runtime buffer: quantizing rows [START, START+n) must touch
    exactly those rows — the plan-replay contract (no recompile per offset)."""
    H, S_MAX, D, START, N = 2, 64, 64, 21, 7
    rng = np.random.default_rng(1)
    kv = (rng.standard_normal((H, S_MAX, D)) * 2).astype(np.float16)

    kv_buf, codes_buf, scales_buf, start_buf = _bufs(kv, S_MAX)
    kv_buf.numpy.reshape(H, S_MAX, D)[:] = kv
    codes_buf.numpy[:] = 99  # sentinel outside the written range
    start_buf.numpy[:] = START
    kv_quantize_q8_range[(H, N)](kv_buf, start_buf, _ring0(), codes_buf, scales_buf,
                                 S_MAX=S_MAX, HEAD_DIM=D, SRC_OFFSET=START * D,
                                 SRC_HEAD_STRIDE=S_MAX * D, SRC_SEQ_STRIDE=D)
    materialize_many([codes_buf])
    gpu_sync()

    codes_got = codes_buf.numpy.copy().reshape(H, S_MAX, D)
    codes_ref, _ = _quantize_ref(kv[:, START:START + N])
    _assert_codes(codes_got[:, START:START + N], codes_ref, kv[:, START:START + N])
    assert (codes_got[:, :START] == 99).all(), "wrote below START"
    assert (codes_got[:, START + N:] == 99).all(), "wrote past START+N"


def test_kv_q8_round_trip_error_bound():
    """quantize -> dequant reconstruction error bounded by scale/2 per element,
    and dequant(quantize(x)) re-quantizes to identical codes (idempotence)."""
    H, S, D, S_MAX = 2, 33, 256, 64
    rng = np.random.default_rng(2)
    kv = (rng.standard_normal((H, S, D)) * 5).astype(np.float16)

    kv_buf, codes_buf, scales_buf, start_buf = _bufs(kv, S_MAX)
    start_buf.numpy[:] = 0
    out_buf = _alloc_aligned((H * S_MAX * D,), float16)
    kv_quantize_q8_range[(H, S)](kv_buf, start_buf, _ring0(), codes_buf, scales_buf,
                                 S_MAX=S_MAX, HEAD_DIM=D,
                                 SRC_HEAD_STRIDE=S_MAX * D, SRC_SEQ_STRIDE=D)
    end_buf = _alloc_aligned((1,), int32)
    end_buf.numpy[:] = S
    kv_dequant_q8_range[(H, (S + 63) // 64)](codes_buf, scales_buf, end_buf, out_buf,
                                             S_MAX=S_MAX, HEAD_DIM=D)
    materialize_many([codes_buf, scales_buf, out_buf])
    gpu_sync()

    got = out_buf.numpy.copy().reshape(H, S_MAX, D)[:, :S].astype(np.float32)
    ref = kv.astype(np.float32)
    scales = scales_buf.numpy.copy().reshape(H, S_MAX, D // 32)[:, :S].astype(np.float32)
    # err <= d*(1/2 + 127*2^-11) + output-fp16 rounding: the half quantization
    # step, plus the fp16 rounding of the stored scale amplified by |q| <= 127,
    # plus ulp(q*d)/2 on the fp16 result.
    bound = np.repeat(scales, 32, axis=-1) * (0.5 + 127 * 2.0**-11) + np.abs(ref) * 2.0**-10
    err = np.abs(got - ref)
    assert (err <= bound).all(), f"max err {err.max()} exceeds q8 bound"

    # Idempotence: re-quantize the dequantized values -> identical codes/scales.
    codes2_ref, scales2_ref = _quantize_ref(got.astype(np.float16))
    codes1 = codes_buf.numpy.copy().reshape(H, S_MAX, D)[:, :S]
    scales1 = scales_buf.numpy.copy().reshape(H, S_MAX, D // 32)[:, :S]
    assert np.array_equal(codes2_ref, codes1)
    assert np.array_equal(scales2_ref, scales1)


def _u32_view(buf):
    from alloy._compiler.dtypes import uint32
    from alloy._runtime.alloy_buffer import AlloyBuffer
    v = AlloyBuffer(buf._parent_handle, buf._offset, (buf.size // 4,), (4,), uint32,
                    raw_ptr=buf._raw_ptr, total_nbytes=buf.metal_nbytes)
    buf._view_of(v)
    return v


def _decode_attention_q8(kv_k, kv_v, q, pos, kv_group, splits, sliding_window=0,
                         codes_u32=False):
    """Quantize the (ring or linear) cache, run the q8 decode kernel + combine,
    return (out (heads, D), codes/scales arrays for the reference)."""
    H, S_MAX, D = kv_k.shape
    heads = H * kv_group
    BH = heads

    bufs = {}
    for name, kv in (("k", kv_k), ("v", kv_v)):
        kv_buf = _alloc_aligned((H * S_MAX * D,), float16)
        kv_buf.numpy[:] = kv.reshape(-1)
        codes = _alloc_aligned((H * S_MAX * D,), int8)
        scales = _alloc_aligned((H * S_MAX * (D // 32),), float16)
        start = _alloc_aligned((1,), int32)
        start.numpy[:] = 0
        kv_quantize_q8_range[(H, S_MAX)](kv_buf, start, _ring0(), codes, scales,
                                         S_MAX=S_MAX, HEAD_DIM=D,
                                         SRC_HEAD_STRIDE=S_MAX * D, SRC_SEQ_STRIDE=D)
        bufs[name] = (codes, scales)

    q_buf = _alloc_aligned((heads * D,), float32)
    q_buf.numpy[:] = q.reshape(-1)
    pos_buf = _alloc_aligned((1,), int32)
    pos_buf.numpy[:] = pos
    partial_o = _alloc_aligned((BH * splits * D,), float32)
    partial_lse = _alloc_aligned((BH * splits,), float32)
    out_buf = _alloc_aligned((BH * D,), float32)

    dummy_new = _alloc_aligned((H * D,), float16)  # unused: WRITE_KV=0 (read-only)
    kc_u32 = _u32_view(bufs["k"][0]) if codes_u32 else bufs["k"][0]
    vc_u32 = _u32_view(bufs["v"][0]) if codes_u32 else bufs["v"][0]
    attention_decode_vector_split_q8[(BH, splits)](
        q_buf, dummy_new, dummy_new, pos_buf,
        bufs["k"][0], bufs["k"][1], bufs["v"][0], bufs["v"][1],
        kc_u32, vc_u32,
        partial_o, partial_lse,
        BH=BH, HEADS_PER_BATCH=heads, HEAD_DIM=D,
        Q_OFFSET=0, Q_BATCH_STRIDE=heads * D, Q_HEAD_STRIDE=D,
        NK_HEAD_STRIDE=D, NV_HEAD_STRIDE=D, WRITE_KV=0,
        KC_HEAD_STRIDE=S_MAX * D, KC_SEQ_STRIDE=D,
        KS_HEAD_STRIDE=S_MAX * (D // 32), KS_SEQ_STRIDE=D // 32,
        VC_HEAD_STRIDE=S_MAX * D, VC_SEQ_STRIDE=D,
        VS_HEAD_STRIDE=S_MAX * (D // 32), VS_SEQ_STRIDE=D // 32,
        KV_GROUP=kv_group, SPLITS=splits, SLIDING_WINDOW=sliding_window,
        CUSTOM_SCALE=0, CODES_U32=1 if codes_u32 else 0,
    )
    attention_decode_combine_vector[(BH,)](
        partial_o, partial_lse, out_buf,
        BH=BH, HEADS_PER_BATCH=heads, HEAD_DIM=D, SPLITS=splits,
    )
    materialize_many([out_buf] + [b for pair in bufs.values() for b in pair])
    gpu_sync()

    deq = {}
    for name in ("k", "v"):
        codes, scales = bufs[name]
        c = codes.numpy.copy().reshape(H, S_MAX, D).astype(np.float32)
        s = scales.numpy.copy().reshape(H, S_MAX, D // 32).astype(np.float32)
        deq[name] = c * np.repeat(s, 32, axis=-1)
    return out_buf.numpy.copy().reshape(BH, D), deq["k"], deq["v"]


def _ref_attention(deq_k, deq_v, q, pos, kv_group, sliding_window=0):
    """f64 reference over the dequantized (ring or linear) cache."""
    heads, D = q.shape
    lo = max(0, pos + 1 - sliding_window) if sliding_window else 0
    js = np.arange(lo, pos + 1)
    slots = js % sliding_window if sliding_window else js
    out = np.empty((heads, D))
    for h in range(heads):
        kh = deq_k[h // kv_group, slots].astype(np.float64)
        vh = deq_v[h // kv_group, slots].astype(np.float64)
        scores = kh @ q[h].astype(np.float64) / np.sqrt(D)
        p = np.exp(scores - scores.max())
        p /= p.sum()
        out[h] = p @ vh
    return out


def test_attention_decode_q8_matches_reference():
    rng = np.random.default_rng(3)
    cases = ((128, 2, 4, False), (64, 1, 2, False), (256, 4, 8, False),
             (512, 1, 4, False), (512, 1, 4, True))
    for D, kv_group, splits, u32 in cases:
        H, S_MAX, pos = 2, 512, 400
        kv_k = (rng.standard_normal((H, S_MAX, D)) * 2).astype(np.float16)
        kv_v = (rng.standard_normal((H, S_MAX, D)) * 2).astype(np.float16)
        q = rng.standard_normal((H * kv_group, D)).astype(np.float32)

        got, deq_k, deq_v = _decode_attention_q8(kv_k, kv_v, q, pos, kv_group, splits,
                                                 codes_u32=u32)
        ref = _ref_attention(deq_k, deq_v, q, pos, kv_group)
        err = np.abs(got - ref).max()
        assert err < 2e-3, f"D={D} group={kv_group} splits={splits} u32={u32}: max err {err}"


def test_attention_decode_q8_sliding_window():
    rng = np.random.default_rng(4)
    H, SW, D, pos = 1, 128, 128, 400  # window slid well past zero
    # Ring cache: slot j % SW holds token j for the live window only.
    kv_k = np.zeros((H, SW, D), dtype=np.float16)
    kv_v = np.zeros((H, SW, D), dtype=np.float16)
    for j in range(pos + 1 - SW, pos + 1):
        kv_k[:, j % SW] = (rng.standard_normal((H, D)) * 2).astype(np.float16)
        kv_v[:, j % SW] = (rng.standard_normal((H, D)) * 2).astype(np.float16)
    q = rng.standard_normal((2, D)).astype(np.float32)

    got, deq_k, deq_v = _decode_attention_q8(
        kv_k, kv_v, q, pos, kv_group=2, splits=4, sliding_window=SW
    )
    ref = _ref_attention(deq_k, deq_v, q, pos, kv_group=2, sliding_window=SW)
    err = np.abs(got - ref).max()
    assert err < 2e-3, f"sliding-window q8 decode: max err {err}"


def test_kv_quantize_q8_range_token_write():
    """Token-write variant: strided projection-output source, ring slot
    pos % SW, codes/scales bit-match the bulk quantizer's for that row."""

    H, S_MAX, D, SW, POS = 3, 64, 128, 64, 199  # ring slot 199 % 64 = 7
    rng = np.random.default_rng(5)
    # Source laid out (S=1, H, D) with a leading pad to exercise OFFSET+stride.
    src = (rng.standard_normal((H, D + 16)) * 2).astype(np.float16)
    new = src[:, 16:]  # head h at offset 16 + h*(D+16)

    src_buf = _alloc_aligned((H * (D + 16),), float16)
    src_buf.numpy[:] = src.reshape(-1)
    codes_buf = _alloc_aligned((H * S_MAX * D,), int8)
    codes_buf.numpy[:] = 99
    scales_buf = _alloc_aligned((H * S_MAX * (D // 32),), float16)
    pos_buf = _alloc_aligned((1,), int32)
    pos_buf.numpy[:] = POS

    kv_quantize_q8_range[(H, 1)](src_buf, pos_buf, _ring0(), codes_buf, scales_buf,
                                 S_MAX=S_MAX, HEAD_DIM=D,
                                 SRC_OFFSET=16, SRC_HEAD_STRIDE=D + 16,
                                 SLIDING_WINDOW=SW)
    materialize_many([codes_buf, scales_buf])
    gpu_sync()

    codes_ref, scales_ref = _quantize_ref(new[:, None, :])
    slot = POS % SW
    codes_got = codes_buf.numpy.copy().reshape(H, S_MAX, D)
    scales_got = scales_buf.numpy.copy().reshape(H, S_MAX, D // 32)
    _assert_codes(codes_got[:, slot][:, None], codes_ref, new[:, None, :])
    assert np.array_equal(scales_got[:, slot], scales_ref[:, 0])
    assert (np.delete(codes_got, slot, axis=1) == 99).all(), "wrote outside ring slot"


def test_attention_cache_q8_handler_end_to_end():
    """Drive the FX handler directly with AlloyBuffers: token-quantize writes
    (K and V from strided projection outputs) + q8 attention + combine, ordered
    by the lazy dispatcher's dependency groups. Reference: f64 attention over
    the dequantized cache INCLUDING the newly written position."""
    from alloy_torch.ops.kv_quant import _attention_cache_q8_handler

    KV_H, GROUP, D, S_MAX, POS = 2, 2, 128, 512, 300
    heads = KV_H * GROUP
    rng = np.random.default_rng(6)

    hist_k = (rng.standard_normal((KV_H, POS, D)) * 2).astype(np.float16)
    hist_v = (rng.standard_normal((KV_H, POS, D)) * 2).astype(np.float16)
    new_k = (rng.standard_normal((KV_H, D)) * 2).astype(np.float16)
    new_v = (rng.standard_normal((KV_H, D)) * 2).astype(np.float16)
    q = rng.standard_normal((heads, D)).astype(np.float32)

    caches = {}
    prime = []
    for name, hist in (("k", hist_k), ("v", hist_v)):
        kv_buf, codes, scales, start = _bufs(hist, S_MAX)
        start.numpy[:] = 0
        kv_quantize_q8_range[(KV_H, POS)](kv_buf, start, _ring0(), codes, scales,
                                          S_MAX=S_MAX, HEAD_DIM=D,
                                          SRC_HEAD_STRIDE=S_MAX * D,
                                          SRC_SEQ_STRIDE=D)
        prime += [codes, scales]
        caches[name] = (codes.reshape((1, KV_H, S_MAX, D)),
                        scales.reshape((1, KV_H, S_MAX, D // 32)))
    # Materialize the history priming BEFORE the handler runs: lazy-op
    # dependency tracking is per AlloyBuffer object, and the handler reads the
    # caches through its own root views — without this barrier the attention op
    # has no ordering edge against the bulk quantize above. (In production the
    # backend maps each torch storage to ONE AlloyBuffer, so cache
    # readers/writers share an object and the hazard can't occur.)
    materialize_many(prime)
    gpu_sync()

    def _f16_4d(arr, shape):
        b = _alloc_aligned((int(arr.size),), float16)
        b.numpy[:] = arr.reshape(-1)
        return b.reshape(shape)

    new_k_buf = _f16_4d(new_k, (1, KV_H, 1, D))
    new_v_buf = _f16_4d(new_v, (1, KV_H, 1, D))
    q_buf = _alloc_aligned((heads * D,), float32)
    q_buf.numpy[:] = q.reshape(-1)
    pos_buf = _alloc_aligned((1,), int32)
    pos_buf.numpy[:] = POS

    out = _attention_cache_q8_handler(
        q_buf.reshape((1, heads, 1, D)), new_k_buf, new_v_buf, pos_buf,
        caches["k"][0], caches["k"][1], caches["v"][0], caches["v"][1],
    )
    materialize_many([out, caches["k"][0], caches["k"][1],
                      caches["v"][0], caches["v"][1]])
    gpu_sync()
    got = out.numpy.copy().reshape(heads, D)

    deq = {}
    for name in ("k", "v"):
        codes, scales = caches[name]
        c = codes.numpy.copy().reshape(KV_H, S_MAX, D).astype(np.float32)
        s = scales.numpy.copy().reshape(KV_H, S_MAX, D // 32).astype(np.float32)
        deq[name] = c * np.repeat(s, 32, axis=-1)
    # The handler's token kernels must have written position POS.
    codes_k_ref, _ = _quantize_ref(new_k[:, None, :])
    _assert_codes(
        caches["k"][0].numpy.copy().reshape(KV_H, S_MAX, D)[:, POS][:, None],
        codes_k_ref, new_k[:, None, :],
    )

    ref = _ref_attention(deq["k"], deq["v"], q, POS, GROUP)
    err = np.abs(got - ref).max()
    assert err < 2e-3, f"handler end-to-end: max err {err}"


def test_attention_cache_q8_handler_prefill_branch():
    """seq_len > 1: the chunk's K/V quantize durably into the codes caches
    (extern-write registration keeps the ops alive; the test drains and
    materializes them like the backend does at graph output) while the
    attention itself runs on the dequant-to-scratch materialize fallback
    through the stock prefill path. Reference: causal f64 attention over
    [dequantized-q8 history | exact-fp16 chunk]."""
    from alloy_torch.compile_window import compile_window
    from alloy_torch.extern_kv import drain_extern_kv_writes
    from alloy_torch.ops.kv_quant import _attention_cache_q8_handler

    KV_H, GROUP, D, S_MAX, START, CHUNK = 2, 2, 64, 512, 200, 64
    heads = KV_H * GROUP
    rng = np.random.default_rng(7)

    hist_k = (rng.standard_normal((KV_H, START, D)) * 2).astype(np.float16)
    hist_v = (rng.standard_normal((KV_H, START, D)) * 2).astype(np.float16)
    new_k = (rng.standard_normal((KV_H, CHUNK, D)) * 2).astype(np.float16)
    new_v = (rng.standard_normal((KV_H, CHUNK, D)) * 2).astype(np.float16)
    q = rng.standard_normal((heads, CHUNK, D)).astype(np.float32)

    caches = {}
    prime = []
    for name, hist in (("k", hist_k), ("v", hist_v)):
        kv_buf, codes, scales, start = _bufs(hist, S_MAX)
        start.numpy[:] = 0
        kv_quantize_q8_range[(KV_H, START)](kv_buf, start, _ring0(), codes, scales,
                                            S_MAX=S_MAX, HEAD_DIM=D,
                                            SRC_HEAD_STRIDE=S_MAX * D,
                                            SRC_SEQ_STRIDE=D)
        prime += [codes, scales]
        caches[name] = (codes.reshape((1, KV_H, S_MAX, D)),
                        scales.reshape((1, KV_H, S_MAX, D // 32)))
    materialize_many(prime)
    gpu_sync()

    def _f16_4d(arr, shape):
        b = _alloc_aligned((int(arr.size),), float16)
        b.numpy[:] = arr.reshape(-1)
        return b.reshape(shape)

    new_k_buf = _f16_4d(new_k, (1, KV_H, CHUNK, D))
    new_v_buf = _f16_4d(new_v, (1, KV_H, CHUNK, D))
    q_buf = _alloc_aligned((heads * CHUNK * D,), float32)
    q_buf.numpy[:] = q.reshape(-1)
    pos_buf = _alloc_aligned((CHUNK,), int32)
    pos_buf.numpy[:] = np.arange(START, START + CHUNK, dtype=np.int32)

    compile_window.q_start_pos = START
    try:
        out = _attention_cache_q8_handler(
            q_buf.reshape((1, heads, CHUNK, D)), new_k_buf, new_v_buf, pos_buf,
            caches["k"][0], caches["k"][1], caches["v"][0], caches["v"][1],
        )
        extern = drain_extern_kv_writes()
        materialize_many([out] + extern)
        gpu_sync()
    finally:
        compile_window.q_start_pos = 0

    # Durable write: the chunk's codes must be in the cache at [START, START+CHUNK).
    codes_chunk_ref, _ = _quantize_ref(new_k)
    codes_got = caches["k"][0].numpy.copy().reshape(KV_H, S_MAX, D)
    _assert_codes(codes_got[:, START:START + CHUNK], codes_chunk_ref, new_k)

    # Attention vs causal reference: history AND chunk both come from the
    # dequantized codes — the scratch is dequant-filled over [0, START+CHUNK)
    # and the stock path attends read-only (write_kv=False), so prefill reads
    # exactly what decode will read later.
    deq = {}
    for name in ("k", "v"):
        codes, scales = caches[name]
        c = codes.numpy.copy().reshape(KV_H, S_MAX, D)[:, :START + CHUNK].astype(np.float32)
        s = scales.numpy.copy().reshape(KV_H, S_MAX, D // 32)[:, :START + CHUNK].astype(np.float32)
        deq[name] = c * np.repeat(s, 32, axis=-1)
    k_full, v_full = deq["k"], deq["v"]

    got = np.asarray(out.numpy).reshape(out.shape)
    assert got.shape == (1, heads, CHUNK, D), got.shape
    q16 = q.astype(np.float16).astype(np.float64)
    max_err = 0.0
    for h in range(heads):
        kh = k_full[h // GROUP].astype(np.float64)
        vh = v_full[h // GROUP].astype(np.float64)
        for i in range(CHUNK):
            n = START + i + 1
            scores = kh[:n] @ q16[h, i] / np.sqrt(D)
            p = np.exp(scores - scores.max())
            p /= p.sum()
            ref_row = p @ vh[:n]
            max_err = max(max_err, np.abs(got[0, h, i] - ref_row).max())
    assert max_err < 2e-2, f"prefill branch: max err {max_err}"


def test_kv_quantize_q8_ring_end_bound():
    """attention_kv_write's single-writer contract on the quantize path: with
    RING_END set, a padded chunk longer than the window writes only positions
    [end-SW, end) — each ring slot gets exactly one writer, and pad rows past
    the real length never quantize into the ring (the gemma4 <pad> corruption)."""
    H, SW, D = 1, 8, 64
    CHUNK, REAL = 24, 12  # padded chunk 24 rows, real rows 12, window 8
    rng = np.random.default_rng(8)
    src = (rng.standard_normal((H, CHUNK, D)) * 2).astype(np.float16)

    src_buf = _alloc_aligned((H * CHUNK * D,), float16)
    src_buf.numpy[:] = src.reshape(-1)
    codes = _alloc_aligned((H * SW * D,), int8)
    scales = _alloc_aligned((H * SW * (D // 32),), float16)
    start = _alloc_aligned((1,), int32)
    start.numpy[:] = 0
    ring = _alloc_aligned((1,), int32)
    ring.numpy[:] = REAL - 1  # last real row index; start=0 so end = REAL

    kv_quantize_q8_range[(H, CHUNK)](src_buf, start, ring, codes, scales,
                                     S_MAX=SW, HEAD_DIM=D,
                                     SRC_HEAD_STRIDE=CHUNK * D, SRC_SEQ_STRIDE=D,
                                     SLIDING_WINDOW=SW)
    materialize_many([codes, scales])
    gpu_sync()

    got = codes.numpy.copy().reshape(H, SW, D)
    for pos in range(REAL - SW, REAL):  # the surviving window [4, 12)
        ref, _ = _quantize_ref(src[:, pos:pos + 1])
        _assert_codes(got[:, pos % SW][:, None], ref, src[:, pos:pos + 1])
