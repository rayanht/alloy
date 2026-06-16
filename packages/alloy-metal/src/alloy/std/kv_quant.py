"""KV-cache q8_0 quantize/dequantize kernels (encode direction).

The cache layout is the generation cache's: K or V as (KV_H, S_MAX, HEAD_DIM)
fp16 (batch=1), quantized per token into int8 codes (same logical shape) plus
one fp16 scale per 32-element block along head_dim — the same normalized
symmetric layout the Q8_0 weight kernels consume (`quant.py:dot_q8_0`:
int8 payload + fp16 scales, zero_point 0), so attention reads reuse the
existing `_dequant_scale` load descriptor.

The 32-element block is structural, not tunable: one Metal simdgroup (32
lanes) handles exactly one block, so the per-block max-abs is a single
`simd_max` with no shared memory or cross-group sync. ggml q8_0 convention:
scale d = max|x|/127, q = round(x/d) (round half away from zero — MSL
`round()` ≡ C `roundf`), d == 0 ⇒ q = 0.

Both kernels take the token range as (START 1-elem int32 buffer, grid dim 1 =
token count): runtime-positioned like the attention kernels' Q_START_POS, so
compiled plans replay them at any cache offset without recompiling. Grid =
(KV_H, n_tokens), HEAD_DIM threads per program (one simdgroup per block).
"""

import alloy as al


@al.kernel
def kv_quantize_q8_range(SRC, START, LAST_REAL, CODES: al.output, SCALES: al.output,
                         S_MAX: al.constexpr, HEAD_DIM: al.constexpr,
                         SRC_OFFSET: al.constexpr = 0,
                         SRC_HEAD_STRIDE: al.constexpr = 0,
                         SRC_SEQ_STRIDE: al.constexpr = 0,
                         SLIDING_WINDOW: al.constexpr = 0):
    """Quantize grid-dim-1 tokens per head from a strided fp16 source (the
    projection output / chunk rows, indexed chunk-locally by t) into the codes
    cache at logical positions [START, START+grid1) — ring slot
    pos % SLIDING_WINDOW when set. Decode is the grid (KV_H, 1) case; prefill
    chunks use grid (KV_H, seq_len). Dispatched before the attention read,
    ordered by the codes-buffer dependency.

    `LAST_REAL` ((1,) runtime int, sliding-window only) is the chunk's last
    REAL row index (real end = START + LAST_REAL + 1) —
    `attention_kv_write`'s single-writer contract: when set (>= 0) only
    positions [end-SW, end) write, so a PADDED chunk longer than the window
    can't race ring slots or clobber real rows with quantized pad garbage
    (gemma4: 4096-row padded chunk into a 512-slot ring turned whole
    transcripts into <pad>). Negative = "no bound"; unmanaged paths (decode's
    single-token writes, handler tests) ride the no-bound default."""
    h = al.program_id(0)
    t = al.program_id(1)
    start = al.load(START)
    pos = start + t
    keep = al.cast(1, al.int32)
    if SLIDING_WINDOW > 0:
        last_real = al.cast(al.load(LAST_REAL), al.int32)
        ring_end = start + last_real + 1
        below = al.where(pos + SLIDING_WINDOW < ring_end, 1, 0)
        beyond = al.where(pos >= ring_end, 1, 0)
        keep = al.where(last_real >= 0, 1 - below - beyond, 1)
        write_pos = pos % SLIDING_WINDOW
    else:
        write_pos = pos
    d = al.arange(0, HEAD_DIM)

    x = al.cast(
        al.load(SRC + SRC_OFFSET + h * SRC_HEAD_STRIDE + t * SRC_SEQ_STRIDE + d),
        al.float32,
    )
    m = al.simd_reduce(al.abs(x), op="max")        # per-32-block max|x|
    inv = al.where(m > 0.0, 127.0 / m, 0.0)

    row = (h * S_MAX + write_pos) * HEAD_DIM
    al.store(CODES + row + d, al.cast(al.round(x * inv), al.int8), mask=keep > 0)
    srow = (h * S_MAX + write_pos) * (HEAD_DIM // 32)
    al.store(SCALES + srow + d // 32, al.cast(m / 127.0, al.float16),
             mask=((d % 32) == 0) & (keep > 0))


@al.kernel
def kv_dequant_q8_range(CODES, SCALES, END, OUT: al.output,
                        S_MAX: al.constexpr, HEAD_DIM: al.constexpr,
                        END_OFFSET: al.constexpr = 0,
                        SLIDING_WINDOW: al.constexpr = 0,
                        TOKENS_PER_PROG: al.constexpr = 64):
    """Dequantize PHYSICAL cache slots [0, min(END+END_OFFSET, window)) into
    the fp16 scratch (same cache layout) — the materialize fallback for
    prefill reads. Grid dim 1 spans the static slot count; rows
    at or past the runtime end are masked off. END is the chunk's start
    position; END_OFFSET (static chunk length) extends the range over the
    chunk's own just-quantized rows, so the scratch is the SINGLE source the
    read-only prefill attention needs — and prefill attends exactly the codes
    decode will read later. Ring caches are slot-to-slot: physical slot j
    holds logical token (j + k*window); the prefill attention's own ring
    addressing resolves logical positions."""
    h = al.program_id(0)
    base = al.program_id(1) * TOKENS_PER_PROG
    end = al.load(END) + END_OFFSET
    if SLIDING_WINDOW > 0:
        phys_end = al.minimum(end, al.cast(SLIDING_WINDOW, al.int32))
    else:
        phys_end = end
    d = al.arange(0, HEAD_DIM)

    # TOKENS_PER_PROG rows per program (GPU loop, not unrolled): a grid of
    # one program per slot at a 262k-native cache was ~262k mostly-masked
    # threadgroup launches per layer per chunk — measured +1.4s/request on
    # qwen3.5:4b prefill. Rows at/past the runtime end mask off their store;
    # their loads clamp to the last slot (in-bounds garbage, never written).
    for _t in range(0, TOKENS_PER_PROG, 1):
        tok = base + _t
        safe = al.minimum(al.cast(tok, al.int32), al.cast(S_MAX - 1, al.int32))
        row = (h * S_MAX + safe) * HEAD_DIM
        srow = (h * S_MAX + safe) * (HEAD_DIM // 32)
        q = al.cast(al.load(CODES + row + d), al.float32)
        scale = al.cast(al.load(SCALES + srow + d // 32), al.float32)
        al.store(OUT + row + d, al.cast(q * scale, al.float16), mask=tok < phys_end)


@al.tunable(UNROLL=[8, 16, 32])
@al.kernel
def attention_decode_vector_split_q8(
    Q,
    new_K,
    new_V,
    cache_pos_buf,
    K_codes,
    K_scales,
    V_codes,
    V_scales,
    K_codes_u32,
    V_codes_u32,
    partial_O: al.output,
    partial_lse: al.output,
    BH: al.constexpr = 1,
    HEADS_PER_BATCH: al.constexpr = 1,
    HEAD_DIM: al.constexpr = 128,
    Q_OFFSET: al.constexpr = 0,
    Q_BATCH_STRIDE: al.constexpr = 0,
    Q_HEAD_STRIDE: al.constexpr = 0,
    NK_OFFSET: al.constexpr = 0,
    NK_HEAD_STRIDE: al.constexpr = 0,
    NV_OFFSET: al.constexpr = 0,
    NV_HEAD_STRIDE: al.constexpr = 0,
    KC_HEAD_STRIDE: al.constexpr = 0,
    KC_SEQ_STRIDE: al.constexpr = 128,
    KS_HEAD_STRIDE: al.constexpr = 0,
    KS_SEQ_STRIDE: al.constexpr = 4,
    VC_HEAD_STRIDE: al.constexpr = 0,
    VC_SEQ_STRIDE: al.constexpr = 128,
    VS_HEAD_STRIDE: al.constexpr = 0,
    VS_SEQ_STRIDE: al.constexpr = 4,
    KV_GROUP: al.constexpr = 1,
    SPLITS: al.constexpr = 1,
    SLIDING_WINDOW: al.constexpr = 0,
    CUSTOM_SCALE: al.constexpr = 0,
    WRITE_KV: al.constexpr = 1,
    CODES_U32: al.constexpr = 0,
    UNROLL: al.constexpr = 16,
):
    """q8_0 read-only twin of `attention.attention_decode_vector_split`.

    Same M=1 flash-decoding structure (32-thread TG, lane owns
    PER_LANE=HEAD_DIM/32 consecutive dims, UNROLL-pipelined simd_reduce
    scores, exp2-frame block softmax, identical (partial_O, partial_lse)
    contract so the existing combine kernels are reused unchanged) — but K/V
    come from int8 codes + per-32-block fp16 scales. A lane's PER_LANE dims
    always sit inside ONE 32-block (PER_LANE divides 32), so dequant is one
    scale load per (lane, position) factored out of the lane's partial dot.

    WRITE_KV=1 fuses the new token's quantize-write, mirroring the fp16
    kernel: lanes 0..D/32-1 each scan one 32-elem block of new_K/new_V
    (scalar; runs once per dispatch, negligible vs the K-loop), write codes +
    scale, then the loop reads position pos back. Every split writes identical
    bytes — the same redundant-write-then-read safety the fp16 WRITE_KV path
    uses. Standalone `kv_quantize_q8_range` dispatches cost ~2 extra
    dispatches/layer/step in the chunked decode plan — the measured reason
    decode tpot trailed fp16 (~10.0 vs 8.05 ms @16k) while the kernel itself
    microbenched at near-parity. WRITE_KV=0 is the gemma4 KV-shared read.

    Codes load as char4 + dot4 (float4(char4) is one hardware convert) on the
    PER_LANE%4==0 path — instruction-parity with the fp16 kernel; the original
    per-byte bring-up loads were instruction-bound and 0.95x fp16 at depth.
    """
    D = HEAD_DIM
    SCALE = CUSTOM_SCALE if (CUSTOM_SCALE is not None and CUSTOM_SCALE > 0) else 1.0 / (D**0.5)
    SCALE2 = SCALE * 1.4426950408889634  # fold log2(e) into QK -> exp2 frame
    LN2 = 0.6931471805599453
    PER_LANE = D // 32

    bh = al.program_id(0)
    split_idx = al.program_id(1)
    batch = bh // HEADS_PER_BATCH
    head = bh - batch * HEADS_PER_BATCH
    kv_head = head // KV_GROUP if KV_GROUP > 1 else head

    Qh = Q + Q_OFFSET + batch * Q_BATCH_STRIDE + head * Q_HEAD_STRIDE
    KCh = K_codes + kv_head * KC_HEAD_STRIDE
    KSh = K_scales + kv_head * KS_HEAD_STRIDE
    VCh = V_codes + kv_head * VC_HEAD_STRIDE
    VSh = V_scales + kv_head * VS_HEAD_STRIDE

    cache_pos = al.load(cache_pos_buf)
    if SLIDING_WINDOW > 0:
        loop_lo = al.maximum(al.cast(0, al.int32), al.cast(cache_pos + 1 - SLIDING_WINDOW, al.int32))
    else:
        loop_lo = al.cast(0, al.int32)
    N_KV_ACTIVE = al.cast(cache_pos + 1, al.int32)

    lane = al.arange(0, 32)
    lane_off = lane * PER_LANE
    BLK = lane_off // 32  # the (single) 32-block this lane's dims live in

    # Same active-window split partition as the fp16 kernel (empty splits
    # write a -inf lse that combine ignores).
    SPAN = al.cast(N_KV_ACTIVE, al.int32) - loop_lo
    CHUNK = (SPAN + SPLITS - 1) // SPLITS
    split_start = loop_lo + al.cast(split_idx, al.int32) * CHUNK
    split_end = al.minimum(split_start + CHUNK, al.cast(N_KV_ACTIVE, al.int32))

    if WRITE_KV:
        # Fused quantize-write of the new token (per-32-block, ggml q8_0):
        # lane b < D/32 scans block b. The address clamps keep idle lanes
        # (b >= D/32) in bounds; their stores are masked off. Only the split
        # whose K-range contains the new position writes — it is the only one
        # that reads position pos back, and there is no inter-TG visibility
        # guarantee, so each reader must be its own writer (and non-readers
        # need no write at all). All BH×SPLITS TGs writing redundantly
        # measured +255us/step fixed overhead at 36 layers.
        pos_i32 = al.cast(cache_pos, al.int32)
        need = al.where((pos_i32 >= split_start) & (pos_i32 < split_end), 1, 0)
        if SLIDING_WINDOW > 0:
            w_pos = cache_pos % SLIDING_WINDOW
        else:
            w_pos = cache_pos
        blk_lane = al.cast(al.arange(0, 32), al.int32)
        blk_ok = (blk_lane < (HEAD_DIM // 32)) & (need > 0)
        blk_base = al.minimum(blk_lane, al.cast(HEAD_DIM // 32 - 1, al.int32)) * 32
        NKh = new_K + NK_OFFSET + kv_head * NK_HEAD_STRIDE
        NVh = new_V + NV_OFFSET + kv_head * NV_HEAD_STRIDE
        kmax = 0.0
        vmax = 0.0
        for _e in al.unroll(range(32)):
            kmax = al.maximum(kmax, al.abs(al.cast(al.load(NKh + blk_base + _e), al.float32)))
            vmax = al.maximum(vmax, al.abs(al.cast(al.load(NVh + blk_base + _e), al.float32)))
        kinv = al.where(kmax > 0.0, 127.0 / kmax, 0.0)
        vinv = al.where(vmax > 0.0, 127.0 / vmax, 0.0)
        kc_row = w_pos * KC_SEQ_STRIDE
        vc_row = w_pos * VC_SEQ_STRIDE
        for _e in al.unroll(range(32)):
            kq = al.round(al.cast(al.load(NKh + blk_base + _e), al.float32) * kinv)
            vq = al.round(al.cast(al.load(NVh + blk_base + _e), al.float32) * vinv)
            al.store(KCh + kc_row + blk_base + _e, al.cast(kq, al.int8), mask=blk_ok)
            al.store(VCh + vc_row + blk_base + _e, al.cast(vq, al.int8), mask=blk_ok)
        al.store(KSh + w_pos * KS_SEQ_STRIDE + blk_lane, al.cast(kmax / 127.0, al.float16), mask=blk_ok)
        al.store(VSh + w_pos * VS_SEQ_STRIDE + blk_lane, al.cast(vmax / 127.0, al.float16), mask=blk_ok)

    o = [0.0] * PER_LANE
    m = -1e30
    l = 0.0

    T_MAIN = split_start + ((split_end - split_start) // UNROLL) * UNROLL
    if PER_LANE % 4 == 0:
        # ---- vec4 path (D 128/256/512): instruction-parity with the fp16
        # kernel's hot loop. One char4 load + one dot4 per 4 codes (dot4 wraps
        # operands in float4() — float4(char4) is a single hardware convert),
        # plus ONE scale fetch/mul per (lane, position): the per-byte bring-up
        # loads made the kernel instruction-bound and 0.95x fp16 at depth.
        NVEC = PER_LANE // 4
        q = [al.load4_vec(Qh + lane_off + 4 * _c) for _c in range(NVEC)]
        # CODES_U32 (PER_LANE==16, D=512): a lane's 16 codes per position are
        # exactly one uint4 — ONE 16-byte load + 4 as_char4 reinterprets vs
        # four char4 loads. The kernel is load-issue-bound; this quarters the
        # K/V code load instructions. Strides derive from the int8 units.
        KU32h = K_codes_u32 + kv_head * (KC_HEAD_STRIDE // 4)
        VU32h = V_codes_u32 + kv_head * (VC_HEAD_STRIDE // 4)
        for jb in range(split_start, T_MAIN, UNROLL):
            sc = [0.0] * UNROLL
            for _t in al.unroll(range(UNROLL)):
                if SLIDING_WINDOW > 0:
                    jslot = (jb + _t) % SLIDING_WINDOW
                else:
                    jslot = jb + _t
                ks = al.cast(al.load(KSh + jslot * KS_SEQ_STRIDE + BLK), al.float32)
                partial = 0.0
                if CODES_U32:
                    w = al.load4_vec(KU32h + jslot * (KC_SEQ_STRIDE // 4) + lane * 4)
                    for _c in al.unroll(range(4)):
                        partial = partial + al.dot4(q[_c], al.as_char4(w, _c))
                else:
                    for _c in al.unroll(range(NVEC)):
                        partial = partial + al.dot4(
                            q[_c], al.load4_vec(KCh + jslot * KC_SEQ_STRIDE + lane_off + 4 * _c)
                        )
                sc[_t] = al.simd_reduce(partial * ks) * SCALE2
            block_max = m
            for _t in al.unroll(range(UNROLL)):
                block_max = al.maximum(block_max, sc[_t])
            rescale = al.exp2(m - block_max)
            l = l * rescale
            for _i in al.unroll(range(PER_LANE)):
                o[_i] = o[_i] * rescale
            for _t in al.unroll(range(UNROLL)):
                if SLIDING_WINDOW > 0:
                    jslot = (jb + _t) % SLIDING_WINDOW
                else:
                    jslot = jb + _t
                p = al.exp2(sc[_t] - block_max)
                l = l + p
                vs = al.cast(al.load(VSh + jslot * VS_SEQ_STRIDE + BLK), al.float32)
                pvs = p * vs
                if CODES_U32:
                    wv = al.load4_vec(VU32h + jslot * (VC_SEQ_STRIDE // 4) + lane * 4)
                    for _c in al.unroll(range(4)):
                        cv = al.as_char4(wv, _c)
                        for _k in al.unroll(range(4)):
                            o[4 * _c + _k] = o[4 * _c + _k] + pvs * al.unpack4(cv, _k)
                else:
                    for _c in al.unroll(range(NVEC)):
                        v = al.load4_vec(VCh + jslot * VC_SEQ_STRIDE + lane_off + 4 * _c)
                        for _k in al.unroll(range(4)):
                            o[4 * _c + _k] = o[4 * _c + _k] + pvs * al.unpack4(v, _k)
            m = block_max
        for j in range(T_MAIN, split_end, 1):
            if SLIDING_WINDOW > 0:
                jslot = j % SLIDING_WINDOW
            else:
                jslot = j
            ks = al.cast(al.load(KSh + jslot * KS_SEQ_STRIDE + BLK), al.float32)
            partial = 0.0
            if CODES_U32:
                w = al.load4_vec(KU32h + jslot * (KC_SEQ_STRIDE // 4) + lane * 4)
                for _c in al.unroll(range(4)):
                    partial = partial + al.dot4(q[_c], al.as_char4(w, _c))
            else:
                for _c in al.unroll(range(NVEC)):
                    partial = partial + al.dot4(
                        q[_c], al.load4_vec(KCh + jslot * KC_SEQ_STRIDE + lane_off + 4 * _c)
                    )
            score = al.simd_reduce(partial * ks) * SCALE2
            block_max = al.maximum(m, score)
            rescale = al.exp2(m - block_max)
            p = al.exp2(score - block_max)
            l = l * rescale + p
            vs = al.cast(al.load(VSh + jslot * VS_SEQ_STRIDE + BLK), al.float32)
            pvs = p * vs
            if CODES_U32:
                wv = al.load4_vec(VU32h + jslot * (VC_SEQ_STRIDE // 4) + lane * 4)
                for _c in al.unroll(range(4)):
                    cv = al.as_char4(wv, _c)
                    for _k in al.unroll(range(4)):
                        o[4 * _c + _k] = o[4 * _c + _k] * rescale + pvs * al.unpack4(cv, _k)
            else:
                for _c in al.unroll(range(NVEC)):
                    v = al.load4_vec(VCh + jslot * VC_SEQ_STRIDE + lane_off + 4 * _c)
                    for _k in al.unroll(range(4)):
                        o[4 * _c + _k] = o[4 * _c + _k] * rescale + pvs * al.unpack4(v, _k)
            m = block_max
    else:
        # ---- scalar path (PER_LANE=2, D=64 — llama3.2:1b) ----
        q = [al.cast(al.load(Qh + lane_off + _i), al.float32) for _i in range(PER_LANE)]
        for jb in range(split_start, T_MAIN, UNROLL):
            sc = [0.0] * UNROLL
            for _t in al.unroll(range(UNROLL)):
                if SLIDING_WINDOW > 0:
                    jslot = (jb + _t) % SLIDING_WINDOW
                else:
                    jslot = jb + _t
                ks = al.cast(al.load(KSh + jslot * KS_SEQ_STRIDE + BLK), al.float32)
                partial = 0.0
                for _i in al.unroll(range(PER_LANE)):
                    partial = partial + q[_i] * al.cast(
                        al.load(KCh + jslot * KC_SEQ_STRIDE + lane_off + _i), al.float32
                    )
                sc[_t] = al.simd_reduce(partial * ks) * SCALE2
            block_max = m
            for _t in al.unroll(range(UNROLL)):
                block_max = al.maximum(block_max, sc[_t])
            rescale = al.exp2(m - block_max)
            l = l * rescale
            for _i in al.unroll(range(PER_LANE)):
                o[_i] = o[_i] * rescale
            for _t in al.unroll(range(UNROLL)):
                if SLIDING_WINDOW > 0:
                    jslot = (jb + _t) % SLIDING_WINDOW
                else:
                    jslot = jb + _t
                p = al.exp2(sc[_t] - block_max)
                l = l + p
                vs = al.cast(al.load(VSh + jslot * VS_SEQ_STRIDE + BLK), al.float32)
                pvs = p * vs
                for _i in al.unroll(range(PER_LANE)):
                    o[_i] = o[_i] + pvs * al.cast(
                        al.load(VCh + jslot * VC_SEQ_STRIDE + lane_off + _i), al.float32
                    )
            m = block_max
        for j in range(T_MAIN, split_end, 1):
            if SLIDING_WINDOW > 0:
                jslot = j % SLIDING_WINDOW
            else:
                jslot = j
            ks = al.cast(al.load(KSh + jslot * KS_SEQ_STRIDE + BLK), al.float32)
            partial = 0.0
            for _i in al.unroll(range(PER_LANE)):
                partial = partial + q[_i] * al.cast(
                    al.load(KCh + jslot * KC_SEQ_STRIDE + lane_off + _i), al.float32
                )
            score = al.simd_reduce(partial * ks) * SCALE2
            block_max = al.maximum(m, score)
            rescale = al.exp2(m - block_max)
            p = al.exp2(score - block_max)
            l = l * rescale + p
            vs = al.cast(al.load(VSh + jslot * VS_SEQ_STRIDE + BLK), al.float32)
            pvs = p * vs
            for _i in al.unroll(range(PER_LANE)):
                o[_i] = o[_i] * rescale + pvs * al.cast(
                    al.load(VCh + jslot * VC_SEQ_STRIDE + lane_off + _i), al.float32
                )
            m = block_max

    l_safe = al.maximum(l, 1e-30)
    POh = partial_O + (bh * SPLITS + split_idx) * D
    for _i in al.unroll(range(PER_LANE)):
        al.store(POh + lane_off + _i, o[_i] / l_safe, mask=bh < BH)
    PLseh = partial_lse + (bh * SPLITS + split_idx)
    al.store(PLseh, m * LN2 + al.log(al.maximum(l, 1e-30)), mask=(bh < BH) & (lane < 1))
