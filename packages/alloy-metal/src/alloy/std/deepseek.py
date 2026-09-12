"""DeepSeek-V4.1 kernels: sparse latent attention, the lightning indexer and its
exact top-k selection, hyper-connection (mHC) mixing, the Engram gate, the biased
sqrt-softplus router, the ratio-r KV compressor, and the fp8 / fp4 cache rounding.

Activations are fp32; KV caches fp16 (the fp8-E4M3 / fp4-E2M1 rounded values with
their block scales are exactly representable). Every row-parallel kernel is one
program per row, 32 lanes (one simdgroup) unless noted.
"""

import alloy as al

NEG_INF = -1e30
LANES = 32


# --- rounding helpers (traced inline) -------------------------------------------


def _rne(x):
    """Round half to even (MSL `round` is half-away-from-zero)."""
    f = al.floor(x)
    d = x - f
    odd = (f - 2.0 * al.floor(f * 0.5)) > 0.5
    up = al.where(d > 0.5, 1.0, al.where(d < 0.5, 0.0, al.where(odd, 1.0, 0.0)))
    return f + up


def _pow2_ceil(x):
    """2^ceil(log2(x)) for normal positive x, by exponent arithmetic (exact)."""
    bits = al.bitcast(x, al.int32)
    e = ((bits >> 23) & 0xFF) - 127
    mant = bits & 0x7FFFFF
    e_ceil = al.where(mant != 0, e + 1, e)
    return al.bitcast((e_ceil + 127) << 23, al.float32)


def _round_e4m3(v):
    """Round |v| <= 448 to the nearest fp8 E4M3 value (RNE), sign preserved."""
    a = al.abs(v)
    bits = al.bitcast(a, al.int32)
    lsb = (bits >> 20) & 1
    rb = bits + 0x7FFFF + lsb
    rb = rb - (rb & 0xFFFFF)
    normal = al.bitcast(rb, al.float32)
    subn = _rne(a * 512.0) * (1.0 / 512.0)
    r = al.where(a < (1.0 / 64.0), subn, normal)
    r = al.minimum(r, 448.0)
    return al.where(v < 0.0, -r, r)


def _round_e2m1(v):
    """Round |v| <= 6 to the nearest fp4 E2M1 value (RNE on the encoding)."""
    a = al.abs(v)
    q = al.where(a < 2.0, 0.5, al.where(a < 4.0, 1.0, 2.0))
    r = _rne(a / q) * q
    return al.where(v < 0.0, -r, r)


def _store(ptr, value, f16, mask):
    if f16:
        value = al.cast(value, al.float16)
    al.store(ptr, value, mask=mask)


@al.kernel
def ds_block_round(
    X,
    DEP,                      # a view of OUT (or a dummy): orders this write after OUT's previous writer
    ROW0,                     # (1,) int32 output row offset (writes into a cache slice)
    OUT: al.output,
    N: al.constexpr,          # row width
    MODE: al.constexpr,       # 0: fp8 E4M3 / E8M0 scale, 1: fp4 E2M1 / E8M0, 2: fp4 E2M1 / E4M3, 3: no rounding
    BLOCK: al.constexpr,      # elements per scale (32, 32, 16)
    OUT_F16: al.constexpr = 0,
    NUM_THREADS: al.constexpr = 32,
):
    """Quantize-dequantize each BLOCK of a row through the reference cache format
    (`act_quant` / `fp4_act_quant` with inplace=True): per-block absmax scale, clamped
    values rounded to the narrow grid, rescaled. One row per program, 32 lanes
    covering 32 // BLOCK blocks per step."""
    GROUPS = LANES // BLOCK
    row = al.program_id(0)
    out_row = row + al.cast(al.load(ROW0 + 0), al.int32)
    lane = al.arange(0, LANES)
    grp = lane // BLOCK
    _dep = al.load(DEP + 0)
    for g in range(0, N, LANES):
        v = al.cast(al.load(X + row * N + g + lane), al.float32)
        a = al.abs(v)
        amax = a
        for _g in al.unroll(range(GROUPS)):
            amax = al.where(grp == _g, al.max(al.where(grp == _g, a, 0.0)), amax)
        if MODE == 0:
            s = _pow2_ceil(al.maximum(amax, 1e-4) * (1.0 / 448.0))
            y = _round_e4m3(al.clamp(v / s, -448.0, 448.0)) * s
        elif MODE == 1:
            s = _pow2_ceil(al.maximum(amax, 6.0 * 1.1754943508222875e-38) * (1.0 / 6.0))
            y = _round_e2m1(al.clamp(v / s, -6.0, 6.0)) * s
        elif MODE == 2:
            s = _round_e4m3(al.maximum(amax, 6.0 / 512.0) * (1.0 / 6.0))
            y = _round_e2m1(al.clamp(v / s, -6.0, 6.0)) * s
        else:
            y = v
        _store(OUT + out_row * N + g + lane, y, OUT_F16, lane < LANES)


# --- rope -----------------------------------------------------------------------


@al.kernel
def ds_rope_pairs(
    X,
    COS,                      # (max_pos, ROT/2)
    SIN,
    POS,                      # (ROWS,) int32 absolute position per row
    OUT: al.output,
    WIDTH: al.constexpr,      # row width (HEADS * HEAD_DIM)
    HEAD_DIM: al.constexpr,
    ROT: al.constexpr,        # rotated tail dims per head
    INVERSE: al.constexpr = 0,
    OUT_F16: al.constexpr = 0,
    NUM_THREADS: al.constexpr = 32,   # == ROT // 2
):
    """Interleaved-pair complex rotation of the last ROT dims of every head
    (`apply_rotary_emb`): (x0, x1) -> (x0 c - x1 s, x0 s + x1 c); INVERSE conjugates.
    The leading HEAD_DIM - ROT dims copy through. Grid (ROWS, HEADS); ROT/2 lanes."""
    HALF = ROT // 2
    PASS = HEAD_DIM - ROT
    row = al.program_id(0)
    head = al.program_id(1)
    base = row * WIDTH + head * HEAD_DIM
    pos = al.cast(al.load(POS + row), al.int32)
    i = al.arange(0, HALF)
    x0 = al.cast(al.load(X + base + PASS + 2 * i), al.float32)
    x1 = al.cast(al.load(X + base + PASS + 2 * i + 1), al.float32)
    c = al.load(COS + pos * HALF + i)
    s = al.load(SIN + pos * HALF + i)
    if INVERSE:
        s = -s
    _store(OUT + base + PASS + 2 * i, x0 * c - x1 * s, OUT_F16, i < HALF)
    _store(OUT + base + PASS + 2 * i + 1, x0 * s + x1 * c, OUT_F16, i < HALF)
    for p in range(0, PASS, HALF):
        offs = p + i
        v = al.cast(al.load(X + base + offs, mask=offs < PASS, other=0.0), al.float32)
        _store(OUT + base + offs, v, OUT_F16, mask=offs < PASS)


# --- sparse attention --------------------------------------------------------------


@al.kernel
def ds_sparse_attn(
    Q,                        # (ROWS, HEADS*D) f32, rope applied
    WIN_KV,                   # (win_rows, D) f16 window source
    WIN_IDX,                  # (ROWS, WIN_N) int32 rows of WIN_KV, -1 absent
    COMP_KV,                  # (comp_rows, D) f16 compressed cache
    COMP_IDX,                 # (ROWS, COMP_N) int32 rows of COMP_KV, -1 absent
    SINK,                     # (HEADS,) f32
    OUT: al.output,           # (ROWS, HEADS*D) f32
    HEADS: al.constexpr,
    HEAD_DIM: al.constexpr,
    WIN_N: al.constexpr,
    COMP_N: al.constexpr,
    SCALE: al.constexpr,
):
    """Gathered attention where the 512-dim latent is both key and value
    (`sparse_attn`): softmax over the selected rows plus the per-head sink; absent
    rows (-1) contribute exp(-inf) = 0. Grid (ROWS, HEADS); one simdgroup per
    (row, head), each lane owning D/32 dims. Branch-free."""
    D = HEAD_DIM
    PER_LANE = D // LANES
    NVEC = PER_LANE // 4
    row = al.program_id(0)
    head = al.program_id(1)
    lane = al.arange(0, LANES)
    lane_off = lane * PER_LANE
    qbase = Q + row * HEADS * D + head * D
    q = [al.load4_vec(qbase + lane_off + 4 * _c) for _c in range(NVEC)]
    o = [0.0] * PER_LANE
    m = NEG_INF
    l = 0.0
    for j in range(0, WIN_N, 1):
        idx = al.cast(al.load(WIN_IDX + row * WIN_N + j), al.int32)
        present = idx >= 0
        r = al.where(present, idx, 0)
        partial = 0.0
        for _c in al.unroll(range(NVEC)):
            partial = partial + al.dot4(q[_c], al.load4_vec(WIN_KV + r * D + lane_off + 4 * _c))
        score = al.where(present, al.simd_reduce(partial) * SCALE, NEG_INF)
        mn = al.maximum(m, score)
        alpha = al.exp(m - mn)
        p = al.exp(score - mn)
        l = l * alpha + p
        for _c in al.unroll(range(NVEC)):
            v = al.load4_vec(WIN_KV + r * D + lane_off + 4 * _c)
            for _k in al.unroll(range(4)):
                o[4 * _c + _k] = o[4 * _c + _k] * alpha + p * al.unpack4(v, _k)
        m = mn
    for j in range(0, COMP_N, 1):
        idx = al.cast(al.load(COMP_IDX + row * COMP_N + j), al.int32)
        present = idx >= 0
        r = al.where(present, idx, 0)
        partial = 0.0
        for _c in al.unroll(range(NVEC)):
            partial = partial + al.dot4(q[_c], al.load4_vec(COMP_KV + r * D + lane_off + 4 * _c))
        score = al.where(present, al.simd_reduce(partial) * SCALE, NEG_INF)
        mn = al.maximum(m, score)
        alpha = al.exp(m - mn)
        p = al.exp(score - mn)
        l = l * alpha + p
        for _c in al.unroll(range(NVEC)):
            v = al.load4_vec(COMP_KV + r * D + lane_off + 4 * _c)
            for _k in al.unroll(range(4)):
                o[4 * _c + _k] = o[4 * _c + _k] * alpha + p * al.unpack4(v, _k)
        m = mn
    sink = al.load(SINK + head)
    l = l + al.exp(sink - m)
    obase = OUT + row * HEADS * D + head * D
    for _i in al.unroll(range(PER_LANE)):
        al.store(obase + lane_off + _i, o[_i] / l, mask=lane < LANES)


# --- indexer ---------------------------------------------------------------------------


@al.kernel
def ds_index_score(
    Q,                        # (ROWS, HEADS*D) f32 rope'd + fp4-rounded index queries
    K,                        # (n_keys, D) f16 index keys
    W,                        # (ROWS, HEADS) f32 head weights (already scaled)
    VISIBLE,                  # (ROWS,) int32 keys visible to the row (causal)
    MASK,                     # (ROWS, CAP) uint8 candidate mask (HAS_MASK) else unused
    COUNT,                    # (1,) int32 number of keys
    SCORE: al.output,         # (ROWS, CAP) f32, -inf where masked; columns >= COUNT untouched
    HEADS: al.constexpr,
    HEAD_DIM: al.constexpr,
    CAP: al.constexpr,        # row stride (allocated key capacity)
    HAS_MASK: al.constexpr = 0,
    BLOCK: al.constexpr = 64,
):
    """score[s, t] = sum_h relu(q[s,h] . k[t]) * w[s,h] over the causally visible
    (and candidate-masked) keys. Grid (ROWS, ceil(CAP/BLOCK)); one lane per key."""
    D = HEAD_DIM
    N_KEYS = CAP
    row = al.program_id(0)
    t = al.program_id(1) * BLOCK + al.arange(0, BLOCK)
    n = al.cast(al.load(COUNT + 0), al.int32)
    valid = t < n
    ts = al.where(valid, t, 0)
    acc = 0.0
    for h in range(HEADS):
        w = al.load(W + row * HEADS + h)
        dot = 0.0
        for d in range(0, D, 4):
            qv = al.load4_vec(Q + row * HEADS * D + h * D + d)
            kv = al.load4_vec(K + ts * D + d)
            dot = dot + al.dot4(qv, kv)
        acc = acc + al.relu(dot) * w
    vis = al.cast(al.load(VISIBLE + row), al.int32)
    keep = valid & (t < vis)
    if HAS_MASK:
        keep = keep & (al.cast(al.load(MASK + row * N_KEYS + ts), al.int32) != 0)
    al.store(SCORE + row * N_KEYS + t, al.where(keep, acc, NEG_INF), mask=valid)


def _ordered_key(score):
    """Float -> int32 with the same ordering (larger float, larger int)."""
    bits = al.bitcast(score, al.int32)
    return al.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)


@al.kernel
def ds_topk_select(
    SCORE,                    # (ROWS, CAP) f32
    COUNT,                    # (1,) int32 number of valid columns N (<= CAP)
    OUT: al.output,           # (ROWS, K) int32 selected indices ascending; -1 for -inf picks and slots >= min(K, N)
    CAP: al.constexpr,        # row stride
    K: al.constexpr,
    IDX_BITS: al.constexpr,   # ceil(log2(CAP)) + 1
    NEG: al.constexpr = NEG_INF,   # scores at or below this are "absent" -> -1
):
    """Exact top-min(K, N) per row under the total order (score desc, index asc): a
    33-step bisection on the ordered score bits finds the K-th value, an IDX_BITS-step
    bisection on the index resolves ties at that value, then one ordered pass appends
    the selected indices (simdgroup prefix sums keep them sorted). Deterministic."""
    row = al.program_id(0)
    lane = al.arange(0, LANES)
    base = SCORE + row * CAP
    N = al.cast(al.load(COUNT + 0), al.int32)
    k_eff = al.minimum(al.cast(K, al.int32), N)
    # threshold key T: the largest key with count(key >= T) >= k (int64 midpoints —
    # the int32 span overflows)
    lo = al.cast(-2147483648, al.int64)
    hi = al.cast(2147483647, al.int64)
    for _it in range(33):
        mid = al.cast(lo + ((hi - lo) >> 1), al.int32)
        cnt = 0.0
        for j in range(0, N, LANES):
            t = j + lane
            s = al.load(base + t, mask=t < N, other=NEG_INF)
            cnt = cnt + al.where((t < N) & (_ordered_key(s) >= mid), 1.0, 0.0)
        enough = al.simd_reduce(cnt) >= al.cast(k_eff, al.float32)
        lo = al.where(enough, al.cast(mid, al.int64), lo)
        hi = al.where(enough, hi, al.cast(mid, al.int64) - 1)
        hi = al.maximum(hi, lo)
    thresh = al.cast(lo, al.int32)
    gt = 0.0
    for j in range(0, N, LANES):
        t = j + lane
        s = al.load(base + t, mask=t < N, other=NEG_INF)
        gt = gt + al.where((t < N) & (_ordered_key(s) > thresh), 1.0, 0.0)
    need = al.cast(k_eff, al.float32) - al.simd_reduce(gt)
    # smallest index I with count(key == thresh and t <= I) >= need
    ilo = al.cast(0, al.int32)
    ihi = al.maximum(N - 1, al.cast(0, al.int32))
    for _it in range(IDX_BITS):
        imid = al.cast(ilo + ((ihi - ilo) >> 1), al.int32)
        cnt = 0.0
        for j in range(0, N, LANES):
            t = j + lane
            s = al.load(base + t, mask=t < N, other=NEG_INF)
            cnt = cnt + al.where((t < N) & (_ordered_key(s) == thresh) & (t <= imid), 1.0, 0.0)
        enough = al.simd_reduce(cnt) >= need
        ihi = al.where(enough, imid, ihi)
        ilo = al.where(enough, ilo, al.minimum(imid + 1, ihi))
    tie_last = ihi
    written = al.cast(0, al.int32)
    for j in range(0, N, LANES):
        t = j + lane
        s = al.load(base + t, mask=t < N, other=NEG_INF)
        key = _ordered_key(s)
        sel = (t < N) & ((key > thresh) | ((key == thresh) & (t <= tie_last)))
        sel_i = al.where(sel, al.cast(1, al.int32), al.cast(0, al.int32))
        pos = written + al.cast(al.simd_prefix_exclusive_sum(sel_i), al.int32)
        out_val = al.where(s > NEG, t, al.cast(-1, al.int32))
        al.store(OUT + row * K + pos, out_val, mask=sel & (pos < k_eff))
        written = written + al.cast(al.simd_reduce(al.cast(sel_i, al.float32)), al.int32)
    for j in range(0, K, LANES):
        slot = j + lane
        al.store(OUT + row * K + slot, al.cast(-1, al.int32), mask=(slot < K) & (slot >= k_eff))


@al.kernel
def ds_block_max(
    SCORE,                    # (ROWS, CAP) f32
    VISIBLE,                  # (ROWS,) int32
    COUNT,                    # (1,) int32 number of keys N
    OUT: al.output,           # (ROWS, NB_CAP) f32 block max, +inf on the row's last visible block
    CAP: al.constexpr,        # score row stride
    BLOCK_SIZE: al.constexpr,
    NB_CAP: al.constexpr,     # output row stride (block capacity)
    NUM_THREADS: al.constexpr = 32,
):
    """Per-block max of the index scores (hierarchical indexer level one); the block
    holding the row's newest position is pinned to +inf so it is always selected.
    Blocks >= ceil(N / BLOCK_SIZE) are untouched."""
    row = al.program_id(0)
    b = al.program_id(1) * LANES + al.arange(0, LANES)
    n = al.cast(al.load(COUNT + 0), al.int32)
    nb = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    valid = b < nb
    mx = NEG_INF
    for i in range(BLOCK_SIZE):
        t = b * BLOCK_SIZE + i
        v = al.load(SCORE + row * CAP + t, mask=valid & (t < n), other=NEG_INF)
        mx = al.maximum(mx, v)
    vis = al.cast(al.load(VISIBLE + row), al.int32)
    last = (vis - 1) // BLOCK_SIZE
    mx = al.where(b == last, 1e30, mx)
    al.store(OUT + row * NB_CAP + b, mx, mask=valid)


@al.kernel
def ds_block_mask_scatter(
    BLOCKS,                   # (ROWS, B) int32 selected block ids, -1 none
    MASK_DEP,                 # a view of MASK: keeps the zero-fill alive and ordered first
    MASK: al.output,          # (ROWS, CAP) uint8, pre-zeroed
    CAP: al.constexpr,        # mask row stride
    BLOCK_SIZE: al.constexpr,
    B: al.constexpr,
    NUM_THREADS: al.constexpr = 8,    # == BLOCK_SIZE
):
    """Expand selected blocks (-1 = none) to a per-key candidate mask. Grid (ROWS, B)."""
    row = al.program_id(0)
    j = al.program_id(1)
    _dep = al.cast(al.load(MASK_DEP + 0), al.int32)
    blk = al.cast(al.load(BLOCKS + row * B + j), al.int32)
    i = al.arange(0, BLOCK_SIZE)
    t = blk * BLOCK_SIZE + i
    al.store(MASK + row * CAP + t, al.cast(1, al.uint8), mask=(blk >= 0) & (t < CAP))


@al.kernel
def ds_zero_u8(OUT: al.output, N: al.constexpr, BLOCK: al.constexpr = 1024):
    pid = al.program_id(0)
    offs = pid * BLOCK + al.arange(0, BLOCK)
    al.store(OUT + offs, al.cast(0, al.uint8), mask=offs < N)


# --- hyper-connections -------------------------------------------------------------


@al.kernel
def ds_hc_mixes(
    X,                        # (ROWS, HC_D) f32 flattened hc stream
    FN,                       # (MIX, HC_D) f32
    OUT: al.output,           # (ROWS, MIX) f32 = (x @ fn.T) * rsqrt(mean(x^2) + eps)
    HC_D: al.constexpr,
    MIX: al.constexpr,
    EPS: al.constexpr,
    BLOCK: al.constexpr = 256,
):
    """The mHC coefficient projection with the RMS division applied after (`hc_mixes`)."""
    row = al.program_id(0)
    lane = al.arange(0, BLOCK)
    sq = 0.0
    acc = [0.0] * MIX
    for k in range(0, HC_D, BLOCK):
        offs = k + lane
        mask = offs < HC_D
        x = al.load(X + row * HC_D + offs, mask=mask, other=0.0)
        sq = sq + x * x
        for _j in al.unroll(range(MIX)):
            acc[_j] = acc[_j] + x * al.load(FN + _j * HC_D + offs, mask=mask, other=0.0)
    rsq = al.rsqrt(al.sum(sq) / HC_D + EPS)
    for _j in al.unroll(range(MIX)):
        al.store(OUT + row * MIX + _j, al.sum(acc[_j]) * rsq, mask=lane < 1)


def _group_sum(v, group, hc):
    """Per-lane sum over the lanes sharing `group` (values 0..hc-1)."""
    out = v
    for _k in al.unroll(range(hc)):
        out = al.where(group == _k, al.sum(al.where(group == _k, v, 0.0)), out)
    return out


@al.kernel
def ds_hc_sinkhorn(
    MIXES,                    # (ROWS, MIX) f32
    SCALE,                    # (3,) f32
    BASE,                     # (MIX,) f32
    PRE: al.output,           # (ROWS, HC)
    POST: al.output,          # (ROWS, HC)
    COMB: al.output,          # (ROWS, HC, HC)
    HC: al.constexpr,
    ITERS: al.constexpr,
    EPS: al.constexpr,
):
    """`hc_split_sinkhorn`: pre = sigmoid(m*s0+b)+eps, post = 2 sigmoid(m*s1+b),
    comb = softmax(m*s2+b)+eps, an initial column normalization, then ITERS-1 rounds
    of row/column normalization. 32 lanes; lane i < HC*HC owns comb[i // HC, i % HC]."""
    HH = HC * HC
    MIX = 2 * HC + HH
    row = al.program_id(0)
    i = al.arange(0, LANES)
    live = i < HH
    ii = al.where(live, i, 0)
    r = al.where(live, ii // HC, -1)
    c = al.where(live, ii - (ii // HC) * HC, -1)
    s0 = al.load(SCALE + 0)
    s1 = al.load(SCALE + 1)
    s2 = al.load(SCALE + 2)
    ic = ii - (ii // HC) * HC
    mpre = al.load(MIXES + row * MIX + ic)
    al.store(PRE + row * HC + i, al.sigmoid(mpre * s0 + al.load(BASE + ic)) + EPS, mask=i < HC)
    mpost = al.load(MIXES + row * MIX + HC + ic)
    al.store(POST + row * HC + i, 2.0 * al.sigmoid(mpost * s1 + al.load(BASE + HC + ic)), mask=i < HC)
    v = al.load(MIXES + row * MIX + 2 * HC + ii) * s2 + al.load(BASE + 2 * HC + ii)
    rowmax = v
    for _k in al.unroll(range(HC)):
        rowmax = al.where(r == _k, al.max(al.where(r == _k, v, NEG_INF)), rowmax)
    e = al.where(live, al.exp(v - rowmax), 0.0)
    comb = e / _group_sum(e, r, HC) + EPS
    comb = comb / (_group_sum(comb, c, HC) + EPS)
    for _it in range(ITERS - 1):
        comb = comb / (_group_sum(comb, r, HC) + EPS)
        comb = comb / (_group_sum(comb, c, HC) + EPS)
    al.store(COMB + row * HH + i, comb, mask=live)


@al.kernel
def ds_hc_pre(
    X,                        # (ROWS, HC, D) f32
    PRE,                      # (ROWS, HC) f32
    OUT: al.output,           # (ROWS, D) f32
    HC: al.constexpr,
    D: al.constexpr,
    BLOCK: al.constexpr = 256,
):
    """Collapse the hc copies: out = sum_c pre[c] * x[c]."""
    row = al.program_id(0)
    lane = al.arange(0, BLOCK)
    for k in range(0, D, BLOCK):
        offs = k + lane
        mask = offs < D
        acc = 0.0
        for _c in al.unroll(range(HC)):
            acc = acc + al.load(PRE + row * HC + _c) * al.load(X + (row * HC + _c) * D + offs, mask=mask, other=0.0)
        al.store(OUT + row * D + offs, acc, mask=mask)


@al.kernel
def ds_hc_post(
    Y,                        # (ROWS, D) f32 sublayer output
    RES,                      # (ROWS, HC, D) f32 residual stream
    POST,                     # (ROWS, HC)
    COMB,                     # (ROWS, HC, HC)
    OUT: al.output,           # (ROWS, HC, D)
    HC: al.constexpr,
    D: al.constexpr,
    BLOCK: al.constexpr = 256,
):
    """Expand back: out[c] = post[c] * y + sum_k comb[c, k] * res[k]. Grid (ROWS, HC)."""
    row = al.program_id(0)
    c = al.program_id(1)
    lane = al.arange(0, BLOCK)
    post = al.load(POST + row * HC + c)
    for k in range(0, D, BLOCK):
        offs = k + lane
        mask = offs < D
        acc = post * al.load(Y + row * D + offs, mask=mask, other=0.0)
        for _k in al.unroll(range(HC)):
            acc = acc + al.load(COMB + (row * HC + c) * HC + _k) * al.load(RES + (row * HC + _k) * D + offs, mask=mask, other=0.0)
        al.store(OUT + (row * HC + c) * D + offs, acc, mask=mask)


# --- engram ------------------------------------------------------------------------


@al.kernel
def ds_engram_gate(
    X,                        # (ROWS, HC, D) f32 stream
    KV,                       # (ROWS, (HC+1)*D) f32: keys for each copy then the value
    QK,                       # (HC, D) f32 q_weight * k_weight
    OUT: al.output,           # (ROWS, HC, D)
    HC: al.constexpr,
    D: al.constexpr,
    EPS: al.constexpr,
    BLOCK: al.constexpr = 256,
):
    """`Engram.forward`: per (row, copy) gate = sigmoid(signed sqrt of the
    normalized weighted dot of the copy against its key), out = x + gate * value."""
    row = al.program_id(0)
    c = al.program_id(1)
    lane = al.arange(0, BLOCK)
    xbase = X + (row * HC + c) * D
    kbase = KV + row * (HC + 1) * D + c * D
    vbase = KV + row * (HC + 1) * D + HC * D
    hh = 0.0
    kk = 0.0
    hk = 0.0
    for k in range(0, D, BLOCK):
        offs = k + lane
        mask = offs < D
        h = al.load(xbase + offs, mask=mask, other=0.0)
        key = al.load(kbase + offs, mask=mask, other=0.0)
        w = al.load(QK + c * D + offs, mask=mask, other=0.0)
        hh = hh + h * h
        kk = kk + key * key
        hk = hk + h * w * key
    rstd = al.rsqrt(al.sum(hh) / D + EPS) * al.rsqrt(al.sum(kk) / D + EPS)
    dot = al.sum(hk) * rstd * (1.0 / (D**0.5))
    mag = al.sqrt(al.maximum(al.abs(dot), 1e-6))
    gate = al.sigmoid(al.where(dot < 0.0, -mag, mag))
    for k in range(0, D, BLOCK):
        offs = k + lane
        mask = offs < D
        h = al.load(xbase + offs, mask=mask, other=0.0)
        v = al.load(vbase + offs, mask=mask, other=0.0)
        al.store(OUT + (row * HC + c) * D + offs, h + gate * v, mask=mask)


# --- moe router --------------------------------------------------------------------


@al.kernel
def ds_router_topk(
    LOGITS,                   # (ROWS, E) f32
    BIAS,                     # (E,) f32 selection bias
    IDX_OUT: al.output,       # (ROWS, K) int32
    W_OUT: al.output,         # (ROWS, K) f32 routing weights
    E: al.constexpr,
    K: al.constexpr,
    ROUTE_SCALE: al.constexpr,
    NORM: al.constexpr = 1,
    SCORE_FUNC: al.constexpr = 2,   # 0 softmax, 1 sigmoid, 2 sqrt(softplus)
    BLOCK: al.constexpr = 512,
):
    """DeepSeek `Gate`: scores from the logits, the bias steers selection only,
    weights are the raw scores of the picks (normalized to sum 1) * ROUTE_SCALE.
    Ties resolve to the lowest expert index."""
    row = al.program_id(0)
    offs = al.arange(0, BLOCK)
    mask = offs < E
    x = al.load(LOGITS + row * E + offs, mask=mask, other=0.0)
    if SCORE_FUNC == 0:
        mx = al.max(al.where(mask, x, NEG_INF))
        ex = al.where(mask, al.exp(x - mx), 0.0)
        score = ex / al.sum(ex)
    elif SCORE_FUNC == 1:
        score = al.sigmoid(x)
    else:
        sp = al.where(x > 20.0, x, al.log(1.0 + al.exp(al.minimum(x, 20.0))))
        score = al.sqrt(sp)
    sel = al.where(mask, score + al.load(BIAS + offs, mask=mask, other=0.0), NEG_INF)
    offs_f = al.cast(offs, al.float32)
    picked = [0.0] * K
    total = 0.0
    for k in al.unroll(range(K)):
        mk = al.max(sel)
        idx = al.min(al.where(sel == mk, offs_f, 1e30))
        al.store(IDX_OUT + row * K + k, al.cast(idx, al.int32), mask=offs < 1)
        picked[k] = al.sum(al.where(offs_f == idx, score, 0.0))
        total = total + picked[k]
        sel = al.where(offs_f == idx, NEG_INF, sel)
    for k in al.unroll(range(K)):
        w = picked[k]
        if NORM:
            w = w / (total + 1e-20)
        al.store(W_OUT + row * K + k, w * ROUTE_SCALE, mask=offs < 1)


# --- compressor ------------------------------------------------------------------------


@al.kernel
def ds_compress_pool(
    KV,                       # (NG*RATIO, D) f32 projected latents in position order
    SCORE,                    # (NG*RATIO, D) f32 gate logits
    OUT: al.output,           # (NG, D) f32 softmax-over-group weighted sum
    RATIO: al.constexpr,
    D: al.constexpr,
    BLOCK: al.constexpr = 256,
):
    """`Compressor`: each group of RATIO consecutive tokens pools into one latent,
    per-channel softmax over the group's gate logits."""
    g = al.program_id(0)
    lane = al.arange(0, BLOCK)
    for k in range(0, D, BLOCK):
        offs = k + lane
        mask = offs < D
        mx = NEG_INF
        for _r in al.unroll(range(RATIO)):
            mx = al.maximum(mx, al.load(SCORE + (g * RATIO + _r) * D + offs, mask=mask, other=0.0))
        den = 0.0
        num = 0.0
        for _r in al.unroll(range(RATIO)):
            e = al.exp(al.load(SCORE + (g * RATIO + _r) * D + offs, mask=mask, other=0.0) - mx)
            den = den + e
            num = num + e * al.load(KV + (g * RATIO + _r) * D + offs, mask=mask, other=0.0)
        al.store(OUT + g * D + offs, num / den, mask=mask)


# --- grouped output projection ------------------------------------------------------


@al.tunable(
    BLOCK_M=[8, 16, 32, 64],
    BLOCK_N=[32, 64, 128],
    BLOCK_K=[32, 64, 128],
)
@al.kernel
def ds_grouped_out_proj(
    A,                        # (M, G*K_G) f32 attention output, group-major columns
    BLK,                      # (G*N_G, (K_G/256)*144) uint8 Q4_K wo_a rows
    C: al.output,             # (M, G*N_G) f32
    K_G: al.constexpr,        # input dims per group
    N_G: al.constexpr,        # output dims per group
    BLOCK_M: al.constexpr = 16,
    BLOCK_N: al.constexpr = 64,
    BLOCK_K: al.constexpr = 64,
):
    """Block-diagonal `wo_a`: output column n belongs to group g = n // N_G and reads
    only A's columns [g*K_G, (g+1)*K_G). A `dot_q4_k` tile with a per-tile A column
    offset; BLOCK_N must divide N_G."""
    M = A.shape[0]
    G_K = A.shape[1]
    N = BLK.shape[0]
    ROW_BYTES = (K_G // 256) * 144
    pm = al.program_id(0)
    pn = al.program_id(1)
    g = (pn * BLOCK_N) // N_G
    rm = pm * BLOCK_M + al.arange(0, BLOCK_M)
    rn = pn * BLOCK_N + al.arange(0, BLOCK_N)
    rk = al.arange(0, BLOCK_K)
    a_ptrs = A + rm[:, None] * G_K + g * K_G + rk[None, :]
    acc = al.zeros((BLOCK_M, BLOCK_N), dtype=al.float32)
    for k in range(0, K_G, BLOCK_K):
        elem_k = k + rk
        a = al.load(a_ptrs, mask=(rm[:, None] < M) & (elem_k[None, :] < K_G))
        b = al.load(
            BLK + rn[:, None] * ROW_BYTES + elem_k[None, :],
            mask=(rn[:, None] < N) & (elem_k[None, :] < K_G),
            _dequant_format="q4_k",
        )
        acc += al.tile_dot(a, b, transpose_rhs=True)
        a_ptrs += BLOCK_K
    al.store(C + rm[:, None] * N + rn[None, :], acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


# --- row movement (window ring, chunk assembly) --------------------------------------


@al.kernel
def ds_copy_rows(
    SRC,                      # (n, D)
    IDX,                      # (ROWS,) int32 source rows
    DST: al.output,           # (m, D)
    D: al.constexpr,
    DST_ROW0: al.constexpr = 0,
    OUT_F16: al.constexpr = 1,
    BLOCK: al.constexpr = 256,
):
    """DST[DST_ROW0 + r] = SRC[IDX[r]]. One program per row."""
    r = al.program_id(0)
    src = al.cast(al.load(IDX + r), al.int32)
    lane = al.arange(0, BLOCK)
    for k in range(0, D, BLOCK):
        offs = k + lane
        mask = offs < D
        v = al.cast(al.load(SRC + src * D + offs, mask=mask, other=0.0), al.float32)
        _store(DST + (r + DST_ROW0) * D + offs, v, OUT_F16, mask)


@al.kernel
def ds_expand_hc(
    X,                        # (ROWS, D) f32
    OUT: al.output,           # (ROWS, HC, D) f32, every copy = x
    HC: al.constexpr,
    D: al.constexpr,
    BLOCK: al.constexpr = 256,
):
    row = al.program_id(0)
    lane = al.arange(0, BLOCK)
    for k in range(0, D, BLOCK):
        offs = k + lane
        mask = offs < D
        v = al.load(X + row * D + offs, mask=mask, other=0.0)
        for _c in al.unroll(range(HC)):
            al.store(OUT + (row * HC + _c) * D + offs, v, mask=mask)


@al.kernel
def ds_scatter_rows(
    SRC,                      # (n, D) f16
    IDX,                      # (ROWS,) int32 destination rows (distinct)
    DEP,                      # a view of DST: orders this write after DST's readers/writers
    DST: al.output,           # (m, D) f16
    D: al.constexpr,
    SRC_ROW0: al.constexpr = 0,
    BLOCK: al.constexpr = 256,
):
    """DST[IDX[r]] = SRC[SRC_ROW0 + r]. One program per row."""
    r = al.program_id(0)
    _dep = al.cast(al.load(DEP + 0), al.float32)
    dst = al.cast(al.load(IDX + r), al.int32)
    lane = al.arange(0, BLOCK)
    for k in range(0, D, BLOCK):
        offs = k + lane
        mask = offs < D
        v = al.cast(al.load(SRC + (r + SRC_ROW0) * D + offs, mask=mask, other=0.0), al.float32)
        al.store(DST + dst * D + offs, al.cast(v, al.float16), mask=mask)


# --- compressor over a chunk with carried partial-group state ------------------------


@al.kernel
def ds_compress_chunk(
    STATE_KV,                 # (RATIO, D) f32 pending rows [0, R) from the previous chunk
    STATE_SCORE,
    KV_C,                     # (T, D) f32 this chunk's projected latents
    SCORE_C,
    OUT: al.output,           # (NG, D) f32 pooled latents, NG = (R + T) // RATIO
    R: al.constexpr,
    RATIO: al.constexpr,
    D: al.constexpr,
    BLOCK: al.constexpr = 256,
):
    """Group g pools concat(state[:R], chunk)[g*RATIO : (g+1)*RATIO] with a per-channel
    softmax over the gate logits (`Compressor`, generalized to any chunk start)."""
    g = al.program_id(0)
    lane = al.arange(0, BLOCK)
    for k in range(0, D, BLOCK):
        offs = k + lane
        mask = offs < D
        mx = NEG_INF
        for _r in al.unroll(range(RATIO)):
            i = g * RATIO + _r
            from_state = i < R
            si = al.where(from_state, i, 0)
            ci = al.where(from_state, 0, i - R)
            s = al.where(from_state, al.load(STATE_SCORE + si * D + offs, mask=mask, other=0.0),
                         al.load(SCORE_C + ci * D + offs, mask=mask, other=0.0))
            mx = al.maximum(mx, s)
        den = 0.0
        num = 0.0
        for _r in al.unroll(range(RATIO)):
            i = g * RATIO + _r
            from_state = i < R
            si = al.where(from_state, i, 0)
            ci = al.where(from_state, 0, i - R)
            s = al.where(from_state, al.load(STATE_SCORE + si * D + offs, mask=mask, other=0.0),
                         al.load(SCORE_C + ci * D + offs, mask=mask, other=0.0))
            v = al.where(from_state, al.load(STATE_KV + si * D + offs, mask=mask, other=0.0),
                         al.load(KV_C + ci * D + offs, mask=mask, other=0.0))
            e = al.exp(s - mx)
            den = den + e
            num = num + e * v
        al.store(OUT + g * D + offs, num / den, mask=mask)


@al.kernel
def ds_compress_tail(
    STATE_KV,                 # (RATIO, D) f32 previous pending rows
    STATE_SCORE,
    KV_C,                     # (T, D) f32
    SCORE_C,
    NEW_KV: al.output,        # (RATIO, D) f32 the REM pending rows after this chunk
    NEW_SCORE: al.output,
    R: al.constexpr,
    T: al.constexpr,
    REM: al.constexpr,
    D: al.constexpr,
    BLOCK: al.constexpr = 256,
):
    """The last REM rows of concat(state[:R], chunk) become the next pending state
    (double-buffered: NEW_* must not alias STATE_*). Grid (REM,)."""
    j = al.program_id(0)
    i = R + T - REM + j
    from_state = i < R
    si = al.where(from_state, i, 0)
    ci = al.where(from_state, 0, i - R)
    lane = al.arange(0, BLOCK)
    for k in range(0, D, BLOCK):
        offs = k + lane
        mask = offs < D
        v = al.where(from_state, al.load(STATE_KV + si * D + offs, mask=mask, other=0.0),
                     al.load(KV_C + ci * D + offs, mask=mask, other=0.0))
        s = al.where(from_state, al.load(STATE_SCORE + si * D + offs, mask=mask, other=0.0),
                     al.load(SCORE_C + ci * D + offs, mask=mask, other=0.0))
        al.store(NEW_KV + j * D + offs, v, mask=mask)
        al.store(NEW_SCORE + j * D + offs, s, mask=mask)


# --- shared expert activation ----------------------------------------------------------


@al.kernel
def ds_swiglu_clamp(
    G,                        # (N,) f32 gate pre-activation
    U,                        # (N,) f32 up
    OUT: al.output,           # (N,) silu(min(g, L)) * clamp(u, -L, L)
    N: al.constexpr,
    LIMIT: al.constexpr,
    BLOCK: al.constexpr = 1024,
):
    pid = al.program_id(0)
    offs = pid * BLOCK + al.arange(0, BLOCK)
    mask = offs < N
    g = al.load(G + offs, mask=mask, other=0.0)
    u = al.load(U + offs, mask=mask, other=0.0)
    if LIMIT > 0:
        g = al.minimum(g, LIMIT)
        u = al.clamp(u, -LIMIT, LIMIT)
    al.store(OUT + offs, g * (1.0 / (1.0 + al.exp(-g))) * u, mask=mask)
