"""Mixture-of-Experts (MoE) routed-expert kernels for Qwen3.5-MoE.

The routed-expert FFN is a *gathered* quantized GEMV: each token routes to
TOP_K experts (chosen at runtime by the router), and per expert computes

    h    = silu(gate[e] @ x) * (up[e] @ x)            # (I,)
    y_e  = down[e] @ h                                 # (H,)
    out  = sum_e  w_e * y_e                            # (H,)

The expert index `e` is read from a runtime routing buffer (the same
data-dependent-dispatch pattern as `attention_strided_runtime_pos`): the grid
is static (token×expert-slot × output tile), and each program loads its expert
index, computes the per-expert weight base offset, and runs the quantized dot.
This keeps the whole MoE block in one static compiled plan with routing as a
runtime INPUT — no per-step Python, no recompile when the routing changes.

Weight layout (matches `GGUFQwen35MoeExperts`, fused gate_up):
  gate_up : GGUF-native Q4_K, (E, 2I, (K/256)*144) raw 144-byte superblocks.
            rows [0:I] are gate, rows [I:2I] are up (fused; row-concat).
  down    : Q6_K, (E, H, (I/256)*210) raw packed blocks.

These are the M=1 decode-shaped kernels (one program per (token,slot) pair,
vec4 K-vectorized over 32 lanes). Prefill (M=128) runs the same kernels with
more (token,slot) programs — correct but GEMV-bound.
"""

import alloy as al
from alloy.std.quant import _q4k_contrib, _q4k_load_y, _q4k_silu


@al.kernel
def moe_router_topk(
    LOGITS,             # (T, NUM_EXPERTS) f32 router logits
    IDX_OUT: al.output,    # (T, TOP_K) int32 selected expert indices
    WEIGHT_OUT: al.output, # (T, TOP_K) f32 routing weights (softmax over the top-k)
    ACTIVE_OUT: al.output, # (1,) int32 = this dispatch's row count (grid axis-0)
    NUM_EXPERTS: al.constexpr,
    TOP_K: al.constexpr,
    BLOCK: al.constexpr = 256,
):
    """Top-K router: per token, select the TOP_K largest logits (descending) and
    softmax over just those K.

    The HF reference softmaxes over all NUM_EXPERTS, takes the top-K probs, then
    renormalizes them to sum to 1. Because softmax is monotonic the global
    denominator cancels, so `renorm(topk(softmax(logits)))` is exactly
    `softmax(topk_logits)` — we select on raw logits and softmax the K winners.
    Requires NUM_EXPERTS <= BLOCK (one tile); Qwen3.5-MoE has 256 experts.

    ACTIVE_OUT records how many rows this launch actually routed — the counting
    sort's row bound must EXACTLY match the slots the router wrote (a looser
    bound scans stale slots, a tighter one leaves phantom slots the perm scan
    still scatters: measured 8.2 KV drift). num_programs is the launch grid, so
    the bound is exact for ANY dispatched grid — full, grid-shrunk, or a
    partial recipe that pinned this dispatch. Every program stores the same
    value (aligned 4B, value-identical: race-free overwrite, no atomics).
    """
    row = al.program_id(0)
    al.store(ACTIVE_OUT + 0, al.cast(al.num_programs(0), al.int32))
    offs = al.arange(0, BLOCK)
    mask = offs < NUM_EXPERTS
    w = al.load(LOGITS + row * NUM_EXPERTS + offs, mask=mask, other=-1e30)
    offs_f = al.cast(offs, al.float32)

    vals = []
    for k in al.unroll(range(TOP_K)):
        mk = al.max(w)                                   # k-th largest logit
        cand = al.where(w == mk, offs_f, 1e30)
        idx = al.min(cand)                               # its index (lowest on tie)
        al.store(IDX_OUT + row * TOP_K + k, al.cast(idx, al.int32))
        vals.append(mk)
        w = al.where(offs_f == idx, -1e30, w)            # mask out for next pass

    m0 = vals[0]                                          # max (selection is descending)
    sumexp = 0.0
    for k in al.unroll(range(TOP_K)):
        sumexp = sumexp + al.exp(vals[k] - m0)
    for k in al.unroll(range(TOP_K)):
        al.store(WEIGHT_OUT + row * TOP_K + k, al.exp(vals[k] - m0) / sumexp)


@al.kernel
def moe_gate_up_silu(
    A,                  # (T, K) activations (fp32), one row per token
    GATE_UP_BLK,        # (E, 2I, (K/256)*144) uint8 fused gate_up native Q4_K blocks
    ROUTING,            # (T*TOP_K,) int32 expert index per (token,slot)
    H_OUT: al.output,   # (T*TOP_K, I) silu(gate)*up
    K: al.constexpr,
    MOE_INTER: al.constexpr,     # I
    TOP_K: al.constexpr,
):
    """Gathered native-Q4_K gate+up matvec with SiLU fusion (decode, one row/slot).

    Mirrors `dot_q4_k_silu_v2` (NSG=1, NR0=1) but folds the routed expert into the
    weight ROW index: gate row `e*2I + col0`, up row `e*2I + col0 + I` inside the
    fused (E, 2I, ...) native block tensor.
    """
    I = MOE_INTER
    NB = K // 256
    NQ = NB // 4
    R = NB - 4 * NQ
    TWO_I = 2 * I

    ts = al.program_id(0)                  # flattened (token, slot) index
    col0 = al.program_id(1)                # output channel in [0, I)
    t = ts // TOP_K                        # activation row (token)
    e = al.cast(al.load(ROUTING + ts), al.int32)
    gate_row = e * TWO_I + col0
    up_row = e * TWO_I + col0 + I

    tid = al.arange(0, 32)
    ix = tid // 8
    it = tid - ix * 8
    iq = it // 4
    ir = it - iq * 4
    o = 64 * iq + 8 * ir

    g0 = 0.0
    u0 = 0.0
    for jj in range(NQ):
        ib = ix + 4 * jj
        yl, yh, sumy = _q4k_load_y(A, t * K + ib * 256 + o)
        g0 = g0 + _q4k_contrib(GATE_UP_BLK, (gate_row * NB + ib) * 144, iq, ir, yl, yh, sumy)
        u0 = u0 + _q4k_contrib(GATE_UP_BLK, (up_row * NB + ib) * 144, iq, ir, yl, yh, sumy)
    if R > 0:
        ib = ix + 4 * NQ
        valid = ix < R
        ibc = al.where(valid, ib, 0)
        yl, yh, sumy = _q4k_load_y(A, t * K + ibc * 256 + o)
        g0 = g0 + al.where(valid, _q4k_contrib(GATE_UP_BLK, (gate_row * NB + ibc) * 144, iq, ir, yl, yh, sumy), 0.0)
        u0 = u0 + al.where(valid, _q4k_contrib(GATE_UP_BLK, (up_row * NB + ibc) * 144, iq, ir, yl, yh, sumy), 0.0)

    g = al.simd_reduce(g0)
    u = al.simd_reduce(u0)
    al.store(H_OUT + ts * I + col0, _q4k_silu(g) * u, mask=(tid < 1))


@al.kernel
def moe_down_combine(
    H_IN,               # (T*TOP_K, I) per-slot gate_up_silu outputs (fp32)
    DOWN_Q6,            # (E, H, (I/256)*210) uint8 Q6_K down weights
    ROUTING,            # (T*TOP_K,) int32 expert index per (token,slot)
    WEIGHTS,            # (T*TOP_K,) f32 routing weight per (token,slot)
    Y_OUT: al.output,   # (T, H) combined routed-expert output
    HID: al.constexpr,           # H (hidden / down output dim)
    MOE_INTER: al.constexpr,     # I (down input dim == gate_up output)
    TOP_K: al.constexpr,
    GROUP_SIZE: al.constexpr = 256,
):
    """Gathered Q6_K down matvec with the routing-weighted slot reduction folded
    in (decode, one program per (token, output channel)).

    Mirrors `dot_q6_k_v2`'s lane layout, but gathers the per-expert down row,
    reads its activation from the slot's `H_IN` row, and accumulates
    `w[slot] * partial` per-lane across the TOP_K slots before the single
    simd reduce — so `y[t,j] = sum_slot w * (down[e] @ h[slot])`.
    """
    I = MOE_INTER
    BLOCK_BYTES = 210
    QL_BYTES = 128
    QH_BYTES = 64
    SCALE_BYTES = 16
    N_GROUPS = I // GROUP_SIZE
    ROW_BYTES = N_GROUPS * BLOCK_BYTES
    E_STRIDE = HID * ROW_BYTES

    t = al.program_id(0)
    j = al.program_id(1)                  # output channel in [0, H)
    lane = al.arange(0, 32)

    acc = 0.0
    for s in range(TOP_K):
        ts = t * TOP_K + s
        e = al.cast(al.load(ROUTING + ts), al.int32)
        w = al.load(WEIGHTS + ts)
        down_row_base = e * E_STRIDE + j * ROW_BYTES
        for g in range(N_GROUPS):
            block_base = down_row_base + g * BLOCK_BYTES
            scale_base = block_base + QL_BYTES + QH_BYTES
            d_base = scale_base + SCALE_BYTES
            d_lo = al.cast(al.load(DOWN_Q6 + d_base), "uint16")
            d_hi = al.cast(al.load(DOWN_Q6 + d_base + 1), "uint16")
            d_bits = al.cast(d_lo | (d_hi << 8), "uint16")
            d = al.cast(al.bitcast(d_bits, al.float16), al.float32)
            for quadrant in range(2):
                ql_base = block_base + quadrant * 64
                qh_base = block_base + QL_BYTES + quadrant * 32
                ql_v4 = al.load4_vec(DOWN_Q6 + ql_base + (lane % 16) * 4)
                nibble_shift = al.where(lane < 16, 0, 4)
                ql_nibbles = (ql_v4 >> nibble_shift) & 0x0F
                qh_v4 = al.load4_vec(DOWN_Q6 + qh_base + (lane % 8) * 4)
                qh_shift = (lane // 8) * 2
                qh_bits = (qh_v4 >> qh_shift) & 0x03
                q_combined = ql_nibbles | (qh_bits << 4)
                q_signed = al.cast(q_combined, al.int32) - 32
                q_f4 = al.cast(q_signed, al.float32)
                k_in_quad = (lane // 8) * 32 + (lane % 8) * 4
                a_v4 = al.load4_vec(H_IN + ts * I + g * GROUP_SIZE + quadrant * 128 + k_in_quad)
                scale_idx = (lane // 4) + quadrant * 8
                scale_raw = al.cast(al.load(DOWN_Q6 + scale_base + scale_idx), al.int32)
                scale = al.where(scale_raw > 127, scale_raw - 256, scale_raw)
                scale_f = al.cast(scale, al.float32)
                acc = acc + w * (d * scale_f * al.dot4(a_v4, q_f4))

    y = al.simd_reduce(acc)
    al.store(Y_OUT + t * HID + j, y, mask=(lane < 1))


# --- Grouped-GEMM prefill path (M>1) -----------------------------------------
# The decode kernels above are GEMV (one program per (token,slot)/(token,channel)):
# correct but they re-read each expert's weights once per routed token, so prefill
# (M=128 chunk) runs ~4.5x above its memory roofline. The grouped path below sorts
# the chunk's routings by expert into BLOCK_M-padded tiles (moe_sort_routing),
# then runs a tiled simdgroup-MMA GEMM per tile that dequantizes the expert weight
# ONCE and reuses it across the tile's tokens — the same dot_q4_k_silu / dot_q6_k
# machinery, with the expert folded into the weight ROW index (rn_eff = e*2I + rn),
# so the existing fused-dequant cooperative load indexes scales[rn_eff*n_groups+g]
# unchanged. Tiles are tile-major (H_OUT[rm]); the token un-permute + routing-weight
# combine happens once, at the end of moe_down_grouped_combine.


# BLOCK_M is NOT tuned: it must equal the sort's PAD_M (tile <-> expert-segment
# alignment), so it's fixed (8 = the MMA floor, lowest padding at ~4 tokens/expert).
# Only BLOCK_N/BLOCK_K are swept.
@al.tunable(
    BLOCK_N=[32, 64, 128],
    BLOCK_K=[64, 128, 256],
)
@al.kernel
def moe_gate_up_grouped(
    A,                  # (T, K) activations (fp32) — tile rows GATHERED via TOK_LD (no a_perm)
    GATE_UP_BLK,        # (E*2I, (K/256)*144) uint8 fused gate_up native Q4_K, expert-stacked rows
    PERM,               # (MAX_ROWS,) int32 padded-row -> source (token,slot) flat idx (valid mask)
    TOK_LD,             # (MAX_ROWS,) int32 padded-row -> source TOKEN row, pads clamped to 0 (moe_row_tokens)
    TILE_EXPERT,        # (MAX_TILES,) int32 tile -> expert id
    TOTAL_ROWS,         # (1,) int32 active-tile row boundary (rows >= this are the empty tail)
    W_ROW,              # (MAX_ROWS,) f32 routing weight per sorted row, pads = 0 (moe_row_tokens)
    H_OUT: al.output,   # (MAX_ROWS, I) w * silu(gate)*up, TILE-MAJOR (parallel to PERM)
    K: al.constexpr,
    MOE_INTER: al.constexpr,     # I
    BLOCK_M: al.constexpr = 8,    # == sort PAD_M; not tuned
    BLOCK_N: al.constexpr = 32,
    BLOCK_K: al.constexpr = 64,
):
    """Grouped native-Q4_K gate+up GEMM with SiLU and the ROUTING WEIGHT folded in
    (prefill, one tile of one expert).

    Mirrors `dot_q4_k_silu` but the tile's expert `e = TILE_EXPERT[pm]` folds into
    the weight row (gate row `e*2I + rn`, up row `e*2I + I + rn`). The tile's A rows are
    GATHERED in-kernel: the cooperative A-load reads source token `TOK_LD[rm]` per row
    (the emitter's gathered-coop-load path; pads are pre-clamped to row 0 by
    moe_row_tokens so the load never dereferences OOB) — there is NO global pre-gathered
    `a_perm` buffer ((T*TOP_K, H) = 17GB at native). The routing weight `w = W_ROW[rm]`
    (pre-resolved per sorted row by moe_row_tokens, pads = 0) multiplies the epilogue
    (`h = w * silu(gate) * up`): by linearity the down GEMM's output rows then carry w, so
    the fused scatter-accumulate down (`moe_down_grouped` reduce-add) needs no separate
    combine pass — and pad h rows are exactly zero. Padding / inactive-tile rows
    (PERM[rm] < 0) are masked out of the store. Branch-free.
    """
    I = MOE_INTER
    TWO_I = 2 * I
    ROW_BYTES = (K // 256) * 144

    pm = al.program_id(0)
    pn = al.program_id(1)
    rm = pm * BLOCK_M + al.arange(0, BLOCK_M)
    rn = pn * BLOCK_N + al.arange(0, BLOCK_N)   # output channel in [0, I)
    rk = al.arange(0, BLOCK_K)

    e = al.cast(al.load(TILE_EXPERT + pm), al.int32)
    e_safe = al.maximum(e, al.cast(0, al.int32))
    src = al.cast(al.load(PERM + rm), al.int32)             # (BLOCK_M,) (token,slot) — valid mask
    valid = src >= 0
    row_tok = al.cast(al.load(TOK_LD + rm), al.int32)       # (BLOCK_M,) source token per sorted row
    w_row = al.load(W_ROW + rm)                             # (BLOCK_M,) f32 routing weight (pads 0)
    # Empty tiles (the static worst-case grid's tail beyond the actual active tiles,
    # which are contiguous in [0, TOTAL_ROWS)) skip the whole K-loop via a runtime
    # bound — no weight cooperative-load, no MMA. Same idiom as attention's empty-split skip.
    total_rows = al.cast(al.load(TOTAL_ROWS + 0), al.int32)
    k_end = al.where(pm * BLOCK_M < total_rows, al.cast(K, al.int32), al.cast(0, al.int32))

    a_ptrs = A + row_tok[:, None] * K + rk[None, :]         # gathered: A[ROW_TOKEN[rm]] (no a_perm)
    gate_row = e_safe * TWO_I + rn                           # (BLOCK_N,)
    up_row = e_safe * TWO_I + I + rn
    acc_gate = al.zeros((BLOCK_M, BLOCK_N), dtype=al.float32)
    acc_up = al.zeros((BLOCK_M, BLOCK_N), dtype=al.float32)

    for k in range(0, k_end, BLOCK_K):
        elem_k = k + rk
        a = al.load(a_ptrs, mask=valid[:, None] & (elem_k[None, :] < K))
        gate = al.load(
            GATE_UP_BLK + gate_row[:, None] * ROW_BYTES + elem_k[None, :],
            mask=elem_k[None, :] < K,
            _dequant_format="q4_k",
        )
        up = al.load(
            GATE_UP_BLK + up_row[:, None] * ROW_BYTES + elem_k[None, :],
            mask=elem_k[None, :] < K,
            _dequant_format="q4_k",
        )
        acc_gate += al.tile_dot(a, gate, transpose_rhs=True)
        acc_up += al.tile_dot(a, up, transpose_rhs=True)
        a_ptrs += BLOCK_K

    silu = acc_gate * (1.0 / (1.0 + al.exp(-acc_gate))) * acc_up * w_row[:, None]
    al.store(H_OUT + rm[:, None] * I + rn[None, :], silu, mask=valid[:, None] & (rn[None, :] < I))


# BLOCK_M fixed (= sort PAD_M); only BLOCK_N/BLOCK_K swept (see moe_gate_up_grouped).
@al.tunable(
    BLOCK_N=[32, 64, 128],
    BLOCK_K=[64, 128, 256],
)
@al.kernel
def moe_down_grouped(
    H_IN,               # (MAX_ROWS, I) tile-major w·gate_up_silu output
    DOWN_Q6,            # (E*H, (I/256)*210) uint8 Q6_K down weights, expert-stacked rows
    PERM,               # (MAX_ROWS,) int32 padded-row -> source (token,slot) flat idx
    TOK_ST,             # (MAX_ROWS,) int32 padded-row -> source TOKEN row, pads = huge sentinel
    TILE_EXPERT,        # (MAX_TILES,) int32 tile -> expert id
    TOTAL_ROWS,         # (1,) int32 active-tile row boundary (rows >= this are the empty tail)
    Y_DEP,              # == Y_OUT buffer, as an INPUT: the atomic scatter is a read-modify-write,
                        # and declaring the read gives the planner the zero-fill -> GEMM edge
                        # (writers alone can be topo-reordered)
    Y_OUT: al.output,   # (T, H) combined routed-expert output — scatter-ACCUMULATED (pre-zeroed)
    HID: al.constexpr,           # H (down output dim)
    MOE_INTER: al.constexpr,     # I (down input dim == gate_up output)
    T_ROWS: al.constexpr,        # T — Y row count; the scatter guard (pads' sentinel fails it)
    MAX_ROWS: al.constexpr,      # H_IN row count — explicit H_IN row bound (see load mask note)
    BLOCK_M: al.constexpr = 8,    # == sort PAD_M; not tuned
    BLOCK_N: al.constexpr = 64,
    BLOCK_K: al.constexpr = 64,
    GROUP_SIZE: al.constexpr = 256,
):
    """Grouped Q6_K down GEMM with the combine FUSED as a scatter-accumulate (prefill,
    one tile of one expert).

    Mirrors `dot_q6_k` with the tile's expert `e = TILE_EXPERT[pm]` folded into the
    down weight row (`e*H + rn`). Reads the tile-major `H_IN` (which already carries the
    routing weight, folded in by `moe_gate_up_grouped`) and atomic-ADDs each row's down
    output directly into `Y_OUT[token]` via the gathered store row `TOK_ST[rm]`
    (`reduce="add"` → the emitter's scatter-accumulate MMA epilogue) — there is NO global
    (MAX_ROWS, H) per-row `partial` buffer (17GB at native) and no separate combine pass.
    Y_OUT must be pre-zeroed (`moe_zero_f32`). Pad rows carry the TOK_ST sentinel, which
    fails the `token < T_ROWS` store guard; a token's TOP_K expert rows land in different
    tiles, hence the atomic add (cross-threadgroup accumulation, order non-deterministic
    at f32-ULP level — the combine sum is associative-reordered, not bit-stable).
    """
    I = MOE_INTER
    BLOCK_BYTES = 210
    N_GROUPS = I // GROUP_SIZE
    ROW_BYTES = N_GROUPS * BLOCK_BYTES

    pm = al.program_id(0)
    pn = al.program_id(1)
    rm = pm * BLOCK_M + al.arange(0, BLOCK_M)
    rn = pn * BLOCK_N + al.arange(0, BLOCK_N)   # output channel in [0, H)
    rk = al.arange(0, BLOCK_K)

    e = al.cast(al.load(TILE_EXPERT + pm), al.int32)
    e_safe = al.maximum(e, al.cast(0, al.int32))
    src = al.cast(al.load(PERM + rm), al.int32)
    valid = src >= 0
    tok = al.cast(al.load(TOK_ST + rm), al.int32)  # scatter row; pads = sentinel >= T_ROWS
    _dep = al.load(Y_DEP + 0)  # value unused — records the RMW read (ordering edge only)
    # Empty tiles (grid tail beyond the contiguous active tiles in [0, TOTAL_ROWS)) skip
    # the K-loop (no weight load / MMA); store guarded by the token bound.
    total_rows = al.cast(al.load(TOTAL_ROWS + 0), al.int32)
    k_end = al.where(pm * BLOCK_M < total_rows, al.cast(I, al.int32), al.cast(0, al.int32))

    a_ptrs = H_IN + rm[:, None] * I + rk[None, :]
    down_row = e_safe * HID + rn                  # fold expert into down weight row
    acc = al.zeros((BLOCK_M, BLOCK_N), dtype=al.float32)

    for k in range(0, k_end, BLOCK_K):
        elem_k = k + rk
        # The `rm < MAX_ROWS` term is always true (the grid covers exactly MAX_ROWS)
        # but it must be EXPLICIT: this kernel's OUTPUT is (T, H), so the dispatch's
        # auto-derived row bound M == T — without a parseable row bound in the mask,
        # the cooperative loader falls back to `_gr < M(=T)` and silently fills every
        # H_IN row >= T with zeros (only tile 0's contribution survived). With the
        # explicit bound the loader parses MAX_ROWS, and since MAX_ROWS % BLOCK_M == 0
        # the check is elided entirely.
        a = al.load(
            a_ptrs,
            mask=valid[:, None] & (rm[:, None] < MAX_ROWS) & (elem_k[None, :] < I),
        )
        b = al.load(
            DOWN_Q6 + down_row[:, None] * ROW_BYTES + elem_k[None, :],
            mask=elem_k[None, :] < I,
            _dequant_format="q6_k",
        )
        acc += al.tile_dot(a, b, transpose_rhs=True)
        a_ptrs += BLOCK_K

    al.store(
        Y_OUT + tok[:, None] * HID + rn[None, :],
        acc,
        mask=(tok[:, None] < T_ROWS) & (rn[None, :] < HID),
        reduce="add",
    )


# BLOCK_M fixed (= sort PAD_M); only BLOCK_N/BLOCK_K swept (see moe_gate_up_grouped).
@al.tunable(
    BLOCK_N=[32, 64, 128],
    BLOCK_K=[64, 128, 256],
)
@al.kernel
def moe_down_grouped_partial(
    H_IN,               # (MAX_ROWS, I) tile-major w·gate_up_silu output
    DOWN_Q6,            # (E*H, (I/256)*210) uint8 Q6_K down weights, expert-stacked rows
    PERM,               # (MAX_ROWS,) int32 padded-row -> source (token,slot) flat idx
    TILE_EXPERT,        # (MAX_TILES,) int32 tile -> expert id
    TOTAL_ROWS,         # (1,) int32 active-tile row boundary
    PARTIAL: al.output,  # (MAX_ROWS, H) per-sorted-row down output (overwrite, no atomics)
    HID: al.constexpr,
    MOE_INTER: al.constexpr,
    MAX_ROWS: al.constexpr,
    BLOCK_M: al.constexpr = 8,    # == sort PAD_M; not tuned
    BLOCK_N: al.constexpr = 64,
    BLOCK_K: al.constexpr = 64,
    GROUP_SIZE: al.constexpr = 256,
):
    """`moe_down_grouped` with the combine UNFUSED: each sorted row's down
    output lands in PARTIAL[rm] for `moe_combine_rows`' fixed-order reduce.
    The production T>1 path (spec verify AND chunk prefill): the fused
    kernel's cross-tile atomic-ADD combine is associative-reordered (not
    bit-stable) — measured ~0.1-0.3-logit run-to-run prefill jitter on
    qwen3.6:35b, which breaks seeded reproducibility and flips near-tie
    argmax between the plain and spec streams. The (MAX_ROWS, H) partial
    buffer the fusion exists to avoid is 285MB transient at the 4096
    chunk (~0.6% chunk-time traffic; the 17GB figure was a native-context
    M=262144 plan production never compiles). Pad rows store garbage
    partials that `moe_combine_rows` never reads (INV_PERM addresses
    active rows only) — the sort's fill-free philosophy. Keep the body
    in sync with `moe_down_grouped`.
    """
    I = MOE_INTER
    BLOCK_BYTES = 210
    N_GROUPS = I // GROUP_SIZE
    ROW_BYTES = N_GROUPS * BLOCK_BYTES

    pm = al.program_id(0)
    pn = al.program_id(1)
    rm = pm * BLOCK_M + al.arange(0, BLOCK_M)
    rn = pn * BLOCK_N + al.arange(0, BLOCK_N)   # output channel in [0, H)
    rk = al.arange(0, BLOCK_K)

    e = al.cast(al.load(TILE_EXPERT + pm), al.int32)
    e_safe = al.maximum(e, al.cast(0, al.int32))
    src = al.cast(al.load(PERM + rm), al.int32)
    valid = src >= 0
    total_rows = al.cast(al.load(TOTAL_ROWS + 0), al.int32)
    k_end = al.where(pm * BLOCK_M < total_rows, al.cast(I, al.int32), al.cast(0, al.int32))

    a_ptrs = H_IN + rm[:, None] * I + rk[None, :]
    down_row = e_safe * HID + rn
    acc = al.zeros((BLOCK_M, BLOCK_N), dtype=al.float32)

    for k in range(0, k_end, BLOCK_K):
        elem_k = k + rk
        a = al.load(
            a_ptrs,
            mask=valid[:, None] & (rm[:, None] < MAX_ROWS) & (elem_k[None, :] < I),
        )
        b = al.load(
            DOWN_Q6 + down_row[:, None] * ROW_BYTES + elem_k[None, :],
            mask=elem_k[None, :] < I,
            _dequant_format="q6_k",
        )
        acc += al.tile_dot(a, b, transpose_rhs=True)
        a_ptrs += BLOCK_K

    al.store(
        PARTIAL + rm[:, None] * HID + rn[None, :],
        acc,
        mask=(rm[:, None] < MAX_ROWS) & (rn[None, :] < HID),
    )


@al.kernel
def moe_combine_rows(
    PARTIAL,            # (MAX_ROWS, H) per-sorted-row down output
    INV_PERM,           # (T*TOP_K,) int32 (token,slot) flat idx -> sorted row
    Y_OUT: al.output,   # (T, H) combined routed-expert output
    HID: al.constexpr,
    TOP_K: al.constexpr,
    BLOCK: al.constexpr = 256,
):
    """Fixed-order routed-expert combine: Y[t] = Σ_s PARTIAL[INV_PERM[t·K+s]]
    in slot order — bit-stable across runs (the routing weight is already
    folded into PARTIAL via gate_up's w_row). Grid (T, ceil(H/BLOCK))."""
    t = al.program_id(0)
    pn = al.program_id(1)
    rh = pn * BLOCK + al.arange(0, BLOCK)
    hmask = rh < HID
    acc = al.zeros((BLOCK,), dtype=al.float32)
    for s in al.unroll(range(TOP_K)):
        row = al.cast(al.load(INV_PERM + t * TOP_K + s), al.int32)
        acc = acc + al.cast(
            al.load(PARTIAL + row * HID + rh, mask=hmask, other=0.0), al.float32
        )
    al.store(Y_OUT + t * HID + rh, acc, mask=hmask)


# --- Routing sort: ROUTING -> PERM / TILE_EXPERT / INV_PERM (the grouped layout) -------
# Counting sort of the chunk's T*TOP_K routings into per-expert PAD_M-padded segments.
# FILL-FREE by design: every output is OVERWRITTEN (computed by a per-program scan), so
# pad/empty positions are never zeroed; they hold recycled garbage but are never TRUSTED —
# `moe_row_tokens` derives per-row validity ANALYTICALLY (from the fully-overwritten
# COUNT/ROW_OFF/TOTAL_ROWS) and emits sanitized gather/scatter indices + weights (pads:
# clamped / sentinel / 0), and the grouped GEMMs' empty-tile tail is skipped via TOTAL_ROWS.
#   moe_sort_block_count      BLOCK_COUNT[b,e] = #{ts in block b: routing==e}  (grid NB*E, scan SORT_B)
#   moe_sort_count_from_blocks COUNT[e] = Σ_b BLOCK_COUNT[b,e]                 (grid E, scan NB)
#   moe_sort_offsets          prefix-sum of PAD_M-padded counts -> ROW_OFF + TOTAL_ROWS
#   moe_tile_expert           TILE_EXPERT[t] = expert owning tile t (grid MAX_TILES, scan E)
#   moe_sort_block_off        BLOCK_OFF[b,e] = ROW_OFF[e] + Σ_{b'<b} BLOCK_COUNT[b',e]  (grid E, scan NB)
#   moe_sort_perm_scan        rank = block-local #{ts'<ts: same expert} -> PERM[BLOCK_OFF+rank]=ts
#   moe_row_tokens            sanitized TOK_LD / TOK_ST / W_ROW per padded row (full overwrite)
#
# The rank computation is BLOCK-DECOMPOSED (block_count -> block_off -> perm_scan):
# a flat rank scan is O(R²) total work in 1-thread threadgroups — measured 25.3 ms/layer
# (23% of qwen3.6:35b prefill) at the production 4096-chunk (R = 32768), scaling 3.5-3.9×
# per R doubling. Splitting R into SORT_B-sized blocks makes it O(E·R + R·SORT_B/2) with
# 1024× the threadgroup parallelism, while staying stable (block-major + in-block order
# = ts order), deterministic, fill-free, and atomic-free (dispatch_plan-robust).


@al.kernel
def moe_sort_count_from_blocks(
    BLOCK_COUNT,
    ACTIVE_ROWS,
    COUNT: al.output,
    R_TOTAL: al.constexpr,
    SORT_B: al.constexpr,
    NUM_EXPERTS: al.constexpr,
    TOP_K: al.constexpr,
):
    """COUNT[e] = Σ_b BLOCK_COUNT[b,e] over the ACTIVE blocks. Replaces the
    retired moe_sort_count_scan (each of E programs re-scanned the full R
    routings: O(E·R) serial loads — 2.1 ms/layer at the 4096-chunk) with an
    O(E·NB) reduction of stage-1's block histograms. Grid (NUM_EXPERTS,);
    overwrite, no atomics.

    ACTIVE_ROWS is a (1,) int32 RUNTIME row bound — the chunk's m_pad under
    grid shrink, the bucket otherwise. The clamp is REQUIRED here, not just an
    optimization: under grid shrink the tail blocks' block_count programs never
    ran, so their BLOCK_COUNT entries are stale pool garbage — summing them
    would corrupt COUNT. min() against R_TOTAL keeps any unset/default value
    (huge) at the full-sum semantics; cdiv rounds the boundary block IN (its
    histogram is active-clamped by moe_sort_block_count itself)."""
    e = al.program_id(0)
    r_act = al.minimum(
        al.cast(al.load(ACTIVE_ROWS + 0), al.int32) * TOP_K,
        al.cast(R_TOTAL, al.int32),
    )
    nb_act = (r_act + (SORT_B - 1)) // SORT_B
    c = al.cast(0, al.int32)
    for b in range(nb_act):
        c = c + al.cast(al.load(BLOCK_COUNT + (b * NUM_EXPERTS + al.cast(e, al.int32))), al.int32)
    al.store(COUNT + e, c)


@al.kernel
def moe_sort_offsets(
    COUNT,
    ROW_OFF: al.output,          # (NUM_EXPERTS,) padded-row base per expert
    TOTAL_ROWS: al.output,       # (1,) total padded rows = active-tile boundary (rows >= this are empty)
    NUM_EXPERTS: al.constexpr,
    PAD_M: al.constexpr,         # padding granularity (== grouped-GEMM BLOCK_M); NOT a thread count
):
    """Exclusive prefix sum of PAD_M-padded per-expert counts -> row base, plus the
    total padded-row count. Active tiles are contiguous [0, TOTAL_ROWS/PAD_M); the
    grouped GEMMs skip rows >= TOTAL_ROWS (the static worst-case grid's empty tail).
    Grid (1,), one thread. Traced loop (NOT al.unroll): register-carried `acc` —
    unrolling 256 experts nests the running sum 256 deep (bracket-depth overflow)."""
    acc = al.cast(0, al.int32)
    for e in range(NUM_EXPERTS):
        c = al.cast(al.load(COUNT + e), al.int32)
        al.store(ROW_OFF + e, acc)
        n_rows = ((c + (PAD_M - 1)) // PAD_M) * PAD_M     # padded rows for this expert
        acc = acc + n_rows
    al.store(TOTAL_ROWS + 0, acc)


@al.kernel
def moe_tile_expert(
    ROW_OFF,
    COUNT,
    TILE_EXPERT: al.output,      # (MAX_TILES,) tile -> expert
    NUM_EXPERTS: al.constexpr,
    PAD_M: al.constexpr,
):
    """TILE_EXPERT[t] = the expert whose padded-row segment [ROW_OFF[e], ROW_OFF[e]+padded)
    contains row t*PAD_M. Grid (MAX_TILES,); scans the NUM_EXPERTS segments (overwrite).
    Empty-tail tiles match no segment -> 0 (harmless; those tiles are skipped via TOTAL_ROWS)."""
    t = al.program_id(0)
    row = al.cast(t, al.int32) * PAD_M
    e_found = al.cast(0, al.int32)
    for e in range(NUM_EXPERTS):
        ro = al.cast(al.load(ROW_OFF + e), al.int32)
        c = al.cast(al.load(COUNT + e), al.int32)
        n_rows = ((c + (PAD_M - 1)) // PAD_M) * PAD_M
        in_seg = (row >= ro) & (row < ro + n_rows)
        e_found = al.where(in_seg, al.cast(e, al.int32), e_found)
    al.store(TILE_EXPERT + t, e_found)


@al.kernel
def moe_sort_block_count(
    ROUTING,
    ACTIVE_ROWS,
    BLOCK_COUNT: al.output,      # (NB*NUM_EXPERTS,) [b*E + e] = #{ts in block b: routing==e}
    R_TOTAL: al.constexpr,
    SORT_B: al.constexpr,        # rank-scan block size (== perm_scan's SORT_B)
    NUM_EXPERTS: al.constexpr,
    TOP_K: al.constexpr,
):
    """Per-(block, expert) routing count — stage 1 of the block-decomposed rank.
    Grid (NB*NUM_EXPERTS,); program (b, e) scans only block b's SORT_B slots
    (overwrite, no atomics). The ACTIVE_ROWS clamp: slots >= active*TOP_K were
    never written by a shrunk router, so counting them would read pool garbage;
    clamping also keeps a boundary block's garbage tail out of the counts."""
    pid = al.program_id(0)
    b = al.cast(pid, al.int32) // NUM_EXPERTS
    e = al.cast(pid, al.int32) % NUM_EXPERTS
    r_act = al.minimum(
        al.cast(al.load(ACTIVE_ROWS + 0), al.int32) * TOP_K,
        al.cast(R_TOTAL, al.int32),
    )
    start = b * SORT_B
    n = al.maximum(al.minimum(start + SORT_B, r_act) - start, al.cast(0, al.int32))
    c = al.cast(0, al.int32)
    for j in range(n):
        c = c + al.where(al.cast(al.load(ROUTING + (start + j)), al.int32) == e,
                         al.cast(1, al.int32), al.cast(0, al.int32))
    al.store(BLOCK_COUNT + (b * NUM_EXPERTS + e), c)


@al.kernel
def moe_sort_block_off(
    BLOCK_COUNT,
    ROW_OFF,
    BLOCK_OFF: al.output,        # (NB*NUM_EXPERTS,) [b*E + e] = ROW_OFF[e] + Σ_{b'<b} BLOCK_COUNT[b',e]
    NB: al.constexpr,            # number of SORT_B blocks = ceil(R/SORT_B)
    NUM_EXPERTS: al.constexpr,
):
    """Per-expert exclusive prefix of the block counts, rebased at the expert's
    padded-segment start — stage 2. Grid (NUM_EXPERTS,); each program carries a
    register prefix over its expert's NB block counts (traced loop, register
    accumulator — same shape as moe_sort_offsets). Under grid shrink the tail
    blocks' BLOCK_COUNT is stale, but stale entries only affect BLOCK_OFF of
    LATER (inactive) blocks, which no shrunk perm_scan program reads."""
    e = al.program_id(0)
    acc = al.cast(al.load(ROW_OFF + e), al.int32)
    for b in range(NB):
        c = al.cast(al.load(BLOCK_COUNT + (b * NUM_EXPERTS + al.cast(e, al.int32))), al.int32)
        al.store(BLOCK_OFF + (b * NUM_EXPERTS + al.cast(e, al.int32)), acc)
        acc = acc + c


@al.kernel
def moe_sort_perm_scan(
    ROUTING,
    BLOCK_OFF,
    PERM: al.output,             # (MAX_ROWS,) padded-row -> (token,slot); pad rows left as-is (unread)
    INV_PERM: al.output,         # (T*TOP_K,) (token,slot) -> padded-row
    ROW_TOKEN: al.output,        # (MAX_ROWS,) padded-row -> source token (ts//TOP_K); pad rows unread
    TOP_K: al.constexpr,
    SORT_B: al.constexpr,        # rank-scan block size (== block_count's SORT_B)
    NUM_EXPERTS: al.constexpr,
):
    """Stable counting-sort scatter without atomics — stage 3. rank = the
    program's BLOCK-LOCAL #{ts' < ts: same expert} (a scan of at most SORT_B-1
    predecessors) + the cross-block prefix baked into BLOCK_OFF[b, e], so each
    (token,slot) lands at the same unique row the flat O(R²) scan produced
    (block-major + in-block order = ts order: bit-identical, still stable).
    Grid (T*TOP_K,); writes PERM[pos], INV_PERM[ts], ROW_TOKEN[pos] (overwrite).
    ROW_TOKEN folds the token = ts//TOP_K so the grouped GEMM can gather A rows
    directly (no a_perm)."""
    ts = al.program_id(0)
    e = al.cast(al.load(ROUTING + ts), al.int32)
    b = al.cast(ts, al.int32) // SORT_B
    base_j = b * SORT_B
    rank = al.cast(0, al.int32)
    local_n = al.cast(ts, al.int32) - base_j
    for j in range(local_n):
        rank = rank + al.where(al.cast(al.load(ROUTING + (base_j + j)), al.int32) == e,
                               al.cast(1, al.int32), al.cast(0, al.int32))
    pos = al.cast(al.load(BLOCK_OFF + (b * NUM_EXPERTS + e)), al.int32) + rank
    al.store(PERM + pos, al.cast(ts, al.int32))
    al.store(INV_PERM + ts, pos)
    al.store(ROW_TOKEN + pos, al.cast(ts // TOP_K, al.int32))


@al.kernel
def moe_row_tokens(
    ROW_TOKEN,                   # (MAX_ROWS,) raw scatter from perm_scan (pad rows = garbage)
    PERM,                        # (MAX_ROWS,) raw (token,slot) scatter (pad rows = garbage)
    WEIGHTS,                     # (T*TOP_K,) f32 routing weight per (token,slot)
    TILE_EXPERT,                 # (MAX_TILES,) tile -> expert (empty tail -> 0)
    ROW_OFF,                     # (NUM_EXPERTS,) padded-row base per expert
    COUNT,                       # (NUM_EXPERTS,) real (un-padded) routing count per expert
    TOTAL_ROWS,                  # (1,) active-row boundary
    TOK_LD: al.output,           # (MAX_ROWS,) gather-LOAD index: token, pad rows CLAMPED to 0
    TOK_ST: al.output,           # (MAX_ROWS,) scatter-STORE index: token, pad rows = huge sentinel
    W_ROW: al.output,            # (MAX_ROWS,) routing weight per sorted row, pad rows = 0.0
    PAD_M: al.constexpr,
    MAX_ROWS: al.constexpr,
    R_TOTAL: al.constexpr,       # T*TOP_K — clamp bound for the pad-row WEIGHTS read
    BLOCK: al.constexpr = 256,
):
    """Sanitize the fill-free sort's per-row indices/weights for in-kernel gather/scatter.

    Validity is ANALYTIC — `rm < TOTAL_ROWS` (active tiles) and `rm - ROW_OFF[e] < COUNT[e]`
    (inside the expert's real rows, not its segment-tail padding) — from buffers that are
    fully overwritten every call, so no fill pass is needed and pad-row garbage in
    ROW_TOKEN/PERM is never trusted. Loads and stores need opposite failure modes: TOK_LD
    clamps pads to row 0 (a harmless in-bounds read, masked from any store) so the gathered
    cooperative LOAD never dereferences out of bounds; TOK_ST maps pads to a huge sentinel
    so the scatter-accumulate STORE's `token < T_ROWS` guard drops them. W_ROW resolves the
    routing weight per sorted row HERE (a plain per-element kernel — the grouped GEMM's
    tile-addressed 1D gather of WEIGHTS[PERM[rm]] miscompiles to base+arange) with pads at
    0.0, so pad h rows are exactly zero. Full overwrite of all outputs (grid covers
    MAX_ROWS)."""
    pid = al.program_id(0)
    rm = pid * BLOCK + al.arange(0, BLOCK)
    m = rm < MAX_ROWS
    e = al.cast(al.load(TILE_EXPERT + rm // PAD_M, mask=m, other=0), al.int32)
    base = al.cast(al.load(ROW_OFF + e, mask=m, other=0), al.int32)
    cnt = al.cast(al.load(COUNT + e, mask=m, other=0), al.int32)
    total = al.cast(al.load(TOTAL_ROWS + 0), al.int32)
    raw = al.cast(al.load(ROW_TOKEN + rm, mask=m, other=0), al.int32)
    src = al.cast(al.load(PERM + rm, mask=m, other=0), al.int32)
    src_safe = al.minimum(al.maximum(src, al.cast(0, al.int32)), al.cast(R_TOTAL - 1, al.int32))
    w = al.load(WEIGHTS + src_safe, mask=m, other=0.0)
    valid = (rm < total) & ((rm - base) < cnt)
    al.store(TOK_LD + rm, al.where(valid, raw, al.cast(0, al.int32)), mask=m)
    al.store(TOK_ST + rm, al.where(valid, raw, al.cast(0x7FFFFFFF, al.int32)), mask=m)
    al.store(W_ROW + rm, al.where(valid, w, 0.0), mask=m)


@al.kernel
def moe_zero_f32(
    Y: al.output,                # (N,) buffer to clear
    N: al.constexpr,
    BLOCK: al.constexpr = 1024,
):
    """Zero-fill for the scatter-accumulate output: the fused down GEMM atomic-ADDs every
    (token, slot) contribution into Y, so Y must start at 0. Targets the same buffer as the
    down GEMM's output slot, so the WAW edge orders the zero before the accumulation."""
    pid = al.program_id(0)
    offs = pid * BLOCK + al.arange(0, BLOCK)
    al.store(Y + offs, al.zeros((BLOCK,), dtype=al.float32), mask=offs < N)
