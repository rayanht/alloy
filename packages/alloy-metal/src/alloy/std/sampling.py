"""Categorical sampling kernel — on-GPU temperature / top-k / top-p / min-p.

This replaces the decode argmax with a full sampler that runs *entirely inside
the compiled plan* (one kernel, no sampling work in Python). It writes the same
`(M,)` int64 token-id output slot that `argmax_last_dim` writes, so the decode
loop's token feedback keeps working untouched.

Two design choices make this sampler-state-free and sort-free:

* **Counter-based RNG.** The uniform for vocab entry `i` at decode step `pos`
  is `hash(seed, pos, i)`. `pos` is the `cache_position` the decode loop
  already advances every step, so reproducible *seeded* sampling needs
  no new per-step state — the counter is already there. Same (seed, prompt) →
  bit-identical stream.

* **Gumbel-max sampling.** A categorical draw from softmax(logits/T) equals
  `argmax_i(logits_i/T + g_i)` with `g_i = -log(-log(u_i))`. That is the *same
  reduction shape* as `argmax_last_dim`, so we reuse it instead of a sort or a
  CDF prefix-sum. top-k / top-p / min-p become a keep-mask: filtered-out entries
  get key `-inf`. The mask thresholds are found by in-kernel branchless binary
  search (count-based for top-k, mass-based for top-p) — a handful of
  bandwidth-bound passes over the vocab row, sub-millisecond against the ~ms
  transformer forward.

`params` layout (float32): `[temperature, top_p, top_k, min_p]`. Disabled
sentinels: `top_p >= 1`, `top_k < 1`, `min_p <= 0` (each lifts its constraint).

`temperature <= 0` is greedy: the kernel returns an exact argmax (raw logits,
zeroed Gumbel, lowest-index tie-break), so the decode path needs only ONE
compiled plan — greedy is just this kernel with temperature 0, and switching
between greedy and sampling is a write to the runtime `params` buffer, not a
recompile. The top-k / top-p bisections are gated behind `if`, so greedy and
pure-temperature decode pay nothing for filters they don't use.
"""

import alloy as al

# Odd 32-bit mixing constants (xxhash / murmur lineage). Wrapping uint32
# multiply is what gives the avalanche; Metal uint arithmetic wraps by default.
_M1 = 0x9E3779B1
_M2 = 0x85EBCA77
_M3 = 0xC2B2AE3D
_MIX_A = 0x7FEB352D
_MIX_B = 0x846CA68B

_NEG_INF = -1e30
_POS_INF = 1e30


def _rng_uniform(seed_u, pos_u, idx_u):
    """Counter-based uint32 hash -> uniform float in (0, 1). Pure function of
    (seed, position, vocab index): no state, reproducible, seed-controlled.

    Every step is wrapped in `cast(uint32)` to force 32-bit truncation: constants
    like 0x9E3779B1 exceed INT_MAX, so Metal types them as 64-bit `long` and the
    expression promotes to 64-bit — without truncation the xorshift mixing pulls in
    high garbage bits. The low 32 bits of a 64-bit op equal true uint32 arithmetic.
    """
    h = al.cast(
        al.cast(seed_u * _M1, al.uint32)
        + al.cast(pos_u * _M2, al.uint32)
        + al.cast(idx_u * _M3, al.uint32),
        al.uint32,
    )
    h = al.cast(h ^ (h >> 16), al.uint32)
    h = al.cast(h * _MIX_A, al.uint32)
    h = al.cast(h ^ (h >> 15), al.uint32)
    h = al.cast(h * _MIX_B, al.uint32)
    h = al.cast(h ^ (h >> 16), al.uint32)
    # Top 24 bits -> [0, 1); clamp off the open endpoints so -log(-log(u)) is finite.
    u = al.cast(h >> 8, al.float32) * (1.0 / 16777216.0)
    return al.clamp(u, 1e-7, 1.0 - 1e-7)


@al.kernel
def dropout_mask_apply(
    X,
    seed,
    out: al.output,
    mask_out: al.output,
    P: al.constexpr,
    SCALE: al.constexpr,
    OFFSET: al.constexpr,
    N: al.constexpr,
    BLOCK_SIZE: al.constexpr = 1024,
):
    """Inverted dropout: keep each element with prob (1-P), scale survivors by
    SCALE = 1/(1-P). The keep decision is the counter-RNG hash of (seed, OFFSET,
    element index): `seed` is a 1-element runtime buffer redrawn from the torch
    generator each forward (so `torch.manual_seed` controls it), OFFSET
    decorrelates stacked dropout layers. `mask_out` stores keep (1/0) for the
    backward."""
    pid = al.program_id(0)
    offs = pid * BLOCK_SIZE + al.arange(0, BLOCK_SIZE)
    m = offs < N
    seed_u = al.cast(al.load(seed + 0), al.uint32)
    u = _rng_uniform(seed_u, al.cast(OFFSET, al.uint32), al.cast(offs, al.uint32))
    keep = u >= P
    x = al.load(X + offs, mask=m)
    al.store(out + offs, al.where(keep, x * SCALE, 0.0), mask=m)
    al.store(mask_out + offs, al.where(keep, 1.0, 0.0), mask=m)


@al.kernel
def apply_token_bitmask(logits, bitmask, out: al.output, BLOCK_SIZE: al.constexpr = 512):
    """Constrained-decoding mask. `bitmask` is the xgrammar token bitmask —
    (M, ceil(N/32)) int32, row-major, bit (token % 32) of word (token // 32) set
    iff the token is allowed. Disallowed tokens are forced to -inf so the
    downstream sampler can never pick them. The grammar/matcher stays in Python;
    this kernel only consumes the bitmask buffer passed as an argument.
    """
    M, N = logits.shape
    row = al.program_id(0)
    words = (N + 31) >> 5  # int32 words per row of the bitmask
    for _ki in range(0, N, BLOCK_SIZE):
        offs = _ki + al.arange(0, BLOCK_SIZE)
        m = offs < N
        v = al.load(logits + row * N + offs, mask=m, other=0.0)
        word = al.load(bitmask + row * words + (offs >> 5), mask=m, other=0)
        # bit `offs & 31` of `word`; (word >> bit) & 1 reads it correctly for
        # signed int32 too (only the LSB after the shift matters).
        allowed = (word >> (offs & 31)) & 1
        al.store(out + row * N + offs, al.where(m & (allowed != 0), v, -1e30), mask=m)


@al.tunable(BLOCK_SIZE=[256, 512, 1024])
@al.kernel
def sample_categorical(
    logits,            # (M, N) float
    position,          # (M,) int64
    seed,              # (1,)  int64
    params,            # (>=4,) float32 — [temperature, top_p, top_k, min_p, ...]
    out: al.output,    # (M,)  int64
    BLOCK_SIZE: al.constexpr = 1024,
    TOPK_ITERS: al.constexpr = 24,
    TOPP_ITERS: al.constexpr = 24,
):
    """Single-threadgroup sampler (one program per row). Used by the per-step
    constrained loop and spec verify; the compiled decode/prefill plan uses the
    split + combine kernels below (bandwidth-bound). One greedy vocab pass (the
    argmax is fused into the max sweep via arg_lane)."""
    M, N = logits.shape
    row = al.program_id(0)

    temperature = al.load(params + 0)
    top_p = al.load(params + 1)
    top_k = al.load(params + 2)
    min_p = al.load(params + 3)
    pos_u = al.cast(al.load(position + row), al.uint32)
    seed_u = al.cast(al.load(seed + 0), al.uint32)

    do_sample = temperature > 0.0
    inv_t = al.where(do_sample, 1.0 / al.maximum(temperature, 1e-6), 1.0)

    s_max = _NEG_INF
    s_min = _POS_INF
    arg_lane = _POS_INF
    for _ki in range(0, N, BLOCK_SIZE):
        offs = _ki + al.arange(0, BLOCK_SIZE)
        mask = offs < N
        s = al.load(logits + row * N + offs, mask=mask, other=0.0) * inv_t
        sm = al.where(mask, s, _NEG_INF)
        offs_f = al.cast(offs, al.float32)
        gt = sm > s_max
        eq = sm == s_max
        arg_lane = al.where(gt, offs_f, al.where(eq, al.minimum(arg_lane, offs_f), arg_lane))
        s_max = al.maximum(s_max, sm)
        s_min = al.minimum(s_min, al.where(mask, s, _POS_INF))
    g_max = al.max(s_max)
    greedy_idx = al.min(al.where(s_max == g_max, arg_lane, _POS_INF))
    s_max = g_max
    s_min = al.min(s_min)

    topk_thresh = _NEG_INF
    if top_k >= 1.0:
        lo_k = s_min
        hi_k = s_max
        for _it in range(TOPK_ITERS):
            mid = (lo_k + hi_k) * 0.5
            cnt = 0.0
            for _ki in range(0, N, BLOCK_SIZE):
                offs = _ki + al.arange(0, BLOCK_SIZE)
                mask = offs < N
                s = al.load(logits + row * N + offs, mask=mask, other=0.0) * inv_t
                cnt = cnt + al.cast(mask & (s >= mid), al.float32)
            too_many = al.sum(cnt) > top_k
            lo_k = al.where(too_many, mid, lo_k)
            hi_k = al.where(too_many, hi_k, mid)
        topk_thresh = hi_k

    topp_thresh = _NEG_INF
    if top_p < 1.0:
        z_acc = 0.0
        for _ki in range(0, N, BLOCK_SIZE):
            offs = _ki + al.arange(0, BLOCK_SIZE)
            mask = offs < N
            s = al.load(logits + row * N + offs, mask=mask, other=0.0) * inv_t
            z_acc = z_acc + al.where(mask, al.exp(s - s_max), 0.0)
        z_target = al.sum(z_acc) * top_p
        lo_p = s_min
        hi_p = s_max
        for _it in range(TOPP_ITERS):
            mid = (lo_p + hi_p) * 0.5
            msum = 0.0
            for _ki in range(0, N, BLOCK_SIZE):
                offs = _ki + al.arange(0, BLOCK_SIZE)
                mask = offs < N
                s = al.load(logits + row * N + offs, mask=mask, other=0.0) * inv_t
                msum = msum + al.where(mask & (s >= mid), al.exp(s - s_max), 0.0)
            enough = al.sum(msum) >= z_target
            lo_p = al.where(enough, mid, lo_p)
            hi_p = al.where(enough, hi_p, mid)
        topp_thresh = lo_p

    minp_thresh = _NEG_INF
    if min_p > 0.0:
        minp_thresh = s_max + al.log(al.maximum(min_p, 1e-30))

    keep_thresh = al.maximum(al.maximum(topk_thresh, topp_thresh), minp_thresh)

    if do_sample:
        key_max = _NEG_INF
        for _ki in range(0, N, BLOCK_SIZE):
            offs = _ki + al.arange(0, BLOCK_SIZE)
            mask = offs < N
            s = al.load(logits + row * N + offs, mask=mask, other=0.0) * inv_t
            u = _rng_uniform(seed_u, pos_u, al.cast(offs, al.uint32))
            g = -al.log(-al.log(u))
            key = al.where(mask & (s >= keep_thresh), s + g, _NEG_INF)
            key_max = al.maximum(key_max, key)
        key_max = al.max(key_max)

        best_idx = _POS_INF
        for _ki in range(0, N, BLOCK_SIZE):
            offs = _ki + al.arange(0, BLOCK_SIZE)
            mask = offs < N
            s = al.load(logits + row * N + offs, mask=mask, other=0.0) * inv_t
            u = _rng_uniform(seed_u, pos_u, al.cast(offs, al.uint32))
            g = -al.log(-al.log(u))
            key = al.where(mask & (s >= keep_thresh), s + g, _NEG_INF)
            cand = al.where(mask & (key == key_max), al.cast(offs, al.float32), _POS_INF)
            best_idx = al.minimum(best_idx, cand)
        al.store(out + row, al.cast(al.min(best_idx), al.int64))
    else:
        al.store(out + row, al.cast(greedy_idx, al.int64))


# Vocab-split count for the decode sampler. The argmax/Gumbel reduction over the
# ~150k-entry vocab is split across this many threadgroups (a single TG can't
# saturate memory bandwidth — it was the ~40-70us/token bottleneck). `params[4]`
# selects the ACTIVE split count at runtime, per request: top-k/p/min-p needs a
# GLOBAL threshold bisection that can't be split, so those set n_splits=1 (split
# 0 covers the whole vocab); greedy / pure-temperature set n_splits=SAMPLE_SPLITS
# for the bandwidth-bound parallel argmax. Idle splits get an empty slice and
# emit a sentinel partial. (Per-request via the same params buffer that already
# carries temperature/top-k — no recompile, no separate plan.)
SAMPLE_SPLITS = 64


@al.tunable(BLOCK_SIZE=[256, 512, 1024])
@al.kernel
def sample_categorical_split(
    logits,            # (M, N) float  — last-position logits, one row per sequence
    position,          # (M,) int64    — per-row RNG counter (cache_position). Every
                       #   M=1 caller (decode/prefill/constrained) passes its (1,)
                       #   cache_position unchanged; the M-row spec verify passes the
                       #   rows' absolute positions, so row j samples with the SAME
                       #   counter plain decode would use at that position.
    seed,              # (1,)  int64   — base RNG seed for this generation
    params,            # (5,)  float32 — [temperature, top_p, top_k, min_p, n_splits]
    partial_val: al.output,  # (M, SP) float32 — this split's winning key value
    partial_idx: al.output,  # (M, SP) float32 — this split's winning vocab index
    BLOCK_SIZE: al.constexpr = 1024,
    TOPK_ITERS: al.constexpr = 24,
    TOPP_ITERS: al.constexpr = 24,
):
    M, N = logits.shape
    SP = partial_val.shape[1]
    row = al.program_id(0)
    sp = al.program_id(1)

    temperature = al.load(params + 0)
    top_p = al.load(params + 1)
    top_k = al.load(params + 2)
    min_p = al.load(params + 3)
    n_splits = al.cast(al.load(params + 4), al.int32)
    pos_u = al.cast(al.load(position + row), al.uint32)
    seed_u = al.cast(al.load(seed + 0), al.uint32)

    # temperature <= 0 is greedy: operate on raw logits (inv_t = 1) and zero the
    # Gumbel noise below, so the result is bit-identical to argmax_last_dim
    # (lowest-index tie-break). Sampling scales by 1/T.
    do_sample = temperature > 0.0
    inv_t = al.where(do_sample, 1.0 / al.maximum(temperature, 1e-6), 1.0)

    # This split's contiguous vocab slice [start, end). At n_splits==1 split 0
    # covers the whole vocab (so the top-k/p bisection below is global/correct)
    # and splits >=1 get start>=N -> empty -> sentinel partial. At n_splits==SP
    # the slices tile the vocab for a bandwidth-bound parallel reduction.
    chunk = (N + n_splits - 1) // n_splits
    start = sp * chunk
    end = al.minimum(start + chunk, N)

    # --- Pass 1: slice extremes (search bounds) + greedy argmax in one sweep.
    # arg_lane tracks, per lane, the lowest offset attaining the lane's running
    # max; after the reduce the slice argmax is the min over lanes hitting it. ---
    s_max = _NEG_INF
    s_min = _POS_INF
    arg_lane = _POS_INF
    for _ki in range(start, end, BLOCK_SIZE):
        offs = _ki + al.arange(0, BLOCK_SIZE)
        mask = offs < end
        s = al.load(logits + row * N + offs, mask=mask, other=0.0) * inv_t
        sm = al.where(mask, s, _NEG_INF)
        offs_f = al.cast(offs, al.float32)
        gt = sm > s_max
        eq = sm == s_max
        arg_lane = al.where(gt, offs_f, al.where(eq, al.minimum(arg_lane, offs_f), arg_lane))
        s_max = al.maximum(s_max, sm)
        s_min = al.minimum(s_min, al.where(mask, s, _POS_INF))
    g_max = al.max(s_max)
    greedy_idx = al.min(al.where(s_max == g_max, arg_lane, _POS_INF))
    s_max = g_max
    s_min = al.min(s_min)

    # --- top-k threshold (only when enabled): largest t with count(s >= t) <= k.
    # Gated; only reached at n_splits==1 (slice == whole vocab). ---
    topk_thresh = _NEG_INF
    if top_k >= 1.0:
        lo_k = s_min
        hi_k = s_max
        for _it in range(TOPK_ITERS):
            mid = (lo_k + hi_k) * 0.5
            cnt = 0.0
            for _ki in range(start, end, BLOCK_SIZE):
                offs = _ki + al.arange(0, BLOCK_SIZE)
                mask = offs < end
                s = al.load(logits + row * N + offs, mask=mask, other=0.0) * inv_t
                cnt = cnt + al.cast(mask & (s >= mid), al.float32)
            too_many = al.sum(cnt) > top_k
            lo_k = al.where(too_many, mid, lo_k)
            hi_k = al.where(too_many, hi_k, mid)
        topk_thresh = hi_k

    # --- top-p threshold (only when enabled): largest t with mass(s >= t) >= p. ---
    topp_thresh = _NEG_INF
    if top_p < 1.0:
        z_acc = 0.0
        for _ki in range(start, end, BLOCK_SIZE):
            offs = _ki + al.arange(0, BLOCK_SIZE)
            mask = offs < end
            s = al.load(logits + row * N + offs, mask=mask, other=0.0) * inv_t
            z_acc = z_acc + al.where(mask, al.exp(s - s_max), 0.0)
        z_target = al.sum(z_acc) * top_p
        lo_p = s_min
        hi_p = s_max
        for _it in range(TOPP_ITERS):
            mid = (lo_p + hi_p) * 0.5
            msum = 0.0
            for _ki in range(start, end, BLOCK_SIZE):
                offs = _ki + al.arange(0, BLOCK_SIZE)
                mask = offs < end
                s = al.load(logits + row * N + offs, mask=mask, other=0.0) * inv_t
                msum = msum + al.where(mask & (s >= mid), al.exp(s - s_max), 0.0)
            enough = al.sum(msum) >= z_target
            lo_p = al.where(enough, mid, lo_p)
            hi_p = al.where(enough, hi_p, mid)
        topp_thresh = lo_p

    # --- min-p threshold (only when enabled): s_i >= s_max + log(min_p). ---
    minp_thresh = _NEG_INF
    if min_p > 0.0:
        minp_thresh = s_max + al.log(al.maximum(min_p, 1e-30))

    keep_thresh = al.maximum(al.maximum(topk_thresh, topp_thresh), minp_thresh)

    if do_sample:
        # --- Gumbel-max over the keep-set within this slice: argmax_i(s_i + g_i).
        key_max = _NEG_INF
        for _ki in range(start, end, BLOCK_SIZE):
            offs = _ki + al.arange(0, BLOCK_SIZE)
            mask = offs < end
            s = al.load(logits + row * N + offs, mask=mask, other=0.0) * inv_t
            u = _rng_uniform(seed_u, pos_u, al.cast(offs, al.uint32))
            g = -al.log(-al.log(u))
            key = al.where(mask & (s >= keep_thresh), s + g, _NEG_INF)
            key_max = al.maximum(key_max, key)
        key_max = al.max(key_max)

        best_idx = _POS_INF
        for _ki in range(start, end, BLOCK_SIZE):
            offs = _ki + al.arange(0, BLOCK_SIZE)
            mask = offs < end
            s = al.load(logits + row * N + offs, mask=mask, other=0.0) * inv_t
            u = _rng_uniform(seed_u, pos_u, al.cast(offs, al.uint32))
            g = -al.log(-al.log(u))
            key = al.where(mask & (s >= keep_thresh), s + g, _NEG_INF)
            cand = al.where(mask & (key == key_max), al.cast(offs, al.float32), _POS_INF)
            best_idx = al.minimum(best_idx, cand)
        al.store(partial_val + row * SP + sp, key_max)
        al.store(partial_idx + row * SP + sp, al.min(best_idx))
    else:
        al.store(partial_val + row * SP + sp, g_max)
        al.store(partial_idx + row * SP + sp, greedy_idx)


@al.kernel
def sample_categorical_combine(
    partial_val,       # (M, SP) float32 — per-split winning key
    partial_idx,       # (M, SP) float32 — per-split winning vocab index
    out: al.output,    # (M,)    int64   — sampled token id
):
    """Reduce the per-split partials to the final token: the global max key, with
    the lowest vocab index on a tie (slices are vocab-ordered, so the per-split
    lowest-index reduce + this min make it bit-identical to a single sweep).
    Empty splits carry _NEG_INF and lose."""
    M, SP = partial_val.shape
    row = al.program_id(0)
    j = al.arange(0, SP)
    v = al.load(partial_val + row * SP + j)
    idx = al.load(partial_idx + row * SP + j)
    v_max = al.max(v)
    best = al.min(al.where(v == v_max, idx, _POS_INF))
    al.store(out + row, al.cast(best, al.int64))
