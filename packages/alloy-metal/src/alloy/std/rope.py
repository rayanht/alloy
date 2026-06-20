"""Rotary embedding kernels."""

import math

import alloy as al


@al.kernel
def rope(x, out: al.output, BLOCK_SIZE: al.constexpr = 256, BASE: al.constexpr = 10000):
    M, D = x.shape
    HALF_D = D // 2
    INV_BASE_LOG2 = math.log2(1.0 / BASE)
    pos = al.program_id(0)
    for _ki in range(0, HALF_D, BLOCK_SIZE):
        j_offs = _ki + al.arange(0, BLOCK_SIZE)
        mask = j_offs < HALF_D
        two_j = j_offs * 2
        inv_freq = al.exp2(two_j * INV_BASE_LOG2 / D)
        angle = pos * inv_freq
        cos_a = al.cos(angle)
        sin_a = al.sin(angle)
        x_even = al.load(x + pos * D + two_j, mask=mask, other=0.0)
        x_odd = al.load(x + pos * D + two_j + 1, mask=mask, other=0.0)
        al.store(out + pos * D + two_j, x_even * cos_a - x_odd * sin_a, mask=mask)
        al.store(out + pos * D + two_j + 1, x_even * sin_a + x_odd * cos_a, mask=mask)


@al.kernel
def rope_cos_sin(
    cache_position,
    inv_freq,
    cos_out: al.output,
    sin_out: al.output,
    HALF_D: al.constexpr,
    BLOCK_SIZE: al.constexpr = 256,
):
    """Rotary cos/sin table for `SEQ_LEN` positions starting at cache_position.

    Collapses HF's `(arange + cache_position) -> cast -> ·inv_freq -> cos/sin`
    into one kernel: one threadgroup per position computes
    `angle = float(m + cache_position) * inv_freq[j]` then cos/sin (integer
    position add, then the f32 multiply).

    cache_position: (1,) int — the chunk's start position.
    inv_freq:       (HALF_D,) f32.
    cos_out/sin_out: (SEQ_LEN, HALF_D) f32 (one row per position).
    """
    m = al.program_id(0)
    cp = al.cast(al.load(cache_position), al.int32)
    pos = al.cast(m + cp, al.float32)
    for _j in range(0, HALF_D, BLOCK_SIZE):
        offs = _j + al.arange(0, BLOCK_SIZE)
        mask = offs < HALF_D
        f = al.cast(al.load(inv_freq + offs, mask=mask, other=0.0), al.float32)
        angle = pos * f
        al.store(cos_out + m * HALF_D + offs, al.cos(angle), mask=mask)
        al.store(sin_out + m * HALF_D + offs, al.sin(angle), mask=mask)


@al.kernel
def rope_apply(
    x, cos, sin, out: al.output, COS_ROWS: al.constexpr = 0, BLOCK_SIZE: al.constexpr = 256
):
    """Apply precomputed rotary embeddings: out = x * cos + rotate_half(x) * sin.

    x: (M, D) — flattened (B*H*S, D) or (S, D)
    cos, sin: (COS_ROWS, D) or (M, D)
    When COS_ROWS > 0, uses modular indexing: cos_row = row % COS_ROWS.
    """
    M, D = x.shape
    HALF_D = D // 2
    row = al.program_id(0)
    cos_row = row % COS_ROWS if COS_ROWS > 0 else row
    for _ki in range(0, HALF_D, BLOCK_SIZE):
        offs = _ki + al.arange(0, BLOCK_SIZE)
        mask = offs < HALF_D
        x1 = al.load(x + row * D + offs, mask=mask, other=0.0)
        x2 = al.load(x + row * D + HALF_D + offs, mask=mask, other=0.0)
        c1 = al.load(cos + cos_row * D + offs, mask=mask, other=0.0)
        s1 = al.load(sin + cos_row * D + offs, mask=mask, other=0.0)
        c2 = al.load(cos + cos_row * D + HALF_D + offs, mask=mask, other=0.0)
        s2 = al.load(sin + cos_row * D + HALF_D + offs, mask=mask, other=0.0)
        al.store(out + row * D + offs, x1 * c1 - x2 * s1, mask=mask)
        al.store(out + row * D + HALF_D + offs, x2 * c2 + x1 * s2, mask=mask)


@al.kernel
def rope_apply_backward(
    dout, cos, sin, dx: al.output, COS_ROWS: al.constexpr = 0, BLOCK_SIZE: al.constexpr = 256
):
    """Backward of rope_apply.

    Forward: out = (x1*c1 - x2*s1, x2*c2 + x1*s2)
    Backward: dx = (dout1*c1 + dout2*s2, dout2*c2 - dout1*s1)

    dout: (M, D) flattened (B*H*S, D); dx: same.
    cos, sin: (COS_ROWS, D) with row % COS_ROWS broadcast indexing.
    """
    M, D = dout.shape
    HALF_D = D // 2
    row = al.program_id(0)
    cos_row = row % COS_ROWS if COS_ROWS > 0 else row
    for _ki in range(0, HALF_D, BLOCK_SIZE):
        offs = _ki + al.arange(0, BLOCK_SIZE)
        mask = offs < HALF_D
        d1 = al.load(dout + row * D + offs, mask=mask, other=0.0)
        d2 = al.load(dout + row * D + HALF_D + offs, mask=mask, other=0.0)
        c1 = al.load(cos + cos_row * D + offs, mask=mask, other=0.0)
        s1 = al.load(sin + cos_row * D + offs, mask=mask, other=0.0)
        c2 = al.load(cos + cos_row * D + HALF_D + offs, mask=mask, other=0.0)
        s2 = al.load(sin + cos_row * D + HALF_D + offs, mask=mask, other=0.0)
        al.store(dx + row * D + offs, d1 * c1 + d2 * s2, mask=mask)
        al.store(dx + row * D + HALF_D + offs, d2 * c2 - d1 * s1, mask=mask)


@al.kernel
def rope_apply_backward_strided(
    dout,
    cos,
    sin,
    dx: al.output,
    BH: al.constexpr,
    HEADS_PER_BATCH: al.constexpr = 1,
    SEQ_LEN: al.constexpr = 1,
    HEAD_DIM: al.constexpr = 64,
    D_OFFSET: al.constexpr = 0,
    D_BATCH_STRIDE: al.constexpr = 0,
    D_HEAD_STRIDE: al.constexpr = 0,
    D_SEQ_STRIDE: al.constexpr = 0,
    COS_ROWS: al.constexpr = 0,
    BLOCK_SIZE: al.constexpr = 256,
):
    """RoPE backward with strided 4D dout. Output dx is BSHD-flat (B*S*H, D):
    row = (batch_idx * S + s) * H + head_idx. The downstream chain
    `permute([0,2,1,3]).view(B*S, H*D).mm(W)` then operates on a tensor that is
    BSHD-contiguous after the permute, so AOT autograd skips the clone."""
    HALF_D = HEAD_DIM // 2
    seq_pos = al.program_id(0)
    bh = al.program_id(1)
    batch_idx = bh // HEADS_PER_BATCH
    head_idx = bh % HEADS_PER_BATCH
    d_base = (
        D_OFFSET + batch_idx * D_BATCH_STRIDE + head_idx * D_HEAD_STRIDE + seq_pos * D_SEQ_STRIDE
    )
    out_row = (batch_idx * SEQ_LEN + seq_pos) * HEADS_PER_BATCH + head_idx
    cos_row = seq_pos % COS_ROWS if COS_ROWS > 0 else seq_pos

    for _ki in range(0, HALF_D, BLOCK_SIZE):
        offs = _ki + al.arange(0, BLOCK_SIZE)
        mask = offs < HALF_D
        d1 = al.load(dout + d_base + offs, mask=mask, other=0.0)
        d2 = al.load(dout + d_base + HALF_D + offs, mask=mask, other=0.0)
        c1 = al.load(cos + cos_row * HEAD_DIM + offs, mask=mask, other=0.0)
        s1 = al.load(sin + cos_row * HEAD_DIM + offs, mask=mask, other=0.0)
        c2 = al.load(cos + cos_row * HEAD_DIM + HALF_D + offs, mask=mask, other=0.0)
        s2 = al.load(sin + cos_row * HEAD_DIM + HALF_D + offs, mask=mask, other=0.0)
        al.store(dx + out_row * HEAD_DIM + offs, d1 * c1 + d2 * s2, mask=mask)
        al.store(dx + out_row * HEAD_DIM + HALF_D + offs, d2 * c2 - d1 * s1, mask=mask)


@al.kernel
def rms_norm_rope_strided(
    x,
    weight,
    cos,
    sin,
    out: al.output,
    BH: al.constexpr,
    HEADS_PER_BATCH: al.constexpr = 1,
    SEQ_LEN: al.constexpr = 1,
    HEAD_DIM: al.constexpr = 64,
    ROTARY_DIM: al.constexpr = 0,
    X_OFFSET: al.constexpr = 0,
    X_BATCH_STRIDE: al.constexpr = 0,
    X_HEAD_STRIDE: al.constexpr = 0,
    X_SEQ_STRIDE: al.constexpr = 0,
    COS_ROWS: al.constexpr = 0,
    EPS: al.constexpr = 1e-6,
    HALF_COS: al.constexpr = 0,
    BLOCK_SIZE: al.constexpr = 128,
):
    """Fused RMSNorm + RoPE for Qwen3-style per-head q/k norm before rotary.

    Replaces (rms_norm strided over head_dim) → (rope_apply_strided) with one
    dispatch. Per-head reduction stays in fp32; the cast-to-weight-dtype
    precision trick from rms_norm is preserved so output matches eager.

    ROTARY_DIM enables partial rotary (Qwen3.5: rotary_dim 64 of head_dim 256):
    the RMS norm covers the full HEAD_DIM, rope rotates only the leading
    ROTARY_DIM dims (cos/sin are laid out (COS_ROWS, ROTARY_DIM)), and the
    trailing HEAD_DIM-ROTARY_DIM dims pass through normalized but un-rotated.
    ROTARY_DIM == 0 means full rotary (ROT = HEAD_DIM).

    HALF_COS != 0 means cos/sin are stored at HALF width (HALF_ROT per row): the
    rotate_half layout makes emb = cat(freqs, freqs), so the two halves of the
    table are identical. The self-cat strip rewrite drops the duplication and
    sets this flag, so the table stride is HALF_ROT and the "second half"
    (c2/s2) re-reads the first (same value).

    Grid: (SEQ_LEN, BH). Each TG processes one (head, seq) row, reads
    head_dim values from strided input, computes rms, normalizes by weight,
    and applies rotary rotation. Output is contiguous (BH*SEQ_LEN, HEAD_DIM).
    """
    ROT = ROTARY_DIM if ROTARY_DIM > 0 else HEAD_DIM
    HALF_ROT = ROT // 2
    PASS = HEAD_DIM - ROT
    # Half-width table: row stride is HALF_ROT and the second half re-reads the
    # first (offset 0); full table: stride ROT, second half at +HALF_ROT.
    COS_STRIDE = HALF_ROT if HALF_COS else ROT
    COS_SECOND = 0 if HALF_COS else HALF_ROT
    seq_pos = al.program_id(0)
    bh = al.program_id(1)
    batch_idx = bh // HEADS_PER_BATCH
    head_idx = bh % HEADS_PER_BATCH

    x_base = (
        X_OFFSET + batch_idx * X_BATCH_STRIDE + head_idx * X_HEAD_STRIDE + seq_pos * X_SEQ_STRIDE
    )
    out_row = bh * SEQ_LEN + seq_pos
    cos_row = seq_pos % COS_ROWS if COS_ROWS > 0 else seq_pos

    # Phase 1: sum-of-squares over the full head_dim.
    sq_sum = 0.0
    for _ki in range(0, HEAD_DIM, BLOCK_SIZE):
        offs = _ki + al.arange(0, BLOCK_SIZE)
        mask = offs < HEAD_DIM
        v = al.cast(al.load(x + x_base + offs, mask=mask, other=0.0), al.float32)
        sq_sum = sq_sum + v * v
    rrms = al.rsqrt(al.sum(sq_sum) / HEAD_DIM + EPS)

    # Phase 2a: re-read the leading ROT dims, rms-normalize with weight, apply
    # paired-half rope. We iterate over HALF_ROT so each lane reads both halves
    # of the rotary band. cos/sin are strided by ROT (partial-rotary table).
    for _ki in range(0, HALF_ROT, BLOCK_SIZE):
        offs = _ki + al.arange(0, BLOCK_SIZE)
        mask = offs < HALF_ROT

        x1 = al.cast(al.load(x + x_base + offs, mask=mask, other=0.0), al.float32)
        x2 = al.cast(al.load(x + x_base + HALF_ROT + offs, mask=mask, other=0.0), al.float32)
        w1 = al.load(weight + offs, mask=mask, other=0.0)
        w2 = al.load(weight + HALF_ROT + offs, mask=mask, other=0.0)
        # Match rms_norm precision: truncate normalized value to weight dtype
        # before the weight multiply (cast back to f32 here for the rope math).
        n1 = al.cast(al.cast(x1 * rrms, w1.dtype) * w1, al.float32)
        n2 = al.cast(al.cast(x2 * rrms, w2.dtype) * w2, al.float32)

        c1 = al.load(cos + cos_row * COS_STRIDE + offs, mask=mask, other=0.0)
        s1 = al.load(sin + cos_row * COS_STRIDE + offs, mask=mask, other=0.0)
        c2 = al.load(cos + cos_row * COS_STRIDE + COS_SECOND + offs, mask=mask, other=0.0)
        s2 = al.load(sin + cos_row * COS_STRIDE + COS_SECOND + offs, mask=mask, other=0.0)
        al.store(out + out_row * HEAD_DIM + offs, n1 * c1 - n2 * s1, mask=mask)
        al.store(out + out_row * HEAD_DIM + HALF_ROT + offs, n2 * c2 + n1 * s2, mask=mask)

    # Phase 2b: pass-through the trailing dims (partial rotary), normalized but
    # un-rotated. Empty loop when ROTARY_DIM == 0 (full rotary).
    for _ki in range(0, PASS, BLOCK_SIZE):
        offs = _ki + al.arange(0, BLOCK_SIZE)
        mask = offs < PASS
        idx = ROT + offs
        xv = al.cast(al.load(x + x_base + idx, mask=mask, other=0.0), al.float32)
        wv = al.load(weight + idx, mask=mask, other=0.0)
        nv = al.cast(al.cast(xv * rrms, wv.dtype) * wv, al.float32)
        al.store(out + out_row * HEAD_DIM + idx, nv, mask=mask)


@al.kernel
def rope_apply_strided(
    x,
    cos,
    sin,
    out: al.output,
    BH: al.constexpr,
    HEADS_PER_BATCH: al.constexpr = 1,
    SEQ_LEN: al.constexpr = 1,
    HEAD_DIM: al.constexpr = 64,
    X_OFFSET: al.constexpr = 0,
    X_BATCH_STRIDE: al.constexpr = 0,
    X_HEAD_STRIDE: al.constexpr = 0,
    X_SEQ_STRIDE: al.constexpr = 0,
    COS_ROWS: al.constexpr = 0,
    BLOCK_SIZE: al.constexpr = 256,
):
    """RoPE with strided 4D input. Reads x at (batch, head, seq) via stride params.

    Grid: (SEQ_LEN, BH). Output is contiguous (BH*SEQ_LEN, HEAD_DIM).
    cos/sin: (COS_ROWS, HEAD_DIM) with modular indexing.
    """
    HALF_D = HEAD_DIM // 2
    seq_pos = al.program_id(0)
    bh = al.program_id(1)
    batch_idx = bh // HEADS_PER_BATCH
    head_idx = bh % HEADS_PER_BATCH

    # Input offset for this (batch, head, seq) position
    x_base = (
        X_OFFSET + batch_idx * X_BATCH_STRIDE + head_idx * X_HEAD_STRIDE + seq_pos * X_SEQ_STRIDE
    )

    # Output row in flattened (BH*S, D) layout
    out_row = bh * SEQ_LEN + seq_pos

    # cos/sin row (modular for broadcast)
    cos_row = seq_pos % COS_ROWS if COS_ROWS > 0 else seq_pos

    for _ki in range(0, HALF_D, BLOCK_SIZE):
        offs = _ki + al.arange(0, BLOCK_SIZE)
        mask = offs < HALF_D
        x1 = al.load(x + x_base + offs, mask=mask, other=0.0)
        x2 = al.load(x + x_base + HALF_D + offs, mask=mask, other=0.0)
        c1 = al.load(cos + cos_row * HEAD_DIM + offs, mask=mask, other=0.0)
        s1 = al.load(sin + cos_row * HEAD_DIM + offs, mask=mask, other=0.0)
        c2 = al.load(cos + cos_row * HEAD_DIM + HALF_D + offs, mask=mask, other=0.0)
        s2 = al.load(sin + cos_row * HEAD_DIM + HALF_D + offs, mask=mask, other=0.0)
        al.store(out + out_row * HEAD_DIM + offs, x1 * c1 - x2 * s1, mask=mask)
        al.store(out + out_row * HEAD_DIM + HALF_D + offs, x2 * c2 + x1 * s2, mask=mask)
