"""Indexing, im2col, and concat kernels."""

import alloy as al


@al.kernel
def im2col_1d(
    x,
    col: al.output,
    IN_C: al.constexpr,
    IN_LEN: al.constexpr,
    OUT_LEN: al.constexpr,
    CK: al.constexpr,
    K: al.constexpr,
    STRIDE: al.constexpr,
    PADDING: al.constexpr,
):
    """Unfold input windows for conv1d: col[b, t, ic*K+ki] = x[b, ic, t*stride+ki-pad].

    x: (batch, IN_C * IN_LEN) flattened
    col: (batch * OUT_LEN, CK) flattened — im2col matrix for GEMM
    Grid: (batch * OUT_LEN,).
    """
    idx = al.program_id(0)
    b = idx // OUT_LEN
    t = idx % OUT_LEN
    x_base = b * (IN_C * IN_LEN)
    col_base = idx * CK

    for ic in range(IN_C):
        for ki in range(K):
            in_pos = t * STRIDE + ki - PADDING
            if in_pos >= 0:
                if in_pos < IN_LEN:
                    val = al.load(x + x_base + ic * IN_LEN + in_pos)
                    al.store(col + col_base + ic * K + ki, val)
                else:
                    al.store(col + col_base + ic * K + ki, 0.0)
            else:
                al.store(col + col_base + ic * K + ki, 0.0)


@al.kernel
def depthwise_conv1d(
    x,
    w,
    out: al.output,
    BATCH: al.constexpr,
    C: al.constexpr,
    IN_LEN: al.constexpr,
    OUT_LEN: al.constexpr,
    K: al.constexpr,
    STRIDE: al.constexpr,
    PADDING: al.constexpr,
):
    """Depthwise Conv1d: one kernel per channel, no cross-channel mixing.

    x:   (BATCH, C, IN_LEN), flat
    w:   (C, K), flat — weight[c, k] applied to x[b, c, l*STRIDE+k-PADDING]
    out: (BATCH, C, OUT_LEN), flat — out[b, c, l] = sum_k w[c,k] * x[b, c, l*STRIDE+k-PADDING]

    Grid: (BATCH * C * OUT_LEN,) — one thread per output element. K is
    typically tiny (4 in GatedDeltaNet) so the inner sum is fully
    unrolled by the MSL emitter.
    """
    idx = al.program_id(0)
    bc_l = idx
    l = bc_l % OUT_LEN
    bc = bc_l // OUT_LEN
    c = bc % C
    b = bc // C
    x_base = b * (C * IN_LEN) + c * IN_LEN
    w_base = c * K
    acc = 0.0
    for ki in range(K):
        in_pos = l * STRIDE + ki - PADDING
        if in_pos >= 0:
            if in_pos < IN_LEN:
                xv = al.load(x + x_base + in_pos)
                wv = al.load(w + w_base + ki)
                acc = acc + xv * wv
    al.store(out + b * (C * OUT_LEN) + c * OUT_LEN + l, acc)


@al.kernel
def im2col_2d(
    x,
    col: al.output,
    IN_C: al.constexpr,
    IN_H: al.constexpr,
    IN_W: al.constexpr,
    OUT_H: al.constexpr,
    OUT_W: al.constexpr,
    KH: al.constexpr,
    KW: al.constexpr,
    CKK: al.constexpr,
    STRIDE_H: al.constexpr,
    STRIDE_W: al.constexpr,
    PAD_H: al.constexpr,
    PAD_W: al.constexpr,
    NUM_THREADS: al.constexpr = 256,
    X_BATCH_STRIDE: al.constexpr = 0,
    X_C_STRIDE: al.constexpr = 0,
    X_H_STRIDE: al.constexpr = 0,
    X_W_STRIDE: al.constexpr = 1,
    X_OFFSET: al.constexpr = 0,
):
    """Unfold 2D input patches for conv2d: im2col matrix for GEMM.

    x: input buffer (possibly non-contiguous via stride constexprs)
    col: (batch * OUT_H * OUT_W, CKK) flattened output
    CKK = IN_C * KH * KW
    Grid: (batch * OUT_H * OUT_W,).
    Threads cooperatively write CKK elements per output position.

    Stride constexprs default to 0, meaning contiguous NCHW layout.
    When X_BATCH_STRIDE=0, strides are computed as:
      batch=IN_C*IN_H*IN_W, channel=IN_H*IN_W, height=IN_W, width=1
    """
    # Resolve strides: 0 means contiguous default
    SB = X_BATCH_STRIDE if X_BATCH_STRIDE > 0 else IN_C * IN_H * IN_W
    SC = X_C_STRIDE if X_C_STRIDE > 0 else IN_H * IN_W
    SH = X_H_STRIDE if X_H_STRIDE > 0 else IN_W
    SW = X_W_STRIDE

    idx = al.program_id(0)
    tid = al.thread_id()
    OUT_HW = OUT_H * OUT_W
    b = idx // OUT_HW
    rem = idx % OUT_HW
    oh = rem // OUT_W
    ow = rem % OUT_W
    x_base = X_OFFSET + b * SB
    col_base = idx * CKK
    KK = KH * KW

    for col_off in range(tid, CKK, NUM_THREADS):
        ic = col_off // KK
        rem_k = col_off % KK
        kh = rem_k // KW
        kw = rem_k % KW
        ih = oh * STRIDE_H + kh - PAD_H
        iw = ow * STRIDE_W + kw - PAD_W
        val = 0.0
        if ih >= 0:
            if ih < IN_H:
                if iw >= 0:
                    if iw < IN_W:
                        val = al.load(x + x_base + ic * SC + ih * SH + iw * SW)
        al.store(col + col_base + col_off, val)


# --- Strided copy (contiguify) ---


@al.kernel
def k_concat_2(
    A,
    B,
    out: al.output,
    N: al.constexpr,
    CAT_TOTAL: al.constexpr = 1,
    SPLIT_D: al.constexpr = 0,
    INNER: al.constexpr = 1,
    A_OUTER_STRIDE: al.constexpr = 0,
    A_CAT_STRIDE: al.constexpr = 0,
    A_INNER_STRIDE: al.constexpr = 1,
    B_OUTER_STRIDE: al.constexpr = 0,
    B_CAT_STRIDE: al.constexpr = 0,
    B_INNER_STRIDE: al.constexpr = 1,
    BLOCK_SIZE: al.constexpr = 1024,
):
    """Concatenate two buffers along an arbitrary dimension.

    Supports non-contiguous inputs via per-dim stride parameters.
    When strides are 0, computes contiguous strides from dimensions.
    """
    pid = al.program_id(0)
    offs = pid * BLOCK_SIZE + al.arange(0, BLOCK_SIZE)
    mask = offs < N
    inner_idx = offs % INNER
    concat_block = offs // INNER
    cat_idx = concat_block % CAT_TOTAL
    outer_idx = concat_block // CAT_TOTAL
    from_a = cat_idx < SPLIT_D
    # Source offsets — use strides if provided, else compute contiguous
    a_os = A_OUTER_STRIDE if A_OUTER_STRIDE > 0 else SPLIT_D * INNER
    a_cs = A_CAT_STRIDE if A_CAT_STRIDE > 0 else INNER
    a_is = A_INNER_STRIDE
    b_os = B_OUTER_STRIDE if B_OUTER_STRIDE > 0 else (CAT_TOTAL - SPLIT_D) * INNER
    b_cs = B_CAT_STRIDE if B_CAT_STRIDE > 0 else INNER
    b_is = B_INNER_STRIDE
    a_src = outer_idx * a_os + cat_idx * a_cs + inner_idx * a_is
    b_src = outer_idx * b_os + (cat_idx - SPLIT_D) * b_cs + inner_idx * b_is
    a_val = al.load(A + a_src, mask=mask & from_a)
    b_val = al.load(B + b_src, mask=mask & (~from_a))
    val = al.where(from_a, a_val, b_val)
    al.store(out + offs, val, mask=mask)


@al.kernel
def k_concat_3(
    A,
    B,
    C,
    out: al.output,
    N: al.constexpr,
    CAT_TOTAL: al.constexpr = 1,
    SPLIT_AB: al.constexpr = 0,
    SPLIT_BC: al.constexpr = 0,
    INNER: al.constexpr = 1,
    A_OUTER_STRIDE: al.constexpr = 0,
    A_CAT_STRIDE: al.constexpr = 0,
    A_INNER_STRIDE: al.constexpr = 1,
    B_OUTER_STRIDE: al.constexpr = 0,
    B_CAT_STRIDE: al.constexpr = 0,
    B_INNER_STRIDE: al.constexpr = 1,
    C_OUTER_STRIDE: al.constexpr = 0,
    C_CAT_STRIDE: al.constexpr = 0,
    C_INNER_STRIDE: al.constexpr = 1,
    BLOCK_SIZE: al.constexpr = 1024,
):
    """Concatenate three buffers along an arbitrary dimension in one pass."""
    pid = al.program_id(0)
    offs = pid * BLOCK_SIZE + al.arange(0, BLOCK_SIZE)
    mask = offs < N
    inner_idx = offs % INNER
    concat_block = offs // INNER
    cat_idx = concat_block % CAT_TOTAL
    outer_idx = concat_block // CAT_TOTAL
    from_a = cat_idx < SPLIT_AB
    from_c = cat_idx >= SPLIT_BC
    from_b = (~from_a) & (~from_c)
    a_os = A_OUTER_STRIDE if A_OUTER_STRIDE > 0 else SPLIT_AB * INNER
    a_cs = A_CAT_STRIDE if A_CAT_STRIDE > 0 else INNER
    a_is = A_INNER_STRIDE
    b_width = SPLIT_BC - SPLIT_AB
    b_os = B_OUTER_STRIDE if B_OUTER_STRIDE > 0 else b_width * INNER
    b_cs = B_CAT_STRIDE if B_CAT_STRIDE > 0 else INNER
    b_is = B_INNER_STRIDE
    c_width = CAT_TOTAL - SPLIT_BC
    c_os = C_OUTER_STRIDE if C_OUTER_STRIDE > 0 else c_width * INNER
    c_cs = C_CAT_STRIDE if C_CAT_STRIDE > 0 else INNER
    c_is = C_INNER_STRIDE
    a_src = outer_idx * a_os + cat_idx * a_cs + inner_idx * a_is
    b_src = outer_idx * b_os + (cat_idx - SPLIT_AB) * b_cs + inner_idx * b_is
    c_src = outer_idx * c_os + (cat_idx - SPLIT_BC) * c_cs + inner_idx * c_is
    a_val = al.load(A + a_src, mask=mask & from_a)
    b_val = al.load(B + b_src, mask=mask & from_b)
    c_val = al.load(C + c_src, mask=mask & from_c)
    val = al.where(from_a, a_val, al.where(from_c, c_val, b_val))
    al.store(out + offs, val, mask=mask)
