"""Quantized GEMM kernels."""

import alloy as al


@al.tunable(
    BLOCK_M=[16, 32, 64],
    BLOCK_N=[32, 64, 128],
    BLOCK_K=[32, 64, 128],
)
@al.kernel
def dot_q8_0(
    A,
    B_q8,
    scales,
    C: al.output,
    GROUP_SIZE: al.constexpr = 32,
    BLOCK_M: al.constexpr = 16,
    BLOCK_N: al.constexpr = 64,
    BLOCK_K: al.constexpr = 64,
):
    """GGUF Q8_0 matmul: C = A @ dequant(B_q8, scales).T.

    The loader normalizes GGUF Q8_0 row blocks into tensor-friendly buffers:
    int8 payload bytes in B_q8[N, K] plus fp16 scales[N, K / 32].
    """
    M, K = A.shape
    N = B_q8.shape[0]
    N_GROUPS = K // GROUP_SIZE

    pm = al.program_id(0)
    pn = al.program_id(1)
    rm = pm * BLOCK_M + al.arange(0, BLOCK_M)
    rn = pn * BLOCK_N + al.arange(0, BLOCK_N)
    rk = al.arange(0, BLOCK_K)

    a_ptrs = A + rm[:, None] * K + rk[None, :]
    acc = al.zeros((BLOCK_M, BLOCK_N), dtype=al.float32)

    for k in range(0, K, BLOCK_K):
        elem_k = k + rk

        a = al.load(a_ptrs, mask=(rm[:, None] < M) & (elem_k[None, :] < K))
        b = al.load(
            B_q8 + rn[:, None] * K + elem_k[None, :],
            mask=(rn[:, None] < N) & (elem_k[None, :] < K),
            _dequant_scale=scales,
            _dequant_zero_point=0,
            _dequant_n_groups=N_GROUPS,
        )

        acc += al.tile_dot(a, b, transpose_rhs=True)
        a_ptrs += BLOCK_K

    c_ptrs = C + rm[:, None] * N + rn[None, :]
    al.store(c_ptrs, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


@al.tunable(NUM_SPLITS=[1, 2, 4], NR0=[1, 2, 4])
@al.kernel
def dot_q8_0_v2(
    A,
    B_q8,
    scales,
    C: al.output,
    GROUP_SIZE: al.constexpr = 32,
    NUM_SPLITS: al.constexpr = 1,
    NR0: al.constexpr = 1,
):
    """Q8_0 matvec: split-K + multi-row amortization, vec4-batched.

    Same shape as `dot_q4_k_v2` minus the nibble unpack and bias.
    Two layers of parallelism:
      NUM_SPLITS simdgroups in the TG cooperatively reduce one K range
      (partials combined through shmem + barrier).
      NR0 output cols per TG: the activation is loaded ONCE per K chunk
      and dotted against NR0 different weight rows, cutting activation
      bandwidth by NR0x (the standard llama.cpp-style multi-row trick).

    The trace tracks named scalars as loop carries with no DCE, so each
    NR0 case is a separate body with exactly the right carry count.
    """
    M, K = A.shape
    N = B_q8.shape[0]
    N_GROUPS = K // GROUP_SIZE
    K_SPLIT = K // NUM_SPLITS

    col_base = al.program_id(0) * NR0
    tid = al.arange(0, 32 * NUM_SPLITS)
    simd_id = tid // 32
    lane = tid % 32

    K_STEP = 256
    PER_LANE = 8
    # The vec4-batched loop covers 32 lanes * PER_LANE = K_STEP elements per
    # iteration with UNMASKED load4_vec, so it can only run over a whole
    # multiple of K_STEP. When K_SPLIT is not a multiple of K_STEP (e.g.
    # gemma3:270m hidden_size=640 → K_SPLIT=640 = 2*256 + 128), the leftover
    # GROUP_SIZE-aligned tail is handled by a masked scalar loop below;
    # without it the bulk loop's last block reads past row `col` into the next
    # weight row (and past the activation), corrupting every M=1/decode matvec.
    _K_VEC = (K_SPLIT // K_STEP) * K_STEP

    def _final_store_one(col_local, val, col):
        # `col_local` is the linear offset into C (= m*N + col for M-loop
        # callers; col alone for M=1 callers). `col` is the column index
        # used for the bounds check (col_local can exceed N once m > 0,
        # so masking on col_local would drop every m > 0 write).
        if NUM_SPLITS == 1:
            al.store(C + col_local, val, mask=(col < N) & (lane < 1))
        else:
            shm = al.shared(NUM_SPLITS, dtype=al.float32)
            al.store(shm + simd_id, val, mask=(lane < 1))
            al.barrier()
            partials = [al.load(shm + s) for s in list(range(NUM_SPLITS))]
            total = partials[0]
            for s in list(range(1, NUM_SPLITS)):
                total = total + partials[s]
            al.store(
                C + col_local,
                total,
                mask=(col < N) & (simd_id < 1) & (lane < 1),
            )

    if NR0 == 1:
        for m in range(M):
            col0 = col_base + 0
            acc0 = 0.0
            for kb in range(0, _K_VEC, K_STEP):
                k = simd_id * K_SPLIT + kb + lane * PER_LANE
                a4_first = al.load4_vec(A + m * K + k)
                a4_second = al.load4_vec(A + m * K + k + 4)
                r0_first = al.load4_vec(B_q8 + col0 * K + k)
                r0_second = al.load4_vec(B_q8 + col0 * K + k + 4)
                s0 = al.cast(al.load(scales + col0 * N_GROUPS + k // GROUP_SIZE), al.float32)
                acc0 = (
                    acc0
                    + al.dot4(a4_first, s0 * al.cast(r0_first, al.float32))
                    + al.dot4(a4_second, s0 * al.cast(r0_second, al.float32))
                )
            for kt in range(_K_VEC, K_SPLIT, GROUP_SIZE):
                kk = kt + lane
                ka = simd_id * K_SPLIT + kk
                kmask = kk < K_SPLIT
                at = al.cast(al.load(A + m * K + ka, mask=kmask), al.float32)
                s0t = al.cast(
                    al.load(scales + col0 * N_GROUPS + ka // GROUP_SIZE, mask=kmask), al.float32
                )
                r0t = al.cast(al.load(B_q8 + col0 * K + ka, mask=kmask), al.float32)
                acc0 = acc0 + s0t * at * r0t
            acc0 = al.simd_reduce(acc0)
            _final_store_one(m * N + col0, acc0, col0)
    elif NR0 == 2:
        for m in range(M):
            col0 = col_base + 0
            col1 = col_base + 1
            acc0 = 0.0
            acc1 = 0.0
            for kb in range(0, _K_VEC, K_STEP):
                k = simd_id * K_SPLIT + kb + lane * PER_LANE
                a4_first = al.load4_vec(A + m * K + k)
                a4_second = al.load4_vec(A + m * K + k + 4)
                r0_first = al.load4_vec(B_q8 + col0 * K + k)
                r0_second = al.load4_vec(B_q8 + col0 * K + k + 4)
                s0 = al.cast(al.load(scales + col0 * N_GROUPS + k // GROUP_SIZE), al.float32)
                acc0 = (
                    acc0
                    + al.dot4(a4_first, s0 * al.cast(r0_first, al.float32))
                    + al.dot4(a4_second, s0 * al.cast(r0_second, al.float32))
                )
                r1_first = al.load4_vec(B_q8 + col1 * K + k)
                r1_second = al.load4_vec(B_q8 + col1 * K + k + 4)
                s1 = al.cast(al.load(scales + col1 * N_GROUPS + k // GROUP_SIZE), al.float32)
                acc1 = (
                    acc1
                    + al.dot4(a4_first, s1 * al.cast(r1_first, al.float32))
                    + al.dot4(a4_second, s1 * al.cast(r1_second, al.float32))
                )
            for kt in range(_K_VEC, K_SPLIT, GROUP_SIZE):
                kk = kt + lane
                ka = simd_id * K_SPLIT + kk
                kmask = kk < K_SPLIT
                at = al.cast(al.load(A + m * K + ka, mask=kmask), al.float32)
                s0t = al.cast(
                    al.load(scales + col0 * N_GROUPS + ka // GROUP_SIZE, mask=kmask), al.float32
                )
                r0t = al.cast(al.load(B_q8 + col0 * K + ka, mask=kmask), al.float32)
                acc0 = acc0 + s0t * at * r0t
                s1t = al.cast(
                    al.load(scales + col1 * N_GROUPS + ka // GROUP_SIZE, mask=kmask), al.float32
                )
                r1t = al.cast(al.load(B_q8 + col1 * K + ka, mask=kmask), al.float32)
                acc1 = acc1 + s1t * at * r1t
            acc0 = al.simd_reduce(acc0)
            acc1 = al.simd_reduce(acc1)
            _final_store_one(m * N + col0, acc0, col0)
            _final_store_one(m * N + col1, acc1, col1)
    else:  # NR0 == 4
        for m in range(M):
            col0 = col_base + 0
            col1 = col_base + 1
            col2 = col_base + 2
            col3 = col_base + 3
            acc0 = 0.0
            acc1 = 0.0
            acc2 = 0.0
            acc3 = 0.0
            for kb in range(0, _K_VEC, K_STEP):
                k = simd_id * K_SPLIT + kb + lane * PER_LANE
                a4_first = al.load4_vec(A + m * K + k)
                a4_second = al.load4_vec(A + m * K + k + 4)
                r0_first = al.load4_vec(B_q8 + col0 * K + k)
                r0_second = al.load4_vec(B_q8 + col0 * K + k + 4)
                s0 = al.cast(al.load(scales + col0 * N_GROUPS + k // GROUP_SIZE), al.float32)
                acc0 = (
                    acc0
                    + al.dot4(a4_first, s0 * al.cast(r0_first, al.float32))
                    + al.dot4(a4_second, s0 * al.cast(r0_second, al.float32))
                )
                r1_first = al.load4_vec(B_q8 + col1 * K + k)
                r1_second = al.load4_vec(B_q8 + col1 * K + k + 4)
                s1 = al.cast(al.load(scales + col1 * N_GROUPS + k // GROUP_SIZE), al.float32)
                acc1 = (
                    acc1
                    + al.dot4(a4_first, s1 * al.cast(r1_first, al.float32))
                    + al.dot4(a4_second, s1 * al.cast(r1_second, al.float32))
                )
                r2_first = al.load4_vec(B_q8 + col2 * K + k)
                r2_second = al.load4_vec(B_q8 + col2 * K + k + 4)
                s2 = al.cast(al.load(scales + col2 * N_GROUPS + k // GROUP_SIZE), al.float32)
                acc2 = (
                    acc2
                    + al.dot4(a4_first, s2 * al.cast(r2_first, al.float32))
                    + al.dot4(a4_second, s2 * al.cast(r2_second, al.float32))
                )
                r3_first = al.load4_vec(B_q8 + col3 * K + k)
                r3_second = al.load4_vec(B_q8 + col3 * K + k + 4)
                s3 = al.cast(al.load(scales + col3 * N_GROUPS + k // GROUP_SIZE), al.float32)
                acc3 = (
                    acc3
                    + al.dot4(a4_first, s3 * al.cast(r3_first, al.float32))
                    + al.dot4(a4_second, s3 * al.cast(r3_second, al.float32))
                )
            for kt in range(_K_VEC, K_SPLIT, GROUP_SIZE):
                kk = kt + lane
                ka = simd_id * K_SPLIT + kk
                kmask = kk < K_SPLIT
                at = al.cast(al.load(A + m * K + ka, mask=kmask), al.float32)
                sg = ka // GROUP_SIZE
                s0t = al.cast(al.load(scales + col0 * N_GROUPS + sg, mask=kmask), al.float32)
                acc0 = acc0 + s0t * at * al.cast(
                    al.load(B_q8 + col0 * K + ka, mask=kmask), al.float32
                )
                s1t = al.cast(al.load(scales + col1 * N_GROUPS + sg, mask=kmask), al.float32)
                acc1 = acc1 + s1t * at * al.cast(
                    al.load(B_q8 + col1 * K + ka, mask=kmask), al.float32
                )
                s2t = al.cast(al.load(scales + col2 * N_GROUPS + sg, mask=kmask), al.float32)
                acc2 = acc2 + s2t * at * al.cast(
                    al.load(B_q8 + col2 * K + ka, mask=kmask), al.float32
                )
                s3t = al.cast(al.load(scales + col3 * N_GROUPS + sg, mask=kmask), al.float32)
                acc3 = acc3 + s3t * at * al.cast(
                    al.load(B_q8 + col3 * K + ka, mask=kmask), al.float32
                )
            acc0 = al.simd_reduce(acc0)
            acc1 = al.simd_reduce(acc1)
            acc2 = al.simd_reduce(acc2)
            acc3 = al.simd_reduce(acc3)
            _final_store_one(m * N + col0, acc0, col0)
            _final_store_one(m * N + col1, acc1, col1)
            _final_store_one(m * N + col2, acc2, col2)
            _final_store_one(m * N + col3, acc3, col3)


@al.tunable()
@al.kernel
def dot_q8_0_v2_rows(A, B_q8, scales, C: al.output, GROUP_SIZE: al.constexpr = 32):
    """Q8_0 GEMV for the draft's multi-token propose, M in 2..8: dequant ONCE,
    read weights ONCE, accumulate all rows. Each row's accumulation matches
    dot_q8_0_v2 (NUM_SPLITS=1) exactly — same lane→k map, same scale-then-dot4
    order — so the draft's M-row propose forward stays numerically consistent
    with its M=1 decode and acceptance doesn't drop. Traced K-loop + 8 named
    accumulators (clamp rows >= M to M-1, store only M); M==1 keeps dot_q8_0_v2."""
    M, K = A.shape
    N = B_q8.shape[0]
    N_GROUPS = K // GROUP_SIZE
    col = al.program_id(0)
    lane = al.arange(0, 32)
    K_STEP = 256
    PER_LANE = 8
    r0 = min(0, M - 1)
    r1 = min(1, M - 1)
    r2 = min(2, M - 1)
    r3 = min(3, M - 1)
    r4 = min(4, M - 1)
    r5 = min(5, M - 1)
    r6 = min(6, M - 1)
    r7 = min(7, M - 1)
    acc0 = 0.0
    acc1 = 0.0
    acc2 = 0.0
    acc3 = 0.0
    acc4 = 0.0
    acc5 = 0.0
    acc6 = 0.0
    acc7 = 0.0
    for kb in range(0, K, K_STEP):
        k = kb + lane * PER_LANE
        s = al.cast(al.load(scales + col * N_GROUPS + k // GROUP_SIZE), al.float32)
        w_first = s * al.cast(al.load4_vec(B_q8 + col * K + k), al.float32)
        w_second = s * al.cast(al.load4_vec(B_q8 + col * K + k + 4), al.float32)
        acc0 = (
            acc0
            + al.dot4(al.load4_vec(A + r0 * K + k), w_first)
            + al.dot4(al.load4_vec(A + r0 * K + k + 4), w_second)
        )
        acc1 = (
            acc1
            + al.dot4(al.load4_vec(A + r1 * K + k), w_first)
            + al.dot4(al.load4_vec(A + r1 * K + k + 4), w_second)
        )
        acc2 = (
            acc2
            + al.dot4(al.load4_vec(A + r2 * K + k), w_first)
            + al.dot4(al.load4_vec(A + r2 * K + k + 4), w_second)
        )
        acc3 = (
            acc3
            + al.dot4(al.load4_vec(A + r3 * K + k), w_first)
            + al.dot4(al.load4_vec(A + r3 * K + k + 4), w_second)
        )
        acc4 = (
            acc4
            + al.dot4(al.load4_vec(A + r4 * K + k), w_first)
            + al.dot4(al.load4_vec(A + r4 * K + k + 4), w_second)
        )
        acc5 = (
            acc5
            + al.dot4(al.load4_vec(A + r5 * K + k), w_first)
            + al.dot4(al.load4_vec(A + r5 * K + k + 4), w_second)
        )
        acc6 = (
            acc6
            + al.dot4(al.load4_vec(A + r6 * K + k), w_first)
            + al.dot4(al.load4_vec(A + r6 * K + k + 4), w_second)
        )
        acc7 = (
            acc7
            + al.dot4(al.load4_vec(A + r7 * K + k), w_first)
            + al.dot4(al.load4_vec(A + r7 * K + k + 4), w_second)
        )
    accs = [acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7]
    for m in _unroll(M):
        al.store(C + m * N + col, al.simd_reduce(accs[m]), mask=(col < N) & (lane < 1))


@al.tunable()
@al.kernel
def dot_q8_0_silu_v2_rows(
    A,
    B_gate_q8,
    gate_scales,
    B_up_q8,
    up_scales,
    C: al.output,
    GROUP_SIZE: al.constexpr = 32,
):
    """Q8_0 gate+up GEMV with SiLU for the draft's multi-token propose, M 2..8:
    dequant gate+up ONCE, read each set ONCE, all rows. Per-row accumulation
    matches dot_q8_0_silu_v2 (M=1) so the draft propose stays consistent with
    decode. Traced K-loop, 16 named accumulators (gate ag, up au per row); clamp
    rows >= M to M-1, store only M."""
    M, K = A.shape
    N = B_gate_q8.shape[0]
    N_GROUPS = K // GROUP_SIZE
    col = al.program_id(0)
    lane = al.arange(0, 32)
    K_STEP = 256
    PER_LANE = 8
    r0 = min(0, M - 1)
    r1 = min(1, M - 1)
    r2 = min(2, M - 1)
    r3 = min(3, M - 1)
    r4 = min(4, M - 1)
    r5 = min(5, M - 1)
    r6 = min(6, M - 1)
    r7 = min(7, M - 1)
    ag0 = 0.0
    ag1 = 0.0
    ag2 = 0.0
    ag3 = 0.0
    ag4 = 0.0
    ag5 = 0.0
    ag6 = 0.0
    ag7 = 0.0
    au0 = 0.0
    au1 = 0.0
    au2 = 0.0
    au3 = 0.0
    au4 = 0.0
    au5 = 0.0
    au6 = 0.0
    au7 = 0.0
    for kb in range(0, K, K_STEP):
        k = kb + lane * PER_LANE
        gs = al.cast(al.load(gate_scales + col * N_GROUPS + k // GROUP_SIZE), al.float32)
        us = al.cast(al.load(up_scales + col * N_GROUPS + k // GROUP_SIZE), al.float32)
        g_first = gs * al.cast(al.load4_vec(B_gate_q8 + col * K + k), al.float32)
        g_second = gs * al.cast(al.load4_vec(B_gate_q8 + col * K + k + 4), al.float32)
        u_first = us * al.cast(al.load4_vec(B_up_q8 + col * K + k), al.float32)
        u_second = us * al.cast(al.load4_vec(B_up_q8 + col * K + k + 4), al.float32)
        ag0 = (
            ag0
            + al.dot4(al.load4_vec(A + r0 * K + k), g_first)
            + al.dot4(al.load4_vec(A + r0 * K + k + 4), g_second)
        )
        au0 = (
            au0
            + al.dot4(al.load4_vec(A + r0 * K + k), u_first)
            + al.dot4(al.load4_vec(A + r0 * K + k + 4), u_second)
        )
        ag1 = (
            ag1
            + al.dot4(al.load4_vec(A + r1 * K + k), g_first)
            + al.dot4(al.load4_vec(A + r1 * K + k + 4), g_second)
        )
        au1 = (
            au1
            + al.dot4(al.load4_vec(A + r1 * K + k), u_first)
            + al.dot4(al.load4_vec(A + r1 * K + k + 4), u_second)
        )
        ag2 = (
            ag2
            + al.dot4(al.load4_vec(A + r2 * K + k), g_first)
            + al.dot4(al.load4_vec(A + r2 * K + k + 4), g_second)
        )
        au2 = (
            au2
            + al.dot4(al.load4_vec(A + r2 * K + k), u_first)
            + al.dot4(al.load4_vec(A + r2 * K + k + 4), u_second)
        )
        ag3 = (
            ag3
            + al.dot4(al.load4_vec(A + r3 * K + k), g_first)
            + al.dot4(al.load4_vec(A + r3 * K + k + 4), g_second)
        )
        au3 = (
            au3
            + al.dot4(al.load4_vec(A + r3 * K + k), u_first)
            + al.dot4(al.load4_vec(A + r3 * K + k + 4), u_second)
        )
        ag4 = (
            ag4
            + al.dot4(al.load4_vec(A + r4 * K + k), g_first)
            + al.dot4(al.load4_vec(A + r4 * K + k + 4), g_second)
        )
        au4 = (
            au4
            + al.dot4(al.load4_vec(A + r4 * K + k), u_first)
            + al.dot4(al.load4_vec(A + r4 * K + k + 4), u_second)
        )
        ag5 = (
            ag5
            + al.dot4(al.load4_vec(A + r5 * K + k), g_first)
            + al.dot4(al.load4_vec(A + r5 * K + k + 4), g_second)
        )
        au5 = (
            au5
            + al.dot4(al.load4_vec(A + r5 * K + k), u_first)
            + al.dot4(al.load4_vec(A + r5 * K + k + 4), u_second)
        )
        ag6 = (
            ag6
            + al.dot4(al.load4_vec(A + r6 * K + k), g_first)
            + al.dot4(al.load4_vec(A + r6 * K + k + 4), g_second)
        )
        au6 = (
            au6
            + al.dot4(al.load4_vec(A + r6 * K + k), u_first)
            + al.dot4(al.load4_vec(A + r6 * K + k + 4), u_second)
        )
        ag7 = (
            ag7
            + al.dot4(al.load4_vec(A + r7 * K + k), g_first)
            + al.dot4(al.load4_vec(A + r7 * K + k + 4), g_second)
        )
        au7 = (
            au7
            + al.dot4(al.load4_vec(A + r7 * K + k), u_first)
            + al.dot4(al.load4_vec(A + r7 * K + k + 4), u_second)
        )
    ags = [ag0, ag1, ag2, ag3, ag4, ag5, ag6, ag7]
    aus = [au0, au1, au2, au3, au4, au5, au6, au7]
    for m in _unroll(M):
        g = al.simd_reduce(ags[m])
        u = al.simd_reduce(aus[m])
        silu = g * (1.0 / (1.0 + al.exp(-g))) * u
        al.store(C + m * N + col, silu, mask=(col < N) & (lane < 1))


@al.tunable(NR0=[1, 2, 4])
@al.kernel
def dot_q8_0_silu_v2(
    A,
    B_gate_q8,
    gate_scales,
    B_up_q8,
    up_scales,
    C: al.output,
    GROUP_SIZE: al.constexpr = 32,
    NR0: al.constexpr = 1,
):
    """Q8_0 gate+up matvec with SiLU fusion + multi-row amortization.

    Same vec4 K-vectorisation as `dot_q8_0_v2`. Each TG owns NR0
    consecutive output cols; activation loaded once per K chunk and
    dotted against NR0 (gate, up) row pairs.
    """
    M, K = A.shape
    N = B_gate_q8.shape[0]
    N_GROUPS = K // GROUP_SIZE

    col_base = al.program_id(0) * NR0
    lane = al.arange(0, 32)
    K_STEP = 256
    PER_LANE = 8

    if NR0 == 1:
        for m in range(M):
            col0 = col_base + 0
            acc_g0 = 0.0
            acc_u0 = 0.0
            for kb in range(0, K, K_STEP):
                k = kb + lane * PER_LANE
                a4_first = al.load4_vec(A + m * K + k)
                a4_second = al.load4_vec(A + m * K + k + 4)
                g0a = al.load4_vec(B_gate_q8 + col0 * K + k)
                g0b = al.load4_vec(B_gate_q8 + col0 * K + k + 4)
                u0a = al.load4_vec(B_up_q8 + col0 * K + k)
                u0b = al.load4_vec(B_up_q8 + col0 * K + k + 4)
                gs0 = al.cast(al.load(gate_scales + col0 * N_GROUPS + k // GROUP_SIZE), al.float32)
                us0 = al.cast(al.load(up_scales + col0 * N_GROUPS + k // GROUP_SIZE), al.float32)
                acc_g0 = (
                    acc_g0
                    + al.dot4(a4_first, gs0 * al.cast(g0a, al.float32))
                    + al.dot4(a4_second, gs0 * al.cast(g0b, al.float32))
                )
                acc_u0 = (
                    acc_u0
                    + al.dot4(a4_first, us0 * al.cast(u0a, al.float32))
                    + al.dot4(a4_second, us0 * al.cast(u0b, al.float32))
                )
            g = al.simd_reduce(acc_g0)
            u = al.simd_reduce(acc_u0)
            silu = g * (1.0 / (1.0 + al.exp(-g))) * u
            al.store(C + m * N + col0, silu, mask=(col0 < N) & (lane < 1))
    elif NR0 == 2:
        for m in range(M):
            col0 = col_base + 0
            col1 = col_base + 1
            acc_g0 = 0.0
            acc_u0 = 0.0
            acc_g1 = 0.0
            acc_u1 = 0.0
            for kb in range(0, K, K_STEP):
                k = kb + lane * PER_LANE
                a4_first = al.load4_vec(A + m * K + k)
                a4_second = al.load4_vec(A + m * K + k + 4)
                g0a = al.load4_vec(B_gate_q8 + col0 * K + k)
                g0b = al.load4_vec(B_gate_q8 + col0 * K + k + 4)
                u0a = al.load4_vec(B_up_q8 + col0 * K + k)
                u0b = al.load4_vec(B_up_q8 + col0 * K + k + 4)
                gs0 = al.cast(al.load(gate_scales + col0 * N_GROUPS + k // GROUP_SIZE), al.float32)
                us0 = al.cast(al.load(up_scales + col0 * N_GROUPS + k // GROUP_SIZE), al.float32)
                acc_g0 = (
                    acc_g0
                    + al.dot4(a4_first, gs0 * al.cast(g0a, al.float32))
                    + al.dot4(a4_second, gs0 * al.cast(g0b, al.float32))
                )
                acc_u0 = (
                    acc_u0
                    + al.dot4(a4_first, us0 * al.cast(u0a, al.float32))
                    + al.dot4(a4_second, us0 * al.cast(u0b, al.float32))
                )
                g1a = al.load4_vec(B_gate_q8 + col1 * K + k)
                g1b = al.load4_vec(B_gate_q8 + col1 * K + k + 4)
                u1a = al.load4_vec(B_up_q8 + col1 * K + k)
                u1b = al.load4_vec(B_up_q8 + col1 * K + k + 4)
                gs1 = al.cast(al.load(gate_scales + col1 * N_GROUPS + k // GROUP_SIZE), al.float32)
                us1 = al.cast(al.load(up_scales + col1 * N_GROUPS + k // GROUP_SIZE), al.float32)
                acc_g1 = (
                    acc_g1
                    + al.dot4(a4_first, gs1 * al.cast(g1a, al.float32))
                    + al.dot4(a4_second, gs1 * al.cast(g1b, al.float32))
                )
                acc_u1 = (
                    acc_u1
                    + al.dot4(a4_first, us1 * al.cast(u1a, al.float32))
                    + al.dot4(a4_second, us1 * al.cast(u1b, al.float32))
                )
            for col_v, acc_g, acc_u in [(col0, acc_g0, acc_u0), (col1, acc_g1, acc_u1)]:
                g = al.simd_reduce(acc_g)
                u = al.simd_reduce(acc_u)
                silu = g * (1.0 / (1.0 + al.exp(-g))) * u
                al.store(C + m * N + col_v, silu, mask=(col_v < N) & (lane < 1))
    else:  # NR0 == 4
        for m in range(M):
            col0 = col_base + 0
            col1 = col_base + 1
            col2 = col_base + 2
            col3 = col_base + 3
            acc_g0 = 0.0
            acc_u0 = 0.0
            acc_g1 = 0.0
            acc_u1 = 0.0
            acc_g2 = 0.0
            acc_u2 = 0.0
            acc_g3 = 0.0
            acc_u3 = 0.0
            for kb in range(0, K, K_STEP):
                k = kb + lane * PER_LANE
                a4_first = al.load4_vec(A + m * K + k)
                a4_second = al.load4_vec(A + m * K + k + 4)
                g0a = al.load4_vec(B_gate_q8 + col0 * K + k)
                g0b = al.load4_vec(B_gate_q8 + col0 * K + k + 4)
                u0a = al.load4_vec(B_up_q8 + col0 * K + k)
                u0b = al.load4_vec(B_up_q8 + col0 * K + k + 4)
                gs0 = al.cast(al.load(gate_scales + col0 * N_GROUPS + k // GROUP_SIZE), al.float32)
                us0 = al.cast(al.load(up_scales + col0 * N_GROUPS + k // GROUP_SIZE), al.float32)
                acc_g0 = (
                    acc_g0
                    + al.dot4(a4_first, gs0 * al.cast(g0a, al.float32))
                    + al.dot4(a4_second, gs0 * al.cast(g0b, al.float32))
                )
                acc_u0 = (
                    acc_u0
                    + al.dot4(a4_first, us0 * al.cast(u0a, al.float32))
                    + al.dot4(a4_second, us0 * al.cast(u0b, al.float32))
                )
                g1a = al.load4_vec(B_gate_q8 + col1 * K + k)
                g1b = al.load4_vec(B_gate_q8 + col1 * K + k + 4)
                u1a = al.load4_vec(B_up_q8 + col1 * K + k)
                u1b = al.load4_vec(B_up_q8 + col1 * K + k + 4)
                gs1 = al.cast(al.load(gate_scales + col1 * N_GROUPS + k // GROUP_SIZE), al.float32)
                us1 = al.cast(al.load(up_scales + col1 * N_GROUPS + k // GROUP_SIZE), al.float32)
                acc_g1 = (
                    acc_g1
                    + al.dot4(a4_first, gs1 * al.cast(g1a, al.float32))
                    + al.dot4(a4_second, gs1 * al.cast(g1b, al.float32))
                )
                acc_u1 = (
                    acc_u1
                    + al.dot4(a4_first, us1 * al.cast(u1a, al.float32))
                    + al.dot4(a4_second, us1 * al.cast(u1b, al.float32))
                )
                g2a = al.load4_vec(B_gate_q8 + col2 * K + k)
                g2b = al.load4_vec(B_gate_q8 + col2 * K + k + 4)
                u2a = al.load4_vec(B_up_q8 + col2 * K + k)
                u2b = al.load4_vec(B_up_q8 + col2 * K + k + 4)
                gs2 = al.cast(al.load(gate_scales + col2 * N_GROUPS + k // GROUP_SIZE), al.float32)
                us2 = al.cast(al.load(up_scales + col2 * N_GROUPS + k // GROUP_SIZE), al.float32)
                acc_g2 = (
                    acc_g2
                    + al.dot4(a4_first, gs2 * al.cast(g2a, al.float32))
                    + al.dot4(a4_second, gs2 * al.cast(g2b, al.float32))
                )
                acc_u2 = (
                    acc_u2
                    + al.dot4(a4_first, us2 * al.cast(u2a, al.float32))
                    + al.dot4(a4_second, us2 * al.cast(u2b, al.float32))
                )
                g3a = al.load4_vec(B_gate_q8 + col3 * K + k)
                g3b = al.load4_vec(B_gate_q8 + col3 * K + k + 4)
                u3a = al.load4_vec(B_up_q8 + col3 * K + k)
                u3b = al.load4_vec(B_up_q8 + col3 * K + k + 4)
                gs3 = al.cast(al.load(gate_scales + col3 * N_GROUPS + k // GROUP_SIZE), al.float32)
                us3 = al.cast(al.load(up_scales + col3 * N_GROUPS + k // GROUP_SIZE), al.float32)
                acc_g3 = (
                    acc_g3
                    + al.dot4(a4_first, gs3 * al.cast(g3a, al.float32))
                    + al.dot4(a4_second, gs3 * al.cast(g3b, al.float32))
                )
                acc_u3 = (
                    acc_u3
                    + al.dot4(a4_first, us3 * al.cast(u3a, al.float32))
                    + al.dot4(a4_second, us3 * al.cast(u3b, al.float32))
                )
            for col_v, acc_g, acc_u in [
                (col0, acc_g0, acc_u0),
                (col1, acc_g1, acc_u1),
                (col2, acc_g2, acc_u2),
                (col3, acc_g3, acc_u3),
            ]:
                g = al.simd_reduce(acc_g)
                u = al.simd_reduce(acc_u)
                silu = g * (1.0 / (1.0 + al.exp(-g))) * u
                al.store(C + m * N + col_v, silu, mask=(col_v < N) & (lane < 1))


@al.tunable(
    BLOCK_M=[32, 64, 128],
    BLOCK_N=[32, 64, 128],
    BLOCK_K=[32, 64, 128],
)
@al.kernel
def dot_q8_0_silu(
    A,
    B_gate_q8,
    gate_scales,
    B_up_q8,
    up_scales,
    C: al.output,
    GROUP_SIZE: al.constexpr = 32,
    BLOCK_M: al.constexpr = 16,
    BLOCK_N: al.constexpr = 32,
    BLOCK_K: al.constexpr = 64,
):
    """GGUF Q8_0 paired matmul: C = silu(A @ gate.T) * (A @ up.T).

    Cooperative tile_dot with the `_dequant_scale` epilogue baked into
    `al.load`; both gate and up weights share the same activation slab
    per K-block so the activation load is amortised across both MMAs.
    """
    M, K = A.shape
    N = B_gate_q8.shape[0]
    N_GROUPS = K // GROUP_SIZE

    pm = al.program_id(0)
    pn = al.program_id(1)
    rm = pm * BLOCK_M + al.arange(0, BLOCK_M)
    rn = pn * BLOCK_N + al.arange(0, BLOCK_N)
    rk = al.arange(0, BLOCK_K)

    a_ptrs = A + rm[:, None] * K + rk[None, :]
    acc_gate = al.zeros((BLOCK_M, BLOCK_N), dtype=al.float32)
    acc_up = al.zeros((BLOCK_M, BLOCK_N), dtype=al.float32)

    for k in range(0, K, BLOCK_K):
        elem_k = k + rk
        a = al.load(a_ptrs, mask=(rm[:, None] < M) & (elem_k[None, :] < K))
        gate = al.load(
            B_gate_q8 + rn[:, None] * K + elem_k[None, :],
            mask=(rn[:, None] < N) & (elem_k[None, :] < K),
            _dequant_scale=gate_scales,
            _dequant_zero_point=0,
            _dequant_n_groups=N_GROUPS,
        )
        up = al.load(
            B_up_q8 + rn[:, None] * K + elem_k[None, :],
            mask=(rn[:, None] < N) & (elem_k[None, :] < K),
            _dequant_scale=up_scales,
            _dequant_zero_point=0,
            _dequant_n_groups=N_GROUPS,
        )
        acc_gate += al.tile_dot(a, gate, transpose_rhs=True)
        acc_up += al.tile_dot(a, up, transpose_rhs=True)
        a_ptrs += BLOCK_K

    silu = acc_gate * (1.0 / (1.0 + al.exp(-acc_gate))) * acc_up
    c_ptrs = C + rm[:, None] * N + rn[None, :]
    al.store(c_ptrs, silu, mask=(rm[:, None] < M) & (rn[None, :] < N))


@al.tunable(
    BLOCK_M=[8, 16, 32, 64, 128],
    BLOCK_N=[32, 64, 128],
    BLOCK_K=[32, 64, 128],
)
@al.kernel
def dot_q4_k(
    A,
    BLK,
    C: al.output,
    BLOCK_M: al.constexpr = 16,
    BLOCK_N: al.constexpr = 64,
    BLOCK_K: al.constexpr = 64,
):
    """GGUF-native Q4_K matmul: C = A @ dequant(BLK).T over 144-byte superblocks.

    Tile-MMA over the cooperative load's fused Q4_K dequant. The M=1 matvec
    specialization lives in dot_q4_k_v2.
    """
    M, K = A.shape
    N = BLK.shape[0]
    BLOCK_BYTES = 144
    N_GROUPS = K // 256
    ROW_BYTES = N_GROUPS * BLOCK_BYTES

    pm = al.program_id(0)
    pn = al.program_id(1)
    rm = pm * BLOCK_M + al.arange(0, BLOCK_M)
    rn = pn * BLOCK_N + al.arange(0, BLOCK_N)
    rk = al.arange(0, BLOCK_K)

    a_ptrs = A + rm[:, None] * K + rk[None, :]
    acc = al.zeros((BLOCK_M, BLOCK_N), dtype=al.float32)

    for k in range(0, K, BLOCK_K):
        elem_k = k + rk
        a = al.load(a_ptrs, mask=(rm[:, None] < M) & (elem_k[None, :] < K))
        b = al.load(
            BLK + rn[:, None] * ROW_BYTES + elem_k[None, :],
            mask=(rn[:, None] < N) & (elem_k[None, :] < K),
            _dequant_format="q4_k",
        )
        acc += al.tile_dot(a, b, transpose_rhs=True)
        a_ptrs += BLOCK_K

    c_ptrs = C + rm[:, None] * N + rn[None, :]
    al.store(c_ptrs, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


@al.kernel
def embedding_q8_0(
    input_ids,
    qweight,
    scales,
    out: al.output,
    NUM_INDICES: al.constexpr,
    WIDTH: al.constexpr,
    GROUP_SIZE: al.constexpr = 32,
    BLOCK_SIZE: al.constexpr = 1024,
):
    """Embedding lookup over normalized GGUF Q8_0 row blocks."""
    pid = al.program_id(0)
    offs = pid * BLOCK_SIZE + al.arange(0, BLOCK_SIZE)
    mask = offs < NUM_INDICES * WIDTH
    row = offs // WIDTH
    col = offs % WIDTH
    groups = WIDTH // GROUP_SIZE
    idx = al.load(input_ids + row, mask=mask)
    scale = al.cast(al.load(scales + idx * groups + col // GROUP_SIZE, mask=mask), al.float32)
    raw = al.cast(al.load(qweight + idx * WIDTH + col, mask=mask), al.float32)
    al.store(out + offs, scale * raw, mask=mask)


@al.kernel
def embedding_q4_k(
    input_ids,
    BLK,
    out: al.output,
    NUM_INDICES: al.constexpr,
    WIDTH: al.constexpr,
    BLOCK_SIZE: al.constexpr = 1024,
):
    """Embedding lookup over GGUF-native Q4_K 144-byte superblocks.

    Per 256-element superblock: d/dmin fp16, 12B packed 6-bit scales/mins
    (get_scale_min_k4), 128B interleaved nibbles. weight = d*sc*nib - dmin*m.
    """
    BLOCK_BYTES = 144
    NB = WIDTH // 256
    ROW_BYTES = NB * BLOCK_BYTES

    pid = al.program_id(0)
    offs = pid * BLOCK_SIZE + al.arange(0, BLOCK_SIZE)
    mask = offs < NUM_INDICES * WIDTH
    row = offs // WIDTH
    col = offs % WIDTH

    idx = al.load(input_ids + row, mask=mask)

    block_base = idx * ROW_BYTES + (col // 256) * BLOCK_BYTES
    sc_base = block_base + 4
    g = (col % 256) // 32
    lane = col % 32

    d_lo = al.cast(al.load(BLK + block_base, mask=mask), "uint16")
    d_hi = al.cast(al.load(BLK + block_base + 1, mask=mask), "uint16")
    d = al.cast(al.bitcast(al.cast(d_lo | (d_hi << 8), "uint16"), al.float16), al.float32)
    dm_lo = al.cast(al.load(BLK + block_base + 2, mask=mask), "uint16")
    dm_hi = al.cast(al.load(BLK + block_base + 3, mask=mask), "uint16")
    dmin = al.cast(al.bitcast(al.cast(dm_lo | (dm_hi << 8), "uint16"), al.float16), al.float32)

    s_j = al.cast(al.load(BLK + sc_base + g, mask=mask), al.int32)
    s_j4 = al.cast(al.load(BLK + sc_base + g + 4, mask=mask), al.int32)
    s_jm4 = al.cast(al.load(BLK + block_base + g, mask=mask), al.int32)  # scales[g-4] = sc_base+g-4
    ge4 = g >= 4
    sc = al.where(ge4, (s_j4 & 0x0F) | (((s_jm4 >> 6) & 0x03) << 4), s_j & 0x3F)
    mn = al.where(ge4, ((s_j4 >> 4) & 0x0F) | (((s_j >> 6) & 0x03) << 4), s_j4 & 0x3F)

    qbyte = al.cast(al.load(BLK + block_base + 16 + (g // 2) * 32 + lane, mask=mask), al.int32)
    nib = al.where((g & 1) > 0, (qbyte >> 4) & 0x0F, qbyte & 0x0F)

    weight = d * al.cast(sc, al.float32) * al.cast(nib, al.float32) - dmin * al.cast(mn, al.float32)
    al.store(out + offs, weight, mask=mask)


@al.tunable(
    BLOCK_M=[8, 16, 32, 64, 128],
    BLOCK_N=[32, 64, 128],
    BLOCK_K=[32, 64, 128],
)
@al.kernel
def dot_q5_0(
    A,
    B_q5,
    qhigh,
    scales,
    C: al.output,
    GROUP_SIZE: al.constexpr = 32,
    BLOCK_M: al.constexpr = 16,
    BLOCK_N: al.constexpr = 64,
    BLOCK_K: al.constexpr = 64,
):
    """GGUF Q5_0 matmul: C = A @ dequant(B_q5, qhigh, scales).T.

    Tiled prefill GEMM, identical in shape to `dot_q4_k` (one (BLOCK_N,
    BLOCK_K) weight tile per K step, reused across all BLOCK_M rows via
    `tile_dot`). The M=1 decode matvec lives in `dot_q5_0_v2`.

    The cooperative-load dequant is extended with `_dequant_high`: Q5_0's
    5th bit lives in the separate `qhigh` buffer (same packed [N, K/2]
    layout as B_q5), so the load rebuilds `nibble | (high_bit << 4)` before
    the symmetric `(q - 16) * scale` dequant. q4_k/q8_0/q6_k don't use it.

      B_q5  : uint8 [N, K/2]  — byte k/2 holds elem k (low nibble) + k+1 (high).
      qhigh : uint8 [N, K/2]  — bit (k%2)*4 holds the 5th bit of elem k.
      scales: fp16  [N, K/32] — one scale per 32-element block.
    """
    M, K = A.shape
    N = B_q5.shape[0]
    PACK_FACTOR = 2
    K_PACKED = K // PACK_FACTOR
    N_GROUPS = K // GROUP_SIZE

    pm = al.program_id(0)
    pn = al.program_id(1)
    rm = pm * BLOCK_M + al.arange(0, BLOCK_M)
    rn = pn * BLOCK_N + al.arange(0, BLOCK_N)
    rk = al.arange(0, BLOCK_K)

    a_ptrs = A + rm[:, None] * K + rk[None, :]
    acc = al.zeros((BLOCK_M, BLOCK_N), dtype=al.float32)

    for k in range(0, K, BLOCK_K):
        elem_k = k + rk
        a = al.load(a_ptrs, mask=(rm[:, None] < M) & (elem_k[None, :] < K))
        b = al.load(
            B_q5 + rn[:, None] * K_PACKED + elem_k[None, :] // PACK_FACTOR,
            mask=(rn[:, None] < N) & (elem_k[None, :] < K),
            _dequant_scale=scales,
            _dequant_high=qhigh,
            _dequant_zero_point=16,
            _dequant_n_groups=N_GROUPS,
        )
        acc += al.tile_dot(a, b, transpose_rhs=True)
        a_ptrs += BLOCK_K

    c_ptrs = C + rm[:, None] * N + rn[None, :]
    al.store(c_ptrs, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


@al.tunable(NUM_SPLITS=[1, 2, 4], NR0=[1, 2, 4])
@al.kernel
def dot_q5_0_v2(
    A,
    B_q5,
    qhigh,
    scales,
    C: al.output,
    GROUP_SIZE: al.constexpr = 32,
    NUM_SPLITS: al.constexpr = 1,
    NR0: al.constexpr = 1,
):
    """Q5_0 matvec: split-K + multi-row amortization, vec4-batched.

    Same shape as `dot_q4_k_v2` (nibble unpack via `interleave_vec4`)
    plus a per-element 5th-bit correction loaded with the same vec4
    pattern. The loader normalises Q5_0 blocks into K-sequential
    buffers, with the high-bit payload packed into nibble positions
    so it goes through the same `load4_vec` + mask/shift dance:

      B_q5  : uint8 [N, K/2]  — packed nibbles, byte at k/2 holds
                                element k (low) and element k+1 (high).
      qhigh : uint8 [N, K/2]  — bit 0 of low nibble holds the 5th bit
                                of element k (even K), bit 0 of high
                                nibble (= bit 4 of byte) holds the
                                5th bit of element k+1 (odd K).
      scales: fp16  [N, K/32] — one fp16 scale per 32-element block.

    Per-element decode: `q = ((nibble | (high_bit << 4)) - 16) * scale`.
    """
    M, K = A.shape
    N = scales.shape[0]
    K_HALF = K // 2
    N_GROUPS = K // GROUP_SIZE
    K_SPLIT = K // NUM_SPLITS

    col_base = al.program_id(0) * NR0
    m = al.program_id(1)
    tid = al.arange(0, 32 * NUM_SPLITS)
    simd_id = tid // 32
    lane = tid % 32

    K_STEP = 256
    PER_LANE = 8
    # PER_LANE divides K cleanly (Q5_0 only fires on tensors whose K
    # is a multiple of 32, the GGUF block size; 32 % PER_LANE == 0).
    # K_STEP, however, need NOT divide K — gemma3:1b's attention
    # tensors have K=1152 (= 4.5 * K_STEP). We clamp the per-lane
    # offset so loads stay in bounds and mask the dot-product
    # contribution to zero on OOB lanes.
    K_TAIL = (K // PER_LANE - 1) * PER_LANE  # last valid lane start

    def _final_store_one(col_local, val, col):
        # See dot_q8_0_v2's _final_store_one: `col_local` is the linear
        # offset into C (= m*N + col), `col` is the column index. The
        # bounds check belongs to col, not col_local (which exceeds N
        # for m > 0).
        if NUM_SPLITS == 1:
            al.store(C + col_local, val, mask=(col < N) & (lane < 1))
        else:
            shm = al.shared(NUM_SPLITS, dtype=al.float32)
            al.store(shm + simd_id, val, mask=(lane < 1))
            al.barrier()
            partials = [al.load(shm + s) for s in list(range(NUM_SPLITS))]
            total = partials[0]
            for s in list(range(1, NUM_SPLITS)):
                total = total + partials[s]
            al.store(
                C + col_local,
                total,
                mask=(col < N) & (simd_id < 1) & (lane < 1),
            )

    def _decode_q5(raw, hraw):
        # raw: vec4 int32, 4 nibble bytes covering 8 K positions.
        # hraw: vec4 int32, 4 hbit bytes — bit 0 of low nibble of byte i
        #       is the 5th bit of element (k + 2i); bit 0 of high nibble
        #       (i.e. bit 4 of the byte) is the 5th bit of element (k + 2i + 1).
        lo = raw & 0x0F  # vec4: elements at even K
        hi = (raw >> 4) & 0x0F  # vec4: elements at odd K
        hb_lo = (hraw & 0x01) << 4  # vec4: high bit << 4, even-K
        hb_hi = ((hraw >> 4) & 0x01) << 4  # vec4: high bit << 4, odd-K
        return (lo | hb_lo) - 16, (hi | hb_hi) - 16

    # M parallelized via program_id(1); each TG handles one (m, NR0-col-block).
    if NR0 == 1:
        col0 = col_base + 0
        acc0 = 0.0
        for kb in range(0, K_SPLIT, K_STEP):
            k = simd_id * K_SPLIT + kb + lane * PER_LANE
            safe_k = al.minimum(al.cast(k, al.int32), al.cast(K_TAIL, al.int32))
            in_bounds = al.cast(al.cast(k, al.int32) <= al.cast(K_TAIL, al.int32), al.float32)
            a4_first = al.load4_vec(A + m * K + safe_k)
            a4_second = al.load4_vec(A + m * K + safe_k + 4)

            raw0 = al.cast(al.load4_vec(B_q5 + col0 * K_HALF + safe_k // 2), al.int32)
            hraw0 = al.cast(al.load4_vec(qhigh + col0 * K_HALF + safe_k // 2), al.int32)
            q5_lo0, q5_hi0 = _decode_q5(raw0, hraw0)
            q0_first = al.interleave_vec4(q5_lo0, q5_hi0, 0)
            q0_second = al.interleave_vec4(q5_lo0, q5_hi0, 1)
            s0 = al.cast(al.load(scales + col0 * N_GROUPS + safe_k // GROUP_SIZE), al.float32)
            acc0 = acc0 + in_bounds * (
                al.dot4(a4_first, s0 * _q4k_u2f(al.cast(q0_first, al.int32)))
                + al.dot4(a4_second, s0 * _q4k_u2f(al.cast(q0_second, al.int32)))
            )
        acc0 = al.simd_reduce(acc0)
        _final_store_one(m * N + col0, acc0, col0)
    elif NR0 == 2:
        col0 = col_base + 0
        col1 = col_base + 1
        acc0 = 0.0
        acc1 = 0.0
        for kb in range(0, K_SPLIT, K_STEP):
            k = simd_id * K_SPLIT + kb + lane * PER_LANE
            safe_k = al.minimum(al.cast(k, al.int32), al.cast(K_TAIL, al.int32))
            in_bounds = al.cast(al.cast(k, al.int32) <= al.cast(K_TAIL, al.int32), al.float32)
            a4_first = al.load4_vec(A + m * K + safe_k)
            a4_second = al.load4_vec(A + m * K + safe_k + 4)

            raw0 = al.cast(al.load4_vec(B_q5 + col0 * K_HALF + safe_k // 2), al.int32)
            hraw0 = al.cast(al.load4_vec(qhigh + col0 * K_HALF + safe_k // 2), al.int32)
            q5_lo0, q5_hi0 = _decode_q5(raw0, hraw0)
            q0_first = al.interleave_vec4(q5_lo0, q5_hi0, 0)
            q0_second = al.interleave_vec4(q5_lo0, q5_hi0, 1)
            s0 = al.cast(al.load(scales + col0 * N_GROUPS + safe_k // GROUP_SIZE), al.float32)
            acc0 = acc0 + in_bounds * (
                al.dot4(a4_first, s0 * _q4k_u2f(al.cast(q0_first, al.int32)))
                + al.dot4(a4_second, s0 * _q4k_u2f(al.cast(q0_second, al.int32)))
            )

            raw1 = al.cast(al.load4_vec(B_q5 + col1 * K_HALF + safe_k // 2), al.int32)
            hraw1 = al.cast(al.load4_vec(qhigh + col1 * K_HALF + safe_k // 2), al.int32)
            q5_lo1, q5_hi1 = _decode_q5(raw1, hraw1)
            q1_first = al.interleave_vec4(q5_lo1, q5_hi1, 0)
            q1_second = al.interleave_vec4(q5_lo1, q5_hi1, 1)
            s1 = al.cast(al.load(scales + col1 * N_GROUPS + safe_k // GROUP_SIZE), al.float32)
            acc1 = acc1 + in_bounds * (
                al.dot4(a4_first, s1 * _q4k_u2f(al.cast(q1_first, al.int32)))
                + al.dot4(a4_second, s1 * _q4k_u2f(al.cast(q1_second, al.int32)))
            )
        acc0 = al.simd_reduce(acc0)
        acc1 = al.simd_reduce(acc1)
        _final_store_one(m * N + col0, acc0, col0)
        _final_store_one(m * N + col1, acc1, col1)
    else:  # NR0 == 4
        col0 = col_base + 0
        col1 = col_base + 1
        col2 = col_base + 2
        col3 = col_base + 3
        acc0 = 0.0
        acc1 = 0.0
        acc2 = 0.0
        acc3 = 0.0
        for kb in range(0, K_SPLIT, K_STEP):
            k = simd_id * K_SPLIT + kb + lane * PER_LANE
            safe_k = al.minimum(al.cast(k, al.int32), al.cast(K_TAIL, al.int32))
            in_bounds = al.cast(al.cast(k, al.int32) <= al.cast(K_TAIL, al.int32), al.float32)
            a4_first = al.load4_vec(A + m * K + safe_k)
            a4_second = al.load4_vec(A + m * K + safe_k + 4)

            raw0 = al.cast(al.load4_vec(B_q5 + col0 * K_HALF + safe_k // 2), al.int32)
            hraw0 = al.cast(al.load4_vec(qhigh + col0 * K_HALF + safe_k // 2), al.int32)
            q5_lo0, q5_hi0 = _decode_q5(raw0, hraw0)
            q0a = al.interleave_vec4(q5_lo0, q5_hi0, 0)
            q0b = al.interleave_vec4(q5_lo0, q5_hi0, 1)
            s0 = al.cast(al.load(scales + col0 * N_GROUPS + safe_k // GROUP_SIZE), al.float32)
            acc0 = acc0 + in_bounds * (
                al.dot4(a4_first, s0 * al.cast(q0a, al.float32))
                + al.dot4(a4_second, s0 * al.cast(q0b, al.float32))
            )

            raw1 = al.cast(al.load4_vec(B_q5 + col1 * K_HALF + safe_k // 2), al.int32)
            hraw1 = al.cast(al.load4_vec(qhigh + col1 * K_HALF + safe_k // 2), al.int32)
            q5_lo1, q5_hi1 = _decode_q5(raw1, hraw1)
            q1a = al.interleave_vec4(q5_lo1, q5_hi1, 0)
            q1b = al.interleave_vec4(q5_lo1, q5_hi1, 1)
            s1 = al.cast(al.load(scales + col1 * N_GROUPS + safe_k // GROUP_SIZE), al.float32)
            acc1 = acc1 + in_bounds * (
                al.dot4(a4_first, s1 * al.cast(q1a, al.float32))
                + al.dot4(a4_second, s1 * al.cast(q1b, al.float32))
            )

            raw2 = al.cast(al.load4_vec(B_q5 + col2 * K_HALF + safe_k // 2), al.int32)
            hraw2 = al.cast(al.load4_vec(qhigh + col2 * K_HALF + safe_k // 2), al.int32)
            q5_lo2, q5_hi2 = _decode_q5(raw2, hraw2)
            q2a = al.interleave_vec4(q5_lo2, q5_hi2, 0)
            q2b = al.interleave_vec4(q5_lo2, q5_hi2, 1)
            s2 = al.cast(al.load(scales + col2 * N_GROUPS + safe_k // GROUP_SIZE), al.float32)
            acc2 = acc2 + in_bounds * (
                al.dot4(a4_first, s2 * al.cast(q2a, al.float32))
                + al.dot4(a4_second, s2 * al.cast(q2b, al.float32))
            )

            raw3 = al.cast(al.load4_vec(B_q5 + col3 * K_HALF + safe_k // 2), al.int32)
            hraw3 = al.cast(al.load4_vec(qhigh + col3 * K_HALF + safe_k // 2), al.int32)
            q5_lo3, q5_hi3 = _decode_q5(raw3, hraw3)
            q3a = al.interleave_vec4(q5_lo3, q5_hi3, 0)
            q3b = al.interleave_vec4(q5_lo3, q5_hi3, 1)
            s3 = al.cast(al.load(scales + col3 * N_GROUPS + safe_k // GROUP_SIZE), al.float32)
            acc3 = acc3 + in_bounds * (
                al.dot4(a4_first, s3 * al.cast(q3a, al.float32))
                + al.dot4(a4_second, s3 * al.cast(q3b, al.float32))
            )
        acc0 = al.simd_reduce(acc0)
        acc1 = al.simd_reduce(acc1)
        acc2 = al.simd_reduce(acc2)
        acc3 = al.simd_reduce(acc3)
        _final_store_one(m * N + col0, acc0, col0)
        _final_store_one(m * N + col1, acc1, col1)
        _final_store_one(m * N + col2, acc2, col2)
        _final_store_one(m * N + col3, acc3, col3)


@al.kernel
def embedding_q5_0(
    input_ids,
    qweight,
    qhigh,
    scales,
    out: al.output,
    NUM_INDICES: al.constexpr,
    WIDTH: al.constexpr,
    GROUP_SIZE: al.constexpr = 32,
    BLOCK_SIZE: al.constexpr = 1024,
):
    """Embedding lookup over normalized GGUF Q5_0 row blocks.

    qhigh layout matches the matvec kernel's: byte at index k/2 has
    the 5th bit of element k (even K) in bit 0 of the low nibble and
    the 5th bit of element k+1 (odd K) in bit 0 of the high nibble.
    """
    pid = al.program_id(0)
    offs = pid * BLOCK_SIZE + al.arange(0, BLOCK_SIZE)
    mask = offs < NUM_INDICES * WIDTH
    row = offs // WIDTH
    col = offs % WIDTH
    groups = WIDTH // GROUP_SIZE
    packed_width = WIDTH // 2
    idx = al.load(input_ids + row, mask=mask)
    raw = al.cast(al.load(qweight + idx * packed_width + col // 2, mask=mask), al.int32)
    nib = (raw >> ((col % 2) * 4)) & 0x0F
    hraw = al.cast(al.load(qhigh + idx * packed_width + col // 2, mask=mask), al.int32)
    high_bit = (hraw >> ((col % 2) * 4)) & 0x01
    q = al.cast((nib | (high_bit << 4)) - 16, al.float32)
    scale = al.cast(al.load(scales + idx * groups + col // GROUP_SIZE, mask=mask), al.float32)
    al.store(out + offs, scale * q, mask=mask)


@al.kernel
def embedding_q6_k(
    input_ids,
    qweight,
    out: al.output,
    NUM_INDICES: al.constexpr,
    WIDTH: al.constexpr,
    GROUP_SIZE: al.constexpr = 256,
    BLOCK_SIZE: al.constexpr = 1024,
):
    """Embedding lookup over GGUF Q6_K row blocks.

    Per 256-element super-block (210 bytes): 128B ql, 64B qh, 16B int8
    sub-scales, 2B fp16 super-scale. Decoder per element mirrors the Q6_K
    matvec decode.
    """
    BLOCK_BYTES = 210
    QL_BYTES = 128
    QH_BYTES = 64
    SCALE_BYTES = 16
    N_GROUPS = WIDTH // GROUP_SIZE
    ROW_BYTES = N_GROUPS * BLOCK_BYTES

    pid = al.program_id(0)
    offs = pid * BLOCK_SIZE + al.arange(0, BLOCK_SIZE)
    mask = offs < NUM_INDICES * WIDTH
    row = offs // WIDTH
    col = offs % WIDTH

    idx = al.load(input_ids + row, mask=mask)

    block_in_row = col // GROUP_SIZE
    pos = col % GROUP_SIZE
    subgroup = pos // 32
    lane = pos % 32

    block_base = idx * ROW_BYTES + block_in_row * BLOCK_BYTES
    scale_base = block_base + QL_BYTES + QH_BYTES
    d_base = scale_base + SCALE_BYTES

    d_lo = al.cast(al.load(qweight + d_base, mask=mask), "uint16")
    d_hi = al.cast(al.load(qweight + d_base + 1, mask=mask), "uint16")
    d_bits = al.cast(d_lo | (d_hi << 8), "uint16")
    d = al.cast(al.bitcast(d_bits, al.float16), al.float32)

    ql_block = subgroup // 4
    ql_shift = ((subgroup % 4) // 2) * 4
    ql_lane_base = (subgroup % 2) * 32
    ql_raw = al.cast(
        al.load(qweight + block_base + ql_block * 64 + ql_lane_base + lane, mask=mask),
        al.int32,
    )
    ql = (ql_raw >> ql_shift) & 0x0F

    qh_base = block_base + QL_BYTES + (subgroup // 4) * 32
    qh_shift = (subgroup % 4) * 2
    qh_raw = al.cast(al.load(qweight + qh_base + lane, mask=mask), al.int32)
    qh = (qh_raw >> qh_shift) & 0x03

    # int->float bit-trick (see dot_q6_k_v2): reinterpret 0x4B000000|u instead
    # of converting, then subtract 8388608+32 to recover q = u - 32.
    q = al.bitcast((ql | (qh << 4)) | 0x4B000000, al.float32) - 8388640.0

    scale_idx_lo = subgroup * 2
    scale_idx_hi = scale_idx_lo + 1
    scale_lo_raw = al.cast(al.load(qweight + scale_base + scale_idx_lo, mask=mask), al.int32)
    scale_hi_raw = al.cast(al.load(qweight + scale_base + scale_idx_hi, mask=mask), al.int32)
    scale_raw = al.where(lane < 16, scale_lo_raw, scale_hi_raw)

    weight = d * _q6k_s2f(scale_raw) * q
    al.store(out + offs, weight, mask=mask)


def _chain(terms):
    """Sum a python list of traced scalars without a leading 0.0 literal."""
    r = terms[0]
    for t in terms[1:]:
        r = r + t
    return r


def _q4k_u2f(x):
    # Unsigned int (x < 2^23) -> float via the bit-trick: 0x4B000000|x ==
    # 8388608.0f + x (exact, ULP=1 at 2^23), reinterpret + subtract instead of
    # the int->float CONVERT pipe. Bit-identical to float(x) (Sterbenz). Feeds
    # every native Q4_K matvec/MoE path (nibbles via _q4k_mf, 6-bit scales below).
    return al.bitcast(x | 0x4B000000, al.float32) - 8388608.0


def _q4k_mf(q, mask):
    return _q4k_u2f(q & mask)


def _q6k_s2f(raw):
    # Signed int8 (raw byte in [0,255]) -> float bit-trick. raw ^ 0x80 ==
    # signed_value + 128 in [0,255] (two's-complement -> offset-binary), so
    # reinterpret 0x4B000000|(raw^0x80) and subtract 8388608+128 to recover the
    # signed scale — dodges both the int->float CONVERT pipe AND the sign `where`.
    # Bit-identical to float((raw>127) ? raw-256 : raw) (|scale|<=128, Sterbenz).
    return al.bitcast((raw ^ 0x80) | 0x4B000000, al.float32) - 8388736.0


def _q4k_load_y(A, base):
    """yl[0..7]=y[0..7], yl[8..15]=y[32..39], yh[0..7]=y[128..135],
    yh[8..15]=y[160..167] (llama ix-stride), plus the 4 sub-block sums."""
    yl = [al.load(A + base + k) for k in (0, 1, 2, 3, 4, 5, 6, 7, 32, 33, 34, 35, 36, 37, 38, 39)]
    yh = [al.load(A + base + k) for k in (128, 129, 130, 131, 132, 133, 134, 135,
                                          160, 161, 162, 163, 164, 165, 166, 167)]
    sumy = (_chain(yl[0:8]), _chain(yl[8:16]), _chain(yh[0:8]), _chain(yh[8:16]))
    return yl, yh, sumy


def _q4k_contrib(BLK, blk, iq, ir, yl, yh, sumy):
    """One superblock, one row — llama's nibble-mask accumulation."""
    d = al.cast(al.load_wide(BLK + blk, "f16"), al.float32)
    dmin = al.cast(al.load_wide(BLK + blk + 2, "f16"), al.float32)
    sc0 = al.cast(al.load_wide(BLK + blk + 4 + iq * 2, "u16"), al.int32)
    sc2 = al.cast(al.load_wide(BLK + blk + 4 + (iq + 2) * 2, "u16"), al.int32)
    sc4 = al.cast(al.load_wide(BLK + blk + 4 + (iq + 4) * 2, "u16"), al.int32)
    sc16_0 = sc0 & 0x3F3F
    sc16_1 = sc2 & 0x3F3F
    sc16_2 = (sc4 & 0x0F0F) | ((sc0 & 0xC0C0) >> 2)
    sc16_3 = ((sc4 >> 4) & 0x0F0F) | ((sc2 & 0xC0C0) >> 2)
    sc8_0 = _q4k_u2f(sc16_0 & 0xFF)
    sc8_1 = _q4k_u2f((sc16_0 >> 8) & 0xFF)
    sc8_2 = _q4k_u2f(sc16_1 & 0xFF)
    sc8_3 = _q4k_u2f((sc16_1 >> 8) & 0xFF)
    sc8_4 = _q4k_u2f(sc16_2 & 0xFF)
    sc8_5 = _q4k_u2f((sc16_2 >> 8) & 0xFF)
    sc8_6 = _q4k_u2f(sc16_3 & 0xFF)
    sc8_7 = _q4k_u2f((sc16_3 >> 8) & 0xFF)
    qb = blk + 16 + (16 * iq + 4 * ir) * 2
    q1 = [al.cast(al.load_wide(BLK + qb + i * 2, "u16"), al.int32) for i in range(4)]
    q2 = [al.cast(al.load_wide(BLK + qb + 64 + i * 2, "u16"), al.int32) for i in range(4)]
    acc1_0 = _chain([yl[2 * i + 0] * _q4k_mf(q1[i], 0x000F) for i in range(4)])
    acc1_1 = _chain([yl[2 * i + 1] * _q4k_mf(q1[i], 0x0F00) for i in range(4)])
    acc1_2 = _chain([yl[2 * i + 8] * _q4k_mf(q1[i], 0x00F0) for i in range(4)])
    acc1_3 = _chain([yl[2 * i + 9] * _q4k_mf(q1[i], 0xF000) for i in range(4)])
    acc2_0 = _chain([yh[2 * i + 0] * _q4k_mf(q2[i], 0x000F) for i in range(4)])
    acc2_1 = _chain([yh[2 * i + 1] * _q4k_mf(q2[i], 0x0F00) for i in range(4)])
    acc2_2 = _chain([yh[2 * i + 8] * _q4k_mf(q2[i], 0x00F0) for i in range(4)])
    acc2_3 = _chain([yh[2 * i + 9] * _q4k_mf(q2[i], 0xF000) for i in range(4)])
    return d * (
        (acc1_0 + acc1_1 * (1.0 / 256.0)) * sc8_0
        + (acc1_2 + acc1_3 * (1.0 / 256.0)) * sc8_1 * (1.0 / 16.0)
        + (acc2_0 + acc2_1 * (1.0 / 256.0)) * sc8_4
        + (acc2_2 + acc2_3 * (1.0 / 256.0)) * sc8_5 * (1.0 / 16.0)
    ) - dmin * (sumy[0] * sc8_2 + sumy[1] * sc8_3 + sumy[2] * sc8_6 + sumy[3] * sc8_7)


def _q4k_silu(g):
    return g * (1.0 / (1.0 + al.exp(-g)))


@al.tunable(NSG=[1, 2, 4], NR0=[1, 2, 4])
@al.kernel
def dot_q4_k_v2(
    A,
    BLK,
    C: al.output,
    NSG: al.constexpr = 1,
    NR0: al.constexpr = 1,
):
    """Q4_K matvec over GGUF-native 144-byte superblocks.

    Literal port of llama.cpp mul_mv_q4_K_f32: ix-stride lane layout, NSG
    rows-parallel simdgroups (NR0 cols each), scalar yl/yh activation, the
    uint16 nibble-mask accumulation with the factored min term. No per-
    sub-block reduction.
    """
    M, K = A.shape
    N = BLK.shape[0]
    NB = K // 256
    NQ = NB // 4
    R = NB - 4 * NQ

    tid = al.arange(0, 32 * NSG)
    simd_id = tid // 32
    lane = tid - simd_id * 32
    ix = lane // 8
    it = lane - ix * 8
    iq = it // 4
    ir = it - iq * 4
    o = 64 * iq + 8 * ir
    col_base = (al.program_id(0) * NSG + simd_id) * NR0

    def _store(col_local, val, col):
        al.store(C + col_local, val, mask=(col < N) & (lane < 1))

    if NR0 == 1:
        for m in range(M):
            col0 = col_base + 0
            acc0 = 0.0
            for jj in range(NQ):
                ib = ix + 4 * jj
                yl, yh, sumy = _q4k_load_y(A, m * K + ib * 256 + o)
                acc0 = acc0 + _q4k_contrib(BLK, (col0 * NB + ib) * 144, iq, ir, yl, yh, sumy)
            if R > 0:
                ib = ix + 4 * NQ
                valid = ix < R
                ibc = al.where(valid, ib, 0)
                yl, yh, sumy = _q4k_load_y(A, m * K + ibc * 256 + o)
                acc0 = acc0 + al.where(valid, _q4k_contrib(BLK, (col0 * NB + ibc) * 144, iq, ir, yl, yh, sumy), 0.0)
            _store(m * N + col0, al.simd_reduce(acc0), col0)
    elif NR0 == 2:
        for m in range(M):
            col0 = col_base + 0
            col1 = col_base + 1
            acc0 = 0.0
            acc1 = 0.0
            for jj in range(NQ):
                ib = ix + 4 * jj
                yl, yh, sumy = _q4k_load_y(A, m * K + ib * 256 + o)
                acc0 = acc0 + _q4k_contrib(BLK, (col0 * NB + ib) * 144, iq, ir, yl, yh, sumy)
                acc1 = acc1 + _q4k_contrib(BLK, (col1 * NB + ib) * 144, iq, ir, yl, yh, sumy)
            if R > 0:
                ib = ix + 4 * NQ
                valid = ix < R
                ibc = al.where(valid, ib, 0)
                yl, yh, sumy = _q4k_load_y(A, m * K + ibc * 256 + o)
                acc0 = acc0 + al.where(valid, _q4k_contrib(BLK, (col0 * NB + ibc) * 144, iq, ir, yl, yh, sumy), 0.0)
                acc1 = acc1 + al.where(valid, _q4k_contrib(BLK, (col1 * NB + ibc) * 144, iq, ir, yl, yh, sumy), 0.0)
            _store(m * N + col0, al.simd_reduce(acc0), col0)
            _store(m * N + col1, al.simd_reduce(acc1), col1)
    else:  # NR0 == 4
        for m in range(M):
            col0 = col_base + 0
            col1 = col_base + 1
            col2 = col_base + 2
            col3 = col_base + 3
            acc0 = 0.0
            acc1 = 0.0
            acc2 = 0.0
            acc3 = 0.0
            for jj in range(NQ):
                ib = ix + 4 * jj
                yl, yh, sumy = _q4k_load_y(A, m * K + ib * 256 + o)
                acc0 = acc0 + _q4k_contrib(BLK, (col0 * NB + ib) * 144, iq, ir, yl, yh, sumy)
                acc1 = acc1 + _q4k_contrib(BLK, (col1 * NB + ib) * 144, iq, ir, yl, yh, sumy)
                acc2 = acc2 + _q4k_contrib(BLK, (col2 * NB + ib) * 144, iq, ir, yl, yh, sumy)
                acc3 = acc3 + _q4k_contrib(BLK, (col3 * NB + ib) * 144, iq, ir, yl, yh, sumy)
            if R > 0:
                ib = ix + 4 * NQ
                valid = ix < R
                ibc = al.where(valid, ib, 0)
                yl, yh, sumy = _q4k_load_y(A, m * K + ibc * 256 + o)
                acc0 = acc0 + al.where(valid, _q4k_contrib(BLK, (col0 * NB + ibc) * 144, iq, ir, yl, yh, sumy), 0.0)
                acc1 = acc1 + al.where(valid, _q4k_contrib(BLK, (col1 * NB + ibc) * 144, iq, ir, yl, yh, sumy), 0.0)
                acc2 = acc2 + al.where(valid, _q4k_contrib(BLK, (col2 * NB + ibc) * 144, iq, ir, yl, yh, sumy), 0.0)
                acc3 = acc3 + al.where(valid, _q4k_contrib(BLK, (col3 * NB + ibc) * 144, iq, ir, yl, yh, sumy), 0.0)
            _store(m * N + col0, al.simd_reduce(acc0), col0)
            _store(m * N + col1, al.simd_reduce(acc1), col1)
            _store(m * N + col2, al.simd_reduce(acc2), col2)
            _store(m * N + col3, al.simd_reduce(acc3), col3)


@al.tunable(NSG=[1, 2, 4], NR0=[1, 2, 4])
@al.kernel
def dot_q4_k_silu_v2(
    A,
    GATE,
    UP,
    C: al.output,
    NSG: al.constexpr = 1,
    NR0: al.constexpr = 1,
):
    """Q4_K gate+up matvec with silu fusion over GGUF-native superblocks.

    silu(A@gate.T) * (A@up.T). Same ix-stride native decode as dot_q4_k_v2;
    gate and up share the activation load per superblock.
    """
    M, K = A.shape
    N = GATE.shape[0]
    NB = K // 256
    NQ = NB // 4
    R = NB - 4 * NQ

    tid = al.arange(0, 32 * NSG)
    simd_id = tid // 32
    lane = tid - simd_id * 32
    ix = lane // 8
    it = lane - ix * 8
    iq = it // 4
    ir = it - iq * 4
    o = 64 * iq + 8 * ir
    col_base = (al.program_id(0) * NSG + simd_id) * NR0

    def _store(col_local, val, col):
        al.store(C + col_local, val, mask=(col < N) & (lane < 1))

    if NR0 == 1:
        for m in range(M):
            col0 = col_base + 0
            g0 = 0.0
            u0 = 0.0
            for jj in range(NQ):
                ib = ix + 4 * jj
                yl, yh, sumy = _q4k_load_y(A, m * K + ib * 256 + o)
                blk = (col0 * NB + ib) * 144
                g0 = g0 + _q4k_contrib(GATE, blk, iq, ir, yl, yh, sumy)
                u0 = u0 + _q4k_contrib(UP, blk, iq, ir, yl, yh, sumy)
            if R > 0:
                ib = ix + 4 * NQ
                valid = ix < R
                ibc = al.where(valid, ib, 0)
                yl, yh, sumy = _q4k_load_y(A, m * K + ibc * 256 + o)
                blk = (col0 * NB + ibc) * 144
                g0 = g0 + al.where(valid, _q4k_contrib(GATE, blk, iq, ir, yl, yh, sumy), 0.0)
                u0 = u0 + al.where(valid, _q4k_contrib(UP, blk, iq, ir, yl, yh, sumy), 0.0)
            g0 = al.simd_reduce(g0)
            u0 = al.simd_reduce(u0)
            _store(m * N + col0, _q4k_silu(g0) * u0, col0)
    elif NR0 == 2:
        for m in range(M):
            col0 = col_base + 0
            col1 = col_base + 1
            g0 = 0.0
            u0 = 0.0
            g1 = 0.0
            u1 = 0.0
            for jj in range(NQ):
                ib = ix + 4 * jj
                yl, yh, sumy = _q4k_load_y(A, m * K + ib * 256 + o)
                b0 = (col0 * NB + ib) * 144
                b1 = (col1 * NB + ib) * 144
                g0 = g0 + _q4k_contrib(GATE, b0, iq, ir, yl, yh, sumy)
                u0 = u0 + _q4k_contrib(UP, b0, iq, ir, yl, yh, sumy)
                g1 = g1 + _q4k_contrib(GATE, b1, iq, ir, yl, yh, sumy)
                u1 = u1 + _q4k_contrib(UP, b1, iq, ir, yl, yh, sumy)
            if R > 0:
                ib = ix + 4 * NQ
                valid = ix < R
                ibc = al.where(valid, ib, 0)
                yl, yh, sumy = _q4k_load_y(A, m * K + ibc * 256 + o)
                b0 = (col0 * NB + ibc) * 144
                b1 = (col1 * NB + ibc) * 144
                g0 = g0 + al.where(valid, _q4k_contrib(GATE, b0, iq, ir, yl, yh, sumy), 0.0)
                u0 = u0 + al.where(valid, _q4k_contrib(UP, b0, iq, ir, yl, yh, sumy), 0.0)
                g1 = g1 + al.where(valid, _q4k_contrib(GATE, b1, iq, ir, yl, yh, sumy), 0.0)
                u1 = u1 + al.where(valid, _q4k_contrib(UP, b1, iq, ir, yl, yh, sumy), 0.0)
            g0 = al.simd_reduce(g0)
            u0 = al.simd_reduce(u0)
            g1 = al.simd_reduce(g1)
            u1 = al.simd_reduce(u1)
            _store(m * N + col0, _q4k_silu(g0) * u0, col0)
            _store(m * N + col1, _q4k_silu(g1) * u1, col1)
    else:  # NR0 == 4
        for m in range(M):
            col0 = col_base + 0
            col1 = col_base + 1
            col2 = col_base + 2
            col3 = col_base + 3
            g0 = 0.0
            u0 = 0.0
            g1 = 0.0
            u1 = 0.0
            g2 = 0.0
            u2 = 0.0
            g3 = 0.0
            u3 = 0.0
            for jj in range(NQ):
                ib = ix + 4 * jj
                yl, yh, sumy = _q4k_load_y(A, m * K + ib * 256 + o)
                g0 = g0 + _q4k_contrib(GATE, (col0 * NB + ib) * 144, iq, ir, yl, yh, sumy)
                u0 = u0 + _q4k_contrib(UP, (col0 * NB + ib) * 144, iq, ir, yl, yh, sumy)
                g1 = g1 + _q4k_contrib(GATE, (col1 * NB + ib) * 144, iq, ir, yl, yh, sumy)
                u1 = u1 + _q4k_contrib(UP, (col1 * NB + ib) * 144, iq, ir, yl, yh, sumy)
                g2 = g2 + _q4k_contrib(GATE, (col2 * NB + ib) * 144, iq, ir, yl, yh, sumy)
                u2 = u2 + _q4k_contrib(UP, (col2 * NB + ib) * 144, iq, ir, yl, yh, sumy)
                g3 = g3 + _q4k_contrib(GATE, (col3 * NB + ib) * 144, iq, ir, yl, yh, sumy)
                u3 = u3 + _q4k_contrib(UP, (col3 * NB + ib) * 144, iq, ir, yl, yh, sumy)
            if R > 0:
                ib = ix + 4 * NQ
                valid = ix < R
                ibc = al.where(valid, ib, 0)
                yl, yh, sumy = _q4k_load_y(A, m * K + ibc * 256 + o)
                g0 = g0 + al.where(valid, _q4k_contrib(GATE, (col0 * NB + ibc) * 144, iq, ir, yl, yh, sumy), 0.0)
                u0 = u0 + al.where(valid, _q4k_contrib(UP, (col0 * NB + ibc) * 144, iq, ir, yl, yh, sumy), 0.0)
                g1 = g1 + al.where(valid, _q4k_contrib(GATE, (col1 * NB + ibc) * 144, iq, ir, yl, yh, sumy), 0.0)
                u1 = u1 + al.where(valid, _q4k_contrib(UP, (col1 * NB + ibc) * 144, iq, ir, yl, yh, sumy), 0.0)
                g2 = g2 + al.where(valid, _q4k_contrib(GATE, (col2 * NB + ibc) * 144, iq, ir, yl, yh, sumy), 0.0)
                u2 = u2 + al.where(valid, _q4k_contrib(UP, (col2 * NB + ibc) * 144, iq, ir, yl, yh, sumy), 0.0)
                g3 = g3 + al.where(valid, _q4k_contrib(GATE, (col3 * NB + ibc) * 144, iq, ir, yl, yh, sumy), 0.0)
                u3 = u3 + al.where(valid, _q4k_contrib(UP, (col3 * NB + ibc) * 144, iq, ir, yl, yh, sumy), 0.0)
            _store(m * N + col0, _q4k_silu(al.simd_reduce(g0)) * al.simd_reduce(u0), col0)
            _store(m * N + col1, _q4k_silu(al.simd_reduce(g1)) * al.simd_reduce(u1), col1)
            _store(m * N + col2, _q4k_silu(al.simd_reduce(g2)) * al.simd_reduce(u2), col2)
            _store(m * N + col3, _q4k_silu(al.simd_reduce(g3)) * al.simd_reduce(u3), col3)




@al.tunable(NSG=[1, 2, 4], NR0=[1, 2, 4])
@al.kernel
def dot_q4_k_gelu_v2(
    A,
    GATE,
    UP,
    C: al.output,
    NSG: al.constexpr = 1,
    NR0: al.constexpr = 1,
):
    """Q4_K gate+up matvec with gelu_tanh fusion over GGUF-native superblocks.

    gelu_tanh(A@gate.T) * (A@up.T). Same ix-stride native decode as
    dot_q4_k_v2; gate and up share the activation load per superblock.
    """
    M, K = A.shape
    N = GATE.shape[0]
    NB = K // 256
    NQ = NB // 4
    R = NB - 4 * NQ

    tid = al.arange(0, 32 * NSG)
    simd_id = tid // 32
    lane = tid - simd_id * 32
    ix = lane // 8
    it = lane - ix * 8
    iq = it // 4
    ir = it - iq * 4
    o = 64 * iq + 8 * ir
    col_base = (al.program_id(0) * NSG + simd_id) * NR0

    def _store(col_local, val, col):
        al.store(C + col_local, val, mask=(col < N) & (lane < 1))

    if NR0 == 1:
        for m in range(M):
            col0 = col_base + 0
            g0 = 0.0
            u0 = 0.0
            for jj in range(NQ):
                ib = ix + 4 * jj
                yl, yh, sumy = _q4k_load_y(A, m * K + ib * 256 + o)
                blk = (col0 * NB + ib) * 144
                g0 = g0 + _q4k_contrib(GATE, blk, iq, ir, yl, yh, sumy)
                u0 = u0 + _q4k_contrib(UP, blk, iq, ir, yl, yh, sumy)
            if R > 0:
                ib = ix + 4 * NQ
                valid = ix < R
                ibc = al.where(valid, ib, 0)
                yl, yh, sumy = _q4k_load_y(A, m * K + ibc * 256 + o)
                blk = (col0 * NB + ibc) * 144
                g0 = g0 + al.where(valid, _q4k_contrib(GATE, blk, iq, ir, yl, yh, sumy), 0.0)
                u0 = u0 + al.where(valid, _q4k_contrib(UP, blk, iq, ir, yl, yh, sumy), 0.0)
            g0 = al.simd_reduce(g0)
            u0 = al.simd_reduce(u0)
            _store(m * N + col0, al.gelu_tanh(g0) * u0, col0)
    elif NR0 == 2:
        for m in range(M):
            col0 = col_base + 0
            col1 = col_base + 1
            g0 = 0.0
            u0 = 0.0
            g1 = 0.0
            u1 = 0.0
            for jj in range(NQ):
                ib = ix + 4 * jj
                yl, yh, sumy = _q4k_load_y(A, m * K + ib * 256 + o)
                b0 = (col0 * NB + ib) * 144
                b1 = (col1 * NB + ib) * 144
                g0 = g0 + _q4k_contrib(GATE, b0, iq, ir, yl, yh, sumy)
                u0 = u0 + _q4k_contrib(UP, b0, iq, ir, yl, yh, sumy)
                g1 = g1 + _q4k_contrib(GATE, b1, iq, ir, yl, yh, sumy)
                u1 = u1 + _q4k_contrib(UP, b1, iq, ir, yl, yh, sumy)
            if R > 0:
                ib = ix + 4 * NQ
                valid = ix < R
                ibc = al.where(valid, ib, 0)
                yl, yh, sumy = _q4k_load_y(A, m * K + ibc * 256 + o)
                b0 = (col0 * NB + ibc) * 144
                b1 = (col1 * NB + ibc) * 144
                g0 = g0 + al.where(valid, _q4k_contrib(GATE, b0, iq, ir, yl, yh, sumy), 0.0)
                u0 = u0 + al.where(valid, _q4k_contrib(UP, b0, iq, ir, yl, yh, sumy), 0.0)
                g1 = g1 + al.where(valid, _q4k_contrib(GATE, b1, iq, ir, yl, yh, sumy), 0.0)
                u1 = u1 + al.where(valid, _q4k_contrib(UP, b1, iq, ir, yl, yh, sumy), 0.0)
            g0 = al.simd_reduce(g0)
            u0 = al.simd_reduce(u0)
            g1 = al.simd_reduce(g1)
            u1 = al.simd_reduce(u1)
            _store(m * N + col0, al.gelu_tanh(g0) * u0, col0)
            _store(m * N + col1, al.gelu_tanh(g1) * u1, col1)
    else:  # NR0 == 4
        for m in range(M):
            col0 = col_base + 0
            col1 = col_base + 1
            col2 = col_base + 2
            col3 = col_base + 3
            g0 = 0.0
            u0 = 0.0
            g1 = 0.0
            u1 = 0.0
            g2 = 0.0
            u2 = 0.0
            g3 = 0.0
            u3 = 0.0
            for jj in range(NQ):
                ib = ix + 4 * jj
                yl, yh, sumy = _q4k_load_y(A, m * K + ib * 256 + o)
                g0 = g0 + _q4k_contrib(GATE, (col0 * NB + ib) * 144, iq, ir, yl, yh, sumy)
                u0 = u0 + _q4k_contrib(UP, (col0 * NB + ib) * 144, iq, ir, yl, yh, sumy)
                g1 = g1 + _q4k_contrib(GATE, (col1 * NB + ib) * 144, iq, ir, yl, yh, sumy)
                u1 = u1 + _q4k_contrib(UP, (col1 * NB + ib) * 144, iq, ir, yl, yh, sumy)
                g2 = g2 + _q4k_contrib(GATE, (col2 * NB + ib) * 144, iq, ir, yl, yh, sumy)
                u2 = u2 + _q4k_contrib(UP, (col2 * NB + ib) * 144, iq, ir, yl, yh, sumy)
                g3 = g3 + _q4k_contrib(GATE, (col3 * NB + ib) * 144, iq, ir, yl, yh, sumy)
                u3 = u3 + _q4k_contrib(UP, (col3 * NB + ib) * 144, iq, ir, yl, yh, sumy)
            if R > 0:
                ib = ix + 4 * NQ
                valid = ix < R
                ibc = al.where(valid, ib, 0)
                yl, yh, sumy = _q4k_load_y(A, m * K + ibc * 256 + o)
                g0 = g0 + al.where(valid, _q4k_contrib(GATE, (col0 * NB + ibc) * 144, iq, ir, yl, yh, sumy), 0.0)
                u0 = u0 + al.where(valid, _q4k_contrib(UP, (col0 * NB + ibc) * 144, iq, ir, yl, yh, sumy), 0.0)
                g1 = g1 + al.where(valid, _q4k_contrib(GATE, (col1 * NB + ibc) * 144, iq, ir, yl, yh, sumy), 0.0)
                u1 = u1 + al.where(valid, _q4k_contrib(UP, (col1 * NB + ibc) * 144, iq, ir, yl, yh, sumy), 0.0)
                g2 = g2 + al.where(valid, _q4k_contrib(GATE, (col2 * NB + ibc) * 144, iq, ir, yl, yh, sumy), 0.0)
                u2 = u2 + al.where(valid, _q4k_contrib(UP, (col2 * NB + ibc) * 144, iq, ir, yl, yh, sumy), 0.0)
                g3 = g3 + al.where(valid, _q4k_contrib(GATE, (col3 * NB + ibc) * 144, iq, ir, yl, yh, sumy), 0.0)
                u3 = u3 + al.where(valid, _q4k_contrib(UP, (col3 * NB + ibc) * 144, iq, ir, yl, yh, sumy), 0.0)
            _store(m * N + col0, al.gelu_tanh(al.simd_reduce(g0)) * al.simd_reduce(u0), col0)
            _store(m * N + col1, al.gelu_tanh(al.simd_reduce(g1)) * al.simd_reduce(u1), col1)
            _store(m * N + col2, al.gelu_tanh(al.simd_reduce(g2)) * al.simd_reduce(u2), col2)
            _store(m * N + col3, al.gelu_tanh(al.simd_reduce(g3)) * al.simd_reduce(u3), col3)


def _q4k_decode_w(BLK, blk, iq, ir):
    """Decode one Q4_K superblock row WITHOUT activation (for row amortization):
    returns (d, dmin, q1, q2, sc8) so a weight read is shared across M rows."""
    d = al.cast(al.load_wide(BLK + blk, "f16"), al.float32)
    dmin = al.cast(al.load_wide(BLK + blk + 2, "f16"), al.float32)
    sc0 = al.cast(al.load_wide(BLK + blk + 4 + iq * 2, "u16"), al.int32)
    sc2 = al.cast(al.load_wide(BLK + blk + 4 + (iq + 2) * 2, "u16"), al.int32)
    sc4 = al.cast(al.load_wide(BLK + blk + 4 + (iq + 4) * 2, "u16"), al.int32)
    sc16_0 = sc0 & 0x3F3F
    sc16_1 = sc2 & 0x3F3F
    sc16_2 = (sc4 & 0x0F0F) | ((sc0 & 0xC0C0) >> 2)
    sc16_3 = ((sc4 >> 4) & 0x0F0F) | ((sc2 & 0xC0C0) >> 2)
    sc8 = (
        _q4k_u2f(sc16_0 & 0xFF),
        _q4k_u2f((sc16_0 >> 8) & 0xFF),
        _q4k_u2f(sc16_1 & 0xFF),
        _q4k_u2f((sc16_1 >> 8) & 0xFF),
        _q4k_u2f(sc16_2 & 0xFF),
        _q4k_u2f((sc16_2 >> 8) & 0xFF),
        _q4k_u2f(sc16_3 & 0xFF),
        _q4k_u2f((sc16_3 >> 8) & 0xFF),
    )
    qb = blk + 16 + (16 * iq + 4 * ir) * 2
    q1 = [al.cast(al.load_wide(BLK + qb + i * 2, "u16"), al.int32) for i in range(4)]
    q2 = [al.cast(al.load_wide(BLK + qb + 64 + i * 2, "u16"), al.int32) for i in range(4)]
    return d, dmin, q1, q2, sc8


def _q4k_apply(w, yl, yh, sumy):
    """Contribution for one row given a pre-decoded weight (_q4k_decode_w)."""
    d, dmin, q1, q2, sc8 = w
    acc1_0 = _chain([yl[2 * i + 0] * _q4k_mf(q1[i], 0x000F) for i in range(4)])
    acc1_1 = _chain([yl[2 * i + 1] * _q4k_mf(q1[i], 0x0F00) for i in range(4)])
    acc1_2 = _chain([yl[2 * i + 8] * _q4k_mf(q1[i], 0x00F0) for i in range(4)])
    acc1_3 = _chain([yl[2 * i + 9] * _q4k_mf(q1[i], 0xF000) for i in range(4)])
    acc2_0 = _chain([yh[2 * i + 0] * _q4k_mf(q2[i], 0x000F) for i in range(4)])
    acc2_1 = _chain([yh[2 * i + 1] * _q4k_mf(q2[i], 0x0F00) for i in range(4)])
    acc2_2 = _chain([yh[2 * i + 8] * _q4k_mf(q2[i], 0x00F0) for i in range(4)])
    acc2_3 = _chain([yh[2 * i + 9] * _q4k_mf(q2[i], 0xF000) for i in range(4)])
    return d * (
        (acc1_0 + acc1_1 * (1.0 / 256.0)) * sc8[0]
        + (acc1_2 + acc1_3 * (1.0 / 256.0)) * sc8[1] * (1.0 / 16.0)
        + (acc2_0 + acc2_1 * (1.0 / 256.0)) * sc8[4]
        + (acc2_2 + acc2_3 * (1.0 / 256.0)) * sc8[5] * (1.0 / 16.0)
    ) - dmin * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7])


@al.tunable()
@al.kernel
def dot_q4_k_silu_v2_rows(
    A,
    GATE,
    UP,
    C: al.output,
):
    """Q4_K gate+up GEMV with silu fusion for verify M (2..8): decode each
    weight superblock ONCE (gate and up), apply to all M rows. The FFN gate+up
    is the majority of a layer's weight bytes, so re-reading them per row is
    what put verify ~40% over the memory roofline. One program per output col
    (grid (N,)); native 144-byte superblocks, ix-stride; 16 named accumulators
    (8 gate + 8 up, the verify K ceiling). Rows >= M clamp to M-1, only M
    stored. M==1 decode keeps using dot_q4_k_silu_v2."""
    M, K = A.shape
    N = GATE.shape[0]
    NB = K // 256
    NQ = NB // 4
    R = NB - 4 * NQ
    col = al.program_id(0)
    tid = al.arange(0, 32)
    ix = tid // 8
    it = tid - ix * 8
    iq = it // 4
    ir = it - iq * 4
    o = 64 * iq + 8 * ir
    r0 = min(0, M - 1)
    r1 = min(1, M - 1)
    r2 = min(2, M - 1)
    r3 = min(3, M - 1)
    r4 = min(4, M - 1)
    r5 = min(5, M - 1)
    r6 = min(6, M - 1)
    r7 = min(7, M - 1)
    ag0 = 0.0
    ag1 = 0.0
    ag2 = 0.0
    ag3 = 0.0
    ag4 = 0.0
    ag5 = 0.0
    ag6 = 0.0
    ag7 = 0.0
    au0 = 0.0
    au1 = 0.0
    au2 = 0.0
    au3 = 0.0
    au4 = 0.0
    au5 = 0.0
    au6 = 0.0
    au7 = 0.0
    for jj in range(NQ):
        ib = ix + 4 * jj
        blk = (col * NB + ib) * 144
        wg = _q4k_decode_w(GATE, blk, iq, ir)
        wu = _q4k_decode_w(UP, blk, iq, ir)
        y0 = _q4k_load_y(A, r0 * K + ib * 256 + o)
        ag0 = ag0 + _q4k_apply(wg, y0[0], y0[1], y0[2])
        au0 = au0 + _q4k_apply(wu, y0[0], y0[1], y0[2])
        y1 = _q4k_load_y(A, r1 * K + ib * 256 + o)
        ag1 = ag1 + _q4k_apply(wg, y1[0], y1[1], y1[2])
        au1 = au1 + _q4k_apply(wu, y1[0], y1[1], y1[2])
        y2 = _q4k_load_y(A, r2 * K + ib * 256 + o)
        ag2 = ag2 + _q4k_apply(wg, y2[0], y2[1], y2[2])
        au2 = au2 + _q4k_apply(wu, y2[0], y2[1], y2[2])
        y3 = _q4k_load_y(A, r3 * K + ib * 256 + o)
        ag3 = ag3 + _q4k_apply(wg, y3[0], y3[1], y3[2])
        au3 = au3 + _q4k_apply(wu, y3[0], y3[1], y3[2])
        y4 = _q4k_load_y(A, r4 * K + ib * 256 + o)
        ag4 = ag4 + _q4k_apply(wg, y4[0], y4[1], y4[2])
        au4 = au4 + _q4k_apply(wu, y4[0], y4[1], y4[2])
        y5 = _q4k_load_y(A, r5 * K + ib * 256 + o)
        ag5 = ag5 + _q4k_apply(wg, y5[0], y5[1], y5[2])
        au5 = au5 + _q4k_apply(wu, y5[0], y5[1], y5[2])
        y6 = _q4k_load_y(A, r6 * K + ib * 256 + o)
        ag6 = ag6 + _q4k_apply(wg, y6[0], y6[1], y6[2])
        au6 = au6 + _q4k_apply(wu, y6[0], y6[1], y6[2])
        y7 = _q4k_load_y(A, r7 * K + ib * 256 + o)
        ag7 = ag7 + _q4k_apply(wg, y7[0], y7[1], y7[2])
        au7 = au7 + _q4k_apply(wu, y7[0], y7[1], y7[2])
    if R > 0:
        ib = ix + 4 * NQ
        valid = ix < R
        ibc = al.where(valid, ib, 0)
        blk = (col * NB + ibc) * 144
        wg = _q4k_decode_w(GATE, blk, iq, ir)
        wu = _q4k_decode_w(UP, blk, iq, ir)
        y0 = _q4k_load_y(A, r0 * K + ibc * 256 + o)
        ag0 = ag0 + al.where(valid, _q4k_apply(wg, y0[0], y0[1], y0[2]), 0.0)
        au0 = au0 + al.where(valid, _q4k_apply(wu, y0[0], y0[1], y0[2]), 0.0)
        y1 = _q4k_load_y(A, r1 * K + ibc * 256 + o)
        ag1 = ag1 + al.where(valid, _q4k_apply(wg, y1[0], y1[1], y1[2]), 0.0)
        au1 = au1 + al.where(valid, _q4k_apply(wu, y1[0], y1[1], y1[2]), 0.0)
        y2 = _q4k_load_y(A, r2 * K + ibc * 256 + o)
        ag2 = ag2 + al.where(valid, _q4k_apply(wg, y2[0], y2[1], y2[2]), 0.0)
        au2 = au2 + al.where(valid, _q4k_apply(wu, y2[0], y2[1], y2[2]), 0.0)
        y3 = _q4k_load_y(A, r3 * K + ibc * 256 + o)
        ag3 = ag3 + al.where(valid, _q4k_apply(wg, y3[0], y3[1], y3[2]), 0.0)
        au3 = au3 + al.where(valid, _q4k_apply(wu, y3[0], y3[1], y3[2]), 0.0)
        y4 = _q4k_load_y(A, r4 * K + ibc * 256 + o)
        ag4 = ag4 + al.where(valid, _q4k_apply(wg, y4[0], y4[1], y4[2]), 0.0)
        au4 = au4 + al.where(valid, _q4k_apply(wu, y4[0], y4[1], y4[2]), 0.0)
        y5 = _q4k_load_y(A, r5 * K + ibc * 256 + o)
        ag5 = ag5 + al.where(valid, _q4k_apply(wg, y5[0], y5[1], y5[2]), 0.0)
        au5 = au5 + al.where(valid, _q4k_apply(wu, y5[0], y5[1], y5[2]), 0.0)
        y6 = _q4k_load_y(A, r6 * K + ibc * 256 + o)
        ag6 = ag6 + al.where(valid, _q4k_apply(wg, y6[0], y6[1], y6[2]), 0.0)
        au6 = au6 + al.where(valid, _q4k_apply(wu, y6[0], y6[1], y6[2]), 0.0)
        y7 = _q4k_load_y(A, r7 * K + ibc * 256 + o)
        ag7 = ag7 + al.where(valid, _q4k_apply(wg, y7[0], y7[1], y7[2]), 0.0)
        au7 = au7 + al.where(valid, _q4k_apply(wu, y7[0], y7[1], y7[2]), 0.0)
    accg = [ag0, ag1, ag2, ag3, ag4, ag5, ag6, ag7]
    accu = [au0, au1, au2, au3, au4, au5, au6, au7]
    for m in _unroll(M):
        g = al.simd_reduce(accg[m])
        u = al.simd_reduce(accu[m])
        al.store(C + m * N + col, _q4k_silu(g) * u, mask=(col < N) & (tid < 1))


@al.tunable(
    BLOCK_M=[8, 16, 32, 64, 128],
    BLOCK_N=[16, 32, 64, 128],
    BLOCK_K=[32, 64, 128],
)
@al.kernel
def dot_q4_k_silu(
    A,
    GATE,
    UP,
    C: al.output,
    BLOCK_M: al.constexpr = 16,
    BLOCK_N: al.constexpr = 32,
    BLOCK_K: al.constexpr = 64,
):
    """GGUF-native Q4_K paired matmul: C = silu(A @ gate.T) * (A @ up.T)."""
    M, K = A.shape
    N = GATE.shape[0]
    BLOCK_BYTES = 144
    N_GROUPS = K // 256
    ROW_BYTES = N_GROUPS * BLOCK_BYTES

    pm = al.program_id(0)
    pn = al.program_id(1)
    rm = pm * BLOCK_M + al.arange(0, BLOCK_M)
    rn = pn * BLOCK_N + al.arange(0, BLOCK_N)
    rk = al.arange(0, BLOCK_K)

    a_ptrs = A + rm[:, None] * K + rk[None, :]
    acc_gate = al.zeros((BLOCK_M, BLOCK_N), dtype=al.float32)
    acc_up = al.zeros((BLOCK_M, BLOCK_N), dtype=al.float32)

    for k in range(0, K, BLOCK_K):
        elem_k = k + rk
        a = al.load(a_ptrs, mask=(rm[:, None] < M) & (elem_k[None, :] < K))
        gate = al.load(
            GATE + rn[:, None] * ROW_BYTES + elem_k[None, :],
            mask=(rn[:, None] < N) & (elem_k[None, :] < K),
            _dequant_format="q4_k",
        )
        up = al.load(
            UP + rn[:, None] * ROW_BYTES + elem_k[None, :],
            mask=(rn[:, None] < N) & (elem_k[None, :] < K),
            _dequant_format="q4_k",
        )
        acc_gate += al.tile_dot(a, gate, transpose_rhs=True)
        acc_up += al.tile_dot(a, up, transpose_rhs=True)
        a_ptrs += BLOCK_K

    silu = acc_gate * (1.0 / (1.0 + al.exp(-acc_gate))) * acc_up
    c_ptrs = C + rm[:, None] * N + rn[None, :]
    al.store(c_ptrs, silu, mask=(rm[:, None] < M) & (rn[None, :] < N))


@al.tunable(
    BLOCK_M=[8, 16, 32, 64, 128],
    BLOCK_N=[32, 64, 128],
    BLOCK_K=[32, 64, 128, 256],
)
@al.kernel
def dot_q6_k(
    A,
    B_q6,
    C: al.output,
    GROUP_SIZE: al.constexpr = 256,
    BLOCK_M: al.constexpr = 16,
    BLOCK_N: al.constexpr = 64,
    BLOCK_K: al.constexpr = 64,
):
    """GGUF Q6_K matmul: C = A @ dequant(B_q6).T.

    Tile-MMA over the cooperative load's fused Q6_K dequant. The M=1
    matvec specialization lives in the dedicated `dot_q6_k_v2` kernel.
    """
    M, K = A.shape
    N = B_q6.shape[0]
    BLOCK_BYTES = 210
    N_GROUPS = K // GROUP_SIZE
    ROW_BYTES = N_GROUPS * BLOCK_BYTES

    pm = al.program_id(0)
    pn = al.program_id(1)
    rm = pm * BLOCK_M + al.arange(0, BLOCK_M)
    rn = pn * BLOCK_N + al.arange(0, BLOCK_N)
    rk = al.arange(0, BLOCK_K)

    a_ptrs = A + rm[:, None] * K + rk[None, :]
    acc = al.zeros((BLOCK_M, BLOCK_N), dtype=al.float32)

    for k in range(0, K, BLOCK_K):
        elem_k = k + rk
        a = al.load(a_ptrs, mask=(rm[:, None] < M) & (elem_k[None, :] < K))
        b = al.load(
            B_q6 + rn[:, None] * ROW_BYTES + elem_k[None, :],
            mask=(rn[:, None] < N) & (elem_k[None, :] < K),
            _dequant_format="q6_k",
        )
        acc += al.tile_dot(a, b, transpose_rhs=True)
        a_ptrs += BLOCK_K

    c_ptrs = C + rm[:, None] * N + rn[None, :]
    al.store(c_ptrs, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


@al.tunable(NUM_SPLITS=[1, 2, 4])
@al.kernel
def dot_q6_k_v2(
    A,
    B_q6,
    C: al.output,
    GROUP_SIZE: al.constexpr = 256,
    NUM_SPLITS: al.constexpr = 1,
):
    """GGUF Q6_K matvec with vec4 K-vectorization.

    Lane layout per quadrant (128 K positions, lanes 0..31):
      subgroup_in_quad = lane // 8 (0..3)
      K_local        = (lane % 8) * 4  (0..28, then per subgroup)
      Quadrant covers K[quadrant*128 .. quadrant*128 + 127]
      Lane handles K[quadrant*128 + subgroup_in_quad*32 + K_local + (0..3)]

    QL byte b in quadrant (b in 0..63):
      low nibble  = K[subgroup0 base + b] = K[quadrant*128 + b] low 4 bits
                                          (for b in 0..31; for b in 32..63: subgroup 1)
      high nibble = K[subgroup2 base + b] = K[quadrant*128 + 64 + b] low 4 bits
                                          (for b in 0..31; for b in 32..63: subgroup 3)
    → lanes 0..15 use LOW nibbles of QL; lanes 16..31 use HIGH nibbles.
       Each lane reads 4 contiguous QL bytes at offset (lane % 16) * 4.

    QH byte at offset b in quadrant (b in 0..31):
      shift 0  = K[quadrant*128 + b] high 2 bits (subgroup 0)
      shift 2  = K[quadrant*128 + 32 + b] high 2 bits (subgroup 1)
      shift 4  = K[quadrant*128 + 64 + b] high 2 bits (subgroup 2)
      shift 6  = K[quadrant*128 + 96 + b] high 2 bits (subgroup 3)
    → all lanes in subgroup s use shift s*2. Each lane reads 4 contiguous QH
      bytes at offset (lane % 8) * 4.

    Scales: 16 bytes per block (2 per subgroup, one for lanes 0..15 of the
    subgroup-half and one for lanes 16..31). Effective group size for scales
    is 16 K. In our layout each lane covers 4 K, so 4 lanes share a scale →
    scale_idx = (lane // 4) + quadrant * 8.

    Per lane per quadrant the kernel issues:
      1× load4_vec on QL (4 bytes)
      1× load4_vec on QH (4 bytes)
      1× load4_vec on A  (4 fp32)
      1× scalar load on scale (1 byte)
    That keeps load count bounded by lanes instead of per-byte unpack work.
    """
    M, K = A.shape
    N = B_q6.shape[0]
    BLOCK_BYTES = 210
    QL_BYTES = 128
    QH_BYTES = 64
    SCALE_BYTES = 16
    N_GROUPS = K // GROUP_SIZE
    GROUPS_PER_SPLIT = N_GROUPS // NUM_SPLITS

    col = al.program_id(0)
    # NUM_SPLITS simdgroups per TG, each reducing GROUPS_PER_SPLIT groups of the
    # K range; partials combined via shmem. Raises the simdgroup count NUM_SPLITS×
    # without growing the grid — recovers occupancy on small-N rows (e.g. down_proj
    # N=2048, K=8192: 1 simdgroup can't hide the long K-scan's memory latency).
    tid = al.arange(0, 32 * NUM_SPLITS)
    simd_id = tid // 32
    lane = tid - simd_id * 32

    def _store_combined(col_local, val, col):
        if NUM_SPLITS == 1:
            al.store(C + col_local, val, mask=(col < N) & (lane < 1))
        else:
            shm = al.shared(NUM_SPLITS, dtype=al.float32)
            al.store(shm + simd_id, val, mask=(lane < 1))
            al.barrier()
            partials = [al.load(shm + s) for s in list(range(NUM_SPLITS))]
            total = partials[0]
            for s in list(range(1, NUM_SPLITS)):
                total = total + partials[s]
            al.store(C + col_local, total, mask=(col < N) & (simd_id < 1) & (lane < 1))

    for m in range(M):
        acc = 0.0
        for g_local in range(GROUPS_PER_SPLIT):
            g = simd_id * GROUPS_PER_SPLIT + g_local
            block_base = col * (N_GROUPS * BLOCK_BYTES) + g * BLOCK_BYTES
            scale_base = block_base + QL_BYTES + QH_BYTES
            d_base = scale_base + SCALE_BYTES

            d_lo = al.cast(al.load(B_q6 + d_base), "uint16")
            d_hi = al.cast(al.load(B_q6 + d_base + 1), "uint16")
            d_bits = al.cast(d_lo | (d_hi << 8), "uint16")
            d = al.cast(al.bitcast(d_bits, al.float16), al.float32)

            for quadrant in range(2):
                ql_base = block_base + quadrant * 64
                qh_base = block_base + QL_BYTES + quadrant * 32

                # Each lane reads 4 QL bytes; lanes 0..15 and 16..31 read the
                # same physical bytes (different nibble halves). Express the
                # nibble pick as a per-lane shift to avoid a vec4 select
                # (alloy's Select emit only handles scalar results today).
                ql_v4 = al.load4_vec(B_q6 + ql_base + (lane % 16) * 4)
                nibble_shift = al.where(lane < 16, 0, 4)
                ql_nibbles = (ql_v4 >> nibble_shift) & 0x0F

                # Each lane reads 4 QH bytes; the 2-bit field selected by
                # shift depends on the lane's subgroup-in-quadrant.
                qh_v4 = al.load4_vec(B_q6 + qh_base + (lane % 8) * 4)
                qh_shift = (lane // 8) * 2
                qh_bits = (qh_v4 >> qh_shift) & 0x03

                # Combine low 4 bits + high 2 bits → 6-bit q in [0, 63], then
                # recover signed q = u - 32 via the int→float BIT-TRICK instead
                # of a convert: 0x4B000000 | u == 8388608.0f + u (exact for
                # u < 2^23, ULP=1 at 2^23), so bitcast - (8388608+32) gives
                # u - 32 on the F32-add pipe, dodging the int→float CONVERT pipe
                # (~14% of this kernel in the Xcode GPU trace). Bit-identical to
                # `float(int(u) - 32)`: 8388608+u and 8388640 are exact and their
                # difference (|·|≤32) is exact by Sterbenz.
                q_combined = ql_nibbles | (qh_bits << 4)
                q_bits = al.cast(q_combined, al.int32) | 0x4B000000
                q_f4 = al.bitcast(q_bits, al.float32) - 8388640.0

                # Activations: 4 fp32 per lane.
                k_in_quad = (lane // 8) * 32 + (lane % 8) * 4
                a_v4 = al.load4_vec(A + m * K + g * GROUP_SIZE + quadrant * 128 + k_in_quad)

                # Scale: 4 lanes share one byte (lanes 0..3 → scale[0], etc.).
                scale_idx = (lane // 4) + quadrant * 8
                scale_raw = al.cast(al.load(B_q6 + scale_base + scale_idx), al.int32)
                scale_f = _q6k_s2f(scale_raw)

                # Per-lane: dot of 4 activations × 4 weights, then scale and
                # block scaling outside the dot4 to keep it cheap.
                acc = acc + d * scale_f * al.dot4(a_v4, q_f4)

        acc = al.simd_reduce(acc)
        _store_combined(m * N + col, acc, col)


def _unroll(n: int) -> tuple[int, ...]:
    """`range`-alias the kernel AST-rewriter leaves as a real Python loop (it
    only rewrites `for x in range(...)`), so loops over it unroll at trace time
    with concrete indices — needed to index the per-row accumulator list."""
    return tuple(range(n))


@al.tunable()
@al.kernel
def dot_q6_k_v2_rows(A, B_q6, C: al.output, GROUP_SIZE: al.constexpr = 256):
    """Q6_K GEMV for verify M (2..8): dequant each block ONCE, read weights
    ONCE, accumulate all rows. TRACED g/quadrant loops (compact MSL; the full
    `_unroll` form collides SSA load names under epilogue fusion) + 8 named
    accumulators (verify K ceiling), rows >= M clamped to M-1, only M stored.
    M==1 decode keeps using dot_q6_k_v2. Crucially the lm_head (q6_k, 248K
    vocab) verifies all rows for one weight read instead of an 8-row MMA."""
    M, K = A.shape
    N = B_q6.shape[0]
    BLOCK_BYTES = 210
    QL_BYTES = 128
    QH_BYTES = 64
    SCALE_BYTES = 16
    N_GROUPS = K // GROUP_SIZE
    col = al.program_id(0)
    lane = al.arange(0, 32)
    r0 = min(0, M - 1)
    r1 = min(1, M - 1)
    r2 = min(2, M - 1)
    r3 = min(3, M - 1)
    r4 = min(4, M - 1)
    r5 = min(5, M - 1)
    r6 = min(6, M - 1)
    r7 = min(7, M - 1)
    acc0 = 0.0
    acc1 = 0.0
    acc2 = 0.0
    acc3 = 0.0
    acc4 = 0.0
    acc5 = 0.0
    acc6 = 0.0
    acc7 = 0.0
    for g in range(N_GROUPS):
        block_base = col * (N_GROUPS * BLOCK_BYTES) + g * BLOCK_BYTES
        scale_base = block_base + QL_BYTES + QH_BYTES
        d_base = scale_base + SCALE_BYTES
        d_lo = al.cast(al.load(B_q6 + d_base), "uint16")
        d_hi = al.cast(al.load(B_q6 + d_base + 1), "uint16")
        d = al.cast(al.bitcast(al.cast(d_lo | (d_hi << 8), "uint16"), al.float16), al.float32)
        for quadrant in range(2):
            ql_base = block_base + quadrant * 64
            qh_base = block_base + QL_BYTES + quadrant * 32
            ql_v4 = al.load4_vec(B_q6 + ql_base + (lane % 16) * 4)
            ql_nibbles = (ql_v4 >> al.where(lane < 16, 0, 4)) & 0x0F
            qh_v4 = al.load4_vec(B_q6 + qh_base + (lane % 8) * 4)
            qh_bits = (qh_v4 >> ((lane // 8) * 2)) & 0x03
            q_f4 = al.bitcast(al.cast(ql_nibbles | (qh_bits << 4), al.int32) | 0x4B000000, al.float32) - 8388640.0
            scale_raw = al.cast(al.load(B_q6 + scale_base + (lane // 4) + quadrant * 8), al.int32)
            ds = d * _q6k_s2f(scale_raw)
            base_k = g * GROUP_SIZE + quadrant * 128 + (lane // 8) * 32 + (lane % 8) * 4
            acc0 = acc0 + ds * al.dot4(al.load4_vec(A + r0 * K + base_k), q_f4)
            acc1 = acc1 + ds * al.dot4(al.load4_vec(A + r1 * K + base_k), q_f4)
            acc2 = acc2 + ds * al.dot4(al.load4_vec(A + r2 * K + base_k), q_f4)
            acc3 = acc3 + ds * al.dot4(al.load4_vec(A + r3 * K + base_k), q_f4)
            acc4 = acc4 + ds * al.dot4(al.load4_vec(A + r4 * K + base_k), q_f4)
            acc5 = acc5 + ds * al.dot4(al.load4_vec(A + r5 * K + base_k), q_f4)
            acc6 = acc6 + ds * al.dot4(al.load4_vec(A + r6 * K + base_k), q_f4)
            acc7 = acc7 + ds * al.dot4(al.load4_vec(A + r7 * K + base_k), q_f4)
    accs = [acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7]
    for m in _unroll(M):
        al.store(C + m * N + col, al.simd_reduce(accs[m]), mask=(col < N) & (lane < 1))


@al.tunable()
@al.kernel
def dot_q4_k_v2_rows(A, BLK, C: al.output):
    """Q4_K GEMV for verify M (2..8): decode each weight superblock ONCE,
    apply to all M rows. Native 144-byte superblocks, ix-stride; one program
    per output col (grid (N,)); 8 named accumulators (the verify K ceiling)
    and clamp rows >= M to row M-1, storing only M. M==1 decode keeps using
    dot_q4_k_v2."""
    M, K = A.shape
    N = BLK.shape[0]
    NB = K // 256
    NQ = NB // 4
    R = NB - 4 * NQ
    col = al.program_id(0)
    tid = al.arange(0, 32)
    ix = tid // 8
    it = tid - ix * 8
    iq = it // 4
    ir = it - iq * 4
    o = 64 * iq + 8 * ir
    r0 = min(0, M - 1)
    r1 = min(1, M - 1)
    r2 = min(2, M - 1)
    r3 = min(3, M - 1)
    r4 = min(4, M - 1)
    r5 = min(5, M - 1)
    r6 = min(6, M - 1)
    r7 = min(7, M - 1)
    acc0 = 0.0
    acc1 = 0.0
    acc2 = 0.0
    acc3 = 0.0
    acc4 = 0.0
    acc5 = 0.0
    acc6 = 0.0
    acc7 = 0.0
    for jj in range(NQ):
        ib = ix + 4 * jj
        w = _q4k_decode_w(BLK, (col * NB + ib) * 144, iq, ir)
        y0 = _q4k_load_y(A, r0 * K + ib * 256 + o)
        acc0 = acc0 + _q4k_apply(w, y0[0], y0[1], y0[2])
        y1 = _q4k_load_y(A, r1 * K + ib * 256 + o)
        acc1 = acc1 + _q4k_apply(w, y1[0], y1[1], y1[2])
        y2 = _q4k_load_y(A, r2 * K + ib * 256 + o)
        acc2 = acc2 + _q4k_apply(w, y2[0], y2[1], y2[2])
        y3 = _q4k_load_y(A, r3 * K + ib * 256 + o)
        acc3 = acc3 + _q4k_apply(w, y3[0], y3[1], y3[2])
        y4 = _q4k_load_y(A, r4 * K + ib * 256 + o)
        acc4 = acc4 + _q4k_apply(w, y4[0], y4[1], y4[2])
        y5 = _q4k_load_y(A, r5 * K + ib * 256 + o)
        acc5 = acc5 + _q4k_apply(w, y5[0], y5[1], y5[2])
        y6 = _q4k_load_y(A, r6 * K + ib * 256 + o)
        acc6 = acc6 + _q4k_apply(w, y6[0], y6[1], y6[2])
        y7 = _q4k_load_y(A, r7 * K + ib * 256 + o)
        acc7 = acc7 + _q4k_apply(w, y7[0], y7[1], y7[2])
    if R > 0:
        ib = ix + 4 * NQ
        valid = ix < R
        ibc = al.where(valid, ib, 0)
        w = _q4k_decode_w(BLK, (col * NB + ibc) * 144, iq, ir)
        y0 = _q4k_load_y(A, r0 * K + ibc * 256 + o)
        acc0 = acc0 + al.where(valid, _q4k_apply(w, y0[0], y0[1], y0[2]), 0.0)
        y1 = _q4k_load_y(A, r1 * K + ibc * 256 + o)
        acc1 = acc1 + al.where(valid, _q4k_apply(w, y1[0], y1[1], y1[2]), 0.0)
        y2 = _q4k_load_y(A, r2 * K + ibc * 256 + o)
        acc2 = acc2 + al.where(valid, _q4k_apply(w, y2[0], y2[1], y2[2]), 0.0)
        y3 = _q4k_load_y(A, r3 * K + ibc * 256 + o)
        acc3 = acc3 + al.where(valid, _q4k_apply(w, y3[0], y3[1], y3[2]), 0.0)
        y4 = _q4k_load_y(A, r4 * K + ibc * 256 + o)
        acc4 = acc4 + al.where(valid, _q4k_apply(w, y4[0], y4[1], y4[2]), 0.0)
        y5 = _q4k_load_y(A, r5 * K + ibc * 256 + o)
        acc5 = acc5 + al.where(valid, _q4k_apply(w, y5[0], y5[1], y5[2]), 0.0)
        y6 = _q4k_load_y(A, r6 * K + ibc * 256 + o)
        acc6 = acc6 + al.where(valid, _q4k_apply(w, y6[0], y6[1], y6[2]), 0.0)
        y7 = _q4k_load_y(A, r7 * K + ibc * 256 + o)
        acc7 = acc7 + al.where(valid, _q4k_apply(w, y7[0], y7[1], y7[2]), 0.0)
    accs = [acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7]
    for m in _unroll(M):
        al.store(C + m * N + col, al.simd_reduce(accs[m]), mask=(col < N) & (tid < 1))


@al.tunable(
    BLOCK_M=[8, 16, 32, 64],
    BLOCK_N=[32, 64, 128],
    BLOCK_K=[32, 64, 128],
    _matvec=[0, 1],
)
@al.kernel
def dot_dequant(
    A,
    B_packed,
    scales,
    C: al.output,
    GROUP_SIZE: al.constexpr = 128,
    BITS: al.constexpr = 4,
    ZERO_POINT: al.constexpr = 8,
    BLOCK_M: al.constexpr = 16,
    BLOCK_N: al.constexpr = 64,
    BLOCK_K: al.constexpr = 64,
    _matvec: al.constexpr = 0,
):
    """Fused dequantize + matmul: C = A @ dequant(B_packed, scales, ZERO_POINT).T

    Uses standard al.load / al.tile_dot. The compiler detects the packed
    address pattern (rk // pack_factor) and generates cooperative loads
    with nibble extraction automatically.
    """
    M, K = A.shape
    N = B_packed.shape[0]
    PACK_FACTOR = 8 // BITS
    K_PACKED = K // PACK_FACTOR
    N_GROUPS = K // GROUP_SIZE

    if _matvec:
        # Matvec: one program per output column, 32 threads reduce K via simd.
        # Process exactly GROUP_SIZE elements per group — no masks on inner loads.
        # Each thread handles PACK_FACTOR elements (1 byte = 2 INT4 nibbles).
        # K-loop covers GROUP_SIZE in GROUP_SIZE/(32*PACK_FACTOR) iterations.
        col = al.program_id(0)
        _SIMD_WIDTH = 32
        _ELEMS_PER_THREAD = PACK_FACTOR  # 2 for INT4: 1 byte per thread
        _STEP = _SIMD_WIDTH * _ELEMS_PER_THREAD  # 64 elements per iteration
        _NIBBLE_MASK = (1 << BITS) - 1
        lane = al.arange(0, _SIMD_WIDTH)
        for m in range(M):
            acc = 0.0
            for g in range(N_GROUPS):
                g_start = g * GROUP_SIZE
                dot_acc = 0.0
                a_sum = 0.0
                for kb in range(0, GROUP_SIZE, _STEP):
                    elem_k = g_start + kb + lane * _ELEMS_PER_THREAD
                    # No mask: GROUP_SIZE is always a multiple of _STEP (128 % 64 = 0)
                    raw = al.load(B_packed + col * K_PACKED + elem_k // PACK_FACTOR)
                    local_dot = 0.0
                    local_a_sum = 0.0
                    for e in range(_ELEMS_PER_THREAD):
                        a_val = al.load(A + m * K + elem_k + e)
                        nibble = al.cast((raw >> (e * BITS)) & _NIBBLE_MASK, al.float32)
                        local_dot = local_dot + a_val * nibble
                        local_a_sum = local_a_sum + a_val
                    dot_acc = dot_acc + al.simd_reduce(local_dot)
                    a_sum = a_sum + al.simd_reduce(local_a_sum)
                scale = al.load(scales + col * N_GROUPS + g, mask=(col < N) & (lane < 1))
                acc = acc + scale * (dot_acc - ZERO_POINT * a_sum)
            al.store(C + m * N + col, acc, mask=(col < N) & (lane < 1))
    else:
        # Tiled MMA path
        pm = al.program_id(0)
        pn = al.program_id(1)
        rm = pm * BLOCK_M + al.arange(0, BLOCK_M)
        rn = pn * BLOCK_N + al.arange(0, BLOCK_N)
        rk = al.arange(0, BLOCK_K)

        a_ptrs = A + rm[:, None] * K + rk[None, :]
        acc = al.zeros((BLOCK_M, BLOCK_N), dtype=al.float32)

        for k in range(0, K, BLOCK_K):
            a = al.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] < K))
            b_packed_ptrs = B_packed + rn[:, None] * K_PACKED + rk[None, :] // PACK_FACTOR
            b_dequant = al.load(
                b_packed_ptrs,
                mask=(rn[:, None] < N) & (rk[None, :] < K),
                _dequant_scale=scales,
                _dequant_zero_point=ZERO_POINT,
                _dequant_n_groups=N_GROUPS,
            )
            acc += al.tile_dot(a, b_dequant, transpose_rhs=True)
            a_ptrs += BLOCK_K

        c_ptrs = C + rm[:, None] * N + rn[None, :]
        al.store(c_ptrs, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


@al.tunable(
    BLOCK_M=[8, 16, 32, 64],
    BLOCK_N=[32, 64, 128],
    BLOCK_K=[32, 64, 128],
    _matvec=[0, 1],
)
@al.kernel
def dot_dequant_silu(
    A,
    B_gate_packed,
    scales_gate,
    B_up_packed,
    scales_up,
    C: al.output,
    N_GATE: al.constexpr,
    GROUP_SIZE: al.constexpr = 128,
    BITS: al.constexpr = 4,
    ZERO_POINT: al.constexpr = 8,
    BLOCK_M: al.constexpr = 16,
    BLOCK_N: al.constexpr = 64,
    BLOCK_K: al.constexpr = 64,
    _matvec: al.constexpr = 0,
):
    """Paired-column dequant GEMM with fused SiLU: C = silu(A @ dequant(gate).T) * (A @ dequant(up).T).

    Two dequant GEMMs sharing the same A tile, SiLU fused in registers.
    """
    M, K = A.shape
    PACK_FACTOR = 8 // BITS
    K_PACKED = K // PACK_FACTOR
    N_GROUPS = K // GROUP_SIZE

    if _matvec:
        col = al.program_id(0)
        _SIMD_WIDTH = 32
        _ELEMS_PER_THREAD = PACK_FACTOR
        _STEP = _SIMD_WIDTH * _ELEMS_PER_THREAD
        _NIBBLE_MASK = (1 << BITS) - 1
        lane = al.arange(0, _SIMD_WIDTH)
        for m in range(M):
            acc_gate = 0.0
            acc_up = 0.0
            for g in range(N_GROUPS):
                g_start = g * GROUP_SIZE
                dot_acc_g = 0.0
                dot_acc_u = 0.0
                a_sum = 0.0
                for kb in range(0, GROUP_SIZE, _STEP):
                    elem_k = g_start + kb + lane * _ELEMS_PER_THREAD
                    raw_g = al.load(B_gate_packed + col * K_PACKED + elem_k // PACK_FACTOR)
                    raw_u = al.load(B_up_packed + col * K_PACKED + elem_k // PACK_FACTOR)
                    local_dot_g = 0.0
                    local_dot_u = 0.0
                    local_a_sum = 0.0
                    for e in range(_ELEMS_PER_THREAD):
                        a_val = al.load(A + m * K + elem_k + e)
                        nibble_g = al.cast((raw_g >> (e * BITS)) & _NIBBLE_MASK, al.float32)
                        nibble_u = al.cast((raw_u >> (e * BITS)) & _NIBBLE_MASK, al.float32)
                        local_dot_g = local_dot_g + a_val * nibble_g
                        local_dot_u = local_dot_u + a_val * nibble_u
                        local_a_sum = local_a_sum + a_val
                    dot_acc_g = dot_acc_g + al.simd_reduce(local_dot_g)
                    dot_acc_u = dot_acc_u + al.simd_reduce(local_dot_u)
                    a_sum = a_sum + al.simd_reduce(local_a_sum)
                scale_g = al.load(
                    scales_gate + col * N_GROUPS + g, mask=(col < N_GATE) & (lane < 1)
                )
                scale_u = al.load(scales_up + col * N_GROUPS + g, mask=(col < N_GATE) & (lane < 1))
                acc_gate = acc_gate + scale_g * (dot_acc_g - ZERO_POINT * a_sum)
                acc_up = acc_up + scale_u * (dot_acc_u - ZERO_POINT * a_sum)
            silu = acc_gate * (1.0 / (1.0 + al.exp(-acc_gate))) * acc_up
            al.store(C + m * N_GATE + col, silu, mask=(col < N_GATE) & (lane < 1))
    else:
        pm = al.program_id(0)
        pn = al.program_id(1)
        rm = pm * BLOCK_M + al.arange(0, BLOCK_M)
        rn = pn * BLOCK_N + al.arange(0, BLOCK_N)
        rk = al.arange(0, BLOCK_K)

        a_ptrs = A + rm[:, None] * K + rk[None, :]
        acc_gate = al.zeros((BLOCK_M, BLOCK_N), dtype=al.float32)
        acc_up = al.zeros((BLOCK_M, BLOCK_N), dtype=al.float32)

        for k in range(0, K, BLOCK_K):
            a = al.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] < K))

            bg_ptrs = B_gate_packed + rn[:, None] * K_PACKED + rk[None, :] // PACK_FACTOR
            bg = al.load(
                bg_ptrs,
                mask=(rn[:, None] < N_GATE) & (rk[None, :] < K),
                _dequant_scale=scales_gate,
                _dequant_zero_point=ZERO_POINT,
                _dequant_n_groups=N_GROUPS,
            )

            bu_ptrs = B_up_packed + rn[:, None] * K_PACKED + rk[None, :] // PACK_FACTOR
            bu = al.load(
                bu_ptrs,
                mask=(rn[:, None] < N_GATE) & (rk[None, :] < K),
                _dequant_scale=scales_up,
                _dequant_zero_point=ZERO_POINT,
                _dequant_n_groups=N_GROUPS,
            )

            acc_gate += al.tile_dot(a, bg, transpose_rhs=True)
            acc_up += al.tile_dot(a, bu, transpose_rhs=True)
            a_ptrs += BLOCK_K

        silu = acc_gate * (1.0 / (1.0 + al.exp(-acc_gate))) * acc_up
        c_ptrs = C + rm[:, None] * N_GATE + rn[None, :]
        al.store(c_ptrs, silu, mask=(rm[:, None] < M) & (rn[None, :] < N_GATE))


# Affine int4 group quant (MLX 4-bit): weight = scale*q + bias per group, q an
# unsigned nibble. qweight packs 2 nibbles/byte, weight k at byte k//2 (low nibble
# if even). Bias folds as dot(a, scale*q + bias) = scale*dot(a,q) + bias*sum(a).


@al.tunable(
    BLOCK_M=[8, 16, 32, 64, 128],
    BLOCK_N=[32, 64, 128],
    BLOCK_K=[32, 64, 128],
)
@al.kernel
def dot_mlx_q4(
    A,
    B_q4,
    scales,
    biases,
    C: al.output,
    GROUP_SIZE: al.constexpr = 128,
    BLOCK_M: al.constexpr = 16,
    BLOCK_N: al.constexpr = 64,
    BLOCK_K: al.constexpr = 64,
):
    """Affine int4 matmul (prefill): cooperative dequant load + tiled MMA."""
    M, K = A.shape
    N = B_q4.shape[0]
    PACK_FACTOR = 2
    K_PACKED = K // PACK_FACTOR
    N_GROUPS = K // GROUP_SIZE

    pm = al.program_id(0)
    pn = al.program_id(1)
    rm = pm * BLOCK_M + al.arange(0, BLOCK_M)
    rn = pn * BLOCK_N + al.arange(0, BLOCK_N)
    rk = al.arange(0, BLOCK_K)

    a_ptrs = A + rm[:, None] * K + rk[None, :]
    acc = al.zeros((BLOCK_M, BLOCK_N), dtype=al.float32)

    for k in range(0, K, BLOCK_K):
        elem_k = k + rk
        a = al.load(a_ptrs, mask=(rm[:, None] < M) & (elem_k[None, :] < K))
        b = al.load(
            B_q4 + rn[:, None] * K_PACKED + elem_k[None, :] // PACK_FACTOR,
            mask=(rn[:, None] < N) & (elem_k[None, :] < K),
            _dequant_scale=scales,
            _dequant_bias=biases,
            _dequant_zero_point=0,
            _dequant_n_groups=N_GROUPS,
        )
        acc += al.tile_dot(a, b, transpose_rhs=True)
        a_ptrs += BLOCK_K

    c_ptrs = C + rm[:, None] * N + rn[None, :]
    al.store(c_ptrs, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


@al.tunable(NUM_SPLITS=[1, 2, 4], NR0=[1, 2, 4])
@al.kernel
def dot_mlx_q4_v2(
    A,
    B_q4,
    scales,
    biases,
    C: al.output,
    GROUP_SIZE: al.constexpr = 128,
    NUM_SPLITS: al.constexpr = 1,
    NR0: al.constexpr = 1,
):
    """Affine int4 matvec (decode): split-K + multi-row amortization, vec4-batched.

    NR0 output cols per TG share one activation load per K chunk; NUM_SPLITS
    simdgroups cooperatively reduce one K range (partials combined via shmem).
    """
    M, K = A.shape
    N = B_q4.shape[0]
    K_PACKED = K // 2
    N_GROUPS = K // GROUP_SIZE
    K_SPLIT = K // NUM_SPLITS

    col_base = al.program_id(0) * NR0
    tid = al.arange(0, 32 * NUM_SPLITS)
    simd_id = tid // 32
    lane = tid % 32

    K_STEP = 256
    PER_LANE = 8

    def _final_store_one(col_local, val, col):
        if NUM_SPLITS == 1:
            al.store(C + col_local, val, mask=(col < N) & (lane < 1))
        else:
            shm = al.shared(NUM_SPLITS, dtype=al.float32)
            al.store(shm + simd_id, val, mask=(lane < 1))
            al.barrier()
            partials = [al.load(shm + s) for s in list(range(NUM_SPLITS))]
            total = partials[0]
            for s in list(range(1, NUM_SPLITS)):
                total = total + partials[s]
            al.store(
                C + col_local,
                total,
                mask=(col < N) & (simd_id < 1) & (lane < 1),
            )

    if NR0 == 1:
        for m in range(M):
            col0 = col_base + 0
            acc0 = 0.0
            for kb in range(0, K_SPLIT, K_STEP):
                k = simd_id * K_SPLIT + kb + lane * PER_LANE
                packed_idx = k // 2
                a4_first = al.load4_vec(A + m * K + k)
                a4_second = al.load4_vec(A + m * K + k + 4)

                raw0 = al.load4_vec(B_q4 + col0 * K_PACKED + packed_idx)
                lo0 = raw0 & 0x0F
                hi0 = (raw0 >> 4) & 0x0F
                q0_first = al.interleave_vec4(lo0, hi0, 0)
                q0_second = al.interleave_vec4(lo0, hi0, 1)
                s0 = al.cast(al.load(scales + col0 * N_GROUPS + k // GROUP_SIZE), al.float32)
                b0 = al.cast(al.load(biases + col0 * N_GROUPS + k // GROUP_SIZE), al.float32)
                w0_first = s0 * _q4k_u2f(al.cast(q0_first, al.int32)) + b0
                w0_second = s0 * _q4k_u2f(al.cast(q0_second, al.int32)) + b0
                acc0 = acc0 + al.dot4(a4_first, w0_first) + al.dot4(a4_second, w0_second)
            acc0 = al.simd_reduce(acc0)
            _final_store_one(m * N + col0, acc0, col0)
    elif NR0 == 2:
        for m in range(M):
            col0 = col_base + 0
            col1 = col_base + 1
            acc0 = 0.0
            acc1 = 0.0
            for kb in range(0, K_SPLIT, K_STEP):
                k = simd_id * K_SPLIT + kb + lane * PER_LANE
                packed_idx = k // 2
                a4_first = al.load4_vec(A + m * K + k)
                a4_second = al.load4_vec(A + m * K + k + 4)

                raw0 = al.load4_vec(B_q4 + col0 * K_PACKED + packed_idx)
                lo0 = raw0 & 0x0F
                hi0 = (raw0 >> 4) & 0x0F
                q0_first = al.interleave_vec4(lo0, hi0, 0)
                q0_second = al.interleave_vec4(lo0, hi0, 1)
                s0 = al.cast(al.load(scales + col0 * N_GROUPS + k // GROUP_SIZE), al.float32)
                b0 = al.cast(al.load(biases + col0 * N_GROUPS + k // GROUP_SIZE), al.float32)
                w0_first = s0 * _q4k_u2f(al.cast(q0_first, al.int32)) + b0
                w0_second = s0 * _q4k_u2f(al.cast(q0_second, al.int32)) + b0
                acc0 = acc0 + al.dot4(a4_first, w0_first) + al.dot4(a4_second, w0_second)

                raw1 = al.load4_vec(B_q4 + col1 * K_PACKED + packed_idx)
                lo1 = raw1 & 0x0F
                hi1 = (raw1 >> 4) & 0x0F
                q1_first = al.interleave_vec4(lo1, hi1, 0)
                q1_second = al.interleave_vec4(lo1, hi1, 1)
                s1 = al.cast(al.load(scales + col1 * N_GROUPS + k // GROUP_SIZE), al.float32)
                b1 = al.cast(al.load(biases + col1 * N_GROUPS + k // GROUP_SIZE), al.float32)
                w1_first = s1 * _q4k_u2f(al.cast(q1_first, al.int32)) + b1
                w1_second = s1 * _q4k_u2f(al.cast(q1_second, al.int32)) + b1
                acc1 = acc1 + al.dot4(a4_first, w1_first) + al.dot4(a4_second, w1_second)
            acc0 = al.simd_reduce(acc0)
            acc1 = al.simd_reduce(acc1)
            _final_store_one(m * N + col0, acc0, col0)
            _final_store_one(m * N + col1, acc1, col1)
    else:  # NR0 == 4
        for m in range(M):
            col0 = col_base + 0
            col1 = col_base + 1
            col2 = col_base + 2
            col3 = col_base + 3
            acc0 = 0.0
            acc1 = 0.0
            acc2 = 0.0
            acc3 = 0.0
            for kb in range(0, K_SPLIT, K_STEP):
                k = simd_id * K_SPLIT + kb + lane * PER_LANE
                packed_idx = k // 2
                a4_first = al.load4_vec(A + m * K + k)
                a4_second = al.load4_vec(A + m * K + k + 4)

                raw0 = al.load4_vec(B_q4 + col0 * K_PACKED + packed_idx)
                lo0 = raw0 & 0x0F
                hi0 = (raw0 >> 4) & 0x0F
                q0_first = al.interleave_vec4(lo0, hi0, 0)
                q0_second = al.interleave_vec4(lo0, hi0, 1)
                s0 = al.cast(al.load(scales + col0 * N_GROUPS + k // GROUP_SIZE), al.float32)
                b0 = al.cast(al.load(biases + col0 * N_GROUPS + k // GROUP_SIZE), al.float32)
                w0_first = s0 * _q4k_u2f(al.cast(q0_first, al.int32)) + b0
                w0_second = s0 * _q4k_u2f(al.cast(q0_second, al.int32)) + b0
                acc0 = acc0 + al.dot4(a4_first, w0_first) + al.dot4(a4_second, w0_second)

                raw1 = al.load4_vec(B_q4 + col1 * K_PACKED + packed_idx)
                lo1 = raw1 & 0x0F
                hi1 = (raw1 >> 4) & 0x0F
                q1_first = al.interleave_vec4(lo1, hi1, 0)
                q1_second = al.interleave_vec4(lo1, hi1, 1)
                s1 = al.cast(al.load(scales + col1 * N_GROUPS + k // GROUP_SIZE), al.float32)
                b1 = al.cast(al.load(biases + col1 * N_GROUPS + k // GROUP_SIZE), al.float32)
                w1_first = s1 * _q4k_u2f(al.cast(q1_first, al.int32)) + b1
                w1_second = s1 * _q4k_u2f(al.cast(q1_second, al.int32)) + b1
                acc1 = acc1 + al.dot4(a4_first, w1_first) + al.dot4(a4_second, w1_second)

                raw2 = al.load4_vec(B_q4 + col2 * K_PACKED + packed_idx)
                lo2 = raw2 & 0x0F
                hi2 = (raw2 >> 4) & 0x0F
                q2_first = al.interleave_vec4(lo2, hi2, 0)
                q2_second = al.interleave_vec4(lo2, hi2, 1)
                s2 = al.cast(al.load(scales + col2 * N_GROUPS + k // GROUP_SIZE), al.float32)
                b2 = al.cast(al.load(biases + col2 * N_GROUPS + k // GROUP_SIZE), al.float32)
                w2_first = s2 * _q4k_u2f(al.cast(q2_first, al.int32)) + b2
                w2_second = s2 * _q4k_u2f(al.cast(q2_second, al.int32)) + b2
                acc2 = acc2 + al.dot4(a4_first, w2_first) + al.dot4(a4_second, w2_second)

                raw3 = al.load4_vec(B_q4 + col3 * K_PACKED + packed_idx)
                lo3 = raw3 & 0x0F
                hi3 = (raw3 >> 4) & 0x0F
                q3_first = al.interleave_vec4(lo3, hi3, 0)
                q3_second = al.interleave_vec4(lo3, hi3, 1)
                s3 = al.cast(al.load(scales + col3 * N_GROUPS + k // GROUP_SIZE), al.float32)
                b3 = al.cast(al.load(biases + col3 * N_GROUPS + k // GROUP_SIZE), al.float32)
                w3_first = s3 * _q4k_u2f(al.cast(q3_first, al.int32)) + b3
                w3_second = s3 * _q4k_u2f(al.cast(q3_second, al.int32)) + b3
                acc3 = acc3 + al.dot4(a4_first, w3_first) + al.dot4(a4_second, w3_second)
            acc0 = al.simd_reduce(acc0)
            acc1 = al.simd_reduce(acc1)
            acc2 = al.simd_reduce(acc2)
            acc3 = al.simd_reduce(acc3)
            _final_store_one(m * N + col0, acc0, col0)
            _final_store_one(m * N + col1, acc1, col1)
            _final_store_one(m * N + col2, acc2, col2)
            _final_store_one(m * N + col3, acc3, col3)


@al.tunable()
@al.kernel
def dot_mlx_q4_v2_rows(A, B_q4, scales, biases, C: al.output, GROUP_SIZE: al.constexpr = 128):
    """Affine int4 GEMV for small M (2..8): dequant ONCE, read weights ONCE,
    accumulate all rows. Rows >= M clamp to M-1 (cheap extra dot4s), store only M."""
    M, K = A.shape
    N = B_q4.shape[0]
    K_PACKED = K // 2
    N_GROUPS = K // GROUP_SIZE
    col = al.program_id(0)
    lane = al.arange(0, 32)
    K_STEP = 256
    PER_LANE = 8
    r0 = min(0, M - 1)
    r1 = min(1, M - 1)
    r2 = min(2, M - 1)
    r3 = min(3, M - 1)
    r4 = min(4, M - 1)
    r5 = min(5, M - 1)
    r6 = min(6, M - 1)
    r7 = min(7, M - 1)
    acc0 = 0.0
    acc1 = 0.0
    acc2 = 0.0
    acc3 = 0.0
    acc4 = 0.0
    acc5 = 0.0
    acc6 = 0.0
    acc7 = 0.0
    for kb in range(0, K, K_STEP):
        k = kb + lane * PER_LANE
        packed_idx = k // 2
        raw = al.load4_vec(B_q4 + col * K_PACKED + packed_idx)
        lo = raw & 0x0F
        hi = (raw >> 4) & 0x0F
        q_first = al.interleave_vec4(lo, hi, 0)
        q_second = al.interleave_vec4(lo, hi, 1)
        s = al.cast(al.load(scales + col * N_GROUPS + k // GROUP_SIZE), al.float32)
        b = al.cast(al.load(biases + col * N_GROUPS + k // GROUP_SIZE), al.float32)
        w_first = s * _q4k_u2f(al.cast(q_first, al.int32)) + b
        w_second = s * _q4k_u2f(al.cast(q_second, al.int32)) + b
        acc0 = (
            acc0
            + al.dot4(al.load4_vec(A + r0 * K + k), w_first)
            + al.dot4(al.load4_vec(A + r0 * K + k + 4), w_second)
        )
        acc1 = (
            acc1
            + al.dot4(al.load4_vec(A + r1 * K + k), w_first)
            + al.dot4(al.load4_vec(A + r1 * K + k + 4), w_second)
        )
        acc2 = (
            acc2
            + al.dot4(al.load4_vec(A + r2 * K + k), w_first)
            + al.dot4(al.load4_vec(A + r2 * K + k + 4), w_second)
        )
        acc3 = (
            acc3
            + al.dot4(al.load4_vec(A + r3 * K + k), w_first)
            + al.dot4(al.load4_vec(A + r3 * K + k + 4), w_second)
        )
        acc4 = (
            acc4
            + al.dot4(al.load4_vec(A + r4 * K + k), w_first)
            + al.dot4(al.load4_vec(A + r4 * K + k + 4), w_second)
        )
        acc5 = (
            acc5
            + al.dot4(al.load4_vec(A + r5 * K + k), w_first)
            + al.dot4(al.load4_vec(A + r5 * K + k + 4), w_second)
        )
        acc6 = (
            acc6
            + al.dot4(al.load4_vec(A + r6 * K + k), w_first)
            + al.dot4(al.load4_vec(A + r6 * K + k + 4), w_second)
        )
        acc7 = (
            acc7
            + al.dot4(al.load4_vec(A + r7 * K + k), w_first)
            + al.dot4(al.load4_vec(A + r7 * K + k + 4), w_second)
        )
    accs = [acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7]
    for m in _unroll(M):
        al.store(C + m * N + col, al.simd_reduce(accs[m]), mask=(col < N) & (lane < 1))


@al.kernel
def embedding_mlx_q4(
    input_ids,
    qweight,
    scales,
    biases,
    out: al.output,
    NUM_INDICES: al.constexpr,
    WIDTH: al.constexpr,
    GROUP_SIZE: al.constexpr = 128,
    BLOCK_SIZE: al.constexpr = 1024,
):
    """Embedding lookup over affine int4 row blocks."""
    pid = al.program_id(0)
    offs = pid * BLOCK_SIZE + al.arange(0, BLOCK_SIZE)
    mask = offs < NUM_INDICES * WIDTH
    row = offs // WIDTH
    col = offs % WIDTH
    groups = WIDTH // GROUP_SIZE
    packed_width = WIDTH // 2
    idx = al.load(input_ids + row, mask=mask)
    raw = al.cast(al.load(qweight + idx * packed_width + col // 2, mask=mask), al.int32)
    q = al.cast((raw >> ((col % 2) * 4)) & 0x0F, al.float32)
    group = col // GROUP_SIZE
    scale = al.cast(al.load(scales + idx * groups + group, mask=mask), al.float32)
    bias = al.cast(al.load(biases + idx * groups + group, mask=mask), al.float32)
    al.store(out + offs, scale * q + bias, mask=mask)


@al.kernel
def dot_mlx_q4_silu_v2(
    A,
    B_gate_q4,
    gate_scales,
    gate_biases,
    B_up_q4,
    up_scales,
    up_biases,
    C: al.output,
    GROUP_SIZE: al.constexpr = 128,
):
    """Affine int4 gate+up matvec (decode) with SiLU fusion: one activation load
    per K chunk dotted against both the gate and up rows, silu(gate)*up in registers."""
    M, K = A.shape
    N = B_gate_q4.shape[0]
    K_PACKED = K // 2
    N_GROUPS = K // GROUP_SIZE
    col0 = al.program_id(0)
    lane = al.arange(0, 32)
    K_STEP = 256
    PER_LANE = 8
    for m in range(M):
        acc_g0 = 0.0
        acc_u0 = 0.0
        for kb in range(0, K, K_STEP):
            k = kb + lane * PER_LANE
            packed_idx = k // 2
            a4_first = al.load4_vec(A + m * K + k)
            a4_second = al.load4_vec(A + m * K + k + 4)

            g0_raw = al.load4_vec(B_gate_q4 + col0 * K_PACKED + packed_idx)
            u0_raw = al.load4_vec(B_up_q4 + col0 * K_PACKED + packed_idx)
            g0_lo = g0_raw & 0x0F
            g0_hi = (g0_raw >> 4) & 0x0F
            u0_lo = u0_raw & 0x0F
            u0_hi = (u0_raw >> 4) & 0x0F
            qg0_first = al.interleave_vec4(g0_lo, g0_hi, 0)
            qg0_second = al.interleave_vec4(g0_lo, g0_hi, 1)
            qu0_first = al.interleave_vec4(u0_lo, u0_hi, 0)
            qu0_second = al.interleave_vec4(u0_lo, u0_hi, 1)
            gs0 = al.cast(al.load(gate_scales + col0 * N_GROUPS + k // GROUP_SIZE), al.float32)
            gb0 = al.cast(al.load(gate_biases + col0 * N_GROUPS + k // GROUP_SIZE), al.float32)
            us0 = al.cast(al.load(up_scales + col0 * N_GROUPS + k // GROUP_SIZE), al.float32)
            ub0 = al.cast(al.load(up_biases + col0 * N_GROUPS + k // GROUP_SIZE), al.float32)
            gw0_first = gs0 * _q4k_u2f(al.cast(qg0_first, al.int32)) + gb0
            gw0_second = gs0 * _q4k_u2f(al.cast(qg0_second, al.int32)) + gb0
            uw0_first = us0 * _q4k_u2f(al.cast(qu0_first, al.int32)) + ub0
            uw0_second = us0 * _q4k_u2f(al.cast(qu0_second, al.int32)) + ub0
            acc_g0 = acc_g0 + al.dot4(a4_first, gw0_first) + al.dot4(a4_second, gw0_second)
            acc_u0 = acc_u0 + al.dot4(a4_first, uw0_first) + al.dot4(a4_second, uw0_second)
        g0 = al.simd_reduce(acc_g0)
        u0 = al.simd_reduce(acc_u0)
        silu0 = g0 * (1.0 / (1.0 + al.exp(-g0))) * u0
        al.store(C + m * N + col0, silu0, mask=(col0 < N) & (lane < 1))
