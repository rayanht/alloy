"""Dense GEMM kernels."""

import alloy as al


@al.tunable(
    BLOCK_M=[8, 16, 32, 64],
    BLOCK_N=[32, 64, 128],
    BLOCK_K=[16, 32, 64],
    _reg=[1, 2, 4],
    options=dict(double_buffer=[0, 1]),
)
@al.kernel
def dot(
    A,
    B,
    C: al.output,
    BLOCK_M: al.constexpr = 64,
    BLOCK_N: al.constexpr = 64,
    BLOCK_K: al.constexpr = 16,
):
    M, K = A.shape
    N = B.shape[1]
    pm = al.program_id(0)
    pn = al.program_id(1)
    rm = pm * BLOCK_M + al.arange(0, BLOCK_M)
    rn = pn * BLOCK_N + al.arange(0, BLOCK_N)
    rk = al.arange(0, BLOCK_K)
    a_ptrs = A + rm[:, None] * K + rk[None, :]
    b_ptrs = B + rk[:, None] * N + rn[None, :]
    acc = al.zeros((BLOCK_M, BLOCK_N), dtype=al.float32)
    for k in range(0, K, BLOCK_K):
        a = al.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] < K))
        b = al.load(b_ptrs, mask=(rk[:, None] < K) & (rn[None, :] < N))
        acc += al.tile_dot(a, b)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N
    c_ptrs = C + rm[:, None] * N + rn[None, :]
    al.store(c_ptrs, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


@al.tunable(
    BLOCK_M=[8, 16, 32, 64],
    BLOCK_N=[8, 32, 64, 128],
    BLOCK_K=[16, 32, 64],
    _reg=[1, 2, 4],
    _matvec=[0, 1],
    _PACKED=[0],
    options=dict(double_buffer=[0, 1]),
)
@al.kernel
def dot_transpose_rhs(
    A,
    B_T,
    C: al.output,
    BLOCK_M: al.constexpr = 64,
    BLOCK_N: al.constexpr = 64,
    BLOCK_K: al.constexpr = 16,
    _matvec: al.constexpr = 0,
    _PACKED: al.constexpr = 0,
):
    M, K = A.shape
    N = B_T.shape[0]
    if _matvec:
        _SIMD_WIDTH = 32
        col = al.program_id(0)
        lane = al.arange(0, _SIMD_WIDTH)
        for m in range(M):
            acc = 0.0
            _STEP = _SIMD_WIDTH * 4  # 128 f16 per vectorized iteration
            _K_VEC = (K // _STEP) * _STEP  # largest multiple of 128 ≤ K
            # Vectorized main loop
            for kb in range(0, _K_VEC, _STEP):
                k = kb + lane * 4
                a_v4 = al.load4_vec(A + m * K + k)
                b_v4 = al.load4_vec(B_T + col * K + k)
                acc = acc + al.dot4(a_v4, b_v4)
            # Scalar tail for remaining elements
            for kb in range(_K_VEC, K, _SIMD_WIDTH):
                k = kb + lane
                a_val = al.load(A + m * K + k, mask=k < K)
                b_val = al.load(B_T + col * K + k, mask=k < K)
                acc = acc + a_val * b_val
            acc = al.simd_reduce(acc)
            al.store(C + m * N + col, acc, mask=(col < N) & (lane < 1))
    elif _PACKED:
        # Tiled MMA with packed weight layout: each (BN, BK) tile is contiguous
        K_TILES = K // BLOCK_K
        pm = al.program_id(0)
        pn = al.program_id(1)
        rm = pm * BLOCK_M + al.arange(0, BLOCK_M)
        rk = al.arange(0, BLOCK_K)
        rn_local = al.arange(0, BLOCK_N)
        a_ptrs = A + rm[:, None] * K + rk[None, :]
        # Tile (pn, tk=0) starts at pn * BN * K_TILES * BK; row stride is BK.
        # pn * BLOCK_N must be the first pid multiply so the dispatch spec
        # records stride=BLOCK_N (not K_TILES) for correct grid computation.
        b_ptrs = (
            B_T + (pn * BLOCK_N) * (K_TILES * BLOCK_K) + rn_local[:, None] * BLOCK_K + rk[None, :]
        )
        acc = al.zeros((BLOCK_M, BLOCK_N), dtype=al.float32)
        for k in range(0, K, BLOCK_K):
            a = al.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] < K))
            b = al.load(b_ptrs, mask=True)
            acc += al.tile_dot(a, b, transpose_rhs=True)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_N * BLOCK_K  # jump to next contiguous tile
        rn = pn * BLOCK_N + al.arange(0, BLOCK_N)
        c_ptrs = C + rm[:, None] * N + rn[None, :]
        al.store(c_ptrs, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))
    else:
        # Tiled MMA path
        pm = al.program_id(0)
        pn = al.program_id(1)
        rm = pm * BLOCK_M + al.arange(0, BLOCK_M)
        rn = pn * BLOCK_N + al.arange(0, BLOCK_N)
        rk = al.arange(0, BLOCK_K)
        a_ptrs = A + rm[:, None] * K + rk[None, :]
        b_ptrs = B_T + rn[:, None] * K + rk[None, :]
        acc = al.zeros((BLOCK_M, BLOCK_N), dtype=al.float32)
        for k in range(0, K, BLOCK_K):
            a = al.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] < K))
            b = al.load(b_ptrs, mask=(rn[:, None] < N) & (rk[None, :] < K))
            acc += al.tile_dot(a, b, transpose_rhs=True)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K
        c_ptrs = C + rm[:, None] * N + rn[None, :]
        al.store(c_ptrs, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


@al.tunable(
    BLOCK_M=[8, 16, 32, 64],
    BLOCK_N=[8, 32, 64, 128],
    BLOCK_K=[16, 32, 64],
    _reg=[1, 2, 4],
    options=dict(double_buffer=[0, 1]),
)
@al.kernel
def dot_transpose_lhs(
    A_T,
    B,
    C: al.output,
    BLOCK_M: al.constexpr = 64,
    BLOCK_N: al.constexpr = 64,
    BLOCK_K: al.constexpr = 16,
):
    """Matmul A.T @ B where A is physically stored as (K, M) contiguous.

    Mirrors dot_transpose_rhs but with the transpose on the left operand:
    input A_T has shape (K, M) and we compute A_T.T @ B → (M, N). Avoids
    materializing an (M, K) contiguous copy of a transposed view (e.g.
    d_W = x.T @ d_y in linear-layer backward).

    No matvec/PACKED paths: matvec in dot_transpose_rhs relies on a
    contiguous 4-wide load along K, which doesn't apply here (A_T accesses
    at fixed m stride by M), and PACKED is specific to quantized weight
    layouts irrelevant for LHS gradients. The tiled MMA path carries all
    tune knobs matching dot_transpose_rhs.
    """
    K, M = A_T.shape
    N = B.shape[1]
    pm = al.program_id(0)
    pn = al.program_id(1)
    rm = pm * BLOCK_M + al.arange(0, BLOCK_M)
    rn = pn * BLOCK_N + al.arange(0, BLOCK_N)
    rk = al.arange(0, BLOCK_K)
    a_t_ptrs = A_T + rk[:, None] * M + rm[None, :]
    b_ptrs = B + rk[:, None] * N + rn[None, :]
    acc = al.zeros((BLOCK_M, BLOCK_N), dtype=al.float32)
    for k in range(0, K, BLOCK_K):
        a_t = al.load(a_t_ptrs, mask=(rk[:, None] < K) & (rm[None, :] < M))
        b = al.load(b_ptrs, mask=(rk[:, None] < K) & (rn[None, :] < N))
        acc += al.tile_dot(a_t, b, transpose_lhs=True)
        a_t_ptrs += BLOCK_K * M
        b_ptrs += BLOCK_K * N
    c_ptrs = C + rm[:, None] * N + rn[None, :]
    al.store(c_ptrs, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


@al.tunable(
    BLOCK_M=[8, 16, 32, 64],
    BLOCK_N=[8, 32, 64, 128],
    BLOCK_K=[16, 32, 64],
    _matvec=[0, 1],
)
@al.kernel
def dot_transpose_rhs_silu(
    A,
    B_gate_T,
    B_up_T,
    C: al.output,
    N_GATE: al.constexpr,
    BLOCK_M: al.constexpr = 64,
    BLOCK_N: al.constexpr = 64,
    BLOCK_K: al.constexpr = 16,
    _matvec: al.constexpr = 0,
):
    """Paired-column GEMM with fused SiLU: C = silu(A @ B_gate.T) * (A @ B_up.T).

    Two GEMMs sharing the same A tile, SiLU fused in registers.
    Grid: (ceil(M/BM), ceil(N_GATE/BN)) for tiled, (N_GATE,) for matvec.
    """
    M, K = A.shape
    if _matvec:
        col = al.program_id(0)
        _SIMD_WIDTH = 32
        lane = al.arange(0, _SIMD_WIDTH)
        for m in range(M):
            acc_gate = 0.0
            acc_up = 0.0
            for kb in range(0, K, _SIMD_WIDTH):
                k = kb + lane
                a_val = al.load(A + m * K + k, mask=k < K)
                bg_val = al.load(B_gate_T + col * K + k, mask=k < K)
                bu_val = al.load(B_up_T + col * K + k, mask=k < K)
                acc_gate = acc_gate + al.simd_reduce(a_val * bg_val)
                acc_up = acc_up + al.simd_reduce(a_val * bu_val)
            silu = acc_gate * (1.0 / (1.0 + al.exp(-acc_gate))) * acc_up
            al.store(C + m * N_GATE + col, silu, mask=(col < N_GATE) & (lane < 1))
    else:
        pm = al.program_id(0)
        pn = al.program_id(1)
        rm = pm * BLOCK_M + al.arange(0, BLOCK_M)
        rn = pn * BLOCK_N + al.arange(0, BLOCK_N)
        rk = al.arange(0, BLOCK_K)
        a_ptrs = A + rm[:, None] * K + rk[None, :]
        bg_ptrs = B_gate_T + rn[:, None] * K + rk[None, :]
        bu_ptrs = B_up_T + rn[:, None] * K + rk[None, :]
        acc_gate = al.zeros((BLOCK_M, BLOCK_N), dtype=al.float32)
        acc_up = al.zeros((BLOCK_M, BLOCK_N), dtype=al.float32)
        for k in range(0, K, BLOCK_K):
            a = al.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] < K))
            bg = al.load(bg_ptrs, mask=(rn[:, None] < N_GATE) & (rk[None, :] < K))
            bu = al.load(bu_ptrs, mask=(rn[:, None] < N_GATE) & (rk[None, :] < K))
            acc_gate += al.tile_dot(a, bg, transpose_rhs=True)
            acc_up += al.tile_dot(a, bu, transpose_rhs=True)
            a_ptrs += BLOCK_K
            bg_ptrs += BLOCK_K
            bu_ptrs += BLOCK_K
        silu = acc_gate * (1.0 / (1.0 + al.exp(-acc_gate))) * acc_up
        c_ptrs = C + rm[:, None] * N_GATE + rn[None, :]
        al.store(c_ptrs, silu, mask=(rm[:, None] < M) & (rn[None, :] < N_GATE))
