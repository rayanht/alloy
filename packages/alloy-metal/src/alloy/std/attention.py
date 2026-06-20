"""Attention kernels."""

import alloy as al


@al.tunable(BLOCK_M=[8, 16, 32, 64], BLOCK_N=[8, 16, 32], options=dict(fuse_loops=[0, 1]))
@al.kernel
def attention(
    Q,
    K,
    V,
    O: al.output,  # noqa: E741
    BH: al.constexpr = 1,
    BLOCK_M: al.constexpr = 16,
    BLOCK_N: al.constexpr = 32,
    causal: al.constexpr = 0,
    KV_GROUP: al.constexpr = 1,
    CUSTOM_SCALE: al.constexpr = 0,
):
    D = Q.shape[1]
    N = Q.shape[0] // BH
    N_KV_BLOCKS = (N + BLOCK_N - 1) // BLOCK_N
    SCALE = CUSTOM_SCALE if (CUSTOM_SCALE is not None and CUSTOM_SCALE > 0) else 1.0 / (D**0.5)
    q_block = al.program_id(0)
    bh = al.program_id(1)
    head_off = bh * N * D
    # GQA: K/V have fewer heads. KV_GROUP = Q_heads / KV_heads (1 for non-GQA).
    if KV_GROUP is not None and KV_GROUP > 1:
        kv_bh = bh // KV_GROUP
        kv_head_off = kv_bh * N * D
    else:
        kv_head_off = head_off
    Qh = Q + head_off
    Kh = K + kv_head_off
    Vh = V + kv_head_off
    Oh = O + head_off
    q_start = q_block * BLOCK_M
    rm = al.arange(0, BLOCK_M)
    rd = al.arange(0, D)
    rn = al.arange(0, BLOCK_N)
    q = al.load(
        Qh + (q_start + rm)[:, None] * D + rd[None, :],
        mask=(q_start + rm[:, None]) < N,
        other=0.0,
    )
    m = -1e30
    l = 0.0  # noqa: E741
    o = al.zeros((BLOCK_M, D), dtype=al.float32)
    for _jb in range(0, N_KV_BLOCKS, 1):
        j = _jb * BLOCK_N
        k_tile = al.load(
            Kh + (j + rn)[:, None] * D + rd[None, :],
            mask=(j + rn[:, None]) < N,
            other=0.0,
        )
        s = al.tile_dot(q, k_tile, transpose_rhs=True)
        s = s * SCALE
        s = al.where((j + rn)[None, :] < N, s, -1e30)
        if causal:
            s = al.where((q_start + rm)[:, None] >= (j + rn)[None, :], s, -1e30)
        bmax = al.max(s, axis=1)
        mn = al.maximum(m, bmax)
        alpha = al.exp(m - mn)
        l = l * alpha  # noqa: E741
        o = o * alpha
        p = al.exp(s - mn)
        l = l + al.sum(p, axis=1)  # noqa: E741
        al.barrier()
        v_tile = al.load(
            Vh + (j + rn)[:, None] * D + rd[None, :],
            mask=(j + rn[:, None]) < N,
            other=0.0,
        )
        o = o + al.tile_dot(p, v_tile)
        m = mn
        al.barrier()
    o = o * (1.0 / l)
    al.store(
        Oh + (q_start + rm)[:, None] * D + rd[None, :],
        o,
        mask=((q_start + rm[:, None]) < N) & (bh < BH),
    )


@al.tunable(BLOCK_M=[8, 16, 32, 64], BLOCK_N=[8, 16, 32], options=dict(fuse_loops=[0, 1]))
@al.kernel
def attention_masked_by_batch(
    Q,
    K,
    V,
    Mask,
    O: al.output,  # noqa: E741
    BH: al.constexpr = 1,
    HEADS_PER_BATCH: al.constexpr = 1,
    BLOCK_M: al.constexpr = 16,
    BLOCK_N: al.constexpr = 32,
    causal: al.constexpr = 0,
    KV_GROUP: al.constexpr = 1,
    CUSTOM_SCALE: al.constexpr = 0,
):
    D = Q.shape[1]
    N = Q.shape[0] // BH
    N_KV_BLOCKS = (N + BLOCK_N - 1) // BLOCK_N
    SCALE = CUSTOM_SCALE if (CUSTOM_SCALE is not None and CUSTOM_SCALE > 0) else 1.0 / (D**0.5)
    q_block = al.program_id(0)
    bh = al.program_id(1)
    batch = bh // HEADS_PER_BATCH
    head_off = bh * N * D
    if KV_GROUP is not None and KV_GROUP > 1:
        kv_bh = bh // KV_GROUP
        kv_head_off = kv_bh * N * D
    else:
        kv_head_off = head_off
    mask_off = batch * N * N
    Qh = Q + head_off
    Kh = K + kv_head_off
    Vh = V + kv_head_off
    Mh = Mask + mask_off
    Oh = O + head_off
    q_start = q_block * BLOCK_M
    rm = al.arange(0, BLOCK_M)
    rd = al.arange(0, D)
    rn = al.arange(0, BLOCK_N)
    q = al.load(
        Qh + (q_start + rm)[:, None] * D + rd[None, :],
        mask=(q_start + rm[:, None]) < N,
        other=0.0,
    )
    m = -1e30
    l = 0.0  # noqa: E741
    o = al.zeros((BLOCK_M, D), dtype=al.float32)
    for _jb in range(0, N_KV_BLOCKS, 1):
        j = _jb * BLOCK_N
        k_tile = al.load(
            Kh + (j + rn)[:, None] * D + rd[None, :],
            mask=(j + rn[:, None]) < N,
            other=0.0,
        )
        s = al.tile_dot(q, k_tile, transpose_rhs=True)
        s = s * SCALE
        al.barrier()
        mask_tile = al.load(
            Mh + (q_start + rm)[:, None] * N + (j + rn)[None, :],
            mask=((q_start + rm)[:, None] < N) & ((j + rn)[None, :] < N),
            other=-1e30,
        )
        s = s + mask_tile
        if causal:
            s = al.where((q_start + rm)[:, None] >= (j + rn)[None, :], s, -1e30)
        bmax = al.max(s, axis=1)
        mn = al.maximum(m, bmax)
        alpha = al.exp(m - mn)
        l = l * alpha  # noqa: E741
        o = o * alpha
        p = al.exp(s - mn)
        l = l + al.sum(p, axis=1)  # noqa: E741
        al.barrier()
        v_tile = al.load(
            Vh + (j + rn)[:, None] * D + rd[None, :],
            mask=(j + rn[:, None]) < N,
            other=0.0,
        )
        o = o + al.tile_dot(p, v_tile)
        m = mn
        al.barrier()
    o = o * (1.0 / l)
    al.store(
        Oh + (q_start + rm)[:, None] * D + rd[None, :],
        o,
        mask=((q_start + rm[:, None]) < N) & (bh < BH),
    )


@al.tunable(BLOCK_M=[8, 16, 32, 64], BLOCK_N=[8, 16, 32], options=dict(fuse_loops=[0, 1]))
@al.kernel
def attention_strided(
    Q,
    K,
    V,
    O: al.output,  # noqa: E741
    BH: al.constexpr = 1,
    HEADS_PER_BATCH: al.constexpr = 1,
    SEQ_LEN: al.constexpr = 1,
    HEAD_DIM: al.constexpr = 1,
    Q_OFFSET: al.constexpr = 0,
    Q_BATCH_STRIDE: al.constexpr = 0,
    Q_HEAD_STRIDE: al.constexpr = 0,
    Q_SEQ_STRIDE: al.constexpr = 0,
    K_OFFSET: al.constexpr = 0,
    K_BATCH_STRIDE: al.constexpr = 0,
    K_HEAD_STRIDE: al.constexpr = 0,
    K_SEQ_STRIDE: al.constexpr = 0,
    V_OFFSET: al.constexpr = 0,
    V_BATCH_STRIDE: al.constexpr = 0,
    V_HEAD_STRIDE: al.constexpr = 0,
    V_SEQ_STRIDE: al.constexpr = 0,
    BLOCK_M: al.constexpr = 16,
    BLOCK_N: al.constexpr = 32,
    causal: al.constexpr = 0,
    KV_GROUP: al.constexpr = 1,
    CUSTOM_SCALE: al.constexpr = 0,
    KV_LEN: al.constexpr = 0,
    Q_START_POS: al.constexpr = 0,
    SLIDING_WINDOW: al.constexpr = 0,
    K_WRAP: al.constexpr = 0,
):
    D = HEAD_DIM
    N = SEQ_LEN
    N_KV = KV_LEN if KV_LEN > 0 else N  # KV cache: K/V may be longer than Q
    N_KV_BLOCKS = (N_KV + BLOCK_N - 1) // BLOCK_N
    SCALE = CUSTOM_SCALE if (CUSTOM_SCALE is not None and CUSTOM_SCALE > 0) else 1.0 / (D**0.5)
    q_block = al.program_id(0)
    bh = al.program_id(1)
    batch = bh // HEADS_PER_BATCH
    head = bh - batch * HEADS_PER_BATCH
    q_head_off = Q_OFFSET + batch * Q_BATCH_STRIDE + head * Q_HEAD_STRIDE
    # GQA: K/V have fewer heads — map Q head to KV head
    kv_head = head // KV_GROUP if KV_GROUP > 1 else head
    k_head_off = K_OFFSET + batch * K_BATCH_STRIDE + kv_head * K_HEAD_STRIDE
    v_head_off = V_OFFSET + batch * V_BATCH_STRIDE + kv_head * V_HEAD_STRIDE
    Qh = Q + q_head_off
    Kh = K + k_head_off
    Vh = V + v_head_off
    # Write output in (B, N, H, D) order
    O_STRIDE = HEADS_PER_BATCH * D  # stride between sequence positions
    Oh = O + batch * N * O_STRIDE + head * D
    q_start = q_block * BLOCK_M
    rm = al.arange(0, BLOCK_M)
    rd = al.arange(0, D)
    rn = al.arange(0, BLOCK_N)
    q = al.load(
        Qh + (q_start + rm)[:, None] * Q_SEQ_STRIDE + rd[None, :],
        mask=(q_start + rm[:, None]) < N,
        other=0.0,
    )
    m = -1e30
    l = 0.0  # noqa: E741
    o = al.zeros((BLOCK_M, D), dtype=al.float32)
    # Block-level causal early-exit. The last Q row in this block lives at
    # absolute position Q_START_POS + q_start + BLOCK_M - 1, so the highest
    # K position it can attend to is the same value. K blocks past
    # `ceil((q_max + 1) / BLOCK_N)` are fully causally-masked, so skip them;
    # the intra-block causal where below trims the boundary block.
    if causal:
        end_kv_blocks = al.minimum(
            al.cast(
                (Q_START_POS + q_start + BLOCK_M + BLOCK_N - 1) // BLOCK_N,
                al.int32,
            ),
            al.cast(N_KV_BLOCKS, al.int32),
        )
    else:
        end_kv_blocks = N_KV_BLOCKS
    # Sliding-window: lowest logical K position any Q row in this block can
    # attend to is `q_start - SW + 1` (rm=0). Skip blocks entirely below
    # that bound; the per-row sliding mask inside the loop trims the
    # straggler columns.
    if SLIDING_WINDOW > 0:
        sw_loop_start = al.maximum(al.cast(0, al.int32), al.cast(Q_START_POS + q_start + 1 - SLIDING_WINDOW, al.int32))
        start_kv_block = sw_loop_start // BLOCK_N
    else:
        start_kv_block = al.cast(0, al.int32)
    for _jb in range(start_kv_block, end_kv_blocks, 1):
        j = _jb * BLOCK_N
        # K_WRAP > 0 means the K/V buffer is a wrap-modulo cache of that
        # physical size: read at slot = logical_pos % K_WRAP. K_WRAP=0
        # reads linearly even when SLIDING_WINDOW > 0 (sliding logic stays
        # active for masking and loop-start clamp on a contiguous buffer).
        if K_WRAP > 0:
            kv_slot = (j + rn) % K_WRAP
        else:
            kv_slot = j + rn
        k_tile = al.load(
            Kh + kv_slot[:, None] * K_SEQ_STRIDE + rd[None, :],
            mask=(j + rn[:, None]) < N_KV,
            other=0.0,
        )
        s = al.tile_dot(q, k_tile, transpose_rhs=True)
        s = s * SCALE
        s = al.where((j + rn)[None, :] < N_KV, s, -1e30)
        if causal and SLIDING_WINDOW > 0:
            # Combined causal + sliding-window mask. The sliding lower bound
            # (k_pos >= q_pos - SW + 1) is reformulated to (k_pos + SW > q_pos)
            # to keep both sides uint and avoid a subtraction that underflows
            # on uint `rm`.
            q_pos = (Q_START_POS + q_start + rm)[:, None]
            k_pos = (j + rn)[None, :]
            s = al.where((k_pos <= q_pos) & (k_pos + SLIDING_WINDOW > q_pos), s, -1e30)
        elif causal:
            # Causal mask uses ABSOLUTE sequence position: Q_START_POS is the
            # prefix length, so (Q_START_POS + q_start + rm) is the sequence
            # index of the q-th row (0 for a full-prompt prefill).
            s = al.where((Q_START_POS + q_start + rm)[:, None] >= (j + rn)[None, :], s, -1e30)
        elif SLIDING_WINDOW > 0:
            q_pos = (Q_START_POS + q_start + rm)[:, None]
            s = al.where(
                (j + rn)[None, :] + SLIDING_WINDOW > q_pos,
                s,
                -1e30,
            )
        bmax = al.max(s, axis=1)
        mn = al.maximum(m, bmax)
        alpha = al.exp(m - mn)
        l = l * alpha  # noqa: E741
        o = o * alpha
        p = al.exp(s - mn)
        l = l + al.sum(p, axis=1)  # noqa: E741
        al.barrier()
        v_tile = al.load(
            Vh + kv_slot[:, None] * V_SEQ_STRIDE + rd[None, :],
            mask=(j + rn[:, None]) < N_KV,
            other=0.0,
        )
        o = o + al.tile_dot(p, v_tile)
        m = mn
        al.barrier()
    o = o * (1.0 / l)
    al.store(
        Oh + (q_start + rm)[:, None] * O_STRIDE + rd[None, :],
        o,
        mask=((q_start + rm[:, None]) < N) & (bh < BH),
    )


@al.tunable(BLOCK_M=[16, 32, 64], BLOCK_N=[32, 64, 128], options=dict(fuse_loops=[0, 1]))
@al.kernel
def attention_strided_runtime_pos(
    Q,
    K,
    V,
    Q_START_POS_BUF,
    O: al.output,  # noqa: E741
    BH: al.constexpr = 1,
    HEADS_PER_BATCH: al.constexpr = 1,
    SEQ_LEN: al.constexpr = 1,
    HEAD_DIM: al.constexpr = 1,
    Q_OFFSET: al.constexpr = 0,
    Q_BATCH_STRIDE: al.constexpr = 0,
    Q_HEAD_STRIDE: al.constexpr = 0,
    Q_SEQ_STRIDE: al.constexpr = 0,
    K_OFFSET: al.constexpr = 0,
    K_BATCH_STRIDE: al.constexpr = 0,
    K_HEAD_STRIDE: al.constexpr = 0,
    K_SEQ_STRIDE: al.constexpr = 0,
    V_OFFSET: al.constexpr = 0,
    V_BATCH_STRIDE: al.constexpr = 0,
    V_HEAD_STRIDE: al.constexpr = 0,
    V_SEQ_STRIDE: al.constexpr = 0,
    BLOCK_M: al.constexpr = 16,
    BLOCK_N: al.constexpr = 32,
    causal: al.constexpr = 1,
    KV_GROUP: al.constexpr = 1,
    CUSTOM_SCALE: al.constexpr = 0,
    KV_LEN: al.constexpr = 0,
    SLIDING_WINDOW: al.constexpr = 0,
    K_WRAP: al.constexpr = 0,
):
    """Warm-prefill variant of `attention_strided` with runtime Q_START_POS.

    `Q_START_POS` is read from a 1-element int32 buffer instead of a constexpr,
    so one compiled plan handles arbitrary cache offsets across multi-turn
    requests. The cold/SDPA-handler path keeps the constexpr variant so Metal
    can constant-fold the causal early-exit's loop bound.
    """
    D = HEAD_DIM
    N = SEQ_LEN
    N_KV = KV_LEN if KV_LEN > 0 else N
    N_KV_BLOCKS = (N_KV + BLOCK_N - 1) // BLOCK_N
    SCALE = CUSTOM_SCALE if (CUSTOM_SCALE is not None and CUSTOM_SCALE > 0) else 1.0 / (D**0.5)
    Q_START_POS = al.load(Q_START_POS_BUF + 0)
    q_block = al.program_id(0)
    bh = al.program_id(1)
    batch = bh // HEADS_PER_BATCH
    head = bh - batch * HEADS_PER_BATCH
    q_head_off = Q_OFFSET + batch * Q_BATCH_STRIDE + head * Q_HEAD_STRIDE
    kv_head = head // KV_GROUP if KV_GROUP > 1 else head
    k_head_off = K_OFFSET + batch * K_BATCH_STRIDE + kv_head * K_HEAD_STRIDE
    v_head_off = V_OFFSET + batch * V_BATCH_STRIDE + kv_head * V_HEAD_STRIDE
    Qh = Q + q_head_off
    Kh = K + k_head_off
    Vh = V + v_head_off
    O_STRIDE = HEADS_PER_BATCH * D
    Oh = O + batch * N * O_STRIDE + head * D
    q_start = q_block * BLOCK_M
    rm = al.arange(0, BLOCK_M)
    rd = al.arange(0, D)
    rn = al.arange(0, BLOCK_N)
    q = al.load(
        Qh + (q_start + rm)[:, None] * Q_SEQ_STRIDE + rd[None, :],
        mask=(q_start + rm[:, None]) < N,
        other=0.0,
    )
    # Fold the attention scale into Q (once, before the K-loop): (q*SCALE)·K =
    # SCALE*(q·K). K stays a raw device Load whose only consumer is the QK Dot,
    # so the planner streams it straight into the MMA (no shmem tile). The
    # scaled-Q logit stays bounded → no f16 overflow.
    q = al.cast(q * SCALE, q.dtype)
    m = -1e30
    l = 0.0  # noqa: E741
    o = al.zeros((BLOCK_M, D), dtype=al.float32)
    if causal:
        # K_WRAP > 0: cache is circular of size K_WRAP. Iterate j over
        # logical positions up to the last Q's logical end — slot = j % SW
        # handles the circular layout, so the N_KV_BLOCKS clamp would
        # incorrectly stop short of positions ≥ SW. The causal+sliding
        # mask below still zeros out positions outside the SW window.
        if K_WRAP > 0:
            end_kv_blocks = al.cast(
                (Q_START_POS + q_start + BLOCK_M + BLOCK_N - 1) // BLOCK_N,
                al.int32,
            )
        else:
            end_kv_blocks = al.minimum(
                al.cast(
                    (Q_START_POS + q_start + BLOCK_M + BLOCK_N - 1) // BLOCK_N,
                    al.int32,
                ),
                al.cast(N_KV_BLOCKS, al.int32),
            )
    else:
        end_kv_blocks = N_KV_BLOCKS
    # cache_min: lowest logical K position still present in the circular
    # cache. After total_writes = Q_START_POS + N (cache_pos + seq_len)
    # writes into a SW-sized cache, the oldest surviving position is
    # max(0, total_writes - SW). For K_WRAP > 0 with wrap (total > SW)
    # the sliding-window lower bound (q_pos - SW + 1) can fall below
    # this, so positions in [q_pos-SW+1, cache_min) are in the model's
    # window but already evicted and must be masked out.
    if SLIDING_WINDOW > 0:
        if K_WRAP > 0:
            cache_min = al.maximum(al.cast(0, al.int32), al.cast(Q_START_POS + N - SLIDING_WINDOW, al.int32))
            sw_loop_start = al.maximum(al.cast(cache_min, al.int32), al.cast(Q_START_POS + q_start + 1 - SLIDING_WINDOW, al.int32))
        else:
            cache_min = al.cast(0, al.int32)
            sw_loop_start = al.maximum(al.cast(0, al.int32), al.cast(Q_START_POS + q_start + 1 - SLIDING_WINDOW, al.int32))
        start_kv_block = sw_loop_start // BLOCK_N
    else:
        cache_min = al.cast(0, al.int32)
        start_kv_block = al.cast(0, al.int32)
    for _jb in range(start_kv_block, end_kv_blocks, 1):
        j = _jb * BLOCK_N
        if K_WRAP > 0:
            kv_slot = (j + rn) % K_WRAP
            # K_WRAP > 0: every slot is in-bounds in the circular cache,
            # so the cooperative load is unconditional. Out-of-window
            # logical positions are zeroed by the causal+sliding+
            # cache-residency masks below — the `(j+rn) < N_KV` row-bound
            # would wrongly mask positions ≥ SW that are valid cache slots.
            k_tile = al.load(
                Kh + kv_slot[:, None] * K_SEQ_STRIDE + rd[None, :],
            )
        else:
            kv_slot = j + rn
            k_tile = al.load(
                Kh + kv_slot[:, None] * K_SEQ_STRIDE + rd[None, :],
                mask=(j + rn[:, None]) < N_KV,
                other=0.0,
            )
        s = al.tile_dot(q, k_tile, transpose_rhs=True)
        # The kv-bound mask only does work when the final K-block overhangs N_KV
        # (N_KV not a multiple of BLOCK_N). When N_KV divides BLOCK_N evenly
        # (native caches are powers of two) every column (j+rn) is < N_KV, so the
        # `where` masks nothing yet costs ~6% of GPU time at depth in predication
        # + the f16↔f32 round-trip. Both operands are constexprs → gate it out at
        # trace time.
        if K_WRAP == 0 and N_KV % BLOCK_N != 0:
            s = al.where((j + rn)[None, :] < N_KV, s, -1e30)
        if causal and SLIDING_WINDOW > 0:
            # Combined causal + sliding-window mask. Move SW to the k side —
            # (k_pos <= q_pos) & (k_pos + SW > q_pos) — to keep both sides uint
            # and avoid a subtraction that underflows on the per-row `rm`.
            q_pos = (Q_START_POS + q_start + rm)[:, None]
            k_pos = (j + rn)[None, :]
            valid = (k_pos <= q_pos) & (k_pos + SLIDING_WINDOW > q_pos)
            if K_WRAP > 0:
                valid = valid & (k_pos >= cache_min)
            s = al.where(valid, s, -1e30)
        elif causal:
            s = al.where((Q_START_POS + q_start + rm)[:, None] >= (j + rn)[None, :], s, -1e30)
        elif SLIDING_WINDOW > 0:
            # Same k-side reformulation: (k_pos + SW > q_pos).
            q_pos = (Q_START_POS + q_start + rm)[:, None]
            s = al.where(
                (j + rn)[None, :] + SLIDING_WINDOW > q_pos,
                s,
                -1e30,
            )
        bmax = al.max(s, axis=1)
        mn = al.maximum(m, bmax)
        alpha = al.exp(m - mn)
        l = l * alpha  # noqa: E741
        o = o * alpha
        p = al.exp(s - mn)
        l = l + al.sum(p, axis=1)  # noqa: E741
        al.barrier()
        if K_WRAP > 0:
            v_tile = al.load(
                Vh + kv_slot[:, None] * V_SEQ_STRIDE + rd[None, :],
            )
        else:
            v_tile = al.load(
                Vh + kv_slot[:, None] * V_SEQ_STRIDE + rd[None, :],
                mask=(j + rn[:, None]) < N_KV,
                other=0.0,
            )
        o = o + al.tile_dot(p, v_tile)
        m = mn
        al.barrier()
    o = o * (1.0 / l)
    al.store(
        Oh + (q_start + rm)[:, None] * O_STRIDE + rd[None, :],
        o,
        mask=((q_start + rm[:, None]) < N) & (bh < BH),
    )


@al.tunable(BLOCK_M=[8, 16, 32, 64], BLOCK_N=[8, 16, 32, 64, 128], options=dict(fuse_loops=[0, 1]))
@al.kernel
def attention_strided_runtime_pos_split(
    Q,
    K,
    V,
    Q_START_POS_BUF,
    partial_O: al.output,
    partial_lse: al.output,
    BH: al.constexpr = 1,
    HEADS_PER_BATCH: al.constexpr = 1,
    SEQ_LEN: al.constexpr = 1,
    HEAD_DIM: al.constexpr = 1,
    Q_OFFSET: al.constexpr = 0,
    Q_BATCH_STRIDE: al.constexpr = 0,
    Q_HEAD_STRIDE: al.constexpr = 0,
    Q_SEQ_STRIDE: al.constexpr = 0,
    K_OFFSET: al.constexpr = 0,
    K_BATCH_STRIDE: al.constexpr = 0,
    K_HEAD_STRIDE: al.constexpr = 0,
    K_SEQ_STRIDE: al.constexpr = 0,
    V_OFFSET: al.constexpr = 0,
    V_BATCH_STRIDE: al.constexpr = 0,
    V_HEAD_STRIDE: al.constexpr = 0,
    V_SEQ_STRIDE: al.constexpr = 0,
    BLOCK_M: al.constexpr = 16,
    BLOCK_N: al.constexpr = 32,
    causal: al.constexpr = 1,
    KV_GROUP: al.constexpr = 1,
    CUSTOM_SCALE: al.constexpr = 0,
    KV_LEN: al.constexpr = 0,
    SLIDING_WINDOW: al.constexpr = 0,
    K_WRAP: al.constexpr = 0,
    SPLITS: al.constexpr = 1,
):
    """Split-KV (flash-decoding-style) variant of `attention_strided_runtime_pos`.

    The causal KV block range [start_kv_block, end_kv_blocks) is partitioned
    evenly across `SPLITS` threadgroups along program_id(2), parallelizing the
    long serial K-scan that dominates deep-context prefill.

    Each (q_block, head, split) emits a per-split normalized softmax result
    `partial_O` + its log-sum-exp `partial_lse`; `attention_combine_splits`
    reduces over splits into the final O. Empty splits (split_lo >= split_hi)
    skip the loop and emit a 0 partial with lse → -inf, which the combine ignores.

    Grid: (Q_BLOCKS, BH, SPLITS) where Q_BLOCKS = ceil(SEQ_LEN / BLOCK_M).
    partial_O:   (SPLITS, BH, N, D)  laid out PO_SPLIT = BH*N*D per split
    partial_lse: (SPLITS, BH, N)     laid out PL_SPLIT = BH*N   per split
    """
    D = HEAD_DIM
    N = SEQ_LEN
    N_KV = KV_LEN if KV_LEN > 0 else N
    N_KV_BLOCKS = (N_KV + BLOCK_N - 1) // BLOCK_N
    SCALE = CUSTOM_SCALE if (CUSTOM_SCALE is not None and CUSTOM_SCALE > 0) else 1.0 / (D**0.5)
    Q_START_POS = al.load(Q_START_POS_BUF + 0)
    q_block = al.program_id(0)
    bh = al.program_id(1)
    split_idx = al.program_id(2)
    batch = bh // HEADS_PER_BATCH
    head = bh - batch * HEADS_PER_BATCH
    q_head_off = Q_OFFSET + batch * Q_BATCH_STRIDE + head * Q_HEAD_STRIDE
    kv_head = head // KV_GROUP if KV_GROUP > 1 else head
    k_head_off = K_OFFSET + batch * K_BATCH_STRIDE + kv_head * K_HEAD_STRIDE
    v_head_off = V_OFFSET + batch * V_BATCH_STRIDE + kv_head * V_HEAD_STRIDE
    Qh = Q + q_head_off
    Kh = K + k_head_off
    Vh = V + v_head_off
    q_start = q_block * BLOCK_M
    rm = al.arange(0, BLOCK_M)
    rd = al.arange(0, D)
    rn = al.arange(0, BLOCK_N)
    q = al.load(
        Qh + (q_start + rm)[:, None] * Q_SEQ_STRIDE + rd[None, :],
        mask=(q_start + rm[:, None]) < N,
        other=0.0,
    )
    # Scale folded into Q — see attention_strided_runtime_pos for the full
    # rationale (keeps K/V raw device Loads so they stream into the MMA).
    LOG2E = 1.4426950408889634
    q = al.cast(q * (SCALE * LOG2E), q.dtype)
    m = -1e30
    l = 0.0  # noqa: E741
    o = al.zeros((BLOCK_M, D), dtype=al.float32)
    if causal:
        if K_WRAP > 0:
            end_kv_blocks = al.cast(
                (Q_START_POS + q_start + BLOCK_M + BLOCK_N - 1) // BLOCK_N,
                al.int32,
            )
        else:
            end_kv_blocks = al.minimum(
                al.cast(
                    (Q_START_POS + q_start + BLOCK_M + BLOCK_N - 1) // BLOCK_N,
                    al.int32,
                ),
                al.cast(N_KV_BLOCKS, al.int32),
            )
    else:
        end_kv_blocks = N_KV_BLOCKS
    if SLIDING_WINDOW > 0:
        if K_WRAP > 0:
            cache_min = al.maximum(al.cast(0, al.int32), al.cast(Q_START_POS + N - SLIDING_WINDOW, al.int32))
            sw_loop_start = al.maximum(al.cast(cache_min, al.int32), al.cast(Q_START_POS + q_start + 1 - SLIDING_WINDOW, al.int32))
        else:
            cache_min = al.cast(0, al.int32)
            sw_loop_start = al.maximum(al.cast(0, al.int32), al.cast(Q_START_POS + q_start + 1 - SLIDING_WINDOW, al.int32))
        start_kv_block = sw_loop_start // BLOCK_N
    else:
        cache_min = al.cast(0, al.int32)
        start_kv_block = al.cast(0, al.int32)
    # Partition the causal block range [start_kv_block, end_kv_blocks) across
    # SPLITS threadgroups. Each takes a contiguous chunk; empty splits skip.
    total_kv_blocks = end_kv_blocks - start_kv_block
    chunk_blocks = (total_kv_blocks + SPLITS - 1) // SPLITS
    split_lo = start_kv_block + al.cast(split_idx, al.int32) * chunk_blocks
    split_hi = al.minimum(split_lo + chunk_blocks, end_kv_blocks)
    for _jb in range(split_lo, split_hi, 1):
        j = _jb * BLOCK_N
        if K_WRAP > 0:
            kv_slot = (j + rn) % K_WRAP
            k_tile = al.load(
                Kh + kv_slot[:, None] * K_SEQ_STRIDE + rd[None, :],
            )
        else:
            kv_slot = j + rn
            k_tile = al.load(
                Kh + kv_slot[:, None] * K_SEQ_STRIDE + rd[None, :],
                mask=(j + rn[:, None]) < N_KV,
                other=0.0,
            )
        s = al.tile_dot(q, k_tile, transpose_rhs=True)
        if K_WRAP == 0 and N_KV % BLOCK_N != 0:
            s = al.where((j + rn)[None, :] < N_KV, s, -1e30)
        if causal and SLIDING_WINDOW > 0:
            q_pos = (Q_START_POS + q_start + rm)[:, None]
            k_pos = (j + rn)[None, :]
            valid = (k_pos <= q_pos) & (k_pos + SLIDING_WINDOW > q_pos)
            if K_WRAP > 0:
                valid = valid & (k_pos >= cache_min)
            s = al.where(valid, s, -1e30)
        elif causal:
            s = al.where((Q_START_POS + q_start + rm)[:, None] >= (j + rn)[None, :], s, -1e30)
        elif SLIDING_WINDOW > 0:
            q_pos = (Q_START_POS + q_start + rm)[:, None]
            s = al.where(
                (j + rn)[None, :] + SLIDING_WINDOW > q_pos,
                s,
                -1e30,
            )
        bmax = al.max(s, axis=1)
        mn = al.maximum(m, bmax)
        alpha = al.exp2(m - mn)
        l = l * alpha  # noqa: E741
        o = o * alpha
        p = al.exp2(s - mn)
        l = l + al.sum(p, axis=1)  # noqa: E741
        al.barrier()
        if K_WRAP > 0:
            v_tile = al.load(
                Vh + kv_slot[:, None] * V_SEQ_STRIDE + rd[None, :],
            )
        else:
            v_tile = al.load(
                Vh + kv_slot[:, None] * V_SEQ_STRIDE + rd[None, :],
                mask=(j + rn[:, None]) < N_KV,
                other=0.0,
            )
        o = o + al.tile_dot(p, v_tile)
        m = mn
        al.barrier()
    # Per-split normalize, guarded for EMPTY splits (split_lo >= split_hi → loop
    # skipped → l == 0). The persistent-MMA pass lifts the post-scale out of the
    # loop and divides by the raw (pre-clamp) l, so a plain reciprocal gives
    # 1/0 = inf and o*(0*inf) = NaN. The branchless safe reciprocal l/(l²+ε) is
    # exactly 0 when l == 0, and 1/l (relative error ε/l² ≤ ε for l ≥ 1) otherwise;
    # ε=1e-12 on the f32 softmax denominator neither underflows nor perturbs.
    inv_l = l / (l * l + 1e-12)
    o = o * inv_l
    l = al.maximum(l, 1e-30)  # noqa: E741  (kept finite for the log in lse)
    PO_SPLIT = BH * N * D
    PL_SPLIT = BH * N
    al.store(
        partial_O + split_idx * PO_SPLIT + bh * (N * D) + (q_start + rm)[:, None] * D + rd[None, :],
        o,
        mask=((q_start + rm[:, None]) < N) & (bh < BH),
    )
    LN2 = 0.6931471805599453
    lse = m * LN2 + al.log(l)
    al.store(
        partial_lse + split_idx * PL_SPLIT + bh * N + (q_start + rm),
        lse,
        mask=((q_start + rm) < N) & (bh < BH),
    )


@al.kernel
def attention_combine_splits(
    partial_O,
    partial_lse,
    O: al.output,  # noqa: E741
    BH: al.constexpr = 1,
    HEADS_PER_BATCH: al.constexpr = 1,
    SEQ_LEN: al.constexpr = 1,
    HEAD_DIM: al.constexpr = 1,
    SPLITS: al.constexpr = 1,
):
    """Reduce SPLITS partials per (query-row, head) into the final attention out.

    Each split wrote a NORMALIZED output o_s and a log-sum-exp lse_s; the final
    output is the softmax(lse)-weighted average of the o_s across splits (online,
    numerically stable). Grid: (SEQ_LEN, BH) — ONE threadgroup per (query-row,
    head), D threads each, so each thread owns a single output element `d` and
    the online-softmax accumulator is 1 float/thread.

    The per-row mapping keeps the accumulator at 1 float/thread; a 2D (BLOCK_M, D)
    tile would lower it to a per-thread float[D] that SPILLS to thread-local memory
    held 32x-redundantly across a row's lanes (~40x slower, erasing the split-K win).
    """
    D = HEAD_DIM
    N = SEQ_LEN
    n = al.program_id(0)
    bh = al.program_id(1)
    batch = bh // HEADS_PER_BATCH
    head = bh - batch * HEADS_PER_BATCH
    rd = al.arange(0, D)
    PO_SPLIT = BH * N * D
    PL_SPLIT = BH * N
    m_g = -1e30
    denom = 0.0
    acc = al.zeros((D,), dtype=al.float32)
    for s in range(0, SPLITS, 1):
        lse_s = al.load(partial_lse + s * PL_SPLIT + bh * N + n)
        o_s = al.load(partial_O + s * PO_SPLIT + bh * (N * D) + n * D + rd)
        m_new = al.maximum(m_g, lse_s)
        scale = al.exp(m_g - m_new)
        w = al.exp(lse_s - m_new)
        acc = acc * scale + o_s * w
        denom = denom * scale + w
        m_g = m_new
    out = acc * (1.0 / denom)
    O_STRIDE = HEADS_PER_BATCH * D
    al.store(O + batch * N * O_STRIDE + head * D + n * O_STRIDE + rd, out, mask=bh < BH)


@al.tunable(BLOCK_M=[8, 16, 32, 64], BLOCK_N=[8, 16, 32], options=dict(fuse_loops=[0, 1]))
@al.kernel
def attention_strided_masked_by_batch(
    Q,
    K,
    V,
    Mask,
    O: al.output,  # noqa: E741
    BH: al.constexpr = 1,
    HEADS_PER_BATCH: al.constexpr = 1,
    SEQ_LEN: al.constexpr = 1,
    HEAD_DIM: al.constexpr = 1,
    Q_OFFSET: al.constexpr = 0,
    Q_BATCH_STRIDE: al.constexpr = 0,
    Q_HEAD_STRIDE: al.constexpr = 0,
    Q_SEQ_STRIDE: al.constexpr = 0,
    K_OFFSET: al.constexpr = 0,
    K_BATCH_STRIDE: al.constexpr = 0,
    K_HEAD_STRIDE: al.constexpr = 0,
    K_SEQ_STRIDE: al.constexpr = 0,
    V_OFFSET: al.constexpr = 0,
    V_BATCH_STRIDE: al.constexpr = 0,
    V_HEAD_STRIDE: al.constexpr = 0,
    V_SEQ_STRIDE: al.constexpr = 0,
    BLOCK_M: al.constexpr = 16,
    BLOCK_N: al.constexpr = 32,
    causal: al.constexpr = 0,
    KV_GROUP: al.constexpr = 1,
    CUSTOM_SCALE: al.constexpr = 0,
    KV_LEN: al.constexpr = 0,
    Q_START_POS: al.constexpr = 0,
):
    D = HEAD_DIM
    N = SEQ_LEN
    N_KV = KV_LEN if KV_LEN > 0 else N  # KV cache: K/V may be longer than Q
    N_KV_BLOCKS = (N_KV + BLOCK_N - 1) // BLOCK_N
    SCALE = CUSTOM_SCALE if (CUSTOM_SCALE is not None and CUSTOM_SCALE > 0) else 1.0 / (D**0.5)
    q_block = al.program_id(0)
    bh = al.program_id(1)
    batch = bh // HEADS_PER_BATCH
    head = bh - batch * HEADS_PER_BATCH
    q_head_off = Q_OFFSET + batch * Q_BATCH_STRIDE + head * Q_HEAD_STRIDE
    kv_head = head // KV_GROUP if KV_GROUP > 1 else head
    k_head_off = K_OFFSET + batch * K_BATCH_STRIDE + kv_head * K_HEAD_STRIDE
    v_head_off = V_OFFSET + batch * V_BATCH_STRIDE + kv_head * V_HEAD_STRIDE
    mask_off = batch * N * N_KV  # mask is [B, q_len, kv_len]
    Qh = Q + q_head_off
    Kh = K + k_head_off
    Vh = V + v_head_off
    Mh = Mask + mask_off
    O_STRIDE = HEADS_PER_BATCH * D
    Oh = O + batch * N * O_STRIDE + head * D
    q_start = q_block * BLOCK_M
    rm = al.arange(0, BLOCK_M)
    rd = al.arange(0, D)
    rn = al.arange(0, BLOCK_N)
    q = al.load(
        Qh + (q_start + rm)[:, None] * Q_SEQ_STRIDE + rd[None, :],
        mask=(q_start + rm[:, None]) < N,
        other=0.0,
    )
    m = -1e30
    l = 0.0  # noqa: E741
    o = al.zeros((BLOCK_M, D), dtype=al.float32)
    for _jb in range(0, N_KV_BLOCKS, 1):
        j = _jb * BLOCK_N
        k_tile = al.load(
            Kh + (j + rn)[:, None] * K_SEQ_STRIDE + rd[None, :],
            mask=(j + rn[:, None]) < N_KV,
            other=0.0,
        )
        s = al.tile_dot(q, k_tile, transpose_rhs=True)
        s = s * SCALE
        al.barrier()
        mask_tile = al.load(
            Mh + (q_start + rm)[:, None] * N_KV + (j + rn)[None, :],
            mask=((q_start + rm)[:, None] < N) & ((j + rn)[None, :] < N_KV),
            other=0.0,
        )
        al.barrier()
        s = al.maximum(s + mask_tile, -1e30)
        s = al.where((j + rn)[None, :] < N_KV, s, -1e30)
        if causal:
            # See attention_strided for Q_START_POS semantics.
            s = al.where((Q_START_POS + q_start + rm)[:, None] >= (j + rn)[None, :], s, -1e30)
        bmax = al.max(s, axis=1)
        mn = al.maximum(m, bmax)
        alpha = al.exp(m - mn)
        l = l * alpha  # noqa: E741
        o = o * alpha
        p = al.exp(s - mn)
        l = l + al.sum(p, axis=1)  # noqa: E741
        al.barrier()
        v_tile = al.load(
            Vh + (j + rn)[:, None] * V_SEQ_STRIDE + rd[None, :],
            mask=(j + rn[:, None]) < N_KV,
            other=0.0,
        )
        o = o + al.tile_dot(p, v_tile)
        m = mn
        al.barrier()
    o = o * (1.0 / l)
    al.store(
        Oh + (q_start + rm)[:, None] * O_STRIDE + rd[None, :],
        o,
        mask=((q_start + rm[:, None]) < N) & (bh < BH),
    )


# fuse_loops=1 is excluded: per-kernel max-diff stays within f32 tolerance
# but the error compounds across decoder layers into NaN at the lm_head.
@al.tunable(BLOCK_M=[8, 16, 32, 64], BLOCK_N=[8, 16, 32], options=dict(fuse_loops=[0]))
@al.kernel
def attention_strided_masked_by_batch_with_lse(
    Q,
    K,
    V,
    Mask,
    O: al.output,  # noqa: E741
    log_sum_exp: al.output,
    BH: al.constexpr = 1,
    HEADS_PER_BATCH: al.constexpr = 1,
    SEQ_LEN: al.constexpr = 1,
    HEAD_DIM: al.constexpr = 1,
    Q_OFFSET: al.constexpr = 0,
    Q_BATCH_STRIDE: al.constexpr = 0,
    Q_HEAD_STRIDE: al.constexpr = 0,
    Q_SEQ_STRIDE: al.constexpr = 0,
    K_OFFSET: al.constexpr = 0,
    K_BATCH_STRIDE: al.constexpr = 0,
    K_HEAD_STRIDE: al.constexpr = 0,
    K_SEQ_STRIDE: al.constexpr = 0,
    V_OFFSET: al.constexpr = 0,
    V_BATCH_STRIDE: al.constexpr = 0,
    V_HEAD_STRIDE: al.constexpr = 0,
    V_SEQ_STRIDE: al.constexpr = 0,
    BLOCK_M: al.constexpr = 16,
    BLOCK_N: al.constexpr = 32,
    causal: al.constexpr = 0,
    KV_GROUP: al.constexpr = 1,
    CUSTOM_SCALE: al.constexpr = 0,
    KV_LEN: al.constexpr = 0,
    Q_START_POS: al.constexpr = 0,
):
    """Fwd attention + lse in one pass, saving a re-read of Q/K/mask for the
    separate logsumexp kernel. Body mirrors `attention_strided_masked_by_batch`;
    the only addition is the `log_sum_exp` store using the final (m, l)."""
    D = HEAD_DIM
    N = SEQ_LEN
    N_KV = KV_LEN if KV_LEN > 0 else N
    N_KV_BLOCKS = (N_KV + BLOCK_N - 1) // BLOCK_N
    SCALE = CUSTOM_SCALE if (CUSTOM_SCALE is not None and CUSTOM_SCALE > 0) else 1.0 / (D**0.5)
    q_block = al.program_id(0)
    bh = al.program_id(1)
    batch = bh // HEADS_PER_BATCH
    head = bh - batch * HEADS_PER_BATCH
    q_head_off = Q_OFFSET + batch * Q_BATCH_STRIDE + head * Q_HEAD_STRIDE
    kv_head = head // KV_GROUP if KV_GROUP > 1 else head
    k_head_off = K_OFFSET + batch * K_BATCH_STRIDE + kv_head * K_HEAD_STRIDE
    v_head_off = V_OFFSET + batch * V_BATCH_STRIDE + kv_head * V_HEAD_STRIDE
    mask_off = batch * N * N_KV
    Qh = Q + q_head_off
    Kh = K + k_head_off
    Vh = V + v_head_off
    Mh = Mask + mask_off
    O_STRIDE = HEADS_PER_BATCH * D
    Oh = O + batch * N * O_STRIDE + head * D
    q_start = q_block * BLOCK_M
    rm = al.arange(0, BLOCK_M)
    rd = al.arange(0, D)
    rn = al.arange(0, BLOCK_N)
    rc = al.arange(0, 1)
    q = al.load(
        Qh + (q_start + rm)[:, None] * Q_SEQ_STRIDE + rd[None, :],
        mask=(q_start + rm[:, None]) < N,
        other=0.0,
    )
    m = -1e30
    l = 0.0  # noqa: E741
    o = al.zeros((BLOCK_M, D), dtype=al.float32)
    for _jb in range(0, N_KV_BLOCKS, 1):
        j = _jb * BLOCK_N
        k_tile = al.load(
            Kh + (j + rn)[:, None] * K_SEQ_STRIDE + rd[None, :],
            mask=(j + rn[:, None]) < N_KV,
            other=0.0,
        )
        s = al.tile_dot(q, k_tile, transpose_rhs=True)
        s = s * SCALE
        al.barrier()
        mask_tile = al.load(
            Mh + (q_start + rm)[:, None] * N_KV + (j + rn)[None, :],
            mask=((q_start + rm)[:, None] < N) & ((j + rn)[None, :] < N_KV),
            other=0.0,
        )
        al.barrier()
        s = al.maximum(s + mask_tile, -1e30)
        s = al.where((j + rn)[None, :] < N_KV, s, -1e30)
        if causal:
            # See attention_strided for Q_START_POS semantics.
            s = al.where((Q_START_POS + q_start + rm)[:, None] >= (j + rn)[None, :], s, -1e30)
        bmax = al.max(s, axis=1)
        mn = al.maximum(m, bmax)
        alpha = al.exp(m - mn)
        l = l * alpha  # noqa: E741
        o = o * alpha
        p = al.exp(s - mn)
        l = l + al.sum(p, axis=1)  # noqa: E741
        al.barrier()
        v_tile = al.load(
            Vh + (j + rn)[:, None] * V_SEQ_STRIDE + rd[None, :],
            mask=(j + rn[:, None]) < N_KV,
            other=0.0,
        )
        o = o + al.tile_dot(p, v_tile)
        m = mn
        al.barrier()
    o = o * (1.0 / l)
    al.store(
        Oh + (q_start + rm)[:, None] * O_STRIDE + rd[None, :],
        o,
        mask=((q_start + rm[:, None]) < N) & (bh < BH),
    )
    # lse = m + log(l) per row. Layout matches attention_strided_logsumexp_masked_by_batch:
    # log_sum_exp[bh * N + q_start + rm]
    lse_tile = al.zeros((BLOCK_M, 1), dtype=al.float32)
    lse_tile = lse_tile + m + al.log(l)
    al.store(
        log_sum_exp + (rm[:, None] * 1 + (bh * N + q_start)) + rc[None, :],
        lse_tile,
        mask=((q_start + rm[:, None]) < N) & (bh < BH),
    )


@al.tunable(BLOCK_M=[8, 16, 32, 64], BLOCK_N=[8, 16, 32], options=dict(fuse_loops=[0, 1]))
@al.kernel
def attention_strided_logsumexp(
    Q,
    K,
    log_sum_exp: al.output,
    BH: al.constexpr = 1,
    HEADS_PER_BATCH: al.constexpr = 1,
    SEQ_LEN: al.constexpr = 1,
    HEAD_DIM: al.constexpr = 1,
    Q_OFFSET: al.constexpr = 0,
    Q_BATCH_STRIDE: al.constexpr = 0,
    Q_HEAD_STRIDE: al.constexpr = 0,
    Q_SEQ_STRIDE: al.constexpr = 0,
    K_OFFSET: al.constexpr = 0,
    K_BATCH_STRIDE: al.constexpr = 0,
    K_HEAD_STRIDE: al.constexpr = 0,
    K_SEQ_STRIDE: al.constexpr = 0,
    BLOCK_M: al.constexpr = 16,
    BLOCK_N: al.constexpr = 32,
    causal: al.constexpr = 0,
    KV_GROUP: al.constexpr = 1,
    CUSTOM_SCALE: al.constexpr = 0,
    KV_LEN: al.constexpr = 0,
):
    D = HEAD_DIM
    N = SEQ_LEN
    N_KV = KV_LEN if KV_LEN > 0 else N
    N_KV_BLOCKS = (N_KV + BLOCK_N - 1) // BLOCK_N
    SCALE = CUSTOM_SCALE if (CUSTOM_SCALE is not None and CUSTOM_SCALE > 0) else 1.0 / (D**0.5)
    q_block = al.program_id(0)
    bh = al.program_id(1)
    batch = bh // HEADS_PER_BATCH
    head = bh - batch * HEADS_PER_BATCH
    kv_head = head // KV_GROUP if KV_GROUP > 1 else head
    Qh = Q + Q_OFFSET + batch * Q_BATCH_STRIDE + head * Q_HEAD_STRIDE
    Kh = K + K_OFFSET + batch * K_BATCH_STRIDE + kv_head * K_HEAD_STRIDE
    q_start = q_block * BLOCK_M
    rm = al.arange(0, BLOCK_M)
    rc = al.arange(0, 1)
    rn = al.arange(0, BLOCK_N)
    rd = al.arange(0, D)
    row_mask = (q_start + rm) < N
    q = al.load(
        Qh + (q_start + rm)[:, None] * Q_SEQ_STRIDE + rd[None, :],
        mask=row_mask[:, None],
        other=0.0,
    )
    m = -1e30
    l = 0.0  # noqa: E741
    for _jb in range(0, N_KV_BLOCKS, 1):
        j = _jb * BLOCK_N
        kv_mask = (j + rn) < N_KV
        k_tile = al.load(
            Kh + (j + rn)[:, None] * K_SEQ_STRIDE + rd[None, :],
            mask=kv_mask[:, None],
            other=0.0,
        )
        s = al.tile_dot(q, k_tile, transpose_rhs=True) * SCALE
        s = al.where(kv_mask[None, :], s, -1e30)
        s = al.where(row_mask[:, None], s, -1e30)
        if causal:
            s = al.where((q_start + rm)[:, None] >= (j + rn)[None, :], s, -1e30)
        bmax = al.max(s, axis=1)
        mn = al.maximum(m, bmax)
        alpha = al.exp(m - mn)
        l = l * alpha  # noqa: E741
        p = al.exp(s - mn)
        l = l + al.sum(p, axis=1)  # noqa: E741
        m = mn
    lse_tile = al.zeros((BLOCK_M, 1), dtype=al.float32)
    lse_tile = lse_tile + m + al.log(l)
    al.store(
        log_sum_exp + (rm[:, None] * 1 + (bh * N + q_start)) + rc[None, :],
        lse_tile,
        mask=row_mask[:, None] & (bh < BH),
    )


@al.tunable(BLOCK_M=[8, 16, 32, 64], BLOCK_N=[8, 16, 32], options=dict(fuse_loops=[0, 1]))
@al.kernel
def attention_strided_logsumexp_masked_by_batch(
    Q,
    K,
    Mask,
    log_sum_exp: al.output,
    BH: al.constexpr = 1,
    HEADS_PER_BATCH: al.constexpr = 1,
    SEQ_LEN: al.constexpr = 1,
    HEAD_DIM: al.constexpr = 1,
    Q_OFFSET: al.constexpr = 0,
    Q_BATCH_STRIDE: al.constexpr = 0,
    Q_HEAD_STRIDE: al.constexpr = 0,
    Q_SEQ_STRIDE: al.constexpr = 0,
    K_OFFSET: al.constexpr = 0,
    K_BATCH_STRIDE: al.constexpr = 0,
    K_HEAD_STRIDE: al.constexpr = 0,
    K_SEQ_STRIDE: al.constexpr = 0,
    BLOCK_M: al.constexpr = 16,
    BLOCK_N: al.constexpr = 32,
    causal: al.constexpr = 0,
    KV_GROUP: al.constexpr = 1,
    CUSTOM_SCALE: al.constexpr = 0,
    KV_LEN: al.constexpr = 0,
):
    D = HEAD_DIM
    N = SEQ_LEN
    N_KV = KV_LEN if KV_LEN > 0 else N
    N_KV_BLOCKS = (N_KV + BLOCK_N - 1) // BLOCK_N
    SCALE = CUSTOM_SCALE if (CUSTOM_SCALE is not None and CUSTOM_SCALE > 0) else 1.0 / (D**0.5)
    q_block = al.program_id(0)
    bh = al.program_id(1)
    batch = bh // HEADS_PER_BATCH
    head = bh - batch * HEADS_PER_BATCH
    kv_head = head // KV_GROUP if KV_GROUP > 1 else head
    Qh = Q + Q_OFFSET + batch * Q_BATCH_STRIDE + head * Q_HEAD_STRIDE
    Kh = K + K_OFFSET + batch * K_BATCH_STRIDE + kv_head * K_HEAD_STRIDE
    Mh = Mask + batch * N * N_KV
    q_start = q_block * BLOCK_M
    rm = al.arange(0, BLOCK_M)
    rc = al.arange(0, 1)
    rn = al.arange(0, BLOCK_N)
    rd = al.arange(0, D)
    row_mask = (q_start + rm) < N
    q = al.load(
        Qh + (q_start + rm)[:, None] * Q_SEQ_STRIDE + rd[None, :],
        mask=row_mask[:, None],
        other=0.0,
    )
    m = -1e30
    l = 0.0  # noqa: E741
    for _jb in range(0, N_KV_BLOCKS, 1):
        j = _jb * BLOCK_N
        kv_mask = (j + rn) < N_KV
        k_tile = al.load(
            Kh + (j + rn)[:, None] * K_SEQ_STRIDE + rd[None, :],
            mask=kv_mask[:, None],
            other=0.0,
        )
        s = al.tile_dot(q, k_tile, transpose_rhs=True) * SCALE
        mask_tile = al.load(
            Mh + (q_start + rm)[:, None] * N_KV + (j + rn)[None, :],
            mask=row_mask[:, None] & kv_mask[None, :],
            other=0.0,
        )
        s = al.maximum(s + mask_tile, -1e30)
        s = al.where(kv_mask[None, :], s, -1e30)
        s = al.where(row_mask[:, None], s, -1e30)
        if causal:
            s = al.where((q_start + rm)[:, None] >= (j + rn)[None, :], s, -1e30)
        bmax = al.max(s, axis=1)
        mn = al.maximum(m, bmax)
        alpha = al.exp(m - mn)
        l = l * alpha  # noqa: E741
        p = al.exp(s - mn)
        l = l + al.sum(p, axis=1)  # noqa: E741
        m = mn
    lse_tile = al.zeros((BLOCK_M, 1), dtype=al.float32)
    lse_tile = lse_tile + m + al.log(l)
    al.store(
        log_sum_exp + (rm[:, None] * 1 + (bh * N + q_start)) + rc[None, :],
        lse_tile,
        mask=row_mask[:, None] & (bh < BH),
    )


@al.kernel
def attention_compute_delta_strided(
    dO,
    O,
    Delta: al.output,
    BH: al.constexpr = 1,
    HEADS_PER_BATCH: al.constexpr = 1,
    SEQ_LEN: al.constexpr = 1,
    HEAD_DIM: al.constexpr = 1,
    GO_OFFSET: al.constexpr = 0,
    GO_BATCH_STRIDE: al.constexpr = 0,
    GO_HEAD_STRIDE: al.constexpr = 0,
    GO_SEQ_STRIDE: al.constexpr = 0,
    O_OFFSET: al.constexpr = 0,
    O_BATCH_STRIDE: al.constexpr = 0,
    O_HEAD_STRIDE: al.constexpr = 0,
    O_SEQ_STRIDE: al.constexpr = 0,
    BLOCK_M: al.constexpr = 16,
):
    """FlashAttention-2 delta precompute: D[bh, s] = sum_d(dO[b,h,s,d] * O[b,h,s,d]).

    Reused by attention_strided_backward_dq and _dkdv so the row-wise dot
    product isn't recomputed on every (KV-block, Q-block) pair.

    Output Delta is contiguous (BH*SEQ_LEN,) f32. Grid: (ceil(SEQ_LEN/BLOCK_M), BH).
    """
    block = al.program_id(0)
    bh = al.program_id(1)
    batch = bh // HEADS_PER_BATCH
    head = bh - batch * HEADS_PER_BATCH

    rm = al.arange(0, BLOCK_M)
    rd = al.arange(0, HEAD_DIM)

    s = block * BLOCK_M
    row_mask = (s + rm) < SEQ_LEN

    go_base = GO_OFFSET + batch * GO_BATCH_STRIDE + head * GO_HEAD_STRIDE
    o_base = O_OFFSET + batch * O_BATCH_STRIDE + head * O_HEAD_STRIDE

    go = al.load(
        dO + go_base + (s + rm)[:, None] * GO_SEQ_STRIDE + rd[None, :],
        mask=row_mask[:, None],
        other=0.0,
    )
    o = al.load(
        O + o_base + (s + rm)[:, None] * O_SEQ_STRIDE + rd[None, :],
        mask=row_mask[:, None],
        other=0.0,
    )
    # Explicit f32 casts: Metal's `bfloat * bfloat` does not auto-promote to float
    # (unlike `half`), so without them the multiply lives in bf16. The ~0.4% bf16
    # product error breaks the FA-2 cancellation `ds = p*(dp - delta)` under K-bias
    # amplification (Qwen K bias 316 → q_proj LoRA grad explodes).
    delta = al.sum(al.cast(go, al.float32) * al.cast(o, al.float32), axis=1)
    al.store(Delta + bh * SEQ_LEN + s + rm, delta, mask=row_mask)


@al.tunable(BLOCK_M=[8, 16, 32, 64], BLOCK_N=[8, 16, 32], options=dict(fuse_loops=[0, 1]))
@al.kernel
def attention_strided_backward_dq(
    dO,
    Q,
    K,
    V,
    LogSumExp,
    Delta,
    dQ: al.output,
    BH: al.constexpr = 1,
    HEADS_PER_BATCH: al.constexpr = 1,
    SEQ_LEN: al.constexpr = 1,
    HEAD_DIM: al.constexpr = 1,
    GO_OFFSET: al.constexpr = 0,
    GO_BATCH_STRIDE: al.constexpr = 0,
    GO_HEAD_STRIDE: al.constexpr = 0,
    GO_SEQ_STRIDE: al.constexpr = 0,
    Q_OFFSET: al.constexpr = 0,
    Q_BATCH_STRIDE: al.constexpr = 0,
    Q_HEAD_STRIDE: al.constexpr = 0,
    Q_SEQ_STRIDE: al.constexpr = 0,
    K_OFFSET: al.constexpr = 0,
    K_BATCH_STRIDE: al.constexpr = 0,
    K_HEAD_STRIDE: al.constexpr = 0,
    K_SEQ_STRIDE: al.constexpr = 0,
    V_OFFSET: al.constexpr = 0,
    V_BATCH_STRIDE: al.constexpr = 0,
    V_HEAD_STRIDE: al.constexpr = 0,
    V_SEQ_STRIDE: al.constexpr = 0,
    DQ_OFFSET: al.constexpr = 0,
    DQ_BATCH_STRIDE: al.constexpr = 0,
    DQ_HEAD_STRIDE: al.constexpr = 0,
    DQ_SEQ_STRIDE: al.constexpr = 0,
    BLOCK_M: al.constexpr = 16,
    BLOCK_N: al.constexpr = 16,
    causal: al.constexpr = 0,
    KV_GROUP: al.constexpr = 1,
    CUSTOM_SCALE: al.constexpr = 0,
    KV_LEN: al.constexpr = 0,
    HIGH_PRECISION: al.constexpr = 0,
):
    D = HEAD_DIM
    N = SEQ_LEN
    N_KV = KV_LEN if KV_LEN > 0 else N
    N_KV_BLOCKS = (N_KV + BLOCK_N - 1) // BLOCK_N
    SCALE = CUSTOM_SCALE if (CUSTOM_SCALE is not None and CUSTOM_SCALE > 0) else 1.0 / (D**0.5)

    q_block = al.program_id(0)
    bh = al.program_id(1)
    batch = bh // HEADS_PER_BATCH
    head = bh - batch * HEADS_PER_BATCH
    kv_head = head // KV_GROUP if KV_GROUP > 1 else head

    GOh = dO + GO_OFFSET + batch * GO_BATCH_STRIDE + head * GO_HEAD_STRIDE
    Qh = Q + Q_OFFSET + batch * Q_BATCH_STRIDE + head * Q_HEAD_STRIDE
    Kh = K + K_OFFSET + batch * K_BATCH_STRIDE + kv_head * K_HEAD_STRIDE
    Vh = V + V_OFFSET + batch * V_BATCH_STRIDE + kv_head * V_HEAD_STRIDE
    dQh = dQ + DQ_OFFSET + batch * DQ_BATCH_STRIDE + head * DQ_HEAD_STRIDE

    q_start = q_block * BLOCK_M
    rm = al.arange(0, BLOCK_M)
    rn = al.arange(0, BLOCK_N)
    rd = al.arange(0, D)
    row_mask = (q_start + rm) < N

    q = al.load(
        Qh + (q_start + rm)[:, None] * Q_SEQ_STRIDE + rd[None, :],
        mask=row_mask[:, None],
        other=0.0,
    )
    go = al.load(
        GOh + (q_start + rm)[:, None] * GO_SEQ_STRIDE + rd[None, :],
        mask=row_mask[:, None],
        other=0.0,
    )
    lse = al.load(LogSumExp + bh * N + q_start + rm, mask=row_mask, other=0.0)
    drow = al.load(Delta + bh * N + q_start + rm, mask=row_mask, other=0.0)
    dq = al.zeros((BLOCK_M, D), dtype=al.float32)

    for _jb in range(0, N_KV_BLOCKS, 1):
        j = _jb * BLOCK_N
        kv_mask = (j + rn) < N_KV
        # Pre-scale K by SCALE inline with the load. Combining the load and
        # the scalar multiply in a single expression lets prologue fusion
        # absorb the multiply into the cooperative-load epilogue, so we keep
        # ONE shmem slot for K (no separate raw-K + K_scaled buffers). Math
        # unchanged: q @ (K * S).T = q @ K.T * S, and ds @ (K * S) = (ds @
        # K) * S — both factor the post-MMA `* SCALE` out and let the dq
        # accumulator stay register-resident via `_opt_persistent_mma`.
        k_tile = (
            al.load(
                Kh + (j + rn)[:, None] * K_SEQ_STRIDE + rd[None, :],
                mask=kv_mask[:, None],
                other=0.0,
            )
            * SCALE
        )
        v_tile = al.load(
            Vh + (j + rn)[:, None] * V_SEQ_STRIDE + rd[None, :],
            mask=kv_mask[:, None],
            other=0.0,
        )
        s = al.tile_dot(q, k_tile, transpose_rhs=True)
        s = al.where(kv_mask[None, :], s, -1e30)
        s = al.where(row_mask[:, None], s, -1e30)
        if causal:
            s = al.where((q_start + rm)[:, None] >= (j + rn)[None, :], s, -1e30)
        # Clamp s - lse to <= 0 so p stays in [0, 1] even when bwd's
        # tile_dot recompute drifts a few ulps above the fwd-stored lse.
        sd = al.minimum(s - lse[:, None], 0.0)
        p = al.exp(sd)
        dp = al.tile_dot(go, v_tile, transpose_rhs=True)
        ds = p * (dp - drow[:, None])
        dq = dq + al.tile_dot(ds, k_tile)

    al.store(
        dQh + (q_start + rm)[:, None] * DQ_SEQ_STRIDE + rd[None, :],
        dq,
        mask=row_mask[:, None] & (bh < BH),
    )


@al.tunable(BLOCK_M=[8, 16, 32, 64], BLOCK_N=[8, 16, 32], options=dict(fuse_loops=[0, 1]))
@al.kernel
def attention_strided_backward_dq_masked_by_batch(
    dO,
    Q,
    K,
    V,
    LogSumExp,
    Mask,
    Delta,
    dQ: al.output,
    BH: al.constexpr = 1,
    HEADS_PER_BATCH: al.constexpr = 1,
    SEQ_LEN: al.constexpr = 1,
    HEAD_DIM: al.constexpr = 1,
    GO_OFFSET: al.constexpr = 0,
    GO_BATCH_STRIDE: al.constexpr = 0,
    GO_HEAD_STRIDE: al.constexpr = 0,
    GO_SEQ_STRIDE: al.constexpr = 0,
    Q_OFFSET: al.constexpr = 0,
    Q_BATCH_STRIDE: al.constexpr = 0,
    Q_HEAD_STRIDE: al.constexpr = 0,
    Q_SEQ_STRIDE: al.constexpr = 0,
    K_OFFSET: al.constexpr = 0,
    K_BATCH_STRIDE: al.constexpr = 0,
    K_HEAD_STRIDE: al.constexpr = 0,
    K_SEQ_STRIDE: al.constexpr = 0,
    V_OFFSET: al.constexpr = 0,
    V_BATCH_STRIDE: al.constexpr = 0,
    V_HEAD_STRIDE: al.constexpr = 0,
    V_SEQ_STRIDE: al.constexpr = 0,
    DQ_OFFSET: al.constexpr = 0,
    DQ_BATCH_STRIDE: al.constexpr = 0,
    DQ_HEAD_STRIDE: al.constexpr = 0,
    DQ_SEQ_STRIDE: al.constexpr = 0,
    BLOCK_M: al.constexpr = 16,
    BLOCK_N: al.constexpr = 16,
    causal: al.constexpr = 0,
    KV_GROUP: al.constexpr = 1,
    CUSTOM_SCALE: al.constexpr = 0,
    KV_LEN: al.constexpr = 0,
    HIGH_PRECISION: al.constexpr = 0,
):
    D = HEAD_DIM
    N = SEQ_LEN
    N_KV = KV_LEN if KV_LEN > 0 else N
    N_KV_BLOCKS = (N_KV + BLOCK_N - 1) // BLOCK_N
    SCALE = CUSTOM_SCALE if (CUSTOM_SCALE is not None and CUSTOM_SCALE > 0) else 1.0 / (D**0.5)

    q_block = al.program_id(0)
    bh = al.program_id(1)
    batch = bh // HEADS_PER_BATCH
    head = bh - batch * HEADS_PER_BATCH
    kv_head = head // KV_GROUP if KV_GROUP > 1 else head

    GOh = dO + GO_OFFSET + batch * GO_BATCH_STRIDE + head * GO_HEAD_STRIDE
    Qh = Q + Q_OFFSET + batch * Q_BATCH_STRIDE + head * Q_HEAD_STRIDE
    Kh = K + K_OFFSET + batch * K_BATCH_STRIDE + kv_head * K_HEAD_STRIDE
    Vh = V + V_OFFSET + batch * V_BATCH_STRIDE + kv_head * V_HEAD_STRIDE
    Mh = Mask + batch * N * N_KV
    dQh = dQ + DQ_OFFSET + batch * DQ_BATCH_STRIDE + head * DQ_HEAD_STRIDE

    q_start = q_block * BLOCK_M
    rm = al.arange(0, BLOCK_M)
    rn = al.arange(0, BLOCK_N)
    rd = al.arange(0, D)
    row_mask = (q_start + rm) < N

    q = al.load(
        Qh + (q_start + rm)[:, None] * Q_SEQ_STRIDE + rd[None, :],
        mask=row_mask[:, None],
        other=0.0,
    )
    go = al.load(
        GOh + (q_start + rm)[:, None] * GO_SEQ_STRIDE + rd[None, :],
        mask=row_mask[:, None],
        other=0.0,
    )
    lse = al.load(LogSumExp + bh * N + q_start + rm, mask=row_mask, other=0.0)
    drow = al.load(Delta + bh * N + q_start + rm, mask=row_mask, other=0.0)
    dq = al.zeros((BLOCK_M, D), dtype=al.float32)

    for _jb in range(0, N_KV_BLOCKS, 1):
        j = _jb * BLOCK_N
        kv_mask = (j + rn) < N_KV
        # Pre-scale K by SCALE inline with the load. Combining the load and
        # the scalar multiply in a single expression lets prologue fusion
        # absorb the multiply into the cooperative-load epilogue, so we keep
        # ONE shmem slot for K (no separate raw-K + K_scaled buffers). Math
        # unchanged: q @ (K * S).T = q @ K.T * S, and ds @ (K * S) = (ds @
        # K) * S — both factor the post-MMA `* SCALE` out and let the dq
        # accumulator stay register-resident via `_opt_persistent_mma`.
        k_tile = (
            al.load(
                Kh + (j + rn)[:, None] * K_SEQ_STRIDE + rd[None, :],
                mask=kv_mask[:, None],
                other=0.0,
            )
            * SCALE
        )
        v_tile = al.load(
            Vh + (j + rn)[:, None] * V_SEQ_STRIDE + rd[None, :],
            mask=kv_mask[:, None],
            other=0.0,
        )
        s = al.tile_dot(q, k_tile, transpose_rhs=True)
        mask_tile = al.load(
            Mh + (q_start + rm)[:, None] * N_KV + (j + rn)[None, :],
            mask=row_mask[:, None] & kv_mask[None, :],
            other=0.0,
        )
        s = al.maximum(s + mask_tile, -1e30)
        s = al.where(kv_mask[None, :], s, -1e30)
        s = al.where(row_mask[:, None], s, -1e30)
        if causal:
            s = al.where((q_start + rm)[:, None] >= (j + rn)[None, :], s, -1e30)
        # Clamp s - lse to <= 0 so p stays in [0, 1] even when bwd's
        # tile_dot recompute drifts a few ulps above the fwd-stored lse
        # (Qwen K bias ~250 makes |s| ~20000, where bf16/f32 cancellation in
        # `s - lse` can flip sign on the peak token and overflow exp).
        sd = al.minimum(s - lse[:, None], 0.0)
        p = al.exp(sd)
        dp = al.tile_dot(go, v_tile, transpose_rhs=True)
        ds = p * (dp - drow[:, None])
        dq = dq + al.tile_dot(ds, k_tile)

    al.store(
        dQh + (q_start + rm)[:, None] * DQ_SEQ_STRIDE + rd[None, :],
        dq,
        mask=row_mask[:, None] & (bh < BH),
    )


@al.tunable(BLOCK_M=[8, 16, 32, 64], BLOCK_N=[8, 16, 32], options=dict(fuse_loops=[0, 1]))
@al.kernel
def attention_strided_backward_dkdv(
    dO,
    Q,
    K,
    V,
    LogSumExp,
    Delta,
    dK: al.output,
    dV: al.output,
    BH: al.constexpr = 1,
    BH_KV: al.constexpr = 1,
    HEADS_PER_BATCH: al.constexpr = 1,
    KV_HEADS_PER_BATCH: al.constexpr = 1,
    SEQ_LEN: al.constexpr = 1,
    HEAD_DIM: al.constexpr = 1,
    GO_OFFSET: al.constexpr = 0,
    GO_BATCH_STRIDE: al.constexpr = 0,
    GO_HEAD_STRIDE: al.constexpr = 0,
    GO_SEQ_STRIDE: al.constexpr = 0,
    Q_OFFSET: al.constexpr = 0,
    Q_BATCH_STRIDE: al.constexpr = 0,
    Q_HEAD_STRIDE: al.constexpr = 0,
    Q_SEQ_STRIDE: al.constexpr = 0,
    K_OFFSET: al.constexpr = 0,
    K_BATCH_STRIDE: al.constexpr = 0,
    K_HEAD_STRIDE: al.constexpr = 0,
    K_SEQ_STRIDE: al.constexpr = 0,
    V_OFFSET: al.constexpr = 0,
    V_BATCH_STRIDE: al.constexpr = 0,
    V_HEAD_STRIDE: al.constexpr = 0,
    V_SEQ_STRIDE: al.constexpr = 0,
    DK_OFFSET: al.constexpr = 0,
    DK_BATCH_STRIDE: al.constexpr = 0,
    DK_HEAD_STRIDE: al.constexpr = 0,
    DK_SEQ_STRIDE: al.constexpr = 0,
    DV_OFFSET: al.constexpr = 0,
    DV_BATCH_STRIDE: al.constexpr = 0,
    DV_HEAD_STRIDE: al.constexpr = 0,
    DV_SEQ_STRIDE: al.constexpr = 0,
    BLOCK_M: al.constexpr = 16,
    BLOCK_N: al.constexpr = 16,
    causal: al.constexpr = 0,
    KV_GROUP: al.constexpr = 1,
    CUSTOM_SCALE: al.constexpr = 0,
    KV_LEN: al.constexpr = 0,
    HIGH_PRECISION: al.constexpr = 0,
):
    D = HEAD_DIM
    N = SEQ_LEN
    N_KV = KV_LEN if KV_LEN > 0 else N
    N_Q_BLOCKS = (N + BLOCK_M - 1) // BLOCK_M
    SCALE = CUSTOM_SCALE if (CUSTOM_SCALE is not None and CUSTOM_SCALE > 0) else 1.0 / (D**0.5)

    k_block = al.program_id(0)
    kv_bh = al.program_id(1)
    batch = kv_bh // KV_HEADS_PER_BATCH
    kv_head = kv_bh - batch * KV_HEADS_PER_BATCH

    Kh = K + K_OFFSET + batch * K_BATCH_STRIDE + kv_head * K_HEAD_STRIDE
    Vh = V + V_OFFSET + batch * V_BATCH_STRIDE + kv_head * V_HEAD_STRIDE
    dKh = dK + DK_OFFSET + batch * DK_BATCH_STRIDE + kv_head * DK_HEAD_STRIDE
    dVh = dV + DV_OFFSET + batch * DV_BATCH_STRIDE + kv_head * DV_HEAD_STRIDE

    k_start = k_block * BLOCK_N
    rm = al.arange(0, BLOCK_M)
    rn = al.arange(0, BLOCK_N)
    rd = al.arange(0, D)
    kv_mask = (k_start + rn) < N_KV

    k_tile = al.load(
        Kh + (k_start + rn)[:, None] * K_SEQ_STRIDE + rd[None, :],
        mask=kv_mask[:, None],
        other=0.0,
    )
    v_tile = al.load(
        Vh + (k_start + rn)[:, None] * V_SEQ_STRIDE + rd[None, :],
        mask=kv_mask[:, None],
        other=0.0,
    )
    dk = al.zeros((BLOCK_N, D), dtype=al.float32)
    dv = al.zeros((BLOCK_N, D), dtype=al.float32)

    for g in range(KV_GROUP):
        q_head = kv_head * KV_GROUP + g
        GOh = dO + GO_OFFSET + batch * GO_BATCH_STRIDE + q_head * GO_HEAD_STRIDE
        Qh = Q + Q_OFFSET + batch * Q_BATCH_STRIDE + q_head * Q_HEAD_STRIDE
        lse_off = (batch * HEADS_PER_BATCH + q_head) * N

        for _ib in range(0, N_Q_BLOCKS, 1):
            i = _ib * BLOCK_M
            row_mask = (i + rm) < N
            q = al.load(
                Qh + (i + rm)[:, None] * Q_SEQ_STRIDE + rd[None, :],
                mask=row_mask[:, None],
                other=0.0,
            )
            go = al.load(
                GOh + (i + rm)[:, None] * GO_SEQ_STRIDE + rd[None, :],
                mask=row_mask[:, None],
                other=0.0,
            )
            lse = al.load(LogSumExp + lse_off + i + rm, mask=row_mask, other=0.0)
            drow = al.load(Delta + lse_off + i + rm, mask=row_mask, other=0.0)

            s_t = al.tile_dot(k_tile, q, transpose_rhs=True) * SCALE
            s_t = al.where(kv_mask[:, None], s_t, -1e30)
            s_t = al.where(row_mask[None, :], s_t, -1e30)
            if causal:
                s_t = al.where((i + rm)[None, :] >= (k_start + rn)[:, None], s_t, -1e30)
            sd_t = al.minimum(s_t - lse[None, :], 0.0)
            p_t = al.exp(sd_t)
            dp_t = al.tile_dot(v_tile, go, transpose_rhs=True)
            dv = dv + al.tile_dot(p_t, go)
            ds_t = p_t * (dp_t - drow[None, :])
            dk = dk + al.tile_dot(ds_t, q)

    # SCALE folded out of the inner accumulator. _opt_persistent_mma matches
    # the bare Add(init, Dot) body; the trailing scalar Mul gets absorbed into
    # Store.transform via _absorb_post_loop_scale so dk/dv stay in simdgroup
    # registers across the whole loop instead of materializing per-iter.
    dk = dk * SCALE

    col_mask = kv_mask[:, None] & (kv_bh < BH_KV)
    al.store(dKh + (k_start + rn)[:, None] * DK_SEQ_STRIDE + rd[None, :], dk, mask=col_mask)
    al.store(dVh + (k_start + rn)[:, None] * DV_SEQ_STRIDE + rd[None, :], dv, mask=col_mask)


@al.tunable(BLOCK_M=[8, 16, 32, 64], BLOCK_N=[8, 16, 32], options=dict(fuse_loops=[0, 1]))
@al.kernel
def attention_strided_backward_dkdv_masked_by_batch(
    dO,
    Q,
    K,
    V,
    LogSumExp,
    Mask,
    Delta,
    dK: al.output,
    dV: al.output,
    BH: al.constexpr = 1,
    BH_KV: al.constexpr = 1,
    HEADS_PER_BATCH: al.constexpr = 1,
    KV_HEADS_PER_BATCH: al.constexpr = 1,
    SEQ_LEN: al.constexpr = 1,
    HEAD_DIM: al.constexpr = 1,
    GO_OFFSET: al.constexpr = 0,
    GO_BATCH_STRIDE: al.constexpr = 0,
    GO_HEAD_STRIDE: al.constexpr = 0,
    GO_SEQ_STRIDE: al.constexpr = 0,
    Q_OFFSET: al.constexpr = 0,
    Q_BATCH_STRIDE: al.constexpr = 0,
    Q_HEAD_STRIDE: al.constexpr = 0,
    Q_SEQ_STRIDE: al.constexpr = 0,
    K_OFFSET: al.constexpr = 0,
    K_BATCH_STRIDE: al.constexpr = 0,
    K_HEAD_STRIDE: al.constexpr = 0,
    K_SEQ_STRIDE: al.constexpr = 0,
    V_OFFSET: al.constexpr = 0,
    V_BATCH_STRIDE: al.constexpr = 0,
    V_HEAD_STRIDE: al.constexpr = 0,
    V_SEQ_STRIDE: al.constexpr = 0,
    DK_OFFSET: al.constexpr = 0,
    DK_BATCH_STRIDE: al.constexpr = 0,
    DK_HEAD_STRIDE: al.constexpr = 0,
    DK_SEQ_STRIDE: al.constexpr = 0,
    DV_OFFSET: al.constexpr = 0,
    DV_BATCH_STRIDE: al.constexpr = 0,
    DV_HEAD_STRIDE: al.constexpr = 0,
    DV_SEQ_STRIDE: al.constexpr = 0,
    BLOCK_M: al.constexpr = 16,
    BLOCK_N: al.constexpr = 16,
    causal: al.constexpr = 0,
    KV_GROUP: al.constexpr = 1,
    CUSTOM_SCALE: al.constexpr = 0,
    KV_LEN: al.constexpr = 0,
    HIGH_PRECISION: al.constexpr = 0,
):
    D = HEAD_DIM
    N = SEQ_LEN
    N_KV = KV_LEN if KV_LEN > 0 else N
    N_Q_BLOCKS = (N + BLOCK_M - 1) // BLOCK_M
    SCALE = CUSTOM_SCALE if (CUSTOM_SCALE is not None and CUSTOM_SCALE > 0) else 1.0 / (D**0.5)

    k_block = al.program_id(0)
    kv_bh = al.program_id(1)
    batch = kv_bh // KV_HEADS_PER_BATCH
    kv_head = kv_bh - batch * KV_HEADS_PER_BATCH

    Kh = K + K_OFFSET + batch * K_BATCH_STRIDE + kv_head * K_HEAD_STRIDE
    Vh = V + V_OFFSET + batch * V_BATCH_STRIDE + kv_head * V_HEAD_STRIDE
    Mh = Mask + batch * N * N_KV
    dKh = dK + DK_OFFSET + batch * DK_BATCH_STRIDE + kv_head * DK_HEAD_STRIDE
    dVh = dV + DV_OFFSET + batch * DV_BATCH_STRIDE + kv_head * DV_HEAD_STRIDE

    k_start = k_block * BLOCK_N
    rm = al.arange(0, BLOCK_M)
    rn = al.arange(0, BLOCK_N)
    rd = al.arange(0, D)
    kv_mask = (k_start + rn) < N_KV

    k_tile = al.load(
        Kh + (k_start + rn)[:, None] * K_SEQ_STRIDE + rd[None, :],
        mask=kv_mask[:, None],
        other=0.0,
    )
    v_tile = al.load(
        Vh + (k_start + rn)[:, None] * V_SEQ_STRIDE + rd[None, :],
        mask=kv_mask[:, None],
        other=0.0,
    )
    dk = al.zeros((BLOCK_N, D), dtype=al.float32)
    dv = al.zeros((BLOCK_N, D), dtype=al.float32)
    for g in range(KV_GROUP):
        q_head = kv_head * KV_GROUP + g
        GOh = dO + GO_OFFSET + batch * GO_BATCH_STRIDE + q_head * GO_HEAD_STRIDE
        Qh = Q + Q_OFFSET + batch * Q_BATCH_STRIDE + q_head * Q_HEAD_STRIDE
        lse_off = (batch * HEADS_PER_BATCH + q_head) * N

        for _ib in range(0, N_Q_BLOCKS, 1):
            i = _ib * BLOCK_M
            row_mask = (i + rm) < N
            q = al.load(
                Qh + (i + rm)[:, None] * Q_SEQ_STRIDE + rd[None, :],
                mask=row_mask[:, None],
                other=0.0,
            )
            go = al.load(
                GOh + (i + rm)[:, None] * GO_SEQ_STRIDE + rd[None, :],
                mask=row_mask[:, None],
                other=0.0,
            )
            lse = al.load(LogSumExp + lse_off + i + rm, mask=row_mask, other=0.0)
            drow = al.load(Delta + lse_off + i + rm, mask=row_mask, other=0.0)

            s_t = al.tile_dot(k_tile, q, transpose_rhs=True) * SCALE
            mask_tile = al.load(
                Mh + (i + rm)[None, :] * N_KV + (k_start + rn)[:, None],
                mask=kv_mask[:, None] & row_mask[None, :],
                other=0.0,
            )
            s_t = al.maximum(s_t + mask_tile, -1e30)
            s_t = al.where(kv_mask[:, None], s_t, -1e30)
            s_t = al.where(row_mask[None, :], s_t, -1e30)
            if causal:
                s_t = al.where((i + rm)[None, :] >= (k_start + rn)[:, None], s_t, -1e30)
            sd_t = al.minimum(s_t - lse[None, :], 0.0)
            p_t = al.exp(sd_t)
            dp_t = al.tile_dot(v_tile, go, transpose_rhs=True)
            dv = dv + al.tile_dot(p_t, go)
            ds_t = p_t * (dp_t - drow[None, :])
            dk = dk + al.tile_dot(ds_t, q)

    # SCALE folded out of the inner accumulator. _opt_persistent_mma matches
    # the bare Add(init, Dot) body; the trailing scalar Mul gets absorbed into
    # Store.transform via _absorb_post_loop_scale so dk/dv stay in simdgroup
    # registers across the whole loop instead of materializing per-iter.
    dk = dk * SCALE

    col_mask = kv_mask[:, None] & (kv_bh < BH_KV)
    al.store(dKh + (k_start + rn)[:, None] * DK_SEQ_STRIDE + rd[None, :], dk, mask=col_mask)
    al.store(dVh + (k_start + rn)[:, None] * DV_SEQ_STRIDE + rd[None, :], dv, mask=col_mask)


@al.tunable(
    BLOCK_M=[8, 16, 32, 64],
    BLOCK_N=[8, 16, 32],
    _matvec=[0, 1],
    options=dict(fuse_loops=[0, 1]),
)
@al.kernel
def attention_kv_update(
    Q,
    new_K,
    new_V,
    cache_pos_buf,
    K_cache,
    V_cache,
    O: al.output,
    BH: al.constexpr = 1,
    HEADS_PER_BATCH: al.constexpr = 1,
    SEQ_LEN: al.constexpr = 1,
    HEAD_DIM: al.constexpr = 1,
    Q_OFFSET: al.constexpr = 0,
    Q_BATCH_STRIDE: al.constexpr = 0,
    Q_HEAD_STRIDE: al.constexpr = 0,
    Q_SEQ_STRIDE: al.constexpr = 0,
    NK_OFFSET: al.constexpr = 0,
    NK_HEAD_STRIDE: al.constexpr = 0,
    NV_OFFSET: al.constexpr = 0,
    NV_HEAD_STRIDE: al.constexpr = 0,
    KC_HEAD_STRIDE: al.constexpr = 0,
    KC_SEQ_STRIDE: al.constexpr = 0,
    VC_HEAD_STRIDE: al.constexpr = 0,
    VC_SEQ_STRIDE: al.constexpr = 0,
    BLOCK_M: al.constexpr = 16,
    BLOCK_N: al.constexpr = 32,
    causal: al.constexpr = 0,
    KV_GROUP: al.constexpr = 1,
    KV_LEN: al.constexpr = 0,
    SLIDING_WINDOW: al.constexpr = 0,
    BLOCK_SIZE: al.constexpr = 256,
    _matvec: al.constexpr = 0,
    CUSTOM_SCALE: al.constexpr = 0,
):
    """Fused KV cache update + attention.

    Identical to attention_strided_masked_by_batch, with an added prologue
    that writes new_K/new_V into K_cache/V_cache at cache_pos before attending.
    For GQA, all Q heads in a group redundantly write identical data.

    Grid: (ceil(SEQ_LEN/BLOCK_M), BH).

    When SLIDING_WINDOW > 0 the cache is a circular buffer of that physical
    size and the write position wraps via modulo (Gemma3 sliding-window).
    """
    D = HEAD_DIM
    N = SEQ_LEN
    N_KV = KV_LEN if KV_LEN > 0 else N
    SCALE = CUSTOM_SCALE if (CUSTOM_SCALE is not None and CUSTOM_SCALE > 0) else 1.0 / (D**0.5)

    q_block = al.program_id(0)
    bh = al.program_id(1)
    batch = bh // HEADS_PER_BATCH
    head = bh - batch * HEADS_PER_BATCH
    kv_head = head // KV_GROUP if KV_GROUP > 1 else head

    Qh = Q + Q_OFFSET + batch * Q_BATCH_STRIDE + head * Q_HEAD_STRIDE
    NKh = new_K + NK_OFFSET + kv_head * NK_HEAD_STRIDE
    NVh = new_V + NV_OFFSET + kv_head * NV_HEAD_STRIDE
    KCh = K_cache + kv_head * KC_HEAD_STRIDE
    VCh = V_cache + kv_head * VC_HEAD_STRIDE
    O_STRIDE = HEADS_PER_BATCH * D
    Oh = O + batch * N * O_STRIDE + head * D

    # --- KV cache write prologue ---
    cache_pos = al.load(cache_pos_buf)
    if SLIDING_WINDOW > 0:
        write_pos = cache_pos % SLIDING_WINDOW
    else:
        write_pos = cache_pos
    if causal:
        N_KV_ACTIVE = al.minimum(cache_pos + N, N_KV)
    else:
        N_KV_ACTIVE = N_KV
    # Sliding-window: restrict the K-loop to the last SLIDING_WINDOW logical
    # positions. The loop iterator stays in LOGICAL position space (so causal
    # masking is unchanged); cache reads modulo into the circular buffer.
    if SLIDING_WINDOW > 0:
        kv_loop_start = al.maximum(al.cast(0, al.int32), al.cast(cache_pos + N - SLIDING_WINDOW, al.int32))
        kv_start_block = kv_loop_start // BLOCK_N
    else:
        kv_start_block = al.cast(0, al.int32)
    N_KV_BLOCKS = (N_KV_ACTIVE + BLOCK_N - 1) // BLOCK_N
    for _di in range(0, D, BLOCK_SIZE):
        offs = _di + al.arange(0, BLOCK_SIZE)
        mask = offs < D
        k_val = al.load(NKh + offs, mask=mask, other=0.0)
        v_val = al.load(NVh + offs, mask=mask, other=0.0)
        al.store(KCh + write_pos * KC_SEQ_STRIDE + offs, k_val, mask=mask)
        al.store(VCh + write_pos * VC_SEQ_STRIDE + offs, v_val, mask=mask)
    al.barrier()

    rd = al.arange(0, D)
    rn = al.arange(0, BLOCK_N)

    if _matvec:
        q_row = q_block
        rm1 = al.arange(0, 1)
        q = al.load(
            Qh + (q_row + rm1)[:, None] * Q_SEQ_STRIDE + rd[None, :],
            mask=((q_row + rm1[:, None]) < N) & (rd[None, :] < D),
            other=0.0,
        )
        new_k_row = al.load(
            NKh + rm1[:, None] * D + rd[None, :],
            mask=rd[None, :] < D,
            other=0.0,
        )
        new_v_row = al.load(
            NVh + rm1[:, None] * D + rd[None, :],
            mask=rd[None, :] < D,
            other=0.0,
        )
        m = -1e30
        l = 0.0
        o = al.zeros((1, D), dtype=al.float32)
        for _jb in range(kv_start_block, N_KV_BLOCKS, 1):
            j = _jb * BLOCK_N
            kv_mask = (j + rn) < N_KV
            is_new_pos = (j + rn) == cache_pos
            if SLIDING_WINDOW > 0:
                kv_slot = (j + rn) % SLIDING_WINDOW
            else:
                kv_slot = j + rn
            k_tile = al.load(
                KCh + kv_slot[:, None] * KC_SEQ_STRIDE + rd[None, :],
                mask=kv_mask[:, None] & (rd[None, :] < D),
                other=0.0,
            )
            k_tile = al.where(is_new_pos[:, None], new_k_row, k_tile)
            s = al.tile_dot(q, k_tile, transpose_rhs=True) * SCALE
            s = al.where(kv_mask[None, :], s, -1e30)
            if causal:
                s = al.where((cache_pos + q_row + rm1)[:, None] >= (j + rn)[None, :], s, -1e30)
            bmax = al.max(s, axis=1)
            mn = al.maximum(m, bmax)
            alpha = al.exp(m - mn)
            l = l * alpha
            o = o * alpha
            p = al.exp(s - mn)
            l = l + al.sum(p, axis=1)
            v_tile = al.load(
                VCh + kv_slot[:, None] * VC_SEQ_STRIDE + rd[None, :],
                mask=kv_mask[:, None] & (rd[None, :] < D),
                other=0.0,
            )
            v_tile = al.where(is_new_pos[:, None], new_v_row, v_tile)
            o = o + al.tile_dot(p, v_tile)
            m = mn
        o = o * (1.0 / l)
        al.store(
            Oh + (q_row + rm1)[:, None] * O_STRIDE + rd[None, :],
            o,
            mask=((q_row + rm1[:, None]) < N) & (rd[None, :] < D) & (bh < BH),
        )
    else:
        # --- Attention (identical to attention_strided_masked_by_batch) ---
        q_start = q_block * BLOCK_M
        rm = al.arange(0, BLOCK_M)
        q = al.load(
            Qh + (q_start + rm)[:, None] * Q_SEQ_STRIDE + rd[None, :],
            mask=(q_start + rm[:, None]) < N,
            other=0.0,
        )
        m = -1e30
        l = 0.0
        o = al.zeros((BLOCK_M, D), dtype=al.float32)
        for _jb in range(kv_start_block, N_KV_BLOCKS, 1):
            j = _jb * BLOCK_N
            if SLIDING_WINDOW > 0:
                kv_slot = (j + rn) % SLIDING_WINDOW
            else:
                kv_slot = j + rn
            k_tile = al.load(
                KCh + kv_slot[:, None] * KC_SEQ_STRIDE + rd[None, :],
                mask=(j + rn[:, None]) < N_KV,
                other=0.0,
            )
            s = al.tile_dot(q, k_tile, transpose_rhs=True)
            s = s * SCALE
            # No external mask — use N_KV bound + causal position mask
            s = al.where((j + rn)[None, :] < N_KV, s, -1e30)
            if causal:
                s = al.where((cache_pos + q_start + rm)[:, None] >= (j + rn)[None, :], s, -1e30)
            bmax = al.max(s, axis=1)
            mn = al.maximum(m, bmax)
            alpha = al.exp(m - mn)
            l = l * alpha
            o = o * alpha
            p = al.exp(s - mn)
            l = l + al.sum(p, axis=1)
            v_tile = al.load(
                VCh + kv_slot[:, None] * VC_SEQ_STRIDE + rd[None, :],
                mask=(j + rn[:, None]) < N_KV,
                other=0.0,
            )
            o = o + al.tile_dot(p, v_tile)
            m = mn
        o = o * (1.0 / l)
        al.store(
            Oh + (q_start + rm)[:, None] * O_STRIDE + rd[None, :],
            o,
            mask=((q_start + rm[:, None]) < N) & (bh < BH),
        )


@al.tunable(
    BLOCK_M=[8, 16, 32, 64],
    BLOCK_N=[8, 16, 32],
    options=dict(fuse_loops=[0, 1]),
)
@al.kernel
def attention_kv_update_split(
    Q,
    new_K,
    new_V,
    cache_pos_buf,
    K_cache,
    V_cache,
    partial_O: al.output,
    partial_lse: al.output,
    BH: al.constexpr = 1,
    HEADS_PER_BATCH: al.constexpr = 1,
    HEAD_DIM: al.constexpr = 1,
    Q_OFFSET: al.constexpr = 0,
    Q_BATCH_STRIDE: al.constexpr = 0,
    Q_HEAD_STRIDE: al.constexpr = 0,
    NK_OFFSET: al.constexpr = 0,
    NK_HEAD_STRIDE: al.constexpr = 0,
    NV_OFFSET: al.constexpr = 0,
    NV_HEAD_STRIDE: al.constexpr = 0,
    KC_HEAD_STRIDE: al.constexpr = 0,
    KC_SEQ_STRIDE: al.constexpr = 0,
    VC_HEAD_STRIDE: al.constexpr = 0,
    VC_SEQ_STRIDE: al.constexpr = 0,
    BLOCK_M: al.constexpr = 8,
    BLOCK_N: al.constexpr = 16,
    KV_GROUP: al.constexpr = 1,
    SPLITS: al.constexpr = 1,
    SLIDING_WINDOW: al.constexpr = 0,
    BLOCK_SIZE: al.constexpr = 256,
    CUSTOM_SCALE: al.constexpr = 0,
):
    """Flash Decoding split-KV attention with fused KV cache update.

    Grid: (BH, SPLITS). Each TG processes one (head, kv-slice) and writes a
    partial output + lse. A companion combine kernel reduces SPLITS partials
    per head into the final output. Causal is assumed (decode).

    BLOCK_M=8 keeps the persistent_mma pattern on V@P (accumulator stays in
    simdgroup registers across the K loop) even though only row 0 is the
    real query — rows 1..BLOCK_M-1 are padding masked to 0.
    """
    D = HEAD_DIM
    SCALE = CUSTOM_SCALE if (CUSTOM_SCALE is not None and CUSTOM_SCALE > 0) else 1.0 / (D**0.5)

    bh = al.program_id(0)
    split_idx = al.program_id(1)
    batch = bh // HEADS_PER_BATCH
    head = bh - batch * HEADS_PER_BATCH
    kv_head = head // KV_GROUP if KV_GROUP > 1 else head

    Qh = Q + Q_OFFSET + batch * Q_BATCH_STRIDE + head * Q_HEAD_STRIDE
    NKh = new_K + NK_OFFSET + kv_head * NK_HEAD_STRIDE
    NVh = new_V + NV_OFFSET + kv_head * NV_HEAD_STRIDE
    KCh = K_cache + kv_head * KC_HEAD_STRIDE
    VCh = V_cache + kv_head * VC_HEAD_STRIDE

    cache_pos = al.load(cache_pos_buf)
    N_KV_ACTIVE = cache_pos + 1

    # Partition the ACTIVE blocks (ceil(fill/BLOCK_N)) across splits, not the
    # allocated cache length: sizing the chunk by the cache size concentrates a
    # fixed fill into the first fill/CHUNK splits and idles the rest when
    # fill << cache. Tracking fill keeps all SPLITS busy regardless of cache budget.
    N_ACTIVE_BLOCKS = al.cast((N_KV_ACTIVE + BLOCK_N - 1) // BLOCK_N, al.int32)
    CHUNK_BLOCKS = (N_ACTIVE_BLOCKS + SPLITS - 1) // SPLITS
    split_block_start = al.cast(split_idx, al.int32) * CHUNK_BLOCKS
    split_block_end = split_block_start + CHUNK_BLOCKS
    active_end = al.minimum(split_block_end, N_ACTIVE_BLOCKS)

    # KV cache write prologue — every (head, split) TG writes the same row
    # idempotently. Cross-TG races are safe (same data). For sliding-window
    # caches, the physical buffer is sized at SLIDING_WINDOW; modulo the
    # position so writes wrap instead of OOBing.
    if SLIDING_WINDOW > 0:
        write_pos = cache_pos % SLIDING_WINDOW
    else:
        write_pos = cache_pos
    for _di in range(0, D, BLOCK_SIZE):
        offs = _di + al.arange(0, BLOCK_SIZE)
        mask = offs < D
        k_val = al.load(NKh + offs, mask=mask, other=0.0)
        v_val = al.load(NVh + offs, mask=mask, other=0.0)
        al.store(KCh + write_pos * KC_SEQ_STRIDE + offs, k_val, mask=mask)
        al.store(VCh + write_pos * VC_SEQ_STRIDE + offs, v_val, mask=mask)
    al.barrier()

    rd = al.arange(0, D)
    rn = al.arange(0, BLOCK_N)
    rm = al.arange(0, BLOCK_M)

    q = al.load(
        Qh + rm[:, None] * 0 + rd[None, :],
        mask=(rm[:, None] < 1) & (rd[None, :] < D),
        other=0.0,
    )

    # Sliding-window: clamp the split's first block to the visible window.
    if SLIDING_WINDOW > 0:
        sw_loop_start = al.maximum(al.cast(0, al.int32), al.cast(cache_pos + 1 - SLIDING_WINDOW, al.int32))
        sw_start_block = sw_loop_start // BLOCK_N
        loop_start = al.maximum(al.cast(split_block_start, al.int32), sw_start_block)
    else:
        loop_start = al.cast(split_block_start, al.int32)

    m = -1e30
    l = 0.0
    o = al.zeros((BLOCK_M, D), dtype=al.float32)
    for _jb in range(loop_start, active_end, 1):
        j = _jb * BLOCK_N
        if SLIDING_WINDOW > 0:
            kv_slot = (j + rn) % SLIDING_WINDOW
        else:
            kv_slot = j + rn
        k_tile = al.load(
            KCh + kv_slot[:, None] * KC_SEQ_STRIDE + rd[None, :],
            mask=(j + rn[:, None]) < N_KV_ACTIVE,
            other=0.0,
        )
        s = al.tile_dot(q, k_tile, transpose_rhs=True)
        s = s * SCALE
        s = al.where((j + rn)[None, :] < N_KV_ACTIVE, s, -1e30)
        bmax = al.max(s, axis=1)
        mn = al.maximum(m, bmax)
        alpha = al.exp(m - mn)
        l = l * alpha
        o = o * alpha
        p = al.exp(s - mn)
        l = l + al.sum(p, axis=1)
        v_tile = al.load(
            VCh + kv_slot[:, None] * VC_SEQ_STRIDE + rd[None, :],
            mask=(j + rn[:, None]) < N_KV_ACTIVE,
            other=0.0,
        )
        o = o + al.tile_dot(p, v_tile)
        m = mn

    # Guard empty-split: when active_end <= split_block_start the loop doesn't
    # run and l stays 0. Without the clamp, 1/l is +inf, 0*inf is NaN, and
    # combine's `o_s * coef` propagates NaN even at coef=0 (IEEE NaN*0 = NaN).
    # Clamping l keeps the math finite; lse = m + log(l) stays sufficiently
    # negative that combine's exp(lse - m_global) is 0 and the split is ignored.
    l = al.maximum(l, 1e-30)
    o = o * (1.0 / l)

    # partial_O is (BH, SPLITS, BLOCK_M, D); partial_lse is (BH, SPLITS, BLOCK_M).
    # Each Q row gets its own (row-strided) slot so the simdgroup-matrix store
    # sees a non-zero row stride — `rm * 0` aliases all 8 MMA rows to one
    # address and clobbers the row-0 result. Combine reads slot 0 only.
    POh = partial_O + (bh * SPLITS + split_idx) * BLOCK_M * D
    PLseh = partial_lse + (bh * SPLITS + split_idx) * BLOCK_M

    al.store(
        POh + rm[:, None] * D + rd[None, :],
        o,
        mask=(rm[:, None] < BLOCK_M) & (rd[None, :] < D) & (bh < BH),
    )
    lse_tile = al.zeros((BLOCK_M, 1), dtype=al.float32)
    lse_tile = lse_tile + m + al.log(l)
    rc1 = al.arange(0, 1)
    al.store(
        PLseh + rm[:, None] * 1 + rc1[None, :],
        lse_tile,
        mask=(rm[:, None] < BLOCK_M) & (rc1[None, :] < 1) & (bh < BH),
    )


@al.kernel
def attention_decode_vector_split(
    Q,
    new_K,
    new_V,
    cache_pos_buf,
    K_cache,
    V_cache,
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
    VC_HEAD_STRIDE: al.constexpr = 0,
    VC_SEQ_STRIDE: al.constexpr = 128,
    KV_GROUP: al.constexpr = 1,
    SPLITS: al.constexpr = 1,
    SLIDING_WINDOW: al.constexpr = 0,
    CUSTOM_SCALE: al.constexpr = 0,
    WRITE_KV: al.constexpr = 1,
    UNROLL: al.constexpr = 16,
):
    """M=1 vector-path flash decoding split-KV — no MMA, no tile_dot.

    32-thread TG (1 simdgroup). Each lane owns PER_LANE=HEAD_DIM/32
    consecutive D-dims of Q and of the output. K[j] is dotted by each
    lane and the per-lane partial reduced across the simdgroup via
    `al.simd_reduce` to get the broadcast scalar score. Softmax state
    (m, l) and per-lane output slice live in registers; zero shared
    memory, zero barriers in the K-loop.

    Grid: (BH, SPLITS). Combined by `attention_decode_combine_vector`
    (one D-row + one scalar per (head, split)).

    Specialisations by HEAD_DIM (constexpr-evaluated at trace):
      D=512 → PER_LANE=16, quad-vec4 path. Gemma 4 global/full layers.
      D=256 → PER_LANE=8, dual-vec4 path. Gemma 3.
      D=128 → PER_LANE=4, fast vec4 path (load4_vec / dot4 / unpack4).
              Qwen3 0.6B, Llama 3.x 3B/8B.
      D=64  → PER_LANE=2, scalar load + manual dot. Llama 3.2 1B.
    Other HEAD_DIMs that divide evenly by 32 take the scalar path too.
    """
    D = HEAD_DIM
    SCALE = CUSTOM_SCALE if (CUSTOM_SCALE is not None and CUSTOM_SCALE > 0) else 1.0 / (D**0.5)
    SCALE2 = SCALE * 1.4426950408889634  # fold log2(e) into the QK scale → exp2 frame
    LN2 = 0.6931471805599453  # log2-frame running max → natural units for the lse
    PER_LANE = D // 32  # 4 for D=128, 2 for D=64

    bh = al.program_id(0)
    split_idx = al.program_id(1)
    batch = bh // HEADS_PER_BATCH
    head = bh - batch * HEADS_PER_BATCH
    kv_head = head // KV_GROUP if KV_GROUP > 1 else head

    Qh = Q + Q_OFFSET + batch * Q_BATCH_STRIDE + head * Q_HEAD_STRIDE
    NKh = new_K + NK_OFFSET + kv_head * NK_HEAD_STRIDE
    NVh = new_V + NV_OFFSET + kv_head * NV_HEAD_STRIDE
    KCh = K_cache + kv_head * KC_HEAD_STRIDE
    VCh = V_cache + kv_head * VC_HEAD_STRIDE

    cache_pos = al.load(cache_pos_buf)
    if SLIDING_WINDOW > 0:
        write_pos = cache_pos % SLIDING_WINDOW
    else:
        write_pos = cache_pos
    # Attend range [loop_lo, cache_pos + 1) — the true logical end. The K-loop
    # walks LOGICAL positions; the circular sliding cache reads slot
    # = j % SLIDING_WINDOW, so the upper bound is the logical end, never the
    # physical buffer size. Full attention reads slot = j directly: the cache is
    # always allocated long enough to hold every position decode reaches, so
    # cache_pos+1 needs no cap against the total length. Sliding window
    # additionally evicts everything older than the window via loop_lo.
    if SLIDING_WINDOW > 0:
        loop_lo = al.maximum(al.cast(0, al.int32), al.cast(cache_pos + 1 - SLIDING_WINDOW, al.int32))
    else:
        loop_lo = al.cast(0, al.int32)
    N_KV_ACTIVE = al.cast(cache_pos + 1, al.int32)

    lane = al.arange(0, 32)
    lane_off = lane * PER_LANE

    # Per-split K range. Partition the ACTIVE WINDOW [loop_lo, N_KV_ACTIVE) evenly
    # across splits — NOT [0, N_KV_ACTIVE): once the window has slid off zero
    # (cache_pos >= SW) the leading [0, loop_lo) positions are evicted, and
    # partitioning from 0 would pile the live window into the last split(s). Sizing
    # CHUNK by the live span keeps all SPLITS busy regardless of cache allocation.
    # Empty splits (split_start >= N_KV_ACTIVE) write a -inf-lse partial.
    SPAN = al.cast(N_KV_ACTIVE, al.int32) - loop_lo
    CHUNK = (SPAN + SPLITS - 1) // SPLITS
    split_start = loop_lo + al.cast(split_idx, al.int32) * CHUNK
    split_end = al.minimum(split_start + CHUNK, al.cast(N_KV_ACTIVE, al.int32))

    if PER_LANE % 4 == 0:
        # ---- vec4 path: HEAD_DIM 128/256/512 (PER_LANE 4/8/16) in ONE body ----
        # Each lane owns PER_LANE output dims = NVEC contiguous vec4 chunks. The
        # accumulator array `o` and Q chunks are carried/iterated with al.unroll
        # instead of being hand-flattened to o0..oN per head-dim. The K-loop is
        # unrolled by UNROLL positions so the UNROLL independent per-position
        # simd_reduce()s pipeline through the shuffle unit (decode attention is
        # reduction-latency-bound at low occupancy) — then UNROLL sequential
        # softmax updates; a scalar tail covers the remainder.
        NVEC = PER_LANE // 4
        if WRITE_KV:
            for _i in al.unroll(range(PER_LANE)):
                al.store(KCh + write_pos * KC_SEQ_STRIDE + lane_off + _i, al.load(NKh + lane_off + _i))
                al.store(VCh + write_pos * VC_SEQ_STRIDE + lane_off + _i, al.load(NVh + lane_off + _i))
        q = [al.load4_vec(Qh + lane_off + 4 * _c) for _c in range(NVEC)]
        o = [0.0] * PER_LANE
        m = -1e30
        l = 0.0
        T_MAIN = split_start + ((split_end - split_start) // UNROLL) * UNROLL
        for jb in range(split_start, T_MAIN, UNROLL):
            sc = [0.0] * UNROLL
            for _t in al.unroll(range(UNROLL)):
                if SLIDING_WINDOW > 0:
                    jslot = (jb + _t) % SLIDING_WINDOW
                else:
                    jslot = jb + _t
                partial = 0.0
                for _c in al.unroll(range(NVEC)):
                    partial = partial + al.dot4(q[_c], al.load4_vec(KCh + jslot * KC_SEQ_STRIDE + lane_off + 4 * _c))
                sc[_t] = al.simd_reduce(partial) * SCALE2
            # FlashAttention-2 block softmax: take the block max over all UNROLL
            # scores, rescale (l, o) into that frame ONCE, then add every
            # contribution in it. Cuts exps from 2*UNROLL to UNROLL+1 and the
            # o-rescales from PER_LANE*UNROLL to PER_LANE; native exp2 (scores are
            # pre-scaled by log2e) over exp.
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
                for _c in al.unroll(range(NVEC)):
                    v = al.load4_vec(VCh + jslot * VC_SEQ_STRIDE + lane_off + 4 * _c)
                    for _k in al.unroll(range(4)):
                        o[4 * _c + _k] = o[4 * _c + _k] + p * al.unpack4(v, _k)
            m = block_max
        for j in range(T_MAIN, split_end, 1):
            if SLIDING_WINDOW > 0:
                jslot = j % SLIDING_WINDOW
            else:
                jslot = j
            partial = 0.0
            for _c in al.unroll(range(NVEC)):
                partial = partial + al.dot4(q[_c], al.load4_vec(KCh + jslot * KC_SEQ_STRIDE + lane_off + 4 * _c))
            score = al.simd_reduce(partial) * SCALE2
            block_max = al.maximum(m, score)
            rescale = al.exp2(m - block_max)
            p = al.exp2(score - block_max)
            l = l * rescale + p
            for _c in al.unroll(range(NVEC)):
                v = al.load4_vec(VCh + jslot * VC_SEQ_STRIDE + lane_off + 4 * _c)
                for _k in al.unroll(range(4)):
                    o[4 * _c + _k] = o[4 * _c + _k] * rescale + p * al.unpack4(v, _k)
            m = block_max
        l_safe = al.maximum(l, 1e-30)
        POh = partial_O + (bh * SPLITS + split_idx) * D
        for _i in al.unroll(range(PER_LANE)):
            al.store(POh + lane_off + _i, o[_i] / l_safe, mask=bh < BH)
        PLseh = partial_lse + (bh * SPLITS + split_idx)
        al.store(PLseh, m * LN2 + al.log(al.maximum(l, 1e-30)), mask=(bh < BH) & (lane < 1))
    else:
        # ---- PER_LANE=2 scalar path (HEAD_DIM=64, Llama 3.2 1B) ----
        # Named scalar carries (q0/q1, k0/k1, v0/v1, o0/o1) so the trace
        # tracks each as a loop carry. PER_LANE=2 means each lane covers
        # 2 contiguous elements via 2 scalar loads — Metal coalesces them
        # into a single float2 op per lane × 32 lanes = one cache line per
        # row, same coalescing as the vec4 path uses for D=128.
        nk0 = al.load(NKh + lane_off + 0)
        nk1 = al.load(NKh + lane_off + 1)
        nv0 = al.load(NVh + lane_off + 0)
        nv1 = al.load(NVh + lane_off + 1)
        al.store(KCh + write_pos * KC_SEQ_STRIDE + lane_off + 0, nk0)
        al.store(KCh + write_pos * KC_SEQ_STRIDE + lane_off + 1, nk1)
        al.store(VCh + write_pos * VC_SEQ_STRIDE + lane_off + 0, nv0)
        al.store(VCh + write_pos * VC_SEQ_STRIDE + lane_off + 1, nv1)

        q0 = al.load(Qh + lane_off + 0)
        q1 = al.load(Qh + lane_off + 1)

        m = -1e30
        l = 0.0
        o0 = 0.0
        o1 = 0.0

        for j in range(split_start, split_end, 1):
            if SLIDING_WINDOW > 0:
                j_slot = j % SLIDING_WINDOW
            else:
                j_slot = j
            k0 = al.load(KCh + j_slot * KC_SEQ_STRIDE + lane_off + 0)
            k1 = al.load(KCh + j_slot * KC_SEQ_STRIDE + lane_off + 1)
            partial = q0 * k0 + q1 * k1
            score = al.simd_reduce(partial) * SCALE

            mn = al.maximum(m, score)
            alpha = al.exp(m - mn)
            p = al.exp(score - mn)
            l = l * alpha + p
            o0 = o0 * alpha
            o1 = o1 * alpha
            m = mn

            v0 = al.load(VCh + j_slot * VC_SEQ_STRIDE + lane_off + 0)
            v1 = al.load(VCh + j_slot * VC_SEQ_STRIDE + lane_off + 1)
            o0 = o0 + p * v0
            o1 = o1 + p * v1

        l_safe = al.maximum(l, 1e-30)
        o0 = o0 / l_safe
        o1 = o1 / l_safe

        POh = partial_O + (bh * SPLITS + split_idx) * D
        al.store(POh + lane_off + 0, o0, mask=bh < BH)
        al.store(POh + lane_off + 1, o1, mask=bh < BH)

        PLseh = partial_lse + (bh * SPLITS + split_idx)
        al.store(
            PLseh, m + al.log(al.maximum(l, 1e-30)), mask=(bh < BH) & (lane < 1)
        )



@al.kernel
def attention_decode_combine_vector(
    partial_O,
    partial_lse,
    O: al.output,
    BH: al.constexpr = 1,
    HEADS_PER_BATCH: al.constexpr = 1,
    HEAD_DIM: al.constexpr = 128,
    SPLITS: al.constexpr = 1,
):
    """Combine for `attention_decode_vector_split` — partial_O is (BH, SPLITS, D),
    partial_lse is (BH, SPLITS). One D-row + one scalar per (head, split)."""
    D = HEAD_DIM
    bh = al.program_id(0)
    batch = bh // HEADS_PER_BATCH
    head = bh - batch * HEADS_PER_BATCH

    rd = al.arange(0, D)

    m_global = -1e30
    sum_weights = 0.0
    for s in range(SPLITS):
        lse_s = al.load(partial_lse + (bh * SPLITS + s))
        new_m = al.maximum(m_global, lse_s)
        alpha = al.exp(m_global - new_m)
        sum_weights = sum_weights * alpha + al.exp(lse_s - new_m)
        m_global = new_m

    inv_sum = 1.0 / sum_weights

    o_accum = al.zeros((D,), dtype=al.float32)
    for s in range(SPLITS):
        lse_s = al.load(partial_lse + (bh * SPLITS + s))
        coef = al.exp(lse_s - m_global) * inv_sum
        o_s = al.load(
            partial_O + (bh * SPLITS + s) * D + rd,
            mask=rd < D,
            other=0.0,
        )
        o_accum = o_accum + o_s * coef

    O_STRIDE = HEADS_PER_BATCH * D
    Oh = O + batch * O_STRIDE + head * D
    al.store(Oh + rd, o_accum, mask=(rd < D) & (bh < BH))


@al.kernel
def attention_kv_update_vector_split_multi(
    Q,
    new_K,
    new_V,
    cache_pos_buf,
    K_cache,
    V_cache,
    partial_O: al.output,
    partial_lse: al.output,
    BH: al.constexpr = 1,
    HEADS_PER_BATCH: al.constexpr = 1,
    HEAD_DIM: al.constexpr = 256,
    Q_OFFSET: al.constexpr = 0,
    Q_BATCH_STRIDE: al.constexpr = 0,
    Q_HEAD_STRIDE: al.constexpr = 0,
    Q_SEQ_STRIDE: al.constexpr = 0,
    NK_OFFSET: al.constexpr = 0,
    NK_HEAD_STRIDE: al.constexpr = 0,
    NK_SEQ_STRIDE: al.constexpr = 0,
    NV_OFFSET: al.constexpr = 0,
    NV_HEAD_STRIDE: al.constexpr = 0,
    NV_SEQ_STRIDE: al.constexpr = 0,
    KC_HEAD_STRIDE: al.constexpr = 0,
    KC_SEQ_STRIDE: al.constexpr = 256,
    VC_HEAD_STRIDE: al.constexpr = 0,
    VC_SEQ_STRIDE: al.constexpr = 256,
    KV_GROUP: al.constexpr = 1,
    K_INPUT: al.constexpr = 2,
    SPLITS: al.constexpr = 1,
    SLIDING_WINDOW: al.constexpr = 0,
    CUSTOM_SCALE: al.constexpr = 0,
    UNROLL: al.constexpr = 8,
    BIDIR_BLOCK: al.constexpr = 0,
):
    """Vector-path multi-token flash decoding with fused KV update — the
    K_INPUT > 1 spec-verify counterpart of `attention_decode_vector_split`.

    BIDIR_BLOCK=1 lifts the per-row causal bound to the whole new-token block
    (every row attends [0, cache_pos + K_INPUT)): the DFlash draft's block
    attention — mask tokens attend bidirectionally within the block and fully to
    the context KV.

    Same 1-simdgroup (32-thread) vec4 path as the M=1 vector decode (PER_LANE =
    HEAD_DIM/32 dims per lane, one `simd_reduce` per score, exp2 softmax frame),
    generalized to K_INPUT query rows: each lane carries K_INPUT Q slices and
    K_INPUT (m, l, o) softmax states, so K and V are read ONCE per position and
    reused across rows. Query row i sits at absolute position cache_pos + i and
    attends causally to [0, cache_pos + i]. Requires HEAD_DIM in {128, 256, 512}
    (PER_LANE 4/8/16, vec4 path); HEAD_DIM 64 stays on the MMA path.

    partial_O: (BH, SPLITS, K_INPUT, HEAD_DIM); partial_lse: (BH, SPLITS, K_INPUT).
    """
    D = HEAD_DIM
    SCALE = CUSTOM_SCALE if (CUSTOM_SCALE is not None and CUSTOM_SCALE > 0) else 1.0 / (D**0.5)
    SCALE2 = SCALE * 1.4426950408889634  # log2(e): fold into the QK scale → exp2 frame
    LN2 = 0.6931471805599453  # log2-frame max → natural-unit lse for the combine
    PER_LANE = D // 32
    NVEC = PER_LANE // 4

    bh = al.program_id(0)
    split_idx = al.program_id(1)
    batch = bh // HEADS_PER_BATCH
    head = bh - batch * HEADS_PER_BATCH
    kv_head = head // KV_GROUP if KV_GROUP > 1 else head

    Qbh = Q + Q_OFFSET + batch * Q_BATCH_STRIDE + head * Q_HEAD_STRIDE
    NKh = new_K + NK_OFFSET + kv_head * NK_HEAD_STRIDE
    NVh = new_V + NV_OFFSET + kv_head * NV_HEAD_STRIDE
    KCh = K_cache + kv_head * KC_HEAD_STRIDE
    VCh = V_cache + kv_head * VC_HEAD_STRIDE

    cache_pos = al.cast(al.load(cache_pos_buf), al.int32)
    N_KV_ACTIVE = cache_pos + K_INPUT  # max over rows (row K_INPUT-1 attends here)
    if SLIDING_WINDOW > 0:
        loop_lo = al.maximum(al.cast(0, al.int32), al.cast(cache_pos + K_INPUT - SLIDING_WINDOW, al.int32))
    else:
        loop_lo = al.cast(0, al.int32)
    SPAN = N_KV_ACTIVE - loop_lo
    CHUNK = (SPAN + SPLITS - 1) // SPLITS
    split_start = loop_lo + al.cast(split_idx, al.int32) * CHUNK
    split_end = al.minimum(split_start + CHUNK, N_KV_ACTIVE)

    lane = al.arange(0, 32)
    lane_off = lane * PER_LANE

    # KV write prologue: K_INPUT new tokens at [cache_pos, cache_pos + K_INPUT).
    for _r in al.unroll(range(K_INPUT)):
        if SLIDING_WINDOW > 0:
            wpos = (cache_pos + _r) % SLIDING_WINDOW
        else:
            wpos = cache_pos + _r
        for _i in al.unroll(range(PER_LANE)):
            al.store(KCh + wpos * KC_SEQ_STRIDE + lane_off + _i, al.load(NKh + _r * NK_SEQ_STRIDE + lane_off + _i))
            al.store(VCh + wpos * VC_SEQ_STRIDE + lane_off + _i, al.load(NVh + _r * NV_SEQ_STRIDE + lane_off + _i))

    # Per-row Q (NVEC vec4 chunks per lane) + per-row softmax state.
    q = [[al.load4_vec(Qbh + _r * Q_SEQ_STRIDE + lane_off + 4 * _c) for _c in range(NVEC)] for _r in range(K_INPUT)]
    o = [[0.0] * PER_LANE for _ in range(K_INPUT)]
    m = [-1e30 for _ in range(K_INPUT)]
    l = [0.0 for _ in range(K_INPUT)]

    T_MAIN = split_start + ((split_end - split_start) // UNROLL) * UNROLL
    for jb in range(split_start, T_MAIN, UNROLL):
        # Scores: K loaded ONCE per position, dotted by every row. K_INPUT*UNROLL
        # independent simd_reduces pipeline through the shuffle unit.
        sc = [[0.0] * UNROLL for _ in range(K_INPUT)]
        for _t in al.unroll(range(UNROLL)):
            if SLIDING_WINDOW > 0:
                jslot = (jb + _t) % SLIDING_WINDOW
            else:
                jslot = jb + _t
            kk = [al.load4_vec(KCh + jslot * KC_SEQ_STRIDE + lane_off + 4 * _c) for _c in range(NVEC)]
            for _r in al.unroll(range(K_INPUT)):
                partial = 0.0
                for _c in al.unroll(range(NVEC)):
                    partial = partial + al.dot4(q[_r][_c], kk[_c])
                s = al.simd_reduce(partial) * SCALE2
                # Causal: row _r (position cache_pos+_r) only sees kv ≤ cache_pos+_r.
                # BIDIR_BLOCK: every row sees the whole block (folds at compile).
                _row_bound = (K_INPUT - 1) if BIDIR_BLOCK else _r
                sc[_r][_t] = al.where((jb + _t) <= (cache_pos + _row_bound), s, -1e30)
        # Per-row block softmax: rescale carried (l, o) into the new running max.
        for _r in al.unroll(range(K_INPUT)):
            block_max = m[_r]
            for _t in al.unroll(range(UNROLL)):
                block_max = al.maximum(block_max, sc[_r][_t])
            rescale = al.exp2(m[_r] - block_max)
            l[_r] = l[_r] * rescale
            for _i in al.unroll(range(PER_LANE)):
                o[_r][_i] = o[_r][_i] * rescale
            m[_r] = block_max
        # V loaded ONCE per position, accumulated into every row's output.
        for _t in al.unroll(range(UNROLL)):
            if SLIDING_WINDOW > 0:
                jslot = (jb + _t) % SLIDING_WINDOW
            else:
                jslot = jb + _t
            vv = [al.load4_vec(VCh + jslot * VC_SEQ_STRIDE + lane_off + 4 * _c) for _c in range(NVEC)]
            for _r in al.unroll(range(K_INPUT)):
                p = al.exp2(sc[_r][_t] - m[_r])
                l[_r] = l[_r] + p
                for _c in al.unroll(range(NVEC)):
                    for _k in al.unroll(range(4)):
                        o[_r][4 * _c + _k] = o[_r][4 * _c + _k] + p * al.unpack4(vv[_c], _k)

    # Scalar tail (the < UNROLL remainder positions).
    for j in range(T_MAIN, split_end, 1):
        if SLIDING_WINDOW > 0:
            jslot = j % SLIDING_WINDOW
        else:
            jslot = j
        kk = [al.load4_vec(KCh + jslot * KC_SEQ_STRIDE + lane_off + 4 * _c) for _c in range(NVEC)]
        vv = [al.load4_vec(VCh + jslot * VC_SEQ_STRIDE + lane_off + 4 * _c) for _c in range(NVEC)]
        for _r in al.unroll(range(K_INPUT)):
            partial = 0.0
            for _c in al.unroll(range(NVEC)):
                partial = partial + al.dot4(q[_r][_c], kk[_c])
            _row_bound = (K_INPUT - 1) if BIDIR_BLOCK else _r
            score = al.where(j <= (cache_pos + _row_bound), al.simd_reduce(partial) * SCALE2, -1e30)
            block_max = al.maximum(m[_r], score)
            rescale = al.exp2(m[_r] - block_max)
            p = al.exp2(score - block_max)
            l[_r] = l[_r] * rescale + p
            for _c in al.unroll(range(NVEC)):
                for _k in al.unroll(range(4)):
                    o[_r][4 * _c + _k] = o[_r][4 * _c + _k] * rescale + p * al.unpack4(vv[_c], _k)
            m[_r] = block_max

    # Per-row partials → ROW-MAJOR (BH, K_INPUT, SPLITS, D) / (BH, K_INPUT, SPLITS):
    # for a fixed (bh, row) the split loop strides by exactly D, so the combine
    # (attention_decode_combine_vector_multi) threads by HEAD_DIM like the M=1
    # vector combine — NOT by the partial's row factor (the split-major layout the
    # MMA combine uses mis-threads when the row count < HEAD_DIM/32). Empty splits
    # write a -inf-frame lse the combine's sentinel guard discards.
    for _r in al.unroll(range(K_INPUT)):
        l_safe = al.maximum(l[_r], 1e-30)
        POh = partial_O + ((bh * K_INPUT + _r) * SPLITS + split_idx) * D
        for _i in al.unroll(range(PER_LANE)):
            al.store(POh + lane_off + _i, o[_r][_i] / l_safe, mask=bh < BH)
        PLseh = partial_lse + (bh * K_INPUT + _r) * SPLITS + split_idx
        al.store(PLseh, m[_r] * LN2 + al.log(al.maximum(l[_r], 1e-30)), mask=(bh < BH) & (lane < 1))


@al.kernel
def attention_decode_combine_vector_multi(
    partial_O,
    partial_lse,
    O: al.output,
    BH: al.constexpr = 1,
    HEADS_PER_BATCH: al.constexpr = 1,
    HEAD_DIM: al.constexpr = 256,
    SPLITS: al.constexpr = 1,
    K_INPUT: al.constexpr = 2,
):
    """Combine for `attention_kv_update_vector_split_multi`. partial_O is ROW-MAJOR
    (BH, K_INPUT, SPLITS, D), partial_lse (BH, K_INPUT, SPLITS): for a fixed
    (head, row) the split loop strides by exactly D, so this threads by HEAD_DIM
    (one D-row + one scalar per (head, row, split)) — the M=1 vector combine
    (`attention_decode_combine_vector`) generalized to K_INPUT query rows via the
    grid. Grid: (BH, K_INPUT).

    NB: the split-major MMA combine (`attention_decode_combine_multi`) mis-threads
    this — its `s*BLOCK_M*D` partial stride makes the planner pick BLOCK_M*32
    threads, under-covering the D output whenever the row count < HEAD_DIM/32.
    The row-major layout here keeps the per-row split stride at D, avoiding that.
    """
    D = HEAD_DIM
    bh = al.program_id(0)
    q_idx = al.program_id(1)
    batch = bh // HEADS_PER_BATCH
    head = bh - batch * HEADS_PER_BATCH
    base = (bh * K_INPUT + q_idx) * SPLITS  # this (head, row)'s first split slot

    rd = al.arange(0, D)

    m_global = -1e30
    sum_weights = 0.0
    for s in range(SPLITS):
        lse_s = al.load(partial_lse + base + s)
        new_m = al.maximum(m_global, lse_s)
        alpha = al.exp(m_global - new_m)
        sum_weights = sum_weights * alpha + al.exp(lse_s - new_m)
        m_global = new_m

    inv_sum = 1.0 / sum_weights

    o_accum = al.zeros((D,), dtype=al.float32)
    for s in range(SPLITS):
        lse_s = al.load(partial_lse + base + s)
        coef = al.exp(lse_s - m_global) * inv_sum
        o_s = al.load(partial_O + (base + s) * D + rd, mask=rd < D, other=0.0)
        # Empty/fully-masked split: -inf-frame sentinel lse + uninitialized
        # partial_O. Select to 0 (a select, not a multiply, so the garbage never
        # propagates) — same guard as attention_decode_combine_multi.
        o_s = al.where(lse_s > -1e29, o_s, 0.0)
        o_accum = o_accum + o_s * coef

    O_SEQ_STRIDE = HEADS_PER_BATCH * D
    O_BATCH_STRIDE = K_INPUT * HEADS_PER_BATCH * D
    Oh = O + batch * O_BATCH_STRIDE + q_idx * O_SEQ_STRIDE + head * D
    al.store(Oh + rd, o_accum, mask=(rd < D) & (bh < BH))


@al.kernel
def attention_decode_combine_vector_par(
    partial_O,
    partial_lse,
    O: al.output,
    BH: al.constexpr = 1,
    HEADS_PER_BATCH: al.constexpr = 1,
    HEAD_DIM: al.constexpr = 128,
    SPLITS: al.constexpr = 1,
):
    """Split-parallel combine for `attention_decode_vector_split`.

    Spreads the reduction over a (BH, HEAD_DIM/4) grid: each 32-lane TG owns ONE
    head and 4 output dims, the 32 lanes split the SPLITS reduction (each does
    SPLITS/32), and a simd_reduce finishes it. Fills the machine and barely grows
    with SPLITS (vs the serial one-TG-per-head combine). Grid: (BH, HEAD_DIM // 4).
    Requires SPLITS a multiple of 32 (the flash decoder picks a power-of-two ≥ 32).
    """
    D = HEAD_DIM
    bh = al.program_id(0)
    tile = al.program_id(1)
    batch = bh // HEADS_PER_BATCH
    head = bh - batch * HEADS_PER_BATCH
    tile_base = tile * 4
    lane = al.arange(0, 32)

    # Pass 1: flash-merge the per-split lse across the 32 lanes (each lane folds
    # its SPLITS/32 splits, then a cross-lane max + rescaled sum).
    local_m = -1e30
    local_sum = 0.0
    for si in al.unroll(range(0, SPLITS, 32)):
        s = si + lane
        lse = al.load(partial_lse + bh * SPLITS + s)
        nm = al.maximum(local_m, lse)
        local_sum = local_sum * al.exp(local_m - nm) + al.exp(lse - nm)
        local_m = nm
    m_global = al.simd_reduce(local_m, op="max")
    sum_weights = al.simd_reduce(local_sum * al.exp(local_m - m_global), op="sum")
    inv_sum = 1.0 / al.maximum(sum_weights, 1e-30)

    # Pass 2: each lane weight-sums its SPLITS/32 splits over the 4 tile dims,
    # then a simd_reduce per dim finishes the cross-lane sum.
    o0 = 0.0
    o1 = 0.0
    o2 = 0.0
    o3 = 0.0
    for si in al.unroll(range(0, SPLITS, 32)):
        s = si + lane
        lse = al.load(partial_lse + bh * SPLITS + s)
        coef = al.exp(lse - m_global) * inv_sum
        po = al.load4_vec(partial_O + (bh * SPLITS + s) * D + tile_base)
        o0 = o0 + coef * al.unpack4(po, 0)
        o1 = o1 + coef * al.unpack4(po, 1)
        o2 = o2 + coef * al.unpack4(po, 2)
        o3 = o3 + coef * al.unpack4(po, 3)
    go0 = al.simd_reduce(o0, op="sum")
    go1 = al.simd_reduce(o1, op="sum")
    go2 = al.simd_reduce(o2, op="sum")
    go3 = al.simd_reduce(o3, op="sum")

    O_STRIDE = HEADS_PER_BATCH * D
    Oh = O + batch * O_STRIDE + head * D + tile_base
    al.store(Oh + 0, go0, mask=(bh < BH) & (lane < 1))
    al.store(Oh + 1, go1, mask=(bh < BH) & (lane < 1))
    al.store(Oh + 2, go2, mask=(bh < BH) & (lane < 1))
    al.store(Oh + 3, go3, mask=(bh < BH) & (lane < 1))


@al.kernel
def attention_decode_combine(
    partial_O,
    partial_lse,
    O: al.output,
    BH: al.constexpr = 1,
    HEADS_PER_BATCH: al.constexpr = 1,
    HEAD_DIM: al.constexpr = 1,
    SPLITS: al.constexpr = 1,
    BLOCK_M: al.constexpr = 8,
):
    """Combine SPLITS partial attention outputs per (bh) into final O.

    Grid: (BH,). partial_O is (BH, SPLITS, BLOCK_M, D); reads slot 0 only.
    partial_lse is (BH, SPLITS, BLOCK_M); reads slot 0 only.

    Two passes: pass 1 builds m_global + sum_weights via online merge;
    pass 2 accumulates each split's pre-normalized partial weighted by
    exp(lse_s - m_global) / sum_weights. Uses 1D tile loads — the 2D
    cooperative load pattern drops the coef multiplier in pass 2.
    """
    D = HEAD_DIM
    bh = al.program_id(0)
    batch = bh // HEADS_PER_BATCH
    head = bh - batch * HEADS_PER_BATCH

    rd = al.arange(0, D)

    m_global = -1e30
    sum_weights = 0.0
    for s in range(SPLITS):
        lse_s = al.load(partial_lse + (bh * SPLITS + s) * BLOCK_M)
        new_m = al.maximum(m_global, lse_s)
        alpha = al.exp(m_global - new_m)
        sum_weights = sum_weights * alpha + al.exp(lse_s - new_m)
        m_global = new_m

    inv_sum = 1.0 / sum_weights

    o_accum = al.zeros((D,), dtype=al.float32)
    for s in range(SPLITS):
        lse_s = al.load(partial_lse + (bh * SPLITS + s) * BLOCK_M)
        coef = al.exp(lse_s - m_global) * inv_sum
        o_s = al.load(
            partial_O + (bh * SPLITS + s) * BLOCK_M * D + rd,
            mask=rd < D,
            other=0.0,
        )
        o_accum = o_accum + o_s * coef

    O_STRIDE = HEADS_PER_BATCH * D
    Oh = O + batch * O_STRIDE + head * D
    al.store(
        Oh + rd,
        o_accum,
        mask=(rd < D) & (bh < BH),
    )


@al.tunable(
    BLOCK_N=[8, 16, 32],
    options=dict(fuse_loops=[0, 1]),
)
@al.kernel
def attention_kv_update_split_multi(
    Q,
    new_K,
    new_V,
    cache_pos_buf,
    K_cache,
    V_cache,
    partial_O: al.output,
    partial_lse: al.output,
    BH: al.constexpr = 1,
    HEADS_PER_BATCH: al.constexpr = 1,
    HEAD_DIM: al.constexpr = 1,
    Q_OFFSET: al.constexpr = 0,
    Q_BATCH_STRIDE: al.constexpr = 0,
    Q_HEAD_STRIDE: al.constexpr = 0,
    Q_SEQ_STRIDE: al.constexpr = 0,
    NK_OFFSET: al.constexpr = 0,
    NK_HEAD_STRIDE: al.constexpr = 0,
    NK_SEQ_STRIDE: al.constexpr = 0,
    NV_OFFSET: al.constexpr = 0,
    NV_HEAD_STRIDE: al.constexpr = 0,
    NV_SEQ_STRIDE: al.constexpr = 0,
    KC_HEAD_STRIDE: al.constexpr = 0,
    KC_SEQ_STRIDE: al.constexpr = 0,
    VC_HEAD_STRIDE: al.constexpr = 0,
    VC_SEQ_STRIDE: al.constexpr = 0,
    KV_GROUP: al.constexpr = 1,
    K_INPUT: al.constexpr = 1,
    BLOCK_M: al.constexpr = 8,
    SPLITS: al.constexpr = 1,
    BLOCK_N: al.constexpr = 8,
    SLIDING_WINDOW: al.constexpr = 0,
    BLOCK_SIZE: al.constexpr = 256,
    CUSTOM_SCALE: al.constexpr = 0,
):
    """Multi-token Flash Decoding split-KV attention with fused KV update.

    Generalizes attention_kv_update_split to K_INPUT > 1 query tokens (mid-
    decode multi-token forward, e.g. speculative-decode verify pass).
    BLOCK_M must be >= K_INPUT (rows >= K_INPUT are padding, masked out).

    Each TG handles one (head, kv-slice). Writes K_INPUT new K/V values to
    cache at positions [cache_pos, cache_pos + K_INPUT). Runs flash attention
    on a (BLOCK_M, BLOCK_N) tile where each row i corresponds to query at
    absolute position cache_pos + i; causal mask makes row i see only kv
    positions <= cache_pos + i.

    partial_O: (BH, SPLITS, BLOCK_M, HEAD_DIM)
    partial_lse: (BH, SPLITS, BLOCK_M)
    """
    D = HEAD_DIM
    SCALE = CUSTOM_SCALE if (CUSTOM_SCALE is not None and CUSTOM_SCALE > 0) else 1.0 / (D**0.5)

    bh = al.program_id(0)
    split_idx = al.program_id(1)
    batch = bh // HEADS_PER_BATCH
    head = bh - batch * HEADS_PER_BATCH
    kv_head = head // KV_GROUP if KV_GROUP > 1 else head

    Qh = Q + Q_OFFSET + batch * Q_BATCH_STRIDE + head * Q_HEAD_STRIDE
    NKh = new_K + NK_OFFSET + kv_head * NK_HEAD_STRIDE
    NVh = new_V + NV_OFFSET + kv_head * NV_HEAD_STRIDE
    KCh = K_cache + kv_head * KC_HEAD_STRIDE
    VCh = V_cache + kv_head * VC_HEAD_STRIDE

    cache_pos = al.cast(al.load(cache_pos_buf), al.int32)

    # KV cache write prologue. Modulo for sliding-window caches so writes
    # beyond the window wrap rather than OOB the (sliding_window-sized) buffer.
    for q_idx in range(K_INPUT):
        if SLIDING_WINDOW > 0:
            write_pos = (cache_pos + q_idx) % SLIDING_WINDOW
        else:
            write_pos = cache_pos + q_idx
        for _di in range(0, D, BLOCK_SIZE):
            offs = _di + al.arange(0, BLOCK_SIZE)
            mask = offs < D
            k_val = al.load(NKh + q_idx * NK_SEQ_STRIDE + offs, mask=mask, other=0.0)
            v_val = al.load(NVh + q_idx * NV_SEQ_STRIDE + offs, mask=mask, other=0.0)
            al.store(KCh + write_pos * KC_SEQ_STRIDE + offs, k_val, mask=mask)
            al.store(VCh + write_pos * VC_SEQ_STRIDE + offs, v_val, mask=mask)
    al.barrier()

    rd = al.arange(0, D)
    rn = al.arange(0, BLOCK_N)
    rm = al.arange(0, BLOCK_M)

    # Active KV range across the whole K_INPUT chunk: positions [0, cache_pos + K_INPUT).
    N_KV_ACTIVE = cache_pos + K_INPUT
    N_ACTIVE_BLOCKS = al.cast((N_KV_ACTIVE + BLOCK_N - 1) // BLOCK_N, al.int32)

    # Partition active blocks across splits (see attention_kv_update_split) so a
    # fixed verify-window fill keeps all SPLITS busy independent of the cache
    # budget, rather than concentrating the fill into the first fill/CHUNK splits.
    CHUNK_BLOCKS = (N_ACTIVE_BLOCKS + SPLITS - 1) // SPLITS
    split_block_start = al.cast(split_idx, al.int32) * CHUNK_BLOCKS
    split_block_end = split_block_start + CHUNK_BLOCKS
    active_end = al.minimum(split_block_end, N_ACTIVE_BLOCKS)
    # Sliding-window: clamp loop start to the visible window for the earliest
    # Q (logical position cache_pos): see only kv positions ≥ cache_pos - SW + 1.
    if SLIDING_WINDOW > 0:
        sw_loop_start = al.maximum(al.cast(0, al.int32), al.cast(cache_pos + K_INPUT - SLIDING_WINDOW, al.int32))
        sw_start_block = sw_loop_start // BLOCK_N
        loop_start = al.maximum(al.cast(split_block_start, al.int32), sw_start_block)
    else:
        loop_start = al.cast(split_block_start, al.int32)

    # Load K_INPUT queries as a (BLOCK_M, D) tile; rows >= K_INPUT are zeros.
    q = al.load(
        Qh + rm[:, None] * Q_SEQ_STRIDE + rd[None, :],
        mask=(rm[:, None] < K_INPUT) & (rd[None, :] < D),
        other=0.0,
    )

    m = -1e30
    l = 0.0
    o = al.zeros((BLOCK_M, D), dtype=al.float32)
    for _jb in range(loop_start, active_end, 1):
        j = _jb * BLOCK_N
        if SLIDING_WINDOW > 0:
            kv_slot = (j + rn) % SLIDING_WINDOW
        else:
            kv_slot = j + rn
        k_tile = al.load(
            KCh + kv_slot[:, None] * KC_SEQ_STRIDE + rd[None, :],
            mask=(j + rn[:, None]) < N_KV_ACTIVE,
            other=0.0,
        )
        s = al.tile_dot(q, k_tile, transpose_rhs=True)
        s = s * SCALE
        s = al.where((j + rn)[None, :] < N_KV_ACTIVE, s, -1e30)
        # Causal mask per row i: kv_pos must be <= cache_pos + i.
        s = al.where(
            (j + rn)[None, :] <= (cache_pos + rm)[:, None],
            s,
            -1e30,
        )
        bmax = al.max(s, axis=1)
        mn = al.maximum(m, bmax)
        alpha = al.exp(m - mn)
        l = l * alpha
        o = o * alpha
        p = al.exp(s - mn)
        l = l + al.sum(p, axis=1)
        v_tile = al.load(
            VCh + kv_slot[:, None] * VC_SEQ_STRIDE + rd[None, :],
            mask=(j + rn[:, None]) < N_KV_ACTIVE,
            other=0.0,
        )
        o = o + al.tile_dot(p, v_tile)
        m = mn

    l = al.maximum(l, 1e-30)
    o = o * (1.0 / l)

    POh = partial_O + (bh * SPLITS + split_idx) * BLOCK_M * D
    PLseh = partial_lse + (bh * SPLITS + split_idx) * BLOCK_M

    al.store(
        POh + rm[:, None] * D + rd[None, :],
        o,
        mask=(rm[:, None] < BLOCK_M) & (rd[None, :] < D) & (bh < BH),
    )
    # Per-row lse stored as 1D — each row carries a distinct query.
    lse_value = m + al.log(l)
    al.store(PLseh + rm, lse_value, mask=(rm < BLOCK_M) & (bh < BH))


@al.kernel
def attention_kv_write(
    new_K,
    new_V,
    cache_pos_buf,
    last_real_buf,
    K_cache: al.output,
    V_cache: al.output,
    HEADS_PER_BATCH: al.constexpr = 1,
    HEAD_DIM: al.constexpr = 1,
    K_INPUT: al.constexpr = 1,
    NK_OFFSET: al.constexpr = 0,
    NK_HEAD_STRIDE: al.constexpr = 0,
    NK_SEQ_STRIDE: al.constexpr = 0,
    NV_OFFSET: al.constexpr = 0,
    NV_HEAD_STRIDE: al.constexpr = 0,
    NV_SEQ_STRIDE: al.constexpr = 0,
    KC_HEAD_STRIDE: al.constexpr = 0,
    KC_SEQ_STRIDE: al.constexpr = 0,
    VC_HEAD_STRIDE: al.constexpr = 0,
    VC_SEQ_STRIDE: al.constexpr = 0,
    SLIDING_WINDOW: al.constexpr = 0,
    BLOCK_SIZE: al.constexpr = 256,
):
    """Copy new_K/new_V rows into K_cache/V_cache at positions
    [cache_pos..cache_pos+K_INPUT). One TG per (position, KV head).

    Grid: (K_INPUT, kv_heads) — POSITION on axis-0 so (a) the one-shot
    grid-shrink recipe sizes the launch to the real prompt length (an M_MAX-
    compiled plan otherwise writes all M_MAX positions, ~64x waste at short
    prompts), and (b) the copy parallelizes over positions instead of a serial
    K_INPUT loop in a single TG per head. K_INPUT is now the axis-0 grid extent,
    not an in-kernel loop bound.

    When SLIDING_WINDOW > 0 the cache is a circular buffer of that physical
    size — writes are mapped to ``pos % SLIDING_WINDOW`` so positions past
    the window wrap. Required for Gemma3 sliding-window layers where the
    physical cache is sized at sliding_window (not max_cache_len): without
    the modulo, prefill positions beyond sliding_window write out of
    bounds and produce NaN.

    `last_real_buf` ((1,) runtime int, sliding-window only) holds the chunk's
    last REAL row index, so the real end position is cache_pos + last_real + 1.
    When set (>= 0) only positions [end-SW, end) write — ONE writer per ring
    slot. Without it a chunk longer than the window has rows s, s+SW, s+2·SW…
    racing for slot s with no ordering guarantee (>80% stale-position K/V,
    non-deterministic). Negative = sentinel "no bound" so unmanaged dispatch
    paths fail open, not silent-empty.
    """
    q_idx = al.program_id(0)
    kv_head = al.program_id(1)
    NKh = new_K + NK_OFFSET + kv_head * NK_HEAD_STRIDE
    NVh = new_V + NV_OFFSET + kv_head * NV_HEAD_STRIDE
    KCh = K_cache + kv_head * KC_HEAD_STRIDE
    VCh = V_cache + kv_head * VC_HEAD_STRIDE

    cache_pos = al.cast(al.load(cache_pos_buf), al.int32)
    skip_row = al.cast(0, al.int32)
    if SLIDING_WINDOW > 0:
        pos = cache_pos + q_idx
        last_real = al.cast(al.load(last_real_buf), al.int32)
        ring_end = cache_pos + last_real + 1
        # skip iff a bound is set AND pos outside [ring_end-SW, ring_end)
        below = al.where(pos + SLIDING_WINDOW < ring_end, 1, 0)
        beyond = al.where(pos >= ring_end, 1, 0)
        skip_row = al.where(last_real >= 0, below + beyond, 0)
        write_pos = pos % SLIDING_WINDOW
    else:
        write_pos = cache_pos + q_idx
    for _di in range(0, HEAD_DIM, BLOCK_SIZE):
        # skip_row > 0 pushes offs past HEAD_DIM so the mask kills the whole
        # row (predicated skip — the DSL has no early return).
        offs = _di + al.arange(0, BLOCK_SIZE) + skip_row * HEAD_DIM
        mask = offs < HEAD_DIM
        k_val = al.load(NKh + q_idx * NK_SEQ_STRIDE + offs, mask=mask, other=0.0)
        v_val = al.load(NVh + q_idx * NV_SEQ_STRIDE + offs, mask=mask, other=0.0)
        al.store(KCh + write_pos * KC_SEQ_STRIDE + offs, k_val, mask=mask)
        al.store(VCh + write_pos * VC_SEQ_STRIDE + offs, v_val, mask=mask)


@al.kernel
def attention_decode_combine_multi(
    partial_O,
    partial_lse,
    O: al.output,
    BH: al.constexpr = 1,
    HEADS_PER_BATCH: al.constexpr = 1,
    HEAD_DIM: al.constexpr = 1,
    SPLITS: al.constexpr = 1,
    K_INPUT: al.constexpr = 1,
    BLOCK_M: al.constexpr = 8,
):
    """Combine SPLITS partials for each of K_INPUT query rows.

    Grid: (BH, K_INPUT). partial_O: (BH, SPLITS, BLOCK_M, HEAD_DIM);
    partial_lse: (BH, SPLITS, BLOCK_M). Reads row `q_idx` of each split.
    """
    D = HEAD_DIM
    bh = al.program_id(0)
    q_idx = al.program_id(1)
    batch = bh // HEADS_PER_BATCH
    head = bh - batch * HEADS_PER_BATCH

    rd = al.arange(0, D)

    m_global = -1e30
    sum_weights = 0.0
    for s in range(SPLITS):
        lse_s = al.load(partial_lse + (bh * SPLITS + s) * BLOCK_M + q_idx)
        new_m = al.maximum(m_global, lse_s)
        alpha = al.exp(m_global - new_m)
        sum_weights = sum_weights * alpha + al.exp(lse_s - new_m)
        m_global = new_m

    inv_sum = 1.0 / sum_weights

    o_accum = al.zeros((D,), dtype=al.float32)
    for s in range(SPLITS):
        lse_s = al.load(partial_lse + (bh * SPLITS + s) * BLOCK_M + q_idx)
        coef = al.exp(lse_s - m_global) * inv_sum
        o_s = al.load(
            partial_O + ((bh * SPLITS + s) * BLOCK_M + q_idx) * D + rd,
            mask=rd < D,
            other=0.0,
        )
        # Empty splits (their KV-block loop never ran) carry the lse=-inf
        # sentinel and an UNINITIALIZED persistent-MMA `partial_O`. coef is ~0 for
        # them, but NaN*0 = NaN would poison the sum — so select 0 by the sentinel
        # (a select, not a multiply, so the NaN never propagates). A fully
        # causal-masked real row hits the same sentinel and is a true zero too.
        o_s = al.where(lse_s > -1e29, o_s, 0.0)
        o_accum = o_accum + o_s * coef

    # Output layout: (batch, q_idx, head, head_dim) — matches (B, S, H, D).
    O_SEQ_STRIDE = HEADS_PER_BATCH * D
    O_BATCH_STRIDE = K_INPUT * HEADS_PER_BATCH * D
    Oh = O + batch * O_BATCH_STRIDE + q_idx * O_SEQ_STRIDE + head * D
    al.store(
        Oh + rd,
        o_accum,
        mask=(rd < D) & (bh < BH),
    )
