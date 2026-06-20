"""Qwen 3.5 GatedDeltaNet (linear-attention) kernels.

These kernels back `torch.ops.alloy.linear_attention_update`. Each
corresponds to one stage of `Qwen3_5GatedDeltaNet.forward`
(transformers ref: modeling_qwen3_5.py:424-543):

  causal_conv1d_with_state_prefill / _decode
    Depthwise causal Conv1d with a rolling input window cached in
    `conv_state`. The conv1d is groups=conv_dim depthwise (one kernel
    per channel; conv_kernel_size=4 for qwen3.5).

  l2norm_last_dim
    Per-vector L2 normalisation along the last dim. Matches
    `l2norm(x, dim=-1, eps=1e-6)` exactly (sum-of-squares, not mean).

  delta_net_gate_compute
    g = -exp(A_log) * softplus(a + dt_bias);  beta = sigmoid(b).

  recurrent_gated_delta_rule
    Single-step decode (modeling_qwen3_5.py:314-355).

  rms_norm_gated
    Fused RMSNormGated — matches `Qwen3_5RMSNormGated.forward`.
"""

import math

import alloy


@alloy.kernel
def silu_inplace(
    x,
    out: alloy.output,
    C: alloy.constexpr,
    BLOCK_SIZE: alloy.constexpr = 256,
):
    """Elementwise SiLU. `out[i] = x[i] * sigmoid(x[i])`.

    Split out from the conv kernel because the MSL emitter doesn't hoist
    post-loop scalar expressions into outer scope (the SiLU on a scalar loop
    carry emits an undeclared buffer index); inside its own per-block
    elementwise loop the same composition lowers cleanly.

    **2D launch** over the (B*S, C) conv output: axis-0 = position (B*S),
    axis-1 = channel block. Position on grid axis-0 lets the prefill recipe
    shrink the SiLU to the real prompt length (axis-0 = S = M at B=1).
    """
    pos = alloy.program_id(0)
    cb = alloy.program_id(1)
    col = cb * BLOCK_SIZE + alloy.arange(0, BLOCK_SIZE)
    mask = col < C
    idx = pos * C + col
    v = alloy.cast(alloy.load(x + idx, mask=mask, other=0.0), alloy.float32)
    alloy.store(out + idx, v * (1.0 / (1.0 + alloy.exp(-v))), mask=mask)


@alloy.kernel
def l2norm_last_dim(
    x,
    out: alloy.output,
    N: alloy.constexpr,
    HEADS: alloy.constexpr,
    EPS: alloy.constexpr = 1e-6,
    BLOCK_SIZE: alloy.constexpr = 256,
):
    """L2-normalise each row of an (M, N) matrix along the last dim.

      out[m, n] = x[m, n] * rsqrt(sum_k x[m, k]^2 + EPS)

    Matches HF `l2norm` (modeling_qwen3_5.py:228-231):
      inv_norm = torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)
      return x * inv_norm

    **2D launch**: axis-0 = position (B*S), axis-1 = head (0..HEADS); the buffer
    row is `pos*HEADS + head`. Putting the sequence position on grid axis-0 lets
    the prefill recipe shrink the launch to the real prompt length (it only
    shrinks axis-0). At B=1 axis-0 = S = M, so the padded tail rows beyond the
    real prompt cost no GPU work.
    """
    pos = alloy.program_id(0)
    head = alloy.program_id(1)
    row = pos * HEADS + head
    sq_sum = 0.0
    for _ki in range(0, N, BLOCK_SIZE):
        offs = _ki + alloy.arange(0, BLOCK_SIZE)
        mask = offs < N
        v = alloy.cast(alloy.load(x + row * N + offs, mask=mask, other=0.0), alloy.float32)
        sq_sum = sq_sum + v * v
    inv_norm = alloy.rsqrt(alloy.sum(sq_sum) + EPS)
    for _ki in range(0, N, BLOCK_SIZE):
        offs = _ki + alloy.arange(0, BLOCK_SIZE)
        mask = offs < N
        v = alloy.cast(alloy.load(x + row * N + offs, mask=mask, other=0.0), alloy.float32)
        alloy.store(out + row * N + offs, v * inv_norm, mask=mask)


@alloy.kernel
def causal_conv1d_with_state_prefill(
    x,
    w,
    tape,
    conv_state: alloy.output,
    out: alloy.output,
    BATCH: alloy.constexpr,
    C: alloy.constexpr,
    S: alloy.constexpr,
    K: alloy.constexpr,
    SAVE_TAPE: alloy.constexpr = 0,
):
    """Causal Conv1d with state for prefill. ONLY computes the conv output —
    does NOT finalize conv_state. Use the companion `conv_state_save_real_pos`
    kernel afterwards to write the saved K-window.

    Split so the state write uses one-thread-per-slot addressing: a masked
    store whose address is `where(is_finalize, slot, 0)` lets non-finalize
    threads in the same simdgroup share addr=slot_0, and alloy's masked-store
    emission doesn't reliably suppress their writes — conv_state[slot_0] gets
    clobbered with zero, corrupting the decode pre-context (4-position
    transient in qwen3.5:4b/9b's L0 GDN output).

    x:          (BATCH, S, C)  — reads `mixed_qkv` directly (no transpose to (B,C,S))
    w:          (C, K) — depthwise weight (squeezed from (C, 1, K))
    conv_state: (BATCH, C, K) — input only (read for the pre-context at
                                s ∈ [0, K-1]); annotated alloy.output so the
                                planner orders conv_state_save_real_pos
                                AFTER this kernel (WAW barrier).
    out:        (BATCH, S, C)
    tape:       (BATCH*S*C,) — spec-verify conv tape (SAVE_TAPE=1 only): the
                kernel TEES each row's x value into a persistent bank, so a
                partial-accept rollback can splice the conv window at any
                boundary. DELIBERATELY not alloy.output: nothing in-plan reads
                the bank (the session reads it host-side between replays), and
                an output annotation puts the shared SAVE_TAPE=0 dummy into
                every conv dispatch's write set — the resulting WAW edges
                reorder the planner's conv/state-save scheduling and corrupt
                the conv pre-context. With SAVE_TAPE=0 (all non-spec paths) the
                store folds away and the param is a dead const pointer; callers
                bind a 1-element dummy.

    **2D launch**: axis-0 = position p over B*S, axis-1 = channel block. Each
    threadgroup convolves one position's BLOCK_C channels (a vector over channels,
    the K-tap loop scalar in the position). Operating in (B,S,C) — the layout
    `mixed_qkv` already has — avoids a (B,S,C)→(B,C,S) transpose copy, AND puts
    the sequence position on grid axis-0 so the prefill recipe shrinks the conv
    to the real prompt length (axis-0 = S = M at B=1).
    """
    p = alloy.program_id(0)        # position index over B*S (B=1 → = s)
    cblk = alloy.program_id(1)
    BLOCK_C = 256
    c = cblk * BLOCK_C + alloy.arange(0, BLOCK_C)
    cmask = c < C
    s = p % S
    b = p // S
    s_i = alloy.cast(s, alloy.int32)
    K_minus_1 = K - 1
    zero_i = alloy.cast(0, alloy.int32)
    K_i = alloy.cast(K, alloy.int32)
    acc = alloy.cast(c * 0, alloy.float32)
    for ki in range(K):
        ki_i = alloy.cast(ki, alloy.int32)
        in_pos = s_i - K_minus_1 + ki_i
        x_pos = alloy.where(in_pos >= 0, in_pos, zero_i)
        state_pos = K_i + in_pos
        st_slot = alloy.where(in_pos >= 0, zero_i, state_pos)
        inp_x = alloy.cast(alloy.load(
            x + b * (S * C) + x_pos * C + c, mask=cmask, other=0.0,
        ), alloy.float32)
        inp_st = alloy.cast(alloy.load(
            conv_state + b * (C * K) + c * K + st_slot, mask=cmask, other=0.0,
        ), alloy.float32)
        inp = alloy.where(in_pos >= 0, inp_x, inp_st)
        wv = alloy.cast(alloy.load(
            w + c * K + ki, mask=cmask, other=0.0,
        ), alloy.float32)
        acc = acc + inp * wv
    if SAVE_TAPE:
        # Tape the row's own x. Kept OUTSIDE the K-loop: the loop is a RUNTIME
        # loop in the emitted MSL and an `if` inside it hits the SSA-merge
        # hazard (see causal_conv1d_with_state_decode's docstring) — corrupting
        # the conv even when the branch is constexpr-dead.
        xx = alloy.load(x + b * (S * C) + s * C + c, mask=cmask, other=0.0)
        alloy.store(tape + b * (S * C) + s * C + c, xx, mask=cmask)
    alloy.store(out + b * (S * C) + s * C + c, acc, mask=cmask)


@alloy.kernel
def conv_state_save_real_pos(
    x,
    real_len_t,
    dep_in,
    conv_state: alloy.output,
    BATCH: alloy.constexpr,
    C: alloy.constexpr,
    S: alloy.constexpr,
    K: alloy.constexpr,
):
    """Write conv_state[b, c, k] = x[b, c, real_len - K + k] for k in [0, K).

    `dep_in` is the conv kernel's output buffer — passed as an unused input
    to create a producer→consumer data-flow edge so the planner schedules this
    kernel AFTER the conv kernel. Without that edge the topological sort can put
    this kernel FIRST, and the conv kernel's pre-context reads at s ∈ [0, K-1]
    pick up real-position bytes we wrote here (4-position transient corrupting
    the qwen3.5:4b/9b L0 GDN output). dep_in[0] is multiplied by zero so the
    save value is unaffected.

    conv_state is annotated `alloy.output` so the planner records this
    dispatch's write — without that the WAW conflict isn't tracked.

    x layout: (B, S, C) — matches the (B,S,C) conv input/output.
    Grid: ceil((BATCH * C * K) / BLOCK_SIZE,).
    """
    BLOCK_SIZE = 256
    pid = alloy.program_id(0)
    offs = pid * BLOCK_SIZE + alloy.arange(0, BLOCK_SIZE)
    N = BATCH * C * K
    mask = offs < N
    k = offs % K
    cb = offs // K
    c = cb % C
    b = cb // C
    rl = alloy.load(real_len_t)
    rl_i = alloy.cast(rl, alloy.int32)
    K_i = alloy.cast(K, alloy.int32)
    k_i = alloy.cast(k, alloy.int32)
    src_s = rl_i - K_i + k_i
    src_idx = b * (S * C) + src_s * C + c
    v = alloy.load(x + src_idx, mask=mask, other=0.0)
    # Read dep_in to force a producer→consumer dep with the conv kernel;
    # multiply by zero so v is unchanged (the read is what orders).
    dep_v = alloy.cast(alloy.load(dep_in + 0), alloy.float32) * 0.0
    v = alloy.cast(v, alloy.float32) + dep_v
    alloy.store(conv_state + b * (C * K) + c * K + k, v, mask=mask)


@alloy.kernel
def _conv_state_finalize_prefill(
    x,
    conv_state: alloy.output,
    BATCH: alloy.constexpr,
    C: alloy.constexpr,
    S: alloy.constexpr,
    K: alloy.constexpr,
):
    """Set conv_state to the last K elements of (prior conv_state || x).
    If S >= K, conv_state[b,c,k] = x[b,S-K+k,c]. If S < K, shift the
    existing state left by S and append x.

    x layout: (B, S, C) — matches the (B,S,C) conv input/output.
    Grid: ceil((BATCH * C * K) / BLOCK_SIZE,) — tile pattern.
    """
    BLOCK_SIZE = 256
    pid = alloy.program_id(0)
    offs = pid * BLOCK_SIZE + alloy.arange(0, BLOCK_SIZE)
    N = BATCH * C * K
    mask = offs < N
    k = offs % K
    cb = offs // K
    c = cb % C
    b = cb // C
    if S >= K:
        v = alloy.cast(alloy.load(
            x + b * (S * C) + (S - K + k) * C + c,
            mask=mask, other=0.0,
        ), alloy.float32)
    else:
        # S < K: pull from x if k - (K - S) >= 0, else from old state at k + S.
        src_i = alloy.cast(k, alloy.int32) - (K - S)
        zero_i = alloy.cast(0, alloy.int32)
        from_x = src_i >= 0
        x_addr = alloy.where(from_x, src_i, zero_i)
        st_addr = alloy.where(from_x, zero_i, alloy.cast(k + S, alloy.int32))
        v_x = alloy.cast(alloy.load(
            x + b * (S * C) + x_addr * C + c, mask=mask, other=0.0,
        ), alloy.float32)
        v_st = alloy.cast(alloy.load(
            conv_state + b * (C * K) + c * K + st_addr, mask=mask, other=0.0,
        ), alloy.float32)
        v = alloy.where(from_x, v_x, v_st)
    alloy.store(conv_state + b * (C * K) + c * K + k, v, mask=mask)


@alloy.kernel
def causal_conv1d_with_state_decode(
    x,
    w,
    conv_state: alloy.output,
    out: alloy.output,
    BATCH: alloy.constexpr,
    C: alloy.constexpr,
    K: alloy.constexpr,
):
    """Decode-step Conv1d. Rolls conv_state by -1, writes new x at
    slot K-1, convolves with weight.

    Grid: ceil((BATCH * C) / BLOCK_SIZE,) — tile pattern.

    Alloy converts the K-loop into a runtime `for (uint ki ...)` and
    doesn't merge SSA values from `if/else` branches across iterations
    — `tap = ...` in the if/else would silently collapse to whichever
    branch happens to dominate at codegen, producing wrong conv output.
    Use `where` to pick `tap` per-iteration.
    """
    BLOCK_SIZE = 256
    pid = alloy.program_id(0)
    offs = pid * BLOCK_SIZE + alloy.arange(0, BLOCK_SIZE)
    N = BATCH * C
    mask = offs < N
    c = offs % C
    b = offs // C
    new_x = alloy.cast(alloy.load(
        x + b * C + c, mask=mask, other=0.0,
    ), alloy.float32)
    acc = alloy.cast(offs * 0, alloy.float32)
    K_minus_1 = K - 1
    for ki in range(K):
        ki_i = alloy.cast(ki, alloy.int32)
        # For ki < K-1, read conv_state[ki+1] (the previous-slot value).
        # For ki == K-1, the "ki+1" would be OOB; clamp to K-1 so the
        # load is safe and use `where` to pick `new_x` instead.
        load_slot = alloy.where(
            ki_i < K_minus_1, ki_i + 1, alloy.cast(K_minus_1, alloy.int32),
        )
        from_state = alloy.cast(alloy.load(
            conv_state + b * (C * K) + c * K + load_slot,
            mask=mask, other=0.0,
        ), alloy.float32)
        tap = alloy.where(ki_i < K_minus_1, from_state, new_x)
        alloy.store(conv_state + b * (C * K) + c * K + ki, tap, mask=mask)
        wv = alloy.cast(alloy.load(
            w + c * K + ki, mask=mask, other=0.0,
        ), alloy.float32)
        acc = acc + tap * wv
    alloy.store(out + b * C + c, acc, mask=mask)


@alloy.kernel
def causal_conv1d_gated_decode(
    bcx,
    w,
    conv_state: alloy.output,
    out: alloy.output,
    BATCH: alloy.constexpr,
    C: alloy.constexpr,
    K: alloy.constexpr,
):
    """LFM2 gated decode short-conv in ONE kernel.

    Collapses the `chunk(bcx) -> b*x -> conv -> c*conv_out` diamond: reads the
    b/c/x column-slices of `bcx` (the in_proj output, (B, 1, 3C) flat) directly,
    forms `new_x = b*x`, runs the depthwise causal conv with rolling state, then
    gates the result by `c`.
    """
    BLOCK_SIZE = 256
    pid = alloy.program_id(0)
    offs = pid * BLOCK_SIZE + alloy.arange(0, BLOCK_SIZE)
    N = BATCH * C
    mask = offs < N
    c = offs % C
    b = offs // C
    base3 = b * (3 * C)
    b_gate = alloy.cast(alloy.load(bcx + base3 + c, mask=mask, other=0.0), alloy.float32)
    x_in = alloy.cast(alloy.load(bcx + base3 + 2 * C + c, mask=mask, other=0.0), alloy.float32)
    new_x = b_gate * x_in
    acc = alloy.cast(offs * 0, alloy.float32)
    K_minus_1 = K - 1
    for ki in range(K):
        ki_i = alloy.cast(ki, alloy.int32)
        load_slot = alloy.where(
            ki_i < K_minus_1, ki_i + 1, alloy.cast(K_minus_1, alloy.int32),
        )
        from_state = alloy.cast(alloy.load(
            conv_state + b * (C * K) + c * K + load_slot, mask=mask, other=0.0,
        ), alloy.float32)
        tap = alloy.where(ki_i < K_minus_1, from_state, new_x)
        alloy.store(conv_state + b * (C * K) + c * K + ki, tap, mask=mask)
        wv = alloy.cast(alloy.load(
            w + c * K + ki, mask=mask, other=0.0,
        ), alloy.float32)
        acc = acc + tap * wv
    c_gate = alloy.cast(alloy.load(bcx + base3 + C + c, mask=mask, other=0.0), alloy.float32)
    alloy.store(out + b * C + c, c_gate * acc, mask=mask)


@alloy.kernel
def delta_net_gate_compute(
    a,
    b,
    A_log,
    dt_bias,
    g_out: alloy.output,
    beta_out: alloy.output,
    BATCH: alloy.constexpr,
    S: alloy.constexpr,
    NV: alloy.constexpr,
):
    """g = -exp(A_log) * softplus(a + dt_bias);  beta = sigmoid(b).
    Per-token-per-head scalar; grid ceil((BATCH * S * NV) / BLOCK_SIZE,).
    """
    BLOCK_SIZE = 256
    pid = alloy.program_id(0)
    offs = pid * BLOCK_SIZE + alloy.arange(0, BLOCK_SIZE)
    N = BATCH * S * NV
    mask = offs < N
    h = offs % NV
    aval = alloy.cast(alloy.load(a + offs, mask=mask, other=0.0), alloy.float32)
    bval = alloy.cast(alloy.load(b + offs, mask=mask, other=0.0), alloy.float32)
    A_log_v = alloy.cast(alloy.load(A_log + h, mask=mask, other=0.0), alloy.float32)
    dt_v = alloy.cast(alloy.load(dt_bias + h, mask=mask, other=0.0), alloy.float32)
    sp_arg = aval + dt_v
    # softplus(x) numerically stable: relu(x) + log1p(exp(-|x|)).
    zero_f = alloy.cast(offs * 0, alloy.float32)
    sp_pos = alloy.where(sp_arg > 0.0, sp_arg, zero_f)
    abs_arg = alloy.where(sp_arg > 0.0, sp_arg, -sp_arg)
    sp = sp_pos + alloy.log(1.0 + alloy.exp(-abs_arg))
    g = -alloy.exp(A_log_v) * sp
    beta = 1.0 / (1.0 + alloy.exp(-bval))
    alloy.store(g_out + offs, g, mask=mask)
    alloy.store(beta_out + offs, beta, mask=mask)


@alloy.kernel
def recurrent_gated_delta_rule(
    q,
    k,
    v,
    g,
    beta,
    real_len_t,
    recurrent_state: alloy.output,
    out: alloy.output,
    BATCH: alloy.constexpr,
    S: alloy.constexpr,
    NV: alloy.constexpr,
    DK: alloy.constexpr,
    DV: alloy.constexpr,
    NK: alloy.constexpr = 0,
    HAS_REAL_LEN: alloy.constexpr = 0,
    SAVE_STEPS: alloy.constexpr = 0,
):
    """Per-token DeltaNet recurrence. Updates `recurrent_state` in
    place; writes per-token output to `out`.

    Layout:
      q, k:            (BATCH, S, NK or NV, DK) — NK<NV means GVA-style
                       repeat: head h reads q/k at h // (NV // NK).
      v:               (BATCH, S, NV, DV)
      g, beta:         (BATCH, S, NV)
      recurrent_state: (BATCH, NV, DK, DV)
      out:             (BATCH, S, NV, DV)

    NK==0 (default) is equivalent to NK==NV (no GVA). For qwen3.5:4b/9b
    NK=16 and NV=32 (n_rep=2). Doing the head repeat inside the kernel
    avoids an expand+contiguous on q/k in the handler, which changes the
    surrounding plan's slot structure enough to break the alloy backend's
    Tensor(c!) mutation propagation on this kernel's recurrent_state
    writes.

    `recurrent_state` must hold the state AFTER the last REAL token so the
    subsequent decode step continues correctly. During bucketed prefill the
    loop runs the full padded length S, but tokens [real_len, S) are padding
    whose recurrence would corrupt the carried state. When HAS_REAL_LEN, the
    state is saved at the iteration where s+1 == real_len (read from
    `real_len_t`) and the post-loop save is skipped. When not HAS_REAL_LEN
    (single-token decode, or exact-length prefill where S == real_len) the
    unconditional post-loop save is correct; `real_len_t` is then unused and
    may be any valid buffer.

    Grid: (BATCH * NV * DV,) — one thread per (batch, head, dv).
    Each thread maintains a per-dv column of the state matrix as a
    (DK,) tile in registers, avoiding the alloy DSL's val_loc='address'
    fallout from a 2D (DK, DV) tile built via broadcast outer product.
    Loop over S serially (state carries across tokens). q must be
    pre-scaled by 1/sqrt(DK) by the caller.
    """
    NK_EFF = NK if NK > 0 else NV
    idx = alloy.program_id(0)
    dv = idx % DV
    bh = idx // DV
    h = bh % NV
    bi = bh // NV
    # GVA value→key head pairing. llama.cpp stores the GGUF value heads in
    # [v_per_k, num_k_heads] order (the `ssm.v_head_reordered` layout), so
    # value head `h` pairs with key head `h % NK_EFF` — NOT `h // n_rep`
    # (HF's repeat_interleave convention, which assumes [num_k_heads,
    # v_per_k] order and produces incoherent output on these weights).
    # Everything else (v/z/g/beta/state/out_proj cols) stays consistently in
    # the GGUF value-head order, so only this cross-reference needs the
    # native pairing. For NV == NK (0.8B/2B) `h % NK_EFF == h`, a no-op.
    h_kv = h % NK_EFF
    rk = alloy.arange(0, DK)
    rk_mask = rk < DK
    state_col_addr = bi * (NV * DK * DV) + h * (DK * DV) + rk * DV + dv

    state = alloy.zeros((DK,), dtype=alloy.float32)
    state = state + alloy.cast(
        alloy.load(recurrent_state + state_col_addr, mask=rk_mask, other=0.0),
        alloy.float32,
    )

    if HAS_REAL_LEN:
        rl_i = alloy.cast(alloy.load(real_len_t + 0), alloy.int32)

    for s in range(S):
        gh_addr = bi * (S * NV) + s * NV + h
        g_val = alloy.cast(alloy.load(g + gh_addr), alloy.float32)
        decay = alloy.exp(g_val)
        state = state * decay

        # q/k indexed by h_kv (= h // n_rep when NK<NV, = h when NK==NV)
        kqkv_base = bi * (S * NK_EFF * DK) + s * (NK_EFF * DK) + h_kv * DK
        k_t = alloy.cast(
            alloy.load(k + kqkv_base + rk, mask=rk_mask, other=0.0),
            alloy.float32,
        )
        q_t = alloy.cast(
            alloy.load(q + kqkv_base + rk, mask=rk_mask, other=0.0),
            alloy.float32,
        )
        v_t = alloy.cast(
            alloy.load(v + bi * (S * NV * DV) + s * (NV * DV) + h * DV + dv),
            alloy.float32,
        )
        beta_t = alloy.cast(alloy.load(beta + gh_addr), alloy.float32)

        # Combine two reductions into one to avoid sharing _red across
        # back-to-back reductions in the same loop iteration (race in
        # alloy's emitter when S > 4):
        #   out_v = sum(new_state * q_t)
        #         = sum((state + k_t*delta) * q_t)
        #         = sum(state * q_t) + delta * sum(k_t * q_t)
        # Compute three sums (kv_mem, qstate, qk) — alloy fuses them
        # into one reduction pass with three accumulators.
        kv_mem = alloy.sum(state * k_t)
        qstate = alloy.sum(state * q_t)
        qk = alloy.sum(k_t * q_t)
        delta = (v_t - kv_mem) * beta_t
        out_v = qstate + delta * qk
        state = state + k_t * delta
        alloy.store(
            out + bi * (S * NV * DV) + s * (NV * DV) + h * DV + dv,
            out_v,
        )
        # SAVE_STEPS (speculative verify): write the state after EVERY token to
        # a (S, BATCH, NV, DK, DV) recurrent_state buffer so a partial-accept
        # rollback is a slot-copy instead of a target re-run. Otherwise the
        # single-state save: at the last REAL token (HAS_REAL_LEN) or post-loop,
        # so a following decode continues from the un-polluted state.
        if SAVE_STEPS:
            alloy.store(
                recurrent_state + s * (BATCH * NV * DK * DV) + state_col_addr,
                state,
                mask=rk_mask,
            )
        elif HAS_REAL_LEN:
            alloy.store(
                recurrent_state + state_col_addr,
                state,
                mask=rk_mask & (rl_i == (s + 1)),
            )

    if (not SAVE_STEPS) and (not HAS_REAL_LEN):
        alloy.store(recurrent_state + state_col_addr, state, mask=rk_mask)


@alloy.kernel
def chunked_gdr_stage1(
    q,
    k,
    g,
    beta,
    real_len_t,
    W_out: alloy.output,
    T_out: alloy.output,
    attn_out: alloy.output,
    qg_out: alloy.output,
    kdecay_out: alloy.output,
    BATCH: alloy.constexpr,
    S: alloy.constexpr,
    NV: alloy.constexpr,
    DK: alloy.constexpr,
    DV: alloy.constexpr,
    NK: alloy.constexpr = 0,
    C: alloy.constexpr = 8,
    HAS_REAL_LEN: alloy.constexpr = 0,
):
    """Stage 1 of the 2-stage chunked gated delta rule: the INTRA-chunk machinery.

    One threadgroup per (batch, value-head, chunk) — fully parallel over all
    NC×NV chunks (no cross-chunk dependency), so the expensive T-inversion /
    intra-chunk attention saturates the GPU and is computed once per chunk.
    Emits the per-chunk quantities the serial scan (stage 2) consumes:
      W   = T @ (kᵦ·exp_gc)         (C, DK)   → W_out      (B,NV,S,DK)
      T   = (I - A)⁻¹               (C, C)    → T_out      (B,NV,NC,C,C)
      attn= tril(q@kᵀ)·dm           (C, C)    → attn_out   (B,NV,NC,C,C)
      qg  = q·exp_gc                (C, DK)   → qg_out     (B,NV,S,DK)
      kd  = k·exp(glast-gc)         (C, DK)   → kdecay_out (B,NV,S,DK)
    decay = exp(Σg_chunk) is NOT stored — stage 2 recomputes it from g (a cheap
    1-D reduce; a scalar store/load round-trips unreliably). v is not needed here
    (stage 2 forms U = T@vᵦ per DV-block). q must be pre-scaled 1/√DK, q/k l2normed.
    """
    NK_EFF = NK if NK > 0 else NV
    NC = S // C
    N_SQUARE = int(math.log2(C)) - 1

    # 2D grid (NC, B*NV): chunk index on axis-0 so the one-shot prefill recipe
    # shrinks the launch to the real chunk count ceil(real_len/C). A 1D
    # (B*NV*NC,) grid would put the chunk in the LOW bits of a flat index (chunk-inner),
    # so axis-0 = B*NV*NC scales with M but truncating it drops whole heads, not
    # pad chunks — un-shrinkable. Chunks are fully independent, so the reorder is
    # free. axis-1 = B*NV is fixed in M.
    c = alloy.program_id(0)
    bh = alloy.program_id(1)
    h = bh % NV
    bi = bh // NV
    h_kv = h % NK_EFF
    t0 = c * C

    rc = alloy.arange(0, C)
    rk = alloy.arange(0, DK)
    zero_cc = alloy.zeros((C, C), dtype=alloy.float32)
    one_cc = alloy.zeros((C, C), dtype=alloy.float32) + 1.0
    eye = alloy.where(rc[:, None] == rc[None, :], one_cc, 0.0)
    L_tril = alloy.where(rc[:, None] >= rc[None, :], one_cc, 0.0)

    g_addr = bi * (S * NV) + (t0 + rc) * NV + h
    g_c = alloy.cast(alloy.load(g + g_addr), alloy.float32)
    beta_c = alloy.cast(alloy.load(beta + g_addr), alloy.float32)
    # Padded prefill: zero g/beta for tokens past the real boundary so the
    # chunk machinery treats them as a no-op (g=0,beta=0 → block-triangular T,
    # zero v_new contribution), making the post-scan state the real-len state.
    if HAS_REAL_LEN:
        keep = alloy.cast((t0 + rc) < alloy.cast(alloy.load(real_len_t + 0), alloy.int32), alloy.float32)
        g_c = g_c * keep
        beta_c = beta_c * keep

    glast = alloy.sum(g_c)

    g_col_bc = g_c[:, None] + zero_cc
    gc_col_mat = alloy.tile_dot(L_tril, g_col_bc)
    gc_row = alloy.sum(alloy.where(rc[:, None] <= rc[None, :], g_col_bc, 0.0), axis=0)
    dm = alloy.where(rc[:, None] >= rc[None, :], alloy.exp(gc_col_mat - gc_row), 0.0)
    gc_col = alloy.sum(gc_col_mat * eye, axis=1)
    exp_gc = alloy.exp(gc_col)

    qk_base = (
        bi * (S * NK_EFF * DK) + (t0 + rc)[:, None] * (NK_EFF * DK)
        + h_kv * DK + rk[None, :]
    )
    q_c = alloy.cast(alloy.load(q + qk_base), alloy.float32)
    k_c = alloy.cast(alloy.load(k + qk_base), alloy.float32)

    attn0 = alloy.tile_dot(q_c, k_c, transpose_rhs=True)
    attn = alloy.where(rc[:, None] >= rc[None, :], attn0 * dm, 0.0)
    qg = q_c * exp_gc
    kb = k_c * beta_c[:, None]
    kk = alloy.tile_dot(kb, k_c, transpose_rhs=True)
    A = alloy.where(rc[:, None] > rc[None, :], (kk * dm) * (-1.0), 0.0)
    T = eye + A
    P = A
    for _sq in alloy.unroll(range(N_SQUARE)):
        P = alloy.tile_dot(P, P)
        T = alloy.tile_dot(T, eye + P)
    W = alloy.tile_dot(T, kb * exp_gc)
    kdecay = k_c * alloy.exp(glast - gc_col)

    # (B,NV,S,DK) row-major; this chunk owns rows [t0, t0+C).
    dk_addr = (
        bi * (NV * S * DK) + h * (S * DK)
        + (t0 + rc)[:, None] * DK + rk[None, :]
    )
    alloy.store(W_out + dk_addr, W)
    alloy.store(qg_out + dk_addr, qg)
    alloy.store(kdecay_out + dk_addr, kdecay)
    # (B,NV,NC,C,C)
    cc_addr = (
        bi * (NV * NC * C * C) + h * (NC * C * C) + c * (C * C)
        + rc[:, None] * C + rc[None, :]
    )
    alloy.store(T_out + cc_addr, T)
    alloy.store(attn_out + cc_addr, attn)


@alloy.kernel
def chunked_gdr_stage2(
    v,
    beta,
    g,
    real_len_t,
    W_in,
    T_in,
    attn_in,
    qg_in,
    kdecay_in,
    recurrent_state: alloy.output,
    out: alloy.output,
    BATCH: alloy.constexpr,
    S: alloy.constexpr,
    NV: alloy.constexpr,
    DK: alloy.constexpr,
    DV: alloy.constexpr,
    C: alloy.constexpr = 8,
    DV_BLOCK: alloy.constexpr = 32,
    HAS_REAL_LEN: alloy.constexpr = 0,
):
    """Stage 2: the serial cross-chunk scan, DV-blocked so the (DK,DV_BLOCK)
    state slice + the (C,DK) per-chunk machinery fit 32KB. One threadgroup per
    (batch, value-head, DV-block). Reads stage-1's intra-chunk results (no
    recompute → no redundancy); the only per-DV-block work is the state matmuls:
      U      = T @ (v·β)            (C, DVB)
      v_new  = U - W @ state        (C, DVB)
      out    = qg @ state + attn @ v_new
      state  = state·decay + kdᵀ @ v_new

    The spec verify does NOT use the chunked path at all: the f32
    T-inverse deviates ~1e-2 from the serial evolution on
    ill-conditioned repetitive rows — enough to flip gate near-ties at
    ~6× the rate budget. It dispatches `recurrent_gdr_dvblock_save`
    (serial numerics, DV-blocked, tees the reconstruct ingredients).
    """
    DVB_PER_H = DV // DV_BLOCK
    NC = S // C

    idx = alloy.program_id(0)
    dvb = idx % DVB_PER_H
    bh = idx // DVB_PER_H
    h = bh % NV
    bi = bh // NV
    dv0 = dvb * DV_BLOCK

    rc = alloy.arange(0, C)
    rk = alloy.arange(0, DK)
    rdv = alloy.arange(0, DV_BLOCK)

    state_addr = (
        bi * (NV * DK * DV) + h * (DK * DV)
        + rk[:, None] * DV + (dv0 + rdv)[None, :]
    )
    state = alloy.cast(alloy.load(recurrent_state + state_addr), alloy.float32)

    # Runtime chunk bound. With HAS_REAL_LEN (padded prefill) stage-1 zeroed
    # g/beta past the real boundary, so every chunk entirely in the pad region
    # is a state-preserving no-op (decay=exp(0)=1, vb=0 → v_new=0 → state
    # unchanged) AND its `out` rows are padding nobody reads. Bounding the
    # serial scan to ceil(real_len / C) skips those chunks, so the scan cost
    # tracks the real prompt length instead of the padded S. Without a real_len
    # (decode / exact-length prefill where S == real_len) the full static NC is
    # correct.
    if HAS_REAL_LEN:
        rl_i = alloy.cast(alloy.load(real_len_t + 0), alloy.int32)
        nc_real = (rl_i + (C - 1)) // C
    else:
        nc_real = NC

    # Runtime loop (NOT unrolled): one threadgroup walks the chunks serially,
    # carrying the (DK, DV_BLOCK) `state` tile across iterations. NC = S // C
    # scales with the prefill length, so unrolling emitted MSL ∝ S → superlinear
    # Metal-shader compile (the one-shot-prefill blocker: ~170s for a 2048-token
    # forward). As a runtime ForLoop the body is emitted once → O(1) compile in S.
    # `state` is a loop-carried 2D shmem tile that is BOTH a GEMM operand
    # (W @ state, qg @ state) and updated each iteration — see the codegen support
    # for shmem-resident carried tiles in msl/compiler.py.
    for c in range(nc_real):
        t0 = c * C
        dk_addr = (
            bi * (NV * S * DK) + h * (S * DK)
            + (t0 + rc)[:, None] * DK + rk[None, :]
        )
        W = alloy.cast(alloy.load(W_in + dk_addr), alloy.float32)         # (C, DK)
        qg = alloy.cast(alloy.load(qg_in + dk_addr), alloy.float32)       # (C, DK)
        kdecay = alloy.cast(alloy.load(kdecay_in + dk_addr), alloy.float32)  # (C, DK)
        cc_addr = (
            bi * (NV * NC * C * C) + h * (NC * C * C) + c * (C * C)
            + rc[:, None] * C + rc[None, :]
        )
        T = alloy.cast(alloy.load(T_in + cc_addr), alloy.float32)         # (C, C)
        attn = alloy.cast(alloy.load(attn_in + cc_addr), alloy.float32)   # (C, C)

        v_addr = (
            bi * (S * NV * DV) + (t0 + rc)[:, None] * (NV * DV)
            + h * DV + (dv0 + rdv)[None, :]
        )
        beta_addr = bi * (S * NV) + (t0 + rc) * NV + h
        beta_c = alloy.cast(alloy.load(beta + beta_addr), alloy.float32)  # (C,)
        # Recompute the chunk's scalar decay = exp(Σ g_chunk) from g (a scalar
        # store/load round-trips unreliably; the 1-D reduce is cheap and the
        # lane-mask fix keeps it from summing the next chunk's g).
        g_c = alloy.cast(alloy.load(g + beta_addr), alloy.float32)        # (C,)
        if HAS_REAL_LEN:
            keep = alloy.cast((t0 + rc) < alloy.cast(alloy.load(real_len_t + 0), alloy.int32), alloy.float32)
            g_c = g_c * keep
            beta_c = beta_c * keep
        decay = alloy.exp(alloy.sum(g_c))
        v_c = alloy.cast(alloy.load(v + v_addr), alloy.float32)           # (C, DVB)
        vb = v_c * beta_c[:, None]
        U = alloy.tile_dot(T, vb)                                         # (C, DVB)
        v_new = U - alloy.tile_dot(W, state)                             # (C, DVB)
        out_c = alloy.tile_dot(qg, state) + alloy.tile_dot(attn, v_new)   # (C, DVB)
        alloy.store(out + v_addr, out_c)
        state = state * decay + alloy.tile_dot(kdecay, v_new, transpose_lhs=True)

    alloy.store(recurrent_state + state_addr, state)


@alloy.kernel
def recurrent_gdr_dvblock_save(
    q,
    k,
    v,
    g,
    beta,
    recurrent_state,
    k_tee,
    g_tee,
    beta_tee,
    v_tee,
    out: alloy.output,
    BATCH: alloy.constexpr,
    S: alloy.constexpr,
    NV: alloy.constexpr,
    DK: alloy.constexpr,
    DV: alloy.constexpr,
    NK: alloy.constexpr = 0,
):
    """The SERIAL per-token DeltaNet recurrence, DV-blocked 8× — the spec
    verify's GDR. Per-column math is OP-FOR-OP `recurrent_gated_delta_rule`
    (serial-grade numerics — the chunked T-inverse path's f32 logit
    deviation flips gate near-ties at ~6× the rate budget on repetitive
    rows), but one program owns EIGHT dv columns: one program per (head, dv)
    re-reads the full 512B k/q rows once PER dv COLUMN (B·NV·DV programs ×
    1KB × S ≈ 96MB/dispatch at S=16); blocking 8 columns into one program
    cuts that 8× while the k/q loads, the qk reduction, and the g/beta
    scalars amortize across the block.

    The eight columns are EXPLICIT NAMED carried (DK,) register tiles
    (state0..state7), not a (DK, 8) tile: the runtime-loop carry contract
    is name-based (ast_rewrite's `defined & modified_in_body`), and a 2D
    carried tile built via broadcast outer product hits the val_loc='address'
    fallout — the (1, DVB) delta lives only on thread/row 0 (the emitter
    row-guards elementwise ops on shmem (1, N) tiles and emits NO inter-thread
    broadcast), so the (DK,1)·(1,DVB) state update silently consumes garbage
    on rows 1+ and the elementwise carry update never writes back to the
    carried shmem tile. Eight 1D columns keep every intermediate either
    per-thread (DK,) or a reduction scalar.

    Does NOT write recurrent_state (read-only seed — slot 0 must stay
    the PRE-round state for the session's `gdr_state_reconstruct`) and
    fills no slot bank; instead it tees k_l2/g/beta/v into the layer's
    STABLE registry buffers for the session's per-round reconstruct.

    Tees dedupe across the DV-block programs of a head via a scalar
    `dvb == 0` mask term (the serial kernel's `rl_i == s+1` masking
    pattern); v is teed per program (its 8 columns are unique to it).

    q must be pre-scaled by 1/sqrt(DK); q/k l2-normed (handler contract).
    Grid: (BATCH * NV * (DV / 8),).
    """
    NK_EFF = NK if NK > 0 else NV
    DV_BLOCK = 8
    DVB_PER_H = DV // DV_BLOCK
    idx = alloy.program_id(0)
    dvb = idx % DVB_PER_H
    bh = idx // DVB_PER_H
    h = bh % NV
    bi = bh // NV
    h_kv = h % NK_EFF
    dv0 = dvb * DV_BLOCK
    rk = alloy.arange(0, DK)
    rk_mask = rk < DK
    rs = alloy.arange(0, S)
    # EVERY store below carries an EXPLICIT mask: the emitter's 1D store
    # guard falls back to the LAST traced Compare (`_mask_expr`,
    # last-compare-wins), so an unmasked store after the `dvb == 0` tee
    # compare silently inherits that guard — 15/16 programs skip their
    # v_tee/out writes. w1_mask elects thread 0 as the single writer for
    # the per-column scalar stores (the value is reduction-broadcast,
    # uniform across threads).
    w1_mask = rk < 1
    dvb0 = dvb == 0
    ktee_mask = rk_mask & dvb0
    col_base = bi * (NV * DK * DV) + h * (DK * DV) + rk * DV + dv0

    state0 = alloy.zeros((DK,), dtype=alloy.float32)
    state0 = state0 + alloy.cast(
        alloy.load(recurrent_state + col_base + 0, mask=rk_mask, other=0.0),
        alloy.float32,
    )
    state1 = alloy.zeros((DK,), dtype=alloy.float32)
    state1 = state1 + alloy.cast(
        alloy.load(recurrent_state + col_base + 1, mask=rk_mask, other=0.0),
        alloy.float32,
    )
    state2 = alloy.zeros((DK,), dtype=alloy.float32)
    state2 = state2 + alloy.cast(
        alloy.load(recurrent_state + col_base + 2, mask=rk_mask, other=0.0),
        alloy.float32,
    )
    state3 = alloy.zeros((DK,), dtype=alloy.float32)
    state3 = state3 + alloy.cast(
        alloy.load(recurrent_state + col_base + 3, mask=rk_mask, other=0.0),
        alloy.float32,
    )
    state4 = alloy.zeros((DK,), dtype=alloy.float32)
    state4 = state4 + alloy.cast(
        alloy.load(recurrent_state + col_base + 4, mask=rk_mask, other=0.0),
        alloy.float32,
    )
    state5 = alloy.zeros((DK,), dtype=alloy.float32)
    state5 = state5 + alloy.cast(
        alloy.load(recurrent_state + col_base + 5, mask=rk_mask, other=0.0),
        alloy.float32,
    )
    state6 = alloy.zeros((DK,), dtype=alloy.float32)
    state6 = state6 + alloy.cast(
        alloy.load(recurrent_state + col_base + 6, mask=rk_mask, other=0.0),
        alloy.float32,
    )
    state7 = alloy.zeros((DK,), dtype=alloy.float32)
    state7 = state7 + alloy.cast(
        alloy.load(recurrent_state + col_base + 7, mask=rk_mask, other=0.0),
        alloy.float32,
    )

    # Tee g/beta once, pre-loop, as strided (S,) tiles — NOT per-step
    # scalar stores (a scalar store/load round-trips unreliably; see the
    # stage-1 decay note). Masked loads: the (S,) tile spans fewer lanes
    # than the threadgroup, and unmasked emission reads `base[tid]` on
    # every lane — out of bounds past S.
    gb_addr = bi * (S * NV) + rs * NV + h
    gb_mask = (rs < S) & dvb0
    g_all = alloy.cast(
        alloy.load(g + gb_addr, mask=rs < S, other=0.0), alloy.float32
    )
    beta_all = alloy.cast(
        alloy.load(beta + gb_addr, mask=rs < S, other=0.0), alloy.float32
    )
    alloy.store(g_tee + gb_addr, g_all, mask=gb_mask)
    alloy.store(beta_tee + gb_addr, beta_all, mask=gb_mask)

    for s in range(S):
        gh_addr = bi * (S * NV) + s * NV + h
        g_val = alloy.cast(alloy.load(g + gh_addr), alloy.float32)
        decay = alloy.exp(g_val)
        beta_t = alloy.cast(alloy.load(beta + gh_addr), alloy.float32)
        kq_base = bi * (S * NK_EFF * DK) + s * (NK_EFF * DK) + h_kv * DK
        k_t = alloy.cast(
            alloy.load(k + kq_base + rk, mask=rk_mask, other=0.0),
            alloy.float32,
        )
        q_t = alloy.cast(
            alloy.load(q + kq_base + rk, mask=rk_mask, other=0.0),
            alloy.float32,
        )
        alloy.store(k_tee + kq_base + rk, k_t, mask=ktee_mask)
        v_base = bi * (S * NV * DV) + s * (NV * DV) + h * DV + dv0
        qk = alloy.sum(k_t * q_t)

        state0 = state0 * decay
        v_0 = alloy.cast(alloy.load(v + v_base + 0), alloy.float32)
        alloy.store(v_tee + v_base + 0, v_0, mask=w1_mask)
        kv_0 = alloy.sum(state0 * k_t)
        qs_0 = alloy.sum(state0 * q_t)
        delta_0 = (v_0 - kv_0) * beta_t
        alloy.store(out + v_base + 0, qs_0 + delta_0 * qk, mask=w1_mask)
        state0 = state0 + k_t * delta_0

        state1 = state1 * decay
        v_1 = alloy.cast(alloy.load(v + v_base + 1), alloy.float32)
        alloy.store(v_tee + v_base + 1, v_1, mask=w1_mask)
        kv_1 = alloy.sum(state1 * k_t)
        qs_1 = alloy.sum(state1 * q_t)
        delta_1 = (v_1 - kv_1) * beta_t
        alloy.store(out + v_base + 1, qs_1 + delta_1 * qk, mask=w1_mask)
        state1 = state1 + k_t * delta_1

        state2 = state2 * decay
        v_2 = alloy.cast(alloy.load(v + v_base + 2), alloy.float32)
        alloy.store(v_tee + v_base + 2, v_2, mask=w1_mask)
        kv_2 = alloy.sum(state2 * k_t)
        qs_2 = alloy.sum(state2 * q_t)
        delta_2 = (v_2 - kv_2) * beta_t
        alloy.store(out + v_base + 2, qs_2 + delta_2 * qk, mask=w1_mask)
        state2 = state2 + k_t * delta_2

        state3 = state3 * decay
        v_3 = alloy.cast(alloy.load(v + v_base + 3), alloy.float32)
        alloy.store(v_tee + v_base + 3, v_3, mask=w1_mask)
        kv_3 = alloy.sum(state3 * k_t)
        qs_3 = alloy.sum(state3 * q_t)
        delta_3 = (v_3 - kv_3) * beta_t
        alloy.store(out + v_base + 3, qs_3 + delta_3 * qk, mask=w1_mask)
        state3 = state3 + k_t * delta_3

        state4 = state4 * decay
        v_4 = alloy.cast(alloy.load(v + v_base + 4), alloy.float32)
        alloy.store(v_tee + v_base + 4, v_4, mask=w1_mask)
        kv_4 = alloy.sum(state4 * k_t)
        qs_4 = alloy.sum(state4 * q_t)
        delta_4 = (v_4 - kv_4) * beta_t
        alloy.store(out + v_base + 4, qs_4 + delta_4 * qk, mask=w1_mask)
        state4 = state4 + k_t * delta_4

        state5 = state5 * decay
        v_5 = alloy.cast(alloy.load(v + v_base + 5), alloy.float32)
        alloy.store(v_tee + v_base + 5, v_5, mask=w1_mask)
        kv_5 = alloy.sum(state5 * k_t)
        qs_5 = alloy.sum(state5 * q_t)
        delta_5 = (v_5 - kv_5) * beta_t
        alloy.store(out + v_base + 5, qs_5 + delta_5 * qk, mask=w1_mask)
        state5 = state5 + k_t * delta_5

        state6 = state6 * decay
        v_6 = alloy.cast(alloy.load(v + v_base + 6), alloy.float32)
        alloy.store(v_tee + v_base + 6, v_6, mask=w1_mask)
        kv_6 = alloy.sum(state6 * k_t)
        qs_6 = alloy.sum(state6 * q_t)
        delta_6 = (v_6 - kv_6) * beta_t
        alloy.store(out + v_base + 6, qs_6 + delta_6 * qk, mask=w1_mask)
        state6 = state6 + k_t * delta_6

        state7 = state7 * decay
        v_7 = alloy.cast(alloy.load(v + v_base + 7), alloy.float32)
        alloy.store(v_tee + v_base + 7, v_7, mask=w1_mask)
        kv_7 = alloy.sum(state7 * k_t)
        qs_7 = alloy.sum(state7 * q_t)
        delta_7 = (v_7 - kv_7) * beta_t
        alloy.store(out + v_base + 7, qs_7 + delta_7 * qk, mask=w1_mask)
        state7 = state7 + k_t * delta_7


@alloy.kernel
def gdr_state_reconstruct(
    k,
    g,
    beta,
    v,
    n_t,
    recurrent_state: alloy.output,
    BATCH: alloy.constexpr,
    S: alloy.constexpr,
    NV: alloy.constexpr,
    DK: alloy.constexpr,
    DV: alloy.constexpr,
    NK: alloy.constexpr = 0,
):
    """Advance the live DeltaNet state (slot 0) by the round's committed
    token count, running the EXACT serial recurrence on the chunked
    verify's teed inputs:

      for j < n:  state *= exp(g_j)
                  delta = (v_j − state·k_j) · β_j
                  state += k_j ⊗ delta

    This is `recurrent_gated_delta_rule`'s math and addressing verbatim
    (minus q/out/bank stores), NOT a replay of the chunked formulation's
    deltas: the chunked T-inverse path deviates from the serial state by
    ~1e-2 on ill-conditioned (repetitive) rows, which trips near-tie
    divergences ~6× over the gate's rate budget; the serial recurrence
    tracks plain decode at ~2e-4. One dispatch per layer per ROUND —
    dispatched by the SESSION between plan replays — instead of a full
    slot bank: materializing all S per-token states costs S × the 2MB fp32
    state in write bandwidth (~0.25ms/layer at S=16) while the session only
    ever consumes ONE of them. n_t holds the runtime token count
    (num_accepted + 1) as FLOAT32 — an int32 buffer binds as float* and the
    bit-pattern reads as a denormal → 0-trip loop. The in-place slot-0
    read+write within one program is the serial kernel's SAVE_STEPS=0 pattern.

    Grid: (BATCH * NV * DV,), like the serial kernel.
    """
    NK_EFF = NK if NK > 0 else NV
    idx = alloy.program_id(0)
    dv = idx % DV
    bh = idx // DV
    h = bh % NV
    bi = bh // NV
    h_kv = h % NK_EFF
    rk = alloy.arange(0, DK)
    rk_mask = rk < DK
    state_col_addr = bi * (NV * DK * DV) + h * (DK * DV) + rk * DV + dv

    n_i = alloy.cast(alloy.load(n_t + 0), alloy.int32)
    state = alloy.cast(
        alloy.load(recurrent_state + state_col_addr, mask=rk_mask, other=0.0),
        alloy.float32,
    )
    for s in range(n_i):
        gh_addr = bi * (S * NV) + s * NV + h
        decay = alloy.exp(alloy.cast(alloy.load(g + gh_addr), alloy.float32))
        state = state * decay
        kqkv_base = bi * (S * NK_EFF * DK) + s * (NK_EFF * DK) + h_kv * DK
        k_t = alloy.cast(
            alloy.load(k + kqkv_base + rk, mask=rk_mask, other=0.0),
            alloy.float32,
        )
        v_t = alloy.cast(
            alloy.load(v + bi * (S * NV * DV) + s * (NV * DV) + h * DV + dv),
            alloy.float32,
        )
        beta_t = alloy.cast(alloy.load(beta + gh_addr), alloy.float32)
        kv_mem = alloy.sum(state * k_t)
        delta = (v_t - kv_mem) * beta_t
        state = state + k_t * delta
    alloy.store(recurrent_state + state_col_addr, state, mask=rk_mask)


@alloy.kernel
def rms_norm_gated(
    x,
    z,
    weight,
    out: alloy.output,
    DV: alloy.constexpr,
    HEADS: alloy.constexpr,
    EPS: alloy.constexpr = 1e-6,
    BLOCK_SIZE: alloy.constexpr = 256,
):
    """Fused gated RMSNorm — matches `Qwen3_5RMSNormGated.forward`.

    Reference (modeling_qwen3_5.py:181-190):
      var = (x^2).mean(-1)
      h = x * rsqrt(var + eps)
      h = w * h           # norm BEFORE gate
      out = h * silu(z)   # gate applied last

    **2D launch**: axis-0 = position (B*S), axis-1 = head (0..HEADS); buffer row is
    `pos*HEADS + head`. The position on axis-0 lets the prefill recipe shrink to
    the real prompt length (it only shrinks axis-0; at B=1 axis-0 = S = M).
    """
    pos = alloy.program_id(0)
    head = alloy.program_id(1)
    row = pos * HEADS + head
    sq_sum = 0.0
    for _ki in range(0, DV, BLOCK_SIZE):
        offs = _ki + alloy.arange(0, BLOCK_SIZE)
        mask = offs < DV
        xv = alloy.cast(alloy.load(x + row * DV + offs, mask=mask, other=0.0), alloy.float32)
        sq_sum = sq_sum + xv * xv
    rrms = alloy.rsqrt(alloy.sum(sq_sum) / DV + EPS)
    for _ki in range(0, DV, BLOCK_SIZE):
        offs = _ki + alloy.arange(0, BLOCK_SIZE)
        mask = offs < DV
        xv = alloy.cast(alloy.load(x + row * DV + offs, mask=mask, other=0.0), alloy.float32)
        zv = alloy.cast(alloy.load(z + row * DV + offs, mask=mask, other=0.0), alloy.float32)
        wv = alloy.load(weight + offs, mask=mask, other=0.0)
        zsig = 1.0 / (1.0 + alloy.exp(-zv))
        normed = xv * rrms
        alloy.store(out + row * DV + offs, (wv * normed) * (zv * zsig), mask=mask)
