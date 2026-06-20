"""Tile kernel planning — analyze tile IR and produce optimization decisions.

plan_tile_kernel() runs a pipeline of passes, each a function(func, plan) → None
that reads the IR and mutates the plan. Passes run in dependency order: thread
model, shared memory layout, column tiling, register blocking, etc.
"""

from __future__ import annotations

from alloy._compiler.dtypes import ALLOY_TO_MSL
from alloy._compiler.tile_ir import (
    BinOp,
    Cast,
    Dot,
    ForLoop,
    IndexStore,
    Load,
    MakeRange,
    Reduce,
    Select,
    SimdMatrixOp,
    SimdReduce,
    Store,
    TileFunction,
    TileKernelPlan,
    TileOp,
    UnaryOp,
    WhileLoop,
    get_op_reads,
    walk_ops,
)

def pick_dot_reg(
    M_dim: int, N_dim: int, override: int = 0, target_n_sg: int | None = None
) -> int:
    """Pick the simdgroup register-tile factor for one Dot.

    `reg` controls the register-tile size: each simdgroup MMA covers a
    (reg*8, reg*8) sub-tile of the dot output. Larger `reg` → fewer
    simdgroups (less launch overhead, higher register pressure per
    simdgroup). The factor must evenly divide both M and N; otherwise
    `_sg_cols = N // (reg*8)` underflows to 0 and the dot is silently
    skipped at codegen.

    Used by the planner to size `n_simdgroups` per-Dot AND by the MSL
    emitter to emit per-Dot guards/accumulators that match — both use this
    picker so per-dot decisions agree. A mixed-shape kernel (e.g. SDPA bwd
    dq has 16x8 and 16x128 dots side by side) gets a different `reg` per dot.

    `target_n_sg`: when set, additionally downsize reg if the default
    (M, N)-only pick would use ≤ ¼ of the available simdgroups. Dropping
    reg by one step quadruples this dot's simdgroup count (each lane gets
    a smaller output tile, more lanes can run in parallel). Needed when a
    small dot shares NUM_THREADS with a large dot — otherwise the small
    dot lets most simdgroups idle during its compute-dense K-loop.
    """
    if override in (1, 2, 4):
        reg = override
        while reg > 1 and (
            reg * 8 > M_dim or reg * 8 > N_dim
            or M_dim % (reg * 8) != 0 or N_dim % (reg * 8) != 0
        ):
            reg //= 2
        return reg
    if M_dim >= 64 and N_dim >= 64 and M_dim % 32 == 0 and N_dim % 32 == 0:
        reg = 4
    elif M_dim >= 16 and N_dim >= 16 and M_dim % 16 == 0 and N_dim % 16 == 0:
        reg = 2
    else:
        reg = 1

    # Underutilization downsize: drop reg if doing so quadruples this dot's
    # n_sg without overshooting the kernel-wide target.
    if target_n_sg is not None and target_n_sg > 1:
        while reg > 1:
            n_sg = (M_dim // (reg * 8)) * (N_dim // (reg * 8))
            if n_sg * 4 > target_n_sg:
                break
            next_reg = reg // 2
            if M_dim % (next_reg * 8) != 0 or N_dim % (next_reg * 8) != 0:
                break
            reg = next_reg
    return reg


# --- Auto shared memory planning ---


def _auto_shmem_plan(
    func: TileFunction,
    pad: int = 4,
    max_cols: int | None = None,
    dtype_of: "dict[str, str] | None" = None,
    skip_loads: "set[str] | None" = None,
) -> dict:
    """Auto-generate shared memory plan from tile IR.

    Returns dict mapping value names to (buf_name, rows, cols, stride).
    Values that can share a buffer (non-overlapping lifetimes, same rows)
    are merged to minimize memory usage.

    Liveness is loop-aware: if a value defined before a loop is read inside
    the loop, its lifetime extends to the end of the loop (it must stay live
    across all iterations).

    When max_cols is set, column dimensions are clamped (for column tiling).

    `dtype_of` is an optional per-value shmem dtype. When provided, slot
    reuse requires the candidate value's dtype to match the slot's existing
    dtype — values whose shmem layouts are different sizes (e.g. bf16 vs
    f32) cannot safely share a buffer. When omitted, all values are treated
    as the same dtype.
    """
    # Collect all ops with timestamps + loop span info
    flat_ops = list(walk_ops(func.ops))

    # Build loop span map: for each ForLoop, record (start_t, end_t) in flat_ops
    loop_spans: list[tuple[int, int]] = []
    _build_loop_spans(func.ops, flat_ops, loop_spans)

    # Find values needing shared memory
    # (name, rows, cols, birth_time, death_time)
    shmem_vals: list[list] = []
    val_names = set()

    for t, op in enumerate(flat_ops):
        if isinstance(op, Load) and op.result and len(op.result.shape) == 2:
            # Device-direct loads stream from device into the MMA — no shmem tile.
            if skip_loads and op.result.name in skip_loads:
                continue
            shmem_vals.append([op.result.name, *op.result.shape, t, t])
            val_names.add(op.result.name)
        if isinstance(op, Dot) and op.result:
            # Persistent MMA accumulators (op.acc set by _opt_persistent_mma)
            # live in simdgroup registers across the loop. Skip the shmem
            # slot unless the result is consumed by ANOTHER Dot (chained GEMM).
            #
            # FA-2 forward's acc_pre_scale path keeps everything in registers:
            # alpha lives in a tiny per-row scratch and the rescale is applied
            # via thread_elements() writes on the persistent accumulator.
            # A full (BLOCK_M, HEAD_DIM) slot would bloat the shmem budget
            # without ever being written, and force mask/exp slots to inherit
            # the wide stride via slot-reuse.
            if op.acc is not None:
                result_name = op.result.name
                # In-loop consumers (rescales like `o = o * alpha`) are emitted
                # as in-register simdgroup ops and need no shmem slot. Only
                # post-loop consumers that aren't a Store (or chained Dot) need
                # a slot to spill into.
                in_loop_end = -1
                for ls, le in loop_spans:
                    if ls <= t <= le and le > in_loop_end:
                        in_loop_end = le
                consumed_by_dot = False
                consumed_by_other = False
                for ct, consumer in enumerate(flat_ops):
                    if ct <= in_loop_end:
                        continue
                    if isinstance(consumer, Dot):
                        if (consumer.lhs is not None and consumer.lhs.name == result_name) or \
                           (consumer.rhs is not None and consumer.rhs.name == result_name):
                            consumed_by_dot = True
                    elif not isinstance(consumer, Store):
                        for v in consumer.operand_values():
                            if v.name == result_name:
                                consumed_by_other = True
                                break
                if not (consumed_by_dot or consumed_by_other):
                    # Store-only consumer writes straight from registers to
                    # device memory — no shmem slot needed.
                    continue
            shmem_vals.append([op.result.name, *op.result.shape, t, t])
            val_names.add(op.result.name)

    # Also add 2D BinOp/UnaryOp/Select results that feed into Dot or Reduce
    # — they need their own shmem slot so simdgroup_load (Dot) or the
    # `_row * stride + _n` shmem walk (Reduce, see _emit_reduce in
    # msl/reductions.py) can read them. Must be done before in-place chain
    # extension so the chain terminates at own-slot nodes. Without the Reduce
    # side of this rule, the M=1 vector-path attention chain `sum(...) →
    # where(...) → max(axis=0)` emits a `_ssel*[...]` shmem reference for the
    # Where output the decl-emission pass never sees, producing MSL with
    # undeclared identifiers and a Metal compile error.
    shmem_consumer_inputs: set[str] = set()
    for op in flat_ops:
        if isinstance(op, Dot):
            if op.lhs:
                shmem_consumer_inputs.add(op.lhs.name)
            if op.rhs:
                shmem_consumer_inputs.add(op.rhs.name)
        elif isinstance(op, Reduce):
            if op.input:
                shmem_consumer_inputs.add(op.input.name)
        elif isinstance(op, Select):
            # A 2D Select reads its branches per-element (`_resolve_2d_elem` /
            # `_elem_access`). A branch produced as a (C,C) BinOp of row-varying
            # (per-thread) and column-varying (local_array) operands has no
            # shmem buffer and resolves to a single per-thread scalar — wrong.
            # Force such 2D branches into shmem so they index [_row,_c].
            for v in (op.true_val, op.false_val):
                if v is not None and len(v.shape) == 2:
                    shmem_consumer_inputs.add(v.name)

    for t, op in enumerate(flat_ops):
        if isinstance(op, (BinOp, UnaryOp, Select)) and op.result:
            name = op.result.name
            # A 2D Select result ALWAYS needs its own shmem slot: `_emit_select`
            # only has a 2D shmem-write path (no local_array output), and a
            # positional/row-varying mask can't be a per-thread constant row.
            # BinOp/UnaryOp keep the consumer gate (they have local_array/in-place
            # output paths and only need a slot when feeding Dot/Reduce).
            needs_slot = isinstance(op, Select) or name in shmem_consumer_inputs
            if (
                needs_slot
                and name not in val_names
                and len(op.result.shape) == 2
            ):
                shmem_vals.append([name, *op.result.shape, t, t])
                val_names.add(name)

    # Update death times based on last read
    for t, op in enumerate(flat_ops):
        reads = get_op_reads(op)
        for sv in shmem_vals:
            if sv[0] in reads:
                sv[4] = max(sv[4], t)

    # Extend death through in-place BinOp/UnaryOp/Select chains.  The emitter
    # writes the result of an in-place elementwise op back into one of its
    # inputs' shmem slots (the "first shared operand").  That parent slot must
    # stay alive until the chain terminates.
    #
    # Chain rules — must match `_emit_binop` / `_emit_unaryop` / `_emit_select`:
    # 1. A BinOp/UnaryOp/Select result has its OWN slot if it is in `val_names`
    #    (Load, Dot, or feeds-into-Dot).  Otherwise it writes in-place to the
    #    slot of its first shared operand.
    # 2. Sibling operands (e.g. `mask` in `s + mask`) are NOT in-place children
    #    of the parent — they live in their own slot and are merely read.
    # 3. An own-slot child (e.g. `p` in `s → exp(s) = p`) pins the parent slot
    #    alive through the own-slot child's BIRTH (the emitter reads the
    #    parent's slot to materialize the child), but the chain does NOT
    #    continue past the own-slot child.
    #
    # Compute which 2D values end up in shared memory (own slot or in-place).
    shared_2d_names = set(val_names)
    changed = True
    while changed:
        changed = False
        for op in flat_ops:
            if (
                isinstance(op, (BinOp, UnaryOp, Select, Cast))
                and op.result
                and len(op.result.shape) == 2
                and op.result.name not in shared_2d_names
            ):
                if any(r in shared_2d_names for r in get_op_reads(op)):
                    shared_2d_names.add(op.result.name)
                    changed = True

    # In-place score-buffer merge (attention softmax). A Select/BinOp/UnaryOp
    # result that feeds a Reduce/Dot normally takes its OWN slot, but when this
    # op is the LAST reader of an element-wise shmem input, it can overwrite that
    # input's slot in place instead — the input is dead, so the write is safe
    # (the emitter's COW still guards any survivor). This collapses the
    # causal-mask / subtract / exp outputs back onto the QK-score slot: ONE
    # (BLOCK_M, BLOCK_N) buffer for the whole softmax instead of two, freeing a
    # shmem resident (deep-prefill split-K attention is shmem-occupancy-bound:
    # 12.8KB/TG → 2 residents; dropping a score buffer → ~8.7KB → 3). Gated to
    # the online-softmax signature so GEMM epilogues keep their own slots.
    if _has_loop_carried_reduce(func):
        _last_read: dict[str, int] = {}
        for t, op in enumerate(flat_ops):
            for r in get_op_reads(op):
                _last_read[r] = t

        def _ew_inplace_parent(op):
            if isinstance(op, (UnaryOp, Cast)):
                cands = (op.input,)
            elif isinstance(op, Select):
                cands = (op.true_val, op.false_val)
            elif isinstance(op, BinOp):
                cands = (op.lhs, op.rhs)
            else:
                return None
            for c in cands:
                if (
                    c is not None
                    and c.name in shared_2d_names
                    and tuple(c.shape) == tuple(op.result.shape)
                ):
                    return c
            return None

        for t, op in enumerate(flat_ops):
            if not isinstance(op, (BinOp, UnaryOp, Select)) or op.result is None:
                continue
            if op.result.name not in val_names:
                continue
            parent = _ew_inplace_parent(op)
            if parent is not None and _last_read.get(parent.name) == t:
                val_names.discard(op.result.name)
                shmem_vals[:] = [sv for sv in shmem_vals if sv[0] != op.result.name]

    def _first_shared_parent(op):
        """Return the name of the shmem value whose slot `op` writes in-place to."""
        if isinstance(op, BinOp):
            sides = (op.lhs, op.rhs)
        elif isinstance(op, (UnaryOp, Cast)):
            sides = (op.input,)
        elif isinstance(op, Select):
            sides = (op.true_val, op.false_val)
        else:
            return None
        for s in sides:
            if s is not None and s.name in shared_2d_names:
                return s.name
        return None

    # Build in-place child map: for each parent shmem value, the set of
    # BinOp/UnaryOp/Select results that write back into the parent's slot.
    inplace_children: dict[str, set[str]] = {}
    for op in flat_ops:
        if not isinstance(op, (BinOp, UnaryOp, Select, Cast)):
            continue
        if not op.result or len(op.result.shape) != 2:
            continue
        if op.result.name in val_names:
            continue  # has its own slot — not in-place on anyone
        parent = _first_shared_parent(op)
        if parent is not None:
            inplace_children.setdefault(parent, set()).add(op.result.name)

    last_read_idx: dict[str, int] = {}
    for t, op in enumerate(flat_ops):
        for name in get_op_reads(op):
            last_read_idx[name] = t  # t increases, so the final write is the max

    # Walk chains; extend each shmem val's death through its in-place children.
    for sv in shmem_vals:
        frontier = {sv[0]}
        visited = set()
        while frontier:
            cur = frontier.pop()
            if cur in visited:
                continue
            visited.add(cur)
            t = last_read_idx.get(cur, -1)
            if t > sv[4]:
                sv[4] = t
            if cur in inplace_children:
                frontier |= inplace_children[cur]

    # Loop-aware liveness: if a value is born before a loop but read inside
    # it, extend death to the end of the loop (value must survive all iters)
    for sv in shmem_vals:
        birth, death = sv[3], sv[4]
        for loop_start, loop_end in loop_spans:
            if birth < loop_start and death >= loop_start:
                sv[4] = max(sv[4], loop_end)

    # Sort by birth time
    shmem_vals.sort(key=lambda x: x[3])

    # Greedy buffer allocation with reuse
    # Each slot: [buf_name, rows, stride, death_time, dtype_or_None]
    slots: list[list] = []
    plan: dict = {}

    for sv in shmem_vals:
        name, rows, cols, birth, death = sv
        if max_cols is not None:
            cols = min(cols, max_cols)
        stride = cols + pad
        val_dt = dtype_of.get(name) if dtype_of else None

        # Try to reuse a slot whose last value died before this one.
        # Both the slot and the value must use the same stride (the max),
        # since they share the same physical shared memory buffer.
        # Per-buffer dtype: a slot's dtype is fixed at allocation; subsequent
        # reusers must match it (or both must be unspecified). Mixing dtypes
        # in a single slot would mean different element sizes (bf16=2B vs
        # f32=4B), so the same byte offset means different element indices.
        reused = False
        for slot in slots:
            buf_name, slot_rows, slot_stride, end_time, slot_dt = slot
            if end_time < birth and slot_rows == rows and slot_dt == val_dt:
                new_stride = max(slot_stride, stride)
                slot[2] = new_stride
                slot[3] = death
                plan[name] = (buf_name, rows, cols, new_stride)
                reused = True
                break

        if not reused:
            buf_name = f"_s{len(slots)}"
            slots.append([buf_name, rows, stride, death, val_dt])
            plan[name] = (buf_name, rows, cols, stride)

    return plan


def _build_loop_spans(ops: list[TileOp], flat_ops: list[TileOp], out: list[tuple[int, int]]):
    """Find (start_t, end_t) in flat_ops for each ForLoop/WhileLoop."""
    for op in ops:
        if isinstance(op, ForLoop):
            if op.body:
                start = _find_flat_idx(flat_ops, op.body[0])
                end = _find_flat_idx(flat_ops, op.body[-1])
                if start >= 0 and end >= 0:
                    out.append((start, end))
            _build_loop_spans(op.body, flat_ops, out)
        elif isinstance(op, WhileLoop):
            all_body = op.cond_body + op.body
            if all_body:
                start = _find_flat_idx(flat_ops, all_body[0])
                end = _find_flat_idx(flat_ops, all_body[-1])
                if start >= 0 and end >= 0:
                    out.append((start, end))
            _build_loop_spans(op.body, flat_ops, out)
            _build_loop_spans(op.cond_body, flat_ops, out)


def _find_flat_idx(flat_ops: list[TileOp], target: TileOp) -> int:
    """Find index of target op in flat_ops by identity."""
    for i, op in enumerate(flat_ops):
        if op is target:
            return i
    return -1


# --- Pipeline ---


def plan_tile_kernel(func: TileFunction) -> TileKernelPlan:
    """Analyze tile IR and produce a complete compilation plan.

    Runs optimization passes in order. Each pass reads the IR and
    annotates the plan. Later passes can depend on earlier decisions.
    """
    plan = TileKernelPlan()

    for optimize in _PASSES:
        optimize(func, plan)

    # Inject NUM_THREADS for runtime dispatch
    func.constexpr_values["NUM_THREADS"] = plan.threads

    return plan


# --- Individual optimization passes ---


def _pass_classify_buffers(func: TileFunction, plan: TileKernelPlan):
    """Classify buffer parameters and detect output buffers."""
    for p in func.params:
        if not p.is_constexpr:
            plan.buffer_params.append(p.name)
    _scan_outputs(func.ops, plan.outputs)


def _pass_detect_dtypes(func: TileFunction, plan: TileKernelPlan):
    """Determine compute, accumulator, and shared memory dtypes."""
    # Classify all buffer dtypes
    for p in func.params:
        if not p.is_constexpr:
            msl_dt = ALLOY_TO_MSL.get(p.dtype, "float")
            plan.buffer_dtypes[p.name] = msl_dt

    # Input dtype = first buffer's type
    for p in func.params:
        if not p.is_constexpr:
            msl_dt = ALLOY_TO_MSL.get(p.dtype, "float")
            if msl_dt != "float":
                plan.dtype = msl_dt
            break

    # Accumulator dtype: float for sub-word types.
    if plan.dtype in ("char", "short", "bfloat"):
        plan.acc_dtype = "float"
    elif plan.dtype == "half":
        plan.acc_dtype = "float"
    else:
        plan.acc_dtype = plan.dtype
    # Shared memory dtype: for sub-word types (char, short), shared memory
    # holds promoted values (float) since simdgroup MMA requires float/half.
    # For bfloat, also promote to float: Metal's `bfloat` has no implicit
    # narrowing from `float`, and softmax/reduce intermediates need f32
    # precision anyway. Load/Store still use the narrow dtype; only the
    # in-kernel intermediate tiles are wider.
    # Packed loads with inline dequant: promote to half (not float) since
    # INT4 precision is fully captured by f16 and half MMA is 2x throughput.
    has_packed_dequant = any(
        isinstance(op, Load)
        and (
            (op.pack_factor > 0 and op.dequant_scale_ptr is not None)
            or op.dequant_format == "q6_k"
            or op.dequant_format == "q4_k"
        )
        for op in walk_ops(func.ops)
    )
    if plan.dtype in ("char", "short", "uchar", "ushort"):
        plan.shmem_dtype = "half" if has_packed_dequant else "float"
    if plan.dtype == "bfloat":
        # For pure-MMA kernels (have Dot, no tile-level Reduce), keep bfloat
        # operands in shmem and let the float accumulator handle precision.
        # Apple Silicon's `simdgroup_multiply_accumulate(float, bfloat, bfloat,
        # float)` is a native intrinsic — running f32 operands through MMA on a
        # bf16 model wastes 2× shmem and 2× MMA throughput. Tile-level Reduce
        # ops still need f32 shmem accumulation, so fall back there.
        #
        # HIGH_PRECISION=1 forces f32 shmem — used by the SDPA backward dq/dkdv
        # kernels when K-bias amplification makes the cancellation
        # `ds = p * (dp - delta)` lose precision through the bf16 shmem spill.
        force_f32 = func.constexpr_values.get("HIGH_PRECISION", 0) == 1
        ops_iter = list(walk_ops(func.ops))
        has_dot = any(isinstance(op, Dot) for op in ops_iter)
        has_reduce = any(isinstance(op, Reduce) for op in ops_iter)
        if has_dot and not has_reduce and not force_f32:
            plan.shmem_dtype = "bfloat"
        else:
            plan.shmem_dtype = "float"
    else:
        plan.shmem_dtype = plan.dtype
    # No padding for float (Apple Silicon has no bank conflicts for aligned float access).
    # Half precision still needs padding to avoid vectorization alignment issues.
    # Packed-dequant kernels plan as "float" (their activation input is f32)
    # but the per-buffer dtype pass narrows every shmem tile to half — without
    # the half pad their 64B-row tiles put transposed simdgroup_loads on a
    # 2-bank stride (4-way conflict on every column read of the B tiles).
    plan.pad = 0 if plan.shmem_dtype == "float" and not has_packed_dequant else 8
    plan.vec_width = 4 if plan.shmem_dtype == "float" else 8


# Apple Silicon hardware cap on threads per threadgroup (apple7-9 all report
# 1024). Exceeding it does NOT raise at dispatch — Metal silently runs only the
# first 1024 threads, so any simdgroup past that never executes and its output
# is left uninitialised (garbage), not an error. Hardcoded for the same reason
# SHMEM_BUDGET is: the compiler layer must not depend on the runtime device.
_MAX_THREADS_PER_THREADGROUP = 1024


def _has_loop_carried_reduce(func: TileFunction) -> bool:
    """True if a Reduce appears inside a For/While loop body.

    This is the online-softmax (attention) signature: per-iteration max/sum
    reductions over the QK-score tile, carried across the K-loop. It excludes
    GEMM epilogues whose reductions (rmsnorm/layernorm) sit at the top level
    after the matmul, never inside a loop — so the attention-specific softmax
    lane parallelism below doesn't perturb their tuned tpr=1 threading.
    """

    def _body_has_reduce(ops: list[TileOp]) -> bool:
        for op in ops:
            if isinstance(op, Reduce):
                return True
            if isinstance(op, ForLoop) and _body_has_reduce(op.body):
                return True
            if isinstance(op, WhileLoop) and (
                _body_has_reduce(op.body) or _body_has_reduce(op.cond_body)
            ):
                return True
        return False

    for op in func.ops:
        if isinstance(op, ForLoop) and _body_has_reduce(op.body):
            return True
        if isinstance(op, WhileLoop) and (
            _body_has_reduce(op.body) or _body_has_reduce(op.cond_body)
        ):
            return True
    return False


def _pass_thread_model(func: TileFunction, plan: TileKernelPlan):
    """Decide thread count and threads-per-row based on IR structure."""
    flat_ops = list(walk_ops(func.ops))

    has_dots = False
    n_simdgroups = 1

    # Auto-select register blocking based on dot dimensions
    ce = func.constexpr_values
    reg_override = ce.get("_reg", 0)

    # Per-dot reg picks: each dot's largest reg that evenly tiles its (M, N).
    # The emitter consults `pick_dot_reg(M, N, override)` per-Dot so a kernel
    # with mixed dot shapes (e.g. SDPA bwd dq's small (16, 8) dot alongside
    # a wider (16, 128) dot) can give the big dot reg=2 while the small dot
    # keeps reg=1; a kernel-global MAX reg would make the small dot's
    # `_sg_cols = N // (reg*8) = 0` and skip execution entirely.
    #
    # `plan.reg_m` / `plan.reg_n` keeps the MIN reg (used as a lower bound
    # for non-dot heuristics like shmem dtype); `n_simdgroups` reflects the
    # launch size required by the WIDEST per-dot tile, so each dot's
    # `if (simd_gid < N)` guard has enough simdgroups available.
    # Two passes: first compute default per-dot reg (shape-only) to find max
    # n_sg across all dots in this kernel, then re-pick per-dot reg with that
    # max as a target so under-parallel dots can downsize.
    dot_dims: list[tuple[int, int]] = []
    for op in flat_ops:
        if isinstance(op, Dot):
            has_dots = True
            if op.transpose_lhs:
                _, M_dim = op.lhs.shape
            else:
                M_dim, _ = op.lhs.shape
            N_dim = op.rhs.shape[0] if op.transpose_rhs else op.rhs.shape[1]
            dot_dims.append((M_dim, N_dim))

    n_simdgroups_dots = 1
    for M_dim, N_dim in dot_dims:
        reg = pick_dot_reg(M_dim, N_dim, override=reg_override)
        sg_rows = max(M_dim // (reg * 8), 1)
        sg_cols = max(N_dim // (reg * 8), 1)
        n_simdgroups_dots = max(n_simdgroups_dots, sg_rows * sg_cols)
    plan.n_sg_target = n_simdgroups_dots if n_simdgroups_dots > 1 else None

    per_dot_reg: list[int] = []
    for M_dim, N_dim in dot_dims:
        reg = pick_dot_reg(
            M_dim, N_dim, override=reg_override, target_n_sg=plan.n_sg_target
        )
        per_dot_reg.append(reg)
    if per_dot_reg:
        plan.reg_m = min(per_dot_reg)
        plan.reg_n = plan.reg_m
        n_simdgroups = max(n_simdgroups, n_simdgroups_dots)

    ce = func.constexpr_values
    block_m = ce.get("BLOCK_M", ce.get("BLOCK_SIZE", None))
    # Whether the row block size is an EXPLICIT constexpr (BLOCK_M/BLOCK_SIZE) vs
    # derived from the largest arange below. The softmax lane-parallel (tpr>1)
    # optimization only knows the block size IS the row count when it's explicit
    # (attention's tunable BLOCK_M). When derived from the largest arange it may
    # be a K/column dim (e.g. DeltaNet stage2's rk/rdv span 16 while the row dim
    # rc spans C=8), so tpr would be sized off the wrong axis and the per-row
    # broadcast loads index OOB. Gate tpr>1 on an explicit block size.
    block_m_explicit = block_m is not None

    # If no explicit block size, derive from the largest MakeRange
    has_make_range = False
    if block_m is None:
        for op in flat_ops:
            if isinstance(op, MakeRange):
                has_make_range = True
                block_m = max(block_m or 0, op.end - op.start)
    else:
        has_make_range = any(isinstance(op, MakeRange) for op in flat_ops)

    # Explicit NUM_THREADS always takes priority (user declares exact thread count)
    explicit_threads = ce.get("NUM_THREADS")
    if explicit_threads is not None and explicit_threads <= 0:
        explicit_threads = None

    # Detect simd_reduce — all threads in one SIMD group cooperate on a single
    # reduction. The threadgroup IS the SIMD group (32 threads), not block_m * tpr.
    has_simd_reduce = any(isinstance(op, SimdReduce) for op in flat_ops)

    if explicit_threads is not None:
        # User kernel with explicit NUM_THREADS constexpr
        plan.threads = explicit_threads
        plan.tpr = 1
    elif has_dots:
        plan.threads = n_simdgroups * 32
        plan.tpr = 1
        # Online-softmax (attention): the per-row mask/exp/max/sum over the
        # QK-score tile is otherwise serial — `has_dots` forces tpr=1, so one
        # thread per row walks BLOCK_N columns while the rest of the
        # n_simdgroups*32 launch idles. Distribute each of the BLOCK_M rows
        # across L lanes of a simdgroup (L | 32, 32/L rows packed per
        # simdgroup) so those passes run lane-parallel with a simd_shuffle_xor
        # butterfly reduction — matching llama.cpp's in-register FA softmax.
        # The launch size is unchanged (so occupancy is unaffected) and the
        # dots, which use their own simd_gid/lane tiling, are untouched. Gated
        # to the loop-carried-reduce signature so GEMM epilogues stay tpr=1.
        if block_m and block_m_explicit and _has_loop_carried_reduce(func):
            lanes_per_row = min(32, plan.threads // block_m)
            if lanes_per_row >= 2:
                # floor to a power of two that evenly tiles a 32-lane simdgroup
                lanes_per_row = 1 << (lanes_per_row.bit_length() - 1)
                if 32 % lanes_per_row == 0 and block_m * lanes_per_row <= plan.threads:
                    plan.tpr = lanes_per_row
    elif has_simd_reduce:
        # simd_reduce: a SIMD group (32 threads) cooperates on each reduction.
        # If the kernel's largest arange spans multiple simdgroups (e.g.
        # arange(0, 32 * NUM_SPLITS) for split-K), launch with all of them
        # so multi-simdgroup work distribution actually executes — otherwise
        # only simd 0 runs, the rest are silently skipped, and post-reduce
        # cross-simdgroup work (shmem partials + barrier + combine) reads
        # uninitialised slots from the missing simdgroups.
        plan.threads = max(32, block_m or 32)
        plan.tpr = 1
    elif block_m is None:
        # Scalar kernel (no arange, no block size) — one thread per program
        plan.threads = 1
        plan.tpr = 1
    elif not has_make_range:
        # No MakeRange: BLOCK_SIZE/BLOCK_M is the total thread count
        # (threadgroup-ops kernels using thread_id, simd, shared, etc.)
        plan.threads = ce.get("NUM_THREADS") or ce.get("BLOCK_SIZE") or ce.get("BLOCK") or block_m
        plan.tpr = 1
    elif block_m <= 32:
        plan.tpr = 32
        plan.threads = block_m * plan.tpr
    else:
        plan.tpr = 1
        plan.threads = block_m

    # Fail loudly if the chosen launch exceeds the hardware threadgroup-thread
    # cap. Without this the overflow is silent garbage — Metal runs only the
    # first 1024 threads, leaving the rest of the output unwritten. Fix by
    # lowering n_simdgroups (bigger reg/BLOCK_M, or head_dim tiling).
    if plan.threads > _MAX_THREADS_PER_THREADGROUP:
        raise ValueError(
            f"kernel {func.name!r}: {plan.threads} threads/threadgroup "
            f"({plan.threads // 32} simdgroups) exceeds the Apple Silicon limit of "
            f"{_MAX_THREADS_PER_THREADGROUP}. Lower n_simdgroups via register "
            f"blocking (larger reg/BLOCK_M) or tile the head_dim."
        )


def _pass_register_resident(func: TileFunction, plan: TileKernelPlan):
    """Skip shared memory when each lane's data fits in registers."""
    flat_ops = list(walk_ops(func.ops))
    has_dots = any(isinstance(op, Dot) for op in flat_ops)

    N = func.constexpr_values.get("N", 0)
    if not has_dots and plan.tpr > 1 and N > 0:
        D = N // plan.tpr
        if D > 0 and N % plan.tpr == 0 and D <= 32:
            plan.register_resident = True


def _compute_per_value_shmem_dtype(
    func: TileFunction, plan: TileKernelPlan, candidate_names: "set[str]"
) -> "dict[str, str]":
    """Per-shmem-value dtype, narrowed where MMA pairing allows.

    Defaults each value to `plan.shmem_dtype` (the kernel-global pick). For
    Load values whose source buffer is narrower (e.g. bf16 input under
    HIGH_PRECISION=1 → global f32 shmem), tries to keep the value in the
    narrower dtype IF the resulting per-value dtype is consistent with the
    Apple Silicon MMA pair-dtype constraint (`simdgroup_multiply_accumulate`
    requires both inputs to be the same matrix dtype — bf16-bf16 or f32-f32,
    not mixed).

    Returns a mapping val_name → MSL dtype string. Only includes values from
    `candidate_names` (the set of values that actually need shmem slots).

    Algorithm: per-value preferred dtype, union-find over MMA-paired values,
    each equivalence class collapses to its widest preferred dtype. (A pair
    where one side is forced to f32 because of OTHER consumers makes the
    whole class f32.)
    """
    flat_ops = list(walk_ops(func.ops))
    op_by_result: dict[str, TileOp] = {}
    for op in flat_ops:
        if op.result is not None:
            op_by_result[op.result.name] = op

    # Promotion order: numeric rank, used to pick the wider dtype in a class.
    PROMOTION = {"char": 0, "uchar": 0, "short": 1, "ushort": 1, "bfloat": 2, "half": 2, "float": 3}

    def _wider(a: str, b: str) -> str:
        return a if PROMOTION.get(a, 99) >= PROMOTION.get(b, 99) else b

    preferred: dict[str, str] = {}
    for name in candidate_names:
        op = op_by_result.get(name)
        if isinstance(op, Load) and op.ptr is not None:
            # Packed loads (INT4 dequant) and Q6_K cooperative loads store the
            # post-dequant OUTPUT in shmem. Q4/Q5/Q6/Q8 information content
            # (4-8 bit nibble × fp16 scale + fp16 bias) fits losslessly into
            # fp16, and pairing the dequant target with a half activation tile
            # lets the MMA run on the half×half→float intrinsic at 2x the
            # FP32 throughput. Prefer half regardless of plan.shmem_dtype —
            # the union-find Load×Load downcast below propagates this back to
            # the activation Load so both MMA operands land at half.
            if op.pack_factor > 0 or op.dequant_format == "q6_k" or op.dequant_format == "q4_k":
                preferred[name] = "half"
                continue
            buf_name = op.ptr.name
            buf_dt = plan.buffer_dtypes.get(buf_name, plan.shmem_dtype)
            # If the source is wider than the kernel-global, keep the
            # global (we never widen a buffer past the global pick); if
            # narrower, prefer the source dtype.
            if PROMOTION.get(buf_dt, 99) > PROMOTION.get(plan.shmem_dtype, 99):
                preferred[name] = plan.shmem_dtype
            else:
                preferred[name] = buf_dt
        elif isinstance(op, Dot) and op.acc_pre_scale is not None:
            # FA-2 persistent-o spill: the Dot result is round-tripped
            # through shmem so the per-row alpha rescale can multiply into
            # the f32 accumulator. The spill is `simdgroup_store(<float
            # matrix>, threadgroup T*)` so T MUST be float; if shmem_dtype
            # is half (f16 inference path) the buffer would otherwise narrow
            # to half and the simdgroup_store template fails to type-check.
            # Force float here regardless of plan.shmem_dtype.
            preferred[name] = "float"
        else:
            preferred[name] = plan.shmem_dtype

    # Pre-mark single-Dot-consumer 2D BinOp/UnaryOp/Select results as
    # bfloat-preferred so union-find collapses their MMA partner classes
    # to bf16 too. Small per-dispatch shmem savings free room for larger
    # BM/BN tiles.
    has_dot_user_pre: dict[str, bool] = {}
    has_nondot_user_pre: dict[str, bool] = {}
    for op in flat_ops:
        for r in get_op_reads(op):
            if isinstance(op, Dot):
                has_dot_user_pre[r] = True
            else:
                has_nondot_user_pre[r] = True
    for name in candidate_names:
        op = op_by_result.get(name)
        if not isinstance(op, (BinOp, UnaryOp, Select)):
            continue
        if not op.result or len(op.result.shape) != 2:
            continue
        if not has_dot_user_pre.get(name) or has_nondot_user_pre.get(name):
            continue
        if preferred.get(name) == "float":
            preferred[name] = "bfloat"

    # Union-find over MMA-paired values
    parent: dict[str, str] = {n: n for n in candidate_names}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Identify "fanout" computed Dot inputs: 2D BinOp/UnaryOp/Select results
    # in shmem that are consumed by BOTH a Dot AND a non-Dot op (typically a
    # downstream BinOp chain). The canonical case is `p_t` in SDPA bwd dkdv:
    #   p_t = exp(s_t - lse)
    #   dv = dv + tile_dot(p_t, go)        # Dot consumer
    #   ds_t = p_t * (dp_t - drow[None,:])  # BinOp consumer
    # Skipping these in the union-find prevents `p_t`'s float dtype from
    # pulling its MMA partner (`go`) wide. Apple Silicon's mixed-dtype MMA
    # accepts `(f32, bf16)` operand pairs at full throughput, so the
    # resulting bf16(go) × f32(p_t) Dot lowers correctly.
    #
    # Single-Dot-consumer values (e.g. `ds_t` in dkdv, `ds` in dq) are NOT
    # excluded — they keep their float dtype in per-buf so the MMA partner
    # they share with critical-precision Loads stays at f32. The bf16-cast
    # pass below handles those separately by overriding the OWN slot's
    # dtype, which the mixed-MMA tolerance turns into a precision-preserving
    # `bf16(ds_t) × f32(K)` multiply.
    fanout_excluded: set[str] = set()
    has_dot_user: dict[str, bool] = {}
    has_nondot_user: dict[str, bool] = {}
    for op in flat_ops:
        for r in get_op_reads(op):
            if isinstance(op, Dot):
                has_dot_user[r] = True
            else:
                has_nondot_user[r] = True
    for name in candidate_names:
        op = op_by_result.get(name)
        if isinstance(op, (BinOp, UnaryOp, Select)) and op.result and len(op.result.shape) == 2:
            if has_dot_user.get(name) and has_nondot_user.get(name):
                fanout_excluded.add(name)

    for op in flat_ops:
        if not isinstance(op, Dot):
            continue
        # Persistent-MMA Dot's accumulator stays in registers, but the lhs/rhs
        # pair still needs matching shmem dtype.
        lhs_n = op.lhs.name if op.lhs is not None else None
        rhs_n = op.rhs.name if op.rhs is not None else None
        if lhs_n in fanout_excluded or rhs_n in fanout_excluded:
            continue
        if lhs_n in preferred and rhs_n in preferred:
            # Float-paired-with-half Load×Load MMA: downcast the float
            # operand to half on cooperative load so the MMA runs the
            # native half×half→float intrinsic (2× throughput vs the
            # mixed `(float, half, float)` path which falls back to f32
            # MMA speed on Apple Silicon). Float accumulator preserves
            # precision through the K-loop; Q's downcast on shmem store
            # is the only precision cost, and it's bounded by the
            # subsequent softmax that the FA-2 attention pattern uses
            # anyway. Pure Loads only — keeps the BinOp/UnaryOp/Select
            # precision-sensitive computed values at their preferred
            # widths.
            lhs_op = op_by_result.get(lhs_n) if lhs_n is not None else None
            rhs_op = op_by_result.get(rhs_n) if rhs_n is not None else None
            # Mixed-precision MMA: if one operand prefers half (a Q4/Q5/Q6/Q8
            # dequant-target Load) and the other prefers float (the
            # activation Load), downcast both to half so we hit the
            # half×half→float intrinsic at 2x FP32 throughput. The activation
            # is small (post-RMSNorm hidden state, ~±10) and fits losslessly
            # in fp16; the dequant target is bit-equivalent in half by
            # construction. The float accumulator preserves K-loop precision.
            if (
                isinstance(lhs_op, Load) and isinstance(rhs_op, Load)
                and {preferred[lhs_n], preferred[rhs_n]} == {"float", "half"}
            ):
                preferred[lhs_n] = "half"
                preferred[rhs_n] = "half"
            union(lhs_n, rhs_n)

    # Collapse each class to its widest preferred dtype
    class_dt: dict[str, str] = {}
    for name in candidate_names:
        root = find(name)
        cur = class_dt.get(root, preferred[name])
        class_dt[root] = _wider(cur, preferred[name])

    return {n: class_dt[find(n)] for n in candidate_names}


def _pass_device_direct(func: TileFunction, plan: TileKernelPlan):
    """Mark Dot operands that should stream from device instead of staging.

    The RHS operand B is independent of sg_m, so each of the `sg_rows = M/TM`
    row-simdgroups reloads the same B bytes. When sg_rows == 1 there is a single
    reader, so cooperative staging is a pure load → barrier → readback round-trip
    with zero sharing — always stream. When sg_rows > 1 multiple row-simdgroups
    reload B, which cooperative staging amortizes; that genuinely pays off for a
    persistent-MMA GEMM contraction (large sg_rows, bounded K-loop, B reused
    across a wide output), so keep it staged there.

    But sg_rows > 1 must NOT force staging for a Flash-Attention score dot. Its
    RHS (K) is reloaded fresh every KV-block iteration, reused only sg_rows (2-8)
    times, and the shmem it occupies caps BLOCK_N over a thousands-deep KV scan —
    the amortization (saving one cheap simdgroup_load) is dwarfed by the lost
    occupancy. `sg_rows` alone is a leaky GEMM proxy (it reads 1 for attention
    only when BLOCK_M <= reg*8). The discriminator is the FA-vs-GEMM signature:
    a non-accumulating score dot (`op.acc is None`) inside a loop-carried softmax
    (`_has_loop_carried_reduce`) streams its RHS regardless of sg_rows; a
    persistent-MMA accumulator dot (`op.acc is not None`) keeps the sg_rows rule.

    (LHS A is symmetric under sg_cols == 1, but in attention Q/p are shared across
    sg_n so they stay staged.) Only a raw 2D Load whose sole consumer is this Dot,
    with an elidable row mask (the streamed load can't bounds-check per element),
    is eligible.
    """
    flat_ops = list(walk_ops(func.ops))
    consumers: dict[str, list] = {}
    for op in flat_ops:
        for r in get_op_reads(op):
            consumers.setdefault(r, []).append(op)
    op_by_result = {op.result.name: op for op in flat_ops if op.result is not None}
    ce = func.constexpr_values
    reg_override = ce.get("_reg", 0)
    # A reduce inside a loop is the online-softmax (flash-attention) signature;
    # GEMM epilogue reduces (rmsnorm/layernorm) sit at the top level. Used to let
    # the FA score dot stream its RHS even when sg_rows > 1 (see docstring).
    fa_kernel = _has_loop_carried_reduce(func)

    def _row_mask_elidable(load_op, rows: int) -> bool:
        # No mask → already unconditional. With a mask, require the KV length to
        # divide the tile so no block overhangs (same condition that elides the
        # cooperative load's row bound). Conservative: only the attention KV case.
        if load_op.mask is None:
            return True
        n_kv = ce.get("KV_LEN") or ce.get("SEQ_LEN")
        return isinstance(n_kv, int) and rows > 0 and n_kv % rows == 0

    for op in flat_ops:
        if not isinstance(op, Dot) or op.rhs is None or len(op.rhs.shape) != 2:
            continue
        if op.transpose_lhs:
            M_dim = op.lhs.shape[1]
        else:
            M_dim = op.lhs.shape[0]
        N_dim = op.rhs.shape[0] if op.transpose_rhs else op.rhs.shape[1]
        reg = pick_dot_reg(M_dim, N_dim, override=reg_override, target_n_sg=plan.n_sg_target)
        sg_rows = M_dim // (reg * 8)
        # sg_rows == 1 always streams. sg_rows > 1 keeps cooperative staging for a
        # persistent-MMA GEMM, but a flash-attention score dot (no accumulator,
        # inside a loop-carried softmax) streams regardless — staging its K only
        # caps BLOCK_N for a cheap reload it never needed.
        is_fa_score_dot = fa_kernel and op.acc is None
        if sg_rows != 1 and not is_fa_score_dot:
            continue
        rhs_op = op_by_result.get(op.rhs.name)
        if (
            isinstance(rhs_op, Load)
            and consumers.get(op.rhs.name) == [op]
            and _row_mask_elidable(rhs_op, op.rhs.shape[0])
        ):
            plan.device_direct_loads[op.rhs.name] = "rhs"


def _pass_shmem_and_column_tiling(func: TileFunction, plan: TileKernelPlan):
    """Plan shared memory layout with buffer reuse and column tiling."""
    if plan.register_resident:
        plan.shmem_plan = {}
        return

    flat_ops = list(walk_ops(func.ops))

    ce = func.constexpr_values
    N = ce.get("N", 0)
    block_m = ce.get("BLOCK_M", ce.get("BLOCK_SIZE", 256))
    sizeof_elem = 4 if plan.shmem_dtype == "float" else 2

    # Estimate shmem with reuse by running the auto plan without col tiling
    n_2d_loads = sum(1 for op in flat_ops if isinstance(op, Load) and len(op.result.shape) == 2)
    SHMEM_BUDGET = 32768
    trial_plan = _auto_shmem_plan(func, pad=plan.pad, skip_loads=plan.device_direct_loads)
    if trial_plan:
        # Count distinct buffers and their sizes
        buf_sizes = {}
        for _, (buf, rows, cols, stride) in trial_plan.items():
            cur = buf_sizes.get(buf, 0)
            buf_sizes[buf] = max(cur, rows * stride)
        shmem_needed = sum(buf_sizes.values()) * sizeof_elem
    else:
        shmem_needed = 0

    # Column tiling only applies to 1D row-per-thread kernels (softmax, etc.)
    # GEMM kernels (with Dot ops) must not be column-tiled — the tuner
    # filters configs that exceed shmem budget instead.
    has_dots = any(isinstance(op, Dot) for op in flat_ops)
    if N > 0 and n_2d_loads > 0 and shmem_needed > SHMEM_BUDGET and not has_dots:
        max_cols = SHMEM_BUDGET // (max(n_2d_loads, 1) * block_m * sizeof_elem) - plan.pad
        block_n = (max_cols // plan.vec_width) * plan.vec_width
        block_n = max(plan.vec_width, block_n)
        for candidate in range(block_n, plan.vec_width - 1, -plan.vec_width):
            if candidate > 0 and N % candidate == 0:
                block_n = candidate
                break
        plan.col_tiled = True
        plan.block_n = block_n

    plan.shmem_plan = _auto_shmem_plan(
        func, pad=plan.pad, max_cols=plan.block_n, skip_loads=plan.device_direct_loads
    )

    # Per-buffer shmem dtype: any time the kernel-global shmem is wider
    # than a Load's source-buffer dtype, try to keep that Load's slot at
    # the narrower source dtype — saves 50% per such buffer. Constrained
    # by Apple Silicon's MMA pair-dtype rule (`simdgroup_multiply_accumulate`
    # requires both inputs to share matrix element type), which the
    # union-find in `_compute_per_value_shmem_dtype` enforces by
    # collapsing each MMA-paired class to its widest preferred dtype.
    #
    # Safe across all f32-shmem kernel classes:
    #   * HIGH_PRECISION=1 SDPA bwd: narrows the bf16 dO/V/mask Loads
    #     whose MMA partner is also bf16 (e.g. dV = pᵀ @ dO pairs the
    #     bf16 dO with the float p, so dO stays float; but mask Loads
    #     and select unpaired Loads can drop to bf16).
    #   * Reduce-driven f32 shmem (SDPA forward, softmax): narrowing
    #     bf16 K/V loads back to bf16 reproduces the standard
    #     bf16×bf16→f32 MMA path that LLM training uses everywhere
    #     else, with no measured forward regression.
    #   * INT4 packed shmem: the dequant-target dtype is float/half
    #     and the packed Load writes its post-dequant OUTPUT into shmem
    #     at that target dtype, not the source `char` bytes. Per-value
    #     `_compute_per_value_shmem_dtype` checks `op.pack_factor > 0`
    #     and pins those Loads to `plan.shmem_dtype`.
    # Also run when shmem_dtype is half but persistent-o (FA-2 forward
    # rescale-accumulate) is in play: that pass spills f32 accumulators via
    # shmem and needs the spill slot at f32 even though the kernel-global
    # dtype is half.
    has_persistent_o = any(
        isinstance(op, Dot) and op.acc_pre_scale is not None for op in flat_ops
    )
    if plan.shmem_plan and (plan.shmem_dtype == "float" or has_persistent_o):
        cand_names = set(plan.shmem_plan.keys())
        per_val = _compute_per_value_shmem_dtype(func, plan, cand_names)
        # Buffer dtype = widest across its values (so a slot holding ANY value
        # that needs f32 stays f32). Only override when the buffer can be
        # narrower than the global shmem dtype.
        PROMOTION = {"char": 0, "uchar": 0, "short": 1, "ushort": 1, "bfloat": 2, "half": 2, "float": 3}
        buf_dt: dict[str, str] = {}
        for val_name, (buf_name, _, _, _) in plan.shmem_plan.items():
            cur = buf_dt.get(buf_name, per_val[val_name])
            if PROMOTION.get(per_val[val_name], 99) > PROMOTION.get(cur, 99):
                cur = per_val[val_name]
            buf_dt[buf_name] = cur
        # Re-run the layout with dtype-aware reuse so values that disagree on
        # dtype don't share a slot — `_auto_shmem_plan` will allocate
        # additional buffers in that case.
        plan.shmem_plan = _auto_shmem_plan(
            func, pad=plan.pad, max_cols=plan.block_n, dtype_of=per_val,
            skip_loads=plan.device_direct_loads,
        )
        # Recompute per-buf dtype against the new (possibly larger) plan
        buf_dt = {}
        for val_name, (buf_name, _, _, _) in plan.shmem_plan.items():
            cur = buf_dt.get(buf_name, per_val[val_name])
            if PROMOTION.get(per_val[val_name], 99) > PROMOTION.get(cur, 99):
                cur = per_val[val_name]
            buf_dt[buf_name] = cur
        plan.shmem_buf_dtype = {bn: dt for bn, dt in buf_dt.items() if dt != plan.shmem_dtype}

    # bf16-cast intermediate Dot inputs: 2D BinOp/UnaryOp/Select results in
    # shmem that are used ONLY as Dot lhs/rhs (not as inputs to other shmem
    # ops) get their owning shmem buffer cast to bf16, even when the
    # kernel-global is float (HP=1 path). Apple Silicon's
    # `simdgroup_multiply_accumulate` accepts mixed-dtype `(Ta, Tb)` operand
    # pairs, so we end up with bf16(ds) × float(K) → float multiply — the
    # SAME per-element precision as the standard bf16 attention path PyTorch
    # uses everywhere, while the upstream `dp - delta` cancellation that
    # HP=1 was specifically added to protect stays in f32 shmem.
    #
    # Grad precision is preserved within the HP=1 tolerance, while the cast
    # saves the per-iter shmem footprint of `ds`. The same pattern fires in
    # `attention_strided_masked_by_batch_with_lse`'s `p = exp(s - lse)`
    # chain feeding `o += p @ v`.
    #
    # Why we don't ALSO narrow K via union-find: K narrowing would cast
    # K to bf16 in shmem, losing precision in BOTH `s = q @ K.T` AND
    # `dq += ds @ K`, which blows past the HP=1 grad tolerance. The
    # MMA-pair-dtype constraint that this would normally enforce is
    # bypassed here ONLY because mixed-dtype MMA is legal on Apple Silicon.
    #
    # Gated to bf16 models: the cast reproduces the bf16×float MMA the model
    # already runs everywhere, so the Dot input loses no precision the model
    # had. On an f32 model (f32 SDPA backward) there is no such bf16 path —
    # casting `ds` to bf16 would inject ~bf16 error into an otherwise-exact
    # gradient (the dq/dkdv backward, where `ds` feeds dQ/dK directly).
    if plan.shmem_plan and plan.shmem_dtype == "float" and plan.dtype == "bfloat":
        op_by_result: dict[str, TileOp] = {}
        for op in walk_ops(func.ops):
            if op.result is not None:
                op_by_result[op.result.name] = op
        # Names that consumers other than Dot also read.
        non_dot_consumed: set[str] = set()
        for op in walk_ops(func.ops):
            if isinstance(op, Dot):
                continue
            for r in get_op_reads(op):
                non_dot_consumed.add(r)
        bf16_cast: set[str] = set()
        for name in plan.shmem_plan:
            op = op_by_result.get(name)
            if not isinstance(op, (BinOp, UnaryOp, Select)):
                continue
            if not op.result or len(op.result.shape) != 2:
                continue
            # Only as Dot input — not consumed elsewhere
            if name in non_dot_consumed:
                continue
            bf16_cast.add(name)
        for name in bf16_cast:
            buf_name = plan.shmem_plan[name][0]
            sharers = [n for n, (b, _, _, _) in plan.shmem_plan.items() if b == buf_name]
            # Only override when the buffer is exclusively held by bf16-cast
            # candidates — otherwise we'd corrupt a sibling f32 value.
            if all(s in bf16_cast for s in sharers):
                plan.shmem_buf_dtype[buf_name] = "bfloat"


def _pass_row_bounds(func: TileFunction, plan: TileKernelPlan):
    """Set row bound guard variable when M is a constexpr."""
    plan.row_bound = "M" if "M" in func.constexpr_values else None


def _pass_double_buffer(func: TileFunction, plan: TileKernelPlan):
    """Enable K-loop double-buffering when requested AND shmem budget allows it.

    Double buffering doubles the shared memory for cooperative load tiles.
    Reject if doubling would exceed the 32KB threadgroup memory limit.
    """
    if not func.options.get("double_buffer"):
        return
    # Estimate shmem with double buffering: each buffer in the plan doubles
    buf_sizes: dict[str, int] = {}
    sizeof = 4 if plan.shmem_dtype == "float" else 2
    for _, (buf, rows, cols, stride) in plan.shmem_plan.items():
        cur = buf_sizes.get(buf, 0)
        buf_sizes[buf] = max(cur, rows * stride * sizeof)
    doubled = sum(buf_sizes.values()) * 2
    if doubled <= 32768:
        plan.double_buffer = True


def _pass_persistent_o_budget(func: TileFunction, plan: TileKernelPlan):
    """Disable persistent-o (FA-2 forward rescale-accumulate) when
    its threadgroup scratch (`_acc_pre_scale_*` + `_acc_post_scale_*`,
    BLOCK_M floats each) would push the kernel over the 32 KB
    threadgroup-memory limit.

    Runs after _pass_shmem_and_column_tiling so the per-buffer dtype pass
    (which widens persistent-o spill slots to float — see the Dot/acc_pre_scale
    branch in _compute_per_value_shmem_dtype) has already settled. The
    accounting here sums real per-buffer dtypes and reserves the scratch.

    For large-HEAD_DIM f32 attention the Q/K/V/Mask tiles already saturate
    the 32 KB threadgroup budget; persistent-o adds spill scratch on top and
    can push the kernel over the limit. Falling back to non-persistent
    codegen keeps the kernel valid at the cost of giving up persistent-o
    gains on shapes without headroom; bf16-narrowed shapes have room to
    spare and keep the optimization.
    """
    if plan.register_resident or not plan.shmem_plan:
        return
    flat_ops = list(walk_ops(func.ops))
    pre_scale_dots = [
        op for op in flat_ops if isinstance(op, Dot) and op.acc_pre_scale is not None
    ]
    post_scale_stores = [
        op for op in flat_ops if isinstance(op, Store) and op.acc_post_scale is not None
    ]
    if not pre_scale_dots and not post_scale_stores:
        return

    DTYPE_BYTES = {"char": 1, "uchar": 1, "short": 2, "ushort": 2,
                   "bfloat": 2, "half": 2, "float": 4}
    buf_sizes: dict[str, int] = {}
    per_buf_dtype = plan.shmem_buf_dtype or {}
    for _, (buf, rows, cols, stride) in plan.shmem_plan.items():
        dt = per_buf_dtype.get(buf, plan.shmem_dtype)
        sizeof = DTYPE_BYTES.get(dt, 4)
        cur = buf_sizes.get(buf, 0)
        buf_sizes[buf] = max(cur, rows * stride * sizeof)
    base_bytes = sum(buf_sizes.values())
    if plan.double_buffer:
        base_bytes *= 2

    extra_bytes = 0
    for op in pre_scale_dots:
        extra_bytes += op.acc.shape[0] * 4
    for op in post_scale_stores:
        producer = next(
            (p for p in flat_ops if isinstance(p, Dot) and op.value is not None
             and p.result.name == op.value.name),
            None,
        )
        if producer is not None and producer.acc is not None:
            extra_bytes += producer.acc.shape[0] * 4

    if base_bytes + extra_bytes <= 32768:
        return

    for op in pre_scale_dots:
        op.acc_pre_scale = None
    for op in post_scale_stores:
        op.acc_post_scale = None


# --- Pass pipeline — order matters (later passes depend on earlier ones) ---

_PASSES = [
    _pass_classify_buffers,
    _pass_detect_dtypes,
    _pass_thread_model,  # depends on: dtypes (for reg sizing)
    _pass_register_resident,  # depends on: thread_model (tpr)
    _pass_device_direct,  # depends on: thread_model (n_sg_target); before shmem
    _pass_shmem_and_column_tiling,  # depends on: register_resident, dtypes
    _pass_row_bounds,
    _pass_double_buffer,
]


# --- Helpers ---


def _scan_outputs(ops: list[TileOp], outputs: set[str]):
    """Find all store target buffer names."""
    for op in walk_ops(ops):
        if isinstance(op, Store) and op.ptr:
            outputs.add(op.ptr.name)
        elif isinstance(op, SimdMatrixOp) and op.op == "store":
            if len(op.args) >= 2 and op.args[1]:
                outputs.add(op.args[1].name)
        elif isinstance(op, IndexStore) and op.base:
            outputs.add(op.base.name)
