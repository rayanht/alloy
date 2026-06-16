"""Cross-LazyOp row-pass fusion — generic composer.

Collapses a connected subgraph of row-local LazyOps (elem ops +
row_reduce_sum) into a single kernel by rebuilding the IR.

No new @al.kernel templates are written. The composer:

  1. Builds a DAG over input LazyOps using `shares_allocation()` edges
     (reshape-safe, unlike raw id() comparisons on buffer_args).
  2. Grows a group outward from a row_reduce_sum seed, admitting
     elem ops whose producers are all inside the group ∪ external
     inputs, until no more can be added.
  3. Classifies every external input buffer by **shape-total**:
         M*N  → FULL_2D
         M    → ROW_BCAST
         N    → COL_BCAST
         1    → SCALAR
  4. Extracts each LazyOp's scalar compute chain — BinOp / UnaryOp /
     Select / Compare / TernaryOp producing the stored value, plus
     Reduce for row_reduce_sum. Index-math (dtype i32/bool) is
     discarded; the composer re-emits clean per-row indexing.
  5. Assembles a fresh TileFunction with row-per-threadgroup dispatch,
     one Load per external input at the natural rank, a single RowPass
     containing the remapped compute (inter-op flows wired directly
     in SSA, intermediate tensor stores elided), and one Store of the
     final output.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
from typing import cast

from alloy._compiler.tile_ir import (
    BinOp,
    Compare,
    Constant,
    Copy,
    ForLoop,
    Layout,
    Load,
    MakeRange,
    ProgramId,
    Reduce,
    RowPass,
    Select,
    Splat,
    Store,
    TernaryOp,
    TileBuilder,
    TileFunction,
    TileOp,
    TileParam,
    TileValue,
    UnaryOp,
    walk_ops,
)
from alloy._dispatch.lazy import LazyOp
from alloy._runtime.alloy_buffer import AlloyBuffer


# ─── Access pattern classification by shape-total ───


class AccessKind(Enum):
    FULL_2D = "full_2d"
    ROW_BCAST = "row_bcast"
    COL_BCAST = "col_bcast"
    SCALAR = "scalar"


def _total(shape) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n


def _classify(arr: AlloyBuffer, M: int, N: int) -> AccessKind | None:
    t = _total(arr.shape)
    if t == M * N:
        return AccessKind.FULL_2D
    if t == M:
        return AccessKind.ROW_BCAST
    if t == N:
        return AccessKind.COL_BCAST
    if t == 1:
        return AccessKind.SCALAR
    return None


# ─── Role of a LazyOp within a candidate group ───


class Role(Enum):
    ELEM_FLAT = "elem_flat"
    ROW_REDUCE_SUM = "row_reduce_sum"


_ELEM_ALLOWED = (
    ProgramId, MakeRange, Constant, Splat, Load, Store, Copy,
    BinOp, UnaryOp, Select, Compare, TernaryOp,
)


def _is_row_reduce(op: LazyOp) -> bool:
    """Structural check: does the kernel reduce each row to a scalar?

    Any kernel built by _make_row_reduce (row_reduce_sum, row_mean,
    row_reduce_max, …) lowers to `ForLoop(body: Load+combine) → Reduce(axis=0)
    → [optional scalar tail] → Store(scalar)`. Two conditions:

      1. At least one Reduce(axis=0) somewhere in the ops.
      2. The top-level Store writes a SCALAR value (shape ()), not a tile.

    The second condition distinguishes row-reduce kernels from layernorm /
    softmax / attention, which all contain internal Reduces but store a
    per-row tile of shape (BLOCK_SIZE,). Without it, _is_row_reduce would
    seed ROW_PASS groups on any kernel that happens to reduce internally,
    and the extractor would then walk a much bigger op graph than it can
    handle.
    """
    func = op.func
    if func is None:
        return False
    has_row_reduce = any(
        isinstance(o, Reduce) and o.axis == 0 for o in walk_ops(func.ops)
    )
    if not has_row_reduce:
        return False
    # Distinguish row-reduce from layernorm/softmax/attention, all of which
    # contain internal Reduces but store a per-row TILE (shape (BLOCK_SIZE,)).
    # A row-reduce stores a per-row scalar.
    for o in func.ops:
        if isinstance(o, Store) and o.value is not None:
            return o.value.shape == ()
    return False


def _role(op: LazyOp) -> Role | None:
    if op.kernel is None or op.func is None:
        return None
    if _is_row_reduce(op):
        return Role.ROW_REDUCE_SUM
    if all(isinstance(o, _ELEM_ALLOWED) for o in op.func.ops):
        # Require real per-element compute (BinOp/UnaryOp/Select/Compare/
        # TernaryOp feeding the Store). Kernels whose Store.value is a raw
        # Load result (strided_copy_4d, strided_copy_3d) are structurally
        # in _ELEM_ALLOWED but their offsets encode a layout change the
        # composer's canonical row-major `pid*N + offs` can't reproduce.
        has_compute = any(
            isinstance(o, (BinOp, UnaryOp, Select, Compare, TernaryOp))
            and o.result is not None
            and o.result.dtype not in ("i32", "i64", "bool")
            for o in op.func.ops
        )
        if not has_compute:
            return None
        return Role.ELEM_FLAT
    return None


# ─── Matcher: grow a row-pass group from a seed ───


@dataclass
class RowPassGroup:
    op_indices: list[int]
    seed_idx: int
    M: int
    N: int


def _shares_alloc(a: AlloyBuffer, b: AlloyBuffer) -> bool:
    try:
        return a.shares_allocation(b)
    except Exception:
        return False


def _find_producer_in_group(
    ops: list[LazyOp], arr: AlloyBuffer, group: set[int]
) -> int | None:
    """Return the index of an op in `group` that writes a buffer sharing
    storage with `arr`, or None."""
    for j in group:
        for pn, out_arr in ops[j].buffer_args:
            if pn in ops[j].output_params and _shares_alloc(out_arr, arr):
                return j
    return None


def grow_group(
    ops: list[LazyOp],
    seed: int,
    planned: set[int],
    roots: set[int] | None = None,
) -> RowPassGroup | None:
    """Expand a group starting at a row_reduce_sum seed.

    Admission rule: an op is added if its role is ELEM_FLAT or
    ROW_REDUCE_SUM, and every one of its inputs either shares storage
    with an op already in the group or has a shape-total compatible
    with the seed's (M, N) layout.
    """
    if seed in planned:
        return None
    if _role(ops[seed]) is not Role.ROW_REDUCE_SUM:
        return None
    cv = ops[seed].constexpr_values
    M = cv.get("M")
    N = cv.get("N")
    if not isinstance(M, int) or not isinstance(N, int):
        return None
    # Two composer paths: single-chunk for N ≤ 1024 (one thread per column),
    # chunked for larger N (column loop with reduce carry). compose_chunked
    # below only handles the common softmax-bwd shape (one Reduce whose input
    # is external FULL_2D, post-reduce elem chain ending in a single
    # terminal-root sink). Cases outside that fall back to per-op.

    group: set[int] = {seed}
    # The reduce's input is external — an upstream LazyOp wrote it. We do
    # NOT pull that upstream op into the group; its output becomes one of
    # the group's external inputs. Starting the group strictly at the
    # reduce keeps it a single-output chain.

    # Build the TRUE producer→consumers DAG via LazyOp.input_producers.
    # shares_allocation() is too permissive because temp buffers get
    # reused across unrelated layers, which causes the walker to absorb
    # the entire training step into one group. input_producers tracks
    # the actual data-flow edges recorded at queue time.
    op_pos = {id(o): i for i, o in enumerate(ops)}
    consumers_of: dict[int, list[int]] = {i: [] for i in range(len(ops))}
    for i, o in enumerate(ops):
        for _pn, producer in o.input_producers.items():
            p = op_pos.get(id(producer))
            if p is not None:
                consumers_of[p].append(i)

    # Admission check: role + shape compatibility.
    def _admittable(k: int) -> bool:
        role_y = _role(ops[k])
        if role_y is None:
            return False
        if role_y is Role.ROW_REDUCE_SUM:
            cv_y = ops[k].constexpr_values
            if cv_y.get("M") != M or cv_y.get("N") != N:
                return False
            return True
        for pn, arr in ops[k].buffer_args:
            if pn in ops[k].output_params:
                continue
            if _classify(arr, M, N) is None:
                return False
        return True

    # Bidirectional fixed-point: grow forward (consumers), then backward
    # (producers of in-group ops). Backward admission is required for shapes
    # like LN-bwd where two independent reduction branches share a common
    # downstream elementwise — forward BFS only captures one branch.
    #
    # MAX_SPAN bounds growth on either side of the seed. The admittability
    # check (role + shape classify) prevents cross-layer absorption; the
    # span cap is a pathological-graph guard, not a correctness gate, and
    # must allow ops BEFORE the seed so backward closure can pull in
    # diamonds like `add → row_mean → sub`.
    MAX_SPAN = 20
    window_lo = max(0, seed - MAX_SPAN)
    window_hi = seed + MAX_SPAN
    changed = True
    while changed:
        changed = False
        # Forward step.
        for x in list(group):
            for y in consumers_of.get(x, []):
                if y in group or y in planned:
                    continue
                if y > window_hi or y < window_lo:
                    continue
                if _admittable(y):
                    group.add(y)
                    changed = True
        # Backward step: admit producers of in-group ops.
        for y in list(group):
            for pname, producer in ops[y].input_producers.items():
                p = op_pos.get(id(producer))
                if p is None or p in group or p in planned:
                    continue
                if p > window_hi or p < window_lo:
                    continue
                if _admittable(p):
                    group.add(p)
                    changed = True

    if len(group) < 3:
        return None
    roles = {_role(ops[i]) for i in group}
    if Role.ELEM_FLAT not in roles or Role.ROW_REDUCE_SUM not in roles:
        return None

    sink_ops = _find_sinks(ops, group, roots or set())
    if not sink_ops:
        return None
    # Multi-output is supported — each sink becomes a separate Store in
    # the fused kernel (either a per-element tile Store for shape (M*N,)
    # outputs or a per-row-scalar Store for shape (M,) outputs). Cap the
    # fan-out to keep param counts manageable.
    if len(sink_ops) > 6:
        return None

    # Sort-order guard. dispatch_entries sorts fused groups by max(op_indices).
    # If an external op inside [lo, hi] reads a sink's output buffer, it
    # gets sorted BEFORE the fused dispatch and the dep-group builder then
    # separates them — the external consumer lands in an earlier group and
    # reads stale memory before the fused kernel writes. Reject in that case.
    # Roots whose only consumers sit at positions > hi (typical for
    # saved-for-bwd tensors) pass this check and fuse correctly.
    if group:
        lo, hi = min(group), max(group)
        sink_out_keys: set = set()
        for si in sink_ops:
            for pn, arr in ops[si].buffer_args:
                if pn in ops[si].output_params:
                    sink_out_keys.add(arr.buffer_key)
        for j in range(lo + 1, hi):
            if j in group:
                continue
            op_j = ops[j]
            bad = False
            for pn, arr in op_j.buffer_args:
                if pn in op_j.output_params:
                    continue
                if arr.buffer_key in sink_out_keys:
                    bad = True
                    break
            if bad:
                return None

    return RowPassGroup(op_indices=sorted(group), seed_idx=seed, M=M, N=N)


def _find_sinks(
    ops: list[LazyOp], group: set[int], roots: set[int] | None = None
) -> list[int]:
    """Ops in `group` whose output must be stored externally.

    A sink is any op in group whose value escapes the group:
      - An op outside the group consumes its output (in-batch fan-out), or
      - It is a root (externally materialized — e.g. saved for backward in a
        later batch). This includes TERMINAL roots (no in-batch consumers)
        and INTERMEDIATE roots (in-batch consumer inside the group — the
        register value flows to downstream in-group ops AND a Store emits
        the per-row-scalar to the root's output buffer).

    Sort-order safety: grow_group rejects a group if any root has an
    external in-batch consumer whose position falls inside
    [min(group), max(group)], which would otherwise trigger the same
    sort-by-max miscompile as task #41.
    """
    op_pos = {id(o): i for i, o in enumerate(ops)}
    sinks: set[int] = set()
    for j, other in enumerate(ops):
        if j in group:
            continue
        for _pn, producer in other.input_producers.items():
            p = op_pos.get(id(producer))
            if p is not None and p in group:
                sinks.add(p)
    if roots:
        for j in group:
            if j in roots:
                sinks.add(j)
    return sorted(sinks)


# ─── Extract per-op compute from TileFunction IR ───


# Constant is included so float literals consumed by the compute chain (e.g.
# the eps in a scale-free RMSNorm's `add(mean, eps)` → `k_add_scalar`) are kept
# in compute_ops and re-emitted in the fused kernel. Without it the backward
# walk dropped the Constant, leaving its operand reference (`c9`) undeclared in
# the emitted MSL (the v_norm fusion compile error). Int/bool constants used as
# index math stay filtered out by the dtype guard below.
_COMPUTE_TYPES = (BinOp, UnaryOp, Select, Compare, TernaryOp, Constant)


@dataclass
class LoadSlot:
    """A Load inside a LazyOp's IR: which param it reads, which TileValue
    it produces, what access kind the buffer has."""
    param_name: str
    value_name: str    # TileValue.name produced by the Load
    kind: AccessKind
    arr: AlloyBuffer


@dataclass
class ExtractedOp:
    role: Role
    # LazyOp's original slot: input buffer args in param order, output arr.
    loads: list[LoadSlot]
    output_param: str
    output_arr: AlloyBuffer
    compute_ops: list[TileOp]   # scalar ops in order (no Loads / Stores / index math)
    stored_value_name: str      # the TileValue fed into the final Store
    # For row_reduce_sum: the Reduce op's input TileValue name (which in
    # the original IR is the ForLoop carry — we reduce over the per-row
    # load in the composed kernel instead).
    reduce_input_param: str | None = None


def _extract_elem(op: LazyOp, M: int, N: int) -> ExtractedOp | None:
    func = op.func
    # Build Load table keyed by param name.
    loads: dict[str, LoadSlot] = {}
    for body_op in func.ops:
        if not isinstance(body_op, Load) or body_op.ptr is None:
            continue
        pn = body_op.ptr.name
        # Find corresponding buffer_args entry.
        arr = None
        for pname, a in op.buffer_args:
            if pname == pn and pname not in op.output_params:
                arr = a
                break
        if arr is None:
            continue
        kind = _classify(arr, M, N)
        if kind is None:
            return None
        loads[pn] = LoadSlot(
            param_name=pn, value_name=body_op.result.name, kind=kind, arr=arr
        )

    # Output
    out_param = None
    out_arr = None
    for pn, arr in op.buffer_args:
        if pn in op.output_params:
            out_param = pn
            out_arr = arr
            break
    if out_param is None or out_arr is None:
        return None

    # Find the Store's source value.
    store_op = next((o for o in func.ops if isinstance(o, Store)), None)
    if store_op is None or store_op.value is None:
        return None
    stored = store_op.value.name

    # Collect compute ops (float dtype, excluding index math) reachable
    # from `stored`.
    name_to_op: dict[str, TileOp] = {
        o.result.name: o for o in func.ops if o.result is not None
    }
    keep: set[str] = set()
    visit = [stored]
    while visit:
        nm = visit.pop()
        if nm in keep or nm not in name_to_op:
            continue
        node = name_to_op[nm]
        if not isinstance(node, _COMPUTE_TYPES):
            continue
        if node.result.dtype in ("i32", "i64", "bool"):
            continue
        keep.add(nm)
        for v in node.operand_values():
            visit.append(v.name)

    compute_ops = [
        o for o in func.ops
        if o.result is not None and o.result.name in keep
    ]

    # Require real per-element compute. Kernels whose Store.value is just a
    # Load result (e.g. strided_copy_4d's layout reshuffle) yield an empty
    # compute chain — the composer would replace their stride-specific Load
    # offsets with its canonical row-major offs_2d and produce wrong data.
    # Rejecting here keeps those kernels out of ROW_PASS groups.
    if not compute_ops:
        return None

    return ExtractedOp(
        role=Role.ELEM_FLAT,
        loads=list(loads.values()),
        output_param=out_param,
        output_arr=out_arr,
        compute_ops=compute_ops,
        stored_value_name=stored,
    )


def _extract_reduce(op: LazyOp, M: int, N: int) -> ExtractedOp | None:
    func = op.func
    # row_reduce_sum kernel: one non-output buffer param (x) read inside
    # ForLoop; output is a per-row scalar.
    in_param = None
    in_arr = None
    for pn, arr in op.buffer_args:
        if pn not in op.output_params:
            in_param = pn
            in_arr = arr
            break
    if in_param is None:
        return None
    kind = _classify(in_arr, M, N)
    if kind is not AccessKind.FULL_2D:
        return None

    out_param = None
    out_arr = None
    for pn, arr in op.buffer_args:
        if pn in op.output_params:
            out_param = pn
            out_arr = arr
            break
    if out_param is None:
        return None

    reduce_op = next((o for o in walk_ops(func.ops) if isinstance(o, Reduce)), None)
    if reduce_op is None or reduce_op.result is None:
        return None

    store_op = next((o for o in func.ops if isinstance(o, Store)), None)
    if store_op is None or store_op.value is None:
        return None

    # Walk backward from Store.value collecting the Reduce and any scalar
    # tail ops (e.g. row_mean's `Div(reduce_result, N)`). We stop recursion
    # AT the Reduce — its input lives inside the ForLoop and is replaced by
    # the compose-local accumulator, so we don't need to pull the loop body
    # ops into the extract.
    name_to_op: dict[str, TileOp] = {
        o.result.name: o for o in walk_ops(func.ops) if o.result is not None
    }
    keep: set[str] = set()
    visit = [store_op.value.name]
    while visit:
        nm = visit.pop()
        if nm in keep or nm not in name_to_op:
            continue
        node = name_to_op[nm]
        if isinstance(node, Reduce):
            keep.add(nm)
            continue  # stop at Reduce — don't recurse into its carry input
        if isinstance(node, (Load, ProgramId, MakeRange, Splat, Copy, ForLoop, Store)):
            continue
        keep.add(nm)
        for v in node.operand_values():
            visit.append(v.name)

    compute_ops = [
        o for o in walk_ops(func.ops)
        if o.result is not None and o.result.name in keep
    ]

    return ExtractedOp(
        role=Role.ROW_REDUCE_SUM,
        loads=[LoadSlot(param_name=in_param, value_name="", kind=kind, arr=in_arr)],
        output_param=out_param,
        output_arr=out_arr,
        compute_ops=compute_ops,
        stored_value_name=store_op.value.name,
        reduce_input_param=in_param,
    )


def extract(op: LazyOp, M: int, N: int) -> ExtractedOp | None:
    role = _role(op)
    if role is Role.ELEM_FLAT:
        return _extract_elem(op, M, N)
    if role is Role.ROW_REDUCE_SUM:
        return _extract_reduce(op, M, N)
    return None


# ─── Composer: build one fused TileFunction ───


def _remap_operands(op: TileOp, remap: dict[str, TileValue]) -> TileOp:
    new_op = copy.copy(op)
    new_op.remap(remap)
    return new_op


class NamedDType:
    name: str


def _dtype_short(dt) -> str:
    name = cast(NamedDType, dt).name.lower() if hasattr(dt, "name") else str(dt).lower()
    return {
        "float32": "f32", "float16": "f16", "bfloat16": "bf16",
        "int32": "i32", "int64": "i64", "int8": "i8", "uint8": "u8",
    }.get(name, "f32")


def _pick_block_size(N: int) -> int:
    v = 1
    while v < N:
        v *= 2
    return max(min(v, 1024), 32)


def _param_tv(p: TileParam) -> TileValue:
    return TileValue(name=p.name, shape=(), layout=Layout.REPLICATED, dtype=p.dtype)


def compose(
    ops: list[LazyOp],
    group: RowPassGroup,
    roots: set[int] | None = None,
) -> tuple[TileFunction, list[AlloyBuffer], tuple[int, int, int]] | None:
    """Build the fused TileFunction. Returns (func, buf_arrs, grid)."""
    M, N = group.M, group.N
    indices = group.op_indices

    # Extract each op's compute descriptor.
    extracted: list[ExtractedOp] = []
    for i in indices:
        ex = extract(ops[i], M, N)
        if ex is None:
            return None
        extracted.append(ex)

    # Identify internal buffers (outputs of ops in the group).
    id_to_group_pos: dict[int, int] = {}
    for pos, ex in enumerate(extracted):
        id_to_group_pos[id(ex.output_arr)] = pos

    # Sinks: ops in the group whose output is consumed outside the group,
    # or which are externally-observed roots.
    sink_lazy_idxs = _find_sinks(ops, set(indices), roots)
    if not sink_lazy_idxs:
        return None
    sink_positions = [indices.index(si) for si in sink_lazy_idxs]

    # Determine internal flows via input_producers (true dataflow). An
    # op's input is "internal" iff its producer is another LazyOp in the
    # same group.
    op_pos_of = {id(o): k for k, o in enumerate(ops)}
    group_set = set(indices)

    def _is_internal_input(lazy_idx: int, param_name: str) -> tuple[bool, int | None]:
        lazy_op = ops[lazy_idx]
        producer = lazy_op.input_producers.get(param_name)
        if producer is None:
            return False, None
        prod_gidx = op_pos_of.get(id(producer))
        if prod_gidx is None or prod_gidx not in group_set:
            return False, None
        return True, indices.index(prod_gidx)

    # Gather unique external inputs (first-encounter order).
    external: list[AlloyBuffer] = []
    external_name: dict[int, str] = {}
    used_names: set[str] = set()

    def _fresh_pname(base: str) -> str:
        n = base
        i = 0
        while n in used_names:
            i += 1
            n = f"{base}_{i}"
        used_names.add(n)
        return n

    for pos, ex in enumerate(extracted):
        lazy_idx = indices[pos]
        for slot in ex.loads:
            internal, _ = _is_internal_input(lazy_idx, slot.param_name)
            if internal:
                continue
            bid = id(slot.arr)
            if bid in external_name:
                continue
            external_name[bid] = _fresh_pname(slot.param_name)
            external.append(slot.arr)

    # Emit one output param per sink.
    sink_outputs: list[tuple[int, AlloyBuffer, str]] = []    # (sink_pos, buffer, param_name)
    for sp in sink_positions:
        sink_outputs.append(
            (sp, extracted[sp].output_arr, _fresh_pname(extracted[sp].output_param))
        )

    # ─── Build the TileFunction ───
    builder = TileBuilder("_fused_row_pass")
    # Params (order matters — buf_arrs indices map 1:1).
    for arr in external:
        dt = _dtype_short(arr.dtype)
        builder.add_param(external_name[id(arr)], is_constexpr=False, dtype=dt)
    for _sp, out_arr, out_pname in sink_outputs:
        builder.add_param(out_pname, is_constexpr=False, dtype=_dtype_short(out_arr.dtype))
    builder.add_param("M", is_constexpr=True)
    builder.add_param("N", is_constexpr=True)
    builder.add_param("BLOCK_SIZE", is_constexpr=True)

    BLOCK_SIZE = _pick_block_size(N)
    builder.set_constexprs({"M": M, "N": N, "BLOCK_SIZE": BLOCK_SIZE})

    # Row-per-threadgroup prologue.
    pid = builder.program_id(0)
    rng = builder.make_range(0, BLOCK_SIZE)
    n_const = builder.constant(N, dtype="i32")
    mask = builder.compare("lt", rng, n_const)
    row_off = builder.binop("mul", pid, n_const)
    offs_2d = builder.binop("add", row_off, rng)

    # Load each external input at its natural rank.
    param_tv = {p.name: _param_tv(p) for p in builder.func.params if not p.is_constexpr}
    loaded: dict[int, TileValue] = {}    # id(AlloyBuffer) → TileValue

    for arr in external:
        pname = external_name[id(arr)]
        ptr = param_tv[pname]
        dt = _dtype_short(arr.dtype)
        kind = _classify(arr, M, N)
        if kind is AccessKind.FULL_2D:
            v = _load_tile(builder, ptr, offs_2d, mask, dt, shape=(BLOCK_SIZE,))
        elif kind is AccessKind.ROW_BCAST:
            v = _load_scalar(builder, ptr, pid, dt)
        elif kind is AccessKind.COL_BCAST:
            v = _load_tile(builder, ptr, rng, mask, dt, shape=(BLOCK_SIZE,))
        elif kind is AccessKind.SCALAR:
            zero = builder.constant(0, dtype="i32")
            v = _load_scalar(builder, ptr, zero, dt)
        else:
            return None
        loaded[id(arr)] = v

    # Walk ops in group order, remap SSA, build the RowPass body.
    rp_ops: list[TileOp] = []
    remap: dict[str, TileValue] = {}
    produces: dict[int, TileValue] = {}    # group pos → final TileValue

    for pos, ex in enumerate(extracted):
        lazy_idx = indices[pos]
        # Seed remap for this op's Loads.
        for slot in ex.loads:
            internal, producer_pos = _is_internal_input(lazy_idx, slot.param_name)
            if internal and producer_pos is not None:
                if producer_pos in produces and slot.value_name:
                    remap[slot.value_name] = produces[producer_pos]
                continue
            ext_tv = loaded.get(id(slot.arr))
            if ext_tv is None:
                return None
            if slot.value_name:
                remap[slot.value_name] = ext_tv

        if ex.role is Role.ELEM_FLAT:
            for old in ex.compute_ops:
                new_op = _remap_operands(old, remap)
                old_res = old.result
                # Each LazyOp's compute was traced with BLOCK_SIZE=1024 (the
                # default for _make_elementwise_*), so old_res.shape is
                # (1024,). In the fused kernel the column width is
                # _pick_block_size(N), which is tighter (e.g. 128 when
                # N=100). Remap the result shape so per-element computes
                # line up with the external Loads we emitted at shape
                # (BLOCK_SIZE,); leave scalar/replicated results (from
                # earlier Reduce outputs feeding BinOp as a broadcast
                # operand) untouched.
                #
                # Scalar-chain ops (e.g. `add(row_mean_result, eps)` followed
                # by `k_rsqrt`) have all operands remapped to scalar TileValues
                # — their result must stay scalar, else a scalar Store at
                # out[pid] would overflow an (M,) buffer by tile writes of
                # size (BLOCK_SIZE,).
                operand_shapes = [v.shape for v in new_op.operand_values()]
                if operand_shapes and all(s == () for s in operand_shapes):
                    new_shape: tuple[int, ...] = ()
                    new_layout = Layout.REPLICATED
                elif old_res.shape and len(old_res.shape) == 1:
                    new_shape = (BLOCK_SIZE,)
                    new_layout = old_res.layout
                else:
                    new_shape = old_res.shape
                    new_layout = old_res.layout
                new_res = TileValue(
                    name=builder._fresh("v"),
                    shape=new_shape,
                    layout=new_layout,
                    dtype=old_res.dtype,
                )
                new_op.result = new_res
                rp_ops.append(new_op)
                remap[old_res.name] = new_res
            produces[pos] = remap[ex.stored_value_name]
        else:
            src_param = ex.reduce_input_param
            slot = next(s for s in ex.loads if s.param_name == src_param)
            internal, producer_pos = _is_internal_input(lazy_idx, src_param)
            if internal and producer_pos is not None:
                reduce_input_tv = produces.get(producer_pos)
            else:
                reduce_input_tv = loaded.get(id(slot.arr))
            if reduce_input_tv is None:
                return None
            # Emit Reduce + any scalar tail (row_mean's Div-by-N, etc.)
            # captured by _extract_reduce. Tail ops reference the Reduce's
            # result by its original IR name; remap to the freshly-created
            # TileValue so operands point at the compose-local version.
            local_remap: dict[str, TileValue] = {}
            last_tv: TileValue | None = None
            for old in ex.compute_ops:
                if isinstance(old, Reduce):
                    new_res = TileValue(
                        name=builder._fresh("v"),
                        shape=(),
                        layout=Layout.REPLICATED,
                        dtype=reduce_input_tv.dtype,
                    )
                    rp_ops.append(
                        Reduce(result=new_res, input=reduce_input_tv,
                               axis=0, op=old.op or "sum")
                    )
                else:
                    new_op = _remap_operands(old, local_remap)
                    old_res = old.result
                    new_res = TileValue(
                        name=builder._fresh("v"),
                        shape=old_res.shape,
                        layout=old_res.layout,
                        dtype=old_res.dtype,
                    )
                    new_op.result = new_res
                    rp_ops.append(new_op)
                local_remap[old.result.name] = new_res
                last_tv = new_res
            # produces[pos] points at the final stored value (last op in the
            # tail), matching ex.stored_value_name.
            produces[pos] = local_remap.get(ex.stored_value_name, last_tv)

    # Emit the RowPass.
    builder.func.add_op(RowPass(ops=rp_ops, writeback=set()))

    # One Store per sink. Tile-shaped values (per-element) use the (M*N,)
    # offset layout and row mask; scalar values (per-row-scalar sinks like
    # row_mean / rstd) store at out[pid].
    for sp, _out_arr, out_pname in sink_outputs:
        out_ptr = param_tv[out_pname]
        val = produces.get(sp)
        if val is None:
            return None
        if val.shape == (BLOCK_SIZE,):
            builder.func.add_op(
                Store(ptr=out_ptr, offsets=offs_2d, value=val, mask=mask)
            )
        elif val.shape == ():
            builder.func.add_op(
                Store(ptr=out_ptr, offsets=pid, value=val)
            )
        else:
            return None

    buf_arrs = list(external) + [sink[1] for sink in sink_outputs]
    grid = (M, 1, 1)
    return builder.func, buf_arrs, grid


# ─── Chunked composer: handles N > BLOCK_SIZE via column-chunk ForLoops ───


def compose_chunked(
    ops: list[LazyOp],
    group: RowPassGroup,
    roots: set[int] | None = None,
) -> tuple[TileFunction, list[AlloyBuffer], tuple[int, int, int]] | None:
    """Multi-chunk variant for N > BLOCK_SIZE.

    Emits:
        pid = program_id
        # init per-reduce scalar accumulators
        for each Reduce:
            init = constant(identity_for_op)
        # phase 1: accumulate across chunks
        ForLoop(_ki, 0, N, BLOCK_SIZE) carried=[init → acc_tile]:
            offs = _ki + make_range(0, BLOCK_SIZE)
            mask = offs < N
            addr = pid * N + offs
            # Load the reduce's input (external FULL_2D)
            v = Load(ptr, addr, mask)
            acc_tile = acc + v       # sum accumulation
        # butterfly each acc_tile → scalar
        reduced = Reduce(acc_tile, axis=0, op="sum")
        # phase 2: emit sink stores using reduced scalars + re-loaded inputs
        ForLoop(_ki, 0, N, BLOCK_SIZE):
            # re-loads, post-reduce elem compute, Store

    MVP scope:
      * One Reduce whose input is an external FULL_2D buffer (no in-group
        pre-compute feeding the reduce).
      * All external inputs are FULL_2D.
      * Sinks are either the Reduce output itself (stored once per row)
        or a downstream elem op producing a (M·N,) tile.
    Anything else: return None so the caller falls back.
    """
    M, N = group.M, group.N
    indices = group.op_indices

    extracted: list[ExtractedOp] = []
    for i in indices:
        ex = extract(ops[i], M, N)
        if ex is None:
            return None
        extracted.append(ex)

    op_pos_of = {id(o): k for k, o in enumerate(ops)}
    group_set = set(indices)

    def _is_internal(lazy_idx: int, param_name: str) -> tuple[bool, int | None]:
        producer = ops[lazy_idx].input_producers.get(param_name)
        if producer is None:
            return False, None
        prod = op_pos_of.get(id(producer))
        if prod is None or prod not in group_set:
            return False, None
        return True, indices.index(prod)

    # Dataflow: for each group position, its in-group producer positions.
    deps: dict[int, list[int]] = {p: [] for p in range(len(extracted))}
    for pos, ex in enumerate(extracted):
        lazy_idx = indices[pos]
        for slot in ex.loads:
            internal, prod_pos = _is_internal(lazy_idx, slot.param_name)
            if internal and prod_pos is not None:
                deps[pos].append(prod_pos)

    reduce_positions = [
        pos for pos, ex in enumerate(extracted) if ex.role is Role.ROW_REDUCE_SUM
    ]
    if not reduce_positions:
        return None  # only worth chunking when a Reduce drives the cost

    # MVP: require reduce inputs come directly from external buffers.
    for rpos in reduce_positions:
        lazy_idx = indices[rpos]
        param = extracted[rpos].reduce_input_param
        internal, _ = _is_internal(lazy_idx, param)
        if internal:
            return None

    # Partition: pre-reduce = {reduce ops themselves}; post-reduce = everything else.
    pre_reduce: set[int] = set(reduce_positions)
    post_reduce: set[int] = set(range(len(extracted))) - pre_reduce

    # Sinks (terminal roots included via _find_sinks).
    sink_lazy_idxs = _find_sinks(ops, set(indices), roots)
    if not sink_lazy_idxs:
        return None
    sink_positions = [indices.index(si) for si in sink_lazy_idxs]

    # MVP: split sinks into "reduce-output" sinks (stored as scalar per row)
    # and "post-reduce-elem" sinks (stored per-chunk via ForLoop).
    post_reduce_sinks = [sp for sp in sink_positions if sp in post_reduce]
    reduce_output_sinks = [sp for sp in sink_positions if sp in reduce_positions]

    # Gather external inputs — require all FULL_2D for the chunked path.
    external: list[AlloyBuffer] = []
    external_name: dict[int, str] = {}
    used_names: set[str] = set()

    def _fresh_pname(base: str) -> str:
        n = base
        i = 0
        while n in used_names:
            i += 1
            n = f"{base}_{i}"
        used_names.add(n)
        return n

    for pos, ex in enumerate(extracted):
        lazy_idx = indices[pos]
        for slot in ex.loads:
            internal, _ = _is_internal(lazy_idx, slot.param_name)
            if internal:
                continue
            if slot.kind is not AccessKind.FULL_2D:
                return None  # MVP
            bid = id(slot.arr)
            if bid in external_name:
                continue
            external_name[bid] = _fresh_pname(slot.param_name)
            external.append(slot.arr)

    sink_outputs: list[tuple[int, AlloyBuffer, str]] = []
    for sp in sink_positions:
        out_arr = extracted[sp].output_arr
        out_pname = _fresh_pname(extracted[sp].output_param)
        sink_outputs.append((sp, out_arr, out_pname))

    # ─── Build the TileFunction ───
    BLOCK_SIZE = 1024
    builder = TileBuilder("_fused_row_pass_chunked")
    for arr in external:
        builder.add_param(
            external_name[id(arr)], is_constexpr=False, dtype=_dtype_short(arr.dtype)
        )
    for _sp, out_arr, out_pname in sink_outputs:
        builder.add_param(
            out_pname, is_constexpr=False, dtype=_dtype_short(out_arr.dtype)
        )
    builder.add_param("M", is_constexpr=True)
    builder.add_param("N", is_constexpr=True)
    builder.add_param("BLOCK_SIZE", is_constexpr=True)
    builder.set_constexprs({"M": M, "N": N, "BLOCK_SIZE": BLOCK_SIZE})

    pid = builder.program_id(0)
    n_const = builder.constant(N, dtype="i32")
    row_off = builder.binop("mul", pid, n_const)

    param_tv = {p.name: _param_tv(p) for p in builder.func.params if not p.is_constexpr}

    # Init per-reduce scalar accumulators (one per reduce).
    reduce_init_scalar: dict[int, TileValue] = {}
    for rpos in reduce_positions:
        init = builder.constant(0.0, dtype="f32")
        reduce_init_scalar[rpos] = init

    # ─── Phase 1 body ───
    phase1_body: list[TileOp] = []

    def _fresh_name(prefix: str) -> str:
        return builder._fresh(prefix)

    loop_var1 = TileValue(name="_ki", shape=(), layout=Layout.REPLICATED, dtype="i32")

    rng1_tv = TileValue(
        name=_fresh_name("rng"), shape=(BLOCK_SIZE,), layout=Layout.BLOCKED, dtype="i32"
    )
    phase1_body.append(MakeRange(result=rng1_tv, start=0, end=BLOCK_SIZE))

    offs1_tv = TileValue(
        name=_fresh_name("offs"), shape=(BLOCK_SIZE,), layout=Layout.BLOCKED, dtype="i32"
    )
    phase1_body.append(BinOp(result=offs1_tv, op="add", lhs=loop_var1, rhs=rng1_tv))

    mask1_tv = TileValue(
        name=_fresh_name("cmp"), shape=(BLOCK_SIZE,), layout=Layout.BLOCKED, dtype="bool"
    )
    phase1_body.append(Compare(result=mask1_tv, op="lt", lhs=offs1_tv, rhs=n_const))

    addr1_tv = TileValue(
        name=_fresh_name("addr"), shape=(BLOCK_SIZE,), layout=Layout.BLOCKED, dtype="i32"
    )
    phase1_body.append(BinOp(result=addr1_tv, op="add", lhs=row_off, rhs=offs1_tv))

    carried: list[tuple[TileValue, TileValue]] = []
    reduce_final_tile: dict[int, TileValue] = {}

    for rpos in reduce_positions:
        ex = extracted[rpos]
        slot = next(s for s in ex.loads if s.param_name == ex.reduce_input_param)
        arr = slot.arr
        pname = external_name[id(arr)]
        ptr = param_tv[pname]
        dt = _dtype_short(arr.dtype)

        load_tv = TileValue(
            name=_fresh_name("ld"), shape=(BLOCK_SIZE,), layout=Layout.BLOCKED, dtype=dt
        )
        phase1_body.append(
            Load(result=load_tv, ptr=ptr, offsets=addr1_tv, mask=mask1_tv, other=0.0)
        )
        init_scalar = reduce_init_scalar[rpos]
        acc_tile = TileValue(
            name=_fresh_name("acc"), shape=(BLOCK_SIZE,), layout=Layout.BLOCKED, dtype="f32"
        )
        # sum: acc = init + v; subsequent iterations: acc = acc_prev + v (carried)
        phase1_body.append(BinOp(result=acc_tile, op="add", lhs=init_scalar, rhs=load_tv))
        carried.append((init_scalar, acc_tile))
        reduce_final_tile[rpos] = acc_tile

    builder.func.add_op(
        ForLoop(var="_ki", start=0, end=N, step=BLOCK_SIZE, body=phase1_body, carried=carried)
    )

    # Butterfly reduce each carry → row scalar, then emit any scalar tail
    # ops captured by _extract_reduce (e.g. row_mean's Div-by-N).
    reduced_scalar: dict[int, TileValue] = {}
    for rpos in reduce_positions:
        final_tile = reduce_final_tile[rpos]
        ex = extracted[rpos]
        local_remap: dict[str, TileValue] = {}
        last_tv: TileValue | None = None
        for old in ex.compute_ops:
            if isinstance(old, Reduce):
                new_res = TileValue(
                    name=_fresh_name("red"), shape=(), layout=Layout.REPLICATED, dtype="f32"
                )
                builder.func.add_op(
                    Reduce(result=new_res, input=final_tile,
                           axis=0, op=old.op or "sum")
                )
            else:
                new_op = _remap_operands(old, local_remap)
                old_res = old.result
                new_res = TileValue(
                    name=_fresh_name("v"),
                    shape=old_res.shape,
                    layout=old_res.layout,
                    dtype=old_res.dtype,
                )
                new_op.result = new_res
                builder.func.add_op(new_op)
            local_remap[old.result.name] = new_res
            last_tv = new_res
        reduced_scalar[rpos] = local_remap.get(ex.stored_value_name, last_tv)

    # Store reduce-output sinks directly at out[pid] (no loop).
    for sp, _out_arr, out_pname in sink_outputs:
        if sp not in reduce_output_sinks:
            continue
        out_ptr = param_tv[out_pname]
        builder.func.add_op(
            Store(ptr=out_ptr, offsets=pid, value=reduced_scalar[sp])
        )

    # ─── Phase 2: post-reduce elem compute + Store per chunk ───
    if post_reduce_sinks:
        phase2_body: list[TileOp] = []
        loop_var2 = TileValue(name="_kj", shape=(), layout=Layout.REPLICATED, dtype="i32")

        rng2 = TileValue(
            name=_fresh_name("rng"), shape=(BLOCK_SIZE,), layout=Layout.BLOCKED, dtype="i32"
        )
        phase2_body.append(MakeRange(result=rng2, start=0, end=BLOCK_SIZE))

        offs2 = TileValue(
            name=_fresh_name("offs"), shape=(BLOCK_SIZE,), layout=Layout.BLOCKED, dtype="i32"
        )
        phase2_body.append(BinOp(result=offs2, op="add", lhs=loop_var2, rhs=rng2))

        mask2 = TileValue(
            name=_fresh_name("cmp"), shape=(BLOCK_SIZE,), layout=Layout.BLOCKED, dtype="bool"
        )
        phase2_body.append(Compare(result=mask2, op="lt", lhs=offs2, rhs=n_const))

        addr2 = TileValue(
            name=_fresh_name("addr"), shape=(BLOCK_SIZE,), layout=Layout.BLOCKED, dtype="i32"
        )
        phase2_body.append(BinOp(result=addr2, op="add", lhs=row_off, rhs=offs2))

        # Re-load every external input the post-reduce ops need.
        ext_load_tv: dict[int, TileValue] = {}  # id(arr) → Load TileValue
        for arr in external:
            pname = external_name[id(arr)]
            ptr = param_tv[pname]
            dt = _dtype_short(arr.dtype)
            ld = TileValue(
                name=_fresh_name("ld"), shape=(BLOCK_SIZE,), layout=Layout.BLOCKED, dtype=dt
            )
            phase2_body.append(
                Load(result=ld, ptr=ptr, offsets=addr2, mask=mask2, other=0.0)
            )
            ext_load_tv[id(arr)] = ld

        # Remap and emit compute for post-reduce ops in topological order.
        remap: dict[str, TileValue] = {}
        produces: dict[int, TileValue] = {}
        for pos in sorted(post_reduce):
            ex = extracted[pos]
            lazy_idx = indices[pos]
            for slot in ex.loads:
                internal, producer_pos = _is_internal(lazy_idx, slot.param_name)
                if internal and producer_pos is not None:
                    if producer_pos in produces and slot.value_name:
                        remap[slot.value_name] = produces[producer_pos]
                    elif producer_pos in reduced_scalar and slot.value_name:
                        remap[slot.value_name] = reduced_scalar[producer_pos]
                    continue
                ext_tv = ext_load_tv.get(id(slot.arr))
                if ext_tv is None:
                    return None
                if slot.value_name:
                    remap[slot.value_name] = ext_tv

            if ex.role is not Role.ELEM_FLAT:
                return None  # MVP: reduces only appear in pre_reduce

            for old in ex.compute_ops:
                new_op = _remap_operands(old, remap)
                old_res = old.result
                if old_res.shape and len(old_res.shape) == 1:
                    new_shape = (BLOCK_SIZE,)
                else:
                    new_shape = old_res.shape
                new_res = TileValue(
                    name=_fresh_name("v"),
                    shape=new_shape,
                    layout=old_res.layout,
                    dtype=old_res.dtype,
                )
                new_op.result = new_res
                phase2_body.append(new_op)
                remap[old_res.name] = new_res
            produces[pos] = remap[ex.stored_value_name]

        # Stores inside phase 2.
        for sp, _oa, out_pname in sink_outputs:
            if sp not in post_reduce_sinks:
                continue
            out_ptr = param_tv[out_pname]
            val = produces.get(sp)
            if val is None or val.shape != (BLOCK_SIZE,):
                return None
            phase2_body.append(
                Store(ptr=out_ptr, offsets=addr2, value=val, mask=mask2)
            )

        builder.func.add_op(
            ForLoop(var="_kj", start=0, end=N, step=BLOCK_SIZE, body=phase2_body, carried=[])
        )

    buf_arrs = list(external) + [sink[1] for sink in sink_outputs]
    grid = (M, 1, 1)
    return builder.func, buf_arrs, grid


# ─── Load helpers (manual — mirror TileBuilder API with explicit shape) ───


def _load_tile(
    builder: TileBuilder, ptr: TileValue, offsets: TileValue,
    mask: TileValue, dtype: str, shape: tuple[int, ...]
) -> TileValue:
    v = TileValue(
        name=builder._fresh("ld"),
        shape=shape,
        layout=Layout.BLOCKED,
        dtype=dtype,
    )
    builder.func.add_op(
        Load(result=v, ptr=ptr, offsets=offsets, mask=mask, other=0.0)
    )
    return v


def _load_scalar(
    builder: TileBuilder, ptr: TileValue, offset: TileValue, dtype: str
) -> TileValue:
    v = TileValue(
        name=builder._fresh("ld"),
        shape=(),
        layout=Layout.REPLICATED,
        dtype=dtype,
    )
    builder.func.add_op(Load(result=v, ptr=ptr, offsets=offset))
    return v
