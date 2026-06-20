"""Multi-root elementwise fusion.

Generalizes ELEM_CHAIN (linear primary-input chain) to a sub-DAG of
elementwise ops that share a launch grid. Typical case: optimizer update
steps (AdamW) where three sibling chains (m_new, v_new, param_new) read
shared inputs (grad, m, v, param, scalars) and produce multiple mutated
outputs at the same per-element grid.

Pipeline:
    find_islands(ops) → list[Island]      # matcher (DAG analysis)
    compile_multi_root(queue, island)     # composer (IR synthesis)

Contract:
    * All island ops produce tiles with the same canonical launch geometry.
    * Intermediate values consumed outside the island are emitted as explicit
      Stores (writebacks). Every mutation-root is a writeback.
    * Island ops are topologically ordered. Each op reads either in-island
      producer values (from registers, via substitution) or external
      buffers (via fresh Loads added to the scaffold).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from alloy._compiler.tile_ir import (
    Compare,
    Constant,
    Layout,
    Load,
    Splat,
    Store,
    TileFunction,
    TileParam,
    TileValue,
    shallow_clone_for_fusion,
)
from alloy._dispatch.buf_utils import _NON_FUSABLE_ELEM_KERNELS, _normalize_grid
from alloy._dispatch.fusion_transform import (
    IRSubstitution,
    _buffer_dtype_name,
    _build_buf_map,
    _extract_and_lower,
    _find_ir_store,
    _resolve_extra_arr,
    _unique_name,
)
from alloy._dispatch.fusion_types import FusionUnsupported
from alloy._dispatch.lazy import LazyOp
from alloy._runtime.alloy_buffer import AlloyBuffer


# ===================================================================
# Section 1: Matcher
# ===================================================================


@dataclass
class Island:
    """A connected sub-DAG of elementwise ops sharing a launch grid."""

    indices: list[int]
    writebacks: set[int] = field(default_factory=set)
    order: list[int] = field(default_factory=list)


def _flat_elems(shape: tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


def _canonical_grid(op: LazyOp) -> tuple[int, ...]:
    """Bucket ops by launch geometry.

    Two elementwise ops fuse iff each output element index maps to the same
    (thread_x, thread_y, thread_z) lane. The launch grid × output element
    count captures this. Scalar-shape ops (output numel == 1) get their own
    bucket so they stay in the existing small-chain ELEM_CHAIN path.
    """
    out_shape: tuple[int, ...] = ()
    for pn in op.output_params:
        s = op.buffer_shapes.get(pn)
        if s is not None:
            out_shape = tuple(int(d) for d in s)
            break
    flat = _flat_elems(out_shape) if out_shape else 0
    return (tuple(int(g) for g in op.grid), flat)


def _is_fusable_elem(op: LazyOp) -> bool:
    if not op.is_elem_op():
        return False
    k = op.kernel
    if k is None:
        return False
    if k.name in _NON_FUSABLE_ELEM_KERNELS:
        return False
    return True


def _op_externals_emittable(
    op: LazyOp,
    op_idx: dict[int, int],
    island: set[int],
    island_flat: int,
) -> bool:
    """Check this op's external inputs (those NOT produced in-island) for emittability.

    The composer can load per-element buffers (flat matches the island) and
    scalar broadcasts (flat == 1). Row/column/strided broadcasts aren't
    synthesized — islands that would need them get skipped so the offending
    op stays as an individual dispatch (or a smaller island forms around it).
    """
    for pname, arr in op.buffer_args:
        if pname in op.output_params:
            continue
        producer = op.input_producers.get(pname)
        if producer is not None and op_idx.get(id(producer)) in island:
            continue
        flat = 1 if arr._strides and any(s == 0 for s in arr._strides) else arr.size
        if flat != 1 and flat != island_flat:
            return False
    return True


def _grow_island(
    ops: list[LazyOp],
    op_idx: dict[int, int],
    consumed_by: dict[int, list[int]],
    planned: set[int],
    seed: int,
) -> set[int]:
    """BFS from seed over elem producers + consumers sharing the canonical grid.

    Skips ops whose external inputs use broadcast patterns the composer
    can't synthesize (anything beyond per-element or scalar). A rejected
    op stops the walk at that boundary — we keep the rest of the island
    and let the producer/consumer dispatch as a separate kernel.
    """
    target = _canonical_grid(ops[seed])
    if target[1] <= 1:
        return set()  # scalar bucket — leave to ELEM_CHAIN
    island_flat = target[1]
    island: set[int] = set()
    stack = [seed]
    while stack:
        i = stack.pop()
        if i in island or i in planned:
            continue
        if not _is_fusable_elem(ops[i]):
            continue
        if _canonical_grid(ops[i]) != target:
            continue
        if not _op_externals_emittable(ops[i], op_idx, island, island_flat):
            continue
        island.add(i)
        for producer in ops[i].input_producers.values():
            pi = op_idx.get(id(producer))
            if pi is None:
                continue
            if pi not in island:
                stack.append(pi)
            # Siblings sharing this producer are reachable even when the producer
            # itself is non-fusable (e.g. cos & sin both read the `cat`/emb
            # output, whose only common node is the non-fusable concat). The
            # loop-top fusability/grid guards drop the non-fusable consumers.
            for sib in consumed_by.get(pi, []):
                if sib not in island:
                    stack.append(sib)
        for c in consumed_by.get(i, []):
            if c not in island:
                stack.append(c)
    return island


def _topo_sort(
    ops: list[LazyOp],
    island: set[int],
    op_idx: dict[int, int],
) -> list[int] | None:
    """Kahn's algorithm restricted to the island. Returns None on cycle."""
    in_deg: dict[int, int] = {i: 0 for i in island}
    deps: dict[int, list[int]] = defaultdict(list)
    for i in island:
        for producer in ops[i].input_producers.values():
            pi = op_idx.get(id(producer))
            if pi is not None and pi in island:
                deps[pi].append(i)
                in_deg[i] += 1
    order: list[int] = []
    ready = sorted([i for i, d in in_deg.items() if d == 0])
    while ready:
        i = ready.pop(0)
        order.append(i)
        for j in deps[i]:
            in_deg[j] -= 1
            if in_deg[j] == 0:
                ready.append(j)
        ready.sort()
    if len(order) != len(island):
        return None
    return order


def _writebacks(
    ops: list[LazyOp],
    island: set[int],
    consumed_by: dict[int, list[int]],
    roots: set[int],
) -> set[int]:
    """Island members whose output must materialize into a buffer."""
    wb: set[int] = set()
    for i in island:
        if i in roots:
            wb.add(i)
            continue
        for c in consumed_by.get(i, []):
            if c not in island:
                wb.add(i)
                break
    return wb


def _externals_emittable(
    ops: list[LazyOp],
    island: set[int],
    op_idx: dict[int, int],
    island_flat: int,
) -> bool:
    """Every external input to an island op must be loadable by the composer.

    The composer only knows two Load patterns: per-element (flat matches the
    island) and scalar broadcast (flat == 1 via stride-0 dims). Anything else
    — row/column broadcast of a partial tile, strided view, etc. — would
    require replicating the per-op IndexTransform machinery inside the fused
    kernel, which the current composer doesn't do.
    """
    for i in island:
        op = ops[i]
        for pname, arr in op.buffer_args:
            if pname in op.output_params:
                continue
            producer = op.input_producers.get(pname)
            if producer is not None and op_idx.get(id(producer)) in island:
                continue  # in-island producer — loaded from registers
            flat = 1 if arr._strides and any(s == 0 for s in arr._strides) else arr.size
            if flat != 1 and flat != island_flat:
                return False
    return True


_METAL_MAX_BUFFERS = 30  # Metal kernels support buffers 0..30 (31 total)


def _estimate_param_count(
    ops: list[LazyOp],
    island: set[int],
    op_idx: dict[int, int],
    writebacks: set[int],
) -> int:
    """Count unique external inputs + writebacks this island would materialize.

    Used as a cap before accepting an island: Metal kernels can bind at most
    31 buffers, so islands producing 100+ params aren't compilable.
    """
    seen_buffers: set[tuple[int, int]] = set()
    for i in island:
        op = ops[i]
        for pname, arr in op.buffer_args:
            if pname in op.output_params:
                continue
            producer = op.input_producers.get(pname)
            if producer is not None and op_idx.get(id(producer)) in island:
                continue
            seen_buffers.add(arr.buffer_key)
    wb_buffers: set[tuple[int, int]] = set()
    for i in writebacks:
        op = ops[i]
        for pname, arr in op.buffer_args:
            if pname in op.output_params:
                wb_buffers.add(arr.buffer_key)
                break
    return len(seen_buffers) + len(wb_buffers)


def find_islands(
    ops: list[LazyOp],
    op_idx: dict[int, int],
    consumed_by: dict[int, list[int]],
    planned: set[int],
    roots: set[int],
    *,
    min_size: int = 3,
    max_params: int = _METAL_MAX_BUFFERS,
) -> list[Island]:
    """Enumerate maximal MULTI_ROOT islands."""
    visited: set[int] = set()
    islands: list[Island] = []
    for i in range(len(ops)):
        if i in visited or i in planned:
            continue
        if not _is_fusable_elem(ops[i]):
            continue
        island = _grow_island(ops, op_idx, consumed_by, planned, i)
        if not island:
            continue
        visited.update(island)
        wb = _writebacks(ops, island, consumed_by, roots)
        if not wb:
            continue  # nothing observable — dead island
        # min_size leaves small LINEAR chains (1 writeback) to ELEM_CHAIN, but a
        # multi-OUTPUT fan-out (>=2 writebacks) — e.g. sincos sharing one emb
        # load — can't be expressed by ELEM_CHAIN's single-output linear model,
        # so admit those below min_size.
        if len(island) < min_size and len(wb) < 2:
            continue
        order = _topo_sort(ops, island, op_idx)
        if order is None:
            continue  # cyclic — can't fuse
        island_flat = _canonical_grid(ops[next(iter(island))])[1]
        if not _externals_emittable(ops, island, op_idx, island_flat):
            continue
        # Metal caps total buffer params per kernel. When an island crosses
        # the cap, drop it — a later pass (e.g. when some member gets pulled
        # into an anchor group) will expose smaller sub-islands.
        if _estimate_param_count(ops, island, op_idx, wb) > max_params:
            continue
        # Sort-order guard: dispatch_entries are sorted by max(island_indices)
        # for fused groups. If any external op inside [min(island), max(island)]
        # reads a writeback's output buffer, it gets sorted BEFORE the fused
        # dispatch and the dep-group builder separates them — the external
        # consumer would land in an earlier group and read stale memory before
        # the fused kernel writes.
        lo, hi = order[0], order[-1]
        wb_out_keys: set = set()
        for wb_idx in wb:
            for pn, arr in ops[wb_idx].buffer_args:
                if pn in ops[wb_idx].output_params:
                    wb_out_keys.add(arr.buffer_key)
        unorderable = False
        for j in range(lo + 1, hi):
            if j in island:
                continue
            for pn, arr in ops[j].buffer_args:
                if pn in ops[j].output_params:
                    continue
                if arr.buffer_key in wb_out_keys:
                    unorderable = True
                    break
            if unorderable:
                break
        if unorderable:
            continue
        islands.append(Island(indices=sorted(island), writebacks=wb, order=order))
    return islands


# ===================================================================
# Section 2: Composer
# ===================================================================


def _output_buf(op: LazyOp) -> AlloyBuffer | None:
    buf_args = dict(op.buffer_args)
    for pn in op.output_params:
        b = buf_args.get(pn)
        if b is not None:
            return b
    return None


def _find_preamble_values(func: TileFunction) -> tuple[TileValue, TileValue]:
    """Locate (offsets, mask) TileValues in a lowered elem func.

    Every elementwise kernel has the same preamble shape: ProgramId → scale
    by BLOCK_SIZE → add MakeRange → offsets; Compare(offsets < N) → mask.
    """
    for op in func.ops:
        if isinstance(op, Compare) and op.op == "lt" and op.result is not None:
            if op.lhs is None:
                continue
            return op.lhs, op.result
    raise FusionUnsupported("multi-root: seed has no preamble offsets/mask")


def _insert_zero_splat(
    func: TileFunction, insert_idx: int, tile_shape: tuple[int, ...], taken: set[str]
) -> TileValue:
    """Emit Constant(0) + Splat(0, shape) and return the splatted TileValue."""
    c_name = _unique_name("mr_c0", taken)
    taken.add(c_name)
    s_name = _unique_name("mr_spl0", taken)
    taken.add(s_name)
    c_tv = TileValue(c_name, (), Layout.REPLICATED, "i32")
    s_tv = TileValue(s_name, tile_shape, Layout.BLOCKED, "i32")
    func.ops.insert(insert_idx, Constant(result=c_tv, value=0))
    func.ops.insert(insert_idx + 1, Splat(result=s_tv, value=c_tv, shape=tile_shape))
    return s_tv


def _index_after_preamble(func: TileFunction) -> int:
    """Return the index into func.ops just after the bounds-check Compare."""
    for i, op in enumerate(func.ops):
        if isinstance(op, Compare) and op.op == "lt":
            return i + 1
    return len(func.ops)


def _arr_flat(arr: AlloyBuffer) -> int:
    """Effective element count — 1 for stride-0 broadcasts, else total numel."""
    if arr._strides and any(s == 0 for s in arr._strides):
        return 1
    return arr.size


def _emit_external_load(
    func: TileFunction,
    param_name: str,
    arr: AlloyBuffer,
    offsets: TileValue,
    mask: TileValue,
    tile_shape: tuple[int, ...],
    island_flat: int,
    taken_names: set[str],
) -> TileValue:
    """Emit Load(ptr=param_name, ...) into func. Per-element vs scalar."""
    dtype_name = _buffer_dtype_name(arr)
    ptr_tv = TileValue(param_name, (), Layout.REPLICATED, dtype_name)

    input_flat = _arr_flat(arr)
    insert_at = _index_after_preamble(func)

    if input_flat == island_flat:
        use_offsets = offsets
    elif input_flat == 1:
        use_offsets = _insert_zero_splat(func, insert_at, tile_shape, taken_names)
        insert_at += 2
    else:
        raise FusionUnsupported(
            f"multi-root: external input flat={input_flat} doesn't match "
            f"island flat={island_flat} and isn't scalar"
        )

    res_name = _unique_name("mr_ld", taken_names)
    taken_names.add(res_name)
    res_tv = TileValue(res_name, tile_shape, Layout.BLOCKED, dtype_name)
    load_op = Load(
        result=res_tv,
        ptr=ptr_tv,
        offsets=use_offsets,
        mask=mask,
        other=0.0,
        transform=[],
        transform_extras={},
        transform_source_name=None,
        row_indices=None,
        col_indices=None,
        row_stride=None,
        base_offset=None,
        addr_transposed=False,
        pack_factor=0,
        pack_bits=0,
        dequant_scale_ptr=None,
        dequant_zero_point=0.0,
        dequant_n_groups=0,
    )
    func.ops.insert(insert_at, load_op)
    return res_tv


def _find_existing_load_result(func: TileFunction, ptr_name: str) -> TileValue | None:
    for op in func.ops:
        if (
            isinstance(op, Load)
            and op.ptr is not None
            and op.ptr.name == ptr_name
            and op.result is not None
        ):
            return op.result
    return None


def _param_for_buffer(
    buffer_map: dict[str, AlloyBuffer],
    arr: AlloyBuffer,
) -> str | None:
    for pname, bound in buffer_map.items():
        if bound is arr or bound.shares_allocation(arr):
            return pname
    return None


def _add_param(
    func: TileFunction,
    buffer_map: dict[str, AlloyBuffer],
    arr: AlloyBuffer,
    taken_names: set[str],
    base: str,
) -> str:
    name = _unique_name(base, taken_names)
    taken_names.add(name)
    func.params.append(TileParam(name=name, is_constexpr=False, dtype=_buffer_dtype_name(arr)))
    buffer_map[name] = arr
    return name


def _emit_store(
    func: TileFunction,
    insert_idx: int,
    param_name: str,
    arr: AlloyBuffer,
    value: TileValue,
    offsets: TileValue,
    mask: TileValue,
) -> None:
    dtype_name = _buffer_dtype_name(arr)
    ptr_tv = TileValue(param_name, (), Layout.REPLICATED, dtype_name)
    store = Store(
        result=None,
        ptr=ptr_tv,
        offsets=offsets,
        value=value,
        mask=mask,
        transform=[],
        transform_extras={},
        transform_source_name=None,
        col_slice=None,
        row_indices=None,
        col_indices=None,
        row_stride=None,
        base_offset=None,
    )
    func.ops.insert(insert_idx, store)


def compile_multi_root(
    queue: list[LazyOp],
    island: Island,
) -> tuple[TileFunction, list[AlloyBuffer], tuple[int, int, int], None]:
    """Compile a MULTI_ROOT island into a single fused TileFunction.

    Returns (func, buffer_list, grid, None) matching compile_elem_chain.
    """
    order = island.order
    writebacks = island.writebacks

    seed_idx = order[0]
    seed_op = queue[seed_idx]

    func = shallow_clone_for_fusion(seed_op.func)
    func.name = "_fused_multi_root"
    buffer_map = _build_buf_map(seed_op, func)
    grid = _normalize_grid(seed_op.grid)

    offsets, mask = _find_preamble_values(func)
    tile_shape = offsets.shape

    # Seed's Store: extract value, then drop. Stores are re-emitted for every
    # writeback at the tail; the matcher ensures the seed is not itself a
    # writeback, so dropping here is safe.
    seed_store = _find_ir_store(func)
    if seed_store is None or seed_store.value is None:
        raise FusionUnsupported("multi-root: seed op has no Store")
    seed_store_value = seed_store.value
    func.ops = [o for o in func.ops if o is not seed_store]

    taken_names: set[str] = {op.result.name for op in func.ops if op.result is not None}
    taken_names.update(p.name for p in func.params)

    alloc_resolved: dict[tuple[int, int], TileValue] = {}
    seed_out = _output_buf(seed_op)
    if seed_out is not None:
        alloc_resolved[seed_out.buffer_key] = seed_store_value

    island_flat = _canonical_grid(seed_op)[1]

    def _resolve(arr: AlloyBuffer) -> TileValue:
        hit = alloc_resolved.get(arr.buffer_key)
        if hit is not None:
            return hit
        existing_pname = _param_for_buffer(buffer_map, arr)
        if existing_pname is not None:
            exist_tv = _find_existing_load_result(func, existing_pname)
            if exist_tv is not None:
                return exist_tv
            return _emit_external_load(
                func,
                existing_pname,
                arr,
                offsets,
                mask,
                tile_shape,
                island_flat,
                taken_names,
            )
        new_pname = _add_param(func, buffer_map, arr, taken_names, "mr_in")
        return _emit_external_load(
            func,
            new_pname,
            arr,
            offsets,
            mask,
            tile_shape,
            island_flat,
            taken_names,
        )

    sub = IRSubstitution()

    for pos in range(1, len(order)):
        op = queue[order[pos]]
        _sub_func, xf, sub_buf_map = _extract_and_lower(op, pos)

        step_keys: list[str] = []

        primary_arr = sub_buf_map.get(xf.input_param_name)
        if primary_arr is None:
            raise FusionUnsupported(f"multi-root: op {order[pos]} missing primary buffer")
        primary_resolved = _resolve(primary_arr)
        sub.add(xf.input_value_name, primary_resolved)
        step_keys.append(xf.input_value_name)
        # If remap aliased the primary ptr inside xf.ops (rare but possible
        # when the primary also appeared as an extra), catch that too.
        if xf.input_param_name != xf.input_value_name:
            sub.add(xf.input_param_name, primary_resolved)
            step_keys.append(xf.input_param_name)

        for ename, eval_tv in xf.extra_inputs.items():
            earr = _resolve_extra_arr(ename, sub_buf_map, xf.input_param_name)
            if earr is None:
                raise FusionUnsupported(
                    f"multi-root: unresolved extra '{ename}' on op {order[pos]}"
                )
            # After extract_ir_transform remaps extra Load results → ptr
            # TileValues, the transform_ops reference the param name (ename).
            # IRSubstitution matches by TileValue.name, so key on ename.
            resolved_tv = _resolve(earr)
            sub.add(ename, resolved_tv)
            step_keys.append(ename)
            if eval_tv is not None and eval_tv.name != ename:
                sub.add(eval_tv.name, resolved_tv)
                step_keys.append(eval_tv.name)

        prefix = f"mr{pos}"
        step_ops = sub.apply_ops(xf.ops, result_prefix=prefix)
        for sop in step_ops:
            if sop.result is not None:
                taken_names.add(sop.result.name)

        func.ops.extend(step_ops)

        out_tv = step_ops[-1].result if step_ops else None
        out_arr = _output_buf(op)
        if out_tv is not None and out_arr is not None:
            alloc_resolved[out_arr.buffer_key] = out_tv

        # Scope substitutions to THIS step — local param names (x, y) can
        # recur in subsequent steps with different bindings.
        for k in step_keys:
            sub._map.pop(k, None)

    # Phase 3: emit writeback Stores
    wb_list = sorted(writebacks, key=lambda i: order.index(i))
    for wb_idx in wb_list:
        op = queue[wb_idx]
        out_arr = _output_buf(op)
        if out_arr is None:
            raise FusionUnsupported(f"multi-root: writeback op {wb_idx} has no output")
        value_tv = alloc_resolved.get(out_arr.buffer_key)
        if value_tv is None:
            raise FusionUnsupported(f"multi-root: writeback op {wb_idx} has no resolved TileValue")
        existing = _param_for_buffer(buffer_map, out_arr)
        if existing is not None:
            param_name = existing
        else:
            param_name = _add_param(func, buffer_map, out_arr, taken_names, "mr_out")
        _emit_store(func, len(func.ops), param_name, out_arr, value_tv, offsets, mask)

    # Drop buffer params that aren't referenced by any Load/Store (seed's
    # original output is orphaned when the seed isn't a writeback).
    referenced: set[str] = set()
    for op in func.ops:
        if isinstance(op, (Load, Store)) and op.ptr is not None:
            referenced.add(op.ptr.name)
    func.params = [p for p in func.params if p.is_constexpr or p.name in referenced]

    buf_arrs = [buffer_map[p.name] for p in func.params if not p.is_constexpr]
    return func, buf_arrs, grid, None
