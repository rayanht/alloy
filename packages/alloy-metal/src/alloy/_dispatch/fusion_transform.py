"""Fusion IR transforms — extraction, chain composition, and anchor context.

Operates on TileFunction IR: extracts transforms from elementwise kernels,
composes transform chains, and attaches them to anchor Load/Store ops.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field

from alloy._compiler.tile_ir import (
    ColumnSliceInfo,
    ExpandDims,
    ForLoop,
    IfElse,
    Layout,
    Load,
    MakeRange,
    ProgramId,
    Splat,
    Store,
    TileFunction,
    TileOp,
    TileParam,
    TileValue,
    WhileLoop,
    shallow_clone_for_fusion,
    walk_ops,
)
from alloy._compiler.fusion_transforms import (
    BroadcastTransform,
    ColumnBroadcastTransform,
    IdentityTransform,
    IndexTransform,
    RowBroadcastTransform,
    ScalarBroadcastTransform,
    ScatterTransform,
    StridedTransform,
)
from alloy._dispatch.fusion_types import FusionUnsupported
from alloy._dispatch.lazy import LazyOp
from alloy._runtime.alloy_buffer import AlloyBuffer


# ===================================================================
# Section 4: IR extraction (from lowered elem TileFunctions)
# ===================================================================


def extract_ir_transform(
    elem_func: TileFunction,
) -> (
    tuple[list[TileOp], str, TileValue | None, str | None, str | None, dict[str, TileValue | None]]
    | None
):
    """Extract the computation between Load and Store in a lowered elem TileFunction.

    Returns (transform_ops, load_result_name, store_value, load_ptr_name,
             store_ptr_name, extra_load_ptrs).
    """
    op_map: dict[str, TileOp] = {}
    loads: list[Load] = []
    store: Store | None = None
    for op in elem_func.ops:
        if op.result:
            op_map[op.result.name] = op
        if isinstance(op, Load):
            loads.append(op)
        if isinstance(op, Store):
            store = op

    if store is None or not loads:
        return None

    def _deps(name: str, visited: set[str] | None = None) -> set[str]:
        if visited is None:
            visited = set()
        if name in visited:
            return set()
        visited.add(name)
        op = op_map.get(name)
        if op is None or isinstance(op, (ProgramId, MakeRange)):
            return set()
        if isinstance(op, Load):
            return {name}
        result: set[str] = set()
        for v in op.operand_values():
            result |= _deps(v.name, visited)
        return result

    load_deps: set[str] = _deps(store.value.name) if store.value else set()

    shared_load: Load | None = None
    extra_loads: list[Load] = []
    for ld in loads:
        if ld.result and ld.result.name in load_deps:
            if shared_load is None:
                shared_load = ld
            else:
                extra_loads.append(ld)

    if shared_load is None:
        return None

    compute_names: set[str] = set()

    def _collect(name: str) -> None:
        if name in compute_names:
            return
        op = op_map.get(name)
        if op is None:
            return
        if isinstance(op, (Load, ProgramId, MakeRange, ExpandDims, Splat)):
            return
        compute_names.add(name)
        for v in op.operand_values():
            _collect(v.name)

    if store.value:
        _collect(store.value.name)

    # Clone each extracted op so downstream remaps mutate the copy, not the
    # shared IR. copy.copy suffices: these are elementwise ops whose fields
    # are scalar TileValue refs — no shared list/dict state.
    transform_ops: list[TileOp] = [
        copy.copy(op) for op in elem_func.ops if op.result and op.result.name in compute_names
    ]

    extra_load_ptrs: dict[str, TileValue | None] = {}
    if extra_loads:
        shared_ptr = shared_load.ptr.name if shared_load.ptr else None
        remap: dict[str, TileValue] = {}
        for ld in extra_loads:
            ptr_name = ld.ptr.name if ld.ptr else None
            if ptr_name == shared_ptr:
                alias = f"_xtra_{ld.result.name}"
                alias_tv = TileValue(
                    alias,
                    ld.ptr.shape if ld.ptr else (),
                    ld.ptr.layout if ld.ptr else Layout.REPLICATED,
                    ld.ptr.dtype if ld.ptr else "f32",
                )
                extra_load_ptrs[alias] = ld.result
                remap[ld.result.name] = alias_tv
            else:
                extra_load_ptrs[ptr_name] = ld.result
                remap[ld.result.name] = ld.ptr
        for op in transform_ops:
            op.remap(remap)

    return (
        transform_ops,
        shared_load.result.name,
        store.value,
        shared_load.ptr.name if shared_load.ptr else None,
        store.ptr.name if store.ptr else None,
        extra_load_ptrs,
    )


def inject_prologue_ir(
    anchor_func: TileFunction, elem_func: TileFunction, anchor_input_param: str
) -> bool:
    """Fuse a lowered elem as prologue on an anchor's Load."""
    result = extract_ir_transform(elem_func)
    if result is None:
        return False
    transform_ops, _, _, _, _, _ = result

    found = False
    first = True
    for op in walk_ops(anchor_func.ops):
        if isinstance(op, Load) and op.ptr and op.ptr.name == anchor_input_param:
            if first:
                op.transform = transform_ops
                first = False
            else:
                op.transform = [copy.copy(t) for t in transform_ops]
            found = True
    return found


# ===================================================================
# Section 5: Buffer/IR helpers for chain composition
# ===================================================================


def _buf_elem_count(arr: AlloyBuffer) -> int:
    """Physical element count — accounts for broadcast (stride-0) buffers."""
    if arr.ndim and arr._strides:
        if all(s == 0 for s in arr._strides):
            return 1
        max_offset = 0
        for dim_size, stride in zip(arr._shape, arr._strides):
            if dim_size > 1 and stride > 0:
                max_offset += (dim_size - 1) * stride
        return max_offset // arr.dtype.itemsize + 1 if max_offset > 0 else 1
    return arr.size


def _find_ir_store(func: TileFunction) -> Store | None:
    for op in walk_ops(func.ops):
        if isinstance(op, Store):
            return op
    return None


def _buffer_dtype_name(arr: AlloyBuffer) -> str:
    return arr._dtype.ir


def _set_param_dtype(func: TileFunction, param_name: str, arr: AlloyBuffer | None) -> None:
    if arr is None:
        return
    dtype_name = _buffer_dtype_name(arr)
    for param in func.params:
        if param.name == param_name:
            param.dtype = dtype_name
            break


def _index_transform_for_extra(earr: AlloyBuffer) -> IndexTransform:
    """Build the IndexTransform for an extra buffer from its strides/shape.

    Returns a self-contained transform — no constexpr references.
    For 2D extras, the row stride is encoded in StridedTransform.
    """
    if not (earr.ndim and earr._strides):
        # No stride info → flat identity
        return IdentityTransform()

    itemsize = earr.dtype.itemsize
    elem_strides = tuple(s // itemsize for s in earr._strides)

    # All-zero strides → scalar broadcast
    if all(s == 0 for s in elem_strides):
        return ScalarBroadcastTransform()

    # 1D: flat identity
    if earr.ndim == 1:
        return IdentityTransform()

    # 2D: encode the row stride directly
    if earr.ndim == 2:
        return StridedTransform(row_stride=elem_strides[-2])

    # >2D: check if contiguous (flat identity works)
    is_contiguous = elem_strides[-1] == 1
    for i in range(len(earr._shape) - 2, -1, -1):
        if elem_strides[i] != elem_strides[i + 1] * earr._shape[i + 1]:
            is_contiguous = False
            break
    if is_contiguous:
        return IdentityTransform()

    # Non-contiguous >2D: scatter decomposition
    return ScatterTransform(nd_shape=earr._shape, nd_strides=elem_strides)


def _add_extra_param(
    func: TileFunction, buffer_map: dict[str, AlloyBuffer], ename: str, earr: AlloyBuffer
) -> None:
    """Add an extra buffer param to a TileFunction.

    The IndexTransform is built separately by _register_extra_params and
    attached to the IR node; all indexing info lives on the transform.
    """
    if ename in buffer_map:
        return
    buffer_map[ename] = earr
    func.params.append(TileParam(name=ename, is_constexpr=False, dtype=_buffer_dtype_name(earr)))


def _build_buf_map(op: LazyOp, func: TileFunction) -> dict[str, AlloyBuffer]:
    """Map param names to their buffer arrays.

    Contiguifies stride-0 broadcast views — fused kernels use flat cooperative
    loads that can't handle stride-0 dims with size > 1.
    """
    buf_params = [p.name for p in func.params if not p.is_constexpr]
    buf_map: dict[str, AlloyBuffer] = {}
    for i, pname in enumerate(buf_params):
        if i < len(op.buffer_args):
            arr = op.buffer_args[i][1]
            if arr._strides and any(s == 0 and d > 1 for s, d in zip(arr._strides, arr._shape)):
                arr = arr.contiguous()
            buf_map[pname] = arr
    return buf_map


def _maybe_restride_extra(extra_arr: AlloyBuffer, op: LazyOp) -> AlloyBuffer:
    """If the chain's shared input was view+permuted, restride the extra to match."""
    shared_strides_cv: tuple[int, ...] | None = None
    shared_shape_cv: tuple[int, ...] | None = None
    for bname, _ in op.buffer_args:
        cv = op.constexpr_values.get(f"_{bname}_strides")
        if cv is not None and isinstance(cv, tuple):
            shared_strides_cv = cv
            shared_shape_cv = op.constexpr_values.get(f"_{bname}_shape")
            break

    if shared_strides_cv is None:
        return extra_arr

    if shared_shape_cv is None:
        return extra_arr
    if tuple(extra_arr._shape) != tuple(shared_shape_cv):
        return extra_arr

    extra_isz = extra_arr._dtype.itemsize
    extra_elem_strides = tuple(s // extra_isz for s in extra_arr._strides)
    if extra_elem_strides == tuple(shared_strides_cv):
        return extra_arr

    perm = sorted(range(len(shared_strides_cv)), key=lambda d: -shared_strides_cv[d])
    return extra_arr.transpose(*perm)


# --- Transform attachment ---


def _attach_transform_to_stores(
    func: TileFunction,
    param_name: str,
    transform_ops: list[TileOp],
    chain_source_name: str | None = None,
    round_acc_for_eager: bool = False,
) -> bool:
    found = False
    for op in walk_ops(func.ops):
        if isinstance(op, Store) and op.ptr and op.ptr.name == param_name:
            op.transform = [copy.copy(t) for t in transform_ops]
            if chain_source_name:
                op.transform_source_name = chain_source_name
            if round_acc_for_eager:
                op.round_acc_for_eager = True
            found = True
    return found


def _attach_transform_to_loads(
    func: TileFunction,
    param_name: str,
    transform_ops: list[TileOp],
    chain_source_name: str | None = None,
) -> bool:
    found = False
    for op in walk_ops(func.ops):
        if isinstance(op, Load) and op.ptr and op.ptr.name == param_name:
            op.transform = [copy.copy(t) for t in transform_ops]
            if chain_source_name:
                op.transform_source_name = chain_source_name
            found = True
    return found


def _clone_stores_for_output(
    func: TileFunction,
    src_param: str,
    new_param: str,
    transform_ops: list[TileOp],
    chain_source_name: str | None = None,
    round_acc_for_eager: bool = False,
) -> None:
    def _clone_store(src: Store) -> Store:
        # transform_extras gets its own copy so downstream mutation doesn't
        # leak to the original.
        new = copy.copy(src)
        new.ptr = TileValue(new_param, (), Layout.REPLICATED, src.ptr.dtype)
        new.transform = [copy.copy(t) for t in transform_ops]
        new.transform_source_name = chain_source_name
        new.transform_extras = dict(src.transform_extras)
        if round_acc_for_eager:
            new.round_acc_for_eager = True
        return new

    idx = 0
    ops = func.ops
    while idx < len(ops):
        op = ops[idx]
        if isinstance(op, Store) and op.ptr and op.ptr.name == src_param:
            ops.insert(idx + 1, _clone_store(op))
            idx += 2
            continue
        nested_bodies: tuple[list[TileOp], ...] = ()
        if isinstance(op, ForLoop):
            nested_bodies = (op.body,)
        elif isinstance(op, WhileLoop):
            nested_bodies = (op.cond_body, op.body)
        elif isinstance(op, IfElse):
            nested_bodies = (op.body, op.orelse)
        for body in nested_bodies:
            sub_idx = 0
            while sub_idx < len(body):
                sub_op = body[sub_idx]
                if isinstance(sub_op, Store) and sub_op.ptr and sub_op.ptr.name == src_param:
                    body.insert(sub_idx + 1, _clone_store(sub_op))
                    sub_idx += 2
                    continue
                sub_idx += 1
        idx += 1


def _set_col_slice_on_stores(
    func: TileFunction,
    param_name: str,
    info: ColumnSliceInfo,
) -> None:
    """Set col_slice on all Store nodes targeting param_name."""
    for op in walk_ops(func.ops):
        if isinstance(op, Store) and op.ptr and op.ptr.name == param_name:
            op.col_slice = info


# --- Name uniqueness ---


def _unique_name(base: str, taken: set[str]) -> str:
    if base not in taken:
        return base
    k = 2
    while f"{base}_{k}" in taken:
        k += 1
    return f"{base}_{k}"


# ===================================================================
# Section 6: Chain composition via walk+rebuild substitution
# ===================================================================


@dataclass
class IRSubstitution:
    """Maps old TileValue names to new TileValues.

    Build once from the chain's buffer bindings, then apply to every op in
    the transform.
    """

    _map: dict[str, TileValue] = field(default_factory=dict)

    def add(self, old_name: str, new_value: TileValue) -> None:
        """Register a substitution: old_name → new_value."""
        self._map[old_name] = new_value

    def get(self, name: str) -> TileValue | None:
        """Look up a substitution."""
        return self._map.get(name)

    def apply_ops(self, ops: list[TileOp], result_prefix: str | None = None) -> list[TileOp]:
        """Apply substitution to ops, creating new ops.

        If result_prefix is given, renames results for uniqueness. Prefix
        mappings are LOCAL to this call so they don't pollute subsequent steps.
        """
        local_map = dict(self._map)
        new_ops: list[TileOp] = []
        for op in ops:
            new_op = self._apply_with_map(op, local_map)
            if new_op.result is not None and result_prefix is not None:
                old_name = op.result.name
                new_name = f"{result_prefix}_{new_op.result.name}"
                new_op.result = TileValue(
                    new_name, new_op.result.shape, new_op.result.layout, new_op.result.dtype
                )
                local_map[old_name] = new_op.result
            new_ops.append(new_op)
        return new_ops

    def _apply_with_map(self, op: TileOp, mapping: dict[str, TileValue]) -> TileOp:
        """Create a new op with operands substituted from the given map."""
        new_op = copy.copy(op)
        new_op.remap(mapping)
        if new_op.result is not None:
            new_op.result = TileValue(
                new_op.result.name, new_op.result.shape, new_op.result.layout, new_op.result.dtype
            )
        return new_op


@dataclass
class ExtractedTransform:
    ops: list[TileOp]
    input_value_name: str
    input_param_name: str | None
    output_param_name: str | None
    extra_inputs: dict[str, TileValue | None]


@dataclass
class ComposedTransform:
    ops: list[TileOp]
    final_out_arr: AlloyBuffer | None
    extra_params: dict[str, AlloyBuffer]
    source_param_name: str | None


@dataclass
class AnchorFusionContext:
    anchor_op: LazyOp
    func: TileFunction
    constexpr_values: dict[str, int | float | bool | tuple[int, ...]]
    buffer_map: dict[str, AlloyBuffer]
    buffer_params: list[str]
    output_idx: int
    output_param: str
    anchor_out_arr: AlloyBuffer | None


def _extract_and_lower(
    op: LazyOp, step_idx: int
) -> tuple[TileFunction, ExtractedTransform, dict[str, AlloyBuffer]]:
    """Extract transform from a LazyOp's stored TileFunction. No re-tracing.

    Shallow-clone is safe: extract_ir_transform copy.copies each extracted op
    before any operand remap, so shared BinOp/UnaryOp/Compare instances stay
    immutable from fusion's perspective.
    """
    func = shallow_clone_for_fusion(op.func)
    raw = extract_ir_transform(func)
    if raw is None:
        raise FusionUnsupported(
            f"Cannot extract transform from elem op {step_idx}", op_idx=step_idx
        )
    xf_ops, load_name, _store_value, load_ptr, store_ptr, extra = raw
    result = ExtractedTransform(
        ops=xf_ops,
        input_value_name=load_name,
        input_param_name=load_ptr,
        output_param_name=store_ptr,
        extra_inputs=extra,
    )
    buf_map = _build_buf_map(op, func)
    return func, result, buf_map


def _identify_chain_input(
    xf: ExtractedTransform,
    buf_map: dict[str, AlloyBuffer],
    prev_out_arr: AlloyBuffer | None,
    step_idx: int,
) -> tuple[str, bool]:
    """Identify which input connects to the chain.

    Returns (input_name, is_swap) where is_swap=True if an extra was
    identified as the chain input (primary becomes extra).
    """
    if prev_out_arr is None:
        return xf.input_value_name, False

    read_arr = buf_map.get(xf.input_param_name)
    if read_arr is not None and read_arr.shares_allocation(prev_out_arr):
        # The epilogue's primary input must cover the WHOLE anchor output. A
        # strict sub-slice (different offset, or fewer elements — e.g. the Q
        # slice of a batched QKV dot) would narrow the fused dot to that slice's
        # columns: the dot computes the full weight's N columns but writes them
        # into the slice-sized buffer, so sibling slices are written out of
        # bounds / read from never-materialized memory. Refuse so the dot
        # compiles standalone and the epilogue runs unfused over the slices.
        # Compare by element COUNT, not shape tuple: a reshape of the full output
        # (same offset + numel, e.g. K-rope cast to f16 for the KV cache) covers
        # everything and must stay fusable.
        if read_arr._offset != prev_out_arr._offset or read_arr.size != prev_out_arr.size:
            raise FusionUnsupported(
                f"elem op {step_idx} primary reads a sub-slice of the anchor "
                "output; keep the batched dot unfused",
                op_idx=step_idx,
            )
        return xf.input_value_name, False

    for ename in xf.extra_inputs:
        earr = buf_map.get(ename)
        if earr is not None and earr.shares_allocation(prev_out_arr):
            return ename, True

    raise FusionUnsupported(
        f"Elem op {step_idx} primary load doesn't chain from previous output",
        op_idx=step_idx,
    )


def _resolve_extra_arr(
    ename: str, buf_map: dict[str, AlloyBuffer], input_param_name: str | None
) -> AlloyBuffer | None:
    extra_arr = buf_map.get(ename)
    if extra_arr is None and ename.startswith("_xtra_"):
        primary_arr = buf_map.get(input_param_name)
        if primary_arr is not None:
            extra_arr = primary_arr
    return extra_arr


def _resolve_extra_alloc(
    earr: AlloyBuffer,
    alloc_resolved: dict[int, TileValue],
    anchor_out_arr: AlloyBuffer | None,
    step_idx: int,
) -> TileValue | str | None:
    """Resolve an extra buffer via allocation_id lookup.

    Returns:
    - TileValue: substitute with this value (known intermediate or prev output)
    - str: chain source name (extra refers to anchor output)
    - None: unknown buffer, must be added as extra param
    """
    resolved = alloc_resolved.get(earr.buffer_key)
    if resolved is not None:
        return resolved

    if anchor_out_arr is not None and earr.shares_allocation(anchor_out_arr):
        # Sharing the allocation only lets the in-register anchor value be
        # reused if the extra reads the SAME range. A different column slice
        # (e.g. the `up` half of a batched gate_up at offset != the anchor's
        # gate half) is a sibling slice the fused dot never materializes. Bail
        # so the dot materializes fully and the epilogue runs unfused.
        if earr._offset == anchor_out_arr._offset and earr._shape == anchor_out_arr._shape:
            return f"_csrc_{step_idx}"
        # A sibling column slice of the anchor. Absorbing this epilogue would
        # narrow the dot to the gate columns and read `up` from the
        # never-materialized full output. Refuse: the batched dot compiles
        # standalone and the epilogue runs as a separate pass over the slices.
        raise FusionUnsupported(
            f"epilogue extra at step {step_idx} reads a sibling column slice "
            "of the anchor output; keep the batched dot unfused",
            op_idx=step_idx,
        )

    return None


def _compose_chain(
    elem_ops: list[LazyOp],
    anchor_out_arr: AlloyBuffer | None,
    taken_names: set[str] | None = None,
) -> ComposedTransform:
    """Compose transforms from a chain of elem ops via walk+rebuild substitution.

    For each elem op: extract its computation, build an IRSubstitution mapping
    its inputs to the previous chain result, and apply to produce new ops.
    Extra buffers are resolved via allocation_id lookup.
    """
    sub = IRSubstitution()
    extra_params: dict[str, AlloyBuffer] = {}
    taken: set[str] = set(taken_names) if taken_names else set()

    # allocation_id → TileValue for all known buffers in the chain.
    alloc_resolved: dict[tuple[int, int], TileValue] = {}

    prev_out_arr: AlloyBuffer | None = anchor_out_arr
    final_out_arr: AlloyBuffer | None = None
    chain_source_name: str | None = None
    chain_origin_name: str | None = None
    composed_ops: list[TileOp] = []
    last_load_name: str | None = None

    for i, op in enumerate(elem_ops):
        _func, xf, buf_map = _extract_and_lower(op, i)

        chain_input, is_swap = _identify_chain_input(xf, buf_map, prev_out_arr, i)
        effective_input_param = chain_input if is_swap else xf.input_param_name
        last_load_name = xf.input_value_name

        # Handle primary/extra swap: primary becomes extra. Resolve the primary
        # like extras — it may be a chain intermediate (in registers), not an
        # external buffer. Substitutions added here reference THIS op's local
        # parameter names (e.g. "x", "y") which can collide with subsequent
        # steps; track which keys we add so they can be removed after apply_ops.
        step_sub_keys: list[str] = []
        if is_swap:
            primary_arr = buf_map.get(xf.input_param_name)
            if primary_arr is not None:
                resolved_primary = _resolve_extra_alloc(
                    primary_arr, alloc_resolved, anchor_out_arr, i
                )
                if isinstance(resolved_primary, TileValue):
                    sub.add(xf.input_value_name, resolved_primary)
                    step_sub_keys.append(xf.input_value_name)
                elif isinstance(resolved_primary, str):
                    sub.add(
                        xf.input_value_name,
                        TileValue(resolved_primary, (), Layout.REPLICATED, "f32"),
                    )
                    step_sub_keys.append(xf.input_value_name)
                    chain_source_name = resolved_primary
                else:
                    unique_pname = _unique_name(xf.input_param_name, taken)
                    sub.add(
                        xf.input_value_name,
                        TileValue(unique_pname, (), Layout.REPLICATED, "f32"),
                    )
                    step_sub_keys.append(xf.input_value_name)
                    extra_params[unique_pname] = _maybe_restride_extra(primary_arr, op)
                    taken.add(unique_pname)

            # At step 0 only: map the chain input (the extra identified as
            # chain flow) back to the primary's load result name so the chain
            # source name appears in the composed ops. At step > 0, the chain
            # input is mapped to prev_result below.
            if i == 0:
                sub.add(
                    chain_input,
                    TileValue(xf.input_value_name, (), Layout.REPLICATED, "f32"),
                )
                step_sub_keys.append(chain_input)
                # The composed IR uses xf.input_value_name as the chain-source
                # placeholder (via the substitution above); the emitter needs
                # this explicit pointer.
                if chain_source_name is None:
                    chain_source_name = xf.input_value_name

        if i == 0:
            # The chain input appears in composed IR under xf.input_value_name
            # in both swap and non-swap cases, which is also the chain source
            # name, so the downstream origin-name fixup is a no-op.
            chain_origin_name = xf.input_value_name
        elif composed_ops and composed_ops[-1].result is not None:
            target = chain_input if is_swap else xf.input_value_name
            sub.add(target, composed_ops[-1].result)
            step_sub_keys.append(target)

        # Resolve extra inputs via allocation_id lookup.
        for ename in list(xf.extra_inputs):
            if is_swap and ename == chain_input:
                continue  # already handled as chain input
            earr = _resolve_extra_arr(ename, buf_map, effective_input_param)
            if earr is None:
                continue

            # Pre-check: extra shares allocation with prev output → same as chain flow.
            # Must check BEFORE alloc lookup because at step 0, prev_out = anchor_out
            # and alloc_resolved is empty, so the anchor_out branch would incorrectly
            # treat this as a chain source reference.
            if (
                prev_out_arr is not None
                and earr.shares_allocation(prev_out_arr)
                and earr._offset == prev_out_arr._offset
                and earr._shape == prev_out_arr._shape
            ):
                chain_flow_name = chain_input if is_swap else xf.input_value_name
                existing = sub.get(chain_flow_name)
                if existing is not None:
                    sub.add(ename, existing)
                else:
                    sub.add(ename, TileValue(chain_flow_name, (), Layout.REPLICATED, "f32"))
                step_sub_keys.append(ename)
                continue

            resolved = _resolve_extra_alloc(earr, alloc_resolved, anchor_out_arr, i)

            if isinstance(resolved, TileValue):
                sub.add(ename, resolved)
                step_sub_keys.append(ename)
            elif isinstance(resolved, str):
                # Anchor output reference → chain source
                sub.add(ename, TileValue(resolved, (), Layout.REPLICATED, "f32"))
                step_sub_keys.append(ename)
                chain_source_name = resolved
            else:
                # Unknown buffer → add as extra param
                earr_final = _maybe_restride_extra(buf_map[ename], op)
                unique = _unique_name(ename, taken)
                if unique != ename:
                    sub.add(ename, TileValue(unique, (), Layout.REPLICATED, "f32"))
                    step_sub_keys.append(ename)
                extra_params[unique] = earr_final
                taken.add(unique)

        prefix = f"_c{i}" if i > 0 else None
        step_ops = sub.apply_ops(xf.ops, result_prefix=prefix)
        composed_ops.extend(step_ops)

        # Discard substitutions keyed by THIS step's local names: reusing them
        # aliases later steps that share a local name (every elementwise kernel
        # has x/y params).
        for k in step_sub_keys:
            sub._map.pop(k, None)

        # Track this step's output for next iteration
        step_out_arr = buf_map.get(xf.output_param_name)
        if step_out_arr is not None:
            final_out_arr = step_out_arr
            prev_out_arr = step_out_arr
            if step_ops and step_ops[-1].result is not None:
                alloc_resolved[step_out_arr.buffer_key] = step_ops[-1].result

    if (
        chain_source_name is not None
        and chain_origin_name is not None
        and chain_source_name != chain_origin_name
    ):
        fixup = IRSubstitution()
        fixup.add(chain_origin_name, TileValue(chain_source_name, (), Layout.REPLICATED, "f32"))
        composed_ops = fixup.apply_ops(composed_ops)

    if chain_source_name:
        source = chain_source_name
    elif len(elem_ops) == 1:
        source = last_load_name
    else:
        source = None

    return ComposedTransform(
        ops=composed_ops,
        final_out_arr=final_out_arr,
        extra_params=extra_params,
        source_param_name=source,
    )


# --- Extra sizes and constexpr propagation ---


def _build_broadcast_transform(
    earr: AlloyBuffer,
    esize: int,
    total_elems: int,
    chain_ops: list[LazyOp],
    anchor_out_shape: tuple[int, ...] | None,
    func_constexprs: dict[str, int | float | bool | tuple[int, ...]],
) -> IndexTransform:
    """Build the IndexTransform for a broadcast extra (esize < total_elems).

    Determines inner_repeat from stride metadata, then disambiguates
    row-broadcast vs column-broadcast vs general modular wrap.
    """
    inner_repeat = 1
    row_stride = 0

    # A single-element extra is unconditionally a scalar broadcast. Decide this
    # before the M/N disambiguation: when esize == M (decode, M == 1, with a
    # scalar operand like gemma4's per-layer `layer_scalar` of shape (1,)) that
    # path misclassifies it as a ColumnBroadcast, emitting `y[m*N + col]`
    # instead of `y[0]` and reading out of bounds of the 1-element buffer.
    if esize == 1:
        return ScalarBroadcastTransform()

    # Non-trailing broadcast (e.g. attention mask `[B, 1, S, S]` against
    # output `[B, H, S, S]`) — the broadcast dim sits between leading
    # batch dims and trailing data dims, so neither `i % size` nor
    # `(i / inner_repeat) % size` produces the right index. Use the
    # chain op's constexpr stride metadata, which encodes a 0 in the
    # broadcast dim, and let ScatterTransform emit the N-D unravel.
    for cop in chain_ops:
        for bname, barr in cop.buffer_args:
            if barr is earr:
                cv_strides = cop.constexpr_values.get(f"_{bname}_strides")
                cv_shape = cop.constexpr_values.get(f"_{bname}_shape")
                if (
                    cv_strides
                    and cv_shape
                    and len(cv_strides) == len(cv_shape)
                    and len(cv_strides) >= 3
                    and any(
                        st == 0 and s > 1
                        for st, s in zip(cv_strides[:-1], cv_shape[:-1], strict=True)
                    )
                    # Trailing-broadcast case (cv_strides[-1] == 0) is
                    # already handled below via the `inner_repeat` path.
                    and cv_strides[-1] != 0
                ):
                    return ScatterTransform(
                        nd_shape=tuple(int(s) for s in cv_shape),
                        nd_strides=tuple(int(st) for st in cv_strides),
                    )
                break

    if earr._strides and earr.ndim >= 2 and earr._strides[-1] == 0 and earr._shape[-1] > 1:
        inner_repeat = earr._shape[-1]
    else:
        for cop in chain_ops:
            for bname, barr in cop.buffer_args:
                if barr is earr:
                    cv_strides = cop.constexpr_values.get(f"_{bname}_strides")
                    cv_shape = cop.constexpr_values.get(f"_{bname}_shape")
                    if cv_strides and cv_shape and cv_strides[-1] == 0 and cv_shape[-1] > 1:
                        n_data_dims = sum(
                            1 for s, st in zip(cv_shape, cv_strides) if s > 1 and st > 0
                        )
                        if n_data_dims > earr.ndim:
                            inner_repeat = 1
                            for sd, st in zip(cv_shape, cv_strides):
                                if st == 0 and sd > 1:
                                    inner_repeat *= sd
                        elif anchor_out_shape is not None:
                            consumer_input_shape = None
                            for cn, cb in cop.buffer_args:
                                if cn not in cop.kernel._output_params:
                                    if cb.size == total_elems:
                                        consumer_input_shape = cb.shape
                                        break
                            if consumer_input_shape is not None:

                                def _strip_ones(s: tuple[int, ...]) -> tuple[int, ...]:
                                    i = 0
                                    while i < len(s) - 1 and s[i] == 1:
                                        i += 1
                                    return s[i:]

                                if _strip_ones(consumer_input_shape) != _strip_ones(
                                    anchor_out_shape
                                ):
                                    inner_repeat = 1
                                    for sd, st in zip(cv_shape, cv_strides):
                                        if st == 0 and sd > 1:
                                            inner_repeat *= sd
                    break
            if inner_repeat > 1:
                break

    # 1D extras with no inner_repeat: disambiguate broadcast direction using M/N
    if row_stride == 0 and inner_repeat == 1:
        M = func_constexprs.get("M")
        N = func_constexprs.get("N")
        # Fall back to anchor output shape when the kernel doesn't expose M/N as
        # constexprs (e.g. `dot_q6_k` derives them from `A.shape` / `B.shape[0]`).
        # Without this fallback, the disambiguation defaults to a generic
        # modular-wrap broadcast that addresses the bias as `y[m*N + col]`
        # instead of `y[col]`, reading out-of-bounds for every row except the
        # first.
        if (not isinstance(M, int) or not isinstance(N, int)) and anchor_out_shape is not None:
            if len(anchor_out_shape) == 2:
                if not isinstance(M, int):
                    M = int(anchor_out_shape[0])
                if not isinstance(N, int):
                    N = int(anchor_out_shape[1])
        if isinstance(M, int) and isinstance(N, int):
            if esize == M and esize != N:
                return ColumnBroadcastTransform()
            if esize == N:
                return RowBroadcastTransform(size=N)

    if row_stride == 0 and inner_repeat > 1:
        N = func_constexprs.get("N")
        if not isinstance(N, int) and anchor_out_shape is not None and len(anchor_out_shape) == 2:
            N = int(anchor_out_shape[1])
        if isinstance(N, int):
            row_stride = N

    return BroadcastTransform(size=esize, inner_repeat=inner_repeat, row_stride=row_stride)


def _build_extra_transform(
    earr: AlloyBuffer,
    total_elems: int | None,
    chain_ops: list[LazyOp],
    anchor_out_shape: tuple[int, ...] | None,
    func_constexprs: dict[str, int | float | bool | tuple[int, ...]],
) -> IndexTransform:
    """Build the IndexTransform for an extra buffer.

    Dispatch order:
    1. Scalar broadcast (all-zero strides)
    2. Broadcast (esize < total_elems) — with row/column/general disambiguation
    3. Scatter (non-contiguous >2D)
    4. Strided (2D or 1D-in-2D-context with known N)
    5. Identity (flat 1D only)
    """
    base = _index_transform_for_extra(earr)
    if isinstance(base, ScalarBroadcastTransform):
        return base
    if isinstance(base, ScatterTransform):
        return base

    esize = _buf_elem_count(earr)
    if total_elems is not None and esize < total_elems:
        return _build_broadcast_transform(
            earr,
            esize,
            total_elems,
            chain_ops,
            anchor_out_shape,
            func_constexprs,
        )

    # IdentityTransform is only safe in flat (1D) contexts. In 2D tile
    # contexts, tile_2d() must produce a self-contained expression with
    # an explicit row stride — not a reference to an ambient `_N` constexpr.
    # Upgrade to StridedTransform(N) when we know the column count.
    if isinstance(base, IdentityTransform):
        N = func_constexprs.get("N")
        if isinstance(N, int) and N > 0:
            return StridedTransform(row_stride=N)

    return base


def _attach_extras_to_ops(
    func: TileFunction,
    extras: dict[str, IndexTransform],
    transform_filter: Callable[[TileOp], bool] | None = None,
) -> None:
    """Attach IndexTransform instances to Load/Store nodes that have transforms."""
    if not extras:
        return
    for ir_op in walk_ops(func.ops):
        if not isinstance(ir_op, (Load, Store)) or not ir_op.transform:
            continue
        if transform_filter is not None and not transform_filter(ir_op):
            continue
        for name, xf in extras.items():
            ir_op.transform_extras[name] = xf


def _register_extra_params(
    func: TileFunction,
    buffer_map: dict[str, AlloyBuffer],
    extra: dict[str, AlloyBuffer],
    total_elems: int,
    chain_ops: list[LazyOp],
    *,
    transform_filter: Callable[[TileOp], bool] | None = None,
    anchor_out_shape: tuple[int, ...] | None = None,
) -> None:
    extras: dict[str, IndexTransform] = {}
    for ename, earr in extra.items():
        _add_extra_param(func, buffer_map, ename, earr)
        extras[ename] = _build_extra_transform(
            earr,
            total_elems,
            chain_ops,
            anchor_out_shape,
            func.constexpr_values,
        )
    _attach_extras_to_ops(func, extras, transform_filter=transform_filter)


def _attach_transform(
    func: TileFunction,
    param_name: str,
    composed: ComposedTransform,
    *,
    load: bool,
    label: str,
    round_acc_for_eager: bool = False,
) -> None:
    target = "load" if load else "store"
    if load:
        ok = _attach_transform_to_loads(
            func,
            param_name,
            composed.ops,
            chain_source_name=composed.source_param_name,
        )
    else:
        ok = _attach_transform_to_stores(
            func,
            param_name,
            composed.ops,
            chain_source_name=composed.source_param_name,
            round_acc_for_eager=round_acc_for_eager,
        )
    if not ok:
        raise FusionUnsupported(f"Failed to attach {label} transform to {target} '{param_name}'")


def _epilogue_has_tile_consumer(chain_ops: list[LazyOp], anchor_size: int) -> bool:
    """Detect if any chain op is a tile binop (multi-input same-shape as anchor).

    Distinguishes residual-stream `add(gemm_out, residual_tile)` (2+ same-size
    inputs) from bias `add(gemm_out, bias_broadcast)` (1 same-size input).
    Flags epilogues where eager rounds the gemm acc to bf16 before the binop;
    fused single-dispatch otherwise keeps f32 acc through the add, diverging
    from eager.
    """
    if anchor_size <= 0:
        return False
    for op in chain_ops:
        same_size_inputs = 0
        for pn, buf in op.buffer_args:
            if pn in op.kernel._output_params:
                continue
            if buf.size == anchor_size:
                same_size_inputs += 1
        if same_size_inputs >= 2:
            return True
    return False


def _apply_chain(
    func: TileFunction,
    buffer_map: dict[str, AlloyBuffer],
    chain_ops: list[LazyOp],
    anchor_out_arr: AlloyBuffer | None,
    param_name: str,
    total_elems: int,
    label: str,
    load: bool = False,
    clone_param: str | None = None,
    bound_arr: AlloyBuffer | None = None,
) -> ComposedTransform:
    composed: ComposedTransform = _compose_chain(
        chain_ops,
        anchor_out_arr,
        taken_names={p.name for p in func.params},
    )
    round_acc = (
        not load
        and anchor_out_arr is not None
        and _epilogue_has_tile_consumer(chain_ops, anchor_out_arr.size)
    )

    if clone_param is not None:
        bound_arr = composed.final_out_arr if composed.final_out_arr is not None else anchor_out_arr
        dtype_name = _buffer_dtype_name(bound_arr) if bound_arr is not None else "f32"
        func.params.append(TileParam(name=clone_param, is_constexpr=False, dtype=dtype_name))
        _clone_stores_for_output(
            func,
            param_name,
            clone_param,
            composed.ops,
            chain_source_name=composed.source_param_name,
            round_acc_for_eager=round_acc,
        )
        target_param = clone_param

        def transform_filter(op: TileOp, cp: str = clone_param) -> bool:
            return isinstance(op, Store) and op.ptr is not None and op.ptr.name == cp
    else:
        _attach_transform(
            func,
            param_name,
            composed,
            load=load,
            label=label,
            round_acc_for_eager=round_acc,
        )
        target_param = param_name
        transform_type = Load if load else Store

        def transform_filter(
            op: TileOp, pn: str = param_name, tt: type[Load] | type[Store] = transform_type
        ) -> bool:
            return isinstance(op, tt) and op.ptr is not None and op.ptr.name == pn

        if bound_arr is None and not load:
            bound_arr = composed.final_out_arr

    if bound_arr is not None:
        buffer_map[target_param] = bound_arr
        _set_param_dtype(func, target_param, bound_arr)

    _register_extra_params(
        func,
        buffer_map,
        composed.extra_params,
        total_elems,
        chain_ops,
        transform_filter=transform_filter,
        anchor_out_shape=anchor_out_arr.shape if anchor_out_arr is not None else None,
    )
    return composed


# ===================================================================
# Section 7: Anchor compilation context
# ===================================================================


def _build_anchor_context(anchor_op: LazyOp) -> AnchorFusionContext:
    output_idx = anchor_op.kernel._output_idx
    func = shallow_clone_for_fusion(anchor_op.func)
    func.name = "_fused"
    cv = dict(anchor_op.constexpr_values)
    # Inject M/N from the anchor's output shape — the emitter needs them to
    # disambiguate epilogue broadcast direction (row vs column).
    buffer_params = [p.name for p in func.params if not p.is_constexpr]
    _out_param = buffer_params[output_idx] if output_idx < len(buffer_params) else "out"
    out_shape = anchor_op.buffer_shapes.get(_out_param)
    if out_shape is not None and len(out_shape) == 2:
        cv.setdefault("M", out_shape[0])
        cv.setdefault("N", out_shape[1])
    func.constexpr_values = cv
    buffer_map = _build_buf_map(anchor_op, func)
    buffer_params = [p.name for p in func.params if not p.is_constexpr]
    output_param = buffer_params[output_idx] if output_idx < len(buffer_params) else "out"
    return AnchorFusionContext(
        anchor_op=anchor_op,
        func=func,
        constexpr_values=cv,
        buffer_map=buffer_map,
        buffer_params=buffer_params,
        output_idx=output_idx,
        output_param=output_param,
        anchor_out_arr=buffer_map.get(output_param),
    )


def _resolve_anchor_input_param(ctx: AnchorFusionContext, produced_arr: AlloyBuffer | None) -> str:
    for inp_idx in range(min(ctx.output_idx, len(ctx.anchor_op.buffer_args))):
        _, inp_arr = ctx.anchor_op.buffer_args[inp_idx]
        if produced_arr is not None and inp_arr.shares_allocation(produced_arr):
            return ctx.buffer_params[inp_idx]
    raise FusionUnsupported("Failed to resolve anchor input for prologue fusion")


def _apply_anchor_prologue(
    ctx: AnchorFusionContext, pro_indices: list[int], queue: list[LazyOp]
) -> None:
    if not pro_indices:
        return
    last_pro_op = queue[pro_indices[-1]]
    last_pro_func, last_xf, last_pro_buf_map = _extract_and_lower(last_pro_op, pro_indices[-1])
    last_pro_out_arr = last_pro_buf_map.get(last_xf.output_param_name)
    anchor_input_param = _resolve_anchor_input_param(ctx, last_pro_out_arr)

    if len(pro_indices) == 1:
        if not inject_prologue_ir(ctx.func, last_pro_func, anchor_input_param):
            raise FusionUnsupported(f"Failed to inject prologue into load '{anchor_input_param}'")
        pro_src_arr = last_pro_buf_map.get(last_xf.input_param_name)
        if pro_src_arr is not None:
            ctx.buffer_map[anchor_input_param] = pro_src_arr
            _set_param_dtype(ctx.func, anchor_input_param, pro_src_arr)
        extra_params: dict[str, AlloyBuffer] = {}
        for extra_name in last_xf.extra_inputs:
            extra_arr: AlloyBuffer | None = _resolve_extra_arr(
                extra_name, last_pro_buf_map, last_xf.input_param_name
            )
            if extra_arr is None:
                for orig_name, orig_arr in last_pro_buf_map.items():
                    if orig_name != last_xf.input_param_name and orig_name not in ctx.buffer_map:
                        extra_arr = orig_arr
                        break
            if extra_arr is not None:
                extra_params[extra_name] = extra_arr
        _register_extra_params(
            ctx.func,
            ctx.buffer_map,
            extra_params,
            _buf_elem_count(pro_src_arr),
            [last_pro_op],
            transform_filter=lambda op, aip=anchor_input_param: (
                isinstance(op, Load) and op.ptr is not None and op.ptr.name == aip
            ),
        )
        return

    # Multi-step prologue chain
    first_pro_op = queue[pro_indices[0]]
    _first_pro_func, first_xf, first_pro_buf_map = _extract_and_lower(first_pro_op, pro_indices[0])
    _apply_chain(
        ctx.func,
        ctx.buffer_map,
        [queue[idx] for idx in pro_indices],
        first_pro_buf_map.get(first_xf.input_param_name),
        param_name=anchor_input_param,
        total_elems=_buf_elem_count(first_pro_buf_map.get(first_xf.input_param_name)),
        label="prologue chain",
        load=True,
        bound_arr=first_pro_buf_map.get(first_xf.input_param_name),
    )


def _evaluate_anchor_grid(ctx: AnchorFusionContext) -> tuple[int, int, int]:
    spec = ctx.func.dispatch_spec
    if spec is not None and (spec.grid_axes or spec.outputs):
        _, grid_3d, _ = spec.evaluate_dispatch(
            ctx.constexpr_values,
            ctx.anchor_op.buffer_shapes,
            grid_override=ctx.anchor_op.grid,
            kernel_name="_fused",
        )
        return grid_3d
    return ctx.anchor_op.grid or (1, 1, 1)
