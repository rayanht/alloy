"""Address decomposition and dispatch write-pattern analysis for tracing."""

from __future__ import annotations

from typing import TypeAlias

from alloy._compiler.dispatch_spec import OutputWritePattern
from alloy._compiler.tile_ir import BinOp, Constant, Copy, ExpandDims, TileValue, UnaryOp
from alloy._compiler.trace.value import TracedValue, TraceCtx, _ctx

ExtractedAddress2D: TypeAlias = tuple[
    TileValue | None, TileValue | None, int | None, TileValue | None, bool
]

# ---------------------------------------------------------------------------
# Pointer decomposition — extract (base_ptr, offsets) for load/store
# ---------------------------------------------------------------------------


def _apply_stride_decomposition(ctx: TraceCtx, ptr_tv: TileValue, off_tv: TileValue) -> TileValue:
    """If the buffer has stride constexprs, decompose flat offset → strided offset.

    The stride metadata is injected by _queue_op as constexprs:
      _{name}_shape: tuple (logical view shape)
      _{name}_strides: tuple (element strides)

    Transforms: flat_idx → i0*s0 + i1*s1 + ... + iN*sN
    where iK = (flat_idx // prod(shape[K+1:])) % shape[K]

    The view's element offset is NOT applied here — runtime binding adds it
    via the Metal buffer offset, so baking it in MSL would double-apply.
    """
    pname = ptr_tv.name
    cv = ctx.constexpr_values
    shape_key = f"_{pname}_shape"
    if shape_key not in cv:
        return off_tv  # contiguous buffer, no transformation

    shape = cv[shape_key]
    strides = cv[f"_{pname}_strides"]
    ndim = len(shape)

    if ndim == 0:
        return off_tv

    b = ctx.builder
    # Decompose flat_idx into per-dimension indices and apply strides.
    # All arithmetic is integer (matching the offset tile's dtype).
    # result = i0*s0 + i1*s1 + ... + iN*sN
    dt = off_tv.dtype  # "i32" or "f32" depending on trace
    remaining = off_tv
    result = None

    for dim in range(ndim - 1, -1, -1):
        d = shape[dim]
        s = strides[dim]
        d_tv = b.splat(b.constant(d, dt), off_tv.shape)
        i_dim = b.binop("mod", remaining, d_tv) if dim > 0 else remaining
        if dim > 0:
            remaining = b.binop("floordiv", remaining, d_tv)
        if s == 0:
            continue  # broadcast dim, contributes nothing
        if s == 1:
            term = i_dim
        else:
            s_tv = b.splat(b.constant(s, dt), off_tv.shape)
            term = b.binop("mul", i_dim, s_tv)
        if result is None:
            result = term
        else:
            result = b.binop("add", result, term)

    if result is None:
        # All strides were 0 (pure broadcast) — every thread reads at offset 0
        result = b.splat(b.constant(0, dt), off_tv.shape)
    return result


def _decompose_ptr(tv: TracedValue) -> tuple[TileValue, TileValue]:
    """Walk pointer tracking to find the root buffer and offsets.

    The root buffer is the original buffer parameter (is_ptr=True, no ptr_base).
    The offsets are the full computed expression (tv._ptr_offsets._tv), which
    has the correct broadcast shape from TileBuilder arithmetic.
    """
    if tv._is_ptr and tv._ptr_base is None:
        # Bare buffer pointer with no offset — treat as offset 0 (scalar load).
        b = _ctx().builder
        zero = b.constant(0, "i32")
        return tv._tv, zero

    if tv._ptr_base is not None:
        # Walk up to find the root buffer param
        base = tv._ptr_base
        while base._ptr_base is not None:
            base = base._ptr_base
        if not base._is_ptr:
            raise ValueError("Pointer base is not a buffer param")
        # offsets = the full add expression (already has correct broadcast shape)
        off = tv._ptr_offsets._tv if tv._ptr_offsets is not None else tv._tv
        return base._tv, off

    raise ValueError("Not a pointer expression — cannot decompose for load/store")


# ---------------------------------------------------------------------------
# 2D address extraction — detect row[:, None] * stride + col[None, :]
# ---------------------------------------------------------------------------


def _extract_2d_addr_standard(offsets: TileValue):
    """Extract standard 2D addressing: row[:, None] * stride + col[None, :]."""
    if offsets.rank != 2:
        return None, None, None, None, False

    op_map = _ctx().op_map

    # Walk through Copy ops to find the underlying structure
    top = op_map.get(offsets.name)
    while isinstance(top, Copy) and top.source is not None:
        top = op_map.get(top.source.name)
    if not isinstance(top, BinOp) or top.op != "add":
        return None, None, None, None, False

    lhs, rhs = top.lhs, top.rhs
    if len(lhs.shape) == 2 and lhs.shape[1] == 1:
        row_val, col_val = lhs, rhs
    elif len(rhs.shape) == 2 and rhs.shape[1] == 1:
        row_val, col_val = rhs, lhs
    else:
        return None, None, None, None, False

    # Walk row_val to find mul(expanded_row, stride), accumulating EVERY scalar
    # addend on the way down. The address may carry more than one program-derived
    # scalar base — e.g. multi-head q/k loads address `bi*(S·NK·DK) + h_kv*DK +
    # (t0+rc)*stride + col`, where `bi*…` and `h_kv*DK` sit at different add
    # levels. Collect all and sum them.
    scalar_addends: list = []
    cur = op_map.get(row_val.name)
    while isinstance(cur, BinOp) and cur.op == "add":
        found = False
        for child, other in [(cur.lhs, cur.rhs), (cur.rhs, cur.lhs)]:
            child_op = op_map.get(child.name)
            if (
                isinstance(child_op, BinOp)
                and child_op.op == "mul"
                and child.shape
                and child.shape != ()
            ):
                if other.shape == ():
                    scalar_addends.append(other)
                cur = child_op
                found = True
                break
        if found:
            break
        walked = False
        for child, other in [(cur.lhs, cur.rhs), (cur.rhs, cur.lhs)]:
            if child.shape and len(child.shape) >= 1 and child.shape[-1] == 1:
                if other.shape == ():
                    scalar_addends.append(other)
                cur = op_map.get(child.name)
                walked = True
                break
        if not walked:
            break

    if not isinstance(cur, BinOp) or cur.op != "mul":
        return None, None, None, None, False

    scalar_addend = None
    for s in scalar_addends:
        scalar_addend = s if scalar_addend is None else _ctx().builder.binop("add", scalar_addend, s)

    a, b = cur.lhs, cur.rhs
    if a.shape == ():
        stride_val, expanded_val = a, b
    elif b.shape == ():
        stride_val, expanded_val = b, a
    else:
        return None, None, None, None, False

    stride_op = op_map.get(stride_val.name)
    if isinstance(stride_op, Constant):
        stride_int = int(stride_op.value)
    elif stride_val.name in _ctx().constexpr_values:
        stride_int = int(_ctx().constexpr_values[stride_val.name])
    else:
        return None, None, None, None, False

    exp_op = op_map.get(expanded_val.name)
    if isinstance(exp_op, ExpandDims):
        row_1d = exp_op.input
    else:
        return None, None, None, None, False

    col_exp = op_map.get(col_val.name)
    if isinstance(col_exp, ExpandDims):
        col_1d = col_exp.input
    else:
        col_1d = col_val

    return row_1d, col_1d, stride_int, scalar_addend


def _extract_2d_addr(offsets: TileValue):
    """Extract semantic 2D addressing from offsets IR.

    Handles both standard (row[:, None] * stride + col[None, :]) and
    transposed (col[None, :] * stride + row[:, None]) patterns.

    Returns (row_1d, col_1d, stride_int, scalar_addend, addr_transposed).
    """
    # Try standard pattern first
    result = _extract_2d_addr_standard(offsets)
    if result[0] is not None:
        return (*result, False)

    # Try swapped: stride on the [None, :] term instead of [:, None]
    if offsets.rank != 2:
        return None, None, None, None, False

    op_map = _ctx().op_map
    top = op_map.get(offsets.name)
    while isinstance(top, Copy) and top.source is not None:
        top = op_map.get(top.source.name)
    if not isinstance(top, BinOp) or top.op != "add":
        return None, None, None, None, False

    lhs, rhs = top.lhs, top.rhs
    # Identify the (N, 1) term as row_val and the (1, M) term as col_val
    if len(lhs.shape) == 2 and lhs.shape[1] == 1:
        row_val, col_val = lhs, rhs
    elif len(rhs.shape) == 2 and rhs.shape[1] == 1:
        row_val, col_val = rhs, lhs
    else:
        return None, None, None, None, False

    # Check if col_val has the stride (transposed pattern)
    # Walk through add(scalar, ...) to find the mul(expanded, stride)
    col_cur = op_map.get(col_val.name)
    scalar_addend = None
    while isinstance(col_cur, BinOp) and col_cur.op == "add":
        for child, other in [(col_cur.lhs, col_cur.rhs), (col_cur.rhs, col_cur.lhs)]:
            child_op = op_map.get(child.name)
            if (
                isinstance(child_op, BinOp)
                and child_op.op == "mul"
                and child.shape
                and child.shape != ()
            ):
                if other.shape == ():
                    scalar_addend = other
                col_cur = child_op
                break
        else:
            break
    if not isinstance(col_cur, BinOp) or col_cur.op != "mul":
        return None, None, None, None, False

    a, b = col_cur.lhs, col_cur.rhs
    if a.shape == ():
        stride_val, expanded_val = a, b
    elif b.shape == ():
        stride_val, expanded_val = b, a
    else:
        return None, None, None, None, False

    stride_op = op_map.get(stride_val.name)
    if isinstance(stride_op, Constant):
        stride_int = int(stride_op.value)
    elif stride_val.name in _ctx().constexpr_values:
        stride_int = int(_ctx().constexpr_values[stride_val.name])
    else:
        return None, None, None, None, False

    # The expanded column value should be expand_dims(..., axis=0) → (1, M)
    col_exp = op_map.get(expanded_val.name)
    if isinstance(col_exp, ExpandDims):
        col_1d = col_exp.input
    else:
        return None, None, None, None, False

    # The row value is expand_dims(..., axis=1) → (N, 1), no stride
    row_exp = op_map.get(row_val.name)
    if isinstance(row_exp, ExpandDims):
        row_1d = row_exp.input
    else:
        row_1d = row_val

    # Transposed: col has the stride, row has stride 1.
    # Return as (col_1d, row_1d, stride, scalar_addend) — col becomes the
    # "row" for the cooperative load since it carries the stride.
    # addr_transposed=True tells the coop load to scatter-store transposed.
    return col_1d, row_1d, stride_int, scalar_addend, True


def _match_stride_to_dim(buf_shape, pid_stride):
    """Match a pid stride to a buffer dimension in row-major layout.

    Returns (dim_idx, dim_size) if the stride corresponds to indexing
    dimension dim_idx, or None if no match.
    """
    stride_acc = 1
    for dim_idx in reversed(range(len(buf_shape))):
        if stride_acc == pid_stride:
            return dim_idx, buf_shape[dim_idx]
        stride_acc *= buf_shape[dim_idx]
    # Stride exceeds all individual dims → batch-level indexing.
    # The bound = total_elements / pid_stride (number of batches).
    total = 1
    for d in buf_shape:
        total *= d
    if pid_stride > 0 and total % pid_stride == 0:
        return -1, total // pid_stride
    return None


def _record_pid_dim_from_address(addr):
    """Detect which buffer dimension each pid indexes from address structure.

    For single-axis addresses, matches the pid stride from spec.axes against
    the buffer's row-major layout.

    For multi-axis addresses (pid_axis == "multi"), walks the IR per-axis
    via _find_store_pid_stride to recover individual strides, then matches
    each one independently.
    """
    if addr._ptr_base is None or addr._ptr_offsets is None:
        return
    offsets = addr._ptr_offsets
    pid_axis = offsets._pid_axis

    ctx = _ctx()
    spec = ctx.spec

    # Walk up to root buffer ptr
    base = addr._ptr_base
    while base._ptr_base is not None:
        base = base._ptr_base
    buf_name = base._tv.name
    buf_shape = ctx.shape_vars.get(buf_name) or ctx.builder.func.buffer_shapes.get(buf_name)
    if not buf_shape or len(buf_shape) < 1:
        return

    if pid_axis == "multi":
        # Multi-axis: resolve each unresolved axis independently.
        off_tv = offsets._tv
        for axis in list(spec.axes.keys()):
            if spec.axes[axis]["bound"] is not None and axis in spec.pid_dim_map:
                continue
            ax_info = spec.axes[axis]
            is_blocked = ax_info["block"] > 1

            if is_blocked:
                # Blocked pid (pid*BLOCK + arange): use grid-level stride for
                # standard dim matching. No total/stride — bound comes from
                # masks or the composite solver.
                grid_stride = ax_info.get("stride") or 1
                match = _match_stride_to_dim(buf_shape, grid_stride)
                if match is not None and match[0] >= 0:
                    dim_idx, dim_size = match
                    if ax_info["bound"] is None:
                        ax_info["bound"] = dim_size
                    spec.pid_dim_map[axis] = (buf_name, dim_idx, dim_size)
            else:
                # Scalar pid (batch-like): walk IR for actual cumulative stride,
                # then match including total/stride for batch detection.
                actual_stride = _find_store_pid_stride(off_tv, axis)
                if actual_stride <= 0:
                    continue
                match = _match_stride_to_dim(buf_shape, actual_stride)
                if match is not None:
                    dim_idx, dim_size = match
                    if ax_info["bound"] is None:
                        ax_info["bound"] = dim_size
                    if dim_idx >= 0:
                        spec.pid_dim_map[axis] = (buf_name, dim_idx, dim_size)
        return

    if not isinstance(pid_axis, int):
        return
    if pid_axis not in spec.axes:
        return
    if spec.axes[pid_axis]["bound"] is not None and pid_axis in spec.pid_dim_map:
        return

    pid_stride = spec.axes[pid_axis].get("stride") or 1
    match = _match_stride_to_dim(buf_shape, pid_stride)
    if match is not None:
        dim_idx, dim_size = match
        if spec.axes[pid_axis]["bound"] is None:
            spec.axes[pid_axis]["bound"] = dim_size
        if dim_idx >= 0:
            spec.pid_dim_map[pid_axis] = (buf_name, dim_idx, dim_size)


def _find_pid_axes_in_ir(tv: TileValue) -> set[int]:
    """Walk IR backward from tv to find all pid axes that contribute."""
    ctx = _ctx()
    pid_tvs = ctx.spec.pid_tvs
    op_map = ctx.op_map
    axes = set()
    visited = set()
    stack = [tv]
    while stack:
        cur = stack.pop()
        if cur.name in visited:
            continue
        visited.add(cur.name)
        if cur.name in pid_tvs:
            axes.add(pid_tvs[cur.name])
            continue
        op = op_map.get(cur.name)
        if isinstance(op, BinOp):
            stack.extend([op.lhs, op.rhs])
        elif isinstance(op, ExpandDims):
            stack.append(op.input)
        elif isinstance(op, Copy):
            if op.source:
                stack.append(op.source)
        elif isinstance(op, UnaryOp):
            stack.append(op.input)
    return axes


def _find_store_pid_stride(off_tv: TileValue, pid_axis: int) -> int:
    """Find the stride of a pid axis in a store address expression.

    Walks the IR backward from off_tv to find mul(pid, constant).
    Returns 1 if the pid appears without multiplication, 0 if not found.
    """
    ctx = _ctx()
    op_map = ctx.op_map
    pid_tvs = ctx.spec.pid_tvs

    def _walk(tv, depth=0):
        if depth > 20:
            return None
        if tv.name in pid_tvs and pid_tvs[tv.name] == pid_axis:
            return 1
        op = op_map.get(tv.name)
        if op is None:
            return None
        if isinstance(op, BinOp) and op.op == "mul":
            for val, other in [(op.lhs, op.rhs), (op.rhs, op.lhs)]:
                sub = _walk(val, depth + 1)
                if sub is not None:
                    other_op = op_map.get(other.name)
                    if isinstance(other_op, Constant):
                        return sub * int(other_op.value)
                    if other.name in ctx.constexpr_values:
                        return sub * int(ctx.constexpr_values[other.name])
                    return sub
            return None
        if isinstance(op, BinOp) and op.op == "add":
            l = _walk(op.lhs, depth + 1)  # noqa: E741
            r = _walk(op.rhs, depth + 1)
            if l is not None and r is None:
                return l
            if r is not None and l is None:
                return r
            if l is not None and r is not None:
                return max(l, r)
            return None
        if isinstance(op, ExpandDims):
            return _walk(op.input, depth + 1)
        if isinstance(op, Copy) and op.source:
            return _walk(op.source, depth + 1)
        return None

    result = _walk(off_tv)
    return result if result is not None else 0


def _build_write_pattern(
    param_name, addr, value, off_tv, row_indices, col_indices, row_stride, base_offset
):
    """Build an OutputWritePattern from store address analysis.

    Determines output dimensionality from the store address structure,
    NOT from the tile value shape.
    """

    ctx = _ctx()
    spec = ctx.spec
    dtype = value._tv.dtype
    value_shape = value._tv.shape

    # Case 1: 2D address (detected by _extract_2d_addr)
    if row_stride is not None and row_indices is not None:
        row_pids = _find_pid_axes_in_ir(row_indices)
        if base_offset is not None:
            row_pids |= _find_pid_axes_in_ir(base_offset)
        dim0 = ("bound", sorted(row_pids)) if row_pids else ("const", row_stride)
        dim1 = ("const", row_stride)
        return OutputWritePattern(param_name, dtype, value_shape, [dim0, dim1])

    # Non-2D: analyze the store offsets directly
    off_pid_axis = None
    if addr._ptr_offsets is not None:
        off_pid_axis = addr._ptr_offsets._pid_axis

    if isinstance(off_pid_axis, int) and off_pid_axis in spec.axes:
        # Find the pid stride in the store address (not global spec.axes stride)
        store_stride = _find_store_pid_stride(off_tv, off_pid_axis)
        off_shape = off_tv.shape if off_tv else ()

        if off_shape == () or store_stride <= 1:
            # Case 2: Scalar store (out + pid) or pid with stride 1
            return OutputWritePattern(param_name, dtype, value_shape, [("grid", [off_pid_axis])])

        # Distinguish blocked flat store vs row-per-pid 2D store.
        # Blocked flat: pid is part of a blocked offset (block > 1 in spec.axes),
        # meaning pid*BLOCK + arange tiles one contiguous dimension → 1D output.
        # Row-per-pid: pid is scalar (block == 1), indexes rows while arange
        # indexes columns within each row → 2D output.
        ax_block = spec.axes[off_pid_axis]["block"]
        if ax_block > 1:
            # Case 3a: Blocked flat store (pid*BLOCK + arange) → 1D output
            return OutputWritePattern(param_name, dtype, value_shape, [("bound", [off_pid_axis])])

        # Case 3b: Row-per-pid pattern (softmax, layernorm) → 2D output
        return OutputWritePattern(
            param_name,
            dtype,
            value_shape,
            [("bound", [off_pid_axis]), ("const", store_stride)],
        )

    elif off_pid_axis == "multi":
        # Multi-axis in store address — decompose per-axis from IR.
        # Find each axis's stride in the store address to determine
        # which axis is "rows" (highest stride) and which is "cols".
        axis_strides = {}
        for ax in spec.axes:
            s = _find_store_pid_stride(off_tv, ax)
            if s > 0:
                axis_strides[ax] = s
        if len(axis_strides) >= 2:
            # Highest stride axis indexes rows, lowest indexes cols
            sorted_axes = sorted(axis_strides, key=lambda a: axis_strides[a], reverse=True)
            row_ax = sorted_axes[0]
            # row extent = bound of row axis, col extent = row axis stride
            # (row_stride = distance between rows = number of cols)
            return OutputWritePattern(
                param_name,
                dtype,
                value_shape,
                [("bound", [row_ax]), ("const", axis_strides[row_ax])],
            )
        # Single axis resolvable from multi — treat like single-axis
        if len(axis_strides) == 1:
            ax = next(iter(axis_strides))
            s = axis_strides[ax]
            if s > 1:
                return OutputWritePattern(
                    param_name, dtype, value_shape, [("bound", [ax]), ("const", s)]
                )
            return OutputWritePattern(param_name, dtype, value_shape, [("grid", [ax])])
        # No strides found — fall back to flat
        all_axes = sorted(spec.axes.keys())
        return OutputWritePattern(param_name, dtype, value_shape, [("bound", all_axes)])

    else:
        # No pid in store address. If there ARE pid axes with bounds
        # (e.g., transpose where floordiv/mod break pid tracking),
        # use product of all bounds as flat output extent.
        axes_with_bounds = [ax for ax, info in spec.axes.items() if info.get("bound") is not None]
        if axes_with_bounds:
            return OutputWritePattern(
                param_name, dtype, value_shape, [("bound", sorted(axes_with_bounds))]
            )
        arange_size = off_tv.shape[0] if off_tv and off_tv.shape else 1
        return OutputWritePattern(param_name, dtype, value_shape, [("const", arange_size)])
