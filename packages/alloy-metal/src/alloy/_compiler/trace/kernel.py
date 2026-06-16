"""Kernel trace entrypoint."""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Callable, Mapping
from typing import cast

from alloy._compiler.dispatch_spec import (
    AxisSpec,
    CeilDiv,
    Const,
    DispatchContract,
    Expr,
    Mul,
    OutputSpec,
    OutputWritePattern,
    Sym,
)
from alloy._compiler.tile_ir import TileBuilder, TileFunction, TileOp, TileValue
from alloy._compiler.trace.ast_rewrite import _rewrite_kernel_source
from alloy._compiler.trace.control_flow import (
    _trace_flow,
    _trace_for_var,
    _trace_if_else,
    _trace_if_enter,
    _trace_if_exit,
    _trace_loop_cond,
    _trace_loop_enter,
    _trace_loop_exit,
)
from alloy._compiler.trace.value import (
    ConstexprValue,
    TracedValue,
    SpecBuilder,
    TraceCtx,
    _tls,
)

TraceCallable = Callable[..., None]
TraceCallArg = TracedValue | ConstexprValue

# ---------------------------------------------------------------------------
# trace_kernel — the main entry point
# ---------------------------------------------------------------------------


def trace_kernel(
    fn,
    kernel_name: str,
    constexpr_values: Mapping[str, ConstexprValue],
    buffer_dtypes: dict[str, str] | None = None,
    param_names: list[str] | None = None,
    constexpr_params: set[str] | None = None,
    source: str | None = None,
    buffer_shapes: dict[str, tuple[int, ...]] | None = None,
    output_params: set[str] | None = None,
) -> TileFunction:
    """Trace a kernel function to produce tile IR.

    Instead of parsing the AST, we execute fn() with TracedValue proxies
    for buffer params and concrete values for constexpr params. Each DSL
    operation builds TileOps into a TileBuilder.

    Args:
        fn: The kernel function to trace.
        kernel_name: Name for the generated kernel.
        constexpr_values: Dict of constexpr param name -> value.
        buffer_dtypes: Dict of buffer param name -> alloy dtype string.
        param_names: Ordered list of parameter names.
        constexpr_params: Set of constexpr param names.
        source: Kernel source code (for AST rewrite).
        buffer_shapes: Dict of buffer param name -> array shape.
                       When provided, TracedValue.shape returns the array
                       shape so kernel code can use x.shape[0] etc.

    Returns:
        TileFunction ready for tile_opt/tile_plan/tile_msl.
    """
    constexpr_values = dict(constexpr_values)
    buffer_dtypes = buffer_dtypes or {}

    if param_names is None or constexpr_params is None:
        sig = inspect.signature(fn)
        if param_names is None:
            param_names = list(sig.parameters.keys())
        if constexpr_params is None:
            constexpr_params = set()
            for pname, param in sig.parameters.items():
                ann = param.annotation
                if ann is inspect.Parameter.empty:
                    continue
                if isinstance(ann, str) and ann in {"constexpr", "al.constexpr"}:
                    constexpr_params.add(pname)
                elif isinstance(ann, type) and ann.__name__ == "ConstExprType":
                    constexpr_params.add(pname)

    builder = TileBuilder(kernel_name)
    buffer_params: dict[str, TracedValue] = {}
    call_args: dict[str, TraceCallArg] = {}

    buffer_shapes = buffer_shapes or {}
    builder.func.buffer_shapes = dict(buffer_shapes)

    for pname in param_names:
        if pname in constexpr_params:
            builder.add_param(pname, is_constexpr=True)
            val = constexpr_values.get(pname)
            if val is None:
                sig = inspect.signature(fn)
                p = sig.parameters.get(pname)
                if p is not None and p.default is not inspect.Parameter.empty:
                    val = cast(ConstexprValue, p.default)
            call_args[pname] = val
        else:
            dtype = buffer_dtypes.get(pname, "f32")
            tv = builder.add_param(pname, is_constexpr=False, dtype=dtype)
            assert tv is not None
            arr_shape = buffer_shapes.get(pname)
            traced = TracedValue(tv, is_ptr=True, array_shape=arr_shape, array_dtype=dtype)
            buffer_params[pname] = traced
            call_args[pname] = traced

    builder.set_constexprs(constexpr_values)

    if output_params is None:
        output_params = set()

    spec_builder = SpecBuilder(
        pid_tvs={},
        arange_tvs={},
        axes={},
        composite_bounds=[],
        output_writes={},
        bindings={},
        pid_dim_map={},
        output_params=output_params,
    )
    ctx = TraceCtx(
        builder=builder,
        op_map={},
        buffer_params=buffer_params,
        constexpr_values=constexpr_values,
        alloc_vars=set(),
        shape_vars={},
        spec=spec_builder,
        buffer_dtypes=dict(buffer_dtypes) if buffer_dtypes else {},
    )

    orig_add_op = builder.func.add_op

    def _tracking_add_op(op: TileOp) -> TileValue | None:
        if op.result:
            ctx.op_map[op.result.name] = op
        return orig_add_op(op)

    builder.func.add_op = _tracking_add_op

    trace_fn: TraceCallable = fn
    if source is None:
        source_attr = fn.__dict__.get("_alloy_source")
        if isinstance(source_attr, str):
            source = source_attr
    if source is None:
        source = textwrap.dedent(inspect.getsource(fn))
    rewritten_tree = _rewrite_kernel_source(ast.parse(source))
    if rewritten_tree is not None:
        code = compile(rewritten_tree, f"<traced:{kernel_name}>", "exec")
        ns = {
            **fn.__globals__,
            "_trace_loop_enter": _trace_loop_enter,
            "_trace_for_var": _trace_for_var,
            "_trace_loop_cond": _trace_loop_cond,
            "_trace_loop_exit": _trace_loop_exit,
            "_trace_if_enter": _trace_if_enter,
            "_trace_if_else": _trace_if_else,
            "_trace_if_exit": _trace_if_exit,
            "_trace_flow": _trace_flow,
        }
        # Inject closure variables so rewritten functions can access them
        if fn.__closure__ and fn.__code__.co_freevars:
            for name, cell in zip(fn.__code__.co_freevars, fn.__closure__):
                ns[name] = cell.cell_contents
        exec(code, ns)
        for node in ast.walk(rewritten_tree):
            if isinstance(node, ast.FunctionDef):
                trace_fn = cast(TraceCallable, ns[node.name])
                break

    _tls.trace_ctx = ctx
    try:
        pos_args = [call_args[p] for p in param_names]
        trace_fn(*pos_args)
    finally:
        _tls.trace_ctx = None
        builder.func.add_op = orig_add_op

    func = builder.build()
    func.shape_vars = ctx.shape_vars
    if buffer_shapes:
        func.buffer_shapes = dict(buffer_shapes)

    sb = ctx.spec

    # --- Composite constraint solving for missing axis bounds ---
    for total, strides in sb.composite_bounds:
        missing = [ax for ax in sb.axes if sb.axes[ax]["bound"] is None]
        if not missing:
            break
        known_product = 1
        for ax, info in sb.axes.items():
            if info["bound"] is not None and ax not in missing:
                known_product *= info["bound"]
        if len(missing) == 1 and known_product > 0 and total % known_product == 0:
            sb.axes[missing[0]]["bound"] = total // known_product
        elif len(missing) >= 2 and strides:
            sorted_missing = sorted(missing, key=lambda ax: strides.get(ax, 0), reverse=True)
            remaining = total // known_product
            for ax in sorted_missing:
                s = strides.get(ax)
                if s and s > 0 and remaining % s == 0:
                    sb.axes[ax]["bound"] = remaining // s
                    remaining = s
                elif remaining > 0:
                    sb.axes[ax]["bound"] = remaining
                    remaining = 1

    # --- Helper: build an Expr for an axis bound ---
    def _bound_expr(axis: int) -> Expr | None:
        """Return an expression for an axis bound, or None if unresolved.

        Uses Sym when the bound traces to a buffer shape dimension (enabling
        symbolic re-evaluation), Const for purely constexpr-derived bounds.
        The FromInputShape binding is always registered so resolve_bindings
        can provide the concrete value at dispatch time.
        """
        info = sb.axes.get(axis)
        if info is None or info["bound"] is None:
            return None
        mapping = sb.pid_dim_map.get(axis)
        if mapping is not None:
            buf_name, dim_idx, _ = mapping
            sym_name = f"{buf_name}_dim{dim_idx}"
            if sym_name in sb.bindings:
                return Sym(sym_name)
        return Const(info["bound"])

    # --- Build grid axes (symbolic bounds where possible) ---
    grid_axes: dict[int, AxisSpec] = {}
    unresolved: list[int] = []
    for axis, info in sb.axes.items():
        arange_block = info["block"]
        assert arange_block is not None
        pid_stride = info.get("stride")
        if arange_block > 1 and isinstance(pid_stride, int) and pid_stride > arange_block:
            effective_block = pid_stride
        else:
            effective_block = arange_block
        if info["bound"] is None:
            unresolved.append(axis)
            continue
        bound = _bound_expr(axis)
        assert bound is not None
        grid_axes[axis] = AxisSpec(block=Const(effective_block), bound=bound)

    # --- Output shape from write patterns (symbolic where possible) ---
    outputs: dict[str, OutputSpec] = {}
    unresolved_outputs: dict[str, list[int]] = {}
    for pname, patterns in sb.output_writes.items():
        if not patterns:
            continue

        # Validate: all patterns with same ndim must agree on dim kinds
        by_ndim: dict[int, list[OutputWritePattern]] = {}
        for p in patterns:
            by_ndim.setdefault(len(p.dims), []).append(p)
        for ndim, group in by_ndim.items():
            if len(group) < 2:
                continue
            ref = group[0]
            for other in group[1:]:
                for i, ((k1, _), (k2, _)) in enumerate(zip(ref.dims, other.dims)):
                    if k1 != k2:
                        raise RuntimeError(
                            f"Incompatible write patterns for output '{pname}': "
                            f"store dim {i} has kind '{k1}' vs '{k2}'. "
                            f"Cannot merge contradictory rank/dim mappings."
                        )

        # Select best pattern: highest ndim, then largest tile
        best = max(
            patterns,
            key=lambda p: (
                len(p.dims),
                (
                    (p.value_shape[0] * p.value_shape[1])
                    if len(p.value_shape) >= 2
                    else p.value_shape[0]
                    if p.value_shape
                    else 0
                ),
            ),
        )

        shape_dims: list[Expr] = []
        for dim_idx, (kind, info) in enumerate(best.dims):
            if kind == "const":
                shape_dims.append(Const(info))
            elif kind == "bound":
                # Symbolic product of bounds for the listed pid axes
                expr: Expr | None = None
                all_resolved = True
                for ax in info:
                    be = _bound_expr(ax)
                    if be is None:
                        unresolved_outputs.setdefault(pname, []).append(dim_idx)
                        all_resolved = False
                        break
                    expr = be if expr is None else Mul(expr, be)
                if all_resolved and expr is not None:
                    shape_dims.append(expr)
                elif not all_resolved:
                    shape_dims.append(Const(1))  # placeholder
            elif kind == "grid":
                # Symbolic grid count: ceildiv(bound, block) per axis
                expr: Expr | None = None
                all_resolved = True
                for ax in info:
                    be = _bound_expr(ax)
                    if be is None:
                        unresolved_outputs.setdefault(pname, []).append(dim_idx)
                        all_resolved = False
                        break
                    ga = grid_axes.get(ax)
                    ax_expr = CeilDiv(be, ga.block) if ga is not None else be
                    expr = ax_expr if expr is None else Mul(expr, ax_expr)
                if all_resolved and expr is not None:
                    shape_dims.append(expr)
                elif not all_resolved:
                    shape_dims.append(Const(1))  # placeholder

        if shape_dims:
            outputs[pname] = OutputSpec(shape=tuple(shape_dims), dtype=best.dtype)
        else:
            outputs[pname] = OutputSpec(shape=(Const(1),), dtype=best.dtype)

    if grid_axes or outputs or unresolved:
        func.dispatch_spec = DispatchContract(
            grid_axes=grid_axes,
            outputs=outputs,
            bindings=dict(sb.bindings),
            unresolved_axes=unresolved,
            unresolved_outputs=unresolved_outputs,
        )

    return func
