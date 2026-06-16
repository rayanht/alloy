"""Elementwise transform MSL emitter methods."""

from __future__ import annotations

from alloy._compiler.dtypes import from_ir
from alloy._compiler.fusion_transforms import StridedTransform
from alloy._compiler.msl.math import (
    BINOP_FUNC_MSL,
    BINOP_MSL,
    CMPOP_MSL,
    UNARY_FUNC_MSL,
    UNARY_MSL,
    fmt_const,
    get_extra_transform,
)
from alloy._compiler.tile_ir import (
    BinOp,
    BoolOp,
    Cast,
    Compare,
    Constant,
    Select,
    TernaryOp,
    TileValue,
    UnaryOp,
)


class TransformEmitterMixin:
    def _emit_inline_transform(
        self,
        transform: list,
        type_name: str,
        var_name: str,
        broadcast_scalar: bool = False,
        row_stride: int | None = None,
        chain_source_name: str | None = None,
        extra_transforms: dict | None = None,
        preload_extras: dict[str, str] | None = None,
    ):
        """Emit inline elementwise transform on a variable.

        Unified handler for both load transforms (vec4, broadcast_scalar=True)
        and store transforms (scalar, broadcast_scalar=False).
        extra_transforms: dict[str, IndexTransform] from the IR node.
        preload_extras: optional dict[str, str] mapping an extra buffer
            param name to a pre-emitted MSL expression. When present, the
            transform reads the extra from this expression instead of
            emitting a fresh scalar load of `name[idx]`. Used by the MMA
            epilogue to vectorize adjacent-column reads.
        """
        self._current_extras = extra_transforms
        # Count uses for multi-use temporaries
        _OPERAND_ATTRS = ("lhs", "rhs", "input", "a", "b", "c", "cond", "true_val", "false_val")
        use_count: dict[str, int] = {}
        for t_op in transform:
            for v in t_op.operand_values():
                use_count[v.name] = use_count.get(v.name, 0) + 1

        # Find the source name (non-produced, non-constant input)
        source_name = chain_source_name
        if source_name is None:
            for t_op in transform:
                if isinstance(t_op, Constant):
                    continue
                for v in t_op.operand_values():
                    produced = any(t2.result and t2.result.name == v.name for t2 in transform)
                    if not produced:
                        source_name = v.name
                        break
                if source_name:
                    break

        expr_map: dict[str, str] = {}
        if source_name:
            expr_map[source_name] = var_name
        if preload_extras:
            expr_map.update(preload_extras)
        tmp_idx = 0

        def _resolve(val):
            if val is None:
                return "0"
            name = val.name
            if name in expr_map:
                expr = expr_map[name]
                if broadcast_scalar and val.shape == ():
                    if type_name in ("bfloat4", "bfloat2", "half4", "half2"):
                        elem = type_name.rstrip("0123456789")
                        return f"{type_name}({elem}({expr}))"
                    return f"{type_name}({expr})"
                return expr
            if val.shape == () and any(p.name == name for p in self.func.params):
                xf = get_extra_transform(name, self._current_extras)
                if not broadcast_scalar:
                    # Store transform: use IndexTransform for 2D tile addressing
                    idx_expr = xf.tile_2d("_gm", "_gn")
                    return f"float({name}[{idx_expr}])"
                # Load transform: boundary-safe vectorized read
                # Extract row stride from the transform for vec4 addressing
                rs = (
                    xf.row_stride
                    if isinstance(xf, StridedTransform)
                    else (
                        row_stride
                        if row_stride is not None
                        else self.func.constexpr_values.get("N", 1)
                    )
                )
                K_bound = self.func.constexpr_values.get(
                    "K", self.func.constexpr_values.get("N", 0)
                )
                M_bound = self.func.constexpr_values.get("M", 0)
                if K_bound and M_bound:
                    return (
                        f"((_gr < {M_bound} && _gc + 3u < {K_bound}) "
                        f"? *(device const {type_name}*)({name} + _gr * {rs}u + _gc) "
                        f": {type_name}(0.0f))"
                    )
                return f"*(device const {type_name}*)({name} + _gr * {rs}u + _gc)"
            scalar = self._resolve_scalar(val)
            if broadcast_scalar and val.shape == ():
                # Metal rejects functional-style cast `bfloat4(0.125f)` —
                # scalar→vector requires a same-dtype scalar first. Pre-
                # cast to the element type for half/bfloat vec types.
                if type_name in ("bfloat4", "bfloat2", "half4", "half2"):
                    elem = type_name.rstrip("0123456789")
                    return f"{type_name}({elem}({scalar}))"
                return f"{type_name}({scalar})"
            return scalar

        for t_op in transform:
            if isinstance(t_op, Constant):
                expr_map[t_op.result.name] = fmt_const(t_op.value)
                continue

            if isinstance(t_op, BinOp):
                func_name = BINOP_FUNC_MSL.get(t_op.op)
                l, r = _resolve(t_op.lhs), _resolve(t_op.rhs)  # noqa: E741
                if func_name:
                    result_expr = f"{func_name}({l}, {r})"
                else:
                    sym = BINOP_MSL.get(t_op.op)
                    if not sym:
                        continue
                    result_expr = f"({l} {sym} {r})"
            elif isinstance(t_op, UnaryOp):
                func_name = UNARY_FUNC_MSL.get(t_op.op)
                inp = _resolve(t_op.input)
                if func_name:
                    result_expr = f"{func_name}({inp})"
                else:
                    sym = UNARY_MSL.get(t_op.op)
                    if not sym:
                        continue
                    result_expr = f"({sym}{inp})"
            elif isinstance(t_op, TernaryOp):
                a, b, c = _resolve(t_op.a), _resolve(t_op.b), _resolve(t_op.c)
                result_expr = f"{t_op.op}({a}, {b}, {c})"
            elif isinstance(t_op, Select):
                cond = _resolve(t_op.cond)
                tv = _resolve(t_op.true_val)
                fv = _resolve(t_op.false_val)
                result_expr = f"(({cond}) ? ({tv}) : ({fv}))"
            elif isinstance(t_op, Compare):
                l, r = _resolve(t_op.lhs), _resolve(t_op.rhs)  # noqa: E741
                sym = CMPOP_MSL.get(t_op.op, "==")
                result_expr = f"({l} {sym} {r})"
            elif isinstance(t_op, BoolOp):
                l, r = _resolve(t_op.lhs), _resolve(t_op.rhs)  # noqa: E741
                sym = "&&" if t_op.op == "and" else "||"
                result_expr = f"({l} {sym} {r})"
            elif isinstance(t_op, Cast):
                inp = _resolve(t_op.input)
                msl_type = from_ir(t_op.target_dtype).msl
                result_expr = f"_alloy_cast_{msl_type}({inp})"
            else:
                continue

            name = t_op.result.name
            if use_count.get(name, 0) > 1:
                tmp_name = f"_t{tmp_idx}"
                tmp_idx += 1
                self._emit(f"{type_name} {tmp_name} = {type_name}({result_expr});")
                expr_map[name] = tmp_name
            else:
                expr_map[name] = result_expr

        # Assign final expression to the variable
        last_name = None
        for t_op in reversed(transform):
            if not isinstance(t_op, Constant) and t_op.result:
                last_name = t_op.result.name
                break
        if last_name and last_name in expr_map:
            final_expr = expr_map[last_name]
            if final_expr != var_name:
                self._emit(f"{var_name} = {type_name}({final_expr});")

    def _emit_load_transform(self, transform: list, vec_type: str):
        """Emit inline elementwise transform on _val (vec4)."""
        self._emit_inline_transform(transform, vec_type, "_val", broadcast_scalar=True)

    def _emit_store_transform(
        self,
        transform: list,
        scalar_type: str,
        var_name: str,
        row_stride: int | None = None,
        chain_source_name: str | None = None,
        extra_transforms: dict | None = None,
        preload_extras: dict[str, str] | None = None,
    ):
        """Emit inline elementwise transform on a scalar store value."""
        self._emit_inline_transform(
            transform,
            scalar_type,
            var_name,
            broadcast_scalar=False,
            row_stride=row_stride,
            chain_source_name=chain_source_name,
            extra_transforms=extra_transforms,
            preload_extras=preload_extras,
        )

    def _resolve_scalar(self, val: TileValue) -> str:
        """Resolve a scalar TileValue to an MSL expression, even if not yet emitted."""
        expr = self._exprs.get(val.name)
        if expr is not None:
            return expr
        op = self._op_map.get(val.name)
        if isinstance(op, Constant):
            return fmt_const(op.value)
        return val.name
