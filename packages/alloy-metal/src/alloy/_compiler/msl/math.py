"""Scalar expression formatting for MSL emission."""

from __future__ import annotations

import math
from collections.abc import Callable

from alloy._compiler.dtypes import ALLOY_TO_MSL, from_ir
from alloy._compiler.fusion_transforms import IdentityTransform, IndexTransform
from alloy._compiler.tile_ir import (
    BinOp,
    BoolOp,
    Cast,
    Compare,
    Constant,
    Select,
    TernaryOp,
    TileOp,
    TileValue,
    UnaryOp,
)

MSLConstant = bool | int | float | str | None
GetExpr = Callable[[TileValue], str]

BINOP_MSL: dict[str, str] = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
    "mod": "%",
    "floordiv": "/",
    "and": "&",
    "or": "|",
    "bitand": "&",
    "bitor": "|",
    "bitxor": "^",
    "lshift": "<<",
    "rshift": ">>",
}

CMPOP_MSL: dict[str, str] = {
    "lt": "<",
    "le": "<=",
    "gt": ">",
    "ge": ">=",
    "eq": "==",
    "ne": "!=",
}

UNARY_MSL: dict[str, str] = {
    "neg": "-",
    "not": "!",
    "bitnot": "~",
}

UNARY_FUNC_MSL: dict[str, str] = {
    "exp": "exp",
    "log": "log",
    "sqrt": "sqrt",
    "rsqrt": "rsqrt",
    "tanh": "tanh",
    "erf": "_alloy_erf",
    "sin": "sin",
    "cos": "cos",
    "abs": "abs",
    "ceil": "ceil",
    "floor": "floor",
    "round": "round",
    "exp2": "exp2",
    "log2": "log2",
}

BINOP_FUNC_MSL: dict[str, str] = {
    "max": "max",
    "min": "min",
    "maximum": "max",
    "minimum": "min",
}

I64_HELPERS = """inline ulong _alloy_u64_add(ulong a, ulong b) {
    uint2 av = as_type<uint2>(a);
    uint2 bv = as_type<uint2>(b);
    uint lo = av.x + bv.x;
    uint carry = lo < av.x ? 1u : 0u;
    uint hi = av.y + bv.y + carry;
    return as_type<ulong>(uint2(lo, hi));
}

inline ulong _alloy_u64_sub(ulong a, ulong b) {
    uint2 av = as_type<uint2>(a);
    uint2 bv = as_type<uint2>(b);
    uint lo = av.x - bv.x;
    uint borrow = av.x < bv.x ? 1u : 0u;
    uint hi = av.y - bv.y - borrow;
    return as_type<ulong>(uint2(lo, hi));
}

inline long _alloy_i64_add(long a, long b) {
    return as_type<long>(_alloy_u64_add(as_type<ulong>(a), as_type<ulong>(b)));
}

inline long _alloy_i64_sub(long a, long b) {
    return as_type<long>(_alloy_u64_sub(as_type<ulong>(a), as_type<ulong>(b)));
}

// Mixed-precision overloads for max/min — resolves ambiguity when
// half and float operands are mixed (e.g. shmem half vs float literal).
inline float max(half a, float b) { return max(float(a), b); }
inline float max(float a, half b) { return max(a, float(b)); }
inline float min(half a, float b) { return min(float(a), b); }
inline float min(float a, half b) { return min(a, float(b)); }
// Type-cast helpers that work on both scalars and vec<T, N>. Needed because
// the prologue/epilogue eval_expr_chain may run inside a cooperative-load
// vector context (e.g. `bfloat4`) where the bare `float(v)` cast Metal emits
// for `Cast` is rejected (`functional-style cast from 'bfloat4' to 'float'
// is not allowed`). The vector overload returns vec<float, N> element-wise.
template<typename T> static inline float _alloy_cast_float(T v) { return float(v); }
template<typename T, int N> static inline vec<float, N> _alloy_cast_float(vec<T, N> v) { return vec<float, N>(v); }
template<typename T> static inline bfloat _alloy_cast_bfloat(T v) { return bfloat(v); }
template<typename T, int N> static inline vec<bfloat, N> _alloy_cast_bfloat(vec<T, N> v) { return vec<bfloat, N>(v); }
template<typename T> static inline half _alloy_cast_half(T v) { return half(v); }
template<typename T, int N> static inline vec<half, N> _alloy_cast_half(vec<T, N> v) { return vec<half, N>(v); }
template<typename T> static inline int _alloy_cast_int(T v) { return int(v); }
template<typename T, int N> static inline vec<int, N> _alloy_cast_int(vec<T, N> v) { return vec<int, N>(v); }
template<typename T> static inline ushort _alloy_cast_ushort(T v) { return ushort(v); }
template<typename T, int N> static inline vec<ushort, N> _alloy_cast_ushort(vec<T, N> v) { return vec<ushort, N>(v); }
template<typename T> static inline long _alloy_cast_long(T v) { return long(v); }
template<typename T, int N> static inline vec<long, N> _alloy_cast_long(vec<T, N> v) { return vec<long, N>(v); }
template<typename T> static inline uint _alloy_cast_uint(T v) { return uint(v); }
template<typename T, int N> static inline vec<uint, N> _alloy_cast_uint(vec<T, N> v) { return vec<uint, N>(v); }
// Width-polymorphic bit REINTERPRET (as_type, not a numeric convert — no convert
// pipe). The scalar `as_type<float>` form is a size-mismatch error on a vec<int,4>
// operand, so route bitcast-to-float through overloads that pick the matching
// scalar/vector target (used by the Q6_K/Q4_K int->float dequant bit-trick).
template<typename T> static inline float _alloy_bitcast_float(T v) { return as_type<float>(v); }
template<typename T, int N> static inline vec<float, N> _alloy_bitcast_float(vec<T, N> v) { return as_type<vec<float, N>>(v); }
template<typename T> static inline uchar _alloy_cast_uchar(T v) { return uchar(v); }
template<typename T, int N> static inline vec<uchar, N> _alloy_cast_uchar(vec<T, N> v) { return vec<uchar, N>(v); }
template<typename T> static inline ulong _alloy_cast_ulong(T v) { return ulong(v); }
template<typename T, int N> static inline vec<ulong, N> _alloy_cast_ulong(vec<T, N> v) { return vec<ulong, N>(v); }
template<typename T> static inline short _alloy_cast_short(T v) { return short(v); }
template<typename T, int N> static inline vec<short, N> _alloy_cast_short(vec<T, N> v) { return vec<short, N>(v); }
template<typename T> static inline char _alloy_cast_char(T v) { return char(v); }
template<typename T, int N> static inline vec<char, N> _alloy_cast_char(vec<T, N> v) { return vec<char, N>(v); }
// Abramowitz & Stegun erf approximation (max error < 1.5e-7).
// Metal stdlib does not provide erf().
inline float _alloy_erf(float x) {
    float ax = abs(x);
    float t = 1.0f / (1.0f + 0.3275911f * ax);
    float t2 = t * t;
    float y = 1.0f - (0.254829592f * t - 0.284496736f * t2
        + 1.421413741f * (t2 * t) - 1.453152027f * (t2 * t2)
        + 1.061405429f * (t2 * t2 * t)) * exp(-ax * ax);
    return x < 0.0f ? -y : y;
}
"""


def fmt_const(value: MSLConstant) -> str:
    """Format a Python constant as an MSL literal."""
    if isinstance(value, float):
        if math.isinf(value):
            return "-INFINITY" if value < 0 else "INFINITY"
        if value != int(value) or value == 0.0:
            return f"{value}f"
        if value < 0 and abs(value) >= 1e20:
            return f"({value}f)"
        return f"{value}"
    return str(value)


def format_scalar_op(op: TileOp, get_expr: GetExpr) -> str | None:
    """Format a scalar tile IR op as an MSL expression string."""
    if isinstance(op, Constant):
        return fmt_const(op.value)
    if isinstance(op, BinOp):
        l = get_expr(op.lhs)  # noqa: E741
        r = get_expr(op.rhs)
        if op.result.dtype == "i64":
            if op.op == "add":
                return f"_alloy_i64_add({l}, {r})"
            if op.op == "sub":
                return f"_alloy_i64_sub({l}, {r})"
        if op.result.dtype == "u64":
            if op.op == "add":
                return f"_alloy_u64_add({l}, {r})"
            if op.op == "sub":
                return f"_alloy_u64_sub({l}, {r})"
        func_name = BINOP_FUNC_MSL.get(op.op)
        if func_name:
            return f"{func_name}({l}, {r})"
        sym = BINOP_MSL.get(op.op)
        if sym:
            return f"({l} {sym} {r})"
        return f"/* unknown binop {op.op} */({l}, {r})"
    if isinstance(op, UnaryOp):
        inp = get_expr(op.input)
        if op.op == "bitcast":
            target_msl = ALLOY_TO_MSL.get(op.result.dtype, "float")
            if target_msl == "float":
                # Width-polymorphic so it works in a vec4 (opaque_vec) context.
                return f"_alloy_bitcast_float({inp})"
            return f"as_type<{target_msl}>({inp})"
        func_name = UNARY_FUNC_MSL.get(op.op)
        if func_name:
            return f"{func_name}({inp})"
        sym = UNARY_MSL.get(op.op)
        if sym:
            return f"({sym}{inp})"
        return f"/* unknown unary {op.op} */({inp})"
    if isinstance(op, Compare):
        l = get_expr(op.lhs)  # noqa: E741
        r = get_expr(op.rhs)
        return f"({l} {CMPOP_MSL[op.op]} {r})"
    if isinstance(op, BoolOp):
        l = get_expr(op.lhs)  # noqa: E741
        r = get_expr(op.rhs)
        sym = "&&" if op.op == "and" else "||"
        return f"({l} {sym} {r})"
    if isinstance(op, TernaryOp):
        a = get_expr(op.a)
        b = get_expr(op.b)
        c = get_expr(op.c)
        return f"{op.op}({a}, {b}, {c})"
    if isinstance(op, Select):
        c = get_expr(op.cond)
        t = get_expr(op.true_val)
        f = get_expr(op.false_val)
        return f"(({c}) ? ({t}) : ({f}))"
    return None


def get_extra_transform(name: str, extras: dict[str, IndexTransform] | None) -> IndexTransform:
    """Look up the IndexTransform for an extra buffer."""
    if extras and name in extras:
        return extras[name]
    return IdentityTransform()


def eval_expr_chain(
    transform: list[TileOp],
    source_expr: str,
    store_offs: str | None = None,
    extra_transforms: dict[str, IndexTransform] | None = None,
    chain_source_name: str | None = None,
    operand_exprs: dict[str, str] | None = None,
) -> str:
    """Evaluate a chain of tile IR ops into an MSL expression string.

    `operand_exprs` maps a non-source operand's IR name to a pre-resolved MSL
    expression. Use it for operands that are NOT device buffers — e.g. a
    loop-carried per-thread register feeding an epilogue store transform. Without
    it a store transform (`store_offs is not None`) indexes EVERY extra operand
    as `name[store_offs]`, which is correct for a fused residual buffer but emits
    an undeclared global-indexed array for a register value.
    """
    operand_exprs = operand_exprs or {}
    produced: set[str] = set()
    for t_op in transform:
        if t_op.result:
            produced.add(t_op.result.name)

    source_name = chain_source_name
    non_produced: set[str] = set()
    for t_op in transform:
        if isinstance(t_op, Constant):
            continue
        for val in t_op.operand_values():
            if val.name not in produced:
                non_produced.add(val.name)
                if source_name is None:
                    source_name = val.name

    expr_map: dict[str, str] = {}
    if source_name is not None:
        expr_map[source_name] = source_expr
    for name in non_produced:
        if name == source_name:
            continue
        if name in operand_exprs:
            expr_map[name] = operand_exprs[name]
        elif store_offs is not None:
            transform_for_extra = (extra_transforms or {}).get(name, IdentityTransform())
            idx_expr = transform_for_extra.flat(store_offs)
            expr_map[name] = f"float({name}[{idx_expr}])"
        else:
            expr_map[name] = name

    for t_op in transform:
        if isinstance(t_op, Constant):
            expr_map[t_op.result.name] = fmt_const(t_op.value)
            continue
        if isinstance(t_op, BinOp):
            l = expr_map.get(t_op.lhs.name, t_op.lhs.name) if t_op.lhs else "0"  # noqa: E741
            r = expr_map.get(t_op.rhs.name, t_op.rhs.name) if t_op.rhs else "0"
            sym = BINOP_MSL.get(t_op.op)
            func_name = BINOP_FUNC_MSL.get(t_op.op)
            if func_name:
                expr_map[t_op.result.name] = f"{func_name}({l}, {r})"
            elif sym:
                expr_map[t_op.result.name] = f"({l} {sym} {r})"
        elif isinstance(t_op, UnaryOp):
            inp = expr_map.get(t_op.input.name, t_op.input.name) if t_op.input else "0"
            func_name = UNARY_FUNC_MSL.get(t_op.op)
            sym = UNARY_MSL.get(t_op.op)
            if func_name:
                expr_map[t_op.result.name] = f"{func_name}({inp})"
            elif sym:
                expr_map[t_op.result.name] = f"({sym}{inp})"
        elif isinstance(t_op, TernaryOp):
            a = expr_map.get(t_op.a.name, t_op.a.name) if t_op.a else "0"
            b = expr_map.get(t_op.b.name, t_op.b.name) if t_op.b else "0"
            c = expr_map.get(t_op.c.name, t_op.c.name) if t_op.c else "0"
            expr_map[t_op.result.name] = f"{t_op.op}({a}, {b}, {c})"
        elif isinstance(t_op, Select):
            cond = expr_map.get(t_op.cond.name, t_op.cond.name) if t_op.cond else "0"
            tv = expr_map.get(t_op.true_val.name, t_op.true_val.name) if t_op.true_val else "0"
            fv = expr_map.get(t_op.false_val.name, t_op.false_val.name) if t_op.false_val else "0"
            expr_map[t_op.result.name] = f"(({cond}) ? ({tv}) : ({fv}))"
        elif isinstance(t_op, Compare):
            l = expr_map.get(t_op.lhs.name, t_op.lhs.name) if t_op.lhs else "0"  # noqa: E741
            r = expr_map.get(t_op.rhs.name, t_op.rhs.name) if t_op.rhs else "0"
            sym = CMPOP_MSL.get(t_op.op, "==")
            expr_map[t_op.result.name] = f"({l} {sym} {r})"
        elif isinstance(t_op, BoolOp):
            l = expr_map.get(t_op.lhs.name, t_op.lhs.name) if t_op.lhs else "0"  # noqa: E741
            r = expr_map.get(t_op.rhs.name, t_op.rhs.name) if t_op.rhs else "0"
            sym = "&&" if t_op.op == "and" else "||"
            expr_map[t_op.result.name] = f"({l} {sym} {r})"
        elif isinstance(t_op, Cast):
            inp = expr_map.get(t_op.input.name, t_op.input.name) if t_op.input else "0"
            msl_type = from_ir(t_op.target_dtype).msl
            expr_map[t_op.result.name] = f"_alloy_cast_{msl_type}({inp})"

    for t_op in reversed(transform):
        if not isinstance(t_op, Constant) and t_op.result and t_op.result.name in expr_map:
            return expr_map[t_op.result.name]
    return source_expr
