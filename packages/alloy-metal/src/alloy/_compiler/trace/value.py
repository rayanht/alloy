"""Trace-time value proxies and active tracing context."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Literal, TypeAlias

from alloy._compiler.dispatch_spec import BindingSource, FromInputShape, OutputWritePattern
from alloy._compiler.tile_ir import IndexLoad, IndexStore, Layout, TileBuilder, TileOp, TileValue

PidAxis: TypeAlias = int | Literal["multi"]
MaskBounds: TypeAlias = dict[str, int | PidAxis]
ConstexprValue: TypeAlias = (
    bool | int | float | str | tuple[int, ...] | tuple[float, ...] | tuple[str, ...] | None
)

# ---------------------------------------------------------------------------
# Trace context — thread-local state active during tracing
# ---------------------------------------------------------------------------


@dataclass
class SpecBuilder:
    """Builds DispatchContract during tracing by observing key operations."""

    # pid TileValue name → axis
    pid_tvs: dict[str, int]
    # arange TileValue name → range size
    arange_tvs: dict[str, int]
    # axis → {"block": int, "bound": int|None}
    axes: dict[int, dict[str, int | None]]
    # composite bounds from multi-axis comparisons: [(total_bound, {axis: stride})]
    composite_bounds: list[tuple[int, dict[int, int]]]
    # param_name → list[OutputWritePattern] — ALL stores, merged at assembly
    output_writes: dict[str, list[OutputWritePattern]]
    # pid axis → (buf_name, dim_idx, dim_size) — which buffer dim each pid indexes
    pid_dim_map: dict[int, tuple[str, int, int]]
    # symbol_name → BindingSource (FromConstexpr | FromInputShape | FromDerived)
    bindings: dict[str, BindingSource]
    # output param names
    output_params: set[str]


@dataclass
class TraceCtx:
    builder: TileBuilder
    # op_map: TileValue name -> producing TileOp (for 2D address extraction)
    op_map: dict[str, TileOp]
    # buffer param names in order (for ptr decomposition)
    buffer_params: dict[str, TracedValue]
    # constexpr values
    constexpr_values: dict[str, ConstexprValue]
    # shared/local alloc TracedValues (for index load/store detection)
    alloc_vars: set[int]  # id(TracedValue)
    # shape_vars: param_name -> shape tuple, recorded when kernel body accesses .shape
    shape_vars: dict[str, tuple[int, ...]]
    # spec builder for DispatchContract
    spec: SpecBuilder
    # original buffer dtypes (before type promotion) for packed load detection
    buffer_dtypes: dict[str, str] | None = None


_tls: threading.local = threading.local()


def _ctx() -> TraceCtx:
    try:
        ctx = _tls.trace_ctx
    except AttributeError:
        ctx = None
    if ctx is None:
        raise RuntimeError("Not inside a kernel trace")
    return ctx


def _active() -> bool:
    """Return True if we're currently inside a kernel trace."""
    try:
        return _tls.trace_ctx is not None
    except AttributeError:
        return False


def _add_op(op: TileOp) -> TileValue | None:
    ctx = _ctx()
    if op.result:
        ctx.op_map[op.result.name] = op
    return ctx.builder.func.add_op(op)


# ---------------------------------------------------------------------------
# TracedValue — proxy object that records operations via __add__ etc.
# ---------------------------------------------------------------------------


class TracedValue:
    """Symbolic value during kernel tracing.

    Wraps a TileValue and records operations into the active TileBuilder
    when Python operators are used.
    """

    __slots__ = (
        "_tv",
        "_is_ptr",
        "_ptr_base",
        "_ptr_offsets",
        "_array_shape",
        "_array_dtype",
        "_pid_axis",
        "_mask_bounds",
    )

    def __init__(
        self,
        tv: TileValue,
        *,
        is_ptr: bool = False,
        ptr_base: "TracedValue | None" = None,
        ptr_offsets: "TracedValue | None" = None,
        array_shape: tuple[int, ...] | None = None,
        array_dtype: str | None = None,
        pid_axis: PidAxis | None = None,
        mask_bounds: MaskBounds | None = None,
    ):
        self._tv = tv
        self._is_ptr = is_ptr
        self._ptr_base = ptr_base
        self._ptr_offsets = ptr_offsets
        self._array_shape = array_shape
        self._array_dtype = array_dtype
        self._pid_axis = pid_axis
        self._mask_bounds = mask_bounds

    @property
    def shape(self) -> tuple[int, ...]:
        """Array shape for user code (x.shape), or tile shape for computed values."""
        if self._array_shape is not None:
            try:
                c = _ctx()
                if c is not None and self._is_ptr and self._tv.name:
                    c.shape_vars[self._tv.name] = self._array_shape
                    pname = self._tv.name
                    for dim_idx in range(len(self._array_shape)):
                        c.spec.bindings[f"{pname}_dim{dim_idx}"] = FromInputShape(pname, dim_idx)
            except Exception:
                pass
            return self._array_shape
        return self._tv.shape

    @property
    def dtype(self) -> str:
        if self._array_dtype is not None:
            return self._array_dtype
        return self._tv.dtype

    @property
    def ndim(self) -> int:
        return len(self.shape)

    # --- Arithmetic operators ---

    def __add__(self, other):
        other = _ensure_traced(other)
        # Pointer arithmetic: don't emit add(ptr, offsets) — just track lineage.
        # The result's _tv IS the offsets. The ptr base is carried separately.
        if self._is_ptr:
            # ptr + offsets → result is the offsets, base is the ptr
            return TracedValue(other._tv, ptr_base=self, ptr_offsets=other)
        if other._is_ptr:
            # offsets + ptr → result is the offsets, base is the ptr
            return TracedValue(self._tv, ptr_base=other, ptr_offsets=self)
        tv = _ctx().builder.binop("add", self._tv, other._tv)
        if self._ptr_base is not None:
            # Chained ptr arithmetic: combine pid_axis from existing offsets + new term
            prev_ax = self._ptr_offsets._pid_axis if self._ptr_offsets is not None else None
            new_ax = other._pid_axis
            if prev_ax is not None and new_ax is not None and prev_ax != new_ax:
                combined = "multi"
            else:
                combined = prev_ax if prev_ax is not None else new_ax
            return TracedValue(
                tv,
                ptr_base=self._ptr_base,
                ptr_offsets=TracedValue(tv, pid_axis=combined),
            )
        if other._ptr_base is not None:
            prev_ax = other._ptr_offsets._pid_axis if other._ptr_offsets is not None else None
            new_ax = self._pid_axis
            if prev_ax is not None and new_ax is not None and prev_ax != new_ax:
                combined = "multi"
            else:
                combined = prev_ax if prev_ax is not None else new_ax
            return TracedValue(
                tv,
                ptr_base=other._ptr_base,
                ptr_offsets=TracedValue(tv, pid_axis=combined),
            )
        # Propagate pid_axis and detect blocked offset (pid*BLOCK + arange)
        s_ax = self._pid_axis
        o_ax = other._pid_axis
        if s_ax is not None and o_ax is not None and s_ax != o_ax:
            axis = "multi"  # composite: depends on multiple pid axes
        else:
            axis = s_ax if s_ax is not None else o_ax
        result = TracedValue(tv, pid_axis=axis)
        spec = _ctx().spec
        # Propagate arange membership through addition
        s_arange = spec.arange_tvs.get(self._tv.name)
        o_arange = spec.arange_tvs.get(other._tv.name)
        block = s_arange or o_arange
        if block:
            spec.arange_tvs[tv.name] = block  # result is also arange-derived
        # Detect pid_expr + arange → blocked offset
        if block and axis is not None and isinstance(axis, int) and axis in spec.axes:
            spec.axes[axis]["block"] = max(spec.axes[axis]["block"], block)
        return result

    def __radd__(self, other):
        return _ensure_traced(other).__add__(self)

    def __sub__(self, other):
        other = _ensure_traced(other)
        tv = _ctx().builder.binop("sub", self._tv, other._tv)
        return TracedValue(tv)

    def __rsub__(self, other):
        return _ensure_traced(other).__sub__(self)

    def __mul__(self, other):
        raw_other = other
        other = _ensure_traced(other)
        tv = _ctx().builder.binop("mul", self._tv, other._tv)
        axis = self._pid_axis if self._pid_axis is not None else other._pid_axis
        # Record pid stride: pid(axis) * constant → the constant is the stride.
        # First write wins, capturing the block-level stride (e.g. BLOCK_M) for
        # grid computation. Address-level strides (e.g. M*K for batched GEMM) are
        # recovered from the IR at load/store sites by _find_store_pid_stride.
        if isinstance(axis, int) and axis != "multi":
            spec = _ctx().spec
            if isinstance(raw_other, (int, float)) and self._pid_axis is not None:
                spec.axes.setdefault(axis, {"block": 1, "bound": None})
                spec.axes[axis].setdefault("stride", None)
                if spec.axes[axis]["stride"] is None:
                    spec.axes[axis]["stride"] = int(raw_other)
            elif isinstance(raw_other, TracedValue) and raw_other._pid_axis is None:
                # other might be a concrete-valued TracedValue (from constexpr)
                pass
        return TracedValue(tv, pid_axis=axis)

    def __rmul__(self, other):
        return _ensure_traced(other).__mul__(self)

    def __truediv__(self, other):
        other = _ensure_traced(other)
        tv = _ctx().builder.binop("div", self._tv, other._tv)
        return TracedValue(tv)

    def __rtruediv__(self, other):
        return _ensure_traced(other).__truediv__(self)

    def __floordiv__(self, other):
        other = _ensure_traced(other)
        tv = _ctx().builder.binop("floordiv", self._tv, other._tv)
        return TracedValue(tv)

    def __rfloordiv__(self, other):
        return _ensure_traced(other).__floordiv__(self)

    def __mod__(self, other):
        other = _ensure_traced(other)
        tv = _ctx().builder.binop("mod", self._tv, other._tv)
        return TracedValue(tv)

    def __rmod__(self, other):
        return _ensure_traced(other).__mod__(self)

    def __neg__(self):
        tv = _ctx().builder.unary("neg", self._tv)
        return TracedValue(tv)

    def __invert__(self):
        tv = _ctx().builder.unary("bitnot", self._tv)
        return TracedValue(tv)

    def __and__(self, other):
        other = _ensure_traced(other)
        tv = _ctx().builder.binop("bitand", self._tv, other._tv)
        # Merge mask bounds from both sides
        merged = {}
        if self._mask_bounds:
            merged.update(self._mask_bounds)
        if hasattr(other, "_mask_bounds") and other._mask_bounds:
            merged.update(other._mask_bounds)
        return TracedValue(tv, mask_bounds=merged if merged else None)

    def __rand__(self, other):
        return _ensure_traced(other).__and__(self)

    def __or__(self, other):
        other = _ensure_traced(other)
        tv = _ctx().builder.binop("bitor", self._tv, other._tv)
        return TracedValue(tv)

    def __ror__(self, other):
        return _ensure_traced(other).__or__(self)

    def __xor__(self, other):
        other = _ensure_traced(other)
        tv = _ctx().builder.binop("bitxor", self._tv, other._tv)
        return TracedValue(tv)

    def __rxor__(self, other):
        return _ensure_traced(other).__xor__(self)

    def __lshift__(self, other):
        other = _ensure_traced(other)
        tv = _ctx().builder.binop("lshift", self._tv, other._tv)
        return TracedValue(tv)

    def __rlshift__(self, other):
        return _ensure_traced(other).__lshift__(self)

    def __rshift__(self, other):
        other = _ensure_traced(other)
        tv = _ctx().builder.binop("rshift", self._tv, other._tv)
        return TracedValue(tv)

    def __rrshift__(self, other):
        return _ensure_traced(other).__rshift__(self)

    # --- Comparison operators ---

    def __lt__(self, other):
        raw_bound = other
        other = _ensure_traced(other)
        tv = _ctx().builder.compare("lt", self._tv, other._tv)
        # Track mask bound: if RHS is a concrete int, record for spec
        bounds = None
        axis = self._pid_axis
        if isinstance(raw_bound, (int, float)) and axis is not None:
            spec = _ctx().spec
            bound = int(raw_bound)
            if axis == "multi":
                # Composite: record total bound + per-axis strides for constraint solving
                strides = {}
                for ax_id, ax_info in spec.axes.items():
                    s = ax_info.get("stride")
                    if s is not None:
                        strides[ax_id] = s
                spec.composite_bounds.append((bound, strides))
            elif isinstance(axis, int) and axis in spec.axes:
                cur = spec.axes[axis]["bound"]
                if cur is None or bound < cur:
                    spec.axes[axis]["bound"] = bound
            bounds = {"axis": axis, "bound": bound}
        elif isinstance(raw_bound, (int, float)):
            bounds = {"bound": int(raw_bound)}
        return TracedValue(tv, mask_bounds=bounds, pid_axis=axis)

    def __le__(self, other):
        other = _ensure_traced(other)
        tv = _ctx().builder.compare("le", self._tv, other._tv)
        return TracedValue(tv)

    def __gt__(self, other):
        other = _ensure_traced(other)
        tv = _ctx().builder.compare("gt", self._tv, other._tv)
        return TracedValue(tv)

    def __ge__(self, other):
        other = _ensure_traced(other)
        tv = _ctx().builder.compare("ge", self._tv, other._tv)
        return TracedValue(tv)

    def __eq__(self, other):
        other = _ensure_traced(other)
        tv = _ctx().builder.compare("eq", self._tv, other._tv)
        return TracedValue(tv)

    def __ne__(self, other):
        other = _ensure_traced(other)
        tv = _ctx().builder.compare("ne", self._tv, other._tv)
        return TracedValue(tv)

    # --- Augmented assignment support ---

    def __iadd__(self, other):
        return self.__add__(other)

    def __isub__(self, other):
        return self.__sub__(other)

    def __imul__(self, other):
        return self.__mul__(other)

    def __itruediv__(self, other):
        return self.__truediv__(other)

    # --- Subscript: x[:, None], x[None, :], arr[idx] ---

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            # x[:, None] -> expand_dims(x, axis=1)
            if isinstance(key[0], slice) and key[0] == slice(None) and key[1] is None:
                tv = _ctx().builder.expand_dims(self._tv, axis=1)
                return TracedValue(tv, pid_axis=self._pid_axis)
            # x[None, :] -> expand_dims(x, axis=0)
            if key[0] is None and isinstance(key[1], slice) and key[1] == slice(None):
                tv = _ctx().builder.expand_dims(self._tv, axis=0)
                return TracedValue(tv, pid_axis=self._pid_axis)
        # arr[idx] -> IndexLoad
        idx = _ensure_traced(key)
        v = TileValue(
            name=_ctx().builder._fresh("ild"),
            shape=(),
            layout=Layout.REPLICATED,
            dtype=self._tv.dtype,
        )
        _add_op(IndexLoad(result=v, base=self._tv, index=idx._tv))
        return TracedValue(v)

    def __setitem__(self, key, value):
        idx = _ensure_traced(key)
        val = _ensure_traced(value)
        _add_op(IndexStore(base=self._tv, index=idx._tv, value=val._tv))

    # --- Prevent bool() coercion (would break `if traced_val:`) ---

    def __bool__(self):
        raise RuntimeError(
            "Cannot use TracedValue in a boolean context (if/while). "
            "Use al.where() for conditional select."
        )

    # --- repr ---

    def __repr__(self):
        return f"TracedValue({self._tv.name}: {self._tv.dtype}{list(self._tv.shape)})"


def _ensure_traced(val) -> TracedValue:
    """Wrap Python scalars as IR constants."""
    if isinstance(val, TracedValue):
        return val
    if isinstance(val, (int, float)):
        dtype = "f32" if isinstance(val, float) else "i32"
        tv = _ctx().builder.constant(val, dtype=dtype)
        return TracedValue(tv)
    raise TypeError(f"Cannot trace value of type {type(val)}")
