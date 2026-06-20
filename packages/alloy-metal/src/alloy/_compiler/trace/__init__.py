"""Trace-based kernel compilation — execute Python to build tile IR.

Executes the kernel with TracedValue proxies. Each DSL operation (load, store,
exp, etc.) appends a TileOp to a TileBuilder. The result is a TileFunction —
same type consumed by tile_opt/tile_plan/tile_msl.
"""

from __future__ import annotations

from typing import cast

from alloy._compiler.trace.ast_rewrite import _rewrite_kernel_source as _rewrite_kernel_source
from alloy._compiler.trace.addressing import (
    _apply_stride_decomposition,
    _build_write_pattern,
    _decompose_ptr,
    _extract_2d_addr,
    _record_pid_dim_from_address,
)
from alloy._compiler.trace.kernel import trace_kernel as trace_kernel
from alloy._compiler.trace.control_flow import (
    _trace_flow as _trace_flow,
    _trace_for_var as _trace_for_var,
    _trace_if_else as _trace_if_else,
    _trace_if_enter as _trace_if_enter,
    _trace_if_exit as _trace_if_exit,
    _trace_loop_cond as _trace_loop_cond,
    _trace_loop_enter as _trace_loop_enter,
    _trace_loop_exit as _trace_loop_exit,
    trace_if as trace_if,
)
from alloy._compiler.trace.value import (
    TracedValue,
    _active as _active,
    _add_op,
    _ctx,
    _ensure_traced,
)

from alloy._compiler.dtypes import from_name
from alloy._compiler.tile_ir import (
    Atomic,
    Barrier,
    BinOp,
    Cast,
    Constant,
    CoopLoad,
    Copy4,
    DebugPrint,
    IndexLoad,
    Layout,
    LocalAlloc,
    NumPrograms,
    SharedAlloc,
    SimdMatrixOp,
    SimdOp,
    ThreadId,
    TileValue,
    UnaryOp,
    Zeros,
)

# ---------------------------------------------------------------------------
# DSL operations — called during tracing to build IR
# ---------------------------------------------------------------------------


class NamedDType:
    __name__: str


def _dtype_name(dtype: object) -> str:
    return cast(NamedDType, dtype).__name__ if hasattr(dtype, "__name__") else str(dtype)


def trace_program_id(axis: int) -> TracedValue:
    ctx = _ctx()
    tv = ctx.builder.program_id(axis)
    spec = ctx.spec
    spec.pid_tvs[tv.name] = axis
    if axis not in spec.axes:
        spec.axes[axis] = {"block": 1, "bound": None}
    return TracedValue(tv, pid_axis=axis)


def trace_arange(start: int, end: int) -> TracedValue:
    ctx = _ctx()
    tv = ctx.builder.make_range(start, end)
    ctx.spec.arange_tvs[tv.name] = end - start
    return TracedValue(tv)


def trace_load(
    addr,
    *,
    mask=None,
    other=0.0,
    _dequant_scale=None,
    _dequant_bias=None,
    _dequant_zero_point=0,
    _dequant_n_groups=0,
    _dequant_group_size=0,
    _dequant_format="",
    _dequant_high=None,
) -> TracedValue:
    """Trace al.load(ptr + offsets, mask=..., other=...)."""
    addr = _ensure_traced(addr)
    ctx = _ctx()

    # shared/local alloc: index load
    if addr._ptr_base is not None and id(addr._ptr_base) in ctx.alloc_vars:
        base_tv = addr._ptr_base._tv
        idx_tv = addr._ptr_offsets._tv if addr._ptr_offsets is not None else addr._tv
        v = TileValue(
            name=ctx.builder._fresh("ild"),
            shape=(),
            layout=Layout.REPLICATED,
            dtype="f32",
        )
        _add_op(IndexLoad(result=v, base=base_tv, index=idx_tv))
        return TracedValue(v)

    _record_pid_dim_from_address(addr)
    ptr_tv, off_tv = _decompose_ptr(addr)
    # Strided buffer: decompose flat offset into multi-dimensional strided access.
    # The stride metadata was injected by _queue_op as constexprs.
    off_tv = _apply_stride_decomposition(ctx, ptr_tv, off_tv)
    mask_tv = _ensure_traced(mask)._tv if mask is not None else None
    dtype = ptr_tv.dtype if ptr_tv else "f32"
    row_indices, col_indices, row_stride, base_offset, addr_transposed = _extract_2d_addr(off_tv)
    # Packed addressing (col = original_col // pack_factor) for integer buffer
    # types only — check the ORIGINAL buffer dtype, not the promoted dtype.
    pack_factor = 0
    pack_bits = 0
    ptr_name = ptr_tv.name if ptr_tv else ""
    original_dtype = ctx.buffer_dtypes.get(ptr_name, dtype) if ctx.buffer_dtypes else dtype
    if col_indices is not None and original_dtype in ("u8", "i8", "char", "uchar"):
        col_op = ctx.op_map.get(col_indices.name)
        if isinstance(col_op, BinOp) and col_op.op == "floordiv":
            divisor_op = ctx.op_map.get(col_op.rhs.name) if col_op.rhs else None
            if isinstance(divisor_op, Constant) and int(divisor_op.value) in (1, 2, 4, 8):
                pack_factor = int(divisor_op.value)
                pack_bits = 8 // pack_factor
                col_indices = col_op.lhs  # unwrap: use original column
        elif _dequant_scale is not None:
            pack_factor = 1
            pack_bits = 8
    # Resolve dequant scale pointer if provided
    dq_scale_tv = _ensure_traced(_dequant_scale)._tv if _dequant_scale is not None else None
    dq_bias_tv = _ensure_traced(_dequant_bias)._tv if _dequant_bias is not None else None
    dq_high_tv = _ensure_traced(_dequant_high)._tv if _dequant_high is not None else None

    tv = ctx.builder.load(
        ptr_tv,
        off_tv,
        mask=mask_tv,
        other=other,
        dtype=dtype,
        row_indices=row_indices,
        col_indices=col_indices,
        row_stride=row_stride,
        base_offset=base_offset,
        addr_transposed=addr_transposed,
        pack_factor=pack_factor,
        pack_bits=pack_bits,
        dequant_scale_ptr=dq_scale_tv,
        dequant_bias_ptr=dq_bias_tv,
        dequant_zero_point=float(_dequant_zero_point),
        dequant_n_groups=_dequant_n_groups,
        dequant_format=_dequant_format,
        dequant_high_ptr=dq_high_tv,
    )
    return TracedValue(tv)


def trace_store(addr, value, *, mask=None, reduce=""):
    """Trace al.store(ptr + offsets, value, mask=..., reduce=...).

    reduce="add" makes overlapping stores atomic_fetch_add (scatter-accumulate),
    e.g. the MoE grouped-down combine adding each expert tile into Y[token]."""
    addr = _ensure_traced(addr)
    value = _ensure_traced(value)
    _record_pid_dim_from_address(addr)
    ptr_tv, off_tv = _decompose_ptr(addr)
    off_tv = _apply_stride_decomposition(_ctx(), ptr_tv, off_tv)
    mask_traced = _ensure_traced(mask) if mask is not None else None
    mask_tv = mask_traced._tv if mask_traced is not None else None
    row_indices, col_indices, row_stride, base_offset, _addr_t = _extract_2d_addr(off_tv)
    ctx = _ctx()
    ctx.builder.store(
        ptr_tv,
        off_tv,
        value._tv,
        mask=mask_tv,
        row_indices=row_indices,
        col_indices=col_indices,
        row_stride=row_stride,
        base_offset=base_offset,
        reduce=reduce,
    )

    # Record write pattern for every store to an output param (not first-write-wins)
    param_name = ptr_tv.name
    if param_name in ctx.spec.output_params:
        pattern = _build_write_pattern(
            param_name,
            addr,
            value,
            off_tv,
            row_indices,
            col_indices,
            row_stride,
            base_offset,
        )
        ctx.spec.output_writes.setdefault(param_name, []).append(pattern)


def trace_zeros(shape, *, dtype=None) -> TracedValue:
    if dtype is not None:
        if isinstance(dtype, str):
            ir_dtype = from_name(dtype).ir
        else:
            # al.float32 sentinel etc
            ir_dtype = from_name(_dtype_name(dtype)).ir
    else:
        ir_dtype = "f32"
    if isinstance(shape, int):
        shape = (shape,)
    v = TileValue(
        name=_ctx().builder._fresh("z"),
        shape=shape,
        layout=Layout.REPLICATED,
        dtype=ir_dtype,
    )
    _add_op(Zeros(result=v, shape=shape, dtype=ir_dtype))
    return TracedValue(v)


def trace_dot(lhs, rhs, *, transpose_rhs=False, transpose_lhs=False) -> TracedValue:
    lhs = _ensure_traced(lhs)
    rhs = _ensure_traced(rhs)
    tv = _ctx().builder.dot(
        lhs._tv, rhs._tv, transpose_rhs=transpose_rhs, transpose_lhs=transpose_lhs
    )
    return TracedValue(tv)


def trace_barrier():
    _add_op(Barrier())


def trace_thread_id() -> TracedValue:
    v = TileValue(
        name=_ctx().builder._fresh("tid"),
        shape=(),
        layout=Layout.REPLICATED,
        dtype="i32",
    )
    _add_op(ThreadId(result=v))
    return TracedValue(v)


def trace_num_programs(axis=0) -> TracedValue:
    v = TileValue(
        name=_ctx().builder._fresh("np"),
        shape=(),
        layout=Layout.REPLICATED,
        dtype="i32",
    )
    _add_op(NumPrograms(result=v, axis=axis))
    return TracedValue(v)


def trace_shared(size, dtype=None) -> TracedValue:
    if isinstance(size, TracedValue):
        raise ValueError("shared() size must be a compile-time constant")
    v = TileValue(
        name=_ctx().builder._fresh("shm"),
        shape=(),
        layout=Layout.REPLICATED,
        dtype="f32",
    )
    _add_op(SharedAlloc(result=v, size=int(size)))
    tv = TracedValue(v, is_ptr=True)
    _ctx().alloc_vars.add(id(tv))
    return tv


def trace_local(size, dtype=None) -> TracedValue:
    if isinstance(size, TracedValue):
        raise ValueError("local() size must be a compile-time constant")
    v = TileValue(
        name=_ctx().builder._fresh("loc"),
        shape=(),
        layout=Layout.REPLICATED,
        dtype="f32",
    )
    _add_op(LocalAlloc(result=v, size=int(size)))
    tv = TracedValue(v, is_ptr=True)
    _ctx().alloc_vars.add(id(tv))
    return tv


def trace_debug_print(fmt: str, *args):
    traced_args = [_ensure_traced(a)._tv for a in args]
    _add_op(DebugPrint(fmt=fmt, args=traced_args))


def trace_cast(x, dtype) -> TracedValue:
    x = _ensure_traced(x)
    if isinstance(dtype, str):
        target = from_name(dtype).ir
    else:
        target = from_name(_dtype_name(dtype)).ir
    v = TileValue(
        name=_ctx().builder._fresh("cast"),
        shape=x._tv.shape,
        layout=x._tv.layout,
        dtype=target,
    )
    _add_op(Cast(result=v, input=x._tv, target_dtype=target))
    return TracedValue(v)


def trace_bitcast(x, dtype) -> TracedValue:
    x = _ensure_traced(x)
    if isinstance(dtype, str):
        target = from_name(dtype).ir
    else:
        target = from_name(_dtype_name(dtype)).ir
    v = TileValue(
        name=_ctx().builder._fresh("bc"),
        shape=x._tv.shape,
        layout=x._tv.layout,
        dtype=target,
    )
    _add_op(UnaryOp(result=v, op="bitcast", input=x._tv))
    return TracedValue(v)


def trace_where(cond, x, y) -> TracedValue:
    cond = _ensure_traced(cond)
    x = _ensure_traced(x)
    y = _ensure_traced(y)
    tv = _ctx().builder.select(cond._tv, x._tv, y._tv)
    return TracedValue(tv)


def trace_fma(a, b, c) -> TracedValue:
    a, b, c = _ensure_traced(a), _ensure_traced(b), _ensure_traced(c)
    tv = _ctx().builder.ternary("fma", a._tv, b._tv, c._tv)
    return TracedValue(tv)


def trace_clamp(x, lo, hi) -> TracedValue:
    x, lo, hi = _ensure_traced(x), _ensure_traced(lo), _ensure_traced(hi)
    tv = _ctx().builder.ternary("clamp", x._tv, lo._tv, hi._tv)
    return TracedValue(tv)


def trace_maximum(x, y) -> TracedValue:
    x, y = _ensure_traced(x), _ensure_traced(y)
    tv = _ctx().builder.binop("max", x._tv, y._tv)
    return TracedValue(tv)


def trace_minimum(x, y) -> TracedValue:
    x, y = _ensure_traced(x), _ensure_traced(y)
    tv = _ctx().builder.binop("min", x._tv, y._tv)
    return TracedValue(tv)


# Unary math
def _make_trace_unary(op_name):
    def fn(x) -> TracedValue:
        x = _ensure_traced(x)
        tv = _ctx().builder.unary(op_name, x._tv)
        return TracedValue(tv)

    fn.__name__ = f"trace_{op_name}"
    return fn


trace_exp = _make_trace_unary("exp")
trace_log = _make_trace_unary("log")
trace_sqrt = _make_trace_unary("sqrt")
trace_rsqrt = _make_trace_unary("rsqrt")
trace_tanh = _make_trace_unary("tanh")
trace_erf = _make_trace_unary("erf")
trace_sin = _make_trace_unary("sin")
trace_cos = _make_trace_unary("cos")
trace_abs = _make_trace_unary("abs")
trace_ceil = _make_trace_unary("ceil")
trace_floor = _make_trace_unary("floor")
trace_round = _make_trace_unary("round")  # MSL round(): half away from zero, matches ggml roundf
trace_exp2 = _make_trace_unary("exp2")
trace_log2 = _make_trace_unary("log2")


# Compound ops
def trace_sigmoid(x) -> TracedValue:
    x = _ensure_traced(x)
    b = _ctx().builder
    neg_x = b.unary("neg", x._tv)
    exp_neg = b.unary("exp", neg_x)
    one = b.constant(1.0, dtype="f32")
    denom = b.binop("add", one, exp_neg)
    tv = b.binop("div", one, denom)
    return TracedValue(tv)


def trace_relu(x) -> TracedValue:
    x = _ensure_traced(x)
    b = _ctx().builder
    zero = b.constant(0.0, dtype="f32")
    tv = b.binop("max", x._tv, zero)
    return TracedValue(tv)


def trace_gelu(x) -> TracedValue:
    """Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))."""
    x = _ensure_traced(x)
    b = _ctx().builder
    half = b.constant(0.5, dtype="f32")
    inv_sqrt2 = b.constant(0.7071067811865476, dtype="f32")
    one = b.constant(1.0, dtype="f32")
    erf_val = b.unary("erf", b.binop("mul", x._tv, inv_sqrt2))
    tv = b.binop("mul", b.binop("mul", x._tv, half), b.binop("add", one, erf_val))
    return TracedValue(tv)


def trace_gelu_tanh(x) -> TracedValue:
    """Approximate GELU using tanh."""
    x = _ensure_traced(x)
    b = _ctx().builder
    c1 = b.constant(0.5, dtype="f32")
    c2 = b.constant(0.7978845608, dtype="f32")
    c3 = b.constant(0.044715, dtype="f32")
    one = b.constant(1.0, dtype="f32")
    x3 = b.binop("mul", x._tv, b.binop("mul", x._tv, x._tv))
    inner = b.binop("add", x._tv, b.binop("mul", c3, x3))
    tanh_arg = b.binop("mul", c2, inner)
    clamp_lo = b.constant(-10.0, dtype="f32")
    clamp_hi = b.constant(10.0, dtype="f32")
    tanh_arg = b.ternary("clamp", tanh_arg, clamp_lo, clamp_hi)
    tanh_val = b.unary("tanh", tanh_arg)
    sum_val = b.binop("add", one, tanh_val)
    half_x = b.binop("mul", x._tv, c1)
    tv = b.binop("mul", half_x, sum_val)
    return TracedValue(tv)


# Tile reductions
def trace_sum(x, axis=0) -> TracedValue:
    x = _ensure_traced(x)
    tv = _ctx().builder.reduce(x._tv, axis, "sum")
    return TracedValue(tv)


def trace_load4_vec(ptr) -> TracedValue:
    """Vectorized load: read 4 consecutive f16 at ptr. ptr must include offsets."""
    ptr = _ensure_traced(ptr)
    base_tv, off_tv = _decompose_ptr(ptr)
    tv = _ctx().builder.load4_vec(base_tv, off_tv)
    return TracedValue(tv)


def trace_load_wide(ptr, dtype) -> TracedValue:
    """Read one scalar of a wider type at a BYTE offset from a byte buffer
    (reinterpret load). `dtype` is an IR string ("u16", "f16", "u32", ...) or a
    DType (whose str() is its ir). Matches llama.cpp's `(uint16_t*)scales`."""
    ptr = _ensure_traced(ptr)
    base_tv, off_tv = _decompose_ptr(ptr)
    tv = _ctx().builder.load_wide(base_tv, off_tv, str(dtype))
    return TracedValue(tv)


def trace_dot4(a, b) -> TracedValue:
    """Vectorized dot product: dot(float4(half4_a), float4(half4_b)) → f32 scalar."""
    a = _ensure_traced(a)
    b = _ensure_traced(b)
    tv = _ctx().builder.dot4(a._tv, b._tv)
    return TracedValue(tv)


def trace_unpack4(a, lane) -> TracedValue:
    """Extract one component (lane in 0..3) of a vec4 (load4_vec result)
    as a scalar f32. Lets a kernel issue one vec4 load and use the four
    components as register-resident scalars in downstream FMAs."""
    a = _ensure_traced(a)
    tv = _ctx().builder.unpack4(a._tv, int(lane))
    return TracedValue(tv)


def trace_as_char4(a, lane) -> TracedValue:
    """Reinterpret uint component `lane` of a uint4 (load4_vec on a u32-viewed
    buffer) as char4, promoted to float4 — for dot4 against quantized codes
    fetched 16-at-a-time with one load."""
    a = _ensure_traced(a)
    tv = _ctx().builder.as_char4(a._tv, int(lane))
    return TracedValue(tv)


def trace_interleave_vec4(lo, hi, half) -> TracedValue:
    """Interleave two vec4s. half=0 returns (lo.x, hi.x, lo.y, hi.y);
    half=1 returns (lo.z, hi.z, lo.w, hi.w). For unpacking packed nibble pairs
    into K-aligned vectors that can dot4 with consecutive activations.
    """
    lo = _ensure_traced(lo)
    hi = _ensure_traced(hi)
    tv = _ctx().builder.interleave_vec4(lo._tv, hi._tv, int(half))
    return TracedValue(tv)


def trace_simd_reduce(x, op="sum") -> TracedValue:
    """SIMD cross-lane reduction (emits simd_sum/simd_max)."""
    x = _ensure_traced(x)
    tv = _ctx().builder.simd_reduce(x._tv, op)
    return TracedValue(tv)


def trace_max(x, axis=0) -> TracedValue:
    x = _ensure_traced(x)
    tv = _ctx().builder.reduce(x._tv, axis, "max")
    return TracedValue(tv)


def trace_min(x, axis=0) -> TracedValue:
    x = _ensure_traced(x)
    tv = _ctx().builder.reduce(x._tv, axis, "min")
    return TracedValue(tv)


# Atomics
def _make_trace_atomic(op_name):
    def fn(ptr, index, val_or_expected, desired_or_none=None) -> TracedValue:
        ptr = _ensure_traced(ptr)
        index = _ensure_traced(index)
        op = op_name[len("atomic_") :]
        if op == "cas":
            # atomic_cas(ptr, index, expected, desired)
            expected = _ensure_traced(val_or_expected)
            desired = _ensure_traced(desired_or_none)
            v = TileValue(
                name=_ctx().builder._fresh("atm"),
                shape=(),
                layout=Layout.REPLICATED,
                dtype="i32",
            )
            _add_op(
                Atomic(
                    result=v,
                    op=op,
                    ptr=ptr._tv,
                    index=index._tv,
                    value=desired._tv,
                    expected=expected._tv,
                )
            )
        else:
            # atomic_*(ptr, index, val)
            val = _ensure_traced(val_or_expected)
            dtype = "f32" if op.endswith("_float") else "i32"
            v = TileValue(
                name=_ctx().builder._fresh("atm"),
                shape=(),
                layout=Layout.REPLICATED,
                dtype=dtype,
            )
            _add_op(
                Atomic(
                    result=v,
                    op=op,
                    ptr=ptr._tv,
                    index=index._tv,
                    value=val._tv,
                )
            )
        return TracedValue(v)

    fn.__name__ = f"trace_{op_name}"
    return fn


trace_atomic_add = _make_trace_atomic("atomic_add")
trace_atomic_max = _make_trace_atomic("atomic_max")
trace_atomic_min = _make_trace_atomic("atomic_min")
trace_atomic_cas = _make_trace_atomic("atomic_cas")
trace_atomic_xchg = _make_trace_atomic("atomic_xchg")
trace_atomic_and = _make_trace_atomic("atomic_and")
trace_atomic_or = _make_trace_atomic("atomic_or")
trace_atomic_xor = _make_trace_atomic("atomic_xor")
trace_atomic_add_float = _make_trace_atomic("atomic_add_float")
trace_atomic_max_float = _make_trace_atomic("atomic_max_float")
trace_atomic_min_float = _make_trace_atomic("atomic_min_float")


# SIMD ops
def _make_trace_simd(op_name):
    def fn(*args) -> TracedValue:
        traced_args = [_ensure_traced(a)._tv for a in args]
        op = op_name[len("simd_") :]
        v = TileValue(
            name=_ctx().builder._fresh("simd"),
            shape=(),
            layout=Layout.REPLICATED,
            dtype="f32",
        )
        _add_op(SimdOp(result=v, op=op, args=traced_args))
        return TracedValue(v)

    fn.__name__ = f"trace_{op_name}"
    return fn


trace_simd_shuffle_xor = _make_trace_simd("simd_shuffle_xor")
trace_simd_shuffle = _make_trace_simd("simd_shuffle")
trace_simd_shuffle_up = _make_trace_simd("simd_shuffle_up")
trace_simd_shuffle_down = _make_trace_simd("simd_shuffle_down")
trace_simd_prefix_exclusive_sum = _make_trace_simd("simd_prefix_exclusive_sum")
trace_simd_prefix_inclusive_sum = _make_trace_simd("simd_prefix_inclusive_sum")
trace_simd_all = _make_trace_simd("simd_all")
trace_simd_any = _make_trace_simd("simd_any")
trace_simd_id = _make_trace_simd("simd_id")
trace_simd_lane_id = _make_trace_simd("simd_lane_id")


# SIMD matrix ops
def trace_simd_matrix() -> TracedValue:
    v = TileValue(
        name=_ctx().builder._fresh("smat"),
        shape=(),
        layout=Layout.REPLICATED,
        dtype="f32",
    )
    _add_op(SimdMatrixOp(result=v, op="create"))
    return TracedValue(v)


def trace_simd_load(source, offset, stride, transpose=False) -> TracedValue:
    args = [_ensure_traced(a)._tv for a in (source, offset, stride)]
    stride_int = int(stride) if isinstance(stride, (int, float)) else None
    v = TileValue(
        name=_ctx().builder._fresh("smat"),
        shape=(),
        layout=Layout.REPLICATED,
        dtype="f32",
    )
    _add_op(
        SimdMatrixOp(
            result=v,
            op="load",
            args=args,
            stride=stride_int,
            transpose=transpose,
        )
    )
    return TracedValue(v)


def trace_simd_mma(acc, a, b) -> TracedValue:
    args = [_ensure_traced(x)._tv for x in (acc, a, b)]
    v = TileValue(
        name=_ctx().builder._fresh("smat"),
        shape=(),
        layout=Layout.REPLICATED,
        dtype="f32",
    )
    _add_op(SimdMatrixOp(result=v, op="mma", args=args))
    return TracedValue(v)


def trace_simd_store(mat, dest, offset, stride):
    args = [_ensure_traced(x)._tv for x in (mat, dest, offset, stride)]
    stride_int = int(stride) if isinstance(stride, (int, float)) else None
    _add_op(SimdMatrixOp(op="store", args=args, stride=stride_int))


def trace_coop_load(shared_buf, src_ptr, size):
    dst = _ensure_traced(shared_buf)
    src = _ensure_traced(src_ptr)
    count = _ensure_traced(size)
    # src_ptr is a pointer expression — need the full ptr+offsets value for CoopLoad.
    # If it has pointer tracking, reconstruct the add for the emitter.
    if src._ptr_base is not None:
        base = src._ptr_base
        while base._ptr_base is not None:
            base = base._ptr_base
        src_tv = _ctx().builder.binop("add", base._tv, src._tv)
    else:
        src_tv = src._tv
    _add_op(CoopLoad(dst=dst._tv, src=src_tv, count=count._tv))


def trace_copy4(dst, dst_offset, src_ptr):
    dst = _ensure_traced(dst)
    dst_off = _ensure_traced(dst_offset)
    src = _ensure_traced(src_ptr)
    # Decompose src pointer if possible
    src_ptr_tv, src_off_tv = _decompose_ptr(src)
    _add_op(
        Copy4(
            dst=dst._tv,
            dst_offset=dst_off._tv,
            src_ptr=src_ptr_tv,
            src_offset=src_off_tv,
        )
    )
