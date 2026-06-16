"""Elementwise handlers for torch op lowering."""

import math
from typing import TYPE_CHECKING

import alloy
import torch

from alloy._compiler.dtypes import float32, from_torch_dtype
from alloy._dispatch.buf_utils import _alloc_aligned, _alloc_scratch
from alloy._runtime.alloy_buffer import AlloyBuffer
from alloy._runtime.buffer_ops import (
    _compare_nd,
    _expand_buf,
    k_bitwise_and_nd,
    k_bitwise_or_nd,
    k_eq_nd,
    k_ne_nd,
)
from alloy_torch.ops.casting import _to_copy
from alloy_torch.ops.common import (
    _broadcast_shapes,
    _dtype_of,
    _is_bool_dtype,
    _numel,
    _reference_kernel_dtype,
    _shape_of,
)
from alloy_torch.ops.creation import _full
from alloy_torch.ops.values import _broadcast_layout, _coerce_lazy_value, _coerce_mask_numeric

if TYPE_CHECKING:
    output = AlloyBuffer
    constexpr = int
else:
    output = alloy.output
    constexpr = alloy.constexpr

ScalarValue = AlloyBuffer | bool | int | float | torch.Tensor
OptionalScalar = ScalarValue | None


@alloy.kernel
def k_where_nd(
    cond,
    x,
    y,
    out: output,
    N: constexpr,
    OUT0: constexpr = 1,
    OUT1: constexpr = 1,
    OUT2: constexpr = 1,
    OUT3: constexpr = 1,
    C_STR0: constexpr = 0,
    C_STR1: constexpr = 0,
    C_STR2: constexpr = 0,
    C_STR3: constexpr = 0,
    X_STR0: constexpr = 0,
    X_STR1: constexpr = 0,
    X_STR2: constexpr = 0,
    X_STR3: constexpr = 0,
    Y_STR0: constexpr = 0,
    Y_STR1: constexpr = 0,
    Y_STR2: constexpr = 0,
    Y_STR3: constexpr = 0,
    BLOCK_SIZE: constexpr = 1024,
):
    pid = alloy.program_id(0)
    offs = pid * BLOCK_SIZE + alloy.arange(0, BLOCK_SIZE)
    mask = offs < N
    rem = offs
    i3 = rem % OUT3
    rem = rem // OUT3
    i2 = rem % OUT2
    rem = rem // OUT2
    i1 = rem % OUT1
    i0 = rem // OUT1
    c_offs = i0 * C_STR0 + i1 * C_STR1 + i2 * C_STR2 + i3 * C_STR3
    x_offs = i0 * X_STR0 + i1 * X_STR1 + i2 * X_STR2 + i3 * X_STR3
    y_offs = i0 * Y_STR0 + i1 * Y_STR1 + i2 * Y_STR2 + i3 * Y_STR3
    c = alloy.load(cond + c_offs, mask=mask)
    xv = alloy.load(x + x_offs, mask=mask)
    yv = alloy.load(y + y_offs, mask=mask)
    alloy.store(out + offs, alloy.where(c != 0, xv, yv), mask=mask)


def _buf_add(a: AlloyBuffer, b: AlloyBuffer, *, alpha: float = 1) -> AlloyBuffer:
    if alpha != 1:
        return a + b * alpha
    return a + b


def _buf_add_scalar(a: AlloyBuffer, b: float | int, *, alpha: float = 1) -> AlloyBuffer:
    """aten.add.Scalar for tensor plus Python scalar."""
    return a + (b * alpha)


def _buf_sub(a: AlloyBuffer, b: AlloyBuffer, *, alpha: float = 1) -> AlloyBuffer:
    if alpha != 1:
        return a - b * alpha
    return a - b


def _bitwise_and_tensor(a: AlloyBuffer, b: AlloyBuffer) -> AlloyBuffer:
    bool_like = _is_bool_dtype(_dtype_of(a)) or _is_bool_dtype(_dtype_of(b))
    if bool_like:
        return _coerce_mask_numeric(a) * _coerce_mask_numeric(b)

    out_dtype = _reference_kernel_dtype(a, b)
    if out_dtype is not None:
        out_shape = tuple(_broadcast_shapes(_shape_of(a), _shape_of(b)))
        lhs = _coerce_lazy_value(a, dtype=out_dtype)
        rhs = _coerce_lazy_value(b, dtype=out_dtype)
        padded_out, lhs_strides = _broadcast_layout(lhs, out_shape)
        _, rhs_strides = _broadcast_layout(rhs, out_shape)
        out_arr = _alloc_scratch(out_shape, from_torch_dtype(out_dtype))
        out = k_bitwise_and_nd(
            lhs,
            rhs,
            out_arr,
            N=_numel(out_shape),
            OUT0=padded_out[0],
            OUT1=padded_out[1],
            OUT2=padded_out[2],
            OUT3=padded_out[3],
            X_STR0=lhs_strides[0],
            X_STR1=lhs_strides[1],
            X_STR2=lhs_strides[2],
            X_STR3=lhs_strides[3],
            Y_STR0=rhs_strides[0],
            Y_STR1=rhs_strides[1],
            Y_STR2=rhs_strides[2],
            Y_STR3=rhs_strides[3],
        )
        return out.reshape(out_shape)
    raise RuntimeError(
        f"Alloy bitwise_and: unsupported shape combination "
        f"(a.shape={a.shape}, b.shape={b.shape}, ndim={a.ndim})"
    )


def _bitwise_or_tensor(a: AlloyBuffer, b: AlloyBuffer) -> AlloyBuffer:
    bool_like = _is_bool_dtype(_dtype_of(a)) or _is_bool_dtype(_dtype_of(b))
    if bool_like:
        # OR of 0/1 masks: a + b - a*b (stays in {0,1}), mirroring the AND path's
        # `a * b`. Used by gemma4 audio's blocked attention-mask construction.
        ma, mb = _coerce_mask_numeric(a), _coerce_mask_numeric(b)
        return ma + mb - ma * mb

    out_dtype = _reference_kernel_dtype(a, b)
    if out_dtype is not None:
        out_shape = tuple(_broadcast_shapes(_shape_of(a), _shape_of(b)))
        lhs = _coerce_lazy_value(a, dtype=out_dtype)
        rhs = _coerce_lazy_value(b, dtype=out_dtype)
        padded_out, lhs_strides = _broadcast_layout(lhs, out_shape)
        _, rhs_strides = _broadcast_layout(rhs, out_shape)
        out_arr = _alloc_scratch(out_shape, from_torch_dtype(out_dtype))
        out = k_bitwise_or_nd(
            lhs,
            rhs,
            out_arr,
            N=_numel(out_shape),
            OUT0=padded_out[0],
            OUT1=padded_out[1],
            OUT2=padded_out[2],
            OUT3=padded_out[3],
            X_STR0=lhs_strides[0],
            X_STR1=lhs_strides[1],
            X_STR2=lhs_strides[2],
            X_STR3=lhs_strides[3],
            Y_STR0=rhs_strides[0],
            Y_STR1=rhs_strides[1],
            Y_STR2=rhs_strides[2],
            Y_STR3=rhs_strides[3],
        )
        return out.reshape(out_shape)
    raise RuntimeError(
        f"Alloy bitwise_or: unsupported shape combination "
        f"(a.shape={a.shape}, b.shape={b.shape}, ndim={a.ndim})"
    )


def _clamp(x: AlloyBuffer, min_val: OptionalScalar = None, max_val: OptionalScalar = None) -> AlloyBuffer:
    return x.clamp(min_val, max_val)


def _where_self(condition: AlloyBuffer, x: AlloyBuffer, y: AlloyBuffer) -> AlloyBuffer:
    cond_buf = _coerce_lazy_value(condition)
    x_buf = _coerce_lazy_value(x)
    y_buf = _coerce_lazy_value(y)
    out_shape = tuple(_broadcast_shapes(condition.shape, x.shape, y.shape))
    if out_shape == ():
        cond_val = float(cond_buf) if cond_buf._dtype.is_float() else int(cond_buf)
        picked = x_buf if cond_val != 0 else y_buf
        picked_val = float(picked)
        out_dtype = x_buf._dtype if x_buf._dtype.is_float() else float32
        out_buf = _alloc_aligned((), out_dtype)
        out_buf.write_scalar(picked_val)
        return out_buf
    if cond_buf.ndim == 0:
        cond_val = float(cond_buf) if cond_buf._dtype.is_float() else int(cond_buf)
        picked = x_buf if cond_val != 0 else y_buf
        if picked.shape == out_shape:
            return _to_copy(picked, dtype=picked._dtype.to_torch_dtype())
        expanded = _expand_buf(picked, out_shape)
        return _to_copy(expanded, dtype=expanded._dtype.to_torch_dtype())
    if x_buf._dtype.is_float() and y_buf._dtype.is_float():
        if not cond_buf._dtype.is_float():
            cond_buf = _to_copy(cond_buf, dtype=torch.float32)
        # k_where_nd reads each operand with the strides `_broadcast_layout`
        # derives from `buf._strides`. The kernel dispatch contiguizes input
        # buffers, so a non-contiguous view (e.g. a `transpose`/`permute` of
        # the operand) would be read with its pre-contiguize permuted strides
        # → wrong elements. Force contiguity so the strides match the memory
        # the kernel actually sees. (`.contiguous()` is a no-op when already
        # contiguous; broadcast size-1 dims are preserved and still map to
        # stride 0 below.)
        cond_buf = cond_buf.contiguous()
        x_buf = x_buf.contiguous()
        y_buf = y_buf.contiguous()
        # >4D: the 4D stride layout (OUT0..OUT3) collapses a *contiguous* >4D
        # tensor cleanly, but a broadcast (stride-0) axis can't be folded, so it
        # silently drops a dim. Materialize each broadcast operand to the full
        # output shape (gemma4 audio's block-attention masked_fill broadcasts a
        # (1,1,blocks,chunk,ctx) mask + a scalar fill over the heads axis).
        if len(out_shape) > 4:
            if tuple(cond_buf.shape) != out_shape:
                cond_buf = _expand_buf(cond_buf, out_shape).contiguous()
            if tuple(x_buf.shape) != out_shape:
                x_buf = _expand_buf(x_buf, out_shape).contiguous()
            if tuple(y_buf.shape) != out_shape:
                y_buf = _expand_buf(y_buf, out_shape).contiguous()
        padded_out, c_strides = _broadcast_layout(cond_buf, out_shape)
        _, x_strides = _broadcast_layout(x_buf, out_shape)
        _, y_strides = _broadcast_layout(y_buf, out_shape)
        return k_where_nd(
            cond_buf,
            x_buf,
            y_buf,
            N=math.prod(out_shape),
            OUT0=padded_out[0],
            OUT1=padded_out[1],
            OUT2=padded_out[2],
            OUT3=padded_out[3],
            C_STR0=c_strides[0],
            C_STR1=c_strides[1],
            C_STR2=c_strides[2],
            C_STR3=c_strides[3],
            X_STR0=x_strides[0],
            X_STR1=x_strides[1],
            X_STR2=x_strides[2],
            X_STR3=x_strides[3],
            Y_STR0=y_strides[0],
            Y_STR1=y_strides[1],
            Y_STR2=y_strides[2],
            Y_STR3=y_strides[3],
        ).reshape(out_shape)
    orig_dtype = x_buf._dtype.to_torch_dtype()
    x_f32 = _to_copy(x_buf, dtype=torch.float32)
    y_f32 = _to_copy(y_buf, dtype=torch.float32)
    result_f32 = _where_self(cond_buf, x_f32, y_f32)
    return _to_copy(result_f32, dtype=orig_dtype)


def _eq(a: ScalarValue, b: ScalarValue) -> AlloyBuffer:
    return _compare_nd(k_eq_nd, a, b)


def _ne(a: ScalarValue, b: ScalarValue) -> AlloyBuffer:
    return _compare_nd(k_ne_nd, a, b)


def _pow_tensor_scalar(x: AlloyBuffer, exponent: float) -> AlloyBuffer:
    if exponent == 0:
        return _full(x.shape, 1, dtype=x._dtype.to_torch_dtype())
    if exponent == 1:
        return x
    if exponent == 2:
        return x * x
    if exponent == 3:
        return x * x * x
    # Exact closed forms for the common fractional/negative exponents (gemma4
    # emits a raw pow(x, -0.5) rsqrt in norms the rms_norm rewrite doesn't fuse).
    if exponent == -0.5:
        return x.rsqrt()
    if exponent == 0.5:
        return x.sqrt()
    if exponent == -1:
        return _reciprocal(x)
    raise RuntimeError(
        f"Alloy pow: exponent={exponent} not supported on GPU (only 0, 1, 2, 3, -1, 0.5, -0.5)"
    )


def _pow_scalar_tensor(base: float, exponent: AlloyBuffer) -> AlloyBuffer:
    if base == 1.0:
        return _full(exponent.shape, 1, dtype=exponent._dtype.to_torch_dtype())
    if base <= 0.0:
        raise RuntimeError(f"pow.Scalar with non-positive base {base} not supported")
    log_base = math.log(base)
    scaled = exponent * log_base
    return scaled.exp()


def _reciprocal(x: AlloyBuffer) -> AlloyBuffer:
    one = _full(x.shape, 1.0, dtype=x._dtype.to_torch_dtype())
    return one / x
