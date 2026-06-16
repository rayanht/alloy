"""Scalar and dtype coercion helpers for torch op lowering."""

import torch

from alloy._compiler.dtypes import from_torch_dtype
from alloy._dispatch.buf_utils import _alloc_aligned
from alloy._runtime.alloy_buffer import AlloyBuffer
from alloy_torch.ops.casting import _to_copy
from alloy_torch.ops.common import _dtype_of

ScalarValue = AlloyBuffer | bool | int | float | torch.Tensor


def _scalar_lazy_buffer(value: ScalarValue, *, dtype: torch.dtype | None = None) -> AlloyBuffer:
    target_dtype = dtype or _dtype_of(value) or torch.float32
    buf = _alloc_aligned((), from_torch_dtype(target_dtype))
    buf.write_scalar(value)
    return buf


def _broadcast_layout(
    buf: AlloyBuffer, out_shape: tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    itemsize = buf._dtype.itemsize
    ndim = len(buf._shape)
    pad = len(out_shape) - ndim
    strides = tuple(stride // itemsize for stride in buf._strides)
    shape = buf._shape
    broadcast_strides = (0,) * pad + tuple(
        0 if shape[index] == 1 and out_shape[pad + index] != 1 else strides[index]
        for index in range(ndim)
    )
    out_ndim = len(out_shape)
    pad4 = 4 - out_ndim
    return (1,) * pad4 + out_shape, (0,) * pad4 + broadcast_strides


_WIDENING_PROMOTIONS: frozenset[tuple[torch.dtype, torch.dtype]] = frozenset(
    ((torch.float16, torch.float32),)
)


def _coerce_lazy_value(value: AlloyBuffer, *, dtype: torch.dtype | None = None) -> AlloyBuffer:
    if dtype is not None and value._dtype.to_torch_dtype() != dtype:
        if (value._dtype.to_torch_dtype(), dtype) in _WIDENING_PROMOTIONS:
            return value  # MSL auto-promotes in arithmetic.
        return _to_copy(value, dtype=dtype)
    return value


def _coerce_mask_numeric(value: ScalarValue) -> AlloyBuffer:
    if isinstance(value, AlloyBuffer):
        if _dtype_of(value) == torch.bool and str(value.dtype) == "int32":
            return value  # already int32, no conversion needed
        if _dtype_of(value) == torch.bool:
            return value
        return _coerce_lazy_value(value, dtype=torch.int32)
    if isinstance(value, bool):
        return _scalar_lazy_buffer(int(value), dtype=torch.int32)
    raise TypeError(f"Expected AlloyBuffer or bool mask, got {type(value)!r}")
