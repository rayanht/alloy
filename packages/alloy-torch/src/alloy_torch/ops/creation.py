"""Tensor creation handlers for torch op lowering."""

from collections.abc import Sequence
import ctypes

import torch

from alloy._compiler.dtypes import from_torch_dtype
from alloy._dispatch.buf_utils import _alloc_aligned
from alloy._runtime.alloy_buffer import AlloyBuffer
from alloy_torch.ops.common import _dtype_of, _normalize_shape

ScalarFill = AlloyBuffer | bool | int | float | torch.Tensor | None
ShapeLike = int | Sequence[int]


def _full(
    shape: ShapeLike,
    fill_value: ScalarFill,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool = False,
) -> AlloyBuffer:
    del device, layout, pin_memory
    out_shape = _normalize_shape(shape)
    out_dtype = from_torch_dtype(dtype or _dtype_of(fill_value) or torch.float32)
    out = _alloc_aligned(out_shape, out_dtype)
    out[...] = fill_value
    return out


def _zeros(
    shape: ShapeLike,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool = False,
) -> AlloyBuffer:
    return _full(shape, 0, dtype=dtype, device=device, layout=layout, pin_memory=pin_memory)


def _new_empty_strided(
    ref: AlloyBuffer,
    size: ShapeLike,
    stride: Sequence[int],
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | str | None = None,
    pin_memory: bool = False,
) -> AlloyBuffer:
    del stride, layout, device, pin_memory
    shape = _normalize_shape(size)
    out_dtype = from_torch_dtype(dtype) if dtype is not None else ref._dtype
    return _alloc_aligned(shape, out_dtype)


def _ones_like(
    x: AlloyBuffer,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | str | None = None,
    pin_memory: bool = False,
    memory_format: torch.memory_format | None = None,
) -> AlloyBuffer:
    del memory_format
    out_dtype = dtype or x._dtype.to_torch_dtype()
    return _full(x.shape, 1, dtype=out_dtype, device=device, layout=layout, pin_memory=pin_memory)


def _full_like(
    x: AlloyBuffer,
    fill_value: ScalarFill,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | str | None = None,
    pin_memory: bool = False,
    memory_format: torch.memory_format | None = None,
) -> AlloyBuffer:
    del memory_format
    out_dtype = dtype or x._dtype.to_torch_dtype()
    return _full(
        x.shape,
        fill_value,
        dtype=out_dtype,
        device=device,
        layout=layout,
        pin_memory=pin_memory,
    )


def _scalar_tensor(
    value: ScalarFill,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    layout: torch.layout | None = None,
) -> AlloyBuffer:
    del device, layout
    return _full((), value, dtype=dtype)


def _arange_default(
    end: int | float,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool = False,
) -> AlloyBuffer:
    del device, layout, pin_memory
    return _arange_start_step(0, int(end), 1, dtype=dtype)


def _arange_start(
    start: int | float,
    end: int | float,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool = False,
) -> AlloyBuffer:
    del device, layout, pin_memory
    return _arange_start_step(int(start), int(end), 1, dtype=dtype)


def _arange_start_step(
    start: int | float,
    end: int | float | None = None,
    step: int | float = 1,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool = False,
) -> AlloyBuffer:
    del device, layout, pin_memory
    if end is None:
        start, end = 0, start
    start_i = int(start)
    end_i = int(end)
    step_i = int(step)
    n = len(range(start_i, end_i, step_i))
    buf = _alloc_aligned((n,), from_torch_dtype(dtype or torch.int64))

    itemsize = buf._dtype.itemsize
    ptr = buf.data_ptr
    if buf._dtype.ir in ("f16", "f32", "f64", "bf16"):
        float_ctype = {2: ctypes.c_uint16, 4: ctypes.c_float, 8: ctypes.c_double}[itemsize]
        for i in range(n):
            ctypes.cast(ptr + i * itemsize, ctypes.POINTER(float_ctype))[0] = float(
                start_i + i * step_i
            )
    else:
        int_ctype = {
            1: ctypes.c_int8,
            2: ctypes.c_int16,
            4: ctypes.c_int32,
            8: ctypes.c_int64,
        }[itemsize]
        for i in range(n):
            ctypes.cast(ptr + i * itemsize, ctypes.POINTER(int_ctype))[0] = start_i + i * step_i
    return buf
