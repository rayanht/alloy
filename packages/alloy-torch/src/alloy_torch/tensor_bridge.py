"""Torch tensor construction helpers shared by backend and training hooks."""

from __future__ import annotations

import ctypes
import weakref
from typing import TYPE_CHECKING

import torch

from alloy._compiler.dtypes import DType
from alloy._runtime import _metal_ext
from alloy._runtime.alloy_buffer import _compute_contiguous_strides

if TYPE_CHECKING:
    import numpy as np

IR_TO_TORCH: dict[str, torch.dtype] = {
    "f16": torch.float16,
    "f32": torch.float32,
    "bf16": torch.bfloat16,
    "i8": torch.int8,
    "i16": torch.int16,
    "i32": torch.int32,
    "i64": torch.int64,
    "u8": torch.uint8,
    "u32": torch.uint32,
}


def make_tensor_from_ptr(
    base_ptr: int,
    shape: tuple[int, ...],
    dtype: DType,
    byte_offset: int = 0,
    total_nbytes: int = 0,
    strides_bytes: tuple[int, ...] | None = None,
    make_contiguous: bool = False,
) -> torch.Tensor:
    """Create a torch tensor backed by a raw Metal pointer."""
    torch_dt = IR_TO_TORCH.get(dtype.ir, torch.float32)
    itemsize = dtype.itemsize
    if strides_bytes is None:
        strides_bytes = _compute_contiguous_strides(shape, itemsize)
    strides_elem = tuple(s // itemsize for s in strides_bytes)
    is_contiguous = strides_bytes == _compute_contiguous_strides(shape, itemsize)

    if is_contiguous and byte_offset == 0:
        elems = 1
        for s in shape:
            elems *= s
        if total_nbytes and elems * itemsize > total_nbytes:
            raise RuntimeError(
                f"alloy: {shape} of {dtype.ir} needs {elems * itemsize} bytes but the "
                f"buffer holds {total_nbytes} — a view past the end would read "
                f"unowned memory"
            )
        return torch.frombuffer(
            (ctypes.c_uint8 * (elems * itemsize)).from_address(base_ptr),
            dtype=torch_dt,
            count=elems,
        ).reshape(shape)

    # Non-contiguous view or offset: create a strided view.
    min_needed = (
        (sum(max(0, s - 1) * st for s, st in zip(shape, strides_bytes)) + itemsize + byte_offset)
        if shape
        else itemsize
    )
    if total_nbytes == 0:
        total_nbytes = max(min_needed, 1)
    elif min_needed > total_nbytes:
        elems = 1
        for s in shape:
            elems *= s
        contig_needed = byte_offset + elems * itemsize
        if contig_needed <= total_nbytes:
            strides_bytes = _compute_contiguous_strides(shape, itemsize)
            strides_elem = tuple(s // itemsize for s in strides_bytes)
        else:
            return torch.zeros(shape, dtype=torch_dt)
    buf_elems = total_nbytes // itemsize
    flat = torch.frombuffer(
        (ctypes.c_uint8 * total_nbytes).from_address(base_ptr),
        dtype=torch_dt,
        count=buf_elems,
    )
    offset_elems = byte_offset // itemsize
    view = torch.as_strided(flat, shape, strides_elem, offset_elems)
    return view.contiguous() if make_contiguous else view


def alloy_empty(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """Uninitialized torch tensor whose storage is an alloy-owned Metal buffer.
    The handle is released when the tensor's storage dies."""
    elems = 1
    for s in shape:
        elems *= int(s)
    nbytes = elems * dtype.itemsize
    handle = _metal_ext.buf_alloc(nbytes)
    raw = (ctypes.c_uint8 * nbytes).from_address(_metal_ext.buf_ptr(handle))
    weakref.finalize(raw, release_handle, handle)
    return torch.frombuffer(raw, dtype=dtype, count=elems).reshape(shape)


def release_handle(handle: int) -> None:
    try:
        _metal_ext.buf_release(handle)
    except Exception:
        pass


def alloy_tensor_from_numpy(arr: "np.ndarray") -> torch.Tensor:
    """Copy `arr` into an alloy-owned Metal buffer and return a torch view of it."""
    import numpy as np

    src = np.ascontiguousarray(arr)
    out = alloy_empty(tuple(src.shape), torch.from_numpy(np.empty(0, src.dtype)).dtype)
    np.copyto(out.numpy(), src)
    return out
