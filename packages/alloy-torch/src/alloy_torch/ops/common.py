"""Shared shape and dtype helpers for torch op lowering."""

from collections.abc import Sequence
import operator
from functools import reduce

import torch

from alloy._runtime.alloy_buffer import AlloyBuffer

_FLOAT_DTYPES: frozenset[torch.dtype] = frozenset(
    (torch.bfloat16, torch.float16, torch.float32, torch.float64)
)
_INTEGER_DTYPES: frozenset[torch.dtype] = frozenset(
    (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
)

_IR_TO_TORCH: dict[str, torch.dtype] = {
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

ScalarLike = AlloyBuffer | bool | int | float | torch.Tensor | None


def _numel(shape: tuple[int, ...]) -> int:
    if not shape:
        return 1
    return reduce(operator.mul, shape, 1)


def _broadcast_shapes(*shapes: tuple[int, ...]) -> tuple[int, ...]:
    """Compute broadcast output shape without touching torch dispatch."""
    ndim = max(len(shape) for shape in shapes)
    padded = [(1,) * (ndim - len(shape)) + shape for shape in shapes]
    return tuple(max(dims) for dims in zip(*padded))


def _pad_shape_4d(shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    """Pad shape to exactly 4D for kernel constexprs."""
    padded = (1,) * (4 - len(shape)) + shape if len(shape) < 4 else shape[-4:]
    return (padded[0], padded[1], padded[2], padded[3])


def _elem_strides_4d(buffer: AlloyBuffer) -> tuple[int, int, int, int]:
    """Element strides from an AlloyBuffer's backing allocation, padded to 4D."""
    itemsize = buffer._dtype.itemsize
    element_strides = tuple(stride // itemsize for stride in buffer._strides)
    padded = (
        (0,) * (4 - len(element_strides)) + element_strides
        if len(element_strides) < 4
        else element_strides[-4:]
    )
    return (padded[0], padded[1], padded[2], padded[3])


def _normalize_shape(shape: int | Sequence[int]) -> tuple[int, ...]:
    if isinstance(shape, Sequence) and not isinstance(shape, str):
        return tuple(int(dim) for dim in shape)
    return (int(shape),)


def _normalize_dim(dim: int, ndim: int) -> int:
    return dim if dim >= 0 else dim + ndim


def _expand_lazy_buffer(buf: AlloyBuffer, out_shape: tuple[int, ...]) -> AlloyBuffer:
    """Broadcast to out_shape by setting strides to 0 for broadcast dims."""
    if buf.shape == out_shape:
        return buf

    src_shape = buf.shape
    pad = len(out_shape) - len(src_shape)
    padded_shape = (1,) * pad + src_shape
    padded_strides = (0,) * pad + buf._strides
    new_strides = tuple(
        0 if padded_shape[i] == 1 and out_shape[i] > 1 else padded_strides[i]
        for i in range(len(out_shape))
    )

    new_buf = AlloyBuffer(
        buf._parent_handle,
        buf._offset,
        buf._shape,
        buf._strides,
        buf._dtype,
        raw_ptr=buf._raw_ptr,
        total_nbytes=buf._total_nbytes,
    )
    new_buf.reinterpret(out_shape, new_strides)
    buf._view_of(new_buf)
    return new_buf


def _root_flat_buf(buf: AlloyBuffer) -> AlloyBuffer:
    """Return a contiguous 1D view of buf's entire root allocation.

    Used when a kernel handles strided access via explicit offset/stride
    constexprs. The caller has already extracted offset and strides from
    the original view; this just gives the kernel a flat, contiguous buffer
    to index into, so _queue_op never sees non-contiguous data.
    """
    root = AlloyBuffer(
        buf._parent_handle,
        buf._offset,
        buf._shape,
        buf._strides,
        buf._dtype,
        raw_ptr=buf._raw_ptr,
        total_nbytes=buf.metal_nbytes,
    )
    root.root_flat()
    buf._view_of(root)
    return root


def _dtype_of(value: ScalarLike) -> torch.dtype | None:
    if isinstance(value, AlloyBuffer):
        return value._dtype.to_torch_dtype()
    if isinstance(value, torch.Tensor):
        return value.dtype
    if isinstance(value, bool):
        return torch.bool
    if isinstance(value, int):
        return torch.int64
    if isinstance(value, float):
        return torch.float32
    return None


def _shape_of(value: ScalarLike) -> tuple[int, ...]:
    if isinstance(value, AlloyBuffer):
        return value.shape
    if isinstance(value, torch.Tensor):
        return tuple(value.shape)
    if isinstance(value, (bool, int, float)):
        return ()
    raise TypeError(f"Cannot determine shape for {type(value)!r}")


def _is_float_dtype(dtype: torch.dtype | None) -> bool:
    return dtype in _FLOAT_DTYPES


def _is_integer_dtype(dtype: torch.dtype | None) -> bool:
    return dtype in _INTEGER_DTYPES


def _is_bool_dtype(dtype: torch.dtype | None) -> bool:
    return dtype == torch.bool


def _reference_float_dtype(*values: ScalarLike) -> torch.dtype | None:
    for value in values:
        dtype = _dtype_of(value)
        if _is_float_dtype(dtype):
            return dtype
    return None


def _reference_kernel_dtype(*values: ScalarLike) -> torch.dtype | None:
    float_dtype = _reference_float_dtype(*values)
    if float_dtype is not None:
        return float_dtype
    for value in values:
        dtype = _dtype_of(value)
        if _is_integer_dtype(dtype):
            return dtype
    return None
