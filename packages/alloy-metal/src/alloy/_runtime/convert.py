"""Shared framework-tensor → AlloyBuffer conversion."""

from __future__ import annotations

from alloy._compiler.dtypes import (
    DType,
    from_numpy,
    float32,
    float16,
    int64,
    int32,
    int16,
    int8,
    uint8,
)
from alloy._runtime.alloy_buffer import AlloyBuffer
import numpy as np

# torch dtype string → Alloy DType
_TORCH_DTYPES: dict[str, DType] = {
    "torch.float32": float32,
    "torch.float16": float16,
    "torch.int64": int64,
    "torch.int32": int32,
    "torch.int16": int16,
    "torch.int8": int8,
    "torch.uint8": uint8,
}


def to_alloy_buffer(arg) -> AlloyBuffer:
    """Convert a framework tensor (torch/mlx/numpy) to AlloyBuffer.

    Raises TypeError if the input cannot be converted.
    Zero-copy for contiguous CPU tensors.
    """
    if isinstance(arg, AlloyBuffer):
        return arg

    if isinstance(arg, np.ndarray):
        buf = AlloyBuffer.from_raw_ptr(
            arg.ctypes.data,
            tuple(arg.shape),
            tuple(arg.strides),
            from_numpy(arg.dtype),
            arg.nbytes,
        )
        buf._np_ref = arg
        return buf

    # torch.Tensor
    if hasattr(arg, "detach") and hasattr(arg, "data_ptr") and hasattr(arg, "device"):
        if str(arg.device).startswith("mps"):
            import torch  # scoped: optional dep — only reached when caller passes a torch.Tensor

            torch.mps.synchronize()
            arg = arg.cpu()
        if not arg.is_contiguous():
            arg = arg.contiguous()
        ptr = arg.data_ptr()
        shape = tuple(arg.shape)
        strides = tuple(s * arg.element_size() for s in arg.stride())
        dt = _TORCH_DTYPES.get(str(arg.dtype), float32)
        nbytes = arg.nelement() * arg.element_size()
        buf = AlloyBuffer.from_raw_ptr(ptr, shape, strides, dt, nbytes)
        buf._ext_ref = arg
        return buf

    # mlx.array
    if type(arg).__module__.startswith("mlx"):
        import mlx.core as mx  # noqa: PLC0415

        mx.eval(arg)
        arr = np.array(arg, copy=False)
        buf = AlloyBuffer.from_raw_ptr(
            arr.ctypes.data,
            tuple(arr.shape),
            tuple(arr.strides),
            from_numpy(arr.dtype),
            arr.nbytes,
        )
        buf._np_ref = arr
        return buf

    raise TypeError(
        f"Cannot convert {type(arg).__name__} to AlloyBuffer. "
        f"Expected AlloyBuffer, numpy array, torch.Tensor, or mlx.array."
    )
