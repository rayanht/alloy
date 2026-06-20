"""Interop between MLX arrays and Alloy Metal buffers.

MLX arrays on Apple Silicon use unified memory, so an evaluated array views as a
numpy array without copying (np.array(arr, copy=False)), then wraps as a
MetalBuffer: mlx.array → numpy view → MetalBuffer → GPU kernel.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
from alloy._runtime.metal import MetalBuffer, MetalDevice, default_device

_MLX_TO_NP_DTYPE = {
    mx.float32: np.float32,
    mx.float16: np.float16,
    mx.int32: np.int32,
    mx.int16: np.int16,
    mx.int8: np.int8,
}

_NP_TO_MLX_DTYPE = {v: k for k, v in _MLX_TO_NP_DTYPE.items()}


def array_to_buffer(array: mx.array, device: MetalDevice | None = None) -> MetalBuffer:
    """Convert an MLX array to a MetalBuffer.

    Args:
        array: Input MLX array.
        device: MetalDevice to use. Defaults to the system GPU.

    Returns:
        MetalBuffer sharing memory with the MLX array.
    """
    if device is None:
        device = default_device()

    mx.eval(array)
    arr = np.array(array, copy=False)
    buf = MetalBuffer.from_numpy(device, arr)
    # Keep MLX array alive so its memory isn't freed
    buf._mlx_ref = array
    return buf


def buffer_to_array(
    buffer: MetalBuffer,
    dtype: mx.Dtype,
    shape: tuple[int, ...],
) -> mx.array:
    """View a MetalBuffer as an MLX array.

    Args:
        buffer: Source MetalBuffer.
        dtype: MLX dtype for the result.
        shape: Shape for the result.

    Returns:
        mlx.array viewing the buffer's data.
    """
    np_dtype = _MLX_TO_NP_DTYPE.get(dtype)
    if np_dtype is None:
        raise ValueError(f"Unsupported dtype: {dtype}")

    arr = buffer.to_numpy(np_dtype, shape)
    return mx.array(arr)


def array_to_numpy(array: mx.array) -> np.ndarray:
    """Convert an MLX array to a numpy array for Alloy kernel dispatch."""
    mx.eval(array)
    return np.array(array, copy=False)
