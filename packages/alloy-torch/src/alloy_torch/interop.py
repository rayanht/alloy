"""Interop between PyTorch tensors and Alloy Metal buffers.

On Apple Silicon, CPU tensors live in unified memory — the same physical pages
are accessible by both CPU and GPU. For contiguous CPU tensors, this path can bind the
tensor storage directly:

  torch.Tensor (CPU) → .numpy() → MetalBuffer.from_numpy() → GPU kernel

Non-contiguous tensors are made contiguous first, and MPS tensors require a copy to CPU.
The typical Alloy workflow avoids MPS tensors because Alloy is the GPU backend.
"""

from __future__ import annotations

import numpy as np
import torch
from alloy._runtime.metal import MetalBuffer, MetalDevice, default_device

_TORCH_TO_NP_DTYPE = {
    torch.float32: np.float32,
    torch.float16: np.float16,
    torch.int64: np.int64,
    torch.int32: np.int32,
    torch.int16: np.int16,
    torch.int8: np.int8,
    torch.uint8: np.uint8,
    torch.bool: np.uint8,
    torch.uint64: np.uint64,
}


def tensor_to_buffer(tensor: torch.Tensor, device: MetalDevice | None = None) -> MetalBuffer:
    """Convert a torch.Tensor to a MetalBuffer.

    Direct storage binding for contiguous CPU tensors on Apple Silicon.
    Non-contiguous tensors and MPS tensors are copied first.

    Args:
        tensor: Input tensor (CPU or MPS).
        device: MetalDevice to use. Defaults to the system GPU.

    Returns:
        MetalBuffer sharing memory with a contiguous CPU tensor or a copied buffer.
    """
    if device is None:
        device = default_device()

    if tensor.device.type == "mps":
        torch.mps.synchronize()
        tensor = tensor.cpu()

    # Alloy represents bool as int32 internally (comparison results, mask kernels),
    # but a torch bool tensor is 1 byte/element. Binding its storage directly would
    # make the int32-typed buffer read 4 packed bool bytes as one int (0x01010101).
    # Widen to int32 here so the 1-byte data becomes the expected 4-byte 0/1.
    if tensor.dtype == torch.bool:
        tensor = tensor.to(torch.int32)

    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    arr = tensor.detach().numpy()
    buf = MetalBuffer.from_numpy(device, arr)
    # Keep tensor alive so its memory isn't freed
    buf._torch_ref = tensor
    return buf


def buffer_to_tensor(
    buffer: MetalBuffer,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    device: str = "cpu",
) -> torch.Tensor:
    """View a MetalBuffer as a PyTorch tensor.

    Args:
        buffer: Source MetalBuffer.
        dtype: PyTorch dtype for the result.
        shape: Shape for the result.
        device: Target device ('cpu' or 'mps'). CPU can share the MetalBuffer storage.

    Returns:
        torch.Tensor sharing memory with the buffer (CPU) or a copy (MPS).
    """
    np_dtype = _TORCH_TO_NP_DTYPE.get(dtype)
    if np_dtype is None:
        raise ValueError(f"Unsupported dtype: {dtype}")

    arr = buffer.to_numpy(np_dtype, shape)
    t = torch.from_numpy(arr)

    if device == "mps":
        t = t.to("mps")

    return t


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a torch.Tensor to a numpy array for Alloy kernel dispatch.

    Direct view for contiguous CPU tensors. MPS and non-contiguous tensors are copied.
    """
    if tensor.device.type == "mps":
        torch.mps.synchronize()
        tensor = tensor.cpu()

    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    return tensor.detach().numpy()
