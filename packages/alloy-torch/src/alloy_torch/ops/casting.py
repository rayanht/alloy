"""Cast and copy handlers for torch op lowering."""

from typing import TYPE_CHECKING

import alloy
import torch

from alloy._compiler.dtypes import from_torch_dtype
from alloy._dispatch.buf_utils import _alloc_scratch
from alloy._dispatch.kernel import KernelFunction
from alloy._runtime.alloy_buffer import AlloyBuffer
from alloy._runtime.buffer_ops import _expand_buf

if TYPE_CHECKING:
    output = AlloyBuffer
    constexpr = int
else:
    output = alloy.output
    constexpr = alloy.constexpr


@alloy.kernel
def k_to_bool_mask(
    x,
    out: output,
    N: constexpr,
    BLOCK_SIZE: constexpr = 1024,
):
    pid = alloy.program_id(0)
    offs = pid * BLOCK_SIZE + alloy.arange(0, BLOCK_SIZE)
    mask = offs < N
    value = alloy.load(x + offs, mask=mask)
    alloy.store(out + offs, alloy.cast(value != 0, "int32"), mask=mask)


@alloy.kernel
def k_copy(x, out: output, N: constexpr, BLOCK_SIZE: constexpr = 1024):
    pid = alloy.program_id(0)
    offs = pid * BLOCK_SIZE + alloy.arange(0, BLOCK_SIZE)
    mask = offs < N
    alloy.store(out + offs, alloy.load(x + offs, mask=mask), mask=mask)


def _make_cast_kernel(name: str, target_dtype: str) -> KernelFunction:
    @alloy.kernel
    def k(x, out: output, N: constexpr, BLOCK_SIZE: constexpr = 1024):
        pid = alloy.program_id(0)
        offs = pid * BLOCK_SIZE + alloy.arange(0, BLOCK_SIZE)
        mask = offs < N
        value = alloy.load(x + offs, mask=mask)
        alloy.store(out + offs, alloy.cast(value, target_dtype), mask=mask)

    k.name = name
    return k


_CAST_KERNELS: dict[torch.dtype, KernelFunction] = {
    torch.float16: _make_cast_kernel("cast_f16", "float16"),
    torch.bfloat16: _make_cast_kernel("cast_bf16", "bfloat16"),
    torch.float32: _make_cast_kernel("cast_f32", "float32"),
    torch.int64: _make_cast_kernel("cast_i64", "int64"),
    torch.int32: _make_cast_kernel("cast_i32", "int32"),
    torch.int16: _make_cast_kernel("cast_i16", "int16"),
    torch.int8: _make_cast_kernel("cast_i8", "int8"),
    torch.uint8: _make_cast_kernel("cast_u8", "uint8"),
    torch.uint64: _make_cast_kernel("cast_u64", "uint64"),
}


def _to_copy(
    x: AlloyBuffer,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool = False,
    memory_format: torch.memory_format | None = None,
    non_blocking: bool = False,
) -> AlloyBuffer:
    del device, layout, pin_memory, memory_format, non_blocking
    target_dtype = dtype or x._dtype.to_torch_dtype()
    source_dtype = x._dtype.to_torch_dtype()
    if (source_dtype, target_dtype) == (torch.float16, torch.float32):
        return x
    if source_dtype == target_dtype:
        return x
    out_arr = _alloc_scratch(x.shape, from_torch_dtype(target_dtype))
    if target_dtype == torch.bool:
        return k_to_bool_mask(x, out_arr, N=x.size)
    cast_kernel = _CAST_KERNELS.get(target_dtype)
    if cast_kernel is None:
        raise NotImplementedError(f"Alloy _to_copy cast does not support dtype {target_dtype}")
    return cast_kernel(x, out_arr, N=x.size)


def _copy(self: AlloyBuffer, src: AlloyBuffer, non_blocking: bool = False) -> AlloyBuffer:
    del non_blocking
    if src._dtype != self._dtype:
        src = _to_copy(src, dtype=self._dtype.to_torch_dtype())
    if src.shape != self.shape:
        src = _expand_buf(src, self.shape)
    out = _alloc_scratch(self.shape, self._dtype)
    return k_copy(src, out, N=out.size)
