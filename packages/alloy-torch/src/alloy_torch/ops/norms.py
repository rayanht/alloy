"""Normalization handlers for torch op lowering."""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import cast

import torch

from alloy._compiler.dtypes import from_torch_dtype
from alloy._dispatch.buf_utils import _alloc_aligned, _alloc_scratch
from alloy._dispatch.kernel import KernelFunction
from alloy._runtime.alloy_buffer import AlloyBuffer
from alloy.std.gemm import dot_transpose_rhs
from alloy.std.norm import layernorm, rms_norm, rms_norm_backward
from alloy.std.reductions import mean
from alloy_torch.ops.casting import _to_copy
from alloy_torch.ops.common import _IR_TO_TORCH, _normalize_shape, _numel
from alloy_torch.ops.creation import _full
from alloy_torch.ops.linalg import _addmm, _mm

NormalizedShape = int | Sequence[int]
LayerNormArg = NormalizedShape | AlloyBuffer | None

_LAYER_NORM_KERNEL = cast(KernelFunction, layernorm)
_MEAN_KERNEL = cast(KernelFunction, mean)
_DOT_TRANSPOSE_RHS_KERNEL = cast(KernelFunction, dot_transpose_rhs)
_RMS_NORM_KERNEL = cast(KernelFunction, rms_norm)
_RMS_NORM_BACKWARD_KERNEL = cast(KernelFunction, rms_norm_backward)


def _normalize_layer_shape(value: LayerNormArg) -> tuple[int, ...]:
    if value is None or isinstance(value, AlloyBuffer):
        raise NotImplementedError(
            f"layer norm expected normalized shape, got {type(value).__name__}"
        )
    return _normalize_shape(value)


def _optional_buffer(value: LayerNormArg, name: str) -> AlloyBuffer | None:
    if value is None:
        return None
    if isinstance(value, AlloyBuffer):
        return value
    raise NotImplementedError(
        f"layer norm expected AlloyBuffer or None for {name}, got {type(value).__name__}"
    )


def _native_group_norm(
    x: AlloyBuffer,
    weight: AlloyBuffer | None,
    bias: AlloyBuffer | None,
    N: int,
    C: int,
    HxW: int,
    group: int,
    eps: float = 1e-5,
) -> tuple[AlloyBuffer, AlloyBuffer, AlloyBuffer]:
    """GroupNorm: normalize each of `group` channel groups independently.

    Input: (N, C, HxW) flattened. Reshape to (N*G, C//G * HxW), normalize
    each row, reshape back, apply per-channel weight and bias.
    """
    channels_per_group = C // group
    elems_per_group = channels_per_group * HxW
    flat = x.reshape((N * group, elems_per_group))

    ones = _full((elems_per_group,), 1, dtype=x._dtype.to_torch_dtype())
    zeros_buf = _full((elems_per_group,), 0, dtype=x._dtype.to_torch_dtype())
    normed = _LAYER_NORM_KERNEL(flat, ones, zeros_buf, EPS=eps)

    normed = normed.reshape((N, C, HxW))
    if weight is not None:
        normed = normed * weight.reshape((1, C, 1))
    if bias is not None:
        normed = normed + bias.reshape((1, C, 1))

    out = normed.reshape(x.shape)
    mean_dummy = _alloc_aligned((N, group), x.dtype)
    rstd_dummy = _alloc_aligned((N, group), x.dtype)
    return out, mean_dummy, rstd_dummy


def _native_layer_norm(
    x: AlloyBuffer,
    normalized_shape: NormalizedShape,
    weight: AlloyBuffer | None,
    bias: AlloyBuffer | None,
    eps: float = 1e-5,
) -> tuple[AlloyBuffer, AlloyBuffer, AlloyBuffer]:
    x = x.contiguous()
    norm_shape = _normalize_shape(normalized_shape)
    if not norm_shape:
        raise NotImplementedError("native_layer_norm requires a non-empty normalized shape")
    if x.shape[-len(norm_shape) :] != norm_shape:
        raise NotImplementedError(
            f"native_layer_norm expected trailing dims {norm_shape}, got {x.shape}"
        )

    rows = math.prod(x.shape[: -len(norm_shape)]) if len(x.shape) > len(norm_shape) else 1
    cols = math.prod(norm_shape)
    flat = x.reshape((rows, cols))

    gamma = _full((cols,), 1, dtype=x._dtype.to_torch_dtype()) if weight is None else weight
    beta = _full((cols,), 0, dtype=x._dtype.to_torch_dtype()) if bias is None else bias
    out = _LAYER_NORM_KERNEL(flat, gamma, beta, EPS=eps)

    # Use E[X^2] - E[X]^2 instead of mean((x - mean)^2). The broadcast-subtract
    # plus row-wise reduce chain mis-reduces at cols < 64 and breaks tiny hidden
    # dim training backward saved tensors.
    row_mean = _MEAN_KERNEL(flat, axis=1)
    ex2 = _MEAN_KERNEL(flat * flat, axis=1)
    var = ex2 - row_mean * row_mean
    rstd_flat = (var.reshape(rows, 1) + eps).rsqrt()

    batch_shape = x.shape[: -len(norm_shape)]
    return (
        out.reshape(x.shape),
        row_mean.reshape(batch_shape + (1,)),
        rstd_flat.reshape(batch_shape + (1,)),
    )


def _fused_gemm_layernorm(
    is_addmm: bool,
    gemm_args: tuple[AlloyBuffer, ...],
    residual: AlloyBuffer | None,
    ln_args: tuple[LayerNormArg, ...],
    eps: float = 1e-5,
) -> tuple[tuple[AlloyBuffer, None, None], AlloyBuffer]:
    """Fused GEMM + residual + LayerNorm with dual output."""
    if is_addmm:
        mat1, mat2 = gemm_args[1], gemm_args[2]
    else:
        mat1, mat2 = gemm_args[0], gemm_args[1]

    norm_shape = _normalize_layer_shape(ln_args[0])
    ln_weight = _optional_buffer(ln_args[1], "weight") if len(ln_args) > 1 else None
    ln_bias = _optional_buffer(ln_args[2], "bias") if len(ln_args) > 2 else None

    buf_x = mat1
    buf_w = mat2
    rows = _numel(buf_x.shape[:-1])
    cols_out = buf_w.shape[-1]
    cols = math.prod(norm_shape)

    torch_dtype = buf_x._dtype.to_torch_dtype()
    gamma = ln_weight if ln_weight is not None else _full((cols,), 1, dtype=torch_dtype)
    beta = ln_bias if ln_bias is not None else _full((cols,), 0, dtype=torch_dtype)

    original_shape = buf_x.shape[:-1] + (cols_out,)
    if is_addmm:
        gemm_out = _addmm(gemm_args[0], mat1, mat2)
    else:
        gemm_out = _mm(mat1, mat2)

    if residual is not None and len(residual.shape) > len(original_shape):
        original_shape = residual.shape

    if residual is not None:
        gemm_for_add = (
            gemm_out.reshape(original_shape) if gemm_out.shape != original_shape else gemm_out
        )
        res_result = gemm_for_add + residual
    else:
        res_result = gemm_out

    flat_res = res_result.reshape((rows, cols))
    ln_out = _LAYER_NORM_KERNEL(flat_res, gamma, beta, EPS=eps)
    ln_result = ln_out.reshape(original_shape)
    res_result = (
        res_result.reshape(original_shape) if res_result.shape != original_shape else res_result
    )
    return ((ln_result, None, None), res_result)


def _fused_gemm_rmsnorm(
    gemm_args: tuple[AlloyBuffer, ...],
    residual: AlloyBuffer,
    weight: AlloyBuffer,
    eps: float = 1e-6,
) -> tuple[AlloyBuffer, AlloyBuffer, AlloyBuffer]:
    """Fused GEMM + residual + RMSNorm with dual output."""
    mat1, mat2 = gemm_args[0], gemm_args[1]
    buf_x = mat1
    buf_bt = mat2
    rows = _numel(buf_x.shape[:-1])
    reduction = buf_x.shape[-1]
    cols = buf_bt.shape[0]

    flat_x = buf_x.reshape((rows, reduction))
    flat_res = residual.reshape((rows, cols))
    flat_weight = weight.reshape((cols,))

    if len(residual.shape) > len(buf_x.shape):
        original_shape = residual.shape
    else:
        original_shape = buf_x.shape[:-1] + (cols,)

    out = _alloc_scratch((rows, cols), flat_x.dtype)
    gemm_out = _DOT_TRANSPOSE_RHS_KERNEL(flat_x, buf_bt, out)
    res_sum = gemm_out + flat_res
    rms_out, rsqrt = _fused_rms_norm(res_sum, flat_weight, eps=eps)
    rms_result = rms_out.reshape(original_shape)
    res_result = res_sum.reshape(original_shape)
    rsqrt_shape = original_shape[:-1] + (1,)
    return (rms_result, res_result, rsqrt.reshape(rsqrt_shape))


def _alloy_gemm_residual_layernorm_handler(
    mat1: AlloyBuffer,
    mat2: AlloyBuffer,
    bias: AlloyBuffer | None,
    residual: AlloyBuffer | None,
    ln_weight: AlloyBuffer | None,
    ln_bias: AlloyBuffer | None,
    normalized_shape: NormalizedShape,
    eps: float,
) -> tuple[AlloyBuffer, AlloyBuffer, AlloyBuffer]:
    is_addmm = bias is not None
    gemm_args = (bias, mat1, mat2) if is_addmm and bias is not None else (mat1, mat2)
    ln_args: tuple[LayerNormArg, ...] = (normalized_shape, ln_weight, ln_bias)
    ln_tuple, res_sum = _fused_gemm_layernorm(is_addmm, gemm_args, residual, ln_args, eps=eps)
    ln_result = ln_tuple[0]
    return ln_result, res_sum, ln_result


def _alloy_gemm_residual_rmsnorm_handler(
    mat1: AlloyBuffer,
    mat2: AlloyBuffer,
    residual: AlloyBuffer,
    weight: AlloyBuffer,
    eps: float,
) -> tuple[AlloyBuffer, AlloyBuffer, AlloyBuffer]:
    return _fused_gemm_rmsnorm((mat1, mat2), residual, weight, eps=eps)


def _fused_rms_norm(
    x: AlloyBuffer, weight: AlloyBuffer, eps: float = 1e-6
) -> tuple[AlloyBuffer, AlloyBuffer]:
    """Fused RMSNorm via alloy.std.norm.rms_norm. Returns (normed, rsqrt)."""
    f32 = from_torch_dtype(torch.float32)
    shape = x.shape
    cols = shape[-1]
    rows = 1
    for dim in shape[:-1]:
        rows *= dim
    flat_x = x.reshape((rows, cols))
    flat_w = weight.reshape((cols,))
    out_buf = _alloc_scratch((rows, cols), flat_x.dtype)
    rsqrt_buf = _alloc_scratch((rows,), f32)
    _RMS_NORM_KERNEL(flat_x, flat_w, out_buf, rsqrt_buf, EPS=eps)
    result = out_buf
    out_dtype = weight.dtype
    if out_dtype != result._dtype:
        torch_out = _IR_TO_TORCH.get(out_dtype.ir, torch.float16)
        result = _to_copy(result, dtype=torch_out)
    if len(shape) > 2:
        result = result.reshape(shape)
    rsqrt_shape = shape[:-1] + (1,)
    return result, rsqrt_buf.reshape(rsqrt_shape)


def _fused_rms_norm_backward(
    x: AlloyBuffer, dy: AlloyBuffer, weight: AlloyBuffer, rrms: AlloyBuffer
) -> AlloyBuffer:
    """Fused RMSNorm backward via alloy.std.norm.rms_norm_backward."""
    shape = dy.shape
    cols = shape[-1]
    rows = 1
    for dim in shape[:-1]:
        rows *= dim
    flat_x = x.reshape((rows, cols))
    flat_dy = dy.reshape((rows, cols))
    flat_w = weight.reshape((cols,))
    flat_rrms = rrms.reshape((rows,))
    out_buf = _alloc_aligned((rows, cols), flat_dy.dtype)
    _RMS_NORM_BACKWARD_KERNEL(flat_x, flat_dy, flat_w, flat_rrms, out_buf)
    if len(shape) > 2:
        out_buf = out_buf.reshape(shape)
    return out_buf
