"""Reduction handlers for torch op lowering."""

from collections.abc import Sequence
import math
from typing import cast

import numpy as np
import torch

from alloy._compiler.dtypes import float32, from_torch_dtype, int64
from alloy._dispatch.buf_utils import _alloc_aligned
from alloy._dispatch.kernel import KernelFunction
from alloy._runtime.alloy_buffer import AlloyBuffer
from alloy._runtime.buffer_ops import _compare_nd, k_ne_nd
from alloy.std.reductions import (
    argmax_last_dim,
    cross_entropy_fused_bwd,
    cross_entropy_fused_fwd,
    mean,
    reduce_any,
    reduce_max,
    reduce_sum,
    softmax,
)
from alloy_torch.ops.common import _normalize_dim
from alloy_torch.ops.casting import _to_copy
from alloy_torch.ops.linalg import _mm

ReductionDim = int | Sequence[int] | None

_mean = cast(KernelFunction, mean)
_cross_entropy_fwd = cast(KernelFunction, cross_entropy_fused_fwd)
_cross_entropy_bwd = cast(KernelFunction, cross_entropy_fused_bwd)
_argmax_last_dim = cast(KernelFunction, argmax_last_dim)
_reduce_any = cast(KernelFunction, reduce_any)
_reduce_sum = cast(KernelFunction, reduce_sum)
_reduce_max = cast(KernelFunction, reduce_max)
_softmax_kernel = cast(KernelFunction, softmax)
_cumsum_tril_cache: dict[tuple[int, str], AlloyBuffer] = {}


def _softmax(x: AlloyBuffer, dim: int = -1, half_to_float: bool = False) -> AlloyBuffer:
    if half_to_float:
        raise NotImplementedError("aten._softmax(..., half_to_float=True) is not supported yet")

    if x.ndim == 0:
        return x

    dim = _normalize_dim(dim, x.ndim)
    if dim != x.ndim - 1:
        axes = list(range(x.ndim))
        axes[dim], axes[-1] = axes[-1], axes[dim]
        transposed = x.transpose(*axes)
        out = _softmax(transposed, dim=-1)
        inverse = [0] * len(axes)
        for i, axis in enumerate(axes):
            inverse[axis] = i
        return out.transpose(*inverse)

    if x.ndim == 1:
        out = _softmax_kernel(x.reshape((1, x.shape[0])))
        return out.reshape(x.shape)

    flat_rows = math.prod(x.shape[:-1])
    cols = x.shape[-1]
    out = _softmax_kernel(x.reshape((flat_rows, cols)))
    return out.reshape(x.shape)


def _argmax(x: AlloyBuffer, dim: int | None = None, keepdim: bool = False) -> AlloyBuffer:
    if x.ndim == 0:
        out = _alloc_aligned((1,), int64)
        out.write_scalar(0)
        return out.reshape((1,)) if keepdim else out.reshape(())

    if dim is None:
        flat = x.reshape((1, x.size))
        out = _alloc_aligned((1,), int64)
        result = _argmax_last_dim(flat, out)
        return result.reshape((1,)) if keepdim else result.reshape(())

    dim = _normalize_dim(dim, x.ndim)
    if dim != x.ndim - 1:
        raise NotImplementedError("aten.argmax is only supported over the last dimension")

    outer = math.prod(x.shape[:-1]) if x.ndim > 1 else 1
    flat = x.reshape((outer, x.shape[-1]))
    out = _alloc_aligned((outer,), int64)
    result = _argmax_last_dim(flat, out)
    if keepdim:
        return result.reshape(x.shape[:-1] + (1,))
    return result.reshape(x.shape[:-1] if x.ndim > 1 else ())


def _mean_dim(
    x: AlloyBuffer,
    dim: ReductionDim,
    keepdim: bool = False,
    *,
    dtype: torch.dtype | None = None,
) -> AlloyBuffer:
    del dtype
    shape = x.shape
    ndim = len(shape)
    if dim is None:
        flat = x.reshape((x.size,))
        return _mean(flat)
    if isinstance(dim, Sequence) and not isinstance(dim, str):
        if len(dim) != 1:
            raise RuntimeError(f"Alloy mean: multi-dim reduction {dim} not supported on GPU")
        dim = dim[0]
    dim = _normalize_dim(int(dim), ndim)
    dim_size = shape[dim]
    outer = 1
    for i in range(ndim):
        if i != dim:
            outer *= shape[i]
    if dim == ndim - 1:
        flat = x.reshape((outer, dim_size))
    else:
        axes = [i for i in range(ndim) if i != dim] + [dim]
        flat = x.transpose(*axes).reshape((outer, dim_size))
    reduced = _mean(flat, axis=1)
    out_shape = list(shape)
    if keepdim:
        out_shape[dim] = 1
    else:
        out_shape.pop(dim)
    return reduced.reshape(tuple(out_shape) if out_shape else (1,))


def _var_correction(
    x: AlloyBuffer,
    dim: ReductionDim = None,
    *,
    correction: int | float | None = None,
    keepdim: bool = False,
) -> AlloyBuffer:
    if correction is None:
        correction = 1
    shape = x.shape
    ndim = len(shape)
    if dim is None:
        reduce_dims = tuple(range(ndim))
    elif isinstance(dim, Sequence) and not isinstance(dim, str):
        reduce_dims = tuple(int(d) for d in dim)
    else:
        reduce_dims = (int(dim),)

    if len(reduce_dims) != 1:
        raise RuntimeError(f"Alloy var: multi-dim reduction {reduce_dims} not supported on GPU")
    reduce_dim = _normalize_dim(reduce_dims[0], ndim)
    reduced_count = shape[reduce_dim]
    denom = float(reduced_count - correction)
    if denom <= 0:
        raise RuntimeError(
            f"Alloy var: correction={correction} leaves non-positive denominator ({denom})"
        )

    ex = _mean_dim(x, reduce_dim, keepdim=keepdim)
    ex2 = _mean_dim(x * x, reduce_dim, keepdim=keepdim)
    biased = ex2 - ex * ex
    if abs(reduced_count - denom) < 1e-12:
        return biased
    return biased * (reduced_count / denom)


def _sum_dim(
    x: AlloyBuffer,
    dim: ReductionDim,
    keepdim: bool = False,
    *,
    dtype: torch.dtype | None = None,
) -> AlloyBuffer:
    del dtype
    shape = x.shape
    if dim is None or (isinstance(dim, Sequence) and not isinstance(dim, str) and len(dim) == 0):
        flat = x.reshape(1, x.size)
        result = _reduce_sum(flat)
        if not keepdim:
            return result.reshape(())
        return result.reshape(tuple(1 for _ in shape))
    if isinstance(dim, Sequence) and not isinstance(dim, str):
        dims = sorted((_normalize_dim(int(d), len(shape)) for d in dim), reverse=True)
    else:
        dims = [_normalize_dim(int(dim), len(shape))]
    result = x
    for reduce_dim in dims:
        current_shape = result.shape
        ndim = len(current_shape)
        reduced_size = current_shape[reduce_dim]
        if reduced_size == 1:
            if keepdim:
                new_shape = current_shape
            else:
                new_shape = current_shape[:reduce_dim] + current_shape[reduce_dim + 1 :]
            result = result.reshape(new_shape) if new_shape else result
            continue
        remaining = math.prod(current_shape[:reduce_dim]) * math.prod(
            current_shape[reduce_dim + 1 :]
        )
        if reduce_dim != ndim - 1:
            perm = list(range(ndim))
            perm.append(perm.pop(reduce_dim))
            result = result.transpose(*perm)
        flat = result.reshape((remaining, reduced_size))
        summed = _reduce_sum(flat, axis=1)
        if keepdim:
            new_shape = current_shape[:reduce_dim] + (1,) + current_shape[reduce_dim + 1 :]
        else:
            new_shape = current_shape[:reduce_dim] + current_shape[reduce_dim + 1 :]
        result = summed.reshape(new_shape) if new_shape else summed
    return result


def _amax_dim(x: AlloyBuffer, dim: ReductionDim = None, keepdim: bool = False) -> AlloyBuffer:
    """aten.amax — max-reduce over dim(s). Mirrors _sum_dim with reduce_max."""
    shape = x.shape
    if dim is None or (isinstance(dim, Sequence) and not isinstance(dim, str) and len(dim) == 0):
        flat = x.reshape(1, x.size)
        result = _reduce_max(flat)
        return result.reshape(tuple(1 for _ in shape)) if keepdim else result.reshape(())
    if isinstance(dim, Sequence) and not isinstance(dim, str):
        dims = sorted((_normalize_dim(int(d), len(shape)) for d in dim), reverse=True)
    else:
        dims = [_normalize_dim(int(dim), len(shape))]
    result = x
    for reduce_dim in dims:
        current_shape = result.shape
        ndim = len(current_shape)
        reduced_size = current_shape[reduce_dim]
        if reduced_size == 1:
            new_shape = current_shape if keepdim else current_shape[:reduce_dim] + current_shape[reduce_dim + 1 :]
            result = result.reshape(new_shape) if new_shape else result
            continue
        remaining = math.prod(current_shape[:reduce_dim]) * math.prod(current_shape[reduce_dim + 1 :])
        if reduce_dim != ndim - 1:
            perm = list(range(ndim))
            perm.append(perm.pop(reduce_dim))
            result = result.transpose(*perm)
        flat = result.reshape((remaining, reduced_size))
        reduced = _reduce_max(flat, axis=1)
        if keepdim:
            new_shape = current_shape[:reduce_dim] + (1,) + current_shape[reduce_dim + 1 :]
        else:
            new_shape = current_shape[:reduce_dim] + current_shape[reduce_dim + 1 :]
        result = reduced.reshape(new_shape) if new_shape else reduced
    return result


def _any_dim(x: AlloyBuffer, dim: int, keepdim: bool = False) -> AlloyBuffer:
    shape = x.shape
    ndim = len(shape)
    dim = _normalize_dim(dim, ndim)
    dim_size = shape[dim]
    outer = 1
    for i in range(ndim):
        if i != dim:
            outer *= shape[i]
    if dim == ndim - 1:
        flat = x.reshape((outer, dim_size))
    else:
        axes = [i for i in range(ndim) if i != dim] + [dim]
        flat = x.transpose(*axes).reshape((outer, dim_size))
    reduced = _reduce_any(flat, axis=1)
    out_shape = list(shape)
    if keepdim:
        out_shape[dim] = 1
    else:
        out_shape.pop(dim)
    return reduced.reshape(tuple(out_shape))


def _log_softmax(x: AlloyBuffer, dim: int = -1, half_to_float: bool = False) -> AlloyBuffer:
    """log_softmax = log(softmax(x)). Numerically: x - logsumexp(x, dim)."""
    return _softmax(x, dim=dim, half_to_float=half_to_float).log()


def _cumsum(x: AlloyBuffer, dim: int, *, dtype: torch.dtype | None = None) -> AlloyBuffer:
    """Cumulative sum via upper-triangular matmul."""
    shape = x.shape
    ndim = len(shape)
    dim = _normalize_dim(int(dim), ndim)
    cols = shape[dim]

    outer = math.prod(shape[:dim])
    inner = math.prod(shape[dim + 1 :])
    rows = outer * inner

    if dim == ndim - 1:
        flat = x.reshape(rows, cols)
    else:
        perm = list(range(ndim))
        perm.append(perm.pop(dim))
        flat = x.transpose(*perm).reshape(rows, cols)

    src_torch_dtype = x._dtype.to_torch_dtype()
    if src_torch_dtype != torch.float32:
        flat = _to_copy(flat, dtype=torch.float32)

    cache_key = (cols, "f32")
    if cache_key not in _cumsum_tril_cache:
        triu_np = np.triu(np.ones((cols, cols), dtype=np.float32))
        triu_buf = _alloc_aligned((cols, cols), float32)
        triu_buf.numpy[:] = triu_np
        _cumsum_tril_cache[cache_key] = triu_buf
    tril_buf = _cumsum_tril_cache[cache_key]

    result = _mm(flat, tril_buf)

    out_torch_dtype = (
        dtype
        if dtype is not None
        else (torch.int64 if src_torch_dtype == torch.bool else src_torch_dtype)
    )
    if out_torch_dtype != torch.float32:
        result = _to_copy(result, dtype=out_torch_dtype)

    if dim == ndim - 1:
        return result.reshape(shape[:dim] + (cols,) + shape[dim + 1 :])

    result = result.reshape(shape[:dim] + shape[dim + 1 :] + (cols,))
    inverse_perm = list(range(ndim))
    inverse_perm.insert(dim, inverse_perm.pop(-1))
    return result.transpose(*inverse_perm)


def _alloy_cross_entropy_fwd_fused_handler(
    logits: AlloyBuffer, labels: AlloyBuffer, ignore_index: int
) -> tuple[AlloyBuffer, AlloyBuffer, AlloyBuffer]:
    f32 = from_torch_dtype(torch.float32)
    if logits._dtype != f32:
        logits = _to_copy(logits, dtype=torch.float32)
    vocab = int(logits.shape[-1])
    rows = 1
    for dim in logits.shape[:-1]:
        rows *= int(dim)
    logits_2d = logits.reshape((rows, vocab))
    labels_1d = labels.reshape((rows,))
    if labels_1d._dtype != from_torch_dtype(torch.int32):
        labels_1d = _to_copy(labels_1d, dtype=torch.int32)

    per_row = _alloc_aligned((rows,), f32)
    lse = _alloc_aligned((rows,), f32)
    _cross_entropy_fwd(
        logits_2d,
        labels_1d,
        per_row,
        lse,
        IGNORE_INDEX=int(ignore_index),
        BLOCK_SIZE=1024,
    )
    valid_mask = _compare_nd(k_ne_nd, labels_1d, ignore_index)
    n_valid = valid_mask.to(from_torch_dtype(torch.float32)).sum()
    loss = per_row.sum() / n_valid
    return loss, lse, n_valid


def _alloy_cross_entropy_bwd_fused_handler(
    logits: AlloyBuffer,
    labels: AlloyBuffer,
    lse: AlloyBuffer,
    n_valid: AlloyBuffer,
    grad_loss: AlloyBuffer,
    ignore_index: int,
) -> AlloyBuffer:
    f32 = from_torch_dtype(torch.float32)
    orig_dtype = logits._dtype
    if logits._dtype != f32:
        logits = _to_copy(logits, dtype=torch.float32)
    orig_shape = logits.shape
    vocab = int(logits.shape[-1])
    rows = 1
    for dim in logits.shape[:-1]:
        rows *= int(dim)
    logits_2d = logits.reshape((rows, vocab))
    labels_1d = labels.reshape((rows,))
    if labels_1d._dtype != from_torch_dtype(torch.int32):
        labels_1d = _to_copy(labels_1d, dtype=torch.int32)

    grad_loss_flat = grad_loss.reshape((1,)) if len(grad_loss.shape) == 0 else grad_loss
    n_valid_flat = n_valid.reshape((1,)) if len(n_valid.shape) == 0 else n_valid
    grad_scale = (grad_loss_flat / n_valid_flat).reshape((1,))

    d_logits = _alloc_aligned((rows, vocab), f32)
    _cross_entropy_bwd(
        logits_2d,
        labels_1d,
        lse,
        grad_scale,
        d_logits,
        IGNORE_INDEX=int(ignore_index),
        BLOCK_SIZE=1024,
    )
    d_out = d_logits.reshape(orig_shape)
    if orig_dtype != f32:
        d_out = _to_copy(d_out, dtype=orig_dtype.to_torch_dtype())
    return d_out
