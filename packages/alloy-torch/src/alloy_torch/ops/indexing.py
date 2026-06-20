"""Indexing, gather, scatter, and padding handlers for torch op lowering."""

from collections.abc import Sequence
import math
from typing import TYPE_CHECKING

import alloy
import numpy as np
import torch

from alloy._compiler.dtypes import int32
from alloy._dispatch.buf_utils import _alloc_aligned
from alloy._runtime.alloy_buffer import AlloyBuffer
from alloy_torch.ops.casting import _to_copy, k_copy
from alloy_torch.ops.common import (
    _broadcast_shapes,
    _dtype_of,
    _elem_strides_4d,
    _expand_lazy_buffer,
    _is_bool_dtype,
    _normalize_dim,
    _numel,
    _pad_shape_4d,
    _shape_of,
)
from alloy_torch.ops.concat import _cat
from alloy_torch.ops.creation import _arange_start_step, _full
from alloy_torch.ops.values import _coerce_lazy_value, _coerce_mask_numeric
from alloy_torch.ops.views import _select_int

if TYPE_CHECKING:
    output = AlloyBuffer
    constexpr = int
else:
    output = alloy.output
    constexpr = alloy.constexpr

IndexValue = AlloyBuffer | torch.Tensor | slice | int | None
IndexSequence = tuple[IndexValue, ...] | list[IndexValue]
ScalarIndexValue = AlloyBuffer | torch.Tensor | bool | int


@alloy.kernel
def k_scatter_last_2d(
    base,
    idx,
    out: output,
    ROWS: constexpr,
    COLS: constexpr,
    IDX_COLS: constexpr,
    VALUE: constexpr,
    BLOCK_SIZE: constexpr = 1024,
):
    pid = alloy.program_id(0)
    offs = pid * BLOCK_SIZE + alloy.arange(0, BLOCK_SIZE)
    mask = offs < ROWS * COLS
    row = offs // COLS
    col = offs % COLS
    val = alloy.load(base + offs, mask=mask)
    for k in range(IDX_COLS):
        target = alloy.load(idx + row * IDX_COLS + k, mask=mask)
        val = alloy.where(col == target, VALUE, val)
    alloy.store(out + offs, val, mask=mask)


@alloy.kernel
def k_scatter_add_last_2d(
    base,
    idx,
    src,
    out: output,
    ROWS: constexpr,
    COLS: constexpr,
    IDX_COLS: constexpr,
    BLOCK_SIZE: constexpr = 1024,
):
    pid = alloy.program_id(0)
    offs = pid * BLOCK_SIZE + alloy.arange(0, BLOCK_SIZE)
    mask = offs < ROWS * COLS
    row = offs // COLS
    col = offs % COLS
    val = alloy.load(base + offs, mask=mask)
    for k in range(IDX_COLS):
        target = alloy.load(idx + row * IDX_COLS + k, mask=mask)
        s = alloy.load(src + row * IDX_COLS + k, mask=mask)
        val = alloy.where(col == target, val + s, val)
    alloy.store(out + offs, val, mask=mask)


@alloy.kernel
def k_index_put_add_rows(
    index,
    src,
    out: output,
    NUM_IDX: constexpr,
    COLS: constexpr,
    BLOCK_SIZE: constexpr = 1024,
):
    """Scatter-add `src[i]` into `out[index[i]]` (one program per src element).
    Collisions accumulate via an atomic-add store, so the per-row sum order is
    non-deterministic at f32-ULP. `out` carries the running value and must be
    pre-initialised to the target."""
    pid = alloy.program_id(0)
    offs = pid * BLOCK_SIZE + alloy.arange(0, BLOCK_SIZE)
    mask = offs < NUM_IDX * COLS
    i = offs // COLS
    c = offs % COLS
    row = alloy.load(index + i, mask=mask)
    v = alloy.load(src + offs, mask=mask)
    alloy.store(out + row * COLS + c, v, mask=mask, reduce="add")


@alloy.kernel
def k_gather_2d(
    x,
    idx0,
    idx1,
    out: output,
    N: constexpr,
    COLS: constexpr,
    D0: constexpr = 1,
    D1: constexpr = 1,
    D2: constexpr = 1,
    D3: constexpr = 1,
    I0S0: constexpr = 0,
    I0S1: constexpr = 0,
    I0S2: constexpr = 0,
    I0S3: constexpr = 0,
    I1S0: constexpr = 0,
    I1S1: constexpr = 0,
    I1S2: constexpr = 0,
    I1S3: constexpr = 0,
    BLOCK_SIZE: constexpr = 1024,
):
    pid = alloy.program_id(0)
    offs = pid * BLOCK_SIZE + alloy.arange(0, BLOCK_SIZE)
    mask = offs < N
    rem = offs
    i3 = rem % D3
    rem = rem // D3
    i2 = rem % D2
    rem = rem // D2
    i1 = rem % D1
    i0 = rem // D1
    row = alloy.load(idx0 + i0 * I0S0 + i1 * I0S1 + i2 * I0S2 + i3 * I0S3, mask=mask)
    col = alloy.load(idx1 + i0 * I1S0 + i1 * I1S1 + i2 * I1S2 + i3 * I1S3, mask=mask)
    src = row * COLS + col
    alloy.store(out + offs, alloy.load(x + src, mask=mask), mask=mask)


@alloy.kernel
def k_gather_rows_2d(
    weight,
    indices,
    out: output,
    NUM_INDICES: constexpr,
    WIDTH: constexpr,
    BLOCK_SIZE: constexpr = 1024,
):
    pid = alloy.program_id(0)
    offs = pid * BLOCK_SIZE + alloy.arange(0, BLOCK_SIZE)
    mask = offs < NUM_INDICES * WIDTH
    cols = offs % WIDTH
    rows = offs // WIDTH
    idx = alloy.load(indices + rows, mask=mask)
    alloy.store(out + offs, alloy.load(weight + idx * WIDTH + cols, mask=mask), mask=mask)


@alloy.kernel
def k_index_copy_dim2_4d(
    base,
    index,
    src,
    out: output,
    B: constexpr,
    H: constexpr,
    L: constexpr,
    D: constexpr,
    T: constexpr,
    S_STR0: constexpr,
    S_STR1: constexpr,
    S_STR2: constexpr,
    S_STR3: constexpr,
    SRC_OFF: constexpr = 0,
    BLOCK_SIZE: constexpr = 1024,
):
    pid = alloy.program_id(0)
    offs = pid * BLOCK_SIZE + alloy.arange(0, BLOCK_SIZE)
    mask = offs < B * H * L * D
    rem = offs
    d = rem % D
    rem = rem // D
    seq = rem % L
    rem = rem // L
    h = rem % H
    b = rem // H
    out_val = alloy.load(base + offs, mask=mask)
    for t in range(T):
        pos = alloy.load(index + t)
        write_mask = mask & (seq == pos)
        src_offs = SRC_OFF + b * S_STR0 + h * S_STR1 + t * S_STR2 + d * S_STR3
        src_val = alloy.load(src + src_offs, mask=write_mask)
        out_val = alloy.where(write_mask, src_val, out_val)
    alloy.store(out + offs, out_val, mask=mask)


@alloy.kernel
def k_cache_scatter_dim2_4d(
    cache,
    index,
    src,
    H: constexpr,
    L: constexpr,
    D: constexpr,
    T: constexpr,
    S_STR0: constexpr,
    S_STR1: constexpr,
    S_STR2: constexpr,
    S_STR3: constexpr,
    SRC_OFF: constexpr = 0,
    BLOCK_SIZE: constexpr = 256,
):
    pid = alloy.program_id(0)
    offs = pid * BLOCK_SIZE + alloy.arange(0, BLOCK_SIZE)
    total = T * H * D
    mask = offs < total
    rem = offs
    d = rem % D
    rem = rem // D
    h = rem % H
    t = rem // H
    pos = alloy.load(index + t, mask=mask)
    src_offs = SRC_OFF + h * S_STR1 + t * S_STR2 + d * S_STR3
    val = alloy.load(src + src_offs, mask=mask)
    dst_offs = h * L * D + pos * D + d
    alloy.store(cache + dst_offs, val, mask=mask)


@alloy.kernel
def k_cache_write_arange(
    target,
    index,
    src,
    out: output,
    B: constexpr,
    H: constexpr,
    L: constexpr,
    D: constexpr,
    T: constexpr,
    S_STR0: constexpr,
    S_STR1: constexpr,
    S_STR2: constexpr,
    S_STR3: constexpr,
    SRC_OFF: constexpr = 0,
    BLOCK_SIZE: constexpr = 1024,
):
    """Cache-write specialized for contiguous arange indices: a single-pass
    O(B*H*L*D) kernel (vs k_index_copy_dim2_4d's O(L*T*H*D) inner-loop scan).
    Each output element copies from target (seq index outside the new range) or
    src (inside).

    Assumes `index[i] == index[0] + i` for all i in [0, T) — HF's
    StaticLayer.update builds cache_position as `arange(kv_length) +
    cumulative_length`. For non-contiguous indices, use k_index_copy_dim2_4d.
    """
    pid = alloy.program_id(0)
    offs = pid * BLOCK_SIZE + alloy.arange(0, BLOCK_SIZE)
    mask = offs < B * H * L * D
    rem = offs
    d = rem % D
    rem = rem // D
    seq = rem % L
    rem = rem // L
    h = rem % H
    b = rem // H

    start = alloy.load(index + 0)
    t_for_seq = seq - start
    use_src = mask & (t_for_seq >= 0) & (t_for_seq < T)
    # Clamp t to [0, T) regardless of use_src so the offset never points past
    # src's buffer: Metal's masked load with an OOB pointer is undefined and
    # propagates garbage for small T where the unmasked address exceeds src.size.
    t_clamped = alloy.maximum(
        alloy.minimum(alloy.cast(t_for_seq, alloy.int32), alloy.cast(T - 1, alloy.int32)),
        alloy.cast(0, alloy.int32),
    )

    target_val = alloy.load(target + offs, mask=mask)
    src_offs = SRC_OFF + b * S_STR0 + h * S_STR1 + t_clamped * S_STR2 + d * S_STR3
    src_val = alloy.load(src + src_offs, mask=use_src)

    val = alloy.where(use_src, src_val, target_val)
    alloy.store(out + offs, val, mask=mask)


@alloy.kernel
def k_slice_scatter_1d(
    base,
    src,
    out: output,
    N: constexpr,
    BEFORE: constexpr,
    DIM_SIZE: constexpr,
    AFTER: constexpr,
    START: constexpr,
    END: constexpr,
    STEP: constexpr,
    SRC_DIM_SIZE: constexpr,
    BLOCK_SIZE: constexpr = 1024,
):
    pid = alloy.program_id(0)
    offs = pid * BLOCK_SIZE + alloy.arange(0, BLOCK_SIZE)
    mask = offs < N
    after_idx = offs % AFTER
    tmp = offs // AFTER
    dim_idx = tmp % DIM_SIZE
    before_idx = tmp // DIM_SIZE
    in_range = (dim_idx >= START) & (dim_idx < END)
    rel = dim_idx - START
    on_step = (rel % STEP) == 0
    in_slice = in_range & on_step
    src_dim = rel // STEP
    src_off = before_idx * SRC_DIM_SIZE * AFTER + src_dim * AFTER + after_idx
    base_val = alloy.load(base + offs, mask=mask, other=0.0)
    src_val = alloy.load(src + src_off, mask=mask & in_slice, other=0.0)
    val = alloy.where(in_slice, src_val, base_val)
    alloy.store(out + offs, val, mask=mask)


def _slice_scatter(
    x: AlloyBuffer,
    src: AlloyBuffer,
    dim: int = 0,
    start: int | None = None,
    end: int | None = None,
    step: int = 1,
) -> AlloyBuffer:
    dim = _normalize_dim(dim, x.ndim)
    dim_size = x.shape[dim]
    s = 0 if start is None else int(start)
    e = dim_size if end is None else int(end)
    if s < 0:
        s += dim_size
    if e < 0:
        e += dim_size
    s = max(0, min(s, dim_size))
    e = max(s, min(e, dim_size))
    step = int(step)

    before = math.prod(x.shape[:dim]) if dim > 0 else 1
    after = math.prod(x.shape[dim + 1 :]) if dim + 1 < len(x.shape) else 1
    src_dim_size = (e - s + step - 1) // step

    x_c = x.contiguous()
    src_c = _coerce_lazy_value(src.contiguous(), dtype=x._dtype.to_torch_dtype())
    out = _alloc_aligned(x.shape, x._dtype)
    return k_slice_scatter_1d(
        x_c,
        src_c,
        out,
        N=x.size,
        BEFORE=before,
        DIM_SIZE=dim_size,
        AFTER=after,
        START=s,
        END=e,
        STEP=step,
        SRC_DIM_SIZE=src_dim_size,
    ).reshape(x.shape)


def _select_backward(
    grad_out: AlloyBuffer,
    input_sizes: Sequence[int],
    dim: int,
    index: int,
) -> AlloyBuffer:
    """aten.select_backward: scatter grad into a zero tensor at [..., dim=index, ...].

    AOT emits this to undo ``aten.select.int`` during backward. Concatenate
    ``[zeros_before, grad.unsqueeze(dim), zeros_after]`` along ``dim`` — each
    piece is contiguous and the grad buffer stays within its own allocation. The
    ``eq(arange) + where`` decomposition mis-scatters on small head_dim SDPA
    paths, and ``_slice_scatter`` returns garbage when its out-of-slice ``src``
    offsets go negative.
    """
    sizes: tuple[int, ...] = tuple(int(size) for size in input_sizes)
    ndim = len(sizes)
    dim = _normalize_dim(int(dim), ndim)
    idx = int(index)
    if idx < 0:
        idx += sizes[dim]
    total = sizes[dim]
    src_shape: tuple[int, ...] = sizes[:dim] + (1,) + sizes[dim + 1 :]
    grad_unsq = grad_out.reshape(src_shape).contiguous()
    dtype = grad_out._dtype.to_torch_dtype()

    if total == 1:
        return grad_unsq
    pieces: list[AlloyBuffer] = []
    if idx > 0:
        before: tuple[int, ...] = sizes[:dim] + (idx,) + sizes[dim + 1 :]
        pieces.append(_full(before, 0, dtype=dtype))
    pieces.append(grad_unsq)
    if idx + 1 < total:
        after: tuple[int, ...] = sizes[:dim] + (total - idx - 1,) + sizes[dim + 1 :]
        pieces.append(_full(after, 0, dtype=dtype))
    return _cat(pieces, dim)


def _is_full_slice_index(item: IndexValue) -> bool:
    return item is None or (isinstance(item, slice) and item == slice(None))


def _normalize_cache_write_index(
    target: AlloyBuffer,
    dim: int,
    index: AlloyBuffer,
    source: AlloyBuffer,
) -> tuple[AlloyBuffer, AlloyBuffer, AlloyBuffer, tuple[int, int, int, int], int] | None:
    dim = _normalize_dim(int(dim), target.ndim)

    if dim != 2 or target.ndim != 4 or source.ndim != 4 or index.ndim != 1:
        return None
    if tuple(source.shape[:2]) != tuple(target.shape[:2]):
        return None
    if source.shape[-1] != target.shape[-1]:
        return None
    if source.shape[2] != index.shape[0]:
        return None
    if target._dtype.to_torch_dtype() != source._dtype.to_torch_dtype():
        source = _to_copy(source, dtype=target._dtype.to_torch_dtype())
    itemsize = source._dtype.itemsize
    src_stride_values = tuple(stride // itemsize for stride in source._strides)
    src_strides = (
        src_stride_values[0],
        src_stride_values[1],
        src_stride_values[2],
        src_stride_values[3],
    )
    return target, index, source, src_strides, dim


def _cache_write_dim2_4d(
    target: AlloyBuffer, dim: int, index: AlloyBuffer, source: AlloyBuffer, *, inplace: bool = False
) -> AlloyBuffer:
    normalized = _normalize_cache_write_index(target, dim, index, source)
    if normalized is None:
        raise RuntimeError(
            f"Alloy _cache_write_dim2_4d: unsupported index pattern "
            f"(dim={dim}, target.ndim={target.ndim}, source.ndim={source.ndim}, "
            f"index.ndim={index.ndim if hasattr(index, 'ndim') else '?'})"
        )

    target, index_buf, source, src_strides, _ = normalized
    if tuple(target.shape) == tuple(source.shape):
        return source

    _, heads, length, dim_size = target.shape
    seq_len = index_buf.shape[0]

    if inplace and target.shape[0] == 1:
        k_cache_scatter_dim2_4d(
            target,
            index_buf,
            source,
            H=heads,
            L=length,
            D=dim_size,
            T=seq_len,
            S_STR0=src_strides[0],
            S_STR1=src_strides[1],
            S_STR2=src_strides[2],
            S_STR3=src_strides[3],
            SRC_OFF=0,
        )
        return target

    for check_buf, check_name in ((target, "target"), (source, "source")):
        itemsize = check_buf._dtype.itemsize
        elem_strides = tuple(stride // itemsize for stride in check_buf._strides)
        max_off = sum((size - 1) * abs(stride) for size, stride in zip(check_buf._shape, elem_strides))
        if (max_off + 1) * itemsize > check_buf.nbytes:
            fixed = check_buf.contiguous()
            if check_name == "source":
                source = fixed
                src_strides = tuple(stride // itemsize for stride in source._strides)
            else:
                target = fixed

    idx_itemsize = index_buf._dtype.itemsize
    idx_elem_strides = tuple(stride // idx_itemsize for stride in index_buf._strides)
    idx_max = sum((size - 1) * abs(stride) for size, stride in zip(index_buf._shape, idx_elem_strides))
    if (idx_max + 1) * idx_itemsize > index_buf.nbytes:
        index_buf = index_buf.contiguous()

    out_arr = _alloc_aligned(target.shape, target.dtype)
    B, H, L, D = target.shape
    T = index_buf.shape[0]
    # T > 1 means prefill / multi-token decode where cache_position is a
    # contiguous arange. k_cache_write_arange does a single bandwidth-bound
    # pass over the cache instead of the O(L*T*H*D) inner-loop scan in
    # k_index_copy_dim2_4d. T == 1 is the steady-state decode case — the
    # scan kernel is already fast there.
    if T > 1:
        total = B * H * L * D
        grid = ((total + 1023) // 1024,)
        return k_cache_write_arange[grid](
            target,
            index_buf,
            source,
            out_arr,
            B=B,
            H=H,
            L=L,
            D=D,
            T=T,
            S_STR0=src_strides[0],
            S_STR1=src_strides[1],
            S_STR2=src_strides[2],
            S_STR3=src_strides[3],
            SRC_OFF=0,
        )
    return k_index_copy_dim2_4d(
        target,
        index_buf,
        source,
        out_arr,
        B=B,
        H=H,
        L=L,
        D=D,
        T=T,
        S_STR0=src_strides[0],
        S_STR1=src_strides[1],
        S_STR2=src_strides[2],
        S_STR3=src_strides[3],
        SRC_OFF=0,
    )


def _single_indexed_dim(indices: IndexSequence) -> tuple[int, AlloyBuffer]:
    if not isinstance(indices, (tuple, list)):
        raise NotImplementedError("Alloy index_put expects a tuple/list of indices")
    indexed = [(i, item) for i, item in enumerate(indices) if not _is_full_slice_index(item)]
    if len(indexed) != 1:
        raise NotImplementedError("Alloy index_put currently supports exactly one indexed dimension")
    dim, index = indexed[0]
    if not isinstance(index, AlloyBuffer):
        raise NotImplementedError("Alloy index_put expects AlloyBuffer indices")
    return dim, index


def _index_put_accumulate(
    target: AlloyBuffer, dim: int, index: AlloyBuffer, values: AlloyBuffer, *, inplace: bool
) -> AlloyBuffer:
    """index_put(accumulate=True): scatter-add `values` rows into `target` at
    `index` along the leading dim — the embedding backward (grad_weight[token] +=
    grad_out). Repeated tokens accumulate through an atomic-add store, so the
    per-row sum order is non-deterministic at f32-ULP."""
    if dim != 0:
        raise NotImplementedError(
            "Alloy index_put(accumulate=True) supports indexing the leading dim only"
        )
    rows = target.shape[0]
    cols = math.prod(target.shape[1:]) if target.ndim > 1 else 1
    num_idx = index.size
    index_i32 = _to_copy(index.reshape((num_idx,)), dtype=torch.int32)
    values_2d = values.reshape((num_idx, cols)).contiguous()
    if inplace:
        out = target.reshape((rows, cols))
    else:
        out = _alloc_aligned((rows, cols), target.dtype)
        src = target.reshape((rows, cols)).contiguous()
        k_copy[((rows * cols + 1023) // 1024,)](src, out, N=rows * cols)
    k_index_put_add_rows[((num_idx * cols + 1023) // 1024,)](
        index_i32, values_2d, out, NUM_IDX=num_idx, COLS=cols
    )
    return out.reshape(target.shape)


def _index_put(
    target: AlloyBuffer, indices: IndexSequence, values: AlloyBuffer, accumulate: bool = False
) -> AlloyBuffer:
    if accumulate:
        dim, index = _single_indexed_dim(indices)
        return _index_put_accumulate(target, dim, index, values, inplace=False)
    if not isinstance(indices, (tuple, list)):
        raise NotImplementedError("Alloy index_put expects a tuple/list of indices")

    indexed_dims = [(idx, item) for idx, item in enumerate(indices) if not _is_full_slice_index(item)]
    if len(indexed_dims) != 1:
        raise NotImplementedError(
            "Alloy index_put currently supports exactly one indexed dimension"
        )
    dim, index = indexed_dims[0]
    if not isinstance(index, AlloyBuffer):
        raise NotImplementedError("Alloy index_put expects AlloyBuffer indices")
    if any(not _is_full_slice_index(item) for idx, item in enumerate(indices) if idx != dim):
        raise NotImplementedError(
            "Alloy index_put only supports full slices on non-indexed dimensions"
        )
    return _cache_write_dim2_4d(target, dim, index, values)


def _index_put_inplace(
    target: AlloyBuffer, indices: IndexSequence, values: AlloyBuffer, accumulate: bool = False
) -> AlloyBuffer:
    if accumulate:
        dim, index = _single_indexed_dim(indices)
        return _index_put_accumulate(target, dim, index, values, inplace=True)
    if not isinstance(indices, (tuple, list)):
        raise NotImplementedError("Alloy index_put expects a tuple/list of indices")
    indexed_dims = [(idx, item) for idx, item in enumerate(indices) if not _is_full_slice_index(item)]
    if len(indexed_dims) != 1:
        raise NotImplementedError(
            "Alloy index_put currently supports exactly one indexed dimension"
        )
    dim, index = indexed_dims[0]
    if not isinstance(index, AlloyBuffer):
        raise NotImplementedError("Alloy index_put expects AlloyBuffer indices")
    return _cache_write_dim2_4d(target, dim, index, values, inplace=True)


def _extract_static_scalar_int(value: ScalarIndexValue) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return int(value.detach().cpu().item())
        return None
    if value._materializer is None and value.size == 1:
        return int(value.read_scalar())
    producer = value._producer
    if producer is None:
        return None
    inputs = [(name, arr) for name, arr in producer.buffer_args if name not in producer.output_params]
    if len(inputs) != 1:
        return None
    _, arr = inputs[0]
    try:
        return int(arr.reshape(()).item())
    except Exception:
        return None


def _array_is_identity_range(arr: np.ndarray, width: int) -> bool:
    if arr.size != width:
        return False
    try:
        flat = arr.reshape(width)
    except Exception:
        return False
    return bool((flat == np.arange(width, dtype=flat.dtype)).all())


def _is_identity_column_index(value: AlloyBuffer | torch.Tensor, width: int) -> bool:
    if isinstance(value, torch.Tensor):
        return _array_is_identity_range(value.detach().cpu().numpy(), width)
    if value._materializer is None:
        return _array_is_identity_range(value.numpy, width)
    producer = value._producer
    if producer is None:
        return False
    if producer.kernel.name not in {"k_add_scalar", "add_scalar"}:
        return False
    base = None
    scalar = None
    for name, arr in producer.buffer_args:
        if name in producer.output_params:
            continue
        shape = producer.buffer_shapes[name] if name in producer.buffer_shapes else arr.shape
        if _numel(tuple(shape) if isinstance(shape, tuple) else tuple(shape)) == 1:
            try:
                scalar = int(arr.reshape(()).item())
            except Exception:
                return False
        else:
            base = arr
    return scalar == 0 and base is not None and _array_is_identity_range(base, width)


def _free_2d_identity_index(x: AlloyBuffer, indices: IndexSequence) -> AlloyBuffer | None:
    if not isinstance(indices, (tuple, list)) or len(indices) != 2:
        return None
    if any(index is None or isinstance(index, slice) and index == slice(None) for index in indices):
        return None

    if x.ndim != 2:
        return None

    row_idx, col_idx = indices
    if row_idx is None or isinstance(row_idx, slice):
        return None
    if not isinstance(col_idx, (AlloyBuffer, torch.Tensor)):
        return None
    out_shape = tuple(_broadcast_shapes(_shape_of(row_idx), _shape_of(col_idx)))
    if _numel(out_shape) != x.shape[1]:
        return None

    row = _extract_static_scalar_int(row_idx)
    if row is None or row < 0 or row >= x.shape[0]:
        return None
    if not _is_identity_column_index(col_idx, x.shape[1]):
        return None

    selected = _select_int(x, 0, row)
    reshaped = selected.reshape(out_shape)
    if _is_bool_dtype(_dtype_of(x)):
        return _coerce_mask_numeric(reshaped)
    return reshaped


def _index_tensor(x: AlloyBuffer, indices: IndexSequence) -> AlloyBuffer:
    free = _free_2d_identity_index(x, indices)
    if free is not None:
        return free

    if x.ndim == 2 and len(indices) == 1 and indices[0] is not None:
        idx = indices[0]
        if not isinstance(idx, AlloyBuffer):
            raise RuntimeError("Alloy index.Tensor expects AlloyBuffer indices")
        idx = _coerce_lazy_value(idx, dtype=torch.int64)
        idx_shape = idx.shape
        out = k_gather_rows_2d(x, idx, NUM_INDICES=_numel(idx_shape), WIDTH=x.shape[1])
        return out.reshape(idx_shape + (x.shape[1],))

    if x.ndim == 2 and len(indices) == 2:
        idx0 = indices[0]
        idx1 = indices[1]
        if not isinstance(idx0, AlloyBuffer) or not isinstance(idx1, AlloyBuffer):
            raise RuntimeError("Alloy 2D index.Tensor expects AlloyBuffer indices")
        out_shape = _broadcast_shapes(idx0.shape, idx1.shape)
        if idx0.shape != out_shape:
            idx0 = _expand_lazy_buffer(idx0, out_shape)
        if idx1.shape != out_shape:
            idx1 = _expand_lazy_buffer(idx1, out_shape)
        n = math.prod(out_shape)
        rows, cols = x.shape
        flat_x = x.reshape((rows * cols,))
        es0 = _elem_strides_4d(idx0)
        es1 = _elem_strides_4d(idx1)
        dims = _pad_shape_4d(out_shape)
        out = k_gather_2d(
            flat_x,
            idx0,
            idx1,
            N=n,
            COLS=cols,
            D0=dims[0],
            D1=dims[1],
            D2=dims[2],
            D3=dims[3],
            I0S0=es0[0],
            I0S1=es0[1],
            I0S2=es0[2],
            I0S3=es0[3],
            I1S0=es1[0],
            I1S1=es1[1],
            I1S2=es1[2],
            I1S3=es1[3],
        )
        result = out.reshape(out_shape)
        if _is_bool_dtype(_dtype_of(x)):
            return _coerce_mask_numeric(result)
        return result

    if (
        x.ndim == 3
        and len(indices) >= 2
        and indices[0] is None
        and isinstance(indices[1], AlloyBuffer)
        and (len(indices) == 2 or indices[2] is None)
    ):
        # x[:, idx, :] — gather along dim 1. Used by HF causal LMs with
        # logits_to_keep=tensor to slice hidden_states down to specific
        # positions before lm_head.
        idx = _coerce_lazy_value(indices[1], dtype=torch.int64)
        idx_shape = idx.shape
        batch, seq_len, hidden = x.shape
        if batch != 1:
            raise RuntimeError(
                f"_index_tensor 3D [:, idx, :] only supports batch=1, got batch={batch}"
            )
        flat = x.reshape((seq_len, hidden))
        gathered = k_gather_rows_2d(flat, idx, NUM_INDICES=_numel(idx_shape), WIDTH=hidden)
        return gathered.reshape((batch,) + idx_shape + (hidden,))

    if x.ndim == 4 and len(indices) == 4 and indices[0] is None and indices[1] is None:
        idx_h, idx_w = indices[2], indices[3]
        if not isinstance(idx_h, AlloyBuffer) or not isinstance(idx_w, AlloyBuffer):
            raise RuntimeError("Alloy 4D spatial index.Tensor expects AlloyBuffer indices")
        batch, channels, height, width = x.shape
        out_shape = _broadcast_shapes(idx_h.shape, idx_w.shape)
        if idx_h.shape != out_shape:
            idx_h = _expand_lazy_buffer(idx_h, out_shape)
        if idx_w.shape != out_shape:
            idx_w = _expand_lazy_buffer(idx_w, out_shape)

        idx_h_i32 = _to_copy(idx_h, dtype=torch.int32)
        idx_w_i32 = _to_copy(idx_w, dtype=torch.int32)
        flat_idx = idx_h_i32 * width + idx_w_i32.reshape((-1,))

        x_flat = x.reshape((batch * channels * height * width,))
        out_n = flat_idx.shape[0]
        n_rows = batch * channels
        hw = height * width

        row_offsets = _arange_start_step(0, n_rows, 1, dtype=torch.int32) * hw
        flat_col_i32 = (
            _to_copy(flat_idx, dtype=torch.int32) if flat_idx._dtype.ir != "i32" else flat_idx
        )
        flat_col = flat_col_i32.reshape((1, out_n))
        src_idx = row_offsets + flat_col
        src_idx_flat = src_idx.reshape((n_rows * out_n,))
        total_out = n_rows * out_n
        result = k_gather_rows_2d(x_flat, src_idx_flat, NUM_INDICES=total_out, WIDTH=1)
        return result.reshape((batch, channels) + tuple(out_shape))

    idx_info = [str(index.shape) if isinstance(index, (AlloyBuffer, torch.Tensor)) else "None" for index in indices]
    raise RuntimeError(
        f"No GPU output for _index_tensor: x.shape={x.shape} x.ndim={x.ndim} "
        f"n_indices={len(indices)} idx_shapes={idx_info}"
    )


@alloy.kernel
def k_gather(
    src,
    idx,
    out: output,
    N: constexpr,
    INNER: constexpr,
    DIM_SIZE: constexpr,
    IDX_DIM: constexpr,
    BLOCK_SIZE: constexpr = 1024,
):
    pid = alloy.program_id(0)
    offs = pid * BLOCK_SIZE + alloy.arange(0, BLOCK_SIZE)
    mask = offs < N
    inner_idx = offs % INNER
    rem = offs // INNER
    outer_idx = rem // IDX_DIM
    gather_idx = alloy.load(idx + offs, mask=mask)
    src_offs = outer_idx * (DIM_SIZE * INNER) + gather_idx * INNER + inner_idx
    alloy.store(out + offs, alloy.load(src + src_offs, mask=mask), mask=mask)


def _gather(x: AlloyBuffer, dim: int, index: AlloyBuffer) -> AlloyBuffer:
    shape = x.shape
    idx_shape = index.shape
    dim = _normalize_dim(dim, len(shape))
    inner = math.prod(shape[dim + 1 :])
    out = k_gather(x, index, N=index.size, INNER=inner, DIM_SIZE=shape[dim], IDX_DIM=idx_shape[dim])
    return out.reshape(idx_shape)


def _scatter_value(
    base: AlloyBuffer,
    dim: int,
    index: AlloyBuffer,
    value: float | int,
) -> AlloyBuffer:
    dim = _normalize_dim(int(dim), base.ndim)
    if base.ndim != 2 or dim != 1:
        raise NotImplementedError(
            f"Alloy scatter.value supports 2D tensors with dim=-1, got ndim={base.ndim} dim={dim}"
        )
    if index.ndim != 2 or index.shape[0] != base.shape[0]:
        raise NotImplementedError(
            f"Alloy scatter.value expects 2D index with matching rows, got {index.shape}"
        )
    rows, cols = base.shape
    idx_cols = index.shape[1]
    out = _alloc_aligned(base.shape, base._dtype)
    base_c = base if base.is_contiguous() else _to_copy(base, dtype=base._dtype.to_torch_dtype())
    idx_c = index.contiguous()
    if idx_c._dtype.ir != "i32":
        narrow = _alloc_aligned(idx_c.shape, int32)
        narrow.numpy[:] = idx_c.numpy.astype("int32")
        idx_c = narrow
    return k_scatter_last_2d(
        base_c, idx_c, out, ROWS=rows, COLS=cols, IDX_COLS=idx_cols, VALUE=float(value)
    )


def _scatter_add(
    base: AlloyBuffer,
    dim: int,
    index: AlloyBuffer,
    src: AlloyBuffer,
) -> AlloyBuffer:
    dim = _normalize_dim(int(dim), base.ndim)
    if base.ndim != 2 or dim != 1:
        raise NotImplementedError(
            f"Alloy scatter_add supports 2D tensors with dim=-1, got ndim={base.ndim} dim={dim}"
        )
    if index.ndim != 2 or src.ndim != 2:
        raise NotImplementedError(
            f"Alloy scatter_add expects 2D index/src, got index={index.shape} src={src.shape}"
        )
    if index.shape != src.shape or index.shape[0] != base.shape[0]:
        raise NotImplementedError(
            f"Alloy scatter_add shape mismatch base={base.shape} index={index.shape} src={src.shape}"
        )
    rows, cols = base.shape
    idx_cols = index.shape[1]
    out = _alloc_aligned(base.shape, base._dtype)
    base_c = base if base.is_contiguous() else _to_copy(base, dtype=base._dtype.to_torch_dtype())
    src_c = src if src.is_contiguous() else _to_copy(src, dtype=src._dtype.to_torch_dtype())
    idx_c = index.contiguous()
    if idx_c._dtype.ir != "i32":
        narrow = _alloc_aligned(idx_c.shape, int32)
        narrow.numpy[:] = idx_c.numpy.astype("int32")
        idx_c = narrow
    return k_scatter_add_last_2d(
        base_c, idx_c, src_c, out, ROWS=rows, COLS=cols, IDX_COLS=idx_cols
    )


def _embedding(weight: AlloyBuffer, indices: AlloyBuffer, *_, **__) -> AlloyBuffer:
    if weight.ndim != 2:
        raise NotImplementedError(f"aten.embedding requires rank-2 weight, got {weight.shape}")
    index_buf = _coerce_lazy_value(indices, dtype=torch.int64)
    idx_shape = index_buf.shape
    out = k_gather_rows_2d(weight, index_buf, NUM_INDICES=_numel(idx_shape), WIDTH=weight.shape[1])
    return out.reshape(idx_shape + (weight.shape[1],))


def _constant_pad_nd(x: AlloyBuffer, pad: list[int], value: float = 0.0) -> AlloyBuffer:
    """`aten.constant_pad_nd` — pad (or, for negative pad, crop) each trailing
    dim by a constant `value`."""
    ndim = x.ndim
    if all(p == 0 for p in pad):
        return x
    out = x
    torch_dtype = x._dtype.to_torch_dtype()
    n_pad_dims = len(pad) // 2
    for i in range(n_pad_dims):
        left, right = int(pad[2 * i]), int(pad[2 * i + 1])
        if left == 0 and right == 0:
            continue
        dim = ndim - 1 - i
        if left < 0 or right < 0:  # negative pad crops that side
            out = out.slice(dim, max(0, -left), out.shape[dim] - max(0, -right))
        parts: list[AlloyBuffer] = []
        if left > 0:
            shp = list(out.shape)
            shp[dim] = left
            parts.append(_full(tuple(shp), value, dtype=torch_dtype))
        parts.append(out)
        if right > 0:
            shp = list(out.shape)
            shp[dim] = right
            parts.append(_full(tuple(shp), value, dtype=torch_dtype))
        if len(parts) > 1:
            out = _cat(parts, dim)
    return out
