"""GPU strided-to-contiguous copy.

The strided_copy_4d kernel and contiguify_lazy helper live here instead of
in alloy.std/_lazy.py because:
  - alloy.std imports alloy (for @al.kernel) which imports _core -> _kernel -> _lazy
  - _lazy.py needs strided_copy_4d to contiguify non-contiguous buffers
  - Putting the kernel here avoids the cycle: this file only depends on
    _dispatch._buf_utils and _runtime.alloy_buffer (leaf modules).
"""

from __future__ import annotations

import alloy as al
from alloy._dispatch.buf_utils import _alloc_aligned
from alloy._runtime.alloy_buffer import AlloyBuffer


@al.kernel
def strided_copy_4d(
    x,
    out: al.output,
    N: al.constexpr,
    D0: al.constexpr,
    D1: al.constexpr,
    D2: al.constexpr,
    D3: al.constexpr,
    S0: al.constexpr,
    S1: al.constexpr,
    S2: al.constexpr,
    S3: al.constexpr,
    SRC_OFFSET: al.constexpr = 0,
    BLOCK_SIZE: al.constexpr = 1024,
):
    """Copy from strided source to contiguous output (up to 4D).

    Launched 2D: **axis-0 is the row index** over the leading dims (D0·D1·D2),
    axis-1 tiles the innermost dim D3 by BLOCK_SIZE. Putting the row (which, for the
    `(B, S, feat)` / `(B·S, feat)` layouts the DeltaNet + attention paths produce, is
    the M = sequence-position dimension at B=1) on grid axis-0 lets the grid-shrink
    prefill grid recipe shrink the copy to the real prompt length — the recipe only
    ever shrinks axis-0. A flat 1D `ceil(N/BLOCK)` grid would bury
    M inside `offs`, with grid[0] = ceil(M·feat/BLOCK) > M, so the recipe's divisible
    M-tile model couldn't touch it and every layout copy would run at the padded M_MAX.
    Copies whose row count does NOT scale with M (e.g. a `(B, C, S)` transpose, rows
    = B·C) keep their full grid — axis-0 doesn't change with M, so the recipe leaves
    them alone, which is correct (those have M innermost and aren't row-shrinkable).
    """
    row = al.program_id(0)
    cb = al.program_id(1)
    col = cb * BLOCK_SIZE + al.arange(0, BLOCK_SIZE)
    cmask = col < D3
    i2 = row % D2
    tmp = row // D2
    i1 = tmp % D1
    i0 = tmp // D1
    src_idx = SRC_OFFSET + i0 * S0 + i1 * S1 + i2 * S2 + col * S3
    val = al.load(x + src_idx, mask=cmask, other=0.0)
    al.store(out + row * D3 + col, val, mask=cmask)


@al.kernel
def strided_copy_5d(
    x,
    out: al.output,
    N: al.constexpr,
    D0: al.constexpr,
    D1: al.constexpr,
    D2: al.constexpr,
    D3: al.constexpr,
    D4: al.constexpr,
    S0: al.constexpr,
    S1: al.constexpr,
    S2: al.constexpr,
    S3: al.constexpr,
    S4: al.constexpr,
    SRC_OFFSET: al.constexpr = 0,
    BLOCK_SIZE: al.constexpr = 1024,
):
    """Copy from strided source to contiguous output (5D).

    Used when a 5D ``.contiguous()`` call lands on a tensor whose strides
    can't be merged into 4D without losing information (e.g. the
    ``permute(1,3,0,2,4)`` output from QKV-split SDPA-backward plumbing at
    batch > 1, where every adjacent-dim stride product disagrees).
    """
    pid = al.program_id(0)
    offs = pid * BLOCK_SIZE + al.arange(0, BLOCK_SIZE)
    mask = offs < N
    i4 = offs % D4
    t1 = offs // D4
    i3 = t1 % D3
    t2 = t1 // D3
    i2 = t2 % D2
    t3 = t2 // D2
    i1 = t3 % D1
    i0 = t3 // D1
    src_idx = SRC_OFFSET + i0 * S0 + i1 * S1 + i2 * S2 + i3 * S3 + i4 * S4
    val = al.load(x + src_idx, mask=mask, other=0.0)
    al.store(out + offs, val, mask=mask)


def contiguify_lazy(lb: AlloyBuffer) -> AlloyBuffer:
    """GPU-copy a non-contiguous AlloyBuffer to a contiguous one.

    Uses strided_copy_4d. The copy is lazy — it becomes part of the
    consuming kernel's materializer chain, dispatched in the same
    Metal command buffer with no extra sync.
    """
    if lb.is_contiguous() and lb._offset == 0:
        return lb
    itemsize = lb._dtype.itemsize
    shape = lb._shape
    ndim = len(shape)
    strides = tuple(s // itemsize for s in lb._strides)

    if ndim == 0:
        return lb

    # Pick a safe 4D projection: try to merge adjacent dims whose strides
    # are compatible (stride[i] == shape[i+1] * stride[i+1]). If none of
    # the adjacencies can be merged, we need the 5D kernel.
    def _try_merge_to_4d(shape, strides):
        if len(shape) <= 4:
            return shape, strides
        for i in range(len(shape) - 1):
            if strides[i] == shape[i + 1] * strides[i + 1]:
                merged_shape = shape[:i] + (shape[i] * shape[i + 1],) + shape[i + 2 :]
                merged_strides = strides[:i] + (strides[i + 1],) + strides[i + 2 :]
                return _try_merge_to_4d(merged_shape, merged_strides)
        return shape, strides

    if ndim == 5:
        shape, strides = _try_merge_to_4d(shape, strides)
        ndim = len(shape)

    if ndim == 1:
        d = (1, 1, 1, shape[0])
        s = (0, 0, 0, strides[0])
    elif ndim == 2:
        d = (1, 1, shape[0], shape[1])
        s = (0, 0, strides[0], strides[1])
    elif ndim == 3:
        d = (1, shape[0], shape[1], shape[2])
        s = (0, strides[0], strides[1], strides[2])
    elif ndim == 4:
        d = shape
        s = strides
    elif ndim == 5:
        # No mergeable pair — fall through to strided_copy_5d below.
        d = shape
        s = strides
    else:
        raise RuntimeError(f"contiguify_lazy: unsupported ndim={ndim}")

    N = lb.size
    src_offset = lb._offset // itemsize

    root = AlloyBuffer(
        lb._parent_handle,
        lb._offset,
        lb._shape,
        lb._strides,
        lb._dtype,
        raw_ptr=lb._raw_ptr,
        total_nbytes=lb.metal_nbytes,
    )
    root.root_flat()
    lb._view_of(root)

    out = _alloc_aligned(lb._shape, lb._dtype)
    if ndim == 5:
        return strided_copy_5d(
            root,
            out,
            N=N,
            D0=d[0],
            D1=d[1],
            D2=d[2],
            D3=d[3],
            D4=d[4],
            S0=s[0],
            S1=s[1],
            S2=s[2],
            S3=s[3],
            S4=s[4],
            SRC_OFFSET=src_offset,
        )
    # 2D launch: axis-0 = rows over the leading dims, axis-1 = column tiles over
    # the innermost dim (BLOCK_SIZE=1024, matching the kernel default). Axis-0 being
    # the row index is what makes the grid-shrunk chunk prefill recipe able to shrink layout
    # copies whose rows are the M dimension (B=1 → rows = S).
    rows = d[0] * d[1] * d[2]
    col_tiles = (d[3] + 1023) // 1024
    return strided_copy_4d[(rows, col_tiles)](
        root,
        out,
        N=N,
        D0=d[0],
        D1=d[1],
        D2=d[2],
        D3=d[3],
        S0=s[0],
        S1=s[1],
        S2=s[2],
        S3=s[3],
        SRC_OFFSET=src_offset,
    )
