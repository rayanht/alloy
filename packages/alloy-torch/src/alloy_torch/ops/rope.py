"""RoPE custom-op handlers for torch op lowering."""

from __future__ import annotations

from typing import cast

from alloy._compiler.dtypes import float32
from alloy._dispatch.buf_utils import _alloc_aligned, _alloc_scratch
from alloy._dispatch.kernel import KernelFunction
from alloy._runtime.alloy_buffer import AlloyBuffer
from alloy.std.rope import (
    rms_norm_rope_strided,
    rope_apply,
    rope_apply_backward,
    rope_apply_backward_strided,
    rope_apply_strided,
    rope_cos_sin,
)
from alloy_torch.ops.common import _root_flat_buf
from alloy_torch.ops.concat import _cat
from alloy_torch.ops.norms import _fused_rms_norm

_ROPE_APPLY_KERNEL = cast(KernelFunction, rope_apply)
_ROPE_APPLY_STRIDED_KERNEL = cast(KernelFunction, rope_apply_strided)
_RMS_NORM_ROPE_STRIDED_KERNEL = cast(KernelFunction, rms_norm_rope_strided)
_ROPE_APPLY_BACKWARD_KERNEL = cast(KernelFunction, rope_apply_backward)
_ROPE_APPLY_BACKWARD_STRIDED_KERNEL = cast(KernelFunction, rope_apply_backward_strided)
_ROPE_COS_SIN_KERNEL = cast(KernelFunction, rope_cos_sin)


def _rope_table(
    cache_position: AlloyBuffer, inv_freq: AlloyBuffer, seq_len: int
) -> tuple[AlloyBuffer, AlloyBuffer]:
    """Single-kernel rotary cos/sin table: collapses HF's
    (arange+cache_position)->cast->·inv_freq->cos/sin into one dispatch. cos/sin
    are (1, seq_len, HALF_D), shared across all layers' rope."""
    half_d = inv_freq.shape[-1]
    cp = cache_position.reshape((cache_position.size,))
    inv_flat = inv_freq.reshape((half_d,))
    cos = _alloc_scratch((1, seq_len, half_d), float32)
    sin = _alloc_scratch((1, seq_len, half_d), float32)
    _ROPE_COS_SIN_KERNEL[(seq_len,)](cp, inv_flat, cos, sin, HALF_D=half_d)
    return cos, sin


def _fused_rms_norm_rope(
    x: AlloyBuffer, weight: AlloyBuffer, cos: AlloyBuffer, sin: AlloyBuffer, eps: float,
    cos_half: bool = False,
) -> AlloyBuffer:
    """Fused rms_norm + rope_apply for per-head Q/K norm before rotary. Non-canonical
    rope layouts (vision's (B,S,H,D)) are permuted to canonical, run, permuted back —
    the rms_norm normalizes the head_dim (last axis), which the permute preserves.

    cos_half: cos/sin are stored at half the rotary width (the rotate_half
    self-cat duplicate dropped by the rope self-cat strip rewrite). The true
    rotary span is 2x the table width, and the kernel reads the table at half
    stride with the second half re-reading the first."""
    perm = _rope_canonical_perm(x.shape, cos.shape)
    if perm is not None:
        out = _fused_rms_norm_rope_canonical(
            x.permute(perm).contiguous(), weight,
            cos.permute(perm).contiguous(), sin.permute(perm).contiguous(), eps, cos_half,
        )
        return out.permute(_invert_perm(perm))
    return _fused_rms_norm_rope_canonical(x, weight, cos, sin, eps, cos_half)


def _fused_rms_norm_rope_canonical(
    x: AlloyBuffer, weight: AlloyBuffer, cos: AlloyBuffer, sin: AlloyBuffer, eps: float,
    cos_half: bool = False,
) -> AlloyBuffer:
    """Fused rms_norm + rope_apply for per-head Q/K norm before rotary.

    Falls back to the unfused composition when the input is contiguous (the
    strided dispatch is what enables the per-head reduction; for contiguous
    inputs the rms_norm kernel runs row-by-row anyway).
    """
    shape = x.shape
    # rotary_dim is the cos/sin table width; < head_dim means partial rotary
    # (Qwen3.5: rotate the leading 64 of 256, pass the rest through). Partial
    # rotary only arises in per-head BHSD layout, so route it through the
    # strided fused kernel even when seq_len==1 makes the permuted view
    # nominally contiguous — the contiguous fallback below is full-rotary only.
    # table_dim is the stored cos/sin width; with cos_half the rotate_half
    # duplicate is dropped, so the true rotary span is twice the table width.
    table_dim = cos.shape[-1]
    rotary_dim = table_dim * 2 if cos_half else table_dim
    is_partial = len(shape) == 4 and rotary_dim != shape[-1]
    if len(shape) == 4 and (is_partial or not x.is_contiguous() or cos_half):
        batch, heads, seq_len, head_dim = shape
        itemsize = x._dtype.itemsize
        elem_strides = tuple(stride // itemsize for stride in x._strides)
        offset_elems = x._offset // itemsize
        base_buf = _root_flat_buf(x)
        total_heads = batch * heads

        cos_seq = cos.size // table_dim if table_dim > 0 else 1
        flat_cos = cos.reshape((cos_seq, table_dim))
        flat_sin = sin.reshape((cos_seq, table_dim))
        rows = total_heads * seq_len
        cos_rows = cos_seq if cos_seq != rows else 0

        out_buf = _alloc_scratch((total_heads * seq_len, head_dim), x.dtype)
        result = _RMS_NORM_ROPE_STRIDED_KERNEL[(seq_len, total_heads)](
            base_buf,
            weight,
            flat_cos,
            flat_sin,
            out_buf,
            BH=total_heads,
            HEADS_PER_BATCH=heads,
            SEQ_LEN=seq_len,
            HEAD_DIM=head_dim,
            ROTARY_DIM=rotary_dim if rotary_dim != head_dim else 0,
            X_OFFSET=offset_elems,
            X_BATCH_STRIDE=elem_strides[0],
            X_HEAD_STRIDE=elem_strides[1],
            X_SEQ_STRIDE=elem_strides[2],
            COS_ROWS=cos_rows,
            EPS=eps,
            HALF_COS=1 if cos_half else 0,
        )
        result = result.reshape(shape)
        return result
    # Contiguous fallback: chain unfused rms_norm + rope_apply. The strided
    # fused dispatch above is a perf optimization; correctness must work for
    # the contiguous shape too because the rewrite_rms_norm_rope pass can
    # match patterns where the post-norm transpose ends up absorbed elsewhere
    # (e.g. Gemma3's q_norm-before-transpose forward feeds AOT a contiguous
    # 4D tensor into the fused op).
    normed_pair = _fused_rms_norm(x, weight, eps)
    normed = normed_pair[0] if isinstance(normed_pair, tuple) else normed_pair
    return _fused_rope_apply(normed, cos, sin)


def _rope_canonical_perm(
    x_shape: tuple[int, ...], cos_shape: tuple[int, ...]
) -> tuple[int, ...] | None:
    """The rope kernel flattens x to (rows, head_dim) and broadcasts cos via
    `row % cos_seq`, which only recovers the cos-varying axis when that axis sits
    just before head_dim (dim -2). Text Q/K are (B,H,S,D) with cos varying along S
    (dim -2) — canonical. Vision applies rope in (B,S,H,D) (cos varies along S at
    dim 1, broadcasts over heads at dim -2) — NOT canonical, so the modular index
    picks the wrong cos row. Return a permutation that moves the cos-varying axis
    to -2 (so the same kernel is correct), or None if already canonical.

    Conservative: only the same-rank, single-varying-axis case (the vision layout);
    anything else is left to the existing path so the text rope is untouched.
    """
    nd = len(x_shape)
    if nd != len(cos_shape) or nd < 3:
        return None
    varying = [d for d in range(nd - 1) if cos_shape[d] != 1]
    if len(varying) != 1 or varying[0] == nd - 2:
        return None
    va = varying[0]
    return tuple(d for d in range(nd - 1) if d != va) + (va, nd - 1)


def _invert_perm(perm: tuple[int, ...]) -> tuple[int, ...]:
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return tuple(inv)


def _fused_rope_apply(x: AlloyBuffer, cos: AlloyBuffer, sin: AlloyBuffer) -> AlloyBuffer:
    """Dispatch fused RoPE: out = x*cos + rotate_half(x)*sin. Non-canonical layouts
    (e.g. vision's (B,S,H,D)) are permuted to canonical (cos-varying axis at -2),
    run through the same kernel, then permuted back."""
    perm = _rope_canonical_perm(x.shape, cos.shape)
    if perm is not None and cos.shape[-1] == x.shape[-1] and x._dtype.itemsize == 2:
        # Full-rotary non-canonical layout (vision's (B,S,H,D), cos broadcasting
        # over heads) IN 16-BIT: compute with elementwise primitives instead of the
        # permute→fused-kernel→permute-back path. That path reads out-of-bounds in
        # f16 (a masked vector load past the permuted buffer picks up uninitialized
        # memory — benign as zeros in f32, NaN/garbage in f16), which is what blocks
        # native-f16 vision. The elementwise form composes cleanly and is exact, so
        # it matches the kernel; restricting it to 16-bit keeps f32 vision (which
        # already works) bit-for-bit on the fast fused kernel.
        return _rope_apply_elementwise(x, cos, sin)
    if perm is not None:
        out = _fused_rope_apply_canonical(
            x.permute(perm).contiguous(),
            cos.permute(perm).contiguous(),
            sin.permute(perm).contiguous(),
        )
        return out.permute(_invert_perm(perm))
    return _fused_rope_apply_canonical(x, cos, sin)


def _rope_apply_elementwise(x: AlloyBuffer, cos: AlloyBuffer, sin: AlloyBuffer) -> AlloyBuffer:
    """out = x*cos + rotate_half(x)*sin, where rotate_half(x) = cat(-x2, x1).
    Pure elementwise (slice/neg/cat/mul/add) — robust across layouts and dtypes;
    cos/sin broadcast against x. Full rotary only (cos width == head_dim)."""
    d = x.shape[-1]
    half = d // 2
    x1 = x.slice(x.ndim - 1, 0, half)
    x2 = x.slice(x.ndim - 1, half, d)
    rotated = _cat([x2 * -1.0, x1], dim=x.ndim - 1)
    return x * cos + rotated * sin


def _fused_rope_apply_canonical(
    x: AlloyBuffer, cos: AlloyBuffer, sin: AlloyBuffer
) -> AlloyBuffer:
    shape = x.shape
    head_dim = shape[-1]
    rows = 1
    for dim in shape[:-1]:
        rows *= dim

    cos_seq = cos.size // head_dim if head_dim > 0 else 1
    flat_cos = cos.reshape((cos_seq, head_dim))
    flat_sin = sin.reshape((cos_seq, head_dim))
    cos_rows = cos_seq if cos_seq != rows else 0

    if len(shape) == 4 and not x.is_contiguous():
        batch, heads, seq_len, head_dim = shape
        itemsize = x._dtype.itemsize
        elem_strides = tuple(stride // itemsize for stride in x._strides)
        offset_elems = x._offset // itemsize
        base_buf = _root_flat_buf(x)
        total_heads = batch * heads
        out_buf = _alloc_scratch((total_heads * seq_len, head_dim), x.dtype)
        grid = (seq_len, total_heads)
        result = _ROPE_APPLY_STRIDED_KERNEL[grid](
            base_buf,
            flat_cos,
            flat_sin,
            out_buf,
            BH=total_heads,
            HEADS_PER_BATCH=heads,
            SEQ_LEN=seq_len,
            HEAD_DIM=head_dim,
            X_OFFSET=offset_elems,
            X_BATCH_STRIDE=elem_strides[0],
            X_HEAD_STRIDE=elem_strides[1],
            X_SEQ_STRIDE=elem_strides[2],
            COS_ROWS=cos_rows,
        )
    else:
        flat_x = x.reshape((rows, head_dim))
        result = _ROPE_APPLY_KERNEL(flat_x, flat_cos, flat_sin, COS_ROWS=cos_rows)

    if len(shape) > 2:
        result = result.reshape(shape)
    return result


def _fused_rope_apply_backward(
    dout: AlloyBuffer, cos: AlloyBuffer, sin: AlloyBuffer
) -> AlloyBuffer:
    """Backward of rope_apply: dx = (dout1*c1 + dout2*s2, dout2*c2 - dout1*s1)."""
    shape = dout.shape
    head_dim = shape[-1]
    rows = 1
    for dim in shape[:-1]:
        rows *= dim

    if cos._dtype != dout._dtype:
        cos = cos.to(dout._dtype)
    if sin._dtype != dout._dtype:
        sin = sin.to(dout._dtype)

    cos_seq = cos.size // head_dim if head_dim > 0 else 1
    flat_cos = cos.reshape((cos_seq, head_dim))
    flat_sin = sin.reshape((cos_seq, head_dim))
    cos_rows = cos_seq if cos_seq != rows else 0

    if len(shape) == 4 and not dout.is_contiguous():
        batch, heads, seq_len, head_dim = shape
        itemsize = dout._dtype.itemsize
        elem_strides = tuple(stride // itemsize for stride in dout._strides)
        offset_elems = dout._offset // itemsize
        base_buf = _root_flat_buf(dout)
        total_heads = batch * heads
        out_buf = _alloc_aligned((total_heads * seq_len, head_dim), dout.dtype)
        grid = (seq_len, total_heads)
        result = _ROPE_APPLY_BACKWARD_STRIDED_KERNEL[grid](
            base_buf,
            flat_cos,
            flat_sin,
            out_buf,
            BH=total_heads,
            HEADS_PER_BATCH=heads,
            SEQ_LEN=seq_len,
            HEAD_DIM=head_dim,
            D_OFFSET=offset_elems,
            D_BATCH_STRIDE=elem_strides[0],
            D_HEAD_STRIDE=elem_strides[1],
            D_SEQ_STRIDE=elem_strides[2],
            COS_ROWS=cos_rows,
        )
        result = result.reshape((batch, seq_len, heads, head_dim)).transpose(1, 2)
    else:
        flat_dout = dout.reshape((rows, head_dim))
        result = _ROPE_APPLY_BACKWARD_KERNEL(flat_dout, flat_cos, flat_sin, COS_ROWS=cos_rows)
        if len(shape) > 2:
            result = result.reshape(shape)
    return result
