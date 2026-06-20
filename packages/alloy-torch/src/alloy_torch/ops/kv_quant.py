"""FX handler for the q8_0 quantized-KV attention op."""

from __future__ import annotations

import math
from typing import cast

import torch

from alloy._compiler.dtypes import float16, uint32
from alloy._dispatch.buf_utils import _alloc_scratch
from alloy._dispatch.kernel import KernelFunction
from alloy._runtime.alloy_buffer import AlloyBuffer
from alloy.std.kv_quant import (
    attention_decode_vector_split_q8,
    kv_dequant_q8_range,
    kv_quantize_q8_range,
)
from alloy_torch.ops.attention import (
    _ATTENTION_DECODE_COMBINE_VECTOR_KERNEL,
    _ATTENTION_DECODE_COMBINE_VECTOR_PAR_KERNEL,
    _attention_cache_handler,
    _choose_flash_decoding_splits,
    _no_bound_buf,
)
from alloy_torch.extern_kv import note_extern_kv_write
from alloy_torch.ops.casting import _to_copy
from alloy_torch.ops.common import _root_flat_buf

_KV_QUANTIZE_Q8_RANGE_KERNEL = cast(KernelFunction, kv_quantize_q8_range)
_KV_DEQUANT_Q8_RANGE_KERNEL = cast(KernelFunction, kv_dequant_q8_range)
_ATTENTION_DECODE_VECTOR_SPLIT_Q8_KERNEL = cast(KernelFunction, attention_decode_vector_split_q8)


def _strides_elems(buf: AlloyBuffer) -> tuple[tuple[int, ...], int]:
    itemsize = buf._dtype.itemsize
    return tuple(s // itemsize for s in buf._strides), buf._offset // itemsize


def _u32_root(root: AlloyBuffer) -> AlloyBuffer:
    """uint32 view over an int8 codes root (same storage): lets the D=512
    decode path fetch 16 codes per load4_vec (uint4) while the fused write
    keeps byte-granular int8 stores through the original root. `_view_of`
    propagates the pending producer so lazy collection still chains."""
    v = AlloyBuffer(
        root._parent_handle,
        root._offset,
        (root.size // 4,),
        (4,),
        uint32,
        raw_ptr=root._raw_ptr,
        total_nbytes=root.metal_nbytes,
    )
    root._view_of(v)
    return v


def _attention_cache_q8_handler(
    q: AlloyBuffer,
    new_k: AlloyBuffer,
    new_v: AlloyBuffer,
    cache_pos: AlloyBuffer,
    k_codes: AlloyBuffer,
    k_scales: AlloyBuffer,
    v_codes: AlloyBuffer,
    v_scales: AlloyBuffer,
    scale: float = -1.0,
    sliding_window: int = 0,
    write_kv: bool = True,
    last_real: AlloyBuffer | None = None,
) -> AlloyBuffer:
    q_shape = q.shape
    if len(q_shape) != 4:
        raise ValueError(f"attention_cache_q8 expects 4D Q, got {len(q_shape)}D")
    batch, heads, seq_len, head_dim = q_shape
    if head_dim % 32 != 0:
        raise ValueError(f"attention_cache_q8 needs head_dim % 32 == 0, got {head_dim}")
    _, kv_heads, s_max, _ = k_codes.shape
    kv_group = heads // kv_heads

    pos_buf = cache_pos
    if pos_buf._dtype.itemsize != 4:
        pos_buf = _to_copy(pos_buf, dtype=torch.int32)
    if len(pos_buf.shape) and pos_buf.shape[0] != 1:
        # Prefill passes cache_position as an arange; the quantize/dequant
        # kernels read element 0 = the chunk's start position.
        pos_buf = pos_buf.slice(0, 0, 1)

    # One root view per cache buffer, shared between the quantize ops' WRITE
    # and the attention/dequant ops' READ (per-object dependency tracking).
    k_codes_root = _root_flat_buf(k_codes)
    k_scales_root = _root_flat_buf(k_scales)
    v_codes_root = _root_flat_buf(v_codes)
    v_scales_root = _root_flat_buf(v_scales)

    # --- durable write (prefill only): bulk-quantize the chunk's K/V rows.
    # Decode fuses the single-token quantize into the attention kernel
    # (WRITE_KV) — two standalone dispatches per layer per step drag tpot below
    # fp16. write_kv=False is the gemma4 KV-shared read: the source layer already
    # quantize-wrote this step; new_k/new_v are throwaway views, never read.
    if write_kv and seq_len > 1:
        for new, codes_root, scales_root in (
            (new_k, k_codes_root, k_scales_root),
            (new_v, v_codes_root, v_scales_root),
        ):
            n_strides, n_offset = _strides_elems(new)
            _KV_QUANTIZE_Q8_RANGE_KERNEL[(kv_heads, seq_len)](
                _root_flat_buf(new),
                pos_buf,
                last_real if last_real is not None else _no_bound_buf(),
                codes_root,
                scales_root,
                S_MAX=s_max,
                HEAD_DIM=head_dim,
                SRC_OFFSET=n_offset,
                SRC_HEAD_STRIDE=n_strides[1],
                SRC_SEQ_STRIDE=n_strides[2],
                SLIDING_WINDOW=sliding_window,
            )

    if seq_len > 1:
        if write_kv:
            # Prefill: nothing in-plan reads the chunk's own codes — keep the
            # quantize ops alive at graph output.
            for root in (k_codes_root, k_scales_root, v_codes_root, v_scales_root):
                note_extern_kv_write(root)
        # Materialize fallback: history + the just-quantized chunk
        # ([0, chunk_start + seq_len)) dequantize into fp16 scratch in cache
        # layout; the stock cache-attention path runs read-only against it.
        # Single writer per scratch: a write_kv=True delegation would queue
        # attention_kv_write as a second writer through its own root views,
        # shadowing the dequant's materializer on the shared storage — the lazy
        # collector follows buffers' current materializers, not the queue-time
        # input_producers snapshot, and silently drops the dequant.
        k_scratch = _alloc_scratch((1, kv_heads, s_max, head_dim), float16)
        v_scratch = _alloc_scratch((1, kv_heads, s_max, head_dim), float16)
        for codes_root, scales_root, scratch in (
            (k_codes_root, k_scales_root, k_scratch),
            (v_codes_root, v_scales_root, v_scratch),
        ):
            _KV_DEQUANT_Q8_RANGE_KERNEL[(kv_heads, (s_max + 63) // 64)](
                codes_root,
                scales_root,
                pos_buf,
                scratch,
                S_MAX=s_max,
                HEAD_DIM=head_dim,
                END_OFFSET=seq_len,
                SLIDING_WINDOW=sliding_window,
                TOKENS_PER_PROG=64,
            )
        return _attention_cache_handler(
            q,
            new_k,
            new_v,
            cache_pos,
            k_scratch,
            v_scratch,
            scale,
            sliding_window,
            write_kv=False,
        )

    # --- decode: read-only q8 flash decoding + shared combine ---
    default_scale = 1.0 / math.sqrt(head_dim)
    if (
        scale is None
        or scale <= 0
        or math.isclose(float(scale), default_scale, rel_tol=1e-6, abs_tol=1e-6)
    ):
        custom_scale = 0
    else:
        custom_scale = float(scale)

    # Mirror the fp16 path's _q_to_cache_dtype contract: Q (and with it the
    # partials and the attention output) goes fp16, so downstream o_proj GEMVs
    # see the same activation dtype/configs as the fp16 path. Leaving Q fp32
    # propagates f32 activations into the tuned-for-fp16 GEMVs.
    if q._dtype.itemsize == 4:
        q = _to_copy(q, dtype=torch.float16)
    q_strides, q_offset = _strides_elems(q)
    nk_strides, nk_offset = _strides_elems(new_k)
    nv_strides, nv_offset = _strides_elems(new_v)
    kc_strides, _ = _strides_elems(k_codes)
    ks_strides, _ = _strides_elems(k_scales)
    vc_strides, _ = _strides_elems(v_codes)
    vs_strides, _ = _strides_elems(v_scales)

    total_heads = batch * heads
    splits = max(1, _choose_flash_decoding_splits(total_heads, s_max, simdgroups_per_tg=1))
    out = _alloc_scratch((batch * seq_len * heads, head_dim), q.dtype)
    partial_o = _alloc_scratch((total_heads, splits, head_dim), q.dtype)
    partial_lse = _alloc_scratch((total_heads, splits), q.dtype)

    codes_u32 = head_dim // 32 == 16  # PER_LANE==16 (D=512): uint4-packed loads
    _ATTENTION_DECODE_VECTOR_SPLIT_Q8_KERNEL[(total_heads, splits)](
        _root_flat_buf(q),
        _root_flat_buf(new_k),
        _root_flat_buf(new_v),
        pos_buf,
        k_codes_root,
        k_scales_root,
        v_codes_root,
        v_scales_root,
        _u32_root(k_codes_root) if codes_u32 else k_codes_root,
        _u32_root(v_codes_root) if codes_u32 else v_codes_root,
        partial_o,
        partial_lse,
        BH=total_heads,
        HEADS_PER_BATCH=heads,
        HEAD_DIM=head_dim,
        Q_OFFSET=q_offset,
        Q_BATCH_STRIDE=q_strides[0],
        Q_HEAD_STRIDE=q_strides[1],
        NK_OFFSET=nk_offset,
        NK_HEAD_STRIDE=nk_strides[1],
        NV_OFFSET=nv_offset,
        NV_HEAD_STRIDE=nv_strides[1],
        KC_HEAD_STRIDE=kc_strides[1],
        KC_SEQ_STRIDE=kc_strides[2],
        KS_HEAD_STRIDE=ks_strides[1],
        KS_SEQ_STRIDE=ks_strides[2],
        VC_HEAD_STRIDE=vc_strides[1],
        VC_SEQ_STRIDE=vc_strides[2],
        VS_HEAD_STRIDE=vs_strides[1],
        VS_SEQ_STRIDE=vs_strides[2],
        KV_GROUP=kv_group,
        SPLITS=splits,
        SLIDING_WINDOW=sliding_window,
        CUSTOM_SCALE=custom_scale,
        WRITE_KV=1 if write_kv else 0,
        CODES_U32=1 if codes_u32 else 0,
    )
    combine = (
        _ATTENTION_DECODE_COMBINE_VECTOR_PAR_KERNEL
        if splits >= 32
        else _ATTENTION_DECODE_COMBINE_VECTOR_KERNEL
    )
    grid = (total_heads, head_dim // 4) if splits >= 32 else (total_heads,)
    result = combine[grid](
        partial_o,
        partial_lse,
        out,
        BH=total_heads,
        HEADS_PER_BATCH=heads,
        HEAD_DIM=head_dim,
        SPLITS=splits,
    )
    return result.reshape((batch, seq_len, heads, head_dim)).transpose(0, 2, 1, 3)
