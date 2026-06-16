"""Attention handlers for torch op lowering."""

from __future__ import annotations

import math
from typing import cast

import torch

from alloy._compiler.dtypes import float32, from_torch_dtype, int64
from alloy._dispatch.buf_utils import _alloc_aligned, _alloc_scratch
from alloy._dispatch.dispatch import DispatchEngine
from alloy._dispatch.kernel import KernelFunction
from alloy._dispatch.lazy import LazyOp
from alloy._runtime.alloy_buffer import AlloyBuffer
from alloy._runtime.tune_configs import resolve_config
from alloy.std.attention import (
    attention_compute_delta_strided,
    attention_decode_combine,
    attention_decode_combine_multi,
    attention_decode_combine_vector,
    attention_decode_combine_vector_multi,
    attention_decode_combine_vector_par,
    attention_decode_vector_split,
    attention_kv_update,
    attention_kv_update_split,
    attention_kv_update_split_multi,
    attention_kv_update_vector_split_multi,
    attention_kv_write,
    attention_strided,
    attention_strided_backward_dkdv,
    attention_strided_backward_dkdv_masked_by_batch,
    attention_strided_backward_dq,
    attention_strided_backward_dq_masked_by_batch,
    attention_strided_logsumexp,
    attention_strided_masked_by_batch_with_lse,
    attention_strided_runtime_pos,
    attention_strided_runtime_pos_split,
    attention_combine_splits,
)
from alloy_torch.compile_window import compile_window
from alloy_torch.extern_kv import note_extern_kv_write
from alloy_torch.ops.casting import _to_copy
from alloy_torch.ops.common import _expand_lazy_buffer, _root_flat_buf

_ATTENTION_KV_UPDATE_KERNEL = cast(KernelFunction, attention_kv_update)
_ATTENTION_KV_UPDATE_SPLIT_KERNEL = cast(KernelFunction, attention_kv_update_split)
_ATTENTION_KV_UPDATE_SPLIT_MULTI_KERNEL = cast(KernelFunction, attention_kv_update_split_multi)
_ATTENTION_KV_UPDATE_VECTOR_SPLIT_MULTI_KERNEL = cast(KernelFunction, attention_kv_update_vector_split_multi)
_ATTENTION_DECODE_COMBINE_VECTOR_MULTI_KERNEL = cast(KernelFunction, attention_decode_combine_vector_multi)
_ATTENTION_KV_WRITE_KERNEL = cast(KernelFunction, attention_kv_write)
_ATTENTION_DECODE_VECTOR_SPLIT_KERNEL = cast(KernelFunction, attention_decode_vector_split)
_ATTENTION_DECODE_COMBINE_KERNEL = cast(KernelFunction, attention_decode_combine)
_ATTENTION_DECODE_COMBINE_VECTOR_KERNEL = cast(KernelFunction, attention_decode_combine_vector)
_ATTENTION_DECODE_COMBINE_VECTOR_PAR_KERNEL = cast(KernelFunction, attention_decode_combine_vector_par)
_ATTENTION_DECODE_COMBINE_MULTI_KERNEL = cast(KernelFunction, attention_decode_combine_multi)
_ATTENTION_STRIDED_KERNEL = cast(KernelFunction, attention_strided)
_ATTENTION_STRIDED_RUNTIME_POS_KERNEL = cast(KernelFunction, attention_strided_runtime_pos)
_ATTENTION_STRIDED_RUNTIME_POS_SPLIT_KERNEL = cast(
    KernelFunction, attention_strided_runtime_pos_split
)
_ATTENTION_COMBINE_SPLITS_KERNEL = cast(KernelFunction, attention_combine_splits)
_ATTENTION_STRIDED_LSE_KERNEL = cast(KernelFunction, attention_strided_logsumexp)
_ATTENTION_STRIDED_MASKED_LSE_KERNEL = cast(
    KernelFunction, attention_strided_masked_by_batch_with_lse
)
_ATTENTION_DELTA_KERNEL = cast(KernelFunction, attention_compute_delta_strided)
_ATTENTION_BACKWARD_DQ_KERNEL = cast(KernelFunction, attention_strided_backward_dq)
_ATTENTION_BACKWARD_DKDV_KERNEL = cast(KernelFunction, attention_strided_backward_dkdv)
_ATTENTION_BACKWARD_DQ_MASKED_KERNEL = cast(
    KernelFunction, attention_strided_backward_dq_masked_by_batch
)
_ATTENTION_BACKWARD_DKDV_MASKED_KERNEL = cast(
    KernelFunction, attention_strided_backward_dkdv_masked_by_batch
)

AttentionConstexprs = dict[str, int | float]
AttentionBuffers = list[tuple[str, AlloyBuffer]]

# Inert "no ring bound" constant for dispatch sites without a managed
# `last_real` operand — allocated once, never mutated (negative = unbounded).
_NO_BOUND: AlloyBuffer | None = None


def _no_bound_buf() -> AlloyBuffer:
    global _NO_BOUND
    if _NO_BOUND is None:
        buf = _alloc_aligned((1,), int64)
        DispatchEngine.default().untrack_alloc(buf.base_ptr)
        buf.numpy[:] = -1
        _NO_BOUND = buf
    return _NO_BOUND

_kbias_cache: set[tuple[int, ...]] = set()

# Max query length routed through the spec-decode multi-token verify kernel.
# seq_len in [2, _MAX_VERIFY_K] uses attention_kv_update_multi; above it goes
# to the prefill (strided) path; ==1 is single-token decode. Canonical home is
# here (the lower layer where the handlers live); `multi_token_attention`
# imports it. Used by `_attention_cache_handler`'s runtime dispatch.
# 16 covers the block-16 DFlash verify (the z-lab drafts are trained at
# block 16); seq_len 9..16 occurs only in spec verify, so plain decode and
# prefill routing are unaffected.
_MAX_VERIFY_K = 16

# Warm-prefill and grid-shrink compile flags live on
# `alloy_torch.compile_window` (the trace-boundary channel); the SDPA
# handler reads them below. Grid-shrink rationale: a split-K plan's `splits`
# is M_MAX-derived and its combine grid is (M_MAX, heads), neither of which
# shrinks per request — single-pass has grid (q_blocks, heads) whose q_blocks
# axis shrinks cleanly, and its causal early-exit bounds each real query's
# K-scan to its own position, so attention cost scales with the real prompt.


def _attention_kv_update_static_kv_len(k_cache_shape: tuple[int, ...]) -> int:
    if len(k_cache_shape) != 4:
        raise ValueError(
            f"attention_kv_update expects 4D K cache, got {len(k_cache_shape)}D"
        )
    return k_cache_shape[2]


def _q_to_cache_dtype(q: AlloyBuffer, k_cache: AlloyBuffer) -> AlloyBuffer:
    """Downcast Q to the KV-cache dtype (f16) at the dispatch boundary so the
    attention simdgroup MMA runs at f16 tensor-core throughput (~2x) instead of
    being forced to f32 by a mismatched operand: the planner gates both MMA
    operands to one shmem-tile dtype, so an f32 Q drags K up to f32 and halves
    the Q·K / P·V dot throughput. Q is born f32 from q_proj; a model-forward
    cast is normalized away by the FX attention rewrite (it sees through
    `_to_copy`), so the downcast must happen here where the kernel's operand
    buffer dtype is set. Scores, softmax, and output still accumulate in f32
    in-kernel, so this is precision-neutral for 1/sqrt(d)-scaled attention."""
    if q._dtype != k_cache._dtype:
        return _to_copy(q, dtype=k_cache._dtype.to_torch_dtype())
    return q


def _parse_bhsd(buf: AlloyBuffer) -> tuple[int, int, int, int, int, int, int, int] | None:
    """Extract (batch, heads, seq, dim, offset, batch_stride, head_stride, seq_stride)."""
    shape = buf._shape
    ndim = len(shape)
    itemsize = buf._dtype.itemsize
    elem_strides = tuple(stride // itemsize for stride in buf._strides)
    offset = buf._offset // itemsize

    if elem_strides[-1] != 1:
        return None

    if ndim == 2:
        return 1, 1, shape[0], shape[1], offset, 0, 0, elem_strides[0]
    if ndim == 3:
        return 1, shape[0], shape[1], shape[2], offset, 0, elem_strides[0], elem_strides[1]
    if ndim == 4:
        return (
            shape[0],
            shape[1],
            shape[2],
            shape[3],
            offset,
            elem_strides[0],
            elem_strides[1],
            elem_strides[2],
        )
    return None


def _legal_fallback_blocks(head_dim: int, block: int) -> tuple[int, int]:
    """Thread/shmem-legal (block_m, block_n) when no tuned config exists.

    The o-accumulator dot (p @ v, N = head_dim) spawns
    (block_m/tile)*(head_dim/tile) simdgroups — tile = 16 when block_m % 16 == 0
    (pick_dot_reg picks reg=2), else 8 — and >32 simdgroups busts the 1024
    thread/threadgroup limit. Counterintuitively a LARGER block_m can be legal
    where a smaller one isn't: at head_dim 512, block_m 8 is reg=1 → 64
    simdgroups (illegal) but block_m 16 is reg=2 → 32 (legal). So where the
    square default overflows, step up to the reg=2 block and shrink block_n until
    the Q + 2·(K,V) shmem tiles fit 32 KB. A no-op for the head dims (≤256) where
    the default already fits — those keep their tuned/larger blocks."""
    def n_simdgroups(bm: int) -> int:
        tile = 16 if bm % 16 == 0 else 8
        return (bm // tile) * (head_dim // tile)

    if head_dim <= 0 or (head_dim < 256 and n_simdgroups(block) <= 32):
        return block, block
    # head_dim >= 256: BLOCK_M must drop to 16 even where the simdgroup count
    # is legal — at head_dim 256 the Q tile + f32 o-accumulator alone overrun
    # the 32KB budget for ANY BLOCK_M=32 config (measured: (32, 8) emits
    # 42240B, the planner cannot column-tile Dot kernels), while BLOCK_M=16
    # compiles at every BLOCK_N up to 128. The (bm + 2*bn) shmem loop below
    # underestimates (no accumulator term), so it alone cannot catch this.
    block_m, block_n = 16, (min(block, 32) if head_dim >= 256 else block)
    while block_n > 8 and (block_m + 2 * block_n) * head_dim * 2 > 32768:
        block_n //= 2
    return block_m, block_n


def _resolved_blocks_for_attention(
    kernel: KernelFunction,
    constexpr_kwargs: AttentionConstexprs,
    buffer_args: AttentionBuffers,
    fallback_block: int,
) -> tuple[int, int]:
    """Resolve BLOCK_M and BLOCK_N for an SDPA kernel from the tune cache.

    Goes through `resolve_config` — the SAME entry point the kernel-level
    `_resolve_tune` uses — so cross-cutting key transforms apply here too.
    A previous duplicate of the dict lookup bypassed the grid-shrink
    representative-M cap (`_apply_oneshot_cap`): at native M_MAX the handler
    missed the tuned entry and fell back to BLOCK_M=8/BLOCK_N=32 while the
    kernel-level resolution (capped) hit BLOCK_M=16/BLOCK_N=128 — a 10x
    attention slowdown per one-shot dispatch that the resolve-probe (which
    patches resolve_config) couldn't see.
    """
    key_values: dict[str, int] = {}
    if kernel._tune_key is not None:
        for key in kernel._tune_key:
            if key in constexpr_kwargs:
                key_values[key] = int(constexpr_kwargs[key])
    else:
        for key in kernel._constexpr_params:
            if key not in kernel._tune_tuned_params and key in constexpr_kwargs:
                key_values[key] = int(constexpr_kwargs[key])
        for param_name, arg in buffer_args:
            for dim_index, dim in enumerate(arg.shape):
                key_values[f"_{param_name}_dim{dim_index}"] = int(dim)

    cfg = resolve_config(kernel.name, key_values)
    head_dim = int(constexpr_kwargs.get("HEAD_DIM", 0))
    # A kernel-wide CONSERVATIVE DEFAULT is not shape-aware: attention's
    # (32, 64) busts the 32KB shmem budget at head_dim >= 256 (gemma4's
    # one-shot recipe-probe compile dies on it — the probe's off-tune
    # SEQ_LEN misses every entry). Route defaults through the head_dim-aware
    # fallback; only configs TUNED for a matching shape are trusted.
    if not cfg.constexprs or cfg.is_default:
        return _legal_fallback_blocks(head_dim, fallback_block)
    block_m = int(cfg.constexprs.get("BLOCK_M", fallback_block))
    block_n = int(cfg.constexprs.get("BLOCK_N", fallback_block))
    # Validate thread-legality: a tuned/stale config can carry a block that busts
    # the 1024-thread limit at large head_dim — e.g. head_dim 512 with block_m 32 →
    # (32/16)*(512/16) = 64 simdgroups = 2048 threads. (This bites gemma4's head_dim
    # 512 global layers once split-K is off for them: they take this single-pass
    # path, and a stale split-K-era config would otherwise emit an illegal kernel.)
    # Fall back to the legal block rather than fail in the thread-model planner.
    if head_dim > 0:
        tile = 16 if block_m % 16 == 0 else 8
        if (block_m // tile) * (head_dim // tile) > 32:
            return _legal_fallback_blocks(head_dim, fallback_block)
    return block_m, block_n


def _attention_block_size(
    head_dim: int,
    *,
    backward: bool = False,
    seq_len: int | None = None,
    input_dtype: torch.dtype | None = None,
) -> int:
    if backward:
        max_block = 32768 // (6 * max(head_dim, 1) * 4)
    else:
        in_bytes = 2 if input_dtype in (torch.bfloat16, torch.float16) else 4
        per_block = max(head_dim, 1) * (3 * in_bytes + 4) + 8
        max_block = 32768 // per_block

    if max_block >= 32:
        block = 32
    elif max_block >= 16:
        block = 16
    else:
        block = 8

    if seq_len is not None and seq_len > 0:
        capped = 8
        while capped * 2 <= seq_len and capped * 2 <= block:
            capped *= 2
        block = capped
    if head_dim < 16:
        block = min(block, max(head_dim, 8))
    return block


def _reshape_attention_mask(
    attn_mask: AlloyBuffer,
    *,
    batch: int,
    heads: int,
    q_len: int,
    k_len: int,
) -> AlloyBuffer | None:
    """Reshape attention mask to (B*q_len, k_len) for the masked kernel."""
    if not attn_mask._dtype.is_float():
        # Bool/int mask -> additive: 0 where keep (True/1), -1e30 where masked
        # (False/0). `(mask - 1) * 1e30` instead of where(mask, 0, -1e30) — the
        # scalar-broadcast path of `where` is buggy, and this is a clean two-op
        # elementwise chain. Only runs for non-float masks; float masks (the text
        # path's additive masks) skip this entirely.
        mask_f = attn_mask.to(torch.float32)
        attn_mask = (mask_f - 1.0) * 1e30

    ndim = attn_mask.ndim
    shape = attn_mask._shape
    if ndim == 2:
        return _broadcast_mask_to_batch(attn_mask, batch, q_len, k_len)
    if ndim == 3:
        if shape[0] == 1 and batch > 1:
            expanded = _expand_lazy_buffer(attn_mask, (batch, q_len, k_len))
            return expanded.reshape((batch * q_len, k_len))
        return attn_mask.reshape((batch * q_len, k_len))
    if ndim == 4:
        if shape[1] != 1 and shape[1] != heads:
            return None
        if shape[1] == heads and heads > 1:
            return None
        squeezed = attn_mask.reshape((shape[0], shape[2], shape[3]))
        if shape[0] == 1 and batch > 1:
            squeezed = _expand_lazy_buffer(squeezed, (batch, q_len, k_len))
        return squeezed.reshape((batch * q_len, k_len))
    return None


def _broadcast_mask_to_batch(
    mask: AlloyBuffer,
    batch: int,
    q_len: int,
    k_len: int,
) -> AlloyBuffer:
    """Broadcast a (q, k) mask to (B*q, k) via expand+contiguify."""
    if batch == 1:
        return mask.reshape((q_len, k_len))
    expanded = mask.reshape((1, q_len, k_len))
    expanded = _expand_lazy_buffer(expanded, (batch, q_len, k_len))
    return expanded.reshape((batch * q_len, k_len))


def _gpu_sdpa(
    q: AlloyBuffer,
    k: AlloyBuffer,
    v: AlloyBuffer,
    *,
    attn_mask: AlloyBuffer | None = None,
    is_causal: bool = False,
    scale: float | None = None,
    _kv_group: int = 1,
    need_lse: bool = True,
) -> tuple[AlloyBuffer, AlloyBuffer | None]:
    if _has_bias_add_in_producer_chain(k):
        _kbias_cache.add(_sdpa_shape_key(q, k))
    q_info = _parse_bhsd(q)
    k_info = _parse_bhsd(k)
    v_info = _parse_bhsd(v)
    if q_info is None or k_info is None or v_info is None:
        raise ValueError("Invalid buffer shapes for SDPA")

    batch, heads, q_len, head_dim, q_off, q_bs, q_hs, q_ss = q_info
    k_batch, k_heads, kv_len, k_dim, k_off, k_bs, k_hs, k_ss = k_info
    v_batch, v_heads, v_len, v_dim, v_off, v_bs, v_hs, v_ss = v_info

    if batch != k_batch or batch != v_batch:
        raise ValueError("Batch dimensions do not match")
    if k_dim != head_dim or v_dim != head_dim or v_len != kv_len:
        raise ValueError("Input dimensions do not match")
    if not q._dtype.is_float():
        raise ValueError("Q buffer is not a float type")

    # Bucketed prefill: the K/V tensors come straight from HF's StaticLayer
    # cache, which always returns the full max_context view (e.g. 2048).
    # For COLD prefill the first q_len positions are real and the rest are
    # zero/garbage; slice to [0..q_len) so the kernel doesn't iterate over
    # irrelevant positions.
    # For WARM prefill the cache holds a populated prefix the model attends
    # to via causal attention, so K/V keep the full cache extent.
    q_start_pos = compile_window.q_start_pos
    effective_kv_len = q_start_pos + q_len
    # COLD prefill: slice K/V/mask to the real-data region — the rest of
    # the cache is uninitialized and the plan's KV_LEN constexpr can
    # safely bake `q_len`.
    # WARM prefill: leave K/V at full cache size.
    # Dynamo caches the alloy plan by graph bytecode + input shape, so a
    # per-call effective_kv_len would bake a stale KV_LEN constexpr that
    # the next warm call at a different start_pos would silently reuse,
    # truncating K/V iteration and producing wrong attention. The HF
    # causal mask (sliced only on q, full kv extent) already encodes
    # per-row causality including the warm-start offset, so the kernel
    # needs no Q_START_POS / effective_kv_len constexpr — it iterates
    # the full cache and the mask zeroes everything past each row's
    # allowed range.
    #
    # Only valid for CAUSAL attention: the slice drops K/V[q_len:kv_len] on the
    # premise those positions are future/garbage (a bucketed causal cache). A
    # NON-causal call with a full real K/V (q_len < kv_len) — e.g. whisper's
    # cross-attention prefill (4 SOT queries over 1500 real encoder frames) —
    # must attend ALL kv; slicing there silently drops 1496/1500 frames. Gate on
    # causality (is_causal, or a causal mask present) so cross-attention is left whole.
    if (q_len > 1 and q_start_pos == 0 and kv_len > effective_kv_len
            and (is_causal or attn_mask is not None)):
        k = k.slice(2, 0, effective_kv_len)
        v = v.slice(2, 0, effective_kv_len)
        if attn_mask is not None:
            attn_mask = attn_mask.slice(len(attn_mask._shape) - 1, 0, effective_kv_len)
        k_info = _parse_bhsd(k)
        v_info = _parse_bhsd(v)
        if k_info is None or v_info is None:
            raise ValueError("Invalid sliced K/V shape")
        _, k_heads, kv_len, _, k_off, k_bs, k_hs, k_ss = k_info
        _, v_heads, v_len, _, v_off, v_bs, v_hs, v_ss = v_info

    kv_group = _kv_group
    if heads != k_heads or heads != v_heads:
        if k_heads == v_heads and heads % k_heads == 0:
            kv_group = heads // k_heads
        else:
            raise ValueError("Head dimensions do not match")

    custom_scale = None
    default_scale = 1.0 / math.sqrt(head_dim)
    if scale is not None and not math.isclose(
        float(scale), default_scale, rel_tol=1e-6, abs_tol=1e-6
    ):
        custom_scale = float(scale)

    causal = is_causal
    mask_arr = None
    if attn_mask is not None:
        mask_arr = _reshape_attention_mask(
            attn_mask,
            batch=batch,
            heads=heads,
            q_len=q_len,
            k_len=kv_len,
        )
        if mask_arr is None:
            raise ValueError("Invalid attention mask")

    total_heads = batch * heads
    block = _attention_block_size(
        head_dim,
        seq_len=min(q_len, kv_len) if kv_len > 0 else q_len,
        input_dtype=q._dtype.to_torch_dtype() if hasattr(q._dtype, "to_torch_dtype") else None,
    )
    out_size = batch * q_len * heads * head_dim
    out_buf = _alloc_scratch((out_size,), q._dtype)

    q_flat = _root_flat_buf(q)
    k_flat = _root_flat_buf(k)
    v_flat = _root_flat_buf(v)

    stride_kwargs: AttentionConstexprs = {
        "BH": total_heads,
        "HEADS_PER_BATCH": heads,
        "SEQ_LEN": q_len,
        "KV_LEN": kv_len,
        "HEAD_DIM": head_dim,
        "Q_OFFSET": q_off,
        "Q_BATCH_STRIDE": q_bs,
        "Q_HEAD_STRIDE": q_hs,
        "Q_SEQ_STRIDE": q_ss,
        "K_OFFSET": k_off,
        "K_BATCH_STRIDE": k_bs,
        "K_HEAD_STRIDE": k_hs,
        "K_SEQ_STRIDE": k_ss,
        "V_OFFSET": v_off,
        "V_BATCH_STRIDE": v_bs,
        "V_HEAD_STRIDE": v_hs,
        "V_SEQ_STRIDE": v_ss,
        "causal": 1 if causal else 0,
        "KV_GROUP": kv_group,
        "CUSTOM_SCALE": custom_scale or 0,
        # Cold-prefill / SDPA-handler path bakes q_start_pos as a constexpr
        # so Metal can constant-fold the causal early-exit's loop bound.
        # Warm prefill goes through `attention_prefill_warm`'s dedicated
        # `attention_strided_runtime_pos` kernel which takes Q_START_POS
        # as a runtime buffer.
        "Q_START_POS": int(q_start_pos),
    }

    log_sum_exp_buf = _alloc_scratch((batch * q_len * heads,), from_torch_dtype(torch.float32))

    def _with_blocks(kernel: KernelFunction, buf_args: AttentionBuffers) -> AttentionConstexprs:
        block_m, block_n = _resolved_blocks_for_attention(
            kernel, stride_kwargs, buf_args, fallback_block=block
        )
        return {**stride_kwargs, "BLOCK_M": block_m, "BLOCK_N": block_n}

    if mask_arr is None:
        strided_kwargs = _with_blocks(
            _ATTENTION_STRIDED_KERNEL,
            [
                ("Q", q_flat),
                ("K", k_flat),
                ("V", v_flat),
                ("O", out_buf),
            ],
        )
        out = _ATTENTION_STRIDED_KERNEL(
            q_flat,
            k_flat,
            v_flat,
            out_buf,
            **strided_kwargs,
        )
        if need_lse:
            # Training / CPU-export path: emit lse via a second full attention
            # pass. Inference (`_scaled_dot_product_attention`) sets
            # `need_lse=False` and skips this — lse on the no-mask path was
            # being discarded immediately by the caller, costing one full
            # extra K-loop per attention layer.
            log_sum_exp = _ATTENTION_STRIDED_LSE_KERNEL(
                q_flat,
                k_flat,
                log_sum_exp_buf,
                **_with_blocks(
                    _ATTENTION_STRIDED_LSE_KERNEL,
                    [("Q", q_flat), ("K", k_flat), ("log_sum_exp", log_sum_exp_buf)],
                ),
            )
        else:
            log_sum_exp = None
    else:
        masked_kwargs = _with_blocks(
            _ATTENTION_STRIDED_MASKED_LSE_KERNEL,
            [
                ("Q", q_flat),
                ("K", k_flat),
                ("V", v_flat),
                ("Mask", mask_arr),
                ("O", out_buf),
                ("log_sum_exp", log_sum_exp_buf),
            ],
        )
        out, log_sum_exp = cast(
            tuple[AlloyBuffer, AlloyBuffer],
            _ATTENTION_STRIDED_MASKED_LSE_KERNEL(
                q_flat,
                k_flat,
                v_flat,
                mask_arr,
                out_buf,
                log_sum_exp_buf,
                **masked_kwargs,
            ),
        )

    ndim = len(q._shape)
    if ndim <= 2:
        lse_out = log_sum_exp.reshape((q_len,)) if log_sum_exp is not None else None
        return out, lse_out
    if ndim == 3:
        out_bhnd = out.reshape((q_len, heads, head_dim)).transpose(1, 0, 2)
        lse_out = log_sum_exp.reshape((heads, q_len)) if log_sum_exp is not None else None
        return out_bhnd, lse_out
    out_bhnd = out.reshape((batch, q_len, heads, head_dim)).transpose(0, 2, 1, 3)
    lse_out = log_sum_exp.reshape((batch, heads, q_len)) if log_sum_exp is not None else None
    return out_bhnd, lse_out


def _scaled_dot_product_attention(
    query: AlloyBuffer,
    key: AlloyBuffer,
    value: AlloyBuffer,
    attn_mask: AlloyBuffer | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: float | None = None,
    enable_gqa: bool = False,
    _kv_group: int = 1,
) -> AlloyBuffer:
    if dropout_p != 0.0:
        raise NotImplementedError("Alloy backend only supports dropout_p=0 for inference")
    if enable_gqa:
        raise NotImplementedError("Alloy backend does not support enable_gqa yet")
    out, _ = _gpu_sdpa(
        query,
        key,
        value,
        attn_mask=attn_mask,
        is_causal=is_causal,
        scale=scale,
        _kv_group=_kv_group,
        need_lse=False,
    )
    return out


def _scaled_dot_product_flash_attention_for_cpu(
    query: AlloyBuffer,
    key: AlloyBuffer,
    value: AlloyBuffer,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    attn_mask: AlloyBuffer | None = None,
    scale: float | None = None,
    _kv_group: int = 1,
) -> tuple[AlloyBuffer, AlloyBuffer]:
    if dropout_p != 0.0:
        raise NotImplementedError("Alloy backend only supports dropout_p=0 for inference")
    return _gpu_sdpa(
        query,
        key,
        value,
        attn_mask=attn_mask,
        is_causal=is_causal,
        scale=scale,
        _kv_group=_kv_group,
    )


def _sdpa_shape_key(query: AlloyBuffer, key: AlloyBuffer) -> tuple[int, ...]:
    """Stable key identifying an SDPA call across fwd/bwd."""
    return tuple(query.shape) + tuple(key.shape)


def _has_bias_add_in_producer_chain(buf: AlloyBuffer, max_depth: int = 12) -> bool:
    """True if `buf`'s lazy producer chain contains a gemm + 1D-bias add."""
    visited: set[int] = set()
    frontier: list[tuple[LazyOp, int]] = []
    if buf._producer is not None:
        frontier.append((buf._producer, 0))
    while frontier:
        op, depth = frontier.pop()
        if id(op) in visited or depth > max_depth:
            continue
        visited.add(id(op))
        if op.kernel.name == "add":
            sizes = []
            for param_name, buffer_arg in op.buffer_args:
                if param_name in op.kernel._output_params:
                    continue
                sizes.append(buffer_arg.size)
            if len(sizes) == 2 and min(sizes) > 0 and max(sizes) // min(sizes) >= 16:
                return True
        for producer in op.input_producers.values():
            if producer is not None:
                frontier.append((producer, depth + 1))
    return False


def _gpu_sdpa_backward(
    grad_out: AlloyBuffer,
    query: AlloyBuffer,
    key: AlloyBuffer,
    value: AlloyBuffer,
    out: AlloyBuffer,
    logsumexp: AlloyBuffer,
    *,
    attn_mask: AlloyBuffer | None = None,
    is_causal: bool = False,
    scale: float | None = None,
) -> tuple[AlloyBuffer, AlloyBuffer, AlloyBuffer]:
    q_info = _parse_bhsd(query)
    k_info = _parse_bhsd(key)
    v_info = _parse_bhsd(value)
    go_info = _parse_bhsd(grad_out)
    out_info = _parse_bhsd(out)
    if q_info is None or k_info is None or v_info is None or go_info is None or out_info is None:
        raise ValueError("Invalid buffer shapes for SDPA backward")

    batch, heads, q_len, head_dim, q_off, q_bs, q_hs, q_ss = q_info
    k_batch, k_heads, kv_len, k_dim, k_off, k_bs, k_hs, k_ss = k_info
    v_batch, v_heads, v_len, v_dim, v_off, v_bs, v_hs, v_ss = v_info
    go_batch, go_heads, go_len, go_dim, go_off, go_bs, go_hs, go_ss = go_info
    out_batch, out_heads, out_len, out_dim, out_off, out_bs, out_hs, out_ss = out_info

    if batch != k_batch or batch != v_batch or batch != go_batch or batch != out_batch:
        raise ValueError("Batch dimensions do not match in SDPA backward")
    if heads != go_heads or heads != out_heads or q_len != go_len or q_len != out_len:
        raise ValueError("Query/output shapes do not match in SDPA backward")
    if head_dim != k_dim or head_dim != v_dim or head_dim != go_dim or head_dim != out_dim:
        raise ValueError("Head dimensions do not match in SDPA backward")
    if kv_len != v_len or k_heads != v_heads:
        raise ValueError("Key/value shapes do not match in SDPA backward")

    if heads != k_heads:
        if heads % k_heads != 0:
            raise ValueError("Query heads must be divisible by KV heads in SDPA backward")
        kv_group = heads // k_heads
    else:
        kv_group = 1

    if logsumexp.size != batch * heads * q_len:
        raise ValueError("logsumexp shape does not match SDPA backward inputs")

    custom_scale = None
    default_scale = 1.0 / math.sqrt(head_dim)
    if scale is not None and not math.isclose(
        float(scale), default_scale, rel_tol=1e-6, abs_tol=1e-6
    ):
        custom_scale = float(scale)

    mask_arr = None
    if attn_mask is not None:
        mask_arr = _reshape_attention_mask(
            attn_mask,
            batch=batch,
            heads=heads,
            q_len=q_len,
            k_len=kv_len,
        )
        if mask_arr is None:
            raise ValueError("Invalid attention mask in SDPA backward")

    q_flat = _root_flat_buf(query)
    k_flat = _root_flat_buf(key)
    v_flat = _root_flat_buf(value)
    go_flat = _root_flat_buf(grad_out)
    out_flat = _root_flat_buf(out)
    lse_flat = _root_flat_buf(logsumexp)

    dq_elements = batch * heads * q_len * head_dim
    dk_elements = batch * k_heads * kv_len * head_dim
    dv_elements = batch * k_heads * kv_len * head_dim
    dq_root = _alloc_aligned((dq_elements,), query.dtype)
    dk_root = _alloc_aligned((dk_elements,), key.dtype)
    dv_root = _alloc_aligned((dv_elements,), value.dtype)

    block = _attention_block_size(head_dim, backward=True, seq_len=min(q_len, kv_len))
    total_q_heads = batch * heads
    total_kv_heads = batch * k_heads
    stride_kwargs_no_block: AttentionConstexprs = {
        "BH": total_q_heads,
        "BH_KV": total_kv_heads,
        "HEADS_PER_BATCH": heads,
        "KV_HEADS_PER_BATCH": k_heads,
        "SEQ_LEN": q_len,
        "KV_LEN": kv_len,
        "HEAD_DIM": head_dim,
        "GO_OFFSET": go_off,
        "GO_BATCH_STRIDE": go_bs,
        "GO_HEAD_STRIDE": go_hs,
        "GO_SEQ_STRIDE": go_ss,
        "Q_OFFSET": q_off,
        "Q_BATCH_STRIDE": q_bs,
        "Q_HEAD_STRIDE": q_hs,
        "Q_SEQ_STRIDE": q_ss,
        "K_OFFSET": k_off,
        "K_BATCH_STRIDE": k_bs,
        "K_HEAD_STRIDE": k_hs,
        "K_SEQ_STRIDE": k_ss,
        "V_OFFSET": v_off,
        "V_BATCH_STRIDE": v_bs,
        "V_HEAD_STRIDE": v_hs,
        "V_SEQ_STRIDE": v_ss,
        "DQ_OFFSET": 0,
        "DQ_BATCH_STRIDE": heads * q_len * head_dim,
        "DQ_HEAD_STRIDE": head_dim,
        "DQ_SEQ_STRIDE": heads * head_dim,
        "DK_OFFSET": 0,
        "DK_BATCH_STRIDE": k_heads * kv_len * head_dim,
        "DK_HEAD_STRIDE": head_dim,
        "DK_SEQ_STRIDE": k_heads * head_dim,
        "DV_OFFSET": 0,
        "DV_BATCH_STRIDE": k_heads * kv_len * head_dim,
        "DV_HEAD_STRIDE": head_dim,
        "DV_SEQ_STRIDE": k_heads * head_dim,
        "causal": 1 if is_causal else 0,
        "KV_GROUP": kv_group,
        "CUSTOM_SCALE": custom_scale or 0,
    }

    delta = _alloc_aligned((total_q_heads * q_len,), float32)
    delta_grid = ((q_len + block - 1) // block, total_q_heads)
    _ATTENTION_DELTA_KERNEL[delta_grid](
        go_flat,
        out_flat,
        delta,
        BH=total_q_heads,
        HEADS_PER_BATCH=heads,
        SEQ_LEN=q_len,
        HEAD_DIM=head_dim,
        GO_OFFSET=go_off,
        GO_BATCH_STRIDE=go_bs,
        GO_HEAD_STRIDE=go_hs,
        GO_SEQ_STRIDE=go_ss,
        O_OFFSET=out_off,
        O_BATCH_STRIDE=out_bs,
        O_HEAD_STRIDE=out_hs,
        O_SEQ_STRIDE=out_ss,
        BLOCK_M=block,
    )

    high_precision = 1 if _sdpa_shape_key(query, key) in _kbias_cache else 0
    base_bwd_kwargs: AttentionConstexprs = {
        **stride_kwargs_no_block,
        "HIGH_PRECISION": high_precision,
    }

    if mask_arr is None:
        dq_kernel = _ATTENTION_BACKWARD_DQ_KERNEL
        dkdv_kernel = _ATTENTION_BACKWARD_DKDV_KERNEL
        dq_buf_args = [
            ("dO", go_flat),
            ("Q", q_flat),
            ("K", k_flat),
            ("V", v_flat),
            ("LogSumExp", lse_flat),
            ("Delta", delta),
            ("dQ", dq_root),
        ]
        dkdv_buf_args = [
            ("dO", go_flat),
            ("Q", q_flat),
            ("K", k_flat),
            ("V", v_flat),
            ("LogSumExp", lse_flat),
            ("Delta", delta),
            ("dK", dk_root),
            ("dV", dv_root),
        ]
    else:
        dq_kernel = _ATTENTION_BACKWARD_DQ_MASKED_KERNEL
        dkdv_kernel = _ATTENTION_BACKWARD_DKDV_MASKED_KERNEL
        dq_buf_args = [
            ("dO", go_flat),
            ("Q", q_flat),
            ("K", k_flat),
            ("V", v_flat),
            ("LogSumExp", lse_flat),
            ("Mask", mask_arr),
            ("Delta", delta),
            ("dQ", dq_root),
        ]
        dkdv_buf_args = [
            ("dO", go_flat),
            ("Q", q_flat),
            ("K", k_flat),
            ("V", v_flat),
            ("LogSumExp", lse_flat),
            ("Mask", mask_arr),
            ("Delta", delta),
            ("dK", dk_root),
            ("dV", dv_root),
        ]

    dq_bm, dq_bn = _resolved_blocks_for_attention(
        dq_kernel, base_bwd_kwargs, dq_buf_args, fallback_block=block
    )
    dkdv_bm, dkdv_bn = _resolved_blocks_for_attention(
        dkdv_kernel, base_bwd_kwargs, dkdv_buf_args, fallback_block=block
    )

    dq_grid = ((q_len + dq_bm - 1) // dq_bm, total_q_heads)
    dkdv_grid = ((kv_len + dkdv_bn - 1) // dkdv_bn, total_kv_heads)
    dq_kwargs: AttentionConstexprs = {**base_bwd_kwargs, "BLOCK_M": dq_bm, "BLOCK_N": dq_bn}
    dkdv_kwargs: AttentionConstexprs = {
        **base_bwd_kwargs,
        "BLOCK_M": dkdv_bm,
        "BLOCK_N": dkdv_bn,
    }

    if mask_arr is None:
        dq_result = dq_kernel[dq_grid](
            go_flat, q_flat, k_flat, v_flat, lse_flat, delta, dq_root, **dq_kwargs
        )
        dk_result, dv_result = cast(
            tuple[AlloyBuffer, AlloyBuffer],
            dkdv_kernel[dkdv_grid](
                go_flat,
                q_flat,
                k_flat,
                v_flat,
                lse_flat,
                delta,
                dk_root,
                dv_root,
                **dkdv_kwargs,
            ),
        )
    else:
        dq_result = dq_kernel[dq_grid](
            go_flat, q_flat, k_flat, v_flat, lse_flat, mask_arr, delta, dq_root, **dq_kwargs
        )
        dk_result, dv_result = cast(
            tuple[AlloyBuffer, AlloyBuffer],
            dkdv_kernel[dkdv_grid](
                go_flat,
                q_flat,
                k_flat,
                v_flat,
                lse_flat,
                mask_arr,
                delta,
                dk_root,
                dv_root,
                **dkdv_kwargs,
            ),
        )

    dq_out = dq_result.reshape((batch, q_len, heads, head_dim)).transpose(1, 2)
    dk_out = dk_result.reshape((batch, kv_len, k_heads, head_dim)).transpose(1, 2)
    dv_out = dv_result.reshape((batch, kv_len, k_heads, head_dim)).transpose(1, 2)

    return dq_out, dk_out, dv_out


def _scaled_dot_product_flash_attention_for_cpu_backward(
    grad_out: AlloyBuffer,
    query: AlloyBuffer,
    key: AlloyBuffer,
    value: AlloyBuffer,
    out: AlloyBuffer,
    logsumexp: AlloyBuffer,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    attn_mask: AlloyBuffer | None = None,
    scale: float | None = None,
) -> tuple[AlloyBuffer, AlloyBuffer, AlloyBuffer]:
    if dropout_p != 0.0:
        raise NotImplementedError("Alloy backend only supports dropout_p=0 for SDPA backward")
    return _gpu_sdpa_backward(
        grad_out,
        query,
        key,
        value,
        out,
        logsumexp,
        attn_mask=attn_mask,
        is_causal=is_causal,
        scale=scale,
    )


def _attention_cache_handler(
    q: AlloyBuffer,
    new_k: AlloyBuffer,
    new_v: AlloyBuffer,
    cache_pos: AlloyBuffer,
    k_cache: AlloyBuffer,
    v_cache: AlloyBuffer,
    scale: float,
    sliding_window: int = 0,
    write_kv: bool = True,
    last_real: AlloyBuffer | None = None,
) -> AlloyBuffer:
    """Unified cache-attention dispatcher. Picks the kernel path from the
    runtime query length so the model body carries no seq_len branch (Dynamo
    traces decode + prefill as one graph; alloy records a separate plan per
    concrete seq_len at run-0, so each path keeps its own optimal kernels):

      seq_len == 1            -> flash-decode (split-K)         [decode]
      2 <= seq_len <= MAX_VK  -> multi-token verify             [spec-decode]
      seq_len  > MAX_VK       -> strided runtime-pos prefill    [prefill]

    `scale` is forwarded to every path. The decode/verify kernels default to
    1/sqrt(head_dim) when it matches (gemma3/qwen/llama), but gemma4 uses
    scaling=1.0, so the explicit scale must reach them. Mirrors the in-handler
    seq_len branch `linear_attention_update` already uses for DeltaNet."""
    seq_len = q.shape[2]
    if seq_len == 1:
        return _attention_kv_update_handler(
            q, new_k, new_v, cache_pos, k_cache, v_cache,
            scale, sliding_window=sliding_window, write_kv=write_kv,
        )
    if seq_len <= _MAX_VERIFY_K:
        # The multi-token verify path has no read-only variant (it fuses the
        # KV write into the attention kernel); read-only is decode + prefill only.
        if not write_kv:
            raise ValueError(f"write_kv=False unsupported on the verify path (seq_len={seq_len})")
        # Verify rows route to the tiled prefill kernel (KV write + strided
        # runtime-pos MMA, split-K at depth), NOT the vector multi kernel:
        # the vector path's 1-simdgroup serial scan is depth-toxic in the
        # production schedule — 5.4ms/layer at 12K ctx vs ~0.1ms bandwidth
        # bound (the spec depth-decay root cause; it profiles flat in
        # isolation because its slice re-reads hit SLC, so only the
        # group-pipelined profile sees it). The draft's bidir block
        # attention keeps the vector multi path (bidir has no prefill
        # equivalent).
        return _attention_prefill_warm_handler(
            q, new_k, new_v, cache_pos, k_cache, v_cache,
            scale, sliding_window=sliding_window,
        )
    return _attention_prefill_warm_handler(
        q, new_k, new_v, cache_pos, k_cache, v_cache, scale,
        sliding_window=sliding_window, write_kv=write_kv, last_real=last_real,
    )


def _attention_kv_update_handler(
    q: AlloyBuffer,
    new_k: AlloyBuffer,
    new_v: AlloyBuffer,
    cache_pos: AlloyBuffer,
    k_cache: AlloyBuffer,
    v_cache: AlloyBuffer,
    scale: float = -1.0,
    sliding_window: int = 0,
    write_kv: bool = True,
) -> AlloyBuffer:
    q_shape = q.shape
    if len(q_shape) != 4:
        raise ValueError(f"attention_kv_update expects 4D Q, got {len(q_shape)}D")
    batch, heads, seq_len, head_dim = q_shape
    _, kv_heads, _, _ = new_k.shape
    kv_group = heads // kv_heads
    q = _q_to_cache_dtype(q, k_cache)  # f16 MMA on the padded-MMA decode-split path

    # Honor a non-default attention scale (gemma4 uses scaling=1.0, not
    # 1/sqrt(head_dim)). CUSTOM_SCALE=0 keeps the kernel's 1/sqrt(d) default —
    # which equals the passed scale for gemma3/qwen/llama, so they are unchanged.
    default_scale = 1.0 / math.sqrt(head_dim)
    if scale is None or scale <= 0 or math.isclose(
        float(scale), default_scale, rel_tol=1e-6, abs_tol=1e-6
    ):
        custom_scale = 0
    else:
        custom_scale = float(scale)

    kv_len = _attention_kv_update_static_kv_len(k_cache.shape)

    q_itemsize = q._dtype.itemsize
    q_strides = tuple(stride // q_itemsize for stride in q._strides)
    q_offset = q._offset // q_itemsize
    q_base = _root_flat_buf(q)

    nk_itemsize = new_k._dtype.itemsize
    nk_strides = tuple(stride // nk_itemsize for stride in new_k._strides)
    nk_offset = new_k._offset // nk_itemsize
    nk_base = _root_flat_buf(new_k)

    nv_itemsize = new_v._dtype.itemsize
    nv_strides = tuple(stride // nv_itemsize for stride in new_v._strides)
    nv_offset = new_v._offset // nv_itemsize
    nv_base = _root_flat_buf(new_v)

    kc_itemsize = k_cache._dtype.itemsize
    kc_strides = tuple(stride // kc_itemsize for stride in k_cache._strides)
    kc_base = _root_flat_buf(k_cache)
    vc_itemsize = v_cache._dtype.itemsize
    vc_strides = tuple(stride // vc_itemsize for stride in v_cache._strides)
    vc_base = _root_flat_buf(v_cache)

    pos_buf = cache_pos
    if pos_buf._dtype.itemsize != 4:
        pos_buf = _to_copy(pos_buf, dtype=torch.int32)

    total_heads = batch * heads
    out = _alloc_scratch((batch * seq_len * heads, head_dim), q.dtype)

    # Vector-decode path supports PER_LANE ∈ {2, 4, 8, 16} → HEAD_DIM ∈ {64, 128, 256, 512}.
    _vector_decode_head_dims = (64, 128, 256, 512)
    if (
        seq_len == 1
        and kv_len >= 256
        and head_dim in _vector_decode_head_dims
    ):
        # M=1 vector path — no MMA, no shmem in the K-loop. 32-thread TGs
        # (1 simdgroup each), each lane owns PER_LANE=HEAD_DIM/32 dims.
        # Vec4 fast path for PER_LANE=4 (HEAD_DIM=128 — qwen3, llama
        # 3.x 3B/8B); scalar path for PER_LANE=2 (HEAD_DIM=64 — llama
        # 3.2 1B); dual-vec4 path for PER_LANE=8 (HEAD_DIM=256 — gemma3);
        # quad-vec4 path for PER_LANE=16 (HEAD_DIM=512 — gemma4 global layers).
        # 1-simdgroup TG needs ~8× more splits than the padded-MMA path to
        # match its GPU occupancy.
        splits = _choose_flash_decoding_splits(
            total_heads, kv_len, simdgroups_per_tg=1
        )
    else:
        splits = (
            _choose_flash_decoding_splits(total_heads, kv_len, simdgroups_per_tg=8)
            if seq_len == 1
            else 1
        )
    use_vector = (
        seq_len == 1
        and splits > 1
        and kv_len >= 256
        and head_dim in _vector_decode_head_dims
    )
    # Read-only attend (write_kv=False) is only implemented on the vector path
    # (the gemma4 KV-shared global layers, head_dim 512). Every other decode
    # path writes the cache unconditionally, so fail loudly rather than silently
    # corrupting the source cache if read-only ever reaches one.
    if not write_kv and not use_vector:
        raise ValueError(
            "write_kv=False requires the vector-decode path "
            f"(got seq_len={seq_len}, splits={splits}, kv_len={kv_len}, head_dim={head_dim})"
        )
    if use_vector:
        partial_o = _alloc_scratch((total_heads, splits, head_dim), q.dtype)
        partial_lse = _alloc_scratch((total_heads, splits), q.dtype)
        _ATTENTION_DECODE_VECTOR_SPLIT_KERNEL[(total_heads, splits)](
            q_base,
            nk_base,
            nv_base,
            pos_buf,
            kc_base,
            vc_base,
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
            VC_HEAD_STRIDE=vc_strides[1],
            VC_SEQ_STRIDE=vc_strides[2],
            KV_GROUP=kv_group,
            SPLITS=splits,
            SLIDING_WINDOW=sliding_window,
            CUSTOM_SCALE=custom_scale,
            WRITE_KV=1 if write_kv else 0,
        )
        # Split-parallel combine when there are enough splits to fill it (SPLITS
        # is a power of two, so >=32 ⇒ multiple of 32, the par kernel's lane
        # tiling); otherwise the serial one-TG-per-head combine.
        if splits >= 32:
            result = _ATTENTION_DECODE_COMBINE_VECTOR_PAR_KERNEL[(total_heads, head_dim // 4)](
                partial_o,
                partial_lse,
                out,
                BH=total_heads,
                HEADS_PER_BATCH=heads,
                HEAD_DIM=head_dim,
                SPLITS=splits,
            )
        else:
            result = _ATTENTION_DECODE_COMBINE_VECTOR_KERNEL[(total_heads,)](
                partial_o,
                partial_lse,
                out,
                BH=total_heads,
                HEADS_PER_BATCH=heads,
                HEAD_DIM=head_dim,
                SPLITS=splits,
            )
        return result.reshape((batch, seq_len, heads, head_dim)).transpose(0, 2, 1, 3)

    if seq_len == 1 and splits > 1 and kv_len >= 256:
        # Flash Decoding split-KV: BH alone underfills the GPU on small
        # models. Split the KV axis into SPLITS chunks to launch
        # BH × SPLITS threadgroups; a tiny combine kernel reduces the
        # partials per head. Only worth it when the combine dispatch cost
        # is amortized by enough split parallelism — gated to splits > 1.
        block_m = 8
        partial_o = _alloc_scratch((total_heads, splits, block_m, head_dim), q.dtype)
        partial_lse = _alloc_scratch((total_heads, splits, block_m), q.dtype)
        _ATTENTION_KV_UPDATE_SPLIT_KERNEL[(total_heads, splits)](
            q_base,
            nk_base,
            nv_base,
            pos_buf,
            kc_base,
            vc_base,
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
            VC_HEAD_STRIDE=vc_strides[1],
            VC_SEQ_STRIDE=vc_strides[2],
            KV_GROUP=kv_group,
            SPLITS=splits,
            BLOCK_M=block_m,
            SLIDING_WINDOW=sliding_window,
            CUSTOM_SCALE=custom_scale,
        )
        result = _ATTENTION_DECODE_COMBINE_KERNEL[(total_heads,)](
            partial_o,
            partial_lse,
            out,
            BH=total_heads,
            HEADS_PER_BATCH=heads,
            HEAD_DIM=head_dim,
            SPLITS=splits,
            BLOCK_M=block_m,
        )
        return result.reshape((batch, seq_len, heads, head_dim)).transpose(0, 2, 1, 3)

    grid = ((seq_len + 31) // 32, total_heads)
    result = _ATTENTION_KV_UPDATE_KERNEL[grid](
        q_base,
        nk_base,
        nv_base,
        pos_buf,
        kc_base,
        vc_base,
        out,
        BH=total_heads,
        HEADS_PER_BATCH=heads,
        SEQ_LEN=seq_len,
        HEAD_DIM=head_dim,
        Q_OFFSET=q_offset,
        Q_BATCH_STRIDE=q_strides[0],
        Q_HEAD_STRIDE=q_strides[1],
        Q_SEQ_STRIDE=q_strides[2],
        NK_OFFSET=nk_offset,
        NK_HEAD_STRIDE=nk_strides[1],
        NV_OFFSET=nv_offset,
        NV_HEAD_STRIDE=nv_strides[1],
        KC_HEAD_STRIDE=kc_strides[1],
        KC_SEQ_STRIDE=kc_strides[2],
        VC_HEAD_STRIDE=vc_strides[1],
        VC_SEQ_STRIDE=vc_strides[2],
        KV_GROUP=kv_group,
        KV_LEN=kv_len,
        causal=1,
        SLIDING_WINDOW=sliding_window,
        CUSTOM_SCALE=custom_scale,
    )
    return result.reshape((batch, seq_len, heads, head_dim)).transpose(0, 2, 1, 3)


def _attention_kv_update_multi_bidir_handler(
    q: AlloyBuffer,
    new_k: AlloyBuffer,
    new_v: AlloyBuffer,
    cache_pos: AlloyBuffer,
    k_cache: AlloyBuffer,
    v_cache: AlloyBuffer,
    scale: float = -1.0,
) -> AlloyBuffer:
    """DFlash draft block attention. Fused KV write of
    the block rows + attention where every row sees the whole block plus the
    full context KV."""
    return _attention_kv_update_multi_handler(
        q, new_k, new_v, cache_pos, k_cache, v_cache, scale,
        sliding_window=0, bidir_block=True,
    )


def _spec_kv_write_handler(
    k: AlloyBuffer,
    v: AlloyBuffer,
    cache_pos: AlloyBuffer,
    k_cache: AlloyBuffer,
    v_cache: AlloyBuffer,
) -> AlloyBuffer:
    """Write-only KV row store for the DFlash observe/fusion plan: M rows of
    (B, KV_H, M, D) k/v land in the (B, KV_H, S, D) caches at
    [cache_pos, cache_pos+M). Returns k unchanged (keeps the dispatch live in
    the lazy collector; the cache mutation is the effect). The cache writes
    are semantically side effects nothing in-plan reads - register them as
    extern roots or the collector DCEs the dispatch (the gemma4 ring-write
    precedent)."""
    # Register the ROOT buffers (the kernel writes roots; materialize_many
    # on a view does not chase root writes — the gemma4 ring-write precedent).
    note_extern_kv_write(k_cache)
    note_extern_kv_write(v_cache)
    _, kv_heads, seq_len, head_dim = k.shape
    nk_itemsize = k._dtype.itemsize
    nk_strides = tuple(s // nk_itemsize for s in k._strides)
    nk_offset = k._offset // nk_itemsize
    nv_itemsize = v._dtype.itemsize
    nv_strides = tuple(s // nv_itemsize for s in v._strides)
    nv_offset = v._offset // nv_itemsize
    kc_itemsize = k_cache._dtype.itemsize
    kc_strides = tuple(s // kc_itemsize for s in k_cache._strides)
    vc_itemsize = v_cache._dtype.itemsize
    vc_strides = tuple(s // vc_itemsize for s in v_cache._strides)
    pos_buf = cache_pos
    if pos_buf._dtype.itemsize != 4:
        pos_buf = _to_copy(pos_buf, dtype=torch.int32)
    _ATTENTION_KV_WRITE_KERNEL[(seq_len, kv_heads)](
        _root_flat_buf(k),
        _root_flat_buf(v),
        pos_buf,
        _no_bound_buf(),  # unused at SLIDING_WINDOW=0
        k_cache,
        v_cache,
        HEADS_PER_BATCH=kv_heads,
        HEAD_DIM=head_dim,
        K_INPUT=seq_len,
        NK_OFFSET=nk_offset,
        NK_HEAD_STRIDE=nk_strides[1],
        NK_SEQ_STRIDE=nk_strides[2],
        NV_OFFSET=nv_offset,
        NV_HEAD_STRIDE=nv_strides[1],
        NV_SEQ_STRIDE=nv_strides[2],
        KC_HEAD_STRIDE=kc_strides[1],
        KC_SEQ_STRIDE=kc_strides[2],
        VC_HEAD_STRIDE=vc_strides[1],
        VC_SEQ_STRIDE=vc_strides[2],
        SLIDING_WINDOW=0,
    )
    return k


def _attention_kv_update_multi_handler(
    q: AlloyBuffer,
    new_k: AlloyBuffer,
    new_v: AlloyBuffer,
    cache_pos: AlloyBuffer,
    k_cache: AlloyBuffer,
    v_cache: AlloyBuffer,
    scale: float = -1.0,
    sliding_window: int = 0,
    bidir_block: bool = False,
) -> AlloyBuffer:
    """Multi-token decode attention with fused KV cache update.

    For mid-decode multi-token forwards (e.g. speculative-decode verify).
    Q has shape (B, H, K_INPUT, D) where K_INPUT > 1; cache_pos is a scalar
    (start position for the K_INPUT new K/V writes). Each query at
    cache_pos+i attends causally to KV positions [0, cache_pos+i+1).

    `bidir_block` lifts the per-row causal bound to the whole new block
    (every row attends [0, cache_pos+K_INPUT)) — the DFlash draft's block
    attention. Vector path only (HEAD_DIM 128/256/512).
    """
    q_shape = q.shape
    if len(q_shape) != 4:
        raise ValueError(f"attention_kv_update_multi expects 4D Q, got {len(q_shape)}D")
    batch, heads, seq_len, head_dim = q_shape
    q = _q_to_cache_dtype(q, k_cache)  # f16 MMA on the spec-verify multi-token path
    _, kv_heads, _, _ = new_k.shape
    kv_group = heads // kv_heads

    # Honor a non-default attention scale (gemma4 scaling=1.0); see
    # _attention_kv_update_handler. CUSTOM_SCALE=0 keeps the 1/sqrt(d) default.
    default_scale = 1.0 / math.sqrt(head_dim)
    if scale is None or scale <= 0 or math.isclose(
        float(scale), default_scale, rel_tol=1e-6, abs_tol=1e-6
    ):
        custom_scale = 0
    else:
        custom_scale = float(scale)

    kv_len = _attention_kv_update_static_kv_len(k_cache.shape)
    K_INPUT = seq_len
    # BLOCK_M padded to next multiple of 8 (simdgroup-friendly).
    block_m = max(8, ((K_INPUT + 7) // 8) * 8)

    q_itemsize = q._dtype.itemsize
    q_strides = tuple(stride // q_itemsize for stride in q._strides)
    q_offset = q._offset // q_itemsize
    q_base = _root_flat_buf(q)

    nk_itemsize = new_k._dtype.itemsize
    nk_strides = tuple(stride // nk_itemsize for stride in new_k._strides)
    nk_offset = new_k._offset // nk_itemsize
    nk_base = _root_flat_buf(new_k)

    nv_itemsize = new_v._dtype.itemsize
    nv_strides = tuple(stride // nv_itemsize for stride in new_v._strides)
    nv_offset = new_v._offset // nv_itemsize
    nv_base = _root_flat_buf(new_v)

    kc_itemsize = k_cache._dtype.itemsize
    kc_strides = tuple(stride // kc_itemsize for stride in k_cache._strides)
    kc_base = _root_flat_buf(k_cache)
    vc_itemsize = v_cache._dtype.itemsize
    vc_strides = tuple(stride // vc_itemsize for stride in v_cache._strides)
    vc_base = _root_flat_buf(v_cache)

    pos_buf = cache_pos
    if pos_buf._dtype.itemsize != 4:
        pos_buf = _to_copy(pos_buf, dtype=torch.int32)

    total_heads = batch * heads
    out = _alloc_scratch((batch * seq_len * heads, head_dim), q.dtype)

    # Vector multi-token path for the vec4 head dims (128/256/512, PER_LANE 4/8/16):
    # 1-simdgroup 32-thread TGs that read K/V once per position and dot every query
    # row (one simd_reduce per row), simdgroups_per_tg=1 → cap-128 splits like the
    # M=1 decode. ~10× the padded-MMA path (BLOCK_M=8 for ≤8 real rows, cap-64
    # splits), which stays the fallback for HEAD_DIM 64 or a tiny cache.
    if head_dim in (128, 256, 512) and kv_len >= 256:
        splits = _choose_flash_decoding_splits(total_heads, kv_len, simdgroups_per_tg=1)
        if splits < 1:
            splits = 1
        # Row-major partial (BH, K_INPUT, SPLITS, D) so the combine threads by
        # HEAD_DIM, not the row count — see attention_decode_combine_vector_multi.
        partial_o = _alloc_scratch((total_heads, K_INPUT, splits, head_dim), q.dtype)
        partial_lse = _alloc_scratch((total_heads, K_INPUT, splits), q.dtype)
        _ATTENTION_KV_UPDATE_VECTOR_SPLIT_MULTI_KERNEL[(total_heads, splits)](
            q_base,
            nk_base,
            nv_base,
            pos_buf,
            kc_base,
            vc_base,
            partial_o,
            partial_lse,
            BH=total_heads,
            HEADS_PER_BATCH=heads,
            HEAD_DIM=head_dim,
            Q_OFFSET=q_offset,
            Q_BATCH_STRIDE=q_strides[0],
            Q_HEAD_STRIDE=q_strides[1],
            Q_SEQ_STRIDE=q_strides[2],
            NK_OFFSET=nk_offset,
            NK_HEAD_STRIDE=nk_strides[1],
            NK_SEQ_STRIDE=nk_strides[2],
            NV_OFFSET=nv_offset,
            NV_HEAD_STRIDE=nv_strides[1],
            NV_SEQ_STRIDE=nv_strides[2],
            KC_HEAD_STRIDE=kc_strides[1],
            KC_SEQ_STRIDE=kc_strides[2],
            VC_HEAD_STRIDE=vc_strides[1],
            VC_SEQ_STRIDE=vc_strides[2],
            KV_GROUP=kv_group,
            K_INPUT=K_INPUT,
            SPLITS=splits,
            SLIDING_WINDOW=sliding_window,
            CUSTOM_SCALE=custom_scale,
            BIDIR_BLOCK=1 if bidir_block else 0,
        )
        result = _ATTENTION_DECODE_COMBINE_VECTOR_MULTI_KERNEL[(total_heads, K_INPUT)](
            partial_o,
            partial_lse,
            out,
            BH=total_heads,
            HEADS_PER_BATCH=heads,
            HEAD_DIM=head_dim,
            SPLITS=splits,
            K_INPUT=K_INPUT,
        )
        return result.reshape((batch, seq_len, heads, head_dim)).transpose(0, 2, 1, 3)

    # Padded-MMA fallback (HEAD_DIM 64, or kv_len < 256).
    if bidir_block:
        raise ValueError(
            "attention_kv_update_multi_bidir requires the vector path "
            f"(HEAD_DIM in 128/256/512 and kv_len >= 256; got HEAD_DIM={head_dim}, "
            f"kv_len={kv_len})"
        )
    splits = _choose_flash_decoding_splits(total_heads, kv_len)
    if splits < 1:
        splits = 1
    partial_o = _alloc_scratch((total_heads, splits, block_m, head_dim), q.dtype)
    partial_lse = _alloc_scratch((total_heads, splits, block_m), q.dtype)

    _ATTENTION_KV_UPDATE_SPLIT_MULTI_KERNEL[(total_heads, splits)](
        q_base,
        nk_base,
        nv_base,
        pos_buf,
        kc_base,
        vc_base,
        partial_o,
        partial_lse,
        BH=total_heads,
        HEADS_PER_BATCH=heads,
        HEAD_DIM=head_dim,
        Q_OFFSET=q_offset,
        Q_BATCH_STRIDE=q_strides[0],
        Q_HEAD_STRIDE=q_strides[1],
        Q_SEQ_STRIDE=q_strides[2],
        NK_OFFSET=nk_offset,
        NK_HEAD_STRIDE=nk_strides[1],
        NK_SEQ_STRIDE=nk_strides[2],
        NV_OFFSET=nv_offset,
        NV_HEAD_STRIDE=nv_strides[1],
        NV_SEQ_STRIDE=nv_strides[2],
        KC_HEAD_STRIDE=kc_strides[1],
        KC_SEQ_STRIDE=kc_strides[2],
        VC_HEAD_STRIDE=vc_strides[1],
        VC_SEQ_STRIDE=vc_strides[2],
        KV_GROUP=kv_group,
        K_INPUT=K_INPUT,
        BLOCK_M=block_m,
        SPLITS=splits,
        SLIDING_WINDOW=sliding_window,
        CUSTOM_SCALE=custom_scale,
    )
    result = _ATTENTION_DECODE_COMBINE_MULTI_KERNEL[(total_heads, K_INPUT)](
        partial_o,
        partial_lse,
        out,
        BH=total_heads,
        HEADS_PER_BATCH=heads,
        HEAD_DIM=head_dim,
        SPLITS=splits,
        K_INPUT=K_INPUT,
        BLOCK_M=block_m,
    )
    return result.reshape((batch, seq_len, heads, head_dim)).transpose(0, 2, 1, 3)


def _attention_prefill_cold_handler(
    q: AlloyBuffer,
    new_k: AlloyBuffer,
    new_v: AlloyBuffer,
    cache_pos: AlloyBuffer,
    k_cache: AlloyBuffer,
    v_cache: AlloyBuffer,
    scale: float,
    sliding_window: int = 0,
    last_real: AlloyBuffer | None = None,
) -> AlloyBuffer:
    """Cold prefill with fused KV write and the regular cold SDPA kernel.

    The write and read both use the root cache buffers. That keeps the lazy
    mutation dependency visible to plan construction while still baking
    Q_START_POS=0 and KV_LEN=seq_len into `attention_strided`.
    """
    q_shape = q.shape
    if len(q_shape) != 4:
        raise ValueError(f"attention_prefill_cold expects 4D Q, got {len(q_shape)}D")
    batch, heads, seq_len, head_dim = q_shape
    _, kv_heads, _, _ = new_k.shape
    kv_group = heads // kv_heads
    q = _q_to_cache_dtype(q, k_cache)  # f16 MMA on the cold-prefill strided path

    q_itemsize = q._dtype.itemsize
    q_strides = tuple(stride // q_itemsize for stride in q._strides)
    q_offset = q._offset // q_itemsize
    q_base = _root_flat_buf(q)

    nk_itemsize = new_k._dtype.itemsize
    nk_strides = tuple(stride // nk_itemsize for stride in new_k._strides)
    nk_offset = new_k._offset // nk_itemsize
    nk_base = _root_flat_buf(new_k)

    nv_itemsize = new_v._dtype.itemsize
    nv_strides = tuple(stride // nv_itemsize for stride in new_v._strides)
    nv_offset = new_v._offset // nv_itemsize
    nv_base = _root_flat_buf(new_v)

    kc_itemsize = k_cache._dtype.itemsize
    kc_strides = tuple(stride // kc_itemsize for stride in k_cache._strides)
    kc_base = _root_flat_buf(k_cache)
    vc_itemsize = v_cache._dtype.itemsize
    vc_strides = tuple(stride // vc_itemsize for stride in v_cache._strides)
    vc_base = _root_flat_buf(v_cache)

    pos_buf = cache_pos
    if pos_buf._dtype.itemsize != 4:
        pos_buf = _to_copy(pos_buf, dtype=torch.int32)

    _ATTENTION_KV_WRITE_KERNEL[(seq_len, kv_heads)](
        nk_base,
        nv_base,
        pos_buf,
        last_real if last_real is not None else _no_bound_buf(),
        kc_base,
        vc_base,
        HEADS_PER_BATCH=heads,
        HEAD_DIM=head_dim,
        K_INPUT=seq_len,
        NK_OFFSET=nk_offset,
        NK_HEAD_STRIDE=nk_strides[1],
        NK_SEQ_STRIDE=nk_strides[2],
        NV_OFFSET=nv_offset,
        NV_HEAD_STRIDE=nv_strides[1],
        NV_SEQ_STRIDE=nv_strides[2],
        KC_HEAD_STRIDE=kc_strides[1],
        KC_SEQ_STRIDE=kc_strides[2],
        VC_HEAD_STRIDE=vc_strides[1],
        VC_SEQ_STRIDE=vc_strides[2],
        SLIDING_WINDOW=sliding_window,
    )

    # Sliding-window prefill > SW needs a linear (non-wrap) K/V buffer to
    # attend against. The cache state after the kv_write above only holds
    # the last SW positions in modular slot order, so per-Q-row attention
    # against the cache reads overwritten slots for early positions.
    # Pay one extra fp16 kv_write into a contiguous temp buffer here; the
    # cache write above still runs so the post-prefill cache state matches
    # HF for the next decode / warm-prefill turn.
    use_temp_kv = sliding_window > 0 and seq_len > sliding_window
    if use_temp_kv:
        # The attend below reads the linear temp copy, so nothing in-plan
        # reads the ring write above — register it as a side-effect root or
        # the collector DCEs it and the post-prefill ring stays EMPTY (gemma4
        # sliding layers: decode against zero windows, garbage output).
        note_extern_kv_write(kc_base)
        note_extern_kv_write(vc_base)
        cache_dtype = k_cache._dtype
        temp_k = _alloc_scratch(
            (batch * kv_heads * seq_len * head_dim,), cache_dtype
        )
        temp_v = _alloc_scratch(
            (batch * kv_heads * seq_len * head_dim,), cache_dtype
        )
        tk_head_stride = seq_len * head_dim
        tk_seq_stride = head_dim
        _ATTENTION_KV_WRITE_KERNEL[(seq_len, kv_heads)](
            nk_base,
            nv_base,
            pos_buf,
            _no_bound_buf(),  # unused at SLIDING_WINDOW=0 (linear temp)
            temp_k,
            temp_v,
            HEADS_PER_BATCH=heads,
            HEAD_DIM=head_dim,
            K_INPUT=seq_len,
            NK_OFFSET=nk_offset,
            NK_HEAD_STRIDE=nk_strides[1],
            NK_SEQ_STRIDE=nk_strides[2],
            NV_OFFSET=nv_offset,
            NV_HEAD_STRIDE=nv_strides[1],
            NV_SEQ_STRIDE=nv_strides[2],
            KC_HEAD_STRIDE=tk_head_stride,
            KC_SEQ_STRIDE=tk_seq_stride,
            VC_HEAD_STRIDE=tk_head_stride,
            VC_SEQ_STRIDE=tk_seq_stride,
            SLIDING_WINDOW=0,
        )

    total_heads = batch * heads
    out = _alloc_scratch((batch * seq_len * heads * head_dim,), q.dtype)
    block = _attention_block_size(
        head_dim,
        seq_len=seq_len,
        input_dtype=q._dtype.to_torch_dtype() if hasattr(q._dtype, "to_torch_dtype") else None,
    )
    default_scale = 1.0 / math.sqrt(head_dim)
    custom_scale = (
        float(scale)
        if scale > 0 and not math.isclose(float(scale), default_scale, rel_tol=1e-6, abs_tol=1e-6)
        else 0
    )
    # K/V source: cache for the common path; linear temp for sliding-window
    # prefill > SW. Strides differ between the two layouts; K_WRAP=0 always
    # because both layouts are linear w.r.t. the K-loop iteration.
    if use_temp_kv:
        k_buf = temp_k
        v_buf = temp_v
        k_head_stride = seq_len * head_dim
        k_seq_stride = head_dim
        v_head_stride = seq_len * head_dim
        v_seq_stride = head_dim
    else:
        k_buf = kc_base
        v_buf = vc_base
        k_head_stride = kc_strides[1]
        k_seq_stride = kc_strides[2]
        v_head_stride = vc_strides[1]
        v_seq_stride = vc_strides[2]
    strided_kwargs: AttentionConstexprs = {
        "BH": total_heads,
        "HEADS_PER_BATCH": heads,
        "SEQ_LEN": seq_len,
        "KV_LEN": seq_len,
        "HEAD_DIM": head_dim,
        "Q_OFFSET": q_offset,
        "Q_BATCH_STRIDE": q_strides[0],
        "Q_HEAD_STRIDE": q_strides[1],
        "Q_SEQ_STRIDE": q_strides[2],
        "K_OFFSET": 0,
        "K_BATCH_STRIDE": 0,
        "K_HEAD_STRIDE": k_head_stride,
        "K_SEQ_STRIDE": k_seq_stride,
        "V_OFFSET": 0,
        "V_BATCH_STRIDE": 0,
        "V_HEAD_STRIDE": v_head_stride,
        "V_SEQ_STRIDE": v_seq_stride,
        "causal": 1,
        "KV_GROUP": kv_group,
        "CUSTOM_SCALE": custom_scale,
        "Q_START_POS": 0,
        "SLIDING_WINDOW": sliding_window,
        "K_WRAP": 0,
    }
    block_m, block_n = _resolved_blocks_for_attention(
        _ATTENTION_STRIDED_KERNEL,
        strided_kwargs,
        [
            ("Q", q_base),
            ("K", k_buf),
            ("V", v_buf),
            ("O", out),
        ],
        fallback_block=block,
    )
    strided_kwargs["BLOCK_M"] = block_m
    strided_kwargs["BLOCK_N"] = block_n
    _ATTENTION_STRIDED_KERNEL(
        q_base,
        k_buf,
        v_buf,
        out,
        **strided_kwargs,
    )
    return out.reshape((batch, seq_len, heads, head_dim)).transpose(0, 2, 1, 3)


def _attention_prefill_warm_handler(
    q: AlloyBuffer,
    new_k: AlloyBuffer,
    new_v: AlloyBuffer,
    cache_pos: AlloyBuffer,
    k_cache: AlloyBuffer,
    v_cache: AlloyBuffer,
    scale: float,
    sliding_window: int = 0,
    last_real: AlloyBuffer | None = None,
    *,
    # keyword-only: FX calls handlers positionally per the op schema, so any
    # parameter not in the schema must not occupy a positional slot.
    write_kv: bool = True,
) -> AlloyBuffer:
    """Prefill attention with fused KV write and runtime Q_START_POS.

    Used for BOTH cold (cache_pos==0) and warm-suffix (cache_pos>0) prefill
    on full-attention layers — the patched forward routes all non-sliding
    prefill here so cold and warm share one compiled plan (`cache_pos` is a
    runtime buffer, so start_pos varies without recompiling). Q has shape
    (B, H, K_INPUT, D); `cache_pos` holds the absolute position of Q row 0.

      1. Write new_k / new_v into K_cache / V_cache at positions
         [cache_pos..cache_pos+K_INPUT) via `attention_kv_write`.
      2. Run causal attention over the full cache via the dedicated
         `attention_strided_runtime_pos` kernel, passing `cache_pos` as
         Q_START_POS_BUF so the kernel applies the absolute-position
         causal mask at runtime.

    Runtime vs constexpr Q_START_POS is GPU-free here: the causal early-exit
    clamps the K-scan to the query position regardless of KV_LEN, so reading
    the full cache costs the same as a sliced read (measured ±0% at min/p10).
    Cold and warm then collapse to one kernel specialization → one plan → one
    compile. Sliding-window prefill keeps the separate cold handler
    (its >SW case needs a linear temp-KV layout this path doesn't build).
    """
    q_shape = q.shape
    if len(q_shape) != 4:
        raise ValueError(f"attention_prefill_warm expects 4D Q, got {len(q_shape)}D")
    batch, heads, seq_len, head_dim = q_shape
    _, kv_heads, _, _ = new_k.shape
    kv_group = heads // kv_heads
    kv_len = _attention_kv_update_static_kv_len(k_cache.shape)
    K_INPUT = seq_len

    q = _q_to_cache_dtype(q, k_cache)  # f16 MMA — see _q_to_cache_dtype

    q_itemsize = q._dtype.itemsize
    q_strides = tuple(stride // q_itemsize for stride in q._strides)
    q_offset = q._offset // q_itemsize
    q_base = _root_flat_buf(q)

    nk_itemsize = new_k._dtype.itemsize
    nk_strides = tuple(stride // nk_itemsize for stride in new_k._strides)
    nk_offset = new_k._offset // nk_itemsize
    nk_base = _root_flat_buf(new_k)

    nv_itemsize = new_v._dtype.itemsize
    nv_strides = tuple(stride // nv_itemsize for stride in new_v._strides)
    nv_offset = new_v._offset // nv_itemsize
    nv_base = _root_flat_buf(new_v)

    kc_itemsize = k_cache._dtype.itemsize
    kc_strides = tuple(stride // kc_itemsize for stride in k_cache._strides)
    kc_base = _root_flat_buf(k_cache)
    vc_itemsize = v_cache._dtype.itemsize
    vc_strides = tuple(stride // vc_itemsize for stride in v_cache._strides)
    vc_base = _root_flat_buf(v_cache)

    pos_buf = cache_pos
    if pos_buf._dtype.itemsize != 4:
        pos_buf = _to_copy(pos_buf, dtype=torch.int32)

    # Phase 1: write new_k / new_v into the cache at [cache_pos..cache_pos+K_INPUT).
    # Skipped for read-only attend (write_kv=False) — gemma4's KV-shared global
    # layers own no K/V projection and attend the SOURCE layer's already-written
    # cache, so new_k/new_v are a throwaway shape-only view and must not be written.
    if write_kv:
        _ATTENTION_KV_WRITE_KERNEL[(K_INPUT, kv_heads)](
            nk_base,
            nv_base,
            pos_buf,
            last_real if last_real is not None else _no_bound_buf(),
            kc_base,
            vc_base,
            HEADS_PER_BATCH=heads,
            HEAD_DIM=head_dim,
            K_INPUT=K_INPUT,
            NK_OFFSET=nk_offset,
            NK_HEAD_STRIDE=nk_strides[1],
            NK_SEQ_STRIDE=nk_strides[2],
            NV_OFFSET=nv_offset,
            NV_HEAD_STRIDE=nv_strides[1],
            NV_SEQ_STRIDE=nv_strides[2],
            KC_HEAD_STRIDE=kc_strides[1],
            KC_SEQ_STRIDE=kc_strides[2],
            VC_HEAD_STRIDE=vc_strides[1],
            VC_SEQ_STRIDE=vc_strides[2],
            SLIDING_WINDOW=sliding_window,
        )

    # Phase 2: causal attention over the full (now-updated) cache, using
    # the runtime-Q_START_POS variant so a single compiled plan handles
    # arbitrary cache offsets across multi-turn requests.
    total_heads = batch * heads
    out = _alloc_scratch((batch * seq_len * heads * head_dim,), q.dtype)
    block = _attention_block_size(
        head_dim,
        seq_len=min(seq_len, kv_len) if kv_len > 0 else seq_len,
        input_dtype=q.dtype if hasattr(q, "dtype") else None,
    )
    default_scale = 1.0 / math.sqrt(head_dim)
    custom_scale = (
        float(scale)
        if scale > 0 and not math.isclose(float(scale), default_scale, rel_tol=1e-6, abs_tol=1e-6)
        else 0
    )
    strided_kwargs: AttentionConstexprs = {
        "BH": total_heads,
        "HEADS_PER_BATCH": heads,
        "SEQ_LEN": seq_len,
        "KV_LEN": kv_len,
        "HEAD_DIM": head_dim,
        "Q_OFFSET": q_offset,
        "Q_BATCH_STRIDE": q_strides[0],
        "Q_HEAD_STRIDE": q_strides[1],
        "Q_SEQ_STRIDE": q_strides[2],
        "K_OFFSET": 0,
        "K_BATCH_STRIDE": 0,
        "K_HEAD_STRIDE": kc_strides[1],
        "K_SEQ_STRIDE": kc_strides[2],
        "V_OFFSET": 0,
        "V_BATCH_STRIDE": 0,
        "V_HEAD_STRIDE": vc_strides[1],
        "V_SEQ_STRIDE": vc_strides[2],
        "causal": 1,
        "KV_GROUP": kv_group,
        "CUSTOM_SCALE": custom_scale,
        "SLIDING_WINDOW": sliding_window,
        # K_WRAP=sliding_window routes K/V reads through the circular
        # cache (slot = pos % SW). Required once the conversation's
        # cache_pos+seq_len exceeds SW; harmless when it doesn't (slot
        # = pos). Two cooperating pieces:
        #   - msl/memory.py: cooperative-load emitter applies the
        #     modulus per-thread as `(base + _r) % SW` (the bare
        #     `(j+rn) % SW` would compile to `((j+tid) % SW) + _r`,
        #     wrong granularity).
        #   - attention_strided_runtime_pos: K_WRAP>0 path drops the
        #     `j+rn < N_KV` row mask (every wrapped slot is in-bounds)
        #     and removes the N_KV_BLOCKS clamp on end_kv_blocks so
        #     the loop reaches positions >= SW.
        "K_WRAP": sliding_window,
    }
    block_m, block_n = _resolved_blocks_for_attention(
        _ATTENTION_STRIDED_RUNTIME_POS_KERNEL,
        strided_kwargs,
        [
            ("Q", q_base),
            ("K", kc_base),
            ("V", vc_base),
            ("Q_START_POS_BUF", pos_buf),
            ("O", out),
        ],
        fallback_block=block,
    )
    strided_kwargs["BLOCK_M"] = block_m
    strided_kwargs["BLOCK_N"] = block_n

    # Split-K for deep prefill: the single-pass grid (q_blocks, total_heads) is
    # ~one wave on the M4 Max for small-head models, so a deep K-scan leaves a
    # grid tail. Fanning the scan across SPLITS (program_id(2)) fills the machine;
    # the per-row combine reduces the partials ~free. Gated to plain causal,
    # head_dim ≤ 256: sliding-window stays single-pass (a window spans only part of
    # the K range, so splitting the full range wastes work on out-of-window splits,
    # and the window already bounds the scan). head_dim 512 (gemma4's global layers)
    # is EXCLUDED: the split kernel's (BLOCK_M, 512) f32 o-accumulator + barriers
    # race across simdgroups there — non-deterministic KV (Δ≈20 run-to-run, bisected
    # to 6fe6bbc "enable split-K prefill for head_dim 512"), so those layers take the
    # deterministic single-pass path (the head_dim-512 split-K gain was small anyway;
    # the partial f32 write is 4× heavier at 512). `splits` is shape-derived, so a
    # given plan always takes the same branch and compiles consistently.
    splits = 1
    if sliding_window == 0 and head_dim <= 256 and not compile_window.grid_shrink_active():
        q_blocks = (seq_len + block_m - 1) // block_m
        if seq_len <= _MAX_VERIFY_K:
            # Verify-width dispatch (spec decode): q_blocks == 1, so the
            # q-axis contributes no parallelism and the static split slicing
            # (slice = KV_LEN/SPLITS) decides everything — only
            # ceil(depth/slice) splits carry work at runtime. The prefill
            # chooser's cap of 8 gives 32K-position slices on a 262K-native
            # cache: at 12K depth a single split per head scans serially
            # (~3.4ms/layer, the spec depth-decay residual). Slice at the
            # decode path's ~2K granularity instead so working-split count
            # tracks depth.
            splits = max(1, min(128, kv_len // 2048))
        else:
            splits = _choose_prefill_splits(q_blocks, total_heads, kv_len, block_n)
    if splits > 1:
        f32 = from_torch_dtype(torch.float32)
        partial_o = _alloc_scratch((splits, total_heads, seq_len, head_dim), f32)
        partial_lse = _alloc_scratch((splits, total_heads, seq_len), f32)
        split_kwargs: AttentionConstexprs = {**strided_kwargs, "SPLITS": splits}
        sblock_m, sblock_n = _resolved_blocks_for_attention(
            _ATTENTION_STRIDED_RUNTIME_POS_SPLIT_KERNEL,
            split_kwargs,
            [
                ("Q", q_base),
                ("K", kc_base),
                ("V", vc_base),
                ("Q_START_POS_BUF", pos_buf),
                ("partial_O", partial_o),
                ("partial_lse", partial_lse),
            ],
            fallback_block=block,
        )
        split_kwargs["BLOCK_M"] = sblock_m
        split_kwargs["BLOCK_N"] = sblock_n
        split_q_blocks = (seq_len + sblock_m - 1) // sblock_m
        _ATTENTION_STRIDED_RUNTIME_POS_SPLIT_KERNEL[(split_q_blocks, total_heads, splits)](
            q_base,
            kc_base,
            vc_base,
            pos_buf,
            partial_o,
            partial_lse,
            **split_kwargs,
        )
        _ATTENTION_COMBINE_SPLITS_KERNEL[(seq_len, total_heads)](
            partial_o,
            partial_lse,
            out,
            BH=total_heads,
            HEADS_PER_BATCH=heads,
            SEQ_LEN=seq_len,
            HEAD_DIM=head_dim,
            SPLITS=splits,
        )
    else:
        _ATTENTION_STRIDED_RUNTIME_POS_KERNEL(
            q_base,
            kc_base,
            vc_base,
            pos_buf,
            out,
            **strided_kwargs,
        )
    return out.reshape((batch, seq_len, heads, head_dim)).transpose(0, 2, 1, 3)


_APPLE9_MAX_CORES = 40


def _choose_flash_decoding_splits(
    total_heads: int,
    kv_len: int,
    *,
    simdgroups_per_tg: int = 8,
) -> int:
    """Pick a power-of-two SPLITS for flash decoding.

    Each split adds GPU parallelism (good) and one extra combine dispatch
    per layer (CPU/encoder overhead). Target ~2x oversubscription of the
    M4 Max so smaller-head models (Qwen3 16 heads, Llama 32 heads) keep
    all cores busy at decode time. Decode wall time scales as
    kv_len / splits in the absence of bandwidth bottlenecks, so more
    splits flattens the growth-with-context curve in multi-turn chat.

    `simdgroups_per_tg` is the per-TG simdgroup count of the kernel that
    will receive these splits. The padded-MMA path uses 256-thread TGs
    (8 simdgroups each) and benefits from ~2 TGs per core (target_tgs=80).
    The PER_LANE=2 / PER_LANE=4 vector path uses 32-thread TGs
    (1 simdgroup each); to reach the same compute occupancy each "vector
    TG" must be matched ~8× by more splits, so target_tgs scales with
    `simdgroups_per_tg`'s inverse.
    """
    # The 1-simdgroup vector path pairs with a split-PARALLEL combine
    # (attention_decode_combine_vector_par), so extra splits are nearly free —
    # oversubscribe harder (4x) and lift the cap to 128 there to shrink the
    # split kernel's grid-tail drain (measured ~3% on qwen2.5:3b at 16k depth).
    # The padded-MMA path keeps its serial one-TG-per-head combine: 2x / cap 64.
    oversub = 4 if simdgroups_per_tg == 1 else 2
    cap = 128 if simdgroups_per_tg == 1 else 64
    target_simdgroups = _APPLE9_MAX_CORES * 8 * oversub
    target_tgs = max(1, target_simdgroups // max(1, simdgroups_per_tg))
    raw = max(1, target_tgs // max(1, total_heads))
    splits = 1
    while splits < raw and splits < cap:
        splits *= 2
    # Each split must have enough KV positions to be worth a dispatch.
    # 32 is the smallest chunk that keeps the simdgroup pipeline full.
    while splits > 1 and kv_len // splits < 32:
        splits //= 2
    return splits


def _choose_prefill_splits(
    q_blocks: int, total_heads: int, kv_len: int, block_n: int
) -> int:
    """Pick a power-of-two SPLITS for split-K PREFILL attention.

    The single-pass prefill grid is (q_blocks, total_heads). At a 128-token
    chunk that is ~128 TGs for a 16-head model — barely one wave on the M4
    Max's 40 cores, so a deep-context K-scan leaves a grid tail: occupancy
    falls to ~10% as the last few TGs finish alone. Splitting the K-scan along
    program_id(2) multiplies the TG count by SPLITS, filling the machine and
    shortening the tail (measured: split-only 1.30x GPU at depth 16k for 3b).
    The combine that reduces the partials is ~free (per-row register
    accumulator, ~7us), so the only added cost is the per-split partial write.

    Split only when (a) the grid underfills AND (b) the scan is long enough
    that each split still carries real work — splitting a short scan just adds
    fixed per-TG overhead (Q load, partial write) with nothing to parallelize.
    Returns 1 (single-pass, no partials) when not worth it.
    """
    existing_tgs = q_blocks * total_heads
    kv_blocks = (kv_len + block_n - 1) // block_n
    # Target ~4 waves of 256-thread TGs to bury the tail (≈40 cores × ~3
    # resident × ~4). 3b's 128-TG grid lands on SPLITS=4 (→512 TGs), its
    # measured optimum; a 32-head model (256 TGs) lands on SPLITS=2.
    target_tgs = _APPLE9_MAX_CORES * 12
    raw = max(1, target_tgs // max(1, existing_tgs))
    splits = 1
    while splits < raw and splits < 8:
        splits *= 2
    # Each split needs a deep-enough sub-scan to amortize its partial write;
    # below ~8 BLOCK_N tiles (~1024 positions at BLOCK_N=128) the scan is too
    # short to bother, so shallow/cold prefill stays single-pass.
    while splits > 1 and kv_blocks // splits < 8:
        splits //= 2
    return splits
