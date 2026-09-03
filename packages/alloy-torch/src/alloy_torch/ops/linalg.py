"""Linear algebra handlers for torch op lowering."""

from collections.abc import Callable, Sequence
import ctypes
from dataclasses import dataclass

import numpy as np

from alloy._compiler.dtypes import float32
from alloy._dispatch.buf_utils import _alloc_aligned, _alloc_scratch
from alloy._dispatch.dispatch import _engine
from alloy._dispatch.kernel import KernelFunction
from alloy._runtime.alloy_buffer import AlloyBuffer
from alloy._runtime.tune_configs import resolve_config
from alloy.std.elementwise import gelu_tanh_mul, silu_mul
from alloy.std.gemm import dot, dot_transpose_lhs, dot_transpose_rhs, dot_transpose_rhs_silu
from alloy.std.quant import (
    dot_dequant,
    dot_dequant_silu,
    dot_q4_k,
    dot_q4_k_silu_v2,
    dot_q4_k_gelu_v2,
    dot_q4_k_silu_v2_rows,
    dot_q4_k_v2,
    dot_q4_k_v2_rows,
    dot_mlx_q4,
    dot_mlx_q4_silu_v2,
    dot_mlx_q4_v2,
    dot_mlx_q4_v2_rows,
    dot_q5_0,
    dot_q5_0_v2,
    dot_q6_k,
    dot_q6_k_v2,
    dot_q6_k_v2_rows,
    dot_q8_0,
    dot_q8_0_silu_v2,
    dot_q8_0_silu_v2_rows,
    dot_q8_0_v2,
    dot_q8_0_v2_rows,
    embedding_q4_k,
    embedding_mlx_q4,
    embedding_q5_0,
    embedding_q6_k,
    embedding_q8_0,
)
from alloy_torch.mode import is_training_mode_enabled
from alloy_torch.ops.concat import _cat
from alloy_torch.ops.creation import _full
from alloy_torch.ops.views import _select_int

_mm_batched_cache: dict[tuple[int, ...], tuple[AlloyBuffer, list[int]]] = {}

BatchedMMBiases = Sequence[AlloyBuffer | None] | None


def _transpose_base_2d(value: AlloyBuffer) -> AlloyBuffer | None:
    """Detect if value is a transposed 2D view and return the untransposed base."""
    if len(value._shape) != 2:
        return None
    rows, cols = value._shape
    itemsize = value._dtype.itemsize
    if value._strides != (itemsize, rows * itemsize):
        return None

    base_buf = AlloyBuffer(
        value._parent_handle,
        value._offset,
        value._shape,
        value._strides,
        value._dtype,
        raw_ptr=value._raw_ptr,
        total_nbytes=value._total_nbytes,
    )
    base_buf.reinterpret((cols, rows), (rows * itemsize, itemsize))
    return value._view_of(base_buf)


def _ensure_zero_offset(x: AlloyBuffer) -> AlloyBuffer:
    """Ensure GEMM inputs bind at allocation base with contiguous layout."""
    if x._offset != 0 or not x.is_contiguous():
        return x.contiguous()
    return x


def _addmm(
    bias: AlloyBuffer,
    mat1: AlloyBuffer,
    mat2: AlloyBuffer,
    *,
    beta: float = 1.0,
    alpha: float = 1.0,
) -> AlloyBuffer:
    out = _mm(mat1, mat2)
    if alpha != 1:
        out = out * alpha
    if beta == 0:
        return out
    if beta != 1:
        bias = bias * beta
    return out + bias


def _bmm(a: AlloyBuffer, b: AlloyBuffer) -> AlloyBuffer:
    if a.ndim != 3 or b.ndim != 3:
        raise NotImplementedError(f"aten.bmm requires rank-3 tensors, got {a.shape} and {b.shape}")
    batch, rows, reduction = a.shape
    rhs_reduction, cols = b.shape[1], b.shape[2]
    if reduction == 1 and rhs_reduction == 1:
        return a * b

    a = a.contiguous()
    b = b.contiguous()
    slices: list[AlloyBuffer] = []
    for batch_index in range(batch):
        a_slice = _select_int(a, 0, batch_index)
        b_slice = _select_int(b, 0, batch_index)
        out_slice = _mm(a_slice, b_slice)
        slices.append(out_slice.reshape(1, rows, cols))

    while len(slices) > 1:
        pairs: list[AlloyBuffer] = []
        for index in range(0, len(slices) - 1, 2):
            pairs.append(_cat((slices[index], slices[index + 1]), dim=0))
        if len(slices) % 2 == 1:
            pairs.append(slices[-1])
        slices = pairs
    return slices[0]


def _read_int_scalar(buf: AlloyBuffer) -> int:
    return int(float(buf.read_scalar()))


def _alloy_batched_mm_handler(
    x: AlloyBuffer,
    weights: Sequence[AlloyBuffer],
    biases: BatchedMMBiases = None,
) -> tuple[AlloyBuffer, ...]:
    weight_bufs: list[AlloyBuffer] = []
    for weight in weights:
        base = _transpose_base_2d(weight)
        if base is None:
            return tuple(_mm(x, weight) for weight in weights)
        weight_bufs.append(base)

    sizes = [weight_buf.shape[0] for weight_buf in weight_bufs]
    cache_key = tuple(weight_buf.data_ptr for weight_buf in weight_bufs)
    cached = _mm_batched_cache.get(cache_key)
    if cached is not None:
        concat_buf, sizes = cached
    else:
        reduction = weight_bufs[0].shape[1]
        concat_buf = _alloc_aligned((sum(sizes), reduction), weight_bufs[0].dtype)

        itemsize = concat_buf._dtype.itemsize
        byte_offset = 0
        for weight_buf in weight_bufs:
            row_bytes = weight_buf.shape[0] * reduction * itemsize
            ctypes.memmove(concat_buf.data_ptr + byte_offset, weight_buf.data_ptr, row_bytes)
            byte_offset += row_bytes
        _mm_batched_cache[cache_key] = (concat_buf, sizes)

    out = _alloc_scratch((x.shape[0], concat_buf.shape[0]), x.dtype)
    result = dot_transpose_rhs(x, concat_buf, out)
    total_cols = sum(sizes)

    if biases is not None and any(bias is not None for bias in biases):
        bias_dtype = next(bias._dtype for bias in biases if bias is not None)
        concat_bias = _alloc_aligned((total_cols,), bias_dtype)
        ctypes.memset(concat_bias.data_ptr, 0, total_cols * bias_dtype.itemsize)
        byte_offset = 0
        for index, size in enumerate(sizes):
            bias = biases[index] if index < len(biases) else None
            if bias is not None:
                nbytes = bias.size * bias._dtype.itemsize
                ctypes.memmove(concat_bias.data_ptr + byte_offset, bias.data_ptr, nbytes)
            byte_offset += size * bias_dtype.itemsize
        result = result + concat_bias

    result_2d = result.reshape((x.shape[0], total_cols))
    outputs: list[AlloyBuffer] = []
    col = 0
    for size in sizes:
        output = result_2d.slice(1, col, col + size)
        output._materializer = result._materializer
        outputs.append(output)
        col += size
    return tuple(outputs)


def _alloy_dequant_mm_handler(
    activations: AlloyBuffer,
    packed_weights: AlloyBuffer,
    scales: AlloyBuffer,
    zeros: AlloyBuffer,
    group_size: int,
) -> AlloyBuffer:
    out = _alloc_scratch((activations.shape[0], packed_weights.shape[0]), activations.dtype)
    return dot_dequant(
        activations,
        packed_weights,
        scales,
        out,
        GROUP_SIZE=group_size,
        BITS=4,
        ZERO_POINT=_read_int_scalar(zeros),
    )


def _alloy_batched_dequant_mm_handler(
    activations: AlloyBuffer,
    packed_weights: Sequence[AlloyBuffer],
    scales_list: Sequence[AlloyBuffer],
    zeros_list: Sequence[AlloyBuffer],
    group_size: int,
) -> tuple[AlloyBuffer, ...]:
    sizes = [weight.shape[0] for weight in packed_weights]
    packed_cols = packed_weights[0].shape[1]
    groups = scales_list[0].shape[1]
    total_cols = sum(sizes)

    concat_packed = _alloc_aligned((total_cols, packed_cols), packed_weights[0].dtype)
    byte_offset = 0
    for weight in packed_weights:
        row_bytes = weight.shape[0] * packed_cols * concat_packed._dtype.itemsize
        ctypes.memmove(concat_packed.data_ptr + byte_offset, weight.data_ptr, row_bytes)
        byte_offset += row_bytes

    concat_scales = _alloc_aligned((total_cols, groups), scales_list[0].dtype)
    byte_offset = 0
    for scale in scales_list:
        row_bytes = scale.shape[0] * groups * scale._dtype.itemsize
        ctypes.memmove(concat_scales.data_ptr + byte_offset, scale.data_ptr, row_bytes)
        byte_offset += row_bytes

    result = dot_dequant(
        activations,
        concat_packed,
        concat_scales,
        GROUP_SIZE=group_size,
        BITS=4,
        ZERO_POINT=8,
    )

    result_2d = result.reshape((activations.shape[0], total_cols))
    outputs: list[AlloyBuffer] = []
    col = 0
    for size in sizes:
        output = result_2d.slice(1, col, col + size)
        output._materializer = result._materializer
        outputs.append(output)
        col += size
    return tuple(outputs)


def _alloy_dot_silu_handler(
    x: AlloyBuffer, gate_weight: AlloyBuffer, up_weight: AlloyBuffer
) -> AlloyBuffer:
    gate_base = _transpose_base_2d(gate_weight)
    up_base = _transpose_base_2d(up_weight)
    if gate_base is None or up_base is None:
        gate_out = _mm(x, gate_weight)
        up_out = _mm(x, up_weight)
        # silu_mul = silu(g)*up, NOT sigmoid(g)*up — keep the `g *` factor.
        fused_out = _alloc_scratch(gate_out.shape, gate_out.dtype)
        return silu_mul(gate_out, up_out, fused_out, N=gate_out.size).reshape(gate_out.shape)

    rows = x.shape[0]
    gate_cols = gate_base.shape[0]
    out = _alloc_scratch((rows, gate_cols), x.dtype)
    if rows <= 8:
        # GEMV regime: the fused kernel's shared activation load wins.
        return dot_transpose_rhs_silu(x, gate_base, up_base, out, N_GATE=gate_cols)
    # Tiled: two singles + silu_mul beat the fused dual-accumulator kernel
    # (1.05-1.07x at M=512/4096 — 3-shmem-tile + 2× acc-register occupancy cost).
    # The explicit silu_mul out keeps the handler's output dtype (auto-alloc
    # would promote f16 to f32).
    gate_out = _alloc_scratch((rows, gate_cols), x.dtype)
    silu_out = _alloc_scratch((rows, gate_cols), x.dtype)
    g = dot_transpose_rhs(x, gate_base, gate_out)
    u = dot_transpose_rhs(x, up_base, out)
    return silu_mul(g, u, silu_out, N=rows * gate_cols).reshape((rows, gate_cols))


def _alloy_dequant_silu_handler(
    x: AlloyBuffer,
    gate_packed: AlloyBuffer,
    gate_scales: AlloyBuffer,
    up_packed: AlloyBuffer,
    up_scales: AlloyBuffer,
    zeros: AlloyBuffer,
    group_size: int,
) -> AlloyBuffer:
    rows = x.shape[0]
    gate_cols = gate_packed.shape[0]
    out = _alloc_scratch((rows, gate_cols), x.dtype)
    return dot_dequant_silu(
        x,
        gate_packed,
        gate_scales,
        up_packed,
        up_scales,
        out,
        N_GATE=gate_cols,
        GROUP_SIZE=group_size,
        BITS=4,
        ZERO_POINT=_read_int_scalar(zeros),
    )


Launch = Callable[[int], tuple[tuple[int, ...], dict[str, int]]]
Weights = tuple[AlloyBuffer, ...]


def nr0_for(N: int) -> int:
    return 4 if N % 4 == 0 else (2 if N % 2 == 0 else 1)


def launch_nr0(N: int) -> tuple[tuple[int, ...], dict[str, int]]:
    nr0 = nr0_for(N)
    return (N // nr0,), {"NR0": nr0}


def launch_q5_0(N: int) -> tuple[tuple[int, ...], dict[str, int]]:
    nr0 = nr0_for(N)
    return (N // nr0, 1), {"NR0": nr0}


def launch_q4_k(N: int) -> tuple[tuple[int, ...], dict[str, int]]:
    """(NSG, NR0) for the native Q4_K decode matvec — N % (NSG*NR0) must be 0.
    Default NR0=2/NSG=2 (M4 Max)."""
    nsg, nr0 = (2, 2) if N % 4 == 0 else ((2, 1) if N % 2 == 0 else (1, 1))
    return (N // (nsg * nr0),), {"NSG": nsg, "NR0": nr0}


def launch_q4_k_fused(N: int) -> tuple[tuple[int, ...], dict[str, int]]:
    """(NSG, NR0) for the native Q4_K gate+up fused matvec (silu/gelu) —
    default NR0=1/NSG=2."""
    nsg, nr0 = (2, 1) if N % 2 == 0 else (1, 1)
    return (N // (nsg * nr0),), {"NSG": nsg, "NR0": nr0}


def launch_q6_k(N: int) -> tuple[tuple[int, ...], dict[str, int]]:
    """NUM_SPLITS=1: split-K regresses full-model decode — the shmem-barrier
    K-reduction dominates at the vocab-wide lm_head."""
    return (N,), {"NUM_SPLITS": 1}


def launch_mlx_q4(N: int) -> tuple[tuple[int, ...], dict[str, int]]:
    """(NUM_SPLITS, NR0) for the affine int4 decode matvec — N % NR0 == 0. NR0=2
    beats 4 (fewer threadgroups hurt occupancy more than the extra activation reuse
    helps); NUM_SPLITS>1 regresses (the shmem-barrier K-reduction)."""
    return (N // (2 if N % 2 == 0 else 1),), {"NUM_SPLITS": 1, "NR0": 2 if N % 2 == 0 else 1}


def launch_per_column(N: int) -> tuple[tuple[int, ...], dict[str, int]]:
    return (N,), {}


def no_constexprs(weights: Weights) -> dict[str, int]:
    return {}


def mlx_group_size(weights: Weights) -> dict[str, int]:
    """group_size from the qweight/scales column counts."""
    return {"GROUP_SIZE": (weights[0].shape[1] * 2) // weights[1].shape[1]}


def rows_always(weights: Weights) -> bool:
    return True


def rows_k_multiple_of_256(weights: Weights) -> bool:
    return weights[0].shape[1] % 256 == 0


@dataclass(frozen=True)
class FusedQuantGemm:
    matvec: KernelFunction
    matvec_launch: Launch
    rows: KernelFunction | None
    epilogue: KernelFunction


@dataclass(frozen=True)
class QuantGemm:
    """One quantized weight format's kernel set. `n_index` is the weight arg
    whose dim 0 is N; `rows` is the M<=4 GEMV (None = straight to tiled)."""

    n_index: int
    tiled: KernelFunction
    matvec: KernelFunction
    matvec_launch: Launch
    rows: KernelFunction | None
    rows_ok: Callable[[Weights], bool]
    embedding: KernelFunction
    embed_width: Callable[[Weights], int]
    constexprs: Callable[[Weights], dict[str, int]]
    fused: dict[str, FusedQuantGemm]


Q8_0 = QuantGemm(
    n_index=0,
    tiled=(dot_q8_0),
    matvec=(dot_q8_0_v2),
    matvec_launch=launch_nr0,
    rows=(dot_q8_0_v2_rows),
    rows_ok=rows_k_multiple_of_256,
    embedding=(embedding_q8_0),
    embed_width=lambda w: w[0].shape[1],
    constexprs=no_constexprs,
    fused={
        "silu": FusedQuantGemm(
            (dot_q8_0_silu_v2), launch_nr0,
            (dot_q8_0_silu_v2_rows), silu_mul,
        ),
    },
)

Q5_0 = QuantGemm(
    n_index=2,
    tiled=(dot_q5_0),
    matvec=(dot_q5_0_v2),
    matvec_launch=launch_q5_0,
    rows=None,
    rows_ok=rows_always,
    embedding=(embedding_q5_0),
    embed_width=lambda w: w[0].shape[1] * 2,
    constexprs=no_constexprs,
    fused={},
)

Q4_K = QuantGemm(
    n_index=0,
    tiled=(dot_q4_k),
    matvec=(dot_q4_k_v2),
    matvec_launch=launch_q4_k,
    rows=(dot_q4_k_v2_rows),
    rows_ok=rows_always,
    embedding=(embedding_q4_k),
    embed_width=lambda w: (w[0].shape[1] // 144) * 256,
    constexprs=no_constexprs,
    fused={
        "silu": FusedQuantGemm(
            (dot_q4_k_silu_v2), launch_q4_k_fused,
            (dot_q4_k_silu_v2_rows), silu_mul,
        ),
        "gelu": FusedQuantGemm(
            (dot_q4_k_gelu_v2), launch_q4_k_fused, None, gelu_tanh_mul,
        ),
    },
)

Q6_K = QuantGemm(
    n_index=0,
    tiled=(dot_q6_k),
    matvec=(dot_q6_k_v2),
    matvec_launch=launch_q6_k,
    rows=(dot_q6_k_v2_rows),
    rows_ok=rows_always,
    embedding=(embedding_q6_k),
    embed_width=lambda w: (w[0].shape[1] // 210) * 256,
    constexprs=no_constexprs,
    fused={},
)

MLX_Q4 = QuantGemm(
    n_index=0,
    tiled=(dot_mlx_q4),
    matvec=(dot_mlx_q4_v2),
    matvec_launch=launch_mlx_q4,
    rows=(dot_mlx_q4_v2_rows),
    rows_ok=rows_always,
    embedding=(embedding_mlx_q4),
    embed_width=lambda w: w[0].shape[1] * 2,
    constexprs=mlx_group_size,
    fused={
        "silu": FusedQuantGemm(
            (dot_mlx_q4_silu_v2), launch_per_column, None, silu_mul,
        ),
    },
)


def quant_mm(fmt: QuantGemm, activations: AlloyBuffer, *weights: AlloyBuffer) -> AlloyBuffer:
    """M == 1 is the decode matvec; M <= 4 (MTP propose) the rows GEMV, which is
    issue-bound (one program per output column) and loses to the tiled GEMM from
    the DFlash verify width (M >= 8) up."""
    M = activations.shape[0]
    N = weights[fmt.n_index].shape[0]
    out = _alloc_scratch((M, N), activations.dtype)
    extra = fmt.constexprs(weights)
    if M == 1:
        grid, cx = fmt.matvec_launch(N)
        return fmt.matvec[grid](activations, *weights, out, **extra, **cx)
    if fmt.rows is not None and M <= 4 and fmt.rows_ok(weights):
        return fmt.rows[(N,)](activations, *weights, out, **extra)
    return fmt.tiled(activations, *weights, out, **extra)


def quant_mm_fused(
    fmt: QuantGemm, variant: str, activations: AlloyBuffer, *weights: AlloyBuffer,
) -> AlloyBuffer:
    """gate+up projection with the activation epilogue. The GEMV regimes keep the
    fused kernel (the shared activation load dominates); tiled prefill runs two
    single GEMMs + the epilogue, which beats the fused dual-accumulator kernel by
    ~6-9% (3 shmem tiles + 2x the acc registers cost more occupancy than the
    shared A pass saves)."""
    fused = fmt.fused[variant]
    half = len(weights) // 2
    gate, up = weights[:half], weights[half:]
    M = activations.shape[0]
    N = gate[fmt.n_index].shape[0]
    out = _alloc_scratch((M, N), activations.dtype)
    extra = fmt.constexprs(gate)
    if M == 1:
        grid, cx = fused.matvec_launch(N)
        return fused.matvec[grid](activations, *gate, *up, out, **extra, **cx)
    if fused.rows is not None and M <= 4 and fmt.rows_ok(gate):
        return fused.rows[(N,)](activations, *gate, *up, out, **extra)
    gate_out = _alloc_scratch((M, N), activations.dtype)
    g = fmt.tiled(activations, *gate, gate_out, **extra)
    u = fmt.tiled(activations, *up, out, **extra)
    return fused.epilogue(g, u, N=M * N).reshape((M, N))


def quant_embedding(fmt: QuantGemm, input_ids: AlloyBuffer, *weights: AlloyBuffer) -> AlloyBuffer:
    index_buf = input_ids.contiguous()
    width = fmt.embed_width(weights)
    out = _alloc_scratch(input_ids.shape + (width,), float32)
    return fmt.embedding(
        index_buf, *weights, out,
        NUM_INDICES=index_buf.size, WIDTH=width, **fmt.constexprs(weights),
    )


_gguf_batched_concat_cache: dict[
    tuple[int, ...], tuple[tuple[AlloyBuffer, ...], list[int]]
] = {}


def _row_concat_buffers(
    buffer_lists: tuple[Sequence[AlloyBuffer], ...],
    sizes: list[int],
) -> tuple[AlloyBuffer, ...]:
    """Concatenate each parallel sequence of weight buffers row-wise (dim 0).

    Every buffer in `buffer_lists[k]` has shape `(rows_i, cols_k)` for some
    common `cols_k` and per-weight rows. We allocate one buffer per slot of
    shape `(sum(rows_i), cols_k)` and memcopy each source's rows in order."""
    total_rows = sum(sizes)
    concats: list[AlloyBuffer] = []
    for buffers in buffer_lists:
        prototype = buffers[0]
        row_bytes = prototype.shape[1] * prototype._dtype.itemsize
        concat = _alloc_aligned((total_rows, prototype.shape[1]), prototype.dtype)
        off = 0
        for buf in buffers:
            ctypes.memmove(concat.data_ptr + off, buf.data_ptr, buf.shape[0] * row_bytes)
            off += buf.shape[0] * row_bytes
        concats.append(concat)
    return tuple(concats)


def _slice_batched_result(
    result: AlloyBuffer, rows: int, sizes: list[int]
) -> tuple[AlloyBuffer, ...]:
    """Split the concatenated GEMM output into contiguous per-weight buffers.

    The raw slice along dim=1 produces a strided view whose row stride is
    `sum(sizes)` instead of `size_i`. Downstream HF code calls `.view()`
    on the projection output to split it into (B, S, num_heads, head_dim) —
    which silently misinterprets the strided view as contiguous when
    `size_i` happens to divide cleanly into the layout (qwen3:0.6b's
    1024-col K/V on a 2048-col concat works by coincidence) but reads
    garbage when it doesn't (qwen2.5:0.5b's 128-col K/V on a 1152-col
    concat). `.contiguous()` runs a lazy Metal-side copy so the buffer
    handed to torch matches the Meta impl's `torch.empty((rows, N_i))`
    contract without breaking the lazy pipeline.
    """
    total_cols = sum(sizes)
    result_2d = result.reshape((rows, total_cols))
    outputs: list[AlloyBuffer] = []
    col = 0
    for size in sizes:
        # Zero-copy: a decode-time slice of the batched result is a dense block
        # that aliases the result. rows > 1 (prefill) is genuinely gapped and
        # copies. Epilogue fusions that read a sibling slice (gate_up gelu*mul)
        # keep the batched dot unfused so the full output is materialized first
        # (see _resolve_extra_alloc).
        outputs.append(result_2d.slice(1, col, col + size).as_dense_view())
        col += size
    return tuple(outputs)


def _resolve_concat(
    primary: Sequence[AlloyBuffer],
    extras: tuple[Sequence[AlloyBuffer], ...],
) -> tuple[tuple[AlloyBuffer, ...], list[int]]:
    sizes = [p.shape[0] for p in primary]
    cache_key = tuple(p.data_ptr for p in primary)
    cached = _gguf_batched_concat_cache.get(cache_key)
    if cached is not None:
        return cached
    concats = _row_concat_buffers((primary,) + extras, sizes)
    _gguf_batched_concat_cache[cache_key] = (concats, sizes)
    return concats, sizes


def quant_mm_batched(
    fmt: QuantGemm, activations: AlloyBuffer, *weight_lists: Sequence[AlloyBuffer],
) -> tuple[AlloyBuffer, ...]:
    others = [i for i in range(len(weight_lists)) if i != fmt.n_index]
    concats, sizes = _resolve_concat(
        weight_lists[fmt.n_index], tuple(weight_lists[i] for i in others),
    )
    weights: list[AlloyBuffer] = list(concats)
    for i, concat in zip([fmt.n_index, *others], concats):
        weights[i] = concat
    result = quant_mm(fmt, activations, *weights)
    return _slice_batched_result(result, activations.shape[0], sizes)


def _pack_weight(w: AlloyBuffer, block_n: int, block_k: int) -> AlloyBuffer:
    """Repack (N, K) weight so each (BLOCK_N, BLOCK_K) tile is contiguous."""
    rows, cols = w.shape
    packed = _alloc_aligned((rows, cols), w.dtype)
    itemsize = w.dtype.itemsize
    src = np.frombuffer(
        (ctypes.c_uint8 * (rows * cols * itemsize)).from_address(w.data_ptr), dtype=np.uint8
    ).reshape(rows // block_n, block_n, cols // block_k, block_k * itemsize)
    dst = np.frombuffer(
        (ctypes.c_uint8 * (rows * cols * itemsize)).from_address(packed.data_ptr), dtype=np.uint8
    ).reshape(rows // block_n, cols // block_k, block_n, block_k * itemsize)
    np.copyto(dst, src.transpose(0, 2, 1, 3))
    return packed


# Only pad large M. The bounds-checked partial-M-tile path corrupts only for a
# large M inside a deep f16 graph (gemma4 vision: 2520-patch encoder, 280-token
# pooler); a small M is correct on the bounds path, so padding it is both
# unnecessary and harmful — the un-padded residual/elementwise that consumes the
# GEMM output then shape-mismatches the padded rows under epilogue fusion. Text
# never has a large non-aligned M (prefill chunks at 128, decodes at 1), so this
# is vision-only in practice.
_PAD_M_MIN = 256


def _pad_m_to_tile(buf: AlloyBuffer, m_axis: int) -> tuple[AlloyBuffer, int]:
    """Pad `buf` along `m_axis` (the GEMM's output-rows / M dimension) up to a
    multiple of 64 with zero rows; returns (padded_buf, original_M).

    Routes an M not divisible by the tile height through the fast,
    non-bounds-checked GEMM path. The masked partial-last-M-tile codegen
    corrupts f16 output inside a large dispatch graph (gemma4 vision
    o_proj/MLP at M=2520, the pooler at M=280): the kernel is correct in
    isolation and at M % 64 == 0, yet the full encoder forward emits
    all-NaN/all-zero — a memory-layout-dependent hazard in the divergent
    partial-tile path that only the fast path avoids. 64 is the largest
    BLOCK_M, so every resolved config tiles the padded M exactly. No-op for
    already-aligned M and for M < `_PAD_M_MIN`.
    """
    m = buf.shape[m_axis]
    m_pad = (-m) % 64
    if m < _PAD_M_MIN or m_pad == 0:
        return buf, m
    shp = list(buf.shape)
    shp[m_axis] = m_pad
    pad_zeros = _full(tuple(shp), 0.0, dtype=buf._dtype.to_torch_dtype())
    return _cat([buf, pad_zeros], m_axis), m


def _mm(lhs: AlloyBuffer, rhs: AlloyBuffer) -> AlloyBuffer:
    squeeze_result = False
    if lhs.ndim == 1:
        lhs = lhs.reshape((1, lhs.shape[0]))
        squeeze_result = True

    lhs_t_base = _transpose_base_2d(lhs)
    if lhs_t_base is not None and lhs._offset == 0:
        lhs_t_base = _ensure_zero_offset(lhs_t_base)
        rhs = _ensure_zero_offset(rhs)
        cols = rhs.shape[1]
        # lhs_t_base is (K, M); the M dimension is axis 1.
        lhs_t_base, orig_rows = _pad_m_to_tile(lhs_t_base, 1)
        rows = lhs_t_base.shape[1]
        out = _alloc_scratch((rows, cols), lhs.dtype)
        result = dot_transpose_lhs(lhs_t_base, rhs, out)
        if rows != orig_rows:
            result = result.slice(0, 0, orig_rows)
        if squeeze_result:
            result = result.reshape(result.shape[-1])
        return result

    lhs = _ensure_zero_offset(lhs)
    rhs_t_base = _transpose_base_2d(rhs)
    if rhs_t_base is not None:
        rhs_t_base = _ensure_zero_offset(rhs_t_base)
        lhs, orig_rows = _pad_m_to_tile(lhs, 0)
        rows, cols = lhs.shape[0], rhs_t_base.shape[0]
        reduction = lhs.shape[1]
        out = _alloc_scratch((rows, cols), lhs.dtype)
        packed: AlloyBuffer | None = None
        if rows > 1 and cols > 1 and not is_training_mode_enabled():
            key: dict[str, int] = {
                "_A_dim0": rows,
                "_A_dim1": reduction,
                "_B_T_dim0": cols,
                "_B_T_dim1": reduction,
                "_C_dim0": rows,
                "_C_dim1": cols,
            }
            config = resolve_config("dot_transpose_rhs", key)
            block_n = config.constexprs.get("BLOCK_N", 32)
            block_k = config.constexprs.get("BLOCK_K", 16)
            # Never pack under a matvec config: the kernel branches on _matvec
            # BEFORE _PACKED, so the matvec body would read the (BN,BK)-tiled
            # packed weight as row-major — silent garbage.
            if (
                not config.constexprs.get("_matvec")
                and cols % block_n == 0
                and reduction % block_k == 0
                and block_n * block_k >= 512
            ):
                packed = _pack_weight(rhs_t_base, block_n, block_k)
                _engine.untrack_alloc(packed.base_ptr)
        if packed is not None:
            result = dot_transpose_rhs(lhs, packed, out, _PACKED=1)
        elif rows == 1:
            # M=1 decode: the tiled path tiles only N (32 threadgroups for a
            # 1024-wide projection) and starves the GPU. The matvec body is one
            # program per output column → grid (N,1,1), well-occupied at bandwidth.
            result = dot_transpose_rhs(lhs, rhs_t_base, out, _matvec=1)
        else:
            result = dot_transpose_rhs(lhs, rhs_t_base, out)
        if rows != orig_rows:
            result = result.slice(0, 0, orig_rows)
    else:
        rhs = _ensure_zero_offset(rhs)
        lhs, orig_rows = _pad_m_to_tile(lhs, 0)
        rows, red = lhs.shape
        cols = rhs.shape[1]
        out = _alloc_scratch((rows, cols), lhs.dtype)
        # `dot` has no dispatch_spec, so an auto-grid launch collapses to one
        # threadgroup and leaves every tile past the first uninitialised. Launch
        # an explicit grid over the resolved config's tiling.
        cfg = resolve_config("dot", {
            "_A_dim0": rows, "_A_dim1": red, "_B_dim0": red,
            "_B_dim1": cols, "_C_dim0": rows, "_C_dim1": cols,
        }).constexprs
        bm = cfg.get("BLOCK_M", 64)
        bn = cfg.get("BLOCK_N", 64)
        bk = cfg.get("BLOCK_K", 16)
        grid = ((rows + bm - 1) // bm, (cols + bn - 1) // bn)
        result = dot[grid](lhs, rhs, out, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk)
        if rows != orig_rows:
            result = result.slice(0, 0, orig_rows)

    if squeeze_result:
        result = result.reshape(result.shape[-1])
    return result
