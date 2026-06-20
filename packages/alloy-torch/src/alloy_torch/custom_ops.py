"""Custom torch op schemas and meta kernels used by Alloy FX rewrites."""

from __future__ import annotations

from collections.abc import Sequence

import torch

_alloy_lib: torch.library.Library = torch.library.Library("alloy", "DEF")

_alloy_lib.define("rms_norm(Tensor x, Tensor weight, float eps) -> (Tensor, Tensor)")
_alloy_lib.define("rms_norm_backward(Tensor x, Tensor dy, Tensor weight, Tensor rrms) -> Tensor")
_alloy_lib.define("rope_apply(Tensor x, Tensor cos, Tensor sin) -> Tensor")
_alloy_lib.define(
    "rope_table(Tensor cache_position, Tensor inv_freq, int seq_len) -> (Tensor, Tensor)"
)
_alloy_lib.define(
    "rms_norm_rope(Tensor x, Tensor weight, Tensor cos, Tensor sin, float eps, "
    "bool cos_half=False) -> Tensor"
)
_alloy_lib.define("rope_apply_backward(Tensor dout, Tensor cos, Tensor sin) -> Tensor")
_alloy_lib.define("batched_mm(Tensor x, Tensor[] weights, Tensor[]? biases=None) -> Tensor[]")
_alloy_lib.define(
    "gemm_residual_layernorm(Tensor mat1, Tensor mat2, Tensor? bias, "
    "Tensor? residual, Tensor ln_weight, Tensor ln_bias, int[] normalized_shape, float eps) "
    "-> (Tensor, Tensor, Tensor)"
)
_alloy_lib.define(
    "gemm_residual_rmsnorm(Tensor mat1, Tensor mat2, Tensor residual, "
    "Tensor weight, float eps) -> (Tensor, Tensor, Tensor)"
)
_alloy_lib.define(
    "dequant_mm(Tensor activations, Tensor packed_weights, Tensor scales, "
    "Tensor zeros, int group_size) -> Tensor"
)
_alloy_lib.define(
    "batched_dequant_mm(Tensor activations, Tensor[] packed_weights, "
    "Tensor[] scales, Tensor[] zeros, int group_size) -> Tensor[]"
)
_alloy_lib.define("dot_silu(Tensor x, Tensor gate_weight, Tensor up_weight) -> Tensor")
_alloy_lib.define(
    "dequant_silu(Tensor x, Tensor gate_packed, Tensor gate_scales, "
    "Tensor up_packed, Tensor up_scales, Tensor zeros, int group_size) -> Tensor"
)
# GGUF-native Q4_K: weights are the raw 144-byte superblocks (N, blocks_per_row*144)
# uint8; K = (blocks.shape[1] // 144) * 256.
_alloy_lib.define(
    "gguf_q4_k_mm(Tensor activations, Tensor blocks) -> Tensor"
)
_alloy_lib.define(
    "batched_gguf_q4_k_mm(Tensor activations, Tensor[] blocks) -> Tensor[]"
)
_alloy_lib.define(
    "gguf_q4_k_silu(Tensor activations, Tensor gate_blocks, Tensor up_blocks) -> Tensor"
)
_alloy_lib.define(
    "gguf_q4_k_gelu(Tensor activations, Tensor gate_blocks, Tensor up_blocks) -> Tensor"
)
# Qwen3.5-MoE routed experts: router top-k over `router_logits`, then the
# gathered gate_up (Q4_K-native, fused) + down (Q6_K) per-expert GEMV with the
# routing weight combine. Returns the routed output (T, H); the shared expert is
# added by the caller. hidden=(T,H), router_logits=(T,num_experts).
_alloy_lib.define(
    "gguf_moe_routed(Tensor hidden, Tensor router_logits, Tensor gate_up_blocks, "
    "Tensor down_qweight, int num_experts, int top_k, int moe_intermediate) -> Tensor"
)
_alloy_lib.define(
    "gguf_q4_k_embedding(Tensor input_ids, Tensor blocks) -> Tensor"
)
# Affine int4 group quant (MLX 4-bit): qweight 2 nibbles/byte (N, K//2) uint8,
# fp16 scales+biases (N, K//group); weight = scale*q + bias.
_alloy_lib.define(
    "mlx_q4_mm(Tensor activations, Tensor qweight, Tensor scales, Tensor biases) -> Tensor"
)
_alloy_lib.define(
    "batched_mlx_q4_mm(Tensor activations, Tensor[] qweights, "
    "Tensor[] scales, Tensor[] biases) -> Tensor[]"
)
_alloy_lib.define(
    "mlx_q4_silu(Tensor activations, Tensor gate_qweight, Tensor gate_scales, "
    "Tensor gate_biases, Tensor up_qweight, Tensor up_scales, Tensor up_biases) -> Tensor"
)
_alloy_lib.define(
    "mlx_q4_embedding(Tensor input_ids, Tensor qweight, Tensor scales, Tensor biases) -> Tensor"
)
_alloy_lib.define(
    "gguf_q5_0_mm(Tensor activations, Tensor qweight, Tensor qhigh, Tensor scales) -> Tensor"
)
_alloy_lib.define(
    "gguf_q5_0_embedding(Tensor input_ids, Tensor qweight, Tensor qhigh, Tensor scales) -> Tensor"
)
_alloy_lib.define(
    "batched_gguf_q5_0_mm(Tensor activations, Tensor[] qweights, "
    "Tensor[] qhighs, Tensor[] scales) -> Tensor[]"
)
_alloy_lib.define("gguf_q6_k_mm(Tensor activations, Tensor packed_weights) -> Tensor")
_alloy_lib.define(
    "gguf_q6_k_embedding(Tensor input_ids, Tensor packed_weights) -> Tensor"
)
_alloy_lib.define(
    "batched_gguf_q6_k_mm(Tensor activations, Tensor[] packed_weights) -> Tensor[]"
)
_alloy_lib.define("gguf_q8_0_embedding(Tensor input_ids, Tensor qweight, Tensor scales) -> Tensor")
_alloy_lib.define("gguf_q8_0_mm(Tensor activations, Tensor qweight, Tensor scales) -> Tensor")
_alloy_lib.define(
    "batched_gguf_q8_0_mm(Tensor activations, Tensor[] qweights, Tensor[] scales) -> Tensor[]"
)
_alloy_lib.define(
    "gguf_q8_0_silu(Tensor activations, Tensor gate_qweight, Tensor gate_scales, "
    "Tensor up_qweight, Tensor up_scales) -> Tensor"
)
_alloy_lib.define(
    "attention_kv_update(Tensor q, Tensor new_k, Tensor new_v, "
    "Tensor cache_pos, Tensor k_cache, Tensor v_cache, "
    "float scale=-1.0, int sliding_window=0) -> Tensor"
)
_alloy_lib.define(
    "attention_kv_update_multi(Tensor q, Tensor new_k, Tensor new_v, "
    "Tensor cache_pos, Tensor k_cache, Tensor v_cache, "
    "float scale=-1.0, int sliding_window=0) -> Tensor"
)
_alloy_lib.define(
    "attention_prefill_warm(Tensor q, Tensor new_k, Tensor new_v, "
    "Tensor cache_pos, Tensor k_cache, Tensor v_cache, float scale, "
    "int sliding_window=0, Tensor? last_real=None) -> Tensor"
)
_alloy_lib.define(
    "attention_prefill_cold(Tensor q, Tensor new_k, Tensor new_v, "
    "Tensor cache_pos, Tensor k_cache, Tensor v_cache, float scale, "
    "int sliding_window=0, Tensor? last_real=None) -> Tensor"
)
# Unified cache-attention op: one FX node for decode (seq_len==1), spec-decode
# verify (seq_len<=_MAX_VERIFY_K), and prefill (seq_len>_MAX_VERIFY_K). The
# handler picks the kernel path from the runtime seq_len, so the model body has
# no seq_len branch and Dynamo traces decode + prefill as ONE graph.
_alloy_lib.define(
    "attention_cache(Tensor q, Tensor new_k, Tensor new_v, "
    "Tensor cache_pos, Tensor k_cache, Tensor v_cache, float scale, "
    "int sliding_window=0, bool write_kv=True, Tensor? last_real=None) -> Tensor"
)
# q8_0 quantized-KV decode attention: the codes/scales caches replace the fp16
# K/V caches. Cache buffers are plain Tensor inputs (alloy-owned, static-address;
# kernels mutate them GPU-side) — same contract as `attention_cache`.
_alloy_lib.define(
    "attention_cache_q8(Tensor q, Tensor new_k, Tensor new_v, "
    "Tensor cache_pos, Tensor k_codes, Tensor k_scales, "
    "Tensor v_codes, Tensor v_scales, "
    "float scale=-1.0, int sliding_window=0, bool write_kv=True, "
    "Tensor? last_real=None) -> Tensor"
)
# DFlash draft block attention: like attention_kv_update_multi but every query
# row attends the WHOLE new-token block (bidirectional within block) plus the
# full context KV — the block diffusion mask. Same fused KV write of the block
# rows at [cache_pos, cache_pos+M).
_alloy_lib.define(
    "attention_kv_update_multi_bidir(Tensor q, Tensor new_k, Tensor new_v, "
    "Tensor cache_pos, Tensor k_cache, Tensor v_cache, "
    "float scale=-1.0) -> Tensor"
)
# Write-only KV row store for the DFlash observe/fusion plan: writes M rows of
# (B, KV_H, M, D) k/v into the (B, KV_H, S, D) caches at
# [cache_pos, cache_pos+M). Returns k unchanged (a token output so the lazy
# collector keeps the dispatch live; the cache mutation is the real effect).
_alloy_lib.define(
    "spec_kv_write(Tensor k, Tensor v, Tensor cache_pos, "
    "Tensor k_cache, Tensor v_cache) -> Tensor"
)
_alloy_lib.define(
    "cross_entropy_fwd_fused(Tensor logits, Tensor labels, int ignore_index) "
    "-> (Tensor, Tensor, Tensor)"
)
# On-GPU categorical sampler — replaces the decode argmax when sampling is
# requested. `logits` is (..., V); reduces the last dim to a sampled token id
# (..., ) int64, mirroring `argmax(dim=-1)`. `position` (cache_position) is the
# RNG counter; `seed` + `params` ([temperature, top_p, top_k, min_p]) are stable.
_alloy_lib.define(
    "sample_categorical(Tensor logits, Tensor position, Tensor seed, Tensor params) -> Tensor"
)
# Qwen 3.5 GatedDeltaNet (linear-attention) layer core. Subsumes the causal
# Conv1d (with rolling state), the chunked-/recurrent- gated delta rule, and the
# RMSNormGated. conv_state and recurrent_state are marked mutable so AOT autograd
# doesn't lift the in-graph state update — same contract as `attention_kv_update`.
#
# Shape contract:
#   mixed_qkv:        (B, S, conv_dim) = (B, S, key_dim*2 + value_dim)
#   z:                (B, S, value_dim)
#   a:                (B, S, num_v_heads)
#   b:                (B, S, num_v_heads)
#   conv_state:       (B, conv_dim, conv_kernel_size) — rolling input window
#   recurrent_state:  (B, num_v_heads, head_k_dim, head_v_dim)
#   conv1d_w:         (conv_dim, 1, conv_kernel_size) — depthwise (groups=conv_dim)
#   A_log:            (num_v_heads,)
#   dt_bias:          (num_v_heads,)
#   norm_w:           (head_v_dim,)
# Output: (B, S, value_dim) — post-norm pre-out_proj
_alloy_lib.define(
    "linear_attention_update("
    "Tensor mixed_qkv, "
    "Tensor z, "
    "Tensor a, "
    "Tensor b, "
    "Tensor(c!) conv_state, "
    "Tensor(r!) recurrent_state, "
    "Tensor conv1d_w, "
    "Tensor A_log, "
    "Tensor dt_bias, "
    "Tensor norm_w, "
    "int num_k_heads, int num_v_heads, "
    "int head_k_dim, int head_v_dim, "
    "int conv_kernel_size, "
    "float norm_eps, "
    "bool has_previous_state, "
    "Tensor? real_len=None"
    ") -> Tensor"
)
_alloy_lib.define(
    "cross_entropy_bwd_fused(Tensor logits, Tensor labels, Tensor lse, "
    "Tensor n_valid, Tensor grad_loss, int ignore_index) -> Tensor"
)
# LFM2 short-conv (conv-mixer) layer core. Subsumes the causal depthwise Conv1d
# (kernel size conv_kernel_size) with rolling state. `conv_state` is marked
# mutable so AOT autograd keeps the in-graph state update. Unlike DeltaNet there
# is NO SiLU and NO recurrent rule — the conv emits the linear conv directly; the
# in_proj `B*x` gate and the post-conv `C*` gate stay in the FX graph.
#
# Shape contract:
#   bx:           (B, S, C)   — the gated `B * x` channels (C = hidden_size)
#   conv_state:   (B, C, conv_kernel_size) — rolling input window
#   conv1d_w:     (C, 1, conv_kernel_size) — depthwise (groups=C)
#   real_len:     (1,) int — last REAL position+1 in a padded prefill chunk,
#                 so conv_state is saved from real tokens (None on decode).
# Output: (B, S, C) — the conv output, pre `C*` gate and out_proj.
_alloy_lib.define(
    "short_conv_update("
    "Tensor bx, "
    "Tensor(c!) conv_state, "
    "Tensor conv1d_w, "
    "int conv_kernel_size, "
    "bool has_previous_state, "
    "Tensor? real_len=None"
    ") -> Tensor"
)


@torch.library.impl(_alloy_lib, "short_conv_update", "Meta")
def _short_conv_update_meta(
    bx: torch.Tensor,
    conv_state: torch.Tensor,
    conv1d_w: torch.Tensor,
    conv_kernel_size: int,
    has_previous_state: bool,
    real_len: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty_like(bx)


# Gated warm-decode short-conv: takes the FULL in_proj output `bcx` (B, 1, 3C)
# and does b*x -> conv -> c* in one kernel, collapsing the LFM2 conv diamond.
# Output: (B, 1, C), the post-`C*`-gate conv (feeds out_proj directly).
_alloy_lib.define(
    "short_conv_gated("
    "Tensor bcx, "
    "Tensor(c!) conv_state, "
    "Tensor conv1d_w, "
    "int conv_kernel_size, "
    "bool has_previous_state"
    ") -> Tensor"
)


@torch.library.impl(_alloy_lib, "short_conv_gated", "Meta")
def _short_conv_gated_meta(
    bcx: torch.Tensor,
    conv_state: torch.Tensor,
    conv1d_w: torch.Tensor,
    conv_kernel_size: int,
    has_previous_state: bool,
) -> torch.Tensor:
    b, s, c3 = bcx.shape
    return torch.empty((b, s, c3 // 3), dtype=bcx.dtype, device=bcx.device)


@torch.library.impl(_alloy_lib, "attention_cache_q8", "Meta")
def _attention_cache_q8_meta(
    q: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    cache_pos: torch.Tensor,
    k_codes: torch.Tensor,
    k_scales: torch.Tensor,
    v_codes: torch.Tensor,
    v_scales: torch.Tensor,
    scale: float = -1.0,
    sliding_window: int = 0,
    write_kv: bool = True,
    last_real: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty_like(q)


@torch.library.impl(_alloy_lib, "attention_kv_update", "Meta")
def _attention_kv_update_meta(
    q: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    cache_pos: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: float = -1.0,
    sliding_window: int = 0,
) -> torch.Tensor:
    return torch.empty_like(q)


@torch.library.impl(_alloy_lib, "attention_kv_update_multi", "Meta")
def _attention_kv_update_multi_meta(
    q: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    cache_pos: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: float = -1.0,
    sliding_window: int = 0,
) -> torch.Tensor:
    return torch.empty_like(q)


@torch.library.impl(_alloy_lib, "attention_kv_update_multi_bidir", "Meta")
def _attention_kv_update_multi_bidir_meta(
    q: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    cache_pos: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: float = -1.0,
) -> torch.Tensor:
    return torch.empty_like(q)


@torch.library.impl(_alloy_lib, "spec_kv_write", "Meta")
def _spec_kv_write_meta(
    k: torch.Tensor,
    v: torch.Tensor,
    cache_pos: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(k)


@torch.library.impl(_alloy_lib, "attention_prefill_warm", "Meta")
def _attention_prefill_warm_meta(
    q: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    cache_pos: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: float,
    sliding_window: int = 0,
    last_real: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty_like(q)


@torch.library.impl(_alloy_lib, "attention_prefill_cold", "Meta")
def _attention_prefill_cold_meta(
    q: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    cache_pos: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: float,
    sliding_window: int = 0,
    last_real: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty_like(q)


@torch.library.impl(_alloy_lib, "attention_cache", "Meta")
def _attention_cache_meta(
    q: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    cache_pos: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: float,
    sliding_window: int = 0,
    write_kv: bool = True,
    last_real: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty_like(q)


@torch.library.impl(_alloy_lib, "linear_attention_update", "Meta")
def _linear_attention_update_meta(
    mixed_qkv: torch.Tensor,
    z: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    conv_state: torch.Tensor,
    recurrent_state: torch.Tensor,
    conv1d_w: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_w: torch.Tensor,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_kernel_size: int,
    norm_eps: float,
    has_previous_state: bool,
    real_len: torch.Tensor | None = None,
) -> torch.Tensor:
    # Output is (B, S, value_dim); `z` already has that shape contract.
    return torch.empty_like(z)


@torch.library.impl(_alloy_lib, "cross_entropy_fwd_fused", "Meta")
def _cross_entropy_fwd_fused_meta(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid_rows = logits.shape[0] * (1 if logits.ndim == 2 else logits.shape[1])
    if labels.ndim == 1:
        valid_rows = labels.shape[0]
    loss = torch.empty((), dtype=torch.float32, device=logits.device)
    lse = torch.empty((valid_rows,), dtype=torch.float32, device=logits.device)
    n_valid = torch.empty((), dtype=torch.float32, device=logits.device)
    return loss, lse, n_valid


@torch.library.impl(_alloy_lib, "sample_categorical", "Meta")
def _sample_categorical_meta(
    logits: torch.Tensor,
    position: torch.Tensor,
    seed: torch.Tensor,
    params: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(logits.shape[:-1], dtype=torch.int64, device=logits.device)


@torch.library.impl(_alloy_lib, "cross_entropy_bwd_fused", "Meta")
def _cross_entropy_bwd_fused_meta(
    logits: torch.Tensor,
    labels: torch.Tensor,
    lse: torch.Tensor,
    n_valid: torch.Tensor,
    grad_loss: torch.Tensor,
    ignore_index: int,
) -> torch.Tensor:
    return torch.empty_like(logits)


@torch.library.impl(_alloy_lib, "gemm_residual_rmsnorm", "Meta")
def _gemm_residual_rmsnorm_meta(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out_shape = residual.shape
    rsqrt_shape = out_shape[:-1] + (1,)
    return (
        torch.empty(out_shape, dtype=residual.dtype, device=residual.device),
        torch.empty(out_shape, dtype=residual.dtype, device=residual.device),
        torch.empty(rsqrt_shape, dtype=torch.float32, device=residual.device),
    )


@torch.library.impl(_alloy_lib, "dequant_mm", "Meta")
def _dequant_mm_meta(
    activations: torch.Tensor,
    packed_weights: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    rows = activations.shape[0]
    cols = packed_weights.shape[0]
    return torch.empty((rows, cols), dtype=activations.dtype, device=activations.device)


@torch.library.impl(_alloy_lib, "batched_dequant_mm", "Meta")
def _batched_dequant_mm_meta(
    activations: torch.Tensor,
    packed_weights: Sequence[torch.Tensor],
    scales: Sequence[torch.Tensor],
    zeros: Sequence[torch.Tensor],
    group_size: int,
) -> list[torch.Tensor]:
    rows = activations.shape[0]
    return [
        torch.empty((rows, packed.shape[0]), dtype=activations.dtype, device=activations.device)
        for packed in packed_weights
    ]


@torch.library.impl(_alloy_lib, "batched_gguf_q4_k_mm", "Meta")
def _batched_gguf_q4_k_mm_meta(
    activations: torch.Tensor,
    blocks: Sequence[torch.Tensor],
) -> list[torch.Tensor]:
    rows = activations.shape[0]
    return [
        torch.empty((rows, b.shape[0]), dtype=activations.dtype, device=activations.device)
        for b in blocks
    ]


@torch.library.impl(_alloy_lib, "batched_gguf_q8_0_mm", "Meta")
def _batched_gguf_q8_0_mm_meta(
    activations: torch.Tensor,
    qweights: Sequence[torch.Tensor],
    scales: Sequence[torch.Tensor],
) -> list[torch.Tensor]:
    rows = activations.shape[0]
    return [
        torch.empty((rows, qw.shape[0]), dtype=activations.dtype, device=activations.device)
        for qw in qweights
    ]


@torch.library.impl(_alloy_lib, "batched_gguf_q5_0_mm", "Meta")
def _batched_gguf_q5_0_mm_meta(
    activations: torch.Tensor,
    qweights: Sequence[torch.Tensor],
    qhighs: Sequence[torch.Tensor],
    scales: Sequence[torch.Tensor],
) -> list[torch.Tensor]:
    rows = activations.shape[0]
    return [
        torch.empty((rows, sc.shape[0]), dtype=activations.dtype, device=activations.device)
        for sc in scales
    ]


@torch.library.impl(_alloy_lib, "batched_gguf_q6_k_mm", "Meta")
def _batched_gguf_q6_k_mm_meta(
    activations: torch.Tensor,
    packed_weights: Sequence[torch.Tensor],
) -> list[torch.Tensor]:
    rows = activations.shape[0]
    return [
        torch.empty((rows, pw.shape[0]), dtype=activations.dtype, device=activations.device)
        for pw in packed_weights
    ]


@torch.library.impl(_alloy_lib, "rms_norm", "Meta")
def _rms_norm_meta(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    rsqrt_shape = x.shape[:-1] + (1,)
    return torch.empty_like(x), torch.empty(rsqrt_shape, dtype=torch.float32, device=x.device)


@torch.library.impl(_alloy_lib, "rope_table", "Meta")
def _rope_table_meta(
    cache_position: torch.Tensor, inv_freq: torch.Tensor, seq_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    half_d = inv_freq.shape[-1]
    cos = torch.empty((1, seq_len, half_d), dtype=torch.float32, device=cache_position.device)
    return cos, torch.empty_like(cos)


@torch.library.impl(_alloy_lib, "rms_norm_backward", "Meta")
def _rms_norm_backward_meta(
    x: torch.Tensor, dy: torch.Tensor, weight: torch.Tensor, rrms: torch.Tensor
) -> torch.Tensor:
    return torch.empty_like(dy)


@torch.library.impl(_alloy_lib, "rope_apply", "Meta")
def _rope_meta(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.impl(_alloy_lib, "rms_norm_rope", "Meta")
def _rms_norm_rope_meta(
    x: torch.Tensor, weight: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
    eps: float, cos_half: bool = False,
) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.impl(_alloy_lib, "rope_apply_backward", "Meta")
def _rope_backward_meta(
    dout: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    shape: tuple[int, ...] = tuple(dout.shape)
    if len(shape) == 4:
        batch, heads, seq, dim = shape
        return torch.empty_strided(
            (batch, heads, seq, dim),
            (seq * heads * dim, dim, heads * dim, 1),
            dtype=dout.dtype,
            device=dout.device,
        )
    return torch.empty_like(dout)


@torch.library.impl(_alloy_lib, "batched_mm", "Meta")
def _batched_mm_meta(
    x: torch.Tensor,
    weights: Sequence[torch.Tensor],
    biases: Sequence[torch.Tensor] | None = None,
) -> list[torch.Tensor]:
    return [
        torch.empty(x.shape[:-1] + (weight.shape[-1],), device=x.device, dtype=x.dtype)
        for weight in weights
    ]


@torch.library.impl(_alloy_lib, "dot_silu", "Meta")
def _dot_silu_meta(x: torch.Tensor, gate_weight: torch.Tensor, up_weight: torch.Tensor) -> torch.Tensor:
    rows = x.shape[0]
    cols = gate_weight.shape[0]
    return torch.empty((rows, cols), dtype=x.dtype, device=x.device)


@torch.library.impl(_alloy_lib, "dequant_silu", "Meta")
def _dequant_silu_meta(
    x: torch.Tensor,
    gate_packed: torch.Tensor,
    gate_scales: torch.Tensor,
    up_packed: torch.Tensor,
    up_scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    rows = x.shape[0]
    cols = gate_packed.shape[0]
    return torch.empty((rows, cols), dtype=x.dtype, device=x.device)


@torch.library.impl(_alloy_lib, "gguf_q8_0_mm", "Meta")
def _gguf_q8_0_mm_meta(
    activations: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    rows = activations.shape[0]
    cols = qweight.shape[0]
    return torch.empty((rows, cols), dtype=activations.dtype, device=activations.device)


@torch.library.impl(_alloy_lib, "gguf_q8_0_embedding", "Meta")
def _gguf_q8_0_embedding_meta(
    input_ids: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(input_ids.shape + (qweight.shape[1],), dtype=torch.float32, device=input_ids.device)


@torch.library.impl(_alloy_lib, "gguf_q4_k_mm", "Meta")
def _gguf_q4_k_mm_meta(
    activations: torch.Tensor,
    blocks: torch.Tensor,
) -> torch.Tensor:
    rows = activations.shape[0]
    cols = blocks.shape[0]
    return torch.empty((rows, cols), dtype=activations.dtype, device=activations.device)


@torch.library.impl(_alloy_lib, "mlx_q4_mm", "Meta")
def _mlx_q4_mm_meta(
    activations: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
) -> torch.Tensor:
    rows = activations.shape[0]
    cols = qweight.shape[0]
    return torch.empty((rows, cols), dtype=activations.dtype, device=activations.device)


@torch.library.impl(_alloy_lib, "batched_mlx_q4_mm", "Meta")
def _batched_mlx_q4_mm_meta(
    activations: torch.Tensor,
    qweights: Sequence[torch.Tensor],
    scales: Sequence[torch.Tensor],
    biases: Sequence[torch.Tensor],
) -> list[torch.Tensor]:
    rows = activations.shape[0]
    return [
        torch.empty((rows, qw.shape[0]), dtype=activations.dtype, device=activations.device)
        for qw in qweights
    ]


@torch.library.impl(_alloy_lib, "mlx_q4_silu", "Meta")
def _mlx_q4_silu_meta(
    activations: torch.Tensor,
    gate_qweight: torch.Tensor,
    gate_scales: torch.Tensor,
    gate_biases: torch.Tensor,
    up_qweight: torch.Tensor,
    up_scales: torch.Tensor,
    up_biases: torch.Tensor,
) -> torch.Tensor:
    rows = activations.shape[0]
    cols = gate_qweight.shape[0]
    return torch.empty((rows, cols), dtype=activations.dtype, device=activations.device)


@torch.library.impl(_alloy_lib, "mlx_q4_embedding", "Meta")
def _mlx_q4_embedding_meta(
    input_ids: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(
        input_ids.shape + (qweight.shape[1] * 2,), dtype=torch.float32, device=input_ids.device
    )


@torch.library.impl(_alloy_lib, "gguf_moe_routed", "Meta")
def _gguf_moe_routed_meta(
    hidden: torch.Tensor,
    router_logits: torch.Tensor,
    gate_up_blocks: torch.Tensor,
    down_qweight: torch.Tensor,
    num_experts: int,
    top_k: int,
    moe_intermediate: int,
) -> torch.Tensor:
    # Routed output is one hidden-width vector per token: (T, H).
    return torch.empty(
        (hidden.shape[0], hidden.shape[1]), dtype=torch.float32, device=hidden.device
    )


@torch.library.impl(_alloy_lib, "gguf_q4_k_embedding", "Meta")
def _gguf_q4_k_embedding_meta(
    input_ids: torch.Tensor,
    blocks: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(
        input_ids.shape + ((blocks.shape[1] // 144) * 256,),
        dtype=torch.float32,
        device=input_ids.device,
    )


@torch.library.impl(_alloy_lib, "gguf_q6_k_mm", "Meta")
def _gguf_q6_k_mm_meta(
    activations: torch.Tensor,
    packed_weights: torch.Tensor,
) -> torch.Tensor:
    rows = activations.shape[0]
    cols = packed_weights.shape[0]
    return torch.empty((rows, cols), dtype=activations.dtype, device=activations.device)


@torch.library.impl(_alloy_lib, "gguf_q6_k_embedding", "Meta")
def _gguf_q6_k_embedding_meta(
    input_ids: torch.Tensor,
    packed_weights: torch.Tensor,
) -> torch.Tensor:
    # WIDTH = (packed row bytes / 210) * 256
    width = (packed_weights.shape[1] // 210) * 256
    return torch.empty(
        input_ids.shape + (width,),
        dtype=torch.float32,
        device=input_ids.device,
    )


@torch.library.impl(_alloy_lib, "gguf_q5_0_mm", "Meta")
def _gguf_q5_0_mm_meta(
    activations: torch.Tensor,
    qweight: torch.Tensor,
    qhigh: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    rows = activations.shape[0]
    cols = scales.shape[0]
    return torch.empty((rows, cols), dtype=activations.dtype, device=activations.device)


@torch.library.impl(_alloy_lib, "gguf_q5_0_embedding", "Meta")
def _gguf_q5_0_embedding_meta(
    input_ids: torch.Tensor,
    qweight: torch.Tensor,
    qhigh: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(
        input_ids.shape + (qweight.shape[1] * 2,),
        dtype=torch.float32,
        device=input_ids.device,
    )


@torch.library.impl(_alloy_lib, "gguf_q4_k_silu", "Meta")
def _gguf_q4_k_silu_meta(
    activations: torch.Tensor,
    gate_blocks: torch.Tensor,
    up_blocks: torch.Tensor,
) -> torch.Tensor:
    rows = activations.shape[0]
    cols = gate_blocks.shape[0]
    return torch.empty((rows, cols), dtype=activations.dtype, device=activations.device)


@torch.library.impl(_alloy_lib, "gguf_q4_k_gelu", "Meta")
def _gguf_q4_k_gelu_meta(
    activations: torch.Tensor,
    gate_blocks: torch.Tensor,
    up_blocks: torch.Tensor,
) -> torch.Tensor:
    rows = activations.shape[0]
    cols = gate_blocks.shape[0]
    return torch.empty((rows, cols), dtype=activations.dtype, device=activations.device)


@torch.library.impl(_alloy_lib, "gguf_q8_0_silu", "Meta")
def _gguf_q8_0_silu_meta(
    activations: torch.Tensor,
    gate_qweight: torch.Tensor,
    gate_scales: torch.Tensor,
    up_qweight: torch.Tensor,
    up_scales: torch.Tensor,
) -> torch.Tensor:
    rows = activations.shape[0]
    cols = gate_qweight.shape[0]
    return torch.empty((rows, cols), dtype=activations.dtype, device=activations.device)
