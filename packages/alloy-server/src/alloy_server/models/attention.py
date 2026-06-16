"""Module-replacement attention forward that uses alloy cache-attention ops.

Why this exists: torch.compile / AOT autograd lifts in-place mutations on
static-address tensors (like HF's KV cache) OUT of the FX graph. The
single-token decode path works because the FX rewrite reconstructs the
write from a matched index_put/index_copy node that lands in the K-chain;
for multi-token decode the chain doesn't contain a matchable mutation and
SDPA reads from the pre-update placeholder, producing wrong output.

Workaround: monkey-patch Qwen3Attention.forward globally to route the
verify-shape range (2 <= seq_len <= _MAX_VERIFY_K) through
`alloy.attention_kv_update_multi`. For AlloyStaticCache we also route cold
prefill and single-token decode through the custom ops so Q can remain fp32
while K/V cache storage and reads stay fp16; upstream SDPA's fake/meta path
requires Q/K/V dtypes to match and rejects that mixed contract before Alloy
lowering can run. The original Qwen3 forward is NOT called from the patched
forward because Dynamo treats captured closures with `**kwargs` as opaque,
inserting two graph breaks per layer (56 breaks per cold forward on 28-layer
Qwen3) that destroy fusion and plan compilation. Inlining keeps the fallback
path single-graph.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn
from transformers.cache_utils import StaticSlidingWindowLayer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb as llama_apply_rotary_pos_emb,
    eager_attention_forward as llama_eager_attention_forward,
)
from transformers.models.gemma3.modeling_gemma3 import Gemma3Attention
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4TextAttention,
    apply_rotary_pos_emb as gemma4_apply_rotary_pos_emb,
    eager_attention_forward as gemma4_eager_attention_forward,
)
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    apply_rotary_pos_emb as qwen3_apply_rotary_pos_emb,
    eager_attention_forward as qwen3_eager_attention_forward,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5GatedDeltaNet,
    Qwen3_5TextModel,
    apply_rotary_pos_emb as qwen3_5_apply_rotary_pos_emb,
    eager_attention_forward as qwen3_5_eager_attention_forward,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)
# Qwen3.5-MoE shares the qwen3.5 backbone verbatim (gated attention, GatedDeltaNet,
# rope, delta rule all byte-identical) — only the FFN differs (MoE vs dense). So the
# same patched forwards apply to the MoE attention/DeltaNet classes unchanged.
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeAttention,
    Qwen3_5MoeGatedDeltaNet,
    Qwen3_5MoeTextModel,
)
# LFM2 (LiquidAI): a conv/attention hybrid — short-conv layers (Lfm2ShortConv)
# interleave with GQA attention (Lfm2Attention, per-head q/k RMSNorm, no gate).
from transformers.models.lfm2.modeling_lfm2 import (
    Lfm2Attention,
    Lfm2ShortConv,
    apply_rotary_pos_emb as lfm2_apply_rotary_pos_emb,
    eager_attention_forward as lfm2_eager_attention_forward,
)
import numpy as np
import torch.nn.functional as F

import alloy_torch.custom_ops  # noqa: F401
from alloy_torch.ops.attention import _MAX_VERIFY_K
from alloy._runtime.alloy_buffer import materialize_many

# Process-wide flag toggled by `PrefillEngine.chunk_step` (and verify warmup)
# to tell the patched Qwen3 forward whether to take the alloy custom-op
# path for the current model.forward call.
#
# Why this can't be `past_key_values.layers[*].is_initialized`:
# `ContiguousKV.persistent_caches` reuses one StaticCache instance across whole
# conversations. After the first eager_compile_all warm-up,
# `is_initialized` is True forever — so a cold-prefill request on a
# fresh conversation would be misrouted to the runtime-Q_START_POS warm
# op (substantially slower than the constexpr cold kernel). The bool
# toggle below is the unambiguous "is this forward call semantically
# warm?" signal.
#
# Dynamo specialises on Python booleans, so the cold and warm cases
# produce two distinct compiled plans rather than recompiling per turn.
USE_ALLOY_WARM_OP: bool = False

# Side channel for DeltaNet attention mask. The alloy chunked prefill path
# pads input_ids; without masking the recurrent state at pad positions, decode
# reads contaminated state. Keyed by id(cache_params) so multiple in-flight
# caches don't collide. The mask is (1, bucket) int64 with 1 for real, 0 pad.
#
# IMPLEMENTATION NOTE: the mask is stashed as a tensor attribute on the
# cache_params object (not in a module-level dict keyed by id()). The dict +
# id() pattern forced a dynamo graph break inside the patched GDN forward,
# which silently disabled the alloy backend's Tensor(c!) mutation tracking on
# `linear_attention_update` — conv_states / recurrent_states stayed at their
# zero-init across the call, producing degenerate decode (repetition cycles
# after a few tokens). Plain attribute access on the Python cache object lets
# dynamo trace through cleanly and keeps the mutation propagation intact.
def set_deltanet_attn_mask(cache_params: Any, mask: torch.Tensor | None) -> None:
    """Copy the prefill-pad mask into each linear-attention layer's
    pre-allocated `alloy_attn_mask` buffer. The patched DeltaNet forward
    reads that buffer directly (a normal tensor attribute on the cache
    layer, marked `mark_static_address`) — which dynamo traces without a
    graph break, preserving the alloy backend's Tensor(c!) mutation
    propagation on the linear_attention_update op."""
    if cache_params is None or mask is None:
        return
    seq_len = int(mask.shape[1])
    for layer in cache_params.layers:
        try:
            buf = layer.alloy_attn_mask
        except AttributeError:
            continue
        if seq_len > buf.shape[1]:
            continue
        buf[:, :seq_len].copy_(mask)
        if seq_len < buf.shape[1]:
            buf[:, seq_len:].fill_(0)


# ---------------------------------------------------------------------------
# Speculative tap collection. transformers 5.8 implements
# output_hidden_states via an output-capturing context manager that graph-breaks
# under Dynamo, so taps ride a traceable side channel instead: tapped decoder
# layers are wrapped with TapLayer, which appends its output hidden to the
# module-global TAP_VALUES list when TAPS_ENABLED — Dynamo's side-effect
# tracking threads the appended tensors through the graph, and the wrapper
# module (verify/prefill) reads + returns them as ordinary outputs. Pinned-plan
# replay never runs the python; the taps are just extra plan outputs.
TAP_VALUES: list = []
TAPS_ENABLED = False


def set_taps_enabled(value: bool) -> None:
    global TAPS_ENABLED
    TAPS_ENABLED = bool(value)


def tap_values_clear() -> None:
    TAP_VALUES.clear()


def tap_values() -> list:
    return list(TAP_VALUES)


class TapLayer(nn.Module):
    """Transparent decoder-layer wrapper that mirrors its output hidden into
    the tap sink. Installed once per tapped layer by install_taps()."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, *args, **kwargs):
        out = self.inner(*args, **kwargs)
        if TAPS_ENABLED:
            hidden = out[0] if isinstance(out, tuple) else out
            TAP_VALUES.append(hidden)
        return out


def install_taps(model: nn.Module, layer_ids: tuple) -> None:
    """Wrap the tapped decoder layers (idempotent). Installs BEFORE any
    compile so every specialization sees the wrapper; the TAPS_ENABLED read
    is a Dynamo guard, so tapless forwards (decode) carry no append."""
    layers = model.model.layers
    for lid in layer_ids:
        if not isinstance(layers[lid], TapLayer):
            layers[lid] = TapLayer(layers[lid])
    torch._dynamo.reset()


def set_use_alloy_warm_op(value: bool) -> None:
    global USE_ALLOY_WARM_OP
    USE_ALLOY_WARM_OP = bool(value)


def current_use_alloy_warm_op() -> bool:
    return USE_ALLOY_WARM_OP


# Optional per-layer output capture for correctness debugging.
# Enable via ALLOY_LAYER_CAPTURE=1; outputs land in `/tmp/alloy_captures/`.
LAYER_CAPTURES: dict[str, Any] = {}


CAPTURE_DIR = "/tmp/alloy_captures"

# Read once at import. The capture hook is `@torch._dynamo.disable`d, so a *call*
# to it forces a dynamo graph break at every site even when it would no-op at
# runtime — that added ~76 graph breaks (2 per layer) to the prefill forward,
# the dominant share of all breaks. Gating the call behind this module-level
# constant lets dynamo fold the `if` to a constant and prune the call entirely
# when capture is off, so the break disappears in the (normal) disabled case.
# Set ALLOY_LAYER_CAPTURE=1 before import to enable; breaks then return, which
# is fine for the diagnostic path.
LAYER_CAPTURE_ENABLED = os.environ.get("ALLOY_LAYER_CAPTURE") == "1"


def capture_layer_output(name: str, tensor) -> None:
    """Traceable gate around the disabled capture impl. When capture is off
    (the default) dynamo constant-folds the guard and drops the call, so no
    graph break is introduced."""
    if LAYER_CAPTURE_ENABLED:
        capture_layer_output_impl(name, tensor)


@torch._dynamo.disable
def capture_layer_output_impl(name: str, tensor) -> None:
    """Save a hidden_state to disk. Disabled-on-dynamo so it doesn't get
    traced. Writes to /tmp/alloy_captures/ to avoid storing tensors in Python
    globals (which would break dynamo guards across re-compiles)."""
    if os.environ.get("ALLOY_LAYER_CAPTURE") != "1":
        return
    seq_len = tensor.shape[1] if len(tensor.shape) >= 2 else 0
    full_name = f"{name}_S{seq_len}"
    os.makedirs(CAPTURE_DIR, exist_ok=True)
    path = f"{CAPTURE_DIR}/{full_name}.npy"
    if os.path.exists(path):
        return  # First occurrence wins
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().float().numpy()
    else:
        materialize_many([tensor])
        arr = np.array(tensor.numpy).copy()
    np.save(path, arr)


def get_layer_captures() -> dict[str, Any]:
    return dict(LAYER_CAPTURES)


def clear_layer_captures() -> None:
    LAYER_CAPTURES.clear()


def use_alloy_cache_op(layer: Any) -> bool:
    # Check the per-layer dtype FIRST: for AlloyStaticCache (the daemon path)
    # it's always set, so the `or` short-circuits and dynamo never reads the
    # `USE_ALLOY_WARM_OP` module global — which it would otherwise guard on,
    # splitting the cold-prefill graph (global False) from warm (global True)
    # and forcing a second compile. The global only matters for the non-Alloy
    # spec-decode verify path (cache_dtype None + warm op forced); reading it
    # last preserves that behaviour without the steady-state guard.
    return layer._alloy_cache_dtype is not None or USE_ALLOY_WARM_OP


def cache_write_dtype(layer: Any, name: str, fallback: torch.dtype) -> torch.dtype:
    # `name` is "keys" or "values" — only two valid callers — so branch
    # explicitly rather than reflecting on `layer`.
    if name == "keys":
        cache_tensor = layer.keys
    elif name == "values":
        cache_tensor = layer.values
    else:
        raise ValueError(f"unexpected cache name: {name!r}")
    if cache_tensor is not None:
        return cache_tensor.dtype
    # `_alloy_cache_dtype` lives on Alloy*Layer subclasses; the upstream HF
    # `StaticLayer` doesn't have it. AttributeError -> no preferred dtype.
    try:
        target_dtype = layer._alloy_cache_dtype
    except AttributeError:
        target_dtype = None
    return fallback if target_dtype is None else target_dtype


def alloy_cache_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer: Any,
    cache_position: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    key_dtype = cache_write_dtype(layer, "keys", k.dtype)
    value_dtype = cache_write_dtype(layer, "values", v.dtype)
    write_k = k.to(key_dtype) if k.dtype != key_dtype else k
    write_v = v.to(value_dtype) if v.dtype != value_dtype else v
    # Quantized-KV layer (q8_0): the codes/scales buffer set replaces the fp16
    # K/V tensors entirely (allocated eagerly at construction — no lazy init).
    # Only Alloy*Layer carries the attribute (None on the fp16 path); the
    # non-Alloy spec-decode verify path uses plain HF layers, hence the
    # AttributeError guard (same pattern as `cache_write_dtype`). The
    # `is not None` outcome is a stable per-layer Dynamo constant.
    try:
        keys_q = layer.alloy_keys_q
    except AttributeError:
        keys_q = None
    # The chunk's last-real-row bound for the sliding ring write — a per-cache
    # pinned tensor (see AlloyStaticCache), passed as a real op operand so it
    # rides the plan as dataflow. None (plain HF layers, spec verify path)
    # falls back to the handler's unbounded sentinel.
    try:
        last_real = layer.alloy_last_real
    except AttributeError:
        last_real = None
    if keys_q is not None:
        sliding_q8 = (
            int(layer.max_cache_len) if isinstance(layer, StaticSlidingWindowLayer) else 0
        )
        return torch.ops.alloy.attention_cache_q8(
            q, write_k, write_v, cache_position[:1],
            keys_q, layer.alloy_keys_scales,
            layer.alloy_values_q, layer.alloy_values_scales,
            float(scale), sliding_window=sliding_q8, last_real=last_real,
        )
    if not layer.is_initialized or layer.keys is None or layer.values is None:
        layer.lazy_initialization(write_k, write_v)
    cache_pos_scalar = cache_position[:1]
    seq_len = q.shape[2]
    # Sliding-window layers (gemma3) have a circular K/V cache whose
    # physical size is ``layer.max_cache_len`` (HF caps it to sliding_window).
    # Pass it through so the kernel modulos writes; full-attention layers
    # carry sliding=0 which is the linear-write path.
    sliding = int(layer.max_cache_len) if isinstance(layer, StaticSlidingWindowLayer) else 0
    # Full-attention (sliding == 0: qwen3.5/qwen3/qwen2.5/llama/deepseek):
    # ONE op for decode (seq_len==1), spec-decode verify (<=_MAX_VERIFY_K), and
    # prefill (>_MAX_VERIFY_K). The handler picks the kernel path from runtime
    # seq_len, so the traced graph has no seq_len branch and Dynamo can compile
    # decode + prefill as a single graph. Each concrete seq_len still records
    # its own plan at run-0, so decode keeps flash-decode and prefill keeps the
    # strided runtime-pos kernel — kernels (and TPOT/prefill throughput) are
    # unchanged; only the trace is shared.
    if sliding == 0:
        return torch.ops.alloy.attention_cache(
            q, write_k, write_v, cache_pos_scalar, layer.keys, layer.values,
            float(scale), sliding_window=sliding, last_real=last_real,
        )
    # Sliding-window (gemma3): keep the explicit seq_len branch. Its >SW
    # prefill needs the cold path's linear temp-KV buffer, which the unified
    # op's prefill path doesn't build; `USE_ALLOY_WARM_OP` / `compile_window.q_start_pos`
    # gate the cold-vs-warm K/V slice extent here.
    if seq_len == 1:
        return torch.ops.alloy.attention_kv_update(
            q, write_k, write_v, cache_pos_scalar, layer.keys, layer.values,
            float(scale), sliding_window=sliding,
        )
    if seq_len <= _MAX_VERIFY_K:
        return torch.ops.alloy.attention_kv_update_multi(
            q, write_k, write_v, cache_pos_scalar, layer.keys, layer.values,
            float(scale), sliding_window=sliding,
        )
    if not USE_ALLOY_WARM_OP:
        return torch.ops.alloy.attention_prefill_cold(
            q, write_k, write_v, cache_pos_scalar, layer.keys, layer.values, float(scale),
            sliding_window=sliding, last_real=last_real,
        )
    return torch.ops.alloy.attention_prefill_warm(
        q, write_k, write_v, cache_pos_scalar, layer.keys, layer.values,
        float(scale), sliding_window=sliding, last_real=last_real,
    )


def gemm_cache_attention(
    q: torch.Tensor,
    k_full: torch.Tensor,
    v_full: torch.Tensor,
    cache_position: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Non-fused causal attention via GEMM + softmax.

    For head_dim the fused flash-attention kernel can't fit: gemma4's global
    layers are head_dim 512, where the `[BLOCK_M, 512]` f32 o-accumulator needs
    64 simdgroup tiles (128 floats/thread) — it overflows the register budget,
    spills to shared memory, blows the 32 KB threadgroup limit and the kernel
    silently returns zeros. alloy's `batched_mm`/`softmax` kernels have no such
    limit, so route these layers through them.

    `q` is `[B, H, N, D]`; `k_full`/`v_full` are `[B, KV_H, L, D]` (the full
    KV-cache view for non-shared layers, or the reused full-length K/V for
    shared layers). Causal over absolute positions: query row `i` (absolute
    position `cache_position[i]`) attends to KV `[0, cache_position[i]]`, which
    also masks out unwritten cache slots (they sit beyond the query position).
    Scores are accumulated in fp32 — the fused kernel's MMA accumulates fp32
    too — while the KV cache itself stays fp16 (the invariant): `k_full`/`v_full`
    arrive at the fp16 cache dtype.
    """
    groups = q.shape[1] // k_full.shape[1]
    kr = k_full.repeat_interleave(groups, dim=1) if groups > 1 else k_full
    vr = v_full.repeat_interleave(groups, dim=1) if groups > 1 else v_full
    scores = torch.matmul(q.float(), kr.float().transpose(-2, -1)) * float(scale)
    kv_idx = torch.arange(k_full.shape[2], device=q.device)
    future = kv_idx[None, :] > cache_position[:, None]
    scores = scores.masked_fill(future[None, None], float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, vr.float()).to(q.dtype)


def alloy_qwen3_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values=None,
    **kwargs: Any,
):
    """Patched Qwen3Attention.forward.

    Three runtime branches, all guarded on Python state Dynamo specialises:
      - AlloyStaticCache
        → alloy.attention_kv_update / multi / prefill_warm, including cold
        prefill and single-token decode, so mixed fp32-Q/fp16-KV never enters
        upstream SDPA.
      - 2 <= seq_len <= _MAX_VERIFY_K with a non-Alloy populated cache and
        `USE_ALLOY_WARM_OP`
        → alloy.attention_kv_update_multi (fused kv-write + multi-token
        attention, used by spec-decode verify).
      - seq_len > _MAX_VERIFY_K with a non-Alloy populated cache and
        `USE_ALLOY_WARM_OP`
        → alloy.attention_prefill_warm (runtime Q_START_POS warm-suffix
        prefill — multi-turn safe).
      - otherwise
        → inlined standard cache.update + SDPA fallback.
    """
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    q = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    k = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    q, k = qwen3_apply_rotary_pos_emb(q, k, cos, sin)

    if past_key_values is not None:
        layer = past_key_values.layers[self.layer_idx]
        if use_alloy_cache_op(layer):
            attn_output = alloy_cache_attention(
                q, k, v, layer, kwargs["cache_position"], self.scaling
            )
            attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output, None

    # Cold / single-token path — inlined upstream Qwen3Attention.forward
    # body so Dynamo traces straight through with no graph breaks. Use the
    # exact `.get_interface(..., fallback)` shape upstream uses; the
    # bytecode-level match keeps Dynamo's specialisation behaviour aligned
    # with the un-patched forward.
    if past_key_values is not None:
        k, v = past_key_values.update(k, v, self.layer_idx)

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, qwen3_eager_attention_forward
    )

    attn_output, attn_weights = attention_interface(
        self,
        q,
        k,
        v,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def alloy_qwen3_5_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
    past_key_values=None,
    **kwargs: Any,
):
    """Patched Qwen3_5Attention.forward.

    Same alloy cache-op routing as Qwen3 plus the qwen3.5-specific gated
    attention head:
      - `q_proj` is 2x as wide as for qwen3 — output reshaped to
        (B, S, H, 2*D), split via `torch.chunk(2, dim=-1)` into the
        attention query (B, S, H, D) and a per-element gate of the
        same shape. The gate is reshaped to (B, S, H*D) for the
        post-attention multiplication.
      - After attention compute (whether through alloy cache ops or
        upstream SDPA), `attn_output = attn_output * sigmoid(gate)`
        before `o_proj`.
    No sliding window; qwen3.5 full-attention layers attend to the
    full causal context.
    """
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    capture_layer_output(f"L{self.layer_idx}_in", hidden_states)

    # 2x-wide q_proj → split into query and gate over the last dim.
    q_full = self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2)
    q_view, gate = torch.chunk(q_full, 2, dim=-1)
    gate = gate.reshape(*input_shape, -1)
    q = self.q_norm(q_view.contiguous().view(hidden_shape)).transpose(1, 2)
    k_raw = self.k_proj(hidden_states).view(hidden_shape)
    k = self.k_norm(k_raw).transpose(1, 2)
    v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    q, k = qwen3_5_apply_rotary_pos_emb(q, k, cos, sin)

    if past_key_values is not None:
        layer = past_key_values.layers[self.layer_idx]
        if use_alloy_cache_op(layer):
            attn_output = alloy_cache_attention(
                q, k, v, layer, kwargs["cache_position"], self.scaling
            )
            attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
            attn_output = attn_output * torch.sigmoid(gate)
            attn_output = self.o_proj(attn_output)
            capture_layer_output(f"L{self.layer_idx}_modout", attn_output)
            return attn_output, None

    if past_key_values is not None:
        k, v = past_key_values.update(k, v, self.layer_idx)

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, qwen3_5_eager_attention_forward
    )

    attn_output, attn_weights = attention_interface(
        self,
        q,
        k,
        v,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = attn_output * torch.sigmoid(gate)
    attn_output = self.o_proj(attn_output)
    capture_layer_output(f"L{self.layer_idx}_modout", attn_output)
    return attn_output, attn_weights


@torch._dynamo.disable
def deltanet_body_torch(
    self,
    mixed_qkv,
    z,
    a,
    b,
    conv_state,
    recurrent_state,
    has_previous_state,
):
    """Pure-torch DeltaNet body (matches HF eager). Diagnostic replacement for alloy.linear_attention_update."""
    batch_size, seq_len, _ = mixed_qkv.shape
    z_reshaped = z.reshape(batch_size, seq_len, -1, self.head_v_dim)
    mixed_qkv_bcs = mixed_qkv.transpose(1, 2).contiguous()  # (B, C, S)

    conv_kernel_size = self.conv_kernel_size
    if has_previous_state and seq_len == 1:
        state_rolled = torch.cat([conv_state[:, :, 1:], mixed_qkv_bcs], dim=-1)
        conv_state.copy_(state_rolled)
        weight = self.conv1d.weight.squeeze(1)
        mixed_qkv_bcs = F.silu((state_rolled * weight.unsqueeze(0)).sum(dim=-1, keepdim=True))
    else:
        if has_previous_state:
            mixed_qkv_bcs = torch.cat([conv_state, mixed_qkv_bcs], dim=-1)
        new_conv_state = F.pad(mixed_qkv_bcs, (conv_kernel_size - mixed_qkv_bcs.shape[-1], 0))
        conv_state.copy_(new_conv_state)
        weight = self.conv1d.weight
        bias = self.conv1d.bias
        x_pad = F.pad(mixed_qkv_bcs, (conv_kernel_size - 1, 0))
        conv_out = F.conv1d(x_pad, weight, bias, padding=0, groups=mixed_qkv_bcs.shape[1])
        mixed_qkv_bcs = F.silu(conv_out[:, :, : mixed_qkv_bcs.shape[-1]])
        if has_previous_state:
            mixed_qkv_bcs = mixed_qkv_bcs[:, :, -seq_len:]

    mixed_qkv = mixed_qkv_bcs.transpose(1, 2)
    key_dim = self.num_k_heads * self.head_k_dim
    value_dim = self.num_v_heads * self.head_v_dim
    query, key, value = torch.split(mixed_qkv, [key_dim, key_dim, value_dim], dim=-1)
    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)
    beta_t = b.sigmoid()
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    if self.num_v_heads // self.num_k_heads > 1:
        n_rep = self.num_v_heads // self.num_k_heads
        query = query.repeat_interleave(n_rep, dim=2)
        key = key.repeat_interleave(n_rep, dim=2)

    if has_previous_state and seq_len == 1:
        core_attn_out, last_recurrent_state = torch_recurrent_gated_delta_rule(
            query, key, value, g=g, beta=beta_t, initial_state=recurrent_state,
            output_final_state=True, use_qk_l2norm_in_kernel=True,
        )
    else:
        core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
            query, key, value, g=g, beta=beta_t,
            initial_state=recurrent_state if has_previous_state else None,
            output_final_state=True, use_qk_l2norm_in_kernel=True,
        )
    if last_recurrent_state is not None:
        recurrent_state.copy_(last_recurrent_state.to(recurrent_state.dtype))

    core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
    z_flat = z_reshaped.reshape(-1, self.head_v_dim)
    core_attn_out = self.norm(core_attn_out, z_flat)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
    return core_attn_out


def alloy_qwen3_5_update_linear_attn_mask(self, attention_mask, past_key_values):
    """Patched `Qwen3_5TextModel._update_linear_attn_mask` — always returns
    None.

    Upstream branches on `past_key_values.has_previous_state()` (a Python
    bool that flips False→True after the first prefill chunk). Dynamo guards
    on it, which splits the cold-prefill graph from the warm-prefill graph and
    forces a second full compile of the model. Our patched GDN forward ignores
    the passed `attention_mask` entirely — it reads `layer.alloy_attn_mask`
    instead — so this method's result is dead on the alloy path. Returning
    None unconditionally drops the bool read (no guard → cold and warm share
    one plan) without changing behaviour. Full-attention layers are unaffected:
    they take `causal_mask`, not this value (modeling_qwen3_5.py:1284)."""
    return None


def alloy_qwen3_5_gated_delta_net_forward(
    self,
    hidden_states: torch.Tensor,
    cache_params: Any = None,
    attention_mask: torch.Tensor | None = None,
):
    """Patched Qwen3_5GatedDeltaNet.forward.

    The model's eager forward writes the conv- and recurrent-state caches
    via `cache_params.update_conv_state(...)` / `update_recurrent_state(...)`,
    which torch.compile / AOT autograd lifts OUT of the FX graph — the
    cache never actually updates, decode reads zero state, output is
    garbage tokens (the same class of bug `attention_kv_update` solves
    for the regular KV cache).

    Replace the body with a single custom op call that subsumes the
    causal Conv1d, the chunked/recurrent delta rule, and RMSNormGated.
    The op declares `conv_state` and `recurrent_state` as mutable
    inputs so AOT keeps the state update in-graph. Linear projections
    (in_proj_*, out_proj) stay in the FX graph and route through the
    GGUFQ8_0 fast paths as usual.
    """
    batch_size, seq_len, _ = hidden_states.shape
    capture_layer_output(f"L{self.layer_idx}_in", hidden_states)

    layer = cache_params.layers[self.layer_idx]

    # Eager-fast projections (Q8_0 mm), traced into the FX graph as usual.
    mixed_qkv = self.in_proj_qkv(hidden_states)              # (B, S, conv_dim)
    z = self.in_proj_z(hidden_states)                        # (B, S, value_dim)
    b = self.in_proj_b(hidden_states)                        # (B, S, num_v_heads)
    a = self.in_proj_a(hidden_states)                        # (B, S, num_v_heads)
    if self.layer_idx == 0:
        capture_layer_output("L0_qkv", mixed_qkv)
        capture_layer_output("L0_z", z)
        capture_layer_output("L0_a", a)
        capture_layer_output("L0_b", b)

    # Pad mask: read from cache layer's pre-allocated static-address tensor
    # (1 for real tokens, 0 for pads). Apply AFTER projections to zero pad
    # contributions. Reading via a tensor attribute on the cache layer keeps
    # dynamo from graph-breaking, preserving the alloy backend's Tensor(c!)
    # mutation propagation on linear_attention_update.
    full_mask = layer.alloy_attn_mask
    alloy_attn_mask = full_mask[:, :seq_len].float()
    mixed_qkv = mixed_qkv * alloy_attn_mask.unsqueeze(-1)
    z = z * alloy_attn_mask.unsqueeze(-1)
    # a/b: set pad positions to -1e9 so recurrent rule no-ops there.
    pad_neg = (1.0 - alloy_attn_mask).unsqueeze(-1) * -1e9
    a = a * alloy_attn_mask.unsqueeze(-1) + pad_neg
    b = b * alloy_attn_mask.unsqueeze(-1) + pad_neg
    real_len = alloy_attn_mask.sum(dim=1).to(torch.int64)

    # `has_previous_state` only gates the conv1d DECODE branch in the handler
    # (`if has_previous_state and seq_len == 1`); for prefill (seq_len > 1) it
    # is ignored. Reading the Python bool unconditionally makes dynamo guard
    # on it, which splits the cold-prefill graph (flag False on chunk 0) from
    # the warm-prefill graph (flag True on chunk 1+) — forcing a second full
    # compile. Under dynamic=False `seq_len` is a trace-time constant, so
    # gating the read on `seq_len == 1` folds the branch away during prefill:
    # cold and warm then trace identically (no flag guard) and share ONE plan.
    # Decode still reads the real value (and it's stably True there).
    has_previous_state = bool(layer.has_previous_state) if seq_len == 1 else False

    core_out = torch.ops.alloy.linear_attention_update(
        mixed_qkv, z, a, b,
        layer.conv_states, layer.recurrent_states,
        self.conv1d.weight,
        self.A_log, self.dt_bias, self.norm.weight,
        self.num_k_heads, self.num_v_heads,
        self.head_k_dim, self.head_v_dim,
        self.conv_kernel_size,
        float(self.layer_norm_epsilon),
        has_previous_state,
        real_len,
    )
    layer.has_previous_state = True

    if self.layer_idx == 0:
        capture_layer_output("L0_core_out", core_out)
    output = self.out_proj(core_out)
    capture_layer_output(f"L{self.layer_idx}_modout", output)
    return output


def alloy_lfm2_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
    past_key_values=None,
    **kwargs: Any,
):
    """Patched Lfm2Attention.forward.

    Plain GQA with per-head q/k RMSNorm (`q_layernorm`/`k_layernorm`, applied to
    the (B, S, H, head_dim) view before the (1,2) transpose — the same layout
    Qwen3's q_norm uses), standard rotate-half RoPE, and `out_proj`. No gate, no
    sliding window. Same alloy cache-op routing as Qwen3: AlloyStaticCache layers
    go through `alloy_cache_attention` (mixed fp32-Q/fp16-KV never reaches
    upstream SDPA); everything else falls back to the inlined cache.update + SDPA.
    """
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    q = self.q_layernorm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    k = self.k_layernorm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    q, k = lfm2_apply_rotary_pos_emb(q, k, cos, sin)

    if past_key_values is not None:
        layer = past_key_values.layers[self.layer_idx]
        if use_alloy_cache_op(layer):
            attn_output = alloy_cache_attention(
                q, k, v, layer, kwargs["cache_position"], self.scaling
            )
            attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
            attn_output = self.out_proj(attn_output)
            return attn_output, None

    # Cold / single-token fallback — inlined upstream Lfm2Attention body.
    if past_key_values is not None:
        k, v = past_key_values.update(k, v, self.layer_idx)

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, lfm2_eager_attention_forward
    )
    attn_output, attn_weights = attention_interface(
        self, q, k, v, attention_mask, dropout=0.0, scaling=self.scaling, **kwargs
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.out_proj(attn_output)
    return attn_output, attn_weights


def alloy_lfm2_short_conv_forward(
    self,
    hidden_states: torch.Tensor,
    past_key_values=None,
    cache_position: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
):
    """Patched Lfm2ShortConv.forward.

    The eager forward writes the conv-state cache via `.copy_()`, which
    torch.compile / AOT autograd lifts OUT of the FX graph — decode then reads
    zero state → garbage (the same class of bug `linear_attention_update` solves
    for DeltaNet). Replace the conv body with `short_conv_update`, whose
    `conv_state` operand is mutable so the write stays in-graph. The in_proj /
    out_proj linears + the `B*x` / `C*` gates stay in the FX graph and route
    through the GGUF Q4_K fast paths.

    LFM2 conv: `BCx = in_proj(x)`; split into (B, C, x); `Bx = B * x`; causal
    depthwise Conv1d (no activation); `y = C * conv_out`; `out_proj(y)`.
    """
    batch_size, seq_len, _ = hidden_states.shape
    layer = past_key_values.layers[self.layer_idx]

    bcx = self.in_proj(hidden_states)              # (B, S, 3*hidden) — Q4_K mm

    # Gate the has_previous_state read on seq_len==1 so cold vs warm prefill
    # trace identically (no Python-bool guard split) — only the decode conv
    # branch consumes it. (See the DeltaNet forward for the full rationale.)
    has_previous_state = bool(layer.has_previous_state) if seq_len == 1 else False

    if has_previous_state:
        # Warm decode: fuse chunk -> b*x -> conv -> c* into one gated kernel,
        # reading the b/c/x column-slices of bcx directly. Collapses the conv
        # diamond (in_proj GEMV -> mul -> conv -> mul) to a single dispatch.
        conv_out = torch.ops.alloy.short_conv_gated(
            bcx,
            layer.conv_states,
            self.conv.weight,
            self.L_cache,
            has_previous_state,
        )
        layer.has_previous_state = True
        return self.out_proj(conv_out)

    b_gate, c_gate, x = bcx.chunk(3, dim=-1)       # each (B, S, hidden)
    bx = b_gate * x

    # Real prompt length for the conv-state save in a padded prefill chunk —
    # read from the layer's pad-mask buffer (set per chunk by the prefill
    # engine; all-ones for decode). Same mechanism the DeltaNet forward uses.
    full_mask = layer.alloy_attn_mask
    real_len = full_mask[:, :seq_len].sum(dim=1).to(torch.int64)

    conv_out = torch.ops.alloy.short_conv_update(
        bx,
        layer.conv_states,
        self.conv.weight,
        self.L_cache,
        has_previous_state,
        real_len,
    )
    layer.has_previous_state = True

    y = c_gate * conv_out
    return self.out_proj(y)


def alloy_llama_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values=None,
    **kwargs: Any,
):
    """Patched LlamaAttention.forward — mirrors the Qwen3 variant but
    drops the Qwen3-only q_norm/k_norm post-projection normalisation and
    the sliding_window kwarg. Same three-way routing:
      - AlloyStaticCache → attention_kv_update / multi / prefill_warm.
      - past_key_values is None OR USE_ALLOY_WARM_OP=False on non-Alloy
        cache → inlined standard cache.update + SDPA.
      - 2 <= seq_len <= _MAX_VERIFY_K on warm non-Alloy cache
        → attention_kv_update_multi.
      - seq_len > _MAX_VERIFY_K on warm non-Alloy cache
        → attention_prefill_warm.

    Closes the warm-prefill gap on Llama: avoids a full-cold reprefill
    for every conversation turn and instead scales with suffix length.
    """
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    # Llama has no q_norm / k_norm.
    q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    q, k = llama_apply_rotary_pos_emb(q, k, cos, sin)

    if past_key_values is not None:
        layer = past_key_values.layers[self.layer_idx]
        if use_alloy_cache_op(layer):
            attn_output = alloy_cache_attention(
                q, k, v, layer, kwargs["cache_position"], self.scaling
            )
            attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output, None

    # Cold / single-token path — inlined upstream LlamaAttention.forward
    # body. Same graph-break rationale as the Qwen3 variant.
    if past_key_values is not None:
        k, v = past_key_values.update(k, v, self.layer_idx)

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, llama_eager_attention_forward
    )

    attn_output, attn_weights = attention_interface(
        self,
        q,
        k,
        v,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def alloy_gemma3_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
    past_key_values=None,
    **kwargs: Any,
):
    """Patched Gemma3Attention.forward — same structure as Qwen3 (q_norm /
    k_norm post-projection) but accepts Gemma3's argument order (no
    position_ids positional). Alloy custom-op cache routing is identical to
    Qwen3; sliding-window layers store a windowed cache that the kv_update
    kernel naturally respects via cache_position % window.
    """
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    # Match upstream Gemma3Attention exactly: project → view → transpose,
    # then apply q_norm/k_norm. The contiguous-fallback path in
    # _fused_rms_norm_rope handles the resulting strided 4D input.
    q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    q = self.q_norm(q)
    k = self.k_norm(k)

    cos, sin = position_embeddings
    q, k = qwen3_apply_rotary_pos_emb(q, k, cos, sin)

    if past_key_values is not None:
        layer = past_key_values.layers[self.layer_idx]
        if use_alloy_cache_op(layer):
            attn_output = alloy_cache_attention(
                q, k, v, layer, kwargs["cache_position"], self.scaling
            )
            attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output, None

    # Cold / single-token fallback. Gemma3 passes sliding_window to the
    # attention_interface; reuse Qwen3's eager forward (signature matches).
    if past_key_values is not None:
        k, v = past_key_values.update(k, v, self.layer_idx)

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, qwen3_eager_attention_forward
    )

    attn_output, attn_weights = attention_interface(
        self,
        q,
        k,
        v,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def alloy_gemma4_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
    shared_kv_states: dict | None = None,
    past_key_values=None,
    **kwargs: Any,
):
    """Patched Gemma4TextAttention.forward.

    Mirrors upstream — per-layer-type head_dim (sliding 256 / global
    `global_head_dim` 512), q_norm/k_norm AND v_norm, RoPE applied per-tensor at
    unsqueeze_dim=2 before the (1,2) transpose, cross-layer KV sharing via
    `shared_kv_states` for the trailing `num_kv_shared_layers`, scaling=1.0.

    Routing (gemma3-style, proper handler — no SDPA on the cache path):
      * Non-shared layers → `alloy_cache_attention` (write + attend on the fp16
        KV cache, with scale=self.scaling).
      * Shared layers (no cache layer of their own) → cache-less attention via
        the standard interface against the source layer's K/V, which are kept at
        the **fp16 cache dtype** (the invariant). Matching the query dtype to it
        keeps the masked-attention kernel's shmem tiles at fp16 — essential at
        head_dim 512, where fp32 tiles (16 KB each) overflow the 32 KB
        threadgroup budget.
    """
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    cos, sin = position_embeddings

    # Upstream applies RoPE at unsqueeze_dim=2 on the [B,S,H,D] tensor, THEN
    # transposes to [B,H,S,D]. That is mathematically identical to transposing
    # first and applying RoPE at unsqueeze_dim=1 (the per-position rotation is
    # the same per head), but the latter is the layout alloy's rope_apply op +
    # rms_norm_rope fusion expect (qwen3/gemma3 emit it that way). Emitting the
    # gemma4 pre-transpose form instead makes the rope rewrite/fusion apply the
    # rotation against the wrong broadcast axis. Transpose-then-rope here.
    query_states = self.q_proj(hidden_states).view(hidden_shape)
    query_states = self.q_norm(query_states).transpose(1, 2)
    query_states = gemma4_apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=1)
    # The head_dim<=256 (sliding) layers route through the fused cache-attention
    # kernel, which shares the fp16 KV cache dtype on its shmem Q tile (the cache
    # invariant; the MMA still accumulates in fp32). Cast Q to fp16 there.
    #
    # The head_dim-512 (global) layers DON'T touch that kernel — they go through
    # the non-fused GEMM path (`gemm_cache_attention`), which upcasts every
    # operand to fp32 internally. Casting Q to fp16 first is pure precision loss
    # for them, and gemma4 amplifies it: scaling=1.0 (not 1/sqrt(d)) means the
    # head_dim-512 dot product is a sum of 512 terms with no shrink, so fp16 Q
    # rounding (~1e-3) accumulates into a materially wrong score. Keep Q at fp32
    # for the GEMM layers; the KV cache stays fp16 (the invariant is on K/V).
    if self.head_dim <= 256:
        query_states = query_states.to(torch.float16)

    if self.is_kv_shared_layer:
        # Trailing shared layers reuse the full-length K/V of the last
        # non-shared layer of their type (no k/v projections exist). They keep
        # their own cache entry (the gemma4 cache is not trimmed) and route the
        # reused K/V through the same cached attention op as everyone else — its
        # kernel fits head_dim 512, the cache-less one doesn't.
        shared_entry = shared_kv_states[self.layer_type]
        if len(shared_entry) == 4:
            # Quantized full-type source (codes, k_scales, v_codes, v_scales) —
            # consumed by the head_dim-512 shared branch below; these views are
            # throwaway placeholders for the unused new_k/new_v slots.
            key_states, value_states = shared_entry[0], shared_entry[2]
        else:
            key_states, value_states = shared_entry
        layer = past_key_values.layers[self.layer_idx] if past_key_values is not None else None
    else:
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = (
            self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states
        )
        key_states = self.k_norm(key_states).transpose(1, 2)
        key_states = gemma4_apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=1)
        value_states = self.v_norm(value_states)
        value_states = value_states.transpose(1, 2)
        layer = past_key_values.layers[self.layer_idx] if past_key_values is not None else None
        if self.store_full_length_kv and self.head_dim <= 256:
            # Sliding type: expose the current K/V for this type's shared layers
            # at the fp16 cache dtype. (The full type stores its POST-update
            # full-length K/V from the head_dim-512 branch below — the shared
            # full layers need every accumulated position, not just this step's.)
            kv_dtype = (
                cache_write_dtype(layer, "keys", key_states.dtype)
                if layer is not None
                else key_states.dtype
            )
            shared_kv_states[self.layer_type] = (
                key_states.to(kv_dtype),
                value_states.to(kv_dtype),
            )

    # Global/full layers are head_dim 512. They route through the fused alloy
    # KV-cache attention op for BOTH prefill and decode (the GEMM+softmax
    # fallback is kept only for the non-alloy path). Two head_dim-512 fixes make
    # the fused kernels fit Apple Silicon's hardware limits:
    #   - DECODE: the vector-decode kernel keeps the 512-dim output in registers
    #     (PER_LANE=16 quad-vec4, no simdgroup-matrix tiles).
    #   - PREFILL: BLOCK_M=16 makes the p@v dot use reg=2 (16×16 tiles), so
    #     n_simdgroups = (16/16)*(512/16) = 32 → 1024 threads (the 1024
    #     threads/threadgroup cap; reg=1 at BLOCK_M=8 needed 64 sg = 2048).
    # Q is cast to fp16 to match the fp16 cache (uniform-dtype tiles, which also
    # keeps the K/V shmem tiles fp16 so the prefill kernel fits ~26 KB < 32 KB);
    # scores accumulate in fp32 regardless. KV-SHARED global layers own no K/V
    # projection — they attend the SOURCE layer's already-written cache READ-ONLY
    # (write_kv=False elides the cache write so the source's live K/V isn't
    # clobbered).
    if self.head_dim > 256:
        cache_position = kwargs["cache_position"]
        if (
            not self.is_kv_shared_layer
            and layer is not None
            and use_alloy_cache_op(layer)
        ):
            # Non-shared global layer, prefill AND decode, through the fused
            # alloy KV-cache op. Q cast to fp16 to match the fp16 cache so the
            # prefill flash kernel's K/V shmem tiles are fp16 (BLOCK_N×512×2B =
            # 8 KB each → ~17 KB total, fits the 32 KB threadgroup budget at
            # head_dim 512). Scores accumulate in fp32.
            attn_output = alloy_cache_attention(
                query_states.to(torch.float16),
                key_states,
                value_states,
                layer,
                cache_position,
                self.scaling,
            )
            if self.store_full_length_kv:
                # Post-update full-length K/V for this type's shared layers
                # (the op wrote this step into layer.keys/values — or, on the
                # quantized-KV path, into the codes/scales set; the consumer
                # branches on tuple arity).
                try:
                    keys_q_full = layer.alloy_keys_q
                except AttributeError:
                    keys_q_full = None
                if keys_q_full is not None:
                    shared_kv_states[self.layer_type] = (
                        keys_q_full, layer.alloy_keys_scales,
                        layer.alloy_values_q, layer.alloy_values_scales,
                    )
                else:
                    shared_kv_states[self.layer_type] = (layer.keys, layer.values)
            attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
            return self.o_proj(attn_output), None
        if self.is_kv_shared_layer and layer is not None and use_alloy_cache_op(layer):
            # KV-shared global layer (prefill AND decode): attend the SOURCE
            # layer's full-length cache read-only via the fused kernel (decode →
            # vector-512; prefill → strided_runtime_pos with the KV-write phase
            # skipped). This layer owns no K/V projection; new_k/new_v are a
            # throwaway [B,KVH,1,D] view (unused — write_kv=False elides the cache
            # write), so the source cache is never mutated here. The source ran
            # earlier in the layer loop and wrote [cache_pos, cache_pos+seq_len)
            # into its full-length cache (allocated at max_cache_len), so the
            # causal mask sees [0, cache_pos+seq_len) and the vector path's
            # kv_len>=256 gate holds.
            shared = shared_kv_states[self.layer_type]
            if len(shared) == 4:
                # Quantized source cache: attend the codes read-only. The
                # new_k/new_v slots are throwaway views (write_kv=False skips
                # the quantize dispatches entirely — re-quantizing here would
                # clobber the source's codes with garbage).
                kq, ks, vq, vs = shared
                attn_output = torch.ops.alloy.attention_cache_q8(
                    query_states.to(torch.float16),
                    kq[:, :, :1, :],
                    vq[:, :, :1, :],
                    cache_position[:1],
                    kq, ks, vq, vs,
                    float(self.scaling),
                    sliding_window=0,
                    write_kv=False,
                )
                attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
                return self.o_proj(attn_output), None
            k_full, v_full = shared
            attn_output = torch.ops.alloy.attention_cache(
                query_states.to(torch.float16),
                k_full[:, :, :1, :],
                v_full[:, :, :1, :],
                cache_position[:1],
                k_full,
                v_full,
                float(self.scaling),
                sliding_window=0,
                write_kv=False,
            )
            attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
            return self.o_proj(attn_output), None
        if self.is_kv_shared_layer:
            # Non-alloy fallback: reused full-length K/V from the source layer.
            k_full, v_full = key_states, value_states
        elif layer is not None and use_alloy_cache_op(layer):
            # Write this step's K/V into the cache at the absolute `cache_position`
            # (NOT layer.update, which writes at the cache's internal
            # cumulative_length — that can disagree with the mask's
            # cache_position across incremental decode steps), then attend over
            # the whole cache. The causal mask drops the unwritten tail.
            kv_k = cache_write_dtype(layer, "keys", key_states.dtype)
            kv_v = cache_write_dtype(layer, "values", value_states.dtype)
            write_k = key_states.to(kv_k)
            write_v = value_states.to(kv_v)
            if not layer.is_initialized or layer.keys is None or layer.values is None:
                layer.lazy_initialization(write_k, write_v)
            layer.keys.index_copy_(2, cache_position, write_k)
            layer.values.index_copy_(2, cache_position, write_v)
            k_full, v_full = layer.keys, layer.values
            if self.store_full_length_kv:
                # Post-update full-length K/V for this type's shared layers.
                shared_kv_states[self.layer_type] = (k_full, v_full)
        else:
            k_full, v_full = key_states, value_states
        attn_output = gemm_cache_attention(
            query_states, k_full, v_full, cache_position, self.scaling
        )
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), None

    # Non-shared layers route through the alloy KV-cache attention op (write +
    # attend on the fp16 cache), exactly like the gemma3 path.
    if layer is not None and use_alloy_cache_op(layer):
        attn_output = alloy_cache_attention(
            query_states, key_states, value_states, layer, kwargs["cache_position"], self.scaling
        )
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), None

    # Shared layers (no cache) — and the non-alloy fallback — attend against the
    # explicit K/V via the standard interface. The reused K/V are the source
    # layer's full-length states at the fp16 cache dtype, so no cache round-trip.
    #
    # Full (non-sliding) shared layers are pure causal attention — drop the mask
    # buffer and let the interface use is_causal. That routes to the non-masked
    # attention kernel, which keeps the `o` accumulator in registers and so fits
    # the 32 KB threadgroup budget at head_dim 512 (the masked-with-lse kernel
    # spills `o` to shmem → overflows at 512). This is exactly equivalent to the
    # causal mask, not an approximation. Sliding shared layers genuinely need the
    # window mask (head_dim 256, fits) — keep it, trimmed to the K/V length (its
    # leading columns are the real positions; the tail is unwritten cache).
    if not self.is_sliding:
        attention_mask = None
    elif attention_mask is not None and attention_mask.shape[-1] != key_states.shape[-2]:
        attention_mask = attention_mask[..., : key_states.shape[-2]]
    # Run the attention at the K/V (fp16 cache) dtype — uniform dtype for the
    # masked attention op and fp16 shmem tiles (head_dim 512 fits 32 KB).
    if query_states.dtype != key_states.dtype:
        query_states = query_states.to(key_states.dtype)
    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, gemma4_eager_attention_forward
    )
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


INSTALLED = False


def install_multi_token_attention(model: nn.Module) -> None:
    """Patch the alloy-aware forward onto every attention class we know
    about. Idempotent. Calls `torch._dynamo.reset()` so any pre-existing
    compiled graphs that captured the old forward are invalidated.

    Patches Qwen3 and Llama unconditionally — the patch is a class-level
    mutation and harmless for models that aren't loaded; if the bench
    later boots a different model class the wired patch is already there.

    Must be called BEFORE the torch.compile()'d wrapper sees its first
    real input — installing mid-stream forces a recompile.
    """
    global INSTALLED
    if INSTALLED:
        return

    Qwen3Attention.forward = alloy_qwen3_attention_forward
    LlamaAttention.forward = alloy_llama_attention_forward
    # Qwen2 attention layout matches Llama (no q_norm/k_norm); bias on QKV
    # is handled inside the Linear modules. Reuse the Llama forward.
    Qwen2Attention.forward = alloy_llama_attention_forward
    # Gemma3 has Qwen3-style q_norm/k_norm but its own forward signature
    # (no `position_ids` positional arg, sliding_window per-layer).
    Gemma3Attention.forward = alloy_gemma3_attention_forward
    # Gemma4: per-layer-type head_dim (256/512), q/k/v-norm, cross-layer KV
    # sharing via shared_kv_states, RoPE at unsqueeze_dim=2, scaling=1.0.
    Gemma4TextAttention.forward = alloy_gemma4_attention_forward
    # Qwen3.5 attention layers — gated full-attention (q_proj 2x wide,
    # output * sigmoid(gate)). The linear-attention/SSM layers are
    # different classes (`Qwen3_5GatedDeltaNet`) and pass through
    # transformers' eager DeltaNet forward unchanged.
    Qwen3_5Attention.forward = alloy_qwen3_5_attention_forward
    # Linear-attention layers — custom op subsumes Conv1d + delta rule
    # + RMSNormGated so the cache mutations stay in-graph.
    Qwen3_5GatedDeltaNet.forward = alloy_qwen3_5_gated_delta_net_forward
    # Drop the `has_previous_state()` Python-bool read in the text model's
    # linear-attn mask helper — it guards-splits cold vs warm prefill and its
    # result is unused on our path (GDN forward reads layer.alloy_attn_mask).
    Qwen3_5TextModel._update_linear_attn_mask = alloy_qwen3_5_update_linear_attn_mask
    # Qwen3.5-MoE: identical backbone, so the same three patches apply to the
    # MoE attention / DeltaNet / text-model classes (only the FFN — the MoE
    # block — differs, and that's driven by GGUFQwen35MoeBlock + gguf_moe_routed).
    Qwen3_5MoeAttention.forward = alloy_qwen3_5_attention_forward
    Qwen3_5MoeGatedDeltaNet.forward = alloy_qwen3_5_gated_delta_net_forward
    Qwen3_5MoeTextModel._update_linear_attn_mask = alloy_qwen3_5_update_linear_attn_mask
    # LFM2: GQA attention (per-head q/k norm, no gate) + short-conv mixer.
    # The conv forward routes through `short_conv_update` so the conv-state
    # cache write stays in-graph; in_proj/out_proj stay as Q4_K linears.
    Lfm2Attention.forward = alloy_lfm2_attention_forward
    Lfm2ShortConv.forward = alloy_lfm2_short_conv_forward
    # Qwen3_5RMSNorm forward uses `output * (1 + weight)`. The GGUF converter
    # pre-adds 1 to norm.weight tensors; the alloy gguf processor (Qwen35TensorProcessor)
    # now subtracts 1 at load time to restore the raw trained weight, matching
    # Gemma's convention. So the HF forward applies the correct `(1 + raw_weight)`
    # multiplier without needing a patch here.

    # Qwen3.5 MRoPE: leave the upstream compile path intact. Python-level
    # overrides (a lambda returning freqs[0], or a full rotary-forward
    # replacement avoiding slice_scatter) regressed end-to-end quality even
    # though they were mathematically equivalent to the intact 1D rotary.
    # The right fix lives deeper in alloy's compile of slice_scatter on the
    # (start=1/2, end=33/30, step=3) MRoPE pattern, not at the Python layer.

    torch._dynamo.reset()
    INSTALLED = True
