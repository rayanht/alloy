"""Static cache helpers for Alloy generation."""

from __future__ import annotations

import ctypes
from collections.abc import Callable, Mapping
from typing import TypeAlias, cast

import torch
import transformers
from transformers.cache_utils import (
    Cache,
    CacheLayerMixin,
    LinearAttentionCacheLayerMixin,
    LinearAttentionLayer,
    StaticCache,
    StaticLayer,
    StaticSlidingWindowLayer,
    is_torchdynamo_compiling,
)

from alloy._runtime import _metal_ext
from alloy_server.kv_format import KVFormat

ConfigValue: TypeAlias = (
    None | bool | int | float | str | list["ConfigValue"] | dict[str, "ConfigValue"]
)
CacheKwargs: TypeAlias = dict[str, object] | None
# nbytes -> alloy buffer handle. None = buf_alloc (Metal-owned pages);
# PagedKV passes a pool-slice allocator so cache tensors land in the pool.
# Stored on each layer (`_alloy_alloc`) so the lazy-init path allocates from
# the same pool as construction.
TensorAlloc: TypeAlias = Callable[[int], int] | None


def _alloc_quantized_kv(
    layer: "AlloyStaticLayer | AlloyStaticSlidingWindowLayer",
    kv_format: KVFormat,
    max_batch_size: int,
    num_kv_heads: int,
    phys_len: int,
    head_dim: int,
    alloc: TensorAlloc = None,
) -> None:
    """Allocate the quantized KV buffer set in place of fp16 keys/values:
    int8 codes (same logical shape) + fp16 scales per 32-elem head_dim block.
    The fp16 K/V tensors are never allocated — that's the memory win; the
    attention path reads/writes codes via `alloy.attention_cache_q8`."""
    assert head_dim % kv_format.block_elems == 0, (head_dim, kv_format.block_elems)
    codes_shape = (max_batch_size, num_kv_heads, phys_len, head_dim)
    scales_shape = (max_batch_size, num_kv_heads, phys_len, head_dim // kv_format.block_elems)
    layer.alloy_keys_q = _alloy_owned_empty(codes_shape, torch.int8, alloc)
    layer.alloy_keys_scales = _alloy_owned_empty(scales_shape, torch.float16, alloc)
    layer.alloy_values_q = _alloy_owned_empty(codes_shape, torch.int8, alloc)
    layer.alloy_values_scales = _alloy_owned_empty(scales_shape, torch.float16, alloc)
    layer._alloy_kv_format = kv_format
    if not is_torchdynamo_compiling():
        for t in (layer.alloy_keys_q, layer.alloy_keys_scales,
                  layer.alloy_values_q, layer.alloy_values_scales):
            torch._dynamo.mark_static_address(t)


class AlloyStaticLayer(StaticLayer):
    def __init__(
        self,
        max_cache_len: int,
        cache_dtype: torch.dtype | None = None,
        *,
        max_batch_size: int = 1,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        kv_format: KVFormat | None = None,
        alloc: TensorAlloc = None,
    ) -> None:
        super().__init__(max_cache_len=max_cache_len)
        self._alloy_cache_dtype = cache_dtype
        self._alloy_alloc = alloc
        # Quantized-KV buffer set (None on the fp16 path) — plain attributes so
        # the patched attention's `is not None` branch is a stable per-layer
        # Dynamo constant, like `_alloy_cache_dtype`.
        self.alloy_keys_q = None
        self.alloy_keys_scales = None
        self.alloy_values_q = None
        self.alloy_values_scales = None
        if kv_format is not None and cache_dtype is not None and num_kv_heads is not None and head_dim is not None:
            _alloc_quantized_kv(self, kv_format, max_batch_size, num_kv_heads, max_cache_len, head_dim, alloc)
            self.dtype = cache_dtype
            self.device = self.alloy_keys_q.device
            self.max_batch_size = max_batch_size
            self.num_heads = num_kv_heads
            self.k_head_dim = head_dim
            self.v_head_dim = head_dim
            self.cumulative_length = self.cumulative_length.to(self.device)
            if not is_torchdynamo_compiling():
                torch._dynamo.mark_static_address(self.cumulative_length)
            self.is_initialized = True
            return
        # Eagerly allocate K/V in alloy-owned memory at construction (mirroring
        # AlloyLinearAttentionLayer's conv/recurrent states) instead of lazily
        # on the first forward. Lazy init calls buf_alloc/buf_ptr (nanobind) +
        # memset/from_address (ctypes) — none traceable by dynamo — so doing it
        # inside the compiled forward forces a graph break per layer per buffer
        # (~50 across a hybrid model, the only avoidable breaks left). Allocating
        # here, outside the compiled region, keeps the prefill forward
        # break-free. Shapes come from config; fall back to lazy init when the
        # head dims aren't known (mark_static_address runs because construction
        # is always in eager mode).
        if cache_dtype is not None and num_kv_heads is not None and head_dim is not None:
            kv_shape = (max_batch_size, num_kv_heads, max_cache_len, head_dim)
            self.keys = _alloy_owned_empty(kv_shape, cache_dtype, alloc)
            self.values = _alloy_owned_empty(kv_shape, cache_dtype, alloc)
            self.dtype = cache_dtype
            self.device = self.keys.device
            self.max_batch_size = max_batch_size
            self.num_heads = num_kv_heads
            self.k_head_dim = head_dim
            self.v_head_dim = head_dim
            self.cumulative_length = self.cumulative_length.to(self.device)
            if not is_torchdynamo_compiling():
                torch._dynamo.mark_static_address(self.keys)
                torch._dynamo.mark_static_address(self.values)
                torch._dynamo.mark_static_address(self.cumulative_length)
            self.is_initialized = True

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        if self._alloy_cache_dtype is None:
            super().lazy_initialization(key_states, value_states)
            return
        _alloy_lazy_initialization(self, key_states, value_states)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: CacheKwargs = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # HF's StaticLayer.update increments self.cumulative_length per call,
        # producing one tiny add_cast_i64 dispatch per layer per step —
        # every full_attention layer advances in lockstep, so the per-layer
        # increment is redundant work. Skip the in-graph increment here;
        # AlloyStaticCache aliases one cumulative_length tensor across all
        # AlloyStaticLayers so a single wrapper-level .add_() advances
        # every layer's view at once.
        write_key_states, write_value_states = _cache_write_views(self, key_states, value_states)
        if not self.is_initialized:
            self.lazy_initialization(write_key_states, write_value_states)
        kv_length = write_key_states.shape[-2]
        # For decode (kv_length==1), cumulative_length is shape (1,) so reuse
        # it directly as cache_position — avoids a per-layer arange + add.
        if kv_length == 1:
            cache_position = self.cumulative_length
        else:
            cache_position = (
                torch.arange(kv_length, device=self.device) + self.cumulative_length
            )
        try:
            self.keys.index_copy_(2, cache_position, write_key_states)
            self.values.index_copy_(2, cache_position, write_value_states)
        except NotImplementedError:
            self.keys[:, :, cache_position] = write_key_states
            self.values[:, :, cache_position] = write_value_states
        return _attention_dtype_views(
            self.keys, self.values, key_states, value_states
        )

    def get_seq_length(self, cache_position: torch.Tensor | None = None) -> int:
        # `update()` skips the per-layer `cumulative_length_int` increment (the wrapper
        # advances the aliased `cumulative_length` tensor instead), so the base
        # StaticLayer.get_seq_length — which returns that int — is frozen at 0. Models
        # that derive `position_ids = arange(seq) + get_seq_length()` (whisper's decoder)
        # then pin every decode token to position 0. Return the aliased tensor, like
        # AlloyStaticSlidingWindowLayer. (LLMs dodge this by using cache_position.)
        return self.cumulative_length if self.is_initialized else 0


class AlloyStaticSlidingWindowLayer(StaticSlidingWindowLayer):
    def __init__(
        self,
        max_cache_len: int,
        sliding_window: int,
        cache_dtype: torch.dtype | None = None,
        *,
        max_batch_size: int = 1,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        kv_format: KVFormat | None = None,
        alloc: TensorAlloc = None,
    ) -> None:
        super().__init__(max_cache_len=max_cache_len, sliding_window=sliding_window)
        self._alloy_cache_dtype = cache_dtype
        self._alloy_alloc = alloc
        self.alloy_keys_q = None
        self.alloy_keys_scales = None
        self.alloy_values_q = None
        self.alloy_values_scales = None
        if kv_format is not None and cache_dtype is not None and num_kv_heads is not None and head_dim is not None:
            # token-granular formats are ring-safe (whole-byte codes per token);
            # tiled formats are not — refuse rather than corrupt.
            if kv_format.group_tokens != 1:
                raise ValueError(
                    f"KV format {kv_format.name!r} (group_tokens="
                    f"{kv_format.group_tokens}) is not ring-safe for sliding-window layers"
                )
            # `self.max_cache_len` is the HF-capped window size after super().__init__.
            _alloc_quantized_kv(self, kv_format, max_batch_size, num_kv_heads, self.max_cache_len, head_dim, alloc)
            self.dtype = cache_dtype
            self.device = self.alloy_keys_q.device
            self.max_batch_size = max_batch_size
            self.num_heads = num_kv_heads
            self.k_head_dim = head_dim
            self.v_head_dim = head_dim
            if isinstance(self.cumulative_length, torch.Tensor):
                self.cumulative_length = self.cumulative_length.to(self.device)
                if not is_torchdynamo_compiling():
                    torch._dynamo.mark_static_address(self.cumulative_length)
            self.is_initialized = True
            return
        # Eagerly allocate K/V here (mirroring AlloyStaticLayer) instead of lazily
        # on the first forward. Lazy init runs INSIDE the compiled forward where
        # `is_torchdynamo_compiling()` is True, so `_alloy_lazy_initialization`
        # SKIPS `mark_static_address` — leaving the sliding cache buffers as
        # non-static addresses. Dynamo then mishandles them (symbolic-shape /
        # changing-address path), silently producing garbage output that
        # compounds over the sliding layers. AlloyStaticLayer dodges this because
        # it allocates at construction (eager mode). `self.max_cache_len` is the
        # HF-capped window size after `super().__init__`.
        if cache_dtype is not None and num_kv_heads is not None and head_dim is not None:
            kv_shape = (max_batch_size, num_kv_heads, self.max_cache_len, head_dim)
            self.keys = _alloy_owned_empty(kv_shape, cache_dtype, alloc)
            self.values = _alloy_owned_empty(kv_shape, cache_dtype, alloc)
            self.dtype = cache_dtype
            self.device = self.keys.device
            self.max_batch_size = max_batch_size
            self.num_heads = num_kv_heads
            self.k_head_dim = head_dim
            self.v_head_dim = head_dim
            if isinstance(self.cumulative_length, torch.Tensor):
                self.cumulative_length = self.cumulative_length.to(self.device)
            if not is_torchdynamo_compiling():
                torch._dynamo.mark_static_address(self.keys)
                torch._dynamo.mark_static_address(self.values)
                if isinstance(self.cumulative_length, torch.Tensor):
                    torch._dynamo.mark_static_address(self.cumulative_length)
            self.is_initialized = True

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        if self._alloy_cache_dtype is None:
            super().lazy_initialization(key_states, value_states)
            return
        _alloy_lazy_initialization(self, key_states, value_states)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: CacheKwargs = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        write_key_states, write_value_states = _cache_write_views(self, key_states, value_states)
        keys, values = super().update(write_key_states, write_value_states, cache_kwargs)
        return _attention_dtype_views(keys, values, key_states, value_states)

    def get_seq_length(self) -> int:
        # HF StaticSlidingWindowLayer.get_seq_length returns `cumulative_length_int`,
        # a Python int advanced only inside `update()`. The alloy decode path routes
        # attention through the `attention_kv_update` custom op and BYPASSES
        # `update()`, so `cumulative_length_int` is frozen at 0 — and the model's
        # `position_ids = arange(seq) + past_key_values.get_seq_length()` then pins
        # every decoded token's RoPE to position 0 while the cache is written at the
        # correct slot, so Q/K rotations no longer match the cached keys and decode
        # output is garbage. (Gemma3 hits this because its layer 0 is a sliding
        # layer, which is what the cache-level `get_seq_length` reads.) Return the
        # aliased `cumulative_length` tensor instead — the one the generation wrapper
        # advances with a single in-graph `.add_()` — mirroring AlloyStaticLayer /
        # StaticLayer.get_seq_length and keeping decode positions correct.
        return self.cumulative_length if self.is_initialized else 0


# Speculative-verify slot-bank width for DeltaNet recurrent states. attach_spec
# sets this to the drafter's verify width BEFORE any cache is constructed
# (single-user server contract — same pattern as set_use_alloy_warm_op). Plain
# decode reads ONLY slot 0, so the default is 1 — a wider bank is pure waste
# without a drafter (the bank is `recurrent_states`' leading dim, replicated
# per layer AND per paged KV slice: at width 8 on qwen3.5:4b that was ~2.7 GB
# committed across the 8-slice table for a feature that wasn't attached).
_SPEC_SLOT_BANK = 1


def set_spec_slot_bank(value: int) -> None:
    global _SPEC_SLOT_BANK
    _SPEC_SLOT_BANK = int(value)


class AlloyLinearAttentionLayer(LinearAttentionLayer):
    """Linear-attention (GatedDeltaNet) layer cache for Qwen 3.5.

    Hybrid stack interleaves these with full-attention layers
    (3 linear : 1 full, per `config.layer_types`). The linear layer
    keeps two state tensors:

      - `conv_states`:      (B, conv_dim, conv_kernel_size) — rolling
        window of past inputs for the depthwise causal Conv1d.
      - `recurrent_states`: (B, num_v_heads, head_k_dim, head_v_dim) —
        GatedDeltaNet recurrent state.

    Unlike the upstream `LinearAttentionLayer` which lazy-initialises
    in `update_conv_state` / `update_recurrent_state`, we eagerly
    allocate both states at construction time in alloy-owned memory.
    Reason: the alloy custom op `linear_attention_update` reads
    `layer.conv_states` / `layer.recurrent_states` directly as op
    inputs — they need to be real tensors at FX-trace time, not None.
    Eager allocation also guarantees stable storage_ptrs across calls
    so Dynamo doesn't recompile when the lazy-init flips.

    `_alloy_cache_dtype = None` keeps `use_alloy_cache_op(layer)`
    returning False for linear layers — the gated-attention forward
    patch only routes ATTENTION layers through `alloy_cache_attention`.
    The DeltaNet path uses the dedicated `linear_attention_update`
    custom op directly.
    """

    _alloy_cache_dtype: torch.dtype | None = None

    def __init__(
        self,
        config: transformers.PretrainedConfig | None = None,
        *,
        max_batch_size: int = 1,
        cache_dtype: torch.dtype | None = None,
        alloc: TensorAlloc = None,
    ) -> None:
        super().__init__(config=config)
        self._alloy_alloc = alloc
        if config is None:
            return
        if hasattr(config, "get_text_config"):
            text = config.get_text_config(decoder=True)
        else:
            text = config
        # Direct attribute access — errors explicitly if missing so we
        # catch new hybrid models that diverge from this contract.
        conv_kernel_size = int(text.linear_conv_kernel_dim)
        num_k_heads = int(text.linear_num_key_heads)
        num_v_heads = int(text.linear_num_value_heads)
        head_k_dim = int(text.linear_key_head_dim)
        head_v_dim = int(text.linear_value_head_dim)
        key_dim = num_k_heads * head_k_dim
        value_dim = num_v_heads * head_v_dim
        conv_dim = key_dim * 2 + value_dim
        # DeltaNet states are kept in fp32 regardless of the KV cache_dtype.
        # The recurrent kernel computes in fp32 and writes back; saving as fp16
        # truncates each step, and errors compound across decode iterations to
        # produce degenerate outputs after a handful of tokens.
        dtype = torch.float32
        conv_shape = (max_batch_size, conv_dim, conv_kernel_size)
        # Leading dim = K-step bank (the speculative-verify width). Decode and
        # prefill write only slot [0] (the recurrent kernel's SAVE_STEPS=0 path,
        # which uses `state_col_addr` = offset into slot 0); a serial SAVE_STEPS
        # spec verify writes per-token states [0..S-1] so a partial-accept
        # rollback is a free slot-copy instead of a target re-run. Extra slots
        # are inert otherwise. Sized by `set_spec_slot_bank` (attach_spec sets
        # it BEFORE any cache is built): 8 covers the small-width slot-bank
        # drafters (MTP/PLD); chunk-aligned widths (DFlash block 16) use the
        # dvblock+reconstruct verify, which never writes past slot 0, so
        # attach_spec sizes the bank to 1 (vs 16 slots ≈ +400MB resident on
        # qwen3.5:4b).
        rec_shape = (_SPEC_SLOT_BANK, max_batch_size, num_v_heads, head_k_dim, head_v_dim)
        self.conv_states = _alloy_owned_empty(conv_shape, dtype, alloc)
        self.recurrent_states = _alloy_owned_empty(rec_shape, dtype, alloc)
        # Pad mask buffer: int64 (1, max_prefill_bucket) — 1 for real tokens,
        # 0 for pads. Pre-allocated so the patched DeltaNet forward can read
        # it via `layer.alloy_attn_mask` (a normal tensor attribute that
        # dynamo traces) instead of going through a Python dict lookup that
        # forces a graph break and disables Tensor(c!) mutation propagation
        # on the linear_attention_update op. Updated via .copy_() each
        # prefill call. Initialised to all-ones so decode (seq_len=1)
        # always sees "no pads".
        # Must span the longest single prefill. Chunked prefill caps at the chunk
        # size, but grid-shrunk chunk prefill drives seq_len up to the model's native
        # context (the cache is sized there too), so a hardcoded 4096 cap would
        # clamp the mask (`full_mask[:, :seq_len]`) and corrupt DeltaNet pad
        # handling for M_MAX > 4096. int64 × native ≈ a few MB/layer — negligible.
        max_mask = int(text.max_position_embeddings)
        self.alloy_attn_mask = _alloy_owned_empty((max_batch_size, max_mask), torch.int64, alloc)
        self.alloy_attn_mask.fill_(1)
        self.is_conv_states_initialized = True
        self.is_recurrent_states_initialized = True
        self.dtype = dtype
        self.device = self.conv_states.device
        self.max_batch_size = max_batch_size
        self.conv_kernel_size = conv_kernel_size
        if not is_torchdynamo_compiling():
            torch._dynamo.mark_static_address(self.conv_states)
            torch._dynamo.mark_static_address(self.recurrent_states)
            torch._dynamo.mark_static_address(self.alloy_attn_mask)


class AlloyShortConvLayer(AlloyLinearAttentionLayer):
    """LFM2 short-conv (conv-mixer) layer cache.

    The conv-only sibling of `AlloyLinearAttentionLayer`: LFM2's mixer is a
    causal depthwise Conv1d with NO recurrent rule, so it keeps ONLY the rolling
    `conv_states` window — `recurrent_states` stays None. Subclassing
    `AlloyLinearAttentionLayer` means every `isinstance(layer,
    AlloyLinearAttentionLayer)` hybrid check (kv.py / prefix.py / spec.py)
    automatically treats it as position-bound state; those sites guard
    `recurrent_states` with `is not None`.

    `conv_states` is `(B, hidden_size, conv_L_cache)` — the conv operates on the
    `B * x` channels (hidden_size wide), kernel length `conv_L_cache`. The pad
    mask (`alloy_attn_mask`) is carried so the patched conv forward derives the
    real prompt length for the conv-state save in a padded prefill chunk.
    """

    def __init__(
        self,
        config: transformers.PretrainedConfig | None = None,
        *,
        max_batch_size: int = 1,
        cache_dtype: torch.dtype | None = None,
        alloc: TensorAlloc = None,
    ) -> None:
        # Skip AlloyLinearAttentionLayer.__init__ (it reads DeltaNet-only config
        # keys and allocates recurrent_states); go straight to the mixin base.
        LinearAttentionLayer.__init__(self, config=config)
        self._alloy_alloc = alloc
        if config is None:
            return
        text = config.get_text_config(decoder=True) if hasattr(config, "get_text_config") else config
        conv_kernel_size = int(text.conv_L_cache)
        conv_dim = int(text.hidden_size)
        # Conv state kept in fp32 (the conv kernels accumulate in fp32; the
        # window is tiny so fp16 buys nothing and would truncate the carry).
        dtype = torch.float32
        conv_shape = (max_batch_size, conv_dim, conv_kernel_size)
        self.conv_states = _alloy_owned_empty(conv_shape, dtype, alloc)
        self.recurrent_states = None
        max_mask = int(text.max_position_embeddings)
        self.alloy_attn_mask = _alloy_owned_empty((max_batch_size, max_mask), torch.int64, alloc)
        self.alloy_attn_mask.fill_(1)
        self.is_conv_states_initialized = True
        self.is_recurrent_states_initialized = False
        self.dtype = dtype
        self.device = self.conv_states.device
        self.max_batch_size = max_batch_size
        self.conv_kernel_size = conv_kernel_size
        if not is_torchdynamo_compiling():
            torch._dynamo.mark_static_address(self.conv_states)
            torch._dynamo.mark_static_address(self.alloy_attn_mask)


class AlloyStaticCache(StaticCache):
    """Static cache with Alloy decode semantics.

    When ``cache_dtype`` is set, layers allocate K/V in that dtype from the
    first lazy initialization. Patched attention routes mixed fp32-Q/fp16-KV
    calls through Alloy custom ops so Dynamo never has to trace upstream SDPA
    with mismatched Q/K/V dtypes.

    All ``AlloyStaticLayer`` instances share a single ``cumulative_length``
    tensor so the model wrapper can advance every layer's view with one
    in-graph ``.add_()`` instead of 28. See ``AlloyStaticLayer.update``.
    """

    def __init__(
        self,
        config: transformers.PretrainedConfig,
        max_cache_len: int,
        max_batch_size: int = 1,
        offloading: bool = False,
        offload_only_non_sliding: bool = True,
        cache_dtype: torch.dtype | None = None,
        kv_format: KVFormat | None = None,
        alloc: TensorAlloc = None,
    ) -> None:
        text_config = config.get_text_config(decoder=True)
        values = cast(dict[str, ConfigValue], text_config.to_dict())
        # Encoder-decoder models (whisper) carry the decoder geometry as
        # `decoder_layers` / `decoder_attention_heads` / `d_model` on the full
        # config; `get_text_config(decoder=True).to_dict()` drops them. Use the full
        # config dict so `_layer_count` / `_full_attn_kv_dims` read the decoder fields.
        if text_config.to_dict().get("is_encoder_decoder"):
            values = cast(dict[str, ConfigValue], config.to_dict())
        layer_types = _cache_layer_types(values)
        layers: list[CacheLayerMixin] = [
            _make_cache_layer(
                layer_type, max_cache_len, max_batch_size, config, values, cache_dtype,
                kv_format, alloc,
            )
            for layer_type in layer_types
        ]

        # Alias cumulative_length across every attention layer — full AND
        # sliding-window. Combined with AlloyStaticLayer.update skipping the
        # per-layer .add_, this lets the model wrapper do ONE in-graph
        # cumulative_length.add_(kv_length) to advance every layer's view of
        # the current position.
        #
        # Sliding layers MUST be included: in a hybrid stack (Gemma3's 5:1
        # sliding:full pattern) layer 0 is a sliding-window layer, and
        # generation.py advances `layers[0].cumulative_length`. The model's
        # `position_ids = arange(seq) + past_key_values.get_seq_length()`
        # reads the cumulative_length off a (different) layer; if the sliding
        # layers carry their own un-aliased tensors, the wrapper's advance
        # never reaches the layer get_seq_length() inspects. That pins decode
        # `position_ids` at 0 — every decoded token gets RoPE for position 0
        # while the cache is written at the correct slot, so Q/K rotations
        # mismatch the cached keys and decode produces garbage. (Full-attn
        # models dodged this only because layer 0 was already the shared
        # tensor.) Alias every alloy attention layer to ONE tensor so the
        # single wrapper advance moves all views, sliding included.
        #
        # For hybrid caches (qwen3.5: linear-attention layers interleaved with
        # full attention), the linear layers don't normally carry a
        # cumulative_length attribute. Alias the shared tensor onto linear
        # layers as well so the wrapper-level access stays uniform.
        shared: torch.Tensor | None = None
        for layer in layers:
            if not isinstance(layer, (AlloyStaticLayer, AlloyStaticSlidingWindowLayer)):
                continue
            if shared is None:
                shared = layer.cumulative_length
            else:
                layer.cumulative_length = shared
        if shared is not None:
            for layer in layers:
                if isinstance(layer, LinearAttentionCacheLayerMixin):
                    layer.cumulative_length = shared

        # Chunk's last REAL row index ((1,) int64, -1 = no bound): the sliding
        # ring write's single-writer bound, passed into the attention cache
        # ops as the `last_real` operand by the patched attention. ONE tensor
        # aliased onto every attention layer (the cumulative_length pattern);
        # PrefillEngine.chunk_step fills it with real_len-1 per chunk and
        # resets to -1 so decode, spec verify, and unmanaged forwards write
        # unbounded (fail-open).
        self.alloy_last_real = _alloy_owned_empty((1,), torch.int64, alloc)
        self.alloy_last_real.fill_(-1)
        if not is_torchdynamo_compiling():
            torch._dynamo.mark_static_address(self.alloy_last_real)
        for layer in layers:
            if isinstance(layer, (AlloyStaticLayer, AlloyStaticSlidingWindowLayer)):
                layer.alloy_last_real = self.alloy_last_real

        Cache.__init__(
            self,
            layers=layers,
            offloading=offloading,
            offload_only_non_sliding=offload_only_non_sliding,
        )


def _alloy_lazy_initialization(
    layer: AlloyStaticLayer | AlloyStaticSlidingWindowLayer,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> None:
    # Callers (AlloyStaticLayer.lazy_initialization / AlloyStaticSlidingWindowLayer)
    # only reach this branch after checking `_alloy_cache_dtype is not None`,
    # so direct access is safe (and `dtype` is not None here).
    dtype = cast(torch.dtype, layer._alloy_cache_dtype)
    key_states = key_states.to(dtype=dtype) if key_states.dtype != dtype else key_states
    value_states = value_states.to(dtype=dtype) if value_states.dtype != dtype else value_states

    layer.dtype, layer.device = key_states.dtype, key_states.device
    layer.max_batch_size, layer.num_heads = key_states.shape[:2]
    layer.v_head_dim = value_states.shape[-1]
    layer.k_head_dim = key_states.shape[-1]
    layer.keys = _alloy_owned_empty(
        (layer.max_batch_size, layer.num_heads, layer.max_cache_len, layer.k_head_dim),
        dtype=layer.dtype,
        alloc=layer._alloy_alloc,
    )
    layer.values = _alloy_owned_empty(
        (layer.max_batch_size, layer.num_heads, layer.max_cache_len, layer.v_head_dim),
        dtype=layer.dtype,
        alloc=layer._alloy_alloc,
    )
    layer.cumulative_length = layer.cumulative_length.to(layer.device)
    if not is_torchdynamo_compiling():
        torch._dynamo.mark_static_address(layer.keys)
        torch._dynamo.mark_static_address(layer.values)
        torch._dynamo.mark_static_address(layer.cumulative_length)
    layer.is_initialized = True


def _alloy_owned_empty(
    shape: tuple[int, ...], dtype: torch.dtype, alloc: TensorAlloc = None,
) -> torch.Tensor:
    itemsize = torch.empty((), dtype=dtype).element_size()
    nbytes = itemsize
    for dim in shape:
        nbytes *= int(dim)
    handle = (alloc or _metal_ext.buf_alloc)(nbytes)
    aligned_ptr = _metal_ext.buf_ptr(handle)
    # No memset: a Metal shared buffer is anonymous memory, so macOS zero-fills
    # each page on first fault — it reads as zero until written, identical to an
    # explicit memset but without committing physical pages up front. This keeps
    # allocation lazy so a full-context KV cache (e.g. qwen3.5's 256k) costs only
    # the pages actually filled, not its entire virtual size. Reuse paths that
    # need a clean slate (linear-attention recurrent/conv state) re-zero
    # explicitly; full-attention KV is never read past cumulative_length (causal
    # mask), so it needs no init. (Mirrors buf_alloc, which dropped its memset in
    # 4ea7f64 for the same reason.)
    raw = (ctypes.c_uint8 * nbytes).from_address(aligned_ptr)
    flat = torch.frombuffer(raw, dtype=torch.uint8)
    new_storage = flat.untyped_storage()
    new_storage._alloy_keepalive = (raw, handle)  # type: ignore[attr-defined]
    out = torch.empty(0, dtype=dtype)
    out.set_(new_storage, 0, shape, _contiguous_strides(shape))
    return out


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides: list[int] = []
    stride = 1
    for dim in reversed(shape):
        strides.append(stride)
        stride *= int(dim)
    return tuple(reversed(strides))


def _attention_dtype_views(
    keys: torch.Tensor,
    values: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return keys, values


def _cache_write_views(
    layer: StaticLayer,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    keys = layer.keys
    values = layer.values
    if not layer.is_initialized or keys is None or values is None:
        # `_alloy_cache_dtype` is only declared on Alloy*Layer subclasses; the
        # upstream `StaticLayer` doesn't have it. AttributeError signals
        # "no preferred cache dtype" — return inputs unchanged.
        try:
            target_dtype = layer._alloy_cache_dtype  # type: ignore[attr-defined]
        except AttributeError:
            target_dtype = None
        if target_dtype is not None:
            key_states = key_states.to(dtype=target_dtype)
            value_states = value_states.to(dtype=target_dtype)
        return key_states, value_states
    write_key_states = key_states if key_states.dtype == keys.dtype else key_states.to(dtype=keys.dtype)
    write_value_states = (
        value_states if value_states.dtype == values.dtype else value_states.to(dtype=values.dtype)
    )
    return write_key_states, write_value_states


def _make_cache_layer(
    layer_type: str,
    max_cache_len: int,
    max_batch_size: int,
    config: transformers.PretrainedConfig,
    values: Mapping[str, ConfigValue],
    cache_dtype: torch.dtype | None,
    kv_format: KVFormat | None = None,
    alloc: TensorAlloc = None,
) -> AlloyStaticLayer | AlloyStaticSlidingWindowLayer | AlloyLinearAttentionLayer:
    # Sliding/chunked layers stay fp16 even when a KV format is active: their
    # cache is BOUNDED (window-sized — no growing-KV memory or bandwidth win),
    # and their chunk>window prefill needs the cold-path linear-temp machinery
    # (`attention_prefill_cold`) that the unified q8 op's materialize fallback
    # does not build — quantizing them produced all-<pad> gemma4 output even
    # with the single-writer ring bound in place. The growing full-attention
    # layers (gemma4: the 7 d512 globals) are the quantization win and DO
    # quantize.
    if layer_type == "sliding_attention":
        num_kv_heads, head_dim = _full_attn_kv_dims(values)
        return AlloyStaticSlidingWindowLayer(
            max_cache_len=max_cache_len,
            sliding_window=_required_int(values, "sliding_window"),
            cache_dtype=cache_dtype,
            max_batch_size=max_batch_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            alloc=alloc,
        )
    if layer_type == "chunked_attention":
        num_kv_heads, head_dim = _full_attn_kv_dims(values)
        return AlloyStaticSlidingWindowLayer(
            max_cache_len=max_cache_len,
            sliding_window=_required_int(values, "attention_chunk_size"),
            cache_dtype=cache_dtype,
            max_batch_size=max_batch_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            alloc=alloc,
        )
    if layer_type == "linear_attention":
        # GatedDeltaNet layer — pre-allocated conv_state + recurrent_state.
        return AlloyLinearAttentionLayer(
            config=config,
            max_batch_size=max_batch_size,
            cache_dtype=cache_dtype,
            alloc=alloc,
        )
    if layer_type == "conv":
        # LFM2 short-conv layer — pre-allocated conv_state only (no recurrent).
        return AlloyShortConvLayer(
            config=config,
            max_batch_size=max_batch_size,
            cache_dtype=cache_dtype,
            alloc=alloc,
        )
    num_kv_heads, head_dim = _full_attn_kv_dims(values)
    # gemma4 full-attention layers use a wider per-head dim (`global_head_dim`,
    # 512) than its sliding layers (`head_dim`, 256); the cache K/V for these
    # layers must match or the global-layer Q@Kᵀ shapes mismatch. Other models
    # don't set global_head_dim, so this is a no-op for them.
    global_head_dim = _int_value(values, "global_head_dim")
    if global_head_dim is not None:
        head_dim = global_head_dim
    return AlloyStaticLayer(
        max_cache_len=max_cache_len,
        cache_dtype=cache_dtype,
        max_batch_size=max_batch_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        kv_format=kv_format,
        alloc=alloc,
    )


def _full_attn_kv_dims(
    values: Mapping[str, ConfigValue],
) -> tuple[int | None, int | None]:
    """KV head count + per-head dim for eager K/V allocation, or (None, None)
    if the config doesn't expose them (then the layer lazy-inits as before)."""
    # `decoder_attention_heads` is the encoder-decoder (whisper) name for the head count.
    n_heads = _int_value(values, "num_attention_heads") or _int_value(values, "decoder_attention_heads")
    num_kv_heads = _int_value(values, "num_key_value_heads")
    if num_kv_heads is None:
        # No GQA field (whisper MHA, older MHA configs): KV heads == attention heads.
        # GQA models always set num_key_value_heads, so this never changes them.
        num_kv_heads = n_heads
    head_dim = _int_value(values, "head_dim")
    if head_dim is None:
        hidden = _int_value(values, "hidden_size") or _int_value(values, "d_model")
        head_dim = (hidden // n_heads) if (hidden and n_heads) else None
    return num_kv_heads, head_dim


def _cache_layer_types(values: Mapping[str, ConfigValue]) -> tuple[str, ...]:
    explicit_layer_types = _string_tuple(values, "layer_types")
    layer_count = _layer_count(values)
    if explicit_layer_types is not None:
        layer_types = explicit_layer_types
    elif _int_value(values, "sliding_window") is not None:
        layer_types = tuple("sliding_attention" for _ in range(layer_count))
    elif _int_value(values, "attention_chunk_size") is not None:
        layer_types = tuple("chunked_attention" for _ in range(layer_count))
    else:
        layer_types = tuple("full_attention" for _ in range(layer_count))

    shared_layers = _int_value(values, "num_kv_shared_layers")
    if shared_layers is None or shared_layers == 0:
        return layer_types
    # gemma4 (identified by global_head_dim): its trailing `num_kv_shared_layers`
    # reuse an earlier layer's K/V. Those reused K/V are head_dim 512 on the full
    # layers, and the cache-LESS attention kernel overflows the 32 KB threadgroup
    # budget at 512 — whereas the `attention_cache` kernel (used by cached layers)
    # fits. So keep a cache entry for every gemma4 layer and route the shared
    # layers' (reused) K/V through the cached attention op. The K/V values are
    # identical to the source layer's, so attention is exact; only the
    # memory-sharing optimization is given up (a follow-up). Other models keep
    # the trimmed (shared) cache.
    if _int_value(values, "global_head_dim") is not None:
        return layer_types
    return layer_types[:-shared_layers]


def _layer_count(values: Mapping[str, ConfigValue]) -> int:
    layer_count = _int_value(values, "num_hidden_layers")
    if layer_count is not None:
        return layer_count
    # Encoder-decoder (whisper): the cache covers the decoder's self-attention.
    layer_count = _int_value(values, "decoder_layers")
    if layer_count is not None:
        return layer_count
    return _required_int(values, "n_layer")


def _required_int(values: Mapping[str, ConfigValue], key: str) -> int:
    value = _int_value(values, key)
    if value is None:
        raise ValueError(f"model config is missing integer field {key!r}")
    return value


def _int_value(values: Mapping[str, ConfigValue], key: str) -> int | None:
    value = values.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _string_tuple(values: Mapping[str, ConfigValue], key: str) -> tuple[str, ...] | None:
    value = values.get(key)
    if not isinstance(value, list):
        return None
    strings: list[str] = []
    for item in value:
        if not isinstance(item, str):
            return None
        strings.append(item)
    return tuple(strings)
