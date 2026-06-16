"""KVStore: the cache-ownership seam between generation engines and KV memory.

ContiguousKV is the default implementation — one StaticCache per (batch,
max_len), always sized to the model's native context, with the machine-derived
fill budget. PagedKV is the opt-in paged variant behind the same interface.
"""

from __future__ import annotations

import ctypes
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

import torch
import transformers
from transformers.cache_utils import StaticCache

from alloy import get_logger
from alloy._runtime import _metal_ext
from alloy_server.cache import (
    AlloyLinearAttentionLayer,
    AlloyStaticCache,
    AlloyStaticLayer,
)
from alloy_server.kv_format import KVFormat

if TYPE_CHECKING:
    from alloy_server.speculative.session import SpecSession

logger = get_logger("alloy_server.generation")


class KVStore(Protocol):
    """What the engines need from a KV backing store."""

    def acquire(self, batch_size: int, max_len: int) -> StaticCache: ...
    def cache_len_for(self, required: int) -> int: ...
    @property
    def max_cache_len(self) -> int: ...
    @property
    def max_fill(self) -> int: ...
    def fit_to_budget(self, prompt_len: int, max_new_tokens: int, *, extra: int = 0) -> int: ...
    def clear_tail(self, cache: StaticCache, start: int) -> None: ...
    def reclaim_beyond(self, cache: StaticCache, start: int) -> int: ...
    def pressure_level(self) -> int: ...


class ContiguousKV:
    """Per-(batch, max_len) persistent StaticCaches + the fill budget."""

    # Leave 10% of the working set for activations, scratch buffers, and the
    # fixed (sliding-window + linear-attention) cache state the per-token
    # figure excludes; never budget below a usable floor.
    FILL_HEADROOM_FRAC = 0.10
    MIN_FILL = 512

    def __init__(
        self,
        model: transformers.PreTrainedModel,
        cache_dtype: torch.dtype,
        kv_format: KVFormat | None,
        max_cache_len: int,
        bookmark_slots: int,
    ) -> None:
        self.model = model
        self.cache_dtype = cache_dtype
        self.kv_format = kv_format
        self.native_len = max_cache_len
        # Worst-case prefix-bookmark count, reserved out of the fill budget.
        self.bookmark_slots = bookmark_slots
        # Derived lazily from the machine's memory budget on first use.
        self.fill_budget: int | None = None
        # Attached spec-decode session; its drafter state grows per token and
        # is counted into the fill budget. Set by AlloyGenerator.attach_spec.
        self.spec: SpecSession | None = None
        # Per-(batch_size, max_len) cache instances, reused across
        # conversations so the alloy backend's storage_ptr-keyed
        # _cached_input_check doesn't invalidate the warm-prefill plan
        # on every new conversation.
        self.persistent_caches: dict[tuple[int, int], StaticCache] = {}
        # Bumped whenever the cache tensors are repointed to a different
        # slice's storage. Pinned-plan consumers that replay with
        # args_stable (spec verify) and per-storage scratch (the DeltaNet
        # conv tape / GDR round buffers, keyed by data_ptr) watch this to
        # rebind — they would otherwise target the slice the plan was pinned
        # against. 0 and never-incremented on ContiguousKV (no slices).
        self.slice_epoch = 0

    def cache_len_for(self, required: int) -> int:
        """The KV-cache length for a request needing `required` positions. The
        cache is always the model's native context (a single size — no buckets).
        The memory budget is enforced upstream by `fit_to_budget` (which clamps
        generation to fit), so this is just a final native-bound safety net."""
        if required > self.native_len:
            raise ValueError(
                f"request needs {required} positions, exceeding the model's native "
                f"context {self.native_len}"
            )
        return self.native_len

    @property
    def max_cache_len(self) -> int:
        """The KV-cache size — the model's native context."""
        return self.native_len

    @property
    def max_fill(self) -> int:
        """Max KV positions a single request may fill, derived from the machine's
        GPU working-set budget (Metal `recommendedMaxWorkingSetSize`) minus the
        resident model weights — so a long context can't exhaust memory on a
        small machine."""
        if self.fill_budget is None:
            self.fill_budget = self.derive_fill_budget()
        return self.fill_budget

    def fit_to_budget(self, prompt_len: int, max_new_tokens: int, *, extra: int = 0) -> int:
        """Clamp `max_new_tokens` so prompt + generation fits the memory budget
        (`max_fill`); raise a clear, actionable error if the prompt alone does
        not fit. `extra` reserves positions for transient overshoot (e.g. the
        spec-decode verify window). The cache stays native regardless."""
        budget = self.max_fill
        if prompt_len + extra + 1 > budget:
            raise ValueError(
                f"prompt is {prompt_len} tokens, but this machine's memory budget "
                f"fits ~{budget} tokens of context (model native is "
                f"{self.native_len}). Reduce the prompt or use a smaller model."
            )
        room = budget - prompt_len - extra - 1
        return min(max_new_tokens, room)

    def derive_fill_budget(self) -> int:
        """positions = (working_set·(1−headroom) − weight_bytes) / kv_bytes_per_token,
        clamped to [MIN_FILL, native]. Any failure → native (no effective cap):
        the budget is a safety valve, never a hard gate on a capable machine."""
        try:
            # scoped: GPU/Metal runtime dep — keep generation importable without a device
            from alloy._runtime.metal import default_device
            working_set = int(default_device().recommended_max_working_set_size)
        except Exception:  # noqa: BLE001 -- no device signal → don't bound fill
            return self.native_len
        if working_set <= 0:
            return self.native_len
        per_token = self.kv_bytes_per_token()
        if per_token <= 0:
            return self.native_len
        avail = (
            working_set * (1.0 - self.FILL_HEADROOM_FRAC)
            - self.weight_bytes()
            - self.bookmark_budget_bytes()
        )
        budget = int(avail // per_token)
        return max(self.MIN_FILL, min(self.native_len, budget))

    def bookmark_budget_bytes(self) -> int:
        """Worst-case bytes the prefix-bookmark deque can pin: maxlen copies
        of the position-bound state (every cache-layer tensor without a
        cache_len-sized dim — the hybrid's recurrent/conv state). Subtracted
        from the fill budget so bookmarks can't push the working set past
        the headroom the budget promises."""
        cache = self.acquire(1, self.native_len)
        state_bytes = 0
        for layer in cache.layers:
            for value in vars(layer).values():
                if isinstance(value, torch.Tensor) and self.native_len not in value.shape:
                    state_bytes += value.numel() * value.element_size()
        return state_bytes * self.bookmark_slots

    def kv_bytes_per_token(self) -> int:
        """Bytes the KV cache grows by per added token, summed over layers with a
        growing cache. Only full-attention layers grow with context; sliding-
        window layers cap at their window and linear-attention layers keep
        fixed-size state, so neither scales per token. Introspects a native cache
        — the authoritative shapes (handles GQA, gemma4's mixed head dims, and
        hybrid models without re-deriving the layer-type logic)."""
        cache = self.acquire(1, self.native_len)
        per_token = 0
        for layer in cache.layers:
            # Only full-attention layers grow with context. Sliding-window
            # (AlloyStaticSlidingWindowLayer) caps at its window and linear-
            # attention (AlloyLinearAttentionLayer) keeps fixed state — neither
            # subclasses AlloyStaticLayer, so this selects exactly the growing
            # layers, whose `keys` is (batch, n_kv_heads, max_cache_len, head_dim).
            # `keys` is None for a dynamically-allocated cache (no eager buffers);
            # skipping it yields 0 → `derive_fill_budget` falls back to native.
            if isinstance(layer, AlloyStaticLayer) and layer.keys is not None:
                keys = layer.keys
                per_token += 2 * int(keys.shape[1]) * int(keys.shape[3]) * keys.element_size()
            elif isinstance(layer, AlloyStaticLayer) and layer.alloy_keys_q is not None:
                # Quantized cache: int8 codes + fp16 scales per token, K and V.
                codes, scales = layer.alloy_keys_q, layer.alloy_keys_scales
                per_token += 2 * int(codes.shape[1]) * (
                    int(codes.shape[3]) * codes.element_size()
                    + int(scales.shape[3]) * scales.element_size()
                )
        # Speculative drafter state grows with the sequence too (DFlash ctx-KV
        # rows); the drafter reports its own per-token footprint.
        if self.spec is not None:
            per_token += int(self.spec.drafter.state_bytes_per_token())
        return per_token

    def weight_bytes(self) -> int:
        """Resident weight memory: deduped sum of parameter + buffer storage.
        Quantized GGUF weights are packed byte buffers (numel·element_size is
        exact); dense tensors count at their real dtype."""
        seen: set[int] = set()
        total = 0
        for t in (*self.model.parameters(), *self.model.buffers()):
            st = t.untyped_storage()
            ptr = st.data_ptr()
            if ptr in seen:
                continue
            seen.add(ptr)
            total += st.nbytes()
        return total

    def tensor_alloc(self) -> "Callable[[int], int] | None":
        """Raw allocator for cache tensors; None = per-tensor buf_alloc.
        PagedKV returns its pool-slice allocator."""
        return None

    def supports_slices(self) -> bool:
        """True when the store can hold several KV slices and rebind the
        cache between them (PagedKV). False = single-slot semantics."""
        return False

    def wire_slices(self, slices: "list[KVSlice]") -> None:
        """Dispatch-wire the slices' VA resident off the request path. No-op
        for contiguous (one buffer, wired on first use and reused)."""
        return None

    def has_position_bound_state(self, cache: StaticCache) -> bool:
        """True for hybrids (DeltaNet recurrent/conv state): a mid-sequence
        truncation resume is only valid through a bookmark at or before the
        cut — the recurrent state exists only at a saved END. Resuming past
        a divergence without one produces malformed output."""
        return any(isinstance(layer, AlloyLinearAttentionLayer) for layer in cache.layers)

    def cache_for(self, batch_size: int, max_len: int) -> StaticCache:
        """The persistent cache object WITHOUT the acquire-time state reset.
        Slice binding owns state initialization in the multi-slice flow."""
        cache = self.persistent_caches.get((batch_size, max_len))
        return cache if cache is not None else self.acquire(batch_size, max_len)

    def reclaim_beyond(self, cache: StaticCache, start: int) -> int:
        """Return committed KV pages at positions >= `start` to the kernel.
        Contiguous caches have no page-level reclaim — no-op."""
        return 0

    def pressure_level(self) -> int:
        """System memory-pressure level (0 normal / 1 warn / 2 critical);
        only meaningful when the store can reclaim."""
        return 0

    def clear_tail(self, cache: StaticCache, start: int) -> None:
        """Zero every attention layer's K/V at cache positions [start, end).

        Used by cross-layer KV-sharing models (gemma4) before decode so a
        shared layer that doesn't rewrite the bucket tail can't attend the
        previous request's stale K/V there. Positions [0, start) (the live
        prompt, including any warm-prefill prefix) are untouched.
        """
        for layer in cache.layers:
            keys = layer.keys
            values = layer.values
            if keys is not None and start < keys.shape[2]:
                keys[:, :, start:].zero_()
            if values is not None and start < values.shape[2]:
                values[:, :, start:].zero_()

    def acquire(self, batch_size: int, max_len: int) -> StaticCache:
        # Reuse the same StaticCache instance (= same tensor storage_ptrs)
        # across conversations. The alloy backend's `_cached_input_check`
        # in `_execute_plan` matches by storage_ptr, so allocating a fresh
        # cache per conversation would invalidate the warm-prefill plan
        # and force a noticeable per-conversation rebuild hiccup on every
        # new conversation. Reusing the cache keeps storage_ptrs stable.
        # Safety: we reset cumulative_length to 0 so the model writes
        # fresh K/V at positions [0..N) for the new conversation; any
        # stale K/V left at higher positions is masked out by the
        # model's causal attention as long as decode only reads
        # cache[0..cur_pos+1).
        key = (batch_size, max_len)
        cache = self.persistent_caches.get(key)
        if cache is None:
            cache = AlloyStaticCache(
                config=self.model.config,
                max_cache_len=max_len,
                max_batch_size=batch_size,
                cache_dtype=self.cache_dtype,
                kv_format=self.kv_format,
                alloc=self.tensor_alloc(),
            )
            self.persistent_caches[key] = cache
        else:
            # Reset cumulative_length only. Stale K/V at positions beyond
            # the new prompt's length is harmless: the model's causal
            # attention only reads cache[0..cur_pos+1) so anything past
            # cumulative_length never reaches the softmax.
            #
            # Linear-attention (GatedDeltaNet) layers ARE order-dependent:
            # conv_states + recurrent_states carry the entire history,
            # not a position window. Reset them along with the layer's
            # has_previous_state flag so a fresh conversation starts
            # from zero state instead of inheriting warmup or prior-
            # request residue.
            for layer in cache.layers:
                layer.cumulative_length.fill_(0)
                if isinstance(layer, AlloyLinearAttentionLayer):
                    layer.conv_states.zero_()
                    if layer.recurrent_states is not None:  # None for LFM2 conv layers
                        layer.recurrent_states.zero_()
                    layer.has_previous_state = False
        return cache


class PagedPool:
    """One mach_vm reservation wrapped in a single MTLBuffer, carved into
    page-aligned slices (`pool_slice` registers each as a first-class alloy
    buffer, so every pointer/handle path resolves it with the right Metal
    offset). Pages commit on first touch; `pool_reclaim` returns them.
    Batch-1 today is a bump allocator over the slice table; the free list
    arrives with multi-batch."""

    PAGE = 16384

    def __init__(self, nbytes: int) -> None:
        self.handle = _metal_ext.pool_create(nbytes)
        self.nbytes = int(nbytes)
        self.top = 0
        self.slices: list[tuple[int, int, int]] = []  # (handle, start, nbytes)

    def alloc(self, nbytes: int) -> int:
        """Carve the next page-aligned slice -> alloy buffer handle."""
        start = self.top
        handle = _metal_ext.pool_slice(self.handle, start, nbytes)
        self.top = -(-(start + nbytes) // self.PAGE) * self.PAGE
        self.slices.append((handle, start, nbytes))
        return handle


# Concurrent resumable conversations: each holds a full KV slice in the pool.
# Committed pages scale with what each conversation actually filled; the LRU
# entry is evicted (pages reclaimed, storages reused) when a new conversation
# needs a slot. Sized for agentic traffic: interleaved sessions plus the side
# calls clients fire around every real turn.
MAX_SLICES = 8

# Per-conversation chunk-boundary resume points (see PrefixCache.mark_prefix);
# halved when full so coverage stays even. Slim capture: spec slot 0 only,
# ~1/SPEC_SLOT_BANK of a full turn-end bookmark.
PREFIX_MARK_CAP = 8


class KVSlice:
    """One conversation's physical KV backing: a storage per cache tensor
    (canonical walk order) plus the python state the graph cache guards on.
    The cache OBJECT never changes — binding a slice repoints every cache
    tensor's storage with `set_()`; the pinned plans re-resolve bindings via
    the existing storage-change check on the next dispatch."""

    def __init__(self, storages: list[torch.UntypedStorage]) -> None:
        self.storages = storages
        # Per-layer has_previous_state for linear-attention layers (None for
        # others) — Dynamo specializes decode on this bool, so a slice must
        # carry it (both specializations are compiled by eager_compile_all).
        self.has_prev: list[bool | None] = []


class PagedKV(ContiguousKV):
    """ContiguousKV with every cache tensor carved from one vm-reserved pool.

    Same cache objects, same kernels, same plans — slices are virtual-
    contiguous, so nothing above the allocator changes. What the pool adds:
    page-level reclaim (`reclaim_beyond`: dead conversations hand their
    KV pages back to the kernel) and the memory-pressure signal. Opt-in via
    ALLOY_KV=paged.
    """

    # Tensors safe to reclaim: KV rows past the live position are dead by the
    # causal-mask contract. Deliberately NOT the DeltaNet conv/recurrent state
    # or alloy_attn_mask — reclaimed pages can come back STALE, not zeroed
    # (E1), and those tensors' semantics assume their contents.
    KV_ROW_ATTRS = (
        "keys", "values",
        "alloy_keys_q", "alloy_keys_scales", "alloy_values_q", "alloy_values_scales",
    )

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.pool: PagedPool | None = None

    def tensor_alloc(self) -> Callable[[int], int]:
        if self.pool is None:
            # Reserve virtual space for the maximum the machine could ever
            # hold (the fill budget bounds actual commitment). The
            # reservation is pure VA — no Metal buffer spans it (each slice
            # wraps its own range, so Metal's wire-whole-buffer-at-first-use
            # behaviour stays per-tensor) and pages commit on first write.
            try:
                # scoped: GPU/Metal runtime dep — keep generation importable without a device
                from alloy._runtime.metal import default_device
                working_set = int(default_device().recommended_max_working_set_size)
            except Exception:  # noqa: BLE001 -- no device signal → modest fixed reserve
                working_set = 16 << 30
            self.pool = PagedPool(working_set)
        return self.pool.alloc

    def pressure_level(self) -> int:
        return int(_metal_ext.memory_pressure_level())

    def supports_slices(self) -> bool:
        return True

    def bookmark_budget_bytes(self) -> int:
        """Turn-end bookmarks plus the multi-slice prefix marks: worst case
        MAX_SLICES conversations × PREFIX_MARK_CAP slim marks, each ~1/8 of
        the full position-bound state (spec slot 0 only)."""
        base = super().bookmark_budget_bytes()
        if self.bookmark_slots <= 0:
            return base
        per_full = base // self.bookmark_slots
        return base + MAX_SLICES * PREFIX_MARK_CAP * per_full // 8

    def slice_entries(self, cache: StaticCache) -> list[tuple[str, torch.Tensor]]:
        """Every cache tensor as (attr_name, tensor) in a canonical, deduped
        order — THE walk every slice operation indexes by. Aliased tensors
        (the shared cumulative_length / alloy_last_real) appear once — one
        set_() repoints every layer's view because they are one object."""
        seen: set[int] = set()
        out: list[tuple[str, torch.Tensor]] = []
        for layer in cache.layers:
            for name, value in sorted(vars(layer).items()):
                if isinstance(value, torch.Tensor) and id(value) not in seen:
                    seen.add(id(value))
                    out.append((name, value))
        return out

    def slice_tensors(self, cache: StaticCache) -> list[torch.Tensor]:
        return [t for _name, t in self.slice_entries(cache)]

    def current_slice(self, cache: StaticCache) -> KVSlice:
        """Adopt the cache's CURRENT storages as a slice (slice 0 — the
        storages construction allocated from the pool)."""
        kvslice = KVSlice([t.untyped_storage() for t in self.slice_tensors(cache)])
        self.snapshot_slice(cache, kvslice)
        return kvslice

    def create_slice(self, cache: StaticCache) -> KVSlice:
        """Allocate a fresh pool-backed storage set matching the cache's
        tensors. Contents are uninitialized — `init_slice` after binding."""
        storages: list[torch.UntypedStorage] = []
        for t in self.slice_tensors(cache):
            nbytes = t.untyped_storage().nbytes()
            handle = self.tensor_alloc()(nbytes)
            raw = (ctypes.c_uint8 * nbytes).from_address(_metal_ext.buf_ptr(handle))
            storage = torch.frombuffer(raw, dtype=torch.uint8).untyped_storage()
            storage._alloy_keepalive = (raw, handle)  # type: ignore[attr-defined]
            storages.append(storage)
        return KVSlice(storages)

    def wire_slices(self, slices: list[KVSlice]) -> None:
        """Pre-wire every slice buffer's VA resident in ONE dispatch, off the
        request path. A bytesNoCopy slice is native-context-VA, which Metal
        wires on first encoder use (~14ms/GB) — paid per fresh conversation
        otherwise. A dispatch-touch wires it now WITHOUT touching
        phys_footprint (E1), so the demand-paged memory win is preserved."""
        handles = [
            storage._alloy_keepalive[1]  # type: ignore[attr-defined]
            for kvslice in slices
            for storage in kvslice.storages
        ]
        if handles:
            _metal_ext.wire_buffers(handles)

    def snapshot_slice(self, cache: StaticCache, kvslice: KVSlice) -> None:
        """Refresh a slice's records from the live cache before switching
        away: storage objects (the backend's input conversion may have
        repointed plain-torch tensors like HF's cumulative_length into alloy
        memory) and the per-layer has_previous_state bools."""
        kvslice.storages = [t.untyped_storage() for t in self.slice_tensors(cache)]
        kvslice.has_prev = [
            layer.has_previous_state if isinstance(layer, AlloyLinearAttentionLayer) else None
            for layer in cache.layers
        ]

    def bind_slice(self, cache: StaticCache, kvslice: KVSlice) -> None:
        """Repoint every cache tensor to the slice's storages and restore the
        slice's python state. Pinned plans self-heal: the next dispatch's
        storage-change check rebuilds (handle, offset) bindings."""
        tensors = self.slice_tensors(cache)
        if len(tensors) != len(kvslice.storages):
            raise RuntimeError(
                f"slice shape drift: cache has {len(tensors)} tensors, "
                f"slice has {len(kvslice.storages)} storages"
            )
        moved = False
        with torch.no_grad():
            for t, storage in zip(tensors, kvslice.storages):
                if t.untyped_storage() is not storage:
                    t.set_(storage, 0, t.shape, t.stride())
                    moved = True
        for layer, flag in zip(cache.layers, kvslice.has_prev or []):
            if flag is not None and isinstance(layer, AlloyLinearAttentionLayer):
                layer.has_previous_state = flag
        if moved:
            self.slice_epoch += 1

    def init_slice(self, cache: StaticCache) -> None:
        """Cold-start init of the BOUND slice — replicates construction-time
        state for a fresh or LRU-reused storage set. A new pool slice reads
        zero, a reused one reads stale; neither matches the constructor's
        invariants (alloy_attn_mask all-ones, alloy_last_real -1), and
        DeltaNet state must be explicitly zeroed either way."""
        for layer in cache.layers:
            layer.cumulative_length.fill_(0)
            if isinstance(layer, AlloyLinearAttentionLayer):
                layer.conv_states.zero_()
                if layer.recurrent_states is not None:  # None for LFM2 conv layers
                    layer.recurrent_states.zero_()
                layer.alloy_attn_mask.fill_(1)
                layer.has_previous_state = False
        cache.alloy_last_real.fill_(-1)

    def fork_rows(self, cache: StaticCache, src: KVSlice, rows: int) -> None:
        """Copy KV rows [0, rows) from `src` into the BOUND slice (the eager
        page-copy fork: a divergent request resumes a shared prefix without
        hijacking the source conversation). Full-attention tensors copy the
        row range; window-bounded (sliding) tensors copy whole. DeltaNet
        state is NOT copied — it is only valid at a sequence end, so the
        caller restores a bookmark or stays cold."""
        entries = self.slice_entries(cache)
        if len(entries) != len(src.storages):
            raise RuntimeError("fork across mismatched slice shapes")
        with torch.no_grad():
            for (name, dst), src_storage in zip(entries, src.storages):
                if name not in self.KV_ROW_ATTRS:
                    continue
                view = torch.empty(0, dtype=dst.dtype)
                view.set_(src_storage, 0, dst.shape, dst.stride())
                if dst.ndim == 4 and dst.shape[2] == self.native_len:
                    dst[:, :, :rows].copy_(view[:, :, :rows])
                else:
                    dst.copy_(view)  # window-bounded ring — whole tensor

    def reclaim_slice(self, kvslice: KVSlice) -> int:
        """Return an (unbound) slice's committed pages to the kernel."""
        _metal_ext.gpu_sync()
        total = 0
        for storage in kvslice.storages:
            keepalive = vars(storage).get("_alloy_keepalive")
            if keepalive is None:
                continue
            total += _metal_ext.pool_reclaim(keepalive[1], 0, storage.nbytes())
        return total

    def reclaim_beyond(self, cache: StaticCache, start: int) -> int:
        """MADV_FREE_REUSABLE every KV row at positions >= `start`. Returns
        bytes reclaimed. Syncs the GPU first — reclaim must never race an
        in-flight command buffer (deferred decode waits)."""
        if self.pool is None:
            return 0
        t0 = time.perf_counter()
        _metal_ext.gpu_sync()
        total = 0
        ranges = 0
        for layer in cache.layers:
            attrs = vars(layer)
            for name in self.KV_ROW_ATTRS:
                t = attrs.get(name)
                if not isinstance(t, torch.Tensor) or t.ndim != 4:
                    continue
                keepalive = vars(t.untyped_storage()).get("_alloy_keepalive")
                if keepalive is None:
                    continue
                handle = keepalive[1]
                batch, heads, seq = t.shape[0], t.shape[1], t.shape[2]
                if start >= seq:
                    continue  # sliding-window layers cap below a deep start
                item = t.element_size()
                base = t.data_ptr() - _metal_ext.buf_ptr(handle)
                # (B, H, S, D) is contiguous: rows [start, seq) of each (b, h)
                # plane are one byte range. Range count is the layout
                # instrumentation feeding the seq-major decision.
                for b in range(batch):
                    for h in range(heads):
                        plane = base + (b * t.stride(0) + h * t.stride(1)) * item
                        total += _metal_ext.pool_reclaim(
                            handle,
                            plane + start * t.stride(2) * item,
                            (seq - start) * t.stride(2) * item,
                        )
                        ranges += 1
        logger.info(
            "kv_pages_reclaimed", start=start, ranges=ranges,
            mb=round(total / (1 << 20), 1),
            ms=round((time.perf_counter() - t0) * 1e3, 2),
        )
        return total
