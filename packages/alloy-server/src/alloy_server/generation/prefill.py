"""PrefillEngine: chunked prefill (fixed chunk + grid-shrunk excess).

Owns the compiled prefill module, the chunk loop, and the per-call grid
override path (recipe discovery, static safety gate, opt-in empirical
validator). The PlanStore supplies pinned plans/inputs; the KVStore supplies
caches for the discovery/validation probes.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import torch
import transformers
from transformers.cache_utils import StaticCache, StaticSlidingWindowLayer

from alloy import get_logger
from alloy._dispatch.buf_utils import is_record_only, set_record_only
from alloy_torch.backend import (
    _execute_plan,
    _grow_plan_pool,
    capture_plan,
    release_plan_intermediates,
)
from alloy_server.cache import AlloyLinearAttentionLayer
from alloy_torch.compile_window import compile_window, grid_shrink_compile
from alloy_server.generation.kv import ContiguousKV
from alloy_server.generation.plans import PlanStore
from alloy_server.models.attention import (
    set_deltanet_attn_mask,
    set_taps_enabled,
    set_use_alloy_warm_op,
    tap_values,
    tap_values_clear,
)
from alloy_server.speculative.contract import TapBatch

if TYPE_CHECKING:
    from alloy_torch.backend import CompiledPlan
    from alloy_server.speculative.session import SpecSession

logger = get_logger("alloy_server.generation")

# Final-segment split size: the last prefix mark lands this many tokens
# before the prompt end (see `chunked`). 64-multiple keeps the shrink grid
# tile-aligned.
TAIL_MARK = 256


def grid_shrink_safe(d, m_max: int) -> bool:
    """Is it safe to shrink this dispatch's axis-0 threadgroup prefix to cover
    only the first m_pad M-rows of its work?

    Safe by construction when grid[0] == m_max (one threadgroup per M row —
    rms_norm, the 2D-rework strided copies). Otherwise the grid is either
    M-block-tiled or 1D-flattened, and the threadgroup prefix covers the first
    ceil(m_pad/m_max) FRACTION of the kernel's linear work — which equals "the
    first m_pad M-rows" only if M is the OUTERMOST axis of every buffer it
    writes. Counter-example: the rope-table
    broadcast `mul` writes freqs = inv_freq ⊗ positions as (1, n_freqs, M) — M
    INNERMOST — so its shrunk prefix wrote "freq row 0, first chunk of
    positions" instead of "all freqs for the first m_pad positions", leaving
    the real rows' tables for freqs >= 1 holding whatever the uninitialized
    pool contained.

    The gate checks the recording-time view layout of every written buffer:
    an M-bearing axis (extent % m_max == 0) must be the max-stride axis. A
    buffer with no M-bearing axis doesn't constrain the shrink (per-feature /
    scalar writes). Missing dims metadata pins the dispatch (full grid —
    always correct, never corrupt)."""
    if d.grid[0] == m_max:
        return True
    if not d.write_dims:
        return False
    for dims in d.write_dims:
        axes = [(e, s) for e, s in dims if e > 1]
        if not axes:
            continue
        m_bearing = [(e, s) for e, s in axes if e % m_max == 0]
        if not m_bearing:
            continue
        max_stride = max(s for _, s in axes)
        if not any(s == max_stride for _, s in m_bearing):
            return False
    return True


def compute_grid_shrink_recipe(
    max_disp: list,
    probe_disp: list,
    m_max: int,
    m_probe: int,
) -> dict[int, list[tuple[int, int]]]:
    """Grid-shrink recipe: which dispatch grid axes scale
    with the prompt length M, read off the COMPILED m_max plan.

    The chunk plan is compiled once at sequence length `m_max`; per request we
    shrink it to an exact threadgroup count for the real prompt. Grid **axis 0**
    is the M-scaling dimension for every prefill kernel — GEMM M-tiles, attention
    Q-blocks, the combine's M rows, rope/norm rows, and the row count of every
    elementwise op; axes 1/2 are N-tiles / heads / splits / column blocks, fixed
    in M. So we only shrink axis 0.

    Axis 0's extent is LINEAR in M: ext(M) = ext_max * M / m_max, so per request
    the shrunk extent is ceil(M_pad * ext_max / m_max). We record `ext_max` (axis
    0's max-length extent) and reconstruct from it + m_max with integer arithmetic
    (exact even for a fractional per-position factor). The forms a prefill grid
    takes, and how each is DETECTED as M-linear:
      - block-tiled: ext = ceil(M / BLOCK_M)         (GEMMs, attention, 2D norms)
        → ext_max divides m_max. Detected by `m_max % ext_max == 0`. Use the
        m_max plan ALONE (not the probe extent): the probe is an UNTUNED shape, so
        a GEMM's BLOCK_M there can differ by CONFIG, breaking proportionality.
      - 1D-flattened: ext = ceil(M * cols / BLOCK)   (generic elementwise: the
        DeltaNet mask muls, SiLU sigmoid_mul, etc., gridded as (rows*col_blocks,))
        → ext_max = M * (cols/BLOCK). When cols/BLOCK is integer, ext_max is a
        multiple of m_max (`ext_max % m_max == 0`). When it's FRACTIONAL (e.g.
        qwen2.5:0.5b intermediate 4864 / BLOCK 1024 = 4.75), neither divisibility
        holds — but elementwise BLOCK is FIXED (not tuned), so the probe IS exactly
        proportional: `ext_max * m_probe == ext_probe * m_max`. That's the third
        gate. (It can't replace divisibility — for tuned-BLOCK GEMMs the probe is
        NOT proportional — but as a union branch it only adds the fractional case.)

    All gates require `ext_max != ext_probe` first (axis 0 actually changes with M,
    excluding M-independent axis-0 dispatches like the M=1 lm-head gather).

      - AFFINE: ext = base + k*M with base > 0   (grids whose tile count carries a
        constant offset — the MoE grouped GEMMs grid max_tiles = NUM_EXPERTS + M*TOP_K/
        PAD_M, i.e. 256 + M, and the sort/sanitize passes derived from it). Neither
        divisibility nor probe-proportionality holds (the +base breaks both), so at
        native these ran the FULL M_MAX grid every call — and their full-buffer
        writes faulted in ~the whole intermediate pool (~117GB on qwen3.6:35b at
        262144) regardless of prompt length. Detected by an exact two-point affine
        fit: k = (ext_max-ext_probe)/(m_max-m_probe) as an exact rational, base =
        ext_max - k*m_max > 0. Tuned-BLOCK GEMMs cannot mis-match here: their
        block-tiled extents divide m_max, so the divisibility gate claims them
        first, probe-independent (the §10.2 pitfall stays fixed).

    Dispatches are matched by (debug_name, occurrence) so a dispatch-count
    mismatch (split-K SPLITS shifting per shape) only drops the unmatched ones.
    Returns {flat_dispatch_idx: [(axis, base, lin), ...]} where the per-request
    shrunk extent is `base + ceil(m_pad * lin / m_max)` — base=0, lin=ext_max
    reproduces the proportional forms exactly.
    """
    probe_by_name: dict[str, list] = {}
    for d in probe_disp:
        probe_by_name.setdefault(d.debug_name, []).append(d)
    name_seen: dict[str, int] = {}
    recipe: dict[int, list[tuple[int, int, int]]] = {}
    for i, dmax in enumerate(max_disp):
        nm = dmax.debug_name
        occ = name_seen.get(nm, 0)
        name_seen[nm] = occ + 1
        cands = probe_by_name.get(nm)
        if cands is None or occ >= len(cands):
            continue
        ext_max = dmax.grid[0]
        ext_probe = cands[occ].grid[0]
        if ext_max <= 1 or ext_max == ext_probe:
            continue
        if not grid_shrink_safe(dmax, m_max):
            logger.info(
                "grid_shrink_static_pin",
                dispatch_idx=i,
                kernel=nm,
                grid=list(dmax.grid),
            )
            continue
        # axis 0 scales with M (extent changes — excludes the M=1 lm-head gather
        # and any M-independent axis-0) AND is M-linear: ext_max divides m_max
        # (block-tiled) OR is a multiple of m_max (integer-factor flattened) OR is
        # exactly proportional to the probe (fractional-factor flattened, whose
        # fixed elementwise BLOCK keeps the probe proportional).
        if (
            m_max % ext_max == 0
            or ext_max % m_max == 0
            or ext_max * m_probe == ext_probe * m_max
        ):
            recipe[i] = [(0, 0, ext_max)]
            continue
        # AFFINE union branch: ext = base + k*M, exact rational two-point fit.
        dm = m_max - m_probe
        dext = ext_max - ext_probe
        if dext > 0 and (dext * m_max) % dm == 0:
            lin = (dext * m_max) // dm  # k*m_max, exact
            base = ext_max - lin
            if base > 0:
                recipe[i] = [(0, base, lin)]
    return recipe


def grid_shrink_updates(
    plan: "CompiledPlan", m_pad: int
) -> list[tuple[int, int, int, int]] | None:
    """Per-dispatch grid overrides for a grid-shrunk prefill of `m_pad` rows, built
    from the plan's discovered recipe. M-dependent axes shrink to
    ceil(m_pad * ext_max / M_MAX) (linear in M — covers block-tiled and
    1D-flattened grids alike); every other axis keeps its registered (max-length)
    extent. Returns None when the plan has no recipe — the caller then dispatches
    the full max-length grid (still correct, just not shrunk)."""
    recipe = plan._grid_shrink_recipe
    if not recipe:
        return None
    m_max = plan._grid_shrink_m_max
    updates: list[tuple[int, int, int, int]] = []
    for flat_idx, axes in recipe.items():
        g = list(plan.dispatches[flat_idx].grid)
        for axis, base, lin in axes:
            # base + ceil(m_pad*lin/m_max): proportional forms have base=0; affine
            # grids (MoE max_tiles = NUM_EXPERTS + M) carry their constant offset.
            # ceil can only round UP toward the registered max extent — extra tiles
            # mask out exactly like the full grid's padding did.
            g[axis] = base + (m_pad * lin + m_max - 1) // m_max
        updates.append((flat_idx, g[0], g[1], g[2]))
    return updates


class ChunkPrefill(torch.nn.Module):
    """One prefill chunk on a fixed-size (padded) input. Returns the sampled
    token for the single real last position via
    `logits_to_keep=tensor([last_real_pos])`.

    The point of the fixed size: input_ids shape is ONE production chunk size
    (`chunk_prefill_size`, 4096 on the server), so Dynamo specializes the FX
    graph once and per-shape kernel tuning is done once — every prompt runs
    full chunks plus one partial chunk whose excess is GRID-SHRUNK away
    (per-call threadgroup override; pad rows cost no GPU work on the pinned
    plan).

    Why right-padding works without explicit attention masks: causal attention
    at position i only reads cache[0..i+1). Real tokens (positions [0,real])
    only see other real tokens. Pad tokens (positions [real,chunk)) compute
    bogus logits but we throw them away — we only ask for the logit at
    position real-1 via logits_to_keep. Subsequent decode starts at position
    real and only attends to cache[0..real+1), never reading the bogus pad
    K/V at positions [real,chunk).
    """

    model: transformers.PreTrainedModel

    def __init__(
        self,
        model: transformers.PreTrainedModel,
        tap_layers: tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        self.model = model
        # Speculative-drafter taps: when a drafter is
        # attached before the prefill module builds, every chunk also emits
        # the tapped decoder layers' hidden states (hidden_states[i+1] is the
        # OUTPUT of layer i); unrequested layers stay dead lazy buffers.
        self.tap_layers = tuple(tap_layers)
        # Splits the FX-graph-cache signature: a tapped prefill graph is a
        # different graph from the tapless one (same source, same model).
        self._cache_variant = f"taps={self.tap_layers}" if tap_layers else None

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: StaticCache,
        cache_position: torch.Tensor,
        last_real_pos: torch.Tensor,
        attention_mask: torch.Tensor,
        seed: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        # attention_mask (1, chunk): 1 for real tokens, 0 for pads. Stashed
        # as a tensor attribute on the cache_params so DeltaNet layers can
        # read it via plain attribute access — a dict keyed by id(cache_params)
        # forces a dynamo graph break that loses the alloy backend's Tensor(c!)
        # mutation propagation on the linear_attention_update op.
        set_deltanet_attn_mask(past_key_values, attention_mask)
        output = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=cache_position,
            logits_to_keep=last_real_pos,
        )
        # Shared cumulative_length advance (see GreedyNextToken.forward).
        # PrefillEngine.chunked overrides this eagerly with fill_(start_pos +
        # real_len) post-forward, so the in-graph value is corrected for the
        # padded-chunk case — but tracing the .add_ keeps cumulative_length
        # marked as an input mutation and stable across the pinned-storage
        # path.
        past_key_values.layers[0].cumulative_length.add_(input_ids.shape[1])
        logits = output.logits
        if logits is None:
            raise ValueError("causal LM output did not include logits")
        # Sample (or argmax when greedy) the first token on-GPU, mirroring
        # `GreedyNextToken`. The kept logit is at the last real position
        # (logits_to_keep=last_real_pos), so logits[:, -1:, :] is its row.
        token = torch.ops.alloy.sample_categorical(
            logits[:, -1:, :], cache_position, seed, params
        )
        if self.tap_layers:
            return (token, *tap_values())
        return token


# The multimodal `inputs_embeds` prefill always chunks at this size, independent
# of the text chunk: the embeds path has no pinned plan and no grid shrink, so a
# large chunk would pay REAL pad-row GPU work on every short multimodal prompt.
EMBEDS_CHUNK_SIZE = 128


class EmbedTokens(torch.nn.Module):
    """Text embeddings (+ gemma4 per-layer-input embeddings) for multimodal prefill.

    Both embeddings are quantized (they run via alloy dispatch, not eager), so they
    are computed here; the vision features are spliced into `embeds` in Python and
    fed back as `inputs_embeds`. `input_ids` must already have image-placeholder
    slots replaced by the pad token — matching HF `Gemma4Model.forward`, which
    substitutes PAD so the lookup never OOVs and the per-layer-input carries the PAD
    identity at image slots (the image embedding itself is overwritten by the
    splice). The precomputed per-layer-input lets the decoder skip its
    inputs_embeds→input_ids reverse path (which can't run on a quantized embedding).
    """

    model: transformers.PreTrainedModel
    has_ple: bool

    def __init__(self, model: transformers.PreTrainedModel) -> None:
        super().__init__()
        self.model = model
        # gemma4 carries per-layer embeddings (PLE); plain-text multimodal archs
        # (qwen3.5 vision) don't define the field at all. A 0/missing field means
        # no PLE — the embeds→inputs_embeds prefill path skips it.
        cfg = model.model.config
        self.has_ple = (
            "hidden_size_per_layer_input" in vars(cfg)
            and bool(cfg.hidden_size_per_layer_input)
        )

    def forward(self, input_ids: torch.Tensor):
        embeds = self.model.get_input_embeddings()(input_ids)
        if self.has_ple:
            per_layer = self.model.model.get_per_layer_inputs(input_ids, embeds)
            return embeds, per_layer
        return embeds, None


class ChunkPrefillEmbeds(torch.nn.Module):
    """Bucketed prefill from `inputs_embeds` (vision features already spliced in)
    plus the precomputed `per_layer_inputs`, sampling the first token on-GPU. The
    multimodal counterpart of `BucketedPrefill`: identical padding / warm-op /
    sampling semantics, but the input is embeddings (so the quantized embed_tokens
    isn't re-run and the vision splice survives) and `per_layer_inputs` is forwarded
    through `**kwargs` so the decoder skips its embeds→ids reverse path."""

    model: transformers.PreTrainedModel

    def __init__(self, model: transformers.PreTrainedModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        per_layer_inputs: torch.Tensor | None,
        past_key_values: StaticCache,
        cache_position: torch.Tensor,
        last_real_pos: torch.Tensor,
        attention_mask: torch.Tensor,
        seed: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        set_deltanet_attn_mask(past_key_values, attention_mask)
        output = self.model(
            inputs_embeds=inputs_embeds,
            per_layer_inputs=per_layer_inputs,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=cache_position,
            logits_to_keep=last_real_pos,
        )
        past_key_values.layers[0].cumulative_length.add_(inputs_embeds.shape[1])
        logits = output.logits
        if logits is None:
            raise ValueError("causal LM output did not include logits")
        return torch.ops.alloy.sample_categorical(
            logits[:, -1:, :], cache_position, seed, params
        )


class PrefillEngine:
    """Chunked prefill with pinned-plan replay + grid-shrunk partial chunks."""

    def __init__(
        self,
        model: transformers.PreTrainedModel,
        plans: PlanStore,
        kv: ContiguousKV,
        *,
        chunk_prefill_size: int,
        grid_shrink: bool,
        prefill_chunks: tuple[int, ...],
        pad_token_id: int,
        cache_dtype: torch.dtype,
    ) -> None:
        self.model = model
        self.plans = plans
        self.kv = kv
        self.chunk_prefill_size = chunk_prefill_size
        self.grid_shrink = grid_shrink
        self.prefill_chunks = prefill_chunks
        self.pad_token_id = pad_token_id
        self.cache_dtype = cache_dtype
        # Attached spec-decode session (taps + observe). Set by attach_spec.
        self.spec: SpecSession | None = None
        # Compiled modules INJECTED by AlloyGenerator.build_modules
        # (eager_compile_all is the mandatory precondition). This engine is a
        # pure executor: it dispatches them, it never compiles. `module` is the
        # chunked-prefill wrapper (one cached plan per unique chunk, SEQ_LEN
        # constexpr); the two embed modules are the multimodal text-embed
        # lookup + chunked inputs_embeds prefill (built only for vision/audio
        # models).
        self.module: torch.nn.Module | None = None
        self.embed_module_compiled: torch.nn.Module | None = None
        self.embeds_module: torch.nn.Module | None = None
        # Whether warm prefill needs its own plan. Only sliding-window models
        # fork the prefill graph (cold builds the linear temp-KV that
        # `attention_prefill_cold` needs at chunk > window); everywhere else
        # Q_START_POS is a runtime buffer and one plan serves cold + warm.
        self.warm_split: bool | None = None

    def sliding_split(self, cache: StaticCache) -> bool:
        if self.warm_split is None:
            self.warm_split = any(
                isinstance(layer, StaticSlidingWindowLayer) for layer in cache.layers
            )
        return self.warm_split

    def plan_key(self, cache: StaticCache, chunk: int, is_warm: bool) -> tuple[int, bool]:
        return (chunk, is_warm and self.sliding_split(cache))

    def run(
        self,
        input_ids: torch.Tensor,
        cache: StaticCache,
        *,
        start_pos: int = 0,
        on_chunk: Callable[[int], None] | None = None,
    ) -> torch.Tensor:
        """Prefill the prompt suffix: THE one text-prefill entry point — every
        caller (run pipeline, warm-suffix, constrained, spec-decode) lands here,
        and it is nothing but `chunked` at the production chunk size. With a
        shrink-capable chunk plan, full chunks saturate the GPU and the final
        partial chunk runs at an exact shrunk grid, so no padded rows cost GPU
        work anywhere."""
        return self.chunked(input_ids, cache, start_pos=start_pos, on_chunk=on_chunk)

    def chunked(
        self,
        input_ids: torch.Tensor,
        cache: StaticCache,
        *,
        start_pos: int = 0,
        chunk: int | None = None,
        on_chunk: Callable[[int], None] | None = None,
    ) -> torch.Tensor:
        """Prefill `input_ids` by looping `chunk_step` over fixed-size chunks
        (default: the production chunk size; pass `chunk` to force another
        size, e.g. a 128-chunk reference in an A/B harness — the off-size
        chunk compiles lazily through the handler path). First chunk is cold
        (or warm if start_pos > 0); subsequent chunks are warm. Returns the
        argmax at the last real position of the FINAL chunk."""
        chunk = chunk or self.chunk_prefill_size
        real_len = int(input_ids.shape[1])
        assert real_len > 0
        last_token: torch.Tensor | None = None
        pos = 0
        while pos < real_len:
            end = min(pos + chunk, real_len)
            # Split the final segment so a chunk boundary (= a prefix-mark
            # resume point) lands TAIL_MARK tokens before the prompt end —
            # full-prefix forks (a request matching everything but the last
            # few mutated tokens) otherwise resume a whole partial chunk
            # back. Multi-chunk prefills only: the extra small forward costs
            # ~0.4s of fixed per-chunk overhead, a measured 7% pp hit on a
            # single-chunk prefill but <1% on the long prompts whose forks
            # the mark exists for. Applies in both KV modes so prefill stays
            # bit-identical across them.
            if end == real_len and pos > 0 and end - pos > 2 * TAIL_MARK:
                end -= TAIL_MARK
            slice_ids = input_ids[:, pos:end]
            sp = start_pos + pos
            last_token = self.chunk_step(slice_ids, cache, chunk, start_pos=sp)
            pos = end
            # Interior chunk boundary: the cache holds a consistent state for
            # positions [0, start_pos+pos) — a resume point a prefix bookmark
            # can capture. The final boundary is the turn end, covered by the
            # post-generation bookmark.
            if on_chunk is not None and pos < real_len:
                on_chunk(start_pos + pos)
        assert last_token is not None
        return last_token

    def chunk_step(
        self,
        input_ids: torch.Tensor,
        cache: StaticCache,
        chunk: int,
        *,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """Pad input_ids to `chunk` length and run prefill. Returns the next
        token (argmax of logits at the last real position).

        Pad K/V written to cache positions [start_pos + real_len, start_pos +
        chunk) are bogus but never read again — subsequent causal decode at
        position start_pos + real_len only attends to cache[0..start_pos +
        real_len + 1). We reset every layer's cumulative_length to
        `start_pos + real_len` so position_ids on the next forward starts
        from the correct value (the model auto-increments it by the input's
        seq dim — which would be `chunk`, not `real_len`).

        `start_pos > 0` is the warm-prefill case: cache positions [0, start_pos)
        are already populated from the previous request and the model attends
        to them via causal attention; we only prefill the suffix.
        """
        real_len = int(input_ids.shape[1])
        device = input_ids.device
        is_warm = start_pos > 0
        pinned_in = self.plans.pinned_inputs_for_chunk(chunk, device)
        pinned_input_ids, pinned_cache_position, pinned_last_real_pos, pinned_attn_mask = pinned_in
        with torch.no_grad():
            pinned_input_ids[:, :real_len].copy_(input_ids[:, :real_len])
            if real_len < chunk:
                pinned_input_ids[:, real_len:].fill_(self.pad_token_id)
            torch.arange(chunk, device=device, out=pinned_cache_position).add_(start_pos)
            pinned_last_real_pos[0] = real_len - 1
            pinned_attn_mask[:, :real_len].fill_(1)
            pinned_attn_mask[:, real_len:].fill_(0)
        compile_window.q_start_pos = start_pos
        set_use_alloy_warm_op(is_warm)
        taps_on = self.spec is not None and bool(self.spec.drafter.taps.layer_ids)
        set_taps_enabled(taps_on)
        if taps_on:
            tap_values_clear()  # host-side: in-graph clears break the pin (guards)
        # Sliding-ring write bound: only positions [end-window, end) of this
        # chunk write the ring — one writer per slot. Without it a chunk
        # longer than the window races (rows s, s+W, … contend for slot s;
        # measured >80% stale slots, non-deterministic, on gemma4). The bound
        # rides `cache.alloy_last_real` (a pinned plan operand — see
        # AlloyStaticCache); reset to unbounded in finally so decode and
        # spec verify write fail-open. Non-Alloy caches carry no buffer.
        try:
            cache_last_real = cache.alloy_last_real
        except AttributeError:
            cache_last_real = None
        if cache_last_real is not None:
            cache_last_real.fill_(real_len - 1)
        try:
            pinned_pair = self.plans.prefill_plans.get(
                self.plan_key(cache, chunk, is_warm)
            )
            if pinned_pair is not None:
                plan, args, next_token_idx, tap_idxs = pinned_pair
                # Grid shrink: dispatch exactly the real prompt length (padded
                # up to the tile multiple) against the chunk-compiled plan —
                # only the M-tiled dispatches' grids change per call, so the
                # padding rows beyond `m_pad` cost no GPU work. This is a PLAN
                # property: a plan compiled without a shrink recipe (small
                # chunks below the shrink threshold) yields grid_updates=None
                # and keeps its full registered grid.
                m_pad = (real_len + 63) // 64 * 64
                # Request-bounded pool: grow the M-outer intermediates to
                # cover this prompt (monotone high-water; doubles to
                # amortize, capped at the plan's M_MAX). Rare path — the
                # steady state pays nothing.
                if plan._pool_trunc and m_pad > plan._pool_bound:
                    _grow_plan_pool(plan, max(m_pad, 2 * plan._pool_bound))
                updates = grid_shrink_updates(plan, m_pad)
                # Record the shrunk grid so `alloy.visualize` can profile
                # this plan at the same launch the run just dispatched.
                plan._last_grid_shrink_updates = updates
                # `_execute_plan` returns the FX-flat output tuple (or a
                # single tensor when the graph has just one output). The
                # entries other than `next_token_idx` are AOT input-mutation
                # tracebacks (per-layer cumulative_length / keys / values)
                # that fold back into the input storages via aliasing — we
                # don't need to return them.
                result = _execute_plan(plan, args, grid_updates=updates)
                tap_tensors: tuple[torch.Tensor, ...] = ()
                if isinstance(result, tuple):
                    next_token = cast(torch.Tensor, result[next_token_idx])
                    if tap_idxs:
                        tap_tensors = tuple(result[i] for i in tap_idxs)
                else:
                    next_token = cast(torch.Tensor, result)
            else:
                module_out = self.module(
                    input_ids=pinned_input_ids,
                    past_key_values=cache,
                    cache_position=pinned_cache_position,
                    last_real_pos=pinned_last_real_pos,
                    attention_mask=pinned_attn_mask,
                    seed=self.plans.seed,
                    params=self.plans.params,
                )
                tap_tensors = ()
                if isinstance(module_out, tuple):
                    next_token = cast(torch.Tensor, module_out[0])
                    tap_tensors = tuple(module_out[1:])
                else:
                    next_token = cast(torch.Tensor, module_out)
        finally:
            compile_window.q_start_pos = 0
            set_use_alloy_warm_op(False)
            set_taps_enabled(False)
            if cache_last_real is not None:
                cache_last_real.fill_(-1)
        # The model's forward incremented every layer's cumulative_length by
        # `chunk`. Real-position decode wants it at `start_pos + real_len`.
        # has_previous_state is the SAME contract: the module path sets it
        # True inside the DN forward, but a pinned-plan replay has no python
        # side effects — without replicating it here, the request's FIRST
        # decode token runs the has_previous_state=False S=1 graph, whose
        # conv path computes a correct OUTPUT (it reads the saved pre-
        # context) but NEVER WRITES the token's input into conv_states. The
        # dropped column corrupts every window for the next conv_kernel_size
        # positions — a silent plain-decode wrongness near the prefill
        # boundary.
        for layer in cache.layers:
            layer.cumulative_length.fill_(start_pos + real_len)
            if isinstance(layer, AlloyLinearAttentionLayer):
                layer.has_previous_state = True
        # Drafter taps: hand this chunk's tapped hidden
        # states to the attached drafter — rows [0, real_len) are real, the
        # padded tail is dead (TapBatch.rows bounds validity; the tensors stay
        # chunk-wide for fixed shapes). record-only compiles skip (phantom
        # buffers, no data).
        if tap_tensors and self.spec is not None and not is_record_only():
            self.spec.drafter.observe(
                input_ids[0, :real_len].tolist(),
                TapBatch(start=start_pos, rows=real_len, layers=tap_tensors),
                start_pos,
            )
        return next_token

    def discover_grid_shrink_recipe(self, device: torch.device, max_cache: int) -> None:
        """Attach the chunk plan's M-dependent grid recipe (which dispatch
        grid axes scale with the prompt length, and their tile block) by tracing
        the SAME plan at a second, shorter length and diffing the resolved grids.
        Runs once in `eager_compile_all`; the recipe rides on the pinned plan."""
        if not self.grid_shrink:
            return
        m_max = self.chunk_prefill_size
        pinned = self.plans.prefill_plans.get((m_max, False))
        if pinned is None:
            return
        max_plan = pinned[0]
        # The second discovery point is a fresh trace (NOT a pinned chunk, whose
        # plan is a replay we can't re-resolve) at a multiple of 64 chosen to
        # satisfy two competing constraints:
        #   - FAR ENOUGH below m_max that every M-tiled grid axis changes tile
        #     count: a dispatch with row-block B has M-extent ceil(M/B), so a
        #     probe within B of m_max rounds to the SAME count (m_max-64 vs m_max
        #     at B=128 both give 32 → nothing detected).
        #   - CLOSE ENOUGH that the attention split-K structure (SPLITS resolved
        #     per shape) is identical, so the two plans have the same dispatch
        #     count and align by index (halving m_max changed SPLITS: 663 vs 767
        #     dispatches → unalignable).
        # A 256-row step changes ceil(M/B) for every block size up to 256 while
        # staying in m_max's structural regime. Fall back to ~half for an m_max
        # too small to step 256.
        m_probe = m_max - 256
        while m_probe >= 64 and m_probe in self.prefill_chunks:
            m_probe -= 64
        if m_probe < 64:
            m_probe = (m_max // 2 // 64) * 64
            while m_probe >= 64 and m_probe in self.prefill_chunks:
                m_probe -= 64
        if m_probe < 64:
            return  # no room for a second discovery point; keep the full grid
        probe_cache = self.kv.acquire(1, max_cache)
        # Trace the probe in grid-shrink mode so its attention is single-pass too —
        # the same structure as the pinned chunk plan, so the attention
        # dispatches align and land in the recipe (without this the probe would
        # take the handler's split-K branch and the attention would diverge).
        # The window caps the probe's M-scaled configs to the same
        # representative-M tune the pinned max plan resolved against, so the
        # two plans pick identical tile configs (→ identical grids) and the
        # recipe diff isn't polluted by a config mismatch. m_probe (not m_max)
        # — the probe is compiled at m_probe.
        # Record-only: the recipe only needs the probe plan's resolved grids
        # (metadata), so build it without allocating M=m_probe intermediates or
        # running the GPU — same memory bound as the M_MAX plan compile.
        set_record_only(True)
        try:
            with grid_shrink_compile(m_probe), torch.inference_mode():
                probe_in = torch.zeros((1, m_probe - 1), dtype=torch.long, device=device)
                with capture_plan() as slot:
                    self.chunk_step(probe_in, probe_cache, m_probe, start_pos=0)
        finally:
            set_record_only(False)
        probe_plan = slot.plan
        if probe_plan is None:
            return
        # The two plans may have different dispatch COUNTS (attention split-K
        # resolves per shape); `compute_grid_shrink_recipe` aligns by name +
        # occurrence, so a count mismatch only leaves the unalignable dispatches
        # (the differing attention split structure) on the full grid.
        recipe = compute_grid_shrink_recipe(
            max_plan.dispatches, probe_plan.dispatches, m_max, m_probe
        )
        max_plan._grid_shrink_recipe = recipe
        max_plan._grid_shrink_m_max = m_max
        # The probe was traced ONLY to diff grids (its .dispatches); its
        # M=m_probe intermediate pool (~the pinned plan's size) is now dead —
        # free it so it doesn't leak for the process (no Metal GC).
        release_plan_intermediates(probe_plan)
        # The static shrink-safety gate (write-layout provenance) is the
        # production mechanism — zero load-time cost. The empirical validator
        # (6 full-chunk prefills + a bisect per culprit, 2-27s/load measured)
        # stays as an opt-in bring-up harness for new models/kernels whose
        # layouts the static gate cannot see.
        n_pinned = 0
        if os.environ.get("ALLOY_GRID_SHRINK_VALIDATE", "") not in ("", "0"):
            n_pinned = self.validate_grid_shrink_recipe(max_plan, m_max, max_cache)
        warm = self.plans.prefill_plans.get((m_max, True))
        if warm is not None and len(warm[0].dispatches) == len(max_plan.dispatches):
            warm[0]._grid_shrink_recipe = max_plan._grid_shrink_recipe
            warm[0]._grid_shrink_m_max = m_max
        logger.info(
            "grid_shrink_recipe",
            m_max=m_max,
            m_probe=m_probe,
            n_dispatches=len(max_plan.dispatches),
            n_m_dependent=len(recipe),
            n_pinned=n_pinned,
        )

    def validate_grid_shrink_recipe(
        self, plan: "CompiledPlan", m_max: int, max_cache: int
    ) -> int:
        """Empirically validate the grid-shrink recipe on the compiled plan,
        auto-pinning any dispatch whose shrunk grid corrupts a short prompt.
        Returns the number of pinned dispatches. Runs once in eager-compile.

        Soundness gap the two-point grid diff cannot close: a 1D-flattened
        kernel whose output stores M INNERMOST has an M-linear grid extent —
        indistinguishable from a row-major flatten by grids alone — but its
        shrunk threadgroup prefix covers the wrong elements. Example: a
        rope-table broadcast `mul` (inv_freq ⊗ positions, logical (1, freqs, M))
        shrinks to "freq row 0, first chunk of positions" instead of "all freqs
        for the first m_pad positions", leaving real rows' tables for freqs >= 1
        reading uninitialized pool memory while the KV cache stays bit-exact.

        The check is content-based and kernel-agnostic: prime every
        intermediate by prefilling prompt X at the FULL chunk length (m_pad ==
        m_max -> all grids full), then prefill a shorter prompt Y twice — once
        shrunk (production grids), once full — and require identical next-token
        + KV rows. A dispatch whose shrink misses real work leaves X's
        wrong-content values in rows Y needs, so the comparison fails for
        content-dependent and content-independent corruption alike; a bisect
        over the recipe prefix then pins the culprit. Pinning keeps the recipe
        ENTRY but sets (base=ext_max, lin=0) — the dispatch gets an explicit
        full grid every call (grid overrides are sticky in C++, so entries must
        never just disappear). Worst case pins everything: full grids are
        always correct, just unshrunk. The X-priming run doubles as the pool
        warmup: every pad row a masked read can touch holds finite values
        afterwards, never uninitialized memory."""
        recipe = plan._grid_shrink_recipe
        if not recipe:
            return 0
        t0 = time.perf_counter()
        vocab = int(self.model.config.get_text_config().vocab_size)
        rng = torch.Generator().manual_seed(0x05EED)
        ids_x = torch.randint(0, vocab, (1, m_max), generator=rng, dtype=torch.long)
        len_y = max(64, (m_max // 2) - 27)
        ids_y = torch.randint(0, vocab, (1, len_y), generator=rng, dtype=torch.long)
        keys = sorted(recipe)
        original = {i: recipe[i] for i in keys}

        def pin_entry(i: int) -> list[tuple[int, int, int]]:
            return [(axis, plan.dispatches[i].grid[axis], 0) for axis, _b, _l in original[i]]

        def prefill_token(ids: torch.Tensor, pinned: set[int]) -> tuple[int, StaticCache]:
            plan._grid_shrink_recipe = {
                i: (pin_entry(i) if i in pinned else original[i]) for i in keys
            }
            cache = self.kv.acquire(1, max_cache)
            with torch.inference_mode():
                nt = self.chunk_step(ids, cache, m_max, start_pos=0)
            return int(nt[0, 0].item()), cache

        def snapshot(cache: StaticCache, rows: int) -> list[torch.Tensor]:
            out: list[torch.Tensor] = []
            for layer in cache.layers:
                if hasattr(layer, "keys") and layer.keys is not None:
                    out.append(layer.keys[:, :, :rows].clone())
                    out.append(layer.values[:, :, :rows].clone())
                if hasattr(layer, "conv_states") and layer.conv_states is not None:
                    out.append(layer.conv_states.clone())
                if hasattr(layer, "recurrent_states") and layer.recurrent_states is not None:
                    out.append(layer.recurrent_states.clone())
            return out

        def max_diff(a_list: list[torch.Tensor], b_list: list[torch.Tensor]) -> float:
            d = 0.0
            for a, b in zip(a_list, b_list):
                d = max(d, float((a.float() - b.float()).abs().max().item()))
            return d

        # Noise floor: plans with atomic scatter-accumulate (the fused MoE
        # prefill's reduce="add" stores) are not bit-stable run-to-run — two
        # IDENTICAL full-grid runs differ at f32 ULP and can flip a garbage-
        # prompt argmax. Measure that jitter on two full-grid runs and compare
        # shrunk-vs-full against it: corruption (wrong-CONTENT values from the
        # X priming left in rows Y needs) is orders of magnitude above ULP.
        prefill_token(ids_x, set())
        tok_f1, cache_f1 = prefill_token(ids_y, set(keys))
        snap_f1 = snapshot(cache_f1, len_y)
        tok_f2, cache_f2 = prefill_token(ids_y, set(keys))
        snap_f2 = snapshot(cache_f2, len_y)
        noise = max_diff(snap_f1, snap_f2)
        deterministic = noise == 0.0 and tok_f1 == tok_f2
        threshold = 0.0 if deterministic else max(noise * 64.0, 1e-4)
        if not deterministic:
            logger.info(
                "grid_shrink_recipe_noise_floor", noise=noise, threshold=threshold
            )

        def probe(pinned: set[int]) -> bool:
            """True when the shrunk-Y run matches the full-Y run (bit-exact for
            deterministic plans; within the measured noise floor otherwise)."""
            prefill_token(ids_x, set())          # prime pads at full coverage
            tok_s, cache_s = prefill_token(ids_y, pinned)
            snap_s = snapshot(cache_s, len_y)
            tok_f, cache_f = prefill_token(ids_y, set(keys))
            snap_f = snapshot(cache_f, len_y)
            if deterministic and tok_s != tok_f:
                return False
            return max_diff(snap_s, snap_f) <= threshold

        pinned: set[int] = set()
        try:
            for _round in range(8):
                if probe(pinned):
                    break
                # Bisect the smallest prefix of (unpinned) recipe keys whose
                # pinning fixes the mismatch; its last key is a culprit.
                free = [i for i in keys if i not in pinned]
                lo, hi = 0, len(free)  # pin free[:hi] fixes; free[:lo] doesn't
                if not probe(pinned | set(free)):
                    # Even all-full mismatches: non-grid corruption — bail to
                    # full grids and let the caller's log surface it.
                    pinned = set(keys)
                    logger.error("grid_shrink_recipe_unfixable", m_max=m_max)
                    break
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    if probe(pinned | set(free[:mid])):
                        hi = mid
                    else:
                        lo = mid
                culprit = free[hi - 1]
                pinned.add(culprit)
                logger.warning(
                    "grid_shrink_recipe_pinned",
                    dispatch_idx=culprit,
                    kernel=plan.dispatches[culprit].debug_name,
                    grid=list(plan.dispatches[culprit].grid),
                )
            else:
                pinned = set(keys)
                logger.error("grid_shrink_recipe_unfixable", m_max=m_max)
        finally:
            plan._grid_shrink_recipe = {
                i: (pin_entry(i) if i in pinned else original[i]) for i in keys
            }
            # Leave the cache reset for the first real request.
            self.kv.acquire(1, max_cache)
            logger.info(
                "grid_shrink_recipe_validated",
                took_ms=round((time.perf_counter() - t0) * 1000.0, 1),
                n_pinned=len(pinned),
            )
        return len(pinned)

    def chunked_embeds(
        self,
        embeds: torch.Tensor,
        per_layer_inputs: torch.Tensor | None,
        cache: StaticCache,
        *,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """Embeds counterpart of `chunked`: loop `chunk_step_embeds` over
        fixed-size chunks. Returns the sampled token at the last real position
        of the final chunk.

        Always chunks at `EMBEDS_CHUNK_SIZE` (NOT the text chunk): the embeds
        path has no pinned plan and no grid shrink, so a large chunk would pay
        REAL pad-row GPU work (up to chunk-1 rows) on every short multimodal
        prompt — the opposite trade of the text path. Port the shrink machinery
        before unifying."""
        chunk = EMBEDS_CHUNK_SIZE
        real_len = int(embeds.shape[1])
        assert real_len > 0
        last_token: torch.Tensor | None = None
        pos = 0
        while pos < real_len:
            end = min(pos + chunk, real_len)
            slice_pl = None if per_layer_inputs is None else per_layer_inputs[:, pos:end]
            last_token = self.chunk_step_embeds(
                embeds[:, pos:end], slice_pl, cache, chunk, start_pos=start_pos + pos,
            )
            pos = end
        assert last_token is not None
        return last_token

    def chunk_step_embeds(
        self,
        embeds: torch.Tensor,
        per_layer_inputs: torch.Tensor | None,
        cache: StaticCache,
        chunk: int,
        *,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """Embeds counterpart of `chunk_step`: pad `embeds` (and the
        per-layer-inputs) to `chunk`, run the multimodal prefill module, sample the
        first token. Same pad/warm/cumulative-length semantics as the token path;
        pad rows are zero embeddings whose bogus K/V is never read. Dispatches the
        injected `embeds_module` (built by AlloyGenerator.build_modules); not
        pinned (no eager_compile_all plan)."""
        real_len = int(embeds.shape[1])
        device = embeds.device
        is_warm = start_pos > 0
        # Reuse the token path's pinned aux buffers (cache_position / last_real_pos
        # / attention_mask) for matching dtypes; only the embeds + per-layer inputs
        # are multimodal-specific.
        _, _, _, attn_template = self.plans.pinned_inputs_for_chunk(chunk, device)
        padded = torch.zeros((1, chunk, embeds.shape[2]), dtype=embeds.dtype, device=device)
        padded[:, :real_len].copy_(embeds[:, :real_len])
        padded_pl: torch.Tensor | None = None
        if per_layer_inputs is not None:
            padded_pl = torch.zeros(
                (1, chunk, per_layer_inputs.shape[2], per_layer_inputs.shape[3]),
                dtype=per_layer_inputs.dtype, device=device,
            )
            padded_pl[:, :real_len].copy_(per_layer_inputs[:, :real_len])
        cache_position = torch.arange(chunk, dtype=torch.int32, device=device).add_(start_pos)
        last_real_pos = torch.tensor([real_len - 1], dtype=torch.long, device=device)
        attn_mask = torch.zeros_like(attn_template)
        attn_mask[:, :real_len].fill_(1)
        compile_window.q_start_pos = start_pos
        set_use_alloy_warm_op(is_warm)
        taps_on = self.spec is not None and bool(self.spec.drafter.taps.layer_ids)
        set_taps_enabled(taps_on)
        if taps_on:
            tap_values_clear()  # host-side: in-graph clears break the pin (guards)
        try:
            cache_last_real = cache.alloy_last_real
        except AttributeError:
            cache_last_real = None
        if cache_last_real is not None:
            cache_last_real.fill_(real_len - 1)
        try:
            next_token = cast(torch.Tensor, self.embeds_module(
                inputs_embeds=padded,
                per_layer_inputs=padded_pl,
                past_key_values=cache,
                cache_position=cache_position,
                last_real_pos=last_real_pos,
                attention_mask=attn_mask,
                seed=self.plans.seed,
                params=self.plans.params,
            ))
        finally:
            compile_window.q_start_pos = 0
            set_use_alloy_warm_op(False)
            if cache_last_real is not None:
                cache_last_real.fill_(-1)
        for layer in cache.layers:
            layer.cumulative_length.fill_(start_pos + real_len)
            # Replicate the module path's DN side effect on pinned replays —
            # see `chunked`'s epilogue.
            if isinstance(layer, AlloyLinearAttentionLayer):
                layer.has_previous_state = True
        return next_token
