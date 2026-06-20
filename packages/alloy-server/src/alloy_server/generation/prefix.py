"""PrefixCache: warm-prefix reuse, bookmarks, truncation heal, side-call
preservation.

Owns prefix lineage: the live (cache_len, tokens, cache) save slot, the
bookmark deque (hybrid recurrent-state snapshots), token-LCP matching, and the
heal that closes a truncated turn.
"""

from __future__ import annotations

import collections
import contextlib

import torch
from transformers.cache_utils import StaticCache

from alloy import get_logger
from alloy_server.cache import AlloyLinearAttentionLayer
from alloy_server.generation.decode import DecodeEngine
from alloy_server.generation.kv import MAX_SLICES, PREFIX_MARK_CAP, ContiguousKV, KVSlice
from alloy_server.generation.plans import PlanStore

logger = get_logger("alloy_server.generation")

# Below this many matching tokens the warm-splice bookkeeping isn't worth it.
MIN_PREFIX = 8

class Conversation:
    """One resumable conversation: its KV slice, token history, and bookmarks."""

    def __init__(self, kvslice: KVSlice, cache_len: int, bookmark_slots: int) -> None:
        self.kvslice = kvslice
        self.cache_len = cache_len
        self.tokens: list[int] = []
        self.bookmarks: collections.deque = collections.deque(maxlen=bookmark_slots)
        # Chunk-boundary resume points, ascending by length. Same entry shape
        # as bookmarks: (cache_len, tokens, layer-state clones).
        self.prefix_marks: list[tuple[int, list[int], list[dict]]] = []


def contains_subseq(seq: list[int], sub: tuple[int, ...]) -> bool:
    """Return True iff `sub` appears contiguously inside `seq`."""
    if not sub or len(sub) > len(seq):
        return False
    first = sub[0]
    n_sub = len(sub)
    for i in range(len(seq) - n_sub + 1):
        if seq[i] == first and tuple(seq[i:i + n_sub]) == sub:
            return True
    return False


class PrefixCache:
    """The conversation's resumable-prefix state across requests."""

    def __init__(
        self,
        kv: ContiguousKV,
        plans: PlanStore,
        decode: DecodeEngine,
        *,
        eos_token_ids: tuple[int, ...],
        close_think_seq: tuple[int, ...],
        mid_think_heal_seq: tuple[int, ...],
        post_think_heal_seq: tuple[int, ...],
        auto_injects: bool,
        bookmark_slots: int,
    ) -> None:
        self.kv = kv
        self.plans = plans
        self.decode = decode
        self.eos_token_ids = eos_token_ids
        # Heal tokens for max_new_tokens truncation. `close_think_seq` is used
        # purely for detection: if absent from the decoded stream, truncation
        # happened inside a `<think>` block (a sequence, not a single id —
        # Qwen3's GGUF tokenizer renders `</think>` as 3 ordinary tokens).
        # `mid_think_heal_seq` closes the block + writes a brief notice +
        # turn-end; `post_think_heal_seq` is just the turn-end token.
        self.close_think_seq = close_think_seq
        self.mid_think_heal_seq = mid_think_heal_seq
        self.post_think_heal_seq = post_think_heal_seq
        # True when the chat template inserts assistant-turn markers the model
        # itself must emit during generation (e.g. Qwen3's auto
        # `<think>\n\n</think>\n\n`); decode tokens then don't round-trip
        # through the next turn's re-render, so the LCP stops at the prompt
        # boundary.
        self.auto_injects = auto_injects
        # Warm-prefill state: (cache_len, full_token_ids, cache). On the next
        # request, the longest byte-for-byte prefix match against
        # `full_token_ids` is reused from the cache and only the suffix is
        # prefilled. Single slot.
        self.state: tuple[int, list[int], StaticCache] | None = None
        # Multi-slice conversation table (paged mode): LRU-ordered resumable
        # conversations, each with its own KV slice. `bound` is the entry the
        # cache object's storages currently point at; `state`/`bookmarks`
        # mirror it so the single-slot logic below operates on whichever
        # conversation is bound.
        self.conversations: list[Conversation] = []
        self.bound: Conversation | None = None
        # Pre-warmed spare slices (paged): allocated + dispatch-wired at startup
        # by prewarm_slices. open_conversation draws from here before allocating,
        # so a fresh conversation reuses an already-wired slice and skips Metal's
        # per-slice first-encoder-use VA wiring on its first prefill.
        self.free_slices: list[KVSlice] = []
        self.bookmark_slots = bookmark_slots
        # Prefix BOOKMARKS — resume points beyond the single live slot. The
        # live cache physically holds a trie of every prefix whose rows
        # haven't been overwritten (branches share their head and only write
        # past it); what a branch destroys is the hybrid's position-bound
        # DeltaNet recurrent/conv state, which exists only at a sequence END.
        # A bookmark is that state plus the token list as the rows-valid
        # witness (valid iff its tokens prefix the live state's tokens).
        self.bookmarks: collections.deque = collections.deque(maxlen=bookmark_slots)

    def save(self, cache_len: int, tokens: list[int], cache: StaticCache) -> None:
        """Set the live resume slot to (cache_len, tokens, cache)."""
        self.state = (cache_len, tokens, cache)
        if self.bound is not None:
            self.bound.cache_len = cache_len
            self.bound.tokens = list(tokens)

    def extend(self, decoded: list[int]) -> None:
        """Extend the live slot's tokens with newly decoded ids so the next
        turn's LCP can reach through the assistant boundary."""
        if self.state is None:
            return
        cache_len, prior_tokens, cache = self.state
        self.state = (cache_len, prior_tokens + list(decoded), cache)
        if self.bound is not None:
            self.bound.tokens = list(self.state[1])

    def reset(self) -> None:
        """Clear the saved warm-prefill prefix and the pinned decode plans.
        Called whenever an incoming request isn't a clean continuation of the
        previous one, to keep the token-level LCP from matching unrelated
        prefixes by coincidence AND to keep spec-decode plan replay from
        reusing a decode plan whose pinned dispatch shape no longer matches
        the new request (the pin is shape-specific).

        Multi-slice mode keeps the conversation table: "not a continuation of
        the LAST conversation" is the case the table exists for — the next
        request matches whichever conversation it actually extends."""
        self.state = None
        self.plans.clear_decode_state()

    def match(
        self,
        input_ids: torch.Tensor,
        cache_len: int,
        batch_size: int,
        prompt_len: int,
    ) -> tuple[StaticCache, int]:
        """Return (cache, prefix_len). prefix_len > 0 means warm-prefill: the
        returned cache is the previous request's cache, with cumulative_length
        truncated to the longest byte-for-byte prefix match between
        `input_ids` and the saved token sequence. prefix_len == 0 means
        cold path.

        We compute the actual LCP (not strict-extension) so that chats
        sharing a long history but diverging at the end (e.g. user edited
        the last message, or the chat template re-tokenized the assistant
        response into different special-token forms) still get partial
        reuse for the matching prefix.
        """
        if self.kv.supports_slices():
            return self.match_paged(input_ids, cache_len, batch_size, prompt_len)
        state = self.state
        if state is None or batch_size != 1 or state[0] != cache_len:
            return self.kv.acquire(batch_size, cache_len), 0
        prev_tokens = state[1]
        prev_cache = state[2]
        prev_len = len(prev_tokens)

        # Cold-but-cache-reuse path. Reset cumulative_length on the same-shape
        # cache we already own and hand it back to prefill, avoiding the
        # buf_alloc + memmove pass over the full K/V storage on every
        # cache-miss request. Prefill overwrites [0..bucket) before decode
        # reads it, so stale K/V doesn't leak.
        def reset_and_reuse() -> tuple[StaticCache, int]:
            # Full cold reset, mirroring acquire(): linear-attention state
            # carries the ENTIRE history — not resetting it leaks the previous
            # conversation's DeltaNet residue into the new prefill.
            for layer in prev_cache.layers:
                layer.cumulative_length.fill_(0)
                if isinstance(layer, AlloyLinearAttentionLayer):
                    layer.conv_states.zero_()
                    if layer.recurrent_states is not None:  # None for LFM2 conv layers
                        layer.recurrent_states.zero_()
                    layer.has_previous_state = False
            return prev_cache, 0

        if prev_len < MIN_PREFIX:
            return reset_and_reuse()
        new_row = input_ids[0].tolist()
        # Longest exact-match prefix between saved tokens and the new input;
        # reuse the saved cache up to there and prefill only the suffix. Capped
        # at prompt_len - 1 so the suffix prefill always has at least one token
        # to produce a next-token logit. An LCP short of MIN_PREFIX falls back
        # to cold (but still reuses the cache storage).
        max_check = min(prev_len, prompt_len - 1)
        lcp = 0
        while lcp < max_check and int(new_row[lcp]) == prev_tokens[lcp]:
            lcp += 1
        if lcp < prev_len:
            # The input DIVERGES from the live sequence before its end — a
            # mid-sequence resume has no valid linear-attention state (it
            # only exists at a saved END). Look for a bookmark at or before
            # the divergence: its tokens must still prefix the live sequence
            # (rows intact) and prefix the input (resume-correct; implied by
            # len <= lcp). Longest valid bookmark wins.
            for bookmark in sorted(
                self.bookmarks, key=lambda b: len(b[1]), reverse=True,
            ):
                b_cache_len, b_tokens, _ = bookmark
                b_len = len(b_tokens)
                if b_cache_len != cache_len or not (MIN_PREFIX <= b_len <= lcp):
                    continue
                if b_tokens == prev_tokens[:b_len]:
                    return prev_cache, self.restore_bookmark(prev_cache, bookmark)
            # No bookmark covers the divergence. On hybrids the LCP
            # truncation below would resume with the saved END's DeltaNet
            # state — malformed output (a request sharing a long prefix but
            # diverging gets the donor's end-of-reply state). Cold instead.
            if self.kv.has_position_bound_state(prev_cache):
                return reset_and_reuse()
        if lcp < MIN_PREFIX:
            return reset_and_reuse()
        for layer in prev_cache.layers:
            layer.cumulative_length.fill_(lcp)
        return prev_cache, lcp

    def match_paged(
        self,
        input_ids: torch.Tensor,
        cache_len: int,
        batch_size: int,
        prompt_len: int,
    ) -> tuple[StaticCache, int]:
        """Multi-slice match: pick the conversation with the longest token
        LCP. A clean extension binds that conversation and resumes warm; a
        DIVERGENCE forks (`fork_conversation`) so the source stays intact;
        a fresh conversation opens only when NO conversation shares
        MIN_PREFIX. Eviction is LRU + memory-pressure only."""
        if batch_size != 1:
            return self.kv.acquire(batch_size, cache_len), 0
        cache = self.kv.cache_for(batch_size, cache_len)
        if self.bound is None:
            # Adopt the construction-time storages as the first conversation.
            conv = Conversation(self.kv.current_slice(cache), cache_len, self.bookmark_slots)
            self.conversations.append(conv)
            self.bound = conv
            self.bookmarks = conv.bookmarks
        if self.kv.pressure_level() > 0:
            self.evict_cold(cache)

        new_row = input_ids[0].tolist()
        max_check_cap = prompt_len - 1
        best: Conversation | None = None
        best_lcp = 0
        for conv in self.conversations:
            if conv.cache_len != cache_len or len(conv.tokens) < MIN_PREFIX:
                continue
            cap = min(len(conv.tokens), max_check_cap)
            lcp = 0
            while lcp < cap and int(new_row[lcp]) == conv.tokens[lcp]:
                lcp += 1
            if lcp > best_lcp:
                best_lcp, best = lcp, conv
        logger.info(
            "kv_match", n_convs=len(self.conversations), prompt=prompt_len,
            best_lcp=best_lcp,
            best_tokens=len(best.tokens) if best else 0,
            best_marks=len(best.prefix_marks) if best else 0,
            best_bookmarks=len(best.bookmarks) if best else 0,
        )
        if best is None or best_lcp < MIN_PREFIX:
            return self.open_conversation(cache, cache_len), 0
        if best_lcp < len(best.tokens):
            return self.fork_conversation(best, best_lcp, cache, cache_len)
        self.bind_conversation(best, cache)
        for layer in cache.layers:
            layer.cumulative_length.fill_(best_lcp)
        return cache, best_lcp

    def fork_conversation(
        self,
        src: Conversation,
        lcp: int,
        cache: StaticCache,
        cache_len: int,
    ) -> tuple[StaticCache, int]:
        """A request that DIVERGES from `src` before its end must never resume
        in place — truncation hijacks src (the post-generation save() rewrites
        its lineage and the prefill overwrites its tail rows), destroying
        interleaved sessions' warmth. Fork instead: copy the shared rows into a
        fresh conversation and resume there; src stays intact and resumable.

        Resume point: hybrids carry position-bound DeltaNet state that only
        exists at a saved end — a bookmark at or before the divergence sets
        it; without one the fork is cold (resuming past the divergence with
        someone else's end-state emits malformed turns). Pure-attention
        models resume at the LCP directly."""
        resume = lcp
        restore: tuple[int, list[int], list[dict]] | None = None
        if self.kv.has_position_bound_state(cache):
            # Longest resume point at/before the divergence: turn-end
            # bookmarks AND chunk-boundary prefix marks.
            for bookmark in sorted(
                [*src.bookmarks, *src.prefix_marks],
                key=lambda b: len(b[1]), reverse=True,
            ):
                b_cache_len, b_tokens, _ = bookmark
                if b_cache_len != cache_len or not (MIN_PREFIX <= len(b_tokens) <= lcp):
                    continue
                if b_tokens == src.tokens[:len(b_tokens)]:
                    restore = bookmark
                    resume = len(b_tokens)
                    break
            if restore is None:
                return self.open_conversation(cache, cache_len), 0
        self.open_conversation(cache, cache_len)
        self.kv.fork_rows(cache, src.kvslice, resume)
        if restore is not None:
            resume = self.restore_bookmark(cache, restore)
        else:
            for layer in cache.layers:
                layer.cumulative_length.fill_(resume)
        assert self.bound is not None
        self.bound.tokens = list(src.tokens[:resume])
        # Inherit every source resume point at/before the fork's resume row —
        # the fork's rows are copies of exactly those rows, so the snapshots
        # (immutable; shared by reference) are valid here too. Without this,
        # forked conversations are mark-less and the NEXT divergence from one
        # (every Claude Code turn — it mutates an earlier message's reminder
        # block, so turns are never clean extensions) goes fully cold.
        inherited = sorted(
            (m for m in [*src.prefix_marks, *src.bookmarks]
             if m[0] == cache_len and len(m[1]) <= resume),
            key=lambda m: len(m[1]),
        )
        while len(inherited) > PREFIX_MARK_CAP:
            inherited = inherited[1::2]
        self.bound.prefix_marks = inherited
        self.state = (cache_len, self.bound.tokens, cache)
        logger.info(
            "kv_slice_fork", rows=resume, src_tokens=len(src.tokens),
            inherited_marks=len(inherited),
        )
        return cache, resume

    def bind_conversation(self, conv: Conversation, cache: StaticCache) -> None:
        """Bind `conv`'s slice (snapshotting the outgoing one) and point the
        single-slot mirrors (`state`, `bookmarks`) at it."""
        if conv is not self.bound:
            if self.bound is not None:
                self.kv.snapshot_slice(cache, self.bound.kvslice)
            self.kv.bind_slice(cache, conv.kvslice)
            self.bound = conv
            logger.info("kv_slice_switch", tokens=len(conv.tokens))
        self.conversations.remove(conv)
        self.conversations.append(conv)  # LRU bump
        self.state = (conv.cache_len, conv.tokens, cache)
        self.bookmarks = conv.bookmarks

    def prewarm_slices(self, cache: StaticCache) -> None:
        """Allocate the full spare-slice set at startup and dispatch-wire their
        VA resident in one pass, off the request path. Metal wires a buffer's
        full VA on first encoder use (~14ms/GB), and a paged slice is
        native-context-VA — so a fresh conversation's first prefill otherwise
        eats that stall (~100ms on qwen3.5:4b). Dispatch-wiring (NOT
        requestResidency) keeps it OFF phys_footprint, preserving the
        demand-paged memory win. The construction storages are conversation 0;
        create MAX_SLICES-1 spares."""
        if not self.kv.supports_slices():
            return
        for _ in range(MAX_SLICES - 1 - len(self.free_slices)):
            self.free_slices.append(self.kv.create_slice(cache))
        self.kv.wire_slices(self.free_slices)

    def open_conversation(self, cache: StaticCache, cache_len: int) -> StaticCache:
        """Open a new conversation: a pre-warmed spare slice, a fresh slice, or
        the LRU entry's storages (pages reclaimed) once the table is full.
        Cold-initializes the bound slice — fresh pool pages read zero but a
        reused slice reads stale, and either way the constructor invariants
        (attn-mask ones, last_real -1) must be re-established."""
        if self.bound is not None:
            self.kv.snapshot_slice(cache, self.bound.kvslice)
        if self.free_slices:
            kvslice = self.free_slices.pop()
        elif len(self.conversations) >= MAX_SLICES:
            victim = next(c for c in self.conversations if c is not self.bound)
            self.conversations.remove(victim)
            self.kv.reclaim_slice(victim.kvslice)
            kvslice = victim.kvslice
            logger.info("kv_slice_evicted", tokens=len(victim.tokens))
        else:
            kvslice = self.kv.create_slice(cache)
        conv = Conversation(kvslice, cache_len, self.bookmark_slots)
        self.conversations.append(conv)
        self.kv.bind_slice(cache, conv.kvslice)
        self.kv.init_slice(cache)
        self.bound = conv
        self.state = None
        self.bookmarks = conv.bookmarks
        return cache

    def evict_cold(self, cache: StaticCache) -> None:
        """Memory pressure: reclaim every non-bound conversation's pages and
        drop the entries (their KV rows may not survive the kernel's
        reclaim, so they are no longer resumable)."""
        cold = [c for c in self.conversations if c is not self.bound]
        for conv in cold:
            self.kv.reclaim_slice(conv.kvslice)
            self.conversations.remove(conv)
        if cold:
            logger.warning("kv_pressure_evict", n=len(cold))

    def invalidate_live(self) -> None:
        """The bound cache state is no longer trustworthy (an unmanaged
        generate path ran). Drop the bound conversation in paged mode; clear
        the single slot either way."""
        if self.bound is not None:
            self.conversations.remove(self.bound)
            self.bound = None
        self.state = None

    def bookmark(self) -> None:
        """Bookmark the live slot as a resume point: clone every cache-layer
        tensor WITHOUT a cache_len-sized dim (DeltaNet conv/recurrent state,
        cumulative_length — the position-bound state; K/V rows stay in the
        live cache and are validated by token-prefix comparison at restore
        time). 406 MB / ~5ms on qwen3.5:4b; bytes on pure-attention models."""
        state = self.state
        if state is None:
            return
        cache_len, tokens, cache = state
        newest = self.bookmarks[-1] if self.bookmarks else None
        if newest is not None and newest[1] == tokens:
            return  # identical resume point already bookmarked
        layers: list[dict[str, torch.Tensor]] = []
        for layer in cache.layers:
            entry: dict[str, torch.Tensor] = {}
            for name, value in vars(layer).items():
                if isinstance(value, torch.Tensor) and cache_len not in value.shape:
                    entry[name] = value.clone()
            layers.append(entry)
        self.bookmarks.append((cache_len, list(tokens), layers))

    def restore_bookmark(
        self, cache: StaticCache, bookmark: tuple[int, list[int], list[dict]],
    ) -> int:
        """Copy a bookmark's position-bound state back into the live cache and
        return its resume position. Caller has already validated the rows.
        Attribute names come from the same `vars(layer)` walk that saved them.
        Slim entries (prefix marks: spec slot 0 only) narrow-copy into the
        leading slots; slots past the copy are stale, which is sound — spec
        verify writes its slots before reading them. The restored state is
        primed, so the linear layers' has_previous_state flips True (a fork
        binds a fresh slice where it starts False)."""
        _, tokens, layers = bookmark
        for layer, entry in zip(cache.layers, layers):
            live = vars(layer)
            for name, saved in entry.items():
                dst = live[name]
                if saved.shape != dst.shape:
                    dst.narrow(0, 0, saved.shape[0]).copy_(saved)
                else:
                    dst.copy_(saved)
            if entry and isinstance(layer, AlloyLinearAttentionLayer):
                layer.has_previous_state = True
        for layer in cache.layers:
            layer.cumulative_length.fill_(len(tokens))
        return len(tokens)

    def mark_prefix(self, pos: int, tokens: list[int], cache: StaticCache) -> None:
        """Chunk-boundary resume point: slim-clone the position-bound state
        (spec slot 0 only — ~1/8 of a turn-end bookmark) for positions
        [0, pos) into the bound conversation's prefix marks. These are what
        let a fork resume INSIDE a long shared prefix (a new session or
        sub-agent sharing a 30k+ system block) instead of going cold: the
        residual re-prefill is bounded by one mark spacing. Capped with
        halving so coverage stays even."""
        conv = self.bound
        if conv is None or not self.kv.has_position_bound_state(cache):
            return
        if pos < MIN_PREFIX or pos > len(tokens):
            return
        layers: list[dict[str, torch.Tensor]] = []
        for layer in cache.layers:
            entry: dict[str, torch.Tensor] = {}
            for name, value in vars(layer).items():
                if not isinstance(value, torch.Tensor) or conv.cache_len in value.shape:
                    continue
                # Spec slot bank: only slot 0 carries the resume state.
                entry[name] = value[:1].clone() if name == "recurrent_states" else value.clone()
            layers.append(entry)
        conv.prefix_marks.append((conv.cache_len, list(tokens[:pos]), layers))
        if len(conv.prefix_marks) > PREFIX_MARK_CAP:
            conv.prefix_marks = conv.prefix_marks[1::2]  # halve, keep coverage even

    @contextlib.contextmanager
    def preserving(self, side_total: int):
        """Run a foreign side request without losing the conversation's warm
        prefix (Claude Code interleaves topic-detection / bash-prefix calls
        with the main turns; without preservation each one evicts the prefix
        and costs the next turn a full cold re-prefill).

        There is exactly ONE live cache the pinned prefill/decode plans are
        bound to — a side request writes ITS prefill+decode into that cache's
        head no matter what cache object is passed around (the pins hold
        frozen buffer bindings). So instead of a second cache: snapshot the
        head rows the side request will overwrite (`side_total` = its prompt +
        decode budget), null the live slot for the duration so the side
        request runs the plain cold path (a partial token-LCP match against
        the saved prefix would warm-resume mid-sequence, which is unsound for
        hybrids — DeltaNet's recurrent state only exists at the saved end),
        and restore bytes + state after.

        Snapshot cost: `side_total` rows of K/V per attention layer plus the
        small recurrent/conv state tensors — tens of MB / ~ms for the 1-3k
        token side calls this exists for. The policy gate (caller) only
        preserves when the side request is much smaller than the saved
        prefix, so the copy is always the cheap direction.
        """
        if self.kv.supports_slices():
            # Multi-slice mode needs no snapshot: the side request opens (or
            # LCP-matches) its own conversation slice via `match_paged`, and
            # the main conversation's slice is rebound on its next turn.
            yield
            return
        state = self.state
        if state is None:
            yield
            return
        cache_len, _, cache = state
        rows = min(side_total, cache_len)
        snapshot: list[dict[str, tuple[int | None, torch.Tensor]]] = []
        for layer in cache.layers:
            entry: dict[str, tuple[int | None, torch.Tensor]] = {}
            for name, value in vars(layer).items():
                if not isinstance(value, torch.Tensor):
                    continue
                # Tensors with a cache_len-sized dim (attention K/V) need only
                # the head rows; everything else (DeltaNet conv/recurrent
                # state, cumulative_length) is small — clone whole.
                seq_dims = [i for i, d in enumerate(value.shape) if d == cache_len]
                if seq_dims:
                    entry[name] = (seq_dims[0], value.narrow(seq_dims[0], 0, rows).clone())
                else:
                    entry[name] = (None, value.clone())
            snapshot.append(entry)
        self.state = None
        try:
            yield
        finally:
            for layer, entry in zip(cache.layers, snapshot):
                live = vars(layer)
                for name, (dim, saved) in entry.items():
                    if dim is None:
                        live[name].copy_(saved)
                    else:
                        live[name].narrow(dim, 0, saved.shape[dim]).copy_(saved)
            self.state = state

    def heal_truncated(
        self,
        tokens: list[torch.Tensor],
        cache: StaticCache,
        cache_len: int,
        prompt_len: int,
        device: torch.device,
    ) -> None:
        """Amortise the next-turn cold-prefill cost into a few extra decode
        steps at this turn's end.

        When max_new_tokens fires before the model emits turn-end, the cache
        is mid-emission. The current state has no `</think>` close, no
        `<|im_end|>` — splicing those wrappers into the next turn's input_ids
        produces out-of-distribution attention and degenerate output.

        Fix: detect mid-`<think>` truncation by counting open/close think
        tokens in the decoded portion, then append `</think>` (if needed)
        and turn-end and run them through the decode module so the model
        writes K/V at those positions. The 1-2 extra dispatches cost ~5-10ms
        but unlock fast warm-prefill on every subsequent turn.
        """
        last_id = int(tokens[-1][0, -1].item())
        if last_id in self.eos_token_ids:
            return
        # Total emitted positions so far: count tokens in `tokens` after the
        # input_ids tensor (tokens[0]). Each subsequent tensor is shape (1,k)
        # — usually k=1, occasionally larger when a prefill returns the
        # bootstrap token.
        decoded: list[int] = []
        for tensor in tokens[1:]:
            decoded.extend(int(v) for v in tensor.flatten().tolist())
        if self.close_think_seq and not contains_subseq(decoded, self.close_think_seq):
            heal_ids = list(self.mid_think_heal_seq)
            heal_kind = "mid_think"
        else:
            heal_ids = list(self.post_think_heal_seq)
            heal_kind = "post_think"
        if not heal_ids:
            return
        logger.warning("truncation_healed", kind=heal_kind, n_heal_tokens=len(heal_ids))
        # cumulative_length after the decode loop is `prompt_len + decoded_len`
        # (HF auto-incremented on each forward). Read it from the cache to be
        # robust against early-exit-via-EOS paths.
        cur_pos = int(cache.layers[0].cumulative_length.item())
        # Use the pinned tensors so the decode plan's storage_ptr-keyed input
        # check stays valid across heal + the next turn's decode loop. Each
        # heal step appends a fresh tensor to `tokens`; the caller reads the
        # appended ids into seq.healed.
        for tid in heal_ids:
            self.plans.token_input[0, 0] = tid
            self.plans.cache_position[0] = cur_pos
            self.decode.next_token(
                self.plans.token_input, cache, self.plans.cache_position,
            )
            tokens.append(
                torch.tensor([[tid]], dtype=torch.long, device=device)
            )
            cur_pos += 1
