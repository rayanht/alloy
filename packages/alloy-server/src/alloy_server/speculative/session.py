"""SpecSession — the method-independent half of speculative decoding.

Owns the round loop, verification, acceptance, target-state rollback, grammar
masking, and instrumentation. The drafter plugs in through the Drafter protocol
(contract.py).

Verification: position-keyed Gumbel exact-match. The verify plan emits
per-position LOGITS (+ the drafter's tap hidden states); the session samples
each row with `sample_categorical` at that row's absolute cache position —
the same counter-based RNG plain decode uses — greedy (temperature 0, argmax
with lowest-index tie-break) and sampled through one mechanism. The committed
stream matches non-spec decode at the same seed up to the MEASURED numerics
envelope: M>1 projections accumulate in
a different order than M=1, the hybrid recurrence amplifies that with
in-window row depth, so a deep-row commit can pick a different
still-high-probability token at a decision point. Inherent, not a bug; the
gate bounds divergence rate and plausibility instead of asserting
bit-identity. Grammar masking applies the per-row xgrammar bitmask between
logits and sampling, mirroring `_masked_sample`.

Target-state rollback:
- KV rows past the committed point are dead (cumulative_length is set once per
  round; the next verify overwrites them).
- DeltaNet recurrent state: the verify plan compiles with SAVE_STEPS so slot[j]
  holds the state after row j; rollback is an immediate slot[j]→slot[0] copy
  (shared memory, CPU-side — see the in-loop comment for why it must not be
  deferred onto the next verify's command buffer).
- DeltaNet conv window: reconstructable from (pre-verify snapshot + final
  window) only when the verify is no longer than the window itself
  (M <= conv_kernel_size). Beyond that a partial accept restores the snapshot
  and REPLAYS the committed tokens through the single-token decode plan —
  correct but slow.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Iterator

import torch

from alloy._dispatch.buf_utils import _alloc_aligned
from alloy._compiler.dtypes import float32, int32, int64
from alloy._runtime.alloy_buffer import materialize_many
from alloy._runtime.convert import to_alloy_buffer
from alloy.std import apply_token_bitmask, sample_categorical
from alloy.std.delta_net import gdr_state_reconstruct
from alloy_server.models.attention import set_deltanet_attn_mask, set_use_alloy_warm_op
from alloy_server.cache import AlloyLinearAttentionLayer
from alloy_torch.backend import _execute_plan
from alloy_torch.ops.delta_net import alias_dn_scratch, conv_tape_tensor, gdr_round_bufs

from .contract import Drafter, Proposal, RoundStats, SpecMetrics, TapBatch

if TYPE_CHECKING:
    from alloy_server.generation.generator import AlloyGenerator


class VerifyState:
    """One pinned verify plan (per M) + its IO bindings."""

    def __init__(self, plan, args, in_buf, pos_buf, logits_idx, tap_idxs, post_norm_idx):
        self.plan = plan
        self.args = args
        self.in_buf = in_buf
        self.pos_buf = pos_buf
        self.logits_idx = logits_idx
        self.tap_idxs = tap_idxs
        self.post_norm_idx = post_norm_idx
        wanted = [logits_idx, *tap_idxs]
        if post_norm_idx >= 0:
            wanted.append(post_norm_idx)
        self.wanted = frozenset(wanted)


class SpecSession:
    """Drives speculative decoding for one (generator, drafter) pair.

    Construction is cheap; `warmup()` (from eager_compile_all) pins the verify
    plan(s) and the drafter's plans. `run()` serves one request and yields
    committed token ids — the caller (server _Generation / bench) handles EOS
    and stop strings exactly as for plain decode."""

    def __init__(self, gen: "AlloyGenerator", drafter: Drafter) -> None:
        self.gen = gen
        self.drafter = drafter
        self.metrics = SpecMetrics()
        self.last_metrics: SpecMetrics | None = None
        self._verify_states: dict[int, VerifyState] = {}
        self._sample_bufs: dict[int, dict] = {}
        # Committed-count input for the per-round gdr_state_reconstruct
        # dispatch (tape-replay DN rollback). FLOAT32, not int32: the
        # kernel's scalar load binds float and would read int bits as a
        # denormal -> cast -> 0-trip loop (a silent no-op; found as spec
        # output cycling with period tau).
        self._recon_n = _alloc_aligned((1,), float32)
        # Verify width: anchor + up to max_draft_tokens proposal rows.
        self.verify_m = 1 + drafter.max_draft_tokens
        # Last KV slice epoch the verify plan + DeltaNet scratch were bound
        # to. A multi-slice switch (paged KV) repoints the cache storage out
        # from under the pinned verify plan; `_rebind_slice` re-establishes
        # both before the next verify. -1 = never bound.
        self._bound_epoch = -1
        # Per-DeltaNet-layer (conv_states, recurrent_states) storage pointers
        # the scratch was PINNED against at warmup, in layer order. A slice
        # switch aliases that scratch from these to the new slice's pointers.
        self._warmup_dn_ptrs: list[tuple[int, int]] = []

    # ------------------------------------------------------------------ setup

    def warmup(self) -> None:
        """Pin the M-row verify plan and warm the drafter. Called from
        eager_compile_all; idempotent."""
        if self.verify_m not in self._verify_states:
            from alloy_server.generation.spec import pin_verify_plan  # scoped: avoid import cycle (generation.spec imports session types)

            self._verify_states[self.verify_m] = pin_verify_plan(
                self.gen,
                self.verify_m,
                self.drafter.taps,
            )
            cache = self.gen.kv.cache_for(1, self.gen.kv.max_cache_len)
            self._warmup_dn_ptrs = [
                (layer.conv_states.untyped_storage().data_ptr(),
                 layer.recurrent_states.untyped_storage().data_ptr())
                for _, layer in self._dn_layers(cache)
            ]
        self.drafter.warmup()

    def _bufs_for(self, m: int) -> dict:
        bufs = self._sample_bufs.get(m)
        if bufs is None:
            vocab = int(self.gen.model.config.vocab_size)
            words = (vocab + 31) // 32
            bufs = {
                "vocab": vocab,
                "masked": _alloc_aligned((m, vocab), float32),
                "bitmask": _alloc_aligned((m, words), int32),
                "bitmask_cpu": torch.empty((m, words), dtype=torch.int32),
                "pos": _alloc_aligned((m,), int64),
                "pos_cpu": torch.empty((m,), dtype=torch.int64),
                "seed": _alloc_aligned((1,), int64),
                "params": _alloc_aligned((4,), float32),
                "choices": _alloc_aligned((m,), int64),
            }
            self._sample_bufs[m] = bufs
        return bufs

    # ------------------------------------------------------ verify + sampling

    def _verify(
        self,
        toks: list[int],
        start: int,
        vs: VerifyState,
        matcher,
        bufs: dict,
    ) -> tuple[list[int], TapBatch | None, bool]:
        """One verify forward over `toks` at [start, start+M): replay the
        pinned plan, then per-row position-keyed (masked) sampling. Returns
        (choices, taps, grammar_dead) where choices[j] is the target's chosen
        token after toks[:j+1] and grammar_dead marks rows the matcher
        rejected (guaranteed mismatch past the first)."""
        gen = self.gen
        m = len(toks)
        device = gen.plans.token_input.device
        self._rebind_slice(vs)
        set_use_alloy_warm_op(True)
        try:
            vs.in_buf.copy_(torch.tensor([toks], dtype=torch.long, device=device))
            vs.pos_buf.copy_(torch.arange(start, start + m, dtype=torch.long, device=device))
            res = _execute_plan(
                vs.plan,
                vs.args,
                wanted_outputs=vs.wanted,
                args_stable=True,
            )
            outs = res if isinstance(res, tuple) else (res,)
        finally:
            set_use_alloy_warm_op(False)

        logits = outs[vs.logits_idx]  # (1, M, V) torch tensor (alloy-backed)
        taps = None
        if vs.tap_idxs or vs.post_norm_idx >= 0:
            taps = TapBatch(
                start=start,
                rows=m,
                layers=tuple(outs[i] for i in vs.tap_idxs),
                post_norm=outs[vs.post_norm_idx] if vs.post_norm_idx >= 0 else None,
            )

        # Grammar: advance a speculative copy of the matcher across the
        # proposal rows, collecting a per-row bitmask. Row j's mask constrains
        # the choice AFTER toks[:j+1]; an invalid draft token kills rows > j.
        grammar_dead = False
        if matcher is not None:
            cpu_mask = bufs["bitmask_cpu"]
            advanced = 0
            for j in range(m):
                matcher.fill_next_token_bitmask(cpu_mask, index=j)
                if j + 1 < m:
                    if grammar_dead or not matcher.accept_token(toks[j + 1]):
                        grammar_dead = True
                        # Rows past an invalid draft can never commit; their
                        # mask content is irrelevant — reuse row j's.
                        cpu_mask[j + 1 :] = cpu_mask[j]
                        break
                    advanced += 1
            # Rewind the speculative advance; the caller re-advances through
            # the genuinely committed tokens only.
            if advanced:
                matcher.rollback(advanced)
            bufs["bitmask"].copy_from(cpu_mask.data_ptr())

        # Row-parallel position-keyed sampling (lazy alloy kernels — one
        # command buffer at the sync). Greedy is temperature==0 inside the
        # kernel; row j's RNG counter is its absolute position, so each row
        # samples exactly what plain decode would sample there (§3.4). Both
        # kernels are dispatch_spec-less — explicit one-program-per-row grids.
        vocab = bufs["vocab"]
        logits_buf = to_alloy_buffer(logits.reshape(m, vocab))
        if matcher is not None:
            apply_token_bitmask[(m,)](logits_buf, bufs["bitmask"], bufs["masked"])
            logits_buf = bufs["masked"]
        bufs["pos_cpu"].copy_(torch.arange(start, start + m, dtype=torch.int64))
        bufs["pos"].copy_from(bufs["pos_cpu"].data_ptr())
        choices = bufs["choices"]
        sample_categorical[(m,)](
            logits_buf,
            bufs["pos"],
            bufs["seed"],
            bufs["params"],
            choices,
        )
        choices.sync()
        return [int(v) for v in choices.numpy.reshape(-1)], taps, grammar_dead

    # ------------------------------------------------------- state rollback

    def _rebind_slice(self, vs: VerifyState) -> None:
        """Re-establish the verify plan + DeltaNet scratch against the
        CURRENTLY bound KV slice when a multi-slice switch has repointed the
        cache storage since the last verify. The verify plan replays with
        args_stable (so it never re-scans for moved storage on its own) and
        the conv-tape / round-buffer registries are data_ptr-keyed — both are
        pinned to whichever slice was bound at warmup, so without this a
        request on any other conversation's slice reads the wrong KV (or
        misses the scratch lookup and crashes). Runs once per switch, after
        cast — `_verify` calls it before every round but the epoch only moves
        on a bind."""
        epoch = self.gen.kv.slice_epoch
        if epoch == self._bound_epoch:
            return
        cache = self.gen.kv.cache_for(1, self.gen.kv.max_cache_len)
        # Alias the warmup scratch to the bound slice's storage so the
        # rollback's data_ptr-keyed lookups land on the buffers the pinned
        # plan actually tees into (the plan writes fixed addresses; the slice
        # only moves the cache K/V the plan reads/writes via its input slots).
        for (cp, rp), (_, layer) in zip(self._warmup_dn_ptrs, self._dn_layers(cache)):
            alias_dn_scratch(
                cp, layer.conv_states.untyped_storage().data_ptr(),
                rp, layer.recurrent_states.untyped_storage().data_ptr(),
            )
        # Force the next replay to rebuild (slot, handle, offset) from the
        # args' current storage instead of the stale pinned bindings.
        vs.plan._cached_input_updates = None
        vs.plan._cached_input_check = None
        self._bound_epoch = epoch

    def _dn_layers(self, cache) -> list[tuple[int, AlloyLinearAttentionLayer]]:
        return [
            (i, layer)
            for i, layer in enumerate(cache.layers)
            if isinstance(layer, AlloyLinearAttentionLayer)
        ]

    # ---------------------------------------------------------------- run

    def run(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        matcher=None,
    ) -> Iterator[int]:
        """Speculative generation for one request. Yields committed new tokens;
        the stream is bit-identical to the equivalent non-spec decode at the
        same pinned seed/params (greedy and sampled alike), and grammar-valid
        when a matcher is given."""
        gen = self.gen
        drafter = self.drafter
        if int(input_ids.shape[0]) != 1:
            raise ValueError("speculative decode requires batch size 1")
        if max_new_tokens <= 0:
            return
        device = input_ids.device
        prompt_len = int(input_ids.shape[1])
        m_max = self.verify_m
        max_new_tokens = gen.kv.fit_to_budget(
            prompt_len,
            max_new_tokens,
            extra=m_max - 1,
        )
        cache_len = gen.kv.cache_len_for(prompt_len + max_new_tokens + m_max)
        metrics = SpecMetrics()
        self.metrics = metrics

        stop_ids = set(gen.eos_token_ids)
        if matcher is not None:
            stop_ids.update(int(t) for t in matcher.stop_token_ids)

        with torch.inference_mode():
            # Warm up BEFORE touching the cache: pinning the verify plan
            # primes the ONE persistent cache with dummy tokens, so doing it
            # lazily after the request prefill would destroy the prefilled
            # state. Production warms in eager_compile_all; this covers
            # direct API use.
            vs = self._verify_states.get(m_max)
            if vs is None:
                self.warmup()
                vs = self._verify_states[m_max]

            # Warm prefix: the target reuses its LCP rows
            # as in stream_chunks_fast, and the drafter truncates to the SAME
            # boundary — its per-position state is position-aligned and
            # append-only, so prior turns' rows stay valid for the shared
            # prefix and the suffix prefill's taps fill the rest. On a cold
            # request prefix_len == 0 == full drafter reset.
            bufs = self._bufs_for(m_max)
            bufs["seed"].write_scalar(int(gen.plans.seed[0].item()))
            bufs["params"].copy_from(gen.plans.params.contiguous().data_ptr())

            cache, prefix_len = gen.prefix.match(
                input_ids,
                cache_len,
                1,
                prompt_len,
            )
            drafter.truncate(prefix_len)

            if matcher is None:
                # The prefill samples the first token through the SAME pinned
                # prefill plan plain decode uses.
                suffix = input_ids if prefix_len == 0 else input_ids[:, prefix_len:]
                first_t = gen.prefill.run(suffix, cache, start_pos=prefix_len)
                first = int(first_t[0, 0].item())
            else:
                # Constrained: the first token must be GRAMMAR-MASKED, so the
                # prefill stops one short and the last prompt token runs a
                # masked decode step (the run_constrained pattern — an
                # unmasked prefill sample would be matcher-rejected or
                # grammar-invalid).
                head = input_ids[:, : prompt_len - 1]
                if head.shape[1] > prefix_len:
                    gen.prefill.run(head[:, prefix_len:], cache, start_pos=prefix_len)
                first = self._plain_step(
                    cache,
                    int(input_ids[0, prompt_len - 1]),
                    prompt_len - 1,
                    matcher,
                    bufs,
                )
            gen.prefix.save(
                cache_len, [int(t) for t in input_ids[0].tolist()], cache,
            )
            drafter.observe(input_ids[0].tolist(), None, 0)

            committed: list[int] = [first]
            yield first
            if first in stop_ids:
                self.last_metrics = metrics
                return
            if matcher is not None:
                if not matcher.accept_token(first):
                    self.last_metrics = metrics
                    return
                if matcher.is_terminated():
                    self.last_metrics = metrics
                    return

            dn_layers = self._dn_layers(cache)
            kc = int(dn_layers[0][1].conv_states.shape[2]) if dn_layers else 0
            anchor = first
            sp = prompt_len  # absolute position of `anchor`

            # No boundary buffer/wash machinery: cold-boundary divergence is
            # handled at the prefill epilogues, which replicate the module
            # path's `has_previous_state = True` side effect on pinned-plan
            # replays (see generation), so plain decode and verify agree from a
            # cold boundary.
            while len(committed) < max_new_tokens:
                t0 = time.perf_counter()
                proposal: Proposal = drafter.propose(anchor, sp)
                n_prop = len(proposal.tokens)
                t_draft = time.perf_counter()

                if n_prop == 0:
                    # Miss path (PLD with no n-gram hit): one plain decode
                    # step through the generator's pinned decode machinery.
                    nt = self._plain_step(cache, anchor, sp, matcher, bufs)
                    metrics.add(
                        RoundStats(
                            proposed=0,
                            accepted=0,
                            bonus=True,
                            draft_us=(t_draft - t0) * 1e6,
                            host_us=(time.perf_counter() - t_draft) * 1e6,
                        )
                    )
                    committed.append(nt)
                    drafter.observe([anchor], None, sp)
                    yield nt
                    if nt in stop_ids or (matcher is not None and self._matcher_step(matcher, nt)):
                        break
                    anchor, sp = nt, sp + 1
                    continue

                # Pad short proposals to the pinned width with the anchor id —
                # rows are verified like any other; a lucky match is still the
                # target's own choice (lossless either way).
                pad = m_max - 1 - n_prop
                toks = [anchor, *proposal.tokens] + [anchor] * pad
                m = len(toks)

                # Pre-verify conv snapshot: source for splice columns that
                # predate the verify rows (boundaries < kc-1 rows in).
                conv_snap = {i: l.conv_states.clone() for i, l in dn_layers}
                set_deltanet_attn_mask(
                    cache,
                    torch.ones((1, m), dtype=torch.long, device=device),
                )
                choices, taps, _dead = self._verify(toks, sp, vs, matcher, bufs)
                t_verify = time.perf_counter()

                # Longest matching prefix over the proposal rows; padded rows
                # may extend acceptance (a padded match is still the target's
                # own choice).
                num_accepted = 0
                for j in range(m - 1):
                    if toks[j + 1] == choices[j]:
                        num_accepted += 1
                    else:
                        break
                bonus = choices[num_accepted]
                accepted = toks[1 : 1 + num_accepted]

                if dn_layers:
                    full = num_accepted == m - 1
                    round_bufs = [
                        gdr_round_bufs(
                            layer.recurrent_states.untyped_storage().data_ptr()
                        )
                        for _, layer in dn_layers
                    ]
                    if all(rb is not None for rb in round_bufs):
                        # Tape-replay rollback: the verify plan
                        # (recurrent_gdr_dvblock_save) left slot 0 at the
                        # PRE-round state and teed k/g/beta/v. Advance slot 0
                        # by the committed count with one tiny dispatch per
                        # layer (vs the serial kernel's full S-slot bank fill
                        # — pure state-write bandwidth, ~0.25ms/layer).
                        self._recon_n.numpy[0] = float(num_accepted + 1)
                        recs = []
                        for (_, layer), rb in zip(dn_layers, round_bufs):
                            rec_b = to_alloy_buffer(layer.recurrent_states)
                            gdr_state_reconstruct[(rb["BATCH"] * rb["NV"] * rb["DV"],)](
                                rb["k"], rb["g"], rb["beta"], rb["v"],
                                self._recon_n, rec_b,
                                BATCH=rb["BATCH"], S=rb["S"], NV=rb["NV"],
                                DK=rb["DK"], DV=rb["DV"], NK=rb["NK"],
                            )
                            recs.append(rec_b)
                        materialize_many(recs)
                    elif num_accepted > 0:
                        # Serial SAVE_STEPS slot-bank rollback (MTP/PLD's
                        # non-chunk-aligned widths). Mutually exclusive with
                        # the reconstruct path BY THE SAME GATE
                        # (gdr_verify_uses_reconstruct): when that gate
                        # holds, attach_spec sized the bank to 1 slot and
                        # this indexing would be out of bounds — but the
                        # registry is then complete, so this branch is
                        # unreachable.
                        for _, layer in dn_layers:
                            layer.recurrent_states[0].copy_(
                                layer.recurrent_states[num_accepted],
                            )
                    if not full:
                        for i, layer in dn_layers:
                            tape = conv_tape_tensor(
                                layer.conv_states.untyped_storage().data_ptr()
                            )  # (M, conv_dim)
                            cols = []
                            for j in range(kc):
                                row = num_accepted - (kc - 1) + j
                                if row >= 0:
                                    cols.append(tape[row].unsqueeze(-1))  # (C, 1)
                                else:
                                    cols.append(conv_snap[i][0, :, kc + row].unsqueeze(-1))
                            layer.conv_states.copy_(torch.cat(cols, dim=-1).unsqueeze(0))
                    cache.layers[0].cumulative_length.fill_(sp + num_accepted + 1)
                else:
                    cache.layers[0].cumulative_length.fill_(sp + num_accepted + 1)
                t_state = time.perf_counter()

                # Drafter ingests ALL forwarded rows (fixed shapes); the
                # committed-pointer move below makes overshoot rows dead.
                drafter.observe(toks, taps, sp)
                drafter.truncate(sp + num_accepted + 1)

                new_tokens = [*accepted, bonus]
                room = max_new_tokens - len(committed)
                if len(new_tokens) > room:
                    new_tokens = new_tokens[:room]

                metrics.add(
                    RoundStats(
                        proposed=n_prop,
                        accepted=min(num_accepted, n_prop),
                        bonus=len(new_tokens) > num_accepted,
                        draft_us=(t_draft - t0) * 1e6,
                        verify_us=(t_verify - t_draft) * 1e6,
                        state_us=(t_state - t_verify) * 1e6,
                        host_us=(time.perf_counter() - t_state) * 1e6,
                    )
                )

                done = False
                for tok in new_tokens:
                    committed.append(tok)
                    yield tok
                    if tok in stop_ids:
                        done = True
                        break
                    if matcher is not None and self._matcher_step(matcher, tok):
                        done = True
                        break
                if done:
                    break
                anchor = new_tokens[-1]
                sp = sp + len(new_tokens)

            # Next-turn warm prefix: extend the saved tokens through this
            # turn's output and bookmark the resume point (the streaming
            # path's save semantics).
            gen.prefix.extend(committed)
            gen.prefix.bookmark()

        self.last_metrics = metrics

    # ------------------------------------------------------------- helpers

    def _matcher_step(self, matcher, tok: int) -> bool:
        """Advance the real matcher; True = terminate the stream."""
        if not matcher.accept_token(tok):
            return True
        return bool(matcher.is_terminated())

    def _plain_step(self, cache, anchor: int, sp: int, matcher, bufs) -> int:
        """One non-speculative decode step (the PLD miss path / M=1 round),
        through the generator's compiled decode module — identical numerics to
        plain decode."""
        gen = self.gen
        gen.plans.token_input[0, 0] = anchor
        gen.plans.cache_position[0] = sp
        if matcher is None:
            nt = gen.decode.next_token(
                gen.plans.token_input,
                cache,
                gen.plans.cache_position,
            )
            return int(nt[0, 0].item())
        logits = gen.decode.next_logits(gen.plans.token_input, cache)
        return gen.masked_sample(
            logits,
            matcher,
            sp,
            gen.constrained_buffers(bufs["vocab"]),
        )
