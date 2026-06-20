"""DecodeEngine: the M=1 token-production loop.

Owns the compiled decode module (GreedyNextToken) and the per-step decode
loop with its chunked-plan steady state. The PlanStore supplies the pinned
input tensors and the captured replay/chunk plans.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import cast

import torch
import transformers
from transformers.cache_utils import StaticCache

from alloy._runtime import _metal_ext
from alloy_torch.backend import _execute_plan, capture_plan
from alloy_server.generation.plans import PlanStore


class GreedyNextToken(torch.nn.Module):
    model: transformers.PreTrainedModel

    def __init__(self, model: transformers.PreTrainedModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: StaticCache,
        cache_position: torch.Tensor,
        seed: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        # logits_to_keep=1 restricts lm_head to the last position. Without
        # this the lm_head dispatch runs over every prompt token even
        # though all but the final logit is discarded — a substantial
        # prefill regression on long prompts.
        output = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=cache_position,
            logits_to_keep=1,
        )
        # All AlloyStaticLayer instances alias one cumulative_length tensor, so
        # this single .add_ advances every layer's view; AOT's input-mutation
        # epilogue propagates it back before the next decode step reads it.
        past_key_values.layers[0].cumulative_length.add_(input_ids.shape[1])
        logits = output.logits
        if logits is None:
            raise ValueError("causal LM output did not include logits")
        # Logits ride along as a second output for the constrained loop, so one
        # decode plan serves both paths.
        last_logits = logits[:, -1:, :]
        token = torch.ops.alloy.sample_categorical(
            last_logits, cache_position, seed, params
        )
        return token, last_logits


class DecodeEngine:
    """Compiled decode module + the per-step/chunked decode loop."""

    def __init__(self, model: transformers.PreTrainedModel, plans: PlanStore) -> None:
        self.model = model
        self.plans = plans
        self.module: torch.nn.Module | None = None

    def next_token(
        self,
        input_ids: torch.Tensor,
        cache: StaticCache,
        cache_position: torch.Tensor,
    ) -> torch.Tensor:
        token, _logits = self.module(
            input_ids=input_ids,
            past_key_values=cache,
            cache_position=cache_position,
            seed=self.plans.seed,
            params=self.plans.params,
        )
        return cast(torch.Tensor, token)

    def next_logits(self, input_ids: torch.Tensor, cache: StaticCache) -> torch.Tensor:
        """Last-position logits — the decode plan's second output."""
        _token, logits = self.module(
            input_ids=input_ids,
            past_key_values=cache,
            cache_position=self.plans.cache_position,
            seed=self.plans.seed,
            params=self.plans.params,
        )
        return cast(torch.Tensor, logits)

    def loop(
        self,
        cache: StaticCache,
        cache_len: int,
        prompt_len: int,
        first_token: torch.Tensor,
        max_new_tokens: int,
        chunks: tuple[int, ...] = (8,),
    ) -> Iterator[int]:
        """The per-step decode loop shared by every pipeline exit, yielding
        each (1,1) next-token tensor.

        Steps 1-2 run through the compiled module (`next_token`); step 2 also
        captures the decode plan + its flat args. The steady state cascades
        through the chunked plans (`PlanStore.decode_chunk`: N iterations +
        GPU-side token feedback in one command buffer), largest `chunks` size
        that still fits first, falling back to single-step plan replay via
        `_execute_plan` for the tail / when the chunk plan can't be built. The
        cascade matters: a 128-token request on `(32, 8)` pays 3×32 + 3×8 + ≤7
        single-step command buffers instead of 3×32 + 29 — the single-step CB
        commit/wait is the dominant constant overhead. EOS is the caller's
        concern; a chunk overshoots past EOS by up to its size − 1 positions."""
        plans = self.plans
        plans.token_input.copy_(first_token)
        replay = plans.decode_replays.get(cache_len)
        step = 1
        while step <= min(2, max_new_tokens - 1):
            plans.cache_position[0] = prompt_len + step - 1
            if step == 2 and replay is None:
                with capture_plan() as slot:
                    next_token = self.next_token(
                        plans.token_input, cache, plans.cache_position,
                    )
                replay = plans.capture_decode_replay(slot, cache_len)
            else:
                next_token = self.next_token(
                    plans.token_input, cache, plans.cache_position,
                )
            yield int(next_token[0, 0].item())
            plans.token_input.copy_(next_token)
            step += 1
        chunk_states: list[tuple[int, tuple]] = []
        if replay is not None:
            for size in sorted({c for c in chunks if c > 1}, reverse=True):
                state = plans.decode_chunk(cache_len, size)
                if state is not None:
                    chunk_states.append((size, state))
        while step < max_new_tokens:
            plans.cache_position[0] = prompt_len + step - 1
            if replay is None:
                # Plan capture failed (graph break) — module call per step.
                next_token = self.next_token(
                    plans.token_input, cache, plans.cache_position,
                )
            elif (chunk_state := next(
                (s for s in chunk_states if max_new_tokens - step >= s[0]), None,
            )) is not None:
                size, (handle, gen_tokens, _keepalive) = chunk_state
                _metal_ext.dispatch_plan(handle, replay[0]._cached_input_updates)
                # gen_tokens is the chunk's int64 token ids; the GPU feedback
                # already advanced token_input/cache_position.
                yield from gen_tokens.tolist()
                step += size
                continue
            else:
                plan, args, out_idx, prop, wanted = replay
                res = _execute_plan(plan, args, wanted_outputs=wanted, args_stable=True)
                outs = res if isinstance(res, tuple) else (res,)
                for o_idx, arg_idx in prop:
                    args[arg_idx].copy_(outs[o_idx])
                next_token = outs[out_idx]
            yield int(next_token[0, 0].item())
            plans.token_input.copy_(next_token)
            step += 1
