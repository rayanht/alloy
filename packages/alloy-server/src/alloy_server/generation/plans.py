"""PlanStore: pinned compiled plans, replay capture, and compile windows.

Owns every piece of pinned-plan state the generation engines replay: the
prefill/decode plan tables, the chunked-decode registrations, and the pinned
input tensors whose stable storage pointers the captured plans bind.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import torch

from alloy._dispatch.buf_utils import _alloc_aligned, set_record_only
from alloy._compiler.dtypes import int64
from alloy._runtime import _metal_ext
from alloy_torch.backend import (
    CapturedPlan,
    InputSlot,
    OutputSlot,
    register_decode_chunk_plan,
)
from alloy_torch.tensor_bridge import make_tensor_from_ptr
from alloy.std.sampling import SAMPLE_SPLITS
from alloy_torch.compile_window import grid_shrink_compile

# GPU-side feedback for the chunked decode plan: copy the sampled token into
# the next iteration's token input, record it in the generated buffer (bound
# at byte offset 8*i per iteration), and advance cache_position. One thread —
# it's a scalar update between full decode iterations inside one command buffer.
DECODE_UPDATE_MSL = """
#include <metal_stdlib>
using namespace metal;
kernel void alloy_decode_chunk_update(
    device const long* token_out [[buffer(0)]],
    device long* token_in [[buffer(1)]],
    device int* cache_position [[buffer(2)]],
    device long* generated [[buffer(3)]]) {
  long token = token_out[0];
  token_in[0] = token;
  generated[0] = token;
  cache_position[0] = cache_position[0] + 1;
}
"""

# 8-byte scalar copy for AOT input-mutation propagation between chunk
# iterations (per-layer cumulative_length etc.) — the GPU-side equivalent of
# the `args[arg_idx].copy_(outs[o_idx])` the single-step replay does in Python.
DECODE_COPY8_MSL = """
#include <metal_stdlib>
using namespace metal;
kernel void alloy_decode_chunk_copy8(
    device const long* src [[buffer(0)]],
    device long* dst [[buffer(1)]]) {
  dst[0] = src[0];
}
"""

# 8-byte scalar self-increment for the cumulative_length counter mutation. The
# model's `cumulative_length.add_(input_ids.shape[1])` is a self-increment
# (= +1 at M=1 decode) whose only consumer is the AOT mutation writeback.
# Folding it here lets the chunked plan elide that standalone `add` dispatch
# (disconnected from the compute DAG, it drained the pipeline every token) and
# its copy8 propagation — the counter advances in the feedback instead.
DECODE_INCR8_MSL = """
#include <metal_stdlib>
using namespace metal;
kernel void alloy_decode_chunk_incr8(device long* dst [[buffer(0)]]) {
  dst[0] = dst[0] + 1;
}
"""

# Token feedback + ONE folded counter increment in a single dispatch. The common
# decode shape has exactly one self-increment counter (the shared
# cumulative_length); folding its incr8 into the update kernel makes the
# per-iteration feedback ONE dispatch instead of two, removing a PSO switch + a
# 1-thread launch from every token's feedback group.
DECODE_UPDATE_INCR1_MSL = """
#include <metal_stdlib>
using namespace metal;
kernel void alloy_decode_chunk_update_incr1(
    device const long* token_out [[buffer(0)]],
    device long* token_in [[buffer(1)]],
    device int* cache_position [[buffer(2)]],
    device long* generated [[buffer(3)]],
    device long* counter [[buffer(4)]]) {
  long token = token_out[0];
  token_in[0] = token;
  generated[0] = token;
  cache_position[0] = cache_position[0] + 1;
  counter[0] = counter[0] + 1;
}
"""

DECODE_CHUNK_PSOS: tuple[int, int, int, int] | None = None


def decode_chunk_psos() -> tuple[int, int, int, int]:
    """(update, copy8, incr8, update_incr1) PSO handles for the chunked decode
    plan, compiled once."""
    global DECODE_CHUNK_PSOS
    if DECODE_CHUNK_PSOS is None:
        DECODE_CHUNK_PSOS = (
            _metal_ext.compile_msl(DECODE_UPDATE_MSL, "alloy_decode_chunk_update"),
            _metal_ext.compile_msl(DECODE_COPY8_MSL, "alloy_decode_chunk_copy8"),
            _metal_ext.compile_msl(DECODE_INCR8_MSL, "alloy_decode_chunk_incr8"),
            _metal_ext.compile_msl(DECODE_UPDATE_INCR1_MSL, "alloy_decode_chunk_update_incr1"),
        )
    return DECODE_CHUNK_PSOS


class PlanStore:
    """Pinned plans + the stable-storage tensors their captured args bind."""

    def __init__(self, hidden_size: int, grid_shrink: bool) -> None:
        self.hidden_size = hidden_size
        self.grid_shrink = grid_shrink
        # Pinned (plan, args, next_token_idx, tap_idxs) per (prefill_chunk,
        # is_warm). When set, prefill bypasses torch.compile/Dynamo entirely
        # and replays the plan directly.
        self.prefill_plans: dict[tuple[int, bool], tuple] = {}
        # input_ids / cache_position / last_real_pos / attention_mask tensors
        # held stable per prefill chunk so the captured args' storage pointers
        # don't invalidate between calls.
        self.prefill_inputs: dict[int, tuple[torch.Tensor, ...]] = {}
        # Pinned decode plan keyed by cache length (single entry under the
        # native length; the key is kept for the lookup call sites).
        self.decode_plans: dict[int, object] = {}
        # Per-cache_len decode replay state (plan, flat args, token-out idx,
        # mutation-propagation pairs, wanted outputs).
        self.decode_replays: dict[int, tuple] = {}
        # Chunked decode plans keyed by (cache_len, chunk): the decode plan's
        # dispatch list repeated `chunk` times in ONE registered plan with
        # GPU-side token feedback between iterations.
        self.decode_chunk_plans: dict[tuple[int, int], tuple] = {}
        # Pinned (1,1) token input and (1,) cache_position for the per-step
        # decode loop. The plan's input check matches tensors by storage_ptr —
        # fresh-per-call tensors would force the slow input-rebind path.
        self.token_input = torch.zeros((1, 1), dtype=torch.long)
        self.cache_position = torch.zeros((1,), dtype=torch.int32)
        # Sampling config as stable input slots. Default = greedy (temperature
        # 0 -> exact argmax). Written in place; the GPU reads them live, so
        # switching greedy<->sampling needs no recompile.
        # params layout: [temperature, top_p, top_k, min_p, n_splits]. n_splits
        # is the sampler's active vocab-split count (SAMPLE_SPLITS for greedy/
        # pure-temp, 1 when a top-k/p/min-p filter needs the global bisection),
        # set per request in generator.
        self.seed = torch.zeros((1,), dtype=torch.long)
        self.params = torch.tensor(
            [0.0, 1.0, 0.0, 0.0, float(SAMPLE_SPLITS)], dtype=torch.float32
        )

    def pinned_inputs_for_chunk(
        self, chunk: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Lazily allocate the (input_ids, cache_position, last_real_pos,
        attention_mask) tensors pinned to this prefill chunk. attention_mask
        is (1, chunk) with 1 for real tokens and 0 for padding — used by
        DeltaNet layers to mask pad positions out of the recurrent state."""
        existing = self.prefill_inputs.get(chunk)
        if existing is not None:
            return existing
        quad = (
            torch.zeros((1, chunk), dtype=torch.long, device=device),
            torch.arange(chunk, dtype=torch.int32, device=device),
            torch.zeros((1,), dtype=torch.long, device=device),
            torch.ones((1, chunk), dtype=torch.long, device=device),
        )
        self.prefill_inputs[chunk] = quad
        return quad

    def pin_prefill_plan(self, chunk: int, is_warm: bool, captured: CapturedPlan) -> None:
        """Pin (plan, args, next_token_idx, tap_idxs) from a `capture_plan()`
        scope for the prefill replay fast path. Caller has run a prefill at
        (chunk, is_warm) inside the scope twice (build + cached execute), so
        `captured` holds the executed plan with `_cached_input_updates`
        populated.

        `next_token_idx` is the FX flat-output entry for the returned
        next_token (shape (1, 1) int64); other entries are AOT input-mutation
        tracebacks for cache state. tap entries are drafter tap outputs."""
        plan = captured.plan
        args = captured.args
        if plan is None or args is None:
            return
        next_token_idx = -1
        tap_idxs: list[int] = []
        for i, entry in enumerate(plan.output_mapping):
            if not isinstance(entry, OutputSlot):
                continue
            if entry.dtype.ir == "i64" and entry.shape == (1, 1):
                next_token_idx = i
            elif entry.shape == (1, chunk, self.hidden_size):
                tap_idxs.append(i)  # drafter tap outputs, in return order
        if next_token_idx < 0:
            return
        self.prefill_plans[(chunk, is_warm)] = (
            plan, args, next_token_idx, tuple(tap_idxs),
        )

    def find_plan_input_slot(self, plan, tensor: torch.Tensor) -> int | None:
        """Slot index of the INPUT slot bound to `tensor` (by storage identity)."""
        checks = plan._cached_input_check
        if checks is None:
            return None
        storage_ptr = tensor.untyped_storage().data_ptr()
        data_ptr = tensor.data_ptr()
        arg_idx = None
        for check in checks:
            if check.storage_ptr == storage_ptr and check.data_ptr == data_ptr:
                arg_idx = check.arg_idx
                break
        if arg_idx is None:
            return None
        for si, plan_slot in enumerate(plan.slots):
            if isinstance(plan_slot, InputSlot) and plan_slot.arg_idx == arg_idx:
                return si
        return None

    def decode_chunk(self, cache_len: int, chunk: int):
        """(plan_handle, generated-token tensor, keepalive) for the chunked
        decode plan — `chunk` decode iterations + GPU-side feedback in one
        registered plan / one command buffer. Built lazily from the captured
        replay state; None when the replay slots can't be resolved (the loop
        then stays per-step)."""
        key = (cache_len, chunk)
        state = self.decode_chunk_plans.get(key)
        if state is not None:
            return state
        replay = self.decode_replays.get(cache_len)
        if replay is None:
            return None
        plan, _args, out_idx, prop, _wanted = replay
        entry = plan.output_mapping[out_idx]
        token_in = self.find_plan_input_slot(plan, self.token_input)
        pos = self.find_plan_input_slot(plan, self.cache_position)
        if token_in is None or pos is None:
            return None
        # Split AOT input mutations into two kinds. A self-increment counter
        # (cumulative_length: out = cumulative_length + 1, produced by a scalar
        # add whose ONLY consumer is this writeback) folds into the GPU feedback
        # as `dst += 1`, eliding the producing add (a dispatch disconnected from
        # the compute DAG that drained the pipeline every token). Everything
        # else keeps the copy8 src->dst propagation.
        producer_of: dict[int, int] = {}
        for di, d in enumerate(plan.dispatches):
            for ws in d.write_slot_indices:
                producer_of.setdefault(ws, di)
        read_anywhere: set[int] = set()
        for d in plan.dispatches:
            read_anywhere.update(
                si for si in d.buf_slot_indices if si not in d.write_slot_indices
            )
        prop_pairs: list[tuple[int, int, int]] = []
        incr_slots: list[int] = []
        skip_dispatch: set[int] = set()
        for o_idx, _arg_idx in prop:
            e = plan.output_mapping[o_idx]
            src_slot = e.slot_idx
            dst_slot = plan.mutation_input_slots[o_idx]
            pi = producer_of.get(src_slot)
            foldable = False
            if pi is not None and src_slot not in read_anywhere:
                pd = plan.dispatches[pi]
                pd_reads = {
                    si for si in pd.buf_slot_indices if si not in pd.write_slot_indices
                }
                # Self-increment: the producer reads the destination (the counter)
                # and writes only this output, so out = f(counter) feeding nothing
                # else. At M=1 decode the advance is +1 (input_ids.shape[1]==1).
                if dst_slot in pd_reads and len(set(pd.write_slot_indices)) == 1:
                    foldable = True
            if foldable:
                incr_slots.append(dst_slot)
                skip_dispatch.add(pi)
            else:
                prop_pairs.append((src_slot, e.byte_offset, dst_slot))
        update_pso, copy_pso, incr_pso, update_incr1_pso = decode_chunk_psos()
        gen = _alloc_aligned((chunk,), int64)
        handle = register_decode_chunk_plan(
            plan,
            token_input_slot_idx=token_in,
            cache_position_slot_idx=pos,
            token_out_slot_idx=entry.slot_idx,
            token_out_byte_offset=entry.byte_offset,
            generated_handle=gen._parent_handle,
            generated_nbytes=gen.metal_nbytes,
            chunk=chunk,
            update_pso=update_pso,
            copy_pso=copy_pso,
            incr_pso=incr_pso,
            update_incr1_pso=update_incr1_pso,
            prop_slot_pairs=tuple(prop_pairs),
            incr_slots=tuple(incr_slots),
            skip_dispatch_indices=frozenset(skip_dispatch),
        )
        gen_tokens = make_tensor_from_ptr(
            gen.base_ptr, (chunk,), int64, total_nbytes=gen.metal_nbytes,
        )
        state = (handle, gen_tokens, gen)
        self.decode_chunk_plans[key] = state
        return state

    def capture_decode_replay(self, slot: CapturedPlan, cache_len: int):
        """Pin (plan, args, token-out idx, mutation-propagation pairs) from a
        `capture_plan()` scope around a pinned decode step — the per-step
        replay state for the decode loop. Returns None if the plan wasn't
        captured (graph break) so the loop falls back to module calls."""
        plan, args = slot.plan, slot.args
        if plan is None or args is None or plan._cached_input_updates is None:
            return None
        out_idx = -1
        for i, entry in enumerate(plan.output_mapping):
            # The sampled-token output is (1,1) i64; take the LAST such output
            # (matches the slot the decode graph's sampler writes).
            if isinstance(entry, OutputSlot) and entry.dtype.ir == "i64" and entry.shape == (1, 1):
                out_idx = i
        if out_idx < 0:
            return None
        # AOT's runtime epilogue copies mutated outputs back into their input
        # tensors after each module call (per-layer cumulative_length etc.).
        # Raw plan replay bypasses AOT, so replicate it for the mutations not
        # already remapped in-plan: copy output[o] into the flat arg the
        # mutated input slot binds.
        prop: list[tuple[int, int]] = []
        for o_idx, in_slot_idx in plan.mutation_input_slots.items():
            if o_idx >= len(plan.output_mapping):
                continue
            entry = plan.output_mapping[o_idx]
            if not isinstance(entry, OutputSlot):
                continue
            if isinstance(plan.slots[entry.slot_idx], InputSlot):
                continue  # already remapped in-plan — nothing to propagate
            in_slot = plan.slots[in_slot_idx]
            if not isinstance(in_slot, InputSlot):
                continue
            prop.append((o_idx, in_slot.arg_idx))
        self.decode_plans[cache_len] = plan
        # Only materialize the outputs the loop reads — the token + the
        # mutation sources. The rest (per-layer KV writeback tracebacks)
        # alias input storages and need no Python-side tensor per step.
        wanted = frozenset({out_idx} | {o for o, _ in prop})
        replay = (plan, args, out_idx, tuple(prop), wanted)
        self.decode_replays[cache_len] = replay
        return replay

    def clear_decode_state(self) -> None:
        self.decode_plans.clear()
        self.decode_replays.clear()
        self.decode_chunk_plans.clear()

    @contextlib.contextmanager
    def compile_window(self, chunk: int | None = None) -> Iterator[None]:
        """Production plan-compile window: record-only + grid-shrink flags.

        Any compile-triggering forward OUTSIDE `eager_compile_all` (the capture
        paths behind `alloy profile` / `alloy inspect`) must run inside this
        window, for two reasons:

        - record-only gives run-0 phantom intermediates and skips the GPU. A
          REAL run-0 materializes every intermediate of the whole forward at
          once (each an individually-allocated Metal buffer held live by the
          dispatch recording, Metal wiring full residency at first encoder
          use) — 100+ GB on qwen3.6:35b's 4096-chunk prefill, an instant OOM.
        - the grid-shrink flags make the captured plan THE production plan
          (single-pass shrinkable attention, M-saturated config resolution,
          request-bounded intermediate pool). Without them the plan compiles
          with the handler's split-K choice and an unbounded pool.

        `chunk` is the prefill chunk size; `None` is the M=1 decode/verify
        window (record-only alone, matching the decode loop).
        """
        shrinkable = chunk is not None and self.grid_shrink
        with contextlib.ExitStack() as stack:
            if chunk is not None:
                stack.enter_context(grid_shrink_compile(chunk if shrinkable else 0))
            set_record_only(True)
            stack.callback(set_record_only, False)
            yield
