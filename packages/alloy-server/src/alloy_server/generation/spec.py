"""Spec-decode adapter: verify module, verify-plan pinning, session attach.

The `speculative/` package owns the drafter contract and the round loop; this
module is the generation-side glue — the multi-token verify forward and the
pinned verify plan the session replays each round.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
import transformers
from transformers.cache_utils import StaticCache

from alloy_torch.backend import OutputSlot, capture_plan
from alloy_server.cache import (
    AlloyLinearAttentionLayer,
    set_spec_slot_bank,
)
from alloy_server.models.attention import (
    install_taps,
    set_deltanet_attn_mask,
    set_taps_enabled,
    set_use_alloy_warm_op,
    tap_values,
    tap_values_clear,
)
from alloy_torch.compile_window import compile_window
from alloy_torch.ops.delta_net import (
    gdr_verify_uses_reconstruct,
    prealloc_conv_tape,
    prealloc_gdr_round_bufs,
)

if TYPE_CHECKING:
    from alloy_server.generation.generator import AlloyGenerator
    from alloy_server.speculative.session import SpecSession


class VerifySpecLogits(torch.nn.Module):
    """Multi-token verify for the speculative session:
    returns per-position LOGITS — sampling happens outside the plan,
    position-keyed per row, so spec output is bit-identical to plain decode at
    the same seed — plus the drafter's tap hidden states.

    Tap layer outputs ride HF's `output_hidden_states`
    (hidden_states[i+1] == output of decoder layer i); layers nobody returns
    are dead lazy buffers, so the flag costs nothing on the GPU."""

    model: transformers.PreTrainedModel

    def __init__(
        self,
        model: transformers.PreTrainedModel,
        layer_ids: tuple[int, ...],
        post_norm: bool,
    ) -> None:
        super().__init__()
        self.model = model
        self.layer_ids = tuple(layer_ids)
        self.post_norm = bool(post_norm)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: StaticCache,
        cache_position: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        # Sink clears happen HOST-SIDE before each compile-time call (the
        # pin loop / module fallback): an in-graph clear makes Dynamo guard
        # on the global list's ENTRY state, and call 0's replayed appends
        # then force a recompile on call 1 — the pin's cached-execute capture
        # never happens.
        out = self.model.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=cache_position,
        )
        hidden = out.last_hidden_state
        past_key_values.layers[0].cumulative_length.add_(input_ids.shape[1])
        logits = self.model.lm_head(hidden)
        outputs = [logits]
        if self.layer_ids:
            outputs.extend(tap_values())
        if self.post_norm:
            outputs.append(hidden)
        return tuple(outputs)


def prealloc_dn_scratch(cache: StaticCache, m: int) -> None:
    """(Re)allocate the per-layer DeltaNet verify scratch — conv tape + GDR
    round buffers — keyed by the CURRENT conv/recurrent storage pointers.
    Idempotent per (storage, m). Called at pin time and again whenever a
    slice switch repoints the cache to different storage (the registries are
    data_ptr-keyed, so each slice gets its own entry)."""
    for layer in cache.layers:
        if not isinstance(layer, AlloyLinearAttentionLayer):
            continue
        prealloc_conv_tape(
            layer.conv_states.untyped_storage().data_ptr(),
            m, int(layer.conv_states.shape[1]),
        )
        rec = layer.recurrent_states
        nv, dk, dv = int(rec.shape[2]), int(rec.shape[3]), int(rec.shape[4])
        key_dim = (int(layer.conv_states.shape[1]) - nv * dv) // 2
        prealloc_gdr_round_bufs(
            rec.untyped_storage().data_ptr(),
            m, nv, dk, dv, key_dim // dk,
        )


def attach_spec_session(gen: AlloyGenerator, drafter) -> SpecSession:
    """Bind a contract drafter to the generator and create the SpecSession
    that owns its round loop. One drafter per generator; `eager_compile_all`
    warms the session's plans."""
    from alloy_server.speculative.session import SpecSession  # scoped: avoid import cycle (speculative imports generation types)

    drafter.bind(gen)
    if drafter.taps.layer_ids:
        install_taps(gen.model, drafter.taps.layer_ids)
    session = SpecSession(gen, drafter)
    # DeltaNet recurrent slot bank. Must run before any cache is
    # constructed — attach precedes eager_compile_all in every flow.
    # Chunk-aligned verify widths (DFlash block 16, PLD block 8) dispatch
    # the dvblock+reconstruct path, which never writes past slot 0 — one
    # slot suffices (16 → 1 ≈ −400MB resident on qwen3.5:4b). Non-aligned
    # widths (MTP) fill the per-row bank via the serial SAVE_STEPS kernel
    # and need one slot per verify row.
    text = gen.model.config
    if hasattr(text, "get_text_config"):
        text = text.get_text_config(decoder=True)
    dv = (
        int(text.linear_value_head_dim)
        if hasattr(text, "linear_value_head_dim")
        else 0
    )
    if dv and gdr_verify_uses_reconstruct(session.verify_m, dv):
        set_spec_slot_bank(1)
    else:
        set_spec_slot_bank(max(8, session.verify_m))
    return session


def pin_verify_plan(gen: AlloyGenerator, m: int, taps):
    """Compile + pin the M-row spec-verify plan (logits + tap hiddens,
    SAVE_STEPS per-token DeltaNet state slots) against the persistent
    max-cache. The session replays it via `_execute_plan` each round —
    zero Dynamo."""
    from alloy_server.speculative.session import VerifyState  # scoped: avoid import cycle (speculative imports generation types)

    device = next(gen.model.parameters()).device
    verify = cast(
        torch.nn.Module,
        torch.compile(
            VerifySpecLogits(gen.model, taps.layer_ids, taps.post_norm),
            backend="alloy", dynamic=False,
        ),
    )
    cache = gen.kv.acquire(1, gen.kv.max_cache_len)
    prefill_len = 8
    with torch.inference_mode():
        gen.decode.next_token(
            torch.zeros((1, prefill_len), dtype=torch.long, device=device), cache,
            torch.arange(prefill_len, dtype=torch.int32, device=device),
        )
        pin_in = torch.zeros((1, m), dtype=torch.long, device=device)
        pin_pos = torch.arange(prefill_len, prefill_len + m, dtype=torch.int32, device=device)
        set_deltanet_attn_mask(cache, torch.ones((1, m), dtype=torch.long, device=device))
        # Conv tape banks and the verify GDR round buffers must exist
        # BEFORE the recorded pin runs — in-recording allocations
        # classify as pooled intermediates and the replayed tee writes
        # would land in recycled pool memory.
        prealloc_dn_scratch(cache, m)
        set_use_alloy_warm_op(True)
        compile_window.spec_save_steps = True
        compile_window.q_start_pos = prefill_len
        set_taps_enabled(bool(taps.layer_ids))
        try:
            with capture_plan() as slot:
                for _ in range(2):  # build (run 0) + cached execute (run 1)
                    # Host-side sink clear: an in-graph clear makes Dynamo
                    # guard on the global list's ENTRY state, and run 0's
                    # replayed appends then force a recompile on run 1 —
                    # the capture never reaches the cached-execute path.
                    tap_values_clear()
                    verify(input_ids=pin_in, past_key_values=cache, cache_position=pin_pos)
        finally:
            set_use_alloy_warm_op(False)
            compile_window.spec_save_steps = False
            set_taps_enabled(False)
            compile_window.q_start_pos = 0  # stale start corrupts later cold prefills
    plan, args = slot.plan, slot.args
    if plan is None or args is None or plan._cached_input_updates is None:
        raise RuntimeError(
            f"spec verify plan capture failed at M={m} (graph break?) — "
            "speculative decoding needs the pinned-plan fast path"
        )
    # Outputs follow the module's return order (logits, *taps, post_norm?):
    # _build_output_mapping preserves FX output order. Identify by ordered
    # shape-walk over the OutputSlot entries.
    hidden_size = gen.model.config.hidden_size
    vocab = gen.model.config.vocab_size
    logits_idx = -1
    hidden_idxs: list[int] = []
    for i, entry in enumerate(plan.output_mapping):
        if not isinstance(entry, OutputSlot):
            continue
        if entry.shape == (1, m, vocab):
            logits_idx = i
        elif entry.shape == (1, m, hidden_size):
            hidden_idxs.append(i)
    n_taps = len(taps.layer_ids)
    expected_hidden = n_taps + (1 if taps.post_norm else 0)
    if logits_idx < 0 or len(hidden_idxs) != expected_hidden:
        raise RuntimeError(
            f"spec verify plan outputs malformed: logits_idx={logits_idx}, "
            f"hidden outputs {len(hidden_idxs)} != expected {expected_hidden}"
        )
    tap_idxs = tuple(hidden_idxs[:n_taps])
    post_norm_idx = hidden_idxs[n_taps] if taps.post_norm else -1
    return VerifyState(
        plan, args, pin_in, pin_pos, logits_idx, tap_idxs, post_norm_idx,
    )
