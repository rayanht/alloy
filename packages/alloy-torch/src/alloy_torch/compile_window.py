"""Compile-window flags: the single trace-boundary channel for plan compiles.

Op handlers run inside Dynamo tracing with fixed ATen signatures, so a flag
that steers a plan compile cannot be threaded through `model.forward` as an
argument. All such flags live here, on one explicit object, set around
compile windows (and per chunk_step, where a mid-stream compile may fire).
They act at trace/plan-compile time only — plan replays never read them.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from alloy._runtime.tune_configs import set_grid_shrink_resolve_cap


@dataclass
class CompileWindow:
    # >0: the plan being compiled is shrink-capable at M=shrink_m. Forces
    # single-pass attention (a split-K plan's grids don't shrink per request),
    # caps M-scaled config resolution to the representative-M tune, and bounds
    # the M-outer intermediate pool.
    shrink_m: int = 0
    # Tuner-only: force single-pass attention WITHOUT the resolve-cap/pool
    # coupling, so the shrink-chunk tune benchmarks the kernels the
    # shrink-capable plan runs.
    single_pass_attention: bool = False
    # Warm-prefill start position. >0: the KV holds a populated prefix, so
    # the SDPA handler keeps the full K/V extent instead of the cold
    # slice-to-q_len; the maskless `attention_strided` bakes the value as its
    # causal early-exit Q_START_POS constexpr.
    q_start_pos: int = 0
    # DeltaNet bakes SAVE_STEPS=1 (+ conv tape) for the spec verify plan.
    spec_save_steps: bool = False

    def grid_shrink_active(self) -> bool:
        return self.shrink_m > 0 or self.single_pass_attention


compile_window = CompileWindow()


@contextmanager
def grid_shrink_compile(m: int) -> Iterator[None]:
    """Shrink-capable compile window at M=m; m == 0 is a plain compile."""
    compile_window.shrink_m = int(m)
    set_grid_shrink_resolve_cap(m)
    try:
        yield
    finally:
        compile_window.shrink_m = 0
        set_grid_shrink_resolve_cap(0)
