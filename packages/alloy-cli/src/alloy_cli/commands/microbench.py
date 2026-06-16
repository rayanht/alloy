"""`alloy microbench <model> <kernel>` — clock-pinned timing for one kernel at
its real production config.

GPU clock state varies several-fold from one run to the next, so a meaningful
measurement needs the clock pinned high. This captures the kernel's exact
production dispatch (constexprs and buffers resolved from a real model forward),
warms it heavily to drive the clock to its ceiling, then reports the median GPU
time and spread over many timed runs.

To compare an old kernel against a new one, run this before and after the change
and compare the medians — each run pins the clock the same way.

  alloy microbench qwen2.5:3b attention_strided_runtime_pos --depth 16384
  alloy microbench qwen2.5:3b attention_decode_vector_split --decode
"""

from __future__ import annotations

from typing import Annotated

import numpy as np
import typer
from rich.console import Console

from alloy._runtime import _metal_ext
from alloy._runtime.tune import dispatch_captured
from alloy_cli.capture import _DEFAULT_DEPTH, capture_kernel_dispatch

console = Console()

_WARMUP = 60  # heavy warmup to drive the GPU clock to its ceiling before timing


def microbench(
    model: Annotated[str, typer.Argument(help="model name, e.g. qwen2.5:3b")],
    kernel: Annotated[str, typer.Argument(help="kernel name, e.g. attention_strided_runtime_pos")],
    depth: Annotated[
        int,
        typer.Option("--depth", help="prefill cache depth to capture at (matches `alloy bench --depths`)."),
    ] = _DEFAULT_DEPTH,
    decode: Annotated[
        bool,
        typer.Option("--decode", help="time the kernel from the decode (M=1) forward; prefill is the default."),
    ] = False,
    runs: Annotated[
        int,
        typer.Option("--runs", help="timed runs after warmup."),
    ] = 100,
) -> None:
    """Report clock-pinned GPU timing for one kernel at its production config."""
    console.print(f"[cyan]loading[/] {model} …")
    try:
        matches, all_names = capture_kernel_dispatch(model, kernel, depth=depth, decode=decode)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if not matches:
        hint = "" if not all_names else f" Kernels seen: {', '.join(all_names)}"
        other = "prefill" if decode else "decode"
        raise typer.BadParameter(
            f"{kernel!r} was not dispatched by {model}'s "
            f"{'decode' if decode else 'prefill'} forward.{hint} "
            f"(Try --{'no-decode' if decode else 'decode'} for the {other} pass.)"
        )
    if len(matches) > 1:
        # A kernel dispatched at several shapes (e.g. q/k/v/o projections). Time the
        # first; name the rest so the user can disambiguate with a fuller name.
        console.print(
            f"[yellow]note[/] {kernel!r} matched {len(matches)} dispatches; timing the first "
            f"({matches[0].name}). Pass a more specific name to pick another.",
            soft_wrap=True,
        )
    disp = matches[0]
    cap = disp.captured

    # One timed run = total GPU µs across the dispatch(es) the kernel emits (a
    # single kernel → one dispatch; a composed replay → the composed cost).
    captured_times: list[float] = []
    orig_dispatch = _metal_ext.dispatch

    def hook(groups):
        r = orig_dispatch(groups)
        if isinstance(r, dict) and "gpu" in r:
            captured_times.append(r["gpu"] * 1000.0)
        return r

    def run() -> float:
        captured_times.clear()
        dispatch_captured(cap)
        return sum(captured_times)

    console.print(
        f"[cyan]timing[/] {disp.name} ({disp.pass_name}) — {_WARMUP} warmup + {runs} runs, clock-pinned …"
    )

    _metal_ext.dispatch = hook
    try:
        for _ in range(_WARMUP):
            run()
        times = [run() for _ in range(runs)]
    finally:
        _metal_ext.dispatch = orig_dispatch

    arr = np.sort(np.asarray(times))
    median = float(np.median(arr))
    p20 = float(arr[len(arr) // 5])
    p80 = float(arr[(len(arr) * 4) // 5])
    lo = float(arr[0])
    console.print(f"\n[bold]{disp.name}[/]  ({disp.pass_name})")
    console.print(
        f"  median {median:7.1f} µs   (p20–p80 {p20:.0f}–{p80:.0f}, min {lo:.0f}, n {len(arr)})",
        highlight=False,
    )
