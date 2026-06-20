"""`alloy profile <model> [--depth N]` — emit interactive HTML viz for the
prefill and decode dispatch plans via `alloy.visualize`.

Profiles the production paths on synthetic tokens at cache depth `--depth` (the
same workload as `alloy bench --depths`), so a slow (model, depth) bench row
profiles 1:1 here. Writes `<model>_prefill.html` and `<model>_decode.html` —
each with the FX graph DAG, the compiled dispatch DAG (data-flow edges),
per-kernel GPU timing, and the buffer/weight breakdown. Prefill is the real
chunked cold→warm loop and decode is the M=1 step at that depth.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import webbrowser
from pathlib import Path
from typing import Annotated

import torch
import typer
from rich.console import Console

import alloy
from alloy._runtime import _metal_ext
from alloy._runtime.tune import dispatch_captured
from alloy_cli.capture import _DEFAULT_DEPTH, build_capture, capture_kernel_dispatch

console = Console()


def _capture_gputrace(
    model: str, kernel: str, *, depth: int, decode: bool, mtp: bool, out_dir: Path
) -> None:
    """Write a Metal GPU trace of one kernel dispatched in isolation at its real
    production config, for opening in Xcode (Performance / Counters).

    Captures the kernel's true dispatch from a model forward (constexprs +
    buffers), warms it up so the trace is pure dispatch, then records a few
    replays inside an MTLCaptureManager window.
    """
    if os.environ.get("MTL_CAPTURE_ENABLED") != "1":
        os.environ["MTL_CAPTURE_ENABLED"] = "1"
        os.execvpe(sys.argv[0], sys.argv, os.environ)

    console.print(f"[cyan]loading[/] {model} …")
    try:
        matches, all_names = capture_kernel_dispatch(model, kernel, depth=depth, decode=decode, mtp=mtp)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if not matches:
        hint = f" Kernels seen: {', '.join(all_names)}" if all_names else ""
        raise typer.BadParameter(f"{kernel!r} was not dispatched by {model}.{hint}")
    disp = matches[0]
    if len(matches) > 1:
        console.print(f"[yellow]note[/] matched {len(matches)} dispatches; capturing {disp.name}.")
    cap = disp.captured

    for _ in range(3):  # warm up (compile + plan) outside the capture window
        dispatch_captured(cap)

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = model.replace(":", "_").replace("/", "_")
    out_path = out_dir / f"{safe}.{disp.name}.gputrace"
    if out_path.exists():
        shutil.rmtree(out_path, ignore_errors=True)

    err = _metal_ext.capture_start(str(out_path))
    if err:
        console.print(f"[red]{err}[/]", soft_wrap=True)
        raise typer.Exit(1)
    for _ in range(5):
        dispatch_captured(cap)
    _metal_ext.capture_stop()
    console.print(f"[green]wrote[/] {disp.name} ({disp.pass_name}) GPU trace", soft_wrap=True)
    console.print(out_path.as_uri(), soft_wrap=True, highlight=False)
    console.print(
        "open in Xcode → pick a compute dispatch → Performance / Counters", highlight=False
    )


def _print_kernel_summary(title: str, kernels: list[dict] | None) -> None:
    """Aggregate per-dispatch GPU timings by base kernel name (stripping the
    trailing per-layer index) and print a top-by-total-μs table — so a 36-layer
    model shows one row per kernel with an xN instance count."""
    if not kernels:
        console.print(f"\n[bold]=== {title} — no GPU timings captured ===[/]")
        return
    agg: dict[str, list[float]] = {}
    for k in kernels:
        base = re.sub(r"_\d+$", "", k["name"])
        slot = agg.setdefault(base, [0.0, 0])
        slot[0] += k["gpu_us"]
        slot[1] += 1
    total = sum(us for us, _ in agg.values())
    console.print(f"\n[bold]=== {title} — top kernels by total GPU μs ===[/]")
    for base, (us, count) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:15]:
        pct = us / total * 100 if total else 0.0
        if pct < 0.1:  # skip negligible rows; TOTAL still reflects everything
            break
        console.print(f"   {base:<30}{us:>8.0f}μs  ({pct:4.1f}%)  x{count}", highlight=False)
    console.print(f"   {'TOTAL':<30}{total:>8.0f}μs", highlight=False)


def profile(
    model: Annotated[str, typer.Argument(help="model name, e.g. qwen2.5:3b")],
    depth: Annotated[
        int,
        typer.Option(
            "--depth",
            help="Prefill cache depth to profile — a synthetic N-token prompt run "
                 "through the production chunked prefill. Matches `alloy bench --depths`, "
                 "so a slow (model, depth) bench row profiles 1:1 here. Default 4096.",
        ),
    ] = _DEFAULT_DEPTH,
    batch: Annotated[
        int | None,
        typer.Option(help="embedding models only: batch size (pins a single shape)"),
    ] = None,
    seq: Annotated[
        int | None,
        typer.Option(help="embedding models only: sequence length (pins a single shape)"),
    ] = None,
    mtp: Annotated[
        bool,
        typer.Option(
            "--mtp",
            help="also profile the MTP self-speculation round (M=2 verify + M=1 draft) "
            "at --depth, so the per-kernel breakdown shows the speculation overhead. "
            "Requires a model that ships an MTP head (qwen3.5).",
        ),
    ] = False,
    capture: Annotated[
        str | None,
        typer.Option(
            "--capture",
            help="write a Metal GPU trace (.gputrace) of this one kernel dispatched "
            "in isolation at its production config, for opening in Xcode.",
        ),
    ] = None,
    decode: Annotated[
        bool,
        typer.Option("--decode", help="with --capture: trace from the decode (M=1) forward; prefill is the default."),
    ] = False,
    decode_only: Annotated[
        bool,
        typer.Option("--decode-only", help="only profile the decode (M=1) pass — skip prefill and the grid-shrunk chunk."),
    ] = False,
    kv_quant: Annotated[
        str | None,
        typer.Option(
            "--kv-quant",
            help="profile with the quantized KV cache active (e.g. q8_0) — the "
            "passes dispatch the q8 attention variants production serves. "
            "Exported as ALLOY_KV_QUANT.",
        ),
    ] = None,
    out_dir: Annotated[
        Path,
        typer.Option(help="directory to write the output files into"),
    ] = Path("."),
    open_browser: Annotated[
        bool,
        typer.Option("--open/--no-open", help="open the HTML in the default browser when done"),
    ] = False,
) -> None:
    """Capture dispatch-plan visualizations via `alloy.visualize`.

    Causal models emit a prefill (chunked, at `--depth`) + decode pass on
    synthetic tokens; embedding models emit a short and a long (batch, seq) pass
    by default, or a single shape via --batch/--seq (--depth is ignored).

    With `--capture <kernel>`, writes a GPU trace of that one kernel in isolation
    (for Xcode) rather than the HTML visualizations.
    """
    if kv_quant is not None:
        os.environ["ALLOY_KV_QUANT"] = kv_quant
    if capture is not None:
        _capture_gputrace(model, capture, depth=depth, decode=decode, mtp=mtp, out_dir=out_dir)
        return

    console.print(f"[cyan]loading[/] {model} …")
    try:
        cap = build_capture(model, depth=depth, batch=batch, seq=seq, mtp=mtp)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if decode_only:
        cap.passes = [p for p in cap.passes if p.name == "decode"]
        if not cap.passes:
            raise typer.BadParameter(f"--decode-only has no decode pass for {model} (embedders have none)")

    out_dir = out_dir.resolve()  # absolute so the "wrote" lines are cmd+clickable
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = model.replace(":", "_").replace("/", "_")

    written: list[Path] = []
    with torch.inference_mode():
        for p in cap.passes:
            path = out_dir / f"{safe}_{p.name}.html"
            console.print(f"[cyan]profiling {p.name}[/] ({p.detail}) …")
            if p.setup is not None:
                p.setup()
            kernels = alloy.visualize(
                p.run, str(path), f"{model} {p.label}", plans=p.plans, compile_ctx=p.compile_ctx
            )
            _print_kernel_summary(p.label, kernels)
            written.append(path)

    # Print a file:// URL on its own line, then open it directly: many terminals
    # route both bare paths and file:// links to the editor, so a click won't
    # reach the browser; webbrowser.open hands the file to the default browser.
    console.print("[green]wrote[/]", soft_wrap=True)
    for path in written:
        console.print(path.as_uri(), soft_wrap=True, highlight=False)
        if open_browser:
            webbrowser.open(path.as_uri())
