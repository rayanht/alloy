"""`alloy inspect <model> <kernel> [--level msl|ir]` — dump the exact MSL (or
tile IR) a real model forward executes for a given kernel.

Unlike `al.inspect(kernel, **constexprs)` (which re-derives from constexprs you
supply), this runs the model's real forward passes (prefill + decode for causal
models; the embedder's per-shape encoder forward for embedding models), observes
every kernel the model actually compiles with its production-resolved
constexprs, and writes the captured source to a .log file — so you see exactly
what runs in the server. For embedders, --batch/--seq pick the shape.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated

import torch
import typer
from rich.console import Console

from alloy._compiler.tile_ir import dump_tile_ir
from alloy._dispatch.observe import set_compile_observer
from alloy._runtime import _metal_ext
from alloy_cli.capture import _DEFAULT_DEPTH, build_capture

console = Console()


def _print_pso(name: str, variants: dict[str, dict]) -> None:
    """Compile each captured MSL variant and print its pipeline stats.

    `variants` is keyed by MSL source (so the source is available even under
    --level ir). maxTotalThreadsPerThreadgroup is a register-pressure proxy
    (1024 = registers light; lower = register-limited). The shmem-limited
    residency (pool ÷ static threadgroup memory) is the occupancy ceiling — e.g.
    a 17 KB tile against a 32 KB pool allows only one resident threadgroup.
    """
    pool = int(_metal_ext.device_info()["max_threadgroup_memory_length"])
    nvar = len(variants)
    console.print(
        f"[bold]{name}[/] — {nvar} variant{'s' if nvar != 1 else ''}, "
        f"threadgroup-memory pool {pool / 1024:.0f} KB"
    )
    for i, (msl_text, entry) in enumerate(variants.items()):
        if nvar > 1:
            console.print(f"  [cyan]variant {i + 1}/{nvar}[/] {entry['constexprs']}", soft_wrap=True)
        m = re.search(r"kernel\s+void\s+(\w+)", msl_text)
        fn_name = m.group(1) if m else name
        try:
            h = _metal_ext.compile_msl(msl_text, fn_name)
        except Exception as exc:  # noqa: BLE001 — surface the Metal compile/link error
            console.print(f"    [red]compile error[/] {exc}", soft_wrap=True)
            continue
        mtpt = _metal_ext.pipeline_max_threads(h)
        tew = _metal_ext.pipeline_thread_width(h)
        shmem = _metal_ext.pipeline_static_threadgroup_memory(h)
        reg = "registers light" if mtpt >= 1024 else "register-limited (<1024)"
        console.print(f"    maxThreadsPerTG = {mtpt:>4}  ({reg})", highlight=False)
        console.print(f"    threadExecWidth = {tew:>4}", highlight=False)
        if shmem > 0:
            residents = pool // shmem if shmem else 0
            plural = "s" if residents != 1 else ""
            console.print(
                f"    threadgroupMem  = {shmem:>5} B ({shmem / 1024:.2f} KB)"
                f"  → {residents} resident TG{plural} by shmem",
                highlight=False,
            )
        else:
            console.print(
                "    threadgroupMem  =     0 B (none static, or set dynamically at encode)",
                highlight=False,
            )


def inspect(
    model: Annotated[str, typer.Argument(help="model name, e.g. qwen2.5:3b")],
    kernel: Annotated[str, typer.Argument(help="kernel name, e.g. dot_q4_k_silu_v2")],
    depth: Annotated[
        int,
        typer.Option(
            "--depth",
            help="Prefill cache depth to capture at (synthetic prompt, chunked like "
                 "production); picks the warm-prefill kernel variant at that offset. "
                 "Matches `alloy bench --depths`. Default 4096.",
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
    level: Annotated[
        str,
        typer.Option(help="'msl' (default) or 'ir' (tile IR)"),
    ] = "msl",
    pso: Annotated[
        bool,
        typer.Option(
            "--pso",
            help="report compiled-pipeline statistics: max threads per "
            "threadgroup (a register-pressure proxy), SIMD width, and threadgroup "
            "memory with the residency it permits against the device pool.",
        ),
    ] = False,
    out_dir: Annotated[
        Path,
        typer.Option(help="directory to write the .log file into"),
    ] = Path("."),
) -> None:
    """Dump the real MSL/IR a model forward executes for a kernel (or its PSO stats)."""
    if level not in ("msl", "ir"):
        raise typer.BadParameter("--level must be 'msl' or 'ir'")

    console.print(f"[cyan]loading[/] {model} …")
    try:
        cap = build_capture(model, depth=depth, batch=batch, seq=seq)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    # name -> {msl: {constexprs, source}} — dedup identical variants (a kernel
    # dispatched at several shapes, e.g. dot_q4_k for q/k/v/o, keeps each).
    captured: dict[str, dict[str, dict]] = {}

    def observer(name: str, constexprs: dict, shapes, msl: str, func: object) -> None:
        variants = captured.setdefault(name, {})
        if msl not in variants:
            variants[msl] = {
                "constexprs": constexprs,
                "source": dump_tile_ir(func) if level == "ir" else msl,
            }

    set_compile_observer(observer)
    try:
        with torch.inference_mode():
            passes_desc = " + ".join(p.detail for p in cap.passes)
            console.print(f"[cyan]forward[/] {passes_desc} …")
            for p in cap.passes:
                if p.setup is not None:
                    p.setup()
                p.run()
    finally:
        set_compile_observer(None)

    if kernel in captured:
        name = kernel
    else:
        subs = sorted(n for n in captured if kernel.lower() in n.lower())
        if len(subs) == 1:
            name = subs[0]
        elif subs:
            raise typer.BadParameter(f"{kernel!r} matches multiple kernels: {', '.join(subs)}")
        else:
            raise typer.BadParameter(
                f"{kernel!r} was not dispatched by {model}. "
                f"Kernels seen: {', '.join(sorted(captured))}"
            )

    variants = captured[name]

    if pso:
        _print_pso(name, variants)
        return

    blocks: list[str] = []
    for i, entry in enumerate(variants.values()):
        if len(variants) > 1:
            blocks.append(f"// ===== variant {i + 1}/{len(variants)} — {entry['constexprs']} =====")
        blocks.append(entry["source"])

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = model.replace(":", "_").replace("/", "_")
    out_path = out_dir / f"{safe}.{name}.{level}.log"
    out_path.write_text("\n\n".join(blocks) + "\n")

    nvar = len(variants)
    console.print(
        f"[green]wrote[/] {level} for {name} ({nvar} variant{'s' if nvar != 1 else ''})",
        soft_wrap=True,
    )
    console.print(str(out_path), soft_wrap=True, highlight=False)
