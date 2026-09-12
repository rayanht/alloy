"""`alloy run`: one prompt through the DeepSeek-V4.1 streaming engine."""

from __future__ import annotations

import sys
import time

import numpy as np
import typer
from rich.console import Console

from alloy_server.models import resolve_model
from alloy_server.models.deepseek_v41.engine import DeepseekV41Engine, GenerationStats
from alloy_server.models.deepseek_v41.handler import build_deepseek41_tokenizer

console = Console(stderr=True)


def run(
    model: str = typer.Argument(..., metavar="MODEL", help="Local path to the first GGUF shard, or a model ref."),
    prompt: str = typer.Option(..., "-p", "--prompt", help="The prompt (a user turn unless --raw)."),
    max_new_tokens: int = typer.Option(64, "-n", "--max-new-tokens"),
    raw: bool = typer.Option(False, "--raw", help="Feed the prompt verbatim instead of through the chat template."),
    temperature: float = typer.Option(0.0, "--temperature"),
    context: int = typer.Option(131072, "-c", "--ctx", help="Cache length to allocate (<= the model's native context)."),
    chunk: int = typer.Option(2048, "--chunk", help="Prefill chunk size."),
    arena_gb: float | None = typer.Option(
        None, "--arena-gb", help="Expert arena size; default derives from the GPU working set.",
    ),
    layers: int | None = typer.Option(None, "--layers", help="Truncate to the first N layers (partial-download bring-up)."),
) -> None:
    resolved = resolve_model(model)
    arch = resolved.architecture()
    if arch != "deepseek41":
        raise typer.BadParameter(f"`alloy run` drives the deepseek41 engine; {model} is {arch} (use `alloy serve`).")
    t0 = time.perf_counter()
    engine = DeepseekV41Engine.from_path(
        resolved.path,
        allow_missing=layers is not None,
        max_seq_len=context,
        chunk_size=chunk,
        arena_bytes=int(arena_gb * (1 << 30)) if arena_gb else None,
        n_layers=layers,
    )
    tokenizer = build_deepseek41_tokenizer(engine.split)
    console.print(f"[dim]loaded in {time.perf_counter() - t0:.1f}s[/dim]")
    if raw:
        text = prompt
    else:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
        )
    ids = np.asarray(tokenizer.encode(text, add_special_tokens=raw), dtype=np.int64)
    eos = (int(tokenizer.eos_token_id),) if tokenizer.eos_token_id is not None else ()
    stats = GenerationStats()
    out: list[int] = []
    for tok in engine.generate(ids, max_new_tokens, eos_ids=eos, temperature=temperature, stats=stats):
        out.append(tok)
        sys.stdout.write(tokenizer.decode(out[-1:]))
        sys.stdout.flush()
    sys.stdout.write("\n")
    tpot = stats.decode_s / max(stats.decode_tokens, 1)
    console.print(
        f"[dim]prompt {stats.prompt_tokens} tok in {stats.prefill_s:.1f}s "
        f"({stats.prompt_tokens / max(stats.prefill_s, 1e-9):.0f} tok/s) · "
        f"{stats.decode_tokens} tok decoded, {tpot * 1000:.0f} ms/tok · "
        f"experts: {stats.expert_hits} hits / {stats.expert_misses} misses, "
        f"{stats.expert_bytes / (1 << 30):.1f} GiB read[/dim]"
    )
