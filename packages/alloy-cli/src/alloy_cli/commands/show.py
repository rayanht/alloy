"""`alloy show <model>` — metadata for one model, read straight from disk
(Ollama store / HuggingFace cache / local path). GGUF and MLX."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import gguf
import typer
from rich.console import Console
from rich.table import Table

from alloy_server.discover import hf_hub_dir, ollama_models_dir
from alloy_server.gguf import ResolvedGGUF
from alloy_server.mlx import ResolvedMLX
from alloy_server.models import resolve_model

console = Console()


def show(
    model: Annotated[
        str,
        typer.Argument(
            help="model ref: ./model.gguf, an MLX repo/dir, Org/Repo:Q4_K_M, or an Ollama name",
        ),
    ],
) -> None:
    """Resolve a model ref and print its on-disk metadata."""
    try:
        resolved = resolve_model(model)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc

    if isinstance(resolved, ResolvedMLX):
        _show_mlx(resolved)
    else:
        _show_gguf(resolved)


def _show_gguf(resolved: ResolvedGGUF) -> None:
    reader = gguf.GGUFReader(str(resolved.path))
    fields = reader.fields
    arch_field = fields.get("general.architecture")
    arch = str(arch_field.contents()) if arch_field is not None else "?"

    table = Table(show_header=False, box=None)
    table.add_column(style="bold")
    table.add_column()
    table.add_row("ref", resolved.ref)
    table.add_row("path", str(resolved.path))
    table.add_row("source", _source_of(resolved.path))
    table.add_row("size", _human_size(_file_size(resolved.path)))
    table.add_row("architecture", arch)

    ctx = fields.get(f"{arch}.context_length")
    if ctx is not None:
        table.add_row("context_length", str(ctx.contents()))

    # The remaining general.* scalars (name, size label, quant version, …);
    # skip arrays (tokenizer vocab etc.) which aren't useful here.
    for key in sorted(fields):
        if not key.startswith("general.") or key == "general.architecture":
            continue
        value = fields[key].contents()
        if isinstance(value, (str, int, float)):
            table.add_row(key[len("general."):], str(value))

    console.print(table)


def _show_mlx(resolved: ResolvedMLX) -> None:
    config = resolved.config
    size = sum(_file_size(p) for p in resolved.safetensors)
    table = Table(show_header=False, box=None)
    table.add_column(style="bold")
    table.add_column()
    table.add_row("ref", resolved.ref)
    table.add_row("path", str(resolved.model_dir))
    table.add_row("source", _source_of(resolved.model_dir))
    table.add_row("size", _human_size(size))
    table.add_row("architecture", resolved.architecture())
    quant = config.get("quantization") or config.get("quantization_config")
    if isinstance(quant, dict):
        table.add_row("quantization", f"{quant.get('bits')}-bit group {quant.get('group_size')}")
    for key in ("max_position_embeddings", "hidden_size", "num_hidden_layers", "vocab_size"):
        if key in config:
            table.add_row(key, str(config[key]))
    console.print(table)


def _source_of(path: Path) -> str:
    p = str(path)
    if p.startswith(str(ollama_models_dir())):
        return "ollama"
    if p.startswith(str(hf_hub_dir())):
        return "huggingface"
    return "local"


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _human_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    value = float(n)
    for unit in ("KB", "MB", "GB", "TB"):
        value /= 1024
        if value < 1024:
            return f"{value:.2f} {unit}"
    return f"{value:.2f} PB"
