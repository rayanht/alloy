"""`alloy compile <model>` — pre-compile the dispatch plan for faster TTFT."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

console = Console()


def compile_(
    model: Annotated[str, typer.Argument(help="model name")],
) -> None:
    """Pre-compile a model's dispatch plan and cache it under ~/.alloy/cache/."""
    console.print(
        f"[yellow]compile is not yet implemented in the CLI.[/] "
        f"Args captured: model={model}. "
        f"Plan caching happens lazily on first dispatch today."
    )
    raise typer.Exit(1)
