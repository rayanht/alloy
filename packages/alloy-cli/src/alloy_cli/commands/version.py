"""`alloy version` — print CLI version, git sha, build date."""

from __future__ import annotations

import platform
import sys

import typer
from rich.console import Console

from alloy_cli.version import __version__

console = Console()


def version() -> None:
    """Print Alloy version, build, and platform info."""
    console.print(f"[bold]alloy[/bold] {__version__}")
    console.print(f"  python: {sys.version.split()[0]}")
    console.print(f"  platform: {platform.platform()}")
    console.print(f"  arch: {platform.machine()}")
    raise typer.Exit(0)
