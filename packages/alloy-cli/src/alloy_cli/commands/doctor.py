"""`alloy doctor` — diagnostics: platform, server reachability, loaded models."""

from __future__ import annotations

import platform

import typer
from rich.console import Console
from rich.table import Table

from alloy_cli.client import ServerClient

console = Console()


def doctor() -> None:
    """Run diagnostics. Exit non-zero if any check fails."""
    rows: list[tuple[str, str, str]] = []
    failed = False

    is_macos = platform.system() == "Darwin"
    rows.append(("platform", "ok" if is_macos else "FAIL", platform.platform()))
    if not is_macos:
        failed = True

    is_arm = platform.machine() == "arm64"
    rows.append(("architecture", "ok" if is_arm else "warn", platform.machine()))

    client = ServerClient()
    health = client.healthz()
    rows.append((
        f"server @ {client.base_url}",
        "ok" if health.reachable else "down",
        health.version or "(unreachable — start with `alloy serve -m <model>`)",
    ))
    if health.reachable:
        rows.append((
            "served model",
            "ok" if health.model else "FAIL",
            f"{health.model} ({health.kind})" if health.model else "none",
        ))

    table = Table(show_header=True, header_style="bold")
    table.add_column("check")
    table.add_column("status")
    table.add_column("detail")
    for row in rows:
        status = row[1]
        style = {"ok": "green", "FAIL": "red", "warn": "yellow"}.get(status, "")
        table.add_row(row[0], f"[{style}]{status}[/]" if style else status, row[2])
    console.print(table)

    raise typer.Exit(1 if failed else 0)
