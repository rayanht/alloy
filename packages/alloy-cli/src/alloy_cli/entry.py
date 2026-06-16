"""Console-script entry point.

The lean `pip install alloy-kit` ships this module (kernel library only — no
torch, typer, etc.). Importing the full CLI then fails on a missing `serve`-extra
dependency; we turn that into one clear line instead of a stray ImportError.
"""

from __future__ import annotations

import sys

# Dependencies that only arrive with the `serve` extra. A miss on any of these
# means the user installed the lean base and is trying to run the CLI/server.
_SERVE_DEPS = frozenset(
    {"typer", "rich", "questionary", "httpx", "torch", "transformers"}
)


def main() -> None:
    try:
        from alloy_cli.cli import app
    except ModuleNotFoundError as exc:
        if exc.name in _SERVE_DEPS:
            sys.stderr.write(
                "The alloy CLI and server need the `serve` extra:\n\n"
                "    pip install 'alloy-kit[serve]'\n\n"
            )
            raise SystemExit(1) from exc
        raise
    app()
