"""Alloy CLI."""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Callable
from typing import TypeAlias, cast

ServerMain: TypeAlias = Callable[[tuple[str, ...] | None], int]


def main(argv: tuple[str, ...] | None = None) -> int:
    args = tuple(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "serve":
        return _run_server_module(args[1:])

    parser = argparse.ArgumentParser(
        prog="alloy",
        description="Alloy: GPU kernels on Apple Silicon",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("serve", help="Start the Alloy OpenAI-compatible server")

    namespace = parser.parse_args(args)
    if namespace.command is None:
        parser.print_help()
        return 1
    return 0


def _run_server_module(argv: tuple[str, ...]) -> int:
    try:
        module = importlib.import_module("alloy_server")
    except ModuleNotFoundError as error:
        if error.name in {"alloy_torch", "alloy_server"}:
            print("alloy serve requires the alloy-server package", file=sys.stderr)
            return 1
        raise

    server_main = cast(ServerMain, module.main)
    try:
        return server_main(argv)
    except SystemExit as error:
        code = error.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        print(code, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
