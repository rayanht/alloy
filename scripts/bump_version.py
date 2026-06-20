#!/usr/bin/env python3
"""Bump the Alloy version everywhere in one shot.

The version is declared in several places (`version.py`, the workspace packages,
the packaging pyproject). This is the single entry point that keeps them in
lockstep — never edit them by hand.

    python scripts/bump_version.py 0.2.0     # set everywhere
    python scripts/bump_version.py --show     # print the current version

Release flow: bump, commit, then tag `vX.Y.Z`. The publish workflow verifies the
tag matches `packaging/pyproject.toml`.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# `version.py` is the canonical display source; the rest mirror it. Each
# pattern must match exactly one version declaration in its file.
CANONICAL = "packages/alloy-cli/src/alloy_cli/version.py"
TARGETS: list[tuple[str, str]] = [
    (CANONICAL, r'(__version__ = ")[^"]+(")'),
    ("packages/alloy-cli/pyproject.toml", r'(?m)^(version = ")[^"]+(")'),
    ("packages/alloy-metal/pyproject.toml", r'(?m)^(version = ")[^"]+(")'),
    ("packages/alloy-torch/pyproject.toml", r'(?m)^(version = ")[^"]+(")'),
    ("packages/alloy-server/pyproject.toml", r'(?m)^(version = ")[^"]+(")'),
    ("packages/alloy-mlx/pyproject.toml", r'(?m)^(version = ")[^"]+(")'),
    ("packaging/pyproject.toml", r'(?m)^(version = ")[^"]+(")'),
    ("pyproject.toml", r'(?m)^(version = ")[^"]+(")'),
]


def current() -> str:
    text = (ROOT / CANONICAL).read_text()
    match = re.search(r'__version__ = "([^"]+)"', text)
    if match is None:
        sys.exit(f"error: no __version__ in {CANONICAL}")
    return match.group(1)


def bump(new: str) -> None:
    for rel, pattern in TARGETS:
        path = ROOT / rel
        text = path.read_text()
        updated, n = re.subn(pattern, rf"\g<1>{new}\g<2>", text)
        if n != 1:
            sys.exit(f"error: {rel}: expected exactly 1 version match, found {n}")
        path.write_text(updated)
        print(f"  {rel}")
    # uv.lock pins the workspace package versions; resync it so the lock never
    # lags the bumped pyproject.tomls.
    subprocess.run(["uv", "lock"], cwd=ROOT, check=True)
    print("  uv.lock")


def main() -> None:
    args = sys.argv[1:]
    if args == ["--show"]:
        print(current())
        return
    if len(args) != 1:
        sys.exit("usage: bump_version.py X.Y.Z | --show")
    new = args[0]
    if not re.fullmatch(r"\d+\.\d+\.\d+([abrc]\d+|\.dev\d+|rc\d+)?", new):
        sys.exit(f"error: invalid version {new!r}")
    print(f"bumping {current()} -> {new}")
    bump(new)


if __name__ == "__main__":
    main()
