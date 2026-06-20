"""Opt-in per-kernel compile observer.

When an observer is installed, the dispatch and fusion compile paths call
`notify_compiled` with the resolved (name, constexprs, shapes, msl, tile_func)
for every kernel they compile — letting `alloy inspect` capture the exact MSL/IR
a real model forward executes. None by default.
"""

from __future__ import annotations

from collections.abc import Callable

_compile_observer: Callable[..., None] | None = None


def set_compile_observer(fn: Callable[..., None] | None) -> None:
    """Install (or clear with None) the per-kernel compile observer."""
    global _compile_observer
    _compile_observer = fn


def notify_compiled(name, constexprs, shapes, msl, tile_func) -> None:
    """Forward a compiled kernel to the observer, if one is installed."""
    if _compile_observer is not None:
        _compile_observer(name, constexprs, shapes, msl, tile_func)
