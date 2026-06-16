"""Shared FX graph helpers for rewrite passes."""

from __future__ import annotations

import operator

import torch
import torch.fx


TRANSPARENT_TARGETS = frozenset(
    (
        torch.ops.aten.view.default,
        torch.ops.aten.clone.default,
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.reshape.default,
    )
)


def find_single_consumer(node: torch.fx.Node) -> torch.fx.Node | None:
    """Return the sole non-getitem tensor consumer of node."""
    users = [
        user
        for user in node.users
        if user.op in ("call_function", "call_method") and user.target is not operator.getitem
    ]
    return users[0] if len(users) == 1 else None


def collect_transparent(node: torch.fx.node.Argument) -> tuple[torch.fx.Node, list[torch.fx.Node]]:
    """Walk backward through view/clone/reshape to find the underlying node."""
    if not isinstance(node, torch.fx.Node):
        raise TypeError(f"expected FX Node, got {type(node).__name__}")

    skipped: list[torch.fx.Node] = []
    current = node
    while current.op == "call_function" and current.target in TRANSPARENT_TARGETS:
        skipped.append(current)
        if not current.args:
            break
        next_node = current.args[0]
        if not isinstance(next_node, torch.fx.Node):
            break
        current = next_node
    return current, skipped
