"""Cleanup rewrite passes for Alloy torch FX graphs."""

from __future__ import annotations

import operator

import torch
import torch.fx


_TO_TARGETS = frozenset((torch.ops.aten._to_copy.default, torch.ops.aten.to.dtype))
_VIEW_OPS = frozenset(
    (
        torch.ops.aten.view.default,
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.reshape.default,
    )
)
_DTYPE_BYTES: dict[torch.dtype, int] = {
    torch.float32: 4,
    torch.int32: 4,
    torch.uint32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.int16: 2,
    torch.uint16: 2,
    torch.int8: 1,
    torch.uint8: 1,
    torch.int64: 8,
    torch.uint64: 8,
}


def topological_sort(graph: torch.fx.Graph) -> None:
    """Fix topological order after FX rewrites."""
    for _ in range(len(list(graph.nodes))):
        moved = False
        node_pos: dict[torch.fx.Node, int] = {node: i for i, node in enumerate(graph.nodes)}
        for node in list(graph.nodes):
            if node.op in ("placeholder", "get_attr", "output"):
                continue
            my_pos = node_pos[node]
            max_dep_pos = -1
            max_dep_node: torch.fx.Node | None = None
            for dep in node.all_input_nodes:
                dep_pos = node_pos.get(dep, -1)
                if dep_pos > max_dep_pos:
                    max_dep_pos = dep_pos
                    max_dep_node = dep
            if max_dep_pos > my_pos and max_dep_node is not None:
                max_dep_node.append(node)
                moved = True
                break
        if not moved:
            break


def rewrite_fold_identities(graph: torch.fx.Graph) -> int:
    """Fold arithmetic identity ops: mul(x, 1), div(x, 1), add(x, 0), sub(x, 0)."""
    identity_values = {
        torch.ops.aten.mul.Tensor: (1, 1.0),
        torch.ops.aten.mul.Scalar: (1, 1.0),
        torch.ops.aten.div.Tensor: (1, 1.0),
        torch.ops.aten.div.Scalar: (1, 1.0),
        torch.ops.aten.add.Tensor: (0, 0.0),
        torch.ops.aten.add.Scalar: (0, 0.0),
        torch.ops.aten.sub.Tensor: (0, 0.0),
        torch.ops.aten.sub.Scalar: (0, 0.0),
        operator.mul: (1, 1.0),
        operator.add: (0, 0.0),
    }
    non_commutative_targets = frozenset(
        (
            torch.ops.aten.div.Tensor,
            torch.ops.aten.div.Scalar,
            torch.ops.aten.sub.Tensor,
            torch.ops.aten.sub.Scalar,
        )
    )

    count = 0
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target not in identity_values:
            continue
        if len(node.args) != 2:
            continue
        lhs, rhs = node.args
        identity = identity_values[node.target]
        if isinstance(rhs, (int, float)) and rhs in identity and isinstance(lhs, torch.fx.Node):
            node.replace_all_uses_with(lhs)
            graph.erase_node(node)
            count += 1
        elif (
            isinstance(lhs, (int, float))
            and lhs in identity
            and node.target not in non_commutative_targets
            and isinstance(rhs, torch.fx.Node)
        ):
            node.replace_all_uses_with(rhs)
            graph.erase_node(node)
            count += 1
    return count


def rewrite_strip_f32_upcasts(graph: torch.fx.Graph) -> int:
    """Strip _to_copy(f16->f32) and same-dtype casts that add dispatches."""
    count = 0
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target not in _TO_TARGETS:
            continue
        if len(node.args) < 1:
            continue
        input_arg = node.args[0]
        if not isinstance(input_arg, torch.fx.Node):
            continue
        input_val = input_arg.meta.get("val")
        if input_val is None or not hasattr(input_val, "dtype"):
            continue
        target_dtype = node.kwargs.get("dtype")
        if not isinstance(target_dtype, torch.dtype):
            continue
        if (input_val.dtype == torch.float16 and target_dtype == torch.float32) or (
            input_val.dtype == target_dtype
        ):
            node.replace_all_uses_with(input_arg)
            graph.erase_node(node)
            count += 1
    return count


def rewrite_strip_lossless_roundtrip_cast(graph: torch.fx.Graph) -> int:
    """Strip `_to_copy(_to_copy(x, wider), x.dtype)` when the roundtrip is lossless."""
    count = 0
    for outer in list(graph.nodes):
        if outer.op != "call_function" or outer.target not in _TO_TARGETS:
            continue
        target_dtype = outer.kwargs.get("dtype")
        if not isinstance(target_dtype, torch.dtype) or len(outer.args) < 1:
            continue
        inner = outer.args[0]
        if not isinstance(inner, torch.fx.Node):
            continue
        if inner.op != "call_function" or inner.target not in _TO_TARGETS:
            continue
        intermediate_dtype = inner.kwargs.get("dtype")
        if not isinstance(intermediate_dtype, torch.dtype) or len(inner.args) < 1:
            continue
        source = inner.args[0]
        if not isinstance(source, torch.fx.Node):
            continue
        source_val = source.meta.get("val")
        if source_val is None or not hasattr(source_val, "dtype"):
            continue
        source_dtype = source_val.dtype
        if source_dtype != target_dtype:
            continue
        if _DTYPE_BYTES.get(intermediate_dtype, 0) <= _DTYPE_BYTES.get(source_dtype, 0):
            continue
        outer.replace_all_uses_with(source)
        graph.erase_node(outer)
        count += 1
    return count


def rewrite_simplify_views(graph: torch.fx.Graph) -> int:
    """Simplify alias and redundant view chains using FX shape metadata."""
    count = 0
    for node in list(graph.nodes):
        if node.op != "call_function":
            continue

        if node.target == torch.ops.aten.alias.default:
            if len(node.args) < 1:
                continue
            input_arg = node.args[0]
            if not isinstance(input_arg, torch.fx.Node):
                continue
            node.replace_all_uses_with(input_arg)
            graph.erase_node(node)
            count += 1
            continue

        if node.target not in _VIEW_OPS:
            continue
        if len(node.args) < 2:
            continue
        src = node.args[0]
        if not isinstance(src, torch.fx.Node):
            continue
        if src.target not in _VIEW_OPS or len(src.users) != 1:
            continue
        if len(src.args) < 1:
            continue
        root = src.args[0]
        if not isinstance(root, torch.fx.Node):
            continue
        root_meta = root.meta.get("val")
        if root_meta is None:
            continue
        if not hasattr(root_meta, "is_contiguous") or not root_meta.is_contiguous():
            continue
        node.args = (root, node.args[1])
        graph.erase_node(src)
        count += 1

    return count
