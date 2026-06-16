"""Activation rewrite passes for Alloy torch FX graphs."""

from __future__ import annotations

from dataclasses import dataclass
import operator

import torch
import torch.fx

from alloy_torch.rewrites.graph_utils import find_single_consumer


_ADD_TARGETS = {torch.ops.aten.add.Tensor, operator.add}
_MUL_TARGETS = {torch.ops.aten.mul.Tensor, operator.mul}
_TANH_TARGETS = {torch.ops.aten.tanh.default}


@dataclass(frozen=True)
class GeluTanhMatch:
    input_node: torch.fx.Node
    output_node: torch.fx.Node
    consumed: tuple[torch.fx.Node, ...]


def _arg_node(node: torch.fx.Node, index: int) -> torch.fx.Node | None:
    if len(node.args) <= index:
        return None
    arg = node.args[index]
    return arg if isinstance(arg, torch.fx.Node) else None


def _find_inner_add(coeff_mul: torch.fx.Node) -> torch.fx.Node | None:
    lhs = _arg_node(coeff_mul, 0)
    rhs = _arg_node(coeff_mul, 1)
    if lhs is not None and lhs.target in _ADD_TARGETS:
        return lhs
    if rhs is not None and rhs.target in _ADD_TARGETS:
        return rhs
    return None


def _find_gelu_input(half_x: torch.fx.Node) -> torch.fx.Node | None:
    lhs = _arg_node(half_x, 0)
    rhs = _arg_node(half_x, 1)
    if lhs is not None and lhs.op != "get_attr":
        return lhs
    if rhs is not None and rhs.op != "get_attr":
        return rhs
    return None


def _collect_scaled_pow3(scaled: torch.fx.Node, consumed: set[torch.fx.Node]) -> None:
    consumed.add(scaled)
    for arg in scaled.args:
        if not isinstance(arg, torch.fx.Node) or arg.target not in _MUL_TARGETS:
            continue
        consumed.add(arg)
        for nested_arg in arg.args:
            if isinstance(nested_arg, torch.fx.Node) and nested_arg.target in _MUL_TARGETS:
                consumed.add(nested_arg)


def _find_gelu_tanh_chain(tanh_node: torch.fx.Node) -> GeluTanhMatch | None:
    if tanh_node.op != "call_function" or tanh_node.target not in _TANH_TARGETS:
        return None

    tanh_input = _arg_node(tanh_node, 0)
    if tanh_input is None or tanh_input.target not in _MUL_TARGETS:
        return None

    add_one = find_single_consumer(tanh_node)
    if add_one is None or add_one.target not in _ADD_TARGETS or len(add_one.args) < 2:
        return None
    other_arg = add_one.args[1] if add_one.args[0] is tanh_node else add_one.args[0]
    if isinstance(other_arg, int | float) and other_arg != 1.0:
        return None

    mul_final = find_single_consumer(add_one)
    if mul_final is None or mul_final.target not in _MUL_TARGETS or len(mul_final.args) < 2:
        return None

    half_x = mul_final.args[0] if mul_final.args[1] is add_one else mul_final.args[1]
    if not isinstance(half_x, torch.fx.Node) or half_x.target not in _MUL_TARGETS:
        return None
    x_candidate = _find_gelu_input(half_x)
    if x_candidate is None:
        return None

    inner_add = _find_inner_add(tanh_input)
    if inner_add is None or len(inner_add.args) < 2:
        return None
    lhs, rhs = inner_add.args[0], inner_add.args[1]
    if lhs is not x_candidate and rhs is not x_candidate:
        return None

    consumed: set[torch.fx.Node] = {
        tanh_node,
        add_one,
        mul_final,
        half_x,
        tanh_input,
        inner_add,
    }
    scaled = rhs if lhs is x_candidate else lhs
    if isinstance(scaled, torch.fx.Node):
        _collect_scaled_pow3(scaled, consumed)
    if isinstance(other_arg, torch.fx.Node):
        consumed.add(other_arg)

    return GeluTanhMatch(
        input_node=x_candidate,
        output_node=mul_final,
        consumed=tuple(consumed),
    )


def rewrite_gelu_tanh(graph: torch.fx.Graph) -> int:
    """Replace decomposed GELU tanh chains with aten.gelu nodes."""
    count = 0
    for node in list(graph.nodes):
        match = _find_gelu_tanh_chain(node)
        if match is None:
            continue

        with graph.inserting_after(match.output_node):
            new_node = graph.call_function(
                torch.ops.aten.gelu.default,
                args=(match.input_node,),
                kwargs={"approximate": "tanh"},
            )
        match.output_node.replace_all_uses_with(new_node)

        for dead in reversed(match.consumed):
            if len(dead.users) == 0:
                graph.erase_node(dead)
        count += 1

    return count
