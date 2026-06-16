"""INT4 dequant rewrite passes for Alloy torch FX graphs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import operator
from typing import cast
from typing import TypeGuard

import alloy_torch.custom_ops  # noqa: F401
import torch
import torch.fx

from alloy_torch.rewrites.graph_utils import TRANSPARENT_TARGETS
from alloy_torch.rewrites.graph_utils import collect_transparent


_MM_TARGETS: frozenset[torch.fx.node.Target] = frozenset((torch.ops.aten.mm.default,))
_MUL_TARGETS: frozenset[torch.fx.node.Target] = frozenset((torch.ops.aten.mul.Tensor, operator.mul))
_SUB_TARGETS: frozenset[torch.fx.node.Target] = frozenset((torch.ops.aten.sub.Tensor, operator.sub))
_PERMUTE_TARGETS: frozenset[torch.fx.node.Target] = frozenset((torch.ops.aten.permute.default,))
_VIEW_TARGETS: frozenset[torch.fx.node.Target] = frozenset(
    (torch.ops.aten.view.default, torch.ops.aten.reshape.default)
)
_UNSQUEEZE_TARGETS: frozenset[torch.fx.node.Target] = frozenset((torch.ops.aten.unsqueeze.default,))
_CAT_TARGETS: frozenset[torch.fx.node.Target] = frozenset((torch.ops.aten.cat.default,))
_TO_COPY_TARGETS: frozenset[torch.fx.node.Target] = frozenset((torch.ops.aten._to_copy.default,))
_FLOOR_TARGETS: frozenset[torch.fx.node.Target] = frozenset((torch.ops.aten.floor.default,))
_DIV_TARGETS: frozenset[torch.fx.node.Target] = frozenset((torch.ops.aten.div.Tensor,))
_SIGMOID_TARGETS: frozenset[torch.fx.node.Target] = frozenset((torch.ops.aten.sigmoid.default,))
_DQ_TARGETS: frozenset[torch.fx.node.Target] = frozenset((torch.ops.alloy.dequant_mm.default,))


@dataclass(frozen=True)
class DequantMmMatch:
    activations: torch.fx.Node
    packed_weights: torch.fx.node.Argument
    scales: torch.fx.node.Argument
    zeros: torch.fx.node.Argument
    group_size: torch.fx.node.Argument
    mm_node: torch.fx.Node
    consumed: tuple[torch.fx.Node, ...]


def _arg(node: torch.fx.Node, index: int) -> torch.fx.node.Argument | None:
    if len(node.args) <= index:
        return None
    return node.args[index]


def _arg_node(node: torch.fx.Node, index: int) -> torch.fx.Node | None:
    arg = _arg(node, index)
    return arg if isinstance(arg, torch.fx.Node) else None


def _is_target(
    node: torch.fx.node.Argument | None, targets: frozenset[torch.fx.node.Target]
) -> TypeGuard[torch.fx.Node]:
    return isinstance(node, torch.fx.Node) and node.target in targets


def _is_permute_10(node: torch.fx.Node) -> bool:
    dims = _arg(node, 1)
    return isinstance(dims, (list, tuple)) and tuple(dims) == (1, 0)


def _find_dequant_mm_chain(mm_node: torch.fx.Node) -> DequantMmMatch | None:
    """Walk backward from an mm node to find the full INT4 dequant pattern."""
    if not _is_target(mm_node, _MM_TARGETS):
        return None

    activations = _arg_node(mm_node, 0)
    rhs = _arg_node(mm_node, 1)
    if activations is None or rhs is None:
        return None

    if not _is_target(rhs, _PERMUTE_TARGETS) or not _is_permute_10(rhs):
        return None
    consumed: list[torch.fx.Node] = [mm_node, rhs]

    view_flat = _arg_node(rhs, 0)
    if not _is_target(view_flat, _VIEW_TARGETS):
        return None
    consumed.append(view_flat)

    mul_dequant = _arg_node(view_flat, 0)
    if not _is_target(mul_dequant, _MUL_TARGETS):
        return None
    consumed.append(mul_dequant)

    sub_dequant = _arg_node(mul_dequant, 0)
    scales_unsq = _arg_node(mul_dequant, 1)
    if not _is_target(sub_dequant, _SUB_TARGETS) or not _is_target(scales_unsq, _UNSQUEEZE_TARGETS):
        return None
    consumed.extend([sub_dequant, scales_unsq])
    scales_node = _arg(scales_unsq, 0)
    if scales_node is None:
        return None

    view_grouped = _arg_node(sub_dequant, 0)
    zeros_unsq = _arg_node(sub_dequant, 1)
    if not _is_target(zeros_unsq, _UNSQUEEZE_TARGETS):
        return None
    consumed.append(zeros_unsq)
    zeros_node = _arg(zeros_unsq, 0)
    if zeros_node is None:
        return None

    if not _is_target(view_grouped, _VIEW_TARGETS):
        return None
    consumed.append(view_grouped)
    grouped_shape = _arg(view_grouped, 1)
    if not isinstance(grouped_shape, (list, tuple)) or len(grouped_shape) != 3:
        return None
    group_size = grouped_shape[2]

    cat_flat = _arg_node(view_grouped, 0)
    if cat_flat is None:
        return None
    while _is_target(cat_flat, _VIEW_TARGETS):
        consumed.append(cat_flat)
        next_node = _arg_node(cat_flat, 0)
        if next_node is None:
            return None
        cat_flat = next_node

    if not _is_target(cat_flat, _CAT_TARGETS):
        return None
    consumed.append(cat_flat)

    cat_inputs = _arg(cat_flat, 0)
    if not isinstance(cat_inputs, (list, tuple)) or len(cat_inputs) != 2:
        return None
    lo_unsq = cat_inputs[0]
    hi_unsq = cat_inputs[1]
    if not _is_target(lo_unsq, _UNSQUEEZE_TARGETS) or not _is_target(hi_unsq, _UNSQUEEZE_TARGETS):
        return None
    if not isinstance(lo_unsq, torch.fx.Node) or not isinstance(hi_unsq, torch.fx.Node):
        return None
    consumed.extend([lo_unsq, hi_unsq])

    lo_node = _arg_node(lo_unsq, 0)
    hi_node = _arg_node(hi_unsq, 0)
    if lo_node is None or hi_node is None:
        return None

    if not _is_target(hi_node, _FLOOR_TARGETS):
        return None
    consumed.append(hi_node)
    hi_div = _arg_node(hi_node, 0)
    if not _is_target(hi_div, _DIV_TARGETS):
        return None
    consumed.append(hi_div)

    if not _is_target(lo_node, _SUB_TARGETS):
        return None
    consumed.append(lo_node)
    lo_mul = _arg_node(lo_node, 1)
    if not _is_target(lo_mul, _MUL_TARGETS):
        return None
    consumed.append(lo_mul)
    lo_floor = _arg_node(lo_mul, 0)
    if not _is_target(lo_floor, _FLOOR_TARGETS):
        return None
    consumed.append(lo_floor)
    lo_div = _arg_node(lo_floor, 0)
    if not _is_target(lo_div, _DIV_TARGETS):
        return None
    consumed.append(lo_div)

    to_copy_lo = _arg_node(lo_div, 0)
    to_copy_hi = _arg_node(hi_div, 0)
    to_copy_node = _arg_node(lo_node, 0)
    if to_copy_node is None:
        return None

    if _is_target(to_copy_node, _TO_COPY_TARGETS):
        consumed.append(to_copy_node)
        packed_weights = _arg(to_copy_node, 0)
    elif to_copy_node is to_copy_lo or to_copy_node is to_copy_hi:
        if not _is_target(to_copy_node, _TO_COPY_TARGETS):
            return None
        consumed.append(to_copy_node)
        packed_weights = _arg(to_copy_node, 0)
    else:
        return None
    if packed_weights is None:
        return None

    return DequantMmMatch(
        activations=activations,
        packed_weights=packed_weights,
        scales=scales_node,
        zeros=zeros_node,
        group_size=group_size,
        mm_node=mm_node,
        consumed=tuple(consumed),
    )


def rewrite_dequant_mm(graph: torch.fx.Graph) -> int:
    """Replace decomposed dequant-to-mm chains with alloy.dequant_mm."""
    count = 0
    for node in list(graph.nodes):
        match = _find_dequant_mm_chain(node)
        if match is None:
            continue

        with graph.inserting_after(match.mm_node):
            new_node = graph.call_function(
                torch.ops.alloy.dequant_mm.default,
                args=(
                    match.activations,
                    match.packed_weights,
                    match.scales,
                    match.zeros,
                    match.group_size,
                ),
            )
        match.mm_node.replace_all_uses_with(new_node)

        for dead in reversed(match.consumed):
            if len(dead.users) == 0:
                graph.erase_node(dead)
        count += 1

    return count


def rewrite_batched_dequant_mm(graph: torch.fx.Graph) -> int:
    """Replace groups of consecutive dequant_mm ops with the same activations."""
    count = 0
    nodes = list(graph.nodes)
    consumed_ids: set[int] = set()

    for i, node in enumerate(nodes):
        if id(node) in consumed_ids:
            continue
        if node.op != "call_function" or node.target != torch.ops.alloy.dequant_mm.default:
            continue

        activations = _arg_node(node, 0)
        group_size = _arg(node, 4)
        if activations is None or group_size is None:
            continue

        source, _ = collect_transparent(activations)

        group: list[torch.fx.Node] = [node]
        for cand in nodes[i + 1 :]:
            if cand.op != "call_function":
                continue
            if cand.target == torch.ops.alloy.dequant_mm.default:
                cand_activations = _arg_node(cand, 0)
                cand_group_size = _arg(cand, 4)
                if cand_activations is None or cand_group_size is None:
                    break
                cand_source, _ = collect_transparent(cand_activations)
                if cand_source is source and cand_group_size == group_size:
                    group.append(cand)
                    continue
                break
            if cand.target in TRANSPARENT_TARGETS or cand.target == torch.ops.aten.permute.default:
                continue
            break

        if len(group) < 2:
            continue

        lhs = _arg_node(group[0], 0)
        if lhs is None:
            continue
        packed_weights: list[torch.fx.node.Argument] = []
        scales_list: list[torch.fx.node.Argument] = []
        zeros_list: list[torch.fx.node.Argument] = []
        for group_node in group:
            packed_weight = _arg(group_node, 1)
            scales = _arg(group_node, 2)
            zeros = _arg(group_node, 3)
            if packed_weight is None or scales is None or zeros is None:
                break
            packed_weights.append(packed_weight)
            scales_list.append(scales)
            zeros_list.append(zeros)
        if len(packed_weights) != len(group):
            continue

        first_node = group[0]
        with graph.inserting_before(first_node):
            new_node = graph.call_function(
                torch.ops.alloy.batched_dequant_mm.default,
                args=(lhs, packed_weights, scales_list, zeros_list, group_size),
            )
            getitems: list[torch.fx.Node] = []
            for k in range(len(group)):
                getitem = graph.call_function(operator.getitem, args=(new_node, k))
                getitems.append(getitem)

        for dm_node, getitem in zip(group, getitems, strict=True):
            dm_node.replace_all_uses_with(getitem)
            consumed_ids.add(id(dm_node))

        for dead in reversed(group):
            if len(dead.users) == 0:
                graph.erase_node(dead)

        count += 1

    return count


def _rewrite_batched_gguf_mm(
    graph: torch.fx.Graph,
    op_target: torch.fx.node.Target,
    batched_op_target: torch.fx.node.Target,
    n_weight_args: int,
) -> int:
    """Group same-format gguf_*_mm ops that share their LHS source — even
    when interleaved by other ops — into one batched dispatch. Generic over
    Q4_K (1: native blocks), Q5_0 (3: qw, qhigh, scales),
    Q8_0 (2: qw, scales), Q6_K (1: packed)."""
    count = 0
    nodes = list(graph.nodes)
    consumed_ids: set[int] = set()

    for i, node in enumerate(nodes):
        if id(node) in consumed_ids:
            continue
        if node.op != "call_function" or node.target != op_target:
            continue

        activations = _arg_node(node, 0)
        if activations is None:
            continue
        source, _ = collect_transparent(activations)

        group: list[torch.fx.Node] = [node]
        for cand in nodes[i + 1 :]:
            if id(cand) in consumed_ids:
                continue
            if cand.op != "call_function" or cand.target != op_target:
                continue
            cand_act = _arg_node(cand, 0)
            if cand_act is None:
                continue
            cand_source, _ = collect_transparent(cand_act)
            if cand_source is source:
                group.append(cand)

        if len(group) < 2:
            continue

        lhs = _arg_node(group[0], 0)
        if lhs is None:
            continue
        arg_lists: list[list[torch.fx.node.Argument]] = [[] for _ in range(n_weight_args)]
        ok = True
        for group_node in group:
            for k in range(n_weight_args):
                arg = _arg(group_node, 1 + k)
                if arg is None:
                    ok = False
                    break
                arg_lists[k].append(arg)
            if not ok:
                break
        if not ok or len(arg_lists[0]) != len(group):
            continue

        with graph.inserting_before(group[0]):
            new_node = graph.call_function(
                batched_op_target,
                args=(lhs, *arg_lists),
            )
            getitems = [
                graph.call_function(operator.getitem, args=(new_node, k))
                for k in range(len(group))
            ]

        for mm_node, getitem in zip(group, getitems, strict=True):
            mm_node.replace_all_uses_with(getitem)
            consumed_ids.add(id(mm_node))

        for dead in reversed(group):
            if len(dead.users) == 0:
                graph.erase_node(dead)

        count += 1

    return count


def rewrite_batched_gguf_q4_k_mm(graph: torch.fx.Graph) -> int:
    return _rewrite_batched_gguf_mm(
        graph,
        torch.ops.alloy.gguf_q4_k_mm.default,
        torch.ops.alloy.batched_gguf_q4_k_mm.default,
        n_weight_args=1,
    )


def rewrite_batched_gguf_q5_0_mm(graph: torch.fx.Graph) -> int:
    return _rewrite_batched_gguf_mm(
        graph,
        torch.ops.alloy.gguf_q5_0_mm.default,
        torch.ops.alloy.batched_gguf_q5_0_mm.default,
        n_weight_args=3,
    )


def rewrite_batched_gguf_q8_0_mm(graph: torch.fx.Graph) -> int:
    return _rewrite_batched_gguf_mm(
        graph,
        torch.ops.alloy.gguf_q8_0_mm.default,
        torch.ops.alloy.batched_gguf_q8_0_mm.default,
        n_weight_args=2,
    )


def rewrite_batched_gguf_q6_k_mm(graph: torch.fx.Graph) -> int:
    return _rewrite_batched_gguf_mm(
        graph,
        torch.ops.alloy.gguf_q6_k_mm.default,
        torch.ops.alloy.batched_gguf_q6_k_mm.default,
        n_weight_args=1,
    )


def rewrite_batched_mlx_q4_mm(graph: torch.fx.Graph) -> int:
    return _rewrite_batched_gguf_mm(
        graph,
        torch.ops.alloy.mlx_q4_mm.default,
        torch.ops.alloy.batched_mlx_q4_mm.default,
        n_weight_args=3,
    )


def rewrite_dequant_mm_silu(graph: torch.fx.Graph) -> int:
    """Replace dequant gate/up SiLU pairs with alloy.dequant_silu."""
    consumed_ids: set[int] = set()
    count = 0

    for node in list(graph.nodes):
        if id(node) in consumed_ids:
            continue
        if node.op != "call_function" or node.target not in _DQ_TARGETS:
            continue

        gate_dq = node
        cur = gate_dq
        gate_views: list[torch.fx.Node] = []
        while True:
            users = [user for user in cur.users if user.op == "call_function"]
            if len(users) != 1:
                break
            next_node = users[0]
            if next_node.target in TRANSPARENT_TARGETS:
                gate_views.append(next_node)
                cur = next_node
                continue
            break

        sigmoid_users = [
            user
            for user in cur.users
            if user.op == "call_function" and user.target in _SIGMOID_TARGETS
        ]
        if len(sigmoid_users) != 1:
            continue
        sigmoid_node = sigmoid_users[0]

        silu_mul_users = [
            user
            for user in sigmoid_node.users
            if user.op == "call_function" and user.target in _MUL_TARGETS
        ]
        if len(silu_mul_users) != 1:
            continue
        silu_mul_node = silu_mul_users[0]

        final_mul_users = [
            user
            for user in silu_mul_node.users
            if user.op == "call_function" and user.target in _MUL_TARGETS
        ]
        if len(final_mul_users) != 1:
            continue
        mul_node = final_mul_users[0]

        other_arg = mul_node.args[1] if mul_node.args[0] is silu_mul_node else mul_node.args[0]
        if not isinstance(other_arg, torch.fx.Node):
            continue
        up_views: list[torch.fx.Node] = []
        trace = other_arg
        while trace.op == "call_function" and trace.target in TRANSPARENT_TARGETS:
            up_views.append(trace)
            traced_arg = _arg_node(trace, 0)
            if traced_arg is None:
                break
            trace = traced_arg

        if trace.target not in _DQ_TARGETS:
            continue
        up_dq = trace

        gate_activations = _arg_node(gate_dq, 0)
        up_activations = _arg_node(up_dq, 0)
        if gate_activations is None or up_activations is None:
            continue
        gate_source, _ = collect_transparent(gate_activations)
        up_source, _ = collect_transparent(up_activations)
        if gate_source is not up_source:
            continue

        gate_packed = _arg(gate_dq, 1)
        gate_scales = _arg(gate_dq, 2)
        zeros = _arg(gate_dq, 3)
        group_size = _arg(gate_dq, 4)
        up_packed = _arg(up_dq, 1)
        up_scales = _arg(up_dq, 2)
        if (
            gate_packed is None
            or gate_scales is None
            or zeros is None
            or group_size is None
            or up_packed is None
            or up_scales is None
        ):
            continue

        with graph.inserting_after(mul_node):
            silu_node = graph.call_function(
                torch.ops.alloy.dequant_silu.default,
                args=(
                    gate_activations,
                    gate_packed,
                    gate_scales,
                    up_packed,
                    up_scales,
                    zeros,
                    group_size,
                ),
            )

        result_node = silu_node
        for view_node in gate_views:
            view_target = cast(Callable[..., torch.fx.node.Argument], view_node.target)
            with graph.inserting_after(result_node):
                result_node = graph.call_function(
                    view_target,
                    args=(result_node,) + view_node.args[1:],
                    kwargs=dict(view_node.kwargs),
                )

        mul_node.replace_all_uses_with(result_node)

        consumed = (
            [mul_node, silu_mul_node, sigmoid_node]
            + list(reversed(gate_views))
            + list(reversed(up_views))
        )
        consumed.extend([gate_dq, up_dq])
        for dead in consumed:
            if len(dead.users) == 0:
                graph.erase_node(dead)

        consumed_ids.update(
            id(dead) for dead in (gate_dq, up_dq, sigmoid_node, silu_mul_node, mul_node)
        )
        consumed_ids.update(id(view_node) for view_node in gate_views + up_views)
        count += 1

    return count
