"""GEMM rewrite passes for Alloy torch FX graphs."""

from __future__ import annotations

from collections.abc import Callable
import operator
from typing import cast

import alloy_torch.custom_ops  # noqa: F401
import torch
import torch.fx
from torch._ops import OpOverload

from alloy_torch.rewrites.graph_utils import TRANSPARENT_TARGETS
from alloy_torch.rewrites.graph_utils import collect_transparent
from alloy_torch.rewrites.graph_utils import find_single_consumer


_BATCHED_MM_TARGETS = {torch.ops.aten.mm.default, torch.ops.aten.addmm.default}
_ADD_TARGETS = {torch.ops.aten.add.Tensor, operator.add}
_GEMM_TARGETS = {torch.ops.aten.addmm.default, torch.ops.aten.mm.default}
_MM_TARGETS = {torch.ops.aten.mm.default}
_MUL_TARGETS = {torch.ops.aten.mul.Tensor, operator.mul}
_GGUF_SILU_TARGETS: dict[OpOverload, OpOverload] = {
    torch.ops.alloy.gguf_q4_k_mm.default: torch.ops.alloy.gguf_q4_k_silu.default,
    torch.ops.alloy.gguf_q8_0_mm.default: torch.ops.alloy.gguf_q8_0_silu.default,
    torch.ops.alloy.mlx_q4_mm.default: torch.ops.alloy.mlx_q4_silu.default,
}
# gemma's MLP gate uses gelu(tanh) instead of silu. dot_q4_k_gelu_v2 mirrors the
# silu kernel with the al.gelu_tanh activation.
_GGUF_GELU_TARGETS: dict[OpOverload, OpOverload] = {
    torch.ops.alloy.gguf_q4_k_mm.default: torch.ops.alloy.gguf_q4_k_gelu.default,
}
_GELU_TARGETS = {torch.ops.aten.gelu.default}
_RMS_SKIP_TARGETS = TRANSPARENT_TARGETS | {
    torch.ops.aten._to_copy.default,
    torch.ops.aten.to.dtype,
}
_SIGMOID_TARGETS = {torch.ops.aten.sigmoid.default}
_WEIGHT_VIEW_TARGETS = {
    torch.ops.aten.permute.default,
    torch.ops.aten.t.default,
    torch.ops.aten.transpose.int,
}


def _arg_node(node: torch.fx.Node, index: int) -> torch.fx.Node | None:
    if len(node.args) <= index:
        return None
    arg = node.args[index]
    return arg if isinstance(arg, torch.fx.Node) else None


def _mm_lhs(node: torch.fx.Node) -> torch.fx.Node | None:
    """Return the LHS activation input of an mm or addmm node."""
    if node.target == torch.ops.aten.addmm.default:
        return _arg_node(node, 1)
    return _arg_node(node, 0)


def _mm_rhs(node: torch.fx.Node) -> torch.fx.Node | None:
    """Return the RHS weight input of an mm or addmm node."""
    if node.target == torch.ops.aten.addmm.default:
        return _arg_node(node, 2)
    return _arg_node(node, 1)


def _mm_bias(node: torch.fx.Node) -> torch.fx.node.Argument | None:
    """Return the bias of an addmm node, or None for mm."""
    if node.target != torch.ops.aten.addmm.default or len(node.args) == 0:
        return None
    return node.args[0]


def rewrite_gemm_residual_layernorm(graph: torch.fx.Graph) -> int:
    """Replace mm/addmm to residual LayerNorm chains with a fused custom op."""
    count = 0
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target not in _GEMM_TARGETS:
            continue

        cur = node
        skipped: list[torch.fx.Node] = []
        residual_node: torch.fx.node.Argument | None = None
        ln_node: torch.fx.Node | None = None

        while True:
            next_node = find_single_consumer(cur)
            if next_node is None:
                break

            if next_node.target in TRANSPARENT_TARGETS:
                skipped.append(next_node)
                cur = next_node
                continue

            if next_node.target == torch.ops.aten.add.Tensor:
                if len(next_node.args) < 2:
                    break
                if next_node.args[0] is cur:
                    residual_node = next_node.args[1]
                elif next_node.args[1] is cur:
                    residual_node = next_node.args[0]
                else:
                    break
                skipped.append(next_node)
                for user in next_node.users:
                    if (
                        user.op == "call_function"
                        and user.target == torch.ops.aten.native_layer_norm.default
                        and _arg_node(user, 0) is next_node
                    ):
                        ln_node = user
                        break
                if ln_node is not None:
                    break
                cur = next_node
                continue

            if (
                next_node.target == torch.ops.aten.native_layer_norm.default
                and _arg_node(next_node, 0) is cur
            ):
                ln_node = next_node
                break

            break

        if ln_node is None:
            continue

        saves_mean_or_invstd = any(
            user.op == "call_function"
            and user.target is operator.getitem
            and len(user.args) >= 2
            and user.args[1] in (1, 2)
            and len(user.users) > 0
            for user in ln_node.users
        )
        if saves_mean_or_invstd:
            continue

        gemm_node = node
        if gemm_node.target == torch.ops.aten.addmm.default:
            if len(gemm_node.args) < 3:
                continue
            bias_node = gemm_node.args[0]
            mat1 = _arg_node(gemm_node, 1)
            mat2 = _arg_node(gemm_node, 2)
        else:
            bias_node = None
            mat1 = _arg_node(gemm_node, 0)
            mat2 = _arg_node(gemm_node, 1)
        if mat1 is None or mat2 is None:
            continue

        ln_weight = ln_node.args[2] if len(ln_node.args) > 2 else None
        ln_bias = ln_node.args[3] if len(ln_node.args) > 3 else None
        ln_eps = ln_node.args[4] if len(ln_node.args) > 4 else 1e-5
        normalized_shape = ln_node.args[1] if len(ln_node.args) > 1 else None

        with graph.inserting_after(ln_node):
            new_node = graph.call_function(
                torch.ops.alloy.gemm_residual_layernorm.default,
                args=(
                    mat1,
                    mat2,
                    bias_node,
                    residual_node,
                    ln_weight,
                    ln_bias,
                    normalized_shape,
                    ln_eps,
                ),
            )

        with graph.inserting_after(new_node):
            gi_ln = graph.call_function(operator.getitem, args=(new_node, 0))
        with graph.inserting_after(gi_ln):
            gi_res = graph.call_function(operator.getitem, args=(new_node, 1))

        add_node = (
            skipped[-1] if skipped and skipped[-1].target == torch.ops.aten.add.Tensor else None
        )
        if add_node is not None:
            for user in list(add_node.users):
                if user is not ln_node and user is not new_node:
                    user.replace_input_with(add_node, gi_res)

        for user in list(ln_node.users):
            if user is new_node:
                continue
            if (
                user.op == "call_function"
                and user.target is operator.getitem
                and len(user.args) >= 2
                and user.args[1] == 0
            ):
                user.replace_all_uses_with(gi_ln)

        pos_map = {graph_node: i for i, graph_node in enumerate(graph.nodes)}
        for getitem in (gi_ln, gi_res):
            getitem_pos = pos_map[getitem]
            for user in list(getitem.users):
                if user in pos_map and pos_map[user] < getitem_pos:
                    if user.target == torch.ops.aten._to_copy.default:
                        continue
                    gi_res.append(user)

        consumed = [ln_node] + list(reversed(skipped)) + [gemm_node]
        for dead in consumed:
            if len(dead.users) == 0:
                graph.erase_node(dead)

        count += 1

    return count


def rewrite_gemm_residual_rmsnorm(graph: torch.fx.Graph) -> int:
    """Replace mm to residual RMSNorm chains with a fused custom op."""
    count = 0
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target is not torch.ops.alloy.rms_norm.default:
            continue
        if len(node.args) < 2:
            continue

        rms_input = _arg_node(node, 0)
        if rms_input is None:
            continue
        rms_weight = node.args[1]
        rms_eps = node.args[2] if len(node.args) > 2 else 1e-6

        cur = rms_input
        skipped: list[torch.fx.Node] = []
        while cur.op == "call_function" and cur.target in _RMS_SKIP_TARGETS:
            skipped.append(cur)
            next_node = _arg_node(cur, 0)
            if next_node is None:
                break
            cur = next_node

        if cur.op != "call_function" or cur.target not in _ADD_TARGETS:
            continue
        add_node = cur

        add_lhs = _arg_node(add_node, 0)
        add_rhs = _arg_node(add_node, 1)
        if add_lhs is None or add_rhs is None:
            continue

        lhs_src, lhs_views = collect_transparent(add_lhs)
        rhs_src, rhs_views = collect_transparent(add_rhs)

        gemm_node: torch.fx.Node
        residual_node: torch.fx.Node
        view_nodes: list[torch.fx.Node]
        if lhs_src.op == "call_function" and lhs_src.target in _GEMM_TARGETS:
            gemm_node, residual_node, view_nodes = lhs_src, add_rhs, lhs_views
        elif rhs_src.op == "call_function" and rhs_src.target in _GEMM_TARGETS:
            gemm_node, residual_node, view_nodes = rhs_src, add_lhs, rhs_views
        else:
            continue

        if gemm_node.target != torch.ops.aten.mm.default:
            continue

        gemm_chain = [gemm_node] + view_nodes + [add_node]
        if any(len(chain_node.users) != 1 for chain_node in gemm_chain[:-1]):
            continue

        mat1 = _arg_node(gemm_node, 0)
        mat2 = _arg_node(gemm_node, 1)
        if mat1 is None or mat2 is None:
            continue

        if mat2.op == "call_function" and mat2.target in _WEIGHT_VIEW_TARGETS:
            unwrapped_mat2 = _arg_node(mat2, 0)
            if unwrapped_mat2 is None:
                continue
            mat2 = unwrapped_mat2

        rms_gi_out: torch.fx.Node | None = None
        rms_gi_rsqrt: torch.fx.Node | None = None
        for user in list(node.users):
            if (
                user.op != "call_function"
                or user.target is not operator.getitem
                or len(user.args) < 2
            ):
                continue
            if user.args[1] == 0:
                rms_gi_out = user
            elif user.args[1] == 1:
                rms_gi_rsqrt = user

        with graph.inserting_after(node):
            new_node = graph.call_function(
                torch.ops.alloy.gemm_residual_rmsnorm.default,
                args=(mat1, mat2, residual_node, rms_weight, rms_eps),
            )

        with graph.inserting_after(new_node):
            gi_rms = graph.call_function(operator.getitem, args=(new_node, 0))
        with graph.inserting_after(gi_rms):
            gi_res = graph.call_function(operator.getitem, args=(new_node, 1))
        with graph.inserting_after(gi_res):
            gi_rsqrt = graph.call_function(operator.getitem, args=(new_node, 2))

        if rms_gi_out is not None:
            rms_gi_out.replace_all_uses_with(gi_rms)
            graph.erase_node(rms_gi_out)
        else:
            node.replace_all_uses_with(gi_rms)
        if rms_gi_rsqrt is not None:
            rms_gi_rsqrt.replace_all_uses_with(gi_rsqrt)
            graph.erase_node(rms_gi_rsqrt)

        for user in list(add_node.users):
            if user is not node and user is not new_node:
                user.replace_input_with(add_node, gi_res)
        for skipped_node in skipped:
            for user in list(skipped_node.users):
                if user is not node and user is not new_node:
                    user.replace_input_with(skipped_node, gi_res)

        pos_map = {graph_node: i for i, graph_node in enumerate(graph.nodes)}
        for getitem in (gi_rms, gi_res):
            getitem_pos = pos_map[getitem]
            for user in list(getitem.users):
                if user in pos_map and pos_map[user] < getitem_pos:
                    if user.target == torch.ops.aten._to_copy.default:
                        continue
                    gi_res.append(user)

        consumed = (
            [node] + list(reversed(skipped)) + [add_node] + list(reversed(view_nodes)) + [gemm_node]
        )
        for dead in consumed:
            if len(dead.users) == 0:
                graph.erase_node(dead)

        count += 1

    return count


def rewrite_batched_mm(graph: torch.fx.Graph) -> int:
    """Replace groups of mm/addmm ops with the same LHS with a single batched mm."""
    count = 0
    nodes = list(graph.nodes)
    consumed_ids: set[int] = set()

    for i, node in enumerate(nodes):
        if id(node) in consumed_ids:
            continue
        if node.op != "call_function" or node.target not in _BATCHED_MM_TARGETS:
            continue

        node_lhs = _mm_lhs(node)
        if node_lhs is None:
            continue
        source, first_views = collect_transparent(node_lhs)
        if not all(len(view_node.users) == 1 for view_node in first_views):
            continue

        group: list[torch.fx.Node] = [node]
        all_lhs_views = list(first_views)
        for cand in nodes[i + 1 :]:
            if cand.op != "call_function":
                continue
            if cand.target in _BATCHED_MM_TARGETS:
                cand_lhs = _mm_lhs(cand)
                if cand_lhs is None:
                    break
                cand_source, cand_views = collect_transparent(cand_lhs)
                if cand_source is source and all(
                    len(view_node.users) == 1 for view_node in cand_views
                ):
                    group.append(cand)
                    all_lhs_views.extend(cand_views)
                    continue
                break
            if cand.target in TRANSPARENT_TARGETS or cand.target == torch.ops.aten.permute.default:
                continue
            break

        if len(group) < 2:
            continue

        lhs = _mm_lhs(group[0])
        if lhs is None:
            continue
        weights: list[torch.fx.Node] = []
        for mm_node in group:
            weight = _mm_rhs(mm_node)
            if weight is None:
                break
            weights.append(weight)
        if len(weights) != len(group):
            continue

        biases = [_mm_bias(mm_node) for mm_node in group]
        has_bias = any(bias is not None for bias in biases)

        first_mm = group[0]
        first_rhs = _mm_rhs(first_mm)
        if first_rhs is None:
            continue
        for weight in weights:
            if weight is not first_rhs:
                first_mm.prepend(weight)

        with graph.inserting_before(first_mm):
            new_node = graph.call_function(
                torch.ops.alloy.batched_mm.default,
                args=(lhs, weights, biases if has_bias else None),
            )
            getitems: list[torch.fx.Node] = []
            for k in range(len(group)):
                getitem = graph.call_function(operator.getitem, args=(new_node, k))
                getitems.append(getitem)

        for mm_node, getitem in zip(group, getitems, strict=True):
            mm_node.replace_all_uses_with(getitem)

        for mm_node in reversed(group):
            if len(mm_node.users) == 0:
                graph.erase_node(mm_node)
        for view_node in reversed(all_lhs_views):
            if len(view_node.users) == 0:
                graph.erase_node(view_node)

        consumed_ids.update(id(group_node) for group_node in group)
        consumed_ids.update(id(view_node) for view_node in all_lhs_views)
        count += 1

    return count


def rewrite_batched_mm_silu(graph: torch.fx.Graph) -> int:
    """Replace gate/up mm SiLU pairs with alloy.dot_silu."""
    consumed_ids: set[int] = set()
    count = 0

    for node in list(graph.nodes):
        if id(node) in consumed_ids:
            continue
        if node.op != "call_function" or node.target not in _MM_TARGETS:
            continue

        gate_mm = node
        cur = gate_mm
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

        if trace.target not in _MM_TARGETS:
            continue
        up_mm = trace

        gate_lhs = _arg_node(gate_mm, 0)
        up_lhs = _arg_node(up_mm, 0)
        if gate_lhs is None or up_lhs is None:
            continue
        gate_source, _ = collect_transparent(gate_lhs)
        up_source, _ = collect_transparent(up_lhs)
        if gate_source is not up_source:
            continue

        x_node = gate_lhs
        gate_w = _arg_node(gate_mm, 1)
        up_w = _arg_node(up_mm, 1)
        if gate_w is None or up_w is None:
            continue

        with graph.inserting_after(mul_node):
            silu_node = graph.call_function(
                torch.ops.alloy.dot_silu.default,
                args=(x_node, gate_w, up_w),
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
        consumed.extend([gate_mm, up_mm])
        for dead in consumed:
            if len(dead.users) == 0:
                graph.erase_node(dead)

        consumed_ids.update(
            id(dead) for dead in (gate_mm, up_mm, sigmoid_node, silu_mul_node, mul_node)
        )
        consumed_ids.update(id(view_node) for view_node in gate_views + up_views)
        count += 1

    return count


def rewrite_gguf_mm_silu(graph: torch.fx.Graph) -> int:
    """Replace GGUF gate/up SiLU pairs with paired quantized matmul."""
    consumed_ids: set[int] = set()
    count = 0

    for node in list(graph.nodes):
        if id(node) in consumed_ids:
            continue
        if node.op != "call_function" or node.target not in _GGUF_SILU_TARGETS:
            continue

        gate_mm = node
        cur = gate_mm
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

        if trace.target != gate_mm.target:
            continue
        up_mm = trace

        gate_lhs = _arg_node(gate_mm, 0)
        up_lhs = _arg_node(up_mm, 0)
        if gate_lhs is None or up_lhs is None:
            continue
        gate_source, _ = collect_transparent(gate_lhs)
        up_source, _ = collect_transparent(up_lhs)
        if gate_source is not up_source:
            continue

        # Generic over weight-arg count: native Q4_K mm has 1 weight tensor
        # (blocks); the fused op takes (lhs, *gate_weights, *up_weights).
        gate_weights = list(gate_mm.args[1:])
        up_weights = list(up_mm.args[1:])
        if (
            not gate_weights
            or len(gate_weights) != len(up_weights)
            or any(w is None for w in gate_weights + up_weights)
        ):
            continue

        if not isinstance(gate_mm.target, OpOverload):
            continue
        fused_target = _GGUF_SILU_TARGETS[gate_mm.target]
        fused_args = (gate_lhs, *gate_weights, *up_weights)
        with graph.inserting_after(mul_node):
            silu_node = graph.call_function(
                fused_target,
                args=fused_args,
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
        consumed.extend([gate_mm, up_mm])
        for dead in consumed:
            if len(dead.users) == 0:
                graph.erase_node(dead)

        consumed_ids.update(
            id(dead) for dead in (gate_mm, up_mm, sigmoid_node, silu_mul_node, mul_node)
        )
        consumed_ids.update(id(view_node) for view_node in gate_views + up_views)
        count += 1

    return count


def rewrite_gguf_mm_gelu(graph: torch.fx.Graph) -> int:
    """Replace GGUF gate/up gelu(tanh) pairs with the fused gate+up gelu matmul.

    Mirror of rewrite_gguf_mm_silu: gate_mm -> [views] -> gelu(approximate=tanh)
    -> mul(up_mm). Only the tanh approximation is fused (dot_q4_k_gelu_v2 uses
    al.gelu_tanh)."""
    consumed_ids: set[int] = set()
    count = 0

    for node in list(graph.nodes):
        if id(node) in consumed_ids:
            continue
        if node.op != "call_function" or node.target not in _GGUF_GELU_TARGETS:
            continue

        gate_mm = node
        cur = gate_mm
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

        gelu_users = [
            user
            for user in cur.users
            if user.op == "call_function"
            and user.target in _GELU_TARGETS
            and user.kwargs.get("approximate") == "tanh"
        ]
        if len(gelu_users) != 1:
            continue
        gelu_node = gelu_users[0]

        final_mul_users = [
            user
            for user in gelu_node.users
            if user.op == "call_function" and user.target in _MUL_TARGETS
        ]
        if len(final_mul_users) != 1:
            continue
        mul_node = final_mul_users[0]

        other_arg = mul_node.args[1] if mul_node.args[0] is gelu_node else mul_node.args[0]
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

        if trace.target != gate_mm.target:
            continue
        up_mm = trace

        gate_lhs = _arg_node(gate_mm, 0)
        up_lhs = _arg_node(up_mm, 0)
        if gate_lhs is None or up_lhs is None:
            continue
        gate_source, _ = collect_transparent(gate_lhs)
        up_source, _ = collect_transparent(up_lhs)
        if gate_source is not up_source:
            continue

        # Generic over weight-arg count (native Q4_K: 1 weight tensor each).
        gate_weights = list(gate_mm.args[1:])
        up_weights = list(up_mm.args[1:])
        if (
            not gate_weights
            or len(gate_weights) != len(up_weights)
            or any(w is None for w in gate_weights + up_weights)
        ):
            continue

        if not isinstance(gate_mm.target, OpOverload):
            continue
        fused_target = _GGUF_GELU_TARGETS[gate_mm.target]
        fused_args = (gate_lhs, *gate_weights, *up_weights)
        with graph.inserting_after(mul_node):
            gelu_fused = graph.call_function(fused_target, args=fused_args)

        result_node = gelu_fused
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
            [mul_node, gelu_node]
            + list(reversed(gate_views))
            + list(reversed(up_views))
        )
        consumed.extend([gate_mm, up_mm])
        for dead in consumed:
            if len(dead.users) == 0:
                graph.erase_node(dead)

        consumed_ids.update(id(dead) for dead in (gate_mm, up_mm, gelu_node, mul_node))
        consumed_ids.update(id(view_node) for view_node in gate_views + up_views)
        count += 1

    return count
