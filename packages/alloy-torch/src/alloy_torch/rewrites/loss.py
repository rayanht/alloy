"""Loss rewrite passes for Alloy torch FX graphs."""

from __future__ import annotations

import operator

import alloy_torch.custom_ops  # noqa: F401
import torch
import torch.fx


def _arg_node(node: torch.fx.Node, index: int) -> torch.fx.Node | None:
    if len(node.args) <= index:
        return None
    arg = node.args[index]
    return arg if isinstance(arg, torch.fx.Node) else None


def rewrite_cross_entropy_fwd(graph: torch.fx.Graph) -> int:
    """Collapse decomposed F.cross_entropy(reduction='mean') forward."""
    count = 0
    output_node = next((node for node in graph.nodes if node.op == "output"), None)
    if output_node is None:
        return 0

    for node in list(graph.nodes):
        if node.op != "call_function" or node.target is not torch.ops.aten._log_softmax.default:
            continue
        if len(node.args) < 2 or node.args[1] != 1:
            continue
        flat_logits = _arg_node(node, 0)
        if flat_logits is None:
            continue

        alias_node: torch.fx.Node | None = None
        gather_node: torch.fx.Node | None = None
        for user in node.users:
            if user.target is torch.ops.aten.alias.default:
                alias_node = user
            elif user.target is torch.ops.aten.gather.default:
                gather_node = user
        if alias_node is None or gather_node is None:
            continue
        if len(gather_node.args) < 3 or gather_node.args[1] != 1:
            continue
        unsqueeze_node = _arg_node(gather_node, 2)
        if (
            unsqueeze_node is None
            or unsqueeze_node.op != "call_function"
            or unsqueeze_node.target is not torch.ops.aten.unsqueeze.default
        ):
            continue
        where_lbl = _arg_node(unsqueeze_node, 0)
        if where_lbl is None or where_lbl.target is not torch.ops.aten.where.self:
            continue
        ne_lbl = _arg_node(where_lbl, 0)
        if (
            ne_lbl is None
            or ne_lbl.target is not torch.ops.aten.ne.Scalar
            or len(ne_lbl.args) < 2
            or ne_lbl.args[1] != -100
        ):
            continue
        flat_labels = _arg_node(ne_lbl, 0)
        if flat_labels is None:
            continue

        squeeze_node: torch.fx.Node | None = None
        for user in gather_node.users:
            if user.target is torch.ops.aten.squeeze.dims:
                squeeze_node = user
                break
        if squeeze_node is None:
            continue

        neg_node: torch.fx.Node | None = None
        for user in squeeze_node.users:
            if user.target is torch.ops.aten.neg.default:
                neg_node = user
                break
        if neg_node is None:
            continue

        where_1: torch.fx.Node | None = None
        for user in neg_node.users:
            if user.target is torch.ops.aten.where.self:
                where_1 = user
                break
        if where_1 is None:
            continue

        sum_2: torch.fx.Node | None = None
        for user in where_1.users:
            if user.target is torch.ops.aten.sum.dim_IntList:
                sum_2 = user
                break
        if sum_2 is None or len(sum_2.args) < 2 or sum_2.args[1] not in ([], None):
            continue

        div_node: torch.fx.Node | None = None
        for user in sum_2.users:
            if user.target is torch.ops.aten.div.Tensor:
                div_node = user
                break
        if div_node is None:
            continue
        n_valid_f32 = _arg_node(div_node, 1)
        if n_valid_f32 is None or n_valid_f32.target is not torch.ops.aten._to_copy.default:
            continue
        if len(output_node.args) < 1 or not isinstance(output_node.args[0], tuple):
            continue
        out_args = output_node.args[0]

        with graph.inserting_before(div_node):
            new_op = graph.call_function(
                torch.ops.alloy.cross_entropy_fwd_fused.default,
                args=(flat_logits, flat_labels, -100),
            )
            new_loss = graph.call_function(operator.getitem, args=(new_op, 0))
            new_lse = graph.call_function(operator.getitem, args=(new_op, 1))
            new_n_valid = graph.call_function(operator.getitem, args=(new_op, 2))

        div_node.replace_all_uses_with(new_loss)

        output_node.args = (
            tuple(
                new_lse
                if arg is alias_node
                else flat_logits
                if arg is node
                else new_n_valid
                if arg is n_valid_f32
                else arg
                for arg in out_args
            ),
        )

        count += 1
    return count


def rewrite_cross_entropy_bwd(graph: torch.fx.Graph) -> int:
    """Collapse decomposed F.cross_entropy backward paired with the fwd rewrite."""
    count = 0
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target is not torch.ops.aten.scatter.value:
            continue
        if len(node.args) != 4 or node.args[3] != -1.0 or node.args[2] is None:
            continue

        full_like_node = _arg_node(node, 0)
        where_lbl = _arg_node(node, 2)
        if full_like_node is None or where_lbl is None or node.args[1] != 1:
            continue
        if (
            full_like_node.op != "call_function"
            or full_like_node.target is not torch.ops.aten.full_like.default
            or len(full_like_node.args) < 2
            or full_like_node.args[1] != 0
        ):
            continue
        logits_placeholder = _arg_node(full_like_node, 0)
        if logits_placeholder is None:
            continue

        if where_lbl.op != "call_function" or where_lbl.target is not torch.ops.aten.where.self:
            continue
        ne_labels = _arg_node(where_lbl, 0)
        unsqueeze_lbl = _arg_node(where_lbl, 1)
        if (
            unsqueeze_lbl is None
            or unsqueeze_lbl.op != "call_function"
            or unsqueeze_lbl.target is not torch.ops.aten.unsqueeze.default
        ):
            continue
        labels_node = _arg_node(unsqueeze_lbl, 0)
        if labels_node is None:
            continue
        if (
            ne_labels is None
            or ne_labels.op != "call_function"
            or ne_labels.target is not torch.ops.aten.ne.Scalar
            or len(ne_labels.args) < 2
            or ne_labels.args[1] != -100
        ):
            continue

        scatter_users = list(node.users)
        if len(scatter_users) != 1:
            continue
        scaled_oh = scatter_users[0]
        if scaled_oh.op != "call_function" or scaled_oh.target is not torch.ops.aten.mul.Tensor:
            continue
        if len(scaled_oh.args) < 2:
            continue
        masked_grad = (
            scaled_oh.args[1]
            if scaled_oh.args[0] is node
            else scaled_oh.args[0]
            if scaled_oh.args[1] is node
            else None
        )
        if not isinstance(masked_grad, torch.fx.Node):
            continue
        if masked_grad.op != "call_function" or masked_grad.target is not torch.ops.aten.where.self:
            continue
        div_node = _arg_node(masked_grad, 1)
        if div_node is None or div_node.target is not torch.ops.aten.div.Tensor:
            continue
        tangents_node = _arg_node(div_node, 0)
        n_valid_node = _arg_node(div_node, 1)
        if tangents_node is None or n_valid_node is None:
            continue

        effective_scaled_oh = scaled_oh
        if len(scaled_oh.users) == 1:
            sole = next(iter(scaled_oh.users))
            if (
                sole.op == "call_function"
                and sole.target is torch.ops.aten._to_copy.default
                and sole.kwargs.get("dtype") == torch.float32
            ):
                effective_scaled_oh = sole

        sub_node: torch.fx.Node | None = None
        row_sum_node: torch.fx.Node | None = None
        for user in effective_scaled_oh.users:
            if user.target is torch.ops.aten.sum.dim_IntList:
                row_sum_node = user
            elif user.target is torch.ops.aten.sub.Tensor:
                sub_node = user
        if sub_node is None or row_sum_node is None:
            continue
        if len(sub_node.args) < 2:
            continue
        probs_rsum = (
            sub_node.args[1]
            if sub_node.args[0] is effective_scaled_oh
            else sub_node.args[0]
            if sub_node.args[1] is effective_scaled_oh
            else None
        )
        if not isinstance(probs_rsum, torch.fx.Node):
            continue
        if probs_rsum.target is not torch.ops.aten.mul.Tensor:
            continue

        exp_node: torch.fx.Node | None = None
        for arg in probs_rsum.args:
            if isinstance(arg, torch.fx.Node) and arg.target is torch.ops.aten.exp.default:
                exp_node = arg
                break
        if exp_node is None:
            continue

        lse_source = _arg_node(exp_node, 0)
        while isinstance(lse_source, torch.fx.Node) and (
            lse_source.target is torch.ops.aten.alias.default
            or (
                lse_source.target is torch.ops.aten._to_copy.default
                and lse_source.kwargs.get("dtype") == torch.float32
            )
        ):
            next_source = _arg_node(lse_source, 0)
            if next_source is None:
                break
            lse_source = next_source
        if lse_source is None:
            continue

        with graph.inserting_before(sub_node):
            new_node = graph.call_function(
                torch.ops.alloy.cross_entropy_bwd_fused.default,
                args=(
                    logits_placeholder,
                    labels_node,
                    lse_source,
                    n_valid_node,
                    tangents_node,
                    -100,
                ),
            )
        sub_node.replace_all_uses_with(new_node)
        count += 1
    return count
