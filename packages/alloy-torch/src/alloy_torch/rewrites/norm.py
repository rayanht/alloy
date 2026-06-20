"""Normalization rewrite passes for Alloy torch FX graphs."""

from __future__ import annotations

from dataclasses import dataclass
import operator

import alloy_torch.custom_ops  # noqa: F401
import torch
import torch.fx

from alloy_torch.rewrites.graph_utils import find_single_consumer


_ADD_TARGETS = {torch.ops.aten.add.Tensor, operator.add}
_MUL_TARGETS = {torch.ops.aten.mul.Tensor, operator.mul}
_TO_TARGETS = {torch.ops.aten._to_copy.default, torch.ops.aten.to.dtype}
_TO_COPY_TARGETS = {torch.ops.aten._to_copy.default}
_ALIAS_TARGETS = {torch.ops.aten.alias.default}


@dataclass(frozen=True)
class RmsNormForwardMatch:
    input_node: torch.fx.Node
    weight_node: torch.fx.Node | None  # None => affine-free RMSNorm (no learnable scale)
    eps: float
    output_node: torch.fx.Node
    rsqrt_node: torch.fx.Node
    consumed: tuple[torch.fx.Node, ...]


@dataclass(frozen=True)
class RmsNormBackwardMatch:
    dy_node: torch.fx.Node
    weight_node: torch.fx.Node
    x_node: torch.fx.Node
    rrms_node: torch.fx.Node
    consumed: tuple[torch.fx.Node, ...]


def _arg_node(node: torch.fx.Node, index: int) -> torch.fx.Node | None:
    if len(node.args) <= index:
        return None
    arg = node.args[index]
    return arg if isinstance(arg, torch.fx.Node) else None


def _is_call_target(node: torch.fx.Node, target) -> bool:
    return node.op == "call_function" and node.target is target


def _other_node(node: torch.fx.Node, known: torch.fx.Node) -> torch.fx.Node | None:
    if len(node.args) < 2:
        return None
    lhs, rhs = node.args[0], node.args[1]
    if lhs is known and isinstance(rhs, torch.fx.Node):
        return rhs
    if rhs is known and isinstance(lhs, torch.fx.Node):
        return lhs
    return None


def _is_rsqrt_like(node: torch.fx.Node) -> bool:
    """The reciprocal-sqrt step of an RMSNorm.

    Most models emit `aten.rsqrt`. Gemma4's RMSNorm uses
    `torch.pow(mean_squared, -0.5)`, which decomposes to `aten.pow.Tensor_Scalar`
    with exponent -0.5 — mathematically identical. Accept both so gemma4 routes
    to the f32-safe `alloy.rms_norm` kernel instead of the decomposed ATen path
    (which overflows f16 on gemma4's large pre-norm activations).
    """
    if node.target is torch.ops.aten.rsqrt.default:
        return True
    return (
        node.op == "call_function"
        and node.target is torch.ops.aten.pow.Tensor_Scalar
        and len(node.args) >= 2
        and node.args[1] == -0.5
    )


def _find_rms_norm_chain(pow_node: torch.fx.Node) -> RmsNormForwardMatch | None:
    if not _is_call_target(pow_node, torch.ops.aten.pow.Tensor_Scalar):
        return None
    if len(pow_node.args) < 2 or pow_node.args[1] != 2:
        return None

    pow_input = _arg_node(pow_node, 0)
    if pow_input is None:
        return None
    consumed: list[torch.fx.Node] = [pow_node]

    mean_node = find_single_consumer(pow_node)
    if mean_node is None or mean_node.target != torch.ops.aten.mean.dim:
        return None
    consumed.append(mean_node)

    add_node = find_single_consumer(mean_node)
    if add_node is None or add_node.target not in _ADD_TARGETS or len(add_node.args) < 2:
        return None
    eps_arg = add_node.args[1] if add_node.args[0] is mean_node else add_node.args[0]
    if not isinstance(eps_arg, int | float):
        return None
    consumed.append(add_node)

    rsqrt_node = find_single_consumer(add_node)
    if rsqrt_node is None or not _is_rsqrt_like(rsqrt_node):
        return None
    consumed.append(rsqrt_node)

    mul_node = find_single_consumer(rsqrt_node)
    if mul_node is None or mul_node.target not in _MUL_TARGETS:
        return None
    x_node = _other_node(mul_node, rsqrt_node)
    if x_node is not pow_input:
        return None
    consumed.append(mul_node)

    current = mul_node
    next_node = find_single_consumer(current)
    if next_node is not None and next_node.target in _TO_TARGETS:
        current = next_node
        consumed.append(current)

    # The learnable-scale multiply is optional. An affine-free RMSNorm (no
    # weight) ends at `x * rsqrt(...)` and feeds the next op directly — gemma4's
    # vision pre-projection norm (`mul_2 -> view -> projection mm`). Without it the
    # norm stays on the decomposed ATen path, whose variance square overflows f16
    # on the pooler's ~5e4 input -> rsqrt(inf)=0 -> all-zero features. weight_node
    # is None then; the rewrite synthesizes a ones weight.
    weight_mul = find_single_consumer(current)
    if weight_mul is None:
        return None
    if weight_mul.target in _MUL_TARGETS:
        weight_node = _other_node(weight_mul, current)
        if weight_node is None:
            return None
        consumed.append(weight_mul)
        output_node: torch.fx.Node = weight_mul
    else:
        # Single non-mul consumer => affine-free RMSNorm (no learnable scale).
        weight_node = None
        output_node = current

    if pow_input.target in _TO_TARGETS and len(pow_input.users) == 2:
        original_input = _arg_node(pow_input, 0)
        if original_input is None:
            return None
        consumed.insert(0, pow_input)
        return RmsNormForwardMatch(
            input_node=original_input,
            weight_node=weight_node,
            eps=float(eps_arg),
            output_node=output_node,
            rsqrt_node=rsqrt_node,
            consumed=tuple(consumed),
        )

    return RmsNormForwardMatch(
        input_node=pow_input,
        weight_node=weight_node,
        eps=float(eps_arg),
        output_node=output_node,
        rsqrt_node=rsqrt_node,
        consumed=tuple(consumed),
    )


def _strip_alias(node: torch.fx.Node) -> torch.fx.Node:
    current = node
    while current.target in _ALIAS_TARGETS:
        next_node = _arg_node(current, 0)
        if next_node is None:
            return current
        current = next_node
    return current


def _is_cast_to(node: torch.fx.Node, dtype: torch.dtype) -> bool:
    return (
        node.op == "call_function"
        and node.target in _TO_COPY_TARGETS
        and node.kwargs.get("dtype") == dtype
    )


def _try_rms_norm_backward(cast_node: torch.fx.Node) -> RmsNormBackwardMatch | None:
    if not _is_cast_to(cast_node, torch.bfloat16):
        return None
    add_node = _arg_node(cast_node, 0)
    if (
        add_node is None
        or add_node.target is not torch.ops.aten.add.Tensor
        or len(add_node.args) != 2
    ):
        return None

    lhs = _arg_node(add_node, 0)
    rhs = _arg_node(add_node, 1)
    if lhs is None or rhs is None:
        return None

    for first, second in ((lhs, rhs), (rhs, lhs)):
        result = _try_dx_chain(cast_node, add_node, first, second)
        if result is not None:
            return result
    return None


def _try_dx_chain(
    cast_node: torch.fx.Node,
    add_node: torch.fx.Node,
    mul_g_rrms: torch.fx.Node,
    mul_xterm: torch.fx.Node,
) -> RmsNormBackwardMatch | None:
    if mul_g_rrms.target is not torch.ops.aten.mul.Tensor or len(mul_g_rrms.args) != 2:
        return None

    g_f32: torch.fx.Node | None = None
    rrms_node: torch.fx.Node | None = None
    lhs = _arg_node(mul_g_rrms, 0)
    rhs = _arg_node(mul_g_rrms, 1)
    if lhs is None or rhs is None:
        return None
    for maybe_g, maybe_rrms in ((lhs, rhs), (rhs, lhs)):
        if _is_cast_to(maybe_g, torch.float32):
            g_f32 = maybe_g
            rrms_node = maybe_rrms
            break
    if g_f32 is None or rrms_node is None:
        return None

    mul_18 = _arg_node(g_f32, 0)
    if mul_18 is None or mul_18.target is not torch.ops.aten.mul.Tensor or len(mul_18.args) != 2:
        return None
    dy_node = _arg_node(mul_18, 0)
    weight_node = _arg_node(mul_18, 1)
    if dy_node is None or weight_node is None:
        return None

    if mul_xterm.target is not torch.ops.aten.mul.Tensor or len(mul_xterm.args) != 2:
        return None
    xterm_lhs = _arg_node(mul_xterm, 0)
    xterm_rhs = _arg_node(mul_xterm, 1)
    if xterm_lhs is None or xterm_rhs is None:
        return None

    div_node: torch.fx.Node | None = None
    mul23: torch.fx.Node | None = None
    for maybe_div, maybe_mul in ((xterm_lhs, xterm_rhs), (xterm_rhs, xterm_lhs)):
        if (
            maybe_div.target is torch.ops.aten.div.Scalar
            and maybe_mul.target is torch.ops.aten.mul.Scalar
        ):
            div_node = maybe_div
            mul23 = maybe_mul
            break
    if div_node is None or mul23 is None:
        return None

    if len(mul23.args) < 2 or mul23.args[1] != 2.0:
        return None
    pow5 = _arg_node(mul23, 0)
    if (
        pow5 is None
        or pow5.target is not torch.ops.aten.pow.Tensor_Scalar
        or len(pow5.args) < 2
        or pow5.args[1] != 1.0
    ):
        return None
    x_f32 = _arg_node(pow5, 0)
    if x_f32 is None:
        return None

    expand_node = _arg_node(div_node, 0)
    if expand_node is None or expand_node.target is not torch.ops.aten.expand.default:
        return None
    mul22 = _arg_node(expand_node, 0)
    if mul22 is None or mul22.target is not torch.ops.aten.mul.Tensor or len(mul22.args) != 2:
        return None

    mul21: torch.fx.Node | None = None
    pow4: torch.fx.Node | None = None
    mul22_lhs = _arg_node(mul22, 0)
    mul22_rhs = _arg_node(mul22, 1)
    if mul22_lhs is None or mul22_rhs is None:
        return None
    for maybe_mul, maybe_pow in ((mul22_lhs, mul22_rhs), (mul22_rhs, mul22_lhs)):
        if (
            maybe_mul.target is torch.ops.aten.mul.Scalar
            and len(maybe_mul.args) >= 2
            and maybe_mul.args[1] == -0.5
        ):
            mul21 = maybe_mul
            pow4 = maybe_pow
            break
    if (
        mul21 is None
        or pow4 is None
        or pow4.target is not torch.ops.aten.pow.Tensor_Scalar
        or len(pow4.args) < 2
        or pow4.args[1] != 3
    ):
        return None

    sum_node = _arg_node(mul21, 0)
    if sum_node is None or sum_node.target is not torch.ops.aten.sum.dim_IntList:
        return None
    mul19 = _arg_node(sum_node, 0)
    if mul19 is None or mul19.target is not torch.ops.aten.mul.Tensor or len(mul19.args) != 2:
        return None
    if not (
        (mul19.args[0] is g_f32 and mul19.args[1] is x_f32)
        or (mul19.args[1] is g_f32 and mul19.args[0] is x_f32)
    ):
        return None

    return RmsNormBackwardMatch(
        dy_node=dy_node,
        weight_node=weight_node,
        x_node=x_f32,
        rrms_node=_strip_alias(rrms_node),
        consumed=(
            cast_node,
            add_node,
            mul_g_rrms,
            g_f32,
            mul_18,
            mul_xterm,
            div_node,
            mul23,
            pow5,
            expand_node,
            mul22,
            mul21,
            sum_node,
            mul19,
            pow4,
        ),
    )


def rewrite_rms_norm_backward(graph: torch.fx.Graph) -> int:
    """Replace AOT-decomposed RMSNorm backward chains with alloy.rms_norm_backward."""
    count = 0
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target not in _TO_COPY_TARGETS:
            continue
        if node.kwargs.get("dtype") not in (torch.bfloat16, torch.float16):
            continue
        match = _try_rms_norm_backward(node)
        if match is None:
            continue
        with graph.inserting_after(node):
            new_node = graph.call_function(
                torch.ops.alloy.rms_norm_backward.default,
                args=(match.x_node, match.dy_node, match.weight_node, match.rrms_node),
            )
        node.replace_all_uses_with(new_node)
        for dead in reversed(match.consumed):
            if len(dead.users) == 0:
                graph.erase_node(dead)
        count += 1
    return count


def rewrite_rms_norm(graph: torch.fx.Graph) -> int:
    """Replace decomposed RMSNorm chains with alloy.rms_norm nodes."""
    count = 0
    consumed_set: set[int] = set()
    for node in list(graph.nodes):
        if id(node) in consumed_set:
            continue
        match = _find_rms_norm_chain(node)
        if match is None:
            continue

        weight_node = match.weight_node
        if weight_node is None:
            # Affine-free RMSNorm: the kernel takes a weight tensor, so feed it
            # ones (its `x_normed * weight` step becomes the identity).
            val = match.input_node.meta.get("val")
            n = int(val.shape[-1])
            with graph.inserting_before(match.output_node):
                weight_node = graph.call_function(
                    torch.ops.aten.full.default,
                    ([n], 1.0),
                    {"dtype": val.dtype, "device": val.device},
                )
        with graph.inserting_after(match.output_node):
            new_node = graph.call_function(
                torch.ops.alloy.rms_norm.default,
                args=(match.input_node, weight_node, match.eps),
            )
        with graph.inserting_after(new_node):
            gi_out = graph.call_function(operator.getitem, args=(new_node, 0))
        with graph.inserting_after(gi_out):
            gi_rsqrt = graph.call_function(operator.getitem, args=(new_node, 1))

        consumed_ids = {id(node) for node in match.consumed}
        match.output_node.replace_all_uses_with(gi_out)
        for user in list(match.rsqrt_node.users):
            if id(user) not in consumed_ids:
                user.replace_input_with(match.rsqrt_node, gi_rsqrt)

        for dead in reversed(match.consumed):
            if len(dead.users) == 0:
                graph.erase_node(dead)
        consumed_set.update(consumed_ids)
        count += 1

    return count
