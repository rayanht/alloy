"""RoPE rewrite passes for Alloy torch FX graphs."""

from __future__ import annotations

from dataclasses import dataclass
import operator

import alloy_torch.custom_ops  # noqa: F401
import torch
import torch.fx


_ADD_TARGETS = {torch.ops.aten.add.Tensor, operator.add}
_MUL_TARGETS = {torch.ops.aten.mul.Tensor, operator.mul}
_SUB_TARGETS = {torch.ops.aten.sub.Tensor, operator.sub}
_NEG_TARGETS = {torch.ops.aten.neg.default, operator.neg}
_CAT_TARGETS = {torch.ops.aten.cat.default, torch.cat}
_SLICE_TARGETS = {torch.ops.aten.slice.Tensor, operator.getitem}
_SLICE_SCATTER_TARGETS = {torch.ops.aten.slice_scatter.default}
_FULL_TARGETS = {torch.ops.aten.full.default}
_SLICE_T_TARGETS = {torch.ops.aten.slice.Tensor}
_COS_VIEW_TARGETS = {
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.view.default,
    torch.ops.aten._to_copy.default,
    torch.ops.aten.to.dtype,
}


@dataclass(frozen=True)
class RopeMatch:
    x_node: torch.fx.Node
    cos_node: torch.fx.Node
    sin_node: torch.fx.Node
    consumed: tuple[torch.fx.Node, ...]


@dataclass(frozen=True)
class SliceInfo:
    source: torch.fx.Node
    dim: int
    start: int
    end: int


@dataclass(frozen=True)
class SliceScatterInfo:
    node: torch.fx.Node
    base: torch.fx.Node
    source: torch.fx.Node
    dim: int
    start: int
    end: int


def _arg_node(node: torch.fx.Node, index: int) -> torch.fx.Node | None:
    if len(node.args) <= index:
        return None
    arg = node.args[index]
    return arg if isinstance(arg, torch.fx.Node) else None


def _arg_int(node: torch.fx.Node, index: int) -> int | None:
    if len(node.args) <= index:
        return None
    arg = node.args[index]
    return arg if isinstance(arg, int) else None


def _is_zero_full(node: torch.fx.Node) -> bool:
    if node.op != "call_function" or node.target not in _FULL_TARGETS or len(node.args) < 2:
        return False
    fill_value = node.args[1]
    return isinstance(fill_value, int | float) and fill_value == 0


def _slice_of(node: torch.fx.Node) -> SliceInfo | None:
    """Return slice fields for aten.slice.Tensor(src, dim, start, end)."""
    if node.op != "call_function" or node.target not in _SLICE_T_TARGETS:
        return None
    source = _arg_node(node, 0)
    dim = _arg_int(node, 1)
    start = _arg_int(node, 2)
    end = _arg_int(node, 3)
    if source is None or dim is None or start is None or end is None:
        return None
    return SliceInfo(source=source, dim=dim, start=start, end=end)


def _slice_scatter_of(node: torch.fx.Node) -> SliceScatterInfo | None:
    if node.op != "call_function" or node.target not in _SLICE_SCATTER_TARGETS:
        return None
    base = _arg_node(node, 0)
    source = _arg_node(node, 1)
    dim = _arg_int(node, 2)
    start = _arg_int(node, 3)
    end = _arg_int(node, 4)
    if base is None or source is None or dim is None or start is None or end is None:
        return None
    if not _is_zero_full(base):
        return None
    return SliceScatterInfo(
        node=node,
        base=base,
        source=source,
        dim=dim,
        start=start,
        end=end,
    )


def rewrite_rope(graph: torch.fx.Graph) -> int:
    """Replace decomposed RoPE chains with alloy.rope_apply nodes."""
    count = 0
    consumed_ids: set[int] = set()

    for node in list(graph.nodes):
        if id(node) in consumed_ids:
            continue
        if node.op != "call_function":
            continue

        if node.target in _ADD_TARGETS and len(node.args) == 2:
            lhs = _arg_node(node, 0)
            rhs = _arg_node(node, 1)
            if lhs is not None and rhs is not None:
                match = _try_rope_form1(node, lhs, rhs)
                if match is not None:
                    _replace_rope_match(graph, node, match)
                    consumed_ids.update(id(dead) for dead in match.consumed)
                    count += 1
                    continue

        if node.target in _CAT_TARGETS:
            match = _try_rope_form2(node)
            if match is not None:
                _replace_rope_match(graph, node, match)
                consumed_ids.update(id(dead) for dead in match.consumed)
                count += 1

    return count


def _replace_rope_match(graph: torch.fx.Graph, node: torch.fx.Node, match: RopeMatch) -> None:
    with graph.inserting_after(node):
        new_node = graph.call_function(
            torch.ops.alloy.rope_apply.default,
            args=(match.x_node, match.cos_node, match.sin_node),
        )
    node.replace_all_uses_with(new_node)
    for dead in reversed(match.consumed):
        if len(dead.users) == 0:
            graph.erase_node(dead)


def _try_rope_backward(add_node: torch.fx.Node) -> RopeMatch | None:
    lhs = _arg_node(add_node, 0)
    rhs = _arg_node(add_node, 1)
    if lhs is None or rhs is None:
        return None

    scat_sum: torch.fx.Node | None = None
    mul_cos: torch.fx.Node | None = None
    for maybe_scat, maybe_mul in ((lhs, rhs), (rhs, lhs)):
        if (
            maybe_scat.op == "call_function"
            and maybe_scat.target in _ADD_TARGETS
            and maybe_mul.op == "call_function"
            and maybe_mul.target in _MUL_TARGETS
        ):
            scat_sum = maybe_scat
            mul_cos = maybe_mul
            break
    if scat_sum is None or mul_cos is None:
        return None

    ss_a = _arg_node(scat_sum, 0)
    ss_b = _arg_node(scat_sum, 1)
    if ss_a is None or ss_b is None:
        return None
    info_a = _slice_scatter_of(ss_a)
    info_b = _slice_scatter_of(ss_b)
    if info_a is None or info_b is None or info_a.dim != info_b.dim:
        return None

    if info_a.start > 0 and info_b.start == 0:
        neg_scatter, slice_scatter = info_a, info_b
    elif info_b.start > 0 and info_a.start == 0:
        neg_scatter, slice_scatter = info_b, info_a
    else:
        return None

    neg_src = neg_scatter.source
    hi_src = slice_scatter.source
    if neg_src.op != "call_function" or neg_src.target not in _NEG_TARGETS:
        return None
    lo_slice = _arg_node(neg_src, 0)
    if lo_slice is None:
        return None
    lo_info = _slice_of(lo_slice)
    if lo_info is None or lo_info.dim != info_a.dim or lo_info.start != 0:
        return None

    hi_info = _slice_of(hi_src)
    if hi_info is None or hi_info.dim != info_a.dim or hi_info.start != lo_info.end:
        return None
    if lo_info.source is not hi_info.source:
        return None
    mul_sin = lo_info.source

    if (
        mul_sin.op != "call_function"
        or mul_sin.target not in _MUL_TARGETS
        or len(mul_sin.args) != 2
        or len(mul_cos.args) != 2
    ):
        return None

    sin_lhs = _arg_node(mul_sin, 0)
    sin_rhs = _arg_node(mul_sin, 1)
    cos_lhs = _arg_node(mul_cos, 0)
    cos_rhs = _arg_node(mul_cos, 1)
    if sin_lhs is None or sin_rhs is None or cos_lhs is None or cos_rhs is None:
        return None

    dout: torch.fx.Node | None = None
    sin_node: torch.fx.Node | None = None
    cos_node: torch.fx.Node | None = None
    for maybe_dout, maybe_sin in ((sin_lhs, sin_rhs), (sin_rhs, sin_lhs)):
        for maybe_cos_dout, maybe_cos in ((cos_lhs, cos_rhs), (cos_rhs, cos_lhs)):
            if maybe_dout is maybe_cos_dout:
                dout = maybe_dout
                sin_node = maybe_sin
                cos_node = maybe_cos
                break
        if dout is not None:
            break
    if dout is None or sin_node is None or cos_node is None:
        return None

    return RopeMatch(
        x_node=dout,
        cos_node=cos_node,
        sin_node=sin_node,
        consumed=(
            mul_sin,
            lo_slice,
            hi_src,
            neg_src,
            info_a.base,
            info_b.base,
            info_a.node,
            info_b.node,
            scat_sum,
            mul_cos,
        ),
    )


def rewrite_rope_backward(graph: torch.fx.Graph) -> int:
    """Replace AOT-decomposed RoPE backward with alloy.rope_apply_backward."""
    count = 0
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target not in _ADD_TARGETS:
            continue
        match = _try_rope_backward(node)
        if match is None:
            continue
        with graph.inserting_after(node):
            new_node = graph.call_function(
                torch.ops.alloy.rope_apply_backward.default,
                args=(match.x_node, match.cos_node, match.sin_node),
            )
        node.replace_all_uses_with(new_node)
        graph.erase_node(node)
        for dead in reversed(match.consumed):
            if len(dead.users) == 0:
                graph.erase_node(dead)
        count += 1
    return count


def _try_rope_form1(
    add_node: torch.fx.Node, lhs: torch.fx.Node, rhs: torch.fx.Node
) -> RopeMatch | None:
    if lhs.target not in _MUL_TARGETS or rhs.target not in _MUL_TARGETS:
        return None

    if _has_cat(rhs):
        x_cos_mul, rot_sin_mul = lhs, rhs
    elif _has_cat(lhs):
        x_cos_mul, rot_sin_mul = rhs, lhs
    else:
        return None

    x_node: torch.fx.Node | None = None
    cos_node: torch.fx.Node | None = None
    for arg in x_cos_mul.args:
        if not isinstance(arg, torch.fx.Node):
            continue
        if cos_node is None and arg.op == "call_function" and arg.target in _COS_VIEW_TARGETS:
            cos_node = arg
        elif x_node is None:
            x_node = arg
        else:
            cos_node = arg
    if x_node is None or cos_node is None:
        return None

    cat_node: torch.fx.Node | None = None
    sin_node: torch.fx.Node | None = None
    for arg in rot_sin_mul.args:
        if not isinstance(arg, torch.fx.Node):
            continue
        if arg.target in _CAT_TARGETS:
            cat_node = arg
        else:
            sin_node = arg
    if cat_node is None or sin_node is None:
        return None

    cat_args = cat_node.args[0]
    if not isinstance(cat_args, (list, tuple)) or len(cat_args) != 2:
        return None
    neg_node = cat_args[0]
    x1_node = cat_args[1]
    if not isinstance(neg_node, torch.fx.Node) or neg_node.target not in _NEG_TARGETS:
        return None
    if not isinstance(x1_node, torch.fx.Node) or x1_node.target not in _SLICE_TARGETS:
        return None
    x2_node = _arg_node(neg_node, 0)
    if x2_node is None or x2_node.target not in _SLICE_TARGETS:
        return None

    return RopeMatch(
        x_node=x_node,
        cos_node=cos_node,
        sin_node=sin_node,
        consumed=(x_cos_mul, rot_sin_mul, cat_node, neg_node, x1_node, x2_node, add_node),
    )


def _has_cat(mul_node: torch.fx.Node) -> bool:
    return any(
        isinstance(arg, torch.fx.Node) and arg.target in _CAT_TARGETS for arg in mul_node.args
    )


def _try_rope_form2(cat_node: torch.fx.Node) -> RopeMatch | None:
    cat_args = cat_node.args[0]
    if not isinstance(cat_args, (list, tuple)) or len(cat_args) != 2:
        return None
    cat_dim = cat_node.args[1] if len(cat_node.args) > 1 else cat_node.kwargs.get("dim", 0)
    if not isinstance(cat_dim, int):
        return None

    first_half = cat_args[0]
    second_half = cat_args[1]
    if not isinstance(first_half, torch.fx.Node) or not isinstance(second_half, torch.fx.Node):
        return None
    if first_half.target not in _SUB_TARGETS or second_half.target not in _ADD_TARGETS:
        return None

    sub_a = _arg_node(first_half, 0)
    sub_b = _arg_node(first_half, 1)
    add_a = _arg_node(second_half, 0)
    add_b = _arg_node(second_half, 1)
    if sub_a is None or sub_b is None or add_a is None or add_b is None:
        return None
    if any(node.target not in _MUL_TARGETS for node in (sub_a, sub_b, add_a, add_b)):
        return None

    sa0 = _arg_node(sub_a, 0)
    sa1 = _arg_node(sub_a, 1)
    sb0 = _arg_node(sub_b, 0)
    sb1 = _arg_node(sub_b, 1)
    aa0 = _arg_node(add_a, 0)
    aa1 = _arg_node(add_a, 1)
    ab0 = _arg_node(add_b, 0)
    ab1 = _arg_node(add_b, 1)
    if (
        sa0 is None
        or sa1 is None
        or sb0 is None
        or sb1 is None
        or aa0 is None
        or aa1 is None
        or ab0 is None
        or ab1 is None
    ):
        return None

    x1: torch.fx.Node | None = None
    x2: torch.fx.Node | None = None
    cos_node: torch.fx.Node | None = None
    sin_node: torch.fx.Node | None = None
    for maybe_x1, maybe_cos in ((sa0, sa1), (sa1, sa0)):
        if maybe_x1 in (ab0, ab1) and maybe_cos in (aa0, aa1):
            maybe_sin = ab1 if ab0 is maybe_x1 else ab0
            maybe_x2 = sb1 if sb0 is maybe_sin else sb0 if sb1 is maybe_sin else None
            if maybe_x2 is None:
                continue
            aa_other = aa1 if aa0 is maybe_cos else aa0
            if maybe_x2 is not aa_other:
                continue
            x1 = maybe_x1
            x2 = maybe_x2
            cos_node = maybe_cos
            sin_node = maybe_sin
            break

    if x1 is None or x2 is None or cos_node is None or sin_node is None:
        return None
    if x1.target not in _SLICE_TARGETS or x2.target not in _SLICE_TARGETS:
        return None
    x1_src = _arg_node(x1, 0)
    x2_src = _arg_node(x2, 0)
    if x1_src is None or x1_src is not x2_src:
        return None

    return RopeMatch(
        x_node=x1_src,
        cos_node=cos_node,
        sin_node=sin_node,
        consumed=(
            sub_a,
            sub_b,
            add_a,
            add_b,
            first_half,
            second_half,
            x1,
            x2,
            cat_node,
        ),
    )


@dataclass
class PartialRotaryMatch:
    """A partial-rotary RoPE split around a permuted per-head norm output."""

    permute_node: torch.fx.Node
    cat_node: torch.fx.Node
    rot_slice: torch.fx.Node
    pass_slice: torch.fx.Node
    rotary_dim: int


def _match_partial_rotary(rope_node: torch.fx.Node) -> PartialRotaryMatch | None:
    """Match Qwen3.5-style partial rotary fed by a permuted norm output:

        perm  = permute(getitem(rms,0), [...])      # (B, H, S, D)
        rot   = perm[..., 0:rotary_dim]
        roped = alloy.rope_apply(rot, cos, sin)
        pass_ = perm[..., rotary_dim:D]
        out   = cat([roped, pass_], dim=-1)

    Returns the permute, the cat to replace, both slices, and rotary_dim, or
    None if `rope_node`'s input isn't a leading last-dim slice of such a permute.
    """
    rot_slice = rope_node.args[0]
    if not isinstance(rot_slice, torch.fx.Node):
        return None
    if rot_slice.op != "call_function" or rot_slice.target is not torch.ops.aten.slice.Tensor:
        return None
    if len(rot_slice.args) < 4:
        return None
    perm, sdim, sstart, send = rot_slice.args[0], rot_slice.args[1], rot_slice.args[2], rot_slice.args[3]
    if not isinstance(perm, torch.fx.Node):
        return None
    if perm.op != "call_function" or perm.target is not torch.ops.aten.permute.default:
        return None
    val = perm.meta.get("val")
    if val is None or not hasattr(val, "shape"):
        return None
    ndim = len(val.shape)
    head_dim = int(val.shape[-1])
    last = ndim - 1
    if sdim not in (last, -1) or sstart != 0:
        return None
    if not isinstance(send, int) or send <= 0 or send >= head_dim:
        return None
    rotary_dim = send
    # permute must fan out to exactly the rope slice + a pass-through slice
    if len(perm.users) != 2:
        return None
    pass_candidates = [u for u in perm.users if u is not rot_slice]
    if len(pass_candidates) != 1:
        return None
    pass_slice = pass_candidates[0]
    if pass_slice.op != "call_function" or pass_slice.target is not torch.ops.aten.slice.Tensor:
        return None
    if len(pass_slice.args) < 4 or pass_slice.args[0] is not perm:
        return None
    if pass_slice.args[1] not in (last, -1) or pass_slice.args[2] != rotary_dim:
        return None
    # roped output and pass-through must recombine in a single cat([roped, pass])
    if len(rope_node.users) != 1:
        return None
    cat_node = next(iter(rope_node.users))
    if cat_node.op != "call_function" or cat_node.target not in _CAT_TARGETS:
        return None
    cat_inputs = cat_node.args[0]
    if not isinstance(cat_inputs, (list, tuple)) or len(cat_inputs) != 2:
        return None
    if cat_inputs[0] is not rope_node or cat_inputs[1] is not pass_slice:
        return None
    cat_dim = cat_node.args[1] if len(cat_node.args) > 1 else 0
    if cat_dim not in (last, -1):
        return None
    return PartialRotaryMatch(perm, cat_node, rot_slice, pass_slice, rotary_dim)


def rewrite_rms_norm_rope(graph: torch.fx.Graph) -> int:
    """Fuse alloy.rms_norm -> getitem(0) -> permute -> alloy.rope_apply.

    Qwen3 emits this chain for per-head q_norm/k_norm before rotary:
        normed_4d = alloy.rms_norm(x_4d, w, eps)[0]   # (B, S, H, D)
        permuted  = normed_4d.permute(0, 2, 1, 3)     # (B, H, S, D)
        roped     = alloy.rope_apply(permuted, cos, sin)
    Since rms_norm only touches the last dim D, permute commutes with it,
    so this is equivalent to:
        x_perm    = x_4d.permute(0, 2, 1, 3)
        roped     = alloy.rms_norm_rope(x_perm, w, cos, sin, eps)
    The new permute is a zero-cost stride view; the win is collapsing two
    Metal dispatches into one fused kernel.

    Also fires for the no-permute variant where rope_apply consumes the
    normed tensor directly.
    """
    count = 0
    for node in list(graph.nodes):
        if node.op != "call_function":
            continue
        if node.target is not torch.ops.alloy.rope_apply.default:
            continue
        if len(node.args) != 3:
            continue
        rope_input = node.args[0]
        if not isinstance(rope_input, torch.fx.Node):
            continue

        permute_node: torch.fx.Node | None = None
        partial: PartialRotaryMatch | None = None
        normed = rope_input
        if normed.op == "call_function" and normed.target is torch.ops.aten.slice.Tensor:
            # Partial rotary: rope consumes a leading last-dim slice of the
            # permuted norm output; the tail passes through to a cat.
            partial = _match_partial_rotary(node)
            if partial is None:
                continue
            permute_node = partial.permute_node
            normed = permute_node.args[0]
            if not isinstance(normed, torch.fx.Node):
                continue
        elif normed.op == "call_function" and normed.target is torch.ops.aten.permute.default:
            permute_node = normed
            if len(permute_node.args) != 2:
                continue
            normed_candidate = permute_node.args[0]
            if not isinstance(normed_candidate, torch.fx.Node):
                continue
            if len(permute_node.users) != 1:
                continue
            normed = normed_candidate

        if normed.op != "call_function" or normed.target is not operator.getitem:
            continue
        if len(normed.args) != 2 or normed.args[1] != 0:
            continue
        rms = normed.args[0]
        if not isinstance(rms, torch.fx.Node):
            continue
        if rms.op != "call_function" or rms.target is not torch.ops.alloy.rms_norm.default:
            continue
        if len(normed.users) != 1:
            continue
        rsqrt_consumers = [
            user for user in rms.users
            if user is not normed and not (
                user.op == "call_function" and user.target is operator.getitem
                and len(user.args) == 2 and user.args[1] == 1 and len(user.users) == 0
            )
        ]
        if rsqrt_consumers:
            continue

        x_node, weight_node, eps_arg = rms.args
        cos_node, sin_node = node.args[1], node.args[2]

        with graph.inserting_after(node):
            if permute_node is not None:
                permute_args = (x_node,) + tuple(permute_node.args[1:])
                x_for_fused = graph.call_function(
                    torch.ops.aten.permute.default, args=permute_args
                )
                with graph.inserting_after(x_for_fused):
                    fused = graph.call_function(
                        torch.ops.alloy.rms_norm_rope.default,
                        args=(x_for_fused, weight_node, cos_node, sin_node, eps_arg),
                    )
            else:
                fused = graph.call_function(
                    torch.ops.alloy.rms_norm_rope.default,
                    args=(x_node, weight_node, cos_node, sin_node, eps_arg),
                )

        if partial is not None:
            # The fused op already emits the full head (roped band + normalized
            # pass-through), so it replaces the cat. Tear down cat -> slices ->
            # rope -> permute in dependency order.
            partial.cat_node.replace_all_uses_with(fused)
            graph.erase_node(partial.cat_node)
            if len(partial.pass_slice.users) == 0:
                graph.erase_node(partial.pass_slice)
            graph.erase_node(node)
            if len(partial.rot_slice.users) == 0:
                graph.erase_node(partial.rot_slice)
            if permute_node is not None and len(permute_node.users) == 0:
                graph.erase_node(permute_node)
        else:
            node.replace_all_uses_with(fused)
            graph.erase_node(node)
            if permute_node is not None and len(permute_node.users) == 0:
                graph.erase_node(permute_node)
        # Erase the getitem(0) and any unused getitem(1)/rsqrt sibling, then
        # erase rms itself if nothing left consumes it. Skip a node if it's
        # already been removed earlier in this loop (e.g. normed appearing
        # twice in rms.users iteration order).
        for dead_user in list(rms.users):
            if dead_user.graph is graph and len(dead_user.users) == 0:
                graph.erase_node(dead_user)
        if rms.graph is graph and len(rms.users) == 0:
            graph.erase_node(rms)
        count += 1
    return count


def rewrite_rope_halve_self_cat(graph: torch.fx.Graph) -> int:
    """Drop the rotate_half self-concat feeding cos/sin.

    HF rotary builds emb = cat([freqs, freqs], -1), then cos/sin over emb, so the
    cos/sin table's two halves are identical. rms_norm_rope reads both halves, so
    the duplication is pure waste: a `cat` (k_concat_2) dispatch plus double-width
    cos/sin. Rewrite cos/sin to read `freqs` directly (half width) and flag the
    consuming rms_norm_rope ops cos_half=True — the kernel then reads the half
    table (stride HALF_ROT, the second half re-reads the first), bit-identical.

    With the cat gone, the freqs `mul` (inv_freq*pos) feeds cos/sin directly, so
    the sibling multi-root fusion folds mul+cos+sin into one dispatch.

    Gated to forward inference: fires only when every cos/sin consumer is an
    alloy.rms_norm_rope, so training graphs (rope backward reads the full table)
    stay full-width and correct.
    """
    cos_sin_targets = (torch.ops.aten.cos.default, torch.ops.aten.sin.default)
    rope_target = torch.ops.alloy.rms_norm_rope.default
    count = 0
    for cat_node in list(graph.nodes):
        if cat_node.op != "call_function" or cat_node.target not in _CAT_TARGETS:
            continue
        cat_args = cat_node.args
        lst = cat_args[0] if cat_args else None
        dim = cat_args[1] if len(cat_args) > 1 else 0
        if not (isinstance(lst, (list, tuple)) and len(lst) == 2 and lst[0] is lst[1]):
            continue  # not a self-concat
        if dim != -1:
            continue  # only the rotate_half last-dim duplication
        freqs = lst[0]
        if not isinstance(freqs, torch.fx.Node):
            continue
        users = list(cat_node.users)
        if not users or any(
            u.op != "call_function" or u.target not in cos_sin_targets for u in users
        ):
            continue
        # Every cos/sin output must feed (optionally through one unsqueeze/view)
        # only rms_norm_rope ops — otherwise halving corrupts another consumer.
        rope_ops: list[torch.fx.Node] = []
        ok = True
        for cs in users:
            for mid in cs.users:
                is_rope = mid.op == "call_function" and mid.target is rope_target
                terminals = [mid] if is_rope else list(mid.users)
                for t in terminals:
                    if t.op != "call_function" or t.target is not rope_target:
                        ok = False
                        break
                    rope_ops.append(t)
                if not ok:
                    break
            if not ok:
                break
        if not ok:
            continue
        for cs in users:
            cs.replace_input_with(cat_node, freqs)
        for rope in set(rope_ops):
            if len(rope.args) == 5:
                rope.args = (*rope.args, True)
        if len(cat_node.users) == 0:
            graph.erase_node(cat_node)
        count += 1
    return count


# ── rope cos/sin table → single alloy.rope_table op ──

_BMM_TARGET = torch.ops.aten.bmm.default
_COS_TARGET = torch.ops.aten.cos.default
_SIN_TARGET = torch.ops.aten.sin.default
_ARANGE_TARGETS = {torch.ops.aten.arange.start_step, torch.ops.aten.arange.default}
_RT_SHAPE_TARGETS = {
    torch.ops.aten.view.default,
    torch.ops.aten.expand.default,
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.permute.default,
    torch.ops.aten._to_copy.default,
    torch.ops.aten.reshape.default,
}


def _rt_strip(node: torch.fx.Node) -> torch.fx.Node:
    """Strip view/expand/unsqueeze/permute/cast wrappers down to the source."""
    cur = node
    while cur.op == "call_function" and cur.target in _RT_SHAPE_TARGETS:
        nxt = _arg_node(cur, 0)
        if nxt is None:
            break
        cur = nxt
    return cur


def _rt_subtree_has_arange(node: torch.fx.Node, depth: int = 0) -> bool:
    if depth > 14:
        return False
    if node.op == "call_function" and node.target in _ARANGE_TARGETS:
        return True
    return any(_rt_subtree_has_arange(a, depth + 1) for a in node.all_input_nodes)


def _rt_trace_bmm(x: torch.fx.Node) -> torch.fx.Node | None:
    cur: torch.fx.Node | None = x
    while cur is not None and cur.op == "call_function" and cur.target in _RT_SHAPE_TARGETS:
        cur = _arg_node(cur, 0)
    if cur is not None and cur.op == "call_function" and cur.target is _BMM_TARGET:
        return cur
    return None


def rewrite_rope_table(graph: torch.fx.Graph) -> int:
    """Collapse HF's rotary cos/sin table into one alloy.rope_table op.

    Matches `cos(X)` / `sin(X)` sharing `X = permute(view(bmm(inv_freq, positions)))`
    where `positions = arange(0, M) + cache_position`, and replaces both with
    `alloy.rope_table(cache_position, inv_freq, M)` — one kernel computes
    `cos/sin(float(m + cache_position) · inv_freq[j])`. Runs after
    rope.halve_self_cat, so the self-cat is already gone (cos reads `permute`,
    half-table); the consuming rms_norm_rope already has cos_half=True.
    """
    count = 0
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target is not _COS_TARGET:
            continue
        cos = node
        x = _arg_node(cos, 0)
        if x is None:
            continue
        sin = next(
            (u for u in x.users if u.op == "call_function" and u.target is _SIN_TARGET), None
        )
        if sin is None:
            continue
        bmm = _rt_trace_bmm(x)
        if bmm is None:
            continue
        a = _arg_node(bmm, 0)
        b = _arg_node(bmm, 1)
        if a is None or b is None:
            continue
        if _rt_subtree_has_arange(a):
            pos_side, invf_side = a, b
        elif _rt_subtree_has_arange(b):
            pos_side, invf_side = b, a
        else:
            continue
        positions = _rt_strip(pos_side)  # the `add` = arange + cache_position
        inv_freq = _rt_strip(invf_side)  # the inv_freq buffer
        if positions.op != "call_function" or positions.target not in _ADD_TARGETS:
            continue
        arange_node = next(
            (o for o in positions.all_input_nodes if o.target in _ARANGE_TARGETS), None
        )
        cache_position = next(
            (o for o in positions.all_input_nodes if o.target not in _ARANGE_TARGETS), None
        )
        if arange_node is None or cache_position is None:
            continue
        # arange must start at 0 (pos = m + cache_position assumes it)
        if arange_node.target is torch.ops.aten.arange.start_step and arange_node.args and arange_node.args[0] != 0:
            continue
        m_val = positions.meta.get("val")
        if m_val is None or m_val.ndim < 1:
            continue
        seq_len = int(m_val.shape[-1])

        with graph.inserting_before(cos):
            rt = graph.call_function(
                torch.ops.alloy.rope_table.default, (cache_position, inv_freq, seq_len)
            )
            gi_cos = graph.call_function(operator.getitem, (rt, 0))
            gi_sin = graph.call_function(operator.getitem, (rt, 1))
        cos.replace_all_uses_with(gi_cos)
        sin.replace_all_uses_with(gi_sin)
        for dead in (cos, sin):
            if len(dead.users) == 0:
                graph.erase_node(dead)
        count += 1
    return count
