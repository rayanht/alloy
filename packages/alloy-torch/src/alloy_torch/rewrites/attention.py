"""Attention rewrite passes for Alloy torch FX graphs."""

from __future__ import annotations

import operator

import alloy_torch.custom_ops  # noqa: F401
import torch
import torch.fx
from torch._ops import OpOverload


_VIEW_TARGETS = {
    torch.ops.aten.view.default,
    torch.ops.aten._unsafe_view.default,
    torch.ops.aten.reshape.default,
}
_TRANSPOSE_TARGETS = {torch.ops.aten.transpose.int}
_SDPA_TARGETS = {
    torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default,
    torch.ops.aten.scaled_dot_product_attention.default,
}
_WHERE_TARGETS = {torch.ops.aten.where.self}
_SCALAR_TENSOR_TARGETS = {torch.ops.aten.scalar_tensor.default}
_LE_TARGETS = {torch.ops.aten.le.Tensor, torch.ops.aten.le.Scalar}
_INDEX_PUT_TARGETS = {torch.ops.aten.index_put.default}
_INDEX_COPY_TARGETS: set[OpOverload | str] = {
    torch.ops.aten.index_copy.default,
    torch.ops.aten.index_copy_.default,
    "index_copy",
    "index_copy_",
}
_CLONE_TARGETS = {torch.ops.aten.clone.default}
_TO_TARGETS = {
    torch.ops.aten._to_copy.default,
    torch.ops.aten.to.dtype,
}
_VIEW_ONLY_TARGETS = frozenset(
    {
        torch.ops.aten.view.default,
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.reshape.default,
    }
)


def _arg_node(node: torch.fx.Node, index: int) -> torch.fx.Node | None:
    if len(node.args) <= index:
        return None
    arg = node.args[index]
    return arg if isinstance(arg, torch.fx.Node) else None


def _tensor_shape(node: torch.fx.Node) -> tuple[int, ...] | None:
    value = node.meta.get("val")
    if not isinstance(value, torch.Tensor):
        return None
    return tuple(int(dim) for dim in value.shape)


def _is_single_token_decode(q_node: torch.fx.Node, pos_node: torch.fx.Node) -> bool:
    q_shape = _tensor_shape(q_node)
    if q_shape is not None and (len(q_shape) != 4 or q_shape[2] != 1):
        return False

    pos_shape = _tensor_shape(pos_node)
    if pos_shape is None:
        return True
    pos_count = 1
    for dim in pos_shape:
        pos_count *= dim
    return pos_count == 1


_MAX_MULTI_TOKEN_DECODE = 16


def _decode_chunk_size(q_node: torch.fx.Node, pos_node: torch.fx.Node) -> int | None:
    """Return K for a [1, _MAX_MULTI_TOKEN_DECODE] multi-token decode pattern,
    or None if the shape doesn't match a fast-path-able decode.

    Specifically: Q is (B, H, K, D) and cache_position has exactly K entries.
    K=1 is the single-token decode case; K > 1 is speculative-decode verify.
    Large K (prefill, K > _MAX_MULTI_TOKEN_DECODE) falls back to the standard
    SDPA path, since our multi-token kernel padding to BLOCK_M = ceil(K/8)*8
    becomes inefficient at large K.
    """
    q_shape = _tensor_shape(q_node)
    pos_shape = _tensor_shape(pos_node)
    if q_shape is None:
        # No shape metadata — fall through to the single-token check, which
        # accepts shape-less nodes as single-token by default.
        return 1 if _is_single_token_decode(q_node, pos_node) else None
    if len(q_shape) != 4:
        return None
    k = q_shape[2]
    if not (1 <= k <= _MAX_MULTI_TOKEN_DECODE):
        return None
    if pos_shape is None:
        return k
    pos_count = 1
    for dim in pos_shape:
        pos_count *= dim
    if pos_count != k:
        return None
    return k


def rewrite_gqa_expansion(graph: torch.fx.Graph) -> int:
    """Collapse GQA unsqueeze->expand->clone->view chains on SDPA K/V inputs."""
    sdpa_targets = {torch.ops.aten.scaled_dot_product_attention.default}
    if hasattr(torch.ops.aten, "_scaled_dot_product_flash_attention_for_cpu"):
        sdpa_targets.add(torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default)

    count = 0
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target not in sdpa_targets:
            continue
        k_node = _arg_node(node, 1)
        v_node = _arg_node(node, 2)
        if k_node is None or v_node is None:
            continue

        k_result = _unwrap_gqa(k_node)
        v_result = _unwrap_gqa(v_node)
        if k_result is None or v_result is None:
            continue
        k_src, k_group, k_chain = k_result
        v_src, v_group, v_chain = v_result
        if k_group != v_group:
            continue

        new_args = list(node.args)
        new_args[1] = k_src
        new_args[2] = v_src
        node.args = tuple(new_args)
        node.kwargs = {**node.kwargs, "_kv_group": k_group}

        view_k = k_chain[0]
        view_v = v_chain[0]
        for end_node in (view_k, view_v):
            src = k_src if end_node is view_k else v_src
            for user in list(end_node.users):
                if user.op == "output":
                    user.replace_input_with(end_node, src)

        for chain_node in k_chain + v_chain:
            if len(chain_node.users) == 0:
                graph.erase_node(chain_node)

        count += 1

    return count


def rewrite_gqa_expansion_backward(graph: torch.fx.Graph) -> int:
    """Drop redundant GQA-expanded dK/dV foldback after SDPA backward."""
    sdpa_backward_targets = {
        torch.ops.aten._scaled_dot_product_flash_attention_for_cpu_backward.default
    }
    if hasattr(torch.ops.aten, "_scaled_dot_product_flash_attention_backward"):
        sdpa_backward_targets.add(
            torch.ops.aten._scaled_dot_product_flash_attention_backward.default
        )

    count = 0
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target not in sdpa_backward_targets:
            continue
        for user in list(node.users):
            if user.op != "call_function" or user.target is not operator.getitem:
                continue
            if len(user.args) < 2 or user.args[1] not in (1, 2):
                continue
            view_node = next(iter(user.users), None)
            if view_node is None or view_node.target is not torch.ops.aten.view.default:
                continue
            if len(view_node.args) < 2:
                continue
            view_shape = view_node.args[1]
            if not (isinstance(view_shape, (list, tuple)) and len(view_shape) == 5):
                continue
            sum_node = next(iter(view_node.users), None)
            if sum_node is None or sum_node.target is not torch.ops.aten.sum.dim_IntList:
                continue
            sum_dims = sum_node.args[1] if len(sum_node.args) >= 2 else None
            if not (
                isinstance(sum_dims, (list, tuple)) and len(sum_dims) == 1 and sum_dims[0] == 2
            ):
                continue
            squeeze_node = next(iter(sum_node.users), None)
            if squeeze_node is None or squeeze_node.target is not torch.ops.aten.squeeze.dims:
                continue
            squeeze_dims = squeeze_node.args[1] if len(squeeze_node.args) >= 2 else None
            if not (
                isinstance(squeeze_dims, (list, tuple))
                and len(squeeze_dims) == 1
                and squeeze_dims[0] == 2
            ):
                continue
            squeeze_node.replace_all_uses_with(user)
            for dead in (squeeze_node, sum_node, view_node):
                if len(dead.users) == 0:
                    graph.erase_node(dead)
            count += 1
    return count


def _unwrap_gqa(kv_node: torch.fx.Node) -> tuple[torch.fx.Node, int, list[torch.fx.Node]] | None:
    """Walk back view->clone->expand->unsqueeze."""
    chain: list[torch.fx.Node] = []
    current = kv_node
    for expected in (
        torch.ops.aten.view.default,
        torch.ops.aten.clone.default,
        torch.ops.aten.expand.default,
        torch.ops.aten.unsqueeze.default,
    ):
        if current.op != "call_function" or current.target != expected:
            return None
        chain.append(current)
        next_node = _arg_node(current, 0)
        if next_node is None:
            return None
        current = next_node

    expand_node = chain[2]
    unsqueeze_node = chain[3]

    if len(unsqueeze_node.args) < 2 or unsqueeze_node.args[1] != 2:
        return None
    if len(expand_node.args) < 2:
        return None
    expand_shape = expand_node.args[1]
    if not isinstance(expand_shape, (list, tuple)) or len(expand_shape) != 5:
        return None
    kv_group = expand_shape[2]
    if not isinstance(kv_group, int) or kv_group < 2:
        return None

    return current, kv_group, chain


def rewrite_strip_bhsd_flatten(graph: torch.fx.Graph) -> int:
    """Remove view(B*H, S, D) after transpose when attention can consume BHSD strides."""
    count = 0
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target not in _VIEW_TARGETS:
            continue
        if len(node.args) < 2:
            continue
        src = _arg_node(node, 0)
        if src is None or src.target not in _TRANSPOSE_TARGETS:
            continue
        target_shape = node.args[1]
        src_shape = src.meta.get("val")
        if src_shape is None or not hasattr(src_shape, "shape"):
            continue
        src_dims = src_shape.shape
        if len(src_dims) != 4:
            continue
        if not isinstance(target_shape, (list, tuple)) or len(target_shape) != 3:
            continue
        batch, heads, seq_len, dim = src_dims
        if list(target_shape) != [batch * heads, seq_len, dim]:
            continue
        node.replace_all_uses_with(src)
        graph.erase_node(node)
        count += 1
    return count


def rewrite_causal_mask_to_is_causal(graph: torch.fx.Graph) -> int:
    """Replace where(le_causal_mask, 0, -inf) feeding SDPA with is_causal=True."""
    count = 0
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target not in _WHERE_TARGETS:
            continue
        if len(node.args) < 3:
            continue
        cond = _arg_node(node, 0)
        true_val = node.args[1]
        false_val = node.args[2]
        if cond is None:
            continue
        true_node = true_val if isinstance(true_val, torch.fx.Node) else None
        false_node = false_val if isinstance(false_val, torch.fx.Node) else None
        if true_node is None or false_node is None:
            continue
        if not (_is_scalar_node(true_node, 0.0) and _is_scalar_node(false_node, float("-inf"))):
            continue
        if not _traces_to_le(cond):
            continue

        cond_meta = cond.meta.get("val")
        if cond_meta is not None and hasattr(cond_meta, "shape"):
            shape = cond_meta.shape
            if len(shape) >= 2 and shape[-2] != shape[-1]:
                continue

        users = list(node.users)
        if len(users) != 1:
            continue
        sdpa_node = users[0]
        if sdpa_node.op != "call_function" or sdpa_node.target not in _SDPA_TARGETS:
            continue
        if sdpa_node.kwargs.get("attn_mask") is not node:
            continue

        new_kwargs = dict(sdpa_node.kwargs)
        del new_kwargs["attn_mask"]
        new_kwargs["is_causal"] = True
        sdpa_node.kwargs = new_kwargs
        count += 1

    return count


def _traces_to_le(node: torch.fx.Node) -> bool:
    skip_targets = {torch.ops.aten.expand.default, torch.ops.aten.unsqueeze.default}
    current = node
    for _ in range(12):
        if current.target in _LE_TARGETS:
            return True
        if current.target in skip_targets:
            next_node = _arg_node(current, 0)
            if next_node is None:
                return False
            current = next_node
        else:
            return False
    return False


def _is_scalar_node(node: torch.fx.Node, value: float) -> bool:
    if node.target not in _SCALAR_TENSOR_TARGETS or not node.args:
        return False
    scalar_value = node.args[0]
    if isinstance(scalar_value, int | float):
        if value != value:
            return isinstance(scalar_value, float) and scalar_value != scalar_value
        return float(scalar_value) == value
    return False


# ---------------------------------------------------------------------------
# rewrite_eager_attention_to_sdpa — collapse the AOT-decomposed
# matmul(Q, K^T) → softmax → matmul(attn, V) chain into one
# aten.scaled_dot_product_attention node.
#
# Triggered by nomic-bert (and any model whose attention head is the textbook
# `eager_attention_forward`): AOT lowers each 4D matmul to
# `view(clone(expand(_)))→bmm→view` which produces strided_copy_4d at runtime
# (3 copies per matmul × 2 matmuls per layer = ~6 redundant copies per
# attention block). Replacing with SDPA lets the alloy attention kernel
# consume the 4D Q/K/V directly with no contiguify.
# ---------------------------------------------------------------------------

_BMM_TARGETS = {torch.ops.aten.bmm.default}
_PERMUTE_TARGETS = {torch.ops.aten.permute.default, torch.ops.aten.transpose.int}
_ADD_TARGETS = {torch.ops.aten.add.Tensor}
_MUL_TARGETS = {torch.ops.aten.mul.Tensor, torch.ops.aten.mul.Scalar}
_EXPAND_TARGETS = {torch.ops.aten.expand.default}
_SOFTMAX_TARGETS = {torch.ops.aten._softmax.default}


def _unwrap_expand_clone_view_to_4d(
    node: torch.fx.Node,
) -> tuple[torch.fx.Node, tuple[int, ...]] | None:
    """`view(clone(expand(x_4d)))` or `view(expand(x_4d))` → (x_4d, expand_shape).
    Returns None if the pattern doesn't match.

    The clone is optional because AOT elides it when expand is an identity
    (e.g. batch=1 expand from (1, H, S, D) to (1, H, S, D) doesn't need a
    contiguify). Shape comes from the expand's FX meta — alloy custom ops
    like rope_apply may not propagate meta to their outputs."""
    if node.target not in _VIEW_TARGETS:
        return None
    mid = _arg_node(node, 0)
    if mid is None:
        return None
    if mid.target in _CLONE_TARGETS:
        expand = _arg_node(mid, 0)
    else:
        expand = mid
    if expand is None or expand.target not in _EXPAND_TARGETS:
        return None
    expand_shape = _tensor_shape(expand)
    if expand_shape is None or len(expand_shape) != 4:
        return None
    inner = _arg_node(expand, 0)
    if inner is None:
        return None
    return inner, expand_shape


def _maybe_strip_permute(node: torch.fx.Node) -> tuple[torch.fx.Node, list[int] | None]:
    """If node is a permute/transpose of a 4D tensor, return (inner_node, perm).
    Otherwise return (node, None)."""
    if node.target in _PERMUTE_TARGETS:
        inner = _arg_node(node, 0)
        if inner is None:
            return node, None
        if node.target == torch.ops.aten.transpose.int:
            a, b = node.args[1], node.args[2]
            if not (isinstance(a, int) and isinstance(b, int)):
                return node, None
            inner_shape = _tensor_shape(inner)
            if inner_shape is None:
                return node, None
            perm = list(range(len(inner_shape)))
            perm[a], perm[b] = perm[b], perm[a]
            return inner, perm
        # aten.permute.default: explicit permutation list
        perm = node.args[1]
        if not isinstance(perm, (list, tuple)):
            return node, None
        return inner, [int(p) for p in perm]
    return node, None


def _scalar_from_mul(mul_node: torch.fx.Node) -> tuple[torch.fx.Node, float] | None:
    """Extract `(tensor_arg, scalar)` from `mul(tensor, scalar)`. Returns None
    when no scalar arg present."""
    if mul_node.target not in _MUL_TARGETS or len(mul_node.args) < 2:
        return None
    a, b = mul_node.args[0], mul_node.args[1]
    if isinstance(a, torch.fx.Node) and isinstance(b, (int, float)):
        return a, float(b)
    if isinstance(b, torch.fx.Node) and isinstance(a, (int, float)):
        return b, float(a)
    return None


def rewrite_eager_attention_to_sdpa(graph: torch.fx.Graph) -> int:
    """Find `_softmax → bmm → view → permute` attention chains and replace
    with aten.scaled_dot_product_attention. Patterns left alone if Q/K/V or
    output reshape don't match the canonical AOT decomposition exactly."""
    count = 0
    for softmax in list(graph.nodes):
        if softmax.op != "call_function" or softmax.target not in _SOFTMAX_TARGETS:
            continue

        # --- back-trace softmax input → bmm(Q_view, K_view) → Q, K, scale, mask
        sm_in = _arg_node(softmax, 0)
        if sm_in is None:
            continue
        mask_node: torch.fx.Node | None = None
        scaled = sm_in
        if sm_in.target in _ADD_TARGETS:
            lhs = _arg_node(sm_in, 0)
            rhs = _arg_node(sm_in, 1)
            if lhs is None or rhs is None:
                continue
            # The branch with the bmm chain is the scaled scores; the other
            # branch is the mask.
            scaled, mask_node = lhs, rhs
            if scaled.target not in _MUL_TARGETS:
                scaled, mask_node = rhs, lhs
        if scaled.target not in _MUL_TARGETS:
            continue
        scaled_pair = _scalar_from_mul(scaled)
        if scaled_pair is None:
            continue
        qkt_view, scale = scaled_pair
        if qkt_view.target not in _VIEW_TARGETS:
            continue
        qkt_bmm = _arg_node(qkt_view, 0)
        if qkt_bmm is None or qkt_bmm.target not in _BMM_TARGETS:
            continue
        q_view = _arg_node(qkt_bmm, 0)
        k_view = _arg_node(qkt_bmm, 1)
        if q_view is None or k_view is None:
            continue
        q_unwrap = _unwrap_expand_clone_view_to_4d(q_view)
        k_unwrap = _unwrap_expand_clone_view_to_4d(k_view)
        if q_unwrap is None or k_unwrap is None:
            continue
        q_4d, q_shape = q_unwrap
        k_inner_raw, k_inner_shape = k_unwrap
        # K side carries the (..., 1, 3, 2) transpose for K^T — peel it.
        k_4d, k_perm = _maybe_strip_permute(k_inner_raw)
        if k_perm is not None and k_perm != [0, 1, 3, 2]:
            continue

        # --- forward-trace softmax → clone → expand → view → bmm(attn_view, V_view)
        if len(softmax.users) != 1:
            continue
        attn_clone = next(iter(softmax.users))
        if attn_clone.target not in _CLONE_TARGETS:
            continue
        expand_users = list(attn_clone.users)
        if len(expand_users) != 1 or expand_users[0].target not in _EXPAND_TARGETS:
            continue
        attn_expand = expand_users[0]
        view_users = list(attn_expand.users)
        if len(view_users) != 1 or view_users[0].target not in _VIEW_TARGETS:
            continue
        attn_view = view_users[0]
        attn_bmm_users = [u for u in attn_view.users if u.target in _BMM_TARGETS]
        if len(attn_bmm_users) != 1:
            continue
        attn_bmm = attn_bmm_users[0]
        v_view = (
            _arg_node(attn_bmm, 1)
            if _arg_node(attn_bmm, 0) is attn_view
            else _arg_node(attn_bmm, 0)
        )
        if v_view is None:
            continue
        v_unwrap = _unwrap_expand_clone_view_to_4d(v_view)
        if v_unwrap is None:
            continue
        v_4d, v_shape = v_unwrap

        # --- output reshape: bmm → view(B,H,S,D) — replace this view with the
        # SDPA result so downstream permute/clone/view stays untouched.
        out_users = list(attn_bmm.users)
        if len(out_users) != 1 or out_users[0].target not in _VIEW_TARGETS:
            continue
        out_view = out_users[0]

        # Shape sanity from the expand outputs we already gathered.
        # k_inner_shape is the K^T shape (B,H,D,S); the K passed to SDPA is
        # the un-transposed (B,H,S,D) — same as v_shape, so use that.
        if q_shape[:2] != v_shape[:2] or q_shape[3] != v_shape[3]:
            continue

        # --- splice in SDPA
        with graph.inserting_before(out_view):
            sdpa = graph.call_function(
                torch.ops.aten.scaled_dot_product_attention.default,
                args=(q_4d, k_4d, v_4d),
                kwargs={"attn_mask": mask_node, "scale": scale, "is_causal": False},
            )
        # Propagate output shape meta for downstream rewrites that read it.
        out_meta = out_view.meta.get("val")
        if out_meta is not None:
            sdpa.meta["val"] = out_meta
        out_view.replace_all_uses_with(sdpa)
        count += 1
    return count


def _trace_kv_cache_source(
    kv_input: torch.fx.Node,
) -> tuple[torch.fx.Node, torch.fx.Node, torch.fx.Node, torch.fx.Node] | None:
    """Trace back from attention K/V input to a StaticCache update op."""
    current = kv_input
    while current.target in _VIEW_ONLY_TARGETS or current.target in _TO_TARGETS:
        next_node = _arg_node(current, 0)
        if next_node is None:
            return None
        current = next_node

    if current.target in _CLONE_TARGETS:
        clone_source = _arg_node(current, 0)
        if clone_source is None:
            return None
        current = clone_source

    while current.target in (torch.ops.aten.expand.default, torch.ops.aten.unsqueeze.default):
        next_node = _arg_node(current, 0)
        if next_node is None:
            return None
        current = next_node

    if current.target not in _INDEX_PUT_TARGETS:
        return _trace_index_copy_cache_source(current)

    cache_node = _arg_node(current, 0)
    source_node = _arg_node(current, 2)
    if cache_node is None or source_node is None or len(current.args) < 2:
        return None
    indices = current.args[1]
    if not isinstance(indices, (list, tuple)) or len(indices) < 3:
        return None
    if indices[0] is not None or indices[1] is not None:
        return None
    pos_node = indices[2]
    if not isinstance(pos_node, torch.fx.Node):
        return None
    return current, cache_node, pos_node, source_node


def _trace_index_copy_cache_source(
    current: torch.fx.Node,
) -> tuple[torch.fx.Node, torch.fx.Node, torch.fx.Node, torch.fx.Node] | None:
    if current.target not in _INDEX_COPY_TARGETS or len(current.args) < 4:
        return None
    dim = current.args[1]
    if not isinstance(dim, int) or dim != 2:
        return None
    cache_node = _arg_node(current, 0)
    pos_node = _arg_node(current, 2)
    source_node = _arg_node(current, 3)
    if cache_node is None or pos_node is None or source_node is None:
        return None
    return current, cache_node, pos_node, source_node


def _unwrap_cache_node(cache_node: torch.fx.Node) -> torch.fx.Node:
    if cache_node.op == "call_function" and cache_node.target in _CLONE_TARGETS:
        clone_source = _arg_node(cache_node, 0)
        if clone_source is not None:
            return clone_source
    return cache_node


def rewrite_attention_kv_update(graph: torch.fx.Graph) -> int:
    """Replace index_put(K/V cache) + clone + attention with fused attention_kv_update."""
    count = 0
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target not in _SDPA_TARGETS:
            continue

        q_node = _arg_node(node, 0)
        k_input = _arg_node(node, 1)
        v_input = _arg_node(node, 2)
        if q_node is None or k_input is None or v_input is None:
            continue

        k_result = _trace_kv_cache_source(k_input)
        if k_result is None:
            continue
        k_iput, k_cache, k_pos, k_source = k_result

        v_result = _trace_kv_cache_source(v_input)
        if v_result is None:
            continue
        v_iput, v_cache, v_pos, v_source = v_result
        k_cache = _unwrap_cache_node(k_cache)
        v_cache = _unwrap_cache_node(v_cache)

        if k_pos is not v_pos:
            continue
        if k_cache.op != "placeholder" or v_cache.op != "placeholder":
            continue
        chunk = _decode_chunk_size(q_node, k_pos)
        if chunk is None:
            continue

        target_op = (
            torch.ops.alloy.attention_kv_update.default
            if chunk == 1
            else torch.ops.alloy.attention_kv_update_multi.default
        )
        with graph.inserting_after(node):
            new_node = graph.call_function(
                target_op,
                args=(q_node, k_source, v_source, k_pos, k_cache, v_cache),
            )

        for user in list(node.users):
            if (
                user.op == "call_function"
                and user.target is operator.getitem
                and len(user.args) >= 2
                and user.args[1] == 0
            ):
                user.replace_all_uses_with(new_node)
                if len(user.users) == 0:
                    graph.erase_node(user)

        if len(node.users) == 0:
            graph.erase_node(node)

        for index_put_node, cache_node in ((k_iput, k_cache), (v_iput, v_cache)):
            index_put_node.replace_all_uses_with(cache_node)
            if len(index_put_node.users) == 0:
                graph.erase_node(index_put_node)

        count += 1

    return count
