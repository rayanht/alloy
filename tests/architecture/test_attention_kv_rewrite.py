"""Rewrite coverage for decode-time KV cache updates."""

from __future__ import annotations

import operator

import torch

from alloy_torch.rewrites.attention import rewrite_attention_kv_update


def test_functional_cache_update_rewrites_to_fused_decode_attention() -> None:
    graph = torch.fx.Graph()
    q = graph.placeholder("q")
    new_k = graph.placeholder("new_k")
    new_v = graph.placeholder("new_v")
    cache_pos = graph.placeholder("cache_pos")
    k_cache = graph.placeholder("k_cache")
    v_cache = graph.placeholder("v_cache")
    q.meta["val"] = torch.empty((1, 16, 1, 128))
    cache_pos.meta["val"] = torch.empty((1,), dtype=torch.int64)

    k_clone = graph.call_function(torch.ops.aten.clone.default, args=(k_cache,))
    v_clone = graph.call_function(torch.ops.aten.clone.default, args=(v_cache,))
    k_updated = graph.call_function(
        torch.ops.aten.index_put.default,
        args=(k_clone, [None, None, cache_pos], new_k),
    )
    v_updated = graph.call_function(
        torch.ops.aten.index_put.default,
        args=(v_clone, [None, None, cache_pos], new_v),
    )
    k_view = graph.call_function(
        torch.ops.aten._to_copy.default,
        args=(k_updated,),
        kwargs={"dtype": torch.float32},
    )
    v_view = graph.call_function(
        torch.ops.aten._to_copy.default,
        args=(v_updated,),
        kwargs={"dtype": torch.float32},
    )
    attention = graph.call_function(
        torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default,
        args=(q, k_view, v_view),
        kwargs={"dropout_p": 0.0, "is_causal": False},
    )
    out = graph.call_function(operator.getitem, args=(attention, 0))
    graph.output(out)

    changed = rewrite_attention_kv_update(graph)
    graph.eliminate_dead_code()
    graph.lint()

    targets = tuple(node.target for node in graph.nodes if node.op == "call_function")
    assert changed == 1
    assert torch.ops.alloy.attention_kv_update.default in targets
    assert torch.ops.aten.index_copy.default not in targets
    assert torch.ops.aten.index_put.default not in targets
    assert torch.ops.aten.clone.default not in targets
    assert torch.ops.aten._to_copy.default not in targets
    assert torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default not in targets
