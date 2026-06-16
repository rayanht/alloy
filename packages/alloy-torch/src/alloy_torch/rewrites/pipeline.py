"""Ordered FX rewrite pipeline for the Alloy torch backend."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import torch
import torch.fx

from alloy import get_logger
from alloy_torch.rewrites import activation
from alloy_torch.rewrites import attention
from alloy_torch.rewrites import auto_functionalize
from alloy_torch.rewrites import cleanup
from alloy_torch.rewrites import dequant
from alloy_torch.rewrites import gemm
from alloy_torch.rewrites import loss
from alloy_torch.rewrites import norm
from alloy_torch.rewrites import rope

logger = get_logger("alloy_torch.rewrites")

RewriteFn = Callable[[torch.fx.Graph], int]


@dataclass(frozen=True)
class RewriteStats:
    name: str
    changed: int
    nodes_before: int
    nodes_after: int


@dataclass(frozen=True)
class RewritePass:
    name: str
    fn: RewriteFn

    def run(self, graph: torch.fx.Graph) -> RewriteStats:
        nodes_before = _node_count(graph)
        changed = self.fn(graph)
        nodes_after = _node_count(graph)
        return RewriteStats(
            name=self.name,
            changed=changed,
            nodes_before=nodes_before,
            nodes_after=nodes_after,
        )


def _node_count(graph: torch.fx.Graph) -> int:
    return sum(1 for _ in graph.nodes)


def _strip_orphaned_f16_downcasts(graph: torch.fx.Graph) -> int:
    count = 0
    to_targets = {torch.ops.aten._to_copy.default, torch.ops.aten.to.dtype}
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target not in to_targets:
            continue
        if node.kwargs.get("dtype") != torch.float16:
            continue
        replacement = cast(torch.fx.Node, node.args[0])
        node.replace_all_uses_with(replacement)
        graph.erase_node(node)
        count += 1
    return count


def _toposort_and_dce(graph: torch.fx.Graph) -> int:
    nodes_before = _node_count(graph)
    cleanup.topological_sort(graph)
    graph.eliminate_dead_code()
    return max(nodes_before - _node_count(graph), 0)


REWRITE_PIPELINE: tuple[RewritePass, ...] = (
    RewritePass(
        "auto_functionalize.unwrap",
        auto_functionalize.rewrite_unwrap_auto_functionalized,
    ),
    RewritePass("loss.cross_entropy_fwd", loss.rewrite_cross_entropy_fwd),
    RewritePass("loss.cross_entropy_bwd", loss.rewrite_cross_entropy_bwd),
    RewritePass("cleanup.fold_identities", cleanup.rewrite_fold_identities),
    RewritePass("cleanup.strip_f32_upcasts", cleanup.rewrite_strip_f32_upcasts),
    RewritePass(
        "cleanup.strip_lossless_roundtrip_cast",
        cleanup.rewrite_strip_lossless_roundtrip_cast,
    ),
    RewritePass("cleanup.simplify_views", cleanup.rewrite_simplify_views),
    RewritePass("gemm.gelu_tanh", activation.rewrite_gelu_tanh),
    RewritePass("dequant.dequant_mm", dequant.rewrite_dequant_mm),
    RewritePass("gemm.residual_layernorm", gemm.rewrite_gemm_residual_layernorm),
    RewritePass("norm.rms_norm", norm.rewrite_rms_norm),
    RewritePass("norm.rms_norm_backward", norm.rewrite_rms_norm_backward),
    RewritePass("gemm.residual_rmsnorm", gemm.rewrite_gemm_residual_rmsnorm),
    RewritePass("rope.forward", rope.rewrite_rope),
    RewritePass("cleanup.dce_pre_rms_rope_fusion", _toposort_and_dce),
    RewritePass("rope.rms_norm_rope", rope.rewrite_rms_norm_rope),
    RewritePass("rope.halve_self_cat", rope.rewrite_rope_halve_self_cat),
    RewritePass("rope.table", rope.rewrite_rope_table),
    RewritePass("rope.backward", rope.rewrite_rope_backward),
    RewritePass("cleanup.strip_f16_downcasts_pre_silu", _strip_orphaned_f16_downcasts),
    RewritePass(
        "attention.causal_mask_to_is_causal", attention.rewrite_causal_mask_to_is_causal
    ),
    RewritePass("attention.gqa_expansion", attention.rewrite_gqa_expansion),
    RewritePass(
        "attention.gqa_expansion_backward",
        attention.rewrite_gqa_expansion_backward,
    ),
    RewritePass("gemm.batched_mm_silu", gemm.rewrite_batched_mm_silu),
    RewritePass("gemm.gguf_mm_silu", gemm.rewrite_gguf_mm_silu),
    RewritePass("gemm.gguf_mm_gelu", gemm.rewrite_gguf_mm_gelu),
    RewritePass("gemm.batched_mm", gemm.rewrite_batched_mm),
    RewritePass(
        "attention.eager_to_sdpa", attention.rewrite_eager_attention_to_sdpa,
    ),
    RewritePass("attention.strip_bhsd_flatten", attention.rewrite_strip_bhsd_flatten),
    RewritePass("attention.kv_update", attention.rewrite_attention_kv_update),
    RewritePass("cleanup.toposort_dce_before_dequant_batching", _toposort_and_dce),
    RewritePass("dequant.dequant_mm_silu", dequant.rewrite_dequant_mm_silu),
    RewritePass("dequant.batched_dequant_mm", dequant.rewrite_batched_dequant_mm),
    RewritePass(
        "dequant.batched_gguf_q4_k_mm", dequant.rewrite_batched_gguf_q4_k_mm,
    ),
    RewritePass(
        "dequant.batched_gguf_q5_0_mm", dequant.rewrite_batched_gguf_q5_0_mm,
    ),
    RewritePass(
        "dequant.batched_gguf_q6_k_mm", dequant.rewrite_batched_gguf_q6_k_mm,
    ),
    RewritePass(
        "dequant.batched_gguf_q8_0_mm", dequant.rewrite_batched_gguf_q8_0_mm,
    ),
    RewritePass(
        "dequant.batched_mlx_q4_mm", dequant.rewrite_batched_mlx_q4_mm,
    ),
    RewritePass("cleanup.toposort_dce_final", _toposort_and_dce),
    RewritePass("cleanup.strip_f16_downcasts_final", _strip_orphaned_f16_downcasts),
)


def run_rewrite_pipeline(graph: torch.fx.Graph) -> tuple[RewriteStats, ...]:
    return tuple(rewrite_pass.run(graph) for rewrite_pass in REWRITE_PIPELINE)


def rewrite_fx_graph(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    t0 = time.perf_counter()
    n_nodes_before = sum(1 for _ in gm.graph.nodes)
    stats = run_rewrite_pipeline(gm.graph)
    gm.graph.eliminate_dead_code()
    gm.recompile()
    n_nodes_after = sum(1 for _ in gm.graph.nodes)
    per_pass = {stat.name: stat.changed for stat in stats if stat.changed}
    logger.debug(
        "rewrite_pipeline_complete",
        n_passes=len(stats),
        n_passes_matched=len(per_pass),
        total_matches=sum(per_pass.values()),
        n_nodes_before=n_nodes_before,
        n_nodes_after=n_nodes_after,
        took_ms=round((time.perf_counter() - t0) * 1000.0, 1),
        per_pass=per_pass,
    )
    return gm
