"""Tests for fusion analysis — _plan_fusion.

Pure analysis: no GPU, no compilation. Tests the DAG analysis that decides
which ops to fuse together.
"""

import numpy as np
import alloy as al
from alloy._dispatch.lazy import _collect_pending_ops
from alloy._dispatch.fusion_analysis import FusionKind, _plan_fusion
from tests.helpers import k_scale, k_bias, k_relu, get_unary_kernel


def _build_chain(*kernels, N=4096):
    """Queue a chain of elem kernels, return (list[LazyOp], list[AlloyBuffer])."""
    grid = (N + 1023) // 1024
    x = np.random.RandomState(42).randn(N).astype(np.float32)
    results = []
    prev = x
    for kfn in kernels:
        out = np.zeros(N, dtype=np.float32)
        r = kfn[grid](prev, out, N=N)
        results.append(r)
        prev = r
    ops, _ = _collect_pending_ops(tuple(results))
    return ops, results


class TestPlanFusion:
    def test_single_op_is_individual(self):
        ops, _ = _build_chain(k_scale)
        plans = _plan_fusion(ops)
        assert len(plans) == 1
        assert plans[0][1].kind == FusionKind.INDIVIDUAL

    def test_two_elem_form_chain(self):
        ops, _ = _build_chain(k_scale, k_bias)
        plans = _plan_fusion(ops)
        chain_plans = [p for _, p in plans if p.kind == FusionKind.ELEM_CHAIN]
        assert len(chain_plans) == 1
        assert len(chain_plans[0].chain) == 2

    def test_three_elem_form_single_chain(self):
        ops, _ = _build_chain(k_scale, k_bias, k_relu)
        plans = _plan_fusion(ops)
        chain_plans = [p for _, p in plans if p.kind == FusionKind.ELEM_CHAIN]
        assert len(chain_plans) == 1
        assert len(chain_plans[0].chain) == 3

    def test_anchor_epilogue(self):
        """dot → elem should produce ANCHOR plan with epilogue."""
        M, N, K = 64, 64, 64
        A = np.random.randn(M, K).astype(np.float32) * 0.1
        B = np.random.randn(K, N).astype(np.float32) * 0.1

        r1 = al.dot(A, B, BLOCK_M=32, BLOCK_N=32, BLOCK_K=16)
        out = np.zeros(M * N, dtype=np.float32)
        r2 = k_scale[((M * N + 1023) // 1024,)](r1, out, N=M * N)

        ops, _ = _collect_pending_ops((r2,))
        plans = _plan_fusion(ops)
        anchor_plans = [p for _, p in plans if p.kind == FusionKind.ANCHOR]
        assert len(anchor_plans) == 1
        assert anchor_plans[0].epi_chain

    def test_all_ops_covered(self):
        """Every op index appears in exactly one plan."""
        ops, _ = _build_chain(k_scale, k_bias, k_relu)
        plans = _plan_fusion(ops)
        all_indices = set()
        for _, plan in plans:
            all_indices.update(plan.indices)
        assert all_indices == set(range(len(ops)))

    def test_dot_fanout_absorbed_by_anchor_not_multiroot(self):
        """Two elem siblings reading a GEMM output fold into the dot's epilogue
        (one ANCHOR plan), not a standalone MULTI_ROOT kernel. Multi-root defers
        to the anchor because its store absorbs both for free."""
        M, N, K = 64, 64, 64
        A = np.random.randn(M, K).astype(np.float32) * 0.1
        B = np.random.randn(K, N).astype(np.float32) * 0.1
        r1 = al.dot(A, B, BLOCK_M=32, BLOCK_N=32, BLOCK_K=16)
        g = ((M * N + 1023) // 1024,)
        left = k_scale[g](r1, np.zeros(M * N, dtype=np.float32), N=M * N)
        right = k_bias[g](r1, np.zeros(M * N, dtype=np.float32), N=M * N)

        ops, _ = _collect_pending_ops((left, right))
        roots = {id(o) for o in ops if o.kernel.name in ("k_scale", "k_bias")}
        plans = _plan_fusion(ops, roots)
        kinds = [p.kind for _, p in plans]
        assert FusionKind.MULTI_ROOT not in kinds
        anchor = [p for _, p in plans if p.kind == FusionKind.ANCHOR]
        assert len(anchor) == 1 and anchor[0].epi_chain

    def test_siblings_over_nonanchor_form_multiroot(self):
        """cos & sin sharing a non-fusable concat (the rope emb table) have no
        absorbing anchor, so they still fuse into one MULTI_ROOT kernel."""
        L = 2048
        freqs = np.random.randn(L).astype(np.float32)
        g = ((2 * L + 1023) // 1024,)
        emb = al.std.k_concat_2[g](freqs, freqs, N=2 * L, CAT_TOTAL=2, SPLIT_D=1, INNER=L)
        cos = get_unary_kernel("cos")[g](emb, np.zeros(2 * L, dtype=np.float32), N=2 * L)
        sin = get_unary_kernel("sin")[g](emb, np.zeros(2 * L, dtype=np.float32), N=2 * L)

        ops, _ = _collect_pending_ops((cos, sin))
        roots = {id(o) for o in ops if o.kernel.name in ("k_cos", "k_sin")}
        plans = _plan_fusion(ops, roots)
        mroot = [p for _, p in plans if p.kind == FusionKind.MULTI_ROOT]
        assert len(mroot) == 1
        cos_sin = {i for i, o in enumerate(ops) if o.kernel.name in ("k_cos", "k_sin")}
        assert set(mroot[0].indices) == cos_sin
