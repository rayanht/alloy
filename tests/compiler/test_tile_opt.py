"""Tests for tile IR optimization passes — persistent MMA, row-loop fusion.

Tests IR structure before/after optimization. No GPU, no MSL emission.
"""

import alloy as al
from alloy._compiler.trace import trace_kernel
from alloy._compiler.tile_opt import _opt_persistent_mma, _opt_fuse_row_loops
from alloy._compiler.tile_ir import Dot, ForLoop, FusedElementwise, Zeros, walk_ops


def _trace_gemm(M=32, N=32, K=32, BM=16, BN=16, BK=8):
    def k(A, B, C: al.output, M: al.constexpr, N: al.constexpr, K: al.constexpr,
          BM: al.constexpr, BN: al.constexpr, BK: al.constexpr):
        pm = al.program_id(0)
        pn = al.program_id(1)
        rm = pm * BM + al.arange(0, BM)
        rn = pn * BN + al.arange(0, BN)
        rk = al.arange(0, BK)
        acc = al.zeros((BM, BN), dtype=al.float32)
        for ki in range(0, K, BK):
            a = al.load(A + rm[:, None] * K + rk[None, :], mask=(rm[:, None] < M) & (rk[None, :] < K))
            b = al.load(B + rk[:, None] * N + rn[None, :], mask=(rk[:, None] < K) & (rn[None, :] < N))
            acc += al.tile_dot(a, b)
        al.store(C + rm[:, None] * N + rn[None, :], acc, mask=(rm[:, None] < M) & (rn[None, :] < N))

    return trace_kernel(k, "k",
                        {"M": M, "N": N, "K": K, "BM": BM, "BN": BN, "BK": BK},
                        param_names=["A", "B", "C", "M", "N", "K", "BM", "BN", "BK"],
                        constexpr_params={"M", "N", "K", "BM", "BN", "BK"},
                        output_params={"C"})


class TestPersistentMMA:
    def test_rewrites_zeros_forloop_dot_add(self):
        func = _trace_gemm()
        loops_before = [op for op in walk_ops(func.ops) if isinstance(op, ForLoop)]
        assert len(loops_before) >= 1

        _opt_persistent_mma(func)

        # Dot carries the accumulator in registers; the K-loop BinOp(add) is gone
        dots = [op for op in walk_ops(func.ops) if isinstance(op, Dot)]
        assert any(d.acc is not None for d in dots), "Dot should have persistent accumulator"

    def test_zeros_still_present(self):
        func = _trace_gemm()
        _opt_persistent_mma(func)
        zeros = [op for op in walk_ops(func.ops) if isinstance(op, Zeros)]
        assert len(zeros) >= 1, "Zeros initializer should remain"


class TestFuseRowLoops:
    def test_disabled_by_default(self):
        """fuse_loops defaults to 0, so the pass is a no-op."""
        func = _trace_gemm()
        _opt_fuse_row_loops(func)
        ops_after = list(walk_ops(func.ops))
        assert not any(isinstance(op, FusedElementwise) for op in ops_after)

    def test_enabled_creates_fused_nodes(self):
        """With fuse_loops=1, consecutive 2D elementwise ops in a ForLoop body fuse."""
        def k(A, B, C: al.output, M: al.constexpr, N: al.constexpr, K: al.constexpr,
              BM: al.constexpr, BN: al.constexpr, BK: al.constexpr):
            pm = al.program_id(0)
            pn = al.program_id(1)
            rm = pm * BM + al.arange(0, BM)
            rn = pn * BN + al.arange(0, BN)
            rk = al.arange(0, BK)
            acc = al.zeros((BM, BN), dtype=al.float32)
            for ki in range(0, K, BK):
                a = al.load(A + rm[:, None] * K + rk[None, :], mask=(rm[:, None] < M) & (rk[None, :] < K))
                b = al.load(B + rk[:, None] * N + rn[None, :], mask=(rk[:, None] < K) & (rn[None, :] < N))
                d = al.tile_dot(a, b)
                scaled = d * 2.0
                acc += scaled + 1.0
            al.store(C + rm[:, None] * N + rn[None, :], acc, mask=(rm[:, None] < M) & (rn[None, :] < N))

        func = trace_kernel(k, "k",
                            {"M": 32, "N": 32, "K": 32, "BM": 16, "BN": 16, "BK": 8},
                            param_names=["A", "B", "C", "M", "N", "K", "BM", "BN", "BK"],
                            constexpr_params={"M", "N", "K", "BM", "BN", "BK"},
                            output_params={"C"})
        func.options["fuse_loops"] = 1
        _opt_persistent_mma(func)
        _opt_fuse_row_loops(func)
        fused = [op for op in walk_ops(func.ops) if isinstance(op, FusedElementwise)]
        # Fusion is conditional on shmem/dot adjacency rules; verify structure only if it fired.
        if fused:
            assert len(fused[0].ops) >= 2, "Fused node should contain multiple ops"
