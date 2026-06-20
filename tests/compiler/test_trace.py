"""Tests for kernel tracing — TracedValue, pointer decomposition, spec assembly."""

import alloy as al
from alloy._compiler.trace import trace_kernel
from alloy._compiler.tile_ir import Load, Store, ForLoop, walk_ops


def _trace(fn, name, ce, **kwargs):
    return trace_kernel(fn, name, ce, param_names=list(fn.__code__.co_varnames[:fn.__code__.co_argcount]),
                        constexpr_params={p for p in ce}, source=None, **kwargs)


class TestTraceBasic:
    def test_elementwise_produces_load_store(self):
        def k(x, out: al.output, N: al.constexpr):
            pid = al.program_id(0)
            offs = pid * 1024 + al.arange(0, 1024)
            mask = offs < N
            al.store(out + offs, al.load(x + offs, mask=mask), mask=mask)

        func = trace_kernel(k, "k", {"N": 4096},
                            param_names=["x", "out", "N"],
                            constexpr_params={"N"},
                            output_params={"out"})
        loads = [op for op in walk_ops(func.ops) if isinstance(op, Load)]
        stores = [op for op in walk_ops(func.ops) if isinstance(op, Store)]
        assert len(loads) == 1
        assert len(stores) == 1

    def test_gemm_produces_dot_and_forloop(self):
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
                rk = rk  # satisfy loop variable requirement
            al.store(C + rm[:, None] * N + rn[None, :], acc, mask=(rm[:, None] < M) & (rn[None, :] < N))

        func = trace_kernel(k, "k", {"M": 32, "N": 32, "K": 32, "BM": 16, "BN": 16, "BK": 8},
                            param_names=["A", "B", "C", "M", "N", "K", "BM", "BN", "BK"],
                            constexpr_params={"M", "N", "K", "BM", "BN", "BK"},
                            output_params={"C"})
        loops = [op for op in walk_ops(func.ops) if isinstance(op, ForLoop)]
        assert len(loops) >= 1

    def test_2d_load_has_semantic_addressing(self):
        def k(A, out: al.output, M: al.constexpr, N: al.constexpr, BM: al.constexpr):
            pm = al.program_id(0)
            rm = pm * BM + al.arange(0, BM)
            rn = al.arange(0, N)
            a = al.load(A + rm[:, None] * N + rn[None, :], mask=rm[:, None] < M)
            al.store(out + rm[:, None] * N + rn[None, :], a * 2.0, mask=rm[:, None] < M)

        func = trace_kernel(k, "k", {"M": 32, "N": 16, "BM": 16},
                            param_names=["A", "out", "M", "N", "BM"],
                            constexpr_params={"M", "N", "BM"},
                            output_params={"out"})
        loads = [op for op in walk_ops(func.ops) if isinstance(op, Load)]
        assert len(loads) >= 1
        ld = loads[0]
        assert ld.row_indices is not None or ld.offsets is not None


class TestDispatchSpecFromTrace:
    def test_1d_spec_inferred(self):
        def k(x, out: al.output, N: al.constexpr):
            pid = al.program_id(0)
            offs = pid * 1024 + al.arange(0, 1024)
            mask = offs < N
            al.store(out + offs, al.load(x + offs, mask=mask) * 2.0, mask=mask)

        func = trace_kernel(k, "k", {"N": 4096},
                            param_names=["x", "out", "N"],
                            constexpr_params={"N"},
                            output_params={"out"})
        spec = func.dispatch_spec
        assert spec is not None
        assert 0 in spec.grid_axes

    def test_2d_spec_two_axes(self):
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

        func = trace_kernel(k, "k", {"M": 64, "N": 64, "K": 64, "BM": 32, "BN": 32, "BK": 16},
                            param_names=["A", "B", "C", "M", "N", "K", "BM", "BN", "BK"],
                            constexpr_params={"M", "N", "K", "BM", "BN", "BK"},
                            output_params={"C"})
        spec = func.dispatch_spec
        assert spec is not None
        assert 0 in spec.grid_axes and 1 in spec.grid_axes
