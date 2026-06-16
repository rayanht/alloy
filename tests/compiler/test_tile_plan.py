"""Tests for tile kernel planning — thread model, shmem, column tiling, dtypes.

Tests plan_tile_kernel output. No GPU, no MSL emission.
"""

import alloy as al
import pytest
from alloy._compiler.trace import trace_kernel
from alloy._compiler.tile_plan import plan_tile_kernel
from alloy._compiler.tile_opt import optimize_tile_ir


def _trace_and_plan(fn, name, ce, **kwargs):
    func = trace_kernel(fn, name, ce,
                        param_names=list(ce.keys()) if not kwargs.get("param_names") else kwargs["param_names"],
                        constexpr_params=set(ce.keys()) if not kwargs.get("constexpr_params") else kwargs["constexpr_params"],
                        output_params=kwargs.get("output_params", set()))
    optimize_tile_ir(func)
    return plan_tile_kernel(func)


class TestThreadModel:
    def test_elementwise_threads_equal_block(self):
        def k(x, out: al.output, N: al.constexpr):
            pid = al.program_id(0)
            offs = pid * 256 + al.arange(0, 256)
            mask = offs < N
            al.store(out + offs, al.load(x + offs, mask=mask) * 2.0, mask=mask)

        plan = _trace_and_plan(k, "k", {"N": 4096},
                               param_names=["x", "out", "N"],
                               constexpr_params={"N"},
                               output_params={"out"})
        assert plan.threads == 256 * plan.tpr

    def test_gemm_threads_from_simdgroups(self):
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

        plan = _trace_and_plan(k, "k",
                               {"M": 64, "N": 64, "K": 64, "BM": 32, "BN": 32, "BK": 16},
                               param_names=["A", "B", "C", "M", "N", "K", "BM", "BN", "BK"],
                               constexpr_params={"M", "N", "K", "BM", "BN", "BK"},
                               output_params={"C"})
        # 32x32 with reg=2 → 2x2 simdgroups → 4*32 = 128 threads
        assert plan.threads % 32 == 0
        assert plan.threads >= 32

    def test_scalar_kernel_one_thread(self):
        def k(x, out: al.output, N: al.constexpr):
            pid = al.program_id(0)
            al.store(out + pid, al.load(x + pid) * 2.0)

        plan = _trace_and_plan(k, "k", {"N": 64},
                               param_names=["x", "out", "N"],
                               constexpr_params={"N"},
                               output_params={"out"})
        assert plan.threads == 1


class TestDtypeDetection:
    def test_f32_default(self):
        def k(x, out: al.output, N: al.constexpr):
            pid = al.program_id(0)
            offs = pid * 256 + al.arange(0, 256)
            mask = offs < N
            al.store(out + offs, al.load(x + offs, mask=mask), mask=mask)

        plan = _trace_and_plan(k, "k", {"N": 1024},
                               param_names=["x", "out", "N"],
                               constexpr_params={"N"},
                               output_params={"out"})
        assert plan.dtype == "float"
        assert plan.acc_dtype == "float"


class TestShmemPlan:
    def test_gemm_has_shmem_buffers(self):
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

        plan = _trace_and_plan(k, "k",
                               {"M": 32, "N": 32, "K": 32, "BM": 16, "BN": 16, "BK": 8},
                               param_names=["A", "B", "C", "M", "N", "K", "BM", "BN", "BK"],
                               constexpr_params={"M", "N", "K", "BM", "BN", "BK"},
                               output_params={"C"})
        assert plan.shmem_plan, "GEMM should have shared memory plan"

    def test_elementwise_no_shmem(self):
        def k(x, out: al.output, N: al.constexpr):
            pid = al.program_id(0)
            offs = pid * 256 + al.arange(0, 256)
            mask = offs < N
            al.store(out + offs, al.load(x + offs, mask=mask) * 2.0, mask=mask)

        plan = _trace_and_plan(k, "k", {"N": 1024},
                               param_names=["x", "out", "N"],
                               constexpr_params={"N"},
                               output_params={"out"})
        assert not plan.shmem_plan


class TestRegisterBlocking:
    @pytest.mark.parametrize("BM,BN,expected_reg", [
        (64, 64, 4),   # large → 4x4
        (16, 16, 2),   # medium → 2x2
        (8, 8, 1),     # small → 1x1
    ])
    def test_auto_register_blocking(self, BM, BN, expected_reg):
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

        plan = _trace_and_plan(k, "k",
                               {"M": BM, "N": BN, "K": 16, "BM": BM, "BN": BN, "BK": 8},
                               param_names=["A", "B", "C", "M", "N", "K", "BM", "BN", "BK"],
                               constexpr_params={"M", "N", "K", "BM", "BN", "BK"},
                               output_params={"C"})
        assert plan.reg_m == expected_reg
