"""Tests for MSL codegen primitives — only test primitive emission patterns.

Does NOT assert patterns on complex kernels (those change with optimizations).
Tests only: scalar ops, load/store, barriers, thread_id, atomics, etc.
"""

import alloy as al
import pytest
from alloy._compiler.msl.math import fmt_const, format_scalar_op
from alloy._compiler.tile_ir import BinOp, UnaryOp, Constant, TileValue, Layout


class TestFormatConst:
    @pytest.mark.parametrize(
        "val,expected",
        [
            (0.0, "0.0f"),
            (1.5, "1.5f"),
            (42, "42"),
            (float("inf"), "INFINITY"),
            (float("-inf"), "-INFINITY"),
        ],
    )
    def test_format(self, val, expected):
        assert fmt_const(val) == expected


class TestFormatScalarOp:
    def _v(self, name, dtype="f32"):
        return TileValue(name, (), Layout.REPLICATED, dtype)

    def test_binop_add(self):
        op = BinOp(result=self._v("r"), op="add", lhs=self._v("a"), rhs=self._v("b"))
        assert format_scalar_op(op, lambda v: v.name) == "(a + b)"

    def test_binop_max(self):
        op = BinOp(result=self._v("r"), op="max", lhs=self._v("a"), rhs=self._v("b"))
        assert format_scalar_op(op, lambda v: v.name) == "max(a, b)"

    def test_unary_exp(self):
        op = UnaryOp(result=self._v("r"), op="exp", input=self._v("x"))
        assert format_scalar_op(op, lambda v: v.name) == "exp(x)"

    def test_unary_neg(self):
        op = UnaryOp(result=self._v("r"), op="neg", input=self._v("x"))
        assert format_scalar_op(op, lambda v: v.name) == "(-x)"

    def test_constant(self):
        op = Constant(result=self._v("c"), value=3.14)
        assert format_scalar_op(op, lambda v: v.name) == "3.14f"


class TestPrimitiveCodegen:
    def test_program_id_emits_gid(self):
        @al.kernel
        def k(out: al.output, N: al.constexpr):
            pid = al.program_id(0)
            al.store(out + pid, 1.0)

        msl = k.compile_to_msl(N=64)
        assert "threadgroup_position_in_grid" in msl

    def test_thread_id_emits_tid(self):
        @al.kernel
        def k(out: al.output, N: al.constexpr):
            tid = al.thread_id()
            al.store(out + tid, 1.0)

        msl = k.compile_to_msl(N=256)
        assert "thread_index_in_threadgroup" in msl or "tid" in msl

    def test_barrier_emits_threadgroup_barrier(self):
        @al.kernel
        def k(x, out: al.output, N: al.constexpr, BLOCK_SIZE: al.constexpr):
            pid = al.program_id(0)
            tid = al.thread_id()
            buf = al.shared(64)
            buf[tid] = al.load(x + pid * 64 + tid)
            al.barrier()
            al.store(out + pid * 64 + tid, buf[tid])

        msl = k.compile_to_msl(N=64, BLOCK_SIZE=64)
        assert "threadgroup_barrier(mem_flags::mem_threadgroup)" in msl

    def test_shared_alloc_emits_threadgroup_array(self):
        @al.kernel
        def k(x, out: al.output, N: al.constexpr, BLOCK_SIZE: al.constexpr):
            tid = al.thread_id()
            buf = al.shared(256)
            buf[tid] = al.load(x + tid)
            al.barrier()
            al.store(out + tid, buf[tid])

        msl = k.compile_to_msl(N=256, BLOCK_SIZE=256)
        assert "threadgroup" in msl
        assert "[256]" in msl

    def test_local_alloc_emits_thread_array(self):
        @al.kernel
        def k(x, out: al.output, N: al.constexpr):
            pid = al.program_id(0)
            arr = al.local(4)
            arr[0] = al.load(x + pid)
            al.store(out + pid, arr[0])

        msl = k.compile_to_msl(N=64)
        assert "float" in msl
        assert "[4]" in msl

    def test_mask_guard_emits_if(self):
        @al.kernel
        def k(x, out: al.output, N: al.constexpr):
            pid = al.program_id(0)
            offs = pid * 1024 + al.arange(0, 1024)
            mask = offs < N
            al.store(out + offs, al.load(x + offs, mask=mask), mask=mask)

        msl = k.compile_to_msl(N=1024)
        assert "if (" in msl

    def test_output_buffer_not_const(self):
        @al.kernel
        def k(x, out: al.output, N: al.constexpr):
            pid = al.program_id(0)
            al.store(out + pid, al.load(x + pid))

        msl = k.compile_to_msl(N=64)
        for line in msl.split("\n"):
            if "out" in line and "buffer" in line:
                assert "const" not in line

    def test_input_buffer_is_const(self):
        @al.kernel
        def k(x, out: al.output, N: al.constexpr):
            pid = al.program_id(0)
            al.store(out + pid, al.load(x + pid))

        msl = k.compile_to_msl(N=64)
        for line in msl.split("\n"):
            if "x" in line and "buffer(0)" in line:
                assert "const" in line
