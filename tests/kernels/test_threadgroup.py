"""GPU correctness tests for threadgroup primitives — shared, barrier, local, atomics."""

import alloy as al
import numpy as np


class TestLocalArray:
    def test_roundtrip(self):
        @al.kernel
        def k(x, out: al.output, D: al.constexpr):
            row = al.program_id(0)
            arr = al.local(4)
            for d in range(4):
                arr[d] = al.load(x + row * 4 + d)
            for d in range(4):
                al.store(out + row * 4 + d, arr[d] * 2.0)

        N = 16
        x = np.arange(N * 4, dtype=np.float32)
        np.testing.assert_allclose(k[(N,)](x, np.zeros_like(x), D=4), x * 2.0)

    def test_accumulate_loop(self):
        @al.kernel
        def k(x, out: al.output, D: al.constexpr):
            row = al.program_id(0)
            arr = al.local(8)
            for d in range(8):
                arr[d] = al.load(x + row * 8 + d)
            acc = 0.0
            for d in range(8):
                acc = acc + arr[d]
            al.store(out + row, acc)

        N = 4
        x = np.arange(N * 8, dtype=np.float32)
        expected = x.reshape(N, 8).sum(axis=1)
        np.testing.assert_allclose(k[(N,)](x, np.zeros(N, dtype=np.float32), D=8), expected)


class TestSharedMemory:
    def test_copy_via_barrier(self):
        @al.kernel
        def k(x, out: al.output, N: al.constexpr, BLOCK_SIZE: al.constexpr):
            pid = al.program_id(0)
            tid = al.thread_id()
            buf = al.shared(64)
            idx = pid * 64 + tid
            buf[tid] = al.load(x + idx)
            al.barrier()
            al.store(out + idx, buf[tid])

        x = np.arange(256, dtype=np.float32)
        np.testing.assert_allclose(k[(4,)](x, np.zeros(256, dtype=np.float32),
                                           N=256, BLOCK_SIZE=64), x)

    def test_reverse_via_shared(self):
        @al.kernel
        def k(x, out: al.output, N: al.constexpr, BLOCK_SIZE: al.constexpr):
            pid = al.program_id(0)
            tid = al.thread_id()
            buf = al.shared(32)
            idx = pid * 32 + tid
            buf[tid] = al.load(x + idx)
            al.barrier()
            al.store(out + idx, buf[31 - tid])

        x = np.arange(32, dtype=np.float32)
        np.testing.assert_allclose(k[(1,)](x, np.zeros(32, dtype=np.float32),
                                           N=32, BLOCK_SIZE=32), x[::-1])


class TestAtomics:
    def test_atomic_add_all_to_zero(self):
        @al.kernel
        def k(out: al.output, N: al.constexpr):
            al.atomic_add(out, 0, 1)

        result = k[(64,)](np.zeros(1, dtype=np.int32), N=64)
        assert int(np.asarray(result)[0]) == 64

    def test_atomic_add_per_element(self):
        @al.kernel
        def k(out: al.output, N: al.constexpr):
            pid = al.program_id(0)
            al.atomic_add(out, pid, 1)

        N = 32
        result = k[(N,)](np.zeros(N, dtype=np.int32), N=N)
        np.testing.assert_array_equal(np.asarray(result), np.ones(N, dtype=np.int32))

    def test_atomic_max(self):
        @al.kernel
        def k(data, out: al.output, N: al.constexpr):
            pid = al.program_id(0)
            v = al.load(data + pid)
            al.atomic_max(out, 0, v)

        N = 256
        x = np.random.default_rng(42).integers(-1000, 1000, size=N).astype(np.int32)
        result = k[(N,)](x, np.array([-999999], dtype=np.int32), N=N)
        assert int(np.asarray(result)[0]) == int(x.max())

    def test_atomic_min(self):
        @al.kernel
        def k(data, out: al.output, N: al.constexpr):
            pid = al.program_id(0)
            v = al.load(data + pid)
            al.atomic_min(out, 0, v)

        N = 256
        x = np.random.default_rng(42).integers(-1000, 1000, size=N).astype(np.int32)
        result = k[(N,)](x, np.array([999999], dtype=np.int32), N=N)
        assert int(np.asarray(result)[0]) == int(x.min())

    def test_atomic_cas(self):
        @al.kernel
        def k(out: al.output, N: al.constexpr):
            # CAS: if out[0] == 0, set to 42. Only first thread succeeds.
            al.atomic_cas(out, 0, 0, 42)

        result = k[(1,)](np.zeros(1, dtype=np.int32), N=1)
        assert int(np.asarray(result)[0]) == 42
