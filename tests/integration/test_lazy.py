"""Integration tests for lazy evaluation — materializer chain, branchy graphs, views."""

import alloy as al
import numpy as np
from alloy._dispatch.lazy import _collect_pending_ops
from alloy._runtime.alloy_buffer import materialize_many
from alloy._runtime.metal import default_dispatcher
from tests.helpers import k_add, k_scale


class TestLazyDispatch:
    def test_single_kernel(self):
        N = 4096
        x = np.ones(N, dtype=np.float32)
        y = np.ones(N, dtype=np.float32) * 2
        result = k_add[(N + 1023) // 1024](x, y, np.zeros(N, dtype=np.float32), N=N)
        np.testing.assert_allclose(result, 3.0)

    def test_chained_kernels(self):
        N = 4096
        x = np.ones(N, dtype=np.float32)
        y = np.ones(N, dtype=np.float32) * 2
        r1 = k_add[(N + 1023) // 1024](x, y, np.zeros(N, dtype=np.float32), N=N)
        r2 = k_scale[(N + 1023) // 1024](r1, np.zeros(N, dtype=np.float32), N=N)
        np.testing.assert_allclose(r2, 6.0)

    def test_sync_materializes(self):
        N = 100_000
        x = np.ones(N, dtype=np.float32)
        y = np.ones(N, dtype=np.float32)
        result = k_add[(N + 1023) // 1024](x, y, np.zeros(N, dtype=np.float32), N=N)
        result.sync()
        np.testing.assert_allclose(result, 2.0)


class TestMaterializeMany:
    def test_branchy_graph_single_flush(self):
        M = N = K = 32
        rng = np.random.default_rng(0)
        a = rng.standard_normal((M, K)).astype(np.float32)
        b = rng.standard_normal((K, N)).astype(np.float32)
        ones = np.ones(M * N, dtype=np.float32)

        base = al.dot(a, b)
        base_flat = base.reshape(M * N)
        left = al.std.add(base_flat, ones, N=M * N).reshape(M, N)
        right = al.std.sub(base_flat, ones, N=M * N).reshape(M, N)

        d = default_dispatcher()
        before = d.dispatch_count
        materialize_many((left, right))
        assert d.dispatch_count - before == 1  # single command buffer

        np.testing.assert_allclose(np.asarray(left), a @ b + 1.0, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(np.asarray(right), a @ b - 1.0, rtol=1e-4, atol=1e-4)

    def test_view_ops_preserve_producer_edges(self):
        M = N = K = 32
        rng = np.random.default_rng(1)
        a = rng.standard_normal((M, K)).astype(np.float32)
        b = rng.standard_normal((K, N)).astype(np.float32)
        ones = np.ones(M * N, dtype=np.float32)

        base = al.dot(a, b)
        flat = base.reshape(M * N)
        left = al.std.add(flat, ones, N=M * N).reshape(M, N)
        right = al.std.sub(flat, ones, N=M * N).reshape(M, N)

        ops, _ = _collect_pending_ops((left, right))
        assert len(ops) == 3
        assert ops[1].input_producers["x"] is ops[0]
        assert ops[2].input_producers["x"] is ops[0]
