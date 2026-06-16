"""Integration tests for CacheManager and DispatchEngine."""

from typing import cast

from alloy._compiler.tile_ir import TileFunction
from alloy._dispatch.cache import CacheManager, KernelKey
from alloy._dispatch.dispatch import DispatchEngine, _engine
from alloy._runtime.metal import CompiledKernel


TRACE_FUNC = cast(TileFunction, None)
PIPELINE = cast(CompiledKernel, None)


class TestSingletons:
    def test_cache_manager_singleton(self):
        assert CacheManager.default() is CacheManager.default()

    def test_engine_singleton(self):
        assert DispatchEngine.default() is DispatchEngine.default()

    def test_engine_owns_cache(self):
        assert _engine.cache is CacheManager.default()


class TestCacheManagerClear:
    def test_clear_resets_all(self, tmp_path):
        cm = CacheManager(cache_dir=tmp_path)
        key = KernelKey("def f(): pass")

        cm.put_msl(key, "kernel void f() {}")
        cm.put_pipeline("hash1", "dev1", PIPELINE)
        cm.trace_cache[("k",)] = (TRACE_FUNC, (1, 1, 1), None)
        cm.opaque_cache[("k",)] = (PIPELINE, (1, 1, 1), (32, 1, 1))
        cm.fused_cache[("k",)] = []

        cm.clear()

        assert cm.get_msl(key) is None
        assert cm.get_pipeline("hash1", "dev1") is None
        assert not cm.trace_cache and not cm.opaque_cache and not cm.fused_cache
        assert not cm._msl_dir.exists()

    def test_clear_dispatch_preserves_msl(self, tmp_path):
        cm = CacheManager(cache_dir=tmp_path)
        key = KernelKey("def f(): pass")
        cm.put_msl(key, "kernel void f() {}")
        cm.trace_cache[("k",)] = (TRACE_FUNC, (1, 1, 1), None)

        cm.clear_dispatch()

        assert cm.get_msl(key) is not None  # L1 preserved
        assert cm.trace_cache  # L3 preserved (value-keyed, safe across batches)

    def test_stats(self, tmp_path):
        cm = CacheManager(cache_dir=tmp_path)
        key = KernelKey("def f(): pass")
        cm.get_msl(key)  # miss
        cm.put_msl(key, "kernel void f() {}")
        cm.get_msl(key)  # hit

        s = cm.stats()
        assert s["l1_hits"] == 1
        assert s["l1_misses"] == 1
