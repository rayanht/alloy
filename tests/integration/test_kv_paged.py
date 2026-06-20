"""PagedKV contract tests — no model, no GPU.

Guards the multi-slice dispatch surface: `supports_slices` silently
falling back to ContiguousKV's False disables the whole conversation
table (forks, prefix marks, slice switching) while the bit-exactness
gauntlets stay green.
"""

from types import SimpleNamespace

from alloy_server.generation.kv import ContiguousKV, PagedKV


def make_store(cls):
    return cls(
        model=SimpleNamespace(config=None),
        cache_dtype=None,
        kv_format=None,
        max_cache_len=1024,
        bookmark_slots=4,
    )


def test_paged_supports_slices():
    assert make_store(PagedKV).supports_slices() is True


def test_contiguous_is_single_slot():
    store = make_store(ContiguousKV)
    assert store.supports_slices() is False
    assert store.tensor_alloc() is None
    assert store.reclaim_beyond(SimpleNamespace(layers=[]), 0) == 0
