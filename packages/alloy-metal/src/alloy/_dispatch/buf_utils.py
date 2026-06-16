"""Buffer utilities, allocation, and module-level state.

Leaf module — no dependencies on other alloy._core submodules.
"""

from __future__ import annotations
from alloy._compiler.dtypes import float32, DType
from alloy.log import get_logger


from alloy._dispatch.fusion_types import DispatchLaunch
from alloy._runtime import _metal_ext as _ext
from alloy._runtime.alloy_buffer import AlloyBuffer, _compute_contiguous_strides

# Save builtins before they're shadowed by tile reduction API (al.max, al.min, al.sum)
_builtin_max = max
_builtin_min = min
_builtin_sum = sum

_STRIDE_NATIVE_KERNELS = frozenset(
    {
        "strided_copy_4d",
        "strided_copy_5d",
        "k_concat_2",
        "k_cache_write_arange",
        "k_index_copy_dim2_4d",
        "k_gather_2d",
        "im2col_2d",
        "attention_strided",
        "attention_strided_masked_by_batch",
        "attention_strided_logsumexp",
        "attention_strided_logsumexp_masked_by_batch",
        "attention_strided_backward_dq",
        "attention_strided_backward_dq_masked_by_batch",
        "attention_strided_backward_dkdv",
        "attention_strided_backward_dkdv_masked_by_batch",
        "rope_apply_strided",
    }
)

_NON_FUSABLE_ELEM_KERNELS = frozenset(
    {
        "k_gather_rows_2d",  # data-dependent indexing (index buffer)
        "k_gather_2d",  # data-dependent indexing (two index buffers)
        "k_index_2d_nd",  # data-dependent indexing
        "k_index_copy_dim2_4d",  # scatter with loop over positions
        "k_cache_scatter_dim2_4d",  # scatter to cache at indexed positions
        "k_copy",  # identity copy — no point fusing
        "strided_copy_4d",  # strided source addressing (infrastructure kernel)
        "strided_copy_5d",  # strided source addressing (5D variant)
        "k_concat_2",  # reads from two source buffers by position
        "im2col_1d",  # complex loop nest with conditional loads
        "im2col_2d",  # complex loop nest with conditional loads
    }
)


def _batch_to_v2(
    entries: list[DispatchLaunch],
) -> list[tuple[int, list[tuple[int, int, int]], tuple[int, int, int], tuple[int, int, int]]]:
    """Convert dispatch batch to (pso_handle, [(ptr, nbytes, offset)...], grid, tg) tuples for C++ dispatch."""
    v2 = []
    for entry in entries:
        buf_ptrs = [(b.base_ptr, b.metal_nbytes, b._offset) for b in entry.buffers]
        v2.append((entry.pso_handle, buf_ptrs, entry.grid, entry.threadgroup))
    return v2


def _normalize_grid(grid: int | float | tuple[int, ...] | list[int]) -> tuple[int, int, int]:
    """Normalize a grid or threadgroup to a 3D int tuple."""
    if isinstance(grid, (int, float)):
        return (int(grid), 1, 1)
    if not isinstance(grid, tuple):
        grid = tuple(grid)
    n = len(grid)
    if n == 0:
        return (1, 1, 1)
    if n == 1:
        return (int(grid[0]), 1, 1)
    if n == 2:
        return (int(grid[0]), int(grid[1]), 1)
    return (int(grid[0]), int(grid[1]), int(grid[2]))


# --- Debug mode ---

_debug_mode = False


def set_debug(enabled: bool = True) -> None:
    """Enable or disable debug mode."""
    global _debug_mode
    _debug_mode = enabled


def get_debug() -> bool:
    return _debug_mode


# --- Module-level state ---

logger = get_logger("alloy.runtime")

# Threshold above which an allocation is operationally interesting
# (one INFO event per ≥2 GB buffer — typically KV caches, lm_head).
_LARGE_BUFFER_BYTES = 2048 * 1024 * 1024

_aligned_pool: dict[int, list[object]] = {}
_alloy_handle_map: dict[int, int] = {}
_alloy_buf_map: dict[int, AlloyBuffer] = {}
_alloc_ptrs_this_run: set[int] = set()

# --- Record-only compile mode ---
#
# A plan is built entirely from dispatch *metadata* (PSO handles, buffer
# ptrs/offsets/nbytes, grids) — `_compile_to_plan` never reads an intermediate's
# *contents*. So `eager_compile_all` can record the plan WITHOUT executing the
# GPU and WITHOUT allocating real Metal storage for kernel-produced
# intermediates: it gives them "phantom" AlloyBuffers (real shape/dtype/nbytes,
# a unique fake ptr, no MTLBuffer). Peak compile memory then drops from
# O(M_MAX × layers) to weights + the bounded liveness pool — so M_MAX can scale
# to the model's native context. The GPU dispatch is skipped (the phantom ptrs
# can't be bound to an encoder anyway); the recorded plan is byte-identical to
# the one a real run-0 would produce, and run-1+ (`_execute_plan`) is untouched.
_record_only_mode = [False]
# Fake ptrs live in a high reserved range that cannot collide with real heap
# addresses (macOS arm64 user pointers are far lower) or 0 ("no ptr").
_PHANTOM_PTR_BASE = 0x7000_0000_0000_0000
_phantom_ptr_counter = [_PHANTOM_PTR_BASE]


def set_record_only(enabled: bool) -> None:
    """Toggle record-only compile (phantom intermediates + no GPU dispatch)."""
    _record_only_mode[0] = bool(enabled)


def is_record_only() -> bool:
    return _record_only_mode[0]


def _alloc_scratch(shape: tuple[int, ...], dtype: DType = float32) -> AlloyBuffer:
    """Handler-created kernel intermediate/scratch buffer.

    Like `_alloc_aligned` but (a) honours record-only mode (phantom, no Metal
    page) and (b) registers the ptr as an INTERMEDIATE so `_compile_to_plan`
    pools it (and never transiently treats a phantom as a WeightSlot). Use for
    buffers whose contents a kernel produces and that are never read on the CPU
    during tracing — e.g. the MoE handler's gather/sort/expert scratch.
    """
    buf = _alloc_phantom(shape, dtype) if _record_only_mode[0] else _alloc_aligned(shape, dtype)
    _alloc_ptrs_this_run.add(buf.base_ptr)
    return buf


def _alloc_phantom(shape: tuple[int, ...], dtype: DType = float32) -> AlloyBuffer:
    """Metadata-only buffer for record-only compile: NO Metal allocation.

    Carries the real shape/dtype/nbytes and a unique fake ptr so `_record_for_plan`
    and `_compile_to_plan` see a valid identity, but holds no GPU page. The fake
    ptr must never be dereferenced — record-only skips the GPU dispatch, and no
    handler reads an intermediate's contents during tracing.
    """
    if not isinstance(dtype, DType):
        raise TypeError(f"dtype must be a DType, got {type(dtype)}")
    shape = tuple(int(s) for s in shape)
    count = 1
    for s in shape:
        count *= s
    nbytes = count * dtype.itemsize
    ptr = _phantom_ptr_counter[0]
    # Advance past this allocation's footprint so distinct phantoms get disjoint
    # ptr ranges (keeps base/data-ptr identity unique, like real allocations).
    _phantom_ptr_counter[0] += max(nbytes, 64)
    strides = _compute_contiguous_strides(shape, dtype.itemsize)
    buf = AlloyBuffer(-1, 0, shape, strides, dtype, raw_ptr=ptr, total_nbytes=nbytes)
    _alloy_buf_map[ptr] = buf
    return buf


def _free_aligned(buf: AlloyBuffer) -> None:
    """Release an `_alloc_aligned` buffer's Metal pages and drop the bookkeeping
    that pins its Python wrapper. For transient GPU scratch (e.g. the GGUF
    quant-repack buffers) that is copied to CPU and never used again — without
    this the Metal pages live for the process lifetime (no pool, no GC for
    Metal buffers), which at load was ~3.5 GB of leaked repack scratch on
    qwen3.5:4b. No-op on phantom/raw buffers (no Metal handle)."""
    handle = buf._parent_handle
    if handle < 0:
        return
    ptr = _ext.buf_ptr(handle)
    _ext.buf_release(handle)
    _alloy_handle_map.pop(ptr, None)
    _alloy_buf_map.pop(ptr, None)
    buf._backing_arr = None  # drop the numpy view over now-freed pages


def is_phantom_buffer(buf: AlloyBuffer) -> bool:
    """True if `buf` is a record-only phantom (metadata-only, no Metal page).

    Its fake ptr lives in the reserved `_PHANTOM_PTR_BASE` range and must NEVER be
    dereferenced — reading it (`.numpy`, `read_scalar`) is a raw memory access at
    `0x7000…` and SEGFAULTS (not a catchable Python exception). Callers that read
    buffer contents (e.g. the tuner's snapshot) must skip phantoms.
    """
    return buf._parent_handle < 0 and buf._raw_ptr >= _PHANTOM_PTR_BASE


def _unique_lazy_buffers(buffers: tuple[AlloyBuffer, ...] | list[AlloyBuffer]) -> list[AlloyBuffer]:
    """Deduplicate lazy buffers by identity."""
    unique: list[AlloyBuffer] = []
    seen: set[int] = set()
    for lb in buffers:
        key = id(lb)
        if key in seen:
            continue
        seen.add(key)
        unique.append(lb)
    return unique


def _alloc_aligned(shape: tuple[int, ...], dtype: DType = float32) -> AlloyBuffer:
    """Allocate a page-aligned buffer for Metal zero-copy dispatch."""
    if not isinstance(dtype, DType):
        raise TypeError(f"dtype must be a DType, got {type(dtype)}")
    shape = tuple(int(s) for s in shape)
    count = 1
    for s in shape:
        count *= s
    nbytes = count * dtype.itemsize
    try:
        _arr, handle, ptr = _ext.alloc_typed(nbytes, tuple(shape), dtype.ir)
    except Exception as exc:
        logger.error(
            "buffer_alloc_failed",
            nbytes=nbytes, shape=shape, dtype=dtype.ir, error=str(exc),
        )
        raise
    if nbytes >= _LARGE_BUFFER_BYTES:
        logger.info("large_buffer_allocated", nbytes=nbytes, shape=shape, dtype=dtype.ir)
    _alloy_handle_map[ptr] = handle
    strides = _compute_contiguous_strides(tuple(shape), dtype.itemsize)
    buf = AlloyBuffer(handle, 0, tuple(shape), strides, dtype)
    # CRITICAL: retain the backing allocation to prevent GC from freeing the
    # page-aligned memory while the Metal buffer still references it.
    buf._backing_arr = _arr
    _alloy_buf_map[ptr] = buf
    return buf
