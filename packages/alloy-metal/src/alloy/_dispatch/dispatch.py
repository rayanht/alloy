"""Dispatch — fuse, compile, and dispatch batches of pending LazyOps.

Single entry point: DispatchEngine.dispatch_ops().
Pipeline: _lazy._materialize_many → DispatchEngine.dispatch_ops
          → _fusion._plan_fusion + _compile_fusion_plans → GPU
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from alloy._dispatch.buf_utils import (
    _alloc_ptrs_this_run,
    _alloy_handle_map,
    _batch_to_v2,
    _unique_lazy_buffers,
    is_record_only,
)
from alloy._dispatch.cache import CacheManager, CachedFusionEntry
from alloy._dispatch.lazy import LazyOp
from alloy._dispatch.fusion_analysis import _plan_fusion
from alloy._dispatch.fusion_types import (
    DispatchLaunch,
    FusedPair,
    FusionPlan,
    Grid3D,
    PlanBufferBinding,
    RecordedDispatch,
)
from alloy._dispatch.fusion_compile import _compile_fusion_plans
from alloy._dispatch.dup_fuse_cast import dup_fuse_casts
from alloy.log import get_logger
from alloy._runtime import _metal_ext
from alloy._runtime import profile
from alloy._runtime.alloy_buffer import AlloyBuffer
from alloy._runtime.metal import default_dispatcher

logger = get_logger("alloy.dispatch")

# Zero-cost timing returned when a dispatch is recorded but not executed
# (record-only compile). Never mutated.
_EMPTY_TIMING: dict[str, float] = {"gpu": 0.0, "encode": 0.0, "wait": 0.0, "copy": 0.0}

if TYPE_CHECKING:
    from alloy._dispatch.kernel import KernelFunction


# --- DispatchEngine ---


class DispatchEngine:
    """Orchestrates fusion, compilation, and Metal dispatch.

    Owns dependency group analysis, Metal command buffer submission,
    plan recording for the compiled plan path, and allocation tracking.
    """

    __slots__ = (
        "_cache",
        "op_profile",
        "_plan_record",
        "_plan_record_paused",
        "_plan_buf_map",
    )

    _default: DispatchEngine | None = None

    def __init__(self, cache: CacheManager | None = None) -> None:
        self._cache: CacheManager = cache or CacheManager.default()
        self.op_profile: dict[int, dict[str, float]] = {}
        self._plan_record: list[RecordedDispatch] | None = None
        self._plan_record_paused: bool = False
        self._plan_buf_map: dict[int, AlloyBuffer] = {}

    @property
    def cache(self) -> CacheManager:
        return self._cache

    def clear(self) -> None:
        """Reset all dispatch state and caches."""
        self._cache.clear()
        self.op_profile.clear()
        self._plan_record = None
        self._plan_record_paused = False
        self._plan_buf_map.clear()
        _alloc_ptrs_this_run.clear()
        _alloy_handle_map.clear()
        _metal_ext.clear_buffer_cache()

    def clear_run(self) -> None:
        """Reset per-run state. Called at start of each compiled() invocation."""
        _alloc_ptrs_this_run.clear()
        self._cache.clear_dispatch()

    # --- Allocation tracking ---

    def track_alloc(self, ptr: int) -> None:
        """Mark a pointer as an alloy-allocated intermediate."""
        _alloc_ptrs_this_run.add(ptr)

    def untrack_alloc(self, ptr: int) -> None:
        """Remove a pointer from intermediate tracking (constants, weights)."""
        _alloc_ptrs_this_run.discard(ptr)

    @property
    def alloc_ptrs(self) -> set[int]:
        """Current set of alloy-allocated intermediate pointers."""
        return _alloc_ptrs_this_run

    # --- Plan recording API ---

    def start_recording(self) -> None:
        """Start recording dispatches for compiled plan."""
        self._plan_record = []

    def stop_recording(
        self,
    ) -> tuple[list[RecordedDispatch], dict[int, AlloyBuffer]] | None:
        """Stop recording and return (dispatches, buf_map). Returns None if empty."""
        rec = self._plan_record
        self._plan_record = None
        if not rec:
            return None
        buf_map = dict(self._plan_buf_map)
        self._plan_buf_map.clear()
        return rec, buf_map

    # --- Dispatch ---

    def dispatch_ops(
        self,
        ops: list[LazyOp],
        roots: tuple[AlloyBuffer, ...] | list[AlloyBuffer] | None = None,
    ) -> tuple[dict[str, float], float]:
        """Fuse, compile, and dispatch a batch of pending ops."""
        if not ops:
            return {"gpu": 0.0, "encode": 0.0, "wait": 0.0, "copy": 0.0}, 0.0

        # Duplicate widening casts whose prologue absorption is otherwise
        # blocked (multi-consumer or save-for-bwd root). Each duplicate
        # becomes a single-consumer non-root chain the existing prologue
        # path can absorb naturally.
        _root_op_ids: set[int] = {
            id(lb._producer)
            for lb in _unique_lazy_buffers(roots or ())
            if lb._producer is not None
        }
        ops = dup_fuse_casts(ops, _root_op_ids)

        for o in ops:
            o._dispatched = True
            _populate_buf_sets(o)

        _p = profile._profile_enabled
        _t = time.perf_counter_ns()

        cache = self._cache
        cache_key = _fusion_cache_key(ops)
        cached = cache.fused_cache.get(cache_key)

        if cached is not None:
            # ─── Cache hit: swap buffer pointers into cached PSOs ───
            dispatches: list[DispatchLaunch] = []
            op_indices_list: list[frozenset[int]] = []
            for entry in cached:
                bufs = tuple(ops[oi].buffer_args[pi][1] for oi, pi in entry.buffer_slots)
                dispatches.append(
                    DispatchLaunch(
                        kernel=entry.pso_handle,
                        buffers=bufs,
                        grid=entry.grid,
                        threadgroup=entry.threadgroup,
                        write_indices=entry.write_indices,
                        debug_name=entry.debug_name,
                    )
                )
                op_indices_list.append(entry.op_indices)

            self._record_for_plan(dispatches)
            fusion_ms = (time.perf_counter_ns() - _t) / 1e6 if _p else 0.0
            # Record-only compile: the plan is already recorded; skip the GPU
            # dispatch (phantom intermediates have no MTLBuffer to bind).
            timing = (
                _EMPTY_TIMING if is_record_only()
                else _group_and_dispatch(dispatches, ops, op_indices_list)
            )

            if _p:
                rec = profile.DispatchRecord(name=f"batch({len(dispatches)})")
                rec.phases[profile.FUSION] = fusion_ms
                rec.phases[profile.GPU] = timing.get("gpu", 0.0)
                rec.phases[profile.ENCODE] = timing.get("encode", 0.0)
                rec.phases[profile.WAIT] = timing.get("wait", 0.0)
                rec.phases[profile.COPY] = timing.get("copy", 0.0)
                rec.phases[profile.TOTAL] = (
                    fusion_ms + timing.get("encode", 0.0) + timing.get("wait", 0.0)
                )
                profile.get_accumulator().records.append(rec)
            return timing, fusion_ms

        # ─── Cache miss: fuse + compile ───
        root_op_ids = _root_op_ids
        plans: list[tuple[int, FusionPlan]] = _plan_fusion(ops, root_op_ids)
        root_indices: set[int] = {i for i, o in enumerate(ops) if id(o) in root_op_ids}
        fused_pairs: list[FusedPair] = _compile_fusion_plans(
            ops,
            plans,
            roots=root_indices,
            compile_individual=self.compile_kernel,
        )

        all_dispatches: list[DispatchLaunch] = [d for d, _ in fused_pairs]
        all_op_indices: list[frozenset[int]] = [frozenset(indices) for _, indices in fused_pairs]

        # Cache: map each buffer in each dispatch back to (op_idx, param_idx)
        ptr_to_loc: dict[int, tuple[int, int]] = {}
        for oi, o in enumerate(ops):
            for pi, (_, arg) in enumerate(o.buffer_args):
                ptr_to_loc[arg.data_ptr] = (oi, pi)
        entries: list[CachedFusionEntry] = []
        cacheable = True
        for di, dispatch in enumerate(all_dispatches):
            slots: list[tuple[int, int]] = []
            for buf in dispatch.buffers:
                loc = ptr_to_loc.get(buf.data_ptr)
                if loc is None:
                    cacheable = False
                    break
                slots.append(loc)
            if not cacheable:
                break
            entries.append(
                CachedFusionEntry(
                    pso_handle=dispatch.pso_handle,
                    grid=dispatch.grid,
                    threadgroup=dispatch.threadgroup,
                    buffer_slots=tuple(slots),
                    op_indices=all_op_indices[di],
                    write_indices=dispatch.write_indices,
                    debug_name=dispatch.debug_name,
                )
            )
        if cacheable:
            cache.fused_cache[cache_key] = entries

        self._record_for_plan(all_dispatches)

        fusion_ms = (time.perf_counter_ns() - _t) / 1e6 if _p else 0.0
        timing = (
            _EMPTY_TIMING if is_record_only()
            else _group_and_dispatch(all_dispatches, ops, all_op_indices)
        )

        if _p:
            rec = profile.DispatchRecord(name=f"batch({len(all_dispatches)},miss)")
            rec.phases[profile.FUSION] = fusion_ms
            rec.phases[profile.GPU] = timing.get("gpu", 0.0)
            rec.phases[profile.ENCODE] = timing.get("encode", 0.0)
            rec.phases[profile.WAIT] = timing.get("wait", 0.0)
            rec.phases[profile.COPY] = timing.get("copy", 0.0)
            rec.phases[profile.TOTAL] = (
                fusion_ms + timing.get("encode", 0.0) + timing.get("wait", 0.0)
            )
            profile.get_accumulator().records.append(rec)

        return timing, fusion_ms

    def _record_for_plan(self, dispatches: list[DispatchLaunch]) -> None:
        """Record dispatches for the compiled plan (C++ dispatch_plan)."""
        if self._plan_record is None or self._plan_record_paused:
            return
        # Local import keeps _dispatch.py independent of the runtime layer
        # until plan recording is actually engaged.  # scoped: avoid cycle
        from alloy._runtime.metal import CompiledKernel, pso_source  # scoped: leaf util

        for dispatch in dispatches:
            buf_info: list[PlanBufferBinding] = []
            for buf in dispatch.buffers:
                root_ptr = buf.base_ptr
                nbytes = buf.metal_nbytes
                # A flat (rank<=1) binding carries no layout; prefer the
                # pre-flatten provenance reshape() stashed so the grid-shrink
                # shrink gate can see which axis the kernel walks outermost.
                dims = buf._pre_flatten_dims
                if dims is None or len(buf.shape) > 1:
                    dims = tuple(zip(buf.shape, buf.strides))
                buf_info.append(
                    PlanBufferBinding(
                        root_ptr=root_ptr,
                        byte_offset=buf._offset,
                        nbytes=nbytes,
                        dims=dims,
                    )
                )
                if root_ptr not in self._plan_buf_map:
                    self._plan_buf_map[root_ptr] = buf
            # Pull MSL + function name either from a live CompiledKernel
            # (the common path) or fall back to the pso registry that
            # from_msl populates by handle.
            msl_source = ""
            function_name = ""
            kernel = dispatch.kernel
            if isinstance(kernel, CompiledKernel):
                msl_source = kernel._msl_source
                function_name = kernel._function_name
            if not msl_source:
                src = pso_source(dispatch.pso_handle)
                if src is not None:
                    msl_source, function_name = src
            self._plan_record.append(
                RecordedDispatch(
                    pso_handle=dispatch.pso_handle,
                    buffers=tuple(buf_info),
                    grid=dispatch.grid,
                    threadgroup=dispatch.threadgroup,
                    write_indices=dispatch.write_indices,
                    debug_name=dispatch.debug_name,
                    msl_source=msl_source,
                    function_name=function_name,
                )
            )

    def register_plan(
        self,
        dispatches: list[tuple[int, list[int], list[int], Grid3D, Grid3D]],
        slots: list[tuple[int, int, int, int]],
        groups: list[list[int]],
        written_slots: list[int] | None = None,
    ) -> int:
        """Register a compiled plan with the dispatch engine."""
        return _metal_ext.register_plan(dispatches, slots, groups, written_slots or [])

    def dispatch_plan(
        self,
        plan_handle: int,
        input_updates: list[tuple[int, int, int]],
        n_dispatches: int = 0,
        defer_wait: bool = False,
        pre_copies: list[tuple[int, int, int, int, int]] | None = None,
        grid_updates: list[tuple[int, int, int, int]] | None = None,
    ) -> dict[str, float]:
        """Dispatch a registered plan with new input arrays.

        `pre_copies` are GPU-side bulk blit copies (dst_handle, dst_offset,
        src_handle, src_offset, nbytes) encoded at the head of the plan's
        command buffer — used by speculative decode to propagate DeltaNet
        recurrent state without per-layer Python `.copy_()` dispatches.

        `grid_updates` are per-call launch-grid overrides (flat_dispatch_idx,
        gx, gy, gz) — grid-shrunk chunk prefill keeps the plan compiled once at the max
        sequence length but dispatches an exact threadgroup count for the real
        prompt length, so padding tiles cost no GPU work.
        """
        default_dispatcher().dispatch_count += n_dispatches
        return _metal_ext.dispatch_plan(
            plan_handle, input_updates, defer_wait=defer_wait,
            pre_copies=pre_copies if pre_copies is not None else [],
            grid_updates=grid_updates if grid_updates is not None else [],
        )

    @staticmethod
    def gpu_sync() -> None:
        """Wait for any pending async command buffer."""
        _metal_ext.gpu_sync()

    # --- Single-op compilation ---

    def compile_kernel(self, kernel: KernelFunction, op: LazyOp) -> DispatchLaunch:
        """Compile a single LazyOp into a typed launch record."""
        return kernel._compile_op(op)

    # --- Singleton ---

    _default_lock: threading.Lock = threading.Lock()

    @classmethod
    def default(cls) -> DispatchEngine:
        """Get or create the default DispatchEngine instance. Thread-safe."""
        if cls._default is None:
            with cls._default_lock:
                if cls._default is None:
                    cls._default = DispatchEngine()
        return cls._default

    @classmethod
    def set_default(cls, instance: DispatchEngine) -> None:
        """Override the default instance (for testing)."""
        cls._default = instance


# Module-level singleton for internal use
_engine = DispatchEngine.default()


# --- Helpers ---


def _populate_buf_sets(op: LazyOp) -> None:
    """Populate read_bufs/write_bufs from buffer_args."""
    op.read_bufs = set()
    op.write_bufs = set()
    for pname, arg in op.buffer_args:
        handle = (
            arg._parent_handle if arg._parent_handle >= 0 else _alloy_handle_map.get(arg.data_ptr)
        )
        # Also check base_ptr — a view at base+offset shares the same
        # allocation as the buffer at base. Without this, dependency analysis
        # assigns different handles to overlapping memory, missing RAW/WAR
        # conflicts and corrupting training backward gradients.
        if handle is None and arg.base_ptr != arg.data_ptr:
            handle = _alloy_handle_map.get(arg.base_ptr)
        if handle is None:
            # External buffer (torch tensor): use negative base_ptr as sentinel
            # so all views of the same storage share one dependency handle.
            ptr = arg._raw_ptr or arg.data_ptr
            if ptr:
                handle = -ptr  # negative to avoid collision with alloy handles
            else:
                continue
        if pname in op.output_params:
            op.write_bufs.add(handle)
        else:
            op.read_bufs.add(handle)


def _fusion_cache_key(ops: list[LazyOp]) -> tuple[object, ...]:
    """Build cache key from op signatures and buffer-sharing edges."""
    op_keys = tuple(o._cache_key for o in ops if o._cache_key)
    ptr_to_origin: dict[int, tuple[int, int]] = {}
    edges: list[tuple[int, int, int, int]] = []
    for oi, o in enumerate(ops):
        for pi, (_, arg) in enumerate(o.buffer_args):
            ptr = arg.data_ptr
            if ptr in ptr_to_origin:
                edges.append((oi, pi, *ptr_to_origin[ptr]))
            ptr_to_origin[ptr] = (oi, pi)
    return (op_keys, tuple(edges))


def _build_dependency_groups(
    dispatches: list[DispatchLaunch],
    read_sets: list[set[int]],
    write_sets: list[set[int]],
) -> list[list[DispatchLaunch]]:
    """Partition dispatches into barrier-free groups based on read/write conflicts.

    Groups execute serially with a barrier between them, so a dispatch
    appended to the last group already sees every earlier group's writes.
    The only conflict that forces a new group is with the currently-open
    (last) group. Each dispatch does one intersection check against the
    last group's aggregated reads + writes — O(N) total rather than the
    O(N × G) scan a per-prior-group check would require.
    """
    if not dispatches:
        return []
    groups: list[tuple[list[DispatchLaunch], set[int], set[int]]] = []
    for i, dispatch in enumerate(dispatches):
        reads = read_sets[i]
        writes = write_sets[i]
        if not groups:
            groups.append(([dispatch], set(writes), set(reads)))
            continue
        # Fast path: no reads/writes → no conflict possible, just append.
        if not reads and not writes:
            groups[-1][0].append(dispatch)
            continue
        # Check only the last (open) group for RAW / WAR / WAW conflicts.
        _, g_writes, g_reads = groups[-1]
        if (reads & g_writes) or (writes & g_reads) or (writes & g_writes):
            groups.append(([dispatch], set(writes), set(reads)))
        else:
            groups[-1][0].append(dispatch)
            g_writes.update(writes)
            g_reads.update(reads)
    return [g[0] for g in groups]


def _group_and_dispatch(
    dispatches: list[DispatchLaunch],
    ops: list[LazyOp],
    op_indices_per_dispatch: list[frozenset[int]],
) -> dict[str, float]:
    """Build dependency groups from fused op read/write sets, then dispatch."""
    if not dispatches:
        return {"gpu": 0.0, "encode": 0.0, "wait": 0.0, "copy": 0.0}
    read_sets: list[set[int]] = []
    write_sets: list[set[int]] = []
    for indices in op_indices_per_dispatch:
        reads: set[int] = set()
        writes: set[int] = set()
        for oi in indices:
            if oi < len(ops):
                reads.update(ops[oi].read_bufs)
                writes.update(ops[oi].write_bufs)
        read_sets.append(reads)
        write_sets.append(writes)
    groups = _build_dependency_groups(dispatches, read_sets, write_sets)
    default_dispatcher().dispatch_count += len(dispatches)
    try:
        return _metal_ext.dispatch([_batch_to_v2(g) for g in groups])
    except Exception as exc:
        logger.error(
            "metal_dispatch_error",
            n_dispatches=len(dispatches),
            n_groups=len(groups),
            error=str(exc),
        )
        raise
