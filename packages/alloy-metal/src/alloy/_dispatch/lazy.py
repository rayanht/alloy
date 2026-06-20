"""Lazy evaluation graph — LazyOp, materialization, and _queue_op dispatch.

Owns the deferred compile+dispatch pipeline: _queue_op creates LazyOps,
Materializer chains them until data is needed, then flushes via
DispatchEngine.dispatch_ops.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from alloy._dispatch.buf_utils import (
    _NON_FUSABLE_ELEM_KERNELS,
    _alloc_aligned,
    _alloc_phantom,
    _alloc_ptrs_this_run,
    _unique_lazy_buffers,
    is_record_only,
)
from alloy._compiler.dtypes import DType, float32, from_ir
from alloy.log import get_logger
from alloy._runtime import profile
from alloy._runtime.alloy_buffer import AlloyBuffer

logger = get_logger("alloy.dispatch")

if TYPE_CHECKING:
    from alloy._compiler.dispatch_spec import DispatchContract
    from alloy._compiler.tile_ir import TileFunction
    from alloy._dispatch.kernel import KernelFunction
    from alloy._dispatch.fusion_types import Grid3D


# ===================================================================
# LazyOp — a kernel call queued during lazy mode
# ===================================================================


class LazyOp:
    """A kernel call queued during lazy mode, analyzed at flush time."""

    kernel: KernelFunction
    grid: tuple[int, int, int]
    func: TileFunction
    constexpr_values: dict[str, int | float | bool | tuple[int, ...]]
    buffer_args: list[tuple[str, AlloyBuffer]]
    buffer_dtypes: dict[str, str]
    buffer_shapes: dict[str, tuple[int, ...]]
    output_params: set[str]
    input_producers: dict[str, LazyOp]
    _cache_key: tuple[object, ...] | None
    _dispatched: bool
    read_bufs: set[int]
    write_bufs: set[int]

    __slots__ = (
        "kernel",
        "grid",
        "func",
        "constexpr_values",
        "buffer_args",
        "buffer_dtypes",
        "buffer_shapes",
        "output_params",
        "input_producers",
        "_cache_key",
        "_dispatched",
        "read_bufs",
        "write_bufs",
    )

    def __init__(
        self,
        kernel: KernelFunction | None = None,
        grid: tuple[int, int, int] | None = None,
        func: TileFunction | None = None,
        constexpr_values: dict[str, int | float | bool | tuple[int, ...]] | None = None,
        buffer_args: list[tuple[str, AlloyBuffer]] | None = None,
        buffer_dtypes: dict[str, str] | None = None,
        buffer_shapes: dict[str, tuple[int, ...]] | None = None,
        output_params: set[str] | None = None,
        input_producers: dict[str, LazyOp] | None = None,
    ) -> None:
        self.kernel = kernel
        self.grid = grid
        self.func = func
        self.constexpr_values = constexpr_values if constexpr_values is not None else {}
        self.buffer_args = buffer_args if buffer_args is not None else []
        self.buffer_dtypes = buffer_dtypes if buffer_dtypes is not None else {}
        self.buffer_shapes = buffer_shapes if buffer_shapes is not None else {}
        self.output_params = output_params if output_params is not None else set()
        self.input_producers = input_producers if input_producers is not None else {}
        self._cache_key = None
        self._dispatched = False
        self.read_bufs = set()
        self.write_bufs = set()

    def is_elem_op(self) -> bool:
        k = self.kernel
        if k.name in _NON_FUSABLE_ELEM_KERNELS:
            return False
        if not (not k._has_tg_ops and not k._has_non_elem and not k._is_tile):
            return False
        return True


# ===================================================================
# Output allocation
# ===================================================================


def _allocate_outputs(
    output_params: set[str],
    out_shape: tuple[int, ...] | None,
    out_shapes: dict[str, tuple[int, ...]],
    spec: DispatchContract | None,
    buffer_dtypes: dict[str, str],
    buffer_args: list[tuple[str, AlloyBuffer]],
    buffer_shapes: dict[str, tuple[int, ...]],
) -> set[str]:
    """Allocate output buffers for params not provided by the caller. Returns names of allocated outputs."""
    allocated: set[str] = set()
    if out_shape is None:
        return allocated
    for pname in output_params:
        spec_shape = out_shapes.get(pname)
        fb = out_shape
        if spec_shape and fb:
            if _prod(spec_shape) < _prod(fb):
                spec_shape = None
        shape = spec_shape if spec_shape is not None else fb
        out_dtype: DType = float32
        if spec is not None and pname in spec.outputs:
            out_dtype = from_ir(spec.outputs[pname].dtype)
        # Record-only compile: kernel outputs are phantom (no Metal page). Their
        # contents are never read during tracing, and the GPU dispatch is skipped.
        out_buf = (
            _alloc_phantom(shape, out_dtype)
            if is_record_only()
            else _alloc_aligned(shape, out_dtype)
        )
        _alloc_ptrs_this_run.add(out_buf.base_ptr)
        buffer_args.append((pname, out_buf))
        buffer_shapes[pname] = shape
        buffer_dtypes[pname] = out_buf._dtype.ir
        allocated.add(pname)
    return allocated


def _prod(shape: tuple[int, ...]) -> int:
    r = 1
    for d in shape:
        r *= d
    return r


# ===================================================================
# Enqueue: create LazyOp + Materializer, attach to outputs
# ===================================================================


def _enqueue_and_return(
    kernel: KernelFunction,
    func: TileFunction,
    grid: Grid3D,
    constexpr_values: dict[str, int | float | bool | tuple[int, ...]],
    buffer_args: list[tuple[str, AlloyBuffer]],
    buffer_dtypes: dict[str, str],
    buffer_shapes: dict[str, tuple[int, ...]],
    input_producers: dict[str, LazyOp],
    lazy_inputs: list[AlloyBuffer],
    cache_key: tuple[object, ...],
    allocated_outputs: set[str] | None = None,
) -> AlloyBuffer | tuple[AlloyBuffer, ...] | None:
    """Create LazyOp + Materializer, attach to output buffers, return results."""
    queued_op = LazyOp.__new__(LazyOp)
    queued_op.kernel = kernel
    queued_op.grid = grid
    queued_op.func = func
    queued_op.constexpr_values = constexpr_values
    queued_op.buffer_args = buffer_args
    queued_op.buffer_dtypes = buffer_dtypes
    queued_op.buffer_shapes = buffer_shapes
    queued_op.output_params = kernel._output_params
    queued_op.input_producers = input_producers
    queued_op._cache_key = cache_key
    queued_op._dispatched = False
    queued_op.read_bufs = set()
    queued_op.write_bufs = set()

    mat = Materializer(queued_op, lazy_inputs, cache_key=cache_key)

    if kernel._output_params:
        outputs: list[AlloyBuffer] = []
        for pname, arg in buffer_args:
            if pname in kernel._output_params:
                arg._materializer = mat
                arg._producer = queued_op
                arg._owns_aligned = allocated_outputs is None or pname in allocated_outputs
                outputs.append(arg)
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    # Side-effect kernel — attach to inputs
    for lb in lazy_inputs:
        lb._materializer = mat
    if not lazy_inputs:
        mat()
    return None


# ===================================================================
# _queue_op — the unified dispatch entry point
# ===================================================================


def _queue_op(
    kernel: KernelFunction,
    grid: tuple[int, ...] | None,
    args: tuple[AlloyBuffer | np.ndarray | int | float, ...],
    kwargs: dict[str, int | float | bool],
) -> AlloyBuffer | tuple[AlloyBuffer, ...] | None:
    """Unified entry: resolve inputs, trace kernel, derive grid, queue for lazy dispatch."""
    _p = profile._profile_enabled
    if _p:
        _total_t0 = time.perf_counter_ns()
        _queue_t0 = time.perf_counter_ns()

    # --- Resolve inputs, constexprs, trace ---
    inputs = kernel.resolve_inputs(args)
    constexprs = kernel.resolve_constexprs(
        kwargs, inputs.buffer_dtypes, inputs.buffer_args, inputs.lazy_inputs
    )
    inputs.stride_meta_into(constexprs.values)

    if _p:
        _queue_ms = (time.perf_counter_ns() - _queue_t0) / 1e6

    traced = kernel.trace_and_plan(
        constexprs.values,
        constexprs.compiler_options,
        inputs.buffer_dtypes,
        inputs.buffer_shapes,
        inputs.buffer_args,
        grid,
    )

    if _p:
        _trace_ms = traced.trace_ms
        _grid_ms = traced.grid_ms

    allocated = _allocate_outputs(
        kernel._output_params,
        traced.out_shape,
        traced.out_shapes,
        traced.spec,
        inputs.buffer_dtypes,
        inputs.buffer_args,
        inputs.buffer_shapes,
    )

    # --- Build dispatch cache key ---
    _cv_items = tuple((k, v) for k, v in constexprs.values.items() if k != "_input_shapes")
    dispatch_cache_key = (
        kernel._source,
        traced.func.fingerprint,
        _cv_items,
        tuple(inputs.buffer_dtypes.items()),
        tuple(inputs.buffer_shapes.items()),
        traced.grid,
    )

    if _p:
        from alloy._dispatch.dispatch import (
            _engine,
        )  # scoped: avoid cycle (_dispatch imports LazyOp from this module)

        _engine.op_profile[id(kernel)] = {
            "trace_ms": _trace_ms,
            "grid_ms": _grid_ms,
            "queue_ms": _queue_ms,
            "total_t0": _total_t0,
        }

    # --- Enqueue ---
    return _enqueue_and_return(
        kernel,
        traced.func,
        traced.grid,
        constexprs.values,
        inputs.buffer_args,
        inputs.buffer_dtypes,
        inputs.buffer_shapes,
        inputs.input_producers,
        inputs.lazy_inputs,
        dispatch_cache_key,
        allocated,
    )


# ===================================================================
# Data containers for resolve_inputs / resolve_constexprs / trace_and_plan
# ===================================================================


@dataclass
class ResolvedInputs:
    """Result of KernelFunction.resolve_inputs()."""

    buffer_args: list[tuple[str, AlloyBuffer]]
    buffer_shapes: dict[str, tuple[int, ...]]
    buffer_dtypes: dict[str, str]
    input_producers: dict[str, LazyOp]
    lazy_inputs: list[AlloyBuffer]
    _stride_meta: dict[str, int | tuple[int, ...]] = field(default_factory=dict)

    def stride_meta_into(
        self, constexpr_values: dict[str, int | float | bool | tuple[int, ...]]
    ) -> None:
        """Merge stride metadata into constexpr values."""
        constexpr_values.update(self._stride_meta)


@dataclass
class ResolvedConstexprs:
    """Result of KernelFunction.resolve_constexprs()."""

    values: dict[str, int | float | bool | tuple[int, ...]]
    compiler_options: dict[str, int | list[int]]


@dataclass
class TraceResult:
    """Result of KernelFunction.trace_and_plan()."""

    func: TileFunction
    grid: Grid3D
    out_shape: tuple[int, ...] | None
    out_shapes: dict[str, tuple[int, ...]]
    spec: DispatchContract | None
    trace_ms: float = 0.0
    grid_ms: float = 0.0


# ===================================================================
# Materialization infrastructure
# ===================================================================


def _collect_pending_ops(
    roots: tuple[AlloyBuffer, ...] | list[AlloyBuffer],
) -> tuple[list[LazyOp], list[AlloyBuffer]]:
    """Collect pending LazyOps from materializer chain in topological order."""
    ops: list[LazyOp] = []
    synced_buffers: list[AlloyBuffer] = []
    for lb in _unique_lazy_buffers(roots):
        if lb._materializer is not None and not lb._materializer._done:
            lb._materializer._collect_ops(ops, synced_buffers)
            synced_buffers.append(lb)
    return ops, _unique_lazy_buffers(synced_buffers)


def _finalize_materializers(
    roots: tuple[AlloyBuffer, ...] | list[AlloyBuffer],
    synced_buffers: list[AlloyBuffer],
) -> None:
    """Finalize lazy state after materializing a set of roots."""
    for lb in _unique_lazy_buffers(synced_buffers):
        lb._materializer = None
    for lb in _unique_lazy_buffers(roots):
        lb._materializer = None


def _materialize_many(roots: tuple[AlloyBuffer, ...] | list[AlloyBuffer]) -> None:
    """Flush all pending LazyOps reachable from roots."""
    ops, synced_buffers = _collect_pending_ops(roots)
    if not ops:
        return
    from alloy._dispatch.dispatch import (
        _engine,
    )  # scoped: circular — _dispatch imports LazyOp from this module

    _t0 = time.perf_counter()
    logger.debug("materialize_start", n_ops=len(ops), n_roots=len(roots))
    _engine.dispatch_ops(ops, roots=roots)
    _finalize_materializers(roots, synced_buffers)
    logger.debug(
        "materialize_end",
        n_ops=len(ops),
        n_roots=len(roots),
        took_ms=round((time.perf_counter() - _t0) * 1000.0, 2),
    )


class Materializer:
    """Compiles+dispatches a LazyOp and its unmaterialized inputs in one command buffer.

    Uses __slots__ to avoid Python GC traversal of this object.
    """

    __slots__ = ("_op", "_inputs", "_done", "_cache_key")

    def __init__(
        self,
        op: LazyOp,
        inputs: list[AlloyBuffer],
        cache_key: tuple[object, ...] | None = None,
    ) -> None:
        self._op: LazyOp = op
        self._inputs: list[AlloyBuffer] = inputs
        self._done: bool = False
        self._cache_key: tuple[object, ...] | None = cache_key

    def __call__(self) -> None:
        if self._done:
            return
        self._done = True

        _p = profile._profile_enabled
        if _p:
            _t = time.perf_counter_ns()
        op = self._op

        ops, synced_buffers = _collect_pending_ops(self._inputs)
        if not op._dispatched:
            ops.append(op)
        if not ops:
            return

        from alloy._dispatch.dispatch import (
            _engine,
        )  # scoped: avoid cycle (_dispatch imports LazyOp from this module)

        timing, _fusion_ms = _engine.dispatch_ops(ops)

        if _p:
            rec = profile.DispatchRecord(name=op.kernel.name)
            rec.phases[profile.QUEUE] = _engine.op_profile.get(id(op), {}).get("queue_ms", 0.0)
            rec.phases[profile.FUSION] = _fusion_ms
            rec.phases[profile.GPU] = timing["gpu"]
            rec.phases[profile.ENCODE] = timing["encode"]
            rec.phases[profile.WAIT] = timing["wait"]
            rec.phases[profile.COPY] = timing["copy"]
            if _engine.op_profile.get(id(op), {}).get("total_t0", 0):
                rec.phases[profile.TOTAL] = (
                    time.perf_counter_ns() - _engine.op_profile.get(id(op), {}).get("total_t0", 0)
                ) / 1e6
            profile.get_accumulator().records.append(rec)

        _finalize_materializers(self._inputs, synced_buffers)

    def _collect_ops(self, out: list[LazyOp], synced: list[AlloyBuffer]) -> None:
        """Collect LazyOps in topological order (inputs before consumers).

        Iterative post-order DFS: a fully-lazy graph with no mid-trace syncs
        (e.g. a multi-layer conformer once `constant_pad_nd`/`unfold` stay lazy)
        chains thousands of ops deep and overflows Python's recursion limit, so
        an explicit stack replaces the call stack. `_done` is set on first
        encounter (pre-order) to dedup shared sub-DAGs; `out.append` happens on
        the post-visit so every input lands ahead of its consumer."""
        stack: list[tuple[Materializer, bool]] = [(self, False)]
        while stack:
            node, post = stack.pop()
            if post:
                if not node._op._dispatched:
                    out.append(node._op)
                continue
            if node._done:
                continue
            node._done = True
            stack.append((node, True))
            for lb in reversed(node._inputs):
                if lb._materializer is not None and not lb._materializer._done:
                    synced.append(lb)
                    stack.append((lb._materializer, False))
