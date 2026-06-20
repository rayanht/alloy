"""Fusion compilation — plan execution, Metal compilation, and orchestration.

Takes FusionPlan decisions from _fusion_analysis and produces typed Metal launches.
Uses _fusion_transform for IR composition.

Pipeline: _compile_fusion_plans(ops, plans) → list[FusedPair]
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from alloy._dispatch.buf_utils import _builtin_min, _normalize_grid
from alloy._compiler.tile_ir import ColumnSliceInfo, Store, TileFunction, shallow_clone_for_fusion
from alloy._compiler.tile_msl import emit_msl_from_tile_ir
from alloy._dispatch.fusion_analysis import _fused_name
from alloy.log import get_logger
from alloy._dispatch.fusion_types import (
    DispatchLaunch,
    FusedPair,
    FusionKind,
    FusionPlan,
    FusionUnsupported,
    Grid3D,
)
from alloy._dispatch.fusion_transform import (
    _apply_anchor_prologue,
    _apply_chain,
    _buf_elem_count,
    _build_anchor_context,
    _build_buf_map,
    _evaluate_anchor_grid,
    _find_ir_store,
    _set_col_slice_on_stores,
)
from alloy._dispatch.lazy import LazyOp
from alloy._dispatch.multi_root import Island, compile_multi_root
from alloy._dispatch.reduce_fold import compose_reduce_fold
from alloy._dispatch.row_pass import (
    RowPassGroup,
    _find_sinks,
    _is_row_reduce,
    compose,
    compose_chunked,
)
from alloy._runtime.alloy_buffer import AlloyBuffer
from alloy._dispatch.cache import CacheManager, msl_hash
from alloy._dispatch.observe import notify_compiled
from alloy._runtime.metal import CompiledKernel, default_device

if TYPE_CHECKING:
    from alloy._dispatch.kernel import KernelFunction

CompileIndividual = Callable[["KernelFunction", LazyOp], DispatchLaunch]

logger = get_logger("alloy.fusion")

# Dedupe set for fusion_compile_error: the same codegen failure fires once per
# plan that touches the kernel during a warmup sweep. Warn once per
# (kernel, pass_name) per process; ALLOY_LOG_FUSION=debug logs the repeats.
_warned_fusion_failures: set[tuple[str, str]] = set()


# ===================================================================
# Column-slice epilogue detection
# ===================================================================


def _detect_column_slice_epilogue(
    first_epi_op: LazyOp, anchor_out_arr: AlloyBuffer | None
) -> tuple[int, int, int] | None:
    """Detect if an epilogue op reads a column slice of the anchor output.

    Returns (col_start, col_end, slice_width) if the first epilogue op's input
    is a column slice (non-contiguous view with row_stride > slice_width) of the
    anchor output. Returns None if it reads the full output or a different buffer.
    """
    if anchor_out_arr is None:
        return None

    cv = first_epi_op.constexpr_values
    input_param = None
    input_buf = None
    for pn, buf in first_epi_op.buffer_args:
        if pn not in first_epi_op.output_params:
            input_param = pn
            input_buf = buf
            break
    if input_buf is None or not input_buf.shares_allocation(anchor_out_arr):
        return None

    shape_key = f"_{input_param}_shape"
    strides_key = f"_{input_param}_strides"
    if shape_key not in cv or strides_key not in cv:
        return None

    shape = cv[shape_key]
    strides = cv[strides_key]
    if not isinstance(shape, tuple) or not isinstance(strides, tuple):
        return None
    # View element offset lives on the buffer; runtime binding applies it.
    offset = input_buf._offset // input_buf._dtype.itemsize if input_buf._offset else 0

    if len(shape) != 2 or len(strides) != 2:
        return None

    _rows, slice_width = shape
    row_stride, col_stride = strides

    # Column slice: row_stride > slice_width (rows span the full parent width)
    if col_stride != 1 or row_stride <= slice_width:
        return None

    col_start = offset  # element offset = column start (when col_stride=1)
    col_end = col_start + slice_width
    return (col_start, col_end, slice_width)


# ===================================================================
# Section 8: Entry points — compile elem chain / anchor with chains
# ===================================================================


def compile_elem_chain(
    queue: list[LazyOp], indices: list[int]
) -> tuple[TileFunction, list[AlloyBuffer], Grid3D, None]:
    """Resolve standalone elem chain using IR-level fusion."""
    first_op = queue[indices[0]]
    func = shallow_clone_for_fusion(first_op.func)
    func.name = "_fused"
    buffer_map = _build_buf_map(first_op, func)
    grid = _normalize_grid(first_op.grid)

    if len(indices) > 1:
        store_op = _find_ir_store(func)
        if store_op is None or store_op.ptr is None:
            raise FusionUnsupported("No Store op in first elem function")

        out_param_name = store_op.ptr.name
        first_out_arr = buffer_map.get(out_param_name)
        if first_out_arr is None:
            raise FusionUnsupported(f"No output array for store param '{out_param_name}'")

        remaining_ops = [queue[idx] for idx in indices[1:]]
        total_elems = func.constexpr_values.get("N")
        if total_elems is None:
            total_elems = grid[0] * grid[1] * grid[2] if isinstance(grid, tuple) else 1
        composed = _apply_chain(
            func,
            buffer_map,
            remaining_ops,
            first_out_arr,
            param_name=out_param_name,
            total_elems=total_elems,
            label="elem-chain",
        )

        # Elem chains: ensure the Store's transform_source_name is set so
        # the emitter knows which operand is the chain flow (maps to _ev).
        # _compose_chain sets this for epilogues (via anchor_out_arr) but
        # not for elem chains where the source is the first op's store value.
        if composed.source_param_name is None:
            store_op2 = _find_ir_store(func)
            if (
                store_op2 is not None
                and store_op2.transform
                and store_op2.transform_source_name is None
            ):
                # The chain flow enters the transform as the store value.
                # Find the first non-produced operand that matches a local
                # variable (not a buffer param) — that's the chain flow.
                buf_params = {p.name for p in func.params if not p.is_constexpr}
                produced = {op.result.name for op in store_op2.transform if op.result}
                for t_op in store_op2.transform:
                    for v in t_op.operand_values():
                        if v.name not in produced and v.name not in buf_params:
                            store_op2.transform_source_name = v.name
                            break
                    if store_op2.transform_source_name:
                        break

    buf_arrs = [buffer_map[p.name] for p in func.params if not p.is_constexpr]
    return func, buf_arrs, grid, None


def compile_anchor_with_chains(
    anchor_op: LazyOp,
    epi_indices: list[int],
    pro_indices: list[int],
    queue: list[LazyOp],
    extra_epi_chains: list[tuple[list[int], list[LazyOp]]] | None = None,
) -> tuple[TileFunction, list[AlloyBuffer], tuple[int, int, int]]:
    """Resolve anchor + absorbed elems entirely at IR level."""
    ctx = _build_anchor_context(anchor_op)
    if ctx.anchor_out_arr is None:
        raise FusionUnsupported("Anchor op has no output array")

    if epi_indices:
        # Detect column-slice epilogue: first epilogue op reads a column slice
        # of the anchor output (shares allocation, has stride metadata with
        # row_stride > slice_width). Use clone_param so the anchor's Store
        # stays untouched and a new Store writes the epilogue result.
        first_epi_op = queue[epi_indices[0]]
        epi_slice_info = _detect_column_slice_epilogue(first_epi_op, ctx.anchor_out_arr)

        if epi_slice_info is not None:
            col_start, col_end, slice_width = epi_slice_info
            _apply_chain(
                ctx.func,
                ctx.buffer_map,
                [queue[idx] for idx in epi_indices],
                ctx.anchor_out_arr,
                param_name=ctx.output_param,
                total_elems=_buf_elem_count(ctx.anchor_out_arr),
                label="epilogue",
                clone_param="_epi_slice",
            )
            _set_col_slice_on_stores(
                ctx.func,
                "_epi_slice",
                ColumnSliceInfo(col_start=col_start, col_end=col_end, out_stride=slice_width),
            )
        else:
            _apply_chain(
                ctx.func,
                ctx.buffer_map,
                [queue[idx] for idx in epi_indices],
                ctx.anchor_out_arr,
                param_name=ctx.output_param,
                total_elems=_buf_elem_count(ctx.anchor_out_arr),
                label="epilogue",
            )

    if extra_epi_chains:
        for branch_idx, (branch_indices, branch_queue) in enumerate(extra_epi_chains):
            clone_name = f"_branch_{branch_idx}"
            _apply_chain(
                ctx.func,
                ctx.buffer_map,
                [branch_queue[idx] for idx in branch_indices],
                ctx.anchor_out_arr,
                param_name=ctx.output_param,
                total_elems=_buf_elem_count(ctx.anchor_out_arr),
                label=f"branch_{branch_idx}",
                clone_param=clone_name,
            )
            # Column-slice branch: detect and set on Store nodes
            first_branch_op = branch_queue[branch_indices[0]]
            branch_slice = _detect_column_slice_epilogue(first_branch_op, ctx.anchor_out_arr)
            if branch_slice is not None:
                cs, ce, sw = branch_slice
                _set_col_slice_on_stores(
                    ctx.func,
                    clone_name,
                    ColumnSliceInfo(col_start=cs, col_end=ce, out_stride=sw),
                )

    _apply_anchor_prologue(ctx, pro_indices, queue)

    buf_arrs = [ctx.buffer_map[p.name] for p in ctx.func.params if not p.is_constexpr]
    return ctx.func, buf_arrs, _evaluate_anchor_grid(ctx)


# ===================================================================
# Section 9: Fusion plan compilation (orchestration + degradation)
# ===================================================================


@dataclass(slots=True)
class FusionCompileContext:
    ops: list[LazyOp]
    roots: set[int]
    compile_individual: CompileIndividual
    dispatch_entries: list[FusedPair]

    def compile_one(self, idx: int) -> DispatchLaunch:
        kernel = self.ops[idx].kernel
        if kernel is None:
            raise RuntimeError(f"Cannot compile op {idx}: missing kernel")
        return self.compile_individual(kernel, self.ops[idx])

    def append(self, dispatch: DispatchLaunch, indices: set[int]) -> None:
        self.dispatch_entries.append((dispatch, indices))

    def fallback(self, indices: set[int]) -> None:
        logger.debug("fusion_fallback", n_ops=len(indices), indices=sorted(indices))
        for idx in sorted(indices):
            self.append(self.compile_one(idx), {idx})


class IndividualFusionCompiler:
    name = "individual"
    kind = FusionKind.INDIVIDUAL

    def compile(self, plan: FusionPlan, context: FusionCompileContext) -> None:
        idx = plan.idx
        if idx is None:
            raise RuntimeError("Individual fusion plan missing op index")
        context.append(context.compile_one(idx), plan.indices)


class AnchorFusionCompiler:
    name = "anchor"
    kind = FusionKind.ANCHOR

    def compile(self, plan: FusionPlan, context: FusionCompileContext) -> None:
        anchor_idx = plan.anchor_idx
        if anchor_idx is None:
            context.fallback(plan.indices)
            return

        anchor_kernel = context.ops[anchor_idx].kernel
        if anchor_kernel is None:
            context.fallback(plan.indices)
            return

        pro_chain = plan.pro_chain
        epi_chain = plan.epi_chain
        extra_branches = plan.extra_branches
        pro_queue = [context.ops[idx] for idx in pro_chain]
        epi_queue = [context.ops[idx] for idx in epi_chain]
        combined = pro_queue + epi_queue
        n_pro = len(pro_queue)

        extra_epi: list[tuple[list[int], list[LazyOp]]] | None = None
        if extra_branches:
            extra_epi = [
                (list(range(len(branch))), [context.ops[idx] for idx in branch])
                for branch in extra_branches
            ]

        try:
            func, buf_arrs, grid_3d = compile_anchor_with_chains(
                context.ops[anchor_idx],
                epi_indices=list(range(n_pro, n_pro + len(epi_queue))),
                pro_indices=list(range(n_pro)),
                queue=combined,
                extra_epi_chains=extra_epi,
            )
            all_names = pro_chain + [anchor_idx] + epi_chain
            for branch in extra_branches:
                all_names.extend(branch)
            func.name = _fused_name(context.ops, all_names)
            compiled, msl = _compile_fused(func)
        except FusionUnsupported:
            context.fallback(plan.indices)
            return
        except FusionCompileError as exc:
            warn_key = (func.name, self.name)
            if warn_key not in _warned_fusion_failures:
                _warned_fusion_failures.add(warn_key)
                logger.warning(
                    "fusion_compile_error",
                    kernel=func.name, pass_name=self.name, error=str(exc),
                )
            else:
                logger.debug(
                    "fusion_compile_error_repeat",
                    kernel=func.name, pass_name=self.name,
                )
            context.fallback(plan.indices)
            return

        tg_3d = _compute_tg(compiled, msl)
        write_indices = _write_indices_from_params(func, anchor_kernel._output_params)
        context.append(
            DispatchLaunch(
                kernel=compiled,
                buffers=tuple(buf_arrs),
                grid=grid_3d,
                threadgroup=tg_3d,
                write_indices=write_indices,
                debug_name=func.name,
            ),
            plan.indices,
        )


class RowPassCompiler:
    name = "row_pass"
    kind = FusionKind.ROW_PASS

    def compile(self, plan: FusionPlan, context: FusionCompileContext) -> None:
        indices = sorted(plan.indices)
        if not indices:
            context.fallback(plan.indices)
            return

        seed = indices[0]
        m_n: tuple[int, int] | None = None
        for gi in indices:
            if context.ops[gi].kernel and _is_row_reduce(context.ops[gi]):
                cv = context.ops[gi].constexpr_values
                if isinstance(cv.get("M"), int) and isinstance(cv.get("N"), int):
                    m_n = (cv["M"], cv["N"])
                    break
        if m_n is None:
            context.fallback(plan.indices)
            return

        group = RowPassGroup(op_indices=indices, seed_idx=seed, M=m_n[0], N=m_n[1])
        if m_n[1] <= 1024:
            result = compose(context.ops, group, roots=context.roots)
        else:
            result = compose_chunked(context.ops, group, roots=context.roots)
        if result is None:
            context.fallback(plan.indices)
            return

        func, buf_arrs, grid = result
        func.name = _fused_name(context.ops, indices)
        try:
            compiled, msl = _compile_fused(func)
        except FusionCompileError as exc:
            warn_key = (func.name, self.name)
            if warn_key not in _warned_fusion_failures:
                _warned_fusion_failures.add(warn_key)
                logger.warning(
                    "fusion_compile_error",
                    kernel=func.name, pass_name=self.name, error=str(exc),
                )
            else:
                logger.debug(
                    "fusion_compile_error_repeat",
                    kernel=func.name, pass_name=self.name,
                )
            context.fallback(plan.indices)
            return

        tg_3d = _compute_tg(compiled, msl)
        sinks_count = len(_find_sinks(context.ops, set(indices), context.roots))
        write_indices = frozenset(range(len(buf_arrs) - sinks_count, len(buf_arrs)))
        context.append(
            DispatchLaunch(
                kernel=compiled,
                buffers=tuple(buf_arrs),
                grid=grid,
                threadgroup=tg_3d,
                write_indices=write_indices,
                debug_name=func.name,
            ),
            plan.indices,
        )


class MultiRootCompiler:
    name = "multi_root"
    kind = FusionKind.MULTI_ROOT

    def compile(self, plan: FusionPlan, context: FusionCompileContext) -> None:
        topo_order = plan.chain or sorted(plan.indices)
        island = Island(
            indices=sorted(plan.indices),
            writebacks=set(plan.epi_chain or []),
            order=topo_order,
        )
        try:
            func, buf_arrs, grid_3d, _ = compile_multi_root(context.ops, island)
            func.name = _fused_name(context.ops, topo_order)
            compiled, msl = _compile_fused(func)
        except FusionUnsupported:
            context.fallback(plan.indices)
            return
        except FusionCompileError as exc:
            warn_key = (func.name, self.name)
            if warn_key not in _warned_fusion_failures:
                _warned_fusion_failures.add(warn_key)
                logger.warning(
                    "fusion_compile_error",
                    kernel=func.name, pass_name=self.name, error=str(exc),
                )
            else:
                logger.debug(
                    "fusion_compile_error_repeat",
                    kernel=func.name, pass_name=self.name,
                )
            context.fallback(plan.indices)
            return

        tg_3d = _compute_tg(compiled, msl)
        write_indices = _fused_write_indices(func)
        context.append(
            DispatchLaunch(
                kernel=compiled,
                buffers=tuple(buf_arrs),
                grid=grid_3d,
                threadgroup=tg_3d,
                write_indices=write_indices,
                debug_name=func.name,
            ),
            plan.indices,
        )


class RmsNormFoldCompiler:
    name = "reduce_fold"
    kind = FusionKind.REDUCE_FOLD

    def compile(self, plan: FusionPlan, context: FusionCompileContext) -> None:
        gemv_idx = plan.anchor_idx
        rms_idx = plan.pro_chain[0] if plan.pro_chain else None
        if gemv_idx is None or rms_idx is None:
            context.fallback(plan.indices)
            return
        gemv_op = context.ops[gemv_idx]
        rms_op = context.ops[rms_idx]
        fused_name = f"{rms_op.kernel.name}_{gemv_op.kernel.name}"
        try:
            func, buf_arrs, grid_3d = compose_reduce_fold(gemv_op, rms_op)
            compiled, msl = _compile_fused(func)
        except FusionUnsupported:
            context.fallback(plan.indices)
            return
        except FusionCompileError as exc:
            warn_key = (fused_name, self.name)
            if warn_key not in _warned_fusion_failures:
                _warned_fusion_failures.add(warn_key)
                logger.warning("fusion_compile_error", kernel=fused_name,
                               pass_name=self.name, error=str(exc))
            context.fallback(plan.indices)
            return

        tg_3d = _compute_tg(compiled, msl)
        write_indices = _write_indices_from_params(func, gemv_op.kernel._output_params)
        context.append(
            DispatchLaunch(
                kernel=compiled,
                buffers=tuple(buf_arrs),
                grid=grid_3d,
                threadgroup=tg_3d,
                write_indices=write_indices,
                debug_name=func.name,
            ),
            plan.indices,
        )


class ElementChainCompiler:
    name = "elem_chain"
    kind = FusionKind.ELEM_CHAIN

    def compile(self, plan: FusionPlan, context: FusionCompileContext) -> None:
        chain = plan.chain
        if not chain:
            context.fallback(plan.indices)
            return
        try:
            func, buf_arrs, grid_3d, _ = compile_elem_chain(
                [context.ops[idx] for idx in chain],
                list(range(len(chain))),
            )
            func.name = _fused_name(context.ops, chain)
            compiled, msl = _compile_fused(func)
        except FusionUnsupported:
            context.fallback(plan.indices)
            return
        except FusionCompileError as exc:
            warn_key = (func.name, self.name)
            if warn_key not in _warned_fusion_failures:
                _warned_fusion_failures.add(warn_key)
                logger.warning(
                    "fusion_compile_error",
                    kernel=func.name, pass_name=self.name, error=str(exc),
                )
            else:
                logger.debug(
                    "fusion_compile_error_repeat",
                    kernel=func.name, pass_name=self.name,
                )
            context.fallback(plan.indices)
            return

        tg_3d = _compute_tg(compiled, msl)
        write_indices = _fused_write_indices(func)
        context.append(
            DispatchLaunch(
                kernel=compiled,
                buffers=tuple(buf_arrs),
                grid=grid_3d,
                threadgroup=tg_3d,
                write_indices=write_indices,
                debug_name=func.name,
            ),
            plan.indices,
        )


FusionPlanCompiler = (
    IndividualFusionCompiler
    | AnchorFusionCompiler
    | RowPassCompiler
    | MultiRootCompiler
    | ElementChainCompiler
    | RmsNormFoldCompiler
)

FUSION_PLAN_COMPILERS: dict[FusionKind, FusionPlanCompiler] = {
    compiler.kind: compiler
    for compiler in (
        IndividualFusionCompiler(),
        AnchorFusionCompiler(),
        RowPassCompiler(),
        MultiRootCompiler(),
        ElementChainCompiler(),
        RmsNormFoldCompiler(),
    )
}


def _compile_fusion_plans(
    ops: list[LazyOp],
    plans: list[tuple[int, FusionPlan]],
    roots: set[int] | None = None,
    compile_individual: CompileIndividual | None = None,
) -> list[FusedPair]:
    """Compile fusion plans into typed launch records.

    Takes the output of _plan_fusion (analysis) and produces
    (dispatch_launch, op_indices_set) pairs ready for Metal dispatch.
    """
    if compile_individual is None:
        raise RuntimeError("_compile_fusion_plans requires an individual compile callback")
    context = FusionCompileContext(
        ops=ops,
        roots=roots or set(),
        compile_individual=compile_individual,
        dispatch_entries=[],
    )

    for _, plan in plans:
        compiler = FUSION_PLAN_COMPILERS.get(plan.kind)
        if compiler is None:
            logger.warning("fusion_unknown_plan_kind", kind=str(plan.kind))
            context.fallback(plan.indices)
            continue
        compiler.compile(plan, context)

    # Sort by the maximum op index for fused groups (ensures all inputs
    # are dispatched before the fused kernel), minimum for singletons.
    context.dispatch_entries.sort(key=lambda x: max(x[1]) if len(x[1]) > 1 else _builtin_min(x[1]))

    n_fused = sum(1 for _, idxs in context.dispatch_entries if len(idxs) > 1)
    n_individual = sum(1 for _, idxs in context.dispatch_entries if len(idxs) == 1)
    logger.debug(
        "fusion_pass_completed",
        n_in=len(ops),
        n_plans=len(plans),
        n_dispatches_out=len(context.dispatch_entries),
        n_fused=n_fused,
        n_individual=n_individual,
    )
    return context.dispatch_entries


def _write_indices_from_params(func: TileFunction, output_params: set[str]) -> frozenset[int]:
    """Map output param names to buffer indices in the fused function."""
    buf_params = [p for p in func.params if not p.is_constexpr]
    return frozenset(i for i, p in enumerate(buf_params) if p.name in output_params)


def _fused_write_indices(func: TileFunction) -> frozenset[int]:
    """Determine which buffer indices are outputs in a fused TileFunction.

    Scans the IR for Store ops and maps their target param names to
    indices in the non-constexpr param list (which maps 1:1 to buf_arrs).
    """
    store_params: set[str] = set()
    for op in func.ops:
        if isinstance(op, Store) and op.ptr is not None:
            store_params.add(op.ptr.name)

    buf_params = [p for p in func.params if not p.is_constexpr]
    return frozenset(i for i, p in enumerate(buf_params) if p.name in store_params)


# ===================================================================
# Section 10: Metal compilation helpers
# ===================================================================


class FusionCompileError(RuntimeError):
    """Raised when a fused kernel fails to compile (e.g. shmem overflow).

    Caught by _compile_fusion_plans to fall back to individual compilation.
    """

    pass


_FUSED_EMIT_CACHE: dict[str, tuple[CompiledKernel, str]] = {}


def _compile_fused(func: TileFunction) -> tuple[CompiledKernel, str]:
    """Compile a fused TileFunction to a (CompiledKernel, msl_source).

    Process-local cache keyed by `func.fingerprint` (hash of dump_tile_ir)
    short-circuits both `emit_msl_from_tile_ir` and the downstream PSO lookup
    when a structurally identical TileFunction was compiled earlier (training
    graphs emit many fused funcs that fingerprint identically).
    """
    key = func.fingerprint
    cached = _FUSED_EMIT_CACHE.get(key)
    if cached is not None:
        return cached
    msl = emit_msl_from_tile_ir(func)
    device = default_device()
    cache = CacheManager.default()
    hash_key = msl_hash(msl)
    device_key = f"{device.name}|{device.gpu_family}"
    compiled = cache.get_pipeline(hash_key, device_key)
    if compiled is None:
        try:
            compiled = CompiledKernel.from_msl(device, msl, func.name)
        except RuntimeError as e:
            raise FusionCompileError(str(e)) from e
        cache.put_pipeline(hash_key, device_key, compiled)
    result = (compiled, msl)
    _FUSED_EMIT_CACHE[key] = result
    notify_compiled(func.name, dict(func.constexpr_values), None, msl, func)
    return result


def _compute_tg(compiled: CompiledKernel, msl_source: str | None = None) -> tuple[int, int, int]:
    """Compute threadgroup size from MSL source or compiled kernel limits."""
    if msl_source is not None:
        m = re.search(r"NUM_THREADS\s*=\s*(\d+)", msl_source)
        if m:
            return _normalize_grid(int(m.group(1)))
    max_t = compiled.max_total_threads_per_threadgroup
    sw = compiled.thread_execution_width
    tg = _builtin_min(max_t, 1024)
    tg = (tg // sw) * sw
    return _normalize_grid(tg)
