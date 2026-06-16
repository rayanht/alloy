"""KernelFunction — the compiled (or to-be-compiled) kernel object.

Owns the Python function, AST, source code, signature parsing, kernel
classification, tracing, compilation, and tune config resolution.
Dispatch logic (_queue_op) lives in _lazy.py.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
import time
from collections.abc import Callable
from typing import cast

import numpy as np

import alloy._dispatch.buf_utils
from alloy._dispatch.buf_utils import (
    _NON_FUSABLE_ELEM_KERNELS,
    _STRIDE_NATIVE_KERNELS,
    _alloc_aligned,
    _normalize_grid,
)
from alloy._compiler.classify import (
    has_non_elem_constructs,
    has_threadgroup_ops,
    is_tile_kernel,
)
from alloy._compiler.dispatch_spec import DISPATCH_CONTRACT_VERSION
from alloy._compiler.tile_msl import emit_msl_from_tile_ir
from alloy._compiler.trace import (
    _active,
    _rewrite_kernel_source,
    _trace_flow,
    _trace_for_var,
    _trace_if_else,
    _trace_if_enter,
    _trace_if_exit,
    _trace_loop_cond,
    _trace_loop_enter,
    _trace_loop_exit,
    trace_kernel,
)
from alloy._dispatch.cache import CacheManager, KernelKey, msl_hash
from alloy._dispatch.dispatch import _engine
from alloy._dispatch.observe import notify_compiled
from alloy._dispatch.lazy import (
    LazyOp,
    ResolvedConstexprs,
    ResolvedInputs,
    TraceResult,
    _queue_op,
)
from alloy._dispatch.fusion_types import DispatchLaunch
from alloy._dispatch.fusion_compile import _compute_tg
from alloy._runtime import profile
from alloy._runtime.alloy_buffer import AlloyBuffer, _compute_contiguous_strides
from alloy._runtime.convert import to_alloy_buffer
from alloy._runtime.metal import CompiledKernel, default_device
from alloy._runtime.tune_configs import TuneConfig, generate_configs, resolve_config

KernelSource = Callable[..., None]


# --- Kernel launcher — the object returned by ``kernel[grid]`` ---


class KernelLauncher:
    """Holds grid config, ready to dispatch when called."""

    def __init__(self, kernel_fn: KernelFunction, grid: tuple[int, ...]) -> None:
        self.kernel_fn = kernel_fn
        self.grid = grid

    def __call__(
        self, *args: AlloyBuffer | np.ndarray | int | float, **kwargs: int | float | bool
    ) -> AlloyBuffer:
        return cast(AlloyBuffer, _queue_op(self.kernel_fn, self.grid, args, kwargs))


# --- KernelFunction ---


class KernelFunction:
    """Wraps a user-written kernel function.

    Created by the ``@al.kernel`` decorator.  Stores the Python AST,
    supports ``kernel(args)`` (auto-grid) or ``kernel[grid](args)`` launch syntax,
    and JIT-compiles to MSL on first invocation (cached thereafter).
    """

    def __init__(self, fn: KernelSource, *, _source: str | None = None) -> None:
        self.fn = fn
        self.name: str = fn.__name__
        if _source is not None:
            self._source = _source
        elif hasattr(fn, "_alloy_source"):
            self._source = fn._alloy_source
        else:
            self._source = textwrap.dedent(inspect.getsource(fn))
        self._ast: ast.Module = ast.parse(self._source)
        self._init_metadata()

    def _init_metadata(self) -> None:
        """Initialize all derived metadata from _source and _ast."""
        self._param_names: list[str] = []
        self._constexpr_params: set[str] = set()
        self._output_params: set[str] = set()
        self._parse_signature()

        self._is_tile: bool = is_tile_kernel(self._ast)
        self._has_tg_ops: bool = has_threadgroup_ops(self._ast) if not self._is_tile else False
        self._has_non_elem: bool = (
            has_non_elem_constructs(self._ast) if not self._is_tile else False
        )

        self._buf_params: list[str] = [
            p for p in self._param_names if p not in self._constexpr_params
        ]
        self._output_idx: int = next(
            (i for i, p in enumerate(self._buf_params) if p in self._output_params), 1
        )
        self._sig: inspect.Signature = inspect.signature(self.fn)

        # Tuning state (set by @al.tunable decorator)
        self._tune_configs: list[TuneConfig] | None = None
        self._tune_key: list[str] | None = []
        self._tune_tuned_params: set[str] = set()
        self._tune_cache: dict[tuple[object, ...], TuneConfig] = {}
        self._benchmark_options: dict[str, int | list[int]] | None = None

        self._constexpr_defaults: dict[str, int | float | bool] = {}
        for pname in self._param_names:
            if pname not in self._constexpr_params:
                continue
            param = self._sig.parameters.get(pname)
            if param is not None and param.default is not inspect.Parameter.empty:
                self._constexpr_defaults[pname] = param.default

        if self._output_params:
            seen_output = False
            for p in self._buf_params:
                if p in self._output_params:
                    seen_output = True
                elif seen_output:
                    raise TypeError(
                        f"Kernel '{self.name}': al.output param must come after all "
                        f"input buffer params. '{p}' appears after an al.output param."
                    )

    def _parse_signature(self) -> None:
        """Walk the AST to find parameter names and constexpr/output annotations."""
        for node in ast.walk(self._ast):
            if isinstance(node, ast.FunctionDef):
                self._param_names = [arg.arg for arg in node.args.args]
                for arg in node.args.args:
                    if arg.annotation is None:
                        continue
                    ann = arg.annotation
                    if isinstance(ann, ast.Attribute) and ann.attr == "constexpr":
                        self._constexpr_params.add(arg.arg)
                    elif isinstance(ann, ast.Name) and ann.id == "constexpr":
                        self._constexpr_params.add(arg.arg)
                    if isinstance(ann, ast.Attribute) and ann.attr == "output":
                        self._output_params.add(arg.arg)
                    elif isinstance(ann, ast.Name) and ann.id == "output":
                        self._output_params.add(arg.arg)
                break

    # ---- launch syntax ----

    def __call__(
        self, *args: AlloyBuffer | np.ndarray | int | float, **kwargs: int | float | bool
    ) -> AlloyBuffer:
        """Call kernel: ``kernel(x, N=1024)`` or ``kernel(x, out, N=1024)``."""
        if _active():
            if not hasattr(self, "_trace_fn"):
                rewritten = _rewrite_kernel_source(ast.parse(self._source))
                if rewritten is not None:
                    code = compile(rewritten, f"<traced:{self.name}>", "exec")
                    ns = {
                        **self.fn.__globals__,
                        "_trace_loop_enter": _trace_loop_enter,
                        "_trace_for_var": _trace_for_var,
                        "_trace_loop_cond": _trace_loop_cond,
                        "_trace_loop_exit": _trace_loop_exit,
                        "_trace_if_enter": _trace_if_enter,
                        "_trace_if_else": _trace_if_else,
                        "_trace_if_exit": _trace_if_exit,
                        "_trace_flow": _trace_flow,
                    }
                    exec(code, ns)
                    for node in ast.walk(rewritten):
                        if isinstance(node, ast.FunctionDef):
                            self._trace_fn = cast(KernelSource, ns[node.name])
                            break
                    else:
                        self._trace_fn = self.fn
                else:
                    self._trace_fn = self.fn
            return cast(AlloyBuffer, self._trace_fn(*args, **kwargs))
        return cast(AlloyBuffer, _queue_op(self, None, args, kwargs))

    def __getitem__(self, grid: int | tuple[int, ...] | None) -> KernelLauncher:
        """``kernel[grid_x]`` or ``kernel[gx, gy, gz]`` launch syntax."""
        if not isinstance(grid, tuple):
            grid = (grid,)
        return KernelLauncher(self, grid)

    # ---- input resolution ----

    def resolve_inputs(
        self,
        args: tuple[AlloyBuffer | np.ndarray | int | float, ...],
    ) -> ResolvedInputs:
        """Process buffer args: contiguify, extract stride metadata, collect lazy inputs."""
        stride_meta: dict[str, int | tuple[int, ...]] = {}
        buffer_shapes: dict[str, tuple[int, ...]] = {}
        buffer_dtypes: dict[str, str] = {}
        buffer_args: list[tuple[str, AlloyBuffer]] = []
        input_producers: dict[str, LazyOp] = {}
        lazy_inputs: list[AlloyBuffer] = []

        for i, pname in enumerate(self._buf_params):
            if i >= len(args):
                break
            arg = args[i]
            if isinstance(arg, AlloyBuffer):
                # A contig-but-offset view (e.g. `x[1:]`) passes is_contiguous
                # but its _offset is non-zero — without the rebase branch,
                # base_ptr binding loses the offset and the kernel reads from
                # the parent's element 0.
                needs_rebase = not arg.is_contiguous() or arg._offset != 0
                if (
                    needs_rebase
                    and pname not in self._output_params
                    and self.name not in _STRIDE_NATIVE_KERNELS
                ):
                    if (
                        self._is_tile
                        or self._has_tg_ops
                        or self._has_non_elem
                        or self.name in _NON_FUSABLE_ELEM_KERNELS
                    ):
                        arg = arg.contiguous()
                    else:
                        itemsize = arg._dtype.itemsize
                        elem_strides = tuple(s // itemsize for s in arg._strides)
                        # Stride meta drives stride decomposition for non-contig
                        # views. The view's byte offset is NOT baked into MSL —
                        # runtime binding (_execute_plan / _batch_to_v2) applies
                        # it via the Metal buffer offset. Baking it here would
                        # double-apply when the compiled plan also adds the
                        # tensor's storage_offset at bind time.
                        stride_meta[f"_{pname}_shape"] = arg._shape
                        stride_meta[f"_{pname}_strides"] = elem_strides
                        root_buf = AlloyBuffer(
                            arg._parent_handle,
                            arg._offset,
                            (arg.metal_nbytes // itemsize,),
                            _compute_contiguous_strides((arg.metal_nbytes // itemsize,), itemsize),
                            arg._dtype,
                            raw_ptr=arg._raw_ptr,
                            total_nbytes=arg.metal_nbytes,
                        )
                        buffer_args.append((pname, root_buf))
                        buffer_shapes[pname] = (arg.size,)
                        buffer_dtypes[pname] = root_buf._dtype.ir
                        if arg._producer is not None:
                            input_producers[pname] = arg._producer
                        lazy_inputs.append(arg)
                        continue
                buffer_shapes[pname] = arg.shape
                buffer_dtypes[pname] = arg._dtype.ir
                if arg._producer is not None:
                    input_producers[pname] = arg._producer
                lazy_inputs.append(arg)
                buffer_args.append((pname, arg))
            else:
                arg = to_alloy_buffer(arg)
                # External buffers used as al.output would get a one-time Metal copy
                # that diverges from the original memory. Replace with a fresh alloy
                # buffer of the same shape so the kernel output is GPU-native.
                if pname in self._output_params and arg._parent_handle < 0:
                    arg = _alloc_aligned(arg.shape, arg._dtype)
                buffer_shapes[pname] = arg.shape
                buffer_dtypes[pname] = arg._dtype.ir
                buffer_args.append((pname, arg))

        return ResolvedInputs(
            buffer_args, buffer_shapes, buffer_dtypes, input_producers, lazy_inputs, stride_meta
        )

    # ---- constexpr resolution ----

    def resolve_constexprs(
        self,
        kwargs: dict[str, int | float | bool],
        buffer_dtypes: dict[str, str],
        buffer_args: list[tuple[str, AlloyBuffer]],
        lazy_inputs: list[AlloyBuffer],
    ) -> ResolvedConstexprs:
        """Resolve constexpr values from kwargs, tuning, and defaults."""
        values: dict[str, int | float | bool | tuple[int, ...]] = {}
        tune_cfg = (
            self._resolve_tune(kwargs, buffer_dtypes, buffer_args) if self._tune_configs else None
        )
        tuned = tune_cfg.constexprs if tune_cfg else {}
        options: dict[str, int | list[int]] = dict(tune_cfg.options) if tune_cfg else {}
        bench_opts = self._benchmark_options
        if bench_opts:
            options.update(bench_opts)
        for pname in self._constexpr_params:
            if pname in kwargs:
                values[pname] = kwargs[pname]
            elif pname in tuned:
                values[pname] = tuned[pname]
            elif pname in self._constexpr_defaults:
                values[pname] = self._constexpr_defaults[pname]
        unfilled = [p for p in self._param_names if p in self._constexpr_params and p not in values]
        if unfilled:
            dims: list[int] = []
            for pname in self._buf_params:
                shape = buffer_dtypes.get(pname)  # check existence via dtypes
                if shape is not None:
                    for bn, ba in buffer_args:
                        if bn == pname:
                            dims.extend(ba.shape)
                            break
            for i, pname in enumerate(unfilled):
                if i < len(dims):
                    values[pname] = dims[i]
        return ResolvedConstexprs(values, options)

    # ---- trace + grid derivation ----

    def trace_and_plan(
        self,
        constexpr_values: dict[str, int | float | bool | tuple[int, ...]],
        compiler_options: dict[str, int | list[int]],
        buffer_dtypes: dict[str, str],
        buffer_shapes: dict[str, tuple[int, ...]],
        buffer_args: list[tuple[str, AlloyBuffer]],
        grid: tuple[int, ...] | None,
    ) -> TraceResult:
        """Trace kernel, derive grid and output shapes. Uses trace cache."""
        _p = profile._profile_enabled
        _tc_key = (
            DISPATCH_CONTRACT_VERSION,
            self._source,
            tuple(constexpr_values.items()),
            tuple(buffer_dtypes.items()),
            tuple(buffer_shapes.items()),
            grid,
        )
        _tc_hit = _engine.cache.trace_cache.get(_tc_key)
        if _tc_hit is not None:
            func, grid, out_shape = _tc_hit
            return TraceResult(
                func=func, grid=grid, out_shape=out_shape, out_shapes={}, spec=func.dispatch_spec
            )

        if _p:
            _t0 = time.perf_counter_ns()
        func = trace_kernel(
            self.fn,
            self.name,
            constexpr_values,
            buffer_dtypes=buffer_dtypes,
            param_names=self._param_names,
            constexpr_params=self._constexpr_params,
            source=self._source,
            buffer_shapes=buffer_shapes,
            output_params=self._output_params,
        )
        func.options = compiler_options
        trace_ms = (time.perf_counter_ns() - _t0) / 1e6 if _p else 0.0
        if _p:
            _t0 = time.perf_counter_ns()

        spec = func.dispatch_spec
        provided_outputs = {pname for pname, _ in buffer_args if pname in self._output_params}
        need_outputs = self._output_params - provided_outputs if self._output_params else None

        out_shapes: dict[str, tuple[int, ...]] = {}
        out_shape: tuple[int, ...] | None = None
        if spec is not None and (spec.grid_axes or spec.outputs):
            _, grid, out_shapes = spec.evaluate_dispatch(
                constexpr_values,
                buffer_shapes,
                grid_override=grid,
                output_params=need_outputs,
                kernel_name=self.name,
            )
            out_shape = next(iter(out_shapes.values()), None) if out_shapes else None
        else:
            grid = _normalize_grid(grid or (1, 1, 1))

        if out_shape is None and need_outputs:
            out_shape = (grid[0],)
        if (
            need_outputs
            and len(self._output_params) >= 2
            and "M" in constexpr_values
            and "N" in constexpr_values
        ):
            _mn = (int(constexpr_values["M"]), int(constexpr_values["N"]))
            if out_shape is None or (len(out_shape) == 1 and out_shape[0] < _mn[0] * _mn[1]):
                out_shape = _mn

        grid_ms = (time.perf_counter_ns() - _t0) / 1e6 if _p else 0.0
        _engine.cache.trace_cache[_tc_key] = (func, grid, out_shape)
        return TraceResult(
            func=func,
            grid=grid,
            out_shape=out_shape,
            out_shapes=out_shapes,
            spec=spec,
            trace_ms=trace_ms,
            grid_ms=grid_ms,
        )

    # ---- tune resolution ----

    def _resolve_tune(
        self,
        kwargs: dict[str, int | float | bool],
        buffer_dtypes: dict[str, str],
        buffer_args: list[tuple[str, AlloyBuffer]],
    ) -> TuneConfig | None:
        """Look up static config for the current dispatch shapes."""
        key_values: dict[str, int] = {}
        if self._tune_key is not None:
            for k in self._tune_key:
                if k in kwargs:
                    key_values[k] = int(kwargs[k])
        else:
            for k in self._constexpr_params:
                if k not in self._tune_tuned_params and k in kwargs:
                    key_values[k] = int(kwargs[k])
            for pname, arg in buffer_args:
                for di, dim in enumerate(arg.shape):
                    key_values[f"_{pname}_dim{di}"] = int(dim)

        # In-memory cache (avoids repeated dict lookups for the same shape)
        cache_key = tuple(sorted(key_values.items()))
        cached = self._tune_cache.get(cache_key)
        if cached is not None:
            return cached

        cfg = resolve_config(self.name, key_values)
        # Wrap in TuneConfig for resolve_constexprs
        result = TuneConfig(constexprs=dict(cfg.constexprs), options=dict(cfg.options))
        self._tune_cache[cache_key] = result
        return result

    # ---- compilation (called during flush by fusion engine) ----

    def _compile_op(self, op: LazyOp) -> DispatchLaunch:
        """Compile a LazyOp into one typed Metal launch."""
        _p = profile._profile_enabled
        rec: profile.DispatchRecord | None = None
        if _p:
            rec = profile.DispatchRecord(name=self.name)
            rec.phases[profile.QUEUE] = _engine.op_profile.get(id(op), {}).get("queue_ms", 0.0)
            rec.phases[profile.TRACE] = _engine.op_profile.get(id(op), {}).get("trace_ms", 0.0)
            rec.phases[profile.GRID] = _engine.op_profile.get(id(op), {}).get("grid_ms", 0.0)

        device = default_device()
        constexpr_values = dict(op.constexpr_values)
        buffer_args = op.buffer_args
        buffer_dtypes = dict(op.buffer_dtypes)
        grid = op.grid

        if op.func is not None:
            for param in op.func.params:
                if not param.is_constexpr and param.name in buffer_dtypes:
                    param.dtype = buffer_dtypes[param.name]

        buffer_shapes: dict[str, tuple[int, ...]] = {pn: a.shape for pn, a in buffer_args}

        if rec:
            _t = time.perf_counter_ns()
        cache_key = (
            op._cache_key
            if op._cache_key is not None
            else (
                self._source,
                op.func.fingerprint,
                tuple(constexpr_values.items()),
                tuple(buffer_dtypes.items()),
                tuple(buffer_shapes.items()),
                grid,
            )
        )
        if rec:
            rec.phases[profile.CACHE_KEY] = (time.perf_counter_ns() - _t) / 1e6

        cached = _engine.cache.opaque_cache.get(cache_key)
        if cached is not None:
            compiled, grid_3d, tg_3d = cached
            buffers = tuple(a for _, a in buffer_args)
            if rec:
                rec.cache_level = "dispatch"
                rec.grid = grid_3d
                rec.threadgroup = tg_3d
                profile.get_accumulator().record_cache_hit("dispatch")
                rec._total_t0 = _engine.op_profile.get(id(op), {}).get("total_t0", 0)
                profile.get_accumulator().records.append(rec)
            return DispatchLaunch(
                kernel=compiled,
                buffers=buffers,
                grid=grid_3d,
                threadgroup=tg_3d,
                write_indices=frozenset(
                    i for i, (pn, _) in enumerate(buffer_args) if pn in self._output_params
                ),
                debug_name=self.name,
            )

        if rec:
            _t = time.perf_counter_ns()
        buffers = tuple(a for _, a in buffer_args)
        if rec:
            rec.phases[profile.BUF_PREP] = (time.perf_counter_ns() - _t) / 1e6

        if self.name == "dot_transpose_rhs":
            a_shape = buffer_shapes.get("A")
            bt_shape = buffer_shapes.get("B_T")
            if a_shape and len(a_shape) == 2:
                op.func.constexpr_values["M"] = a_shape[0]
                op.func.constexpr_values["K"] = a_shape[1]
            if bt_shape and len(bt_shape) == 2:
                op.func.constexpr_values["N"] = bt_shape[0]

        cache = CacheManager.default()
        kernel_key = KernelKey(
            self._source,
            {
                **constexpr_values,
                "_dtypes": buffer_dtypes,
                "_shapes": buffer_shapes,
                "_tile_ir": op.func.fingerprint,
            },
        )

        msl_source = cache.get_msl(kernel_key)
        if msl_source is None:
            op.func.buffer_dtypes = buffer_dtypes
            if rec:
                _t = time.perf_counter_ns()
            msl_source = emit_msl_from_tile_ir(
                op.func, debug=alloy._dispatch.buf_utils._debug_mode
            )
            if rec:
                rec.phases[profile.CODEGEN] = (time.perf_counter_ns() - _t) / 1e6
                rec.cache_level = "miss"
                profile.get_accumulator().record_cache_miss()
            cache.put_msl(kernel_key, msl_source)
        elif rec:
            rec.cache_level = "msl"
            profile.get_accumulator().record_cache_hit("msl")

        notify_compiled(self.name, dict(constexpr_values), buffer_shapes, msl_source, op.func)

        hash_key = msl_hash(msl_source)
        device_key = f"{device.name}|{device.gpu_family}"
        compiled = cache.get_pipeline(hash_key, device_key)
        if compiled is None:
            if rec:
                _t = time.perf_counter_ns()
            compiled = CompiledKernel.from_msl(device, msl_source, self.name)
            if rec:
                rec.phases[profile.COMPILE] = (time.perf_counter_ns() - _t) / 1e6
            cache.put_pipeline(hash_key, device_key, compiled)
        elif rec and rec.cache_level != "miss":
            rec.cache_level = "pipeline"
            profile.get_accumulator().record_cache_hit("pipeline")

        tg_3d = _compute_tg(compiled, msl_source)
        grid_3d = grid
        _engine.cache.opaque_cache[cache_key] = (compiled, grid_3d, tg_3d)

        if rec:
            rec.grid = grid_3d
            rec.threadgroup = tg_3d
            rec._total_t0 = _engine.op_profile.get(id(op), {}).get("total_t0", 0)
            profile.get_accumulator().records.append(rec)

        return DispatchLaunch(
            kernel=compiled,
            buffers=buffers,
            grid=grid_3d,
            threadgroup=tg_3d,
            write_indices=frozenset(
                i for i, (pn, _) in enumerate(buffer_args) if pn in self._output_params
            ),
            debug_name=self.name,
        )

    # ---- MSL compilation ----

    def _compile(
        self,
        constexpr_values: dict[str, int | float | bool | tuple[int, ...]],
        buffer_dtypes: dict[str, str] | None = None,
        buffer_shapes: dict[str, tuple[int, ...]] | None = None,
        debug: bool = False,
    ) -> str:
        """Compile kernel to MSL via tracing."""
        func = trace_kernel(
            self.fn,
            self.name,
            constexpr_values,
            buffer_dtypes=buffer_dtypes,
            param_names=self._param_names,
            constexpr_params=self._constexpr_params,
            source=self._source,
            buffer_shapes=buffer_shapes,
            output_params=self._output_params,
        )
        return emit_msl_from_tile_ir(func, debug=debug)

    def compile_to_msl(
        self, **kwargs: int | float | bool | str | tuple[int, ...] | dict[str, str]
    ) -> str:
        """Compile the kernel to MSL source."""
        constexpr_values: dict[str, int | float | bool | tuple[int, ...]] = dict(
            self._constexpr_defaults
        )
        buffer_shapes: dict[str, tuple[int, ...]] = {}
        buffer_dtypes: dict[str, str] = {}
        buf_params = set(self._param_names) - self._constexpr_params - self._output_params
        for k, v in kwargs.items():
            if k in buf_params and isinstance(v, tuple):
                buffer_shapes[k] = v
            elif k in self._constexpr_params:
                constexpr_values[k] = v
            elif k == "buffer_dtypes":
                buffer_dtypes = v
            else:
                constexpr_values[k] = v
        return self._compile(
            constexpr_values,
            buffer_dtypes=buffer_dtypes or None,
            buffer_shapes=buffer_shapes or None,
            debug=alloy._dispatch.buf_utils._debug_mode,
        )


def kernel(fn: KernelSource, *, _source: str | None = None) -> KernelFunction:
    """Decorator to mark a function as an Alloy GPU kernel."""
    return KernelFunction(fn, _source=_source)


def tunable(
    *,
    key: list[str] | None = None,
    threadgroup_size: list[tuple[int, ...]] | None = None,
    options: dict[str, list[int]] | None = None,
    **param_ranges: list[int],
) -> Callable[[KernelFunction], KernelFunction]:
    """Decorator to declare tuning search space for a kernel."""
    tg_sizes = [_normalize_grid(t) for t in threadgroup_size] if threadgroup_size else None
    configs = generate_configs(param_ranges, tg_sizes, option_ranges=options)
    tuned_params = set(param_ranges.keys())

    def decorator(kf: KernelFunction) -> KernelFunction:
        if not isinstance(kf, KernelFunction):
            raise TypeError(
                "@al.tunable must be applied BEFORE @al.kernel:\n"
                "    @al.tunable(...)\n"
                "    @al.kernel\n"
                "    def my_kernel(...):"
            )
        kf._tune_configs = configs
        kf._tune_key = key
        kf._tune_tuned_params = tuned_params
        return kf

    return decorator
