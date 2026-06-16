"""Offline kernel tuner — produces static JSON config files.

Usage:
    import alloy as al
    al.tune(model, sample_inputs)                    # tune all kernels for a model
    al.tune(al.std.dot_transpose_rhs, M=16, K=2048, N=5632)  # tune one kernel
    al.tune_report()                                 # show what's tuned
"""

from __future__ import annotations

import json
import importlib
import os
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from alloy._compiler.dtypes import float32, from_name
from alloy._dispatch import kernel
from alloy._dispatch.buf_utils import (
    _alloy_buf_map,
    _alloc_aligned,
    is_phantom_buffer,
    set_record_only,
)
from alloy._dispatch.dispatch import _engine
from alloy._dispatch.kernel import KernelFunction
from alloy.log import get_logger
from alloy._runtime import _metal_ext
from alloy._runtime.alloy_buffer import AlloyBuffer, materialize_many
from alloy._runtime.device import detect_device
from alloy._runtime.tune_configs import (
    _CONSERVATIVE_DEFAULTS,
    _PIPELINE_VERSION,
    _STATIC_CONFIGS,
    TuneConfig,
    _load_configs,
    user_config_file,
)

logger = get_logger("alloy.tune")

# alloy_torch mirrors the training-mode flag for its compile pipeline; it
# registers these hooks when IT imports (alloy_torch.mode). alloy must not
# import alloy_torch itself — that drags torch + transformers (~2.3s) into
# every consumer of the base package, including each CLI invocation.
_training_mode_reader: Callable[[], bool] | None = None
_training_mode_writer: Callable[[bool], None] | None = None
_fallback_training_mode: bool = False


def register_training_mode_hooks(
    reader: Callable[[], bool], writer: Callable[[bool], None]
) -> None:
    """Called by alloy_torch.mode at import. Adopts any mode set before
    alloy_torch loaded, so the two flags stay in lockstep like they did
    when this module imported alloy_torch eagerly."""
    global _training_mode_reader, _training_mode_writer
    _training_mode_reader = reader
    _training_mode_writer = writer
    if _fallback_training_mode:
        writer(True)


def _training_mode_enabled() -> bool:
    if _training_mode_reader is None:
        return _fallback_training_mode
    return _training_mode_reader()


def _set_training_mode(mode: bool) -> None:
    global _fallback_training_mode
    _fallback_training_mode = mode
    if _training_mode_writer is not None:
        _training_mode_writer(mode)
    _metal_ext.set_training_mode(mode)
    _metal_ext._training_mode_flag = mode


# ---------------------------------------------------------------------------
# Shape extraction
# ---------------------------------------------------------------------------


@dataclass
class CapturedKernel:
    """Captured kernel dispatch: name, shape key, kwargs, and actual buffer args."""

    name: str
    key_values: dict[str, int]
    kwargs: dict[str, Any]
    buffer_args: list[tuple[str, Any]]  # (param_name, AlloyBuffer)
    buffer_snapshots: dict[str, np.ndarray]
    kernel: Any  # KernelFunction reference
    # Single-kernel `al.tune(kernel, ..., buffer_dtypes={...})` callers pass
    # explicit per-buffer dtypes here so `_make_test_buffers` synthesizes
    # test data at the production dtype mix (e.g. Q=f32, K=V=f16 cold
    # prefill against an fp16 KV cache). When None, model-tuning's path
    # populates buffer dtypes from the captured AlloyBuffers in
    # `buffer_args` instead.
    buffer_dtypes_override: dict[str, Any] | None = None
    # Explicit launch grid, when the dispatch used `kernel[grid](...)` (e.g.
    # split-K attention's SPLITS axis can't be derived from inputs). None means
    # the kernel derives its own grid. Captured so `dispatch_captured` can replay
    # grid-requiring kernels faithfully.
    grid: tuple[int, ...] | None = None
    # FULL ordered buffer list (inputs AND outputs) as actually dispatched —
    # `buffer_args` above is filtered to inputs for the tuner's snapshot path, but
    # a faithful replay (`dispatch_captured`) must pass every positional buffer,
    # including pre-allocated outputs the kernel writes (e.g. split-K's pO/pL).
    replay_buffer_args: list[tuple[str, Any]] = field(default_factory=list)


def _extract_shapes(
    model, inputs, training: bool = False, record_only: bool = False
) -> list[CapturedKernel]:
    """Compile model, run one forward (+ optional backward) pass, collect kernel dispatch info.

    `record_only` runs the extraction forward without executing the GPU or
    allocating real intermediate storage (phantom buffers) — shapes are captured
    from the handler-path resolve regardless. Use for LARGE-M forwards (one-shot
    prefill at M_MAX) that would otherwise hold O(M_MAX × layers) of activations
    and OOM. The per-kernel benchmark afterward synthesizes its own bounded test
    buffers, so it needs only the shapes (+ small real-INPUT snapshots, which stay
    real in record-only). At large M the big activations exceed the 16M-element
    snapshot cap anyway (synthesized random either way), so nothing is lost; at
    small M leave it off so realistic Q/K/V/Mask snapshots are captured."""
    collected: list[CapturedKernel] = []
    seen: set[tuple[str, tuple]] = set()

    # The explicit launch grid of the dispatch currently being resolved. Set by
    # the `_queue_op` wrapper just before it drives resolve → `_resolve_tune`
    # (same synchronous call), so `_intercepting_resolve` reads the grid that
    # belongs to this exact dispatch. None for auto-grid kernels.
    grid_holder: dict[str, tuple | None] = {"grid": None}

    # Intercept `resolve_constexprs`, NOT `_resolve_tune`: the tune resolve only
    # runs for kernels with registered tune configs, so hooking it makes every
    # UNTUNED kernel (the MoE sort chain, casts, strided copies) invisible to
    # `alloy microbench` / `alloy profile --capture`. resolve_constexprs runs
    # for every dispatch and receives the same (kwargs, buffer_args) the tune
    # hook saw.
    orig_resolve = KernelFunction.resolve_constexprs

    def _intercepting_resolve(self, kwargs, buffer_dtypes, buffer_args, lazy_inputs):
        key_values: dict[str, int] = {}
        # Truthiness (not `is not None`) on purpose: an untuned kernel carries
        # the default EMPTY tune key, which must fall through to the
        # constexpr+dims key so its shape variants stay distinct.
        if self._tune_key:
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

        dedup_key = (self.name, tuple(sorted(key_values.items())))
        if dedup_key not in seen:
            seen.add(dedup_key)
            # Snapshot the input buffers NOW, before later kernels mutate or
            # the dispatcher recycles them. Capturing the AlloyBuffer
            # references and reading them after the forward returns NaN —
            # the storage has been overwritten or freed by then.
            input_bufs = [
                (pn, arg)
                for pn, arg in buffer_args
                if pn not in self._output_params
            ]
            snaps = _snapshot_small_buffers(input_bufs)
            collected.append(
                CapturedKernel(
                    name=self.name,
                    key_values=key_values,
                    kwargs=dict(kwargs),
                    buffer_args=input_bufs,
                    buffer_snapshots=snaps,
                    kernel=self,
                    grid=grid_holder["grid"],
                    replay_buffer_args=list(buffer_args),
                )
            )

        return orig_resolve(self, kwargs, buffer_dtypes, buffer_args, lazy_inputs)

    KernelFunction.resolve_constexprs = _intercepting_resolve

    # Capture each dispatch's explicit launch grid. `_queue_op` is imported into
    # the `kernel` module namespace (both `kernel(...)` and `kernel[grid](...)`
    # call it there), so patch it there. It runs resolve → `_resolve_tune`
    # synchronously, so stashing the grid first makes it visible to the
    # interceptor for this dispatch.
    orig_queue_op = kernel._queue_op

    def _grid_capturing_queue_op(kernel, grid, args, kwargs):
        grid_holder["grid"] = grid
        return orig_queue_op(kernel, grid, args, kwargs)

    kernel._queue_op = _grid_capturing_queue_op
    _prev_training = False
    try:
        # Model-level tuning compiles a torch model via the alloy backend,
        # so it needs both torch and alloy_torch at call time. Keep both
        # as call-site imports so plain kernel tuning and the rest of
        # alloy-metal stay torch-free.
        importlib.import_module("alloy_torch")
        import torch  # scoped: optional dep, only needed for model-level tuning

        if training:
            _prev_training = _training_mode_enabled()
            _set_training_mode(True)

        # dynamic=False so AOT autograd emits the same static-shape graph
        # the production runtime hits (AlloyGenerator and embedder both pin
        # shapes per bucket). Under default dynamic shapes, the decomposed
        # attention bmm chain takes a different form that the eager-to-sdpa
        # rewrite doesn't match — tuning would then capture the bmm
        # kernels instead of the fused attention kernel production actually
        # dispatches.
        compiled = torch.compile(model, backend="alloy", dynamic=False)

        def _call():
            if isinstance(inputs, Mapping):
                return compiled(**inputs)
            if isinstance(inputs, (list, tuple)):
                return compiled(*inputs)
            return compiled(inputs)

        if training:
            out = _call()
            def _sum_scalar(v):
                if isinstance(v, torch.Tensor):
                    return v.sum() if v.dtype.is_floating_point else None
                if hasattr(v, "to_tuple"):
                    return _sum_scalar(v.to_tuple())
                if isinstance(v, Mapping):
                    return _sum_scalar(tuple(v.values()))
                if isinstance(v, (list, tuple)):
                    parts = [_sum_scalar(x) for x in v if x is not None]
                    parts = [p for p in parts if p is not None]
                    return sum(parts) if parts else None
                return None
            loss = _sum_scalar(out)
            if loss is None:
                raise RuntimeError("training=True but no float tensor in output")
            loss.backward()
        else:
            # inference_mode (not no_grad) so AOT decomposes attention the
            # same way production does — under no_grad the view-clone-expand
            # chain leaves AOT as bmm-of-views, which our eager-to-sdpa
            # rewrite doesn't see, and tune captures bmm/dot kernels
            # instead of the fused attention kernel.
            with torch.inference_mode():
                # record_only: phantom intermediates + no GPU dispatch, so a
                # large-M forward records every kernel's shape without holding all
                # activations (the OOM otherwise). Reset before benchmarking, which
                # must run real dispatches to time configs.
                set_record_only(record_only)
                try:
                    _call()
                finally:
                    set_record_only(False)
    finally:
        KernelFunction.resolve_constexprs = orig_resolve
        kernel._queue_op = orig_queue_op
        if training:
            _set_training_mode(_prev_training)

    # Snapshots were captured inline above (at dispatch time) — they hold
    # the live data the kernel actually saw. Reading after the forward can
    # hit freed/recycled buffers and produce NaN, which disables the tuner's
    # correctness comparison (NaN != NaN at every threshold).
    return collected


def capture_kernel_for_replay(
    net,
    inputs,
    record_only: bool = False,
) -> list[CapturedKernel]:
    """Capture every kernel a single `net(**inputs)` forward dispatches, ready to
    replay. Thin public wrapper over `_extract_shapes` for the perf CLIs (`alloy
    microbench`, `alloy profile --capture`): they need the production-resolved
    constexprs + real input buffers of a specific kernel, NOT a full tune. The
    caller filters the result by name and replays via `benchmark_config` /
    `dispatch_captured`.

    `record_only=True` captures without executing the GPU or allocating real
    intermediate storage — required for large-M forwards (a real run-0 of a
    4096-chunk prefill on a 35B MoE holds 100+ GB of activations). Captured
    intermediate buffers are then phantom; `dispatch_captured` materializes
    them (snapshot contents where one was taken, zeros otherwise) on first
    replay. Zeros are timing-faithful for structurally-bounded kernels but NOT
    for kernels whose trip count comes from a captured intermediate's CONTENTS
    (e.g. a TOTAL_ROWS-style runtime bound reading 0 → empty dispatch) — poke
    real values into the materialized buffer before timing those.
    """
    return _extract_shapes(net, inputs, record_only=record_only)


def dispatch_captured(captured: CapturedKernel) -> tuple:
    """Replay one captured kernel dispatch with its real buffers + grid. Returns
    the output buffer tuple, materialized. Mirrors `benchmark_config`'s
    `_dispatch` minus the timing/correctness bookkeeping — the caller owns the
    timing loop so it can keep the GPU clock pinned across repeated replays.

    Buffers captured under record-only are phantom (no Metal storage) — they
    materialize as real zero-filled allocations on first replay, cached back
    onto the record so every later replay reuses the same storage. Zero
    contents are timing-faithful for structurally-bounded kernels; see
    `capture_kernel_for_replay` for the content-bounded caveat.
    """
    kwargs = dict(captured.kwargs)
    # Replay every positional buffer the dispatch used (inputs + pre-allocated
    # outputs), in order — falling back to the input-only list for records
    # captured before replay_buffer_args existed.
    buf_args = captured.replay_buffer_args or captured.buffer_args
    if any(isinstance(a, AlloyBuffer) and is_phantom_buffer(a) for _, a in buf_args):
        buf_args = [
            (pn, _alloc_aligned(a.shape, a.dtype))
            if isinstance(a, AlloyBuffer) and is_phantom_buffer(a)
            else (pn, a)
            for pn, a in buf_args
        ]
        captured.replay_buffer_args = buf_args
    lazy = []
    for pname, arr in buf_args:
        if isinstance(arr, AlloyBuffer):
            lazy.append(arr)
        else:
            lazy.append(AlloyBuffer(arr=arr, shape=arr.shape, dtype=arr.dtype))
    # Grid-requiring kernels (split-K attention's SPLITS axis) replay through the
    # captured explicit grid; auto-grid kernels derive their own.
    launcher = captured.kernel[captured.grid] if captured.grid is not None else captured.kernel
    result = launcher(*lazy, **kwargs)
    out = (result,) if isinstance(result, AlloyBuffer) else result
    if isinstance(out, tuple):
        materialize_many(out)
    return out


def _snapshot_small_buffers(input_bufs: list[tuple[str, Any]]) -> dict[str, np.ndarray]:
    """Capture trace-time buffer contents so the tuner can correctness-check
    against realistic data, not random noise.

    A noise fallback (`np.random.randn() * 0.1` for Q/K/V/Mask) is unsafe for
    attention: random masks don't respect causality, so configs that produce
    different numerical artifacts on that noise agree within the f32 max-diff
    threshold yet diverge on the real causal mask.

    Cap at 16 M elements (~64 MB at f32) so we capture Q/K/V/Mask for all
    reasonable shapes while skipping embedding/lm_head weights.

    `arg.numpy` doesn't gpu-sync (observation 10164), so for lazy buffers
    (intermediate activations like Q/K/V projections) it can return the
    pre-materialization underlying storage — zeros for freshly-allocated
    intermediates. Force materialization via `materialize_many` and sync,
    so the snapshot reflects what the kernel actually saw.
    """
    # Skip record-only phantoms: their fake ptr can't be read (segfault, not a
    # catchable exception). In record-only extraction the kept snapshots are the
    # real INPUT buffers; phantom intermediates get synthesized at benchmark time.
    snapshots: dict[str, np.ndarray] = {}
    materializable = [
        arg
        for _, arg in input_bufs
        if isinstance(arg, AlloyBuffer)
        and arg.size <= 16 * 1024 * 1024
        and not is_phantom_buffer(arg)
    ]
    if materializable:
        try:
            materialize_many(materializable)
        except Exception:
            pass
        try:
            _metal_ext.gpu_sync()
        except Exception:
            pass
    for pname, arg in input_bufs:
        if (
            not isinstance(arg, AlloyBuffer)
            or arg.size > 16 * 1024 * 1024
            or is_phantom_buffer(arg)
        ):
            continue
        try:
            snapshots[pname] = np.asarray(arg.numpy).copy()
        except Exception:
            continue
    return snapshots


# ---------------------------------------------------------------------------
# Config pruning
# ---------------------------------------------------------------------------


def prune_configs(
    kernel_name: str,
    configs: list[TuneConfig],
    key_values: dict[str, int],
) -> list[TuneConfig]:
    """Filter configs before benchmarking using shape-conditional rules.

    Static rules from aggregate tuning data live here — values that never
    win at certain shapes get dropped before we waste GPU time on them.
    """
    out = list(configs)

    # ── dot_transpose_rhs: matvec=1 only ever wins at small M.
    # Across 153 tuned shapes the matvec specialization wins 90% at M=1
    # and 18% at M∈[2,16], then 0% at M≥17. Skip _matvec=1 entirely above
    # M=32 — saves roughly half the search space at the lm_head / qkv /
    # ffn-projection shapes that dominate prefill+training.
    if kernel_name in ("dot_transpose_rhs", "dot_transpose_rhs_silu"):
        m = key_values.get("_A_dim0")
        if m is not None and m > 32:
            out = [c for c in out if c.constexprs.get("_matvec", 0) != 1]

    return out


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------


def _validate_config(config: TuneConfig, key_values: dict[str, int]) -> bool:
    """Quick pre-flight checks before dispatching a config."""
    ce = config.constexprs
    bm = ce.get("BLOCK_M", 32)
    bn = ce.get("BLOCK_N", 32)
    bk = ce.get("BLOCK_K", 32)

    # Shared memory budget: (BLOCK_M + BLOCK_N) * BLOCK_K * 4 bytes (f32 worst case)
    shmem = (bm * (bk + 8) + bn * (bk + 8)) * 4
    if shmem > 32768:
        return False

    # Matvec path needs K >= 4 for vectorized loads
    if ce.get("_matvec", 0) and bk < 4:
        return False

    return True


_TUNE_DEBUG = bool(os.environ.get("ALLOY_TUNE_DEBUG"))


def _cfg_label(config: TuneConfig) -> str:
    """One-line constexpr/option summary for ALLOY_TUNE_DEBUG output."""
    parts = [f"{k}={v}" for k, v in config.constexprs.items()]
    if config.options:
        parts.append(f"opts={config.options}")
    return " ".join(parts) or "(default)"


def _combine_splitk_partials(p_o: np.ndarray, p_lse: np.ndarray) -> np.ndarray:
    """lse-weighted softmax reduction of split-K attention partials → final output.

    `p_o` (SPLITS, BH, N, D) is each split's NORMALIZED softmax output, `p_lse`
    (SPLITS, BH, N) its log-sum-exp. Mirrors `attention_combine_splits`:
    `out = Σ_s softmax_s(lse) · o_s`. Empty splits carry a huge-negative sentinel
    lse → ~0 weight, so they drop out automatically (no sentinel masking needed).

    This is what the tuner must correctness-check, NOT the raw partials: which
    split a boundary KV position lands in shifts with BLOCK_N, so each split's
    normalized `o_s` and `lse_s` are config-dependent intermediates — only this
    combined output (what `attention_combine_splits` feeds downstream) is
    invariant. Verified against f32 truth: the combined output stays ~1e-5 from
    ground truth even where a single split's partial_O differs by 5e-2 across
    BLOCK_N (gemma4:e4b head_dim-512), so comparing raw partials falsely rejects
    correct large-BLOCK_N configs.
    """
    m = p_lse.max(axis=0)
    w = np.exp(p_lse - m[None, ...])
    denom = w.sum(axis=0)
    out = (w[..., None] * p_o).sum(axis=0)
    return out / np.maximum(denom, 1e-30)[..., None]


def benchmark_config(
    kernel,
    config: TuneConfig,
    input_arrays: list[tuple[str, Any]],
    reference_output: Any = None,
    n_warmup: int = 5,
    n_runs: int = 15,
    correctness_threshold: float | None = None,
    extra_kwargs: dict[str, Any] | None = None,
) -> float | None:
    """Benchmark a single config. Returns trimmed-mean GPU time in µs, or None if failed.

    If reference_output is provided, validates correctness by comparing against it.
    """
    if not _validate_config(config, {}):
        if _TUNE_DEBUG:
            ce = config.constexprs
            bm, bn, bk = ce.get("BLOCK_M", 32), ce.get("BLOCK_N", 32), ce.get("BLOCK_K", 32)
            shmem = (bm * (bk + 8) + bn * (bk + 8)) * 4
            print(f"    [skip:preflight] {_cfg_label(config)}  shmem_est={shmem}B(>32768) or matvec_bk<4", flush=True)
        return None

    # Intercept dispatch to capture GPU timestamps
    gpu_us_list: list[float] = []
    orig_dispatch = _metal_ext.dispatch

    def capturing_dispatch(groups):
        r = orig_dispatch(groups)
        if isinstance(r, dict) and "gpu" in r:
            gpu_us_list.append(r["gpu"] * 1000)
        return r

    _metal_ext.dispatch = capturing_dispatch

    try:
        # Merge captured kernel constexprs (SEQ_LEN, HEAD_DIM, KV_GROUP, HIGH_PRECISION,
        # strides, etc.) under config.constexprs (BLOCK_M, BLOCK_N, ...). Without this,
        # the kernel runs with default constexprs (SEQ_LEN=1, HEAD_DIM=1, ...), which
        # measures a degenerate no-op workload and makes the tuner's "best" config
        # meaningless for the real shape.
        kwargs = dict(extra_kwargs) if extra_kwargs else {}
        kwargs.update(config.constexprs)
        kernel._benchmark_options = config.options

        def _make_inputs():
            lazy = []
            for pname, arr in input_arrays:
                if isinstance(arr, AlloyBuffer):
                    lazy.append(arr)
                else:
                    lazy.append(AlloyBuffer(arr=arr, shape=arr.shape, dtype=arr.dtype))
            return lazy

        def _dispatch():
            lazy = _make_inputs()
            result = kernel(*lazy, **kwargs)
            if isinstance(result, tuple):
                materialize_many(result)
            elif isinstance(result, AlloyBuffer):
                materialize_many((result,))
            return result

        for _ in range(n_warmup):
            gpu_us_list.clear()
            try:
                _dispatch()
            except Exception as e:
                if _TUNE_DEBUG:
                    msg = (str(e).strip().splitlines() or [type(e).__name__])[0]
                    print(f"    [skip:dispatch] {_cfg_label(config)}  {msg[:180]}", flush=True)
                return None

        gpu_us_list.clear()
        try:
            result = _dispatch()
        except Exception as e:
            if _TUNE_DEBUG:
                msg = (str(e).strip().splitlines() or [type(e).__name__])[0]
                print(f"    [skip:dispatch] {_cfg_label(config)}  {msg[:180]}", flush=True)
            return None

        out_bufs = (result,) if isinstance(result, AlloyBuffer) else result
        if isinstance(out_bufs, tuple):
            for bi, buf in enumerate(out_bufs):
                if isinstance(buf, AlloyBuffer) and buf.size > 0:
                    arr = buf.numpy
                    if arr is not None and np.any(np.isnan(arr)):
                        if _TUNE_DEBUG:
                            stats = []
                            for _pn, _a in input_arrays:
                                _av = np.asarray(_a.numpy) if isinstance(_a, AlloyBuffer) else np.asarray(_a)
                                if _av is not None and _av.size:
                                    _nan = "/NaN" if np.isnan(_av).any() else ""
                                    stats.append(f"{_pn}[{float(np.nanmin(_av)):.1e},{float(np.nanmax(_av)):.1e}]{_nan}")
                            print(f"    [skip:NaN] {_cfg_label(config)}  out[{bi}] {float(np.isnan(arr).mean()):.1%} NaN | in: {' '.join(stats)}", flush=True)
                        return None

        # Full correctness comparison if reference provided. The reference is a
        # numpy snapshot taken immediately after `_compute_reference` ran (see
        # snapshot logic there) — comparing AlloyBuffer-to-AlloyBuffer would
        # be a no-op, since `_dispatch` overwrites the same output buffer the
        # reference run wrote into.
        if reference_output is not None and correctness_threshold is not None:
            if isinstance(reference_output, tuple):
                ref_arrays = reference_output
            elif isinstance(reference_output, np.ndarray):
                ref_arrays = (reference_output,)
            elif isinstance(reference_output, AlloyBuffer):
                ref_arrays = (np.asarray(reference_output.numpy).copy(),)
            else:
                ref_arrays = ()
            if isinstance(out_bufs, tuple) and ref_arrays:
                # Split-K attention emits (partial_O[S,BH,N,D], partial_lse[S,BH,N]).
                # Validate the lse-weighted COMBINED output, not the raw per-split
                # partials (which legitimately shift across BLOCK_N — see
                # `_combine_splitk_partials`). Comparing partials directly falsely
                # rejected correct large-BLOCK_N configs (gemma4:e4b head_dim-512),
                # pinning it to a ~7× slower BLOCK_N=8.
                go = np.asarray(out_bufs[0].numpy).astype(np.float32) if (out_bufs and isinstance(out_bufs[0], AlloyBuffer)) else None
                is_splitk = (
                    len(out_bufs) == 2 and go is not None and go.ndim == 4
                    and len(ref_arrays) == 2 and isinstance(ref_arrays[0], np.ndarray)
                    and isinstance(out_bufs[1], AlloyBuffer) and isinstance(ref_arrays[1], np.ndarray)
                    and ref_arrays[0].shape == go.shape
                    and out_bufs[1].numpy.ndim == 3 and out_bufs[1].numpy.shape == go.shape[:3]
                    and ref_arrays[1].shape == go.shape[:3]
                )
                if is_splitk:
                    g = _combine_splitk_partials(go, np.asarray(out_bufs[1].numpy).astype(np.float32))
                    r = _combine_splitk_partials(ref_arrays[0].astype(np.float32), ref_arrays[1].astype(np.float32))
                    # A non-finite where the other is finite is a real failure → inf → reject.
                    cdiff = np.where(np.isfinite(g) & np.isfinite(r), np.abs(g - r), np.inf)
                    cmax = float(cdiff.max()) if cdiff.size else 0.0
                    cbad = float((cdiff > correctness_threshold).mean()) if cdiff.size else 0.0
                    # Reject only WIDESPREAD disagreement. Split-K's f16 online softmax
                    # and the borderline empty-split classification (a split covering a
                    # causally-marginal KV position flips empty↔non-empty on an fp-edge
                    # call) are non-deterministic at a SPARSE set of elements — the SAME
                    # config differs from itself run-to-run at ~0.5% of elements, and the
                    # streamed-K path is bit-accurate on synthetic data (matches f32 to
                    # 1e-4 at every scale/offset). Comparing raw per-split partials at
                    # max-abs would falsely reject correct large-BLOCK_N configs: those
                    # are config-dependent intermediates the combine reconciles. A
                    # genuinely wrong config diverges everywhere (or NaNs), not at 0.5%
                    # of elements.
                    if cbad > 0.02 or cmax > 0.25:
                        if _TUNE_DEBUG:
                            fd = np.where(np.isfinite(cdiff), cdiff, 0.0)
                            fi = int(fd.argmax())
                            idx = tuple(int(x) for x in np.unravel_index(fi, g.shape))
                            print(
                                f"    [skip:precision] {_cfg_label(config)}  combined "
                                f"frac>{correctness_threshold:.0e}={cbad:.2%} max={cmax:.3e} @ {idx} "
                                f"(got={float(g.flat[fi]):.3e} ref={float(r.flat[fi]):.3e})",
                                flush=True,
                            )
                        return None
                    if _TUNE_DEBUG:
                        print(f"    [ok] {_cfg_label(config)}  combined frac>{correctness_threshold:.0e}={cbad:.2%} max={cmax:.3e}", flush=True)
                    out_bufs = ()  # handled — skip the per-partial loop below
                for oi, (ob, rb) in enumerate(zip(out_bufs, ref_arrays)):
                    if isinstance(ob, AlloyBuffer) and isinstance(rb, np.ndarray):
                        o_arr = np.asarray(ob.numpy).astype(np.float32)
                        r_arr = rb.astype(np.float32)
                        if o_arr.shape == r_arr.shape:
                            # Empty-split sentinel: split-K attention marks splits covering no
                            # (or underflow-only) KV with a huge sentinel lse (m_init·LN2 ≈ -6.9e29).
                            # Which splits are "empty" is a borderline, tile-config-dependent fp
                            # call, so the raw lse flips between the sentinel and a tiny real value
                            # across configs even though partial_O agrees and those splits carry
                            # negligible mass (the combine weights a sentinel lse to ~0). Mask
                            # sentinel-magnitude entries so the check validates the real attention
                            # values, not empty-split bookkeeping — otherwise every HEAD_DIM=256
                            # config is wrongly rejected and only a slow fallback survives.
                            sentinel = (np.abs(o_arr) > 1e20) | (np.abs(r_arr) > 1e20)
                            diff = np.where(sentinel, 0.0, np.abs(o_arr - r_arr))
                            max_diff = float(diff.max()) if diff.size else 0.0
                            if max_diff > correctness_threshold:
                                if _TUNE_DEBUG:
                                    fi = int(diff.argmax())
                                    idx = tuple(int(x) for x in np.unravel_index(fi, o_arr.shape))
                                    nbad = int((diff > correctness_threshold).sum())
                                    # Cross-reference the companion output (partial_O vs lse) at
                                    # the same leading indices: distinguishes a benign empty-split
                                    # lse sentinel (companion ~0 in both → combine ignores it) from
                                    # a dropped real contribution (companion real but excluded → bug).
                                    extra = ""
                                    if len(out_bufs) == 2 and isinstance(out_bufs[1 - oi], AlloyBuffer):
                                        cg = np.asarray(out_bufs[1 - oi].numpy).astype(np.float32)
                                        cr = np.asarray(ref_arrays[1 - oi]).astype(np.float32)
                                        if cg.ndim >= len(idx):
                                            extra = (
                                                f"  companion out[{1 - oi}]@{idx}: "
                                                f"|got|={float(np.abs(cg[idx]).max()):.2e} |ref|={float(np.abs(cr[idx]).max()):.2e}"
                                            )
                                    print(
                                        f"    [skip:precision] {_cfg_label(config)}  out[{oi}] "
                                        f"max_diff={max_diff:.3e} @ {idx} (got={float(o_arr.flat[fi]):.3e} ref={float(r_arr.flat[fi]):.3e}) "
                                        f"nbad={nbad}/{o_arr.size}{extra}",
                                        flush=True,
                                    )
                                return None
                            if _TUNE_DEBUG:
                                print(f"    [ok] {_cfg_label(config)}  out[{oi}] max_diff={max_diff:.3e} <= thr={correctness_threshold:.1e} (masked {int(sentinel.sum())} sentinel)", flush=True)

        # Measurement
        gpu_times: list[float] = []
        for _ in range(n_runs):
            gpu_us_list.clear()
            try:
                _dispatch()
            except Exception:
                return None
            if gpu_us_list:
                gpu_times.append(gpu_us_list[-1])

        if not gpu_times:
            return None

        # Trimmed mean: drop top/bottom 20%
        gpu_times.sort()
        n = len(gpu_times)
        trim = max(1, n // 5)
        trimmed = gpu_times[trim : n - trim]
        if not trimmed:
            return gpu_times[n // 2]
        return sum(trimmed) / len(trimmed)

    except Exception:
        return None
    finally:
        _metal_ext.dispatch = orig_dispatch
        kernel._benchmark_options = None
        # Clear caches + global buf map that holds strong refs to AlloyBuffers,
        # preventing GC from releasing Metal allocations between configs.
        _engine.clear_run()
        _alloy_buf_map.clear()


def _compute_reference(kernel, input_arrays, config=None, extra_kwargs=None):
    """Compute reference output using a safe config and snapshot it.

    The kernel writes into the output buffers from ``input_arrays``. Subsequent
    `benchmark_config` dispatches reuse those same buffers, so the live
    reference would be silently overwritten before any correctness comparison.
    Snapshot each AlloyBuffer output to a numpy array immediately so the
    reference survives across later dispatches.
    """
    if config is None:
        config = TuneConfig(constexprs={"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32, "_reg": 1})

    kwargs = dict(extra_kwargs) if extra_kwargs else {}
    kwargs.update(config.constexprs)
    kernel._benchmark_options = config.options

    lazy = []
    for pname, arr in input_arrays:
        if isinstance(arr, AlloyBuffer):
            lazy.append(arr)
        else:
            lazy.append(AlloyBuffer(arr=arr, shape=arr.shape, dtype=arr.dtype))

    result = kernel(*lazy, **kwargs)
    if isinstance(result, tuple):
        materialize_many(result)
        snapshots = tuple(
            (np.asarray(b.numpy).copy() if isinstance(b, AlloyBuffer) else b) for b in result
        )
    elif isinstance(result, AlloyBuffer):
        materialize_many((result,))
        snapshots = np.asarray(result.numpy).copy()
    else:
        snapshots = result

    kernel._benchmark_options = None
    return snapshots


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------


def _make_test_buffers(
    kernel,
    key_values: dict[str, int],
    buffer_dtypes: dict[str, Any] | None = None,
    buffer_snapshots: dict[str, np.ndarray] | None = None,
):
    """Create random test buffers matching the kernel's parameter signature and the given shapes.

    `buffer_dtypes` maps each buffer name to its captured dtype (from the trace
    pass). When provided, allocates each buffer at its real production dtype so
    cooperative loads compile to the same MSL the pipeline runs (bf16 vs f32 is
    not just a 2× memory cost — it picks a different MMA intrinsic). When
    omitted, falls back to f32 for everything.
    """
    buffers: list[tuple[str, AlloyBuffer]] = []

    # Reconstruct buffer shapes from key_values
    # key_values has entries like _A_dim0=16, _A_dim1=2048, _B_dim0=768, etc.
    buf_shapes: dict[str, list[int]] = {}
    for k, v in key_values.items():
        if k.startswith("_") and "_dim" in k:
            parts = k.rsplit("_dim", 1)
            buf_name = parts[0][1:]  # strip leading _
            dim_idx = int(parts[1])
            if buf_name not in buf_shapes:
                buf_shapes[buf_name] = []
            while len(buf_shapes[buf_name]) <= dim_idx:
                buf_shapes[buf_name].append(1)
            buf_shapes[buf_name][dim_idx] = v

    for buf_name, shape in buf_shapes.items():
        shape_tuple = tuple(shape)
        dtype = (buffer_dtypes or {}).get(buf_name, float32)
        buf = _alloc_aligned(shape_tuple, dtype)
        # Fill with small random values via numpy view (works for f32; for bf16
        # the .numpy view returns the underlying uint16 storage, fill with f32
        # noise then cast).
        try:
            arr = buf.numpy
            snapshot = (buffer_snapshots or {}).get(buf_name)
            # A captured snapshot can be non-finite — e.g. a buffer the extraction
            # forward left NaN (gemma4's head_dim-512 Q under a fallback block).
            # Feeding NaN/inf makes every config produce NaN and the kernel
            # un-tunable, so fall back to random when the snapshot isn't all-finite.
            snapshot_ok = (
                snapshot is not None
                and snapshot.shape == arr.shape
                and (not np.issubdtype(snapshot.dtype, np.floating) or bool(np.isfinite(snapshot).all()))
            )
            if snapshot_ok:
                np.copyto(arr, snapshot.astype(arr.dtype, copy=False))
            else:
                np.copyto(arr, np.random.randn(*shape_tuple).astype(arr.dtype) * 0.1)
        except Exception:
            pass
        buffers.append((buf_name, buf))

    return buffers


def tune_kernel(
    kernel,
    key_values: dict[str, int],
    input_arrays: list[tuple[str, Any]] | None = None,
    device: str | None = None,
    extra_kwargs: dict[str, Any] | None = None,
    buffer_dtypes: dict[str, Any] | None = None,
    buffer_snapshots: dict[str, np.ndarray] | None = None,
) -> dict | None:
    """Tune a single kernel at a specific shape. Returns best entry dict or None."""
    if device is None:
        device = detect_device()

    configs = kernel._tune_configs
    if not configs:
        return None

    configs = prune_configs(kernel.name, configs, key_values)
    if not configs:
        return None

    if input_arrays is None:
        input_arrays = _make_test_buffers(
            kernel,
            key_values,
            buffer_dtypes=buffer_dtypes,
            buffer_snapshots=buffer_snapshots,
        )
    if not input_arrays:
        return None

    # Keep the tolerance tight for f32 compute (errors compound through many
    # downstream transformer layers), but use the f16 tolerance when the kernel's
    # float *inputs* are f16 — the output then carries normal f16 rounding
    # (~1e-3) that varies across tile configs and must not be flagged incorrect.
    # The gate is "an f32 float INPUT", which excludes both 4-byte int index
    # buffers (Q_START_POS / cache_position) and the f32 OUTPUT buffer (the
    # f32-accumulated result — its dtype reflects the accumulator, not the MMA
    # operand precision). Treating either as f32 compute forced the 0.0001
    # threshold on f16-Q attention, rejecting every tile whose f16 rounding
    # diverged from the reference (f16-Q at head_dim 256 produces ~2e-4 diffs)
    # and leaving only a slow fallback config.
    threshold = 0.01
    _f32_trigger = None
    for pname, buf in input_arrays:
        if (
            isinstance(buf, AlloyBuffer)
            and pname not in kernel._output_params
            and buf._dtype == float32
        ):
            threshold = 0.0001
            _f32_trigger = pname
            break
    if _TUNE_DEBUG:
        _why = f"f32 input '{_f32_trigger}'" if _f32_trigger else "no f32 input"
        print(f"  [threshold] {kernel.name}: {threshold:.1e} ({_why})", flush=True)

    # Compute reference output with conservative config. If that fails (e.g.,
    # shmem overflow at HEAD_DIM=128), fall back through the candidate sweep
    # itself: try each config in order, use the first one that produces a
    # NaN-free output as the reference. Without a reference, the tuner cannot
    # validate correctness and can pick a fast-but-wrong config (the dq kernel
    # silently produces wrong output at BM=16 BN=8 with HEAD_DIM=128).
    ref_cfg = _CONSERVATIVE_DEFAULTS.get(kernel.name)
    reference = None
    if ref_cfg:
        ref_tune_cfg = TuneConfig(
            constexprs=dict(ref_cfg.constexprs), options=dict(ref_cfg.options)
        )
    else:
        ref_tune_cfg = TuneConfig(constexprs={"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32})
    try:
        reference = _compute_reference(kernel, input_arrays, ref_tune_cfg, extra_kwargs=extra_kwargs)
    except Exception:
        # Conservative config didn't compile/run. Walk the sweep and use the
        # first compiling config as the reference. Configs the kernel rejects
        # (compile failure, shmem overflow) are silently skipped here.
        for cand in configs:
            try:
                reference = _compute_reference(kernel, input_arrays, cand, extra_kwargs=extra_kwargs)
                break
            except Exception:
                continue
    if reference is None:
        print(
            f"  WARNING: no reference output for {kernel.name} "
            "(no candidate config compiled at this shape) — skipping kernel.",
            flush=True,
        )
        return None

    n_total = len(configs)
    results: list[tuple[TuneConfig, float | None]] = []
    best_time = float("inf")
    best_config = None
    t0 = time.perf_counter()

    for i, cfg in enumerate(configs):
        t = benchmark_config(
            kernel,
            cfg,
            input_arrays,
            reference_output=reference,
            correctness_threshold=threshold,
            extra_kwargs=extra_kwargs,
        )
        results.append((cfg, t))
        if t is not None and t < best_time:
            best_time = t
            best_config = cfg
        # Progress
        elapsed = time.perf_counter() - t0
        if (i + 1) % 50 == 0 or i == n_total - 1:
            status = f"{best_time:.1f}µs" if best_config else "no valid"
            print(
                f"  [{i + 1}/{n_total}] {elapsed:.1f}s elapsed, best={status}",
                flush=True,
            )

    if best_config is None:
        print(f"  WARNING: no valid config found for {kernel.name}", flush=True)
        return None

    return {
        "key": key_values,
        "config": best_config.constexprs,
        "options": best_config.options if best_config.options else {},
        "gpu_us": round(best_time, 2),
        "source": "al.tune",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# JSON merge
# ---------------------------------------------------------------------------


def _config_file_path(device: str) -> Path:
    """Default write target for a device's tuned configs — the user data dir.

    Not the shipped package dir (read-only on a system install, wiped on
    upgrade); serving overlays this file on the shipped baseline at dispatch."""
    return user_config_file(device)


def _load_existing(path: Path) -> dict:
    """Load existing JSON config file, or return empty structure."""
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("pipeline_version") == _PIPELINE_VERSION:
                return data
        except (json.JSONDecodeError, KeyError):
            pass
    return {
        "pipeline_version": _PIPELINE_VERSION,
        "device": "",
        "generated": "",
        "entries": {},
    }


def _merge_entry(existing_entries: list[dict], new_entry: dict) -> list[dict]:
    """Merge a new entry into an existing list, updating by key match.

    Hand-tuned entries (source='hand') are never overwritten by al.tune.
    """
    new_key = tuple(sorted(new_entry["key"].items()))
    for i, e in enumerate(existing_entries):
        if tuple(sorted(e["key"].items())) == new_key:
            if e.get("source") == "hand":
                return existing_entries  # preserve hand-tuned
            existing_entries[i] = new_entry
            return existing_entries
    existing_entries.append(new_entry)
    return existing_entries


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def tune(
    target,
    inputs=None,
    *,
    device: str | None = None,
    output: str | None = None,
    training: bool = False,
    only: str | None = None,
    record_only: bool = False,
    q_start_pos: int | None = None,
    **kwargs,
):
    """Tune kernel configs and write to JSON.

    Args:
        target: A model (torch.nn.Module) or a kernel (KernelFunction)
        inputs: Sample inputs for the model (required if target is a model)
        device: Device name override (default: auto-detect)
        output: Output file path override (default: configs/{device}.json)
        training: Capture backward+optimizer kernels by running an autograd pass.
        only: Optional regex; if provided, only kernels whose name matches
            ``re.search(only, name)`` are tuned. Existing entries for other
            kernels are preserved untouched in the JSON. Useful for re-tuning
            a single subsystem (e.g. ``only=r"attention_strided_backward"``)
            without re-running the full sweep.
        record_only: extract shapes without dispatching the GPU (phantom
            intermediates). The per-kernel benchmark synthesizes its own
            buffers, so only shapes are needed — this skips the whole (untuned,
            slow) extraction forward, which at a warm prefill offset runs the
            full-context attention over every layer just to recover shapes.
        q_start_pos: re-inject this absolute cache offset into the runtime
            ``Q_START_POS_BUF`` of every captured warm-prefill attention. The
            offset sets the attention's causal K-loop trip count, so the tuner
            must benchmark at the production offset; record_only drops the live
            snapshot, so the caller (which knows the offset) supplies it here.
        **kwargs: For kernel tuning, shape dimensions (M=16, K=2048, N=5632)
    """
    if device is None:
        device = detect_device()

    name_filter = re.compile(only) if only else None
    _tune_t0 = time.perf_counter()
    logger.info(
        "tune_session_start",
        target=type(target).__name__,
        device=device,
        training=training,
        only=only,
    )

    captured: list[CapturedKernel] = []
    if isinstance(target, KernelFunction):
        # Single kernel tuning — synthesize buffers
        if not kwargs:
            raise ValueError("Provide shape dimensions: al.tune(kernel, M=16, K=2048, N=5632)")
        # `buffer_dtypes` is a per-buffer dtype map consumed by
        # `_make_test_buffers` to allocate test buffers at production dtypes;
        # it must not enter the tune cache key (dict values are unhashable
        # and dtype is already encoded in the buffer itself).
        single_buffer_dtypes = kwargs.pop("buffer_dtypes", None)
        captured = [
            CapturedKernel(
                name=target.name,
                key_values=kwargs,
                kwargs=kwargs,
                buffer_args=[],
                buffer_snapshots={},
                kernel=target,
                buffer_dtypes_override=single_buffer_dtypes,
            )
        ]
    else:
        # Model tuning — extract shapes with real buffers
        if inputs is None:
            raise ValueError("Provide sample inputs: al.tune(model, sample_inputs)")
        print(f"Extracting kernel shapes (training={training})...", flush=True)
        captured = _extract_shapes(
            target, inputs, training=training, record_only=record_only
        )
        # The warm-prefill attention reads its absolute cache offset from a
        # runtime Q_START_POS_BUF, not a constexpr — so the captured shape is
        # offset-independent but the benchmark's causal K-loop trip count is
        # NOT. record_only drops the live snapshot; re-inject the known offset
        # so the tuner benchmarks the warm extent, not a degenerate cold scan.
        if q_start_pos is not None:
            for cap in captured:
                if any(pn == "Q_START_POS_BUF" for pn, _ in cap.buffer_args):
                    cap.buffer_snapshots["Q_START_POS_BUF"] = np.array(
                        [q_start_pos], dtype=np.int32
                    )
        print(f"Found {len(captured)} unique (kernel, shape) pairs", flush=True)

    out_path = Path(output) if output else _config_file_path(device)
    data = _load_existing(out_path)
    data["device"] = device
    data["generated"] = datetime.now(timezone.utc).isoformat()

    # Suspend training-mode for the tune sweep. ``AlloyBuffer.__del__``
    # otherwise refuses to call ``buf_release`` while training mode is
    # on — a guard that exists for ``tensor.set_()`` zero-copy storage
    # in the compiled-plan path. The tuner doesn't use compiled plans,
    # but every output buffer ``_dispatch()`` allocates inherits the
    # same wrapper; without the release path enabled they pile up at
    # ~32 MB per dispatch × n_warmup+1+n_runs (=21) per config × hundreds
    # of configs and the OS OOM-kills the run mid-sweep on FFN shapes.
    prev_training = _training_mode_enabled()
    if prev_training:
        _set_training_mode(False)

    # Tune each (kernel, shape) pair
    summary: dict[str, list[dict]] = {}
    if name_filter is not None:
        kept = [c for c in captured if name_filter.search(c.name)]
        skipped = len(captured) - len(kept)
        if not kept:
            print(
                f"No kernels matched only={name_filter.pattern!r}; nothing to tune.",
                flush=True,
            )
            return summary
        print(
            f"Filter only={name_filter.pattern!r}: tuning {len(kept)} kernels, "
            f"skipping {skipped} non-matching.",
            flush=True,
        )
        captured = kept
    for i, cap in enumerate(captured):
        kernel = cap.kernel
        if not kernel._tune_configs:
            continue

        # Compact shape display
        dims = {k: v for k, v in cap.key_values.items() if k.startswith("_") and "dim" in k}
        shape_str = " ".join(f"{k}={v}" for k, v in sorted(dims.items()))
        n_configs = len(kernel._tune_configs)

        print(
            f"\n[{i + 1}/{len(captured)}] Tuning {cap.name} ({shape_str}) — {n_configs} configs",
            flush=True,
        )

        # Always synthesize fresh test buffers from the captured shapes.
        # The AlloyBuffers in ``cap.buffer_args`` are wrappers around storage
        # that gets released when the torch tensors from the trace pass go
        # out of scope (training mode skips ``__del__`` for those handles
        # but Dynamo still drops the tensors). By the time we reach the
        # later captured kernels, several of those buffer wrappers point at
        # freed memory — every dispatch raises silently, ``benchmark_config``
        # returns ``None``, and tuning prints "no valid config found" for
        # the rest of the run. Synthesizing per-shape costs one alloc and
        # is the only reliable path.
        # Pass cap.kwargs (the actual constexprs from the captured dispatch:
        # SEQ_LEN, HEAD_DIM, KV_GROUP, HIGH_PRECISION, all strides) so the kernel
        # measures the real workload, not a degenerate default-constexpr no-op.
        # The tunable knobs (BLOCK_M, BLOCK_N, ...) are stripped so per-config
        # values from the sweep aren't shadowed.
        extras = {k: v for k, v in cap.kwargs.items() if k not in kernel._tune_tuned_params}
        # Capture each input buffer's dtype so synthesized test data matches
        # production (bf16 vs f32 picks different MMA intrinsics + memory
        # bandwidth, so f32 test buffers don't reflect bf16 production cost).
        # Single-kernel callers can override via `buffer_dtypes=` kwarg to
        # `al.tune()`, exposed here through `cap.buffer_dtypes_override`.
        if cap.buffer_dtypes_override is not None:
            buf_dtypes = {
                pn: from_name(dt) if isinstance(dt, str) else dt
                for pn, dt in cap.buffer_dtypes_override.items()
            }
        else:
            buf_dtypes = {pn: arg._dtype for pn, arg in cap.buffer_args}
        entry = tune_kernel(
            kernel, cap.key_values,
            input_arrays=None, device=device,
            extra_kwargs=extras, buffer_dtypes=buf_dtypes,
            buffer_snapshots=cap.buffer_snapshots,
        )
        if entry is not None:
            kernel_entries = data["entries"].setdefault(cap.name, [])
            data["entries"][cap.name] = _merge_entry(kernel_entries, entry)
            summary.setdefault(cap.name, []).append(entry)
            opts = entry.get("options") or {}
            opts_str = f" options={opts}" if opts else ""
            print(
                f"  Best: {entry['config']}{opts_str} @ {entry['gpu_us']}µs",
                flush=True,
            )
            logger.info(
                "kernel_tune_complete",
                kernel=cap.name,
                shape=shape_str,
                n_configs=n_configs,
                best_config=entry["config"],
                best_options=opts or None,
                best_gpu_us=entry["gpu_us"],
            )

    # Restore training-mode flag for whatever runs after this tune call.
    if prev_training:
        _set_training_mode(True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}", flush=True)
    logger.info(
        "config_write_to_disk",
        path=str(out_path),
        n_kernels=len(data["entries"]),
        n_entries=sum(len(v) for v in data["entries"].values()),
    )
    logger.info(
        "tune_session_complete",
        n_kernels_tuned=len(summary),
        n_kernels_skipped=len(captured) - len(summary),
        took_s=round(time.perf_counter() - _tune_t0, 2),
    )

    return summary


def tune_report(device: str | None = None):
    """Print a summary of tuned configs for the given device."""
    if device is None:
        device = detect_device()

    _load_configs(device)

    total = sum(len(v) for v in _STATIC_CONFIGS.values())
    print(f"{device}: {total} entries across {len(_STATIC_CONFIGS)} kernels")
    for kernel_name, entries in sorted(_STATIC_CONFIGS.items()):
        print(f"  {kernel_name}: {len(entries)} shapes")
