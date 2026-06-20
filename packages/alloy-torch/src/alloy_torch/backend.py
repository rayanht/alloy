"""torch.compile backend that lowers AOT FX graphs to Alloy kernels."""

from __future__ import annotations
from alloy._compiler.dtypes import DType, from_torch_dtype, uint8
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal
from alloy._dispatch.dispatch import DispatchEngine, _engine
from alloy._dispatch.fusion_types import RecordedDispatch

import ctypes
import operator
import pickle
import os
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any
from alloy._dispatch.buf_utils import (
    _alloc_aligned,
    _alloy_buf_map,
    _alloy_handle_map,
    is_record_only,
)

import torch
import torch.fx
from functorch.compile import make_boxed_func
from alloy import get_logger
from alloy._runtime import _metal_ext
from alloy._runtime.alloy_buffer import AlloyBuffer, _compute_contiguous_strides, materialize_many
from alloy._runtime.tune_configs import GRID_SHRINK_REP_M
from torch._dynamo.backends.common import aot_autograd

from alloy_torch.decomps import get_alloy_decompositions
from alloy_torch.mode import is_training_mode_enabled
from alloy_torch.compile_window import compile_window
from alloy_torch.ops.linalg import _mm_batched_cache
from alloy_torch.ops.dropout import refresh_dropout_seed
from alloy_torch.extern_kv import drain_extern_kv_writes
from alloy_torch.ops.registry import FX_CALL_HANDLERS
from alloy_torch.tensor_bridge import IR_TO_TORCH, make_tensor_from_ptr
from alloy_torch.rewrites.pipeline import rewrite_fx_graph


logger = get_logger("alloy_torch.backend")

_UNRESOLVED = object()

_all_compiled_plans: list = []  # for visualization


@dataclass
class CapturedPlan:
    """The (plan, args) a compiled module produced or executed inside a
    `capture_plan()` scope."""

    plan: "CompiledPlan | None" = None
    args: tuple | None = None


# Compiled-module calls record into the innermost (top-of-stack) slot only.
_capture_stack: list[CapturedPlan] = []


@contextmanager
def capture_plan() -> "Iterator[CapturedPlan]":
    """Scope a (plan, args) capture. The slot holds the MOST RECENT call's plan
    when the block exits — wrap the warmup call(s) whose plan you want to pin
    (typically a run-0 build then a run-1 execute, leaving the slot holding the
    executed plan with `_cached_input_updates` populated)."""
    slot = CapturedPlan()
    _capture_stack.append(slot)
    try:
        yield slot
    finally:
        _capture_stack.pop()


@dataclass(frozen=True, slots=True)
class InputPtrInfo:
    """Run-0 metadata for a torch input storage pointer."""

    arg_idx: int
    view_offset: int


def _to_lazy_input(
    value: Any,
) -> AlloyBuffer | tuple[AlloyBuffer, ...] | list[AlloyBuffer] | dict[str, AlloyBuffer]:

    if isinstance(value, torch.Tensor):
        # Alloy bool is int32 internally; a torch bool tensor is 1 byte/element.
        # Binding its storage with int32 would read 4 packed bool bytes as one int
        # (0x01010101). Widen to int32 (kept alive via _ext_ref below).
        if value.dtype == torch.bool:
            value = value.to(torch.int32)
        # Use storage base ptr (not view data_ptr) so views of the same storage
        # (Q/K/V from split QKV) share a Metal buffer and the kernel can stride
        # across the full allocation.
        storage = value.untyped_storage()
        ptr = storage.data_ptr()
        offset_bytes = value.storage_offset() * value.element_size()
        shape = tuple(value.shape)
        strides = tuple(s * value.element_size() for s in value.stride())
        # from_torch_dtype handles bfloat16/uint16/32 that the numpy map would
        # silently coerce to float32 at the wrong itemsize.
        dtype = from_torch_dtype(value.dtype)
        nbytes = storage.nbytes()
        lb = AlloyBuffer.from_raw_ptr(ptr, shape, strides, dtype, nbytes)
        lb._offset = offset_bytes
        lb._ext_ref = value  # prevent torch from freeing the storage
        return lb
    if isinstance(value, (tuple, list)):
        return type(value)(_to_lazy_input(item) for item in value)
    return value


# --- Output conversion ---


def _abv_to_torch(abv: AlloyBuffer) -> torch.Tensor:
    """Create a torch tensor from an AlloyBuffer. No numpy."""
    dt = abv._dtype
    torch_dt = IR_TO_TORCH.get(dt.ir, torch.float32)
    elems = abv.size
    if elems == 0:
        return torch.empty(abv._shape, dtype=torch_dt)
    itemsize = dt.itemsize

    if abv.is_contiguous():
        nbytes = elems * itemsize
        raw = (ctypes.c_uint8 * nbytes).from_address(abv.data_ptr)
        flat = torch.frombuffer(raw, dtype=torch_dt, count=elems)
        # Keep the AlloyBuffer alive on the STORAGE so alias/view ops (sharing
        # storage) also keep the Metal buffer from being freed.
        flat.untyped_storage()._alloy_ref = (abv, raw)
        if abv._shape != (elems,):
            return flat.reshape(abv._shape)
        return flat

    # Non-contiguous. Default path is a numpy view + torch.from_numpy (numpy's
    # as_strided tolerates strides that read past the backing storage, e.g. GQA
    # K/V caches with a larger max-seq footprint).
    #
    # bf16 fallback: torch.from_numpy rejects ml_dtypes.bfloat16, so build the
    # view from storage via torch.frombuffer + torch.as_strided. That path is
    # strict about bounds, so `flat` must cover the furthest element the strides
    # reach.
    if torch_dt is torch.bfloat16:
        if abv._parent_handle >= 0:
            total_nbytes = _metal_ext.buf_nbytes(abv._parent_handle)
            base_ptr = _metal_ext.buf_ptr(abv._parent_handle)
        else:
            total_nbytes = abv._total_nbytes
            base_ptr = abv._raw_ptr
        # Span the strided view reaches: last addressable element + one more,
        # plus abv._offset, capped against the backing allocation.
        max_idx_bytes = abv._offset + sum(
            max(0, s - 1) * st for s, st in zip(abv._shape, abv._strides)
        ) + itemsize
        span_nbytes = min(max_idx_bytes, total_nbytes)
        span_elems = span_nbytes // itemsize
        raw = (ctypes.c_uint8 * span_nbytes).from_address(base_ptr)
        flat = torch.frombuffer(raw, dtype=torch_dt, count=span_elems)
        elem_offset = abv._offset // itemsize
        strides_elem = tuple(s // itemsize for s in abv._strides)
        result = torch.as_strided(flat, abv._shape, strides_elem, storage_offset=elem_offset)
        result.untyped_storage()._alloy_ref = (abv, raw)
        return result

    np_view = abv.numpy
    result = torch.from_numpy(np_view)
    result.untyped_storage()._alloy_ref = (abv, np_view)
    return result


def _convert_to_torch_output(
    value: AlloyBuffer | tuple[AlloyBuffer, ...] | list[AlloyBuffer] | dict[str, AlloyBuffer],
) -> torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor] | dict[str, torch.Tensor]:
    if value is None:
        return None
    if isinstance(value, list):
        return [_convert_to_torch_output(item) for item in value]
    elif isinstance(value, tuple):
        return tuple(_convert_to_torch_output(item) for item in value)
    elif isinstance(value, Mapping):
        return {key: _convert_to_torch_output(item) for key, item in value.items()}
    return _abv_to_torch(value)


def _to_torch_output(
    value: Any,
) -> torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor] | dict[str, torch.Tensor]:
    materialize_many(value)
    return _convert_to_torch_output(value)


def _dummy_torch_output(value: Any) -> Any:
    """Shape/dtype-matching zeros for record-only compile (phantom outputs hold
    no data, so the real bytes are never read; the caller discards them)."""
    if value is None:
        return None
    if isinstance(value, list):
        return [_dummy_torch_output(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_dummy_torch_output(v) for v in value)
    if isinstance(value, Mapping):
        return {key: _dummy_torch_output(v) for key, v in value.items()}
    if isinstance(value, AlloyBuffer):
        torch_dt = IR_TO_TORCH.get(value._dtype.ir, torch.float32)
        return torch.zeros(value._shape, dtype=torch_dt)
    return value


# --- Compilation helpers ---


def _get_nested_attr(module: torch.nn.Module, target: str) -> object:
    value: object = module
    for part in target.split("."):
        if not isinstance(value, torch.nn.Module):
            if hasattr(value, "__dict__") and part in value.__dict__:
                value = value.__dict__[part]
                continue
            raise AttributeError(f"{type(value).__name__} has no nested attribute {part!r}")
        if part in value._modules:
            value = value._modules[part]
        elif part in value._parameters:
            value = value._parameters[part]
        elif part in value._buffers:
            value = value._buffers[part]
        elif part in value.__dict__:
            value = value.__dict__[part]
        else:
            raise AttributeError(f"{value._get_name()} has no attribute {part!r}")
    return value


def _unsupported_targets(gm: torch.fx.GraphModule) -> list[str]:
    unsupported: list[str] = []
    for node in gm.graph.nodes:
        if node.op in {"call_function", "call_method"} and node.target not in FX_CALL_HANDLERS:
            unsupported.append(str(node.target))
    return unsupported


def _prepack_constant(value: AlloyBuffer | tuple | list | dict) -> Any:
    if isinstance(value, AlloyBuffer):
        src = value
        if src.nbytes == 0:
            buf = _alloc_aligned(src._shape, src._dtype)
            _engine.untrack_alloc(buf.base_ptr)  # not an intermediate
            return buf
        if not src.is_contiguous():
            src = value.contiguous()
        src.sync()
        buf = _alloc_aligned(src._shape, src._dtype)
        buf.copy_from(src)
        _engine.untrack_alloc(buf.base_ptr)  # constant, not intermediate
        return buf
    if isinstance(value, (tuple, list)):
        return type(value)(_prepack_constant(item) for item in value)
    return value


def _resolve_arg(arg: Any, env: dict[torch.fx.Node, Any]) -> Any:
    if isinstance(arg, torch.fx.Node):
        return env[arg]
    if isinstance(arg, (tuple, list)):
        return type(arg)(_resolve_arg(item, env) for item in arg)
    if isinstance(arg, dict):
        return {key: _resolve_arg(value, env) for key, value in arg.items()}
    if isinstance(arg, slice):
        return slice(
            _resolve_arg(arg.start, env),
            _resolve_arg(arg.stop, env),
            _resolve_arg(arg.step, env),
        )
    return arg


def _resolve_const_arg(arg: Any, env: dict[torch.fx.Node, Any]) -> Any:
    if isinstance(arg, torch.fx.Node):
        return env.get(arg, _UNRESOLVED)
    if isinstance(arg, (tuple, list)):
        items = type(arg)(_resolve_const_arg(item, env) for item in arg)
        return _UNRESOLVED if any(item is _UNRESOLVED for item in items) else items
    if isinstance(arg, slice):
        start = _resolve_const_arg(arg.start, env)
        stop = _resolve_const_arg(arg.stop, env)
        step = _resolve_const_arg(arg.step, env)
        if any(item is _UNRESOLVED for item in (start, stop, step)):
            return _UNRESOLVED
        return slice(start, stop, step)
    if isinstance(arg, dict):
        items = {key: _resolve_const_arg(value, env) for key, value in arg.items()}
        return _UNRESOLVED if any(value is _UNRESOLVED for value in items.values()) else items
    return arg


def _build_constant_env(gm: torch.fx.GraphModule):
    """Constant-fold all nodes whose inputs are fully resolved constants.

    Any call_function node with a handler in FX_CALL_HANDLERS is foldable
    if all its args/kwargs resolve to already-folded values. No whitelist —
    the only gate is whether the handler succeeds without exception.
    """
    const_env = {}
    for node in gm.graph.nodes:
        if node.op == "get_attr":
            value = _get_nested_attr(gm, node.target)
            if isinstance(value, torch.Tensor):
                continue
            const_env[node] = _prepack_constant(_to_lazy_input(value))
            continue
        if node.op != "call_function":
            continue
        if node.target not in FX_CALL_HANDLERS:
            continue
        args = _resolve_const_arg(node.args, const_env)
        kwargs = _resolve_const_arg(node.kwargs, const_env)
        if args is _UNRESOLVED or kwargs is _UNRESOLVED:
            continue
        try:
            const_env[node] = _prepack_constant(FX_CALL_HANDLERS[node.target](*args, **kwargs))
        except Exception:
            continue
    return const_env


class Opcode(Enum):
    INPUT = 0
    CONST = 1
    ATTR = 2
    CALL = 3
    GETITEM = 4
    OUTPUT = 5


def _build_execution_plan(
    gm: torch.fx.GraphModule, constant_env: dict
) -> tuple[list[tuple[Opcode, int, Any]], int]:
    """Compile FX graph into a flat instruction list for execution.

    Returns list of (opcode, result_value_index, slot_data).
    All node references are replaced with integer indices into a values array.
    """
    node_to_idx: dict[torch.fx.Node, int] = {}
    plan: list[tuple[Opcode, int, Any]] = []
    next_idx: int = 0
    nodes = list(gm.graph.nodes)

    def _arg_index(arg):
        """Convert an FX arg to a value index or literal."""
        if isinstance(arg, torch.fx.Node):
            return ("ref", node_to_idx[arg])
        if isinstance(arg, (tuple, list)):
            items = [_arg_index(a) for a in arg]
            has_ref = any(isinstance(a, tuple) and a[0] == "ref" for a in items)
            if has_ref:
                kind = "tuple" if isinstance(arg, tuple) else "list"
                return (kind, items)
            return ("lit", type(arg)(a[1] for a in items))
        if isinstance(arg, dict):
            items = {k: _arg_index(v) for k, v in arg.items()}
            has_ref = any(isinstance(v, tuple) and v[0] == "ref" for v in items.values())
            if has_ref:
                return ("dict", items)
            return ("lit", {k: v[1] for k, v in items.items()})
        if isinstance(arg, slice):
            return ("slice", _arg_index(arg.start), _arg_index(arg.stop), _arg_index(arg.step))
        return ("lit", arg)

    def _emit_getitem_users(parent_vi, node):
        nonlocal next_idx
        for user in node.users:
            if user.op == "call_function" and user.target is operator.getitem:
                ui = next_idx
                next_idx += 1
                node_to_idx[user] = ui
                plan.append((Opcode.GETITEM, ui, (parent_vi, user.args[1])))

    for node in nodes:
        if node in constant_env:
            vi = next_idx
            next_idx += 1
            node_to_idx[node] = vi
            plan.append((Opcode.CONST, vi, constant_env[node]))
            _emit_getitem_users(vi, node)
            continue

        if node.op == "placeholder":
            vi = next_idx
            next_idx += 1
            node_to_idx[node] = vi
            plan.append((Opcode.INPUT, vi, None))
            continue

        if node.op == "get_attr":
            vi = next_idx
            next_idx += 1
            node_to_idx[node] = vi
            plan.append((Opcode.ATTR, vi, node.target))
            continue

        if node.op in {"call_function", "call_method"}:
            handler = FX_CALL_HANDLERS[node.target]
            arg_slots = [_arg_index(a) for a in node.args]
            kwarg_slots = {k: _arg_index(v) for k, v in node.kwargs.items()}
            vi = next_idx
            next_idx += 1
            node_to_idx[node] = vi
            plan.append((Opcode.CALL, vi, (handler, arg_slots, kwarg_slots)))
            _emit_getitem_users(vi, node)
            continue

        if node.op == "output":
            out_slots = (
                [_arg_index(a) for a in node.args[0]]
                if isinstance(node.args[0], (tuple, list))
                else [_arg_index(node.args[0])]
            )
            plan.append((Opcode.OUTPUT, -1, out_slots))
            break

        raise RuntimeError(f"Alloy backend: unsupported node kind {node.op!r}")

    return plan, next_idx


def _resolve_from_values(slot, values):
    """Resolve an arg slot to its value from the values array."""
    kind = slot[0]
    if kind == "ref":
        return values[slot[1]]
    if kind == "lit":
        return slot[1]
    if kind == "tuple":
        return tuple(_resolve_from_values(s, values) for s in slot[1])
    if kind == "list":
        return [_resolve_from_values(s, values) for s in slot[1]]
    if kind == "dict":
        return {k: _resolve_from_values(v, values) for k, v in slot[1].items()}
    if kind == "slice":
        return slice(
            _resolve_from_values(slot[1], values),
            _resolve_from_values(slot[2], values),
            _resolve_from_values(slot[3], values),
        )
    raise ValueError(f"Unknown arg slot kind: {kind!r}")


def _extract_mutation_map(gm: torch.fx.GraphModule) -> dict[int, int]:
    """Parse AOT Autograd's input-mutation annotations from graph meta.

    Returns {output_idx: input_arg_idx} for each output that is marked as
    an in-place mutation of an input placeholder. The arg_idx is the
    placeholder's position among call arguments, not among gm.graph.nodes.
    """
    source_to_arg_idx: dict[str, int] = {}
    arg_idx = 0
    for node in gm.graph.nodes:
        if node.op != "placeholder":
            continue
        desc = node.meta.get("desc")
        if desc is not None:
            # AOT tags each placeholder with an origin description; repr() is a
            # stable key matching the output-mutation annotations.
            source_to_arg_idx[repr(desc)] = arg_idx
        arg_idx += 1

    mutation_map: dict[int, int] = {}
    for node in gm.graph.nodes:
        if node.op != "output":
            continue
        out_meta = node.meta.get("desc")
        if out_meta:
            for out_idx, entry in enumerate(out_meta):
                if not hasattr(entry, "mutated_input"):
                    continue
                mutated = entry.mutated_input
                if mutated is None:
                    continue
                src_key = repr(mutated)
                if src_key in source_to_arg_idx:
                    mutation_map[out_idx] = source_to_arg_idx[src_key]
        # auto_functionalize.unwrap stamps a {out_idx → arg_idx} sidecar dict
        # when it unwraps HOPs, since AOT's `mutated_input` annotations hide
        # inside the HOP and won't appear above.
        sidecar = node.meta.get("alloy_auto_functionalized_mutations")
        if isinstance(sidecar, dict):
            for k, v in sidecar.items():
                mutation_map[int(k)] = int(v)
        break
    return mutation_map


def _has_host_reachable_output(gm: torch.fx.GraphModule) -> bool:
    """True if the plan's tail gpu_sync is necessary.

    AOT tags each output as InputMutation / SavedForBackwards / Grad / None /
    Plain. Sync only when no downstream compiled region will implicitly sync:
      - Plain output WITHOUT any SavedForBackwards: a pure-forward exit
        (inference, or training with grad disabled) — no bwd/opt follows.
      - All InputMutation: the optimizer-like tail of a training step.
    Mixed Plain+SavedForBackwards (training fwd or loss subgraph) defers to
    whoever runs next.
    """
    for node in gm.graph.nodes:
        if node.op != "output":
            continue
        descs = node.meta.get("desc")
        if not descs:
            return True  # unknown — be safe
        has_plain = False
        has_saved = False
        has_mutation = False
        has_other = False
        for entry in descs:
            if entry is None:
                continue
            type_name = type(entry).__name__
            if type_name == "PlainAOTOutput":
                has_plain = True
            elif type_name == "SavedForBackwardsAOTOutput":
                has_saved = True
            elif type_name == "InputMutationAOTOutput":
                has_mutation = True
            elif type_name in ("GradAOTOutput", "NoneType"):
                pass
            else:
                has_other = True  # unknown — be safe
        if has_other:
            return True
        # Any Plain output may be read on the host (.item()/.cpu()/.numpy()), so
        # sync. Skipping sync on training-fwd (Plain + SavedForBackwards) breaks
        # fwd-only callers that read loss/logits without a subsequent bwd.
        if has_plain:
            return True
        # Optimizer-like tail (all mutations): the host read after the training
        # step must see finalised data.
        if has_mutation and not has_saved:
            return True
        return False
    return True


# --- FX-graph capture (dev-loop graph cache) ---------------------------------
# The ATen graph _compile_fx receives is a function of the HF model code, the
# custom-op registry, the AOT decompositions, and the generation wrapper modules
# — invariant to kernels/emitter/fusion/dispatch and the rewrites/ops handlers
# (which run inside _compile_fx). Capturing it PRE-rewrite lets _graph_cache
# replay a load without Dynamo+AOT.
_graph_capture_stack: list = []


# FX/Dynamo debug-provenance meta keys. They dominate the serialized cache
# (stack_trace alone is ~79% of a 4MB graph), so drop them before pickling.
_DEBUG_META_KEYS = frozenset(
    ("stack_trace", "from_node", "nn_module_stack", "source_fn_stack", "seq_nr")
)


def _sanitize_meta(meta: dict) -> dict:
    """Best-effort picklable copy of a node's meta dict. Tensor values (incl.
    unpicklable FakeTensors) become meta-device tensors keeping shape/dtype/
    stride; debug-provenance keys are dropped; everything else is kept iff it
    survives pickle. Preserves AOT's `desc` annotations, which
    `_extract_mutation_map` and `_has_host_reachable_output` depend on — dropping
    them breaks input-mutation handling (wrong DeltaNet state on replayed
    hybrids)."""
    out: dict = {}
    for k, v in meta.items():
        if k in _DEBUG_META_KEYS:
            continue
        if isinstance(v, torch.Tensor):
            out[k] = torch.empty_strided(v.shape, v.stride(), dtype=v.dtype, device="meta")
            continue
        try:
            pickle.dumps(v)
        except Exception:
            continue
        out[k] = v
    return out


def _sanitized_graph_copy(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Deep-copy gm's graph with every node.meta reduced to a picklable form."""
    new_graph = torch.fx.Graph()
    val_map: dict = {}
    out = new_graph.graph_copy(gm.graph, val_map)
    out_node = new_graph.output(out)
    for old_node, new_node in val_map.items():
        new_node.meta = _sanitize_meta(old_node.meta)
    old_out = next(n for n in gm.graph.nodes if n.op == "output")
    out_node.meta = _sanitize_meta(old_out.meta)
    new_gm = torch.fx.GraphModule(gm, new_graph)
    new_gm.meta.update(_sanitize_meta(gm.meta))
    return new_gm


def _compile_fx(gm: torch.fx.GraphModule, example_inputs: Sequence[Any]):
    _compile_t0 = time.perf_counter()
    if _graph_capture_stack:
        _graph_capture_stack[-1].note_graph(_sanitized_graph_copy(gm))
    # Clear caches keyed by data pointers — torch recycles addresses from freed models
    _mm_batched_cache.clear()
    _alloy_buf_map.clear()
    _metal_ext.clear_buffer_cache()
    host_reachable_output = _has_host_reachable_output(gm)
    # Graph rewrites: collapse decomposed subgraphs into single custom op nodes.
    # Runs BEFORE the unsupported-op check so rewrites can introduce ops the
    # check would reject (auto_functionalized_v2 → direct call).
    gm = rewrite_fx_graph(gm)
    # Harvest mutation metadata AFTER rewrites: auto_functionalized_v2 hides AOT's
    # `entry.mutated_input` annotations behind the HOP, so the unwrap pass stamps
    # a sidecar dict on the output node's meta. `_extract_mutation_map` merges both.
    mutation_map = _extract_mutation_map(gm)

    if os.environ.get("ALLOY_DEBUG_MUTMAP") == "1":
        _out = next(n for n in gm.graph.nodes if n.op == "output")
        _n_ph = sum(1 for n in gm.graph.nodes if n.op == "placeholder")
        print(
            f"[mutmap] n_ph={_n_ph} map={mutation_map} "
            f"out_desc={'Y' if _out.meta.get('desc') else 'N'} "
            f"sidecar={_out.meta.get('alloy_auto_functionalized_mutations')}",
            flush=True,
        )

    if os.environ.get("ALLOY_DUMP_FX") == "1":
        os.makedirs("/tmp/alloy_fx", exist_ok=True)
        _path = f"/tmp/alloy_fx/graph_{id(gm)}.txt"
        with open(_path, "w") as _f:
            _f.write(str(gm.graph))
        print(f"[alloy] dumped FX to {_path}")

    unsupported = _unsupported_targets(gm)
    if unsupported:
        ops_str = "\n  ".join(unsupported)
        logger.error("unsupported_fx_op_encountered", ops=list(unsupported))
        raise RuntimeError(
            "Alloy backend: unsupported FX op(s):\n"
            f"  {ops_str}\n"
            "Add handlers to alloy_torch.ops.registry before compiling this graph."
        )

    constant_env = _build_constant_env(gm)
    plan, n_values = _build_execution_plan(gm, constant_env)
    n_nodes = sum(1 for _ in gm.graph.nodes)
    n_inputs = sum(1 for n in gm.graph.nodes if n.op == "placeholder")
    n_outputs = sum(1 for n in gm.graph.nodes if n.op == "output")
    logger.debug(
        "fx_graph_compiled",
        graph_id=f"g{id(gm):x}",
        n_nodes=n_nodes,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        n_unique_targets=len({n.target for n in gm.graph.nodes if n.op == "call_function"}),
        took_ms=round((time.perf_counter() - _compile_t0) * 1000.0, 1),
    )
    runtime_attr_cache: dict[str, Any] = {}
    _run_count: int = 0
    # Graph compiler state: [None] → not built, [CompiledPlan] → ready
    _compiled_plan: CompiledPlan | None = None
    _uses_dropout = any(
        n.op == "call_function" and n.target is torch.ops.aten.native_dropout.default
        for n in gm.graph.nodes
    )

    def compiled(*args):
        nonlocal _run_count
        nonlocal _compiled_plan

        # Dropout's keep mask is a per-forward draw from the torch generator;
        # refresh the shared seed each call so masks vary per step and stay under
        # torch.manual_seed.
        if _uses_dropout and is_training_mode_enabled():
            refresh_dropout_seed()

        # Record (plan, args) into the active capture slot, if any. Args are
        # captured on EVERY call so the slot always holds a valid args tuple even
        # when an AOT specialisation recompiles. The plan is recorded on both the
        # replay (below) and compile (OUTPUT) paths.
        _slot = _capture_stack[-1] if _capture_stack else None
        if _slot is not None:
            _slot.args = args

        _engine.clear_run()

        if _compiled_plan is not None:
            if _slot is not None:
                _slot.plan = _compiled_plan
            # Release handler-path AlloyBuffers deferred from plan compilation.
            # Safe here: the plan path is active, handler-path data is unreferenced.
            cleanup = _compiled_plan.pending_buf_cleanup
            if cleanup is not None:
                for arr in cleanup:
                    try:
                        _metal_ext.buf_release(arr._parent_handle)
                    except Exception:
                        pass
                _compiled_plan.pending_buf_cleanup = None
            return _execute_plan(_compiled_plan, args)

        # --- Full path: build lazy graph via handlers ---
        values = [None] * n_values
        arg_idx = 0

        # Track which input args map to which pointers
        input_ptrs: dict[int, InputPtrInfo] = {}

        plan_recording = _run_count == 0
        if plan_recording:
            _engine.start_recording()

        for opcode, vi, slot in plan:
            if opcode == Opcode.CONST:
                values[vi] = slot
            elif opcode == Opcode.INPUT:
                values[vi] = _to_lazy_input(args[arg_idx])
                v = values[vi]
                if isinstance(v, AlloyBuffer):
                    # Key by base_ptr so sliced-view inputs are recognised when
                    # recorded dispatches reference the parent storage; keep
                    # data_ptr too for output-aliasing.
                    bp = v.base_ptr
                    input_ptrs[bp] = InputPtrInfo(arg_idx, v._offset)
                    dp = v.data_ptr
                    if dp != bp and dp not in input_ptrs:
                        input_ptrs[dp] = InputPtrInfo(arg_idx, v._offset)
                arg_idx += 1
            elif opcode == Opcode.ATTR:
                if is_training_mode_enabled():
                    values[vi] = _to_lazy_input(_get_nested_attr(gm, slot))
                else:
                    cached = runtime_attr_cache.get(slot)
                    if cached is None:
                        cached = _prepack_constant(_to_lazy_input(_get_nested_attr(gm, slot)))
                        runtime_attr_cache[slot] = cached
                    values[vi] = cached
            elif opcode == Opcode.CALL:
                handler, arg_slots, kwarg_slots = slot
                call_args = tuple(_resolve_from_values(s, values) for s in arg_slots)
                call_kwargs = {k: _resolve_from_values(v, values) for k, v in kwarg_slots.items()}
                result = handler(*call_args, **call_kwargs)
                values[vi] = result
            elif opcode == Opcode.GETITEM:
                src_idx, item_idx = slot
                src_val = values[src_idx]
                if src_val is None:
                    src_op = plan[src_idx] if src_idx < len(plan) else None
                    info = ""
                    if src_op and src_op[0] == Opcode.CALL:
                        h = src_op[2][0]
                        info = h.__name__ if hasattr(h, "__name__") else repr(h)
                    raise TypeError(
                        f"getitem[{item_idx}] on None at vi={vi}, src={src_idx}, handler={info}"
                    )
                values[vi] = src_val[item_idx]
            elif opcode == Opcode.OUTPUT:
                out = tuple(_resolve_from_values(s, values) for s in slot)
                out_value = out[0] if len(out) == 1 else out
                # KV-cache writes nothing in-graph reads (sliding-window cold
                # prefill attends a linear temp copy) are unreachable from the
                # outputs — materialize the registered side-effect roots so those
                # writes land in the recorded plan instead of being DCE'd.
                extern_writes = drain_extern_kv_writes()
                if extern_writes:
                    materialize_many(extern_writes)
                if is_record_only():
                    # Record-only compile: flush the lazy graph so the dispatch
                    # recording is populated (no GPU, phantom intermediates), but
                    # never read the phantom output — return shaped dummies.
                    materialize_many(out_value)
                    result = _dummy_torch_output(out_value)
                else:
                    result = _to_torch_output(out_value)

                _run_count += 1

                if plan_recording:
                    recording = _engine.stop_recording()
                    if recording is not None:
                        recorded, buf_map = recording
                        cp = _compile_to_plan(
                            recorded,
                            input_ptrs,
                            out,
                            alloc_ptrs=set(_engine.alloc_ptrs),
                            buf_map=buf_map,
                            mutation_map=mutation_map,
                            host_reachable_output=host_reachable_output,
                        )
                        _compiled_plan = cp
                        if _slot is not None:
                            _slot.plan = cp
                        _all_compiled_plans.append(cp)
                return result

        raise RuntimeError("Alloy backend: graph terminated without an output node")

    # FX-graph capture: record the first real call's flat args + flat outputs so
    # _graph_cache can serialize an input spec (params/buffers vs caller kwargs vs
    # lifted constants) and the user-output index. Inert once the scope pops.
    final_fn = compiled
    if _graph_capture_stack:
        _cap = _graph_capture_stack[-1]
        _gidx = len(_cap.graphs) - 1

        def _capturing(*args, _inner=compiled, _cap=_cap, _gidx=_gidx):
            out = _inner(*args)
            if _graph_capture_stack and _graph_capture_stack[-1] is _cap:
                _cap.note_run(_gidx, args, out)
            return out

        final_fn = _capturing

    # Mark the compiled function as "boxed" so AOT Autograd's runtime wrapper
    # runs its input-mutation copyback against our outputs — else mutations the
    # graph writes (optimiser state) never propagate back to the input tensors.
    boxed_fn = make_boxed_func(final_fn)
    # The FX-graph cache calls this boxed fn RAW (no AOT runtime wrapper), so it
    # must apply the input-mutation copyback itself — expose the post-rewrite map.
    boxed_fn._alloy_mutation_map = mutation_map
    return boxed_fn


# ---------------------------------------------------------------------------
# Graph compiler: compiled dispatch plan
# ---------------------------------------------------------------------------

# C++ slot type codes (must match alloy_metal.mm register_plan)
_SLOT_INPUT = 0
_SLOT_WEIGHT = 1
_SLOT_INTERMEDIATE = 2


@dataclass
class InputSlot:
    """Dynamic input buffer, re-bound each run."""

    arg_idx: int
    nbytes: int
    root_ptr: int
    view_offset: int


@dataclass
class WeightSlot:
    """Stable model weight or constant."""

    nbytes: int
    root_ptr: int


@dataclass
class IntermediateSlot:
    """Alloy-allocated temporary, recycled via liveness analysis."""

    nbytes: int
    root_ptr: int
    physical_idx: int = -1  # set by _liveness_analysis


PlanSlot = InputSlot | WeightSlot | IntermediateSlot


@dataclass
class PlanDispatch:
    """A single kernel dispatch in the compiled plan."""

    pso_handle: int
    debug_name: str
    buf_slot_indices: list[int]
    buf_offsets: list[int]
    buf_identity_offsets: tuple[int, ...]
    grid: tuple[int, int, int]
    tg: tuple[int, int, int]
    write_slot_indices: set[int] = field(default_factory=set)
    # MSL source + entry-point name for plan-cache serialisation. Empty when the
    # pso_handle wasn't routed through CompiledKernel.from_msl (non-cacheable).
    msl_source: str = ""
    function_name: str = ""
    # (extent, byte_stride) axes of every buffer this dispatch WRITES. The
    # grid-shrink recipe uses this to refuse dispatches whose written M axis is
    # not outermost (their shrunk threadgroup prefix would cover wrong elements).
    write_dims: tuple[tuple[tuple[int, int], ...], ...] = ()


@dataclass(frozen=True, slots=True)
class InputUpdateCacheCheck:
    """Identity of one runtime input arg for cached input_updates reuse."""

    arg_idx: int
    storage_ptr: int
    data_ptr: int


# --- Output mapping types ---
# Each FX output maps to one of these. Used by _execute_plan to reconstruct
# torch tensors from plan buffer slots after GPU dispatch.


@dataclass(frozen=True, slots=True)
class OutputNone:
    """Output that produces None."""

    pass


@dataclass(frozen=True, slots=True)
class OutputConst:
    """Output for a zero/empty constant tensor."""

    shape: tuple[int, ...]
    dtype: DType


@dataclass(frozen=True, slots=True)
class OutputPassthrough:
    """Output that passes an input arg through (e.g. weight saved for backward)."""

    arg_idx: int
    shape: tuple[int, ...]
    dtype: DType
    byte_offset: int


@dataclass(frozen=True, slots=True)
class OutputSlot:
    """Output backed by a plan buffer slot."""

    slot_idx: int
    shape: tuple[int, ...]
    dtype: DType
    byte_offset: int
    strides_bytes: tuple[int, ...] | None


OutputEntry = OutputNone | OutputConst | OutputPassthrough | OutputSlot


@dataclass
class CompiledPlan:
    """Fully-initialized dispatch plan ready for C++ execution."""

    dispatches: list[PlanDispatch]
    slots: list[InputSlot | WeightSlot | IntermediateSlot]
    output_mapping: list[OutputEntry]
    physical_bufs: list[int]
    dep_groups: list[list[int]]
    phys_arrays: list[AlloyBuffer]
    weight_bindings: dict[int, AlloyBuffer]
    plan_handle: int
    pending_buf_cleanup: list[AlloyBuffer] | None = None
    # True if any output is a user/host-reachable tensor (PlainAOTOutput). When
    # False, _execute_plan defers the tail gpu_sync — nothing between this plan
    # and the next is read on the CPU, and Metal enforces queue order.
    host_reachable_output: bool = True
    # Mutable runtime state — grows across _execute_plan calls
    _storage_handles: dict[int, int] = field(default_factory=dict)
    _align_refs: list[tuple[Any, int]] = field(default_factory=list)
    # Cached input_updates: skip O(n_slots) scan+build on steady-state calls
    _cached_input_updates: list[tuple[int, int, int]] | None = None
    _cached_input_check: tuple[InputUpdateCacheCheck, ...] | None = None
    # AOT mutation outputs that correspond to runtime InputSlots. Some can't be
    # remapped in-place (the original input is read later in the plan), but decode
    # fast paths still need to know which input scalar advances.
    mutation_input_slots: dict[int, int] = field(default_factory=dict)
    # Grid-shrink recipe: {flat_dispatch_idx: [(axis, ext_max), ...]}. Identifies
    # which dispatch grid axes scale linearly with the prompt length M, recording
    # each axis's max-length extent so the shrink dispatch launches
    # ceil(M_pad * ext_max / M_MAX) threadgroups against this max-length-compiled
    # plan instead of the full grid. Covers block-tiled axes (ext = ceil(M/block))
    # and 1D-flattened elementwise grids (ext = M * cols). None until the
    # generator's two-point grid discovery fills it in. _grid_shrink_m_max is the
    # M_MAX the plan compiled at.
    _grid_shrink_recipe: dict[int, list[tuple[int, int, int]]] | None = None
    _grid_shrink_m_max: int = 0
    # Request-bounded intermediate pool (shrink-capable plans only). Metal wires
    # FULL buffer residency at first encoder use, so an M_MAX-sized pool wires ~its
    # whole VA on the first call regardless of the shrunk grids (+54GB on
    # qwen3.5:0.8b at native, +0 on the second call). M-outer pool buffers are
    # allocated at a high-water prompt bound and grown on demand (_grow_plan_pool):
    # _pool_trunc maps phys_idx -> bytes per M row, allocation is
    # aligned(per_row * _pool_bound). 0/None = full-size pool.
    _pool_bound: int = 0
    _pool_m_max: int = 0
    _pool_trunc: dict[int, int] | None = None
    # Grid overrides applied on the most recent grid-shrunk dispatch
    # (flat_dispatch_idx, gx, gy, gz). Stashed by `PrefillEngine.chunk_step` so
    # `alloy.visualize` profiles at the SAME shrunk grid the run dispatched.
    # None for non-shrink plans → profiling uses the registered grid.
    _last_grid_shrink_updates: list[tuple[int, int, int, int]] | None = None


# --- Request-bounded grid-shrink intermediate pool ---------------------------
# Metal wires FULL buffer residency at first encoder use, so an M_MAX-compiled
# shrink-capable plan's pool wires ~its entire VA on the first call no matter how
# far the launch grids were shrunk (+54GB on qwen3.5:0.8b at native; qwen3.6:35b's
# 76GB pool cannot fit 128GB at all). Pool buffers whose every access is
# M-OUTERMOST row-major (offset for m rows bounded by m * row_bytes from the
# buffer start — the same invariant that makes grid-shrink correct) are allocated
# at a high-water prompt bound and grown on demand for a longer prompt (rare,
# monotone).
#
# M-outer-ness is decided per SLOT from the kernels touching it, deny-first: a
# kernel with a known non-M-outer access pattern (head-major rope/attention tiles,
# the heads-outer DeltaNet GDR scratch, strided copies/transposes) pins every slot
# it touches; otherwise the kernel must be in the M-OUTER allowlist (GEMM/GEMV
# (M, N) outputs, 2D elementwise over (M, cols), the (B,S,C) conv pipeline,
# (B*S, heads) norms, the MoE grouped pipeline). Unknown pins — worst case a
# full-size buffer, never corruption. Hard gates: the slot must not back a plan
# OUTPUT, its nbytes must be an exact multiple of M_MAX (pure-linear M scaling),
# and every static dispatch offset must sit in the first 64 rows.
_POOL_M_OUTER_PREFIXES = (
    "dot",
    "mul",
    "sigmoid_mul",
    "silu_inplace",
    "k_add_scalar_rms_norm",
    "rms_norm",
    "l2norm_last_dim",
    "embedding_",
    "causal_conv1d_with_state_prefill",
    "conv_state_save_real_pos",
    "moe_",
)


def _phys_aligned_size(nbytes: int) -> int:
    aligned = ((nbytes + 16383) // 16384) * 16384
    return aligned if aligned > 0 else 16384


def _pool_access_m_outer(d: PlanDispatch, writes: bool, m_max: int) -> bool:
    """Is this dispatch's access to a slot M-OUTERMOST (offset bounded by
    rows*row_bytes from the buffer start)? Direction-aware where a kernel mixes
    layouts: the rope family READS its (M, hidden) input M-outer but WRITES
    head-major (heads*M, D) tiles; single-pass attention reads those head-major
    tiles but writes its O M-outer (B*S*H, D). M-outer strided copies are
    2D-gridded (rows on axis 0 == M); 1D/flat copies can be M-inner transposes and
    pin. The DeltaNet chunked-GDR scratch is heads-outer — always pinned. Unknown
    kernels pin."""
    name = d.debug_name
    if "transpose" in name:
        return False
    if "gdr_stage1" in name:
        # stage1 READS the l2norm'd q/k and gate buffers M-outer ((B*S*heads, D));
        # its W/T/at/qg/kd scratch WRITES are heads-outer.
        return not writes
    if "gdr_stage2" in name:
        # stage2's core_attn WRITE is M-outer ((B*S*NV, DV)); its scratch READS
        # target slots already pinned by stage1's writes.
        return True
    if "gdr" in name:
        return False
    if "strided_copy" in name:
        # M-outer contiguify copies are 2D-gridded with rows == B*S == M on axis 0;
        # flat/transpose copies have grid[0] = ceil(N/block) != m_max.
        return d.grid[0] == m_max
    if "rope" in name:
        return not writes
    if "attention_strided" in name:
        return writes
    return name.startswith(_POOL_M_OUTER_PREFIXES)


def _classify_truncatable_slots(
    slots: "list[InputSlot | WeightSlot | IntermediateSlot]",
    dispatches: list[PlanDispatch],
    output_mapping: "list[OutputEntry]",
    m_max: int,
) -> dict[int, int]:
    """Map truncatable INTERMEDIATE slot indices to their bytes-per-M-row (the
    M-outer gates — see the module comment above). Runs BEFORE liveness so the
    allocator keeps truncatable and pinned slots in separate pools (a shared phys
    buffer would be pinned by its most conservative slot)."""
    output_slot_set = {e.slot_idx for e in output_mapping if isinstance(e, OutputSlot)}
    slot_ok: dict[int, bool] = {}
    max_offset: dict[int, int] = {}
    written: set[int] = set()
    for d in dispatches:
        written.update(d.write_slot_indices)
        for si, off in zip(d.buf_slot_indices, d.buf_offsets):
            w = si in d.write_slot_indices
            ok = _pool_access_m_outer(d, w, m_max)
            slot_ok[si] = slot_ok.get(si, True) and ok
            if off > max_offset.get(si, 0):
                max_offset[si] = off

    trunc: dict[int, int] = {}
    for si, slot in enumerate(slots):
        if not isinstance(slot, IntermediateSlot):
            continue
        if (
            si not in output_slot_set
            and si in written
            and slot.nbytes % m_max == 0
            and slot.nbytes // m_max > 0
            and slot_ok.get(si, False)
            and max_offset.get(si, 0) < 64 * (slot.nbytes // m_max)
        ):
            trunc[si] = slot.nbytes // m_max
    return trunc


def release_plan_intermediates(plan: "CompiledPlan") -> None:
    """Free a discarded plan's intermediate pool (its `phys_arrays`). Alloy has
    no Metal-buffer GC, so a plan compiled then thrown away during warmup (the
    grid-shrink probe) otherwise leaks its full M-bounded pool. Weight bindings
    are SHARED with the pinned plans and never touched."""
    seen: set[int] = set()
    for arr in plan.phys_arrays:
        h = arr._parent_handle
        if h >= 0 and h not in seen:
            seen.add(h)
            _metal_ext.buf_release(h)
    plan.phys_arrays = []


def _grow_plan_pool(plan: "CompiledPlan", new_bound: int) -> None:
    """Grow a request-bounded plan pool to `new_bound` M rows: reallocate every
    truncated physical buffer at the new bound and rebind its slots in C++.
    Monotone high-water; no-op if the plan already covers `new_bound`."""
    if not plan._pool_trunc:
        return
    new_bound = min(int(new_bound), plan._pool_m_max)
    if new_bound <= plan._pool_bound:
        return
    updates: list[tuple[int, int, int]] = []
    for phys_idx, per_row in plan._pool_trunc.items():
        arr = _alloc_aligned((_phys_aligned_size(per_row * new_bound),), uint8)
        ptr = arr.base_ptr
        _alloy_handle_map.pop(ptr, None)
        _alloy_buf_map.pop(ptr, None)
        DispatchEngine.default().untrack_alloc(ptr)
        # Preserve old contents (truncatable slots are dispatch-written and
        # recomputed every call, but copying the prefix is cheap).
        old_arr = plan.phys_arrays[phys_idx]
        copied = min(old_arr.metal_nbytes, arr.metal_nbytes)
        ctypes.memmove(arr.base_ptr, old_arr.base_ptr, copied)
        # Zero the fresh tail: Metal buffer contents start UNDEFINED, and pad rows
        # beyond the request's m_pad are read by masked paths assuming finite
        # values (0 * NaN = NaN).
        if arr.metal_nbytes > copied:
            ctypes.memset(arr.base_ptr + copied, 0, arr.metal_nbytes - copied)
        # Replace the Python-side owner; the old AlloyBuffer frees on GC after the
        # C++ rebind below stops referencing its Metal buffer.
        plan.phys_arrays[phys_idx] = arr
        for si, s in enumerate(plan.slots):
            if isinstance(s, IntermediateSlot) and s.physical_idx == phys_idx:
                updates.append((si, arr._parent_handle, arr.metal_nbytes))
    _metal_ext.update_plan_intermediate_slots(plan.plan_handle, updates)
    old = plan._pool_bound
    plan._pool_bound = new_bound
    logger.info(
        "grid_shrink_pool_grown",
        old_bound=old,
        new_bound=new_bound,
        n_phys=len(plan._pool_trunc),
        n_slots=len(updates),
    )


def _compile_to_plan(
    recorded_dispatches: list[RecordedDispatch],
    input_ptrs: dict[int, InputPtrInfo],
    output_values,
    alloc_ptrs: set[int],
    buf_map: dict[int, AlloyBuffer],
    mutation_map: dict[int, int] | None = None,
    host_reachable_output: bool = True,
) -> CompiledPlan:
    """Build and finalize a CompiledPlan from recorded run-0 dispatches.

    Returns a fully-initialized plan registered with C++ and ready for dispatch.
    """
    if not recorded_dispatches:
        raise RuntimeError("No recorded dispatches for compiled plan")

    # --- Phase 1: Build slots and dispatches ---

    slots: list[InputSlot | WeightSlot | IntermediateSlot] = []
    dispatches: list[PlanDispatch] = []
    ptr_to_slot: dict[int, int] = {}  # root_ptr OR handle → slot_index

    def _get_or_create_slot(root_ptr: int, nbytes: int):
        if root_ptr in ptr_to_slot:
            idx = ptr_to_slot[root_ptr]
            if nbytes > slots[idx].nbytes:
                slots[idx].nbytes = nbytes
            return idx
        idx = len(slots)
        ptr_to_slot[root_ptr] = idx
        binding = buf_map.get(root_ptr)
        if binding is not None and binding._parent_handle >= 0:
            ptr_to_slot[binding._parent_handle] = idx

        if root_ptr in input_ptrs:
            info = input_ptrs[root_ptr]
            slots.append(
                InputSlot(
                    arg_idx=info.arg_idx,
                    nbytes=nbytes,
                    root_ptr=root_ptr,
                    view_offset=info.view_offset,
                )
            )
        elif root_ptr in alloc_ptrs:
            slots.append(IntermediateSlot(nbytes=nbytes, root_ptr=root_ptr))
        else:
            slots.append(WeightSlot(nbytes=nbytes, root_ptr=root_ptr))
        return idx

    for entry in recorded_dispatches:
        slot_indices = []
        offsets = []
        identity_offsets: list[int] = []
        write_slots = set()
        write_dims: list[tuple[tuple[int, int], ...]] = []
        for bi, binding in enumerate(entry.buffers):
            root_ptr = binding.root_ptr
            offset = binding.byte_offset
            nbytes = binding.nbytes
            slot_idx = _get_or_create_slot(root_ptr, nbytes)
            slot = slots[slot_idx]
            slot_indices.append(slot_idx)
            identity_offsets.append(offset)
            # Zero the dispatch-level offset for INPUT slots: dispatch_plan
            # already sets ``slot.base_offset = a.data_ptr() - storage_ptr``
            # per-call. Keep any additional offset introduced inside the graph
            # itself (for example x[4:8] when x is the runtime argument).
            if isinstance(slot, InputSlot):
                offsets.append(offset - slot.view_offset)
            else:
                offsets.append(offset)
            if bi in entry.write_indices:
                write_slots.add(slot_idx)
                write_dims.append(binding.dims)
        dispatches.append(
            PlanDispatch(
                entry.pso_handle,
                entry.debug_name,
                slot_indices,
                offsets,
                tuple(identity_offsets),
                entry.grid,
                entry.threadgroup,
                write_slots,
                msl_source=entry.msl_source,
                function_name=entry.function_name,
                write_dims=tuple(write_dims),
            )
        )

    # Slots written by dispatches MUST be INTERMEDIATE, not WEIGHT. Fused kernels
    # can produce outputs at pointers not tracked by alloc_ptrs, misclassifying
    # them as WEIGHT. (The spec-verify conv tape bank dodges this deliberately:
    # its kernel param carries no alloy.output annotation, so it never enters a
    # write set and keeps the stable WEIGHT binding.)
    written_by_any = set()
    for d in dispatches:
        written_by_any.update(d.write_slot_indices)
    for si in written_by_any:
        slot = slots[si]
        if isinstance(slot, WeightSlot):
            slots[si] = IntermediateSlot(nbytes=slot.nbytes, root_ptr=slot.root_ptr)

    # --- Phase 2: Dead dispatch elimination ---

    def _is_constant_slot(si):
        return isinstance(slots[si], WeightSlot) and si not in written_by_any

    n_before = len(dispatches)
    changed = True
    # Track slots that had a dispatch writing them on run 0 but got
    # elided because all reads were constants. Their handler-path buffer
    # already holds the correct result (run 0 dispatched them), so rebind
    # those slots as WEIGHTs to carry the baked value instead of an
    # uninitialised intermediate.
    folded_to_weight: set[int] = set()
    # Record-only compile never executes the GPU, so a constant-read dispatch's
    # output was NOT computed — folding it to a baked WeightSlot would carry
    # garbage. Keep those dispatches live; run-1 executes them and writes the real
    # value.
    _allow_const_fold = not is_record_only()
    while changed:
        changed = False
        surviving = []
        for d in dispatches:
            read_slots = set(d.buf_slot_indices) - d.write_slot_indices
            if (
                _allow_const_fold
                and read_slots
                and all(_is_constant_slot(si) for si in read_slots)
                and d.write_slot_indices
            ):
                written_by_any -= d.write_slot_indices
                folded_to_weight.update(d.write_slot_indices)
                changed = True
            elif not read_slots and not d.write_slot_indices:
                changed = True
            else:
                surviving.append(d)
        dispatches = surviving
    # Reclassify folded-output slots as WeightSlot pointing at the run-0 buffer so
    # register_plan captures its metal handle — else `weight_bindings` lookup omits
    # them and the consumer reads an uninitialised phys_array.
    for si in folded_to_weight:
        slot = slots[si]
        if isinstance(slot, IntermediateSlot):
            slots[si] = WeightSlot(nbytes=slot.nbytes, root_ptr=slot.root_ptr)
    n_elim = n_before - len(dispatches)
    if n_elim:
        logger.debug(f"PLAN: eliminated {n_elim} constant dispatches")

    # --- Phase 3: CSE ---
    _MAX_CSE_CONTENT_BYTES = 64
    _cse_first: dict[tuple, PlanDispatch] = {}
    _slot_alias: dict[int, int] = {}
    redundant: set[int] = set()

    def _slot_key(
        si: int,
    ) -> tuple[Literal["c"], bytes] | tuple[Literal["s"], int] | tuple[Literal["w"], int, int]:
        s = slots[si]
        if isinstance(s, WeightSlot) and si not in written_by_any:
            if s.nbytes <= _MAX_CSE_CONTENT_BYTES and buf_map:
                b = buf_map.get(s.root_ptr)
                if b is not None:
                    try:
                        return ("c", b.numpy.tobytes())
                    except Exception:
                        pass
            return ("w", s.root_ptr, s.nbytes)
        return ("s", si)

    for di, d in enumerate(dispatches):
        # Apply aliases accumulated from earlier dedup decisions — else two
        # dispatches that match only AFTER an input gets aliased key differently
        # and miss the fold. E.g. each block runs `cast_f32(causal_mask) →
        # where(...)`; the casts dedup to 1, but the wheres each reference the
        # pre-alias cast output and all survive.
        write_slots = {_slot_alias.get(si, si) for si in d.write_slot_indices}
        read_parts = sorted(
            (
                _slot_alias.get(si, si),
                identity_offset,
            )
            for si, identity_offset in zip(d.buf_slot_indices, d.buf_identity_offsets)
            if _slot_alias.get(si, si) not in write_slots
        )
        if not read_parts or not d.write_slot_indices:
            continue
        # Never CSE a dispatch that writes an INPUT slot: that is an in-place
        # mutation of a persistent buffer (a KV-cache write) that MUST run on its
        # own buffer. The CSE key is (pso, read slots) only — it ignores write
        # slots, which breaks for cache writes: gemma4's cross-layer KV sharing has
        # many shared layers write the SAME reused K/V into DIFFERENT per-layer
        # caches, so the writes share a pso + read slots yet aren't redundant.
        # Merging them aliases distinct cache buffers and drops all-but-one write,
        # leaving those layers' caches unwritten (decode attends zeros).
        if any(
            isinstance(slots[_slot_alias.get(si, si)], InputSlot)
            for si in d.write_slot_indices
        ):
            continue
        key_parts = [(_slot_key(si), identity_offset) for si, identity_offset in read_parts]
        key = (d.pso_handle, tuple(key_parts))
        if key in _cse_first:
            first_d = _cse_first[key]
            fw = sorted(first_d.write_slot_indices)
            cw = sorted(d.write_slot_indices)
            if len(fw) == len(cw):
                for f, c in zip(fw, cw):
                    _slot_alias[c] = f
                redundant.add(di)
        else:
            _cse_first[key] = d

    if _slot_alias:
        for d in dispatches:
            d.buf_slot_indices = [_slot_alias.get(si, si) for si in d.buf_slot_indices]
            d.write_slot_indices = {_slot_alias.get(si, si) for si in d.write_slot_indices}
        n_before2 = len(dispatches)
        dispatches = [d for di, d in enumerate(dispatches) if di not in redundant]
        n_cse = n_before2 - len(dispatches)
        if n_cse:
            logger.debug(f"PLAN: CSE eliminated {n_cse} redundant dispatches")

    # --- Phase 4: Output mapping, liveness, dependency groups ---
    output_mapping = _build_output_mapping(output_values, ptr_to_slot, input_ptrs)

    # Mutation remap: if AOT tagged output i as an in-place mutation of input arg
    # j, route the dispatch computing it straight to arg j's storage. This turns
    # AOT's `input.copy_(output)` into a no-op and removes the final
    # Intermediate → Input memmove on CPU.
    mutation_input_slots: dict[int, int] = {}
    if mutation_map:
        mutation_input_slots = _apply_mutation_remap(
            output_mapping, dispatches, slots, input_ptrs, mutation_map
        )

    # Request-bounded shrink pool: classify M-outer slots BEFORE liveness so
    # the allocator keeps them in pools separate from pinned slots (see
    # _classify_truncatable_slots).
    pool_bound = 0
    pool_m_max = compile_window.shrink_m
    trunc_slots: dict[int, int] = {}
    if pool_m_max:
        pool_bound = min(pool_m_max, GRID_SHRINK_REP_M)
        trunc_slots = _classify_truncatable_slots(
            slots, dispatches, output_mapping, pool_m_max
        )

    physical_bufs = _liveness_analysis(
        slots, dispatches, output_mapping, trunc_slots=trunc_slots
    )
    dep_groups = _plan_dependency_groups(slots, dispatches)

    # --- Phase 5: Allocate physical buffers and resolve weights ---

    pool_trunc: dict[int, int] = {}
    for si, per_row in trunc_slots.items():
        phys = slots[si].physical_idx
        if phys is not None and per_row > pool_trunc.get(phys, 0):
            pool_trunc[phys] = per_row

    phys_arrays: list[AlloyBuffer] = []
    for phys_idx, asize in enumerate(physical_bufs):
        per_row = pool_trunc.get(phys_idx)
        alloc = _phys_aligned_size(per_row * pool_bound) if per_row else asize
        arr = _alloc_aligned((alloc,), uint8)
        ptr = arr.base_ptr
        _alloy_handle_map.pop(ptr, None)
        _alloy_buf_map.pop(ptr, None)
        DispatchEngine.default().untrack_alloc(ptr)
        # Metal buffer contents start UNDEFINED. Shrunk grids never write pad rows,
        # and masked paths read them assuming finite values (0 * NaN = NaN) — zero
        # the pool once so no request observes uninitialized memory.
        ctypes.memset(ptr, 0, arr.metal_nbytes)
        phys_arrays.append(arr)
    if pool_trunc:
        full = sum(physical_bufs)
        bounded = sum(a.metal_nbytes for a in phys_arrays)
        logger.info(
            "grid_shrink_pool_bounded",
            m_max=pool_m_max,
            bound=pool_bound,
            n_truncated=len(pool_trunc),
            n_phys=len(physical_bufs),
            full_gb=round(full / 1e9, 2),
            bounded_gb=round(bounded / 1e9, 2),
        )

    weight_bindings: dict[int, AlloyBuffer] = {}
    for si, slot in enumerate(slots):
        if isinstance(slot, WeightSlot):
            arr = buf_map.get(slot.root_ptr)
            if arr is not None:
                weight_bindings[si] = arr
            else:
                raise RuntimeError(
                    f"Weight slot {si} (ptr={slot.root_ptr:#x}) not found in buf_map"
                )

    # --- Phase 6: Register plan in C++ ---

    dispatches_data = [
        (d.pso_handle, list(d.buf_slot_indices), list(d.buf_offsets), d.grid, d.tg)
        for d in dispatches
    ]

    slots_data = []
    for si, slot in enumerate(slots):
        if isinstance(slot, InputSlot):
            slots_data.append((_SLOT_INPUT, slot.arg_idx, 0, slot.nbytes))
        elif isinstance(slot, WeightSlot):
            arr = weight_bindings[si]
            slots_data.append((_SLOT_WEIGHT, -1, arr._parent_handle, arr.metal_nbytes))
        else:
            arr = phys_arrays[slot.physical_idx]
            slots_data.append((_SLOT_INTERMEDIATE, -1, arr._parent_handle, arr.metal_nbytes))

    written_slot_indices = sorted({si for d in dispatches for si in d.write_slot_indices})
    plan_handle = DispatchEngine.default().register_plan(
        dispatches_data, slots_data, dep_groups, written_slot_indices
    )
    n_weight = sum(1 for s in slots_data if s[0] == _SLOT_WEIGHT)
    n_input = sum(1 for s in slots_data if s[0] == _SLOT_INPUT)
    n_intermediate = sum(1 for s in slots_data if s[0] == _SLOT_INTERMEDIATE)
    weight_bytes = sum(s[3] for s in slots_data if s[0] == _SLOT_WEIGHT)
    logger.debug(
        "plan_built",
        plan_handle=plan_handle,
        n_dispatches=len(dispatches_data),
        n_slots=len(slots_data),
        n_weight_slots=n_weight,
        n_input_slots=n_input,
        n_intermediate_slots=n_intermediate,
        n_dep_groups=len(dep_groups),
        weight_bytes=weight_bytes,
    )

    # --- Phase 7: Training cleanup bookkeeping ---

    pending_buf_cleanup = None
    if is_training_mode_enabled():
        kept_ptrs = {arr.base_ptr for arr in phys_arrays}
        kept_ptrs.update(arr.base_ptr for arr in weight_bindings.values())
        pending = [
            arr
            for ptr, arr in buf_map.items()
            if ptr not in kept_ptrs and arr._parent_handle >= 0
        ]
        if pending:
            pending_buf_cleanup = pending

    return CompiledPlan(
        dispatches=dispatches,
        slots=slots,
        output_mapping=output_mapping,
        physical_bufs=physical_bufs,
        dep_groups=dep_groups,
        phys_arrays=phys_arrays,
        weight_bindings=weight_bindings,
        plan_handle=plan_handle,
        pending_buf_cleanup=pending_buf_cleanup,
        host_reachable_output=host_reachable_output,
        mutation_input_slots=mutation_input_slots,
        _pool_bound=pool_bound if pool_trunc else 0,
        _pool_m_max=pool_m_max if pool_trunc else 0,
        _pool_trunc=pool_trunc or None,
    )


def _build_output_mapping(
    output_values,
    ptr_to_slot: dict[int, int],
    input_ptrs: dict[int, InputPtrInfo] | None = None,
) -> list[OutputEntry]:
    """Map FX output values to typed OutputEntry descriptors."""
    flat = output_values if isinstance(output_values, (tuple, list)) else [output_values]
    mapping: list[OutputEntry] = []
    for v in flat:
        mapping.extend(_classify_plan_output(v, ptr_to_slot, input_ptrs))
    return mapping


def _apply_mutation_remap(
    output_mapping: list[OutputEntry],
    dispatches: list[PlanDispatch],
    slots: list[InputSlot | WeightSlot | IntermediateSlot],
    input_ptrs: dict[int, InputPtrInfo],
    mutation_map: dict[int, int],
) -> dict[int, int]:
    """Route mutation-output intermediates through the corresponding InputSlots.

    For each (output_idx → input_arg_idx) entry, find the intermediate slot
    that the output maps to, find (or synthesize) the InputSlot for that
    arg, and rewrite every dispatch's buf_slot_indices + write_slot_indices
    to reference the InputSlot instead. The OutputEntry is rewritten in
    place so _execute_plan returns the input tensor unchanged.
    """
    # Index InputSlots by arg_idx (pick the one whose offset aligns — in
    # practice there's exactly one InputSlot per arg storage).
    arg_to_slot: dict[int, int] = {}
    for si, slot in enumerate(slots):
        if isinstance(slot, InputSlot):
            arg_to_slot.setdefault(slot.arg_idx, si)

    # Plans produce compact OutputSlot entries only after _classify_plan_output
    # has flattened things; use its OutputSlot→output_idx ordering.
    remap: dict[int, int] = {}  # intermediate_slot → input_slot
    mutation_input_slots: dict[int, int] = {}
    remapped_outputs: dict[int, list[tuple[int, OutputSlot]]] = {}
    for out_idx, arg_idx in mutation_map.items():
        if out_idx >= len(output_mapping):
            continue
        entry = output_mapping[out_idx]
        if not isinstance(entry, OutputSlot):
            continue
        old_slot_idx = entry.slot_idx
        if not isinstance(slots[old_slot_idx], IntermediateSlot):
            continue
        new_slot_idx = arg_to_slot.get(arg_idx)
        if new_slot_idx is None:
            # No dispatch uses the input arg as a buffer — create one to retarget
            # the write. The mutated-input nbytes matches the intermediate's.
            inter = slots[old_slot_idx]
            root_ptr = None
            input_info = None
            for ptr, info in input_ptrs.items():
                if info.arg_idx == arg_idx:
                    root_ptr = ptr
                    input_info = info
                    break
            if root_ptr is None or input_info is None:
                continue
            new_slot_idx = len(slots)
            slots.append(
                InputSlot(
                    arg_idx=arg_idx,
                    nbytes=inter.nbytes,
                    root_ptr=root_ptr,
                    view_offset=input_info.view_offset,
                )
            )
            arg_to_slot[arg_idx] = new_slot_idx
        mutation_input_slots[out_idx] = new_slot_idx
        remap[old_slot_idx] = new_slot_idx
        remapped_outputs.setdefault(old_slot_idx, []).append((out_idx, entry))
        output_mapping[out_idx] = OutputSlot(
            slot_idx=new_slot_idx,
            shape=entry.shape,
            dtype=entry.dtype,
            byte_offset=entry.byte_offset,
            strides_bytes=entry.strides_bytes,
        )

    if not remap:
        return mutation_input_slots

    safe_remap: dict[int, int] = {}
    for old_slot_idx, new_slot_idx in remap.items():
        wrote_mutation = False
        input_read_after_write = False
        for d in dispatches:
            if (
                wrote_mutation
                and new_slot_idx in d.buf_slot_indices
                and new_slot_idx not in d.write_slot_indices
            ):
                input_read_after_write = True
                break
            if old_slot_idx in d.write_slot_indices:
                wrote_mutation = True
        if not input_read_after_write:
            safe_remap[old_slot_idx] = new_slot_idx

    skipped_slots = set(remap) - set(safe_remap)
    if skipped_slots:
        for old_slot_idx in skipped_slots:
            for out_idx, entry in remapped_outputs.get(old_slot_idx, []):
                output_mapping[out_idx] = entry
        remap = safe_remap

    if not remap:
        return mutation_input_slots

    for d in dispatches:
        d.buf_slot_indices = [remap.get(si, si) for si in d.buf_slot_indices]
        d.write_slot_indices = {remap.get(si, si) for si in d.write_slot_indices}
    return mutation_input_slots


def _classify_plan_output(
    value, ptr_to_slot: dict[int, int], input_ptrs: dict[int, InputPtrInfo] | None = None
) -> list[OutputEntry]:
    """Classify a single output value for the compiled plan."""
    if value is None:
        return [OutputNone()]
    if isinstance(value, AlloyBuffer):
        buf = value
        if buf.size == 0:
            return [OutputConst(tuple(buf._shape), buf._dtype)]
        # 0-d outputs may be produced by a kernel (e.g. compiled AdamW's
        # bias-correction scalar), so resolve via slot lookup first and
        # only fall back to OutputConst when the buffer isn't wired to a
        # dispatch write or input passthrough.
        result = _find_plan_output_slot(buf, ptr_to_slot)
        if result is not None:
            return [result]
        if buf.ndim == 0:
            if input_ptrs is not None:
                ptr = buf.data_ptr
                if ptr in input_ptrs:
                    info = input_ptrs[ptr]
                    return [OutputPassthrough(info.arg_idx, tuple(buf._shape), buf._dtype, buf._offset)]
                base = buf.base_ptr
                if base != ptr and base in input_ptrs:
                    info = input_ptrs[base]
                    return [OutputPassthrough(info.arg_idx, tuple(buf._shape), buf._dtype, buf._offset)]
            return [OutputConst(tuple(buf._shape), buf._dtype)]
        if input_ptrs is not None:
            # Buffer not in any dispatch slot — check if it's an input arg
            # passed through (e.g. weight saved for backward).
            ptr = buf.data_ptr
            if ptr in input_ptrs:
                info = input_ptrs[ptr]
                return [OutputPassthrough(info.arg_idx, tuple(buf._shape), buf._dtype, buf._offset)]
            base = buf.base_ptr
            if base != ptr and base in input_ptrs:
                info = input_ptrs[base]
                return [OutputPassthrough(info.arg_idx, tuple(buf._shape), buf._dtype, buf._offset)]
        raise RuntimeError(
            "Alloy backend internal error: unmapped non-scalar AlloyBuffer output "
            f"shape={tuple(buf._shape)} dtype={buf._dtype} "
            f"ptr={buf.data_ptr:#x} base_ptr={buf.base_ptr:#x} "
            f"parent_handle={buf._parent_handle}"
        )
    if isinstance(value, (tuple, list)):
        results: list[OutputEntry] = []
        for item in value:
            results.extend(_classify_plan_output(item, ptr_to_slot, input_ptrs))
        return results
    raise ValueError(f"Unsupported output value type: {type(value)}")


def _find_plan_output_slot(
    buf: AlloyBuffer, ptr_to_slot: dict[int, int]
) -> OutputSlot | None:
    """Find which plan slot an AlloyBuffer maps to."""
    shape = tuple(buf._shape)
    dtype = buf._dtype
    byte_offset = buf._offset
    strides = tuple(buf._strides)

    # Check handle first — unique per allocation, immune to pointer recycling
    if buf._parent_handle >= 0 and buf._parent_handle in ptr_to_slot:
        return OutputSlot(ptr_to_slot[buf._parent_handle], shape, dtype, byte_offset, strides)

    ptr = buf.data_ptr
    if ptr in ptr_to_slot:
        return OutputSlot(ptr_to_slot[ptr], shape, dtype, 0, strides)

    # Fallback to base_ptr with offset
    base_ptr = buf.base_ptr
    if base_ptr != ptr and base_ptr in ptr_to_slot:
        return OutputSlot(ptr_to_slot[base_ptr], shape, dtype, byte_offset, strides)

    return None


def _liveness_analysis(
    slots: list[InputSlot | WeightSlot | IntermediateSlot],
    dispatches: list[PlanDispatch],
    output_mapping: list[OutputEntry],
    trunc_slots: dict[int, int] | None = None,
) -> list[int]:
    """Assign intermediate buffers to reusable physical allocations.

    Mutates IntermediateSlot.physical_idx in place.
    Returns list of physical buffer sizes.
    """
    intermediate_slots = [i for i, s in enumerate(slots) if isinstance(s, IntermediateSlot)]
    if not intermediate_slots:
        return []

    inter_set = set(intermediate_slots)

    # Compute live ranges: slot_idx → (first_use_di, last_use_di)
    live_ranges = {}
    n_dispatches = len(dispatches)
    for di, d in enumerate(dispatches):
        for si in d.buf_slot_indices:
            if si not in inter_set:
                continue
            if si not in live_ranges:
                live_ranges[si] = [di, di]
            else:
                live_ranges[si][1] = di

    # The plan repeats every call. Intermediate slots that are never WRITTEN
    # by any dispatch hold data from outside (constants, pre-computed values).
    # They must survive across plan cycles — extend their range to [0, N].
    written_slots = set()
    for d in dispatches:
        written_slots.update(d.write_slot_indices)
    for slot_idx in live_ranges:
        if slot_idx not in written_slots:
            live_ranges[slot_idx] = [0, n_dispatches]

    # Output slots are read AFTER all dispatches — they must NOT share
    # physical buffers with any other slot.
    output_slot_set = {e.slot_idx for e in output_mapping if isinstance(e, OutputSlot)}

    # Sort by first occurrence (start of live range)
    intervals = sorted(live_ranges.items(), key=lambda x: x[1][0])

    # Linear scan allocation grouped by aligned size
    def _aligned_size(nbytes):
        aligned = ((nbytes + 16383) // 16384) * 16384
        return aligned if aligned > 0 else 16384

    # Pool key: (aligned_size, truncatable). Request-bounded shrink pools
    # must not mix truncatable (M-outer) and pinned slots on one physical
    # buffer — a shared buffer is pinned by its most conservative slot.
    trunc_set = set(trunc_slots or ())
    pool: dict[tuple[int, bool], list[tuple[int, int]]] = {}
    physical_bufs: list[int] = []

    # Optional escape hatch: disable liveness-based buffer pooling so every
    # intermediate gets its own physical buffer. Useful for debugging suspected
    # aliasing bugs in the dispatch plan.
    no_reuse = os.environ.get("ALLOY_NO_BUFFER_REUSE") == "1"

    for slot_idx, (start, end) in intervals:
        nbytes = slots[slot_idx].nbytes
        asize = _aligned_size(nbytes)

        # Output slots get exclusive physical buffers — no sharing
        if slot_idx in output_slot_set or no_reuse:
            phys_idx = len(physical_bufs)
            physical_bufs.append(asize)
            slots[slot_idx].physical_idx = phys_idx
            continue

        # Find a free physical buffer of this aligned size + truncation class
        key = (asize, slot_idx in trunc_set)
        candidates = pool.get(key, [])
        assigned = False
        for ci in range(len(candidates)):
            phys_idx, last_end = candidates[ci]
            if last_end < start:
                slots[slot_idx].physical_idx = phys_idx
                candidates[ci] = (phys_idx, end)
                assigned = True
                break

        if not assigned:
            phys_idx = len(physical_bufs)
            physical_bufs.append(asize)
            slots[slot_idx].physical_idx = phys_idx
            pool.setdefault(key, []).append((phys_idx, end))

    # Intermediates referenced by NO dispatch never entered `live_ranges`, so they
    # never got a physical_idx (stays -1). Give each its own buffer so plan
    # registration can resolve it instead of indexing the pool with -1.
    for slot_idx in intermediate_slots:
        if slots[slot_idx].physical_idx < 0:
            logger.debug("unreferenced_intermediate", slot=slot_idx, nbytes=slots[slot_idx].nbytes)
            slots[slot_idx].physical_idx = len(physical_bufs)
            physical_bufs.append(_aligned_size(slots[slot_idx].nbytes))

    return physical_bufs


def _plan_dependency_groups(
    slots: list[InputSlot | WeightSlot | IntermediateSlot],
    dispatches: list[PlanDispatch],
) -> list[list[int]]:
    """Partition dispatches into dependency groups with Metal memory barriers.

    Metal compute dispatches need explicit memoryBarrierWithScope between
    dispatches that share buffers (read-after-write, write-after-read,
    write-after-write). We build groups where no dispatch reads a slot
    written by another dispatch in the same group.
    """
    if not dispatches:
        return []
    n = len(dispatches)

    # Map slot index → physical resource ID for dependency analysis.
    # Two slots can share the same physical buffer (liveness reuse), so we
    # must track dependencies on PHYSICAL buffers, not logical slots.
    def _phys_id(si: int) -> int:
        s = slots[si]
        if isinstance(s, IntermediateSlot) and s.physical_idx >= 0:
            return -(s.physical_idx + 1)  # negative to distinguish from slot ids
        return si  # INPUT/WEIGHT slots are unique

    read_sets: list[set[int]] = []
    write_sets: list[set[int]] = []
    for d in dispatches:
        reads = {_phys_id(si) for si in d.buf_slot_indices if si not in d.write_slot_indices}
        writes = {_phys_id(si) for si in d.write_slot_indices}
        read_sets.append(reads)
        write_sets.append(writes)

    # Greedy grouping: extend current group until a conflict is found.
    groups: list[list[int]] = []
    g_writes: set[int] = set()
    g_reads: set[int] = set()
    cur: list[int] = []
    for i in range(n):
        conflict = bool(
            (read_sets[i] & g_writes)  # RAW: we read something the group wrote
            | (write_sets[i] & g_reads)  # WAR: we write something the group reads
            | (write_sets[i] & g_writes)  # WAW: we write something the group wrote
        )
        if conflict and cur:
            groups.append(cur)
            cur = [i]
            g_writes = set(write_sets[i])
            g_reads = set(read_sets[i])
        else:
            cur.append(i)
            g_writes |= write_sets[i]
            g_reads |= read_sets[i]
    if cur:
        groups.append(cur)
    return groups


def _execute_plan(plan: CompiledPlan, args: tuple[torch.Tensor, ...],
                  pre_copies: list[tuple[int, int, int, int, int]] | None = None,
                  wanted_outputs: frozenset[int] | None = None,
                  args_stable: bool = False,
                  grid_updates: list[tuple[int, int, int, int]] | None = None):
    """Execute a compiled plan via C++ dispatch_plan.

    Args:
        plan: fully-initialized CompiledPlan
        args: tuple of torch.Tensors (inputs)
        pre_copies: optional GPU-side bulk copies (dst_handle, dst_offset,
            src_handle, src_offset, nbytes) run at the head of the plan's
            command buffer (speculative-decode state propagation).
        wanted_outputs: if given, only reconstruct these output indices
            (others returned as None) — the spec loop reads a single argmax
            output but the verify plan declares ~30 (cache mutations etc.);
            skipping the unread wrappers cuts per-round Python materially.
        grid_updates: optional per-call launch-grid overrides (flat_dispatch_idx,
            gx, gy, gz) — grid-shrunk prefill dispatches an exact threadgroup count
            for the real prompt length against the max-length-compiled plan.

    Returns:
        tuple of output torch.Tensors
    """

    # The plan was compiled with bool inputs widened to int32 (see
    # _to_lazy_input). Run-1+ rebinds the CALLER's args, still 1-byte bool —
    # binding against the 4-byte int32 slot reads 0x01010101 garbage, so widen
    # here too. Only graphs with bool inputs (vision's padding mask) pay this.
    if any(isinstance(a, torch.Tensor) and a.dtype == torch.bool for a in args):
        args = tuple(
            a.to(torch.int32) if isinstance(a, torch.Tensor) and a.dtype == torch.bool else a
            for a in args
        )

    # Fast path: if all INPUT storages are unchanged since last call,
    # reuse cached input_updates — skip O(n_slots) scan+build entirely.
    plan_handle = plan.plan_handle
    cached = plan._cached_input_updates
    if args_stable and cached is not None:
        # Caller guarantees the input tensors (and their storages) are the same
        # objects every call — the spec loop replays a fixed (plan, args) pair.
        # Skip the O(n_input_args) storage-change scan entirely. `cached` stays
        # non-None so the rebuild block below is skipped.
        input_updates = cached
    elif cached is not None:
        # Verify no storage changed (O(n_unique_input_args), not O(n_slots))
        stale = False
        for check in plan._cached_input_check:
            arg = args[check.arg_idx]
            if (
                arg.untyped_storage().data_ptr() != check.storage_ptr
                or arg.data_ptr() != check.data_ptr
            ):
                stale = True
                break
        if not stale:
            input_updates = cached
        else:
            plan._cached_input_updates = None
            plan._cached_input_check = None
            cached = None  # fall through to full rebuild

    if cached is None:
        # Convert INPUT tensor storages to alloy buffers. After set_(), the
        # tensor lives in alloy memory. optimizer.step() modifies it directly.
        _needs_scan = False
        for slot in plan.slots:
            if isinstance(slot, InputSlot):
                a = args[slot.arg_idx]
                if (
                    isinstance(a, torch.Tensor)
                    and a.untyped_storage().data_ptr() not in plan._storage_handles
                ):
                    _needs_scan = True
                    break
        if _needs_scan:
            for slot in plan.slots:
                if not isinstance(slot, InputSlot):
                    continue
                a = args[slot.arg_idx]
                if not isinstance(a, torch.Tensor):
                    continue
                storage = a.untyped_storage()
                sp = storage.data_ptr()
                if sp in plan._storage_handles:
                    continue
                # Check if already an alloy buffer (e.g. from saved_tensors_hooks).
                # If so, just record the handle — no allocation or copy needed.
                existing_handle = _metal_ext.buf_handle_for_ptr(sp)
                if existing_handle >= 0:
                    plan._storage_handles[sp] = existing_handle
                    continue
                nbytes = storage.nbytes()
                handle = _metal_ext.buf_alloc(nbytes)
                aligned_ptr = _metal_ext.buf_ptr(handle)
                ctypes.memmove(aligned_ptr, sp, nbytes)
                raw = (ctypes.c_uint8 * nbytes).from_address(aligned_ptr)
                flat = torch.frombuffer(raw, dtype=torch.uint8)
                new_storage = flat.untyped_storage()
                plan._storage_handles[new_storage.data_ptr()] = handle
                with torch.no_grad():
                    a.set_(new_storage, a.storage_offset(), a.shape, a.stride())
                plan._align_refs.append((raw, handle))

        # Build input slot updates: (slot_idx, handle, offset)
        input_updates = []
        for si, slot in enumerate(plan.slots):
            if isinstance(slot, InputSlot):
                a = args[slot.arg_idx]
                storage = a.untyped_storage()
                sp = storage.data_ptr()
                handle = plan._storage_handles.get(sp, 0)
                offset = a.data_ptr() - sp
                input_updates.append((si, handle, offset))

        # Cache for subsequent calls. Store both the storage pointer and the
        # current data pointer: same-storage views can have different offsets.
        seen_args: dict[int, InputUpdateCacheCheck] = {}
        for slot in plan.slots:
            if isinstance(slot, InputSlot) and slot.arg_idx not in seen_args:
                arg = args[slot.arg_idx]
                seen_args[slot.arg_idx] = InputUpdateCacheCheck(
                    arg_idx=slot.arg_idx,
                    storage_ptr=arg.untyped_storage().data_ptr(),
                    data_ptr=arg.data_ptr(),
                )
        plan._cached_input_updates = input_updates
        plan._cached_input_check = tuple(seen_args.values())

    # Dispatch async: commit the command buffer but don't wait. Output
    # reconstruction below creates tensor wrappers (no data read); sync before
    # returning so the caller sees valid data.
    #
    # Record-only (eager warmup): `_cached_input_updates` is populated above —
    # the reason the warmup runs this replay. The GPU forward's output is
    # discarded, so skip it; this makes a full-M_MAX warmup replay ~free instead
    # of a real M_MAX forward (catastrophic at native context).
    _skip_gpu = is_record_only()
    if not _skip_gpu:
        DispatchEngine.default().dispatch_plan(
            plan_handle, input_updates, n_dispatches=len(plan.dispatches),
            defer_wait=True, pre_copies=pre_copies, grid_updates=grid_updates,
        )

    results = []
    for oi, entry in enumerate(plan.output_mapping):
        if wanted_outputs is not None and oi not in wanted_outputs:
            # Skip reconstructing outputs the caller won't read. In-place cache
            # mutations already happened GPU-side; their wrappers are unused.
            results.append(None)
            continue
        if isinstance(entry, OutputNone):
            results.append(None)
        elif isinstance(entry, OutputConst):
            results.append(
                torch.zeros(entry.shape, dtype=IR_TO_TORCH.get(entry.dtype.ir, torch.float32))
            )
        elif isinstance(entry, OutputPassthrough):
            t = args[entry.arg_idx]
            if entry.byte_offset:
                results.append(
                    make_tensor_from_ptr(t.data_ptr(), entry.shape, entry.dtype, entry.byte_offset)
                )
            elif tuple(t.shape) != entry.shape:
                results.append(t.reshape(entry.shape))
            else:
                results.append(t)
        elif isinstance(entry, OutputSlot):
            slot = plan.slots[entry.slot_idx]

            if isinstance(slot, InputSlot):
                t = args[slot.arg_idx]
                if entry.byte_offset or (
                    entry.strides_bytes
                    and entry.strides_bytes
                    != _compute_contiguous_strides(entry.shape, t.element_size())
                ):
                    results.append(
                        make_tensor_from_ptr(
                            t.data_ptr(),
                            entry.shape,
                            entry.dtype,
                            entry.byte_offset,
                            total_nbytes=t.untyped_storage().nbytes(),
                            strides_bytes=entry.strides_bytes,
                        )
                    )
                elif tuple(t.shape) != entry.shape:
                    results.append(t.reshape(entry.shape))
                else:
                    results.append(t)
            elif isinstance(slot, WeightSlot):
                wb = plan.weight_bindings[entry.slot_idx]
                results.append(
                    make_tensor_from_ptr(
                        wb.base_ptr,
                        entry.shape,
                        entry.dtype,
                        entry.byte_offset,
                        max(wb.metal_nbytes, slot.nbytes),
                        strides_bytes=entry.strides_bytes,
                    )
                )
            else:
                arr = plan.phys_arrays[slot.physical_idx]
                phys_total = max(arr.metal_nbytes, slot.nbytes)
                eff_offset = entry.byte_offset
                eff_strides = entry.strides_bytes
                if eff_offset < 0 or eff_offset >= phys_total:
                    eff_offset = 0
                    eff_strides = None
                results.append(
                    make_tensor_from_ptr(
                        arr.base_ptr,
                        entry.shape,
                        entry.dtype,
                        eff_offset,
                        phys_total,
                        strides_bytes=eff_strides,
                    )
                )

    # Sync only if the host can observe this plan's outputs.
    # `host_reachable_output` is True when any output is a PlainAOTOutput
    # (returned to user code); False when every output is saved-for-bwd / gradient
    # / input mutation, consumed by the next plan's GPU dispatches under Metal's
    # queue-order handoff.
    # Training: gradients/input-mutations are flagged not host-reachable, but the
    # eager optimizer reads param.grad on the HOST, so sync — else opt.step()
    # races and reads the zero-initialised grad buffer.
    if (plan.host_reachable_output or is_training_mode_enabled()) and not _skip_gpu:
        DispatchEngine.gpu_sync()

    if not results or all(r is None for r in results):
        logger.warning(
            "Plan produced no results: %d mappings, kinds=%s",
            len(plan.output_mapping),
            [type(e).__name__ for e in plan.output_mapping],
        )
        return None
    if len(results) == 1:
        return results[0]
    return tuple(results)


def register_decode_chunk_plan(
    plan: CompiledPlan,
    *,
    token_input_slot_idx: int,
    cache_position_slot_idx: int,
    token_out_slot_idx: int,
    token_out_byte_offset: int,
    generated_handle: int,
    generated_nbytes: int,
    chunk: int,
    update_pso: int,
    copy_pso: int,
    incr_pso: int = -1,
    update_incr1_pso: int = -1,
    prop_slot_pairs: tuple[tuple[int, int, int], ...] = (),
    incr_slots: tuple[int, ...] = (),
    skip_dispatch_indices: frozenset[int] = frozenset(),
) -> int:
    """Register a synthetic plan that runs `chunk` decode iterations in ONE
    command buffer: `plan`'s dispatch list repeated `chunk` times, with a 1-thread
    update dispatch between iterations doing the GPU-side feedback (token_out →
    token_in, cache_position += 1, generated[i] = token). Python dispatches one
    chunk per `dispatch_plan` call instead of one token, so the commit/wait cost
    spreads over `chunk` tokens.

    Slot indices are PRESERVED from `plan` (the generated buffer rides as one
    extra stable slot at the end), so `plan._cached_input_updates` binds the chunk
    plan unchanged. `prop_slot_pairs` are (src_slot, src_byte_offset,
    dst_input_slot) 8-byte scalar propagations for AOT input mutations not
    remapped in-plan (per-layer cumulative_length) — applied GPU-side after each
    iteration's update, mirroring AOT's runtime epilogue.
    """
    slots_data = []
    for si, slot in enumerate(plan.slots):
        if isinstance(slot, InputSlot):
            slots_data.append((_SLOT_INPUT, slot.arg_idx, 0, slot.nbytes))
        elif isinstance(slot, WeightSlot):
            arr = plan.weight_bindings[si]
            slots_data.append((_SLOT_WEIGHT, -1, arr._parent_handle, arr.metal_nbytes))
        else:
            arr = plan.phys_arrays[slot.physical_idx]
            slots_data.append((_SLOT_INTERMEDIATE, -1, arr._parent_handle, arr.metal_nbytes))
    gen_slot = len(slots_data)
    slots_data.append((_SLOT_WEIGHT, -1, generated_handle, generated_nbytes))

    # Elide the producer dispatches of folded counter mutations (their output is
    # consumed only by the mutation writeback, now an incr8 in the feedback).
    # Build the per-iteration base + dep-groups over the KEPT dispatches.
    keep = [di for di in range(len(plan.dispatches)) if di not in skip_dispatch_indices]
    remap = {old: new for new, old in enumerate(keep)}
    base = [
        (
            plan.dispatches[di].pso_handle,
            list(plan.dispatches[di].buf_slot_indices),
            list(plan.dispatches[di].buf_offsets),
            plan.dispatches[di].grid,
            plan.dispatches[di].tg,
        )
        for di in keep
    ]
    base_dep_groups: list[list[int]] = []
    for group in plan.dep_groups:
        g = [remap[gi] for gi in group if gi in remap]
        if g:
            base_dep_groups.append(g)
    one = (1, 1, 1)
    # Fold the single counter increment into the update dispatch (common decode
    # shape: one shared cumulative_length) so the feedback group runs ONE dispatch
    # instead of update+incr8. 0 or >=2 counters keep the separate-incr8 path.
    fold_counter = (
        incr_slots[0] if (len(incr_slots) == 1 and update_incr1_pso >= 0) else None
    )
    dispatches_data: list[tuple] = []
    dep_groups: list[list[int]] = []
    for i in range(chunk):
        off = len(dispatches_data)
        dispatches_data.extend(base)
        for group in base_dep_groups:
            dep_groups.append([gi + off for gi in group])
        update_idx = len(dispatches_data)
        if fold_counter is not None:
            dispatches_data.append((
                update_incr1_pso,
                [token_out_slot_idx, token_input_slot_idx, cache_position_slot_idx,
                 gen_slot, fold_counter],
                [token_out_byte_offset, 0, 0, 8 * i, 0],
                one, one,
            ))
        else:
            dispatches_data.append((
                update_pso,
                [token_out_slot_idx, token_input_slot_idx, cache_position_slot_idx, gen_slot],
                [token_out_byte_offset, 0, 0, 8 * i],
                one, one,
            ))
        # The update and the scalar copies touch disjoint buffers (update:
        # token_in/cache_position/generated; copies: distinct mutation in/out
        # slots), so they share ONE dependency group — no barrier between them,
        # dropping a per-token GPU pipeline drain (~10µs end-of-token bubble).
        feedback_group = [update_idx]
        for src_slot, src_off, dst_slot in prop_slot_pairs:
            dispatches_data.append(
                (copy_pso, [src_slot, dst_slot], [src_off, 0], one, one)
            )
            feedback_group.append(len(dispatches_data) - 1)
        # Folded counter mutations: self-increment the destination in place
        # (dst += 1), replacing the elided producer add + its copy8. The single
        # `fold_counter` is done inside update_incr1 above.
        for dst_slot in incr_slots:
            if dst_slot == fold_counter:
                continue
            dispatches_data.append((incr_pso, [dst_slot], [0], one, one))
            feedback_group.append(len(dispatches_data) - 1)
        dep_groups.append(feedback_group)

    written = {si for di in keep for si in plan.dispatches[di].write_slot_indices}
    written |= {token_input_slot_idx, cache_position_slot_idx, gen_slot}
    written |= {dst for _, _, dst in prop_slot_pairs}
    written |= set(incr_slots)
    return DispatchEngine.default().register_plan(
        dispatches_data, slots_data, dep_groups, sorted(written)
    )


def _bw_compiler_inference_path(gm, example_inputs):
    """Backward compiler for the inference AOT backend.

    Reaching here means AOT decided backward is reachable (params have
    requires_grad=True, not under no_grad). If the user never called
    set_training_mode(True), warn: they're either training (should flip the flag)
    or forgot eval()/no_grad.
    """
    if not is_training_mode_enabled():
        # The training-preview boundary forbids loading `alloy_torch.training` at
        # import time; pull it in only when a backward fires without the flag.
        from alloy_torch.training import warn_if_backward_without_mode  # scoped: training-boundary contract — keep `alloy_torch` import inference-only

        warn_if_backward_without_mode()
    return _compile_fx(gm, example_inputs)


_AOT_BACKEND = aot_autograd(
    fw_compiler=_compile_fx,
    bw_compiler=_bw_compiler_inference_path,
    inference_compiler=_compile_fx,
    decompositions=get_alloy_decompositions,
)


_training_aot_backend = None


def _get_training_backend():
    global _training_aot_backend
    if _training_aot_backend is None:
        _training_aot_backend = aot_autograd(
            fw_compiler=_compile_fx,
            inference_compiler=_compile_fx,
            decompositions=lambda: get_alloy_decompositions(training=True),
        )
    return _training_aot_backend


def alloy_backend(gm: torch.fx.GraphModule, example_inputs: Sequence[Any], *, mode=None, **kwargs):
    """Compile an FX graph to an Alloy-backed callable."""
    if mode and mode != "default":
        logger.debug("alloy backend: mode=%r accepted, using the standard path", mode)
    if kwargs:
        logger.debug(f"alloy backend: ignoring extra kwargs {sorted(kwargs)}")
    backend = _get_training_backend() if is_training_mode_enabled() else _AOT_BACKEND
    return backend(gm, example_inputs)
