"""FX-graph cache: skip Dynamo + AOT on repeat compiles of the same graph.

THE DEV-LOOP PROBLEM. ~85% of `eager_compile_all` is the torch.compile
front-end (Dynamo bytecode trace + FakeTensor meta-prop + FX proxying) re-run
on every process start — measured 14.75s on qwen3.5:0.8b, of which alloy's own
pipeline is ~10%. A plan-level cache is useless for development (any kernel
edit invalidates it), but the ATen FX graph that Dynamo+AOT produce sits at a
layer ABOVE everything a kernel developer iterates on: it depends only on the
HF model code, the compat patches, the generation wrapper modules, the custom
op registry, and the AOT decompositions. It is INVARIANT to std/* kernels, the
tile compiler/emitter, the fusion engine, the dispatch layer — and even to
rewrites/ and ops/ handlers, because those run INSIDE `_compile_fx` at replay.

So: cache the PRE-rewrite ATen graph `_compile_fx` receives, keyed by the
graph-affecting sources. On a hit, feed it straight back to `_compile_fx` —
no Dynamo, no AOT — and the whole alloy pipeline still executes fresh.

WHAT A CACHE ENTRY HOLDS
- the sanitized graph (node.meta reduced to meta-device 'val' tensors),
- the input spec: for each flat graph arg, where it comes from — a module
  param/buffer (FQN), a caller kwarg (name), a StaticCache field
  (kwarg, layer index, attr), or a lifted constant (serialized) — recorded by
  matching tensor identity on the first real call,
- the user-output index: which flat output the wrapped module returns,
  matched by identity against the Dynamo path's return value.

REPLAY contract matches the pinned-plan path that production already runs:
calling `_compile_fx`'s product directly bypasses AOT's runtime epilogue, but
alloy plans write mutated inputs (KV cache, cumulative_length) in place
through the converted storages — the same contract `PrefillEngine.chunk_step`'s
pinned fast path relies on. The plan-capture hooks (`capture_plan`) live
inside `compiled`, so eager_compile pinning works identically on both paths.

Anything unexpected (graph breaks producing != 1 graph, an unmatchable arg, a
return value not found in the flat outputs, a deserialization error) falls
back to the plain torch.compile path for that entry, permanently and loudly.
ALLOY_FX_CACHE=0 disables the whole mechanism.
"""

from __future__ import annotations

import hashlib
import importlib
import operator
import os
import pathlib
import time
from typing import Any

import torch
import transformers

from alloy import get_logger
from alloy_torch import backend
from alloy_torch.backend import _compile_fx, _graph_capture_stack
from alloy_torch.compile_window import compile_window
from alloy_server.models.attention import current_use_alloy_warm_op

logger = get_logger("alloy_torch.graph_cache")

_CACHE_VERSION = (
    6  # v6: drop debug-provenance node.meta (stack_trace etc.) — ~5x smaller payload
)
_CACHE_DIR = pathlib.Path(
    os.environ.get(
        "ALLOY_FX_CACHE_DIR", str(pathlib.Path.home() / ".cache" / "alloy" / "fx_graphs")
    )
)

# Sources whose changes alter the traced ATen graph. Kernel / emitter / fusion /
# dispatch / rewrites / ops-handler edits do NOT appear here by design — that is
# the whole point of caching at this layer.
_FINGERPRINT_FILES = (
    "generation.py",
    "cache.py",
    "multi_token_attention.py",
    "_custom_ops.py",
    "_decomps.py",
    "_qwen3_5_compat.py",
    "_gemma4_compat.py",
    "_gemma4_audio.py",
    "embedding.py",
    "mtp.py",
    "_modality.py",
)

_source_fp: str | None = None


def _source_fingerprint() -> str:
    global _source_fp
    if _source_fp is None:
        h = hashlib.sha256()
        h.update(
            f"v{_CACHE_VERSION}|torch={torch.__version__}|tf={transformers.__version__}".encode()
        )
        pkg = pathlib.Path(__file__).parent
        for name in _FINGERPRINT_FILES:
            p = pkg / name
            if p.exists():
                h.update(name.encode())
                h.update(p.read_bytes())
        _source_fp = h.hexdigest()[:16]
    return _source_fp


def _enabled() -> bool:
    return os.environ.get("ALLOY_FX_CACHE", "1") not in ("0", "")  # "" guards empty-set


class GraphCapture:
    """Capture scope pushed onto backend._graph_capture_stack: collects the
    sanitized ATen graph(s) compiled inside it and each one's first real call
    (flat args + flat outputs)."""

    def __init__(self) -> None:
        self.graphs: list[torch.fx.GraphModule] = []
        self.runs: dict[int, tuple[tuple, Any]] = {}

    def note_graph(self, gm: torch.fx.GraphModule) -> None:
        self.graphs.append(gm)

    def note_run(self, gidx: int, args: tuple, out: Any) -> None:
        if gidx not in self.runs:
            self.runs[gidx] = (args, out)


def _serialize_target(target: Any) -> tuple:
    if isinstance(target, str):
        return ("attr", target)
    if isinstance(target, torch._ops.OpOverload):
        return ("op", str(target))  # e.g. "aten.mm.default"
    if isinstance(target, torch._ops.HigherOrderOperator):
        # Named singletons in the higher_order namespace (auto_functionalized_v2
        # wraps alloy's mutable custom ops) — restore by name.
        return ("hop", str(target))
    if callable(target) and target.__module__ in ("operator", "builtins", "_operator"):
        return ("callable", target.__module__, target.__name__)
    raise RuntimeError(f"unsupported fx target: {target!r}")


def _deserialize_target(entry: tuple) -> Any:
    kind = entry[0]
    if kind == "attr":
        return entry[1]
    if kind == "op":
        # Resolve through the NORMAL attribute protocol (attrgetter) — calling
        # __getattr__ directly on torch.ops bypasses its namespace caching and
        # mints fresh OpOverload objects that fail the rewrites' identity/set
        # checks (every view node silently stopped matching _VIEW_OPS).
        return operator.attrgetter(entry[1])(torch.ops)
    if kind == "hop":
        return operator.attrgetter(f"higher_order.{entry[1]}")(torch.ops)
    if kind == "callable":
        return vars(importlib.import_module(entry[1]))[entry[2]]
    raise RuntimeError(f"unsupported target entry: {entry!r}")


def _encode_arg(a: Any) -> Any:
    if isinstance(a, torch.fx.Node):
        return ("__node__", a.name)
    if isinstance(a, torch._ops.OpOverload):
        return ("__op__", str(a))
    if isinstance(a, tuple):
        return ("__tuple__", [_encode_arg(x) for x in a])
    if isinstance(a, list):
        return ("__list__", [_encode_arg(x) for x in a])
    if isinstance(a, dict):
        return ("__dict__", {k: _encode_arg(v) for k, v in a.items()})
    if isinstance(a, slice):
        return ("__slice__", a.start, a.stop, a.step)
    return ("__leaf__", a)


def _decode_arg(a: Any, env: dict) -> Any:
    tag = a[0]
    if tag == "__node__":
        return env[a[1]]
    if tag == "__op__":
        return _deserialize_target(("op", a[1]))
    if tag == "__tuple__":
        return tuple(_decode_arg(x, env) for x in a[1])
    if tag == "__list__":
        return [_decode_arg(x, env) for x in a[1]]
    if tag == "__dict__":
        return {k: _decode_arg(v, env) for k, v in a[1].items()}
    if tag == "__slice__":
        return slice(a[1], a[2], a[3])
    return a[1]


def _fetch_attr(gm: torch.nn.Module, qualname: str) -> Any:
    """Resolve a dotted get_attr target via the module's explicit registries
    (no dynamic getattr: _parameters / _buffers / _modules / __dict__)."""
    mod: torch.nn.Module = gm
    parts = qualname.split(".")
    for p in parts[:-1]:
        mod = mod._modules[p]
    leaf = parts[-1]
    if leaf in mod._parameters:
        return mod._parameters[leaf]
    if leaf in mod._buffers:
        return mod._buffers[leaf]
    return mod.__dict__[leaf]


def _serialize_gm(gm: torch.fx.GraphModule) -> dict:
    nodes = []
    attrs: dict[str, torch.Tensor] = {}
    for n in gm.graph.nodes:
        if n.op == "get_attr":
            t = _fetch_attr(gm, n.target)
            if not torch.is_tensor(t):
                raise RuntimeError(f"non-tensor get_attr {n.target}: {type(t).__name__}")
            attrs[n.target] = t.detach().clone()
        nodes.append(
            {
                "name": n.name,
                "op": n.op,
                "target": _serialize_target(n.target) if n.op != "output" else ("attr", "output"),
                "args": _encode_arg(tuple(n.args)),
                "kwargs": _encode_arg(dict(n.kwargs)),
                "meta": backend._sanitize_meta(n.meta),
            }
        )
    return {"nodes": nodes, "attrs": attrs, "gm_meta": backend._sanitize_meta(gm.meta)}


def _deserialize_gm(payload: dict) -> torch.fx.GraphModule:
    root = torch.nn.Module()
    for qualname, t in payload["attrs"].items():
        mod = root
        parts = qualname.split(".")
        for p in parts[:-1]:
            if p not in mod._modules:
                mod.add_module(p, torch.nn.Module())
            mod = mod._modules[p]
        mod.register_buffer(parts[-1], t, persistent=False)
    graph = torch.fx.Graph()
    env: dict[str, torch.fx.Node] = {}
    for rec in payload["nodes"]:
        args = _decode_arg(rec["args"], env)
        kwargs = _decode_arg(rec["kwargs"], env)
        if rec["op"] == "output":
            node = graph.output(args[0] if len(args) == 1 else args)
        elif rec["op"] == "placeholder":
            node = graph.placeholder(rec["target"][1])
        else:
            node = graph.create_node(
                rec["op"], _deserialize_target(rec["target"]), args, kwargs, rec["name"]
            )
        node.name = rec["name"]
        node.meta = rec["meta"]
        env[rec["name"]] = node
    gm = torch.fx.GraphModule(root, graph)
    gm.meta.update(payload["gm_meta"])
    return gm


_CACHE_TENSOR_ATTRS = ("keys", "values", "conv_states", "recurrent_states", "cumulative_length")


def _cache_field(layer: Any, attr: str) -> Any:
    if attr == "keys":
        return layer.keys if hasattr(layer, "keys") else None
    if attr == "values":
        return layer.values if hasattr(layer, "values") else None
    if attr == "conv_states":
        return layer.conv_states if hasattr(layer, "conv_states") else None
    if attr == "recurrent_states":
        return layer.recurrent_states if hasattr(layer, "recurrent_states") else None
    if attr == "cumulative_length":
        return layer.cumulative_length if hasattr(layer, "cumulative_length") else None
    return None


def _id_to_source(module: torch.nn.Module, kwargs: dict) -> dict[int, tuple]:
    id2src: dict[int, tuple] = {}
    for fqn, p in module.named_parameters(remove_duplicate=False):
        id2src.setdefault(id(p), ("param", fqn))
    for fqn, b in module.named_buffers(remove_duplicate=False):
        id2src.setdefault(id(b), ("buffer", fqn))
    for name, v in kwargs.items():
        if torch.is_tensor(v):
            id2src.setdefault(id(v), ("kwarg", name))
        elif hasattr(v, "layers"):  # StaticCache-shaped object
            for i, layer in enumerate(v.layers):
                for attr in _CACHE_TENSOR_ATTRS:
                    t = _cache_field(layer, attr)
                    if torch.is_tensor(t):
                        id2src.setdefault(id(t), ("cache", name, i, attr))
                # Side-channel state stashed as plain tensor attributes on the
                # cache layer (e.g. DeltaNet's `alloy_attn_mask` pad mask) must
                # resolve live at replay. Const-baking it freezes mutable state:
                # an all-zero capture-time snapshot reads as "every position is
                # padding" forever — found as garbage decode on every hybrid.
                for attr, t in vars(layer).items():
                    if torch.is_tensor(t):
                        id2src.setdefault(id(t), ("cache", name, i, attr))
    return id2src


def _resolve_spec(spec: list[tuple], module: torch.nn.Module, kwargs: dict) -> list[torch.Tensor]:
    params = dict(module.named_parameters(remove_duplicate=False))
    buffers = dict(module.named_buffers(remove_duplicate=False))
    args: list[torch.Tensor] = []
    for entry in spec:
        kind = entry[0]
        if kind == "param":
            args.append(params[entry[1]])
        elif kind == "buffer":
            args.append(buffers[entry[1]])
        elif kind == "kwarg":
            args.append(kwargs[entry[1]])
        elif kind == "cache":
            layer = kwargs[entry[1]].layers[entry[2]]
            t = _cache_field(layer, entry[3])
            if t is None:
                t = vars(layer).get(entry[3])
            if not torch.is_tensor(t):
                raise KeyError(f"cache field {entry[2]}.{entry[3]} missing at replay")
            args.append(t)
        elif kind == "const":
            args.append(entry[1])
        else:
            raise KeyError(f"unknown spec entry {kind}")
    return args


class GraphCachingModule:
    """Drop-in for `torch.compile(module, backend='alloy', dynamic=False)`.

    Per call signature (shapes/dtypes of the kwarg leaves + the trace-time
    mode globals), the first process EVER pays the Dynamo trace and persists
    the ATen graph + input spec; every later process replays it through
    `_compile_fx` directly. In-process repeat calls go through whichever
    callable the first call built (Dynamo's guard cache / the replay closure's
    `_compiled_plan` fast path — identical steady-state behavior)."""

    def __init__(self, module: torch.nn.Module, label: str) -> None:
        self._module = module
        self._label = label
        self._dynamo: Any = None
        # sig -> ("replay", boxed_fn, spec, out_idx) | ("dynamo",)
        self._entries: dict[str, tuple] = {}
        self._model_fp: str | None = None

    # -- keys ---------------------------------------------------------------

    def _model_fingerprint(self) -> str:
        if self._model_fp is None:
            h = hashlib.sha256()
            for fqn, p in self._module.named_parameters(remove_duplicate=False):
                h.update(f"{fqn}:{tuple(p.shape)}:{p.dtype}".encode())
            for fqn, b in self._module.named_buffers(remove_duplicate=False):
                h.update(f"buf:{fqn}:{tuple(b.shape)}:{b.dtype}".encode())
            self._model_fp = h.hexdigest()[:16]
        return self._model_fp

    def _signature(self, kwargs: dict) -> str:
        h = hashlib.sha256()
        parts: list[str] = []
        h.update(self._label.encode())
        h.update(_source_fingerprint().encode())
        h.update(self._model_fingerprint().encode())
        # Module-level graph variants (e.g. ChunkPrefill's drafter tap
        # layers) are constructor STATE, invisible to source/model
        # fingerprints — without this a tapless cached graph replays for a
        # tapped module and the taps silently vanish.
        variant = vars(self._module).get("_cache_variant")
        if variant is not None:
            parts.append(f"variant:{variant}")
            h.update(f"variant:{variant}".encode())
        flags = f"warm={current_use_alloy_warm_op()}|shrink={compile_window.grid_shrink_active()}"
        parts.append(flags)
        h.update(flags.encode())
        for name in sorted(kwargs):
            v = kwargs[name]
            if torch.is_tensor(v):
                part = f"{name}:{tuple(v.shape)}:{v.dtype}"
                parts.append(part)
                h.update(part.encode())
            elif v is None:
                parts.append(f"{name}:None")
                h.update(f"{name}:None".encode())
            elif hasattr(v, "layers"):
                for i, layer in enumerate(v.layers):
                    for attr in _CACHE_TENSOR_ATTRS:
                        t = _cache_field(layer, attr)
                        if torch.is_tensor(t):
                            part = f"{name}.{i}.{attr}:{tuple(t.shape)}:{t.dtype}"
                            parts.append(part)
                            h.update(part.encode())
                    # Dynamo SPECIALIZES on python-bool cache state (guards),
                    # so it must split the signature too: the GDN forward
                    # branches on `layer.has_previous_state`, and a decode
                    # graph captured at False (fresh cache, eager-compile)
                    # replayed against True runs the prefill conv variant with
                    # zero pre-context — silent decode drift on every hybrid.
                    hps = vars(layer).get("has_previous_state")
                    if isinstance(hps, bool):
                        part = f"{name}.{i}.has_previous_state:{hps}"
                        parts.append(part)
                        h.update(part.encode())
            else:
                parts.append(f"{name}:{type(v).__name__}")
                h.update(f"{name}:{type(v).__name__}".encode())
        sig = h.hexdigest()[:24]
        if os.environ.get("ALLOY_DEBUG_SIG") == "1":
            print(f"[sig] {self._label} {sig} :: {'|'.join(parts)}", flush=True)
        return sig

    def _path(self, sig: str) -> pathlib.Path:
        return _CACHE_DIR / f"{sig}.pt"

    # -- call ---------------------------------------------------------------

    def __call__(self, **kwargs):
        if not _enabled():
            return self._call_dynamo(**kwargs)
        sig = self._signature(kwargs)
        entry = self._entries.get(sig)
        if entry is None:
            entry = self._load(sig)
            if entry is not None:
                self._entries[sig] = entry
            else:
                # Capture executes the call (through Dynamo) as a side effect
                # and registers the signature; return its output directly.
                return self._capture(sig, kwargs)
        if entry[0] == "replay":
            _, boxed, spec, out_idx, mut_map = entry
            args = _resolve_spec(spec, self._module, kwargs)
            outs = boxed(args)
            # AOT's runtime-wrapper epilogue: propagate input mutations the
            # plan didn't internalize back into the input tensors. Where the
            # plan DID internalize a mutation, _execute_plan already returns
            # the input tensor itself — skipped via the identity check.
            for o_idx, a_idx in mut_map.items():
                out_t = outs[o_idx]
                arg_t = args[a_idx]
                if torch.is_tensor(out_t) and out_t is not arg_t:
                    arg_t.copy_(out_t)
            # A live (Dynamo-orchestrated) call replays the forward's python
            # side effects on the cache object; the replayed graph performs
            # only the tensor work. Replicate the one transition our forwards
            # depend on: the call wrote every linear-attention layer's
            # conv/recurrent state, so the lazy-init flag must flip — without
            # this, downstream signatures keep computing has_previous_state=
            # False and pin the unprimed decode graph (garbage decode).
            for v in kwargs.values():
                if not torch.is_tensor(v) and hasattr(v, "layers"):
                    for layer in v.layers:
                        if vars(layer).get("has_previous_state") is False:
                            layer.has_previous_state = True
            if isinstance(out_idx, list):
                return tuple(outs[i] for i in out_idx)
            return outs[out_idx]
        return self._call_dynamo(**kwargs)

    def _call_dynamo(self, **kwargs):
        if self._dynamo is None:
            self._dynamo = torch.compile(self._module, backend="alloy", dynamic=False)
        return self._dynamo(**kwargs)

    # -- replay -------------------------------------------------------------

    def _load(self, sig: str) -> tuple | None:
        path = self._path(sig)
        if not path.exists():
            return None
        t0 = time.perf_counter()
        try:
            payload = torch.load(path, weights_only=False)
            gm = _deserialize_gm(payload["graph"])
            spec = payload["spec"]
            out_idx = payload["out_idx"]
            boxed = _compile_fx(gm, [])
            # AOT's runtime epilogue propagates input mutations the plan does
            # not internalize (`arg.copy_(out)` — e.g. gemma4's read-after-
            # write cumulative_length advance). The raw boxed fn has no AOT
            # wrapper, so the replay branch must apply the same epilogue; the
            # post-rewrite map (desc + auto_functionalized sidecar entries)
            # rides on the boxed fn — same reason the C++ greedy loop carries
            # mutation_pairs.
            mut_map = boxed._alloy_mutation_map
            logger.info(
                "fx_graph_cache_hit",
                label=self._label,
                sig=sig,
                took_ms=round((time.perf_counter() - t0) * 1000.0, 1),
            )
            return ("replay", boxed, spec, out_idx, mut_map)
        except Exception as exc:
            logger.warning("fx_graph_cache_load_failed", label=self._label, sig=sig, error=str(exc))
            try:
                path.unlink()
            except OSError:
                pass
            return None

    # -- capture ------------------------------------------------------------

    def _capture(self, sig: str, kwargs: dict):
        """Run the call through Dynamo under a capture scope, persist the graph
        + input spec, and return the call's output. The signature is pinned to
        the Dynamo callable for the rest of the process (its guard cache makes
        repeat calls cheap; replay is for the NEXT process)."""
        cap = GraphCapture()
        _graph_capture_stack.append(cap)
        try:
            out = self._call_dynamo(**kwargs)
        finally:
            _graph_capture_stack.pop()
        try:
            self._save(sig, cap, out, kwargs)
        except Exception as exc:
            logger.warning(
                "fx_graph_cache_capture_failed", label=self._label, sig=sig, error=str(exc)
            )
        self._entries[sig] = ("dynamo",)
        return out

    def _save(self, sig: str, cap: GraphCapture, user_out: Any, kwargs: dict) -> None:
        if len(cap.graphs) != 1:
            raise RuntimeError(f"expected 1 graph, got {len(cap.graphs)} (graph break?)")
        if 0 not in cap.runs:
            raise RuntimeError("no run captured for the compiled graph")
        args, flat_out = cap.runs[0]

        def _flat_index(o: Any) -> int:
            for i, fo in enumerate(flat_out):
                if fo is o:
                    return i
            raise RuntimeError("module return not found in flat outputs (aliased by AOT epilogue)")

        # out_idx is an int for a single-tensor return, or a list of ints for a
        # tuple return (the decode module yields (token, logits)). Replay rebuilds
        # the tuple from the same flat outputs.
        if torch.is_tensor(user_out):
            out_idx: int | list[int] = _flat_index(user_out)
        elif isinstance(user_out, tuple) and user_out and all(torch.is_tensor(o) for o in user_out):
            out_idx = [_flat_index(o) for o in user_out]
        else:
            raise RuntimeError(f"unsupported module return ({type(user_out).__name__})")
        id2src = _id_to_source(self._module, kwargs)
        spec: list[tuple] = []
        baked: list[str] = []
        for a in args:
            if not torch.is_tensor(a):
                raise RuntimeError(f"non-tensor flat arg ({type(a).__name__})")
            src = id2src.get(id(a))
            if src is None:
                if a.numel() > 1_000_000:
                    raise RuntimeError(f"unmatched large arg {tuple(a.shape)} {a.dtype}")
                src = ("const", a.detach().clone())
                baked.append(f"{tuple(a.shape)}:{a.dtype}")
            spec.append(src)
        if baked:
            # Const-baked args replay as capture-time snapshots. Correct only
            # for true constants (AOT-lifted tables); anything stateful must
            # surface as a param/buffer/kwarg/cache source above.
            logger.info("fx_graph_cache_const_baked", label=self._label, args=baked)
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"graph": _serialize_gm(cap.graphs[0]), "spec": spec, "out_idx": out_idx},
            self._path(sig),
        )
        logger.info("fx_graph_cache_saved", label=self._label, sig=sig, n_args=len(spec))


def graph_caching_compile(module: torch.nn.Module, label: str):
    """torch.compile(module, backend="alloy", dynamic=False), with the FX-graph
    cache in front when enabled."""
    if not _enabled():
        return torch.compile(module, backend="alloy", dynamic=False)
    return GraphCachingModule(module, label)
