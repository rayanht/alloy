"""Shared helper: build the real forward passes a model dispatches.

Used by `alloy profile` and `alloy inspect` to capture the kernels a model
actually runs in production — not an idealized re-derivation. The result is a
`ModelCapture`: a model kind plus an ordered list of `CapturePass`es that the
commands iterate uniformly.

Two model kinds:

  - **causal** (Qwen, Llama, …): two passes — prefill then decode — over a
    real KV cache. Mirrors the generator's attention path (fp32 Q + fp16 KV
    via the alloy cache-attention ops).
  - **embedder** (nomic-bert): one pass per (batch, seq) shape over the
    encoder's single forward — no cache, no decode. Drives the exact compiled
    forward the embedding server pins, via `nomic_bert.build_nomic_forward`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from alloy._runtime.tune import capture_kernel_for_replay
from alloy_torch.compile_window import compile_window
from alloy_server import resolve_prefill_policy
from alloy_server.cache import AlloyStaticCache
from alloy_server.generation.generator import AlloyGenerator
from alloy_server.generation.spec import VerifySpecLogits
from alloy_server.gguf import ResolvedGGUF
from alloy_server.kv_format import resolve_kv_format
from alloy_server.models import load_native_causal_lm, resolve_model
from alloy_server.models.attention import set_deltanet_attn_mask, set_use_alloy_warm_op
from alloy_server.models.mtp import MTPDraftStep, load_quantized_mtp
from alloy_server.models.nomic_bert import build_nomic_forward

# Default prefill depth for `alloy profile` / `alloy inspect`. A representative
# production context and a tier of `alloy bench --depths {512,4096,16384}`, so a
# slow (model, depth) row from the bench profiles 1:1 here. Override with --depth.
_DEFAULT_DEPTH = 4096

# GGUF `general.architecture` values that route to the encoder/embedding path.
_EMBEDDER_ARCHITECTURES = frozenset({"nomic-bert"})

# Embedder shape defaults. With no --batch/--seq, profile a short and a long
# bucket so the output mirrors the causal prefill/decode pair and surfaces the
# win-short / lose-long contrast directly. Either flag pins a single shape,
# filling the unpassed dimension from these.
_DEFAULT_EMBED_BATCH = 8
_DEFAULT_EMBED_SEQ = 128
_EMBED_SHORT_SHAPE = (8, 32)
_EMBED_LONG_SHAPE = (8, 128)


@dataclass
class CapturePass:
    """One forward the commands drive through `visualize` / kernel-observe.

    `setup` (if present) runs once before the pass — e.g. the causal decode
    fills the KV cache to the prompt length. `run` is the zero-arg thunk that
    executes the forward (called repeatedly by `visualize`).
    """

    name: str  # filename slug + console verb, e.g. "prefill", "long"
    label: str  # appended to the model name for the HTML title
    detail: str  # console parenthetical, e.g. "seq_len=42, ctx=2048"
    run: Callable[[], object]
    setup: Callable[[], None] | None = None
    # Explicit plans to profile (passed to `alloy.visualize(plans=…)`). Set for
    # passes that replay an eager-compiled pinned plan — the grid-shrunk partial
    # chunk — which never re-enters the torch.compile backend, so the default
    # auto-capture from `_all_compiled_plans` would find nothing. None → auto.
    # Late-bound: the shrunk pass's setup fills it in (pinned plans only exist
    # after its eager_compile_all).
    plans: list | None = None
    # Context-manager factory `alloy.visualize` wraps around its FIRST run()
    # call (the compile run after the dynamo reset). Causal passes set the
    # generator's `plan_compile_window` so the compile is record-only (phantom
    # intermediates, no GPU — a REAL run-0 holds every intermediate of the
    # forward live at once; measured 100+ GB on qwen3.6:35b's 4096-chunk
    # prefill) and the captured plan carries the production grid-shrink
    # properties. None → the compile run executes unwrapped (embedder /
    # vision / pinned-plan passes, whose first run is small or plan-replay).
    compile_ctx: Callable[[], object] | None = None


@dataclass
class ModelCapture:
    """A model's kind plus the ordered passes to capture."""

    kind: str  # "causal" | "embedder"
    passes: list[CapturePass] = field(default_factory=list)


def _model_architecture(model: str) -> str:
    return resolve_model(model).architecture()


def build_capture(
    model: str,
    *,
    depth: int = _DEFAULT_DEPTH,
    batch: int | None = None,
    seq: int | None = None,
    mtp: bool = False,
) -> ModelCapture:
    """Resolve `model`'s kind and build its capture passes.

    Causal models profile the production chunked-prefill + decode paths on
    SYNTHETIC tokens at cache depth `depth` — the same methodology as
    `alloy bench --depths`, so a slow (model, depth) row from the bench profiles
    1:1 here (clean bench → profile chain for someone optimizing a model). The
    KV cache is sized to the model's native context. Prefill emits BOTH shapes
    generation dispatches: the full chunk (saturated GEMMs, compile-path
    auto-capture) and the grid-shrunk partial last chunk (pinned-plan replay,
    profiled at the shrunk launch). Embedders ignore `depth` (they use
    `batch`/`seq`).

    `mtp=True` appends the MTP self-speculation round's two extra forwards — the
    hidden-emitting M=2 verify and the M=1 draft (block + shared lm_head) — so
    the per-kernel breakdown exposes what a spec round costs over plain decode
    (the lever for lowering the acceptance break-even). Causal-only; raises if
    the model ships no MTP head.
    """
    if _model_architecture(model) in _EMBEDDER_ARCHITECTURES:
        if mtp:
            raise ValueError("--mtp is only meaningful for causal models")
        return _build_embedder_capture(model, batch=batch, seq=seq)
    return _build_causal_capture(model, depth=depth, mtp=mtp)


def _build_causal_capture(
    model: str, *, depth: int = _DEFAULT_DEPTH, mtp: bool = False
) -> ModelCapture:
    loaded = load_native_causal_lm(model)
    hf_model = loaded.model.eval()
    # Build the production generator and profile ITS forwards — that's what makes
    # this "the kernels the model actually runs". from_model installs the
    # multi-token-attention patch, sizes the KV cache to the model's native
    # context, and owns the exact chunked-prefill (cold→warm, per-chunk DeltaNet
    # mask + cumulative_length threading) and decode paths the server runs. We do
    # NOT eager_compile: that pins plans and bypasses Dynamo, but `visualize`
    # reads `_all_compiled_plans` (the torch.compile path), so keep prefill/decode
    # on the compiled path. (The grid-shrunk pass below is the exception — its
    # whole point is the pinned exact-grid plan, so it eager-compiles in its
    # setup and runs LAST, after every auto-capture pass has been profiled.)
    gen = AlloyGenerator.from_model(
        hf_model, cache_dtype=torch.float16, vision=loaded.vision, audio=loaded.audio,
        chunk_prefill_size=resolve_prefill_policy(),
    )
    # The engines are pure executors that dispatch INJECTED modules; build them
    # (cheap — the compile is lazy on first call, which is what `visualize`
    # captures). Profile/inspect deliberately skip the full eager_compile_all
    # (it pins plans + bypasses Dynamo), but the modules must still exist.
    gen.build_modules()
    context = gen.max_cache_len  # native context (derived, never passed)
    chunk = gen.chunk_prefill_size
    if not (1 <= depth < context):
        raise ValueError(f"--depth {depth} must be in [1, {context}) for {model}")

    # Profile ONE chunk-sized forward at the DEEPEST offset (depth − chunk): its
    # full-attention layers scan the whole `depth` KV, so the per-kernel breakdown
    # is the kernel mix at max depth and `wall − GPU-busy` exposes the per-chunk
    # dispatch overhead — WITHOUT re-running all ceil(depth/chunk) chunks (visualize
    # replays each pass ~29× for timing, so a full chunked loop would be ~29·n_chunks
    # forwards). The cold prime lays down the DeltaNet recurrent state + KV (both
    # depth-independent in cost); cumulative_length then drives the attention scan
    # extent. Synthetic tokens are throughput-equivalent to prose (matches
    # `alloy bench --depths`).
    vocab = int(hf_model.config.vocab_size)
    chunk_len = min(chunk, depth)
    offset = depth - chunk_len  # cold (0) when depth <= chunk, else warm at depth−chunk
    g = torch.Generator().manual_seed(0)
    chunk_ids = torch.randint(
        1024, max(1025, vocab - 256), (1, chunk_len), generator=g, dtype=torch.long,
    )
    n_chunks = (depth + chunk - 1) // chunk

    def _set_fill(cache: object, pos: int) -> None:
        # One cumulative_length aliased across layers drives the full-attention
        # scan extent; it scans `pos` positions regardless of KV content (cost is
        # content-independent, which is all profiling needs).
        cache.layers[0].cumulative_length.fill_(pos)

    prefill_cache = gen.kv.acquire(1, context)

    def prefill_setup() -> None:
        gen.reset_prefix_state()
        _set_fill(prefill_cache, 0)
        # The cold prime compiles the cold-prefill graph — it MUST run inside
        # the production plan-compile window (record-only + grid-shrink
        # globals): a real run-0 of a large chunk OOMs the machine, and the
        # window is what makes the compiled plan the production plan. Same
        # for every other compile-triggering prime below.
        with gen.plan_compile_window(chunk):
            gen.prefill.chunk_step(chunk_ids, prefill_cache, chunk, start_pos=0)  # cold prime

    def prefill_fn() -> object:
        _set_fill(prefill_cache, offset)  # warm chunk attends [0, offset+chunk_len) = depth
        return gen.prefill.chunk_step(chunk_ids, prefill_cache, chunk, start_pos=offset)

    decode_cache = gen.kv.acquire(1, context)
    decode_input = chunk_ids[:, -1:].clone()
    decode_pos = torch.tensor([depth], dtype=torch.int32)

    def decode_setup() -> None:
        gen.reset_prefix_state()
        _set_fill(decode_cache, 0)
        # The prefill pass's visualize ran a dynamo reset, so this prime
        # recompiles the cold graph — window it (see prefill_setup).
        with gen.plan_compile_window(chunk):
            gen.prefill.chunk_step(chunk_ids, decode_cache, chunk, start_pos=0)  # establish state
        _set_fill(decode_cache, depth)  # decode attention scans the full depth

    def decode_fn() -> object:
        return gen.decode.next_token(decode_input, decode_cache, decode_pos)

    passes = [
        CapturePass(
            name="prefill",
            label=f"prefill chunk @ depth {depth}",
            detail=f"one {chunk_len}-token chunk at offset {offset} (prod runs {n_chunks}×{chunk})",
            run=prefill_fn,
            setup=prefill_setup,
            compile_ctx=lambda: gen.plan_compile_window(chunk),
        ),
        CapturePass(
            name="decode",
            label=f"decode (pos={depth})",
            detail="seq_len=1",
            run=decode_fn,
            setup=decode_setup,
            compile_ctx=lambda: gen.plan_compile_window(),  # M=1: record-only alone
        ),
    ]

    # MTP self-speculation: the two extra forwards a spec round runs on top of
    # plain decode (M=2 verify + M=1 draft), so the per-kernel breakdown shows
    # what the round costs and where the acceptance break-even can be lowered.
    if mtp:
        passes.extend(
            _build_mtp_passes(model, gen, depth=depth, chunk=chunk, chunk_ids=chunk_ids)
        )

    # Modality front-ends (vision today; audio / qwen3.5-vision later) each expose
    # their profilable forwards via capture_targets() — same seam `alloy tune`
    # uses. Each becomes its own pass, written to `<model>_<name>.html`.
    if loaded.vision is not None:
        for target in loaded.vision.capture_targets():
            stage = torch.compile(target.module, backend="alloy", dynamic=False)
            passes.append(
                CapturePass(
                    name=target.name,
                    label=target.label,
                    detail="modality front-end (fixed shape)",
                    run=lambda s=stage, i=target.inputs: s(**i),
                    setup=target.setup,
                )
            )

    # Grid-shrunk partial chunk: production prefills ⌊depth/chunk⌋ full chunks
    # then ONE partial chunk whose M-tiled grids shrink to the real remainder —
    # via the eager-compiled PINNED plan (the server's replay path). The
    # compile-path prefill pass above cannot show this (grid shrink is a
    # plan-replay feature), so this pass eager-compiles in its setup, replays
    # the pinned plan, and profiles at the shrunk grid the run dispatched
    # (`plans=[…]` + `_last_grid_shrink_updates`). It MUST stay the LAST pass:
    # eager_compile_all pins plans, after which gen forwards replay them and
    # the earlier passes' `_all_compiled_plans` auto-capture would find nothing.
    rem = depth % chunk
    synthesized = rem == 0
    shrunk_len = rem if rem > 0 else min(chunk // 2, depth)
    shrunk_offset = depth - shrunk_len
    shrunk_m_pad = (shrunk_len + 63) // 64 * 64
    shrunk_ids = torch.randint(
        1024, max(1025, vocab - 256), (1, shrunk_len), generator=g, dtype=torch.long,
    )
    shrunk_cache = gen.kv.acquire(1, context)

    def shrunk_setup() -> None:
        gen.eager_compile_all()
        gen.reset_prefix_state()
        _set_fill(shrunk_cache, 0)
        gen.prefill.chunk_step(chunk_ids, shrunk_cache, chunk, start_pos=0)  # cold prime
        # Late-bind the pinned plan (it only exists after eager_compile_all):
        # warm for a mid-prompt partial chunk, cold when the prompt fits one.
        pinned = gen.plans.prefill_plans.get((chunk, shrunk_offset > 0))
        if pinned is not None:
            shrunk_pass.plans = [pinned[0]]

    def shrunk_fn() -> object:
        _set_fill(shrunk_cache, shrunk_offset)
        return gen.prefill.chunk_step(shrunk_ids, shrunk_cache, chunk, start_pos=shrunk_offset)

    shrunk_pass = CapturePass(
        name="prefill_shrunk",
        label=f"shrunk last chunk @ depth {depth}",
        detail=(
            f"{shrunk_len}-token partial chunk at offset {shrunk_offset}, "
            f"grid shrunk to {shrunk_m_pad} of {chunk}"
            + (" (representative: depth is a chunk multiple)" if synthesized else "")
        ),
        run=shrunk_fn,
        setup=shrunk_setup,
    )
    passes.append(shrunk_pass)

    return ModelCapture(kind="causal", passes=passes)


def _build_mtp_passes(
    model: str, gen: object, *, depth: int, chunk: int, chunk_ids: object
) -> list[CapturePass]:
    """The two extra forwards an MTP spec round runs on top of plain decode: the
    hidden-emitting M=2 verify (full backbone at seq_len=2) and the M=1 draft
    (one MTP block + the shared lm_head). Both profiled at cache `depth` so their
    attention scans the same KV extent as the decode pass — the cost delta vs
    decode is exactly the speculation overhead the round must amortize.
    """
    resolved = resolve_model(model)
    if not isinstance(resolved, ResolvedGGUF):
        raise ValueError(f"{model}: --mtp speculation is only available for GGUF models")
    blob = resolved.path
    try:
        mtp = load_quantized_mtp(gen.model, blob, quantize=True)
    except (StopIteration, KeyError) as exc:  # no full-attention layer / no mtp.* tensors
        raise ValueError(f"{model} ships no MTP head — --mtp is not applicable") from exc
    mtp.bind_runtime(
        gen.model.model.embed_tokens,
        gen.model.model.rotary_emb,
        (1.0 + gen.model.model.norm.weight).detach(),
    )
    hidden = int(gen.model.config.hidden_size)
    context = gen.max_cache_len
    prefill_len = min(chunk, depth)

    # --- verify (M=2): full backbone at seq_len=2, emits (argmax, hidden). Primed
    # like the decode pass (one cold chunk lays down DeltaNet + KV state, then
    # cumulative_length drives the attention scan to `depth`). ---
    verify = torch.compile(VerifySpecLogits(gen.model, (), True), backend="alloy", dynamic=False)
    vcache = gen.kv.acquire(1, context)
    v_in = chunk_ids[:, -1:].repeat(1, 2).contiguous()  # [t, t]
    v_pos = torch.tensor([depth, depth + 1], dtype=torch.int32)

    def verify_setup() -> None:
        gen.reset_prefix_state()
        vcache.layers[0].cumulative_length.fill_(0)
        set_deltanet_attn_mask(vcache, torch.ones((1, prefill_len), dtype=torch.long))
        # Compile-triggering prime → production window (see prefill_setup).
        with gen.plan_compile_window(chunk):
            gen.prefill.chunk_step(chunk_ids, vcache, chunk, start_pos=0)
        vcache.layers[0].cumulative_length.fill_(depth)
        set_deltanet_attn_mask(vcache, torch.ones((1, 2), dtype=torch.long))

    def verify_fn() -> object:
        # Spec globals are read at trace (first run) and baked into the plan; set
        # them around the call and reset so a following pass starts clean.
        set_use_alloy_warm_op(True)
        compile_window.spec_save_steps = True
        compile_window.q_start_pos = depth
        try:
            return verify(input_ids=v_in, past_key_values=vcache, cache_position=v_pos)
        finally:
            set_use_alloy_warm_op(False)
            compile_window.spec_save_steps = False
            compile_window.q_start_pos = 0

    # --- draft (M=1): one MTP full-attention block + the shared lm_head, over the
    # MTP's own single-layer cache. cache_position=depth makes its attention scan
    # `depth` slots (content-independent cost, same trick as decode). ---
    draft = torch.compile(MTPDraftStep(mtp), backend="alloy", dynamic=False)
    mcache = AlloyStaticCache(
        config=mtp.cache_config, max_cache_len=context,
        max_batch_size=1, cache_dtype=gen.cache_dtype,
    )
    dpos = torch.tensor([[depth]], dtype=torch.long)
    dcp = torch.tensor([depth], dtype=torch.int32)
    te = torch.zeros((1, 1, hidden), dtype=gen.cache_dtype)
    hid = torch.zeros((1, 1, hidden), dtype=gen.cache_dtype)
    # Real partial-rope cos/sin for position `depth`, straight from the model's
    # rotary_emb (gives the correct (1, 1, rotary_dim) shape — no head-dim math).
    cos, sin = gen.model.model.rotary_emb(te, dpos)
    cos = cos.to(gen.cache_dtype).contiguous()
    sin = sin.to(gen.cache_dtype).contiguous()

    def draft_setup() -> None:
        mcache.reset()

    def draft_fn() -> object:
        return draft(te, hid, cos, sin, dpos, mcache, dcp)

    return [
        CapturePass(
            name="mtp_verify",
            label=f"mtp verify M=2 (pos={depth})",
            detail="spec-verify: full backbone at seq_len=2, emits hidden",
            run=verify_fn,
            setup=verify_setup,
            compile_ctx=lambda: gen.plan_compile_window(),  # M=2: record-only alone
        ),
        CapturePass(
            name="mtp_draft",
            label=f"mtp draft M=1 (pos={depth})",
            detail="one MTP block + shared lm_head",
            run=draft_fn,
            setup=draft_setup,
            compile_ctx=lambda: gen.plan_compile_window(),  # M=1: record-only alone
        ),
    ]


@dataclass
class KernelDispatch:
    """A captured kernel dispatch ready to replay (timing / GPU capture).

    `captured` is the runtime tuner's `CapturedKernel` (KernelFunction +
    production-resolved constexprs + real input buffers). `name` is its resolved
    kernel name; `pass_name` records which forward (prefill / decode) it came from.
    """

    name: str
    pass_name: str
    captured: object  # alloy._runtime.tune.CapturedKernel


def capture_kernel_dispatch(
    model: str,
    kernel: str,
    *,
    depth: int = _DEFAULT_DEPTH,
    decode: bool = False,
    mtp: bool = False,
) -> tuple[list[KernelDispatch], list[str]]:
    """Capture the production dispatch(es) of `kernel` for `model`, ready to replay.

    Drives the model's real chunked-prefill forward at cache `depth` (or the M=1
    decode forward when `decode=True`) — the SAME warm-offset path `alloy tune`
    and `alloy inspect` use — and captures the target kernel with its
    production-resolved constexprs and real input buffers. This is what lets
    `alloy microbench` / `alloy profile --capture` replay the EXACT kernel the
    server runs with zero hand-typed constexprs or buffer shapes.

    Returns `(matches, all_names)`: `matches` are the captured dispatches whose
    name matches `kernel` (exact, else substring); `all_names` is every kernel the
    forward dispatched (for a useful "not found — kernels seen: …" error).
    """
    loaded = load_native_causal_lm(model)
    net = loaded.model.eval()
    context = int(net.config.max_position_embeddings)
    # from_model installs the multi-token-attention patch + wires chunked prefill,
    # so the captured forward exercises the exact production dispatch path. The
    # server's chunk policy matters: without it the default 128-chunk makes
    # `alloy microbench` time an M=128 dispatch that production (4096-chunk)
    # never runs at depths > 128.
    gen = AlloyGenerator.from_model(net, chunk_prefill_size=resolve_prefill_policy())
    chunk = gen.chunk_prefill_size
    if not (1 <= depth < context):
        raise ValueError(f"--depth {depth} must be in [1, {context}) for {model}")
    cfg = net.config

    if decode:
        # Decode kernels (M=1) only appear after state exists. Establish DeltaNet
        # recurrent + KV state with one compiled prefill, then capture the M=1
        # forward — matching `alloy tune`'s decode setup exactly.
        decode_cache = AlloyStaticCache(
            cfg, max_cache_len=context, cache_dtype=torch.float16,
            kv_format=resolve_kv_format(None),
        )
        compiled = torch.compile(net, backend="alloy", dynamic=False)
        with torch.inference_mode(), gen.plan_compile_window(chunk):
            # Window: a REAL run-0 of a full chunk holds every forward
            # intermediate live at once (100+ GB on a 35B MoE) — the prime
            # only needs the compile + python state side effects, not KV
            # contents (capture timing is content-independent).
            compiled(
                input_ids=torch.zeros((1, chunk), dtype=torch.long),
                position_ids=torch.arange(chunk).unsqueeze(0),
                cache_position=torch.arange(chunk, dtype=torch.int32),
                past_key_values=decode_cache,
                use_cache=True,
            )
        inputs = {
            "input_ids": torch.zeros((1, 1), dtype=torch.long),
            "position_ids": torch.tensor([[chunk]]),
            "cache_position": torch.tensor([chunk], dtype=torch.int32),
            "past_key_values": decode_cache,
            "use_cache": True,
        }
        pass_name = "decode"
    elif mtp:
        # MTP self-speculation verify: the M=2 backbone forward at cache `depth` —
        # seq_len=2 routes the projections + lm_head to the `_v2_rows` GEMVs and
        # full-attention to the multi-token kernel, under the K=2 DeltaNet mask +
        # spec globals the verify runs in production. Prime DeltaNet/KV like decode.

        cache = AlloyStaticCache(cfg, max_cache_len=context, cache_dtype=torch.float16)
        compiled = torch.compile(net, backend="alloy", dynamic=False)
        set_deltanet_attn_mask(cache, torch.ones((1, chunk), dtype=torch.long))
        with torch.inference_mode(), gen.plan_compile_window(chunk):
            # Window: see the decode prime above — compile + state only.
            compiled(
                input_ids=torch.zeros((1, chunk), dtype=torch.long),
                position_ids=torch.arange(chunk).unsqueeze(0),
                cache_position=torch.arange(chunk, dtype=torch.int32),
                past_key_values=cache,
                use_cache=True,
            )
        cache.layers[0].cumulative_length.fill_(depth)  # verify attention scans full depth
        set_deltanet_attn_mask(cache, torch.ones((1, 2), dtype=torch.long))
        set_use_alloy_warm_op(True)
        compile_window.spec_save_steps = True
        compile_window.q_start_pos = depth
        inputs = {
            "input_ids": torch.zeros((1, 2), dtype=torch.long),
            "position_ids": torch.tensor([[depth, depth + 1]]),
            "cache_position": torch.tensor([depth, depth + 1], dtype=torch.int32),
            "past_key_values": cache,
            "use_cache": True,
        }
        pass_name = "mtp_verify"
    else:
        # Prefill: the chunk shape generation dispatches, at a representative WARM
        # offset (depth − chunk). attention_strided_runtime_pos's causal early-exit
        # clamps the K-scan to the offset, so the warm offset benchmarks the real
        # large-extent attention — capturing at offset 0 would replay a tiny K-loop.
        chunk_len = min(chunk, depth)
        offset = depth - chunk_len
        compile_window.q_start_pos = offset
        cache = AlloyStaticCache(cfg, max_cache_len=context, cache_dtype=torch.float16)
        inputs = {
            "input_ids": torch.zeros((1, chunk_len), dtype=torch.long),
            "position_ids": torch.arange(offset, offset + chunk_len).unsqueeze(0),
            "cache_position": torch.arange(offset, offset + chunk_len, dtype=torch.int32),
            "past_key_values": cache,
            "use_cache": True,
        }
        pass_name = "prefill"

    with torch.inference_mode():
        if pass_name == "prefill":
            # The prefill capture forward is the explosive one (chunk-sized
            # run-0): capture record-only inside the production window, so
            # memory stays bounded AND the resolved kernels are the production
            # (grid-shrink-mode) variants. Phantom intermediates materialize
            # zero-filled on first replay (`dispatch_captured`). The decode /
            # mtp captures above stay real — M=1/M=2 intermediates are tiny
            # and real buffers replay with realistic contents.
            with gen.plan_compile_window(chunk_len):
                captured = capture_kernel_for_replay(net, inputs, record_only=True)
        else:
            captured = capture_kernel_for_replay(net, inputs)

    all_names = sorted({c.name for c in captured})
    matches = [KernelDispatch(name=c.name, pass_name=pass_name, captured=c) for c in captured if c.name == kernel]
    if not matches:
        matches = [
            KernelDispatch(name=c.name, pass_name=pass_name, captured=c)
            for c in captured
            if kernel.lower() in c.name.lower()
        ]
    return matches, all_names


def _build_embedder_capture(
    model: str, *, batch: int | None, seq: int | None
) -> ModelCapture:
    nf = build_nomic_forward(model)
    ceiling = nf.meta.context_length
    vocab = nf.meta.vocab_size

    if batch is None and seq is None:
        shapes = [("short", *_EMBED_SHORT_SHAPE), ("long", *_EMBED_LONG_SHAPE)]
    else:
        b = batch if batch is not None else _DEFAULT_EMBED_BATCH
        s = seq if seq is not None else _DEFAULT_EMBED_SEQ
        shapes = [(f"b{b}_s{s}", b, s)]

    passes: list[CapturePass] = []
    for name, b, s in shapes:
        if s > ceiling:
            raise ValueError(f"seq {s} exceeds model context length {ceiling}")
        # Content is irrelevant to kernel shape/timing; the mask is all-ones so
        # attention runs the full seq×seq cost (the long-seq case we profile),
        # never a padded short-circuit. Pinned for a static compiled plan.
        input_ids = (torch.arange(b * s, dtype=torch.long) % max(1, vocab - 1)).reshape(b, s)
        input_ids = input_ids.contiguous()
        attention_mask = torch.ones((b, s), dtype=torch.long)
        for pinned in (input_ids, attention_mask):
            torch._dynamo.mark_static_address(pinned)

        def run(ids: object = input_ids, mask: object = attention_mask) -> object:
            return nf.forward(input_ids=ids, attention_mask=mask)

        passes.append(
            CapturePass(
                name=name,
                label=f"{name} (batch={b}, seq={s})",
                detail=f"batch={b}, seq={s}",
                run=run,
            )
        )

    return ModelCapture(kind="embedder", passes=passes)
