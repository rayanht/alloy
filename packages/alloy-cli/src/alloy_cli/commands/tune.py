"""`alloy tune <model>` — run the offline tuner; writes optimal configs.

Tunes every ``@al.tunable`` kernel at the shapes generation actually dispatches
under **chunked prefill**: the chunk shape (M = chunk size, default 128) for the
prefill pass and M = 1 for decode. Results merge into the per-device user config
(``$ALLOY_CONFIG_DIR`` or ``~/.local/share/alloy/configs/{device}.json``, device
auto-detected), which serving overlays on the package-shipped baseline.

This is deliberately NOT a prefill-bucket sweep — that approach is obsolete.
Generation prefills in fixed-size chunks (``AlloyGenerator(chunk_prefill_size=…)``),
so a model only ever hits two M dimensions: the chunk size and 1. Tuning those
two shapes is sufficient and ~8x cheaper than the old per-bucket sweep.
"""

from __future__ import annotations

import os
from typing import Annotated

import torch
import typer
from rich.console import Console

import alloy
from alloy._runtime.device import detect_device
from alloy._runtime.tune_configs import GRID_SHRINK_REP_M, user_config_file
from alloy_torch.compile_window import compile_window
from alloy_server.cache import AlloyStaticCache
from alloy_server.generation.generator import AlloyGenerator
from alloy_server.generation.spec import VerifySpecLogits
from alloy_server.gguf import ResolvedGGUF
from alloy_server.kv_format import resolve_kv_format
from alloy_server.models import load_native_causal_lm, model_kind, resolve_model
from alloy_server.models.attention import (
    set_deltanet_attn_mask,
    set_taps_enabled,
    set_use_alloy_warm_op,
)
from alloy_server.models.whisper import tune_whisper
from alloy_server.speculative.dflash import DFlashDrafter, resolve_dflash_checkpoint
from alloy_server.speculative.mtp import MTPDrafter
from alloy_server.speculative.pld import PromptLookupDrafter

console = Console()

# Generation's default chunked-prefill chunk size (AlloyGenerator chunk_prefill).
_DEFAULT_CHUNK = 4096

# Representative WARM cache offset the prefill attention is benchmarked at. The
# runtime-pos attention's causal early-exit makes its cost scale with the cache
# offset, but the tile config saturates past a few thousand K-positions: the
# config optimal at an 8K-position K-loop is optimal at native (262K), and
# benchmarking the native offset is intractable (~7ms/dispatch × 20 runs × N
# configs). The cache is still allocated at native, so the shape key (KV_LEN) is
# unchanged; only the benchmarked early-exit trip count shrinks.
_WARM_TUNE_OFFSET = 4096


def tune(
    model: Annotated[str, typer.Argument(help="model name, e.g. qwen3.5:4b")],
    only: Annotated[
        str | None,
        typer.Option(help="regex: tune only kernels whose name matches (re-tune one subsystem)"),
    ] = None,
    output: Annotated[
        str | None,
        typer.Option(
            help="write tuned configs to this file instead of the default "
            "per-device path ($ALLOY_CONFIG_DIR or "
            "~/.local/share/alloy/configs/{device}.json)"
        ),
    ] = None,
    chunk: Annotated[
        int, typer.Option(help="chunked-prefill chunk size to tune the prefill pass at")
    ] = _DEFAULT_CHUNK,
    skip_decode: Annotated[
        bool, typer.Option("--skip-decode", help="tune only the prefill chunk shape")
    ] = False,
    shrink_max: Annotated[
        int,
        typer.Option(
            help="also tune the grid-shrink chunk prefill forward (0 = skip). Pass the "
            "deployment chunk size (e.g. 4096); the tuner benchmarks at the "
            "representative GRID_SHRINK_REP_M (tile configs are M-saturated, and a native-M "
            "tune is intractable — its cold attention is O(M^2)), and the native "
            "shrink-capable plan caps its M-scaled config keys down to that at resolution."
        ),
    ] = _DEFAULT_CHUNK,
    only_shrink: Annotated[
        bool,
        typer.Option(
            "--only-shrink",
            help="tune ONLY the grid-shrink chunk shape (skip the small-chunk and "
            "decode steps). Use to add shrink-chunk configs to a model whose chunk/decode "
            "configs are already tuned — those are stable, so re-tuning them is waste. "
            "Requires --shrink-max.",
        ),
    ] = False,
    kv_quant: Annotated[
        str | None,
        typer.Option(
            "--kv-quant",
            help="tune with the quantized KV cache active (e.g. q8_0) so the "
            "forwards dispatch — and the tuner benchmarks — the q8 attention "
            "variants production serves. Exported as ALLOY_KV_QUANT.",
        ),
    ] = None,
    spec: Annotated[
        str | None,
        typer.Option(
            "--spec",
            help="also tune the speculative-decoding forwards for this drafter "
            "(dflash | mtp | pld)"
        ),
    ] = None,
    only_spec: Annotated[
        bool,
        typer.Option(
            "--only-spec",
            help="tune ONLY the speculative-decoding shapes (skip the chunk, "
            "decode, and shrink steps). Use to add/refresh spec configs on a "
            "model whose generation configs are already tuned — those are "
            "stable, so re-tuning them is waste. Requires --spec.",
        ),
    ] = False,
) -> None:
    """Tune a model's kernels at the chunked-prefill (M=chunk) and decode (M=1) shapes.

    Always tunes at the model's native context (``max_position_embeddings``) — the
    only context worth tuning at. Prefill attention is keyed by the cache length,
    native-max KV allocation is free (demand-paged, so it costs only the pages
    actually filled), and decode is context-robust (no KV_LEN specialization), so
    production allocates and deploys at native. A sub-native tune has no use: its
    configs wouldn't match the native deployment.
    """
    if kv_quant is not None:
        # Resolved at cache construction; an unknown name raises with the
        # available set, never silently tuning fp16.
        os.environ["ALLOY_KV_QUANT"] = kv_quant

    only_kw = {"only": only} if only else {}
    out_kw = {"output": output} if output else {}

    config_dest = output or str(user_config_file(detect_device()))

    if only_shrink and not shrink_max:
        raise typer.BadParameter("--only-shrink requires --shrink-max <M>")
    if only_spec and not spec:
        raise typer.BadParameter("--only-spec requires --spec <drafter>")

    # Transcription (whisper) isn't a CausalLM — it tunes the encoder + decoder
    # prefill/decode forwards (HF generate owns the loop), not the chunked-prefill
    # backbone. Branch before load_native_causal_lm, which only builds CausalLMs.
    resolved = resolve_model(model)
    if model_kind(resolved.architecture()) == "transcription":
        console.print(f"[bold]tuning[/] {model} (whisper) — encoder + decoder prefill/decode")
        tune_whisper(resolved.location, only=only, output=output)
        console.print(f"[bold green]tuned[/] — written to {config_dest}")
        return

    loaded = load_native_causal_lm(model)
    net = loaded.model.eval()
    # Native context (max_position_embeddings); the GGUF loader maps
    # `<arch>.context_length` onto this field.
    context = int(net.config.max_position_embeddings)
    label = f"only={only!r}" if only else "all kernels"
    console.print(
        f"[bold]tuning[/] {model} @ context={context} — {label}, "
        f"prefill M={chunk} + decode M=1"
    )
    # Configure the model as production does (multi-token attention patch for
    # hybrid/gated-attention models, chunked prefill) so the tuner's forward
    # exercises the real dispatch path. Only the in-place side effect on `net`
    # is kept; the generator object itself is unused here.
    AlloyGenerator.from_model(net, chunk_prefill_size=chunk)
    cfg = net.config
    # Honor ALLOY_KV_QUANT: the tune forwards must dispatch the SAME kernels
    # production serves — a quantized KV deployment tunes the q8 attention
    # variants, not the fp16 ones.
    kv_format = resolve_kv_format(None)
    set_use_alloy_warm_op(False)

    # 1) Prefill: the single chunk shape generation dispatches, benchmarked at a
    # representative WARM offset. Chunked prefill runs this chunk shape at cache
    # offsets 0..(context-chunk) through `attention_strided_runtime_pos`, whose
    # causal early-exit clamps the K-scan to Q_START_POS+chunk, so its cost scales
    # with the offset. Tuning at offset 0 shrinks the benchmarked K-loop to ~chunk
    # positions, so the tuner picks tile sizes optimal for a tiny attention (8x8)
    # that are ~8x too slow at the warm offsets real prefill hits. We tune at
    # _WARM_TUNE_OFFSET — past the K-length where the tile config saturates — so
    # the pick matches the native offset but stays cheap to benchmark. The cache
    # is native-sized, so KV_LEN (the shape key) is unchanged; only the runtime
    # early-exit trip count shrinks. GEMMs/norms are offset-independent (M=chunk).
    #
    # record_only: the extraction forward only needs SHAPES — the per-kernel
    # benchmark synthesizes its own buffers. The offset lives in a runtime
    # Q_START_POS_BUF, lost when the forward is skipped, so re-inject it via
    # q_start_pos so the attention benchmarks the warm extent.
    if not only_shrink and not only_spec:
        warm_pos = min(_WARM_TUNE_OFFSET, context - chunk)
        compile_window.q_start_pos = warm_pos
        cache = AlloyStaticCache(cfg, max_cache_len=context, cache_dtype=torch.float16, kv_format=kv_format)
        # logits_to_keep=1: production chunked prefill also keeps only the last
        # token (lm_head at M=1, tuned by the decode pass), so don't tune a phantom
        # (chunk, vocab) lm_head. The backbone still tunes at M=chunk.
        alloy.tune(
            net,
            {
                "input_ids": torch.zeros((1, chunk), dtype=torch.long),
                "position_ids": torch.arange(warm_pos, warm_pos + chunk).unsqueeze(0),
                "cache_position": torch.arange(warm_pos, warm_pos + chunk, dtype=torch.int32),
                "past_key_values": cache,
                "use_cache": True,
                "logits_to_keep": 1,
            },
            record_only=True,
            q_start_pos=warm_pos,
            **only_kw, **out_kw,
        )
        console.print(f"  [green]done[/] prefill (M={chunk}, warm offset={warm_pos})")

    # 2) Decode: a single token continuing from populated cache state. Run one
    # prefill forward first so the DeltaNet conv/recurrent + KV state exist and
    # `has_previous_state` is set, then the tuner's M=1 forward takes the decode
    # branch (not the cold single-token path).
    if not skip_decode and not only_shrink and not only_spec:
        decode_cache = AlloyStaticCache(cfg, max_cache_len=context, cache_dtype=torch.float16, kv_format=kv_format)
        compiled = torch.compile(net, backend="alloy", dynamic=False)
        with torch.inference_mode():
            compiled(
                input_ids=torch.zeros((1, chunk), dtype=torch.long),
                position_ids=torch.arange(chunk).unsqueeze(0),
                cache_position=torch.arange(chunk, dtype=torch.int32),
                past_key_values=decode_cache,
                use_cache=True,
            )
        alloy.tune(
            net,
            {
                "input_ids": torch.zeros((1, 1), dtype=torch.long),
                "position_ids": torch.tensor([[chunk]]),
                "cache_position": torch.tensor([chunk], dtype=torch.int32),
                "past_key_values": decode_cache,
                "use_cache": True,
            },
            **only_kw, **out_kw,
        )
        console.print("  [green]done[/] decode (M=1)")

    # 3) Grid-shrink chunk prefill: the production large chunk in a SINGLE cold
    # forward. The shrink-capable plan compiles at SEQ_LEN=chunk, so its GEMM /
    # attention / norm / DeltaNet kernels resolve configs by the large-chunk shape —
    # distinct from the M=chunk entries above. Without this the large-chunk
    # dispatch falls back to untuned configs despite the larger, more efficient
    # GEMMs.
    #
    # We tune at the REPRESENTATIVE M (GRID_SHRINK_REP_M), NOT the deployment M_MAX:
    # tile configs are M-saturated past a few thousand rows (M=16384 == M=262144
    # configs for 3/4 GEMM shapes), and tuning at native M_MAX is intractable — the
    # cold single-pass attention is O(M^2) (~1hr+/shape at 262144). A larger-chunk
    # plan caps its M-scaled config keys down to GRID_SHRINK_REP_M at resolution,
    # so it resolves against this tune.
    if shrink_max and not only_spec:
        m = (min(shrink_max, context, GRID_SHRINK_REP_M) // 64) * 64
        compile_window.q_start_pos = 0
        # Force single-pass attention — the branch the shrink-capable dispatch
        # uses (the exact-grid path forces single-pass for shrinkability);
        # otherwise the tuner benchmarks the split-K kernels the shrink-capable
        # plan never runs, and the single-pass attention falls back to untuned
        # configs. NOT shrink_m: the tuner WRITES configs, it doesn't resolve, so
        # it must not couple in the resolve cap / bounded pool.
        compile_window.single_pass_attention = True
        os_cache = AlloyStaticCache(cfg, max_cache_len=context, cache_dtype=torch.float16, kv_format=kv_format)
        try:
            # logits_to_keep=1 — match production (`ChunkPrefill`): chunk prefill
            # keeps ONLY the last token's logits, so the lm_head runs at M=1
            # (already tuned by the decode pass), NOT M=M_MAX. Without this the
            # tuner benchmarks a phantom (M_MAX, vocab) lm_head GEMM production
            # never dispatches — at M_MAX=16384, vocab≈248k that's a 16 GB output
            # and ~580s/shape, the dominant tune cost. The backbone still runs at
            # M=M_MAX (logits_to_keep only slices the final lm_head).
            # record_only: the shape-extraction forward at M=M_MAX would hold
            # O(M_MAX × layers) of activations and OOM; it captures every kernel's
            # shape with phantom intermediates and no GPU dispatch, while the
            # per-kernel benchmark still runs real.
            alloy.tune(
                net,
                {
                    "input_ids": torch.zeros((1, m), dtype=torch.long),
                    "position_ids": torch.arange(0, m).unsqueeze(0),
                    "cache_position": torch.arange(0, m, dtype=torch.int32),
                    "past_key_values": os_cache,
                    "use_cache": True,
                    "logits_to_keep": 1,
                },
                record_only=True,
                # Cold shrink chunk (start_pos=0); pin the runtime offset so the
                # single-pass attention benchmarks the cold extent explicitly,
                # not record_only's dropped-snapshot default.
                q_start_pos=0,
                **only_kw, **out_kw,
            )
        finally:
            compile_window.single_pass_attention = False
        console.print(f"  [green]done[/] grid-shrink chunk prefill (M={m}, cold, single-pass)")

    # 4) Speculative decoding: the M=block verify forward + the drafter's own
    # modules. The verify runs with the production flags (warm op, SAVE_STEPS,
    # taps) against a primed cache at a warm offset, so the dot_*_v2_rows quant
    # kernels and the SAVE_STEPS recurrent tune at the verify shapes; the drafter
    # contributes its propose/observe forwards via tune_targets().
    if spec:
        gen = AlloyGenerator.from_model(net, chunk_prefill_size=chunk)
        if spec == "pld":
            drafter = PromptLookupDrafter()
        elif spec in ("mtp", "dflash"):
            if not isinstance(resolved, ResolvedGGUF):
                raise typer.BadParameter(f"--spec {spec} is only available for GGUF models")
            if spec == "mtp":
                drafter = MTPDrafter(resolved.path)
            else:
                drafter = DFlashDrafter(resolve_dflash_checkpoint(model), resolved.path)
        else:
            raise typer.BadParameter(f"unknown --spec {spec!r}; known: dflash, mtp, pld")
        gen.attach_spec(drafter)
        b = 1 + drafter.max_draft_tokens
        # Prime a cache (one chunk forward) so the verify takes the warm path,
        # then tune the verify forward at M=b — mirroring generation.spec.pin_verify_plan.
        spec_cache = AlloyStaticCache(cfg, max_cache_len=context, cache_dtype=torch.float16, kv_format=kv_format)
        compiled = torch.compile(net, backend="alloy", dynamic=False)
        with torch.inference_mode():
            compiled(
                input_ids=torch.zeros((1, chunk), dtype=torch.long),
                position_ids=torch.arange(chunk).unsqueeze(0),
                cache_position=torch.arange(chunk, dtype=torch.int32),
                past_key_values=spec_cache,
                use_cache=True,
            )
        verify = VerifySpecLogits(net, drafter.taps.layer_ids, drafter.taps.post_norm)
        set_deltanet_attn_mask(spec_cache, torch.ones((1, b), dtype=torch.long))
        set_use_alloy_warm_op(True)
        compile_window.spec_save_steps = True
        set_taps_enabled(bool(drafter.taps.layer_ids))
        compile_window.q_start_pos = chunk
        try:
            alloy.tune(
                verify,
                {
                    "input_ids": torch.zeros((1, b), dtype=torch.long),
                    "past_key_values": spec_cache,
                    "cache_position": torch.arange(chunk, chunk + b, dtype=torch.int32),
                },
                **only_kw, **out_kw,
            )
        finally:
            set_use_alloy_warm_op(False)
            compile_window.spec_save_steps = False
            set_taps_enabled(False)
            compile_window.q_start_pos = 0
        console.print(f"  [green]done[/] spec verify (M={b}, {spec})")
        tune_targets = drafter.tune_targets() if spec == "dflash" else []
        for spec_label, module, inputs in tune_targets:
            console.print(f"  [cyan]tuning[/] {spec_label} …")
            alloy.tune(module, inputs, **only_kw, **out_kw)
            console.print(f"  [green]done[/] {spec_label}")

    # Modality front-ends expose their tunable forwards via capture_targets().
    # Their shapes are fixed and context-independent (a ViT attends its full
    # patch count, not the text KV-context), so they tune once and merge by shape.
    if loaded.vision is not None:
        for target in loaded.vision.capture_targets():
            if target.setup is not None:
                target.setup()
            console.print(f"  [cyan]tuning[/] {target.label} …")
            alloy.tune(target.module, target.inputs, **only_kw, **out_kw)
            console.print(f"  [green]done[/] {target.label}")

    console.print(f"[bold green]tuned[/] — written to {config_dest}")
