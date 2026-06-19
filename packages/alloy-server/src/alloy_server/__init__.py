"""Alloy LLM inference + serving stack: GGUF loading, the batch-1 generation
engine, speculative decoding, and the OpenAI/Ollama/Anthropic-compatible server.

Builds on the alloy_torch backend; alloy_torch never imports this package.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING, ContextManager, cast

# Suppress transformers load-noise + optional sklearn/scipy imports before any
# model class loads (import side effect; see _compat). Kept first so it runs
# ahead of the alloy_server.models import below.
import alloy_server.compat  # noqa: F401
import torch
import transformers
# Register the "alloy" torch.compile backend + the alloy.* custom ops (gguf_*_mm,
# sample_categorical, …) BEFORE any moved submodule loads: gguf.quant and others
# read `torch.ops.alloy.*` at import time. Every `import alloy_server.*` runs this
# __init__ first, so registering here covers all entry points.
import alloy_torch.backend  # noqa: F401  registers the alloy torch.compile backend
import alloy_torch.custom_ops  # noqa: F401  registers the alloy.* custom ops

from alloy import get_logger
from alloy_server.constrain import Constraint, GrammarFactory, xgrammar_tool_format
from alloy_server.gguf import ResolvedGGUF
from alloy_server.models import (
    ResolvedModel,
    check_arch_supported,
    load_resolved_causal_lm,
    model_kind,
    resolve_model,
)
from alloy_server.models.modality import ModalityEncoder
from alloy_server.reasoning import (
    ReasoningProtocol,
    resolve_reasoning_protocol,
)
from alloy_server.schema import (
    ChatMessage,
    TokenDecoder,
    TokenGenerator,
    TokenStreamer,
    ChatTokenizer,
    TextTokenCounter,
    NativeModelLoader,
    NativeGeneratorBuilder,
    MultimodalEncoder,
    MultimodalGenerator,
    MultimodalStreamer,
    ApplySampling,
    ServedModel,
    RequestError,
)
from alloy_server.session import chat_template_extras
from alloy_server.runner import ModelRunner, GenerationWorker
from alloy_server.transport import AlloyServer, PortCollisionError
from alloy_server.modality import TRANSCRIPTION, Modality
from alloy_server.generation.generator import AlloyGenerator
from alloy_server.generation.sequence import MultimodalInputs, SamplingParams, Sequence
from alloy_server.discover import discover_all
from alloy_server.speculative.mtp import MTPDrafter
from alloy_server.speculative.pld import PromptLookupDrafter
from alloy_server.speculative.dflash import DFlashDrafter, resolve_dflash_checkpoint

if TYPE_CHECKING:
    from alloy_server.embedding import EmbeddingModel

logger = get_logger("alloy_server")


@dataclass(frozen=True, slots=True)
class ServerConfig:
    model: str | None
    hf_id: str | None
    host: str
    port: int
    allow_downloads: bool
    spec: str | None = None
    force: bool = False


def create_server(
    host: str,
    port: int,
    *,
    served: object | None = None,
    modality: "Modality | None" = None,
    chat_model: ServedModel | None = None,
    embedding_model: "EmbeddingModel | None" = None,
    spec: str | None = None,
) -> AlloyServer:
    def installed_chat_names() -> tuple[str, ...]:
        # The served chat model (if any) first, then GGUF models discovered on disk
        # (Ollama store / HF cache) — a catalog listing for client model pickers,
        # not a load surface. `discovered_chat_names` is looked up dynamically so
        # tests can monkeypatch it.
        served_names = (chat_model.name,) if chat_model is not None else ()
        seen = set(served_names)
        discovered: list[str] = []
        for entry in discovered_chat_names():
            if entry not in seen:
                discovered.append(entry)
                seen.add(entry)
        return served_names + tuple(discovered)

    return AlloyServer(
        host, port, served=served, modality=modality,
        chat_model=chat_model, embedding_model=embedding_model,
        spec=spec, installed_chat_names=installed_chat_names,
    )


def create_generation_served_model(
    name: str,
    encode_messages: ChatTokenizer,
    decode: TokenDecoder,
    generate: TokenGenerator,
    stream_token_ids: TokenStreamer,
    count_tokens: TextTokenCounter,
    tokenizer: transformers.PreTrainedTokenizerBase | None = None,
    reset_prefix_state: Callable[[], None] | None = None,
    preserve_context: "Callable[[int], ContextManager] | None" = None,
    eos_token_ids: frozenset[int] = frozenset(),
    apply_sampling: ApplySampling | None = None,
    last_timings: Callable[[], dict] | None = None,
    reasoning: ReasoningProtocol | None = None,
    encode_multimodal: MultimodalEncoder | None = None,
    generate_multimodal: MultimodalGenerator | None = None,
    stream_multimodal_ids: MultimodalStreamer | None = None,
) -> ServedModel:
    runner = ModelRunner(
        name=name, encode_messages=encode_messages, decode=decode, generate=generate,
        stream_token_ids=stream_token_ids, count_tokens=count_tokens, tokenizer=tokenizer,
        reset_prefix_state=reset_prefix_state, preserve_context=preserve_context,
        eos_token_ids=eos_token_ids, apply_sampling=apply_sampling, last_timings=last_timings,
        reasoning=reasoning, encode_multimodal=encode_multimodal,
        generate_multimodal=generate_multimodal, stream_multimodal_ids=stream_multimodal_ids,
    )
    worker = GenerationWorker(runner)
    return ServedModel(
        name=name, complete=worker.complete, stream=worker.stream, count_tokens=count_tokens,
        apply_sampling=apply_sampling, reasoning=reasoning,
    )


def chat_message_dict(m: ChatMessage) -> dict:
    """Render a ChatMessage to the dict shape apply_chat_template expects,
    including assistant tool_calls so multi-turn tool conversations round-trip."""
    d: dict = {"role": m.role, "content": m.content}
    if m.tool_calls:
        d["tool_calls"] = [
            {"type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}
            for tc in m.tool_calls
        ]
    return d


def chat_message_dict_mm(m: ChatMessage) -> dict:
    """Like `chat_message_dict` but renders content as a parts list when the
    message carries images/audio, so the chat template emits one placeholder per
    attachment (images then audio, in order) ahead of the text."""
    if not m.images and not m.audio:
        return chat_message_dict(m)
    content: list[dict] = [{"type": "image"} for _ in m.images]
    content += [{"type": "audio"} for _ in m.audio]
    if m.content:
        content.append({"type": "text", "text": m.content})
    return {"role": m.role, "content": content}


def build_multimodal_hooks(
    vision: ModalityEncoder | None,
    audio: ModalityEncoder | None,
    tokenizer: transformers.PreTrainedTokenizerBase,
    generator: AlloyGenerator,
) -> "tuple[MultimodalEncoder, MultimodalGenerator, MultimodalStreamer]":
    """Build the (encode, generate, stream) hooks for multimodal requests. `encode`
    runs each modality's tower on its attachments, renders the chat with one
    placeholder per attachment, asks each adapter to expand its placeholders into
    soft-token runs, and returns (input_ids, features ordered by text position,
    placeholder positions). Model-agnostic: gemma4 ships both a vision and an audio
    front-end, so either or both may be present."""

    def encode_multimodal(
        messages: tuple[ChatMessage, ...], enable_thinking: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_feats = (
            [vision.encode(i) for m in messages for i in m.images] if vision else []
        )
        aud_feats = (
            [audio.encode(a) for m in messages for a in m.audio] if audio else []
        )
        msgs = [chat_message_dict_mm(m) for m in messages]
        extras = dict(chat_template_extras(tokenizer, enable_thinking))
        try:
            text = cast(str, tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, **extras,
            ))
            if vision is not None and img_feats:
                text = vision.expand_text(text, img_feats)
            if audio is not None and aud_feats:
                text = audio.expand_text(text, aud_feats)
        except Exception as exc:
            raise RequestError(
                HTTPStatus.BAD_REQUEST, "invalid_request",
                f"could not encode multimodal request: {exc}",
            ) from exc
        enc = tokenizer(text, add_special_tokens=False)["input_ids"]
        row = enc[0] if enc and isinstance(enc[0], list) else enc
        input_ids = torch.tensor([row], dtype=torch.long)
        # Order features by the text position of their placeholder runs: walk the
        # token row and, at each maximal run of an image/audio token, pop the next
        # feature from that modality (expand_text sized each run to its feature
        # rows). This keeps `features[k]` aligned with the k-th placeholder slot
        # even when image and audio runs interleave.
        img_tok = vision.placeholder_token_id if vision is not None else None
        aud_tok = audio.placeholder_token_id if audio is not None else None
        img_iter, aud_iter = iter(img_feats), iter(aud_feats)
        ordered: list[torch.Tensor] = []
        i = 0
        while i < len(row):
            if img_tok is not None and row[i] == img_tok:
                feats = next(img_iter)
            elif aud_tok is not None and row[i] == aud_tok:
                feats = next(aud_iter)
            else:
                i += 1
                continue
            ordered.append(feats)
            i += int(feats.shape[0])
        mm_tokens = {t for t in (img_tok, aud_tok) if t is not None}
        positions = torch.tensor(
            [j for j, tok in enumerate(row) if tok in mm_tokens], dtype=torch.long
        )
        features = torch.cat(ordered, dim=0)
        return input_ids, features, positions

    def generate_multimodal(
        input_ids: torch.Tensor, features: torch.Tensor,
        positions: torch.Tensor, max_tokens: int,
    ) -> torch.Tensor:
        seq = Sequence(
            input_ids=input_ids, max_new_tokens=max_tokens,
            embeds=MultimodalInputs(features=features, positions=positions),
        )
        ids = list(generator.run(seq))
        new = torch.tensor([ids], dtype=torch.long, device=input_ids.device)
        return torch.cat([input_ids, new], dim=1)

    def stream_multimodal_ids(
        input_ids: torch.Tensor, features: torch.Tensor,
        positions: torch.Tensor, max_tokens: int,
    ) -> Iterator[int]:
        yield from generator.run(Sequence(
            input_ids=input_ids, max_new_tokens=max_tokens, stream=True,
            embeds=MultimodalInputs(features=features, positions=positions),
        ))

    return encode_multimodal, generate_multimodal, stream_multimodal_ids


def tokenizer_chat_encoder(tokenizer: transformers.PreTrainedTokenizerBase) -> ChatTokenizer:
    """Apply the model's chat template to messages, return token ids."""

    def encode_messages(
        messages: tuple[ChatMessage, ...], tools: tuple[dict, ...] = (),
        enable_thinking: bool | None = None,
    ) -> torch.Tensor:
        msgs = [chat_message_dict(m) for m in messages]
        extras = dict(chat_template_extras(tokenizer, enable_thinking))
        if tools:
            extras["tools"] = list(tools)
        try:
            rendered = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
                **extras,
            )
        except Exception as exc:
            logger.warning(
                "chat_template_render_failed",
                n_messages=len(messages), error_class=type(exc).__name__, error=str(exc),
            )
            # Chat templates `raise_exception(...)` to reject inputs they can't
            # represent — llama3.2 ">1 tool call per turn", gemma3 "roles must
            # alternate", qwen3.5 "no user query" / "system must be first". These
            # are client-input errors: surface the template's own message as a 400
            # instead of letting a Jinja TemplateError escape as a 500.
            raise RequestError(
                HTTPStatus.BAD_REQUEST, "invalid_request",
                f"these messages cannot be rendered by the model's chat template: {exc}",
            ) from exc
        if not isinstance(rendered, str):
            raise TypeError("apply_chat_template returned non-string")
        batch = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        input_ids = batch["input_ids"]
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("tokenizer returned non-tensor input_ids")
        return input_ids.to(dtype=torch.long)

    return encode_messages


def resolve_prefill_policy() -> int:
    """The production prefill chunk size (ALLOY_CHUNK_PREFILL_SIZE, default 4096).

    LARGE-CHUNK prefill is the production path: chunks of 4096 saturate the
    GEMMs and the final partial chunk runs at an exact grid-shrunk size, so no
    padded rows cost GPU work. Faster than 128-chunking at every depth >= 512 on
    every tracked model (qwen3.6:35b 1.08-1.25x, qwen3.5:0.8b 1.13-1.23x).
    Requires the per-model shrink-chunk tune (`alloy tune <m> --shrink-max 4096
    --only-shrink`); an untuned model still runs (fallback configs), just
    slower. ALLOY_CHUNK_PREFILL_SIZE=128 reproduces the historical small-chunk
    prefill exactly (chunks below generation._MIN_SHRINK_CHUNK compile the
    classic plans, no shrink machinery).

    Benchmarks that want production parity (alloy_cli.benchmark, the release
    scorecard's correctness worker) MUST resolve through this function rather
    than hardcoding from_model kwargs — otherwise they measure a path
    production never runs.
    """
    chunk_env = os.environ.get("ALLOY_CHUNK_PREFILL_SIZE")
    return int(chunk_env) if chunk_env and int(chunk_env) > 0 else 4096


def build_alloy_generator(
    model: transformers.PreTrainedModel,
    cache_dtype: torch.dtype,
    chat_template_auto_injects: bool = False,
    close_think_seq: tuple[int, ...] = (),
    mid_think_heal_seq: tuple[int, ...] = (),
    post_think_heal_seq: tuple[int, ...] = (),
    vision: "ModalityEncoder | None" = None,
    audio: "ModalityEncoder | None" = None,
) -> AlloyGenerator:
    return AlloyGenerator.from_model(
        model,
        cache_dtype=cache_dtype,
        chat_template_auto_injects=chat_template_auto_injects,
        close_think_seq=close_think_seq,
        mid_think_heal_seq=mid_think_heal_seq,
        post_think_heal_seq=post_think_heal_seq,
        chunk_prefill_size=resolve_prefill_policy(),
        vision=vision,
        audio=audio,
    )


TRUNCATED_RESPONSE_NOTICE = "Response truncated.\n"


def chat_template_uses_think_blocks(
    tokenizer: transformers.PreTrainedTokenizerBase,
) -> bool:
    """Detect whether this model's chat template participates in the
    ``<think>...</think>`` protocol — Qwen3-style reasoning models thread
    a ``<think>`` marker through the rendered template (either in the
    generation prompt or in conditional logic around saved assistant
    content). For Llama, Gemma3, plain instruct chat templates etc., the
    literal ``</think>`` is just a token sequence the model could emit in
    normal text and carries no protocol meaning.

    Detection strategy: inspect the raw chat-template source and the
    rendered generation prompt. Reasoning-enabled templates reference
    ``<think>`` literally either in the source (to inject / branch on
    it) or in the rendered prompt (to push the model into a think
    block). Plain-text models do neither.
    """
    template = tokenizer.chat_template if hasattr(tokenizer, "chat_template") else None
    if isinstance(template, str) and "<think>" in template:
        return True
    try:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": "ping"}],
            tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        return False
    return isinstance(rendered, str) and "<think>" in rendered


def resolve_heal_tokens(
    tokenizer: transformers.PreTrainedTokenizerBase,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Resolve heal sequences for mid-think and post-think truncation.

    Returns `(close_think_seq, mid_think_heal_seq, post_think_heal_seq)`:
      - `close_think_seq`: token ids for `</think>` (multi-token in Qwen3's
        GGUF tokenizer). Used to detect "are we inside a think block" by
        scanning the decoded stream for this subsequence.
      - `mid_think_heal_seq`: tokens to append when truncated inside a
        `<think>` block. Closes the think block with a `</think>` + the
        post-think truncated-notice text + turn-end. The resulting cache
        ends in a "completed turn with brief post-think answer" state so
        the next turn's auto-injected `<think>\n` starts a fresh think
        cycle rather than the model echoing a "brief-think" style.
      - `post_think_heal_seq`: tokens to append when truncated past
        `</think>` — just turn-end.

    Models whose chat template doesn't participate in the ``<think>`` /
    ``</think>`` protocol (Llama, Gemma3, plain instruct models) get an
    empty ``close_think_seq`` and the post-think heal for both branches —
    truncation just appends turn-end with no spurious ``</think>`` notice.
    """
    def encode(text: str) -> tuple[int, ...]:
        try:
            ids = tokenizer.encode(text, add_special_tokens=False)
        except Exception:
            return ()
        return tuple(int(i) for i in ids)

    def single(text: str) -> int | None:
        ids = encode(text)
        return ids[0] if len(ids) == 1 else None

    # Prefer the chat-template turn-end token (`<|im_end|>` for Qwen3,
    # `<|eot_id|>` for Llama-3 instruct). Fall back to `eos_token_id`.
    turn_end = single("<|im_end|>") or single("<|eot_id|>")
    if turn_end is None and tokenizer.eos_token_id is not None:
        turn_end = int(tokenizer.eos_token_id)
    if turn_end is None:
        return (), (), ()
    post_think_heal = (turn_end,)
    # Mid-think heal applies only when the chat template uses think blocks
    # and `</think>` encodes to multiple tokens (so a half-emitted block
    # can't simply be terminated by a single id). Otherwise both branches
    # of `_heal_truncated` get the plain turn-end.
    if chat_template_uses_think_blocks(tokenizer):
        close_seq = encode("</think>")
        if len(close_seq) > 1:
            mid_text = "\n</think>\n\n" + TRUNCATED_RESPONSE_NOTICE
            mid_think_heal = encode(mid_text) + (turn_end,)
        else:
            mid_think_heal = post_think_heal
        return close_seq, mid_think_heal, post_think_heal
    return (), post_think_heal, post_think_heal


def chat_template_auto_injects_assistant_markers(
    tokenizer: transformers.PreTrainedTokenizerBase,
) -> bool:
    """Detect whether the tokenizer's chat template inserts text between
    `<role-marker>\\n` and the assistant content we supplied — i.e. markers
    the model must emit explicitly during generation. Examples:

    - Qwen3: injects `<think>\\n\\n</think>\\n\\n` before assistant content
      that doesn't already start with `<think>`. Saved decode tokens
      (which DO start with `<think>`) won't round-trip through
      (decode -> chat-template re-render -> tokenize), so warm-prefill
      can only safely reuse the prompt portion of the cache.
    - Llama: no auto-injection. Saved decode tokens line up with the
      next turn's re-rendered prompt, enabling full prefix reuse.

    Probe with sentinel content rather than checking model class — robust
    across the long tail of model-specific templates without a registry.
    """
    user = "PROBE_USER_TEXT_xxyyzz"
    asst = "PROBE_ASSISTANT_TEXT_xxyyzz"
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": user},
         {"role": "assistant", "content": asst}],
        tokenize=False, add_generation_prompt=False,
    )
    if not isinstance(rendered, str):
        return False
    asst_idx = rendered.find(asst)
    if asst_idx < 0:
        return False
    role_marker = "assistant\n"
    role_idx = rendered.rfind(role_marker, 0, asst_idx)
    if role_idx < 0:
        return False
    return asst_idx > role_idx + len(role_marker)


def attach_spec_drafter(generator: AlloyGenerator, resolved: ResolvedModel, spec: str) -> None:
    """Build + attach the named drafter and log it.
    Missing draft weights raise a clear, actionable error at STARTUP."""
    if spec == "pld":
        drafter = PromptLookupDrafter()
    elif spec in ("mtp", "dflash"):
        if not isinstance(resolved, ResolvedGGUF):
            raise SystemExit(f"alloy serve: --spec {spec} is only available for GGUF models")
        if spec == "mtp":
            drafter = MTPDrafter(resolved.path)
        else:
            drafter = DFlashDrafter(resolve_dflash_checkpoint(resolved.ref), resolved.path)
    else:
        raise ValueError(f"unknown --spec {spec!r}; known: dflash, mtp, pld")
    generator.attach_spec(drafter)
    logger.info("spec_attached", drafter=spec, model=resolved.ref)


def create_native_served_model(
    config: ServerConfig,
    resolved: ResolvedModel,
    load_model: NativeModelLoader = load_resolved_causal_lm,
    build_generator: NativeGeneratorBuilder = build_alloy_generator,
) -> ServedModel:
    model_load_t0 = time.perf_counter()
    logger.info(
        "model_load_requested", model=resolved.ref,
        location=str(resolved.location), source=resolved.format,
    )
    loaded = load_model(resolved)
    tokenizer = loaded.tokenizer
    if tokenizer is None:
        raise RuntimeError(f"tokenizer was not loaded for {resolved.ref}")
    # The whole serving surface renders messages through the chat template
    # (even /api/generate wraps its prompt as a user turn). A GGUF with no
    # template is a base (non-instruct) model that alloy can't serve as chat —
    # fail with an actionable message instead of a transformers stack trace.
    # (Multimodal arches like gemma4 ship no template either, but the loader
    # injects one before we get here, so only genuine base models trip this.)
    if not tokenizer.chat_template:
        raise SystemExit(
            f"alloy serve: {resolved.ref!r} has no chat template — it looks like a "
            f"base (non-instruct) model. alloy serves chat completions, which need "
            f"an instruct/chat GGUF; pick an instruct variant of this model."
        )
    auto_injects = chat_template_auto_injects_assistant_markers(tokenizer)
    close_think_seq, mid_think_heal_seq, post_think_heal_seq = resolve_heal_tokens(tokenizer)
    generator = build_generator(
        loaded.model.eval(),
        torch.float16,
        chat_template_auto_injects=auto_injects,
        close_think_seq=close_think_seq,
        mid_think_heal_seq=mid_think_heal_seq,
        post_think_heal_seq=post_think_heal_seq,
        # A multimodal GGUF surfaces vision/audio front-ends; hand them to the
        # generator so eager_compile_all warms their plans too (one entry point).
        vision=loaded.vision,
        audio=loaded.audio,
    )
    native_context: int | None = None
    max_fill: int | None = None
    if isinstance(generator, AlloyGenerator):
        # Sizes the DeltaNet slot bank; must precede max_fill, which builds the cache.
        if config.spec:
            attach_spec_drafter(generator, resolved, config.spec)
        native_context = generator.max_cache_len
        max_fill = generator.max_fill
        # Pre-compile cold + warm prefill + decode + verify + the vision tower
        # (if any) at the native cache. No real request pays torch.compile cost.
        generator.eager_compile_all()

    logger.info(
        "model_loaded",
        model=resolved.ref,
        source=resolved.format,
        took_s=round(time.perf_counter() - model_load_t0, 2),
        n_params=sum(p.numel() for p in loaded.model.parameters()),
        native_context=native_context,
        max_fill=max_fill,
    )

    encode_messages = tokenizer_chat_encoder(tokenizer)

    # Drop EOS-style stop tokens before decoding so they don't leak into
    # streamed content, but KEEP non-stop specials like `<think>` /
    # `</think>` so reasoning-trace UIs can detect the boundary. With
    # `skip_special_tokens=True` qwen3.5's single-token `</think>`
    # (id 248069) is stripped and the Mac App can't split thinking from
    # answer.
    decode_strip_ids: frozenset[int] = (
        frozenset(generator.eos_token_ids)
        if isinstance(generator, AlloyGenerator)
        else frozenset()
    )
    def decode(token_ids: torch.Tensor) -> str:
        ids = [
            int(token_ids[index].item())
            for index in range(int(token_ids.shape[0]))
        ]
        ids = [t for t in ids if t not in decode_strip_ids]
        return cast(str, tokenizer.decode(ids, skip_special_tokens=False))

    # Lazy grammar factory (built on first constrained request, so unconstrained
    # deployments never pay the xgrammar TokenizerInfo cost). `reasoning` is
    # DETECTED from the chat template (a non-empty </think> heal sequence), never
    # hardcoded per model — so the tool grammar gates correctly for any reasoning model.
    grammar: list = [None]

    def build_matcher(constraint: Constraint):
        if grammar[0] is None:
            grammar[0] = GrammarFactory(
                tokenizer, int(loaded.model.config.vocab_size),
                tool_format=xgrammar_tool_format(resolved.ref),
                reasoning=bool(close_think_seq),
                stop_token_ids=(
                    sorted(generator.eos_token_ids)
                    if isinstance(generator, AlloyGenerator) else None
                ),
            )
        factory = grammar[0]
        if constraint.kind == "tool" and not factory.supports_tool_forcing:
            return None  # no tool-call grammar for this model -> unconstrained
        return factory.matcher(constraint)

    def generate(
        input_ids: torch.Tensor, max_new_tokens: int, constraint: Constraint | None = None,
    ) -> torch.Tensor:
        if (
            isinstance(generator, AlloyGenerator)
            and generator.spec is not None
            and int(input_ids.shape[0]) == 1
        ):
            matcher = build_matcher(constraint) if constraint is not None else None
            if constraint is None or matcher is not None:
                toks = list(generator.spec.run(
                    input_ids, max_new_tokens=max_new_tokens, matcher=matcher,
                ))
                m = generator.spec.last_metrics
                if m is not None and m.rounds:
                    logger.info(
                        "spec_request",
                        drafter=generator.spec.drafter.name,
                        rounds=m.rounds, tau=round(m.tau, 2),
                        acceptance=round(m.acceptance, 2),
                    )
                new = torch.tensor([toks], dtype=torch.long, device=input_ids.device)
                return torch.cat([input_ids, new], dim=1)
        matcher = None
        if constraint is not None and isinstance(generator, AlloyGenerator):
            matcher = build_matcher(constraint)
            if matcher is None:
                constraint = None  # no grammar for this model -> unconstrained
        if isinstance(generator, AlloyGenerator) and int(input_ids.shape[0]) == 1:
            seq = Sequence(
                input_ids=input_ids, max_new_tokens=max_new_tokens,
                sampling=pending_sampling[0], constraint=matcher,
            )
            for _ in generator.run(seq):
                pass
            new = torch.tensor(
                [seq.generated + seq.healed], dtype=torch.long, device=input_ids.device,
            )
            return torch.cat([input_ids, new], dim=1)
        return generator.generate(input_ids, max_new_tokens=max_new_tokens)

    def stream_token_ids(
        input_ids: torch.Tensor, max_new_tokens: int, constraint: Constraint | None = None,
    ) -> Iterator[int]:
        if (
            isinstance(generator, AlloyGenerator)
            and generator.spec is not None
            and int(input_ids.shape[0]) == 1
        ):
            matcher = build_matcher(constraint) if constraint is not None else None
            if constraint is None or matcher is not None:
                yield from generator.spec.run(
                    input_ids, max_new_tokens=max_new_tokens, matcher=matcher,
                )
                m = generator.spec.last_metrics
                if m is not None and m.rounds:
                    logger.info(
                        "spec_request",
                        drafter=generator.spec.drafter.name,
                        rounds=m.rounds,
                        tau=round(m.tau, 2),
                        acceptance=round(m.acceptance, 2),
                        draft_ms=round(m.draft_us / 1000.0, 1),
                        verify_ms=round(m.verify_us / 1000.0, 1),
                    )
                return
        matcher = build_matcher(constraint) if constraint is not None else None
        yield from generator.run(Sequence(
            input_ids=input_ids, max_new_tokens=max_new_tokens,
            sampling=pending_sampling[0], stream=True, constraint=matcher,
        ))

    count_tokens = tokenizer_text_counter(tokenizer)

    # Sampling for the NEXT request, threaded into its Sequence. The spec and
    # constrained paths still read the pinned buffers directly, so apply the
    # params there too (in-place pinned-buffer write, never a recompile).
    pending_sampling: list[SamplingParams | None] = [None]

    def apply_sampling(params: SamplingParams) -> None:
        pending_sampling[0] = params
        if isinstance(generator, AlloyGenerator):
            generator.plans.params[0] = float(params.temperature)
            generator.plans.params[1] = float(params.top_p)
            generator.plans.params[2] = float(params.top_k)
            generator.plans.params[3] = float(params.min_p)
            generator.plans.seed[0] = int(params.seed)

    # Vision: a multimodal GGUF (gemma4) carries a dense vision front-end the loader
    # surfaces as `loaded.vision`. When present, wire the multimodal (encode,
    # generate, stream) hooks so image requests run end-to-end through alloy's
    # quantized decode — no per-model branching here.
    mm_encode: MultimodalEncoder | None = None
    mm_generate: MultimodalGenerator | None = None
    mm_stream: MultimodalStreamer | None = None
    if isinstance(generator, AlloyGenerator) and (
        loaded.vision is not None or loaded.audio is not None
    ):
        # Modality plans were already compiled by generator.eager_compile_all above
        # (the generator owns the adapters) — here we only wire the request hooks.
        mm_encode, mm_generate, mm_stream = build_multimodal_hooks(
            loaded.vision, loaded.audio, tokenizer, generator,
        )
        logger.info(
            "multimodal_adapter_loaded", model=resolved.ref,
            vision=loaded.vision is not None, audio=loaded.audio is not None,
        )

    return create_generation_served_model(
        config.model, encode_messages, decode, generate, stream_token_ids,
        count_tokens, tokenizer=tokenizer,
        reset_prefix_state=(
            generator.reset_prefix_state if isinstance(generator, AlloyGenerator) else None
        ),
        preserve_context=(
            generator.preserving_prefix if isinstance(generator, AlloyGenerator) else None
        ),
        eos_token_ids=frozenset(generator.eos_token_ids)
            if isinstance(generator, AlloyGenerator) else frozenset(),
        apply_sampling=apply_sampling if isinstance(generator, AlloyGenerator) else None,
        last_timings=(
            (lambda: generator.last_gen_timings)
            if isinstance(generator, AlloyGenerator) else None
        ),
        reasoning=resolve_reasoning_protocol(tokenizer, close_think_seq),
        encode_multimodal=mm_encode,
        generate_multimodal=mm_generate,
        stream_multimodal_ids=mm_stream,
    )


def parse_server_config(argv: tuple[str, ...] | None = None) -> ServerConfig:
    parser = server_arg_parser()
    namespace = parser.parse_args(argv)
    return ServerConfig(
        model=cast(str | None, namespace.model),
        hf_id=cast(str | None, namespace.hf_id),
        host=cast(str, namespace.host),
        port=cast(int, namespace.port),
        allow_downloads=cast(bool, namespace.allow_downloads),
        spec=cast("str | None", namespace.spec),
        force=cast(bool, namespace.force),
    )


def run_server(config: ServerConfig) -> None:
    """Run the alloy server in the foreground (`alloy serve -m <model>`).

    One process serves exactly one model, loaded + pre-compiled here and
    held for the life of the process. The ref is resolved to a concrete GGUF
    file (local path / HF repo / Ollama name) and the runtime kind (chat vs
    embedding) is read from the GGUF's own `general.architecture` — no model
    name is special-cased.
    """
    if config.model is None:
        raise SystemExit("alloy serve requires a model: `alloy serve -m <model>`")
    check_port_collision(config.host, config.port)
    try:
        resolved = resolve_model(config.model)
        arch = resolved.architecture()
        check_arch_supported(arch, force=config.force)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"alloy serve: {exc}")
    kind = model_kind(arch)
    server_kwargs: dict = {"spec": config.spec}
    if kind == "chat":
        server_kwargs["chat_model"] = create_native_served_model(config, resolved)
    elif kind == "embed":
        from alloy_server.models.nomic_bert import load_ollama_gguf_embedder  # scoped: pulls transformers.NomicBertModel + huggingface_hub side effects; load only when an embed model is actually served

        server_kwargs["embedding_model"] = load_ollama_gguf_embedder(resolved)
    else:  # transcription (whisper)
        from alloy_server.models.whisper import load_whisper_gguf  # scoped: pulls transformers.Whisper + the alloy backend; load only when an STT model is served

        server_kwargs["served"] = load_whisper_gguf(str(resolved.location), name=config.model)
        server_kwargs["modality"] = TRANSCRIPTION
    server = create_server(config.host, config.port, **server_kwargs)
    logger.info(
        "server_started",
        host=config.host,
        port=server.server_port,
        model=config.model,
        kind=kind,
        spec=config.spec,
    )
    try:
        server.serve_forever()
    finally:
        logger.info("server_stopping", port=server.server_port)
        server.server_close()


def discovered_chat_names() -> list[str]:
    """GGUF model refs discovered on disk (Ollama store + HF cache), in the form
    `alloy serve -m` accepts. A purely informational catalog for client model
    pickers — the arch gate at serve time is what decides loadability, so no
    pre-filtering here."""
    out: list[str] = []
    seen: set[str] = set()
    for model in discover_all():
        if model.name in seen:
            continue
        out.append(model.name)
        seen.add(model.name)
    return out


def check_port_collision(host: str, port: int) -> None:
    """Probe `host:port` before binding so we can surface a useful
    error instead of a raw `OSError: Address already in use`.

    Spec §7.4: if the squatter answers `/api/version` with Ollama's
    version string, point the user at `alloy doctor`. Don't auto-
    fall-back to a different port — that would silently break the
    drop-in promise.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(0.25)
        try:
            sock.connect((host, port))
        except (ConnectionRefusedError, socket.timeout, OSError):
            return  # port is free
    finally:
        sock.close()

    version = ""
    try:
        with urllib.request.urlopen(
            f"http://{host}:{port}/api/version", timeout=0.5,
        ) as resp:
            payload = json.loads(resp.read())
            version = str(payload.get("version", ""))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        pass

    # Ours has `-alloy` suffix; Ollama returns `"0.1.32"`-style versions.
    # Check alloy-self FIRST so a second alloy server on the same port gets a
    # specific message instead of being mis-classified as "not Ollama".
    if "-alloy" in version:
        raise PortCollisionError(
            f"port {port} is already in use by another alloy server. "
            "Stop it or pick a different `--port`.",
        )
    if version:
        raise PortCollisionError(
            f"port {port} is already taken by Ollama. options:\n"
            f"  (a) `alloy serve --port {port + 1}` and set client "
            f"`OLLAMA_HOST=http://{host}:{port + 1}`\n"
            "  (b) `launchctl unload ~/Library/LaunchAgents/com.ollama.ollama.plist`\n"
            "  (c) uninstall Ollama\n"
            "see `alloy doctor` for details.",
        )
    raise PortCollisionError(
        f"port {port} is already taken by another process (not Ollama). "
        "stop it or pick a different `--port`.",
    )


def main(argv: tuple[str, ...] | None = None) -> int:
    config = parse_server_config(argv)
    try:
        run_server(config)
    except KeyboardInterrupt:
        return 130
    except PortCollisionError as exc:
        # Print the actionable message on stderr instead of a stack
        # trace and exit non-zero.
        print(f"alloy serve: {exc}", file=sys.stderr)
        return 78  # EX_CONFIG — convention for "config error, fix and restart"
    return 0


def tokenizer_text_counter(tokenizer: transformers.PreTrainedTokenizerBase) -> TextTokenCounter:
    """Return a callable that counts tokens via the model's tokenizer."""

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    return count


def server_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alloy serve",
        description="Start the Alloy OpenAI-compatible server",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="the one model this process serves (chat or embedding): a local "
             "path (./model.gguf), a HuggingFace GGUF repo (Org/Repo:Q4_K_M), "
             "or an Ollama name (qwen3.5:4b); required",
    )
    parser.add_argument("--hf-id", default=None, help="local Hugging Face model id or path")
    parser.add_argument("--host", default="127.0.0.1", help="bind host")
    parser.add_argument("--port", default=11434, type=int, help="bind port (default 11434, matches Ollama)")
    parser.add_argument(
        "--allow-downloads",
        action="store_true",
        help="allow HF tokenizer to download missing model files",
    )
    parser.add_argument(
        "--spec",
        default=os.environ.get("ALLOY_SPEC") or None,
        help="speculative-decoding drafter (dflash | mtp | pld); env ALLOY_SPEC."
             " Opt-in; missing draft weights fail at startup with the pull hint.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="attempt to load a GGUF whose architecture isn't in the supported "
             "set (best-effort; may fail later in compile).",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
