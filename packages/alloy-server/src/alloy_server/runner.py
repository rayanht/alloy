"""ModelRunner (a served model's capabilities) + GenerationWorker (the request
orchestration that drives them).

`ModelRunner` bundles the callables a served model exposes — encode / decode /
generate / stream / count, sampling, multimodal hooks. `GenerationWorker` owns
the per-request orchestration: warm-prefix reuse + branch/preserve policy (via
ConversationStore), the cold/warm prefill choice, generation, and reporting
finish/timings back through the per-request `record` callback. One worker per
served model; today it runs inline (the serve loop serializes it).
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import ContextManager

import torch
import transformers

from alloy_server.constrain import Constraint
from alloy_server.schema import (
    ApplySampling,
    ChatMessage,
    ChatTokenizer,
    MultimodalEncoder,
    MultimodalGenerator,
    MultimodalStreamer,
    TextTokenCounter,
    TokenDecoder,
    TokenGenerator,
    TokenStreamer,
)
from alloy_server.reasoning import ReasoningProtocol
from alloy_server.session import ConversationStore
from alloy_server.stops import finish_reason


def validate_token_matrix(token_ids: torch.Tensor, label: str) -> None:
    if token_ids.ndim != 2:
        raise ValueError(f"{label} must have shape (batch, sequence)")
    if int(token_ids.shape[0]) != 1:
        raise ValueError(f"{label} must have batch size 1")
    if token_ids.dtype != torch.long:
        raise ValueError(f"{label} must use torch.long dtype")


@dataclass(frozen=True, slots=True)
class ModelRunner:
    """The capabilities a served model exposes; the worker drives these."""
    name: str
    encode_messages: ChatTokenizer
    decode: TokenDecoder
    generate: TokenGenerator
    stream_token_ids: TokenStreamer
    count_tokens: TextTokenCounter
    tokenizer: transformers.PreTrainedTokenizerBase | None = None
    reset_prefix_state: Callable[[], None] | None = None
    preserve_context: "Callable[[int], ContextManager] | None" = None
    eos_token_ids: frozenset[int] = frozenset()
    apply_sampling: ApplySampling | None = None
    last_timings: Callable[[], dict] | None = None
    reasoning: ReasoningProtocol | None = None
    encode_multimodal: MultimodalEncoder | None = None
    generate_multimodal: MultimodalGenerator | None = None
    stream_multimodal_ids: MultimodalStreamer | None = None


class GenerationWorker:
    """Per-request orchestration over one `ModelRunner`: warm reuse + branch /
    preserve policy, cold/warm prefill, generation, and per-request finish/timings
    reporting. `complete` / `stream` match the `ServedModel` contract."""

    def __init__(self, runner: ModelRunner) -> None:
        self.runner = runner
        # Recent (messages, input_ids, decoded) states for warm-prefix reuse + the
        # branch/preserve/continuation policy.
        self.convo = ConversationStore(runner.tokenizer, runner.eos_token_ids, runner.reasoning)
        self.multimodal = (
            runner.encode_multimodal is not None and runner.generate_multimodal is not None
        )

    def report(
        self, record: "Callable[[str, dict], None] | None", token_ids: list[int], max_tokens: int,
    ) -> None:
        # Report finish (length vs stop) + timings onto the per-request Generation
        # (via its record callback). Stop-sequence termination is detected at the
        # request layer; here we only know EOS-vs-length from the token count.
        if record is not None:
            reason = finish_reason(token_ids, max_tokens, self.runner.eos_token_ids)
            timings = self.runner.last_timings
            record(reason, dict(timings()) if timings else {})

    def complete(
        self, messages: tuple[ChatMessage, ...], max_tokens: int, tools: tuple[dict, ...] = (),
        constraint: Constraint | None = None, enable_thinking: bool | None = None,
        record: "Callable[[str, dict], None] | None" = None,
    ) -> str:
        runner = self.runner
        if self.multimodal and any(m.images or m.audio for m in messages):
            assert runner.encode_multimodal is not None and runner.generate_multimodal is not None
            # A follow-up about the same image LCP-matches through the image prefix
            # and reuses its cached KV (features baked in on the first turn); an
            # unrelated request resets so no stale prefix matches by coincidence.
            if not self.convo.is_continuation(messages) and runner.reset_prefix_state is not None:
                runner.reset_prefix_state()
            input_ids, feats, positions = runner.encode_multimodal(messages, enable_thinking)
            validate_token_matrix(input_ids, "encoded prompt")
            output_ids = runner.generate_multimodal(input_ids, feats, positions, max_tokens)
            prompt_len = int(input_ids.shape[1])
            new_tokens = output_ids[0, prompt_len:]
            new_ids = new_tokens.tolist()
            self.report(record, new_ids, max_tokens)
            self.convo.save(messages, input_ids[0].tolist(), new_ids)
            return runner.decode(new_tokens)
        input_ids: torch.Tensor | None = None
        # Warm-prefix reuse works WITH tools: the tool definitions live in the
        # saved prefix (system message) already, and the spliced continuation is
        # tool-independent — so a multi-turn tool conversation reuses the large
        # system+tools+user block instead of re-prefilling it every turn.
        # Constrained decode runs its own (cold) prefill, so skip warm there.
        if constraint is None:
            input_ids = self.convo.warm_input_ids(messages, enable_thinking)
        preserve = False
        if input_ids is None:
            input_ids = runner.encode_messages(messages, tools, enable_thinking)
            preserve = self.convo.preserve_foreign(input_ids)
            # Cold path: drop any saved warm prefix so the generator's
            # token-level LCP doesn't reuse stale cache K/V by matching
            # chat-template wrapper tokens that are identical across
            # unrelated conversations. A small foreign request instead runs
            # inside the preserve context and leaves the prefix alone.
            if not preserve and runner.reset_prefix_state is not None:
                runner.reset_prefix_state()
        validate_token_matrix(input_ids, "encoded prompt")
        ctx = (
            runner.preserve_context(int(input_ids.shape[1]) + max_tokens + 32)
            if preserve and runner.preserve_context is not None
            else contextlib.nullcontext()
        )
        with ctx:
            output_ids = runner.generate(input_ids, max_tokens, constraint)
        validate_token_matrix(output_ids, "generated output")
        prompt_len = int(input_ids.shape[1])
        if int(output_ids.shape[1]) < prompt_len:
            raise ValueError("generated output is shorter than encoded prompt")
        new_tokens = output_ids[0, prompt_len:]
        new_ids = new_tokens.tolist()
        self.report(record, new_ids, max_tokens)
        if not preserve:
            self.convo.save(messages, input_ids[0].tolist(), new_ids)
        return runner.decode(new_tokens)

    def stream(
        self, messages: tuple[ChatMessage, ...], max_tokens: int, tools: tuple[dict, ...] = (),
        constraint: Constraint | None = None, enable_thinking: bool | None = None,
        record: "Callable[[str, dict], None] | None" = None,
    ) -> Iterator[str]:
        """Per-token streaming. Yields the text delta added by each new token.
        Re-decoding the running id list (not just the new token) is required for
        BPE-style tokenizers where a single id rendered in isolation can produce
        different bytes than the same id with context (multi-byte UTF-8 fragments,
        surrogates). Mirrors `complete`'s warm-splice setup so the generator's
        LCP-based warm prefill matches through the assistant turn boundary."""
        runner = self.runner
        if self.multimodal and any(m.images or m.audio for m in messages):
            assert runner.encode_multimodal is not None and runner.stream_multimodal_ids is not None
            if not self.convo.is_continuation(messages) and runner.reset_prefix_state is not None:
                runner.reset_prefix_state()
            mm_input_ids, feats, positions = runner.encode_multimodal(messages, enable_thinking)
            validate_token_matrix(mm_input_ids, "encoded prompt")
            accumulated: list[int] = []
            decoded_prev = ""
            for token_id in runner.stream_multimodal_ids(mm_input_ids, feats, positions, max_tokens):
                accumulated.append(int(token_id))
                full = runner.decode(torch.tensor(accumulated, dtype=torch.long))
                if full.endswith("�"):
                    continue
                delta = full[len(decoded_prev):]
                decoded_prev = full
                if delta:
                    yield delta
            self.report(record, accumulated, max_tokens)
            self.convo.save(messages, mm_input_ids[0].tolist(), list(accumulated))
            return
        input_ids: torch.Tensor | None = None
        if constraint is None:
            input_ids = self.convo.warm_input_ids(messages, enable_thinking)
        preserve = False
        if input_ids is None:
            input_ids = runner.encode_messages(messages, tools, enable_thinking)
            preserve = self.convo.preserve_foreign(input_ids)
            # Cold path: drop saved warm prefix so the generator's token-level LCP
            # doesn't reuse stale K/V by matching wrapper tokens shared across
            # unrelated conversations. A small foreign request instead runs inside
            # the preserve context and leaves the prefix alone.
            if not preserve and runner.reset_prefix_state is not None:
                runner.reset_prefix_state()
        validate_token_matrix(input_ids, "encoded prompt")

        accumulated_ids: list[int] = []
        decoded_so_far = ""
        ctx = (
            runner.preserve_context(int(input_ids.shape[1]) + max_tokens + 32)
            if preserve and runner.preserve_context is not None
            else contextlib.nullcontext()
        )
        with ctx:
            for token_id in runner.stream_token_ids(input_ids, max_tokens, constraint):
                accumulated_ids.append(int(token_id))
                full = runner.decode(torch.tensor(accumulated_ids, dtype=torch.long))
                # A multi-byte character whose UTF-8 bytes span several byte-fallback
                # tokens (deepseek's `｜`/`▁` glyphs, CJK, emoji) decodes to a trailing
                # U+FFFD until its last byte arrives. Hold the incomplete tail back:
                # emitting it leaks a replacement char AND misaligns the length-based
                # delta, corrupting every later delta. Wait for the bytes to complete.
                if full.endswith("�"):
                    continue
                delta = full[len(decoded_so_far):]
                decoded_so_far = full
                if delta:
                    yield delta

        # Save state on normal completion so the next turn can warm-splice
        # (messages-seen-this-turn, prompt-tokens, decoded-tokens). Runs only when
        # the stream is fully consumed (no stop sequence cut it short).
        self.report(record, accumulated_ids, max_tokens)
        if not preserve:
            self.convo.save(messages, input_ids[0].tolist(), list(accumulated_ids))
