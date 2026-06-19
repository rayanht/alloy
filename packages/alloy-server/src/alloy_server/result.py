"""Per-request response assembly.

`Generation` drives one request's text generation through the model, applies
stop-string filtering, splits reasoning from answer, and buffers tool calls —
producing the dialect-agnostic (reasoning, content, tool_calls, finish_reason)
the dialect renderers turn into wire shapes.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from alloy_server.schema import ChatCompletionRequest, ToolCall
from alloy_server.reasoning import (
    ReasoningProtocol,
    consume_reasoning_open,
    split_reasoning,
)
from alloy_server.stops import filter_stops, partial_suffix_len
from alloy_server.toolcalls import (
    TOOL_CALL_OPENERS,
    extract_tool_calls,
    find_tool_opener,
)


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """One request's assembled output: dialect-agnostic. finish_reason and
    timings are per-request fields (the served model reports them through
    `Generation.record_generation` during the run)."""
    content: str
    reasoning: str
    tool_calls: tuple[ToolCall, ...]
    finish_reason: str
    timings: dict = field(default_factory=dict)


class Generation:
    """Drives one request's text generation: applies stop sequences and records
    why it ended, so writers can emit a correct finish_reason. Consume `stream()`
    or `text()` fully BEFORE reading `finish_reason()` / `stop_sequence`."""

    def __init__(self, request: ChatCompletionRequest) -> None:
        self.request = request
        self.stop_sequence: str | None = None  # the stop string that fired, if any
        self.tool_calls: tuple[ToolCall, ...] = ()  # set by text() when tools active
        self.reasoning_content: str = ""  # set by text() for reasoning models
        # Per-request finish (length vs stop) + phase timings, reported by the
        # served model through `record_generation` when it knows the token count.
        self.length_finish: str | None = None
        self.timings: dict = {}

    def record_generation(self, finish: str, timings: dict) -> None:
        """Callback the served model invokes once it has generated, so the
        length-vs-stop reason and timings ride this per-request object."""
        self.length_finish = finish
        self.timings = timings

    def raw_stream(self) -> Iterator[str]:
        """Stop-filtered text deltas, no tool-call handling."""
        req = self.request
        # Pass constraint/enable_thinking/record only when meaningful, so stub
        # models that don't accept those kwargs keep working.
        deltas = req.model.stream(req.messages, req.max_tokens, req.tools, **self.extra_kw())

        def hit(s: str) -> None:
            self.stop_sequence = s

        yield from filter_stops(deltas, req.stop, on_stop=hit)

    def split_reasoning_stream(
        self, deltas: Iterator[str], proto: ReasoningProtocol | None,
    ) -> Iterator[tuple[str, str]]:
        """Tag each delta ("reasoning", …) or ("content", …) for the given protocol.
        The pre-close span is reasoning; the close marker is held back across deltas
        (it can split mid-token), and the opener (`<think>` / `<|channel>thought\\n`)
        is consumed from the head. `proto is None` -> everything is content."""
        if proto is None:
            for d in deltas:
                if d:
                    yield ("content", d)
            return
        buf = ""
        in_content = False
        content_started = False  # strip whitespace between the close marker and the answer
        header_done = not proto.open  # consume the opener before emitting reasoning
        for d in deltas:
            if in_content:
                if not content_started:
                    d = d.lstrip()
                    if not d:
                        continue
                    content_started = True
                if d:
                    yield ("content", d)
                continue
            buf += d
            if not header_done:
                buf, header_done = consume_reasoning_open(buf, proto)
                if not header_done:
                    continue
            idx = buf.find(proto.close)
            if idx >= 0:
                if buf[:idx]:
                    yield ("reasoning", buf[:idx])
                after = buf[idx + len(proto.close):].lstrip()
                in_content = True
                if after:
                    content_started = True
                    yield ("content", after)
                buf = ""
                continue
            # No close marker yet: emit the safe prefix, hold back a tail that could
            # be the start of the close marker.
            hold = partial_suffix_len(buf, proto.close)
            safe, buf = buf[: len(buf) - hold], buf[len(buf) - hold:]
            if safe:
                yield ("reasoning", safe)
        if buf and not in_content:  # stream ended mid-reasoning (truncated): flush it
            if not header_done:
                buf, _ = consume_reasoning_open(buf, proto)
            if buf:
                yield ("reasoning", buf)

    def stream(self) -> Iterator[tuple[str, str]]:
        """Stream (kind, text) deltas, kind in {"reasoning", "content"}. For
        reasoning models the pre-`</think>` span is tagged "reasoning"; the rest is
        "content". With tools active, a tool-call opener marker ANYWHERE in the
        content switches to buffering (models routinely emit prose before the
        call): preceding text streams normally, everything from the opener on is
        parsed into `self.tool_calls` at the end. A partial opener at a delta
        boundary is held back so it can never leak as text. Bare-JSON calls have
        no marker and are only detected at content start. Runs on the
        post-reasoning text, so a reasoning model's tool call is still detected."""
        req = self.request
        # thinking off: no `<think>` opens, so the splitter (which assumes it
        # starts mid-think) would tag the whole answer as reasoning.
        proto = None if req.enable_thinking is False else req.model.reasoning
        tagged = self.split_reasoning_stream(self.raw_stream(), proto)
        if not req.tools:
            yield from tagged
            return
        buf = ""          # unemitted text (held-back partial opener tail)
        tool_buf = ""     # everything from the opener on
        mode: str | None = None  # None=deciding (bare-JSON window), "text", "tool"
        for kind, d in tagged:
            if kind == "reasoning":
                yield ("reasoning", d)
                continue
            if mode == "tool":
                tool_buf += d
                continue
            buf += d
            if mode is None:
                stripped = buf.lstrip()
                if not stripped:
                    continue
                if stripped[0] == "{":   # bare JSON (llama3.2): start-only
                    mode = "tool"
                    tool_buf, buf = buf, ""
                    continue
                mode = "text"
            idx = find_tool_opener(buf)
            if idx >= 0:
                if buf[:idx]:
                    yield ("content", buf[:idx])
                mode = "tool"
                tool_buf, buf = buf[idx:], ""
                continue
            # No opener yet: emit the safe prefix, hold back a tail that could
            # be the start of one (same pattern as stop strings / `</think>`).
            hold = max(partial_suffix_len(buf, m) for m in TOOL_CALL_OPENERS)
            safe, buf = buf[: len(buf) - hold], buf[len(buf) - hold:]
            if safe:
                yield ("content", safe)
        if mode == "tool":
            content, self.tool_calls = extract_tool_calls(tool_buf, True)
            if not self.tool_calls and content:
                yield ("content", content)  # misclassified (e.g. an answer that began with '{')
        elif buf:
            yield ("content", buf)  # flush the held-back partial-opener tail

    def extra_kw(self) -> dict:
        """Model kwargs threaded into complete/stream: constraint / enable_thinking
        only when set, and the `record` callback so finish/timings ride this
        per-request object. Served models accept `**kwargs`; the real native model
        reads `record`."""
        req = self.request
        kw: dict = {"record": self.record_generation}
        if req.constraint is not None:
            kw["constraint"] = req.constraint
        if req.enable_thinking is not None:
            kw["enable_thinking"] = req.enable_thinking
        return kw

    def text(self) -> str:
        """Full assistant content. Reasoning (the model's chain-of-thought span, per
        its `reasoning` protocol) is split into `self.reasoning_content`; tool calls
        are parsed out of the post-reasoning text into `self.tool_calls`. Stop
        strings force the (early-halting) raw stream."""
        req = self.request
        if req.stop or req.constraint is not None:
            # stop strings AND constrained decode both need the per-step path.
            raw = "".join(self.raw_stream())
        else:
            # grid-shrink fast path; reason comes from the model afterwards.
            raw = req.model.complete(req.messages, req.max_tokens, req.tools, **self.extra_kw())
        # Reasoning comes before the answer / tool call, so split it off first.
        rest = raw
        if req.model.reasoning is not None:
            self.reasoning_content, rest = split_reasoning(raw, req.model.reasoning)
        content, self.tool_calls = extract_tool_calls(rest, bool(req.tools))
        return content

    def complete(self) -> GenerationResult:
        """Run to completion (non-streaming) and bundle the per-request result."""
        content = self.text()
        return GenerationResult(
            content=content, reasoning=self.reasoning_content,
            tool_calls=self.tool_calls, finish_reason=self.finish_reason(),
            timings=self.timings,
        )

    def finish_reason(self) -> str:
        """OpenAI vocabulary: 'tool_calls' if the model called tools, 'length' if
        it hit max_tokens, else 'stop' (EOS or stop sequence). The length-vs-stop
        reason is reported per-request via `record_generation`."""
        if self.tool_calls:
            return "tool_calls"
        if self.stop_sequence is not None:
            return "stop"
        return self.length_finish or "stop"

    def ollama_done_reason(self) -> str:
        """Ollama has no 'tool_calls' done_reason — tool calls ride in
        message.tool_calls and done_reason stays 'stop'."""
        fr = self.finish_reason()
        return "stop" if fr == "tool_calls" else fr

    def anthropic_stop(self) -> tuple[str, str | None]:
        """(stop_reason, stop_sequence) in Anthropic's vocabulary."""
        if self.tool_calls:
            return "tool_use", None
        if self.stop_sequence is not None:
            return "stop_sequence", self.stop_sequence
        return ("max_tokens", None) if self.finish_reason() == "length" else ("end_turn", None)
