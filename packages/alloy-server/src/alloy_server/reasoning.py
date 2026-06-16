"""Reasoning / thinking output handling for the server.

How a model delimits chain-of-thought in its *output* (the `</think>` family vs
gemma4's `<|channel>` block), how to split it from the answer, and how to detect
the protocol from a chat template. The per-request `enable_thinking` toggle and
the dialect surfacing live with the dialects; this module is the protocol + the
text-level split.
"""

from __future__ import annotations

from dataclasses import dataclass

import transformers


@dataclass(frozen=True, slots=True)
class ReasoningProtocol:
    """How a model delimits chain-of-thought in its *output*. `close` ends the
    reasoning span; `open` begins it. For the `</think>` family the opener sits in
    the prompt (qwen3/3.5) or the model emits it (deepseek), so it's stripped when
    present. gemma4 emits a channel block `<|channel>thought\\n…\\n<channel|>`: the
    opener carries a channel-name line (`open_has_name_line`) dropped with it."""

    close: str
    open: str
    open_has_name_line: bool = False


THINK_PROTOCOL = ReasoningProtocol(close="</think>", open="<think>")
CHANNEL_PROTOCOL = ReasoningProtocol(close="<channel|>", open="<|channel>", open_has_name_line=True)


def strip_reasoning_open(reasoning: str, proto: ReasoningProtocol) -> str:
    """Drop the protocol's opener (and, for channel protocols, the channel-name
    line that follows) from the front of a reasoning span."""
    if proto.open and proto.open in reasoning:
        reasoning = reasoning.split(proto.open, 1)[-1]
        if proto.open_has_name_line:
            nl = reasoning.find("\n")
            reasoning = reasoning[nl + 1:] if nl >= 0 else reasoning
    return reasoning


def consume_reasoning_open(buf: str, proto: ReasoningProtocol) -> tuple[str, bool]:
    """Streaming opener strip: return (remaining, done). done=False means the opener
    is present at the head of `buf` but not yet complete (wait for more deltas)."""
    if not proto.open:
        return buf, True
    if not buf:
        return buf, False
    if buf.startswith(proto.open):
        rest = buf[len(proto.open):]
        if proto.open_has_name_line:
            nl = rest.find("\n")
            return (buf, False) if nl < 0 else (rest[nl + 1:], True)
        return rest, True
    if proto.open.startswith(buf):
        return buf, False  # buf is a partial prefix of the opener
    return buf, True  # no opener at the head — reasoning starts directly


def split_reasoning(text: str, proto: ReasoningProtocol) -> tuple[str, str]:
    """Split a reasoning model's output into (reasoning, content) on the protocol's
    close marker. No close marker (thinking off, or cut off) -> ("", text)."""
    idx = text.find(proto.close)
    if idx < 0:
        return "", text
    reasoning = strip_reasoning_open(text[:idx], proto).strip()
    content = text[idx + len(proto.close):].lstrip()
    return reasoning, content


def resolve_reasoning_protocol(
    tokenizer: transformers.PreTrainedTokenizerBase, close_think_seq: tuple[int, ...],
) -> ReasoningProtocol | None:
    """Pick a model's reasoning protocol from its chat template: the `</think>`
    family (qwen3/3.5, deepseek — a non-empty close-think heal sequence) or gemma4's
    channel block (`<|channel>`/`<|think|>` in the template). None = non-reasoning."""
    if close_think_seq:
        return THINK_PROTOCOL
    template = tokenizer.chat_template if hasattr(tokenizer, "chat_template") else None
    if isinstance(template, str) and ("<|channel>" in template or "<|think|>" in template):
        return CHANNEL_PROTOCOL
    return None
