"""Grammar-constrained decoding support.

xgrammar lives entirely here (Python): it compiles a schema/grammar and runs the
per-step matcher. The GPU side never sees it — `AlloyGenerator.run_constrained`
only consumes the token bitmask the matcher fills. A `Constraint` is the
dialect-agnostic spec the request layer produces; `GrammarFactory` turns it into a
fresh xgrammar matcher (compiled grammars cached per model).

Tool forcing uses xgrammar's model-aware structural tags. Crucially these are
*reasoning-gated*: with `reasoning=True` the grammar stays free during the model's
`<think>` span and only forces the tool call after the reasoning-end marker, so
thinking and forced tool calls coexist. The `reasoning` flag is supplied by the
caller, which DETECTS whether the model emits a reasoning span from its chat
template (a non-empty `</think>` heal sequence) — never assumed per model family."""

from __future__ import annotations

import json
from dataclasses import dataclass

import transformers
import xgrammar
from xgrammar.structural_tag import TagsWithSeparatorFormat, TriggeredTagsFormat


# alloy model family -> xgrammar structural-tag format. Matched in order, first
# substring wins, so more specific keys come first. None = forcing unsupported
# (e.g. gemma, which has no xgrammar tool format).
_XG_FORMATS: tuple[tuple[str, str], ...] = (
    ("qwen3.5", "qwen_3_5"),
    ("qwen3.6", "qwen_3_5"),
    ("qwen35", "qwen_3_5"),
    ("qwen36", "qwen_3_5"),
    ("qwen", "qwen_3"),
    ("llama", "llama"),
    ("deepseek", "deepseek_r1"),
)


def xgrammar_tool_format(model_ref: str) -> str | None:
    """xgrammar structural-tag format for a model ref, or None if tool forcing
    isn't supported for that model (caller then degrades to unconstrained)."""
    name = model_ref.lower()
    for key, fmt in _XG_FORMATS:
        if key in name:
            return fmt
    return None


@dataclass(frozen=True, slots=True)
class Constraint:
    """What to constrain generation to. `kind`:
    - "json": any valid JSON value (response_format json_object / Ollama format=json)
    - "json_schema": JSON matching `schema_json` (a json.dumps'd JSON schema)
    - "tool": a tool call — `tools_json` is the tool list, `tool_choice_json` is
      "required" / "auto" / a named-tool choice dict.

    `single_call` enforces a single tool call structurally (parallel_tool_calls=false
    / disable_parallel_tool_use): exactly one under "required"/named, at most one
    under "auto". Ignored for json/json_schema."""

    kind: str
    schema_json: str | None = None
    tools_json: str | None = None
    tool_choice_json: str | None = None
    single_call: bool = False


class GrammarFactory:
    """Per-model grammar compiler + cache. `matcher(constraint)` returns a fresh,
    stateful GrammarMatcher (compiled grammars are cached; matchers are not).

    `tool_format` is the xgrammar structural-tag format for tool forcing (None if
    unsupported); `reasoning` gates the tool grammar on the model's reasoning span."""

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizerBase,
        vocab_size: int,
        *,
        tool_format: str | None = None,
        reasoning: bool = False,
        stop_token_ids: list[int] | None = None,
    ) -> None:
        # Pass the model's REAL generation stop tokens explicitly. Left to
        # auto-detection, xgrammar reads `tokenizer.eos_token_id` — and a
        # GGUF-loaded tokenizer can carry a bogus one (qwen3.5: '</s>' id
        # 248321, OUT OF RANGE for the 248320 vocab), leaving the matcher
        # with NO stop tokens. A completed grammar then allows zero tokens:
        # every logit masks to -1e30, greedy argmax lands on token id 0
        # ('!'), and the loop emits garbage until max_tokens.
        if stop_token_ids is not None:
            stop_token_ids = [t for t in stop_token_ids if 0 <= t < vocab_size]
        info = xgrammar.TokenizerInfo.from_huggingface(
            tokenizer, vocab_size=vocab_size,
            stop_token_ids=stop_token_ids or None,
        )
        self._compiler = xgrammar.GrammarCompiler(info)
        self._tool_format = tool_format
        self._reasoning = reasoning
        self._cache: dict[tuple, object] = {}

    @property
    def supports_tool_forcing(self) -> bool:
        return self._tool_format is not None

    def _compiled(self, c: Constraint):
        key = (c.kind, c.schema_json, c.tools_json, c.tool_choice_json, c.single_call)
        grammar = self._cache.get(key)
        if grammar is None:
            grammar = self._compile(c)
            self._cache[key] = grammar
        return grammar

    def _compile(self, c: Constraint):
        if c.kind == "json":
            return self._compiler.compile_builtin_json_grammar()
        if c.kind == "json_schema":
            schema = json.loads(c.schema_json) if c.schema_json else {"type": "object"}
            return self._compiler.compile_json_schema(schema)
        if c.kind == "tool":
            if self._tool_format is None:
                raise ValueError("tool forcing unsupported for this model")
            tools = json.loads(c.tools_json) if c.tools_json else []
            choice = json.loads(c.tool_choice_json) if c.tool_choice_json else "required"
            tag = xgrammar.get_model_structural_tag(
                self._tool_format, tools=tools, tool_choice=choice,
                reasoning=self._reasoning,
            )
            if c.single_call:
                # Close the tool-call repetition structurally rather than emitting
                # several and truncating at parse time (which would make streamed
                # tool_calls deltas dishonest). stop_after_first + the format's
                # at_least_one gives exactly-one under "required"/named and
                # at-most-one under "auto". Named choice is a lone tag (no-op).
                fmt = tag.format
                if isinstance(fmt, (TagsWithSeparatorFormat, TriggeredTagsFormat)):
                    fmt.stop_after_first = True
            return self._compiler.compile_structural_tag(tag)
        raise ValueError(f"unknown constraint kind: {c.kind!r}")

    def matcher(self, constraint: Constraint):
        return xgrammar.GrammarMatcher(self._compiled(constraint))
