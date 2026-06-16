"""Unit tests for the server's pure generation helpers: stop-string filtering,
finish-reason determination, and tool-call parsing/emission (no HTTP, no model)."""

import base64
import contextlib
import json
from http import HTTPStatus

import pytest
import torch

from alloy_server.constrain import xgrammar_tool_format
from alloy_server import (
    create_generation_served_model,
    tokenizer_chat_encoder,
)
from alloy_server.session import (
    chat_template_extras,
    reconstruct_warm_input_ids,
)
from alloy_server.schema import (
    ChatCompletionRequest,
    ChatMessage,
    RequestError,
    ServedModel,
)
from alloy_server.reasoning import (
    CHANNEL_PROTOCOL,
    THINK_PROTOCOL,
    split_reasoning,
)
from alloy_server.stops import filter_stops, finish_reason
from alloy_server.toolcalls import coerce_tool_json, extract_tool_calls
from alloy_server.result import Generation
from alloy_server.dialects import openai, ollama, anthropic
from alloy_server.dialects.common import (
    decode_anthropic_image_source,
    decode_image_source,
    messages_field,
    parse_content_with_images,
    parse_request_tool_calls,
    tool_constraint,
)


def test_stream_holds_back_incomplete_multibyte_char():
    # A glyph whose UTF-8 bytes span byte-fallback tokens (e.g. deepseek's `｜`,
    # CJK, emoji) decodes to a trailing U+FFFD until its last byte arrives. The
    # stream must not leak `�` and must keep the length-based delta aligned.
    decoded = {(1,): "a", (1, 2): "a�", (1, 2, 3): "a｜b"}

    def decode(t: torch.Tensor) -> str:
        return decoded[tuple(int(x) for x in t.tolist())]

    def stream_token_ids(input_ids, max_tokens, constraint=None):
        yield from (1, 2, 3)

    model = create_generation_served_model(
        name="t",
        encode_messages=lambda messages, tools=(), enable_thinking=None: torch.tensor([[0]], dtype=torch.long),
        decode=decode,
        generate=lambda *a, **k: torch.tensor([[0]], dtype=torch.long),
        stream_token_ids=stream_token_ids,
        count_tokens=len,
    )
    out = list(model.stream((ChatMessage(role="user", content="hi"),), 8))
    assert "".join(out) == "a｜b" and "�" not in "".join(out)
    assert out == ["a", "｜b"]  # step 2 held back; step 3 emits the completed glyph


def _stream_model(chunks: list[str]) -> ServedModel:
    return ServedModel(
        name="t",
        complete=lambda messages, max_tokens, tools=(), **kw: "".join(chunks),
        stream=lambda messages, max_tokens, tools=(), **kw: iter(chunks),
        count_tokens=len,
    )


def _tools_req(model: ServedModel) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=model, messages=(ChatMessage(role="user", content="hi"),),
        max_tokens=64, tools=({"type": "function", "function": {"name": "f"}},),
    )


def test_stream_buffers_tool_call_yielding_no_text():
    gen = Generation(_tools_req(_stream_model(
        ["<tool_call>", '{"name": "f",', ' "arguments": {"x": 1}}', "</tool_call>"])))
    assert list(gen.stream()) == []          # tool call buffered, not streamed as text
    assert gen.tool_calls[0].name == "f" and gen.tool_calls[0].arguments == {"x": 1}
    assert gen.finish_reason() == "tool_calls"


def test_stream_detects_tool_call_after_prose():
    # qwen3.5 routinely emits prose before the call. The prose must stream as
    # content; the call (from the opener on) parses into tool_calls instead of
    # leaking as raw XML (this was Claude Code's "tool calls render as text").
    gen = Generation(_tools_req(_stream_model([
        "I'll check that for you. ",
        "<tool_call><function=f><parameter=x>\n1\n</parameter>",
        "</function></tool_call>",
    ])))
    out = list(gen.stream())
    assert "".join(t for k, t in out if k == "content") == "I'll check that for you. "
    assert "<" not in "".join(t for _, t in out)  # no XML leaked
    assert gen.tool_calls[0].name == "f" and gen.tool_calls[0].arguments == {"x": 1}
    assert gen.finish_reason() == "tool_calls"


def test_stream_holds_back_partial_opener_at_delta_boundary():
    # An opener split across deltas must never leak its head as content.
    gen = Generation(_tools_req(_stream_model([
        "Sure. <tool",
        "_call>", '{"name": "f", "arguments": {"x": 1}}', "</tool_call>",
    ])))
    out = list(gen.stream())
    assert "".join(t for k, t in out if k == "content") == "Sure. "
    assert gen.tool_calls[0].name == "f"


def test_stream_partial_opener_lookalike_flushes_as_text():
    # Text that ENDS with something opener-like but never completes it must
    # still be delivered once the stream ends.
    gen = Generation(_tools_req(_stream_model(["a < b and <tool"])))
    out = list(gen.stream())
    assert "".join(t for k, t in out if k == "content") == "a < b and <tool"
    assert gen.tool_calls == ()


def test_stream_streams_text_when_no_tool_call():
    gen = Generation(_tools_req(_stream_model(["Hel", "lo wor", "ld"])))
    # stream() yields (kind, text); a non-reasoning model tags everything "content".
    out = list(gen.stream())
    assert all(kind == "content" for kind, _ in out)
    assert "".join(t for _, t in out) == "Hello world"
    assert gen.tool_calls == ()


def _reasoning_stream_model(chunks: list[str]) -> ServedModel:
    return ServedModel(
        name="r",
        complete=lambda messages, max_tokens, tools=(), **k: "".join(chunks),
        stream=lambda messages, max_tokens, tools=(), **k: iter(chunks),
        count_tokens=len, reasoning=THINK_PROTOCOL)


def test_stream_splits_reasoning_from_content():
    gen = Generation(_req(_reasoning_stream_model(["think", "ing", "</think>", "the ", "answer"])))
    out = list(gen.stream())
    assert "".join(t for k, t in out if k == "reasoning") == "thinking"
    assert "".join(t for k, t in out if k == "content") == "the answer"


def test_stream_reasoning_marker_split_across_deltas():
    # `</think>` arriving in fragments must still split cleanly (held back until whole).
    gen = Generation(_req(_reasoning_stream_model(["reason", "</thi", "nk>", "answer"])))
    out = list(gen.stream())
    assert "".join(t for k, t in out if k == "reasoning") == "reason"
    assert "".join(t for k, t in out if k == "content") == "answer"
    # the partial marker must never leak into either stream
    assert "</thi" not in "".join(t for _, t in out)


def test_stream_reasoning_then_tool_call():
    # A reasoning model's tool call follows </think>; tool buffering runs on the
    # post-reasoning content, so it's detected (not streamed as text).
    model = ServedModel(
        name="r",
        complete=lambda messages, max_tokens, tools=(), **k: "",
        stream=lambda messages, max_tokens, tools=(), **k: iter(
            ["reasoning", "</think>", "<tool_call>", '{"name":"f","arguments":{"x":1}}', "</tool_call>"]),
        count_tokens=len, reasoning=THINK_PROTOCOL)
    gen = Generation(_tools_req(model))
    out = list(gen.stream())
    assert "".join(t for k, t in out if k == "reasoning") == "reasoning"
    assert "".join(t for k, t in out if k == "content") == ""  # tool call buffered, not streamed
    assert gen.tool_calls[0].name == "f" and gen.tool_calls[0].arguments == {"x": 1}


def test_anthropic_tools_normalization():
    out = anthropic.tools([{"name": "f", "description": "d", "input_schema": {"type": "object"}}])
    assert out[0] == {"type": "function", "function": {
        "name": "f", "description": "d", "parameters": {"type": "object"}}}


def test_anthropic_message_to_chat_tool_use_and_result():
    asst = anthropic.message_to_chat("assistant", [
        {"type": "tool_use", "id": "t1", "name": "f", "input": {"x": 1}}])
    assert asst[0].role == "assistant" and asst[0].tool_calls[0].name == "f"
    assert asst[0].tool_calls[0].id == "t1" and asst[0].tool_calls[0].arguments == {"x": 1}
    usr = anthropic.message_to_chat("user", [
        {"type": "tool_result", "tool_use_id": "t1", "content": "42"}])
    assert usr[0].role == "tool" and usr[0].content == "42" and usr[0].tool_call_id == "t1"


def test_anthropic_payload_emits_tool_use():
    model = ServedModel(
        name="t",
        complete=lambda messages, max_tokens, tools=(), **kw: '{"name": "get_weather", "arguments": {"city": "Paris"}}',
        stream=lambda messages, max_tokens, tools=(), **kw: iter([""]),
        count_tokens=len,
    )
    out = anthropic.messages_payload(model, {
        "model": "t", "max_tokens": 50,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"name": "get_weather", "input_schema": {}}],
    })
    tu = [b for b in out["content"] if b["type"] == "tool_use"]
    assert tu[0]["name"] == "get_weather" and tu[0]["input"] == {"city": "Paris"}
    assert out["stop_reason"] == "tool_use"


def test_anthropic_system_role_messages_hoist_into_leading_system():
    # Claude Code sends a system-role message INSIDE `messages` (its skills
    # list) in addition to the top-level `system` field. Chat templates only
    # represent a leading system turn (qwen3.5 raises "System message must be
    # at the beginning"), so both must merge into one leading system message.
    model = ServedModel(
        name="t",
        complete=lambda messages, max_tokens, tools=(), **kw: "",
        stream=lambda messages, max_tokens, tools=(), **kw: iter([""]),
        count_tokens=len,
    )
    req = anthropic.messages_request(model, {
        "model": "t", "max_tokens": 50,
        "system": [{"type": "text", "text": "top-level"}],
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "skills list"},
            {"role": "user", "content": "again"},
        ],
    })
    roles = [m.role for m in req.messages]
    assert roles == ["system", "user", "user"]
    assert req.messages[0].content == "top-level\n\nskills list"

    # An unknown role still 400s.
    with pytest.raises(RequestError):
        anthropic.messages_request(model, {
            "model": "t", "max_tokens": 50,
            "messages": [{"role": "tool", "content": "x"}],
        })


def test_foreign_small_request_preserves_warm_state():
    # A small request that isn't a continuation (Claude Code side calls:
    # topic detection, bash-prefix checks) must run inside the generator's
    # preserve context instead of resetting the warm prefix — each eviction
    # costs the next main turn a full cold re-prefill.
    resets: list[int] = []
    preserved: list[int] = []

    @contextlib.contextmanager
    def preserve_context(side_total: int):
        preserved.append(side_total)
        yield

    def stream_token_ids(input_ids, max_tokens, constraint=None):
        yield from (1, 2)

    def encode(messages, tools=(), enable_thinking=None):
        n = sum(len(m.content) for m in messages)
        return torch.tensor([[0] * n], dtype=torch.long)

    model = create_generation_served_model(
        name="t", encode_messages=encode, decode=lambda t: "x",
        generate=lambda *a, **k: torch.tensor([[0]], dtype=torch.long),
        stream_token_ids=stream_token_ids, count_tokens=len,
        reset_prefix_state=lambda: resets.append(1),
        preserve_context=preserve_context,
    )
    big = (ChatMessage(role="user", content="A" * 1000),)
    small = (ChatMessage(role="user", content="B" * 10),)

    list(model.stream(big, 8))    # establishes the saved state (1000 + 2 tokens)
    assert not preserved
    n_resets = len(resets)

    list(model.stream(small, 8))  # foreign small: preserve context, no reset
    assert preserved == [10 + 8 + 32]  # prompt + max_tokens + margin
    assert len(resets) == n_resets

    # The small request must NOT have claimed the saved state: a second small
    # request still preserves against the BIG state (10*2 <= 1002). Had the
    # first small claimed it (saved=12), this one would evict (20 > 12).
    list(model.stream(small, 8))
    assert len(preserved) == 2
    assert len(resets) == n_resets

    # A big foreign request evicts and claims the primary cache.
    list(model.stream((ChatMessage(role="user", content="C" * 999),), 8))
    assert len(preserved) == 2  # not preserved
    assert len(resets) == n_resets + 1


class FakeTokenizer:
    """Minimal tokenizer for warm-reconstruction tests: char ords as ids,
    0 as the (only) special/EOS token, template = role-tagged concat."""

    chat_template = "fake"

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        return {"input_ids": torch.tensor([[ord(c) for c in text]], dtype=torch.long)}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(int(i)) for i in ids if int(i) != 0 or not skip_special_tokens)

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        out = "".join(f"[{m['role']}]{m['content']}" for m in msgs)
        return out + "[assistant]"


def test_warm_reconstruct_matches_post_reasoning_content():
    # Reasoning models stream the chain-of-thought separately, so clients echo
    # back ONLY the post-`</think>` content. The reconstruction must match
    # that form — comparing against the full decoded text (reasoning included)
    # bailed every text turn of a thinking model and forced a full cold
    # re-prefill (Claude Code's 30s-per-turn symptom).
    tok = FakeTokenizer()
    saved_messages = (
        ChatMessage(role="system", content="sys"),
        ChatMessage(role="user", content="say hello"),
    )
    messages = saved_messages + (
        ChatMessage(role="assistant", content="hello"),  # post-reasoning only
        ChatMessage(role="user", content="again"),
    )
    saved_input_ids = [ord(c) for c in "[system]sys[user]say hello[assistant]"]
    saved_decoded = [ord(c) for c in "THINKING</think>hello"] + [0]  # 0 = EOS

    warm = reconstruct_warm_input_ids(
        tok, messages, saved_messages, saved_input_ids, saved_decoded,
        frozenset({0}), None, THINK_PROTOCOL,
    )
    assert warm is not None
    # The cache-covered prefix must be exactly what was prefilled + emitted.
    assert warm[0].tolist()[: len(saved_input_ids) + len(saved_decoded)] == (
        saved_input_ids + saved_decoded
    )

    # Without the reasoning protocol the old behavior bailed — pin that the
    # protocol is what makes the match work, so it isn't dropped by accident.
    cold = reconstruct_warm_input_ids(
        tok, messages, saved_messages, saved_input_ids, saved_decoded,
        frozenset({0}), None, None,
    )
    assert cold is None


def test_branching_side_call_does_not_evict_warm_state():
    # Claude Code follows each real turn with a +2-shaped side call (same
    # history + a synthetic user message). Both the side call AND the real
    # next turn extend the SAME assistant boundary — a branch. With a single
    # saved state the side call claimed it and the real turn went cold; with
    # the state deque, reconstruction finds the shared base for both.
    encodes: list[int] = []

    def encode(messages, tools=(), enable_thinking=None):
        encodes.append(len(messages))
        text = "".join(f"[{m.role}]{m.content}" for m in messages)
        return torch.tensor([[ord(c) for c in text]], dtype=torch.long)

    def stream_token_ids(input_ids, max_tokens, constraint=None):
        yield from (ord("h"), ord("i"), 0)  # "hi" + EOS

    model = create_generation_served_model(
        name="t", encode_messages=encode,
        decode=lambda t: "".join(chr(int(x)) for x in t.tolist() if int(x) != 0),
        generate=lambda *a, **k: torch.tensor([[0]], dtype=torch.long),
        stream_token_ids=stream_token_ids, count_tokens=len,
        tokenizer=FakeTokenizer(), eos_token_ids=frozenset({0}),
    )

    base = (ChatMessage(role="user", content="question one"),)
    list(model.stream(base, 8))                 # turn 1: cold (encoded)
    assert len(encodes) == 1

    asst = ChatMessage(role="assistant", content="hi")
    side = base + (asst, ChatMessage(role="user", content="title this chat"))
    list(model.stream(side, 8))                 # side call: warm splice, no encode
    assert len(encodes) == 1

    real = base + (asst, ChatMessage(role="user", content="question two"))
    list(model.stream(real, 8))                 # real turn: warm via the OLDER state
    assert len(encodes) == 1


def test_anthropic_billing_header_block_is_stripped():
    # Claude Code's "x-anthropic-billing-header: ...; cch=<hash>;" system block
    # mutates every request; keeping it would byte-destabilize the system
    # prompt and force a full cold re-prefill each turn (warm prefill needs
    # prior messages byte-identical).
    model = ServedModel(
        name="t",
        complete=lambda messages, max_tokens, tools=(), **kw: "",
        stream=lambda messages, max_tokens, tools=(), **kw: iter([""]),
        count_tokens=len,
    )
    def req(cch):
        return anthropic.messages_request(model, {
            "model": "t", "max_tokens": 50,
            "system": [
                {"type": "text", "text": f"x-anthropic-billing-header: cc_version=2.1; cch={cch};"},
                {"type": "text", "text": "You are an agent."},
            ],
            "messages": [{"role": "user", "content": "hi"}],
        })
    r1, r2 = req("91606"), req("a03b8")
    assert r1.messages[0].content == "You are an agent."
    assert r1.messages[0].content == r2.messages[0].content  # byte-stable across turns


def test_extract_tool_calls_wrapped_block():
    text = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
    content, calls = extract_tool_calls(text, active=True)
    assert content == "" and len(calls) == 1
    assert calls[0].name == "get_weather" and calls[0].arguments == {"city": "Paris"}


def test_extract_tool_calls_bare_json():
    content, calls = extract_tool_calls('{"name": "f", "arguments": {"x": 1}}', active=True)
    assert content == "" and calls[0].name == "f" and calls[0].arguments == {"x": 1}


def test_extract_tool_calls_double_brace_quirk():
    # Some small Qwen GGUFs copy the template's literal `{{...}}` example.
    _, calls = extract_tool_calls('{{"name": "f", "arguments": {"x": 1}}}', active=True)
    assert calls[0].name == "f" and calls[0].arguments == {"x": 1}


def test_extract_tool_calls_inactive_passthrough():
    content, calls = extract_tool_calls('{"name": "f", "arguments": {}}', active=False)
    assert calls == () and content == '{"name": "f", "arguments": {}}'


def test_extract_tool_calls_arguments_as_string():
    _, calls = extract_tool_calls('{"name": "f", "arguments": "{\\"x\\": 2}"}', active=True)
    assert calls[0].arguments == {"x": 2}


# --- per-family tool-call formats (ground truth captured from each model) ----

def test_extract_tool_calls_qwen35_xml():
    # qwen3.5 emits XML <function=..><parameter=..> inside <tool_call> tags, with
    # leading reasoning + </think>. Typed values (int/bool/array) coerce.
    text = ("reasoning\n</think>\n\n<tool_call>\n<function=get_weather>\n"
            "<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"
            "<tool_call>\n<function=f>\n<parameter=n>\n3\n</parameter>\n"
            "<parameter=flag>\ntrue\n</parameter>\n<parameter=tags>\n[\"a\",\"b\"]\n"
            "</parameter>\n</function>\n</tool_call>")
    _, calls = extract_tool_calls(text, active=True)
    assert [c.name for c in calls] == ["get_weather", "f"]
    assert calls[0].arguments == {"city": "Paris"}
    assert calls[1].arguments == {"n": 3, "flag": True, "tags": ["a", "b"]}


def test_extract_tool_calls_deepseek():
    text = ("Okay let me think.</think>"
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
            "```json\n{\"city\": \"Paris\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>")
    content, calls = extract_tool_calls(text, active=True)
    assert len(calls) == 1 and calls[0].name == "get_weather"
    assert calls[0].arguments == {"city": "Paris"}
    assert content == "Okay let me think.</think>"  # tool block stripped, prose kept


def test_extract_tool_calls_gemma4():
    flat = '<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>'
    _, calls = extract_tool_calls(flat, active=True)
    assert calls[0].name == "get_weather" and calls[0].arguments == {"city": "Paris"}
    # typed scalars + nested object + array
    nested = ('<|tool_call>call:f{opts:{unit:<|"|>celsius<|"|>,n:2},days:3,'
              'metric:true,tags:[<|"|>a<|"|>,<|"|>b<|"|>]}<tool_call|>')
    _, calls = extract_tool_calls(nested, active=True)
    assert calls[0].arguments == {
        "opts": {"unit": "celsius", "n": 2}, "days": 3, "metric": True, "tags": ["a", "b"]}


def test_enable_thinking_openai():
    assert openai.enable_thinking({"chat_template_kwargs": {"enable_thinking": False}}) is False
    assert openai.enable_thinking({"chat_template_kwargs": {"enable_thinking": True}}) is True
    assert openai.enable_thinking({"reasoning_effort": "none"}) is False
    assert openai.enable_thinking({"reasoning_effort": "high"}) is True
    assert openai.enable_thinking({}) is None  # absent -> model default


def test_enable_thinking_ollama():
    assert ollama.enable_thinking({"think": True}) is True
    assert ollama.enable_thinking({"think": False}) is False
    assert ollama.enable_thinking({"think": "high"}) is True  # effort string
    assert ollama.enable_thinking({"think": "none"}) is False
    assert ollama.enable_thinking({}) is None


def test_enable_thinking_anthropic():
    assert anthropic.enable_thinking({"type": "enabled"}) is True
    assert anthropic.enable_thinking({"type": "disabled"}) is False
    assert anthropic.enable_thinking(None) is None


def test_chat_template_extras_respects_enable_thinking():
    class Tok:
        chat_template = "...{% if enable_thinking %}<think>{% endif %}..."
    tok = Tok()
    assert chat_template_extras(tok) == {"enable_thinking": True}            # default ON
    assert chat_template_extras(tok, True) == {"enable_thinking": True}
    assert chat_template_extras(tok, False) == {"enable_thinking": False}   # explicit OFF

    class Plain:
        chat_template = "no thinking marker here"
    assert chat_template_extras(Plain(), False) == {}  # non-reasoning template: no-op


def test_split_reasoning_think_protocol():
    # qwen3/3.5 leave <think> in the prompt -> output is `reasoning</think>answer`.
    assert split_reasoning("thinking hard</think>the answer", THINK_PROTOCOL) == ("thinking hard", "the answer")
    # a leading <think> (if the model emits it) is stripped; surrounding ws trimmed.
    assert split_reasoning("<think>\nreason\n</think>\n\nanswer", THINK_PROTOCOL) == ("reason", "answer")
    # no marker (thinking off / cut off) -> all content, no reasoning.
    assert split_reasoning("no markers here", THINK_PROTOCOL) == ("", "no markers here")


def test_split_reasoning_channel_protocol():
    # gemma4 emits <|channel>thought\n...<channel|>answer (captured from the model).
    g = "<|channel>thought\nHere is my reasoning\nstep two<channel|>The ball costs $0.05."
    assert split_reasoning(g, CHANNEL_PROTOCOL) == (
        "Here is my reasoning\nstep two", "The ball costs $0.05.")
    # cut off before the channel closes -> all reasoning, no content.
    assert split_reasoning("<|channel>thought\npartial", CHANNEL_PROTOCOL) == ("", "<|channel>thought\npartial")


def test_stream_splits_reasoning_channel_protocol():
    # gemma4 channel thinking, streamed: header (<|channel>thought\n) + close
    # (<channel|>) stripped, split across deltas.
    chunks = ["<|channel>", "thought\n", "Here is ", "my reasoning", "<channel|>", "The answer"]
    model = ServedModel(
        name="g",
        complete=lambda messages, max_tokens, tools=(), **k: "".join(chunks),
        stream=lambda messages, max_tokens, tools=(), **k: iter(chunks),
        count_tokens=len, reasoning=CHANNEL_PROTOCOL)
    out = list(Generation(_req(model)).stream())
    assert "".join(t for k, t in out if k == "reasoning") == "Here is my reasoning"
    assert "".join(t for k, t in out if k == "content") == "The answer"
    assert "<|channel>" not in "".join(t for _, t in out)
    assert "<channel|>" not in "".join(t for _, t in out)


def _reasoning_model(raw: str, reasoning) -> ServedModel:
    return ServedModel(
        name="r",
        complete=lambda messages, max_tokens, tools=(), **kw: raw,
        stream=lambda messages, max_tokens, tools=(), **kw: iter([raw]),
        count_tokens=len, reasoning=reasoning)


def test_generation_separates_reasoning_for_reasoning_model():
    gen = Generation(_req(_reasoning_model("let me think</think>42", reasoning=THINK_PROTOCOL)))
    assert gen.text() == "42"
    assert gen.reasoning_content == "let me think"


def test_generation_no_reasoning_split_for_plain_model():
    # A non-reasoning model: never split, even if </think> appears literally.
    gen = Generation(_req(_reasoning_model("plain</think>still here", reasoning=None)))
    assert gen.text() == "plain</think>still here"
    assert gen.reasoning_content == ""


def test_generation_reasoning_then_tool_call():
    # Reasoning is split first, then the tool call is parsed from the remainder.
    raw = 'reasoning</think><tool_call>\n{"name": "f", "arguments": {"x": 1}}\n</tool_call>'
    req = ChatCompletionRequest(
        model=_reasoning_model(raw, reasoning=THINK_PROTOCOL),
        messages=(ChatMessage(role="user", content="hi"),),
        max_tokens=64, tools=({"type": "function", "function": {"name": "f"}},))
    gen = Generation(req)
    assert gen.text() == "" and gen.reasoning_content == "reasoning"
    assert gen.tool_calls[0].name == "f" and gen.tool_calls[0].arguments == {"x": 1}


def test_messages_field_maps_developer_to_system():
    # "developer" (OpenAI o1) must not be silently dropped by templates that only
    # branch on system/user/assistant — normalize it to system at parse.
    msgs = messages_field({"messages": [
        {"role": "developer", "content": "be terse"},
        {"role": "user", "content": "hi"}]}, "messages")
    assert msgs[0].role == "system" and msgs[0].content == "be terse"
    assert msgs[1].role == "user"


def test_encode_messages_template_rejection_is_400():
    # A chat template that raise_exception()s (llama >1 tool call, gemma3 role
    # alternation, ...) must surface as a 400 with the template's message, not a 500.
    class RaisingTok:
        chat_template = "{% if x %}y{% endif %}"

        def apply_chat_template(self, *a, **k):
            raise ValueError("This model only supports single tool-calls at once!")

        def __call__(self, *a, **k):  # pragma: no cover - never reached
            return {"input_ids": None}

    encode = tokenizer_chat_encoder(RaisingTok())
    with pytest.raises(RequestError) as ei:
        encode((ChatMessage(role="user", content="hi"),))
    assert ei.value.status == HTTPStatus.BAD_REQUEST
    assert "single tool-calls" in ei.value.message


def test_xgrammar_tool_format_per_family():
    # qwen3.5 must NOT collapse to qwen_3 — it uses the XML (qwen_3_5) grammar.
    assert xgrammar_tool_format("qwen3.5:4b") == "qwen_3_5"
    assert xgrammar_tool_format("qwen3.5:0.8b") == "qwen_3_5"
    assert xgrammar_tool_format("qwen3:0.6b") == "qwen_3"
    assert xgrammar_tool_format("qwen2.5:3b") == "qwen_3"
    assert xgrammar_tool_format("llama3.2:1b") == "llama"
    assert xgrammar_tool_format("deepseek-r1:1.5b") == "deepseek_r1"
    assert xgrammar_tool_format("gemma4:e2b") is None  # no xgrammar gemma format


def test_coerce_tool_json_parameters_fallback():
    assert coerce_tool_json('{"name": "f", "parameters": {"a": 1}}') == {"name": "f", "arguments": {"a": 1}}
    assert coerce_tool_json("not json") is None


def test_parse_request_tool_calls_openai_and_ollama_shapes():
    openai = parse_request_tool_calls([
        {"id": "call_1", "type": "function", "function": {"name": "f", "arguments": '{"x": 1}'}},
    ])
    assert openai[0].id == "call_1" and openai[0].arguments == {"x": 1}
    ollama = parse_request_tool_calls([{"function": {"name": "g", "arguments": {"y": 2}}}])
    assert ollama[0].name == "g" and ollama[0].arguments == {"y": 2}


def _tool_model(output: str) -> ServedModel:
    return ServedModel(
        name="t",
        complete=lambda messages, max_tokens, tools=(), **kw: output,
        stream=lambda messages, max_tokens, tools=(), **kw: iter([output]),
        count_tokens=len,
    )


_TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]


def test_openai_payload_emits_tool_calls():
    model = _tool_model('<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>')
    out = openai.chat_completion_payload(
        model,
        {"model": "t", "messages": [{"role": "user", "content": "hi"}], "tools": _TOOLS},
    )
    choice = out["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    tc = choice["message"]["tool_calls"][0]
    assert tc["type"] == "function" and tc["function"]["name"] == "get_weather"
    assert json.loads(tc["function"]["arguments"]) == {"city": "Paris"}


def test_ollama_payload_emits_tool_calls():
    model = _tool_model('{"name": "get_weather", "arguments": {"city": "Tokyo"}}')
    out = ollama.chat_payload(
        model,
        {"model": "t", "messages": [{"role": "user", "content": "hi"}],
         "stream": False, "tools": _TOOLS},
    )
    assert out["done_reason"] == "stop"  # Ollama keeps stop; tool_calls in message
    tc = out["message"]["tool_calls"][0]
    assert tc["function"]["name"] == "get_weather" and tc["function"]["arguments"] == {"city": "Tokyo"}


def test_no_tools_no_tool_calls():
    model = _tool_model("just a normal answer")
    out = openai.chat_completion_payload(
        model, {"model": "t", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert out["choices"][0]["message"]["content"] == "just a normal answer"
    assert out["choices"][0]["message"].get("tool_calls") is None
    assert out["choices"][0]["finish_reason"] == "stop"


def test_finish_reason_function():
    eos = frozenset({2})
    assert finish_reason([5, 6, 2], 10, eos) == "stop"        # natural EOS under cap
    assert finish_reason([5, 6, 7, 8], 4, eos) == "length"    # hit cap, no EOS (no heal)
    assert finish_reason([5, 6, 7, 8, 2], 4, eos) == "length" # cap, healed EOS past cap
    assert finish_reason([5, 6, 7, 2], 4, eos) == "stop"      # natural EOS exactly at cap
    assert finish_reason([5, 6], 10, eos) == "stop"           # under cap, no EOS


def _collect(chunks, stops):
    return "".join(filter_stops(iter(chunks), tuple(stops)))


def test_filter_stops_basic_and_holdback():
    assert _collect(["Hello", " world"], []) == "Hello world"
    assert _collect(["abcSTOPdef"], ["STOP"]) == "abc"          # within one delta
    assert _collect(["ab", "cST", "OPdef"], ["STOP"]) == "abc"  # split across deltas
    assert _collect(["abcST"], ["STOP"]) == "abcST"             # partial never completes
    assert _collect(["aXbYc"], ["Y", "X"]) == "a"               # earliest wins
    assert _collect(["STOPxyz"], ["STOP"]) == ""                # stop at start
    assert _collect(["a", "\n", "b"], ["\n\n"]) == "a\nb"       # held \n flushed, no leak
    assert _collect(["l", "\n", "\n", "x"], ["\n\n"]) == "l"    # \n\n is the stop


def test_filter_stops_reports_which_stop():
    fired = []
    list(filter_stops(iter(["aSTOPb"]), ("STOP", "X"), on_stop=fired.append))
    assert fired == ["STOP"]


def _model(*, finish, deltas):
    def complete(messages, max_tokens, tools=(), record=None, **kw):
        if record is not None:
            record(finish, {})
        return "".join(deltas)

    def stream(messages, max_tokens, tools=(), record=None, **kw):
        yield from deltas
        if record is not None:
            record(finish, {})

    return ServedModel(name="t", complete=complete, stream=stream, count_tokens=len)


def _req(model, *, stop=()):
    return ChatCompletionRequest(
        model=model,
        messages=(ChatMessage(role="user", content="hi"),),
        max_tokens=8,
        stop=tuple(stop),
    )


def test_finish_reason_length():
    gen = Generation(_req(_model(finish="length", deltas=["abc"])))
    assert gen.text() == "abc"
    assert gen.finish_reason() == "length"
    assert gen.anthropic_stop() == ("max_tokens", None)


def test_finish_reason_eos_stop():
    gen = Generation(_req(_model(finish="stop", deltas=["abc"])))
    assert gen.text() == "abc"
    assert gen.finish_reason() == "stop"
    assert gen.anthropic_stop() == ("end_turn", None)


def test_finish_reason_stop_sequence_overrides():
    # Even though the model would report "length", a stop sequence took priority.
    gen = Generation(_req(_model(finish="length", deltas=["abSTOPcd"]), stop=["STOP"]))
    assert gen.text() == "ab"
    assert gen.finish_reason() == "stop"
    assert gen.stop_sequence == "STOP"
    assert gen.anthropic_stop() == ("stop_sequence", "STOP")


def test_finish_reason_none_model_defaults_stop():
    model = ServedModel(
        name="stub",
        complete=lambda messages, max_tokens, tools=(), **kw: "x",
        stream=lambda messages, max_tokens, tools=(), **kw: iter(["x"]),
        count_tokens=len,
    )
    gen = Generation(_req(model))
    assert gen.text() == "x"
    assert gen.finish_reason() == "stop"


# --- constrained-decoding constraint parsing -------------------------------

_SCHEMA = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}


def test_openai_response_format_json_object():
    c = openai.response_format_constraint({"type": "json_object"})
    assert c is not None and c.kind == "json" and c.schema_json is None


def test_openai_response_format_json_schema():
    c = openai.response_format_constraint(
        {"type": "json_schema", "json_schema": {"name": "p", "schema": _SCHEMA}})
    assert c is not None and c.kind == "json_schema"
    assert json.loads(c.schema_json) == _SCHEMA


def test_openai_response_format_json_schema_without_schema_falls_back_to_json():
    c = openai.response_format_constraint({"type": "json_schema", "json_schema": {}})
    assert c is not None and c.kind == "json"


def test_openai_response_format_text_and_garbage_are_none():
    assert openai.response_format_constraint({"type": "text"}) is None
    assert openai.response_format_constraint("json") is None
    assert openai.response_format_constraint(None) is None


def test_ollama_format_constraint_json_and_schema():
    assert ollama.format_constraint("json").kind == "json"
    c = ollama.format_constraint(_SCHEMA)
    assert c is not None and c.kind == "json_schema"
    assert json.loads(c.schema_json) == _SCHEMA
    assert ollama.format_constraint("") is None
    assert ollama.format_constraint({}) is None


_NAMED_OPENAI = {"type": "function", "function": {"name": "get_weather"}}
_NAMED_ANTHROPIC = {
    "type": "allowed_tools",
    "allowed_tools": {
        "mode": "required",
        "tools": [{"type": "function", "function": {"name": "get_weather"}}],
    },
}


def test_tool_constraint_required_and_named_and_auto():
    tools = ({"type": "function", "function": {"name": "get_weather"}},)
    # auto / no tools / none -> no constraint (model decides; fast path)
    assert tool_constraint(tools, "auto", None) is None
    assert tool_constraint((), "required", None) is None
    assert tool_constraint(tools, "none", None) is None
    # required -> "required" choice (one-or-more, parallel allowed by default)
    c = tool_constraint(tools, "required", None)
    assert c is not None and c.kind == "tool" and c.single_call is False
    assert json.loads(c.tool_choice_json) == "required"
    assert json.loads(c.tools_json) == list(tools)


def test_tool_constraint_named_dialect_difference():
    tools = ({"type": "function", "function": {"name": "get_weather"}},)
    # OpenAI named (default) -> {type:function}: exactly one, always.
    c = tool_constraint(tools, "required", "get_weather")
    assert c is not None and json.loads(c.tool_choice_json) == _NAMED_OPENAI
    # Anthropic named -> allowed_tools{required}: >=1 of the tool by default.
    c = tool_constraint(tools, "required", "get_weather", named_exactly_one=False)
    assert c is not None and json.loads(c.tool_choice_json) == _NAMED_ANTHROPIC


def test_tool_constraint_single_call():
    tools = ({"type": "function", "function": {"name": "get_weather"}},)
    # required + single_call -> exactly one (choice stays "required", flag set)
    c = tool_constraint(tools, "required", None, single_call=True)
    assert c is not None and c.single_call is True
    assert json.loads(c.tool_choice_json) == "required"
    # auto + single_call -> constrain to at-most-one (choice "auto", flag set)
    c = tool_constraint(tools, "auto", None, single_call=True)
    assert c is not None and c.kind == "tool" and c.single_call is True
    assert json.loads(c.tool_choice_json) == "auto"
    # auto WITHOUT single_call stays unconstrained (fast path)
    assert tool_constraint(tools, "auto", None, single_call=False) is None
    # Anthropic named + single_call -> exactly one of that tool (allowed_tools + flag)
    c = tool_constraint(tools, "required", "get_weather", single_call=True,
                         named_exactly_one=False)
    assert c is not None and c.single_call is True
    assert json.loads(c.tool_choice_json) == _NAMED_ANTHROPIC


# --- Multimodal image input parsing (OpenAI / Ollama / Anthropic) -------------

_FAKE_JPEG = b"\xff\xd8\xff\xe0fake-image-bytes"
_FAKE_B64 = base64.b64encode(_FAKE_JPEG).decode()


def test_decode_image_source_data_url_and_bare_base64():
    # data: URL strips the prefix and decodes the payload
    assert decode_image_source(f"data:image/jpeg;base64,{_FAKE_B64}") == _FAKE_JPEG
    # bare base64 (Ollama `images`) decodes directly
    assert decode_image_source(_FAKE_B64) == _FAKE_JPEG


def test_decode_image_source_invalid_base64_is_400():
    with pytest.raises(RequestError) as exc:
        decode_image_source("not!valid!base64!")
    assert exc.value.status == HTTPStatus.BAD_REQUEST


def test_parse_content_with_images_string_and_parts():
    # plain string content -> no images, no audio
    assert parse_content_with_images("hello") == ("hello", (), ())
    # parts list: text joined, image_url decoded
    text, images, audio = parse_content_with_images([
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_FAKE_B64}"}},
    ])
    assert text == "what is this?"
    assert images == (_FAKE_JPEG,)
    assert audio == ()
    # input_audio part decoded into the audio channel
    text, images, audio = parse_content_with_images([
        {"type": "text", "text": "transcribe"},
        {"type": "input_audio", "input_audio": {"data": _FAKE_B64, "format": "wav"}},
    ])
    assert text == "transcribe" and images == () and audio == (_FAKE_JPEG,)


def test_messages_field_openai_image_part_and_ollama_images():
    # OpenAI: image carried as a content part
    oa = messages_field({"messages": [
        {"role": "user", "content": [
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": _FAKE_B64}},
        ]},
    ]}, "messages")
    assert oa[0].content == "describe" and oa[0].images == (_FAKE_JPEG,)
    # Ollama: images carried out-of-band on the message
    ol = messages_field({"messages": [
        {"role": "user", "content": "describe", "images": [_FAKE_B64]},
    ]}, "messages")
    assert ol[0].content == "describe" and ol[0].images == (_FAKE_JPEG,)
    # text-only message still parses with no images
    txt = messages_field({"messages": [{"role": "user", "content": "hi"}]}, "messages")
    assert txt[0].images == ()


def test_anthropic_image_block_extraction():
    msgs = anthropic.message_to_chat("user", [
        {"type": "text", "text": "what is in this image?"},
        {"type": "image", "source": {
            "type": "base64", "media_type": "image/jpeg", "data": _FAKE_B64,
        }},
    ])
    assert len(msgs) == 1
    assert msgs[0].content == "what is in this image?"
    assert msgs[0].images == (_FAKE_JPEG,)


def test_decode_anthropic_image_source_base64_and_url_and_invalid():
    assert decode_anthropic_image_source(
        {"type": "base64", "data": _FAKE_B64}
    ) == _FAKE_JPEG
    with pytest.raises(RequestError):
        decode_anthropic_image_source({"type": "base64"})  # missing data
