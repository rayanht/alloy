"""Per-dialect unit tests: parse (payload -> ChatCompletionRequest) and render
(Result -> wire shape) in isolation, plus the path -> Dialect routing. No HTTP."""

from __future__ import annotations

import json
from collections.abc import Iterator
from http import HTTPStatus

import pytest

from alloy_server.dialects import (
    ANTHROPIC,
    OLLAMA_CHAT,
    OLLAMA_GENERATE,
    OPENAI,
    chat_dialect_for_path,
    error_dialect_for_path,
)
from alloy_server.dialects.base import NotSupported
from alloy_server.schema import ChatCompletionRequest, ServedModel
from alloy_server.reasoning import THINK_PROTOCOL


def _stub(*, reasoning: bool = False, text: str = "echo") -> ServedModel:
    prefix = "<think>thinking</think>" if reasoning else ""

    def complete(messages, max_tokens, tools=(), **kwargs) -> str:
        return prefix + text

    def stream(messages, max_tokens, tools=(), **kwargs) -> Iterator[str]:
        full = prefix + text
        for i in range(0, len(full), 3):
            yield full[i:i + 3]

    return ServedModel(
        name="m", complete=complete, stream=stream, count_tokens=len,
        reasoning=THINK_PROTOCOL if reasoning else None,
    )


# ---- routing ----

def test_chat_dialect_routing():
    assert chat_dialect_for_path("/v1/chat/completions") is OPENAI
    assert chat_dialect_for_path("/api/chat") is OLLAMA_CHAT
    assert chat_dialect_for_path("/api/generate") is OLLAMA_GENERATE
    assert chat_dialect_for_path("/v1/messages") is ANTHROPIC
    assert chat_dialect_for_path("/api/tags") is None


def test_error_dialect_routing():
    assert error_dialect_for_path("/v1/messages") is ANTHROPIC
    assert error_dialect_for_path("/api/embed").name == "ollama"
    assert error_dialect_for_path("/healthz").name == "ollama"
    assert error_dialect_for_path("/v1/chat/completions").name == "openai"


# ---- wants_stream defaults differ per dialect ----

def test_wants_stream_defaults():
    assert OPENAI.wants_stream({}) is False
    assert OLLAMA_CHAT.wants_stream({}) is True   # ollama defaults stream=true
    assert ANTHROPIC.wants_stream({}) is False


# ---- error envelopes ----

def test_render_error_envelopes():
    oa = OPENAI.render_error(HTTPStatus.NOT_FOUND, "model_not_found", "nope")
    assert oa == {"error": {"message": "nope", "type": "model_not_found", "code": "model_not_found"}}
    ol = OLLAMA_CHAT.render_error(HTTPStatus.NOT_FOUND, "model_not_found", "nope")
    assert ol == {"error": "nope"}
    an = ANTHROPIC.render_error(HTTPStatus.NOT_FOUND, "model_not_found", "nope")
    assert an["type"] == "error"
    assert an["error"]["type"] == "not_found_error"
    assert an["request_id"] is None
    # unknown code falls back to api_error
    assert ANTHROPIC.render_error(HTTPStatus.BAD_REQUEST, "weird", "x")["error"]["type"] == "api_error"


# ---- OpenAI ----

def test_openai_parse_chat():
    req = OPENAI.parse_chat(_stub(), {
        "messages": [{"role": "user", "content": "hi"}], "stop": ["X"], "max_tokens": 5,
    })
    assert isinstance(req, ChatCompletionRequest)
    assert req.messages[-1].content == "hi"
    assert req.stop == ("X",)
    assert req.max_tokens == 5


def test_openai_response_format_constraint():
    req = OPENAI.parse_chat(_stub(), {
        "messages": [{"role": "user", "content": "hi"}],
        "response_format": {"type": "json_object"},
    })
    assert req.constraint is not None and req.constraint.kind == "json"


def test_openai_render_chat():
    out = OPENAI.render_chat(_stub(text="hello"), {"messages": [{"role": "user", "content": "hi"}]})
    assert out["object"] == "chat.completion"
    assert out["choices"][0]["message"]["content"] == "hello"
    assert out["choices"][0]["finish_reason"] == "stop"


def test_openai_render_chat_stream_frames():
    req = OPENAI.parse_chat(_stub(text="hi"), {"messages": [{"role": "user", "content": "x"}]})
    frames = list(OPENAI.render_chat_stream(req))
    assert all(isinstance(f, bytes) for f in frames)
    assert frames[0].startswith(b"data: ")
    assert frames[-1] == b"data: [DONE]\n\n"


def test_openai_render_catalog_and_embeddings():
    cat = OPENAI.render_catalog(("a", "b"))
    assert cat["object"] == "list" and [m["id"] for m in cat["data"]] == ["a", "b"]


# ---- Ollama ----

def test_ollama_parse_options_and_format():
    req = OLLAMA_CHAT.parse_chat(_stub(), {
        "messages": [{"role": "user", "content": "x"}],
        "options": {"num_predict": 7, "stop": ["Z"]},
        "format": "json",
    })
    assert req.max_tokens == 7
    assert req.stop == ("Z",)
    assert req.constraint is not None and req.constraint.kind == "json"


def test_ollama_render_chat_done_frame():
    req = OLLAMA_CHAT.parse_chat(_stub(text="hi"), {"messages": [{"role": "user", "content": "x"}]})
    frames = list(OLLAMA_CHAT.render_chat_stream(req))
    assert json.loads(frames[-1])["done"] is True


def test_ollama_generate_parse_and_render():
    req = OLLAMA_GENERATE.parse_chat(_stub(), {"prompt": "hello"})
    assert req.messages[-1].content == "hello"
    out = OLLAMA_GENERATE.render_chat(_stub(text="hi"), {"prompt": "hello"})
    assert out["response"] == "hi" and out["done"] is True


def test_ollama_render_catalog_shape():
    cat = OLLAMA_CHAT.render_catalog(("m",))
    assert cat["models"][0]["name"] == "m" and "details" in cat["models"][0]


# ---- Anthropic ----

def test_anthropic_system_hoist():
    req = ANTHROPIC.parse_chat(_stub(), {
        "max_tokens": 16, "system": "be nice",
        "messages": [{"role": "user", "content": "x"}],
    })
    assert req.messages[0].role == "system" and "be nice" in req.messages[0].content


def test_anthropic_render_chat_blocks():
    out = ANTHROPIC.render_chat(_stub(text="hi"), {
        "max_tokens": 16, "messages": [{"role": "user", "content": "x"}],
    })
    assert out["type"] == "message"
    assert out["content"][0]["type"] == "text" and out["content"][0]["text"] == "hi"
    assert out["stop_reason"] == "end_turn"


def test_anthropic_render_stream_events():
    req = ANTHROPIC.parse_chat(_stub(text="hi"), {
        "max_tokens": 16, "messages": [{"role": "user", "content": "x"}],
    })
    body = b"".join(ANTHROPIC.render_chat_stream(req)).decode()
    assert "event: message_start" in body
    assert "event: message_stop" in body


def test_anthropic_unsupported_endpoints():
    with pytest.raises(NotSupported):
        ANTHROPIC.render_catalog(("m",))
    with pytest.raises(NotSupported):
        ANTHROPIC.render_embeddings(None, {})  # type: ignore[arg-type]
