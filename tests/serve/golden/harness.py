"""Wire-format golden harness for the server.

Fires a fixed request corpus across all three dialects (OpenAI / Ollama /
Anthropic) at the live server with deterministic stub models, and records each
response as a normalized structure: status, app-meaningful headers, and the
body parsed into JSON / SSE-event-list / NDJSON-line-list with volatile fields
(uuids, timestamps, durations) masked.

The golden is STRUCTURAL, not literal bytes: streaming chunk boundaries may
legitimately differ between transports (a thread-per-request stdlib server vs an
async one), but the reconstructed event sequence, framing markers (`data: `, `[DONE]`,
Anthropic `event:` names), field set, and ordering must stay identical. That is
exactly the wire contract the dialect renderers own. Transport-injected headers
(Server, Date, Connection, Transfer-Encoding) are dropped — they differ by
transport by design; only content-type and CORS headers are kept. (An async
transport would change chunking, not the decoded frames.)

Both `capture_wire_golden.py` (writes the golden) and `test_wire_golden.py`
(replays + asserts) call `capture_all()`, so the capture and the gate can never
drift.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from http.client import HTTPConnection
from threading import Thread

import alloy_server
from alloy_server.embedding import EmbeddingModel
from alloy_server import (
    AlloyServer,
    ChatMessage,
    ServedModel,
    create_server,
)
from alloy_server.reasoning import THINK_PROTOCOL

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "wire_golden.json")

# Response keys whose values are non-deterministic (uuids, wall-clock, timing).
# Masked to a constant before diffing so the golden is stable run-to-run.
_VOLATILE_KEYS = frozenset({
    "id", "created", "created_at", "modified_at",
    "total_duration", "load_duration",
    "prompt_eval_duration", "eval_duration",
})
# Response headers that carry app meaning (dialect content-type, CORS). Every
# other header is transport-injected and dropped.
_KEPT_HEADER_PREFIXES = ("content-type", "access-control-")


# ----------------------------------------------------------------------------
# Stub models — deterministic, no GPU. The chat stub switches behaviour on the
# last user message so one model exercises text / tool-call / stop paths.
# ----------------------------------------------------------------------------

_TOOL_TEXT = (
    '<tool_call>\n{"name": "get_weather", "arguments": {"location": "Paris"}}\n</tool_call>'
)


def _chat_text(messages: tuple[ChatMessage, ...]) -> str:
    last = messages[-1].content if messages else ""
    if "weather" in last.lower():
        return _TOOL_TEXT
    return f"echo: {last}"


def _make_chat_stub(name: str, *, reasoning: bool) -> ServedModel:
    prefix = "<think>let me consider this</think>" if reasoning else ""

    def complete(messages, max_tokens, tools=(), **kwargs) -> str:
        return prefix + _chat_text(messages)

    def stream(messages, max_tokens, tools=(), **kwargs) -> Iterator[str]:
        text = prefix + _chat_text(messages)
        for i in range(0, len(text), 5):
            yield text[i:i + 5]

    return ServedModel(
        name=name,
        complete=complete,
        stream=stream,
        count_tokens=len,
        reasoning=THINK_PROTOCOL if reasoning else None,
    )


def _make_embed_stub(name: str = "alloy-test:embed", dim: int = 4) -> EmbeddingModel:
    def embed(texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * dim
            for i, ch in enumerate(t):
                v[i % dim] += float(ord(ch))
            out.append([round(x, 4) for x in v])
        return out

    return EmbeddingModel(
        name=name, embed=embed, dimensions=dim, count_tokens=len, max_batch=8,
    )


# ----------------------------------------------------------------------------
# Request corpus
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Case:
    key: str                 # golden record id
    server: str              # "chat" | "reasoning" | "embed"
    method: str              # "GET" | "POST" | "OPTIONS"
    path: str
    body: dict | None = None
    headers: dict = field(default_factory=dict)


def _chat_body(extra: dict) -> dict:
    base = {"model": "alloy-test:tiny", "messages": [{"role": "user", "content": "hello"}]}
    base.update(extra)
    return base


_WEATHER_TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    },
}]


def corpus() -> list[Case]:
    cases: list[Case] = []

    # ---- OpenAI ----
    cases += [
        Case("openai.models", "chat", "GET", "/v1/models"),
        Case("openai.chat.text", "chat", "POST", "/v1/chat/completions",
             _chat_body({"stream": False})),
        Case("openai.chat.stream", "chat", "POST", "/v1/chat/completions",
             _chat_body({"stream": True})),
        Case("openai.chat.stop", "chat", "POST", "/v1/chat/completions",
             {"model": "alloy-test:tiny",
              "messages": [{"role": "user", "content": "alpha beta gamma"}],
              "stream": False, "stop": ["beta"]}),
        Case("openai.chat.tools", "chat", "POST", "/v1/chat/completions",
             {"model": "alloy-test:tiny",
              "messages": [{"role": "user", "content": "what is the weather"}],
              "tools": _WEATHER_TOOLS, "stream": False}),
        Case("openai.chat.tools.stream", "chat", "POST", "/v1/chat/completions",
             {"model": "alloy-test:tiny",
              "messages": [{"role": "user", "content": "what is the weather"}],
              "tools": _WEATHER_TOOLS, "stream": True}),
        Case("openai.chat.reasoning", "reasoning", "POST", "/v1/chat/completions",
             _chat_body({"stream": False})),
        Case("openai.chat.reasoning.stream", "reasoning", "POST", "/v1/chat/completions",
             _chat_body({"stream": True})),
        Case("openai.chat.unknown_model", "chat", "POST", "/v1/chat/completions",
             {"model": "nope", "messages": [{"role": "user", "content": "x"}]}),
        Case("openai.embeddings", "embed", "POST", "/v1/embeddings",
             {"model": "alloy-test:embed", "input": ["hello", "world"]}),
    ]

    # ---- Ollama ----
    cases += [
        Case("ollama.version", "chat", "GET", "/api/version"),
        Case("ollama.tags", "chat", "GET", "/api/tags"),
        Case("ollama.show", "chat", "POST", "/api/show", {"name": "alloy-test:tiny"}),
        Case("ollama.show.unknown", "chat", "POST", "/api/show", {"name": "nope"}),
        Case("ollama.chat.text", "chat", "POST", "/api/chat",
             {"model": "alloy-test:tiny",
              "messages": [{"role": "user", "content": "hello"}], "stream": False}),
        Case("ollama.chat.stream", "chat", "POST", "/api/chat",
             {"model": "alloy-test:tiny",
              "messages": [{"role": "user", "content": "streaming"}], "stream": True}),
        Case("ollama.chat.stop", "chat", "POST", "/api/chat",
             {"model": "alloy-test:tiny",
              "messages": [{"role": "user", "content": "alpha beta gamma"}],
              "stream": False, "options": {"stop": ["beta"]}}),
        Case("ollama.chat.tools", "chat", "POST", "/api/chat",
             {"model": "alloy-test:tiny",
              "messages": [{"role": "user", "content": "what is the weather"}],
              "tools": _WEATHER_TOOLS, "stream": False}),
        Case("ollama.generate.text", "chat", "POST", "/api/generate",
             {"model": "alloy-test:tiny", "prompt": "hello", "stream": False}),
        Case("ollama.generate.stream", "chat", "POST", "/api/generate",
             {"model": "alloy-test:tiny", "prompt": "hello", "stream": True}),
        Case("ollama.embed", "embed", "POST", "/api/embed",
             {"model": "alloy-test:embed", "input": ["hello", "world"]}),
        Case("ollama.embeddings.legacy", "embed", "POST", "/api/embeddings",
             {"model": "alloy-test:embed", "prompt": "hello"}),
        Case("healthz", "chat", "GET", "/healthz"),
    ]

    # ---- Anthropic ----
    cases += [
        Case("anthropic.messages.text", "chat", "POST", "/v1/messages",
             {"model": "alloy-test:tiny", "max_tokens": 64,
              "messages": [{"role": "user", "content": "hello"}]}),
        Case("anthropic.messages.stream", "chat", "POST", "/v1/messages",
             {"model": "alloy-test:tiny", "max_tokens": 64, "stream": True,
              "messages": [{"role": "user", "content": "hello"}]}),
        Case("anthropic.messages.system", "chat", "POST", "/v1/messages",
             {"model": "alloy-test:tiny", "max_tokens": 64, "system": "be concise",
              "messages": [{"role": "user", "content": "hello"}]}),
        Case("anthropic.messages.tools", "chat", "POST", "/v1/messages",
             {"model": "alloy-test:tiny", "max_tokens": 64,
              "tools": [{"name": "get_weather", "description": "Get weather",
                         "input_schema": {"type": "object",
                                          "properties": {"location": {"type": "string"}}}}],
              "messages": [{"role": "user", "content": "what is the weather"}]}),
        Case("anthropic.messages.reasoning", "reasoning", "POST", "/v1/messages",
             {"model": "alloy-test:tiny", "max_tokens": 64,
              "messages": [{"role": "user", "content": "hello"}]}),
    ]

    # ---- CORS ----
    cases += [
        Case("cors.preflight", "chat", "OPTIONS", "/v1/chat/completions",
             headers={"Origin": "http://localhost:3000",
                      "Access-Control-Request-Method": "POST"}),
        Case("cors.get_with_origin", "chat", "GET", "/v1/models",
             headers={"Origin": "http://localhost:3000"}),
    ]

    return cases


# ----------------------------------------------------------------------------
# Normalization
# ----------------------------------------------------------------------------

def _mask(value):
    if isinstance(value, dict):
        return {k: ("<volatile>" if k in _VOLATILE_KEYS else _mask(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [_mask(v) for v in value]
    return value


def _parse_body(content_type: str, raw: bytes):
    """Body → JSON object / SSE event list / NDJSON line list, volatile-masked."""
    text = raw.decode("utf-8", errors="replace")
    if "text/event-stream" in content_type:
        events = []
        for block in text.split("\n\n"):
            block = block.strip("\n")
            if not block:
                continue
            event_name = None
            data = None
            for line in block.split("\n"):
                if line.startswith("event:"):
                    event_name = line[len("event:"):].strip()
                elif line.startswith("data:"):
                    data = line[len("data:"):].strip()
            if data == "[DONE]":
                events.append({"event": event_name, "data": "[DONE]"})
            elif data is not None:
                events.append({"event": event_name, "data": _mask(json.loads(data))})
        return {"kind": "sse", "events": events}
    if "x-ndjson" in content_type:
        lines = [_mask(json.loads(ln)) for ln in text.splitlines() if ln.strip()]
        return {"kind": "ndjson", "lines": lines}
    if not text:
        return {"kind": "empty"}
    try:
        return {"kind": "json", "value": _mask(json.loads(text))}
    except json.JSONDecodeError:
        return {"kind": "text", "value": text}


def _kept_headers(headers) -> dict:
    out = {}
    for key, val in headers.items():
        kl = key.lower()
        if any(kl.startswith(p) for p in _KEPT_HEADER_PREFIXES):
            out[kl] = val
    return dict(sorted(out.items()))


# ----------------------------------------------------------------------------
# Server lifecycle + firing
# ----------------------------------------------------------------------------

@contextmanager
def _serve(model_kwargs: dict) -> Iterator[int]:
    server: AlloyServer = create_server("127.0.0.1", 0, **model_kwargs)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _fire(port: int, case: Case) -> dict:
    conn = HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        headers = dict(case.headers)
        body = None
        if case.body is not None:
            body = json.dumps(case.body).encode()
            headers.setdefault("Content-Type", "application/json")
        conn.request(case.method, case.path, body=body, headers=headers)
        resp = conn.getresponse()
        ctype = resp.headers.get("content-type", "")
        raw = resp.read()
        return {
            "status": resp.status,
            "headers": _kept_headers(resp.headers),
            "body": _parse_body(ctype, raw),
        }
    finally:
        conn.close()


@contextmanager
def _isolated_discovery() -> Iterator[None]:
    """Neutralize on-disk model discovery so /api/tags and /v1/models list only
    the served stub (otherwise the golden depends on what's cached on the box)."""
    saved = alloy_server.discovered_chat_names
    alloy_server.discovered_chat_names = lambda: []
    try:
        yield
    finally:
        alloy_server.discovered_chat_names = saved


def capture_all() -> dict:
    """Build the three stub servers, fire the corpus, return {case_key: record}."""
    cases = corpus()
    with _isolated_discovery(), ExitStack() as stack:
        ports = {
            "chat": stack.enter_context(
                _serve({"chat_model": _make_chat_stub("alloy-test:tiny", reasoning=False)})),
            "reasoning": stack.enter_context(
                _serve({"chat_model": _make_chat_stub("alloy-test:tiny", reasoning=True)})),
            "embed": stack.enter_context(
                _serve({"embedding_model": _make_embed_stub()})),
        }
        records: dict[str, dict] = {}
        for case in cases:
            records[case.key] = _fire(ports[case.server], case)
    return records
