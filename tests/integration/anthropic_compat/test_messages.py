"""Integration tests for the Anthropic-compatible /v1/messages endpoint.

Verifies wire-shape against Anthropic's published API contract:
- Non-streaming response with content blocks + usage.
- SSE event sequence: message_start, content_block_start, content_block_delta(*),
  content_block_stop, message_delta, message_stop.
- System-prompt handling (string and content-block forms).
- Anthropic error envelope (`{"type": "error", "error": {"type": "...", "message": "..."}}`).
"""

from __future__ import annotations

import json
from http.client import HTTPConnection

JSON_HEADERS = {"content-type": "application/json"}


def _post(
    port: int,
    path: str,
    payload: dict,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    body = json.dumps(payload).encode()
    headers = {**JSON_HEADERS, **(headers or {})}
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("POST", path, body=body, headers=headers)
    response = conn.getresponse()
    raw = response.read()
    out_headers = {k.lower(): v for k, v in response.getheaders()}
    conn.close()
    return response.status, out_headers, raw


def _parse_sse_events(raw: bytes) -> list[tuple[str, dict]]:
    """Parse `event: <name>\\ndata: {...}\\n\\n` SSE frames into a list of
    (event_name, payload) tuples.
    """
    events: list[tuple[str, dict]] = []
    current_event: str | None = None
    current_data: str | None = None
    for line in raw.decode().splitlines():
        if line.startswith("event:"):
            current_event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            current_data = line[len("data:"):].strip()
        elif line == "":
            if current_event is not None and current_data is not None:
                events.append((current_event, json.loads(current_data)))
            current_event = None
            current_data = None
    return events


# ---- non-streaming ---------------------------------------------------------


def test_v1_messages_non_streaming_returns_anthropic_shape(port: int) -> None:
    status, _, body = _post(
        port,
        "/v1/messages",
        {
            "model": "alloy-test:tiny",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert status == 200
    payload = json.loads(body)
    assert payload["type"] == "message"
    assert payload["role"] == "assistant"
    assert payload["model"] == "alloy-test:tiny"
    assert payload["stop_reason"] == "end_turn"
    assert payload["stop_sequence"] is None
    assert payload["content"] == [{"type": "text", "text": "hello"}]
    assert payload["usage"]["input_tokens"] == 5  # len("hello") via stub counter
    assert payload["usage"]["output_tokens"] == 5
    assert payload["id"].startswith("msg_alloy_")


def test_v1_messages_with_system_string(port: int) -> None:
    status, _, body = _post(
        port,
        "/v1/messages",
        {
            "model": "alloy-test:tiny",
            "max_tokens": 32,
            "system": "You are concise.",
            "messages": [{"role": "user", "content": "echo"}],
        },
    )
    assert status == 200
    payload = json.loads(body)
    # The stub echoes the last user message regardless of the system prompt;
    # we just verify the system prompt was accepted (no 400).
    assert payload["content"] == [{"type": "text", "text": "echo"}]


def test_v1_messages_with_system_as_content_blocks(port: int) -> None:
    """Anthropic also accepts `system` as a list of content blocks."""
    status, _, body = _post(
        port,
        "/v1/messages",
        {
            "model": "alloy-test:tiny",
            "max_tokens": 32,
            "system": [{"type": "text", "text": "concise"}],
            "messages": [{"role": "user", "content": "echo"}],
        },
    )
    assert status == 200
    payload = json.loads(body)
    assert payload["content"] == [{"type": "text", "text": "echo"}]


def test_v1_messages_with_content_blocks_input(port: int) -> None:
    """User-message content can be a list of `{"type":"text", ...}` blocks."""
    status, _, body = _post(
        port,
        "/v1/messages",
        {
            "model": "alloy-test:tiny",
            "max_tokens": 32,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "part-one "},
                        {"type": "text", "text": "part-two"},
                    ],
                }
            ],
        },
    )
    assert status == 200
    payload = json.loads(body)
    # Non-text content blocks are ignored by this text-only endpoint.
    assert payload["content"] == [{"type": "text", "text": "part-one part-two"}]


def test_v1_messages_drops_unknown_blocks_silently(port: int) -> None:
    """Unknown content-block types (e.g. `document`) are dropped without
    error so future-aware clients don't break — they just receive a
    response built from the blocks we do understand. (Image/tool blocks
    are NOT dropped: they parse into vision input and tool turns.)
    """
    status, _, body = _post(
        port,
        "/v1/messages",
        {
            "model": "alloy-test:tiny",
            "max_tokens": 32,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "before "},
                        {"type": "document", "source": {"data": "b64..."}},
                        {"type": "text", "text": "after"},
                    ],
                }
            ],
        },
    )
    assert status == 200
    payload = json.loads(body)
    assert payload["content"] == [{"type": "text", "text": "before after"}]


# ---- streaming -------------------------------------------------------------


def test_v1_messages_streaming_emits_full_event_sequence(port: int) -> None:
    status, headers, body = _post(
        port,
        "/v1/messages",
        {
            "model": "alloy-test:tiny",
            "max_tokens": 32,
            "stream": True,
            "messages": [{"role": "user", "content": "streaming-test"}],
        },
    )
    assert status == 200
    assert headers["content-type"] == "text/event-stream"
    events = _parse_sse_events(body)
    names = [name for name, _ in events]
    # The exact sequence Anthropic's SDK expects.
    assert names[0] == "message_start"
    assert names[1] == "content_block_start"
    # 1+ content_block_delta entries between start and stop.
    deltas = [p for name, p in events if name == "content_block_delta"]
    assert len(deltas) >= 1
    assert names[-3] == "content_block_stop"
    assert names[-2] == "message_delta"
    assert names[-1] == "message_stop"

    # Reconstruct the streamed text from delta events.
    reconstructed = "".join(d["delta"]["text"] for d in deltas)
    assert reconstructed == "streaming-test"

    # message_start carries initial input_tokens; output_tokens is the
    # Anthropic sentinel `1` (not `0`).
    start_payload = events[0][1]
    assert start_payload["message"]["usage"]["input_tokens"] == len("streaming-test")
    assert start_payload["message"]["usage"]["output_tokens"] == 1

    # message_delta carries the final stop_reason + output_tokens.
    final_delta = events[-2][1]
    assert final_delta["delta"]["stop_reason"] == "end_turn"
    assert final_delta["delta"]["stop_sequence"] is None
    assert final_delta["usage"]["output_tokens"] > 0


# ---- errors (Anthropic envelope) -------------------------------------------


def test_v1_messages_unknown_model_returns_anthropic_404(port: int) -> None:
    status, _, body = _post(
        port,
        "/v1/messages",
        {
            "model": "nope",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "x"}],
        },
    )
    assert status == 404
    payload = json.loads(body)
    assert payload["type"] == "error"
    assert payload["error"]["type"] == "not_found_error"
    assert "nope" in payload["error"]["message"]
    # Anthropic error envelopes carry a top-level `request_id` field that
    # the SDK exposes as a property; missing it returns None.
    assert "request_id" in payload


def test_v1_messages_missing_max_tokens_returns_400(port: int) -> None:
    status, _, body = _post(
        port,
        "/v1/messages",
        {
            "model": "alloy-test:tiny",
            "messages": [{"role": "user", "content": "x"}],
        },
    )
    assert status == 400
    payload = json.loads(body)
    assert payload["error"]["type"] == "invalid_request_error"
    assert "max_tokens" in payload["error"]["message"]


def test_v1_messages_invalid_role_returns_400(port: int) -> None:
    # `system` is NOT invalid: it hoists into the leading system message
    # (Claude Code sends its skills list that way).
    status, _, body = _post(
        port,
        "/v1/messages",
        {
            "model": "alloy-test:tiny",
            "max_tokens": 16,
            "messages": [{"role": "tool", "content": "x"}],
        },
    )
    assert status == 400
    payload = json.loads(body)
    assert payload["error"]["type"] == "invalid_request_error"
    assert "user" in payload["error"]["message"]


def test_v1_messages_empty_messages_returns_400(port: int) -> None:
    status, _, body = _post(
        port,
        "/v1/messages",
        {"model": "alloy-test:tiny", "max_tokens": 16, "messages": []},
    )
    assert status == 400
    payload = json.loads(body)
    assert payload["error"]["type"] == "invalid_request_error"


def test_v1_messages_empty_content_returns_400(port: int) -> None:
    """A user message that carries nothing (empty block list) must 400 —
    matches Anthropic's API behavior."""
    status, _, body = _post(
        port,
        "/v1/messages",
        {
            "model": "alloy-test:tiny",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": []}],
        },
    )
    assert status == 400
    payload = json.loads(body)
    assert payload["error"]["type"] == "invalid_request_error"
    assert "empty" in payload["error"]["message"]


def test_v1_messages_null_content_returns_400(port: int) -> None:
    status, _, body = _post(
        port,
        "/v1/messages",
        {
            "model": "alloy-test:tiny",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": None}],
        },
    )
    assert status == 400
    payload = json.loads(body)
    assert payload["error"]["type"] == "invalid_request_error"


# ---- /v1/embeddings ----


def test_v1_embeddings_with_unregistered_model_returns_404(port: int) -> None:
    """A chat-only fixture is absent from the embedding registry."""
    status, _, body = _post(
        port,
        "/v1/embeddings",
        {"model": "alloy-test:tiny", "input": "hello"},
    )
    assert status == 404
    payload = json.loads(body)
    assert payload["error"]["type"] == "model_not_served"
