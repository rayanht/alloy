"""Streaming wire-frame encoders shared by the dialect render_*_stream functions.

Each returns the exact bytes for one SSE frame / NDJSON line, so the transport
just writes them — no formatting in the handler.
"""

from __future__ import annotations

import json

from alloy_server.schema import JsonObject


def sse_data(payload: JsonObject) -> bytes:
    """OpenAI-style SSE frame: `data: {json}\\n\\n`."""
    return f"data: {json.dumps(payload)}\n\n".encode()


def sse_event(event_name: str, payload: JsonObject) -> bytes:
    """Anthropic-style named SSE frame: `event: <name>\\ndata: {json}\\n\\n`.
    Clients route on the event name; missing it breaks the SDK delta accumulator."""
    return f"event: {event_name}\ndata: {json.dumps(payload)}\n\n".encode()


def sse_done() -> bytes:
    """OpenAI-only stream terminator. NOT used by Anthropic (its SDK throws on a
    `[DONE]` sentinel; `message_stop` is the canonical terminator there)."""
    return b"data: [DONE]\n\n"


def ndjson(payload: JsonObject) -> bytes:
    """Ollama NDJSON line: one compact JSON object + newline."""
    return json.dumps(payload).encode() + b"\n"
