"""Pytest fixtures for the Anthropic Messages API tests.

Spawns a live `AlloyServer` in a background thread with a stub `ServedModel`
that echoes the last user message. Not shared with ollama_compat/conftest.py:
pytest discovers conftest.py per directory, and importing across dirs creates
package-loading order surprises.
"""

from __future__ import annotations

from collections.abc import Iterator
from threading import Thread

import pytest

from alloy_server import (
    AlloyServer,
    ChatMessage,
    ServedModel,
    create_server,
)


def _complete(messages: tuple[ChatMessage, ...], max_tokens: int, tools: tuple[dict, ...] = (), **kwargs) -> str:
    return f"{messages[-1].content[:max_tokens]}"


def _stream(messages: tuple[ChatMessage, ...], max_tokens: int, tools: tuple[dict, ...] = (), **kwargs) -> Iterator[str]:
    text = _complete(messages, max_tokens)
    for chunk in (text[i:i + 4] for i in range(0, len(text), 4)):
        yield chunk


def _count_tokens(text: str) -> int:
    return len(text)


def make_stub_model(name: str = "alloy-test:tiny") -> ServedModel:
    return ServedModel(name=name, complete=_complete, stream=_stream, count_tokens=_count_tokens)


@pytest.fixture()
def server() -> Iterator[AlloyServer]:
    instance = create_server("127.0.0.1", 0, chat_model=make_stub_model())
    thread = Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    try:
        yield instance
    finally:
        instance.shutdown()
        thread.join(timeout=5)
        instance.server_close()


@pytest.fixture()
def port(server: AlloyServer) -> int:
    return server.server_address[1]
