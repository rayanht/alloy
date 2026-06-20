"""Pytest fixtures for the Ollama-compatible HTTP API tests.

Spawns a real `AlloyServer` in a background thread with a stub
`ServedModel`. The stub echoes the last user message and uses character
count as a token-count proxy — fast and deterministic for snapshot tests.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from threading import Thread

import pytest

import alloy_server
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
    """A live HTTP server on a random port, shut down at teardown. The
    background thread is a daemon so leaks don't keep pytest alive.
    """
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


@pytest.fixture(autouse=True)
def _reset_origins_env() -> Iterator[None]:
    """Each test gets a clean ALLOY_ORIGINS view; reverted at teardown."""
    prior = os.environ.pop("ALLOY_ORIGINS", None)
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop("ALLOY_ORIGINS", None)
        else:
            os.environ["ALLOY_ORIGINS"] = prior


@pytest.fixture(autouse=True)
def _isolate_disk_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(alloy_server, "discovered_chat_names", lambda: [])
