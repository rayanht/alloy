"""Port-collision detection (spec §7.4) + clean-error regression.

Two server launch regressions:

  1. When the bind port is already taken, `socketserver` calls
     `self.server_close()` on the half-constructed instance — any
     attribute that cleanup path touches must exist before
     `super().__init__()`, or the port collision surfaces as an
     unhelpful AttributeError. This test pins clean failure.

  2. We weren't detecting an Ollama squatter at all — launchd just
     crash-looped on `OSError: Address already in use`. Spec §7.4
     requires us to probe `/api/version` and emit a clear error.
"""

from __future__ import annotations

import json
import socket
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import pytest

from alloy_server import (
    AlloyServer,
    PortCollisionError,
    ServedModel,
    check_port_collision,
)


class OllamaStub(BaseHTTPRequestHandler):
    """Minimal stand-in for an Ollama install on port 11434."""

    def do_GET(self) -> None:  # noqa: N802 -- BaseHTTPRequestHandler convention
        if self.path == "/api/version":
            body = json.dumps({"version": "0.5.4"}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return  # silence


class AlloyStub(BaseHTTPRequestHandler):
    """Stand-in for another alloy server already on the port."""

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/version":
            body = json.dumps({"version": "0.5.4-alloy"}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return


@pytest.fixture
def ollama_squatter() -> Iterator[int]:
    srv = HTTPServer(("127.0.0.1", 0), OllamaStub)
    thread = Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield srv.server_port
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=2.0)


@pytest.fixture
def alloy_squatter() -> Iterator[int]:
    srv = HTTPServer(("127.0.0.1", 0), AlloyStub)
    thread = Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield srv.server_port
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=2.0)


# ----- spec §7.4 collision detection --------------------------------------


def test_free_port_is_a_noop() -> None:
    """On a free port, the check must return silently."""
    check_port_collision("127.0.0.1", 1)  # privileged port — refused, treated as free


def test_ollama_on_port_raises_with_actionable_message(
    ollama_squatter: int,
) -> None:
    with pytest.raises(PortCollisionError) as info:
        check_port_collision("127.0.0.1", ollama_squatter)
    message = str(info.value)
    assert "Ollama" in message
    # The three remediation paths from the spec must all appear.
    assert "alloy serve --port" in message
    assert "launchctl unload" in message
    assert "uninstall Ollama" in message


def test_alloy_squatter_reports_alloy_self_collision(
    alloy_squatter: int,
) -> None:
    """Another alloy server answers `/api/version` with `*-alloy` —
    we must NOT call this Ollama OR "another process (not Ollama)";
    those messages confused users who hit a stale server. The
    alloy-self branch points them at the right remediation."""
    with pytest.raises(PortCollisionError) as info:
        check_port_collision("127.0.0.1", alloy_squatter)
    message = str(info.value)
    assert "another alloy server" in message
    assert "--port" in message
    # And explicitly NOT the Ollama variant — different remediations.
    assert "Ollama" not in message
    assert "launchctl unload" not in message


def test_non_http_squatter_raises_generic_collision() -> None:
    """A non-HTTP process on the port (e.g. raw TCP). We can't probe
    `/api/version`, so the message is the generic "another process"
    variant."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    port = sock.getsockname()[1]
    try:
        with pytest.raises(PortCollisionError) as info:
            check_port_collision("127.0.0.1", port)
        message = str(info.value)
        # Non-HTTP squatter takes the same non-Ollama branch.
        assert "not Ollama" in message
        assert "alloy serve --port" not in message
    finally:
        sock.close()


# ----- AttributeError regression on failed bind ---------------------------


def test_failed_bind_raises_port_collision_not_attribute_error() -> None:
    """A port-in-use bind triggers `server_close()` cleanup on the
    half-constructed instance; the visible failure must be the friendly
    `PortCollisionError` (same type `check_port_collision` emits), not
    a raw `OSError(EADDRINUSE)` or an AttributeError from cleanup
    touching not-yet-initialized attributes.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    port = sock.getsockname()[1]
    try:
        with pytest.raises(PortCollisionError) as info:
            AlloyServer(
                "127.0.0.1", port,
                chat_model=ServedModel(
                    name="t",
                    complete=lambda messages, max_tokens, tools=(), **kw: "",
                    stream=lambda messages, max_tokens, tools=(), **kw: iter([""]),
                    count_tokens=len,
                ),
            )
        assert f"port {port}" in str(info.value)
        assert "lsof" in str(info.value)
    finally:
        sock.close()
