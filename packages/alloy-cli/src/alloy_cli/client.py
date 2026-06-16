"""HTTP client wrapper for the alloy server.

The server (`alloy serve`) listens on `http://127.0.0.1:11434` by default
(Ollama-compatible port). This module is the single point of HTTP knowledge
in the CLI — every client command goes through `ServerClient`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 11434


def _env_host() -> str:
    return os.environ.get("ALLOY_HOST", DEFAULT_HOST)


def _env_port() -> int | None:
    """Explicit `ALLOY_PORT` env override, or None if unset."""
    raw = os.environ.get("ALLOY_PORT")
    return int(raw) if raw else None


def _base_url() -> str:
    return f"http://{_env_host()}:{_env_port() or DEFAULT_PORT}"


@dataclass(frozen=True, slots=True)
class ServerStatus:
    reachable: bool
    version: str | None
    model: str | None
    """The one model the server serves (None when unreachable)."""
    kind: str | None
    """`"chat"` or `"embedding"` (None when unreachable)."""


class ServerClient:
    """Thin sync HTTP client over the server's OpenAI/Ollama/Anthropic APIs."""

    def __init__(self, base_url: str | None = None, timeout: float = 60.0) -> None:
        self._base = base_url or _base_url()
        self._timeout = timeout

    @property
    def base_url(self) -> str:
        return self._base

    def healthz(self) -> ServerStatus:
        try:
            response = httpx.get(f"{self._base}/healthz", timeout=2.0)
        except httpx.HTTPError:
            return ServerStatus(reachable=False, version=None, model=None, kind=None)
        if response.status_code != 200:
            return ServerStatus(reachable=False, version=None, model=None, kind=None)
        payload = response.json()
        return ServerStatus(
            reachable=True,
            version=payload.get("version"),
            model=payload.get("model"),
            kind=payload.get("kind"),
        )

    def tags(self) -> dict:
        """GET /api/tags — installed models."""
        return self._get_json("/api/tags")

    def _get_json(self, path: str) -> dict:
        response = httpx.get(f"{self._base}{path}", timeout=self._timeout)
        response.raise_for_status()
        return response.json()
