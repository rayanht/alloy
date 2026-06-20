"""The Dialect protocol: the uniform surface the handler dispatches a request
through, one implementation per wire format (OpenAI / Ollama / Anthropic).

A path resolves to a Dialect and the handler drives the chat pipeline uniformly:
wants_stream -> parse_chat -> render_chat / render_chat_stream, with render_error
for the dialect's error envelope. Endpoints a given dialect does not serve
(Anthropic has no embeddings or catalog) raise NotSupported.
"""

from __future__ import annotations

from collections.abc import Iterator
from http import HTTPStatus
from typing import TYPE_CHECKING, Protocol

from alloy_server.schema import ChatCompletionRequest, JsonObject, ServedModel

if TYPE_CHECKING:
    from alloy_server.embedding import EmbeddingModel


class NotSupported(Exception):
    """Raised by a Dialect for an endpoint it does not serve."""


class Dialect(Protocol):
    name: str
    stream_content_type: str

    def wants_stream(self, payload: JsonObject) -> bool: ...

    def parse_chat(self, model: ServedModel, payload: JsonObject) -> ChatCompletionRequest: ...

    def render_chat(self, model: ServedModel, payload: JsonObject) -> JsonObject: ...

    def render_chat_stream(self, request: ChatCompletionRequest) -> Iterator[bytes]: ...

    def render_error(self, status: HTTPStatus, code: str, message: str) -> JsonObject: ...

    def render_catalog(self, names: tuple[str, ...]) -> JsonObject: ...

    def render_embeddings(self, model: "EmbeddingModel", payload: JsonObject) -> JsonObject: ...
