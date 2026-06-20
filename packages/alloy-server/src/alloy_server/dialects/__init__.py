"""Wire-format dialects and the path -> Dialect routing.

`chat_dialect_for_path` resolves a chat endpoint to its dialect;
`error_dialect_for_path` resolves any path to the dialect whose error envelope it
answers with.
"""

from __future__ import annotations

from collections.abc import Iterator
from http import HTTPStatus
from typing import TYPE_CHECKING

from alloy_server.dialects import anthropic, ollama, openai
from alloy_server.dialects.base import Dialect, NotSupported
from alloy_server.schema import ChatCompletionRequest, JsonObject, ServedModel

if TYPE_CHECKING:
    from alloy_server.embedding import EmbeddingModel


class OpenAIDialect:
    name = "openai"
    stream_content_type = "text/event-stream"

    def wants_stream(self, payload: JsonObject) -> bool:
        return openai.stream_field(payload)

    def parse_chat(self, model: ServedModel, payload: JsonObject) -> ChatCompletionRequest:
        return openai.chat_completion_request(model, payload)

    def render_chat(self, model: ServedModel, payload: JsonObject) -> JsonObject:
        return openai.chat_completion_payload(model, payload)

    def render_chat_stream(self, request: ChatCompletionRequest) -> Iterator[bytes]:
        return openai.render_chat_stream(request)

    def render_error(self, status: HTTPStatus, code: str, message: str) -> JsonObject:
        return openai.error_payload(status, code, message)

    def render_catalog(self, names: tuple[str, ...]) -> JsonObject:
        return openai.models_payload(names)

    def render_embeddings(self, model: "EmbeddingModel", payload: JsonObject) -> JsonObject:
        return openai.embeddings_payload(model, payload)


class OllamaChatDialect:
    name = "ollama"
    stream_content_type = "application/x-ndjson"

    def wants_stream(self, payload: JsonObject) -> bool:
        return ollama.stream_field(payload)

    def parse_chat(self, model: ServedModel, payload: JsonObject) -> ChatCompletionRequest:
        return ollama.chat_request(model, payload)

    def render_chat(self, model: ServedModel, payload: JsonObject) -> JsonObject:
        return ollama.chat_payload(model, payload)

    def render_chat_stream(self, request: ChatCompletionRequest) -> Iterator[bytes]:
        return ollama.render_chat_stream(request)

    def render_error(self, status: HTTPStatus, code: str, message: str) -> JsonObject:
        return ollama.error_payload(status, code, message)

    def render_catalog(self, names: tuple[str, ...]) -> JsonObject:
        return ollama.tags_payload(names)

    def render_embeddings(self, model: "EmbeddingModel", payload: JsonObject) -> JsonObject:
        return ollama.embed_payload(model, payload)


class OllamaGenerateDialect(OllamaChatDialect):
    """`/api/generate` — same wire family as `/api/chat`, prompt-shaped request +
    `response` payload."""

    def parse_chat(self, model: ServedModel, payload: JsonObject) -> ChatCompletionRequest:
        return ollama.generate_request(model, payload)

    def render_chat(self, model: ServedModel, payload: JsonObject) -> JsonObject:
        return ollama.generate_payload(model, payload)

    def render_chat_stream(self, request: ChatCompletionRequest) -> Iterator[bytes]:
        return ollama.render_generate_stream(request)


class AnthropicDialect:
    name = "anthropic"
    stream_content_type = "text/event-stream"

    def wants_stream(self, payload: JsonObject) -> bool:
        return openai.stream_field(payload)  # Anthropic stream defaults false, like OpenAI

    def parse_chat(self, model: ServedModel, payload: JsonObject) -> ChatCompletionRequest:
        return anthropic.messages_request(model, payload)

    def render_chat(self, model: ServedModel, payload: JsonObject) -> JsonObject:
        return anthropic.messages_payload(model, payload)

    def render_chat_stream(self, request: ChatCompletionRequest) -> Iterator[bytes]:
        return anthropic.render_messages_stream(request)

    def render_error(self, status: HTTPStatus, code: str, message: str) -> JsonObject:
        return anthropic.error_payload(status, code, message)

    def render_catalog(self, names: tuple[str, ...]) -> JsonObject:
        raise NotSupported("anthropic /v1/messages has no catalog endpoint")

    def render_embeddings(self, model: "EmbeddingModel", payload: JsonObject) -> JsonObject:
        raise NotSupported("anthropic /v1/messages has no embeddings endpoint")


OPENAI: Dialect = OpenAIDialect()
OLLAMA_CHAT: Dialect = OllamaChatDialect()
OLLAMA_GENERATE: Dialect = OllamaGenerateDialect()
ANTHROPIC: Dialect = AnthropicDialect()

# Chat endpoints -> the dialect that parses + renders them.
CHAT_DIALECTS: dict[str, Dialect] = {
    "/v1/chat/completions": OPENAI,
    "/api/chat": OLLAMA_CHAT,
    "/api/generate": OLLAMA_GENERATE,
    "/v1/messages": ANTHROPIC,
}


def chat_dialect_for_path(path: str) -> Dialect | None:
    return CHAT_DIALECTS.get(path)


def error_dialect_for_path(path: str) -> Dialect:
    """The dialect whose error envelope a failed request on `path` answers with."""
    if path == "/v1/messages":
        return ANTHROPIC
    if path.startswith("/api/") or path == "/healthz":
        return OLLAMA_CHAT
    return OPENAI
