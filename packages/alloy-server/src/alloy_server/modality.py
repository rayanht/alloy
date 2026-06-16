"""The modality registry — the server's extension point.

A `Modality` bundles what varies per served kind: its healthz `kind`, the `task`
its endpoints serve, and the `Endpoint`s (path → dialect + task). One server serves
exactly one model of one modality. The transport is a generic task-router over this
registry: it resolves a path to an `Endpoint`, looks up the `(dialect, task)`
`TaskHandler`, and dispatches.

Adding a modality (e.g. transcription, image↔text) = append a `Modality` to
`MODALITIES` + register its `(dialect, task)` handlers in `DIALECT_TASKS`, with zero
edits to the dispatch core. `TaskHandler`s wrap the existing dialect methods, so the
dialect classes stay untouched.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

from alloy_server.dialects import (
    ANTHROPIC,
    OLLAMA_CHAT,
    OLLAMA_GENERATE,
    OPENAI,
    audio,
    ollama,
)
from alloy_server.dialects.base import Dialect
from alloy_server.schema import JsonObject


@dataclass(frozen=True, slots=True)
class TaskHandler:
    """How one dialect handles one task: a synchronous `render`, plus the streaming
    trio (`parse` request → `render_stream` frames, gated by `wants_stream`) for
    tasks that can stream. Embedding-style tasks set only `render`."""

    render: Callable[[object, JsonObject], JsonObject]
    parse: Callable[[object, JsonObject], object] | None = None
    render_stream: Callable[[object], Iterator[bytes]] | None = None
    wants_stream: Callable[[JsonObject], bool] | None = None
    stream_content_type: str | None = None


@dataclass(frozen=True, slots=True)
class Endpoint:
    path: str
    dialect: str  # key into DIALECT_TASKS with `task`
    task: str
    request: str = "json"  # "json" | "multipart"


@dataclass(frozen=True, slots=True)
class Modality:
    kind: str  # /healthz kind: "chat" | "embedding" | "transcription"
    tasks: frozenset[str]  # the tasks this modality's endpoints serve
    endpoints: tuple[Endpoint, ...]


def chat_task(dialect: Dialect) -> TaskHandler:
    """Wrap a dialect's chat methods (streaming-capable) as a TaskHandler."""
    return TaskHandler(
        render=dialect.render_chat,
        parse=dialect.parse_chat,
        render_stream=dialect.render_chat_stream,
        wants_stream=dialect.wants_stream,
        stream_content_type=dialect.stream_content_type,
    )


# (dialect, task) -> handler. Chat handlers wrap the dialect methods; embed handlers
# call the per-dialect render directly (no streaming).
DIALECT_TASKS: dict[tuple[str, str], TaskHandler] = {
    ("openai", "chat"): chat_task(OPENAI),
    ("ollama", "chat"): chat_task(OLLAMA_CHAT),
    ("ollama_generate", "chat"): chat_task(OLLAMA_GENERATE),
    ("anthropic", "chat"): chat_task(ANTHROPIC),
    ("openai", "embed"): TaskHandler(render=OPENAI.render_embeddings),
    ("ollama", "embed"): TaskHandler(render=OLLAMA_CHAT.render_embeddings),
    ("ollama_legacy", "embed"): TaskHandler(render=ollama.legacy_embeddings_payload),
    ("openai", "transcribe"): TaskHandler(
        render=audio.transcription_payload, parse=audio.parse_transcription,
        render_stream=audio.stream_transcription, wants_stream=audio.wants_stream,
        stream_content_type="text/event-stream",
    ),
    ("openai", "translate"): TaskHandler(
        render=audio.translation_payload, parse=audio.parse_translation,
        render_stream=audio.stream_transcription, wants_stream=audio.wants_stream,
        stream_content_type="text/event-stream",
    ),
}


CHAT = Modality(
    kind="chat", tasks=frozenset({"chat"}),
    endpoints=(
        Endpoint("/v1/chat/completions", "openai", "chat"),
        Endpoint("/api/chat", "ollama", "chat"),
        Endpoint("/api/generate", "ollama_generate", "chat"),
        Endpoint("/v1/messages", "anthropic", "chat"),
    ),
)

EMBED = Modality(
    kind="embedding", tasks=frozenset({"embed"}),
    endpoints=(
        Endpoint("/v1/embeddings", "openai", "embed"),
        Endpoint("/api/embed", "ollama", "embed"),
        Endpoint("/api/embeddings", "ollama_legacy", "embed"),
    ),
)

TRANSCRIPTION = Modality(
    kind="transcription", tasks=frozenset({"transcribe", "translate"}),
    endpoints=(
        Endpoint("/v1/audio/transcriptions", "openai", "transcribe", request="multipart"),
        Endpoint("/v1/audio/translations", "openai", "translate", request="multipart"),
    ),
)

MODALITIES: tuple[Modality, ...] = (CHAT, EMBED, TRANSCRIPTION)

# path -> Endpoint, across all modalities (every endpoint is always routed; the
# served modality decides which tasks actually run — see served_model_for).
ENDPOINT_INDEX: dict[str, Endpoint] = {ep.path: ep for m in MODALITIES for ep in m.endpoints}

# task -> human noun for the cross-kind 404 ("not a <noun> model").
TASK_NOUN: dict[str, str] = {
    "chat": "chat", "embed": "embedding", "transcribe": "transcription",
    "translate": "transcription",
}
