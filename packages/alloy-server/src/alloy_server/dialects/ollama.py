"""Ollama dialect: `/api/chat`, `/api/generate`, `/api/embed`(+legacy), and the
read-only catalog (`/api/tags`, `/api/show`). Parses Ollama-shaped requests into
ChatCompletionRequest and renders Ollama-shaped responses (JSON, NDJSON stream).
Shared parse primitives from dialects.common."""

from __future__ import annotations

import datetime
import json
import time
from collections.abc import Iterator
from http import HTTPStatus
from typing import TYPE_CHECKING

from alloy_server.constrain import Constraint
from alloy_server.dialects import wire
from alloy_server.dialects.common import (
    apply_request_sampling,
    messages_field,
    parse_max_tokens,
    parse_stop,
    parse_tool_choice,
    parse_tools,
    resolve_embedding_request,
    sampling_from,
    string_field,
    tool_constraint,
)
from alloy_server.schema import (
    ChatCompletionRequest,
    ChatMessage,
    JsonObject,
    JsonValue,
    RequestError,
    ServedModel,
)
from alloy_server.result import Generation
from alloy_server.toolcalls import ollama_tool_calls

if TYPE_CHECKING:
    from alloy_server.embedding import EmbeddingModel

DEFAULT_OLLAMA_NUM_PREDICT = 2048


def format_constraint(value: JsonValue) -> Constraint | None:
    """Ollama `format`: "json" or an inline JSON schema."""
    if value == "json":
        return Constraint(kind="json")
    if isinstance(value, dict) and value:
        return Constraint(kind="json_schema", schema_json=json.dumps(value))
    return None


def enable_thinking(payload: JsonObject) -> bool | None:
    """Ollama `think`: bool, or an effort string (newer Ollama). None = absent."""
    think = payload.get("think")
    if isinstance(think, bool):
        return think
    if isinstance(think, str):
        return think.lower() not in ("", "false", "none")
    return None


def options(payload: JsonObject) -> JsonObject:
    opts = payload.get("options")
    return opts if isinstance(opts, dict) else {}


def stream_field(payload: JsonObject) -> bool:
    """Ollama defaults stream=true when the field is absent (unlike OpenAI)."""
    value = payload.get("stream", True)
    if not isinstance(value, bool):
        raise RequestError(HTTPStatus.BAD_REQUEST, "invalid_request", "stream must be a boolean")
    return value


def now() -> str:
    return datetime.datetime.now(tz=datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def max_tokens(payload: JsonObject) -> int:
    """Ollama nests generation options under `options.num_predict`.

    Default 2048 is enough for typical single-turn chat responses
    without truncation. Real Ollama defaults to `-1` (no cap until
    natural EOS or context overflow); `parse_max_tokens` has no
    "unlimited" sentinel, so 2048 is a safe middle ground.
    """
    opts = payload.get("options")
    if isinstance(opts, dict):
        value = opts.get("num_predict")
        if value is not None:
            return parse_max_tokens(value)
    return DEFAULT_OLLAMA_NUM_PREDICT


def error_payload(status: HTTPStatus, code: str, message: str) -> JsonObject:
    """Ollama-shaped error envelope: flat string in the `error` field (matches
    Ollama 0.x; clients like LangChain/llama-index/Continue parse `error` as a
    string and break on a nested object)."""
    return {"error": message}


def model_details() -> JsonObject:
    """Empty Ollama `details` envelope. Clients commonly destructure this
    so we ship the full shape with empty values rather than `{}`.
    """
    return {
        "parent_model": "",
        "format": "safetensors",
        "family": "",
        "families": [],
        "parameter_size": "",
        "quantization_level": "",
    }


def tags_payload(names: tuple[str, ...]) -> JsonObject:
    """`/api/tags` — list of installed models in Ollama's shape.

    Shows the *installed* catalog from disk (plus the served model),
    so client model pickers have something to render.
    """
    return {
        "models": [
            {
                "name": name,
                "model": name,
                "modified_at": now(),
                "size": 0,
                "digest": "",
                "details": model_details(),
            }
            for name in names
        ],
    }


def show_payload() -> JsonObject:
    """`/api/show` — model metadata. v1 returns a minimal envelope; richer
    metadata (Modelfile, template, GGUF parameter counts) is a future
    enhancement that depends on the GGUF loader exposing manifest fields.
    """
    return {
        "modelfile": "",
        "parameters": "",
        "template": "",
        "details": model_details(),
        "model_info": {},
        "capabilities": ["completion"],
    }


def chat_request(model: ServedModel, payload: JsonObject) -> ChatCompletionRequest:
    messages = messages_field(payload, "messages")
    new_tokens = max_tokens(payload)
    opts = options(payload)
    apply_request_sampling(model, sampling_from(opts))
    mode, forced = parse_tool_choice(payload.get("tool_choice"))
    tools = () if mode == "none" else parse_tools(payload.get("tools"))
    constraint = (
        format_constraint(payload.get("format"))
        or tool_constraint(tools, mode, forced)
    )
    return ChatCompletionRequest(
        model=model, messages=messages, max_tokens=new_tokens,
        stop=parse_stop(opts.get("stop")),
        tools=tools, tool_choice=mode, forced_tool=forced, constraint=constraint,
        enable_thinking=enable_thinking(payload),
    )


def generate_request(model: ServedModel, payload: JsonObject) -> ChatCompletionRequest:
    """`/api/generate` takes a `prompt` field; we wrap it as a single user
    message so the same downstream complete/stream callables apply.
    """
    prompt = string_field(payload, "prompt")
    messages = (ChatMessage(role="user", content=prompt),)
    new_tokens = max_tokens(payload)
    opts = options(payload)
    apply_request_sampling(model, sampling_from(opts))
    return ChatCompletionRequest(
        model=model, messages=messages, max_tokens=new_tokens,
        stop=parse_stop(opts.get("stop")),
        enable_thinking=enable_thinking(payload),
    )


def chat_payload(model: ServedModel, payload: JsonObject) -> JsonObject:
    request = chat_request(model, payload)
    started_ns = time.monotonic_ns()
    gen = Generation(request)
    content = gen.text()
    total_ns = time.monotonic_ns() - started_ns
    count = request.model.count_tokens
    prompt_tokens = sum(count(message.content) for message in request.messages)
    completion_tokens = count(content)
    message: JsonObject = {"role": "assistant", "content": content}
    if gen.reasoning_content:
        message["thinking"] = gen.reasoning_content  # ollama's reasoning field
    if gen.tool_calls:
        message["tool_calls"] = ollama_tool_calls(gen.tool_calls)
    return {
        "model": request.model.name,
        "created_at": now(),
        "message": message,
        "done": True,
        "done_reason": gen.ollama_done_reason(),
        "total_duration": total_ns,
        "load_duration": 0,
        "prompt_eval_count": prompt_tokens,
        "prompt_eval_duration": 0,
        "eval_count": completion_tokens,
        "eval_duration": total_ns,
    }


def generate_payload(model: ServedModel, payload: JsonObject) -> JsonObject:
    # `prompt_eval_count` here counts message content tokens only; chat-
    # template wrappers add ~10-15 tokens per turn that we don't surface in
    # v1. Tools that derive tokens/sec from this field will overestimate
    # slightly. Future enhancement: when ServedModel exposes a
    # template-aware token counter.
    request = generate_request(model, payload)
    started_ns = time.monotonic_ns()
    gen = Generation(request)
    content = gen.text()
    total_ns = time.monotonic_ns() - started_ns
    count = request.model.count_tokens
    prompt_tokens = sum(count(message.content) for message in request.messages)
    completion_tokens = count(content)
    result: JsonObject = {
        "model": request.model.name,
        "created_at": now(),
        "response": content,
        "done": True,
        "done_reason": gen.finish_reason(),
        "context": [],
        "total_duration": total_ns,
        "load_duration": 0,
        "prompt_eval_count": prompt_tokens,
        "prompt_eval_duration": 0,
        "eval_count": completion_tokens,
        "eval_duration": total_ns,
    }
    if gen.reasoning_content:
        result["thinking"] = gen.reasoning_content
    return result


def render_chat_stream(request: ChatCompletionRequest) -> Iterator[bytes]:
    """NDJSON frames for a streamed `/api/chat`: assistant message deltas
    (content / thinking), a non-final tool_calls frame, then the terminal done
    frame with timings."""
    name = request.model.name
    started_ns = time.monotonic_ns()
    prompt_tokens = sum(request.model.count_tokens(m.content) for m in request.messages)
    eval_count = 0
    gen = Generation(request)
    for kind, text in gen.stream():
        if not text:
            continue
        eval_count += request.model.count_tokens(text)
        message = ({"role": "assistant", "content": "", "thinking": text}
                   if kind == "reasoning" else {"role": "assistant", "content": text})
        yield wire.ndjson({"model": name, "created_at": now(), "message": message, "done": False})
    if gen.tool_calls:
        yield wire.ndjson({
            "model": name, "created_at": now(),
            "message": {"role": "assistant", "content": "",
                        "tool_calls": ollama_tool_calls(gen.tool_calls)},
            "done": False,
        })
    total_ns = time.monotonic_ns() - started_ns
    yield wire.ndjson({
        "model": name, "created_at": now(),
        "message": {"role": "assistant", "content": ""},
        "done": True, "done_reason": gen.ollama_done_reason(),
        "total_duration": total_ns, "load_duration": 0,
        "prompt_eval_count": prompt_tokens, "prompt_eval_duration": 0,
        "eval_count": eval_count, "eval_duration": total_ns,
    })


def render_generate_stream(request: ChatCompletionRequest) -> Iterator[bytes]:
    """NDJSON frames for a streamed `/api/generate`."""
    name = request.model.name
    started_ns = time.monotonic_ns()
    prompt_tokens = sum(request.model.count_tokens(m.content) for m in request.messages)
    eval_count = 0
    gen = Generation(request)
    for kind, text in gen.stream():
        if not text:
            continue
        eval_count += request.model.count_tokens(text)
        chunk = ({"response": "", "thinking": text} if kind == "reasoning"
                 else {"response": text})
        yield wire.ndjson({"model": name, "created_at": now(), **chunk, "done": False})
    total_ns = time.monotonic_ns() - started_ns
    yield wire.ndjson({
        "model": name, "created_at": now(), "response": "",
        "done": True, "done_reason": gen.finish_reason(), "context": [],
        "total_duration": total_ns, "load_duration": 0,
        "prompt_eval_count": prompt_tokens, "prompt_eval_duration": 0,
        "eval_count": eval_count, "eval_duration": total_ns,
    })


def embed_payload(model: "EmbeddingModel", payload: JsonObject) -> JsonObject:
    """Ollama new `/api/embed` response — supports batched input."""
    inputs = resolve_embedding_request(model, payload)
    started_ns = time.monotonic_ns()
    vectors = model.embed(inputs)
    total_ns = time.monotonic_ns() - started_ns
    prompt_tokens = sum(model.count_tokens(text) for text in inputs)
    return {
        "model": model.name,
        "embeddings": vectors,
        "total_duration": total_ns,
        "load_duration": 0,
        "prompt_eval_count": prompt_tokens,
    }


def legacy_embeddings_payload(model: "EmbeddingModel", payload: JsonObject) -> JsonObject:
    """Ollama legacy `/api/embeddings` response — single string only, flat
    `embedding` field (singular). Kept for older clients still pinned to
    the pre-`/api/embed` shape.
    """
    prompt = string_field(payload, "prompt")
    vectors = model.embed([prompt])
    return {"embedding": vectors[0]}
