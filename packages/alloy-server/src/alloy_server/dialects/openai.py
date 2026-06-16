"""OpenAI dialect: `/v1/chat/completions`, `/v1/models`, `/v1/embeddings`.

Parses OpenAI-shaped requests into the dialect-agnostic ChatCompletionRequest and
renders the responses (JSON, SSE stream). Shared parse primitives come from
dialects.common; the response assembler from result.
"""

from __future__ import annotations

import json
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
    tool_constraint,
)
from alloy_server.schema import (
    ChatCompletionRequest,
    JsonObject,
    JsonValue,
    RequestError,
    ServedModel,
)
from alloy_server.result import Generation
from alloy_server.toolcalls import openai_tool_calls

if TYPE_CHECKING:
    from alloy_server.embedding import EmbeddingModel


def response_format_constraint(value: JsonValue) -> Constraint | None:
    """OpenAI response_format -> Constraint (json_object / json_schema)."""
    if not isinstance(value, dict):
        return None
    t = value.get("type")
    if t == "json_object":
        return Constraint(kind="json")
    if t == "json_schema":
        js = value.get("json_schema")
        schema = js.get("schema") if isinstance(js, dict) else None
        if isinstance(schema, dict):
            return Constraint(kind="json_schema", schema_json=json.dumps(schema))
        return Constraint(kind="json")
    return None


def enable_thinking(payload: JsonObject) -> bool | None:
    """OpenAI reasoning toggle: `chat_template_kwargs.enable_thinking` (the vLLM/
    SGLang passthrough, explicit) or `reasoning_effort` ("none" disables, any other
    level enables). None when neither is present (keep the model default)."""
    ctk = payload.get("chat_template_kwargs")
    if isinstance(ctk, dict) and isinstance(ctk.get("enable_thinking"), bool):
        return ctk["enable_thinking"]
    effort = payload.get("reasoning_effort")
    if isinstance(effort, str):
        return effort.lower() != "none"
    return None


def stream_field(payload: JsonObject) -> bool:
    value = payload.get("stream", False)
    if not isinstance(value, bool):
        raise RequestError(HTTPStatus.BAD_REQUEST, "invalid_request", "stream must be a boolean")
    return value


def chat_completion_request(model: ServedModel, payload: JsonObject) -> ChatCompletionRequest:
    messages = messages_field(payload, "messages")
    max_tokens = parse_max_tokens(payload.get("max_tokens", 16))
    apply_request_sampling(model, sampling_from(payload))
    mode, forced = parse_tool_choice(payload.get("tool_choice"))
    tools = () if mode == "none" else parse_tools(payload.get("tools"))
    # parallel_tool_calls defaults true (OpenAI); only explicit false forces single.
    single_call = payload.get("parallel_tool_calls") is False
    constraint = (
        response_format_constraint(payload.get("response_format"))
        or tool_constraint(tools, mode, forced, single_call)
    )
    return ChatCompletionRequest(
        model=model, messages=messages, max_tokens=max_tokens,
        stop=parse_stop(payload.get("stop")),
        tools=tools, tool_choice=mode, forced_tool=forced, constraint=constraint,
        enable_thinking=enable_thinking(payload),
    )


def chat_completion_chunk(model_name: str, delta: JsonObject, finish_reason: str | None) -> JsonObject:
    return {
        "id": "chatcmpl-alloy-local",
        "object": "chat.completion.chunk",
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


def render_chat_stream(request: ChatCompletionRequest) -> Iterator[bytes]:
    """SSE frames for a streamed chat completion: content / reasoning_content
    deltas, buffered tool calls as tool_call deltas (id+name, then arguments),
    a finish-reason frame, and the `[DONE]` terminator."""
    name = request.model.name
    first_chunk = True
    gen = Generation(request)
    for kind, text in gen.stream():
        if not text:
            continue
        field = "reasoning_content" if kind == "reasoning" else "content"
        delta: JsonObject = {field: text}
        if first_chunk:
            delta["role"] = "assistant"
            first_chunk = False
        yield wire.sse_data(chat_completion_chunk(name, delta, None))
    for index, tc in enumerate(gen.tool_calls):
        head: JsonObject = {"tool_calls": [
            {"index": index, "id": tc.id, "type": "function",
             "function": {"name": tc.name, "arguments": ""}},
        ]}
        if first_chunk:
            head["role"] = "assistant"
            first_chunk = False
        yield wire.sse_data(chat_completion_chunk(name, head, None))
        yield wire.sse_data(chat_completion_chunk(name, {"tool_calls": [
            {"index": index, "function": {"arguments": json.dumps(tc.arguments)}},
        ]}, None))
    yield wire.sse_data(chat_completion_chunk(name, {}, gen.finish_reason()))
    yield wire.sse_done()


def error_payload(status: HTTPStatus, code: str, message: str) -> JsonObject:
    """OpenAI-shaped error envelope: nested `error` object."""
    return {"error": {"message": message, "type": code, "code": code}}


def models_payload(names: tuple[str, ...]) -> JsonObject:
    return {
        "object": "list",
        "data": [
            {"id": name, "object": "model", "owned_by": "alloy"}
            for name in names
        ],
    }


def chat_completion_payload(model: ServedModel, payload: JsonObject) -> JsonObject:
    request = chat_completion_request(model, payload)
    result = Generation(request).complete()
    count = request.model.count_tokens
    completion_tokens = count(result.content)
    prompt_tokens = sum(count(message.content) for message in request.messages)

    message: JsonObject = {"role": "assistant", "content": result.content}
    if result.reasoning:
        # De-facto OpenAI-compatible field (vLLM/SGLang/DeepSeek) for chain-of-thought.
        message["reasoning_content"] = result.reasoning
    if result.tool_calls:
        # OpenAI sends content:null alongside tool_calls when there's no text.
        message["content"] = result.content or None
        message["tool_calls"] = openai_tool_calls(result.tool_calls)
    return {
        "id": "chatcmpl-alloy-local",
        "object": "chat.completion",
        "model": request.model.name,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": result.finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "timings": dict(result.timings),
        },
    }


def embeddings_payload(model: "EmbeddingModel", payload: JsonObject) -> JsonObject:
    """OpenAI-shape `/v1/embeddings` response."""
    inputs = resolve_embedding_request(model, payload)
    vectors = model.embed(inputs)
    prompt_tokens = sum(model.count_tokens(text) for text in inputs)
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": vec}
            for i, vec in enumerate(vectors)
        ],
        "model": model.name,
        "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
    }
