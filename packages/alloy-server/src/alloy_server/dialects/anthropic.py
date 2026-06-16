"""Anthropic dialect: `/v1/messages`.

Parses Anthropic-shaped requests (system field + content blocks: text, image,
tool_use, tool_result) into ChatCompletionRequest and renders Anthropic-shaped
responses (JSON, SSE stream). Shared parse primitives from dialects.common; the
response assembler from result.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from http import HTTPStatus
from typing import cast

from alloy_server.dialects import wire
from alloy_server.dialects.common import (
    apply_request_sampling,
    decode_anthropic_image_source,
    parse_max_tokens,
    parse_stop,
    sampling_from,
    tool_constraint,
)
from alloy_server.schema import (
    ChatCompletionRequest,
    ChatMessage,
    JsonObject,
    JsonValue,
    RequestError,
    ServedModel,
    ToolCall,
)
from alloy_server.result import Generation

# Claude Code prepends a system block of billing telemetry whose `cch=` hash
# CHANGES EVERY REQUEST: "x-anthropic-billing-header: cc_version=...; cch=..;".
# It's transport metadata, not prompt content — and the per-request mutation
# poisons warm prefill: `try_reconstruct_warm_input_ids` requires prior messages
# byte-identical, so one volatile system block forces a full cold re-prefill of
# the whole (huge) system+tools prompt on every single turn.
VOLATILE_SYSTEM_BLOCK_PREFIXES = ("x-anthropic-billing-header:",)

# Anthropic's error envelope uses a small fixed set of `type` strings. Map our
# internal RequestError codes onto them so /v1/messages emits the shape the SDK
# expects.
ERROR_TYPES: dict[str, str] = {
    "model_not_found": "not_found_error",
    "model_not_served": "not_found_error",
    "not_found": "not_found_error",
    "invalid_request": "invalid_request_error",
    "invalid_json": "invalid_request_error",
    "not_supported": "api_error",
}


def error_payload(status: HTTPStatus, code: str, message: str) -> JsonObject:
    """Anthropic-shaped error envelope: nested `error` with `type` from a fixed
    enum and a top-level `request_id` (the SDK pattern-matches on the type)."""
    return {
        "type": "error",
        "error": {"type": ERROR_TYPES.get(code, "api_error"), "message": message},
        "request_id": None,
    }


def tool_choice(value: JsonValue) -> tuple[str, str | None]:
    """Anthropic tool_choice ({type: auto|any|none|tool}) -> (mode, forced_tool)."""
    if isinstance(value, dict):
        t = value.get("type")
        if t == "any":
            return "required", None
        if t == "none":
            return "none", None
        if t == "tool" and isinstance(value.get("name"), str):
            return "required", value["name"]
    return "auto", None


def enable_thinking(value: JsonValue) -> bool | None:
    """Anthropic `thinking`: {type: "enabled"|"disabled"}. None = absent."""
    if isinstance(value, dict):
        t = value.get("type")
        if t == "enabled":
            return True
        if t == "disabled":
            return False
    return None


def tools(value: JsonValue) -> tuple[dict, ...]:
    """Normalize Anthropic tools ([{name, description, input_schema}]) to the
    OpenAI/template shape the chat template renders."""
    if not isinstance(value, list):
        return ()
    out: list[dict] = []
    for t in value:
        if not isinstance(t, dict) or not isinstance(t.get("name"), str):
            continue
        out.append({"type": "function", "function": {
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": t.get("input_schema", {}),
        }})
    return tuple(out)


def flatten_content(content: JsonValue) -> str:
    """Anthropic content is either a plain string or a list of content blocks.

    Text-only flatten for content that must reduce to a string: the `system`
    field, system-role messages, and tool_result content. Non-text blocks are
    silently dropped (the one real gap: images nested inside a tool_result).
    User/assistant messages do NOT come through here — `message_to_chat` handles
    their image/tool_use/tool_result blocks. Concatenation matches what the model
    would see if the blocks were laid out in order.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    pieces: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_obj = cast(JsonObject, block)
        if block_obj.get("type") != "text":
            continue
        text = block_obj.get("text")
        if isinstance(text, str):
            pieces.append(text)
    return "".join(pieces)


def system_text(content: JsonValue) -> str:
    """System content flatten: `flatten_content` minus transport telemetry blocks
    (see `VOLATILE_SYSTEM_BLOCK_PREFIXES`)."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    kept: list[JsonValue] = []
    for block in content:
        if isinstance(block, dict):
            text = block.get("text")
            if isinstance(text, str) and text.startswith(VOLATILE_SYSTEM_BLOCK_PREFIXES):
                continue
        kept.append(block)
    return flatten_content(kept)


def tool_result_text(content: JsonValue) -> str:
    """A tool_result's content is a string or a list of text blocks."""
    if isinstance(content, str):
        return content
    return flatten_content(content)


def message_to_chat(role: str, content: JsonValue) -> list[ChatMessage]:
    """Map one Anthropic message to chat messages, expanding tool_use (assistant
    function calls) and tool_result (a user message that becomes a tool turn).
    Returns [] for genuinely empty content so the caller can 400 it."""
    if isinstance(content, str):
        return [ChatMessage(role=role, content=content)] if content else []
    if not isinstance(content, list):
        return []
    text_parts: list[str] = []
    calls: list[ToolCall] = []
    tool_results: list[tuple[str | None, str]] = []
    images: list[bytes] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text" and isinstance(block.get("text"), str):
            text_parts.append(block["text"])
        elif btype == "image":
            source = block.get("source")
            if isinstance(source, dict):
                images.append(decode_anthropic_image_source(source))
        elif btype == "tool_use" and isinstance(block.get("name"), str):
            tid = block.get("id")
            calls.append(ToolCall(
                id=tid if isinstance(tid, str) else f"call_{uuid.uuid4().hex[:24]}",
                name=block["name"],
                arguments=block.get("input", {}) if isinstance(block.get("input"), dict) else {},
            ))
        elif btype == "tool_result":
            tuid = block.get("tool_use_id")
            tool_results.append((
                tuid if isinstance(tuid, str) else None,
                tool_result_text(block.get("content")),
            ))
    out: list[ChatMessage] = []
    # tool_result blocks (sent on a user message) become tool-role turns.
    for tuid, txt in tool_results:
        out.append(ChatMessage(role="tool", content=txt, tool_call_id=tuid))
    text = "".join(text_parts)
    # Emit the text/tool_calls/image message only if it carries something. A pure
    # tool_result turn produced only the tool-role messages above; a message
    # with no text, tool_use, tool_result, or image yields [] (empty -> 400).
    if text or calls or images:
        out.append(ChatMessage(
            role=role, content=text, tool_calls=tuple(calls), images=tuple(images),
        ))
    return out


def messages_request(model: ServedModel, payload: JsonObject) -> ChatCompletionRequest:
    """Translate an Anthropic `/v1/messages` request to our internal shape.

    Anthropic uses a `system` field (string or list of content blocks) plus
    `messages` with role in {user, assistant, system}. User/assistant content
    blocks go through `message_to_chat` (text, image, tool_use, tool_result).
    System content — the `system` field plus any system-role messages in the
    array (Claude Code sends its skills list that way) — is flattened to text and
    hoisted into ONE leading system message: chat templates only represent a
    leading system turn (qwen3.5 raises "System message must be at the
    beginning"), so an in-place mapping would 400 at encode time.
    """
    max_tokens_value = payload.get("max_tokens")
    if max_tokens_value is None:
        raise RequestError(
            HTTPStatus.BAD_REQUEST, "invalid_request", "max_tokens is required",
        )
    max_tokens = parse_max_tokens(max_tokens_value)

    messages_raw = payload.get("messages")
    if not isinstance(messages_raw, list) or not messages_raw:
        raise RequestError(
            HTTPStatus.BAD_REQUEST, "invalid_request", "messages must be a non-empty list",
        )

    system_parts: list[str] = []
    system = payload.get("system")
    if isinstance(system, str) and system:
        system_parts.append(system)
    elif isinstance(system, list):
        text = system_text(system)
        if text:
            system_parts.append(text)

    conversation: list[ChatMessage] = []
    for item in messages_raw:
        if not isinstance(item, dict):
            raise RequestError(
                HTTPStatus.BAD_REQUEST, "invalid_request", "each message must be an object",
            )
        message = cast(JsonObject, item)
        role = message.get("role")
        if role == "system":
            text = system_text(message.get("content"))
            if not text:
                raise RequestError(
                    HTTPStatus.BAD_REQUEST, "invalid_request",
                    "message content must not be empty (image blocks alone are not supported in v1)",
                )
            system_parts.append(text)
            continue
        if role not in ("user", "assistant"):
            raise RequestError(
                HTTPStatus.BAD_REQUEST, "invalid_request",
                "message role must be 'user', 'assistant', or 'system'",
            )
        expanded = message_to_chat(cast(str, role), message.get("content"))
        # A message must carry SOMETHING — text, a tool_use, or a tool_result.
        if not expanded:
            raise RequestError(
                HTTPStatus.BAD_REQUEST, "invalid_request",
                "message content must not be empty (image blocks alone are not supported in v1)",
            )
        conversation.extend(expanded)

    chat_messages: list[ChatMessage] = []
    if system_parts:
        chat_messages.append(ChatMessage(role="system", content="\n\n".join(system_parts)))
    chat_messages.extend(conversation)

    # Anthropic exposes temperature / top_p / top_k (no min_p).
    apply_request_sampling(model, sampling_from(payload, min_p=False))
    tc = payload.get("tool_choice")
    mode, forced = tool_choice(tc)
    request_tools = () if mode == "none" else tools(payload.get("tools"))
    # disable_parallel_tool_use lives INSIDE the tool_choice object (Anthropic):
    # at-most-one under auto, exactly-one under any/tool. A named {type:tool} forces
    # the tool but is parallel-capable by default (named_exactly_one=False), unlike
    # OpenAI's {type:function} which is exactly-one.
    single_call = isinstance(tc, dict) and tc.get("disable_parallel_tool_use") is True
    return ChatCompletionRequest(
        model=model, messages=tuple(chat_messages), max_tokens=max_tokens,
        stop=parse_stop(payload.get("stop_sequences")),
        tools=request_tools, tool_choice=mode, forced_tool=forced,
        constraint=tool_constraint(request_tools, mode, forced, single_call, named_exactly_one=False),
        enable_thinking=enable_thinking(payload.get("thinking")),
    )


def messages_payload(model: ServedModel, payload: JsonObject) -> JsonObject:
    request = messages_request(model, payload)
    gen = Generation(request)
    content = gen.text()
    count = request.model.count_tokens
    prompt_tokens = sum(count(message.content) for message in request.messages)
    completion_tokens = count(content)
    stop_reason, stop_sequence = gen.anthropic_stop()
    blocks: list[JsonObject] = []
    if gen.reasoning_content:
        # Anthropic surfaces reasoning as a thinking block, ordered before text.
        blocks.append({"type": "thinking", "thinking": gen.reasoning_content})
    if content or not gen.tool_calls:
        blocks.append({"type": "text", "text": content})
    for tc in gen.tool_calls:
        blocks.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments})
    return {
        "id": f"msg_alloy_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": blocks,
        "model": request.model.name,
        "stop_reason": stop_reason,
        "stop_sequence": stop_sequence,
        "usage": {"input_tokens": prompt_tokens, "output_tokens": completion_tokens},
    }


def render_messages_stream(request: ChatCompletionRequest) -> Iterator[bytes]:
    """Anthropic SSE event stream: message_start, lazily-opened content blocks
    (thinking / text / tool_use) with deltas, then message_delta + message_stop.
    `output_tokens` starts at the sentinel 1 (Anthropic convention). A mid-stream
    generation error becomes an `event: error` frame — headers are already
    committed, so a JSON error body is no longer possible."""
    msg_id = f"msg_alloy_{uuid.uuid4().hex[:24]}"
    prompt_tokens = sum(request.model.count_tokens(m.content) for m in request.messages)
    yield wire.sse_event("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id, "type": "message", "role": "assistant", "content": [],
            "model": request.model.name, "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": prompt_tokens, "output_tokens": 1},
        },
    })
    output_tokens = 0
    cur_kind: str | None = None
    block_index = -1
    try:
        gen = Generation(request)
        for kind, text in gen.stream():
            if not text:
                continue
            output_tokens += request.model.count_tokens(text)
            if kind != cur_kind:
                if cur_kind is not None:
                    yield wire.sse_event("content_block_stop",
                                         {"type": "content_block_stop", "index": block_index})
                block_index += 1
                block = ({"type": "thinking", "thinking": ""} if kind == "reasoning"
                         else {"type": "text", "text": ""})
                yield wire.sse_event("content_block_start", {
                    "type": "content_block_start", "index": block_index, "content_block": block})
                cur_kind = kind
            delta = ({"type": "thinking_delta", "thinking": text} if kind == "reasoning"
                     else {"type": "text_delta", "text": text})
            yield wire.sse_event("content_block_delta", {
                "type": "content_block_delta", "index": block_index, "delta": delta})
    except Exception as exc:  # noqa: BLE001 — last-resort mid-stream error frame
        yield wire.sse_event("error", {
            "type": "error",
            "error": {"type": "api_error", "message": f"generation failed: {exc}"},
        })
        return

    stop_reason, stop_sequence = gen.anthropic_stop()
    # Anthropic requires a non-empty content array: emit an empty text block if
    # nothing streamed and there are no tool calls.
    if cur_kind is None and not gen.tool_calls:
        block_index += 1
        yield wire.sse_event("content_block_start", {
            "type": "content_block_start", "index": block_index,
            "content_block": {"type": "text", "text": ""}})
        cur_kind = "content"
    if cur_kind is not None:
        yield wire.sse_event("content_block_stop", {"type": "content_block_stop", "index": block_index})
    for tc in gen.tool_calls:
        block_index += 1
        idx = block_index
        yield wire.sse_event("content_block_start", {
            "type": "content_block_start", "index": idx,
            "content_block": {"type": "tool_use", "id": tc.id, "name": tc.name, "input": {}},
        })
        yield wire.sse_event("content_block_delta", {
            "type": "content_block_delta", "index": idx,
            "delta": {"type": "input_json_delta", "partial_json": json.dumps(tc.arguments)},
        })
        yield wire.sse_event("content_block_stop", {"type": "content_block_stop", "index": idx})
    yield wire.sse_event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": stop_sequence},
        "usage": {"output_tokens": output_tokens},
    })
    yield wire.sse_event("message_stop", {"type": "message_stop"})
