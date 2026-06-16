"""Dialect-agnostic request-parsing primitives shared across OpenAI / Ollama /
Anthropic: sampling, stop strings, tool/message/image parsing, the tool-call
grammar constraint. Each dialect module composes these into its own request
builder. No HTTP, no rendering."""

from __future__ import annotations

import base64
import binascii
import json
import urllib.request
import uuid
from http import HTTPStatus
from typing import TYPE_CHECKING, cast

from alloy_server.constrain import Constraint
from alloy_server.generation.sequence import SamplingParams
from alloy_server.schema import (
    ChatMessage,
    JsonObject,
    JsonValue,
    RequestError,
    ServedModel,
    ToolCall,
)

if TYPE_CHECKING:
    from alloy_server.embedding import EmbeddingModel


def opt_float(d: JsonObject, key: str, default: float) -> float:
    v = d.get(key)
    # bool is an int subclass — exclude it so `"stream": true` can't read as 1.0.
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    return default


def opt_int(d: JsonObject, key: str, default: int) -> int:
    v = d.get(key)
    if isinstance(v, int) and not isinstance(v, bool):
        return int(v)
    return default


def sampling_from(d: JsonObject, *, top_k: bool = True, min_p: bool = True) -> SamplingParams:
    """Parse common sampling fields from a flat dict — OpenAI/Anthropic top-level
    or an Ollama `options` block. Absent temperature => 0 => greedy."""
    return SamplingParams(
        temperature=opt_float(d, "temperature", 0.0),
        top_p=opt_float(d, "top_p", 1.0),
        top_k=opt_int(d, "top_k", 0) if top_k else 0,
        min_p=opt_float(d, "min_p", 0.0) if min_p else 0.0,
        seed=opt_int(d, "seed", 0),
    )


def parse_stop(value: JsonValue) -> tuple[str, ...]:
    """Normalize a stop field (string, list of strings, or absent) to a tuple
    of non-empty strings. OpenAI accepts a bare string or a list; Ollama and
    Anthropic use lists."""
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, list):
        return tuple(s for s in value if isinstance(s, str) and s)
    return ()


def apply_request_sampling(model: ServedModel, params: SamplingParams) -> None:
    """Push parsed sampling onto the model before generation. No-op for stub
    models (apply_sampling is None). Single-user local: each chat request resets
    sampling, so an omitted temperature reverts to greedy."""
    if model.apply_sampling is not None:
        model.apply_sampling(params)


def string_field(payload: JsonObject, key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise RequestError(HTTPStatus.BAD_REQUEST, "invalid_request", f"{key} must be a non-empty string")
    return value


def parse_max_tokens(value: JsonValue) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RequestError(HTTPStatus.BAD_REQUEST, "invalid_request", "max_tokens must be an integer")
    if value < 1:
        raise RequestError(HTTPStatus.BAD_REQUEST, "invalid_request", "max_tokens must be positive")
    return value


def parse_tools(value: JsonValue) -> tuple[dict, ...]:
    """Tool/function definitions (OpenAI/Ollama shape), passed through to the chat
    template which renders the signatures into the prompt."""
    if not isinstance(value, list):
        return ()
    return tuple(cast(dict, t) for t in value if isinstance(t, dict))


def parse_request_tool_calls(value: JsonValue) -> tuple[ToolCall, ...]:
    """Parse tool_calls echoed back on an assistant message (OpenAI/Ollama shape:
    [{id?, type, function:{name, arguments}}]). OpenAI sends arguments as a JSON
    string; Ollama sends an object. Both normalize to a dict."""
    if not isinstance(value, list):
        return ()
    calls: list[ToolCall] = []
    for tc in value:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function")
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
            continue
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, ValueError):
                args = {}
        tid = tc.get("id") if isinstance(tc.get("id"), str) else f"call_{uuid.uuid4().hex[:24]}"
        calls.append(ToolCall(
            id=tid, name=fn["name"], arguments=args if isinstance(args, dict) else {},
        ))
    return tuple(calls)


def decode_image_source(source: str) -> bytes:
    """Decode one image reference to raw bytes. Accepts a `data:` URL, an
    `http(s)` URL (fetched), or a bare base64 string (Ollama `images`)."""
    src = source.strip()
    if src.startswith("data:"):
        # data:[<mediatype>][;base64],<data>
        _, _, b64 = src.partition(",")
        src = b64
    elif src.startswith(("http://", "https://")):
        try:
            with urllib.request.urlopen(src, timeout=30) as resp:  # noqa: S310
                return cast(bytes, resp.read())
        except Exception as exc:
            raise RequestError(
                HTTPStatus.BAD_REQUEST, "invalid_request",
                f"failed to fetch image url: {exc}",
            ) from exc
    try:
        return base64.b64decode(src, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RequestError(
            HTTPStatus.BAD_REQUEST, "invalid_request", "invalid base64 image data",
        ) from exc


def decode_anthropic_image_source(source: JsonObject) -> bytes:
    """Decode an Anthropic image block's `source` ({type: base64, data} or
    {type: url, url}) to raw bytes."""
    stype = source.get("type")
    if stype == "url" and isinstance(source.get("url"), str):
        return decode_image_source(cast(str, source["url"]))
    data = source.get("data")
    if isinstance(data, str):
        return decode_image_source(data)
    raise RequestError(
        HTTPStatus.BAD_REQUEST, "invalid_request", "unsupported image source",
    )


def parse_content_with_images(
    content: JsonValue,
) -> tuple[str, tuple[bytes, ...], tuple[bytes, ...]]:
    """Parse a message `content` field into (text, image_bytes, audio_bytes).
    Accepts a plain string or an OpenAI-style parts list. Image parts (`image_url`)
    carry a `data:`/`http(s)` URL or bare base64; audio parts (`input_audio`) carry
    base64 (or a `data:` URL) in `input_audio.data`."""
    if content is None:
        return "", (), ()
    if isinstance(content, str):
        return content, (), ()
    if not isinstance(content, list):
        raise RequestError(
            HTTPStatus.BAD_REQUEST, "invalid_request",
            "message content must be a string or a list of content parts",
        )
    texts: list[str] = []
    images: list[bytes] = []
    audio: list[bytes] = []
    for part in content:
        if not isinstance(part, dict):
            raise RequestError(HTTPStatus.BAD_REQUEST, "invalid_request", "each content part must be an object")
        ptype = part.get("type")
        if ptype == "text":
            text = part.get("text")
            if isinstance(text, str):
                texts.append(text)
        elif ptype in ("image_url", "image"):
            url = part.get("image_url", part.get("url"))
            if isinstance(url, dict):
                url = url.get("url")
            if not isinstance(url, str):
                raise RequestError(HTTPStatus.BAD_REQUEST, "invalid_request", "image part missing url")
            images.append(decode_image_source(url))
        elif ptype in ("input_audio", "audio"):
            spec = part.get("input_audio", part.get("audio"))
            data = spec.get("data") if isinstance(spec, dict) else spec
            if not isinstance(data, str):
                raise RequestError(HTTPStatus.BAD_REQUEST, "invalid_request", "audio part missing data")
            audio.append(decode_image_source(data))  # generic data:/base64 → bytes
        # Unknown part types are ignored (forward-compat with video parts).
    return "\n".join(texts), tuple(images), tuple(audio)


def messages_field(payload: JsonObject, key: str) -> tuple[ChatMessage, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or not value:
        raise RequestError(HTTPStatus.BAD_REQUEST, "invalid_request", f"{key} must be a non-empty list")

    messages: list[ChatMessage] = []
    for item in value:
        if not isinstance(item, dict):
            raise RequestError(HTTPStatus.BAD_REQUEST, "invalid_request", "each message must be an object")
        message = cast(JsonObject, item)
        role = message.get("role")
        if not isinstance(role, str) or not role:
            raise RequestError(HTTPStatus.BAD_REQUEST, "invalid_request", "message role must be a string")
        if role == "developer":
            # OpenAI's o1-era rename of the system role. Models without a native
            # developer role (every family but gemma4, which treats them
            # identically) would silently drop the message — normalize to system.
            role = "system"
        tool_calls = parse_request_tool_calls(message.get("tool_calls"))
        # content is a string, an OpenAI parts list (text + image_url +
        # input_audio), or null for assistant tool-call messages.
        content, part_images, part_audio = parse_content_with_images(message.get("content"))
        # Ollama carries images out-of-band on the message (`images`: [base64,...]).
        ollama_images = message.get("images")
        if isinstance(ollama_images, list):
            part_images = part_images + tuple(
                decode_image_source(img) for img in ollama_images if isinstance(img, str)
            )
        tcid = message.get("tool_call_id")
        messages.append(ChatMessage(
            role=role, content=content, tool_calls=tool_calls,
            tool_call_id=tcid if isinstance(tcid, str) else None,
            images=part_images, audio=part_audio,
        ))
    return tuple(messages)


def parse_tool_choice(value: JsonValue) -> tuple[str, str | None]:
    """OpenAI tool_choice -> (mode, forced_tool). mode in auto/none/required."""
    if isinstance(value, str) and value in ("auto", "none", "required"):
        return value, None
    if isinstance(value, dict):
        fn = value.get("function")
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            return "required", fn["name"]
    return "auto", None


def tool_constraint(
    tools: tuple[dict, ...], mode: str, forced_tool: str | None, single_call: bool = False,
    *, named_exactly_one: bool = True,
) -> Constraint | None:
    """A tool-call grammar constraint, or None to leave the decode unconstrained.

    `single_call` is parallel_tool_calls=false / disable_parallel_tool_use=true.
    - "required" / "any": must call >=1; one-or-more by default (parallel allowed),
      single_call -> exactly one.
    - "auto": unconstrained by default (the fast path; tool calls are parsed from
      free text — zero/one/several). Only when single_call is set do we constrain it
      — to enforce "at most one" structurally — accepting the per-step decode path.
    - named: forces that one tool. The ONE dialect difference is the default count:
      OpenAI's {type:function} means exactly one call (`named_exactly_one=True`);
      Anthropic's {type:tool} forces the tool but may call it more than once by
      default (`named_exactly_one=False`), capped at one by disable_parallel_tool_use.

    The native layer compiles the choice to the model's tool-call grammar; if the
    model has no known tool format it degrades to unconstrained."""
    if not tools or mode == "none":
        return None
    if forced_tool is not None:
        if named_exactly_one:
            # OpenAI: a specific function => exactly that one call. xgrammar's
            # {type:function} is a lone tag (no repetition); single_call is moot.
            choice: JsonValue = {"type": "function", "function": {"name": forced_tool}}
        else:
            # Anthropic: forced but parallel-capable — >=1 of the named tool by
            # default. allowed_tools{required, [tool]} is a repeatable tag, so
            # single_call's stop_after_first caps it at exactly one.
            choice = {
                "type": "allowed_tools",
                "allowed_tools": {
                    "mode": "required",
                    "tools": [{"type": "function", "function": {"name": forced_tool}}],
                },
            }
    elif mode == "required":
        choice = "required"
    elif mode == "auto" and single_call:
        choice = "auto"
    else:  # auto without single_call -> fast path, no constraint
        return None
    return Constraint(
        kind="tool", tools_json=json.dumps(list(tools)),
        tool_choice_json=json.dumps(choice), single_call=single_call,
    )


def normalize_embed_input(value: JsonValue) -> list[str]:
    """OpenAI and Ollama-new both accept `input` as a string or list of
    strings. We normalize to a non-empty list of strings; non-conforming
    inputs raise a 400 before we touch the model.
    """
    if isinstance(value, str):
        if not value:
            raise RequestError(
                HTTPStatus.BAD_REQUEST, "invalid_request", "input must not be empty",
            )
        return [value]
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise RequestError(
                    HTTPStatus.BAD_REQUEST, "invalid_request",
                    "input list elements must be strings",
                )
            out.append(item)
        if not out:
            raise RequestError(
                HTTPStatus.BAD_REQUEST, "invalid_request", "input must be non-empty",
            )
        return out
    raise RequestError(
        HTTPStatus.BAD_REQUEST, "invalid_request",
        "input must be a string or list of strings",
    )


def resolve_embedding_request(model: "EmbeddingModel", payload: JsonObject) -> list[str]:
    inputs = normalize_embed_input(payload.get("input"))
    if len(inputs) > model.max_batch:
        raise RequestError(
            HTTPStatus.BAD_REQUEST, "invalid_request",
            f"batch size {len(inputs)} exceeds model max {model.max_batch}",
        )
    return inputs
