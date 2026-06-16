"""Dialect-agnostic data model + type aliases for the server.

The request/message structs every dialect parses into and every renderer reads
from, plus the callable contracts a served model exposes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from http import HTTPStatus
from typing import TypeAlias

import torch

from alloy_server.constrain import Constraint
from alloy_server.embedding import TextTokenCounter
from alloy_server.generation.generator import AlloyGenerator
from alloy_server.generation.sequence import SamplingParams
from alloy_server.gguf import LoadedGGUFCausalLM, ResolvedGGUF
from alloy_server.mlx import ResolvedMLX
from alloy_server.reasoning import ReasoningProtocol

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class Rendered:
    """A non-JSON response body carrying its own content-type (e.g. transcription
    text/srt/vtt). A handler returns this instead of a JsonObject when the response
    isn't application/json; the transport emits it verbatim."""
    content_type: str
    body: str


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A function call the model emitted. `arguments` is the parsed object."""
    id: str
    name: str
    arguments: dict


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: str
    content: str
    # Populated for assistant messages that called tools (round-tripped into the
    # next prompt) and for parsed model output.
    tool_calls: tuple[ToolCall, ...] = ()
    # Set on role="tool" result messages to link back to the call (OpenAI).
    tool_call_id: str | None = None
    # Raw bytes of any images attached to this (user) message, in prompt order.
    # Decoded from OpenAI `image_url` parts / Ollama `images` / Anthropic image
    # blocks. Only used by vision-capable served models; ignored otherwise.
    images: tuple[bytes, ...] = ()
    # Raw bytes of any audio clips attached to this (user) message, in prompt order.
    # Decoded from OpenAI `input_audio` parts. Only used by audio-capable served
    # models (gemma4); ignored otherwise.
    audio: tuple[bytes, ...] = ()


# Tools/finish are carried on the request; the model contract takes a `tools`
# list so the chat template can inject the function signatures into the prompt.
ChatResponder: TypeAlias = Callable[..., str]
ChatStreamer: TypeAlias = Callable[..., Iterator[str]]
TokenDecoder: TypeAlias = Callable[[torch.Tensor], str]
# generate/stream take an optional constraint (grammar) for constrained decoding.
TokenGenerator: TypeAlias = Callable[..., torch.Tensor]
TokenStreamer: TypeAlias = Callable[..., Iterator[int]]
ChatTokenizer: TypeAlias = Callable[..., torch.Tensor]
NativeModelLoader: TypeAlias = Callable[[ResolvedGGUF | ResolvedMLX], LoadedGGUFCausalLM]
NativeGeneratorBuilder: TypeAlias = Callable[..., AlloyGenerator]
# Multimodal (vision) hooks: encode messages+images -> (input_ids, image_features,
# image_positions); generate/stream consume those alongside max_tokens.
MultimodalEncoder: TypeAlias = Callable[
    ["tuple[ChatMessage, ...]", "bool | None"],
    "tuple[torch.Tensor, torch.Tensor, torch.Tensor]",
]
MultimodalGenerator: TypeAlias = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, int], torch.Tensor,
]
MultimodalStreamer: TypeAlias = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, int], Iterator[int],
]


ApplySampling: TypeAlias = Callable[[SamplingParams], None]


@dataclass(frozen=True, slots=True)
class ServedModel:
    name: str
    complete: ChatResponder
    stream: ChatStreamer
    count_tokens: TextTokenCounter
    # Set decode sampling for the next generation. None for stub/echo models
    # that don't sample. The native model threads this into the next Sequence.
    apply_sampling: ApplySampling | None = None
    # The reasoning protocol if this is a thinking model — how its output delimits
    # chain-of-thought (`</think>` family or gemma4's `<|channel>` block) — used to
    # split reasoning into a separate field. None for non-reasoning models.
    reasoning: ReasoningProtocol | None = None


@dataclass(frozen=True, slots=True)
class RequestError(Exception):
    status: HTTPStatus
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class ChatCompletionRequest:
    model: ServedModel
    messages: tuple[ChatMessage, ...]
    max_tokens: int
    # Stop strings: generation halts at the first occurrence and the stop string
    # (and anything after) is excluded from the output. Empty = no stop strings.
    stop: tuple[str, ...] = ()
    # Tool/function definitions (OpenAI shape: [{type, function:{name, description,
    # parameters}}]) injected into the prompt via the chat template. Empty = none.
    tools: tuple[dict, ...] = ()
    # "auto" (model decides) or "required" (must call a tool). "none" drops tools.
    tool_choice: str = "auto"
    # When set, force a call to exactly this function (named tool_choice).
    forced_tool: str | None = None
    # Grammar constraint for the decode (JSON / structured output / forced tool
    # call). None = unconstrained. Built from response_format / forcing tool_choice.
    constraint: Constraint | None = None
    # Per-request reasoning toggle for thinking models (Ollama `think`, OpenAI
    # reasoning_effort / chat_template_kwargs, Anthropic `thinking`). None = model
    # default (thinking ON, matching ollama); True/False is an explicit override.
    enable_thinking: bool | None = None
