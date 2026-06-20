from __future__ import annotations

import json
from collections.abc import Iterator
from http.client import HTTPConnection
from pathlib import Path
from threading import Thread
from typing import TypeAlias, cast

import pytest
import torch
import transformers

from alloy_server import (
    ChatMessage,
    NativeGeneratorBuilder,
    ServedModel,
    create_generation_served_model,
    create_native_served_model,
    create_server,
    parse_server_config,
)
from alloy_server.generation.generator import AlloyGenerator
from alloy_server.gguf import (
    GGUFLoadReport,
    LoadedGGUFCausalLM,
    ResolvedGGUF,
)
from alloy_server.models import check_arch_supported

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def _complete(messages: tuple[ChatMessage, ...], max_tokens: int, tools: tuple[dict, ...] = (), **kwargs) -> str:
    return f"{messages[-1].content} / {max_tokens}"


def _stream(messages: tuple[ChatMessage, ...], max_tokens: int, tools: tuple[dict, ...] = (), **kwargs) -> Iterator[str]:
    yield _complete(messages, max_tokens)


def _encode_prompt(
    messages: tuple[ChatMessage, ...], tools: tuple[dict, ...] = (),
    enable_thinking: bool | None = None,
) -> torch.Tensor:
    roles = [m.role for m in messages]
    contents = [m.content for m in messages]
    assert roles == ["system", "user"]
    assert contents == ["concise", "hello"]
    return torch.tensor([[11, 12, 13]], dtype=torch.long)


def _generate_tokens(
    input_ids: torch.Tensor, max_new_tokens: int, constraint: object = None,
) -> torch.Tensor:
    assert input_ids.shape == (1, 3)
    assert max_new_tokens == 2
    return torch.tensor([[11, 12, 13, 21, 22]], dtype=torch.long)


def _decode_tokens(token_ids: torch.Tensor) -> str:
    # Prefix-coherent fake (BPE-style): decode([t1, t2]) starts with
    # decode([t1]). The streaming path computes deltas by stripping the
    # previous step's decoded prefix, so a fake that swung between
    # unrelated outputs would corrupt the delta math.
    pieces = {21: "decoded", 22: " response"}
    if token_ids.ndim == 1:
        ids = token_ids.tolist()
    else:
        ids = token_ids.flatten().tolist()
    return "".join(pieces.get(int(i), "?") for i in ids)


def _stream_generated_token_ids(
    input_ids: torch.Tensor, max_new_tokens: int, constraint: object = None,
) -> Iterator[int]:
    assert input_ids.shape == (1, 3)
    assert max_new_tokens == 2
    yield 21
    yield 22


def _count_tokens(text: str) -> int:
    return len([part for part in text.split() if part])


def _request_json(
    port: int,
    method: str,
    path: str,
    body: dict[str, JsonValue] | None = None,
) -> tuple[int, dict[str, JsonValue]]:
    connection = HTTPConnection("127.0.0.1", port, timeout=5)
    payload = json.dumps(body).encode() if body is not None else None
    headers = {"content-type": "application/json"} if body is not None else {}
    connection.request(method, path, body=payload, headers=headers)
    response = connection.getresponse()
    response_body = response.read().decode()
    connection.close()
    return response.status, cast(dict[str, JsonValue], json.loads(response_body))


def _request_raw(
    port: int,
    method: str,
    path: str,
    body: dict[str, JsonValue],
) -> tuple[int, str, str]:
    connection = HTTPConnection("127.0.0.1", port, timeout=5)
    payload = json.dumps(body).encode()
    connection.request(method, path, body=payload, headers={"content-type": "application/json"})
    response = connection.getresponse()
    response_body = response.read().decode()
    content_type = response.getheader("content-type", "")
    connection.close()
    return response.status, content_type, response_body


def test_models_endpoint_returns_openai_list_shape() -> None:
    server = create_server("127.0.0.1", 0, chat_model=ServedModel("tiny", _complete, _stream, _count_tokens))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, payload = _request_json(server.server_port, "GET", "/v1/models")
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert status == 200
    assert payload["object"] == "list"
    data = payload["data"]
    assert isinstance(data, list)
    assert data[0] == {"id": "tiny", "object": "model", "owned_by": "alloy"}


def test_chat_completion_returns_assistant_message() -> None:
    server = create_server("127.0.0.1", 0, chat_model=ServedModel("tiny", _complete, _stream, _count_tokens))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, payload = _request_json(
            server.server_port,
            "POST",
            "/v1/chat/completions",
            {
                "model": "tiny",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 7,
                "stream": False,
            },
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert status == 200
    assert payload["object"] == "chat.completion"
    assert payload["model"] == "tiny"
    choices = payload["choices"]
    assert isinstance(choices, list)
    choice = choices[0]
    assert isinstance(choice, dict)
    message = choice["message"]
    assert isinstance(message, dict)
    assert message == {"role": "assistant", "content": "hello / 7"}


def test_chat_completion_unknown_model_returns_structured_404() -> None:
    server = create_server("127.0.0.1", 0, chat_model=ServedModel("tiny", _complete, _stream, _count_tokens))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, payload = _request_json(
            server.server_port,
            "POST",
            "/v1/chat/completions",
            {
                "model": "missing",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 7,
                "stream": False,
            },
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert status == 404
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["code"] == "model_not_served"


def test_chat_completion_stream_returns_sse_chunks() -> None:
    server = create_server("127.0.0.1", 0, chat_model=ServedModel("tiny", _complete, _stream, _count_tokens))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, content_type, body = _request_raw(
            server.server_port,
            "POST",
            "/v1/chat/completions",
            {
                "model": "tiny",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 7,
                "stream": True,
            },
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert status == 200
    assert content_type == "text/event-stream"
    data_lines = [line.removeprefix("data: ") for line in body.splitlines() if line.startswith("data: ")]
    assert data_lines[-1] == "[DONE]"
    chunks = [cast(dict[str, JsonValue], json.loads(line)) for line in data_lines[:-1]]
    assert [chunk["object"] for chunk in chunks] == ["chat.completion.chunk", "chat.completion.chunk"]
    assert chunks[0]["model"] == "tiny"
    first_choices = chunks[0]["choices"]
    assert isinstance(first_choices, list)
    first_choice = first_choices[0]
    assert isinstance(first_choice, dict)
    assert first_choice["delta"] == {"role": "assistant", "content": "hello / 7"}
    final_choices = chunks[1]["choices"]
    assert isinstance(final_choices, list)
    final_choice = final_choices[0]
    assert isinstance(final_choice, dict)
    assert final_choice["finish_reason"] == "stop"


def test_generation_served_model_decodes_only_new_tokens() -> None:
    model = create_generation_served_model(
        "tiny",
        _encode_prompt,
        _decode_tokens,
        _generate_tokens,
        _stream_generated_token_ids,
        _count_tokens,
    )

    content = model.complete(
        (
            ChatMessage(role="system", content="concise"),
            ChatMessage(role="user", content="hello"),
        ),
        2,
    )

    assert content == "decoded response"
    # `stream()` emits per-token deltas, so assert on the joined result
    # rather than a single-chunk shape — clients concatenate the same way.
    chunks = list(
        model.stream(
            (
                ChatMessage(role="system", content="concise"),
                ChatMessage(role="user", content="hello"),
            ),
            2,
        )
    )
    assert len(chunks) >= 1
    assert "".join(chunks) == "decoded response"


def test_parse_server_config_accepts_optional_hf_id() -> None:
    config = parse_server_config(
        (
            "--model",
            "qwen3:0.6b",
            "--hf-id",
            "Qwen/Qwen3-0.6B",
            "--host",
            "127.0.0.1",
            "--port",
            "11435",
        )
    )

    assert config.model == "qwen3:0.6b"
    assert config.hf_id == "Qwen/Qwen3-0.6B"
    assert config.host == "127.0.0.1"
    assert config.port == 11435
    # Context is not a config field: the cache is always native.
    assert config.allow_downloads is False


def test_parse_server_config_does_not_require_hf_id() -> None:
    config = parse_server_config(
        (
            "--model",
            "qwen3:0.6b",
            "--host",
            "127.0.0.1",
            "--port",
            "11435",
        )
    )

    assert config.model == "qwen3:0.6b"
    assert config.hf_id is None


def test_unsupported_gguf_arch_rejected_unless_forced() -> None:
    # The serve gate is metadata-driven: an arch outside the supported set is
    # refused; `--force` is the escape hatch.
    with pytest.raises(ValueError, match="not in the supported set"):
        check_arch_supported("some-unknown-arch")
    check_arch_supported("some-unknown-arch", force=True)  # no raise
    check_arch_supported("qwen35")  # a supported arch: no raise


def test_native_served_model_loads_gguf_model_before_decode() -> None:
    calls: list[str] = []
    build_calls: list[torch.dtype] = []

    class FakeModel(torch.nn.Module):
        pass

    class FakeTokenizer:
        eos_token_id = 99
        chat_template = "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}"

        def apply_chat_template(
            self,
            messages: list[dict[str, str]],
            *,
            tokenize: bool,
            add_generation_prompt: bool,
        ) -> str:
            assert tokenize is False
            if not add_generation_prompt:
                parts = "".join(f"<|{m['role']}|>{m['content']}" for m in messages)
                return parts + "<|assistant|>"
            assert [m["role"] for m in messages] == ["user"]
            assert [m["content"] for m in messages] == ["hello"]
            return "<|user|>hello<|assistant|>"

        def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
            return []

        def __call__(
            self,
            prompt: str,
            *,
            return_tensors: str,
            add_special_tokens: bool,
        ) -> dict[str, torch.Tensor]:
            assert prompt == "<|user|>hello<|assistant|>"
            assert return_tensors == "pt"
            assert add_special_tokens is False
            return {"input_ids": torch.tensor([[7, 8]], dtype=torch.long)}

        def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str:
            return f"decoded:{ids}"

    class FakeGenerator:
        def generate(self, input_ids: torch.Tensor, *, max_new_tokens: int) -> torch.Tensor:
            assert input_ids.tolist() == [[7, 8]]
            assert max_new_tokens == 2
            return torch.tensor([[7, 8, 21, 22]], dtype=torch.long)

        def stream_token_ids(self, input_ids: torch.Tensor, *, max_new_tokens: int) -> Iterator[int]:
            assert input_ids.tolist() == [[7, 8]]
            assert max_new_tokens == 2
            yield 21
            yield 22

        def run(self, seq: object) -> Iterator[int]:
            # Native streaming drives generator.run(Sequence) for the
            # non-AlloyGenerator fallback; `generate` still uses .generate.
            assert seq.input_ids.tolist() == [[7, 8]]  # type: ignore[attr-defined]
            assert seq.max_new_tokens == 2  # type: ignore[attr-defined]
            yield 21
            yield 22

    fake_model = cast(transformers.PreTrainedModel, FakeModel())

    def load_fake_model(source: ResolvedGGUF) -> LoadedGGUFCausalLM:
        calls.append(source.ref)
        return LoadedGGUFCausalLM(
            model=fake_model,
            tokenizer=cast(transformers.PreTrainedTokenizerBase, FakeTokenizer()),
            report=cast(GGUFLoadReport, None),
        )

    def build_fake_generator(
        model: transformers.PreTrainedModel,
        cache_dtype: torch.dtype,
        *_args: object,
        **_kwargs: object,
    ) -> AlloyGenerator:
        assert model is fake_model
        build_calls.append(cache_dtype)
        return cast(AlloyGenerator, FakeGenerator())

    config = parse_server_config(("--model", "qwen3:0.6b"))
    resolved = ResolvedGGUF(ref="qwen3:0.6b", path=Path("/fake/qwen3.gguf"), digest="x")

    served = create_native_served_model(
        config,
        resolved,
        load_model=load_fake_model,
        build_generator=cast(NativeGeneratorBuilder, build_fake_generator),
    )

    assert served.complete((ChatMessage("user", "hello"),), 2) == "decoded:[21, 22]"
    joined = "".join(served.stream((ChatMessage("user", "hello"),), 2))
    assert "21" in joined and "22" in joined and joined.index("21") < joined.index("22")
    assert calls == ["qwen3:0.6b"]
    assert build_calls == [torch.float16]
