"""Integration tests for the embedding endpoints (spec §5.6).

A stub `EmbeddingModel` returns deterministic vectors keyed off character
sums, so tests are fast and reproducible without loading a real encoder. The
real encoder path (`load_ollama_gguf_embedder`) is exercised indirectly: same
`EmbeddingModel` shape, same `embed(texts) -> list[list[float]]` contract.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from http.client import HTTPConnection
from threading import Thread

import pytest

from alloy_server.embedding import EmbeddingModel
from alloy_server import (
    AlloyServer,
    create_server,
)


_DIM = 4


def _stub_vector(text: str, dim: int = _DIM) -> list[float]:
    """Deterministic vector: char ord sums into dim slots, L2-normalized."""
    bucket = [0.0] * dim
    for i, ch in enumerate(text):
        bucket[i % dim] += float(ord(ch))
    norm = math.sqrt(sum(x * x for x in bucket)) or 1.0
    return [x / norm for x in bucket]


def _stub_embed_model(name: str = "alloy-test:embed", max_batch: int = 4) -> EmbeddingModel:
    def embed(texts: list[str]) -> list[list[float]]:
        if len(texts) > max_batch:
            raise ValueError(f"batch {len(texts)} > {max_batch}")
        return [_stub_vector(t) for t in texts]

    return EmbeddingModel(
        name=name,
        embed=embed,
        dimensions=_DIM,
        count_tokens=len,
        max_batch=max_batch,
    )


@pytest.fixture()
def server() -> Iterator[AlloyServer]:
    instance = create_server(
        "127.0.0.1", 0,
        embedding_model=_stub_embed_model(),
    )
    thread = Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    try:
        yield instance
    finally:
        instance.shutdown()
        thread.join(timeout=5)
        instance.server_close()


@pytest.fixture()
def port(server: AlloyServer) -> int:
    return server.server_address[1]


def _post(port: int, path: str, payload: dict) -> tuple[int, dict]:
    body = json.dumps(payload).encode()
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("POST", path, body=body, headers={"content-type": "application/json"})
    response = conn.getresponse()
    raw = response.read()
    conn.close()
    return response.status, json.loads(raw) if raw else {}


# ---- /v1/embeddings (OpenAI shape) -----------------------------------------


def test_v1_embeddings_single_string_input(port: int) -> None:
    status, payload = _post(
        port, "/v1/embeddings",
        {"model": "alloy-test:embed", "input": "hello"},
    )
    assert status == 200
    assert payload["object"] == "list"
    assert payload["model"] == "alloy-test:embed"
    assert len(payload["data"]) == 1
    item = payload["data"][0]
    assert item["object"] == "embedding"
    assert item["index"] == 0
    assert len(item["embedding"]) == _DIM
    assert payload["usage"]["prompt_tokens"] == 5
    assert payload["usage"]["total_tokens"] == 5


def test_v1_embeddings_batched_input(port: int) -> None:
    status, payload = _post(
        port, "/v1/embeddings",
        {"model": "alloy-test:embed", "input": ["alpha", "beta", "gamma"]},
    )
    assert status == 200
    assert len(payload["data"]) == 3
    assert [d["index"] for d in payload["data"]] == [0, 1, 2]
    # Deterministic stub: different inputs → different vectors.
    vectors = [d["embedding"] for d in payload["data"]]
    assert vectors[0] != vectors[1] != vectors[2]
    # All vectors are unit-normalized (within fp32 tolerance).
    for v in vectors:
        norm_sq = sum(x * x for x in v)
        assert abs(norm_sq - 1.0) < 1e-5


def test_v1_embeddings_unknown_model_returns_404(port: int) -> None:
    status, payload = _post(port, "/v1/embeddings", {"model": "nope", "input": "x"})
    assert status == 404
    assert payload["error"]["type"] == "model_not_served"


def test_v1_embeddings_batch_too_large_returns_400(port: int) -> None:
    """Stub's max_batch is 4; a 5-input request should 400 before reaching the model."""
    status, payload = _post(
        port, "/v1/embeddings",
        {"model": "alloy-test:embed", "input": ["a", "b", "c", "d", "e"]},
    )
    assert status == 400
    assert "batch size" in payload["error"]["message"]


def test_v1_embeddings_empty_input_returns_400(port: int) -> None:
    status, payload = _post(
        port, "/v1/embeddings", {"model": "alloy-test:embed", "input": ""},
    )
    assert status == 400
    assert payload["error"]["type"] == "invalid_request"


def test_v1_embeddings_non_string_list_item_returns_400(port: int) -> None:
    status, payload = _post(
        port, "/v1/embeddings",
        {"model": "alloy-test:embed", "input": ["ok", 42]},
    )
    assert status == 400
    assert "strings" in payload["error"]["message"]


# ---- /api/embed (Ollama new) ----------------------------------------------


def test_api_embed_batched_returns_ollama_shape(port: int) -> None:
    status, payload = _post(
        port, "/api/embed",
        {"model": "alloy-test:embed", "input": ["one", "two"]},
    )
    assert status == 200
    assert payload["model"] == "alloy-test:embed"
    assert len(payload["embeddings"]) == 2
    assert len(payload["embeddings"][0]) == _DIM
    assert payload["prompt_eval_count"] == len("one") + len("two")
    assert payload["total_duration"] >= 0
    assert payload["load_duration"] == 0


def test_api_embed_single_string_input(port: int) -> None:
    status, payload = _post(
        port, "/api/embed",
        {"model": "alloy-test:embed", "input": "hi"},
    )
    assert status == 200
    assert len(payload["embeddings"]) == 1


def test_api_embed_unknown_model_uses_ollama_error_envelope(port: int) -> None:
    status, payload = _post(port, "/api/embed", {"model": "nope", "input": "x"})
    assert status == 404
    assert isinstance(payload["error"], str), "Ollama clients expect flat string errors"
    assert "nope" in payload["error"]


# ---- /api/embeddings (Ollama legacy) ---------------------------------------


def test_api_embeddings_legacy_single_prompt(port: int) -> None:
    """Legacy /api/embeddings uses `prompt` (not `input`) and returns a
    flat `embedding` (singular) field rather than `embeddings` (plural).
    """
    status, payload = _post(
        port, "/api/embeddings",
        {"model": "alloy-test:embed", "prompt": "hello"},
    )
    assert status == 200
    assert "embedding" in payload  # singular
    assert "embeddings" not in payload
    assert len(payload["embedding"]) == _DIM


def test_api_embeddings_legacy_missing_prompt_returns_400(port: int) -> None:
    status, payload = _post(
        port, "/api/embeddings", {"model": "alloy-test:embed"},
    )
    assert status == 400


def test_api_embeddings_legacy_unknown_model_returns_404(port: int) -> None:
    status, payload = _post(
        port, "/api/embeddings", {"model": "nope", "prompt": "x"},
    )
    assert status == 404
    assert isinstance(payload["error"], str)


# ---- one model per server: kind mismatches 404 -----------------------------


def test_embedding_server_rejects_chat_requests(port: int) -> None:
    """An embedding-serving process has no chat model; chat endpoints
    404 with a pointer at starting a chat server."""
    status, payload = _post(
        port, "/v1/chat/completions",
        {"model": "alloy-test:embed", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert status == 404
