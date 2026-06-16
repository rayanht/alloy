"""Integration tests for Ollama-compatible endpoints on the Alloy daemon.

Each test spins up the live HTTP server (via the `server` / `port` fixtures
in `conftest.py`) and hits it with `http.client`. We snapshot the JSON wire
shape — actual model behavior is stubbed in the fixture.
"""

from __future__ import annotations

import json
import os
from http.client import HTTPConnection

import pytest


PROBE_HEADERS_JSON = {"content-type": "application/json"}


def _get(port: int, path: str, headers: dict[str, str] | None = None) -> tuple[int, dict[str, str], bytes]:
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("GET", path, headers=headers or {})
    response = conn.getresponse()
    body = response.read()
    out_headers = {k.lower(): v for k, v in response.getheaders()}
    conn.close()
    return response.status, out_headers, body


def _post(
    port: int,
    path: str,
    payload: dict | None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    body = json.dumps(payload).encode() if payload is not None else b""
    headers = {**PROBE_HEADERS_JSON, **(headers or {})}
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("POST", path, body=body, headers=headers)
    response = conn.getresponse()
    raw = response.read()
    out_headers = {k.lower(): v for k, v in response.getheaders()}
    conn.close()
    return response.status, out_headers, raw


def _delete(port: int, path: str, payload: dict | None) -> tuple[int, dict[str, str], bytes]:
    body = json.dumps(payload).encode() if payload is not None else b""
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("DELETE", path, body=body, headers=PROBE_HEADERS_JSON)
    response = conn.getresponse()
    raw = response.read()
    out_headers = {k.lower(): v for k, v in response.getheaders()}
    conn.close()
    return response.status, out_headers, raw


def _options(port: int, path: str, headers: dict[str, str]) -> tuple[int, dict[str, str]]:
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("OPTIONS", path, headers=headers)
    response = conn.getresponse()
    response.read()
    out_headers = {k.lower(): v for k, v in response.getheaders()}
    conn.close()
    return response.status, out_headers


def _ndjson_lines(raw: bytes) -> list[dict]:
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


# ---- read-only endpoints ---------------------------------------------------


def test_api_version_returns_string(port: int) -> None:
    status, _, body = _get(port, "/api/version")
    assert status == 200
    payload = json.loads(body)
    assert isinstance(payload["version"], str)
    assert payload["version"]  # non-empty


def test_api_tags_returns_models_list(port: int) -> None:
    status, _, body = _get(port, "/api/tags")
    assert status == 200
    payload = json.loads(body)
    assert "models" in payload
    names = [m["name"] for m in payload["models"]]
    assert names == ["alloy-test:tiny"]
    # Shape: each entry has name/model/modified_at/size/digest/details.
    entry = payload["models"][0]
    assert set(entry).issuperset({"name", "model", "modified_at", "size", "digest", "details"})


def test_api_ps_is_removed(port: int) -> None:
    """One serve process = one model, fixed at startup. The dynamic
    loaded-models surface (/api/ps, /api/load, /api/unload) is gone;
    what's served lives in /healthz."""
    status, _, _ = _get(port, "/api/ps")
    assert status == 404


def test_api_show_returns_metadata(port: int) -> None:
    status, _, body = _post(port, "/api/show", {"name": "alloy-test:tiny"})
    assert status == 200
    payload = json.loads(body)
    assert set(payload).issuperset({"modelfile", "parameters", "template", "details", "capabilities"})
    assert "completion" in payload["capabilities"]


def test_api_show_unknown_model_returns_404(port: int) -> None:
    status, _, body = _post(port, "/api/show", {"name": "nope"})
    assert status == 404
    payload = json.loads(body)
    assert isinstance(payload["error"], str), "Ollama clients expect flat string errors"
    assert "nope" in payload["error"]


def test_healthz_reports_the_served_model(port: int) -> None:
    status, _, body = _get(port, "/healthz")
    assert status == 200
    payload = json.loads(body)
    assert payload["status"] == "ok"
    assert payload["model"] == "alloy-test:tiny"
    assert payload["kind"] == "chat"


# ---- chat / generate -------------------------------------------------------


def test_api_chat_non_streaming(port: int) -> None:
    status, _, body = _post(
        port,
        "/api/chat",
        {
            "model": "alloy-test:tiny",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
            "options": {"num_predict": 5},
        },
    )
    assert status == 200
    payload = json.loads(body)
    assert payload["model"] == "alloy-test:tiny"
    assert payload["message"]["role"] == "assistant"
    assert payload["message"]["content"] == "hello"
    assert payload["done"] is True
    assert payload["done_reason"] == "stop"
    assert payload["prompt_eval_count"] == 5  # len("hello")
    assert payload["eval_count"] == 5


def test_api_chat_streaming_ndjson(port: int) -> None:
    status, headers, body = _post(
        port,
        "/api/chat",
        {
            "model": "alloy-test:tiny",
            "messages": [{"role": "user", "content": "streaming-test"}],
            "stream": True,
            "options": {"num_predict": 20},
        },
    )
    assert status == 200
    assert headers["content-type"] == "application/x-ndjson"
    events = _ndjson_lines(body)
    # All events except the last have done=False; last is done=True.
    assert events[-1]["done"] is True
    assert all(not e["done"] for e in events[:-1])
    # Reconstructed text matches the stub's echo.
    reconstructed = "".join(e["message"]["content"] for e in events[:-1])
    assert reconstructed == "streaming-test"


def test_api_chat_defaults_to_streaming_when_field_absent(port: int) -> None:
    """Ollama treats stream as true-by-default (unlike OpenAI's false default)."""
    status, headers, _ = _post(
        port,
        "/api/chat",
        {
            "model": "alloy-test:tiny",
            "messages": [{"role": "user", "content": "x"}],
        },
    )
    assert status == 200
    assert headers["content-type"] == "application/x-ndjson"


def test_api_chat_unknown_model_returns_404(port: int) -> None:
    status, _, body = _post(
        port,
        "/api/chat",
        {"model": "nope", "messages": [{"role": "user", "content": "x"}], "stream": False},
    )
    assert status == 404
    payload = json.loads(body)
    assert isinstance(payload["error"], str)
    assert "nope" in payload["error"]


def test_api_chat_stop_truncates_non_streaming(port: int) -> None:
    # The echo stub returns the user content; a stop string halts the output
    # before its first occurrence and excludes it.
    status, _, body = _post(
        port,
        "/api/chat",
        {
            "model": "alloy-test:tiny",
            "messages": [{"role": "user", "content": "alpha beta gamma"}],
            "stream": False,
            "options": {"num_predict": 50, "stop": ["beta"]},
        },
    )
    assert status == 200
    assert json.loads(body)["message"]["content"] == "alpha "


def test_api_chat_stop_truncates_streaming(port: int) -> None:
    # Stop also applies to the streamed deltas (including a stop split across
    # the stub's 4-char chunks); the reassembled text matches the non-stream cut.
    status, _, body = _post(
        port,
        "/api/chat",
        {
            "model": "alloy-test:tiny",
            "messages": [{"role": "user", "content": "alpha beta gamma"}],
            "stream": True,
            "options": {"num_predict": 50, "stop": ["beta"]},
        },
    )
    assert status == 200
    events = _ndjson_lines(body)
    reconstructed = "".join(e["message"]["content"] for e in events[:-1])
    assert reconstructed == "alpha "


def test_api_chat_stop_earliest_of_multiple_wins(port: int) -> None:
    status, _, body = _post(
        port,
        "/api/chat",
        {
            "model": "alloy-test:tiny",
            "messages": [{"role": "user", "content": "one two three four"}],
            "stream": False,
            "options": {"num_predict": 50, "stop": ["four", "two"]},
        },
    )
    assert status == 200
    assert json.loads(body)["message"]["content"] == "one "


def test_api_chat_no_stop_returns_full_echo(port: int) -> None:
    status, _, body = _post(
        port,
        "/api/chat",
        {
            "model": "alloy-test:tiny",
            "messages": [{"role": "user", "content": "alpha beta gamma"}],
            "stream": False,
            "options": {"num_predict": 50},
        },
    )
    assert status == 200
    assert json.loads(body)["message"]["content"] == "alpha beta gamma"


def test_api_generate_non_streaming(port: int) -> None:
    status, _, body = _post(
        port,
        "/api/generate",
        {"model": "alloy-test:tiny", "prompt": "hi-there", "stream": False, "options": {"num_predict": 10}},
    )
    assert status == 200
    payload = json.loads(body)
    assert payload["response"] == "hi-there"
    assert payload["done"] is True


def test_api_generate_streaming_ndjson(port: int) -> None:
    status, headers, body = _post(
        port,
        "/api/generate",
        {"model": "alloy-test:tiny", "prompt": "abcd-efgh", "stream": True, "options": {"num_predict": 20}},
    )
    assert status == 200
    assert headers["content-type"] == "application/x-ndjson"
    events = _ndjson_lines(body)
    reconstructed = "".join(e["response"] for e in events[:-1])
    assert reconstructed == "abcd-efgh"
    # The final event must include a `context` field — Continue and other
    # /api/generate-based loops thread this back into the next request.
    assert events[-1]["context"] == []


def test_api_generate_non_streaming_includes_context(port: int) -> None:
    _, _, body = _post(
        port,
        "/api/generate",
        {"model": "alloy-test:tiny", "prompt": "x", "stream": False, "options": {"num_predict": 4}},
    )
    payload = json.loads(body)
    assert payload["context"] == []


def test_multi_turn_chat_is_stateless(port: int) -> None:
    """daemon does not maintain server-side conversation state.

    Two sequential requests share a common prefix but differ in their final
    user message. If the daemon were stateful (e.g. accumulating history
    across requests), the second response would echo the first request's
    final message instead of its own. The stub model echoes the last user
    message, so each response should equal its own request's last message.
    """
    common_prefix = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "first"},
    ]
    req_a = {
        "model": "alloy-test:tiny",
        "messages": [*common_prefix, {"role": "user", "content": "second"}],
        "stream": False,
        "options": {"num_predict": 32},
    }
    req_b = {
        "model": "alloy-test:tiny",
        "messages": [*common_prefix, {"role": "user", "content": "third"}],
        "stream": False,
        "options": {"num_predict": 32},
    }
    response_a = json.loads(_post(port, "/api/chat", req_a)[2])
    response_b = json.loads(_post(port, "/api/chat", req_b)[2])
    assert response_a["message"]["content"] == "second"
    assert response_b["message"]["content"] == "third"


# ---- copy / delete (removed: the served model is immutable) ----------------


def test_api_copy_is_removed(port: int) -> None:
    status, _, _ = _post(
        port,
        "/api/copy",
        {"source": "alloy-test:tiny", "destination": "alloy-test:alias"},
    )
    assert status == 404


def test_api_delete_is_removed(port: int) -> None:
    # With no DELETE endpoints left, do_DELETE itself is gone; the stdlib
    # handler answers 501 for the unsupported method.
    status, _, _ = _delete(port, "/api/delete", {"name": "alloy-test:tiny"})
    assert status in (404, 501)


# ---- 501-stubs --------------------------------------------------------------


@pytest.mark.parametrize("path", ["/api/create", "/api/push", "/api/blobs/sha256-deadbeef"])
def test_deferred_endpoints_return_501(port: int, path: str) -> None:
    status, _, body = _post(port, path, {})
    assert status == 501
    payload = json.loads(body)
    assert isinstance(payload["error"], str), "Ollama clients expect flat string errors"
    assert "alloy v1" in payload["error"]
    assert path in payload["error"]


# ---- embeddings (see tests/integration/embeddings/ for full coverage) ----


def test_api_embed_unregistered_model_returns_ollama_404(port: int) -> None:
    """A chat-only fixture is absent from the embedding registry."""
    status, _, body = _post(port, "/api/embed", {"model": "alloy-test:tiny", "input": "hi"})
    assert status == 404
    payload = json.loads(body)
    assert isinstance(payload["error"], str)
    assert "alloy-test:tiny" in payload["error"]


def test_api_embeddings_legacy_unregistered_model_returns_ollama_404(port: int) -> None:
    status, _, body = _post(port, "/api/embeddings", {"model": "alloy-test:tiny", "prompt": "hi"})
    assert status == 404
    payload = json.loads(body)
    assert isinstance(payload["error"], str)


# ---- CORS -------------------------------------------------------------------


def test_cors_preflight_allowed_localhost_origin(port: int) -> None:
    status, headers = _options(
        port,
        "/api/chat",
        {
            "origin": "http://localhost:3000",
            "access-control-request-method": "POST",
            "access-control-request-headers": "content-type",
        },
    )
    assert status == 204
    assert headers["access-control-allow-origin"] == "http://localhost:3000"
    assert "POST" in headers["access-control-allow-methods"]
    assert headers["access-control-max-age"] == "86400"


def test_cors_preflight_disallowed_origin_omits_allow_header(port: int) -> None:
    status, headers = _options(
        port,
        "/api/chat",
        {
            "origin": "https://evil.example",
            "access-control-request-method": "POST",
        },
    )
    # Preflight still returns 204 (browsers do the actual block based on the
    # absence of Access-Control-Allow-Origin matching the request origin).
    assert status == 204
    assert "access-control-allow-origin" not in headers


def test_cors_get_response_echoes_allowed_origin(port: int) -> None:
    status, headers, _ = _get(port, "/api/version", headers={"origin": "http://localhost:5173"})
    assert status == 200
    assert headers["access-control-allow-origin"] == "http://localhost:5173"
    assert headers["vary"] == "origin"


def test_alloy_origins_env_extends_default(port: int) -> None:
    os.environ["ALLOY_ORIGINS"] = "https://app.example.com"
    status, headers, _ = _get(
        port, "/api/version", headers={"origin": "https://app.example.com"}
    )
    assert status == 200
    assert headers["access-control-allow-origin"] == "https://app.example.com"


def test_alloy_origins_env_replace_strips_defaults(port: int) -> None:
    os.environ["ALLOY_ORIGINS"] = "=https://only.this"
    status, headers, _ = _get(port, "/api/version", headers={"origin": "http://localhost:3000"})
    # The `=` prefix replaces the default origin list entirely.
    assert status == 200
    assert "access-control-allow-origin" not in headers


def test_cors_omitted_when_no_origin_header(port: int) -> None:
    status, headers, _ = _get(port, "/api/version")
    assert status == 200
    assert "access-control-allow-origin" not in headers
