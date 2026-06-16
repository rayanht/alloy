"""CLI-level tests for `alloy serve` — the foreground server command.

`alloy serve [-m MODEL]` runs the HTTP server in the foreground
(llama.cpp/vLLM-style). The typer flag → argv → argparse → ServerConfig
chain is the only way the CLI's flags reach the server process, so each
flag round-trip is pinned here with `run_server` mocked out.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

import alloy_server
from alloy_cli.cli import app


@pytest.fixture
def captured_config(monkeypatch: pytest.MonkeyPatch) -> dict[str, alloy_server.ServerConfig]:
    captured: dict[str, alloy_server.ServerConfig] = {}

    def fake_run_server(config: alloy_server.ServerConfig) -> None:
        captured["config"] = config

    monkeypatch.setattr(alloy_server, "run_server", fake_run_server)
    return captured


def test_serve_without_model_is_a_usage_error(
    captured_config: dict[str, alloy_server.ServerConfig],
) -> None:
    """One serve process = one model; `-m` is required."""
    runner = CliRunner()
    result = runner.invoke(app, ["serve"])
    assert result.exit_code != 0
    assert "config" not in captured_config


def test_serve_model_flag_preloads(
    captured_config: dict[str, alloy_server.ServerConfig],
) -> None:
    """`-m`/`--model` loads + pre-compiles the model at startup."""
    runner = CliRunner()
    result = runner.invoke(app, ["serve", "-m", "qwen3:0.6b"])
    assert result.exit_code == 0, result.output
    assert captured_config["config"].model == "qwen3:0.6b"


def test_serve_host_port_flags(
    captured_config: dict[str, alloy_server.ServerConfig],
) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["serve", "-m", "qwen3:0.6b", "--host", "0.0.0.0", "--port", "11435"])
    assert result.exit_code == 0, result.output
    assert captured_config["config"].host == "0.0.0.0"
    assert captured_config["config"].port == 11435


def test_serve_embedding_model_via_m(
    captured_config: dict[str, alloy_server.ServerConfig],
) -> None:
    """An embedding model is served the same way as a chat model: `-m`.
    The kind is resolved from the name at startup."""
    runner = CliRunner()
    result = runner.invoke(app, ["serve", "-m", "nomic-embed-text"])
    assert result.exit_code == 0, result.output
    assert captured_config["config"].model == "nomic-embed-text"
