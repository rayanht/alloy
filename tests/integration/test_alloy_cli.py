from __future__ import annotations

from pytest import CaptureFixture

from alloy.cli import main as alloy_main


def test_alloy_cli_exposes_serve_help(capsys: CaptureFixture[str]) -> None:
    exit_code = alloy_main(("serve", "--help"))

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Start the Alloy OpenAI-compatible server" in output
    assert "--hf-id" in output
