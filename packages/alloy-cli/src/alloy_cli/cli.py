"""Typer entrypoint that wires every `alloy` subcommand."""

from __future__ import annotations

import importlib

import typer
import typer.core
from typer.main import get_command_from_info
from typer.models import CommandInfo

import alloy

# name -> (module, function, short help)
_COMMANDS: dict[str, tuple[str, str, str]] = {
    "serve": ("alloy_cli.commands.serve", "serve",
              "Start the alloy server in the foreground (OpenAI/Ollama/Anthropic APIs)."),
    "launch": ("alloy_cli.commands.launch", "launch",
               "Launch an AI coding tool wired to the running alloy server."),
    "list": ("alloy_cli.commands.list_cmd", "list_models",
             "List installed GGUF models grouped by source."),
    "show": ("alloy_cli.commands.show", "show",
             "Resolve a model ref to its GGUF and print its header metadata."),
    "bench": ("alloy_cli.commands.bench", "bench",
              "Benchmark prefill (pp) + decode (tg) tok/s across a cache-depth sweep."),
    "tune": ("alloy_cli.commands.tune", "tune",
             "Tune a model's kernels at the chunked-prefill (M=chunk) and decode (M=1) shapes."),
    "profile": ("alloy_cli.commands.profile", "profile",
                "Capture dispatch-plan visualizations via alloy.visualize."),
    "microbench": ("alloy_cli.commands.microbench", "microbench",
                   "Report clock-pinned GPU timing for one kernel at its production config."),
    "inspect": ("alloy_cli.commands.inspect_cmd", "inspect",
                "Dump the real MSL/IR a model forward executes for a kernel (or its PSO stats)."),
    "pack": ("alloy_cli.commands.pack", "pack",
             "Build a distributable .alloypack for on-device Apple inference."),
    "pack-publish": ("alloy_cli.commands.pack", "pack_publish",
                     "Assemble catalog.json from built packs and (optionally) publish to HuggingFace."),
    "compile": ("alloy_cli.commands.compile_cmd", "compile_",
                "Pre-compile a model's dispatch plan and cache it under ~/.alloy/cache/."),
    "doctor": ("alloy_cli.commands.doctor", "doctor",
               "Run diagnostics. Exit non-zero if any check fails."),
    "version": ("alloy_cli.commands.version", "version",
                "Print Alloy version, build, and platform info."),
}

_EPILOG = (
    "Getting started:\n"
    "  alloy serve -m qwen3:0.6b   — start the server (foreground)\n"
    "  then point any OpenAI / Ollama / Anthropic client at http://127.0.0.1:11434"
)


def _build_command(name: str) -> typer.core.TyperCommand:
    module, attr, _ = _COMMANDS[name]
    func = vars(importlib.import_module(module))[attr]
    return get_command_from_info(
        CommandInfo(name=name, callback=func),
        pretty_exceptions_short=False,
        rich_markup_mode="markdown",
    )


class LazyTyperGroup(typer.core.TyperGroup):
    def list_commands(self, ctx: typer.Context) -> list[str]:
        return list(_COMMANDS)

    def get_command(self, ctx: typer.Context, name: str):
        return _build_command(name) if name in _COMMANDS else None

    def format_commands(self, ctx: typer.Context, formatter) -> None:
        rows = [(name, short) for name, (_, _, short) in _COMMANDS.items()]
        with formatter.section("Commands"):
            formatter.write_dl(rows)


app = typer.Typer(
    cls=LazyTyperGroup,
    name="alloy",
    help="Alloy — local LLM inference for Apple Silicon. Drop-in Ollama replacement.",
    epilog=_EPILOG,
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
    rich_markup_mode=None,
)


@app.callback()
def _root(
    log_level: str | None = typer.Option(
        None, "--log-level",
        help="Set log level (debug, info, warning, error). Overrides ALLOY_LOG.",
        case_sensitive=False,
    ),
) -> None:
    """Root callback — applies global flags before any subcommand runs."""
    if log_level is not None:
        alloy.configure_logging(level=log_level)


if __name__ == "__main__":
    app()
