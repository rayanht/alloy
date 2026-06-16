"""`alloy launch <target>` — start an AI coding tool wired to the local alloy server.

Sets the right environment variables so the target tool talks to the
running alloy server instead of a cloud API, then execs into it. The
model is whatever the server is serving — launch never picks, loads, or
starts anything; if no server is running it errors and points at
`alloy serve -m <model>`.

Supported targets:

  claude    — Claude Code CLI (Anthropic SDK)
  codex     — OpenAI Codex CLI
  aider     — Aider AI pair programmer
  goose     — Block Goose agent
  cursor    — Cursor editor (macOS app)
  windsurf  — Windsurf editor (macOS app)

Any other tool can be wired via --openai or --anthropic.
"""

from __future__ import annotations

import os
import shutil
from typing import Annotated

import typer
from rich.console import Console

from alloy_cli.client import DEFAULT_HOST, DEFAULT_PORT, ServerClient

console = Console()

# ---------------------------------------------------------------------------
# Target definitions
# ---------------------------------------------------------------------------
# Placeholders in env values:
#   {base_url}  — http://<host>:<port>
#   {model}     — the served model name (read from the server's /healthz)
#
# "cmd"     — executable on PATH to exec into
# "mac_app" — macOS app name for `open -a` (GUI apps)
# "extra_args" — prepended to the exec argv (with placeholder expansion)

_ANTHROPIC_ENV = {
    "ANTHROPIC_BASE_URL": "{base_url}",
    "ANTHROPIC_API_KEY": "alloy",
    "ANTHROPIC_MODEL": "{model}",
}

_OPENAI_ENV = {
    "OPENAI_BASE_URL": "{base_url}/v1",
    "OPENAI_API_KEY": "alloy",
}

_TARGETS: dict[str, dict] = {
    "claude": {
        "cmd": "claude",
        "description": "Claude Code",
        # DISABLE_NON_ESSENTIAL_MODEL_CALLS: Claude Code fires side calls
        # (topic detection, suggestions) in parallel with the main turn.
        # Alloy serves ONE model with ONE KV cache — alternating requests
        # evict the conversation's warm prefix and force a full re-prefill
        # of the (large) system+tools prompt every turn.
        "env": {**_ANTHROPIC_ENV, "DISABLE_NON_ESSENTIAL_MODEL_CALLS": "1"},
    },
    "codex": {
        "cmd": "codex",
        "description": "OpenAI Codex CLI",
        "env": _OPENAI_ENV,
    },
    "aider": {
        "cmd": "aider",
        "description": "Aider",
        "env": _OPENAI_ENV,
        "extra_args": ["--model", "{model}"],
    },
    "goose": {
        "cmd": "goose",
        "description": "Block Goose",
        "env": _OPENAI_ENV,
    },
    "cursor": {
        "mac_app": "Cursor",
        "description": "Cursor",
        "env": _OPENAI_ENV,
    },
    "windsurf": {
        "mac_app": "Windsurf",
        "description": "Windsurf",
        "env": _OPENAI_ENV,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _expand(template: str, base_url: str, model: str) -> str:
    return template.replace("{base_url}", base_url).replace("{model}", model)


def _build_env(target_env: dict[str, str], base_url: str, model: str) -> dict[str, str]:
    env = os.environ.copy()
    for key, val in target_env.items():
        env[key] = _expand(val, base_url, model)
    return env


def _served_chat_model(client: ServerClient, base_url: str) -> str:
    """The one model the running server serves, from /healthz. Errors out
    (exit 1) when no server is running or it serves an embedding model."""
    status = client.healthz()
    if not status.reachable:
        console.print(
            f"[red]no alloy server running at {base_url}[/]\n"
            f"start one first: [bold]alloy serve -m <model>[/] "
            f"(e.g. alloy serve -m qwen3.5:4b), then re-run this command."
        )
        raise typer.Exit(1)
    if status.model is None or status.kind != "chat":
        served = f" (serving embedding model {status.model!r})" if status.model else ""
        console.print(
            f"[red]the alloy server at {base_url} serves no chat model{served}[/]\n"
            f"restart it with a chat model: [bold]alloy serve -m <model>[/]"
        )
        raise typer.Exit(1)
    return status.model


def _launch_mac_app(
    app_name: str, description: str, env: dict[str, str], base_url: str, model: str
) -> None:
    # Build --env key=value pairs for env vars that differ from the current env.
    open_env_args: list[str] = []
    for k, v in env.items():
        if os.environ.get(k) != v:
            open_env_args += ["--env", f"{k}={v}"]

    console.print(
        f"[green]launching[/] {description} "
        f"[dim]→ alloy/{model} at {base_url}[/dim]"
    )
    os.execvp("open", ["open", "-a", app_name, *open_env_args])


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------

def launch(
    target: Annotated[
        str,
        typer.Argument(
            help="Tool to launch: claude, codex, aider, goose, cursor, windsurf — "
                 "or any executable with --openai / --anthropic.",
        ),
    ],
    host: Annotated[str, typer.Option("--host", help="Alloy server host.")] = DEFAULT_HOST,
    port: Annotated[int, typer.Option("--port", help="Alloy server port.")] = DEFAULT_PORT,
    openai: Annotated[
        bool,
        typer.Option("--openai", help="Wire OPENAI_BASE_URL/OPENAI_API_KEY (unknown OpenAI-compatible tool)."),
    ] = False,
    anthropic: Annotated[
        bool,
        typer.Option("--anthropic", help="Wire ANTHROPIC_BASE_URL/ANTHROPIC_API_KEY/ANTHROPIC_MODEL (unknown Anthropic-compatible tool)."),
    ] = False,
) -> None:
    """Launch an AI coding tool wired to the running alloy server.

    Uses whatever model the server is serving (read from /healthz) and
    execs into the target tool with the right environment variables set
    so it talks to alloy instead of a cloud API. Errors out if no server
    is running — start one with `alloy serve -m <model>` first.

    \b
    Examples:
      alloy serve -m qwen3.5:4b      # in another terminal
      alloy launch claude
      alloy launch codex
      alloy launch my-tool --openai
    """
    cfg = _TARGETS.get(target.lower())
    if cfg is None and not openai and not anthropic:
        names = ", ".join(sorted(_TARGETS))
        console.print(
            f"[red]unknown target:[/] {target!r}\n"
            f"Known targets: {names}\n"
            f"For a custom tool pass --openai or --anthropic."
        )
        raise typer.Exit(1)

    if cfg is None:
        cfg = {
            "cmd": target,
            "description": target,
            "env": _OPENAI_ENV if openai else _ANTHROPIC_ENV,
        }

    # Explicit flag overrides the target's default API dialect.
    if openai:
        cfg = dict(cfg, env=_OPENAI_ENV)
    elif anthropic:
        cfg = dict(cfg, env=_ANTHROPIC_ENV)

    base_url = f"http://{host}:{port}"
    client = ServerClient(base_url=base_url)
    model = _served_chat_model(client, base_url)
    env = _build_env(cfg["env"], base_url, model)

    # --- Launch target --------------------------------------------------
    mac_app = cfg.get("mac_app")
    if mac_app:
        _launch_mac_app(mac_app, cfg["description"], env, base_url, model)
        return

    cmd = cfg.get("cmd", target)
    exe = shutil.which(cmd)
    if exe is None:
        console.print(
            f"[red]error:[/] {cmd!r} not found on PATH — "
            f"install {cfg['description']} first."
        )
        raise typer.Exit(1)

    extra_args = [_expand(a, base_url, model) for a in cfg.get("extra_args", [])]
    console.print(
        f"[green]launching[/] {cfg['description']} "
        f"[dim]→ alloy/{model} at {base_url}[/dim]"
    )
    os.execvpe(exe, [exe, *extra_args], env)
