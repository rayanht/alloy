"""`alloy serve -m MODEL` — run the Alloy HTTP server in the foreground.

Serves until interrupted (Ctrl-C), llama.cpp/vLLM-style: no LaunchAgent,
no background process, no client/daemon split. One serve process serves
exactly one model — chat or embedding, decided by the name — loaded and
pre-compiled at startup and held for the life of the process. Serving a
different model means restarting with a different `-m`.
"""

from __future__ import annotations

import os
from typing import Annotated

import typer

import alloy_server


def serve(
    model: Annotated[
        str,
        typer.Option(
            "--model", "-m",
            help="The one model this process serves (chat or embedding). "
                 "A local path (./model.gguf), a HuggingFace GGUF repo "
                 "(Org/Repo:Q4_K_M), or an Ollama name (qwen3.5:4b).",
        ),
    ],
    host: Annotated[str, typer.Option("--host", help="bind host")] = "127.0.0.1",
    port: Annotated[
        int, typer.Option("--port", help="bind port (Ollama-compat default)"),
    ] = 11434,
    kv_quant: Annotated[
        str | None,
        typer.Option(
            "--kv-quant",
            help="quantize the KV cache (e.g. q8_0). Opt-in and lossy; the "
            "default fp16 cache stays bit-identical. Exported as ALLOY_KV_QUANT.",
        ),
    ] = None,
    spec: Annotated[
        str | None,
        typer.Option(
            "--spec",
            help="speculative decoding drafter (dflash | mtp | pld). Opt-in; "
            "dflash needs the z-lab draft downloaded (hf download "
            "z-lab/<Model>-DFlash) — a missing draft fails at startup with "
            "the exact command.",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="load a GGUF whose architecture isn't in the supported set "
            "(best-effort; may fail later in compile).",
        ),
    ] = False,
) -> None:
    """Start the alloy server in the foreground (OpenAI/Ollama/Anthropic APIs)."""
    if kv_quant is not None:
        # Resolved at generator construction (AlloyGenerator.from_model); an
        # unknown name raises there with the available set, never silently fp16.
        os.environ["ALLOY_KV_QUANT"] = kv_quant

    argv: list[str] = [
        "--host", host,
        "--port", str(port),
        "--model", model,
    ]
    if spec is not None:
        argv += ["--spec", spec]
    if force:
        argv += ["--force"]
    raise typer.Exit(alloy_server.main(tuple(argv)))
