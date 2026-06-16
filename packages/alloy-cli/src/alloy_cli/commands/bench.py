"""`alloy bench` — run the local LLM benchmark.

Default: pp512 / tg128 (prefill 512 tokens, decode 128) on the native KV cache.
Pass several `--depths` for a depth sweep, or `--dataset multimodal --image
PATH` to drive an image + prompt through the vision tower.

  alloy bench qwen3:0.6b llama3.2:1b                       # pp512 / tg128
  alloy bench qwen3:0.6b --depths 512 4096 16384           # depth sweep
  alloy bench gemma4:e2b --dataset multimodal --image cat.jpg
  alloy bench nomic-embed-text --dataset embeddings        # encoder tok/s
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import typer
from rich.console import Console

from alloy_cli import benchmark

console = Console()


class Dataset(str, Enum):
    synthetic = "synthetic"
    multimodal = "multimodal"
    embeddings = "embeddings"


def bench(
    models: list[str] = typer.Argument(  # noqa: B008
        ...,
        metavar="MODELS...",
        help="Model refs to benchmark, e.g. qwen3:0.6b llama3.2:1b.",
    ),
    dataset: Dataset = typer.Option(
        Dataset.synthetic,
        "--dataset",
        help="synthetic (pp/tg), multimodal (image + prompt), or embeddings (encoder tok/s).",
        case_sensitive=False,
    ),
    depths: list[int] = typer.Option(  # noqa: B008
        [512],
        "--depths",
        help="Prefill depth(s). One value (default 512) reports pp/tg; several "
             "(e.g. 512 4096 16384) run a depth sweep.",
    ),
    depth_gen: int = typer.Option(
        128, "--depth-gen", help="Decode length for the tg measurement (128 -> tg128)."
    ),
    reps: int = typer.Option(
        3, "--reps", help="Timed repetitions per depth; median reported."
    ),
    image: Path | None = typer.Option(
        None,
        "--image",
        help="multimodal only: path to the image file sent with each request.",
    ),
    mm_prompt: str = typer.Option(
        "Describe this image in detail.",
        "--mm-prompt",
        help="multimodal only: the text prompt paired with the image.",
    ),
    mm_reps: int = typer.Option(
        5, "--mm-reps", help="multimodal only: timed requests per model."
    ),
    mm_max_output: int = typer.Option(
        128, "--mm-max-output", help="multimodal only: per-request output cap (num_predict)."
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit machine-readable JSON instead of rich tables."
    ),
) -> None:
    """Benchmark prefill (pp) + decode (tg) tok/s across a cache-depth sweep."""
    argv: list[str] = [
        "--models", *models,
        "--dataset", dataset.value,
        "--depths", *(str(d) for d in depths),
        "--depth-gen", str(depth_gen),
        "--reps", str(reps),
    ]
    if dataset is Dataset.multimodal:
        if image is None:
            console.print("[red]--dataset multimodal requires --image PATH[/]")
            raise typer.Exit(1)
        argv += [
            "--image", str(image),
            "--mm-prompt", mm_prompt,
            "--mm-reps", str(mm_reps),
            "--mm-max-output", str(mm_max_output),
        ]
    if json_output:
        argv.append("--json")

    raise typer.Exit(benchmark.main(argv))
