"""`alloy list` — installed GGUF models, a read-only unified view over the
Ollama store (`~/.ollama/models`) and the HuggingFace cache
(`~/.cache/huggingface/hub`).

`name` is the ref you'd pass to `alloy serve -m` (an HF repo expands to one row
per quant, `<org>/<repo>:<quant>`). quant/params come from the Ollama config
blob, or from the GGUF filename for HF.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

from alloy_server.discover import discover_huggingface, discover_ollama

console = Console()

# GGUF quant labels (Q4_K_M, Q8_0, IQ4_XS, BF16, …) and a parameter-size token
# (135M, 7B, 1.5B) as they appear in HF GGUF filenames.
_QUANT_RE = re.compile(r"(?i)(IQ\d+_\w+|Q\d+_K_[SML]|Q\d+_K|Q\d+_[01]|Q\d+_\d+|BF16|F16|F32)")
_PARAMS_RE = re.compile(r"(?i)(?<![A-Za-z0-9.])(\d+(?:\.\d+)?[BMK])(?![A-Za-z0-9])")


@dataclass(frozen=True, slots=True)
class ModelRow:
    name: str
    source: str
    params: str
    quant: str
    size_bytes: int


def list_models() -> None:
    """List installed GGUF models grouped by source."""
    rows = _ollama_rows() + _hf_rows()
    if not rows:
        console.print(
            "[dim]no GGUF models found in the Ollama or HuggingFace caches. "
            "fetch one with `hf download <org>/<repo> <file>.gguf`.[/]"
        )
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("name", style="bold", overflow="fold")
    table.add_column("source", style="dim")
    table.add_column("params", justify="right")
    table.add_column("quant", justify="right", style="dim")
    table.add_column("size", justify="right")

    # Group by family (the name before `:`) so related sizes/quants sit
    # together — qwen3.5:0.8b next to qwen3.5:2b, every quant of an HF repo in
    # one block — ordered by parameter count within each family.
    by_family: dict[str, list[ModelRow]] = {}
    for row in rows:
        by_family.setdefault(row.name.split(":", 1)[0], []).append(row)

    for i, family in enumerate(sorted(by_family, key=str.lower)):
        if i > 0:
            table.add_section()
        for row in sorted(by_family[family], key=_params_numeric):
            table.add_row(
                row.name, row.source, row.params or "—", row.quant or "—",
                _human_size(row.size_bytes),
            )
    console.print(table)


def _params_numeric(row: ModelRow) -> float:
    """Parse a param label into a sort key: '2.3B' → 2.3e9, '873.44M' → 8.7e8.
    Unparseable falls back to size in bytes for a stable within-family order."""
    raw = row.params.strip().upper()
    multipliers = {"B": 1e9, "M": 1e6, "K": 1e3}
    if raw and raw[-1] in multipliers:
        try:
            return float(raw[:-1]) * multipliers[raw[-1]]
        except ValueError:
            pass
    return float(row.size_bytes)


def _ollama_rows() -> list[ModelRow]:
    out: list[ModelRow] = []
    for model in discover_ollama():
        cfg = _ollama_config(Path(model.path))
        out.append(ModelRow(
            name=model.name,
            source="ollama",
            params=str(cfg.get("model_type") or ""),
            quant=str(cfg.get("file_type") or ""),
            size_bytes=model.size_bytes,
        ))
    return out


def _ollama_config(manifest_path: Path) -> dict:
    """The manifest's config blob (`blobs/sha256-<hex>` under the same store
    root) — a small JSON carrying Ollama's `model_type` / `file_type`."""
    try:
        manifest = json.loads(manifest_path.read_bytes())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    config = manifest.get("config") if isinstance(manifest, dict) else None
    if not isinstance(config, dict):
        return {}
    digest = config.get("digest")
    if not isinstance(digest, str) or ":" not in digest:
        return {}
    # manifest_path: <root>/manifests/registry.ollama.ai/<ns>/<name>/<tag>
    root = manifest_path.parents[4]
    try:
        data = json.loads((root / "blobs" / digest.replace(":", "-")).read_bytes())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _hf_rows() -> list[ModelRow]:
    out: list[ModelRow] = []
    for repo in discover_huggingface():
        for gguf in _repo_ggufs(Path(repo.path)):
            quant = _first_match(_QUANT_RE, gguf.name)
            out.append(ModelRow(
                name=f"{repo.name}:{quant}" if quant else repo.name,
                source="huggingface",
                params=_first_match(_PARAMS_RE, gguf.name),
                quant=quant,
                size_bytes=_file_size(gguf),
            ))
    return out


def _repo_ggufs(repo_dir: Path) -> list[Path]:
    """GGUF files in the repo's snapshots, deduplicated by filename."""
    seen: dict[str, Path] = {}
    for gguf in sorted((repo_dir / "snapshots").glob("*/*.gguf")):
        seen.setdefault(gguf.name, gguf)
    return list(seen.values())


def _first_match(pattern: re.Pattern[str], text: str) -> str:
    match = pattern.search(text)
    return match.group(1) if match else ""


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size  # follows the snapshot symlink to the blob
    except OSError:
        return 0


def _human_size(n: int) -> str:
    """Bytes → 'X.XX GB' (binary units, IEC convention used by `ls -h`)."""
    if n < 1024:
        return f"{n} B"
    value = float(n)
    for unit in ("KB", "MB", "GB", "TB"):
        value /= 1024
        if value < 1024:
            return f"{value:.2f} {unit}"
    return f"{value:.2f} PB"
