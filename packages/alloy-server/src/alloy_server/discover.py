"""Discover GGUF models already on disk in the Ollama and HuggingFace caches.

Read-only: we only walk existing dirs and report what we find — alloy never
downloads, stores, or curates models (use `hf download` / an existing Ollama
install). Backs `alloy list`, `alloy show`, and the server's `/api/tags`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True, slots=True)
class DiscoveredModel:
    """One model found on disk."""

    name: str
    """Serve-able ref: `<base>:<tag>` (Ollama) or `<org>/<repo>` (HF)."""
    source: str
    """`ollama` or `huggingface`."""
    path: str
    """Filesystem path that gave us the hit (manifest file or snapshot dir)."""
    size_bytes: int = 0

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "source": self.source,
            "path": self.path,
            "size_bytes": self.size_bytes,
        }


def ollama_models_dir() -> Path:
    """Honors `OLLAMA_MODELS` env (Ollama's own override), else `~/.ollama/models`."""
    override = os.environ.get("OLLAMA_MODELS")
    if override:
        return Path(override)
    return Path("~/.ollama/models").expanduser()


def hf_hub_dir() -> Path:
    """Honors `HUGGINGFACE_HUB_CACHE` / `HF_HOME`, else `~/.cache/huggingface/hub`."""
    hub = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hub:
        return Path(hub)
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home) / "hub"
    return Path("~/.cache/huggingface/hub").expanduser()


def discover_ollama(root: Path | None = None) -> list[DiscoveredModel]:
    """List models in Ollama's manifest tree.

    Layout: `<root>/manifests/registry.ollama.ai/<namespace>/<name>/<tag>` where
    each `<tag>` is an OCI manifest JSON file. Ollama shows `library` models as
    bare `name:tag`, so we follow suit. Size is the sum of layer blob sizes.
    """
    models_root = root or ollama_models_dir()
    base = models_root / "manifests" / "registry.ollama.ai"
    if not base.is_dir():
        return []
    out: list[DiscoveredModel] = []
    for ns_dir in _safe_iterdir(base):
        if not ns_dir.is_dir():
            continue
        for name_dir in _safe_iterdir(ns_dir):
            if not name_dir.is_dir():
                continue
            for tag_file in _safe_iterdir(name_dir):
                if not tag_file.is_file() or tag_file.name.startswith("."):
                    continue
                display_base = (
                    name_dir.name if ns_dir.name == "library"
                    else f"{ns_dir.name}/{name_dir.name}"
                )
                out.append(DiscoveredModel(
                    name=f"{display_base}:{tag_file.name}",
                    source="ollama",
                    path=str(tag_file),
                    size_bytes=_ollama_manifest_size(tag_file),
                ))
    return out


def _ollama_manifest_size(manifest_path: Path) -> int:
    """Sum `config.size` + `layers[*].size` in an Ollama OCI manifest. 0 on a
    parse miss — discovery should still surface the model even unsized."""
    try:
        with manifest_path.open("rb") as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError):
        return 0
    total = 0
    config = manifest.get("config")
    if isinstance(config, dict):
        total += int(config.get("size", 0) or 0)
    layers = manifest.get("layers", [])
    if isinstance(layers, list):
        for layer in layers:
            if isinstance(layer, dict):
                total += int(layer.get("size", 0) or 0)
    return total


def discover_huggingface(root: Path | None = None) -> list[DiscoveredModel]:
    """List `models--<org>--<repo>` entries in the HF hub cache that contain at
    least one `.gguf` (alloy serves GGUF, so a safetensors-only repo isn't
    serve-able and is skipped). Half-pulled caches (no snapshot) are skipped."""
    base = root or hf_hub_dir()
    if not base.is_dir():
        return []
    out: list[DiscoveredModel] = []
    for entry in _safe_iterdir(base):
        if not entry.is_dir() or not entry.name.startswith("models--"):
            continue
        parts = entry.name[len("models--"):].split("--")
        if len(parts) < 2:
            continue
        org, repo = parts[0], "--".join(parts[1:])
        snapshots = entry / "snapshots"
        if not snapshots.is_dir():
            continue
        if not any(snapshots.glob("*/*.gguf")):
            continue
        out.append(DiscoveredModel(
            name=f"{org}/{repo}",
            source="huggingface",
            path=str(entry),
            size_bytes=_dir_size(entry / "blobs"),
        ))
    return out


def discover_all() -> list[DiscoveredModel]:
    """Both sources, deduplicated by `(source, name)`."""
    seen: set[tuple[str, str]] = set()
    out: list[DiscoveredModel] = []
    for model in (*discover_ollama(), *discover_huggingface()):
        key = (model.source, model.name)
        if key in seen:
            continue
        seen.add(key)
        out.append(model)
    return out


def _safe_iterdir(p: Path) -> Iterable[Path]:
    try:
        return sorted(p.iterdir())
    except (OSError, PermissionError):
        return []


def _dir_size(p: Path) -> int:
    """Sum of regular-file sizes under `p` (symlinks resolved to their blob)."""
    if not p.is_dir():
        return 0
    total = 0
    try:
        for root, _, files in os.walk(p, followlinks=False):
            for name in files:
                fp = Path(root) / name
                try:
                    target = fp.resolve() if fp.is_symlink() else fp
                    total += target.stat().st_size
                except OSError:
                    continue
    except OSError:
        pass
    return total
