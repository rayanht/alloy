"""MLX model-reference resolution: local dir / HuggingFace cache -> a model
directory (config.json + *.safetensors + tokenizer)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from alloy_server.gguf.resolve import hf_snapshot_dir, looks_like_local_path


@dataclass(frozen=True, slots=True)
class ResolvedMLX:
    """An MLX model reference resolved to a local directory. `config` is the
    parsed config.json; `safetensors` are the weight shards."""

    ref: str
    model_dir: Path
    safetensors: tuple[Path, ...]
    config: dict
    digest: str
    format: ClassVar[str] = "mlx"

    @property
    def location(self) -> Path:
        return self.model_dir

    def architecture(self) -> str:
        return str(self.config["model_type"])


def mlx_quantization(config: dict) -> dict | None:
    """The MLX quantization block (`{group_size, bits}`) from a config dict, or
    None if the model isn't MLX-quantized."""
    quant = config.get("quantization") or config.get("quantization_config")
    if isinstance(quant, dict) and quant.get("bits") is not None:
        return quant
    return None


def looks_like_mlx_dir(directory: Path) -> bool:
    config_path = directory / "config.json"
    if not config_path.is_file() or not list(directory.glob("*.safetensors")):
        return False
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return mlx_quantization(config) is not None


def resolve_mlx_dir(ref: str, directory: Path) -> ResolvedMLX:
    config = json.loads((directory / "config.json").read_text())
    shards = tuple(sorted(directory.glob("*.safetensors")))
    payload = "".join(f"{p.name}:{p.stat().st_size}:{p.stat().st_mtime_ns}" for p in shards)
    digest = "mlx-" + hashlib.sha256(payload.encode()).hexdigest()[:40]
    return ResolvedMLX(ref=ref, model_dir=directory, safetensors=shards, config=config, digest=digest)


def resolve_mlx(ref: str) -> ResolvedMLX | None:
    """Resolve a ref to an MLX model dir, or None if it isn't an MLX model.
    Checks a local dir, then the local HF cache."""
    ref = ref.strip()
    if looks_like_local_path(ref):
        path = Path(ref).expanduser()
        if path.is_dir() and looks_like_mlx_dir(path):
            return resolve_mlx_dir(ref, path)
        return None
    if "/" in ref:
        repo_id = ref.partition(":")[0]
        snapshot = hf_snapshot_dir(repo_id)
        if snapshot is not None and looks_like_mlx_dir(snapshot):
            return resolve_mlx_dir(ref, snapshot)
    return None
