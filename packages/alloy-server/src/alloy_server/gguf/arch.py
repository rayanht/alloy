"""GGUF architecture detection. Reading `general.architecture` is pure GGUF
parsing; the supported-arch gate and a model's kind are derived from the
ModelHandler registry (`models/registry.py`), the source of truth."""

from __future__ import annotations

from pathlib import Path

from alloy_server.gguf.transformers_compat import walk_gguf_metadata


def gguf_architecture(path: Path) -> str:
    """Read `general.architecture` from a GGUF file's metadata header.

    Uses the single-pass walk with an early stop at the target key, so it
    reads only the leading KV entries (architecture is the first) and never
    touches the tokenizer arrays. `gguf.GGUFReader` would eagerly parse every
    KV field — for a 248k-token vocab that is ~9s of pure metadata parsing
    just to read one string."""
    fields, _ = walk_gguf_metadata(str(path), stop_at="general.architecture")
    arch = fields.get("general.architecture")
    if not isinstance(arch, str):
        raise ValueError(f"{path}: GGUF has no string `general.architecture` field")
    return arch
