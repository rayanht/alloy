"""GGUF architecture detection: read `general.architecture` from the header."""

from __future__ import annotations

from pathlib import Path

from alloy_server.gguf.transformers_compat import walk_gguf_metadata


def gguf_architecture(path: Path) -> str:
    """Read `general.architecture` from a GGUF file's metadata header.

    Single-pass walk with an early stop at the target key, reading only the
    leading KV entries and never the tokenizer arrays. `gguf.GGUFReader` eagerly
    parses every KV field — ~9s of metadata parsing for a 248k-token vocab just
    to read one string."""
    fields, _ = walk_gguf_metadata(str(path), stop_at="general.architecture")
    arch = fields.get("general.architecture")
    if not isinstance(arch, str):
        raise ValueError(f"{path}: GGUF has no string `general.architecture` field")
    return arch
