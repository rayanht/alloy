"""KV-cache quantization formats.

A `KVFormat` describes everything the cache, the write path, the attention
read path, and the memory accounting need to know about one quantized KV
layout. Formats are registered here and resolved once at generator init from
`alloy serve --kv-quant <name>` / `ALLOY_KV_QUANT`; `None` (format "none").
"""

from __future__ import annotations

import os
from dataclasses import dataclass

BLOCK_ELEMS_Q8_0 = 32  # quant group along head_dim, matches Q8_0 weight kernels


@dataclass(frozen=True)
class KVFormat:
    name: str
    # Effective bits per element INCLUDING scale metadata — feeds
    # `_kv_bytes_per_token` / `_derive_fill_budget`
    bits_per_elem_k: float
    bits_per_elem_v: float
    block_elems: int  # quant group along head_dim
    # 1 = token-granular (no cross-token state: ring-safe, flush-free).
    # >1 = tiled (tier 2); inadmissible on sliding-window ring layers.
    group_tokens: int

    def code_bytes_per_row(self, head_dim: int) -> int:
        """Packed code bytes for one token's K (or V) vector in one head."""
        assert head_dim % self.block_elems == 0, (head_dim, self.block_elems)
        return head_dim  # q8_0: 1 byte/elem; sub-byte formats override via bits

    def scale_count_per_row(self, head_dim: int) -> int:
        return head_dim // self.block_elems


Q8_0 = KVFormat(
    name="q8_0",
    # 8 code bits + fp16 scale per 32 elems = 8.5 bits/elem, KV bytes ÷1.88.
    bits_per_elem_k=8.5,
    bits_per_elem_v=8.5,
    block_elems=BLOCK_ELEMS_Q8_0,
    group_tokens=1,
)

KV_FORMATS: dict[str, KVFormat] = {Q8_0.name: Q8_0}


def resolve_kv_format(name: str | None) -> KVFormat | None:
    """Resolve a format name (CLI flag wins over ALLOY_KV_QUANT). `None` /
    "none" → the fp16 cache. Unknown names raise with the available set —
    a misspelled format must never silently fall back to fp16."""
    if name is None:
        name = os.environ.get("ALLOY_KV_QUANT")
    if name is None or name == "none":
        return None
    try:
        return KV_FORMATS[name]
    except KeyError:
        raise ValueError(
            f"unknown KV format {name!r}; available: none, {', '.join(sorted(KV_FORMATS))}"
        ) from None
