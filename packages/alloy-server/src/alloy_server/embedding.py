"""Embedding model contract.

An `EmbeddingModel` is a parallel concept to `ServedModel`: same name-keyed
registry on the server, but encoder-only — single forward pass, no KV cache, no
autoregressive decode. The alloy-native loader lives in `models/nomic_bert.py`
(`load_ollama_gguf_embedder`); this module just defines the contract it returns.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

# A model's token counter (`text -> token count`). Lives here (the model layer)
# rather than in `server/` so the embed/transcription loaders don't pull the
# server package (a models<->server cycle).
TextTokenCounter = Callable[[str], int]

EmbedFn = Callable[[list[str]], list[list[float]]]

_DEFAULT_MAX_BATCH = 64


@dataclass(frozen=True, slots=True)
class EmbeddingModel:
    """An encoder served alongside the chat models.

    `embed(texts)` returns a list of float vectors, one per input. The
    implementation owns batching internally; callers pass the full batch they
    want and trust the model to chunk if needed (or raise if the batch exceeds
    the model's `max_batch`).
    """

    name: str
    embed: EmbedFn
    dimensions: int
    count_tokens: TextTokenCounter
    max_batch: int = _DEFAULT_MAX_BATCH
