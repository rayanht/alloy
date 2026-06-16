"""GGUF loading infrastructure: resolution, metadata cache, quantized modules,
tokenizer/SPM, and the causal-LM tensor-load orchestrator.

This package owns the loading mechanics; the per-architecture handlers live in
the sibling ``models/`` package, which depends on this one (never the reverse).
The public surface below is the package API consumers import from
``alloy_server.gguf``. resolve / arch / meta / tokenizer / quant are submodules;
loader.py is the orchestrator.
"""

from __future__ import annotations

from alloy_server.gguf.arch import gguf_architecture
from alloy_server.gguf.loader import (
    CausalLMHooks,
    GGUFLoadReport,
    LoadedGGUFCausalLM,
    load_gguf_causal_lm,
)
from alloy_server.gguf.quant import (
    GGUFQ4_KEmbedding,
    GGUFQ4_KLinear,
    GGUFQ5_0Embedding,
    GGUFQ5_0Linear,
    GGUFQ6_KEmbedding,
    GGUFQ6_KLinear,
    GGUFQ8_0Embedding,
    GGUFQ8_0Linear,
    replace_quantized_weight,
    tensor_quantization,
)
from alloy_server.gguf.resolve import ResolvedGGUF, resolve_gguf

__all__ = [
    "CausalLMHooks",
    "GGUFLoadReport",
    "GGUFQ4_KEmbedding",
    "GGUFQ4_KLinear",
    "GGUFQ5_0Embedding",
    "GGUFQ5_0Linear",
    "GGUFQ6_KEmbedding",
    "GGUFQ6_KLinear",
    "GGUFQ8_0Embedding",
    "GGUFQ8_0Linear",
    "LoadedGGUFCausalLM",
    "ResolvedGGUF",
    "gguf_architecture",
    "load_gguf_causal_lm",
    "replace_quantized_weight",
    "resolve_gguf",
    "tensor_quantization",
]
