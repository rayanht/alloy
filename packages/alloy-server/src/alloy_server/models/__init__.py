"""Model-compat package: per-architecture handlers behind a registry.

Each architecture is one module implementing a ModelHandler — config/tensor
fixups, attention patches, vision/audio adapters, chat template, expert install —
registered via ``@register``. ``load_model`` dispatches arch -> handler; the
loading mechanics live in the sibling ``gguf/`` and ``mlx/`` packages. An
unrecognized arch (only under ``--force``) falls back to a bare ``CausalLMHandler``.
"""

from __future__ import annotations

from alloy_server.models.registry import (
    REGISTRY,
    LoadedModel,
    ModelHandler,
    apply_transformers_patches,
    check_arch_supported,
    ResolvedModel,
    load_model,
    load_native_causal_lm,
    load_resolved_causal_lm,
    model_kind,
    register,
    resolve_model,
)

# Import every arch module for its `@register` side effect.
from alloy_server.models import gemma4  # noqa: F401
from alloy_server.models import llama  # noqa: F401
from alloy_server.models import nomic_bert  # noqa: F401
from alloy_server.models import qwen3_5  # noqa: F401
from alloy_server.models import whisper  # noqa: F401

apply_transformers_patches()

__all__ = [
    "REGISTRY",
    "LoadedModel",
    "ModelHandler",
    "apply_transformers_patches",
    "check_arch_supported",
    "ResolvedModel",
    "load_model",
    "load_native_causal_lm",
    "load_resolved_causal_lm",
    "model_kind",
    "register",
    "resolve_model",
]
