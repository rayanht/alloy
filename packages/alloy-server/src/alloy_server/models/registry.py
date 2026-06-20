"""Model-handler registry + the unified load entry.

`load_model(resolved) -> LoadedModel` reads a resolved model's architecture, looks
up its `ModelHandler`, gates support, and dispatches to `handler.load(...)`,
returning a kind-tagged result the server consumes uniformly (chat / embed /
transcription). Each arch registers a handler via `@register(...)`; an unrecognized
arch (only reachable under `--force`) falls back to a bare `CausalLMHandler`.

The registry is the source of truth for both the supported-arch set
(`check_arch_supported`) and a model's kind (`model_kind`). A causal-LM handler is
injected into `gguf.load_causal_lm` as the `CausalLMHooks`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from alloy_server.gguf import (
    LoadedGGUFCausalLM,
    ResolvedGGUF,
    resolve_gguf,
)
from alloy_server.mlx import ResolvedMLX, resolve_mlx
from alloy_server.models.base import CausalLMHandler

ResolvedModel = ResolvedGGUF | ResolvedMLX


class ModelHandler(Protocol):
    """The capabilities the registry + loader drive. Causal-LM handlers subclass
    `CausalLMHandler` (which satisfies `gguf.CausalLMHooks` for the load path);
    the whisper/nomic handlers provide a custom `load`."""

    arch: tuple[str, ...]
    kind: str

    def load(self, source: ResolvedGGUF, **kwargs: object) -> object: ...

    def apply_transformers_patches(self) -> None: ...


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """The kind-tagged result of `load_model` — the unified message the server
    dispatches on. `payload` is the per-kind loaded object: a
    `LoadedGGUFCausalLM` (chat), an `EmbeddingModel` (embed), or a
    `TranscriptionModel` (transcription). Typed `object` so the loader package
    doesn't pull the server-coupled embed/transcription types."""

    kind: str
    payload: object


REGISTRY: dict[str, ModelHandler] = {}


def register(*archs: str):
    """Class decorator: instantiate the handler and bind it to each `arch`
    (the GGUF `general.architecture` string)."""

    def decorator(cls: type) -> type:
        instance = cls()
        for arch in archs:
            REGISTRY[arch] = instance
        return cls

    return decorator


def apply_transformers_patches() -> None:
    """Apply every registered handler's transformers-registry contributions
    (GGUF_CONFIG_MAPPING / converters). Idempotent; runs at package import before
    any GGUF parse."""
    for handler in set(REGISTRY.values()):
        handler.apply_transformers_patches()


# Fallback for an arch with no registered handler — only reached for an
# unrecognized arch loaded under `--force` (the gate rejects it otherwise). A
# bare CausalLMHandler loads it through the generic path with no-op hooks.
DEFAULT_HANDLER = CausalLMHandler()


def handler_for(arch: str) -> ModelHandler:
    """The handler bound to `arch`, or the bare default for an unregistered arch."""
    return REGISTRY.get(arch, DEFAULT_HANDLER)


def model_kind(arch: str) -> str:
    """The served kind for `arch` (`"chat"` | `"embed"` | `"transcription"`),
    read from its handler — the registry is the source of truth."""
    return handler_for(arch).kind


def check_arch_supported(arch: str, *, force: bool = False) -> None:
    """Gate on whether `arch` has a registered handler. `--force` bypasses it
    (the bare default handler then attempts a generic load)."""
    if force or arch in REGISTRY:
        return
    supported = ", ".join(sorted(REGISTRY))
    raise ValueError(
        f"GGUF architecture {arch!r} is not in the supported set ({supported}); "
        f"pass --force to attempt loading it anyway."
    )


def resolve_model(ref: str, *, root: Path | None = None) -> ResolvedModel:
    """Resolve a model reference to a concrete on-disk source. An MLX-quantized
    model dir/repo wins; otherwise it's a GGUF (local path / HF repo / Ollama)."""
    resolved_mlx = resolve_mlx(ref)
    if resolved_mlx is not None:
        return resolved_mlx
    return resolve_gguf(ref, root=root)


def load_model(resolved: ResolvedModel, *, force: bool = False) -> LoadedModel:
    """Gate a resolved model on its architecture and load it through the matching
    handler — returning a kind-tagged `LoadedModel`."""
    arch = resolved.architecture()
    check_arch_supported(arch, force=force)
    handler = handler_for(arch)
    return LoadedModel(kind=handler.kind, payload=handler.load(resolved))


def load_resolved_causal_lm(
    source: ResolvedModel,
    *,
    dtype: object | None = None,
    load_tokenizer: bool = True,
) -> LoadedGGUFCausalLM:
    """Load an already-resolved causal LM via its handler (the arch gate is the
    caller's)."""
    handler = handler_for(source.architecture())
    return handler.load(source, dtype=dtype, load_tokenizer=load_tokenizer)


def load_native_causal_lm(
    ref: str,
    *,
    dtype: object | None = None,
    load_tokenizer: bool = True,
    root: Path | None = None,
    force: bool = False,
) -> LoadedGGUFCausalLM:
    """Resolve a model reference (MLX dir/repo, GGUF local path / HF repo / Ollama
    name), gate on its architecture, and load it through the matching handler."""
    source = resolve_model(ref, root=root)
    check_arch_supported(source.architecture(), force=force)
    return load_resolved_causal_lm(source, dtype=dtype, load_tokenizer=load_tokenizer)
