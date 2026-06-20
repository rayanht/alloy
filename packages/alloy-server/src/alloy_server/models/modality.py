"""Modality seam: the protocol a model's non-text front-ends (vision, audio)
implement, the capture target the offline tools (`alloy tune` / `alloy profile`)
consume, and the `ModalityAdapter` base that holds the scaffolding shared across
adapters (placeholder expansion). Per-adapter internals (state-dict mapping, CPU
bookkeeping, compiled stages) live in each arch's module."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch


@dataclass(frozen=True)
class CaptureTarget:
    """One tunable/profilable forward a modality contributes to the offline tools.

    `alloy tune` runs `alloy.tune(module, inputs)` on it; `alloy profile` compiles
    `module` and visualizes `module(**inputs)`. `module` is the UNcompiled stage and
    `inputs` are fixed-shape representative kwargs (the shape is what tuning keys on,
    so dummy values are fine). `setup` warms any prerequisite state before the
    capture (e.g. a prefill before decode); the stateless vision stages leave it
    None."""

    name: str  # filename-safe slug, e.g. "vision_encode"
    label: str  # human label, e.g. "vision encode (patch + 16-layer ViT)"
    module: torch.nn.Module
    inputs: dict
    setup: Callable[[], None] | None = None


@runtime_checkable
class ModalityEncoder(Protocol):
    """A model's dense non-text front-end (vision OR audio) + its placeholder
    encoding. Built by the model handler when a multimodal GGUF is loaded, consumed
    model-agnostically by the served model to run image/audio requests through
    alloy's quantized decode."""

    # The placeholder token whose embedding a spliced feature replaces.
    placeholder_token_id: int

    def encode(self, data: bytes) -> torch.Tensor:
        """Input bytes (or a PIL image / waveform pair) -> `(num_soft_tokens,
        text_hidden)` features in the language model's embedding space."""
        ...

    def expand_text(self, text: str, features: list[torch.Tensor]) -> str:
        """Expand each placeholder in the chat-rendered `text` into the model's
        soft-token run, sized to each item's feature rows."""
        ...

    def eager_compile_all(self) -> None:
        """Compile this modality's alloy plans ahead of the first real request."""
        ...

    def capture_targets(self) -> list[CaptureTarget]:
        """The tunable/profilable forwards this modality dispatches (its compiled
        stages + fixed-shape representative inputs). `alloy tune` / `alloy profile`
        iterate these per modality."""
        ...


class ModalityAdapter:
    """Base for the per-arch vision/audio adapters. Holds the placeholder-expansion
    logic shared across modalities; subclasses set the markers + provide the
    modality-specific `encode` / `eager_compile_all` / `capture_targets`.

    `PLACEHOLDER` is the single token the chat template emits per item; each
    placeholder is replaced by `OPEN + PLACEHOLDER * num_soft_tokens + CLOSE`
    (gemma4 wraps with begin/end markers; a modality that just replicates the
    placeholder leaves OPEN/CLOSE empty)."""

    placeholder_token_id: int
    PLACEHOLDER: str
    OPEN: str = ""
    CLOSE: str = ""
    ITEM_NOUN: str = "item"

    def expand_text(self, text: str, features: list[torch.Tensor]) -> str:
        segments = text.split(self.PLACEHOLDER)
        if len(segments) - 1 != len(features):
            raise ValueError(
                f"{self.ITEM_NOUN} placeholder count ({len(segments) - 1}) != "
                f"{self.ITEM_NOUN}s ({len(features)})"
            )
        pieces = [segments[0]]
        for i, feats in enumerate(features):
            pieces.append(self.OPEN + self.PLACEHOLDER * int(feats.shape[0]) + self.CLOSE)
            pieces.append(segments[i + 1])
        return "".join(pieces)
