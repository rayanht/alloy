"""Sequence: the request as a first-class object.

Exactly one Sequence is live at a time (single-user serving), but the
generation pipeline operates on the Sequence, not on generator-global state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True, slots=True)
class SamplingParams:
    """Decode sampling config parsed from a request. temperature <= 0 is greedy
    (the on-GPU sampler returns an exact argmax). Defaults are greedy, so a
    request that omits all sampling fields decodes deterministically."""

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    seed: int = 0


@dataclass(slots=True)
class MultimodalInputs:
    """Vision/audio soft tokens for a request: `features[i]` replaces the
    embedding at `positions[i]` (an image/audio placeholder slot in the
    rendered prompt)."""

    features: torch.Tensor  # (n, hidden)
    positions: torch.Tensor  # (n,) absolute prompt positions


@dataclass(slots=True)
class Sequence:
    """One generation request and its produced state.

    `sampling=None` leaves the pinned sampling buffers untouched (greedy unless
    a previous request set otherwise); a SamplingParams writes them at request
    start. `stream=True` selects the small decode-chunk cascade so tokens reach
    the consumer in ~8-token bursts; non-streaming requests use larger chunks
    to amortize command-buffer commits.
    """

    input_ids: torch.Tensor  # (1, prompt_len), int64
    max_new_tokens: int
    sampling: SamplingParams | None = None
    stream: bool = False
    constraint: object | None = None  # xgrammar matcher (decode-step mask hook)
    embeds: MultimodalInputs | None = None  # multimodal prefill input mode
    ignore_eos: bool = False  # decode exactly max_new_tokens (benchmarks: fixed tg count)
    # Filled by the pipeline:
    generated: list[int] = field(default_factory=list)  # decoded ids (incl. EOS)
    healed: list[int] = field(default_factory=list)  # truncation-heal ids
    finish_reason: str | None = None  # "stop" | "length"

    @property
    def prompt_len(self) -> int:
        return int(self.input_ids.shape[1])
