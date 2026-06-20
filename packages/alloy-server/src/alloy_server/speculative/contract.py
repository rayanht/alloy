"""
A *drafter* proposes tokens; the *session* (session.py) owns everything
method-independent: the round loop, the Gumbel exact-match verify, acceptance,
target-state rollback, grammar masking, and instrumentation. A drafter sees
only tokens, tap hidden states, and positions.

Contract rules every drafter must satisfy:

- **Position-keyed, append-only state.** Per-position drafter state (DFlash
  context-KV rows, PLD's token ring) is keyed by absolute sequence position and
  only ever appended or truncated from the tail, so it's trie-safe under
  conversation branches.
- **Fixed shapes.** `observe()` always receives ALL forwarded rows of a verify
  pass (or prefill chunk); acceptance only moves the session's committed-length
  pointer. Overshoot rows are dead — overwritten by the next round, never read.
- **The anchor is clean.** `propose()` conditions on the last committed token
  (always the previous round's bonus — the one committed token the target has
  not yet forwarded). Drafter state covers positions [0, position); the anchor
  enters by token id.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import torch

if TYPE_CHECKING:
    from alloy_server.generation.generator import AlloyGenerator


@dataclass(frozen=True)
class TargetTaps:
    """Which intermediate hidden states the target must emit for a drafter.

    `layer_ids` are decoder-layer indices whose OUTPUT hidden states are
    emitted (DFlash: the 5 fused context-feature layers). `post_norm` adds the
    post-final-norm hidden (the lm_head input — what the MTP head consumes).
    """

    layer_ids: tuple[int, ...] = ()
    post_norm: bool = False

    @property
    def empty(self) -> bool:
        return not self.layer_ids and not self.post_norm


@dataclass
class TapBatch:
    """Tap hidden states for one forwarded token range [start, start+rows).

    `layers[i]` is (1, rows, H) for TargetTaps.layer_ids[i]; `post_norm` is
    (1, rows, H) when requested. Tensors are the verify/prefill plan's pinned
    output buffers — VALID ONLY until the next target forward, so a drafter that
    needs them later must consume them in observe().
    """

    start: int
    rows: int
    layers: tuple[torch.Tensor, ...] = ()
    post_norm: torch.Tensor | None = None


@dataclass
class Proposal:
    """A linear draft: `tokens` extend the anchor. Empty = no proposal this
    round (the session runs a plain decode step)."""

    tokens: list[int]


@dataclass
class RoundStats:
    """Standardized per-round instrumentation, every drafter, every round."""

    proposed: int
    accepted: int
    bonus: bool
    draft_us: float = 0.0
    verify_us: float = 0.0
    state_us: float = 0.0
    host_us: float = 0.0


@dataclass
class SpecMetrics:
    """Aggregated over one `SpecSession.run()` call (one request)."""

    rounds: int = 0
    proposed: int = 0
    accepted: int = 0
    committed: int = 0
    draft_us: float = 0.0
    verify_us: float = 0.0
    state_us: float = 0.0
    host_us: float = 0.0
    per_round: list[RoundStats] = field(default_factory=list)

    def add(self, r: RoundStats) -> None:
        self.rounds += 1
        self.proposed += r.proposed
        self.accepted += r.accepted
        self.committed += r.accepted + (1 if r.bonus else 0)
        self.draft_us += r.draft_us
        self.verify_us += r.verify_us
        self.state_us += r.state_us
        self.host_us += r.host_us

    @property
    def tau(self) -> float:
        """Mean committed tokens per round (the paper's acceptance length)."""
        return self.committed / self.rounds if self.rounds else 0.0

    @property
    def acceptance(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0


class Drafter(Protocol):
    """What a speculative-decoding method implements. See module docstring for
    the contract rules (positional state, fixed shapes, clean anchor)."""

    name: str
    max_draft_tokens: int
    taps: TargetTaps

    def bind(self, gen: "AlloyGenerator") -> None:
        """Called once at attach: share embed/lm_head/rope with the target,
        allocate persistent state buffers (native-context-sized, demand-paged),
        build modules. No compilation here — that's warmup()."""
        ...

    def warmup(self) -> None:
        """Compile + pin all draft-side plans. Called from eager_compile_all
        (and lazily before the first round if eager compile was skipped)."""
        ...

    def observe(self, tokens: list[int], taps: TapBatch | None, start: int) -> None:
        """The target forwarded `tokens` at positions [start, start+len).
        Ingest tap hidden states into per-position drafter state. Called for
        every prefill chunk and every verify pass — including rows past the
        accepted point (fixed shapes; the session's later truncate()/pointer
        movement makes overshoot rows dead)."""
        ...

    def propose(self, anchor: int, position: int) -> Proposal:
        """Draft up to max_draft_tokens continuing `anchor` at `position`.
        Drafter state must cover [0, position) — the session guarantees it."""
        ...

    def truncate(self, length: int) -> None:
        """Drop drafter state at positions >= length (warm-prefix LCP
        truncation, branch rewind, round overshoot reset)."""
        ...

    def state_bytes_per_token(self) -> int:
        """Per-token bytes of persistent drafter state — feeds the generator's
        fill-budget accounting (_kv_bytes_per_token)."""
        ...

    def snapshot_head(self, rows: int) -> object | None:
        """Snapshot per-position state for positions [0, rows) ahead of a
        foreign side request that will overwrite the cache head
        (AlloyGenerator.preserving_prefix). None when the drafter keeps no
        per-position GPU state (PLD)."""
        ...

    def restore_head(self, snap: object) -> None:
        """Restore a snapshot_head() payload."""
        ...
