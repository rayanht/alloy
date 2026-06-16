"""Prompt-lookup drafter (PLD)

Proposes continuations by matching the trailing n-gram of the committed
sequence against earlier occurrences in (prompt + generated) and replaying
what followed. Zero weights, zero GPU state, zero training; wins on
repetition-heavy content (file echoes, identifiers, edit loops — the
Claude Code shape) and degrades to plain decode on a miss (empty proposal →
the session runs a single decode step).

Contract notes: per-position state is just the token mirror (a numpy ring,
trivially positional/append-only); taps are not requested; snapshot/restore
for side requests is a list slice.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .contract import Proposal, TapBatch, TargetTaps

if TYPE_CHECKING:
    from alloy_server.generation.generator import AlloyGenerator


class PromptLookupDrafter:
    """N-gram prompt lookup. `ngram_max` down to `ngram_min` are tried in
    order; the LAST earlier occurrence wins (recency favors local repetition).
    """

    name = "pld"
    taps = TargetTaps()

    def __init__(
        self,
        max_draft_tokens: int = 7,
        ngram_max: int = 3,
        ngram_min: int = 2,
        capacity: int = 1 << 18,
    ) -> None:
        # 7 keeps the verify at M=8, inside attention_kv_update_multi's
        # _MAX_VERIFY_K fast path (block 16 is M-C1).
        self.max_draft_tokens = int(max_draft_tokens)
        self.ngram_max = int(ngram_max)
        self.ngram_min = int(ngram_min)
        self._buf = np.zeros((capacity,), dtype=np.int64)
        self._len = 0

    # ------------------------------------------------------------- contract

    def bind(self, gen: "AlloyGenerator") -> None:
        self._len = 0

    def warmup(self) -> None:
        pass  # nothing to compile

    def observe(self, tokens: list[int], taps: TapBatch | None, start: int) -> None:
        end = start + len(tokens)
        if end > self._buf.shape[0]:
            grown = np.zeros((max(end, self._buf.shape[0] * 2),), dtype=np.int64)
            grown[: self._len] = self._buf[: self._len]
            self._buf = grown
        self._buf[start:end] = tokens
        self._len = max(self._len, end)

    def propose(self, anchor: int, position: int) -> Proposal:
        # Effective sequence: positions [0, position) from the mirror + anchor.
        n_ctx = min(self._len, position)
        if n_ctx < self.ngram_min:
            return Proposal([])
        seq = self._buf[:n_ctx]
        k = self.max_draft_tokens
        for n in range(self.ngram_max, self.ngram_min - 1, -1):
            if n_ctx + 1 < n:
                continue
            # Pattern = last n tokens of (seq + [anchor]).
            pattern = np.empty((n,), dtype=np.int64)
            pattern[:-1] = seq[n_ctx - (n - 1):] if n > 1 else pattern[:0]
            pattern[-1] = anchor
            # Vectorized scan for pattern occurrences ENDING before `position`
            # (an occurrence ending at i proposes seq[i+1 : i+1+k]).
            window = n_ctx - n + 1  # candidate pattern starts in seq
            if window <= 0:
                continue
            hit = np.ones((window,), dtype=bool)
            for j in range(n):
                hit &= seq[j : j + window] == pattern[j]
            idxs = np.flatnonzero(hit)
            if idxs.shape[0] == 0:
                continue
            cont_start = int(idxs[-1]) + n  # token AFTER the matched n-gram
            cont = seq[cont_start : cont_start + k]
            if cont.shape[0] == 0:
                continue
            return Proposal([int(t) for t in cont])
        return Proposal([])

    def truncate(self, length: int) -> None:
        if length < self._len:
            self._len = length

    def state_bytes_per_token(self) -> int:
        return 0  # CPU mirror only; no GPU state in the fill budget

    def snapshot_head(self, rows: int) -> object | None:
        return None  # no per-position GPU state to protect

    def restore_head(self, snap: object) -> None:
        pass
