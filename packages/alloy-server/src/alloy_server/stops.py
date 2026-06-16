"""Stop-string handling and finish-reason resolution.

Pure text helpers, no model or HTTP: hold back partial stop-string / marker
matches across streamed deltas so a stop split over token boundaries never
leaks, and classify why a generation ended (stop vs length).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator


def earliest_stop(text: str, stops: tuple[str, ...]) -> tuple[int, str] | None:
    """(index, stop_string) of the earliest stop-string occurrence, or None."""
    best: tuple[int, str] | None = None
    for s in stops:
        if not s:
            continue
        i = text.find(s)
        if i != -1 and (best is None or i < best[0]):
            best = (i, s)
    return best


def stop_holdback(text: str, stops: tuple[str, ...]) -> int:
    """Longest suffix of `text` that is a (proper) prefix of some stop string.
    That suffix must be withheld from streaming output — the next token could
    complete a stop, in which case we'd have already leaked part of it."""
    best = 0
    for s in stops:
        if not s:
            continue
        for k in range(min(len(text), len(s) - 1), 0, -1):
            if text.endswith(s[:k]):
                best = max(best, k)
                break
    return best


def filter_stops(
    deltas: Iterator[str],
    stops: tuple[str, ...],
    on_stop: Callable[[str], None] | None = None,
) -> Iterator[str]:
    """Wrap a text-delta stream with stop-string handling: emit text up to the
    first stop (exclusive) then terminate, holding back any suffix that could be
    the start of a stop so a partial match never leaks. Terminating early stops
    the underlying generator (the consumer stops pulling). `on_stop` is called
    with the matched stop string when one fires."""
    if not stops:
        yield from deltas
        return
    acc = ""
    emitted = 0
    for delta in deltas:
        if not delta:
            continue
        acc += delta
        hit = earliest_stop(acc, stops)
        if hit is not None:
            cut, which = hit
            if cut > emitted:
                yield acc[emitted:cut]
            if on_stop is not None:
                on_stop(which)
            return
        safe_end = len(acc) - stop_holdback(acc, stops)
        if safe_end > emitted:
            yield acc[emitted:safe_end]
            emitted = safe_end
    # Stream ended (EOS / max_tokens) with no stop hit: flush the held-back tail.
    if len(acc) > emitted:
        yield acc[emitted:]


def finish_reason(token_ids: list[int], max_tokens: int, eos_token_ids: frozenset[int]) -> str:
    """Why a greedy generation ended. 'length' if it hit max_tokens without the
    model stopping on its own, else 'stop' (natural EOS).

    A max_tokens truncation is *healed* by appending a turn-end token PAST the
    cap, so the last token is EOS even though the model never stopped — `n >
    max_tokens` catches that. `n == max_tokens` with a non-EOS tail (heal
    disabled) is also length; a natural EOS lands at or under the cap."""
    n = len(token_ids)
    if n > max_tokens:
        return "length"
    hit_eos = n > 0 and token_ids[-1] in eos_token_ids
    return "length" if (not hit_eos and n >= max_tokens) else "stop"


def partial_suffix_len(buf: str, marker: str) -> int:
    """Length of the longest suffix of `buf` that is a proper prefix of `marker` —
    how much to hold back so a marker split across stream deltas isn't emitted as
    text before it can be matched whole."""
    for k in range(min(len(buf), len(marker) - 1), 0, -1):
        if buf.endswith(marker[:k]):
            return k
    return 0
