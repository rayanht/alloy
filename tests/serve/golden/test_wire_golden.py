"""Wire-format golden gate.

Replays the dialect corpus against the current server and asserts every record
matches `wire_golden.json` byte-for-byte (structurally — see harness docstring).
This is the regression net for the whole server refactor: any change to dialect
parsing/rendering, SSE/NDJSON framing, error envelopes, or CORS that isn't a
deliberate, re-captured wire change trips here.
"""

from __future__ import annotations

import json

import harness


def _golden() -> dict:
    with open(harness.GOLDEN_PATH) as fh:
        return json.load(fh)


def test_wire_format_matches_golden() -> None:
    golden = _golden()
    actual = harness.capture_all()

    assert set(actual) == set(golden), (
        f"corpus drift: only-in-actual={sorted(set(actual) - set(golden))}, "
        f"only-in-golden={sorted(set(golden) - set(actual))} "
        f"(re-run capture_wire_golden.py if the corpus changed on purpose)"
    )
    mismatches = [key for key in golden if actual[key] != golden[key]]
    assert not mismatches, (
        "wire-format regressions in: " + ", ".join(mismatches)
        + "\nfirst diff:\n"
        + json.dumps({"key": mismatches[0],
                      "expected": golden[mismatches[0]],
                      "actual": actual[mismatches[0]]}, indent=1)
        if mismatches else ""
    )
