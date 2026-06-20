"""Regenerate the wire-format golden.

    uv run python tests/serve/golden/capture_wire_golden.py

Fires the dialect corpus at the current server with stub models and writes the
normalized records to `wire_golden.json`. `test_wire_golden.py` replays the same
corpus and asserts equality. Re-run this ONLY when a wire-format change is
intended.
"""

from __future__ import annotations

import json
import os
import sys


def main() -> None:
    sys.path.insert(0, os.path.dirname(__file__))
    import harness  # scoped: sibling module, needs sys.path set just above

    records = harness.capture_all()
    with open(harness.GOLDEN_PATH, "w") as fh:
        json.dump(records, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print(f"wrote {len(records)} records to {harness.GOLDEN_PATH}")


if __name__ == "__main__":
    main()
