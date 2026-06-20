"""Extern-KV write registry: liveness roots for side-effect cache writes.

The lazy collector only keeps ops reachable from the graph outputs, so a cache
write whose value is never read in-plan is dead-code-eliminated (the
sliding-window cold prefill hits this when chunk > window: the attend reads the
linear temp copy, not the ring). Handlers register the written cache buffers
here; the backend drains the list at graph OUTPUT and materializes them so the
write lands in the plan.
"""

from __future__ import annotations

from alloy._runtime.alloy_buffer import AlloyBuffer

EXTERN_KV_WRITES: list[AlloyBuffer] = []


def note_extern_kv_write(buf: AlloyBuffer) -> None:
    EXTERN_KV_WRITES.append(buf)


def drain_extern_kv_writes() -> list[AlloyBuffer]:
    if not EXTERN_KV_WRITES:
        return []
    drained = list(EXTERN_KV_WRITES)
    EXTERN_KV_WRITES.clear()
    return drained
