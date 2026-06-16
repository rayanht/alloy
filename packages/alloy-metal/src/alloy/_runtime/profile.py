"""Profiling infrastructure for Alloy dispatch pipeline.

Enable with ALLOY_PROFILE=1 or al.set_profile(True).
Zero overhead when disabled: single bool check at each instrumentation point.
"""

import json
import os
from dataclasses import dataclass, field

# --- Phase constants ---
QUEUE = "queue"  # _queue_op preamble: arg processing, constexpr, alloc
TRACE = "trace"  # trace_kernel: AST→IR
GRID = "grid"  # grid derivation from IR
CACHE_KEY = "cache_key"  # cache key construction
CODEGEN = "codegen"  # emit_msl_from_tile_ir
COMPILE = "compile"  # Metal shader compilation
BUF_PREP = "buf_prep"  # MetalBuffer wrapping
FUSION = "fusion"  # graph walk + IR composition + compile
ENCODE = "encode"  # C++: nanobind unpack + MTLBuffer lookup + command encoding
WAIT = "wait"  # C++: commit + waitUntilCompleted
COPY = "copy"  # C++: memcpy back for non-aligned buffers
GPU = "gpu"  # GPU execution only (Metal timestamp)
TOTAL = "total"  # wall-clock: _queue_op entry → GPU completion

_ALL_PHASES = [
    QUEUE,
    TRACE,
    GRID,
    CACHE_KEY,
    CODEGEN,
    COMPILE,
    BUF_PREP,
    FUSION,
    ENCODE,
    WAIT,
    COPY,
    GPU,
    TOTAL,
]

# --- Global enable flag ---
_profile_enabled = os.environ.get("ALLOY_PROFILE", "0") == "1"


@dataclass
class DispatchRecord:
    """One kernel dispatch with per-phase timing."""

    name: str
    phases: dict = field(default_factory=dict)
    cache_level: str = "miss"  # "dispatch" (L0), "msl" (L1), "pipeline" (L2), "miss"
    n_bufs: int = 0
    n_zero_copy: int = 0
    n_copied: int = 0
    grid: tuple = (0, 0, 0)
    threadgroup: tuple = (0, 0, 0)
    _total_t0: int = 0  # perf_counter_ns start for wall-clock total


class ProfileAccumulator:
    """Collects DispatchRecords, computes summaries."""

    def __init__(self):
        self.records: list[DispatchRecord] = []
        self._dispatch_hits = 0
        self._dispatch_misses = 0
        self._msl_hits = 0
        self._pipeline_hits = 0

    def record_cache_hit(self, level: str):
        if level == "dispatch":
            self._dispatch_hits += 1
        elif level == "msl":
            self._msl_hits += 1
        elif level == "pipeline":
            self._pipeline_hits += 1

    def record_cache_miss(self):
        self._dispatch_misses += 1

    def summary(self) -> str:
        if not self.records:
            return "No profiling data collected."

        # Phase totals
        phase_totals = {p: 0.0 for p in _ALL_PHASES}
        phase_counts = {p: 0 for p in _ALL_PHASES}
        for rec in self.records:
            for p in _ALL_PHASES:
                if p in rec.phases:
                    phase_totals[p] += rec.phases[p]
                    phase_counts[p] += 1

        total_bufs = sum(r.n_bufs for r in self.records)
        total_zc = sum(r.n_zero_copy for r in self.records)
        total_copied = sum(r.n_copied for r in self.records)

        lines = []
        lines.append(f"Alloy Profile: {len(self.records)} dispatches")
        lines.append("")
        lines.append(f"{'Phase':<12} {'Total ms':>10} {'Count':>6} {'Avg ms':>10}")
        lines.append("-" * 42)
        for p in _ALL_PHASES:
            if phase_counts[p] > 0:
                avg = phase_totals[p] / phase_counts[p]
                lines.append(f"{p:<12} {phase_totals[p]:>10.3f} {phase_counts[p]:>6} {avg:>10.3f}")

        lines.append("")
        lines.append(
            f"Cache: dispatch={self._dispatch_hits} msl={self._msl_hits} "
            f"pipeline={self._pipeline_hits} miss={self._dispatch_misses}"
        )
        if total_bufs > 0:
            lines.append(
                f"Buffers: {total_bufs} total, {total_zc} zero-copy, "
                f"{total_copied} copied ({100 * total_copied / total_bufs:.0f}%)"
            )

        return "\n".join(lines)

    def to_json(self) -> str:
        data = {
            "dispatches": len(self.records),
            "phase_totals": {},
            "cache": {
                "dispatch_hits": self._dispatch_hits,
                "msl_hits": self._msl_hits,
                "pipeline_hits": self._pipeline_hits,
                "misses": self._dispatch_misses,
            },
            "records": [],
        }
        for p in _ALL_PHASES:
            total = sum(r.phases.get(p, 0.0) for r in self.records)
            if total > 0:
                data["phase_totals"][p] = round(total, 4)
        for rec in self.records:
            data["records"].append(
                {
                    "name": rec.name,
                    "phases": {k: round(v, 4) for k, v in rec.phases.items()},
                    "cache_level": rec.cache_level,
                    "n_bufs": rec.n_bufs,
                    "n_zero_copy": rec.n_zero_copy,
                    "n_copied": rec.n_copied,
                    "grid": list(rec.grid),
                    "threadgroup": list(rec.threadgroup),
                }
            )
        return json.dumps(data, indent=2)

    def reset(self):
        self.records.clear()
        self._dispatch_hits = 0
        self._dispatch_misses = 0
        self._msl_hits = 0
        self._pipeline_hits = 0


# --- Singleton ---
_accumulator: ProfileAccumulator | None = None


def get_accumulator() -> ProfileAccumulator:
    global _accumulator
    if _accumulator is None:
        _accumulator = ProfileAccumulator()
    return _accumulator


# --- Public API ---


def set_profile(enabled: bool = True):
    global _profile_enabled
    _profile_enabled = enabled


def profile_summary() -> str:
    return get_accumulator().summary()


def profile_json() -> str:
    return get_accumulator().to_json()


def profile_reset():
    get_accumulator().reset()
