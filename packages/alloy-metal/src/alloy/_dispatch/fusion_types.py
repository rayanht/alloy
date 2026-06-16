"""Shared fusion contracts.

This module is intentionally leaf-level: analysis, row-pass, multi-root,
transform, and compilation can all import these records without forming cycles.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from alloy._runtime.alloy_buffer import AlloyBuffer
from alloy._runtime.metal import CompiledKernel

Grid3D = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class DispatchLaunch:
    """One Metal kernel launch after fusion and compilation."""

    kernel: CompiledKernel | int
    buffers: tuple[AlloyBuffer, ...]
    grid: Grid3D
    threadgroup: Grid3D
    write_indices: frozenset[int]
    debug_name: str

    @property
    def pso_handle(self) -> int:
        if isinstance(self.kernel, CompiledKernel):
            return self.kernel._handle
        return self.kernel


@dataclass(frozen=True, slots=True)
class PlanBufferBinding:
    """Buffer identity captured for compiled-plan replay."""

    root_ptr: int
    byte_offset: int
    nbytes: int
    # (extent, byte_stride) per axis of the bound view. Consumed by the
    # grid-shrink recipe to verify the M axis is OUTERMOST in every
    # written buffer — a 1D-flattened kernel whose output stores M innermost
    # (e.g. the rope-table broadcast (1, freqs, M)) covers the WRONG elements
    # when its threadgroup prefix is shrunk, so it must keep the full grid.
    dims: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True, slots=True)
class RecordedDispatch:
    """A typed launch record consumed by the torch compiled-plan builder."""

    pso_handle: int
    buffers: tuple[PlanBufferBinding, ...]
    grid: Grid3D
    threadgroup: Grid3D
    write_indices: frozenset[int]
    debug_name: str
    # MSL + entry point needed by the L5 plan cache to recompile the
    # pso_handle in a fresh process. Empty string for handles that bypass
    # CompiledKernel.from_msl (the cached-fusion-dispatch path); those
    # dispatches make the plan non-cacheable and we fall back to fresh
    # compilation rather than serialise a partial plan.
    msl_source: str = ""
    function_name: str = ""


FusedPair = tuple[DispatchLaunch, set[int]]


class FusionUnsupported(Exception):
    """Raised when a fusion group cannot be compiled."""

    def __init__(self, reason: str, *, op_idx: int | None = None) -> None:
        self.reason = reason
        self.op_idx = op_idx
        super().__init__(reason)


class FusionKind(enum.Enum):
    INDIVIDUAL = "individual"
    ANCHOR = "anchor"
    ELEM_CHAIN = "elem_chain"
    ROW_PASS = "row_pass"
    MULTI_ROOT = "multi_root"
    REDUCE_FOLD = "reduce_fold"


class FusionPlan:
    """A single fusion decision from analysis. No compilation info."""

    __slots__ = (
        "kind",
        "indices",
        "anchor_idx",
        "pro_chain",
        "epi_chain",
        "extra_branches",
        "idx",
        "chain",
    )

    def __init__(
        self,
        kind: FusionKind,
        indices: set[int],
        anchor_idx: int | None = None,
        pro_chain: list[int] | None = None,
        epi_chain: list[int] | None = None,
        extra_branches: list[list[int]] | None = None,
        idx: int | None = None,
        chain: list[int] | None = None,
    ) -> None:
        self.kind = kind
        self.indices = indices
        self.anchor_idx = anchor_idx
        self.pro_chain = pro_chain or []
        self.epi_chain = epi_chain or []
        self.extra_branches = extra_branches or []
        self.idx = idx
        self.chain = chain or []


@dataclass(slots=True)
class FusionGroup:
    """A proposed set of ops to fuse into one Metal dispatch."""

    op_indices: list[int]
    kind: FusionKind
    anchor_idx: int | None = None
    pro_chain: list[int] = field(default_factory=list)
    epi_chain: list[int] = field(default_factory=list)
    extra_branches: list[list[int]] = field(default_factory=list)

    @property
    def indices(self) -> set[int]:
        return set(self.op_indices)
