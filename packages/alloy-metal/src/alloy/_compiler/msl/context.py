"""Shared MSL emitter context records."""

from __future__ import annotations

from dataclasses import dataclass

from alloy._compiler.dtypes import from_ir
from alloy._compiler.tile_ir import TileValue


def msl_dtype_for_value(val: TileValue | None, fallback: str) -> str:
    """Map a TileValue dtype to an MSL type, falling back when dtype is absent."""
    if val is None or not val.dtype:
        return fallback
    try:
        return from_ir(val.dtype).msl
    except KeyError:
        return fallback


@dataclass(frozen=True, slots=True)
class ValLoc:
    """Location of a tile IR value in the emitter."""

    kind: str
    name: str = ""
    stride: int = 0


PER_THREAD = ValLoc("per_thread")
ADDRESS = ValLoc("address")
MMA = ValLoc("mma")
PERSISTENT_MMA = ValLoc("persistent_mma")
