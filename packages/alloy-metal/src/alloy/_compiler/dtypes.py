"""Alloy dtype system — single source of truth for all type information.

DType is the canonical representation: a lightweight frozen object carrying
IR name, MSL type, itemsize, and ctype.  Singletons exist for every
supported type.  Numpy and torch conversions happen at boundaries only.
"""

from __future__ import annotations

import ctypes as _ct
import importlib

import numpy as np


class DType:
    """Alloy's canonical dtype.  One singleton per supported type."""

    __slots__ = ("ir", "msl", "itemsize", "_np_str", "_ctype")

    def __init__(self, ir: str, msl: str, itemsize: int, np_str: str, ctype: type):
        self.ir = ir
        self.msl = msl
        self.itemsize = itemsize
        self._np_str = np_str
        self._ctype = ctype

    def to_numpy(self):
        """Return the numpy dtype.  Only call at interop boundaries."""
        if self._np_str == "bfloat16":
            ml_dtypes = importlib.import_module("ml_dtypes")
            return np.dtype(ml_dtypes.bfloat16)
        return np.dtype(self._np_str)

    def to_torch_dtype(self):
        """Return the torch dtype."""
        import torch  # scoped: optional dep — keeps `alloy-metal` torch-free at import time

        _NP_STR_TO_TORCH_DTYPE: dict[str, torch.dtype] = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "int8": torch.int8,
            "int16": torch.int16,
            "int32": torch.int32,
            "int64": torch.int64,
            "uint8": torch.uint8,
        }
        return _NP_STR_TO_TORCH_DTYPE[self._np_str]

    def __repr__(self) -> str:
        return self.ir

    def __hash__(self) -> int:
        return hash(self.ir)

    def __eq__(self, other) -> bool:
        if isinstance(other, DType):
            return self.ir == other.ir
        return NotImplemented

    def is_float(self) -> bool:
        return self._np_str in ("float32", "float16", "bfloat16")

    def is_integer(self) -> bool:
        return self._np_str in (
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
        )


# ── Singletons ──────────────────────────────────────────────────────

float32 = DType("f32", "float", 4, "float32", _ct.c_float)
float16 = DType("f16", "half", 2, "float16", _ct.c_uint16)
bfloat16 = DType("bf16", "bfloat", 2, "bfloat16", _ct.c_uint16)
int64 = DType("i64", "long", 8, "int64", _ct.c_int64)
int32 = DType("i32", "int", 4, "int32", _ct.c_int32)
int16 = DType("i16", "short", 2, "int16", _ct.c_int16)
int8 = DType("i8", "char", 1, "int8", _ct.c_int8)
uint64 = DType("u64", "ulong", 8, "uint64", _ct.c_uint64)
uint32 = DType("u32", "uint", 4, "uint32", _ct.c_uint32)
uint16 = DType("u16", "ushort", 2, "uint16", _ct.c_uint16)
uint8 = DType("u8", "uchar", 1, "uint8", _ct.c_uint8)

_ALL = [float32, float16, bfloat16, int64, int32, int16, int8, uint64, uint32, uint16, uint8]


# ── Lookup tables ───────────────────────────────────────────────────

# IR name → DType
_BY_IR: dict[str, DType] = {d.ir: d for d in _ALL}

# (numpy kind char, itemsize) → DType
_BY_NUMPY_KIND: dict[tuple[str, int], DType] = {
    ("f", 4): float32,
    ("f", 2): float16,
    ("i", 8): int64,
    ("i", 4): int32,
    ("i", 2): int16,
    ("i", 1): int8,
    ("u", 8): uint64,
    ("u", 4): uint32,
    ("u", 2): uint16,
    ("u", 1): uint8,
    ("b", 1): uint8,
    ("V", 2): bfloat16,  # ml_dtypes bfloat16 shows as void/2
}

# Python/user name → DType
_BY_NAME: dict[str, DType] = {d._np_str: d for d in _ALL}

# IR name → MSL type (dict form for callers that need it)
ALLOY_TO_MSL: dict[str, str] = {d.ir: d.msl for d in _ALL}


# ── Conversion functions ────────────────────────────────────────────


def from_numpy(np_dtype) -> DType:
    """Convert a numpy dtype (or numpy type like np.float32) to DType."""
    d = np.dtype(np_dtype)
    return _BY_NUMPY_KIND[(d.kind, d.itemsize)]


def from_torch_dtype(torch_dtype) -> DType:
    """Convert a torch dtype to DType."""
    import torch  # scoped: optional dep — keeps `alloy-metal` torch-free at import time

    _BY_TORCH_DTYPE: dict[torch.dtype, DType] = {
        torch.float32: float32,
        torch.float16: float16,
        torch.bfloat16: bfloat16,
        torch.int8: int8,
        torch.int16: int16,
        torch.int32: int32,
        torch.int64: int64,
        torch.uint8: uint8,
        torch.uint16: uint16,
        torch.uint32: uint32,
        torch.uint64: uint64,
        torch.bool: int32,
    }

    return _BY_TORCH_DTYPE[torch_dtype]


def from_ir(name: "str | DType") -> DType:
    """Look up DType by IR name ('f32', 'i32', etc.). Passes through DType."""
    if isinstance(name, DType):
        return name
    return _BY_IR[name]


def from_name(name: str) -> DType:
    """Convert a Python name ('float32') or IR name ('f32') to DType."""
    if name in _BY_IR:
        return _BY_IR[name]
    return _BY_NAME.get(name, float32)
