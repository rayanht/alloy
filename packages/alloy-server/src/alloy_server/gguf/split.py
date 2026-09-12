"""Split-GGUF index: one metadata table + a tensor location table spanning shards.

llama.cpp splits (`name-00001-of-000NN.gguf`) put the model metadata in the first
shard and each tensor in exactly one shard, with per-shard headers. This reads every
shard header once (numpy memmaps, no tensor data touched) and exposes tensors by
name as (path, offset, nbytes, type, shape) — the addressing a streaming weight store
needs — plus zero-copy memmap views for tensors that are loaded resident.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import gguf
import numpy as np

SPLIT_RE = re.compile(r"^(?P<prefix>.*)-(?P<no>\d{5})-of-(?P<count>\d{5})\.gguf$")


@dataclass(frozen=True, slots=True)
class TensorLocation:
    name: str
    path: Path
    data_offset: int
    n_bytes: int
    tensor_type: gguf.GGMLQuantizationType
    shape: tuple[int, ...]  # torch order (reversed GGUF dims)
    n_elements: int


def shard_paths(first: Path) -> list[Path]:
    """Every shard of the split `first` belongs to (or just `[first]`)."""
    m = SPLIT_RE.match(first.name)
    if m is None:
        return [first]
    count = int(m.group("count"))
    paths = [first.with_name(f"{m.group('prefix')}-{i:05d}-of-{count:05d}.gguf") for i in range(1, count + 1)]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"split GGUF is missing shards: {missing}")
    return paths


def field_value(field: gguf.ReaderField):
    """A ReaderField's python value (scalar, string, or list)."""
    return field.contents()


class SplitGGUF:
    """Metadata + tensor index over all shards; readers are kept for memmap views."""

    def __init__(self, first: Path) -> None:
        self.paths = shard_paths(Path(first))
        self.readers: list[gguf.GGUFReader] = []
        self.tensors: dict[str, TensorLocation] = {}
        self.kv: dict = {}
        for i, path in enumerate(self.paths):
            reader = gguf.GGUFReader(str(path))
            self.readers.append(reader)
            if i == 0:
                self.kv = {name: field_value(field) for name, field in reader.fields.items()}
            for t in reader.tensors:
                if t.name in self.tensors:
                    raise ValueError(f"tensor {t.name} appears in more than one shard")
                self.tensors[t.name] = TensorLocation(
                    name=t.name,
                    path=path,
                    data_offset=int(t.data_offset),
                    n_bytes=int(t.n_bytes),
                    tensor_type=gguf.GGMLQuantizationType(int(t.tensor_type)),
                    shape=tuple(reversed(tuple(int(d) for d in t.shape.tolist()))),
                    n_elements=int(t.n_elements),
                )
        expected = self.kv.get("split.tensors.count")
        if expected is not None and int(expected) != len(self.tensors):
            raise ValueError(f"split declares {expected} tensors, shards carry {len(self.tensors)}")

    @property
    def architecture(self) -> str:
        return str(self.kv["general.architecture"])

    def shapes(self) -> dict[str, tuple[int, ...]]:
        return {name: loc.shape for name, loc in self.tensors.items()}

    def raw(self, name: str) -> np.ndarray:
        """The tensor's packed bytes as a memmap view (uint8, [n_bytes])."""
        loc = self.tensors[name]
        reader = self.readers[self.paths.index(loc.path)]
        mm = reader.data
        return mm[loc.data_offset : loc.data_offset + loc.n_bytes]

    def dequantized(self, name: str) -> np.ndarray:
        """The tensor as fp32 in torch shape (dense types are cast, quantized types
        dequantized through gguf-py). For resident, non-hot-path tensors."""
        loc = self.tensors[name]
        data = self.raw(name)
        qtype = loc.tensor_type
        if qtype == gguf.GGMLQuantizationType.F32:
            out = data.view(np.float32)
        elif qtype == gguf.GGMLQuantizationType.F16:
            out = data.view(np.float16).astype(np.float32)
        elif qtype == gguf.GGMLQuantizationType.BF16:
            out = (data.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        else:
            block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
            rows = loc.n_elements // loc.shape[-1]
            blocks = data.reshape(rows, loc.shape[-1] // block_size * type_size)
            out = gguf.quants.dequantize(blocks, qtype)
        return np.ascontiguousarray(out.reshape(loc.shape))

    def packed_rows(self, name: str) -> np.ndarray:
        """A quantized tensor's native blocks as [rows..., row_bytes] uint8 (the layout
        the alloy quant kernels index)."""
        loc = self.tensors[name]
        qtype = loc.tensor_type
        block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
        row_bytes = loc.shape[-1] // block_size * type_size
        return self.raw(name).reshape(*loc.shape[:-1], row_bytes)
