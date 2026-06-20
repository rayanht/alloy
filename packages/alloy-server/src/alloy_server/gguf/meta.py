"""Parsed-GGUF metadata cache.

`build_gguf_meta` runs HF's `load_gguf_checkpoint` + `gguf.GGUFReader` once and
extracts a small sidecar (parsed config dict + per-tensor name/type/shape/offset)
so subsequent loads rebuild ReaderTensor-like views from a plain mmap. The
arch-specific config fixups (qwen35 / qwen35moe / gemma4) run here.
"""

from __future__ import annotations

import pickle
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import gguf
import numpy as np
from gguf.quants import quant_shape_to_byte_shape
from transformers.modeling_gguf_pytorch_utils import load_gguf_checkpoint, read_field

GGUFConfigValue = str | int | float | bool | list[int] | list[float] | list[str] | None
GGUFConfig = dict[str, GGUFConfigValue]
ParsedGGUF = dict[str, GGUFConfig]


# Bump when the cached on-disk layout changes (new fields, different
# pickle protocol, different field semantics). Old files are ignored.
GGUF_META_CACHE_VERSION = 4


@dataclass(frozen=True, slots=True)
class CachedTensorMeta:
    """Sidecar tensor metadata. Pickled; tensor data never enters the cache —
    only its byte offset into the GGUF blob, which we mmap at load time."""

    name: str
    tensor_type_value: int  # gguf.GGMLQuantizationType.value
    dims: tuple[int, ...]
    n_elements: int
    n_bytes: int
    data_offset: int


@dataclass(slots=True)
class LiveTensor:
    """Runtime stand-in for gguf.ReaderTensor. Same attrs the downstream
    helpers (tensor_quantization, processed_quantized_weight,
    dense_tensor_weights) actually touch: .name, .tensor_type, .data."""

    name: str
    tensor_type: gguf.GGMLQuantizationType
    data: np.ndarray


def gguf_meta_cache_path(digest: str) -> Path:
    return Path.home() / ".cache" / "alloy" / "gguf-meta" / f"v{GGUF_META_CACHE_VERSION}_{digest}.pkl"


def build_gguf_meta(
    blob_path: Path,
    config_fixup: Callable[[dict, gguf.GGUFReader], None],
) -> tuple[ParsedGGUF, str, list[CachedTensorMeta]]:
    """Slow path: run HF's load_gguf_checkpoint + gguf.GGUFReader once and
    extract the metadata we need to rebuild ReaderTensor-like views from a
    plain mmap on subsequent loads. `config_fixup` is the handler's arch-specific
    config transform (e.g. resolving per-layer GGUF arrays); the fixed config is
    what gets cached, so this runs only on a cache miss."""
    parsed = cast(ParsedGGUF, load_gguf_checkpoint(str(blob_path), return_tensors=False))
    reader = gguf.GGUFReader(str(blob_path))
    architecture = read_required_string(reader, "general.architecture")
    config_fixup(parsed["config"], reader)
    # Cache the vision sub-config scalars (`<arch>.vision.*`) — the flat text-config
    # parse drops them, but the vision adapter needs them to size its ViT. Plain
    # scalars only (pickle-safe); the vision tensor data comes from the mmap views.
    parsed["vision_metadata"] = {
        name: field.contents()
        for name, field in reader.fields.items()
        if ".vision." in name
    }
    # Same for the audio sub-config scalars (`<arch>.audio.*`) — the conformer
    # adapter needs them to size its encoder.
    parsed["audio_metadata"] = {
        name: field.contents()
        for name, field in reader.fields.items()
        if ".audio." in name
    }
    tensors_meta = [
        CachedTensorMeta(
            name=t.name,
            tensor_type_value=int(t.tensor_type),
            dims=tuple(int(d) for d in t.shape.tolist()),
            n_elements=int(t.n_elements),
            n_bytes=int(t.n_bytes),
            data_offset=int(t.data_offset),
        )
        for t in reader.tensors
    ]
    return parsed, architecture, tensors_meta


def load_or_build_gguf_meta(
    blob_path: Path, digest: str, config_fixup: Callable[[dict, gguf.GGUFReader], None]
) -> tuple[ParsedGGUF, str, list[CachedTensorMeta]]:
    """Return cached metadata if present, else parse the GGUF file once and
    persist a small sidecar (~MB, contains offsets only — no tensor data).
    `config_fixup` (the handler's arch transform) runs only on a cache miss."""
    cache_path = gguf_meta_cache_path(digest)
    if cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                payload = pickle.load(f)
            return payload["parsed"], payload["architecture"], payload["tensors"]
        except (pickle.UnpicklingError, KeyError, EOFError, ImportError, AttributeError):
            # Corrupt, or written by code whose pickled classes have since moved
            # (e.g. a package rename) — rebuild from the GGUF.
            cache_path.unlink(missing_ok=True)
    parsed, architecture, tensors_meta = build_gguf_meta(blob_path, config_fixup)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        pickle.dump(
            {"parsed": parsed, "architecture": architecture, "tensors": tensors_meta},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    tmp_path.replace(cache_path)
    return parsed, architecture, tensors_meta


def live_tensors_from_meta(
    mm: np.memmap, tensors_meta: list[CachedTensorMeta]
) -> list[LiveTensor]:
    """Rebuild ReaderTensor.data views from raw mmap bytes. Mirrors
    gguf.GGUFReader._build_tensors (dtype/shape table for dense + quantized
    types) without re-parsing tens of thousands of field-info bytes."""
    live: list[LiveTensor] = []
    for tm in tensors_meta:
        ggml_type = gguf.GGMLQuantizationType(tm.tensor_type_value)
        # gguf stores dims reversed-from-numpy: see GGUFReader._build_tensors.
        np_dims: tuple[int, ...] = tuple(reversed(tm.dims))
        if ggml_type == gguf.GGMLQuantizationType.F16:
            item_type: np.dtype = np.dtype(np.float16)
            item_count = tm.n_elements
        elif ggml_type == gguf.GGMLQuantizationType.BF16:
            # Raw bf16 bits; converted to f32 in `dense_tensor_weights`. Two
            # bytes/element with the plain logical shape (np_dims already set).
            item_type = np.dtype(np.uint16)
            item_count = tm.n_elements
        elif ggml_type == gguf.GGMLQuantizationType.F32:
            item_type = np.dtype(np.float32)
            item_count = tm.n_elements
        elif ggml_type == gguf.GGMLQuantizationType.F64:
            item_type = np.dtype(np.float64)
            item_count = tm.n_elements
        elif ggml_type == gguf.GGMLQuantizationType.I8:
            item_type = np.dtype(np.int8)
            item_count = tm.n_elements
        elif ggml_type == gguf.GGMLQuantizationType.I16:
            item_type = np.dtype(np.int16)
            item_count = tm.n_elements
        elif ggml_type == gguf.GGMLQuantizationType.I32:
            item_type = np.dtype(np.int32)
            item_count = tm.n_elements
        elif ggml_type == gguf.GGMLQuantizationType.I64:
            item_type = np.dtype(np.int64)
            item_count = tm.n_elements
        else:
            item_type = np.dtype(np.uint8)
            item_count = tm.n_bytes
            np_dims = quant_shape_to_byte_shape(np_dims, ggml_type)
        end = tm.data_offset + item_count * item_type.itemsize
        view = mm[tm.data_offset:end].view(item_type)[:item_count].reshape(np_dims)
        live.append(LiveTensor(name=tm.name, tensor_type=ggml_type, data=view))
    return live


def read_required_string(reader: gguf.GGUFReader, key: str) -> str:
    values = read_field(reader, key)
    if not values:
        raise KeyError(f"GGUF metadata key missing: {key}")
    if len(values) != 1 or not isinstance(values[0], str):
        raise TypeError(f"GGUF metadata key {key} must contain exactly one string")
    return values[0]
