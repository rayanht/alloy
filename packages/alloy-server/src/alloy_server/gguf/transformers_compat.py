"""Generic transformers GGUF-loader patches + the fast in-RAM metadata walk.

Architecture-agnostic loading mechanics: a single-pass mmap walk that replaces
`gguf.GGUFReader`'s per-element parse, a fast
`load_gguf_checkpoint(return_tensors=False)` path, memoization of
`get_tensor_name_map`, and the split-out-text-config → GGUF-arch translation the
weight map needs. They patch transformers at import.

`BYPASS_CONFIG_FIXUPS` lets archs whose config needs a fixup on code paths that
bypass the main loader (the AutoTokenizer build calls `load_gguf_checkpoint`
directly, then feeds the config to `AutoConfig.for_model`) register their fixup.
"""

from __future__ import annotations

import functools
import mmap
import os
import struct
from collections.abc import Callable

import gguf
import numpy as np
from transformers import modeling_gguf_pytorch_utils, tokenization_utils_tokenizers
from transformers.integrations.ggml import GGUF_CONFIG_DEFAULTS_MAPPING
from transformers.modeling_gguf_pytorch_utils import GGUF_TO_TRANSFORMERS_MAPPING
from transformers.models.auto import tokenization_auto

# Arch (GGUF `model_type`) -> config fixup, applied on loader paths that bypass
# the main `load_causal_lm` (e.g. AutoTokenizer). Populated by handlers at
# `apply_transformers_patches`.
BYPASS_CONFIG_FIXUPS: dict[str, Callable[[dict], None]] = {}


# HF split-out text-decoder model_type -> gguf-py MODEL_ARCH name. Upstream's
# get_gguf_hf_weights_map has a hardcoded elif chain missing recent split-out
# text configs (gemma4's decoder is `gemma4_text`, gguf-py knows `gemma4`;
# qwen3.5 is `qwen3_5_text` -> `qwen35`).
GGUF_MODEL_TYPE_TO_ARCH = {
    "qwen3_5_text": "qwen35",
    "qwen3_5_moe_text": "qwen35moe",
    "gemma4_text": "gemma4",
}


# Use the fast in-RAM metadata walk for return_tensors=False loads. Escape hatch:
# set ALLOY_DISABLE_FAST_GGUF_METADATA=1 to force transformers' GGUFReader path
# everywhere (e.g. to A/B a suspected parse divergence).
FAST_GGUF_METADATA = os.environ.get("ALLOY_DISABLE_FAST_GGUF_METADATA") != "1"

# GGUF value type tags (gguf-py GGUFValueType).
GGUF_STR, GGUF_ARR = 8, 9
GGUF_SCALAR_NP = {
    0: np.uint8, 1: np.int8, 2: np.uint16, 3: np.int16, 4: np.uint32,
    5: np.int32, 6: np.float32, 7: np.uint8, 10: np.uint64, 11: np.int64, 12: np.float64,
}
GGUF_SCALAR_SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
GGUF_MAGIC = 0x46554747  # 'GGUF' little-endian


def walk_gguf_metadata(path, *, stop_at=None):
    """Single-pass walk of a GGUF file's metadata over an mmap. Returns
    ``(kv_fields, tensor_names)`` where values are already decoded to the
    Python forms transformers' `_gguf_parse_value` produces (utf-8 strings,
    int/float/bool scalars, lists for arrays).

    This replaces `gguf.GGUFReader`, which eagerly parses every KV field by
    slicing the memmap per array element — for a 248k-token vocab that is
    ~18M numpy memmap `__array_finalize__` calls (tens of seconds). The walk
    here reads each array region in one pass and never re-views per element.
    The mmap means we read no more than the OS pages the metadata touches and
    impose no fixed prefix bound (the KV section can be arbitrarily large).

    ``stop_at`` short-circuits as soon as that KV key is read, returning the
    fields seen so far and an empty tensor-name list — for a single header
    field (e.g. ``general.architecture``, the first KV entry) this skips the
    huge tokenizer arrays that follow entirely.
    """
    with open(path, "rb") as handle:
        mm = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
    try:
        offset = 0
        magic, _ = struct.unpack_from("<II", mm, offset)
        offset += 8
        if magic != GGUF_MAGIC:
            raise ValueError(f"not a GGUF file: magic={magic:#x}")
        n_tensors, n_kv = struct.unpack_from("<QQ", mm, offset)
        offset += 16

        def read_str(off):
            (length,) = struct.unpack_from("<Q", mm, off)
            off += 8
            return mm[off:off + length], off + length

        def read_value(off, vtype):
            if vtype == GGUF_STR:
                raw, off = read_str(off)
                return raw.decode("utf-8"), off
            if vtype == GGUF_ARR:
                (itype,) = struct.unpack_from("<I", mm, off)
                off += 4
                (alen,) = struct.unpack_from("<Q", mm, off)
                off += 8
                if itype == GGUF_STR:
                    vals = [None] * alen
                    for i in range(alen):
                        (slen,) = struct.unpack_from("<Q", mm, off)
                        off += 8
                        vals[i] = mm[off:off + slen].decode("utf-8")
                        off += slen
                    return vals, off
                arr = np.frombuffer(mm, dtype=GGUF_SCALAR_NP[itype], count=alen, offset=off)
                off += GGUF_SCALAR_SZ[itype] * alen
                if itype in (6, 12):
                    return [float(x) for x in arr], off
                if itype == 7:
                    return [bool(x) for x in arr], off
                return [int(x) for x in arr], off
            raw = np.frombuffer(mm, dtype=GGUF_SCALAR_NP[vtype], count=1, offset=off)[0]
            if vtype == 7:
                return bool(raw), off + GGUF_SCALAR_SZ[vtype]
            if vtype in (6, 12):
                return float(raw), off + GGUF_SCALAR_SZ[vtype]
            return int(raw), off + GGUF_SCALAR_SZ[vtype]

        kv_fields = {}
        for _ in range(n_kv):
            key_bytes, offset = read_str(offset)
            (vtype,) = struct.unpack_from("<I", mm, offset)
            offset += 4
            value, offset = read_value(offset, vtype)
            key = key_bytes.decode("utf-8")
            kv_fields[key] = value
            if stop_at is not None and key == stop_at:
                return kv_fields, []

        # Tensor-info records (name, n_dims, dims[], type, data_offset). We
        # only need the names (for the tie_word_embeddings probe below).
        tensor_names = []
        for _ in range(n_tensors):
            name_bytes, offset = read_str(offset)
            tensor_names.append(name_bytes.decode("utf-8"))
            (n_dims,) = struct.unpack_from("<I", mm, offset)
            offset += 4 + 8 * n_dims + 4 + 8  # dims + type(u32) + offset(u64)
        return kv_fields, tensor_names
    finally:
        mm.close()


def arch_needs_original_loader(architecture, model_name):
    """True for architectures whose config `load_gguf_checkpoint` builds with
    logic the fast path does NOT replicate (rope_scaling reconstruction,
    stablelm qkv_bias probing, t5 gated-act, gemma3/minimax/lfm2 model_type
    rewrites, MoE arch renames, mistral-via-llama). For those, fall back to the
    original loader so the model config never silently diverges. The supported
    models (qwen2/qwen3/qwen35, llama, deepseek=qwen2) need none of this."""
    a = architecture
    if "t5" in a or "stablelm" in a or "gpt_oss" in a or "gpt-oss" in a:
        return True
    if "qwen2moe" in a or "qwen3moe" in a or "minimax" in a or "lfm2" in a:
        return True
    if "gemma3" in a:
        return True
    if "llama" in a and model_name and "mistral" in model_name:
        return True
    return False


def fast_load_gguf_checkpoint(gguf_checkpoint_path):
    """Fast equivalent of `load_gguf_checkpoint(path, return_tensors=False)`.

    Reproduces transformers' field-rename loop + the metadata-only fixups
    (tie_word_embeddings, config defaults, vocab_size-from-tokens) exactly,
    using `GGUF_TO_TRANSFORMERS_MAPPING`, so the returned `{config, tokenizer,
    tokenizer_config}` dict is byte-equal to the original's.

    Returns ``None`` for architectures that need the original loader's extra
    config logic (see `arch_needs_original_loader`) — the caller then falls back.
    """
    fields, tensor_names = walk_gguf_metadata(gguf_checkpoint_path)
    architecture = fields["general.architecture"]
    model_name = fields.get("general.name")
    if arch_needs_original_loader(architecture, model_name):
        return None
    updated = architecture
    if "llama" in architecture and model_name and "mistral" in model_name:
        updated = "mistral"
    if "qwen2moe" in architecture:
        updated = "qwen2_moe"
    elif "gpt_oss" in architecture or "gpt-oss" in architecture:
        updated = "gpt_oss"
    elif "qwen3moe" in architecture:
        updated = "qwen3_moe"
    elif "minimax-m2" in architecture:
        updated = "minimax_m2"

    parsed = {key: {} for key in GGUF_TO_TRANSFORMERS_MAPPING}
    parsed["config"]["tie_word_embeddings"] = (
        all(name != "output.weight" for name in tensor_names)
        or architecture in ("falcon", "bloom")
    )
    defaults = GGUF_CONFIG_DEFAULTS_MAPPING.get(
        updated, GGUF_CONFIG_DEFAULTS_MAPPING.get(architecture) or {}
    )
    for key, value in defaults.items():
        parsed["config"].setdefault(key, value)

    for gguf_key, value in fields.items():
        renamed_key = gguf_key.replace(architecture, updated)
        split = renamed_key.split(".")
        prefix, config_key = split[0], ".".join(split[1:])
        if isinstance(value, str) and architecture in value:
            value = value.replace(architecture, updated)
        for parameter, parameter_renames in GGUF_TO_TRANSFORMERS_MAPPING.items():
            if prefix in parameter_renames and config_key in parameter_renames[prefix]:
                dst = parameter_renames[prefix][config_key]
                if dst == -1:
                    continue
                if dst is not None:
                    parsed[parameter][dst] = value

    if "vocab_size" not in parsed["config"]:
        tokens = parsed["tokenizer"].get("tokens")
        if tokens is not None:
            parsed["config"]["vocab_size"] = len(tokens)
    return parsed


ORIGINAL_LOAD_GGUF_CHECKPOINT = modeling_gguf_pytorch_utils.load_gguf_checkpoint


def patched_load_gguf_checkpoint(gguf_checkpoint_path, return_tensors=False, model_to_load=None):
    """Wrap load_gguf_checkpoint so registered arch fixups (`BYPASS_CONFIG_FIXUPS`)
    apply even on code paths that bypass the main loader (e.g. the AutoTokenizer
    path calls load_gguf_checkpoint directly, then feeds the parsed config to
    `AutoConfig.for_model`).

    For metadata-only loads (`return_tensors=False`), parse via the fast in-RAM
    walk instead of the per-element GGUFReader.
    """
    parsed = None
    if not return_tensors and FAST_GGUF_METADATA:
        parsed = fast_load_gguf_checkpoint(gguf_checkpoint_path)
    if parsed is None:
        parsed = ORIGINAL_LOAD_GGUF_CHECKPOINT(
            gguf_checkpoint_path, return_tensors=return_tensors, model_to_load=model_to_load
        )
    config_dict = parsed.get("config") if isinstance(parsed, dict) else None
    if isinstance(config_dict, dict):
        fixup = BYPASS_CONFIG_FIXUPS.get(config_dict.get("model_type"))
        if fixup is not None:
            fixup(config_dict)
    return parsed


ORIGINAL_GET_GGUF_HF_WEIGHTS_MAP = modeling_gguf_pytorch_utils.get_gguf_hf_weights_map


def patched_get_gguf_hf_weights_map(hf_model, processor, model_type=None, num_layers=None, qual_name=""):
    """Wrap get_gguf_hf_weights_map to translate split-out text model_types to
    the GGUF arch name gguf-py's MODEL_ARCH_NAMES expects."""
    effective_model_type = (
        hf_model.config.model_type if model_type is None else model_type
    )
    translated = GGUF_MODEL_TYPE_TO_ARCH.get(effective_model_type)
    if translated is not None:
        return ORIGINAL_GET_GGUF_HF_WEIGHTS_MAP(
            hf_model, processor,
            model_type=translated, num_layers=num_layers, qual_name=qual_name,
        )
    return ORIGINAL_GET_GGUF_HF_WEIGHTS_MAP(
        hf_model, processor,
        model_type=model_type, num_layers=num_layers, qual_name=qual_name,
    )


# Memoize gguf.get_tensor_name_map. `get_gguf_hf_weights_map` rebuilds the name
# map once per module as it recurses the model tree (~535x for a 32-layer model),
# and each build constructs+deep-copies a large mapping config (~3.4s of a cold
# load). The map depends only on (arch, n_blocks) and is used read-only, so one
# shared instance per key is safe.
ORIGINAL_GET_TENSOR_NAME_MAP = gguf.get_tensor_name_map


@functools.lru_cache(maxsize=None)
def memoized_get_tensor_name_map(arch, n_blocks):
    return ORIGINAL_GET_TENSOR_NAME_MAP(arch, n_blocks)


def install_transformers_patches() -> None:
    """Install the generic loader patches. Idempotent (re-binding the same
    wrappers is a no-op); called at gguf-package import."""
    modeling_gguf_pytorch_utils.get_gguf_hf_weights_map = patched_get_gguf_hf_weights_map
    modeling_gguf_pytorch_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint
    # Several modules rebind load_gguf_checkpoint via `from ... import`, so patch
    # each host module's symbol too — otherwise that import site keeps calling the
    # slow original. `tokenization_auto` drives class/config resolution;
    # `tokenization_utils_tokenizers` (TokenizersBackend) builds the fast tokenizer
    # from a gguf_file and is the hottest of the three.
    tokenization_auto.load_gguf_checkpoint = patched_load_gguf_checkpoint
    tokenization_utils_tokenizers.load_gguf_checkpoint = patched_load_gguf_checkpoint
    gguf.get_tensor_name_map = memoized_get_tensor_name_map


install_transformers_patches()
