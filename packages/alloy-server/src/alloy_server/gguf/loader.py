"""GGUF causal-LM load orchestrator + the Qwen3.5-MoE expert install.

`load_causal_lm(source, hooks)` walks the GGUF tensor table, swaps quantized
Linear/Embedding modules in (`gguf.quant`), attaches the tokenizer
(`gguf.tokenizer`), and runs the arch-specific steps through the `CausalLMHooks`
the caller injects (the `models/` ModelHandler) — config fixup, tensor-map fixup,
expert install (`post_load`), chat template, vision/audio adapters,
allowed-missing-keys. No arch dispatch lives here; the loader is pure mechanics.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol, cast

import gguf
import numpy as np
import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.initialization import no_init_weights
from transformers.modeling_gguf_pytorch_utils import (
    TENSOR_PROCESSORS,
    GGUFTensor,
    TensorProcessor,
    get_gguf_hf_weights_map,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from alloy import get_logger
from alloy_server.gguf.meta import ParsedGGUF, live_tensors_from_meta, load_or_build_gguf_meta
from alloy_server.gguf.quant import (
    GGUFQuantEmbedding,
    GGUFQuantLinear,
    GGUFQuantization,
    dense_tensor_weights,
    quant_buffer_names,
    replace_quantized_weight,
    tensor_quantization,
    tie_quantized_output_embeddings,
    torch_tensor_from_numpy,
)
from alloy_server.gguf.resolve import ResolvedGGUF
from alloy_server.gguf.tokenizer import build_tokenizer

logger = get_logger("alloy_server.gguf")


class CausalLMHooks(Protocol):
    """The arch-specific steps the loader delegates. Implemented by the `models/`
    ModelHandler and injected into `load_causal_lm`; the loader itself has no arch
    dispatch. `config_fixup` runs at metadata-build time (it needs the GGUFReader
    and its result is cached); the rest run per load."""

    def config_fixup(self, config_dict: dict, reader: gguf.GGUFReader) -> None: ...

    def fixup_tensor_map(self, tensor_key_mapping: dict, tensor_names) -> None: ...

    def post_load(self, model: torch.nn.Module, tensors: list, config) -> None: ...

    def chat_template(self) -> str | None: ...

    def build_vision(self, tensors: list, vision_meta: dict, model, tokenizer) -> object | None: ...

    def build_audio(self, tensors: list, audio_meta: dict, model) -> object | None: ...

    def allowed_missing_keys(self, model) -> set[str]: ...


@dataclass(frozen=True, slots=True)
class GGUFLoadReport:
    source: ResolvedGGUF
    quantized_linear_count: int
    quantized_linear_counts: tuple[tuple[GGUFQuantization, int], ...]
    quantized_embedding_count: int
    quantized_embedding_counts: tuple[tuple[GGUFQuantization, int], ...]
    dense_tensor_count: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LoadedGGUFCausalLM:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase | None
    report: GGUFLoadReport
    # The dense vision front-end for multimodal GGUFs (gemma4), else None. Built by
    # the handler's `build_vision` hook so the serving layer never special-cases a
    # model. Typed `object` because `ModalityEncoder` lives in `models/` (which
    # this package must not import); it is a `models.modality.ModalityEncoder`.
    vision: object | None = None
    # The dense audio front-end (gemma4 Conformer), else None — same protocol shape.
    audio: object | None = None


def load_gguf_causal_lm(
    source: ResolvedGGUF,
    hooks: CausalLMHooks,
    *,
    dtype: torch.dtype | None = None,
    load_tokenizer: bool = True,
    hf_tokenizer_id: str | None = None,
) -> LoadedGGUFCausalLM:
    load_t0 = time.perf_counter()
    # AutoTokenizer.from_pretrained does its own (slow) GGUF parse that's
    # independent of weight loading. Kick it off on a worker thread up front so it
    # overlaps with the weight pass instead of running serially after.
    tokenizer_pool: ThreadPoolExecutor | None = None
    tokenizer_future = None
    if load_tokenizer:
        tokenizer_pool = ThreadPoolExecutor(max_workers=1)
        tokenizer_future = tokenizer_pool.submit(
            build_tokenizer, source.path, hf_tokenizer_id
        )

    # GGUF metadata cache: parsed config dict + per-tensor (name, type, shape,
    # byte-offset), keyed by the file's content digest. `hooks.config_fixup` runs
    # only on a cache miss (the fixed config is what gets cached).
    parsed, architecture, tensors_meta = load_or_build_gguf_meta(
        source.path, source.digest, hooks.config_fixup
    )
    config_dict = dict(parsed["config"])
    model_type_value = config_dict.pop("model_type")
    if not isinstance(model_type_value, str):
        raise TypeError(f"GGUF config model_type must be a string, got {model_type_value!r}")

    config = AutoConfig.for_model(model_type_value, **config_dict)
    # Multimodal GGUFs (gemma4: text + vision + audio) map to a COMPOSITE config
    # (text_config + vision_config + audio_config), but the GGUF metadata is a FLAT
    # dict describing only the text decoder. Rebuild the decoder directly from the
    # flat dict via the text sub-config's own model_type so every GGUF dim lands on
    # the text config. Text-only GGUFs parse to a flat config with no text_config,
    # so this is a no-op for them.
    if hasattr(config, "text_config"):
        text_model_type = config.text_config.model_type
        config = AutoConfig.for_model(text_model_type, **config_dict)
    # GGUF tensors overwrite every weight a moment later, so allocating the
    # full-precision skeleton is pure waste — and at scale catastrophic (a 35B
    # f32/f16 skeleton is 70-140 GB held until each tensor is replaced).
    # `init_empty_weights(include_buffers=False)` puts every PARAMETER on meta
    # (no storage) while leaving BUFFERS real and computed — so rotary `inv_freq`
    # and friends, which the GGUF doesn't carry, are still correct. The quant
    # replacement loop + `load_state_dict(assign=True)` attach the real storage.
    with no_init_weights(), init_empty_weights(include_buffers=False):
        model = cast(PreTrainedModel, AutoModelForCausalLM.from_config(config))

    processor_class = cast(type[TensorProcessor], TENSOR_PROCESSORS.get(architecture, TensorProcessor))
    processor = processor_class(config=parsed["config"])
    tensor_key_mapping = cast(dict[str, str], get_gguf_hf_weights_map(model, processor))
    hooks.fixup_tensor_map(tensor_key_mapping, [tm.name for tm in tensors_meta])

    blob_mm = cast(np.memmap, np.memmap(str(source.path), dtype=np.uint8, mode="r"))
    tensors = live_tensors_from_meta(blob_mm, tensors_meta)
    state_dict: dict[str, torch.Tensor] = {}
    quantized_linear_counts: Counter[GGUFQuantization] = Counter()
    quantized_embedding_counts: Counter[GGUFQuantization] = Counter()
    quantized_linear_count = 0
    quantized_embedding_count = 0
    dense_tensor_count = 0
    for tensor in tensors:
        if tensor.name not in tensor_key_mapping:
            continue
        quantization = tensor_quantization(tensor)
        if quantization is not None:
            packed = processed_quantized_weight(processor, tensor, tensor_key_mapping, parsed)
            mapped_name = tensor_key_mapping[packed.name]
            replaced = replace_quantized_weight(model, mapped_name, packed.weights, quantization)
            if replaced == "linear":
                quantized_linear_counts[quantization] += 1
                quantized_linear_count += 1
            else:
                quantized_embedding_counts[quantization] += 1
                quantized_embedding_count += 1
            continue

        weights = dense_tensor_weights(tensor)
        processed = processor.process(
            weights=weights,
            name=tensor.name,
            tensor_key_mapping=tensor_key_mapping,
            parsed_parameters=parsed,
        )
        if processed.name not in tensor_key_mapping:
            continue
        mapped_name = tensor_key_mapping[processed.name]
        state_dict[mapped_name] = torch_tensor_from_numpy(processed.weights, dtype=dtype)
        dense_tensor_count += 1

    # Arch-specific post-load step (e.g. Qwen3.5-MoE installs its quantized fused
    # expert containers here, before load_state_dict, so the dense expert params
    # are replaced — not reported missing).
    hooks.post_load(model, tensors, model.config)

    incompatible = model.load_state_dict(state_dict, strict=False, assign=True)
    if bool(config.tie_word_embeddings):
        tie_quantized_output_embeddings(model)
    model.eval()

    tokenizer: PreTrainedTokenizerBase | None = None
    if tokenizer_future is not None:
        tokenizer = tokenizer_future.result()
    if tokenizer_pool is not None:
        tokenizer_pool.shutdown(wait=False)

    # Some GGUFs carry no chat_template (gemma4: ollama renders it with a built-in
    # Go renderer). The handler supplies the bundled template; attach it only if
    # the tokenizer didn't already carry one.
    chat_template = hooks.chat_template()
    if chat_template is not None and tokenizer is not None and not tokenizer.chat_template:
        tokenizer.chat_template = chat_template

    # Multimodal GGUFs carry dense vision/audio towers alongside the text decoder.
    # The handler builds the adapters (returns None for text-only / non-mm archs)
    # so the serving layer consumes them generically.
    vision = hooks.build_vision(tensors, parsed.get("vision_metadata", {}), model, tokenizer)
    audio = hooks.build_audio(tensors, parsed.get("audio_metadata", {}), model)

    allowed_missing = allowed_missing_keys(model) | hooks.allowed_missing_keys(model)
    missing_keys = tuple(key for key in incompatible.missing_keys if key not in allowed_missing)
    unexpected_keys = tuple(incompatible.unexpected_keys)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "GGUF load produced incompatible model state: "
            f"missing={missing_keys} unexpected={unexpected_keys}"
        )

    report = GGUFLoadReport(
        source=source,
        quantized_linear_count=quantized_linear_count,
        quantized_linear_counts=tuple(sorted(quantized_linear_counts.items())),
        quantized_embedding_count=quantized_embedding_count,
        quantized_embedding_counts=tuple(sorted(quantized_embedding_counts.items())),
        dense_tensor_count=dense_tensor_count,
        missing_keys=tuple(incompatible.missing_keys),
        unexpected_keys=unexpected_keys,
    )
    cfg = model.config
    logger.info(
        "load_complete",
        path=str(source.path),
        model_name=source.ref,
        took_s=round(time.perf_counter() - load_t0, 2),
        n_params=sum(p.numel() for p in model.parameters()),
        architecture=type(model).__name__,
        n_layers=int(cfg.num_hidden_layers),
        vocab_size=int(cfg.vocab_size),
        hidden_size=int(cfg.hidden_size),
        quant_linear=dict(report.quantized_linear_counts),
        quant_embedding=dict(report.quantized_embedding_counts),
        n_dense_tensors=report.dense_tensor_count,
        tokenizer_loaded=tokenizer is not None,
    )
    return LoadedGGUFCausalLM(
        model=model, tokenizer=tokenizer, report=report, vision=vision, audio=audio
    )


def processed_quantized_weight(
    processor: TensorProcessor,
    tensor: gguf.ReaderTensor,
    tensor_key_mapping: Mapping[str, str],
    parsed: ParsedGGUF,
) -> GGUFTensor:
    processed = processor.process(
        weights=tensor.data,
        name=tensor.name,
        tensor_key_mapping=tensor_key_mapping,
        parsed_parameters=parsed,
    )
    if processed.name is None:
        raise RuntimeError(f"GGUF processor dropped quantized tensor {tensor.name}")
    if processed.name not in tensor_key_mapping:
        raise RuntimeError(f"GGUF processor produced unmapped tensor {processed.name}")
    return processed


def allowed_missing_keys(model: PreTrainedModel) -> set[str]:
    """The quantized-buffer + tied-head keys that legitimately won't appear in the
    GGUF state dict. Arch-specific extras (e.g. MoE expert buffers) are added by
    the handler's `allowed_missing_keys` hook."""
    allowed: set[str] = set()
    for name, module in model.named_modules():
        if isinstance(module, (GGUFQuantLinear, GGUFQuantEmbedding)):
            allowed.update(f"{name}.{buf}" for buf in quant_buffer_names(module))
    config = model.config
    if bool(config.tie_word_embeddings):
        allowed.add("lm_head.weight")
    return allowed
