"""MLX causal-LM load orchestrator.

Builds the HF skeleton from config.json, swaps quantized Linear/Embedding for the
affine-int4 modules (`mlx.quant`), loads the dense tensors, ties the head, and
attaches the tokenizer.
"""

from __future__ import annotations

import time
from contextlib import ExitStack
from typing import cast

import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.initialization import no_init_weights
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from alloy import get_logger
from alloy_server.gguf.loader import CausalLMHooks, GGUFLoadReport, LoadedGGUFCausalLM
from alloy_server.mlx.quant import (
    MLXQ4Embedding,
    MLXQ4Linear,
    mlx_qweight_to_uint8,
    replace_mlx_quantized_weight,
    tie_mlx_output_embeddings,
)
from alloy_server.mlx.resolve import ResolvedMLX, mlx_quantization

logger = get_logger("alloy_server.mlx")

WEIGHT_SUFFIX = ".weight"


def build_mlx_config(resolved: ResolvedMLX):
    config_dict = dict(resolved.config)
    config_dict.pop("quantization", None)
    config_dict.pop("quantization_config", None)
    model_type = config_dict.pop("model_type")
    if not isinstance(model_type, str):
        raise TypeError(f"MLX config model_type must be a string, got {model_type!r}")
    config = AutoConfig.for_model(model_type, **config_dict)
    if hasattr(config, "text_config"):
        config = AutoConfig.for_model(config.text_config.model_type, **config_dict)
    return config


def load_mlx_causal_lm(
    resolved: ResolvedMLX,
    hooks: CausalLMHooks,
    *,
    dtype: torch.dtype | None = None,
    load_tokenizer: bool = True,
) -> LoadedGGUFCausalLM:
    load_t0 = time.perf_counter()
    quant = mlx_quantization(resolved.config)
    if quant is None:
        raise ValueError(f"{resolved.ref}: not an MLX-quantized model")
    if int(quant.get("bits", 0)) != 4:
        raise NotImplementedError(f"only MLX 4-bit is supported, got {quant}")

    config = build_mlx_config(resolved)
    with no_init_weights(), init_empty_weights(include_buffers=False):
        model = cast(PreTrainedModel, AutoModelForCausalLM.from_config(config))

    dense_dtype = dtype if dtype is not None else torch.float32
    param_shapes = {name: tuple(p.shape) for name, p in model.named_parameters()}
    state_dict: dict[str, torch.Tensor] = {}
    quantized_linear = 0
    quantized_embedding = 0
    dense_count = 0

    with ExitStack() as stack:
        key_index = {}
        for shard in resolved.safetensors:
            handle = stack.enter_context(safe_open(str(shard), framework="pt"))
            for key in handle.keys():
                key_index[key] = handle
        quant_bases = {
            key[: -len(WEIGHT_SUFFIX)]
            for key in key_index
            if key.endswith(WEIGHT_SUFFIX) and (key[: -len(WEIGHT_SUFFIX)] + ".scales") in key_index
        }
        for key, handle in key_index.items():
            if key.endswith((".scales", ".biases")):
                continue
            base = key[: -len(WEIGHT_SUFFIX)] if key.endswith(WEIGHT_SUFFIX) else None
            if base is not None and base in quant_bases:
                qweight = mlx_qweight_to_uint8(handle.get_tensor(key))
                scales = key_index[base + ".scales"].get_tensor(base + ".scales").to(torch.float16)
                biases = key_index[base + ".biases"].get_tensor(base + ".biases").to(torch.float16)
                kind = replace_mlx_quantized_weight(model, key, qweight, scales, biases)
                if kind == "linear":
                    quantized_linear += 1
                else:
                    quantized_embedding += 1
            else:
                tensor = handle.get_tensor(key)
                if tensor.is_floating_point():
                    tensor = tensor.to(dense_dtype)
                # MLX stores conv1d weights with the kernel axis last; HF wants it second.
                want = param_shapes.get(key)
                if want is not None and tuple(tensor.shape) != want and tensor.ndim >= 2 \
                        and tuple(tensor.transpose(-1, -2).shape) == want:
                    tensor = tensor.transpose(-1, -2).contiguous()
                state_dict[key] = tensor
                dense_count += 1

    hooks.post_load(model, [], model.config)
    incompatible = model.load_state_dict(state_dict, strict=False, assign=True)
    if bool(config.tie_word_embeddings):
        tie_mlx_output_embeddings(model)
    model.eval()

    allowed_missing = mlx_allowed_missing_keys(model) | hooks.allowed_missing_keys(model)
    missing_keys = tuple(k for k in incompatible.missing_keys if k not in allowed_missing)
    unexpected_keys = tuple(incompatible.unexpected_keys)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"MLX load produced incompatible model state: missing={missing_keys} "
            f"unexpected={unexpected_keys}"
        )

    tokenizer: PreTrainedTokenizerBase | None = None
    if load_tokenizer:
        tokenizer = cast(
            PreTrainedTokenizerBase,
            AutoTokenizer.from_pretrained(str(resolved.model_dir), local_files_only=True),
        )

    report = GGUFLoadReport(
        source=cast("object", resolved),  # ResolvedMLX; downstream reads only .ref
        quantized_linear_count=quantized_linear,
        quantized_linear_counts=(("mlx_q4", quantized_linear),) if quantized_linear else (),
        quantized_embedding_count=quantized_embedding,
        quantized_embedding_counts=(("mlx_q4", quantized_embedding),) if quantized_embedding else (),
        dense_tensor_count=dense_count,
        missing_keys=tuple(incompatible.missing_keys),
        unexpected_keys=unexpected_keys,
    )
    cfg = model.config
    logger.info(
        "load_complete",
        model_name=resolved.ref,
        took_s=round(time.perf_counter() - load_t0, 2),
        n_params=sum(p.numel() for p in model.parameters()),
        architecture=type(model).__name__,
        n_layers=int(cfg.num_hidden_layers),
        vocab_size=int(cfg.vocab_size),
        quant_linear=quantized_linear,
        quant_embedding=quantized_embedding,
        n_dense_tensors=dense_count,
        tokenizer_loaded=tokenizer is not None,
    )
    return LoadedGGUFCausalLM(model=model, tokenizer=tokenizer, report=report)


def mlx_allowed_missing_keys(model: PreTrainedModel) -> set[str]:
    """Quantized-buffer + tied-head keys that legitimately aren't in the dense
    state dict."""
    allowed: set[str] = set()
    for name, module in model.named_modules():
        if isinstance(module, (MLXQ4Linear, MLXQ4Embedding)):
            allowed.update(f"{name}.{buf}" for buf in ("qweight", "scales", "biases"))
    if bool(model.config.tie_word_embeddings):
        allowed.add("lm_head.weight")
    return allowed
