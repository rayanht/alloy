"""Load a Whisper GGUF whose tensors carry HF-native names (e.g.
oxide-lab/whisper-tiny-GGUF) into a transformers WhisperForConditionalGeneration.

The config is derived from tensor shapes (Whisper always has head_dim 64, so
heads = d_model // 64); the tokenizer / feature extractor / generation config are
NOT in the GGUF and come from the canonical openai/whisper-* tokenizer.

This module is the eager reference path: Q4_0 weights are dequantized to a dense
dtype here. The alloy-compiled + native-q4_0 paths build on this loader.
"""

from __future__ import annotations

import numpy as np
import torch
from gguf import GGUFReader
from gguf.constants import GGMLQuantizationType
from transformers import WhisperConfig, WhisperForConditionalGeneration

WHISPER_HEAD_DIM = 64  # every Whisper size uses head_dim 64; heads = d_model // 64


def torch_shape(reader_shape) -> tuple[int, ...]:
    """GGUF stores dims reversed vs torch (ne[0] = innermost/contiguous)."""
    return tuple(int(x) for x in reversed(list(reader_shape)))


def raw_bytes(data: np.ndarray, n_blocks: int, block_bytes: int) -> np.ndarray:
    raw = np.ascontiguousarray(data).reshape(-1).view(np.uint8)
    assert raw.size == n_blocks * block_bytes, f"byte size {raw.size} != {n_blocks * block_bytes}"
    return raw.reshape(n_blocks, block_bytes)


def dequant_q4_0(data: np.ndarray, n_elements: int) -> np.ndarray:
    """Q4_0 block = 18 bytes: fp16 scale d, then 32 4-bit nibbles.
    value[j] = d*((qs[j]&0xF)-8) for j<16; value[j+16] = d*((qs[j]>>4)-8)."""
    assert n_elements % 32 == 0
    blocks = raw_bytes(data, n_elements // 32, 18)
    scale = blocks[:, 0:2].copy().view(np.float16).astype(np.float32)  # (nb, 1)
    qs = blocks[:, 2:18].astype(np.int16)
    lo, hi = (qs & 0x0F) - 8, (qs >> 4) - 8
    vals = np.concatenate([lo, hi], axis=1).astype(np.float32)
    return (vals * scale).reshape(n_elements)


def dequant_q4_1(data: np.ndarray, n_elements: int) -> np.ndarray:
    """Q4_1 block = 20 bytes: fp16 scale d, fp16 min m, then 32 4-bit nibbles.
    value[j] = d*(qs[j]&0xF)+m for j<16; value[j+16] = d*(qs[j]>>4)+m."""
    assert n_elements % 32 == 0
    blocks = raw_bytes(data, n_elements // 32, 20)
    scale = blocks[:, 0:2].copy().view(np.float16).astype(np.float32)  # (nb, 1)
    minv = blocks[:, 2:4].copy().view(np.float16).astype(np.float32)   # (nb, 1)
    qs = blocks[:, 4:20].astype(np.int16)
    lo, hi = qs & 0x0F, qs >> 4
    vals = np.concatenate([lo, hi], axis=1).astype(np.float32)
    return (vals * scale + minv).reshape(n_elements)


def dequant_q8_0(data: np.ndarray, n_elements: int) -> np.ndarray:
    """Q8_0 block = 34 bytes: fp16 scale d, then 32 int8 values. value[i] = d*q[i]."""
    assert n_elements % 32 == 0
    blocks = raw_bytes(data, n_elements // 32, 34)
    scale = blocks[:, 0:2].copy().view(np.float16).astype(np.float32)  # (nb, 1)
    q = blocks[:, 2:34].copy().view(np.int8).astype(np.float32)        # (nb, 32)
    return (q * scale).reshape(n_elements)


DEQUANT = {
    int(GGMLQuantizationType.Q4_0): dequant_q4_0,
    int(GGMLQuantizationType.Q4_1): dequant_q4_1,
    int(GGMLQuantizationType.Q8_0): dequant_q8_0,
}


def tensor_to_torch(tensor, dtype: torch.dtype) -> torch.Tensor:
    """Decode one GGUF tensor (F32/F16/Q4_0/Q4_1/Q8_0) to a torch tensor in HF orientation."""
    shape = torch_shape(tensor.shape)
    n = int(np.prod(shape)) if shape else 1
    tt = int(tensor.tensor_type)
    if tt == int(GGMLQuantizationType.F32):
        flat = np.asarray(tensor.data, dtype=np.float32).reshape(-1)
    elif tt == int(GGMLQuantizationType.F16):
        flat = np.asarray(tensor.data, dtype=np.float16).reshape(-1).astype(np.float32)
    elif tt in DEQUANT:
        flat = DEQUANT[tt](np.asarray(tensor.data), n)
    else:
        raise NotImplementedError(f"{tensor.name}: unsupported GGUF type {GGMLQuantizationType(tt).name}")
    return torch.from_numpy(np.ascontiguousarray(flat.reshape(shape))).to(dtype)


def whisper_config_from_gguf(reader: GGUFReader) -> WhisperConfig:
    """Derive a WhisperConfig from the GGUF tensor shapes + layer counts."""
    by_name = {t.name: t for t in reader.tensors}

    def layer_count(prefix: str) -> int:
        idxs = {
            int(nm[len(prefix):].split(".", 1)[0])
            for nm in by_name
            if nm.startswith(prefix)
        }
        return len(idxs)

    conv1 = torch_shape(by_name["model.encoder.conv1.weight"].shape)  # (d_model, n_mels, 3)
    d_model, n_mels = conv1[0], conv1[1]
    ffn = torch_shape(by_name["model.encoder.layers.0.fc1.weight"].shape)[0]
    enc_pos = torch_shape(by_name["model.encoder.embed_positions.weight"].shape)[0]
    dec_pos = torch_shape(by_name["model.decoder.embed_positions.weight"].shape)[0]
    vocab = torch_shape(by_name["model.decoder.embed_tokens.weight"].shape)[0]
    heads = d_model // WHISPER_HEAD_DIM
    return WhisperConfig(
        vocab_size=vocab,
        num_mel_bins=n_mels,
        d_model=d_model,
        encoder_layers=layer_count("model.encoder.layers."),
        decoder_layers=layer_count("model.decoder.layers."),
        encoder_attention_heads=heads,
        decoder_attention_heads=heads,
        encoder_ffn_dim=ffn,
        decoder_ffn_dim=ffn,
        max_source_positions=enc_pos,
        max_target_positions=dec_pos,
        tie_word_embeddings=True,
    )


def whisper_state_dict_from_gguf(reader: GGUFReader, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Dequantize every GGUF tensor to a torch state dict keyed by HF param name.
    Skips `mel_filters` (a bundled extra, not an HF parameter)."""
    return {
        t.name: tensor_to_torch(t, dtype)
        for t in reader.tensors
        if t.name != "mel_filters"
    }


def build_whisper_eager(
    path: str, dtype: torch.dtype = torch.float32,
) -> tuple[WhisperForConditionalGeneration, list[str], list[str]]:
    """Build an eager WhisperForConditionalGeneration from the GGUF (no alloy).
    Returns (model, missing_keys, unexpected_keys). `proj_out.weight` is expected
    missing (tied to decoder.embed_tokens) and re-tied here."""
    reader = GGUFReader(path)
    config = whisper_config_from_gguf(reader)
    model = WhisperForConditionalGeneration(config)
    state = whisper_state_dict_from_gguf(reader, dtype)
    result = model.load_state_dict(state, strict=False)
    model.tie_weights()
    return model.to(dtype).eval(), list(result.missing_keys), list(result.unexpected_keys)
