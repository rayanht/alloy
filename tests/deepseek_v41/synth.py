"""Synthetic tiny `deepseek41` models: random weights in the real GGUF layout.

The oracle (`reference.ReferenceModel`) and the GGUF must see identical effective
weights, so every quantized tensor is quantized first and the oracle gets the
dequantized result.
"""

from __future__ import annotations

from pathlib import Path

import gguf
import numpy as np
import torch

from alloy_server.gguf import kquant
from alloy_server.models.deepseek_v41.config import DeepseekV41Config, gguf_kv_for_config
from alloy_server.models.deepseek_v41.engram import build_engram_config

Q4_K = gguf.GGMLQuantizationType.Q4_K
Q6_K = gguf.GGMLQuantizationType.Q6_K
F32 = gguf.GGMLQuantizationType.F32
BF16 = gguf.GGMLQuantizationType.BF16


def tiny_config(*, seed: int = 0, n_layers: int = 6, max_seq_len: int = 64) -> DeepseekV41Config:
    """The real model's layer structure in miniature: 2 SWA layers, ratio-2 encoder
    layers with one KV source, ratio-1 decoder layers with one KV source, a Reindex
    layer, a candidate source, engram on two layers."""
    if n_layers != 6:
        raise ValueError("tiny_config is laid out for 6 layers")
    vocab = 512
    rng = np.random.default_rng(seed)
    compressed_vocab = 64
    token_map = rng.integers(0, compressed_vocab, size=vocab, dtype=np.int64)
    token_map[:compressed_vocab] = np.arange(compressed_vocab)  # every compressed id is hit
    engram = build_engram_config(
        layer_ids=(1, 3),
        n_heads=4,
        head_dim=256,
        max_ngram_size=3,
        vocab_size=100,
        token_map=token_map,
        compressed_vocab_size=compressed_vocab,
        pad_token_id=2,
    )
    return DeepseekV41Config(
        vocab_size=vocab,
        dim=256,
        n_layers=6,
        n_heads=4,
        head_dim=256,
        rope_head_dim=64,
        q_lora_rank=256,
        o_lora_rank=128,
        o_groups=2,
        norm_eps=1e-20,
        max_seq_len=max_seq_len,
        window_size=8,
        compress_ratios=(0, 0, 2, 2, 1, 1),
        kv_source_layers=(2, 4),
        index_source_layers=(2, 4, 5),
        index_n_heads=2,
        index_head_dim=128,
        index_topk=4,
        candidate_source_layer=4,
        candidate_topk_blocks=3,
        candidate_block_size=2,
        rope_theta=10000.0,
        compress_rope_theta=160000.0,
        rope_factor=16.0,
        original_seq_len=32,
        beta_fast=32.0,
        beta_slow=1.0,
        n_routed_experts=8,
        n_shared_experts=1,
        n_activated_experts=2,
        moe_inter_dim=256,
        score_func="sqrtsoftplus",
        route_scale=1.5,
        norm_topk_prob=True,
        swiglu_limit=10.0,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        engram=engram,
    )


def down_qtype(layer: int) -> gguf.GGMLQuantizationType:
    """The real file mixes Q6_K and Q4_K down experts across layers."""
    return Q6_K if layer % 2 == 0 else Q4_K


def tensor_specs(cfg: DeepseekV41Config) -> dict[str, tuple[tuple[int, ...], gguf.GGMLQuantizationType]]:
    """name -> (torch shape, gguf type), mirroring the real deepseek41 GGUF."""
    d, hc = cfg.dim, cfg.hc_mult
    specs: dict[str, tuple[tuple[int, ...], gguf.GGMLQuantizationType]] = {
        "token_embd.weight": ((cfg.vocab_size, d), Q4_K),
        "output.weight": ((cfg.vocab_size, d), Q6_K),
        "output_norm.weight": ((d,), F32),
    }
    for i in range(cfg.n_layers):
        p = f"blk.{i}."
        specs.update(
            {
                p + "attn_norm.weight": ((d,), F32),
                p + "ffn_norm.weight": ((d,), F32),
                p + "attn_q_a.weight": ((cfg.q_lora_rank, d), Q4_K),
                p + "attn_q_a_norm.weight": ((cfg.q_lora_rank,), F32),
                p + "attn_q_b.weight": ((cfg.n_heads * cfg.head_dim, cfg.q_lora_rank), Q4_K),
                p + "attn_kv.weight": ((cfg.head_dim, d), Q4_K),
                p + "attn_kv_a_norm.weight": ((cfg.head_dim,), F32),
                p + "attn_output_a.weight": ((cfg.o_groups * cfg.o_lora_rank, cfg.n_heads * cfg.head_dim // cfg.o_groups), Q4_K),
                p + "attn_output_b.weight": ((d, cfg.o_groups * cfg.o_lora_rank), Q4_K),
                p + "attn_sinks.weight": ((cfg.n_heads,), F32),
                p + "hc_attn_fn.weight": ((cfg.hc_mix_dim, hc * d), Q4_K),
                p + "hc_attn_base.weight": ((cfg.hc_mix_dim,), F32),
                p + "hc_attn_scale.weight": ((3,), F32),
                p + "hc_ffn_fn.weight": ((cfg.hc_mix_dim, hc * d), Q4_K),
                p + "hc_ffn_base.weight": ((cfg.hc_mix_dim,), F32),
                p + "hc_ffn_scale.weight": ((3,), F32),
                p + "ffn_gate_inp.weight": ((cfg.n_routed_experts, d), BF16),
                p + "exp_probs_b.bias": ((cfg.n_routed_experts,), F32),
                p + "exp_probs_b_vl.bias": ((cfg.n_routed_experts,), F32),
                p + "ffn_gate_exps.weight": ((cfg.n_routed_experts, cfg.moe_inter_dim, d), Q4_K),
                p + "ffn_up_exps.weight": ((cfg.n_routed_experts, cfg.moe_inter_dim, d), Q4_K),
                p + "ffn_down_exps.weight": ((cfg.n_routed_experts, d, cfg.moe_inter_dim), down_qtype(i)),
                p + "ffn_gate_shexp.weight": ((cfg.moe_inter_dim, d), Q4_K),
                p + "ffn_up_shexp.weight": ((cfg.moe_inter_dim, d), Q4_K),
                p + "ffn_down_shexp.weight": ((d, cfg.moe_inter_dim), down_qtype(i)),
            }
        )
        if cfg.is_kv_source(i):
            specs[p + "attn_compressor_kv.weight"] = ((cfg.head_dim, d), Q4_K)
            specs[p + "attn_compressor_norm.weight"] = ((cfg.head_dim,), F32)
            if cfg.compress_ratios[i] > 1:
                specs[p + "attn_compressor_gate.weight"] = ((cfg.head_dim, d), Q4_K)
            specs[p + "indexer.attn_k.weight"] = ((cfg.index_head_dim, cfg.head_dim), Q4_K)
            specs[p + "indexer.k_norm.weight"] = ((cfg.index_head_dim,), F32)
        if cfg.is_index_source(i):
            specs[p + "indexer.attn_q_b.weight"] = ((cfg.index_n_heads * cfg.index_head_dim, cfg.q_lora_rank), Q4_K)
            specs[p + "indexer.proj.weight"] = ((cfg.index_n_heads, d), Q4_K)
        e = cfg.engram
        if e is not None and i in e.layer_ids:
            rows = e.num_embeddings[e.layer_ids.index(i)]
            specs[p + "engram_embd.weight"] = ((rows, e.head_dim), Q4_K)
            specs[p + "engram_k.weight"] = ((hc, d), Q4_K)
            specs[p + "engram_q.weight"] = ((hc, d), Q4_K)
            specs[p + "engram_wkv.weight"] = ((d * (hc + 1), e.n_hash_cols * e.head_dim), Q4_K)
    return specs


def random_tensor(name: str, shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    """Trained-model-like magnitudes so the forward stays in range."""
    if name.endswith("_norm.weight") or name.endswith("norm.weight"):
        return (1.0 + 0.1 * rng.standard_normal(shape)).astype(np.float32)
    if "hc_" in name and name.endswith("scale.weight"):
        return (0.1 + 0.05 * rng.random(shape)).astype(np.float32)
    if "hc_" in name and name.endswith("base.weight"):
        return (0.5 * rng.standard_normal(shape)).astype(np.float32)
    if name.endswith("attn_sinks.weight") or name.endswith(".bias"):
        return (0.5 * rng.standard_normal(shape)).astype(np.float32)
    if name.endswith("engram_k.weight") or name.endswith("engram_q.weight"):
        return (1.0 + 0.1 * rng.standard_normal(shape)).astype(np.float32)
    if len(shape) == 1:
        return rng.standard_normal(shape).astype(np.float32)
    fan_in = shape[-1]
    return (rng.standard_normal(shape) / np.sqrt(fan_in)).astype(np.float32)


def quantize(arr: np.ndarray, qtype: gguf.GGMLQuantizationType) -> tuple[np.ndarray, np.ndarray]:
    """(packed bytes for the GGUF, the dequantized fp32 the oracle uses)."""
    if qtype == F32:
        return arr.astype(np.float32), arr.astype(np.float32)
    if qtype == BF16:
        bits = (arr.astype(np.float32).view(np.uint32) + 0x8000) >> 16  # round-to-nearest-even-ish
        back = (bits.astype(np.uint32) << 16).view(np.float32)
        return bits.astype(np.uint16), back
    packed = kquant.quantize(arr.astype(np.float32), qtype)
    return packed, gguf.quants.dequantize(packed, qtype).astype(np.float32)


class SyntheticModel:
    """Random weights for `cfg`: `oracle` holds dequantized fp32 torch tensors, `packed`
    the GGUF payloads (name -> (bytes/array, qtype, torch shape))."""

    def __init__(self, cfg: DeepseekV41Config, *, seed: int = 0) -> None:
        self.cfg = cfg
        rng = np.random.default_rng(seed + 1)
        self.oracle: dict[str, torch.Tensor] = {}
        self.packed: dict[str, tuple[np.ndarray, gguf.GGMLQuantizationType, tuple[int, ...]]] = {}
        for name, (shape, qtype) in tensor_specs(cfg).items():
            arr = random_tensor(name, shape, rng)
            packed, deq = quantize(arr, qtype)
            self.oracle[name] = torch.from_numpy(deq.reshape(shape))
            self.packed[name] = (packed, qtype, shape)

    def write_gguf(self, path: Path, *, arch: str = "deepseek41") -> Path:
        writer = gguf.GGUFWriter(str(path), arch)
        for key, value in gguf_kv_for_config(self.cfg, arch=arch).items():
            if key == "general.architecture":
                continue
            if isinstance(value, bool):
                writer.add_bool(key, value)
            elif isinstance(value, int):
                if abs(value) >= 2**31:
                    writer.add_key_value(key, value, gguf.GGUFValueType.INT64)
                else:
                    writer.add_uint32(key, value) if value >= 0 else writer.add_int32(key, value)
            elif isinstance(value, float):
                writer.add_float32(key, value)
            elif isinstance(value, list):
                if all(isinstance(v, int) for v in value) and any(abs(v) >= 2**31 for v in value):
                    writer.add_key_value(key, value, gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.UINT64)
                elif all(isinstance(v, int) for v in value):
                    writer.add_key_value(key, value, gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.INT32)
                else:
                    writer.add_array(key, value)
            else:
                writer.add_string(key, str(value))
        for name, (packed, qtype, shape) in self.packed.items():
            if qtype in (F32,):
                writer.add_tensor(name, packed.reshape(shape))
            elif qtype == BF16:
                writer.add_tensor(name, packed.reshape(shape), raw_dtype=BF16)
            else:
                writer.add_tensor(name, packed.reshape(*shape[:-1], -1), raw_dtype=qtype)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        return path
