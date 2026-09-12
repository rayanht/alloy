"""Resident DeepSeek-V4.1 weights as alloy buffers.

Everything except the routed experts (`ExpertStore`) and the engram tables
(`EngramTables`) is loaded from the split GGUF into Metal-shared memory: quantized
matrices stay in their native GGUF block layout (`(rows, row_bytes)` uint8, indexed
by the quant kernels), small tensors are dequantized to fp32.
"""

from __future__ import annotations

from dataclasses import dataclass

import gguf
import numpy as np

from alloy._compiler.dtypes import float32, uint8
from alloy._dispatch.buf_utils import _alloc_aligned
from alloy._runtime.alloy_buffer import AlloyBuffer
from alloy_server.gguf.split import SplitGGUF
from alloy_server.models.deepseek_v41.config import DeepseekV41Config

Q4_K = gguf.GGMLQuantizationType.Q4_K
Q6_K = gguf.GGMLQuantizationType.Q6_K


@dataclass(frozen=True)
class QuantMatrix:
    """A GGUF-native quantized matrix: `blk` is (n_out, row_bytes) uint8."""

    blk: AlloyBuffer
    qtype: gguf.GGMLQuantizationType
    n_out: int
    n_in: int


def f32_buffer(arr: np.ndarray) -> AlloyBuffer:
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    buf = _alloc_aligned(tuple(arr.shape), float32)
    buf.numpy[:] = arr
    return buf


def u8_buffer(arr: np.ndarray) -> AlloyBuffer:
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    buf = _alloc_aligned(tuple(arr.shape), uint8)
    buf.numpy[:] = arr
    return buf


class WeightLoader:
    def __init__(self, split: SplitGGUF) -> None:
        self.split = split

    def dense(self, name: str) -> AlloyBuffer:
        return f32_buffer(self.split.dequantized(name))

    def quant(self, name: str) -> QuantMatrix:
        loc = self.split.tensors[name]
        if loc.tensor_type not in (Q4_K, Q6_K):
            raise ValueError(f"{name}: expected Q4_K/Q6_K, got {loc.tensor_type.name}")
        rows = self.split.packed_rows(name)
        if rows.ndim != 2:
            raise ValueError(f"{name}: expected a 2D matrix, got shape {loc.shape}")
        return QuantMatrix(blk=u8_buffer(rows), qtype=loc.tensor_type, n_out=loc.shape[0], n_in=loc.shape[1])

    def has(self, name: str) -> bool:
        return name in self.split.tensors


@dataclass
class LayerWeights:
    attn_norm: AlloyBuffer
    ffn_norm: AlloyBuffer
    q_a: QuantMatrix
    q_a_norm: AlloyBuffer
    q_b: QuantMatrix
    kv: QuantMatrix
    kv_norm: AlloyBuffer
    out_a: QuantMatrix
    out_b: QuantMatrix
    sinks: AlloyBuffer
    hc_attn_fn: AlloyBuffer
    hc_attn_base: AlloyBuffer
    hc_attn_scale: AlloyBuffer
    hc_ffn_fn: AlloyBuffer
    hc_ffn_base: AlloyBuffer
    hc_ffn_scale: AlloyBuffer
    gate_inp: AlloyBuffer  # (E, d) f32 router
    probs_bias: AlloyBuffer
    shexp_gate: QuantMatrix
    shexp_up: QuantMatrix
    shexp_down: QuantMatrix
    compressor_kv: QuantMatrix | None = None
    compressor_gate: QuantMatrix | None = None
    compressor_norm: AlloyBuffer | None = None
    indexer_q_b: QuantMatrix | None = None
    indexer_proj: QuantMatrix | None = None
    indexer_k: QuantMatrix | None = None
    indexer_k_norm: AlloyBuffer | None = None
    engram_qk: AlloyBuffer | None = None  # (hc, d) f32 q_weight * k_weight
    engram_wkv: QuantMatrix | None = None


class ModelWeights:
    def __init__(self, split: SplitGGUF, cfg: DeepseekV41Config) -> None:
        self.cfg = cfg
        w = WeightLoader(split)
        self.token_embd = w.quant("token_embd.weight")
        self.output = w.quant("output.weight")
        self.output_norm = w.dense("output_norm.weight")
        self.layers: list[LayerWeights] = []
        for i in range(cfg.n_layers):
            p = f"blk.{i}."
            layer = LayerWeights(
                attn_norm=w.dense(p + "attn_norm.weight"),
                ffn_norm=w.dense(p + "ffn_norm.weight"),
                q_a=w.quant(p + "attn_q_a.weight"),
                q_a_norm=w.dense(p + "attn_q_a_norm.weight"),
                q_b=w.quant(p + "attn_q_b.weight"),
                kv=w.quant(p + "attn_kv.weight"),
                kv_norm=w.dense(p + "attn_kv_a_norm.weight"),
                out_a=w.quant(p + "attn_output_a.weight"),
                out_b=w.quant(p + "attn_output_b.weight"),
                sinks=w.dense(p + "attn_sinks.weight"),
                hc_attn_fn=w.dense(p + "hc_attn_fn.weight"),
                hc_attn_base=w.dense(p + "hc_attn_base.weight"),
                hc_attn_scale=w.dense(p + "hc_attn_scale.weight"),
                hc_ffn_fn=w.dense(p + "hc_ffn_fn.weight"),
                hc_ffn_base=w.dense(p + "hc_ffn_base.weight"),
                hc_ffn_scale=w.dense(p + "hc_ffn_scale.weight"),
                gate_inp=w.dense(p + "ffn_gate_inp.weight"),
                probs_bias=w.dense(p + "exp_probs_b.bias"),
                shexp_gate=w.quant(p + "ffn_gate_shexp.weight"),
                shexp_up=w.quant(p + "ffn_up_shexp.weight"),
                shexp_down=w.quant(p + "ffn_down_shexp.weight"),
            )
            if cfg.is_kv_source(i):
                layer.compressor_kv = w.quant(p + "attn_compressor_kv.weight")
                layer.compressor_norm = w.dense(p + "attn_compressor_norm.weight")
                if cfg.compress_ratios[i] > 1:
                    layer.compressor_gate = w.quant(p + "attn_compressor_gate.weight")
                layer.indexer_k = w.quant(p + "indexer.attn_k.weight")
                layer.indexer_k_norm = w.dense(p + "indexer.k_norm.weight")
            if cfg.is_index_source(i):
                layer.indexer_q_b = w.quant(p + "indexer.attn_q_b.weight")
                layer.indexer_proj = w.quant(p + "indexer.proj.weight")
            if cfg.engram is not None and i in cfg.engram.layer_ids:
                qk = split.dequantized(p + "engram_q.weight") * split.dequantized(p + "engram_k.weight")
                layer.engram_qk = f32_buffer(qk)
                layer.engram_wkv = w.quant(p + "engram_wkv.weight")
            self.layers.append(layer)
