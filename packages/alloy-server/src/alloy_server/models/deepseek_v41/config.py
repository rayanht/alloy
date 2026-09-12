"""DeepSeek-V4.1 model configuration, read from the `deepseek41` GGUF metadata.

Field names follow the reference `inference/model.py` `ModelArgs`. The GGUF carries
the per-layer compress ratios, the indexer/hc/engram constants and the engram hash
tables' primes/offsets/multipliers; the KV and index source layers are inferred from
which layers ship compressor / indexer tensors. The hierarchical-indexer candidate
constants are not in the GGUF and default to the published config.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Expert gating functions as llama.cpp encodes `expert_gating_func`.
GATING_FUNCS = {1: "softmax", 2: "sigmoid", 4: "sqrtsoftplus"}


@dataclass(frozen=True)
class EngramConfig:
    layer_ids: tuple[int, ...]
    n_heads: int
    head_dim: int
    max_ngram_size: int
    # Unpadded table rows, one per engram layer (the `engram_embd` tensor's row count).
    num_embeddings: tuple[int, ...]
    multipliers: np.ndarray  # [n_layers, max_ngram_size] int64
    primes: np.ndarray  # [n_layers, max_ngram_size - 1, n_heads] int64
    offsets: np.ndarray  # [n_layers, max_ngram_size - 1, n_heads] int64
    token_map: np.ndarray  # [vocab] int64, raw token id -> compressed id
    pad_id: int  # already in the compressed id space

    @property
    def n_hash_cols(self) -> int:
        return (self.max_ngram_size - 1) * self.n_heads


@dataclass(frozen=True)
class DeepseekV41Config:
    vocab_size: int
    dim: int
    n_layers: int
    n_heads: int
    head_dim: int
    rope_head_dim: int
    q_lora_rank: int
    o_lora_rank: int
    o_groups: int
    norm_eps: float
    max_seq_len: int
    # attention
    window_size: int
    compress_ratios: tuple[int, ...]  # one per layer (any trailing draft-layer entries dropped)
    kv_source_layers: tuple[int, ...]
    index_source_layers: tuple[int, ...]
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    candidate_source_layer: int
    candidate_topk_blocks: int
    candidate_block_size: int
    # rope
    rope_theta: float
    compress_rope_theta: float
    rope_factor: float
    original_seq_len: int
    beta_fast: float
    beta_slow: float
    # moe
    n_routed_experts: int
    n_shared_experts: int
    n_activated_experts: int
    moe_inter_dim: int
    score_func: str
    route_scale: float
    norm_topk_prob: bool
    swiglu_limit: float
    # hyper-connections
    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float
    engram: EngramConfig | None = None
    bos_token_id: int = 0
    eos_token_id: int = 1
    pad_token_id: int = 2

    @property
    def nope_head_dim(self) -> int:
        return self.head_dim - self.rope_head_dim

    @property
    def hc_mix_dim(self) -> int:
        return (2 + self.hc_mult) * self.hc_mult

    def compress_ratio(self, layer: int) -> int:
        return self.compress_ratios[layer]

    def is_kv_source(self, layer: int) -> bool:
        return layer in self.kv_source_layers

    def is_index_source(self, layer: int) -> bool:
        return layer in self.index_source_layers

    def kv_source_for(self, layer: int) -> int:
        """The layer whose compressed KV cache `layer` reads (itself if a source)."""
        if not self.compress_ratios[layer]:
            raise ValueError(f"layer {layer} has no compressed KV")
        src = max(s for s in self.kv_source_layers if s <= layer)
        if self.compress_ratios[src] != self.compress_ratios[layer]:
            raise ValueError(f"layer {layer} reads KV from layer {src} with a different ratio")
        return src

    def index_source_for(self, layer: int) -> int:
        if not self.compress_ratios[layer]:
            raise ValueError(f"layer {layer} has no indexer")
        return max(s for s in self.index_source_layers if s <= layer)

    def uses_candidates(self, layer: int) -> bool:
        return 0 <= self.candidate_source_layer < layer

    def rope_params(self, layer: int) -> tuple[float, int]:
        """(theta, original_seq_len) for a layer: compressed layers use the compress
        theta with YaRN, pure sliding-window layers the base theta without it."""
        if self.compress_ratios[layer]:
            return self.compress_rope_theta, self.original_seq_len
        return self.rope_theta, 0


# Hierarchical sparse indexer constants (HF config.json) — absent from the GGUF.
CANDIDATE_SOURCE_LAYER = 20
CANDIDATE_TOPK_BLOCKS = 2048
CANDIDATE_BLOCK_SIZE = 8


def config_from_gguf_kv(
    kv: dict, tensor_shapes: dict[str, tuple[int, ...]], *, arch: str = "deepseek41", n_layers: int | None = None,
) -> DeepseekV41Config:
    """Build the config from a GGUF's key/value table (`kv`, already parsed to
    python values) and its tensor name -> torch shape map. `n_layers` truncates the
    model (bring-up on a partial download)."""
    tensor_names = set(tensor_shapes)

    def k(key: str, default=None):
        full = f"{arch}.{key}"
        if full in kv:
            return kv[full]
        if default is None:
            raise KeyError(f"GGUF metadata key missing: {full}")
        return default

    n_layers = int(k("block_count")) if n_layers is None else n_layers
    compress_ratios = tuple(int(r) for r in k("attention.compress_ratios"))[:n_layers]
    if len(compress_ratios) != n_layers:
        raise ValueError(f"compress_ratios has {len(compress_ratios)} entries for {n_layers} layers")
    kv_sources = tuple(
        i for i in range(n_layers) if f"blk.{i}.attn_compressor_kv.weight" in tensor_names
    )
    index_sources = tuple(
        i for i in range(n_layers) if f"blk.{i}.indexer.attn_q_b.weight" in tensor_names
    )
    if not kv_sources or not index_sources:
        raise ValueError("no compressor / indexer tensors found — not a deepseek41 GGUF")

    gating = int(k("expert_gating_func"))
    if gating not in GATING_FUNCS:
        raise ValueError(f"unknown expert_gating_func {gating}")

    engram = None
    engram_layers = k("engram.layer_ids", ())
    if engram_layers:
        layer_ids = tuple(int(i) for i in engram_layers if int(i) < n_layers)
        n_heads = int(k("engram.head_count"))
        max_ngram = int(k("engram.max_ngram_size"))
        n_l = len(layer_ids)
        engram = EngramConfig(
            layer_ids=layer_ids,
            n_heads=n_heads,
            head_dim=int(k("engram.key_length")),
            max_ngram_size=max_ngram,
            num_embeddings=tuple(int(tensor_shapes[f"blk.{i}.engram_embd.weight"][0]) for i in layer_ids),
            multipliers=np.asarray(k("engram.multipliers"), dtype=np.int64).reshape(n_l, max_ngram),
            primes=np.asarray(k("engram.primes"), dtype=np.int64).reshape(n_l, max_ngram - 1, n_heads),
            offsets=np.asarray(k("engram.offsets"), dtype=np.int64).reshape(n_l, max_ngram - 1, n_heads),
            token_map=np.asarray(k("engram.token_map"), dtype=np.int64),
            pad_id=int(k("engram.pad_id")),
        )

    return DeepseekV41Config(
        vocab_size=len(kv["tokenizer.ggml.tokens"]) if "tokenizer.ggml.tokens" in kv else int(k("vocab_size")),
        dim=int(k("embedding_length")),
        n_layers=n_layers,
        n_heads=int(k("attention.head_count")),
        head_dim=int(k("attention.key_length")),
        rope_head_dim=int(k("rope.dimension_count")),
        q_lora_rank=int(k("attention.q_lora_rank")),
        o_lora_rank=int(k("attention.output_lora_rank")),
        o_groups=int(k("attention.output_group_count")),
        norm_eps=float(k("attention.layer_norm_rms_epsilon")),
        max_seq_len=int(k("context_length")),
        window_size=int(k("attention.sliding_window")),
        compress_ratios=compress_ratios,
        kv_source_layers=kv_sources,
        index_source_layers=index_sources,
        index_n_heads=int(k("attention.indexer.head_count")),
        index_head_dim=int(k("attention.indexer.key_length")),
        index_topk=int(k("attention.indexer.top_k")),
        candidate_source_layer=int(k("attention.candidate_source_layer", CANDIDATE_SOURCE_LAYER)),
        candidate_topk_blocks=int(k("attention.candidate_topk_blocks", CANDIDATE_TOPK_BLOCKS)),
        candidate_block_size=int(k("attention.candidate_block_size", CANDIDATE_BLOCK_SIZE)),
        rope_theta=float(k("rope.freq_base")),
        compress_rope_theta=float(k("attention.compress_rope_freq_base")),
        rope_factor=float(k("rope.scaling.factor", 1.0)),
        original_seq_len=int(k("rope.scaling.original_context_length", 0)),
        beta_fast=float(k("rope.scaling.yarn_beta_fast", 32.0)),
        beta_slow=float(k("rope.scaling.yarn_beta_slow", 1.0)),
        n_routed_experts=int(k("expert_count")),
        n_shared_experts=int(k("expert_shared_count")),
        n_activated_experts=int(k("expert_used_count")),
        moe_inter_dim=int(k("expert_feed_forward_length")),
        score_func=GATING_FUNCS[gating],
        route_scale=float(k("expert_weights_scale")),
        norm_topk_prob=bool(k("expert_weights_norm")),
        swiglu_limit=float(k("swiglu_clamp_exp")[0]) if f"{arch}.swiglu_clamp_exp" in kv else 0.0,
        hc_mult=int(k("hyper_connection.count")),
        hc_sinkhorn_iters=int(k("hyper_connection.sinkhorn_iterations")),
        hc_eps=float(k("hyper_connection.epsilon")),
        engram=engram,
        bos_token_id=int(kv.get("tokenizer.ggml.bos_token_id", 0)),
        eos_token_id=int(kv.get("tokenizer.ggml.eos_token_id", 1)),
        pad_token_id=int(kv.get("tokenizer.ggml.padding_token_id", 2)),
    )


def gguf_kv_for_config(config: DeepseekV41Config, *, arch: str = "deepseek41") -> dict:
    """The inverse of `config_from_gguf_kv`: the metadata a synthetic `deepseek41`
    GGUF needs so that reading it back yields `config` (tests)."""
    gating = {v: k for k, v in GATING_FUNCS.items()}[config.score_func]
    kv: dict = {
        f"{arch}.block_count": config.n_layers,
        f"{arch}.context_length": config.max_seq_len,
        f"{arch}.embedding_length": config.dim,
        f"{arch}.attention.head_count": config.n_heads,
        f"{arch}.attention.head_count_kv": 1,
        f"{arch}.attention.key_length": config.head_dim,
        f"{arch}.attention.value_length": config.head_dim,
        f"{arch}.rope.dimension_count": config.rope_head_dim,
        f"{arch}.attention.q_lora_rank": config.q_lora_rank,
        f"{arch}.attention.output_lora_rank": config.o_lora_rank,
        f"{arch}.attention.output_group_count": config.o_groups,
        f"{arch}.attention.layer_norm_rms_epsilon": config.norm_eps,
        f"{arch}.attention.sliding_window": config.window_size,
        f"{arch}.attention.compress_ratios": list(config.compress_ratios),
        f"{arch}.attention.indexer.head_count": config.index_n_heads,
        f"{arch}.attention.indexer.key_length": config.index_head_dim,
        f"{arch}.attention.indexer.top_k": config.index_topk,
        f"{arch}.attention.candidate_source_layer": config.candidate_source_layer,
        f"{arch}.attention.candidate_topk_blocks": config.candidate_topk_blocks,
        f"{arch}.attention.candidate_block_size": config.candidate_block_size,
        f"{arch}.rope.freq_base": config.rope_theta,
        f"{arch}.attention.compress_rope_freq_base": config.compress_rope_theta,
        f"{arch}.rope.scaling.factor": config.rope_factor,
        f"{arch}.rope.scaling.original_context_length": config.original_seq_len,
        f"{arch}.rope.scaling.yarn_beta_fast": config.beta_fast,
        f"{arch}.rope.scaling.yarn_beta_slow": config.beta_slow,
        f"{arch}.expert_count": config.n_routed_experts,
        f"{arch}.expert_shared_count": config.n_shared_experts,
        f"{arch}.expert_used_count": config.n_activated_experts,
        f"{arch}.expert_feed_forward_length": config.moe_inter_dim,
        f"{arch}.expert_gating_func": gating,
        f"{arch}.expert_weights_scale": config.route_scale,
        f"{arch}.expert_weights_norm": config.norm_topk_prob,
        f"{arch}.swiglu_clamp_exp": [config.swiglu_limit] * config.n_layers,
        f"{arch}.hyper_connection.count": config.hc_mult,
        f"{arch}.hyper_connection.sinkhorn_iterations": config.hc_sinkhorn_iters,
        f"{arch}.hyper_connection.epsilon": config.hc_eps,
        f"{arch}.vocab_size": config.vocab_size,
        "tokenizer.ggml.bos_token_id": config.bos_token_id,
        "tokenizer.ggml.eos_token_id": config.eos_token_id,
        "tokenizer.ggml.padding_token_id": config.pad_token_id,
    }
    e = config.engram
    if e is not None:
        kv.update(
            {
                f"{arch}.engram.layer_ids": list(e.layer_ids),
                f"{arch}.engram.head_count": e.n_heads,
                f"{arch}.engram.key_length": e.head_dim,
                f"{arch}.engram.max_ngram_size": e.max_ngram_size,
                f"{arch}.engram.multipliers": [int(x) for x in e.multipliers.reshape(-1)],
                f"{arch}.engram.primes": [int(x) for x in e.primes.reshape(-1)],
                f"{arch}.engram.offsets": [int(x) for x in e.offsets.reshape(-1)],
                f"{arch}.engram.token_map": [int(x) for x in e.token_map],
                f"{arch}.engram.pad_id": e.pad_id,
            }
        )
    return kv
