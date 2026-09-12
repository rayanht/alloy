"""DeepSeek-V4.1 generation engine: load the split GGUF, size the expert arena to the
machine, run chunked prefill + greedy/sampled decode over `DeepseekV41Model`."""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from alloy import get_logger
from alloy_server.gguf.split import SplitGGUF
from alloy_server.models.deepseek_v41.config import DeepseekV41Config, config_from_gguf_kv
from alloy_server.models.deepseek_v41.experts import EngramTables, ExpertStore
from alloy_server.models.deepseek_v41.model import DeepseekV41Model, SequenceState
from alloy_server.models.deepseek_v41.weights import ModelWeights

logger = get_logger("alloy_server.deepseek_v41")

# Working-set headroom left for activations, the engram page cache and the OS.
ARENA_HEADROOM_BYTES = 12 << 30
DEFAULT_MAX_SEQ_LEN = 131072


def resident_weight_bytes(split: SplitGGUF) -> int:
    return sum(
        loc.n_bytes for name, loc in split.tensors.items()
        if "_exps." not in name and "engram_embd" not in name
    )


def kv_bytes(cfg: DeepseekV41Config, max_seq_len: int) -> int:
    total = 0
    for layer in cfg.kv_source_layers:
        groups = max(1, max_seq_len // cfg.compress_ratios[layer])
        total += groups * (cfg.head_dim + cfg.index_head_dim) * 2
    total += cfg.n_layers * (cfg.window_size + 4096) * cfg.head_dim * 2
    return total


def default_arena_bytes(split: SplitGGUF, cfg: DeepseekV41Config, max_seq_len: int) -> int:
    """Expert arena budget: the GPU working set minus resident weights, caches and
    headroom; `ALLOY_EXPERT_ARENA_GB` overrides."""
    override = os.environ.get("ALLOY_EXPERT_ARENA_GB")
    if override:
        return int(float(override) * (1 << 30))
    from alloy._runtime.metal import default_device  # scoped: Metal device init

    working_set = int(default_device().recommended_max_working_set_size)
    budget = working_set - resident_weight_bytes(split) - kv_bytes(cfg, max_seq_len) - ARENA_HEADROOM_BYTES
    return max(budget, 4 << 30)


@dataclass
class GenerationStats:
    prompt_tokens: int = 0
    reused_tokens: int = 0
    prefill_s: float = 0.0
    decode_tokens: int = 0
    decode_s: float = 0.0
    expert_misses: int = 0
    expert_hits: int = 0
    expert_bytes: int = 0
    step_times: list[float] = field(default_factory=list)


class DeepseekV41Engine:
    def __init__(
        self,
        split: SplitGGUF,
        *,
        max_seq_len: int | None = None,
        chunk_size: int = 2048,
        arena_bytes: int | None = None,
        rounding: bool = True,
        io_threads: int = 8,
        bypass_page_cache: bool = True,
        n_layers: int | None = None,
    ) -> None:
        self.split = split
        self.cfg = config_from_gguf_kv(split.kv, split.shapes(), n_layers=n_layers)
        self.max_seq_len = min(self.cfg.max_seq_len, max_seq_len or DEFAULT_MAX_SEQ_LEN)
        self.chunk_size = chunk_size
        t0 = time.perf_counter()
        self.weights = ModelWeights(split, self.cfg)
        budget = arena_bytes if arena_bytes is not None else default_arena_bytes(split, self.cfg, self.max_seq_len)
        self.experts = ExpertStore(
            split, self.cfg, budget_bytes=budget, io_threads=io_threads, bypass_page_cache=bypass_page_cache,
        )
        self.engram = EngramTables(split, self.cfg)
        self.model = DeepseekV41Model(
            self.cfg, self.weights, self.experts, self.engram, max_seq_len=self.max_seq_len, rounding=rounding,
        )
        self.state: SequenceState = self.model.new_state()
        # every token the live state has consumed (prompt + generated), for prefix reuse
        self.history: list[int] = []
        logger.info(
            "deepseek41_loaded",
            took_s=round(time.perf_counter() - t0, 1),
            max_seq_len=self.max_seq_len,
            arena_gb=round(budget / (1 << 30), 1),
            arena_slots={a.layout.down_qtype.name: a.slots for a in self.experts.arenas.values()},
            resident_gb=round(resident_weight_bytes(split) / (1 << 30), 2),
        )

    @classmethod
    def from_path(cls, path: Path | str, *, allow_missing: bool = False, **kwargs) -> "DeepseekV41Engine":
        return cls(SplitGGUF(Path(path), allow_missing=allow_missing), **kwargs)

    def reset(self) -> None:
        self.state = self.model.new_state()
        self.history = []

    def resume_point(self, ids: np.ndarray) -> int:
        """How many leading tokens of `ids` the live state already holds. Only a
        prompt that extends the whole history resumes (the multi-turn append); any
        divergence restarts cold — the caches are position-indexed but the window /
        compressor state has no rewind."""
        n = self.state.pos
        if n == 0 or len(ids) <= n:
            return 0
        if self.history[:n] == ids[:n].tolist():
            return n
        return 0

    def prefill(self, ids: np.ndarray) -> np.ndarray:
        """Chunked prefill from the current position; returns the last token's logits."""
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        logits = None
        for start in range(0, len(ids), self.chunk_size):
            chunk = ids[start : start + self.chunk_size]
            logits = self.model.forward(self.state, chunk)
            self.history.extend(int(t) for t in chunk)
        return logits[0]

    def warmup(self) -> None:
        """JIT every kernel variant (prefill chunk + decode step) on a scratch state."""
        ids = np.array([self.cfg.bos_token_id, 3, 4, 5], dtype=np.int64)
        for _ in self.generate(ids, 2):
            pass
        self.reset()

    def generate(
        self,
        prompt_ids: np.ndarray,
        max_new_tokens: int,
        *,
        eos_ids: tuple[int, ...] = (),
        temperature: float = 0.0,
        seed: int = 0,
        stats: GenerationStats | None = None,
    ) -> Iterator[int]:
        """Prefill the prompt then decode token by token (greedy when temperature is 0)."""
        rng = np.random.default_rng(seed)
        prompt_ids = np.asarray(prompt_ids, dtype=np.int64).reshape(-1)
        resume = self.resume_point(prompt_ids)
        if resume == 0:
            self.reset()
        suffix = prompt_ids[resume:]
        if self.state.pos + len(suffix) + max_new_tokens > self.max_seq_len:
            raise ValueError(
                f"prompt {len(prompt_ids)} + {max_new_tokens} new tokens exceeds max_seq_len {self.max_seq_len}"
            )
        misses0, hits0, bytes0 = self.experts.misses, self.experts.hits, self.experts.bytes_read
        t0 = time.perf_counter()
        logits = self.prefill(suffix)
        t1 = time.perf_counter()
        if stats is not None:
            stats.prompt_tokens += len(suffix)
            stats.reused_tokens += resume
            stats.prefill_s += t1 - t0
        for i in range(max_new_tokens):
            token = self.sample(logits, temperature, rng)
            yield token
            if stats is not None:
                stats.decode_tokens += 1
            if token in eos_ids or i == max_new_tokens - 1:
                break
            t_step = time.perf_counter()
            logits = self.model.forward(self.state, np.array([token], dtype=np.int64))[0]
            self.history.append(token)
            if stats is not None:
                stats.step_times.append(time.perf_counter() - t_step)
        if stats is not None:
            stats.decode_s += time.perf_counter() - t1
            stats.expert_misses += self.experts.misses - misses0
            stats.expert_hits += self.experts.hits - hits0
            stats.expert_bytes += self.experts.bytes_read - bytes0

    @staticmethod
    def sample(logits: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
        if temperature <= 0:
            return int(np.argmax(logits))
        z = logits.astype(np.float64) / temperature
        z -= z.max()
        p = np.exp(z)
        p /= p.sum()
        return int(rng.choice(len(p), p=p))
