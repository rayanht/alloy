"""Engram n-gram hashing (CPU side).

Port of the reference `engram.py`: every position is hashed as the 2..max_ngram-grams
ending there, over the compressed token-id space, one bucket range per (n-gram size,
head). Hash ids depend only on the token sequence, so they are computed on the CPU
ahead of the forward and drive the table row gather.
"""

from __future__ import annotations

import numpy as np

from alloy_server.models.deepseek_v41.config import EngramConfig

DEAD = -1


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    f = 41
    while f * f <= n:
        if n % f == 0 or n % (f + 2) == 0:
            return False
        f += 6
    return True


def next_unused_prime(start: int, seen: set[int]) -> int:
    candidate = start + 1
    while not is_prime(candidate) or candidate in seen:
        candidate += 1
    return candidate


def hash_multipliers(layer_ids: tuple[int, ...], max_ngram_size: int, compressed_vocab_size: int) -> np.ndarray:
    """One odd multiplier per (layer, lookback), bounded so id * multiplier fits int64."""
    bound = max(1, (np.iinfo(np.int64).max // compressed_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        gen = np.random.default_rng(10007 * layer_id)
        values = gen.integers(low=0, high=bound, size=(max_ngram_size,), dtype=np.int64)
        rows.append(values * 2 + 1)
    return np.stack(rows)


def bucket_primes(n_layers: int, max_ngram_size: int, n_heads: int, vocab_size: int) -> np.ndarray:
    """[n_layers, max_ngram_size - 1, n_heads] distinct primes above `vocab_size - 1`,
    drawn in layer / n-gram / head order and never reused."""
    seen: set[int] = set()
    out = np.zeros((n_layers, max_ngram_size - 1, n_heads), dtype=np.int64)
    for layer in range(n_layers):
        for ngram in range(max_ngram_size - 1):
            current = vocab_size - 1
            for head in range(n_heads):
                current = next_unused_prime(current, seen)
                seen.add(current)
                out[layer, ngram, head] = current
    return out


def bucket_offsets(primes: np.ndarray) -> np.ndarray:
    """Cumulative bucket starts per layer, in the flat (n-gram, head) order."""
    flat = primes.reshape(primes.shape[0], -1)
    offsets = np.zeros_like(flat)
    offsets[:, 1:] = np.cumsum(flat[:, :-1], axis=1)
    return offsets.reshape(primes.shape)


def build_engram_config(
    *,
    layer_ids: tuple[int, ...],
    n_heads: int,
    head_dim: int,
    max_ngram_size: int,
    vocab_size: int,
    token_map: np.ndarray,
    compressed_vocab_size: int,
    pad_token_id: int,
) -> EngramConfig:
    """The full layout from scratch (the converter's job for a real model; tests)."""
    primes = bucket_primes(len(layer_ids), max_ngram_size, n_heads, vocab_size)
    return EngramConfig(
        layer_ids=layer_ids,
        n_heads=n_heads,
        head_dim=head_dim,
        max_ngram_size=max_ngram_size,
        num_embeddings=tuple(int(primes[i].sum()) for i in range(len(layer_ids))),
        multipliers=hash_multipliers(layer_ids, max_ngram_size, compressed_vocab_size),
        primes=primes,
        offsets=bucket_offsets(primes),
        token_map=np.asarray(token_map, dtype=np.int64),
        pad_id=int(token_map[pad_token_id]),
    )


class EngramHasher:
    """Maps positions to engram table rows, keeping the compressed-id history so
    decode steps hash against the tokens before them."""

    def __init__(self, config: EngramConfig, max_seq_len: int) -> None:
        self.config = config
        self.history = np.empty(max_seq_len, dtype=np.int64)

    def hash_ids(self, input_ids: np.ndarray, start_pos: int, token_mask: np.ndarray | None = None) -> np.ndarray:
        """Rows for positions [start_pos, start_pos + T). `token_mask` False marks
        tokens that take no part in an n-gram (image spans). Returns
        [T, n_engram_layers, n_hash_cols] int64."""
        cfg = self.config
        ids = np.asarray(input_ids, dtype=np.int64).reshape(-1)
        seqlen = ids.shape[0]
        compressed = cfg.token_map[ids]
        if token_mask is not None:
            compressed = np.where(token_mask, compressed, DEAD)
        self.history[start_pos : start_pos + seqlen] = compressed

        positions = np.arange(start_pos, start_pos + seqlen)
        tokens = np.empty((seqlen, cfg.max_ngram_size), dtype=np.int64)
        blocked = np.zeros(seqlen, dtype=bool)
        for shift in range(cfg.max_ngram_size):
            source = self.history[np.maximum(positions - shift, 0)]
            blocked = blocked | (positions < shift) | (source == DEAD)
            tokens[:, shift] = np.where(blocked, cfg.pad_id, source)

        # XOR the multiplied ids in one lookback at a time: after step i the running
        # value is the (i+1)-gram hash, each landing in its own prime-sized bucket.
        products = tokens[:, None, :] * cfg.multipliers[None, :, :]  # [T, L, n]
        rolling = products[..., 0]
        cols = []
        for i in range(1, cfg.max_ngram_size):
            rolling = np.bitwise_xor(rolling, products[..., i])
            cols.append(rolling[..., None] % cfg.primes[None, :, i - 1, :])  # [T, L, H]
        hashes = np.concatenate(cols, axis=-1)  # [T, L, (n-1)*H]
        return hashes + cfg.offsets.reshape(1, cfg.offsets.shape[0], -1)
