"""EngramHasher against a direct port of the reference `NgramHashState.forward`."""

import numpy as np
import torch

from alloy_server.models.deepseek_v41.engram import EngramHasher
from tests.deepseek_v41.synth import tiny_config


def reference_hashes(cfg, input_ids: torch.Tensor, start_pos: int, cache: torch.Tensor, token_mask=None):
    e = cfg.engram
    primes = torch.as_tensor(e.primes)
    offsets = torch.as_tensor(e.offsets.reshape(e.offsets.shape[0], -1))
    multipliers = torch.as_tensor(e.multipliers)
    token_map = torch.as_tensor(e.token_map)
    batch, seqlen = input_ids.shape
    compressed = token_map[input_ids]
    if token_mask is not None:
        compressed = torch.where(token_mask, compressed, -1)
    cache[:batch, start_pos : start_pos + seqlen] = compressed
    positions = torch.arange(start_pos, start_pos + seqlen).expand(batch, seqlen)
    tokens, blocked = [], torch.zeros_like(positions, dtype=torch.bool)
    for shift in range(e.max_ngram_size):
        source = cache[:batch].gather(1, (positions - shift).clamp_min(0))
        blocked = blocked | (positions < shift) | (source == -1)
        tokens.append(torch.where(blocked, e.pad_id, source))
    tokens = torch.stack(tokens, dim=-1)
    products = tokens.unsqueeze(2) * multipliers
    rolling, hashes = products[..., 0], []
    for i in range(1, e.max_ngram_size):
        rolling = torch.bitwise_xor(rolling, products[..., i])
        hashes.append(rolling.unsqueeze(-1) % primes[:, i - 1])
    return torch.cat(hashes, dim=-1) + offsets


def test_hasher_matches_reference_prefill_then_decode():
    cfg = tiny_config()
    rng = np.random.default_rng(3)
    ids = rng.integers(0, cfg.vocab_size, size=40, dtype=np.int64)
    mask = rng.random(40) > 0.2
    mask[:3] = True
    hasher = EngramHasher(cfg.engram, cfg.max_seq_len)
    cache = torch.empty(1, cfg.max_seq_len, dtype=torch.int64)

    ours = hasher.hash_ids(ids[:25], 0, mask[:25])
    ref = reference_hashes(cfg, torch.as_tensor(ids[:25]).unsqueeze(0), 0, cache, torch.as_tensor(mask[:25]).unsqueeze(0))
    np.testing.assert_array_equal(ours, ref[0].numpy())
    for pos in range(25, 40):
        ours = hasher.hash_ids(ids[pos : pos + 1], pos, mask[pos : pos + 1])
        ref = reference_hashes(cfg, torch.as_tensor(ids[pos : pos + 1]).unsqueeze(0), pos, cache, torch.as_tensor(mask[pos : pos + 1]).unsqueeze(0))
        np.testing.assert_array_equal(ours, ref[0].numpy())
    assert ours.shape == (1, 2, cfg.engram.n_hash_cols)
    # every row lands inside its layer's table
    for layer_idx, rows in enumerate(cfg.engram.num_embeddings):
        assert ours[:, layer_idx].max() < rows
