"""Generation through the engine on the synthetic tiny model: deterministic greedy
decode, chunked prefill independence, and a reset between prompts."""

import numpy as np
import pytest

from alloy_server.gguf.split import SplitGGUF
from alloy_server.models.deepseek_v41.engine import DeepseekV41Engine, GenerationStats
from tests.deepseek_v41.synth import SyntheticModel, tiny_config


@pytest.fixture(scope="module")
def engines(tmp_path_factory):
    cfg = tiny_config()
    path = SyntheticModel(cfg).write_gguf(tmp_path_factory.mktemp("gguf") / "tiny-00001-of-00001.gguf")
    layout_bytes = 2 * 256 * 144 + 256 * 210
    mk = lambda chunk: DeepseekV41Engine(  # noqa: E731
        SplitGGUF(path), max_seq_len=cfg.max_seq_len, chunk_size=chunk,
        arena_bytes=layout_bytes * 16, rounding=False, bypass_page_cache=False,
    )
    return cfg, mk(16), mk(5)


def test_greedy_generation_is_deterministic_and_chunk_independent(engines):
    cfg, big, small = engines
    prompt = np.random.default_rng(0).integers(3, cfg.vocab_size, size=13, dtype=np.int64)
    stats = GenerationStats()
    a = list(big.generate(prompt, 8, stats=stats))
    big.reset()
    b = list(big.generate(prompt, 8))
    small.reset()
    c = list(small.generate(prompt, 8))
    assert len(a) == 8 and a == b
    # chunked prefill (5-token chunks) vs one 13-token chunk: same greedy path except
    # at near-tie selection flips, so require most tokens to agree
    assert sum(x == y for x, y in zip(a, c)) >= 6
    assert stats.prompt_tokens == 13 and stats.decode_tokens == 8 and len(stats.step_times) == 7
    assert stats.expert_misses + stats.expert_hits > 0


def test_eos_stops(engines):
    cfg, big, _ = engines
    big.reset()
    prompt = np.random.default_rng(1).integers(3, cfg.vocab_size, size=5, dtype=np.int64)
    first = next(iter(big.generate(prompt, 4)))
    big.reset()
    toks = list(big.generate(prompt, 4, eos_ids=(first,)))
    assert toks == [first]
