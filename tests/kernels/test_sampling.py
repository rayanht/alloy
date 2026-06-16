"""GPU correctness tests for the on-GPU categorical sampler (std.sample_categorical).

The sampler replaces the decode argmax: temperature / top-k / top-p / min-p applied
on-GPU via Gumbel-max, with a counter-based RNG keyed on (seed, position, vocab index).
`position` stands in for the decode step's cache_position, so varying it across calls
mimics autoregressive decode.
"""

import numpy as np
import pytest

from alloy._compiler.dtypes import float32, int64
from alloy._dispatch.buf_utils import _alloc_aligned, _alloc_scratch
from alloy._runtime.convert import to_alloy_buffer
from alloy.std import sample_categorical
from alloy.std.sampling import (
    SAMPLE_SPLITS,
    sample_categorical_combine,
    sample_categorical_split,
)


def _sample(logits_np, position, seed, temperature, top_p, top_k, min_p):
    """Production split-K path: split (n_splits vocab slices) + combine, with
    n_splits chosen exactly as the generator does (1 when a filter needs the
    global bisection, SAMPLE_SPLITS otherwise)."""
    m = logits_np.shape[0]
    filtered = top_p < 1.0 or top_k >= 1.0 or min_p > 0.0
    n_splits = 1 if filtered else SAMPLE_SPLITS
    params = np.array([temperature, top_p, top_k, min_p, float(n_splits)], dtype=np.float32)
    pv = _alloc_scratch((m, SAMPLE_SPLITS), float32)
    pi = _alloc_scratch((m, SAMPLE_SPLITS), float32)
    out = _alloc_aligned((m,), int64)
    sample_categorical_split[(m, SAMPLE_SPLITS)](
        to_alloy_buffer(logits_np.astype(np.float32)),
        to_alloy_buffer(np.array([position] * m, dtype=np.int64)),
        to_alloy_buffer(np.array([seed], dtype=np.int64)),
        to_alloy_buffer(params),
        pv,
        pi,
    )
    sample_categorical_combine[(m,)](pv, pi, out)
    out.sync()
    return np.asarray(out.numpy).reshape(m)


def _sample_single_tg(logits_np, position, seed, temperature, top_p, top_k, min_p):
    """The single-threadgroup kernel (constrained loop / spec verify path)."""
    m = logits_np.shape[0]
    out = _alloc_aligned((m,), int64)
    sample_categorical(
        to_alloy_buffer(logits_np.astype(np.float32)),
        to_alloy_buffer(np.array([position] * m, dtype=np.int64)),
        to_alloy_buffer(np.array([seed], dtype=np.int64)),
        to_alloy_buffer(np.array([temperature, top_p, top_k, min_p], dtype=np.float32)),
        out,
    )
    out.sync()
    return np.asarray(out.numpy).reshape(m)


def test_split_and_single_tg_agree():
    # The split-K plan path and the single-TG constrained/spec path must give the
    # same token across modes (greedy, pure-temperature, top-k, top-p).
    rng = np.random.default_rng(123)
    logits = rng.standard_normal((1, 257)).astype(np.float32)
    for temp, tp, tk, mp in [(0.0, 1.0, 0.0, 0.0), (0.8, 1.0, 0.0, 0.0),
                             (1.0, 1.0, 5.0, 0.0), (1.0, 0.7, 0.0, 0.0)]:
        for pos in (0, 31):
            a = int(_sample(logits, pos, pos * 5 + 2, temp, tp, tk, mp)[0])
            b = int(_sample_single_tg(logits, pos, pos * 5 + 2, temp, tp, tk, mp)[0])
            assert a == b, f"split {a} != single {b} for ({temp},{tp},{tk},{mp}) pos={pos}"


def _softmax(x, t=1.0):
    z = (x / t) - (x / t).max()
    e = np.exp(z)
    return e / e.sum()


def test_greedy_temperature_zero_is_exact_argmax():
    # temperature=0 must be bit-identical to argmax (lowest-index tie-break),
    # independent of seed/position — this is what lets one plan serve both.
    rng = np.random.default_rng(7)
    for _ in range(20):
        logits = rng.standard_normal((1, 200)).astype(np.float32)
        gold = int(logits.argmax())
        for pos in (0, 17, 999):
            tok = int(_sample(logits, pos, pos * 3 + 1, 0.0, 1.0, 0.0, 0.0)[0])
            assert tok == gold, f"greedy != argmax: {tok} vs {gold}"

    # exact-tie tie-break: lowest index wins (matches argmax_last_dim).
    tie = np.zeros((1, 8), dtype=np.float32)
    tie[0, 3] = tie[0, 6] = 5.0
    assert int(_sample(tie, 42, 123, 0.0, 1.0, 0.0, 0.0)[0]) == 3


def test_deterministic_per_seed_position():
    logits = np.random.default_rng(0).standard_normal((1, 16)).astype(np.float32)
    a = _sample(logits, 5, 1234, 1.0, 1.0, 0.0, 0.0)[0]
    b = _sample(logits, 5, 1234, 1.0, 1.0, 0.0, 0.0)[0]
    assert a == b


def test_top_k_1_is_greedy():
    logits = np.random.default_rng(1).standard_normal((1, 32)).astype(np.float32)
    gold = int(logits.argmax())
    picks = {int(_sample(logits, p, p * 7 + 1, 1.0, 1.0, 1.0, 0.0)[0]) for p in range(40)}
    assert picks == {gold}


@pytest.mark.slow
def test_matches_softmax_distribution():
    n, ns = 8, 6000
    logits = np.random.default_rng(2).standard_normal((1, n)).astype(np.float32)
    probs = _softmax(logits[0], t=1.0)
    counts = np.zeros(n)
    for p in range(ns):
        counts[int(_sample(logits, p, 99, 1.0, 1.0, 0.0, 0.0)[0])] += 1
    assert np.abs(counts / ns - probs).max() < 0.02


@pytest.mark.slow
def test_temperature_increases_entropy():
    n, ns = 8, 4000
    logits = np.random.default_rng(3).standard_normal((1, n)).astype(np.float32)

    def ent(d):
        return -(d * np.log(d + 1e-12)).sum()

    def emp(temp):
        c = np.zeros(n)
        for p in range(ns):
            c[int(_sample(logits, p, 7, temp, 1.0, 0.0, 0.0)[0])] += 1
        return c / ns

    assert ent(emp(4.0)) > ent(emp(1.0))


@pytest.mark.slow
def test_top_k_restricts_support():
    n, k = 12, 3
    logits = np.random.default_rng(4).standard_normal((1, n)).astype(np.float32)
    topk_ids = set(np.argsort(logits[0])[-k:].tolist())
    seen = {int(_sample(logits, p, 3, 1.0, 1.0, float(k), 0.0)[0]) for p in range(4000)}
    assert seen.issubset(topk_ids)


@pytest.mark.slow
def test_top_p_nucleus_restricts_support():
    n, p_thresh = 8, 0.6
    logits = np.random.default_rng(5).standard_normal((1, n)).astype(np.float32)
    probs = _softmax(logits[0])
    order = np.argsort(probs)[::-1]
    keep_n = int(np.searchsorted(np.cumsum(probs[order]), p_thresh) + 1)
    nucleus = set(order[:keep_n].tolist())
    seen = {int(_sample(logits, p, 11, 1.0, p_thresh, 0.0, 0.0)[0]) for p in range(6000)}
    assert seen.issubset(nucleus)


@pytest.mark.slow
def test_min_p_restricts_support():
    # min_p keeps tokens with prob >= min_p * max_prob.
    n, min_p = 10, 0.3
    logits = np.random.default_rng(6).standard_normal((1, n)).astype(np.float32)
    probs = _softmax(logits[0])
    keep = set(np.nonzero(probs >= min_p * probs.max())[0].tolist())
    seen = {int(_sample(logits, p, 13, 1.0, 1.0, 0.0, min_p)[0]) for p in range(5000)}
    assert seen.issubset(keep)
