"""The alloy-native forward against the torch oracle on the synthetic tiny model.

Cache rounding is off on both sides: the fp4/fp8 rounding kernels are bit-exact
(test_kernels) but the indexer's relu'd scores tie exactly at 0 and the top-k /
candidate selection of a random model flips on ~1e-3 GEMM noise, so rows are
compared individually and a small number of selection flips is tolerated.
"""

import numpy as np
import pytest

from alloy_server.gguf.split import SplitGGUF
from alloy_server.models.deepseek_v41.config import config_from_gguf_kv
from alloy_server.models.deepseek_v41.experts import EngramTables, ExpertStore
from alloy_server.models.deepseek_v41.model import DeepseekV41Model
from alloy_server.models.deepseek_v41.reference import ReferenceModel
from alloy_server.models.deepseek_v41.weights import ModelWeights
from tests.deepseek_v41.synth import SyntheticModel, tiny_config

ROW_TOL = 2e-2      # per-row max |err| / max |ref|
MIN_MATCH = 0.8     # fraction of rows that must be within ROW_TOL


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    cfg = tiny_config()
    synth = SyntheticModel(cfg)
    path = synth.write_gguf(tmp_path_factory.mktemp("gguf") / "tiny-00001-of-00001.gguf")
    split = SplitGGUF(path)
    read_cfg = config_from_gguf_kv(split.kv, split.shapes())
    weights = ModelWeights(split, read_cfg)
    layout_bytes = 2 * 256 * 144 + 256 * 210
    # 8 slots per arena: a whole layer fits, but the 3 layers sharing an arena evict each other
    experts = ExpertStore(split, read_cfg, budget_bytes=layout_bytes * 16, bypass_page_cache=False)
    engram = EngramTables(split, read_cfg)
    model = DeepseekV41Model(read_cfg, weights, experts, engram, max_seq_len=cfg.max_seq_len, rounding=False)
    return cfg, synth, model


def row_matches(got, ref):
    got = np.asarray(got, dtype=np.float32).reshape(ref.shape[0], -1)
    ref = np.asarray(ref, dtype=np.float32).reshape(ref.shape[0], -1)
    scale = max(float(np.abs(ref).max()), 1e-6)
    err = np.abs(got - ref).max(axis=1) / scale
    return err <= ROW_TOL, err


def assert_rows(got, ref, msg):
    ok, err = row_matches(got, ref)
    frac = float(ok.mean())
    assert frac >= MIN_MATCH, f"{msg}: only {frac:.0%} rows within tol; per-row err {np.round(err, 3)}"


def test_teacher_forced_layers_match_oracle(tiny):
    """Every layer's attention / MoE / engram output given the oracle's input stream."""
    cfg, synth, model = tiny
    rng = np.random.default_rng(11)
    ids = rng.integers(3, cfg.vocab_size, size=20, dtype=np.int64)
    oracle = ReferenceModel(cfg, synth.oracle, cfg.max_seq_len, rounding=False)
    _, ref_streams = oracle.forward(ids, 0, all_logits=True)
    teacher = [(s.numpy(), p.numpy()) for s, p in zip(ref_streams, oracle.pres)]
    trace: list[dict] = []
    model.forward(model.new_state(), ids, all_logits=True, trace=trace, teacher=teacher)
    for layer, rec in enumerate(trace):
        assert_rows(rec["attn"], oracle.attn_outs[layer].numpy(), f"layer {layer} attention")
        assert_rows(rec["moe"], oracle.moe_outs[layer].numpy(), f"layer {layer} moe")
        assert_rows(rec["x"], ref_streams[layer].numpy(), f"layer {layer} stream")
        if "engram" in rec:
            ok, _ = row_matches(rec["engram"], oracle.engram_outs[layer].numpy())
            assert ok.all(), f"layer {layer} engram"


def test_prefill_and_decode_match_oracle(tiny):
    cfg, synth, model = tiny
    rng = np.random.default_rng(11)
    ids = rng.integers(3, cfg.vocab_size, size=23, dtype=np.int64)
    oracle = ReferenceModel(cfg, synth.oracle, cfg.max_seq_len, rounding=False)
    ref_logits, _ = oracle.forward(ids, 0, all_logits=True)
    state = model.new_state()
    got = model.forward(state, ids[:20], all_logits=True)
    assert_rows(got, ref_logits[:20].numpy(), "one-shot prefill logits")
    steps = np.stack([model.forward(state, ids[pos : pos + 1])[0] for pos in range(20, 23)])
    assert_rows(steps, ref_logits[20:23].numpy(), "decode logits")


def test_chunked_prefill_matches_one_shot(tiny):
    cfg, synth, model = tiny
    rng = np.random.default_rng(12)
    ids = rng.integers(3, cfg.vocab_size, size=27, dtype=np.int64)
    one = model.forward(model.new_state(), ids, all_logits=True)
    state = model.new_state()
    chunks = []
    for chunk in (7, 9, 11):
        start = state.pos
        chunks.append(model.forward(state, ids[start : start + chunk], all_logits=True))
    assert_rows(np.concatenate(chunks), one, "chunked vs one-shot prefill")


def test_expert_arena_streams(tiny):
    cfg, synth, model = tiny
    assert all(a.slots < cfg.n_layers * cfg.n_routed_experts for a in model.experts.arenas.values())
    assert model.experts.misses > 0
