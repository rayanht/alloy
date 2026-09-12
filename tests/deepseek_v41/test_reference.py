"""The torch oracle's cache handling: a one-shot prefill must equal prefill + decode."""

import numpy as np
import torch

from alloy_server.models.deepseek_v41.reference import ReferenceModel, fp4_round, fp8_round, round_e2m1
from tests.deepseek_v41.synth import SyntheticModel, tiny_config


def test_fp4_rounding_grid():
    x = torch.tensor([0.0, 0.2, 0.25, 0.26, 0.75, 1.24, 1.25, 1.26, 1.75, 2.5, 2.6, 3.5, 5.0, 5.1, 6.0, -2.5, -0.25])
    expect = torch.tensor([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 3.0, 4.0, 4.0, 6.0, 6.0, -2.0, -0.0])
    torch.testing.assert_close(round_e2m1(x), expect)


def test_block_rounding_is_idempotent():
    x = torch.randn(4, 64) * 3
    y = fp8_round(x, 32)
    torch.testing.assert_close(fp8_round(y, 32), y)
    z = fp4_round(x, 16, scale_e4m3=True)
    torch.testing.assert_close(fp4_round(z, 16, scale_e4m3=True), z)
    z = fp4_round(x, 32, scale_e4m3=False)
    torch.testing.assert_close(fp4_round(z, 32, scale_e4m3=False), z)


def test_prefill_equals_prefill_plus_decode():
    cfg = tiny_config()
    synth = SyntheticModel(cfg)
    rng = np.random.default_rng(0)
    ids = rng.integers(3, cfg.vocab_size, size=23, dtype=np.int64)

    one_shot = ReferenceModel(cfg, synth.oracle, cfg.max_seq_len)
    logits_full, streams_full = one_shot.forward(ids, 0, all_logits=True)

    stepped = ReferenceModel(cfg, synth.oracle, cfg.max_seq_len)
    logits_a, streams_a = stepped.forward(ids[:20], 0, all_logits=True)
    torch.testing.assert_close(logits_a, logits_full[:20], rtol=1e-4, atol=1e-4)
    for pos in range(20, 23):
        logits_b, streams_b = stepped.forward(ids[pos : pos + 1], pos)
        torch.testing.assert_close(streams_b[-1][0], streams_full[-1][pos], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(logits_b[0], logits_full[pos], rtol=1e-4, atol=1e-4)
