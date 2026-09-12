"""Synthetic deepseek41 GGUF round trip: config + tensors read back through SplitGGUF."""

import dataclasses

import numpy as np

from alloy_server.gguf.split import SplitGGUF
from alloy_server.models.deepseek_v41.config import config_from_gguf_kv
from tests.deepseek_v41.synth import SyntheticModel, tiny_config


def test_roundtrip(tmp_path):
    cfg = tiny_config()
    model = SyntheticModel(cfg)
    path = model.write_gguf(tmp_path / "tiny-00001-of-00001.gguf")
    split = SplitGGUF(path)
    assert split.architecture == "deepseek41"
    read = config_from_gguf_kv(split.kv, split.shapes())
    # GGUF floats are f32
    expect = dataclasses.replace(
        cfg, engram=None, norm_eps=float(np.float32(cfg.norm_eps)), hc_eps=float(np.float32(cfg.hc_eps)),
    )
    assert dataclasses.replace(read, engram=None) == expect
    for name, a in vars(read.engram).items():
        b = vars(cfg.engram)[name]
        if isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        else:
            assert a == b, name
    for name, oracle in model.oracle.items():
        got = split.dequantized(name)
        assert got.shape == tuple(oracle.shape), name
        np.testing.assert_allclose(got, oracle.numpy(), rtol=0, atol=0, err_msg=name)
    assert split.packed_rows("blk.0.ffn_down_exps.weight").shape == (8, 256, 210)
    assert split.packed_rows("blk.1.ffn_down_exps.weight").shape == (8, 256, 144)
