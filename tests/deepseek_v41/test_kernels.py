"""GPU kernels for DeepSeek-V4.1 against the torch oracle (`reference.py`)."""

import gguf
import numpy as np
import torch

from alloy._compiler.dtypes import float16, float32, int32, uint8
from alloy._dispatch.buf_utils import _alloc_aligned
from alloy._runtime._metal_ext import gpu_sync
from alloy._runtime.alloy_buffer import materialize_many
from alloy.std.deepseek import (
    ds_block_mask_scatter,
    ds_block_max,
    ds_block_round,
    ds_compress_pool,
    ds_engram_gate,
    ds_grouped_out_proj,
    ds_hc_mixes,
    ds_hc_post,
    ds_hc_pre,
    ds_hc_sinkhorn,
    ds_index_score,
    ds_rope_pairs,
    ds_router_topk,
    ds_sparse_attn,
    ds_topk_select,
    ds_zero_u8,
)
from alloy_server.gguf import kquant
from alloy_server.models.deepseek_v41 import reference


def buf(arr, dtype):
    arr = np.ascontiguousarray(arr)
    b = _alloc_aligned(tuple(arr.shape), dtype)
    b.numpy[:] = arr
    return b


def out(shape, dtype):
    return _alloc_aligned(tuple(shape), dtype)


def run(*bufs):
    materialize_many(list(bufs))
    gpu_sync()
    return [np.array(b.numpy) for b in bufs]


def idx_bits(n):
    b = 1
    while (1 << b) < n:
        b += 1
    return b + 1


def test_block_round_modes():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((6, 128)) * 3).astype(np.float32)
    xt = torch.from_numpy(x)
    zero = buf(np.zeros(1, dtype=np.int32), int32)
    for mode, block, expect in (
        (0, 32, reference.fp8_round(xt, 32)),
        (1, 32, reference.fp4_round(xt, 32, scale_e4m3=False)),
        (2, 16, reference.fp4_round(xt, 16, scale_e4m3=True)),
    ):
        o = out((6, 128), float32)
        ds_block_round[(6,)](buf(x, float32), o.slice(0, 0, 1), zero, o, N=128, MODE=mode, BLOCK=block)
        (got,) = run(o)
        np.testing.assert_array_equal(got, expect.numpy(), err_msg=f"mode {mode}")
        o16 = out((6, 128), float16)
        ds_block_round[(6,)](buf(x, float32), o16.slice(0, 0, 1), zero, o16, N=128, MODE=mode, BLOCK=block, OUT_F16=1)
        (got16,) = run(o16)
        np.testing.assert_array_equal(got16.astype(np.float32), expect.numpy(), err_msg=f"mode {mode} f16")


def test_rope_pairs_forward_inverse():
    rng = np.random.default_rng(1)
    rows, heads, d, rot = 5, 3, 128, 64
    x = rng.standard_normal((rows, heads * d)).astype(np.float32)
    freqs = reference.precompute_freqs_cis(rot, 64, 32, 160000.0, 16.0, 32.0, 1.0)
    pos = np.array([0, 7, 8, 33, 63], dtype=np.int32)
    cos = freqs.real.numpy().astype(np.float32)
    sin = freqs.imag.numpy().astype(np.float32)
    xt = torch.from_numpy(x).view(rows, heads, d)
    f = freqs[torch.from_numpy(pos).long()]
    expect = torch.cat([xt[..., :-rot], reference.apply_rotary(xt[..., -rot:], f)], dim=-1).reshape(rows, -1)
    o = out((rows, heads * d), float32)
    ds_rope_pairs[(rows, heads)](buf(x, float32), buf(cos, float32), buf(sin, float32), buf(pos, int32), o,
                                 WIDTH=heads * d, HEAD_DIM=d, ROT=rot, NUM_THREADS=rot // 2)
    (got,) = run(o)
    np.testing.assert_allclose(got, expect.numpy(), rtol=1e-5, atol=1e-5)
    inv = out((rows, heads * d), float32)
    ds_rope_pairs[(rows, heads)](o, buf(cos, float32), buf(sin, float32), buf(pos, int32), inv,
                                 WIDTH=heads * d, HEAD_DIM=d, ROT=rot, INVERSE=1, NUM_THREADS=rot // 2)
    (back,) = run(inv)
    np.testing.assert_allclose(back, x, rtol=1e-5, atol=1e-5)


def test_sparse_attn_two_sources():
    rng = np.random.default_rng(2)
    rows, heads, d = 4, 4, 128
    win_rows, comp_rows, win_n, comp_n = 10, 12, 6, 5
    q = rng.standard_normal((rows, heads * d)).astype(np.float32)
    win_kv = rng.standard_normal((win_rows, d)).astype(np.float16)
    comp_kv = rng.standard_normal((comp_rows, d)).astype(np.float16)
    win_idx = rng.integers(-1, win_rows, size=(rows, win_n)).astype(np.int32)
    comp_idx = rng.integers(-1, comp_rows, size=(rows, comp_n)).astype(np.int32)
    win_idx[3] = -1  # a row with nothing visible in the window
    sink = rng.standard_normal(heads).astype(np.float32)
    scale = d**-0.5
    kv_all = torch.cat([torch.from_numpy(win_kv), torch.from_numpy(comp_kv)]).float()
    ci = torch.from_numpy(comp_idx)
    idx_all = torch.cat([torch.from_numpy(win_idx), torch.where(ci >= 0, ci + win_rows, -1)], dim=1)
    expect = reference.sparse_attn(torch.from_numpy(q).view(rows, heads, d), kv_all, torch.from_numpy(sink), idx_all, scale)
    o = out((rows, heads * d), float32)
    ds_sparse_attn[(rows, heads)](buf(q, float32), buf(win_kv, float16), buf(win_idx, int32), buf(comp_kv, float16),
                                  buf(comp_idx, int32), buf(sink, float32), o,
                                  HEADS=heads, HEAD_DIM=d, WIN_N=win_n, COMP_N=comp_n, SCALE=scale)
    (got,) = run(o)
    np.testing.assert_allclose(got, expect.reshape(rows, -1).numpy(), rtol=1e-4, atol=1e-4)


def test_index_score_and_topk():
    rng = np.random.default_rng(3)
    rows, heads, d, n = 6, 2, 128, 70
    q = rng.standard_normal((rows, heads * d)).astype(np.float32)
    k = rng.standard_normal((n, d)).astype(np.float16)
    w = rng.standard_normal((rows, heads)).astype(np.float32)
    visible = np.array([1, 5, 20, 33, 70, 70], dtype=np.int32)
    mask = (rng.random((rows, n)) > 0.3).astype(np.uint8)
    qt = torch.from_numpy(q).view(rows, heads, d)
    score = torch.einsum("shd,td->sht", qt, torch.from_numpy(k).float())
    score = (score.relu() * torch.from_numpy(w).unsqueeze(-1)).sum(1)
    causal = torch.arange(n)[None, :] < torch.from_numpy(visible)[:, None]
    expect_plain = score.masked_fill(~causal, -1e30)
    expect_masked = expect_plain.masked_fill(torch.from_numpy(mask) == 0, -1e30)
    for has_mask, expect in ((0, expect_plain), (1, expect_masked)):
        o = out((rows, n), float32)
        ds_index_score[(rows, (n + 63) // 64)](buf(q, float32), buf(k, float16), buf(w, float32), buf(visible, int32),
                                              buf(mask, uint8), buf(np.array([n], dtype=np.int32), int32), o,
                                              HEADS=heads, HEAD_DIM=d, CAP=n, HAS_MASK=has_mask)
        (got,) = run(o)
        np.testing.assert_allclose(got, expect.numpy(), rtol=1e-4, atol=1e-3)
    # top-k with ties (relu zeros) and -inf rows; lowest index wins ties
    sc = expect_masked.numpy().copy()
    sc[2, 3:9] = 0.5  # a tie block
    K = 8
    o = out((rows, K), int32)
    ds_topk_select[(rows,)](buf(sc, float32), buf(np.array([n], dtype=np.int32), int32), o, CAP=n, K=K, IDX_BITS=idx_bits(n))
    (got,) = run(o)
    exp_idx = reference.topk_lowest_index(torch.from_numpy(sc), K)
    exp_sorted = torch.where(torch.from_numpy(sc).gather(1, exp_idx) > -1e30, exp_idx, -1)
    for r in range(rows):
        e = exp_sorted[r].numpy()
        g = got[r]
        assert sorted(e[e >= 0].tolist()) == sorted(g[g >= 0].tolist()), r
        assert (e < 0).sum() == (g < 0).sum(), r
        valid = g[g >= 0]
        assert np.all(np.diff(valid) > 0), "ascending"


def test_candidate_blocks():
    rng = np.random.default_rng(4)
    rows, n, bs, top_b = 4, 37, 4, 3
    nb = (n + bs - 1) // bs
    score = rng.standard_normal((rows, n)).astype(np.float32)
    visible = np.array([3, 10, 22, 37], dtype=np.int32)
    unreachable = np.arange(n)[None, :] >= visible[:, None]
    score[unreachable] = -1e30
    score_ref = torch.from_numpy(score).masked_fill(torch.from_numpy(unreachable), -torch.inf)
    expect = reference.select_candidate_blocks(score_ref, torch.from_numpy(visible)[:, None], top_b, bs)
    bm = out((rows, nb), float32)
    count = buf(np.array([n], dtype=np.int32), int32)
    ds_block_max[(rows, (nb + 31) // 32)](buf(score, float32), buf(visible, int32), count, bm, CAP=n, BLOCK_SIZE=bs, NB_CAP=nb)
    sel = out((rows, top_b), int32)
    ds_topk_select[(rows,)](bm, buf(np.array([nb], dtype=np.int32), int32), sel, CAP=nb, K=top_b, IDX_BITS=idx_bits(nb))
    mask = out((rows, n), uint8)
    ds_zero_u8[((rows * n + 1023) // 1024,)](mask, N=rows * n)
    ds_block_mask_scatter[(rows, top_b)](sel, mask.slice(0, 0, 1), mask, CAP=n, BLOCK_SIZE=bs, B=top_b, NUM_THREADS=bs)
    (got,) = run(mask)
    np.testing.assert_array_equal(got.astype(bool), expect.numpy())


def test_hc_mixes_sinkhorn_pre_post():
    rng = np.random.default_rng(5)
    rows, hc, d = 3, 4, 256
    mix = (2 + hc) * hc
    x = rng.standard_normal((rows, hc, d)).astype(np.float32)
    fn = (rng.standard_normal((mix, hc * d)) / 32).astype(np.float32)
    scale = np.array([0.1, 0.12, 0.15], dtype=np.float32)
    base = (rng.standard_normal(mix) * 0.5).astype(np.float32)
    eps, hc_eps, iters = 1e-20, 1e-6, 20
    flat = torch.from_numpy(x).flatten(1)
    mixes_ref = (flat @ torch.from_numpy(fn).T) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + eps)
    pre_ref, post_ref, comb_ref = reference.hc_split_sinkhorn(
        mixes_ref, torch.from_numpy(scale), torch.from_numpy(base), hc, iters, hc_eps,
    )
    mixes = out((rows, mix), float32)
    ds_hc_mixes[(rows,)](buf(x.reshape(rows, -1), float32), buf(fn, float32), mixes, HC_D=hc * d, MIX=mix, EPS=eps)
    pre, post, comb = out((rows, hc), float32), out((rows, hc), float32), out((rows, hc, hc), float32)
    ds_hc_sinkhorn[(rows,)](mixes, buf(scale, float32), buf(base, float32), pre, post, comb, HC=hc, ITERS=iters, EPS=hc_eps)
    got_mix, got_pre, got_post, got_comb = run(mixes, pre, post, comb)
    np.testing.assert_allclose(got_mix, mixes_ref.numpy(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(got_pre, pre_ref.numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(got_post, post_ref.numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(got_comb, comb_ref.numpy(), rtol=1e-4, atol=1e-5)
    y = rng.standard_normal((rows, d)).astype(np.float32)
    o_pre = out((rows, d), float32)
    ds_hc_pre[(rows,)](buf(x, float32), pre, o_pre, HC=hc, D=d)
    o_post = out((rows, hc, d), float32)
    ds_hc_post[(rows, hc)](buf(y, float32), buf(x, float32), post, comb, o_post, HC=hc, D=d)
    got_pre_o, got_post_o = run(o_pre, o_post)
    np.testing.assert_allclose(
        got_pre_o, reference.ReferenceModel.hc_pre(torch.from_numpy(x), pre_ref).numpy(), rtol=1e-4, atol=1e-4,
    )
    np.testing.assert_allclose(
        got_post_o,
        reference.ReferenceModel.hc_post(torch.from_numpy(y), torch.from_numpy(x), post_ref, comb_ref).numpy(),
        rtol=1e-4, atol=1e-4,
    )


def test_engram_gate():
    rng = np.random.default_rng(6)
    rows, hc, d = 3, 4, 256
    x = rng.standard_normal((rows, hc, d)).astype(np.float32)
    kv = rng.standard_normal((rows, (hc + 1) * d)).astype(np.float32)
    qw = (1 + 0.1 * rng.standard_normal((hc, d))).astype(np.float32)
    kw = (1 + 0.1 * rng.standard_normal((hc, d))).astype(np.float32)
    eps = 1e-20
    kvt = torch.from_numpy(kv)
    key, value = kvt.split([hc * d, d], dim=-1)
    key = key.unflatten(-1, (hc, d))
    h = torch.from_numpy(x)
    rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps)
    dot = (h * torch.from_numpy(qw * kw) * key).sum(-1) * rstd * d**-0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
    expect = h + gate.unsqueeze(-1) * value.unsqueeze(1)
    o = out((rows, hc, d), float32)
    ds_engram_gate[(rows, hc)](buf(x, float32), buf(kv, float32), buf(qw * kw, float32), o, HC=hc, D=d, EPS=eps)
    (got,) = run(o)
    np.testing.assert_allclose(got, expect.numpy(), rtol=1e-4, atol=1e-4)


def test_router_topk():
    rng = np.random.default_rng(7)
    rows, e, k = 5, 8, 2
    logits = rng.standard_normal((rows, e)).astype(np.float32)
    bias = (rng.standard_normal(e) * 0.5).astype(np.float32)
    lt = torch.from_numpy(logits)
    scores = torch.nn.functional.softplus(lt).sqrt()
    idx_ref = reference.topk_lowest_index(scores + torch.from_numpy(bias), k)
    w_ref = scores.gather(1, idx_ref)
    w_ref = w_ref / (w_ref.sum(-1, keepdim=True) + 1e-20) * 1.5
    idx, w = out((rows, k), int32), out((rows, k), float32)
    ds_router_topk[(rows,)](buf(logits, float32), buf(bias, float32), idx, w, E=e, K=k, ROUTE_SCALE=1.5)
    got_idx, got_w = run(idx, w)
    np.testing.assert_array_equal(got_idx, idx_ref.numpy())
    np.testing.assert_allclose(got_w, w_ref.numpy(), rtol=1e-5, atol=1e-6)


def test_compress_pool():
    rng = np.random.default_rng(8)
    ng, ratio, d = 5, 2, 256
    kv = rng.standard_normal((ng * ratio, d)).astype(np.float32)
    score = rng.standard_normal((ng * ratio, d)).astype(np.float32)
    expect = (torch.from_numpy(kv).view(ng, ratio, d) * torch.from_numpy(score).view(ng, ratio, d).softmax(1)).sum(1)
    o = out((ng, d), float32)
    ds_compress_pool[(ng,)](buf(kv, float32), buf(score, float32), o, RATIO=ratio, D=d)
    (got,) = run(o)
    np.testing.assert_allclose(got, expect.numpy(), rtol=1e-5, atol=1e-5)


def test_grouped_out_proj():
    rng = np.random.default_rng(9)
    m, g, k_g, n_g = 5, 2, 512, 128
    a = rng.standard_normal((m, g * k_g)).astype(np.float32)
    w = (rng.standard_normal((g * n_g, k_g)) / 20).astype(np.float32)
    packed = kquant.quantize_q4_k(w)
    deq = gguf.quants.dequantize(packed, gguf.GGMLQuantizationType.Q4_K)
    expect = torch.einsum(
        "sgd,grd->sgr", torch.from_numpy(a).view(m, g, k_g), torch.from_numpy(deq).view(g, n_g, k_g),
    ).reshape(m, -1)
    o = out((m, g * n_g), float32)
    ds_grouped_out_proj[((m + 15) // 16, g * n_g // 64)](buf(a, float32), buf(packed, uint8), o, K_G=k_g, N_G=n_g)
    (got,) = run(o)
    np.testing.assert_allclose(got, expect.numpy(), rtol=2e-2, atol=1e-2)
