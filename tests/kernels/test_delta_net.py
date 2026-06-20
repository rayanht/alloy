"""GPU correctness tests for the 2-stage chunked gated delta rule (the prefill
delta-net kernel: chunked_gdr_stage1 + chunked_gdr_stage2).

Regression coverage for two emitter bugs the kernel exercises:
  * 1D-reduction sub-tile mask — the per-chunk `sum(g_c)` over a (C,) tile must
    NOT sum threadgroup lanes beyond C (the cooperative load fills them with the
    NEXT chunk's g); without it the carried state is scaled by exp(Σg_next).
  * Multi-head address — the cooperative (C,DK) q/k load must keep the `h_kv*DK`
    base offset (the NV=16 case below); dropping it made every head read head 0.
"""

import pytest
import torch
import torch.nn.functional as F

from alloy.std.delta_net import chunked_gdr_stage1, chunked_gdr_stage2
from alloy._compiler.dtypes import float32
from alloy._dispatch.buf_utils import _alloc_aligned
from alloy._runtime._metal_ext import gpu_sync
from alloy._runtime.alloy_buffer import materialize_many


def _torch_chunked_ref(ql, kl, v, g, beta, C):
    """Per-chunk inverse-based gated delta rule (ql/kl pre-normed, ql pre-scaled).
    ql,kl,v: (S, D); g,beta: (S,). Returns (out (S,D), final_state (D,D))."""
    S, D = ql.shape
    state = torch.zeros(D, D, dtype=torch.float64)
    outs = []
    for c in range(S // C):
        sl = slice(c * C, (c + 1) * C)
        qc, kc, vc = ql[sl].double(), kl[sl].double(), v[sl].double()
        gc_, bc = g[sl].double(), beta[sl].double()
        gcum = torch.cumsum(gc_, 0)
        dm = torch.tril(torch.exp(gcum[:, None] - gcum[None, :]))
        kb = kc * bc[:, None]
        A = -torch.tril(kb @ kc.T * dm, -1)
        T = torch.linalg.inv(torch.eye(C, dtype=torch.float64) - A)
        W = T @ (kb * gcum[:, None].exp())
        U = T @ (vc * bc[:, None])
        attn = torch.tril(qc @ kc.T) * dm
        v_new = U - W @ state
        out = (qc * gcum[:, None].exp()) @ state + attn @ v_new
        glast = gcum[-1]
        state = state * glast.exp() + (kc * (glast - gcum)[:, None].exp()).T @ v_new
        outs.append(out)
    return torch.cat(outs, 0).float(), state.float()


def _to_buf(t):
    b = _alloc_aligned((t.numel(),), float32)
    b.numpy[:] = t.reshape(-1).contiguous().numpy()
    return b


def _run_2stage(B, NV, DK, DV, S, C, DVB, real_len=None, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(B, S, NV, DK)
    k = torch.randn(B, S, NV, DK)
    v = torch.randn(B, S, NV, DV)
    g = -torch.rand(B, S, NV) * 0.3
    beta = torch.rand(B, S, NV)
    ql = F.normalize(q, dim=-1, eps=1e-6) * (DK ** -0.5)
    kl = F.normalize(k, dim=-1, eps=1e-6)

    hrl = 1 if real_len is not None else 0
    rl = real_len if real_len is not None else S
    qa, ka, va, ga, ba = _to_buf(ql), _to_buf(kl), _to_buf(v), _to_buf(g), _to_buf(beta)
    rla = _to_buf(torch.tensor([float(rl)]))
    NC = S // C
    W_o = _alloc_aligned((B * NV * S * DK,), float32)
    qg_o = _alloc_aligned((B * NV * S * DK,), float32)
    kd_o = _alloc_aligned((B * NV * S * DK,), float32)
    T_o = _alloc_aligned((B * NV * NC * C * C,), float32)
    at_o = _alloc_aligned((B * NV * NC * C * C,), float32)
    chunked_gdr_stage1[(NC, B * NV)](
        qa, ka, ga, ba, rla, W_o, T_o, at_o, qg_o, kd_o,
        BATCH=B, S=S, NV=NV, DK=DK, DV=DV, NK=NV, C=C, HAS_REAL_LEN=hrl)
    rec = _alloc_aligned((B * NV * DK * DV,), float32)
    rec.numpy[:] = 0.0
    outk = _alloc_aligned((B * S * NV * DV,), float32)
    outk.numpy[:] = 0.0
    ret = chunked_gdr_stage2[(B * NV * (DV // DVB),)](
        va, ba, ga, rla, W_o, T_o, at_o, qg_o, kd_o, rec, outk,
        BATCH=B, S=S, NV=NV, DK=DK, DV=DV, C=C, DV_BLOCK=DVB, HAS_REAL_LEN=hrl)
    materialize_many([ret, rec, outk])
    gpu_sync()
    o_k = torch.from_numpy(outk.numpy.copy()).reshape(B, S, NV, DV)
    s_k = torch.from_numpy(rec.numpy.copy()).reshape(B, NV, DK, DV)
    return ql, kl, v, g, beta, o_k, s_k


@pytest.mark.parametrize("NV,S,C,DVB", [(1, 16, 8, 16), (2, 24, 8, 8), (16, 128, 8, 8)])
def test_2stage_chunked(NV, S, C, DVB):
    """Stage1 (parallel intra-chunk) + stage2 (DV-blocked scan). The NV=16 case
    guards the multi-head cooperative-load address (the `h_kv*DK` base offset
    must reach each head, not head 0)."""
    d = 128 if NV == 16 else 16
    ql, kl, v, g, beta, o_k, s_k = _run_2stage(1, NV, d, d, S, C, DVB)
    for h in range(NV):
        o_ref, s_ref = _torch_chunked_ref(
            ql[0, :, h], kl[0, :, h], v[0, :, h], g[0, :, h], beta[0, :, h], C)
        assert (o_ref - o_k[0, :, h]).abs().max().item() < 3e-3, f"head {h} out"
        assert (s_ref - s_k[0, h]).abs().max().item() < 1e-2, f"head {h} state"


def test_2stage_padded_prefill():
    """HAS_REAL_LEN: g/beta zeroed past real_len makes padding a no-op so the
    post-scan state is the real-len state and out[:real_len] is correct."""
    real_len = 64
    ql, kl, v, g, beta, o_k, s_k = _run_2stage(1, 16, 128, 128, 128, 8, 8, real_len=real_len)
    for h in range(16):
        o_ref, s_ref = _torch_chunked_ref(
            ql[0, :real_len, h], kl[0, :real_len, h], v[0, :real_len, h],
            g[0, :real_len, h], beta[0, :real_len, h], 8)
        assert (o_ref - o_k[0, :real_len, h]).abs().max().item() < 3e-3, f"head {h} out[:rl]"
        assert (s_ref - s_k[0, h]).abs().max().item() < 1e-2, f"head {h} state"
