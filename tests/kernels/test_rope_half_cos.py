"""rms_norm_rope_strided: the HALF_COS table must be bit-identical to the full
table. The rotate_half layout makes emb = cat(freqs, freqs), so the cos/sin
table's two halves are equal; the self-cat strip rewrite drops the duplication
and runs the kernel with HALF_COS=1 (half-width table, stride HALF_ROT, second
half re-reads the first). Covers both full rotary (ROTARY_DIM==0) and partial
rotary (ROTARY_DIM>0)."""

import numpy as np
import pytest

from alloy.std import rms_norm_rope_strided
from alloy._runtime.convert import to_alloy_buffer


def _run(x_flat, weight, cos_tab, sin_tab, *, H, S, head_dim, rot, cos_rows, half):
    out = to_alloy_buffer(np.zeros((H * S, head_dim), np.float32))
    rms_norm_rope_strided[(S, H)](
        to_alloy_buffer(x_flat),
        to_alloy_buffer(weight),
        to_alloy_buffer(cos_tab),
        to_alloy_buffer(sin_tab),
        out,
        BH=H,
        HEADS_PER_BATCH=H,
        SEQ_LEN=S,
        HEAD_DIM=head_dim,
        ROTARY_DIM=rot if rot != head_dim else 0,
        X_OFFSET=0,
        X_BATCH_STRIDE=H * S * head_dim,
        X_HEAD_STRIDE=S * head_dim,
        X_SEQ_STRIDE=head_dim,
        COS_ROWS=cos_rows,
        EPS=1e-6,
        HALF_COS=half,
    )
    return np.asarray(out).copy()


@pytest.mark.parametrize("head_dim,rot", [(8, 8), (128, 128), (256, 64), (8, 4)])
def test_half_cos_matches_full(head_dim, rot):
    H, S = 2, 3
    half_rot = rot // 2
    rng = np.random.default_rng(0)
    x = rng.standard_normal((H * S * head_dim,)).astype(np.float32)
    weight = rng.standard_normal((head_dim,)).astype(np.float32)
    # Arbitrary rotary angles per (seq, rotary-pair).
    freqs = rng.standard_normal((S, half_rot)).astype(np.float32)

    half_cos = np.cos(freqs)
    half_sin = np.sin(freqs)
    # Full table duplicates the half along the rotary axis (the self-cat).
    full_cos = np.concatenate([half_cos, half_cos], axis=1)
    full_sin = np.concatenate([half_sin, half_sin], axis=1)

    out_full = _run(
        x, weight, full_cos, full_sin,
        H=H, S=S, head_dim=head_dim, rot=rot, cos_rows=S, half=0,
    )
    out_half = _run(
        x, weight, half_cos, half_sin,
        H=H, S=S, head_dim=head_dim, rot=rot, cos_rows=S, half=1,
    )
    np.testing.assert_array_equal(out_half, out_full)
