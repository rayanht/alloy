"""Regression coverage for GEMM epilogue fusion with a broadcast-add mask.

The pattern matters for encoder-style attention with additive padding masks
(BERT, nomic-bert, etc.). The mask has shape `[B, 1, S, S]`, while the dot
anchor is emitted per head as `(S, S)`, so epilogue lowering must preserve the
broadcast stride instead of indexing the mask with the per-head output offset.
"""

from __future__ import annotations

import torch

import alloy_torch  # noqa: F401 — registers the alloy torch.compile backend






def _matmul_scale_mask(q: torch.Tensor, k: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """matmul + scale + broadcast-add."""
    scores = torch.matmul(q, k.transpose(-1, -2)) * (1.0 / (q.size(-1) ** 0.5))
    return scores + mask


def test_broadcast_add_after_scaled_matmul_eager_vs_alloy() -> None:
    """Same FX pattern, eager vs alloy.compile."""
    torch.manual_seed(0)
    batch, heads, seq, head_dim = 2, 12, 13, 64
    query = torch.randn(batch, heads, seq, head_dim, dtype=torch.float16, device="cpu")
    key = torch.randn(batch, heads, seq, head_dim, dtype=torch.float16, device="cpu")
    # Mask: batch element 1 has padding from position 4 onwards.
    mask = torch.zeros(batch, 1, seq, seq, dtype=torch.float16, device="cpu")
    mask[1, 0, :, 4:] = torch.finfo(torch.float16).min

    eager_out = _matmul_scale_mask(query, key, mask).to(torch.float32)
    compiled = torch.compile(_matmul_scale_mask, backend="alloy", dynamic=False)
    alloy_out = compiled(query, key, mask).to(torch.float32)

    # At the masked positions, the score must be the fp16 min sentinel
    # (-65504). Eager gets it right; alloy used to silently drop the
    # mask add when it was absorbed into the GEMM epilogue. After the
    # ScatterTransform fix, the result lands within fp16 rounding noise
    # of the sentinel (the saturated value's representable neighbors).
    max_diff = (eager_out - alloy_out).abs().max().item()
    assert max_diff < 10.0, (
        f"alloy compile diverges from eager — max abs diff {max_diff:.4f}"
    )
    # Sanity: at the masked positions, magnitude must be near the fp16
    # min sentinel (this is what proves the mask is applied, not just
    # that the unmasked numerics happen to be small).
    masked = alloy_out[1, 0, 0, 4:7]
    assert (masked < -60000.0).all(), (
        f"alloy fails to apply broadcast mask: got {masked.tolist()}"
    )
