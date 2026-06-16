"""Transformer encoder block through the Alloy torch.compile backend.

Pre-norm multi-head self-attention (via scaled_dot_product_attention) + a GELU
MLP with residual connections, compiled to fused Metal kernels via
`torch.compile(model, backend="alloy")` and checked against eager.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import alloy_torch  # noqa: F401  imports register the "alloy" backend


class EncoderBlock(nn.Module):
    def __init__(self, d_model=256, n_heads=8, d_ff=1024):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x):
        B, T, C = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, n_heads, T, d_head)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, T, C)
        x = x + self.proj(attn)
        x = x + self.ff(self.norm2(x))
        return x


def main() -> None:
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    model = EncoderBlock().eval()
    x = torch.randn(2, 64, 256)  # (batch, seq_len, d_model)

    expected = model(x)
    compiled = torch.compile(model, backend="alloy")
    result = compiled(x)

    err = float((result - expected).abs().max())
    print(f"Transformer {tuple(x.shape)} -> {tuple(result.shape)}: max_abs_err={err:.2e}")
    assert torch.allclose(result, expected, rtol=1e-3, atol=1e-3)
    print("PASSED")


if __name__ == "__main__":
    main()
