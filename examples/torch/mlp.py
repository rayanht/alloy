"""MLP through the Alloy torch.compile backend.

A multi-layer perceptron (Linear + LayerNorm + GELU) compiled to fused Metal
kernels via `torch.compile(model, backend="alloy")` and checked against eager.
"""

import torch
import torch.nn as nn
import alloy_torch  # noqa: F401  imports register the "alloy" backend


class MLP(nn.Module):
    def __init__(self, d_in=256, d_hidden=512, d_out=10, depth=3):
        super().__init__()
        layers, d = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(d, d_hidden), nn.LayerNorm(d_hidden), nn.GELU()]
            d = d_hidden
        layers.append(nn.Linear(d, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def main() -> None:
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    model = MLP().eval()
    x = torch.randn(32, 256)

    expected = model(x)
    compiled = torch.compile(model, backend="alloy")
    result = compiled(x)

    err = float((result - expected).abs().max())
    print(f"MLP {tuple(x.shape)} -> {tuple(result.shape)}: max_abs_err={err:.2e}")
    assert torch.allclose(result, expected, rtol=1e-3, atol=1e-4)
    print("PASSED")


if __name__ == "__main__":
    main()
