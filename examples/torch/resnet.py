"""Small ResNet through the Alloy torch.compile backend.

A GroupNorm ResNet for image classification, compiled to fused Metal kernels via
`torch.compile(model, backend="alloy")` and checked against eager. GroupNorm +
strided convolutions stand in for BatchNorm + max-pooling (the backend lowers
conv / group_norm / residual adds, not batch_norm / pooling).
"""

import torch
import torch.nn as nn
import alloy_torch  # noqa: F401  imports register the "alloy" backend


class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride=1, groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(groups, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, cout)
        self.proj = None
        if stride != 1 or cin != cout:
            self.proj = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride=stride, bias=False),
                nn.GroupNorm(groups, cout),
            )

    def forward(self, x):
        identity = x if self.proj is None else self.proj(x)
        h = torch.relu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return torch.relu(h + identity)


class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
        )
        self.layer1 = ResBlock(32, 32)
        self.layer2 = ResBlock(32, 64, stride=2)
        self.layer3 = ResBlock(64, 128, stride=2)
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.flatten(2).mean(-1)  # global average pool over the spatial grid
        return self.head(x)


def main() -> None:
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    model = ResNet().eval()
    x = torch.randn(4, 3, 32, 32)  # CIFAR-sized batch

    expected = model(x)
    compiled = torch.compile(model, backend="alloy")
    result = compiled(x)

    err = float((result - expected).abs().max())
    print(f"ResNet {tuple(x.shape)} -> {tuple(result.shape)}: max_abs_err={err:.2e}")
    assert torch.allclose(result, expected, rtol=1e-3, atol=1e-3)
    print("PASSED")


if __name__ == "__main__":
    main()
