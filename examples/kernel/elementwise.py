"""Elementwise operations with fused math.

Multiple math operations compile into a single fused GPU kernel
with no intermediate buffers — each thread loads once, computes
everything, and stores once.
"""

import alloy as al
import numpy as np


@al.kernel
def gelu(x_ptr, out_ptr: al.output, N: al.constexpr) -> None:
    pid = al.program_id(0)
    offs = pid * 1024 + al.arange(0, 1024)
    mask = offs < N
    x = al.load(x_ptr + offs, mask=mask)
    result = x * 0.5 * (1.0 + al.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
    al.store(out_ptr + offs, result, mask=mask)


@al.kernel
def sigmoid(x_ptr, out_ptr: al.output, N: al.constexpr) -> None:
    pid = al.program_id(0)
    offs = pid * 1024 + al.arange(0, 1024)
    mask = offs < N
    x = al.load(x_ptr + offs, mask=mask)
    al.store(out_ptr + offs, 1.0 / (1.0 + al.exp(-x)), mask=mask)


@al.kernel
def silu(x_ptr, out_ptr: al.output, N: al.constexpr) -> None:
    pid = al.program_id(0)
    offs = pid * 1024 + al.arange(0, 1024)
    mask = offs < N
    x = al.load(x_ptr + offs, mask=mask)
    al.store(out_ptr + offs, x * al.sigmoid(x), mask=mask)


def main() -> None:
    np.random.seed(42)
    N = 4096
    x = np.random.randn(N).astype(np.float32)
    out = np.zeros(N, dtype=np.float32)
    grid = (N + 1023) // 1024

    print("=== GELU kernel MSL ===")
    al.inspect(gelu, N=N)
    print()

    r = gelu[grid](x, out, N=N)
    x64 = x.astype(np.float64)
    expected = (x64 * 0.5 * (1.0 + np.tanh(0.7978845608 * (x64 + 0.044715 * x64**3)))).astype(
        np.float32
    )
    err = np.max(np.abs(np.array(r) - expected))
    print(f"GELU N={N}: error={err:.2e}")
    assert err < 1e-5

    out = np.zeros(N, dtype=np.float32)
    r = sigmoid[grid](x, out, N=N)
    err = np.max(np.abs(np.array(r) - 1.0 / (1.0 + np.exp(-x))))
    print(f"Sigmoid N={N}: error={err:.2e}")
    assert err < 1e-6

    out = np.zeros(N, dtype=np.float32)
    r = silu[grid](x, out, N=N)
    err = np.max(np.abs(np.array(r) - x * (1.0 / (1.0 + np.exp(-x)))))
    print(f"SiLU N={N}: error={err:.2e}")
    assert err < 1e-6

    print("All PASSED")


if __name__ == "__main__":
    main()
