"""Row-wise softmax, two ways.

Demonstrates writing a softmax kernel using Alloy's low-level primitives
(load/store/arange/masks) and compares it to the builtin `al.softmax`.
"""

import alloy as al
import numpy as np


# --- Approach 1: Manual kernel with Alloy primitives ---
# Each program handles one row. Uses threadgroup reductions.
@al.kernel
def softmax_manual(x_ptr, out_ptr: al.output, M: al.constexpr, N: al.constexpr) -> None:
    row = al.program_id(0)
    col = al.arange(0, 1024)
    mask = col < N

    x = al.load(x_ptr + row * N + col, mask=mask, other=-1e30)
    m = al.max(x)
    e = al.exp(x - m)
    s = al.sum(e)
    al.store(out_ptr + row * N + col, e / s, mask=mask)


# --- Approach 2: Builtin op (production use) ---
# al.softmax generates an optimized kernel with threadgroup reductions,
# shared memory, and simdgroup operations — much faster for large N.


def main() -> None:
    np.random.seed(42)
    M, N = 128, 512
    x = np.random.randn(M, N).astype(np.float32)

    # Show the generated MSL for the manual kernel
    print("=== Manual softmax MSL (first 20 lines) ===")
    msl = softmax_manual.compile_to_msl(M=M, N=N)
    for line in msl.split("\n")[:20]:
        print(f"  {line}")
    print("  ...")

    result = np.asarray(al.softmax(x))

    exp_x = np.exp(x - x.max(axis=1, keepdims=True))
    expected = exp_x / exp_x.sum(axis=1, keepdims=True)
    max_error = np.max(np.abs(result - expected))
    max_sum_error = np.max(np.abs(result.sum(axis=1) - 1.0))

    print(f"\nal.softmax({M}x{N})")
    print(f"Max row sum error: {max_sum_error:.2e}")
    print(f"Max value error: {max_error:.2e}")
    assert max_error < 1e-5
    print("PASSED")


if __name__ == "__main__":
    main()
