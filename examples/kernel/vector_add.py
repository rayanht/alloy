"""Vector addition on Metal GPU."""

import alloy as al
import numpy as np


@al.kernel
def vector_add(x_ptr, y_ptr, out_ptr: al.output, N: al.constexpr) -> None:
    pid = al.program_id(0)
    block_size = 1024
    offsets = pid * block_size + al.arange(0, block_size)
    mask = offsets < N
    x = al.load(x_ptr + offsets, mask=mask)
    y = al.load(y_ptr + offsets, mask=mask)
    al.store(out_ptr + offsets, x + y, mask=mask)


def main() -> None:
    N = 8192
    x = np.random.randn(N).astype(np.float32)
    y = np.random.randn(N).astype(np.float32)
    out = np.zeros(N, dtype=np.float32)

    print("=== Generated MSL ===")
    al.inspect(vector_add, N=N)
    print()

    grid = (N + 1023) // 1024
    result = vector_add[grid](x, y, out, N=N)

    expected = x + y
    max_error = np.max(np.abs(np.array(result) - expected))
    print(f"N = {N}")
    print(f"Max error: {max_error:.2e}")
    assert max_error < 1e-5
    print("PASSED")


if __name__ == "__main__":
    main()
