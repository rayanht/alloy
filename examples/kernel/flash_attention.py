"""FlashAttention — online softmax with no N*N matrix.

Shows the complete FlashAttention algorithm using Alloy primitives.
Each thread streams over all K/V positions, maintaining a running max
and sum for numerically stable online softmax.
"""

import alloy as al
import numpy as np


@al.kernel
def flash_attn(q, k, v, out: al.output, N: al.constexpr, D: al.constexpr) -> None:
    """Online softmax attention — no N*N matrix materialized.

    2D grid: each thread handles one output element O[row, col].
    """
    row = al.program_id(0)
    col = al.program_id(1)
    scale = al.rsqrt(D * 1.0)

    m = -1e30
    l = 0.0  # noqa: E741
    o = 0.0

    for j in range(N):
        dot = 0.0
        for d in range(D):
            dot += al.load(q + row * D + d) * al.load(k + j * D + d)
        s = dot * scale

        m_new = al.maximum(m, s)
        alpha = al.exp(m - m_new)
        p = al.exp(s - m_new)

        l = l * alpha + p  # noqa: E741
        o = o * alpha + p * al.load(v + j * D + col)
        m = m_new

    al.store(out + row * D + col, o / l)


def numpy_attention(Q, K, V):
    D = Q.shape[-1]
    S = Q @ K.T / np.sqrt(D)
    S_max = S.max(axis=-1, keepdims=True)
    P = np.exp(S - S_max)
    P = P / P.sum(axis=-1, keepdims=True)
    return P @ V


def main() -> None:
    np.random.seed(42)
    N, D = 64, 32
    Q = np.random.randn(N, D).astype(np.float32) * 0.3
    K_mat = np.random.randn(N, D).astype(np.float32) * 0.3
    V = np.random.randn(N, D).astype(np.float32) * 0.3
    expected = numpy_attention(Q, K_mat, V)

    # --- Manual FlashAttention ---
    print("=== FlashAttention (online softmax) ===")
    print("Generated kernel (first 20 lines):")
    msl = flash_attn.compile_to_msl(N=N, D=D)
    for line in msl.split("\n")[:20]:
        print(f"  {line}")
    print("  ...\n")

    out = np.zeros((N, D), dtype=np.float32)
    result = flash_attn[(N, D)](Q, K_mat, V, out, N=N, D=D)
    err = np.max(np.abs(np.array(result).reshape(N, D) - expected))
    print(f"  N={N} D={D}: error={err:.2e}")
    assert err < 1e-3

    # --- Builtin al.attention ---
    print("\n=== al.attention() (tiled, simdgroup MMA) ===")
    result2 = al.attention(Q, K_mat, V)
    err2 = np.max(np.abs(np.array(result2) - expected))
    print(f"  N={N} D={D}: error={err2:.2e}")
    assert err2 < 1e-3

    print("\nAll PASSED")


if __name__ == "__main__":
    main()
