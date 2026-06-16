"""Matrix multiply — naive loop vs. optimized tiled GEMM.

Shows two approaches:
1. Naive matmul with explicit for-loop (one element per thread)
2. Tiled GEMM using al.tile_dot with simdgroup MMA
"""

import alloy as al
import numpy as np


# --- Approach 1: Naive matmul ---
# Each thread computes one element of C.
# Simple but slow — no tiling, no shared memory.
@al.kernel
def matmul_naive(
    a_ptr, b_ptr, c_ptr: al.output, M: al.constexpr, N: al.constexpr, K: al.constexpr
) -> None:
    row = al.program_id(0)
    col = al.program_id(1)
    acc = 0.0
    for i in range(K):
        acc += al.load(a_ptr + row * K + i) * al.load(b_ptr + i * N + col)
    al.store(c_ptr + row * N + col, acc)


# --- Approach 2: Tiled GEMM with simdgroup MMA ---
# Uses blocking, shared memory, and hardware matrix multiply.
@al.kernel
def matmul_tiled(
    a_ptr,
    b_ptr,
    c_ptr: al.output,
    M: al.constexpr,
    N: al.constexpr,
    K: al.constexpr,
    BLOCK_M: al.constexpr,
    BLOCK_N: al.constexpr,
    BLOCK_K: al.constexpr,
) -> None:
    pm = al.program_id(0)
    pn = al.program_id(1)
    rm = pm * BLOCK_M + al.arange(0, BLOCK_M)
    rn = pn * BLOCK_N + al.arange(0, BLOCK_N)
    rk = al.arange(0, BLOCK_K)
    a_ptrs = a_ptr + rm[:, None] * K + rk[None, :]
    b_ptrs = b_ptr + rk[:, None] * N + rn[None, :]
    acc = al.zeros((BLOCK_M, BLOCK_N), dtype=al.float32)
    for k in range(0, K, BLOCK_K):
        a = al.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] < K))
        b = al.load(b_ptrs, mask=(rk[:, None] < K) & (rn[None, :] < N))
        acc += al.tile_dot(a, b)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N
    c_ptrs = c_ptr + rm[:, None] * N + rn[None, :]
    al.store(c_ptrs, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


def main() -> None:
    np.random.seed(42)
    M, N, K = 128, 128, 128
    A = np.random.randn(M, K).astype(np.float32) * 0.1
    B = np.random.randn(K, N).astype(np.float32) * 0.1
    expected = A @ B

    # --- Naive ---
    print("=== Naive matmul MSL (first 20 lines) ===")
    msl = matmul_naive.compile_to_msl(M=M, N=N, K=K)
    for line in msl.split("\n")[:20]:
        print(f"  {line}")
    print("  ...")

    C_naive = np.zeros((M, N), dtype=np.float32)
    r = matmul_naive[(M, N)](A, B, C_naive, M=M, N=N, K=K)
    err = np.max(np.abs(np.array(r).reshape(M, N) - expected))
    print(f"\nNaive {M}x{N}x{K}: error={err:.2e}")
    assert err < 1e-2

    # --- Tiled ---
    BM, BN, BK = 32, 32, 16
    C_tiled = np.zeros((M, N), dtype=np.float32)
    grid_m = (M + BM - 1) // BM
    grid_n = (N + BN - 1) // BN
    r = matmul_tiled[(grid_m, grid_n)](
        A, B, C_tiled, M=M, N=N, K=K, BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK
    )
    err = np.max(np.abs(np.array(r).reshape(M, N) - expected))
    print(f"Tiled {M}x{N}x{K}: error={err:.2e}")
    assert err < 1e-2

    # --- Builtin al.dot ---
    r = al.dot(A, B)
    err = np.max(np.abs(np.array(r) - expected))
    print(f"al.dot {M}x{N}x{K}: error={err:.2e}")
    assert err < 1e-2

    print("\nAll PASSED")


if __name__ == "__main__":
    main()
