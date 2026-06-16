"""GPU histogram — count value frequencies with atomics.

Demonstrates: atomic_add, masked loads, constexpr block sizing.
"""

import alloy as al
import numpy as np

BLOCK = 256


@al.kernel
def histogram(bin_indices, bins: al.output, N: al.constexpr, BLOCK_SIZE: al.constexpr) -> None:
    pid = al.program_id(0)
    offsets = pid * BLOCK_SIZE + al.arange(0, BLOCK_SIZE)
    mask = offsets < N
    bin_idx = al.load(bin_indices + offsets, mask=mask, other=-1.0)
    if bin_idx >= 0.0:
        al.atomic_add(bins, al.cast(bin_idx, al.int32), 1)


if __name__ == "__main__":
    N = 1_000_000
    NUM_BINS = 64

    data = np.random.rand(N).astype(np.float32)
    bin_indices = np.clip((data * NUM_BINS).astype(np.int32), 0, NUM_BINS - 1).astype(np.float32)
    bins = np.zeros(NUM_BINS, dtype=np.int32)

    grid = (N + BLOCK - 1) // BLOCK
    result = histogram[grid](bin_indices, bins, N=N, BLOCK_SIZE=BLOCK)
    bins = np.asarray(result)

    ref = np.bincount(bin_indices.astype(np.int32), minlength=NUM_BINS)
    max_diff = np.max(np.abs(bins - ref))
    print(f"Histogram of {N:,} values into {NUM_BINS} bins")
    print(f"Max bin count diff vs NumPy: {max_diff}")
    print(f"Total counted: {bins.sum()} (expected {N})")
    assert bins.sum() == N
    print("PASSED")
