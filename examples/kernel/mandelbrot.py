"""Mandelbrot set — massively parallel iteration on 2D grid.

Demonstrates: 2D grids, for loops, constexpr parameters,
divergent iteration counts per thread.
"""

import alloy as al
import numpy as np


@al.kernel
def mandelbrot(out: al.output, W: al.constexpr, H: al.constexpr, MAX_ITER: al.constexpr) -> None:
    px = al.program_id(0)
    py = al.program_id(1)

    cr = -2.0 + px * 3.0 / W
    ci = -1.5 + py * 3.0 / H

    zr = 0.0
    zi = 0.0
    count = 0
    for i in range(MAX_ITER):
        zr2 = zr * zr
        zi2 = zi * zi
        if zr2 + zi2 < 4.0:
            zi = 2.0 * zr * zi + ci
            zr = zr2 - zi2 + cr
            count = count + 1

    al.store(out + py * W + px, count * 1.0 / MAX_ITER)


if __name__ == "__main__":
    W, H = 1920, 1080
    MAX_ITER = 256
    out = np.zeros(H * W, dtype=np.float32)

    result = mandelbrot[W, H](out, W=W, H=H, MAX_ITER=MAX_ITER)
    img = np.array(result).reshape(H, W)

    n_inside = (img >= 1.0).sum()
    n_outside = (img < 1.0).sum()
    print(f"Mandelbrot {W}x{H}, max_iter={MAX_ITER}")
    print(f"Inside set: {n_inside:,} pixels ({100 * n_inside / (W * H):.1f}%)")
    print(f"Outside set: {n_outside:,} pixels")
    print(f"Average iterations: {img.mean() * MAX_ITER:.1f}")
