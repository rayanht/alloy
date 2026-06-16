"""N-body gravitational simulation.

Demonstrates: for-loops, autotuning, constexpr block sizing.
"""

from typing import cast

import alloy as al
import numpy as np
from alloy._runtime.alloy_buffer import AlloyBuffer, materialize_many


@al.kernel
def nbody_forces(
    px, py, mass, ax: al.output, ay: al.output, N: al.constexpr, BLOCK_SIZE: al.constexpr
) -> None:
    pid = al.program_id(0)
    offs = pid * BLOCK_SIZE + al.arange(0, BLOCK_SIZE)
    mask = offs < N
    xi = al.load(px + offs, mask=mask)
    yi = al.load(py + offs, mask=mask)
    acc_x = 0.0
    acc_y = 0.0
    for j in range(N):
        xj = al.load(px + j)
        yj = al.load(py + j)
        mj = al.load(mass + j)
        dx = xj - xi
        dy = yj - yi
        dist_sq = dx * dx + dy * dy + 1e-6
        inv_dist = al.rsqrt(dist_sq)
        inv_dist3 = inv_dist * inv_dist * inv_dist
        acc_x = acc_x + mj * dx * inv_dist3
        acc_y = acc_y + mj * dy * inv_dist3
    al.store(ax + offs, acc_x, mask=mask)
    al.store(ay + offs, acc_y, mask=mask)


@al.kernel
def integrate(
    ax,
    ay,
    dt_buf,
    px_in,
    py_in,
    vx_in,
    vy_in,
    px: al.output,
    py: al.output,
    vx: al.output,
    vy: al.output,
    N: al.constexpr,
) -> None:
    pid = al.program_id(0)
    block_size = 1024
    offs = pid * block_size + al.arange(0, block_size)
    mask = offs < N
    dt = al.load(dt_buf + 0)
    cur_vx = al.load(vx_in + offs, mask=mask) + al.load(ax + offs, mask=mask) * dt
    cur_vy = al.load(vy_in + offs, mask=mask) + al.load(ay + offs, mask=mask) * dt
    al.store(vx + offs, cur_vx, mask=mask)
    al.store(vy + offs, cur_vy, mask=mask)
    al.store(px + offs, al.load(px_in + offs, mask=mask) + cur_vx * dt, mask=mask)
    al.store(py + offs, al.load(py_in + offs, mask=mask) + cur_vy * dt, mask=mask)


if __name__ == "__main__":
    N = 512
    DT = 0.001
    STEPS = 100

    px = np.random.randn(N).astype(np.float32) * 10
    py = np.random.randn(N).astype(np.float32) * 10
    vx = np.zeros(N, dtype=np.float32)
    vy = np.zeros(N, dtype=np.float32)
    mass = np.ones(N, dtype=np.float32)
    dt_buf = np.array([DT], dtype=np.float32)
    force_block_size = 256
    force_grid = (N + force_block_size - 1) // force_block_size
    int_grid = (N + 1023) // 1024

    print(f"N-body simulation: {N} bodies, {STEPS} steps")

    for step in range(STEPS):
        ax, ay = cast(
            tuple[AlloyBuffer, AlloyBuffer],
            nbody_forces[force_grid](px, py, mass, N=N, BLOCK_SIZE=force_block_size),
        )
        px, py, vx, vy = cast(
            tuple[AlloyBuffer, AlloyBuffer, AlloyBuffer, AlloyBuffer],
            integrate[int_grid](ax, ay, dt_buf, px, py, vx, vy, N=N),
        )
        materialize_many((px, py, vx, vy))

    al.sync()
    vx_np = np.asarray(vx)
    vy_np = np.asarray(vy)
    px_np = np.asarray(px)
    py_np = np.asarray(py)
    energy = 0.5 * np.sum(mass * (vx_np**2 + vy_np**2))
    print(f"Final kinetic energy: {energy:.4f}")
    print(f"Position spread: x=[{px_np.min():.2f}, {px_np.max():.2f}], y=[{py_np.min():.2f}, {py_np.max():.2f}]")
