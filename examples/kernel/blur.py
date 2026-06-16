"""2D box blur — image processing with Alloy.

Demonstrates: 2D grids, nested loops, conditional logic.
"""

import alloy as al
import numpy as np


@al.kernel
def blur(src, dst: al.output, W: al.constexpr, H: al.constexpr) -> None:
    x = al.program_id(0)
    y = al.program_id(1)
    acc = 0.0
    count = 0
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            nx = x + dx
            ny = y + dy
            if nx >= 0:
                if nx < W:
                    if ny >= 0:
                        if ny < H:
                            acc = acc + al.load(src + ny * W + nx)
                            count = count + 1
    al.store(dst + y * W + x, acc / count)


if __name__ == "__main__":
    W, H = 1024, 1024
    img = np.random.rand(H, W).astype(np.float32)

    result = blur[W, H](img.ravel(), W=W, H=H)
    out = np.asarray(result).reshape(H, W)

    y, x = 540, 960
    expected = img[y - 1 : y + 2, x - 1 : x + 2].mean()
    print(f"Center pixel [{y},{x}]: GPU={out[y, x]:.6f}, ref={expected:.6f}")
    assert abs(out[y, x] - expected) < 1e-5

    y, x = 0, 0
    expected = img[0:2, 0:2].mean()
    print(f"Corner pixel [{y},{x}]: GPU={out[y, x]:.6f}, ref={expected:.6f}")
    assert abs(out[y, x] - expected) < 1e-5
    print("PASSED")
