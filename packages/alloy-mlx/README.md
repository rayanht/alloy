# alloy-mlx

MLX-array interop for [Alloy](https://github.com/rayanht/alloy).

Pass MLX arrays directly into Alloy kernels (their storage is unified Apple
Silicon memory) and consume Alloy buffers as MLX arrays without a copy on the
supported layouts.

Requires macOS on Apple Silicon and the `alloy-metal` and `mlx` packages.
