"""MLX 4-bit model support: load MLX-quantized safetensors (affine int4 group
quant) through the same HF skeleton + per-arch handlers as the GGUF path."""

from alloy_server.mlx.loader import load_mlx_causal_lm
from alloy_server.mlx.resolve import ResolvedMLX, mlx_quantization, resolve_mlx

__all__ = [
    "ResolvedMLX",
    "load_mlx_causal_lm",
    "mlx_quantization",
    "resolve_mlx",
]
