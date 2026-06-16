"""Quantized modules for MLX 4-bit (affine int4 group quant) + the weight swap.

MLX stores each quantized weight as `weight` (uint32, 8 nibbles/word,
low-order-first), `scales` and `biases` (one per `group_size` along the
contraction dim); dequant is `weight = scale*q + bias`, q an unsigned nibble.
The kernels read qweight as 2 nibbles/byte in consecutive-weight order, which is
the uint32 buffer viewed little-endian — so the swap is a zero-cost reinterpret.
"""

from __future__ import annotations

import torch
from transformers.modeling_utils import PreTrainedModel

from alloy_server.gguf.quant import module_for_parameter, module_parent

MLX_Q4_MM = torch.ops.alloy.mlx_q4_mm.default
MLX_Q4_EMBEDDING = torch.ops.alloy.mlx_q4_embedding.default


def mlx_qweight_to_uint8(weight: torch.Tensor) -> torch.Tensor:
    """MLX packed weight (uint32, [N, in//8]) -> uint8 [N, in//2], a pure
    little-endian reinterpret (byte k//2 holds weights k, k+1)."""
    return weight.contiguous().view(torch.uint8)


class MLXQ4Linear(torch.nn.Module):
    """Linear backed by affine int4 group quant: qweight (out, in//2) uint8,
    fp16 scales/biases (out, in//group_size)."""

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        biases: torch.Tensor,
        bias: torch.nn.Parameter | None = None,
    ) -> None:
        super().__init__()
        validate_mlx_buffers(qweight, scales, biases, rows=out_features, dim=in_features, kind="linear")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.register_buffer("qweight", qweight.contiguous())
        self.register_buffer("scales", scales.contiguous())
        self.register_buffer("biases", biases.contiguous())
        if bias is not None:
            self.bias = bias
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape
        flat = x.reshape(-1, self.in_features).contiguous()
        out = MLX_Q4_MM(flat, self.qweight, self.scales, self.biases)
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*x_shape[:-1], self.out_features)


class MLXQ4Embedding(torch.nn.Module):
    """Embedding backed by affine int4 group quant."""

    def __init__(
        self,
        *,
        num_embeddings: int,
        embedding_dim: int,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        biases: torch.Tensor,
    ) -> None:
        super().__init__()
        validate_mlx_buffers(
            qweight, scales, biases, rows=num_embeddings, dim=embedding_dim, kind="embedding"
        )
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.register_buffer("qweight", qweight.contiguous())
        self.register_buffer("scales", scales.contiguous())
        self.register_buffer("biases", biases.contiguous())

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return MLX_Q4_EMBEDDING(input_ids, self.qweight, self.scales, self.biases)


def validate_mlx_buffers(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    *,
    rows: int,
    dim: int,
    kind: str,
) -> None:
    """Validate shapes/dtypes; group_size (the scale-column count) must be a multiple of 8."""
    if qweight.dtype is not torch.uint8:
        raise TypeError(f"MLX {kind} qweight must be uint8, got {qweight.dtype}")
    if scales.dtype is not torch.float16 or biases.dtype is not torch.float16:
        raise TypeError(f"MLX {kind} scales/biases must be float16")
    if tuple(qweight.shape) != (rows, dim // 2) or scales.shape != biases.shape or scales.shape[0] != rows:
        raise ValueError(
            f"MLX {kind} buffer shapes inconsistent: qweight {tuple(qweight.shape)}, "
            f"scales {tuple(scales.shape)} (rows={rows} dim={dim})"
        )
    groups = scales.shape[1]
    if groups == 0 or dim % groups != 0 or (dim // groups) % 8 != 0:
        raise ValueError(f"MLX {kind} group size {dim}/{groups} unsupported (must divide dim, multiple of 8)")


def replace_mlx_quantized_weight(
    model: torch.nn.Module,
    hf_weight_name: str,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
) -> str:
    if not hf_weight_name.endswith(".weight"):
        raise NotImplementedError(f"MLX quantized tensor does not map to a weight: {hf_weight_name}")
    module = module_for_parameter(model, hf_weight_name)
    module_path = hf_weight_name[: -len(".weight")]
    parent, child_name = module_parent(model, module_path)
    if isinstance(module, torch.nn.Linear):
        parent._modules[child_name] = MLXQ4Linear(
            in_features=module.in_features,
            out_features=module.out_features,
            qweight=qweight,
            scales=scales,
            biases=biases,
            bias=module.bias if module.bias is not None else None,
        )
        return "linear"
    if isinstance(module, torch.nn.Embedding):
        parent._modules[child_name] = MLXQ4Embedding(
            num_embeddings=module.num_embeddings,
            embedding_dim=module.embedding_dim,
            qweight=qweight,
            scales=scales,
            biases=biases,
        )
        return "embedding"
    raise NotImplementedError(
        f"MLX quantized tensor maps to unsupported module: {hf_weight_name} -> {type(module).__name__}"
    )


def tie_mlx_output_embeddings(model: PreTrainedModel) -> None:
    """Tie a quantized input embedding into the LM head (the head reuses the
    embedding's packed buffers)."""
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if not isinstance(input_embeddings, MLXQ4Embedding):
        model.tie_weights()
        return
    if output_embeddings is not None and not isinstance(output_embeddings, torch.nn.Linear):
        raise TypeError(
            f"Tied MLX embeddings require a Linear output head, got {type(output_embeddings).__name__}"
        )
    model.set_output_embeddings(
        MLXQ4Linear(
            in_features=input_embeddings.embedding_dim,
            out_features=input_embeddings.num_embeddings,
            qweight=input_embeddings.qweight,
            scales=input_embeddings.scales,
            biases=input_embeddings.biases,
        )
    )
