"""Quantized GGUF modules and the weight-installation helpers.

The `GGUFQ*` Linear/Embedding modules hold GGUF-native packed weights and route
their forward through the `alloy.gguf_q*_mm` / `gguf_q*_embedding` Metal kernels.
All four formats share two table-driven bases (`GGUFQuantLinear` /
`GGUFQuantEmbedding`): a `QuantFormat` declares the format's buffers (name, dtype,
column count) + kernel ops, and the eight public class names are thin subclasses
that pin one `FORMAT`. The names are kept (not collapsed) so `isinstance` checks,
`mtp.py`, and embedding-tying stay unchanged.

The `replace_*` helpers swap an HF `Linear`/`Embedding` for the matching quantized
module at load time; `tie_quantized_output_embeddings` mirrors the input embedding
into a tied LM head.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import gguf
import numpy as np
import torch
from transformers.modeling_utils import PreTrainedModel

GGUFQuantization = Literal["q4_k", "q5_0", "q6_k", "q8_0"]

GGUF_Q8_0_MM = torch.ops.alloy.gguf_q8_0_mm.default
GGUF_Q4_K_MM = torch.ops.alloy.gguf_q4_k_mm.default
GGUF_Q5_0_MM = torch.ops.alloy.gguf_q5_0_mm.default
GGUF_Q8_0_EMBEDDING = torch.ops.alloy.gguf_q8_0_embedding.default
GGUF_Q4_K_EMBEDDING = torch.ops.alloy.gguf_q4_k_embedding.default
GGUF_Q5_0_EMBEDDING = torch.ops.alloy.gguf_q5_0_embedding.default
GGUF_Q6_K_MM = torch.ops.alloy.gguf_q6_k_mm.default
GGUF_Q6_K_EMBEDDING = torch.ops.alloy.gguf_q6_k_embedding.default


@dataclass(frozen=True)
class QuantBuffer:
    """One packed buffer of a quant format: its attribute name, dtype, and the
    column count as a function of the contraction dim (in_features for Linear,
    embedding_dim for Embedding)."""

    name: str
    dtype: torch.dtype
    cols: Callable[[int], int]


@dataclass(frozen=True)
class QuantFormat:
    name: GGUFQuantization
    align: int  # the contraction dim must be divisible by this
    mm_op: Callable
    embed_op: Callable
    buffers: tuple[QuantBuffer, ...]

    @property
    def buffer_names(self) -> tuple[str, ...]:
        return tuple(b.name for b in self.buffers)


QUANT_FORMATS: dict[GGUFQuantization, QuantFormat] = {
    "q8_0": QuantFormat("q8_0", 32, GGUF_Q8_0_MM, GGUF_Q8_0_EMBEDDING, (
        QuantBuffer("qweight", torch.int8, lambda k: k),
        QuantBuffer("scales", torch.float16, lambda k: k // 32),
    )),
    "q5_0": QuantFormat("q5_0", 32, GGUF_Q5_0_MM, GGUF_Q5_0_EMBEDDING, (
        QuantBuffer("qweight", torch.uint8, lambda k: k // 2),
        QuantBuffer("qhigh", torch.uint8, lambda k: k // 2),
        QuantBuffer("scales", torch.float16, lambda k: k // 32),
    )),
    "q4_k": QuantFormat("q4_k", 256, GGUF_Q4_K_MM, GGUF_Q4_K_EMBEDDING, (
        QuantBuffer("blocks", torch.uint8, lambda k: (k // 256) * 144),
    )),
    "q6_k": QuantFormat("q6_k", 256, GGUF_Q6_K_MM, GGUF_Q6_K_EMBEDDING, (
        QuantBuffer("qweight", torch.uint8, lambda k: (k // 256) * 210),
    )),
}


def validate_quant_buffers(
    fmt: QuantFormat, buffers: dict[str, torch.Tensor], *, rows: int, dim: int, kind: str
) -> None:
    if set(buffers) != set(fmt.buffer_names):
        raise ValueError(
            f"{fmt.name} {kind} expects buffers {fmt.buffer_names}, got {tuple(buffers)}"
        )
    if dim % fmt.align != 0:
        raise ValueError(f"{fmt.name} {kind} dim {dim} not divisible by {fmt.align}")
    for b in fmt.buffers:
        t = buffers[b.name]
        if t.dtype is not b.dtype:
            raise TypeError(
                f"{fmt.name} {kind} buffer {b.name!r} requires {b.dtype}, got {t.dtype}"
            )
        if t.ndim != 2:
            raise ValueError(
                f"{fmt.name} {kind} buffer {b.name!r} must be rank-2, got {tuple(t.shape)}"
            )
        expected = (rows, b.cols(dim))
        if tuple(t.shape) != expected:
            raise ValueError(
                f"{fmt.name} {kind} buffer {b.name!r} shape {tuple(t.shape)} != {expected} "
                f"(rows={rows} dim={dim})"
            )


class GGUFQuantLinear(torch.nn.Module):
    """Linear backed by GGUF-native packed weights. Subclasses pin one `FORMAT`,
    which selects the buffers + the `gguf_q*_mm` kernel."""

    FORMAT: QuantFormat

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        bias: torch.nn.Parameter | None = None,
        **buffers: torch.Tensor,
    ) -> None:
        super().__init__()
        validate_quant_buffers(
            self.FORMAT, buffers, rows=out_features, dim=in_features, kind="linear"
        )
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        for b in self.FORMAT.buffers:
            self.register_buffer(b.name, buffers[b.name].contiguous())
        if bias is not None:
            self.bias = bias
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape
        flat = x.reshape(-1, self.in_features).contiguous()
        out = self.FORMAT.mm_op(flat, *(self._buffers[b.name] for b in self.FORMAT.buffers))
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*x_shape[:-1], self.out_features)


class GGUFQuantEmbedding(torch.nn.Module):
    """Embedding backed by GGUF-native packed weights. Subclasses pin one
    `FORMAT`, which selects the buffers + the `gguf_q*_embedding` kernel."""

    FORMAT: QuantFormat

    def __init__(
        self,
        *,
        num_embeddings: int,
        embedding_dim: int,
        embed_scale: float = 1.0,
        **buffers: torch.Tensor,
    ) -> None:
        super().__init__()
        validate_quant_buffers(
            self.FORMAT, buffers, rows=num_embeddings, dim=embedding_dim, kind="embedding"
        )
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.embed_scale = float(embed_scale)
        for b in self.FORMAT.buffers:
            self.register_buffer(b.name, buffers[b.name].contiguous())

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.FORMAT.embed_op(input_ids, *(self._buffers[b.name] for b in self.FORMAT.buffers))
        if self.embed_scale != 1.0:
            out = out * self.embed_scale
        return out


class GGUFQ8_0Linear(GGUFQuantLinear):
    """Linear backed by normalized GGUF Q8_0 row blocks (int8 qweight + fp16 scales)."""

    FORMAT = QUANT_FORMATS["q8_0"]


class GGUFQ5_0Linear(GGUFQuantLinear):
    """Linear backed by normalized GGUF Q5_0 row blocks (K-sequential nibble qweight,
    bit-packed qhigh, fp16 scales — see `split_q5_0_weight`)."""

    FORMAT = QUANT_FORMATS["q5_0"]


class GGUFQ4_KLinear(GGUFQuantLinear):
    """Linear backed by GGUF-native Q4_K 144-byte superblocks."""

    FORMAT = QUANT_FORMATS["q4_k"]


class GGUFQ6_KLinear(GGUFQuantLinear):
    """Linear backed by GGUF Q6_K 210-byte row blocks."""

    FORMAT = QUANT_FORMATS["q6_k"]


class GGUFQ8_0Embedding(GGUFQuantEmbedding):
    """Embedding backed by normalized GGUF Q8_0 row blocks."""

    FORMAT = QUANT_FORMATS["q8_0"]


class GGUFQ5_0Embedding(GGUFQuantEmbedding):
    """Embedding backed by normalized GGUF Q5_0 row blocks."""

    FORMAT = QUANT_FORMATS["q5_0"]


class GGUFQ4_KEmbedding(GGUFQuantEmbedding):
    """Embedding backed by GGUF-native Q4_K 144-byte superblocks."""

    FORMAT = QUANT_FORMATS["q4_k"]


class GGUFQ6_KEmbedding(GGUFQuantEmbedding):
    """Embedding backed by GGUF Q6_K row blocks."""

    FORMAT = QUANT_FORMATS["q6_k"]


def quant_buffer_names(module: GGUFQuantLinear | GGUFQuantEmbedding) -> tuple[str, ...]:
    """The packed-buffer attribute names a quantized module carries — drives the
    `allowed_missing_keys` whitelist without an isinstance chain."""
    return module.FORMAT.buffer_names


def tensor_quantization(tensor: gguf.ReaderTensor) -> GGUFQuantization | None:
    tensor_type = int(tensor.tensor_type)
    if tensor_type == int(gguf.GGMLQuantizationType.Q4_K):
        return "q4_k"
    if tensor_type == int(gguf.GGMLQuantizationType.Q5_0):
        return "q5_0"
    if tensor_type == int(gguf.GGMLQuantizationType.Q6_K):
        return "q6_k"
    if tensor_type == int(gguf.GGMLQuantizationType.Q8_0):
        return "q8_0"
    if tensor_type in {
        int(gguf.GGMLQuantizationType.F32),
        int(gguf.GGMLQuantizationType.F16),
        int(gguf.GGMLQuantizationType.BF16),
    }:
        # BF16 (gemma4's per_layer_token_embd) is a 16-bit float, not a block
        # quant — it goes through the dense path, dequantized in
        # `dense_tensor_weights`.
        return None
    raise NotImplementedError(f"Unsupported quantized GGUF tensor type={tensor_type}")


def dense_tensor_weights(tensor: gguf.ReaderTensor) -> np.ndarray:
    tensor_type = int(tensor.tensor_type)
    if tensor_type in {
        int(gguf.GGMLQuantizationType.F32),
        int(gguf.GGMLQuantizationType.F16),
    }:
        return np.asarray(tensor.data)
    if tensor_type == int(gguf.GGMLQuantizationType.BF16):
        # numpy has no native bfloat16: the raw payload is loaded as uint16
        # (see `live_tensors_from_meta`). bf16 -> f32 is just the high 16 bits
        # of the f32 layout; downcast to f16 so the large gemma4
        # per_layer_token_embd (vocab 262144 x per-layer dim) stays 16-bit
        # rather than doubling to f32 (which OOMs alloc + cache warmup).
        u16 = np.asarray(tensor.data, dtype=np.uint16)
        return (u16.astype(np.uint32) << 16).view(np.float32).astype(np.float16)
    raise NotImplementedError(f"Quantized GGUF tensor must stay packed, got type={tensor_type}")


def replace_quantized_weight(
    model: torch.nn.Module,
    hf_weight_name: str,
    weights: np.ndarray,
    quantization: GGUFQuantization,
) -> Literal["linear", "embedding"]:
    if not hf_weight_name.endswith(".weight"):
        raise NotImplementedError(f"Quantized GGUF tensor does not map to a weight: {hf_weight_name}")
    module = module_for_parameter(model, hf_weight_name)
    if isinstance(module, torch.nn.Linear):
        replace_linear_with_quantized(model, hf_weight_name, weights, quantization)
        return "linear"
    if isinstance(module, torch.nn.Embedding):
        replace_embedding_with_quantized(model, hf_weight_name, weights, quantization)
        return "embedding"
    raise NotImplementedError(
        "Quantized GGUF tensor maps to unsupported module: "
        f"{hf_weight_name} -> {type(module).__name__}"
    )


def replace_linear_with_quantized(
    model: torch.nn.Module,
    hf_weight_name: str,
    weights: np.ndarray,
    quantization: GGUFQuantization,
) -> None:
    module_path = hf_weight_name[: -len(".weight")]
    parent, child_name = module_parent(model, module_path)
    child = parent._modules[child_name]
    if not isinstance(child, torch.nn.Linear):
        raise TypeError(f"Expected Linear at {module_path}, got {type(child).__name__}")
    bias = child.bias if child.bias is not None else None

    if quantization == "q4_k":
        # GGUF Q4_K is already 144-byte superblocks (out, blocks_per_row*144);
        # store the raw bytes — the kernel decodes them natively.
        blocks = torch.from_numpy(np.array(weights, copy=True))
        parent._modules[child_name] = GGUFQ4_KLinear(
            blocks=blocks,
            in_features=child.in_features,
            out_features=child.out_features,
            bias=bias,
        )
    elif quantization == "q5_0":
        qweight, qhigh, scales = split_q5_0_weight(
            weights=weights,
            in_features=child.in_features,
            out_features=child.out_features,
        )
        parent._modules[child_name] = GGUFQ5_0Linear(
            qweight=qweight,
            qhigh=qhigh,
            scales=scales,
            in_features=child.in_features,
            out_features=child.out_features,
            bias=bias,
        )
    elif quantization == "q6_k":
        qweight = torch.from_numpy(np.array(weights, copy=True))
        parent._modules[child_name] = GGUFQ6_KLinear(
            qweight=qweight,
            in_features=child.in_features,
            out_features=child.out_features,
            bias=bias,
        )
    else:
        qweight, scales = split_q8_0_weight(
            weights=weights,
            in_features=child.in_features,
            out_features=child.out_features,
        )
        parent._modules[child_name] = GGUFQ8_0Linear(
            qweight=qweight,
            scales=scales,
            in_features=child.in_features,
            out_features=child.out_features,
            bias=bias,
        )


def replace_embedding_with_quantized(
    model: torch.nn.Module,
    hf_weight_name: str,
    weights: np.ndarray,
    quantization: GGUFQuantization,
) -> None:
    module_path = hf_weight_name[: -len(".weight")]
    parent, child_name = module_parent(model, module_path)
    child = parent._modules[child_name]
    if not isinstance(child, torch.nn.Embedding):
        raise TypeError(f"Expected Embedding at {module_path}, got {type(child).__name__}")
    # Gemma3's input embedding multiplies the lookup by sqrt(hidden_size) via
    # `Gemma3TextScaledWordEmbedding.scalar_embed_scale`. Preserve it so the
    # quantized replacement keeps the model arithmetically faithful.
    embed_scale = float(child.scalar_embed_scale) if hasattr(child, "scalar_embed_scale") else 1.0

    if quantization == "q4_k":
        blocks = torch.from_numpy(np.array(weights, copy=True))
        parent._modules[child_name] = GGUFQ4_KEmbedding(
            blocks=blocks,
            num_embeddings=child.num_embeddings,
            embedding_dim=child.embedding_dim,
            embed_scale=embed_scale,
        )
    elif quantization == "q5_0":
        qweight, qhigh, scales = split_q5_0_weight(
            weights=weights,
            in_features=child.embedding_dim,
            out_features=child.num_embeddings,
        )
        parent._modules[child_name] = GGUFQ5_0Embedding(
            qweight=qweight,
            qhigh=qhigh,
            scales=scales,
            num_embeddings=child.num_embeddings,
            embedding_dim=child.embedding_dim,
            embed_scale=embed_scale,
        )
    elif quantization == "q8_0":
        qweight, scales = split_q8_0_weight(
            weights=weights,
            in_features=child.embedding_dim,
            out_features=child.num_embeddings,
        )
        parent._modules[child_name] = GGUFQ8_0Embedding(
            qweight=qweight,
            scales=scales,
            num_embeddings=child.num_embeddings,
            embedding_dim=child.embedding_dim,
            embed_scale=embed_scale,
        )
    elif quantization == "q6_k":
        qweight = torch.from_numpy(np.array(weights, copy=True))
        parent._modules[child_name] = GGUFQ6_KEmbedding(
            qweight=qweight,
            num_embeddings=child.num_embeddings,
            embedding_dim=child.embedding_dim,
            embed_scale=embed_scale,
        )
    else:
        raise NotImplementedError(f"unsupported embedding quantization: {quantization}")


def split_q5_0_weight(
    *,
    weights: np.ndarray,
    in_features: int,
    out_features: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Repack Q5_0 wire blocks into K-sequential nibble + bit-packed
    high-bit + fp16 scale buffers.

    Wire format per 32-element block (22 bytes):
      `{fp16 d; uint8 qh[4]; uint8 qs[16]}` where for l in 0..15:
        element[l]    = (qs[l] & 0xF) | ((qh[l] high bit) << 4) - 16
        element[l+16] = (qs[l] >> 4)  | ((qh[l+16] high bit) << 4) - 16
        qh32 = qh[0..3] viewed as little-endian uint32

    Output layout (K-sequential, same nibble-packed layout for both
    buffers so the kernel can `load4_vec` + interleave_vec4 both):
      qweight[N, K/2]  uint8 — byte at k/2 packs element k (low nibble)
                                and element k+1 (high nibble).
      qhigh  [N, K/2]  uint8 — byte at k/2 packs element k's 5th bit
                                (bit 0 of low nibble) and element k+1's
                                5th bit (bit 0 of high nibble = bit 4
                                of the byte).
      scales [N, K/32] fp16  — one fp16 scale per 32-element block.

    Storage cost: 8.5 bits/elem (vs 5.5 bits/elem wire). The slack is
    in qhigh's nibbles (only 1 of 4 bits per nibble is used). Worth
    it for the simple kernel — bit-packing harder would mean a custom
    vec4-shuffle primitive in the DSL.
    """
    if weights.dtype != np.uint8:
        raise TypeError(f"Q5_0 weights must be uint8 bytes, got {weights.dtype}")
    if in_features % 32 != 0:
        raise ValueError(f"Q5_0 in_features must be divisible by 32, got {in_features}")

    blocks_per_row = in_features // 32
    expected_shape = (out_features, blocks_per_row * 22)
    if weights.shape != expected_shape:
        raise ValueError(
            "Q5_0 packed weight shape does not match linear dimensions: "
            f"weights={tuple(weights.shape)} expected={expected_shape}"
        )

    packed = np.ascontiguousarray(weights)
    blocks = packed.reshape(out_features, blocks_per_row, 22)

    # fp16 d at bytes 0..1 per block
    scales = np.array(
        np.ascontiguousarray(blocks[:, :, 0:2]).view(np.float16).reshape(
            out_features, blocks_per_row,
        ),
        dtype=np.float16, copy=True,
    )

    # qh32 (uint32 little-endian per block): bit l is element l's 5th bit.
    qh_bytes = np.ascontiguousarray(blocks[:, :, 2:6])
    qh32 = qh_bytes.view(np.uint32).reshape(out_features, blocks_per_row)

    # qhigh layout: byte at k/2 = highbit_k | (highbit_(k+1) << 4).
    # 16 bytes per block covering 32 K positions.
    shifts_lo = (np.arange(16, dtype=np.uint32) * 2)        # bit positions for even K
    shifts_hi = shifts_lo + 1                                # bit positions for odd K
    qh_lo = ((qh32[:, :, None] >> shifts_lo) & 1).astype(np.uint8)
    qh_hi = ((qh32[:, :, None] >> shifts_hi) & 1).astype(np.uint8)
    qhigh = (qh_lo | (qh_hi << 4)).reshape(out_features, blocks_per_row * 16)

    # qs: 16 bytes per block, wire interleaved (low nibble = element l,
    # high nibble = element l+16). Reassemble K-sequentially.
    qs = blocks[:, :, 6:22]                           # (N, B, 16) uint8
    low_nibs = qs & 0x0F                              # (N, B, 16) -> elements 0..15
    high_nibs = (qs >> 4) & 0x0F                      # (N, B, 16) -> elements 16..31
    seq = np.concatenate((low_nibs, high_nibs), axis=2)  # (N, B, 32)
    even = seq[:, :, 0::2]
    odd = seq[:, :, 1::2]
    qweight = (even | (odd << 4)).astype(np.uint8).reshape(
        out_features, blocks_per_row * 16,
    )

    return (
        torch.from_numpy(np.array(qweight, copy=True)),
        torch.from_numpy(np.array(qhigh, copy=True)),
        torch.from_numpy(scales),
    )


def split_q8_0_weight(
    *,
    weights: np.ndarray,
    in_features: int,
    out_features: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if weights.dtype != np.uint8:
        raise TypeError(f"Q8_0 weights must be uint8 bytes, got {weights.dtype}")
    if in_features % 32 != 0:
        raise ValueError(f"Q8_0 in_features must be divisible by 32, got {in_features}")

    groups = in_features // 32
    expected_shape = (out_features, groups * 34)
    if weights.shape != expected_shape:
        raise ValueError(
            "Q8_0 packed weight shape does not match linear dimensions: "
            f"weights={tuple(weights.shape)} expected={expected_shape}"
        )

    packed = np.ascontiguousarray(weights)
    blocks = packed.reshape(out_features, groups, 34)
    scale_bytes = np.ascontiguousarray(blocks[:, :, :2])
    scales = np.array(scale_bytes.view(np.float16).reshape(out_features, groups), copy=True)
    qweight = np.array(blocks[:, :, 2:].reshape(out_features, in_features).view(np.int8), copy=True)
    return torch.from_numpy(qweight), torch.from_numpy(scales)


def module_for_parameter(model: torch.nn.Module, hf_name: str) -> torch.nn.Module:
    module_path = hf_name[: -len(".weight")]
    parent, child_name = module_parent(model, module_path)
    module = parent._modules[child_name]
    if module is None:
        raise KeyError(f"Module path contains empty module: {module_path}")
    return module


def module_parent(model: torch.nn.Module, module_path: str) -> tuple[torch.nn.Module, str]:
    parts = module_path.split(".")
    parent = model
    for part in parts[:-1]:
        if part not in parent._modules:
            raise KeyError(f"Module path not found: {module_path}")
        child = parent._modules[part]
        if child is None:
            raise KeyError(f"Module path contains empty module: {module_path}")
        parent = child
    return parent, parts[-1]


def torch_tensor_from_numpy(weights: np.ndarray, *, dtype: torch.dtype | None) -> torch.Tensor:
    tensor = torch.from_numpy(np.array(weights, copy=True))
    if tensor.is_floating_point() and dtype is not None:
        return tensor.to(dtype=dtype)
    return tensor


def tie_quantized_output_embeddings(model: PreTrainedModel) -> None:
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if isinstance(input_embeddings, GGUFQ8_0Embedding):
        if output_embeddings is not None and not isinstance(output_embeddings, torch.nn.Linear):
            raise TypeError(
                "Tied Q8_0 GGUF embeddings require a Linear output head, "
                f"got {type(output_embeddings).__name__}"
            )
        if output_embeddings is not None and output_embeddings.bias is not None:
            raise NotImplementedError("Tied Q8_0 GGUF output head bias is not supported")
        model.set_output_embeddings(
            GGUFQ8_0Linear(
                qweight=input_embeddings.qweight,
                scales=input_embeddings.scales,
                in_features=input_embeddings.embedding_dim,
                out_features=input_embeddings.num_embeddings,
            )
        )
        return
    if isinstance(input_embeddings, GGUFQ4_KEmbedding):
        if output_embeddings is not None and not isinstance(output_embeddings, torch.nn.Linear):
            raise TypeError(
                "Tied Q4_K GGUF embeddings require a Linear output head, "
                f"got {type(output_embeddings).__name__}"
            )
        if output_embeddings is not None and output_embeddings.bias is not None:
            raise NotImplementedError("Tied Q4_K GGUF output head bias is not supported")
        model.set_output_embeddings(
            GGUFQ4_KLinear(
                blocks=input_embeddings.blocks,
                in_features=input_embeddings.embedding_dim,
                out_features=input_embeddings.num_embeddings,
            )
        )
        return
    if isinstance(input_embeddings, GGUFQ5_0Embedding):
        if output_embeddings is not None and not isinstance(output_embeddings, torch.nn.Linear):
            raise TypeError(
                "Tied Q5_0 GGUF embeddings require a Linear output head, "
                f"got {type(output_embeddings).__name__}"
            )
        if output_embeddings is not None and output_embeddings.bias is not None:
            raise NotImplementedError("Tied Q5_0 GGUF output head bias is not supported")
        model.set_output_embeddings(
            GGUFQ5_0Linear(
                qweight=input_embeddings.qweight,
                qhigh=input_embeddings.qhigh,
                scales=input_embeddings.scales,
                in_features=input_embeddings.embedding_dim,
                out_features=input_embeddings.num_embeddings,
            )
        )
        return
    if isinstance(input_embeddings, GGUFQ6_KEmbedding):
        if output_embeddings is not None and not isinstance(output_embeddings, torch.nn.Linear):
            raise TypeError(
                "Tied Q6_K GGUF embeddings require a Linear output head, "
                f"got {type(output_embeddings).__name__}"
            )
        if output_embeddings is not None and output_embeddings.bias is not None:
            raise NotImplementedError("Tied Q6_K GGUF output head bias is not supported")
        model.set_output_embeddings(
            GGUFQ6_KLinear(
                qweight=input_embeddings.qweight,
                in_features=input_embeddings.embedding_dim,
                out_features=input_embeddings.num_embeddings,
            )
        )
        return
    model.tie_weights()
