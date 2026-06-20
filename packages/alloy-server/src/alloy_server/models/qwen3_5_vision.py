"""Qwen 3.5 vision tower on alloy — the image modality for arch=`qwen35`.

Qwen 3.5's GGUF ships a full vision tower (`v.*`: a Qwen2.5-VL-style ViT —
Conv3d-as-linear patch embed, interpolated learned 2D position embeddings,
2D-RoPE bidirectional blocks, a 2×2 spatial merger) that the text-only loader
dropped (see `models/qwen3_5.py:apply_qwen35_post_load_config_fixup`). This module
loads those weights into transformers' native `Qwen3_5VisionModel` and runs the
whole tensor compute through alloy's Metal dispatch.

The position-derived bookkeeping (the interpolated pos-embed, the 2D-RoPE
cos/sin, the padding key-mask) is precomputed on CPU — the parts HF expresses as
`linspace`/`tolist`/`cumsum` index math, which don't belong in a fixed-shape GPU
plan — so the alloy graph is pure tensor compute (matmuls, attention, norms,
the merger). Dynamic resolution: pad the variable patch count to a fixed bucket
and mask the padding keys; the merger's padding outputs are gathered off at the
end.
"""

from __future__ import annotations

import io
from typing import cast

import numpy as np
import torch
from PIL import Image
from transformers import Qwen2VLImageProcessor, Qwen3_5VisionConfig, Qwen3_5VisionModel
from transformers.initialization import no_init_weights

from alloy_server.models.modality import CaptureTarget, ModalityAdapter

# Patch bucket: the variable per-image patch count is padded up to this fixed
# size so the ViT compiles to one plan. The image processor's max_pixels caps
# the real count below it; china.jpg (26×38 grid) is 988 patches. A multiple of
# the 2×2 merge unit (×4) so padding never splits a valid merge group.
PATCH_BUCKET = 1024
# Image-processor pixel budget (in patches): bounds the real patch count under
# the bucket. min keeps small images from collapsing to a few tokens.
MIN_PATCHES = 64
MAX_PATCHES = PATCH_BUCKET
# Additive mask value for padding keys. fp16-safe (within ±65504): alloy's fused
# attention kernel runs fp16, where the usual -1e30 overflows to -inf and
# corrupts the softmax/LSE (cosine 0.81); -3e4 masks cleanly (cosine 0.99).
MASK_NEG = -30000.0


def build_qwen35_vision_config(kv: dict) -> Qwen3_5VisionConfig:
    """Build `Qwen3_5VisionConfig` from the GGUF `qwen35.vision.*` metadata.

    transformers' GGUF config parse only maps the text decoder; the vision
    sub-config is reconstructed here. `intermediate_size` / `out_hidden_size`
    aren't in the GGUF scalars — they're read off the merger/MLP tensor shapes
    by the caller and passed through `kv` under synthetic keys."""
    return Qwen3_5VisionConfig(
        depth=int(kv["qwen35.vision.block_count"]),
        hidden_size=int(kv["qwen35.vision.embedding_length"]),
        num_heads=int(kv["qwen35.vision.attention.head_count"]),
        intermediate_size=int(kv["_intermediate_size"]),
        out_hidden_size=int(kv["_out_hidden_size"]),
        patch_size=int(kv["qwen35.vision.patch_size"]),
        spatial_merge_size=int(kv["qwen35.vision.spatial_merge_size"]),
        temporal_patch_size=int(kv["qwen35.vision.temporal_patch_size"]),
        num_position_embeddings=int(kv["_num_position_embeddings"]),
        in_channels=int(kv["qwen35.vision.num_channels"]),
    )


def qwen35_vision_state_dict(
    tensors, config: Qwen3_5VisionConfig, dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Map GGUF `v.*` vision tensors to a `Qwen3_5VisionModel` state dict.

    GGUF dense tensors (`np.array(t.data)`) are already HF `(out, in)` oriented.
    Two structural transforms: the GGUF carries *separate* `attn_q/k/v` but HF
    wants a *combined* `qkv` (concat along the output dim), and the patch embed
    is a Conv3d whose flattened weight `(out, in, kT, kH, kW)` loads from the
    GGUF's `(out, in·kT·kH·kW)` blob."""
    data: dict[str, np.ndarray] = {}
    for t in tensors:
        if t.name.startswith("v."):
            data[t.name] = np.asarray(t.data).astype(np.float32)

    d, ic = config.hidden_size, config.in_channels
    kt, ps = config.temporal_patch_size, config.patch_size

    def T(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(arr).to(dtype)

    sd: dict[str, torch.Tensor] = {
        "patch_embed.proj.weight": T(data["v.patch_embed.weight"].reshape(d, ic, kt, ps, ps)),
        "patch_embed.proj.bias": T(data["v.patch_embed.bias"]),
        "pos_embed.weight": T(data["v.pos_embed.weight"]),
        "merger.norm.weight": T(data["v.merger.norm.weight"]),
        "merger.norm.bias": T(data["v.merger.norm.bias"]),
        "merger.linear_fc1.weight": T(data["v.merger.linear_fc1.weight"]),
        "merger.linear_fc1.bias": T(data["v.merger.linear_fc1.bias"]),
        "merger.linear_fc2.weight": T(data["v.merger.linear_fc2.weight"]),
        "merger.linear_fc2.bias": T(data["v.merger.linear_fc2.bias"]),
    }
    for blk in range(config.depth):
        g, h = f"v.blk.{blk}.", f"blocks.{blk}."
        sd[h + "attn.qkv.weight"] = T(np.concatenate(
            [data[g + "attn_q.weight"], data[g + "attn_k.weight"], data[g + "attn_v.weight"]], axis=0))
        sd[h + "attn.qkv.bias"] = T(np.concatenate(
            [data[g + "attn_q.bias"], data[g + "attn_k.bias"], data[g + "attn_v.bias"]], axis=0))
        sd[h + "attn.proj.weight"] = T(data[g + "attn_out.weight"])
        sd[h + "attn.proj.bias"] = T(data[g + "attn_out.bias"])
        sd[h + "mlp.linear_fc1.weight"] = T(data[g + "mlp.linear_fc1.weight"])
        sd[h + "mlp.linear_fc1.bias"] = T(data[g + "mlp.linear_fc1.bias"])
        sd[h + "mlp.linear_fc2.weight"] = T(data[g + "mlp.linear_fc2.weight"])
        sd[h + "mlp.linear_fc2.bias"] = T(data[g + "mlp.linear_fc2.bias"])
        sd[h + "norm1.weight"] = T(data[g + "norm1.weight"])
        sd[h + "norm1.bias"] = T(data[g + "norm1.bias"])
        sd[h + "norm2.weight"] = T(data[g + "norm2.weight"])
        sd[h + "norm2.bias"] = T(data[g + "norm2.bias"])
    return sd


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class Qwen35VitPatch(torch.nn.Module):
    """Patch embed + position embed: (1, bucket, in·kT·ps·ps) -> (1, bucket,
    hidden). The Conv3d collapses to a linear (kernel == patch, stride == kernel,
    so it's a per-patch projection). A separate alloy plan from the encoder: the
    patch GEMM's output feeds both `norm1→qkv` and the block residual, and fusing
    that reconvergence into the first block corrupts the encode — materializing
    the patch+pos output between them keeps both correct."""

    def __init__(self, vision: Qwen3_5VisionModel) -> None:
        super().__init__()
        w = vision.patch_embed.proj.weight
        self.patch_w = torch.nn.Parameter(w.reshape(w.shape[0], -1).contiguous(), requires_grad=False)
        self.patch_b = torch.nn.Parameter(vision.patch_embed.proj.bias.detach(), requires_grad=False)

    def forward(self, patch_input: torch.Tensor, pos_embeds: torch.Tensor) -> torch.Tensor:
        return patch_input @ self.patch_w.t() + self.patch_b + pos_embeds


class Qwen35VitEncode(torch.nn.Module):
    """Fixed-shape (bucketed) qwen3.5 ViT encoder: 24 bidirectional 2D-RoPE blocks
    over the patch+pos embeddings -> (1, bucket, hidden). The position bookkeeping
    (RoPE cos/sin, padding key-mask) is precomputed on CPU and passed in, so the
    graph is pure tensor compute — one alloy plan.

    Attention uses F.scaled_dot_product_attention (alloy's fused kernel) with an
    fp16-safe additive key-mask (see `MASK_NEG`). The fused kernel runs fp16
    (~cosine-0.99 vs an fp32 reference) and avoids materializing the (S×S) scores
    matrix per block."""

    def __init__(self, vision: Qwen3_5VisionModel) -> None:
        super().__init__()
        cfg = vision.config
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.hidden_size // cfg.num_heads
        self.scaling = self.head_dim**-0.5
        self.blocks = vision.blocks

    def forward(
        self,
        x: torch.Tensor,             # (1, bucket, hidden) — patch+pos embeddings
        cos: torch.Tensor,           # (1, S, 1, head_dim)
        sin: torch.Tensor,           # (1, S, 1, head_dim)
        key_mask: torch.Tensor,      # (1, 1, 1, S) additive: 0 valid, MASK_NEG padding
    ) -> torch.Tensor:
        b, s, _ = x.shape
        for blk in self.blocks:
            h = blk.norm1(x)
            qkv = blk.attn.qkv(h).reshape(b, s, 3, self.num_heads, self.head_dim)
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # (1, S, heads, hd)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
            q = q.transpose(1, 2)  # (1, heads, S, hd)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            attn = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=key_mask, scale=self.scaling,
            )
            attn = attn.transpose(1, 2).reshape(b, s, -1)
            x = x + blk.attn.proj(attn)
            x = x + blk.mlp(blk.norm2(x))
        return x


class Qwen35VitMerge(torch.nn.Module):
    """Fixed-shape 2×2 spatial merger: encoder output (1, bucket, hidden) ->
    (bucket/merge_unit, out_hidden) language-model-space soft tokens. A separate
    alloy plan from the encoder so the encoder output materializes between them
    — fusing the merger across the encoder boundary corrupts it."""

    def __init__(self, vision: Qwen3_5VisionModel) -> None:
        super().__init__()
        self.merger = vision.merger

    def forward(self, enc: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.merger(enc[0]))


class Qwen35VisionAdapter(ModalityAdapter):
    """Turns image bytes into qwen3.5 language-model-space vision features, and
    encodes qwen3.5's image-placeholder expansion. Implements the `ModalityEncoder`
    protocol the served model consumes. Owns the dense ViT (loaded from the GGUF
    `v.*` tensors, compiled through alloy) + the HF image processor.

    The chat template wraps each image as `<|vision_start|><|image_pad|>
    <|vision_end|>`; the inherited `expand_text` blows the single `<|image_pad|>`
    up to one per soft token (no begin/end markers), and `encode` returns
    `(num_soft_tokens, text_hidden)`."""

    PLACEHOLDER = "<|image_pad|>"
    ITEM_NOUN = "image"

    def __init__(
        self, tensors, vision_kv: dict, image_token_id: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        config = build_qwen35_vision_config(vision_kv)
        # Skip HF's random weight init: load_state_dict overwrites every
        # parameter from the GGUF, so the normal_/uniform_ fill (~2.5s for this
        # tower) is pure waste.
        with no_init_weights():
            vision = Qwen3_5VisionModel(config).eval()
        vision.load_state_dict(
            qwen35_vision_state_dict(tensors, config, dtype), strict=True
        )
        self.vision = vision.to(dtype)
        self.config = config
        self.dtype = dtype
        self.placeholder_token_id = image_token_id
        self.merge = config.spatial_merge_size
        # Three compiled stages — patch+pos, encoder, merger — each materializing
        # between them. Both boundaries (patch→encoder and encoder→merger) corrupt
        # if fused.
        self.patch_mod = Qwen35VitPatch(vision)
        self.encode_mod = Qwen35VitEncode(vision)
        self.merge_mod = Qwen35VitMerge(vision)
        self.vit_patch = cast(
            torch.nn.Module, torch.compile(self.patch_mod, backend="alloy", dynamic=False))
        self.vit_encode = cast(
            torch.nn.Module, torch.compile(self.encode_mod, backend="alloy", dynamic=False))
        self.vit_merge = cast(
            torch.nn.Module, torch.compile(self.merge_mod, backend="alloy", dynamic=False))
        # qwen3.5 vision normalizes with mean/std = 0.5 (per the GGUF
        # `qwen35.vision.image_mean/std`), NOT the OpenAI-CLIP stats the processor
        # defaults to — wrong normalization shifts every pixel and corrupts the
        # features (e.g. fine-grained landmark mis-ID). Read them from the GGUF.
        image_mean = [float(x) for x in vision_kv.get("qwen35.vision.image_mean", [0.5, 0.5, 0.5])]
        image_std = [float(x) for x in vision_kv.get("qwen35.vision.image_std", [0.5, 0.5, 0.5])]
        self.image_processor = Qwen2VLImageProcessor(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            merge_size=config.spatial_merge_size,
            image_mean=image_mean,
            image_std=image_std,
            min_pixels=MIN_PATCHES * config.patch_size**2,
            max_pixels=MAX_PATCHES * config.patch_size**2,
        )

    def bookkeeping(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Precompute (on CPU) the fixed-bucket inputs the alloy graph needs: the
        padded patch input, interpolated pos-embed, 2D-RoPE cos/sin, the padding
        key-mask, and the valid soft-token count. Uses the eager model's own
        `fast_pos_embed_interpolate` / `rot_pos_emb` (pure index math) for the
        bookkeeping."""
        s = int(pixel_values.shape[0])
        bucket = PATCH_BUCKET
        if s > bucket:
            raise ValueError(f"image has {s} patches; raise PATCH_BUCKET ({bucket})")

        pos = self.vision.fast_pos_embed_interpolate(grid_thw).to(self.dtype)  # (S, hidden)
        rot = self.vision.rot_pos_emb(grid_thw)  # (S, hd/2)
        emb = torch.cat((rot, rot), dim=-1)  # (S, hd)
        cos_s, sin_s = emb.cos().to(self.dtype), emb.sin().to(self.dtype)

        def pad(x: torch.Tensor, fill: float = 0.0) -> torch.Tensor:
            out = x.new_full((bucket, *x.shape[1:]), fill)
            out[:s] = x
            return out

        patch_in = pad(pixel_values.to(self.dtype)).unsqueeze(0)         # (1, B, F)
        pos_b = pad(pos).unsqueeze(0)                                     # (1, B, hidden)
        cos_b = pad(cos_s, 1.0).unsqueeze(0).unsqueeze(2)                 # (1, B, 1, hd)
        sin_b = pad(sin_s, 0.0).unsqueeze(0).unsqueeze(2)                 # (1, B, 1, hd)
        key_mask = torch.full((1, 1, 1, bucket), MASK_NEG, dtype=self.dtype)
        key_mask[..., :s] = 0.0                                          # attend valid keys only
        n_soft = s // (self.merge * self.merge)
        return patch_in, pos_b, cos_b, sin_b, key_mask, n_soft

    def encode(self, image) -> torch.Tensor:
        """`image` is raw encoded bytes or a PIL.Image. Returns `(num_soft_tokens,
        text_hidden)` — one feature row per image soft token."""
        if isinstance(image, (bytes, bytearray)):
            image = Image.open(io.BytesIO(image)).convert("RGB")
        processed = self.image_processor(images=[image], return_tensors="pt")
        pixel_values = processed["pixel_values"]
        grid_thw = processed["image_grid_thw"]
        patch_in, pos_b, cos_b, sin_b, key_mask, n_soft = self.bookkeeping(
            pixel_values, grid_thw
        )
        with torch.inference_mode():
            x0 = self.vit_patch(patch_in, pos_b)
            enc = self.vit_encode(x0, cos_b, sin_b, key_mask)
            feats = self.vit_merge(enc)
        return feats[:n_soft]  # drop the padding merge groups

    def eager_compile_all(self) -> None:
        """Compile the three alloy vision plans (patch / encode / merge) ahead of
        the first request. The ViT input shape is the fixed bucket, so one dummy
        image compiles them all."""
        self.encode(Image.new("RGB", (64, 64)))

    def capture_targets(self) -> list[CaptureTarget]:
        """The two compiled ViT stages + fixed-shape inputs from a dummy image,
        for `alloy tune` / `alloy profile`. The bucket shape IS the production
        shape; the merge stage's encoder input is synthesized (values are
        irrelevant to kernel shape/timing)."""
        processed = self.image_processor(
            images=[Image.new("RGB", (64, 64))], return_tensors="pt"
        )
        patch_in, pos_b, cos_b, sin_b, key_mask, _ = self.bookkeeping(
            processed["pixel_values"], processed["image_grid_thw"]
        )
        x0 = torch.zeros((1, PATCH_BUCKET, self.config.hidden_size), dtype=self.dtype)
        return [
            CaptureTarget(
                name="vision_patch",
                label="vision patch embed + position embed",
                module=self.patch_mod,
                inputs={"patch_input": patch_in, "pos_embeds": pos_b},
            ),
            CaptureTarget(
                name="vision_encode",
                label="vision encode (24-layer ViT)",
                module=self.encode_mod,
                inputs={"x": x0, "cos": cos_b, "sin": sin_b, "key_mask": key_mask},
            ),
            CaptureTarget(
                name="vision_merge",
                label="vision merge (2×2 spatial merger + projector)",
                module=self.merge_mod,
                inputs={"enc": x0},
            ),
        ]

def gguf_has_vision_qwen35(tensors) -> bool:
    """True if these GGUF tensors include qwen3.5 vision weights (vs text-only)."""
    return any(t.name.startswith("v.blk.") for t in tensors)


def read_tensor_shapes(tensors) -> dict[str, tuple[int, ...]]:
    out: dict[str, tuple[int, ...]] = {}
    for t in tensors:
        if t.name in ("v.merger.linear_fc1.weight", "v.merger.linear_fc2.weight",
                      "v.pos_embed.weight", "v.blk.0.mlp.linear_fc1.weight"):
            out[t.name] = tuple(int(x) for x in np.asarray(t.data).shape)
    return out


def build_qwen35_vision_adapter(tensors, vision_kv: dict, image_token_id: int) -> Qwen35VisionAdapter:
    """Construct the qwen3.5 vision adapter from live GGUF tensors + the cached
    `qwen35.vision.*` metadata. Called by the GGUF loader's arch dispatch.

    `intermediate_size`, `out_hidden_size`, and `num_position_embeddings` aren't
    in the GGUF scalar metadata — derive them from the tensor shapes (MLP fc1
    out-dim, merger fc2 out-dim, pos-embed table length) and pass through the
    config builder under synthetic keys."""
    shapes = read_tensor_shapes(tensors)
    kv = dict(vision_kv)
    kv["_intermediate_size"] = shapes["v.blk.0.mlp.linear_fc1.weight"][0]   # (out, in)
    kv["_out_hidden_size"] = shapes["v.merger.linear_fc2.weight"][0]        # (out, in)
    kv["_num_position_embeddings"] = shapes["v.pos_embed.weight"][0]        # (num_pos, hidden)
    return Qwen35VisionAdapter(tensors, kv, image_token_id)
