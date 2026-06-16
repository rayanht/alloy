"""gemma4 model handler (arch=`gemma4`): a multimodal (text + vision + audio)
GGUF. The handler owns the gemma4-specific GGUF interpretation (config fixup,
tensor-map fixup, transformers-registry patches, chat template) and builds the
dense vision (SigLIP RoPE ViT) + audio (USM Conformer) towers as alloy-compiled
`ModalityAdapter`s.

Ported from the former `_gemma4_compat.py` + `_gemma4_audio.py` shims. The vision
tower runs in f32; the audio tower in native f16 (the conformer is bit-near-exact
at f16, and two alloy properties make it work — f16/bf16 scalar decode + the
functional out-of-place conformer forwards below).
"""

from __future__ import annotations

import importlib.resources
import io
import re
import types
from typing import cast

import gguf
import numpy as np
import soundfile
import torch
from gguf.quants import dequantize
from PIL import Image
from transformers import (
    Gemma4AudioConfig,
    Gemma4AudioFeatureExtractor,
    Gemma4AudioModel,
    Gemma4ImageProcessor,
    Gemma4VisionConfig,
)
from transformers.initialization import no_init_weights
from transformers.integrations.ggml import GGUF_CONFIG_MAPPING, GGUF_TO_FAST_CONVERTERS
from transformers.masking_utils import create_bidirectional_mask
from transformers.modeling_gguf_pytorch_utils import GGUF_SUPPORTED_ARCHITECTURES
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4MultimodalEmbedder,
    Gemma4VisionModel,
    sliding_window_mask_function,
)

from alloy_server.models.base import CausalLMHandler
from alloy_server.models.modality import CaptureTarget, ModalityAdapter
from alloy_server.models.registry import register

# gemma4 multimodal token ids (from the HF Gemma4Config defaults): the image
# placeholder whose embedding is replaced by a vision feature, plus the
# begin/end-of-image brackets the processor wraps the soft-token run in.
GEMMA4_IMAGE_TOKEN_ID = 258880
GEMMA4_BOI_TOKEN_ID = 255999
GEMMA4_EOI_TOKEN_ID = 258882
GEMMA4_AUDIO_TOKEN_ID = 258881

GGUF_DENSE = (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.BF16)


def gemma4_chat_template() -> str:
    """The official gemma4 chat template (google/gemma-4-E2B-it/chat_template.jinja).

    gemma4's GGUF doesn't embed a chat_template, and — unlike gemma2/3 — Google
    ships it as a standalone `chat_template.jinja` rather than inside
    tokenizer_config.json. Bundled in the package so the GGUF-loaded tokenizer can
    render gemma4 chats (`<|turn>{role}\n…<turn|>` markers, with tool-call /
    thinking-channel macros), matching ollama's `RENDERER gemma4`.
    """
    return (
        importlib.resources.files("alloy_server")
        .joinpath("gemma4_chat_template.jinja")
        .read_text(encoding="utf-8")
    )


# Scalar GGUF metadata key (sans `gemma4.` prefix) -> Gemma4TextConfig field.
# `attention.key_length` (512) is the FULL-attention head dim -> global_head_dim;
# `attention.key_length_swa` (256) is the sliding head dim -> head_dim. The two
# rope freq bases (1e6 full / 1e4 sliding) already match the Gemma4TextConfig
# defaults, so they're intentionally not mapped here.
GEMMA4_CONFIG_MAPPING = {
    "context_length": "max_position_embeddings",
    "block_count": "num_hidden_layers",
    "embedding_length": "hidden_size",
    "attention.head_count": "num_attention_heads",
    "attention.head_count_kv": "num_key_value_heads",
    "attention.key_length": "global_head_dim",
    "attention.key_length_swa": "head_dim",
    "attention.layer_norm_rms_epsilon": "rms_norm_eps",
    "attention.sliding_window": "sliding_window",
    "attention.shared_kv_layers": "num_kv_shared_layers",
    "embedding_length_per_layer_input": "hidden_size_per_layer_input",
    "final_logit_softcapping": "final_logit_softcapping",
    "vocab_size": "vocab_size",
}


def gguf_array(reader, name):
    """Read a GGUF metadata array field as a plain Python list, or None."""
    field = reader.fields.get(name)
    if field is None:
        return None
    contents = field.contents()
    if isinstance(contents, np.ndarray):
        return contents.tolist()
    if isinstance(contents, (list, tuple)):
        return list(contents)
    return [contents]


def apply_gemma4_post_load_config_fixup(config_dict: dict, reader) -> None:
    """Resolve the per-layer GGUF arrays into the HF fields Gemma4TextConfig
    expects. transformers' GGUF parse maps the scalar keys (via
    `GEMMA4_CONFIG_MAPPING`) but drops the per-layer arrays, so read them straight
    from the GGUF reader here.

    - `feed_forward_length` is per-layer ([6144]*15 + [12288]*20 on e2b). The
      base `intermediate_size` is the min; the wide tail is exactly `2*base` on
      the KV-shared layers, which HF reproduces via `use_double_wide_mlp` — so
      set the flag rather than carry a list.
    - `sliding_window_pattern` is a per-layer bool list (True == sliding-window
      attention) -> `layer_types`.
    """
    ffl = gguf_array(reader, "gemma4.feed_forward_length")
    if ffl is not None:
        ffl_ints = [int(x) for x in ffl]
        base = min(ffl_ints)
        config_dict["intermediate_size"] = base
        config_dict["use_double_wide_mlp"] = any(x == 2 * base for x in ffl_ints)

    pattern = gguf_array(reader, "gemma4.attention.sliding_window_pattern")
    if pattern is not None:
        config_dict["layer_types"] = [
            "sliding_attention" if bool(t) else "full_attention" for t in pattern
        ]


def fixup_gemma4_tensor_map(tensor_key_mapping: dict, tensor_names) -> None:
    """Add `.weight` aliases for gemma4's raw-Parameter weights.

    gguf-py maps gemma4's bare nn.Parameters (e.g. `model.layers.N.layer_scalar`,
    which has no `.weight` suffix) to a suffixless GGUF name
    (`blk.N.layer_output_scale`), but llama.cpp actually writes the tensor WITH a
    `.weight` suffix (`blk.N.layer_output_scale.weight`). The load loop looks up
    tensors by their real name and would miss it. For every mapping key whose
    bare name isn't an actual tensor but `<key>.weight` is, add the `.weight`
    alias pointing at the same HF parameter.
    """
    names = set(tensor_names)
    for gguf_name in list(tensor_key_mapping.keys()):
        weighted = f"{gguf_name}.weight"
        if gguf_name not in names and weighted in names:
            tensor_key_mapping[weighted] = tensor_key_mapping[gguf_name]


# ─────────────────────────────── vision ───────────────────────────────


def build_gemma4_vision_config(kv: dict) -> Gemma4VisionConfig:
    """Build the `Gemma4VisionConfig` from the GGUF `gemma4.vision.*` metadata.

    transformers' GGUF config parse only maps the text-decoder scalars, so the
    vision sub-config (a SigLIP-style RoPE ViT) is reconstructed here. `use_clipped
    _linears=True` matches the GGUF, which carries per-linear `input/output min/max`
    clip stats that load into the `Gemma4ClippableLinear` buffers."""
    return Gemma4VisionConfig(
        hidden_size=int(kv["gemma4.vision.embedding_length"]),
        num_hidden_layers=int(kv["gemma4.vision.block_count"]),
        num_attention_heads=int(kv["gemma4.vision.attention.head_count"]),
        intermediate_size=int(kv["gemma4.vision.feed_forward_length"]),
        patch_size=int(kv["gemma4.vision.patch_size"]),
        rms_norm_eps=float(kv["gemma4.vision.attention.layer_norm_epsilon"]),
        use_clipped_linears=True,
    )


# gemma4 vision/projector GGUF tensor names are unknown to transformers' GGUF→HF
# weight map. These tables encode the GGUF→HF correspondence, verified 1:1 against
# `Gemma4ForConditionalGeneration` (659 tensors, 0 shape mismatches).
# Per-layer linears (under `…encoder.layers.{N}.`):
V_LIN = {
    "attn_q": "self_attn.q_proj.linear", "attn_k": "self_attn.k_proj.linear",
    "attn_v": "self_attn.v_proj.linear", "attn_out": "self_attn.o_proj.linear",
    "ffn_gate": "mlp.gate_proj.linear", "ffn_up": "mlp.up_proj.linear",
    "ffn_down": "mlp.down_proj.linear",
}
# Per-layer norms (4 block norms + per-head q/k norms):
V_NORM = {
    "ln1": "input_layernorm", "ln2": "pre_feedforward_layernorm",
    "attn_post_norm": "post_attention_layernorm", "ffn_post_norm": "post_feedforward_layernorm",
    "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm",
}
V_CLIP = ("input_max", "input_min", "output_max", "output_min")
V_LAYER_RE = re.compile(r"v\.blk\.(\d+)\.(\w+?)\.(weight|input_max|input_min|output_max|output_min)$")


def gemma4_vision_target(name: str, prefix: str) -> tuple[str | None, str]:
    """GGUF vision tensor name -> (hf_param_name, transform) or (None, reason).
    `transform`: 'linear'/'direct' = use as-is (gguf data is already HF [out,in]
    orientation), 'patch' = flatten (768,3,16,16)->(768,768), 'clip' = (1,)->scalar."""
    if name.startswith(("mm.a.", "a.")):
        return None, "audio"  # audio modality handled separately
    if name == "mm.input_projection.weight":
        return prefix + "embed_vision.embedding_projection.weight", "linear"
    if name == "v.patch_embd.weight":
        return prefix + "vision_tower.patch_embedder.input_proj.weight", "patch"
    if name == "v.position_embd.weight":
        return prefix + "vision_tower.patch_embedder.position_embedding_table", "direct"
    m = V_LAYER_RE.match(name)
    if m is None:
        return None, "unknown"
    blk, sub, kind = m.group(1), m.group(2), m.group(3)
    base = f"{prefix}vision_tower.encoder.layers.{blk}."
    if kind in V_CLIP and sub in V_LIN:
        return base + V_LIN[sub].removesuffix(".linear") + "." + kind, "clip"
    if kind == "weight" and sub in V_LIN:
        return base + V_LIN[sub] + ".weight", "linear"
    if kind == "weight" and sub in V_NORM:
        return base + V_NORM[sub] + ".weight", "direct"
    return None, "unknown"


def gemma4_vision_tensor(t, how: str, dtype: torch.dtype) -> torch.Tensor:
    if t.tensor_type in GGUF_DENSE:
        arr = np.array(t.data)  # gguf reader already yields HF (out,in) numpy orientation
    else:
        arr = dequantize(np.array(t.data), t.tensor_type).reshape(tuple(int(x) for x in reversed(t.shape)))
    x = torch.from_numpy(arr.astype(np.float32))
    if how == "patch":
        x = x.reshape(x.shape[0], -1)  # (768,3,16,16) -> Linear weight (768, 3*16*16)
    elif how == "clip":
        x = x.reshape(())
    return x.to(dtype)


def gemma4_vision_state_dict(tensors, dtype: torch.dtype, prefix: str = "model.") -> dict[str, torch.Tensor]:
    """Map gemma4 GGUF vision + projector tensors to a {hf_name: tensor} state dict
    for `Gemma4ForConditionalGeneration.load_state_dict` (weights + clip-stat
    buffers). Audio (`mm.a.*`/`a.*`) is skipped (handled by the audio adapter).
    `tensors` is an iterable of gguf ReaderTensor-likes."""
    sd: dict[str, torch.Tensor] = {}
    for t in tensors:
        if not t.name.startswith(("v.", "mm.")):
            continue
        hf, how = gemma4_vision_target(t.name, prefix)
        if hf is not None:
            sd[hf] = gemma4_vision_tensor(t, how, dtype)
    return sd


class Gemma4VisionHolder(torch.nn.Module):
    """The dense (eager) half of gemma4 multimodal: the SigLIP RoPE ViT vision
    tower + the projector that lifts its output into the language model's
    embedding space. Runs on CPU in float32 (alloy's quantized decode is a
    separate path; image features splice into its text embeddings)."""

    def __init__(self, vision_config: Gemma4VisionConfig, text_config) -> None:
        super().__init__()
        # Skip HF's random init: the GGUF load_state_dict overwrites every param.
        with no_init_weights():
            self.vision_tower = Gemma4VisionModel(vision_config)
            self.embed_vision = Gemma4MultimodalEmbedder(vision_config, text_config)

    def get_image_features(
        self, pixel_values: torch.Tensor, image_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        vision_out = self.vision_tower(
            pixel_values=pixel_values, pixel_position_ids=image_position_ids,
        )
        return self.embed_vision(inputs_embeds=vision_out.last_hidden_state)


class Gemma4VitEncode(torch.nn.Module):
    """Fixed-shape patch-embed + 16-layer ViT encoder -> (1, max_patches, hidden).
    The integer position bookkeeping (patch one-hot selector, padding-keep mask) is
    precomputed on CPU and passed in, so the graph is pure tensor compute (matmuls,
    attention, norms) — one alloy plan, like the text decoder."""

    def __init__(self, vision_tower: torch.nn.Module) -> None:
        super().__init__()
        self.vision_tower = vision_tower

    def forward(
        self,
        pixel_values: torch.Tensor,
        pos_one_hot: torch.Tensor,
        keep: torch.Tensor,
        attn_mask: torch.Tensor,
        pixel_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        pe = self.vision_tower.patch_embedder
        # Patch embed: input projection + position embedding (one_hot @ table, the
        # one_hot precomputed). `keep` zeros padding patches (HF's where(padding,0)).
        x = pe.input_proj(2.0 * (pixel_values - 0.5))
        pos = (pos_one_hot @ pe.position_embedding_table).sum(dim=1) * keep
        return cast(torch.Tensor, self.vision_tower.encoder(
            inputs_embeds=x + pos,
            attention_mask=attn_mask,
            pixel_position_ids=pixel_position_ids,
        ).last_hidden_state)


class Gemma4VitPool(torch.nn.Module):
    """Fixed-shape pooler + projector: encoder output -> (1, output_length,
    text_hidden). The pooling grid average is a matmul with precomputed one-hot/k^2
    weights, then the projector lifts to LM space. A separate alloy plan from the
    encoder so the encoder output materializes between them (fusing the pooling
    matmul across the encoder boundary produces NaN)."""

    def __init__(self, vision_tower: torch.nn.Module, embed_vision: torch.nn.Module) -> None:
        super().__init__()
        self.pooler = vision_tower.pooler
        self.embed_vision = embed_vision
        self.root_hidden_size = float(vision_tower.pooler.root_hidden_size)

    def forward(
        self, enc: torch.Tensor, keep: torch.Tensor, pool_weights_t: torch.Tensor
    ) -> torch.Tensor:
        pooled = (pool_weights_t @ (enc * keep).float()).to(enc.dtype) * self.root_hidden_size
        return cast(torch.Tensor, self.embed_vision(inputs_embeds=pooled))


class Gemma4VisionAdapter(ModalityAdapter):
    """Turns image bytes into language-model-space vision features for gemma4, and
    encodes gemma4's image-placeholder expansion. Implements the `Modality`
    protocol the served model consumes — the server stays model-agnostic.

    Owns the dense vision tower + projector (loaded from the GGUF's `v.*`/`mm.*`
    tensors) and the HF image processor. `encode` reproduces HF
    `get_image_features`: pre-patchified pixels (dynamic resolution) → ViT →
    projector → `(num_soft_tokens, text_hidden)`. The caller splices these into the
    placeholder slots of the decoder's text embeddings.
    """

    placeholder_token_id = GEMMA4_IMAGE_TOKEN_ID
    PLACEHOLDER = "<|image|>"
    OPEN = "<|image>"
    CLOSE = "<image|>"
    ITEM_NOUN = "image"

    def __init__(
        self, tensors, vision_kv: dict, text_config, dtype: torch.dtype = torch.float32,
    ) -> None:
        holder = Gemma4VisionHolder(build_gemma4_vision_config(vision_kv), text_config)
        state = gemma4_vision_state_dict(tensors, dtype, prefix="")
        # strict=False tolerates registered-but-not-in-GGUF buffers (rope inv_freq);
        # the 1:1 GGUF↔HF map is verified separately (659 tensors, 0 mismatches).
        holder.load_state_dict(state, strict=False)
        self.holder = holder.eval().to(dtype)
        # The whole vision tensor compute (patch + 16-layer encoder + pooler +
        # projector) runs through alloy's Metal dispatch as one fixed-shape plan,
        # like the text decoder; only the position bookkeeping + the variable
        # valid-token gather stay on CPU (see bookkeeping).
        self.encode_mod = Gemma4VitEncode(holder.vision_tower)
        self.pool_mod = Gemma4VitPool(holder.vision_tower, holder.embed_vision)
        self.vit_encode = cast(
            torch.nn.Module,
            torch.compile(self.encode_mod, backend="alloy", dynamic=False),
        )
        self.vit_pool = cast(
            torch.nn.Module,
            torch.compile(self.pool_mod, backend="alloy", dynamic=False),
        )
        self.image_processor = Gemma4ImageProcessor()
        self.dtype = dtype

    def bookkeeping(self, pid: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Precompute (on CPU) the position-derived selectors the fixed-shape alloy
        graph needs: the patch position one-hot, the padding-keep mask, the encoder
        attention mask, the pooling-grid weight matrix, and the valid-bin mask."""
        vt = self.holder.vision_tower
        pe = vt.patch_embedder
        padding = (pid == -1).all(dim=-1)  # (1, S) True = padding patch
        attn_mask = ~padding
        keep = (~padding).to(self.dtype).unsqueeze(-1)  # (1, S, 1)
        clamped = pid.clamp(min=0)
        pos_one_hot = torch.nn.functional.one_hot(
            clamped, num_classes=pe.position_embedding_size
        ).permute(0, 2, 1, 3).to(self.dtype)  # (1, 2, S, pos_emb_size)
        # Pooling grid: assign each patch to a k*k bin, weight 1/k^2 (HF's avg pool).
        seq_len = int(pid.shape[1])
        length = seq_len // (vt.config.pooling_kernel_size ** 2)
        k = int((seq_len // length) ** 0.5)
        max_x = clamped[..., 0].max(dim=-1, keepdim=True)[0] + 1
        kernel_idxs = torch.div(clamped, k, rounding_mode="floor")
        kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
        weights = torch.nn.functional.one_hot(kernel_idxs.long(), length).to(self.dtype) / (k * k)
        pool_weights_t = weights.transpose(1, 2)  # (1, length, S)
        pool_mask = torch.logical_not((weights == 0).all(dim=1))  # (1, length) valid bins
        return pos_one_hot, keep, attn_mask, pool_weights_t, pool_mask

    def encode(self, image) -> torch.Tensor:
        """`image` is raw encoded bytes or a PIL.Image. Returns `(num_soft_tokens,
        text_hidden)` — one feature row per image soft token."""
        if isinstance(image, (bytes, bytearray)):
            image = Image.open(io.BytesIO(image)).convert("RGB")
        processed = self.image_processor(images=[image], return_tensors="pt")
        pixel_values = processed["pixel_values"].to(self.dtype)
        pid = processed["image_position_ids"]
        pos_one_hot, keep, attn_mask, pool_weights_t, pool_mask = self.bookkeeping(pid)
        with torch.inference_mode():
            enc = self.vit_encode(pixel_values, pos_one_hot, keep, attn_mask, pid)
            feats = self.vit_pool(enc, keep, pool_weights_t)
        return feats[0][pool_mask[0]]  # CPU gather of the valid soft tokens

    def eager_compile_all(self) -> None:
        """Compile the two alloy vision plans (encode / pool) ahead of the first
        real request — the ViT input shape is fixed, so one dummy image compiles
        both stages."""
        self.encode(Image.new("RGB", (64, 64)))

    def capture_targets(self) -> list[CaptureTarget]:
        """The two compiled vision stages + fixed-shape inputs from a dummy image,
        for `alloy tune` / `alloy profile`."""
        processed = self.image_processor(
            images=[Image.new("RGB", (64, 64))], return_tensors="pt"
        )
        pid = processed["image_position_ids"]
        pixel_values = processed["pixel_values"].to(self.dtype)
        pos_one_hot, keep, attn_mask, pool_weights_t, _ = self.bookkeeping(pid)
        hidden = self.holder.vision_tower.config.hidden_size
        enc = torch.zeros((1, int(pid.shape[1]), hidden), dtype=self.dtype)
        return [
            CaptureTarget(
                name="vision_encode",
                label="vision encode (patch + 16-layer ViT)",
                module=self.encode_mod,
                inputs={
                    "pixel_values": pixel_values,
                    "pos_one_hot": pos_one_hot,
                    "keep": keep,
                    "attn_mask": attn_mask,
                    "pixel_position_ids": pid,
                },
            ),
            CaptureTarget(
                name="vision_pool",
                label="vision pool + projector",
                module=self.pool_mod,
                inputs={"enc": enc, "keep": keep, "pool_weights_t": pool_weights_t},
            ),
        ]


def gguf_has_vision(tensors) -> bool:
    """True if these GGUF tensors include gemma4 vision weights (vs text-only)."""
    return any(t.name.startswith("v.blk.") for t in tensors)


# ─────────────────────────────── audio ───────────────────────────────


def build_gemma4_audio_config(kv: dict) -> Gemma4AudioConfig:
    """Build the conformer config from the GGUF `gemma4.audio.*` metadata. The HF
    defaults already match the shipped e2b/e4b geometry; override the handful of
    keys the GGUF carries so a future variant with different dims still resolves."""
    cfg = Gemma4AudioConfig()
    if "gemma4.audio.embedding_length" in kv:
        cfg.hidden_size = int(kv["gemma4.audio.embedding_length"])
    if "gemma4.audio.block_count" in kv:
        cfg.num_hidden_layers = int(kv["gemma4.audio.block_count"])
    if "gemma4.audio.attention.head_count" in kv:
        cfg.num_attention_heads = int(kv["gemma4.audio.attention.head_count"])
    if "gemma4.audio.conv_kernel_size" in kv:
        cfg.conv_kernel_size = int(kv["gemma4.audio.conv_kernel_size"])
    if "gemma4.audio.attention.layer_norm_epsilon" in kv:
        cfg.rms_norm_eps = float(kv["gemma4.audio.attention.layer_norm_epsilon"])
    cfg.use_clipped_linears = True  # GGUF carries per-linear input/output clip stats
    return cfg


A_CLIP = ("input_max", "input_min", "output_max", "output_min")
# GGUF clipped-linear → HF Gemma4ClippableLinear submodule path (relative to the layer).
A_LINEAR = {
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_out": "self_attn.post",
    "ffn_up": "feed_forward1.ffw_layer_1",
    "ffn_down": "feed_forward1.ffw_layer_2",
    "ffn_up_1": "feed_forward2.ffw_layer_1",
    "ffn_down_1": "feed_forward2.ffw_layer_2",
    "conv_pw1": "lconv1d.linear_start",
    "conv_pw2": "lconv1d.linear_end",
}
# GGUF plain weight/param/norm → HF path (relative to the layer).
A_DIRECT = {
    "linear_pos.weight": "self_attn.relative_k_proj.weight",
    "per_dim_scale.weight": "self_attn.per_dim_scale",
    "conv_dw.weight": "lconv1d.depthwise_conv1d.weight",
    "conv_norm.weight": "lconv1d.conv_norm.weight",
    "norm_conv.weight": "lconv1d.pre_layer_norm.weight",
    "ffn_norm.weight": "feed_forward1.pre_layer_norm.weight",
    "ffn_post_norm.weight": "feed_forward1.post_layer_norm.weight",
    "ffn_norm_1.weight": "feed_forward2.pre_layer_norm.weight",
    "ffn_post_norm_1.weight": "feed_forward2.post_layer_norm.weight",
    "ln1.weight": "norm_pre_attn.weight",
    "ln2.weight": "norm_post_attn.weight",
    "layer_pre_norm.weight": "norm_out.weight",
}
# GGUF non-block names → HF tower path. `mm.a.fc` is the tower's final 1024→1536
# output_proj (carries a bias); `mm.a.*` is otherwise the LM-space projector.
A_NONBLOCK = {
    "a.pre_encode.out.weight": "subsample_conv_projection.input_proj_linear.weight",
    "a.conv1d.0.weight": "subsample_conv_projection.layer0.conv.weight",
    "a.conv1d.0.norm.weight": "subsample_conv_projection.layer0.norm.weight",
    "a.conv1d.1.weight": "subsample_conv_projection.layer1.conv.weight",
    "a.conv1d.1.norm.weight": "subsample_conv_projection.layer1.norm.weight",
    "mm.a.fc.weight": "output_proj.weight",
    "mm.a.fc.bias": "output_proj.bias",
}


def audio_target(name: str) -> tuple[str | None, str]:
    """GGUF audio tensor name → (HF tower param, transform) or (None, reason)."""
    if name in A_NONBLOCK:
        return A_NONBLOCK[name], "direct"
    m = re.match(r"a\.blk\.(\d+)\.(.+)", name)
    if m is None:
        return None, "not-tower"  # e.g. mm.a.input_projection (the LM projector)
    layer, rest = f"layers.{m.group(1)}.", m.group(2)
    for ggml, hf in A_LINEAR.items():
        if rest == f"{ggml}.weight":
            return layer + hf + ".linear.weight", "direct"
        for clip in A_CLIP:
            if rest == f"{ggml}.{clip}":
                return layer + hf + "." + clip, "clip"
    if rest in A_DIRECT:
        how = "conv_dw" if rest == "conv_dw.weight" else (
            "scale" if rest == "per_dim_scale.weight" else "direct"
        )
        return layer + A_DIRECT[rest], how
    return None, "unmapped"


def audio_load_tensor(t, how: str, dtype: torch.dtype) -> torch.Tensor:
    """Decode a GGUF tensor to `dtype`. bf16 has no numpy type, so its raw uint16
    bits are reinterpreted into the float32 high bits, then cast."""
    if t.tensor_type == gguf.GGMLQuantizationType.BF16:
        u16 = np.ascontiguousarray(np.array(t.data)).view(np.uint16)
        arr = (u16.astype(np.uint32) << 16).view(np.float32)
    elif t.tensor_type in GGUF_DENSE:  # F32 / F16 — gguf yields the typed values
        arr = np.array(t.data).astype(np.float32)
    else:
        arr = dequantize(np.array(t.data), t.tensor_type).reshape(
            tuple(int(x) for x in reversed(t.shape))
        )
    x = torch.from_numpy(np.ascontiguousarray(arr).astype(np.float32))
    if how == "clip":
        x = x.reshape(())
    elif how == "conv_dw":  # (C, K) → depthwise Conv1d weight (C, 1, K)
        x = x.reshape(x.shape[0], 1, x.shape[1])
    elif how == "scale":
        x = x.reshape(-1)
    return x.to(dtype)


def gemma4_audio_state_dict(tensors, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """GGUF `a.*` + `mm.a.fc` → `Gemma4AudioModel` state dict (tower)."""
    state: dict[str, torch.Tensor] = {}
    for t in tensors:
        if not t.name.startswith(("a.", "mm.a.fc")):
            continue
        hf, how = audio_target(t.name)
        if hf is not None:
            state[hf] = audio_load_tensor(t, how, dtype)
    return state


def gemma4_audio_embedder_state_dict(tensors, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """GGUF `mm.a.input_projection` → `Gemma4MultimodalEmbedder` state dict. The
    pre-projection norm is scale-free (no weight tensor)."""
    for t in tensors:
        if t.name == "mm.a.input_projection.weight":
            return {"embedding_projection.weight": audio_load_tensor(t, "direct", dtype)}
    raise ValueError("mm.a.input_projection.weight not found in GGUF audio tensors")


# Functional conformer forwards. The HF forwards mutate in place; alloy miscompiles
# that in f16 (an AOT-functionalization × fusion interaction). These reproduce them
# out-of-place.


def audio_ff_forward(self, hidden):
    residual = hidden
    hidden = self.pre_layer_norm(hidden)
    hidden = self.ffw_layer_1(hidden)
    hidden = self.act_fn(hidden)
    hidden = self.ffw_layer_2(hidden)
    hidden = self.post_layer_norm(hidden)
    return hidden * self.post_layer_scale + residual


def audio_lconv_forward(self, hidden):
    residual = hidden
    hidden = self.pre_layer_norm(hidden)
    hidden = self.linear_start(hidden)
    hidden = torch.nn.functional.glu(hidden, dim=-1)
    hidden = self.depthwise_conv1d(hidden.transpose(1, 2)).transpose(1, 2)
    hidden = self.conv_norm(hidden)
    hidden = self.act_fn(hidden)
    hidden = self.linear_end(hidden)
    return hidden + residual


def audio_layer_forward(self, hidden, attention_mask=None, position_embeddings=None, **kwargs):
    gc = min(self.gradient_clipping, torch.finfo(hidden.dtype).max)  # 1e10 overflows f16
    hidden = self.feed_forward1(hidden)
    residual = hidden
    hidden = torch.clamp(hidden, -gc, gc)
    hidden = self.norm_pre_attn(hidden)
    hidden, _ = self.self_attn(
        hidden_states=hidden, position_embeddings=position_embeddings, attention_mask=attention_mask
    )
    hidden = torch.clamp(hidden, -gc, gc)
    hidden = self.norm_post_attn(hidden)
    hidden = hidden + residual
    hidden = self.lconv1d(hidden)
    hidden = self.feed_forward2(hidden)
    hidden = torch.clamp(hidden, -gc, gc)
    hidden = self.norm_out(hidden)
    return hidden


def patch_audio_functional(tower: Gemma4AudioModel) -> Gemma4AudioModel:
    for layer in tower.layers:
        layer.forward = types.MethodType(audio_layer_forward, layer)
        layer.feed_forward1.forward = types.MethodType(audio_ff_forward, layer.feed_forward1)
        layer.feed_forward2.forward = types.MethodType(audio_ff_forward, layer.feed_forward2)
        layer.lconv1d.forward = types.MethodType(audio_lconv_forward, layer.lconv1d)
    return tower


class Gemma4AudioModule(torch.nn.Module):
    """Fixed-shape alloy graph: subsample → 12 conformer layers (precomputed mask)
    → output_proj → LM-space projector. Returns `(1, frames, text_hidden)`."""

    def __init__(self, tower: Gemma4AudioModel, embedder: Gemma4MultimodalEmbedder):
        super().__init__()
        self.tower = tower
        self.embedder = embedder

    def forward(self, input_features, input_mask, mask_5d, position_embeddings):
        hidden, _ = self.tower.subsample_conv_projection(input_features, input_mask)
        for layer in self.tower.layers[: self.tower.config.num_hidden_layers]:
            hidden = layer(
                hidden, attention_mask=mask_5d, position_embeddings=position_embeddings
            )
        hidden = self.tower.output_proj(hidden)
        return self.embedder(hidden)


class Gemma4AudioAdapter(ModalityAdapter):
    """Turns audio bytes into language-model-space soft tokens for gemma4, and
    encodes gemma4's audio-placeholder expansion. Implements the `Modality`
    protocol shape the served model consumes (audio is just another modality)."""

    placeholder_token_id = GEMMA4_AUDIO_TOKEN_ID
    PLACEHOLDER = "<|audio|>"
    OPEN = "<|audio>"
    CLOSE = "<audio|>"
    ITEM_NOUN = "audio"

    def __init__(self, tensors, audio_kv: dict, text_config, dtype: torch.dtype = torch.float16):
        cfg = build_gemma4_audio_config(audio_kv)
        # Skip HF's random init: the GGUF load_state_dict overwrites every param.
        with no_init_weights():
            tower = Gemma4AudioModel(cfg)
            embedder = Gemma4MultimodalEmbedder(cfg, text_config)
        tower.load_state_dict(gemma4_audio_state_dict(tensors, torch.float32), strict=False)
        embedder.load_state_dict(
            gemma4_audio_embedder_state_dict(tensors, torch.float32), strict=False
        )
        self.cfg = cfg
        self.dtype = dtype
        # Eager (CPU) tower for the mask/position bookkeeping; the same weights,
        # functionally patched, run the compiled forward.
        self.tower = patch_audio_functional(tower).eval().to(dtype)
        self.embedder = embedder.eval().to(dtype)
        self.module = Gemma4AudioModule(self.tower, self.embedder)
        self.compiled = cast(
            torch.nn.Module, torch.compile(self.module, backend="alloy", dynamic=False)
        )
        self.feature_extractor = Gemma4AudioFeatureExtractor()

    def bookkeeping(self, input_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Precompute the 5D blocked attention mask + relative-position embeddings
        the fixed-shape graph needs. The subsample halves the frame mask per conv
        layer (`mask[:, ::2]` ×2); the rest is HF's own mask construction run on
        CPU (bool/index math that doesn't belong in a Metal plan)."""
        output_mask = input_mask[:, ::2][:, ::2]
        frames = int(output_mask.shape[1])
        hidden = torch.zeros((1, frames, self.cfg.hidden_size), dtype=self.dtype)
        with torch.inference_mode():
            mask_4d = create_bidirectional_mask(
                config=self.cfg,
                inputs_embeds=hidden,
                attention_mask=output_mask,
                and_mask_function=sliding_window_mask_function(
                    (self.cfg.attention_context_left - 1, self.cfg.attention_context_right)
                ),
            )
            mask_5d = self.tower._convert_4d_mask_to_blocked_5d(mask_4d)
            position_embeddings = self.tower.rel_pos_enc(hidden)
        return mask_5d, position_embeddings

    def compute_features(self, input_features: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
        mask_5d, pos = self.bookkeeping(input_mask)
        with torch.inference_mode():
            feats = self.compiled(input_features.to(self.dtype), input_mask, mask_5d, pos)
        return feats[0].float()  # (frames, text_hidden)

    def encode(self, audio) -> torch.Tensor:
        """`audio` is raw encoded bytes (wav/flac/…) or a (waveform, sr) pair.
        Returns `(num_soft_tokens, text_hidden)` — one row per audio soft token."""
        waveform = self.decode_audio(audio)
        proc = self.feature_extractor(
            [waveform], sampling_rate=self.feature_extractor.sampling_rate, return_tensors="pt"
        )
        return self.compute_features(proc["input_features"], proc["input_features_mask"])

    def decode_audio(self, audio) -> np.ndarray:
        """Decode audio bytes (or a (waveform, sr) pair) to a 16 kHz mono float32
        waveform — the rate the feature extractor expects."""
        target_sr = self.feature_extractor.sampling_rate
        if isinstance(audio, tuple):
            waveform, sr = audio
            waveform = np.asarray(waveform, dtype=np.float32)
        else:
            waveform, sr = soundfile.read(
                io.BytesIO(bytes(audio)), dtype="float32", always_2d=False
            )
        if waveform.ndim > 1:  # downmix to mono
            waveform = waveform.mean(axis=1)
        if sr != target_sr:
            import librosa  # scoped: heavy optional dep, only when the clip needs resampling

            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
        return np.ascontiguousarray(waveform, dtype=np.float32)

    def eager_compile_all(self) -> None:
        """Compile the alloy audio plan ahead of the first request. Unlike the
        fixed-shape ViT, the conformer length follows the clip duration, so warm
        with a ~1s tone (the plan recompiles per length, like text prefill)."""
        tone = (0.1 * np.sin(2 * np.pi * 220.0 / 16000 * np.arange(16000))).astype(np.float32)
        self.encode((tone, 16000))

    def capture_targets(self) -> list[CaptureTarget]:
        """The compiled audio stage + fixed-shape inputs from a ~1s dummy tone, for
        `alloy tune` / `alloy profile`."""
        tone = (0.1 * np.sin(2 * np.pi * 220.0 / 16000 * np.arange(16000))).astype(np.float32)
        proc = self.feature_extractor(
            [tone], sampling_rate=self.feature_extractor.sampling_rate, return_tensors="pt"
        )
        input_features = proc["input_features"].to(self.dtype)
        input_mask = proc["input_features_mask"]
        mask_5d, pos = self.bookkeeping(input_mask)
        return [
            CaptureTarget(
                name="audio_encode",
                label="audio conformer (subsample + 12 layers + projector)",
                module=self.module,
                inputs={
                    "input_features": input_features,
                    "input_mask": input_mask,
                    "mask_5d": mask_5d,
                    "position_embeddings": pos,
                },
            )
        ]


def gguf_has_audio(tensors) -> bool:
    return any(t.name.startswith("a.blk.") for t in tensors)


# ─────────────────────────────── handler ───────────────────────────────


@register("gemma4")
class Gemma4Handler(CausalLMHandler):
    """gemma4 (multimodal: text + vision + audio)."""

    arch = ("gemma4",)
    kind = "chat"
    config_mapping = GEMMA4_CONFIG_MAPPING

    def apply_transformers_patches(self) -> None:
        if "gemma4" in GGUF_CONFIG_MAPPING:
            return
        GGUF_CONFIG_MAPPING["gemma4"] = GEMMA4_CONFIG_MAPPING
        # gemma4 uses the same SPM Gemma tokenizer family as gemma3.
        GGUF_TO_FAST_CONVERTERS["gemma4"] = GGUF_TO_FAST_CONVERTERS["gemma3_text"]
        if "gemma4" not in GGUF_SUPPORTED_ARCHITECTURES:
            GGUF_SUPPORTED_ARCHITECTURES.append("gemma4")

    def config_fixup(self, config_dict: dict, reader) -> None:
        apply_gemma4_post_load_config_fixup(config_dict, reader)

    def fixup_tensor_map(self, tensor_key_mapping: dict, tensor_names) -> None:
        fixup_gemma4_tensor_map(tensor_key_mapping, tensor_names)

    def chat_template(self) -> str | None:
        return gemma4_chat_template()

    def build_vision(self, tensors, vision_meta, model, tokenizer):
        if not gguf_has_vision(tensors):
            return None
        return Gemma4VisionAdapter(tensors, vision_meta, model.config)

    def build_audio(self, tensors, audio_meta, model):
        if not gguf_has_audio(tensors):
            return None
        return Gemma4AudioAdapter(tensors, audio_meta, model.config)
