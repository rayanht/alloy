"""Qwen 3.5 (arch=`qwen35`) + Qwen3.5-MoE (`qwen35moe`) model handlers.

Owns the qwen3.5-specific GGUF interpretation — the config mappings, the hybrid
linear/full-attention config fixup, the GGUF→HF tensor processor, the
transformers-registry patches — plus the MoE quantized expert install
(`GGUFQwen35MoeExperts`/`GGUFQwen35MoeBlock`) and the vision adapter wiring. The
generic GGUF-loading infrastructure lives in `gguf/transformers_compat.py`.

Qwen 3.5 is a hybrid: full-attention layers (gated; `attn_q/k/v/o`) interleave
with linear-attention layers (`Qwen3_5GatedDeltaNet`, the `ssm_*` tensors).
`attention.head_count_kv` is a per-layer list (0 = linear, >0 = full-attn) — the
GGUF-side mapping can't express this, so the post-load fixup drops both
`num_key_value_heads` and `layer_types` into the parsed config. Qwen3.5-MoE
shares this backbone; only the FFN (a fine-grained MoE block) differs.
"""

from __future__ import annotations

import re

import gguf
import numpy as np
import torch
from transformers import modeling_gguf_pytorch_utils
from transformers.integrations.ggml import (
    GGUF_CONFIG_MAPPING,
    GGUF_TO_FAST_CONVERTERS,
)
from transformers.modeling_gguf_pytorch_utils import (
    GGUF_SUPPORTED_ARCHITECTURES,
    GGUFTensor,
)

from alloy_server.gguf.quant import module_for_parameter, module_parent
from alloy_server.gguf.transformers_compat import BYPASS_CONFIG_FIXUPS
from alloy_server.models.base import CausalLMHandler
from alloy_server.models.qwen3_5_vision import build_qwen35_vision_adapter, gguf_has_vision_qwen35
from alloy_server.models.registry import register


# Config-key translation. `attention.head_count_kv` is intentionally
# omitted from this mapping — it's per-layer in the GGUF and needs the
# layer-list fix-up in `apply_qwen35_post_load_config_fixup` below.
QWEN3_5_CONFIG_MAPPING = {
    "context_length": "max_position_embeddings",
    "block_count": "num_hidden_layers",
    "feed_forward_length": "intermediate_size",
    "embedding_length": "hidden_size",
    "rope.freq_base": "rope_theta",
    "attention.head_count": "num_attention_heads",
    # Per-layer list in the GGUF; transformers will land it as a list
    # under `num_key_value_heads`. The post-load fixup splits it into
    # `layer_types` + scalar `num_key_value_heads`.
    "attention.head_count_kv": "num_key_value_heads",
    "attention.key_length": "head_dim",
    "attention.layer_norm_rms_epsilon": "rms_norm_eps",
    "vocab_size": "vocab_size",
    "ssm.conv_kernel": "linear_conv_kernel_dim",
    "ssm.state_size": "linear_key_head_dim",
    # `ssm.group_count` is the KEY/group head count → `linear_num_key_heads`.
    # Mapping it to *value* (as a prior revision did) collapsed NV to NK=16
    # and built the value projections at half width, so the GVA 4B/9B models
    # failed to load with a Q4_K shape mismatch.
    "ssm.group_count": "linear_num_key_heads",
    # `ssm.inner_size` (= num_value_heads * value_head_dim) must be mapped for
    # the GGUF loader to derive `linear_num_value_heads` = inner_size/head_dim
    # — 32 for the GVA 4B/9B, 16 for 0.8B/2B. Without this entry NV is left
    # unset. The loader consumes the value into linear_num_value_heads; the
    # `linear_inner_size` landing key does not survive into the HF config.
    "ssm.inner_size": "linear_inner_size",
}


# Qwen3.5-MoE (GGUF arch `qwen35moe`, shipped by ollama as "qwen3.6:35b"):
# the same hybrid linear/full-attention backbone as the dense models, but the
# dense FFN is replaced by a fine-grained MoE block. The four extra GGUF keys
# describe the experts; everything else maps exactly as in the dense case.
QWEN3_5_MOE_CONFIG_MAPPING = {
    **QWEN3_5_CONFIG_MAPPING,
    "expert_count": "num_experts",
    "expert_used_count": "num_experts_per_tok",
    "expert_feed_forward_length": "moe_intermediate_size",
    "expert_shared_feed_forward_length": "shared_expert_intermediate_size",
}


DT_BIAS_PATTERN = ".linear_attn.dt_bias"


class Qwen35TensorProcessor(modeling_gguf_pytorch_utils.TensorProcessor):
    """GGUF→HF tensor adapter for Qwen 3.5 hybrid layers.

    Two transformations the upstream MambaTensorProcessor would *almost*
    cover, except its `"ssm_a" in name` substring match overzealously
    also matches `ssm_alpha` and `ssm_beta` (Q8_0-quantized projection
    weights, not the A_log scalar). Match the trailing component exactly.

      - `blk.N.ssm_conv1d.weight`: GGUF stores (D, 4); PyTorch's depthwise
        Conv1d expects (D, 1, 4) with groups=D. Insert a size-1 dim.
      - `blk.N.ssm_a`: GGUF stores raw A values (negative); the model
        parameter is `A_log = log(-A)`. Apply the Mamba-convention transform.

    Also overrides `perform_fallback_tensor_mapping` to wire
    `linear_attn.dt_bias` to `blk.N.ssm_dt` (that HF→GGUF entry isn't in
    gguf-py's qwen35 TensorNameMap yet).
    """

    def process(self, weights, name, **kwargs):
        trailing = name.rsplit(".", 1)[-1]
        if name.endswith("ssm_conv1d.weight"):
            weights = np.expand_dims(weights, axis=1)
        elif trailing == "ssm_a":
            weights = np.log(-weights)
        elif "norm.weight" in name and "ssm_norm" not in name:
            # llama.cpp's qwen35 converter pre-adds 1 to ALL norm.weight tensors
            # EXCEPT linear_attn.norm.weight (RMSNormGated, stored raw). The HF
            # Qwen3_5RMSNorm forward applies `(1 + weight) * normed`, so subtract
            # here to restore the raw trained weight (Gemma convention). Skip
            # ssm_norm (== linear_attn.norm), the RMSNormGated stored raw.
            weights = weights - 1
        return GGUFTensor(weights, name, {})

    def perform_fallback_tensor_mapping(
        self, gguf_to_hf_name_map, suffix, qual_name, hf_name,
    ):
        if hf_name.endswith(DT_BIAS_PATTERN):
            # `model.layers.{N}.linear_attn.dt_bias` -> `blk.{N}.ssm_dt`
            head = hf_name[: -len(DT_BIAS_PATTERN)]  # e.g., "model.layers.0"
            layer_num = head.rsplit(".", 1)[-1]
            gguf_to_hf_name_map[f"blk.{layer_num}.ssm_dt{suffix}"] = qual_name + hf_name


def apply_qwen35_hybrid_linear_fixup(config_dict: dict) -> None:
    """Fill the per-layer attention / linear-attention HF keys shared by the
    dense and MoE Qwen3.5 hybrids. Mutates `config_dict` in place. Idempotent."""
    # head_count_kv may already have landed under a translated key
    # (`num_key_value_heads`) as a list — handle both shapes.
    raw_list = config_dict.pop("qwen35.attention.head_count_kv", None)
    if raw_list is None and isinstance(config_dict.get("num_key_value_heads"), list):
        raw_list = config_dict.pop("num_key_value_heads")
    if isinstance(raw_list, list):
        layer_types = [
            "linear_attention" if int(h) == 0 else "full_attention" for h in raw_list
        ]
        config_dict["layer_types"] = layer_types
        attention_kv_counts = [int(h) for h in raw_list if int(h) > 0]
        if attention_kv_counts:
            config_dict["num_key_value_heads"] = max(attention_kv_counts)
        else:
            config_dict["num_key_value_heads"] = config_dict.get("num_attention_heads", 1)

    # Linear-attention head counts. `linear_num_key_heads` comes from
    # `ssm.group_count`. The VALUE head count is `ssm.inner_size // head_dim`:
    # 32 for the GVA 4B/9B (inner_size=4096, head_dim=128), 16 for 0.8B/2B.
    # Derive it from the temporary `linear_inner_size` landing key.
    state_size = config_dict.get("linear_key_head_dim")
    inner_size = config_dict.pop("linear_inner_size", None)
    if inner_size is not None and state_size:
        config_dict["linear_num_value_heads"] = int(inner_size) // int(state_size)
    elif config_dict.get("linear_num_value_heads") is None:
        config_dict["linear_num_value_heads"] = int(
            config_dict.get("linear_num_key_heads", 1)
        )

    # ssm.state_size is the shared per-head dim for both key and value heads.
    if state_size is not None:
        config_dict["linear_value_head_dim"] = int(state_size)


def apply_qwen35_post_load_config_fixup(config_dict: dict) -> None:
    """Post-process a GGUF-parsed dense Qwen3.5 config into HF keys, then force
    the text-only model class. Mutates `config_dict` in place. Idempotent."""
    apply_qwen35_hybrid_linear_fixup(config_dict)
    config_dict["architectures"] = ["Qwen3_5ForCausalLM"]
    config_dict["model_type"] = "qwen3_5_text"
    # Drop vision.* / ssm.* keys that AutoConfig doesn't understand on
    # Qwen3_5TextConfig (it errors on extras under strict mode).
    for k in list(config_dict.keys()):
        if k.startswith("qwen35.vision") or k.startswith("vision.") or k.startswith("ssm."):
            config_dict.pop(k, None)


def apply_qwen35moe_post_load_config_fixup(config_dict: dict) -> None:
    """Post-process a GGUF-parsed Qwen3.5-MoE config (`qwen35moe`) into HF keys,
    then force the text-only MoE model class. Mutates in place. Idempotent."""
    apply_qwen35_hybrid_linear_fixup(config_dict)
    config_dict["architectures"] = ["Qwen3_5MoeForCausalLM"]
    config_dict["model_type"] = "qwen3_5_moe_text"
    for k in list(config_dict.keys()):
        if (
            k.startswith("qwen35moe.")
            or k.startswith("qwen35.vision")
            or k.startswith("vision.")
            or k.startswith("ssm.")
        ):
            config_dict.pop(k, None)


def patch_transformers_ggml() -> None:
    if "qwen3_5" in GGUF_CONFIG_MAPPING:
        return
    modeling_gguf_pytorch_utils.TENSOR_PROCESSORS["qwen35"] = Qwen35TensorProcessor
    GGUF_CONFIG_MAPPING["qwen3_5"] = QWEN3_5_CONFIG_MAPPING
    GGUF_CONFIG_MAPPING["qwen35"] = QWEN3_5_CONFIG_MAPPING
    GGUF_TO_FAST_CONVERTERS["qwen3_5"] = GGUF_TO_FAST_CONVERTERS["qwen3"]
    GGUF_TO_FAST_CONVERTERS["qwen35"] = GGUF_TO_FAST_CONVERTERS["qwen3"]
    GGUF_TO_FAST_CONVERTERS["qwen3_5_text"] = GGUF_TO_FAST_CONVERTERS["qwen3"]
    if "qwen3_5" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen3_5")
    if "qwen35" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen35")

    modeling_gguf_pytorch_utils.TENSOR_PROCESSORS["qwen35moe"] = Qwen35TensorProcessor
    GGUF_CONFIG_MAPPING["qwen3_5_moe"] = QWEN3_5_MOE_CONFIG_MAPPING
    GGUF_CONFIG_MAPPING["qwen35moe"] = QWEN3_5_MOE_CONFIG_MAPPING
    GGUF_TO_FAST_CONVERTERS["qwen3_5_moe"] = GGUF_TO_FAST_CONVERTERS["qwen3"]
    GGUF_TO_FAST_CONVERTERS["qwen35moe"] = GGUF_TO_FAST_CONVERTERS["qwen3"]
    GGUF_TO_FAST_CONVERTERS["qwen3_5_moe_text"] = GGUF_TO_FAST_CONVERTERS["qwen3"]
    if "qwen3_5_moe" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen3_5_moe")
    if "qwen35moe" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen35moe")

    # The AutoTokenizer load path calls load_gguf_checkpoint directly (bypassing
    # the main loader), then feeds the parsed config to AutoConfig.for_model —
    # which needs the qwen35 -> qwen3_5_text model_type rewrite. Register the
    # fixups so the patched loader applies them on that path too.
    BYPASS_CONFIG_FIXUPS["qwen35"] = apply_qwen35_post_load_config_fixup
    BYPASS_CONFIG_FIXUPS["qwen35moe"] = apply_qwen35moe_post_load_config_fixup


# ─────────────────────────────── MoE experts ───────────────────────────────


class GGUFQwen35MoeExperts(torch.nn.Module):
    """Stacked, quantized Qwen3.5-MoE routed experts (arch `qwen35moe`).

    Replaces HF's `Qwen3_5MoeExperts`, which holds the experts as dense 3D
    `nn.Parameter` stacks. alloy's quant loader otherwise only swaps
    `Linear`/`Embedding` modules, so the GGUF's per-expert quantized weights have
    no home — this module is that home.

      - `gate_up` is the GGUF `ffn_gate_exps` + `ffn_up_exps` fused along the
        output rows (rows `[0:I]` gate, `[I:2I]` up), both GGUF-native Q4_K
        144-byte superblocks. The row concat is bit-exact — no requantization.
      - `down` is `ffn_down_exps` (Q6_K, raw blocks), stacked over experts.

    Storage only: the grouped per-expert matmul lives in the `gguf_moe_routed`
    op, driven by `GGUFQwen35MoeBlock`. `forward` raises.
    """

    gate_up_blocks: torch.Tensor
    down_qweight: torch.Tensor

    def __init__(
        self,
        *,
        gate_up_blocks: torch.Tensor,
        down_qweight: torch.Tensor,
        num_experts: int,
        hidden_size: int,
        moe_intermediate_size: int,
    ) -> None:
        super().__init__()
        if hidden_size % 256 != 0 or moe_intermediate_size % 256 != 0:
            raise ValueError(
                "Qwen3.5-MoE expert dims must be divisible by 256: "
                f"hidden={hidden_size} moe_intermediate={moe_intermediate_size}"
            )
        gate_up_out = 2 * moe_intermediate_size
        gate_up_row_bytes = (hidden_size // 256) * 144
        down_row_bytes = (moe_intermediate_size // 256) * 210
        expected = {
            "gate_up_blocks": (num_experts, gate_up_out, gate_up_row_bytes),
            "down_qweight": (num_experts, hidden_size, down_row_bytes),
        }
        actual = {
            "gate_up_blocks": tuple(gate_up_blocks.shape),
            "down_qweight": tuple(down_qweight.shape),
        }
        if actual != expected:
            raise ValueError(
                f"Qwen3.5-MoE expert weight shapes do not match config: "
                f"got {actual} expected {expected}"
            )
        if gate_up_blocks.dtype is not torch.uint8 or down_qweight.dtype is not torch.uint8:
            raise TypeError("Qwen3.5-MoE expert weights must be uint8 packed blocks")

        self.num_experts = int(num_experts)
        self.hidden_size = int(hidden_size)
        self.moe_intermediate_size = int(moe_intermediate_size)
        self.register_buffer("gate_up_blocks", gate_up_blocks.contiguous())
        self.register_buffer("down_qweight", down_qweight.contiguous())

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        raise RuntimeError(
            "GGUFQwen35MoeExperts is a weight holder; compute is driven by "
            "GGUFQwen35MoeBlock via the gguf_moe_routed op, not this forward."
        )


class GGUFQwen35MoeBlock(torch.nn.Module):
    """Quantized Qwen3.5-MoE sparse block (replaces `Qwen3_5MoeSparseMoeBlock`).

    Drives the routed path through the `gguf_moe_routed` custom op (router top-k →
    gathered gate_up SiLU → gathered down + combine), bypassing HF's per-expert
    Python loop + `aten.topk`. The shared expert + sigmoid gate ride the existing
    dense/quant handlers.

    forward: out = gguf_moe_routed(x, x@Wr.T, expert_q...) + sigmoid(Wg x)·shared(x)
    """

    def __init__(
        self,
        orig_block: torch.nn.Module,
        *,
        num_experts: int,
        top_k: int,
        moe_intermediate: int,
    ) -> None:
        super().__init__()
        self.gate = orig_block.gate                       # router (dense .weight)
        self.experts = orig_block.experts                 # GGUFQwen35MoeExperts
        self.shared_expert = orig_block.shared_expert
        self.shared_expert_gate = orig_block.shared_expert_gate
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.moe_intermediate = int(moe_intermediate)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        in_shape = hidden_states.shape
        x = hidden_states.reshape(-1, in_shape[-1])
        router_logits = torch.nn.functional.linear(x, self.gate.weight)
        routed = torch.ops.alloy.gguf_moe_routed(
            x,
            router_logits,
            self.experts.gate_up_blocks,
            self.experts.down_qweight,
            self.num_experts,
            self.top_k,
            self.moe_intermediate,
        )
        shared = self.shared_expert(x)
        shared = torch.sigmoid(self.shared_expert_gate(x)) * shared
        out = routed + shared
        return out.reshape(in_shape)


def install_qwen35moe_experts(
    model: torch.nn.Module,
    tensors: list[gguf.ReaderTensor],
    *,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    moe_intermediate_size: int,
) -> int:
    """Replace each layer's HF `Qwen3_5MoeExperts` with a quantized
    `GGUFQwen35MoeExperts`, fusing the GGUF `ffn_gate_exps`+`ffn_up_exps` (Q4_K)
    into one `gate_up` stack and stacking `ffn_down_exps` (Q6_K). Mutates `model`
    in place; must run before `load_state_dict` so the dense expert params are gone
    (not reported missing). Returns the number of layers installed."""
    by_name = {t.name: t for t in tensors}
    # Match only the main decoder layers (`blk.N.*`). The GGUF also carries an MTP
    # head whose layer is itself MoE (`mtp.layers.0.ffn_*_exps`); the HF causal LM
    # has no MTP module, so those tensors are unused by the base load and must not
    # be routed in (the spec-decode MTP path consumes them separately).
    gate_pattern = re.compile(r"blk\.(\d+)\.ffn_gate_exps\.weight")
    gate_names = sorted(
        (name for name in by_name if gate_pattern.fullmatch(name)),
        key=lambda s: int(gate_pattern.fullmatch(s).group(1)),
    )
    installed = 0
    for gate_name in gate_names:
        layer = gate_pattern.fullmatch(gate_name).group(1)
        up_name = f"blk.{layer}.ffn_up_exps.weight"
        down_name = f"blk.{layer}.ffn_down_exps.weight"
        if up_name not in by_name or down_name not in by_name:
            raise RuntimeError(
                f"Qwen3.5-MoE layer {layer} missing expert tensors: "
                f"have gate={gate_name!r} up={up_name in by_name} down={down_name in by_name}"
            )
        # gate/up: GGUF-native Q4_K (E, moe_intermediate, hidden_bytes). Row-concat
        # the raw superblocks along the output rows -> (E, 2*moe_intermediate, ...).
        # Bit-exact (no requantization) since gate/up rows are independent.
        gate_blk = torch.from_numpy(np.array(by_name[gate_name].data, copy=True))
        up_blk = torch.from_numpy(np.array(by_name[up_name].data, copy=True))
        gate_up_blocks = torch.cat([gate_blk, up_blk], dim=1)
        # down: Q6_K raw blocks (E, hidden, moe_intermediate_bytes), stacked as-is.
        down_qweight = torch.from_numpy(np.array(by_name[down_name].data, copy=True))
        experts = GGUFQwen35MoeExperts(
            gate_up_blocks=gate_up_blocks,
            down_qweight=down_qweight,
            num_experts=num_experts,
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
        )
        mlp = module_for_parameter(model, f"model.layers.{layer}.mlp.weight")
        if not isinstance(mlp._modules["experts"], torch.nn.Module):
            raise TypeError(f"No experts module at model.layers.{layer}.mlp.experts")
        mlp._modules["experts"] = experts
        # Wrap the whole sparse block so its forward drives the quantized routed
        # path (gguf_moe_routed) instead of HF's per-expert loop + aten.topk.
        layer_mod, mlp_name = module_parent(model, f"model.layers.{layer}.mlp")
        layer_mod._modules[mlp_name] = GGUFQwen35MoeBlock(
            mlp,
            num_experts=num_experts,
            top_k=top_k,
            moe_intermediate=moe_intermediate_size,
        )
        installed += 1
    return installed


# ─────────────────────────────── handlers ───────────────────────────────


@register("qwen35")
class Qwen35Handler(CausalLMHandler):
    """Qwen 3.5 (dense hybrid: gated attention + GatedDeltaNet)."""

    arch = ("qwen35",)
    kind = "chat"
    config_mapping = QWEN3_5_CONFIG_MAPPING

    def apply_transformers_patches(self) -> None:
        patch_transformers_ggml()

    def config_fixup(self, config_dict: dict, reader) -> None:
        apply_qwen35_post_load_config_fixup(config_dict)

    def build_vision(self, tensors, vision_meta, model, tokenizer):
        if not gguf_has_vision_qwen35(tensors):
            return None
        # qwen3.5's image-placeholder token (`<|image_pad|>`) — the slot whose
        # embedding each spliced vision feature replaces.
        image_pad_id = (
            tokenizer.convert_tokens_to_ids("<|image_pad|>") if tokenizer is not None else None
        )
        if isinstance(image_pad_id, int) and image_pad_id >= 0:
            return build_qwen35_vision_adapter(tensors, vision_meta, image_pad_id)
        return None


@register("qwen35moe")
class Qwen35MoeHandler(CausalLMHandler):
    """Qwen3.5-MoE (the dense hybrid backbone + a fine-grained MoE FFN)."""

    arch = ("qwen35moe",)
    kind = "chat"
    config_mapping = QWEN3_5_MOE_CONFIG_MAPPING

    def apply_transformers_patches(self) -> None:
        patch_transformers_ggml()

    def config_fixup(self, config_dict: dict, reader) -> None:
        apply_qwen35moe_post_load_config_fixup(config_dict)

    def post_load(self, model, tensors, config) -> None:
        install_qwen35moe_experts(
            model,
            tensors,
            num_experts=int(config.num_experts),
            top_k=int(config.num_experts_per_tok),
            hidden_size=int(config.hidden_size),
            moe_intermediate_size=int(config.moe_intermediate_size),
        )

    def allowed_missing_keys(self, model) -> set[str]:
        allowed: set[str] = set()
        for name, module in model.named_modules():
            if isinstance(module, GGUFQwen35MoeExperts):
                allowed.add(f"{name}.gate_up_blocks")
                allowed.add(f"{name}.down_qweight")
        return allowed
