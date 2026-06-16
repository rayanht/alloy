"""Causal-LM handler base.

`CausalLMHandler` is the base every chat arch subclasses: its `load` runs the
shared `gguf.load_causal_lm` with the handler itself as the `CausalLMHooks`, and
its hook methods are no-ops by default — an arch overrides only what it needs.
Archs with no arch-specific GGUF interpretation (llama / qwen2 / qwen3 / gemma3)
register a bare `CausalLMHandler` (see `models/llama.py`); the registry also
falls back to a bare one for an unrecognized arch loaded under `--force`.
"""

from __future__ import annotations

import torch

from alloy_server.gguf import LoadedGGUFCausalLM, ResolvedGGUF, load_gguf_causal_lm
from alloy_server.mlx import ResolvedMLX, load_mlx_causal_lm

# Models whose GGUF-embedded tokenizers don't include the chat-template special
# tokens used by the model's chat_template (e.g. DeepSeek-R1 distills). When the
# ref matches, the loader prefers the HF tokenizer for the mapped repo (must be in
# the local HF cache) and falls back to GGUF on miss.
HF_TOKENIZER_OVERRIDES: dict[str, str] = {
    "deepseek-r1:1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
}


class CausalLMHandler:
    """Base chat handler. Satisfies `gguf.CausalLMHooks` (all hooks no-op by
    default); `load` injects `self` as the hooks into the shared loader."""

    arch: tuple[str, ...] = ()
    kind: str = "chat"
    config_mapping: dict | None = None

    def apply_transformers_patches(self) -> None:
        return None

    def load(
        self,
        source: ResolvedGGUF | ResolvedMLX,
        *,
        dtype: torch.dtype | None = None,
        load_tokenizer: bool = True,
    ) -> LoadedGGUFCausalLM:
        if isinstance(source, ResolvedMLX):
            return load_mlx_causal_lm(
                source, self, dtype=dtype, load_tokenizer=load_tokenizer
            )
        hf_tokenizer_id = HF_TOKENIZER_OVERRIDES.get(source.ref) if load_tokenizer else None
        return load_gguf_causal_lm(
            source,
            self,
            dtype=dtype,
            load_tokenizer=load_tokenizer,
            hf_tokenizer_id=hf_tokenizer_id,
        )

    # --- CausalLMHooks (no-ops by default; archs override what they need) ---

    def config_fixup(self, config_dict: dict, reader) -> None:
        return None

    def fixup_tensor_map(self, tensor_key_mapping: dict, tensor_names) -> None:
        return None

    def post_load(self, model: torch.nn.Module, tensors, config) -> None:
        return None

    def chat_template(self) -> str | None:
        return None

    def build_vision(self, tensors, vision_meta: dict, model, tokenizer):
        return None

    def build_audio(self, tensors, audio_meta: dict, model):
        return None

    def allowed_missing_keys(self, model) -> set[str]:
        return set()
