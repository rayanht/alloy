"""Default-route model handlers: llama / qwen2 / qwen3 / gemma3 / lfm2.

These archs need no arch-specific GGUF interpretation — transformers' GGUF bridge
handles their config + tensors, and the generic loader path (`gguf.load_causal_lm`
with no-op hooks) loads them as-is. So each handler is a thin `CausalLMHandler`
registering its GGUF `general.architecture` string.

The one exception is lfm2's tokenizer converter (transformers has no
`GGUF_TO_FAST_CONVERTERS['lfm2']`): its byte-level BPE matches llama3's, so the
handler points lfm2 at the llama converter. Their attention forwards run through
the alloy cache ops via `models.attention.install_multi_token_attention`,
installed by the generator.
"""

from __future__ import annotations

from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

from alloy_server.models.base import CausalLMHandler
from alloy_server.models.registry import register


@register("llama")
class LlamaHandler(CausalLMHandler):
    arch = ("llama",)


@register("qwen2")
class Qwen2Handler(CausalLMHandler):
    arch = ("qwen2",)


@register("qwen3")
class Qwen3Handler(CausalLMHandler):
    arch = ("qwen3",)


@register("gemma3")
class Gemma3Handler(CausalLMHandler):
    arch = ("gemma3",)


@register("lfm2")
class Lfm2Handler(CausalLMHandler):
    """LiquidAI LFM2. transformers bridges lfm2 end-to-end EXCEPT the tokenizer
    converter: its GGUF tokenizer is `gpt2`-model + `lfm2` pre-type, which
    llama.cpp maps to the llama3 byte-level BPE — so the llama converter is the
    correct one."""

    arch = ("lfm2",)

    def apply_transformers_patches(self) -> None:
        if "lfm2" not in GGUF_TO_FAST_CONVERTERS:
            GGUF_TO_FAST_CONVERTERS["lfm2"] = GGUF_TO_FAST_CONVERTERS["llama"]
