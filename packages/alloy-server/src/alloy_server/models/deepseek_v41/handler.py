"""The `deepseek41` model handler: resolves the split GGUF, builds the streaming
engine and the tokenizer. Not an HF causal LM — the payload is `LoadedDeepseekV41`."""

from __future__ import annotations

from dataclasses import dataclass

from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from alloy_server.gguf import ResolvedGGUF
from alloy_server.gguf.split import SplitGGUF
from alloy_server.gguf.tokenizer import load_gguf_tokenizer
from alloy_server.models.deepseek_v41.engine import DeepseekV41Engine
from alloy_server.models.registry import register

HF_TOKENIZER_ID = "deepseek-ai/DeepSeek-V4.1-Flash"


@dataclass
class LoadedDeepseekV41:
    engine: DeepseekV41Engine
    tokenizer: PreTrainedTokenizerBase

    @property
    def eos_token_ids(self) -> tuple[int, ...]:
        ids = {int(self.tokenizer.eos_token_id)} if self.tokenizer.eos_token_id is not None else set()
        ids.add(self.engine.cfg.eos_token_id)
        return tuple(sorted(ids))


def build_deepseek41_tokenizer(split: SplitGGUF) -> PreTrainedTokenizerBase:
    """The HF tokenizer when its files are in the local cache, else the GGUF-embedded
    one; the chat template comes from the GGUF metadata either way."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(HF_TOKENIZER_ID, local_files_only=True)
    except OSError:
        tokenizer = load_gguf_tokenizer(split.paths[0])
    template = split.kv.get("tokenizer.chat_template")
    if template and not tokenizer.chat_template:
        tokenizer.chat_template = template
    return tokenizer


@register("deepseek41")
class DeepseekV41Handler:
    arch = ("deepseek41",)
    kind = "chat"

    def apply_transformers_patches(self) -> None:
        return None

    def load(self, source: ResolvedGGUF, **kwargs: object) -> LoadedDeepseekV41:
        split = SplitGGUF(source.path)
        engine = DeepseekV41Engine(split, **{k: v for k, v in kwargs.items() if k in ENGINE_OPTIONS})
        return LoadedDeepseekV41(engine=engine, tokenizer=build_deepseek41_tokenizer(split))


ENGINE_OPTIONS = {"max_seq_len", "chunk_size", "arena_bytes", "rounding", "io_threads", "bypass_page_cache", "n_layers"}
