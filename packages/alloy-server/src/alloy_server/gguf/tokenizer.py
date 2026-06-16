"""GGUF tokenizer loading: HF-override fallback, GGUF control-token registration,
and the Unigram→SPM bigram-merge encode patch.

transformers' GGUF importer materialises the vocab as a tokenizers Unigram model,
but llama.cpp tokenises gemma/llama with the SPM bigram-merge algorithm. When the
loaded model is Unigram (the broken path), `load_gguf_tokenizer` swaps encode for
a faithful Python port of llama.cpp's SPM (see alloy_server.spm).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import cast

import torch
from tokenizers import AddedToken
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from alloy_server.gguf.transformers_compat import walk_gguf_metadata
from alloy_server.spm import SPM_SPACE_MARKER, encode_spm


def build_tokenizer(blob_path: Path, hf_tokenizer_id: str | None = None) -> PreTrainedTokenizerBase:
    if hf_tokenizer_id is not None:
        try:
            return cast(
                PreTrainedTokenizerBase,
                AutoTokenizer.from_pretrained(hf_tokenizer_id, local_files_only=True),
            )
        except Exception:
            pass
    return load_gguf_tokenizer(blob_path)


def load_gguf_tokenizer(blob_path: Path) -> PreTrainedTokenizerBase:
    """Load a GGUF-embedded tokenizer.

    transformers' GGUFGemmaConverter materialises the GGUF vocab as a
    tokenizers Unigram model, but llama.cpp tokenises gemma3 with the SPM
    bigram-merge algorithm (BPE-style priority queue keyed by GGUF score),
    not Unigram Viterbi. The two algorithms produce different segmentations
    for the same input — Unigram falls back to per-character splits because
    GGUF scores are merge priorities (``-rank``), not log-probabilities.

    When the loaded model is Unigram (the broken path), wrap the tokenizer
    so encoding goes through a faithful Python port of llama.cpp's SPM
    algorithm. Decoding, chat-template rendering, and special-token attrs
    stay on the HF tokenizer (they don't depend on the encode model).
    """
    tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(
            str(blob_path.parent),
            gguf_file=blob_path.name,
            local_files_only=True,
        ),
    )
    # Register GGUF type=3 (CONTROL) and type=4 (USER_DEFINED) tokens as
    # single tokens. transformers' GGUFTokenizer importer only adds a small
    # set of these (e.g. <|im_start|>, <|im_end|>, <|endoftext|>) and drops
    # the rest — including chat-format control tokens like <think>/</think>
    # that qwen3.5's chat template emits. Without these registered as added
    # tokens, BPE splits them into sub-pieces (<th, ink, >) and the model
    # sees a malformed prompt.
    control_token_ids = extract_gguf_control_tokens(blob_path)
    existing_ids = set(tokenizer.added_tokens_decoder.keys())
    missing = [
        (text, tid) for text, tid in control_token_ids.items()
        if tid not in existing_ids
    ]
    if missing:
        to_add = [
            AddedToken(text, special=True, normalized=False, single_word=False)
            for text, _ in sorted(missing, key=lambda kv: kv[1])
        ]
        tokenizer.add_tokens(to_add, special_tokens=True)
    state = json.loads(tokenizer.backend_tokenizer.to_str())
    model = state.get("model", {})
    if model.get("type") != "Unigram":
        return tokenizer
    vocab = model.get("vocab")
    if not vocab:
        return tokenizer
    sample = [s for _, s in vocab[:32] if isinstance(s, (int, float))]
    if len(sample) < len(vocab[:32]):
        return tokenizer
    if all(s <= 0 for s in sample):
        # gemma3 / classic llama.cpp SPM: GGUF scores are -rank (log-prob-like),
        # so the SPM max-heap merges the highest (least-negative) score first.
        pass
    elif all(s >= 0 for s in sample) and sample == sorted(sample):
        # gemma4 (tokenizer.ggml.pre == "gemma4"): GGUF stores scores as the
        # ASCENDING token rank (score[i] == i), not -rank. Negate so the SPM
        # max-heap still merges the lowest-rank (most common) token first —
        # otherwise the guard rejects it and the tokenizer falls back to the
        # broken per-character Unigram split.
        vocab = [[t, -s] for t, s in vocab]
    else:
        return tokenizer
    install_spm_encoder(tokenizer, vocab, control_token_ids)
    return tokenizer


def extract_gguf_control_tokens(blob_path: Path) -> dict[str, int]:
    """Return ``{token_text: id}`` for GGUF tokens with token_type=3 (CONTROL)
    or token_type=4 (USER_DEFINED).

    Both types must be matched as single tokens against the input *before* BPE
    runs. Without this, BPE happily splits ``<think>``/``</think>``/``<tool_call>``
    etc. into multi-piece sequences the model never saw, producing wrong prompts
    that derail the model's chat-template alignment.
    """
    # Fast single-pass metadata walk instead of gguf.GGUFReader, which
    # re-parses the entire 248k-token vocab per element (~10s). We only need
    # the tokens + token_type arrays.
    fields, _ = walk_gguf_metadata(str(blob_path))
    tokens = fields.get("tokenizer.ggml.tokens")
    types = fields.get("tokenizer.ggml.token_type")
    if tokens is None or types is None:
        return {}
    return {tokens[i]: i for i, t in enumerate(types) if int(t) in (3, 4)}


def install_spm_encoder(
    tokenizer: PreTrainedTokenizerBase,
    vocab: list[list],
    control_tokens: dict[str, int],
) -> None:
    """Replace the tokenizer's encode methods with the shared SPM bigram-merge port.

    Algorithm (see ``src/llama-vocab.cpp::llm_tokenizer_spm_session::tokenize``):
      1. Split text into per-character symbols, linked in a doubly-linked list.
      2. Push every adjacent (left, right) pair whose concatenation is a vocab
         token onto a max-heap keyed by GGUF score (highest = most common).
      3. Pop the top bigram, merge into ``left`` and unlink ``right``,
         then push the two new neighbour pairs.
      4. Repeat until the queue is empty.
      5. Walk the surviving symbols left-to-right, emitting either the
         matched vocab id or, on miss, byte-fallback ``<0xNN>`` token ids.
    """
    token_to_id: dict[str, int] = {tok: i for i, (tok, _) in enumerate(vocab)}
    scores: list[float] = [score for _, score in vocab]
    if control_tokens:
        ctrl_pattern = re.compile(
            "|".join(re.escape(t) for t in sorted(control_tokens, key=len, reverse=True))
        )
    else:
        ctrl_pattern = None

    def encode_piece(text: str) -> list[int]:
        normalised = text.replace(" ", SPM_SPACE_MARKER)
        return encode_spm(normalised, token_to_id=token_to_id, scores=scores)

    def encode_with_specials(text: str, add_special_tokens: bool) -> list[int]:
        ids: list[int] = []
        if ctrl_pattern is None:
            ids.extend(encode_piece(text))
        else:
            pos = 0
            for m in ctrl_pattern.finditer(text):
                if m.start() > pos:
                    ids.extend(encode_piece(text[pos:m.start()]))
                ids.append(control_tokens[m.group(0)])
                pos = m.end()
            if pos < len(text):
                ids.extend(encode_piece(text[pos:]))
        if add_special_tokens and tokenizer.bos_token_id is not None:
            if not ids or ids[0] != tokenizer.bos_token_id:
                ids = [tokenizer.bos_token_id, *ids]
        return ids

    # `tokenizer(text)` resolves __call__ via the class, so instance-level
    # patching is invisible to it. Promote the instance to a dynamic
    # subclass that owns the new __call__ + encode methods.
    base_cls = type(tokenizer)

    class GGUFSPMTokenizer(base_cls):
        def encode(self, text, add_special_tokens: bool = True, **kwargs):
            if not isinstance(text, str):
                return base_cls.encode(self, text, add_special_tokens=add_special_tokens, **kwargs)
            return encode_with_specials(text, add_special_tokens)

        def __call__(self, text, return_tensors=None, add_special_tokens: bool = True, **kwargs):
            if not isinstance(text, str):
                return base_cls.__call__(
                    self, text,
                    return_tensors=return_tensors,
                    add_special_tokens=add_special_tokens,
                    **kwargs,
                )
            ids = encode_with_specials(text, add_special_tokens)
            if return_tensors == "pt":
                return {
                    "input_ids": torch.tensor([ids], dtype=torch.long),
                    "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
                }
            return {"input_ids": [ids], "attention_mask": [[1] * len(ids)]}

    tokenizer.__class__ = GGUFSPMTokenizer
