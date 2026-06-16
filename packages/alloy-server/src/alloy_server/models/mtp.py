"""Qwen3.5 Multi-Token-Prediction (MTP) self-speculation head.

Qwen3-Next / Qwen3.5 ships a single DeepSeek-V3-style MTP layer in its GGUF
(`mtp.*` tensors) that the stock HF `Qwen3_5ForCausalLM` drops on load
(`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`). It predicts token i+2 from
the main model's hidden state at position i plus the embedding of token i+1, and
is trained to match the target's own distribution — so it's a near-free, high-
acceptance draft for speculative decoding (one extra transformer block reusing
the main model's hidden state + shared embedding/head, instead of a separate
draft model).

Forward (depth 1):

    e = embed(t_{i+1})                                   # shared input embedding
    x = fc( concat[ norm_e(e), norm_h(h_i) ] )           # 2*hidden -> hidden
    x = block(x)                                          # one full-attn layer
    logits = lm_head( norm(x) )                           # shared output head

Validated against the Qwen3.5 reference (ml-explore/mlx-lm#990) and offline against
the target's own greedy trace. Measured round acceptance is content-dependent — ~42%
on code, ~56% on predictable prose (qwen3.5:4b, greedy). MTP weight PRECISION does NOT
move it: fp16 and Q4_K weights give bit-identical acceptance (the draft output is an
argmax, robust to the Q4_K logit perturbation) while the fp16 block is ~10% slower per
round, so Q4_K is the right default — see load_quantized_mtp. The concat order is
[embedding, hidden]; the hidden input is the backbone's pre-final-norm residual
(recovered from the post-norm head input, see below).

`h_i` is the main model's pre-`output_norm` residual. We don't have it directly
(the compiled forward only surfaces the post-norm head input `H`), but RMSNorm
is scale-invariant per direction and `H = (raw/rms(raw)) * output_norm`, so
`norm_h(raw)` is recovered exactly from `norm_h(H / output_norm)` — see
`recover_pre_norm_hidden`.

The block, norms and rope are run through alloy (the same kernels the main model
uses), so execution is correct by construction; the eager HF path is unreliable
here because alloy's fused graph does not expose clean per-layer hidden states.
"""

from __future__ import annotations

import copy

import gguf
import numpy as np
import torch
from torch import nn
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5RMSNorm,
)

from alloy_server.gguf import (
    GGUFQ6_KLinear,
    replace_quantized_weight,
    tensor_quantization,
)


def recover_pre_norm_hidden(post_norm_hidden: torch.Tensor, output_norm_weight: torch.Tensor) -> torch.Tensor:
    """Recover the (direction of the) pre-`output_norm` residual from the post-norm
    head input. `H = (raw / rms(raw)) * w`, so `H / w = raw / rms(raw)`, which has
    the same direction as `raw` — and RMSNorm only depends on direction, so feeding
    this to the MTP's `pre_fc_norm_hidden` reproduces `pre_fc_norm_hidden(raw)`."""
    return post_norm_hidden / output_norm_weight


class Qwen35MTP(nn.Module):
    """Single-layer MTP head. `full_attn_layer_idx` selects a full-attention layer
    type so the block is built as full attention (matching the `mtp.layers.0.*`
    tensors)."""

    def __init__(self, config, full_attn_layer_idx: int):
        super().__init__()
        h = config.hidden_size
        eps = config.rms_norm_eps
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(h, eps=eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(h, eps=eps)
        self.fc = nn.Linear(2 * h, h, bias=False)
        # Build the block under a 1-layer all-full-attention config so it owns
        # cache slot 0 — its attention then has a clean StaticCache to attend
        # into (alloy's single-token decode attention can't run cacheless).
        block_cfg = copy.copy(config)
        block_cfg.num_hidden_layers = 1
        block_cfg.layer_types = ["full_attention"]
        self.cache_config = block_cfg
        self.layers = nn.ModuleList([Qwen3_5DecoderLayer(block_cfg, 0)])
        self.norm = Qwen3_5RMSNorm(h, eps=eps)
        self.lm_head = nn.Linear(h, config.vocab_size, bias=False)
        # Optional pruned top-K draft head (set by install_pruned_draft_head): a
        # gathered [K, hidden] subset of lm_head's rows. When present, the draft
        # projects only K vocab entries and remaps the argmax back to a real token id
        # via `shortlist` — lossless because the verify stays full-vocab. lm_head untouched.
        self.draft_head: nn.Module | None = None

    def forward(
        self,
        token_embeds: torch.Tensor,        # (B, S, hidden) embeddings of t_{i+1}
        hidden: torch.Tensor,              # (B, S, hidden) recovered pre-norm hidden h_i
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        past_key_values=None,
        cache_position: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Concat order is [embedding, hidden] (matches the Qwen3.5 reference:
        # `fc(concat([pre_fc_norm_embedding(e), pre_fc_norm_hidden(h)]))`).
        x = self.fc(
            torch.cat(
                [self.pre_fc_norm_embedding(token_embeds), self.pre_fc_norm_hidden(hidden)],
                dim=-1,
            )
        )
        out = self.layers[0](
            x,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )
        out = out[0] if isinstance(out, tuple) else out
        # Only the last position predicts the next draft token, so apply the head to
        # that position alone — applying it across a whole window is ~8x wasted
        # vocab-projection work. Use the pruned top-K head when installed.
        head = self.draft_head if self.draft_head is not None else self.lm_head
        return head(self.norm(out[:, -1:]))

    def bind_runtime(self, embed_tokens, rotary_emb, output_norm_weight: torch.Tensor) -> None:
        """Attach the backbone's (quantized) input embedding, rotary module, and
        `output_norm` weight so `draft_forward` runs entirely GPU-side: embedding
        gather, pre-norm-hidden recovery (post-norm / output_norm), and rope all
        happen *inside* the compiled draft plan — the steady-state loop only mutates
        token-id / hidden / position buffers, with zero python tensor setup."""
        self.embed = embed_tokens
        self.rotary = rotary_emb
        self.register_buffer("output_norm", output_norm_weight, persistent=False)

    def draft_forward(self, token_embeds, hidden_post_norm, cos, sin, position_ids, past_key_values, cache_position):
        """GPU-side draft step. The per-round inputs are the gathered `token_embeds`
        (B,M,H), the post-norm `hidden_post_norm` (B,M,H), and the precomputed rotary
        `cos`/`sin` (B,M,rotary_dim). The compiled function is ONLY the MTP block
        (`forward`) plus the elementwise pre-norm recovery — the embedding gather and
        the rotary derivation are kept OUT of the graph (both constant-fold under
        torch.compile when done in-plan). Returns the last position's argmax."""
        hidden = hidden_post_norm / self.output_norm  # recover dir of pre-norm residual
        out = self.forward(
            token_embeds, hidden, (cos, sin), None, position_ids,
            past_key_values=past_key_values, cache_position=cache_position,
        )
        idx = out[:, -1].argmax(-1)  # (B,) argmax over the head's output rows
        if self.draft_head is None:
            return idx  # full head: argmax index IS the vocab id
        # Pruned head: idx indexes the shortlist; remap to the real vocab id IN-PLAN
        # (the C++ spec loop reads this i64 straight from the plan output).
        return torch.nn.functional.embedding(idx, self.shortlist).reshape(idx.shape)


# GGUF tensor name -> Qwen35MTP submodule param name. `attn_q`/`attn_k` are NOT
# permuted (qwen35 uses NeoX rope; the base GGUF processor is a no-op for it).
MTP_KEY_MAP = {
    "mtp.fc.weight": "fc.weight",
    "mtp.norm.weight": "norm.weight",
    "mtp.pre_fc_norm_embedding.weight": "pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight": "pre_fc_norm_hidden.weight",
    "mtp.layers.0.attn_norm.weight": "layers.0.input_layernorm.weight",
    "mtp.layers.0.attn_q.weight": "layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.attn_k.weight": "layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.attn_v.weight": "layers.0.self_attn.v_proj.weight",
    "mtp.layers.0.attn_output.weight": "layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.attn_q_norm.weight": "layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.attn_k_norm.weight": "layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.post_attention_norm.weight": "layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.ffn_gate.weight": "layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.ffn_up.weight": "layers.0.mlp.up_proj.weight",
    "mtp.layers.0.ffn_down.weight": "layers.0.mlp.down_proj.weight",
}


class MTPDraftStep(torch.nn.Module):
    """Compile target for the GPU-side draft step (returns the next draft token id)."""

    def __init__(self, mtp: Qwen35MTP) -> None:
        super().__init__()
        self.mtp = mtp

    def forward(self, token_embeds, hidden_post_norm, cos, sin, position_ids, past_key_values, cache_position):
        return self.mtp.draft_forward(
            token_embeds, hidden_post_norm, cos, sin, position_ids, past_key_values, cache_position
        )


def load_mtp_state_dict(gguf_tensors: dict[str, torch.Tensor], lm_head_weight: torch.Tensor) -> dict[str, torch.Tensor]:
    """Build a `Qwen35MTP` state dict from dequantized GGUF `mtp.*` tensors plus the
    shared output head (`output.weight`)."""
    sd = {hf: gguf_tensors[gg] for gg, hf in MTP_KEY_MAP.items() if gg in gguf_tensors}
    sd["lm_head.weight"] = lm_head_weight
    return sd


def gguf_special_ids(reader, vocab: int) -> set[int]:
    """In-range control/user-defined token ids from GGUF `tokenizer.ggml.token_type`
    (3=CONTROL, 4=USER_DEFINED) — the special tokens the draft must stay able to emit.
    Qwen3.5 places these at the TOP of the vocab, so a plain top-K-by-id shortlist
    would drop every one of them (<|im_start|>, <|im_end|>, tool tokens, …)."""
    field = reader.fields.get("tokenizer.ggml.token_type")
    if field is None or not field.data:
        return set()
    return {i for i, idx in enumerate(field.data)
            if i < vocab and int(field.parts[idx][0]) in (3, 4)}


def build_draft_shortlist(special_ids, vocab: int, k: int) -> torch.Tensor:
    """The K vocab ids the pruned draft head projects: ALL in-range special ids
    force-included, padded to K with the lowest-id remaining tokens. Sorted int64."""
    special = sorted({int(i) for i in special_ids if 0 <= int(i) < vocab})
    if len(special) >= k:
        return torch.tensor(special, dtype=torch.int64)
    sset = set(special)
    fill, i = [], 0
    while len(fill) < k - len(special) and i < vocab:
        if i not in sset:
            fill.append(i)
        i += 1
    return torch.tensor(sorted(special + fill), dtype=torch.int64)


def install_pruned_draft_head(mtp: Qwen35MTP, model, shortlist: torch.Tensor, reader=None) -> None:
    """Give the MTP draft a gathered top-K head (rows `shortlist` of the shared lm_head)
    plus the [K,1] int64 `shortlist` remap buffer. lm_head itself is untouched — the
    verify keeps the full vocab, so this is lossless. Q6_K heads gather packed bytes
    bit-exactly; other head types dequantize the K rows to fp16 (needs `reader`)."""
    head = model.lm_head
    sl = shortlist.to(torch.int64)
    k = int(sl.numel())
    h = int(head.in_features)
    if isinstance(head, GGUFQ6_KLinear):
        pruned: nn.Module = GGUFQ6_KLinear(
            qweight=head.qweight[sl].contiguous().clone(),  # (K, row_bytes) bit-exact rows
            in_features=h, out_features=k, bias=None,
        )
    elif isinstance(head, nn.Linear):
        pruned = nn.Linear(h, k, bias=False)
        pruned.weight = nn.Parameter(head.weight.detach()[sl].contiguous().clone(), requires_grad=False)
    elif reader is not None:  # other quant (e.g. Q8_0 head on 0.8b): dequant K rows to fp16
        ot = next(x for x in reader.tensors if x.name == "output.weight")
        full = torch.from_numpy(np.array(gguf.dequantize(ot.data, ot.tensor_type))).reshape(-1, h)
        pruned = nn.Linear(h, k, bias=False)
        pruned.weight = nn.Parameter(full[sl].to(torch.float16).contiguous(), requires_grad=False)
    else:
        raise TypeError(f"pruned draft head: unsupported lm_head {type(head).__name__}")
    mtp.draft_head = pruned
    mtp.register_buffer("shortlist", sl.reshape(k, 1), persistent=False)


def load_quantized_mtp(model, blob_path, quantize: bool = True, draft_topk: int | None = None) -> Qwen35MTP:
    """Attach an MTP head to `model` and load the `mtp.*` GGUF tensors.

    `quantize=True` (default) routes the fc/attn/FFN through the SAME alloy
    quantized-weight path the backbone uses. `quantize=False` dequantizes them to
    fp16 instead (precision does NOT affect draft acceptance, so quantize=True is
    strictly better; the fp16 path is a debug knob). The lm_head is always shared
    with the (already-quantized) backbone head.

    `draft_topk=K` prunes the draft's vocab projection to a K-token shortlist (all
    special tokens + top-K-by-id), remapping argmax->vocab in-plan. Lossless and
    cuts the draft lm_head — the draft pass's dominant cost — to ~K/vocab."""
    cfg = model.config
    full_idx = next(i for i, lt in enumerate(cfg.layer_types) if "full" in lt)
    mtp = Qwen35MTP(cfg, full_idx)
    model.mtp = mtp
    reader = gguf.GGUFReader(str(blob_path))
    for t in reader.tensors:
        if not t.name.startswith("mtp"):
            continue
        rel = MTP_KEY_MAP.get(t.name)
        if rel is None or rel == "lm_head.weight":
            continue  # head is shared with the backbone, set below
        q = tensor_quantization(t)
        if quantize and q is not None:
            replace_quantized_weight(model, "mtp." + rel, np.asarray(t.data), q)
        else:
            w = torch.from_numpy(np.array(gguf.dequantize(t.data, t.tensor_type)))
            p = model.get_parameter("mtp." + rel)
            p.data = w.to(torch.float16)
    mtp.lm_head = model.lm_head  # shared, already-quantized output head
    if draft_topk is not None:
        vocab = int(cfg.vocab_size)
        shortlist = build_draft_shortlist(gguf_special_ids(reader, vocab), vocab, draft_topk)
        install_pruned_draft_head(mtp, model, shortlist, reader=reader)
    return mtp
