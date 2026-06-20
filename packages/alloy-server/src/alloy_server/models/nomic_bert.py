"""Load nomic-bert (nomic-embed-text) GGUFs via HuggingFace's built-in
`NomicBertModel` class.

We don't write the architecture ourselves — `transformers >= 5.x` ships
`NomicBertModel` as a stdlib model (no `trust_remote_code` needed).
This module is just a GGUF→HF state_dict adapter:

  1. Read the GGUF blob from `~/.ollama/models/`.
  2. Build a `NomicBertConfig` from the GGUF metadata.
  3. Instantiate `NomicBertModel` with empty weights.
  4. Map GGUF tensor names → HF state_dict keys, transposing linear
     weights (GGUF stores `(in, out)`, torch wants `(out, in)`) and
     splitting the fused `attn_qkv` into separate `q/k/v_proj`.
  5. Wrap forward with mean-pool + L2 normalize → `EmbeddingModel`.

Tokenizer: rebuild a BERT WordPiece tokenizer from the GGUF token list
via `tokenizers.WordPiece`.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import gguf
import numpy as np
import torch
import transformers
import unicodedata
from transformers.initialization import no_init_weights

from alloy_torch.backend import _execute_plan, capture_plan
from alloy_server.embedding import EmbeddingModel
from alloy_server.gguf import ResolvedGGUF, resolve_gguf
from alloy_server.models.registry import register

SPACE_MARKER = "▁"  # SPM ▁ — word-start marker used by nomic-bert's GGUF vocab

# Bucketed (batch, seq) shapes the loader pre-compiles at startup. Every
# alloy.compile is shape-specific (`dynamic=False`), so padding `embed()` up to the
# nearest bucket means real requests hit a hot kernel. batch=1 covers the full seq
# range to the 2048 ceiling (single-shot retrieval / RAG-query); batch=8 covers
# short-to-medium seq (bulk indexing) — larger (batch, seq) combos OOM under fp16
# attention scores and drop to the compile-on-first-sight slow path.
EMBED_PRECOMPILE_SHAPES: tuple[tuple[int, int], ...] = (
    (1, 32),
    (1, 128),
    (1, 512),
    (1, 2048),
    (8, 32),
    (8, 128),
    (8, 512),
)

EMBED_SEQ_BUCKETS: tuple[int, ...] = tuple(
    sorted({seq for _, seq in EMBED_PRECOMPILE_SHAPES})
)


@dataclass(frozen=True, slots=True)
class NomicTokenizer:
    """Greedy longest-match WordPiece tokenizer (BERT algorithm) with the
    nomic-bert GGUF's SPM-style vocab (▁-prefixed word starts).

    The GGUF declares `tokenizer.ggml.model: 'bert'` and uses uniform
    `-1000.0` scores, confirming this is WordPiece — not Unigram-Viterbi
    or BPE bigram-merge. Algorithm matches `llama-vocab.cpp::llm_tokenizer_wpm`:

      1. BERT-uncased normalize (NFD strip-accents, lowercase, clean).
      2. Split on whitespace.
      3. For each word, greedily match the longest vocab prefix —
         prepending ▁ on the first piece. Continuation pieces have no
         marker. On no match, emit [UNK] for the whole word.
      4. Wrap with [CLS] ... [SEP].
    """

    token_to_id: dict[str, int]
    cls_id: int
    sep_id: int
    pad_id: int
    unk_id: int
    max_length: int

    def encode(self, text: str) -> list[int]:
        words = bert_normalize(text).split()
        piece_ids: list[int] = []
        for word in words:
            piece_ids.extend(self.encode_word(word))
        usable = self.max_length - 2
        if len(piece_ids) > usable:
            piece_ids = piece_ids[:usable]
        return [self.cls_id, *piece_ids, self.sep_id]

    def encode_batch(self, texts: list[str]) -> tuple[list[list[int]], list[list[int]]]:
        per_text = [self.encode(t) for t in texts]
        natural_max = max(len(ids) for ids in per_text)
        # Round up to the smallest pre-compiled bucket so we hit a
        # warm alloy kernel. The longest bucket caps at the model's
        # context length anyway, so we don't pad beyond what the
        # tokenizer would already truncate to.
        padded_len = bucket_seq_len(natural_max, self.max_length)
        input_ids = [ids + [self.pad_id] * (padded_len - len(ids)) for ids in per_text]
        attention_mask = [
            [1] * len(ids) + [0] * (padded_len - len(ids)) for ids in per_text
        ]
        return input_ids, attention_mask

    def encode_word(self, word: str) -> list[int]:
        if not word:
            return []
        out: list[int] = []
        remaining = word
        first = True
        while remaining:
            prefix = SPACE_MARKER + remaining if first else remaining
            matched_id: int | None = None
            matched_consume = 0
            # Longest-prefix match. Stop at length 2 on the first piece
            # (must include at least one char beyond ▁) and length 1 on
            # continuations.
            min_len = 2 if first else 1
            for end in range(len(prefix), min_len - 1, -1):
                piece = prefix[:end]
                tid = self.token_to_id.get(piece)
                if tid is not None:
                    matched_id = tid
                    matched_consume = end - 1 if first else end
                    break
            if matched_id is None:
                return [self.unk_id]
            out.append(matched_id)
            remaining = remaining[matched_consume:]
            first = False
        return out


class LastHiddenStateOnly(torch.nn.Module):
    """Wraps NomicBertModel to a single-tensor output (last_hidden_state). After
    AOT autograd the flat output tuple has multiple entries; a single output keeps
    the compiled-plan output index trivial (`_execute_plan` returns it directly)."""

    def __init__(self, base: torch.nn.Module) -> None:
        super().__init__()
        self.base = base

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.base(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


@dataclass(slots=True)
class PinnedBucket:
    """Per-(batch, seq) pinned plan + reusable input storage. Request time copies
    real inputs into the pinned tensors and calls `_execute_plan(plan, args)`
    directly, never crossing the torch.compile wrapper, dynamo guards, or AOT
    autograd."""

    batch: int
    seq: int
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    plan: object  # CompiledPlan; opaque to this module
    args: tuple


@dataclass(frozen=True, slots=True)
class NomicMeta:
    hidden_size: int
    num_layers: int
    num_heads: int
    ffn_size: int
    vocab_size: int
    context_length: int
    layer_norm_eps: float
    rope_freq_base: float


@dataclass(frozen=True, slots=True)
class NomicForward:
    """The compiled (or eager) nomic-bert forward plus its tokenizer/meta.

    Shared between production (`load_ollama_gguf_embedder`, which pins request
    buckets and adds mean-pool/normalize) and the capture path (`alloy profile` /
    `alloy inspect`, which profiles a chosen shape). Both drive this identical
    compiled `forward`, so captured plans / MSL match what the server dispatches.

    `forward` is the `torch.compile(..., backend='alloy', dynamic=False)`
    wrapper (or the raw `wrapped` module when `compile_backend is None`).
    """

    forward: object
    wrapped: torch.nn.Module
    tokenizer: NomicTokenizer
    meta: NomicMeta
    pad_id: int


def build_nomic_forward(
    model_ref: ResolvedGGUF | str,
    *,
    root: Path | None = None,
    dtype: torch.dtype = torch.float16,
    compile_backend: str | None = "alloy",
) -> NomicForward:
    """Load a nomic-bert GGUF and return its compiled forward + tokenizer +
    meta. Stops short of pinning request buckets or building the pooling
    wrapper — callers add what they need on top (production pins + pools,
    capture profiles one shape). See `load_ollama_gguf_embedder` for the
    rationale behind CPU-resident weights and `attn_implementation='eager'`.
    """
    source = model_ref if isinstance(model_ref, ResolvedGGUF) else resolve_gguf(model_ref, root=root)
    reader = gguf.GGUFReader(str(source.path))
    meta = read_nomic_meta(reader)

    config = transformers.NomicBertConfig(
        vocab_size=meta.vocab_size,
        hidden_size=meta.hidden_size,
        num_hidden_layers=meta.num_layers,
        num_attention_heads=meta.num_heads,
        intermediate_size=meta.ffn_size,
        max_position_embeddings=meta.context_length,
        layer_norm_eps=meta.layer_norm_eps,
        rope_parameters={"rope_type": "default", "rope_theta": meta.rope_freq_base},
        type_vocab_size=2,
        attn_implementation="eager",
    )

    with no_init_weights():
        base = transformers.NomicBertModel(config)
    state_dict = gguf_to_hf_state_dict(reader, meta)
    base.load_state_dict(state_dict, strict=True)
    base.to(dtype=dtype, device="cpu")
    base.eval()

    wrapped = LastHiddenStateOnly(base).eval()
    tokenizer = build_nomic_tokenizer(reader, max_length=meta.context_length)
    forward = (
        torch.compile(wrapped, backend=compile_backend, dynamic=False)
        if compile_backend is not None
        else wrapped
    )
    return NomicForward(
        forward=forward,
        wrapped=wrapped,
        tokenizer=tokenizer,
        meta=meta,
        pad_id=tokenizer.pad_id,
    )


def load_ollama_gguf_embedder(
    model_ref: ResolvedGGUF | str,
    *,
    root: Path | None = None,
    name: str | None = None,
    dtype: torch.dtype = torch.float16,
    max_batch: int = 64,
    compile_backend: str | None = "alloy",
) -> EmbeddingModel:
    """Load a nomic-bert GGUF (local path / HF repo / Ollama name) into an
    `EmbeddingModel` runnable via `/api/embed` / `/v1/embeddings`.

    Compute path:
      - Weights are placed on CPU (host-addressable). Alloy's runtime
        memcpy's them into shared-memory Metal buffers on first dispatch
        and caches the buffers — subsequent calls are zero-copy.
      - The forward is wrapped in `torch.compile(model, backend='alloy')`
        with `dynamic=False`; one compile per unique input shape.
      - `attn_implementation='eager'` is forced; the MPS-SDPA fallback
        op (`aten._scaled_dot_product_attention_math_for_mps`) is not in
        alloy's FX handler list, so we route through explicit Q@K + softmax
        + @V which alloy lowers cleanly.

    Pass `compile_backend=None` to skip torch.compile and run the raw HF
    forward on CPU (debug / eager comparison).

    Why not `device='mps'`: torch tensors on MPS have device-private
    `data_ptr()` values in the ~46 GB virtual range that aren't host-
    addressable; alloy's `get_buffer` falls back to a host `memcpy` for
    unrecognized pointers and faults with SIGBUS. CPU-resident weights
    sidestep that entirely while still reaping the alloy-compiled GEMM
    throughput.
    """
    nf = build_nomic_forward(
        model_ref, root=root, dtype=dtype, compile_backend=compile_backend
    )
    meta = nf.meta
    wrapped = nf.wrapped
    tokenizer = nf.tokenizer
    pad_id = nf.pad_id

    # For every request-time shape, pre-allocate stable input storage + bind a
    # CompiledPlan, then dispatch via `_execute_plan` — no torch.compile, no dynamo,
    # no FX walk in the request thread.
    pinned_buckets: dict[tuple[int, int], PinnedBucket] = {}
    embed_lock = threading.Lock()

    if compile_backend is not None:
        forward = nf.forward
        eager_compile_embed_with_pinning(
            forward, meta, pad_id, pinned_buckets,
        )
        # Fallback for shapes the caller hits outside pre-compiled buckets
        # (rare in practice — `NomicTokenizer.encode_batch` pads to a
        # bucket, but a request might still come in with a batch size we
        # didn't pre-cover). Falls through to the lazy/AOT path.
        fallback_forward = forward
    else:
        fallback_forward = wrapped

    def embed(texts: list[str]) -> list[list[float]]:
        if len(texts) > max_batch:
            raise ValueError(f"batch size {len(texts)} exceeds max {max_batch}")
        if not texts:
            return []
        ids_batch, mask_batch = tokenizer.encode_batch(texts)
        real_batch = len(texts)
        seq = len(ids_batch[0])
        bucket_batch = bucket_batch_size(real_batch)
        ids_tensor = torch.tensor(ids_batch, dtype=torch.long, device="cpu")
        mask_tensor = torch.tensor(mask_batch, dtype=torch.long, device="cpu")

        with embed_lock:
            bucket = pinned_buckets.get((bucket_batch, seq))
            if bucket is not None:
                # Hot path: pad to bucket_batch with the pad token /
                # zero mask, copy into pinned storage, dispatch the plan.
                bucket.input_ids[:real_batch].copy_(ids_tensor)
                if bucket_batch > real_batch:
                    bucket.input_ids[real_batch:].fill_(pad_id)
                bucket.attention_mask[:real_batch].copy_(mask_tensor)
                if bucket_batch > real_batch:
                    bucket.attention_mask[real_batch:].fill_(0)
                last_hidden = _execute_plan(bucket.plan, bucket.args)
            else:
                # Cold-shape fallback: still correct, but pays the alloy
                # compile cost on first sight of this exact shape.
                with torch.no_grad():
                    last_hidden = fallback_forward(
                        input_ids=ids_tensor, attention_mask=mask_tensor,
                    )

        # `last_hidden` is (bucket_batch, seq, hidden) — slice to the real batch
        # before pooling (padded rows have mask 0; the slice saves a few FLOPs).
        if last_hidden.shape[0] != real_batch:
            last_hidden = last_hidden[:real_batch]
            mask_for_pool = mask_tensor
        else:
            mask_for_pool = mask_tensor
        pooled = mean_pool(last_hidden, mask_for_pool)
        normalized = torch.nn.functional.normalize(pooled.to(torch.float32), p=2.0, dim=-1)
        return normalized.cpu().tolist()

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text))

    ref_name = model_ref.ref if isinstance(model_ref, ResolvedGGUF) else model_ref
    return EmbeddingModel(
        name=name or ref_name,
        embed=embed,
        dimensions=meta.hidden_size,
        count_tokens=count_tokens,
        max_batch=max_batch,
    )


def read_nomic_meta(reader: gguf.GGUFReader) -> NomicMeta:
    arch = required_string(reader, "general.architecture")
    if arch != "nomic-bert":
        raise ValueError(f"expected nomic-bert architecture; got {arch!r}")
    return NomicMeta(
        hidden_size=required_int(reader, "nomic-bert.embedding_length"),
        num_layers=required_int(reader, "nomic-bert.block_count"),
        num_heads=required_int(reader, "nomic-bert.attention.head_count"),
        ffn_size=required_int(reader, "nomic-bert.feed_forward_length"),
        vocab_size=len(reader.fields["tokenizer.ggml.tokens"].data),
        context_length=required_int(reader, "nomic-bert.context_length"),
        layer_norm_eps=required_float(reader, "nomic-bert.attention.layer_norm_epsilon"),
        rope_freq_base=required_float(reader, "nomic-bert.rope.freq_base"),
    )


def gguf_to_hf_state_dict(reader: gguf.GGUFReader, meta: NomicMeta) -> dict[str, torch.Tensor]:
    """Build a state_dict matching HF NomicBertModel's keys from the GGUF
    tensor layout. GGUF stores tensors in ggml's column-major view;
    `to_numpy` reverses the shape so we get row-major numpy arrays that
    already match torch's `(out, in)` weight convention — no further
    transposes needed. The fused `attn_qkv` projection is split into
    separate q/k/v rows for HF's NomicBertAttention.
    """
    tensors = {t.name: t for t in reader.tensors}

    def fetch(name: str) -> torch.Tensor:
        if name not in tensors:
            raise KeyError(f"missing tensor: {name}")
        return torch.from_numpy(to_numpy(tensors[name]).copy())

    state: dict[str, torch.Tensor] = {}

    state["embeddings.word_embeddings.weight"] = fetch("token_embd.weight")
    state["embeddings.token_type_embeddings.weight"] = fetch("token_types.weight")
    state["embeddings.LayerNorm.weight"] = fetch("token_embd_norm.weight")
    state["embeddings.LayerNorm.bias"] = fetch("token_embd_norm.bias")

    hidden = meta.hidden_size
    for i in range(meta.num_layers):
        gprefix = f"blk.{i}"
        hprefix = f"layers.{i}"

        qkv = fetch(f"{gprefix}.attn_qkv.weight")
        if qkv.shape != (3 * hidden, hidden):
            raise ValueError(f"unexpected QKV shape for block {i}: {qkv.shape}")
        state[f"{hprefix}.self_attn.q_proj.weight"] = qkv[:hidden].contiguous()
        state[f"{hprefix}.self_attn.k_proj.weight"] = qkv[hidden : 2 * hidden].contiguous()
        state[f"{hprefix}.self_attn.v_proj.weight"] = qkv[2 * hidden :].contiguous()
        state[f"{hprefix}.self_attn.o_proj.weight"] = fetch(f"{gprefix}.attn_output.weight")

        state[f"{hprefix}.post_attention_layernorm.weight"] = fetch(f"{gprefix}.attn_output_norm.weight")
        state[f"{hprefix}.post_attention_layernorm.bias"] = fetch(f"{gprefix}.attn_output_norm.bias")
        state[f"{hprefix}.mlp.gate_proj.weight"] = fetch(f"{gprefix}.ffn_gate.weight")
        state[f"{hprefix}.mlp.up_proj.weight"] = fetch(f"{gprefix}.ffn_up.weight")
        state[f"{hprefix}.mlp.down_proj.weight"] = fetch(f"{gprefix}.ffn_down.weight")
        state[f"{hprefix}.post_mlp_layernorm.weight"] = fetch(f"{gprefix}.layer_output_norm.weight")
        state[f"{hprefix}.post_mlp_layernorm.bias"] = fetch(f"{gprefix}.layer_output_norm.bias")

    return state


def to_numpy(tensor: gguf.ReaderTensor) -> np.ndarray:
    name = tensor.tensor_type.name
    if name == "F32":
        arr = np.frombuffer(tensor.data, dtype=np.float32)
    elif name == "F16":
        arr = np.frombuffer(tensor.data, dtype=np.float16)
    else:
        raise ValueError(f"unsupported tensor dtype for nomic-bert: {name}")
    # GGUF shape is little-endian per-dim; ReaderTensor.shape mirrors the
    # on-disk layout which puts fastest-varying dim first. Reverse to get
    # the row-major shape numpy expects.
    return arr.reshape(tuple(int(s) for s in tensor.shape)[::-1])


def eager_compile_embed_with_pinning(
    forward,
    meta: NomicMeta,
    pad_id: int,
    pinned_buckets: dict[tuple[int, int], PinnedBucket],
) -> None:
    """For every (batch, seq) bucket: allocate stable input storage,
    run the compiled wrapper TWICE (first call builds the plan; second
    routes through `_execute_plan` and captures the args tuple), then
    pin (plan, args) into `pinned_buckets`. Request time then hits the pinned
    plan via `_execute_plan(plan, args)` directly."""
    started = time.monotonic()
    pinned_shapes: list[tuple[int, int]] = []
    skipped_shapes: list[tuple[int, int]] = []
    for batch, seq in EMBED_PRECOMPILE_SHAPES:
        if seq > meta.context_length:
            continue
        # Stable storage that survives across compile and request-time
        # `.copy_()` mutations. Filled with `pad_id` / 0 mask so the
        # compile-time forward sees attention-mask=0 everywhere (no real
        # compute, just shape priming).
        input_ids = torch.full((batch, seq), pad_id, dtype=torch.long, device="cpu")
        attention_mask = torch.zeros((batch, seq), dtype=torch.long, device="cpu")

        with torch.no_grad(), capture_plan() as slot:
            # Call 1: compile + plan registration.
            forward(input_ids=input_ids, attention_mask=attention_mask)
            # Call 2: hits the cached plan path so the slot ends holding the
            # executed plan + its args (the wrapper's plan branch).
            forward(input_ids=input_ids, attention_mask=attention_mask)

        plan = slot.plan
        args = slot.args
        if plan is None or args is None:
            skipped_shapes.append((batch, seq))
            continue
        pinned_buckets[(batch, seq)] = PinnedBucket(
            batch=batch,
            seq=seq,
            input_ids=input_ids,
            attention_mask=attention_mask,
            plan=plan,
            args=args,
        )
        pinned_shapes.append((batch, seq))
    elapsed = time.monotonic() - started
    skipped_note = f"  skipped={skipped_shapes}" if skipped_shapes else ""
    print(
        f"[alloy] eager_compile_embed: pinned {len(pinned_shapes)} plans "
        f"{pinned_shapes} in {elapsed:.1f}s{skipped_note}",
        flush=True,
    )


def bucket_seq_len(natural_len: int, ceiling: int) -> int:
    """Round a natural seq length up to the smallest pre-compiled seq bucket
    so every request hits a pinned plan. Caps at the model's context length."""
    for bucket in EMBED_SEQ_BUCKETS:
        if bucket >= natural_len:
            return min(bucket, ceiling)
    return min(EMBED_SEQ_BUCKETS[-1], ceiling)


def bucket_batch_size(real_batch: int) -> int:
    """Round up to the smallest pre-compiled batch bucket. Falls through
    to `real_batch` for sizes beyond what we pre-pinned (the embed()
    fallback path then pays the alloy compile cost on first sight)."""
    batches = sorted({b for b, _ in EMBED_PRECOMPILE_SHAPES})
    for b in batches:
        if b >= real_batch:
            return b
    return real_batch


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def build_nomic_tokenizer(reader: gguf.GGUFReader, *, max_length: int) -> NomicTokenizer:
    tokens_field = reader.fields["tokenizer.ggml.tokens"]
    tokens: list[str] = []
    for idx in tokens_field.data:
        part = tokens_field.parts[idx]
        tokens.append(bytes(part).decode("utf-8", errors="replace"))
    return NomicTokenizer(
        token_to_id={token: index for index, token in enumerate(tokens)},
        cls_id=required_int(reader, "tokenizer.ggml.cls_token_id"),
        sep_id=required_int(reader, "tokenizer.ggml.seperator_token_id"),
        pad_id=required_int(reader, "tokenizer.ggml.padding_token_id"),
        unk_id=required_int(reader, "tokenizer.ggml.unknown_token_id"),
        max_length=max_length,
    )


def bert_normalize(text: str) -> str:
    """nomic-bert tokenizer normalisation: NFD strip-accents + control-
    char filter + whitespace collapse. Notably DOES NOT lowercase.

    The GGUF declares `tokenizer.ggml.model: 'bert'` and ships a
    lowercase-only vocab, but Ollama / llama.cpp's nomic-bert path does
    NOT lowercase before vocab lookup — uppercase words fall through to
    [UNK]. Verified empirically: `embed('Apple')` produces the same
    vector as encoding `[CLS] [UNK] [SEP]` exactly. We mirror that so
    embeddings match Ollama's reference bit-for-bit.
    """
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    cleaned_chars: list[str] = []
    for c in stripped:
        codepoint = ord(c)
        if codepoint == 0 or codepoint == 0xFFFD:
            continue
        if c.isspace() or codepoint < 0x20:
            cleaned_chars.append(" ")
            continue
        cleaned_chars.append(c)
    return " ".join("".join(cleaned_chars).split())


def required_string(reader: gguf.GGUFReader, key: str) -> str:
    return cast(str, reader.fields[key].contents())


def required_int(reader: gguf.GGUFReader, key: str) -> int:
    return int(cast(int, reader.fields[key].contents()))


def required_float(reader: gguf.GGUFReader, key: str) -> float:
    return float(cast(float, reader.fields[key].contents()))


@register("nomic-bert")
class NomicBertHandler:
    """nomic-bert text embeddings. A custom-`load()` handler (own GGUF reader +
    pinned-plan compile, not the causal-LM tensor loop); `kind="embed"` routes the
    server to the embedding modality."""

    arch = ("nomic-bert",)
    kind = "embed"

    def apply_transformers_patches(self) -> None:
        return None

    def load(self, source: ResolvedGGUF) -> EmbeddingModel:
        return load_ollama_gguf_embedder(source)


__all__ = ["NomicBertHandler", "load_ollama_gguf_embedder"]
