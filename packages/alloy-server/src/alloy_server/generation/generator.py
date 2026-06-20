"""Minimal greedy generation runtime for compiled causal language models."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Self, cast

import torch
import transformers
from transformers.cache_utils import StaticCache

if TYPE_CHECKING:
    from alloy_server.speculative.session import SpecSession

from alloy_torch.backend import (
    capture_plan,
    release_plan_intermediates,
)
from alloy import get_logger
from alloy._runtime.convert import to_alloy_buffer
from alloy._dispatch.buf_utils import _alloc_aligned, set_record_only
from alloy._compiler.dtypes import float32, int32, int64
from alloy.std import apply_token_bitmask, sample_categorical
from alloy.std.sampling import SAMPLE_SPLITS
from alloy_server.models.modality import ModalityEncoder
from alloy_server.cache import (
    AlloyLinearAttentionLayer,
)
from alloy_torch.compile_window import grid_shrink_compile
from alloy_server.kv_format import KVFormat, resolve_kv_format
from alloy_server.models.attention import install_multi_token_attention
from alloy_server.graph_cache import graph_caching_compile
from alloy_server.generation.decode import DecodeEngine, GreedyNextToken
from alloy_server.generation.prefill import (
    ChunkPrefill,
    ChunkPrefillEmbeds,
    EmbedTokens,
    PrefillEngine,
)
from alloy_server.generation.prefix import PrefixCache
from alloy_server.generation.kv import ContiguousKV, PagedKV
from alloy_server.generation.plans import PlanStore
from alloy_server.generation.sequence import MultimodalInputs, Sequence
from alloy_server.generation.spec import attach_spec_session

logger = get_logger("alloy_server.generation")

# Chunks at or above this compile a grid-shrink-capable prefill plan (single-pass
# attention + per-call grid override for partial chunks). Below it, small-chunk
# plans keep split-K attention with no shrink machinery — pad waste is bounded by
# the chunk size.
MIN_SHRINK_CHUNK = 256

# Prefix-bookmark deque capacity; also reserved out of the KV fill budget
# (ContiguousKV.bookmark_budget_bytes).
BOOKMARK_SLOTS = 4

class AlloyGenerator:
    """Greedy token generator backed by the Alloy torch.compile backend."""

    model: transformers.PreTrainedModel
    cache_dtype: torch.dtype
    # The compiled chunk sizes. Used to exclude already-compiled sizes from the
    # grid-shrink discovery probe.
    prefill_chunks: tuple[int, ...]
    pad_token_id: int
    # Optional vision/audio front-ends, precompiled by eager_compile_all.
    vision: ModalityEncoder | None
    audio: ModalityEncoder | None

    def __init__(
        self,
        model: transformers.PreTrainedModel,
        cache_dtype: torch.dtype,
        chat_template_auto_injects: bool = False,
        close_think_seq: tuple[int, ...] = (),
        mid_think_heal_seq: tuple[int, ...] = (),
        post_think_heal_seq: tuple[int, ...] = (),
        chunk_prefill_size: int = 128,
        vision: ModalityEncoder | None = None,
        audio: ModalityEncoder | None = None,
        kv_format: KVFormat | None = None,
    ) -> None:
        self.model = model
        # Modality front-ends: unused during text decode (Sequence.embeds carries
        # precomputed features), but owned here so eager_compile_all can
        # precompile their plans alongside the text decoder.
        self.vision = vision
        self.audio = audio
        self.cache_dtype = cache_dtype
        # Most-recent run() phase timings; the server surfaces them under
        # usage.timings.
        self.last_gen_timings: dict[str, float | int] = {}
        # Cache ownership + fill budget, always sized to the model's native
        # context. kv_format is the opt-in quantized-KV format (`--kv-quant` /
        # ALLOY_KV_QUANT); None = the fp16 cache. ALLOY_KV=paged carves cache
        # tensors from one vm-reserved pool with page-level reclaim.
        kv_cls = PagedKV if os.environ.get("ALLOY_KV", "").lower() == "paged" else ContiguousKV
        self.kv = kv_cls(
            model=model,
            cache_dtype=cache_dtype,
            kv_format=kv_format,
            max_cache_len=native_context(model),
            bookmark_slots=BOOKMARK_SLOTS,
        )
        # Prefill loops `PrefillEngine.chunk_step` over fixed-size chunks of
        # `chunk_prefill_size` (batch 1). The server passes 4096 (saturating
        # GEMMs, partial chunks grid-shrunk to the real prompt length).
        #
        # Grid shrink is a property of the compiled chunk plan, engaged for
        # chunks >= MIN_SHRINK_CHUNK: the plan compiles once at SEQ_LEN=chunk
        # (single-pass attention, representative-M config resolution,
        # request-bounded pool), and `PrefillEngine.chunk_step` dispatches an
        # exact threadgroup count for each call's real length via a per-call
        # grid override (the recipe from `_discover_grid_shrink_recipe`) — so
        # padding tiles cost no GPU work; the grid varies per request, not the
        # plan. Small chunks skip the shrink machinery and keep split-K
        # attention. MoE participates: the router records its launched row count
        # in-kernel and the counting sort bounds its scan by it, so shrunk
        # partial chunks never scan the unwritten routing-slot tail.
        if chunk_prefill_size < 1:
            raise ValueError("chunk_prefill_size must be >= 1")
        chunk = min(int(chunk_prefill_size), self.kv.max_cache_len)
        if chunk >= MIN_SHRINK_CHUNK:
            chunk -= chunk % 64
        self.chunk_prefill_size = chunk
        self.grid_shrink = chunk >= MIN_SHRINK_CHUNK
        self.prefill_chunks = (chunk,)
        # Hard-freeze every parameter / buffer before any torch.compile
        # tracing. Without this Dynamo + fake_tensor allocate autograd metadata
        # (versions, view-tracking, grad slots) for every weight, even though
        # this graph is never differentiated.
        for param in model.parameters():
            param.requires_grad_(False)
        for buf in model.buffers():
            buf.requires_grad_(False)
        # chat_template_auto_injects is True when the chat template inserts
        # assistant-turn markers the model must emit during generation (e.g.
        # Qwen3's auto `<think>\n\n</think>\n\n`). When True, warm-prefill saves
        # only the prompt portion of the cache: the decode tokens don't
        # round-trip through (decode -> re-render -> tokenize) for the next turn,
        # so saving them would let LCP extend past where saved tokens diverge
        # from the re-render. When False, save the full output_row so the next
        # turn skips re-prefilling the response. (Carried by
        # PrefixCache.auto_injects below.)
        self.plans = PlanStore(
            hidden_size=int(model.config.hidden_size), grid_shrink=self.grid_shrink,
        )
        self.decode = DecodeEngine(model, self.plans)
        # Chunked prefill + grid shrink. Pad token is never read by attention
        # (the causal mask blocks pad positions), so any in-vocab id works; 0 is
        # safe for every supported tokenizer.
        self.prefill = PrefillEngine(
            model,
            self.plans,
            self.kv,
            chunk_prefill_size=self.chunk_prefill_size,
            grid_shrink=self.grid_shrink,
            prefill_chunks=self.prefill_chunks,
            pad_token_id=0,
            cache_dtype=cache_dtype,
        )
        # End-of-turn token ids. Without these the per-step decode loop runs to
        # max_new_tokens, past the model's real stopping point — output stays
        # coherent for a paragraph then degenerates into hallucinated prompts.
        self.eos_token_ids: tuple[int, ...] = resolve_eos_tokens(model)
        self.constrained_bufs: dict[int, object] = {}  # keyed by vocab size
        # Pad token for the embeds (multimodal) prefill path.
        self.pad_token_id = 0
        # Warm-prefix reuse, bookmarks, truncation heal, side-call snapshot.
        self.prefix = PrefixCache(
            self.kv,
            self.plans,
            self.decode,
            eos_token_ids=self.eos_token_ids,
            close_think_seq=close_think_seq,
            mid_think_heal_seq=mid_think_heal_seq,
            post_think_heal_seq=post_think_heal_seq,
            auto_injects=chat_template_auto_injects,
            bookmark_slots=BOOKMARK_SLOTS,
        )
        # Cross-layer KV sharing (gemma4: `num_kv_shared_layers` trailing layers
        # reuse an earlier layer's K/V). Cache reuse in `kv.acquire` /
        # warm-prefill in `PrefixCache.match` assumes stale K/V beyond the new
        # prompt is masked out by causal attention — but a shared sliding layer
        # doesn't rewrite the padded-bucket tail it shares, so the previous
        # conversation's K/V survives there and the shared layer attends it,
        # leaking cross-prompt content. For these models `generate()` clears the
        # cache tail past the warm prefix before prefilling (`kv.clear_tail`).
        try:
            num_shared = model.config.num_kv_shared_layers
        except AttributeError:
            num_shared = 0
        self.kv_sharing = int(num_shared or 0) > 0
        # Install the multi-token attention monkey-patch up front. It routes K
        # in [2, _MAX_VERIFY_K] through `attention_kv_update_multi` (fused
        # kv-update + attention, for spec-decode verify) and K > _MAX_VERIFY_K
        # against a populated cache through `attention_prefill_warm` (runtime
        # Q_START_POS for warm-prefill). The patch ends with
        # `torch._dynamo.reset()`, so installing it lazily would force a
        # mid-request recompile; eager install keeps the patched forward visible
        # to every compile.
        self.install_multi_token_patch_once()
        self.spec: "SpecSession | None" = None

    @classmethod
    def from_model(
        cls,
        model: transformers.PreTrainedModel,
        *,
        cache_dtype: torch.dtype | None = None,
        chat_template_auto_injects: bool = False,
        close_think_seq: tuple[int, ...] = (),
        mid_think_heal_seq: tuple[int, ...] = (),
        post_think_heal_seq: tuple[int, ...] = (),
        chunk_prefill_size: int = 128,
        vision: ModalityEncoder | None = None,
        audio: ModalityEncoder | None = None,
        kv_format: KVFormat | None = None,
    ) -> Self:
        model.eval()
        resolved_cache_dtype = cache_dtype if cache_dtype is not None else infer_cache_dtype(model)
        # No explicit format -> honor ALLOY_KV_QUANT; unset/"none" = the fp16 cache.
        resolved_kv_format = kv_format if kv_format is not None else resolve_kv_format(None)
        return cls(
            model,
            resolved_cache_dtype,
            chat_template_auto_injects=chat_template_auto_injects,
            close_think_seq=close_think_seq,
            mid_think_heal_seq=mid_think_heal_seq,
            post_think_heal_seq=post_think_heal_seq,
            chunk_prefill_size=chunk_prefill_size,
            vision=vision,
            audio=audio,
            kv_format=resolved_kv_format,
        )

    @property
    def max_cache_len(self) -> int:
        """The KV-cache size — the model's native context."""
        return self.kv.max_cache_len

    @property
    def max_fill(self) -> int:
        """Max KV positions a single request may fill (see ContiguousKV.max_fill)."""
        return self.kv.max_fill

    # --- Constrained decoding (grammar / JSON / forced tool calls) ----------
    # Per-step Python loop: forward -> logits, mask with the grammar bitmask,
    # sample on-GPU, advance the matcher. The matcher (xgrammar) stays in
    # Python; the GPU only consumes the bitmask buffer (~10us/step).

    def constrained_buffers(self, vocab: int) -> dict:
        bufs = cast("dict | None", self.constrained_bufs.get(vocab))
        if bufs is None:
            words = (vocab + 31) // 32
            bufs = {
                "masked": _alloc_aligned((1, vocab), float32),
                "bitmask": _alloc_aligned((1, words), int32),
                "token": _alloc_aligned((1,), int64),
                "pos": _alloc_aligned((1,), int64),
                "seed": _alloc_aligned((1,), int64),
                "params": _alloc_aligned((4,), float32),
                "bitmask_cpu": torch.empty((1, words), dtype=torch.int32),
            }
            self.constrained_bufs[vocab] = bufs
        return bufs

    def masked_sample(self, logits: torch.Tensor, matcher, pos: int, bufs: dict) -> int:
        vocab = int(logits.shape[-1])
        # Grammar mask (xgrammar, CPU) -> GPU buffer.
        matcher.fill_next_token_bitmask(bufs["bitmask_cpu"])
        bufs["bitmask"].copy_from(bufs["bitmask_cpu"].data_ptr())
        bufs["pos"].write_scalar(pos)  # RNG counter for this step
        logits_buf = to_alloy_buffer(logits).reshape((1, vocab))
        apply_token_bitmask(logits_buf, bufs["bitmask"], bufs["masked"])
        sample_categorical(
            bufs["masked"], bufs["pos"], bufs["seed"], bufs["params"], bufs["token"],
        )
        bufs["token"].sync()
        return int(bufs["token"].numpy.reshape(-1)[0])

    def splice_and_prefill_embeds(
        self,
        mm: MultimodalInputs,
        input_ids: torch.Tensor,
        cache: StaticCache,
        prefix_len: int,
    ) -> torch.Tensor:
        """Multimodal prefill input mode: text embeddings from the quantized
        `embed_tokens` (alloy dispatch), vision/audio features spliced into the
        placeholder slots, spliced embeddings prefilled via the embeds chunk
        loop. Decode afterwards is plain text generation.

        Warm-prefix reuse works across vision turns as for text: placeholder
        tokens are identical turn to turn, so a follow-up LCP-matches through the
        image prefix and reuses its cached KV (features baked in at turn 1); only
        the cold suffix is prefilled, and only its placeholder slots spliced."""
        pad_id = self.model.config.pad_token_id or 0
        cold = mm.positions >= prefix_len
        suffix_positions = mm.positions[cold] - prefix_len
        suffix_features = mm.features[cold]
        suffix_ids = input_ids[:, prefix_len:].clone()
        if int(suffix_positions.numel()) > 0:
            suffix_ids[0, suffix_positions] = pad_id  # PAD the OOV image id
        embeds, per_layer_inputs = self.prefill.embed_module_compiled(input_ids=suffix_ids)
        embeds = embeds.clone()
        if int(suffix_positions.numel()) > 0:
            embeds[0, suffix_positions] = suffix_features.to(embeds.dtype)
        return self.prefill.chunked_embeds(
            embeds, per_layer_inputs, cache, start_pos=prefix_len,
        )

    def run(self, seq: Sequence) -> Iterator[int]:
        """THE batch-1 generation pipeline: budget → warm/cold prefill →
        decode → heal → prefix save. Yields each decoded token id; fills
        `seq.generated` / `seq.healed` / `seq.finish_reason`. Every public
        generate/stream entry point wraps this.

        `seq.stream` selects the decode-chunk cascade: (8,) streams in ~8-token
        bursts; non-streaming uses (32, 8) — the 32 chunk amortizes the
        command-buffer commit, the 8 cascade keeps the tail off single-step
        command buffers. Token output is identical either way (chunking is
        GPU-side batching; EOS overshoot is never emitted).
        """
        self.require_modules()
        input_ids = seq.input_ids
        validate_input_ids(input_ids)
        if int(input_ids.shape[0]) != 1:
            raise ValueError("run requires batch size 1 (Sequence is single-request)")
        if seq.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if seq.max_new_tokens == 0:
            return
        if seq.embeds is not None and seq.constraint is not None:
            raise ValueError("constrained multimodal decode is not supported")
        if seq.embeds is not None and int(seq.embeds.features.shape[0]) != int(
            seq.embeds.positions.shape[0]
        ):
            raise ValueError(
                f"multimodal features has {int(seq.embeds.features.shape[0])} rows "
                f"but {int(seq.embeds.positions.shape[0])} placeholder positions"
            )
        if seq.sampling is not None:
            s = seq.sampling
            # In-place writes into the pinned buffers the compiled plans bind;
            # the GPU reads them live.
            self.plans.params[0] = float(s.temperature)
            self.plans.params[1] = float(s.top_p)
            self.plans.params[2] = float(s.top_k)
            self.plans.params[3] = float(s.min_p)
            # Sampler split count: a top-k/p/min-p filter needs a global threshold
            # bisection that can't be split across threadgroups, so it runs
            # single-split (split 0 = whole vocab); greedy / pure-temperature use
            # the full SAMPLE_SPLITS for the bandwidth-bound parallel argmax.
            filtered = float(s.top_p) < 1.0 or float(s.top_k) >= 1.0 or float(s.min_p) > 0.0
            self.plans.params[4] = 1.0 if filtered else float(SAMPLE_SPLITS)
            self.plans.seed[0] = int(s.seed)

        prompt_len = seq.prompt_len
        max_new_tokens = self.kv.fit_to_budget(prompt_len, seq.max_new_tokens)
        cache_len = self.kv.cache_len_for(prompt_len + max_new_tokens + 1)
        t_prefill_start = time.perf_counter()
        cache, prefix_len = self.prefix.match(input_ids, cache_len, 1, prompt_len)
        # Cross-layer KV-sharing models (gemma4): clear stale K/V in the cache
        # tail past the warm prefix before prefilling. A shared sliding layer
        # doesn't rewrite the padded-bucket tail it shares, so the previous
        # request's K/V at [prefix_len, max_cache_len) survives and the shared
        # layer attends it, leaking content. Reused positions [0, prefix_len)
        # stay valid, so continuation still prefills only the delta.
        if self.kv_sharing:
            self.kv.clear_tail(cache, prefix_len)
        # Warm-prefill uses a suffix-sized bucket. The SDPA handler reads
        # `compile_window.q_start_pos` (set in `PrefillEngine.chunk_step`) to
        # keep the full K/V extent instead of the cold slice-to-bucket; the
        # causal mask carries the absolute-position offsets.
        suffix_input = input_ids if prefix_len == 0 else input_ids[:, prefix_len:]
        # ignore_eos: decode the full max_new_tokens regardless of EOS — a fixed
        # tg count for throughput benchmarks (an early EOS on random tokens would
        # otherwise corrupt the measurement).
        eos = () if seq.ignore_eos else self.eos_token_ids

        if seq.constraint is not None:
            yield from self.run_constrained(
                seq, cache, cache_len, prefix_len, max_new_tokens, t_prefill_start,
            )
            return

        prompt_tokens = [int(t) for t in input_ids[0].tolist()]
        with torch.inference_mode():
            if seq.embeds is None:
                # Chunk-boundary prefix marks give later forks resume points
                # inside this prompt's long prefix.
                next_token = self.prefill.run(
                    suffix_input, cache, start_pos=prefix_len,
                    on_chunk=lambda pos: self.prefix.mark_prefix(pos, prompt_tokens, cache),
                )
            else:
                next_token = self.splice_and_prefill_embeds(
                    seq.embeds, input_ids, cache, prefix_len,
                )
            t_prefill_end = time.perf_counter()
            # Warm-prefill state save: the chat-template-rendered input_ids
            # (= what was prefilled), extended with the decoded tokens at the end
            # so the next turn's LCP reaches through the assistant turn boundary.
            self.prefix.save(cache_len, prompt_tokens, cache)

            first_tok = int(next_token[0, 0].item())
            seq.generated.append(first_tok)
            yield first_tok
            if not (eos and first_tok in eos):
                for tok in self.decode.loop(
                    cache, cache_len, prompt_len, next_token, max_new_tokens,
                    chunks=(8,) if seq.stream else (32, 8),
                ):
                    seq.generated.append(tok)
                    yield tok
                    if eos and tok in eos:
                        break
            seq.finish_reason = (
                "stop" if (eos and seq.generated[-1] in eos) else "length"
            )
            # If decode hit max_new_tokens without EOS, run the close-think +
            # turn-end tokens through the model so the cache ends "turn done" —
            # the next turn's warm splice then reads consistent K/V instead of
            # mid-emission state. Heal ids extend the saved prefix (real cache
            # rows) but are never yielded.
            heal_tokens: list[torch.Tensor] = [input_ids, next_token] + [
                torch.tensor([[t]], dtype=torch.long, device=input_ids.device)
                for t in seq.generated[1:]
            ]
            before_heal = len(heal_tokens)
            self.prefix.heal_truncated(
                heal_tokens, cache, cache_len, prompt_len, input_ids.device,
            )
            seq.healed = [
                int(t[0, 0].item()) for t in heal_tokens[before_heal:]
            ]
        t_decode_end = time.perf_counter()

        self.prefix.extend(seq.generated + seq.healed)
        self.prefix.bookmark()
        self.log_request_complete(
            seq, prompt_len, prefix_len, t_prefill_start, t_prefill_end, t_decode_end,
        )

    def run_constrained(
        self,
        seq: Sequence,
        cache: StaticCache,
        cache_len: int,
        prefix_len: int,
        max_new_tokens: int,
        t_prefill_start: float,
    ) -> Iterator[int]:
        """Grammar-constrained decode tail of `run`: prefill all but the last
        prompt token, then per-step masked sampling — forward → logits, mask
        with the grammar bitmask, sample on-GPU, advance the matcher.

        Stops on the chat EOS ∪ the grammar's stop tokens — the grammar often
        terminates on a different token (e.g. <|endoftext|>) than the chat EOS
        (<|im_end|>), and the stop token must not be yielded into the output.
        """
        matcher = seq.constraint
        input_ids = seq.input_ids
        prompt_len = seq.prompt_len
        stop_ids = set(self.eos_token_ids)
        stop_ids.update(int(t) for t in matcher.stop_token_ids)
        with torch.inference_mode():
            # Prefill all but the last prompt token; the last token's forward
            # yields the first decode logits (grammar-masked — an unmasked
            # prefill sample would be matcher-rejected or invalid).
            head = input_ids[:, : prompt_len - 1]
            if int(head.shape[1]) > prefix_len:
                self.prefill.run(head[:, prefix_len:], cache, start_pos=prefix_len)
            t_prefill_end = time.perf_counter()
            # Save the prefilled rows only; the last prompt token's row lands
            # on the first masked step and is folded in by the final extend.
            self.prefix.save(
                cache_len, [int(t) for t in input_ids[0, : prompt_len - 1].tolist()], cache,
            )
            bufs = self.constrained_buffers(int(self.model.config.vocab_size))
            bufs["seed"].write_scalar(int(self.plans.seed[0].item()))
            bufs["params"].copy_from(self.plans.params.contiguous().data_ptr())

            pos = prompt_len - 1
            # Last prompt token via the pinned input — a sliced input_ids view's
            # storage offset trips the warmed decode graph's guards, recompiling
            # a second specialization inline.
            self.plans.token_input[0, 0] = int(input_ids[0, prompt_len - 1])
            self.plans.cache_position[0] = pos
            logits = self.decode.next_logits(self.plans.token_input, cache)
            seq.finish_reason = "length"
            for _ in range(max_new_tokens):
                token = self.masked_sample(logits, matcher, pos, bufs)
                if token in stop_ids:
                    seq.finish_reason = "stop"
                    break
                if not matcher.accept_token(token):
                    # The matcher rejected its own sample — an empty/buggy
                    # bitmask (e.g. stop tokens missing from TokenizerInfo)
                    # would otherwise loop garbage to max_tokens.
                    logger.warning("constrained_decode_dead_end", token=token)
                    seq.finish_reason = "stop"
                    break
                seq.generated.append(token)
                yield token
                if matcher.is_terminated():
                    seq.finish_reason = "stop"
                    break
                pos += 1
                self.plans.token_input[0, 0] = token
                self.plans.cache_position[0] = pos
                logits = self.decode.next_logits(self.plans.token_input, cache)
        t_decode_end = time.perf_counter()
        # Fold the last prompt token + accepted tokens into the saved prefix
        # (their cache rows were written by the masked steps).
        self.prefix.extend([int(input_ids[0, prompt_len - 1])] + seq.generated)
        self.prefix.bookmark()
        self.log_request_complete(
            seq, prompt_len, prefix_len, t_prefill_start, t_prefill_end, t_decode_end,
        )

    def log_request_complete(
        self,
        seq: Sequence,
        prompt_len: int,
        prefix_len: int,
        t_prefill_start: float,
        t_prefill_end: float,
        t_decode_end: float,
    ) -> None:
        prefill_ms = (t_prefill_end - t_prefill_start) * 1000.0
        decode_ms = (t_decode_end - t_prefill_end) * 1000.0
        decode_tokens = len(seq.generated) + len(seq.healed)
        tpot_ms = decode_ms / decode_tokens if decode_tokens > 0 else 0.0
        self.last_gen_timings = {
            "prefill_ms": prefill_ms,
            "decode_ms": decode_ms,
            "prompt_tokens": prompt_len,
            "decode_tokens": decode_tokens,
            "prefill_bucket": self.chunk_prefill_size,
            "warm_prefix_len": prefix_len,
        }
        logger.info(
            "request_complete",
            prompt_tokens=prompt_len,
            decode_tokens=decode_tokens,
            prefill_ms=round(prefill_ms, 2),
            decode_ms=round(decode_ms, 2),
            ttft_ms=round(prefill_ms + tpot_ms, 2),
            tpot_ms=round(tpot_ms, 3),
            warm_prefix_len=prefix_len,
            prefill_bucket=self.chunk_prefill_size,
            batch_size=1,
        )

    def generate(self, input_ids: torch.Tensor, *, max_new_tokens: int) -> torch.Tensor:
        validate_input_ids(input_ids)
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if max_new_tokens == 0:
            return input_ids.clone()
        if int(input_ids.shape[0]) > 1:
            return self.generate_batched(input_ids, max_new_tokens=max_new_tokens)
        seq = Sequence(input_ids=input_ids, max_new_tokens=max_new_tokens)
        for _ in self.run(seq):
            pass
        new = torch.tensor(
            [seq.generated + seq.healed], dtype=torch.long, device=input_ids.device,
        )
        return torch.cat([input_ids, new], dim=1)

    def generate_batched(
        self, input_ids: torch.Tensor, *, max_new_tokens: int,
    ) -> torch.Tensor:
        """Batch>1 fallback: no pinned-tensor counterpart exists, so prefill is
        a single per-shape forward and decode is per-step module calls. No warm
        prefix, no heal — a bench/eval path."""
        self.require_modules()
        prompt_len = int(input_ids.shape[1])
        batch_size = int(input_ids.shape[0])
        max_new_tokens = self.kv.fit_to_budget(prompt_len, max_new_tokens)
        cache_len = self.kv.cache_len_for(prompt_len + max_new_tokens + 1)
        tokens: list[torch.Tensor] = [input_ids]
        cache = self.kv.acquire(batch_size, cache_len)
        with torch.inference_mode():
            next_token = self.decode.next_token(
                input_ids,
                cache,
                torch.arange(prompt_len, dtype=torch.int32, device=input_ids.device),
            )
            tokens.append(next_token.clone())
            eos = self.eos_token_ids
            cache_position = torch.empty((1,), dtype=torch.int32, device=input_ids.device)
            for step in range(1, max_new_tokens):
                cache_position[0] = prompt_len + step - 1
                next_token = self.decode.next_token(next_token, cache, cache_position)
                tokens.append(next_token.clone())
                if eos and int(next_token[0, 0].item()) in eos:
                    break
        self.prefix.invalidate_live()
        return torch.cat(tokens, dim=1)

    def preserving_prefix(self, side_total: int):
        """See PrefixCache.preserving (kept here as the server-facing handle)."""
        return self.prefix.preserving(side_total)

    def stream_chunks_fast(
        self, input_ids: torch.Tensor, *, max_new_tokens: int,
    ) -> Iterator[int]:
        """Per-token streaming wrapper over `run` (small decode chunks). If the
        consumer stops iterating early, heal/prefix-extension don't run and the
        next turn pays full prefill."""
        yield from self.run(
            Sequence(input_ids=input_ids, max_new_tokens=max_new_tokens, stream=True)
        )

    def reset_prefix_state(self) -> None:
        """See PrefixCache.reset (kept here as the server/CLI-facing handle)."""
        self.prefix.reset()

    def install_multi_token_patch_once(self) -> None:
        """Install the multi-token attention monkey-patch for any model class
        with a patched forward. The patch emits
        `alloy.attention_kv_update_multi` / `alloy.attention_prefill_warm`
        opaquely so AOT autograd doesn't lift the cache mutation out of the FX
        graph and so warm-suffix prefill scales with new-token count, not full
        prompt length.
        """
        cls_name = type(self.model).__name__
        modules = list(self.model.modules())
        supported = ("Qwen3Attention", "Qwen2Attention", "LlamaAttention", "Gemma3Attention", "Gemma4TextAttention", "Qwen3_5Attention", "Lfm2Attention")
        if any(s in cls_name for s in ("Qwen3", "Qwen2", "Llama", "Gemma3", "Gemma4", "Qwen3_5", "Lfm2")) or any(
            type(m).__name__ in supported for m in modules
        ):
            install_multi_token_attention(self.model)

    def prefill_prompt(self, input_ids: torch.Tensor, cache: StaticCache, cache_len: int) -> int:
        """Prefill the prompt through the same pinned-plan prefill path plain
        `generate` uses, dispatching eagerly-compiled plans via `_execute_plan`.
        Returns the argmax after the prompt (== plain's first token, so spec
        stays token-exact). NOT the verify module, which would re-enter
        torch.compile."""
        nt = self.prefill.run(input_ids, cache, start_pos=0)
        return int(nt[0, 0].item())

    def attach_spec(self, drafter) -> None:
        """Attach a contract drafter (see generation.spec.attach). The
        drafter's per-token state counts against the KV fill budget; the
        prefill engine reads the taps."""
        self.spec = attach_spec_session(self, drafter)
        self.kv.spec = self.spec
        self.prefill.spec = self.spec

    def eager_compile_step_count(self) -> int:
        """How many distinct phases `eager_compile_all` will run.

        Counted upfront so callers can render a determinate progress
        bar: cold prefill per bucket + warm prefill per sub-max bucket
        + decode + (optional) spec-decode verify when a draft is attached.
        """
        max_cache = self.kv.max_cache_len
        n_cold = len(self.prefill_chunks)
        warm_split = self.prefill.sliding_split(self.kv.acquire(1, max_cache))
        n_warm = sum(1 for b in self.prefill_chunks if b < max_cache) if warm_split else 0
        n_decode = 1  # one decode plan: sampled token + logits as outputs
        n_spec = 1 if self.spec is not None else 0
        n_vision = 1 if self.vision is not None else 0
        n_audio = 1 if self.audio is not None else 0
        return n_cold + n_warm + n_decode + n_spec + n_vision + n_audio

    def plan_compile_window(self, bucket: int | None = None) -> Iterator[None]:
        """See PlanStore.compile_window (kept here for the capture CLIs)."""
        return self.plans.compile_window(bucket)

    def require_modules(self) -> None:
        """Enforce the precondition: eager_compile_all (which calls
        build_modules) must run before any generation. The engines have no
        lazy-compile fallback, so a missing module is a caller error surfaced
        here instead of as an opaque None call."""
        if self.decode.module is None or self.prefill.module is None:
            raise RuntimeError(
                "eager_compile_all() must be called before generation "
            )

    def build_modules(self) -> None:
        """Build every compiled module the engines dispatch and inject them.
        Called once at the top of eager_compile_all, after attach_spec so the
        prefill module bakes the drafter's tap layers. Idempotent."""
        if self.decode.module is None:
            self.decode.module = graph_caching_compile(
                GreedyNextToken(self.model), "decode",
            )
        if self.prefill.module is None:
            taps = self.spec.drafter.taps.layer_ids if self.spec is not None else ()
            self.prefill.module = graph_caching_compile(
                ChunkPrefill(self.model, taps), "prefill",
            )
        # Multimodal text-embed lookup + chunked inputs_embeds prefill: only
        # vision/audio models ever dispatch these.
        if (self.vision is not None or self.audio is not None) and self.prefill.embed_module_compiled is None:
            self.prefill.embed_module_compiled = torch.compile(
                EmbedTokens(self.model), backend="alloy", dynamic=False,
            )
            self.prefill.embeds_module = torch.compile(
                ChunkPrefillEmbeds(self.model), backend="alloy", dynamic=False,
            )

    def eager_compile_all(
        self,
        *,
        progress: "Callable[[int, int, str], None] | None" = None,
    ) -> None:
        """Pre-compile every prefill bucket (cold + warm) + decode + verify
        before the first real request, so no production call pays torch.compile
        cost inline.

        - Cold prefill: each prefill bucket, fresh cache.
        - Warm prefill: each sub-max prefill bucket, primed cache with
          is_initialized=True (the multi-turn follow-up path).
        - Decode: one (1,1) call against a primed cache.
        - Speculative session: delegated to SpecSession.warmup() when a
          drafter is attached.

        Synthetic inputs are zeros; the alloy backend's plan compile depends on
        shape + dtype + device only.
        """
        self.build_modules()
        device = next(self.model.parameters()).device
        max_cache = self.kv.max_cache_len
        t0 = time.perf_counter()
        n_cold = 0
        total_steps = self.eager_compile_step_count()
        step = 0
        logger.info(
            "eager_compile_start",
            n_prefill_buckets=len(self.prefill_chunks),
            max_cache=max_cache,
            total_steps=total_steps,
        )
        # Each (bucket, is_warm) compiles on call 1, then is exercised via
        # `_execute_plan` on call 2 (inside a `capture_plan()` scope) to pin
        # (plan, args) for the torch.compile-bypass fast path in
        # `PrefillEngine.chunk_step`. Without the second call the captured plan
        # has no `_cached_input_updates`, which the plan-replay path needs.
        for bucket in self.prefill_chunks:
            step += 1
            if progress is not None:
                progress(step, total_steps, f"prefill cold · bucket {bucket}")
            real_len = max(1, bucket - 1)
            cache = self.kv.acquire(1, max_cache)
            # Shrink-capable chunks compile attention single-pass with M-saturated
            # config resolution and a request-bounded pool; small chunks keep the
            # handler's split-K choice.
            shrink_m = bucket if self.grid_shrink else 0
            with grid_shrink_compile(shrink_m), torch.inference_mode():
                dummy = torch.zeros((1, real_len), dtype=torch.long, device=device)
                # Record-only through both calls: call 1 (run-0) builds the plan
                # from dispatch metadata; call 2 (run-1) builds + caches
                # `_cached_input_updates` for the pinned fast path. Both are
                # metadata built before GPU dispatch, so neither runs the GPU.
                set_record_only(True)
                try:
                    # Call 1 compiles a full plan WITHOUT _cached_input_updates;
                    # call 2 compiles the separate plan we pin (with the updates).
                    # The two don't share a pool, so call 1's intermediate pool
                    # (~1.5 GB) is dead the moment call 2 pins — capture and free.
                    with capture_plan() as build_slot:
                        self.prefill.chunk_step(dummy, cache, bucket, start_pos=0)
                    # Reset cache to fresh state so the second prefill takes the
                    # same cold path (start_pos=0 + cumulative_length=0).
                    for layer in cache.layers:
                        layer.cumulative_length.fill_(0)
                    with capture_plan() as slot:
                        self.prefill.chunk_step(dummy, cache, bucket, start_pos=0)
                finally:
                    set_record_only(False)
                self.plans.pin_prefill_plan(bucket, is_warm=False, captured=slot)
                if build_slot.plan is not None and build_slot.plan is not slot.plan:
                    release_plan_intermediates(build_slot.plan)
            n_cold += 1
        # Decode plan. The decode graph reads the cache, token, and
        # cache_position as runtime inputs and is position-independent, so a
        # fresh cache suffices to compile + capture it. The first call builds the
        # plan; subsequent calls populate `_cached_input_updates`. cache.update
        # can flip a guard on the first execute, so loop until the captured plan
        # is execute-ready (cap at 4 to bound a pathological recompile), then pin.
        step += 1
        if progress is not None:
            progress(step, total_steps, "decode")
        decode_cache = self.kv.acquire(1, max_cache)
        # Prime the cache (has_previous_state=True) so warmup compiles only the
        # True decode graph. Production prefill's epilogue
        # (PrefillEngine.chunk_step) sets the flag True before any decode, so
        # production decode is always the True graph. Primed, warmup is a
        # deterministic 2 passes (build + replay); the loop stays as a safety net.
        for layer in decode_cache.layers:
            if isinstance(layer, AlloyLinearAttentionLayer):
                layer.has_previous_state = True
        with torch.inference_mode():
            self.plans.token_input.zero_()
            with capture_plan() as slot:
                warm_i = 0
                while warm_i < 4 and (
                    slot.plan is None or slot.plan._cached_input_updates is None
                ):
                    self.plans.cache_position[0] = warm_i
                    # Record-only, like the prefill capture. Executing for real
                    # wires the WHOLE native KV resident on the decode plan's
                    # first encoder use (+8 GB @262144), and isn't needed: the
                    # has_previous_state flip that compiles both decode
                    # specializations is a Python side effect of the traced
                    # forward, and Dynamo's recompile is independent of
                    # record_only, so both graphs compile and
                    # _cached_input_updates is still captured.
                    set_record_only(True)
                    try:
                        self.decode.next_token(
                            self.plans.token_input,
                            decode_cache,
                            self.plans.cache_position,
                        )
                    finally:
                        set_record_only(False)
                    warm_i += 1
            if slot.plan is not None and slot.plan._cached_input_updates is not None:
                self.plans.decode_plans[max_cache] = slot.plan
        smallest_bucket = self.prefill_chunks[0]
        n_warm = 0
        for bucket in self.prefill_chunks:
            # Non-sliding models replay the cold plan for warm prefill
            # (`PrefillEngine.plan_key` drops is_warm) — nothing to compile here.
            if bucket >= max_cache or not self.prefill.warm_split:
                continue
            step += 1
            if progress is not None:
                progress(step, total_steps, f"prefill warm · bucket {bucket}")
            warm_cache = self.kv.acquire(1, max_cache)
            # Single-pass attention for the shrink-capable warm plan too (the prime
            # below replays the pinned chunk plan, so its split-K choice is
            # unaffected by the flag).
            shrink_m = bucket if self.grid_shrink else 0
            with grid_shrink_compile(shrink_m), torch.inference_mode():
                prime_input = torch.zeros(
                    (1, max(1, smallest_bucket - 1)), dtype=torch.long, device=device,
                )
                # The whole warm step is record-only. The prime advances the
                # cache's cumulative_length via a CPU fill in
                # PrefillEngine.chunk_step, so start_pos is correct; then run-0
                # builds the warm plan and the capture replay caches its updates.
                set_record_only(True)
                try:
                    self.prefill.chunk_step(prime_input, warm_cache, smallest_bucket, start_pos=0)
                    start_pos = int(warm_cache.layers[0].cumulative_length.item())
                    warm_input = torch.zeros(
                        (1, max(1, bucket - 1)), dtype=torch.long, device=device,
                    )
                    # Build call compiles a discarded warm plan; the capture below
                    # pins a separate one — free the build pool.
                    with capture_plan() as build_slot:
                        self.prefill.chunk_step(warm_input, warm_cache, bucket, start_pos=start_pos)
                    # Reset cumulative_length back to the warm offset so the replay
                    # lands at the same start_pos.
                    for layer in warm_cache.layers:
                        layer.cumulative_length.fill_(start_pos)
                    with capture_plan() as slot:
                        self.prefill.chunk_step(warm_input, warm_cache, bucket, start_pos=start_pos)
                finally:
                    set_record_only(False)
                self.plans.pin_prefill_plan(bucket, is_warm=True, captured=slot)
                if build_slot.plan is not None and build_slot.plan is not slot.plan:
                    release_plan_intermediates(build_slot.plan)
            n_warm += 1
        # Discover which grid axes scale with the prompt length (so per-request
        # dispatch is exactly sized) once both chunk plans are pinned. No-op for
        # small (non-shrink) chunks.
        self.prefill.discover_grid_shrink_recipe(device, max_cache)
        # The validator's probe prefills (when enabled) dirty the live cache;
        # leave generation state clean for the first real request.
        self.prefix.invalidate_live()
        # Speculative session: pin the M-row verify plan and the drafter's own
        # plans so the first spec request pays no compile cost inline.
        if self.spec is not None:
            step += 1
            if progress is not None:
                progress(step, total_steps, f"spec {self.spec.drafter.name}")
            self.spec.warmup()
        # ModalityEncoder front-ends compile their alloy plans here too, so one
        # eager_compile_all() warms the whole served model.
        if self.vision is not None:
            step += 1
            if progress is not None:
                progress(step, total_steps, "vision")
            self.vision.eager_compile_all()
        # The conformer length follows the clip duration, so this warms one
        # representative length (others recompile on first use, like text prefill).
        if self.audio is not None:
            step += 1
            if progress is not None:
                progress(step, total_steps, "audio")
            self.audio.eager_compile_all()
        # Paged: allocate the spare slice set and dispatch-wire every buffer's
        # VA resident now, off the request path, so a fresh conversation reuses a
        # pre-wired slice instead of paying Metal's per-slice first-encoder-use
        # wiring on its first prefill. No-op for contiguous.
        self.prefix.prewarm_slices(self.kv.cache_for(1, max_cache))
        elapsed = time.perf_counter() - t0
        logger.info(
            "eager_compile_complete",
            n_cold=n_cold,
            n_warm=n_warm,
            n_plans=n_cold + n_warm + 1,  # cold + warm + decode
            took_ms=round(elapsed * 1000.0, 1),
        )

    def warmup(self) -> None:
        """Pre-compile the decode plan at the native cache size and pin it.

        Runs a tiny prefill + 2 decode steps so the decode plan is compiled,
        executed (populating _cached_input_updates), and pinned in
        plans.decode_plans — the first real request then skips the compile.
        """
        prompt_len = 4
        input_ids = torch.tensor([list(range(prompt_len))], dtype=torch.long)
        cache_len = self.kv.max_cache_len
        cache = self.kv.acquire(1, cache_len)
        with torch.inference_mode():
            out_tok = self.decode.next_token(
                input_ids, cache, torch.arange(prompt_len, dtype=torch.int32)
            )
            cache_position = torch.empty((1,), dtype=torch.int32)

            cache_position[0] = prompt_len
            decode_token = out_tok.clone()
            out_tok = self.decode.next_token(decode_token, cache, cache_position)
            decode_token.copy_(out_tok)

            cache_position[0] = prompt_len + 1
            with capture_plan() as slot:
                out_tok = self.decode.next_token(decode_token, cache, cache_position)
            self.plans.decode_plans[cache_len] = slot.plan


def infer_cache_dtype(model: transformers.PreTrainedModel) -> torch.dtype:
    for parameter in model.parameters():
        return parameter.dtype
    return torch.float32


def native_context(model: transformers.PreTrainedModel) -> int:
    """The model's native context length (``max_position_embeddings``) — the
    cache size production allocates and tunes at. Read off the text sub-config so
    multimodal models (gemma4) resolve their decoder context."""
    config = model.config
    if hasattr(config, "get_text_config"):
        config = config.get_text_config(decoder=True)
    return int(config.max_position_embeddings)


def resolve_eos_tokens(model: transformers.PreTrainedModel) -> tuple[int, ...]:
    """Collect end-of-generation token ids from model + generation configs.

    Chat-tuned models (Qwen3, Llama 3.1+) terminate turns on `<|im_end|>` /
    `<|eot_id|>`, which `generation_config.eos_token_id` carries as a list;
    `config.eos_token_id` is usually the legacy `<|endoftext|>` only.
    Union both so the per-step decode loop stops on either.
    """
    ids: set[int] = set()

    def collect(value: object) -> None:
        if isinstance(value, int) and value >= 0:
            ids.add(value)
        elif isinstance(value, (list, tuple)):
            for v in value:
                if isinstance(v, int) and v >= 0:
                    ids.add(v)

    config = model.config
    bos = config.bos_token_id if hasattr(config, "bos_token_id") else None
    if hasattr(config, "eos_token_id"):
        collect(config.eos_token_id)
    gen_config = model.generation_config if hasattr(model, "generation_config") else None
    if gen_config is not None and hasattr(gen_config, "eos_token_id"):
        collect(gen_config.eos_token_id)
    # Gemma3/Gemma4 chat-tuned models terminate turns on the end-of-turn
    # marker (id 106 — `<end_of_turn>` for gemma3, `<turn|>` for gemma4) but
    # the GGUF config only ships `<eos>` (id 1). Without 106 in the EOS set,
    # the model emits the turn marker then continues looping. Add it
    # explicitly so the per-step decode loop stops correctly.
    model_name = type(model).__name__
    if "Gemma3" in model_name or "Gemma4" in model_name:
        ids.add(106)
    if isinstance(bos, int):
        ids.discard(bos)
    return tuple(sorted(ids))


def validate_input_ids(input_ids: torch.Tensor) -> None:
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape (batch, sequence)")
    if input_ids.dtype != torch.long:
        raise ValueError("input_ids must use torch.long dtype")
