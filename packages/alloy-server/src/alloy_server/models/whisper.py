"""Serve Whisper STT through alloy: load a Whisper GGUF, compile BOTH the encoder
and the decoder through the alloy backend (Metal), and drive HF generate() for
transcription/translation.

Both halves run on alloy's compiled backend at the model's native precision (the
q8_0 GGUF dequantizes to f32 — whisper-tiny is an f32 model; we do not force f16).
The decoder self-attention routes through alloy's `attention_cache` op against an
`AlloyStaticCache` (a single in-graph `cumulative_length.add_` advances the position,
the GreedyNextToken pattern); cross-attention recomputes K/V from the fixed encoder
output each step. HF generate() owns the seq2seq algorithm (long-form seeking,
timestamps, temperature fallback).

Tokenizer / feature extractor / generation config are NOT in the GGUF and load from
the canonical openai/whisper-* tokenizer (local cache only, never download).
"""

from __future__ import annotations

import types
import zlib
from collections.abc import Iterator

import torch
from transformers import EncoderDecoderCache, GenerationConfig, WhisperProcessor
from transformers.cache_utils import StaticCache
from transformers.generation.logits_process import SuppressTokensLogitsProcessor
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.whisper.modeling_whisper import eager_attention_forward

import alloy
import alloy_torch.backend  # noqa: F401  registers the "alloy" torch.compile backend
from alloy._runtime import _metal_ext
from alloy_server.audio_io import load_audio
from alloy_torch.backend import capture_plan
from alloy_server.gguf import ResolvedGGUF
from alloy_server.models.whisper_compat import build_whisper_eager
from alloy_server.cache import AlloyStaticCache
from alloy_server.generation.plans import PlanStore
from alloy_server.models.attention import alloy_cache_attention, use_alloy_cache_op
from alloy_server.models.registry import register
from alloy_server.transcription import Segment, TranscriptionModel, TranscriptionResult

SAMPLE_RATE = 16000
ENCODER_FRAMES = 3000  # Whisper's encoder is fixed at 30s (3000 mel frames)
CANONICAL_TOKENIZER = "openai/whisper-tiny"


class EncoderForward(torch.nn.Module):
    """Compilable encoder: input_features -> last_hidden_state tensor."""

    def __init__(self, encoder: torch.nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        return self.encoder(input_features).last_hidden_state


class GenerateEncoder(torch.nn.Module):
    """Drop-in for model.model.encoder inside generate(): runs the alloy-compiled
    encoder and returns a BaseModelOutput, casting to the model dtype (alloy returns
    f32). Holds the compiled module (not a self-reference), so no recursion."""

    main_input_name = "input_features"

    def __init__(self, compiled: torch.nn.Module, original: torch.nn.Module, dtype: torch.dtype) -> None:
        super().__init__()
        self.compiled = compiled
        self.config = original.config
        self.compute_dtype = dtype
        # generate() reads encoder.conv1/conv2.stride for frame math; keep them.
        self.conv1 = original.conv1
        self.conv2 = original.conv2

    def forward(self, input_features=None, **kwargs) -> BaseModelOutput:
        out = self.compiled(input_features.to(self.compute_dtype)).to(self.compute_dtype)
        return BaseModelOutput(last_hidden_state=out)


def alloy_whisper_attention(self, hidden_states, key_value_states=None, past_key_values=None,
                            attention_mask=None, output_attentions=False, **kwargs):
    """Patched WhisperAttention.forward for the compiled decoder. Decoder SELF-attention
    routes through alloy's `attention_cache` op (upstream SDPA can't trace the int causal
    mask); CROSS-attention recomputes K/V from the fixed encoder output and runs full
    (non-causal) attention. q is unscaled; self.scaling is passed to the op/interface."""
    is_cross = key_value_states is not None
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    kv_shape = (input_shape[0], -1, self.num_heads, self.head_dim)
    q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2).contiguous()

    cache = None
    if isinstance(past_key_values, EncoderDecoderCache):
        if is_cross:
            # Mark the cross cache populated so HF's decoder doesn't try to write it
            # itself (we recompute cross K/V here every step — caching it under compile
            # is what corrupts the context).
            past_key_values.is_updated[self.layer_idx] = True
            cache = past_key_values.cross_attention_cache
        else:
            cache = past_key_values.self_attention_cache
    elif past_key_values is not None:
        cache = past_key_values

    # Decoder self-attention against an alloy static cache → the alloy op (the aliased
    # cumulative_length tensor is the write-start position; the wrapper advances it once
    # per step, alloy_cache_attention takes [:1]).
    if not is_cross and cache is not None and use_alloy_cache_op(cache.layers[self.layer_idx]):
        layer = cache.layers[self.layer_idx]
        k = self.k_proj(hidden_states).view(kv_shape).transpose(1, 2).contiguous()
        v = self.v_proj(hidden_states).view(kv_shape).transpose(1, 2).contiguous()
        out = alloy_cache_attention(q, k, v, layer, layer.cumulative_length, self.scaling)
        out = out.to(hidden_states.dtype).transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return self.out_proj(out), None

    # Cross-attention (recompute K/V from the fixed encoder output, no mask), or a
    # self-attn call with no alloy cache (eager fallback — keeps its causal mask).
    current = key_value_states if is_cross else hidden_states
    k = self.k_proj(current).view(kv_shape).transpose(1, 2).contiguous()
    v = self.v_proj(current).view(kv_shape).transpose(1, 2).contiguous()
    mask = None if is_cross else attention_mask
    interface = ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation, eager_attention_forward)
    out, weights = interface(self, q.to(k.dtype), k, v, mask, dropout=0.0, scaling=self.scaling,
                             output_attentions=output_attentions, **kwargs)
    return self.out_proj(out.reshape(*input_shape, -1).contiguous()), weights


def cached_suppress_call(self, input_ids, scores):
    """Drop-in for SuppressTokensLogitsProcessor.__call__ that caches the suppress mask.
    HF rebuilds `torch.isin(arange(vocab), suppress_tokens)` — a CONSTANT — every decode
    step (~1.2 ms/token on whisper-tiny's 51865 vocab, the single biggest per-token cost).
    The mask depends only on (vocab_size, device), so cache it; the per-step cost drops to
    a vectorized `where` (~0.04 ms). Bit-identical output."""
    mask = self.__dict__.get("alloy_suppress_mask")
    if mask is None or mask.shape[-1] != scores.shape[-1] or mask.device != scores.device:
        vocab = torch.arange(scores.shape[-1], device=scores.device)
        mask = torch.isin(vocab, self.suppress_tokens.to(scores.device))
        self.alloy_suppress_mask = mask
    return torch.where(mask, -float("inf"), scores)


def install_fast_logits_processors() -> None:
    """Patch SuppressTokensLogitsProcessor to cache its constant mask (see
    cached_suppress_call). Idempotent; whisper's generate builds these processors
    internally, so patching the class is the least-surface fix."""
    if SuppressTokensLogitsProcessor.__call__ is not cached_suppress_call:
        SuppressTokensLogitsProcessor.__call__ = cached_suppress_call


class WhisperNextToken(torch.nn.Module):
    """One whisper decode step, entirely on GPU: decoder → proj_out → + suppress mask →
    sample_categorical. Returns (token (1,1) i64, logits) — the token never leaves the
    GPU, so the chunked decode plan can feed it back GPU-side and amortize the per-step
    command-buffer commit/wait across a chunk (the LLM GreedyNextToken pattern).

    Calls the decoder's RAW class forward (`type(decoder).forward`) to bypass the
    instance's compiled_decode wrapper (used by the HF-generate path) while keeping the
    instance-level alloy attention patches; advances the shared cumulative_length here as
    an AOT input-mutation (folded into the chunk's GPU feedback). `suppress_mask` is an
    additive (1,1,V) constant (0 / -inf) — bit-exact to HF's where(mask, -inf, scores)."""

    def __init__(self, decoder: torch.nn.Module, proj_out: torch.nn.Module) -> None:
        super().__init__()
        self.decoder = decoder
        self.proj_out = proj_out

    def forward(self, input_ids, past_key_values, cache_position, encoder_hidden_states,
                suppress_mask, seed, params):
        out = type(self.decoder).forward(
            self.decoder, input_ids=input_ids, encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values, use_cache=True, cache_position=cache_position,
        )
        past_key_values.self_attention_cache.layers[0].cumulative_length.add_(input_ids.shape[1])
        last = self.proj_out(out.last_hidden_state)[:, -1:, :] + suppress_mask
        token = torch.ops.alloy.sample_categorical(last, cache_position, seed, params)
        return token, last


class WhisperNextTokenTimestamped(torch.nn.Module):
    """WhisperNextToken + the WhisperTimeStampLogitsProcessor expressed as pure on-GPU
    tensor masking (no Python branches), so the timestamp path chunks too. The stateful
    pairing/non-decreasing rules ride on three (1,) buffers carried as in-graph mutations
    — penult_in (the token before last), last_ts_in (the most recent timestamp value, or
    TB-1 if none), num_dec_in (count of decoded tokens) — which PlanStore.decode_chunk
    propagates GPU-side between chunk iterations. Bit-exact to HF (validated)."""

    def __init__(self, decoder, proj_out, vocab_size: int, timestamp_begin: int, eos: int) -> None:
        super().__init__()
        self.decoder = decoder
        self.proj_out = proj_out
        self.tb = timestamp_begin
        self.eos = eos
        self.register_buffer("idx", torch.arange(vocab_size), persistent=False)

    def forward(self, input_ids, penult_in, last_ts_in, num_dec_in, past_key_values,
                cache_position, encoder_hidden_states, suppress_mask, seed, params):
        out = type(self.decoder).forward(
            self.decoder, input_ids=input_ids, encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values, use_cache=True, cache_position=cache_position,
        )
        past_key_values.self_attention_cache.layers[0].cumulative_length.add_(input_ids.shape[1])
        tb, eos, idx, neg = self.tb, self.eos, self.idx, float("-inf")
        s = self.proj_out(out.last_hidden_state)[:, -1:, :] + suppress_mask  # suppress incl. <|notimestamps|>
        last = input_ids.reshape(-1)
        last_ts = last >= tb
        # pairing rules (De Morgan avoids bitwise_not): penult_ts = num<2 OR penult>=tb
        cond_a = last_ts & ((num_dec_in < 2) | (penult_in >= tb))       # must be text → suppress timestamps
        cond_b = last_ts & (num_dec_in >= 2) & (penult_in < tb)         # must be timestamp → suppress text
        s = torch.where(cond_a & (idx >= tb), neg, s)
        s = torch.where(cond_b & (idx < eos), neg, s)
        # non-decreasing: timestamps below the last emitted one are forbidden
        ts_last = last_ts_in + torch.where(cond_b, 0, 1)
        s = torch.where((last_ts_in >= tb) & (idx >= tb) & (idx < ts_last), neg, s)
        # detect-from-logprob: log_softmax denom cancels → logsumexp(ts) vs amax(text)
        force = (torch.logsumexp(s[..., tb:].float(), dim=-1) > torch.amax(s[..., :tb].float(), dim=-1))
        s = torch.where(force.unsqueeze(-1) & (idx < tb), neg, s)
        token = torch.ops.alloy.sample_categorical(s, cache_position, seed, params)
        nt = token.reshape(-1)
        penult_in.copy_(last)
        last_ts_in.copy_(torch.where(nt >= tb, nt, last_ts_in))
        num_dec_in.add_(1)
        return token, s


class WhisperChunkedDecoder:
    """Alloy-owned chunked decode for the no-timestamp greedy path. Amortizes the
    per-step command-buffer commit/wait across a chunk of decode steps via
    PlanStore.decode_chunk (GPU-side token feedback, zero Python per token inside a
    chunk) — ~1.08 ms/token vs ~3.5 with HF's per-token loop. The cumulative_length.add_
    folds into the GPU feedback; enc_hidden + the suppress mask ride as stable inputs.
    Timestamps and sampling stay on the per-token path (the timestamp processor is
    stateful per-token; baking it on-GPU is the remaining M8 work)."""

    def __init__(self, model, dtype: torch.dtype, chunks: tuple[int, ...] = (8,)) -> None:
        cfg = model.config
        gc = model.generation_config
        self.decoder = model.model.decoder
        self.proj = model.proj_out
        self.eos = gc.eos_token_id
        self.chunks = chunks
        self.cache_len = cfg.max_target_positions
        self.module = torch.compile(WhisperNextToken(self.decoder, self.proj),
                                    backend="alloy", dynamic=False)
        self.plans = PlanStore(hidden_size=cfg.d_model, grid_shrink=False)
        self.enc = torch.zeros(1, cfg.max_source_positions, cfg.d_model, dtype=dtype)
        torch._dynamo.mark_static_address(self.enc)
        # additive suppress masks (0 / -inf) — bit-exact to HF's where(mask, -inf, scores).
        # begin = suppress ∪ begin_suppress, applied only to the first decoded token.
        base = torch.zeros(1, 1, cfg.vocab_size)
        base[0, 0, list(gc.suppress_tokens)] = float("-inf")
        self.suppress = torch.zeros(1, 1, cfg.vocab_size)
        torch._dynamo.mark_static_address(self.suppress)
        self.suppress.copy_(base)
        self.begin = base.clone()
        self.begin[0, 0, list(gc.begin_suppress_tokens)] = float("-inf")
        self.replay = None

    def cumulative(self, cache):
        return cache.self_attention_cache.layers[0].cumulative_length

    def decode(self, cache, enc_hidden, prefill_logits, max_new_tokens):
        """Yield generated token ids (greedy, suppress-only) given a primed cache and the
        prefix prefill's last-position logits. Steps 1-2 run the compiled module (step 2
        captures the replay); the steady state cascades through the chunked plans."""
        self.enc.copy_(enc_hidden)
        plans = self.plans
        first = int(torch.argmax((prefill_logits + self.begin)[0, -1]).item())
        yield first
        if first == self.eos:
            return
        plans.token_input.copy_(torch.tensor([[first]]))

        def mod():
            plans.cache_position[0] = self.cumulative(cache).item()
            return self.module(plans.token_input, cache, plans.cache_position, self.enc,
                               self.suppress, plans.seed, plans.params)

        step = 1
        while step <= 2 and step < max_new_tokens:
            if step == 2 and self.replay is None:
                with capture_plan() as slot:
                    tok, _ = mod()
                self.replay = plans.capture_decode_replay(slot, self.cache_len)
            else:
                tok, _ = mod()
            ti = int(tok.reshape(-1)[0])
            yield ti
            if ti == self.eos:
                return
            plans.token_input.copy_(tok.reshape(1, 1))
            step += 1

        chunk_states = []
        if self.replay is not None:
            for size in sorted({c for c in self.chunks if c > 1}, reverse=True):
                state = plans.decode_chunk(self.cache_len, size)
                if state is not None:
                    chunk_states.append((size, state))
        updates = self.replay[0]._cached_input_updates if self.replay is not None else None
        while step < max_new_tokens:
            pick = next((cs for cs in chunk_states if max_new_tokens - step >= cs[0]), None)
            if pick is not None:
                size, (handle, gen_tokens, _) = pick
                _metal_ext.dispatch_plan(handle, updates)
                for ti in gen_tokens.tolist():
                    yield ti
                    if ti == self.eos:
                        return
                step += size
            else:
                tok, _ = mod()
                ti = int(tok.reshape(-1)[0])
                yield ti
                if ti == self.eos:
                    return
                plans.token_input.copy_(tok.reshape(1, 1))
                step += 1


class WhisperTimestampDecoder:
    """Alloy-owned chunked decode for the TIMESTAMP path — WhisperNextTokenTimestamped
    (the on-GPU timestamp processor) + three state buffers (penult, last_ts, num_dec)
    carried through PlanStore.decode_chunk's GPU feedback. Same ~1.08 ms/token as the
    no-timestamp path; bit-exact to HF generate(return_timestamps=True). The first token
    forces an initial timestamp (the max_initial_timestamp rule, computed once on CPU)."""

    def __init__(self, model, dtype: torch.dtype, chunks: tuple[int, ...] = (8,)) -> None:
        cfg = model.config
        gc = model.generation_config
        self.decoder = model.model.decoder
        self.proj = model.proj_out
        self.eos = gc.eos_token_id
        self.tb = gc.no_timestamps_token_id + 1
        self.max_initial = gc.to_dict().get("max_initial_timestamp_index")
        self.chunks = chunks
        self.cache_len = cfg.max_target_positions
        self.module = torch.compile(
            WhisperNextTokenTimestamped(self.decoder, self.proj, cfg.vocab_size, self.tb, self.eos),
            backend="alloy", dynamic=False)
        self.plans = PlanStore(hidden_size=cfg.d_model, grid_shrink=False)
        self.enc = torch.zeros(1, cfg.max_source_positions, cfg.d_model, dtype=dtype)
        torch._dynamo.mark_static_address(self.enc)
        self.suppress = torch.zeros(1, 1, cfg.vocab_size)
        self.suppress[0, 0, list(gc.suppress_tokens)] = float("-inf")
        self.suppress[0, 0, gc.no_timestamps_token_id] = float("-inf")
        torch._dynamo.mark_static_address(self.suppress)
        self.penult = torch.zeros(1, dtype=torch.long)
        self.last_ts = torch.zeros(1, dtype=torch.long)
        self.ndec = torch.zeros(1, dtype=torch.long)
        for buf in (self.penult, self.last_ts, self.ndec):
            torch._dynamo.mark_static_address(buf)
        self.replay = None

    def cumulative(self, cache):
        return cache.self_attention_cache.layers[0].cumulative_length

    def decode(self, cache, enc_hidden, prefill_logits, max_new_tokens):
        """Yield generated token ids (timestamps + text) given a primed cache and the
        prefix prefill's last-position logits."""
        self.enc.copy_(enc_hidden)
        plans, tb = self.plans, self.tb
        # first token: force an initial timestamp in [tb, tb+max_initial] (bit-exact to
        # HF's first-token processors — text is fully suppressed there too).
        first_scores = prefill_logits[0, -1].clone()
        first_scores[:tb] = float("-inf")
        if self.max_initial is not None:
            first_scores[tb + self.max_initial + 1:] = float("-inf")
        first = int(first_scores.argmax())
        yield first
        if first == self.eos:
            return
        self.penult.zero_()
        self.last_ts.fill_(first if first >= tb else tb - 1)
        self.ndec.fill_(1)
        plans.token_input.copy_(torch.tensor([[first]]))

        def mod():
            plans.cache_position[0] = self.cumulative(cache).item()
            return self.module(plans.token_input, self.penult, self.last_ts, self.ndec, cache,
                               plans.cache_position, self.enc, self.suppress, plans.seed, plans.params)

        step = 1
        while step <= 2 and step < max_new_tokens:
            if step == 2 and self.replay is None:
                with capture_plan() as slot:
                    tok, _ = mod()
                self.replay = plans.capture_decode_replay(slot, self.cache_len)
            else:
                tok, _ = mod()
            ti = int(tok.reshape(-1)[0])
            yield ti
            if ti == self.eos:
                return
            plans.token_input.copy_(tok.reshape(1, 1))
            step += 1

        chunk_states = []
        if self.replay is not None:
            for size in sorted({c for c in self.chunks if c > 1}, reverse=True):
                state = plans.decode_chunk(self.cache_len, size)
                if state is not None:
                    chunk_states.append((size, state))
        updates = self.replay[0]._cached_input_updates if self.replay is not None else None
        while step < max_new_tokens:
            pick = next((cs for cs in chunk_states if max_new_tokens - step >= cs[0]), None)
            if pick is not None:
                size, (handle, gen_tokens, _) = pick
                _metal_ext.dispatch_plan(handle, updates)
                for ti in gen_tokens.tolist():
                    yield ti
                    if ti == self.eos:
                        return
                step += size
            else:
                tok, _ = mod()
                ti = int(tok.reshape(-1)[0])
                yield ti
                if ti == self.eos:
                    return
                plans.token_input.copy_(tok.reshape(1, 1))
                step += 1


def install_whisper_decoder_compat(model, dtype: torch.dtype):
    """Patch the decoder for alloy and return its (uncompiled) forward. Self/cross
    attention route through alloy's op (the int causal mask can't trace); the final
    layer_norm stays f32-safe (the @capture_outputs graph break drops it to an eager
    tail); proj_out's input is cast to the weight dtype (it runs eager outside the
    compiled decoder). Shared by serving (caller then compiles `dec_forward`) and the
    offline tuner (caller leaves it eager so `alloy.tune` compiles + captures it)."""
    for layer in model.model.decoder.layers:
        layer.self_attn.forward = types.MethodType(alloy_whisper_attention, layer.self_attn)
        layer.encoder_attn.forward = types.MethodType(alloy_whisper_attention, layer.encoder_attn)
    ln, ln_forward = model.model.decoder.layer_norm, model.model.decoder.layer_norm.forward
    ln.forward = lambda x: ln_forward(x.float()).to(dtype)
    proj_forward, proj_dtype = model.proj_out.forward, model.proj_out.weight.dtype
    model.proj_out.forward = lambda x: proj_forward(x.to(proj_dtype))
    return model.model.decoder.forward


class WhisperTranscriber:
    """Loads a Whisper GGUF, compiles BOTH halves (encoder + decoder) on alloy at the
    model's native precision, eager-compiles every production plan at construction (no
    first-request tax), and transcribes audio. Holds ONE persistent KV cache reused
    across requests (the LLM ContiguousKV model) so the pinned plans never re-bind."""

    def __init__(self, gguf_path: str, tokenizer_ref: str = CANONICAL_TOKENIZER,
                 dtype: torch.dtype = torch.float32) -> None:
        self.dtype = dtype
        install_fast_logits_processors()
        self.processor = WhisperProcessor.from_pretrained(tokenizer_ref, local_files_only=True)
        model, missing, unexpected = build_whisper_eager(gguf_path, dtype)
        if unexpected:
            raise RuntimeError(f"unexpected GGUF tensors not in the Whisper model: {unexpected}")
        model.generation_config = GenerationConfig.from_pretrained(tokenizer_ref, local_files_only=True)

        # Encoder: compiled on alloy, returns a BaseModelOutput cast to the model dtype.
        original_encoder = model.model.encoder
        compiled_enc = torch.compile(EncoderForward(original_encoder), backend="alloy", dynamic=False)
        model.model.encoder = GenerateEncoder(compiled_enc, original_encoder, dtype)

        # Decoder: route self/cross attn through the alloy op + f32-safe tail, then
        # compile the decoder module (the tuner shares the patch but leaves it eager).
        decoder = model.model.decoder
        dec_forward = install_whisper_decoder_compat(model, dtype)

        def compiled_decode(input_ids=None, **kw):
            out = dec_forward(input_ids=input_ids, **kw)
            pkv = kw.get("past_key_values")
            if pkv is not None and input_ids is not None:
                # one in-graph advance of the shared self-attn position (op skips it)
                pkv.self_attention_cache.layers[0].cumulative_length.add_(input_ids.shape[1])
            out.last_hidden_state = out.last_hidden_state.to(dtype)
            return out

        decoder.forward = torch.compile(compiled_decode, backend="alloy", dynamic=False)
        self.model = model

        # ONE persistent KV cache, reused across every request and reset between them
        # (the LLM ContiguousKV model). Its buffers are eager-allocated + mark_static_
        # address'd, so the pinned decoder plan binds them once — a fresh cache per
        # request would change the static addresses and force a Dynamo recompile every
        # call. eager_compile_all then compiles every production plan against it.
        self.kv = self.build_cache()
        self.chunked: WhisperChunkedDecoder | None = None  # alloy-owned chunked decode (lazy)
        self.ts_decoder: WhisperTimestampDecoder | None = None  # alloy-owned timestamp decode (lazy)
        self.eager_compile_all()

    def build_cache(self) -> EncoderDecoderCache:
        """The decoder self-attention cache (alloy op path: AlloyStaticCache, eager-
        allocated) + a StaticCache placeholder for cross-attn (recomputed each step,
        never written)."""
        cfg = self.model.config
        self_cache = AlloyStaticCache(config=cfg, max_cache_len=cfg.max_target_positions, cache_dtype=self.dtype)
        cross = StaticCache(config=cfg, max_cache_len=cfg.max_source_positions)
        return EncoderDecoderCache(self_cache, cross)

    def reset_kv(self) -> None:
        """Return the persistent cache to a fresh state between requests: rewind the
        shared self-attn position and drop the cross-cache populated flags. The K/V
        buffers themselves need no zeroing — the op only attends [0, cumulative_length),
        so stale rows past the write head are never read."""
        for layer in self.kv.self_attention_cache.layers:
            layer.cumulative_length.zero_()
        self.kv.is_updated.clear()

    def eager_compile_all(self) -> None:
        """Compile every plan the production path dispatches — encoder + each decoder
        graph specialization (language probe, forced-prefix prefill, per-step decode) —
        before the first request, so no production call ever pays torch.compile cost
        inline (the LLM eager_compile contract).

        Warms the exact alloy-owned forwards a request dispatches (no HF generate): the
        encoder, language-detect ([SOT] M=1), forced-prefix prefill (M=3/M=4), and both
        chunked decode paths (timestamped + plain) — module compile + replay capture +
        chunk-plan registration — on a silent 30s frame under `no_grad`."""
        self.reset_kv()
        feats = self.mel(torch.zeros(SAMPLE_RATE * 30, dtype=torch.float32))
        enc = self.encode(feats)
        self.detect_language_from_enc(enc)
        self.decode_window(enc, "en", "transcribe")  # prefill M=3 + timestamp chunked decode
        chunked = self.ensure_chunked()
        prefix = self.forced_prefix("en", "transcribe")  # M=4 prefill + plain chunked decode
        self.reset_kv()
        with torch.no_grad():
            out = self.model.model.decoder(
                input_ids=torch.tensor([prefix]), encoder_hidden_states=enc,
                past_key_values=self.kv, use_cache=True, cache_position=torch.arange(len(prefix)))
            logits = self.model.proj_out(out.last_hidden_state)[:, -1:, :]
            list(chunked.decode(self.kv, enc, logits, 32))
        self.reset_kv()

    def mel(self, waveform) -> torch.Tensor:
        return self.processor.feature_extractor(
            waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt",
        ).input_features

    def ensure_chunked(self) -> WhisperChunkedDecoder:
        if self.chunked is None:
            self.chunked = WhisperChunkedDecoder(self.model, self.dtype)
        return self.chunked

    def forced_prefix(self, language: str, task: str) -> list[int]:
        """The forced decoder prompt for the no-timestamp path: SOT, language, task,
        notimestamps (HF builds the same prefix internally)."""
        tok = self.processor.tokenizer
        return [tok.convert_tokens_to_ids(t) for t in
                ("<|startoftranscript|>", f"<|{language}|>", f"<|{task}|>", "<|notimestamps|>")]

    def transcribe_text_fast(self, waveform, language: str | None, task: str) -> tuple[str, str]:
        """No-timestamp chunked decode of one ≤30s window → (text, language). Detects the
        language (one [SOT] forward) when None. ~mlx-parity decode (~1.08 ms/token); the
        model emits no timestamp tokens, so fewer tokens than the timestamped path."""
        chunked = self.ensure_chunked()
        enc = self.encode(self.mel(waveform))
        if language is None:
            language = self.detect_language_from_enc(enc)
        prefix = self.forced_prefix(language, task)
        self.reset_kv()
        with torch.no_grad():
            out = self.model.model.decoder(
                input_ids=torch.tensor([prefix]), encoder_hidden_states=enc,
                past_key_values=self.kv, use_cache=True, cache_position=torch.arange(len(prefix)))
            logits = self.model.proj_out(out.last_hidden_state)[:, -1:, :]
            budget = self.model.config.max_target_positions - len(prefix)
            ids = list(chunked.decode(self.kv, enc, logits, budget))
        return self.processor.batch_decode([prefix + ids], skip_special_tokens=True)[0].strip(), language

    def transcribe_fast(self, audio: bytes | str, *, language: str | None = "en",
                        task: str = "transcribe") -> TranscriptionResult:
        """No-timestamp greedy transcription (≤30s) via the alloy-owned chunked decode —
        text only, no segments. For >30s or segments, use `transcribe`."""
        waveform = load_audio(audio, SAMPLE_RATE)
        text, lang = self.transcribe_text_fast(waveform, language, task)
        return TranscriptionResult(text=text, language=lang,
                                   duration=float(len(waveform)) / SAMPLE_RATE, segments=(), words=())

    def ensure_timestamp_decoder(self) -> WhisperTimestampDecoder:
        if self.ts_decoder is None:
            self.ts_decoder = WhisperTimestampDecoder(self.model, self.dtype)
        return self.ts_decoder

    def encode(self, feats: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model.model.encoder(feats).last_hidden_state

    def detect_language_from_enc(self, enc: torch.Tensor) -> str:
        """Whisper language detection: one [SOT] forward, argmax over the language token
        ids only. Alloy-owned (no HF generate)."""
        lang_to_id = self.model.generation_config.to_dict().get("lang_to_id") or {}
        sot = self.processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        self.reset_kv()
        with torch.no_grad():
            out = self.model.model.decoder(
                input_ids=torch.tensor([[sot]]), encoder_hidden_states=enc,
                past_key_values=self.kv, use_cache=True, cache_position=torch.arange(1))
            logits = self.model.proj_out(out.last_hidden_state)[0, -1].float()
        mask = torch.full_like(logits, float("-inf"))
        mask[list(lang_to_id.values())] = 0.0
        lang_id = int((logits + mask).argmax())
        return {v: k for k, v in lang_to_id.items()}[lang_id][2:-2]

    def extract_segments(self, tokens: list[int], time_offset: float, final: bool
                         ) -> list[tuple[float, float, list[int]]]:
        """Whisper consecutive-timestamp-pair segmentation (bit-exact to HF). Returns
        (start, end, content_tokens) per segment; timestamps decode at 0.02 s. `final`
        (the decode hit EOS / last window) also emits the trailing segment after the last
        consecutive boundary — which has no closing consecutive pair, just <…|tn|> EOS."""
        tb = self.ensure_timestamp_decoder().tb
        is_ts = [tok >= tb for tok in tokens]
        consecutive = [i + 1 for i in range(len(tokens) - 1) if is_ts[i] and is_ts[i + 1]]
        segs: list[tuple[float, float, list[int]]] = []

        def seg(sliced: list[int]) -> None:
            ts = [x for x in sliced if x >= tb]
            if ts:
                segs.append(((sliced[0] - tb) * 0.02 + time_offset if sliced[0] >= tb else (ts[0] - tb) * 0.02 + time_offset,
                             (ts[-1] - tb) * 0.02 + time_offset, [x for x in sliced if x < tb]))

        if consecutive:
            last = 0
            for cut in consecutive:
                seg(tokens[last:cut])
                last = cut
            if final:
                seg(tokens[last:])
        elif any(is_ts):
            seg(tokens)
        return segs

    def decode_window(self, enc: torch.Tensor, language: str, task: str) -> tuple[list[int], bool]:
        """Prefill [SOT, lang, task] and run the alloy-owned chunked timestamp decode for
        one 30 s window. Returns (token ids EOS-stripped, ended) where ended = EOS hit."""
        tsd = self.ensure_timestamp_decoder()
        tok = self.processor.tokenizer
        prefix = [tok.convert_tokens_to_ids(t) for t in
                  ("<|startoftranscript|>", f"<|{language}|>", f"<|{task}|>")]
        self.reset_kv()
        with torch.no_grad():
            out = self.model.model.decoder(
                input_ids=torch.tensor([prefix]), encoder_hidden_states=enc,
                past_key_values=self.kv, use_cache=True, cache_position=torch.arange(len(prefix)))
            logits = self.model.proj_out(out.last_hidden_state)[:, -1:, :]
            budget = self.model.config.max_target_positions - len(prefix)
            ids = list(tsd.decode(self.kv, enc, logits, budget))
        ended = bool(ids) and ids[-1] == tsd.eos
        return (ids[:-1] if ended else ids), ended

    def transcribe(
        self, audio: bytes | str, *, task: str = "transcribe",
        language: str | None = None, prompt: str | None = None,
        temperature: float = 0.0,
    ) -> TranscriptionResult:
        """Alloy-owned transcription (no HF generate): encoder → [language detect] →
        chunked timestamp decode per 30 s window → segments. >30 s seeks window-to-window
        on the last complete segment's end timestamp; the final window emits its trailing
        segment. (For text-only ≤30s with no segments, `transcribe_fast` is faster but
        may pick a different word on an ambiguous token — kept separate for consistency
        with the segmented text here.)"""
        waveform = load_audio(audio, SAMPLE_RATE)
        duration = float(len(waveform)) / SAMPLE_RATE
        window = SAMPLE_RATE * 30
        segments: list[Segment] = []
        content_all: list[int] = []
        seek = 0
        while seek < max(1, len(waveform)):
            is_last = seek + window >= len(waveform)
            enc = self.encode(self.mel(waveform[seek:seek + window]))
            if language is None:
                language = self.detect_language_from_enc(enc)
            offset = float(seek) / SAMPLE_RATE
            ids, _ = self.decode_window(enc, language, task)
            window_segs = self.extract_segments(ids, offset, is_last)
            for start, end, content in window_segs:
                content_all += content
                segments.append(Segment(
                    id=len(segments), seek=seek, start=round(start, 2), end=round(end, 2),
                    text=self.processor.decode(content, skip_special_tokens=True).strip(),
                    tokens=tuple(content), temperature=temperature, avg_logprob=0.0,
                    compression_ratio=compression_ratio(
                        self.processor.decode(content, skip_special_tokens=True).strip()),
                    no_speech_prob=0.0))
            if is_last or not window_segs:
                break
            seek = max(int(round(window_segs[-1][1] * SAMPLE_RATE)), seek + SAMPLE_RATE)
        text = self.processor.decode(content_all, skip_special_tokens=True).strip()
        return TranscriptionResult(text=text, language=language or "en",
                                   duration=duration, segments=tuple(segments))

    def stream_transcribe(
        self, audio: bytes | str, *, task: str = "transcribe",
        language: str | None = None, prompt: str | None = None,
        temperature: float = 0.0,
    ) -> Iterator[str]:
        """Yield transcription text deltas (alloy-owned, no HF generate). The final 30s
        window streams per token (incrementally detokenized, U+FFFD held back); earlier
        long-form windows yield per completed segment, seeking past them. The generator
        runs inline on the caller's GPU worker thread; a consumer close (disconnect)
        stops it via GeneratorExit, so it never races the next request."""
        tsd = self.ensure_timestamp_decoder()
        tok = self.processor.tokenizer
        waveform = load_audio(audio, SAMPLE_RATE)
        window = SAMPLE_RATE * 30
        seek = 0
        while seek < max(1, len(waveform)):
            is_last = seek + window >= len(waveform)
            enc = self.encode(self.mel(waveform[seek:seek + window]))
            if language is None:
                language = self.detect_language_from_enc(enc)
            prefix = [tok.convert_tokens_to_ids(t) for t in
                      ("<|startoftranscript|>", f"<|{language}|>", f"<|{task}|>")]
            self.reset_kv()
            window_ids: list[int] = []
            content: list[int] = []
            prev = ""
            with torch.no_grad():
                out = self.model.model.decoder(
                    input_ids=torch.tensor([prefix]), encoder_hidden_states=enc,
                    past_key_values=self.kv, use_cache=True, cache_position=torch.arange(len(prefix)))
                logits = self.model.proj_out(out.last_hidden_state)[:, -1:, :]
                budget = self.model.config.max_target_positions - len(prefix)
                for ti in tsd.decode(self.kv, enc, logits, budget):
                    if ti == tsd.eos:
                        break
                    window_ids.append(ti)
                    if is_last and ti < tsd.tb:  # final window streams per token
                        content.append(ti)
                        text = self.processor.decode(content, skip_special_tokens=True)
                        if not text.endswith("�") and len(text) > len(prev):
                            yield text[len(prev):]
                            prev = text
            if is_last:
                break
            segs = self.extract_segments(window_ids, float(seek) / SAMPLE_RATE, False)
            for _, _, seg_content in segs:
                seg_text = self.processor.decode(seg_content, skip_special_tokens=True).strip()
                if seg_text:
                    yield seg_text + " "
            if not segs:
                break
            seek = max(int(round(segs[-1][1] * SAMPLE_RATE)), seek + SAMPLE_RATE)


def compression_ratio(text: str) -> float:
    """gzip compression ratio (whisper's repetition/hallucination signal)."""
    data = text.encode("utf-8")
    if not data:
        return 0.0
    return len(data) / len(zlib.compress(data))


def tune_whisper(gguf_path: str, *, tokenizer_ref: str = CANONICAL_TOKENIZER,
                 dtype: torch.dtype = torch.float32, only: str | None = None,
                 output: str | None = None) -> None:
    """Offline-tune the production kernels — encoder (M=1500) + decoder prefill (M=3,
    the timestamped forced prefix) + decode (M=1) — and merge into the per-device
    user config (overlaid on the shipped baseline at serve time).

    Mirrors the LLM `alloy tune`: builds the model UNCOMPILED (patched), so `alloy.tune`
    compiles + captures it at the exact shapes generation dispatches, then benchmarks
    every config per shape. Run BEFORE serving — the serve process bakes the tuned
    configs into its pinned plans at compile time, so tuning in-process is too late."""
    model, _, unexpected = build_whisper_eager(gguf_path, dtype)
    if unexpected:
        raise RuntimeError(f"unexpected GGUF tensors not in the Whisper model: {unexpected}")
    model.generation_config = GenerationConfig.from_pretrained(tokenizer_ref, local_files_only=True)
    install_whisper_decoder_compat(model, dtype)  # patched, left EAGER — alloy.tune compiles it
    cfg = model.config
    enc_len, d_model = cfg.max_source_positions, cfg.d_model
    only_kw = {"only": only} if only else {}
    if output:
        only_kw["output"] = output

    def cache(offset: int) -> EncoderDecoderCache:
        sc = AlloyStaticCache(config=cfg, max_cache_len=cfg.max_target_positions, cache_dtype=dtype)
        cr = StaticCache(config=cfg, max_cache_len=cfg.max_source_positions)
        c = EncoderDecoderCache(sc, cr)
        for layer in c.self_attention_cache.layers:
            layer.cumulative_length.fill_(offset)
        return c

    alloy.tune(EncoderForward(model.model.encoder),
               torch.zeros((1, cfg.num_mel_bins, 2 * enc_len), dtype=dtype), **only_kw)
    enc = BaseModelOutput(torch.zeros((1, enc_len, d_model), dtype=dtype))
    alloy.tune(model, {"encoder_outputs": enc, "decoder_input_ids": torch.zeros((1, 3), dtype=torch.long),
                       "past_key_values": cache(0), "use_cache": True,
                       "cache_position": torch.arange(3)}, **only_kw)
    alloy.tune(model, {"encoder_outputs": enc, "decoder_input_ids": torch.zeros((1, 1), dtype=torch.long),
                       "past_key_values": cache(3), "use_cache": True,
                       "cache_position": torch.tensor([3])}, **only_kw)


def load_whisper_gguf(gguf_path: str, name: str, tokenizer_ref: str = CANONICAL_TOKENIZER) -> TranscriptionModel:
    """Build a served TranscriptionModel from a Whisper GGUF (eager-compiled at __init__)."""
    transcriber = WhisperTranscriber(gguf_path, tokenizer_ref)
    return TranscriptionModel(
        name=name, transcribe=transcriber.transcribe,
        stream_transcribe=transcriber.stream_transcribe,
    )


@register("whisper")
class WhisperHandler:
    """Whisper speech-to-text. A custom-`load()` handler (own GGUF reader + eager
    Whisper build, not the causal-LM tensor loop); `kind="transcription"` routes
    the server to the transcription modality."""

    arch = ("whisper",)
    kind = "transcription"

    def apply_transformers_patches(self) -> None:
        return None

    def load(self, source: ResolvedGGUF) -> TranscriptionModel:
        return load_whisper_gguf(str(source.path), name=source.ref)
