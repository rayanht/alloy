"""Alloy-native DeepSeek-V4.1 forward.

One code path, `forward(state, ids)`, runs a chunk of T >= 1 tokens at the
sequence's current position: chunked prefill and decode are the same function.
Kernels are dispatched directly on alloy buffers through the lazy engine; the only
CPU round trips are the per-layer router read (which experts to make resident) and
the engram row gather.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alloy._compiler.dtypes import float16, float32, int32, uint8
from alloy._dispatch.buf_utils import _alloc_aligned
from alloy._runtime._metal_ext import gpu_sync
from alloy._runtime.alloy_buffer import AlloyBuffer, materialize_many
from alloy.std.deepseek import (
    ds_block_mask_scatter,
    ds_block_max,
    ds_block_round,
    ds_compress_chunk,
    ds_compress_tail,
    ds_copy_rows,
    ds_engram_gate,
    ds_expand_hc,
    ds_grouped_out_proj,
    ds_hc_mixes,
    ds_hc_post,
    ds_hc_pre,
    ds_hc_sinkhorn,
    ds_index_score,
    ds_rope_pairs,
    ds_router_topk,
    ds_sparse_attn,
    ds_swiglu_clamp,
    ds_topk_select,
    ds_zero_u8,
)
from alloy.std.gemm import dot_transpose_rhs
from alloy.std.moe import (
    moe_combine_rows,
    moe_down_combine,
    moe_down_combine_q4k,
    moe_down_grouped_partial,
    moe_down_grouped_partial_q4k,
    moe_gate_up_grouped,
    moe_gate_up_silu,
    moe_row_tokens,
    moe_sort_block_count,
    moe_sort_block_off,
    moe_sort_count_from_blocks,
    moe_sort_offsets,
    moe_sort_perm_scan,
    moe_tile_expert,
)
from alloy.std.norm import rms_norm
from alloy_torch.ops.linalg import Q4_K, Q6_K, quant_embedding, quant_mm
from alloy_server.models.deepseek_v41.config import DeepseekV41Config
from alloy_server.models.deepseek_v41.engram import EngramHasher
from alloy_server.models.deepseek_v41.experts import EngramTables, ExpertStore
from alloy_server.models.deepseek_v41.reference import precompute_freqs_cis
from alloy_server.models.deepseek_v41.weights import ModelWeights, QuantMatrix

SCORE_FUNCS = {"softmax": 0, "sigmoid": 1, "sqrtsoftplus": 2}


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def next_pow2(n: int) -> int:
    b = 1
    while b < n:
        b *= 2
    return b


def idx_bits(n: int) -> int:
    b = 1
    while (1 << b) < n:
        b += 1
    return b + 1


# Index-score / candidate buffers are allocated at a bucketed key capacity so their
# shapes (and the kernels' constexprs) only change every KEY_BUCKET keys — a decode
# step otherwise recompiles the indexer kernels at every position.
KEY_BUCKET = 1024


def key_capacity(n_keys: int) -> int:
    return max(KEY_BUCKET, cdiv(n_keys, KEY_BUCKET) * KEY_BUCKET)


def i32_buffer(arr: np.ndarray) -> AlloyBuffer:
    arr = np.ascontiguousarray(arr, dtype=np.int32)
    buf = _alloc_aligned(tuple(arr.shape), int32)
    buf.numpy[:] = arr
    return buf


def u8_buffer(arr: np.ndarray) -> AlloyBuffer:
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    buf = _alloc_aligned(tuple(arr.shape), uint8)
    buf.numpy[:] = arr
    return buf


def f32_buffer(arr: np.ndarray) -> AlloyBuffer:
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    buf = _alloc_aligned(tuple(arr.shape), float32)
    buf.numpy[:] = arr
    return buf


def read(buf: AlloyBuffer) -> np.ndarray:
    """Materialize + wait, then copy out."""
    buf.sync()
    gpu_sync()
    return np.array(buf.numpy)


def fmt_for(w: QuantMatrix):
    return Q6_K if w.qtype.name == "Q6_K" else Q4_K


def mm(x: AlloyBuffer, w: QuantMatrix) -> AlloyBuffer:
    """x (M, K) f32 @ w^T -> (M, N) f32 through the production quant GEMM dispatch."""
    return quant_mm(fmt_for(w), x, w.blk)


def norm(x: AlloyBuffer, weight: AlloyBuffer, eps: float) -> AlloyBuffer:
    m, n = x.shape
    out = _alloc_aligned((m, n), float32)
    rrms = _alloc_aligned((m,), float32)
    rms_norm[(m,)](x, weight, out, rrms, EPS=eps)
    return out


@dataclass
class RopeTable:
    cos: AlloyBuffer
    sin: AlloyBuffer


class CompressState:
    """Pending partial-group rows of a ratio-r compressor, double-buffered."""

    def __init__(self, ratio: int, dim: int) -> None:
        self.bufs = [
            (_alloc_aligned((ratio, dim), float32), _alloc_aligned((ratio, dim), float32)) for _ in range(2)
        ]
        self.cur = 0

    @property
    def current(self) -> tuple[AlloyBuffer, AlloyBuffer]:
        return self.bufs[self.cur]

    @property
    def other(self) -> tuple[AlloyBuffer, AlloyBuffer]:
        return self.bufs[1 - self.cur]

    def swap(self) -> None:
        self.cur = 1 - self.cur


class SequenceState:
    """One sequence's caches: per-layer sliding windows, per-KV-source compressed
    latents + index keys, compressor partial groups, engram history.

    A persistent buffer is never both read and written within one forward: the
    lazy engine keys dependencies on a buffer's latest producer, so a read
    followed by a write of the same buffer would order the write first. The window
    is therefore a fresh per-forward buffer (the previous history rows followed by
    the chunk's rows) whose last rows are the next forward's history."""

    def __init__(self, cfg: DeepseekV41Config, max_seq_len: int) -> None:
        self.cfg = cfg
        self.max_seq_len = max_seq_len
        self.pos = 0
        # per layer: the last forward's window source (rows in position order)
        self.window: list[AlloyBuffer | None] = [None] * cfg.n_layers
        self.comp_kv: dict[int, AlloyBuffer] = {}
        self.index_k: dict[int, AlloyBuffer] = {}
        self.comp_state: dict[int, CompressState] = {}
        for layer in cfg.kv_source_layers:
            ratio = cfg.compress_ratios[layer]
            groups = max(1, max_seq_len // ratio)
            self.comp_kv[layer] = _alloc_aligned((groups, cfg.head_dim), float16)
            self.index_k[layer] = _alloc_aligned((groups, cfg.index_head_dim), float16)
            if ratio > 1:
                self.comp_state[layer] = CompressState(ratio, cfg.head_dim)
        self.hasher = EngramHasher(cfg.engram, max_seq_len) if cfg.engram is not None else None
        # shared within one forward: the latest top-k selection and candidate mask
        self.comp_idx: AlloyBuffer | None = None
        self.candidates: AlloyBuffer | None = None
        # writes only a later forward reads (the compressor's pending rows) — unreachable
        # from this forward's outputs, so the lazy engine would drop them; flushed at
        # every sync point
        self.pending: list[AlloyBuffer] = []

    def flush(self, *extra: AlloyBuffer) -> None:
        materialize_many(list(extra) + self.pending)
        self.pending.clear()


class DeepseekV41Model:
    def __init__(
        self,
        cfg: DeepseekV41Config,
        weights: ModelWeights,
        experts: ExpertStore,
        engram: EngramTables,
        max_seq_len: int,
        *,
        rounding: bool = True,
    ) -> None:
        self.cfg = cfg
        self.w = weights
        self.rounding = rounding
        self.experts = experts
        self.engram = engram
        self.max_seq_len = max_seq_len
        self.rope: dict[tuple[float, int], RopeTable] = {}
        for layer in range(cfg.n_layers):
            key = cfg.rope_params(layer)
            if key not in self.rope:
                theta, orig = key
                freqs = precompute_freqs_cis(
                    cfg.rope_head_dim, max_seq_len, orig, theta, cfg.rope_factor, cfg.beta_fast, cfg.beta_slow,
                )
                self.rope[key] = RopeTable(
                    cos=f32_buffer(freqs.real.numpy()), sin=f32_buffer(freqs.imag.numpy()),
                )
        self.debug_index: dict = {}
        self.dummy_f16 = _alloc_aligned((1, cfg.head_dim), float16)
        self.dummy_i32 = _alloc_aligned((1,), int32)
        self.dummy_u8 = _alloc_aligned((1,), uint8)
        self.zero_i32 = i32_buffer(np.zeros(1, dtype=np.int32))

    def new_state(self) -> SequenceState:
        return SequenceState(self.cfg, self.max_seq_len)

    # --- building blocks -------------------------------------------------------

    def apply_rope(self, x: AlloyBuffer, pos: AlloyBuffer, layer: int, heads: int, head_dim: int,
                   inverse: bool = False) -> AlloyBuffer:
        rows = x.shape[0]
        table = self.rope[self.cfg.rope_params(layer)]
        out = _alloc_aligned((rows, heads * head_dim), float32)
        rot = self.cfg.rope_head_dim
        ds_rope_pairs[(rows, heads)](
            x, table.cos, table.sin, pos, out,
            WIDTH=heads * head_dim, HEAD_DIM=head_dim, ROT=rot, INVERSE=1 if inverse else 0, NUM_THREADS=rot // 2,
        )
        return out

    def hc_mixes(self, x: AlloyBuffer, layer: int, kind: str) -> tuple[AlloyBuffer, AlloyBuffer, AlloyBuffer]:
        cfg = self.cfg
        L = self.w.layers[layer]
        fn, base, scale = (
            (L.hc_attn_fn, L.hc_attn_base, L.hc_attn_scale) if kind == "attn" else (L.hc_ffn_fn, L.hc_ffn_base, L.hc_ffn_scale)
        )
        T = x.shape[0]
        hc, d = cfg.hc_mult, cfg.dim
        mixes = _alloc_aligned((T, cfg.hc_mix_dim), float32)
        ds_hc_mixes[(T,)](x, fn, mixes, HC_D=hc * d, MIX=cfg.hc_mix_dim, EPS=cfg.norm_eps)
        pre = _alloc_aligned((T, hc), float32)
        post = _alloc_aligned((T, hc), float32)
        comb = _alloc_aligned((T, hc, hc), float32)
        ds_hc_sinkhorn[(T,)](mixes, scale, base, pre, post, comb, HC=hc, ITERS=cfg.hc_sinkhorn_iters, EPS=cfg.hc_eps)
        return pre, post, comb

    def hc_pre(self, x: AlloyBuffer, pre: AlloyBuffer) -> AlloyBuffer:
        T = x.shape[0]
        out = _alloc_aligned((T, self.cfg.dim), float32)
        ds_hc_pre[(T,)](x, pre, out, HC=self.cfg.hc_mult, D=self.cfg.dim)
        return out

    def hc_post(self, y: AlloyBuffer, residual: AlloyBuffer, post: AlloyBuffer, comb: AlloyBuffer) -> AlloyBuffer:
        T = y.shape[0]
        hc, d = self.cfg.hc_mult, self.cfg.dim
        out = _alloc_aligned((T, hc, d), float32)
        ds_hc_post[(T, hc)](y, residual, post, comb, out, HC=hc, D=d)
        return out

    def engram_layer(self, layer: int, x: AlloyBuffer, hash_ids: np.ndarray) -> AlloyBuffer:
        cfg = self.cfg
        L = self.w.layers[layer]
        T = x.shape[0]
        cols = cfg.engram.n_hash_cols
        rows = u8_buffer(self.engram.gather(layer, hash_ids))
        ids = i32_buffer(np.arange(T * cols, dtype=np.int32))
        emb = quant_embedding(Q4_K, ids, rows).reshape((T, cols * cfg.engram.head_dim))
        kv = mm(emb, L.engram_wkv)
        out = _alloc_aligned((T, cfg.hc_mult, cfg.dim), float32)
        ds_engram_gate[(T, cfg.hc_mult)](x, kv, L.engram_qk, out, HC=cfg.hc_mult, D=cfg.dim, EPS=cfg.norm_eps)
        return out

    def round_into(self, x: AlloyBuffer, cache: AlloyBuffer, row0: int, mode: int) -> None:
        """fp8/fp4-round `x` (rows, N) into cache rows [row0, row0 + rows)."""
        rows, n = x.shape
        block = 16 if mode == 2 else 32
        if not self.rounding:
            mode = 3
        row0_buf = i32_buffer(np.array([row0], dtype=np.int32))
        ds_block_round[(rows,)](x, cache.slice(0, 0, 1), row0_buf, cache, N=n, MODE=mode, BLOCK=block, OUT_F16=1)

    # --- attention ----------------------------------------------------------------

    def compress(self, layer: int, x: AlloyBuffer, state: SequenceState, start_pos: int, T: int) -> tuple[AlloyBuffer | None, int, int]:
        """This chunk's new compressed latents (normed, pre-RoPE): (latents, g0, NG)."""
        cfg = self.cfg
        L = self.w.layers[layer]
        ratio = cfg.compress_ratios[layer]
        kv_c = mm(x, L.compressor_kv)
        if ratio == 1:
            return norm(kv_c, L.compressor_norm, cfg.norm_eps), start_pos, T
        r = start_pos % ratio
        ng = (r + T) // ratio
        rem = r + T - ng * ratio
        g0 = (start_pos - r) // ratio
        score_c = mm(x, L.compressor_gate)
        cs = state.comp_state[layer]
        st_kv, st_score = cs.current
        latents = None
        if ng > 0:
            pooled = _alloc_aligned((ng, cfg.head_dim), float32)
            ds_compress_chunk[(ng,)](st_kv, st_score, kv_c, score_c, pooled, R=r, RATIO=ratio, D=cfg.head_dim)
            latents = norm(pooled, L.compressor_norm, cfg.norm_eps)
        if rem > 0:
            new_kv, new_score = cs.other
            ds_compress_tail[(rem,)](st_kv, st_score, kv_c, score_c, new_kv, new_score, R=r, T=T, REM=rem, D=cfg.head_dim)
            state.pending.extend([new_kv, new_score])
            cs.swap()
        return latents, g0, ng

    def indexer(self, layer: int, x: AlloyBuffer, qr: AlloyBuffer, pos: AlloyBuffer, state: SequenceState,
                positions: np.ndarray, n_keys: int, src: int) -> AlloyBuffer:
        cfg = self.cfg
        L = self.w.layers[layer]
        T = x.shape[0]
        ratio = cfg.compress_ratios[layer]
        hi, di = cfg.index_n_heads, cfg.index_head_dim
        q = self.apply_rope(mm(qr, L.indexer_q_b), pos, layer, hi, di)
        q_r = q
        if self.rounding:
            q_r = _alloc_aligned((T, hi * di), float32)
            ds_block_round[(T,)](q, self.dummy_f16, self.zero_i32, q_r, N=hi * di, MODE=1, BLOCK=32)
        w = mm(x, L.indexer_proj) * (di**-0.5 * hi**-0.5)
        visible = i32_buffer((positions + 1) // ratio)
        cap = key_capacity(n_keys)
        count = i32_buffer(np.array([n_keys], dtype=np.int32))
        score = _alloc_aligned((T, cap), float32)
        use_mask = cfg.uses_candidates(layer) and state.candidates is not None
        ds_index_score[(T, cdiv(cap, 64))](
            q_r, state.index_k[src], w, visible, state.candidates if use_mask else self.dummy_u8, count, score,
            HEADS=hi, HEAD_DIM=di, CAP=cap, HAS_MASK=1 if use_mask else 0,
        )
        if layer == cfg.candidate_source_layer:
            bs = cfg.candidate_block_size
            nb_cap = cdiv(cap, bs)
            nb_count = i32_buffer(np.array([cdiv(n_keys, bs)], dtype=np.int32))
            bm = _alloc_aligned((T, nb_cap), float32)
            ds_block_max[(T, cdiv(nb_cap, 32))](score, visible, count, bm, CAP=cap, BLOCK_SIZE=bs, NB_CAP=nb_cap)
            nblk = cfg.candidate_topk_blocks
            sel = _alloc_aligned((T, nblk), int32)
            ds_topk_select[(T,)](bm, nb_count, sel, CAP=nb_cap, K=nblk, IDX_BITS=idx_bits(nb_cap))
            mask = _alloc_aligned((T, cap), uint8)
            ds_zero_u8[(cdiv(T * cap, 1024),)](mask, N=T * cap)
            ds_block_mask_scatter[(T, nblk)](sel, mask.slice(0, 0, 1), mask, CAP=cap, BLOCK_SIZE=bs, B=nblk, NUM_THREADS=bs)
            state.candidates = mask
        k = cfg.index_topk
        idx = _alloc_aligned((T, k), int32)
        ds_topk_select[(T,)](score, count, idx, CAP=cap, K=k, IDX_BITS=idx_bits(cap))
        self.debug_index[layer] = (score, idx, state.candidates)
        return idx

    def attention(self, layer: int, x: AlloyBuffer, state: SequenceState, start_pos: int, T: int,
                  positions: np.ndarray, pos: AlloyBuffer) -> AlloyBuffer:
        cfg = self.cfg
        L = self.w.layers[layer]
        H, D, win = cfg.n_heads, cfg.head_dim, cfg.window_size
        ratio = cfg.compress_ratios[layer]

        qr = norm(mm(x, L.q_a), L.q_a_norm, cfg.norm_eps)
        q = self.apply_rope(mm(qr, L.q_b), pos, layer, H, D)
        kv = self.apply_rope(norm(mm(x, L.kv), L.kv_norm, cfg.norm_eps), pos, layer, 1, D)

        # window source: the previous min(win-1, start_pos) positions (the tail of
        # the last forward's window, in order) followed by this chunk's rows
        w_prev = min(win - 1, start_pos)
        base = start_pos - w_prev
        win_kv = _alloc_aligned((w_prev + T, D), float16)
        if w_prev > 0:
            prev = state.window[layer]
            prev_rows = i32_buffer(prev.shape[0] - w_prev + np.arange(w_prev))
            ds_copy_rows[(w_prev,)](prev, prev_rows, win_kv, D=D, DST_ROW0=0)
        self.round_into(kv, win_kv, w_prev, mode=0)
        state.window[layer] = win_kv
        q_abs = positions[:, None] - win + 1 + np.arange(win)[None, :]
        win_idx = i32_buffer(np.where(q_abs >= 0, q_abs - base, -1))

        comp_kv = self.dummy_f16
        comp_idx = self.dummy_i32
        comp_n = 0
        if ratio:
            src = cfg.kv_source_for(layer)
            n_keys = (start_pos + T) // ratio
            if cfg.is_kv_source(layer):
                latents, g0, ng = self.compress(layer, x, state, start_pos, T)
                if ng > 0:
                    gpos = i32_buffer((g0 + np.arange(ng)) * ratio)
                    k = self.apply_rope(norm(mm(latents, L.indexer_k), L.indexer_k_norm, cfg.norm_eps), gpos, layer, 1, cfg.index_head_dim)
                    self.round_into(k, state.index_k[layer], g0, mode=1)
                    lat = self.apply_rope(latents, gpos, layer, 1, D)
                    self.round_into(lat, state.comp_kv[layer], g0, mode=2)
            if n_keys > 0:
                if cfg.is_index_source(layer):
                    state.comp_idx = self.indexer(layer, x, qr, pos, state, positions, n_keys, src)
                comp_idx = state.comp_idx
                comp_kv = state.comp_kv[src]
                comp_n = comp_idx.shape[1]

        o = _alloc_aligned((T, H * D), float32)
        ds_sparse_attn[(T, H)](
            q, win_kv, win_idx, comp_kv, comp_idx, L.sinks, o,
            HEADS=H, HEAD_DIM=D, WIN_N=win, COMP_N=comp_n, SCALE=D**-0.5,
        )
        o = self.apply_rope(o, pos, layer, H, D, inverse=True)
        g = cfg.o_groups
        k_g = H * D // g
        n_g = cfg.o_lora_rank
        c = _alloc_aligned((T, g * n_g), float32)
        bm, bn, bk = 16, 64, 64
        ds_grouped_out_proj[(cdiv(T, bm), g * n_g // bn)](o, L.out_a.blk, c, K_G=k_g, N_G=n_g, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk)
        return mm(c, L.out_b)

    # --- moe ---------------------------------------------------------------------------

    def moe(self, layer: int, x: AlloyBuffer, state: SequenceState) -> AlloyBuffer:
        cfg = self.cfg
        L = self.w.layers[layer]
        T = x.shape[0]
        E, K, I, d = cfg.n_routed_experts, cfg.n_activated_experts, cfg.moe_inter_dim, cfg.dim
        logits = _alloc_aligned((T, E), float32)
        dot_transpose_rhs(x, L.gate_inp, logits)
        idx = _alloc_aligned((T, K), int32)
        wts = _alloc_aligned((T, K), float32)
        ds_router_topk[(T,)](
            logits, L.probs_bias, idx, wts,
            E=E, K=K, ROUTE_SCALE=cfg.route_scale, NORM=1 if cfg.norm_topk_prob else 0,
            SCORE_FUNC=SCORE_FUNCS[cfg.score_func], BLOCK=next_pow2(E),
        )
        # everything queued so far (including the previous layer's MoE) has run once
        # the routing is readable, so the previous pins can go
        state.flush(idx)
        routing = read(idx).reshape(-1)
        self.experts.release_all()
        slots = self.experts.acquire(layer, routing.tolist())
        slot_ids = i32_buffer(np.array([slots[int(e)] for e in routing], dtype=np.int32))
        arena = self.experts.arena(layer)
        q6 = arena.layout.down_qtype.name == "Q6_K"
        if T == 1:
            h = _alloc_aligned((K, I), float32)
            moe_gate_up_silu[(K, I)](
                x, arena.gate_up, slot_ids, h, K=d, MOE_INTER=I, TOP_K=K, SWIGLU_LIMIT=cfg.swiglu_limit,
            )
            y = _alloc_aligned((1, d), float32)
            down = moe_down_combine if q6 else moe_down_combine_q4k
            down[(1, d)](h, arena.down, slot_ids, wts, y, HID=d, MOE_INTER=I, TOP_K=K)
        else:
            y = self.moe_grouped(x, arena.gate_up, arena.down, q6, arena.slots, slot_ids, wts)
        gate = mm(x, L.shexp_gate)
        up = mm(x, L.shexp_up)
        hs = _alloc_aligned((T, I), float32)
        ds_swiglu_clamp[(cdiv(T * I, 1024),)](gate, up, hs, N=T * I, LIMIT=cfg.swiglu_limit)
        return y + mm(hs, L.shexp_down)

    def moe_grouped(self, x: AlloyBuffer, gate_up: AlloyBuffer, down: AlloyBuffer, q6: bool, slots: int,
                    slot_ids: AlloyBuffer, wts: AlloyBuffer) -> AlloyBuffer:
        """Prefill routed experts as grouped GEMMs (the `gguf_moe_routed` pipeline) in
        arena-slot space: the counting sort keys on slot ids, so tiles index the arena
        directly and no expert -> slot remap is needed."""
        cfg = self.cfg
        T = x.shape[0]
        K, I, d = cfg.n_activated_experts, cfg.moe_inter_dim, cfg.dim
        S = slots
        R = T * K
        pad, sort_b, bn, bk = 8, 128, 64, 64
        max_tiles = S + cdiv(R, pad)
        max_rows = max_tiles * pad
        active = i32_buffer(np.array([T], dtype=np.int32))
        nb = cdiv(R, sort_b)
        block_count = _alloc_aligned((nb * S,), int32)
        moe_sort_block_count[(nb * S,)](slot_ids, active, block_count, R_TOTAL=R, SORT_B=sort_b, NUM_EXPERTS=S, TOP_K=K)
        count = _alloc_aligned((S,), int32)
        moe_sort_count_from_blocks[(S,)](block_count, active, count, R_TOTAL=R, SORT_B=sort_b, NUM_EXPERTS=S, TOP_K=K)
        row_off = _alloc_aligned((S,), int32)
        total = _alloc_aligned((1,), int32)
        moe_sort_offsets[(1,)](count, row_off, total, NUM_EXPERTS=S, PAD_M=pad)
        tile_e = _alloc_aligned((max_tiles,), int32)
        moe_tile_expert[(max_tiles,)](row_off, count, tile_e, NUM_EXPERTS=S, PAD_M=pad)
        block_off = _alloc_aligned((nb * S,), int32)
        moe_sort_block_off[(S,)](block_count, row_off, block_off, NB=nb, NUM_EXPERTS=S)
        perm = _alloc_aligned((max_rows,), int32)
        inv = _alloc_aligned((R,), int32)
        row_token = _alloc_aligned((max_rows,), int32)
        moe_sort_perm_scan[(R,)](slot_ids, block_off, perm, inv, row_token, TOP_K=K, SORT_B=sort_b, NUM_EXPERTS=S)
        tok_ld = _alloc_aligned((max_rows,), int32)
        tok_st = _alloc_aligned((max_rows,), int32)
        w_row = _alloc_aligned((max_rows,), float32)
        moe_row_tokens[(cdiv(max_rows, 256),)](
            row_token, perm, wts, tile_e, row_off, count, total, tok_ld, tok_st, w_row,
            PAD_M=pad, MAX_ROWS=max_rows, R_TOTAL=R, BLOCK=256,
        )
        h = _alloc_aligned((max_rows, I), float32)
        moe_gate_up_grouped[(max_tiles, cdiv(I, bn))](
            x, gate_up, perm, tok_ld, tile_e, total, w_row, h,
            K=d, MOE_INTER=I, BLOCK_M=pad, BLOCK_N=bn, BLOCK_K=bk, SWIGLU_LIMIT=cfg.swiglu_limit,
        )
        partial = _alloc_aligned((max_rows, d), float32)
        down_kernel = moe_down_grouped_partial if q6 else moe_down_grouped_partial_q4k
        down_kernel[(max_tiles, cdiv(d, bn))](
            h, down, perm, tile_e, total, partial,
            HID=d, MOE_INTER=I, MAX_ROWS=max_rows, BLOCK_M=pad, BLOCK_N=bn, BLOCK_K=bk,
        )
        y = _alloc_aligned((T, d), float32)
        moe_combine_rows[(T, cdiv(d, 256))](partial, inv, y, HID=d, TOP_K=K, BLOCK=256)
        return y

    # --- forward -----------------------------------------------------------------------

    def forward(self, state: SequenceState, ids: np.ndarray, *, all_logits: bool = False,
                trace: list | None = None, teacher: list | None = None) -> np.ndarray:
        """Logits for the last token (or every token). `trace` collects each layer's
        intermediates on the host and `teacher` (per layer: (stream, ffn_pre) from the
        oracle) replaces each layer's output with the oracle's — both for bisecting
        the alloy path against the oracle."""
        cfg = self.cfg
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        T = int(ids.shape[0])
        start_pos = state.pos
        if start_pos + T > self.max_seq_len:
            raise ValueError(f"sequence of {start_pos + T} exceeds max_seq_len {self.max_seq_len}")
        positions = np.arange(start_pos, start_pos + T, dtype=np.int32)
        pos = i32_buffer(positions)
        hashes = state.hasher.hash_ids(ids, start_pos) if state.hasher is not None else None
        state.comp_idx = None
        state.candidates = None

        emb = quant_embedding(Q4_K, i32_buffer(ids.astype(np.int32)), self.w.token_embd.blk)
        x = _alloc_aligned((T, cfg.hc_mult, cfg.dim), float32)
        ds_expand_hc[(T,)](emb, x, HC=cfg.hc_mult, D=cfg.dim)
        pre_mix = np.zeros((T, cfg.hc_mult), dtype=np.float32)
        pre_mix[:, 0] = 1.0
        pre = f32_buffer(pre_mix)
        for layer in range(cfg.n_layers):
            rec: dict = {}
            if cfg.engram is not None and layer in cfg.engram.layer_ids:
                x = self.engram_layer(layer, x, hashes[:, cfg.engram.layer_ids.index(layer), :])
                rec["engram"] = x
            attn_pre, attn_post, attn_comb = self.hc_mixes(x, layer, "attn")
            a = norm(self.hc_pre(x, pre), self.w.layers[layer].attn_norm, cfg.norm_eps)
            a = self.attention(layer, a, state, start_pos, T, positions, pos)
            rec["attn"] = a
            x = self.hc_post(a, x, attn_post, attn_comb)
            ffn_pre, ffn_post, ffn_comb = self.hc_mixes(x, layer, "ffn")
            f = norm(self.hc_pre(x, attn_pre), self.w.layers[layer].ffn_norm, cfg.norm_eps)
            f = self.moe(layer, f, state)
            rec["moe"] = f
            x = self.hc_post(f, x, ffn_post, ffn_comb)
            pre = ffn_pre
            rec["x"] = x
            if trace is not None:
                trace.append({k: read(v) for k, v in rec.items()})
            if teacher is not None:
                x = f32_buffer(teacher[layer][0])
                pre = f32_buffer(teacher[layer][1])
        h = norm(self.hc_pre(x, pre), self.w.output_norm, cfg.norm_eps)
        if not all_logits:
            last = _alloc_aligned((1, cfg.dim), float32)
            ds_copy_rows[(1,)](h, i32_buffer(np.array([T - 1], dtype=np.int32)), last, D=cfg.dim, OUT_F16=0)
            h = last
        logits = mm(h, self.w.output)
        state.pos = start_pos + T
        state.flush(logits)
        out = read(logits)
        self.experts.release_all()
        return out
