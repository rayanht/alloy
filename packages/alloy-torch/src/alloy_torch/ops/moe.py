"""FX handler for the Qwen3.5-MoE routed-expert custom op.

`gguf_moe_routed` lowers the routed FFN. The shared expert is added by the caller.
Two paths, branched on the token count T (prefill chunks trace/compile separately
from M=1 decode, so this is a trace-time constant -> two static plans):

- **decode (T == 1)**: gathered quantized GEMV (`moe_gate_up_silu` -> `moe_down_combine`).
  One token, no grouping to exploit, so the gathered GEMV handles it directly.
- **prefill (T > 1)**: grouped GEMM with NO (T*TOP_K, H)-sized global intermediates
  (each was 17GB at a native-context plan). Sort the chunk's T*TOP_K routings into per-expert
  PAD_M-padded segments (`moe_sort_*` — fill-free, atomic-free), sanitize the per-row
  gather/scatter indices + routing weight analytically (`moe_row_tokens`), then two tiled
  simdgroup-MMA GEMMs that dequantize each expert weight once and reuse it across the
  tile: `moe_gate_up_grouped` GATHERS its A rows in-kernel via TOK_LD (the emitter's
  gathered cooperative load — no a_perm) and folds the routing weight into its epilogue;
  `moe_down_grouped` atomic-ADDs each row's output into Y[TOK_ST[rm]] (the emitter's
  reduce="add" scatter-accumulate MMA epilogue — no partial, no separate combine pass).
  ~2x the GEMV at the real shape; the weight-reuse + MMA recover the GEMV's redundant
  per-token weight reads.
- **production T>1 widths (1 < T <= _DET_COMBINE_MAX_T — spec verify AND chunk
  prefill)**: same grouped pipeline, but the down GEMM is the UNFUSED
  `moe_down_grouped_partial` + `moe_combine_rows` fixed-order reduce — bit-stable,
  where the fused atomic-ADD's reordered sums make generation non-deterministic
  (breaks seeded reproducibility; jitters near-tie argmax against the M=1 decode).
  The partial is 285MB transient at the 4096 chunk, MB-scale at verify widths.

Intermediates flow by data dependency -> static plans. Y is pre-zeroed (`moe_zero_f32`),
and the down GEMM takes a VIEW of y taken after the zero as its Y_DEP input — the atomic
scatter is a read-modify-write, and the declared read both keeps the zero alive (y's
_producer is overwritten by the GEMM, so the zero would otherwise be dead-code-eliminated
and a pool-recycled y would start as garbage) and orders it before the accumulation.
"""

from typing import cast

from alloy._compiler.dtypes import float32, int32
from alloy._dispatch.buf_utils import _alloc_scratch
from alloy._dispatch.kernel import KernelFunction
from alloy._runtime.alloy_buffer import AlloyBuffer
from alloy.std.moe import (
    moe_combine_rows,
    moe_down_combine,
    moe_down_grouped,
    moe_down_grouped_partial,
    moe_gate_up_grouped,
    moe_gate_up_silu,
    moe_router_topk,
    moe_row_tokens,
    moe_sort_block_count,
    moe_sort_block_off,
    moe_sort_count_from_blocks,
    moe_sort_offsets,
    moe_sort_perm_scan,
    moe_tile_expert,
    moe_zero_f32,
)

_MOE_ROUTER_TOPK = cast(KernelFunction, moe_router_topk)
_MOE_GATE_UP_SILU = cast(KernelFunction, moe_gate_up_silu)
_MOE_DOWN_COMBINE = cast(KernelFunction, moe_down_combine)
_MOE_GATE_UP_GROUPED = cast(KernelFunction, moe_gate_up_grouped)
_MOE_DOWN_GROUPED = cast(KernelFunction, moe_down_grouped)
_MOE_DOWN_GROUPED_PARTIAL = cast(KernelFunction, moe_down_grouped_partial)
_MOE_COMBINE_ROWS = cast(KernelFunction, moe_combine_rows)
_MOE_COUNT_FROM_BLOCKS = cast(KernelFunction, moe_sort_count_from_blocks)
_MOE_SORT_OFFSETS = cast(KernelFunction, moe_sort_offsets)
_MOE_TILE_EXPERT = cast(KernelFunction, moe_tile_expert)
_MOE_BLOCK_COUNT = cast(KernelFunction, moe_sort_block_count)
_MOE_BLOCK_OFF = cast(KernelFunction, moe_sort_block_off)
_MOE_PERM_SCAN = cast(KernelFunction, moe_sort_perm_scan)
_MOE_ROW_TOKENS = cast(KernelFunction, moe_row_tokens)
_MOE_ZERO_F32 = cast(KernelFunction, moe_zero_f32)

# Grouped-GEMM tile rows == sort padding granularity. 8 = MMA floor (lowest padding
# at the ~4 tokens/expert of a 128-token chunk). MUST equal moe_*_grouped's BLOCK_M.
_PAD_M = 8
# Rank-scan block size for the block-decomposed counting sort. The flat scan was
# O(R²) and 23% of qwen3.6:35b's 4096-chunk prefill; block decomposition costs
# O(E·R) (block_count) + O(E·NB) (count_from_blocks + block_off) + O(R·SORT_B/2)
# (perm_scan's in-block rank), so SORT_B trades perm_scan's scan against the
# NB-walks. Standalone sweep of the 4-kernel chain at R=32768:
# 64 → 621 µs, 128 → 608 µs,
# 256 → 699 µs, 512 → 944 µs (vs 25,535 + 2,148 µs flat perm_scan + count_scan).
_SORT_B = 128
# Fixed grouped-GEMM N/K block (passed explicitly with an explicit grid — the kernels have
# no dispatch_spec, so auto-grid mis-sizes them; tuning these would need that spec).
_GROUPED_BN = 64
_GROUPED_BK = 64
# All production T>1 widths (spec verify 2-16, chunk prefills 128/4096) take the
# UNFUSED down + fixed-order combine: the fused atomic-ADD's reordered sums make
# generation non-deterministic — measured ~0.1-0.3-logit run-to-run jitter on
# 35b prefill logits, which breaks the seeded-determinism contract, flips
# near-tie argmax between the plain and spec streams, and destabilizes the
# gate's teacher-forced classifier (same flip measured delta 0.000 and 3.550
# across runs). The (MAX_ROWS, H) partial that fusion avoids is 285MB transient
# at the 4096 chunk (~0.6% chunk-time traffic) and MB-scale at verify widths;
# the 17GB figure that motivated the fusion was a native-context M=262144 plan
# production never compiles (prefill is chunked). Fused stays for T beyond any
# production chunk.
_DET_COMBINE_MAX_T = 4096


def _next_pow2(n: int) -> int:
    block = 1
    while block < n:
        block *= 2
    return block


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _gguf_moe_routed_handler(
    hidden: AlloyBuffer,            # (T, H)
    router_logits: AlloyBuffer,    # (T, num_experts)
    gate_up_blocks: AlloyBuffer,   # (E, 2I, (H/256)*144) native Q4_K
    down_qweight: AlloyBuffer,     # (E, H, (I/256)*210) Q6_K
    num_experts: int,
    top_k: int,
    moe_intermediate: int,
) -> AlloyBuffer:
    T = hidden.shape[0]
    H = hidden.shape[1]
    I = moe_intermediate
    R = T * top_k

    # 1. Router: top-k expert indices + softmax-over-k weights, per token.
    # `active` is the router's own launched row count — the counting sort's
    # exact bound, written in-kernel so it tracks whatever grid the dispatcher
    # actually launched (full or grid-shrunk), and the router→sort dependency
    # orders the plan.
    idx = _alloc_scratch((R,), int32)
    weights = _alloc_scratch((R,), float32)
    active = _alloc_scratch((1,), int32)
    _MOE_ROUTER_TOPK[(T,)](
        router_logits, idx, weights, active,
        NUM_EXPERTS=num_experts, TOP_K=top_k, BLOCK=_next_pow2(num_experts),
    )

    if T == 1:
        # Decode: gathered GEMV (no token grouping to exploit).
        h = _alloc_scratch((R, I), float32)
        _MOE_GATE_UP_SILU[(R, I)](
            hidden, gate_up_blocks, idx, h,
            K=H, MOE_INTER=I, TOP_K=top_k,
        )
        y = _alloc_scratch((T, H), float32)
        _MOE_DOWN_COMBINE[(T, H)](
            h, down_qweight, idx, weights, y,
            HID=H, MOE_INTER=I, TOP_K=top_k,
        )
        return y

    # Prefill: grouped GEMM. Atomic-free + fill-free sort (every output overwritten by a
    # per-program scan) — robust through dispatch_plan; pad/empty garbage is never consumed.
    pad = _PAD_M
    max_tiles = num_experts + _cdiv(R, pad)
    max_rows = max_tiles * pad

    # 2. Sort routings into per-expert PAD-padded segments, block-decomposed
    # (see std/moe.py): per-block expert histograms feed BOTH the global COUNT
    # (per-expert sum over active blocks — replaced the retired O(E·R)
    # count_scan, 2.1 ms/layer) and the cross-block prefix BLOCK_OFF, so
    # perm_scan's rank scan is bounded by SORT_B instead of R (the flat scan
    # was O(R²) — 25.3 ms/layer at the 4096-chunk, 23% of qwen3.6:35b prefill).
    nb = _cdiv(R, _SORT_B)
    block_count = _alloc_scratch((nb * num_experts,), int32)
    _MOE_BLOCK_COUNT[(nb * num_experts,)](
        idx, active, block_count,
        R_TOTAL=R, SORT_B=_SORT_B, NUM_EXPERTS=num_experts, TOP_K=top_k,
    )
    count = _alloc_scratch((num_experts,), int32)
    _MOE_COUNT_FROM_BLOCKS[(num_experts,)](
        block_count, active, count,
        R_TOTAL=R, SORT_B=_SORT_B, NUM_EXPERTS=num_experts, TOP_K=top_k,
    )
    row_off = _alloc_scratch((num_experts,), int32)
    total = _alloc_scratch((1,), int32)
    _MOE_SORT_OFFSETS[(1,)](count, row_off, total, NUM_EXPERTS=num_experts, PAD_M=pad)
    tile_e = _alloc_scratch((max_tiles,), int32)
    _MOE_TILE_EXPERT[(max_tiles,)](row_off, count, tile_e, NUM_EXPERTS=num_experts, PAD_M=pad)
    block_off = _alloc_scratch((nb * num_experts,), int32)
    _MOE_BLOCK_OFF[(num_experts,)](
        block_count, row_off, block_off, NB=nb, NUM_EXPERTS=num_experts,
    )
    perm = _alloc_scratch((max_rows,), int32)   # only active positions written; pad rows unread
    inv = _alloc_scratch((R,), int32)           # fully written by perm_scan
    # ROW_TOKEN[rm] = (token,slot)//TOP_K — the source token row, scattered by the sort (pad
    # rows hold garbage; fill-free). moe_row_tokens then sanitizes it ANALYTICALLY (validity
    # from the fully-overwritten COUNT/ROW_OFF/TOTAL_ROWS, never trusting pad garbage) into the
    # two per-row gather/scatter index buffers the grouped GEMMs consume.
    row_token = _alloc_scratch((max_rows,), int32)
    _MOE_PERM_SCAN[(R,)](
        idx, block_off, perm, inv, row_token,
        TOP_K=top_k, SORT_B=_SORT_B, NUM_EXPERTS=num_experts,
    )
    tok_ld = _alloc_scratch((max_rows,), int32)  # gather-load index: pads clamped to row 0
    tok_st = _alloc_scratch((max_rows,), int32)  # scatter-store index: pads = sentinel >= T
    w_row = _alloc_scratch((max_rows,), float32)  # routing weight per sorted row: pads = 0
    _MOE_ROW_TOKENS[(_cdiv(max_rows, 256),)](
        row_token, perm, weights, tile_e, row_off, count, total, tok_ld, tok_st, w_row,
        PAD_M=pad, MAX_ROWS=max_rows, R_TOTAL=R, BLOCK=256,
    )

    # 3. Grouped gate_up+SiLU with the routing weight folded in. The tile GATHERS its A
    # rows in-kernel via TOK_LD (the emitter's gathered cooperative load) — no global
    # expert-major a_perm buffer (that was (T*TOP_K, H) = 17GB at a native-context plan).
    # EXPLICIT grid: the auto-grid (dispatch_spec) mis-sizes the M dim for these kernels
    # (gave 6 of max_tiles), so the grid is set here from max_tiles and the (fixed) BLOCK_N.
    bn, bk = _GROUPED_BN, _GROUPED_BK
    h = _alloc_scratch((max_rows, I), float32)
    _MOE_GATE_UP_GROUPED[(max_tiles, _cdiv(I, bn))](
        hidden, gate_up_blocks, perm, tok_ld, tile_e, total,
        w_row, h,
        K=H, MOE_INTER=I, BLOCK_M=pad, BLOCK_N=bn, BLOCK_K=bk,
    )

    # 4a. Production widths (verify + chunk prefill): unfused down -> per-sorted-row
    # PARTIAL, then a fixed slot-order combine. Bit-stable across runs, unlike the
    # fused atomic-ADD (see _DET_COMBINE_MAX_T). Pad partials hold garbage no one
    # reads.
    if T <= _DET_COMBINE_MAX_T:
        partial = _alloc_scratch((max_rows, H), float32)
        _MOE_DOWN_GROUPED_PARTIAL[(max_tiles, _cdiv(H, bn))](
            h, down_qweight, perm, tile_e, total, partial,
            HID=H, MOE_INTER=I, MAX_ROWS=max_rows,
            BLOCK_M=pad, BLOCK_N=bn, BLOCK_K=bk,
        )
        y = _alloc_scratch((T, H), float32)
        _MOE_COMBINE_ROWS[(T, _cdiv(H, 256))](
            partial, inv, y, HID=H, TOP_K=top_k, BLOCK=256,
        )
        return y

    # 4b. Grouped down with the combine FUSED: each tile's down output atomic-adds into
    # Y[TOK_ST[rm]] (reduce="add" scatter-accumulate epilogue) — no global (MAX_ROWS, H)
    # partial buffer (the other 17GB at native) and no separate combine pass.
    y = _alloc_scratch((T, H), float32)
    _MOE_ZERO_F32[(_cdiv(T * H, 1024),)](y, N=T * H, BLOCK=1024)
    # Y_DEP must be a VIEW taken AFTER the zero: the down call overwrites y's _producer,
    # so passing y itself as the dep would make the zero unreachable from the output walk
    # (DCE — it only "worked" because fresh Metal pages happen to be zero; a pooled run-1
    # buffer is recycled garbage). The view object keeps the zero's producer chain, which
    # both keeps the zero alive and orders it before the GEMM (the atomic scatter is a
    # read-modify-write of y).
    y_dep = y.slice(0, 0, 1)
    # y passed BOTH as Y_DEP (input — the atomic scatter is a read-modify-write; the
    # declared read orders the zero before the GEMM) and as Y_OUT (the accumulated output).
    _MOE_DOWN_GROUPED[(max_tiles, _cdiv(H, bn))](
        h, down_qweight, perm, tok_st, tile_e, total, y_dep, y,
        HID=H, MOE_INTER=I, T_ROWS=T, MAX_ROWS=max_rows, BLOCK_M=pad, BLOCK_N=bn, BLOCK_K=bk,
    )
    return y
