"""Handler for `torch.ops.alloy.linear_attention_update` — Qwen 3.5
GatedDeltaNet (linear-attention) layer core.

Subsumes the layer body of the eager `Qwen3_5GatedDeltaNet.forward`
(modeling_qwen3_5.py:424-543) so the conv- and recurrent-state cache writes
stay inside the FX graph (the op schema declares them mutable; otherwise
AOT autograd lifts the `.copy_()` and decode reads zero state).
"""

from __future__ import annotations

from typing import cast

import numpy as np
import torch

from alloy._compiler.dtypes import float32
from alloy._dispatch.buf_utils import _alloc_aligned, _alloc_scratch
from alloy._dispatch.dispatch import _engine
from alloy._dispatch.kernel import KernelFunction
from alloy._runtime.alloy_buffer import AlloyBuffer
from alloy.std.delta_net import (
    _conv_state_finalize_prefill,
    causal_conv1d_with_state_decode,
    causal_conv1d_with_state_prefill,
    chunked_gdr_stage1,
    chunked_gdr_stage2,
    conv_state_save_real_pos,
    delta_net_gate_compute,
    l2norm_last_dim,
    recurrent_gated_delta_rule,
    recurrent_gdr_dvblock_save,
    rms_norm_gated,
    silu_inplace,
)
from alloy_torch.compile_window import compile_window


_CONV_TAPE_BUFS: dict[int, tuple[AlloyBuffer, int, int]] = {}  # ptr → (buf, rows, conv_dim)
_TAPE_DUMMY: AlloyBuffer | None = None


def conv_tape_clear_registry() -> None:
    _CONV_TAPE_BUFS.clear()


def alias_dn_scratch(old_conv_ptr: int, new_conv_ptr: int,
                     old_rec_ptr: int, new_rec_ptr: int) -> None:
    """Re-key a layer's verify scratch from its old (conv, recurrent) storage
    pointers to its new ones. The pinned verify plan writes the conv tape /
    GDR round-buffer tees to fixed addresses; a multi-slice switch repoints the
    cache storage, so the rollback's data_ptr-keyed lookups must alias to the
    same buffers — re-allocating fresh ones would leave the readers on empty
    memory while the plan keeps writing the originals. Scratch is transient
    single-generation memory, so sharing it across slices is correct."""
    conv = _CONV_TAPE_BUFS.get(old_conv_ptr)
    if conv is not None:
        _CONV_TAPE_BUFS[new_conv_ptr] = conv
    rb = _GDR_ROUND_BUFS.get(old_rec_ptr)
    if rb is not None:
        _GDR_ROUND_BUFS[new_rec_ptr] = rb


def prealloc_conv_tape(conv_state_ptr: int, rows: int, conv_dim: int) -> None:
    """Allocate a layer's tape bank outside any plan recording so the pinned
    verify's replayed writes keep landing in this exact memory."""
    entry = _CONV_TAPE_BUFS.get(conv_state_ptr)
    if entry is not None and entry[1] >= rows:
        return
    tape = _alloc_aligned((rows * conv_dim,), float32)
    _engine.untrack_alloc(tape.base_ptr)
    _CONV_TAPE_BUFS[conv_state_ptr] = (tape, rows, conv_dim)


def conv_tape_tensor(conv_state_ptr: int) -> torch.Tensor | None:
    """The taped (rows, conv_dim) f32 inputs for the layer owning this
    conv_state buffer, as a zero-copy torch view. None if never taped."""
    entry = _CONV_TAPE_BUFS.get(conv_state_ptr)
    if entry is None:
        return None
    buf, rows, conv_dim = entry
    return torch.from_numpy(np.asarray(buf)).reshape(rows, conv_dim)


def _tape_dummy() -> AlloyBuffer:
    """1-element placeholder bound to the conv kernel's tape param on every
    SAVE_TAPE=0 dispatch (the store folds away; the buffer is never touched)."""
    global _TAPE_DUMMY
    if _TAPE_DUMMY is None:
        _TAPE_DUMMY = _alloc_aligned((1,), float32)
        _engine.untrack_alloc(_TAPE_DUMMY.base_ptr)
    return _TAPE_DUMMY


# rec_state ptr → round-buffer entry for the verify GDR. The verify plan
# tees k_l2 / g / beta / v in-kernel into these stable untracked buffers
# (`recurrent_gdr_dvblock_save` binds them directly; pooled intermediates may
# be physically recycled and separate lazy copy ops never replay). The
# session's per-round `gdr_state_reconstruct` dispatch re-runs the serial
# recurrence on them to advance slot 0 by the committed token count, for
# chunk-aligned verify widths. Materializing all S per-token states is pure
# state-write bandwidth (S × 2MB fp32 ≈ 0.25ms/layer at S=16) while only one
# is consumed.
_GDR_ROUND_BUFS: dict[int, dict] = {}


def prealloc_gdr_round_bufs(
    rec_state_ptr: int, s: int, nv: int, dk: int, dv: int, nk: int, batch: int = 1
) -> None:
    """Allocate a layer's verify round buffers outside plan recording. No-op
    when `s` isn't a whole number of chunks (MTP/PLD's small widths stay on the
    serial SAVE_STEPS slot-bank path, matching the dispatch gate in the
    handler)."""
    if s % _GDR_C != 0:
        return
    entry = _GDR_ROUND_BUFS.get(rec_state_ptr)
    if entry is not None and entry["S"] >= s:
        return
    kk = _alloc_aligned((batch * s * nk * dk,), float32)
    gg = _alloc_aligned((batch * s * nv,), float32)
    bb = _alloc_aligned((batch * s * nv,), float32)
    vv = _alloc_aligned((batch * s * nv * dv,), float32)
    for buf in (kk, gg, bb, vv):
        _engine.untrack_alloc(buf.base_ptr)
    _GDR_ROUND_BUFS[rec_state_ptr] = {
        "k": kk, "g": gg, "beta": bb, "v": vv,
        "BATCH": batch, "S": s, "NV": nv, "DK": dk, "DV": dv, "NK": nk,
    }


def gdr_round_bufs(rec_state_ptr: int) -> dict | None:
    """The session-side accessor: this layer's verify round buffers and
    dims, or None (serial SAVE_STEPS slot-bank path)."""
    return _GDR_ROUND_BUFS.get(rec_state_ptr)


def gdr_verify_uses_reconstruct(s: int, dv: int) -> bool:
    """True when a SAVE_STEPS verify at width `s` dispatches the
    dvblock+reconstruct path — the recurrent slot bank then never writes past
    slot 0, so attach_spec can size it to 1 slot (DFlash block 16: 16 → 1 slot
    ≈ −400MB resident on qwen3.5:4b). Must mirror the handler's dispatch gate
    below."""
    return s > 1 and s % _GDR_C == 0 and dv % _GDR_DVB == 0


_CONV1D_PREFILL = cast(KernelFunction, causal_conv1d_with_state_prefill)
_CONV1D_FINALIZE = cast(KernelFunction, _conv_state_finalize_prefill)
_CONV1D_DECODE = cast(KernelFunction, causal_conv1d_with_state_decode)
_CONV_STATE_REAL = cast(KernelFunction, conv_state_save_real_pos)
_SILU = cast(KernelFunction, silu_inplace)
_L2NORM = cast(KernelFunction, l2norm_last_dim)
_GATE = cast(KernelFunction, delta_net_gate_compute)
_RECURRENT = cast(KernelFunction, recurrent_gated_delta_rule)
_GDR_STAGE1 = cast(KernelFunction, chunked_gdr_stage1)
_GDR_STAGE2 = cast(KernelFunction, chunked_gdr_stage2)
_GDR_DVBLOCK_SAVE = cast(KernelFunction, recurrent_gdr_dvblock_save)
_RMSGATED = cast(KernelFunction, rms_norm_gated)

# Chunked (2-stage FLA) delta rule for prefill: ~2x the serial recurrent kernel
# at qwen3.5's dims (DK=DV=128, NV=16). Stage 1 is fully parallel over NC×NV
# chunks; stage 2 is the DV-blocked serial scan. Used when the prefill bucket is
# a multiple of the chunk size. Decode (S=1) and small spec-verify widths
# (MTP/PLD) stay on _RECURRENT; chunk-aligned verify widths (DFlash block 16)
# dispatch _GDR_DVBLOCK_SAVE (serial numerics + reconstruct tees).
_GDR_C = 8
_GDR_DVB = 8


def _linear_attention_update_handler(
    mixed_qkv: AlloyBuffer,
    z: AlloyBuffer,
    a: AlloyBuffer,
    b: AlloyBuffer,
    conv_state: AlloyBuffer,
    recurrent_state: AlloyBuffer,
    conv1d_w: AlloyBuffer,
    A_log: AlloyBuffer,
    dt_bias: AlloyBuffer,
    norm_w: AlloyBuffer,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_kernel_size: int,
    norm_eps: float,
    has_previous_state: bool,
    real_len: AlloyBuffer | None = None,
) -> AlloyBuffer:
    """Run the full GatedDeltaNet layer body.

    Pipeline:
      1. Transpose `mixed_qkv` to (B, conv_dim, S) for the conv.
      2. Causal Conv1d with state — prefill (S > 1) or decode (S == 1
         with prior state). Both update `conv_state` in place.
      3. Transpose back to (B, S, conv_dim) and split into q, k, v.
      4. L2-normalise q and k along head_dim.
      5. Pre-scale q by 1/sqrt(head_k_dim).
      6. Compute g = -exp(A_log)*softplus(a+dt_bias);  beta = sigmoid(b).
      7. Recurrent delta rule — updates `recurrent_state` in place,
         emits per-token output.
      8. Fused RMSNormGated with `z`.
      9. Reshape to (B, S, value_dim).
    """
    batch_size, seq_len, conv_dim = mixed_qkv.shape
    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim
    assert conv_dim == 2 * key_dim + value_dim, (
        f"conv_dim mismatch: got {conv_dim}, expected {2 * key_dim + value_dim} "
        f"(2*{key_dim} + {value_dim})"
    )

    # 1. Conv1d reads `mixed_qkv` in its native (B, S, C) layout directly. The
    # conv kernels address (B,S,C) and emit (B,S,C), so the pipeline stays in
    # (B,S,C), avoiding a full-tensor strided copy per layer each way.
    # Weight (C, 1, K) → (C, K).
    w_squeezed = conv1d_w.reshape((conv_dim, conv_kernel_size))
    # conv_state is the cache buffer (Tensor(c!) — mutable input). The kernels
    # use flat row-major addressing so pass it as-is rather than via
    # `.reshape(...)`: the reshape would create a fresh view slot the alloy
    # backend treats as an intermediate, and the kernel writes would never
    # propagate back to the original input slot's storage.
    if has_previous_state and seq_len == 1:
        N_conv = batch_size * conv_dim
        qkv_post_conv = _alloc_scratch((N_conv,), float32)
        # Decode: mixed_qkv (B,1,C) flattens to (B*C,), the layout the decode
        # conv reads.
        _CONV1D_DECODE[((N_conv + 255) // 256,)](
            mixed_qkv.reshape((N_conv,)),
            w_squeezed.reshape((conv_dim * conv_kernel_size,)),
            conv_state,
            qkv_post_conv,
            BATCH=batch_size,
            C=conv_dim,
            K=conv_kernel_size,
        )
        conv_out_shape = (batch_size, 1, conv_dim)
    else:
        N_conv = batch_size * conv_dim * seq_len
        qkv_post_conv = _alloc_scratch((N_conv,), float32)
        if compile_window.spec_save_steps:
            prealloc_conv_tape(conv_state.base_ptr, seq_len, conv_dim)
            tape = _CONV_TAPE_BUFS[conv_state.base_ptr][0]
            save_tape = 1
        else:
            tape = _tape_dummy()
            save_tape = 0
        # 2D grid (B*S, ceil(C/256)): position on axis-0 so the grid-shrink recipe
        # shrinks the conv to the real prompt length; axis-1 tiles the channels.
        _CONV1D_PREFILL[(batch_size * seq_len, (conv_dim + 255) // 256)](
            mixed_qkv.reshape((N_conv,)),
            w_squeezed.reshape((conv_dim * conv_kernel_size,)),
            tape,
            conv_state,
            qkv_post_conv,
            BATCH=batch_size,
            C=conv_dim,
            S=seq_len,
            K=conv_kernel_size,
            SAVE_TAPE=save_tape,
        )
        # Save conv_state from the last K real positions [real_len-K,
        # real_len-1]. Pass qkv_post_conv as dep_in to force the planner to
        # schedule this after the conv kernel (otherwise topo-sort puts it
        # first → conv's pre-context reads pick up real-position bytes).
        if real_len is not None and seq_len > 1:
            _CONV_STATE_REAL[((batch_size * conv_dim * conv_kernel_size + 255) // 256,)](
                mixed_qkv.reshape((N_conv,)),
                real_len.reshape((1,)),
                qkv_post_conv,
                conv_state,
                BATCH=batch_size,
                C=conv_dim,
                S=seq_len,
                K=conv_kernel_size,
            )
        conv_out_shape = (batch_size, seq_len, conv_dim)

    # 3. SiLU. Conv kernels emit the linear conv output; alloy's MSL emitter
    # doesn't hoist post-loop scalar SiLU into outer scope correctly, so it's a
    # separate elementwise pass. 2D grid (B*S, ceil(C/256)) over the (B,S,C)
    # conv output so the grid-shrink recipe shrinks it.
    qkv_silu = _alloc_scratch((N_conv,), float32)
    pos_count = conv_out_shape[0] * conv_out_shape[1]
    _SILU[(pos_count, (conv_dim + 255) // 256)](qkv_post_conv, qkv_silu, C=conv_dim)

    # 4. The conv emits (B, S, C) — split q/k/v directly off dim 2 (the channel
    # dim). Each slice's `.contiguous()` extracts its contiguous (B*S, feat)
    # sub-tensor; those are M-outer copies the grid-shrink recipe shrinks to
    # the real prompt length.
    qkv_bsc = qkv_silu.reshape(conv_out_shape)
    actual_s = conv_out_shape[1]
    q_flat = qkv_bsc.slice(2, 0, key_dim).contiguous().reshape((batch_size * actual_s, key_dim))
    k_flat = (
        qkv_bsc.slice(2, key_dim, 2 * key_dim)
        .contiguous()
        .reshape((batch_size * actual_s, key_dim))
    )
    v_flat = (
        qkv_bsc.slice(2, 2 * key_dim, 2 * key_dim + value_dim)
        .contiguous()
        .reshape((batch_size * actual_s, value_dim))
    )

    # 5. L2-normalise q, k along head_dim. Buffers must be passed flat (1D) —
    # the 2D-view reshape pattern produces zero output through this dispatch
    # path.
    M_qk = batch_size * actual_s * num_k_heads
    N_qk = M_qk * head_k_dim
    q_l2 = _alloc_scratch((N_qk,), float32)
    k_l2 = _alloc_scratch((N_qk,), float32)
    # 2D grid (B*S, num_k_heads): position on axis-0 so the grid-shrunk chunk
    # prefill recipe shrinks the launch to the real prompt length (1D (M_qk,)
    # buried M at the padded max). buffer row = pos*num_k_heads + head.
    bs = batch_size * actual_s
    _L2NORM[(bs, num_k_heads)](q_flat.reshape((N_qk,)), q_l2, N=head_k_dim, HEADS=num_k_heads)
    _L2NORM[(bs, num_k_heads)](k_flat.reshape((N_qk,)), k_l2, N=head_k_dim, HEADS=num_k_heads)

    # GVA-style head repeat is handled inside the recurrent kernel via the NK
    # constexpr — keeping q/k at NK-headed size avoids an expand+contiguous on
    # every layer, which would add buffer operations to the plan and disrupt
    # slot tracking for the kernel's recurrent_state mutation.

    # 6. Pre-scale q by 1/sqrt(DK) (lazy multiply).
    scale = 1.0 / (head_k_dim**0.5)
    q_scaled = q_l2 * scale

    # 7. Gate and beta.
    M_g = batch_size * actual_s * num_v_heads
    # Bypass the alloy `delta_net_gate_compute` kernel — it returns wrong g for
    # S>1 (collapsing the chain into it diverges at token 1). Compute g, beta
    # via direct AlloyBuffer element-wise ops.
    # g = -exp(A_log) * softplus(a + dt_bias)
    # beta = sigmoid(b)
    # A_log, dt_bias are (num_v_heads,) — broadcast over (B, S, NV).
    a_3d = a.contiguous()  # (B, S, NV)
    b_3d = b.contiguous()
    A_log_3d = A_log.reshape((1, 1, num_v_heads))
    dt_bias_3d = dt_bias.reshape((1, 1, num_v_heads))
    sp_arg = a_3d + dt_bias_3d  # broadcast → (B, S, NV)
    # softplus stable: max(0, x) + log(1 + exp(-|x|))
    sp = sp_arg.relu() + ((sp_arg.abs() * -1.0).exp() + 1.0).log()
    g_3d = (A_log_3d.exp() * -1.0) * sp  # (B, S, NV)
    beta_3d = b_3d.sigmoid()
    g = g_3d.reshape((M_g,))
    beta = beta_3d.reshape((M_g,))

    # 8. Recurrent delta rule. During padded prefill (real_len given, S > 1)
    # the recurrence runs the full bucket length but must save
    # `recurrent_state` after only the real tokens — otherwise the carried
    # state is polluted by padding tokens and the following decode degenerates.
    # Pass real_len + HAS_REAL_LEN so the kernel saves at the real boundary.
    # Decode (S == 1) and exact-length prefill keep the unconditional post-loop
    # save (real_len_t is then an unused placeholder).
    has_real_len = real_len is not None and seq_len > 1
    real_len_arg = real_len.reshape((1,)) if has_real_len else g
    N_attn = batch_size * actual_s * num_v_heads * head_v_dim
    core_attn = _alloc_scratch((N_attn,), float32)
    # Prefill uses the 2-stage chunked delta rule when the length is a whole
    # number of chunks. The spec verify does not — the chunked T-inverse's f32
    # deviation (~1e-2 on ill-conditioned repetitive rows) flips gate near-ties
    # at ~6× the rate budget — it dispatches the DV-blocked serial save kernel
    # instead: op-for-op the serial kernel's numerics, with the k/q row loads
    # (one program per (head, dv) re-reads 1KB per step ≈ 96MB/dispatch at
    # S=16) amortized across 8 columns per program. Decode (S == 1) and
    # non-chunk-aligned verify widths (MTP/PLD at small m) stay on the serial
    # kernel.
    use_chunked = (
        actual_s > 1
        and actual_s % _GDR_C == 0
        and head_v_dim % _GDR_DVB == 0
    )
    if use_chunked and compile_window.spec_save_steps:
        # Spec verify: leaves recurrent_state untouched (slot 0 stays the
        # pre-round live state) and tees k_l2/g/beta/v in-kernel into the
        # layer's stable registry buffers, bound directly. Separate lazy copy
        # ops do not work here: extern-root copies materialize outside the
        # captured verify plan and never replay. The session's per-round
        # gdr_state_reconstruct dispatch advances slot 0 by the committed
        # count.
        prealloc_gdr_round_bufs(
            recurrent_state.base_ptr, actual_s,
            num_v_heads, head_k_dim, head_v_dim, num_k_heads, batch_size,
        )
        rb = _GDR_ROUND_BUFS[recurrent_state.base_ptr]
        _GDR_DVBLOCK_SAVE[(batch_size * num_v_heads * (head_v_dim // _GDR_DVB),)](
            q_scaled.reshape((N_qk,)),
            k_l2.reshape((N_qk,)),
            v_flat.reshape((N_attn,)),
            g,
            beta,
            recurrent_state,
            rb["k"],
            rb["g"],
            rb["beta"],
            rb["v"],
            core_attn,
            BATCH=batch_size,
            S=actual_s,
            NV=num_v_heads,
            DK=head_k_dim,
            DV=head_v_dim,
            NK=num_k_heads,
        )
    elif use_chunked:
        NC = actual_s // _GDR_C
        n_dk = batch_size * num_v_heads * actual_s * head_k_dim
        n_cc = batch_size * num_v_heads * NC * _GDR_C * _GDR_C
        W_o = _alloc_scratch((n_dk,), float32)
        qg_o = _alloc_scratch((n_dk,), float32)
        kd_o = _alloc_scratch((n_dk,), float32)
        T_o = _alloc_scratch((n_cc,), float32)
        at_o = _alloc_scratch((n_cc,), float32)
        # 2D grid (NC, B*NV): chunk on axis-0 so the grid-shrink recipe shrinks the
        # intra-chunk launch to ceil(real_len/C) chunks (the parallel stage-1
        # counterpart of stage-2's runtime NC loop bound).
        _GDR_STAGE1[(NC, batch_size * num_v_heads)](
            q_scaled.reshape((N_qk,)),
            k_l2.reshape((N_qk,)),
            g,
            beta,
            real_len_arg,
            W_o,
            T_o,
            at_o,
            qg_o,
            kd_o,
            BATCH=batch_size,
            S=actual_s,
            NV=num_v_heads,
            DK=head_k_dim,
            DV=head_v_dim,
            NK=num_k_heads,
            C=_GDR_C,
            HAS_REAL_LEN=1 if has_real_len else 0,
        )
        _GDR_STAGE2[(batch_size * num_v_heads * (head_v_dim // _GDR_DVB),)](
            v_flat.reshape((N_attn,)),
            beta,
            g,
            real_len_arg,
            W_o,
            T_o,
            at_o,
            qg_o,
            kd_o,
            recurrent_state,
            core_attn,
            BATCH=batch_size,
            S=actual_s,
            NV=num_v_heads,
            DK=head_k_dim,
            DV=head_v_dim,
            C=_GDR_C,
            DV_BLOCK=_GDR_DVB,
            HAS_REAL_LEN=1 if has_real_len else 0,
        )
    else:
        _RECURRENT[(batch_size * num_v_heads * head_v_dim,)](
            q_scaled.reshape((N_qk,)),
            k_l2.reshape((N_qk,)),
            v_flat.reshape((N_attn,)),
            g,
            beta,
            real_len_arg,
            recurrent_state,
            core_attn,
            BATCH=batch_size,
            S=actual_s,
            NV=num_v_heads,
            DK=head_k_dim,
            DV=head_v_dim,
            NK=num_k_heads,
            HAS_REAL_LEN=1 if has_real_len else 0,
            SAVE_STEPS=1 if compile_window.spec_save_steps else 0,
        )

    # 9. Fused RMSNormGated — 1D buffers. 2D grid (B*S, num_v_heads): position on
    # axis-0 so the grid-shrunk chunk prefill recipe shrinks it to the real prompt length.
    M_rg = batch_size * actual_s * num_v_heads
    N_rg = M_rg * head_v_dim
    out = _alloc_scratch((N_rg,), float32)
    _RMSGATED[(batch_size * actual_s, num_v_heads)](
        core_attn,
        z.contiguous().reshape((N_rg,)),
        norm_w.reshape((head_v_dim,)),
        out,
        DV=head_v_dim,
        HEADS=num_v_heads,
        EPS=norm_eps,
    )
    return out.reshape((batch_size, actual_s, value_dim))
