"""Handler for `torch.ops.alloy.short_conv_update` — LFM2 short-conv layer core.

Subsumes the causal depthwise Conv1d (with rolling state) of the eager
`Lfm2ShortConv.forward` so the conv-state cache write stays inside the FX graph
(the op schema declares `conv_state` mutable; otherwise AOT autograd lifts the
`.copy_()` and decode reads zero state → garbage). Reuses the DeltaNet conv
kernels verbatim — LFM2's conv is the same causal depthwise Conv1d, minus the
SiLU activation and the recurrent rule.
"""

from __future__ import annotations

from typing import cast


from alloy._compiler.dtypes import float32
from alloy._dispatch.buf_utils import _alloc_scratch
from alloy._dispatch.kernel import KernelFunction
from alloy._runtime.alloy_buffer import AlloyBuffer
from alloy.std.delta_net import (
    causal_conv1d_gated_decode,
    causal_conv1d_with_state_decode,
    causal_conv1d_with_state_prefill,
    conv_state_save_real_pos,
)
from alloy_torch.ops.delta_net import _tape_dummy

_CONV1D_PREFILL = cast(KernelFunction, causal_conv1d_with_state_prefill)
_CONV1D_DECODE = cast(KernelFunction, causal_conv1d_with_state_decode)
_CONV1D_GATED_DECODE = cast(KernelFunction, causal_conv1d_gated_decode)
_CONV_STATE_REAL = cast(KernelFunction, conv_state_save_real_pos)


def _short_conv_gated_handler(
    bcx: AlloyBuffer,
    conv_state: AlloyBuffer,
    conv1d_w: AlloyBuffer,
    conv_kernel_size: int,
    has_previous_state: bool,
) -> AlloyBuffer:
    """LFM2 gated warm-decode short-conv: b*x -> conv -> c* in one kernel,
    reading the b/c/x column-slices of `bcx` (B, 1, 3C) directly. Only the warm
    decode path reaches here (the patched forward gates on has_previous_state,
    True only at seq_len==1 with primed state)."""
    batch_size, seq_len, c3 = bcx.shape
    conv_dim = c3 // 3
    K = conv_kernel_size
    w_squeezed = conv1d_w.reshape((conv_dim, K))
    n_conv = batch_size * conv_dim
    out = _alloc_scratch((n_conv,), float32)
    _CONV1D_GATED_DECODE[((n_conv + 255) // 256,)](
        bcx.reshape((batch_size * c3,)),
        w_squeezed.reshape((conv_dim * K,)),
        conv_state,
        out,
        BATCH=batch_size,
        C=conv_dim,
        K=K,
    )
    return out.reshape((batch_size, 1, conv_dim))


def _short_conv_update_handler(
    bx: AlloyBuffer,
    conv_state: AlloyBuffer,
    conv1d_w: AlloyBuffer,
    conv_kernel_size: int,
    has_previous_state: bool,
    real_len: AlloyBuffer | None = None,
) -> AlloyBuffer:
    """Causal depthwise Conv1d with rolling state, in (B, S, C) layout.

    Prefill (S > 1): reads `conv_state` as the K-1 pre-context, emits the conv
    output, then `conv_state_save_real_pos` rewrites `conv_state` from the last
    K REAL positions (so a padded final chunk carries the right window into the
    next chunk / decode). Decode (S == 1, warm): the decode kernel rolls
    `conv_state` and writes the new input itself.

    Weight (C, 1, K) → (C, K). conv_state is the cache buffer (Tensor(c!) —
    mutable) and is passed flat; reshaping it would spawn a view slot the alloy
    backend treats as an intermediate, dropping the kernel's writeback.
    """
    batch_size, seq_len, conv_dim = bx.shape
    K = conv_kernel_size
    w_squeezed = conv1d_w.reshape((conv_dim, K))

    if has_previous_state and seq_len == 1:
        N_conv = batch_size * conv_dim
        out = _alloc_scratch((N_conv,), float32)
        _CONV1D_DECODE[((N_conv + 255) // 256,)](
            bx.reshape((N_conv,)),
            w_squeezed.reshape((conv_dim * K,)),
            conv_state,
            out,
            BATCH=batch_size,
            C=conv_dim,
            K=K,
        )
        return out.reshape((batch_size, 1, conv_dim))

    N_conv = batch_size * conv_dim * seq_len
    out = _alloc_scratch((N_conv,), float32)
    # 2D grid (B*S, ceil(C/256)): position on axis-0 so the grid-shrink recipe
    # shrinks the conv to the real prompt length; axis-1 tiles the channels.
    _CONV1D_PREFILL[(batch_size * seq_len, (conv_dim + 255) // 256)](
        bx.reshape((N_conv,)),
        w_squeezed.reshape((conv_dim * K,)),
        _tape_dummy(),
        conv_state,
        out,
        BATCH=batch_size,
        C=conv_dim,
        S=seq_len,
        K=K,
        SAVE_TAPE=0,
    )
    # Save conv_state from the last K REAL positions [real_len-K, real_len-1].
    # Pass `out` as dep_in to force this AFTER the conv kernel (else topo-sort
    # may run it first and the conv's pre-context reads pick up real-position
    # bytes). real_len is always supplied for prefill (computed from the layer
    # pad mask), so the conv state is carried to the next chunk / decode.
    if real_len is not None and seq_len > 1:
        _CONV_STATE_REAL[((batch_size * conv_dim * K + 255) // 256,)](
            bx.reshape((N_conv,)),
            real_len.reshape((1,)),
            out,
            conv_state,
            BATCH=batch_size,
            C=conv_dim,
            S=seq_len,
            K=K,
        )
    return out.reshape((batch_size, seq_len, conv_dim))
