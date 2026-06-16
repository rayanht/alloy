"""Decomposition helpers for the Alloy torch.compile backend."""

from __future__ import annotations

import torch
from torch._decomp import core_aten_decompositions

PRESERVED_OPS = {
    torch.ops.aten.native_layer_norm.default,
    torch.ops.aten._softmax.default,
    torch.ops.aten.scaled_dot_product_attention.default,
    torch.ops.aten._scaled_dot_product_attention_math.default,
    torch.ops.aten.gelu.default,
    torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default,
    # Keep select_backward intact so the handler dispatches a direct
    # slice-scatter-into-zeros rather than the default eq/where/add chain,
    # which mis-scatters on small head_dim SDPA backward paths.
    torch.ops.aten.select_backward.default,
}


def get_alloy_decompositions(*, training: bool = False):
    """Return an AOT decomposition table tuned for the current Alloy op set."""
    decomps = dict(core_aten_decompositions())
    preserved = PRESERVED_OPS
    for op in preserved:
        decomps.pop(op, None)
    return decomps
