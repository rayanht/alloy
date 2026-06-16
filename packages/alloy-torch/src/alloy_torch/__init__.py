"""Alloy-Torch: PyTorch interop and torch.compile backend registration."""

import torch._dynamo
from torch._dynamo.backends import registry as _backend_registry

from alloy_torch.backend import alloy_backend
from alloy_torch.interop import (
    buffer_to_tensor,
    tensor_to_buffer,
    tensor_to_numpy,
)
# Force static shape specialization. With Dynamo's default
# `automatic_dynamic_shapes=True`, a tensor dim that changes between calls
# (e.g., prompt_len across different requests) gets marked dynamic on the
# 3rd recompile and Dynamo retraces with symbolic shapes. The alloy backend
# does not yet handle the symbolic-shape graph correctly and silently emits
# wrong output (first generated token is garbage, downstream tokens loop).
# Static specialization recompiles per shape but produces correct results;
# the bigger cache size limit avoids hitting the fallback path for HTTP
# servers that legitimately see many distinct prompt lengths.
torch._dynamo.config.automatic_dynamic_shapes = False
# gemma3 (and most decoder-stack models) put `self.layer_idx` on the
# attention module — Dynamo treats nn.Module integer attributes as
# static, so each of gemma3:1b's 26 layers becomes a distinct
# specialisation. Without enough room in the recompile cache, late
# layers fall back to eager → custom-op CPU-dispatch failures. Bump
# the limit instead of unspec'ing — `allow_unspec_int_on_nn_module=True`
# breaks `past_key_values.layers[self.layer_idx]` because Python list
# indexing requires a concrete int.
torch._dynamo.config.cache_size_limit = 512
# Bigger contexts (qwen3.5 ctx=4096 has 8 prefill buckets × 18 decoder layers
# × HF's per-layer recompile guards) blow past dynamo's default 256
# accumulated_recompile_limit. Once exceeded, dynamo silently falls back
# to eager — and `gguf_q8_0_mm` only has MPS/Meta dispatch keys, so eager
# tries the CPU backend and crashes with `Could not run alloy::gguf_q8_0_mm
# with arguments from the 'CPU' backend`. Raise the limit to match
# cache_size_limit so prefill compile completes.
torch._dynamo.config.accumulated_recompile_limit = 4096
# Treat scalar-tensor values (e.g. cache.cumulative_length, which carries
# the prefix length per warm-prefill request) as symbolic instead of
# specialising on the literal int. Without this, every conversation's
# first warm-prefill turn triggers a fresh Dynamo specialisation at that
# request's specific Q_START_POS — even though every plan is functionally
# identical and the alloy backend can rebind input storage per call.
torch._dynamo.config.specialize_int = False

# capture_scalar_outputs=True bakes scalar tensor reads (e.g.
# cumulative_length.item()) into the graph as the specific int seen on
# the first call, which causes per-Q_START_POS recompiles on multi-turn
# warm prefill. False lets Dynamo treat those scalars as symbolic; the
# alloy backend lowers them as runtime values, and they only appear in
# offset arithmetic.
torch._dynamo.config.capture_scalar_outputs = False

if "alloy" not in _backend_registry._COMPILER_FNS:
    torch._dynamo.register_backend(alloy_backend, name="alloy")

__all__ = [
    "alloy_backend",
    "buffer_to_tensor",
    "tensor_to_buffer",
    "tensor_to_numpy",
]
