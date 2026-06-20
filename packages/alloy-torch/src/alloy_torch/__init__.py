"""Alloy-Torch: PyTorch interop and torch.compile backend registration."""

import torch._dynamo
from torch._dynamo.backends import registry as _backend_registry

from alloy_torch.backend import alloy_backend
from alloy_torch.interop import (
    buffer_to_tensor,
    tensor_to_buffer,
    tensor_to_numpy,
)
# Static shape specialization. With `automatic_dynamic_shapes=True` a dim that
# changes between calls (prompt_len) gets marked dynamic on the 3rd recompile and
# Dynamo retraces with symbolic shapes, which the alloy backend mislowers to
# silently-wrong output. Static specialization recompiles per shape but is correct.
torch._dynamo.config.automatic_dynamic_shapes = False
# Models put `self.layer_idx` on the attention module; Dynamo treats nn.Module int
# attributes as static, so each layer is a distinct specialisation. Without room in
# the recompile cache, late layers fall back to eager → custom-op CPU-dispatch
# failures. (`allow_unspec_int_on_nn_module=True` breaks
# `past_key_values.layers[self.layer_idx]` — list indexing needs a concrete int.)
torch._dynamo.config.cache_size_limit = 512
# Bigger contexts blow past dynamo's default accumulated_recompile_limit; once
# exceeded dynamo silently falls back to eager, and `gguf_q8_0_mm` has only
# MPS/Meta dispatch keys so the CPU backend crashes. Match cache_size_limit.
torch._dynamo.config.accumulated_recompile_limit = 4096
# Treat scalar-tensor values (cache.cumulative_length) as symbolic, not specialised
# on the literal int — else every warm-prefill turn recompiles at its Q_START_POS
# even though the alloy backend rebinds input storage per call.
torch._dynamo.config.specialize_int = False

# capture_scalar_outputs=True bakes scalar tensor reads (cumulative_length.item())
# into the graph as the first call's int → per-Q_START_POS recompiles. False lets
# the alloy backend lower them as runtime values in offset arithmetic.
torch._dynamo.config.capture_scalar_outputs = False

if "alloy" not in _backend_registry._COMPILER_FNS:
    torch._dynamo.register_backend(alloy_backend, name="alloy")

__all__ = [
    "alloy_backend",
    "buffer_to_tensor",
    "tensor_to_buffer",
    "tensor_to_numpy",
]
