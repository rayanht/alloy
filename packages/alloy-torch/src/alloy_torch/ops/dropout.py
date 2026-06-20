"""FX handlers for `aten.native_dropout` and its backward.

Inverted dropout on GPU. The keep mask is a counter-RNG hash of (seed, layer,
index); `seed` is a 1-element buffer redrawn from the torch global generator
once per forward (so `torch.manual_seed` controls the masks), written by
`refresh_dropout_seed`.
"""

from typing import cast

import torch

from alloy._compiler.dtypes import int32
from alloy._dispatch.buf_utils import _alloc_aligned, _alloc_scratch
from alloy._dispatch.dispatch import _engine
from alloy._dispatch.kernel import KernelFunction
from alloy._runtime.alloy_buffer import AlloyBuffer
from alloy.std.sampling import dropout_mask_apply

from alloy_torch.ops.casting import _to_copy
from alloy_torch.ops.common import _numel
from alloy_torch.ops.creation import _full

_dropout_kernel = cast(KernelFunction, dropout_mask_apply)

# One shared seed buffer for every dropout site, refreshed each forward.
_DROPOUT_SEED: AlloyBuffer | None = None
# Monotonic per-site offset so stacked dropout layers draw decorrelated masks
# from the same per-forward seed.
_layer_offset = 0


def _dropout_seed_buffer() -> AlloyBuffer:
    global _DROPOUT_SEED
    if _DROPOUT_SEED is None:
        _DROPOUT_SEED = _alloc_aligned((1,), int32)
        _DROPOUT_SEED.write_scalar(0)
        # Untrack so the compiled plan binds it as a WEIGHT and reads its live value.
        _engine.untrack_alloc(_DROPOUT_SEED.base_ptr)
    return _DROPOUT_SEED


def refresh_dropout_seed() -> None:
    """Draw a fresh seed from the torch global generator into the shared buffer.
    Drawing through torch keeps dropout under `torch.manual_seed`. Rewinds the
    per-site offset so each trace bakes the same offset for a given layer
    position."""
    global _layer_offset
    _layer_offset = 0
    _dropout_seed_buffer().write_scalar(int(torch.randint(0, 2**31 - 1, (1,)).item()))


def _native_dropout(
    x: AlloyBuffer, p: float, train: bool | None
) -> tuple[AlloyBuffer, AlloyBuffer]:
    n = _numel(x.shape)
    if not train or p == 0.0:
        return x, _full(x.shape, 1, dtype=torch.bool)
    global _layer_offset
    offset = _layer_offset
    _layer_offset += 1
    out = _alloc_scratch(x.shape, x.dtype)
    mask = _alloc_scratch(x.shape, x.dtype)
    grid = ((n + 1023) // 1024,)
    _dropout_kernel[grid](
        x.contiguous(), _dropout_seed_buffer(), out, mask,
        P=float(p), SCALE=1.0 / (1.0 - p), OFFSET=offset, N=n,
    )
    return out.reshape(x.shape), mask.reshape(x.shape)


def _native_dropout_backward(
    grad: AlloyBuffer, mask: AlloyBuffer, scale: float
) -> AlloyBuffer:
    keep = _to_copy(mask, dtype=grad._dtype.to_torch_dtype())
    return grad * keep * float(scale)
