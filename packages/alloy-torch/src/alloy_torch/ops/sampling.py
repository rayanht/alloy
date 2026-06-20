"""FX handler for the on-GPU categorical sampler.

Lowers `torch.ops.alloy.sample_categorical` to the `std.sample_categorical`
kernel: reduces the last (vocab) dim of `logits` to a sampled token id, so the
decode plan's final kernel can be the sampler instead of `argmax_last_dim`.
"""

import math
from typing import cast

from alloy._compiler.dtypes import float32, int64
from alloy._dispatch.buf_utils import _alloc_aligned, _alloc_scratch
from alloy._dispatch.kernel import KernelFunction
from alloy._runtime.alloy_buffer import AlloyBuffer
from alloy.std.sampling import (
    SAMPLE_SPLITS,
    sample_categorical_combine,
    sample_categorical_split,
)

_sample_split = cast(KernelFunction, sample_categorical_split)
_sample_combine = cast(KernelFunction, sample_categorical_combine)


def _sample_categorical_handler(
    logits: AlloyBuffer,
    position: AlloyBuffer,
    seed: AlloyBuffer,
    params: AlloyBuffer,
) -> AlloyBuffer:
    vocab = logits.shape[-1]
    rows = math.prod(logits.shape[:-1]) if logits.ndim > 1 else 1
    flat = logits.reshape((rows, vocab))
    # Split-K argmax/sampler: SAMPLE_SPLITS threadgroups each reduce a vocab
    # slice to a (value, index) partial, then combine reduces the partials to the
    # token — a single TG can't saturate bandwidth over the ~150k vocab. params[4]
    # (n_splits) picks the active split count per request: 1 for top-k/p/min-p
    # (split 0 does the global bisection), SAMPLE_SPLITS otherwise; idle splits
    # emit a sentinel partial. Explicit grids: no dispatch_spec on these kernels.
    partial_val = _alloc_scratch((rows, SAMPLE_SPLITS), float32)
    partial_idx = _alloc_scratch((rows, SAMPLE_SPLITS), float32)
    out = _alloc_aligned((rows,), int64)
    _sample_split[(rows, SAMPLE_SPLITS)](flat, position, seed, params, partial_val, partial_idx)
    result = _sample_combine[(rows,)](partial_val, partial_idx, out)
    return result.reshape(logits.shape[:-1])
