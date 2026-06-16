"""Reduction, softmax, and cross-entropy kernels."""

import alloy as al
from alloy._compiler.op_registry import (
    _reduce_num_groups,
    resolve_reduction_variant,
)
from alloy._compiler.trace import _active as _trace_active
from alloy._runtime.alloy_buffer import AlloyBuffer
from alloy._runtime.convert import to_alloy_buffer


@al.tunable(BLOCK_SIZE=[128, 256, 512, 1024])
@al.kernel
def softmax(x, out: al.output, BLOCK_SIZE: al.constexpr = 256):
    M, N = x.shape
    row = al.program_id(0)
    row_max = -1e30
    for _ki in range(0, N, BLOCK_SIZE):
        offs = _ki + al.arange(0, BLOCK_SIZE)
        mask = offs < N
        v = al.load(x + row * N + offs, mask=mask, other=-1e30)
        row_max = al.maximum(row_max, v)
    row_max = al.max(row_max)
    row_sum = 0.0
    for _ki in range(0, N, BLOCK_SIZE):
        offs = _ki + al.arange(0, BLOCK_SIZE)
        mask = offs < N
        v = al.load(x + row * N + offs, mask=mask, other=-1e30)
        row_sum = row_sum + al.exp(v - row_max)
    row_sum = al.sum(row_sum)
    inv_sum = 1.0 / row_sum
    for _ki in range(0, N, BLOCK_SIZE):
        offs = _ki + al.arange(0, BLOCK_SIZE)
        mask = offs < N
        v = al.load(x + row * N + offs, mask=mask, other=-1e30)
        al.store(out + row * N + offs, al.exp(v - row_max) * inv_sum, mask=mask)


@al.kernel
def cross_entropy(logits, labels, loss: al.output, BLOCK_SIZE: al.constexpr = 256):
    M, N = logits.shape
    row = al.program_id(0)
    row_max = -1e30
    for _ki in range(0, N, BLOCK_SIZE):
        offs = _ki + al.arange(0, BLOCK_SIZE)
        mask = offs < N
        v = al.load(logits + row * N + offs, mask=mask, other=-1e30)
        row_max = al.maximum(row_max, v)
    row_max = al.max(row_max)
    row_sum = 0.0
    for _ki in range(0, N, BLOCK_SIZE):
        offs = _ki + al.arange(0, BLOCK_SIZE)
        mask = offs < N
        v = al.load(logits + row * N + offs, mask=mask, other=-1e30)
        row_sum = row_sum + al.exp(v - row_max)
    row_sum = al.sum(row_sum)
    label = al.load(labels + row)
    logit_at_label = al.load(logits + row * N + label)
    al.store(loss + row, al.log(row_sum) + row_max - logit_at_label)


@al.tunable(BLOCK_SIZE=[256, 512, 1024])
@al.kernel
def argmax_last_dim(x, out: al.output, BLOCK_SIZE: al.constexpr = 512):
    M, N = x.shape
    row = al.program_id(0)
    row_max = -1e30
    for _ki in range(0, N, BLOCK_SIZE):
        offs = _ki + al.arange(0, BLOCK_SIZE)
        mask = offs < N
        v = al.load(x + row * N + offs, mask=mask, other=-1e30)
        row_max = al.maximum(row_max, v)
    row_max = al.max(row_max)

    best_idx = 1e30
    for _ki in range(0, N, BLOCK_SIZE):
        offs = _ki + al.arange(0, BLOCK_SIZE)
        mask = offs < N
        v = al.load(x + row * N + offs, mask=mask, other=-1e30)
        candidate = al.where((v == row_max) & mask, al.cast(offs, al.float32), 1e30)
        best_idx = al.minimum(best_idx, candidate)
    al.store(out + row, al.cast(al.min(best_idx), al.int64))


@al.kernel
def cross_entropy_fused_fwd(
    logits,
    labels,
    loss: al.output,
    lse: al.output,
    IGNORE_INDEX: al.constexpr = -100,
    BLOCK_SIZE: al.constexpr = 256,
):
    """Single-pass CE forward: streams logits once using online softmax
    (running max + sum), emits per-row loss AND per-row logsumexp for the
    backward pass. Replaces the decomposed `_log_softmax + nll_loss_forward`
    chain that materializes multiple (N, V) intermediates.

    Per-row loss is 0 when label == IGNORE_INDEX; the caller does mean
    reduction over valid rows in Python.
    """
    M, N = logits.shape
    row = al.program_id(0)
    label = al.load(labels + row)
    # Online max + sum-of-exp in ONE pass over V.
    m = -1e30
    l = 0.0  # noqa: E741
    for _ki in range(0, N, BLOCK_SIZE):
        offs = _ki + al.arange(0, BLOCK_SIZE)
        mask = offs < N
        v = al.load(logits + row * N + offs, mask=mask, other=-1e30)
        bmax = al.max(v)
        mn = al.maximum(m, bmax)
        alpha = al.exp(m - mn)
        # Zero out lanes past N so they don't contribute to sum
        v_in = al.where(mask, v, -1e30)
        p = al.exp(v_in - mn)
        l = l * alpha + al.sum(p)  # noqa: E741
        m = mn
    row_lse = m + al.log(l)
    al.store(lse + row, row_lse)
    # Load logit at label (clamp to 0 when ignored to avoid OOB indexing).
    ignored = label == IGNORE_INDEX
    safe_label = al.where(ignored, 0, label)
    logit_at_label = al.load(logits + row * N + safe_label)
    row_loss = row_lse - logit_at_label
    row_loss = al.where(ignored, 0.0, row_loss)
    al.store(loss + row, row_loss)


@al.kernel
def cross_entropy_fused_bwd(
    logits,
    labels,
    lse,
    grad_scale,  # scalar (1,): upstream grad already divided by num_valid
    d_logits: al.output,
    IGNORE_INDEX: al.constexpr = -100,
    BLOCK_SIZE: al.constexpr = 256,
):
    """Single-pass CE backward: d_logits[r, c] = grad_scale * (softmax(logits)[r,c] - one_hot(labels[r], c))
    for valid rows (label != IGNORE_INDEX), else 0.

    softmax is recomputed inline from the saved lse so we never materialize
    the full (N, V) softmax tensor.
    """
    M, N = logits.shape
    row = al.program_id(0)
    label = al.load(labels + row)
    row_lse = al.load(lse + row)
    g = al.load(grad_scale + 0)  # scalar broadcast
    ignored = label == IGNORE_INDEX
    # g = 0 if this row is ignored → d_logits row = 0
    g = al.where(ignored, 0.0, g)
    for _ki in range(0, N, BLOCK_SIZE):
        offs = _ki + al.arange(0, BLOCK_SIZE)
        mask = offs < N
        v = al.load(logits + row * N + offs, mask=mask, other=0.0)
        # softmax(v) = exp(v - lse)
        p = al.exp(v - row_lse)
        # subtract one-hot(label) inside the block
        is_label = offs == label
        d = p - al.where(is_label, 1.0, 0.0)
        al.store(d_logits + row * N + offs, d * g, mask=mask)


# --- Reduction kernel templates (closure-based, no string exec) ---


def _make_flat_reduce(name, init_val, combine_fn, finalize_fn, other_val=0.0, mean=False):
    @al.kernel
    def k(
        x,
        out: al.output,
        N: al.constexpr,
        N_PER_GROUP: al.constexpr,
        BLOCK_SIZE: al.constexpr = 256,
    ):
        pid = al.program_id(0)
        base = pid * N_PER_GROUP
        acc = init_val
        for _ki in range(0, N_PER_GROUP, BLOCK_SIZE):
            offs = base + _ki + al.arange(0, BLOCK_SIZE)
            mask = offs < N
            v = al.load(x + offs, mask=mask, other=other_val)
            acc = combine_fn(acc, v)
        result = finalize_fn(acc)
        if mean:
            result = result / N
        al.store(out + pid, result)

    k.name = name
    k._source = f"# {name}\n" + k._source
    return k


def _make_row_reduce(name, init_val, combine_fn, finalize_fn, other_val=0.0, mean=False):
    @al.kernel
    def k(x, out: al.output, M: al.constexpr, N: al.constexpr, BLOCK_SIZE: al.constexpr = 256):
        row = al.program_id(0)
        acc = init_val
        for _ki in range(0, N, BLOCK_SIZE):
            offs = _ki + al.arange(0, BLOCK_SIZE)
            mask = offs < N
            v = al.load(x + row * N + offs, mask=mask, other=other_val)
            acc = combine_fn(acc, v)
        result = finalize_fn(acc)
        if mean:
            result = result / N
        al.store(out + row, result)

    k.name = name
    k._source = f"# {name}\n" + k._source
    return k


def _make_col_reduce(name, init_val, combine_fn, finalize_fn, other_val=0.0, mean=False):
    @al.kernel
    def k(x, out: al.output, M: al.constexpr, N: al.constexpr, BLOCK_SIZE: al.constexpr = 256):
        col = al.program_id(0)
        acc = init_val
        for _ki in range(0, M, BLOCK_SIZE):
            offs = _ki + al.arange(0, BLOCK_SIZE)
            mask = offs < M
            v = al.load(x + offs * N + col, mask=mask, other=other_val)
            acc = combine_fn(acc, v)
        result = finalize_fn(acc)
        if mean:
            result = result / M
        al.store(out + col, result)

    k.name = name
    k._source = f"# {name}\n" + k._source
    return k


# --- Reduction kernel instances ---

_reduction_kernels = {
    "reduce_sum": _make_flat_reduce("reduce_sum", 0.0, lambda a, v: a + v, lambda a: al.sum(a)),
    "reduce_max": _make_flat_reduce(
        "reduce_max", -1e30, lambda a, v: al.maximum(a, v), lambda a: al.max(a), -1e30
    ),
    "reduce_min": _make_flat_reduce(
        "reduce_min", 1e30, lambda a, v: al.minimum(a, v), lambda a: al.min(a), 1e30
    ),
    "mean": _make_flat_reduce("mean", 0.0, lambda a, v: a + v, lambda a: al.sum(a), mean=True),
    "row_reduce_sum": _make_row_reduce(
        "row_reduce_sum", 0.0, lambda a, v: a + v, lambda a: al.sum(a)
    ),
    "row_reduce_max": _make_row_reduce(
        "row_reduce_max", -1e30, lambda a, v: al.maximum(a, v), lambda a: al.max(a), -1e30
    ),
    "row_reduce_min": _make_row_reduce(
        "row_reduce_min", 1e30, lambda a, v: al.minimum(a, v), lambda a: al.min(a), 1e30
    ),
    "row_mean": _make_row_reduce(
        "row_mean", 0.0, lambda a, v: a + v, lambda a: al.sum(a), mean=True
    ),
    "col_reduce_sum": _make_col_reduce(
        "col_reduce_sum", 0.0, lambda a, v: a + v, lambda a: al.sum(a)
    ),
    "col_reduce_max": _make_col_reduce(
        "col_reduce_max", -1e30, lambda a, v: al.maximum(a, v), lambda a: al.max(a), -1e30
    ),
    "col_reduce_min": _make_col_reduce(
        "col_reduce_min", 1e30, lambda a, v: al.minimum(a, v), lambda a: al.min(a), 1e30
    ),
    "col_mean": _make_col_reduce(
        "col_mean", 0.0, lambda a, v: a + v, lambda a: al.sum(a), mean=True
    ),
    "row_reduce_any": _make_row_reduce(
        "row_reduce_any", 0, lambda a, v: al.maximum(a, v), lambda a: al.max(a), 0
    ),
}


@al.kernel
def _mean_div(x_ptr, out_ptr: al.output, N_DIV: al.constexpr):
    """Divide a scalar by N for two-pass mean."""
    offs = al.arange(0, 1)
    v = al.load(x_ptr + offs)
    al.store(out_ptr + offs, v / N_DIV)


def _make_reduce_dispatch(op_name):
    """Create a dispatch function for a reduction op (e.g. al.reduce_sum)."""

    def dispatch(x, *, axis=None, BLOCK_SIZE=256, **kwargs):
        # Inside a trace: call the variant kernel directly for inlining
        if _trace_active():
            M = None if axis is None else 1  # just needs non-None to trigger row/col
            variant = resolve_reduction_variant(op_name, M=M, axis=axis if axis is not None else 1)
            return _reduction_kernels[variant](x, **kwargs)

        is_lazy = isinstance(x, AlloyBuffer)
        if is_lazy:
            x_shape = x.shape
            x_size = x.size
            x_input = x
        else:
            x_input = to_alloy_buffer(x)
            x_shape = x_input.shape
            x_size = x_input.size

        if axis is not None:
            M, N = x_shape if len(x_shape) == 2 else (1, x_size)
        else:
            M = None

        variant_key = resolve_reduction_variant(op_name, M=M, axis=axis if axis is not None else 1)
        kern = _reduction_kernels[variant_key]

        if M is not None:
            return kern(x_input, M=M, N=N, BLOCK_SIZE=BLOCK_SIZE, **kwargs)

        # Flat reduction
        N_val = x_size
        if not is_lazy:
            x_input = x_input.ravel()

        # Compute N_PER_GROUP (elements per threadgroup)
        if N_val > BLOCK_SIZE * 512:
            n_groups, n_per_group = _reduce_num_groups(N_val, BLOCK_SIZE)
        else:
            n_per_group = N_val
        n_per_group = ((n_per_group + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
        if n_per_group == N_val:
            n_per_group += BLOCK_SIZE

        # Two-pass: if N is large enough for multi-group, dispatch pass 1 → partials → pass 2
        if N_val > BLOCK_SIZE * 512:
            if op_name == "mean":
                sum_kern = _reduction_kernels["reduce_sum"]
                partials = sum_kern(
                    x_input, N=N_val, N_PER_GROUP=n_per_group, BLOCK_SIZE=BLOCK_SIZE
                )
                n_per_group_2 = ((n_groups + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
                if n_per_group_2 == n_groups:
                    n_per_group_2 += BLOCK_SIZE
                total = sum_kern(
                    partials.ravel(), N=n_groups, N_PER_GROUP=n_per_group_2, BLOCK_SIZE=BLOCK_SIZE
                )
                return _mean_div[1](total.ravel(), N_DIV=N_val)
            else:
                partials = kern(
                    x_input, N=N_val, N_PER_GROUP=n_per_group, BLOCK_SIZE=BLOCK_SIZE, **kwargs
                )
                n_per_group_2 = ((n_groups + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
                if n_per_group_2 == n_groups:
                    n_per_group_2 += BLOCK_SIZE
                return kern(
                    partials.ravel(),
                    N=n_groups,
                    N_PER_GROUP=n_per_group_2,
                    BLOCK_SIZE=BLOCK_SIZE,
                    **kwargs,
                )

        return kern(x_input, N=N_val, N_PER_GROUP=n_per_group, BLOCK_SIZE=BLOCK_SIZE, **kwargs)

    dispatch.__name__ = op_name
    return dispatch


reduce_sum = _make_reduce_dispatch("reduce_sum")
reduce_max = _make_reduce_dispatch("reduce_max")
reduce_min = _make_reduce_dispatch("reduce_min")
reduce_any = _make_reduce_dispatch("reduce_any")
mean = _make_reduce_dispatch("mean")
