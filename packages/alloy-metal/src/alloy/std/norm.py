"""Normalization kernels."""

import alloy as al


@al.tunable(BLOCK_SIZE=[128, 256, 512, 1024])
@al.kernel
def layernorm(
    x, gamma, beta, out: al.output, EPS: al.constexpr = 1e-5, BLOCK_SIZE: al.constexpr = 256
):
    M, N = x.shape
    row = al.program_id(0)
    row_sum = 0.0
    for _ki in range(0, N, BLOCK_SIZE):
        offs = _ki + al.arange(0, BLOCK_SIZE)
        mask = offs < N
        v = al.load(x + row * N + offs, mask=mask, other=0.0)
        row_sum = row_sum + v
    mean = al.sum(row_sum) / N
    var_sum = 0.0
    for _ki in range(0, N, BLOCK_SIZE):
        offs = _ki + al.arange(0, BLOCK_SIZE)
        mask = offs < N
        v = al.load(x + row * N + offs, mask=mask, other=0.0)
        d = (v - mean) * mask
        var_sum = var_sum + d * d
    inv_std = al.rsqrt(al.sum(var_sum) / N + EPS)
    for _ki in range(0, N, BLOCK_SIZE):
        offs = _ki + al.arange(0, BLOCK_SIZE)
        mask = offs < N
        v = al.load(x + row * N + offs, mask=mask, other=0.0)
        g = al.load(gamma + offs, mask=mask, other=0.0)
        b = al.load(beta + offs, mask=mask, other=0.0)
        normed = (v - mean) * inv_std
        al.store(out + row * N + offs, g * normed + b, mask=mask)


@al.tunable(BLOCK_SIZE=[128, 256, 512, 1024])
@al.kernel
def rms_norm(
    x,
    weight,
    out: al.output,
    rrms_out: al.output,
    EPS: al.constexpr = 1e-6,
    BLOCK_SIZE: al.constexpr = 256,
):
    """RMSNorm with F32 accumulation. Emits per-row rsqrt as a second output
    so AOT autograd's saved-for-backward `rsqrt(mean(x^2)+eps)` can be sourced
    from this kernel rather than recomputed via a parallel decomposed chain."""
    M, N = x.shape
    row = al.program_id(0)
    sq_sum = 0.0
    for _ki in range(0, N, BLOCK_SIZE):
        offs = _ki + al.arange(0, BLOCK_SIZE)
        mask = offs < N
        v = al.cast(al.load(x + row * N + offs, mask=mask, other=0.0), al.float32)
        sq_sum = sq_sum + v * v
    rrms = al.rsqrt(al.sum(sq_sum) / N + EPS)
    al.store(rrms_out + row, rrms)
    for _ki in range(0, N, BLOCK_SIZE):
        offs = _ki + al.arange(0, BLOCK_SIZE)
        mask = offs < N
        v = al.cast(al.load(x + row * N + offs, mask=mask, other=0.0), al.float32)
        w = al.load(weight + offs, mask=mask, other=0.0)
        # Match eager Llama/Qwen RMSNorm precision exactly: truncate the
        # normalized value to the input dtype BEFORE the weight multiply.
        # `out` is bf16/fp16 typed so the store implicitly casts; rebinding
        # through `cast` collapses to the same precision but ensures the
        # subsequent `w * normed` mul happens in the narrow type rather than
        # being implicitly promoted (which would diverge from eager and amp
        # downstream errors when bias values are large — Qwen K bias path).
        normed = al.cast(v * rrms, w.dtype)
        al.store(out + row * N + offs, w * normed, mask=mask)


@al.tunable(BLOCK_SIZE=[128, 256, 512, 1024])
@al.kernel
def rms_norm_backward(
    x,
    dy,
    weight,
    rrms,
    dx: al.output,
    BLOCK_SIZE: al.constexpr = 256,
):
    """RMSNorm backward (dx only — LoRA freezes the norm weight, so dw is
    never requested by AOT in our common path).

    Math: with s = rrms = 1/sqrt(mean(x^2)+eps), per row
        T   = mean(dy * w * x)
        dx  = s * (dy * w - x * s^2 * T)

    Two-pass over the row: first reduces to T, second emits dx. Both passes
    re-read x/dy/w (caching them between passes only saves bandwidth at the
    cost of registers/shmem, and the inner loop is already memory-bound on
    Apple Silicon).
    """
    M, N = x.shape
    row = al.program_id(0)
    s = al.load(rrms + row)
    T = 0.0
    for _ki in range(0, N, BLOCK_SIZE):
        offs = _ki + al.arange(0, BLOCK_SIZE)
        mask = offs < N
        x_i = al.cast(al.load(x + row * N + offs, mask=mask, other=0.0), al.float32)
        dy_i = al.cast(al.load(dy + row * N + offs, mask=mask, other=0.0), al.float32)
        w_i = al.cast(al.load(weight + offs, mask=mask, other=0.0), al.float32)
        T = T + dy_i * w_i * x_i
    T_mean = al.sum(T) / N
    s2T = s * s * T_mean
    for _ki in range(0, N, BLOCK_SIZE):
        offs = _ki + al.arange(0, BLOCK_SIZE)
        mask = offs < N
        x_i = al.cast(al.load(x + row * N + offs, mask=mask, other=0.0), al.float32)
        dy_i = al.cast(al.load(dy + row * N + offs, mask=mask, other=0.0), al.float32)
        w_i = al.cast(al.load(weight + offs, mask=mask, other=0.0), al.float32)
        al.store(dx + row * N + offs, s * (dy_i * w_i - x_i * s2T), mask=mask)
