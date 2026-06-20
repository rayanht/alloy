"""Shared test helpers — reference implementations, kernel definitions, assertions."""

import alloy as al
import numpy as np
from alloy._dispatch.kernel import KernelFunction
from alloy._runtime.metal import default_dispatcher


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------

def ref_softmax(x):
    ex = np.exp(x - x.max(axis=-1, keepdims=True))
    return ex / ex.sum(axis=-1, keepdims=True)


def ref_layernorm(x, gamma=None, beta=None, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    normed = (x - mean) / np.sqrt(var + eps)
    if gamma is not None:
        normed = gamma * normed
    if beta is not None:
        normed = normed + beta
    return normed


def ref_rms_norm(x, w, eps=1e-5):
    rms = np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps)
    return w * x / rms


def ref_attention(Q, K, V, causal=False, scale=None):
    D = Q.shape[-1]
    s = scale or (1.0 / np.sqrt(D))
    S = Q @ K.T * s
    if causal:
        N = Q.shape[0]
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        S = np.where(mask, -1e30, S)
    S_max = S.max(axis=-1, keepdims=True)
    P = np.exp(S - S_max)
    P = P / P.sum(axis=-1, keepdims=True)
    return P @ V


def ref_attention_batched(Q, K, V, causal=False):
    BH = Q.shape[0]
    out = np.zeros_like(Q)
    for i in range(BH):
        out[i] = ref_attention(Q[i], K[i], V[i], causal=causal)
    return out


def ref_gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def ref_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def ref_cross_entropy(logits, labels):
    row_max = logits.max(axis=1, keepdims=True)
    log_sum_exp = row_max.squeeze() + np.log(np.exp(logits - row_max).sum(axis=1))
    logit_at_label = logits[np.arange(len(labels)), labels]
    return log_sum_exp - logit_at_label


def ref_rope(x, base=10000.0):
    M, D = x.shape
    out = np.zeros_like(x)
    for pos in range(M):
        for j in range(D // 2):
            freq = 1.0 / (base ** (2 * j / D))
            angle = pos * freq
            c, s = np.cos(angle), np.sin(angle)
            out[pos, 2 * j] = x[pos, 2 * j] * c - x[pos, 2 * j + 1] * s
            out[pos, 2 * j + 1] = x[pos, 2 * j] * s + x[pos, 2 * j + 1] * c
    return out


# ---------------------------------------------------------------------------
# Test kernels
# ---------------------------------------------------------------------------

@al.kernel
def k_scale(x, out: al.output, N: al.constexpr):
    pid = al.program_id(0)
    offs = pid * 1024 + al.arange(0, 1024)
    mask = offs < N
    al.store(out + offs, al.load(x + offs, mask=mask) * 2.0, mask=mask)


@al.kernel
def k_bias(x, out: al.output, N: al.constexpr):
    pid = al.program_id(0)
    offs = pid * 1024 + al.arange(0, 1024)
    mask = offs < N
    al.store(out + offs, al.load(x + offs, mask=mask) + 1.0, mask=mask)


@al.kernel
def k_relu(x, out: al.output, N: al.constexpr):
    pid = al.program_id(0)
    offs = pid * 1024 + al.arange(0, 1024)
    mask = offs < N
    al.store(out + offs, al.maximum(al.load(x + offs, mask=mask), 0.0), mask=mask)


@al.kernel
def k_gelu(x, out: al.output, N: al.constexpr):
    pid = al.program_id(0)
    offs = pid * 1024 + al.arange(0, 1024)
    mask = offs < N
    al.store(out + offs, al.gelu(al.load(x + offs, mask=mask)), mask=mask)


@al.kernel
def k_sigmoid(x, out: al.output, N: al.constexpr):
    pid = al.program_id(0)
    offs = pid * 1024 + al.arange(0, 1024)
    mask = offs < N
    v = al.load(x + offs, mask=mask)
    al.store(out + offs, 1.0 / (1.0 + al.exp(-v)), mask=mask)


@al.kernel
def k_add(x, y, out: al.output, N: al.constexpr):
    pid = al.program_id(0)
    offs = pid * 1024 + al.arange(0, 1024)
    mask = offs < N
    al.store(out + offs, al.load(x + offs, mask=mask) + al.load(y + offs, mask=mask), mask=mask)


@al.kernel
def k_mul(x, y, out: al.output, N: al.constexpr):
    pid = al.program_id(0)
    offs = pid * 1024 + al.arange(0, 1024)
    mask = offs < N
    al.store(out + offs, al.load(x + offs, mask=mask) * al.load(y + offs, mask=mask), mask=mask)


# ---------------------------------------------------------------------------
# Kernel factory for parametrized unary tests
# ---------------------------------------------------------------------------

# Map of al.<op_name> → @al.kernel function.
_UNARY_KERNELS: dict[str, KernelFunction] = {}

def _register_unary(name, op_fn):
    """Register a unary kernel. op_fn takes (al, x) and returns the result."""
    @al.kernel
    def _k(x, out: al.output, N: al.constexpr):
        pid = al.program_id(0)
        offs = pid * 1024 + al.arange(0, 1024)
        mask = offs < N
        al.store(out + offs, op_fn(al.load(x + offs, mask=mask)), mask=mask)
    _k.name = f"k_{name}"
    _k._source = f"# k_{name}\n" + _k._source
    _k._init_metadata()
    _UNARY_KERNELS[name] = _k

_register_unary("exp", lambda v: al.exp(v))
_register_unary("log", lambda v: al.log(v))
_register_unary("sqrt", lambda v: al.sqrt(v))
_register_unary("rsqrt", lambda v: al.rsqrt(v))
_register_unary("tanh", lambda v: al.tanh(v))
_register_unary("sin", lambda v: al.sin(v))
_register_unary("cos", lambda v: al.cos(v))
_register_unary("abs", lambda v: al.abs(v))
_register_unary("sigmoid", lambda v: 1.0 / (1.0 + al.exp(-v)))
_register_unary("relu", lambda v: al.maximum(v, 0.0))
_register_unary("gelu", lambda v: al.gelu(v))
_register_unary("ceil", lambda v: al.ceil(v))
_register_unary("floor", lambda v: al.floor(v))
_register_unary("exp2", lambda v: al.exp2(v))
_register_unary("log2", lambda v: al.log2(v))

def get_unary_kernel(name: str) -> KernelFunction:
    return _UNARY_KERNELS[name]


# ---------------------------------------------------------------------------
# Dispatch count helper
# ---------------------------------------------------------------------------

def assert_dispatches(expected: int, fn, label: str = ""):
    """Run fn(), assert it produced exactly `expected` GPU dispatches."""
    d = default_dispatcher()
    before = d.dispatch_count
    result = fn()
    actual = d.dispatch_count - before
    assert actual == expected, (
        f"Expected {expected} dispatch(es){f' ({label})' if label else ''}, got {actual}"
    )
    return result
