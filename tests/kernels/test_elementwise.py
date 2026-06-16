"""GPU correctness tests for elementwise operations."""

import alloy as al
import numpy as np
import pytest
from tests.helpers import get_unary_kernel, ref_gelu, ref_sigmoid, k_add


# ---------------------------------------------------------------------------
# Unary ops — parametrized
# ---------------------------------------------------------------------------

_UNARY_CASES = [
    ("exp",     np.exp,                      "any",      1e-5, 1e-5),
    ("log",     np.log,                      "positive", 1e-5, 1e-5),
    ("sqrt",    np.sqrt,                     "positive", 1e-5, 1e-5),
    ("rsqrt",   lambda x: 1 / np.sqrt(x),   "positive", 1e-4, 1e-4),
    ("tanh",    np.tanh,                     "any",      1e-5, 1e-5),
    ("sin",     np.sin,                      "any",      1e-5, 1e-5),
    ("cos",     np.cos,                      "any",      1e-5, 1e-5),
    ("abs",     np.abs,                      "any",      1e-5, 1e-5),
    ("sigmoid", ref_sigmoid,                 "any",      1e-5, 1e-5),
    ("relu",    lambda x: np.maximum(x, 0),  "any",      1e-5, 1e-5),
    ("gelu",    ref_gelu,                    "any",      1e-3, 1e-3),
    ("ceil",    np.ceil,                     "scaled",   1e-5, 1e-5),
    ("floor",   np.floor,                    "scaled",   1e-5, 1e-5),
    ("exp2",    np.exp2,                     "small",    1e-5, 1e-5),
    ("log2",    np.log2,                     "positive", 1e-5, 1e-5),
]


def _make_input(kind, N, rng):
    if kind == "positive":
        return np.abs(rng.standard_normal(N).astype(np.float32)) + 0.01
    if kind == "scaled":
        return rng.standard_normal(N).astype(np.float32) * 10
    if kind == "small":
        return rng.standard_normal(N).astype(np.float32) * 3
    return rng.standard_normal(N).astype(np.float32)


@pytest.mark.parametrize("op,np_fn,input_kind,atol,rtol",
                         _UNARY_CASES, ids=[c[0] for c in _UNARY_CASES])
def test_unary(op, np_fn, input_kind, atol, rtol):
    k = get_unary_kernel(op)
    N = 4096
    rng = np.random.default_rng(42)
    x = _make_input(input_kind, N, rng)
    out = np.zeros_like(x)
    result = k[(N + 1023) // 1024](x, out, N=N)
    np.testing.assert_allclose(result, np_fn(x), atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Binary ops
# ---------------------------------------------------------------------------

@al.kernel
def _k_maximum(x, y, out: al.output, N: al.constexpr):
    pid = al.program_id(0)
    offs = pid * 1024 + al.arange(0, 1024)
    mask = offs < N
    al.store(out + offs, al.maximum(al.load(x + offs, mask=mask), al.load(y + offs, mask=mask)), mask=mask)

@al.kernel
def _k_minimum(x, y, out: al.output, N: al.constexpr):
    pid = al.program_id(0)
    offs = pid * 1024 + al.arange(0, 1024)
    mask = offs < N
    al.store(out + offs, al.minimum(al.load(x + offs, mask=mask), al.load(y + offs, mask=mask)), mask=mask)

@pytest.mark.parametrize("kernel,np_fn", [
    (_k_maximum, np.maximum),
    (_k_minimum, np.minimum),
], ids=["maximum", "minimum"])
def test_binary(kernel, np_fn):
    N = 4096
    rng = np.random.default_rng(42)
    x = rng.standard_normal(N).astype(np.float32)
    y = rng.standard_normal(N).astype(np.float32)
    out = np.zeros(N, dtype=np.float32)
    result = kernel[(N + 1023) // 1024](x, y, out, N=N)
    np.testing.assert_allclose(result, np_fn(x, y))


# ---------------------------------------------------------------------------
# Non-power-of-two N (mask correctness)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N", [1, 7, 3333, 100_000])
def test_non_power_of_two(N):
    k = get_unary_kernel("exp")
    rng = np.random.default_rng(42)
    x = rng.standard_normal(N).astype(np.float32) * 0.5
    out = np.zeros(N, dtype=np.float32)
    result = k[(N + 1023) // 1024](x, out, N=N)
    np.testing.assert_allclose(result, np.exp(x), atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Chained ops within a single kernel
# ---------------------------------------------------------------------------

def test_exp_log_roundtrip():
    @al.kernel
    def k(x, out: al.output, N: al.constexpr):
        pid = al.program_id(0)
        offs = pid * 1024 + al.arange(0, 1024)
        mask = offs < N
        x_val = al.load(x + offs, mask=mask)
        al.store(out + offs, al.exp(al.log(x_val)), mask=mask)

    N = 4096
    x = np.abs(np.random.default_rng(42).standard_normal(N).astype(np.float32)) + 0.01
    out = np.zeros(N, dtype=np.float32)
    np.testing.assert_allclose(k[(N + 1023) // 1024](x, out, N=N), x, atol=1e-5, rtol=1e-5)


def test_vector_add():
    N = 4096
    rng = np.random.default_rng(42)
    x = rng.standard_normal(N).astype(np.float32)
    y = rng.standard_normal(N).astype(np.float32)
    out = np.zeros(N, dtype=np.float32)
    result = k_add[(N + 1023) // 1024](x, y, out, N=N)
    np.testing.assert_allclose(result, x + y, rtol=1e-5)
