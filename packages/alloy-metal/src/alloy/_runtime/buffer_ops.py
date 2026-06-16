"""AlloyBuffer operator implementations — pure alloy-metal, no torch.

Populates _buf_ops on import. Imported by alloy/__init__.py after std
kernels are available.
"""

from __future__ import annotations

import math

import alloy.dsl as _dsl
from alloy._compiler.dtypes import DType, bfloat16, float32, from_torch_dtype, int32, int64
from alloy._dispatch.buf_utils import _alloc_aligned
from alloy._dispatch.contiguify import contiguify_lazy
from alloy._runtime.alloy_buffer import (
    AlloyBuffer,
    _buf_ops,
    _compute_contiguous_strides,
    _product,
)
from alloy.std.elementwise import (
    _make_elementwise_binary,
    _make_elementwise_unary,
    add,
    k_floor,
    k_gelu,
    k_relu,
    k_sigmoid,
    mul,
    neg,
    sub,
)
from alloy.std.gemm import dot_transpose_rhs

import alloy as _al

# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

_k_div = _make_elementwise_binary("k_div", lambda a, b: a / b)
_k_rsqrt = _make_elementwise_unary("k_rsqrt", lambda v: _dsl.rsqrt(v))
_k_exp = _make_elementwise_unary("k_exp", lambda v: _dsl.exp(v))
_k_log = _make_elementwise_unary("k_log", lambda v: _dsl.log(v))
_k_sqrt = _make_elementwise_unary("k_sqrt", lambda v: _dsl.sqrt(v))
_k_sin = _make_elementwise_unary("k_sin", lambda v: _dsl.sin(v))
_k_cos = _make_elementwise_unary("k_cos", lambda v: _dsl.cos(v))
_k_abs = _make_elementwise_unary("k_abs", lambda v: _dsl.abs(v))
_k_erf = _make_elementwise_unary("k_erf", lambda v: _dsl.erf(v))
_k_tanh = _make_elementwise_unary("k_tanh_buf", lambda v: _dsl.tanh(_dsl.clamp(v, -10.0, 10.0)))
_k_gelu_tanh = _make_elementwise_unary(
    "k_gelu_tanh_buf",
    lambda v: (
        0.5
        * v
        * (
            1.0
            + _dsl.tanh(_dsl.clamp(0.7978845608028654 * (v + 0.044715 * v * v * v), -10.0, 10.0))
        )
    ),
)


def _make_strided_binary_nd(name, op_fn):
    """Binary kernel with 4D strided indexing for broadcast N-D ops.

    Non-contig views: resolve_inputs rebases to the parent AlloyBuffer but
    keeps the view's offset on `data_ptr`, so `x + idx` starts at the
    view's first element. Strides here are per-logical-dim; no explicit
    offset constexpr is needed.
    """

    @_al.kernel
    def k(
        x,
        y,
        out: _al.output,
        N: _al.constexpr,
        OUT0: _al.constexpr = 1,
        OUT1: _al.constexpr = 1,
        OUT2: _al.constexpr = 1,
        OUT3: _al.constexpr = 1,
        X_STR0: _al.constexpr = 0,
        X_STR1: _al.constexpr = 0,
        X_STR2: _al.constexpr = 0,
        X_STR3: _al.constexpr = 0,
        Y_STR0: _al.constexpr = 0,
        Y_STR1: _al.constexpr = 0,
        Y_STR2: _al.constexpr = 0,
        Y_STR3: _al.constexpr = 0,
        BLOCK_SIZE: _al.constexpr = 1024,
    ):
        pid = _al.program_id(0)
        offs = pid * BLOCK_SIZE + _al.arange(0, BLOCK_SIZE)
        mask = offs < N
        rem = offs
        i3 = rem % OUT3
        rem = rem // OUT3
        i2 = rem % OUT2
        rem = rem // OUT2
        i1 = rem % OUT1
        i0 = rem // OUT1
        x_offs = i0 * X_STR0 + i1 * X_STR1 + i2 * X_STR2 + i3 * X_STR3
        y_offs = i0 * Y_STR0 + i1 * Y_STR1 + i2 * Y_STR2 + i3 * Y_STR3
        a = _al.load(x + x_offs, mask=mask)
        b = _al.load(y + y_offs, mask=mask)
        _al.store(out + offs, op_fn(a, b), mask=mask)

    k.name = name
    k._source = f"# {name}\n" + k._source
    return k


k_le_nd = _make_strided_binary_nd("k_le_nd", lambda a, b: _dsl.where(a <= b, 1, 0))
k_gt_nd = _make_strided_binary_nd("k_gt_nd", lambda a, b: _dsl.where(a > b, 1, 0))
k_ge_nd = _make_strided_binary_nd("k_ge_nd", lambda a, b: _dsl.where(a >= b, 1, 0))
k_eq_nd = _make_strided_binary_nd("k_eq_nd", lambda a, b: _dsl.where(a == b, 1, 0))
k_ne_nd = _make_strided_binary_nd("k_ne_nd", lambda a, b: _dsl.where(a == b, 0, 1))
k_bitwise_and_nd = _make_strided_binary_nd(
    "k_bitwise_and_nd", lambda a, b: _dsl.cast(a, _dsl.int32) & _dsl.cast(b, _dsl.int32)
)
k_bitwise_or_nd = _make_strided_binary_nd(
    "k_bitwise_or_nd", lambda a, b: _dsl.cast(a, _dsl.int32) | _dsl.cast(b, _dsl.int32)
)
k_logical_not = _make_elementwise_unary("k_logical_not", lambda v: _dsl.where(v == 0, 1, 0))
k_logical_and_nd = _make_strided_binary_nd(
    "k_logical_and_nd",
    lambda a, b: _dsl.where((a != 0) & (b != 0), 1, 0),
)
_k_log1p = _make_elementwise_unary("k_log1p", lambda v: _dsl.log(v + 1.0))


@_al.kernel
def _k_clamp(
    x,
    out: _al.output,
    N: _al.constexpr,
    LO: _al.constexpr = -1e30,
    HI: _al.constexpr = 1e30,
    BLOCK_SIZE: _al.constexpr = 1024,
):
    pid = _al.program_id(0)
    offs = pid * BLOCK_SIZE + _al.arange(0, BLOCK_SIZE)
    mask = offs < N
    v = _al.load(x + offs, mask=mask)
    _al.store(out + offs, _al.minimum(_al.maximum(v, LO), HI), mask=mask)


_BINARY_KERNELS = {"add": add, "sub": sub, "mul": mul, "div": _k_div}


def _make_scalar_binary(name, op_fn):
    """Unary-shaped kernel that applies a scalar constexpr to every element.

    Python scalars in `tensor op scalar` graphs (e.g. embed_out * embed_scale)
    don't need broadcast machinery — bake the scalar into the kernel as a
    constexpr and emit a flat unary pass. Avoids the 0-dim AlloyBuffer +
    strided-broadcast detour entirely.
    """

    @_al.kernel
    def k(
        x,
        out: _al.output,
        N: _al.constexpr,
        SCALAR: _al.constexpr,
        BLOCK_SIZE: _al.constexpr = 1024,
    ):
        pid = _al.program_id(0)
        offs = pid * BLOCK_SIZE + _al.arange(0, BLOCK_SIZE)
        mask = offs < N
        v = _al.load(x + offs, mask=mask)
        _al.store(out + offs, op_fn(v, SCALAR), mask=mask)

    k.name = name
    k._source = f"# {name}\n" + k._source
    return k


_k_add_scalar = _make_scalar_binary("k_add_scalar", lambda v, s: v + s)
_k_sub_scalar = _make_scalar_binary("k_sub_scalar", lambda v, s: v - s)
_k_rsub_scalar = _make_scalar_binary("k_rsub_scalar", lambda v, s: s - v)
_k_mul_scalar = _make_scalar_binary("k_mul_scalar", lambda v, s: v * s)
_k_div_scalar = _make_scalar_binary("k_div_scalar", lambda v, s: v / s)
_k_rdiv_scalar = _make_scalar_binary("k_rdiv_scalar", lambda v, s: s / v)

_SCALAR_BINARY_KERNELS = {
    "add": (_k_add_scalar, _k_add_scalar),  # add is commutative
    "mul": (_k_mul_scalar, _k_mul_scalar),  # mul is commutative
    "sub": (_k_sub_scalar, _k_rsub_scalar),  # (buf-scalar, scalar-buf)
    "div": (_k_div_scalar, _k_rdiv_scalar),
}

_UNARY_KERNELS = {
    "neg": neg,
    "rsqrt": _k_rsqrt,
    "exp": _k_exp,
    "log": _k_log,
    "sqrt": _k_sqrt,
    "sin": _k_sin,
    "cos": _k_cos,
    "abs": _k_abs,
    "sigmoid": k_sigmoid,
    "relu": k_relu,
    "floor": k_floor,
    "tanh": _k_tanh,
    "erf": _k_erf,
    "gelu_none": k_gelu,
    "gelu_tanh": _k_gelu_tanh,
}

_COMPARE_KERNELS = {"le": k_le_nd, "ge": k_ge_nd, "gt": k_gt_nd}

# Cast kernels per alloy DType ir name
_CAST_KERNELS: dict[str, object] = {}


def _get_cast_kernel(ir: str):
    if ir not in _CAST_KERNELS:
        _CAST_KERNELS[ir] = _make_elementwise_unary(
            f"cast_{ir}", lambda v, _ir=ir: _dsl.cast(v, _ir)
        )
    return _CAST_KERNELS[ir]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _broadcast_shapes(*shapes: tuple[int, ...]) -> tuple[int, ...]:
    ndim = max(len(s) for s in shapes)
    padded = [(1,) * (ndim - len(s)) + s for s in shapes]
    return tuple(max(dims) for dims in zip(*padded))


def _is_flat_contiguous(x: AlloyBuffer) -> bool:
    """C-contiguous ignoring size-1 dims (their stride never affects flat
    addressing) — the numpy/torch definition. The elementwise kernels read x
    flat, so a size-1 leading dim with a non-packed stride (e.g. the decode
    rotary freqs permute, shape (1,1,64) strides (64,1,1)) is still a unit-stride
    read and needs no contiguify. The strict is_contiguous() flags it as
    non-contiguous and inserts a wasteful strided_copy."""
    expected = x._dtype.itemsize
    for size, stride in zip(reversed(x._shape), reversed(x._strides)):
        if size == 1:
            continue
        if stride != expected:
            return False
        expected *= size
    return True


def _ensure_contiguous(x: AlloyBuffer) -> AlloyBuffer:
    if _is_flat_contiguous(x):
        return x
    return x.contiguous(force=True)


def _ensure_zero_offset(x: AlloyBuffer) -> AlloyBuffer:
    if x._offset != 0 or not x.is_contiguous():
        return x.contiguous(force=True)
    return x


def _expand_buf(buf: AlloyBuffer, out_shape: tuple[int, ...]) -> AlloyBuffer:
    """Broadcast buf to out_shape by setting strides to 0 for broadcast dims."""
    if buf.shape == out_shape:
        return buf
    pad = len(out_shape) - len(buf.shape)
    padded_shape = (1,) * pad + buf.shape
    padded_strides = (0,) * pad + buf._strides
    new_strides = tuple(
        0 if padded_shape[i] == 1 and out_shape[i] > 1 else padded_strides[i]
        for i in range(len(out_shape))
    )
    new_buf = AlloyBuffer(
        buf._parent_handle,
        buf._offset,
        buf._shape,
        buf._strides,
        buf._dtype,
        raw_ptr=buf._raw_ptr,
        total_nbytes=buf._total_nbytes,
    )
    new_buf.reinterpret(out_shape, new_strides)
    buf._view_of(new_buf)
    return new_buf


def _scalar_buf(value, dtype: DType) -> AlloyBuffer:
    buf = _alloc_aligned((), dtype)
    buf.write_scalar(value)
    return buf


def _coerce_scalar(value, ref_dtype: DType) -> AlloyBuffer:
    if isinstance(value, float):
        return _scalar_buf(value, ref_dtype if ref_dtype.is_float() else float32)
    if isinstance(value, bool):
        return _scalar_buf(int(value), int64)
    if isinstance(value, int):
        return _scalar_buf(value, ref_dtype if not ref_dtype.is_float() else ref_dtype)
    raise TypeError(f"Cannot coerce {type(value).__name__} to AlloyBuffer")


def _broadcast_layout_4d(buf: AlloyBuffer, out_shape: tuple[int, ...]):
    """Return (padded_shape_4d, padded_elem_strides_4d) for N-D strided kernels."""
    itemsize = buf._dtype.itemsize
    ndim = len(buf._shape)
    pad = len(out_shape) - ndim
    strides = tuple(s // itemsize for s in buf._strides)
    broadcast_strides = (0,) * pad + tuple(
        0 if buf._shape[i] == 1 and out_shape[pad + i] != 1 else strides[i] for i in range(ndim)
    )
    n = len(out_shape)
    pad4 = 4 - n
    return (1,) * pad4 + out_shape, (0,) * pad4 + broadcast_strides


def _normalize_dim(dim: int, ndim: int) -> int:
    return dim if dim >= 0 else dim + ndim


def _shape_of(value) -> tuple[int, ...]:
    if isinstance(value, AlloyBuffer):
        return value.shape
    if isinstance(value, (bool, int, float)):
        return ()
    raise TypeError(f"Cannot determine shape for {type(value)!r}")


# ---------------------------------------------------------------------------
# Binary dispatch
# ---------------------------------------------------------------------------


def _annotate_logical(flat: AlloyBuffer, shape: tuple[int, ...]) -> AlloyBuffer:
    """Stamp a flat-born elementwise kernel output with its logical layout.

    Flat 1D kernels walk their output in storage order, so the logical
    (extent, byte_stride) axes tell the compiled-plan recorder which axis is
    outermost — without this, the one-shot grid-shrink recipe cannot tell a
    row-major (M, k) write from a (k, M) one (the rope-table broadcast bug:
    identical flat extents, opposite shrink semantics)."""
    flat._pre_flatten_dims = tuple(
        zip(shape, _compute_contiguous_strides(shape, flat._dtype.itemsize))
    )
    return flat


def _binary_dispatch(kernel_name: str, a, b) -> AlloyBuffer:
    k = _BINARY_KERNELS[kernel_name]
    # Scalar specialization: Python int/float operand stays as a constexpr
    # in a unary kernel, never gets coerced to a 0-dim AlloyBuffer + flat
    # broadcast (which is broken — flat kernel reads y[offs>0] OOB). Covers
    # e.g. `embedding_out * embed_scale` from Gemma3TextScaledWordEmbedding.
    if kernel_name in _SCALAR_BINARY_KERNELS:
        a_scalar = isinstance(a, (int, float, bool))
        b_scalar = isinstance(b, (int, float, bool))
        # Only specialize when the buffer is float — for int buffers the
        # SCALAR constexpr would promote to float in the emitted MSL and
        # corrupt the integer write (e.g. cumulative_length.add_(128) on
        # an i64 buffer).
        if a_scalar and not b_scalar and isinstance(b, AlloyBuffer) and b._dtype.is_float():
            k_buf_scalar, k_scalar_buf = _SCALAR_BINARY_KERNELS[kernel_name]
            buf = _ensure_contiguous(b)
            out = _annotate_logical(k_scalar_buf(buf, N=buf.size, SCALAR=float(a)), buf.shape)
            return out.reshape(buf.shape)
        if b_scalar and not a_scalar and isinstance(a, AlloyBuffer) and a._dtype.is_float():
            k_buf_scalar, _ = _SCALAR_BINARY_KERNELS[kernel_name]
            buf = _ensure_contiguous(a)
            out = _annotate_logical(k_buf_scalar(buf, N=buf.size, SCALAR=float(b)), buf.shape)
            return out.reshape(buf.shape)
    if not isinstance(a, AlloyBuffer):
        a = _coerce_scalar(a, b._dtype)
    if not isinstance(b, AlloyBuffer):
        b = _coerce_scalar(b, a._dtype)
    # Mixed-dtype promotion for bf16↔f32: Metal's `bfloat` doesn't promote
    # implicitly to `float` in kernel expressions, so a kernel specialized
    # on a bf16 operand stays in bf16 precision even when the other operand
    # is f32 and the graph requested f32 compute. Upcast the bf16 operand
    # explicitly to honor AOT autograd's `_to_copy(bf16→f32)` intent.
    if a._dtype == bfloat16 and b._dtype == float32:
        a = _buf_ops["to_dtype"](a, float32)
    elif b._dtype == bfloat16 and a._dtype == float32:
        b = _buf_ops["to_dtype"](b, float32)
    if a.shape == b.shape:
        a = _ensure_contiguous(a)
        b = _ensure_contiguous(b)
        return _annotate_logical(k(a, b, N=a.size), a.shape).reshape(a.shape)
    out_shape = _broadcast_shapes(a.shape, b.shape)
    a = _ensure_contiguous(a)
    b = _ensure_contiguous(b)
    a = _expand_buf(a, out_shape)
    b = _expand_buf(b, out_shape)
    return _annotate_logical(k(a, b, N=_product(out_shape)), out_shape).reshape(out_shape)


# ---------------------------------------------------------------------------
# Unary dispatch
# ---------------------------------------------------------------------------


def _unary_dispatch(kernel_name: str, x: AlloyBuffer) -> AlloyBuffer:
    k = _UNARY_KERNELS[kernel_name]
    x = _ensure_contiguous(x)
    return _annotate_logical(k(x, N=x.size), x.shape).reshape(x.shape)


# ---------------------------------------------------------------------------
# Comparison dispatch — N-D with broadcast strides
# ---------------------------------------------------------------------------


def _compare_nd(kernel_fn, a, b) -> AlloyBuffer:
    if not isinstance(a, AlloyBuffer):
        ref = b._dtype if isinstance(b, AlloyBuffer) else float32
        a = _coerce_scalar(a, ref)
    if not isinstance(b, AlloyBuffer):
        b = _coerce_scalar(b, a._dtype)
    out_shape = _broadcast_shapes(a.shape, b.shape)
    padded_out, lhs_strides = _broadcast_layout_4d(a, out_shape)
    _, rhs_strides = _broadcast_layout_4d(b, out_shape)
    out_arr = _alloc_aligned(out_shape, int32)  # bool as int32
    return kernel_fn(
        a,
        b,
        out_arr,
        N=_product(out_shape),
        OUT0=padded_out[0],
        OUT1=padded_out[1],
        OUT2=padded_out[2],
        OUT3=padded_out[3],
        X_STR0=lhs_strides[0],
        X_STR1=lhs_strides[1],
        X_STR2=lhs_strides[2],
        X_STR3=lhs_strides[3],
        Y_STR0=rhs_strides[0],
        Y_STR1=rhs_strides[1],
        Y_STR2=rhs_strides[2],
        Y_STR3=rhs_strides[3],
    ).reshape(out_shape)


# ---------------------------------------------------------------------------
# Matmul (2D)
# ---------------------------------------------------------------------------


def _transpose_base_2d(value: AlloyBuffer) -> AlloyBuffer | None:
    if len(value._shape) != 2:
        return None
    M, N = value._shape
    itemsize = value._dtype.itemsize
    if value._strides != (itemsize, M * itemsize):
        return None
    base = AlloyBuffer(
        value._parent_handle,
        value._offset,
        value._shape,
        value._strides,
        value._dtype,
        raw_ptr=value._raw_ptr,
        total_nbytes=value._total_nbytes,
    )
    base.reinterpret((N, M), (M * itemsize, itemsize))
    return value._view_of(base)


def _mm(a: AlloyBuffer, b: AlloyBuffer) -> AlloyBuffer:
    squeeze = False
    if a.ndim == 1:
        a = a.reshape((1, a.shape[0]))
        squeeze = True
    a = _ensure_zero_offset(a)
    rhs_t = _transpose_base_2d(b)
    if rhs_t is not None:
        rhs_t = _ensure_zero_offset(rhs_t)
        M, N = a.shape[0], rhs_t.shape[0]
        out = _alloc_aligned((M, N), a.dtype)
        result = dot_transpose_rhs(a, rhs_t, out)
    else:
        b = _ensure_zero_offset(b)
        result = _al.dot(a, b)
    if squeeze:
        result = result.reshape(result.shape[-1])
    return result


# ---------------------------------------------------------------------------
# Clamp
# ---------------------------------------------------------------------------


def _clamp(x: AlloyBuffer, lo=None, hi=None) -> AlloyBuffer:
    x = _ensure_contiguous(x)
    lo_f = float(lo) if lo is not None else -1e30
    hi_f = float(hi) if hi is not None else 1e30
    # An unbounded side may arrive as ±inf (e.g. a Gemma4ClippableLinear whose
    # clip stats are un-set) — fold it to the same ±1e30 sentinel the `None`
    # default uses. MSL has no `inf` literal for the kernel's LO/HI constexpr,
    # and a clamp to ±inf is a no-op anyway.
    if math.isinf(lo_f):
        lo_f = -1e30 if lo_f < 0 else 1e30
    if math.isinf(hi_f):
        hi_f = 1e30 if hi_f > 0 else -1e30
    return _k_clamp(x, N=x.size, LO=lo_f, HI=hi_f).reshape(x.shape)


# ---------------------------------------------------------------------------
# Softmax
# ---------------------------------------------------------------------------


def _softmax(x: AlloyBuffer, dim: int = -1) -> AlloyBuffer:
    if x.ndim == 0:
        return x
    dim = _normalize_dim(dim, x.ndim)
    if dim != x.ndim - 1:
        axes = list(range(x.ndim))
        axes[dim], axes[-1] = axes[-1], axes[dim]
        out = _softmax(x.transpose(*axes), dim=-1)
        inverse = [0] * len(axes)
        for i, axis in enumerate(axes):
            inverse[axis] = i
        return out.transpose(*inverse)
    if x.ndim == 1:
        return _al.softmax(x.reshape((1, x.shape[0]))).reshape(x.shape)
    flat_rows = math.prod(x.shape[:-1])
    cols = x.shape[-1]
    return _al.softmax(x.reshape((flat_rows, cols))).reshape(x.shape)


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------


def _sum_dim(x: AlloyBuffer, dim, keepdim: bool = False) -> AlloyBuffer:
    shape = x.shape
    # aten.sum.dim_IntList(x, []) means "reduce over all dims" (the
    # ForCausalLMLoss cross-entropy path uses this form to produce the
    # scalar loss). Same semantics as dim=None.
    if dim is None or (isinstance(dim, (list, tuple)) and len(dim) == 0):
        result = _al.reduce_sum(x.reshape(1, x.size))
        if not keepdim:
            return result.reshape(())
        return result.reshape(tuple(1 for _ in shape))
    if isinstance(dim, (list, tuple)):
        dims = sorted([_normalize_dim(int(d), len(shape)) for d in dim], reverse=True)
    else:
        dims = [_normalize_dim(int(dim), len(shape))]
    result = x
    for d in dims:
        s = result.shape
        ndim = len(s)
        red = s[d]
        remaining = math.prod(s[:d]) * math.prod(s[d + 1 :])
        if d != ndim - 1:
            perm = list(range(ndim))
            perm.append(perm.pop(d))
            result = result.transpose(*perm)
        flat = result.reshape((remaining, red))
        summed = _al.reduce_sum(flat, axis=1)
        new_shape = s[:d] + (1,) * keepdim + s[d + 1 :]
        result = summed.reshape(new_shape) if new_shape else summed
    return result


def _mean_dim(x: AlloyBuffer, dim, keepdim: bool = False) -> AlloyBuffer:
    shape = x.shape
    ndim = len(shape)
    if dim is None:
        return _al.mean(x.reshape((x.size,)))
    if isinstance(dim, (list, tuple)):
        if len(dim) != 1:
            raise RuntimeError(f"Alloy mean: multi-dim reduction {dim} not supported on GPU")
        dim = dim[0]
    dim = _normalize_dim(int(dim), ndim)
    dim_size = shape[dim]
    outer = math.prod(s for i, s in enumerate(shape) if i != dim)
    if dim == ndim - 1:
        flat = x.reshape((outer, dim_size))
    else:
        axes = [i for i in range(ndim) if i != dim] + [dim]
        flat = x.transpose(*axes).reshape((outer, dim_size))
    reduced = _al.mean(flat, axis=1)
    out_shape = list(shape)
    if keepdim:
        out_shape[dim] = 1
    else:
        out_shape.pop(dim)
    return reduced.reshape(tuple(out_shape) if out_shape else (1,))


# ---------------------------------------------------------------------------
# Type conversion
# ---------------------------------------------------------------------------


def _to_dtype(x: AlloyBuffer, dtype) -> AlloyBuffer:
    """Cast x to the given dtype (alloy DType or torch dtype)."""
    if not isinstance(dtype, DType):
        # Accept torch dtype — convert to alloy DType
        dtype = from_torch_dtype(dtype)
    if x._dtype == dtype:
        return x
    x = _ensure_contiguous(x)
    k = _get_cast_kernel(dtype.ir)
    return k(x, N=x.size).reshape(x.shape)


# ---------------------------------------------------------------------------
# Populate _buf_ops
# ---------------------------------------------------------------------------

# Binary arithmetic
_buf_ops["add"] = lambda a, b: _binary_dispatch("add", a, b)
_buf_ops["mul"] = lambda a, b: _binary_dispatch("mul", a, b)
_buf_ops["sub"] = lambda a, b: _binary_dispatch("sub", a, b)
_buf_ops["div"] = lambda a, b: _binary_dispatch("div", a, b)

# Unary arithmetic
for _name in _UNARY_KERNELS:
    _buf_ops[_name] = lambda x, _n=_name: _unary_dispatch(_n, x)

# Gelu (extra arg)
_buf_ops["gelu"] = lambda x, approx="none": (
    _unary_dispatch("gelu_tanh", x) if approx == "tanh" else _unary_dispatch("gelu_none", x)
)

# Comparison
_buf_ops["le"] = lambda a, b: _compare_nd(k_le_nd, a, b)
_buf_ops["ge"] = lambda a, b: _compare_nd(k_ge_nd, a, b)
_buf_ops["gt"] = lambda a, b: _compare_nd(k_gt_nd, a, b)
_buf_ops["lt"] = lambda a, b: _compare_nd(k_gt_nd, b, a)

# Matmul
_buf_ops["matmul"] = _mm

# Higher-level ops
_buf_ops["clamp"] = _clamp
_buf_ops["softmax"] = _softmax
_buf_ops["sum"] = _sum_dim
_buf_ops["mean"] = _mean_dim
_buf_ops["to_dtype"] = _to_dtype
_buf_ops["contiguify"] = contiguify_lazy
_buf_ops["expand"] = _expand_buf
