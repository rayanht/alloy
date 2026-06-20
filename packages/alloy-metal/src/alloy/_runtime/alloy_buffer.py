"""AlloyBuffer — unified GPU buffer with lazy materialization.

Single type for all buffer operations in Alloy. Tracks GPU memory handle,
shape/strides/dtype, and deferred computation chain. Materializes on first
read access. View operations (reshape/transpose/slice) are zero-copy metadata
changes that share the materialization chain.

materialize_many() imports _materialize_many lazily to break the circular import.
"""

import ctypes
import struct
from typing import Any, Self

import alloy._runtime._metal_ext as _ext
from alloy._compiler.dtypes import DType, float32, from_numpy
import numpy as np


def _product(seq) -> int:
    r = 1
    for s in seq:
        r *= s
    return r


def _normalize_shape(shape) -> tuple[int, ...]:
    if type(shape) is tuple:
        return shape
    if isinstance(shape, (list, tuple)):
        return tuple(shape)
    return (int(shape),)


def _compute_contiguous_strides(shape, itemsize) -> tuple[int, ...]:
    """Compute C-contiguous byte strides for a shape."""
    ndim = len(shape)
    if ndim == 0:
        return ()
    strides = [0] * ndim
    strides[-1] = itemsize
    for i in range(ndim - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


def _reshape_strides(
    old_shape: tuple[int, ...],
    old_strides: tuple[int, ...],
    new_shape: tuple[int, ...],
) -> tuple[int, ...] | None:
    """Compute strides for reshaping a non-contiguous buffer as a view.

    Returns new byte strides if the reshape is valid without copying,
    or None if a copy is required.
    """
    if _product(old_shape) != _product(new_shape):
        return None

    old_s = [(s, st) for s, st in zip(old_shape, old_strides) if s != 1]
    new_s = [s for s in new_shape if s != 1]

    if not old_s:
        return _compute_contiguous_strides(new_shape, old_strides[-1] if old_strides else 1)

    oi = len(old_s) - 1
    ni = len(new_s) - 1
    result_strides = [0] * len(new_s)

    while ni >= 0 and oi >= 0:
        new_elems = new_s[ni]
        old_elems = old_s[oi][0]
        innermost_stride = old_s[oi][1]

        if new_elems == old_elems:
            result_strides[ni] = innermost_stride
            oi -= 1
            ni -= 1
        elif new_elems < old_elems:
            if old_elems % new_elems != 0:
                return None
            result_strides[ni] = innermost_stride
            old_s[oi] = (old_elems // new_elems, innermost_stride * new_elems)
            ni -= 1
        else:
            # Merging source dims into one target dim: result stride is the
            # innermost (rightmost) stride.
            merge_stride = old_s[oi][1]
            while old_elems < new_elems and oi > 0:
                if old_s[oi][1] * old_s[oi][0] != old_s[oi - 1][1]:
                    return None
                oi -= 1
                old_elems *= old_s[oi][0]
            if old_elems != new_elems:
                return None
            result_strides[ni] = merge_stride
            oi -= 1
            ni -= 1

    if ni >= 0 or oi >= 0:
        return None

    full_strides = []
    ri = 0
    for s in new_shape:
        if s == 1:
            if ri < len(result_strides):
                full_strides.append(new_s[ri] * result_strides[ri])
            else:
                full_strides.append(old_strides[-1] if old_strides else 1)
        else:
            full_strides.append(result_strides[ri])
            ri += 1

    return tuple(full_strides)


# Operator dispatch table — populated by buffer_ops.py when the kernel layer loads.
# Keys: 'add', 'mul', 'sub', 'div', 'neg', 'matmul', 'le', 'ge', 'gt', 'eq', 'ne',
#        'rsqrt', 'exp', 'log', 'sqrt', 'sin', 'cos', 'abs', 'sigmoid', 'relu',
#        'floor', 'tanh', 'gelu', 'clamp', 'contiguify', 'expand', 'to_dtype',
#        'softmax', 'sum', 'mean'
_buf_ops: dict[str, object] = {}


def materialize_many(values) -> None:
    """Materialize all pending AlloyBuffers in a nested structure in one flush."""
    pending = []

    def _collect(value) -> None:
        if isinstance(value, AlloyBuffer):
            if value._materializer is not None:
                pending.append(value)
            return
        if isinstance(value, (tuple, list)):
            for item in value:
                _collect(item)
            return
        if isinstance(value, dict):
            for item in value.values():
                _collect(item)

    _collect(values)
    if pending:
        from alloy._dispatch.lazy import (
            _materialize_many,
        )  # scoped: avoid cycle (_lazy imports AlloyBuffer from this module)

        _materialize_many(pending)


class AlloyBuffer:
    """Unified GPU buffer: memory handle + tensor metadata + lazy dispatch.

    _dtype is an alloy DType (not np.dtype). Numpy conversion happens only
    in the .numpy property at the interop boundary.
    """

    __slots__ = (
        "_parent_handle",
        "_raw_ptr",
        "_offset",
        "_shape",
        "_strides",
        "_dtype",  # DType — the canonical dtype
        "_total_nbytes",
        "_np_ref",
        "_materializer",
        "_producer",
        "_owns_aligned",
        "_owner",
        "_ext_ref",
        "_backing_arr",  # retains page-aligned allocation for Metal zero-copy
        "_pre_flatten_dims",  # (extent, byte_stride) axes of the pre-flatten view
    )

    def __init__(
        self,
        parent_handle=-1,
        offset=0,
        shape=(),
        strides=(),
        dtype: DType = float32,
        *,
        raw_ptr=0,
        total_nbytes=0,
        materializer=None,
        producer=None,
        owns_aligned=False,
        owner=None,
        arr=None,
    ):
        # numpy array input: wrap as external buffer
        if arr is not None:
            if isinstance(arr, np.generic) and not isinstance(arr, np.ndarray):
                arr = np.array(arr)
            if isinstance(arr, np.ndarray):
                raw_ptr = arr.ctypes.data
                shape = tuple(arr.shape)
                strides = tuple(arr.strides)
                dtype = from_numpy(arr.dtype)
                total_nbytes = arr.nbytes
                self._np_ref = arr
            else:
                self._np_ref = None
        else:
            self._np_ref = None

        self._parent_handle = parent_handle
        self._raw_ptr = raw_ptr
        self._offset = offset
        self._shape = tuple(int(s) for s in shape) if shape else ()
        self._strides = tuple(int(s) for s in strides) if strides else ()
        self._dtype = dtype if isinstance(dtype, DType) else from_numpy(dtype)
        self._total_nbytes = total_nbytes
        self._materializer = materializer
        self._producer = producer
        self._owns_aligned = owns_aligned
        self._owner = owner
        self._ext_ref = None
        self._backing_arr = None
        # Layout provenance: (extent, byte_stride) axes of the view this buffer
        # was flattened from (set by reshape when it loses rank). A contiguous
        # flatten preserves storage order, so these axes still describe what a
        # linear (1D-gridded) kernel walks — the compiled-plan recorder uses
        # them so the grid-shrink recipe can see whether M is the outermost
        # axis of a written buffer.
        self._pre_flatten_dims = None

    @staticmethod
    def from_raw_ptr(ptr, shape, strides, dtype, total_nbytes):
        """Create a buffer from a raw pointer (torch tensor, numpy array, etc.).

        dtype can be a DType or anything _dtype_from_numpy accepts (np.dtype, etc.).
        """
        if not isinstance(dtype, DType):
            dtype = from_numpy(dtype)
        return AlloyBuffer(-1, 0, shape, strides, dtype, raw_ptr=ptr, total_nbytes=total_nbytes)

    # --- Fast view constructor (no __init__ overhead) ---

    def _view(self, shape, strides, offset=None) -> Self:
        """New AlloyBuffer sharing materialization chain with different metadata."""
        v = AlloyBuffer.__new__(AlloyBuffer)
        v._parent_handle = self._parent_handle
        v._raw_ptr = self._raw_ptr
        v._offset = offset if offset is not None else self._offset
        v._shape = shape
        v._strides = strides
        v._dtype = self._dtype
        v._total_nbytes = self._total_nbytes
        v._np_ref = self._np_ref
        v._materializer = self._materializer
        v._producer = self._producer
        v._owns_aligned = False
        v._owner = self if self._owner is None else self._owner
        v._ext_ref = None
        v._pre_flatten_dims = None
        return v

    def _view_of(self, other: Self) -> Self:
        """Copy lazy fields from self onto other AlloyBuffer. Returns other."""
        other._materializer = self._materializer
        other._producer = self._producer
        other._owns_aligned = False
        other._owner = self if self._owner is None else self._owner
        return other

    # --- GPU memory access ---

    @property
    def handle(self):
        return self._parent_handle

    @property
    def data_ptr(self):
        """Raw data pointer including offset — for aliasing detection."""
        if self._raw_ptr:
            return self._raw_ptr + self._offset
        if self._parent_handle < 0:
            return 0
        return _ext.buf_ptr(self._parent_handle) + self._offset

    @property
    def base_ptr(self):
        """Base allocation pointer (without offset) — for Metal buffer binding."""
        if self._raw_ptr:
            return self._raw_ptr
        return _ext.buf_ptr(self._parent_handle)

    def shares_allocation(self, other: "AlloyBuffer") -> bool:
        """Canonical buffer identity check — do these buffers share the same root allocation?"""
        if self._parent_handle >= 0 and other._parent_handle >= 0:
            return self._parent_handle == other._parent_handle
        return self.base_ptr == other.base_ptr

    @property
    def allocation_id(self) -> int:
        """Hashable identifier for the root allocation."""
        if self._parent_handle >= 0:
            return self._parent_handle
        return self.base_ptr

    @property
    def metal_nbytes(self):
        """Total allocation size for Metal buffer binding."""
        if self._total_nbytes:
            return self._total_nbytes
        return _ext.buf_nbytes(self._parent_handle)

    @property
    def buffer_key(self) -> tuple[int, int]:
        """(allocation_id, byte_offset) — distinguishes views of the same allocation."""
        return (self.allocation_id, self._offset)

    # --- Metadata ---

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> DType:
        return self._dtype

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def size(self) -> int:
        return _product(self._shape) if self._shape else 1

    @property
    def nbytes(self):
        return self.size * self._dtype.itemsize

    @property
    def strides(self) -> tuple[int, ...]:
        return self._strides

    @property
    def itemsize(self):
        return self._dtype.itemsize

    def is_contiguous(self) -> bool:
        """Check if this view is C-contiguous."""
        return self._strides == _compute_contiguous_strides(self._shape, self._dtype.itemsize)

    # --- View operations (zero-copy, share materialization chain) ---

    def reshape(self: Self, *shape) -> "AlloyBuffer":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        shape: tuple[int, ...] = tuple(int(s) for s in shape)
        # Resolve -1 dimension
        if -1 in shape:
            total = _product(self._shape) if self._shape else 1
            known = 1
            neg_idx = -1
            for i, s in enumerate(shape):
                if s == -1:
                    neg_idx = i
                else:
                    known *= s
            shape = tuple(total // known if i == neg_idx else s for i, s in enumerate(shape))
        if self._shape == shape:
            return self
        # Rank-losing reshape (flatten): carry the source layout as provenance
        # so the plan recorder can still see which axis was outermost. Chained
        # reshapes keep the earliest shaped ancestor (storage order is preserved
        # across contiguous reshapes).
        prov = self._pre_flatten_dims
        if prov is None and len(shape) < len(self._shape):
            prov = tuple(zip(self._shape, self._strides))
        if self.is_contiguous():
            v = self._view(shape, _compute_contiguous_strides(shape, self._dtype.itemsize))
            v._pre_flatten_dims = prov
            return v
        new_strides = _reshape_strides(self._shape, self._strides, shape)
        if new_strides is not None:
            v = self._view(shape, new_strides)
            v._pre_flatten_dims = prov
            return v
        return self.contiguous()._view(
            shape, _compute_contiguous_strides(shape, self._dtype.itemsize)
        )

    def transpose(self, *axes) -> Self:
        ndim = len(self._shape)
        if not axes:
            perm = tuple(range(ndim - 1, -1, -1))
        elif len(axes) == 2 and ndim >= 2:
            # Two-arg form: swap two dimensions (like torch.transpose)
            a, b = axes
            perm = list(range(ndim))
            perm[a], perm[b] = perm[b], perm[a]
            perm = tuple(perm)
        else:
            perm = axes
        new_shape: tuple[int, ...] = tuple(self._shape[a] for a in perm)
        new_strides: tuple[int, ...] = tuple(self._strides[a] for a in perm)
        return self._view(new_shape, new_strides)

    def permute(self, dims) -> Self:
        return self.transpose(*_normalize_shape(dims))

    def slice(self, dim, start, end, step=1) -> "AlloyBuffer":
        byte_offset = start * self._strides[dim]
        new_shape: list[int] = list(self._shape)
        new_shape[dim] = (end - start + step - 1) // step
        new_strides: list[int] = list(self._strides)
        if step != 1:
            new_strides[dim] *= step
        return self._view(tuple(new_shape), tuple(new_strides), self._offset + byte_offset)

    def ravel(self) -> "AlloyBuffer":
        return self.reshape((_product(self._shape),))

    def root_flat(self) -> "AlloyBuffer":
        """Mutate this buffer to a flat 1D view of the entire root allocation."""
        itemsize = self._dtype.itemsize
        n = self.metal_nbytes // itemsize
        self._offset = 0
        self._shape = (n,)
        self._strides = (itemsize,)
        return self

    def reinterpret(self, shape, strides, offset=None, dtype=None) -> "AlloyBuffer":
        """Mutate shape/strides/offset/dtype in place. Returns self."""
        self._shape = shape
        self._strides = strides
        if offset is not None:
            self._offset = offset
        if dtype is not None:
            self._dtype = dtype if isinstance(dtype, DType) else from_numpy(dtype)
        return self

    # --- Lazy materialization ---

    def _ensure_synced(self) -> None:
        if self._materializer is not None:
            materialize_many((self,))

    def sync(self) -> None:
        """Force this buffer to materialize."""
        self._ensure_synced()

    # --- Scalar reads: ctypes from DType, no numpy ---

    def read_scalar(self):
        """Read a single scalar from the buffer."""
        ct = self._dtype._ctype
        raw = ctypes.cast(self.data_ptr, ctypes.POINTER(ct))[0]
        # f16/bf16 carry a c_uint16 ctype that surfaces the raw 2-byte bit pattern,
        # not the value (e.g. -8.0 f16 reads back as 51200). Decode the bits via the
        # real float layout (struct, no numpy). Without this, float()/clamp bounds on
        # a 16-bit buffer are garbage.
        if self._dtype.ir == "f16":
            return struct.unpack("<e", struct.pack("<H", raw))[0]
        if self._dtype.ir == "bf16":
            return struct.unpack("<f", struct.pack("<I", raw << 16))[0]
        return raw

    def write_scalar(self, value):
        """Write a scalar value to a 0-d or 1-element buffer."""
        ct = self._dtype._ctype
        if self._dtype.ir == "bf16":
            bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
            ctypes.cast(self.data_ptr, ctypes.POINTER(ct))[0] = ct((bits >> 16) & 0xFFFF).value
            return
        if self._dtype.ir == "f16":
            # c_uint16 ctype would int-truncate the float; encode the half bits.
            ctypes.cast(self.data_ptr, ctypes.POINTER(ct))[0] = struct.unpack(
                "<H", struct.pack("<e", float(value))
            )[0]
            return
        ctypes.cast(self.data_ptr, ctypes.POINTER(ct))[0] = ct(value).value

    def copy_from(self, src) -> None:
        """Copy raw bytes from another AlloyBuffer or from a raw pointer."""
        if isinstance(src, AlloyBuffer):
            n = min(self.nbytes, src.nbytes)
            src_ptr = src.data_ptr
        else:
            n = self.nbytes
            src_ptr = src
        ctypes.memmove(self.data_ptr, src_ptr, n)

    # --- Python read/write (trigger sync) ---

    def __float__(self) -> float:
        self._ensure_synced()
        return float(self.read_scalar())

    def __int__(self) -> int:
        self._ensure_synced()
        return int(self.read_scalar())

    def __bool__(self) -> bool:
        self._ensure_synced()
        return bool(self.read_scalar())

    def __len__(self) -> int:
        return self._shape[0] if self._shape else 0

    def __getitem__(self, key):
        self._ensure_synced()
        return self.numpy[key]

    def __setitem__(self, key, value) -> None:
        self._ensure_synced()
        self.numpy[key] = value

    # --- Numpy interop (boundary only) ---

    @property
    def numpy(self):
        """Return a numpy view of the buffer's memory. No GPU sync."""
        np_dtype = self._dtype.to_numpy()
        if self._parent_handle >= 0:
            total_bytes = _ext.buf_nbytes(self._parent_handle)
            total_elems = total_bytes // self._dtype.itemsize
            # bf16 has no native DLPack code numpy understands, so ask C++
            # for a uint16 view of the same 2-byte storage and reinterpret
            # via ml_dtypes on the Python side.
            if self._dtype.ir == "bf16":
                raw = _ext.buf_numpy(self._parent_handle, (total_elems,), "u16")
                base = raw.view(np_dtype)
            else:
                base = _ext.buf_numpy(self._parent_handle, (total_elems,), self._dtype.ir)
            elem_offset = self._offset // self._dtype.itemsize
            if elem_offset == 0:
                return np.lib.stride_tricks.as_strided(
                    base,
                    shape=self._shape,
                    strides=self._strides,
                )
            return np.lib.stride_tricks.as_strided(
                base[elem_offset:],
                shape=self._shape,
                strides=self._strides,
            )
        arr_type = ctypes.c_uint8 * self._total_nbytes
        raw_buf = (arr_type).from_address(self._raw_ptr)
        base = np.frombuffer(raw_buf, dtype=np_dtype)
        elem_offset = self._offset // self._dtype.itemsize
        return np.lib.stride_tricks.as_strided(
            base[elem_offset:],
            shape=self._shape,
            strides=self._strides,
        )

    def __array__(self, dtype=None, copy=None):

        self._ensure_synced()
        arr = self.numpy
        if dtype is not None and np.dtype(dtype) != arr.dtype:
            return np.asarray(arr, dtype=dtype)
        return arr

    @property
    def __array_interface__(self) -> dict[str, Any]:
        self._ensure_synced()
        return self.numpy.__array_interface__

    # --- Compatibility ---

    @property
    def ctypes(self):
        class CTypes:
            def __init__(self, ptr):
                self.data = ptr

        return CTypes(self.data_ptr)

    @property
    def flags(self):
        class Flags:
            def __init__(self, contig):
                self.c_contiguous = contig

        return Flags(self.is_contiguous())

    # --- Lifecycle ---

    def __del__(self):
        if self._owns_aligned:
            if self._materializer is not None:
                return
            try:
                if _ext._training_mode_flag:
                    return
                from alloy._dispatch.buf_utils import (
                    _alloy_buf_map,
                )  # scoped: avoid cycle (_buf_utils imports AlloyBuffer)

                _alloy_buf_map.pop(self.data_ptr, None)
                # Phantom buffers (record-only compile) have no Metal handle.
                if self._parent_handle >= 0:
                    _ext.buf_release(self._parent_handle)
            except (AttributeError, ImportError):
                pass

    def __repr__(self) -> str:
        if self._materializer is not None:
            return f"AlloyBuffer({self._shape}, dtype={self._dtype}, deferred)"
        src = (
            f"handle={self._parent_handle}"
            if self._parent_handle >= 0
            else f"ptr=0x{self._raw_ptr:x}"
        )
        return (
            f"AlloyBuffer({src}, offset={self._offset}, shape={self._shape}, dtype={self._dtype})"
        )

    # --- Arithmetic operators (dispatch via _buf_ops registry) ---

    def __add__(self, other) -> Self:
        return _buf_ops["add"](self, other)

    def __radd__(self, other) -> Self:
        return _buf_ops["add"](other, self)

    def __mul__(self, other) -> Self:
        return _buf_ops["mul"](self, other)

    def __rmul__(self, other) -> Self:
        return _buf_ops["mul"](other, self)

    def __sub__(self, other) -> Self:
        return _buf_ops["sub"](self, other)

    def __rsub__(self, other) -> Self:
        return _buf_ops["sub"](other, self)

    def __truediv__(self, other) -> Self:
        return _buf_ops["div"](self, other)

    def __rtruediv__(self, other) -> Self:
        return _buf_ops["div"](other, self)

    def __matmul__(self, other) -> Self:
        return _buf_ops["matmul"](self, other)

    def __neg__(self) -> Self:
        return _buf_ops["neg"](self)

    def neg(self) -> Self:
        return _buf_ops["neg"](self)

    # --- Comparison operators ---

    def __le__(self, other) -> Self:
        return _buf_ops["le"](self, other)

    def __ge__(self, other) -> Self:
        return _buf_ops["ge"](self, other)

    def __gt__(self, other) -> Self:
        return _buf_ops["gt"](self, other)

    def __lt__(self, other) -> Self:
        return _buf_ops["lt"](self, other)

    # --- Fluent unary ops ---

    def rsqrt(self) -> Self:
        return _buf_ops["rsqrt"](self)

    def exp(self) -> Self:
        return _buf_ops["exp"](self)

    def log(self) -> Self:
        return _buf_ops["log"](self)

    def sqrt(self) -> Self:
        return _buf_ops["sqrt"](self)

    def sin(self) -> Self:
        return _buf_ops["sin"](self)

    def cos(self) -> Self:
        return _buf_ops["cos"](self)

    def abs(self) -> Self:
        return _buf_ops["abs"](self)

    def erf(self) -> Self:
        return _buf_ops["erf"](self)

    def sigmoid(self) -> Self:
        return _buf_ops["sigmoid"](self)

    def relu(self) -> Self:
        return _buf_ops["relu"](self)

    def floor(self) -> Self:
        return _buf_ops["floor"](self)

    def tanh(self) -> Self:
        return _buf_ops["tanh"](self)

    def gelu(self, approximate: str = "none") -> Self:
        return _buf_ops["gelu"](self, approximate)

    def clamp(self, lo=None, hi=None) -> Self:
        return _buf_ops["clamp"](self, lo, hi)

    # --- Fluent shape/type ops ---

    def _is_dense_block(self) -> bool:
        """True if the elements occupy a C-contiguous block for this shape once
        size-1 dims (whose strides are don't-care) are ignored. Such a buffer is
        already dense in memory, so it can be re-described with contiguous strides
        at the same offset — no copy. The M=1 decode case: a (1, N) slice of an
        (M, concat) batched-projection result is a single contiguous row that only
        looks strided because it carries the parent's concat-width row stride.

        Not folded into `contiguous()`: a view-instead-of-copy aliases the source,
        corrupting callers that mutate the result in place. Used only where the
        consumer is known read-only."""
        expected = self._dtype.itemsize
        for size, stride in zip(reversed(self._shape), reversed(self._strides)):
            if size == 1:
                continue
            if stride != expected:
                return False
            expected *= size
        return True

    def as_dense_view(self) -> "AlloyBuffer":
        """Re-describe a dense-block buffer with contiguous strides at the same
        offset (no copy), else fall back to a real contiguify. Safe ONLY when the
        result is not mutated in place — it may alias the source."""
        if self._is_dense_block():
            return self._view(
                self._shape,
                _compute_contiguous_strides(self._shape, self._dtype.itemsize),
            )
        return self.contiguous()

    def contiguous(self, force: bool = False) -> Self:
        if self.is_contiguous() and self._offset == 0 and not force:
            return self
        return _buf_ops["contiguify"](self)

    def expand(self, *shape) -> Self:
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return _buf_ops["expand"](self, shape)

    def to(self, dtype) -> Self:
        return _buf_ops["to_dtype"](self, dtype)

    # --- Fluent reductions ---

    def softmax(self, dim: int = -1) -> Self:
        return _buf_ops["softmax"](self, dim)

    def sum(self, dim=None, keepdim: bool = False) -> Self:
        return _buf_ops["sum"](self, dim, keepdim)

    def mean(self, dim=None, keepdim: bool = False) -> Self:
        return _buf_ops["mean"](self, dim, keepdim)
