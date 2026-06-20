"""Composable index transforms for fused kernel codegen.

Each transform maps an index expression (MSL string) to a new index expression.
Two modes: flat (1D elem chains) and tile_2d (row, col for 2D tile kernels).
Transforms compose by chaining: scatter(broadcast(flat_idx)).

The codegen calls transform.flat(store_offs) or transform.tile_2d(_gm, _gn)
depending on context. No special cases in the codegen — the transform handles it.
"""

from __future__ import annotations


class IndexTransform:
    """Base class for composable index transforms.

    Subclasses implement flat() and tile_2d() to emit MSL index expressions.
    """

    def flat(self, expr: str) -> str:
        """Transform a flat 1D index expression. Returns MSL expression."""
        raise NotImplementedError

    def tile_2d(self, row_expr: str, col_expr: str) -> str:
        """Transform (row, col) 2D index expressions. Returns MSL expression."""
        raise NotImplementedError


class IdentityTransform(IndexTransform):
    """Passthrough: index unchanged. Used for contiguous same-shape extras."""

    def flat(self, expr: str) -> str:
        return expr

    def tile_2d(self, row_expr: str, col_expr: str) -> str:
        return f"{row_expr} * _N + {col_expr}"


class ScalarBroadcastTransform(IndexTransform):
    """All-zero strides: single element, always index 0."""

    def flat(self, expr: str) -> str:
        return "0"

    def tile_2d(self, row_expr: str, col_expr: str) -> str:
        return "0"


class BroadcastTransform(IndexTransform):
    """Modular indexing for broadcast extras smaller than the output.

    For a buffer of `size` elements consumed by a kernel with `total` elements,
    the index wraps: flat_idx % size.

    When the buffer has trailing stride-0 dims (inner_repeat > 1), the index
    accounts for the repeat: (flat_idx / inner_repeat) % size.
    """

    def __init__(self, size: int, inner_repeat: int = 1, row_stride: int = 0):
        self.size = size
        self.inner_repeat = inner_repeat
        self.row_stride = row_stride

    def flat(self, expr: str) -> str:
        if self.inner_repeat > 1:
            return f"(({expr}) / {self.inner_repeat}u) % {self.size}u"
        return f"({expr}) % {self.size}u"

    def tile_2d(self, row_expr: str, col_expr: str) -> str:
        rs = self.row_stride
        linear = f"({row_expr} * {rs}u + {col_expr})"
        if self.inner_repeat > 1:
            return f"({linear} / {self.inner_repeat}u) % {self.size}u"
        return f"{linear} % {self.size}u"


class RowBroadcastTransform(IndexTransform):
    """Row-stride broadcast: extra has row_stride=0 (bias broadcast along batch dim).

    The buffer is indexed only by column: name[col]. The row dimension is ignored.
    This handles the common case of bias (N,) broadcast to (M, N).

    `size` is the column dimension (N), needed only by `flat()` to wrap with
    modular arithmetic; the tile_2d path ignores the row coordinate directly.
    When N can't be inferred (size=0), `flat()` returns expr unchanged; callers
    using a flat-indexing kernel must pass size to avoid an out-of-bounds index.
    """

    def __init__(self, size: int = 0):
        self.size = size

    def flat(self, expr: str) -> str:
        if self.size > 0:
            return f"({expr}) % {self.size}u"
        return expr

    def tile_2d(self, row_expr: str, col_expr: str) -> str:
        return col_expr


class ColumnBroadcastTransform(IndexTransform):
    """Column-broadcast: extra has size == M (row dim), broadcast across columns.

    The buffer is indexed only by row: name[row]. The column dimension is ignored.
    This handles bias (M,) broadcast to (M, N) — each row gets its bias value.
    """

    def flat(self, expr: str) -> str:
        return expr

    def tile_2d(self, row_expr: str, col_expr: str) -> str:
        return row_expr


class StridedTransform(IndexTransform):
    """General row-stride indexing: name[row * row_stride + col].

    Handles non-contiguous extras where the row stride differs from the
    column count (e.g., a slice of a larger buffer).
    """

    def __init__(self, row_stride: int):
        self.row_stride = row_stride

    def flat(self, expr: str) -> str:
        return expr

    def tile_2d(self, row_expr: str, col_expr: str) -> str:
        return f"{row_expr} * {self.row_stride}u + {col_expr}"


class ScatterTransform(IndexTransform):
    """Permuted indexing for view+permute boundaries.

    Decomposes a flat index into multi-dimensional indices using the source
    shape, then reindexes with the target's strides. This maps from the
    kernel's iteration order to the extra buffer's memory layout.

    For 2D tile mode, decomposes _gm (the row index) into the higher dims
    and uses _gn for the innermost dim.
    """

    def __init__(self, nd_shape: tuple[int, ...], nd_strides: tuple[int, ...]):
        self.nd_shape = nd_shape
        self.nd_strides = nd_strides

    def flat(self, expr: str) -> str:
        if len(self.nd_shape) <= 2:
            # 2D: simple row*stride + col
            rs = self.nd_strides[0] if len(self.nd_strides) > 0 else 0
            return f"({expr}) / {self.nd_shape[-1]}u * {rs}u + ({expr}) % {self.nd_shape[-1]}u"

        # >2D: standard unravel
        # d_i = (flat / prod(shape[i+1:])) % shape[i]
        # index = sum(d_i * stride_i)
        N = self.nd_shape[-1]
        parts = []
        for d in range(len(self.nd_shape) - 1):
            inner_size = 1
            for dd in range(d + 1, len(self.nd_shape)):
                inner_size *= self.nd_shape[dd]
            idx = f"(({expr}) / {inner_size}u) % {self.nd_shape[d]}u"
            parts.append(f"({idx}) * {self.nd_strides[d]}u")
        parts.append(f"(({expr}) % {N}u)")
        return " + ".join(parts)

    def tile_2d(self, row_expr: str, col_expr: str) -> str:
        if len(self.nd_shape) <= 2:
            rs = self.nd_strides[0] if len(self.nd_strides) > 0 else 0
            return f"{row_expr} * {rs}u + {col_expr}"

        # >2D: decompose row into higher dims.
        # Find the column dim: the dim with the finest stride (typically 1).
        # If the last dim has size 1, it's a collapsed permutation dim — skip it
        # and use _gn for the actual column dim.
        col_dim = len(self.nd_shape) - 1  # default: last
        if self.nd_shape[-1] == 1 and len(self.nd_shape) > 2:
            # Last dim collapsed — find the dim with stride 1 (the real column)
            for d in range(len(self.nd_strides)):
                if self.nd_strides[d] == 1:
                    col_dim = d
                    break

        # Decompose row into all dims except col_dim
        # row_product = product of all non-col dims' sizes
        non_col_dims = [d for d in range(len(self.nd_shape)) if d != col_dim]
        parts = []
        for i, d in enumerate(non_col_dims):
            inner_size = 1
            for j in range(i + 1, len(non_col_dims)):
                inner_size *= self.nd_shape[non_col_dims[j]]
            idx = f"({row_expr} / {inner_size}u) % {int(self.nd_shape[d])}u"
            parts.append(f"({idx}) * {int(self.nd_strides[d])}u")
        # Column dim uses _gn directly (scaled by stride if not 1)
        if self.nd_strides[col_dim] != 1:
            parts.append(f"{col_expr} * {int(self.nd_strides[col_dim])}u")
        else:
            parts.append(col_expr)
        return " + ".join(parts)
