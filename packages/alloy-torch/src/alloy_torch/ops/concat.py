"""Concat handlers for torch op lowering."""

from collections.abc import Sequence
import math

from alloy._runtime.alloy_buffer import AlloyBuffer
from alloy.std.indexing import k_concat_2, k_concat_3
from alloy_torch.ops.common import _normalize_dim
from alloy_torch.ops.values import _coerce_lazy_value


def _concat_3d_strides(buf: AlloyBuffer, dim: int) -> tuple[int, int, int] | None:
    """Compute 3-stride native concat metadata for representable views."""
    if buf._offset != 0:
        return None
    shape = buf.shape
    byte_strides = buf._strides
    itemsize = buf._dtype.itemsize
    ndim = len(shape)

    elem_strides = tuple(stride // itemsize for stride in byte_strides)
    for byte_stride, elem_stride in zip(byte_strides, elem_strides):
        if elem_stride * itemsize != byte_stride:
            return None

    cat_stride = elem_strides[dim]

    inner_stride = 1
    if dim + 1 < ndim:
        expected_bytes = itemsize
        for index in range(ndim - 1, dim, -1):
            if byte_strides[index] != expected_bytes:
                return None
            expected_bytes *= shape[index]

    if dim == 0:
        outer_stride = 0
    else:
        outer_stride: int | None = None
        for index in range(dim):
            if shape[index] == 1:
                continue
            contiguous_stride = math.prod(shape[index + 1 : dim])
            if contiguous_stride == 0:
                return None
            candidate = elem_strides[index] // contiguous_stride
            if elem_strides[index] != candidate * contiguous_stride:
                return None
            if outer_stride is None:
                outer_stride = candidate
            elif outer_stride != candidate:
                return None
        if outer_stride is None:
            outer_stride = 0

    return outer_stride, cat_stride, inner_stride


def _cat(values: Sequence[AlloyBuffer], dim: int = 0) -> AlloyBuffer:
    lazy_values = [_coerce_lazy_value(value) for value in values]
    if not lazy_values:
        raise NotImplementedError("aten.cat requires at least one input")
    nonempty_lazy = [value for value in lazy_values if value.size != 0]
    if not nonempty_lazy:
        return lazy_values[0]
    if len(nonempty_lazy) == 1:
        return nonempty_lazy[0]

    if len(nonempty_lazy) == 2:
        a_buf, b_buf = nonempty_lazy
        dim = _normalize_dim(dim, a_buf.ndim)
        out_shape_list = list(a_buf.shape)
        out_shape_list[dim] = a_buf.shape[dim] + b_buf.shape[dim]
        out_shape = tuple(out_shape_list)
        n_elements = math.prod(out_shape)
        cat_total = out_shape[dim]
        split_d = a_buf.shape[dim]
        inner = math.prod(out_shape[dim + 1 :]) if dim + 1 < len(out_shape) else 1

        stride_kwargs: dict[str, int] = {}
        for buf, prefix in ((a_buf, "A"), (b_buf, "B")):
            if buf.is_contiguous():
                continue
            strides_3d = _concat_3d_strides(buf, dim)
            if strides_3d is not None:
                outer_stride, cat_stride, inner_stride = strides_3d
                stride_kwargs[f"{prefix}_OUTER_STRIDE"] = outer_stride
                stride_kwargs[f"{prefix}_CAT_STRIDE"] = cat_stride
                stride_kwargs[f"{prefix}_INNER_STRIDE"] = inner_stride
            elif prefix == "A":
                a_buf = a_buf.contiguous()
            else:
                b_buf = b_buf.contiguous()

        result = k_concat_2(
            a_buf,
            b_buf,
            N=n_elements,
            CAT_TOTAL=cat_total,
            SPLIT_D=split_d,
            INNER=inner,
            **stride_kwargs,
        )
        return result.reshape(out_shape)

    if len(nonempty_lazy) == 3:
        a_buf, b_buf, c_buf = nonempty_lazy
        dim = _normalize_dim(dim, a_buf.ndim)
        out_shape_list = list(a_buf.shape)
        out_shape_list[dim] = a_buf.shape[dim] + b_buf.shape[dim] + c_buf.shape[dim]
        out_shape = tuple(out_shape_list)
        n_elements = math.prod(out_shape)
        cat_total = out_shape[dim]
        split_ab = a_buf.shape[dim]
        split_bc = a_buf.shape[dim] + b_buf.shape[dim]
        inner = math.prod(out_shape[dim + 1 :]) if dim + 1 < len(out_shape) else 1

        stride_kwargs: dict[str, int] = {}
        fallback = False
        for buf, prefix in ((a_buf, "A"), (b_buf, "B"), (c_buf, "C")):
            if buf.is_contiguous():
                continue
            strides_3d = _concat_3d_strides(buf, dim)
            if strides_3d is not None:
                outer_stride, cat_stride, inner_stride = strides_3d
                stride_kwargs[f"{prefix}_OUTER_STRIDE"] = outer_stride
                stride_kwargs[f"{prefix}_CAT_STRIDE"] = cat_stride
                stride_kwargs[f"{prefix}_INNER_STRIDE"] = inner_stride
            else:
                fallback = True
                break
        if not fallback:
            result = k_concat_3(
                a_buf,
                b_buf,
                c_buf,
                N=n_elements,
                CAT_TOTAL=cat_total,
                SPLIT_AB=split_ab,
                SPLIT_BC=split_bc,
                INNER=inner,
                **stride_kwargs,
            )
            return result.reshape(out_shape)

    remaining = list(nonempty_lazy)
    dim = _normalize_dim(dim, remaining[0].ndim)
    while len(remaining) > 1:
        pairs: list[AlloyBuffer] = []
        for index in range(0, len(remaining) - 1, 2):
            pairs.append(_cat((remaining[index], remaining[index + 1]), dim=dim))
        if len(remaining) % 2 == 1:
            pairs.append(remaining[-1])
        remaining = pairs
    return remaining[0]
