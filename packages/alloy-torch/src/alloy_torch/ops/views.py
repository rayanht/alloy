"""View and metadata-only handlers for torch op lowering."""

from collections.abc import Sequence

from alloy._runtime.alloy_buffer import AlloyBuffer
from alloy_torch.ops.common import _expand_lazy_buffer, _normalize_dim, _normalize_shape

ShapeLike = int | Sequence[int]


def _alias(x: AlloyBuffer) -> AlloyBuffer:
    return x


def _transpose_dims(x: AlloyBuffer, dim0: int, dim1: int) -> AlloyBuffer:
    axes = list(range(x.ndim))
    dim0 = _normalize_dim(dim0, x.ndim)
    dim1 = _normalize_dim(dim1, x.ndim)
    axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
    return x.transpose(*axes)


def _slice_tensor(
    x: AlloyBuffer,
    dim: int = 0,
    start: int | None = None,
    end: int | None = None,
    step: int = 1,
) -> AlloyBuffer:
    dim = _normalize_dim(dim, x.ndim)
    dim_size = x.shape[dim]
    slice_start = start if start is not None else 0
    slice_end = end if end is not None else dim_size
    if slice_start < 0:
        slice_start = dim_size + slice_start
    if slice_end < 0:
        slice_end = dim_size + slice_end
    if slice_end > dim_size:
        slice_end = dim_size
    if slice_start > dim_size:
        slice_start = dim_size
    return x.slice(dim, int(slice_start), int(slice_end), int(step))


def _split_tensor(
    x: AlloyBuffer, split_size: int, dim: int = 0
) -> tuple[AlloyBuffer, ...]:
    dim = _normalize_dim(dim, x.ndim)
    size = x.shape[dim]
    result: list[AlloyBuffer] = []
    for start in range(0, size, int(split_size)):
        stop = min(start + int(split_size), size)
        result.append(x.slice(dim, start, stop))
    return tuple(result)


def _split_with_sizes(
    x: AlloyBuffer, split_sizes: Sequence[int], dim: int = 0
) -> tuple[AlloyBuffer, ...]:
    dim = _normalize_dim(dim, x.ndim)
    start = 0
    result: list[AlloyBuffer] = []
    for split_size in split_sizes:
        stop = start + int(split_size)
        result.append(x.slice(dim, start, stop))
        start = stop
    return tuple(result)


def _expand(x: AlloyBuffer, size: ShapeLike) -> AlloyBuffer:
    target = list(_normalize_shape(size))
    offset = len(target) - x.ndim
    for i, dim in enumerate(target):
        if dim == -1:
            target[i] = x.shape[i - offset]
    return _expand_lazy_buffer(x, tuple(target))


def _repeat(x: AlloyBuffer, repeats: ShapeLike) -> AlloyBuffer:
    normalized_repeats = _normalize_shape(repeats)
    pad = len(normalized_repeats) - x.ndim
    if pad > 0:
        x = x.reshape((1,) * pad + x.shape)
    if all(repeat == 1 for repeat in normalized_repeats):
        return x

    shape = x.shape
    interleaved_x: list[int] = []
    for dim_size in shape:
        interleaved_x.extend([1, dim_size])
    x_reshaped = x.reshape(tuple(interleaved_x))

    target: list[int] = []
    for repeat, dim_size in zip(normalized_repeats, shape):
        target.extend([repeat, dim_size])
    x_expanded = _expand_lazy_buffer(x_reshaped, tuple(target))
    out_shape = tuple(repeat * dim_size for repeat, dim_size in zip(normalized_repeats, shape))
    return x_expanded.reshape(out_shape)


def _unsqueeze(x: AlloyBuffer, dim: int) -> AlloyBuffer:
    dim = _normalize_dim(dim, x.ndim + 1)
    shape = list(x.shape)
    shape.insert(dim, 1)
    return x.reshape(shape)


def _squeeze_dims(x: AlloyBuffer, dims: int | Sequence[int]) -> AlloyBuffer:
    dims_tuple = dims if isinstance(dims, Sequence) and not isinstance(dims, str) else (dims,)
    shape = list(x.shape)
    strides = list(x._strides)
    axes = sorted((_normalize_dim(int(dim), x.ndim) for dim in dims_tuple), reverse=True)
    for axis in axes:
        if shape[axis] == 1:
            shape.pop(axis)
            strides.pop(axis)

    new_buf = AlloyBuffer(
        x._parent_handle,
        x._offset,
        x._shape,
        x._strides,
        x._dtype,
        raw_ptr=x._raw_ptr,
        total_nbytes=x._total_nbytes,
    )
    new_buf.reinterpret(tuple(shape), tuple(strides))
    return x._view_of(new_buf)


def _select_int(x: AlloyBuffer, dim: int, index: int) -> AlloyBuffer:
    dim = _normalize_dim(dim, x.ndim)
    idx = int(index)
    if idx < 0:
        idx = x.shape[dim] + idx
    sliced = x.slice(dim, idx, idx + 1)
    return _squeeze_dims(sliced, (dim,))


def _unfold(x: AlloyBuffer, dimension: int, size: int, step: int) -> AlloyBuffer:
    """`aten.unfold` — sliding windows of `size` along `dimension`, strided by
    `step`: `dimension` shrinks to the window count and a new trailing axis of
    `size` is appended."""
    dim = _normalize_dim(int(dimension), x.ndim)
    size, step = int(size), int(step)
    n_windows = (x.shape[dim] - size) // step + 1
    orig = x._strides[dim]
    new_shape = list(x.shape)
    new_shape[dim] = n_windows
    new_shape.append(size)
    new_strides = list(x._strides)
    new_strides[dim] = orig * step
    new_strides.append(orig)
    return x._view(tuple(new_shape), tuple(new_strides))
