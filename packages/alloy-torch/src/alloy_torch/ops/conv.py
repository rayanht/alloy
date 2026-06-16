"""Convolution handlers for torch op lowering."""

from collections.abc import Sequence
from typing import cast

from alloy._dispatch.buf_utils import _alloc_aligned
from alloy._dispatch.kernel import KernelFunction
from alloy._runtime.alloy_buffer import AlloyBuffer
from alloy.std.gemm import dot_transpose_rhs
from alloy.std.indexing import depthwise_conv1d, im2col_1d, im2col_2d

_dot_transpose_rhs = cast(KernelFunction, dot_transpose_rhs)


def _convolution(
    x: AlloyBuffer,
    weight: AlloyBuffer,
    bias: AlloyBuffer | None,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    transposed: bool,
    output_padding: Sequence[int],
    groups: int,
) -> AlloyBuffer:
    """Handle aten.convolution.default for Conv1d and Conv2d."""
    del output_padding
    if transposed or any(d != 1 for d in dilation):
        raise NotImplementedError(
            f"Alloy conv: transposed={transposed}, dilation={dilation} not supported"
        )

    x_shape = x.shape
    w_shape = weight.shape

    # Depthwise Conv1d (one kernel per channel) — used by qwen3.5's
    # GatedDeltaNet causal conv. groups must equal both in_channels and
    # out_channels; weight shape is (C, 1, K). Skip the im2col + GEMM
    # detour and dispatch the per-channel kernel directly.
    if (
        groups != 1
        and len(x_shape) == 3
        and len(w_shape) == 3
        and groups == x_shape[1]
        and groups == w_shape[0]
        and w_shape[1] == 1
    ):
        batch, channels, in_len = x_shape
        _, _, kernel = w_shape
        stride_x = stride[0]
        padding_x = padding[0]
        out_len = (in_len + 2 * padding_x - kernel) // stride_x + 1
        x_contig = x.contiguous().reshape((batch * channels * in_len,))
        w_contig = weight.contiguous().reshape((channels * kernel,))
        out_buf = _alloc_aligned((batch * channels * out_len,), x.dtype)
        result_lazy = depthwise_conv1d[(batch * channels * out_len,)](
            x_contig,
            w_contig,
            out_buf,
            BATCH=batch,
            C=channels,
            IN_LEN=in_len,
            OUT_LEN=out_len,
            K=kernel,
            STRIDE=stride_x,
            PADDING=padding_x,
        )
        result = result_lazy.reshape((batch, channels, out_len))
        if bias is not None:
            result = result + bias.reshape((1, channels, 1))
        return result

    if groups != 1:
        raise NotImplementedError(
            f"Alloy conv: groups={groups} only supported for depthwise Conv1d"
        )

    if len(x_shape) == 3 and len(w_shape) == 3:
        batch, in_c, in_len = x_shape
        out_c, _, kernel = w_shape
        stride_x = stride[0]
        padding_x = padding[0]
        out_len = (in_len + 2 * padding_x - kernel) // stride_x + 1
        channel_kernel = in_c * kernel

        flat_x = x.reshape((batch, in_c * in_len))
        col_buf = _alloc_aligned((batch * out_len * channel_kernel,), x.dtype)
        col_lazy = im2col_1d[batch * out_len](
            flat_x,
            col_buf,
            IN_C=in_c,
            IN_LEN=in_len,
            OUT_LEN=out_len,
            CK=channel_kernel,
            K=kernel,
            STRIDE=stride_x,
            PADDING=padding_x,
        )
        col = col_lazy.reshape((batch * out_len, channel_kernel))

        flat_w = weight.reshape((out_c, channel_kernel))
        gemm_out = _dot_transpose_rhs(flat_w, col)

        if batch == 1:
            result = gemm_out.reshape((1, out_c, out_len))
        else:
            tmp = gemm_out.reshape((out_c, batch, out_len))
            result = tmp.transpose(0, 1).contiguous().reshape((batch, out_c, out_len))

        if bias is not None:
            result = result + bias.reshape((1, out_c, 1))

        return result

    if len(x_shape) == 4 and len(w_shape) == 4:
        batch, in_c, in_h, in_w = x_shape
        out_c, _, kernel_h, kernel_w = w_shape
        stride_h, stride_w = stride[0], stride[1]
        pad_h, pad_w = padding[0], padding[1]
        out_h = (in_h + 2 * pad_h - kernel_h) // stride_h + 1
        out_w = (in_w + 2 * pad_w - kernel_w) // stride_w + 1
        channel_kernel = in_c * kernel_h * kernel_w

        stride_kwargs: dict[str, int] = {}
        if not x.is_contiguous():
            itemsize = x._dtype.itemsize
            element_strides = tuple(s // itemsize for s in x._strides)
            stride_kwargs["X_BATCH_STRIDE"] = element_strides[0]
            stride_kwargs["X_C_STRIDE"] = element_strides[1]
            stride_kwargs["X_H_STRIDE"] = element_strides[2]
            stride_kwargs["X_W_STRIDE"] = element_strides[3]
            stride_kwargs["X_OFFSET"] = x._offset // itemsize
            x_flat = x.root_flat()
        else:
            x_flat = x.reshape((batch, in_c * in_h * in_w))

        col_buf = _alloc_aligned((batch * out_h * out_w * channel_kernel,), x.dtype)
        col_lazy = im2col_2d[batch * out_h * out_w](
            x_flat,
            col_buf,
            IN_C=in_c,
            IN_H=in_h,
            IN_W=in_w,
            OUT_H=out_h,
            OUT_W=out_w,
            KH=kernel_h,
            KW=kernel_w,
            CKK=channel_kernel,
            STRIDE_H=stride_h,
            STRIDE_W=stride_w,
            PAD_H=pad_h,
            PAD_W=pad_w,
            **stride_kwargs,
        )
        col = col_lazy.reshape((batch * out_h * out_w, channel_kernel))

        flat_w = weight.reshape((out_c, channel_kernel))
        out = _alloc_aligned((out_c, batch * out_h * out_w), x.dtype)
        gemm_out = _dot_transpose_rhs(flat_w, col, out)

        if batch == 1:
            result = gemm_out.reshape((1, out_c, out_h, out_w))
        else:
            tmp = gemm_out.reshape((out_c, batch, out_h * out_w))
            result = tmp.transpose(0, 1).contiguous().reshape((batch, out_c, out_h, out_w))

        if bias is not None:
            result = result + bias.reshape((1, out_c, 1, 1))

        return result

    raise NotImplementedError(
        f"Alloy conv: {len(x_shape)}D input not supported (only Conv1d/Conv2d)"
    )
