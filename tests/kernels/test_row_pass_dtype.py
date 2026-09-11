"""Row-pass fusion over non-f32 buffers.

The fused row kernel is typed from the buffer's dtype; getting that wrong reads
the rows at the wrong element width, so the front rows consume double-width data
and the tail reads nothing.
"""

import pytest
import torch

import alloy_torch  # noqa: F401 — registers the "alloy" backend
from alloy._compiler.dtypes import bfloat16, float16, float32, int32
from alloy._dispatch.row_pass import _dtype_short
from alloy_torch.training import set_training_mode

set_training_mode(True)


def test_dtype_short_reads_alloy_dtypes():
    assert _dtype_short(float32) == "f32"
    assert _dtype_short(bfloat16) == "bf16"
    assert _dtype_short(float16) == "f16"
    assert _dtype_short(int32) == "i32"


def test_dtype_short_reads_torch_dtypes():
    assert _dtype_short(torch.float32) == "f32"
    assert _dtype_short(torch.bfloat16) == "bf16"


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(8, 32), (4, 64, 64)])
def test_row_reduce_then_elementwise(dtype, shape):
    """Reduce fused with a following scalar add — the rms-norm forward shape."""
    torch.set_grad_enabled(False)
    torch._dynamo.reset()
    torch.manual_seed(0)
    x = (torch.randn(*shape) + 3.0).to(dtype)

    def fn(t):
        return t.pow(2).mean(-1, keepdim=True) + 1e-6

    got = torch.compile(fn, backend="alloy", dynamic=False)(x).float()
    ref = fn(x).float()
    torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rms_norm_gradient_non_f32(dtype):
    """The gradient a bf16 RMS norm returns to its input."""
    torch.set_grad_enabled(True)
    shape = (8, 32)
    torch.manual_seed(7)
    weight = torch.randn(*shape)

    def fn(t, eps=1e-6):
        return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + eps)

    grads = {}
    for backend in ("eager", "alloy"):
        torch._dynamo.reset()
        torch.manual_seed(0)
        x = (torch.randn(*shape) + 3.0).to(dtype).requires_grad_(True)
        f = fn if backend == "eager" else torch.compile(fn, backend="alloy", dynamic=False)
        (f(x).float() * weight).sum().backward()
        grads[backend] = x.grad.detach().float().clone()

    torch.testing.assert_close(grads["alloy"], grads["eager"], rtol=5e-2, atol=5e-2)
