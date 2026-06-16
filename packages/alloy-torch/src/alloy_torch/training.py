"""Explicit training-preview boundary for Alloy torch integration."""

from __future__ import annotations

import sys
import warnings
from contextlib import AbstractContextManager

import torch

from alloy._compiler.dtypes import DType, from_ir
from alloy._runtime import _metal_ext  # type: ignore[attr-defined]
from alloy_torch.mode import is_training_mode_enabled, set_training_mode_enabled
from alloy_torch.tensor_bridge import IR_TO_TORCH, make_tensor_from_ptr


class AlloyTrainingPreviewWarning(UserWarning):
    """Training-mode dispatch is preview-quality and not production-ready.

    Silence with:
        warnings.filterwarnings("ignore", category=AlloyTrainingPreviewWarning)
    """


_TRAINING_PREVIEW_WARNED = False
_TRAINING_WITHOUT_MODE_WARNED = False


def _emit_training_preview_warning() -> None:
    """Fired once per process from set_training_mode(True)."""
    global _TRAINING_PREVIEW_WARNED
    if _TRAINING_PREVIEW_WARNED:
        return
    _TRAINING_PREVIEW_WARNED = True
    banner = (
        "\n" + "=" * 72 + "\n"
        "  Alloy training mode: PREVIEW - not production-ready.\n\n"
        "  - Numerics may diverge from eager torch on some shapes/models.\n"
        "  - Performance is best-effort; inference is the supported path.\n"
        "  - Please file issues if your model breaks.\n\n"
        "  Silence: warnings.filterwarnings('ignore',\n"
        "             category=alloy_torch.training.AlloyTrainingPreviewWarning)\n"
        + "=" * 72
        + "\n"
    )
    sys.stderr.write(banner)
    sys.stderr.flush()
    warnings.warn(
        "Alloy training is a preview feature - see stderr banner for details.",
        category=AlloyTrainingPreviewWarning,
        stacklevel=3,
    )


def warn_if_backward_without_mode() -> None:
    """Warn once when AOT requests a backward graph without explicit training mode."""
    global _TRAINING_WITHOUT_MODE_WARNED
    if is_training_mode_enabled() or _TRAINING_WITHOUT_MODE_WARNED:
        return
    _TRAINING_WITHOUT_MODE_WARNED = True
    banner = (
        "\n" + "=" * 72 + "\n"
        "  Alloy: backward graph requested without set_training_mode(True).\n\n"
        "  If you intend to TRAIN:\n"
        "    from alloy_torch.training import set_training_mode\n"
        "    set_training_mode(True)   # call BEFORE torch.compile()\n"
        "    (training is preview-quality - see AlloyTrainingPreviewWarning.)\n\n"
        "  If you intend to do INFERENCE:\n"
        "    model.eval()\n"
        "    with torch.no_grad():  # or @torch.inference_mode()\n"
        "        ...                 # silences this warning\n\n"
        "  Without set_training_mode(True), most backward ops will fail.\n" + "=" * 72 + "\n"
    )
    sys.stderr.write(banner)
    sys.stderr.flush()
    warnings.warn(
        "Alloy compiled a backward graph without set_training_mode(True); "
        "see stderr banner for details.",
        category=AlloyTrainingPreviewWarning,
        stacklevel=3,
    )


PackedAlloyTensor = tuple[int, int, tuple[int, ...], tuple[int, ...], torch.dtype]
PackedTensor = torch.Tensor | PackedAlloyTensor

_saved_tensors_hook_ctx: AbstractContextManager[None] | None = None
_pack_ptr_cache: dict[int, int] = {}
_unpack_cache: dict[PackedAlloyTensor, torch.Tensor] = {}
_TORCH_TO_IR_DTYPE: dict[torch.dtype, DType] = {}


def _pack_alloy_tensor(tensor: torch.Tensor) -> PackedTensor:
    """Pack hook: if tensor is in alloy memory, store handle metadata instead of data."""
    storage_ptr = tensor.untyped_storage().data_ptr()
    handle = _pack_ptr_cache.get(storage_ptr, -1)
    if handle < 0:
        handle = _metal_ext.buf_handle_for_ptr(storage_ptr)
        if handle >= 0:
            _pack_ptr_cache[storage_ptr] = handle
        else:
            return tensor
    return (
        handle,
        int(tensor.storage_offset()) * tensor.element_size(),
        tuple(int(s) for s in tensor.shape),
        tuple(int(s) * tensor.element_size() for s in tensor.stride()),
        tensor.dtype,
    )


def _unpack_alloy_tensor(packed: PackedTensor) -> torch.Tensor:
    """Unpack hook: reconstruct tensor from alloy handle metadata, cached."""
    if isinstance(packed, torch.Tensor):
        return packed
    cached = _unpack_cache.get(packed)
    if cached is not None:
        return cached
    handle, byte_offset, shape, strides_bytes, torch_dtype = packed
    base_ptr = _metal_ext.buf_ptr(handle)
    nbytes = _metal_ext.buf_nbytes(handle)
    dt = _TORCH_TO_IR_DTYPE.get(torch_dtype)
    if dt is None:
        raise RuntimeError(f"Unknown dtype in unpack: {torch_dtype}")
    tensor = make_tensor_from_ptr(base_ptr, shape, dt, byte_offset, nbytes, strides_bytes)
    _unpack_cache[packed] = tensor
    return tensor


def _init_torch_to_ir() -> None:
    for ir_name in ("f16", "f32", "bf16", "i8", "i16", "i32", "i64", "u8", "u32"):
        dt = from_ir(ir_name)
        torch_dt = IR_TO_TORCH.get(ir_name)
        if torch_dt is not None:
            _TORCH_TO_IR_DTYPE[torch_dt] = dt


def set_training_mode(mode: bool) -> None:
    """Enable/disable training support. Must be called before torch.compile()."""
    global _saved_tensors_hook_ctx
    if mode:
        _emit_training_preview_warning()
    set_training_mode_enabled(mode)

    _metal_ext.set_training_mode(mode)
    # Prevent buf_release in __del__ during training to avoid non-deterministic
    # buffer corruption from Python's cycle collector.
    _metal_ext._training_mode_flag = mode

    # Install saved_tensors_hooks to keep alloy-backed tensors in page-aligned
    # memory through the backward pass, avoiding large per-step memmoves.
    if mode:
        _init_torch_to_ir()
        _saved_tensors_hook_ctx = torch.autograd.graph.saved_tensors_hooks(
            _pack_alloy_tensor, _unpack_alloy_tensor
        )
        _saved_tensors_hook_ctx.__enter__()
    elif _saved_tensors_hook_ctx is not None:
        _saved_tensors_hook_ctx.__exit__(None, None, None)
        _saved_tensors_hook_ctx = None
