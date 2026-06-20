"""Gradient correctness helpers — strict alloy-vs-CPU-eager comparisons.

Each helper runs the same computation twice on CPU: once with vanilla PyTorch
eager (the ground truth) and once with ``torch.compile(..., backend="alloy")``.
Forward output and parameter/input gradients are compared with tight tolerances.

  * alloy training mode is enabled at import via ``set_training_mode(True)`` so
    the compiled plan decomposes SDPA and routes optimiser updates through alloy
    memory.
  * The callable and inputs come from factories invoked twice (once per backend)
    so the runs share no state — no stale grads, no aliased tensors.
  * ``torch._dynamo.reset()`` between runs forces a fresh AOT trace; otherwise
    the Dynamo cache may hand the alloy backend a graph guarded against
    different inputs.
"""

from __future__ import annotations

from typing import Any, Callable, cast

import numpy as np
import torch
import torch.nn as nn

import alloy_torch  # noqa: F401 — register the alloy backend
from alloy_torch.training import set_training_mode

set_training_mode(True)

InputOptions = dict[str, Any]
InputSpec = tuple[int, ...] | tuple[tuple[int, ...], InputOptions]


# ---------------------------------------------------------------------------
# Output reduction — produces the scalar the backward pass hangs off
# ---------------------------------------------------------------------------


def _reduce_to_scalar(out: Any) -> torch.Tensor:
    """Turn a forward output (tensor, tuple of tensors, or dict) into a scalar.

    Uses ``.float().sum()`` so the backward seed is dtype-independent.
    """
    if isinstance(out, torch.Tensor):
        return out.float().sum()
    if isinstance(out, (list, tuple)):
        total: torch.Tensor | None = None
        for item in out:
            if isinstance(item, torch.Tensor):
                value = _reduce_to_scalar(item)
                total = value if total is None else total + value
        if total is not None:
            return total
    if isinstance(out, dict):
        total: torch.Tensor | None = None
        for item in out.values():
            if isinstance(item, torch.Tensor):
                value = _reduce_to_scalar(item)
                total = value if total is None else total + value
        if total is not None:
            return total
    if hasattr(out, "loss") and isinstance(out.loss, torch.Tensor):
        return out.loss.float()
    raise TypeError(f"Cannot reduce forward output of type {type(out).__name__} to a scalar")


def _collect_tensor_outputs(out: Any) -> list[torch.Tensor]:
    """Flatten the forward output into a list of tensors for forward-diff check."""
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (list, tuple)):
        return [t for x in out for t in _collect_tensor_outputs(x)]
    if isinstance(out, dict):
        return [t for v in out.values() for t in _collect_tensor_outputs(v)]
    if hasattr(out, "loss") and isinstance(out.loss, torch.Tensor):
        # HF-style ModelOutput: check both loss and logits.
        found = [out.loss]
        if hasattr(out, "logits") and isinstance(out.logits, torch.Tensor):
            found.append(out.logits)
        return found
    return []


# ---------------------------------------------------------------------------
# Assertion — one-stop allclose with helpful failure messages
# ---------------------------------------------------------------------------


def _assert_close(
    actual: torch.Tensor,
    ref: torch.Tensor,
    *,
    atol: float,
    rtol: float,
    tag: str,
) -> None:
    if actual.shape != ref.shape:
        raise AssertionError(f"[{tag}] shape mismatch: alloy={tuple(actual.shape)} vs ref={tuple(ref.shape)}")
    diff = (actual.detach().float() - ref.detach().float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    ref_max = ref.detach().float().abs().max().item()
    allowed = atol + rtol * ref_max
    if max_diff > allowed:
        flat_idx = int(diff.flatten().argmax().item())
        a_val = actual.detach().float().flatten()[flat_idx].item()
        r_val = ref.detach().float().flatten()[flat_idx].item()
        raise AssertionError(
            f"[{tag}] max_diff={max_diff:.3e} > atol+rtol*|ref|={allowed:.3e} "
            f"(atol={atol:.1e}, rtol={rtol:.1e}, |ref|.max={ref_max:.3e}, mean_diff={mean_diff:.3e}); "
            f"worst element: alloy={a_val:.6g} vs ref={r_val:.6g}"
        )


# ---------------------------------------------------------------------------
# Core driver
# ---------------------------------------------------------------------------


def check_grads(
    make_fn: Callable[[], nn.Module | Callable[..., Any]],
    make_inputs: Callable[[], tuple[torch.Tensor, ...]],
    *,
    atol: float = 1e-5,
    rtol: float = 1e-4,
    check_input_grads: bool = True,
    check_param_grads: bool = True,
) -> None:
    """Assert that alloy's forward output and gradients match CPU eager.

    ``make_fn`` and ``make_inputs`` are called twice (reference run, alloy run)
    so each side has independent state. Returning the same module instance twice
    would share ``.grad`` buffers between the two backward passes.
    """
    # ---- Reference (CPU eager) ----
    ref_fn = make_fn()
    ref_inputs = make_inputs()
    ref_out = ref_fn(*ref_inputs) if not isinstance(ref_fn, nn.Module) else ref_fn(*ref_inputs)
    ref_loss = _reduce_to_scalar(ref_out)
    ref_loss.backward()
    ref_forward = [t.detach().clone() for t in _collect_tensor_outputs(ref_out)]
    ref_param_grads: dict[str, torch.Tensor] = {}
    if isinstance(ref_fn, nn.Module) and check_param_grads:
        ref_param_grads = {
            n: p.grad.detach().clone()
            for n, p in ref_fn.named_parameters()
            if p.grad is not None
        }
    ref_input_grads: list[torch.Tensor | None] = []
    if check_input_grads:
        ref_input_grads = [
            a.grad.detach().clone() if isinstance(a, torch.Tensor) and a.grad is not None else None
            for a in ref_inputs
        ]

    # ---- Alloy (torch.compile with alloy backend) ----
    torch._dynamo.reset()
    alloy_fn = make_fn()
    alloy_inputs = make_inputs()
    compiled = torch.compile(alloy_fn, backend="alloy") if isinstance(alloy_fn, nn.Module) else torch.compile(alloy_fn, backend="alloy")
    alloy_out = compiled(*alloy_inputs)
    alloy_loss = _reduce_to_scalar(alloy_out)
    alloy_loss.backward()
    alloy_forward = [t.detach().clone() for t in _collect_tensor_outputs(alloy_out)]

    # ---- Forward diff ----
    assert len(alloy_forward) == len(ref_forward), (
        f"forward output count differs: alloy={len(alloy_forward)} ref={len(ref_forward)}"
    )
    for i, (a, r) in enumerate(zip(alloy_forward, ref_forward)):
        _assert_close(a, r, atol=atol, rtol=rtol, tag=f"forward[{i}]")

    # ---- Parameter gradient diff ----
    if isinstance(alloy_fn, nn.Module) and check_param_grads:
        for name, _p in alloy_fn.named_parameters():
            if name not in ref_param_grads:
                continue
            alloy_g = None
            for n, p in alloy_fn.named_parameters():
                if n == name:
                    alloy_g = p.grad
                    break
            assert alloy_g is not None, f"alloy did not produce a grad for param '{name}'"
            _assert_close(alloy_g, ref_param_grads[name], atol=atol, rtol=rtol, tag=f"param:{name}")

    # ---- Input gradient diff ----
    if check_input_grads:
        for i, (a_in, r_grad) in enumerate(zip(alloy_inputs, ref_input_grads)):
            if r_grad is None:
                continue
            assert a_in.grad is not None, f"alloy did not produce a grad for input[{i}]"
            _assert_close(a_in.grad, r_grad, atol=atol, rtol=rtol, tag=f"input[{i}].grad")

    # Drop dynamo cache so the next test traces fresh.
    torch._dynamo.reset()


# ---------------------------------------------------------------------------
# Convenience builders — reduce boilerplate in the test files
# ---------------------------------------------------------------------------


def fn_factory(fn: Callable[..., Any]) -> Callable[[], Callable[..., Any]]:
    """Wrap a stateless function so it looks like a module factory."""
    return lambda: fn


def inputs_factory(
    *shapes_and_kwargs: InputSpec,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
    requires_grad: bool = True,
) -> Callable[[], tuple[torch.Tensor, ...]]:
    """Build an inputs factory that re-materialises deterministic tensors.

    Each entry is either a shape tuple (uses the top-level dtype/requires_grad
    defaults) or a ``(shape, kwargs)`` tuple where kwargs override defaults
    per-input. Override kwargs: ``dtype``, ``requires_grad``, ``low``/``high``
    for integer tensors, ``kind`` = "randn"|"randint"|"arange".
    """

    def _split_spec(spec: InputSpec) -> tuple[tuple[int, ...], InputOptions]:
        if len(spec) == 2 and isinstance(spec[1], dict):
            return cast(tuple[int, ...], spec[0]), spec[1]
        return cast(tuple[int, ...], spec), {}

    def _make() -> tuple[torch.Tensor, ...]:
        g = torch.Generator().manual_seed(seed)
        tensors: list[torch.Tensor] = []
        for spec in shapes_and_kwargs:
            shape, kw = _split_spec(spec)
            kind = cast(str, kw.get("kind", "randn"))
            dt = cast(torch.dtype, kw.get("dtype", dtype))
            rg = cast(bool, kw.get("requires_grad", requires_grad))
            if kind == "randn":
                t = torch.randn(*shape, generator=g, dtype=dt)
            elif kind == "randint":
                low = cast(int, kw.get("low", 0))
                high = cast(int, kw.get("high", 100))
                t = torch.randint(low, high, shape, generator=g, dtype=dt)
            elif kind == "arange":
                t = torch.arange(int(np.prod(shape)), dtype=dt).reshape(shape)
            else:
                raise ValueError(f"Unknown input kind: {kind!r}")
            if rg and dt.is_floating_point:
                t.requires_grad_(True)
            tensors.append(t)
        return tuple(tensors)

    return _make


def module_factory(build: Callable[[], nn.Module], seed: int = 0) -> Callable[[], nn.Module]:
    """Build a module factory that yields deterministic weights each call."""

    def _make() -> nn.Module:
        torch.manual_seed(seed)
        m = build()
        m.eval()
        return m

    return _make
