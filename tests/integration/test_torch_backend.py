"""Integration tests for torch.compile backend — end-to-end model compilation."""

import numpy as np
import pytest
import torch
import torch.nn as nn
import alloy_torch  # noqa: F401
import alloy_torch.backend
from alloy._compiler.dtypes import float32, int64
from alloy._dispatch.buf_utils import _alloc_aligned
from alloy._runtime.metal import default_dispatcher


def _compile_and_compare(model, x, rtol=1e-3, atol=1e-4):
    """Compile model with alloy backend, compare against eager."""
    torch.set_grad_enabled(False)
    torch._dynamo.reset()
    expected = model(x)
    compiled = torch.compile(model, backend="alloy")
    result = compiled(x)
    np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=rtol, atol=atol)
    return compiled


# ---------------------------------------------------------------------------
# Simple models
# ---------------------------------------------------------------------------

class TestSimpleModels:
    def test_linear(self):
        _compile_and_compare(nn.Linear(64, 32).eval(), torch.randn(4, 64))

    def test_two_linears_with_relu(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(64, 128)
                self.fc2 = nn.Linear(128, 32)
            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x)))
        _compile_and_compare(M().eval(), torch.randn(4, 64))

    def test_layer_norm(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(64, 64)
                self.ln = nn.LayerNorm(64)
            def forward(self, x):
                return self.ln(self.fc(x))
        _compile_and_compare(M().eval(), torch.randn(4, 64))

    def test_softmax(self):
        class M(nn.Module):
            def forward(self, x):
                return torch.softmax(x, dim=-1)
        _compile_and_compare(M().eval(), torch.randn(4, 64))

    def test_argmax_last_dim(self):
        class M(nn.Module):
            def forward(self, x):
                return x[:, -1:, :].argmax(dim=-1)

        torch.set_grad_enabled(False)
        torch._dynamo.reset()
        x = torch.randn(2, 3, 257)
        x[0, -1, 19] = 100.0
        x[1, -1, 211] = 100.0
        expected = M().eval()(x)
        compiled = torch.compile(M().eval(), backend="alloy")
        result = compiled(x)
        assert torch.equal(result, expected)

    def test_multi_layer_mlp(self):
        """MLP with multiple layers and activations."""
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(64, 256)
                self.fc2 = nn.Linear(256, 64)
                self.fc3 = nn.Linear(64, 32)
            def forward(self, x):
                return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))
        _compile_and_compare(M().eval(), torch.randn(4, 64))

    def test_rms_norm_standalone(self):
        """RMS norm without weight should still produce correct results through decomposed handlers."""
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(64, 64, bias=False)
            def forward(self, x):
                h = self.fc(x)
                rms = torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
                return h * rms
        _compile_and_compare(M().eval(), torch.randn(4, 64), rtol=1e-2, atol=1e-3)


# ---------------------------------------------------------------------------
# FX rewrites
# ---------------------------------------------------------------------------

class TestFXRewrites:
    def test_gelu_tanh(self):
        class M(nn.Module):
            def forward(self, x):
                return torch.nn.functional.gelu(x, approximate="tanh")
        _compile_and_compare(M().eval(), torch.randn(4, 64), rtol=1e-3, atol=1e-3)

    def test_gelu_exact(self):
        class M(nn.Module):
            def forward(self, x):
                return torch.nn.functional.gelu(x)
        _compile_and_compare(M().eval(), torch.randn(4, 64), rtol=1e-3, atol=1e-3)

    def test_silu(self):
        class M(nn.Module):
            def forward(self, x):
                return torch.nn.functional.silu(x)
        _compile_and_compare(M().eval(), torch.randn(4, 64))


# ---------------------------------------------------------------------------
# SDPA (scaled dot product attention)
# ---------------------------------------------------------------------------

class TestSDPA:
    def test_sdpa_basic(self):
        class M(nn.Module):
            def forward(self, q, k, v):
                return torch.nn.functional.scaled_dot_product_attention(q, k, v)

        torch.set_grad_enabled(False)
        torch._dynamo.reset()
        B, H, N, D = 1, 1, 16, 16
        q = torch.randn(B, H, N, D)
        k = torch.randn(B, H, N, D)
        v = torch.randn(B, H, N, D)
        model = M().eval()
        expected = model(q, k, v)
        compiled = torch.compile(model, backend="alloy")
        result = compiled(q, k, v)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-3, atol=1e-3)

    def test_sdpa_causal(self):
        class M(nn.Module):
            def forward(self, q, k, v):
                return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

        torch.set_grad_enabled(False)
        torch._dynamo.reset()
        B, H, N, D = 1, 1, 16, 16
        q = torch.randn(B, H, N, D)
        k = torch.randn(B, H, N, D)
        v = torch.randn(B, H, N, D)
        model = M().eval()
        expected = model(q, k, v)
        compiled = torch.compile(model, backend="alloy")
        result = compiled(q, k, v)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# Dispatch count
# ---------------------------------------------------------------------------

class TestDispatchCount:
    def test_compiled_plan_fewer_dispatches(self):
        """Run 0 (handler path) compiles; run 2+ uses compiled plan."""
        torch.set_grad_enabled(False)
        torch._dynamo.reset()

        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 32)).eval()
        x = torch.randn(4, 64)
        compiled = torch.compile(model, backend="alloy")

        # Run 0: compilation
        compiled(x)
        # Run 1: handler path
        d = default_dispatcher()
        before = d.dispatch_count
        compiled(x)
        handler_dispatches = d.dispatch_count - before

        # Run 2: compiled plan
        before = d.dispatch_count
        compiled(x)
        plan_dispatches = d.dispatch_count - before

        assert plan_dispatches <= handler_dispatches


class TestCompiledPlanInputs:
    def test_plan_metadata_records_view_offsets_and_outputs(self) -> None:
        torch.set_grad_enabled(False)
        torch._dynamo.reset()

        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                return x[:4] * 2.0, x[4:8] * 3.0

        model = M().eval()
        compiled = torch.compile(model, backend="alloy")
        x = torch.arange(64, dtype=torch.float32).reshape(8, 8)

        with alloy_torch.backend.capture_plan() as slot:
            compiled(x)
            left, right = compiled(x)
        expected_left, expected_right = model(x)

        np.testing.assert_allclose(left.numpy(), expected_left.numpy(), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(right.numpy(), expected_right.numpy(), rtol=1e-5, atol=1e-5)

        plan = slot.plan
        assert isinstance(plan, alloy_torch.backend.CompiledPlan)

        input_slots = [
            slot
            for slot in plan.slots
            if isinstance(slot, alloy_torch.backend.InputSlot)
        ]
        assert [(slot.arg_idx, slot.view_offset) for slot in input_slots] == [(0, 0)]

        assert tuple(dispatch.debug_name for dispatch in plan.dispatches) == (
            "k_mul_scalar",
            "k_mul_scalar",
        )
        assert tuple(tuple(dispatch.buf_offsets) for dispatch in plan.dispatches) == (
            (0, 0),
            (128, 0),
        )
        assert tuple(dispatch.buf_identity_offsets for dispatch in plan.dispatches) == (
            (0, 0),
            (128, 0),
        )
        assert tuple(tuple(group) for group in plan.dep_groups) == ((0, 1),)

        assert len(plan.output_mapping) == 2
        output_entries: list[alloy_torch.backend.OutputSlot] = []
        for entry in plan.output_mapping:
            assert isinstance(entry, alloy_torch.backend.OutputSlot)
            output_entries.append(entry)
        assert [(entry.shape, entry.byte_offset, entry.strides_bytes) for entry in output_entries] == [
            ((4, 8), 0, (32, 4)),
            ((4, 8), 0, (32, 4)),
        ]

    def test_recomputes_input_updates_when_view_offset_changes(self):
        torch.set_grad_enabled(False)
        torch._dynamo.reset()

        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x * 2.0

        model = M().eval()
        compiled = torch.compile(model, backend="alloy")
        base = torch.arange(128, dtype=torch.float32).reshape(16, 8)
        first = base[:4]

        compiled(first)
        compiled(first)
        second = torch.as_strided(first, (4, 8), (8, 1), storage_offset=32)
        result = compiled(second)

        np.testing.assert_allclose(result.numpy(), model(second).numpy(), rtol=1e-5, atol=1e-5)

    def test_cse_distinguishes_same_storage_views_with_different_offsets(self) -> None:
        torch.set_grad_enabled(False)
        torch._dynamo.reset()

        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                return x[:4] * 2.0, x[4:8] * 2.0

        model = M().eval()
        compiled = torch.compile(model, backend="alloy")
        x = torch.arange(64, dtype=torch.float32).reshape(8, 8)

        compiled(x)
        left, right = compiled(x)
        expected_left, expected_right = model(x)

        np.testing.assert_allclose(left.numpy(), expected_left.numpy(), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(right.numpy(), expected_right.numpy(), rtol=1e-5, atol=1e-5)

    def test_dispatch_plan_rejects_invalid_input_handles(self) -> None:
        torch.set_grad_enabled(False)
        torch._dynamo.reset()

        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x * 2.0

        compiled = torch.compile(M().eval(), backend="alloy")
        x = torch.arange(64, dtype=torch.float32)
        with alloy_torch.backend.capture_plan() as slot:
            compiled(x)
            compiled(x)

        plan = slot.plan
        assert isinstance(plan, alloy_torch.backend.CompiledPlan)
        input_slot_idx = next(
            i
            for i, slot in enumerate(plan.slots)
            if isinstance(slot, alloy_torch.backend.InputSlot)
        )

        with pytest.raises(RuntimeError, match="Invalid input buffer handle"):
            alloy_torch.backend._metal_ext.dispatch_plan(
                plan.plan_handle,
                [(input_slot_idx, -123456789, 0)],
            )

        released_handle = alloy_torch.backend._metal_ext.buf_alloc(64)
        alloy_torch.backend._metal_ext.buf_release(released_handle)
        with pytest.raises(RuntimeError, match="Released input buffer handle"):
            alloy_torch.backend._metal_ext.dispatch_plan(
                plan.plan_handle,
                [(input_slot_idx, released_handle, 0)],
            )

    def test_unmapped_non_scalar_output_raises(self) -> None:
        orphan = _alloc_aligned((4,), float32)

        with pytest.raises(RuntimeError, match="unmapped non-scalar AlloyBuffer output"):
            alloy_torch.backend._classify_plan_output(orphan, {}, {})

    def test_mutation_remap_preserves_input_when_plan_reads_it_later(self) -> None:
        slots = [
            alloy_torch.backend.InputSlot(
                arg_idx=0,
                nbytes=8,
                root_ptr=0x1000,
                view_offset=0,
            ),
            alloy_torch.backend.WeightSlot(nbytes=8, root_ptr=0x2000),
            alloy_torch.backend.IntermediateSlot(nbytes=8, root_ptr=0x3000),
            alloy_torch.backend.IntermediateSlot(nbytes=32, root_ptr=0x4000),
        ]
        dispatches = [
            alloy_torch.backend.PlanDispatch(
                pso_handle=1,
                debug_name="increment",
                buf_slot_indices=[0, 1, 2],
                buf_offsets=[0, 0, 0],
                buf_identity_offsets=(0, 0, 0),
                grid=(1, 1, 1),
                tg=(1, 1, 1),
                write_slot_indices={2},
            ),
            alloy_torch.backend.PlanDispatch(
                pso_handle=2,
                debug_name="uses_original",
                buf_slot_indices=[3, 0],
                buf_offsets=[0, 0],
                buf_identity_offsets=(0, 0),
                grid=(1, 1, 1),
                tg=(1, 1, 1),
                write_slot_indices={3},
            ),
        ]
        output_mapping: list[alloy_torch.backend.OutputEntry] = [
            alloy_torch.backend.OutputSlot(
                slot_idx=2,
                shape=(1,),
                dtype=int64,
                byte_offset=0,
                strides_bytes=(8,),
            )
        ]

        mutation_input_slots = alloy_torch.backend._apply_mutation_remap(
            output_mapping,
            dispatches,
            slots,
            {0x1000: alloy_torch.backend.InputPtrInfo(arg_idx=0, view_offset=0)},
            {0: 0},
        )

        assert mutation_input_slots == {0: 0}
        assert output_mapping == [
            alloy_torch.backend.OutputSlot(
                slot_idx=2,
                shape=(1,),
                dtype=int64,
                byte_offset=0,
                strides_bytes=(8,),
            )
        ]
        assert dispatches[0].buf_slot_indices == [0, 1, 2]
        assert dispatches[0].write_slot_indices == {2}


# ---------------------------------------------------------------------------
# Transformer block (exercises multiple rewrites together)
# ---------------------------------------------------------------------------

class TestCompositeModels:
    def test_linear_layernorm_linear(self):
        """Linear → LayerNorm → Linear exercises GEMM + norm fusion."""
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(64, 64)
                self.ln = nn.LayerNorm(64)
                self.fc2 = nn.Linear(64, 32)
            def forward(self, x):
                return self.fc2(self.ln(self.fc1(x)))
        _compile_and_compare(M().eval(), torch.randn(4, 64))

    def test_residual_connection(self):
        """Residual add pattern should compile correctly."""
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(64, 64)
            def forward(self, x):
                return x + torch.relu(self.fc(x))
        _compile_and_compare(M().eval(), torch.randn(4, 64))
