"""Quick integration smoke test — compile a real model through torch.compile.

Catches integration regressions that unit tests miss: FX decomposition changes,
handler gaps, fusion miscompiles, compiled plan pointer rebinding.

Run after any change to the emitter, fusion engine, or torch backend:
    uv run python -m pytest tests/integration/test_smoke.py -x
"""

import numpy as np
import pytest
import torch

try:
    import alloy_torch  # noqa: F401
    import transformers
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="alloy-torch or transformers not installed")


def _compile_run_check(model, inputs, rtol=1e-2, atol=1e-2):
    """Compile with alloy, run 3 times (compile + 2 plan runs), check correctness."""
    torch.set_grad_enabled(False)
    torch._dynamo.reset()

    expected = model(**inputs)
    expected_logits = expected.logits if hasattr(expected, "logits") else expected

    compiled = torch.compile(model, backend="alloy")
    # Run 0: compilation + handler path
    compiled(**inputs)
    # Run 1: compiled plan
    compiled(**inputs)
    # Run 2: compiled plan (the one we check)
    result = compiled(**inputs)
    result_logits = result.logits if hasattr(result, "logits") else result

    diff = (result_logits - expected_logits).abs().max().item()
    assert diff < 1.0, f"Max diff {diff:.4f} exceeds threshold"
    np.testing.assert_allclose(
        result_logits.detach().numpy(), expected_logits.detach().numpy(),
        rtol=rtol, atol=atol,
    )


def test_llama_1l():
    """1-layer Llama — exercises GEMM, RMSNorm, RoPE, SiLU, GQA attention."""
    cfg = transformers.LlamaConfig(
        hidden_size=256, intermediate_size=512, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=1000,
        max_position_embeddings=64,
    )
    model = transformers.LlamaForCausalLM(cfg).eval()
    ids = torch.randint(0, 1000, (1, 8))
    _compile_run_check(model, {"input_ids": ids})


def test_gpt2_1l():
    """1-layer GPT-2 — exercises GEMM, LayerNorm, SDPA, GELU."""
    cfg = transformers.GPT2Config(
        n_embd=256, n_head=4, n_layer=1, vocab_size=1000,
        n_positions=64,
    )
    model = transformers.GPT2LMHeadModel(cfg).eval()
    ids = torch.randint(0, 1000, (1, 8))
    _compile_run_check(model, {"input_ids": ids})
