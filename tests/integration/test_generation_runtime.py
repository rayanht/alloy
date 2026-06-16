from __future__ import annotations

import torch
import torch._dynamo
import pytest
from transformers import GPT2Config, GPT2LMHeadModel

from alloy_server.cache import AlloyStaticCache
from alloy_server.generation.generator import AlloyGenerator
from alloy_server.models.attention import alloy_cache_attention


def tiny_model() -> GPT2LMHeadModel:
    config = GPT2Config(
        n_layer=1,
        n_head=2,
        n_embd=32,
        # >= the default chunked-prefill chunk size (128): a short prompt is
        # padded up to one chunk, so the model's position embeddings must span it.
        n_positions=128,
        vocab_size=128,
        attn_implementation="eager",
    )
    model = GPT2LMHeadModel(config)
    return model.eval()


def test_generator_returns_requested_tokens() -> None:
    torch._dynamo.reset()
    torch.manual_seed(1)
    model = tiny_model()
    reference_model = tiny_model()
    reference_model.load_state_dict(model.state_dict())
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    ref_gen = AlloyGenerator.from_model(reference_model)
    ref_gen.eager_compile_all()
    reference = list(ref_gen.stream_chunks_fast(input_ids, max_new_tokens=5))

    torch._dynamo.reset()
    generator = AlloyGenerator.from_model(model)
    generator.eager_compile_all()
    output = generator.generate(input_ids, max_new_tokens=5)

    assert tuple(output.shape) == (1, 9)
    assert torch.equal(output[:, :4], input_ids)
    assert output[:, 4:].tolist()[0] == reference


def test_generator_streams_token_ids() -> None:
    generator = AlloyGenerator.from_model(tiny_model())
    generator.eager_compile_all()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    tokens = list(generator.stream_chunks_fast(input_ids, max_new_tokens=3))

    assert len(tokens) == 3
    assert all(isinstance(token, int) for token in tokens)


def test_stream_chunks_fast_matches_generate() -> None:
    """The streaming decode loop and `generate` share the same per-step
    Python loop — their outputs must match token for token."""
    torch._dynamo.reset()
    torch.manual_seed(0)
    model = tiny_model()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    reference_model = tiny_model()
    reference_model.load_state_dict(model.state_dict())
    ref_gen = AlloyGenerator.from_model(reference_model)
    ref_gen.eager_compile_all()
    reference = ref_gen.generate(input_ids, max_new_tokens=5)[0, 4:].tolist()

    torch._dynamo.reset()
    generator = AlloyGenerator.from_model(model)
    generator.eager_compile_all()
    streamed = list(generator.stream_chunks_fast(input_ids, max_new_tokens=5))

    assert streamed == reference


def test_static_cache_initializes_kv_in_target_dtype() -> None:
    cache = AlloyStaticCache(tiny_model().config, max_cache_len=8, cache_dtype=torch.float16)
    layer = cache.layers[0]
    key_states = torch.randn((1, 2, 3, 16), dtype=torch.float32)
    value_states = torch.randn((1, 2, 3, 16), dtype=torch.float32)

    keys, values = layer.update(key_states, value_states)

    assert layer.keys is not None and layer.keys.dtype == torch.float16
    assert layer.values is not None and layer.values.dtype == torch.float16
    assert hasattr(layer.keys.untyped_storage(), "_alloy_keepalive")
    assert hasattr(layer.values.untyped_storage(), "_alloy_keepalive")
    assert keys.dtype == torch.float16
    assert values.dtype == torch.float16


def test_alloy_cache_attention_uses_unified_op_for_long_cold_fp16_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Full-attention (sliding == 0) layers route prefill, decode and verify
    # through the single `attention_cache` op; the cold/warm runtime-position
    # ops only survive for sliding-window (gemma3) layers.
    class FakeLayer:
        def __init__(self) -> None:
            self.keys = torch.empty((1, 2, 16, 16), dtype=torch.float16)
            self.values = torch.empty((1, 2, 16, 16), dtype=torch.float16)
            self.is_initialized = True
            self._alloy_cache_dtype = torch.float16

    called: list[str] = []

    def fake_cache(*args, **kwargs):
        called.append("cache")
        return args[0]

    monkeypatch.setattr(torch.ops.alloy, "attention_cache", fake_cache, raising=False)

    q = torch.empty((1, 2, 9, 16), dtype=torch.float32)
    k = torch.empty((1, 2, 9, 16), dtype=torch.float32)
    v = torch.empty((1, 2, 9, 16), dtype=torch.float32)
    cache_position = torch.tensor([0], dtype=torch.long)

    out = alloy_cache_attention(q, k, v, FakeLayer(), cache_position, 0.25)

    assert out is q
    assert called == ["cache"]
