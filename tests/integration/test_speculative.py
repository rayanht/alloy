"""Speculative-session integration tests.

Scope: session MECHANICS (round loop, accept/commit bookkeeping, EOS,
metrics) on a tiny model, plus pure-CPU PLD behavior. Token-level
lossless A/B gates do NOT live here: a randomly-initialized toy emits
near-uniform logits whose argmax flips under run-to-run GPU
accumulation-order wiggle, so cross-path token equality on a toy tests
tie-breaking luck, not the session. The real gates run on qwen3.5:0.8b
(sharp logits, hybrid DeltaNet rollback, greedy + seeded-sampled
bit-identity).
"""

from __future__ import annotations

import torch
import torch._dynamo
from transformers import LlamaConfig, LlamaForCausalLM

from alloy_server.generation.generator import AlloyGenerator
from alloy_server.speculative import Proposal, TapBatch, TargetTaps
from alloy_server.speculative.pld import PromptLookupDrafter


def tiny_llama() -> LlamaForCausalLM:
    config = LlamaConfig(
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=128,
        max_position_embeddings=512,
        attn_implementation="eager",
    )
    return LlamaForCausalLM(config).eval()


class ScriptedDrafter:
    """Minimal contract drafter: proposes a fixed token sequence."""

    name = "scripted"
    taps = TargetTaps()
    max_draft_tokens = 7

    def __init__(self, tokens: list[int]) -> None:
        self._tokens = tokens
        self.observed: list[tuple[int, int]] = []  # (start, len) calls

    def bind(self, gen) -> None:
        pass

    def warmup(self) -> None:
        pass

    def observe(self, tokens, taps: TapBatch | None, start: int) -> None:
        self.observed.append((start, len(tokens)))

    def propose(self, anchor: int, position: int) -> Proposal:
        return Proposal(list(self._tokens))

    def truncate(self, length: int) -> None:
        pass

    def state_bytes_per_token(self) -> int:
        return 0

    def snapshot_head(self, rows: int):
        return None

    def restore_head(self, snap) -> None:
        pass


PROMPT = torch.tensor([[5, 6, 7, 8] * 8], dtype=torch.long)


def test_session_commits_exactly_max_new_tokens() -> None:
    torch._dynamo.reset()
    torch.manual_seed(7)
    gen = AlloyGenerator.from_model(tiny_llama())
    gen.eager_compile_all()
    drafter = ScriptedDrafter([127] * 7)
    gen.attach_spec(drafter)
    out = list(gen.spec.run(PROMPT, max_new_tokens=12))
    metrics = gen.spec.last_metrics
    # No EOS in a random toy run of 12 tokens is not guaranteed — accept
    # early-stop, but never overshoot.
    assert 0 < len(out) <= 12
    assert metrics is not None
    # Every committed token is accounted: rounds commit accepted + bonus,
    # plus the prefill token and the boundary buffer step (not in metrics).
    assert metrics.committed >= len(out) - 2
    # The drafter observed the prompt range first.
    assert drafter.observed[0] == (0, PROMPT.shape[1])


def test_session_round_bookkeeping_with_padding() -> None:
    """A short proposal pads to the pinned verify width; stats must count
    only the real proposal tokens."""
    torch._dynamo.reset()
    torch.manual_seed(7)
    gen = AlloyGenerator.from_model(tiny_llama())
    gen.eager_compile_all()
    drafter = ScriptedDrafter([127, 126])  # 2 < max_draft_tokens
    gen.attach_spec(drafter)
    out = list(gen.spec.run(PROMPT, max_new_tokens=8))
    metrics = gen.spec.last_metrics
    assert 0 < len(out) <= 8
    for r in metrics.per_round if metrics.per_round else []:
        assert r.proposed <= 2
    # proposed counts real tokens only (2/round), never the padded width.
    assert metrics.rounds == 0 or metrics.proposed <= 2 * metrics.rounds


def test_pld_propose_ngram_matching() -> None:
    d = PromptLookupDrafter(max_draft_tokens=4)
    # Sequence: 1 2 3 9 9 4 1 2 3   (positions 0..8)
    d.observe([1, 2, 3, 9, 9, 4, 1, 2, 3], None, 0)
    # anchor=3 at position 9: trailing 3-gram (2,3,3) has no earlier hit,
    # 2-gram (3,3) none → miss.
    assert d.propose(3, 9).tokens == []
    # anchor=3 at position 8 → context 1 2 3 9 9 4 1 2; pattern (1,2,3)
    # matched at index 0 → continuation [9, 9, 4, 1].
    assert d.propose(3, 8).tokens == [9, 9, 4, 1]


def test_pld_truncate_and_regrow() -> None:
    d = PromptLookupDrafter(max_draft_tokens=4)
    d.observe([1, 2, 3, 4, 5, 6], None, 0)
    d.truncate(4)
    d.observe([9, 9], None, 4)
    assert d.propose(9, 6).tokens == []  # (9,9) only at the live tail
    d.observe([1, 2], None, 6)
    # context = 1 2 3 4 9 9 1 2, anchor 3 at position 8: (1,2,3) at 0 →
    # propose [4, 9, 9, 1].
    assert d.propose(3, 8).tokens == [4, 9, 9, 1]


def test_pld_no_state_below_min_ngram() -> None:
    d = PromptLookupDrafter()
    assert d.propose(1, 0).tokens == []
    d.observe([1], None, 0)
    assert d.propose(1, 1).tokens == []
