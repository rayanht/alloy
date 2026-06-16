"""Conversation tracking + warm-prefix reconstruction.

`ConversationStore` owns the recent (messages, input_ids, decoded) states a
served model keeps for warm-prefix reuse: the chat-template-aware reconstruction
that splices the prior turn's emitted token ids back into the next render (so the
generator's token-level LCP matches through the assistant boundary), the
foreign-side-call preserve heuristic, and the continuation test. The token-level
KV reuse downstream is the generator's PrefixCache; this is the message-level
owner that feeds it.
"""

from __future__ import annotations

import collections

import torch
import transformers

from alloy import get_logger
from alloy_server.schema import ChatMessage
from alloy_server.reasoning import ReasoningProtocol, split_reasoning

logger = get_logger("alloy_server.session")

ASST_SENTINEL = "<<<__ALLOY_ASST_PLACEHOLDER__>>>"


def chat_template_extras(
    tokenizer: transformers.PreTrainedTokenizerBase,
    enable_thinking: bool | None = None,
) -> dict[str, bool]:
    """Return extra kwargs for `apply_chat_template`. Reasoning models thread an
    `enable_thinking` flag through their template: when true the `<think>` block is
    left open (`<think>\\n`) so the model produces chain-of-thought; when false it
    is closed immediately (`<think>\\n\\n</think>\\n\\n`) to skip reasoning.

    `enable_thinking` here is the per-request control: None keeps the default
    (thinking ON, matching ollama), True/False is an explicit client override
    (OpenAI reasoning_effort / chat_template_kwargs, Ollama `think`, Anthropic
    `thinking`). For templates that don't support the flag this is a no-op.
    """
    template = tokenizer.chat_template if hasattr(tokenizer, "chat_template") else None
    if isinstance(template, str) and "enable_thinking" in template:
        return {"enable_thinking": True if enable_thinking is None else enable_thinking}
    return {}


def tokenize_text(tokenizer: transformers.PreTrainedTokenizerBase, text: str) -> list[int]:
    batch = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    ids = batch["input_ids"]
    if not isinstance(ids, torch.Tensor):
        raise TypeError("tokenizer returned non-tensor input_ids")
    return ids[0].tolist()


def saved_decoded_forms(
    tokenizer: transformers.PreTrainedTokenizerBase,
    saved_decoded: list[int],
    eos_token_ids: frozenset[int],
) -> tuple[str, str, str]:
    """The three text forms the server can emit for `saved_decoded`:

      - clean: skip_special_tokens=True (strips `<think>`, EOS, everything)
      - full:  skip_special_tokens=False (keeps everything incl. EOS)
      - kept:  skip_special_tokens=False with EOS-only filter — matches
        what `decode()` in `create_native_served_model` actually emits, so
        the assistant message a client echoes back in turn 2 lines up
        with one of these forms even when reasoning markers like
        `</think>` (qwen3.5 single token, id 248069) or `<think>` (qwen3
        id 151667) are part of the stream.
    """
    clean = tokenizer.decode(saved_decoded, skip_special_tokens=True)
    full = tokenizer.decode(saved_decoded, skip_special_tokens=False)
    kept = tokenizer.decode(
        [t for t in saved_decoded if t not in eos_token_ids],
        skip_special_tokens=False,
    )
    return clean, full, kept


def reconstruct_warm_input_ids(
    tokenizer: transformers.PreTrainedTokenizerBase,
    messages: tuple[ChatMessage, ...],
    saved_messages: tuple[ChatMessage, ...],
    saved_input_ids: list[int],
    saved_decoded: list[int],
    eos_token_ids: frozenset[int],
    enable_thinking: bool | None = None,
    reasoning: ReasoningProtocol | None = None,
) -> torch.Tensor | None:
    """Reconstruct input_ids for a clean one-turn continuation, splicing
    the actual model-emitted decoded token ids back into the chat-template
    rendered conversation. The returned tensor's first
    `len(saved_input_ids) + len(saved_decoded)` tokens equal exactly the
    tokens the cache covers, so the generator's existing token-level LCP
    warm prefill matches all the way through and the suffix to prefill is
    only the new chat wrappers + new user message + generation prompt.

    Returns None when:
      - the request isn't a clean extension by exactly one assistant turn
        and one new user message,
      - the chat template's re-render doesn't produce stable token
        boundaries (rare),
      - or the model's emission contains special tokens that don't BPE
        round-trip through tokenize(decode(x)). In that case the assistant
        text re-tokenises to a different sequence than the cache holds, so
        the chat-template's subsequent renders of the assistant message
        would diverge from cache content and produce wrong attention.

    Caller falls back to cold `encode_messages(messages)` when this returns
    None, plus calls `generator.reset_prefix_state()` so the generator's
    own token-level LCP doesn't match an unrelated previous request by
    coincidence (chat-template wrappers tokenise to identical leading ids).
    """
    def bail(reason: str, **extra: object) -> None:
        # A bailed reconstruction costs a FULL cold re-prefill of the whole
        # prompt next; the reason is the first thing to look at whenever
        # turns are mysteriously slow (e.g. a client mutating its system
        # prompt every request — Claude Code's billing-header hash).
        logger.info("warm_reconstruct_bail", reason=reason, **extra)

    if not saved_decoded or not saved_input_ids:
        bail("no_saved_generation")
        return None
    if eos_token_ids and saved_decoded[-1] not in eos_token_ids:
        bail("saved_generation_not_eos_terminated")
        return None
    if len(messages) != len(saved_messages) + 2:
        # Diagnostic: name the first content divergence among the shared
        # leading messages — a client mutating an early block (billing hash,
        # git snapshot, timestamps) silently caps every later request's
        # token-LCP at that point, and the excerpt is how it gets found.
        for i, (prior, incoming) in enumerate(zip(saved_messages, messages, strict=False)):
            if prior.role != incoming.role or prior.content != incoming.content:
                j = next(
                    (k for k, (a, b) in enumerate(zip(prior.content, incoming.content, strict=False)) if a != b),
                    min(len(prior.content), len(incoming.content)),
                )
                bail(
                    "prefix_message_diff", index=i, role=incoming.role, char=j,
                    prior_excerpt=prior.content[max(0, j - 60):j + 120],
                    incoming_excerpt=incoming.content[max(0, j - 60):j + 120],
                )
                break
        bail(
            "not_single_turn_extension",
            n_messages=len(messages), n_saved=len(saved_messages),
        )
        return None
    for i, (prior, incoming) in enumerate(zip(saved_messages, messages, strict=False)):
        if prior.role != incoming.role or prior.content != incoming.content:
            bail(
                "prior_message_changed", index=i, role=incoming.role,
                prior_len=len(prior.content), incoming_len=len(incoming.content),
            )
            return None
    asst_msg = messages[len(saved_messages)]
    new_user_msg = messages[len(saved_messages) + 1]
    # The new turn is a user reply OR a tool result fed back after a tool call.
    if asst_msg.role != "assistant" or new_user_msg.role not in ("user", "tool"):
        bail("not_assistant_plus_reply", roles=(asst_msg.role, new_user_msg.role))
        return None
    if not asst_msg.tool_calls:
        # Normal text turn: the assistant content must match what the cache holds.
        # (For a tool-call turn, saved_decoded IS the model's emission — the
        # `<tool_call>…</tool_call>` tokens — which we splice back verbatim below,
        # so there's no text content to match.)
        asst_text_clean, asst_text_full, asst_text_kept = saved_decoded_forms(
            tokenizer, saved_decoded, eos_token_ids,
        )
        candidates = {asst_text_clean, asst_text_full, asst_text_kept}
        if reasoning is not None:
            # Reasoning models: the server emits the chain-of-thought
            # separately (reasoning_content / thinking / thinking blocks),
            # so the assistant content a client echoes back is the
            # POST-reasoning text only. The clean form can't be split (the
            # close marker is a special token it strips); full/kept keep it.
            candidates.add(split_reasoning(asst_text_full, reasoning)[1])
            candidates.add(split_reasoning(asst_text_kept, reasoning)[1])
        if asst_msg.content not in candidates:
            bail(
                "assistant_content_mismatch",
                incoming_len=len(asst_msg.content), clean_len=len(asst_text_clean),
            )
            return None
    # Note: we deliberately DON'T require BPE round-trip on saved_decoded.
    # The cache K/V at the assistant positions was written for the actual
    # model-emitted ids (which may include special tokens like Qwen3's
    # `<think>` 151667). We feed those exact ids back via the splice; the
    # generator's prefill only needs the chat-template wrappers around
    # them — pre_tokens (= saved input_ids) and post_tokens — to be stable.
    msgs = [{"role": m.role, "content": m.content} for m in saved_messages]
    msgs.append({"role": "assistant", "content": ASST_SENTINEL})
    msgs.append({"role": new_user_msg.role, "content": new_user_msg.content})
    try:
        rendered = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            **chat_template_extras(tokenizer, enable_thinking),
        )
    except Exception:
        # Non-renderable (e.g. role alternation): bail on the warm path and let
        # the cold encode raise the canonical 400.
        return None
    if not isinstance(rendered, str):
        return None
    sentinel_pos = rendered.find(ASST_SENTINEL)
    if sentinel_pos < 0:
        return None
    post_text = rendered[sentinel_pos + len(ASST_SENTINEL):]
    post_tokens = tokenize_text(tokenizer, post_text)
    # The pre-sentinel rendered text is intentionally NOT re-tokenised:
    # saved_input_ids IS what's in cache (the input_ids actually
    # prefilled, which from turn 3 onward contain the model's emitted
    # ids at prior assistant positions rather than the chat-template
    # text-tokenisation of those positions). The message-level
    # continuation check above already proves the incoming conversation
    # extends the saved one by exactly one assistant turn plus one new
    # user turn, so the cache's content is internally consistent with
    # the spliced input_ids built below.
    full = list(saved_input_ids) + list(saved_decoded) + post_tokens
    return torch.tensor([full], dtype=torch.long)


class ConversationStore:
    """Recent (messages, input_ids, decoded) states for warm-prefix reuse.

    Multiple slots because conversations BRANCH: Claude Code follows each real
    turn with a +2-shaped side call (same history + a synthetic user message). A
    single slot would let the side call claim it and send the REAL next turn
    cold, so warm reconstruction tries each saved state newest-first.
    """

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizerBase | None,
        eos_token_ids: frozenset[int],
        reasoning: ReasoningProtocol | None,
        maxlen: int = 4,
    ) -> None:
        self.tokenizer = tokenizer
        self.eos_token_ids = eos_token_ids
        self.reasoning = reasoning
        self.states: collections.deque = collections.deque(maxlen=maxlen)

    def newest(self) -> "tuple[tuple[ChatMessage, ...], list[int], list[int]] | None":
        return self.states[-1] if self.states else None

    def warm_input_ids(
        self, messages: tuple[ChatMessage, ...], enable_thinking: bool | None = None,
    ) -> torch.Tensor | None:
        """Spliced input_ids for a warm one-turn continuation against the newest
        matching saved state, or None for a cold prefill. Tries states newest-first."""
        if self.tokenizer is None:
            return None
        for saved_messages, saved_input_ids, saved_decoded in reversed(self.states):
            ids = reconstruct_warm_input_ids(
                self.tokenizer, messages, saved_messages, saved_input_ids,
                saved_decoded, self.eos_token_ids, enable_thinking, self.reasoning,
            )
            if ids is not None:
                return ids
        return None

    def preserve_foreign(self, input_ids: torch.Tensor) -> bool:
        """True when a request that ISN'T a warm continuation should run on the
        generator's scratch cache instead of evicting the saved warm prefix.
        Clients interleave small side calls with the main conversation (Claude
        Code: topic detection, bash-prefix checks); each eviction costs the next
        main turn a full cold re-prefill of a 30k+-token prompt. Heuristic:
        foreign and at most half the saved sequence — a genuinely new big
        conversation still claims the primary cache, while a small conversation
        that keeps growing eventually outgrows the stale state and claims it."""
        state = self.newest()
        if state is None:
            return False
        saved_total = len(state[1]) + len(state[2])
        return int(input_ids.shape[1]) * 2 <= saved_total

    def is_continuation(self, messages: tuple[ChatMessage, ...]) -> bool:
        """True if `messages` extends the previous turn's saved messages (same
        leading messages, this turn just appends). Gates whether to keep the
        generator's warm prefix: a continuation lets the token LCP reuse the
        cached KV (including a vision turn's image features); an unrelated request
        resets so no stale prefix matches by coincidence."""
        state = self.newest()
        if state is None:
            return False
        prev = state[0]
        if len(prev) > len(messages):
            return False
        return all(
            p.role == c.role and p.content == c.content and p.images == c.images
            for p, c in zip(prev, messages)
        )

    def save(
        self, messages: tuple[ChatMessage, ...], input_ids: list[int], decoded: list[int],
    ) -> None:
        self.states.append((messages, input_ids, decoded))
