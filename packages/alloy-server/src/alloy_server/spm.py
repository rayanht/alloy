"""Python port of llama.cpp's SPM bigram-merge tokenizer.

GGUF stores tokenizer scores as merge priorities (-rank), not log-
probabilities, so the standard SentencePiece Unigram-Viterbi algorithm
falls back to per-character splits and produces wrong segmentations.
llama.cpp instead runs a priority queue keyed by score, repeatedly
merging the highest-scoring adjacent bigram until no more vocab matches
exist. This module is the encoder both `gguf.py` (chat LMs) and
`nomic_bert.py` (encoder embeddings) call.

Algorithm (see ``src/llama-vocab.cpp::llm_tokenizer_spm_session::tokenize``):
  1. Split the (already space-marker-normalised) text into per-character
     symbols, linked via a doubly-linked list.
  2. Push every adjacent (left, right) pair whose concatenation is a vocab
     token onto a max-heap keyed by score (highest first; tie-break on
     left index, lower wins — matches llama.cpp's comparator).
  3. Pop the top bigram, merge into ``left`` and unlink ``right``; push
     the two newly adjacent pairs.
  4. Repeat until the queue is empty.
  5. Walk the surviving symbols left→right, emitting the matched vocab id
     or, on miss, byte-fallback ``<0xNN>`` ids (one per UTF-8 byte).
"""

from __future__ import annotations

import heapq

SPM_SPACE_MARKER = "▁"  # ▁ — SentencePiece word-start marker


def encode_spm(
    text: str,
    *,
    token_to_id: dict[str, int],
    scores: list[float],
) -> list[int]:
    """Encode a single string into token ids using llama.cpp's SPM
    bigram-merge algorithm. The input must already have spaces converted
    to ``▁`` (callers normalise beforehand). Byte-fallback ``<0xNN>`` ids
    are emitted for surviving symbols outside the vocab.
    """
    if not text:
        return []

    sym_text: list[str] = list(text)
    sym_alive: list[bool] = [True] * len(sym_text)
    sym_prev: list[int] = [i - 1 for i in range(len(sym_text))]
    sym_next: list[int] = [i + 1 for i in range(len(sym_text))]
    if sym_next:
        sym_next[-1] = -1

    heap: list[tuple[float, int, int, int, int]] = []
    counter = 0

    def push_bigram(left: int, right: int) -> None:
        nonlocal counter
        if left < 0 or right < 0 or not sym_alive[left] or not sym_alive[right]:
            return
        merged = sym_text[left] + sym_text[right]
        tid = token_to_id.get(merged)
        if tid is None:
            return
        counter += 1
        heap.append((-scores[tid], left, counter, right, len(merged)))
        heapq.heapify(heap)

    for i in range(1, len(sym_text)):
        push_bigram(i - 1, i)

    while heap:
        _neg_score, left, _ctr, right, size = heapq.heappop(heap)
        if not sym_alive[left] or not sym_alive[right]:
            continue
        if len(sym_text[left]) + len(sym_text[right]) != size:
            continue
        sym_text[left] = sym_text[left] + sym_text[right]
        sym_alive[right] = False
        new_next = sym_next[right]
        sym_next[left] = new_next
        if new_next != -1:
            sym_prev[new_next] = left
        push_bigram(sym_prev[left], left)
        push_bigram(left, sym_next[left])

    ids: list[int] = []
    i = 0
    while i != -1 and i < len(sym_text):
        if sym_alive[i]:
            piece = sym_text[i]
            tid = token_to_id.get(piece)
            if tid is not None:
                ids.append(tid)
            else:
                for byte in piece.encode("utf-8"):
                    bt = token_to_id.get(f"<0x{byte:02X}>")
                    if bt is not None:
                        ids.append(bt)
        i = sym_next[i]
    return ids
