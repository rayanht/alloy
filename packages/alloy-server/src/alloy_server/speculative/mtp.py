"""
Wraps the `Qwen35MTP` module (mtp.py — GGUF-loaded single-layer MTP head,
shared embed/rope/lm_head, optional pruned draft head). The session owns
verify/accept/rollback; this drafter owns the draft half: pinned M∈{1,2} draft
plans against the MTP block's own cache, precomputed rotary/embedding tables,
and the per-round staging (embedding gather + post-norm hidden rows).

Draft-input pairing: the MTP block's row for committed token t at position q
consumes (embed(t), post-norm hidden of position q-1) — the hidden BEFORE t.
Hiddens come from the session's verify taps (`TargetTaps(post_norm=True)`); the
prefill emits no taps in v1, so round 0 drafts against a zero hidden and misses
once (h0=zeros).

The draft block's cache only ever absorbs COMMITTED tokens, so it needs no
rollback: `propose` folds the unabsorbed committed tail [*prev_committed,
anchor] into ONE M-token forward (M ∈ {1,2}), whose last-position argmax is the
proposal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gguf
import numpy as np
import torch

from alloy._compiler.dtypes import float16
from alloy._dispatch.buf_utils import _alloc_aligned
from alloy_torch.tensor_bridge import make_tensor_from_ptr
from alloy_torch.backend import OutputSlot, _execute_plan, capture_plan
from alloy_server.cache import AlloyStaticCache
from alloy_server.models.mtp import MTPDraftStep, load_quantized_mtp

from .contract import Proposal, TapBatch, TargetTaps

if TYPE_CHECKING:
    from alloy_server.generation.generator import AlloyGenerator


class MTPDrafter:
    """Qwen3.5 native MTP head as a contract drafter. `blob_path` is the GGUF
    blob carrying the `mtp.*` tensors (the served model's own blob);
    `draft_topk` installs the pruned shortlist head (lossless — verify stays
    full-vocab)."""

    name = "mtp"
    max_draft_tokens = 1
    taps = TargetTaps(post_norm=True)

    def __init__(self, blob_path, quantize: bool = True, draft_topk: int | None = None) -> None:
        self._blob_path = blob_path
        self._quantize = quantize
        self._draft_topk = draft_topk
        self._gen: "AlloyGenerator | None" = None
        self._mtp = None
        self._pins: dict | None = None
        # Committed-token mirror + per-position post-norm hidden staging.
        self._tokens: list[int] = []
        self._hiddens: dict[int, torch.Tensor] = {}  # position -> (H,) fp16
        # The MTP block cache is LOCALLY DENSE (row i is the i-th absorbed
        # token, cache_position counts from 0) while rope rides ABSOLUTE
        # positions via the cos/sin tables.
        self._abs_absorbed = 0   # absolute positions [0, abs_absorbed) absorbed
        self._local_rows = 0     # rows written into the MTP block cache

    # ------------------------------------------------------------- contract

    def bind(self, gen: "AlloyGenerator") -> None:

        self._gen = gen
        mtp = load_quantized_mtp(
            gen.model, self._blob_path, quantize=self._quantize, draft_topk=self._draft_topk,
        )
        # qwen3_5 RMSNorm scales by (1 + weight): the EFFECTIVE output-norm
        # scale for pre-norm recovery is (1 + norm.weight) — dividing by the
        # bare weight mis-scales the hidden and zeroes draft acceptance.
        mtp.bind_runtime(
            gen.model.model.embed_tokens,
            gen.model.model.rotary_emb,
            (1.0 + gen.model.model.norm.weight).detach(),
        )
        self._mtp = mtp

    def warmup(self) -> None:
        if self._pins is not None:
            return
        self._pins = self._pin_draft_plans()

    def observe(self, tokens: list[int], taps: TapBatch | None, start: int) -> None:
        # Token mirror (overwrite semantics).
        end = start + len(tokens)
        if len(self._tokens) < end:
            self._tokens.extend([0] * (end - len(self._tokens)))
        self._tokens[start:end] = tokens
        # Post-norm hidden of row j is the hidden AFTER consuming tokens[j] at
        # position start+j — keyed by that position. Rows past the committed
        # point are pruned by truncate() right after.
        if taps is not None and taps.post_norm is not None:
            rows = taps.post_norm[0]  # (M, H)
            for j in range(taps.rows):
                self._hiddens[start + j] = rows[j].detach().to(torch.float16).clone()

    def truncate(self, length: int) -> None:
        if length < len(self._tokens):
            del self._tokens[length:]
        for pos in [p for p in self._hiddens if p >= length]:
            del self._hiddens[pos]
        if self._abs_absorbed > length:
            # Branch/warm rewind past the absorbed point. The block cache is
            # locally dense, so absolute rewinds don't map to a row pointer —
            # drop it and re-absorb from the anchor (costs a couple of
            # low-context rounds after a rewind, nothing else).
            self._pins["mcache"].reset()
            self._local_rows = 0
            self._abs_absorbed = length

    def propose(self, anchor: int, position: int) -> Proposal:
        if self._pins is None:
            self.warmup()
        pins = self._pins
        # Unabsorbed committed tail + anchor, ending at `position`. The
        # steady state pends <= 2 tokens/round (accepted + bonus); more
        # pending means a fresh/rewound state — absorb just the anchor into
        # the (empty) cache (one expected miss while context rebuilds).
        pending = position + 1 - self._abs_absorbed
        if pending > 2:
            if self._local_rows != 0:
                pins["mcache"].reset()
                self._local_rows = 0
            abs_start = position
        else:
            abs_start = position + 1 - pending
        toks = [*self._tokens[abs_start:position], anchor]
        m = len(toks)
        plan, args, out_idx, te, hid, cos, sin, cp = pins["draft_plans"][m]
        emb = pins["emb_view"]
        cos_full, sin_full = pins["cos_full"], pins["sin_full"]
        for j, t in enumerate(toks):
            p = abs_start + j  # absolute position of this row
            te[0, j].copy_(emb[t])
            h = self._hiddens.get(p - 1)
            if h is None:
                hid[0, j].zero_()
            else:
                hid[0, j].copy_(h)
            cos[0, j].copy_(cos_full[p])
            sin[0, j].copy_(sin_full[p])
        cp.copy_(torch.arange(self._local_rows, self._local_rows + m, dtype=torch.long))
        pins["mcache"].layers[0].cumulative_length.fill_(self._local_rows)
        res = _execute_plan(plan, args, wanted_outputs=frozenset((out_idx,)), args_stable=True)
        out = res[out_idx] if isinstance(res, tuple) else res
        self._local_rows += m
        self._abs_absorbed = position + 1
        return Proposal([int(out.reshape(-1)[0].item())])

    def state_bytes_per_token(self) -> int:
        # One MTP block layer of K/V per token (fp16) + the staged hidden.
        if self._mtp is None:
            return 0
        cfg = self._mtp.cache_config
        kv = 2 * cfg.num_key_value_heads * cfg.head_dim * 2
        return kv + cfg.hidden_size * 2

    def snapshot_head(self, rows: int) -> object | None:
        # MTP block cache rows for [0, rows) + the token/hidden mirrors. The
        # session calls this around foreign side requests; M-C2 wires it.
        layer = self._pins["mcache"].layers[0] if self._pins else None
        if layer is None or layer.keys is None:
            return None
        return (
            layer.keys[:, :, :rows].clone(),
            layer.values[:, :, :rows].clone(),
            list(self._tokens),
            dict(self._hiddens),
            self._absorbed,
        )

    def restore_head(self, snap: object) -> None:
        if snap is None or self._pins is None:
            return
        keys, values, tokens, hiddens, absorbed = snap
        layer = self._pins["mcache"].layers[0]
        rows = keys.shape[2]
        layer.keys[:, :, :rows].copy_(keys)
        layer.values[:, :, :rows].copy_(values)
        self._tokens = tokens
        self._hiddens = hiddens
        self._absorbed = absorbed

    # ------------------------------------------------------------- plumbing

    def _pin_draft_plans(self) -> dict:
        """Compile + pin the M=1 and M=2 MTP draft plans against the block's
        own cache, plus the rotary/embedding tables."""

        gen = self._gen
        device = next(gen.model.parameters()).device
        H = gen.model.config.hidden_size
        draft = torch.compile(MTPDraftStep(self._mtp), backend="alloy", dynamic=False)
        mcache = AlloyStaticCache(
            config=self._mtp.cache_config, max_cache_len=gen.kv.max_cache_len,
            max_batch_size=1, cache_dtype=gen.cache_dtype,
        )
        # Rotary tables (cos/sin per absolute position), fp16, CPU-readable.
        max_pos = gen.kv.max_cache_len
        cos_full, sin_full = gen.model.model.rotary_emb(
            torch.zeros(1, 1, H, dtype=gen.cache_dtype, device=device),
            torch.arange(max_pos, device=device).unsqueeze(0),
        )
        cos_full = cos_full[0].to(gen.cache_dtype).contiguous()
        sin_full = sin_full[0].to(gen.cache_dtype).contiguous()
        rotary_dim = int(cos_full.shape[-1])
        # Dequantized input-embedding table (the quantized embed op is
        # GPU-plan-only; the per-round gather happens host-side into pinned
        # inputs).
        vocab = gen.model.config.vocab_size
        reader = gguf.GGUFReader(str(self._blob_path))
        et = next(x for x in reader.tensors if x.name == "token_embd.weight")
        emb_np = np.ascontiguousarray(gguf.dequantize(et.data, et.tensor_type))
        emb_arr = _alloc_aligned((vocab, H), float16)
        emb_view = make_tensor_from_ptr(
            emb_arr.base_ptr, (vocab, H), float16, total_nbytes=emb_arr.metal_nbytes,
        )
        emb_view.copy_(torch.from_numpy(emb_np).to(gen.cache_dtype).reshape(vocab, H))

        draft_plans = {}
        with torch.inference_mode():
            for m in (1, 2):
                te = torch.zeros((1, m, H), dtype=gen.cache_dtype, device=device)
                hid = torch.zeros((1, m, H), dtype=gen.cache_dtype, device=device)
                cos = torch.zeros((1, m, rotary_dim), dtype=gen.cache_dtype, device=device)
                sin = torch.zeros((1, m, rotary_dim), dtype=gen.cache_dtype, device=device)
                pos = torch.arange(m, device=device).unsqueeze(0)
                cp = torch.arange(m, device=device)
                mcache.reset()
                with capture_plan() as ms:
                    for _ in range(2):
                        draft(te, hid, cos, sin, pos, mcache, cp)
                mi = next(
                    i for i, e in enumerate(ms.plan.output_mapping)
                    if isinstance(e, OutputSlot) and e.dtype.ir == "i64"
                )
                draft_plans[m] = (ms.plan, ms.args, mi, te, hid, cos, sin, cp)
            mcache.reset()
        return {
            "draft_plans": draft_plans, "mcache": mcache,
            "cos_full": cos_full, "sin_full": sin_full,
            "emb_arr": emb_arr, "emb_view": emb_view, "rotary_dim": rotary_dim,
        }
