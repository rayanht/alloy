"""DFlash block-diffusion drafter; arXiv 2602.06036:

The draft is a small qwen3-style transformer (z-lab checkpoint: a few layers at
the target's hidden size, q/k head-norms, full rope at its own theta) that
predicts a whole block in ONE forward: input = [anchor, mask × (block-1)],
attention is bidirectional within the block and attends the *context KV*
through each layer's cross-context window — per-layer K/V projections of the
fused target features (`hidden_norm(fc(cat(target-layer hiddens)))`), cached per
committed position. Mask-slot j's output IS the token at its own position
(diffusion convention), read through the target's shared lm_head.

alloy mapping (everything position-aligned, dead rows by overwrite):
- ctx KV cache: per draft layer, fp16 alloy buffers. `full_attention` layers are
  native-sized and linear (slot i == absolute position i); `sliding_attention`
  layers are a circular ring of `sliding_window` rows (slot == position % window).
  `observe()` writes feature K/V rows for ALL forwarded rows; the committed
  pointer makes overshoot rows dead, and the next round's write overwrites them.
- propose plan: block embeds → draft layers (attention via
  `attention_kv_update_multi_bidir`: fused block-KV write at [pos, pos+B) + every
  row attends the in-window context plus the block) → final norm → shared lm_head
  on the B-1 mask rows → in-plan argmax. The block's own KV rows land in the
  ctx cache but die by the same overwrite rule.
- observe plan: tap tensors → fc → hidden_norm → per-layer
  k_norm(k_proj)/v_proj + rope at absolute positions → `spec_kv_write` (ring-wrapped
  on the sliding layers).
- rope: the draft's own full-head-dim tables (theta from its config — NOT the
  target's partial-rope tables), staged per call from host like the MTP
  drafter; in-plan rope derivation constant-folds and breaks positions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import gguf
import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from torch import nn

from alloy._compiler.dtypes import float16
from alloy._dispatch.buf_utils import _alloc_aligned
from alloy_torch.tensor_bridge import make_tensor_from_ptr
from alloy_torch.backend import OutputSlot, _execute_plan, capture_plan

from .contract import Proposal, TapBatch, TargetTaps

if TYPE_CHECKING:
    from alloy_server.generation.generator import AlloyGenerator


DFLASH_DRAFTS: dict[str, tuple[str, ...]] = {
    "z-lab/Qwen3.5-4B-DFlash": ("qwen3.5:4b", "Qwen3.5-4B"),
    "z-lab/Qwen3.5-9B-DFlash": ("qwen3.5:9b", "Qwen3.5-9B"),
    "z-lab/Qwen3.5-27B-DFlash": ("qwen3.5:27b", "Qwen3.5-27B"),
    "z-lab/Qwen3.5-35B-A3B-DFlash": ("qwen3.5:35b", "Qwen3.5-35B-A3B"),
    "z-lab/Qwen3.5-122B-A10B-DFlash": ("qwen3.5:122b", "Qwen3.5-122B-A10B"),
    "z-lab/Qwen3.6-27B-DFlash": ("qwen3.6:27b", "Qwen3.6-27B"),
    "z-lab/Qwen3.6-35B-A3B-DFlash": ("qwen3.6:35b", "Qwen3.6-35B-A3B"),
}


def dflash_alias_key(ref: str) -> str:
    """Normalize a model reference to its DFlash lookup key. An HF ref keeps only
    its model identity (org, `-GGUF` suffix, and `:quant` tag dropped); an Ollama
    tag (`name:size`, no `/`) is matched verbatim."""
    ref = ref.strip().lower()
    if "/" in ref:
        ref = ref.rsplit("/", 1)[1].partition(":")[0]
        if ref.endswith("-gguf"):
            ref = ref[: -len("-gguf")]
    return ref


DFLASH_CHECKPOINTS = {
    dflash_alias_key(alias): repo
    for repo, aliases in DFLASH_DRAFTS.items()
    for alias in aliases
}


def resolve_dflash_checkpoint(model_name: str) -> Path:
    """z-lab draft checkpoint directory for a served model reference (Ollama tag
    or HF repo, any GGUF mirror)."""
    repo = DFLASH_CHECKPOINTS.get(dflash_alias_key(model_name))
    if repo is None:
        known = ", ".join(aliases[0] for aliases in DFLASH_DRAFTS.values())
        raise ValueError(f"no DFlash draft known for {model_name!r}; known: {known}")
    try:
        path = snapshot_download(repo, local_files_only=True)
    except Exception as exc:
        raise FileNotFoundError(
            f"DFlash draft {repo} not downloaded — run: hf download {repo}"
        ) from exc
    return Path(path)


class DFlashRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dt)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class DFlashAttention(nn.Module):
    """Draft self-attention: Q from the block, KV = [ctx cache ++ block] via
    the fused bidirectional multi-token op."""

    def __init__(self, hidden: int, heads: int, kv_heads: int, head_dim: int, eps: float, sliding_window: int = 0) -> None:
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.sliding_window = sliding_window
        self.q_proj = nn.Linear(hidden, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, hidden, bias=False)
        self.q_norm = DFlashRMSNorm(head_dim, eps)
        self.k_norm = DFlashRMSNorm(head_dim, eps)

    def forward(self, x, cos, sin, ctx_k, ctx_v, cache_pos):
        b, m, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(b, m, self.heads, self.head_dim))
        k = self.k_norm(self.k_proj(x).view(b, m, self.kv_heads, self.head_dim))
        v = self.v_proj(x).view(b, m, self.kv_heads, self.head_dim).transpose(1, 2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        # cos/sin (1, M, head_dim), broadcast over heads.
        cos_b = cos.unsqueeze(1)
        sin_b = sin.unsqueeze(1)
        q = (q * cos_b) + (_rotate_half(q) * sin_b)
        k = (k * cos_b) + (_rotate_half(k) * sin_b)
        attn = torch.ops.alloy.attention_kv_update_multi_bidir(
            q.to(ctx_k.dtype), k.to(ctx_k.dtype), v.to(ctx_k.dtype),
            cache_pos, ctx_k, ctx_v, float(self.head_dim) ** -0.5, self.sliding_window,
        )
        return self.o_proj(attn.transpose(1, 2).reshape(b, m, -1))


class DFlashLayer(nn.Module):
    def __init__(self, hidden: int, inter: int, heads: int, kv_heads: int, head_dim: int, eps: float, sliding_window: int = 0) -> None:
        super().__init__()
        self.input_layernorm = DFlashRMSNorm(hidden, eps)
        self.self_attn = DFlashAttention(hidden, heads, kv_heads, head_dim, eps, sliding_window)
        self.post_attention_layernorm = DFlashRMSNorm(hidden, eps)
        self.mlp = nn.ModuleDict(dict(
            gate_proj=nn.Linear(hidden, inter, bias=False),
            up_proj=nn.Linear(hidden, inter, bias=False),
            down_proj=nn.Linear(inter, hidden, bias=False),
        ))

    def forward(self, x, cos, sin, ctx_k, ctx_v, cache_pos):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, ctx_k, ctx_v, cache_pos)
        h = self.post_attention_layernorm(x)
        h = self.mlp["down_proj"](
            torch.nn.functional.silu(self.mlp["gate_proj"](h)) * self.mlp["up_proj"](h)
        )
        return x + h


class DFlashDraftModel(nn.Module):
    """The z-lab draft: fc (n_taps·H→H) + hidden_norm for the context features,
    N qwen3-style layers, final norm. Embedding + lm_head are the TARGET's
    (tied/quantized) modules, bound at drafter.bind()."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        h = cfg["hidden_size"]
        eps = cfg["rms_norm_eps"]
        self.n_taps = len(cfg["dflash_config"]["target_layer_ids"])
        self.fc = nn.Linear(self.n_taps * h, h, bias=False)
        self.hidden_norm = DFlashRMSNorm(h, eps)
        # Per-layer cross-context window (the checkpoint's trained config):
        # `sliding_attention` layers cross-attend the most recent `sliding_window`
        # target features through a circular ring of that size (qwen3.6: 4096 on
        # 5/6 layers); `full_attention` layers see all of it (window 0, linear cache).
        layer_types = cfg.get("layer_types") or []
        sw = int(cfg.get("sliding_window") or 0)
        self.layers = nn.ModuleList(
            DFlashLayer(
                h, cfg["intermediate_size"], cfg["num_attention_heads"],
                cfg["num_key_value_heads"], cfg["head_dim"], eps,
                sw if i < len(layer_types) and layer_types[i] == "sliding_attention" else 0,
            )
            for i in range(cfg["num_hidden_layers"])
        )
        self.norm = DFlashRMSNorm(h, eps)

    def fuse(self, taps: list[torch.Tensor]) -> torch.Tensor:
        return self.hidden_norm(self.fc(torch.cat(taps, dim=-1)))

    def block_forward(self, x, cos, sin, ctx_ks, ctx_vs, cache_pos):
        for layer, ck, cv in zip(self.layers, ctx_ks, ctx_vs):
            x = layer(x, cos, sin, ck, cv, cache_pos)
        return self.norm(x)


class DFlashBlockStep(nn.Module):
    """Compile target for propose: block embeds → layers → lm_head argmax on
    the mask rows (diffusion: mask slot j's output IS position pos+j's token)."""

    def __init__(self, draft: DFlashDraftModel, lm_head: nn.Module) -> None:
        super().__init__()
        self.draft = draft
        self.lm_head = lm_head

    def forward(self, block_embeds, cos, sin, cache_pos, *ctx):
        n = len(ctx) // 2
        out = self.draft.block_forward(
            block_embeds, cos, sin, ctx[:n], ctx[n:], cache_pos,
        )
        logits = self.lm_head(out[:, 1:])
        return logits.argmax(dim=-1)


class DFlashObserve(nn.Module):
    """Compile target for observe: tap hiddens → fused features → per-layer
    K/V rows written into the ctx caches at [cache_pos, cache_pos+M)."""

    def __init__(self, draft: DFlashDraftModel) -> None:
        super().__init__()
        self.draft = draft

    def forward(self, *args):
        nt = self.draft.n_taps
        taps, cos, sin, cache_pos = args[:nt], args[nt], args[nt + 1], args[nt + 2]
        ctx = args[nt + 3 :]
        n = len(ctx) // 2
        h = self.draft.fuse(list(taps))
        b, m, _ = h.shape
        outs = []
        cos_b = cos.unsqueeze(1)
        sin_b = sin.unsqueeze(1)
        for layer, ck, cv in zip(self.draft.layers, ctx[:n], ctx[n:]):
            attn = layer.self_attn
            k = attn.k_norm(attn.k_proj(h).view(b, m, attn.kv_heads, attn.head_dim)).transpose(1, 2)
            v = attn.v_proj(h).view(b, m, attn.kv_heads, attn.head_dim).transpose(1, 2)
            k = (k * cos_b) + (_rotate_half(k) * sin_b)
            outs.append(torch.ops.alloy.spec_kv_write(
                k.to(ck.dtype), v.to(ck.dtype), cache_pos, ck, cv, attn.sliding_window,
            ))
        return tuple(outs)


class DFlashDrafter:
    """Contract drafter for DFlash block diffusion. The default block size is
    the CHECKPOINT's trained width (the z-lab drafts train at block 16).
    `blob_path` is the served model's GGUF blob (embed table for host-side
    block staging, the MTPDrafter pattern)."""

    name = "dflash"

    def __init__(self, checkpoint: str | Path, blob_path, block_size: int | None = None) -> None:
        self._ckpt = Path(checkpoint)
        self._blob_path = blob_path
        cfg = json.loads((self._ckpt / "config.json").read_text())
        self._cfg = cfg
        dfc = cfg["dflash_config"]
        # block_size lives under dflash_config (qwen3.6 drafts) or at top level
        # (qwen3.5 drafts).
        default_block = cfg.get("block_size", dfc.get("block_size"))
        self.block_size = int(block_size if block_size is not None else default_block)
        self.max_draft_tokens = self.block_size - 1
        self.mask_token_id = int(dfc["mask_token_id"])
        self.taps = TargetTaps(layer_ids=tuple(dfc["target_layer_ids"]))
        self._gen: "AlloyGenerator | None" = None
        self._draft: DFlashDraftModel | None = None
        self._pins: dict | None = None
        self._ctx_len = 0  # ctx rows valid for positions [0, ctx_len)

    # ------------------------------------------------------------- contract

    def bind(self, gen: "AlloyGenerator") -> None:

        self._gen = gen
        cfg = self._cfg
        if cfg["vocab_size"] != gen.model.config.vocab_size:
            raise ValueError(
                f"draft/target vocab mismatch: {cfg['vocab_size']} != "
                f"{gen.model.config.vocab_size}"
            )
        draft = DFlashDraftModel(cfg)
        weights = load_file(str(self._ckpt / "model.safetensors"))
        # assign=True adopts the fp16 tensors as the params; a plain
        # load_state_dict copies into the fp32-constructed params and leaves
        # the module fp32.
        draft.load_state_dict(
            {k: v.to(torch.float16) for k, v in weights.items()},
            strict=True, assign=True,
        )
        draft.eval()
        for p in draft.parameters():
            p.requires_grad_(False)
        self._draft = draft
        # ctx KV caches, per layer: full-attention layers are native-sized and
        # linear (slot == absolute position); sliding layers are a circular ring
        # of `sliding_window` rows (slot == position % window), matching the
        # attention/write kernels' wrap. The ring keeps the cross-context at the
        # trained window and shrinks the resident KV (~6.4 GB → ~1 GB at qwen3.6).
        kvh, hd = cfg["num_key_value_heads"], cfg["head_dim"]
        s_max = gen.kv.max_cache_len
        self._ctx_arrs = []
        self._ctx_k_t: list[torch.Tensor] = []
        self._ctx_v_t: list[torch.Tensor] = []
        for layer in draft.layers:
            rows = layer.self_attn.sliding_window or s_max
            for views in (self._ctx_k_t, self._ctx_v_t):
                arr = _alloc_aligned((1, kvh, rows, hd), float16)
                t = make_tensor_from_ptr(
                    arr.base_ptr, (1, kvh, rows, hd), float16,
                    total_nbytes=arr.metal_nbytes,
                )
                self._ctx_arrs.append(arr)
                views.append(t)
                torch._dynamo.mark_static_address(t)

    def warmup(self) -> None:
        if self._pins is None:
            self._pins = self._pin_plans()
            # Observe plans for the prefill chunk widths (the verify width is
            # pinned in _pin_plans). Prefill taps arrive bucket-padded; each
            # bucket is its own fixed shape. Other widths pin lazily.
            for width in self._gen.prefill_chunks:
                self._pin_observe(width)

    def observe(self, tokens: list[int], taps: TapBatch | None, start: int) -> None:
        if taps is None or not taps.layers:
            return  # rows without taps can't produce features (cold gap)
        if self._pins is None:
            self.warmup()
        width = int(taps.layers[0].shape[1])  # bucket-padded plan width
        plan_entry = self._pins["observe"].get(width)
        if plan_entry is None:
            self._pin_observe(width)
            plan_entry = self._pins["observe"][width]
        p_plan, p_args, t_ins, cos_in, sin_in, pos_in = plan_entry
        for buf, tap in zip(t_ins, taps.layers):
            buf.copy_(tap.to(buf.dtype))
        cos_in.copy_(self._pins["cos_full"][start : start + width].unsqueeze(0))
        sin_in.copy_(self._pins["sin_full"][start : start + width].unsqueeze(0))
        pos_in.copy_(torch.arange(start, start + width, dtype=torch.long))
        _execute_plan(p_plan, p_args, args_stable=True)
        # Only taps.rows rows are REAL; padded rows land in ctx but the
        # pointer keeps them dead (next append overwrites them).
        self._ctx_len = max(self._ctx_len, start + taps.rows)

    def propose(self, anchor: int, position: int) -> Proposal:
        if self._pins is None:
            self.warmup()
        if self._ctx_len < position:
            # Feature gap (cold prompt without prefill taps, or a rewind past
            # saved features): the draft can't attend unwritten ctx rows.
            return Proposal([])
        pins = self._pins
        plan, args, out_idx, emb_in, cos_in, sin_in, pos_in = pins["block"]
        b = self.block_size
        emb_in[0, 0].copy_(pins["emb_view"][anchor])
        # rows 1..B-1 stay the pre-staged mask embedding (warmup wrote them).
        cos_in.copy_(pins["cos_full"][position : position + b].unsqueeze(0))
        sin_in.copy_(pins["sin_full"][position : position + b].unsqueeze(0))
        pos_in.copy_(torch.arange(position, position + b, dtype=torch.long))
        res = _execute_plan(plan, args, wanted_outputs=frozenset((out_idx,)), args_stable=True)
        out = res[out_idx] if isinstance(res, tuple) else res
        return Proposal([int(t) for t in out.reshape(-1).tolist()])

    def truncate(self, length: int) -> None:
        if self._ctx_len > length:
            self._ctx_len = length

    def state_bytes_per_token(self) -> int:
        cfg = self._cfg
        return 2 * cfg["num_hidden_layers"] * cfg["num_key_value_heads"] * cfg["head_dim"] * 2

    def snapshot_head(self, rows: int) -> object | None:
        rows = min(rows, self._ctx_len)
        if rows <= 0:
            return None
        # Per-cache row count: full layers slice [0, rows); ring layers cap at
        # their window (the whole ring is live once ctx_len >= window).
        return (
            [t[:, :, : min(rows, t.shape[2])].clone() for t in self._ctx_k_t],
            [t[:, :, : min(rows, t.shape[2])].clone() for t in self._ctx_v_t],
            self._ctx_len,
        )

    def restore_head(self, snap: object) -> None:
        if snap is None:
            return
        ks, vs, ctx_len = snap
        for t, s in zip(self._ctx_k_t, ks):
            t[:, :, : s.shape[2]].copy_(s)
        for t, s in zip(self._ctx_v_t, vs):
            t[:, :, : s.shape[2]].copy_(s)
        self._ctx_len = ctx_len

    def tune_targets(self) -> list:
        """(label, module, inputs, record_only) tuples for `alloy tune --spec
        dflash` — the draft's propose and observe forwards at production shapes,
        so their GEMM/norm kernels get tuned configs like every other forward
        (the vision capture_targets() pattern). `record_only` is True for the
        large-M (chunk-width) observe target so its extraction doesn't hold the
        full activation graph. Requires bind()."""
        cfg = self._cfg
        h = cfg["hidden_size"]
        hd = cfg["head_dim"]
        b = self.block_size
        ctx = (*self._ctx_k_t, *self._ctx_v_t)
        gen = self._gen

        class ProposeShim(nn.Module):
            def __init__(self, inner, ctx_bufs) -> None:
                super().__init__()
                self.inner = inner
                self._ctx_bufs = ctx_bufs

            def forward(self, block_embeds, cos, sin, cache_pos):
                return self.inner(block_embeds, cos, sin, cache_pos, *self._ctx_bufs)

        class ObserveShim(nn.Module):
            def __init__(self, inner, ctx_bufs) -> None:
                super().__init__()
                self.inner = inner
                self._ctx_bufs = ctx_bufs

            def forward(self, *args):
                return self.inner(*args, *self._ctx_bufs)

        targets = [(
            f"dflash propose (block {b})",
            ProposeShim(DFlashBlockStep(self._draft, gen.model.lm_head), ctx),
            {
                "block_embeds": torch.zeros((1, b, h), dtype=torch.float16),
                "cos": torch.zeros((1, b, hd), dtype=torch.float16),
                "sin": torch.zeros((1, b, hd), dtype=torch.float16),
                "cache_pos": torch.arange(b, dtype=torch.long),
            },
            False,  # small-M (block width): keep realistic snapshots
        )]
        for m in (b, gen.chunk_prefill_size):
            obs_inputs = (
                *(torch.zeros((1, m, h), dtype=torch.float16) for _ in range(self._draft.n_taps)),
                torch.zeros((1, m, hd), dtype=torch.float16),  # cos
                torch.zeros((1, m, hd), dtype=torch.float16),  # sin
                torch.arange(m, dtype=torch.long),  # cache_pos
            )
            targets.append((
                f"dflash observe (M={m})",
                ObserveShim(DFlashObserve(self._draft), ctx),
                obs_inputs,
                m >= gen.chunk_prefill_size,  # large-M observe: record_only
            ))
        return targets

    # ------------------------------------------------------------- plumbing

    def _rope_tables(self) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self._cfg
        d = cfg["head_dim"]
        # rope_theta is top-level (qwen3.5 drafts) or under rope_parameters
        # (qwen3.6 drafts, newer transformers config layout).
        theta = float(cfg.get("rope_theta") or cfg["rope_parameters"]["rope_theta"])
        max_pos = self._gen.kv.max_cache_len
        inv = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
        t = torch.arange(max_pos, dtype=torch.float32)
        freqs = torch.outer(t, inv)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(torch.float16), emb.sin().to(torch.float16)

    def _pin_plans(self) -> dict:

        gen = self._gen
        cfg = self._cfg
        h = cfg["hidden_size"]
        cos_full, sin_full = self._rope_tables()

        # Dequantized embedding table for host-side block staging (qwen3.5
        # ties lm_head to token_embd).
        vocab = cfg["vocab_size"]
        reader = gguf.GGUFReader(str(self._blob_path))
        et = next(x for x in reader.tensors if x.name == "token_embd.weight")
        emb_np = np.ascontiguousarray(gguf.dequantize(et.data, et.tensor_type))
        emb_arr = _alloc_aligned((vocab, h), float16)
        emb_view = make_tensor_from_ptr(
            emb_arr.base_ptr, (vocab, h), float16, total_nbytes=emb_arr.metal_nbytes,
        )
        emb_view.copy_(torch.from_numpy(emb_np).to(torch.float16).reshape(vocab, h))

        b = self.block_size
        ctx = (*self._ctx_k_t, *self._ctx_v_t)

        # --- propose plan (block step) ---
        block = torch.compile(
            DFlashBlockStep(self._draft, gen.model.lm_head),
            backend="alloy", dynamic=False,
        )
        emb_in = torch.zeros((1, b, h), dtype=torch.float16)
        emb_in[0, 1:].copy_(emb_view[self.mask_token_id].unsqueeze(0))
        cos_in = torch.zeros((1, b, cfg["head_dim"]), dtype=torch.float16)
        sin_in = torch.zeros((1, b, cfg["head_dim"]), dtype=torch.float16)
        pos_in = torch.arange(b, dtype=torch.long)
        with torch.inference_mode():
            with capture_plan() as slot:
                for _ in range(2):
                    block(emb_in, cos_in, sin_in, pos_in, *ctx)
        out_idx = next(
            i for i, e in enumerate(slot.plan.output_mapping)
            if isinstance(e, OutputSlot) and e.dtype.ir == "i64"
        )
        block_pin = (slot.plan, slot.args, out_idx, emb_in, cos_in, sin_in, pos_in)
        # Re-stage the mask rows (the capture left them set; keep invariant).
        emb_in[0, 1:].copy_(emb_view[self.mask_token_id].unsqueeze(0))

        pins = {
            "block": block_pin,
            "observe": {},
            "observe_module": torch.compile(
                DFlashObserve(self._draft), backend="alloy", dynamic=False,
            ),
            "cos_full": cos_full,
            "sin_full": sin_full,
            "emb_view": emb_view,
            "emb_arr": emb_arr,
        }
        self._pins = pins
        self._pin_observe(b)
        return pins

    def _pin_observe(self, m: int) -> None:
        """Compile + pin the observe plan at row width `m` (the verify block
        width and each prefill bucket get their own fixed-shape plan)."""

        cfg = self._cfg
        h = cfg["hidden_size"]
        observe = self._pins["observe_module"]
        ctx = (*self._ctx_k_t, *self._ctx_v_t)
        t_ins = [torch.zeros((1, m, h), dtype=torch.float16) for _ in range(self._draft.n_taps)]
        o_cos = torch.zeros((1, m, cfg["head_dim"]), dtype=torch.float16)
        o_sin = torch.zeros((1, m, cfg["head_dim"]), dtype=torch.float16)
        o_pos = torch.arange(m, dtype=torch.long)
        # The pin executes the observe module with ZERO tap inputs at
        # positions [0, m) — writing garbage feature rows into the LIVE ctx
        # cache. A lazy mid-request pin would poison rows the draft attends
        # (acceptance collapses to 0), so snapshot and restore the clobbered
        # rows.
        snaps = [(t, t[:, :, :m].clone()) for t in ctx]
        try:
            with torch.inference_mode():
                with capture_plan() as oslot:
                    for _ in range(2):
                        observe(*t_ins, o_cos, o_sin, o_pos, *ctx)
        finally:
            with torch.inference_mode():
                for t, snap in snaps:
                    t[:, :, :m].copy_(snap)
        self._pins["observe"][m] = (oslot.plan, oslot.args, t_ins, o_cos, o_sin, o_pos)
