"""Pure-torch DeepSeek-V4.1 forward — the numerical oracle for the alloy path.

A faithful port of the reference `inference/model.py` (batch 1, fp32 compute) over a
flat dict of dequantized fp32 weights keyed by GGUF tensor name, with the reference's
tilelang kernels replaced by torch: `sparse_attn`, `hc_split_sinkhorn`, the fp8 /
fp4 quantize-dequantize rounding the caches go through. GEMMs run in fp32 on the
dequantized weights (the reference quantizes activations to fp8 per 32-block for its
fp8/fp4 GEMMs; that is a GEMM-precision choice, not model structure, and is not
reproduced). Slow by construction: layer-level and tiny-model checks only.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from alloy_server.models.deepseek_v41.config import DeepseekV41Config
from alloy_server.models.deepseek_v41.engram import EngramHasher

Weights = dict[str, torch.Tensor]

FP8_MAX = 448.0
FP4_MAX = 6.0
FP4_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def pow2_ceil(x: torch.Tensor) -> torch.Tensor:
    """2^ceil(log2(x)) for x > 0 (the E8M0 scale rounding)."""
    return torch.exp2(torch.ceil(torch.log2(x)))


def round_e4m3(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float8_e4m3fn).to(torch.float32)


def round_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest-even onto the E2M1 value grid (|x| <= 6 assumed)."""
    mag = x.abs()
    grid = FP4_VALUES.to(x.device)
    mid = (grid[1:] + grid[:-1]) / 2
    # bucketize(right=False) lands a midpoint on the lower value; ties go to the
    # even encoding, i.e. the even grid index
    idx = torch.bucketize(mag, mid, right=False)
    tie = torch.isin(mag, mid)
    idx = torch.where(tie & (idx % 2 == 1), idx + 1, idx)
    return torch.copysign(grid[idx], x)


def fp8_round(x: torch.Tensor, block: int = 32) -> torch.Tensor:
    """act_quant(..., scale_fmt="ue8m0", inplace=True): per-block E8M0 scale, E4M3 values."""
    shape = x.shape
    g = x.float().reshape(-1, block)
    amax = g.abs().amax(dim=1, keepdim=True).clamp_min(1e-4)
    s = pow2_ceil(amax / FP8_MAX)
    y = round_e4m3((g / s).clamp(-FP8_MAX, FP8_MAX)) * s
    return y.reshape(shape).to(x.dtype)


def fp4_round(x: torch.Tensor, block: int, scale_e4m3: bool) -> torch.Tensor:
    """fp4_act_quant(..., inplace=True): E2M1 values with an E8M0 (indexer) or E4M3
    (compressed KV) per-block scale."""
    shape = x.shape
    g = x.float().reshape(-1, block)
    amax = g.abs().amax(dim=1, keepdim=True)
    if scale_e4m3:
        amax = amax.clamp_min(6 * 2.0**-9)
        s = round_e4m3(amax / FP4_MAX)
    else:
        amax = amax.clamp_min(6 * 2.0**-126)
        s = pow2_ceil(amax / FP4_MAX)
    y = round_e2m1((g / s).clamp(-FP4_MAX, FP4_MAX)) * s
    return y.reshape(shape).to(x.dtype)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.float()
    return weight.float() * (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps))


def precompute_freqs_cis(dim: int, seqlen: int, original_seq_len: int, base: float, factor: float,
                         beta_fast: float, beta_slow: float) -> torch.Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        def corrected_dim(rotations: float) -> float:
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    freqs = torch.outer(torch.arange(seqlen, dtype=torch.float32), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Interleaved-pair complex rotation over the last dim; `freqs_cis` is [S, d/2]
    broadcast over any head dim between S and d."""
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)).contiguous())
    if inverse:
        freqs_cis = freqs_cis.conj()
    while freqs_cis.ndim < xc.ndim:
        freqs_cis = freqs_cis.unsqueeze(-2)
    return torch.view_as_real(xc * freqs_cis).flatten(-2).to(x.dtype)


def window_topk_idxs(window_size: int, seqlen: int, start_pos: int) -> torch.Tensor:
    """Which ring slots each query attends; -1 = empty. Prefill rows see their own
    causal window (indices into the chunk); decode sees the whole ring."""
    if start_pos == 0:
        end = torch.arange(seqlen).unsqueeze(1)
        idxs = (end - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
        return torch.where(idxs > end, -1, idxs).int()
    oldest = start_pos % window_size + 1
    idxs = torch.cat([torch.arange(oldest, window_size), torch.arange(oldest)])
    return torch.where(idxs > start_pos, -1, idxs).int().unsqueeze(0)


def sparse_attn(q: torch.Tensor, kv: torch.Tensor, sink: torch.Tensor, idxs: torch.Tensor, scale: float) -> torch.Tensor:
    """q [S, H, D], kv [N, D] (key and value), idxs [S, K] with -1 = absent, sink [H]."""
    valid = idxs >= 0
    gathered = kv[idxs.clamp_min(0)]  # [S, K, D]
    scores = torch.einsum("shd,skd->shk", q.float(), gathered.float()) * scale
    scores = scores.masked_fill(~valid[:, None, :], -torch.inf)
    scores = torch.cat([scores, sink.float().view(1, -1, 1).expand(scores.shape[0], -1, 1)], dim=-1)
    p = scores.softmax(-1)[..., :-1]
    return torch.einsum("shk,skd->shd", p, gathered.float())


def topk_lowest_index(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k indices per row with ties resolved to the lowest index (the reference's
    `topk` leaves tie order unspecified; relu'd index scores tie at 0 constantly)."""
    order = torch.sort(scores, dim=-1, descending=True, stable=True).indices
    return order[..., :k]


def select_candidate_blocks(logits: torch.Tensor, compress_lens: torch.Tensor | int, topk_blocks: int,
                            block_size: int) -> torch.Tensor:
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)
    last = (torch.as_tensor(compress_lens) - 1) // block_size
    scores = scores.masked_fill(torch.arange(num_blocks) == last, torch.inf)
    top_idx = topk_lowest_index(scores, min(topk_blocks, num_blocks))
    top_val = scores.gather(-1, top_idx)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top_idx, top_val > -torch.inf)
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


def hc_split_sinkhorn(mixes: torch.Tensor, scale: torch.Tensor, base: torch.Tensor, hc: int, iters: int,
                      eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pre = torch.sigmoid(mixes[:, :hc] * scale[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[:, hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
    comb = (mixes[:, 2 * hc :] * scale[2] + base[2 * hc :]).view(-1, hc, hc)
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


class ReferenceModel:
    """Stateful (KV caches live here) forward over `weights`; `forward(ids, start_pos)`
    returns the last-position logits and the full hc stream for layer-level checks."""

    def __init__(self, config: DeepseekV41Config, weights: Weights, max_seq_len: int, *, rounding: bool = True) -> None:
        self.cfg = config
        self.w = weights
        self.max_seq_len = max_seq_len
        # the fp8 / fp4 cache rounding; off gives a smooth function for bisecting the
        # alloy path (rounding flips are chaotic at the indexer's exact-tie scores)
        self.rounding = rounding
        cfg = config
        self.freqs: dict[int, torch.Tensor] = {}
        for layer in range(cfg.n_layers):
            theta, orig = cfg.rope_params(layer)
            key = int(theta * 7 + orig)
            if key not in self.freqs:
                self.freqs[key] = precompute_freqs_cis(
                    cfg.rope_head_dim, max_seq_len, orig, theta, cfg.rope_factor, cfg.beta_fast, cfg.beta_slow,
                )
        self.window_kv = [torch.zeros(cfg.window_size, cfg.head_dim) for _ in range(cfg.n_layers)]
        self.compress_kv: dict[int, torch.Tensor] = {}
        self.index_k: dict[int, torch.Tensor] = {}
        self.comp_state: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for layer in cfg.kv_source_layers:
            ratio = cfg.compress_ratios[layer]
            self.compress_kv[layer] = torch.zeros(max_seq_len // ratio, cfg.head_dim)
            self.index_k[layer] = torch.zeros(max_seq_len // ratio, cfg.index_head_dim)
            if ratio > 1:
                self.comp_state[layer] = (
                    torch.zeros(ratio, cfg.head_dim),
                    torch.full((ratio, cfg.head_dim), -torch.inf),
                )
        self.hasher = EngramHasher(cfg.engram, max_seq_len) if cfg.engram is not None else None
        # per-forward shared state (sources write before consumers read)
        self.topk_idxs: torch.Tensor | None = None
        self.candidates: torch.Tensor | None = None
        self.debug_index: dict = {}

    def layer_freqs(self, layer: int) -> torch.Tensor:
        theta, orig = self.cfg.rope_params(layer)
        return self.freqs[int(theta * 7 + orig)]

    # --- attention -----------------------------------------------------------

    def compressor(self, layer: int, x: torch.Tensor, start_pos: int) -> torch.Tensor | None:
        cfg, w = self.cfg, self.w
        ratio = cfg.compress_ratios[layer]
        seqlen = x.shape[0]
        kv = x.float() @ w[f"blk.{layer}.attn_compressor_kv.weight"].float().T
        norm_w = w[f"blk.{layer}.attn_compressor_norm.weight"]
        if ratio == 1:
            return rms_norm(kv, norm_w, cfg.norm_eps)
        score = x.float() @ w[f"blk.{layer}.attn_compressor_gate.weight"].float().T
        kv_state, score_state = self.comp_state[layer]
        if start_pos == 0:
            should = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            if remainder:
                kv_state[:remainder] = kv[cutoff:]
                score_state[:remainder] = score[cutoff:]
                kv, score = kv[:cutoff], score[:cutoff]
            kv = kv.unflatten(0, (-1, ratio))
            score = score.unflatten(0, (-1, ratio))
            kv = (kv * score.softmax(dim=1)).sum(dim=1)
        else:
            should = (start_pos + 1) % ratio == 0
            slot = start_pos % ratio
            kv_state[slot] = kv[0]
            score_state[slot] = score[0]
            if should:
                kv = (kv_state * score_state.softmax(dim=0)).sum(dim=0, keepdim=True)
        if not should:
            return None
        return rms_norm(kv, norm_w, cfg.norm_eps)

    def indexer(self, layer: int, x: torch.Tensor, qr: torch.Tensor, latent: torch.Tensor | None,
                start_pos: int, offset: int) -> torch.Tensor:
        cfg, w = self.cfg, self.w
        ratio = cfg.compress_ratios[layer]
        seqlen = x.shape[0]
        rd = cfg.rope_head_dim
        end_pos = start_pos + seqlen
        freqs = self.layer_freqs(layer)
        src = cfg.kv_source_for(layer)
        if cfg.is_kv_source(layer) and latent is not None:
            f = freqs[: seqlen - seqlen % ratio : ratio] if start_pos == 0 else freqs[start_pos + 1 - ratio].unsqueeze(0)
            k = rms_norm(latent @ w[f"blk.{layer}.indexer.attn_k.weight"].float().T,
                         w[f"blk.{layer}.indexer.k_norm.weight"], cfg.norm_eps)
            k = torch.cat([k[..., :-rd], apply_rotary(k[..., -rd:], f)], dim=-1)
            if self.rounding:
                k = fp4_round(k, 32, scale_e4m3=False)
            self.index_k[layer][start_pos // ratio : start_pos // ratio + k.shape[0]] = k
        q = (qr @ w[f"blk.{layer}.indexer.attn_q_b.weight"].float().T).unflatten(-1, (cfg.index_n_heads, cfg.index_head_dim))
        q = torch.cat([q[..., :-rd], apply_rotary(q[..., -rd:], freqs[start_pos:end_pos])], dim=-1)
        if self.rounding:
            q = fp4_round(q, 32, scale_e4m3=False)
        index_k = self.index_k[src][: end_pos // ratio]
        weights = (x.float() @ w[f"blk.{layer}.indexer.proj.weight"].float().T) * (
            cfg.index_head_dim**-0.5 * cfg.index_n_heads**-0.5
        )
        score = torch.einsum("shd,td->sht", q, index_k)
        score = (score.relu() * weights.unsqueeze(-1)).sum(dim=1)  # [S, T']
        if start_pos == 0:
            compress_lens = (torch.arange(1, seqlen + 1) // ratio).unsqueeze(-1)
            score = score.masked_fill(torch.arange(seqlen // ratio) >= compress_lens, -torch.inf)
        else:
            compress_lens = end_pos // ratio
        if layer == cfg.candidate_source_layer:
            self.candidates = select_candidate_blocks(score, compress_lens, cfg.candidate_topk_blocks, cfg.candidate_block_size)
        elif cfg.uses_candidates(layer):
            score = score.masked_fill(~self.candidates, -torch.inf)
        topk = min(cfg.index_topk, end_pos // ratio)
        idxs = topk_lowest_index(score, topk).sort(dim=-1).values
        self.debug_index[layer] = (score, idxs, self.candidates)
        return torch.where(idxs < compress_lens, idxs + offset, -1).int()

    def attention(self, layer: int, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        cfg, w = self.cfg, self.w
        seqlen = x.shape[0]
        rd = cfg.rope_head_dim
        win = cfg.window_size
        freqs = self.layer_freqs(layer)
        f = freqs[start_pos : start_pos + seqlen]
        p = f"blk.{layer}."

        qr = rms_norm(x.float() @ w[p + "attn_q_a.weight"].float().T, w[p + "attn_q_a_norm.weight"], cfg.norm_eps)
        q = (qr @ w[p + "attn_q_b.weight"].float().T).unflatten(-1, (cfg.n_heads, cfg.head_dim))
        q = torch.cat([q[..., :-rd], apply_rotary(q[..., -rd:], f)], dim=-1)

        kv = rms_norm(x.float() @ w[p + "attn_kv.weight"].float().T, w[p + "attn_kv_a_norm.weight"], cfg.norm_eps)
        kv = torch.cat([kv[..., :-rd], apply_rotary(kv[..., -rd:], f)], dim=-1)
        if self.rounding:
            kv = fp8_round(kv, 32)
        ring = self.window_kv[layer]
        if start_pos == 0:
            if seqlen <= win:
                ring[:seqlen] = kv
            else:
                cutoff = seqlen % win
                tail = kv[-win:]
                ring[cutoff:win] = tail[: win - cutoff]
                ring[:cutoff] = tail[win - cutoff :]
            window_kv = kv
        else:
            ring[start_pos % win] = kv[0]
            window_kv = ring
        topk_idxs = window_topk_idxs(win, seqlen, start_pos)

        ratio = cfg.compress_ratios[layer]
        if ratio:
            compress_len = (start_pos + seqlen) // ratio
            latent = self.compressor(layer, x, start_pos) if cfg.is_kv_source(layer) else None
            if cfg.is_index_source(layer):
                if compress_len == 0:
                    idxs = torch.empty(seqlen, 0, dtype=torch.int32)
                else:
                    idxs = self.indexer(layer, x, qr, latent, start_pos, window_kv.shape[0])
                self.topk_idxs = idxs
            else:
                idxs = self.topk_idxs
            if latent is not None:
                fl = freqs[: seqlen - seqlen % ratio : ratio] if start_pos == 0 else freqs[start_pos + 1 - ratio].unsqueeze(0)
                latent = torch.cat([latent[..., :-rd], apply_rotary(latent[..., -rd:], fl)], dim=-1)
                if self.rounding:
                    latent = fp4_round(latent, 16, scale_e4m3=True)
                self.compress_kv[layer][start_pos // ratio : start_pos // ratio + latent.shape[0]] = latent
            src = cfg.kv_source_for(layer)
            kv_all = torch.cat([window_kv, self.compress_kv[src][:compress_len]], dim=0)
            topk_idxs = torch.cat([topk_idxs, idxs], dim=-1)
        else:
            kv_all = window_kv

        o = sparse_attn(q, kv_all, w[p + "attn_sinks.weight"], topk_idxs, cfg.head_dim**-0.5)
        o = torch.cat([o[..., :-rd], apply_rotary(o[..., -rd:], f, inverse=True)], dim=-1)
        # wo_a is block-diagonal over groups: group g projects its own heads only
        o = o.reshape(seqlen, cfg.o_groups, -1)
        wo_a = w[p + "attn_output_a.weight"].float().view(cfg.o_groups, cfg.o_lora_rank, -1)
        o = torch.einsum("sgd,grd->sgr", o, wo_a).flatten(1)
        return o @ w[p + "attn_output_b.weight"].float().T

    # --- moe ------------------------------------------------------------------

    def expert(self, x: torch.Tensor, gate_w: torch.Tensor, up_w: torch.Tensor, down_w: torch.Tensor) -> torch.Tensor:
        gate = x @ gate_w.float().T
        up = x @ up_w.float().T
        if self.cfg.swiglu_limit > 0:
            up = up.clamp(-self.cfg.swiglu_limit, self.cfg.swiglu_limit)
            gate = gate.clamp(max=self.cfg.swiglu_limit)
        return (F.silu(gate) * up) @ down_w.float().T

    def route(self, layer: int, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cfg, w = self.cfg, self.w
        scores = x.float() @ w[f"blk.{layer}.ffn_gate_inp.weight"].float().T
        if cfg.score_func == "softmax":
            scores = scores.softmax(-1)
        elif cfg.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        bias = w[f"blk.{layer}.exp_probs_b.bias"].float()
        indices = (scores + bias).topk(cfg.n_activated_experts, dim=-1).indices
        weights = scores.gather(1, indices)
        if cfg.norm_topk_prob and cfg.n_activated_experts > 1:
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        return weights * cfg.route_scale, indices

    def moe(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        w = self.w
        p = f"blk.{layer}."
        weights, indices = self.route(layer, x)
        y = torch.zeros_like(x, dtype=torch.float32)
        gate_e, up_e, down_e = (w[p + "ffn_gate_exps.weight"], w[p + "ffn_up_exps.weight"], w[p + "ffn_down_exps.weight"])
        for e in indices.unique().tolist():
            rows, slots = torch.where(indices == e)
            y[rows] += weights[rows, slots, None] * self.expert(x[rows], gate_e[e], up_e[e], down_e[e])
        y = y + self.expert(x, w[p + "ffn_gate_shexp.weight"], w[p + "ffn_up_shexp.weight"], w[p + "ffn_down_shexp.weight"])
        return y

    # --- hyper-connections ----------------------------------------------------

    def hc_mixes(self, layer: int, x: torch.Tensor, kind: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg, w = self.cfg, self.w
        p = f"blk.{layer}.hc_{kind}_"
        flat = x.flatten(1).float()
        rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + cfg.norm_eps)
        mixes = (flat @ w[p + "fn.weight"].float().T) * rsqrt
        return hc_split_sinkhorn(mixes, w[p + "scale.weight"].float(), w[p + "base.weight"].float(),
                                 cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps)

    @staticmethod
    def hc_pre(x: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        return (pre.unsqueeze(-1) * x.float()).sum(dim=1)

    @staticmethod
    def hc_post(y: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
        return post.unsqueeze(-1) * y.unsqueeze(1) + (comb.unsqueeze(-1) * residual.unsqueeze(1)).sum(dim=2)

    # --- engram ----------------------------------------------------------------

    def engram(self, layer: int, x: torch.Tensor, hash_ids: torch.Tensor) -> torch.Tensor:
        cfg, w = self.cfg, self.w
        p = f"blk.{layer}.engram_"
        rows = w[p + "embd.weight"][hash_ids].float().flatten(1)  # [T, cols*head_dim]
        kv = rows @ w[p + "wkv.weight"].float().T
        key, value = kv.split([cfg.hc_mult * cfg.dim, cfg.dim], dim=-1)
        key = key.unflatten(-1, (cfg.hc_mult, cfg.dim))
        weight = w[p + "q.weight"].float() * w[p + "k.weight"].float()
        h = x.float()
        rstd = torch.rsqrt(h.square().mean(-1) + cfg.norm_eps) * torch.rsqrt(key.square().mean(-1) + cfg.norm_eps)
        dot = (h * weight * key).sum(-1) * rstd * cfg.dim**-0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        return h + gate.unsqueeze(-1) * value.unsqueeze(1)

    # --- forward -----------------------------------------------------------------

    def forward(self, input_ids: np.ndarray, start_pos: int = 0, *, all_logits: bool = False) -> tuple[torch.Tensor, list[torch.Tensor]]:
        cfg, w = self.cfg, self.w
        ids = torch.as_tensor(np.asarray(input_ids, dtype=np.int64)).reshape(-1)
        hashes = None
        if self.hasher is not None:
            hashes = torch.as_tensor(self.hasher.hash_ids(ids.numpy(), start_pos))
        h = w["token_embd.weight"][ids].float()
        x = h.unsqueeze(1).repeat(1, cfg.hc_mult, 1)  # [T, hc, dim]
        pre_mix = torch.zeros(ids.shape[0], cfg.hc_mult)
        pre_mix[:, 0] = 1.0
        streams = []
        # per-layer intermediates for component-level checks of the alloy path
        self.pres: list[torch.Tensor] = []
        self.attn_outs: list[torch.Tensor] = []
        self.moe_outs: list[torch.Tensor] = []
        self.engram_outs: dict[int, torch.Tensor] = {}
        for layer in range(cfg.n_layers):
            if cfg.engram is not None and layer in cfg.engram.layer_ids:
                x = self.engram(layer, x, hashes[:, cfg.engram.layer_ids.index(layer), :])
                self.engram_outs[layer] = x
            residual = x
            attn_pre, attn_post, attn_comb = self.hc_mixes(layer, x, "attn")
            a = rms_norm(self.hc_pre(x, pre_mix), w[f"blk.{layer}.attn_norm.weight"], cfg.norm_eps)
            a = self.attention(layer, a, start_pos)
            self.attn_outs.append(a)
            x = self.hc_post(a, residual, attn_post, attn_comb)
            residual = x
            ffn_pre, ffn_post, ffn_comb = self.hc_mixes(layer, x, "ffn")
            f = rms_norm(self.hc_pre(x, attn_pre), w[f"blk.{layer}.ffn_norm.weight"], cfg.norm_eps)
            f = self.moe(layer, f)
            self.moe_outs.append(f)
            x = self.hc_post(f, residual, ffn_post, ffn_comb)
            pre_mix = ffn_pre
            self.pres.append(ffn_pre)
            streams.append(x)
        h = rms_norm(self.hc_pre(x, pre_mix), w["output_norm.weight"], cfg.norm_eps)
        if not all_logits:
            h = h[-1:]
        logits = h @ w["output.weight"].float().T
        return logits, streams
