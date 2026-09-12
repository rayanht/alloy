"""numpy ports of llama.cpp's reference Q4_K / Q6_K quantizers (`ggml-quants.c`:
`quantize_row_q4_K_ref`, `quantize_row_q6_K_ref`), vectorized over superblocks.
gguf-py only dequantizes the K-quants; this is the missing half for producing GGUF
payloads in-process (test models, repacking)."""

from __future__ import annotations

import gguf
import numpy as np

QK_K = 256


def nearest_int(x: np.ndarray) -> np.ndarray:
    return np.rint(x).astype(np.int32)


def make_qkx2_quants(x: np.ndarray, weights: np.ndarray, nmax: int, rmin: float, rdelta: float,
                     nstep: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted min/scale search for asymmetric 4-bit sub-blocks.
    x, weights: [B, n]. Returns (scale [B], the_min [B], L [B, n] uint8)."""
    B, n = x.shape
    lo = np.minimum(x.min(axis=1), 0.0)
    hi = x.max(axis=1)
    flat = hi == lo
    span = np.where(flat, 1.0, hi - lo)
    iscale = nmax / span
    scale = 1.0 / iscale
    L = np.clip(nearest_int(iscale[:, None] * (x - lo[:, None])), 0, nmax)
    diff = scale[:, None] * L + lo[:, None] - x
    best_mad = (weights * diff * diff).sum(axis=1)
    sum_w = weights.sum(axis=1)
    sum_x = (weights * x).sum(axis=1)
    cur_min = lo.copy()
    for step in range(nstep):
        isc = (rmin + rdelta * step + nmax) / span
        Laux = np.clip(nearest_int(isc[:, None] * (x - lo[:, None])), 0, nmax)
        lf = Laux.astype(np.float64)
        sum_l = (weights * lf).sum(axis=1)
        sum_l2 = (weights * lf * lf).sum(axis=1)
        sum_xl = (weights * lf * x).sum(axis=1)
        D = sum_w * sum_l2 - sum_l * sum_l
        ok = D > 0
        Dsafe = np.where(ok, D, 1.0)
        this_scale = (sum_w * sum_xl - sum_x * sum_l) / Dsafe
        this_min = (sum_l2 * sum_x - sum_l * sum_xl) / Dsafe
        pos = this_min > 0
        this_min = np.where(pos, 0.0, this_min)
        this_scale = np.where(pos, sum_xl / np.where(sum_l2 > 0, sum_l2, 1.0), this_scale)
        d2 = this_scale[:, None] * lf + this_min[:, None] - x
        mad = (weights * d2 * d2).sum(axis=1)
        better = ok & (mad < best_mad)
        L = np.where(better[:, None], Laux, L)
        best_mad = np.where(better, mad, best_mad)
        scale = np.where(better, this_scale, scale)
        cur_min = np.where(better, this_min, cur_min)
    scale = np.where(flat, 0.0, scale)
    L = np.where(flat[:, None], 0, L)
    cur_min = np.where(flat, lo, cur_min)
    return scale.astype(np.float32), (-cur_min).astype(np.float32), L.astype(np.uint8)


def quantize_q4_k(data: np.ndarray) -> np.ndarray:
    """[..., K] fp32 -> [..., K/256*144] uint8 native Q4_K superblocks."""
    K = data.shape[-1]
    if K % QK_K:
        raise ValueError(f"Q4_K needs a multiple of {QK_K} columns, got {K}")
    x = np.ascontiguousarray(data, dtype=np.float32).reshape(-1, QK_K)
    B = x.shape[0]
    sub = x.reshape(B * 8, 32).astype(np.float64)
    av = np.sqrt((sub * sub).sum(axis=1) / 32)
    weights = av[:, None] + np.abs(sub)
    scales, mins, _ = make_qkx2_quants(sub, weights, 15, -1.0, 0.1, 20)
    scales = scales.reshape(B, 8)
    mins = mins.reshape(B, 8)
    max_scale = scales.max(axis=1)
    max_min = mins.max(axis=1)
    inv_scale = np.where(max_scale > 0, 63.0 / np.where(max_scale > 0, max_scale, 1.0), 0.0)
    inv_min = np.where(max_min > 0, 63.0 / np.where(max_min > 0, max_min, 1.0), 0.0)
    ls = np.minimum(nearest_int(inv_scale[:, None] * scales), 63).astype(np.uint8)
    lm = np.minimum(nearest_int(inv_min[:, None] * mins), 63).astype(np.uint8)
    packed = np.zeros((B, 12), dtype=np.uint8)
    for j in range(8):
        if j < 4:
            packed[:, j] = ls[:, j]
            packed[:, j + 4] = lm[:, j]
        else:
            packed[:, j + 4] = (ls[:, j] & 0xF) | ((lm[:, j] & 0xF) << 4)
            packed[:, j - 4] |= (ls[:, j] >> 4) << 6
            packed[:, j] |= (lm[:, j] >> 4) << 6
    d = (max_scale / 63.0).astype(np.float16)
    dmin = (max_min / 63.0).astype(np.float16)
    # requantize against the 6-bit-rounded scales, exactly as the decoder sees them
    sc = np.where(np.arange(8) < 4, packed[:, :8] & 63, 0)
    mn = np.where(np.arange(8) < 4, packed[:, 4:12] & 63, 0)
    for j in range(4, 8):
        sc[:, j] = (packed[:, j + 4] & 0xF) | ((packed[:, j - 4] >> 6) << 4)
        mn[:, j] = (packed[:, j + 4] >> 4) | ((packed[:, j] >> 6) << 4)
    dj = d.astype(np.float32)[:, None] * sc.astype(np.float32)  # [B, 8]
    dm = dmin.astype(np.float32)[:, None] * mn.astype(np.float32)
    xs = x.reshape(B, 8, 32)
    safe = np.where(dj != 0, dj, 1.0)
    L = np.clip(nearest_int((xs + dm[:, :, None]) / safe[:, :, None]), 0, 15)
    L = np.where((dj != 0)[:, :, None], L, 0).astype(np.uint8).reshape(B, 4, 2, 32)
    qs = L[:, :, 0, :] | (L[:, :, 1, :] << 4)  # [B, 4, 32]
    out = np.empty((B, 144), dtype=np.uint8)
    out[:, 0:2] = d.view(np.uint8).reshape(B, 2)
    out[:, 2:4] = dmin.view(np.uint8).reshape(B, 2)
    out[:, 4:16] = packed
    out[:, 16:] = qs.reshape(B, 128)
    return out.reshape(*data.shape[:-1], K // QK_K * 144)


def make_qx_quants(x: np.ndarray, nmax: int) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric scale search with x² weighting (rmse_type=1). x: [B, n].
    Returns (scale [B], L [B, n] int in [0, 2*nmax))."""
    B, n = x.shape
    amax_idx = np.abs(x).argmax(axis=1)
    mx = x[np.arange(B), amax_idx]
    amax = np.abs(mx)
    zero = amax < 1e-15
    mx_safe = np.where(zero, 1.0, mx)
    w = x * x

    def trial(iscale):
        l = np.clip(nearest_int(iscale[:, None] * x), -nmax, nmax - 1)
        lf = l.astype(np.float64)
        sumlx = (w * x * lf).sum(axis=1)
        suml2 = (w * lf * lf).sum(axis=1)
        return l, sumlx, suml2

    iscale = -nmax / mx_safe
    L, sumlx, suml2 = trial(iscale)
    scale = np.where(suml2 > 0, sumlx / np.where(suml2 > 0, suml2, 1.0), 0.0)
    best = scale * sumlx
    for step in range(-9, 10):
        if step == 0:
            continue
        isc = -(nmax + 0.1 * step) / mx_safe
        l, slx, sl2 = trial(isc)
        better = (sl2 > 0) & (slx * slx > best * sl2)
        L = np.where(better[:, None], l, L)
        scale = np.where(better, slx / np.where(sl2 > 0, sl2, 1.0), scale)
        best = np.where(better, scale * slx, best)
    scale = np.where(zero, 0.0, scale)
    L = np.where(zero[:, None], 0, L) + nmax
    return scale.astype(np.float32), L


def quantize_q6_k(data: np.ndarray) -> np.ndarray:
    """[..., K] fp32 -> [..., K/256*210] uint8 native Q6_K superblocks."""
    K = data.shape[-1]
    if K % QK_K:
        raise ValueError(f"Q6_K needs a multiple of {QK_K} columns, got {K}")
    x = np.ascontiguousarray(data, dtype=np.float32).reshape(-1, QK_K)
    B = x.shape[0]
    sub = x.reshape(B * 16, 16).astype(np.float64)
    scales, _ = make_qx_quants(sub, 32)
    scales = scales.reshape(B, 16)
    amax_idx = np.abs(scales).argmax(axis=1)
    max_scale = scales[np.arange(B), amax_idx]
    dead = max_scale == 0
    iscale = -128.0 / np.where(dead, 1.0, max_scale)
    d = (1.0 / iscale).astype(np.float16)
    d = np.where(dead, np.float16(0), d)
    q_scales = np.clip(nearest_int(iscale[:, None] * scales), -128, 127).astype(np.int8)
    q_scales = np.where(dead[:, None], 0, q_scales).astype(np.int8)
    dj = d.astype(np.float32)[:, None] * q_scales.astype(np.float32)  # [B, 16]
    xs = x.reshape(B, 16, 16)
    safe = np.where(dj != 0, dj, 1.0)
    L = np.clip(nearest_int(xs / safe[:, :, None]), -32, 31) + 32
    L = np.where((dj != 0)[:, :, None], L, 32).astype(np.uint8).reshape(B, 2, 4, 32)  # [B, half, quarter(32), l]
    q = L & 0xF
    h = L >> 4
    ql = np.empty((B, 2, 64), dtype=np.uint8)
    ql[:, :, :32] = q[:, :, 0, :] | (q[:, :, 2, :] << 4)
    ql[:, :, 32:] = q[:, :, 1, :] | (q[:, :, 3, :] << 4)
    qh = h[:, :, 0, :] | (h[:, :, 1, :] << 2) | (h[:, :, 2, :] << 4) | (h[:, :, 3, :] << 6)  # [B, 2, 32]
    out = np.empty((B, 210), dtype=np.uint8)
    out[:, 0:128] = ql.reshape(B, 128)
    out[:, 128:192] = qh.reshape(B, 64)
    out[:, 192:208] = q_scales.view(np.uint8)
    out[:, 208:210] = d.view(np.uint8).reshape(B, 2)
    return out.reshape(*data.shape[:-1], K // QK_K * 210)


def quantize(data: np.ndarray, qtype: gguf.GGMLQuantizationType) -> np.ndarray:
    if qtype == gguf.GGMLQuantizationType.Q4_K:
        return quantize_q4_k(data)
    if qtype == gguf.GGMLQuantizationType.Q6_K:
        return quantize_q6_k(data)
    return gguf.quants.quantize(data, qtype)
