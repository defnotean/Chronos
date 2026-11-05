#!/usr/bin/env python3
"""
Chronos-1T x GLM-5.3 — Runtime Tier Formats, FP8 Block Utilities & Portable Triton GEMV Kernels.

Tier formats (all packed along the input/`K` dimension, one fp16 scale per group):
  - `T1` (`t1_fp4`):   E2M1 (NVFP4 value set) nibbles, group 32  -> 4.500 bits/weight
  - `T2` (`t2_lm2`):   2-bit Lloyd-Max Gaussian codes, group 64   -> 2.250 bits/weight
  - `T3` (`t3_sign1`): 1-bit sign + absmean scale, group 128      -> 1.125 bits/weight
  - `W8` (`w8_fp8`):   stock 128x128 block-FP8 (e4m3 + fp32 `weight_scale_inv`) used for attention / dense /
                       shared experts

All Triton kernels decode codes with integer/bit arithmetic only (no `tl.float8*` casts), so the exact same
kernels run on Colab A100 (`sm_80`), H100 (`sm_90`), RTX PRO 6000 / RTX 50xx (`sm_120`) and DGX Spark GB10
(`sm_121`). Every kernel has a PyTorch reference (`*_torch`) used on CPU, for prefill, and in tests.
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F

try:  # Triton is optional (CPU / Windows-without-triton fall back to PyTorch).
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - environment dependent
    triton = None
    tl = None
    _HAS_TRITON = False

E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_THRESHOLDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
LM2_VALUES = (-1.510417, -0.452780, 0.452780, 1.510417)
LM2_THRESH = 0.981599

T1_GROUP, T2_GROUP, T3_GROUP = 32, 64, 128
TIER_ABSENT, TIER_T1, TIER_T2, TIER_T3, TIER_STAGE = 0, 1, 2, 3, 4
TIER_SHIFT = 24
SLOT_MASK = (1 << TIER_SHIFT) - 1

_LUT_CACHE: dict[tuple[str, str, torch.dtype], torch.Tensor] = {}


def triton_enabled(device: torch.device | str | None = None) -> bool:
    if not _HAS_TRITON or os.environ.get("CHRONOS_GLM53_DISABLE_TRITON", "0") == "1":
        return False
    if device is None:
        return torch.cuda.is_available()
    return torch.device(device).type == "cuda"


def _lut(kind: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (kind, str(device), dtype)
    t = _LUT_CACHE.get(key)
    if t is None:
        if kind == "e2m1":
            pos = torch.tensor(E2M1_VALUES, dtype=torch.float32)
            t = torch.cat([pos, -pos])
        elif kind == "lm2":
            t = torch.tensor(LM2_VALUES, dtype=torch.float32)
        else:
            raise KeyError(kind)
        t = t.to(device=device, dtype=dtype)
        _LUT_CACHE[key] = t
    return t


# =============================================================================================
# 128x128 block-FP8 helpers (stock GLM-5.3 checkpoint format)
# =============================================================================================
def dequant_fp8_block(
    w_fp8: torch.Tensor,
    scale_inv: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str | None = None,
    block: int = 128,
) -> torch.Tensor:
    """Dequantizes a 2D `float8_e4m3fn` weight with `(ceil(N/128), ceil(K/128))` fp32 `weight_scale_inv`.

    Handles shapes that are not multiples of 128 (e.g. `kv_a_proj_with_mqa` has `576` rows)."""
    device = torch.device(device) if device is not None else w_fp8.device
    n, k = w_fp8.shape
    if w_fp8.dtype == torch.uint8:
        w_fp8 = w_fp8.view(torch.float8_e4m3fn)
    w = w_fp8.to(device=device).to(torch.float32)
    s = scale_inv.to(device=device, dtype=torch.float32)
    if n % block == 0 and k % block == 0:
        out = (w.view(n // block, block, k // block, block) * s.view(n // block, 1, k // block, 1)).view(n, k)
    else:
        s_full = s.repeat_interleave(block, dim=0)[:n].repeat_interleave(block, dim=1)[:, :k]
        out = w * s_full
    return out.to(dtype)


def quantize_fp8_block(w: torch.Tensor, block: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of `dequant_fp8_block` (used to build synthetic GLM-5.3-layout test checkpoints)."""
    n, k = w.shape
    nb, kb = math.ceil(n / block), math.ceil(k / block)
    wp = torch.zeros((nb * block, kb * block), dtype=torch.float32, device=w.device)
    wp[:n, :k] = w.float()
    blk = wp.view(nb, block, kb, block)
    amax = blk.abs().amax(dim=(1, 3)).clamp_min(1e-12)
    scale_inv = amax / 448.0
    q = (blk / scale_inv.view(nb, 1, kb, 1)).clamp(-448.0, 448.0).view(nb * block, kb * block)[:n, :k]
    return q.to(torch.float8_e4m3fn).contiguous(), scale_inv.contiguous()


# =============================================================================================
# Tier packers / unpackers (PyTorch reference). Inputs may have arbitrary leading dims: (..., N, K).
# =============================================================================================
def _encode_e2m1(z: torch.Tensor) -> torch.Tensor:
    a = z.abs()
    idx = torch.zeros_like(a, dtype=torch.uint8)
    for ti, th in enumerate(E2M1_THRESHOLDS, start=1):
        idx = torch.where(a >= th, torch.full_like(idx, ti), idx)
    return idx | ((z < 0).to(torch.uint8) << 3)


def pack_t1_fp4(w: torch.Tensor, search: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """E2M1 nibbles (2/byte, low nibble = even k) + fp16 scale per 32 inputs. Optional 4-candidate MSE scale search."""
    *lead, n, k = w.shape
    assert k % T1_GROUP == 0
    g = w.reshape(-1, k // T1_GROUP, T1_GROUP).float()
    amax = g.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    lut = _lut("e2m1", w.device, torch.float32)
    best_s = amax / 6.0
    if search:
        best_err = None
        for f in (1.0, 0.92, 0.84, 0.76):
            s = (amax / 6.0 * f).to(torch.float16).float().clamp_min(1e-10)
            codes = _encode_e2m1(g / s)
            err = (lut[codes.long()] * s - g).pow(2).sum(dim=-1, keepdim=True)
            if best_err is None:
                best_err, best_s = err, s
            else:
                better = err < best_err
                best_err = torch.where(better, err, best_err)
                best_s = torch.where(better, s, best_s)
    s16 = best_s.to(torch.float16).clamp_min(torch.finfo(torch.float16).tiny)
    codes = _encode_e2m1(g / s16.float()).view(-1, k)
    q = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    return q.view(*lead, n, k // 2).contiguous(), s16.view(*lead, n, k // T1_GROUP).contiguous()


def unpack_t1_fp4(q: torch.Tensor, s: torch.Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    *lead, n, kp = q.shape
    k = kp * 2
    lut = _lut("e2m1", q.device, torch.float32)
    q2 = q.reshape(-1, kp)
    codes = torch.stack([q2 & 0x0F, (q2 >> 4) & 0x0F], dim=-1).view(-1, k // T1_GROUP, T1_GROUP)
    vals = lut[codes.long()] * s.reshape(-1, k // T1_GROUP, 1).float()
    return vals.view(*lead, n, k).to(dtype)


def pack_t2_lm2(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    *lead, n, k = w.shape
    assert k % (T2_GROUP) == 0
    g = w.reshape(-1, k // T2_GROUP, T2_GROUP).float()
    lut = _lut("lm2", w.device, torch.float32)
    sigma = g.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-10)
    for _ in range(2):
        z = g / sigma
        c = (z >= -LM2_THRESH).long() + (z >= 0).long() + (z >= LM2_THRESH).long()
        qv = lut[c]
        sigma = ((g * qv).sum(-1, keepdim=True) / qv.pow(2).sum(-1, keepdim=True).clamp_min(1e-12)).clamp_min(1e-10)
    s16 = sigma.to(torch.float16).clamp_min(torch.finfo(torch.float16).tiny)
    z = g / s16.float()
    c = ((z >= -LM2_THRESH).to(torch.uint8) + (z >= 0).to(torch.uint8) + (z >= LM2_THRESH).to(torch.uint8)).view(-1, k)
    q = (c[:, 0::4] | (c[:, 1::4] << 2) | (c[:, 2::4] << 4) | (c[:, 3::4] << 6)).to(torch.uint8)
    return q.view(*lead, n, k // 4).contiguous(), s16.view(*lead, n, k // T2_GROUP).contiguous()


def unpack_t2_lm2(q: torch.Tensor, s: torch.Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    *lead, n, kp = q.shape
    k = kp * 4
    lut = _lut("lm2", q.device, torch.float32)
    q2 = q.reshape(-1, kp)
    codes = torch.stack([(q2 >> sh) & 0x03 for sh in (0, 2, 4, 6)], dim=-1).view(-1, k // T2_GROUP, T2_GROUP)
    vals = lut[codes.long()] * s.reshape(-1, k // T2_GROUP, 1).float()
    return vals.view(*lead, n, k).to(dtype)


def pack_t3_sign1(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    *lead, n, k = w.shape
    assert k % T3_GROUP == 0
    g = w.reshape(-1, k // T3_GROUP, T3_GROUP).float()
    s16 = g.abs().mean(dim=-1, keepdim=True).to(torch.float16).clamp_min(torch.finfo(torch.float16).tiny)
    bits = (w.reshape(-1, k) >= 0).to(torch.uint8)
    q = torch.zeros((bits.shape[0], k // 8), dtype=torch.uint8, device=w.device)
    for b in range(8):
        q |= bits[:, b::8] << b
    return q.view(*lead, n, k // 8).contiguous(), s16.view(*lead, n, k // T3_GROUP).contiguous()


def unpack_t3_sign1(q: torch.Tensor, s: torch.Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    *lead, n, kp = q.shape
    k = kp * 8
    q2 = q.reshape(-1, kp)
    bits = torch.stack([(q2 >> b) & 0x01 for b in range(8)], dim=-1).view(-1, k // T3_GROUP, T3_GROUP)
    vals = (bits.float() * 2.0 - 1.0) * s.reshape(-1, k // T3_GROUP, 1).float()
    return vals.view(*lead, n, k).to(dtype)


PACKERS = {TIER_T1: pack_t1_fp4, TIER_T2: pack_t2_lm2, TIER_T3: pack_t3_sign1}
UNPACKERS = {TIER_T1: unpack_t1_fp4, TIER_T2: unpack_t2_lm2, TIER_T3: unpack_t3_sign1}
PACK_DIV = {TIER_T1: 2, TIER_T2: 4, TIER_T3: 8}
GROUPS = {TIER_T1: T1_GROUP, TIER_T2: T2_GROUP, TIER_T3: T3_GROUP}


def tier_shapes(tier: int, n: int, k: int) -> tuple[tuple[int, int], tuple[int, int]]:
    return (n, k // PACK_DIV[tier]), (n, k // GROUPS[tier])


# =============================================================================================
# Triton kernels
# =============================================================================================
if _HAS_TRITON:

    @triton.jit
    def _decode_e2m1(nib):
        mag = nib & 7
        e = (mag >> 1).to(tl.float32)
        m = (mag & 1).to(tl.float32)
        v = tl.where(e == 0.0, 0.5 * m, (1.0 + 0.5 * m) * tl.exp2(e - 1.0))
        return tl.where((nib & 8) != 0, -v, v)

    @triton.jit
    def _decode_lm2(c):
        mag = tl.where((c == 0) | (c == 3), 1.510417, 0.452780)
        return tl.where(c >= 2, mag, -mag)

    @triton.jit
    def _packed_tile_dot(
        q_base, s_base, x_row, rows, n_mask, K,
        BITS: tl.constexpr, PER: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_KB: tl.constexpr,
    ):
        """Dot of `BLOCK_N` packed rows with x. Every tier stores exactly 16 packed bytes per fp16 scale group
        (T1: 32 x 4-bit, T2: 64 x 2-bit, T3: 128 x 1-bit), so the scale index is simply `byte // 16`."""
        kp = K // PER
        kg = kp // 16
        j = tl.arange(0, PER)
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for b0 in range(0, kp, BLOCK_KB):
            offs_b = b0 + tl.arange(0, BLOCK_KB)
            byte = tl.load(q_base + rows[:, None] * kp + offs_b[None, :], mask=n_mask[:, None], other=0).to(tl.int32)
            x = tl.load(x_row + offs_b[:, None] * PER + j[None, :]).to(tl.float32)
            c = (byte[:, :, None] >> (j[None, None, :] * BITS)) & ((1 << BITS) - 1)
            if BITS == 4:
                v = _decode_e2m1(c)
            elif BITS == 2:
                v = _decode_lm2(c)
            else:
                v = c.to(tl.float32) * 2.0 - 1.0
            part = tl.sum(v * x[None, :, :], axis=2)
            sc = tl.load(s_base + rows[:, None] * kg + offs_b[None, :] // 16, mask=n_mask[:, None], other=0.0)
            acc += tl.sum(part * sc.to(tl.float32), axis=1)
        return acc

    @triton.jit
    def _glm53_tier_gemv_kernel(
        X_ptr, stride_xm,
        Ids_ptr, TierMap_ptr,
        Q1_ptr, S1_ptr, Q2_ptr, S2_ptr, Q3_ptr, S3_ptr, QS_ptr, SS_ptr,
        Out_ptr, stride_om,
        N, K,
        BLOCK_N: tl.constexpr, BLOCK_KB: tl.constexpr,
    ):
        """out[p, :] = W_{expert(p)} @ x[p, :]  for every (token, expert) pair `p`, dispatching on the expert's tier.

        `TierMap[e] = (tier << 24) | slot`; tier 0 = not resident (writes zeros), 4 = DMA staging buffer (T3 format).
        """
        p = tl.program_id(0)
        nb = tl.program_id(1)
        eid = tl.load(Ids_ptr + p)
        code = tl.load(TierMap_ptr + eid)
        tier = code >> 24
        slot = (code & 0xFFFFFF).to(tl.int64)
        offs_n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N
        rows = offs_n.to(tl.int64)
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        x_row = X_ptr + p.to(tl.int64) * stride_xm
        if tier == 1:
            acc = _packed_tile_dot(Q1_ptr + slot * N * (K // 2), S1_ptr + slot * N * (K // 32), x_row, rows, n_mask, K,
                                   4, 2, BLOCK_N, BLOCK_KB)
        elif tier == 2:
            acc = _packed_tile_dot(Q2_ptr + slot * N * (K // 4), S2_ptr + slot * N * (K // 64), x_row, rows, n_mask, K,
                                   2, 4, BLOCK_N, BLOCK_KB)
        elif tier == 3:
            acc = _packed_tile_dot(Q3_ptr + slot * N * (K // 8), S3_ptr + slot * N * (K // 128), x_row, rows, n_mask, K,
                                   1, 8, BLOCK_N, BLOCK_KB)
        elif tier == 4:
            acc = _packed_tile_dot(QS_ptr + slot * N * (K // 8), SS_ptr + slot * N * (K // 128), x_row, rows, n_mask, K,
                                   1, 8, BLOCK_N, BLOCK_KB)
        tl.store(Out_ptr + p.to(tl.int64) * stride_om + offs_n, acc, mask=n_mask)

    @triton.jit
    def _w8_block_gemv_kernel(
        X_ptr, stride_xm, W_ptr, S_ptr, Out_ptr, stride_om,
        N, K, KS,
        BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        """out[m, :] = dequant(W_fp8, 128x128 scale_inv) @ x[m, :] with manual e4m3 decoding (sm_80 safe)."""
        m = tl.program_id(0)
        nb = tl.program_id(1)
        offs_n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N
        rows = offs_n.to(tl.int64)
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        x_row = X_ptr + m.to(tl.int64) * stride_xm
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K
            x = tl.load(x_row + offs_k, mask=k_mask, other=0.0).to(tl.float32)
            b = tl.load(W_ptr + rows[:, None] * K + offs_k[None, :], mask=n_mask[:, None] & k_mask[None, :], other=0).to(tl.int32)
            e = ((b >> 3) & 0xF).to(tl.float32)
            mt = (b & 7).to(tl.float32)
            v = tl.where(e == 0.0, mt * 0.001953125, (8.0 + mt) * tl.exp2(e - 10.0))
            v = tl.where((b & 0x80) != 0, -v, v)
            sc = tl.load(S_ptr + (rows[:, None] // 128) * KS + (offs_k[None, :] // 128), mask=n_mask[:, None] & k_mask[None, :], other=0.0)
            acc += tl.sum(v * sc * x[None, :], axis=1)
        tl.store(Out_ptr + m.to(tl.int64) * stride_om + offs_n, acc, mask=n_mask)


def tier_gemv(
    x: torch.Tensor,
    ids: torch.Tensor,
    tier_map: torch.Tensor,
    bufs: dict[int, tuple[torch.Tensor, torch.Tensor]],
    n_out: int,
    out: torch.Tensor | None = None,
    block_n: int = 16,
    block_kb: int = 64,
) -> torch.Tensor:
    """Pairwise tier-dispatched GEMV: `out[p] = W[ids[p]] @ x[p]` (CUDA-graph safe: no host syncs).

    `block_kb` is in packed bytes; it must divide `K // 8` (T3 bytes per row)."""
    p, k = x.shape
    while (k // 8) % block_kb:
        block_kb //= 2
    if out is None:
        out = torch.empty((p, n_out), dtype=torch.float32, device=x.device)
    dummy_q = torch.empty((1,), dtype=torch.uint8, device=x.device)
    dummy_s = torch.empty((1,), dtype=torch.float16, device=x.device)
    q1, s1 = bufs.get(TIER_T1, (dummy_q, dummy_s))
    q2, s2 = bufs.get(TIER_T2, (dummy_q, dummy_s))
    q3, s3 = bufs.get(TIER_T3, (dummy_q, dummy_s))
    qs, ss = bufs.get(TIER_STAGE, (dummy_q, dummy_s))
    xc = x.contiguous()
    grid = (p, triton.cdiv(n_out, block_n))
    _glm53_tier_gemv_kernel[grid](
        xc, xc.stride(0), ids, tier_map,
        q1, s1, q2, s2, q3, s3, qs, ss,
        out, out.stride(0),
        n_out, k,
        BLOCK_N=block_n, BLOCK_KB=block_kb, num_warps=4,
    )
    return out


def tier_gemv_torch(
    x: torch.Tensor,
    ids: torch.Tensor,
    tier_map: torch.Tensor,
    bufs: dict[int, tuple[torch.Tensor, torch.Tensor]],
    n_out: int,
) -> torch.Tensor:
    """Reference implementation of `tier_gemv` (groups pairs by expert, dequantizes each expert once)."""
    out = torch.zeros((x.shape[0], n_out), dtype=torch.float32, device=x.device)
    ids_l = ids.tolist()
    codes = tier_map[ids.long()].tolist()
    by_code: dict[int, list[int]] = {}
    for pi, c in enumerate(codes):
        by_code.setdefault(int(c), []).append(pi)
    for c, plist in by_code.items():
        tier, slot = c >> TIER_SHIFT, c & SLOT_MASK
        if tier == TIER_ABSENT:
            continue
        fmt = TIER_T3 if tier == TIER_STAGE else tier
        q, s = bufs[tier]
        w = UNPACKERS[fmt](q[slot], s[slot], dtype=torch.float32)
        idx = torch.tensor(plist, device=x.device, dtype=torch.long)
        out[idx] = x[idx].float() @ w.T
    del ids_l
    return out


def fp8_linear(
    x: torch.Tensor,
    w_fp8: torch.Tensor,
    scale_inv: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    gemv_max_rows: int = 4,
) -> torch.Tensor:
    """`y = x @ dequant(W)^T` for stock 128x128 block-FP8 weights (Triton GEMV for decode, dequant+GEMM otherwise)."""
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    n, k = w_fp8.shape
    if x2.shape[0] <= gemv_max_rows and triton_enabled(x2.device):
        out = torch.empty((x2.shape[0], n), dtype=torch.float32, device=x2.device)
        w_u8 = w_fp8.view(torch.uint8) if w_fp8.dtype != torch.uint8 else w_fp8
        xc = x2.contiguous()
        grid = (x2.shape[0], triton.cdiv(n, 64))
        _w8_block_gemv_kernel[grid](
            xc, xc.stride(0), w_u8, scale_inv, out, out.stride(0),
            n, k, scale_inv.shape[1],
            BLOCK_N=64, BLOCK_K=128, num_warps=4,
        )
        return out.to(out_dtype).view(*lead, n)
    w = dequant_fp8_block(w_fp8, scale_inv, dtype=x2.dtype if x2.dtype != torch.float32 else torch.float32)
    return F.linear(x2, w).to(out_dtype).view(*lead, n)
