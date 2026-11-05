"""
Fused Triton Backbone & Tensor-Core MLA Decode Kernels for DeepSeek-V4.1-Flash-1017 + Chronos-1T.

Provides drop-in Triton accelerations for the standalone DeepSeek-V4.1-Flash-1017 `model.py`:
  1. `_triton_rmsnorm_kernel`: Single-pass FP32 variance + rsqrt + BF16 weight multiply.
  2. `_triton_hc_gemv_sqsum_kernel` + `_triton_hc_split_sinkhorn_m4_kernel`:
     Fuses `x.flatten(2).float()`, `rsqrt(mean(x^2)+eps)`, `F.linear(x, hc_fn) * rsqrt`, and 20-iter 4x4 Sinkhorn.
  3. `_triton_hc_pre_rmsnorm_m4_kernel` & `_triton_hc_post_m4_kernel`:
     Fuses 4-stream HyperConnection `hc_pre` + `RMSNorm` into 1 kernel launch, and `hc_post` into 1 kernel launch.
  4. `_triton_act_quant_inplace_32_kernel` & `_triton_fp4_act_quant_inplace_kernel`: Single-kernel in-place quant/dequant.
  5. `_triton_apply_rotary_inplace_kernel`: Zero-copy view RoPE in-place rotation.
  6. Tensor-Core Fused MLA Decode Pipeline (`_fast_attn_forward`):
     - Zero-extra-VRAM fused `[wq_a; wkv]` projection view (`1792 x 5120`).
     - `_triton_mla_qnorm_kvprep_decode_kernel[(2,)]`: Fuses `q_norm` + `kv_norm` + `kv` RoPE + `kv` FP8 `act_quant` + `window_kv_cache` ring-buffer write.
     - `_triton_mla_qrope_scores_tc_kernel`: Uses BF16 Tensor Cores (`tl.dot` across all 64 heads at once) with fused in-register `q` RoPE.
     - `_triton_mla_decode_softmax_kernel`: Exact `bfloat16`-rounded `exp_s` + `attn_sink` normalization.
     - `_triton_mla_decode_out_rope_quant_tc_kernel`: Uses BF16 Tensor Cores (`tl.dot` across all 64 heads) with fused inverse RoPE and in-place block-32 FP8 `act_quant`.
"""
from __future__ import annotations

from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _quantize_fp8_block32_in_reg(x_f32):
    """In-register exact block-32 UE8M0 scale + FP8-E4M3 round-trip dequantization.
    `x_f32` must have shape [..., 32]. Returns `bfloat16` tensor of same shape.
    """
    amax = tl.max(tl.abs(x_f32), axis=-1, keep_dims=True)
    amax = tl.maximum(amax, 1e-4)
    scaled_max = amax * (1.0 / 448.0)
    bits = scaled_max.to(tl.int32, bitcast=True)
    exp_x = (bits >> 23) & 0xFF
    man_bits = bits & 0x7FFFFF
    log2_ceil = exp_x - 127 + tl.where(man_bits != 0, 1, 0)
    s_fp32 = ((log2_ceil + 127) << 23).to(tl.float32, bitcast=True)
    inv_s = 1.0 / s_fp32
    v = tl.minimum(tl.maximum(x_f32 * inv_s, -448.0), 448.0)
    v_abs = tl.abs(v)
    v_sign = tl.where(v < 0.0, -1.0, 1.0)
    v_bits = v_abs.to(tl.uint32, bitcast=True)
    v_exp = (v_bits >> 23) & 0xFF
    is_sub = v_exp < 121
    round_bias = ((v_bits >> 20) & 1) + 0x7FFFF
    v_norm = ((v_bits + round_bias) & 0xFFF00000).to(tl.float32, bitcast=True)
    v_sub = tl.floor(v_abs * 512.0 + 0.5) * (1.0 / 512.0)
    v_q = tl.where(v_abs == 0.0, 0.0, tl.where(is_sub, v_sub, tl.minimum(v_norm, 448.0))) * v_sign
    return (v_q * s_fp32).to(tl.bfloat16)


# -----------------------------------------------------------------------------
# 1. Fused Triton RMSNorm Kernel
# -----------------------------------------------------------------------------
@triton.jit
def _triton_rmsnorm_kernel(
    X_ptr,
    W_ptr,
    Y_ptr,
    eps,
    N_COLS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N_COLS
    x = tl.load(X_ptr + row * N_COLS + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) * (1.0 / N_COLS)
    rstd = tl.rsqrt(var + eps)
    y = (x * rstd * w).to(tl.bfloat16)
    tl.store(Y_ptr + row * N_COLS + offs, y, mask=mask)


def fast_rmsnorm_forward(self: Any, x: torch.Tensor) -> torch.Tensor:
    if x.is_cuda and x.dtype == torch.bfloat16 and x.is_contiguous():
        n_cols = self.dim
        n_rows = x.numel() // n_cols
        y = torch.empty_like(x)
        block_n = triton.next_power_of_2(n_cols)
        num_warps = 4 if block_n <= 1024 else (8 if block_n <= 4096 else 16)
        _triton_rmsnorm_kernel[(n_rows,)](
            x, self.weight, y, self.eps, N_COLS=n_cols, BLOCK_N=block_n, num_warps=num_warps
        )
        return y
    dtype = x.dtype
    xf = x.float()
    var = xf.square().mean(-1, keepdim=True)
    xf = xf * torch.rsqrt(var + self.eps)
    return (self.weight * xf).to(dtype)


# -----------------------------------------------------------------------------
# 2. Fused Triton HyperConnection GEMV + SqSum + 20-Iteration Sinkhorn
# -----------------------------------------------------------------------------
@triton.jit
def _triton_hc_gemv_sqsum_m4_kernel(
    X_ptr,
    W_ptr,
    Mixes_raw_ptr,
    Rsqrt_ptr,
    eps,
):
    """Grid: (n_rows, 24).
    `HC_DIM = 20480 = 20 * 1024` exact (zero mask, zero register spill).
    """
    row = tl.program_id(0)
    out_c = tl.program_id(1)
    x_base = X_ptr + row * 20480
    w_base = W_ptr + out_c * 20480

    acc_dot = tl.zeros([1024], dtype=tl.float32)
    acc_sq = tl.zeros([1024], dtype=tl.float32)
    offs_base = tl.arange(0, 1024)
    for k0 in tl.static_range(0, 20480, 1024):
        offs = k0 + offs_base
        xv = tl.load(x_base + offs).to(tl.float32)
        wv = tl.load(w_base + offs).to(tl.float32)
        acc_dot += xv * wv
        if out_c == 0:
            acc_sq += xv * xv

    dot_val = tl.sum(acc_dot, axis=0)
    tl.store(Mixes_raw_ptr + row * 24 + out_c, dot_val)
    if out_c == 0:
        mean_sq = tl.sum(acc_sq, axis=0) * (1.0 / 20480.0)
        tl.store(Rsqrt_ptr + row, tl.rsqrt(mean_sq + eps))


@triton.jit
def _triton_hc_split_sinkhorn_m4_kernel(
    Mixes_ptr,
    Rsqrt_ptr,
    Scale_ptr,
    Base_ptr,
    Pre_ptr,
    Post_ptr,
    Comb_ptr,
    eps,
    HAS_RSQRT: tl.constexpr,
    SINKHORN_ITERS: tl.constexpr,
):
    row = tl.program_id(0)
    m_base = Mixes_ptr + row * 24
    rsqrt = tl.load(Rsqrt_ptr + row).to(tl.float32) if HAS_RSQRT else 1.0

    s0 = tl.load(Scale_ptr + 0).to(tl.float32)
    s1 = tl.load(Scale_ptr + 1).to(tl.float32)
    s2 = tl.load(Scale_ptr + 2).to(tl.float32)

    offs4 = tl.arange(0, 4)
    m_pre = tl.load(m_base + offs4).to(tl.float32) * rsqrt
    b_pre = tl.load(Base_ptr + offs4).to(tl.float32)
    pre = tl.sigmoid(m_pre * s0 + b_pre) + eps
    tl.store(Pre_ptr + row * 4 + offs4, pre)

    m_post = tl.load(m_base + 4 + offs4).to(tl.float32) * rsqrt
    b_post = tl.load(Base_ptr + 4 + offs4).to(tl.float32)
    post = 2.0 * tl.sigmoid(m_post * s1 + b_post)
    tl.store(Post_ptr + row * 4 + offs4, post)

    r_idx = tl.arange(0, 4)[:, None]
    c_idx = tl.arange(0, 4)[None, :]
    offs16 = 8 + r_idx * 4 + c_idx
    m_comb = tl.load(m_base + offs16).to(tl.float32) * rsqrt
    b_comb = tl.load(Base_ptr + offs16).to(tl.float32)
    comb = m_comb * s2 + b_comb

    row_max = tl.max(comb, axis=1)[:, None]
    comb = tl.exp(comb - row_max)
    comb = comb / tl.sum(comb, axis=1)[:, None] + eps
    comb = comb / (tl.sum(comb, axis=0)[None, :] + eps)

    for _ in tl.static_range(SINKHORN_ITERS - 1):
        comb = comb / (tl.sum(comb, axis=1)[:, None] + eps)
        comb = comb / (tl.sum(comb, axis=0)[None, :] + eps)

    tl.store(Comb_ptr + row * 16 + r_idx * 4 + c_idx, comb)


_hc_scratch_bufs: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}


def _get_hc_scratch(dev: torch.device, n_rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    pair = _hc_scratch_bufs.get(dev)
    if pair is None or pair[0].size(0) < n_rows or pair[0].is_inference():
        with torch.inference_mode(False):
            cap = max(n_rows, 64)
            pair = (
                torch.empty((cap, 24), dtype=torch.float32, device=dev),
                torch.empty((cap,), dtype=torch.float32, device=dev),
            )
            _hc_scratch_bufs[dev] = pair
    return pair


def fast_hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    orig_prefix = mixes.shape[:-1]
    n_rows = mixes.numel() // 24
    m_contig = mixes.contiguous()
    pre = torch.empty((*orig_prefix, 4), dtype=torch.float32, device=mixes.device)
    post = torch.empty((*orig_prefix, 4), dtype=torch.float32, device=mixes.device)
    comb = torch.empty((*orig_prefix, 4, 4), dtype=torch.float32, device=mixes.device)
    _triton_hc_split_sinkhorn_m4_kernel[(n_rows,)](
        m_contig,
        m_contig,
        hc_scale,
        hc_base,
        pre,
        post,
        comb,
        eps,
        HAS_RSQRT=False,
        SINKHORN_ITERS=sinkhorn_iters,
        num_warps=1,
    )
    return pre, post, comb


def fast_block_hc_mixes(
    self: Any, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor
):
    if x.is_cuda and x.dtype == torch.bfloat16 and x.is_contiguous() and self.hc_mult == 4:
        orig_prefix = x.shape[:-2]
        n_rows = x.numel() // 20480
        mixes_raw, rsqrt_buf = _get_hc_scratch(x.device, n_rows)
        _triton_hc_gemv_sqsum_m4_kernel[(n_rows, 24)](
            x,
            hc_fn,
            mixes_raw,
            rsqrt_buf,
            self.norm_eps,
            num_warps=4,
        )
        pre = torch.empty((*orig_prefix, 4), dtype=torch.float32, device=x.device)
        post = torch.empty((*orig_prefix, 4), dtype=torch.float32, device=x.device)
        comb = torch.empty((*orig_prefix, 4, 4), dtype=torch.float32, device=x.device)
        _triton_hc_split_sinkhorn_m4_kernel[(n_rows,)](
            mixes_raw,
            rsqrt_buf,
            hc_scale,
            hc_base,
            pre,
            post,
            comb,
            self.hc_eps,
            HAS_RSQRT=True,
            SINKHORN_ITERS=self.hc_sinkhorn_iters,
            num_warps=1,
        )
        return pre, post, comb
    xf = x.flatten(2).float()
    rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.norm_eps)
    mixes = F.linear(xf, hc_fn) * rsqrt
    return fast_hc_split_sinkhorn(mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps)


# -----------------------------------------------------------------------------
# 3. Fused Triton HC Pre (+ RMSNorm) and HC Post Kernels (hc_mult=4, D=5120)
# -----------------------------------------------------------------------------
@triton.jit
def _triton_hc_pre_rmsnorm_m4_kernel(
    X_ptr,
    PreMix_ptr,
    NormW_ptr,
    Y_ptr,
    eps,
):
    """Fuses `hc_pre(x, pre_mix)` and `RMSNorm(y, norm_w, eps)` in 5 exact tiles of 1024 (`5 * 1024 = 5120`),
    with zero masking and zero register spilling!
    Grid: (n_toks,)
    """
    tok = tl.program_id(0)
    p0 = tl.load(PreMix_ptr + tok * 4 + 0).to(tl.float32)
    p1 = tl.load(PreMix_ptr + tok * 4 + 1).to(tl.float32)
    p2 = tl.load(PreMix_ptr + tok * 4 + 2).to(tl.float32)
    p3 = tl.load(PreMix_ptr + tok * 4 + 3).to(tl.float32)
    x_base = X_ptr + tok * 20480
    y_base = Y_ptr + tok * 5120

    acc_var = tl.zeros([1024], dtype=tl.float32)
    offs_base = tl.arange(0, 1024)
    for d0 in tl.static_range(0, 5120, 1024):
        offs = d0 + offs_base
        x0 = tl.load(x_base + 0 * 5120 + offs).to(tl.float32)
        x1 = tl.load(x_base + 1 * 5120 + offs).to(tl.float32)
        x2 = tl.load(x_base + 2 * 5120 + offs).to(tl.float32)
        x3 = tl.load(x_base + 3 * 5120 + offs).to(tl.float32)
        y_bf16 = (p0 * x0 + p1 * x1 + p2 * x2 + p3 * x3).to(tl.bfloat16)
        tl.store(y_base + offs, y_bf16)
        yf = y_bf16.to(tl.float32)
        acc_var += yf * yf

    var = tl.sum(acc_var, axis=0) * (1.0 / 5120.0)
    rstd = tl.rsqrt(var + eps)

    for d0 in tl.static_range(0, 5120, 1024):
        offs = d0 + offs_base
        yf = tl.load(y_base + offs).to(tl.float32)
        nw = tl.load(NormW_ptr + offs).to(tl.float32)
        tl.store(y_base + offs, (yf * rstd * nw).to(tl.bfloat16))


@triton.jit
def _triton_hc_pre_m4_kernel(
    X_ptr,
    PreMix_ptr,
    Y_ptr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    tok = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    p0 = tl.load(PreMix_ptr + tok * 4 + 0).to(tl.float32)
    p1 = tl.load(PreMix_ptr + tok * 4 + 1).to(tl.float32)
    p2 = tl.load(PreMix_ptr + tok * 4 + 2).to(tl.float32)
    p3 = tl.load(PreMix_ptr + tok * 4 + 3).to(tl.float32)
    x_base = X_ptr + tok * (4 * D) + offs_d
    x0 = tl.load(x_base + 0 * D).to(tl.float32)
    x1 = tl.load(x_base + 1 * D).to(tl.float32)
    x2 = tl.load(x_base + 2 * D).to(tl.float32)
    x3 = tl.load(x_base + 3 * D).to(tl.float32)
    y = (p0 * x0 + p1 * x1 + p2 * x2 + p3 * x3).to(tl.bfloat16)
    tl.store(Y_ptr + tok * D + offs_d, y)


@triton.jit
def _triton_hc_post_m4_kernel(
    X_ptr,
    Res_ptr,
    Post_ptr,
    Comb_ptr,
    Y_ptr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    tok = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    xv = tl.load(X_ptr + tok * D + offs_d).to(tl.float32)
    r_base = Res_ptr + tok * (4 * D) + offs_d
    r0 = tl.load(r_base + 0 * D).to(tl.float32)
    r1 = tl.load(r_base + 1 * D).to(tl.float32)
    r2 = tl.load(r_base + 2 * D).to(tl.float32)
    r3 = tl.load(r_base + 3 * D).to(tl.float32)

    y_base = Y_ptr + tok * (4 * D) + offs_d
    for dst_c in tl.static_range(4):
        post_c = tl.load(Post_ptr + tok * 4 + dst_c).to(tl.float32)
        c0 = tl.load(Comb_ptr + tok * 16 + 0 * 4 + dst_c).to(tl.float32)
        c1 = tl.load(Comb_ptr + tok * 16 + 1 * 4 + dst_c).to(tl.float32)
        c2 = tl.load(Comb_ptr + tok * 16 + 2 * 4 + dst_c).to(tl.float32)
        c3 = tl.load(Comb_ptr + tok * 16 + 3 * 4 + dst_c).to(tl.float32)
        yc = (post_c * xv + c0 * r0 + c1 * r1 + c2 * r2 + c3 * r3).to(tl.bfloat16)
        tl.store(y_base + dst_c * D, yc)


@triton.jit
def _triton_hc_gemv_and_pre_rmsnorm_m4_kernel(
    X_ptr,
    W_ptr,
    PreMix_in_ptr,
    NormW_ptr,
    Mixes_raw_ptr,
    Rsqrt_ptr,
    Y_norm_ptr,
    norm_eps,
):
    """Fuses `_triton_hc_gemv_sqsum_m4_kernel` (`pid_c in 0..23`) and `_triton_hc_pre_rmsnorm_m4_kernel` (`pid_c == 24`)
    into a single kernel launch (`Grid: (n_toks, 25)`), since `hc_pre + RMSNorm` depends on `PreMix_in_ptr` from the
    previous sub-block and has zero data dependency on `Mixes_raw_ptr`.
    """
    row = tl.program_id(0)
    out_c = tl.program_id(1)
    x_base = X_ptr + row * 20480

    if out_c < 24:
        w_base = W_ptr + out_c * 20480
        acc_dot = tl.zeros([1024], dtype=tl.float32)
        acc_sq = tl.zeros([1024], dtype=tl.float32)
        offs_k_base = tl.arange(0, 1024)
        for k0 in tl.static_range(0, 20480, 1024):
            offs_k = k0 + offs_k_base
            xv = tl.load(x_base + offs_k).to(tl.float32)
            wv = tl.load(w_base + offs_k).to(tl.float32)
            acc_dot += xv * wv
            if out_c == 0:
                acc_sq += xv * xv

        dot_val = tl.sum(acc_dot, axis=0)
        tl.store(Mixes_raw_ptr + row * 24 + out_c, dot_val)
        if out_c == 0:
            mean_sq = tl.sum(acc_sq, axis=0) * (1.0 / 20480.0)
            tl.store(Rsqrt_ptr + row, tl.rsqrt(mean_sq + norm_eps))
    else:
        p0 = tl.load(PreMix_in_ptr + row * 4 + 0).to(tl.float32)
        p1 = tl.load(PreMix_in_ptr + row * 4 + 1).to(tl.float32)
        p2 = tl.load(PreMix_in_ptr + row * 4 + 2).to(tl.float32)
        p3 = tl.load(PreMix_in_ptr + row * 4 + 3).to(tl.float32)
        y_base = Y_norm_ptr + row * 5120

        acc_var = tl.zeros([1024], dtype=tl.float32)
        offs_d_base = tl.arange(0, 1024)
        for d0 in tl.static_range(0, 5120, 1024):
            offs_d = d0 + offs_d_base
            x0 = tl.load(x_base + 0 * 5120 + offs_d).to(tl.float32)
            x1 = tl.load(x_base + 1 * 5120 + offs_d).to(tl.float32)
            x2 = tl.load(x_base + 2 * 5120 + offs_d).to(tl.float32)
            x3 = tl.load(x_base + 3 * 5120 + offs_d).to(tl.float32)
            y_bf16 = (p0 * x0 + p1 * x1 + p2 * x2 + p3 * x3).to(tl.bfloat16)
            tl.store(y_base + offs_d, y_bf16)
            yf = y_bf16.to(tl.float32)
            acc_var += yf * yf

        var = tl.sum(acc_var, axis=0) * (1.0 / 5120.0)
        rstd = tl.rsqrt(var + norm_eps)

        for d0 in tl.static_range(0, 5120, 1024):
            offs_d = d0 + offs_d_base
            yf = tl.load(y_base + offs_d).to(tl.float32)
            nw = tl.load(NormW_ptr + offs_d).to(tl.float32)
            tl.store(y_base + offs_d, (yf * rstd * nw).to(tl.bfloat16))


@triton.jit
def _triton_hc_sinkhorn_and_post_m4_kernel(
    X_ptr,
    Res_ptr,
    Mixes_ptr,
    Rsqrt_ptr,
    Scale_ptr,
    Base_ptr,
    Pre_out_ptr,
    Y_ptr,
    hc_eps,
    SINKHORN_ITERS: tl.constexpr,
):
    """Fuses `_triton_hc_split_sinkhorn_m4_kernel` and `_triton_hc_post_m4_kernel` into 1 kernel launch (`Grid: (n_toks, 20)`).
    Each CTA computes the 4x4 Sinkhorn in registers (`~0.3 us`), `pid_d == 0` writes `Pre_out_ptr`, and all 20 CTAs
    compute `Y_ptr` (`[n_toks, 4, 5120]`) in a single vectorized `[4, 256]` store.
    """
    tok = tl.program_id(0)
    pid_d = tl.program_id(1)

    m_base = Mixes_ptr + tok * 24
    rsqrt = tl.load(Rsqrt_ptr + tok).to(tl.float32)
    s0 = tl.load(Scale_ptr + 0).to(tl.float32)
    s1 = tl.load(Scale_ptr + 1).to(tl.float32)
    s2 = tl.load(Scale_ptr + 2).to(tl.float32)

    offs4 = tl.arange(0, 4)
    if pid_d == 0:
        m_pre = tl.load(m_base + offs4).to(tl.float32) * rsqrt
        b_pre = tl.load(Base_ptr + offs4).to(tl.float32)
        pre = tl.sigmoid(m_pre * s0 + b_pre) + hc_eps
        tl.store(Pre_out_ptr + tok * 4 + offs4, pre)

    m_post = tl.load(m_base + 4 + offs4).to(tl.float32) * rsqrt
    b_post = tl.load(Base_ptr + 4 + offs4).to(tl.float32)
    post = 2.0 * tl.sigmoid(m_post * s1 + b_post)

    r_idx = tl.arange(0, 4)[:, None]
    c_idx = tl.arange(0, 4)[None, :]
    offs16 = 8 + r_idx * 4 + c_idx
    m_comb = tl.load(m_base + offs16).to(tl.float32) * rsqrt
    b_comb = tl.load(Base_ptr + offs16).to(tl.float32)
    comb = m_comb * s2 + b_comb

    row_max = tl.max(comb, axis=1)[:, None]
    comb = tl.exp(comb - row_max)
    comb = comb / tl.sum(comb, axis=1)[:, None] + hc_eps
    comb = comb / (tl.sum(comb, axis=0)[None, :] + hc_eps)
    for _ in tl.static_range(SINKHORN_ITERS - 1):
        comb = comb / (tl.sum(comb, axis=1)[:, None] + hc_eps)
        comb = comb / (tl.sum(comb, axis=0)[None, :] + hc_eps)

    offs_d = pid_d * 256 + tl.arange(0, 256)[None, :]
    xv = tl.load(X_ptr + tok * 5120 + offs_d).to(tl.float32)
    r_2d = tl.load(Res_ptr + tok * 20480 + r_idx * 5120 + offs_d).to(tl.float32)
    y_2d = (post[:, None] * xv + tl.sum(comb[:, :, None] * r_2d[:, None, :], axis=0)).to(tl.bfloat16)
    tl.store(Y_ptr + tok * 20480 + r_idx * 5120 + offs_d, y_2d)


def fast_hc_pre(self: Any, x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    if x.is_cuda and x.dtype == torch.bfloat16 and x.is_contiguous():
        b, s, hc, d = x.shape
        n_toks = b * s
        y = torch.empty((b, s, d), dtype=torch.bfloat16, device=x.device)
        _triton_hc_pre_m4_kernel[(n_toks, d // 256)](
            x, pre_mix.contiguous(), y, D=d, BLOCK_D=256, num_warps=4
        )
        return y
    return torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=2).to(x.dtype)


def fast_hc_post(
    self: Any, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor
) -> torch.Tensor:
    if x.is_cuda and x.dtype == torch.bfloat16 and x.is_contiguous() and residual.is_contiguous():
        b, s, d = x.shape
        n_toks = b * s
        y = torch.empty_like(residual)
        _triton_hc_post_m4_kernel[(n_toks, d // 256)](
            x, residual, post.contiguous(), comb.contiguous(), y, D=d, BLOCK_D=256, num_warps=4
        )
        return y
    y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
    return y.type_as(x)


def _get_block_decode_bufs(dev: torch.device) -> dict[str, Any]:
    b = _hc_block_bufs.get(dev)
    if b is None or b["x_res_0"].is_inference():
        with torch.inference_mode(False):
            b = {
                "x_res_0": torch.empty((1, 1, 4, 5120), dtype=torch.bfloat16, device=dev),
                "x_res_1": torch.empty((1, 1, 4, 5120), dtype=torch.bfloat16, device=dev),
                "pre_0": torch.empty((1, 1, 4), dtype=torch.float32, device=dev),
                "pre_1": torch.empty((1, 1, 4), dtype=torch.float32, device=dev),
                "x_norm": torch.empty((1, 1, 5120), dtype=torch.bfloat16, device=dev),
                "x_attn": torch.empty((1, 1, 5120), dtype=torch.bfloat16, device=dev),
                "attn_pre": torch.empty((1, 1, 4), dtype=torch.float32, device=dev),
                "mixes_raw_attn": torch.empty((1, 24), dtype=torch.float32, device=dev),
                "rsqrt_attn": torch.empty((1,), dtype=torch.float32, device=dev),
                "mixes_raw_ffn": torch.empty((1, 24), dtype=torch.float32, device=dev),
                "rsqrt_ffn": torch.empty((1,), dtype=torch.float32, device=dev),
                "x_post1": torch.empty((1, 1, 4, 5120), dtype=torch.bfloat16, device=dev),
                "pos_gpu": torch.zeros((1,), dtype=torch.int32, device=dev),
                "last_pos": -1,
                "graph_pool": torch.cuda.graph_pool_handle(),
            }
            _hc_block_bufs[dev] = b
    return b


_hc_block_bufs: dict[torch.device, dict[str, Any]] = {}


def fast_block_forward(
    self: Any,
    x: torch.Tensor,
    start_pos: int,
    pre_mix: torch.Tensor,
    image_mask: torch.Tensor | None,
    *attn_args: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.is_cuda and x.dtype == torch.bfloat16 and x.is_contiguous() and self.hc_mult == 4 and self.engram is None:
        b, s, _, d = x.shape
        n_toks = b * s
        if n_toks == 1 and d == 5120 and image_mask is None and start_pos > 0:
            bb = _get_block_decode_bufs(x.device)
            if bb["last_pos"] != start_pos:
                bb["pos_gpu"].fill_(start_pos)
                bb["last_pos"] = start_pos

            parity = int(getattr(self, "layer_id", 0)) & 1
            x_in_buf = bb[f"x_res_{parity}"]
            pre_in_buf = bb[f"pre_{parity}"]
            x_out_buf = bb[f"x_res_{1 - parity}"]
            pre_out_buf = bb[f"pre_{1 - parity}"]

            if x.data_ptr() != x_in_buf.data_ptr():
                x_in_buf.copy_(x)
            if pre_mix.data_ptr() != pre_in_buf.data_ptr():
                pre_in_buf.copy_(pre_mix)

            x_norm = bb["x_norm"]
            attn_pre = bb["attn_pre"]
            mixes_raw_attn = bb["mixes_raw_attn"]
            rsqrt_attn = bb["rsqrt_attn"]
            mixes_raw_ffn = bb["mixes_raw_ffn"]
            rsqrt_ffn = bb["rsqrt_ffn"]
            x_post1 = bb["x_post1"]

            def _run_step1_and_attn() -> torch.Tensor:
                _triton_hc_gemv_and_pre_rmsnorm_m4_kernel[(1, 25)](
                    x_in_buf,
                    self.hc_attn_fn,
                    pre_in_buf,
                    self.attn_norm.weight,
                    mixes_raw_attn,
                    rsqrt_attn,
                    x_norm,
                    self.norm_eps,
                    num_warps=4,
                )
                return self.attn(x_norm, start_pos, *attn_args)

            def _run_subblock2(x_attn_t: torch.Tensor) -> None:
                _triton_hc_sinkhorn_and_post_m4_kernel[(1, 20)](
                    x_attn_t,
                    x_in_buf,
                    mixes_raw_attn,
                    rsqrt_attn,
                    self.hc_attn_scale,
                    self.hc_attn_base,
                    attn_pre,
                    x_post1,
                    self.hc_eps,
                    SINKHORN_ITERS=self.hc_sinkhorn_iters,
                    num_warps=4,
                )
                _triton_hc_gemv_and_pre_rmsnorm_m4_kernel[(1, 25)](
                    x_post1,
                    self.hc_ffn_fn,
                    attn_pre,
                    self.ffn_norm.weight,
                    mixes_raw_ffn,
                    rsqrt_ffn,
                    x_norm,
                    self.norm_eps,
                    num_warps=4,
                )
                if getattr(self, "_ffn_takes_image_mask", None) is None:
                    try:
                        x_ffn = self.ffn(x_norm, image_mask=None)
                        self._ffn_takes_image_mask = True
                    except TypeError:
                        x_ffn = self.ffn(x_norm)
                        self._ffn_takes_image_mask = False
                elif self._ffn_takes_image_mask:
                    x_ffn = self.ffn(x_norm, image_mask=None)
                else:
                    x_ffn = self.ffn(x_norm)
                _triton_hc_sinkhorn_and_post_m4_kernel[(1, 20)](
                    x_ffn,
                    x_post1,
                    mixes_raw_ffn,
                    rsqrt_ffn,
                    self.hc_ffn_scale,
                    self.hc_ffn_base,
                    pre_out_buf,
                    x_out_buf,
                    self.hc_eps,
                    SINKHORN_ITERS=self.hc_sinkhorn_iters,
                    num_warps=4,
                )

            use_cg = (
                getattr(self.ffn, "w1_gu_u8", None) is None
                or self.ffn.w1_gu_u8.is_cuda
                or (not getattr(self.ffn, "t3_on_cpu", False))
                or getattr(self.ffn, "_active_mt3", 6) == 0
            )
            if not use_cg:
                xa = _run_step1_and_attn()
                _run_subblock2(xa)
                return x_out_buf, pre_out_buf

            ratio = getattr(self.attn, "compress_ratio", 0)
            compress_len = ((start_pos + 1) // ratio) if ratio else 0
            can_full_graph = (not getattr(self.attn, "is_kv_source", False)) and (compress_len <= 512) and (not attn_args)

            ffn_key = (id(self.ffn), getattr(self.ffn, "t3_on_cpu", False), id(self.attn))
            if getattr(self, "_cg_ffn_key", None) != ffn_key:
                self._cg_ffn_key = ffn_key
                self._cg_full_block = None
                self._cg_subblock2 = None
                self._cg_warmup_count = 0

            if can_full_graph:
                cg_full = getattr(self, "_cg_full_block", None)
                if cg_full is not None:
                    cg_full.replay()
                else:
                    if getattr(self, "_cg_warmup_count", 0) < 1:
                        xa = _run_step1_and_attn()
                        _run_subblock2(xa)
                        self._cg_warmup_count = getattr(self, "_cg_warmup_count", 0) + 1
                    else:
                        g = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g, pool=bb["graph_pool"]):
                            xa = _run_step1_and_attn()
                            _run_subblock2(xa)
                        self._cg_full_block = g
                        g.replay()
                return x_out_buf, pre_out_buf

            x_attn = _run_step1_and_attn()
            cg = getattr(self, "_cg_subblock2", None)
            if cg is not None:
                cg.replay()
            else:
                if getattr(self, "_cg_warmup_count", 0) < 1:
                    _run_subblock2(x_attn)
                    self._cg_warmup_count = getattr(self, "_cg_warmup_count", 0) + 1
                else:
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, pool=bb["graph_pool"]):
                        _run_subblock2(x_attn)
                    self._cg_subblock2 = g
                    g.replay()

            return x_out_buf, pre_out_buf

        if d == 5120:
            orig_prefix = x.shape[:-2]
            mixes_raw, rsqrt_buf = _get_hc_scratch(x.device, n_toks)
            x_norm = torch.empty((b, s, d), dtype=torch.bfloat16, device=x.device)
            attn_pre = torch.empty((*orig_prefix, 4), dtype=torch.float32, device=x.device)
            ffn_pre = torch.empty((*orig_prefix, 4), dtype=torch.float32, device=x.device)
            x_post1 = torch.empty_like(x)
            x_post2 = torch.empty_like(x)

            # Sub-block 1 (Attention)
            _triton_hc_gemv_and_pre_rmsnorm_m4_kernel[(n_toks, 25)](
                x,
                self.hc_attn_fn,
                pre_mix.contiguous(),
                self.attn_norm.weight,
                mixes_raw,
                rsqrt_buf,
                x_norm,
                self.norm_eps,
                num_warps=4,
            )
            x_attn = self.attn(x_norm, start_pos, *attn_args)
            _triton_hc_sinkhorn_and_post_m4_kernel[(n_toks, 20)](
                x_attn.contiguous(),
                x,
                mixes_raw,
                rsqrt_buf,
                self.hc_attn_scale,
                self.hc_attn_base,
                attn_pre,
                x_post1,
                self.hc_eps,
                SINKHORN_ITERS=self.hc_sinkhorn_iters,
                num_warps=4,
            )

            # Sub-block 2 (FFN)
            _triton_hc_gemv_and_pre_rmsnorm_m4_kernel[(n_toks, 25)](
                x_post1,
                self.hc_ffn_fn,
                attn_pre,
                self.ffn_norm.weight,
                mixes_raw,
                rsqrt_buf,
                x_norm,
                self.norm_eps,
                num_warps=4,
            )
            if getattr(self, "_ffn_takes_image_mask", None) is None:
                try:
                    x_ffn = self.ffn(x_norm, image_mask=image_mask)
                    self._ffn_takes_image_mask = True
                except TypeError:
                    x_ffn = self.ffn(x_norm)
                    self._ffn_takes_image_mask = False
            elif self._ffn_takes_image_mask:
                x_ffn = self.ffn(x_norm, image_mask=image_mask)
            else:
                x_ffn = self.ffn(x_norm)
            _triton_hc_sinkhorn_and_post_m4_kernel[(n_toks, 20)](
                x_ffn.contiguous(),
                x_post1,
                mixes_raw,
                rsqrt_buf,
                self.hc_ffn_scale,
                self.hc_ffn_base,
                ffn_pre,
                x_post2,
                self.hc_eps,
                SINKHORN_ITERS=self.hc_sinkhorn_iters,
                num_warps=4,
            )
            return x_post2, ffn_pre

    residual = x
    attn_pre, attn_post, attn_comb = self.hc_mixes(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
    x = self.hc_pre(x, pre_mix)
    if self.engram is not None:
        assert len(attn_args) >= 1
        x = self.engram(x, attn_args[0])
    x = self.attn_norm(x)
    x = self.attn(x, start_pos, *attn_args)
    x = self.hc_post(x, residual, attn_post, attn_comb)

    residual = x
    ffn_pre, ffn_post, ffn_comb = self.hc_mixes(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
    x = self.hc_pre(x, attn_pre)
    x = self.ffn_norm(x)
    try:
        x = self.ffn(x, image_mask=image_mask)
    except TypeError:
        x = self.ffn(x)
    x = self.hc_post(x, residual, ffn_post, ffn_comb)
    return x, ffn_pre


# -----------------------------------------------------------------------------
# 4. Fused Triton In-Place Quantization Kernels (FP8 block-32 & FP4 block-16/32)
# -----------------------------------------------------------------------------
@triton.jit
def _triton_act_quant_inplace_32_kernel(X_ptr, N_BLOCKS: tl.constexpr):
    blk = tl.program_id(0)
    offs = blk * 32 + tl.arange(0, 32)
    x = tl.load(X_ptr + offs).to(tl.float32)
    deq = _quantize_fp8_block32_in_reg(x)
    tl.store(X_ptr + offs, deq)


@triton.jit
def _triton_fp4_act_quant_inplace_kernel(
    X_ptr,
    BLOCK_SIZE: tl.constexpr,
    IS_E4M3_SCALE: tl.constexpr,
):
    blk = tl.program_id(0)
    offs = blk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(X_ptr + offs).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=0)
    if IS_E4M3_SCALE:
        amax = tl.maximum(amax, 6.0 * (1.0 / 512.0))
        s_raw = amax * (1.0 / 6.0)
        sb = s_raw.to(tl.uint32, bitcast=True)
        se = (sb >> 23) & 0xFF
        is_sub = se < 121
        rb = ((sb >> 20) & 1) + 0x7FFFF
        snorm = ((sb + rb) & 0xFFF00000).to(tl.float32, bitcast=True)
        ssub = tl.floor(s_raw * 512.0 + 0.5) * (1.0 / 512.0)
        s_fp32 = tl.where(is_sub, tl.maximum(ssub, 1.0 / 512.0), tl.minimum(snorm, 448.0))
    else:
        amax = tl.maximum(amax, 6.0 * 1.17549435e-38)
        scaled_max = amax * (1.0 / 6.0)
        bits = scaled_max.to(tl.int32, bitcast=True)
        exp_x = (bits >> 23) & 0xFF
        man_bits = bits & 0x7FFFFF
        log2_ceil = exp_x - 127 + tl.where(man_bits != 0, 1, 0)
        s_fp32 = ((log2_ceil + 127) << 23).to(tl.float32, bitcast=True)

    scaled = tl.minimum(tl.maximum(x / s_fp32, -6.0), 6.0)
    ax = tl.abs(scaled)
    sign = tl.where(scaled < 0.0, -1.0, 1.0)
    q_mag = tl.where(
        ax <= 0.25,
        0.0,
        tl.where(
            ax < 0.75,
            0.5,
            tl.where(
                ax <= 1.25,
                1.0,
                tl.where(
                    ax < 1.75,
                    1.5,
                    tl.where(
                        ax <= 2.5,
                        2.0,
                        tl.where(ax < 3.5, 3.0, tl.where(ax <= 5.0, 4.0, 6.0)),
                    ),
                ),
            ),
        ),
    )
    deq = (sign * q_mag * s_fp32).to(tl.bfloat16)
    tl.store(X_ptr + offs, deq)


# -----------------------------------------------------------------------------
# 5. Fused Triton RoPE In-Place Kernel (zero-copy view of complex64 freqs_cis)
# -----------------------------------------------------------------------------
@triton.jit
def _triton_apply_rotary_inplace_kernel(
    X_ptr,
    Freqs_ptr,
    stride_seq,
    stride_vec,
    stride_freq_seq,
    HALF_RD: tl.constexpr,
    INVERSE: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    vec_idx = tl.program_id(1)
    offs = tl.arange(0, HALF_RD)

    f_base = Freqs_ptr + seq_idx * stride_freq_seq
    cos = tl.load(f_base + offs * 2).to(tl.float32)
    sin = tl.load(f_base + offs * 2 + 1).to(tl.float32)
    if INVERSE:
        sin = -sin

    x_base = X_ptr + seq_idx * stride_seq + vec_idx * stride_vec
    re = tl.load(x_base + offs * 2).to(tl.float32)
    im = tl.load(x_base + offs * 2 + 1).to(tl.float32)

    out_re = (re * cos - im * sin).to(tl.bfloat16)
    out_im = (re * sin + im * cos).to(tl.bfloat16)
    tl.store(x_base + offs * 2, out_re)
    tl.store(x_base + offs * 2 + 1, out_im)


# -----------------------------------------------------------------------------
# 6. Fused Tensor-Core MLA Decode Pipeline Kernels
# -----------------------------------------------------------------------------
@triton.jit
def _triton_mla_qnorm_kvprep_decode_kernel(
    QKV_raw_ptr,
    Q_norm_w_ptr,
    KV_norm_w_ptr,
    QR_out_ptr,
    Win_KV_cache_ptr,
    Freqs_base_ptr,
    Pos_ptr,
    eps,
    WIN_SIZE: tl.constexpr,
):
    """Grid: (2,).
    - pid == 0: RMSNorm on `QKV_raw[0:1280]` -> `QR_out[0:1280]`
    - pid == 1: RMSNorm on `QKV_raw[1280:1792]` + RoPE on tail `[448:512]` + block-32 FP8 `act_quant` -> `Win_KV_cache[win_slot, 0:512]`
    """
    pid = tl.program_id(0)
    if pid == 0:
        offs_q = tl.arange(0, 2048)
        m_q = offs_q < 1280
        xq = tl.load(QKV_raw_ptr + offs_q, mask=m_q, other=0.0).to(tl.float32)
        wq = tl.load(Q_norm_w_ptr + offs_q, mask=m_q, other=0.0).to(tl.float32)
        var_q = tl.sum(xq * xq, axis=0) * (1.0 / 1280.0)
        rstd_q = tl.rsqrt(var_q + eps)
        tl.store(QR_out_ptr + offs_q, (xq * rstd_q * wq).to(tl.bfloat16), mask=m_q)
    else:
        start_pos = tl.load(Pos_ptr)
        win_slot = start_pos % WIN_SIZE
        offs_512 = tl.arange(0, 512)
        kv = tl.load(QKV_raw_ptr + 1280 + offs_512).to(tl.float32)
        wkv = tl.load(KV_norm_w_ptr + offs_512).to(tl.float32)
        var_kv = tl.sum(kv * kv, axis=0) * (1.0 / 512.0)
        rstd_kv = tl.rsqrt(var_kv + eps)
        kv_normed = (kv * rstd_kv * wkv).to(tl.bfloat16)

        # Store normed temporarily to Win_KV_cache[win_slot] so we can apply RoPE on 448..511 and quantize 16x32
        dst_ptr = Win_KV_cache_ptr + win_slot * 512
        tl.store(dst_ptr + offs_512, kv_normed)
        tl.debug_barrier()

        offs_32 = tl.arange(0, 32)
        f_ptr = Freqs_base_ptr + start_pos * 64
        cos = tl.load(f_ptr + offs_32 * 2).to(tl.float32)
        sin = tl.load(f_ptr + offs_32 * 2 + 1).to(tl.float32)
        re = tl.load(dst_ptr + 448 + offs_32 * 2).to(tl.float32)
        im = tl.load(dst_ptr + 448 + offs_32 * 2 + 1).to(tl.float32)
        tl.store(dst_ptr + 448 + offs_32 * 2, (re * cos - im * sin).to(tl.bfloat16))
        tl.store(dst_ptr + 448 + offs_32 * 2 + 1, (re * sin + im * cos).to(tl.bfloat16))
        tl.debug_barrier()

        # Block-32 FP8 act_quant across 16 blocks of 32
        r_blk = tl.arange(0, 16)[:, None]
        c_32 = tl.arange(0, 32)[None, :]
        offs_2d = r_blk * 32 + c_32
        kv_2d = tl.load(dst_ptr + offs_2d).to(tl.float32)
        kv_q = _quantize_fp8_block32_in_reg(kv_2d)
        tl.store(dst_ptr + offs_2d, kv_q)


@triton.jit
def _triton_rope_q64_decode_kernel(
    Q_ptr,
    Freqs_base_ptr,
    start_pos,
):
    """Grid: (64,). Rotates the last 64 dims (`448:512`) of `Q[0, 0, h, :]` in place."""
    h = tl.program_id(0)
    offs_32 = tl.arange(0, 32)
    f_ptr = Freqs_base_ptr + start_pos * 64
    cos = tl.load(f_ptr + offs_32 * 2).to(tl.float32)
    sin = tl.load(f_ptr + offs_32 * 2 + 1).to(tl.float32)
    q_base = Q_ptr + h * 512 + 448
    re = tl.load(q_base + offs_32 * 2).to(tl.float32)
    im = tl.load(q_base + offs_32 * 2 + 1).to(tl.float32)
    tl.store(q_base + offs_32 * 2, (re * cos - im * sin).to(tl.bfloat16))
    tl.store(q_base + offs_32 * 2 + 1, (re * sin + im * cos).to(tl.bfloat16))


@triton.jit
def _triton_mla_qrope_scores_tc_kernel(
    Q_ptr,
    W_KV_ptr,
    C_KV_ptr,
    Freqs_base_ptr,
    Scores_ptr,
    start_pos,
    w_len,
    total_len,
    sm_scale,
    STRIDE_SCORES_H: tl.constexpr,
):
    """Computes `Scores[0:64, k_start:k_start+64] = (RoPE(Q) @ KV^T) * sm_scale`
    using BF16 Tensor Cores (`tl.dot`) across all 64 query heads simultaneously,
    with in-register Q RoPE via `tl.interleave` and 100% contiguous KV loads!
    Grid: (cdiv(total_len, 64),)
    """
    blk_k = tl.program_id(0)
    offs_h = tl.arange(0, 64)
    offs_k = blk_k * 64 + tl.arange(0, 64)
    mask_w = offs_k < w_len
    mask_c = (offs_k >= w_len) & (offs_k < total_len)
    mask_k = offs_k < total_len
    w_idx = tl.where(mask_w, offs_k, 0)
    c_idx = tl.where(mask_c, offs_k - w_len, 0)

    acc = tl.zeros([64, 64], dtype=tl.float32)

    for d0 in tl.static_range(0, 448, 64):
        offs_d = d0 + tl.arange(0, 64)
        q_tile = tl.load(Q_ptr + offs_h[:, None] * 512 + offs_d[None, :])
        w_tile = tl.load(W_KV_ptr + w_idx[:, None] * 512 + offs_d[None, :], mask=mask_w[:, None], other=0.0)
        c_tile = tl.load(C_KV_ptr + c_idx[:, None] * 512 + offs_d[None, :], mask=mask_c[:, None], other=0.0)
        kv_tile = tl.where(mask_w[:, None], w_tile, c_tile)
        acc = tl.dot(q_tile, tl.trans(kv_tile), acc)

    # Last 64 dims (448..511): rotate Q in registers via tl.interleave and load KV contiguously!
    offs_32 = tl.arange(0, 32)
    f_ptr = Freqs_base_ptr + start_pos * 64
    cos = tl.load(f_ptr + offs_32 * 2).to(tl.float32)[None, :]
    sin = tl.load(f_ptr + offs_32 * 2 + 1).to(tl.float32)[None, :]
    q_re = tl.load(Q_ptr + offs_h[:, None] * 512 + 448 + offs_32[None, :] * 2).to(tl.float32)
    q_im = tl.load(Q_ptr + offs_h[:, None] * 512 + 448 + offs_32[None, :] * 2 + 1).to(tl.float32)
    rot_re = (q_re * cos - q_im * sin).to(tl.bfloat16)
    rot_im = (q_re * sin + q_im * cos).to(tl.bfloat16)
    q_tail = tl.interleave(rot_re, rot_im)

    offs_d_tail = 448 + tl.arange(0, 64)
    w_tail = tl.load(W_KV_ptr + w_idx[:, None] * 512 + offs_d_tail[None, :], mask=mask_w[:, None], other=0.0)
    c_tail = tl.load(C_KV_ptr + c_idx[:, None] * 512 + offs_d_tail[None, :], mask=mask_c[:, None], other=0.0)
    kv_tail = tl.where(mask_w[:, None], w_tail, c_tail)
    acc = tl.dot(q_tail, tl.trans(kv_tail), acc)

    scores = acc * sm_scale
    tl.store(
        Scores_ptr + offs_h[:, None] * STRIDE_SCORES_H + offs_k[None, :],
        scores,
        mask=mask_k[None, :],
    )


@triton.jit
def _triton_mla_qrope_scores_softmax_tc_kernel(
    Q_ptr,
    W_KV_ptr,
    C_KV_ptr,
    Freqs_base_ptr,
    Scores_ptr,
    P_bf16_ptr,
    Inv_sum_exp_ptr,
    Sink_ptr,
    Pos_ptr,
    sm_scale,
    WIN_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    STRIDE_SCORES_H: tl.constexpr,
):
    """Fuses `_triton_mla_qrope_scores_tc_kernel` and `_triton_mla_decode_softmax_kernel` into 1 kernel launch (`Grid: (4,)`).
    Reads `start_pos` directly from GPU memory (`Pos_ptr`) so the entire kernel is 100% static-shape for `torch.cuda.CUDAGraph`!
    """
    pid_h = tl.program_id(0)
    offs_h = pid_h * 16 + tl.arange(0, 16)
    start_pos = tl.load(Pos_ptr)
    end_pos = start_pos + 1
    w_len = tl.minimum(end_pos, WIN_SIZE)
    c_len = (end_pos // COMPRESS_RATIO) if COMPRESS_RATIO > 0 else 0
    total_len = w_len + c_len

    offs_32 = tl.arange(0, 32)
    f_ptr = Freqs_base_ptr + start_pos * 64
    cos = tl.load(f_ptr + offs_32 * 2).to(tl.float32)[None, :]
    sin = tl.load(f_ptr + offs_32 * 2 + 1).to(tl.float32)[None, :]
    q_re = tl.load(Q_ptr + offs_h[:, None] * 512 + 448 + offs_32[None, :] * 2).to(tl.float32)
    q_im = tl.load(Q_ptr + offs_h[:, None] * 512 + 448 + offs_32[None, :] * 2 + 1).to(tl.float32)
    rot_re = (q_re * cos - q_im * sin).to(tl.bfloat16)
    rot_im = (q_re * sin + q_im * cos).to(tl.bfloat16)
    q_tail = tl.interleave(rot_re, rot_im)
    offs_d_tail = 448 + tl.arange(0, 64)

    m_i = tl.full([16], -float("inf"), dtype=tl.float32)
    for k0 in range(0, total_len, 64):
        offs_k = k0 + tl.arange(0, 64)
        mask_w = offs_k < w_len
        mask_c = (offs_k >= w_len) & (offs_k < total_len)
        mask_k = offs_k < total_len
        w_idx = tl.where(mask_w, offs_k, 0)
        c_idx = tl.where(mask_c, offs_k - w_len, 0)

        acc = tl.zeros([16, 64], dtype=tl.float32)
        for d0 in range(0, 448, 64):
            offs_d = d0 + tl.arange(0, 64)
            q_tile = tl.load(Q_ptr + offs_h[:, None] * 512 + offs_d[None, :])
            w_tile = tl.load(W_KV_ptr + w_idx[:, None] * 512 + offs_d[None, :], mask=mask_w[:, None], other=0.0)
            c_tile = tl.load(C_KV_ptr + c_idx[:, None] * 512 + offs_d[None, :], mask=mask_c[:, None], other=0.0)
            kv_tile = tl.where(mask_w[:, None], w_tile, c_tile)
            acc = tl.dot(q_tile, tl.trans(kv_tile), acc)

        w_tail = tl.load(W_KV_ptr + w_idx[:, None] * 512 + offs_d_tail[None, :], mask=mask_w[:, None], other=0.0)
        c_tail = tl.load(C_KV_ptr + c_idx[:, None] * 512 + offs_d_tail[None, :], mask=mask_c[:, None], other=0.0)
        kv_tail = tl.where(mask_w[:, None], w_tail, c_tail)
        acc = tl.dot(q_tail, tl.trans(kv_tail), acc)

        scores = acc * sm_scale
        scores_masked = tl.where(mask_k[None, :], scores, -float("inf"))
        m_i = tl.maximum(m_i, tl.max(scores_masked, axis=1))
        tl.store(
            Scores_ptr + offs_h[:, None] * STRIDE_SCORES_H + offs_k[None, :],
            scores_masked,
            mask=mask_k[None, :],
        )

    s_max = tl.maximum(m_i, -1e30)
    sink_h = tl.load(Sink_ptr + offs_h).to(tl.float32)
    sum_exp = tl.exp(sink_h - s_max)
    for k0 in range(0, total_len, 64):
        offs_k_s = k0 + tl.arange(0, 64)
        mask_k_s = offs_k_s < total_len
        sc = tl.load(
            Scores_ptr + offs_h[:, None] * STRIDE_SCORES_H + offs_k_s[None, :],
            mask=mask_k_s[None, :],
            other=-float("inf"),
        )
        exp_s = tl.where(mask_k_s[None, :], tl.exp(sc - s_max[:, None]), 0.0)
        sum_exp += tl.sum(exp_s, axis=1)
        tl.store(
            P_bf16_ptr + offs_h[:, None] * STRIDE_SCORES_H + offs_k_s[None, :],
            exp_s.to(tl.bfloat16),
            mask=mask_k_s[None, :],
        )
    tl.store(Inv_sum_exp_ptr + offs_h, 1.0 / sum_exp)


@triton.jit
def _triton_mla_decode_out_rope_quant_tc_kernel(
    P_bf16_ptr,
    Inv_sum_exp_ptr,
    W_KV_ptr,
    C_KV_ptr,
    Freqs_base_ptr,
    Out_ptr,
    Pos_ptr,
    WIN_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    STRIDE_SCORES_H: tl.constexpr,
):
    """Grid: (8,).
    Reads `start_pos` directly from GPU memory (`Pos_ptr`) for 100% `torch.cuda.CUDAGraph` compatibility!
    """
    pid = tl.program_id(0)
    offs_h = tl.arange(0, 64)
    start_pos = tl.load(Pos_ptr)
    end_pos = start_pos + 1
    w_len = tl.minimum(end_pos, WIN_SIZE)
    c_len = (end_pos // COMPRESS_RATIO) if COMPRESS_RATIO > 0 else 0
    total_len = w_len + c_len

    inv_sum = tl.load(Inv_sum_exp_ptr + offs_h)[:, None]
    offs_d = pid * 64 + tl.arange(0, 64)
    acc = tl.zeros([64, 64], dtype=tl.float32)

    for k0 in range(0, total_len, 64):
        offs_k = k0 + tl.arange(0, 64)
        mask_w = offs_k < w_len
        mask_c = (offs_k >= w_len) & (offs_k < total_len)
        mask_k = offs_k < total_len
        w_idx = tl.where(mask_w, offs_k, 0)
        c_idx = tl.where(mask_c, offs_k - w_len, 0)

        p_tile = tl.load(
            P_bf16_ptr + offs_h[:, None] * STRIDE_SCORES_H + offs_k[None, :],
            mask=mask_k[None, :],
            other=0.0,
        )
        w_tile = tl.load(W_KV_ptr + w_idx[:, None] * 512 + offs_d[None, :], mask=mask_w[:, None], other=0.0)
        c_tile = tl.load(C_KV_ptr + c_idx[:, None] * 512 + offs_d[None, :], mask=mask_c[:, None], other=0.0)
        kv_tile = tl.where(mask_w[:, None], w_tile, c_tile)
        acc = tl.dot(p_tile, kv_tile, acc)

    out_bf16 = (acc * inv_sum).to(tl.bfloat16)
    tl.store(Out_ptr + offs_h[:, None] * 512 + offs_d[None, :], out_bf16)
    tl.debug_barrier()

    if pid == 7:
        offs_32 = tl.arange(0, 32)
        f_ptr = Freqs_base_ptr + start_pos * 64
        cos = tl.load(f_ptr + offs_32 * 2).to(tl.float32)[None, :]
        sin = -tl.load(f_ptr + offs_32 * 2 + 1).to(tl.float32)[None, :]
        re = tl.load(Out_ptr + offs_h[:, None] * 512 + 448 + offs_32[None, :] * 2).to(tl.float32)
        im = tl.load(Out_ptr + offs_h[:, None] * 512 + 448 + offs_32[None, :] * 2 + 1).to(tl.float32)
        tl.store(Out_ptr + offs_h[:, None] * 512 + 448 + offs_32[None, :] * 2, (re * cos - im * sin).to(tl.bfloat16))
        tl.store(
            Out_ptr + offs_h[:, None] * 512 + 448 + offs_32[None, :] * 2 + 1, (re * sin + im * cos).to(tl.bfloat16)
        )
        tl.debug_barrier()

    r_blk = tl.arange(0, 128)[:, None]
    c_32 = tl.arange(0, 32)[None, :]
    h_idx = r_blk // 2
    sub_d = (r_blk % 2) * 32 + c_32
    val_32 = tl.load(Out_ptr + h_idx * 512 + pid * 64 + sub_d).to(tl.float32)
    q_32 = _quantize_fp8_block32_in_reg(val_32)
    tl.store(Out_ptr + h_idx * 512 + pid * 64 + sub_d, q_32)


def install_dsv41_triton_backbone(dsv41_model_mod: Any, enable_direct_uva_t3: bool = True) -> dict[str, Any]:
    """Patches `dsv41_model_mod` (`model.py`) in-place with fused Triton backbone & Tensor-Core decode kernels."""
    if not hasattr(dsv41_model_mod, "_orig_act_quant"):
        dsv41_model_mod._orig_act_quant = dsv41_model_mod.act_quant
        dsv41_model_mod._orig_fp4_act_quant = dsv41_model_mod.fp4_act_quant
        dsv41_model_mod._orig_apply_rotary_emb = dsv41_model_mod.apply_rotary_emb
        dsv41_model_mod._orig_sparse_attn = dsv41_model_mod.sparse_attn
        dsv41_model_mod._orig_Attention_forward = dsv41_model_mod.Attention.forward
        dsv41_model_mod._orig_Block_forward = dsv41_model_mod.Block.forward
    orig_act_quant = dsv41_model_mod._orig_act_quant
    orig_fp4_act_quant = dsv41_model_mod._orig_fp4_act_quant
    orig_apply_rotary_emb = dsv41_model_mod._orig_apply_rotary_emb
    orig_sparse_attn = dsv41_model_mod._orig_sparse_attn

    def _fast_act_quant(x: torch.Tensor, block_size: int = 32, scale_fmt=None, scale_dtype=None, inplace: bool = False):
        if (
            inplace
            and x.is_cuda
            and x.is_contiguous()
            and block_size == 32
            and scale_fmt is not None
            and x.dtype == torch.bfloat16
        ):
            n_blocks = x.numel() // 32
            _triton_act_quant_inplace_32_kernel[(n_blocks,)](x, N_BLOCKS=n_blocks, num_warps=1)
            return x
        return orig_act_quant(x, block_size, scale_fmt, scale_dtype, inplace)

    def _fast_fp4_act_quant(x: torch.Tensor, block_size: int = 32, inplace: bool = False, scale_dtype=None):
        if inplace and x.is_cuda and x.is_contiguous() and x.dtype == torch.bfloat16 and block_size in (16, 32):
            n_blocks = x.numel() // block_size
            is_e4m3 = scale_dtype == torch.float8_e4m3fn
            _triton_fp4_act_quant_inplace_kernel[(n_blocks,)](
                x, BLOCK_SIZE=block_size, IS_E4M3_SCALE=is_e4m3, num_warps=1
            )
            return x
        return orig_fp4_act_quant(x, block_size, inplace, scale_dtype)

    def _fast_apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        if x.is_cuda and x.dtype == torch.bfloat16 and x.size(-1) == 64 and x.stride(-1) == 1:
            f_real = torch.view_as_real(freqs_cis)
            if f_real.stride(-1) == 1 and f_real.stride(-2) == 2:
                stride_f_seq = f_real.stride(0) if f_real.ndim >= 2 else 0
                if x.ndim == 3 and x.size(0) == 1:
                    seqlen = x.size(1)
                    _triton_apply_rotary_inplace_kernel[(seqlen, 1)](
                        x, f_real, x.stride(1), 0, stride_f_seq, HALF_RD=32, INVERSE=inverse, num_warps=1
                    )
                    return x
                elif x.ndim == 4 and x.size(0) == 1:
                    seqlen = x.size(1)
                    n_heads = x.size(2)
                    _triton_apply_rotary_inplace_kernel[(seqlen, n_heads)](
                        x, f_real, x.stride(1), x.stride(2), stride_f_seq, HALF_RD=32, INVERSE=inverse, num_warps=1
                    )
                    return x
        return orig_apply_rotary_emb(x, freqs_cis, inverse)

    def _fast_sparse_attn(
        q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor, topk_idxs: torch.Tensor, softmax_scale=None
    ):
        b, s, h, d = q.size()
        if softmax_scale is None:
            softmax_scale = d**-0.5
        topk = topk_idxs.size(-1)
        if topk == 0 or s == 0:
            return torch.zeros_like(q)
        sink_fp32 = attn_sink[:h].float().view(1, 1, h, 1)
        if b == 1 and s <= 64:
            valid = topk_idxs != -1
            safe_idx = topk_idxs.clamp_min(0).long()
            kv_g = kv[0, safe_idx]
            scores = torch.matmul(q.float(), kv_g.float().transpose(-1, -2)) * softmax_scale
            scores = scores.masked_fill(~valid.unsqueeze(2), -torch.inf)
            s_max = scores.amax(dim=-1, keepdim=True).clamp_min(-1e30)
            exp_s = torch.exp(scores - s_max)
            sum_exp = exp_s.sum(dim=-1, keepdim=True) + torch.exp(sink_fp32 - s_max)
            acc_o = torch.matmul(exp_s.to(kv.dtype), kv_g).float()
            return (acc_o / sum_exp).to(q.dtype)
        return orig_sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale)

    # Pre-allocated scratch buffers per CUDA device for zero-allocation decode attention
    _decode_bufs: dict[torch.device, dict[str, torch.Tensor]] = {}

    def _get_decode_bufs(dev: torch.device) -> dict[str, torch.Tensor]:
        b = _decode_bufs.get(dev)
        if b is None or b["qr"].is_inference():
            with torch.inference_mode(False):
                b = {
                    "qr": torch.empty((1, 1, 1280), dtype=torch.bfloat16, device=dev),
                    "scores": torch.empty((64, 1024), dtype=torch.float32, device=dev),
                    "p_bf16": torch.empty((64, 1024), dtype=torch.bfloat16, device=dev),
                    "inv_sum": torch.empty((64,), dtype=torch.float32, device=dev),
                    "o": torch.empty((1, 1, 64, 512), dtype=torch.bfloat16, device=dev),
                }
                _decode_bufs[dev] = b
        return b

    orig_attn_forward = dsv41_model_mod._orig_Attention_forward

    def _fast_attn_forward(self: Any, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        bsz, seqlen, _ = x.size()
        if bsz == 1 and seqlen == 1 and start_pos > 0:
            end_pos = start_pos + 1
            ratio = self.compress_ratio
            compress_len = (end_pos // ratio) if ratio else 0
            if compress_len <= 512:
                rd = self.rope_head_dim
                win = int(self.window_size)
                ratio_i = int(ratio) if ratio else 0
                bufs = _get_decode_bufs(x.device)
                bb = _get_block_decode_bufs(x.device)
                pos_gpu = bb["pos_gpu"]
                if (not torch.cuda.is_current_stream_capturing()) and bb["last_pos"] != start_pos:
                    pos_gpu.fill_(start_pos)
                    bb["last_pos"] = start_pos

                # Lazily fuse [wq_a.weight; wkv.weight] into a zero-extra-VRAM shared storage buffer
                fused_qkv_w = getattr(self, "_wq_a_wkv_fused", None)
                if fused_qkv_w is None or fused_qkv_w.is_inference():
                    with torch.inference_mode(False):
                        fused_qkv_w = torch.cat([self.wq_a.weight.detach(), self.wkv.weight.detach()], dim=0)
                        self.wq_a.weight = nn.Parameter(fused_qkv_w[:1280], requires_grad=False)
                        self.wkv.weight = nn.Parameter(fused_qkv_w[1280:], requires_grad=False)
                        self._wq_a_wkv_fused = fused_qkv_w

                freqs_real = getattr(self, "_freqs_real_cached", None)
                if freqs_real is None or freqs_real.is_inference():
                    with torch.inference_mode(False):
                        freqs_real = torch.view_as_real(self.freqs_cis.detach())
                        self._freqs_real_cached = freqs_real

                # 1. Single cuBLAS GEMV for both wq_a(x) and wkv(x)
                qkv_raw = F.linear(x, fused_qkv_w)
                qr = bufs["qr"]

                # 2. Fused q_norm + kv_norm + kv RoPE + kv FP8 act_quant + window_kv_cache write
                _triton_mla_qnorm_kvprep_decode_kernel[(2,)](
                    qkv_raw,
                    self.q_norm.weight,
                    self.kv_norm.weight,
                    qr,
                    self.window_kv_cache,
                    freqs_real,
                    pos_gpu,
                    self.eps,
                    WIN_SIZE=win,
                    num_warps=4,
                )

                # 3. cuBLAS GEMV for wq_b(qr) (Q RoPE is fused inside _triton_mla_qrope_scores_softmax_tc_kernel!)
                q = F.linear(qr, self.wq_b.weight).view(1, 1, self.n_local_heads, self.head_dim)
                w_kv = self.window_kv_cache

                # 4. Update compressor & indexer k_cache on the 4 kv_source_layers
                if ratio_i:
                    latent = None
                    if self.is_kv_source:
                        latent = self.compressor(x, start_pos)
                        dsv41_model_mod.shared_attn.compress_kv = self.compress_kv_cache
                    if self.is_index_source and self.indexer is not None and self.indexer.owns_k:
                        if self.indexer.freqs_cis is None:
                            self.indexer.freqs_cis = self.freqs_cis
                        if latent is not None:
                            freqs_c = self.freqs_cis[end_pos - ratio_i].unsqueeze(0)
                            k_idx = self.indexer.k_norm(self.indexer.wk(latent))
                            _fast_apply_rotary_emb(k_idx[..., -rd:], freqs_c)
                            _fast_fp4_act_quant(k_idx, dsv41_model_mod.fp4_block_size, True)
                            self.indexer.k_cache[0, start_pos // ratio_i : start_pos // ratio_i + 1] = k_idx[0]
                        dsv41_model_mod.shared_attn.index_k = self.indexer.k_cache
                    if latent is not None:
                        freqs_c = self.freqs_cis[end_pos - ratio_i].unsqueeze(0)
                        _fast_apply_rotary_emb(latent[..., -rd:], freqs_c)
                        _fast_fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)
                        self.compress_kv_cache[0, start_pos // ratio_i : start_pos // ratio_i + 1] = latent[0]
                    c_kv = dsv41_model_mod.shared_attn.compress_kv
                else:
                    c_kv = w_kv

                # 5. Fused Tensor-Core Q-RoPE + Q@KV^T Scores + Softmax -> Tensor-Core P@KV + Inv-RoPE + FP8 Quant
                scores_buf = bufs["scores"]
                p_bf16_buf = bufs["p_bf16"]
                inv_sum_buf = bufs["inv_sum"]
                o = bufs["o"]

                _triton_mla_qrope_scores_softmax_tc_kernel[(4,)](
                    q,
                    w_kv,
                    c_kv,
                    freqs_real,
                    scores_buf,
                    p_bf16_buf,
                    inv_sum_buf,
                    self.attn_sink,
                    pos_gpu,
                    self.softmax_scale,
                    WIN_SIZE=win,
                    COMPRESS_RATIO=ratio_i,
                    STRIDE_SCORES_H=1024,
                    num_warps=4,
                )
                _triton_mla_decode_out_rope_quant_tc_kernel[(8,)](
                    p_bf16_buf,
                    inv_sum_buf,
                    w_kv,
                    c_kv,
                    freqs_real,
                    o,
                    pos_gpu,
                    WIN_SIZE=win,
                    COMPRESS_RATIO=ratio_i,
                    STRIDE_SCORES_H=1024,
                    num_warps=4,
                )

                # 6. Grouped wo_a bmm + wo_b linear directly into per-layer static buffer
                wo_a_t = getattr(self, "_wo_a_t_cached", None)
                if wo_a_t is None or wo_a_t.is_inference():
                    with torch.inference_mode(False):
                        wo_a_t = self.wo_a.weight.detach().view(self.n_local_groups, self.o_lora_rank, -1).transpose(1, 2)
                        self._wo_a_t_cached = wo_a_t
                wo_b_t = getattr(self, "_wo_b_t_cached", None)
                if wo_b_t is None or wo_b_t.is_inference():
                    with torch.inference_mode(False):
                        wo_b_t = self.wo_b.weight.detach().t()
                        self._wo_b_t_cached = wo_b_t
                x_attn_out = getattr(self, "_x_attn_out_buf", None)
                if x_attn_out is None or x_attn_out.device != x.device or x_attn_out.is_inference():
                    with torch.inference_mode(False):
                        x_attn_out = torch.empty((1, 1, self.dim), dtype=torch.bfloat16, device=x.device)
                        self._x_attn_out_buf = x_attn_out
                o_proj = torch.bmm(o.view(self.n_local_groups, 1, -1), wo_a_t).view(1, -1)
                torch.matmul(o_proj, wo_b_t, out=x_attn_out.view(1, self.dim))
                return x_attn_out

        return orig_attn_forward(self, x, start_pos)

    dsv41_model_mod.RMSNorm.forward = fast_rmsnorm_forward
    dsv41_model_mod.hc_split_sinkhorn = fast_hc_split_sinkhorn
    dsv41_model_mod.Block.hc_mixes = fast_block_hc_mixes
    dsv41_model_mod.Block.hc_pre = fast_hc_pre
    dsv41_model_mod.Block.hc_post = fast_hc_post
    dsv41_model_mod.Block.forward = fast_block_forward
    dsv41_model_mod.act_quant = _fast_act_quant
    dsv41_model_mod.fp4_act_quant = _fast_fp4_act_quant
    dsv41_model_mod.apply_rotary_emb = _fast_apply_rotary_emb
    dsv41_model_mod.sparse_attn = _fast_sparse_attn
    dsv41_model_mod.Attention.forward = _fast_attn_forward

    return {
        "rmsnorm": True,
        "hc_gemv_sinkhorn": True,
        "hc_pre_rmsnorm": True,
        "hc_post": True,
        "act_quant_inplace": True,
        "apply_rotary_inplace": True,
        "mla_decode_tc_fused": True,
        "direct_uva_t3": enable_direct_uva_t3,
    }


def enable_direct_uva_tier3_on_model(model: Any) -> int:
    """Switches CPU-offloaded Tier-3 MoE layers to direct zero-copy UVA pinned-memory reads,
    freeing their 7-slot GPU bounce buffers and eliminating the 2 staging kernel launches per layer.
    """
    switched = 0
    for layer in getattr(model, "layers", []):
        moe = getattr(layer, "ffn", None)
        if moe is not None and getattr(moe, "t3_on_cpu", False) and getattr(moe, "w1_gu_cpu", None) is not None:
            if moe.w1_gu_cpu.is_pinned():
                moe.w1_gu_u8 = moe.w1_gu_cpu
                moe.s1_gu_fp8 = moe.s1_gu_cpu
                moe.w1_d_u8 = moe.w1_d_cpu
                moe.s1_d_fp8 = moe.s1_d_cpu
                moe.t3_on_cpu = False
                switched += 1
    torch.cuda.empty_cache()
    return switched
