#!/usr/bin/env python3
"""
Chronos-1T x PseudoBrain (pb27) — Batched 48-Layer Full Checkpoint Converter & Runtime Module (`v6.2`).

Vectorizes the 8-expert holographic group conversion across GPU batches (`(8, rows, cols)`):
- Batched NVFP4 unpacking for all 8 experts simultaneously
- Batched Subspace Eckart-Young SVD (`(8, 1280, 64)`) for fused `gate_up_proj`
- Batched Exact Gram Eigendecomposition (`(8, 640, 640)`) for `down_proj` Full-Spectrum SMW Healer
- Batched 2-Bit Lloyd-Max / Base-3 Ternary / 1-Bit Rademacher Tail Lattice bit-packing
- Resumable 48-layer streaming conversion into `/home/defnotean/pb27/hf/flash_next_chronos_v6`
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chronos_1t.converter.convert_full_layer_512 import (
    COPY_FILES,
    compose_chronos_checkpoint_dir,
)
from chronos_1t.models.chronos_moe_block import (
    LLOYD_MAX_2BIT_LUT,
    NVFP4_LUT,
    build_hadamard_128,
    fold_2d_permuted_hadamard,
    generate_expert_1d_keys,
    pack_nvfp4,
    unbind_2d_permuted_hadamard,
    unpack_nvfp4,
)
from chronos_1t.models.vllm_chronos_moe import dequantize_mxfp8

MODEL_DIR = Path("/home/defnotean/pb27/hf/flash_next_nvfp4")
OUT_CKPT_DIR = Path("/home/defnotean/pb27/hf/flash_next_chronos_v6")


@torch.no_grad()
def unpack_nvfp4_batched(
    packed: torch.Tensor,
    scale_fp8: torch.Tensor,
    scale_2: torch.Tensor,
) -> torch.Tensor:
    """Batched NVFP4 dequantization for shape (B, rows, cols // 2) -> (B, rows, cols) float32."""
    bsz, rows, half_cols = packed.shape
    cols = half_cols * 2
    lut = torch.tensor(NVFP4_LUT, dtype=torch.float32, device=packed.device)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    unpacked = torch.stack([low, high], dim=-1).view(bsz, rows, cols)
    vals = lut[unpacked.long()].view(bsz, rows, cols // 16, 16)
    scales = scale_fp8.to(torch.float32).unsqueeze(-1) * scale_2.view(bsz, 1, 1, 1).to(torch.float32)
    return (vals * scales).view(bsz, rows, cols)


@torch.no_grad()
def batched_full_spectrum_row_gram_smw_healer(
    targets: torch.Tensor,
    v_rt: torch.Tensor,
    s_rt: torch.Tensor,
    expert_router_vecs: torch.Tensor,
    svd_rank: int = 64,
    alpha_global: float = 6.0,
    alpha_local: float = 8.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    v6.8 Whitened-Spectrum 2560-dim SMW Healer for `gate_up_proj` of shape (B=8, 1280, 2560):
    1. Preserves all 2560 input dimensions via `I_2560 + V_rt diag(g) V_rt^T` (zero subspace blindness).
    2. Normalizes `s_rt` against `s_rt[2]` with a flat `0.30` exponent so capability domains in
       `vh_rt[36:128]` receive full SVD priority above the Marchenko-Pastur noise floor.
    3. Uses fast batched subspace iteration (`torch.svd_lowrank`) + compact r x r re-orthogonalization
       and balanced sqrt(Sigma) scaling into `float8_e4m3fn`.
    """
    _, rows, _ = targets.shape
    s_ref = s_rt[2].clamp_min(1e-8)
    g_glob = alpha_global * (s_rt / s_ref).clamp(max=1.2).pow(0.30).view(1, 1, -1)
    inv_glob = g_glob / (1.0 + g_glob)

    t_proj = torch.matmul(targets, v_rt)
    t_w = targets + torch.matmul(t_proj * g_glob, v_rt.T)

    r_unit = F.normalize(expert_router_vecs, dim=-1).unsqueeze(-1)
    t_loc = torch.bmm(t_w, r_unit)
    t_w = t_w + alpha_local * torch.bmm(t_loc, r_unit.transpose(1, 2))

    q_rank = min(svd_rank + 16, rows)
    u_sub, _, _ = torch.svd_lowrank(t_w, q=q_rank, niter=2)
    u_top = u_sub[:, :, :svd_rank].contiguous()

    v_w = torch.bmm(u_top.transpose(1, 2), t_w)

    inv_loc = alpha_local / (1.0 + alpha_local)
    v_loc_proj = torch.bmm(v_w, r_unit)
    v_unw = v_w - inv_loc * torch.bmm(v_loc_proj, r_unit.transpose(1, 2))

    v_glob_proj = torch.matmul(v_unw, v_rt)
    v_true = v_unw - torch.matmul(v_glob_proj * inv_glob, v_rt.T)

    u_v, s_v, vh_v = torch.linalg.svd(v_true, full_matrices=False)
    u_opt = torch.bmm(u_top, u_v)
    sqrt_s = s_v.clamp_min(1e-8).sqrt().unsqueeze(-1)
    u_bal = (u_opt * sqrt_s.transpose(1, 2)).to(torch.float8_e4m3fn)
    v_bal = (vh_v * sqrt_s).to(torch.float8_e4m3fn)
    h_full = torch.bmm(u_bal.float(), v_bal.float())
    return u_bal, v_bal, h_full


@torch.no_grad()
def batched_exact_gram_smw_healer(
    targets: torch.Tensor,
    v_acts: torch.Tensor,
    s_acts: torch.Tensor,
    svd_rank: int = 64,
    alpha: float = 6.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """v6.8 Whitened-Spectrum Balanced-FP8 Full-Spectrum SMW Healer for `down_proj` (B, 2560, 640)."""
    s_ref = s_acts[:, 2:3].clamp_min(1e-8)
    gain = alpha * (s_acts / s_ref).clamp(max=1.2).pow(0.30).unsqueeze(1)
    inv_gain = gain / (1.0 + gain)
    t_proj = torch.bmm(targets, v_acts)
    t_weighted = targets + torch.bmm(t_proj * gain, v_acts.transpose(1, 2))

    q_rank = min(svd_rank + 16, targets.shape[-1])
    u_sub, s_sub, v_sub = torch.svd_lowrank(t_weighted, q=q_rank, niter=2)
    vh_top = v_sub[:, :, :svd_rank].transpose(1, 2).contiguous()
    u_true = u_sub[:, :, :svd_rank] * s_sub[:, :svd_rank].unsqueeze(1)

    vh_proj = torch.bmm(vh_top, v_acts)
    v_true = vh_top - torch.bmm(vh_proj * inv_gain, v_acts.transpose(1, 2))

    u_u, s_u, vh_u = torch.linalg.svd(u_true, full_matrices=False)
    v_rot = torch.bmm(vh_u, v_true)
    v_norms = torch.linalg.norm(v_rot, dim=-1, keepdim=True).clamp_min(1e-8)
    v_unit = v_rot / v_norms
    s_eff = s_u.unsqueeze(-1) * v_norms
    sqrt_s = s_eff.clamp_min(1e-8).sqrt()
    u_bal = (u_u * sqrt_s.transpose(1, 2)).to(torch.float8_e4m3fn)
    v_bal = (v_unit * sqrt_s).to(torch.float8_e4m3fn)
    h_full = torch.bmm(u_bal.float(), v_bal.float())
    return u_bal, v_bal, h_full


@torch.no_grad()
def pack_tail_batched(
    res_tail: torch.Tensor,
    bits: int,
    block_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched tail lattice packer & unpacker with 2-iteration Least-Squares Optimal Block Scale."""
    bsz, rows, cols = res_tail.shape
    device = res_tail.device
    if bits == 2:
        lut = torch.tensor(LLOYD_MAX_2BIT_LUT, dtype=torch.float32, device=device)
        blocks = res_tail.reshape(bsz, -1, block_size)
        sigma = blocks.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-6, max=448.0)

        for _ in range(2):
            z_blk = blocks / sigma.clamp_min(1e-8)
            codes_blk = torch.zeros_like(z_blk, dtype=torch.uint8)
            codes_blk = torch.where(z_blk >= -0.981599, torch.ones_like(codes_blk), codes_blk)
            codes_blk = torch.where(z_blk >= 0.0, torch.full_like(codes_blk, 2), codes_blk)
            codes_blk = torch.where(z_blk >= 0.981599, torch.full_like(codes_blk, 3), codes_blk)
            q_blk = lut[codes_blk.long()]
            sigma_opt = (blocks * q_blk).sum(dim=-1, keepdim=True) / (q_blk.pow(2).sum(dim=-1, keepdim=True) + 1e-12)
            sigma = sigma_opt.clamp(min=1e-6, max=448.0)

        sigma_fp8 = sigma.to(torch.float8_e4m3fn)
        z = (blocks / sigma_fp8.float().clamp_min(1e-8)).reshape(bsz, rows, cols)

        codes = torch.zeros_like(z, dtype=torch.uint8)
        codes = torch.where(z >= -0.981599, torch.ones_like(codes), codes)
        codes = torch.where(z >= 0.0, torch.full_like(codes, 2), codes)
        codes = torch.where(z >= 0.981599, torch.full_like(codes, 3), codes)

        c0 = codes[:, :, 0::4]
        c1 = codes[:, :, 1::4] << 2
        c2 = codes[:, :, 2::4] << 4
        c3 = codes[:, :, 3::4] << 6
        packed_u8 = (c0 | c1 | c2 | c3).to(torch.uint8)
        scales_out = sigma_fp8.view(bsz, rows, cols // block_size)

        vals = lut[codes.long()].view(bsz, rows, cols // block_size, block_size)
        deq = (vals * scales_out.float().unsqueeze(-1)).view(bsz, rows, cols)
        return packed_u8, scales_out, deq

    if bits == 16:
        blocks = res_tail.reshape(bsz, -1, block_size)
        sigma = blocks.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-6, max=448.0)
        sigma_fp8 = sigma.to(torch.float8_e4m3fn)
        z = (blocks / sigma_fp8.float().clamp_min(1e-8)).reshape(bsz, rows, cols)

        trits = torch.ones_like(z, dtype=torch.uint8)
        trits = torch.where(z <= -0.6120, torch.zeros_like(trits), trits)
        trits = torch.where(z >= 0.6120, torch.full_like(trits, 2), trits)

        t0 = trits[:, :, 0::5]
        t1 = trits[:, :, 1::5] * 3
        t2 = trits[:, :, 2::5] * 9
        t3 = trits[:, :, 3::5] * 27
        t4 = trits[:, :, 4::5] * 81
        packed_u8 = (t0 + t1 + t2 + t3 + t4).to(torch.uint8)
        scales_out = sigma_fp8.view(bsz, rows, cols // block_size)

        vals = (trits.float() - 1.0) * 1.2240
        deq = (vals.view(bsz, rows, cols // block_size, block_size) * scales_out.float().unsqueeze(-1)).view(bsz, rows, cols)
        return packed_u8, scales_out, deq

    # 1-bit Rademacher (block_size = 64)
    b64 = 64
    blocks = res_tail.reshape(bsz, -1, b64)
    sigma = (blocks.pow(2).mean(dim=-1, keepdim=True).sqrt() * 0.79788456).clamp(min=1e-6, max=448.0)
    sigma_fp8 = sigma.to(torch.float8_e4m3fn)
    bit_vals = (res_tail >= 0.0).to(torch.uint8)

    packed_u8 = torch.zeros((bsz, rows, cols // 8), dtype=torch.uint8, device=device)
    for b in range(8):
        packed_u8 |= bit_vals[:, :, b::8] << b
    scales_out = sigma_fp8.view(bsz, rows, cols // b64)
    signs = bit_vals.float() * 2.0 - 1.0
    deq = (signs.view(bsz, rows, cols // b64, b64) * scales_out.float().unsqueeze(-1)).view(bsz, rows, cols)
    return packed_u8, scales_out, deq


@torch.no_grad()
def convert_expert_group_batched(
    group_idx: int,
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
    router_group_w: torch.Tensor,
    x_cal: torch.Tensor,
    v_rt_256: torch.Tensor,
    s_rt_256: torch.Tensor,
    h128: torch.Tensor,
    svd_rank: int = 48,
    gu_tail_bits: int = 2,
    down_tail_bits: int = 2,
) -> dict[str, torch.Tensor]:
    """Batched GPU conversion of K=8 experts (`gate_w`: (8, 640, 2560), `down_w`: (8, 2560, 640))."""
    m_cnt = gate_w.shape[0]
    device = gate_w.device
    gu_experts = torch.cat([gate_w, up_w], dim=1)  # (8, 1280, 2560)

    gu_keys = [
        generate_expert_1d_keys(1280, 2560, m, seed_base=101 + group_idx * 8, device=device)
        for m in range(m_cnt)
    ]
    _, _, init_healers_gu = batched_full_spectrum_row_gram_smw_healer(
        gu_experts, v_rt_256, s_rt_256, router_group_w, svd_rank=svd_rank
    )

    s_gu_accum = torch.zeros((1280, 2560), device=device, dtype=torch.float32)
    smooth_gu = gu_experts - init_healers_gu
    for m in range(m_cnt):
        r_s1, r_perm, _, r_s2, c_s1, c_perm, _, c_s2 = gu_keys[m]
        s_gu_accum.add_(fold_2d_permuted_hadamard(smooth_gu[m], r_s1, r_perm, r_s2, c_s1, c_perm, c_s2, h128))

    s_gu_packed, s_gu_scale, s_gu_scale2 = pack_nvfp4(s_gu_accum)
    s_gu_deq = unpack_nvfp4(s_gu_packed, s_gu_scale, s_gu_scale2)

    h_unbind_gu = torch.stack(
        [
            unbind_2d_permuted_hadamard(s_gu_deq, *gu_keys[m][0:1], gu_keys[m][2], gu_keys[m][3], gu_keys[m][4], gu_keys[m][6], gu_keys[m][7], h128)
            for m in range(m_cnt)
        ],
        dim=0,
    )  # (8, 1280, 2560)

    x_sub = x_cal  # all 512 exact normalized router centroids
    y_tail_gu = torch.matmul(x_sub.unsqueeze(0), smooth_gu.transpose(1, 2))
    y_holo_gu = torch.matmul(x_sub.unsqueeze(0), h_unbind_gu.transpose(1, 2))
    beta_gu = (y_tail_gu * y_holo_gu).sum(dim=(1, 2)) / ((y_holo_gu * y_holo_gu).sum(dim=(1, 2)) + 1e-12)

    res_gu = gu_experts - beta_gu.view(m_cnt, 1, 1) * h_unbind_gu
    tail_target_gu = res_gu - init_healers_gu
    gu_tail_packed, gu_tail_scale, e_gu_deq = pack_tail_batched(
        tail_target_gu, bits=gu_tail_bits, block_size=32
    )

    u_gu, v_gu, h_gu_final = batched_full_spectrum_row_gram_smw_healer(
        res_gu - e_gu_deq, v_rt_256, s_rt_256, router_group_w, svd_rank=svd_rank
    )

    rem_after_gu = gu_experts - (h_gu_final + e_gu_deq)
    y_rem_gu = torch.matmul(x_sub.unsqueeze(0), rem_after_gu.transpose(1, 2))
    beta_gu = (y_rem_gu * y_holo_gu).sum(dim=(1, 2)) / ((y_holo_gu * y_holo_gu).sum(dim=(1, 2)) + 1e-12)

    gu_hat_raw = beta_gu.view(m_cnt, 1, 1) * h_unbind_gu + h_gu_final + e_gu_deq
    z_true_gu = torch.matmul(x_sub.unsqueeze(0), gu_experts.transpose(1, 2))
    z_hat_gu = torch.matmul(x_sub.unsqueeze(0), gu_hat_raw.transpose(1, 2))
    gamma_gu = (
        torch.linalg.norm(z_true_gu.reshape(m_cnt, -1), dim=-1)
        / (torch.linalg.norm(z_hat_gu.reshape(m_cnt, -1), dim=-1) + 1e-12)
    ).clamp(0.98, 1.12)

    beta_gu = beta_gu * gamma_gu
    u_gu = (u_gu.float() * gamma_gu.view(m_cnt, 1, 1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    gu_tail_scale = (gu_tail_scale.float() * gamma_gu.view(m_cnt, 1, 1)).clamp(1e-6, 448.0).to(torch.float8_e4m3fn)
    h_gu_final = torch.bmm(u_gu.float(), v_gu.float())
    e_gu_deq = e_gu_deq * gamma_gu.view(m_cnt, 1, 1)

    gu_hat = beta_gu.view(m_cnt, 1, 1) * h_unbind_gu + h_gu_final + e_gu_deq
    rec_g = gu_hat[:, :640, :]
    rec_u = gu_hat[:, 640:, :]

    # 2. Expert-Affinity Weighted SwiGLU Calibration + MP-Denoised Post-Tail SMW Healer for down_proj (8, 2560, 640)
    aff_logits = torch.matmul(router_group_w, x_sub.T)
    aff_weights = (F.softmax(aff_logits * 2.0, dim=-1) * float(x_sub.shape[0])).clamp(min=0.25, max=6.0).sqrt().unsqueeze(-1)

    h_true_c = (F.silu(torch.matmul(x_sub.unsqueeze(0), gate_w.transpose(1, 2))) * torch.matmul(x_sub.unsqueeze(0), up_w.transpose(1, 2))) * aff_weights
    h_hat_c = (F.silu(torch.matmul(x_sub.unsqueeze(0), rec_g.transpose(1, 2))) * torch.matmul(x_sub.unsqueeze(0), rec_u.transpose(1, 2))) * aff_weights

    _, s_h_all, vh_h_all = torch.linalg.svd(h_hat_c, full_matrices=False)
    v_h_batch = vh_h_all[:, :256, :].transpose(1, 2).contiguous()
    s_h_batch = s_h_all[:, :256].contiguous()

    y_true_d = torch.bmm(h_true_c, down_w.transpose(1, 2))
    u_hm = vh_h_all[:, :48, :].transpose(1, 2).contiguous()
    hm_sub = torch.bmm(h_hat_c, u_hm)
    delta_y = y_true_d - torch.bmm(h_hat_c, down_w.transpose(1, 2))
    gram_h = torch.bmm(hm_sub.transpose(1, 2), hm_sub)
    diag_mean = torch.diagonal(gram_h, dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).unsqueeze(-1)
    reg_eye = 0.20 * diag_mean * torch.eye(48, device=device).unsqueeze(0)
    delta_w = torch.bmm(torch.bmm(delta_y.transpose(1, 2), hm_sub), torch.linalg.inv(gram_h + reg_eye))
    down_targets = down_w + 0.30 * torch.bmm(delta_w, u_hm.transpose(1, 2))

    _, _, init_healers_down = batched_exact_gram_smw_healer(down_w, v_h_batch, s_h_batch, svd_rank=svd_rank)

    down_keys = [
        generate_expert_1d_keys(2560, 640, m, seed_base=303 + group_idx * 8, device=device)
        for m in range(m_cnt)
    ]
    s_down_accum = torch.zeros((2560, 640), device=device, dtype=torch.float32)
    smooth_down = down_w - init_healers_down
    for m in range(m_cnt):
        dr_s1, dr_perm, _, dr_s2, dc_s1, dc_perm, _, dc_s2 = down_keys[m]
        s_down_accum.add_(fold_2d_permuted_hadamard(smooth_down[m], dr_s1, dr_perm, dr_s2, dc_s1, dc_perm, dc_s2, h128))

    s_down_packed, s_down_scale, s_down_scale2 = pack_nvfp4(s_down_accum)
    s_down_deq = unpack_nvfp4(s_down_packed, s_down_scale, s_down_scale2)

    h_unbind_down = torch.stack(
        [
            unbind_2d_permuted_hadamard(s_down_deq, *down_keys[m][0:1], down_keys[m][2], down_keys[m][3], down_keys[m][4], down_keys[m][6], down_keys[m][7], h128)
            for m in range(m_cnt)
        ],
        dim=0,
    )

    rem_t_down = down_w - init_healers_down
    y_tail_d = torch.bmm(h_hat_c, rem_t_down.transpose(1, 2))
    y_holo_d = torch.bmm(h_hat_c, h_unbind_down.transpose(1, 2))
    beta_down = (y_tail_d * y_holo_d).sum(dim=(1, 2)) / ((y_holo_d * y_holo_d).sum(dim=(1, 2)) + 1e-12)

    res_down_clean = down_w - beta_down.view(m_cnt, 1, 1) * h_unbind_down
    tail_target_d = res_down_clean - init_healers_down
    down_tail_packed, down_tail_scale, e_down_deq = pack_tail_batched(
        tail_target_d, bits=down_tail_bits, block_size=32
    )

    u_down, v_down, h_down_final = batched_exact_gram_smw_healer(
        (down_targets - beta_down.view(m_cnt, 1, 1) * h_unbind_down) - e_down_deq,
        v_h_batch,
        s_h_batch,
        svd_rank=svd_rank,
    )
    rem_after_d = down_targets - (h_down_final + e_down_deq)
    y_rem_d = torch.bmm(h_hat_c, rem_after_d.transpose(1, 2))
    beta_down = (y_rem_d * y_holo_d).sum(dim=(1, 2)) / ((y_holo_d * y_holo_d).sum(dim=(1, 2)) + 1e-12)

    down_hat_raw = beta_down.view(m_cnt, 1, 1) * h_unbind_down + h_down_final + e_down_deq
    y_hat_d = torch.bmm(h_hat_c, down_hat_raw.transpose(1, 2))
    gamma_down = (
        torch.linalg.norm(y_true_d.reshape(m_cnt, -1), dim=-1)
        / (torch.linalg.norm(y_hat_d.reshape(m_cnt, -1), dim=-1) + 1e-12)
    ).clamp(0.98, 1.12)

    beta_down = beta_down * gamma_down
    u_down = (u_down.float() * gamma_down.view(m_cnt, 1, 1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    down_tail_scale = (down_tail_scale.float() * gamma_down.view(m_cnt, 1, 1)).clamp(1e-6, 448.0).to(torch.float8_e4m3fn)

    return {
        "s_gu.weight": s_gu_packed.contiguous().cpu(),
        "s_gu.weight_scale": s_gu_scale.contiguous().cpu(),
        "s_gu.weight_scale_2": s_gu_scale2.contiguous().cpu(),
        "s_gu.beta": beta_gu.contiguous().cpu().to(torch.float32),
        "s_down.weight": s_down_packed.contiguous().cpu(),
        "s_down.weight_scale": s_down_scale.contiguous().cpu(),
        "s_down.weight_scale_2": s_down_scale2.contiguous().cpu(),
        "s_down.beta": beta_down.contiguous().cpu().to(torch.float32),
        "healer.u_gu": u_gu.contiguous().cpu(),
        "healer.v_gu": v_gu.contiguous().cpu(),
        "healer.u_down": u_down.contiguous().cpu(),
        "healer.v_down": v_down.contiguous().cpu(),
        "tail.gu_bits": torch.tensor(gu_tail_bits, dtype=torch.int32),
        "tail.gu_packed": gu_tail_packed.contiguous().cpu(),
        "tail.gu_scale": gu_tail_scale.contiguous().cpu(),
        "tail.down_bits": torch.tensor(down_tail_bits, dtype=torch.int32),
        "tail.down_packed": down_tail_packed.contiguous().cpu(),
        "tail.down_scale": down_tail_scale.contiguous().cpu(),
    }


def convert_single_layer(
    l_idx: int,
    idx_map: dict[str, str],
    out_dir: Path,
    device: torch.device,
    h128: torch.Tensor,
    gu_bits: int = 2,
    down_bits: int = 2,
    svd_rank: int = 48,
    force: bool = False,
) -> tuple[str, float, float]:
    shard_name = f"chronos-layer-{l_idx:05d}-of-00048.safetensors"
    shard_path = out_dir / shard_name
    if shard_path.is_file() and not force:
        mb = shard_path.stat().st_size / (1024 * 1024)
        return shard_name, mb, 0.0

    t0 = time.perf_counter()
    prefix = f"model.language_model.layers.{l_idx}.mlp"
    st_cache: dict[str, safe_open] = {}

    def get_t(key: str) -> torch.Tensor:
        sf = idx_map[key]
        if sf not in st_cache:
            st_cache[sf] = safe_open(str(MODEL_DIR / sf), framework="pt", device="cpu")
        return st_cache[sf].get_tensor(key)

    router_w = get_t(f"{prefix}.gate.weight").to(device=device, dtype=torch.float32)
    _, s_rt, vh_rt = torch.linalg.svd(router_w, full_matrices=False)
    v_rt_256 = vh_rt[:256, :].T.contiguous()
    s_rt_256 = s_rt[:256].contiguous()
    x_cal = F.rms_norm(router_w, (2560,), eps=1e-6)

    layer_shard_dict: dict[str, torch.Tensor] = {
        f"{prefix}.gate.weight": get_t(f"{prefix}.gate.weight"),
        f"{prefix}.shared_expert_gate.weight": get_t(f"{prefix}.shared_expert_gate.weight"),
        f"{prefix}.shared_expert.down_proj.weight": get_t(f"{prefix}.shared_expert.down_proj.weight"),
        f"{prefix}.shared_expert.down_proj.weight_scale": get_t(f"{prefix}.shared_expert.down_proj.weight_scale"),
        f"{prefix}.shared_expert.gate_proj.weight": get_t(f"{prefix}.shared_expert.gate_proj.weight"),
        f"{prefix}.shared_expert.gate_proj.weight_scale": get_t(f"{prefix}.shared_expert.gate_proj.weight_scale"),
        f"{prefix}.shared_expert.up_proj.weight": get_t(f"{prefix}.shared_expert.up_proj.weight"),
        f"{prefix}.shared_expert.up_proj.weight_scale": get_t(f"{prefix}.shared_expert.up_proj.weight_scale"),
    }

    for g_idx in range(64):
        e_start = g_idx * 8
        gp_list, gs_list, gs2_list = [], [], []
        up_list, us_list, us2_list = [], [], []
        dp_list, ds_list, ds2_list = [], [], []
        for e_id in range(e_start, e_start + 8):
            ep = f"{prefix}.experts.{e_id}"
            gp_list.append(get_t(f"{ep}.gate_proj.weight"))
            gs_list.append(get_t(f"{ep}.gate_proj.weight_scale"))
            gs2_list.append(get_t(f"{ep}.gate_proj.weight_scale_2"))
            up_list.append(get_t(f"{ep}.up_proj.weight"))
            us_list.append(get_t(f"{ep}.up_proj.weight_scale"))
            us2_list.append(get_t(f"{ep}.up_proj.weight_scale_2"))
            dp_list.append(get_t(f"{ep}.down_proj.weight"))
            ds_list.append(get_t(f"{ep}.down_proj.weight_scale"))
            ds2_list.append(get_t(f"{ep}.down_proj.weight_scale_2"))

        gate_w = unpack_nvfp4_batched(
            torch.stack(gp_list).to(device), torch.stack(gs_list).to(device), torch.stack(gs2_list).to(device)
        )
        up_w = unpack_nvfp4_batched(
            torch.stack(up_list).to(device), torch.stack(us_list).to(device), torch.stack(us2_list).to(device)
        )
        down_w = unpack_nvfp4_batched(
            torch.stack(dp_list).to(device), torch.stack(ds_list).to(device), torch.stack(ds2_list).to(device)
        )

        grp_tensors = convert_expert_group_batched(
            group_idx=g_idx,
            gate_w=gate_w,
            up_w=up_w,
            down_w=down_w,
            router_group_w=router_w[e_start : e_start + 8],
            x_cal=x_cal,
            v_rt_256=v_rt_256,
            s_rt_256=s_rt_256,
            h128=h128,
            svd_rank=svd_rank,
            gu_tail_bits=gu_bits,
            down_tail_bits=down_bits,
        )
        gp = f"{prefix}.chronos_groups.{g_idx}"
        for k, v in grp_tensors.items():
            layer_shard_dict[f"{gp}.{k}"] = v

    tmp_path = out_dir / f"{shard_name}.tmp"
    save_file(layer_shard_dict, str(tmp_path))
    tmp_path.replace(shard_path)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    mb = shard_path.stat().st_size / (1024 * 1024)
    return shard_name, mb, dt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument("--end-layer", type=int, default=47)
    parser.add_argument("--layer-step", type=int, default=1)
    parser.add_argument("--gu-bits", type=int, default=2, help="1=1-bit Compact (34.5GB), 16=Ternary (39GB), 2=Ultra (45.0GB)")
    parser.add_argument("--down-bits", type=int, default=2)
    parser.add_argument("--svd-rank", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-compose", action="store_true")
    parser.add_argument("--out-dir", default=str(OUT_CKPT_DIR))
    args = parser.parse_args()

    device = torch.device("cuda:0")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx_map = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())["weight_map"]
    h128 = build_hadamard_128(device)

    print("=" * 116)
    print(f"BATCHED GPU STREAMING CONVERSION (v6.8): LAYERS {args.start_layer}..{args.end_layer} (step={args.layer_step}) -> {out_dir}")
    print("=" * 116)

    converted_map: dict[int, str] = {}
    total_mb = 0.0
    t_all = time.perf_counter()

    for l_idx in range(args.start_layer, args.end_layer + 1, args.layer_step):
        sname, mb, dt = convert_single_layer(
            l_idx=l_idx,
            idx_map=idx_map,
            out_dir=out_dir,
            device=device,
            h128=h128,
            gu_bits=args.gu_bits,
            down_bits=args.down_bits,
            svd_rank=args.svd_rank,
            force=args.force,
        )
        converted_map[l_idx] = sname
        total_mb += mb
        elapsed_total = time.perf_counter() - t_all
        print(
            f"[Layer {l_idx:02d}/47] {sname} | Size: {mb:6.2f} MB | Layer Time: {dt:5.2f}s | Total Elapsed: {elapsed_total:6.1f}s",
            flush=True,
        )

    # Discover all existing converted shards in out_dir so partial/resumed runs include all converted layers
    for existing_l in range(48):
        cand = f"chronos-layer-{existing_l:05d}-of-00048.safetensors"
        if (out_dir / cand).is_file():
            converted_map[existing_l] = cand

    if not args.skip_compose:
        compose_chronos_checkpoint_dir(MODEL_DIR, out_dir, converted_map)
    print("=" * 116)
    print(
        f"[COMPLETE] Converted {len(converted_map)}/48 layers in {out_dir} "
        f"(Active converted MoE disk size: {total_mb / 1024.0:.2f} GiB vs 63.28 GiB stock)"
    )
    print("=" * 116)


if __name__ == "__main__":
    main()
