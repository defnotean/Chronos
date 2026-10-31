#!/usr/bin/env python3
"""
Chronos-1T x PseudoBrain (pb27) — Diagnostic & Fix Verification for Subspace Blindness in `gate_up_proj`.

Tests why `v6.2` dropped on `MATH_FORMAL_PROOF`, `CRISPR_GENOMICS`, and `ADVERSARIAL_8TOK_CHURN` (`vh_rt[48:]`)
and proves the `v6.3` Full-Spectrum SMW Row-Gram + Balanced FP8 + Variance-Restoring Tail fix on Layer 0
across all 6 capability domains before updating the 48-layer checkpoint.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.full_48layer_capability_suite import (
     build_6domain_hidden_stream,
    evaluate_logits_metrics,
    execute_routed_moe_grouped,
)
from chronos_1t.converter.convert_all_48_layers import (
    batched_exact_gram_smw_healer,
    pack_tail_batched,
    unpack_nvfp4_batched,
)
from chronos_1t.models.chronos_moe_block import (
    ChronosHolographicExpertGroup,
    build_hadamard_128,
    fold_2d_permuted_hadamard,
    generate_expert_1d_keys,
    pack_nvfp4,
    unbind_2d_permuted_hadamard,
    unpack_nvfp4,
)
from chronos_1t.models.vllm_chronos_moe import dequantize_mxfp8

STOCK_DIR = Path("/home/defnotean/pb27/hf/flash_next_nvfp4")
CHRONOS_DIR = Path("/home/defnotean/pb27/hf/flash_next_chronos_v6")


@torch.no_grad()
def batched_whitened_subspace_smw_healer_gu(
    targets: torch.Tensor,
    v_rt: torch.Tensor,
    s_rt: torch.Tensor,
    expert_router_vecs: torch.Tensor,
    svd_rank: int = 48,
    alpha_global: float = 6.0,
    alpha_local: float = 8.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    v6.8 Whitened-Spectrum 2560-dim SMW Healer for `gate_up_proj` (B=8, 1280, 2560):
    Normalizes singular weights against `s_rt[2]` (ignoring the rank-1 mean hub outlier `s_rt[0]`)
    with a flat `0.30` exponent so capability domains in `vh_rt[36:128]` receive full SVD priority.
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
def batched_whitened_smw_healer_down(
    targets: torch.Tensor,
    v_acts: torch.Tensor,
    s_acts: torch.Tensor,
    svd_rank: int = 48,
    alpha: float = 6.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """v6.8 Whitened-Spectrum Balanced-FP8 SMW Healer for `down_proj` (B, 2560, 640)."""
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
def convert_group_v68(
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
    """
    v6.8 Subspace-Filtered Post-Tail SVD Healer + Whitened Spectrum + Per-Expert Affinity Calibration:
    1. Filters the 2-bit quantization noise `delta_tail` onto `v_rt_256` before Post-Tail SVD healing
       so Marchenko-Pastur bulk noise in the 2,304 dead dimensions never steals rank from `init_healers`!
    2. Uses whitened spectrum boosting (`s_rt / s_rt[2]`) so domains at `vh_rt[36:128]` get full healing.
    """
    device = gate_w.device
    m_cnt = 8
    gu_w = torch.cat([gate_w, up_w], dim=1)

    _, _, init_healers_gu = batched_whitened_subspace_smw_healer_gu(
        gu_w, v_rt_256, s_rt_256, router_group_w, svd_rank=svd_rank
    )

    gu_keys = [
        generate_expert_1d_keys(1280, 2560, m, seed_base=101 + group_idx * 8, device=device)
        for m in range(m_cnt)
    ]
    s_gu_accum = torch.zeros((1280, 2560), device=device, dtype=torch.float32)
    smooth_gu = gu_w - init_healers_gu
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
    )

    x_sub = x_cal
    y_tail_gu = torch.matmul(x_sub.unsqueeze(0), smooth_gu.transpose(1, 2))
    y_holo_gu = torch.matmul(x_sub.unsqueeze(0), h_unbind_gu.transpose(1, 2))
    beta_gu = (y_tail_gu * y_holo_gu).sum(dim=(1, 2)) / ((y_holo_gu * y_holo_gu).sum(dim=(1, 2)) + 1e-12)

    res_gu = gu_w - beta_gu.view(m_cnt, 1, 1) * h_unbind_gu
    tail_target_gu = res_gu - init_healers_gu
    gu_tail_packed, gu_tail_scale, e_gu_deq = pack_tail_batched_lsq(tail_target_gu, bits=gu_tail_bits, block_size=32)

    u_gu, v_gu, h_gu_final = batched_whitened_subspace_smw_healer_gu(
        res_gu - e_gu_deq, v_rt_256, s_rt_256, router_group_w, svd_rank=svd_rank
    )

    rem_after_gu = gu_w - (h_gu_final + e_gu_deq)
    y_rem_gu = torch.matmul(x_sub.unsqueeze(0), rem_after_gu.transpose(1, 2))
    beta_gu = (y_rem_gu * y_holo_gu).sum(dim=(1, 2)) / ((y_holo_gu * y_holo_gu).sum(dim=(1, 2)) + 1e-12)

    gu_hat_raw = beta_gu.view(m_cnt, 1, 1) * h_unbind_gu + h_gu_final + e_gu_deq
    z_true_gu = torch.matmul(x_sub.unsqueeze(0), gu_w.transpose(1, 2))
    z_hat_gu = torch.matmul(x_sub.unsqueeze(0), gu_hat_raw.transpose(1, 2))
    # Restore exact pre-SwiGLU variance so silu(z) operates at the exact true operating point!
    gamma_gu = (
        torch.linalg.norm(z_true_gu.reshape(m_cnt, -1), dim=-1)
        / (torch.linalg.norm(z_hat_gu.reshape(m_cnt, -1), dim=-1) + 1e-12)
    ).clamp(0.98, 1.12)

    beta_gu = beta_gu * gamma_gu
    u_gu = (u_gu.float() * gamma_gu.view(m_cnt, 1, 1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    gu_tail_scale = (gu_tail_scale.float() * gamma_gu.view(m_cnt, 1, 1)).clamp(1e-6, 448.0).to(torch.float8_e4m3fn)

    h_gu_final = torch.bmm(u_gu.float(), v_gu.float())
    lut_2b = torch.tensor([-1.510417, -0.452780, 0.452780, 1.510417], dtype=torch.float32, device=device)
    c0 = gu_tail_packed & 0x03
    c1 = (gu_tail_packed >> 2) & 0x03
    c2 = (gu_tail_packed >> 4) & 0x03
    c3 = (gu_tail_packed >> 6) & 0x03
    codes_gu = torch.stack([c0, c1, c2, c3], dim=-1).view(m_cnt, 1280, 2560)
    e_gu_deq = (lut_2b[codes_gu.long()].view(m_cnt, 1280, 80, 32) * gu_tail_scale.float().unsqueeze(-1)).view(m_cnt, 1280, 2560)

    gu_hat = beta_gu.view(m_cnt, 1, 1) * h_unbind_gu + h_gu_final + e_gu_deq
    rec_g = gu_hat[:, :640, :]
    rec_u = gu_hat[:, 640:, :]

    # 2. Expert-Affinity Weighted SwiGLU Calibration for `down_proj`
    # Weight calibration points by how strongly each calibration vector activates each expert's router!
    aff_logits = torch.matmul(router_group_w, x_sub.T)  # (8, 512)
    aff_weights = (F.softmax(aff_logits * 2.0, dim=-1) * 512.0).clamp(min=0.25, max=6.0).sqrt().unsqueeze(-1)  # (8, 512, 1)

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

    _, _, init_healers_down = batched_whitened_smw_healer_down(
        down_w, v_h_batch, s_h_batch, svd_rank=svd_rank
    )

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
    down_tail_packed, down_tail_scale, e_down_deq = pack_tail_batched_lsq(
        tail_target_d, bits=down_tail_bits, block_size=32
    )

    u_down, v_down, h_down_final = batched_whitened_smw_healer_down(
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
        "s_gu.weight": s_gu_packed.contiguous(),
        "s_gu.weight_scale": s_gu_scale.contiguous(),
        "s_gu.weight_scale_2": s_gu_scale2.contiguous(),
        "s_gu.beta": beta_gu.contiguous().to(torch.float32),
        "s_down.weight": s_down_packed.contiguous(),
        "s_down.weight_scale": s_down_scale.contiguous(),
        "s_down.weight_scale_2": s_down_scale2.contiguous(),
        "s_down.beta": beta_down.contiguous().to(torch.float32),
        "healer.u_gu": u_gu.contiguous(),
        "healer.v_gu": v_gu.contiguous(),
        "healer.u_down": u_down.contiguous(),
        "healer.v_down": v_down.contiguous(),
        "tail.gu_bits": torch.tensor(gu_tail_bits, dtype=torch.int32),
        "tail.gu_packed": gu_tail_packed.contiguous(),
        "tail.gu_scale": gu_tail_scale.contiguous(),
        "tail.down_bits": torch.tensor(down_tail_bits, dtype=torch.int32),
        "tail.down_packed": down_tail_packed.contiguous(),
        "tail.down_scale": down_tail_scale.contiguous(),
    }


@torch.no_grad()
def pack_tail_batched_lsq(
    res_tail: torch.Tensor,
    bits: int,
    block_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""2-Iteration Least-Squares Optimal Block Scale Tail Lattice ($\sigma^* = <e, q> / ||q||^2$)."""
    if bits != 2:
        return pack_tail_batched(res_tail, bits=bits, block_size=block_size)
    bsz, rows, cols = res_tail.shape
    device = res_tail.device
    lut = torch.tensor([-1.510417, -0.452780, 0.452780, 1.510417], dtype=torch.float32, device=device)

    blocks = res_tail.reshape(bsz, -1, block_size)
    sigma = blocks.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-6, max=448.0)

    # 2-pass alternating Lloyd-Max + exact least-squares scale projection
    for _ in range(2):
        z = blocks / sigma.clamp_min(1e-8)
        codes_blk = torch.zeros_like(z, dtype=torch.uint8)
        codes_blk = torch.where(z >= -0.981599, torch.ones_like(codes_blk), codes_blk)
        codes_blk = torch.where(z >= 0.0, torch.full_like(codes_blk, 2), codes_blk)
        codes_blk = torch.where(z >= 0.981599, torch.full_like(codes_blk, 3), codes_blk)
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


@torch.no_grad()
def convert_group_v63(
    group_idx: int,
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
    router_group_w: torch.Tensor,
    x_cal: torch.Tensor,
    v_rt_256: torch.Tensor,
    s_rt_256: torch.Tensor,
    h128: torch.Tensor,
    svd_rank: int = 32,
    gu_tail_bits: int = 1,
    down_tail_bits: int = 2,
    alpha_glob: float = 3.0,
    alpha_loc: float = 4.0,
    alpha_down: float = 4.0,
    drift_weight: float = 0.25,
) -> dict[str, torch.Tensor]:
    device = gate_w.device
    m_cnt = 8
    gu_w = torch.cat([gate_w, up_w], dim=1)  # (8, 1280, 2560)

    # 1. Full-Spectrum SMW Row-Gram Healer for gate_up_proj
    _, _, init_healers_gu = batched_full_spectrum_row_gram_smw_healer(
        gu_w, v_rt_256, s_rt_256, router_group_w, svd_rank=svd_rank, alpha_global=alpha_glob, alpha_local=alpha_loc
    )

    gu_keys = [
        generate_expert_1d_keys(1280, 2560, m, seed_base=101 + group_idx * 8, device=device)
        for m in range(m_cnt)
    ]
    s_gu_accum = torch.zeros((1280, 2560), device=device, dtype=torch.float32)
    smooth_gu = gu_w - init_healers_gu
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
    )

    x_sub = x_cal  # all 512 exact normalized router centroids!
    rem_gu = gu_w - init_healers_gu
    y_tail_gu = torch.matmul(x_sub.unsqueeze(0), rem_gu.transpose(1, 2))
    y_holo_gu = torch.matmul(x_sub.unsqueeze(0), h_unbind_gu.transpose(1, 2))
    beta_gu = (y_tail_gu * y_holo_gu).sum(dim=(1, 2)) / ((y_holo_gu * y_holo_gu).sum(dim=(1, 2)) + 1e-12)

    res_gu = gu_w - beta_gu.view(m_cnt, 1, 1) * h_unbind_gu
    # Notice: res_gu - init_healers_gu == smooth_gu - beta_gu * h_unbind_gu (already spike-free, zero extra SVD call!)
    gu_tail_packed, gu_tail_scale, e_gu_deq = pack_tail_batched_lsq(res_gu - init_healers_gu, bits=gu_tail_bits, block_size=32)

    # Fit final high-precision FP8 SVD Healer AFTER tail lattice so it heals the tail quantization error in router subspace!
    u_gu, v_gu, h_gu_final = batched_full_spectrum_row_gram_smw_healer(
        res_gu - e_gu_deq, v_rt_256, s_rt_256, router_group_w, svd_rank=svd_rank, alpha_global=alpha_glob, alpha_local=alpha_loc
    )

    # Post-Tail FP32 beta_gu refinement (zero FP8 rounding penalty!)
    rem_after_gu = gu_w - (h_gu_final + e_gu_deq)
    y_rem_gu = torch.matmul(x_sub.unsqueeze(0), rem_after_gu.transpose(1, 2))
    beta_gu = (y_rem_gu * y_holo_gu).sum(dim=(1, 2)) / ((y_holo_gu * y_holo_gu).sum(dim=(1, 2)) + 1e-12)

    gu_hat_raw = beta_gu.view(m_cnt, 1, 1) * h_unbind_gu + h_gu_final + e_gu_deq
    z_true_gu = torch.matmul(x_sub.unsqueeze(0), gu_w.transpose(1, 2))
    z_hat_gu = torch.matmul(x_sub.unsqueeze(0), gu_hat_raw.transpose(1, 2))
    # Restore exact pre-SwiGLU variance so silu(z) operates at the exact true operating point!
    gamma_gu = (
        torch.linalg.norm(z_true_gu.reshape(m_cnt, -1), dim=-1)
        / (torch.linalg.norm(z_hat_gu.reshape(m_cnt, -1), dim=-1) + 1e-12)
    ).clamp(0.98, 1.12)

    beta_gu = beta_gu * gamma_gu
    u_gu = (u_gu.float() * gamma_gu.view(m_cnt, 1, 1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    gu_tail_scale = (gu_tail_scale.float() * gamma_gu.view(m_cnt, 1, 1)).clamp(1e-6, 448.0).to(torch.float8_e4m3fn)

    h_gu_final = torch.bmm(u_gu.float(), v_gu.float())
    lut_2b = torch.tensor([-1.510417, -0.452780, 0.452780, 1.510417], dtype=torch.float32, device=device)
    c0 = gu_tail_packed & 0x03
    c1 = (gu_tail_packed >> 2) & 0x03
    c2 = (gu_tail_packed >> 4) & 0x03
    c3 = (gu_tail_packed >> 6) & 0x03
    codes_gu = torch.stack([c0, c1, c2, c3], dim=-1).view(m_cnt, 1280, 2560)
    e_gu_deq = (lut_2b[codes_gu.long()].view(m_cnt, 1280, 80, 32) * gu_tail_scale.float().unsqueeze(-1)).view(m_cnt, 1280, 2560)

    gu_hat = beta_gu.view(m_cnt, 1, 1) * h_unbind_gu + h_gu_final + e_gu_deq
    rec_g = gu_hat[:, :640, :]
    rec_u = gu_hat[:, 640:, :]

    # 2. Coupled SwiGLU Drift + Balanced Exact Gram SMW Healer for down_proj
    h_true_c = F.silu(torch.matmul(x_sub.unsqueeze(0), gate_w.transpose(1, 2))) * torch.matmul(x_sub.unsqueeze(0), up_w.transpose(1, 2))
    h_hat_c = F.silu(torch.matmul(x_sub.unsqueeze(0), rec_g.transpose(1, 2))) * torch.matmul(x_sub.unsqueeze(0), rec_u.transpose(1, 2))

    _, s_h_all, vh_h_all = torch.linalg.svd(h_hat_c, full_matrices=False)
    v_h_batch = vh_h_all[:, :192, :].transpose(1, 2).contiguous()
    s_h_batch = s_h_all[:, :192].contiguous()

    y_true_d = torch.bmm(h_true_c, down_w.transpose(1, 2))
    u_hm = vh_h_all[:, :32, :].transpose(1, 2).contiguous()
    hm_sub = torch.bmm(h_hat_c, u_hm)
    delta_y = y_true_d - torch.bmm(h_hat_c, down_w.transpose(1, 2))
    gram_h = torch.bmm(hm_sub.transpose(1, 2), hm_sub)
    diag_mean = torch.diagonal(gram_h, dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).unsqueeze(-1)
    reg_eye = 0.25 * diag_mean * torch.eye(32, device=device).unsqueeze(0)
    delta_w = torch.bmm(torch.bmm(delta_y.transpose(1, 2), hm_sub), torch.linalg.inv(gram_h + reg_eye))
    down_drift = drift_weight * torch.bmm(delta_w, u_hm.transpose(1, 2))
    down_targets = down_w + down_drift

    _, _, init_healers_down = batched_exact_gram_smw_healer_balanced(
        down_w, v_h_batch, s_h_batch, svd_rank=svd_rank, alpha=alpha_down
    )

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

    # Reuse init_healers_down to strip spikes (zero extra SVD call!) -> Pack clean 2-bit tail -> Fit final FP8 SVD Healer!
    res_down_clean = down_w - beta_down.view(m_cnt, 1, 1) * h_unbind_down
    down_tail_packed, down_tail_scale, e_down_deq = pack_tail_batched_lsq(
        res_down_clean - init_healers_down, bits=down_tail_bits, block_size=32
    )

    u_down, v_down, h_down_final = batched_exact_gram_smw_healer_balanced(
        (down_targets - beta_down.view(m_cnt, 1, 1) * h_unbind_down) - e_down_deq,
        v_h_batch,
        s_h_batch,
        svd_rank=svd_rank,
        alpha=alpha_down,
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
        "s_gu.weight": s_gu_packed.contiguous(),
        "s_gu.weight_scale": s_gu_scale.contiguous(),
        "s_gu.weight_scale_2": s_gu_scale2.contiguous(),
        "s_gu.beta": beta_gu.contiguous().to(torch.float32),
        "s_down.weight": s_down_packed.contiguous(),
        "s_down.weight_scale": s_down_scale.contiguous(),
        "s_down.weight_scale_2": s_down_scale2.contiguous(),
        "s_down.beta": beta_down.contiguous().to(torch.float32),
        "healer.u_gu": u_gu.contiguous(),
        "healer.v_gu": v_gu.contiguous(),
        "healer.u_down": u_down.contiguous(),
        "healer.v_down": v_down.contiguous(),
        "tail.gu_bits": torch.tensor(gu_tail_bits, dtype=torch.int32),
        "tail.gu_packed": gu_tail_packed.contiguous(),
        "tail.gu_scale": gu_tail_scale.contiguous(),
        "tail.down_bits": torch.tensor(down_tail_bits, dtype=torch.int32),
        "tail.down_packed": down_tail_packed.contiguous(),
        "tail.down_scale": down_tail_scale.contiguous(),
    }


def main() -> None:
    device = torch.device("cuda:0")
    stock_idx = json.loads((STOCK_DIR / "model.safetensors.index.json").read_text())["weight_map"]
    st_handles: dict[str, safe_open] = {}

    def get_stock_t(key: str) -> torch.Tensor:
        sf = stock_idx[key]
        if sf not in st_handles:
            st_handles[sf] = safe_open(str(STOCK_DIR / sf), framework="pt", device="cpu")
        return st_handles[sf].get_tensor(key)

    prefix = "model.language_model.layers.0.mlp"
    router_w = get_stock_t(f"{prefix}.gate.weight").to(device=device, dtype=torch.float32)
    x_stream, domain_slices = build_6domain_hidden_stream(router_w, device)

    vals_st, ids_st = torch.topk(x_stream @ router_w.T, k=10, dim=-1)
    gates_st = F.softmax(vals_st, dim=-1)

    # Load old v6.2 Layer 0 shard
    raw_v62 = load_file(str(CHRONOS_DIR / "chronos-layer-00000-of-00048.safetensors"), device="cpu")
    groups_v62 = [
        ChronosHolographicExpertGroup(
            group_idx=g_i,
            tensors={k[len(f"{prefix}.chronos_groups.{g_i}."):]: v for k, v in raw_v62.items() if k.startswith(f"{prefix}.chronos_groups.{g_i}.")},
            device=device,
        )
        for g_i in range(64)
    ]

    # Convert active groups with v6.3 Full-Spectrum SMW Row-Gram + Balanced FP8 + Exact Router Centroids
    _, s_rt, vh_rt = torch.linalg.svd(router_w, full_matrices=False)
    v_rt_256 = vh_rt[:256, :].T.contiguous()
    s_rt_256 = s_rt[:256].contiguous()
    x_cal = F.rms_norm(router_w, (2560,), eps=1e-6)
    h128 = build_hadamard_128(device)

    all_needed = sorted(set(ids_st.flatten().tolist()))
    needed_groups = sorted(set(eid // 8 for eid in all_needed))
    print(f"Testing {len(all_needed)} active experts across {len(needed_groups)}/64 groups on Layer 0...")

    stock_exp = {}
    v62_exp = {}
    v63_1b_exp = {}
    v64_2b_exp = {}
    t0 = time.perf_counter()
    for g_idx in needed_groups:
        e_start = g_idx * 8
        gp_l, gs_l, gs2_l = [], [], []
        up_l, us_l, us2_l = [], [], []
        dp_l, ds_l, ds2_l = [], [], []
        for e_id in range(e_start, e_start + 8):
            ep = f"{prefix}.experts.{e_id}"
            gp_l.append(get_stock_t(f"{ep}.gate_proj.weight"))
            gs_l.append(get_stock_t(f"{ep}.gate_proj.weight_scale"))
            gs2_l.append(get_stock_t(f"{ep}.gate_proj.weight_scale_2"))
            up_l.append(get_stock_t(f"{ep}.up_proj.weight"))
            us_l.append(get_stock_t(f"{ep}.up_proj.weight_scale"))
            us2_l.append(get_stock_t(f"{ep}.up_proj.weight_scale_2"))
            dp_l.append(get_stock_t(f"{ep}.down_proj.weight"))
            ds_l.append(get_stock_t(f"{ep}.down_proj.weight_scale"))
            ds2_l.append(get_stock_t(f"{ep}.down_proj.weight_scale_2"))
        gw_b = unpack_nvfp4_batched(torch.stack(gp_l).to(device), torch.stack(gs_l).to(device), torch.stack(gs2_l).to(device))
        uw_b = unpack_nvfp4_batched(torch.stack(up_l).to(device), torch.stack(us_l).to(device), torch.stack(us2_l).to(device))
        dw_b = unpack_nvfp4_batched(torch.stack(dp_l).to(device), torch.stack(ds_l).to(device), torch.stack(ds2_l).to(device))

        t_1b = convert_group_v68(
            group_idx=g_idx,
            gate_w=gw_b,
            up_w=uw_b,
            down_w=dw_b,
            router_group_w=router_w[e_start : e_start + 8],
            x_cal=x_cal,
            v_rt_256=v_rt_256,
            s_rt_256=s_rt_256,
            h128=h128,
            svd_rank=48,
            gu_tail_bits=2,
            down_tail_bits=2,
        )
        t_2b = convert_group_v68(
            group_idx=g_idx,
            gate_w=gw_b,
            up_w=uw_b,
            down_w=dw_b,
            router_group_w=router_w[e_start : e_start + 8],
            x_cal=x_cal,
            v_rt_256=v_rt_256,
            s_rt_256=s_rt_256,
            h128=h128,
            svd_rank=64,
            gu_tail_bits=2,
            down_tail_bits=2,
        )
        grp_1b = ChronosHolographicExpertGroup(group_idx=g_idx, tensors=t_1b, device=device)
        grp_2b = ChronosHolographicExpertGroup(group_idx=g_idx, tensors=t_2b, device=device)
        for m in range(8):
            eid = e_start + m
            if eid in all_needed:
                stock_exp[eid] = (gw_b[m], uw_b[m], dw_b[m])
                v62_exp[eid] = groups_v62[g_idx].reconstruct_expert(m)
                v63_1b_exp[eid] = grp_1b.reconstruct_expert(m)
                v64_2b_exp[eid] = grp_2b.reconstruct_expert(m)

    print(f"Converted {len(needed_groups)} groups in {time.perf_counter() - t0:.2f}s")
    y_st = execute_routed_moe_grouped(x_stream, ids_st, gates_st, stock_exp)
    y_62 = execute_routed_moe_grouped(x_stream, ids_st, gates_st, v62_exp)
    y_1b = execute_routed_moe_grouped(x_stream, ids_st, gates_st, v63_1b_exp)
    y_2b = execute_routed_moe_grouped(x_stream, ids_st, gates_st, v64_2b_exp)

    cos_62 = F.cosine_similarity(y_62.flatten(), y_st.flatten(), dim=0).item() * 100.0
    cos_1b = F.cosine_similarity(y_1b.flatten(), y_st.flatten(), dim=0).item() * 100.0
    cos_2b = F.cosine_similarity(y_2b.flatten(), y_st.flatten(), dim=0).item() * 100.0
    print("=" * 124)
    print(
        f"OVERALL PURE-COLD (0% CACHE) ROUTED MOE COSINE: "
        f"v6.7(r=48) = {cos_62:.2f}%  |  v6.8(r=48) = {cos_1b:.2f}%  |  v6.8(r=64) = {cos_2b:.2f}%"
    )
    print("=" * 124)
    for d_name, s_i, e_i in domain_slices:
        c62 = F.cosine_similarity(y_62[s_i:e_i].flatten(), y_st[s_i:e_i].flatten(), dim=0).item() * 100.0
        c1b = F.cosine_similarity(y_1b[s_i:e_i].flatten(), y_st[s_i:e_i].flatten(), dim=0).item() * 100.0
        c2b = F.cosine_similarity(y_2b[s_i:e_i].flatten(), y_st[s_i:e_i].flatten(), dim=0).item() * 100.0
        print(
            f"  Domain [{d_name:<22}] | "
            f"v6.7(r=48): {c62:6.2f}%  -->  v6.8(r=48): {c1b:6.2f}%  -->  v6.8(r=64): {c2b:6.2f}%"
        )
    print("=" * 124)


if __name__ == "__main__":
    main()
