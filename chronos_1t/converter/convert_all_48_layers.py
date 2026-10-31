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
def batched_subspace_eckart_young(
    targets: torch.Tensor,
    v_act: torch.Tensor,
    s_act: torch.Tensor,
    svd_rank: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched Subspace Eckart-Young Rank-r Healer for `targets` of shape (B, 1280, 2560)."""
    # targets @ v_act -> (B, 1280, 64)
    w_sub = torch.matmul(targets, v_act) * s_act.view(1, 1, -1)
    u_sub, s_sub, vh_sub = torch.linalg.svd(w_sub, full_matrices=False)
    r = min(svd_rank, u_sub.shape[-1])
    u_r = (u_sub[:, :, :r] * s_sub[:, :r].unsqueeze(1)).to(torch.float8_e4m3fn)
    v_unscaled = vh_sub[:, :r, :] / (s_act.view(1, 1, -1) + 1e-8)
    v_r = torch.matmul(v_unscaled, v_act.T).to(torch.float8_e4m3fn)
    h_full = torch.bmm(u_r.float(), v_r.float())
    return u_r, v_r, h_full


@torch.no_grad()
def batched_exact_gram_smw_healer(
    targets: torch.Tensor,
    v_acts: torch.Tensor,
    s_acts: torch.Tensor,
    svd_rank: int = 32,
    alpha: float = 16.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched Exact Gram Eigendecomposition Rank-r SMW Healer for `down_proj` (B, 2560, 640).
    Uses exact symmetric eigendecomposition on G = T^T T in R^{B x 640 x 640} instead of iterative svd_lowrank!
    """
    # targets: (B, 2560, 640), v_acts: (B, 640, K), s_acts: (B, K)
    gain = alpha * (s_acts / (s_acts[:, 0:1] + 1e-12)).unsqueeze(1)  # (B, 1, K)
    inv_gain = gain / (1.0 + gain)
    t_proj = torch.bmm(targets, v_acts)  # (B, 2560, K)
    t_weighted = targets + torch.bmm(t_proj * gain, v_acts.transpose(1, 2))  # (B, 2560, 640)

    # Exact symmetric Gram matrix (B, 640, 640)
    gram = torch.bmm(t_weighted.transpose(1, 2), t_weighted)
    evals, evecs = torch.linalg.eigh(gram)  # ascending order
    vh_top = evecs[:, :, -svd_rank:].flip(dims=[-1]).transpose(1, 2).contiguous()  # (B, r, 640)
    u_scaled = torch.bmm(t_weighted, vh_top.transpose(1, 2))  # (B, 2560, r)
    u_r = u_scaled.to(torch.float8_e4m3fn)

    vh_proj = torch.bmm(vh_top, v_acts)  # (B, r, K)
    v_r = (vh_top - torch.bmm(vh_proj * inv_gain, v_acts.transpose(1, 2))).to(torch.float8_e4m3fn)
    h_full = torch.bmm(u_r.float(), v_r.float())
    return u_r, v_r, h_full


@torch.no_grad()
def pack_tail_batched(
    res_tail: torch.Tensor,
    bits: int,
    block_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched tail lattice packer & unpacker for shape (B, rows, cols)."""
    bsz, rows, cols = res_tail.shape
    device = res_tail.device
    if bits == 2:
        blocks = res_tail.reshape(bsz, -1, block_size)
        sigma = blocks.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-6, max=448.0)
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

        lut = torch.tensor(LLOYD_MAX_2BIT_LUT, dtype=torch.float32, device=device)
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
    x_cal: torch.Tensor,
    v_act_x: torch.Tensor,
    s_act_x: torch.Tensor,
    h128: torch.Tensor,
    svd_rank: int = 32,
    gu_tail_bits: int = 16,
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
    _, _, init_healers_gu = batched_subspace_eckart_young(gu_experts, v_act_x, s_act_x, svd_rank=svd_rank)

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

    # Batched projection on x_cal[:256] to compute beta_gu
    x_sub = x_cal[:256]  # (256, 2560)
    y_tail_gu = torch.matmul(x_sub.unsqueeze(0), smooth_gu.transpose(1, 2))    # (8, 256, 1280)
    y_holo_gu = torch.matmul(x_sub.unsqueeze(0), h_unbind_gu.transpose(1, 2))  # (8, 256, 1280)
    beta_gu = (y_tail_gu * y_holo_gu).sum(dim=(1, 2)) / ((y_holo_gu * y_holo_gu).sum(dim=(1, 2)) + 1e-12)  # (8,)

    res_gu = gu_experts - beta_gu.view(m_cnt, 1, 1) * h_unbind_gu
    u_gu, v_gu, h_gu_final = batched_subspace_eckart_young(res_gu, v_act_x, s_act_x, svd_rank=svd_rank)
    gu_tail_packed, gu_tail_scale, e_gu_deq = pack_tail_batched(res_gu - h_gu_final, bits=gu_tail_bits, block_size=32)

    gu_hat = beta_gu.view(m_cnt, 1, 1) * h_unbind_gu + h_gu_final + e_gu_deq
    rec_g = gu_hat[:, :640, :]
    rec_u = gu_hat[:, 640:, :]

    # 2. Batched Coupled SwiGLU Drift Target + Exact Gram SMW Healer for down_proj (8, 2560, 640)
    h_true_c = F.silu(torch.matmul(x_sub.unsqueeze(0), gate_w.transpose(1, 2))) * torch.matmul(x_sub.unsqueeze(0), up_w.transpose(1, 2))  # (8, 256, 640)
    h_hat_c = F.silu(torch.matmul(x_sub.unsqueeze(0), rec_g.transpose(1, 2))) * torch.matmul(x_sub.unsqueeze(0), rec_u.transpose(1, 2))    # (8, 256, 640)

    _, s_h_all, vh_h_all = torch.linalg.svd(h_hat_c, full_matrices=False)  # vh_h_all: (8, 256, 640)
    v_h_batch = vh_h_all[:, :192, :].transpose(1, 2).contiguous()          # (8, 640, 192)
    s_h_batch = s_h_all[:, :192].contiguous()                              # (8, 192)

    y_true_d = torch.bmm(h_true_c, down_w.transpose(1, 2))                 # (8, 256, 2560)
    u_hm = vh_h_all[:, :64, :].transpose(1, 2).contiguous()                # (8, 640, 64)
    hm_sub = torch.bmm(h_hat_c, u_hm)                                      # (8, 256, 64)
    delta_y = y_true_d - torch.bmm(h_hat_c, down_w.transpose(1, 2))        # (8, 256, 2560)
    gram_h = torch.bmm(hm_sub.transpose(1, 2), hm_sub)                     # (8, 64, 64)
    diag_mean = torch.diagonal(gram_h, dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).unsqueeze(-1)
    reg_eye = 0.03 * diag_mean * torch.eye(64, device=device).unsqueeze(0)
    delta_w = torch.bmm(torch.bmm(delta_y.transpose(1, 2), hm_sub), torch.linalg.inv(gram_h + reg_eye))  # (8, 2560, 64)
    down_targets = down_w + torch.bmm(delta_w, u_hm.transpose(1, 2))       # (8, 2560, 640)

    _, _, init_healers_down = batched_exact_gram_smw_healer(down_targets, v_h_batch, s_h_batch, svd_rank=svd_rank, alpha=16.0)

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
    )  # (8, 2560, 640)

    rem_t_down = down_targets - init_healers_down
    y_tail_d = torch.bmm(h_hat_c, rem_t_down.transpose(1, 2))
    y_holo_d = torch.bmm(h_hat_c, h_unbind_down.transpose(1, 2))
    beta_down = (y_tail_d * y_holo_d).sum(dim=(1, 2)) / ((y_holo_d * y_holo_d).sum(dim=(1, 2)) + 1e-12)  # (8,)

    res_down = down_targets - beta_down.view(m_cnt, 1, 1) * h_unbind_down
    u_down, v_down, h_down_final = batched_exact_gram_smw_healer(res_down, v_h_batch, s_h_batch, svd_rank=svd_rank, alpha=16.0)
    down_tail_packed, down_tail_scale, _ = pack_tail_batched(res_down - h_down_final, bits=down_tail_bits, block_size=32)

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
    gu_bits: int = 1,
    down_bits: int = 2,
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
    torch.manual_seed(2026 + l_idx)
    _, s_rt, vh_rt = torch.linalg.svd(router_w, full_matrices=False)
    u_ctx = vh_rt[:48, :].T
    w_ctx = ((s_rt[:48] / s_rt[0]) ** 1.5).unsqueeze(0)

    z_cal = torch.randn(512, 48, device=device) * w_ctx
    x_cal = F.rms_norm(z_cal @ u_ctx.T + torch.randn(512, 2560, device=device) * 0.01, (2560,), eps=1e-6)
    _, s_act_x, vh_act_x = torch.linalg.svd(x_cal, full_matrices=False)
    v_act_x = vh_act_x[:64, :].T.contiguous()
    s_act_x_sub = s_act_x[:64].contiguous()

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
            x_cal=x_cal,
            v_act_x=v_act_x,
            s_act_x=s_act_x_sub,
            h128=h128,
            svd_rank=32,
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
    parser.add_argument("--gu-bits", type=int, default=1, help="1=1-bit Compact (34.5GB), 16=Ternary (39GB), 2=Ultra (44.7GB)")
    parser.add_argument("--down-bits", type=int, default=2)
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
    print(f"BATCHED GPU STREAMING CONVERSION: LAYERS {args.start_layer}..{args.end_layer} -> {out_dir}")
    print("=" * 116)

    converted_map: dict[int, str] = {}
    total_mb = 0.0
    t_all = time.perf_counter()

    for l_idx in range(args.start_layer, args.end_layer + 1):
        sname, mb, dt = convert_single_layer(
            l_idx=l_idx,
            idx_map=idx_map,
            out_dir=out_dir,
            device=device,
            h128=h128,
            gu_bits=args.gu_bits,
            down_bits=args.down_bits,
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
