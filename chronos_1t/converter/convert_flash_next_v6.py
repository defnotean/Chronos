#!/usr/bin/env python3
"""
Chronos-1T x PseudoBrain (pb27) — Real `.safetensors` Converter & Round-Trip Verifier (`v6 Zero-Con`).

Converts `flash_next_nvfp4` (`PB27Qwen4ExpForConditionalGeneration`) routed MoE layers into
authentic byte-packed `.safetensors` shards:
- 8:1 NVFP4 Holographic Superposition Banks (`uint8` packed E2M1 nibbles + `float8_e4m3fn` scales)
- Data-Free Router-Manifold + Coupled SwiGLU Rank-32 SVD Healers (`float8_e4m3fn`)
- Real Bit-Packed Isotropic Gaussian Tail Lattice (`uint8` 1-bit / 2-bit packed + `float8_e4m3fn` scales)
- Composes `/home/defnotean/pb27/hf/flash_next_chronos_v6` with symlinks for non-MoE shards.
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
from tokenizers import Tokenizer

# Allow running from repository root or remote chronos_1t directory
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chronos_1t.models.chronos_moe_block import (
    ChronosHolographicExpertGroup,
    build_hadamard_128,
    fold_2d_permuted_hadamard,
    generate_expert_1d_keys,
    pack_1bit_rademacher,
    pack_2bit_lloyd_max,
    pack_nvfp4,
    pack_sparse_2bit_lloyd_max,
    pack_ternary_base3,
    unbind_2d_permuted_hadamard,
    unpack_1bit_rademacher,
    unpack_2bit_lloyd_max,
    unpack_nvfp4,
    unpack_sparse_2bit_lloyd_max,
    unpack_ternary_base3,
)

MODEL_DIR = Path("/home/defnotean/pb27/hf/flash_next_nvfp4")
OUT_CKPT_DIR = Path("/home/defnotean/pb27/hf/flash_next_chronos_v6")


def dequantize_mxfp8_tensor(weight_fp8: torch.Tensor, scale_e8m0: torch.Tensor, device: torch.device) -> torch.Tensor:
    w = weight_fp8.to(device=device, dtype=torch.float32)
    s_u8 = scale_e8m0.to(device=device, dtype=torch.int32)
    scales = torch.pow(2.0, s_u8.float() - 127.0).unsqueeze(-1)
    w_grouped = w.view(w.shape[0], -1, 32)
    return (w_grouped * scales).view(w.shape[0], -1)


def subspace_eckart_young_healer(
    target: torch.Tensor,
    v_act: torch.Tensor,
    s_act: torch.Tensor,
    svd_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    w_sub = (target @ v_act) * s_act.unsqueeze(0)
    u_sub, s_sub, vh_sub = torch.linalg.svd(w_sub, full_matrices=False)
    r = min(svd_rank, u_sub.shape[1])
    u_r = (u_sub[:, :r] * s_sub[:r].unsqueeze(0)).to(torch.float8_e4m3fn)
    v_unscaled = vh_sub[:r, :] / (s_act.unsqueeze(0) + 1e-8)
    v_r = (v_unscaled @ v_act.T).to(torch.float8_e4m3fn)
    return u_r, v_r, u_r.float() @ v_r.float()


def full_spectrum_smw_healer(
    target: torch.Tensor,
    v_act: torch.Tensor,
    s_act: torch.Tensor,
    svd_rank: int,
    alpha: float = 16.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gain = alpha * (s_act / (s_act[0] + 1e-12)).unsqueeze(0)
    inv_gain = gain / (1.0 + gain)
    t_proj = target @ v_act
    t_weighted = target + (t_proj * gain) @ v_act.T
    q_rank = min(svd_rank + 24, min(target.shape))
    u, s_val, v = torch.svd_lowrank(t_weighted, q=q_rank, niter=2)
    u_r = (u[:, :svd_rank] * s_val[:svd_rank].unsqueeze(0)).to(torch.float8_e4m3fn)
    vh_top = v[:, :svd_rank].T
    v_r = (vh_top - ((vh_top @ v_act) * inv_gain) @ v_act.T).to(torch.float8_e4m3fn)
    return u_r, v_r, u_r.float() @ v_r.float()


@torch.no_grad()
def convert_expert_group_to_packed_tensors(
    group_idx: int,
    gate_w: list[torch.Tensor],
    up_w: list[torch.Tensor],
    down_w: list[torch.Tensor],
    x_cal: torch.Tensor,
    v_act_x: torch.Tensor,
    s_act_x: torch.Tensor,
    h128: torch.Tensor,
    svd_rank: int = 32,
    gu_tail_bits: int = 2,
    down_tail_bits: int = 2,
    gu_keep_ratio: float = 1.0,
    down_keep_ratio: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Converts a group of K=8 experts into authentic byte-packed `.safetensors` tensors."""
    m_cnt = len(gate_w)
    device = gate_w[0].device
    gu_experts = [torch.cat([gate_w[m], up_w[m]], dim=0) for m in range(m_cnt)]

    # 1. Fused gate_up_proj (1280 x 2560) with Shared-V Rank-r Healer
    gu_keys = [
        generate_expert_1d_keys(1280, 2560, m, seed_base=101 + group_idx * 8, device=device)
        for m in range(m_cnt)
    ]
    init_healers_gu = []
    for m in range(m_cnt):
        _, _, hm = subspace_eckart_young_healer(gu_experts[m], v_act_x, s_act_x, svd_rank=svd_rank)
        init_healers_gu.append(hm)

    s_gu_accum = torch.zeros((1280, 2560), device=device, dtype=torch.float32)
    for m in range(m_cnt):
        r_s1, r_perm, _, r_s2, c_s1, c_perm, _, c_s2 = gu_keys[m]
        s_gu_accum = s_gu_accum + fold_2d_permuted_hadamard(
            gu_experts[m] - init_healers_gu[m], r_s1, r_perm, r_s2, c_s1, c_perm, c_s2, h128
        )

    s_gu_packed, s_gu_scale, s_gu_scale2 = pack_nvfp4(s_gu_accum)
    s_gu_deq = unpack_nvfp4(s_gu_packed, s_gu_scale, s_gu_scale2)

    u_gu_list, v_gu_list, beta_gu_list = [], [], []
    gu_tail_packed_list, gu_tail_scale_list, gu_tail_mask_list = [], [], []
    rec_g, rec_u = [], []
    x_col_imp = x_cal.pow(2).mean(dim=0).sqrt().clamp_min(1e-6)

    for m in range(m_cnt):
        r_s1, _, r_inv, r_s2, c_s1, _, c_inv, c_s2 = gu_keys[m]
        h_unbind = unbind_2d_permuted_hadamard(s_gu_deq, r_s1, r_inv, r_s2, c_s1, c_inv, c_s2, h128)
        rem_t = gu_experts[m] - init_healers_gu[m]
        y_tail = x_cal @ rem_t.T
        y_holo = x_cal @ h_unbind.T
        beta_m = float((torch.sum(y_tail * y_holo) / (torch.sum(y_holo * y_holo) + 1e-12)).item())
        beta_gu_list.append(beta_m)

        res_m = gu_experts[m] - beta_m * h_unbind
        u_r, v_r, h_m = subspace_eckart_young_healer(res_m, v_act_x, s_act_x, svd_rank=svd_rank)
        u_gu_list.append(u_r)
        v_gu_list.append(v_r)

        e_tail_raw = res_m - h_m
        if gu_tail_bits == 20:
            tm, tp, ts = pack_sparse_2bit_lloyd_max(e_tail_raw, x_col_imp, keep_ratio=gu_keep_ratio, block_size=64)
            e_tail_deq = unpack_sparse_2bit_lloyd_max(tm, tp, ts, 1280, 2560, block_size=64)
            gu_tail_mask_list.append(tm)
        elif gu_tail_bits == 16:
            tp, ts = pack_ternary_base3(e_tail_raw, block_size=32)
            e_tail_deq = unpack_ternary_base3(tp, ts, block_size=32)
        elif gu_tail_bits == 2:
            tp, ts = pack_2bit_lloyd_max(e_tail_raw, block_size=32)
            e_tail_deq = unpack_2bit_lloyd_max(tp, ts, block_size=32)
        else:
            tp, ts = pack_1bit_rademacher(e_tail_raw, block_size=64)
            e_tail_deq = unpack_1bit_rademacher(tp, ts, block_size=64)

        gu_tail_packed_list.append(tp)
        gu_tail_scale_list.append(ts)

        gu_hat = beta_m * h_unbind + h_m + e_tail_deq
        rec_g.append(gu_hat[:640, :])
        rec_u.append(gu_hat[640:, :])

    # 2. Coupled SwiGLU Drift Target + Full-Spectrum SMW Healer for down_proj (2560 x 640)
    h_true_c = [F.silu(x_cal[:256] @ gate_w[m].T) * (x_cal[:256] @ up_w[m].T) for m in range(m_cnt)]
    h_hat_c = [F.silu(x_cal[:256] @ rec_g[m].T) * (x_cal[:256] @ rec_u[m].T) for m in range(m_cnt)]

    down_keys = [
        generate_expert_1d_keys(2560, 640, m, seed_base=303 + group_idx * 8, device=device)
        for m in range(m_cnt)
    ]
    v_h_list, s_h_list, down_targets, init_healers_down = [], [], [], []
    for m in range(m_cnt):
        hm = h_hat_c[m]
        _, s_h, vh_h = torch.linalg.svd(hm, full_matrices=False)
        v_h_list.append(vh_h[:192, :].T)
        s_h_list.append(s_h[:192])

        y_true_m = h_true_c[m] @ down_w[m].T
        u_hm = vh_h[:64, :].T
        hm_sub = hm @ u_hm
        delta_y = y_true_m - (hm @ down_w[m].T)
        gram = hm_sub.T @ hm_sub
        reg = 0.03 * torch.diagonal(gram).mean() * torch.eye(64, device=device)
        delta_w = (delta_y.T @ hm_sub) @ torch.linalg.inv(gram + reg)
        target_m = down_w[m] + delta_w @ u_hm.T
        down_targets.append(target_m)

        _, _, hm_d = full_spectrum_smw_healer(target_m, v_h_list[m], s_h_list[m], svd_rank=svd_rank, alpha=16.0)
        init_healers_down.append(hm_d)

    s_down_accum = torch.zeros((2560, 640), device=device, dtype=torch.float32)
    for m in range(m_cnt):
        dr_s1, dr_perm, _, dr_s2, dc_s1, dc_perm, _, dc_s2 = down_keys[m]
        s_down_accum = s_down_accum + fold_2d_permuted_hadamard(
            down_w[m] - init_healers_down[m], dr_s1, dr_perm, dr_s2, dc_s1, dc_perm, dc_s2, h128
        )

    s_down_packed, s_down_scale, s_down_scale2 = pack_nvfp4(s_down_accum)
    s_down_deq = unpack_nvfp4(s_down_packed, s_down_scale, s_down_scale2)

    u_down_list, v_down_list, beta_down_list = [], [], []
    down_tail_packed_list, down_tail_scale_list, down_tail_mask_list = [], [], []

    for m in range(m_cnt):
        dr_s1, _, dr_inv, dr_s2, dc_s1, _, dc_inv, dc_s2 = down_keys[m]
        h_unbind_d = unbind_2d_permuted_hadamard(s_down_deq, dr_s1, dr_inv, dr_s2, dc_s1, dc_inv, dc_s2, h128)
        hm = h_hat_c[m]
        rem_t = down_targets[m] - init_healers_down[m]
        y_tail = hm @ rem_t.T
        y_holo = hm @ h_unbind_d.T
        beta_d = float((torch.sum(y_tail * y_holo) / (torch.sum(y_holo * y_holo) + 1e-12)).item())
        beta_down_list.append(beta_d)

        res_d = down_targets[m] - beta_d * h_unbind_d
        u_dr, v_dr, h_dm = full_spectrum_smw_healer(res_d, v_h_list[m], s_h_list[m], svd_rank=svd_rank, alpha=16.0)
        u_down_list.append(u_dr)
        v_down_list.append(v_dr)

        e_tail_d = res_d - h_dm
        h_col_imp = hm.pow(2).mean(dim=0).sqrt().clamp_min(1e-6)
        if down_tail_bits == 20:
            dtm, dtp, dts = pack_sparse_2bit_lloyd_max(e_tail_d, h_col_imp, keep_ratio=down_keep_ratio, block_size=64)
            down_tail_mask_list.append(dtm)
        elif down_tail_bits == 16:
            dtp, dts = pack_ternary_base3(e_tail_d, block_size=32)
        elif down_tail_bits == 2:
            dtp, dts = pack_2bit_lloyd_max(e_tail_d, block_size=32)
        else:
            dtp, dts = pack_1bit_rademacher(e_tail_d, block_size=64)
        down_tail_packed_list.append(dtp)
        down_tail_scale_list.append(dts)

    out_dict = {
        "s_gu.weight": s_gu_packed.cpu(),
        "s_gu.weight_scale": s_gu_scale.cpu(),
        "s_gu.weight_scale_2": s_gu_scale2.cpu(),
        "s_gu.beta": torch.tensor(beta_gu_list, dtype=torch.float32),
        "s_down.weight": s_down_packed.cpu(),
        "s_down.weight_scale": s_down_scale.cpu(),
        "s_down.weight_scale_2": s_down_scale2.cpu(),
        "s_down.beta": torch.tensor(beta_down_list, dtype=torch.float32),
        "healer.u_gu": torch.stack(u_gu_list, dim=0).cpu(),
        "healer.v_gu": torch.stack(v_gu_list, dim=0).cpu(),
        "healer.u_down": torch.stack(u_down_list, dim=0).cpu(),
        "healer.v_down": torch.stack(v_down_list, dim=0).cpu(),
        "tail.gu_bits": torch.tensor(gu_tail_bits, dtype=torch.int32),
        "tail.gu_packed": torch.stack(gu_tail_packed_list, dim=0).cpu(),
        "tail.gu_scale": torch.stack(gu_tail_scale_list, dim=0).cpu(),
        "tail.down_bits": torch.tensor(down_tail_bits, dtype=torch.int32),
        "tail.down_packed": torch.stack(down_tail_packed_list, dim=0).cpu(),
        "tail.down_scale": torch.stack(down_tail_scale_list, dim=0).cpu(),
    }
    if gu_tail_mask_list:
        out_dict["tail.gu_mask"] = torch.stack(gu_tail_mask_list, dim=0).cpu()
    if down_tail_mask_list:
        out_dict["tail.down_mask"] = torch.stack(down_tail_mask_list, dim=0).cpu()
    return out_dict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", default="0,23,47", help="Comma-separated layer indices to convert & verify")
    parser.add_argument("--groups-per-layer", type=int, default=1, help="Number of 8-expert groups per layer (64 = all 512 experts)")
    parser.add_argument("--out-dir", default=str(OUT_CKPT_DIR))
    args = parser.parse_args()

    device = torch.device("cuda:0")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    test_layers = [int(x) for x in args.layers.split(",")]

    print("=" * 124)
    print("STAGE 1: REAL BYTE-PACKED `.safetensors` CONVERSION & ROUND-TRIP VERIFICATION ON DGX SPARK GB10")
    print("=" * 124)

    idx_map = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())["weight_map"]
    tokenizer = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))

    topic_shift_segments = [
        "__global__ void blackwell_sm121_hss_gemv(const half* __restrict__ seed, const half* __restrict__ x, float* out) { int lane = threadIdx.x & 31; ",
        "Sous le pont Mirabeau coule la Seine et nos amours faut-il qu'il m'en souvienne la joie venait toujours apres la peine. ",
        "The Hamiltonian of a transmon superconducting qubit coupled to a microwave resonator in the dispersive regime is H = hbar omega_r a^dagger a + ",
        "CRISPR-Cas9 ribonucleoprotein complexes recognize the protospacer adjacent motif 5'-NGG-3' and cleave double-stranded genomic DNA via HNH and RuvC domains.",
    ]
    seg_ids = [tokenizer.encode(s).ids for s in topic_shift_segments]
    all_ids = [tid for seg in seg_ids for tid in seg]
    token_ids_t = torch.tensor(all_ids, dtype=torch.long)

    with safe_open(str(MODEL_DIR / idx_map["model.language_model.embed_tokens.weight"]), framework="pt", device="cpu") as f:
        embed_w = f.get_tensor("model.language_model.embed_tokens.weight")
        real_tok_embeds = embed_w[token_ids_t].to(device=device, dtype=torch.float32)
        del embed_w

    with safe_open(str(MODEL_DIR / idx_map["lm_head.weight"]), framework="pt", device="cpu") as f:
        lm_head_w = f.get_tensor("lm_head.weight").to(device=device, dtype=torch.bfloat16)

    h128 = build_hadamard_128(device)
    depth_scale = math.sqrt(48.0 / len(test_layers))

    tiers = [
        {"id": "Compact_1bGU_2bDown", "name": "Chronos-v6 Compact (1b GU + 2b Down Packed)", "gu_bits": 1, "down_bits": 2, "gu_keep": 1.00, "down_keep": 1.00},
        {"id": "Ternary_Base3_All", "name": "Chronos-v6.2 Ternary-Base3 (1.6b GU + 1.6b Down)", "gu_bits": 16, "down_bits": 16, "gu_keep": 1.00, "down_keep": 1.00},
        {"id": "Ternary_Hybrid_GU3_D2", "name": "Chronos-v6.2 Ternary-Hybrid (1.6b GU + 2b Down)", "gu_bits": 16, "down_bits": 2, "gu_keep": 1.00, "down_keep": 1.00},
        {"id": "Ultra_2bGU_2bDown", "name": "Chronos-v6 Ultra (2b GU + 2b Down, blk=32)", "gu_bits": 2, "down_bits": 2, "gu_keep": 1.00, "down_keep": 1.00},
    ]

    h_running_before = None
    h_running_cold = {t["id"]: None for t in tiers}
    h_running_ema = {t["id"]: None for t in tiers}
    cold_cos_log = {t["id"]: [] for t in tiers}
    ema_cos_log = {t["id"]: [] for t in tiers}
    disk_bytes_per_group = {t["id"]: 0 for t in tiers}
    conv_ms_per_group = {t["id"]: 0.0 for t in tiers}

    for l_idx in test_layers:
        prefix = f"model.language_model.layers.{l_idx}.mlp"
        st_cache: dict[str, safe_open] = {}

        def get_t(key: str) -> torch.Tensor:
            sf = idx_map[key]
            if sf not in st_cache:
                st_cache[sf] = safe_open(str(MODEL_DIR / sf), framework="pt", device="cpu")
            return st_cache[sf].get_tensor(key)

        router_w = get_t(f"{prefix}.gate.weight").to(device=device, dtype=torch.float32)
        shared_gate_w = get_t(f"{prefix}.shared_expert_gate.weight").to(device=device, dtype=torch.float32)
        sh_down = dequantize_mxfp8_tensor(get_t(f"{prefix}.shared_expert.down_proj.weight"), get_t(f"{prefix}.shared_expert.down_proj.weight_scale"), device)
        sh_gate = dequantize_mxfp8_tensor(get_t(f"{prefix}.shared_expert.gate_proj.weight"), get_t(f"{prefix}.shared_expert.gate_proj.weight_scale"), device)
        sh_up = dequantize_mxfp8_tensor(get_t(f"{prefix}.shared_expert.up_proj.weight"), get_t(f"{prefix}.shared_expert.up_proj.weight_scale"), device)

        gate_w, up_w, down_w = [], [], []
        for e_id in range(8):
            ep = f"{prefix}.experts.{e_id}"
            gate_w.append(unpack_nvfp4(get_t(f"{ep}.gate_proj.weight").to(device), get_t(f"{ep}.gate_proj.weight_scale").to(device), get_t(f"{ep}.gate_proj.weight_scale_2").to(device)))
            up_w.append(unpack_nvfp4(get_t(f"{ep}.up_proj.weight").to(device), get_t(f"{ep}.up_proj.weight_scale").to(device), get_t(f"{ep}.up_proj.weight_scale_2").to(device)))
            down_w.append(unpack_nvfp4(get_t(f"{ep}.down_proj.weight").to(device), get_t(f"{ep}.down_proj.weight_scale").to(device), get_t(f"{ep}.down_proj.weight_scale_2").to(device)))

        torch.manual_seed(2026 + l_idx)
        _, s_rt, vh_rt = torch.linalg.svd(router_w, full_matrices=False)
        u_ctx = vh_rt[:48, :].T
        w_ctx = ((s_rt[:48] / s_rt[0]) ** 1.5).unsqueeze(0)

        topic_states = []
        offset = 0
        for seg_idx, s_ids in enumerate(seg_ids):
            seg_len = len(s_ids)
            band_start = seg_idx * 10
            u_band = vh_rt[band_start : band_start + 12, :].T
            w_band = ((s_rt[band_start : band_start + 12] / s_rt[0]) ** 1.5).unsqueeze(0)
            seg_emb = real_tok_embeds[offset : offset + seg_len]
            offset += seg_len
            anchor = F.normalize(u_band.mean(dim=-1, keepdim=True).T, dim=-1) * 1.15
            tok_var = ((seg_emb @ u_ctx) * w_ctx) @ u_ctx.T * 0.85
            seg_x = F.rms_norm(anchor + tok_var + seg_emb * 0.02, (2560,), eps=1e-6)
            topic_states.append(seg_x)

        x_stream = torch.cat(topic_states, dim=0)
        n_toks = x_stream.shape[0]
        n_prefill = len(seg_ids[0])
        decode_x = x_stream[n_prefill:]

        if h_running_before is None:
            h_running_before = decode_x.clone()
            for t in tiers:
                h_running_cold[t["id"]] = decode_x.clone()
                h_running_ema[t["id"]] = decode_x.clone()

        z_cal = torch.randn(2048, 48, device=device) * w_ctx
        x_cal = F.rms_norm(z_cal @ u_ctx.T + torch.randn(2048, 2560, device=device) * 0.01, (2560,), eps=1e-6)
        _, s_act_x, vh_act_x = torch.linalg.svd(x_cal, full_matrices=False)
        v_act_x = vh_act_x[:64, :].T
        s_act_x_sub = s_act_x[:64]

        router_logits_all = x_stream @ router_w.T
        _, top10_all = torch.topk(router_logits_all, k=10, dim=-1)
        dec_vals, dec_ids = torch.topk(router_logits_all[n_prefill:], k=10, dim=-1)
        dec_gates = F.softmax(dec_vals, dim=-1)
        gate_8 = F.softmax(dec_vals[:, :8], dim=-1)

        sh_gate_score = torch.sigmoid(decode_x @ shared_gate_w.T)
        y_shared = sh_gate_score * ((F.silu(decode_x @ sh_gate.T) * (decode_x @ sh_up.T)) @ sh_down.T)

        y_routed_before = torch.zeros_like(decode_x)
        for m in range(8):
            ym = (F.silu(decode_x @ gate_w[m].T) * (decode_x @ up_w[m].T)) @ down_w[m].T
            y_routed_before = y_routed_before + gate_8[:, m : m + 1] * ym

        # Compute EMA hot mask (32 hot experts: 16 Global Router Prior + 16 24-token EMA)
        global_prior = torch.linalg.norm(router_w @ u_ctx, dim=-1)
        _, prior_top = torch.topk(global_prior, k=16)
        prior_set = set(prior_top.tolist())
        hot_rows = []
        for t_idx in range(n_prefill, n_toks):
            d_i = t_idx - n_prefill
            win_start = max(0, t_idx - 24)
            counts = torch.bincount(top10_all[win_start:t_idx].flatten(), minlength=512)
            _, ema_ids = torch.topk(counts, k=16)
            pset = prior_set | set(ema_ids.tolist())
            row_ids = dec_ids[d_i]
            row_gates = dec_gates[d_i]
            hit_mask = torch.tensor([1.0 if int(i) in pset else 0.0 for i in row_ids], device=device)
            hot_rows.append((row_gates * hit_mask).sum())
        hot_mass = torch.stack(hot_rows).unsqueeze(-1)

        for t in tiers:
            tid = t["id"]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            packed_tensors = convert_expert_group_to_packed_tensors(
                group_idx=0,
                gate_w=gate_w,
                up_w=up_w,
                down_w=down_w,
                x_cal=x_cal,
                v_act_x=v_act_x,
                s_act_x=s_act_x_sub,
                h128=h128,
                svd_rank=32,
                gu_tail_bits=t["gu_bits"],
                down_tail_bits=t["down_bits"],
                gu_keep_ratio=t["gu_keep"],
                down_keep_ratio=t["down_keep"],
            )
            torch.cuda.synchronize()
            conv_ms_per_group[tid] = (time.perf_counter() - t0) * 1000.0

            # Save real `.safetensors` file to disk and measure exact byte size!
            sf_path = out_dir / f"chronos_layer{l_idx}_group0_{tid}.safetensors"
            save_file(packed_tensors, str(sf_path))
            disk_bytes_per_group[tid] = sf_path.stat().st_size

            # Load back from `.safetensors` and execute via ChronosHolographicExpertGroup
            loaded_tensors = load_file(str(sf_path), device="cpu")
            grp_mod = ChronosHolographicExpertGroup(group_idx=0, tensors=loaded_tensors, device=device)

            y_routed_cold = torch.zeros_like(decode_x)
            for m in range(8):
                rg, ru, rd = grp_mod.reconstruct_expert(m)
                ym_cold = (F.silu(decode_x @ rg.T) * (decode_x @ ru.T)) @ rd.T
                y_routed_cold = y_routed_cold + gate_8[:, m : m + 1] * ym_cold

            y_routed_ema = hot_mass * y_routed_before + (1.0 - hot_mass) * y_routed_cold

            c_cold = F.cosine_similarity(y_routed_cold.flatten(), y_routed_before.flatten(), dim=0).item() * 100.0
            c_ema = F.cosine_similarity(y_routed_ema.flatten(), y_routed_before.flatten(), dim=0).item() * 100.0
            cold_cos_log[tid].append(c_cold)
            ema_cos_log[tid].append(c_ema)

            h_running_cold[tid] = h_running_cold[tid] + y_routed_before + (y_routed_cold - y_routed_before) * depth_scale + y_shared
            h_running_ema[tid] = h_running_ema[tid] + y_routed_before + (y_routed_ema - y_routed_before) * depth_scale + y_shared

        h_running_before = h_running_before + y_routed_before + y_shared

    logits_before = (F.rms_norm(h_running_before, (2560,), eps=1e-6).to(torch.bfloat16) @ lm_head_w.T).float()
    probs_before = F.softmax(logits_before, dim=-1)
    log_probs_before = F.log_softmax(logits_before, dim=-1)
    top1_before = logits_before.argmax(dim=-1)
    _, top5_before = torch.topk(logits_before, k=5, dim=-1)

    orig_group_mb = (8 * 3 * 640 * 2560 * 0.5625) / (1024 * 1024)
    orig_48l_gb = (orig_group_mb * 64 * 48) / 1024.0

    print("-" * 124)
    print(f"Stock `flash_next_nvfp4` 8-Expert Group Disk Size: {orig_group_mb:.2f} MB ({orig_48l_gb:.2f} GiB for 48 layers)")
    print("-" * 124)
    print(
        f"{'Packed .safetensors Format (Round-Trip Loaded)':<48} | {'Group MB':>8} | {'48L MoE':>8} | "
        f"{'0-CacheCos':>10} | {'EMA Cos':>8} | {'LogitCos':>8} | {'Top-1':>7} | {'Top-5':>7} | {'KL Div':>8}"
    )
    print("-" * 124)

    for t in tiers:
        tid = t["id"]
        grp_mb = disk_bytes_per_group[tid] / (1024 * 1024)
        moe_48l_gb = (grp_mb * 64 * 48) / 1024.0
        mean_cold = sum(cold_cos_log[tid]) / len(test_layers)
        mean_ema = sum(ema_cos_log[tid]) / len(test_layers)

        logits_ema = (F.rms_norm(h_running_ema[tid], (2560,), eps=1e-6).to(torch.bfloat16) @ lm_head_w.T).float()
        log_probs_ema = F.log_softmax(logits_ema, dim=-1)
        logit_cos = F.cosine_similarity(logits_ema.flatten(), logits_before.flatten(), dim=0).item() * 100.0
        top1_acc = (logits_ema.argmax(dim=-1) == top1_before).float().mean().item() * 100.0
        _, top5_ema = torch.topk(logits_ema, k=5, dim=-1)
        overlap_cnt = sum(len(set(rb) & set(rr)) for rb, rr in zip(top5_before.tolist(), top5_ema.tolist()))
        top5_ovl = (overlap_cnt / (5.0 * logits_before.shape[0])) * 100.0
        kl = torch.sum(probs_before * (log_probs_before - log_probs_ema), dim=-1).mean().item()

        print(
            f"{t['name']:<48} | {grp_mb:6.2f}MB | {moe_48l_gb:6.2f}GB | "
            f"{mean_cold:9.2f}% | {mean_ema:7.2f}% | {logit_cos:7.2f}% | {top1_acc:6.2f}% | {top5_ovl:6.2f}% | {kl:8.5f}"
        )

    print("=" * 124)


if __name__ == "__main__":
    main()
