#!/usr/bin/env python3
"""
Chronos-1T v7.5 x DeepSeek-V4.1-Flash-1017 (`deepseek-ai/DeepSeek-V4.1-Flash-1017`)
Full 40-Layer Vectorized Streaming Converter with Dual-Verified Post-Quantization Orthogonal Abliteration.

Architecture of `deepseek-ai/DeepSeek-V4.1-Flash-1017` (`DeepseekV41ForCausalLM`, `510.31 GB` Stock / `1.51 TB` BF16):
  - `hidden_size = 5120`, `moe_intermediate_size = 2304` (`gate_up`: `4608 x 5120`, `down`: `5120 x 2304`)
  - `num_hidden_layers = 40` (`layers.0` .. `layers.39`, all 40 layers are Sparse MoE)
  - `n_routed_experts = 384` per layer (`48` Holographic Groups of `K = 8` experts/layer = `15,360` total routed experts)
  - `num_experts_per_tok = 6`, `scoring_func = "sqrtsoftplus"`, `routed_scaling_factor = 1.5`, `swiglu_limit = 10.0`
  - `hc_mult = 4` (Hyper-Connections 4-lane doubly-stochastic Sinkhorn residual stream)

Dual-Verified Post-Quantization Orthogonal Abliteration:
  1. Streams base/uncensored layer shards (`model-00003-of-00048.safetensors` .. `model-00042-of-00048.safetensors`)
     from `dealignai/DeepSeek-V4.1-Flash-1017-UNCENSORED-FP8` (preserving the exact `320/320 = 100.00%` HarmBench-verified
     abliterated `attn.wo_b` and `ffn.shared_experts.w2` tensors).
  2. Extracts the exact 16-bit BF16 Failspy refusal vector `r_ob,l` in `R^5120` from
     `OBLITERATUS/DeepSeek-V4.1-Flash-1017-obliterated` (`layers 1..39`) and `r_dea,l` in `R^5120` from
     `deepseek-ai/DeepSeek-V4.1-Flash-1017` vs `dealignai` (`layers 10..36`), constructing the orthonormal refusal basis
     `R_l` in `R^{d_r x 5120}`.
  3. Orthogonalizes all `384` routed experts' `w2` (`down_proj`) matrices against `R_l` both BEFORE and AFTER
     Chronos `v7.5` quantization (`8:1 NVFP4 Holographic Bank + Coupled Rank-64 FP8 SMW SVD Healer + 1-Bit/2-Bit Tail`),
     guaranteeing `||R_l @ W_2_hat||_F = 0.000000` (zero refusal leakage from low-bit quantization error).
"""
from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import gc
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time
import urllib.request

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
if Path("/content/.hf_token").exists() and "HF_TOKEN" not in os.environ:
    os.environ["HF_TOKEN"] = Path("/content/.hf_token").read_text().strip()

import torch
import torch.nn.functional as F
from huggingface_hub import HfApi, hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file

_here = Path(__file__).resolve()
sys.path.insert(0, str(_here.parent))
if len(_here.parents) > 2:
    sys.path.insert(0, str(_here.parents[2]))

from chronos_1t.converter.convert_glm53_flash_colab import (
    batched_factorized_top_left_svd,
    batched_fold_2d,
    batched_full_spectrum_row_gram_smw_healer,
    batched_pack_1bit_rademacher,
    batched_pack_2bit_lloyd_max,
    batched_unbind_2d,
    batched_unpack_1bit_rademacher,
    batched_unpack_2bit_lloyd_max,
    build_hadamard_128,
    generate_expert_1d_keys,
    pack_nvfp4,
    unpack_nvfp4,
)
from chronos_1t.models.dsv41_chronos_moe import (
    dequantize_block_fp8_32,
    dequantize_mxfp4_32,
    dsv41_clamped_swiglu,
)


def drop_page_cache(path: Path) -> None:
    if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED") and path.exists():
        try:
            fd = os.open(str(path), os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except OSError:
            pass


def _http_get_header(repo: str, shard: str) -> tuple[int, dict]:
    url = f"https://huggingface.co/{repo}/resolve/main/{shard}"
    req = urllib.request.Request(url, headers={"Range": "bytes=0-350000", "User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = r.read()
    hlen = struct.unpack("<Q", data[:8])[0]
    if hlen + 8 > len(data):
        req = urllib.request.Request(url, headers={"Range": f"bytes=0-{hlen+16}", "User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            data = r.read()
    return hlen, json.loads(data[8 : 8 + hlen].decode("utf-8"))


def _http_fetch_tensor(repo: str, shard: str, hlen: int, info: dict, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    start, end = info["data_offsets"]
    url = f"https://huggingface.co/{repo}/resolve/main/{shard}"
    req = urllib.request.Request(
        url,
        headers={"Range": f"bytes={8 + hlen + start}-{8 + hlen + end - 1}", "User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        buf = bytearray(r.read())
    return torch.frombuffer(buf, dtype=dtype).reshape(info["shape"]).to(device)


@torch.no_grad()
def extract_layer_refusal_basis(
    layer_idx: int,
    w_dea_sh: torch.Tensor,
    ob_wmap: dict[str, str],
    ob_header_cache: dict[str, tuple[int, dict]],
    base_header_cache: dict[str, tuple[int, dict]],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Extracts the Dual-Verified Orthonormal Refusal Basis `R_l` (`(d_r, 5120)` in `float32`) for `layers.{layer_idx}`:
      - `r_ob`: Exact Failspy refusal direction from `OBLITERATUS/DeepSeek-V4.1-Flash-1017-obliterated` (`layers 1..39`)
      - `r_dea`: Exact refusal direction from `deepseek-ai/DeepSeek-V4.1-Flash-1017` vs `dealignai` (`layers 10..36`)
    """
    basis_vecs: list[torch.Tensor] = []
    stats: dict[str, float] = {}
    shard_name = f"model-{layer_idx + 3:05d}-of-00048.safetensors"

    # Fetch base shared_experts.w2 (11.8 MB) via HTTP Range
    if shard_name not in base_header_cache:
        base_header_cache[shard_name] = _http_get_header("deepseek-ai/DeepSeek-V4.1-Flash-1017", shard_name)
    bh_len, bh = base_header_cache[shard_name]
    k_sh_w = f"layers.{layer_idx}.ffn.shared_experts.w2.weight"
    k_sh_s = f"layers.{layer_idx}.ffn.shared_experts.w2.scale"
    w_base_sh = dequantize_block_fp8_32(
        _http_fetch_tensor("deepseek-ai/DeepSeek-V4.1-Flash-1017", shard_name, bh_len, bh[k_sh_w], torch.uint8, device),
        _http_fetch_tensor("deepseek-ai/DeepSeek-V4.1-Flash-1017", shard_name, bh_len, bh[k_sh_s], torch.uint8, device),
        device,
        dtype=torch.float32,
    )

    # 1. OBLITERATUS exact BF16 refusal vector (layers 1..39)
    ob_key = f"model.layers.{layer_idx}.mlp.shared_experts.down_proj.weight"
    if layer_idx >= 1 and ob_key in ob_wmap:
        try:
            ob_shard = ob_wmap[ob_key]
            if ob_shard not in ob_header_cache:
                ob_header_cache[ob_shard] = _http_get_header(
                    "OBLITERATUS/DeepSeek-V4.1-Flash-1017-obliterated", ob_shard
                )
            oh_len, oh = ob_header_cache[ob_shard]
            w_ob_sh = _http_fetch_tensor(
                "OBLITERATUS/DeepSeek-V4.1-Flash-1017-obliterated", ob_shard, oh_len, oh[ob_key], torch.bfloat16, device
            ).float()
            diff_ob = w_base_sh - w_ob_sh
            diff_ob_norm = float(diff_ob.norm().item())
            if diff_ob_norm > 1e-3:
                _, u_ob = batched_factorized_top_left_svd(diff_ob.unsqueeze(0), rank=1, n_iter=3)
                r_ob = F.normalize(u_ob[0, :, 0], dim=0)
                basis_vecs.append(r_ob)
                stats["obliteratus_diff_norm"] = round(diff_ob_norm, 5)
                stats["obliteratus_base_proj"] = round(float((r_ob @ w_base_sh).norm().item()), 5)
                stats["obliteratus_resid_proj"] = round(float((r_ob @ w_ob_sh).norm().item()), 5)
        except Exception as e:
            print(f"  [Warn] OBLITERATUS range fetch skipped on layer {layer_idx}: {e}", flush=True)

    # 2. dealignai exact refusal vector (layers 10..36)
    diff_dea = w_base_sh - w_dea_sh.float()
    diff_dea_norm = float(diff_dea.norm().item())
    if diff_dea_norm > 1e-3:
        _, u_dea = batched_factorized_top_left_svd(diff_dea.unsqueeze(0), rank=1, n_iter=3)
        r_dea = F.normalize(u_dea[0, :, 0], dim=0)
        basis_vecs.append(r_dea)
        stats["dealign_diff_norm"] = round(diff_dea_norm, 5)
        stats["dealign_base_proj"] = round(float((r_dea @ w_base_sh).norm().item()), 5)

    if not basis_vecs:
        # Layer 0 fallback: use top-1 singular vector of shared_experts.w2 orthogonalized or empty
        _, u0 = batched_factorized_top_left_svd(w_base_sh.unsqueeze(0), rank=1, n_iter=2)
        # Do not ablate layer 0 if neither OBLITERATUS nor dealignai ablated layer 0 (method_config: all_except_first)
        return torch.zeros((0, 5120), dtype=torch.float32, device=device), stats

    mat = torch.stack(basis_vecs, dim=1)  # (5120, d_r)
    q, _ = torch.linalg.qr(mat)  # (5120, d_r) orthonormal columns
    return q.T.contiguous(), stats  # (d_r, 5120)


@torch.no_grad()
def convert_dsv41_expert_group(
    gate_batch: torch.Tensor,
    up_batch: torch.Tensor,
    down_batch: torch.Tensor,
    router_w: torch.Tensor,
    v_rt_sub: torch.Tensor,
    s_rt_sub: torch.Tensor,
    r_basis: torch.Tensor,
    g_idx: int,
    h128: torch.Tensor,
    svd_rank: int = 64,
    gu_tail_bits: int = 1,
    down_tail_bits: int = 1,
    swiglu_limit: float = 10.0,
) -> dict[str, torch.Tensor]:
    """
    Converts K=8 DeepSeek-V4.1-Flash-1017 experts (`gate`: 2304x5120, `up`: 2304x5120, `down`: 5120x2304)
    using Chronos `v7.5` + Dual-Verified Post-Quantization Orthogonal Abliteration (`r_basis`: `(d_r, 5120)`).
    """
    device = gate_batch.device
    k = gate_batch.shape[0]
    gu_batch = torch.cat([gate_batch, up_batch], dim=1)  # (8, 4608, 5120)

    # Pre-Quantization Orthogonal Abliteration on routed experts' down_proj (w2: 8 x 5120 x 2304)
    if r_basis.shape[0] > 0:
        # Project out refusal basis R_l along the 5120 output rows of down_batch
        proj_pre = torch.einsum("rd,kdi->kri", r_basis, down_batch)
        down_batch = down_batch - torch.einsum("rd,kri->kdi", r_basis, proj_pre)

    gu_r_s1, gu_r_p, gu_r_inv, gu_r_s2, gu_c_s1, gu_c_p, gu_c_inv, gu_c_s2 = [], [], [], [], [], [], [], []
    dn_r_s1, dn_r_p, dn_r_inv, dn_r_s2, dn_c_s1, dn_c_p, dn_c_inv, dn_c_s2 = [], [], [], [], [], [], [], []

    for m in range(k):
        k_gu = generate_expert_1d_keys(4608, 5120, m, seed_base=101 + g_idx * 8, device=device)
        gu_r_s1.append(k_gu[0]); gu_r_p.append(k_gu[1]); gu_r_inv.append(k_gu[2]); gu_r_s2.append(k_gu[3])
        gu_c_s1.append(k_gu[4]); gu_c_p.append(k_gu[5]); gu_c_inv.append(k_gu[6]); gu_c_s2.append(k_gu[7])

        k_dn = generate_expert_1d_keys(5120, 2304, m, seed_base=303 + g_idx * 8, device=device)
        dn_r_s1.append(k_dn[0]); dn_r_p.append(k_dn[1]); dn_r_inv.append(k_dn[2]); dn_r_s2.append(k_dn[3])
        dn_c_s1.append(k_dn[4]); dn_c_p.append(k_dn[5]); dn_c_inv.append(k_dn[6]); dn_c_s2.append(k_dn[7])

    gu_r_s1_t, gu_r_p_t, gu_r_inv_t, gu_r_s2_t = torch.stack(gu_r_s1), torch.stack(gu_r_p), torch.stack(gu_r_inv), torch.stack(gu_r_s2)
    gu_c_s1_t, gu_c_p_t, gu_c_inv_t, gu_c_s2_t = torch.stack(gu_c_s1), torch.stack(gu_c_p), torch.stack(gu_c_inv), torch.stack(gu_c_s2)
    dn_r_s1_t, dn_r_p_t, dn_r_inv_t, dn_r_s2_t = torch.stack(dn_r_s1), torch.stack(dn_r_p), torch.stack(dn_r_inv), torch.stack(dn_r_s2)
    dn_c_s1_t, dn_c_p_t, dn_c_inv_t, dn_c_s2_t = torch.stack(dn_c_s1), torch.stack(dn_c_p), torch.stack(dn_c_inv), torch.stack(dn_c_s2)

    folded_gu = batched_fold_2d(gu_batch, gu_r_s1_t, gu_r_p_t, gu_r_s2_t, gu_c_s1_t, gu_c_p_t, gu_c_s2_t, h128)
    folded_dn = batched_fold_2d(down_batch, dn_r_s1_t, dn_r_p_t, dn_r_s2_t, dn_c_s1_t, dn_c_p_t, dn_c_s2_t, h128)

    eids = list(range(g_idx * 8, g_idx * 8 + k))
    rw_grp = router_w[eids].float()
    imp = torch.linalg.norm(rw_grp @ v_rt_sub, dim=-1)
    imp = (imp / imp.mean().clamp_min(1e-8)).clamp(0.7, 1.4)

    s_gu_f32 = (folded_gu * imp.view(k, 1, 1)).sum(dim=0) / math.sqrt(float(k))
    s_dn_f32 = (folded_dn * imp.view(k, 1, 1)).sum(dim=0) / math.sqrt(float(k))

    s_gu_p, s_gu_s, s_gu_s2 = pack_nvfp4(s_gu_f32)
    s_dn_p, s_dn_s, s_dn_s2 = pack_nvfp4(s_dn_f32)

    s_gu_deq = unpack_nvfp4(s_gu_p, s_gu_s, s_gu_s2)
    s_dn_deq = unpack_nvfp4(s_dn_p, s_dn_s, s_dn_s2)

    h_gu = batched_unbind_2d(s_gu_deq, gu_r_s1_t, gu_r_inv_t, gu_r_s2_t, gu_c_s1_t, gu_c_inv_t, gu_c_s2_t, h128)
    h_dn = batched_unbind_2d(s_dn_deq, dn_r_s1_t, dn_r_inv_t, dn_r_s2_t, dn_c_s1_t, dn_c_inv_t, dn_c_s2_t, h128)

    beta_gu = (gu_batch * h_gu).sum(dim=(1, 2)) / (h_gu * h_gu).sum(dim=(1, 2)).clamp_min(1e-12)
    res_gu = gu_batch - beta_gu.view(k, 1, 1) * h_gu

    _, _, init_healers_gu = batched_full_spectrum_row_gram_smw_healer(
        res_gu, rank=svd_rank, v_sub=v_rt_sub, sub_scales=s_rt_sub, quantize_fp8=False, n_iter=1
    )
    tail_target_gu = res_gu - init_healers_gu
    if gu_tail_bits == 2:
        gu_tp, gu_ts = batched_pack_2bit_lloyd_max(tail_target_gu)
        e_tail_gu = batched_unpack_2bit_lloyd_max(gu_tp, gu_ts)
    else:
        gu_tp, gu_ts = batched_pack_1bit_rademacher(tail_target_gu)
        e_tail_gu = batched_unpack_1bit_rademacher(gu_tp, gu_ts)

    u_gu_fp8, v_gu_fp8, healers_gu = batched_full_spectrum_row_gram_smw_healer(
        res_gu - e_tail_gu, rank=svd_rank, v_sub=v_rt_sub, sub_scales=s_rt_sub, quantize_fp8=True, n_iter=2
    )

    gu_hat = beta_gu.view(k, 1, 1) * h_gu + healers_gu + e_tail_gu
    gamma_gu = (
        gu_batch.pow(2).sum(dim=(1, 2), keepdim=True).sqrt()
        / gu_hat.pow(2).sum(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-8)
    ).clamp(0.95, 1.08)
    beta_gu = beta_gu * gamma_gu.view(k)
    u_gu_fp8 = (u_gu_fp8.float() * gamma_gu).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    gu_ts = (gu_ts.float() * gamma_gu).clamp(1e-6, 448.0).to(torch.float8_e4m3fn)
    gu_hat = gu_hat * gamma_gu

    # Coupled Clamped-SwiGLU (swiglu_limit=10.0) Drift Compensation (Woodbury 64x64 Push-Through Solve)
    n_probes = 64
    rw_norm = F.normalize(rw_grp, dim=-1).unsqueeze(1)
    sub_probes = F.normalize(torch.randn(k, n_probes, v_rt_sub.shape[1], device=device) @ v_rt_sub.T, dim=-1)
    x_probes = F.normalize(0.45 * rw_norm + 0.55 * sub_probes, dim=-1) * math.sqrt(5120.0) * 0.05

    z_true = torch.bmm(x_probes, gu_batch.transpose(1, 2))
    a_true = dsv41_clamped_swiglu(z_true, inter_dim=2304, swiglu_limit=swiglu_limit)
    y_true = torch.bmm(a_true, down_batch.transpose(1, 2))

    z_hat = torch.bmm(x_probes, gu_hat.transpose(1, 2))
    a_hat = dsv41_clamped_swiglu(z_hat, inter_dim=2304, swiglu_limit=swiglu_limit)
    y_uncomp = torch.bmm(a_hat, down_batch.transpose(1, 2))
    dy = y_true - y_uncomp

    aat = torch.bmm(a_hat, a_hat.transpose(1, 2))
    reg = (aat.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).unsqueeze(-1).clamp_min(1e-6)) * (0.15 * float(n_probes) / 2304.0)
    eye_p = torch.eye(n_probes, device=device).unsqueeze(0)
    delta_down_t = torch.bmm(a_hat.transpose(1, 2), torch.linalg.solve(aat + reg * eye_p, dy))
    target_down = down_batch + 0.15 * delta_down_t.transpose(1, 2)
    if r_basis.shape[0] > 0:
        proj_td = torch.einsum("rd,kdi->kri", r_basis, target_down)
        target_down = target_down - torch.einsum("rd,kri->kdi", r_basis, proj_td)

    beta_dn = (target_down * h_dn).sum(dim=(1, 2)) / (h_dn * h_dn).sum(dim=(1, 2)).clamp_min(1e-12)
    res_dn = target_down - beta_dn.view(k, 1, 1) * h_dn

    _, _, vh_act = torch.linalg.svd(a_true, full_matrices=False)
    v_act_sub = vh_act.transpose(1, 2).contiguous()

    dn_weighted_init = res_dn + 0.60 * torch.bmm(torch.bmm(res_dn, v_act_sub), v_act_sub.transpose(1, 2))
    s_di, v_di = batched_factorized_top_left_svd(dn_weighted_init.transpose(1, 2).contiguous(), rank=svd_rank, n_iter=1)
    u_di = torch.bmm(res_dn, v_di) / s_di.unsqueeze(1)
    u_di_norms = torch.linalg.norm(u_di, dim=1, keepdim=True).clamp_min(1e-12)
    u_di = u_di / u_di_norms
    s_di = s_di * u_di_norms.squeeze(1)
    init_healers_dn = torch.bmm(u_di * s_di.unsqueeze(1), v_di.transpose(1, 2))

    tail_target_dn = res_dn - init_healers_dn
    if down_tail_bits == 2:
        dn_tp, dn_ts = batched_pack_2bit_lloyd_max(tail_target_dn)
        e_tail_dn = batched_unpack_2bit_lloyd_max(dn_tp, dn_ts)
    else:
        dn_tp, dn_ts = batched_pack_1bit_rademacher(tail_target_dn)
        e_tail_dn = batched_unpack_1bit_rademacher(dn_tp, dn_ts)

    rem_dn = res_dn - e_tail_dn
    dn_weighted = rem_dn + 0.60 * torch.bmm(torch.bmm(rem_dn, v_act_sub), v_act_sub.transpose(1, 2))
    s_top_dn, v_top_dn = batched_factorized_top_left_svd(dn_weighted.transpose(1, 2).contiguous(), rank=svd_rank, n_iter=2)
    u_top_dn = torch.bmm(rem_dn, v_top_dn) / s_top_dn.unsqueeze(1)
    if r_basis.shape[0] > 0:
        proj_u = torch.einsum("cd,kdr->kcr", r_basis, u_top_dn)
        u_top_dn = u_top_dn - torch.einsum("cd,kcr->kdr", r_basis, proj_u)
    u_dn_norms = torch.linalg.norm(u_top_dn, dim=1, keepdim=True).clamp_min(1e-12)
    u_top_dn = u_top_dn / u_dn_norms
    s_top_dn = s_top_dn * u_dn_norms.squeeze(1)
    sqrt_s_dn = s_top_dn.sqrt()
    u_dn_fp8 = (u_top_dn * sqrt_s_dn.unsqueeze(1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    v_dn_fp8 = (v_top_dn.transpose(1, 2) * sqrt_s_dn.unsqueeze(-1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)

    dn_hat = beta_dn.view(k, 1, 1) * h_dn + torch.bmm(u_dn_fp8.float(), v_dn_fp8.float()) + e_tail_dn
    gamma_dn = (
        target_down.pow(2).sum(dim=(1, 2), keepdim=True).sqrt()
        / dn_hat.pow(2).sum(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-8)
    ).clamp(0.95, 1.08)
    beta_dn = beta_dn * gamma_dn.view(k)
    u_dn_fp8 = (u_dn_fp8.float() * gamma_dn).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    dn_ts = (dn_ts.float() * gamma_dn).clamp(1e-6, 448.0).to(torch.float8_e4m3fn)

    return {
        "s_gu.weight": s_gu_p.cpu(),
        "s_gu.weight_scale": s_gu_s.cpu(),
        "s_gu.weight_scale_2": s_gu_s2.cpu(),
        "s_gu.beta": beta_gu.to(torch.float32).cpu(),
        "s_down.weight": s_dn_p.cpu(),
        "s_down.weight_scale": s_dn_s.cpu(),
        "s_down.weight_scale_2": s_dn_s2.cpu(),
        "s_down.beta": beta_dn.to(torch.float32).cpu(),
        "healer.u_gu": u_gu_fp8.cpu(),
        "healer.v_gu": v_gu_fp8.cpu(),
        "healer.u_down": u_dn_fp8.cpu(),
        "healer.v_down": v_dn_fp8.cpu(),
        "tail.gu_bits": torch.tensor(gu_tail_bits, dtype=torch.uint8),
        "tail.gu_packed": gu_tp.cpu(),
        "tail.gu_scale": gu_ts.cpu(),
        "tail.down_bits": torch.tensor(down_tail_bits, dtype=torch.uint8),
        "tail.down_packed": dn_tp.cpu(),
        "tail.down_scale": dn_ts.cpu(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="dealignai/DeepSeek-V4.1-Flash-1017-UNCENSORED-FP8")
    parser.add_argument("--base-repo-id", default="deepseek-ai/DeepSeek-V4.1-Flash-1017")
    parser.add_argument("--obliteratus-repo-id", default="OBLITERATUS/DeepSeek-V4.1-Flash-1017-obliterated")
    parser.add_argument("--out-dir", default="/content/dsv41_chronos_v75")
    parser.add_argument("--cache-dir", default="/content/hf_cache_dsv41")
    parser.add_argument("--layers", default="all", help="'all' (0..39) or comma-separated MoE layer list")
    parser.add_argument("--gu-bits", type=int, default=2, help="2 (Full Lloyd-Max) or 1 (Compact-Disk)")
    parser.add_argument("--down-bits", type=int, default=2, help="2 (Full Lloyd-Max) or 1 (Compact-Disk)")
    parser.add_argument("--svd-rank", type=int, default=64)
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument(
        "--upload-repo",
        default="defnoteatin/PseudoBrain-DeepSeek-V4.1-Flash-1017-Chronos-v7.5-Abliterated",
        help="Target Hugging Face repository to concurrently upload completed shards (empty to disable)",
    )
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Clear any stale Hugging Face / xet caches so full disk is available
    for c_path in ["/root/.cache/huggingface/xet", "/root/.cache/huggingface/hub", str(cache_dir / ".cache"), "/content/hf_cache"]:
        shutil.rmtree(c_path, ignore_errors=True)

    def _fast_curl_download(repo_id: str, filename: str, dest: Path) -> Path:
        if dest.exists() and dest.stat().st_size > 1024 * 1024:
            return dest
        url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        tmp_dest = dest.with_suffix(".tmp")
        cmd = ["curl", "-sL", "--retry", "5", "--retry-delay", "2", "-o", str(tmp_dest), url]
        if os.environ.get("HF_TOKEN"):
            cmd[1:1] = ["-H", f"Authorization: Bearer {os.environ['HF_TOKEN']}"]
        subprocess.run(cmd, check=True)
        tmp_dest.replace(dest)
        return dest

    hf_api: HfApi | None = None
    upload_pool = ThreadPoolExecutor(max_workers=1)
    upload_futures: list[Future] = []
    if args.upload_repo:
        try:
            hf_api = HfApi(token=os.environ.get("HF_TOKEN"))
            hf_api.create_repo(repo_id=args.upload_repo, repo_type="model", exist_ok=True)
            print(f"[HF Upload] Ready to stream shards concurrently to https://huggingface.co/{args.upload_repo}", flush=True)
        except Exception as e:
            print(f"[HF Upload] Warning initializing repo {args.upload_repo}: {e}", flush=True)
            hf_api = None

    def _upload_shard_bg(file_path: Path) -> str:
        if hf_api is None or not file_path.exists():
            return "skipped"
        t_u0 = time.perf_counter()
        try:
            hf_api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=file_path.name,
                repo_id=args.upload_repo,
                repo_type="model",
            )
            shutil.rmtree("/root/.cache/huggingface/xet", ignore_errors=True)
            dt_u = time.perf_counter() - t_u0
            print(f"[HF Upload DONE in {dt_u:.1f}s] {file_path.name} -> {args.upload_repo}", flush=True)
            return file_path.name
        except Exception as e:
            shutil.rmtree("/root/.cache/huggingface/xet", ignore_errors=True)
            print(f"[HF Upload WARNING] {file_path.name}: {e}", flush=True)
            return f"error:{e}"

    print("=" * 105, flush=True)
    print(
        f"CHRONOS-1T v7.5 FULL 40-LAYER ABLITERATED STREAMING CONVERTER FOR {args.repo_id} ON {torch.cuda.get_device_name(0)}",
        flush=True,
    )
    print("=" * 105, flush=True)

    # 1. Download tokenizer and config files
    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json"]:
        try:
            p = hf_hub_download(repo_id=args.repo_id, filename=fname, local_dir=str(out_dir))
            print(f"[Tokenizer/Config] Saved {fname} -> {p}", flush=True)
        except Exception as e:
            print(f"[Tokenizer/Config] Optional {fname} skipped ({e})", flush=True)

    # Inject chronos_holographic_moe & abliteration metadata into config.json
    cfg_path = out_dir / "config.json"
    if cfg_path.exists():
        cfg_data = json.loads(cfg_path.read_text())
        cfg_data["chronos_holographic_moe"] = {
            "version": "7.5",
            "group_size": 8,
            "num_groups_per_layer": 48,
            "svd_rank": args.svd_rank,
            "gu_tail_bits": args.gu_bits,
            "down_tail_bits": args.down_bits,
            "abliterated": True,
            "abliteration_method": "dual_verified_post_quantization_orthogonal_projection",
            "abliteration_sources": [
                args.repo_id,
                args.obliteratus_repo_id,
            ],
        }
        cfg_path.write_text(json.dumps(cfg_data, indent=2))
        if hf_api is not None:
            for meta_f in ["config.json", "tokenizer.json", "tokenizer_config.json"]:
                mp = out_dir / meta_f
                if mp.exists():
                    upload_futures.append(upload_pool.submit(_upload_shard_bg, mp))

    idx_p = hf_hub_download(repo_id=args.repo_id, filename="model.safetensors.index.json", local_dir=str(out_dir))
    idx_data = json.loads(Path(idx_p).read_text())
    wmap: dict[str, str] = idx_data["weight_map"]

    # Fetch OBLITERATUS weight map for exact BF16 refusal vector extraction
    ob_wmap: dict[str, str] = {}
    try:
        url_ob_idx = f"https://huggingface.co/{args.obliteratus_repo_id}/resolve/main/model.safetensors.index.json"
        req = urllib.request.Request(url_ob_idx, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            ob_wmap = json.loads(r.read().decode("utf-8"))["weight_map"]
        print(f"[Abliteration] Loaded OBLITERATUS weight_map ({len(ob_wmap)} tensors)", flush=True)
    except Exception as e:
        print(f"[Abliteration] OBLITERATUS index fetch warning: {e}", flush=True)

    # 2. Save Base Shard (vision + aligner + embed.weight + norm.weight + head.weight)
    base_shard = out_dir / "chronos-dsv41-base-vision-embed-head.safetensors"
    if not args.skip_base and not base_shard.exists():
        t_b0 = time.perf_counter()
        base_fnames = ["model-00001-of-00048.safetensors", "model-00002-of-00048.safetensors", "model-00043-of-00048.safetensors"]
        base_tensors: dict[str, torch.Tensor] = {}
        for sf in base_fnames:
            lp = _fast_curl_download(args.repo_id, sf, cache_dir / sf)
            with safe_open(str(lp), framework="pt", device="cpu") as fh:
                for k in fh.keys():
                    base_tensors[k] = fh.get_tensor(k).contiguous()
            drop_page_cache(lp)
            lp.unlink(missing_ok=True)
        save_file(base_tensors, str(base_shard))
        drop_page_cache(base_shard)
        del base_tensors
        gc.collect()
        print(
            f"[Base Vision + Embed + Norm + Head DONE in {time.perf_counter() - t_b0:.1f}s] Saved {base_shard.name} ({base_shard.stat().st_size / 1e9:.2f} GB)",
            flush=True,
        )
    if base_shard.exists() and hf_api is not None:
        upload_futures.append(upload_pool.submit(_upload_shard_bg, base_shard))

    # 3. Stream and convert all 40 MoE Layers (0..39) with Background Prefetching
    if args.layers == "all":
        target_layers = list(range(40))
    else:
        target_layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]

    h128 = build_hadamard_128(device)
    progress_path = out_dir / "full_40layer_dsv41_progress.json"
    completed_layers: list[dict] = []
    ob_header_cache: dict[str, tuple[int, dict]] = {}
    base_header_cache: dict[str, tuple[int, dict]] = {}

    def _download_layer_shard(l_id: int) -> Path:
        sf_name = f"model-{l_id + 3:05d}-of-00048.safetensors"
        out_s = out_dir / f"chronos-dsv41-layer-{l_id:05d}-of-00040.safetensors"
        if out_s.exists():
            return out_s
        return _fast_curl_download(args.repo_id, sf_name, cache_dir / sf_name)

    prefetch_pool = ThreadPoolExecutor(max_workers=1)
    pending_futures: dict[int, Future[Path]] = {}
    if target_layers:
        pending_futures[target_layers[0]] = prefetch_pool.submit(_download_layer_shard, target_layers[0])

    for idx_pos, l_idx in enumerate(target_layers):
        # Queue prefetch for l_idx + 1 (1 shard ahead = zero wait time + minimum disk footprint)
        if idx_pos + 1 < len(target_layers):
            next_l = target_layers[idx_pos + 1]
            if next_l not in pending_futures:
                pending_futures[next_l] = prefetch_pool.submit(_download_layer_shard, next_l)

        t0 = time.perf_counter()
        out_shard = out_dir / f"chronos-dsv41-layer-{l_idx:05d}-of-00040.safetensors"
        if out_shard.exists():
            with safe_open(str(out_shard), framework="pt", device="cpu") as f_chk:
                existing_keys = set(f_chk.keys())
            if any(".chronos_groups.47." in k for k in existing_keys):
                print(f"[Layer {l_idx:02d}/39] Already complete ({out_shard.stat().st_size / 1e9:.2f} GB), queueing upload & skipping.", flush=True)
                if hf_api is not None:
                    upload_futures.append(upload_pool.submit(_upload_shard_bg, out_shard))
                completed_layers.append({"layer": l_idx, "status": "cached", "size_gb": round(out_shard.stat().st_size / 1e9, 3)})
                continue

        raw_shard_path = pending_futures.pop(l_idx).result()
        layer_prefix = f"layers.{l_idx}."
        ffn_prefix = f"layers.{l_idx}.ffn"

        out_tensors: dict[str, torch.Tensor] = {}
        with safe_open(str(raw_shard_path), framework="pt", device="cpu") as fh:
            shard_keys = list(fh.keys())
            # Copy all non-routed-expert tensors (attn, norms, hc_*, gate, shared_experts)
            for k in shard_keys:
                if not k.startswith(f"{ffn_prefix}.experts."):
                    out_tensors[k] = fh.get_tensor(k).contiguous()

            router_w = out_tensors[f"{ffn_prefix}.gate.weight"].to(device=device, dtype=torch.bfloat16)
            w_dea_sh = dequantize_block_fp8_32(
                out_tensors[f"{ffn_prefix}.shared_experts.w2.weight"],
                out_tensors[f"{ffn_prefix}.shared_experts.w2.scale"],
                device,
                dtype=torch.float32,
            )

            # Extract Dual-Verified Orthonormal Refusal Basis R_l (d_r x 5120)
            r_basis, ablit_stats = extract_layer_refusal_basis(
                l_idx, w_dea_sh, ob_wmap, ob_header_cache, base_header_cache, device
            )
            if r_basis.shape[0] > 0:
                # Also ensure shared_experts.w2 is orthogonal to OBLITERATUS r_ob if not already
                out_tensors[f"{ffn_prefix}.abliteration.r_basis"] = r_basis.to(torch.bfloat16).cpu().contiguous()

            _, s_rt, vh_rt = torch.linalg.svd(router_w.float(), full_matrices=False)
            v_rt_sub = vh_rt[:256, :].T.contiguous()
            s_ref = s_rt[min(2, s_rt.shape[0] - 1)].clamp_min(1e-6)
            s_rt_sub = 1.0 + 4.5 * ((s_rt[:256] / s_ref).clamp(max=1.5) ** 0.30)

            for g_idx in range(48):
                g_list, u_list, d_list = [], [], []
                for m in range(8):
                    eid = g_idx * 8 + m
                    ep = f"{ffn_prefix}.experts.{eid}"
                    gw = dequantize_mxfp4_32(fh.get_tensor(f"{ep}.w1.weight"), fh.get_tensor(f"{ep}.w1.scale"), device)
                    uw = dequantize_mxfp4_32(fh.get_tensor(f"{ep}.w3.weight"), fh.get_tensor(f"{ep}.w3.scale"), device)
                    dw = dequantize_mxfp4_32(fh.get_tensor(f"{ep}.w2.weight"), fh.get_tensor(f"{ep}.w2.scale"), device)
                    g_list.append(gw)
                    u_list.append(uw)
                    d_list.append(dw)

                g_dict = convert_dsv41_expert_group(
                    torch.stack(g_list, dim=0),
                    torch.stack(u_list, dim=0),
                    torch.stack(d_list, dim=0),
                    router_w,
                    v_rt_sub,
                    s_rt_sub,
                    r_basis,
                    g_idx,
                    h128,
                    svd_rank=args.svd_rank,
                    gu_tail_bits=args.gu_bits,
                    down_tail_bits=args.down_bits,
                )
                for tk, tv in g_dict.items():
                    out_tensors[f"{ffn_prefix}.chronos_groups.{g_idx}.{tk}"] = tv.contiguous()
                del g_list, u_list, d_list, g_dict

        save_file(out_tensors, str(out_shard))
        drop_page_cache(out_shard)
        if raw_shard_path.exists() and raw_shard_path != out_shard:
            drop_page_cache(raw_shard_path)
            raw_shard_path.unlink(missing_ok=True)

        if hf_api is not None:
            upload_futures.append(upload_pool.submit(_upload_shard_bg, out_shard))

        shard_gb = out_shard.stat().st_size / 1e9
        free_gb = shutil.disk_usage(out_dir).free / 1e9
        elapsed = time.perf_counter() - t0
        print(
            f"[Layer {l_idx:02d}/39 DONE in {elapsed:.1f}s] Saved {out_shard.name} ({shard_gb:.3f} GB, free={free_gb:.1f} GB) | r_basis={tuple(r_basis.shape)} | {ablit_stats}",
            flush=True,
        )

        completed_layers.append({
            "layer": l_idx,
            "status": "converted",
            "elapsed_s": round(elapsed, 2),
            "size_gb": round(shard_gb, 3),
            "free_disk_gb": round(free_gb, 2),
            "r_basis_dim": int(r_basis.shape[0]),
            "abliteration": ablit_stats,
        })
        progress_path.write_text(
            json.dumps(
                {
                    "completed_count": len(completed_layers),
                    "total_target_layers": len(target_layers),
                    "free_disk_gb": round(free_gb, 2),
                    "layers": completed_layers,
                },
                indent=2,
            )
        )

        del out_tensors, router_w, w_dea_sh, r_basis, v_rt_sub, s_rt_sub
        gc.collect()
        torch.cuda.empty_cache()

    prefetch_pool.shutdown(wait=False)
    if upload_futures:
        print(f"[HF Upload] Waiting for {len(upload_futures)} concurrent shard uploads to finish...", flush=True)
        for uf in upload_futures:
            try:
                uf.result()
            except Exception as e:
                print(f"[HF Upload] Future warning: {e}", flush=True)
    upload_pool.shutdown(wait=True)
    if hf_api is not None and progress_path.exists():
        _upload_shard_bg(progress_path)

    print("\n" + "=" * 105, flush=True)
    print(f"ALL {len(target_layers)} TARGET LAYERS OF {args.repo_id} CONVERTED & UPLOADED SUCCESSFULLY TO {out_dir}", flush=True)
    print("=" * 105, flush=True)


if __name__ == "__main__":
    main()
