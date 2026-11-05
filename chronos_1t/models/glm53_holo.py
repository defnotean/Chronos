#!/usr/bin/env python3
"""
Chronos-1T v7.5 x GLM-5.3 — Holographic Expert-Group Folding (converter side) & Reconstruction (runtime side).

Both directions live in one module so the converter (`convert_glm53_chronos.py`) and the runtime
(`glm53_chronos_moe.py`) can never drift apart. The math is the Chronos v7.5 pipeline used for GLM-5.3-Flash and
DeepSeek-V4.1-Flash (`convert_glm5_expert_group` / `convert_dsv41_expert_group`), generalized to arbitrary
`(hidden, moe_intermediate)` so it covers GLM-5.3's `gate_up 4096 x 6144` / `down 6144 x 2048` experts:

  1. 8:1 NVFP4 holographic bank per group (permuted-Hadamard + Rademacher binding, router-importance weighted)
  2. Coupled rank-64 FP8 SMW SVD healer on the router's principal subspace
  3. 2-bit Lloyd-Max (standard) or 1-bit Rademacher (compact) isotropic tail lattice
  4. Woodbury 64x64 SwiGLU drift compensation folded into `down_proj` + post-tail variance restoration

One deliberate difference vs. Flash/DSV4.1: the per-expert permutation/sign keys are drawn from a **CPU**
`torch.Generator` and then moved to the target device. CUDA Philox streams are not guaranteed to match CPU streams
(or every GPU launch configuration), so CPU keys make a GLM-5.3 checkpoint converted on a Colab A100 bit-exactly
reconstructable on a DGX Spark GB10, an H100, or a CPU. This is recorded as
`chronos_holographic_moe.key_generator = "torch_cpu_v1"` in `config.json`.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

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
    pack_nvfp4,
    unpack_nvfp4,
)
from chronos_1t.models.glm53_chronos_spec import GROUP_SIZE, group_seed_bases

KEY_GENERATOR = "torch_cpu_v1"
GROUP_TENSOR_SUFFIXES = (
    "s_gu.weight", "s_gu.weight_scale", "s_gu.weight_scale_2", "s_gu.beta",
    "s_down.weight", "s_down.weight_scale", "s_down.weight_scale_2", "s_down.beta",
    "healer.u_gu", "healer.v_gu", "healer.u_down", "healer.v_down",
    "tail.gu_bits", "tail.gu_packed", "tail.gu_scale",
    "tail.down_bits", "tail.down_packed", "tail.down_scale",
)

_KEY_CACHE: dict[tuple[int, int, int, int, str], tuple[torch.Tensor, ...]] = {}
_H128_CACHE: dict[str, torch.Tensor] = {}


def hadamard128(device: torch.device) -> torch.Tensor:
    key = str(device)
    h = _H128_CACHE.get(key)
    if h is None:
        h = build_hadamard_128(device)
        _H128_CACHE[key] = h
    return h


def expert_keys(rows: int, cols: int, local_m: int, seed_base: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    """(r_s1, r_perm, r_inv, r_s2, c_s1, c_perm, c_inv, c_s2) drawn from a CPU generator (device independent)."""
    ck = (rows, cols, local_m, seed_base, str(device))
    hit = _KEY_CACHE.get(ck)
    if hit is not None:
        return hit
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed_base + (local_m + 1) * 104729)
    r_s1 = torch.randint(0, 2, (rows,), generator=gen, dtype=torch.int8).float() * 2.0 - 1.0
    r_perm = torch.randperm(rows, generator=gen)
    r_s2 = torch.randint(0, 2, (rows,), generator=gen, dtype=torch.int8).float() * 2.0 - 1.0
    c_s1 = torch.randint(0, 2, (cols,), generator=gen, dtype=torch.int8).float() * 2.0 - 1.0
    c_perm = torch.randperm(cols, generator=gen)
    c_s2 = torch.randint(0, 2, (cols,), generator=gen, dtype=torch.int8).float() * 2.0 - 1.0
    out = tuple(
        t.to(device)
        for t in (r_s1, r_perm, torch.argsort(r_perm), r_s2, c_s1, c_perm, torch.argsort(c_perm), c_s2)
    )
    if len(_KEY_CACHE) < 4096:
        _KEY_CACHE[ck] = out
    return out


def stacked_keys(rows: int, cols: int, seed_base: int, members: list[int], device: torch.device) -> list[torch.Tensor]:
    per = [expert_keys(rows, cols, m, seed_base, device) for m in members]
    return [torch.stack([p[i] for p in per]) for i in range(8)]


def router_subspace(router_w: torch.Tensor, rank: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    """Router principal subspace `V` (H x r) and SMW importance weights, exactly as in the Flash/DSV4.1 converters."""
    _, s_rt, vh_rt = torch.linalg.svd(router_w.float(), full_matrices=False)
    r = min(rank, vh_rt.shape[0])
    v_rt_sub = vh_rt[:r, :].T.contiguous()
    s_ref = s_rt[min(2, s_rt.shape[0] - 1)].clamp_min(1e-6)
    s_rt_sub = 1.0 + 4.5 * ((s_rt[:r] / s_ref).clamp(max=1.5) ** 0.30)
    return v_rt_sub, s_rt_sub


def make_router_probes(
    router_w: torch.Tensor,
    v_rt_sub: torch.Tensor,
    n: int,
    norm_weight: torch.Tensor | None = None,
    seed: int = 2026,
) -> torch.Tensor:
    """Synthetic post-attention-RMSNorm activations living on the router manifold (used for healing & metrics)."""
    dev = router_w.device
    h = router_w.shape[1]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    eids = torch.randint(0, router_w.shape[0], (n,), generator=gen)
    rw = F.normalize(router_w[eids.to(dev)].float(), dim=-1)
    sub = F.normalize(torch.randn(n, v_rt_sub.shape[1], generator=gen).to(dev) @ v_rt_sub.T, dim=-1)
    x = F.normalize(0.45 * rw + 0.55 * sub, dim=-1) * math.sqrt(float(h))
    if norm_weight is not None:
        x = x * norm_weight.to(device=dev, dtype=torch.float32).unsqueeze(0)
    return x


def _cos_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Row cosine in float64 (`F.cosine_similarity` in fp32 drifts by ~0.5% on 25M-element expert matrices)."""
    a2, b2 = a.flatten(1).double(), b.flatten(1).double()
    return (a2 * b2).sum(1) / (a2.norm(dim=1) * b2.norm(dim=1)).clamp_min(1e-30)


@torch.no_grad()
def router_prior(
    router_w: torch.Tensor,
    e_bias: torch.Tensor,
    top_k: int,
    norm_weight: torch.Tensor | None = None,
    layer_idx: int = 0,
    n_manifold: int = 4096,
    n_iso: int = 2048,
) -> torch.Tensor:
    """Expected routed gate mass per expert under the sigmoid `noaux_tc` router (DSV4.1 Upgrade-2 style Monte Carlo
    over router-manifold probes + the RMSNorm-scaled isotropic sphere). Shared by the converter (hot-expert sidecar
    selection) and the runtime (tier ranking) so both pick the same experts."""
    dev = router_w.device
    rw = router_w.float()
    h = rw.shape[1]
    v_sub, _ = router_subspace(rw)
    x = make_router_probes(rw, v_sub, n_manifold, norm_weight, seed=42 + layer_idx)
    gen = torch.Generator(device="cpu").manual_seed(4242 + layer_idx)
    iso = F.normalize(torch.randn(n_iso, h, generator=gen).to(dev), dim=-1) * (h**0.5)
    if norm_weight is not None:
        iso = iso * norm_weight.to(device=dev, dtype=torch.float32)
    x = torch.cat([x, iso])
    scores = torch.sigmoid(x @ rw.T)
    _, ids = torch.topk(scores + e_bias.to(dev, torch.float32), top_k, dim=-1)
    w = scores.gather(1, ids)
    w = w / w.sum(-1, keepdim=True).clamp_min(1e-20)
    return torch.zeros(rw.shape[0], device=dev).scatter_add_(0, ids.flatten(), w.flatten()).cpu()


@torch.no_grad()
def convert_expert_group(
    gate_b: torch.Tensor,
    up_b: torch.Tensor,
    down_b: torch.Tensor,
    router_w: torch.Tensor,
    v_rt_sub: torch.Tensor,
    s_rt_sub: torch.Tensor,
    g_idx: int,
    svd_rank: int = 64,
    gu_tail_bits: int = 2,
    down_tail_bits: int = 2,
    metric_probes: torch.Tensor | None = None,
    tier_metrics: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Folds K=8 GLM-5.3 experts (`gate/up: I x H`, `down: H x I`, fp32 on device) into one Chronos v7.5 group."""
    device = gate_b.device
    k, inter, hid = gate_b.shape
    h128 = hadamard128(device)
    gu_b = torch.cat([gate_b, up_b], dim=1)  # (k, 2I, H)
    members = list(range(k))
    seed_gu, seed_dn = group_seed_bases(g_idx)
    gk = stacked_keys(2 * inter, hid, seed_gu, members, device)
    dk = stacked_keys(hid, inter, seed_dn, members, device)

    folded_gu = batched_fold_2d(gu_b, gk[0], gk[1], gk[3], gk[4], gk[5], gk[7], h128)
    folded_dn = batched_fold_2d(down_b, dk[0], dk[1], dk[3], dk[4], dk[5], dk[7], h128)

    eids = list(range(g_idx * GROUP_SIZE, g_idx * GROUP_SIZE + k))
    rw_grp = router_w[eids].float()
    imp = torch.linalg.norm(rw_grp @ v_rt_sub, dim=-1)
    imp = (imp / imp.mean().clamp_min(1e-8)).clamp(0.7, 1.4)

    s_gu_f32 = (folded_gu * imp.view(k, 1, 1)).sum(dim=0) / math.sqrt(float(k))
    s_dn_f32 = (folded_dn * imp.view(k, 1, 1)).sum(dim=0) / math.sqrt(float(k))
    del folded_gu, folded_dn
    s_gu_p, s_gu_s, s_gu_s2 = pack_nvfp4(s_gu_f32)
    s_dn_p, s_dn_s, s_dn_s2 = pack_nvfp4(s_dn_f32)
    del s_gu_f32, s_dn_f32
    h_gu = batched_unbind_2d(unpack_nvfp4(s_gu_p, s_gu_s, s_gu_s2), gk[0], gk[2], gk[3], gk[4], gk[6], gk[7], h128)
    h_dn = batched_unbind_2d(unpack_nvfp4(s_dn_p, s_dn_s, s_dn_s2), dk[0], dk[2], dk[3], dk[4], dk[6], dk[7], h128)

    beta_gu = (gu_b * h_gu).sum(dim=(1, 2)) / (h_gu * h_gu).sum(dim=(1, 2)).clamp_min(1e-12)
    res_gu = gu_b - beta_gu.view(k, 1, 1) * h_gu
    _, _, init_heal_gu = batched_full_spectrum_row_gram_smw_healer(
        res_gu, rank=svd_rank, v_sub=v_rt_sub, sub_scales=s_rt_sub, quantize_fp8=False, n_iter=1
    )
    if gu_tail_bits == 2:
        gu_tp, gu_ts = batched_pack_2bit_lloyd_max(res_gu - init_heal_gu)
        e_tail_gu = batched_unpack_2bit_lloyd_max(gu_tp, gu_ts)
    else:
        gu_tp, gu_ts = batched_pack_1bit_rademacher(res_gu - init_heal_gu)
        e_tail_gu = batched_unpack_1bit_rademacher(gu_tp, gu_ts)
    del init_heal_gu
    u_gu, v_gu, heal_gu = batched_full_spectrum_row_gram_smw_healer(
        res_gu - e_tail_gu, rank=svd_rank, v_sub=v_rt_sub, sub_scales=s_rt_sub, quantize_fp8=True, n_iter=2
    )
    del res_gu
    gu_hat = beta_gu.view(k, 1, 1) * h_gu + heal_gu + e_tail_gu
    gamma_gu = (
        gu_b.pow(2).sum(dim=(1, 2), keepdim=True).sqrt() / gu_hat.pow(2).sum(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-8)
    ).clamp(0.95, 1.08)
    beta_gu = beta_gu * gamma_gu.view(k)
    u_gu = (u_gu.float() * gamma_gu).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    gu_ts = (gu_ts.float() * gamma_gu).clamp(1e-6, 448.0).to(torch.float8_e4m3fn)
    gu_hat = gu_hat * gamma_gu
    gu_notail = (beta_gu.view(k, 1, 1) * h_gu + torch.bmm(u_gu.float(), v_gu.float())) if metric_probes is not None else None
    del heal_gu, e_tail_gu

    # Coupled SwiGLU drift compensation (Woodbury 64x64 push-through) folded into down_proj
    n_probes = 64
    gen = torch.Generator(device="cpu").manual_seed(7000 + g_idx)
    rw_norm = F.normalize(rw_grp, dim=-1).unsqueeze(1)
    sub_probes = F.normalize(torch.randn(k, n_probes, v_rt_sub.shape[1], generator=gen).to(device) @ v_rt_sub.T, dim=-1)
    x_probes = F.normalize(0.45 * rw_norm + 0.55 * sub_probes, dim=-1) * math.sqrt(float(hid)) * 0.05
    z_true = torch.bmm(x_probes, gu_b.transpose(1, 2))
    a_true = F.silu(z_true[:, :, :inter]) * z_true[:, :, inter:]
    y_true = torch.bmm(a_true, down_b.transpose(1, 2))
    z_hat = torch.bmm(x_probes, gu_hat.transpose(1, 2))
    a_hat = F.silu(z_hat[:, :, :inter]) * z_hat[:, :, inter:]
    dy = y_true - torch.bmm(a_hat, down_b.transpose(1, 2))
    aat = torch.bmm(a_hat, a_hat.transpose(1, 2))
    reg = aat.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).unsqueeze(-1).clamp_min(1e-6) * (
        0.15 * float(n_probes) / float(inter)
    )
    eye_p = torch.eye(n_probes, device=device).unsqueeze(0)
    delta_down_t = torch.bmm(a_hat.transpose(1, 2), torch.linalg.solve(aat + reg * eye_p, dy))
    target_down = down_b + 0.45 * delta_down_t.transpose(1, 2)
    del z_true, z_hat, dy, aat, delta_down_t

    beta_dn = (target_down * h_dn).sum(dim=(1, 2)) / (h_dn * h_dn).sum(dim=(1, 2)).clamp_min(1e-12)
    res_dn = target_down - beta_dn.view(k, 1, 1) * h_dn
    _, _, vh_act = torch.linalg.svd(a_true, full_matrices=False)
    v_act = vh_act.transpose(1, 2).contiguous()
    del a_true, a_hat, vh_act

    def _act_weighted(m: torch.Tensor) -> torch.Tensor:
        return m + 0.60 * torch.bmm(torch.bmm(m, v_act), v_act.transpose(1, 2))

    s_di, v_di = batched_factorized_top_left_svd(_act_weighted(res_dn).transpose(1, 2).contiguous(), rank=svd_rank, n_iter=1)
    u_di = torch.bmm(res_dn, v_di) / s_di.unsqueeze(1)
    u_nrm = torch.linalg.norm(u_di, dim=1, keepdim=True).clamp_min(1e-12)
    init_heal_dn = torch.bmm((u_di / u_nrm) * (s_di * u_nrm.squeeze(1)).unsqueeze(1), v_di.transpose(1, 2))
    if down_tail_bits == 2:
        dn_tp, dn_ts = batched_pack_2bit_lloyd_max(res_dn - init_heal_dn)
        e_tail_dn = batched_unpack_2bit_lloyd_max(dn_tp, dn_ts)
    else:
        dn_tp, dn_ts = batched_pack_1bit_rademacher(res_dn - init_heal_dn)
        e_tail_dn = batched_unpack_1bit_rademacher(dn_tp, dn_ts)
    del init_heal_dn, u_di, v_di
    rem_dn = res_dn - e_tail_dn
    s_top, v_top = batched_factorized_top_left_svd(_act_weighted(rem_dn).transpose(1, 2).contiguous(), rank=svd_rank, n_iter=2)
    u_top = torch.bmm(rem_dn, v_top) / s_top.unsqueeze(1)
    u_nrm = torch.linalg.norm(u_top, dim=1, keepdim=True).clamp_min(1e-12)
    u_top = u_top / u_nrm
    s_top = s_top * u_nrm.squeeze(1)
    sq = s_top.sqrt()
    u_dn = (u_top * sq.unsqueeze(1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    v_dn = (v_top.transpose(1, 2) * sq.unsqueeze(-1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    del rem_dn, res_dn, u_top, v_top
    dn_hat = beta_dn.view(k, 1, 1) * h_dn + torch.bmm(u_dn.float(), v_dn.float()) + e_tail_dn
    gamma_dn = (
        target_down.pow(2).sum(dim=(1, 2), keepdim=True).sqrt()
        / dn_hat.pow(2).sum(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-8)
    ).clamp(0.95, 1.08)
    beta_dn = beta_dn * gamma_dn.view(k)
    u_dn = (u_dn.float() * gamma_dn).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    dn_ts = (dn_ts.float() * gamma_dn).clamp(1e-6, 448.0).to(torch.float8_e4m3fn)
    dn_hat = dn_hat * gamma_dn
    del target_down, e_tail_dn

    tensors = {
        "s_gu.weight": s_gu_p, "s_gu.weight_scale": s_gu_s, "s_gu.weight_scale_2": s_gu_s2,
        "s_gu.beta": beta_gu.to(torch.float32),
        "s_down.weight": s_dn_p, "s_down.weight_scale": s_dn_s, "s_down.weight_scale_2": s_dn_s2,
        "s_down.beta": beta_dn.to(torch.float32),
        "healer.u_gu": u_gu, "healer.v_gu": v_gu, "healer.u_down": u_dn, "healer.v_down": v_dn,
        "tail.gu_bits": torch.tensor(gu_tail_bits, dtype=torch.uint8),
        "tail.gu_packed": gu_tp, "tail.gu_scale": gu_ts,
        "tail.down_bits": torch.tensor(down_tail_bits, dtype=torch.uint8),
        "tail.down_packed": dn_tp, "tail.down_scale": dn_ts,
    }
    tensors = {kk: v.detach().contiguous().cpu() for kk, v in tensors.items()}

    metrics: dict[str, Any] = {}
    if metric_probes is not None:
        dn_notail = beta_dn.view(k, 1, 1) * h_dn + torch.bmm(u_dn.float(), v_dn.float())
        metrics["w_cos_gu"] = _cos_rows(gu_hat, gu_b).tolist()
        metrics["w_cos_down"] = _cos_rows(dn_hat, down_b).tolist()
        metrics["w_cos_gu_notail"] = _cos_rows(gu_notail, gu_b).tolist()
        metrics["w_cos_down_notail"] = _cos_rows(dn_notail, down_b).tolist()
        xp = metric_probes.float()

        def _expert_out(gu: torch.Tensor, dn: torch.Tensor) -> torch.Tensor:
            z = torch.einsum("ph,koh->kpo", xp, gu)
            return torch.einsum("kpi,khi->kph", F.silu(z[..., :inter]) * z[..., inter:], dn)

        y_ref = _expert_out(gu_b, down_b)
        metrics["func_cos"] = _cos_rows(_expert_out(gu_hat, dn_hat), y_ref).tolist()
        metrics["func_cos_notail"] = _cos_rows(_expert_out(gu_notail, dn_notail), y_ref).tolist()
        if tier_metrics:
            from chronos_1t.models import glm53_tiers as T

            for tier, name in ((T.TIER_T1, "t1"), (T.TIER_T2, "t2"), (T.TIER_T3, "t3")):
                pk, up_ = T.PACKERS[tier], T.UNPACKERS[tier]
                rq_gu = up_(*pk(gu_hat), dtype=torch.float32)
                rq_dn = up_(*pk(dn_hat), dtype=torch.float32)
                metrics[f"func_cos_{name}_from_chronos"] = _cos_rows(_expert_out(rq_gu, rq_dn), y_ref).tolist()
                dq_gu = up_(*pk(gu_b), dtype=torch.float32)
                dq_dn = up_(*pk(down_b), dtype=torch.float32)
                metrics[f"func_cos_{name}_from_source"] = _cos_rows(_expert_out(dq_gu, dq_dn), y_ref).tolist()
                del rq_gu, rq_dn, dq_gu, dq_dn
        del dn_notail, y_ref
    return tensors, metrics


@torch.no_grad()
def reconstruct_group_experts(
    g: dict[str, torch.Tensor],
    g_idx: int,
    members: list[int],
    hidden: int,
    inter: int,
    include_tail: bool = True,
    dtype: torch.dtype = torch.float32,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Rebuilds `(gate_up (2I x H), down (H x I))` for `members` of one group. `g` holds the group tensors on device."""
    device = g["s_gu.weight"].device
    h128 = hadamard128(device)
    seed_gu, seed_dn = group_seed_bases(g_idx)
    s_gu = unpack_nvfp4(g["s_gu.weight"], g["s_gu.weight_scale"], g["s_gu.weight_scale_2"])
    s_dn = unpack_nvfp4(g["s_down.weight"], g["s_down.weight_scale"], g["s_down.weight_scale_2"])
    gu_bits = int(g["tail.gu_bits"].item()) if "tail.gu_bits" in g else 2
    dn_bits = int(g["tail.down_bits"].item()) if "tail.down_bits" in g else 2
    out: list[tuple[torch.Tensor, torch.Tensor]] = []
    for m in members:
        kg = expert_keys(2 * inter, hidden, m, seed_gu, device)
        kd = expert_keys(hidden, inter, m, seed_dn, device)
        h_gu = batched_unbind_2d(s_gu, kg[0][None], kg[2][None], kg[3][None], kg[4][None], kg[6][None], kg[7][None], h128)[0]
        h_dn = batched_unbind_2d(s_dn, kd[0][None], kd[2][None], kd[3][None], kd[4][None], kd[6][None], kd[7][None], h128)[0]
        gu = float(g["s_gu.beta"][m]) * h_gu + g["healer.u_gu"][m].float() @ g["healer.v_gu"][m].float()
        dn = float(g["s_down.beta"][m]) * h_dn + g["healer.u_down"][m].float() @ g["healer.v_down"][m].float()
        if include_tail:
            gp, gs = g["tail.gu_packed"][m : m + 1], g["tail.gu_scale"][m : m + 1]
            dp, ds = g["tail.down_packed"][m : m + 1], g["tail.down_scale"][m : m + 1]
            gu += (batched_unpack_2bit_lloyd_max(gp, gs) if gu_bits == 2 else batched_unpack_1bit_rademacher(gp, gs))[0]
            dn += (batched_unpack_2bit_lloyd_max(dp, ds) if dn_bits == 2 else batched_unpack_1bit_rademacher(dp, ds))[0]
        out.append((gu.to(dtype), dn.to(dtype)))
    return out
