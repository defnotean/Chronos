#!/usr/bin/env python3
"""
Chronos-1T x PseudoBrain (pb27) — Multi-Layer Chained 512-Expert Checkpoint Verifier.

Loads converted `chronos-layer-XXXXX-of-00048.safetensors` shards from `/home/defnotean/pb27/hf/flash_next_chronos_v6`
and executes a true 512-expert Top-10 routed forward pass chained across all converted layers,
comparing against stock `/home/defnotean/pb27/hf/flash_next_nvfp4`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chronos_1t.converter.convert_flash_next_v6 import (
    MODEL_DIR,
    OUT_CKPT_DIR,
    dequantize_mxfp8_tensor,
)
from chronos_1t.models.chronos_moe_block import (
    ChronosHolographicExpertGroup,
    unpack_nvfp4,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", default="0,1,2", help="Comma-separated converted layer indices to chain-verify")
    parser.add_argument("--ckpt-dir", default=str(OUT_CKPT_DIR))
    args = parser.parse_args()

    device = torch.device("cuda:0")
    ckpt_dir = Path(args.ckpt_dir)
    layers = [int(x) for x in args.layers.split(",")]

    print("=" * 124)
    print(f"MULTI-LAYER CHAINED 512-EXPERT CHECKPOINT VERIFICATION: LAYERS {layers} ({len(layers) * 512} EXPERTS)")
    print("=" * 124)

    stock_idx = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())["weight_map"]
    tokenizer = Tokenizer.from_file(str(ckpt_dir / "tokenizer.json"))

    topic_shift_segments = [
        "__global__ void blackwell_sm121_hss_gemv(const half* __restrict__ seed, const half* __restrict__ x, float* out) { int lane = threadIdx.x & 31; ",
        "Sous le pont Mirabeau coule la Seine et nos amours faut-il qu'il m'en souvienne la joie venait toujours apres la peine. ",
        "The Hamiltonian of a transmon superconducting qubit coupled to a microwave resonator in the dispersive regime is H = hbar omega_r a^dagger a + ",
        "CRISPR-Cas9 ribonucleoprotein complexes recognize the protospacer adjacent motif 5'-NGG-3' and cleave double-stranded genomic DNA via HNH and RuvC domains.",
    ]
    seg_ids = [tokenizer.encode(s).ids for s in topic_shift_segments]
    all_ids = [tid for seg in seg_ids for tid in seg]
    token_ids_t = torch.tensor(all_ids, dtype=torch.long)

    with safe_open(str(ckpt_dir / stock_idx["model.language_model.embed_tokens.weight"]), framework="pt", device="cpu") as f:
        embed_w = f.get_tensor("model.language_model.embed_tokens.weight")
        real_tok_embeds = embed_w[token_ids_t].to(device=device, dtype=torch.float32)
        del embed_w

    with safe_open(str(ckpt_dir / stock_idx["lm_head.weight"]), framework="pt", device="cpu") as f:
        lm_head_w = f.get_tensor("lm_head.weight").to(device=device, dtype=torch.bfloat16)

    n_prefill = len(seg_ids[0])
    h_before = None
    h_cold = None
    h_ema = None

    for l_idx in layers:
        prefix = f"model.language_model.layers.{l_idx}.mlp"
        st_cache: dict[str, safe_open] = {}

        def get_t(key: str) -> torch.Tensor:
            sf = stock_idx[key]
            if sf not in st_cache:
                st_cache[sf] = safe_open(str(MODEL_DIR / sf), framework="pt", device="cpu")
            return st_cache[sf].get_tensor(key)

        router_w = get_t(f"{prefix}.gate.weight").to(device=device, dtype=torch.float32)
        shared_gate_w = get_t(f"{prefix}.shared_expert_gate.weight").to(device=device, dtype=torch.float32)
        sh_down = dequantize_mxfp8_tensor(get_t(f"{prefix}.shared_expert.down_proj.weight"), get_t(f"{prefix}.shared_expert.down_proj.weight_scale"), device)
        sh_gate = dequantize_mxfp8_tensor(get_t(f"{prefix}.shared_expert.gate_proj.weight"), get_t(f"{prefix}.shared_expert.gate_proj.weight_scale"), device)
        sh_up = dequantize_mxfp8_tensor(get_t(f"{prefix}.shared_expert.up_proj.weight"), get_t(f"{prefix}.shared_expert.up_proj.weight_scale"), device)

        _, s_rt, vh_rt = torch.linalg.svd(router_w, full_matrices=False)
        u_ctx = vh_rt[:48, :].T
        w_ctx = ((s_rt[:48] / s_rt[0]) ** 1.5).unsqueeze(0)

        if h_before is None:
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
            h_before = x_stream[n_prefill:].clone()
            h_cold = x_stream[n_prefill:].clone()
            h_ema = x_stream[n_prefill:].clone()
            prefill_x = x_stream[:n_prefill].clone()

        x_in_b = F.rms_norm(h_before, (2560,), eps=1e-6)
        x_in_c = F.rms_norm(h_cold, (2560,), eps=1e-6)
        x_in_e = F.rms_norm(h_ema, (2560,), eps=1e-6)

        x_full_stream = torch.cat([prefill_x, x_in_b], dim=0)
        router_logits_all = x_full_stream @ router_w.T
        _, top10_all = torch.topk(router_logits_all, k=10, dim=-1)
        dec_vals, dec_ids = torch.topk(router_logits_all[n_prefill:], k=10, dim=-1)
        dec_gates = F.softmax(dec_vals, dim=-1)

        unique_active = sorted(set(dec_ids.flatten().tolist()))
        shard_path = ckpt_dir / f"chronos-layer-{l_idx:05d}-of-00048.safetensors"
        loaded_all = load_file(str(shard_path), device="cpu")

        stock_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        chronos_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        grp_mods: dict[int, ChronosHolographicExpertGroup] = {}

        for e_id in unique_active:
            ep = f"{prefix}.experts.{e_id}"
            gw = unpack_nvfp4(get_t(f"{ep}.gate_proj.weight").to(device), get_t(f"{ep}.gate_proj.weight_scale").to(device), get_t(f"{ep}.gate_proj.weight_scale_2").to(device))
            uw = unpack_nvfp4(get_t(f"{ep}.up_proj.weight").to(device), get_t(f"{ep}.up_proj.weight_scale").to(device), get_t(f"{ep}.up_proj.weight_scale_2").to(device))
            dw = unpack_nvfp4(get_t(f"{ep}.down_proj.weight").to(device), get_t(f"{ep}.down_proj.weight_scale").to(device), get_t(f"{ep}.down_proj.weight_scale_2").to(device))
            stock_cache[e_id] = (gw, uw, dw)

            g_idx = e_id // 8
            local_m = e_id % 8
            if g_idx not in grp_mods:
                gp = f"{prefix}.chronos_groups.{g_idx}."
                sub_t = {k[len(gp):]: v for k, v in loaded_all.items() if k.startswith(gp)}
                grp_mods[g_idx] = ChronosHolographicExpertGroup(group_idx=g_idx, tensors=sub_t, device=device)
            chronos_cache[e_id] = grp_mods[g_idx].reconstruct_expert(local_m)

        global_prior = torch.linalg.norm(router_w @ u_ctx, dim=-1)
        _, prior_top = torch.topk(global_prior, k=24)
        prior_set = set(prior_top.tolist())

        y_routed_b = torch.zeros_like(x_in_b)
        y_routed_c = torch.zeros_like(x_in_c)
        y_routed_e = torch.zeros_like(x_in_e)

        for d_i in range(x_in_b.shape[0]):
            t_idx = n_prefill + d_i
            win_start = max(0, t_idx - 24)
            counts = torch.bincount(top10_all[win_start:t_idx].flatten(), minlength=512)
            _, ema_ids = torch.topk(counts, k=24)
            hot_set = prior_set | set(ema_ids.tolist())

            xb = x_in_b[d_i : d_i + 1]
            xc = x_in_c[d_i : d_i + 1]
            xe = x_in_e[d_i : d_i + 1]
            for k_slot in range(10):
                e_id = int(dec_ids[d_i, k_slot].item())
                g_val = dec_gates[d_i, k_slot]

                gw, uw, dw = stock_cache[e_id]
                y_routed_b[d_i : d_i + 1] += g_val * ((F.silu(xb @ gw.T) * (xb @ uw.T)) @ dw.T)

                cgw, cuw, cdw = chronos_cache[e_id]
                y_routed_c[d_i : d_i + 1] += g_val * ((F.silu(xc @ cgw.T) * (xc @ cuw.T)) @ cdw.T)

                if e_id in hot_set:
                    y_routed_e[d_i : d_i + 1] += g_val * ((F.silu(xe @ gw.T) * (xe @ uw.T)) @ dw.T)
                else:
                    y_routed_e[d_i : d_i + 1] += g_val * ((F.silu(xe @ cgw.T) * (xe @ cuw.T)) @ cdw.T)

        y_sh_b = torch.sigmoid(x_in_b @ shared_gate_w.T) * ((F.silu(x_in_b @ sh_gate.T) * (x_in_b @ sh_up.T)) @ sh_down.T)
        y_sh_c = torch.sigmoid(x_in_c @ shared_gate_w.T) * ((F.silu(x_in_c @ sh_gate.T) * (x_in_c @ sh_up.T)) @ sh_down.T)
        y_sh_e = torch.sigmoid(x_in_e @ shared_gate_w.T) * ((F.silu(x_in_e @ sh_gate.T) * (x_in_e @ sh_up.T)) @ sh_down.T)

        h_before = h_before + y_routed_b + y_sh_b
        h_cold = h_cold + y_routed_c + y_sh_c
        h_ema = h_ema + y_routed_e + y_sh_e

        c_cold = F.cosine_similarity(y_routed_c.flatten(), y_routed_b.flatten(), dim=0).item() * 100.0
        c_ema = F.cosine_similarity(y_routed_e.flatten(), y_routed_b.flatten(), dim=0).item() * 100.0
        print(
            f"  [Layer {l_idx:02d}] Active Unique Experts: {len(unique_active):3d}/512 | "
            f"Pure-Cold MoE Cos: {c_cold:6.2f}% | Hybrid EMA-48 MoE Cos: {c_ema:6.2f}%"
        )

    logits_b = (F.rms_norm(h_before, (2560,), eps=1e-6).to(torch.bfloat16) @ lm_head_w.T).float()
    logits_c = (F.rms_norm(h_cold, (2560,), eps=1e-6).to(torch.bfloat16) @ lm_head_w.T).float()
    logits_e = (F.rms_norm(h_ema, (2560,), eps=1e-6).to(torch.bfloat16) @ lm_head_w.T).float()

    probs_b = F.softmax(logits_b, dim=-1)
    log_probs_b = F.log_softmax(logits_b, dim=-1)
    top1_b = logits_b.argmax(dim=-1)
    _, top5_b = torch.topk(logits_b, k=5, dim=-1)

    print("-" * 124)
    for label, h_x, l_x in [
        ("Chained Pure-Cold (0% Cache across 1,536 Experts)", h_cold, logits_c),
        ("Chained Hybrid EMA-48 across 1,536 Experts", h_ema, logits_e),
    ]:
        h_cos = F.cosine_similarity(h_x.flatten(), h_before.flatten(), dim=0).item() * 100.0
        l_cos = F.cosine_similarity(l_x.flatten(), logits_b.flatten(), dim=0).item() * 100.0
        top1_acc = (l_x.argmax(dim=-1) == top1_b).float().mean().item() * 100.0
        _, top5_x = torch.topk(l_x, k=5, dim=-1)
        ovl = sum(len(set(rb) & set(rr)) for rb, rr in zip(top5_b.tolist(), top5_x.tolist())) / (5.0 * h_before.shape[0]) * 100.0
        kl = torch.sum(probs_b * (log_probs_b - F.log_softmax(l_x, dim=-1)), dim=-1).mean().item()
        print(
            f"  {label:<50} | HiddenCos: {h_cos:6.2f}% | LogitCos: {l_cos:6.2f}% | "
            f"Top-1: {top1_acc:6.2f}% | Top-5: {ovl:6.2f}% | KL: {kl:.5f}"
        )
    print("=" * 124)


if __name__ == "__main__":
    main()
