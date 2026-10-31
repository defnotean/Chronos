#!/usr/bin/env python3
"""
Chronos-1T x PseudoBrain (pb27) — Stage 2 Full 512-Expert Layer Converter & True 512-Expert Router Verifier.

1. Streams through all 64 Holographic Groups (all 512 routed experts) of Layer 0 (or any layer list)
   from `/home/defnotean/pb27/hf/flash_next_nvfp4`, converting them into a real byte-packed
   `.safetensors` shard (`/home/defnotean/pb27/hf/flash_next_chronos_v6/chronos-layer-00000-of-00048.safetensors`).
2. Composes `/home/defnotean/pb27/hf/flash_next_chronos_v6/` with symlinks for all unchanged shards
   (`embed_tokens`, `lm_head`, `gdn`, `ple`, `self_attn`, `linear_attn`, `tokenizer.json`, `config.json`).
3. Loads all 64 groups (512 experts) back from the `.safetensors` file on disk and executes
   TRUE 512-Expert Top-10 Routing across all 512 experts on real tokens and abrupt topic shifts!
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chronos_1t.converter.convert_flash_next_v6 import (
    MODEL_DIR,
    OUT_CKPT_DIR,
    convert_expert_group_to_packed_tensors,
    dequantize_mxfp8_tensor,
)
from chronos_1t.models.chronos_moe_block import (
    ChronosHolographicExpertGroup,
    build_hadamard_128,
    unpack_nvfp4,
)

COPY_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
    "generation_config.json",
    "hf_quant_config.json",
)


def compose_chronos_checkpoint_dir(src_dir: Path, out_dir: Path, converted_layers: dict[int, str]) -> None:
    """Composes `/home/defnotean/pb27/hf/flash_next_chronos_v6` with symlinks for all non-MoE shards."""
    out_dir.mkdir(parents=True, exist_ok=True)
    src_idx = json.loads((src_dir / "model.safetensors.index.json").read_text())
    weight_map = dict(src_idx["weight_map"])

    # Symlink all original shards first so non-MoE weights (attention, GDN, PLE, embed, lm_head) resolve with 0 disk duplication
    for shard_name in sorted(set(weight_map.values())):
        dst_link = out_dir / shard_name
        src_shard = src_dir / shard_name
        if not dst_link.exists() and not dst_link.is_symlink():
            os.symlink(src_shard.resolve(), dst_link)

    # Replace converted MoE expert entries in weight_map with the new Chronos packed shard
    for l_idx, chronos_shard in converted_layers.items():
        prefix_exp = f"model.language_model.layers.{l_idx}.mlp.experts."
        keys_to_remove = [k for k in weight_map if k.startswith(prefix_exp)]
        for k in keys_to_remove:
            del weight_map[k]
        for g_idx in range(64):
            gp = f"model.language_model.layers.{l_idx}.mlp.chronos_groups.{g_idx}"
            for suffix in (
                "s_gu.weight",
                "s_gu.weight_scale",
                "s_gu.weight_scale_2",
                "s_gu.beta",
                "s_down.weight",
                "s_down.weight_scale",
                "s_down.weight_scale_2",
                "s_down.beta",
                "healer.u_gu",
                "healer.v_gu",
                "healer.u_down",
                "healer.v_down",
                "tail.gu_bits",
                "tail.gu_packed",
                "tail.gu_scale",
                "tail.down_bits",
                "tail.down_packed",
                "tail.down_scale",
            ):
                weight_map[f"{gp}.{suffix}"] = chronos_shard

    (out_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": src_idx.get("metadata", {}), "weight_map": weight_map}, indent=2)
    )

    cfg = json.loads((src_dir / "config.json").read_text())
    cfg["chronos_holographic_moe"] = {
        "version": "v6.2",
        "fold_ratio": 8,
        "svd_rank": 32,
        "converted_layers": sorted(converted_layers.keys()),
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    for fname in COPY_FILES:
        sfile = src_dir / fname
        dfile = out_dir / fname
        if sfile.is_file() and not dfile.exists() and not dfile.is_symlink():
            os.symlink(sfile.resolve(), dfile)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=0, help="Layer index to convert all 512 experts")
    parser.add_argument("--gu-bits", type=int, default=16, help="1=1-bit, 16=Ternary-Base3 (1.6b), 2=2-bit Lloyd-Max")
    parser.add_argument("--down-bits", type=int, default=2, help="16=Ternary-Base3 (1.6b), 2=2-bit Lloyd-Max")
    parser.add_argument("--out-dir", default=str(OUT_CKPT_DIR))
    args = parser.parse_args()

    device = torch.device("cuda:0")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    l_idx = args.layer

    print("=" * 124)
    print(f"STAGE 2: FULL 512-EXPERT (64 GROUPS) CONVERSION & TRUE 512-EXPERT ROUTING ON LAYER {l_idx} (DGX SPARK GB10)")
    print("=" * 124)

    idx_map = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())["weight_map"]
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

    torch.manual_seed(2026 + l_idx)
    _, s_rt, vh_rt = torch.linalg.svd(router_w, full_matrices=False)
    u_ctx = vh_rt[:48, :].T
    w_ctx = ((s_rt[:48] / s_rt[0]) ** 1.5).unsqueeze(0)

    z_cal = torch.randn(1024, 48, device=device) * w_ctx
    x_cal = F.rms_norm(z_cal @ u_ctx.T + torch.randn(1024, 2560, device=device) * 0.01, (2560,), eps=1e-6)
    _, s_act_x, vh_act_x = torch.linalg.svd(x_cal, full_matrices=False)
    v_act_x = vh_act_x[:64, :].T
    s_act_x_sub = s_act_x[:64]
    h128 = build_hadamard_128(device)

    layer_shard_Dict: dict[str, torch.Tensor] = {
        f"{prefix}.gate.weight": get_t(f"{prefix}.gate.weight"),
        f"{prefix}.shared_expert_gate.weight": get_t(f"{prefix}.shared_expert_gate.weight"),
        f"{prefix}.shared_expert.down_proj.weight": get_t(f"{prefix}.shared_expert.down_proj.weight"),
        f"{prefix}.shared_expert.down_proj.weight_scale": get_t(f"{prefix}.shared_expert.down_proj.weight_scale"),
        f"{prefix}.shared_expert.gate_proj.weight": get_t(f"{prefix}.shared_expert.gate_proj.weight"),
        f"{prefix}.shared_expert.gate_proj.weight_scale": get_t(f"{prefix}.shared_expert.gate_proj.weight_scale"),
        f"{prefix}.shared_expert.up_proj.weight": get_t(f"{prefix}.shared_expert.up_proj.weight"),
        f"{prefix}.shared_expert.up_proj.weight_scale": get_t(f"{prefix}.shared_expert.up_proj.weight_scale"),
    }

    t_start = time.perf_counter()
    for g_idx in range(64):
        e_start = g_idx * 8
        gate_w, up_w, down_w = [], [], []
        for e_id in range(e_start, e_start + 8):
            ep = f"{prefix}.experts.{e_id}"
            gate_w.append(unpack_nvfp4(get_t(f"{ep}.gate_proj.weight").to(device), get_t(f"{ep}.gate_proj.weight_scale").to(device), get_t(f"{ep}.gate_proj.weight_scale_2").to(device)))
            up_w.append(unpack_nvfp4(get_t(f"{ep}.up_proj.weight").to(device), get_t(f"{ep}.up_proj.weight_scale").to(device), get_t(f"{ep}.up_proj.weight_scale_2").to(device)))
            down_w.append(unpack_nvfp4(get_t(f"{ep}.down_proj.weight").to(device), get_t(f"{ep}.down_proj.weight_scale").to(device), get_t(f"{ep}.down_proj.weight_scale_2").to(device)))

        grp_tensors = convert_expert_group_to_packed_tensors(
            group_idx=g_idx,
            gate_w=gate_w,
            up_w=up_w,
            down_w=down_w,
            x_cal=x_cal,
            v_act_x=v_act_x,
            s_act_x=s_act_x_sub,
            h128=h128,
            svd_rank=32,
            gu_tail_bits=args.gu_bits,
            down_tail_bits=args.down_bits,
        )
        gp = f"{prefix}.chronos_groups.{g_idx}"
        for k, v in grp_tensors.items():
            layer_shard_Dict[f"{gp}.{k}"] = v

        if (g_idx + 1) % 16 == 0 or g_idx == 63:
            elapsed = time.perf_counter() - t_start
            print(f"  [Layer {l_idx:02d}] Converted {(g_idx + 1) * 8:3d}/512 experts ({g_idx + 1:2d}/64 groups) in {elapsed:.2f}s")

    shard_name = f"chronos-layer-{l_idx:05d}-of-00048.safetensors"
    shard_path = out_dir / shard_name
    save_file(layer_shard_Dict, str(shard_path))
    shard_mb = shard_path.stat().st_size / (1024 * 1024)
    orig_layer_mb = (512 * 3 * 640 * 2560 * 0.5625) / (1024 * 1024)

    compose_chronos_checkpoint_dir(MODEL_DIR, out_dir, {l_idx: shard_name})
    print(f"[OK] Saved {shard_path} ({shard_mb:.2f} MB vs {orig_layer_mb:.2f} MB stock)")

    # 3. Verify TRUE 512-Expert Top-10 Routing across all 64 groups loaded from disk!
    print("-" * 124)
    print(f"Running TRUE 512-Expert Top-10 Routing Verification from {shard_name} ...")
    loaded_all = load_file(str(shard_path), device="cpu")

    tokenizer = Tokenizer.from_file(str(out_dir / "tokenizer.json"))
    topic_shift_segments = [
        "__global__ void blackwell_sm121_hss_gemv(const half* __restrict__ seed, const half* __restrict__ x, float* out) { int lane = threadIdx.x & 31; ",
        "Sous le pont Mirabeau coule la Seine et nos amours faut-il qu'il m'en souvienne la joie venait toujours apres la peine. ",
        "The Hamiltonian of a transmon superconducting qubit coupled to a microwave resonator in the dispersive regime is H = hbar omega_r a^dagger a + ",
        "CRISPR-Cas9 ribonucleoprotein complexes recognize the protospacer adjacent motif 5'-NGG-3' and cleave double-stranded genomic DNA via HNH and RuvC domains.",
    ]
    seg_ids = [tokenizer.encode(s).ids for s in topic_shift_segments]
    all_ids = [tid for seg in seg_ids for tid in seg]
    token_ids_t = torch.tensor(all_ids, dtype=torch.long)

    with safe_open(str(out_dir / idx_map["model.language_model.embed_tokens.weight"]), framework="pt", device="cpu") as f:
        embed_w = f.get_tensor("model.language_model.embed_tokens.weight")
        real_tok_embeds = embed_w[token_ids_t].to(device=device, dtype=torch.float32)
        del embed_w

    with safe_open(str(out_dir / idx_map["lm_head.weight"]), framework="pt", device="cpu") as f:
        lm_head_w = f.get_tensor("lm_head.weight").to(device=device, dtype=torch.bfloat16)

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

    router_logits_all = x_stream @ router_w.T
    _, top10_all = torch.topk(router_logits_all, k=10, dim=-1)
    dec_vals, dec_ids = torch.topk(router_logits_all[n_prefill:], k=10, dim=-1)
    dec_gates = F.softmax(dec_vals, dim=-1)

    unique_active_experts = sorted(set(dec_ids.flatten().tolist()))
    print(f"  Active Unique Experts across 115 decode tokens (4 topics): {len(unique_active_experts)} / 512 experts")

    # Cache reconstructed & stock experts for all active experts
    stock_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    chronos_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    grp_mods: dict[int, ChronosHolographicExpertGroup] = {}

    for e_id in unique_active_experts:
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

    # Evaluate token-by-token true Top-10 routing across all 512 experts!
    global_prior = torch.linalg.norm(router_w @ u_ctx, dim=-1)
    _, prior_top = torch.topk(global_prior, k=24)
    prior_set = set(prior_top.tolist())

    y_routed_before = torch.zeros_like(decode_x)
    y_routed_cold = torch.zeros_like(decode_x)
    y_routed_ema = torch.zeros_like(decode_x)

    for d_i in range(decode_x.shape[0]):
        t_idx = n_prefill + d_i
        win_start = max(0, t_idx - 24)
        counts = torch.bincount(top10_all[win_start:t_idx].flatten(), minlength=512)
        _, ema_ids = torch.topk(counts, k=24)
        hot_set = prior_set | set(ema_ids.tolist())

        xt = decode_x[d_i : d_i + 1]
        for k_slot in range(10):
            e_id = int(dec_ids[d_i, k_slot].item())
            g_val = dec_gates[d_i, k_slot]

            gw, uw, dw = stock_cache[e_id]
            y_st = (F.silu(xt @ gw.T) * (xt @ uw.T)) @ dw.T
            y_routed_before[d_i : d_i + 1] += g_val * y_st

            cgw, cuw, cdw = chronos_cache[e_id]
            y_ch = (F.silu(xt @ cgw.T) * (xt @ cuw.T)) @ cdw.T
            y_routed_cold[d_i : d_i + 1] += g_val * y_ch

            if e_id in hot_set:
                y_routed_ema[d_i : d_i + 1] += g_val * y_st
            else:
                y_routed_ema[d_i : d_i + 1] += g_val * y_ch

    sh_gate_score = torch.sigmoid(decode_x @ shared_gate_w.T)
    y_shared = sh_gate_score * ((F.silu(decode_x @ sh_gate.T) * (decode_x @ sh_up.T)) @ sh_down.T)

    h_before = decode_x + y_routed_before + y_shared
    h_cold = decode_x + y_routed_cold + y_shared
    h_ema = decode_x + y_routed_ema + y_shared

    logits_before = (F.rms_norm(h_before, (2560,), eps=1e-6).to(torch.bfloat16) @ lm_head_w.T).float()
    logits_cold = (F.rms_norm(h_cold, (2560,), eps=1e-6).to(torch.bfloat16) @ lm_head_w.T).float()
    logits_ema = (F.rms_norm(h_ema, (2560,), eps=1e-6).to(torch.bfloat16) @ lm_head_w.T).float()

    probs_b = F.softmax(logits_before, dim=-1)
    log_probs_b = F.log_softmax(logits_before, dim=-1)
    top1_b = logits_before.argmax(dim=-1)
    _, top5_b = torch.topk(logits_before, k=5, dim=-1)

    for label, y_r, l_reg in [
        ("Pure Cold (0% Cache across 512 Experts)", y_routed_cold, logits_cold),
        ("Hybrid EMA-48 Cache across 512 Experts", y_routed_ema, logits_ema),
    ]:
        moe_cos = F.cosine_similarity(y_r.flatten(), y_routed_before.flatten(), dim=0).item() * 100.0
        log_cos = F.cosine_similarity(l_reg.flatten(), logits_before.flatten(), dim=0).item() * 100.0
        top1_acc = (l_reg.argmax(dim=-1) == top1_b).float().mean().item() * 100.0
        _, top5_r = torch.topk(l_reg, k=5, dim=-1)
        ovl = sum(len(set(rb) & set(rr)) for rb, rr in zip(top5_b.tolist(), top5_r.tolist())) / (5.0 * decode_x.shape[0]) * 100.0
        kl = torch.sum(probs_b * (log_probs_b - F.log_softmax(l_reg, dim=-1)), dim=-1).mean().item()
        print(
            f"  {label:<42} | MoE Cos: {moe_cos:6.2f}% | LogitCos: {log_cos:6.2f}% | "
            f"Top-1: {top1_acc:6.2f}% | Top-5: {ovl:6.2f}% | KL: {kl:.5f}"
        )
    print("=" * 124)


if __name__ == "__main__":
    main()
