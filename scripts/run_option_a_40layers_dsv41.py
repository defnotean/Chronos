import os
import time
import torch
import torch.nn.functional as F
from safetensors import safe_open

MODEL_DIR = "/home/defnotean/pb27/hf/dsv41_chronos_v75_abliterated"
OUT_PATH = "/home/defnotean/pb27/serving/pb27_vllm/dsv41_shared_experts_option_a_r64.pt"
RANK_OUT = 64
N_KEEP = int(os.environ.get("CHRONOS_OPTION_A_KEEP", "212"))
SWIGLU_LIMIT = 10.0


def ue8m0_to_float32(scale_e8m0: torch.Tensor, device: torch.device) -> torch.Tensor:
    u8 = scale_e8m0.to(device=device).view(torch.uint8).to(torch.int32)
    bits = u8 << 23
    return bits.view(torch.float32)


def dequantize_block_fp8_32(w_fp8: torch.Tensor, scale_e8m0: torch.Tensor, device: torch.device) -> torch.Tensor:
    rows, cols = w_fp8.shape
    b_r = rows // scale_e8m0.shape[0]
    b_c = cols // scale_e8m0.shape[1]
    w = w_fp8.to(device=device).view(torch.float8_e4m3fn).to(torch.float32).view(scale_e8m0.shape[0], b_r, scale_e8m0.shape[1], b_c)
    s = ue8m0_to_float32(scale_e8m0, device).view(scale_e8m0.shape[0], 1, scale_e8m0.shape[1], 1)
    return (w * s).reshape(rows, cols)


def dsv41_clamped_swiglu(gu_out: torch.Tensor, inter_dim: int = 2304, swiglu_limit: float = SWIGLU_LIMIT) -> torch.Tensor:
    gate, up = gu_out.split(inter_dim, dim=-1)
    if swiglu_limit > 0.0:
        gate = gate.clamp(max=swiglu_limit)
        up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
    return F.silu(gate) * up


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t_start = time.time()
    results: dict[int, dict[str, torch.Tensor | float]] = {}

    for l in range(40):
        tl0 = time.time()
        shard_path = f"{MODEL_DIR}/chronos-dsv41-layer-{l:05d}-of-00040.safetensors"

        with safe_open(shard_path, framework="pt", device="cpu") as f:
            keys = set(f.keys())
            tp = f"layers.{l}.ffn" if f"layers.{l}.ffn.gate.weight" in keys else f"model.layers.{l}.ffn"

            router_w = f.get_tensor(f"{tp}.gate.weight").to(device=device, dtype=torch.float32)
            bias_key = f"{tp}.gate.bias" if f"{tp}.gate.bias" in keys else f"{tp}.gate.e_score_correction_bias"
            e_score_bias = f.get_tensor(bias_key).to(device=device, dtype=torch.float32)

            fn_k = f"layers.{l}.ffn_norm.weight"
            if fn_k not in keys and f"model.{fn_k}" in keys:
                fn_k = f"model.{fn_k}"
            ffn_norm_w = f.get_tensor(fn_k).to(device=device, dtype=torch.float32) if fn_k in keys else torch.ones(5120, device=device)

            sh_w1 = dequantize_block_fp8_32(
                f.get_tensor(f"{tp}.shared_experts.w1.weight"),
                f.get_tensor(f"{tp}.shared_experts.w1.scale"),
                device,
            )
            sh_w3 = dequantize_block_fp8_32(
                f.get_tensor(f"{tp}.shared_experts.w3.weight"),
                f.get_tensor(f"{tp}.shared_experts.w3.scale"),
                device,
            )
            sh_gu = torch.cat([sh_w1, sh_w3], dim=0).contiguous()  # (4608, 5120)
            sh_w2 = dequantize_block_fp8_32(
                f.get_tensor(f"{tp}.shared_experts.w2.weight"),
                f.get_tensor(f"{tp}.shared_experts.w2.scale"),
                device,
            ).contiguous()  # (5120, 2304)
            del sh_w1, sh_w3

            # Compute exact prior_mc ranking matching ChronosDsv41SparseMoeBlock.__init__
            gen_mc = torch.Generator(device=device).manual_seed(42 + l)
            x_mc = F.normalize(
                torch.randn((2048, 5120), generator=gen_mc, device=device, dtype=torch.bfloat16), dim=-1
            ).float() * (5120.0**0.5)
            x_mc = x_mc * ffn_norm_w.unsqueeze(0)
            logits_mc = F.linear(x_mc, router_w)
            scores_mc = F.softplus(logits_mc).sqrt() + e_score_bias.unsqueeze(0)
            _, top6_mc = torch.topk(scores_mc, k=6, dim=-1)
            freq_mc = torch.bincount(top6_mc.view(-1), minlength=384).float()
            prior_mc = (freq_mc + scores_mc.mean(dim=0)).cpu().tolist()

            ranked_eids = sorted(range(384), key=lambda e: prior_mc[e], reverse=True)
            keep_l = ranked_eids[:N_KEEP]
            drop_l = ranked_eids[N_KEEP:]

            # Check hit rates for 56 / 56 / 100 under true prior_mc ranking
            top6_flat = top6_mc.view(-1)
            t1_set = set(ranked_eids[:56])
            t2_set = set(ranked_eids[56:112])
            t3_set = set(ranked_eids[112:212])
            top_list = top6_flat.tolist()
            h1 = sum(1 for e in top_list if e in t1_set) / len(top_list) * 100.0
            h2 = sum(1 for e in top_list if e in t2_set) / len(top_list) * 100.0
            h3 = sum(1 for e in top_list if e in t3_set) / len(top_list) * 100.0

            # Canonical expert input directions X384 (384, 5120) scaled by ffn_norm_w
            X384 = F.normalize(router_w, dim=-1) * (5120.0**0.5) * ffn_norm_w.unsqueeze(0)
            torch.manual_seed(42 + l)
            X_K = X384[keep_l]
            mix_idx = torch.randint(0, len(keep_l), (1024, 6), device=device)
            mix_w = torch.rand(1024, 6, 1, device=device)
            X_mix_K = F.normalize((X_K[mix_idx] * mix_w).sum(dim=1), dim=-1) * (5120.0**0.5) * ffn_norm_w.unsqueeze(0)

            X_prot = torch.cat([X_K, X_mix_K[:512]], dim=0)
            H_prot = dsv41_clamped_swiglu(X_prot @ sh_gu.T)  # (N_prot, 2304)
            H_D = dsv41_clamped_swiglu(X384[drop_l] @ sh_gu.T)  # (N_drop, 2304)

            _, _, Vh_P = torch.linalg.svd(H_prot, full_matrices=False)
            Q_P = Vh_P[:384].T  # (2304, 384)
            H_D_perp = H_D - (H_D @ Q_P) @ Q_P.T  # (N_drop, 2304)

            # Router weights on X384[drop_l]
            logits_D = X384[drop_l] @ router_w.T
            scores_D = F.softplus(logits_D).sqrt()
            _, top6_D = torch.topk(scores_D + e_score_bias.unsqueeze(0), k=6, dim=-1)
            w_D = torch.gather(scores_D, 1, top6_D)
            w_D = (w_D / (w_D.sum(dim=-1, keepdim=True) + 1e-20)) * 1.5

            # Group dropped experts by Chronos group g = e_idx // 8
            drop_by_g: dict[int, list[tuple[int, int]]] = {}
            for idx_pos, e_idx in enumerate(drop_l):
                drop_by_g.setdefault(e_idx // 8, []).append((idx_pos, e_idx))

            Y_D_target = torch.zeros((len(drop_l), 5120), device=device, dtype=torch.float32)
            for g, items in drop_by_g.items():
                gp = f"{tp}.chronos_groups.{g}."
                u_gu_g = f.get_tensor(f"{gp}healer.u_gu").to(device=device).view(torch.float8_e4m3fn).float()
                v_gu_g = f.get_tensor(f"{gp}healer.v_gu").to(device=device).view(torch.float8_e4m3fn).float()
                u_dn_g = f.get_tensor(f"{gp}healer.u_down").to(device=device).view(torch.float8_e4m3fn).float()
                v_dn_g = f.get_tensor(f"{gp}healer.v_down").to(device=device).view(torch.float8_e4m3fn).float()
                for idx_pos, e_idx in items:
                    m = e_idx % 8
                    x_e = X384[e_idx : e_idx + 1]
                    z_e = (8.0 / 7.0) * ((x_e @ v_gu_g[m].T) @ u_gu_g[m].T)
                    h_e = dsv41_clamped_swiglu(z_e)
                    y_e = (8.0 / 7.0) * ((h_e @ v_dn_g[m].T) @ u_dn_g[m].T)
                    match = (top6_D[idx_pos] == e_idx).nonzero(as_tuple=True)[0]
                    gate_w = w_D[idx_pos, match[0]].item() if len(match) > 0 else 0.25
                    Y_D_target[idx_pos] = gate_w * y_e.squeeze(0)
                del u_gu_g, v_gu_g, u_dn_g, v_dn_g

            if f"{tp}.abliteration.r_basis" in keys:
                rb = f.get_tensor(f"{tp}.abliteration.r_basis").to(device=device, dtype=torch.float32)
                Y_D_target = Y_D_target - (Y_D_target @ rb.T) @ rb

            lam = 5e-2 * (H_D_perp.norm() ** 2 / float(max(1, len(drop_l))))
            A = H_D_perp @ H_D_perp.T + lam * torch.eye(len(drop_l), device=device)
            dW_sh = (H_D_perp.T @ torch.linalg.solve(A, Y_D_target)).T  # (5120, 2304)

            U_d, S_d, Vh_d = torch.linalg.svd(dW_sh, full_matrices=False)
            U_r = (U_d[:, :RANK_OUT] * S_d[:RANK_OUT].unsqueeze(0)).to(torch.bfloat16).cpu().contiguous()
            V_r = Vh_d[:RANK_OUT, :].to(torch.bfloat16).cpu().contiguous()

            dW_r64 = U_r.to(device=device, dtype=torch.float32) @ V_r.to(device=device, dtype=torch.float32)
            W_new = sh_w2 + 0.05 * dW_r64
            cos_K = F.cosine_similarity(H_prot[:96] @ sh_w2.T, H_prot[:96] @ W_new.T, dim=-1).mean().item()
            fit_D = F.cosine_similarity(H_D @ dW_r64.T, Y_D_target, dim=-1).mean().item()
            dw_norm = dW_r64.norm().item()

            results[l] = {
                "U": U_r,
                "V": V_r,
                "cos_K": cos_K,
                "fit_D": fit_D,
                "dw_norm": dw_norm,
            }
            print(
                f"[{l+1:02d}/40] L{l:02d} ({time.time()-tl0:.2f}s) | hits(56/56/100): T1={h1:.1f}% T2={h2:.1f}% T3={h3:.1f}% Miss={100-h1-h2-h3:.1f}% | dW_norm={dw_norm:.3f} | cos_K={cos_K:.6f} | fit_D={fit_D:.4f}",
                flush=True,
            )

    torch.save(results, OUT_PATH)
    print(f"ALL 40 LAYERS COMPLETE in {time.time()-t_start:.2f}s -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
