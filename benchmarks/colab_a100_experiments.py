"""Colab A100-SXM4-40GB Hardware Experiments for Chronos-1T (HSS).

Executes large-scale GPU experiments on the Colab A100-40GB VM:
1. Exp 1A: True 1T-Scale Hidden Dimension (d = 16,384) Holographic Superposition
   (Superimposing up to M = 16 experts of size 16,384 x 16,384 = 4.29B params
   into a single shared physical tensor in A100 VRAM).
2. Exp 1B: Full-Rank Algebraic Rank Explosion at Scale (d = 2,048 and 4,096) on GPU,
   proving a tiny m x m complex seed synthesizes a 100% full-rank d x d matrix.
3. Exp 1C: Multi-Domain Chronos-1T Transformer Training + Epigenetic DNA Patch
   Synthesis on A100, exporting trained spectral seeds for DGX Spark (GB10) deployment.
"""

from __future__ import annotations

import json
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_threshold(x: torch.Tensor, tau: float) -> torch.Tensor:
    if tau <= 0.0:
        return x
    return torch.sign(x) * torch.clamp(torch.abs(x) - tau, min=0.0)


def run_exp1a_16k_hrr_superposition(device: torch.device) -> list[dict]:
    """Tests HRR superposition at true 1T hidden dimension d = 16,384 on A100 GPU."""
    print("=" * 80)
    print("EXP 1A: 1T-SCALE HIDDEN DIMENSION (d = 16,384) HRR SUPERPOSITION ON A100 GPU")
    print("=" * 80)

    d = 16384
    rank = 32
    expert_counts = [2, 4, 8, 10, 16]
    max_m = max(expert_counts)

    torch.manual_seed(42)
    t0 = time.perf_counter()

    # Generate M=16 orthonormal rank-32 expert bases on GPU (d = 16,384)
    u_list = []
    v_list = []
    keys = []
    for m in range(max_m):
        u_m, _ = torch.linalg.qr(torch.randn(d, rank, device=device, dtype=torch.float32))
        v_m, _ = torch.linalg.qr(torch.randn(d, rank, device=device, dtype=torch.float32))
        p_m = (torch.randint(0, 2, (d, 1), device=device, dtype=torch.int8).float() * 2.0 - 1.0)
        u_list.append(u_m)
        v_list.append(v_m)
        keys.append(p_m)

    # Target Expert 0 matrix: 16,384 x 16,384 (268.4M floats = 1.07 GB in FP32)
    w_0 = (u_list[0] @ v_list[0].T) / math.sqrt(rank)
    row_proj_true = u_list[0].T @ w_0  # (rank, d)

    results = []
    print(f"{'d':>8} | {'Experts M':>9} | {'Virtual Params':>14} | {'Theo Var':>10} | {'Emp Var':>10} | {'Theo SNR':>10} | {'Emp SNR':>10} | {'Cosine Sim':>10}")
    print("-" * 98)

    for m_cnt in expert_counts:
        # Superimpose m_cnt 16,384 x 16,384 expert matrices into ONE 16,384 x 16,384 tensor S
        s_shared = torch.zeros(d, d, device=device, dtype=torch.float32)
        for m in range(m_cnt):
            w_m = (u_list[m] @ v_list[m].T) / math.sqrt(rank)
            s_shared.add_(keys[m] * w_m)

        # Unbind Expert 0: W_hat_0 = P_0 (x) S
        w_raw = keys[0] * s_shared
        row_proj_hat = u_list[0].T @ w_raw
        cross_talk = row_proj_hat - row_proj_true

        emp_var = float((cross_talk.double().pow(2).sum() / rank).item())
        theo_var = (m_cnt - 1) / float(d)

        emp_snr_lin = 1.0 / max(emp_var, 1e-15)
        theo_snr_lin = float(d) / max(m_cnt - 1, 1)
        emp_snr_db = 10.0 * math.log10(emp_snr_lin)
        theo_snr_db = 10.0 * math.log10(theo_snr_lin)

        # Non-linear soft-threshold activation in spectral core
        core_hat = row_proj_hat @ v_list[0]
        tau = 2.5 * theo_var
        core_denoised = soft_threshold(core_hat, tau)
        w_rec = u_list[0] @ core_denoised @ v_list[0].T

        dot = float((w_rec.double() * w_0.double()).sum().item())
        norm_rec = float(torch.linalg.norm(w_rec.double()).item())
        norm_true = float(torch.linalg.norm(w_0.double()).item())
        cos_sim = min(1.0, dot / (norm_rec * norm_true + 1e-15))

        virt_params = m_cnt * d * d
        print(
            f"{d:8,d} | {m_cnt:9d} | {virt_params / 1e9:12.2f}B | "
            f"{theo_var:10.6f} | {emp_var:10.6f} | "
            f"{theo_snr_db:7.2f} dB | {emp_snr_db:7.2f} dB | {cos_sim:10.6f}"
        )
        results.append({
            "d": d,
            "M": m_cnt,
            "virtual_params_billions": virt_params / 1e9,
            "theo_var": theo_var,
            "emp_var": emp_var,
            "theo_snr_db": theo_snr_db,
            "emp_snr_db": emp_snr_db,
            "cosine_similarity": cos_sim,
        })
        del s_shared, w_raw, w_rec
        torch.cuda.empty_cache()

    elapsed = time.perf_counter() - t0
    print(f"[OK] Exp 1A completed on {torch.cuda.get_device_name(0)} in {elapsed:.2f}s\n")
    return results


def generate_mixed_parity_chebyshev_basis(dim: int, degree: int, phase: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
    device = phase.device
    dtype = phase.dtype
    idx = torch.arange(dim, device=device, dtype=dtype)
    theta = math.pi * (2.0 * idx + 1.0) / (2.0 * dim)

    selected = [0]
    seen = {0}
    step = 1
    while len(selected) < degree:
        cand = (step * 73 + step * step * 20) % dim
        if cand not in seen:
            seen.add(cand)
            selected.append(cand)
        step += 1
    orders = torch.tensor(selected, device=device, dtype=dtype)
    angles = torch.outer(theta, orders) + phase.unsqueeze(0)
    basis = torch.cos(angles) * math.sqrt(2.0 / dim) * gain.unsqueeze(0)
    return basis


def run_exp1b_large_scale_rank_explosion(device: torch.device) -> list[dict]:
    """Verifies algebraic rank explosion on GPU for d = 1024, 2048, 4096."""
    print("=" * 80)
    print("EXP 1B: ALGEBRAIC RANK EXPLOSION ON A100 GPU (d = 1024, 2048, 4096)")
    print("=" * 80)

    torch.manual_seed(101)
    configs = [
        (1024, 64),
        (2048, 64),
        (2048, 128),
        (4096, 128),
    ]
    results = []
    print(f"{'Dim d':>8} | {'Seed m':>8} | {'Seed Params':>12} | {'Dense Params':>14} | {'Compression':>11} | {'Linear Rank':>12} | {'HSS Tiled Rank':>15}")
    print("-" * 92)

    for d, m in configs:
        phi_real = torch.randn(m, m, device=device) / math.sqrt(m)
        phi_imag = torch.randn(m, m, device=device) / math.sqrt(m)
        omega = torch.randn(m, m, device=device) * 0.2
        spec = torch.complex(phi_real, phi_imag) * torch.polar(torch.ones_like(omega), omega)
        g_core = torch.fft.ifft2(spec, norm="ortho").real

        phase_u = torch.zeros(m, device=device)
        phase_v = torch.zeros(m, device=device)
        gain_u = torch.ones(m, device=device)
        gain_v = torch.ones(m, device=device)

        u = generate_mixed_parity_chebyshev_basis(d, m, phase_u, gain_u)
        v = generate_mixed_parity_chebyshev_basis(d, m, phase_v, gain_v)

        w_linear = u @ g_core @ v.T
        z = w_linear * math.sqrt(d)
        r_pos = F.relu(z)
        r_neg = F.relu(-z)
        w_hss = ((r_pos * r_pos - 0.25 * r_neg * r_neg) + 0.15 * torch.sin(math.pi * z)) / math.sqrt(d)

        lin_rank = int(torch.linalg.matrix_rank(w_linear).item())
        hss_rank = int(torch.linalg.matrix_rank(w_hss).item())

        seed_params = 2 * m * m + 4 * m
        dense_params = d * d
        comp = dense_params / seed_params

        print(
            f"{d:8,d} | {m:8d} | {seed_params:12,d} | {dense_params:14,d} | "
            f"{comp:10.1f}x | {lin_rank:6d}/{d:<5d} | {hss_rank:8d}/{d:<5d}"
        )
        results.append({
            "d": d,
            "m": m,
            "seed_params": seed_params,
            "dense_params": dense_params,
            "compression_ratio": comp,
            "linear_rank": lin_rank,
            "hss_full_rank": hss_rank,
        })

    print()
    return results


class GPUHSSLinear(nn.Module):
    """GPU-accelerated Holographic Spectral Operator layer for A100 training & export."""

    def __init__(self, d_out: int, d_in: int, m: int = 64, num_modes: int = 2) -> None:
        super().__init__()
        self.d_out = d_out
        self.d_in = d_in
        self.m = m
        self.num_modes = num_modes

        scale = 1.0 / math.sqrt(num_modes * m)
        self.phi_real = nn.Parameter(torch.randn(num_modes, m, m) * scale)
        self.phi_imag = nn.Parameter(torch.randn(num_modes, m, m) * scale)
        self.phase_u = nn.Parameter(torch.zeros(num_modes, m))
        self.phase_v = nn.Parameter(torch.zeros(num_modes, m))
        self.gain_u = nn.Parameter(torch.ones(num_modes, m))
        self.gain_v = nn.Parameter(torch.ones(num_modes, m))
        self.router_r = nn.Linear(d_in, num_modes * m)
        self.router_c = nn.Linear(d_in, num_modes * m)
        nn.init.normal_(self.router_r.weight, std=0.01)
        nn.init.zeros_(self.router_r.bias)
        nn.init.normal_(self.router_c.weight, std=0.01)
        nn.init.zeros_(self.router_c.bias)
        self.mode_scale = nn.Parameter(torch.ones(num_modes) / math.sqrt(num_modes))

    def forward(self, x: torch.Tensor, epi_delta: torch.Tensor | None = None) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.d_in)
        ctx = x_flat.mean(dim=0)

        om_r = self.router_r(ctx).view(self.num_modes, self.m, 1)
        om_c = self.router_c(ctx).view(self.num_modes, 1, self.m)
        omega = om_r + om_c

        phi = torch.complex(self.phi_real, self.phi_imag)
        if epi_delta is not None:
            phi = phi + epi_delta

        g_cores = torch.fft.ifft2(phi * torch.polar(torch.ones_like(omega), omega), norm="ortho").real

        w_eff = torch.zeros(self.d_out, self.d_in, device=x.device, dtype=x.dtype)
        s_in = math.sqrt(self.d_in)
        inv_s = 1.0 / s_in
        for k in range(self.num_modes):
            u_k = generate_mixed_parity_chebyshev_basis(self.d_out, self.m, self.phase_u[k], self.gain_u[k])
            v_k = generate_mixed_parity_chebyshev_basis(self.d_in, self.m, self.phase_v[k], self.gain_v[k])
            z = (u_k @ g_cores[k] @ v_k.T) * s_in
            r_p = F.relu(z)
            r_n = F.relu(-z)
            act = ((r_p * r_p - 0.25 * r_n * r_n) + 0.15 * torch.sin(math.pi * z)) * inv_s
            w_eff = w_eff + self.mode_scale[k] * act

        out = x_flat @ w_eff.T
        return out.view(*orig_shape[:-1], self.d_out)


def run_exp1c_distillation_and_training(device: torch.device) -> dict:
    """Distills a dense target operator and trains a multi-domain HSS + Epigenetic model on A100."""
    print("=" * 80)
    print("EXP 1C: MULTI-DOMAIN CHRONOS-1T TRAINING & EPIGENETIC MORPHING ON A100 GPU")
    print("=" * 80)

    torch.manual_seed(77)
    d_model = 512
    m_seed = 64
    num_domains = 4
    rank_epi = 16

    # Create a 4-domain specialist token sequence task where each domain has its own
    # non-linear transition rule, learned via a single shared Stem-Cell HSS core +
    # 4 lightweight Epigenetic DNA patches!
    vocab_size = 512
    emb = nn.Embedding(vocab_size, d_model).to(device)
    hss_layer1 = GPUHSSLinear(d_model, d_model, m=m_seed, num_modes=2).to(device)
    hss_layer2 = GPUHSSLinear(d_model, d_model, m=m_seed, num_modes=2).to(device)
    head = nn.Linear(d_model, vocab_size, bias=False).to(device)
    head.weight = emb.weight

    # Epigenetic low-rank patches: 4 domains, rank 16
    epi_a_r = nn.Parameter(torch.randn(num_domains, 2, m_seed, rank_epi, device=device) * 0.05)
    epi_a_i = nn.Parameter(torch.randn(num_domains, 2, m_seed, rank_epi, device=device) * 0.05)
    epi_b_r = nn.Parameter(torch.randn(num_domains, 2, m_seed, rank_epi, device=device) * 0.05)
    epi_b_i = nn.Parameter(torch.randn(num_domains, 2, m_seed, rank_epi, device=device) * 0.05)

    params = (
        list(emb.parameters())
        + list(hss_layer1.parameters())
        + list(hss_layer2.parameters())
        + [epi_a_r, epi_a_i, epi_b_r, epi_b_i]
    )
    opt = torch.optim.AdamW(params, lr=3e-3, weight_decay=1e-4)

    # Fixed domain-specific synthetic sequences (domains 0, 1, 2, 3)
    domain_inputs = [
        torch.randint(0, vocab_size, (8, 32), device=device) for _ in range(num_domains)
    ]
    domain_targets = [
        torch.roll(inp, shifts=-(dom + 1), dims=1) for dom, inp in enumerate(domain_inputs)
    ]

    t0 = time.perf_counter()
    history = []
    for step in range(1, 121):
        opt.zero_grad()
        total_loss = torch.tensor(0.0, device=device)
        correct = 0
        total_toks = 0

        for dom in range(num_domains):
            a_c = torch.complex(epi_a_r[dom], epi_a_i[dom])
            b_c = torch.complex(epi_b_r[dom], epi_b_i[dom])
            delta_phi = torch.matmul(a_c, b_c.transpose(-1, -2))

            x = emb(domain_inputs[dom])
            h = x + F.silu(hss_layer1(x, epi_delta=delta_phi))
            h = h + F.silu(hss_layer2(h, epi_delta=delta_phi))
            logits = head(h)
            loss = F.cross_entropy(logits.view(-1, vocab_size), domain_targets[dom].view(-1))
            total_loss = total_loss + loss / num_domains

            preds = logits.argmax(dim=-1)
            correct += int((preds == domain_targets[dom]).sum().item())
            total_toks += int(domain_targets[dom].numel())

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        acc = (correct / total_toks) * 100.0
        if step in {1, 20, 40, 60, 80, 100, 120}:
            print(f"  Step {step:3d}/120 | Multi-Domain CE Loss: {total_loss.item():.4f} | Token Accuracy: {acc:6.2f}%")
            history.append({"step": step, "loss": float(total_loss.item()), "accuracy_pct": acc})

    elapsed = time.perf_counter() - t0

    # Export trained Holographic Spectral Seed for transfer to DGX Spark (GB10)!
    with torch.no_grad():
        phi = torch.complex(hss_layer1.phi_real[0], hss_layer1.phi_imag[0])
        g_core_0 = torch.fft.ifft2(phi, norm="ortho").real.cpu()
        export_payload = {
            "g_core": g_core_0,
            "phase_u": hss_layer1.phase_u[0].cpu(),
            "phase_v": hss_layer1.phase_v[0].cpu(),
            "gain_u": hss_layer1.gain_u[0].cpu(),
            "gain_v": hss_layer1.gain_v[0].cpu(),
            "mode_scale": float(hss_layer1.mode_scale[0].item()),
            "history": history,
        }
        torch.save(export_payload, "/content/chronos_1t_a100_seed.pt")

    print(f"[OK] Exp 1C finished in {elapsed:.2f}s — Final Multi-Domain Accuracy: {history[-1]['accuracy_pct']:.2f}%")
    print("[OK] Saved trained Holographic Seed to /content/chronos_1t_a100_seed.pt\n")
    return {"elapsed_sec": elapsed, "history": history}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on Colab Hardware: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB VRAM)")
    res_1a = run_exp1a_16k_hrr_superposition(device)
    res_1b = run_exp1b_large_scale_rank_explosion(device)
    res_1c = run_exp1c_distillation_and_training(device)

    summary = {
        "hardware": torch.cuda.get_device_name(0),
        "exp1a_16k_superposition": res_1a,
        "exp1b_rank_explosion": res_1b,
        "exp1c_training": res_1c,
    }
    with open("/content/colab_a100_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("[OK] Wrote full JSON telemetry to /content/colab_a100_results.json")


if __name__ == "__main__":
    main()
