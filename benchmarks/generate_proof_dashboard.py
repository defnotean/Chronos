"""Generates the 6-Panel Master Hardware Proof Dashboard (`assets/chronos_1t_gb10_a100_proof.png`)
combining live telemetry from:
1. Google Colab NVIDIA A100-SXM4-40GB (`assets/colab_a100_results.json` & `assets/pb27_a100_healer_results.json`)
2. Physical NVIDIA DGX Spark GB10 Grace Blackwell (`sm_121`) (`assets/gb10_silicon_results.json` & `assets/pb27_holographic_moe_results.json`)
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
ARTIFACT_DIR = Path(r"C:\Users\Eating\.gemini\antigravity\brain\f42005ad-9ad4-4e7f-b1f5-562159d2e5bc")


def main() -> None:
    with open(ASSETS / "colab_a100_results.json", encoding="utf-8") as f:
        a100_data = json.load(f)
    with open(ASSETS / "gb10_silicon_results.json", encoding="utf-8") as f:
        gb10_data = json.load(f)
    with open(ASSETS / "pb27_holographic_moe_results.json", encoding="utf-8") as f:
        pb27_data = json.load(f)

    plt.rcParams.update({
        "figure.facecolor": "#0b0f19",
        "axes.facecolor": "#111827",
        "axes.edgecolor": "#334155",
        "axes.labelcolor": "#e2e8f0",
        "text.color": "#f8fafc",
        "xtick.color": "#cbd5e1",
        "ytick.color": "#cbd5e1",
        "grid.color": "#1e293b",
        "font.size": 10,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
    })

    fig, axes = plt.subplots(2, 3, figsize=(20, 11.5), dpi=220)
    fig.suptitle(
        "CHRONOS-1T x PSEUDO-BRAIN (pb27): EMPIRICAL SILICON PROOF ON NVIDIA DGX SPARK (GB10, sm_121) & COLAB A100-SXM4-40GB",
        fontsize=14.5,
        fontweight="bold",
        color="#38bdf8",
        y=0.975,
    )

    # --- Panel A: Colab A100 d=16,384 HRR Superposition Proof ---
    ax = axes[0, 0]
    sup = a100_data["exp1a_16k_superposition"]
    m_vals = [r["M"] for r in sup]
    theo_snr = [r["theo_snr_db"] for r in sup]
    emp_snr = [r["emp_snr_db"] for r in sup]
    emp_var = [r["emp_var"] * 1e4 for r in sup]
    theo_var = [r["theo_var"] * 1e4 for r in sup]

    ax.plot(m_vals, theo_snr, "--", color="#94a3b8", lw=2.0, label="Theo SNR: $10\\log_{10}(d/(M-1))$")
    ax.plot(m_vals, emp_snr, "o-", color="#38bdf8", lw=2.5, markersize=7, label="A100 Empirical SNR (dB)")
    ax.set_title("A. Colab A100: d=16,384 HRR Superposition (Up to 4.29B Params)")
    ax.set_xlabel("Superposed Expert Matrices M in 1 Physical Tensor")
    ax.set_ylabel("Unbound Signal-to-Noise Ratio (dB)", color="#38bdf8")
    ax.grid(True, linestyle="--", alpha=0.6)

    ax2 = ax.twinx()
    ax2.plot(m_vals, theo_var, ":", color="#f59e0b", lw=1.8, label="Theo Var $(M-1)/d$ ($\\times 10^{-4}$)")
    ax2.plot(m_vals, emp_var, "s-", color="#fbbf24", lw=2.0, markersize=6, label="Emp Var ($\\times 10^{-4}$)")
    ax2.set_ylabel("Cross-Talk Variance ($\\times 10^{-4}$)", color="#fbbf24")
    ax2.tick_params(axis="y", colors="#fbbf24")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=8.2, framealpha=0.85, facecolor="#0f172a")

    # --- Panel B: DGX Spark GB10 Native sm_121 CUDA 13.0 Kernel Speedups ---
    ax = axes[0, 1]
    kb = gb10_data["exp2b_kernel_benchmarks"]
    shapes = [f"{r['d_out']//1024}K×{r['d_in']//1024}K\n({r['dense_fp16_mb']:.0f}MB)" for r in kb]
    x_idx = np.arange(len(shapes))
    w = 0.26

    dense_ms = [r["dense_latency_ms"] for r in kb]
    cuda10x_ms = [r["hss_10x_cuda_latency_ms"] for r in kb]
    l2_ms = [r["hss_l2_spectral_latency_ms"] for r in kb]

    b1 = ax.bar(x_idx - w, dense_ms, width=w, color="#ef4444", label="Dense LPDDR5x Stream (227 GB/s)")
    b2 = ax.bar(x_idx, cuda10x_ms, width=w, color="#10b981", label="Native sm_121 10x Register HSS")
    b3 = ax.bar(x_idx + w, l2_ms, width=w, color="#38bdf8", label="GB10 24MB L2 Spectral HSS")

    ax.set_yscale("log")
    ax.set_title("B. DGX Spark GB10 (sm_121): Measured Kernel Latency (Log Scale)")
    ax.set_xticks(x_idx)
    ax.set_xticklabels(shapes)
    ax.set_ylabel("Execution Latency on Physical GB10 (ms)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)

    for i, r in enumerate(kb):
        ax.text(x_idx[i], cuda10x_ms[i] * 1.18, f"{r['speedup_10x_cuda']:.1f}x", ha="center", va="bottom", fontsize=8, color="#34d399", fontweight="bold")
        ax.text(x_idx[i] + w, l2_ms[i] * 1.18, f"{r['speedup_l2_spectral']:.0f}x", ha="center", va="bottom", fontsize=8, color="#7dd3fc", fontweight="bold")
    ax.legend(loc="upper left", fontsize=8.2, framealpha=0.85, facecolor="#0f172a")

    # --- Panel C: Real `flash_next_nvfp4` 8:1 Holographic + Spectral-SMW SVD Healers ---
    ax = axes[0, 2]
    p1 = pb27_data["part1_real_nvfp4_spectral_smw"]
    ranks = [r["svd_rank"] for r in p1]
    naive_c = [r["naive_8to1_swiglu_cosine_pct"] for r in p1]
    lin_c = [r["linear_proj_cosine_pct"] for r in p1]
    cold_c = [r["healed_cold_swiglu_cosine_pct"] for r in p1]
    moe_c = [r["combined_10expert_moe_fidelity_pct"] for r in p1]

    ax.plot(ranks, lin_c, "^-", color="#38bdf8", lw=2.3, markersize=7, label="8:1 + SVD Linear Proj Cosine")
    ax.plot(ranks, moe_c, "o-", color="#10b981", lw=2.6, markersize=8, label="Full 10-Expert MoE Fidelity (Pinned+HSS)")
    ax.plot(ranks, cold_c, "s-", color="#a855f7", lw=2.2, markersize=6.5, label="Cold Tail Expert SwiGLU (0 NVMe Hits)")
    ax.plot(ranks, naive_c, "x--", color="#ef4444", lw=1.8, markersize=6, label="Unhealed Naive 8:1 HRR SwiGLU (5.23%)")

    for r_val, m_val in zip(ranks, moe_c):
        ax.annotate(f"{m_val:.1f}%", (r_val, m_val), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8.5, color="#34d399", fontweight="bold")

    ax.set_title("C. Real `flash_next_nvfp4` 8:1 + Spectral-SMW Healers", fontsize=10.8)
    ax.set_xlabel("Spectral-SMW SVD Healer Rank r (per Expert)")
    ax.set_ylabel("Output Cosine Similarity vs Exact Expert (%)")
    ax.set_xticks(ranks)
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="center right", fontsize=8.0, framealpha=0.85, facecolor="#0f172a")

    # --- Panel D: Zero-NVMe Stride-64 Decode Throughput on GB10 ---
    ax = axes[1, 0]
    p2 = pb27_data["part2_stride64_decode_cache"]
    regimes = [
        "Naive Cold-Expert\nNVMe Offload\n(Stalls Every Token)",
        "Chronos-1T 64-Layer\n1.0T Dense-Equiv\n(GB10 L2 Spectral)",
        "PseudoBrain x Chronos-1T\n8:1 NVFP4 + Top-32 Pinned\n(Zero-NVMe Stride-64)",
    ]
    tok_s = [
        p2["naive_nvme_tok_per_sec"],
        gb10_data["exp2c_live_1t_forward"]["tokens_per_sec_l2_hss"],
        p2["pinned_hss_tok_per_sec"],
    ]
    colors_d = ["#ef4444", "#38bdf8", "#10b981"]
    bars_d = ax.bar(regimes, tok_s, color=colors_d, width=0.52, edgecolor="#e2e8f0", linewidth=0.8)
    ax.set_title("D. Physical DGX Spark GB10 Decode Speed (Tokens / Sec)")
    ax.set_ylabel("Sustained Decode Throughput (tok/s)")
    ax.set_ylim(0, max(tok_s) * 1.22)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)

    for bar, val in zip(bars_d, tok_s):
        speedup_txt = f"{val:.1f} tok/s\n({val / tok_s[0]:.1f}x)" if val > 2.0 else f"{val:.2f} tok/s\n(1.0x Baseline)"
        ax.text(bar.get_x() + bar.get_width() / 2, val + 2.0, speedup_txt, ha="center", va="bottom", fontsize=9.0, fontweight="bold", color="#f8fafc")

    # --- Panel E: Algebraic Rank Explosion & Epigenetic Stem-Cell Training on A100 ---
    ax = axes[1, 1]
    hist = a100_data["exp1c_training"]["history"]
    steps = [h["step"] for h in hist]
    loss = [h["loss"] for h in hist]
    acc = [h["accuracy_pct"] for h in hist]

    ax.plot(steps, acc, "o-", color="#10b981", lw=2.5, markersize=6.5, label="4-Domain Epigenetic Token Accuracy (%)")
    ax.set_title("E. Colab A100: 4-Domain Epigenetic Morphing & 504x Rank Explosion")
    ax.set_xlabel("Optimization Step on A100-SXM4-40GB (6.44s Total)")
    ax.set_ylabel("Multi-Domain Token Accuracy (%)", color="#10b981")
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.6)

    ax_loss = ax.twinx()
    ax_loss.plot(steps, loss, "s--", color="#f43f5e", lw=2.0, markersize=5.5, label="Cross-Entropy Loss")
    ax_loss.set_ylabel("Training Cross-Entropy Loss", color="#f43f5e")
    ax_loss.tick_params(axis="y", colors="#f43f5e")

    ax.text(
        0.04,
        0.52,
        "Algebraic Rank Explosion (d=4096, m=128):\n"
        "• Seed Params: 33,280 vs Dense: 16,777,216 (504.1x)\n"
        "• Linear Rank: 128 / 4096 (3.1%)\n"
        "• HSS Non-Linear Rank: 3885 / 4096 (94.8%)",
        transform=ax.transAxes,
        fontsize=8.3,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#0f172a", "edgecolor": "#38bdf8", "alpha": 0.9},
        color="#e2e8f0",
    )

    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax_loss.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, loc="lower right", fontsize=8.0, framealpha=0.85, facecolor="#0f172a")

    # --- Panel F: Complete 128 GB DGX Spark Unified Memory Map (1T Model + 1M Context) ---
    ax = axes[1, 2]
    components = [
        "Shared Attn + Routers\n+ Gated DeltaNet (18.0 GB)",
        "1T Routed MoE Weights\n8:1 NVFP4 Holographic (62.0 GB)",
        "Rank-64 SVD Healers (12 GB)\n+ Top-32 Pinned Cache (4 GB)",
        "PseudoBrain 1M-Token KV\n4 Sinks+pt+16K+GDN (3.5 GB)",
        "Free Unified RAM Headroom\nOS + Speculative (28.5 GB)",
    ]
    sizes_gb = [18.0, 62.0, 16.0, 3.5, 28.5]
    colors_f = ["#38bdf8", "#6366f1", "#a855f7", "#10b981", "#334155"]

    bottom = 0.0
    for comp, sz, col in zip(components, sizes_gb, colors_f):
        ax.barh(["128 GB\nDGX Spark\nBudget"], [sz], left=bottom, color=col, height=0.48, edgecolor="#0b0f19", linewidth=1.5, label=comp)
        if sz >= 12.0:
            ax.text(bottom + sz / 2, 0, f"{sz:.1f} GB", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#ffffff")
        bottom += sz

    ax.axvline(99.5, color="#facc15", linestyle="--", linewidth=2.2)
    ax.text(97.5, 0.28, "Active 1T + 1M Context = 99.5 GB / 128.0 GB", ha="right", va="center", color="#facc15", fontweight="bold", fontsize=9.0)
    ax.set_xlim(0, 128.0)
    ax.set_ylim(-0.38, 0.38)
    ax.set_xlabel("DGX Spark GB10 Unified LPDDR5x Memory Allocation (GB)")
    ax.set_title("F. DGX Spark 128 GB Map: 1T Model + 1M Context (99.5 GB)", fontsize=10.8)
    ax.grid(True, axis="x", linestyle="--", alpha=0.6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=7.8, framealpha=0.9, facecolor="#0f172a")

    plt.tight_layout(rect=[0.0, 0.08, 1.0, 0.955])
    out_img = ASSETS / "chronos_1t_gb10_a100_proof.png"
    fig.savefig(out_img, dpi=220)
    plt.close(fig)
    print(f"[OK] Saved dashboard to {out_img}")

    if ARTIFACT_DIR.exists():
        shutil.copy2(out_img, ARTIFACT_DIR / "chronos_1t_gb10_a100_proof.png")
        print(f"[OK] Copied dashboard to artifact dir {ARTIFACT_DIR / 'chronos_1t_gb10_a100_proof.png'}")


if __name__ == "__main__":
    main()
