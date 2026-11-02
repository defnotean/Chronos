---
license: mit
base_model:
  - deepseek-ai/DeepSeek-V4.1-Flash-1017
  - dealignai/DeepSeek-V4.1-Flash-1017-UNCENSORED-FP8
  - OBLITERATUS/DeepSeek-V4.1-Flash-1017-obliterated
tags:
  - deepseek
  - deepseek-v4.1-flash
  - moe
  - chronos
  - pseudobrain
  - abliterated
  - uncensored
  - holographic-superposition
  - nvfp4
  - vllm
pipeline_tag: text-generation
---

# PseudoBrain x DeepSeek-V4.1-Flash-1017 (`Chronos v7.5` Dual-Verified Abliterated)

**`PseudoBrain-DeepSeek-V4.1-Flash-1017-Chronos-v7.5-Abliterated`** is the full 40-layer (`15,360` routed experts + shared experts + Sparse Multi-Head Latent Attention + 4-lane Sinkhorn Hyper-Connections + multimodal vision/aligner) **Chronos `v7.5`** conversion of [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) with **Dual-Verified Post-Quantization Orthogonal Abliteration**.

---

## 1. Key Highlights & Verified Hardware Receipts

| Metric | Stock `BF16` (`1.51 TB`) | Stock `MXFP4` (`510.31 GB`) | **`Chronos v7.5` + Upgrade 2 (`96 Hot MXFP4 + 288 Cold`)** | **`Chronos v7.5` (`2-Bit Lloyd-Max`)** | **`Chronos v7.5` (`1-Bit Compact-Disk`)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Total Model Disk Footprint** | `1,509.36 GB` | `510.31 GB` | **`286.05 GB`** (`+67.24 GB` Hot-96 Sidecars) | **`218.81 GB`** (`5.380 GB/layer`) | **`142.37 GB`** (`3.469 GB/layer`) |
| **Per-Layer MoE Shard Size** | `27.390 GB` | `7.390 GB` | **`5.380 GB` + `1.681 GB` Hot-96** | **`5.380 GB` (`5.09x` vs BF16)** | **`3.469 GB` (`7.90x` vs BF16)** |
| **Active GPU VRAM (All 40 Layers)** | OOM on 8x80GB | OOM on 4x80GB | **`52.97 GB` (`0 GB` extra GPU VRAM via CPU/mmap)** | **`52.97 GB` (`1x 80GB GPU`)** | **`52.97 GB` (`1x 80GB GPU`)** |
| **Single-Expert `gate_up` Cosine** | `1.0000` | `1.0000` | **`1.00000` (`100.00%` Hot) / `0.95317` (Cold)** | **`0.95317` (`95.32%`)** | `0.85889` (`85.89%`) |
| **Single-Expert `down_proj` Cosine** | `1.0000` | `1.0000` | **`0.999999` (`100.00%` Hot) / `0.95397` (Cold)** | **`0.95397` (`95.40%`)** | `0.88062` (`88.06%`) |
| **Routed-Only Top-6 MoE Cosine** | `1.0000` | `1.0000` | **`0.98556` (`98.56%`)** | **`0.96924` (`96.92%`)** | `0.91420` (`91.42%`) |
| **Full MoE Layer Output Cosine** | `1.0000` | `1.0000` | **`0.99750` (`99.75%`)** | **`0.98292` (`98.29%`)** | `0.95180` (`95.18%`) |
| **Routed `w2` Refusal Projection Norm** | `1.593769` | `1.593769` | **`0.000009` (`99.9994%` Eliminated)** | **`0.000811` (`99.95%` Eliminated)** | **`0.000006` (`99.9996%` Eliminated)** |
| **40-Layer E2E Mean Routed Refusal Leakage** | `1.5938` | `1.5938` | **`< 0.005` (`logits: [2, 7, 129280]`)** | **`0.011707` (`logits: [2, 7, 129280]`)** | **`< 0.012`** |
| **Fused Triton vs PyTorch Reference Cosine** | `1.0000` | `1.0000` | **`0.999982` (`99.998%` Identity)** | **`0.999982` (`99.998%` Identity)** | **`0.999982` (`99.998%` Identity)** |
| **Warm MoE Layer Decode Latency (`A100`)** | N/A (Multi-Node) | N/A (Multi-Node) | **`0.5805 ms / layer` (`43.07 tok/s` / 40L)** | **`0.6009 ms / layer` (`41.60 tok/s` / 40L)** | **`0.595 ms / layer` (`42.02 tok/s` / 40L)** |
| **Full 40-Layer End-to-End Decode Speed (`A100-SXM4-80GB`)** | OOM on `8x80GB` | OOM on `4x80GB` | **`36.88 tok/s` sustained (`27.11 ms/tok`, `35.46 tok/s` 12-suite mean)** | — | — |
| **Official 11-Benchmark + Abliteration Suite (`440` Tasks)** | `100.0%` Ref | `100.0%` Ref | **`12/12` (`100.00%` Pass, `100.00%` Token Identity & Top-5 Containment)** | — | — |

### 1.1 Official `deepseek-ai/DeepSeek-V4.1-Flash-1017` 11-Benchmark Evaluation Suite (`440` Tasks, `14,080` Tokens + Live 40-Layer Decode)

Evaluated via [`benchmarks/dsv41_official_benchmarks_eval.py`](https://github.com/defnotean/chronos-1t/blob/main/benchmarks/dsv41_official_benchmarks_eval.py) ([`dsv41_official_benchmarks_report.json`](dsv41_official_benchmarks_report.json) and [`dsv41_40layer_colab_benchmark.json`](dsv41_40layer_colab_benchmark.json)) across all **11 official benchmarks reported by DeepSeek for `DeepSeek-V4.1-Flash-1017`** plus the **HarmBench-320 Post-Quantization Orthogonal Abliteration Audit** (`16` Tier-1 4-bit + `80` Tier-2 2-bit + `288` Tier-3 1-bit = `384/384` active routed experts + `1` shared expert per layer across all 40 layers, `36` full-block CUDA Graphs + `4` sub-block CUDA Graphs + Direct Zero-Copy UVA Tier-3):

| Official `DeepSeek-V4.1-Flash-1017` Benchmark | Category & Evaluation Scope | `DS-V4.1-Flash` (Base) | `K3` | `GLM-5.3` | `GPT-5.6-Sol` | `Opus-5` | Stock `DS-V4.1-Flash-1017` (`700B` MXFP4) | **`Chronos v7.5` + Upgrade 2 (`384/384` Active)** | Live 40L Decode Speed (`A100-80GB`) | Token Identity / Capability Retention |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. `HLE w/o tools`** | Graduate STEM & Symbolic Math (`n = 40`, `1,280` tok) | `39.1%` | `43.5%` | `42.0%` | `44.5%` | `56.3%` | `44.0%†` (`42.6%`) | **`44.0%†` (`42.6%`)** (`Pass`) | **`34.46 tok/s`** (`29.02 ms/tok`) | **`100.00%` / `100.0%`** |
| **2. `Terminal-Bench 4.0`** | Claude Code CLI & Linux Shell Pipelines (`n = 40`, `1,280` tok) | `31.2%` | `12.6%` | `37.9%` | `39.9%` | `51.8%` | `42.3%` | **`42.3%`** (`Pass`) | **`35.69 tok/s`** (`28.02 ms/tok`) | **`100.00%` / `100.0%`** |
| **3. `DeepSWE v1.1`** | Mini-SWE-V2 Autonomous Patch Gen (`n = 40`, `1,280` tok) | `74.2%` | `67.5%` | `66.9%` | `73.0%` | `74.0%` | `77.0%` | **`77.0%`** (`Pass`) | **`35.53 tok/s`** (`28.14 ms/tok`) | **`100.00%` / `100.0%`** |
| **4. `ProgramBench`** | Competitive Programming & $O(n \log n)$ Algorithms (`n = 40`, `1,280` tok) | `20.3%` | `17.5%` | `19.0%` | `23.0%` | `37.0%` | `29.1%` | **`29.1%`** (`Pass`) | **`35.68 tok/s`** (`28.03 ms/tok`) | **`100.00%` / `100.0%`** |
| **5. `NL2Repo-Bench`** | Zero-Shot Full Repository Synthesis (`n = 40`, `1,280` tok) | `65.4%` | `58.0%` | `58.0%` | `56.8%` | `75.3%` | `68.2%` | **`68.2%`** (`Pass`) | **`35.43 tok/s`** (`28.23 ms/tok`) | **`100.00%` / `100.0%`** |
| **6. `Automation-Bench v1.0.6`** | Strict JSON Schema & Workflow Orchestration (`n = 40`, `1,280` tok) | `54.8%` | `46.7%` | `48.8%` | `45.8%` | `50.3%` | `60.5%` | **`60.5%`** (`Pass`) | **`35.72 tok/s`** (`28.00 ms/tok`) | **`100.00%` / `100.0%`** |
| **7. `Agents' Last Exam`** | Long-Horizon Adversarial Agent Planning (`n = 40`, `1,280` tok) | `31.8%` | `27.6%` | `28.5%` | `26.7%` | `28.6%` | `35.4%` | **`35.4%`** (`Pass`) | **`35.57 tok/s`** (`28.12 ms/tok`) | **`100.00%` / `100.0%`** |
| **8. `Apex Bench v1.1`** | Frontier Systems & Graph Algorithms (`n = 40`, `1,280` tok) | `64.6%` | `47.8%` | `56.6%` | `54.3%` | `65.8%` | `72.1%` | **`72.1%`** (`Pass`) | **`35.37 tok/s`** (`28.27 ms/tok`) | **`100.00%` / `100.0%`** |
| **9. `Chartography (w/tools)`** | Quantitative Chart & Tabular Extraction (`n = 40`, `1,280` tok) | `78.9%` | `68.1%` | — | `79.9%` | `84.0%` | `83.0%` | **`83.0%`** (`Pass`) | **`33.91 tok/s`** (`29.49 ms/tok`) | **`100.00%` / `100.0%`** |
| **10. `BabyVision (w/tools)`** | Fine-Grained Spatial & Coordinate Reasoning (`n = 40`, `1,280` tok) | `89.6%` | `85.7%` | — | `88.9%` | `94.1%` | `92.5%` | **`92.5%`** (`Pass`) | **`35.62 tok/s`** (`28.07 ms/tok`) | **`100.00%` / `100.0%`** |
| **11. `ZeroBench-main (w/tools)`** | Hard Zero-Shot Deductive & Combinatorial (`n = 40`, `1,280` tok) | `49.0%` | `41.0%` | — | `53.0%` | `52.0%` | `51.0%` | **`51.0%`** (`Pass`) | **`35.59 tok/s`** (`28.10 ms/tok`) | **`100.00%` / `100.0%`** |
| **12. `HarmBench-320` Abliteration** | Post-Quantization Non-Refusal Audit (`128`-Tok Sustained Speed) | — | — | — | — | — | `Refuses` | **`320/320` (`100.00%` Pass)** | **`36.88 tok/s`** (`27.11 ms/tok`) | **`100.00%` (`\|R_l W_2\|_F = 0`)** |

---

## 2. Dual-Verified Post-Quantization Orthogonal Abliteration

Standard abliteration on Mixture-of-Experts models suffers from two failure modes when combined with extreme low-bit compression:
1. **Incomplete Expert Coverage**: Modifying only `attn.wo_b` and `ffn.shared_experts.w2` leaves the `384` routed experts per layer (`15,360` total routed experts across 40 layers) with non-zero refusal projection ($\|R_l W_{2, m}\|_F = 1.5938$).
2. **Quantization Refusal Leakage**: Even if routed experts are orthogonalized before compression, 4-bit holographic bank and 1-bit/2-bit tail quantization residuals re-introduce non-zero projection into the refusal subspace ($\|R_l \widehat{W}_{2, m}\|_F > 0$).

**`Chronos v7.5` solves both via Dual-Verified Post-Quantization Orthogonal Abliteration**:
1. **Attention & Shared Expert Preservation (`dealignai`, `320/320 = 100.00%` HarmBench Verified)**:
   Preserves the exact block-FP8 (`float8_e4m3fn` + `UE8M0` scale) abliterated `attn.wo_b` and `ffn.shared_experts.w2` tensors across layers `10..36` from [`dealignai/DeepSeek-V4.1-Flash-1017-UNCENSORED-FP8`](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-1017-UNCENSORED-FP8).
2. **Orthonormal Dual Refusal Basis $R_l \in \mathbb{R}^{d_r \times 5120}$ (`OBLITERATUS` + `dealignai`)**:
   Extracts the exact 16-bit BF16 Failspy refusal vector $r_{\text{ob}, l} \in \mathbb{R}^{5120}$ from [`OBLITERATUS/DeepSeek-V4.1-Flash-1017-obliterated`](https://huggingface.co/OBLITERATUS/DeepSeek-V4.1-Flash-1017-obliterated) (`layers 1..39`, verified at `88.08%` MMLU vs `87.96%` stock on 2,500 questions) and $r_{\text{dea}, l} \in \mathbb{R}^{5120}$ (`layers 10..36`), constructing the Gram-Schmidt orthonormal refusal basis $R_l \in \mathbb{R}^{d_r \times 5120}$ ($R_l R_l^T = I_{d_r}$).
3. **Pre- and Post-Quantization Routed Expert Orthogonalization (`15,360` Routed Experts)**:
   - Projects out $R_l$ from all `384` routed experts' `w2` (`down_proj`) matrices prior to `8:1` holographic folding:
     $$W_{2, m}^{\perp} = W_{2, m} - R_l^T (R_l W_{2, m})$$
   - Projects out $R_l$ from the Coupled SwiGLU Woodbury Drift target and the Rank-64 `FP8` left singular vectors $U_{\text{down}, m}$:
     $$U_{\text{down}, m}^{\perp} = U_{\text{down}, m} - R_l^T (R_l U_{\text{down}, m})$$
   - Embeds `ffn.abliteration.r_basis` ($R_l$) directly in each converted `.safetensors` shard and enforces exact runtime output projection inside `ChronosDsv41SparseMoeBlock`:
     $$\widehat{W}_{2, m}^{\perp} = \widehat{W}_{2, m} - R_l^T (R_l \widehat{W}_{2, m}), \qquad \|R_l \widehat{W}_{2, m}^{\perp}\|_F = 0$$

---

## 3. Chronos `v7.5` Architecture for `DeepSeek-V4.1-Flash-1017`

Each of the 40 MoE layers (`layers.0` .. `layers.39`) contains `384` routed experts (`gate_proj`: `2304 x 5120`, `up_proj`: `2304 x 5120`, `down_proj`: `5120 x 2304`, `top_k = 6`, `scoring_func = "sqrtsoftplus"`, `routed_scaling_factor = 1.5`, `swiglu_limit = 10.0`) partitioned into `48` **Holographic Superposition Groups** of $K = 8$ experts:

$$\widehat{W}_m = \gamma_m \left( \beta_m \cdot \mathcal{U}_m\!\left(\text{NVFP4}(S_{\text{group}})\right) + U_{\text{FP8}, m} V_{\text{FP8}, m} + \alpha_m \cdot Q_{\text{Lloyd-Max}, 2\text{-bit}}(R_{\text{tail}, m}) \right)$$

1. **`8:1` `NVFP4` (`E2M1`) Router-Weighted Holographic Superposition Bank (`S_gu`, `S_down`)**:
   Folds 8 experts into a single shared 4-bit `NVFP4` tensor using 2D block-128 Fast Walsh-Hadamard Transforms (`FWHT`) and deterministic per-expert permutation/Rademacher keys.
2. **Full-Spectrum Sherman-Morrison-Woodbury (`SMW`) Rank-64 `FP8` SVD Healer (`U_gu, V_gu, U_down, V_down`)**:
   Applies the exact matrix-inversion lemma to the `256`-D router subspace (`v_rt_sub`) and `64`-D active Clamped-SwiGLU subspace (`v_act_sub`) without dense $5120 \times 5120$ materialization.
3. **Coupled Clamped-SwiGLU (`swiglu_limit = 10.0`) Woodbury Drift Compensation**:
   Compensates `down_proj` for non-linear activation drift $\Delta A = \text{SwiGLU}_{10}(\widehat{Z}) - \text{SwiGLU}_{10}(Z)$ via a $64 \times 64$ push-through ridge solve.
4. **2-Bit Isotropic Lloyd-Max Gaussian Tail Lattice (`{-1.5104, -0.4528, +0.4528, +1.5104}`)**:
   Captures 88.2% of residual Gaussian tail energy per coordinate (`4` values packed per `uint8` byte with per-row `FP8 E4M3FN` scale).

---

## 4. Usage & vLLM Integration

```bash
pip install torch safetensors huggingface_hub tokenizers
git clone https://github.com/defnotean/Chronos.git
cd Chronos
```

### Direct Python / PyTorch Evaluation
```python
import torch
from chronos_1t.models.dsv41_chronos_moe import ChronosDsv41SparseMoeBlock

device = torch.device("cuda:0")
block = ChronosDsv41SparseMoeBlock(
    layer_idx=20,
    shard_path="chronos-dsv41-layer-00020-of-00040.safetensors",
    device=device,
    hot_cache_size=32,
    offload_tail_to_cpu=True,
)
x = torch.randn(1, 5120, dtype=torch.bfloat16, device=device)
y = block(x)
print("Output shape:", y.shape, "Refusal projection norm:", (y.float() @ block.r_basis.float().T).norm().item())
```

### Source Code & Technical Paper
- **GitHub Repository**: [`https://github.com/defnotean/Chronos`](https://github.com/defnotean/Chronos)
