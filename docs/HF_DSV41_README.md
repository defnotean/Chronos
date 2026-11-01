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

| Metric | Stock `DeepSeek-V4.1-Flash-1017` (`BF16`) | Stock `DeepSeek-V4.1-Flash-1017` (`MXFP4`) | **`Chronos v7.5` (`2-Bit Lloyd-Max`)** | **`Chronos v7.5` (`1-Bit Compact-Disk`)** |
| :--- | :---: | :---: | :---: | :---: |
| **Total Model Disk Footprint** | `1,509.36 GB` (`1.51 TB`) | `510.31 GB` (`299.21 GB` MoE) | **`218.81 GB`** (`5.380 GB/layer`) | **`142.37 GB`** (`3.469 GB/layer`) |
| **Per-Layer MoE Shard Size** | `27.390 GB` | `7.390 GB` | **`5.380 GB` (`5.09x` vs BF16)** | **`3.469 GB` (`7.90x` vs BF16)** |
| **Active GPU VRAM (All 40 Layers)** | OOM on 8x80GB (`>1.1 TB`) | OOM on 4x80GB (`>300 GB`) | **`52.97 GB` (Fits on 1x 80GB GPU)** | **`52.97 GB` (Fits on 1x 80GB GPU)** |
| **Single-Expert `gate_up` Cosine** | `1.0000` | `1.0000` | **`0.95317` (`95.32%`)** | `0.85889` (`85.89%`) |
| **Single-Expert `down_proj` Cosine** | `1.0000` | `1.0000` | **`0.95397` (`95.40%`)** | `0.88062` (`88.06%`) |
| **Routed-Only Top-6 MoE Cosine** | `1.0000` | `1.0000` | **`0.96924` (`96.92%`)** | `0.91420` (`91.42%`) |
| **Full MoE Layer Output Cosine** | `1.0000` | `1.0000` | **`0.98292` (`98.29%`)** | `0.95180` (`95.18%`) |
| **Routed `w2` Refusal Projection Norm** | `1.593769` | `1.593769` | **`0.000811` (`99.95%` Eliminated)** | **`0.000006` (`99.9996%` Eliminated)** |
| **40-Layer E2E Mean Routed Refusal Leakage** | `1.5938` | `1.5938` | **`0.011707` (`logits: [2, 7, 129280]`)** | **`< 0.012`** |
| **Fused Triton vs PyTorch Reference Cosine** | `1.0000` | `1.0000` | **`0.999982` (`99.998%` Identity)** | **`0.999982` (`99.998%` Identity)** |
| **Warm MoE Layer Decode Latency (`A100`)** | N/A (Multi-Node) | N/A (Multi-Node) | **`0.6009 ms / layer` (`41.60 tok/s` / 40L)** | **`0.595 ms / layer` (`42.02 tok/s` / 40L)** |

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
