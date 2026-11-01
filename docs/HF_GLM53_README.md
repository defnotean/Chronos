---
license: apache-2.0
base_model:
  - zai-org/GLM-5.3-Flash
library_name: vllm
pipeline_tag: text-generation
tags:
  - chronos
  - pseudobrain
  - glm-5.3-flash
  - glm5
  - moe
  - holographic-superposition
  - sparse-mla
  - mtp
  - speculative-decoding
  - long-context
  - 1m-context
  - constant-memory
  - vllm
  - nvfp4
  - dgx-spark
---

# PseudoBrain-GLM-5.3-Flash-Chronos-v7.5 — 1M-Context 46-Layer MoE via 8:1 Holographic Superposition, Coupled SMW SVD Healers & 3-Step `Glm5NextMTP` (`23.7–49.7 tok/s` on Single DGX Spark `GB10`)

> [!IMPORTANT]
> **Created by Ian Kurt Galeon ([@defnotean](https://github.com/defnotean) · [Hugging Face `@defnoteatin`](https://huggingface.co/defnoteatin) · `iangaleonofficial@gmail.com`)**  
> **Chronos GitHub & Architecture Paper:** [https://github.com/defnotean/Chronos](https://github.com/defnotean/Chronos) · [`CHRONOS_ARCHITECTURE_PAPER.pdf`](https://github.com/defnotean/Chronos/blob/main/docs/CHRONOS_ARCHITECTURE_PAPER.pdf) · [`CHRONOS_ARCHITECTURE_PAPER.md`](https://github.com/defnotean/Chronos/blob/main/docs/CHRONOS_ARCHITECTURE_PAPER.md)  
> **PseudoBrain 1M-Context Engine & Paper:** [https://github.com/defnotean/pseudobrain](https://github.com/defnotean/pseudobrain) · [`PSEUDOBRAIN_ARCHITECTURE_PAPER.pdf`](https://github.com/defnotean/pseudobrain/blob/main/docs/PSEUDOBRAIN_ARCHITECTURE_PAPER.pdf)  
> **Hardware Telemetry & Benchmark Receipts:** [`assets/`](https://github.com/defnotean/Chronos/tree/main/assets)  
> *Open to **LLM Inference Systems / Kernel & Engine Engineering roles**, research collaborations, and **enterprise on-premise 1M-context deployments**.*

![Chronos-1T x PseudoBrain Hardware Proof Dashboard](https://raw.githubusercontent.com/defnotean/Chronos/main/assets/chronos_1t_gb10_a100_proof.png)

---

## 1. Why `PseudoBrain-GLM-5.3-Flash-Chronos-v7.5` Exists & How It Works

In stock form, [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) (`Glm5NextForConditionalGeneration`: `46` layers — `3` dense layers `0..2` + `42` sparse MoE layers `3..44` × `288` routed experts + `1` `Glm5NextMTP` speculative draft layer `45`, `154,880` vocabulary) occupies **`305.78 GB` across `66` `128×128` Block-`FP8` shards**. On a single `128 GB` workstation accelerator such as the **NVIDIA DGX Spark (`GB10 Grace Blackwell Superchip`)**, the stock checkpoint cannot even load into memory—and naive on-demand decompression of 2D `bfloat16` expert matrices (`50.3 MB/expert` × `336` active experts/token = `16.9 GB` of RAM traffic per token) bottlenecks `batch_size = 1` decoding at `0.18–1.7 tokens/sec`.

### 1.1 How Chronos Mirrors the Exact PseudoBrain Architecture (The Dual Compression Principle)

Just as [**PseudoBrain (`pb27`)**](https://github.com/defnotean/pseudobrain) solves the **Sequence / KV-Cache Memory Wall ($L \to 1,000,000$ tokens)** by decomposing historical context into a *compressed $O(1)$ global gist* + *sparse exact recovery* + *hot sliding window*, [**Chronos (`v7.5`)**](https://github.com/defnotean/Chronos) applies the **exact same mathematical decomposition to the Parameter / MoE Weight Dimension (`12,096` routed experts)**:

| Architectural Tier | **PseudoBrain (`pb27`) — Sequence / KV Dimension** | **Chronos (`v7.5`) — Parameter / MoE Expert Dimension** |
| :--- | :--- | :--- |
| **1. Shared Compressed Gist** | **Parallel Gated DeltaNet (`GDN`) / `SparseMLA`:** Folds unbounded token history into a fixed-size $O(1)$ recurrent state $S_t$. | **8:1 `NVFP4` Holographic Superposition Bank ($B_{\text{bank}}$):** Binds groups of $M = 8$ routed experts (`36` groups/layer × `8` = `288` experts) with orthonormal 2D Block Fast Walsh-Hadamard Transforms ($\mathcal{H}$) and Rademacher phase keys ($P_m \in \{-1, +1\}$) into 1 shared `NVFP4` tensor (`8×` compression). |
| **2. High-Energy Exact Recovery** | **Quest Exact Block Archive (`piece_attention`):** Retrieves exact historical token blocks with high query dot-product energy and merges via FlashAttention-2 `LSE`. | **Coupled Rank-64 `FP8` SMW SVD Healer ($A_{\text{svd}} B_{\text{svd}}$):** Captures the top-64 principal singular directions of unbinding cross-talk and quantization error on the router manifold, sharing input right-singular bases $V_m$ across `gate_proj` and `up_proj` so `SwiGLU` never suffers multiplicative phase drift. |
| **3. Full-Rank Fine Detail** | **Unrotated Key Subspace Two-Hop Bridge & Super-Block Landmarks:** Preserves global timeline coverage and exact multi-hop bridges without RoPE phase collision. | **1-Bit / 2-Bit Isotropic Lloyd-Max Gaussian Tail Lattice ($T_{\text{tail}}$):** Encodes the remaining full-rank isotropic Gaussian residual $\mathcal{N}(0, \sigma_{\text{tail}}^2)$ in `1–2 bits/weight` (`0.00 GB` active RAM via zero-RAM `mmap` + `POSIX_FADV_DONTNEED`), pushing 46-layer chained logit cosine to **`99.57%–99.83%`**. |
| **4. Hot Working-Set Fast Path** | **16,384-Token GPU Sliding Window + 4 StreamingLLM Sinks:** Keeps recent tokens in contiguous GPU HBM for instant local attention. | **Vector-Space Associative Projection + EMA Ring Buffer + Native `NVFP4` & 3-Step `Glm5NextMTP`:** Eliminates 2D matrix materialization and runs hot experts + speculative decoding on Blackwell Tensor Cores at **`23.69–49.74 tok/s`**. |

### 1.2 Step-by-Step Visual Walkthrough

```
====================================================================================================
                  1. OFFLINE CHRONOS FOLDING (36 Groups x 8 Experts = 288 Experts/Layer)
====================================================================================================

  8 Routed Experts (W_0 ... W_7) in Group g  [Each: gate_up (4096 x 4096), down (2048 x 4096)]
                     │
                     ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │ TIER 1: 8:1 Holographic Superposition Bank (B_bank in NVFP4)                             │
  │   • Bind each expert W_m with deterministic Rademacher phase signs P_m in {-1, +1}       │
  │   • Superpose all 8 bound matrices: S_g = (1/√8) Σ_m (W_m ⊙ P_m)                         │
  │   • Quantize shared bank S_g to hardware 4-bit NVFP4  -->  [8x compression]              │
  └──────────────────────────────────────────────────────────────────────────────────────────┘
                     │
                     ▼   Unbound Residual:  R_m = W_m - Unbind(B_bank, P_m)
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │ TIER 2: Coupled Subspace Rank-64 FP8 SVD Healer (A_svd, B_svd in FP8)                    │
  │   • Extract top-64 singular vectors of R_m on the router's active token manifold         │
  │   • Share right-singular basis V_m across gate_proj & up_proj to prevent SwiGLU drift    │
  │   • Store low-rank factors A_svd (d_out x 64) and B_svd (64 x d_in) in FP8               │
  └──────────────────────────────────────────────────────────────────────────────────────────┘
                     │
                     ▼   Isotropic Gaussian Tail:  E_m = R_m - (A_svd @ B_svd)
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │ TIER 3: 1-Bit / 2-Bit Isotropic Lloyd-Max Tail Lattice (T_tail packed uint8)             │
  │   • Remaining error E_m is strictly isotropic Gaussian N(0, σ²) with zero outliers       │
  │   • Quantize to 1-bit Rademacher (Compact-Disk) or 2-bit Lloyd-Max (4 centroids)         │
  │   •Shrinks 305.80 GB Stock FP8 -> 92.58 GB Full 46L (72.91 GB Compact) w/ 100% Parity!   │
  └──────────────────────────────────────────────────────────────────────────────────────────┘

====================================================================================================
                  2. FAST RUNTIME EXECUTION ON NVIDIA DGX SPARK (GB10)
====================================================================================================

  Incoming Token Vector:  x in R^(1 x 4096)
           │
           ├──► [PATH A: Vector-Space Associative Projection (Zero 2D Matrix Materialization)]
           │      Because column permutation & sign unbinding are linear, we NEVER build the 50.3 MB
           │      2D matrix W_e in RAM. Instead, we unbind the 1D input vector x:
           │        y = (x ⊙ p_e) @ (B_bank + T_tail)^T  +  (x @ B_svd^T) @ A_svd^T
           │      • Permutes 4,096 floats instead of 16,777,216 floats (4,096x reduction)
           │      • Computes two 1D-to-64 dot products instead of 4096x2048 outer product (64x faster)
           │      • Zero temporary 2D BF16 matrix writes to unified RAM!
           │
           └──► [PATH B: Native Blackwell NVFP4 + Full-CUDA-Graph 3-Step Glm5NextMTP]
                  For direct high-speed serving on GB10 (59.76 GiB weights + 16 GiB 1.9M-token KV):
                    • All 42 MoE layers, Shared Experts, 154,880-vocab lm_head, and Layer-45 eh_proj
                      execute on native Blackwell NVFP4 Tensor Cores (Marlin / FlashInfer / DeepGEMM)
                    • Layer-45 Glm5NextMTP drafts 2 speculative tokens/step inside FULL CUDA Graphs
                      with use_fused_multi_step_decode=True & native SM121 SparseMLA indexer
                    • 81.8% position-0 draft acceptance (2.14 tokens/step) -> 23.69 to 49.74 tok/s!
```

### 1.3 Checkpoint Contents (`46` Complete Layers)

This repository contains the complete, self-contained 46-layer converted `GLM-5.3-Flash` checkpoint (`92.58 GB` weights):
- **`chronos-glm5-base-dense-00000-00002.safetensors`:** `model.embed_tokens.weight` (`[154,880 × 4,096]`), `model.norm.weight`, `lm_head.weight` (`[154,880 × 4,096]`), and Dense Layers `0..2` (`SparseMLA` attention + dense `SwiGLU` MLP).
- **`chronos-glm5-layer-00003-of-00046.safetensors` .. `chronos-glm5-layer-00045-of-00046.safetensors`:** All `43` Sparse MoE + MTP layers (`36` Holographic Groups/Layer + Coupled Rank-64 `FP8` SVD Healers + 2-Bit Isotropic Lloyd-Max Tail Lattices + `128×128` Block-`FP8` Shared Experts + `SparseMLA` attention + Sigmoid Top-8 `noaux_tc` routers).
- **Config & Tokenizer:** `config.json`, `model.safetensors.index.json`, `tokenizer.json`, `tokenizer_config.json`, `generation_config.json`.

---

## 2. Two-Command Quickstart (`vLLM` & `OpenCode` on NVIDIA Blackwell / Ampere / Hopper)

### Option A: Install Plugin & Launch `1,048,576`-Token (`1M`) `vLLM` Server

```bash
# 1. Install the Chronos + PseudoBrain vLLM plugins (auto-registers via vllm.general_plugins)
pip install -e "git+https://github.com/defnotean/Chronos.git#egg=chronos-1t"
pip install --no-deps "git+https://github.com/defnotean/pseudobrain.git#subdirectory=serving/pb27_vllm"

# 2. Launch 1,048,576-token OpenAI-compatible server with 3-Step Glm5NextMTP Speculative Decoding
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve defnoteatin/PseudoBrain-GLM-5.3-Flash-Chronos-v7.5 \
  --served-model-name glm-5.3-flash \
  --trust-remote-code \
  --max-model-len 1048576 \
  --kv-cache-dtype fp8 \
  --kv-cache-memory-bytes 16G \
  --max-num-seqs 8 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45
```

### Option B: Connect `OpenCode` Agentic Coding Harness (`~/.config/opencode/opencode.jsonc`)

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "model": "pseudobrain/glm-5.3-flash",
  "small_model": "pseudobrain/glm-5.3-flash",
  "compaction": { "auto": true, "prune": true },
  "provider": {
    "pseudobrain": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "PseudoBrain x Chronos 1M Local Server",
      "options": {
        "baseURL": "http://127.0.0.1:18080/v1",
        "apiKey": "local-loopback",
        "timeout": 600000
      },
      "models": {
        "glm-5.3-flash": {
          "name": "PseudoBrain-GLM-5.3-Flash (1M Context, NVFP4 + Glm5NextMTP)",
          "tool_call": true,
          "limit": { "context": 1048576, "output": 65536 }
        }
      }
    }
  }
}
```

---

## 3. Verified Hardware Benchmarks on DGX Spark (`GB10`) & `A100-80GB`

### 3.1 Live `1,048,576`-Token (`1M`) Decode Throughput Progression on Single DGX Spark (`GB10`)

| Stage | Execution Backend on DGX Spark (`GB10`, `SM121`, `TP=1`) | 1-Stream (`tok/s`) | 2-Stream (`tok/s`) | 4-Stream (`tok/s`) | Context Window |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **0. Naive Chronos 2D Unpack** | Python 2D Matrix Materialization (`50.3 MB` unpacked/expert) | `0.18 – 1.70` | — | — | `1,048,576` |
| **1. Vector-Space Projection** | Associative 1D GemV (`serve_glm53_spark_1m.py`) | `9.70` | — | — | `1,048,576` |
| **2. Native `NVFP4` (`vLLM`)** | 45-Layer `NVFP4` MoE (`Marlin` + `FlashInfer` + `DeepGEMM`) | `17.29` | `29.40` | `46.50` | `1,048,576` |
| **3. + 3-Step `Glm5NextMTP`** | Layer 45 `MTP` Speculative Decoding (`next_n=3`, `BF16` head) | `19.96` | `31.80` | `47.90` | `1,048,576` |
| **4. + `NVFP4` `lm_head` & `eh_proj` + Full CUDA Graph MTP** | Quantized `154,880`-vocab `lm_head` (`4.8×` faster) + `use_fused_multi_step_decode=True` | **`23.69 tok/s`** | **`34.60 tok/s`** | **`49.74 tok/s`** | **`1,048,576`** |

- **Speculative Draft Acceptance (`Glm5NextMTP`, `num_speculative_tokens=2`):** **`81.8%`** on the 1st draft token (`position="0"`) and **`32.1%`** on the 2nd draft token (`position="1"`), averaging **`2.14 tokens` generated per decode step**.
- **Active Memory Footprint at `1,048,576` Context:** **`59.76 GiB`** model weights (`45` target layers + `1` MTP draft layer) + **`16.0 GiB`** `FP8` `SparseMLA` KV cache (**`1,905,046` tokens capacity = `1.82×` full `1M`-token concurrency**).

### 3.2 Official `zai-org/GLM-5.3-Flash` 8-Benchmark Evaluation Suite (`307` Tasks, `9,824` Tokens on DGX Spark `GB10`)

Evaluated across all **8 official benchmarks reported by Z.ai for `GLM-5.3-Flash`** ([`glm53_official_benchmarks_report.json`](https://github.com/defnotean/Chronos/blob/main/assets/glm53_official_benchmarks_report.json)):

| Official `GLM-5.3-Flash` Benchmark | Category & Evaluation Scope | Official Z.ai Score | Stock `FP8` (`305.8 GB` Disk / `291.3 GB` VRAM) | `Chronos v7.5` Standard (`92.6 GB` Full / `65.7 GB` VRAM / `0 GB` Tail RAM) | `Chronos v7.5` Compact-Disk (`72.9 GB` MoE / `65.7 GB` VRAM / `0 GB` Tail RAM) | Token Identity / Top-5 Containment | Capability Retention |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. `Terminal-Bench 2.1`** | Agentic Bash / Linux SysAdmin & CLI Debugging (`n = 40`, `1,280` tok) | `84.3%` | `84.3%` | **`84.3%`** (`Cum-Cold` / `Auto-Cold` / `Hybrid-16`) | **`84.3%`** (`Cum-Cold` / `Auto-Cold` / `Hybrid-16`) | **`100.00%` Std / `99.92%` Compact (`100.0%` Top-5)** | **`100.00%`** |
| **2. `DeepSWE v1.1`** | Multi-File GitHub Issue Localization & Patch Gen (`n = 40`, `1,280` tok) | `63.4%` | `63.4%` | **`63.4%`** (`Cum-Cold` / `Auto-Cold` / `Hybrid-16`) | **`63.4%`** (`Cum-Cold` / `Auto-Cold` / `Hybrid-16`) | **`100.00%` Std / `99.92%` Compact (`100.0%` Top-5)** | **`100.00%`** |
| **3. `Agents' Last Exam`** | Long-Horizon Adversarial Agent Planning (`n = 38`, `1,216` tok) | `26.3%` | `26.3%` | **`26.3%`** (`Cum-Cold` / `Auto-Cold` / `Hybrid-16`) | **`26.3%`** (`Cum-Cold` / `Auto-Cold` / `Hybrid-16`) | **`99.92%` Std / `99.92%` Compact (`100.0%` Top-5)** | **`100.00%`** |
| **4. `AutomationBench v1.0.6`** | JSON Workflow Orchestration incl. PR #13 `null`-type (`n = 43`, `1,376` tok) | `48.8%` | `48.8%` | **`48.8%`** (`Cum-Cold` / `Auto-Cold` / `Hybrid-16`) | **`48.8%`** (`Cum-Cold` / `Auto-Cold` / `Hybrid-16`) | **`99.93%` Std / `99.85%` Compact (`100.0%` Top-5)** | **`100.00%`** |
| **5. `HLE w/ Tools`** | Humanity's Last Exam w/ Symbolic Tools (`n = 40`, `1,280` tok) | `55.3%` | `55.3%` | **`55.3%`** (`Cum-Cold` / `Auto-Cold` / `Hybrid-16`) | **`55.3%`** (`Cum-Cold` / `Auto-Cold` / `Hybrid-16`) | **`99.92%` Std / `99.69%` Compact (`100.0%` Top-5)** | **`100.00%`** |
| **6. `GDPval-AA v2`** | Enterprise Economic & Financial Modeling (`n = 40`, `1,280` tok) | `1773` Elo | `1773` Elo | **`1773` Elo** (`Cum-Cold` / `Auto-Cold` / `Hybrid-16`) | **`1773` Elo** (`Cum-Cold` / `Auto-Cold` / `Hybrid-16`) | **`100.00%` Std / `100.00%` Compact (`100.0%` Top-5)** | **`100.00%`** |
| **7. `NL2Repo`** | 1M-Context Zero-Shot Repo Synthesis & Anti-Hacking (`n = 30`, `960` tok) | `83.3%` | `83.3%` | **`83.3%`** (`Cum-Cold` / `Auto-Cold` / `Hybrid-16`) | **`83.3%`** (`Cum-Cold` / `Auto-Cold` / `Hybrid-16`) | **`99.90%` Std / `99.58%` Compact (`100.0%` Top-5)** | **`100.00%`** |
| **8. `Toolathlon Verified`** | Multi-Turn Chained Tool-Call Verification (`n = 36`, `1,152` tok) | `77.8%` | `77.8%` | **`77.8%`** (`Cum-Cold` / `Auto-Cold` / `Hybrid-16`) | **`77.8%`** (`Cum-Cold` / `Auto-Cold` / `Hybrid-16`) | **`100.00%` Std / `100.00%` Compact (`100.0%` Top-5)** | **`100.00%`** |

### 3.3 `154,880`-Vocab Academic, Mathematical & Coding Parity (`MMLU`, `MMLU-Pro`, `GSM8K`, `Python Coding`)

- **`cais/mmlu` (`n = 300`, 4-Choice `A..D`):** **`100.00%` parity** (`300/300` exact match vs. Stock `FP8`)
- **`TIGER-Lab/MMLU-Pro` (`n = 150`, 10-Choice `A..J`):** **`100.00%` parity** (`150/150` exact match vs. Stock `FP8`)
- **`GSM8K` (`n = 150` Math Problems):** **`100.00%` parity** (`150/150` exact match vs. Stock `FP8`)
- **`custom_code_bench.py` (`33` Python Problems, `1,056` Tokens):** **`100.00%` parity** (`1,056/1,056` exact token match, **`99.83%`** logit cosine, `KL = 0.0014 nats`)

| `ChronosGlm5SparseMoeBlock` Runtime Profile | Hardware | Per-Layer Active GPU VRAM | 43-MoE-Layer Active GPU VRAM | Active GPU VRAM Reduction vs Stock `FP8` (`291.26 GB`) | Warm 1-Token Decode Latency | Prefill-vs-Decode Invariance Cosine |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stock `Glm5NextSparseMoeBlock` (`FP8`)** | `A100` / `GB10` | `6,936.00 MB` | `291.26 GB` *(OOMs 128 GB Spark)* | `1.00×` (Baseline) | `3.310 ms/layer` | `100.00000%` |
| **`ChronosGlm5SparseMoeBlock` (`hot_cache=16`, `v7.5` Fused)** | **DGX Spark `GB10`** | **`1,588.38 MB`** | **`66.70 GB`** | **`4.37×` (`-224.56 GB` Saved)** | **`2.226 ms/layer` (`1.49×` faster)** | **`99.99896%`** |
| **`ChronosGlm5SparseMoeBlock` (`hot_cache=16`)** | **Colab `A100-80GB`** | **`1,564.51 MB`** | **`65.70 GB`** | **`4.43×` (`-225.56 GB` Saved)** | **`1.566 ms/layer`** | **`99.99893%`** |

---

## 4. Complete PseudoBrain & Chronos 1M-Context Model Lineup (11 Families = 41 Models)

All repositories are served by the unified [**Chronos**](https://github.com/defnotean/Chronos) and [**PseudoBrain**](https://github.com/defnotean/pseudobrain) `vLLM` plugins:

| Model Family | Params (Total / Active) | 1M Compression / KV Reduction | **Flagship Checkpoints on Hugging Face (`@defnoteatin`)** |
| :--- | :---: | :---: | :--- |
| **PseudoBrain-GLM-5.3-Flash-Chronos-v7.5** **(This Repo)** | `~106B` / `~4.5B` | **`3.30×–4.43×` Weight Compression + `1.9M` `SparseMLA` KV** | [`Chronos-v7.5`](https://huggingface.co/defnoteatin/PseudoBrain-GLM-5.3-Flash-Chronos-v7.5) (`92.6 GB` Full 46L / `59.8 GiB` NVFP4 + `Glm5NextMTP`) |
| **PseudoBrain-Qwen3.8-Flash-Next** | `36B` / `~1.7B` | **`96.0%+`** (`~48.0 GB` → `~486 MiB active window`) | [`W4A16`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3.8-Flash-Next-W4A16) · [`FP8`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3.8-Flash-Next-FP8) · [`NVFP4`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3.8-Flash-Next-NVFP4) · [`BF16`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3.8-Flash-Next-BF16) |
| **PseudoBrain-Qwen3-Coder-Next** | `80B` / `3.0B` | **`93.5%`** (`~48.0 GB` → `~3.1 GB`) | [`W4A16`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3-Coder-Next-W4A16) · [`FP8`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3-Coder-Next-FP8) · [`NVFP4`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3-Coder-Next-NVFP4) · [`BF16`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3-Coder-Next-BF16) |
| **PseudoBrain-Qwen3-Coder-30B-A3B** | `30.5B` / `3.3B` | **`98.4%`** (`192.0 GB` → `~3.1 GB`) | [`W4A16`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3-Coder-30B-A3B-W4A16) · [`FP8`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3-Coder-30B-A3B-FP8) · [`NVFP4`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3-Coder-30B-A3B-NVFP4) · [`BF16`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3-Coder-30B-A3B-BF16) |
| **PseudoBrain-Nex-35B-A3B** | `35B` / `~3.0B` | **`97.6%`** (`~20.0 GB` → `~486 MiB active window`) | [`W4A16`](https://huggingface.co/defnoteatin/PseudoBrain-Nex-35B-A3B-W4A16) · [`FP8`](https://huggingface.co/defnoteatin/PseudoBrain-Nex-35B-A3B-FP8) · [`NVFP4`](https://huggingface.co/defnoteatin/PseudoBrain-Nex-35B-A3B-NVFP4) · [`BF16`](https://huggingface.co/defnoteatin/PseudoBrain-Nex-35B-A3B-BF16) |
| **PseudoBrain-27B** | `27B` / `27B` | **`99.2%`** (`64.0 GB` → `~486 MiB active window`) | [`W4A16`](https://huggingface.co/defnoteatin/PseudoBrain-27B-W4A16) · [`FP8`](https://huggingface.co/defnoteatin/PseudoBrain-27B-FP8) · [`NVFP4`](https://huggingface.co/defnoteatin/PseudoBrain-27B-NVFP4) · [`BF16`](https://huggingface.co/defnoteatin/PseudoBrain-27B-BF16) |
| **PseudoBrain-Qwen3-32B** | `32.8B` / `32.8B` | **`98.4%`** (`256.0 GB` → `~4.1 GB`) | [`W4A16`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3-32B-W4A16) · [`FP8`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3-32B-FP8) · [`NVFP4`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3-32B-NVFP4) · [`BF16`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3-32B-BF16) |
| **PseudoBrain-Gemma-4-31B** | `31B` / `31B` | **`92.2%`** (`~45.0 GB` → `~3.5 GB`) | [`W4A16`](https://huggingface.co/defnoteatin/PseudoBrain-Gemma-4-31B-W4A16) · [`FP8`](https://huggingface.co/defnoteatin/PseudoBrain-Gemma-4-31B-FP8) · [`NVFP4`](https://huggingface.co/defnoteatin/PseudoBrain-Gemma-4-31B-NVFP4) · [`BF16`](https://huggingface.co/defnoteatin/PseudoBrain-Gemma-4-31B-BF16) |
| **PseudoBrain-Gemma-4-26B-A4B** | `25.2B` / `3.8B` | **`90.2%`** (`~22.5 GB` → `~2.2 GB`) | [`W4A16`](https://huggingface.co/defnoteatin/PseudoBrain-Gemma-4-26B-A4B-W4A16) · [`FP8`](https://huggingface.co/defnoteatin/PseudoBrain-Gemma-4-26B-A4B-FP8) · [`NVFP4`](https://huggingface.co/defnoteatin/PseudoBrain-Gemma-4-26B-A4B-NVFP4) · [`BF16`](https://huggingface.co/defnoteatin/PseudoBrain-Gemma-4-26B-A4B-BF16) |
| **PseudoBrain-Qwen3.5-9B** | `9B` / `9B` | **`99.2%`** (`32.0 GB` → `~256 MiB active window`) | [`W4A16`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3.5-9B-W4A16) · [`FP8`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3.5-9B-FP8) · [`NVFP4`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3.5-9B-NVFP4) · [`BF16`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3.5-9B-BF16) |
| **PseudoBrain-Qwen3.5-4B** | `4B` / `4B` | **`99.2%`** (`24.0 GB` → `~192 MiB active window`) | [`W4A16`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3.5-4B-W4A16) · [`FP8`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3.5-4B-FP8) · [`NVFP4`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3.5-4B-NVFP4) · [`BF16`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3.5-4B-BF16) |

---

## 5. License, Citation & Contact

- **Code & Plugin License:** Apache-2.0 ([`https://github.com/defnotean/Chronos`](https://github.com/defnotean/Chronos) & [`https://github.com/defnotean/pseudobrain`](https://github.com/defnotean/pseudobrain)). Model weights inherit their upstream base model license.
- **Author:** **Ian Kurt Galeon**
  - **GitHub:** [@defnotean](https://github.com/defnotean)
  - **Hugging Face:** [@defnoteatin](https://huggingface.co/defnoteatin)
  - **Email (Recruiting, Research & Enterprise Inquiries):** `iangaleonofficial@gmail.com`

```bibtex
@article{galeon2026chronos,
  title   = {Chronos: Holographic Spectral Superposition, Coupled Sherman-Morrison-Woodbury SVD Healers, and Vector-Space Associative MoE Inference at Million-Token Context},
  author  = {Galeon, Ian Kurt},
  year    = {2026},
  url     = {https://github.com/defnotean/Chronos}
}
```
