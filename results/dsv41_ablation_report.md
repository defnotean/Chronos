# Chronos DeepSeek-V4.1-Flash Benchmark Ablation & Root-Cause Report

**Target Hardware & Model:** Resident 40-layer Chronos DeepSeek-V4.1-Flash (`GLOBAL_DSV41_MODEL`) on Colab G4 (RTX PRO 6000 96 GB VRAM + 176 GB host RAM; `94.971 GiB` VRAM, `2.506 GiB` free after 48K upgrade).  
**Fixed Evaluation Set:** [`ablate_fixed_600_questions.json`](file:///c:/Users/Eating/Desktop/Chronos%201T/results/ablate_fixed_600_questions.json) (`n = 600`: MMLU `n = 300` stratified across 57 subjects; MMLU-Pro `n = 150` stratified across all 14 categories; GSM8K `n = 150`).  
**Reference Baselines:** Stock `DeepSeek-V4.1-Flash` MMLU `87.96%`; `Qwen Flash Next` MMLU-Pro `81.33%`, GSM8K `96.00%`.  
**Scope Note:** Per explicit override, the optional `A1_ceil1536` single-variable variant was skipped in favor of the combined `A_unleashed` configuration (`CHRONOS_ABLATE_GUARDS=1`, `think_ceiling=3072`, `kv_limit=8192`).

---

## 1. Pre-Change Baseline Configuration (`A0`)

Recorded verbatim in [`ablate_prechange_baseline_config.json`](file:///c:/Users/Eating/Desktop/Chronos%201T/results/ablate_prechange_baseline_config.json):

### 1.1 Inference & Decoding Parameters (`scripts/run_40layer_chimera_dsv41_a100.py`)

| Parameter | Exact Baseline Value / Formula |
| :--- | :--- |
| `max_seq_len` | `int(os.environ.get('CHRONOS_MAX_SEQ_LEN', '4096')) == 4096` |
| `kv_limit` (`benchmarks/run_dsv41_full_long_benchmarks.py`) | `2048` |
| `eff_think_budget` | `max_think_tokens if max_think_tokens > 0 else (min(384, max(128, max_new_tokens // 2)) if in_think else 0)` |
| `over_budget_trigger` | `eff_think_budget > 0 and think_tok_count >= eff_think_budget and (at_sent or rep_hit or think_tok_count >= eff_think_budget + 16)` |
| `rep_thresh` | `5 if in_code else 1` (inside `<think>`, `in_code` is `False` so `rep_thresh == 1`, plus `norm_fourgrams` `-12.0` logit penalty) |
| `should_close_early` | `think_tok_count >= 48 and at_sent and rep_hit` (`at_sent`: suffix in `('.', '!', '?', '\n', ':')`; `rep_hit`: `any(cnt >= 2 for cnt in fourgram_counts[trig].values())`) |

### 1.2 Baseline Suite Budgets (`benchmarks/run_dsv41_full_long_benchmarks.py`)

| Suite | `thinking_mode` | `max_new_tokens` | `max_think_tokens` | Effective `<think>` Cap (`A0`) |
| :--- | :---: | :---: | :---: | :---: |
| `mmlu` | `chat` | `12` | `0` | `0` |
| `mmlu_pro` | `thinking` | `768` | `0` | `384` (prior 60-item run: `256` / `128`) |
| `gsm8k` | `thinking` | `768` | `0` | `384` (prior 60-item run: `320` / `160`) |
| `terminal`, `babyvision`, `chart` | `thinking` | `512` | `0` | `256` |
| `humaneval`, `mbpp`, `apex`, `math`, `ale`, `zerobench`, `swe` | `thinking` | `768` | `0` | `384` |
| `automation` | `chat` | `220` | `0` | `0` |
| `harmbench` | `chat` | `96` | `0` | `0` |

---

## 2. Master Ablation Summary (`n = 600` Fixed Set)

All percentages below are from the verified JSONL/report outputs compiled in [`ablate_A5_wilson_ci.json`](file:///c:/Users/Eating/Desktop/Chronos%201T/results/ablate_A5_wilson_ci.json).

| arm | MMLU | MMLU-Pro | GSM8K | Δ vs A0 | 95% CI | mean tokens | mean think tokens | % of answers where `</think>` was forced |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A0** (Baseline) | `81.67%` (`245/300`) | `74.67%` (`112/150`) | `92.00%` (`138/150`) | `0.00 pp` | `[79.26%, 85.33%]` | `148.89` | `102.62` | `14.67%` (`88/600`) |
| **A1** (No caps + MMLU thinking `1024`, `ceil=384`) | `81.00%` (`243/300`) | `79.33%` (`119/150`) | `92.67%` (`139/150`) | `+1.00 pp` | `[80.32%, 86.26%]` | `249.21` | `169.49` | `18.67%` (`112/600`) |
| **A2** (`CHRONOS_ABLATE_GUARDS=1` on A0) | `81.67%` (`245/300`) | `81.33%` (`122/150`) | `95.33%` (`143/150`) | `+2.50 pp` | `[81.92%, 87.63%]` | `133.85` | `90.25` | `9.00%` (`54/600`) |
| **A_unleashed** (Guards off + `ceil=3072`, `kv=8192`) | `82.67%` (`248/300`) | `82.00%` (`123/150`) | `95.33%` (`143/150`) | `+3.17 pp` | `[82.64%, 88.24%]` | `361.50` | `305.57` | `3.67%` (`22/600`) |

*Notes on Summary Table:*
- `Δ vs A0` and `95% CI` in the master table report overall micro accuracy across all `n = 600` fixed questions (`A0 = 82.50%`, `A1 = 83.50%`, `A2 = 85.00%`, `A_unleashed = 85.67%`) with 95% Wilson confidence intervals. Equal-weight macro averages across the 3 suites are `82.78%` (`A0`), `84.33%` (`A1`, `+1.55 pp`), `86.11%` (`A2`, `+3.33 pp`), and `86.67%` (`A_unleashed`, `+3.89 pp`).
- **GSM8K clean-grader re-scores** (fixing raw `\mathbf{}`, `\text{}`, `\%` formatting and `<number>` prompt-echo mismatches): `A0` `92.00% -> 94.67%` (`142/150`), `A1` `92.67% -> 94.67%` (`142/150`), `A2` `95.33% -> 96.67%` (`145/150`, exceeding Qwen Flash Next's `96.00%`), `A_unleashed` `95.33%` (`143/150`).
- On **MMLU-Pro**, `A2` tied Qwen Flash Next (`81.33%`) and **`A_unleashed` surpassed Qwen Flash Next (`82.00%` vs `81.33%`)**.

---

## 3. Per-Suite Breakdown & Paired McNemar Tests (`A5`)

Exact two-sided paired McNemar transitions against `A0` (`b = A0 fail & arm pass`, `c = A0 pass & arm fail`) from [`ablate_A5_wilson_ci.json`](file:///c:/Users/Eating/Desktop/Chronos%201T/results/ablate_A5_wilson_ci.json):

| Suite | Arm | Pass Rate | 95% Wilson CI | Δ vs A0 | Forced `</think>` % | `mean_tok` / `think_tok` | McNemar `(b, c)` / `net` | Exact `p` vs A0 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MMLU** (`n=300`) | `A0` (`chat 12`) | `81.67%` (`245`) | `[76.90%, 85.64%]` | `0.00 pp` | `0.00%` (`0`) | `1.04 / 0.00` | `—` | `—` |
| | `A1` (`think 1024`) | `81.00%` (`243`) | `[76.18%, 85.04%]` | `-0.67 pp` | `13.33%` (`40`) | `174.86 / 136.80` | `(33, 35)` / `-2` | `0.903597` |
| | `A2` (`chat 12`, guards off) | `81.67%` (`245`) | `[76.90%, 85.64%]` | `0.00 pp` | `0.00%` (`0`) | `1.04 / 0.00` | `(7, 7)` / `0` | `1.000000` |
| | `A_unleashed` (`think 2048`) | `82.67%` (`248`) | `[77.98%, 86.53%]` | `+1.00 pp` | `4.00%` (`12`) | `258.79 / 233.00` | `(29, 26)` / `+3` | `0.787706` |
| **MMLU-Pro** (`n=150`) | `A0` (`768`, `ceil=384`) | `74.67%` (`112`) | `[67.15%, 80.95%]` | `0.00 pp` | `40.00%` (`60`) | `291.15 / 229.74` | `—` | `—` |
| | `A1` (`2048`, `ceil=384`) | `79.33%` (`119`) | `[72.16%, 85.04%]` | `+4.66 pp` | `35.33%` (`53`) | `341.19 / 227.81` | `(13, 6)` / `+7` | `0.167068` |
| | `A2` (`768`, guards off) | `81.33%` (`122`) | `[74.34%, 86.76%]` | `+6.66 pp` | `28.67%` (`43`) | `262.07 / 207.65` | `(16, 6)` / `+10` | `0.052479` |
| | `A_unleashed` (`4096`, `ceil=3072`) | `82.00%` (`123`) | `[75.08%, 87.32%]` | `+7.33 pp` | `6.67%` (`10`) | `610.37 / 560.35` | `(15, 4)` / `+11` | **`0.019211`** |
| **GSM8K** (`n=150`) | `A0` (`768`, `ceil=384`) | `92.00%` (`138`) | `[86.54%, 95.36%]` | `0.00 pp` | `18.67%` (`28`) | `302.33 / 180.73` | `—` | `—` |
| | `A1` (`2048`, `ceil=384`) | `92.67%` (`139`) | `[87.35%, 95.86%]` | `+0.67 pp` | `12.67%` (`19`) | `305.92 / 176.55` | `(9, 8)` / `+1` | `1.000000` |
| | `A2` (`768`, guards off) | `95.33%` (`143`) | `[90.68%, 97.72%]` | `+3.33 pp` | `7.33%` (`11`) | `271.25 / 153.33` | `(9, 4)` / `+5` | `0.266846` |
| | `A_unleashed` (`4096`, `ceil=3072`) | `95.33%` (`143`) | `[90.68%, 97.72%]` | `+3.33 pp` | `0.00%` (`0`) | `318.05 / 195.93` | `(8, 3)` / `+5` | `0.226562` |
| **Overall Micro** (`n=600`) | `A0` | `82.50%` (`495`) | `[79.26%, 85.33%]` | `0.00 pp` | `14.67%` (`88`) | `148.89 / 102.62` | `—` | `—` |
| | `A1` | `83.50%` (`501`) | `[80.32%, 86.26%]` | `+1.00 pp` | `18.67%` (`112`) | `249.21 / 169.49` | `(55, 49)` / `+6` | `0.624144` |
| | `A2` | `85.00%` (`510`) | `[81.92%, 87.63%]` | `+2.50 pp` | `9.00%` (`54`) | `133.85 / 90.25` | `(32, 17)` / `+15` | **`0.044384`** |
| | `A_unleashed` | `85.67%` (`514`) | `[82.64%, 88.24%]` | `+3.17 pp` | `3.67%` (`22`) | `361.50 / 305.57` | `(52, 33)` / `+19` | `0.050251` |

*GSM8K Prompt/Parser Diagnostic:* Disabling the `<think>` repetition guards (`A2`) both eliminated post-thinking repetition loops and stopped premature mid-calculation `</think>` injections (`18.67% -> 7.33%` forced in `A2`, `0.00%` in `A_unleashed`). With the clean GSM8K extractor, `A2` achieves `145/150 = 96.67%` (`[92.43%, 98.57%]`). Notice also that the earlier 60-item benchmark deficit (`63.33%` MMLU-Pro, `83.33%` GSM8K) was largely caused by ultra-tight `128`/`160` think caps prior to raising `max_new_tokens` to `768`.

---

## 4. Refusal Orthogonalization (`A3`) Inspection

From [`ablate_A3_inspection.json`](file:///c:/Users/Eating/Desktop/Chronos%201T/results/ablate_A3_inspection.json):
- `CHRONOS_RUNTIME_R_BASIS = "0"`, and `r_basis is None` on **`40/40`** resident layers.
- Refusal orthogonalization (`dealignai/OBLITERATUS` projection) was baked offline into the safetensors weights of `attn.wo_b`, `shared_experts.w2`, Upgrade-2 hot-96 `w2`, and holographic banks/healers.
- Because the subtracted orthogonal projection deltas are not stored in the converted checkpoint, **`A3` cannot be toggled off at runtime without reconverting the original weights.**

---

## 5. Cold-Tier Compression & Routing Attribution (`A4`)

Read-only analysis of `A1` routing logs in [`ablate_A4_compression_attribution.json`](file:///c:/Users/Eating/Desktop/Chronos%201T/results/ablate_A4_compression_attribution.json), covering **`88,754` authentic prefill tokens** across all `600` questions:

- **Instrumentation Transparency:** In `/content/ablate_A1_routing/*.pt`, decode rows `route_ids[prompt_tokens:]` (`149,512` tokens across `600` files) were all zeros because `A1` used pre-captured CUDAGraph replay during decode. All decode rows were excluded (`r_ids[:prompt_tokens]` only), and [`dsv41_chronos_moe.py`](file:///c:/Users/Eating/Desktop/Chronos%201T/chronos_1t/models/dsv41_chronos_moe.py#L3277-L3286) confirms `hot_mask_lut = (tier_slot_map_gpu >= 0) & (tier_slot_map_gpu < 1024)` matches runtime Tier-1 dispatch.
- **Root-Cause Calibration Skew Discovery:** Across `88,754` authentic prefill tokens, **`overall_hot_hit_rate = 29.47%` (`0.29469`)** — barely above the `25.0%` (`96/384`) uniform-random floor and far short of the `~58%` target (`overall_cold_frac_mean = 0.69598`, i.e., **~70.5% of routed experts hit 2-bit cold CPU/holographic slots** instead of Top-96 native 4-bit GPU slots). Inspection of per-layer hit counts shows severe calibration skew from the synthetic Gaussian prior ranking ([`dsv41_chronos_moe.py`](file:///c:/Users/Eating/Desktop/Chronos%201T/chronos_1t/models/dsv41_chronos_moe.py#L3186-L3212)): e.g., in Layer 0 hot expert `79` received `2` hits while cold expert `206` received `11,627` hits; in Layer 1 hot experts `58, 74, 351, 383` received `0` hits.

### 5.1 Pass Rate by Prefill Cold-Routing Tertile (`A1`, `n = 600`)

| Cohort | Low-Cold `T1` (`n`) | Mid-Cold `T2` (`n`) | High-Cold `T3` (`n`) | `T1 -> T3` Drop | Spearman `rho` (`p`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Overall (`n = 600`)** | **`90.5%`** (`181/200`, `cold=0.6761`) | **`84.0%`** (`168/200`, `cold=0.6928`) | **`76.0%`** (`152/200`, `cold=0.7190`) | **`-14.5 pp`** | `-0.1965` (`p = 1e-6`) |
| **MMLU (`n = 300`)** | **`88.0%`** (`88/100`, `cold=0.6813`) | **`79.0%`** (`79/100`, `cold=0.6963`) | **`76.0%`** (`76/100`, `cold=0.7209`) | **`-12.0 pp`** | `-0.1534` (`p = 0.007775`) |
| **MMLU-Pro (`n = 150`)** | **`92.0%`** (`46/50`, `cold=0.6789`) | **`78.0%`** (`39/50`, `cold=0.6992`) | **`68.0%`** (`34/50`, `cold=0.7306`) | **`-24.0 pp`** | `-0.2827` (`p = 0.000456`) |
| **GSM8K (`n = 150`)** | `90.0%` (`45/50`, `cold=0.6690`) | `92.0%` (`46/50`, `cold=0.6818`) | `96.0%` (`48/50`, `cold=0.6953`) | `+6.0 pp` | `+0.1078` (`p = 0.189215`, n.s.) |

- **Live Promotion Untestability:** Promoting the top-hit cold experts to native 4-bit on GPU cannot be tested live in-session because only `2.506 GiB` VRAM is free and raw 4-bit weights for the 288 cold experts are not loaded in memory. Re-selecting Top-96 using real multi-domain calibration routing histograms (instead of isotropic Gaussian probes) is recommended for the next offline checkpoint rebuild.

---

## 6. Long-Context (`48K`), Prefix KV Caching & Speculative Decoding Smoke Results

Verified on the resident model in [`dsv41_longctx_smoke_report.json`](file:///c:/Users/Eating/Desktop/Chronos%201T/results/dsv41_longctx_smoke_report.json):
- **48K MLA Window (`max_seq_len = 49152`):** Zero-leak upgrade preserving `2.506 / 94.971 GiB` free VRAM after `12K`-token prefill+decode (`prefill_logits_finite = True`, `decode_at_12000_finite = True`). Real-prompt chunked (`300+250`) vs full (`550`) parity achieved `logits_cos = 0.987382` with exact `top1` match (`361 == 361`).
- **`PrefixKVCacheSession` (`12K` Multi-Turn):** Turn 1 prefilled `11,500` tokens in `67.884 s`; Turn 2 reused `11,492` cached prefix tokens and prefilled `508` new tokens in `4.209 s` (**`16.13x` prefill speedup**).
- **Greedy Lossless `M=1` vs PLD `M=7` Speculative Decoding:** `use_speculative=False` (`M=1` CUDAGraph decode) runs deterministically at **`14.17 tok/s`** (`128/128` exact repeatability across repeated runs, and `14.60 tok/s` in `task-2360`). Enabling PLD `M=7` verification (`spec_k=6`, `82.67%` draft acceptance) ran **slower (`12.31 tok/s`)** and diverged at `95/128` because batching `M=7` tokens fans out ~`4.5x` more unique cold CPU experts across PCIe per step and cuBLAS `M`-tiling shifts activations by ~`1 ULP` across FP4 quantization boundaries. Production multi-turn inference therefore keeps `use_speculative=False`.

---

## 7. Ranking of Causes & Untestable Items

Across our 600-question controlled ablations, the gap against Qwen Flash Next on MMLU-Pro (`63.33% -> 82.00%`) and GSM8K (`83.33% -> 95.33% / 96.67%`) and against stock DeepSeek-V4.1-Flash on MMLU (`87.96%` vs our `82.67%`) is explained by four ranked mechanisms: **(1) Aggressive `<think>` 4-gram repetition penalties and `should_close_early` triggers** were the single largest decoding bug, whose removal alone (`A2`) raised MMLU-Pro by `+6.66 pp` (`74.67% -> 81.33%`), GSM8K by `+3.33 pp` (`92.00% -> 95.33%`), and overall accuracy significantly (`+2.50 pp`, McNemar `p = 0.0444`); **(2) The `384`-token `<think>` ceiling and earlier tight `max_think_tokens` caps** prematurely truncated complex reasoning (`40.0%` forced `</think>` on MMLU-Pro in `A0`), so unleashing thinking budgets to `ceil=3072` (`A_unleashed`) slashed forced closures from `14.67%` to `3.67%`, propelled MMLU-Pro to `82.00%` (`+7.33 pp`, `p = 0.0192`, beating Qwen Flash Next's `81.33%`), and nudged MMLU to `82.67%` (`+1.00 pp`); **(3) 2-bit cold-tier expert compression compounded by synthetic Top-96 calibration skew** is the primary remaining bottleneck behind the `-5.29 pp` MMLU deficit vs stock `87.96%`, as prefill routing hits native 4-bit hot experts only `29.47%` of the time (hardly above `25%` chance) and high-cold questions suffer a significant `-14.5 pp` overall (`-24.0 pp` MMLU-Pro, `-12.0 pp` MMLU) accuracy drop vs low-cold questions; and **(4) GSM8K parser/prompt artifacts** accounted for `+1.33` to `+2.67 pp` (`\mathbf{}`/`\%` formatting and `<number>` echo, reaching `96.67%` on clean rescoring of `A2`). Two factors could not be ablated live in-session: **`A3` refusal orthogonalization** (permanently baked offline into weights with `CHRONOS_RUNTIME_R_BASIS="0"` and no stored reversal deltas) and **live 4-bit promotion of hot cold experts in `A4`** (blocked by `2.506 GiB` free VRAM and the absence of raw 4-bit cold expert weights in runtime memory).
