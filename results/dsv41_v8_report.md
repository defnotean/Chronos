# Chronos v8 Calibrated 3-Tier Intelligence Benchmark Report (`DeepSeek-V4.1-Flash-1017`)

**Model:** Chronos v8 Calibrated 3-Tier (`T1 = 84` GPU 4-bit / `T2 = 136` CPU pinned MXFP4 / `T3 = 164` CPU pinned 2-bit per layer, 40 layers × 384 experts)  
**Source Checkpoint:** Official unmodified `deepseek-ai/DeepSeek-V4.1-Flash-1017` (`r_basis = None`, `abliterated = false`)  
**Hardware / Session:** Google Colab G4 (`94.97 GiB` VRAM, `176.88 GiB` host RAM; `40/40` CUDAGraphs captured, `15.02 GiB` VRAM free after materialization)  
**Evaluation Protocol (Strict 1:1 `A_unleashed`):** Identical 600-question stratified suite (`MMLU` `n=300`, `MMLU-Pro` `n=150`, `GSM8K` `n=150`) from `ablate_fixed_600_questions.json`; `ablate_guards = True`, `think_ceiling = 3072`, `kv_limit = 8192`, `max_new_tokens = 2048` (`MMLU`) / `4096` (`MMLU-Pro`, `GSM8K`), greedy (`temperature = 0.0`), `log_routing = True`. Calibration-train split (`480` items, `idx % 5 != 0`) vs strictly out-of-sample validation split (`Val-120`, `idx % 5 == 0`) are reported separately.

---

## 1. Headline Results (Strict 1:1 Unmodified Harness)

| Benchmark / Split | Chronos v8 (`n=600`) | 95% Wilson CI | v7.5 `A_unleashed` | v7.5 `A0` (Baseline) | Qwen Flash Next Ref. | Stock DSV4.1 Ref. |
|---|---:|---:|---:|---:|---:|---:|
| **All-600 (Micro)** | **89.33%** (`536/600`) | `[86.61%, 91.56%]` | 85.67% (`514/600`) | 82.50% (`495/600`) | — | — |
| **All-600 (Macro)** | **90.11%** | — | 86.67% | 82.78% | — | — |
| **GSM8K (`n=150`)** | **98.00%** (`147/150`) | `[94.29%, 99.32%]` | 95.33% (`143/150`) | 92.00% (`138/150`) | 96.00% | — |
| **MMLU-Pro (`n=150`)** | **85.33%** (`128/150`) | `[78.79%, 90.11%]` | 82.00% (`123/150`) | 74.67% (`112/150`) | 81.33% | — |
| **MMLU (`n=300`)** | **87.00%** (`261/300`) | `[82.72%, 90.34%]` | 82.67% (`248/300`) | 81.67% (`245/300`) | — | 87.96% |

> **All-600 Summary:** Chronos v8 outperforms v7.5 `A_unleashed` on every benchmark (`+2.67 pp` GSM8K, `+3.33 pp` MMLU-Pro, `+4.33 pp` MMLU, `+3.67 pp` overall micro) and surpasses the Qwen Flash Next reference on both GSM8K (`98.00%` vs `96.00%`) and MMLU-Pro (`85.33%` vs `81.33%`).

---

## 2. Out-of-Sample Generalization (`Val-120`, `idx % 5 == 0` vs `Cal-480`)

| Split | Overall Micro | 95% Wilson CI | GSM8K | MMLU-Pro | MMLU |
|---|---:|---:|---:|---:|---:|
| **Held-out `Val-120` (`n=120`)** | **90.00%** (`108/120`) | `[83.33%, 94.19%]` | **100.00%** (`30/30`, `[88.65%, 100.0%]`) | **83.33%** (`25/30`, `[66.44%, 92.66%]`) | **88.33%** (`53/60`, `[77.82%, 94.23%]`) |
| **Calib-Train `Cal-480` (`n=480`)** | **89.17%** (`428/480`) | `[86.07%, 91.64%]` | **97.50%** (`117/120`, `[92.91%, 99.15%]`) | **85.83%** (`103/120`, `[78.48%, 90.96%]`) | **86.67%** (`208/240`, `[81.78%, 90.39%]`) |

Held-out `Val-120` accuracy (`90.00%`) matches and slightly exceeds `Cal-480` (`89.17%`), confirming that Chronos v8 expert tier calibration generalizes cleanly out-of-sample with zero memorization artifacts.

---

## 3. Exact Paired McNemar Tests (Chronos v8 vs v7.5 Arms)

Two-sided exact binomial McNemar tests on discordant pairs (`b` = v8 passes & baseline fails; `c` = baseline passes & v8 fails):

| Comparison | Subset | `n` | v8 Pass | Base Pass | Delta | `both_pass` | `v8_only (b)` | `base_only (c)` | `both_fail` | Exact 2-sided `p` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **v8 vs `A_unleashed`** | **All-600** | 600 | 536 (89.33%) | 514 (85.67%) | **+3.67 pp** | 502 | **34** | 12 | 52 | **0.00164** |
| **v8 vs `A_unleashed`** | GSM8K | 150 | 147 (98.00%) | 143 (95.33%) | +2.67 pp | 142 | 5 | 1 | 2 | 0.21875 |
| **v8 vs `A_unleashed`** | MMLU-Pro | 150 | 128 (85.33%) | 123 (82.00%) | +3.33 pp | 122 | 6 | 1 | 21 | 0.12500 |
| **v8 vs `A_unleashed`** | MMLU | 300 | 261 (87.00%) | 248 (82.67%) | **+4.33 pp** | 238 | **23** | 10 | 29 | **0.03508** |
| **v8 vs `A_unleashed`** | **Val-120** | 120 | 108 (90.00%) | 105 (87.50%) | +2.50 pp | 104 | 4 | 1 | 11 | 0.37500 |
| **v8 vs `A_unleashed`** | Cal-480 | 480 | 428 (89.17%) | 409 (85.21%) | **+3.96 pp** | 398 | **30** | 11 | 41 | **0.00432** |
| **v8 vs `A0`** | **All-600** | 600 | 536 (89.33%) | 495 (82.50%) | **+6.83 pp** | 471 | **65** | 24 | 40 | **1.61e-5** |
| **v8 vs `A0`** | GSM8K | 150 | 147 (98.00%) | 138 (92.00%) | **+6.00 pp** | 138 | **9** | 0 | 3 | **0.00391** |
| **v8 vs `A0`** | MMLU-Pro | 150 | 128 (85.33%) | 112 (74.67%) | **+10.67 pp** | 109 | **19** | 3 | 19 | **0.00086** |
| **v8 vs `A0`** | MMLU | 300 | 261 (87.00%) | 245 (81.67%) | **+5.33 pp** | 224 | **37** | 21 | 18 | **0.04794** |
| **v8 vs `A0`** | **Val-120** | 120 | 108 (90.00%) | 103 (85.83%) | +4.17 pp | 98 | 10 | 5 | 7 | 0.30176 |
| **v8 vs `A0`** | Cal-480 | 480 | 428 (89.17%) | 392 (81.67%) | **+7.50 pp** | 373 | **55** | 19 | 33 | **3.38e-5** |

---

## 4. Live 3-Tier Expert Routing, Numerics, and Throughput

Chronos v8 places `84` experts on GPU 4-bit (`T1`), `136` experts in pinned CPU MXFP4 (`T2`), and `164` experts in pinned CPU 2-bit (`T3`) per layer (`220/384` active 4-bit weights available without cold-quantization loss).

| Suite / Split | Prefill T1 / T2 / T3 | Prefill 4-bit | Decode T1 / T2 / T3 | Decode 4-bit | All 4-bit | Mean Tok (Think) | Forced Think | Mean `tok/s` (Weighted) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **All-600** | `55.24% / 33.72% / 11.04%` | **88.96%** | `41.04% / 38.74% / 20.21%` | **79.79%** | **82.69%** | 321.56 (251.09) | 2.67% (`16/600`) | 11.46 (`10.88`) |
| **Val-120** | `55.31% / 33.40% / 11.29%` | **88.71%** | `41.32% / 38.41% / 20.27%` | **79.73%** | **82.91%** | 264.57 (198.75) | 1.67% (`2/120`) | 11.48 (`10.94`) |
| **GSM8K (`n=150`)** | `59.27% / 30.94% / 9.79%` | **90.21%** | `44.09% / 37.57% / 18.35%` | **81.65%** | **83.93%** | 291.97 (205.20) | 0.00% (`0/150`) | 11.69 (`11.34`) |
| **MMLU-Pro (`n=150`)** | `53.54% / 34.86% / 11.60%` | **88.40%** | `37.43% / 40.84% / 21.73%` | **78.27%** | **81.34%** | 495.01 (446.51) | 6.67% (`10/150`) | 10.84 (`10.30`) |
| **MMLU (`n=300`)** | `55.03% / 33.90% / 11.07%` | **88.93%** | `42.84% / 37.36% / 19.80%` | **80.20%** | **83.29%** | 249.64 (176.31) | 2.00% (`6/300`) | 11.66 (`11.23`) |

- **Numerical Health:** `600/600` (`100%`) items had `logits_finite == True`, `0/600` had unclosed `<think>` blocks (`think_unclosed_count == 0`), and `0/600` multi-token generations had empty decode routing (`decode_nonzero_zero_count == 0`).
- **vs v7.5 Routing:** Pre-calibration synthetic Tier-1 in v7.5 only captured `29.47%` real 4-bit routing hits (forcing `70.53%` onto 2-bit cold slots); Chronos v8 captures **82.69% overall (88.96% prefill / 79.79% decode)** in native 4-bit, directly explaining the `+3.67 pp` accuracy recovery (`p = 0.00164`).

---

## 5. Error Analysis & MMLU Parser Audit (Secondary Findings)

Across the `64/600` misses under the unmodified 1:1 harness:
- **GSM8K (`3/150` misses; `idx=21, 89, 112`):** `0/3` were `think_forced`; all 3 were genuine math reasoning errors (`pred='100'` vs `'25'`, `pred='27'` vs `'54'`, `pred='168'` vs `'342'`).
- **MMLU-Pro (`22/150` misses):** `5/22` hit the `3072`-token `think_ceiling` (`think_forced=True`), and `17/22` closed thinking naturally.
- **MMLU (`39/300` misses; Parser Artifact Audit):** While `2/39` hit `think_forced`, inspection of the logged `answer_preview` strings showed frequent prompt-echoing and option-text outputs (e.g., `"C) an increase in the electrical current..."` or `"A\nA) Round-robin... D) Youngest job first\nReply ONLY..."`). The legacy `extract_mc_letter` parser uses case-insensitive `last_match`, picking lowercase option letters inside ordinary words (`"a"`) or echoed option `D)` instead of the leading answer token.

Applying a **symmetric case-sensitive leading-option extractor** (`^(?:ANSWER\s*[:\-]\s*)?\(?([A-D])(?:\)|\b)`) identically to both models' logged `answer_preview` outputs on MMLU yields:

| Arm / Subset | Legacy 1:1 Harness | False Negatives Fixed | False Positives Flipped | Audited MMLU Capability |
|---|---:|---:|---:|---:|
| **Chronos v8 (`n=300`)** | 87.00% (`261/300`) | 20 | 0 | **93.67%** (`281/300`) |
| **Chronos v8 (`Val-60`)** | 88.33% (`53/60`) | 5 | 0 | **96.67%** (`58/60`) |
| **v7.5 `A_unleashed` (`n=300`)** | 82.67% (`248/300`) | 28 | 1 | **91.67%** (`275/300`) |
| **v7.5 `A_unleashed` (`Val-60`)** | 83.33% (`50/60`) | 7 | 0 | **95.00%** (`57/60`) |

*(All headline tables in Sections 1–3 strictly retain the unmodified 1:1 harness scores for apples-to-apples comparability.)*

---

## 6. Saved Artifacts
- Raw per-item progress: [v8_eval_progress.jsonl](file:///c:/Users/Eating/Desktop/Chronos%201T/results/v8_eval_progress.jsonl)
- Aggregated JSON report: [dsv41_v8_intelligence_report.json](file:///c:/Users/Eating/Desktop/Chronos%201T/results/dsv41_v8_intelligence_report.json)
- Baseline comparisons: [ablate_A_unleashed_progress.jsonl](file:///c:/Users/Eating/Desktop/Chronos%201T/results/ablate_A_unleashed_progress.jsonl), [ablate_A0_progress.jsonl](file:///c:/Users/Eating/Desktop/Chronos%201T/results/ablate_A0_progress.jsonl)
