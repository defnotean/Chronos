---
license: other
license_name: glm-5.3
base_model: zai-org/GLM-5.3
library_name: transformers
pipeline_tag: text-generation
tags:
  - chronos
  - holographic-moe
  - glm_moe_dsa
  - dgx-spark
  - colab
---

# PseudoBrain GLM-5.3 Chronos v7.5 (753B, `GlmMoeDsaForCausalLM`)

Chronos v7.5 holographic conversion of [`zai-org/GLM-5.3`](https://huggingface.co/zai-org/GLM-5.3), meant to run on
**one NVIDIA DGX Spark (GB10, 128 GB unified)** through vLLM and on **one Colab GPU (A100/H100 80 GB, G4 96 GB)**
through the standalone Chronos engine. It uses the same tooling as the GLM-5.3-Flash and DeepSeek-V4.1 Chronos
releases.

> [!IMPORTANT]
> GLM-5.3 is a 753B model (`19,456` routed experts). Fitting it on a single 80–128 GB device means roughly
> **1.1 bits per expert weight on average**, which is lossy. The residency planner reports what each device holds,
> and `benchmarks/glm53_real_layer_moe_fidelity.py` measures the per-layer cost on the real weights. Evaluate on your
> own workload before you rely on it.

## Files

| File | Contents |
| :--- | :--- |
| `chronos-glm53-base-embed-head-norm.safetensors` | `embed_tokens`, `lm_head`, final norm (BF16, unchanged) |
| `chronos-glm53-layer-00000..00002-of-00079.safetensors` | dense layers: MLA + DSA indexer + dense MLP, stock 128×128 block-FP8 |
| `chronos-glm53-layer-00003..00077-of-00079.safetensors` | MoE layers: attention/indexer/router/shared expert (stock FP8) + 32 `chronos_groups.*` |
| `chronos-glm53-layer-00078-of-00079.safetensors` | MTP (nextn) layer, same format |
| `chronos-glm53-hot{N}-layer-*.safetensors` (optional) | Upgrade-2 sidecar: top-N experts/layer packed from FP8 directly into runtime T1 |
| `config.json` | stock config + `chronos_holographic_moe` (`arch: glm53`, tail bits, `key_generator: torch_cpu_v1`) |

## Format

Each group of 8 routed experts (`gate_up 4096×6144`, `down 6144×2048`) is stored as:

1. an **8:1 NVFP4 holographic bank** (permuted-Hadamard and Rademacher binding, weighted by router importance)
2. a **rank-64 FP8 SMW SVD healer** per expert, fitted on the router's principal subspace
3. a **2-bit Lloyd-Max tail** (standard) or **1-bit Rademacher tail** (`--compact`)
4. **Woodbury SwiGLU drift compensation** folded into `down`, plus post-tail variance restoration

| | Stock FP8 | Chronos standard (2-bit tails) | Chronos compact (1-bit tails) |
| :--- | ---: | ---: | ---: |
| Routed experts | 734.6 GB | 281.2 GB | 177.9 GB |
| Whole checkpoint | 755.6 GB | ~302 GB | ~199 GB |

The permutation and sign keys come from a CPU `torch.Generator`. That makes a checkpoint converted on an A100
reconstruct **bit-identically** on a GB10, an H100 or a CPU (verified: max |Δ| = 0 between CPU and CUDA
reconstruction of real layer-3 experts).

## Runtime tiers

At load time every resident expert is rebuilt from its group and packed into one of three GPU tiers. Each tier has
a portable Triton decode kernel (integer-only decoding, so it runs on sm_80, sm_90, sm_120 and sm_121):

| Tier | Format | Bytes / expert |
| :--- | :--- | ---: |
| T1 | E2M1 (NVFP4 values), group 32, fp16 scale (4.5 bpw) | 21.2 MB |
| T2 | 2-bit Lloyd-Max, group 64, fp16 scale (2.25 bpw) | 10.6 MB |
| T3 | sign + absmean, group 128, fp16 scale (1.125 bpw) | 5.3 MB |

Default residency plans from `python -m chronos_1t.models.glm53_chronos_spec` (32K context):

| Device | GPU tiers / layer | Pinned host RAM | Expert coverage |
| :--- | :--- | :--- | ---: |
| DGX Spark GB10 (108 GB container) | 193 × T3 | – (unified) | 75.4% |
| Colab A100-80GB High-RAM | 116 × T3 | 140 × T3 (55.7 GB) | 100% |
| Colab H100-80GB | 116 × T3 | 140 × T3 | 100% |
| Colab G4 (RTX PRO 6000, 96 GB) | 155 × T3 | 101 × T3 | 100% |

Experts that are not resident are masked out of the router, and the top-8 weights are renormalized over the
resident experts (the DeepSeek-V4.1 Chronos contract). You can override the plan with `CHRONOS_GLM53_T{1,2,3}_SLOTS`
and `CHRONOS_GLM53_T3_HOST_SLOTS`. Tier ranking uses a router Monte-Carlo prior by default; run once with
`--calibrate` to replace it with real routing counts (`chronos_glm53_expert_stats.json`).

## Measured fidelity on real GLM-5.3 weights

Each cell is the per-expert SwiGLU output cosine against stock FP8, on router-manifold probe activations, for one
group of 8 experts:

| Layer | Chronos (bank+healer+tail) | T1 from Chronos / from FP8 | T2 from Chronos / from FP8 | T3 from Chronos / from FP8 | Bank+healer only |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 3 | 0.954 | 0.949 / 0.995 | 0.867 / 0.889 | 0.656 / 0.661 | 0.559 |
| 40 | 0.912 | 0.903 / 0.989 | 0.807 / 0.865 | 0.535 / 0.565 | 0.268 |

Two takeaways. First, without its tail the bank+healer reconstruction is not usable as a tier. Second, hot experts
should come straight from FP8 (`--hot-experts N`) instead of being re-packed from the holographic group.

## DGX Spark (vLLM v0.31, pb27-vllm)

```bash
huggingface-cli download defnoteatin/PseudoBrain-GLM-5.3-Chronos-v7.5 --local-dir /home/defnotean/pb27/hf/glm53_chronos_v75
bash scripts/start_glm53_vllm.sh glm-5.3 8001 --wait   # refuses to start while another pb27-vllm container runs
```

`chronos_1t.vllm_plugin` installs `patch_glm53_vllm`, which:

- swaps `DeepseekV2MoE` in `vllm.models.deepseek_v32` for `ChronosGlm53SparseMoeBlock`, falling back to the stock
  class for any non-Chronos model
- runs the block through an opaque `torch.library` op, so `torch.compile` and PIECEWISE CUDA graphs work
- keeps vLLM's loader from reading MoE tensors in `chronos-glm53-*` shards

Built tiers are cached in `CHRONOS_GLM53_TIER_CACHE`, so restarts skip reconstruction.

## Colab (standalone engine)

The DSA sparse-MLA backends in vLLM need sm_90 or newer plus DeepGEMM, so Colab A100s use
`chronos_1t/models/glm53_engine.py`. It reproduces HF `transformers` `GlmMoeDsaForCausalLM`: logit cosine is
1.000000 against the reference on CPU, both for full prefill and for token-by-token decode with the DSA top-k
active. It uses an absorbed-MLA latent KV cache (93 KB/token) and stock FP8 attention.

```bash
python scripts/run_glm53_chronos_colab.py --repo defnoteatin/PseudoBrain-GLM-5.3-Chronos-v7.5 \
    --model-dir /content/glm53_chronos --stream-delete --benchmark
python scripts/run_glm53_chronos_colab.py --model-dir /content/glm53_chronos --serve --port 8000   # OpenAI API
```

There is also a notebook: `notebooks/GLM53_Chronos_Colab.ipynb`.

## Reproducing the conversion

```bash
python chronos_1t/converter/convert_glm53_chronos.py \
    --out-dir /content/glm53_chronos_v75 --cache-dir /content/hf_cache_glm53 \
    --upload-repo defnoteatin/PseudoBrain-GLM-5.3-Chronos-v7.5 --delete-after-upload [--compact] [--hot-experts 32]
```

The conversion is resumable. Source shards are processed in shard order, so at most about 3 shards (~16 GB) sit on
disk at once. Uploads run in the background with back-pressure.

## Credits

Base model: Z.ai GLM-5.3 (GLM-5.3 license). Chronos: Ian Kurt Galeon ([@defnotean](https://github.com/defnotean)).
