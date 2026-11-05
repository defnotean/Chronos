#!/usr/bin/env python3
"""
Chronos-1T x GLM-5.3 (`zai-org/GLM-5.3`, `GlmMoeDsaForCausalLM`) — Architecture Spec, Shard Naming & Residency Planner.

Single source of truth shared by the converter (`convert_glm53_chronos.py`), the runtime MoE block
(`glm53_chronos_moe.py`), the Spark vLLM hook (`patch_glm53_vllm.py`) and the Colab engine (`glm53_engine.py`).

GLM-5.3 (full, NOT the Flash variant) facts, read from `zai-org/GLM-5.3/config.json`:
  - `753.3B` params, `755.6 GB` stock 128x128 block-FP8 (`141` shards)
  - `78` decoder layers + `1` MTP/nextn layer (`model.layers.78`)
  - `hidden_size = 6144`, dense `intermediate_size = 12288` (layers `0..2`)
  - `moe_intermediate_size = 2048`, `256` routed experts + `1` shared expert (layers `3..77` and MTP `78`)
  - sigmoid `noaux_tc` top-8 router, `n_group = topk_group = 1`, `norm_topk_prob`, `routed_scaling_factor = 2.5`
  - MLA attention (`q_lora 2048`, `kv_lora 512`, `64` heads, `qk_nope 192`, `qk_rope 64`, `v_head 256`)
    + DSA lightning indexer (`32` heads x `128`, `index_topk = 2048`, `full`/`shared` cross-layer top-k reuse)

Compared with GLM-5.3-Flash (`Glm5NextForConditionalGeneration`, `4096` hidden, `288` experts, `46` layers),
every expert matrix is `1.5x` wider (`gate_up 4096 x 6144`, `down 6144 x 2048`), there are `32` holographic
groups of `K = 8` per layer instead of `36`, and there are `76` MoE layers instead of `43` (`19,456` experts).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
from typing import Any

GLM53_REPO_ID = "zai-org/GLM-5.3"
GLM53_CHRONOS_REPO_ID = "defnoteatin/PseudoBrain-GLM-5.3-Chronos-v7.5"
CHRONOS_VERSION = "7.5"
GROUP_SIZE = 8

# Runtime tier formats (bits per weight include the per-group fp16 scale)
TIER_FORMATS: dict[str, dict[str, Any]] = {
    "t1_fp4": {"bits": 4, "group": 32, "scale_bytes": 2},   # E2M1 codes, fp16 scale / 32  -> 4.500 bpw
    "t2_lm2": {"bits": 2, "group": 64, "scale_bytes": 2},   # Lloyd-Max 4-level, fp16 / 64 -> 2.250 bpw
    "t3_sign1": {"bits": 1, "group": 128, "scale_bytes": 2},  # sign, fp16 absmean / 128     -> 1.125 bpw
}


@dataclass(frozen=True)
class Glm53Arch:
    hidden_size: int = 6144
    intermediate_size: int = 12288
    moe_intermediate_size: int = 2048
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    num_hidden_layers: int = 78
    num_nextn_predict_layers: int = 1
    first_k_dense_replace: int = 3
    routed_scaling_factor: float = 2.5
    norm_topk_prob: bool = True
    n_group: int = 1
    topk_group: int = 1
    vocab_size: int = 154880
    num_attention_heads: int = 64
    q_lora_rank: int = 2048
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    v_head_dim: int = 256
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 2048
    rms_norm_eps: float = 1e-5
    rope_theta: float = 8_000_000.0
    indexer_types: tuple[str, ...] = field(default_factory=tuple)
    eos_token_ids: tuple[int, ...] = (154820, 154827, 154829)

    # ---------- derived ----------
    @property
    def num_total_layers(self) -> int:
        return self.num_hidden_layers + self.num_nextn_predict_layers

    @property
    def mtp_layer_ids(self) -> list[int]:
        return list(range(self.num_hidden_layers, self.num_total_layers))

    @property
    def moe_layer_ids(self) -> list[int]:
        """All sparse layers including the MTP layer(s) (`3..78` = `76` layers for GLM-5.3)."""
        return list(range(self.first_k_dense_replace, self.num_total_layers))

    @property
    def n_groups(self) -> int:
        assert self.n_routed_experts % GROUP_SIZE == 0
        return self.n_routed_experts // GROUP_SIZE

    @property
    def gu_rows(self) -> int:
        return 2 * self.moe_intermediate_size

    @property
    def expert_params(self) -> int:
        return 3 * self.hidden_size * self.moe_intermediate_size

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    def is_full_indexer_layer(self, layer_idx: int) -> bool:
        if not self.indexer_types or layer_idx >= len(self.indexer_types):
            return True
        return self.indexer_types[layer_idx] == "full"

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "Glm53Arch":
        text = cfg.get("text_config", cfg)
        rope = text.get("rope_parameters") or {}
        eos = text.get("eos_token_id", list(cls.eos_token_ids))
        if isinstance(eos, int):
            eos = [eos]
        arch = cls(
            hidden_size=int(text["hidden_size"]),
            intermediate_size=int(text["intermediate_size"]),
            moe_intermediate_size=int(text["moe_intermediate_size"]),
            n_routed_experts=int(text["n_routed_experts"]),
            n_shared_experts=int(text.get("n_shared_experts", 1) or 0),
            num_experts_per_tok=int(text["num_experts_per_tok"]),
            num_hidden_layers=int(text["num_hidden_layers"]),
            num_nextn_predict_layers=int(text.get("num_nextn_predict_layers", 0) or 0),
            first_k_dense_replace=int(text.get("first_k_dense_replace", 0)),
            routed_scaling_factor=float(text.get("routed_scaling_factor", 1.0)),
            norm_topk_prob=bool(text.get("norm_topk_prob", True)),
            n_group=int(text.get("n_group", 1) or 1),
            topk_group=int(text.get("topk_group", 1) or 1),
            vocab_size=int(text["vocab_size"]),
            num_attention_heads=int(text["num_attention_heads"]),
            q_lora_rank=int(text.get("q_lora_rank") or 0),
            kv_lora_rank=int(text["kv_lora_rank"]),
            qk_nope_head_dim=int(text["qk_nope_head_dim"]),
            qk_rope_head_dim=int(text["qk_rope_head_dim"]),
            v_head_dim=int(text["v_head_dim"]),
            index_n_heads=int(text.get("index_n_heads", 32)),
            index_head_dim=int(text.get("index_head_dim", 128)),
            index_topk=int(text.get("index_topk", 2048)),
            rms_norm_eps=float(text.get("rms_norm_eps", 1e-5)),
            rope_theta=float(rope.get("rope_theta", text.get("rope_theta", 8_000_000.0))),
            indexer_types=tuple(text.get("indexer_types") or ()),
            eos_token_ids=tuple(int(e) for e in eos),
        )
        if text.get("scoring_func", "sigmoid") != "sigmoid" or text.get("topk_method", "noaux_tc") != "noaux_tc":
            raise ValueError("Chronos GLM-5.3 runtime only implements the sigmoid noaux_tc router.")
        if text.get("hidden_act", "silu") != "silu":
            raise ValueError("Chronos GLM-5.3 runtime only implements SwiGLU (silu) experts.")
        for dim in (arch.hidden_size, arch.moe_intermediate_size):
            if dim % 128 != 0:
                raise ValueError(f"Holographic folding needs dims divisible by 128 (got {dim}).")
        return arch

    @classmethod
    def from_json_file(cls, path: str | os.PathLike) -> "Glm53Arch":
        return cls.from_config(json.loads(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------------------------
# Shard naming (mirrors `chronos-dsv41-layer-XXXXX-of-00040` / `chronos-glm5-layer-XXXXX-of-00046`)
# ---------------------------------------------------------------------------------------------
def layer_shard_name(layer_idx: int, arch: Glm53Arch | None = None) -> str:
    total = (arch or Glm53Arch()).num_total_layers
    return f"chronos-glm53-layer-{layer_idx:05d}-of-{total:05d}.safetensors"


def base_shard_name() -> str:
    return "chronos-glm53-base-embed-head-norm.safetensors"


def hot_shard_name(layer_idx: int, n_hot: int, arch: Glm53Arch | None = None) -> str:
    """Upgrade-2 style sidecar: the `n_hot` highest-prior experts of a layer packed directly from the stock FP8
    weights into the runtime `T1` format (func cos 0.995 vs 0.949 when re-packed from the holographic group)."""
    total = (arch or Glm53Arch()).num_total_layers
    return f"chronos-glm53-hot{n_hot}-layer-{layer_idx:05d}-of-{total:05d}.safetensors"


def is_hot_sidecar(name: str) -> bool:
    return os.path.basename(name).startswith("chronos-glm53-hot")


def is_glm53_chronos_file(name: str) -> bool:
    return os.path.basename(name).startswith("chronos-glm53-")


def moe_prefix(layer_idx: int) -> str:
    return f"model.layers.{layer_idx}.mlp"


def group_prefix(layer_idx: int, g_idx: int) -> str:
    return f"{moe_prefix(layer_idx)}.chronos_groups.{g_idx}."


def group_seed_bases(g_idx: int) -> tuple[int, int]:
    """Deterministic Rademacher/permutation seeds — identical convention to GLM-5.3-Flash and DSV4.1."""
    return 101 + g_idx * GROUP_SIZE, 303 + g_idx * GROUP_SIZE


# ---------------------------------------------------------------------------------------------
# Byte accounting
# ---------------------------------------------------------------------------------------------
def chronos_group_bytes(arch: Glm53Arch, svd_rank: int = 64, gu_bits: int = 2, down_bits: int = 2) -> dict[str, int]:
    """On-disk bytes of one Chronos v7.5 holographic group (K=8)."""
    h, i2, i = arch.hidden_size, arch.gu_rows, arch.moe_intermediate_size
    k = GROUP_SIZE
    bank_gu = i2 * h // 2 + i2 * h // 16
    bank_dn = h * i // 2 + h * i // 16
    healer = k * svd_rank * (i2 + h) + k * svd_rank * (h + i)

    def _tail(rows: int, cols: int, bits: int) -> int:
        if bits == 2:
            return k * (rows * cols // 4 + rows * cols // 32)
        return k * (rows * cols // 8 + rows * cols // 64)

    tail = _tail(i2, h, gu_bits) + _tail(h, i, down_bits)
    return {"bank": bank_gu + bank_dn, "healer": healer, "tail": tail, "total": bank_gu + bank_dn + healer + tail}


def tier_expert_bytes(arch: Glm53Arch, tier: str) -> int:
    fmt = TIER_FORMATS[tier]
    n = arch.expert_params
    return n * fmt["bits"] // 8 + (n // fmt["group"]) * fmt["scale_bytes"]


def fp8_expert_bytes(arch: Glm53Arch) -> int:
    n = arch.expert_params
    return n + 4 * (n // (128 * 128))


def non_expert_bytes_estimate(arch: Glm53Arch) -> dict[str, int]:
    """Stock bytes of everything that is not a routed expert (attention, dense MLP, shared experts, embeddings)."""
    h = arch.hidden_size
    nh = arch.num_attention_heads
    attn = (
        arch.q_lora_rank * h
        + nh * arch.qk_head_dim * arch.q_lora_rank
        + (arch.kv_lora_rank + arch.qk_rope_head_dim) * h
        + nh * (arch.qk_nope_head_dim + arch.v_head_dim) * arch.kv_lora_rank
        + h * nh * arch.v_head_dim
    )
    n_full = sum(1 for l in range(arch.num_total_layers) if arch.is_full_indexer_layer(l))
    indexer = n_full * (arch.index_n_heads * arch.index_head_dim * arch.q_lora_rank + arch.index_head_dim * h)
    attn_total = attn * arch.num_total_layers + indexer
    dense = arch.first_k_dense_replace * 3 * h * arch.intermediate_size
    shared = len(arch.moe_layer_ids) * 3 * h * arch.moe_intermediate_size * arch.n_shared_experts
    embed_head_bf16 = 2 * arch.vocab_size * h * 2
    mtp_extra = arch.num_nextn_predict_layers * (2 * h * h) * 2
    router = len(arch.moe_layer_ids) * arch.n_routed_experts * h * 2
    return {
        "attention_fp8": attn_total,
        "dense_mlp_fp8": dense,
        "shared_experts_fp8": shared,
        "embed_lm_head_bf16": embed_head_bf16,
        "mtp_eh_proj_bf16": mtp_extra,
        "routers_bf16": router,
        "total": attn_total + dense + shared + embed_head_bf16 + mtp_extra + router,
    }


def kv_bytes_per_token(arch: Glm53Arch, kv_dtype_bytes: int = 2, include_mtp: bool = False) -> int:
    """MLA latent cache (kv_lora + rope) + DSA indexer key cache, per token."""
    n_layers = arch.num_total_layers if include_mtp else arch.num_hidden_layers
    mla = n_layers * (arch.kv_lora_rank + arch.qk_rope_head_dim) * kv_dtype_bytes
    n_full = sum(1 for l in range(n_layers) if arch.is_full_indexer_layer(l))
    idx = n_full * arch.index_head_dim * kv_dtype_bytes
    return mla + idx


# ---------------------------------------------------------------------------------------------
# Device profiles & residency planner
# ---------------------------------------------------------------------------------------------
GiB = 1024**3


@dataclass
class DeviceProfile:
    name: str
    gpu_bytes: int  # usable device-local bytes for the process (unified total on GB10)
    host_bytes: int  # pinned host bytes usable for a zero-copy/DMA tier (0 on unified memory)
    unified: bool
    h2d_gbps: float
    reserve_bytes: int  # CUDA context, activations, CUDA graphs, allocator slack
    notes: str = ""


PROFILES: dict[str, DeviceProfile] = {
    # DGX Spark: 128 GB LPDDR5X unified; the pb27-vllm container is capped at 108g (see start_dsv41_vllm.sh)
    "spark_gb10": DeviceProfile("spark_gb10", int(108e9), 0, True, 273.0, int(7e9), "unified LPDDR5X, sm_121"),
    # Colab A100-SXM4-80GB High-RAM runtime (167 GB host RAM as of Nov 2026; keep ~45 GB for OS / loader / page cache)
    "colab_a100_80g": DeviceProfile("colab_a100_80g", int(79.0e9), int(120e9), False, 22.0, int(5.5e9), "sm_80, PCIe Gen4"),
    "colab_h100_80g": DeviceProfile("colab_h100_80g", int(79.0e9), int(120e9), False, 45.0, int(5.5e9), "sm_90, PCIe Gen5"),
    "colab_g4_96g": DeviceProfile("colab_g4_96g", int(94.5e9), int(150e9), False, 45.0, int(5.5e9), "RTX PRO 6000 Blackwell, sm_120"),
    "colab_l4_24g": DeviceProfile("colab_l4_24g", int(22.0e9), int(50e9), False, 22.0, int(3e9), "sm_89 (too small for GLM-5.3)"),
}


def detect_profile_name() -> str:
    env = os.environ.get("CHRONOS_GLM53_PROFILE", "").strip()
    if env:
        return env
    try:
        import torch

        if not torch.cuda.is_available():
            return "colab_a100_80g"
        props = torch.cuda.get_device_properties(0)
        name = props.name.lower()
        cap = (props.major, props.minor)
        if "gb10" in name or cap == (12, 1):
            return "spark_gb10"
        if "h100" in name or cap == (9, 0):
            return "colab_h100_80g"
        if "rtx pro 6000" in name:
            return "colab_g4_96g"
        if "l4" in name:
            return "colab_l4_24g"
    except Exception:
        pass
    return "colab_a100_80g"


@dataclass
class ResidencyPlan:
    profile: str
    n_moe_layers: int
    t1_slots: int
    t2_slots: int
    t3_gpu_slots: int
    t3_host_slots: int
    coverage: float
    base_bytes: int
    kv_bytes: int
    moe_gpu_bytes: int
    moe_host_bytes: int
    gpu_total_bytes: int
    host_total_bytes: int
    feasible: bool
    notes: list[str]

    @property
    def resident_experts(self) -> int:
        return self.t1_slots + self.t2_slots + self.t3_gpu_slots + self.t3_host_slots

    def as_env(self) -> dict[str, str]:
        return {
            "CHRONOS_GLM53_T1_SLOTS": str(self.t1_slots),
            "CHRONOS_GLM53_T2_SLOTS": str(self.t2_slots),
            "CHRONOS_GLM53_T3_SLOTS": str(self.t3_gpu_slots),
            "CHRONOS_GLM53_T3_HOST_SLOTS": str(self.t3_host_slots),
        }

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["resident_experts"] = self.resident_experts
        return d


def plan_residency(
    arch: Glm53Arch | None = None,
    profile: str | DeviceProfile | None = None,
    ctx_tokens: int = 32768,
    include_mtp: bool = False,
    t1_slots: int | None = None,
    t2_slots: int | None = None,
    base_bytes_override: int | None = None,
) -> ResidencyPlan:
    """
    Picks per-layer tier slot counts so that all MoE layers fit the device:
      1. Reserve non-expert weights (stock FP8 attention / dense / shared experts + BF16 embeddings), KV cache and
         runtime slack.
      2. Cover as many of the `256` experts per layer as possible at `T3` (1.125 bpw), spilling to pinned host RAM on
         discrete GPUs (A100/H100) and pruning the lowest-prior experts if even that is not enough.
      3. Spend what is left upgrading the highest-prior experts to `T2` (2.25 bpw) and `T1` (4.5 bpw).
    """
    arch = arch or Glm53Arch()
    prof = PROFILES[profile] if isinstance(profile, str) else (profile or PROFILES[detect_profile_name()])
    n_layers = len(arch.moe_layer_ids) - (0 if include_mtp else arch.num_nextn_predict_layers)
    notes: list[str] = []

    base = base_bytes_override if base_bytes_override is not None else non_expert_bytes_estimate(arch)["total"]
    if not include_mtp:
        base -= arch.num_nextn_predict_layers * (
            non_expert_bytes_estimate(arch)["attention_fp8"] // arch.num_total_layers
            + 3 * arch.hidden_size * arch.moe_intermediate_size
        )
    kv = kv_bytes_per_token(arch, include_mtp=include_mtp) * ctx_tokens
    gpu_free = prof.gpu_bytes - prof.reserve_bytes - base - kv
    b1, b2, b3 = (tier_expert_bytes(arch, t) for t in ("t1_fp4", "t2_lm2", "t3_sign1"))
    e = arch.n_routed_experts
    top_k = arch.num_experts_per_tok
    if gpu_free <= 0:
        return ResidencyPlan(prof.name, n_layers, 0, 0, 0, 0, 0.0, base, kv, 0, 0, prof.reserve_bytes + base + kv, 0, False,
                             [f"base weights + KV ({(base + kv) / 1e9:.1f} GB) exceed {prof.name} memory"])

    gpu_per_layer = gpu_free // n_layers
    # On discrete GPUs keep `top_k` staging slots on the GPU for host-resident T3 experts.
    staging = 0 if prof.unified else top_k * b3
    host_per_layer = 0 if prof.unified else prof.host_bytes // n_layers

    t1 = 0 if t1_slots is None else t1_slots
    t2 = 0 if t2_slots is None else t2_slots
    budget_gpu = gpu_per_layer - staging - t1 * b1 - t2 * b2
    if budget_gpu < 0:
        notes.append("requested T1/T2 slots exceed the GPU budget; clamping")
        t1, t2 = 0, 0
        budget_gpu = gpu_per_layer - staging
    remaining = e - t1 - t2
    t3_gpu = int(max(0, min(remaining, budget_gpu // b3)))
    remaining -= t3_gpu
    budget_gpu -= t3_gpu * b3
    t3_host = int(max(0, min(remaining, host_per_layer // b3))) if not prof.unified else 0
    remaining -= t3_host

    # Upgrade highest-prior experts with leftover GPU bytes (T3 -> T2 -> T1) when not pinned by the caller.
    if t1_slots is None and t2_slots is None and remaining == 0:
        while t3_gpu + t3_host > 0 and budget_gpu >= (b2 - b3 if t3_gpu > 0 else b2):
            if t3_gpu > 0:
                t3_gpu -= 1
                budget_gpu -= b2 - b3
            else:
                t3_host -= 1
                budget_gpu -= b2
            t2 += 1
        while t2 > 0 and budget_gpu >= b1 - b2:
            t2 -= 1
            t1 += 1
            budget_gpu -= b1 - b2

    resident = t1 + t2 + t3_gpu + t3_host
    coverage = resident / e
    moe_gpu = n_layers * (t1 * b1 + t2 * b2 + t3_gpu * b3 + staging)
    moe_host = n_layers * t3_host * b3
    feasible = resident >= max(top_k, int(0.5 * e))
    if coverage < 1.0:
        notes.append(
            f"{e - resident}/{e} lowest-prior experts per layer are not resident; the router renormalizes over resident experts"
        )
    if t3_host:
        notes.append(f"{t3_host} T3 experts/layer live in pinned host RAM and are DMA-staged on demand ({prof.h2d_gbps:.0f} GB/s)")
    if not feasible:
        notes.append("coverage below 50% — expect severe quality loss; use a bigger GPU or fewer layers")
    return ResidencyPlan(
        profile=prof.name,
        n_moe_layers=n_layers,
        t1_slots=t1,
        t2_slots=t2,
        t3_gpu_slots=t3_gpu,
        t3_host_slots=t3_host,
        coverage=coverage,
        base_bytes=base,
        kv_bytes=kv,
        moe_gpu_bytes=moe_gpu,
        moe_host_bytes=moe_host,
        gpu_total_bytes=prof.reserve_bytes + base + kv + moe_gpu,
        host_total_bytes=moe_host,
        feasible=feasible,
        notes=notes,
    )


def footprint_report(arch: Glm53Arch | None = None, svd_rank: int = 64) -> dict[str, Any]:
    arch = arch or Glm53Arch()
    n_layers = len(arch.moe_layer_ids)
    n_groups = n_layers * arch.n_groups
    std = chronos_group_bytes(arch, svd_rank, 2, 2)
    cmp_ = chronos_group_bytes(arch, svd_rank, 1, 1)
    non_exp = non_expert_bytes_estimate(arch)
    stock_experts = n_layers * arch.n_routed_experts * fp8_expert_bytes(arch)
    return {
        "moe_layers": n_layers,
        "routed_experts": n_layers * arch.n_routed_experts,
        "stock_fp8_routed_gb": round(stock_experts / 1e9, 2),
        "non_expert_gb": round(non_exp["total"] / 1e9, 2),
        "chronos_standard_routed_gb": round(n_groups * std["total"] / 1e9, 2),
        "chronos_compact_routed_gb": round(n_groups * cmp_["total"] / 1e9, 2),
        "chronos_bank_plus_healer_gb": round(n_groups * (std["bank"] + std["healer"]) / 1e9, 2),
        "chronos_standard_total_gb": round((n_groups * std["total"] + non_exp["total"]) / 1e9, 2),
        "chronos_compact_total_gb": round((n_groups * cmp_["total"] + non_exp["total"]) / 1e9, 2),
        "tier_expert_mb": {t: round(tier_expert_bytes(arch, t) / 1e6, 3) for t in TIER_FORMATS},
        "kv_kb_per_token_bf16": round(kv_bytes_per_token(arch) / 1024, 1),
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="GLM-5.3 Chronos footprint & residency planner")
    ap.add_argument("--config", default="", help="Optional path to GLM-5.3 config.json")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--mtp", action="store_true")
    args = ap.parse_args()
    a = Glm53Arch.from_json_file(args.config) if args.config else Glm53Arch()
    print(json.dumps(footprint_report(a), indent=2))
    for pname in PROFILES:
        p = plan_residency(a, pname, ctx_tokens=args.ctx, include_mtp=args.mtp)
        print(
            f"{pname:16s} T1={p.t1_slots:3d} T2={p.t2_slots:3d} T3gpu={p.t3_gpu_slots:3d} T3host={p.t3_host_slots:3d} "
            f"coverage={p.coverage * 100:5.1f}% gpu={p.gpu_total_bytes / 1e9:6.1f}GB host={p.host_total_bytes / 1e9:5.1f}GB "
            f"feasible={p.feasible} {'; '.join(p.notes)}"
        )
