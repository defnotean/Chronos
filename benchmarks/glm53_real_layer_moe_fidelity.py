#!/usr/bin/env python3
"""
Real GLM-5.3 single-layer MoE fidelity under each device's residency plan.

Downloads the 3 source shards of one MoE layer (~16 GB), converts the full layer to Chronos v7.5 (+ an optional
hot-expert sidecar), then compares the MoE output of `ChronosGlm53SparseMoeBlock` under several tier plans
(DGX Spark, Colab A100, all-T1/T2/T3) against the exact stock-FP8 experts on router-manifold probe activations
(post-attention RMSNorm scale applied). Reports routed-output cosine, full MoE (routed + shared) cosine and the
residual-stream cosine `cos(h + y_chronos, h + y_stock)` with `|h| = |y_stock|`-matched synthetic residuals.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chronos_1t.converter.convert_glm53_chronos import HFStreamingSource, convert_layer, fetch_json  # noqa: E402
from chronos_1t.models.glm53_chronos_moe import ChronosGlm53SparseMoeBlock  # noqa: E402
from chronos_1t.models.glm53_chronos_spec import Glm53Arch, hot_shard_name, layer_shard_name, plan_residency  # noqa: E402
from chronos_1t.models.glm53_holo import make_router_probes, router_subspace  # noqa: E402
from chronos_1t.models.glm53_tiers import dequant_fp8_block  # noqa: E402


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    return float((a @ b) / (a.norm() * b.norm()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=40)
    ap.add_argument("--work", default=str(ROOT / ".glm53_real_layer"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--probes", type=int, default=256)
    ap.add_argument("--hot", type=int, default=32)
    args = ap.parse_args()
    dev = torch.device(args.device)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    cfg = fetch_json("zai-org/GLM-5.3", "config.json")
    arch = Glm53Arch.from_config(cfg)
    (work / "config.json").write_text(json.dumps(cfg))
    wmap = fetch_json("zai-org/GLM-5.3", "model.safetensors.index.json")["weight_map"]
    L = args.layer
    src = HFStreamingSource("zai-org/GLM-5.3", work / "src", wmap, [L], keep=True)
    src.prefetch_layer(L)

    shard = work / layer_shard_name(L, arch)
    if not shard.exists():
        t0 = time.perf_counter()
        tensors, rep, hot = convert_layer(src, arch, L, dev, 64, 2, 2, verify=True, hot_experts=args.hot)
        save_file(tensors, str(shard))
        if hot:
            save_file(hot, str(work / hot_shard_name(L, args.hot, arch)))
        (work / f"layer{L}_report.json").write_text(json.dumps({k: v for k, v in rep.items() if k != "keys"}, indent=2))
        print(f"[convert] layer {L} in {time.perf_counter() - t0:.0f}s -> {shard.name} ({shard.stat().st_size / 1e9:.2f} GB); "
              f"func_cos={rep['metrics']['func_cos']}", flush=True)
        del tensors, hot

    pre = f"model.layers.{L}.mlp"
    router = src.get(f"{pre}.gate.weight").to(dev, torch.float32)
    bias = src.get(f"{pre}.gate.e_score_correction_bias").to(dev, torch.float32)
    norm_w = src.get(f"model.layers.{L}.post_attention_layernorm.weight")
    v_sub, _ = router_subspace(router)
    x = make_router_probes(router, v_sub, args.probes, norm_w, seed=99).to(dev)
    scores = torch.sigmoid(x @ router.T)
    _, ids = torch.topk(scores + bias, arch.num_experts_per_tok, dim=-1)
    w = scores.gather(1, ids)
    w = w / w.sum(-1, keepdim=True) * arch.routed_scaling_factor
    y_ref = torch.zeros((args.probes, arch.hidden_size), device=dev, dtype=torch.float64)
    t0 = time.perf_counter()
    for e in torch.unique(ids).tolist():
        rows, slot = torch.where(ids == e)
        g = dequant_fp8_block(src.get(f"{pre}.experts.{e}.gate_proj.weight"), src.get(f"{pre}.experts.{e}.gate_proj.weight_scale_inv"), torch.float32, dev)
        u = dequant_fp8_block(src.get(f"{pre}.experts.{e}.up_proj.weight"), src.get(f"{pre}.experts.{e}.up_proj.weight_scale_inv"), torch.float32, dev)
        d = dequant_fp8_block(src.get(f"{pre}.experts.{e}.down_proj.weight"), src.get(f"{pre}.experts.{e}.down_proj.weight_scale_inv"), torch.float32, dev)
        xs = x[rows]
        ye = (F.silu(xs @ g.T) * (xs @ u.T)) @ d.T
        y_ref.index_add_(0, rows, (ye * w[rows, slot].unsqueeze(-1)).double())
    print(f"[reference] stock-FP8 routed outputs for {args.probes} probes ({len(torch.unique(ids))} experts) in {time.perf_counter() - t0:.0f}s", flush=True)

    plans = {}
    for prof in ("spark_gb10", "colab_a100_80g", "colab_g4_96g"):
        p = plan_residency(arch, prof, ctx_tokens=32768)
        plans[prof] = {"t1": p.t1_slots, "t2": p.t2_slots, "t3": p.t3_gpu_slots + p.t3_host_slots}
    plans["spark+hot32_t1"] = {"t1": 32, "t2": 0, "t3": plans["spark_gb10"]["t3"] - 128}
    plans["all_t3"] = {"t1": 0, "t2": 0, "t3": 256}
    plans["all_t2"] = {"t1": 0, "t2": 256, "t3": 0}
    plans["all_t1"] = {"t1": 256, "t2": 0, "t3": 0}
    results = {}
    blk = None
    for name, plan in plans.items():
        del blk
        torch.cuda.empty_cache() if dev.type == "cuda" else None
        t0 = time.perf_counter()
        blk = ChronosGlm53SparseMoeBlock(L, shard, arch, dev, plan=dict(plan))
        build_s = time.perf_counter() - t0
        xb = x.to(torch.bfloat16) if dev.type == "cuda" else x
        bi, bw = blk.route(xb)
        y = blk.routed_forward(xb, bi, bw).double()
        sh = blk.shared_forward(xb).double()
        h = torch.randn_like(y_ref) * (y_ref.norm(dim=-1, keepdim=True) + sh.norm(dim=-1, keepdim=True)) / (arch.hidden_size ** 0.5)
        r = {
            "plan": plan,
            "coverage": round(len(blk.assign) / arch.n_routed_experts, 3),
            "hot_from_sidecar": blk.hot_loaded,
            "routed_cos": round(cos(y, y_ref), 5),
            "moe_cos": round(cos(y + sh, y_ref + sh), 5),
            "residual_cos": round(cos(h + y + sh, h + y_ref + sh), 5),
            "build_s": round(build_s, 1),
            "resident_gb": round(blk.resident_bytes()["gpu"] / 1e9, 3),
        }
        results[name] = r
        print(f"[{name:16s}] {json.dumps(r)}", flush=True)
    (work / f"layer{L}_plan_fidelity.json").write_text(json.dumps(results, indent=2))
    src.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
