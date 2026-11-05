#!/usr/bin/env python3
"""
End-to-end synthetic validation of the GLM-5.3 Chronos pipeline against Hugging Face `transformers`
(`GlmMoeDsaForCausalLM`, the reference implementation of GLM-5.x DSA models):

  1. builds a tiny random GLM-MoE-DSA model (same code path as GLM-5.3: MLA + DSA indexer with full/shared layers,
     sigmoid noaux_tc router, shared expert, dense first layer) and exports it in the exact GLM-5.3 checkpoint layout
     (128x128 block-FP8 weights + `weight_scale_inv`, per-expert `experts.{e}.{gate,up,down}_proj`)
  2. runs `convert_glm53_chronos.py --source local` on it
  3. checks the standalone engine (`glm53_engine.py`) with exact (stock) experts against HF logits, both for a full
     prefill and for token-by-token decode through the latent KV / indexer caches (sequence > index_topk, so DSA
     sparse selection is exercised)
  4. runs the engine with the Chronos tiered MoE (`ChronosGlm53SparseMoeBlock`) and reports logit fidelity
  5. on CUDA, checks the Triton decode path against the PyTorch grouped path inside the MoE block
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chronos_1t.models import glm53_tiers as T  # noqa: E402
from chronos_1t.models.glm53_chronos_moe import ChronosGlm53SparseMoeBlock  # noqa: E402
from chronos_1t.models.glm53_engine import Glm53ChronosEngine  # noqa: E402

TINY = dict(
    vocab_size=512, hidden_size=256, intermediate_size=512, moe_intermediate_size=128,
    num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4, n_shared_experts=1, n_routed_experts=16,
    routed_scaling_factor=2.5, kv_lora_rank=64, q_lora_rank=128, qk_rope_head_dim=16, v_head_dim=32,
    qk_nope_head_dim=32, n_group=1, topk_group=1, num_experts_per_tok=4, norm_topk_prob=True,
    max_position_embeddings=4096, rms_norm_eps=1e-5, index_topk=8, index_head_dim=32, index_n_heads=8,
    first_k_dense_replace=1, indexer_types=["full", "full", "shared", "full"],
    rope_parameters={"rope_theta": 10000.0, "rope_type": "default"}, eos_token_id=[1], pad_token_id=0,
)


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


def build_and_export(src: Path, seed: int = 0):
    from transformers import GlmMoeDsaConfig, GlmMoeDsaForCausalLM

    torch.manual_seed(seed)
    cfg = GlmMoeDsaConfig(**TINY)
    cfg._attn_implementation = "eager"
    model = GlmMoeDsaForCausalLM(cfg).float().eval()
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "norm" in n and p.dim() == 1:
                p.copy_(1.0 + 0.1 * torch.randn_like(p))
            elif p.dim() >= 2:
                p.normal_(0.0, 0.06 if "embed" not in n else 1.0)
        for layer in model.model.layers:
            if hasattr(layer.mlp, "gate"):
                layer.mlp.gate.e_score_correction_bias.copy_(0.05 * torch.randn_like(layer.mlp.gate.e_score_correction_bias))
            if getattr(layer.self_attn, "indexer", None) is not None:
                layer.self_attn.indexer.k_norm.bias.copy_(0.1 * torch.randn(TINY["index_head_dim"]))

    out: dict[str, torch.Tensor] = {}
    sd = model.state_dict()

    def put_fp8(name: str, w: torch.Tensor) -> torch.Tensor:
        q, s = T.quantize_fp8_block(w.float())
        out[name + ".weight"] = q
        out[name + ".weight_scale_inv"] = s
        return T.dequant_fp8_block(q, s, dtype=torch.float32)

    with torch.no_grad():
        for k, v in sd.items():
            if k.endswith("rotary_emb.inv_freq") or "original_inv_freq" in k:
                continue
            if ".mlp.experts.gate_up_proj" in k:
                pre = k.replace(".experts.gate_up_proj", ".experts")
                deq = []
                for e in range(v.shape[0]):
                    g = put_fp8(f"{pre}.{e}.gate_proj", v[e, : TINY["moe_intermediate_size"]])
                    u = put_fp8(f"{pre}.{e}.up_proj", v[e, TINY["moe_intermediate_size"] :])
                    deq.append(torch.cat([g, u]))
                v.copy_(torch.stack(deq))
            elif ".mlp.experts.down_proj" in k:
                pre = k.replace(".experts.down_proj", ".experts")
                v.copy_(torch.stack([put_fp8(f"{pre}.{e}.down_proj", v[e]) for e in range(v.shape[0])]))
            elif (
                k.endswith(".weight") and v.dim() == 2 and "embed" not in k and "lm_head" not in k
                and ".mlp.gate.weight" not in k and "weights_proj" not in k
            ):
                v.copy_(put_fp8(k[: -len(".weight")], v))
            else:
                out[k] = v.clone().contiguous()
    shards = {}
    for k in out:
        if k.startswith("model.layers."):
            shards[k] = f"model-{int(k.split('.')[2]) + 1:05d}-of-00005.safetensors"
        else:
            shards[k] = "model-00005-of-00005.safetensors"
    src.mkdir(parents=True, exist_ok=True)
    for sf in sorted(set(shards.values())):
        save_file({k: out[k].contiguous() for k, s in shards.items() if s == sf}, str(src / sf))
    (src / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": shards}))
    cfg_json = dict(TINY)
    cfg_json.update({
        "architectures": ["GlmMoeDsaForCausalLM"], "model_type": "glm_moe_dsa", "hidden_act": "silu",
        "scoring_func": "sigmoid", "topk_method": "noaux_tc", "num_nextn_predict_layers": 0,
        "quantization_config": {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128], "activation_scheme": "dynamic"},
    })
    (src / "config.json").write_text(json.dumps(cfg_json, indent=2))
    return model


class RefMoE(torch.nn.Module):
    """Exact stock-weight MoE (same math as HF `GlmMoeDsaMoE`), used to validate the engine independently of Chronos."""

    def __init__(self, hf_mlp, device):
        super().__init__()
        self.mlp = hf_mlp.to(device)

    def forward(self, x):
        return self.mlp(x.float().unsqueeze(0)).squeeze(0).to(x.dtype)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(ROOT / ".glm53_synth"))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    work = Path(args.work)
    shutil.rmtree(work, ignore_errors=True)
    src, dst = work / "stock", work / "chronos"
    dev = torch.device(args.device)
    ok = True

    hf = build_and_export(src)
    print(f"[1] exported tiny GLM-MoE-DSA checkpoint -> {src}")

    cmd = [sys.executable, str(ROOT / "chronos_1t/converter/convert_glm53_chronos.py"), "--source", "local",
           "--source-dir", str(src), "--out-dir", str(dst), "--device", args.device, "--hot-experts", "4"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout[-1500:])
    if res.returncode != 0:
        print(res.stderr[-3000:])
        return 1
    prog = json.loads((dst / "glm53_chronos_progress.json").read_text())
    for l, r in sorted(prog["layers"].items(), key=lambda kv: int(kv[0])):
        if r.get("moe"):
            print(f"    layer {l}: func_cos={r['metrics']['func_cos']} hot={r.get('hot_experts')}")
    print(f"[2] converted -> {dst} (files: {sorted(p.name for p in dst.iterdir())[:8]} ...)")

    seq = torch.randint(2, TINY["vocab_size"], (24,), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        ref_logits = hf(seq[None]).logits[0].float()

    # ---- engine with exact experts ----
    hf_layers = hf.model.layers
    eng = Glm53ChronosEngine(dst, device=dev, max_len=64, moe_factory=lambda l, p, a, d: RefMoE(hf_layers[l].mlp, d), log=lambda s: None)
    full = eng.forward(seq, all_logits=True)
    c_full = cos(full, ref_logits.to(full.device))
    print(f"[3] engine(exact experts) full-prefill logits cos vs HF = {c_full:.6f}  max|d|={(full.cpu() - ref_logits).abs().max():.2e}")
    eng.reset()
    inc = [eng.forward(seq[:12], all_logits=True).cpu()]
    for t in range(12, 24):
        inc.append(eng.forward(seq[t : t + 1]).cpu())
    inc_l = torch.cat(inc)
    c_inc = cos(inc_l, ref_logits)
    agree = (inc_l.argmax(-1) == ref_logits.argmax(-1)).float().mean().item()
    print(f"    engine(exact experts) prefill12+decode12 cos vs HF = {c_inc:.6f} argmax agree={agree:.3f}")
    thr = 0.9999 if dev.type == "cpu" else 0.999  # bf16 activations on CUDA
    ok &= c_full > thr and c_inc > thr

    # ---- engine with Chronos tiered MoE ----
    for plan in ({"t1": 16, "t2": 0, "t3": 0}, {"t1": 4, "t2": 4, "t3": 8}, {"t1": 0, "t2": 0, "t3": 12}):
        eng_c = Glm53ChronosEngine(
            dst, device=dev, max_len=64,
            moe_factory=lambda l, p, a, d, plan=plan: ChronosGlm53SparseMoeBlock(l, p, a, d, plan=dict(plan)),
            log=lambda s: None,
        )
        lc = eng_c.forward(seq, all_logits=True).cpu()
        agree = (lc.argmax(-1) == ref_logits.argmax(-1)).float().mean().item()
        hot = [getattr(L.mlp, "hot_loaded", 0) for L in eng_c.layers[1:]]
        print(f"[4] engine(chronos {plan}) logits cos vs HF = {cos(lc, ref_logits):.5f} argmax agree={agree:.3f} hot_loaded={hot}")
        del eng_c

    # ---- Triton decode path vs grouped path ----
    if dev.type == "cuda" and T.triton_enabled(dev):
        blk = ChronosGlm53SparseMoeBlock(2, dst / "chronos-glm53-layer-00002-of-00004.safetensors", device=dev,
                                         plan={"t1": 4, "t2": 4, "t3": 8})
        x = torch.randn(3, TINY["hidden_size"], device=dev, dtype=torch.bfloat16)
        ids, w = blk.route(x)
        y_tri = blk.routed_forward(x, ids, w)
        blk.use_triton = False
        y_ref = blk.routed_forward(x, ids, w)
        c = cos(y_tri, y_ref)
        print(f"[5] MoE block Triton decode vs grouped path cos = {c:.6f}")
        ok &= c > 0.999
    print("ALL OK" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
