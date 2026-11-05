#!/usr/bin/env python3
"""
Offline self-test for `patch_glm53_vllm` using stub `vllm` modules that mirror the v0.31 import sites
(`vllm.models.deepseek_v32.nvidia.model.DeepseekV2MoE`, `vllm.model_executor.models.deepseek_v2`, the default
loader / weight_utils iterators). Requires the synthetic Chronos checkpoint produced by
`glm53_synthetic_e2e_test.py` (`--work DIR` -> `DIR/chronos`).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import types

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _install_stub_vllm(model_dir: Path, preexisting_glm5_factory: bool) -> dict:
    class DeepseekV2MoE(torch.nn.Module):
        def __init__(self, config=None, parallel_config=None, quant_config=None, reduce_results=True, prefix="",
                     apply_routed_scale_to_output=False):
            super().__init__()
            self.prefix = prefix
            self.stock = True

    def safetensors_weights_iterator(files, *a, **k):
        from safetensors import safe_open

        for f in files:
            with safe_open(f, framework="pt", device="cpu") as fh:
                for n in fh.keys():
                    yield n, fh.get_tensor(n)

    mods = {}
    for name in ["vllm", "vllm.models", "vllm.models.deepseek_v32", "vllm.models.deepseek_v32.nvidia",
                 "vllm.model_executor", "vllm.model_executor.models", "vllm.model_executor.model_loader", "vllm.config"]:
        mods[name] = types.ModuleType(name)
    dv2 = types.ModuleType("vllm.model_executor.models.deepseek_v2")
    dv2.DeepseekV2MoE = DeepseekV2MoE
    if preexisting_glm5_factory:  # what `patch_glm5_vllm.enable_chronos_glm5_vllm_hook` leaves behind
        dv2._OrigDeepseekV2MoE = DeepseekV2MoE
        dv2.DeepseekV2MoE = lambda vllm_config=None, prefix="", **kw: DeepseekV2MoE(vllm_config=vllm_config, prefix=prefix, **kw)
    nv = types.ModuleType("vllm.models.deepseek_v32.nvidia.model")
    nv.DeepseekV2MoE = dv2.DeepseekV2MoE
    wu = types.ModuleType("vllm.model_executor.model_loader.weight_utils")
    wu.safetensors_weights_iterator = safetensors_weights_iterator
    dl = types.ModuleType("vllm.model_executor.model_loader.default_loader")
    dl.safetensors_weights_iterator = safetensors_weights_iterator
    mods.update({
        "vllm.model_executor.models.deepseek_v2": dv2,
        "vllm.models.deepseek_v32.nvidia.model": nv,
        "vllm.model_executor.model_loader.weight_utils": wu,
        "vllm.model_executor.model_loader.default_loader": dl,
    })
    vcfg = types.SimpleNamespace(model_config=types.SimpleNamespace(model=str(model_dir), max_model_len=64), speculative_config=None)
    mods["vllm.config"].get_current_vllm_config = lambda: vcfg
    mods["vllm.model_executor.model_loader"].default_loader = dl
    mods["vllm.model_executor.model_loader"].weight_utils = wu
    sys.modules.update(mods)
    return {"orig": DeepseekV2MoE, "nv": nv, "dv2": dv2, "wu": wu}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chronos-dir", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    mdir = Path(args.chronos_dir)
    ok = True
    for pre in (False, True):
        for k in [k for k in list(sys.modules) if k.startswith("vllm") or k.startswith("chronos_1t.models.patch_glm53")]:
            del sys.modules[k]
        st = _install_stub_vllm(mdir, preexisting_glm5_factory=pre)
        import os

        os.environ.update({"CHRONOS_GLM53_T1_SLOTS": "4", "CHRONOS_GLM53_T2_SLOTS": "4", "CHRONOS_GLM53_T3_SLOTS": "8",
                           "CHRONOS_GLM53_PROFILE": "colab_a100_80g"})
        from chronos_1t.models import patch_glm53_vllm as P

        P._OP_REGISTERED = P._OP_REGISTERED or hasattr(torch.ops, "chronos_glm53") and hasattr(torch.ops.chronos_glm53, "moe_forward")
        assert P.enable_chronos_glm53_vllm_hook()
        cls = st["nv"].DeepseekV2MoE
        assert isinstance(cls, type) and cls is st["dv2"].DeepseekV2MoE, "both import sites must hold the Chronos class"
        stock = cls(config=None, parallel_config=None, prefix="model.layers.0.mlp")  # dense layer -> must stay stock
        moe = cls(config=None, parallel_config=None, quant_config=None, reduce_results=False, prefix="model.layers.2.mlp",
                  apply_routed_scale_to_output=False)
        missing = cls(config=None, parallel_config=None, prefix="model.layers.9.mlp")
        print(f"[glm5-factory-preinstalled={pre}] layer0 -> {type(stock).__name__}, layer2 -> {type(moe).__name__}, "
              f"layer9 -> {type(missing).__name__}")
        ok &= type(moe).__name__ == "ChronosGlm53VllmMoE" and isinstance(missing, st["orig"]) and getattr(missing, "stock", False)
        ok &= isinstance(stock, st["orig"])
        ok &= isinstance(moe, cls)  # vLLM set_moe_parameters isinstance check
        x = torch.randn(3, moe.hidden, device=moe.device)
        y_op = moe(x)
        y_direct = moe.forward_impl(x)
        ok &= torch.allclose(y_op, y_direct)

        class Wrap(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, h):
                return self.m(h) * 2.0

        try:
            yc = torch.compile(Wrap(moe), fullgraph=True, backend="eager")(x)
            comp_ok = torch.allclose(yc, 2 * y_direct)
        except Exception as e:  # pragma: no cover
            print(f"  torch.compile failed: {e}")
            comp_ok = False
        print(f"  custom-op forward == direct: {torch.allclose(y_op, y_direct)}, torch.compile(fullgraph) ok: {comp_ok}")
        ok &= comp_ok

        files = sorted(str(p) for p in mdir.glob("chronos-glm53-*.safetensors"))
        names = [n for n, _ in st["wu"].safetensors_weights_iterator(files)]
        leaked = [n for n in names if (".chronos_groups." in n or ".hot_experts." in n or ".shared_experts." in n
                                       or n.endswith(".mlp.gate.weight") or n.endswith("e_score_correction_bias"))]
        dense_kept = [n for n in names if ".mlp.gate_proj" in n]
        print(f"  iterator: yielded {len(names)} tensors, leaked MoE tensors={len(leaked)}, dense MLP kept={len(dense_kept)}")
        ok &= not leaked and len(dense_kept) > 0
    print("ALL OK" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
