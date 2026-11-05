#!/usr/bin/env python3
"""
Chronos-1T v7.5 x GLM-5.3 — Standalone Inference Engine (Colab A100 / H100 / G4, also runs on DGX Spark and CPU).

vLLM serves `GlmMoeDsaForCausalLM` through its DSA sparse-MLA backends, which only exist for sm_90 / sm_100 /
sm_120 and require DeepGEMM; there is no sm_80 path. This engine is the Colab-side counterpart of
`scripts/run_40layer_chimera_dsv41_a100.py`: a compact, dependency-light GLM-5.3 forward that reproduces the
HF `transformers` `GlmMoeDsaForCausalLM` math exactly:

  * MLA with weight absorption (`q_nope @ W_UK`, `P @ c_kv @ W_UV`): the KV cache is the compressed latent
    (`512 + 64` per token per layer, 93 KB/token in BF16) instead of 64 expanded heads (4.5 MB/token)
  * DSA lightning indexer (`32 x 128`, interleaved RoPE, LayerNorm keys, ReLU head-weighted scores, top-2048) with
    `full` / `shared` cross-layer top-k reuse; for contexts <= `index_topk` the selection is provably all causal
    tokens, so the indexer keys are cached but scoring is skipped
  * stock 128x128 block-FP8 attention / dense / shared-expert weights (Triton W8 GEMV for decode, dequant + GEMM
    for prefill; manual e4m3 decode, so no FP8 tensor cores are required)
  * Chronos tiered MoE (`ChronosGlm53SparseMoeBlock`) on layers `3..77` (MTP layer 78 is not used here)
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gc
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Iterable

import torch
from torch import nn
import torch.nn.functional as F
from safetensors import safe_open

from chronos_1t.models import glm53_tiers as T
from chronos_1t.models.glm53_chronos_spec import (
    Glm53Arch,
    ResidencyPlan,
    base_shard_name,
    detect_profile_name,
    hot_shard_name,
    layer_shard_name,
    plan_residency,
)


class Linear8(nn.Module):
    """Stock block-FP8 linear (`weight` e4m3 viewed as uint8 + fp32 `weight_scale_inv`) or plain BF16 linear."""

    def __init__(self, w: torch.Tensor, s: torch.Tensor | None, device: torch.device) -> None:
        super().__init__()
        self.fp8 = s is not None
        if self.fp8:
            self.register_buffer("w", w.view(torch.uint8).to(device), persistent=False)
            self.register_buffer("s", s.float().to(device), persistent=False)
        else:
            dt = torch.bfloat16 if device.type == "cuda" else torch.float32
            self.register_buffer("w", w.to(device=device, dtype=dt), persistent=False)
            self.s = None
        self.out_features = w.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fp8:
            return T.fp8_linear(x, self.w, self.s, out_dtype=x.dtype)
        return F.linear(x.to(self.w.dtype), self.w).to(x.dtype)

    def dense(self, dtype: torch.dtype) -> torch.Tensor:
        return T.dequant_fp8_block(self.w, self.s, dtype=dtype) if self.fp8 else self.w.to(dtype)


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    return (w.float() * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps))).to(x.dtype)


def rope_interleave(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """HF `apply_rotary_pos_emb_interleave`: rotates (x0,x1),(x2,x3)... and emits [even', odd'] (de-interleaved)."""
    x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(x.dtype)


class Glm53Attention(nn.Module):
    def __init__(self, arch: Glm53Arch, layer_idx: int, get: Callable[[str], torch.Tensor], keys: set, device: torch.device,
                 max_len: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.arch, self.layer_idx, self.dtype = arch, layer_idx, dtype
        p = f"model.layers.{layer_idx}.self_attn."

        def lin(name: str) -> Linear8:
            sk = f"{p}{name}.weight_scale_inv"
            return Linear8(get(f"{p}{name}.weight"), get(sk) if sk in keys else None, device)

        self.q_a_proj, self.q_b_proj = lin("q_a_proj"), lin("q_b_proj")
        self.kv_a_proj = lin("kv_a_proj_with_mqa")
        self.kv_b_proj, self.o_proj = lin("kv_b_proj"), lin("o_proj")
        self.q_a_norm = get(f"{p}q_a_layernorm.weight").float().to(device)
        self.kv_a_norm = get(f"{p}kv_a_layernorm.weight").float().to(device)
        self.has_indexer = f"{p}indexer.wk.weight" in keys
        if self.has_indexer:
            ip = p + "indexer."
            self.idx_wq_b = Linear8(get(ip + "wq_b.weight"), get(ip + "wq_b.weight_scale_inv") if ip + "wq_b.weight_scale_inv" in keys else None, device)
            self.idx_wk = Linear8(get(ip + "wk.weight"), get(ip + "wk.weight_scale_inv") if ip + "wk.weight_scale_inv" in keys else None, device)
            self.idx_kn_w = get(ip + "k_norm.weight").float().to(device)
            self.idx_kn_b = get(ip + "k_norm.bias").float().to(device)
            self.idx_wproj = get(ip + "weights_proj.weight").float().to(device)  # kept fp32 (HF `_keep_in_fp32_modules`)
            self.idx_k = torch.zeros((max_len, arch.index_head_dim), dtype=dtype, device=device)
        self.ckv = torch.zeros((max_len, arch.kv_lora_rank), dtype=dtype, device=device)
        self.kpe = torch.zeros((max_len, arch.qk_rope_head_dim), dtype=dtype, device=device)
        self.scale = arch.qk_head_dim ** -0.5

    def _indexer_topk(self, x: torch.Tensor, q_resid: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, start: int,
                      need_scores: bool) -> torch.Tensor | None:
        a = self.arch
        t = x.shape[0]
        r = a.qk_rope_head_dim
        k = self.idx_wk(x).float()
        k = F.layer_norm(k, (a.index_head_dim,), self.idx_kn_w, self.idx_kn_b, eps=1e-6)
        k = torch.cat([rope_interleave(k[:, :r], cos, sin), k[:, r:]], dim=-1)
        self.idx_k[start : start + t] = k.to(self.dtype)
        if not need_scores:
            return None
        q = self.idx_wq_b(q_resid).float().view(t, a.index_n_heads, a.index_head_dim)
        q = torch.cat([rope_interleave(q[..., :r], cos[:, None], sin[:, None]), q[..., r:]], dim=-1)
        L = start + t
        keys = self.idx_k[:L].float()
        w = (x.float() @ self.idx_wproj.T) * (a.index_n_heads ** -0.5)  # (t, H)
        topk = min(a.index_topk, L)
        out = torch.empty((t, topk), dtype=torch.long, device=x.device)
        chunk = max(1, int(2**27 // max(1, a.index_n_heads * L)))
        pos = torch.arange(L, device=x.device)
        for c0 in range(0, t, chunk):
            c1 = min(t, c0 + chunk)
            sc = F.relu(torch.einsum("thd,ld->thl", q[c0:c1], keys) * (a.index_head_dim ** -0.5))
            idx = torch.einsum("th,thl->tl", w[c0:c1], sc)
            qpos = start + torch.arange(c0, c1, device=x.device)
            idx = idx.masked_fill(pos[None, :] > qpos[:, None], float("-inf"))
            out[c0:c1] = idx.topk(topk, dim=-1).indices
        return out

    def forward(self, x: torch.Tensor, start: int, cos: torch.Tensor, sin: torch.Tensor,
                prev_topk: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
        a = self.arch
        t = x.shape[0]
        nh, dn, dr, dv, c = a.num_attention_heads, a.qk_nope_head_dim, a.qk_rope_head_dim, a.v_head_dim, a.kv_lora_rank
        q_resid = rms_norm(self.q_a_proj(x), self.q_a_norm, a.rms_norm_eps)
        q = self.q_b_proj(q_resid).view(t, nh, dn + dr)
        q_nope, q_pe = q[..., :dn], rope_interleave(q[..., dn:], cos[:, None], sin[:, None])
        kv = self.kv_a_proj(x)
        self.ckv[start : start + t] = rms_norm(kv[:, :c], self.kv_a_norm, a.rms_norm_eps).to(self.dtype)
        self.kpe[start : start + t] = rope_interleave(kv[:, c:], cos, sin).to(self.dtype)
        L = start + t
        sparse = L > a.index_topk
        topk = prev_topk
        if self.has_indexer:
            topk = self._indexer_topk(x, q_resid, cos, sin, start, need_scores=sparse)
        elif sparse and topk is None:
            raise RuntimeError(f"layer {self.layer_idx}: shared DSA layer without a previous full-layer top-k")

        wkv = self.kv_b_proj.dense(torch.float32).view(nh, dn + dv, c)
        w_uk, w_uv = wkv[:, :dn], wkv[:, dn:]
        q_abs = torch.einsum("thd,hdc->thc", q_nope.float(), w_uk)
        ckv, kpe = self.ckv[:L].float(), self.kpe[:L].float()
        out = torch.empty((t, nh, dv), dtype=torch.float32, device=x.device)
        chunk = max(1, int(2**27 // max(1, nh * L)))
        pos = torch.arange(L, device=x.device)
        for c0 in range(0, t, chunk):
            c1 = min(t, c0 + chunk)
            s = (torch.einsum("thc,lc->thl", q_abs[c0:c1], ckv) + torch.einsum("thr,lr->thl", q_pe[c0:c1].float(), kpe)) * self.scale
            qpos = start + torch.arange(c0, c1, device=x.device)
            mask = pos[None, :] > qpos[:, None]
            if sparse:
                keep = torch.zeros((c1 - c0, L), dtype=torch.bool, device=x.device)
                keep.scatter_(1, topk[c0:c1], True)
                mask = mask | ~keep
            s = s.masked_fill(mask[:, None, :], float("-inf"))
            pr = torch.softmax(s, dim=-1)
            o_lat = torch.einsum("thl,lc->thc", pr, ckv)
            out[c0:c1] = torch.einsum("thc,hvc->thv", o_lat, w_uv)
        y = self.o_proj(out.reshape(t, nh * dv).to(x.dtype))
        return y, topk


class Glm53DenseMLP(nn.Module):
    def __init__(self, prefix: str, get: Callable[[str], torch.Tensor], device: torch.device) -> None:
        super().__init__()
        self.gate = Linear8(get(prefix + "gate_proj.weight"), get(prefix + "gate_proj.weight_scale_inv"), device)
        self.up = Linear8(get(prefix + "up_proj.weight"), get(prefix + "up_proj.weight_scale_inv"), device)
        self.down = Linear8(get(prefix + "down_proj.weight"), get(prefix + "down_proj.weight_scale_inv"), device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down((F.silu(self.gate(x).float()) * self.up(x).float()).to(x.dtype))


class Glm53Layer(nn.Module):
    def __init__(self, attn: Glm53Attention, mlp: nn.Module, in_norm: torch.Tensor, post_norm: torch.Tensor, eps: float):
        super().__init__()
        self.attn, self.mlp = attn, mlp
        self.in_norm, self.post_norm, self.eps = in_norm, post_norm, eps

    def forward(self, h: torch.Tensor, start: int, cos: torch.Tensor, sin: torch.Tensor, prev_topk: torch.Tensor | None):
        a, topk = self.attn(rms_norm(h, self.in_norm, self.eps), start, cos, sin, prev_topk)
        h = h + a
        h = h + self.mlp(rms_norm(h, self.post_norm, self.eps)).to(h.dtype)
        return h, topk


MoEFactory = Callable[[int, Path, Glm53Arch, torch.device], nn.Module]


class Glm53ChronosEngine:
    def __init__(
        self,
        model_dir: str | Path,
        device: torch.device | str | None = None,
        max_len: int = 32768,
        plan: ResidencyPlan | None = None,
        moe_factory: MoEFactory | None = None,
        num_layers: int | None = None,
        repo_id: str | None = None,
        delete_shards_after_load: bool = False,
        log: Callable[[str], None] = print,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.log = log
        self.repo_id = repo_id
        self.delete_after = delete_shards_after_load
        self._ensure_file("config.json")
        cfg = json.loads((self.model_dir / "config.json").read_text())
        self.config = cfg
        self.arch = arch = Glm53Arch.from_config(cfg)
        self.max_len = max_len
        self.n_layers = num_layers or arch.num_hidden_layers
        if plan is None and moe_factory is None:
            os.environ.setdefault("CHRONOS_GLM53_CTX", str(max_len))
            plan = plan_residency(arch, detect_profile_name(), ctx_tokens=max_len)
            log(f"[engine] residency plan {plan.profile}: T1={plan.t1_slots} T2={plan.t2_slots} T3={plan.t3_gpu_slots} "
                f"T3host={plan.t3_host_slots} coverage={plan.coverage:.1%} gpu~{plan.gpu_total_bytes / 1e9:.1f} GB "
                f"host~{plan.host_total_bytes / 1e9:.1f} GB | {'; '.join(plan.notes)}")
        self.plan = plan
        self.moe_factory = moe_factory or self._default_moe

        r = arch.qk_rope_head_dim
        inv = 1.0 / (arch.rope_theta ** (torch.arange(0, r, 2, dtype=torch.float32) / r))
        fr = torch.outer(torch.arange(max_len, dtype=torch.float32), inv)
        self.cos, self.sin = fr.cos().to(self.device), fr.sin().to(self.device)

        t0 = time.perf_counter()
        base = self._ensure_file(base_shard_name())
        with safe_open(str(base), framework="pt", device="cpu") as f:
            self.embed = f.get_tensor("model.embed_tokens.weight").to(self.device, self.dtype)
            self.final_norm = f.get_tensor("model.norm.weight").float().to(self.device)
            self.lm_head = f.get_tensor("lm_head.weight").to(self.device, self.dtype)
        self.layers: list[Glm53Layer] = []
        pool = ThreadPoolExecutor(max_workers=1)
        fut = pool.submit(self._ensure_file, layer_shard_name(0, arch))
        for l_idx in range(self.n_layers):
            path = fut.result()
            if l_idx + 1 < self.n_layers:
                fut = pool.submit(self._ensure_file, layer_shard_name(l_idx + 1, arch))
            tl0 = time.perf_counter()
            self.layers.append(self._build_layer(l_idx, path))
            if self.delete_after and self.repo_id:
                path.unlink(missing_ok=True)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                mem = torch.cuda.memory_allocated(self.device) / 1e9
            else:
                mem = 0.0
            log(f"[engine] layer {l_idx:02d}/{self.n_layers - 1} ready in {time.perf_counter() - tl0:.1f}s (gpu alloc {mem:.1f} GB)")
        pool.shutdown()
        gc.collect()
        self.pos = 0
        log(f"[engine] loaded {self.n_layers} layers in {(time.perf_counter() - t0) / 60:.1f} min")

    # ------------------------------------------------------------------------------------------
    def _ensure_file(self, name: str) -> Path:
        p = self.model_dir / name
        if p.exists():
            return p
        if not self.repo_id:
            raise FileNotFoundError(p)
        from huggingface_hub import hf_hub_download

        self.model_dir.mkdir(parents=True, exist_ok=True)
        return Path(hf_hub_download(repo_id=self.repo_id, filename=name, local_dir=str(self.model_dir)))

    def _default_moe(self, l_idx: int, path: Path, arch: Glm53Arch, device: torch.device) -> nn.Module:
        from chronos_1t.models.glm53_chronos_moe import ChronosGlm53SparseMoeBlock

        if self.repo_id and self.plan is not None and self.plan.t1_slots > 0:
            n_hot = int(self.config.get("chronos_holographic_moe", {}).get("hot_experts_per_layer", 0) or 0)
            if n_hot:
                try:
                    self._ensure_file(hot_shard_name(l_idx, n_hot, arch))
                except Exception:
                    pass
        return ChronosGlm53SparseMoeBlock(layer_idx=l_idx, shard_path=path, arch=arch, device=device, plan=self.plan)

    def _build_layer(self, l_idx: int, path: Path) -> Glm53Layer:
        arch = self.arch
        with safe_open(str(path), framework="pt", device="cpu") as f:
            keys = set(f.keys())
            attn = Glm53Attention(arch, l_idx, f.get_tensor, keys, self.device, self.max_len, self.dtype)
            p = f"model.layers.{l_idx}."
            in_norm = f.get_tensor(p + "input_layernorm.weight").float().to(self.device)
            post_norm = f.get_tensor(p + "post_attention_layernorm.weight").float().to(self.device)
            if l_idx in arch.moe_layer_ids:
                mlp = None
            else:
                mlp = Glm53DenseMLP(p + "mlp.", f.get_tensor, self.device)
        if mlp is None:
            mlp = self.moe_factory(l_idx, path, arch, self.device)
        return Glm53Layer(attn, mlp, in_norm, post_norm, arch.rms_norm_eps)

    # ------------------------------------------------------------------------------------------
    def reset(self) -> None:
        self.pos = 0

    @torch.no_grad()
    def forward(self, ids: torch.Tensor, all_logits: bool = False) -> torch.Tensor:
        t = ids.shape[0]
        if self.pos + t > self.max_len:
            raise ValueError(f"context {self.pos + t} exceeds max_len {self.max_len}")
        h = F.embedding(ids.to(self.device), self.embed)
        cos, sin = self.cos[self.pos : self.pos + t], self.sin[self.pos : self.pos + t]
        topk = None
        for layer in self.layers:
            h, topk = layer(h, self.pos, cos, sin, topk)
        self.pos += t
        h = rms_norm(h if all_logits else h[-1:], self.final_norm, self.arch.rms_norm_eps)
        return F.linear(h, self.lm_head).float()

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop_ids: Iterable[int] | None = None,
        on_token: Callable[[int], None] | None = None,
        prefill_chunk: int = 2048,
    ) -> dict[str, Any]:
        self.reset()
        stop = set(stop_ids if stop_ids is not None else self.arch.eos_token_ids)
        ids = torch.tensor(prompt_ids, dtype=torch.long)
        sync = torch.cuda.synchronize if self.device.type == "cuda" else (lambda: None)
        t0 = time.perf_counter()
        logits = None
        for c0 in range(0, len(prompt_ids), prefill_chunk):
            logits = self.forward(ids[c0 : c0 + prefill_chunk])
        sync()
        t_prefill = time.perf_counter() - t0
        out: list[int] = []
        t1 = time.perf_counter()
        gen = torch.Generator(device="cpu").manual_seed(0)
        for _ in range(max_new_tokens):
            lg = logits[-1]
            if temperature <= 0:
                nxt = int(lg.argmax())
            else:
                pr = torch.softmax(lg / temperature, dim=-1).cpu()
                if top_p < 1.0:
                    sp, si = torch.sort(pr, descending=True)
                    keep = (sp.cumsum(0) - sp) < top_p
                    pr = torch.zeros_like(pr).scatter_(0, si[keep], sp[keep])
                nxt = int(torch.multinomial(pr / pr.sum(), 1, generator=gen))
            out.append(nxt)
            if on_token:
                on_token(nxt)
            if nxt in stop or self.pos >= self.max_len:
                break
            logits = self.forward(torch.tensor([nxt]))
        sync()
        t_dec = time.perf_counter() - t1
        return {
            "tokens": out,
            "prefill_tokens": len(prompt_ids),
            "prefill_s": round(t_prefill, 3),
            "decode_tokens": len(out),
            "decode_s": round(t_dec, 3),
            "decode_tok_s": round(len(out) / max(t_dec, 1e-9), 2),
            "prefill_tok_s": round(len(prompt_ids) / max(t_prefill, 1e-9), 1),
        }

    def memory_report(self) -> dict[str, float]:
        gpu = host = 0
        for layer in self.layers:
            rb = getattr(layer.mlp, "resident_bytes", None)
            if rb:
                r = rb()
                gpu += r["gpu"]
                host += r["host"]
        rep = {"moe_gpu_gb": round(gpu / 1e9, 2), "moe_host_gb": round(host / 1e9, 2)}
        if self.device.type == "cuda":
            rep["cuda_allocated_gb"] = round(torch.cuda.memory_allocated(self.device) / 1e9, 2)
            rep["cuda_peak_gb"] = round(torch.cuda.max_memory_allocated(self.device) / 1e9, 2)
        return rep
