#!/usr/bin/env python3
"""
Chronos-1T v7.5 x GLM-5.3 — Tiered Runtime MoE Block (`ChronosGlm53SparseMoeBlock`) for DGX Spark & Colab GPUs.

Drop-in replacement for vLLM's `DeepseekV2MoE` on `GlmMoeDsaForCausalLM` layers `3..78` and for the MoE layers of
the standalone Colab engine (`glm53_engine.py`). Follows the DeepSeek-V4.1 tier contract (`T1` 4-bit / `T2` 2-bit /
`T3` 1-bit slots built at load time from the Chronos holographic shard, router renormalized over resident experts):

  load:    chronos-glm53-layer-XXXXX shard --(bank + healer + tail reconstruct)--> fp32 expert --(pack)--> tier slot
           optional tier cache (`chronos-glm53-tiers-<tag>-layer-XXXXX.safetensors`) makes restarts load in seconds
  decode:  sigmoid noaux_tc top-8 (absent experts masked) -> one Triton tier-dispatch GEMV for gate_up, SwiGLU,
           one for down (no host syncs => CUDA-graph capturable on Spark/vLLM) + FP8 shared expert
  prefill: experts grouped on the host, each used expert dequantized once and run as a dense GEMM
  A100:    `T3` experts that do not fit in HBM live in pinned host RAM and are DMA-staged per step (planner decides)

Environment overrides: `CHRONOS_GLM53_T1_SLOTS`, `CHRONOS_GLM53_T2_SLOTS`, `CHRONOS_GLM53_T3_SLOTS`,
`CHRONOS_GLM53_T3_HOST_SLOTS`, `CHRONOS_GLM53_PROFILE`, `CHRONOS_GLM53_CTX`, `CHRONOS_GLM53_TIER_CACHE` (dir),
`CHRONOS_GLM53_TRACK=1` (record routing counts for calibration), `CHRONOS_GLM53_DISABLE_TRITON=1`.
"""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable

import torch
from torch import nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from chronos_1t.models import glm53_tiers as T
from chronos_1t.models.glm53_chronos_spec import (
    GROUP_SIZE,
    Glm53Arch,
    ResidencyPlan,
    detect_profile_name,
    layer_shard_name,
    moe_prefix,
    plan_residency,
)
from chronos_1t.models.glm53_holo import (
    GROUP_TENSOR_SUFFIXES,
    reconstruct_group_experts,
    router_prior,
)

STATS_FILE = "chronos_glm53_expert_stats.json"
_PLAN_CACHE: dict[tuple, ResidencyPlan] = {}
LIVE_BLOCKS: dict[int, "ChronosGlm53SparseMoeBlock"] = {}


def _evict(path: str) -> None:
    if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
        try:
            fd = os.open(os.path.realpath(path), os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        except OSError:
            pass


def default_plan(arch: Glm53Arch, include_mtp: bool = False) -> ResidencyPlan:
    profile = detect_profile_name()
    ctx = int(os.environ.get("CHRONOS_GLM53_CTX", "32768"))
    key = (profile, ctx, include_mtp, arch.hidden_size, arch.n_routed_experts)
    plan = _PLAN_CACHE.get(key)
    if plan is None:
        plan = plan_residency(arch, profile, ctx_tokens=ctx, include_mtp=include_mtp)
        _PLAN_CACHE[key] = plan
    return plan


def load_expert_stats(model_dir: str | Path | None) -> dict[str, list[float]]:
    if not model_dir:
        return {}
    p = Path(model_dir) / STATS_FILE
    if p.exists():
        try:
            return json.loads(p.read_text()).get("layers", {})
        except Exception:
            return {}
    return {}


def save_expert_stats(model_dir: str | Path) -> Path:
    """Merges routing counts recorded with `CHRONOS_GLM53_TRACK=1` into `chronos_glm53_expert_stats.json`."""
    p = Path(model_dir) / STATS_FILE
    data = json.loads(p.read_text()) if p.exists() else {"layers": {}}
    for l_idx, blk in LIVE_BLOCKS.items():
        if blk.hits is None:
            continue
        old = torch.tensor(data["layers"].get(str(l_idx), [0.0] * blk.n_experts))
        data["layers"][str(l_idx)] = (old + blk.hits.float().cpu()).tolist()
    p.write_text(json.dumps(data))
    return p


class _ChronosDummyExperts(nn.Module):
    """Satisfies vLLM's `MixtureOfExperts` bookkeeping (`moe_layers.append(layer.mlp.experts)`)."""

    def update_expert_map(self) -> None:  # pragma: no cover - vLLM EPLB hook
        return None


class ChronosGlm53SparseMoeBlock(nn.Module):
    def __init__(
        self,
        layer_idx: int = 3,
        shard_path: str | Path | None = None,
        arch: Glm53Arch | None = None,
        device: torch.device | str | None = None,
        plan: ResidencyPlan | dict[str, int] | None = None,
        prior: torch.Tensor | None = None,
        tier_cache_dir: str | Path | None = None,
        gemv_max_pairs: int = 64,
        stage_slots: int = 16,
        vllm_config: Any = None,
        prefix: str = "",
        **_: Any,
    ) -> None:
        super().__init__()
        model_dir = None
        if vllm_config is not None:
            m = re.search(r"layers\.(\d+)", prefix)
            if m:
                layer_idx = int(m.group(1))
            model_dir = Path(vllm_config.model_config.model)
        if arch is None:
            cfg_dir = model_dir or (Path(shard_path).parent if shard_path else None)
            arch = Glm53Arch.from_json_file(cfg_dir / "config.json") if cfg_dir and (cfg_dir / "config.json").exists() else Glm53Arch()
        if shard_path is None:
            if model_dir is None:
                raise ValueError("shard_path or vllm_config is required")
            shard_path = model_dir / layer_shard_name(layer_idx, arch)
        shard_path = Path(shard_path)
        model_dir = model_dir or shard_path.parent
        if device is None:
            device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.arch = arch
        self.layer_idx = layer_idx
        self.hidden = arch.hidden_size
        self.inter = arch.moe_intermediate_size
        self.n_experts = arch.n_routed_experts
        self.top_k = arch.num_experts_per_tok
        self.routed_scaling_factor = arch.routed_scaling_factor
        self.gemv_max_pairs = gemv_max_pairs
        self.use_triton = T.triton_enabled(self.device)

        # vLLM `DeepseekV2MoE` compatibility surface
        self.n_routed_experts = self.n_experts
        self.n_shared_experts = arch.n_shared_experts
        self.n_logical_experts = self.n_experts
        self.n_physical_experts = self.n_experts
        self.n_local_physical_experts = self.n_experts
        self.n_redundant_experts = 0
        self.experts = _ChronosDummyExperts()
        self.is_sequence_parallel = False

        tp = moe_prefix(layer_idx)
        fh = safe_open(str(shard_path), framework="pt", device="cpu")
        keys = set(fh.keys())
        self.router_w = fh.get_tensor(f"{tp}.gate.weight").to(device=self.device, dtype=torch.float32).contiguous()
        self.e_bias = fh.get_tensor(f"{tp}.gate.e_score_correction_bias").to(device=self.device, dtype=torch.float32)

        # Shared expert kept in stock block-FP8 (gate & up fused along rows; I is a multiple of 128)
        def _fp8(name: str) -> tuple[torch.Tensor, torch.Tensor]:
            w = fh.get_tensor(f"{tp}.shared_experts.{name}.weight")
            s = fh.get_tensor(f"{tp}.shared_experts.{name}.weight_scale_inv").float()
            return w.view(torch.uint8), s

        g_w, g_s = _fp8("gate_proj")
        u_w, u_s = _fp8("up_proj")
        d_w, d_s = _fp8("down_proj")
        self.sh_gu_w = torch.cat([g_w, u_w], 0).to(self.device).contiguous()
        self.sh_gu_s = torch.cat([g_s, u_s], 0).to(self.device).contiguous()
        self.sh_dn_w = d_w.to(self.device).contiguous()
        self.sh_dn_s = d_s.to(self.device).contiguous()
        self.sh_inter = g_w.shape[0]

        # ---- residency plan ----
        if plan is None:
            plan = default_plan(arch, include_mtp=layer_idx in arch.mtp_layer_ids)
        if isinstance(plan, ResidencyPlan):
            slots = {"t1": plan.t1_slots, "t2": plan.t2_slots, "t3": plan.t3_gpu_slots, "t3h": plan.t3_host_slots}
            unified = plan.profile == "spark_gb10"
        else:
            slots = {"t1": 0, "t2": 0, "t3": self.n_experts, "t3h": 0, **plan}
            unified = True
        for k_env, k_slot in (("T1_SLOTS", "t1"), ("T2_SLOTS", "t2"), ("T3_SLOTS", "t3"), ("T3_HOST_SLOTS", "t3h")):
            v = os.environ.get(f"CHRONOS_GLM53_{k_env}")
            if v not in (None, ""):
                slots[k_slot] = int(v)
        if self.device.type != "cuda":
            slots["t3"] += slots["t3h"]
            slots["t3h"] = 0
        total = sum(slots.values())
        if total > self.n_experts:  # trim from the coldest tier first
            over = total - self.n_experts
            for k_slot in ("t3h", "t3", "t2", "t1"):
                cut = min(over, slots[k_slot])
                slots[k_slot] -= cut
                over -= cut
        self.slots = slots
        self.unified = unified

        # ---- expert ranking ----
        if prior is None:
            stats = load_expert_stats(model_dir).get(str(layer_idx))
            if stats is not None and len(stats) == self.n_experts:
                prior = torch.tensor(stats, dtype=torch.float32)
            else:
                prior = self._router_prior(fh, keys)
        order = torch.argsort(prior.float().cpu(), descending=True).tolist()
        bounds = [slots["t1"], slots["t2"], slots["t3"], slots["t3h"]]
        tiers = [T.TIER_T1, T.TIER_T2, T.TIER_T3, T.TIER_STAGE]
        assign: dict[int, tuple[int, int]] = {}
        pos = 0
        for tier, cnt in zip(tiers, bounds):
            for s_i in range(cnt):
                assign[order[pos]] = (tier, s_i)
                pos += 1
        self.assign = assign

        # ---- buffers ----
        h, i2, i = self.hidden, 2 * self.inter, self.inter
        self.gu_bufs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.dn_bufs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for tier, cnt in ((T.TIER_T1, slots["t1"]), (T.TIER_T2, slots["t2"]), (T.TIER_T3, slots["t3"])):
            if cnt > 0:
                self.gu_bufs[tier] = self._alloc(tier, cnt, i2, h, self.device)
                self.dn_bufs[tier] = self._alloc(tier, cnt, h, i, self.device)
        self.host_gu: tuple[torch.Tensor, torch.Tensor] | None = None
        self.host_dn: tuple[torch.Tensor, torch.Tensor] | None = None
        if slots["t3h"] > 0:
            self.host_gu = self._alloc(T.TIER_T3, slots["t3h"], i2, h, torch.device("cpu"), pin=True)
            self.host_dn = self._alloc(T.TIER_T3, slots["t3h"], h, i, torch.device("cpu"), pin=True)
            self.stage_slots = max(stage_slots, self.top_k)
            self.gu_bufs[T.TIER_STAGE] = self._alloc(T.TIER_T3, self.stage_slots, i2, h, self.device)
            self.dn_bufs[T.TIER_STAGE] = self._alloc(T.TIER_T3, self.stage_slots, h, i, self.device)
        else:
            self.stage_slots = 0

        tier_map = torch.zeros(self.n_experts, dtype=torch.int32)
        for eid, (tier, s_i) in assign.items():
            tier_map[eid] = (tier << T.TIER_SHIFT) | s_i
        self.tier_map_cpu = tier_map
        self.tier_map = tier_map.to(self.device)
        absent = torch.zeros(self.n_experts, dtype=torch.float32)
        absent[tier_map == 0] = float("-inf")
        self.absent_bias = absent.to(self.device)
        self.host_ids = {eid for eid, (tier, _) in assign.items() if tier == T.TIER_STAGE}

        # ---- fill slots (tier cache -> hot sidecar + reconstruct) ----
        self.hot_path = self._find_hot_sidecar(shard_path) if os.environ.get("CHRONOS_GLM53_USE_HOT", "1") == "1" else None
        self.hot_loaded = 0
        cache_dir = tier_cache_dir or os.environ.get("CHRONOS_GLM53_TIER_CACHE", "")
        tag = f"t1-{slots['t1']}_t2-{slots['t2']}_t3-{slots['t3']}_h-{slots['t3h']}{'_hot' if self.hot_path else ''}"
        cache_path = Path(cache_dir) / f"chronos-glm53-tiers-{tag}-layer-{layer_idx:05d}.safetensors" if cache_dir else None
        if cache_path is not None and cache_path.exists() and self._load_tier_cache(cache_path):
            pass
        else:
            self._build_slots(fh, keys, tp)
            if cache_path is not None:
                self._save_tier_cache(cache_path)
        del fh
        _evict(str(shard_path))
        self.hits = torch.zeros(self.n_experts, dtype=torch.int64, device=self.device) if os.environ.get("CHRONOS_GLM53_TRACK") == "1" else None
        LIVE_BLOCKS[layer_idx] = self
        gc.collect()

    # ------------------------------------------------------------------------------------------
    @staticmethod
    def _alloc(tier: int, cnt: int, n: int, k: int, device: torch.device, pin: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        qs, ss = T.tier_shapes(tier, n, k)
        kw: dict[str, Any] = {"device": device}
        if pin and torch.cuda.is_available():
            kw["pin_memory"] = True
        return (
            torch.empty((cnt, *qs), dtype=torch.uint8, **kw),
            torch.empty((cnt, *ss), dtype=torch.float16, **kw),
        )

    @torch.no_grad()
    def _router_prior(self, fh: Any, keys: set) -> torch.Tensor:
        nk = f"model.layers.{self.layer_idx}.post_attention_layernorm.weight"
        norm_w = fh.get_tensor(nk) if nk in keys else None
        return router_prior(self.router_w, self.e_bias, self.top_k, norm_w, self.layer_idx)

    def _find_hot_sidecar(self, shard_path: Path) -> Path | None:
        cands = sorted(shard_path.parent.glob(f"chronos-glm53-hot*-layer-{self.layer_idx:05d}-of-*.safetensors"))
        return cands[-1] if cands else None

    @torch.no_grad()
    def _build_slots(self, fh: Any, keys: set, tp: str) -> None:
        todo = dict(self.assign)
        n_hot = 0
        if self.hot_path is not None:
            with safe_open(str(self.hot_path), framework="pt", device="cpu") as hf:
                hp = f"{tp}.hot_experts."
                hot_ids = hf.get_tensor(hp + "ids").tolist()
                for j, eid in enumerate(hot_ids):
                    a = todo.get(eid)
                    if a is None or a[0] != T.TIER_T1:
                        continue
                    s_i = a[1]
                    self.gu_bufs[T.TIER_T1][0][s_i].copy_(hf.get_slice(hp + "gu_q")[j])
                    self.gu_bufs[T.TIER_T1][1][s_i].copy_(hf.get_slice(hp + "gu_s")[j])
                    self.dn_bufs[T.TIER_T1][0][s_i].copy_(hf.get_slice(hp + "dn_q")[j])
                    self.dn_bufs[T.TIER_T1][1][s_i].copy_(hf.get_slice(hp + "dn_s")[j])
                    del todo[eid]
                    n_hot += 1
        self.hot_loaded = n_hot
        by_group: dict[int, list[int]] = {}
        for eid in todo:
            by_group.setdefault(eid // GROUP_SIZE, []).append(eid)
        for g_idx, eids in sorted(by_group.items()):
            gp = f"{tp}.chronos_groups.{g_idx}."
            g = {s: fh.get_tensor(gp + s).to(self.device) for s in GROUP_TENSOR_SUFFIXES if gp + s in keys}
            members = [e % GROUP_SIZE for e in eids]
            rec = reconstruct_group_experts(g, g_idx, members, self.hidden, self.inter, include_tail=True)
            for eid, (gu, dn) in zip(eids, rec):
                self._store(eid, gu, dn)
            del g, rec
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    @torch.no_grad()
    def _store(self, eid: int, gu: torch.Tensor, dn: torch.Tensor) -> None:
        tier, s_i = self.assign[eid]
        fmt = T.TIER_T3 if tier == T.TIER_STAGE else tier
        gq, gs = T.PACKERS[fmt](gu.float())
        dq, ds = T.PACKERS[fmt](dn.float())
        if tier == T.TIER_STAGE:
            assert self.host_gu is not None and self.host_dn is not None
            self.host_gu[0][s_i].copy_(gq)
            self.host_gu[1][s_i].copy_(gs)
            self.host_dn[0][s_i].copy_(dq)
            self.host_dn[1][s_i].copy_(ds)
        else:
            self.gu_bufs[tier][0][s_i].copy_(gq)
            self.gu_bufs[tier][1][s_i].copy_(gs)
            self.dn_bufs[tier][0][s_i].copy_(dq)
            self.dn_bufs[tier][1][s_i].copy_(ds)

    def _tier_tensors(self) -> dict[str, torch.Tensor]:
        out = {"tier_map": self.tier_map_cpu}
        for tier, (q, s) in self.gu_bufs.items():
            if tier != T.TIER_STAGE:
                out[f"gu.{tier}.q"], out[f"gu.{tier}.s"] = q, s
        for tier, (q, s) in self.dn_bufs.items():
            if tier != T.TIER_STAGE:
                out[f"dn.{tier}.q"], out[f"dn.{tier}.s"] = q, s
        if self.host_gu is not None and self.host_dn is not None:
            out["gu.host.q"], out["gu.host.s"] = self.host_gu
            out["dn.host.q"], out["dn.host.s"] = self.host_dn
        return out

    def _save_tier_cache(self, path: Path) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            save_file({k: v.detach().cpu().contiguous() for k, v in self._tier_tensors().items()}, str(tmp))
            tmp.replace(path)
            _evict(str(path))
        except Exception as e:  # pragma: no cover - disk dependent
            print(f"[chronos-glm53] tier cache save failed for layer {self.layer_idx}: {e}", flush=True)

    def _load_tier_cache(self, path: Path) -> bool:
        try:
            with safe_open(str(path), framework="pt", device="cpu") as f:
                if not torch.equal(f.get_tensor("tier_map"), self.tier_map_cpu):
                    return False
                for name, dst in self._tier_tensors().items():
                    if name != "tier_map":
                        dst.copy_(f.get_tensor(name))
            _evict(str(path))
            return True
        except Exception:
            return False

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        for _ in weights:
            pass
        return set()

    # ------------------------------------------------------------------------------------------
    def route(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.sigmoid(x.float() @ self.router_w.T)
        choice = scores + self.e_bias + self.absent_bias
        _, ids = torch.topk(choice, self.top_k, dim=-1)
        w = scores.gather(1, ids)
        if self.arch.norm_topk_prob:
            w = w / (w.sum(-1, keepdim=True) + 1e-20)
        return ids, w * self.routed_scaling_factor

    def shared_forward(self, x: torch.Tensor) -> torch.Tensor:
        gu = T.fp8_linear(x, self.sh_gu_w, self.sh_gu_s, out_dtype=torch.float32)
        act = (F.silu(gu[:, : self.sh_inter]) * gu[:, self.sh_inter :]).to(x.dtype)
        return T.fp8_linear(act, self.sh_dn_w, self.sh_dn_s, out_dtype=torch.float32)

    def _stage_host(self, ids_flat: torch.Tensor) -> torch.Tensor:
        """Copies the host-resident experts referenced by `ids_flat` into GPU staging slots; returns a tier map."""
        need = [e for e in dict.fromkeys(ids_flat.tolist()) if e in self.host_ids]
        tmap = self.tier_map
        if not need:
            return tmap
        assert self.host_gu is not None and self.host_dn is not None
        tmap = tmap.clone()
        sg, sd = self.gu_bufs[T.TIER_STAGE], self.dn_bufs[T.TIER_STAGE]
        for j, eid in enumerate(need[: self.stage_slots]):
            hs = self.assign[eid][1]
            sg[0][j].copy_(self.host_gu[0][hs], non_blocking=True)
            sg[1][j].copy_(self.host_gu[1][hs], non_blocking=True)
            sd[0][j].copy_(self.host_dn[0][hs], non_blocking=True)
            sd[1][j].copy_(self.host_dn[1][hs], non_blocking=True)
            tmap[eid] = (T.TIER_STAGE << T.TIER_SHIFT) | j
        return tmap

    def _expert_weights(self, eid: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        a = self.assign.get(eid)
        if a is None:
            return None
        tier, s_i = a
        if tier == T.TIER_STAGE:
            assert self.host_gu is not None and self.host_dn is not None
            gq, gs = (t[s_i].to(self.device, non_blocking=True) for t in self.host_gu)
            dq, ds = (t[s_i].to(self.device, non_blocking=True) for t in self.host_dn)
            fmt = T.TIER_T3
        else:
            gq, gs = (t[s_i] for t in self.gu_bufs[tier])
            dq, ds = (t[s_i] for t in self.dn_bufs[tier])
            fmt = tier
        dt = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        return T.UNPACKERS[fmt](gq, gs, dtype=dt), T.UNPACKERS[fmt](dq, ds, dtype=dt)

    def routed_forward(self, x: torch.Tensor, ids: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        n_tok = x.shape[0]
        pairs = n_tok * self.top_k
        ids_flat = ids.reshape(-1).to(torch.int32)
        if self.use_triton and pairs <= self.gemv_max_pairs and (not self.host_ids or pairs <= self.stage_slots):
            tmap = self._stage_host(ids_flat) if self.host_ids else self.tier_map
            x_rep = x.to(torch.bfloat16).repeat_interleave(self.top_k, dim=0)
            gu = T.tier_gemv(x_rep, ids_flat, tmap, self.gu_bufs, 2 * self.inter)
            act = (F.silu(gu[:, : self.inter]) * gu[:, self.inter :]).to(torch.bfloat16)
            dn = T.tier_gemv(act, ids_flat, tmap, self.dn_bufs, self.hidden)
            return (dn.view(n_tok, self.top_k, self.hidden) * w.unsqueeze(-1)).sum(1)
        # grouped dense path (prefill / CPU / no Triton)
        y = torch.zeros((n_tok, self.hidden), dtype=torch.float32, device=x.device)
        flat_ids = ids.reshape(-1)
        flat_w = w.reshape(-1)
        tok_of = torch.arange(n_tok, device=x.device).repeat_interleave(self.top_k)
        order = torch.argsort(flat_ids)
        sorted_ids = flat_ids[order]
        uniq, counts = torch.unique_consecutive(sorted_ids, return_counts=True)
        start = 0
        for eid, cnt in zip(uniq.tolist(), counts.tolist()):
            sel = order[start : start + cnt]
            start += cnt
            wts = self._expert_weights(int(eid))
            if wts is None:
                continue
            gu_w, dn_w = wts
            xs = x[tok_of[sel]].to(gu_w.dtype)
            gu = F.linear(xs, gu_w).float()
            act = (F.silu(gu[:, : self.inter]) * gu[:, self.inter :]).to(dn_w.dtype)
            y.index_add_(0, tok_of[sel], F.linear(act, dn_w).float() * flat_w[sel].unsqueeze(-1))
        return y

    def forward_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape, orig_dtype = hidden_states.shape, hidden_states.dtype
        x = hidden_states.reshape(-1, self.hidden)
        if x.dtype == torch.float32 and self.device.type == "cuda":
            x = x.to(torch.bfloat16)
        ids, w = self.route(x)
        if self.hits is not None:
            self.hits.index_add_(0, ids.reshape(-1), torch.ones_like(ids.reshape(-1)))
        y = self.routed_forward(x, ids, w) + self.shared_forward(x)
        return y.to(orig_dtype).view(orig_shape)

    def forward(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.forward_impl(hidden_states)

    def extra_repr(self) -> str:
        return (f"layer={self.layer_idx}, experts={self.n_experts}, slots={self.slots}, "
                f"coverage={len(self.assign) / self.n_experts:.3f}, triton={self.use_triton}")

    def resident_bytes(self) -> dict[str, int]:
        gpu = sum(q.numel() + s.numel() * 2 for q, s in list(self.gu_bufs.values()) + list(self.dn_bufs.values()))
        gpu += self.sh_gu_w.numel() + self.sh_dn_w.numel() + self.router_w.numel() * 4
        host = 0
        if self.host_gu is not None and self.host_dn is not None:
            host = sum(q.numel() + s.numel() * 2 for q, s in (self.host_gu, self.host_dn))
        return {"gpu": gpu, "host": host}
