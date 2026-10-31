#!/usr/bin/env python3
"""
Chronos-1T x PseudoBrain (pb27) — Drop-In vLLM `ChronosQwen4ExpSparseMoeBlock` Runtime Layer.

Drop-in replacement for `Qwen4ExpSparseMoeBlock` in `PB27Qwen4ExpForConditionalGeneration`:
1. Holds all 512 routed experts permanently compressed in 64 `ChronosHolographicExpertGroup`
   banks (`736 MB/layer` in Compact mode, `961 MB/layer` in Ultra mode vs `1,350 MB/layer` stock).
2. Maintains an in-place GPU Ring-Buffer (`slot_gu`, `slot_d`) with O(1) CPU slot table
   lookup (`expert_to_slot`), pre-seeded with Global Router Prior hubs and updated on
   cache misses. Avoids 3D weight gather copies by executing `F.linear` directly on in-place
   slot views.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch import nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file

try:
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts as vllm_fused_experts
except Exception:
    vllm_fused_experts = None

import re
from typing import Any, Iterable
from chronos_1t.models.chronos_moe_block import (
    ChronosHolographicExpertGroup,
    build_hadamard_128,
    unpack_nvfp4,
)


def dequantize_mxfp8(weight_fp8: torch.Tensor, scale_e8m0: torch.Tensor, device: torch.device) -> torch.Tensor:
    w = weight_fp8.to(device=device, dtype=torch.float32)
    s_u8 = scale_e8m0.to(device=device, dtype=torch.int32)
    scales = torch.pow(2.0, s_u8.float() - 127.0).unsqueeze(-1)
    w_grouped = w.view(w.shape[0], -1, 32)
    return (w_grouped * scales).view(w.shape[0], -1).to(torch.bfloat16)


class ChronosQwen4ExpSparseMoeBlock(nn.Module):
    """Drop-in replacement for `Qwen4ExpSparseMoeBlock` backed by a `chronos-layer-XXXXX-of-00048.safetensors` shard.
    Supports both direct standalone instantiation and native vLLM `Qwen4ExpSparseMoeBlock(vllm_config, prefix)` instantiation.
    """

    def __init__(
        self,
        layer_idx: int | None = None,
        shard_path: str | Path | None = None,
        device: torch.device | str | None = None,
        top_k: int = 10,
        hot_cache_size: int = 24,
        offload_tail_to_cpu: bool = True,
        mmap_zero_ram: bool = False,
        vllm_config: Any = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if vllm_config is not None:
            if layer_idx is None:
                m = re.search(r"layers\.(\d+)", prefix)
                layer_idx = int(m.group(1)) if m else 0
            if shard_path is None:
                model_dir = Path(vllm_config.model_config.model)
                shard_path = model_dir / f"chronos-layer-{layer_idx:05d}-of-00048.safetensors"
            hf_cfg = getattr(vllm_config.model_config, "hf_config", None)
            chronos_cfg = getattr(hf_cfg, "chronos_holographic_moe", None) or {}
            if isinstance(chronos_cfg, dict):
                hot_cache_size = int(chronos_cfg.get("hot_cache_size", hot_cache_size))
                offload_tail_to_cpu = bool(chronos_cfg.get("offload_tail_to_cpu", offload_tail_to_cpu))
                mmap_zero_ram = bool(chronos_cfg.get("mmap_zero_ram", mmap_zero_ram))

        if layer_idx is None:
            layer_idx = 0
        if shard_path is None:
            shard_path = f"/home/defnotean/pb27/hf/flash_next_chronos_v6/chronos-layer-{layer_idx:05d}-of-00048.safetensors"
        if device is None:
            device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        self.layer_idx = layer_idx
        self.device = device
        self.top_k = top_k
        self.hot_cache_size = max(top_k, hot_cache_size)
        self.offload_tail_to_cpu = offload_tail_to_cpu
        self.mmap_zero_ram = mmap_zero_ram
        self.shard_path_str = str(shard_path)

        tensor_prefix = f"model.language_model.layers.{layer_idx}.mlp"
        self._mmap_handle = safe_open(self.shard_path_str, framework="pt", device="cpu")
        all_keys = set(self._mmap_handle.keys())

        self.router_w = self._mmap_handle.get_tensor(f"{tensor_prefix}.gate.weight").to(device=device, dtype=torch.bfloat16).contiguous()
        self.shared_gate_w = self._mmap_handle.get_tensor(f"{tensor_prefix}.shared_expert_gate.weight").to(device=device, dtype=torch.bfloat16).contiguous()

        sh_g = dequantize_mxfp8(
            self._mmap_handle.get_tensor(f"{tensor_prefix}.shared_expert.gate_proj.weight"),
            self._mmap_handle.get_tensor(f"{tensor_prefix}.shared_expert.gate_proj.weight_scale"),
            device,
        )
        sh_u = dequantize_mxfp8(
            self._mmap_handle.get_tensor(f"{tensor_prefix}.shared_expert.up_proj.weight"),
            self._mmap_handle.get_tensor(f"{tensor_prefix}.shared_expert.up_proj.weight_scale"),
            device,
        )
        self.sh_gu = torch.cat([sh_g, sh_u], dim=0).contiguous()  # (1280, 2560)
        self.sh_down = dequantize_mxfp8(
            self._mmap_handle.get_tensor(f"{tensor_prefix}.shared_expert.down_proj.weight"),
            self._mmap_handle.get_tensor(f"{tensor_prefix}.shared_expert.down_proj.weight_scale"),
            device,
        ).contiguous()  # (2560, 640)

        self.hub_experts_cpu: dict[int, dict[str, torch.Tensor]] = {}
        hub_prefix = f"{tensor_prefix}.hub_experts."
        for k in all_keys:
            if k.startswith(hub_prefix):
                rest = k[len(hub_prefix):]
                eid_str, tensor_name = rest.split(".", 1)
                eid = int(eid_str)
                self.hub_experts_cpu.setdefault(eid, {})[tensor_name] = self._mmap_handle.get_tensor(k)

        shared_h128 = build_hadamard_128(device)
        self.groups = nn.ModuleList()
        for g_idx in range(64):
            gp = f"{tensor_prefix}.chronos_groups.{g_idx}."
            sub_t: dict[str, torch.Tensor] = {}
            for k_suffix in (
                "s_gu.weight", "s_gu.weight_scale", "s_gu.weight_scale_2", "s_gu.beta",
                "s_down.weight", "s_down.weight_scale", "s_down.weight_scale_2", "s_down.beta",
                "healer.u_gu", "healer.v_gu", "healer.u_down", "healer.v_down",
                "tail.gu_bits", "tail.down_bits",
            ):
                full_k = f"{gp}{k_suffix}"
                if full_k in all_keys:
                    sub_t[k_suffix] = self._mmap_handle.get_tensor(full_k)
            if not mmap_zero_ram:
                for k_suffix in ("tail.gu_packed", "tail.gu_scale", "tail.gu_mask", "tail.down_packed", "tail.down_scale", "tail.down_mask"):
                    full_k = f"{gp}{k_suffix}"
                    if full_k in all_keys:
                        sub_t[k_suffix] = self._mmap_handle.get_tensor(full_k)
            self.groups.append(
                ChronosHolographicExpertGroup(
                    group_idx=g_idx,
                    tensors=sub_t,
                    device=device,
                    offload_tail_to_cpu=offload_tail_to_cpu,
                    mmap_zero_ram=mmap_zero_ram,
                    mmap_handle=self._mmap_handle if mmap_zero_ram else None,
                    mmap_prefix=gp,
                    h128=shared_h128,
                )
            )
        if not mmap_zero_ram:
            self._mmap_handle = None
        if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
            try:
                fd = os.open(self.shard_path_str, os.O_RDONLY)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                os.close(fd)
            except OSError:
                pass

        # Pre-allocated in-place GPU Ring-Buffer views (zero weight gather copy at runtime)
        self.cache_gu_buf = torch.empty((self.hot_cache_size, 1280, 2560), dtype=torch.bfloat16, device=device)
        self.cache_d_buf = torch.empty((self.hot_cache_size, 2560, 640), dtype=torch.bfloat16, device=device)
        self.slot_gu = [self.cache_gu_buf[s] for s in range(self.hot_cache_size)]
        self.slot_d = [self.cache_d_buf[s] for s in range(self.hot_cache_size)]

        self.expert_to_slot: list[int] = [-1] * 512
        self.expert_to_slot_gpu = torch.full((512,), -1, dtype=torch.int32, device=device)
        self.slot_to_expert: list[int] = [-1] * self.hot_cache_size
        self.ring_cursor = 0

        self.cache_hits = 0
        self.cache_misses = 0

        # Pre-seed hot cache with top Global Router Prior hub experts
        with torch.no_grad():
            rw_f = self.router_w.float()
            _, _, vh_rt = torch.linalg.svd(rw_f, full_matrices=False)
            u_ctx = vh_rt[:32, :].T
            prior_scores = torch.linalg.norm(rw_f @ u_ctx, dim=-1)
            _, top_hub_ids = torch.topk(prior_scores, k=min(16, self.hot_cache_size // 2))
            for eid in top_hub_ids.tolist():
                self._ensure_expert_slot(int(eid))
            self.cache_hits = 0
            self.cache_misses = 0

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Consumes vLLM AutoWeightsLoader weight stream for this `.mlp` submodule."""
        for _ in weights:
            pass
        return {name for name, _ in self.named_parameters(recurse=True)}

    @torch.no_grad()
    def _ensure_expert_slot(self, expert_id: int) -> int:
        slot = self.expert_to_slot[expert_id]
        if slot >= 0:
            self.cache_hits += 1
            return slot

        self.cache_misses += 1
        slot = self.ring_cursor
        self.ring_cursor = (self.ring_cursor + 1) % self.hot_cache_size

        old_eid = self.slot_to_expert[slot]
        if old_eid >= 0:
            self.expert_to_slot[old_eid] = -1
            self.expert_to_slot_gpu[old_eid] = -1

        if expert_id in self.hub_experts_cpu:
            ht = self.hub_experts_cpu[expert_id]
            gw = unpack_nvfp4(
                ht["gate_proj.weight"].to(self.device, non_blocking=True),
                ht["gate_proj.weight_scale"].to(self.device, non_blocking=True),
                ht["gate_proj.weight_scale_2"].to(self.device, non_blocking=True),
            )
            uw = unpack_nvfp4(
                ht["up_proj.weight"].to(self.device, non_blocking=True),
                ht["up_proj.weight_scale"].to(self.device, non_blocking=True),
                ht["up_proj.weight_scale_2"].to(self.device, non_blocking=True),
            )
            dw = unpack_nvfp4(
                ht["down_proj.weight"].to(self.device, non_blocking=True),
                ht["down_proj.weight_scale"].to(self.device, non_blocking=True),
                ht["down_proj.weight_scale_2"].to(self.device, non_blocking=True),
            )
        else:
            g_idx = expert_id // 8
            local_m = expert_id % 8
            gw, uw, dw = self.groups[g_idx].reconstruct_expert(local_m)

        self.slot_gu[slot].copy_(torch.cat([gw, uw], dim=0).to(torch.bfloat16))
        self.slot_d[slot].copy_(dw.to(torch.bfloat16))

        self.expert_to_slot[expert_id] = slot
        self.expert_to_slot_gpu[expert_id] = slot
        self.slot_to_expert[slot] = expert_id
        return slot

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        orig_dtype = hidden_states.dtype
        x = hidden_states.view(-1, orig_shape[-1]).to(torch.bfloat16)
        n_toks = x.shape[0]

        # 1. Shared Expert Branch (fused gate_up GEMV)
        sh_score = torch.sigmoid(F.linear(x, self.shared_gate_w))
        sh_gu_out = F.linear(x, self.sh_gu)
        y_shared = sh_score * F.linear(F.silu(sh_gu_out[:, :640]) * sh_gu_out[:, 640:], self.sh_down)

        # 2. Routed Top-10 MoE Branch (FP32 router gate for batch-invariant expert selection)
        router_logits = F.linear(x.float(), self.router_w.float())
        top_vals, top_ids = torch.topk(router_logits, k=self.top_k, dim=-1)
        top_gates_f32 = F.softmax(top_vals, dim=-1)
        top_gates = top_gates_f32.to(torch.bfloat16)

        if n_toks == 1:
            eids = top_ids[0].tolist()
            gates = top_gates[0]
            y_routed = torch.zeros_like(x)
            for k_idx, eid in enumerate(eids):
                slot = self._ensure_expert_slot(int(eid))
                gu_out = F.linear(x, self.slot_gu[slot])
                y_exp = F.linear(F.silu(gu_out[:, :640]) * gu_out[:, 640:], self.slot_d[slot])
                y_routed.addcmul_(gates[k_idx : k_idx + 1], y_exp)
            return (y_routed + y_shared).view(orig_shape).to(orig_dtype)

        # Multi-token prefill / batch execution with Frequency-Gated Non-Thrashing Hot Cache
        y_routed = torch.zeros_like(x)
        flat_ids = top_ids.view(-1)
        flat_gates = top_gates.view(-1).float()
        expert_mass = torch.zeros(512, dtype=torch.float32, device=self.device)
        expert_mass.scatter_add_(0, flat_ids, flat_gates)
        expert_mass.scatter_add_(0, top_ids[0], torch.full((self.top_k,), 100.0, dtype=torch.float32, device=self.device))

        unique_eids = torch.unique(top_ids).tolist()
        warm_eids = [int(eid) for eid in unique_eids if self.expert_to_slot[int(eid)] >= 0]
        cold_eids = [int(eid) for eid in unique_eids if self.expert_to_slot[int(eid)] < 0]

        for eid in warm_eids:
            slot = self._ensure_expert_slot(eid)
            mask = top_ids == eid
            tok_indices, slot_indices = torch.where(mask)
            x_sel = x[tok_indices]
            g_sel = top_gates[tok_indices, slot_indices].unsqueeze(-1)
            gu_out = F.linear(x_sel, self.slot_gu[slot])
            act = F.silu(gu_out[:, :640]) * gu_out[:, 640:]
            y_exp = F.linear(act, self.slot_d[slot])
            y_routed.index_add_(0, tok_indices, g_sel * y_exp)

        if cold_eids:
            cold_masses = [float(expert_mass[eid].item()) for eid in cold_eids]
            sorted_cold = [eid for _, eid in sorted(zip(cold_masses, cold_eids), reverse=True)]
            n_admit = min(len(sorted_cold), max(self.top_k, self.hot_cache_size - len(warm_eids)))
            admit_set = set(sorted_cold[:n_admit])

            by_group: dict[int, list[int]] = {}
            for eid in cold_eids:
                by_group.setdefault(eid // 8, []).append(eid)

            admitted_ordered: list[tuple[float, int, torch.Tensor, torch.Tensor]] = []
            for g_idx, g_eids in by_group.items():
                grp = self.groups[g_idx]
                s_gu_c = unpack_nvfp4(grp.s_gu_packed, grp.s_gu_scale, grp.s_gu_scale2)
                s_dn_c = unpack_nvfp4(grp.s_down_packed, grp.s_down_scale, grp.s_down_scale2)
                for eid in g_eids:
                    gw, uw, dw = grp.reconstruct_expert(eid % 8, s_gu_cached=s_gu_c, s_down_cached=s_dn_c)
                    gu_w = torch.cat([gw, uw], dim=0).to(torch.bfloat16)
                    dw_bf16 = dw.to(torch.bfloat16)
                    mask = top_ids == eid
                    tok_indices, slot_indices = torch.where(mask)
                    x_sel = x[tok_indices]
                    g_sel = top_gates[tok_indices, slot_indices].unsqueeze(-1)
                    gu_out = F.linear(x_sel, gu_w)
                    act = F.silu(gu_out[:, :640]) * gu_out[:, 640:]
                    y_exp = F.linear(act, dw_bf16)
                    y_routed.index_add_(0, tok_indices, g_sel * y_exp)
                    if eid in admit_set:
                        admitted_ordered.append((float(expert_mass[eid].item()), eid, gu_w, dw_bf16))

            for _, eid, gu_w, dw_bf16 in sorted(admitted_ordered, key=lambda item: item[0]):
                self.cache_misses += 1
                slot = self.ring_cursor
                self.ring_cursor = (self.ring_cursor + 1) % self.hot_cache_size
                old_eid = self.slot_to_expert[slot]
                if old_eid >= 0:
                    self.expert_to_slot[old_eid] = -1
                    self.expert_to_slot_gpu[old_eid] = -1
                self.slot_gu[slot].copy_(gu_w)
                self.slot_d[slot].copy_(dw_bf16)
                self.expert_to_slot[eid] = slot
                self.expert_to_slot_gpu[eid] = slot
                self.slot_to_expert[slot] = eid

        return (y_routed + y_shared).view(orig_shape).to(orig_dtype)


def main() -> None:
    import time

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    shard = "/home/defnotean/pb27/hf/flash_next_chronos_v6/chronos-layer-00000-of-00048.safetensors"
    t0 = time.perf_counter()
    block = ChronosQwen4ExpSparseMoeBlock(
        layer_idx=0,
        shard_path=shard,
        device=device,
        hot_cache_size=64,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"Loaded ChronosQwen4ExpSparseMoeBlock (512 experts, 64 groups) in {time.perf_counter() - t0:.2f}s")

    torch.manual_seed(42)
    x_prefill = torch.randn(32, 2560, device=device, dtype=torch.bfloat16) * 0.05
    y_pre = block(x_prefill)
    print(f"Prefill 32 tokens complete | Cache hits={block.cache_hits} misses={block.cache_misses}")

    x_dec = x_prefill[:1]
    for _ in range(10):
        _ = block(x_dec)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    for _ in range(200):
        y_dec = block(x_dec)
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t1) * 1000.0 / 200.0
    cos = F.cosine_similarity(y_pre[:1].float(), y_dec.float(), dim=-1).item()
    print(
        f"Single-Token Decode Latency: {ms:.3f} ms/layer | "
        f"Prefill-vs-Decode Cos: {cos * 100:.4f}% | "
        f"Hits={block.cache_hits} Misses={block.cache_misses}"
    )


if __name__ == "__main__":
    main()
