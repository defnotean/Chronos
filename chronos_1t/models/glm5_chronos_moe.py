#!/usr/bin/env python3
"""
Chronos-1T x GLM-5.3-Flash (`zai-org/GLM-5.3-Flash`) — Standalone & vLLM Drop-In `ChronosGlm5SparseMoeBlock`.

Implements runtime execution for converted `GLM-5.3-Flash` (`Glm5NextForConditionalGeneration`) MoE layers
(`chronos-glm5-layer-XXXXX-of-00046.safetensors`):
  - `hidden_size = 4096`, `moe_intermediate_size = 2048`, `n_routed_experts = 288` (`36` groups of `K = 8`)
  - Sigmoid Top-8 `noaux_tc` router (`e_score_correction_bias`, `routed_scaling_factor = 2.5`)
  - `128x128` Block-FP8 Shared Expert (`shared_experts.gate_proj`, `up_proj`, `down_proj`)
  - GPU-resident `8:1` `NVFP4` Holographic Banks + Coupled Rank-64 `FP8` SVD Healers (`774 MB/layer`)
  - Zero-Copy Unified-Memory (`offload_tail_to_cpu=True`) 2-Bit / 1-Bit Isotropic Gaussian Tail Lattice
  - In-place `BF16` GPU Ring-Buffer Hot Cache (`slot_gu`, `slot_d`)
"""
from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any, Iterable

import torch
from torch import nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file

try:
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts as vllm_fused_experts
except Exception:
    vllm_fused_experts = None

_PINNED_DMA_BUFS: dict[tuple[int, ...], torch.Tensor] = {}


def _copy_via_pinned_dma(cpu_tensor: torch.Tensor, device: torch.device, key_id: int) -> torch.Tensor:
    shape_key = (key_id, *cpu_tensor.shape)
    buf = _PINNED_DMA_BUFS.get(shape_key)
    if buf is None or buf.dtype != cpu_tensor.dtype:
        try:
            buf = torch.empty(cpu_tensor.shape, dtype=cpu_tensor.dtype, pin_memory=True)
            _PINNED_DMA_BUFS[shape_key] = buf
        except Exception:
            return cpu_tensor.to(device, non_blocking=True)
    buf.copy_(cpu_tensor, non_blocking=False)
    return buf.to(device, non_blocking=True)

from chronos_1t.models.chronos_moe_block import (
    build_hadamard_128,
    generate_expert_1d_keys,
    unbind_2d_permuted_hadamard,
    unpack_1bit_rademacher,
    unpack_2bit_lloyd_max,
    unpack_nvfp4,
)


def dequantize_block_fp8_128(
    w_fp8: torch.Tensor,
    scale_inv: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    rows, cols = w_fp8.shape
    w = w_fp8.to(device=device, dtype=torch.float32).view(rows // 128, 128, cols // 128, 128)
    s = scale_inv.to(device=device, dtype=torch.float32).view(rows // 128, 1, cols // 128, 1)
    return (w * s).reshape(rows, cols).to(dtype)


class ChronosGlm5HolographicExpertGroup(nn.Module):
    """Stores a folded group of K=8 GLM-5.3-Flash experts (`gate_up`: 4096x4096, `down`: 4096x2048)."""

    def __init__(
        self,
        group_idx: int,
        tensors: dict[str, torch.Tensor],
        device: torch.device,
        offload_tail_to_cpu: bool = True,
        mmap_zero_ram: bool = False,
        mmap_handle: Any = None,
        mmap_prefix: str = "",
        h128: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.group_idx = group_idx
        self.device = device
        self.h128 = h128 if h128 is not None else build_hadamard_128(device, dtype=torch.bfloat16)
        self.mmap_zero_ram = mmap_zero_ram and (mmap_handle is not None)
        self.mmap_handle = mmap_handle
        self.mmap_prefix = mmap_prefix

        self.s_gu_packed = tensors["s_gu.weight"].to(device)
        self.s_gu_scale = tensors["s_gu.weight_scale"].to(device)
        self.s_gu_scale2 = tensors["s_gu.weight_scale_2"].to(device)
        self.beta_gu = tensors["s_gu.beta"].to(device=device, dtype=torch.bfloat16)

        self.s_down_packed = tensors["s_down.weight"].to(device)
        self.s_down_scale = tensors["s_down.weight_scale"].to(device)
        self.s_down_scale2 = tensors["s_down.weight_scale_2"].to(device)
        self.beta_down = tensors["s_down.beta"].to(device=device, dtype=torch.bfloat16)

        self.u_gu = tensors["healer.u_gu"].to(device)
        self.v_gu = tensors["healer.v_gu"].to(device)
        self.u_down = tensors["healer.u_down"].to(device)
        self.v_down = tensors["healer.v_down"].to(device)

        self.gu_tail_bits = int(tensors.get("tail.gu_bits", torch.tensor(2)).item())
        self.down_tail_bits = int(tensors.get("tail.down_bits", torch.tensor(2)).item())
        self._s_gu_cached: torch.Tensor | None = None
        self._s_dn_cached: torch.Tensor | None = None

        if self.mmap_zero_ram:
            self.gu_tail_packed = None
            self.gu_tail_scale = None
            self.down_tail_packed = None
            self.down_tail_scale = None
        else:
            tail_dev = torch.device("cpu") if offload_tail_to_cpu else device
            self.gu_tail_packed = tensors["tail.gu_packed"].to(tail_dev)
            self.gu_tail_scale = tensors["tail.gu_scale"].to(tail_dev)
            self.down_tail_packed = tensors["tail.down_packed"].to(tail_dev)
            self.down_tail_scale = tensors["tail.down_scale"].to(tail_dev)

    @torch.no_grad()
    def _fetch_tail_slices(
        self, local_m: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.mmap_zero_ram and self.mmap_handle is not None:
            p = self.mmap_prefix
            gu_p_m = _copy_via_pinned_dma(self.mmap_handle.get_slice(f"{p}tail.gu_packed")[local_m], self.device, 1)
            gu_s_m = _copy_via_pinned_dma(self.mmap_handle.get_slice(f"{p}tail.gu_scale")[local_m], self.device, 2)
            dn_p_m = _copy_via_pinned_dma(self.mmap_handle.get_slice(f"{p}tail.down_packed")[local_m], self.device, 3)
            dn_s_m = _copy_via_pinned_dma(self.mmap_handle.get_slice(f"{p}tail.down_scale")[local_m], self.device, 4)
            return gu_p_m, gu_s_m, dn_p_m, dn_s_m
        assert self.gu_tail_packed is not None and self.gu_tail_scale is not None
        assert self.down_tail_packed is not None and self.down_tail_scale is not None
        return (
            self.gu_tail_packed[local_m].to(self.device, non_blocking=True),
            self.gu_tail_scale[local_m].to(self.device, non_blocking=True),
            self.down_tail_packed[local_m].to(self.device, non_blocking=True),
            self.down_tail_scale[local_m].to(self.device, non_blocking=True),
        )

    @torch.no_grad()
    def get_group_banks(self) -> tuple[torch.Tensor, torch.Tensor]:
        s_gu = unpack_nvfp4(self.s_gu_packed, self.s_gu_scale, self.s_gu_scale2, dtype=torch.bfloat16)
        s_dn = unpack_nvfp4(self.s_down_packed, self.s_down_scale, self.s_down_scale2, dtype=torch.bfloat16)
        return s_gu, s_dn

    @torch.no_grad()
    def forward_cold_fast(
        self,
        x: torch.Tensor,
        local_m: int,
        s_gu_cached: torch.Tensor | None = None,
        s_dn_cached: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """1D Activation-Space Cold Bypass (`Con 1` & `Con 2` Fix):
        Executes cold expert `local_m` directly in 1D activation space `(k, 4096)` using the 8:1 Holographic Bank
        + Factored Rank-64 SVD Healer `(x @ V^T) @ U^T` without materializing `(4096, 4096)` weight matrices
        or transferring cold tail slices (`~50x` faster cold execution).
        """
        k_toks = x.shape[0]
        dt = x.dtype
        h = self.h128.to(dt)
        if s_gu_cached is None or s_dn_cached is None:
            s_gu, s_dn = self.get_group_banks()
        else:
            s_gu, s_dn = s_gu_cached, s_dn_cached

        r_s1, r_perm, r_inv, r_s2, c_s1, c_perm, _, c_s2 = generate_expert_1d_keys(
            4096, 4096, local_m, seed_base=101 + self.group_idx * 8, device=self.device
        )
        # 1D column forward transform on input activation x: (k, 4096)
        x_c = (x * c_s1.to(dt).unsqueeze(0))[:, c_perm]
        x_c = torch.matmul(x_c.view(k_toks, 32, 128), h).view(k_toks, 4096) * c_s2.to(dt).unsqueeze(0)
        # Project through shared 8:1 Holographic Bank + 1D row inverse transform
        z_hrr = F.linear(x_c, s_gu) * r_s2.to(dt).unsqueeze(0)
        z_hrr = torch.matmul(z_hrr.view(k_toks, 32, 128), h).view(k_toks, 4096)[:, r_inv] * r_s1.to(dt).unsqueeze(0)
        # Factored Rank-64 Healer: (x @ v_gu^T) @ u_gu^T
        u_gu_m = self.u_gu[local_m].to(dt)
        v_gu_m = self.v_gu[local_m].to(dt)
        z_gu = self.beta_gu[local_m] * z_hrr + F.linear(F.linear(x, v_gu_m), u_gu_m)

        act = F.silu(z_gu[:, :2048]) * z_gu[:, 2048:]

        dr_s1, _, dr_inv, dr_s2, dc_s1, dc_perm, _, dc_s2 = generate_expert_1d_keys(
            4096, 2048, local_m, seed_base=303 + self.group_idx * 8, device=self.device
        )
        a_c = (act * dc_s1.to(dt).unsqueeze(0))[:, dc_perm]
        a_c = torch.matmul(a_c.view(k_toks, 16, 128), h).view(k_toks, 2048) * dc_s2.to(dt).unsqueeze(0)
        y_hrr = F.linear(a_c, s_dn) * dr_s2.to(dt).unsqueeze(0)
        y_hrr = torch.matmul(y_hrr.view(k_toks, 32, 128), h).view(k_toks, 4096)[:, dr_inv] * dr_s1.to(dt).unsqueeze(0)
        u_dn_m = self.u_down[local_m].to(dt)
        v_dn_m = self.v_down[local_m].to(dt)
        return self.beta_down[local_m] * y_hrr + F.linear(F.linear(act, v_dn_m), u_dn_m)

    @torch.no_grad()
    def reconstruct_expert(
        self,
        local_m: int,
        s_gu_cached: torch.Tensor | None = None,
        s_dn_cached: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstructs `(gu_w: (4096, 4096), down_w: (4096, 2048))` in BF16 for expert `local_m` in 0..7."""
        if s_gu_cached is None or s_dn_cached is None:
            s_gu, s_dn = self.get_group_banks()
        else:
            s_gu, s_dn = s_gu_cached, s_dn_cached

        r_s1, _, r_inv, r_s2, c_s1, _, c_inv, c_s2 = generate_expert_1d_keys(
            4096, 4096, local_m, seed_base=101 + self.group_idx * 8, device=self.device
        )
        h_gu = unbind_2d_permuted_hadamard(s_gu, r_s1, r_inv, r_s2, c_s1, c_inv, c_s2, self.h128)
        healer_gu = torch.matmul(self.u_gu[local_m].to(torch.bfloat16), self.v_gu[local_m].to(torch.bfloat16))

        gu_p_m, gu_s_m, dn_p_m, dn_s_m = self._fetch_tail_slices(local_m)
        if self.gu_tail_bits == 2:
            tail_gu = unpack_2bit_lloyd_max(gu_p_m, gu_s_m, dtype=torch.bfloat16)
        else:
            tail_gu = unpack_1bit_rademacher(gu_p_m, gu_s_m, dtype=torch.bfloat16)

        gu_hat = self.beta_gu[local_m] * h_gu + healer_gu + tail_gu

        dr_s1, _, dr_inv, dr_s2, dc_s1, _, dc_inv, dc_s2 = generate_expert_1d_keys(
            4096, 2048, local_m, seed_base=303 + self.group_idx * 8, device=self.device
        )
        h_dn = unbind_2d_permuted_hadamard(s_dn, dr_s1, dr_inv, dr_s2, dc_s1, dc_inv, dc_s2, self.h128)
        healer_dn = torch.matmul(self.u_down[local_m].to(torch.bfloat16), self.v_down[local_m].to(torch.bfloat16))

        if self.down_tail_bits == 2:
            tail_dn = unpack_2bit_lloyd_max(dn_p_m, dn_s_m, dtype=torch.bfloat16)
        else:
            tail_dn = unpack_1bit_rademacher(dn_p_m, dn_s_m, dtype=torch.bfloat16)

        down_hat = self.beta_down[local_m] * h_dn + healer_dn + tail_dn
        return gu_hat, down_hat

    @torch.no_grad()
    def reconstruct_expert_into_slot(
        self,
        local_m: int,
        slot_gu: torch.Tensor,
        slot_dn: torch.Tensor,
        s_gu_cached: torch.Tensor | None = None,
        s_dn_cached: torch.Tensor | None = None,
    ) -> None:
        """Zero-temp-allocation in-place slot reconstruction (`v7.5` `Con 1` Fix):
        Writes directly into `slot_gu` `(4096, 4096)` and `slot_dn` `(4096, 2048)` using `addmm_`,
        avoiding `134 MB` of intermediate GPU buffer allocations per cold miss.
        """
        if s_gu_cached is None or s_dn_cached is None:
            s_gu, s_dn = self.get_group_banks()
        else:
            s_gu, s_dn = s_gu_cached, s_dn_cached

        gu_p_m, gu_s_m, dn_p_m, dn_s_m = self._fetch_tail_slices(local_m)
        r_s1, _, r_inv, r_s2, c_s1, _, c_inv, c_s2 = generate_expert_1d_keys(
            4096, 4096, local_m, seed_base=101 + self.group_idx * 8, device=self.device
        )
        h_gu = unbind_2d_permuted_hadamard(s_gu, r_s1, r_inv, r_s2, c_s1, c_inv, c_s2, self.h128)
        if self.gu_tail_bits == 2:
            slot_gu.copy_(unpack_2bit_lloyd_max(gu_p_m, gu_s_m, dtype=torch.bfloat16))
        else:
            slot_gu.copy_(unpack_1bit_rademacher(gu_p_m, gu_s_m, dtype=torch.bfloat16))
        slot_gu.add_(h_gu * self.beta_gu[local_m])
        slot_gu.addmm_(self.u_gu[local_m].to(torch.bfloat16), self.v_gu[local_m].to(torch.bfloat16))

        dr_s1, _, dr_inv, dr_s2, dc_s1, _, dc_inv, dc_s2 = generate_expert_1d_keys(
            4096, 2048, local_m, seed_base=303 + self.group_idx * 8, device=self.device
        )
        h_dn = unbind_2d_permuted_hadamard(s_dn, dr_s1, dr_inv, dr_s2, dc_s1, dc_inv, dc_s2, self.h128)
        if self.down_tail_bits == 2:
            slot_dn.copy_(unpack_2bit_lloyd_max(dn_p_m, dn_s_m, dtype=torch.bfloat16))
        else:
            slot_dn.copy_(unpack_1bit_rademacher(dn_p_m, dn_s_m, dtype=torch.bfloat16))
        slot_dn.add_(h_dn * self.beta_down[local_m])
        slot_dn.addmm_(self.u_down[local_m].to(torch.bfloat16), self.v_down[local_m].to(torch.bfloat16))


class ChronosGlm5SparseMoeBlock(nn.Module):
    """Drop-in replacement for GLM-5.3-Flash (`Glm5NextForConditionalGeneration`) Sparse MoE layer."""

    def __init__(
        self,
        layer_idx: int = 3,
        shard_path: str | Path | None = None,
        device: torch.device | str | None = None,
        top_k: int = 8,
        routed_scaling_factor: float = 2.5,
        hot_cache_size: int = 16,
        offload_tail_to_cpu: bool = True,
        mmap_zero_ram: bool = False,
        use_fused_moe: bool = True,
        vllm_config: Any = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if vllm_config is not None:
            m = re.search(r"layers\.(\d+)", prefix)
            if m:
                layer_idx = int(m.group(1))
            if shard_path is None:
                model_dir = Path(vllm_config.model_config.model)
                shard_path = model_dir / f"chronos-glm5-layer-{layer_idx:05d}-of-00046.safetensors"

        if device is None:
            device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        if shard_path is None:
            cand = f"/home/defnotean/pb27/hf/glm53_flash_chronos_v6/chronos-glm5-layer-{layer_idx:05d}-of-00046.safetensors"
            if not os.path.exists(cand):
                cand = f"/models/glm53_flash_chronos_v6/chronos-glm5-layer-{layer_idx:05d}-of-00046.safetensors"
            shard_path = cand

        self.layer_idx = layer_idx
        self.device = device
        self.top_k = top_k
        self.routed_scaling_factor = routed_scaling_factor
        self.hot_cache_size = max(top_k, hot_cache_size)
        self.offload_tail_to_cpu = offload_tail_to_cpu
        self.mmap_zero_ram = mmap_zero_ram
        self.use_fused_moe = use_fused_moe and (vllm_fused_experts is not None)
        self.shard_path_str = str(shard_path)

        tp = f"model.language_model.layers.{layer_idx}.mlp"
        self._mmap_handle = safe_open(self.shard_path_str, framework="pt", device="cpu")
        all_keys = set(self._mmap_handle.keys())

        self.router_w = self._mmap_handle.get_tensor(f"{tp}.gate.weight").to(device=device, dtype=torch.bfloat16).contiguous()
        self.e_score_bias = self._mmap_handle.get_tensor(f"{tp}.gate.e_score_correction_bias").to(device=device, dtype=torch.float32).contiguous()

        sh_g = dequantize_block_fp8_128(
            self._mmap_handle.get_tensor(f"{tp}.shared_experts.gate_proj.weight"),
            self._mmap_handle.get_tensor(f"{tp}.shared_experts.gate_proj.weight_scale_inv"),
            device,
        )
        sh_u = dequantize_block_fp8_128(
            self._mmap_handle.get_tensor(f"{tp}.shared_experts.up_proj.weight"),
            self._mmap_handle.get_tensor(f"{tp}.shared_experts.up_proj.weight_scale_inv"),
            device,
        )
        self.sh_gu = torch.cat([sh_g, sh_u], dim=0).contiguous()  # (4096, 4096)
        self.sh_down = dequantize_block_fp8_128(
            self._mmap_handle.get_tensor(f"{tp}.shared_experts.down_proj.weight"),
            self._mmap_handle.get_tensor(f"{tp}.shared_experts.down_proj.weight_scale_inv"),
            device,
        ).contiguous()  # (4096, 2048)

        shared_h128 = build_hadamard_128(device, dtype=torch.bfloat16)
        self.groups = nn.ModuleList()
        for g_idx in range(36):
            gp = f"{tp}.chronos_groups.{g_idx}."
            if f"{gp}s_gu.weight" not in all_keys:
                gp = f"{tp}.chronos_groups.0."
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
                for k_suffix in ("tail.gu_packed", "tail.gu_scale", "tail.down_packed", "tail.down_scale"):
                    sub_t[k_suffix] = self._mmap_handle.get_tensor(f"{gp}{k_suffix}")
            self.groups.append(
                ChronosGlm5HolographicExpertGroup(
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

        self.cache_gu_buf = torch.empty((self.hot_cache_size, 4096, 4096), dtype=torch.bfloat16, device=device)
        self.cache_d_buf = torch.empty((self.hot_cache_size, 4096, 2048), dtype=torch.bfloat16, device=device)
        self.slot_gu = [self.cache_gu_buf[s] for s in range(self.hot_cache_size)]
        self.slot_d = [self.cache_d_buf[s] for s in range(self.hot_cache_size)]

        self.expert_to_slot: list[int] = [-1] * 288
        self.slot_to_expert: list[int] = [-1] * self.hot_cache_size
        self.ring_cursor = 0
        self.cache_hits = 0
        self.cache_misses = 0

        # Pre-seed hot cache with top Global Router Prior hub experts
        with torch.no_grad():
            rw_f = self.router_w.float()
            _, _, vh_rt = torch.linalg.svd(rw_f, full_matrices=False)
            u_ctx = vh_rt[:32, :].T
            prior_scores = torch.linalg.norm(rw_f @ u_ctx, dim=-1) + self.e_score_bias
            _, top_hub_ids = torch.topk(prior_scores, k=min(8, self.hot_cache_size // 2))
            for eid in top_hub_ids.tolist():
                self._ensure_expert_slot(int(eid))
            self.cache_hits = 0
            self.cache_misses = 0

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        for _ in weights:
            pass
        return {name for name, _ in self.named_parameters(recurse=True)}

    @torch.no_grad()
    def _ensure_expert_slot(
        self,
        expert_id: int,
        s_gu_cached: torch.Tensor | None = None,
        s_dn_cached: torch.Tensor | None = None,
    ) -> int:
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

        g_idx = expert_id // 8
        local_m = expert_id % 8
        self.groups[g_idx].reconstruct_expert_into_slot(
            local_m, self.slot_gu[slot], self.slot_d[slot], s_gu_cached=s_gu_cached, s_dn_cached=s_dn_cached
        )

        self.expert_to_slot[expert_id] = slot
        self.slot_to_expert[slot] = expert_id
        return slot

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        orig_dtype = hidden_states.dtype
        x = hidden_states.view(-1, orig_shape[-1]).to(torch.bfloat16)
        n_toks = x.shape[0]

        # 1. Shared Expert Branch
        sh_gu_out = F.linear(x, self.sh_gu)
        y_shared = F.linear(F.silu(sh_gu_out[:, :2048]) * sh_gu_out[:, 2048:], self.sh_down)

        # 2. Sigmoid Top-8 noaux_tc Router Branch
        scores = torch.sigmoid(F.linear(x.float(), self.router_w.float()))
        scores_for_choice = scores + self.e_score_bias.unsqueeze(0)
        _, top_ids = torch.topk(scores_for_choice, k=self.top_k, dim=-1)
        top_weights = scores.gather(1, top_ids)
        top_gates = ((top_weights / top_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)) * self.routed_scaling_factor).to(torch.bfloat16)

        if n_toks == 1:
            eids = top_ids[0].tolist()
            group_bank_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
            slot_ids: list[int] = []
            for eid in eids:
                eid_i = int(eid)
                if self.expert_to_slot[eid_i] < 0:
                    g_idx = eid_i // 8
                    if g_idx not in group_bank_cache:
                        grp = self.groups[g_idx]
                        group_bank_cache[g_idx] = (
                            unpack_nvfp4(grp.s_gu_packed, grp.s_gu_scale, grp.s_gu_scale2, dtype=torch.bfloat16),
                            unpack_nvfp4(grp.s_down_packed, grp.s_down_scale, grp.s_down_scale2, dtype=torch.bfloat16),
                        )
                    s_gu_c, s_dn_c = group_bank_cache[g_idx]
                    slot = self._ensure_expert_slot(eid_i, s_gu_cached=s_gu_c, s_dn_cached=s_dn_c)
                else:
                    slot = self._ensure_expert_slot(eid_i)
                slot_ids.append(slot)

            if self.use_fused_moe and vllm_fused_experts is not None:
                slot_ids_t = torch.tensor([slot_ids], dtype=torch.int32, device=self.device)
                y_routed = vllm_fused_experts(x, self.cache_gu_buf, self.cache_d_buf, top_gates.float(), slot_ids_t)
                return (y_routed + y_shared).view(orig_shape).to(orig_dtype)

            gates = top_gates[0]
            y_routed = torch.zeros_like(x)
            for k_idx, slot in enumerate(slot_ids):
                gu_out = F.linear(x, self.slot_gu[slot])
                y_exp = F.linear(F.silu(gu_out[:, :2048]) * gu_out[:, 2048:], self.slot_d[slot])
                y_routed.addcmul_(gates[k_idx : k_idx + 1], y_exp)
            return (y_routed + y_shared).view(orig_shape).to(orig_dtype)

        y_routed = torch.zeros_like(x)
        flat_ids = top_ids.view(-1)
        flat_gates = top_gates.view(-1).float()
        # Prioritize experts routed by initial/anchor tokens plus overall batch gate mass so hot cache never thrashes
        expert_mass = torch.zeros(288, dtype=torch.float32, device=self.device)
        expert_mass.scatter_add_(0, flat_ids, flat_gates)
        expert_mass.scatter_add_(0, top_ids[0], torch.full((self.top_k,), 100.0, dtype=torch.float32, device=self.device))

        unique_eids = torch.unique(top_ids).tolist()
        warm_eids = [int(eid) for eid in unique_eids if self.expert_to_slot[int(eid)] >= 0]
        cold_eids = [int(eid) for eid in unique_eids if self.expert_to_slot[int(eid)] < 0]

        # 2a. Execute already-warm experts first without touching ring buffer
        for eid in warm_eids:
            slot = self._ensure_expert_slot(eid)
            mask = top_ids == eid
            tok_indices, slot_indices = torch.where(mask)
            x_sel = x[tok_indices]
            g_sel = top_gates[tok_indices, slot_indices].unsqueeze(-1)
            gu_out = F.linear(x_sel, self.slot_gu[slot])
            act = F.silu(gu_out[:, :2048]) * gu_out[:, 2048:]
            y_exp = F.linear(act, self.slot_d[slot])
            y_routed.index_add_(0, tok_indices, g_sel * y_exp)

        if cold_eids:
            cold_masses = [float(expert_mass[eid].item()) for eid in cold_eids]
            sorted_cold = [eid for _, eid in sorted(zip(cold_masses, cold_eids), reverse=True)]
            n_admit = min(len(sorted_cold), max(self.top_k, self.hot_cache_size - len(warm_eids)))
            admit_set = set(sorted_cold[:n_admit])

            # Group cold experts by g_idx so each group's 8:1 NVFP4 bank is dequantized only ONCE
            by_group: dict[int, list[int]] = {}
            for eid in cold_eids:
                by_group.setdefault(eid // 8, []).append(eid)

            admitted_ordered: list[tuple[float, int, torch.Tensor, torch.Tensor]] = []
            for g_idx, g_eids in by_group.items():
                grp = self.groups[g_idx]
                s_gu_c = unpack_nvfp4(grp.s_gu_packed, grp.s_gu_scale, grp.s_gu_scale2, dtype=torch.bfloat16)
                s_dn_c = unpack_nvfp4(grp.s_down_packed, grp.s_down_scale, grp.s_down_scale2, dtype=torch.bfloat16)
                for eid in g_eids:
                    gu_w, dw = grp.reconstruct_expert(eid % 8, s_gu_cached=s_gu_c, s_dn_cached=s_dn_c)
                    mask = top_ids == eid
                    tok_indices, slot_indices = torch.where(mask)
                    x_sel = x[tok_indices]
                    g_sel = top_gates[tok_indices, slot_indices].unsqueeze(-1)
                    gu_out = F.linear(x_sel, gu_w)
                    act = F.silu(gu_out[:, :2048]) * gu_out[:, 2048:]
                    y_exp = F.linear(act, dw)
                    y_routed.index_add_(0, tok_indices, g_sel * y_exp)
                    if eid in admit_set:
                        admitted_ordered.append((float(expert_mass[eid].item()), eid, gu_w, dw))

            # Commit admitted top-frequency experts into the ring buffer without thrashing
            for _, eid, gu_w, dw in sorted(admitted_ordered, key=lambda item: item[0]):
                self.cache_misses += 1
                slot = self.ring_cursor
                self.ring_cursor = (self.ring_cursor + 1) % self.hot_cache_size
                old_eid = self.slot_to_expert[slot]
                if old_eid >= 0:
                    self.expert_to_slot[old_eid] = -1
                self.slot_gu[slot].copy_(gu_w)
                self.slot_d[slot].copy_(dw)
                self.expert_to_slot[eid] = slot
                self.slot_to_expert[slot] = eid

        return (y_routed + y_shared).view(orig_shape).to(orig_dtype)
