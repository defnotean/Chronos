#!/usr/bin/env python3
"""
Chronos-1T x DeepSeek-V4.1-Flash-1017 (`deepseek-ai/DeepSeek-V4.1-Flash-1017` / Abliterated)
Standalone & vLLM Drop-In `ChronosDsv41SparseMoeBlock`.

Implements runtime execution for converted `DeepSeek-V4.1-Flash-1017` (`DeepseekV41ForCausalLM`) MoE layers
(`chronos-dsv41-layer-XXXXX-of-00040.safetensors`):
  - `hidden_size = 5120`, `moe_intermediate_size = 2304`, `n_routed_experts = 384` (`48` groups of `K = 8`)
  - `sqrtsoftplus` Top-6 `noaux_tc` router (`gate.bias`, `gate.bias_vl`, `routed_scaling_factor = 1.5`)
  - Clamped SwiGLU activation (`swiglu_limit = 10.0`: `gate.clamp(max=10.0)`, `up.clamp(-10.0, 10.0)`)
  - `32x32` Block-FP8 (`F8_E4M3` + `F8_E8M0` scale) Shared Expert (`shared_experts.w1`, `w3`, `w2`)
  - Dual-Verified Post-Quantization Orthogonal Abliteration (`abliteration.r_basis` in `R^{d_r x 5120}`)
  - GPU-resident `8:1` `NVFP4` Holographic Banks + Coupled Rank-64 `FP8` SVD Healers (`1.32 GB/layer`)
  - Zero-Copy Unified-Memory (`offload_tail_to_cpu=True`) 1-Bit / 2-Bit Isotropic Gaussian Tail Lattice
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

try:
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts as vllm_fused_experts
except Exception:
    vllm_fused_experts = None

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True

    @triton.jit
    def _triton_dsv41_gu_swiglu_kernel(
        X_ptr,
        W_gu_ptr,
        Slots_ptr,
        Gates_ptr,
        Act_ptr,
        swiglu_limit,
        N_INTER: tl.constexpr,
        K_DIM: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        e_idx = tl.program_id(1)
        slot = tl.load(Slots_ptr + e_idx).to(tl.int64)
        gate_scale = tl.load(Gates_ptr + e_idx).to(tl.float32)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        w_base = W_gu_ptr + slot * (2 * N_INTER * K_DIM)
        w1_row_ptr = w_base + offs_n[:, None] * K_DIM
        w3_row_ptr = w_base + (offs_n[:, None] + N_INTER) * K_DIM

        acc1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
        acc3 = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for k_start in range(0, K_DIM, BLOCK_K):
            k_idx = k_start + offs_k
            x_k = tl.load(X_ptr + k_idx).to(tl.float32)
            w1_tile = tl.load(w1_row_ptr + k_idx[None, :]).to(tl.float32)
            w3_tile = tl.load(w3_row_ptr + k_idx[None, :]).to(tl.float32)
            acc1 += tl.sum(w1_tile * x_k[None, :], axis=1)
            acc3 += tl.sum(w3_tile * x_k[None, :], axis=1)

        g = tl.minimum(acc1, swiglu_limit)
        u = tl.maximum(tl.minimum(acc3, swiglu_limit), -swiglu_limit)
        act = (g * tl.sigmoid(g)) * u * gate_scale
        tl.store(Act_ptr + e_idx * N_INTER + offs_n, act.to(tl.bfloat16))

    @triton.jit
    def _triton_dsv41_down_kernel(
        Act_ptr,
        W_down_ptr,
        Slots_ptr,
        Y_partial_ptr,
        D_OUT: tl.constexpr,
        K_INTER: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_d = tl.program_id(0)
        e_idx = tl.program_id(1)
        slot = tl.load(Slots_ptr + e_idx).to(tl.int64)

        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        offs_k = tl.arange(0, BLOCK_K)

        wd_row_ptr = W_down_ptr + slot * (D_OUT * K_INTER) + offs_d[:, None] * K_INTER
        act_base_ptr = Act_ptr + e_idx * K_INTER

        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for k_start in range(0, K_INTER, BLOCK_K):
            k_idx = k_start + offs_k
            a_k = tl.load(act_base_ptr + k_idx).to(tl.float32)
            wd_tile = tl.load(wd_row_ptr + k_idx[None, :]).to(tl.float32)
            acc += tl.sum(wd_tile * a_k[None, :], axis=1)

        tl.store(Y_partial_ptr + e_idx * D_OUT + offs_d, acc)

except Exception:
    _HAS_TRITON = False

from chronos_1t.models.chronos_moe_block import (
    NVFP4_LUT,
    build_hadamard_128,
    generate_expert_1d_keys,
    unbind_2d_permuted_hadamard,
    unpack_1bit_rademacher,
    unpack_2bit_lloyd_max,
    unpack_nvfp4,
)

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


def ue8m0_to_float32(scale_e8m0: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Converts IEEE/OCP `float8_e8m0fnu` (`uint8` exponent `2^(e - 127)`) to `float32` on `device`."""
    if scale_e8m0.dtype == torch.float32:
        return scale_e8m0.to(device)
    u8 = scale_e8m0.to(device=device).view(torch.uint8)
    return (u8.to(torch.int32) << 23).view(torch.float32)


def dequantize_block_fp8_32(
    w_fp8: torch.Tensor,
    scale_e8m0: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantizes a 2D `(rows, cols)` `float8_e4m3fn` tensor with `32x32` `F8_E8M0` block scales."""
    rows, cols = w_fp8.shape
    b_r = rows // scale_e8m0.shape[0]
    b_c = cols // scale_e8m0.shape[1]
    w = w_fp8.to(device=device).view(torch.float8_e4m3fn).to(torch.float32).view(scale_e8m0.shape[0], b_r, scale_e8m0.shape[1], b_c)
    s = ue8m0_to_float32(scale_e8m0, device).view(scale_e8m0.shape[0], 1, scale_e8m0.shape[1], 1)
    return (w * s).reshape(rows, cols).to(dtype)


def dequantize_mxfp4_32(
    w_i8: torch.Tensor,
    scale_e8m0: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantizes a packed `MXFP4` (`int8`/`uint8` `[rows, cols//2]`) tensor + `F8_E8M0` `[rows, cols//32]` scale."""
    rows, packed_cols = w_i8.shape
    cols = packed_cols * 2
    lut = torch.tensor(NVFP4_LUT, dtype=torch.float32, device=device)
    u8 = w_i8.to(device=device).view(torch.uint8)
    low = u8 & 0x0F
    high = (u8 >> 4) & 0x0F
    vals = torch.stack([lut[low.long()], lut[high.long()]], dim=-1).view(rows, cols // 32, 32)
    scales = ue8m0_to_float32(scale_e8m0, device).unsqueeze(-1)
    return (vals * scales).view(rows, cols).to(dtype)


def dsv41_clamped_swiglu(gu_out: torch.Tensor, inter_dim: int = 2304, swiglu_limit: float = 10.0) -> torch.Tensor:
    """DeepSeek-V4.1 training-matched clamped SwiGLU (`gate` clamped above at `+limit`, `up` clamped to `[-limit, +limit]`)."""
    gate, up = gu_out.split(inter_dim, dim=-1)
    if swiglu_limit > 0.0:
        gate = gate.clamp(max=swiglu_limit)
        up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
    return F.silu(gate) * up


class ChronosDsv41HolographicExpertGroup(nn.Module):
    """Stores a folded group of K=8 DeepSeek-V4.1-Flash-1017 experts (`gate_up`: 4608x5120, `down`: 5120x2304)."""

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
        r_basis: torch.Tensor | None = None,
        swiglu_limit: float = 10.0,
    ) -> None:
        super().__init__()
        self.group_idx = group_idx
        self.device = device
        self.h128 = h128 if h128 is not None else build_hadamard_128(device, dtype=torch.bfloat16)
        self.r_basis = r_basis  # Optional (d_r, 5120) orthonormal refusal subspace basis in BF16
        self.swiglu_limit = swiglu_limit
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

        self.gu_tail_bits = int(tensors.get("tail.gu_bits", torch.tensor(1)).item())
        self.down_tail_bits = int(tensors.get("tail.down_bits", torch.tensor(1)).item())

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
            gu_p_m = _copy_via_pinned_dma(self.mmap_handle.get_slice(f"{p}tail.gu_packed")[local_m], self.device, 11)
            gu_s_m = _copy_via_pinned_dma(self.mmap_handle.get_slice(f"{p}tail.gu_scale")[local_m], self.device, 12)
            dn_p_m = _copy_via_pinned_dma(self.mmap_handle.get_slice(f"{p}tail.down_packed")[local_m], self.device, 13)
            dn_s_m = _copy_via_pinned_dma(self.mmap_handle.get_slice(f"{p}tail.down_scale")[local_m], self.device, 14)
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
        """1D Activation-Space Cold Bypass for DeepSeek-V4.1 (`gu`: 4608x5120, `down`: 5120x2304)."""
        k_toks = x.shape[0]
        dt = x.dtype
        h = self.h128.to(dt)
        if s_gu_cached is None or s_dn_cached is None:
            s_gu, s_dn = self.get_group_banks()
        else:
            s_gu, s_dn = s_gu_cached, s_dn_cached

        r_s1, _, r_inv, r_s2, c_s1, c_perm, _, c_s2 = generate_expert_1d_keys(
            4608, 5120, local_m, seed_base=101 + self.group_idx * 8, device=self.device
        )
        x_c = (x * c_s1.to(dt).unsqueeze(0))[:, c_perm]
        x_c = torch.matmul(x_c.view(k_toks, 40, 128), h).view(k_toks, 5120) * c_s2.to(dt).unsqueeze(0)
        z_hrr = F.linear(x_c, s_gu) * r_s2.to(dt).unsqueeze(0)
        z_hrr = torch.matmul(z_hrr.view(k_toks, 36, 128), h).view(k_toks, 4608)[:, r_inv] * r_s1.to(dt).unsqueeze(0)

        u_gu_m = self.u_gu[local_m].to(dt)
        v_gu_m = self.v_gu[local_m].to(dt)
        z_gu = self.beta_gu[local_m] * z_hrr + F.linear(F.linear(x, v_gu_m), u_gu_m)

        act = dsv41_clamped_swiglu(z_gu, inter_dim=2304, swiglu_limit=self.swiglu_limit)

        dr_s1, _, dr_inv, dr_s2, dc_s1, dc_perm, _, dc_s2 = generate_expert_1d_keys(
            5120, 2304, local_m, seed_base=303 + self.group_idx * 8, device=self.device
        )
        a_c = (act * dc_s1.to(dt).unsqueeze(0))[:, dc_perm]
        a_c = torch.matmul(a_c.view(k_toks, 18, 128), h).view(k_toks, 2304) * dc_s2.to(dt).unsqueeze(0)
        y_hrr = F.linear(a_c, s_dn) * dr_s2.to(dt).unsqueeze(0)
        y_hrr = torch.matmul(y_hrr.view(k_toks, 40, 128), h).view(k_toks, 5120)[:, dr_inv] * dr_s1.to(dt).unsqueeze(0)

        u_dn_m = self.u_down[local_m].to(dt)
        v_dn_m = self.v_down[local_m].to(dt)
        y_out = self.beta_down[local_m] * y_hrr + F.linear(F.linear(act, v_dn_m), u_dn_m)
        if self.r_basis is not None:
            rb = self.r_basis.to(dt)
            y_out = y_out - (y_out @ rb.T) @ rb
        return y_out

    @torch.no_grad()
    def reconstruct_expert(
        self,
        local_m: int,
        s_gu_cached: torch.Tensor | None = None,
        s_dn_cached: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstructs `(gu_w: (4608, 5120), down_w: (5120, 2304))` in BF16 for expert `local_m` in 0..7."""
        if s_gu_cached is None or s_dn_cached is None:
            s_gu, s_dn = self.get_group_banks()
        else:
            s_gu, s_dn = s_gu_cached, s_dn_cached

        r_s1, _, r_inv, r_s2, c_s1, _, c_inv, c_s2 = generate_expert_1d_keys(
            4608, 5120, local_m, seed_base=101 + self.group_idx * 8, device=self.device
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
            5120, 2304, local_m, seed_base=303 + self.group_idx * 8, device=self.device
        )
        h_dn = unbind_2d_permuted_hadamard(s_dn, dr_s1, dr_inv, dr_s2, dc_s1, dc_inv, dc_s2, self.h128)
        healer_dn = torch.matmul(self.u_down[local_m].to(torch.bfloat16), self.v_down[local_m].to(torch.bfloat16))

        if self.down_tail_bits == 2:
            tail_dn = unpack_2bit_lloyd_max(dn_p_m, dn_s_m, dtype=torch.bfloat16)
        else:
            tail_dn = unpack_1bit_rademacher(dn_p_m, dn_s_m, dtype=torch.bfloat16)

        down_hat = self.beta_down[local_m] * h_dn + healer_dn + tail_dn
        if self.r_basis is not None:
            rb = self.r_basis.to(torch.float32)
            df = down_hat.float()
            down_hat = (df - rb.T @ (rb @ df)).to(torch.bfloat16)
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
        """Zero-temp-allocation in-place slot reconstruction (`v7.5`) with post-quantization refusal projection."""
        if s_gu_cached is None or s_dn_cached is None:
            s_gu, s_dn = self.get_group_banks()
        else:
            s_gu, s_dn = s_gu_cached, s_dn_cached

        gu_p_m, gu_s_m, dn_p_m, dn_s_m = self._fetch_tail_slices(local_m)
        r_s1, _, r_inv, r_s2, c_s1, _, c_inv, c_s2 = generate_expert_1d_keys(
            4608, 5120, local_m, seed_base=101 + self.group_idx * 8, device=self.device
        )
        h_gu = unbind_2d_permuted_hadamard(s_gu, r_s1, r_inv, r_s2, c_s1, c_inv, c_s2, self.h128)
        if self.gu_tail_bits == 2:
            slot_gu.copy_(unpack_2bit_lloyd_max(gu_p_m, gu_s_m, dtype=torch.bfloat16))
        else:
            slot_gu.copy_(unpack_1bit_rademacher(gu_p_m, gu_s_m, dtype=torch.bfloat16))
        slot_gu.add_(h_gu * self.beta_gu[local_m])
        slot_gu.addmm_(self.u_gu[local_m].to(torch.bfloat16), self.v_gu[local_m].to(torch.bfloat16))

        dr_s1, _, dr_inv, dr_s2, dc_s1, _, dc_inv, dc_s2 = generate_expert_1d_keys(
            5120, 2304, local_m, seed_base=303 + self.group_idx * 8, device=self.device
        )
        h_dn = unbind_2d_permuted_hadamard(s_dn, dr_s1, dr_inv, dr_s2, dc_s1, dc_inv, dc_s2, self.h128)
        if self.down_tail_bits == 2:
            slot_dn.copy_(unpack_2bit_lloyd_max(dn_p_m, dn_s_m, dtype=torch.bfloat16))
        else:
            slot_dn.copy_(unpack_1bit_rademacher(dn_p_m, dn_s_m, dtype=torch.bfloat16))
        slot_dn.add_(h_dn * self.beta_down[local_m])
        slot_dn.addmm_(self.u_down[local_m].to(torch.bfloat16), self.v_down[local_m].to(torch.bfloat16))

        if self.r_basis is not None:
            rb = self.r_basis.to(torch.bfloat16)
            proj = torch.matmul(rb, slot_dn)
            slot_dn.addmm_(rb.T, proj, beta=1.0, alpha=-1.0)


class ChronosDsv41SparseMoeBlock(nn.Module):
    """Drop-in replacement for DeepSeek-V4.1-Flash-1017 (`DeepseekV41ForCausalLM`) Sparse MoE layer."""

    def __init__(
        self,
        layer_idx: int = 0,
        shard_path: str | Path | None = None,
        device: torch.device | str | None = None,
        top_k: int = 6,
        routed_scaling_factor: float = 1.5,
        swiglu_limit: float = 10.0,
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
                shard_path = model_dir / f"chronos-dsv41-layer-{layer_idx:05d}-of-00040.safetensors"

        if device is None:
            device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        if shard_path is None:
            shard_path = f"/content/dsv41_chronos_v75/chronos-dsv41-layer-{layer_idx:05d}-of-00040.safetensors"

        self.layer_idx = layer_idx
        self.device = device
        self.top_k = top_k
        self.routed_scaling_factor = routed_scaling_factor
        self.swiglu_limit = swiglu_limit
        self.hot_cache_size = max(top_k, hot_cache_size)
        self.offload_tail_to_cpu = offload_tail_to_cpu
        self.mmap_zero_ram = mmap_zero_ram
        self.use_fused_moe = use_fused_moe and (vllm_fused_experts is not None)
        self.shard_path_str = str(shard_path)

        tp = f"layers.{layer_idx}.ffn"
        self._mmap_handle = safe_open(self.shard_path_str, framework="pt", device="cpu")
        all_keys = set(self._mmap_handle.keys())
        if f"{tp}.gate.weight" not in all_keys and f"model.{tp}.gate.weight" in all_keys:
            tp = f"model.layers.{layer_idx}.ffn"

        self.router_w = self._mmap_handle.get_tensor(f"{tp}.gate.weight").to(device=device, dtype=torch.bfloat16).contiguous()
        bias_key = f"{tp}.gate.bias" if f"{tp}.gate.bias" in all_keys else f"{tp}.gate.e_score_correction_bias"
        self.e_score_bias = self._mmap_handle.get_tensor(bias_key).to(device=device, dtype=torch.float32).contiguous()
        self.bias_vl = (
            self._mmap_handle.get_tensor(f"{tp}.gate.bias_vl").to(device=device, dtype=torch.float32).contiguous()
            if f"{tp}.gate.bias_vl" in all_keys
            else None
        )

        sh_w1 = dequantize_block_fp8_32(
            self._mmap_handle.get_tensor(f"{tp}.shared_experts.w1.weight"),
            self._mmap_handle.get_tensor(f"{tp}.shared_experts.w1.scale"),
            device,
        )
        sh_w3 = dequantize_block_fp8_32(
            self._mmap_handle.get_tensor(f"{tp}.shared_experts.w3.weight"),
            self._mmap_handle.get_tensor(f"{tp}.shared_experts.w3.scale"),
            device,
        )
        self.sh_gu = torch.cat([sh_w1, sh_w3], dim=0).contiguous()  # (4608, 5120)
        self.sh_down = dequantize_block_fp8_32(
            self._mmap_handle.get_tensor(f"{tp}.shared_experts.w2.weight"),
            self._mmap_handle.get_tensor(f"{tp}.shared_experts.w2.scale"),
            device,
        ).contiguous()  # (5120, 2304)

        self.r_basis: torch.Tensor | None = None
        if f"{tp}.abliteration.r_basis" in all_keys:
            self.r_basis = self._mmap_handle.get_tensor(f"{tp}.abliteration.r_basis").to(device=device, dtype=torch.bfloat16).contiguous()

        shared_h128 = build_hadamard_128(device, dtype=torch.bfloat16)
        self.groups = nn.ModuleList()
        for g_idx in range(48):
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
                ChronosDsv41HolographicExpertGroup(
                    group_idx=g_idx,
                    tensors=sub_t,
                    device=device,
                    offload_tail_to_cpu=offload_tail_to_cpu,
                    mmap_zero_ram=mmap_zero_ram,
                    mmap_handle=self._mmap_handle if mmap_zero_ram else None,
                    mmap_prefix=gp,
                    h128=shared_h128,
                    r_basis=self.r_basis,
                    swiglu_limit=swiglu_limit,
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

        self.shared_slot = self.hot_cache_size
        self.cache_gu_buf = torch.empty((self.hot_cache_size + 1, 4608, 5120), dtype=torch.bfloat16, device=device)
        self.cache_d_buf = torch.empty((self.hot_cache_size + 1, 5120, 2304), dtype=torch.bfloat16, device=device)
        self.cache_gu_buf[self.shared_slot].copy_(self.sh_gu)
        self.cache_d_buf[self.shared_slot].copy_(self.sh_down)
        self.slot_gu = [self.cache_gu_buf[s] for s in range(self.hot_cache_size)]
        self.slot_d = [self.cache_d_buf[s] for s in range(self.hot_cache_size)]

        self.expert_to_slot: list[int] = [-1] * 384
        self.expert_to_slot_gpu = torch.full((384,), -1, dtype=torch.int32, device=device)
        self.slot_to_expert: list[int] = [-1] * self.hot_cache_size
        self.ring_cursor = 0
        self.cache_hits = 0
        self.cache_misses = 0

        self._active_slots_buf = torch.empty((self.top_k + 1,), dtype=torch.int32, device=device)
        self._active_slots_buf[self.top_k] = self.shared_slot
        self._active_gates_buf = torch.ones((self.top_k + 1,), dtype=torch.float32, device=device)
        self._act_inter_buf = torch.empty((self.top_k + 1, 2304), dtype=torch.bfloat16, device=device)
        self._y_partial_buf = torch.empty((self.top_k + 1, 5120), dtype=torch.float32, device=device)

        # Upgrade 2: Pre-seed & pin top Global Router Prior hub experts (or direct MXFP4 hot experts if present in shard)
        self.pinned_slots = 0
        with torch.no_grad():
            rw_f = self.router_w.float()
            _, _, vh_rt = torch.linalg.svd(rw_f, full_matrices=False)
            u_ctx = vh_rt[:32, :].T
            prior_scores = torch.linalg.norm(rw_f @ u_ctx, dim=-1) + self.e_score_bias
            n_pin = min(8, max(0, self.hot_cache_size - self.top_k))
            _, top_hub_ids = torch.topk(prior_scores, k=n_pin)
            for eid in top_hub_ids.tolist():
                self._ensure_expert_slot(int(eid))
            self.pinned_slots = n_pin
            if self.ring_cursor < self.pinned_slots:
                self.ring_cursor = self.pinned_slots
            self.cache_hits = 0
            self.cache_misses = 0

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        for _ in weights:
            pass
        return {name for name, _ in self.named_parameters(recurse=True)}

    def finalize_mega_moe_weights(self) -> None:
        pass

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
        dyn_span = max(1, self.hot_cache_size - self.pinned_slots)
        self.ring_cursor = self.pinned_slots + ((self.ring_cursor - self.pinned_slots + 1) % dyn_span)

        old_eid = self.slot_to_expert[slot]
        if old_eid >= 0:
            self.expert_to_slot[old_eid] = -1
            self.expert_to_slot_gpu[old_eid] = -1

        g_idx = expert_id // 8
        local_m = expert_id % 8
        self.groups[g_idx].reconstruct_expert_into_slot(
            local_m, self.slot_gu[slot], self.slot_d[slot], s_gu_cached=s_gu_cached, s_dn_cached=s_dn_cached
        )

        self.expert_to_slot[expert_id] = slot
        self.expert_to_slot_gpu[expert_id] = slot
        self.slot_to_expert[slot] = expert_id
        return slot

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        image_mask: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        mega_gate_metadata: Any = None,
    ) -> torch.Tensor:
        orig_shape = hidden_states.shape
        orig_dtype = hidden_states.dtype
        x = hidden_states.view(-1, orig_shape[-1]).to(torch.bfloat16)
        n_toks = x.shape[0]

        # 1. sqrtsoftplus Top-6 noaux_tc Router Branch (BF16 GEMV + FP32 nonlinear scoring)
        raw_logits = F.linear(x, self.router_w).float()
        scores = F.softplus(raw_logits).sqrt()
        bias = self.e_score_bias.unsqueeze(0)
        if image_mask is not None and self.bias_vl is not None:
            bias = torch.where(image_mask.view(-1, 1), self.bias_vl.unsqueeze(0), bias)
        _, top_ids = torch.topk(scores + bias, k=self.top_k, dim=-1)
        top_weights = scores.gather(1, top_ids)
        top_gates_f32 = (top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-20)) * self.routed_scaling_factor
        top_gates = top_gates_f32.to(torch.bfloat16)

        if n_toks == 1:
            self._active_slots_buf[: self.top_k].copy_(self.expert_to_slot_gpu.index_select(0, top_ids[0]))
            self._active_gates_buf[: self.top_k].copy_(top_gates_f32[0])

            eids = top_ids[0].tolist()
            if all(self.expert_to_slot[int(eid)] >= 0 for eid in eids):
                self.cache_hits += self.top_k
                slot_ids = [self.expert_to_slot[int(eid)] for eid in eids]
            else:
                group_bank_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
                slot_ids = []
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
                self._active_slots_buf[: self.top_k].copy_(self.expert_to_slot_gpu.index_select(0, top_ids[0]))

            if _HAS_TRITON and x.is_cuda:
                n_active = self.top_k + 1
                _triton_dsv41_gu_swiglu_kernel[(2304 // 16, n_active)](
                    x,
                    self.cache_gu_buf,
                    self._active_slots_buf,
                    self._active_gates_buf,
                    self._act_inter_buf,
                    float(self.swiglu_limit),
                    N_INTER=2304,
                    K_DIM=5120,
                    BLOCK_N=16,
                    BLOCK_K=256,
                    num_warps=4,
                )
                _triton_dsv41_down_kernel[(5120 // 16, n_active)](
                    self._act_inter_buf,
                    self.cache_d_buf,
                    self._active_slots_buf,
                    self._y_partial_buf,
                    D_OUT=5120,
                    K_INTER=2304,
                    BLOCK_D=16,
                    BLOCK_K=256,
                    num_warps=4,
                )
                return self._y_partial_buf.sum(dim=0, keepdim=True).view(orig_shape).to(orig_dtype)

            if self.use_fused_moe:
                sh_gu_out = F.linear(x, self.sh_gu)
                y_shared = F.linear(dsv41_clamped_swiglu(sh_gu_out, 2304, self.swiglu_limit), self.sh_down)
                slot_ids_t = self._active_slots_buf[: self.top_k].unsqueeze(0)
                y_routed = vllm_fused_experts(
                    x, self.cache_gu_buf[: self.hot_cache_size], self.cache_d_buf[: self.hot_cache_size], top_gates, slot_ids_t, inplace=False
                )
                return (y_routed + y_shared).view(orig_shape).to(orig_dtype)

            # Fallback Batched SwiGLU + Pre-Scaled Down-Projection
            gu_all = torch.stack(
                [F.linear(x, self.sh_gu)] + [F.linear(x, self.slot_gu[s]) for s in slot_ids], dim=0
            ).squeeze(1)  # (7, 4608)
            act_all = dsv41_clamped_swiglu(gu_all, 2304, self.swiglu_limit)  # (7, 2304) in 1 fused pass
            act_all[1:].mul_(top_gates[0].unsqueeze(-1))
            y_out = F.linear(act_all[0:1], self.sh_down)
            for k_idx, slot in enumerate(slot_ids):
                y_out.add_(F.linear(act_all[k_idx + 1 : k_idx + 2], self.slot_d[slot]))
            return y_out.view(orig_shape).to(orig_dtype)

        # Shared Expert Branch for n_toks > 1
        sh_gu_out = F.linear(x, self.sh_gu)
        y_shared = F.linear(dsv41_clamped_swiglu(sh_gu_out, 2304, self.swiglu_limit), self.sh_down)

        y_routed = torch.zeros_like(x)
        flat_ids = top_ids.view(-1)
        flat_gates = top_gates.view(-1).float()
        expert_mass = torch.zeros(384, dtype=torch.float32, device=self.device)
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
            act = dsv41_clamped_swiglu(gu_out, 2304, self.swiglu_limit)
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
                s_gu_c = unpack_nvfp4(grp.s_gu_packed, grp.s_gu_scale, grp.s_gu_scale2, dtype=torch.bfloat16)
                s_dn_c = unpack_nvfp4(grp.s_down_packed, grp.s_down_scale, grp.s_down_scale2, dtype=torch.bfloat16)
                for eid in g_eids:
                    gu_w, dw = grp.reconstruct_expert(eid % 8, s_gu_cached=s_gu_c, s_dn_cached=s_dn_c)
                    mask = top_ids == eid
                    tok_indices, slot_indices = torch.where(mask)
                    x_sel = x[tok_indices]
                    g_sel = top_gates[tok_indices, slot_indices].unsqueeze(-1)
                    gu_out = F.linear(x_sel, gu_w)
                    act = dsv41_clamped_swiglu(gu_out, 2304, self.swiglu_limit)
                    y_exp = F.linear(act, dw)
                    y_routed.index_add_(0, tok_indices, g_sel * y_exp)
                    if eid in admit_set:
                        admitted_ordered.append((float(expert_mass[eid].item()), eid, gu_w, dw))

            for _, eid, gu_w, dw in sorted(admitted_ordered, key=lambda item: item[0]):
                self.cache_misses += 1
                slot = self.ring_cursor
                dyn_span = max(1, self.hot_cache_size - self.pinned_slots)
                self.ring_cursor = self.pinned_slots + ((self.ring_cursor - self.pinned_slots + 1) % dyn_span)
                old_eid = self.slot_to_expert[slot]
                if old_eid >= 0:
                    self.expert_to_slot[old_eid] = -1
                    self.expert_to_slot_gpu[old_eid] = -1
                self.slot_gu[slot].copy_(gu_w)
                self.slot_d[slot].copy_(dw)
                self.expert_to_slot[eid] = slot
                self.expert_to_slot_gpu[eid] = slot
                self.slot_to_expert[slot] = eid

        return (y_routed + y_shared).view(orig_shape).to(orig_dtype)
