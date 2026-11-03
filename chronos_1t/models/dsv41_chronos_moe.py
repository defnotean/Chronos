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

import gc
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

    @triton.jit
    def _quant_dequant_e2m1(v):
        s_sign = tl.where(v < 0.0, -1.0, 1.0)
        s_bit = tl.where(v < 0.0, 8, 0).to(tl.uint8)
        a = tl.abs(v)
        idx = (
            tl.where(a >= 0.25, 1, 0)
            + tl.where(a >= 0.75, 1, 0)
            + tl.where(a >= 1.25, 1, 0)
            + tl.where(a >= 1.75, 1, 0)
            + tl.where(a >= 2.5, 1, 0)
            + tl.where(a >= 3.5, 1, 0)
            + tl.where(a >= 5.0, 1, 0)
        ).to(tl.uint8)
        e = idx >> 1
        f = idx & 1
        q_abs = tl.where(e == 0, f << 1, (2 | f) << e).to(tl.float32) * 0.25
        return (s_bit | idx), (s_sign * q_abs)

    @triton.jit
    def _triton_quantize_mse_fp4_32_kernel(
        W_ptr,
        Out_u8_ptr,
        Out_s_bf16_ptr,
        N_BLOCKS_C: tl.constexpr,
    ):
        row = tl.program_id(0)
        blk = tl.program_id(1)
        offs_pair = tl.arange(0, 16)

        base_w = W_ptr + row * (N_BLOCKS_C * 32) + blk * 32
        w_even = tl.load(base_w + offs_pair * 2).to(tl.float32)
        w_odd = tl.load(base_w + offs_pair * 2 + 1).to(tl.float32)

        amax = tl.maximum(tl.max(tl.abs(w_even), axis=0), tl.max(tl.abs(w_odd), axis=0)).to(tl.float32)
        amax = tl.maximum(amax, 1e-12).to(tl.float32)

        best_err = 1e30
        best_s_bf16 = (amax / 24.0).to(tl.bfloat16)
        best_packed = tl.zeros((16,), dtype=tl.uint8)

        for step in range(8):
            div = 6.0 - step * 0.35
            s_cand_bf16 = ((amax / div) * 0.25).to(tl.bfloat16)
            eff_s = tl.maximum(s_cand_bf16.to(tl.float32) * 4.0, 1e-12)
            inv_s = 1.0 / eff_s
            nib_l, q_l = _quant_dequant_e2m1(w_even * inv_s)
            nib_h, q_h = _quant_dequant_e2m1(w_odd * inv_s)
            err_e = w_even - q_l * eff_s
            err_o = w_odd - q_h * eff_s
            mse = tl.sum(err_e * err_e + err_o * err_o, axis=0)
            cond = mse < best_err
            best_err = tl.where(cond, mse, best_err)
            best_s_bf16 = tl.where(cond, s_cand_bf16, best_s_bf16)
            best_packed = tl.where(cond, nib_l | (nib_h << 4), best_packed)

        tl.store(Out_s_bf16_ptr + row * N_BLOCKS_C + blk, best_s_bf16)
        tl.store(Out_u8_ptr + row * (N_BLOCKS_C * 16) + blk * 16 + offs_pair, best_packed)

    @triton.jit
    def _decode_e2m1_x4(nib):
        i32 = nib.to(tl.int32)
        mag = ((((-932957680 >> ((i32 & 7) << 2)) & 0xF) << 1)).to(tl.float32)
        return tl.where((i32 & 8) != 0, -mag, mag)

    @triton.jit
    def _triton_dsv41_hybrid_gu_swiglu_kernel(
        X_ptr,
        W_gu_u8_ptr,
        S_gu_bf16_ptr,
        Sh_gu_fp8_ptr,
        Sh_gu_s_ptr,
        Slots_ptr,
        Gates_ptr,
        Act_ptr,
        swiglu_limit,
        TOP_K: tl.constexpr,
        N_INTER: tl.constexpr,
        N_BLOCKS_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
        sub_blocks_k: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        e_idx = tl.program_id(1)
        gate_scale = tl.load(Gates_ptr + e_idx).to(tl.float32)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        acc1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
        acc3 = tl.zeros((BLOCK_N,), dtype=tl.float32)

        if e_idx < TOP_K:
            slot = tl.maximum(tl.load(Slots_ptr + e_idx).to(tl.int64), 0)
            BYTES_PER_ITER: tl.constexpr = sub_blocks_k * 16
            offs_b = tl.arange(0, BYTES_PER_ITER)
            offs_sb = offs_b // 16

            w_stride_slot = 2 * N_INTER * (N_BLOCKS_K * 16)
            s_stride_slot = 2 * N_INTER * N_BLOCKS_K

            w1_base = W_gu_u8_ptr + slot * w_stride_slot + offs_n[:, None] * (N_BLOCKS_K * 16)
            w3_base = W_gu_u8_ptr + slot * w_stride_slot + (offs_n[:, None] + N_INTER) * (N_BLOCKS_K * 16)
            s1_base = S_gu_bf16_ptr + slot * s_stride_slot + offs_n[:, None] * N_BLOCKS_K
            s3_base = S_gu_bf16_ptr + slot * s_stride_slot + (offs_n[:, None] + N_INTER) * N_BLOCKS_K

            for sb_start in range(0, N_BLOCKS_K, sub_blocks_k):
                b_idx = sb_start * 16 + offs_b
                sb_idx = sb_start + offs_sb
                x_even = tl.load(X_ptr + b_idx * 2).to(tl.float32)
                x_odd = tl.load(X_ptr + b_idx * 2 + 1).to(tl.float32)

                w1_b = tl.load(w1_base + b_idx[None, :])
                s1_f = tl.load(s1_base + sb_idx[None, :]).to(tl.float32)
                v1 = (_decode_e2m1_x4(w1_b & 0x0F) * x_even[None, :] + _decode_e2m1_x4(w1_b >> 4) * x_odd[None, :]) * s1_f
                acc1 += tl.sum(v1, axis=1)

                w3_b = tl.load(w3_base + b_idx[None, :])
                s3_f = tl.load(s3_base + sb_idx[None, :]).to(tl.float32)
                v3 = (_decode_e2m1_x4(w3_b & 0x0F) * x_even[None, :] + _decode_e2m1_x4(w3_b >> 4) * x_odd[None, :]) * s3_f
                acc3 += tl.sum(v3, axis=1)
        else:
            K_DIM: tl.constexpr = N_BLOCKS_K * 32
            BLOCK_K: tl.constexpr = sub_blocks_k * 32
            offs_k = tl.arange(0, BLOCK_K)
            w1_row_ptr = Sh_gu_fp8_ptr + offs_n[:, None] * K_DIM
            w3_row_ptr = Sh_gu_fp8_ptr + (offs_n[:, None] + N_INTER) * K_DIM
            for k_start in range(0, K_DIM, BLOCK_K):
                k_idx = k_start + offs_k
                x_k = tl.load(X_ptr + k_idx).to(tl.float32)
                w1_tile = tl.load(w1_row_ptr + k_idx[None, :]).to(tl.float32)
                w3_tile = tl.load(w3_row_ptr + k_idx[None, :]).to(tl.float32)
                acc1 += tl.sum(w1_tile * x_k[None, :], axis=1)
                acc3 += tl.sum(w3_tile * x_k[None, :], axis=1)
            acc1 = acc1 * tl.load(Sh_gu_s_ptr + offs_n)
            acc3 = acc3 * tl.load(Sh_gu_s_ptr + N_INTER + offs_n)

        g = tl.minimum(acc1, swiglu_limit)
        u = tl.maximum(tl.minimum(acc3, swiglu_limit), -swiglu_limit)
        act = (g * tl.sigmoid(g)) * u * gate_scale
        tl.store(Act_ptr + e_idx * N_INTER + offs_n, act.to(tl.bfloat16))

    @triton.jit
    def _triton_dsv41_hybrid_down_kernel(
        Act_ptr,
        W_d_u8_ptr,
        S_d_bf16_ptr,
        Sh_d_fp8_ptr,
        Sh_d_s_ptr,
        Slots_ptr,
        Y_partial_ptr,
        TOP_K: tl.constexpr,
        D_OUT: tl.constexpr,
        N_BLOCKS_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        sub_blocks_k: tl.constexpr,
    ):
        pid_d = tl.program_id(0)
        e_idx = tl.program_id(1)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        act_base = Act_ptr + e_idx * (N_BLOCKS_K * 32)
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

        if e_idx < TOP_K:
            slot = tl.maximum(tl.load(Slots_ptr + e_idx).to(tl.int64), 0)
            BYTES_PER_ITER: tl.constexpr = sub_blocks_k * 16
            offs_b = tl.arange(0, BYTES_PER_ITER)
            offs_sb = offs_b // 16

            wd_base = W_d_u8_ptr + slot * (D_OUT * N_BLOCKS_K * 16) + offs_d[:, None] * (N_BLOCKS_K * 16)
            sd_base = S_d_bf16_ptr + slot * (D_OUT * N_BLOCKS_K) + offs_d[:, None] * N_BLOCKS_K

            for sb_start in range(0, N_BLOCKS_K, sub_blocks_k):
                b_idx = sb_start * 16 + offs_b
                sb_idx = sb_start + offs_sb
                a_even = tl.load(act_base + b_idx * 2).to(tl.float32)
                a_odd = tl.load(act_base + b_idx * 2 + 1).to(tl.float32)
                wd_b = tl.load(wd_base + b_idx[None, :])
                sd_f = tl.load(sd_base + sb_idx[None, :]).to(tl.float32)
                vd = (_decode_e2m1_x4(wd_b & 0x0F) * a_even[None, :] + _decode_e2m1_x4(wd_b >> 4) * a_odd[None, :]) * sd_f
                acc += tl.sum(vd, axis=1)
        else:
            K_INTER: tl.constexpr = N_BLOCKS_K * 32
            BLOCK_K: tl.constexpr = sub_blocks_k * 32
            offs_k = tl.arange(0, BLOCK_K)
            wd_row_ptr = Sh_d_fp8_ptr + offs_d[:, None] * K_INTER
            for k_start in range(0, K_INTER, BLOCK_K):
                k_idx = k_start + offs_k
                a_k = tl.load(act_base + k_idx).to(tl.float32)
                wd_tile = tl.load(wd_row_ptr + k_idx[None, :]).to(tl.float32)
                acc += tl.sum(wd_tile * a_k[None, :], axis=1)
            acc = acc * tl.load(Sh_d_s_ptr + offs_d)

        tl.store(Y_partial_ptr + e_idx * D_OUT + offs_d, acc)

    @triton.jit
    def _fwht128(v):
        for _ in tl.static_range(7):
            v2 = tl.reshape(v, (64, 2))
            a, b = tl.split(v2)
            j = tl.join(a + b, a - b)
            v = tl.reshape(tl.trans(j), (128,))
        return v * 0.08838834764831845

    @triton.jit
    def _triton_dsv41_router_top6_kernel(
        Logits_ptr,
        Bias_ptr,
        Top_ids_ptr,
        Gates_ptr,
        routed_scaling_factor,
        K_ACT: tl.constexpr,
    ):
        offs = tl.arange(0, 512)
        mask = offs < 384
        logit = tl.load(Logits_ptr + offs, mask=mask, other=-1e30).to(tl.float32)
        sp = tl.where(logit > 20.0, logit, tl.log(1.0 + tl.exp(logit)))
        score = tl.sqrt(sp)
        bias = tl.load(Bias_ptr + offs, mask=mask, other=-1e30).to(tl.float32)
        biased = tl.where(mask, score + bias, -1e30)

        k_offs = tl.arange(0, 8)
        w_vec = tl.zeros((8,), dtype=tl.float32)
        for k in tl.static_range(6):
            best_idx = tl.argmax(biased, axis=0)
            if k < K_ACT:
                tl.store(Top_ids_ptr + k, best_idx.to(tl.int32))
            w_k = tl.max(tl.where(offs == best_idx, score, -1.0), axis=0)
            w_vec = tl.where(k_offs == k, w_k, w_vec)
            biased = tl.where(offs == best_idx, -1e30, biased)

        w_sum6 = tl.sum(w_vec, axis=0) + 1e-20
        gates = (w_vec / w_sum6) * routed_scaling_factor
        tl.store(Gates_ptr + k_offs, gates, mask=k_offs < K_ACT)
        tl.store(Gates_ptr + K_ACT, 1.0)

    @triton.jit
    def _triton_dsv41_fused_pre_unbind_healer_v_kernel(
        X_in_ptr,
        V_fp8_ptr,
        Top_ids_ptr,
        Hot_slot_map_ptr,
        C_perm_ptr,
        C_s1_perm_ptr,
        C_s2_ptr,
        X_hrr_out_ptr,
        H_mid_out_ptr,
        D_IN: tl.constexpr,
        N_UNBIND_BLKS: tl.constexpr,
        X_STRIDE_E: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        e_idx = tl.program_id(1)
        eid = tl.load(Top_ids_ptr + e_idx).to(tl.int64)
        hot_slot = tl.load(Hot_slot_map_ptr + eid)
        if hot_slot >= 0:
            return
        if pid < N_UNBIND_BLKS:
            offs = pid * 128 + tl.arange(0, 128)
            key_base = eid * D_IN + offs
            perm_idx = tl.load(C_perm_ptr + key_base).to(tl.int64)
            s1 = tl.load(C_s1_perm_ptr + key_base).to(tl.float32)
            s2 = tl.load(C_s2_ptr + key_base).to(tl.float32)
            x_val = tl.load(X_in_ptr + e_idx * X_STRIDE_E + perm_idx).to(tl.float32) * s1
            h_val = _fwht128(x_val) * s2
            tl.store(X_hrr_out_ptr + e_idx * D_IN + offs, h_val.to(tl.bfloat16))
        else:
            r_blk = pid - N_UNBIND_BLKS
            offs_r = r_blk * 16 + tl.arange(0, 16)
            offs_k = tl.arange(0, BLOCK_K)
            x_base = X_in_ptr + e_idx * X_STRIDE_E
            v_base = V_fp8_ptr + eid * (64 * D_IN) + offs_r[:, None] * D_IN
            acc = tl.zeros((16,), dtype=tl.float32)
            for k_start in range(0, D_IN, BLOCK_K):
                k_idx = k_start + offs_k
                x_k = tl.load(x_base + k_idx).to(tl.float32)
                v_tile = tl.load(v_base + k_idx[None, :]).to(tl.float32)
                acc += tl.sum(v_tile * x_k[None, :], axis=1)
            tl.store(H_mid_out_ptr + e_idx * 64 + offs_r, acc)

    @triton.jit
    def _triton_dsv41_resident_gu_kernel(
        X_raw_ptr,
        X_hrr_ptr,
        W_hot_gu_u8_ptr,
        S_hot_gu_bf16_ptr,
        W_grp_gu_u8_ptr,
        S_grp_gu_fp8_ptr,
        Cold_scale_gu_ptr,
        Gu_r_s2_ptr,
        U_gu_fp8_ptr,
        H_gu_mid_ptr,
        Top_ids_ptr,
        Hot_slot_map_ptr,
        Gates_ptr,
        Z_hrr_out_ptr,
        Z_gu_healer_out_ptr,
        Act_out_ptr,
        swiglu_limit,
        TOP_K: tl.constexpr,
        SHARED_SLOT: tl.constexpr,
        N_INTER: tl.constexpr,
        N_BLOCKS_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
        sub_blocks_k: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        e_idx = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
        acc3 = tl.zeros((BLOCK_N,), dtype=tl.float32)

        if e_idx < TOP_K:
            eid = tl.load(Top_ids_ptr + e_idx).to(tl.int64)
            hot_slot = tl.load(Hot_slot_map_ptr + eid).to(tl.int64)
        else:
            eid = tl.zeros((), dtype=tl.int64)
            hot_slot = tl.zeros((), dtype=tl.int64) + SHARED_SLOT

        BYTES_PER_ITER: tl.constexpr = sub_blocks_k * 16
        offs_b = tl.arange(0, BYTES_PER_ITER)
        w_stride_slot = 2 * N_INTER * (N_BLOCKS_K * 16)

        if hot_slot >= 0:
            offs_sb = offs_b // 16
            s_stride_slot = 2 * N_INTER * N_BLOCKS_K
            w1_hot_base = W_hot_gu_u8_ptr + hot_slot * w_stride_slot + offs_n[:, None] * (N_BLOCKS_K * 16)
            w3_hot_base = W_hot_gu_u8_ptr + hot_slot * w_stride_slot + (offs_n[:, None] + N_INTER) * (N_BLOCKS_K * 16)
            s1_hot_base = S_hot_gu_bf16_ptr + hot_slot * s_stride_slot + offs_n[:, None] * N_BLOCKS_K
            s3_hot_base = S_hot_gu_bf16_ptr + hot_slot * s_stride_slot + (offs_n[:, None] + N_INTER) * N_BLOCKS_K

            for sb_start in range(0, N_BLOCKS_K, sub_blocks_k):
                b_idx = sb_start * 16 + offs_b
                sb_idx = sb_start + offs_sb
                x_even = tl.load(X_raw_ptr + b_idx * 2).to(tl.float32)
                x_odd = tl.load(X_raw_ptr + b_idx * 2 + 1).to(tl.float32)

                w1_b = tl.load(w1_hot_base + b_idx[None, :])
                s1_f = tl.load(s1_hot_base + sb_idx[None, :]).to(tl.float32)
                v1 = (_decode_e2m1_x4(w1_b & 0x0F) * x_even[None, :] + _decode_e2m1_x4(w1_b >> 4) * x_odd[None, :]) * s1_f
                acc1 += tl.sum(v1, axis=1)

                w3_b = tl.load(w3_hot_base + b_idx[None, :])
                s3_f = tl.load(s3_hot_base + sb_idx[None, :]).to(tl.float32)
                v3 = (_decode_e2m1_x4(w3_b & 0x0F) * x_even[None, :] + _decode_e2m1_x4(w3_b >> 4) * x_odd[None, :]) * s3_f
                acc3 += tl.sum(v3, axis=1)

            gate_scale = tl.load(Gates_ptr + e_idx).to(tl.float32)
            g = tl.minimum(acc1, swiglu_limit)
            u = tl.maximum(tl.minimum(acc3, swiglu_limit), -swiglu_limit)
            act = (g * tl.sigmoid(g)) * u * gate_scale
            tl.store(Act_out_ptr + e_idx * N_INTER + offs_n, act.to(tl.bfloat16))
        else:
            grp_id = eid // 8
            offs_sb_grp = offs_b // 8
            N_SCALES_GRP: tl.constexpr = N_BLOCKS_K * 2
            s_stride_grp = 2 * N_INTER * N_SCALES_GRP
            w1_grp_base = W_grp_gu_u8_ptr + grp_id * w_stride_slot + offs_n[:, None] * (N_BLOCKS_K * 16)
            w3_grp_base = W_grp_gu_u8_ptr + grp_id * w_stride_slot + (offs_n[:, None] + N_INTER) * (N_BLOCKS_K * 16)
            s1_grp_base = S_grp_gu_fp8_ptr + grp_id * s_stride_grp + offs_n[:, None] * N_SCALES_GRP
            s3_grp_base = S_grp_gu_fp8_ptr + grp_id * s_stride_grp + (offs_n[:, None] + N_INTER) * N_SCALES_GRP
            x_hrr_base = X_hrr_ptr + e_idx * (N_BLOCKS_K * 32)

            for sb_start in range(0, N_BLOCKS_K, sub_blocks_k):
                b_idx = sb_start * 16 + offs_b
                sb_idx = sb_start * 2 + offs_sb_grp
                x_even = tl.load(x_hrr_base + b_idx * 2).to(tl.float32)
                x_odd = tl.load(x_hrr_base + b_idx * 2 + 1).to(tl.float32)

                w1_b = tl.load(w1_grp_base + b_idx[None, :])
                s1_f = tl.load(s1_grp_base + sb_idx[None, :]).to(tl.float32)
                v1 = (_decode_e2m1_x4(w1_b & 0x0F) * x_even[None, :] + _decode_e2m1_x4(w1_b >> 4) * x_odd[None, :]) * s1_f
                acc1 += tl.sum(v1, axis=1)

                w3_b = tl.load(w3_grp_base + b_idx[None, :])
                s3_f = tl.load(s3_grp_base + sb_idx[None, :]).to(tl.float32)
                v3 = (_decode_e2m1_x4(w3_b & 0x0F) * x_even[None, :] + _decode_e2m1_x4(w3_b >> 4) * x_odd[None, :]) * s3_f
                acc3 += tl.sum(v3, axis=1)

            cold_s = tl.load(Cold_scale_gu_ptr + eid).to(tl.float32)
            rs2_1 = tl.load(Gu_r_s2_ptr + eid * (2 * N_INTER) + offs_n).to(tl.float32)
            rs2_3 = tl.load(Gu_r_s2_ptr + eid * (2 * N_INTER) + N_INTER + offs_n).to(tl.float32)
            tl.store(Z_hrr_out_ptr + e_idx * (2 * N_INTER) + offs_n, (acc1 * cold_s * rs2_1).to(tl.bfloat16))
            tl.store(Z_hrr_out_ptr + e_idx * (2 * N_INTER) + N_INTER + offs_n, (acc3 * cold_s * rs2_3).to(tl.bfloat16))

            offs_r = tl.arange(0, 64)
            h_mid = tl.load(H_gu_mid_ptr + e_idx * 64 + offs_r).to(tl.float32)
            u_base = U_gu_fp8_ptr + eid * (2 * N_INTER * 64)
            u1_tile = tl.load(u_base + offs_n[:, None] * 64 + offs_r[None, :]).to(tl.float32)
            u3_tile = tl.load(u_base + (offs_n[:, None] + N_INTER) * 64 + offs_r[None, :]).to(tl.float32)
            h1 = tl.sum(u1_tile * h_mid[None, :], axis=1)
            h3 = tl.sum(u3_tile * h_mid[None, :], axis=1)
            tl.store(Z_gu_healer_out_ptr + e_idx * (2 * N_INTER) + offs_n, h1.to(tl.bfloat16))
            tl.store(Z_gu_healer_out_ptr + e_idx * (2 * N_INTER) + N_INTER + offs_n, h3.to(tl.bfloat16))

    @triton.jit
    def _triton_dsv41_fwht_post_kernel(
        Buf_ptr,
        Top_ids_ptr,
        Hot_slot_map_ptr,
        D_OUT: tl.constexpr,
    ):
        blk_idx = tl.program_id(0)
        e_idx = tl.program_id(1)
        eid = tl.load(Top_ids_ptr + e_idx).to(tl.int64)
        hot_slot = tl.load(Hot_slot_map_ptr + eid)
        if hot_slot >= 0:
            return
        offs = blk_idx * 128 + tl.arange(0, 128)
        v = tl.load(Buf_ptr + e_idx * D_OUT + offs).to(tl.float32)
        u = _fwht128(v)
        tl.store(Buf_ptr + e_idx * D_OUT + offs, u.to(tl.bfloat16))

    @triton.jit
    def _triton_dsv41_cold_swiglu_kernel(
        Z_hrr_ptr,
        Z_gu_healer_ptr,
        Top_ids_ptr,
        Hot_slot_map_ptr,
        Gates_ptr,
        Gu_r_inv_ptr,
        Gu_r_s1_ptr,
        Act_out_ptr,
        swiglu_limit,
        N_INTER: tl.constexpr,
    ):
        blk_idx = tl.program_id(0)
        e_idx = tl.program_id(1)
        eid = tl.load(Top_ids_ptr + e_idx).to(tl.int64)
        hot_slot = tl.load(Hot_slot_map_ptr + eid)
        if hot_slot >= 0:
            return
        offs = blk_idx * 128 + tl.arange(0, 128)
        key_base1 = eid * (2 * N_INTER) + offs
        key_base3 = key_base1 + N_INTER
        inv1 = tl.load(Gu_r_inv_ptr + key_base1).to(tl.int64)
        inv3 = tl.load(Gu_r_inv_ptr + key_base3).to(tl.int64)
        s1_1 = tl.load(Gu_r_s1_ptr + key_base1).to(tl.float32)
        s1_3 = tl.load(Gu_r_s1_ptr + key_base3).to(tl.float32)

        z_hrr_base = Z_hrr_ptr + e_idx * (2 * N_INTER)
        z_hlr_base = Z_gu_healer_ptr + e_idx * (2 * N_INTER)
        acc1 = tl.load(z_hrr_base + inv1).to(tl.float32) * s1_1 + tl.load(z_hlr_base + offs).to(tl.float32)
        acc3 = tl.load(z_hrr_base + inv3).to(tl.float32) * s1_3 + tl.load(z_hlr_base + N_INTER + offs).to(tl.float32)

        gate_scale = tl.load(Gates_ptr + e_idx).to(tl.float32)
        g = tl.minimum(acc1, swiglu_limit)
        u = tl.maximum(tl.minimum(acc3, swiglu_limit), -swiglu_limit)
        act = (g * tl.sigmoid(g)) * u * gate_scale
        tl.store(Act_out_ptr + e_idx * N_INTER + offs, act.to(tl.bfloat16))

    @triton.jit
    def _triton_dsv41_resident_down_kernel(
        Act_ptr,
        Act_hrr_ptr,
        W_hot_d_u8_ptr,
        S_hot_d_bf16_ptr,
        W_grp_d_u8_ptr,
        S_grp_d_fp8_ptr,
        Cold_scale_d_ptr,
        Dn_r_s2_ptr,
        U_d_fp8_ptr,
        H_d_mid_ptr,
        Top_ids_ptr,
        Hot_slot_map_ptr,
        Y_hrr_out_ptr,
        Y_out_ptr,
        TOP_K: tl.constexpr,
        SHARED_SLOT: tl.constexpr,
        D_OUT: tl.constexpr,
        N_BLOCKS_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        sub_blocks_k: tl.constexpr,
    ):
        pid_d = tl.program_id(0)
        e_idx = tl.program_id(1)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

        if e_idx < TOP_K:
            eid = tl.load(Top_ids_ptr + e_idx).to(tl.int64)
            hot_slot = tl.load(Hot_slot_map_ptr + eid).to(tl.int64)
        else:
            eid = tl.zeros((), dtype=tl.int64)
            hot_slot = tl.zeros((), dtype=tl.int64) + SHARED_SLOT

        BYTES_PER_ITER: tl.constexpr = sub_blocks_k * 16
        offs_b = tl.arange(0, BYTES_PER_ITER)
        w_stride_slot = D_OUT * (N_BLOCKS_K * 16)

        if hot_slot >= 0:
            offs_sb = offs_b // 16
            act_hot_base = Act_ptr + e_idx * (N_BLOCKS_K * 32)
            wd_hot_base = W_hot_d_u8_ptr + hot_slot * w_stride_slot + offs_d[:, None] * (N_BLOCKS_K * 16)
            sd_hot_base = S_hot_d_bf16_ptr + hot_slot * (D_OUT * N_BLOCKS_K) + offs_d[:, None] * N_BLOCKS_K

            for sb_start in range(0, N_BLOCKS_K, sub_blocks_k):
                b_idx = sb_start * 16 + offs_b
                sb_idx = sb_start + offs_sb
                a_even = tl.load(act_hot_base + b_idx * 2).to(tl.float32)
                a_odd = tl.load(act_hot_base + b_idx * 2 + 1).to(tl.float32)
                wd_b = tl.load(wd_hot_base + b_idx[None, :])
                sd_f = tl.load(sd_hot_base + sb_idx[None, :]).to(tl.float32)
                vd = (_decode_e2m1_x4(wd_b & 0x0F) * a_even[None, :] + _decode_e2m1_x4(wd_b >> 4) * a_odd[None, :]) * sd_f
                acc += tl.sum(vd, axis=1)

            tl.store(Y_out_ptr + e_idx * D_OUT + offs_d, acc)
        else:
            grp_id = eid // 8
            offs_sb_grp = offs_b // 8
            N_SCALES_GRP: tl.constexpr = N_BLOCKS_K * 2
            act_hrr_base = Act_hrr_ptr + e_idx * (N_BLOCKS_K * 32)
            wd_grp_base = W_grp_d_u8_ptr + grp_id * w_stride_slot + offs_d[:, None] * (N_BLOCKS_K * 16)
            sd_grp_base = S_grp_d_fp8_ptr + grp_id * (D_OUT * N_SCALES_GRP) + offs_d[:, None] * N_SCALES_GRP

            for sb_start in range(0, N_BLOCKS_K, sub_blocks_k):
                b_idx = sb_start * 16 + offs_b
                sb_idx = sb_start * 2 + offs_sb_grp
                a_even = tl.load(act_hrr_base + b_idx * 2).to(tl.float32)
                a_odd = tl.load(act_hrr_base + b_idx * 2 + 1).to(tl.float32)
                wd_b = tl.load(wd_grp_base + b_idx[None, :])
                sd_f = tl.load(sd_grp_base + sb_idx[None, :]).to(tl.float32)
                vd = (_decode_e2m1_x4(wd_b & 0x0F) * a_even[None, :] + _decode_e2m1_x4(wd_b >> 4) * a_odd[None, :]) * sd_f
                acc += tl.sum(vd, axis=1)

            cold_s = tl.load(Cold_scale_d_ptr + eid).to(tl.float32)
            drs2 = tl.load(Dn_r_s2_ptr + eid * D_OUT + offs_d).to(tl.float32)
            tl.store(Y_hrr_out_ptr + e_idx * D_OUT + offs_d, (acc * cold_s * drs2).to(tl.bfloat16))

            offs_r = tl.arange(0, 64)
            h_d_mid = tl.load(H_d_mid_ptr + e_idx * 64 + offs_r).to(tl.float32)
            ud_tile = tl.load(U_d_fp8_ptr + eid * (D_OUT * 64) + offs_d[:, None] * 64 + offs_r[None, :]).to(tl.float32)
            hd = tl.sum(ud_tile * h_d_mid[None, :], axis=1)
            tl.store(Y_out_ptr + e_idx * D_OUT + offs_d, hd)

    @triton.jit
    def _triton_dsv41_reduce_down_kernel(
        Y_hrr_ptr,
        Y_out_ptr,
        Top_ids_ptr,
        Hot_slot_map_ptr,
        Dn_r_inv_ptr,
        Dn_r_s1_ptr,
        Y_final_bf16_ptr,
        TOP_K: tl.constexpr,
        D_OUT: tl.constexpr,
    ):
        blk_idx = tl.program_id(0)
        offs = blk_idx * 128 + tl.arange(0, 128)
        acc_routed = tl.load(Y_out_ptr + TOP_K * D_OUT + offs).to(tl.float32)

        for e_idx in range(TOP_K):
            eid = tl.load(Top_ids_ptr + e_idx).to(tl.int64)
            hot_slot = tl.load(Hot_slot_map_ptr + eid)
            y_base_val = tl.load(Y_out_ptr + e_idx * D_OUT + offs).to(tl.float32)
            if hot_slot >= 0:
                acc_routed += y_base_val
            else:
                key_base = eid * D_OUT + offs
                inv_d = tl.load(Dn_r_inv_ptr + key_base).to(tl.int64)
                s1_d = tl.load(Dn_r_s1_ptr + key_base).to(tl.float32)
                hrr_val = tl.load(Y_hrr_ptr + e_idx * D_OUT + inv_d).to(tl.float32) * s1_d
                acc_routed += hrr_val + y_base_val

        tl.store(Y_final_bf16_ptr + offs, acc_routed.to(tl.bfloat16))

    @triton.jit
    def _encode_pos_f32_to_fp8e4m3_u8(val_f32):
        v = tl.minimum(tl.maximum(val_f32, 0.0), 448.0)
        is_sub = v < 0.015625
        sub_u8 = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.rint(v * 512.0), 0.0), 7.0).to(tl.uint8)
        bits = v.to(tl.int32, bitcast=True) + (1 << 19)
        exp4 = tl.minimum(tl.maximum(((bits >> 23) & 0xFF) - 120, 1), 15)
        man3 = (bits >> 20) & 0x07
        norm_u8 = tl.minimum((exp4 << 3) | man3, 126).to(tl.uint8)
        return tl.where(is_sub, sub_u8, norm_u8)

    @triton.jit
    def _decode_pos_fp8e4m3_u8_to_f32(u8_val):
        u = u8_val.to(tl.int32) & 0x7F
        norm_f = ((u + 960) << 20).to(tl.float32, bitcast=True)
        sub_f = u.to(tl.float32) * 0.001953125
        return tl.where(u < 8, sub_f, norm_f)

    @triton.jit
    def _encode_ufp8_e5m3(val_f32):
        v = tl.minimum(tl.maximum(val_f32, 0.0), 480.0)
        bits = v.to(tl.int32, bitcast=True) + (1 << 19)
        u = tl.minimum(tl.maximum((bits >> 20) - 832, 1), 255).to(tl.uint8)
        return tl.where(v > 1.3411045e-7, u, tl.zeros_like(u))

    @triton.jit
    def _decode_ufp8_e5m3(u8_val):
        u = u8_val.to(tl.int32) & 0xFF
        f = ((u + 832) << 20).to(tl.float32, bitcast=True)
        return tl.where(u == 0, 0.0, f)

    @triton.jit
    def _triton_quantize_2bit_g64_kernel(
        W_ptr,
        Out_u8_ptr,
        Out_s_fp8_ptr,
        N_GROUPS_C: tl.constexpr,
    ):
        row = tl.program_id(0)
        grp = tl.program_id(1)
        offs16 = tl.arange(0, 16)
        base_w = W_ptr + row * (N_GROUPS_C * 64) + grp * 64
        w0 = tl.load(base_w + offs16 + 0).to(tl.float32)
        w1 = tl.load(base_w + offs16 + 16).to(tl.float32)
        w2 = tl.load(base_w + offs16 + 32).to(tl.float32)
        w3 = tl.load(base_w + offs16 + 48).to(tl.float32)

        sq_sum = tl.sum(w0 * w0 + w1 * w1 + w2 * w2 + w3 * w3, axis=0)
        sigma = tl.sqrt(sq_sum / 64.0 + 1e-12)
        inv_sig = 1.0 / sigma

        z0 = w0 * inv_sig
        z1 = w1 * inv_sig
        z2 = w2 * inv_sig
        z3 = w3 * inv_sig

        par0 = ((row + grp + offs16) & 1) == 0
        mid0 = tl.where(par0, z0 > 0.0, z0 >= 0.0)
        mid1 = tl.where(par0, z1 >= 0.0, z1 > 0.0)
        mid2 = tl.where(par0, z2 > 0.0, z2 >= 0.0)
        mid3 = tl.where(par0, z3 >= 0.0, z3 > 0.0)

        c0 = (tl.where(z0 >= -0.996, 1, 0) + tl.where(mid0, 1, 0) + tl.where(z0 >= 0.996, 1, 0)).to(tl.uint8)
        c1 = (tl.where(z1 >= -0.996, 1, 0) + tl.where(mid1, 1, 0) + tl.where(z1 >= 0.996, 1, 0)).to(tl.uint8)
        c2 = (tl.where(z2 >= -0.996, 1, 0) + tl.where(mid2, 1, 0) + tl.where(z2 >= 0.996, 1, 0)).to(tl.uint8)
        c3 = (tl.where(z3 >= -0.996, 1, 0) + tl.where(mid3, 1, 0) + tl.where(z3 >= 0.996, 1, 0)).to(tl.uint8)

        q0 = c0.to(tl.float32) - 1.5
        q1 = c1.to(tl.float32) - 1.5
        q2 = c2.to(tl.float32) - 1.5
        q3 = c3.to(tl.float32) - 1.5
        wq_dot = tl.sum(w0 * q0 + w1 * q1 + w2 * q2 + w3 * q3, axis=0)
        q_sq = tl.sum(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3, axis=0)
        packed = c0 | (c1 << 2) | (c2 << 4) | (c3 << 6)
        s_norm = _encode_ufp8_e5m3(tl.maximum(wq_dot, 0.0) / (q_sq + 1e-12))
        tl.store(Out_s_fp8_ptr + row * N_GROUPS_C + grp, s_norm)
        tl.store(Out_u8_ptr + row * (N_GROUPS_C * 16) + grp * 16 + offs16, packed)

    @triton.jit
    def _triton_quantize_1bit_g64_kernel(
        W_ptr,
        Out_u8_ptr,
        Out_s_fp8_ptr,
        N_GROUPS_C: tl.constexpr,
    ):
        row = tl.program_id(0)
        grp = tl.program_id(1)
        offs8 = tl.arange(0, 8)
        base_w = W_ptr + row * (N_GROUPS_C * 64) + grp * 64
        w0 = tl.load(base_w + offs8 + 0).to(tl.float32)
        w1 = tl.load(base_w + offs8 + 8).to(tl.float32)
        w2 = tl.load(base_w + offs8 + 16).to(tl.float32)
        w3 = tl.load(base_w + offs8 + 24).to(tl.float32)
        w4 = tl.load(base_w + offs8 + 32).to(tl.float32)
        w5 = tl.load(base_w + offs8 + 40).to(tl.float32)
        w6 = tl.load(base_w + offs8 + 48).to(tl.float32)
        w7 = tl.load(base_w + offs8 + 56).to(tl.float32)

        l1_sum = tl.sum(
            tl.abs(w0)
            + tl.abs(w1)
            + tl.abs(w2)
            + tl.abs(w3)
            + tl.abs(w4)
            + tl.abs(w5)
            + tl.abs(w6)
            + tl.abs(w7),
            axis=0,
        )
        s_l1 = _encode_ufp8_e5m3(l1_sum * (1.0 / 64.0))

        par1 = ((row + grp + offs8) & 1) == 0
        b0 = tl.where(tl.where(par1, w0 > 0.0, w0 >= 0.0), 1, 0).to(tl.uint8)
        b1 = tl.where(tl.where(par1, w1 >= 0.0, w1 > 0.0), 2, 0).to(tl.uint8)
        b2 = tl.where(tl.where(par1, w2 > 0.0, w2 >= 0.0), 4, 0).to(tl.uint8)
        b3 = tl.where(tl.where(par1, w3 >= 0.0, w3 > 0.0), 8, 0).to(tl.uint8)
        b4 = tl.where(tl.where(par1, w4 > 0.0, w4 >= 0.0), 16, 0).to(tl.uint8)
        b5 = tl.where(tl.where(par1, w5 >= 0.0, w5 > 0.0), 32, 0).to(tl.uint8)
        b6 = tl.where(tl.where(par1, w6 > 0.0, w6 >= 0.0), 64, 0).to(tl.uint8)
        b7 = tl.where(tl.where(par1, w7 >= 0.0, w7 > 0.0), 128, 0).to(tl.uint8)
        packed = b0 | b1 | b2 | b3 | b4 | b5 | b6 | b7

        tl.store(Out_s_fp8_ptr + row * N_GROUPS_C + grp, s_l1)
        tl.store(Out_u8_ptr + row * (N_GROUPS_C * 8) + grp * 8 + offs8, packed)

    @triton.jit
    def _triton_row_perm_scale_kernel(
        In_ptr,
        Out_ptr,
        R_inv_ptr,
        R_s1_ptr,
        C_s2_ptr,
        COLS: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        row = tl.program_id(0)
        pid_c = tl.program_id(1)
        offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        src_row = tl.load(R_inv_ptr + row).to(tl.int64)
        r_s = tl.load(R_s1_ptr + row).to(tl.float32)
        c_s = tl.load(C_s2_ptr + offs_c).to(tl.float32)
        val = tl.load(In_ptr + src_row * COLS + offs_c).to(tl.float32)
        tl.store(Out_ptr + row * COLS + offs_c, (val * (r_s * c_s)).to(tl.bfloat16))

    @triton.jit
    def _triton_col_perm_add_tail_kernel(
        In_ptr,
        Tail_p_ptr,
        Tail_s_ptr,
        Out_ptr,
        C_inv_ptr,
        C_s1_ptr,
        Beta_ptr,
        COLS: tl.constexpr,
        TAIL_BITS: tl.constexpr,
        HAS_TAIL: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        row = tl.program_id(0)
        pid_c = tl.program_id(1)
        offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        src_col = tl.load(C_inv_ptr + offs_c).to(tl.int64)
        c_s = tl.load(C_s1_ptr + offs_c).to(tl.float32)
        beta = tl.load(Beta_ptr).to(tl.float32)
        h_val = tl.load(In_ptr + row * COLS + src_col).to(tl.float32) * (c_s * beta)

        if HAS_TAIL:
            if TAIL_BITS == 2:
                byte_idx = offs_c >> 2
                shift = (offs_c & 3) << 1
                packed = tl.load(Tail_p_ptr + row * (COLS // 4) + byte_idx)
                code = (packed >> shift) & 3
                q_val = tl.where(
                    code < 2,
                    tl.where(code == 0, -1.510417, -0.452780),
                    tl.where(code == 2, 0.452780, 1.510417),
                )
                s_idx = offs_c >> 5
                t_scale = _decode_pos_fp8e4m3_u8_to_f32(tl.load(Tail_s_ptr + row * (COLS // 32) + s_idx))
                h_val = h_val + q_val * t_scale
            else:
                byte_idx = offs_c >> 3
                shift = offs_c & 7
                packed = tl.load(Tail_p_ptr + row * (COLS // 8) + byte_idx)
                bit = (packed >> shift) & 1
                q_val = tl.where(bit != 0, 1.0, -1.0)
                s_idx = offs_c >> 6
                t_scale = _decode_pos_fp8e4m3_u8_to_f32(tl.load(Tail_s_ptr + row * (COLS // 64) + s_idx))
                h_val = h_val + q_val * t_scale

        tl.store(Out_ptr + row * COLS + offs_c, h_val.to(tl.bfloat16))

    @triton.jit
    def _triton_dsv41_tiered_router_kernel(
        Logits_ptr,
        Bias_ptr,
        Tier_slot_map_ptr,
        Active_slots_ptr,
        Gates_ptr,
        routed_scaling_factor,
        SHARED_SLOT: tl.constexpr,
        K_ACT: tl.constexpr,
    ):
        offs = tl.arange(0, 512)
        mask = offs < 384
        logit_raw = tl.load(Logits_ptr + offs, mask=mask, other=-1e30).to(tl.float32)
        logit = tl.where(logit_raw == logit_raw, logit_raw, 0.0)
        sp = tl.where(logit > 20.0, logit, tl.log(1.0 + tl.exp(tl.maximum(logit, -80.0))))
        score = tl.sqrt(tl.maximum(sp, 0.0))
        bias_raw = tl.load(Bias_ptr + offs, mask=mask, other=-1e30).to(tl.float32)
        bias = tl.where(bias_raw == bias_raw, bias_raw, 0.0)
        tslot_all = tl.load(Tier_slot_map_ptr + offs, mask=mask, other=-1).to(tl.int32)
        valid_mask = mask & (tslot_all >= 0)
        biased = tl.where(valid_mask, score + bias, -1e30)

        k_offs = tl.arange(0, 8)
        kept_vec = tl.zeros((8,), dtype=tl.float32)
        for k in tl.static_range(K_ACT):
            best_idx = tl.argmax(biased, axis=0)
            in_bounds = (best_idx >= 0) & (best_idx < 384)
            w_k = tl.max(tl.where(offs == best_idx, score, 0.0), axis=0)
            tslot = tl.load(Tier_slot_map_ptr + best_idx, mask=in_bounds, other=-1).to(tl.int32)
            tl.store(Active_slots_ptr + k, tslot)
            is_kept = tl.where(tslot >= 0, w_k, 0.0)
            kept_vec = tl.where(k_offs == k, is_kept, kept_vec)
            biased = tl.where(offs == best_idx, -1e30, biased)

        w_kept_sum = tl.sum(kept_vec, axis=0) + 1e-20
        gates = (kept_vec / w_kept_sum) * routed_scaling_factor
        tl.store(Gates_ptr + k_offs, gates, mask=k_offs < K_ACT)
        tl.store(Active_slots_ptr + K_ACT, SHARED_SLOT)
        tl.store(Gates_ptr + K_ACT, 1.0)

    @triton.jit
    def _triton_dsv41_tiered_gu_swiglu_kernel(
        X_ptr,
        W4_gu_u8_ptr,
        S4_gu_bf16_ptr,
        W2_gu_u8_ptr,
        S2_gu_fp8_ptr,
        W1_gu_u8_ptr,
        S1_gu_fp8_ptr,
        Active_slots_ptr,
        Gates_ptr,
        Act_out_ptr,
        swiglu_limit,
        N_INTER: tl.constexpr,
        N_BLOCKS_K4: tl.constexpr,
        N_GROUPS_K2: tl.constexpr,
        BLOCK_N: tl.constexpr,
        sub_blocks_k4: tl.constexpr,
        sub_groups_k2: tl.constexpr,
        HAS_TIER2: tl.constexpr = False,
        HAS_TIER3: tl.constexpr = False,
    ):
        pid_n = tl.program_id(0)
        e_idx = tl.program_id(1)
        tslot = tl.load(Active_slots_ptr + e_idx).to(tl.int64)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        if tslot < 0:
            tl.store(Act_out_ptr + e_idx * N_INTER + offs_n, tl.zeros((BLOCK_N,), dtype=tl.bfloat16))
            return

        acc1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
        acc3 = tl.zeros((BLOCK_N,), dtype=tl.float32)

        if tslot < 1024 or (not HAS_TIER2 and not HAS_TIER3):
            slot4 = tslot
            BYTES_PER_ITER4: tl.constexpr = sub_blocks_k4 * 16
            offs_b4 = tl.arange(0, BYTES_PER_ITER4)
            offs_sb4 = offs_b4 // 16
            w_stride4 = 2 * N_INTER * (N_BLOCKS_K4 * 16)
            s_stride4 = 2 * N_INTER * N_BLOCKS_K4
            w1_base4 = W4_gu_u8_ptr + slot4 * w_stride4 + offs_n[:, None] * (N_BLOCKS_K4 * 16)
            w3_base4 = W4_gu_u8_ptr + slot4 * w_stride4 + (offs_n[:, None] + N_INTER) * (N_BLOCKS_K4 * 16)
            s1_base4 = S4_gu_bf16_ptr + slot4 * s_stride4 + offs_n[:, None] * N_BLOCKS_K4
            s3_base4 = S4_gu_bf16_ptr + slot4 * s_stride4 + (offs_n[:, None] + N_INTER) * N_BLOCKS_K4

            for sb_start in range(0, N_BLOCKS_K4, sub_blocks_k4):
                b_idx = sb_start * 16 + offs_b4
                sb_idx = sb_start + offs_sb4
                x_even = tl.load(X_ptr + b_idx * 2).to(tl.float32)
                x_odd = tl.load(X_ptr + b_idx * 2 + 1).to(tl.float32)
                w1_b = tl.load(w1_base4 + b_idx[None, :])
                s1_f = tl.load(s1_base4 + sb_idx[None, :]).to(tl.float32)
                v1 = (_decode_e2m1_x4(w1_b & 0x0F) * x_even[None, :] + _decode_e2m1_x4(w1_b >> 4) * x_odd[None, :]) * s1_f
                acc1 += tl.sum(v1, axis=1)
                w3_b = tl.load(w3_base4 + b_idx[None, :])
                s3_f = tl.load(s3_base4 + sb_idx[None, :]).to(tl.float32)
                v3 = (_decode_e2m1_x4(w3_b & 0x0F) * x_even[None, :] + _decode_e2m1_x4(w3_b >> 4) * x_odd[None, :]) * s3_f
                acc3 += tl.sum(v3, axis=1)
        elif HAS_TIER2 and (tslot < 2048 or not HAS_TIER3):
            slot2 = tslot - 1024
            BYTES_PER_ITER2: tl.constexpr = sub_groups_k2 * 16
            offs_b2 = tl.arange(0, BYTES_PER_ITER2)
            offs_sg2 = offs_b2 // 16
            offs_in16 = offs_b2 & 15
            w_stride2 = 2 * N_INTER * (N_GROUPS_K2 * 16)
            s_stride2 = 2 * N_INTER * N_GROUPS_K2
            w1_base2 = W2_gu_u8_ptr + slot2 * w_stride2 + offs_n[:, None] * (N_GROUPS_K2 * 16)
            w3_base2 = W2_gu_u8_ptr + slot2 * w_stride2 + (offs_n[:, None] + N_INTER) * (N_GROUPS_K2 * 16)
            s1_base2 = S2_gu_fp8_ptr + slot2 * s_stride2 + offs_n[:, None] * N_GROUPS_K2
            s3_base2 = S2_gu_fp8_ptr + slot2 * s_stride2 + (offs_n[:, None] + N_INTER) * N_GROUPS_K2

            for sg_start in range(0, N_GROUPS_K2, sub_groups_k2):
                b_idx = sg_start * 16 + offs_b2
                sg_idx = sg_start + offs_sg2
                x_base_idx = sg_idx * 64 + offs_in16
                x0 = tl.load(X_ptr + x_base_idx + 0).to(tl.float32)
                x1 = tl.load(X_ptr + x_base_idx + 16).to(tl.float32)
                x2 = tl.load(X_ptr + x_base_idx + 32).to(tl.float32)
                x3 = tl.load(X_ptr + x_base_idx + 48).to(tl.float32)
                x_sum = (x0 + x1 + x2 + x3) * 1.5

                w1_b = tl.load(w1_base2 + b_idx[None, :])
                s1_f = _decode_ufp8_e5m3(tl.load(s1_base2 + sg_idx[None, :]))
                v1 = (
                    (w1_b & 3).to(tl.float32) * x0[None, :]
                    + ((w1_b >> 2) & 3).to(tl.float32) * x1[None, :]
                    + ((w1_b >> 4) & 3).to(tl.float32) * x2[None, :]
                    + (w1_b >> 6).to(tl.float32) * x3[None, :]
                    - x_sum[None, :]
                ) * s1_f
                acc1 += tl.sum(v1, axis=1)

                w3_b = tl.load(w3_base2 + b_idx[None, :])
                s3_f = _decode_ufp8_e5m3(tl.load(s3_base2 + sg_idx[None, :]))
                v3 = (
                    (w3_b & 3).to(tl.float32) * x0[None, :]
                    + ((w3_b >> 2) & 3).to(tl.float32) * x1[None, :]
                    + ((w3_b >> 4) & 3).to(tl.float32) * x2[None, :]
                    + (w3_b >> 6).to(tl.float32) * x3[None, :]
                    - x_sum[None, :]
                ) * s3_f
                acc3 += tl.sum(v3, axis=1)
        else:
            slot1 = tslot - 2048
            BYTES_PER_ITER1: tl.constexpr = sub_groups_k2 * 8
            offs_b1 = tl.arange(0, BYTES_PER_ITER1)
            offs_sg1 = offs_b1 // 8
            offs_in8 = offs_b1 & 7
            w_stride1 = 2 * N_INTER * (N_GROUPS_K2 * 8)
            s_stride1 = 2 * N_INTER * N_GROUPS_K2
            w1_base1 = W1_gu_u8_ptr + slot1 * w_stride1 + offs_n[:, None] * (N_GROUPS_K2 * 8)
            w3_base1 = W1_gu_u8_ptr + slot1 * w_stride1 + (offs_n[:, None] + N_INTER) * (N_GROUPS_K2 * 8)
            s1_base1 = S1_gu_fp8_ptr + slot1 * s_stride1 + offs_n[:, None] * N_GROUPS_K2
            s3_base1 = S1_gu_fp8_ptr + slot1 * s_stride1 + (offs_n[:, None] + N_INTER) * N_GROUPS_K2

            for sg_start in range(0, N_GROUPS_K2, sub_groups_k2):
                b_idx = sg_start * 8 + offs_b1
                sg_idx = sg_start + offs_sg1
                x_base_idx = sg_idx * 64 + offs_in8
                x0 = tl.load(X_ptr + x_base_idx + 0).to(tl.float32)
                x1 = tl.load(X_ptr + x_base_idx + 8).to(tl.float32)
                x2 = tl.load(X_ptr + x_base_idx + 16).to(tl.float32)
                x3 = tl.load(X_ptr + x_base_idx + 24).to(tl.float32)
                x4 = tl.load(X_ptr + x_base_idx + 32).to(tl.float32)
                x5 = tl.load(X_ptr + x_base_idx + 40).to(tl.float32)
                x6 = tl.load(X_ptr + x_base_idx + 48).to(tl.float32)
                x7 = tl.load(X_ptr + x_base_idx + 56).to(tl.float32)
                x_half_sum = (x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7) * 0.5

                w1_b = tl.load(w1_base1 + b_idx[None, :])
                s1_f2 = _decode_ufp8_e5m3(tl.load(s1_base1 + sg_idx[None, :])) * 2.0
                v1 = (
                    (w1_b & 1).to(tl.float32) * x0[None, :]
                    + ((w1_b >> 1) & 1).to(tl.float32) * x1[None, :]
                    + ((w1_b >> 2) & 1).to(tl.float32) * x2[None, :]
                    + ((w1_b >> 3) & 1).to(tl.float32) * x3[None, :]
                    + ((w1_b >> 4) & 1).to(tl.float32) * x4[None, :]
                    + ((w1_b >> 5) & 1).to(tl.float32) * x5[None, :]
                    + ((w1_b >> 6) & 1).to(tl.float32) * x6[None, :]
                    + (w1_b >> 7).to(tl.float32) * x7[None, :]
                    - x_half_sum[None, :]
                ) * s1_f2
                acc1 += tl.sum(v1, axis=1)

                w3_b = tl.load(w3_base1 + b_idx[None, :])
                s3_f2 = _decode_ufp8_e5m3(tl.load(s3_base1 + sg_idx[None, :])) * 2.0
                v3 = (
                    (w3_b & 1).to(tl.float32) * x0[None, :]
                    + ((w3_b >> 1) & 1).to(tl.float32) * x1[None, :]
                    + ((w3_b >> 2) & 1).to(tl.float32) * x2[None, :]
                    + ((w3_b >> 3) & 1).to(tl.float32) * x3[None, :]
                    + ((w3_b >> 4) & 1).to(tl.float32) * x4[None, :]
                    + ((w3_b >> 5) & 1).to(tl.float32) * x5[None, :]
                    + ((w3_b >> 6) & 1).to(tl.float32) * x6[None, :]
                    + (w3_b >> 7).to(tl.float32) * x7[None, :]
                    - x_half_sum[None, :]
                ) * s3_f2
                acc3 += tl.sum(v3, axis=1)

        gate_scale = tl.load(Gates_ptr + e_idx).to(tl.float32)
        g = tl.minimum(acc1, swiglu_limit)
        u = tl.maximum(tl.minimum(acc3, swiglu_limit), -swiglu_limit)
        act = (g * tl.sigmoid(g)) * u * gate_scale
        tl.store(Act_out_ptr + e_idx * N_INTER + offs_n, act.to(tl.bfloat16))

    @triton.jit
    def _triton_dsv41_tiered_down_kernel(
        Act_ptr,
        W4_d_u8_ptr,
        S4_d_bf16_ptr,
        W2_d_u8_ptr,
        S2_d_fp8_ptr,
        W1_d_u8_ptr,
        S1_d_fp8_ptr,
        Active_slots_ptr,
        Y_partial_ptr,
        D_OUT: tl.constexpr,
        N_BLOCKS_K4: tl.constexpr,
        N_GROUPS_K2: tl.constexpr,
        BLOCK_D: tl.constexpr,
        sub_blocks_k4: tl.constexpr,
        sub_groups_k2: tl.constexpr,
        HAS_TIER2: tl.constexpr = False,
        HAS_TIER3: tl.constexpr = False,
    ):
        pid_d = tl.program_id(0)
        e_idx = tl.program_id(1)
        tslot = tl.load(Active_slots_ptr + e_idx).to(tl.int64)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        if tslot < 0:
            tl.store(Y_partial_ptr + e_idx * D_OUT + offs_d, tl.zeros((BLOCK_D,), dtype=tl.float32))
            return

        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        act_base = Act_ptr + e_idx * (N_BLOCKS_K4 * 32)

        if tslot < 1024 or (not HAS_TIER2 and not HAS_TIER3):
            slot4 = tslot
            BYTES_PER_ITER4: tl.constexpr = sub_blocks_k4 * 16
            offs_b4 = tl.arange(0, BYTES_PER_ITER4)
            offs_sb4 = offs_b4 // 16
            w_stride4 = D_OUT * (N_BLOCKS_K4 * 16)
            s_stride4 = D_OUT * N_BLOCKS_K4
            wd_base4 = W4_d_u8_ptr + slot4 * w_stride4 + offs_d[:, None] * (N_BLOCKS_K4 * 16)
            sd_base4 = S4_d_bf16_ptr + slot4 * s_stride4 + offs_d[:, None] * N_BLOCKS_K4

            for sb_start in range(0, N_BLOCKS_K4, sub_blocks_k4):
                b_idx = sb_start * 16 + offs_b4
                sb_idx = sb_start + offs_sb4
                a_even = tl.load(act_base + b_idx * 2).to(tl.float32)
                a_odd = tl.load(act_base + b_idx * 2 + 1).to(tl.float32)
                wd_b = tl.load(wd_base4 + b_idx[None, :])
                sd_f = tl.load(sd_base4 + sb_idx[None, :]).to(tl.float32)
                vd = (_decode_e2m1_x4(wd_b & 0x0F) * a_even[None, :] + _decode_e2m1_x4(wd_b >> 4) * a_odd[None, :]) * sd_f
                acc += tl.sum(vd, axis=1)
        elif HAS_TIER2 and (tslot < 2048 or not HAS_TIER3):
            slot2 = tslot - 1024
            BYTES_PER_ITER2: tl.constexpr = sub_groups_k2 * 16
            offs_b2 = tl.arange(0, BYTES_PER_ITER2)
            offs_sg2 = offs_b2 // 16
            offs_in16 = offs_b2 & 15
            w_stride2 = D_OUT * (N_GROUPS_K2 * 16)
            s_stride2 = D_OUT * N_GROUPS_K2
            wd_base2 = W2_d_u8_ptr + slot2 * w_stride2 + offs_d[:, None] * (N_GROUPS_K2 * 16)
            sd_base2 = S2_d_fp8_ptr + slot2 * s_stride2 + offs_d[:, None] * N_GROUPS_K2

            for sg_start in range(0, N_GROUPS_K2, sub_groups_k2):
                b_idx = sg_start * 16 + offs_b2
                sg_idx = sg_start + offs_sg2
                a_base_idx = sg_idx * 64 + offs_in16
                a0 = tl.load(act_base + a_base_idx + 0).to(tl.float32)
                a1 = tl.load(act_base + a_base_idx + 16).to(tl.float32)
                a2 = tl.load(act_base + a_base_idx + 32).to(tl.float32)
                a3 = tl.load(act_base + a_base_idx + 48).to(tl.float32)
                a_sum = (a0 + a1 + a2 + a3) * 1.5
                wd_b = tl.load(wd_base2 + b_idx[None, :])
                sd_f = _decode_ufp8_e5m3(tl.load(sd_base2 + sg_idx[None, :]))
                vd = (
                    (wd_b & 3).to(tl.float32) * a0[None, :]
                    + ((wd_b >> 2) & 3).to(tl.float32) * a1[None, :]
                    + ((wd_b >> 4) & 3).to(tl.float32) * a2[None, :]
                    + (wd_b >> 6).to(tl.float32) * a3[None, :]
                    - a_sum[None, :]
                ) * sd_f
                acc += tl.sum(vd, axis=1)
        else:
            slot1 = tslot - 2048
            BYTES_PER_ITER1: tl.constexpr = sub_groups_k2 * 8
            offs_b1 = tl.arange(0, BYTES_PER_ITER1)
            offs_sg1 = offs_b1 // 8
            offs_in8 = offs_b1 & 7
            w_stride1 = D_OUT * (N_GROUPS_K2 * 8)
            s_stride1 = D_OUT * N_GROUPS_K2
            wd_base1 = W1_d_u8_ptr + slot1 * w_stride1 + offs_d[:, None] * (N_GROUPS_K2 * 8)
            sd_base1 = S1_d_fp8_ptr + slot1 * s_stride1 + offs_d[:, None] * N_GROUPS_K2

            for sg_start in range(0, N_GROUPS_K2, sub_groups_k2):
                b_idx = sg_start * 8 + offs_b1
                sg_idx = sg_start + offs_sg1
                a_base_idx = sg_idx * 64 + offs_in8
                a0 = tl.load(act_base + a_base_idx + 0).to(tl.float32)
                a1 = tl.load(act_base + a_base_idx + 8).to(tl.float32)
                a2 = tl.load(act_base + a_base_idx + 16).to(tl.float32)
                a3 = tl.load(act_base + a_base_idx + 24).to(tl.float32)
                a4 = tl.load(act_base + a_base_idx + 32).to(tl.float32)
                a5 = tl.load(act_base + a_base_idx + 40).to(tl.float32)
                a6 = tl.load(act_base + a_base_idx + 48).to(tl.float32)
                a7 = tl.load(act_base + a_base_idx + 56).to(tl.float32)
                a_half_sum = (a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7) * 0.5
                wd_b = tl.load(wd_base1 + b_idx[None, :])
                sd_f2 = _decode_ufp8_e5m3(tl.load(sd_base1 + sg_idx[None, :])) * 2.0
                vd = (
                    (wd_b & 1).to(tl.float32) * a0[None, :]
                    + ((wd_b >> 1) & 1).to(tl.float32) * a1[None, :]
                    + ((wd_b >> 2) & 1).to(tl.float32) * a2[None, :]
                    + ((wd_b >> 3) & 1).to(tl.float32) * a3[None, :]
                    + ((wd_b >> 4) & 1).to(tl.float32) * a4[None, :]
                    + ((wd_b >> 5) & 1).to(tl.float32) * a5[None, :]
                    + ((wd_b >> 6) & 1).to(tl.float32) * a6[None, :]
                    + (wd_b >> 7).to(tl.float32) * a7[None, :]
                    - a_half_sum[None, :]
                ) * sd_f2
                acc += tl.sum(vd, axis=1)

        tl.store(Y_partial_ptr + e_idx * D_OUT + offs_d, acc)

    _triton_moe_w2_grouped_gemm_m1_kernel = _triton_dsv41_tiered_gu_swiglu_kernel
    _triton_moe_w2_down_gemm_m1_kernel = _triton_dsv41_tiered_down_kernel

    @triton.jit
    def _triton_dsv41_reduce_sum_bf16_kernel(
        Y_partial_ptr,
        Y_out_bf16_ptr,
        N_ACT: tl.constexpr,
        D_OUT: tl.constexpr,
    ):
        blk = tl.program_id(0)
        offs = blk * 128 + tl.arange(0, 128)
        acc = tl.zeros((128,), dtype=tl.float32)
        for e_i in tl.static_range(N_ACT):
            acc += tl.load(Y_partial_ptr + e_i * D_OUT + offs)
        tl.store(Y_out_bf16_ptr + offs, acc.to(tl.bfloat16))

    @triton.jit
    def _triton_stage_t3_dma_kernel(
        Active_slots_ptr,
        W1_gu_dst_ptr,
        W1_gu_src_ptr,
        S1_gu_dst_ptr,
        S1_gu_src_ptr,
        W1_d_dst_ptr,
        W1_d_src_ptr,
        S1_d_dst_ptr,
        S1_d_src_ptr,
        K_ACT: tl.constexpr,
        BLOCK_BYTES: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_pids = tl.num_programs(0)
        b_idx = 0
        for k in range(0, K_ACT):
            s_val = tl.load(Active_slots_ptr + k)
            if s_val >= 2048:
                s1 = (s_val - 2048).to(tl.int64)
                dst_slot = b_idx
                gu_bytes = 4608 * 640
                for off in range(pid * BLOCK_BYTES, gu_bytes, num_pids * BLOCK_BYTES):
                    cols = (off + tl.arange(0, BLOCK_BYTES)).to(tl.int64)
                    m = cols < gu_bytes
                    v = tl.load(W1_gu_src_ptr + s1 * gu_bytes + cols, mask=m)
                    tl.store(W1_gu_dst_ptr + dst_slot * gu_bytes + cols, v, mask=m)
                dn_bytes = 5120 * 288
                for off in range(pid * BLOCK_BYTES, dn_bytes, num_pids * BLOCK_BYTES):
                    cols = (off + tl.arange(0, BLOCK_BYTES)).to(tl.int64)
                    m = cols < dn_bytes
                    v = tl.load(W1_d_src_ptr + s1 * dn_bytes + cols, mask=m)
                    tl.store(W1_d_dst_ptr + dst_slot * dn_bytes + cols, v, mask=m)
                sgu_bytes = 4608 * 80
                for off in range(pid * BLOCK_BYTES, sgu_bytes, num_pids * BLOCK_BYTES):
                    cols = (off + tl.arange(0, BLOCK_BYTES)).to(tl.int64)
                    m = cols < sgu_bytes
                    v = tl.load(S1_gu_src_ptr + s1 * sgu_bytes + cols, mask=m)
                    tl.store(S1_gu_dst_ptr + dst_slot * sgu_bytes + cols, v, mask=m)
                sdn_bytes = 5120 * 36
                for off in range(pid * BLOCK_BYTES, sdn_bytes, num_pids * BLOCK_BYTES):
                    cols = (off + tl.arange(0, BLOCK_BYTES)).to(tl.int64)
                    m = cols < sdn_bytes
                    v = tl.load(S1_d_src_ptr + s1 * sdn_bytes + cols, mask=m)
                    tl.store(S1_d_dst_ptr + dst_slot * sdn_bytes + cols, v, mask=m)
                b_idx += 1

    @triton.jit
    def _triton_remap_t3_slots_kernel(Active_slots_ptr, K_ACT: tl.constexpr):
        b_idx = 0
        for k in range(0, K_ACT):
            s_val = tl.load(Active_slots_ptr + k)
            if s_val >= 2048:
                tl.store(Active_slots_ptr + k, 2048 + b_idx)
                b_idx += 1

    @triton.jit
    def _triton_inv_rope_m1_kernel(
        O_ptr,
        CosSin_ptr,
        Positions_ptr,
        Out_ptr,
        stride_h,
    ):
        head_idx = tl.program_id(0)
        pos = tl.load(Positions_ptr).to(tl.int64)
        cs_base = CosSin_ptr + pos * 64
        o_base = O_ptr + head_idx * stride_h
        out_base = Out_ptr + head_idx * 512

        offs128 = tl.arange(0, 128)
        for c in tl.static_range(3):
            v = tl.load(o_base + c * 128 + offs128)
            tl.store(out_base + c * 128 + offs128, v)
        offs64 = tl.arange(0, 64)
        v_last = tl.load(o_base + 384 + offs64)
        tl.store(out_base + 384 + offs64, v_last)

        offs32 = tl.arange(0, 32)
        cos = tl.load(cs_base + offs32).to(tl.float32)
        sin = tl.load(cs_base + 32 + offs32).to(tl.float32)
        r_even = tl.load(o_base + 448 + offs32 * 2).to(tl.float32)
        r_odd = tl.load(o_base + 448 + offs32 * 2 + 1).to(tl.float32)
        out_e = (r_even * cos + r_odd * sin).to(tl.bfloat16)
        out_o = (r_odd * cos - r_even * sin).to(tl.bfloat16)
        tl.store(out_base + 448 + offs32 * 2, out_e)
        tl.store(out_base + 448 + offs32 * 2 + 1, out_o)

    @triton.jit
    def _triton_bmm_fp4_m1_kernel(
        X_ptr,
        W_u8_ptr,
        S_bf16_ptr,
        Z_out_ptr,
        N_OUT: tl.constexpr,
        N_BLOCKS_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
        sub_blocks_k: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        g_idx = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        BYTES_PER_ITER: tl.constexpr = sub_blocks_k * 16
        offs_b = tl.arange(0, BYTES_PER_ITER)
        offs_sb = offs_b // 16

        x_base = X_ptr + g_idx * (N_BLOCKS_K * 32)
        row_idx = g_idx * N_OUT + offs_n
        w_base = W_u8_ptr + row_idx[:, None] * (N_BLOCKS_K * 16)
        s_base = S_bf16_ptr + row_idx[:, None] * N_BLOCKS_K

        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for sb_start in range(0, N_BLOCKS_K, sub_blocks_k):
            b_idx = sb_start * 16 + offs_b
            sb_idx = sb_start + offs_sb
            x_even = tl.load(x_base + b_idx * 2).to(tl.float32)
            x_odd = tl.load(x_base + b_idx * 2 + 1).to(tl.float32)
            w_b = tl.load(w_base + b_idx[None, :])
            s_f = tl.load(s_base + sb_idx[None, :]).to(tl.float32)
            v = (_decode_e2m1_x4(w_b & 0x0F) * x_even[None, :] + _decode_e2m1_x4(w_b >> 4) * x_odd[None, :]) * s_f
            acc += tl.sum(v, axis=1)

        tl.store(Z_out_ptr + row_idx, acc.to(tl.bfloat16))

    @triton.jit
    def _triton_gemv_fp4_m1_kernel(
        X_ptr,
        W_u8_ptr,
        S_bf16_ptr,
        Y_out_ptr,
        N_OUT: tl.constexpr,
        N_BLOCKS_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
        sub_blocks_k: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N_OUT

        BYTES_PER_ITER: tl.constexpr = sub_blocks_k * 16
        offs_b = tl.arange(0, BYTES_PER_ITER)
        offs_sb = offs_b // 16

        w_base = W_u8_ptr + offs_n[:, None] * (N_BLOCKS_K * 16)
        s_base = S_bf16_ptr + offs_n[:, None] * N_BLOCKS_K

        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for sb_start in range(0, N_BLOCKS_K, sub_blocks_k):
            b_idx = sb_start * 16 + offs_b
            sb_idx = sb_start + offs_sb
            x_even = tl.load(X_ptr + b_idx * 2).to(tl.float32)
            x_odd = tl.load(X_ptr + b_idx * 2 + 1).to(tl.float32)
            w_b = tl.load(w_base + b_idx[None, :], mask=mask_n[:, None], other=0)
            s_f = tl.load(s_base + sb_idx[None, :], mask=mask_n[:, None], other=0.0).to(tl.float32)
            v = (_decode_e2m1_x4(w_b & 0x0F) * x_even[None, :] + _decode_e2m1_x4(w_b >> 4) * x_odd[None, :]) * s_f
            acc += tl.sum(v, axis=1)

        tl.store(Y_out_ptr + offs_n, acc.to(tl.bfloat16), mask=mask_n)

    @triton.jit
    def _triton_quantize_int4_32_kernel(
        W_ptr,
        Out_u8_ptr,
        Out_s_bf16_ptr,
        N_BLOCKS_C: tl.constexpr,
    ):
        row = tl.program_id(0)
        blk = tl.program_id(1)
        base = row * (N_BLOCKS_C * 32) + blk * 32
        offs16 = tl.arange(0, 16)
        w_even = tl.load(W_ptr + base + offs16 * 2).to(tl.float32)
        w_odd = tl.load(W_ptr + base + offs16 * 2 + 1).to(tl.float32)

        amax = tl.maximum(tl.max(tl.abs(w_even), axis=0), tl.max(tl.abs(w_odd), axis=0))
        best_s = tl.maximum(amax / 7.5, 1e-12)
        best_err = 1e30

        for m_step in tl.static_range(9):
            mul = 0.80 + m_step * 0.025
            s_cand = tl.maximum((amax * mul) / 7.5, 1e-12).to(tl.bfloat16).to(tl.float32)
            inv_s = 1.0 / s_cand
            c_e = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.rint(w_even * inv_s + 7.5), 0.0), 15.0)
            c_o = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.rint(w_odd * inv_s + 7.5), 0.0), 15.0)
            d_e = (c_e - 7.5) * s_cand - w_even
            d_o = (c_o - 7.5) * s_cand - w_odd
            err = tl.sum(d_e * d_e + d_o * d_o, axis=0)
            if err < best_err:
                best_err = err
                best_s = s_cand

        inv_best = 1.0 / best_s
        q_e = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.rint(w_even * inv_best + 7.5), 0.0), 15.0).to(tl.uint8)
        q_o = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.rint(w_odd * inv_best + 7.5), 0.0), 15.0).to(tl.uint8)
        tl.store(Out_s_bf16_ptr + row * N_BLOCKS_C + blk, best_s.to(tl.bfloat16))
        tl.store(Out_u8_ptr + row * (N_BLOCKS_C * 16) + blk * 16 + offs16, q_e | (q_o << 4))

    @triton.jit
    def _triton_bmm_int4_m1_kernel(
        X_ptr,
        W_u8_ptr,
        S_bf16_ptr,
        Z_out_ptr,
        N_OUT: tl.constexpr,
        N_BLOCKS_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
        sub_blocks_k: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        g_idx = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        BYTES_PER_ITER: tl.constexpr = sub_blocks_k * 16
        offs_b = tl.arange(0, BYTES_PER_ITER)
        offs_sb = offs_b // 16

        x_base = X_ptr + g_idx * (N_BLOCKS_K * 32)
        row_idx = g_idx * N_OUT + offs_n
        w_base = W_u8_ptr + row_idx[:, None] * (N_BLOCKS_K * 16)
        s_base = S_bf16_ptr + row_idx[:, None] * N_BLOCKS_K

        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for sb_start in range(0, N_BLOCKS_K, sub_blocks_k):
            b_idx = sb_start * 16 + offs_b
            sb_idx = sb_start + offs_sb
            x_even = tl.load(x_base + b_idx * 2).to(tl.float32)
            x_odd = tl.load(x_base + b_idx * 2 + 1).to(tl.float32)
            x_sum = (x_even + x_odd) * 7.5
            w_b = tl.load(w_base + b_idx[None, :])
            s_f = tl.load(s_base + sb_idx[None, :]).to(tl.float32)
            v = ((w_b & 0x0F).to(tl.float32) * x_even[None, :] + (w_b >> 4).to(tl.float32) * x_odd[None, :] - x_sum[None, :]) * s_f
            acc += tl.sum(v, axis=1)

        tl.store(Z_out_ptr + row_idx, acc.to(tl.bfloat16))

    @triton.jit
    def _triton_gemv_int4_m1_kernel(
        X_ptr,
        W_u8_ptr,
        S_bf16_ptr,
        Y_out_ptr,
        N_OUT: tl.constexpr,
        N_BLOCKS_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
        sub_blocks_k: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N_OUT

        BYTES_PER_ITER: tl.constexpr = sub_blocks_k * 16
        offs_b = tl.arange(0, BYTES_PER_ITER)
        offs_sb = offs_b // 16

        w_base = W_u8_ptr + offs_n[:, None] * (N_BLOCKS_K * 16)
        s_base = S_bf16_ptr + offs_n[:, None] * N_BLOCKS_K

        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for sb_start in range(0, N_BLOCKS_K, sub_blocks_k):
            b_idx = sb_start * 16 + offs_b
            sb_idx = sb_start + offs_sb
            x_even = tl.load(X_ptr + b_idx * 2).to(tl.float32)
            x_odd = tl.load(X_ptr + b_idx * 2 + 1).to(tl.float32)
            x_sum = (x_even + x_odd) * 7.5
            w_b = tl.load(w_base + b_idx[None, :], mask=mask_n[:, None], other=0)
            s_f = tl.load(s_base + sb_idx[None, :], mask=mask_n[:, None], other=0.0).to(tl.float32)
            v = ((w_b & 0x0F).to(tl.float32) * x_even[None, :] + (w_b >> 4).to(tl.float32) * x_odd[None, :] - x_sum[None, :]) * s_f
            acc += tl.sum(v, axis=1)

        tl.store(Y_out_ptr + offs_n, acc.to(tl.bfloat16), mask=mask_n)

    @triton.jit
    def _triton_gemv_mxfp8_m1_kernel(
        X_ptr,
        W_fp8_ptr,
        S_u8_ptr,
        Y_out_ptr,
        N_OUT: tl.constexpr,
        N_BLOCKS_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
        sub_blocks_k: tl.constexpr,
        SCALE_ROW_DIV: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N_OUT

        ELEMS_PER_ITER: tl.constexpr = sub_blocks_k * 32
        offs_k = tl.arange(0, ELEMS_PER_ITER)
        offs_sb = offs_k // 32

        w_base = W_fp8_ptr + offs_n[:, None] * (N_BLOCKS_K * 32)
        s_row = offs_n // SCALE_ROW_DIV
        s_base = S_u8_ptr + s_row[:, None] * N_BLOCKS_K

        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for sb_start in range(0, N_BLOCKS_K, sub_blocks_k):
            k_idx = sb_start * 32 + offs_k
            sb_idx = sb_start + offs_sb
            x_val = tl.load(X_ptr + k_idx).to(tl.float32)
            w_val = tl.load(w_base + k_idx[None, :], mask=mask_n[:, None], other=0.0).to(tl.float32)
            s_u8 = tl.load(s_base + sb_idx[None, :], mask=mask_n[:, None], other=127).to(tl.int32)
            s_f = (s_u8 << 23).to(tl.float32, bitcast=True)
            acc += tl.sum((w_val * x_val[None, :]) * s_f, axis=1)

        tl.store(Y_out_ptr + offs_n, acc.to(tl.bfloat16), mask=mask_n)

    @triton.jit
    def _triton_bmm_mxfp8_m1_kernel(
        X_ptr,
        W_fp8_ptr,
        S_u8_ptr,
        Z_out_ptr,
        N_OUT: tl.constexpr,
        N_BLOCKS_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
        sub_blocks_k: tl.constexpr,
        SCALE_ROW_DIV: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        g_idx = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        ELEMS_PER_ITER: tl.constexpr = sub_blocks_k * 32
        offs_k = tl.arange(0, ELEMS_PER_ITER)
        offs_sb = offs_k // 32

        x_base = X_ptr + g_idx * (N_BLOCKS_K * 32)
        row_idx = g_idx * N_OUT + offs_n
        w_base = W_fp8_ptr + row_idx[:, None] * (N_BLOCKS_K * 32)
        s_row = row_idx // SCALE_ROW_DIV
        s_base = S_u8_ptr + s_row[:, None] * N_BLOCKS_K

        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for sb_start in range(0, N_BLOCKS_K, sub_blocks_k):
            k_idx = sb_start * 32 + offs_k
            sb_idx = sb_start + offs_sb
            x_val = tl.load(x_base + k_idx).to(tl.float32)
            w_val = tl.load(w_base + k_idx[None, :]).to(tl.float32)
            s_u8 = tl.load(s_base + sb_idx[None, :]).to(tl.int32)
            s_f = (s_u8 << 23).to(tl.float32, bitcast=True)
            acc += tl.sum((w_val * x_val[None, :]) * s_f, axis=1)

        tl.store(Z_out_ptr + row_idx, acc.to(tl.bfloat16))

except Exception:
    _HAS_TRITON = False
    _triton_bmm_mxfp8_m1_kernel = None

from chronos_1t.models.chronos_moe_block import (
    NVFP4_LUT,
    _EXPERT_KEY_CACHE,
    _get_nvfp4_byte_lut,
    build_hadamard_128,
    generate_expert_1d_keys,
    unbind_2d_permuted_hadamard,
    unpack_1bit_rademacher,
    unpack_2bit_lloyd_max,
    unpack_nvfp4,
)

_PINNED_DMA_BUFS: dict[tuple[Any, ...], torch.Tensor] = {}


def _copy_via_pinned_dma(cpu_tensor: torch.Tensor, device: torch.device, key_id: int) -> torch.Tensor:
    shape_key = (key_id, *cpu_tensor.shape, cpu_tensor.dtype)
    buf = _PINNED_DMA_BUFS.get(shape_key)
    if buf is None:
        try:
            buf = torch.empty(cpu_tensor.shape, dtype=cpu_tensor.dtype, device="cpu", pin_memory=True)
            _PINNED_DMA_BUFS[shape_key] = buf
        except Exception:
            return cpu_tensor.to(device, non_blocking=False)
    buf.copy_(cpu_tensor, non_blocking=False)
    return buf.to(device, non_blocking=False)


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
    lut256 = _get_nvfp4_byte_lut(device, torch.float32)
    u8 = w_i8.to(device=device).view(torch.uint8)
    vals = lut256[u8.to(torch.int32)].view(rows, cols // 32, 32)
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

        self.s_gu_scale2 = tensors["s_gu.weight_scale_2"].to(device)
        self.beta_gu = tensors["s_gu.beta"].to(device=device, dtype=torch.bfloat16)
        self.s_down_scale2 = tensors["s_down.weight_scale_2"].to(device)
        self.beta_down = tensors["s_down.beta"].to(device=device, dtype=torch.bfloat16)

        self.gu_tail_bits = int(tensors.get("tail.gu_bits", torch.tensor(1)).item())
        self.down_tail_bits = int(tensors.get("tail.down_bits", torch.tensor(1)).item())

        if self.mmap_zero_ram:
            self.s_gu_packed = None
            self.s_gu_scale = None
            self.s_down_packed = None
            self.s_down_scale = None
            self.u_gu = None
            self.v_gu = None
            self.u_down = None
            self.v_down = None
            self.gu_tail_packed = None
            self.gu_tail_scale = None
            self.down_tail_packed = None
            self.down_tail_scale = None
        else:
            self.s_gu_packed = tensors["s_gu.weight"].to(device)
            self.s_gu_scale = tensors["s_gu.weight_scale"].to(device)
            self.s_down_packed = tensors["s_down.weight"].to(device)
            self.s_down_scale = tensors["s_down.weight_scale"].to(device)
            self.u_gu = tensors["healer.u_gu"].to(device)
            self.v_gu = tensors["healer.v_gu"].to(device)
            self.u_down = tensors["healer.u_down"].to(device)
            self.v_down = tensors["healer.v_down"].to(device)
            tail_dev = torch.device("cpu") if offload_tail_to_cpu else device
            self.gu_tail_packed = tensors["tail.gu_packed"].to(tail_dev)
            self.gu_tail_scale = tensors["tail.gu_scale"].to(tail_dev)
            self.down_tail_packed = tensors["tail.down_packed"].to(tail_dev)
            self.down_tail_scale = tensors["tail.down_scale"].to(tail_dev)

    @torch.no_grad()
    def _fetch_healer_slices(
        self, local_m: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.mmap_zero_ram and self.mmap_handle is not None:
            p = self.mmap_prefix
            u_gu_m = _copy_via_pinned_dma(self.mmap_handle.get_slice(f"{p}healer.u_gu")[local_m], self.device, 15)
            v_gu_m = _copy_via_pinned_dma(self.mmap_handle.get_slice(f"{p}healer.v_gu")[local_m], self.device, 16)
            u_dn_m = _copy_via_pinned_dma(self.mmap_handle.get_slice(f"{p}healer.u_down")[local_m], self.device, 17)
            v_dn_m = _copy_via_pinned_dma(self.mmap_handle.get_slice(f"{p}healer.v_down")[local_m], self.device, 18)
            return u_gu_m, v_gu_m, u_dn_m, v_dn_m
        assert self.u_gu is not None and self.v_gu is not None
        assert self.u_down is not None and self.v_down is not None
        return self.u_gu[local_m], self.v_gu[local_m], self.u_down[local_m], self.v_down[local_m]

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
        if self.mmap_zero_ram and self.mmap_handle is not None:
            p = self.mmap_prefix
            s_gu_p = _copy_via_pinned_dma(self.mmap_handle.get_tensor(f"{p}s_gu.weight"), self.device, 1)
            s_gu_s = _copy_via_pinned_dma(self.mmap_handle.get_tensor(f"{p}s_gu.weight_scale"), self.device, 2)
            s_dn_p = _copy_via_pinned_dma(self.mmap_handle.get_tensor(f"{p}s_down.weight"), self.device, 3)
            s_dn_s = _copy_via_pinned_dma(self.mmap_handle.get_tensor(f"{p}s_down.weight_scale"), self.device, 4)
            s_gu = unpack_nvfp4(s_gu_p, s_gu_s, self.s_gu_scale2, dtype=torch.bfloat16)
            s_dn = unpack_nvfp4(s_dn_p, s_dn_s, self.s_down_scale2, dtype=torch.bfloat16)
            return s_gu, s_dn
        assert self.s_gu_packed is not None and self.s_gu_scale is not None
        assert self.s_down_packed is not None and self.s_down_scale is not None
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

        u_gu_r, v_gu_r, u_dn_r, v_dn_r = self._fetch_healer_slices(local_m)

        r_s1, _, r_inv, r_s2, c_s1, c_perm, _, c_s2 = generate_expert_1d_keys(
            4608, 5120, local_m, seed_base=101 + self.group_idx * 8, device=self.device
        )
        x_c = (x * c_s1.to(dt).unsqueeze(0))[:, c_perm]
        x_c = torch.matmul(x_c.view(k_toks, 40, 128), h).view(k_toks, 5120) * c_s2.to(dt).unsqueeze(0)
        z_hrr = F.linear(x_c, s_gu) * r_s2.to(dt).unsqueeze(0)
        z_hrr = torch.matmul(z_hrr.view(k_toks, 36, 128), h).view(k_toks, 4608)[:, r_inv] * r_s1.to(dt).unsqueeze(0)

        u_gu_m = u_gu_r.to(dt)
        v_gu_m = v_gu_r.to(dt)
        z_gu = self.beta_gu[local_m] * z_hrr + F.linear(F.linear(x, v_gu_m), u_gu_m)

        act = dsv41_clamped_swiglu(z_gu, inter_dim=2304, swiglu_limit=self.swiglu_limit)

        dr_s1, _, dr_inv, dr_s2, dc_s1, dc_perm, _, dc_s2 = generate_expert_1d_keys(
            5120, 2304, local_m, seed_base=303 + self.group_idx * 8, device=self.device
        )
        a_c = (act * dc_s1.to(dt).unsqueeze(0))[:, dc_perm]
        a_c = torch.matmul(a_c.view(k_toks, 18, 128), h).view(k_toks, 2304) * dc_s2.to(dt).unsqueeze(0)
        y_hrr = F.linear(a_c, s_dn) * dr_s2.to(dt).unsqueeze(0)
        y_hrr = torch.matmul(y_hrr.view(k_toks, 40, 128), h).view(k_toks, 5120)[:, dr_inv] * dr_s1.to(dt).unsqueeze(0)

        u_dn_m = u_dn_r.to(dt)
        v_dn_m = v_dn_r.to(dt)
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

        u_gu_r, v_gu_r, u_dn_r, v_dn_r = self._fetch_healer_slices(local_m)

        r_s1, _, r_inv, r_s2, c_s1, _, c_inv, c_s2 = generate_expert_1d_keys(
            4608, 5120, local_m, seed_base=101 + self.group_idx * 8, device=self.device
        )
        h_gu = unbind_2d_permuted_hadamard(s_gu, r_s1, r_inv, r_s2, c_s1, c_inv, c_s2, self.h128)
        healer_gu = torch.matmul(u_gu_r.to(torch.bfloat16), v_gu_r.to(torch.bfloat16))

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
        healer_dn = torch.matmul(u_dn_r.to(torch.bfloat16), v_dn_r.to(torch.bfloat16))

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
        include_tail: bool = True,
        healer_slices: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        tail_slices: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        """Zero-temp-allocation in-place slot reconstruction (`v7.5`) with post-quantization refusal projection."""
        if s_gu_cached is None or s_dn_cached is None:
            s_gu, s_dn = self.get_group_banks()
        else:
            s_gu, s_dn = s_gu_cached, s_dn_cached

        u_gu_r, v_gu_r, u_dn_r, v_dn_r = (
            healer_slices if healer_slices is not None else self._fetch_healer_slices(local_m)
        )
        r_s1, _, r_inv, r_s2, c_s1, _, c_inv, c_s2 = generate_expert_1d_keys(
            4608, 5120, local_m, seed_base=101 + self.group_idx * 8, device=self.device
        )
        dr_s1, _, dr_inv, dr_s2, dc_s1, _, dc_inv, dc_s2 = generate_expert_1d_keys(
            5120, 2304, local_m, seed_base=303 + self.group_idx * 8, device=self.device
        )

        if _HAS_TRITON and self.device.type == "cuda":
            h = self.h128.to(torch.bfloat16)
            if include_tail:
                gu_p_m, gu_s_m, dn_p_m, dn_s_m = (
                    tail_slices if tail_slices is not None else self._fetch_tail_slices(local_m)
                )
                gu_s_u8 = gu_s_m.view(torch.uint8)
                dn_s_u8 = dn_s_m.view(torch.uint8)
            else:
                gu_p_m = gu_s_u8 = slot_gu.view(torch.uint8)
                dn_p_m = dn_s_u8 = slot_dn.view(torch.uint8)

            x1_gu = torch.matmul(h, (s_gu * r_s2.to(torch.bfloat16).unsqueeze(1)).view(36, 128, 5120)).view(4608, 5120)
            _triton_row_perm_scale_kernel[(4608, 5120 // 256)](
                x1_gu, slot_gu, r_inv, r_s1, c_s2, COLS=5120, BLOCK_C=256, num_warps=4
            )
            x2_gu = torch.matmul(slot_gu.view(4608, 40, 128), h).view(4608, 5120)
            _triton_col_perm_add_tail_kernel[(4608, 5120 // 256)](
                x2_gu,
                gu_p_m,
                gu_s_u8,
                slot_gu,
                c_inv,
                c_s1,
                self.beta_gu[local_m : local_m + 1],
                COLS=5120,
                TAIL_BITS=self.gu_tail_bits,
                HAS_TAIL=include_tail,
                BLOCK_C=256,
                num_warps=4,
            )
            del x1_gu, x2_gu
            slot_gu.addmm_(u_gu_r.to(torch.bfloat16), v_gu_r.to(torch.bfloat16))

            x1_dn = torch.matmul(h, (s_dn * dr_s2.to(torch.bfloat16).unsqueeze(1)).view(40, 128, 2304)).view(5120, 2304)
            _triton_row_perm_scale_kernel[(5120, 2304 // 256)](
                x1_dn, slot_dn, dr_inv, dr_s1, dc_s2, COLS=2304, BLOCK_C=256, num_warps=4
            )
            x2_dn = torch.matmul(slot_dn.view(5120, 18, 128), h).view(5120, 2304)
            _triton_col_perm_add_tail_kernel[(5120, 2304 // 256)](
                x2_dn,
                dn_p_m,
                dn_s_u8,
                slot_dn,
                dc_inv,
                dc_s1,
                self.beta_down[local_m : local_m + 1],
                COLS=2304,
                TAIL_BITS=self.down_tail_bits,
                HAS_TAIL=include_tail,
                BLOCK_C=256,
                num_warps=4,
            )
            del x1_dn, x2_dn
            slot_dn.addmm_(u_dn_r.to(torch.bfloat16), v_dn_r.to(torch.bfloat16))
        else:
            h_gu = unbind_2d_permuted_hadamard(s_gu, r_s1, r_inv, r_s2, c_s1, c_inv, c_s2, self.h128)
            if include_tail:
                gu_p_m, gu_s_m, dn_p_m, dn_s_m = (
                    tail_slices if tail_slices is not None else self._fetch_tail_slices(local_m)
                )
                if self.gu_tail_bits == 2:
                    slot_gu.copy_(unpack_2bit_lloyd_max(gu_p_m, gu_s_m, dtype=torch.bfloat16))
                else:
                    slot_gu.copy_(unpack_1bit_rademacher(gu_p_m, gu_s_m, dtype=torch.bfloat16))
                slot_gu.add_(h_gu * self.beta_gu[local_m])
            else:
                slot_gu.copy_(h_gu * self.beta_gu[local_m])
            slot_gu.addmm_(u_gu_r.to(torch.bfloat16), v_gu_r.to(torch.bfloat16))

            h_dn = unbind_2d_permuted_hadamard(s_dn, dr_s1, dr_inv, dr_s2, dc_s1, dc_inv, dc_s2, self.h128)
            if include_tail:
                if self.down_tail_bits == 2:
                    slot_dn.copy_(unpack_2bit_lloyd_max(dn_p_m, dn_s_m, dtype=torch.bfloat16))
                else:
                    slot_dn.copy_(unpack_1bit_rademacher(dn_p_m, dn_s_m, dtype=torch.bfloat16))
                slot_dn.add_(h_dn * self.beta_down[local_m])
            else:
                slot_dn.copy_(h_dn * self.beta_down[local_m])
            slot_dn.addmm_(u_dn_r.to(torch.bfloat16), v_dn_r.to(torch.bfloat16))

        if self.r_basis is not None:
            rb = self.r_basis.to(torch.bfloat16)
            proj = torch.matmul(rb, slot_dn)
            slot_dn.addmm_(rb.T, proj, beta=1.0, alpha=-1.0)


class _ChronosDummyExperts(nn.Module):
    """Lightweight placeholder satisfying vLLM `DeepseekV4MoE.experts` attribute checks."""

    def __init__(self) -> None:
        super().__init__()
        self.enable_eplb = False
        self.is_Dummy = True


_SHARED_SCRATCH_GU_BF16: dict[torch.device, torch.Tensor] = {}
_SHARED_SCRATCH_D_BF16: dict[torch.device, torch.Tensor] = {}
_SHARED_384_KEYS: dict[torch.device, dict[str, torch.Tensor]] = {}


def _get_shared_384_keys(device: torch.device) -> dict[str, torch.Tensor]:
    cached = _SHARED_384_KEYS.get(device)
    if cached is not None:
        return cached
    gu_c_perm = torch.empty((384, 5120), dtype=torch.int16, device=device)
    gu_c_s1_perm = torch.empty((384, 5120), dtype=torch.bfloat16, device=device)
    gu_c_s2 = torch.empty((384, 5120), dtype=torch.bfloat16, device=device)
    gu_r_s2 = torch.empty((384, 4608), dtype=torch.bfloat16, device=device)
    gu_r_inv = torch.empty((384, 4608), dtype=torch.int16, device=device)
    gu_r_s1 = torch.empty((384, 4608), dtype=torch.bfloat16, device=device)

    dn_c_perm = torch.empty((384, 2304), dtype=torch.int16, device=device)
    dn_c_s1_perm = torch.empty((384, 2304), dtype=torch.bfloat16, device=device)
    dn_c_s2 = torch.empty((384, 2304), dtype=torch.bfloat16, device=device)
    dn_r_s2 = torch.empty((384, 5120), dtype=torch.bfloat16, device=device)
    dn_r_inv = torch.empty((384, 5120), dtype=torch.int16, device=device)
    dn_r_s1 = torch.empty((384, 5120), dtype=torch.bfloat16, device=device)

    for eid in range(384):
        g_idx = eid // 8
        local_m = eid % 8
        r_s1, _, r_inv, r_s2, c_s1, c_perm, _, c_s2 = generate_expert_1d_keys(
            4608, 5120, local_m, seed_base=101 + g_idx * 8, device=device
        )
        gu_c_perm[eid].copy_(c_perm.to(torch.int16))
        gu_c_s1_perm[eid].copy_(c_s1[c_perm].to(torch.bfloat16))
        gu_c_s2[eid].copy_(c_s2.to(torch.bfloat16))
        gu_r_s2[eid].copy_(r_s2.to(torch.bfloat16))
        gu_r_inv[eid].copy_(r_inv.to(torch.int16))
        gu_r_s1[eid].copy_(r_s1.to(torch.bfloat16))

        dr_s1, _, dr_inv, dr_s2, dc_s1, dc_perm, _, dc_s2 = generate_expert_1d_keys(
            5120, 2304, local_m, seed_base=303 + g_idx * 8, device=device
        )
        dn_c_perm[eid].copy_(dc_perm.to(torch.int16))
        dn_c_s1_perm[eid].copy_(dc_s1[dc_perm].to(torch.bfloat16))
        dn_c_s2[eid].copy_(dc_s2.to(torch.bfloat16))
        dn_r_s2[eid].copy_(dr_s2.to(torch.bfloat16))
        dn_r_inv[eid].copy_(dr_inv.to(torch.int16))
        dn_r_s1[eid].copy_(dr_s1.to(torch.bfloat16))

    out = {
        "gu_c_perm": gu_c_perm,
        "gu_c_s1_perm": gu_c_s1_perm,
        "gu_c_s2": gu_c_s2,
        "gu_r_s2": gu_r_s2,
        "gu_r_inv": gu_r_inv,
        "gu_r_s1": gu_r_s1,
        "dn_c_perm": dn_c_perm,
        "dn_c_s1_perm": dn_c_s1_perm,
        "dn_c_s2": dn_c_s2,
        "dn_r_s2": dn_r_s2,
        "dn_r_inv": dn_r_inv,
        "dn_r_s1": dn_r_s1,
    }
    _SHARED_384_KEYS[device] = out
    return out


def _get_shared_bf16_scratch(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    gu = _SHARED_SCRATCH_GU_BF16.get(device)
    d = _SHARED_SCRATCH_D_BF16.get(device)
    if gu is None or d is None:
        gu = torch.empty((4608, 5120), dtype=torch.bfloat16, device=device)
        d = torch.empty((5120, 2304), dtype=torch.bfloat16, device=device)
        _SHARED_SCRATCH_GU_BF16[device] = gu
        _SHARED_SCRATCH_D_BF16[device] = d
    return gu, d


def _quantize_row_fp8(w_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    s = (w_bf16.abs().amax(dim=-1) / 448.0).float().clamp_min(1e-12).contiguous()
    w_fp8 = (w_bf16.float() / s.unsqueeze(-1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).contiguous()
    return w_fp8, s


def _dequantize_fp4_bf16scale_into(w_u8: torch.Tensor, s_bf16: torch.Tensor, out_bf16: torch.Tensor) -> torch.Tensor:
    rows, packed_cols = w_u8.shape
    cols = packed_cols * 2
    lut256 = _get_nvfp4_byte_lut(w_u8.device, torch.bfloat16)
    vals = lut256[w_u8.to(torch.int32)].view(rows, cols // 32, 32)
    scales = (s_bf16 * 4.0).unsqueeze(-1)
    out_bf16.copy_((vals * scales).view(rows, cols))
    return out_bf16


class ChronosDsv41SparseMoeBlock(nn.Module):
    """Drop-in replacement for DeepSeek-V4.1-Flash-1017 (`DeepseekV41ForCausalLM`) Sparse MoE layer."""

    def __init__(
        self,
        layer_idx: int = 0,
        shard_path: str | Path | None = None,
        hot_shard_path: str | Path | None = None,
        device: torch.device | str | None = None,
        top_k: int = 6,
        routed_scaling_factor: float = 1.5,
        swiglu_limit: float = 10.0,
        hot_cache_size: int = 16,
        offload_tail_to_cpu: bool = True,
        mmap_zero_ram: bool = False,
        use_fp4_slot_cache: bool | None = None,
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
        if use_fp4_slot_cache is None:
            use_fp4_slot_cache = bool(mmap_zero_ram and _HAS_TRITON and device.type == "cuda")
        self.use_fp4_slot_cache = bool(use_fp4_slot_cache and _HAS_TRITON and device.type == "cuda")
        self.use_fused_moe = use_fused_moe and (vllm_fused_experts is not None)
        self.shard_path_str = str(shard_path)

        # vLLM DeepseekV4MoE compatibility attributes
        self.use_mega_moe = False
        self.n_logical_experts = 384
        self.n_physical_experts = 384
        self.n_local_physical_experts = 384
        self.n_routed_experts = 384
        self.n_shared_experts = 1
        self.n_redundant_experts = 0
        self.experts = _ChronosDummyExperts()

        tp = f"layers.{layer_idx}.ffn"
        if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_SEQUENTIAL"):
            try:
                _fd_r = os.open(os.path.realpath(self.shard_path_str), os.O_RDONLY)
                os.posix_fadvise(_fd_r, 0, 0, os.POSIX_FADV_SEQUENTIAL)
                os.close(_fd_r)
            except OSError:
                pass
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
        sh_gu_tmp = torch.cat([sh_w1, sh_w3], dim=0).contiguous()  # (4608, 5120)
        sh_down_tmp = dequantize_block_fp8_32(
            self._mmap_handle.get_tensor(f"{tp}.shared_experts.w2.weight"),
            self._mmap_handle.get_tensor(f"{tp}.shared_experts.w2.scale"),
            device,
        ).contiguous()  # (5120, 2304)

        self.r_basis: torch.Tensor | None = None
        self.r_basis_f32: torch.Tensor | None = None
        if os.environ.get("CHRONOS_RUNTIME_R_BASIS", "0") == "1" and f"{tp}.abliteration.r_basis" in all_keys:
            self.r_basis = self._mmap_handle.get_tensor(f"{tp}.abliteration.r_basis").to(device=device, dtype=torch.bfloat16).contiguous()
            self.r_basis_f32 = self.r_basis.float().contiguous()

        if self.r_basis is not None:
            rb = self.r_basis.to(torch.bfloat16)
            sh_down_tmp.addmm_(rb.T, torch.matmul(rb, sh_down_tmp), beta=1.0, alpha=-1.0)

        shared_h128 = build_hadamard_128(device, dtype=torch.bfloat16)
        self.groups = nn.ModuleList()

        # Upgrade 2: Hybrid Direct Unshared MXFP4 Hot Experts + Chronos v7.5 Holographic Cold Experts
        self.hot_expert_to_idx: list[int] = [-1] * 384
        self.hot_expert_ids_list: list[int] = []
        self.num_hot_mxfp4_experts = 0
        self._hot_tensors: dict[str, torch.Tensor] | None = None
        self._hot_handle: Any = None
        self._hot_tp = f"{tp}.hot_experts."
        self._hot_shard_cand = (
            Path(hot_shard_path)
            if hot_shard_path is not None
            else Path(self.shard_path_str).with_name(
                f"chronos-dsv41-upgrade2-hot96-layer-{layer_idx:05d}-of-00040.safetensors"
            )
        )
        self._try_attach_hot_shard()

        default_dec_k = str(self.top_k)
        self.decode_top_k = max(1, min(self.top_k, int(os.environ.get("CHRONOS_DECODE_TOP_K", default_dec_k))))
        self.n4_slots = int(os.environ.get("CHRONOS_TIER1_4BIT_SLOTS", "48"))
        self.n2_slots = int(os.environ.get("CHRONOS_TIER2_2BIT_SLOTS", "56"))
        default_n1 = min(88, max(0, 384 - self.n4_slots - self.n2_slots))
        self.n1_slots = int(os.environ.get("CHRONOS_TIER3_1BIT_SLOTS", str(default_n1)))

        if self.use_fp4_slot_cache:
            self._keys_384 = None
            self.group_gu_u8 = None
            self.group_gu_s_fp8 = None
            self.group_d_u8 = None
            self.group_d_s_fp8 = None
            self.healer_u_gu = None
            self.healer_v_gu = None
            self.healer_u_d = None
            self.healer_v_d = None
            self.cold_scale_gu_f32 = None
            self.cold_scale_d_f32 = None

            self.shared_slot = self.n4_slots
            self.sh_gu_fp8 = None
            self.sh_gu_s = None
            self.sh_d_fp8 = None
            self.sh_d_s = None
            self.sh_gu = None
            self.sh_down = None
            self.cache_gu_buf = None
            self.cache_d_buf = None
            self.slot_gu = []
            self.slot_d = []

            self.w4_gu_u8 = torch.empty((self.n4_slots + 1, 4608, 2560), dtype=torch.uint8, device=device)
            self.s4_gu_bf16 = torch.empty((self.n4_slots + 1, 4608, 160), dtype=torch.bfloat16, device=device)
            self.w4_d_u8 = torch.empty((self.n4_slots + 1, 5120, 1152), dtype=torch.uint8, device=device)
            self.s4_d_bf16 = torch.empty((self.n4_slots + 1, 5120, 72), dtype=torch.bfloat16, device=device)
            self.cache_gu_u8 = self.w4_gu_u8
            self.cache_gu_s_bf16 = self.s4_gu_bf16
            self.cache_d_u8 = self.w4_d_u8
            self.cache_d_s_bf16 = self.s4_d_bf16

            self.w2_gu_u8 = torch.empty((self.n2_slots, 4608, 1280), dtype=torch.uint8, device=device)
            self.s2_gu_fp8 = torch.empty((self.n2_slots, 4608, 80), dtype=torch.uint8, device=device)
            self.w2_d_u8 = torch.empty((self.n2_slots, 5120, 576), dtype=torch.uint8, device=device)
            self.s2_d_fp8 = torch.empty((self.n2_slots, 5120, 36), dtype=torch.uint8, device=device)

            t3_cpu_from = int(os.environ.get("CHRONOS_T3_CPU_FROM_LAYER", "40"))
            self.t3_on_cpu = bool(self.n1_slots > 0 and layer_idx >= t3_cpu_from)
            if self.t3_on_cpu:
                try:
                    self.w1_gu_cpu = torch.empty((self.n1_slots, 4608, 640), dtype=torch.uint8, device="cpu", pin_memory=True)
                    self.s1_gu_cpu = torch.empty((self.n1_slots, 4608, 80), dtype=torch.uint8, device="cpu", pin_memory=True)
                    self.w1_d_cpu = torch.empty((self.n1_slots, 5120, 288), dtype=torch.uint8, device="cpu", pin_memory=True)
                    self.s1_d_cpu = torch.empty((self.n1_slots, 5120, 36), dtype=torch.uint8, device="cpu", pin_memory=True)
                except Exception:
                    self.w1_gu_cpu = torch.empty((self.n1_slots, 4608, 640), dtype=torch.uint8, device="cpu")
                    self.s1_gu_cpu = torch.empty((self.n1_slots, 4608, 80), dtype=torch.uint8, device="cpu")
                    self.w1_d_cpu = torch.empty((self.n1_slots, 5120, 288), dtype=torch.uint8, device="cpu")
                    self.s1_d_cpu = torch.empty((self.n1_slots, 5120, 36), dtype=torch.uint8, device="cpu")
                n1_gpu_alloc = max(1, self.top_k)
            else:
                self.w1_gu_cpu = None
                self.s1_gu_cpu = None
                self.w1_d_cpu = None
                self.s1_d_cpu = None
                n1_gpu_alloc = self.n1_slots

            self.w1_gu_u8 = torch.empty((n1_gpu_alloc, 4608, 640), dtype=torch.uint8, device=device)
            self.s1_gu_fp8 = torch.empty((n1_gpu_alloc, 4608, 80), dtype=torch.uint8, device=device)
            self.w1_d_u8 = torch.empty((n1_gpu_alloc, 5120, 288), dtype=torch.uint8, device=device)
            self.s1_d_fp8 = torch.empty((n1_gpu_alloc, 5120, 36), dtype=torch.uint8, device=device)
            self.tier_slot_map_gpu = torch.full((384,), -1, dtype=torch.int32, device=device)

            _triton_quantize_mse_fp4_32_kernel[(4608, 160)](
                sh_gu_tmp,
                self.w4_gu_u8[self.shared_slot],
                self.s4_gu_bf16[self.shared_slot],
                N_BLOCKS_C=160,
                num_warps=1,
            )
            _triton_quantize_mse_fp4_32_kernel[(5120, 72)](
                sh_down_tmp,
                self.w4_d_u8[self.shared_slot],
                self.s4_d_bf16[self.shared_slot],
                N_BLOCKS_C=72,
                num_warps=1,
            )
            del sh_gu_tmp, sh_down_tmp, sh_w1, sh_w3
        else:
            self._keys_384 = None
            self.group_gu_u8 = None
            self.group_gu_s_fp8 = None
            self.group_d_u8 = None
            self.group_d_s_fp8 = None
            self.healer_u_gu = None
            self.healer_v_gu = None
            self.healer_u_d = None
            self.healer_v_d = None
            self.cold_scale_gu_f32 = None
            self.cold_scale_d_f32 = None
            self.w4_gu_u8 = None
            self.s4_gu_bf16 = None
            self.w4_d_u8 = None
            self.s4_d_bf16 = None
            self.w2_gu_u8 = None
            self.s2_gu_fp8 = None
            self.w2_d_u8 = None
            self.s2_d_fp8 = None
            self.w1_gu_u8 = None
            self.s1_gu_fp8 = None
            self.w1_d_u8 = None
            self.s1_d_fp8 = None
            self.w1_gu_cpu = None
            self.s1_gu_cpu = None
            self.w1_d_cpu = None
            self.s1_d_cpu = None
            self.t3_on_cpu = False
            self.tier_slot_map_gpu = None
            eager_keys = (
                ("s_gu.weight_scale_2", "s_gu.beta", "s_down.weight_scale_2", "s_down.beta", "tail.gu_bits", "tail.down_bits")
                if mmap_zero_ram
                else (
                    "s_gu.weight", "s_gu.weight_scale", "s_gu.weight_scale_2", "s_gu.beta",
                    "s_down.weight", "s_down.weight_scale", "s_down.weight_scale_2", "s_down.beta",
                    "healer.u_gu", "healer.v_gu", "healer.u_down", "healer.v_down",
                    "tail.gu_bits", "tail.down_bits",
                    "tail.gu_packed", "tail.gu_scale", "tail.down_packed", "tail.down_scale",
                )
            )
            for g_idx in range(48):
                gp = f"{tp}.chronos_groups.{g_idx}."
                if f"{gp}s_gu.weight" not in all_keys:
                    gp = f"{tp}.chronos_groups.0."
                sub_t = {}
                for k_suffix in eager_keys:
                    full_k = f"{gp}{k_suffix}"
                    if full_k in all_keys:
                        sub_t[k_suffix] = self._mmap_handle.get_tensor(full_k)
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

            self.shared_slot = self.hot_cache_size
            self.sh_gu_fp8 = None
            self.sh_gu_s = None
            self.sh_d_fp8 = None
            self.sh_d_s = None
            self.cache_gu_u8 = None
            self.cache_gu_s_bf16 = None
            self.cache_d_u8 = None
            self.cache_d_s_bf16 = None
            self.cache_gu_buf = torch.empty((self.hot_cache_size + 1, 4608, 5120), dtype=torch.bfloat16, device=device)
            self.cache_d_buf = torch.empty((self.hot_cache_size + 1, 5120, 2304), dtype=torch.bfloat16, device=device)
            self.cache_gu_buf[self.shared_slot].copy_(sh_gu_tmp)
            self.cache_d_buf[self.shared_slot].copy_(sh_down_tmp)
            del sh_gu_tmp, sh_down_tmp, sh_w1, sh_w3
            self.sh_gu = self.cache_gu_buf[self.shared_slot]
            self.sh_down = self.cache_d_buf[self.shared_slot]
            self.slot_gu = [self.cache_gu_buf[s] for s in range(self.hot_cache_size)]
            self.slot_d = [self.cache_d_buf[s] for s in range(self.hot_cache_size)]

        self.expert_to_slot: list[int] = [-1] * 384
        self.expert_to_slot_gpu = torch.full((384,), -1, dtype=torch.int32, device=device)
        self.slot_to_expert: list[int] = [-1] * (self.n4_slots if self.use_fp4_slot_cache else self.hot_cache_size)
        self.ring_cursor = 0
        self.cache_hits = 0
        self.cache_misses = 0

        self._top_ids_buf = torch.zeros((self.top_k,), dtype=torch.int32, device=device)
        self._active_slots_buf = torch.empty((self.top_k + 1,), dtype=torch.int32, device=device)
        self._active_slots_buf[self.top_k] = self.shared_slot
        self._active_gates_buf = torch.ones((self.top_k + 1,), dtype=torch.float32, device=device)
        self._act_inter_buf = torch.empty((self.top_k + 1, 2304), dtype=torch.bfloat16, device=device)
        self._y_partial_buf = torch.empty((self.top_k + 1, 5120), dtype=torch.float32, device=device)
        if self.use_fp4_slot_cache:
            self._y_final_buf = torch.empty((1, 5120), dtype=torch.bfloat16, device=device)

        self.pinned_slots = 0
        with torch.no_grad():
            if self.use_fp4_slot_cache:
                gen_mc = torch.Generator(device=device).manual_seed(42 + layer_idx)
                x_mc = F.normalize(
                    torch.randn((2048, 5120), generator=gen_mc, device=device, dtype=torch.bfloat16), dim=-1
                ) * (5120.0**0.5)
                fn_k = f"layers.{layer_idx}.ffn_norm.weight"
                if fn_k not in all_keys and f"model.{fn_k}" in all_keys:
                    fn_k = f"model.{fn_k}"
                if fn_k in all_keys:
                    x_mc = x_mc * self._mmap_handle.get_tensor(fn_k).to(
                        device=device, dtype=torch.bfloat16
                    ).unsqueeze(0)
                logits_mc = F.linear(x_mc, self.router_w).float()
                scores_mc = F.softplus(logits_mc).sqrt() + self.e_score_bias.unsqueeze(0)
                _, top6_mc = torch.topk(scores_mc, k=self.top_k, dim=-1)
                freq_mc = torch.bincount(top6_mc.view(-1), minlength=384).float()
                prior_mc = (freq_mc + scores_mc.mean(dim=0)).cpu().tolist()
                del x_mc, logits_mc, scores_mc, top6_mc, freq_mc

                hot_set = set(self.hot_expert_ids_list)
                cold_ranked = sorted(
                    (eid for eid in range(384) if eid not in hot_set),
                    key=lambda e: prior_mc[e],
                    reverse=True,
                )
                hot_ranked = sorted(self.hot_expert_ids_list, key=lambda e: prior_mc[e], reverse=True)
                ranked_eids = hot_ranked + cold_ranked
                tier1_eids = ranked_eids[: self.n4_slots]
                tier2_eids = ranked_eids[self.n4_slots : self.n4_slots + self.n2_slots]
                tier3_eids = ranked_eids[
                    self.n4_slots + self.n2_slots : self.n4_slots + self.n2_slots + self.n1_slots
                ]

                needed_grps = {eid // 8 for eid in tier1_eids if self.hot_expert_to_idx[eid] < 0}
                for eid in tier2_eids:
                    if self.hot_expert_to_idx[eid] < 0:
                        needed_grps.add(eid // 8)
                for eid in tier3_eids:
                    if self.hot_expert_to_idx[eid] < 0:
                        needed_grps.add(eid // 8)
                grp_map: dict[int, ChronosDsv41HolographicExpertGroup] = {}
                for g_idx in sorted(needed_grps):
                    gp = f"{tp}.chronos_groups.{g_idx}."
                    if f"{gp}s_gu.weight" not in all_keys:
                        gp = f"{tp}.chronos_groups.0."
                    sub_t = {
                        "s_gu.weight_scale_2": self._mmap_handle.get_tensor(f"{gp}s_gu.weight_scale_2"),
                        "s_gu.beta": self._mmap_handle.get_tensor(f"{gp}s_gu.beta"),
                        "s_down.weight_scale_2": self._mmap_handle.get_tensor(f"{gp}s_down.weight_scale_2"),
                        "s_down.beta": self._mmap_handle.get_tensor(f"{gp}s_down.beta"),
                        "tail.gu_bits": self._mmap_handle.get_tensor(f"{gp}tail.gu_bits") if f"{gp}tail.gu_bits" in all_keys else torch.tensor(1),
                        "tail.down_bits": self._mmap_handle.get_tensor(f"{gp}tail.down_bits") if f"{gp}tail.down_bits" in all_keys else torch.tensor(1),
                    }
                    grp_map[g_idx] = ChronosDsv41HolographicExpertGroup(
                        group_idx=g_idx,
                        tensors=sub_t,
                        device=device,
                        offload_tail_to_cpu=True,
                        mmap_zero_ram=True,
                        mmap_handle=self._mmap_handle,
                        mmap_prefix=gp,
                        h128=shared_h128,
                        r_basis=self.r_basis,
                        swiglu_limit=swiglu_limit,
                    )

                tier_map_cpu = [-1] * 384
                for s4, eid in enumerate(tier1_eids):
                    tier_map_cpu[eid] = s4
                    self.expert_to_slot[eid] = s4
                    self.slot_to_expert[s4] = eid
                for s2, eid in enumerate(tier2_eids):
                    tier_map_cpu[eid] = 1024 + s2
                for s1, eid in enumerate(tier3_eids):
                    tier_map_cpu[eid] = 2048 + s1
                self.tier_slot_map_gpu.copy_(torch.tensor(tier_map_cpu, dtype=torch.int32, device=device))
                self.expert_to_slot_gpu.copy_(self.tier_slot_map_gpu)

                hot_t1_entries = [
                    (s4, eid) for s4, eid in enumerate(tier1_eids) if self.hot_expert_to_idx[eid] >= 0
                ]
                hot_sub_entries = [
                    (2, s2, eid) for s2, eid in enumerate(tier2_eids) if self.hot_expert_to_idx[eid] >= 0
                ] + [
                    (3, s1, eid) for s1, eid in enumerate(tier3_eids) if self.hot_expert_to_idx[eid] >= 0
                ]
                cold_all_by_group: dict[int, list[tuple[int, int, int]]] = {}
                for s4, eid in enumerate(tier1_eids):
                    if self.hot_expert_to_idx[eid] < 0:
                        cold_all_by_group.setdefault(eid // 8, []).append((1, s4, eid))
                for s2, eid in enumerate(tier2_eids):
                    if self.hot_expert_to_idx[eid] < 0:
                        cold_all_by_group.setdefault(eid // 8, []).append((2, s2, eid))
                for s1, eid in enumerate(tier3_eids):
                    if self.hot_expert_to_idx[eid] < 0:
                        cold_all_by_group.setdefault(eid // 8, []).append((3, s1, eid))

                if hot_sub_entries or cold_all_by_group:
                    scr_gu, scr_d = _get_shared_bf16_scratch(device)
                else:
                    scr_gu, scr_d = None, None

                if (hot_t1_entries or hot_sub_entries) and self._hot_handle is not None:
                    w1_w_all_cpu = self._hot_handle.get_tensor(f"{self._hot_tp}w1.weight")
                    w3_w_all_cpu = self._hot_handle.get_tensor(f"{self._hot_tp}w3.weight")
                    w2_w_all_cpu = self._hot_handle.get_tensor(f"{self._hot_tp}w2.weight")
                    w1_s_all = _copy_via_pinned_dma(
                        self._hot_handle.get_tensor(f"{self._hot_tp}w1.scale"), device, 107
                    )
                    w3_s_all = _copy_via_pinned_dma(
                        self._hot_handle.get_tensor(f"{self._hot_tp}w3.scale"), device, 108
                    )
                    w2_s_all = _copy_via_pinned_dma(
                        self._hot_handle.get_tensor(f"{self._hot_tp}w2.scale"), device, 109
                    )
                    for s4, eid in sorted(hot_t1_entries, key=lambda item: self.hot_expert_to_idx[item[1]]):
                        rel_h = self.hot_expert_to_idx[eid]
                        self.w4_gu_u8[s4, :2304].copy_(
                            _copy_via_pinned_dma(w1_w_all_cpu[rel_h].view(torch.uint8), device, 101),
                            non_blocking=True,
                        )
                        self.w4_gu_u8[s4, 2304:].copy_(
                            _copy_via_pinned_dma(w3_w_all_cpu[rel_h].view(torch.uint8), device, 102),
                            non_blocking=True,
                        )
                        self.w4_d_u8[s4].copy_(
                            _copy_via_pinned_dma(w2_w_all_cpu[rel_h].view(torch.uint8), device, 103),
                            non_blocking=True,
                        )
                        self.s4_gu_bf16[s4, :2304].copy_(
                            ((w1_s_all[rel_h].view(torch.uint8).to(torch.int16) - 2).clamp(1, 254) << 7).view(torch.bfloat16)
                        )
                        self.s4_gu_bf16[s4, 2304:].copy_(
                            ((w3_s_all[rel_h].view(torch.uint8).to(torch.int16) - 2).clamp(1, 254) << 7).view(torch.bfloat16)
                        )
                        self.s4_d_bf16[s4].copy_(
                            ((w2_s_all[rel_h].view(torch.uint8).to(torch.int16) - 2).clamp(1, 254) << 7).view(torch.bfloat16)
                        )

                    if hot_sub_entries:
                        tmp_u8_gu = torch.empty((4608, 2560), dtype=torch.uint8, device=device)
                        tmp_s_gu = torch.empty((4608, 160), dtype=torch.bfloat16, device=device)
                        tmp_u8_d = torch.empty((5120, 1152), dtype=torch.uint8, device=device)
                        tmp_s_d = torch.empty((5120, 72), dtype=torch.bfloat16, device=device)
                        for tier_id, slot_i, eid in sorted(
                            hot_sub_entries, key=lambda item: self.hot_expert_to_idx[item[2]]
                        ):
                            rel_h = self.hot_expert_to_idx[eid]
                            tmp_u8_gu[:2304].copy_(
                                _copy_via_pinned_dma(w1_w_all_cpu[rel_h].view(torch.uint8), device, 110),
                                non_blocking=True,
                            )
                            tmp_u8_gu[2304:].copy_(
                                _copy_via_pinned_dma(w3_w_all_cpu[rel_h].view(torch.uint8), device, 111),
                                non_blocking=True,
                            )
                            tmp_u8_d.copy_(
                                _copy_via_pinned_dma(w2_w_all_cpu[rel_h].view(torch.uint8), device, 112),
                                non_blocking=True,
                            )
                            tmp_s_gu[:2304].copy_(
                                ((w1_s_all[rel_h].view(torch.uint8).to(torch.int16) - 2).clamp(1, 254) << 7).view(torch.bfloat16)
                            )
                            tmp_s_gu[2304:].copy_(
                                ((w3_s_all[rel_h].view(torch.uint8).to(torch.int16) - 2).clamp(1, 254) << 7).view(torch.bfloat16)
                            )
                            tmp_s_d.copy_(
                                ((w2_s_all[rel_h].view(torch.uint8).to(torch.int16) - 2).clamp(1, 254) << 7).view(torch.bfloat16)
                            )
                            _dequantize_fp4_bf16scale_into(tmp_u8_gu, tmp_s_gu, scr_gu)
                            _dequantize_fp4_bf16scale_into(tmp_u8_d, tmp_s_d, scr_d)
                            if self.r_basis is not None:
                                rb = self.r_basis.to(torch.bfloat16)
                                scr_d.addmm_(rb.T, torch.matmul(rb, scr_d), beta=1.0, alpha=-1.0)
                            if tier_id == 2:
                                _triton_quantize_2bit_g64_kernel[(4608, 80)](
                                    scr_gu, self.w2_gu_u8[slot_i], self.s2_gu_fp8[slot_i], N_GROUPS_C=80, num_warps=1
                                )
                                _triton_quantize_2bit_g64_kernel[(5120, 36)](
                                    scr_d, self.w2_d_u8[slot_i], self.s2_d_fp8[slot_i], N_GROUPS_C=36, num_warps=1
                                )
                            elif self.t3_on_cpu:
                                _triton_quantize_1bit_g64_kernel[(4608, 80)](
                                    scr_gu, self.w1_gu_u8[0], self.s1_gu_fp8[0], N_GROUPS_C=80, num_warps=1
                                )
                                _triton_quantize_1bit_g64_kernel[(5120, 36)](
                                    scr_d, self.w1_d_u8[0], self.s1_d_fp8[0], N_GROUPS_C=36, num_warps=1
                                )
                                self.w1_gu_cpu[slot_i].copy_(self.w1_gu_u8[0], non_blocking=False)
                                self.s1_gu_cpu[slot_i].copy_(self.s1_gu_fp8[0], non_blocking=False)
                                self.w1_d_cpu[slot_i].copy_(self.w1_d_u8[0], non_blocking=False)
                                self.s1_d_cpu[slot_i].copy_(self.s1_d_fp8[0], non_blocking=False)
                            else:
                                _triton_quantize_1bit_g64_kernel[(4608, 80)](
                                    scr_gu, self.w1_gu_u8[slot_i], self.s1_gu_fp8[slot_i], N_GROUPS_C=80, num_warps=1
                                )
                                _triton_quantize_1bit_g64_kernel[(5120, 36)](
                                    scr_d, self.w1_d_u8[slot_i], self.s1_d_fp8[slot_i], N_GROUPS_C=36, num_warps=1
                                )
                        del tmp_u8_gu, tmp_s_gu, tmp_u8_d, tmp_s_d
                    del w1_w_all_cpu, w3_w_all_cpu, w2_w_all_cpu, w1_s_all, w3_s_all, w2_s_all

                for g_idx, entries in sorted(cold_all_by_group.items()):
                    grp = grp_map[g_idx]
                    gp = grp.mmap_prefix
                    s_gu_p = _copy_via_pinned_dma(self._mmap_handle.get_tensor(f"{gp}s_gu.weight"), device, 201)
                    s_gu_s = _copy_via_pinned_dma(self._mmap_handle.get_tensor(f"{gp}s_gu.weight_scale"), device, 202)
                    s_dn_p = _copy_via_pinned_dma(self._mmap_handle.get_tensor(f"{gp}s_down.weight"), device, 203)
                    s_dn_s = _copy_via_pinned_dma(self._mmap_handle.get_tensor(f"{gp}s_down.weight_scale"), device, 204)
                    u_gu_g = _copy_via_pinned_dma(self._mmap_handle.get_tensor(f"{gp}healer.u_gu"), device, 205)
                    v_gu_g = _copy_via_pinned_dma(self._mmap_handle.get_tensor(f"{gp}healer.v_gu"), device, 206)
                    u_dn_g = _copy_via_pinned_dma(self._mmap_handle.get_tensor(f"{gp}healer.u_down"), device, 207)
                    v_dn_g = _copy_via_pinned_dma(self._mmap_handle.get_tensor(f"{gp}healer.v_down"), device, 208)
                    gu_p_g = _copy_via_pinned_dma(self._mmap_handle.get_tensor(f"{gp}tail.gu_packed"), device, 209)
                    gu_s_g = _copy_via_pinned_dma(self._mmap_handle.get_tensor(f"{gp}tail.gu_scale"), device, 210)
                    dn_p_g = _copy_via_pinned_dma(self._mmap_handle.get_tensor(f"{gp}tail.down_packed"), device, 211)
                    dn_s_g = _copy_via_pinned_dma(self._mmap_handle.get_tensor(f"{gp}tail.down_scale"), device, 212)
                    s_gu_c = unpack_nvfp4(s_gu_p, s_gu_s, grp.s_gu_scale2, dtype=torch.bfloat16)
                    s_dn_c = unpack_nvfp4(s_dn_p, s_dn_s, grp.s_down_scale2, dtype=torch.bfloat16)
                    del s_gu_p, s_gu_s, s_dn_p, s_dn_s
                    for tier_id, slot_i, eid in entries:
                        m_loc = eid % 8
                        grp.reconstruct_expert_into_slot(
                            m_loc,
                            scr_gu,
                            scr_d,
                            s_gu_cached=s_gu_c,
                            s_dn_cached=s_dn_c,
                            include_tail=True,
                            healer_slices=(u_gu_g[m_loc], v_gu_g[m_loc], u_dn_g[m_loc], v_dn_g[m_loc]),
                            tail_slices=(gu_p_g[m_loc], gu_s_g[m_loc], dn_p_g[m_loc], dn_s_g[m_loc]),
                        )
                        if tier_id == 1:
                            _triton_quantize_mse_fp4_32_kernel[(4608, 160)](
                                scr_gu, self.w4_gu_u8[slot_i], self.s4_gu_bf16[slot_i], N_BLOCKS_C=160, num_warps=1
                            )
                            _triton_quantize_mse_fp4_32_kernel[(5120, 72)](
                                scr_d, self.w4_d_u8[slot_i], self.s4_d_bf16[slot_i], N_BLOCKS_C=72, num_warps=1
                            )
                        elif tier_id == 2:
                            _triton_quantize_2bit_g64_kernel[(4608, 80)](
                                scr_gu, self.w2_gu_u8[slot_i], self.s2_gu_fp8[slot_i], N_GROUPS_C=80, num_warps=1
                            )
                            _triton_quantize_2bit_g64_kernel[(5120, 36)](
                                scr_d, self.w2_d_u8[slot_i], self.s2_d_fp8[slot_i], N_GROUPS_C=36, num_warps=1
                            )
                        elif self.t3_on_cpu:
                            _triton_quantize_1bit_g64_kernel[(4608, 80)](
                                scr_gu, self.w1_gu_u8[0], self.s1_gu_fp8[0], N_GROUPS_C=80, num_warps=1
                            )
                            _triton_quantize_1bit_g64_kernel[(5120, 36)](
                                scr_d, self.w1_d_u8[0], self.s1_d_fp8[0], N_GROUPS_C=36, num_warps=1
                            )
                            self.w1_gu_cpu[slot_i].copy_(self.w1_gu_u8[0], non_blocking=False)
                            self.s1_gu_cpu[slot_i].copy_(self.s1_gu_fp8[0], non_blocking=False)
                            self.w1_d_cpu[slot_i].copy_(self.w1_d_u8[0], non_blocking=False)
                            self.s1_d_cpu[slot_i].copy_(self.s1_d_fp8[0], non_blocking=False)
                        else:
                            _triton_quantize_1bit_g64_kernel[(4608, 80)](
                                scr_gu, self.w1_gu_u8[slot_i], self.s1_gu_fp8[slot_i], N_GROUPS_C=80, num_warps=1
                            )
                            _triton_quantize_1bit_g64_kernel[(5120, 36)](
                                scr_d, self.w1_d_u8[slot_i], self.s1_d_fp8[slot_i], N_GROUPS_C=36, num_warps=1
                            )
                    del s_gu_c, s_dn_c, u_gu_g, v_gu_g, u_dn_g, v_dn_g, gu_p_g, gu_s_g, dn_p_g, dn_s_g

                for g_obj in grp_map.values():
                    g_obj.mmap_handle = None
                del grp_map
                grp = None
                sub_t = None
                self.pinned_slots = self.n4_slots
                self.groups = nn.ModuleList()
                self._mmap_handle = None
                self._hot_handle = None
                self._hot_tensors = None
                if layer_idx == 39:
                    _EXPERT_KEY_CACHE.clear()
                    _PINNED_DMA_BUFS.clear()
                    _SHARED_SCRATCH_GU_BF16.clear()
                    _SHARED_SCRATCH_D_BF16.clear()
                gc.collect()
                if layer_idx % 2 == 1 or layer_idx == 39:
                    torch.cuda.empty_cache()
                if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
                    for fpath in (self.shard_path_str, str(self._hot_shard_cand)):
                        try:
                            rp = os.path.realpath(fpath)
                            if os.path.exists(rp):
                                fd = os.open(rp, os.O_RDONLY)
                                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                                os.close(fd)
                        except OSError:
                            pass
            else:
                rw_f = self.router_w.float()
                _, _, vh_rt = torch.linalg.svd(rw_f, full_matrices=False)
                u_ctx = vh_rt[:32, :].T
                prior_scores = torch.linalg.norm(rw_f @ u_ctx, dim=-1) + self.e_score_bias
                n_pin = min(8, max(0, self.hot_cache_size - self.top_k))
                _, top_hub_ids = torch.topk(prior_scores, k=n_pin)
                for slot_i, eid in enumerate(top_hub_ids.tolist()):
                    self._ensure_expert_slot(int(eid))
                self.pinned_slots = n_pin
                if self.ring_cursor < self.pinned_slots:
                    self.ring_cursor = self.pinned_slots % self.hot_cache_size

            if layer_idx == 2 and self.use_fp4_slot_cache and _HAS_TRITON and self.device.type == "cuda":
                _ = self.forward(torch.zeros((1, 5120), dtype=torch.bfloat16, device=self.device))
            self.cache_hits = 0
            self.cache_misses = 0

        if self.use_fp4_slot_cache and (self.n4_slots + self.n2_slots + self.n1_slots >= 384):
            self._hot_handle = None
            self._mmap_handle = None
            if self.layer_idx >= 39:
                _PINNED_DMA_BUFS.clear()
                _SHARED_SCRATCH_GU_BF16.clear()
                _SHARED_SCRATCH_D_BF16.clear()
            gc.collect()

        if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
            for _ev_p in (self.shard_path_str, str(self._hot_shard_cand)):
                try:
                    _rp = os.path.realpath(_ev_p)
                    if os.path.exists(_rp):
                        fd = os.open(_rp, os.O_RDONLY)
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                        os.close(fd)
                except OSError:
                    pass

    def _try_attach_hot_shard(self) -> bool:
        if self.num_hot_mxfp4_experts > 0:
            return True
        if not self._hot_shard_cand.exists():
            return False
        try:
            if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_SEQUENTIAL"):
                try:
                    fd_h = os.open(os.path.realpath(str(self._hot_shard_cand)), os.O_RDONLY)
                    os.posix_fadvise(fd_h, 0, 0, os.POSIX_FADV_SEQUENTIAL)
                    os.close(fd_h)
                except OSError:
                    pass
            self._hot_handle = safe_open(str(self._hot_shard_cand), framework="pt", device="cpu")
            hot_keys = set(self._hot_handle.keys())
            if f"{self._hot_tp}expert_ids" not in hot_keys and f"layers.{self.layer_idx}.ffn.hot_experts.expert_ids" in hot_keys:
                self._hot_tp = f"layers.{self.layer_idx}.ffn.hot_experts."
            hot_eids = self._hot_handle.get_tensor(f"{self._hot_tp}expert_ids").tolist()
            self.hot_expert_ids_list = [int(x) for x in hot_eids]
            for idx_h, eid_h in enumerate(hot_eids):
                self.hot_expert_to_idx[int(eid_h)] = idx_h
            self.num_hot_mxfp4_experts = len(hot_eids)
            if not self.mmap_zero_ram:
                self._hot_tensors = {
                    k_s: self._hot_handle.get_tensor(f"{self._hot_tp}{k_s}")
                    for k_s in ("w1.weight", "w1.scale", "w3.weight", "w3.scale", "w2.weight", "w2.scale")
                }
                self._hot_handle = None
            return True
        except Exception:
            self._hot_handle = None
            return False

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        for _ in weights:
            pass
        return {name for name, _ in self.named_parameters(recurse=True)}

    def finalize_mega_moe_weights(self) -> None:
        pass

    def defer_finalize(self) -> Any:
        return False

    def defers_finalize(self, num_tokens: int = 1) -> bool:
        return False

    @torch.no_grad()
    def _reconstruct_hot_mxfp4_into(self, h_idx: int, out_gu: torch.Tensor, out_d: torch.Tensor) -> None:
        if self._hot_tensors is not None:
            w1_w = self._hot_tensors["w1.weight"][h_idx]
            w1_s = self._hot_tensors["w1.scale"][h_idx]
            w3_w = self._hot_tensors["w3.weight"][h_idx]
            w3_s = self._hot_tensors["w3.scale"][h_idx]
            w2_w = self._hot_tensors["w2.weight"][h_idx]
            w2_s = self._hot_tensors["w2.scale"][h_idx]
        else:
            w1_w = _copy_via_pinned_dma(self._hot_handle.get_slice(f"{self._hot_tp}w1.weight")[h_idx], self.device, 21)
            w1_s = _copy_via_pinned_dma(self._hot_handle.get_slice(f"{self._hot_tp}w1.scale")[h_idx], self.device, 22)
            w3_w = _copy_via_pinned_dma(self._hot_handle.get_slice(f"{self._hot_tp}w3.weight")[h_idx], self.device, 23)
            w3_s = _copy_via_pinned_dma(self._hot_handle.get_slice(f"{self._hot_tp}w3.scale")[h_idx], self.device, 24)
            w2_w = _copy_via_pinned_dma(self._hot_handle.get_slice(f"{self._hot_tp}w2.weight")[h_idx], self.device, 25)
            w2_s = _copy_via_pinned_dma(self._hot_handle.get_slice(f"{self._hot_tp}w2.scale")[h_idx], self.device, 26)

        out_gu[:2304].copy_(dequantize_mxfp4_32(w1_w, w1_s, self.device, dtype=torch.bfloat16))
        out_gu[2304:].copy_(dequantize_mxfp4_32(w3_w, w3_s, self.device, dtype=torch.bfloat16))
        out_d.copy_(dequantize_mxfp4_32(w2_w, w2_s, self.device, dtype=torch.bfloat16))
        if self.r_basis is not None:
            rb = self.r_basis.to(torch.bfloat16)
            proj = torch.matmul(rb, out_d)
            out_d.addmm_(rb.T, proj, beta=1.0, alpha=-1.0)

    @torch.no_grad()
    def _load_hot_mxfp4_into_fp4_slot(self, h_idx: int, slot: int) -> None:
        assert self.cache_gu_u8 is not None and self.cache_gu_s_bf16 is not None
        assert self.cache_d_u8 is not None and self.cache_d_s_bf16 is not None
        if self._hot_tensors is not None:
            w1_w = self._hot_tensors["w1.weight"][h_idx].to(self.device, non_blocking=True)
            w1_s = self._hot_tensors["w1.scale"][h_idx].to(self.device, non_blocking=True)
            w3_w = self._hot_tensors["w3.weight"][h_idx].to(self.device, non_blocking=True)
            w3_s = self._hot_tensors["w3.scale"][h_idx].to(self.device, non_blocking=True)
            w2_w = self._hot_tensors["w2.weight"][h_idx].to(self.device, non_blocking=True)
            w2_s = self._hot_tensors["w2.scale"][h_idx].to(self.device, non_blocking=True)
        else:
            w1_w = _copy_via_pinned_dma(self._hot_handle.get_slice(f"{self._hot_tp}w1.weight")[h_idx], self.device, 21)
            w1_s = _copy_via_pinned_dma(self._hot_handle.get_slice(f"{self._hot_tp}w1.scale")[h_idx], self.device, 22)
            w3_w = _copy_via_pinned_dma(self._hot_handle.get_slice(f"{self._hot_tp}w3.weight")[h_idx], self.device, 23)
            w3_s = _copy_via_pinned_dma(self._hot_handle.get_slice(f"{self._hot_tp}w3.scale")[h_idx], self.device, 24)
            w2_w = _copy_via_pinned_dma(self._hot_handle.get_slice(f"{self._hot_tp}w2.weight")[h_idx], self.device, 25)
            w2_s = _copy_via_pinned_dma(self._hot_handle.get_slice(f"{self._hot_tp}w2.scale")[h_idx], self.device, 26)

        self.cache_gu_u8[slot, :2304].copy_(w1_w.view(torch.uint8))
        self.cache_gu_u8[slot, 2304:].copy_(w3_w.view(torch.uint8))
        self.cache_d_u8[slot].copy_(w2_w.view(torch.uint8))
        self.cache_gu_s_bf16[slot, :2304].copy_(
            ((w1_s.view(torch.uint8).to(torch.int16) - 2).clamp(1, 254) << 7).view(torch.bfloat16)
        )
        self.cache_gu_s_bf16[slot, 2304:].copy_(
            ((w3_s.view(torch.uint8).to(torch.int16) - 2).clamp(1, 254) << 7).view(torch.bfloat16)
        )
        self.cache_d_s_bf16[slot].copy_(
            ((w2_s.view(torch.uint8).to(torch.int16) - 2).clamp(1, 254) << 7).view(torch.bfloat16)
        )

    @torch.no_grad()
    def _ensure_expert_slot(
        self,
        expert_id: int,
        s_gu_cached: torch.Tensor | None = None,
        s_dn_cached: torch.Tensor | None = None,
        protected_eids: set[int] | None = None,
    ) -> int:
        slot = self.expert_to_slot[expert_id]
        if slot >= 0:
            self.cache_hits += 1
            return slot

        self.cache_misses += 1
        dyn_span = max(1, self.hot_cache_size - self.pinned_slots)
        slot = self.ring_cursor
        for _ in range(dyn_span):
            cand = self.ring_cursor
            self.ring_cursor = self.pinned_slots + ((self.ring_cursor - self.pinned_slots + 1) % dyn_span)
            if protected_eids is None or self.slot_to_expert[cand] not in protected_eids:
                slot = cand
                break

        old_eid = self.slot_to_expert[slot]
        if old_eid >= 0:
            self.expert_to_slot[old_eid] = -1
            self.expert_to_slot_gpu[old_eid] = -1

        if self.num_hot_mxfp4_experts == 0:
            self._try_attach_hot_shard()
        h_idx = self.hot_expert_to_idx[expert_id]
        if self.use_fp4_slot_cache:
            assert self.cache_gu_u8 is not None and self.cache_gu_s_bf16 is not None
            assert self.cache_d_u8 is not None and self.cache_d_s_bf16 is not None
            if h_idx >= 0:
                self._load_hot_mxfp4_into_fp4_slot(h_idx, slot)
            else:
                g_idx = expert_id // 8
                local_m = expert_id % 8
                scr_gu, scr_d = _get_shared_bf16_scratch(self.device)
                self.groups[g_idx].reconstruct_expert_into_slot(
                    local_m, scr_gu, scr_d, s_gu_cached=s_gu_cached, s_dn_cached=s_dn_cached
                )
                _triton_quantize_mse_fp4_32_kernel[(4608, 160)](
                    scr_gu, self.cache_gu_u8[slot], self.cache_gu_s_bf16[slot], N_BLOCKS_C=160, num_warps=1
                )
                _triton_quantize_mse_fp4_32_kernel[(5120, 72)](
                    scr_d, self.cache_d_u8[slot], self.cache_d_s_bf16[slot], N_BLOCKS_C=72, num_warps=1
                )
        else:
            if h_idx >= 0:
                self._reconstruct_hot_mxfp4_into(h_idx, self.slot_gu[slot], self.slot_d[slot])
            else:
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
    def _stage_t3_bounce_slots(self, k_act: int) -> None:
        if not self.t3_on_cpu or self.w1_gu_cpu is None:
            return
        if _HAS_TRITON and _triton_stage_t3_dma_kernel is not None and self.device.type == "cuda" and self.w1_gu_cpu.is_pinned():
            _triton_stage_t3_dma_kernel[(108,)](
                self._active_slots_buf,
                self.w1_gu_u8,
                self.w1_gu_cpu,
                self.s1_gu_fp8,
                self.s1_gu_cpu,
                self.w1_d_u8,
                self.w1_d_cpu,
                self.s1_d_fp8,
                self.s1_d_cpu,
                K_ACT=k_act,
                BLOCK_BYTES=1024,
                num_warps=4,
            )
            _triton_remap_t3_slots_kernel[(1,)](self._active_slots_buf, K_ACT=k_act, num_warps=1)
            return
        slots_list = self._active_slots_buf[:k_act].tolist()
        if not any(s >= 2048 for s in slots_list):
            return
        b_idx = 0
        for idx_k, s_val in enumerate(slots_list):
            if s_val >= 2048:
                s1 = s_val - 2048
                self.w1_gu_u8[b_idx].copy_(self.w1_gu_cpu[s1], non_blocking=True)
                self.s1_gu_fp8[b_idx].copy_(self.s1_gu_cpu[s1], non_blocking=True)
                self.w1_d_u8[b_idx].copy_(self.w1_d_cpu[s1], non_blocking=True)
                self.s1_d_fp8[b_idx].copy_(self.s1_d_cpu[s1], non_blocking=True)
                slots_list[idx_k] = 2048 + b_idx
                b_idx += 1
        self._active_slots_buf[:k_act].copy_(
            torch.tensor(slots_list, dtype=torch.int32, device=self.device)
        )

    @torch.no_grad()
    def _get_slot_bf16(self, slot: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_fp4_slot_cache:
            return self.slot_gu[slot], self.slot_d[slot]
        assert self.cache_gu_u8 is not None and self.cache_gu_s_bf16 is not None
        assert self.cache_d_u8 is not None and self.cache_d_s_bf16 is not None
        scr_gu, scr_d = _get_shared_bf16_scratch(self.device)
        _dequantize_fp4_bf16scale_into(self.cache_gu_u8[slot], self.cache_gu_s_bf16[slot], scr_gu)
        _dequantize_fp4_bf16scale_into(self.cache_d_u8[slot], self.cache_d_s_bf16[slot], scr_d)
        return scr_gu, scr_d

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        mega_gate_metadata: Any = None,
        image_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        orig_shape = hidden_states.shape
        orig_dtype = hidden_states.dtype
        x = hidden_states.view(-1, orig_shape[-1]).to(torch.bfloat16)
        n_toks = x.shape[0]
        if n_toks == 0:
            return hidden_states

        # Zero-sync 100% GPU-Resident Fused Tiered (4-bit MXFP4 + 2-bit Group-64 + 1-bit Group-64) Triton Path
        if self.use_fp4_slot_cache and _HAS_TRITON and x.is_cuda:
            if image_mask is None and input_ids is not None:
                if input_ids.dtype == torch.bool:
                    image_mask = input_ids
                elif self.bias_vl is not None:
                    image_mask = (input_ids >= 129264) & (input_ids < 129269)

            raw_logits = F.linear(x, self.router_w)
            has_t2 = self.n2_slots > 0
            has_t3 = self.n1_slots > 0
            if n_toks == 1:
                out_y = getattr(self, "_out_y_buf", None)
                if out_y is None or out_y.shape != x.shape or out_y.is_inference():
                    with torch.inference_mode(False):
                        out_y = torch.empty_like(x)
                        self._out_y_buf = out_y
                bias_vec = self.e_score_bias
                if image_mask is not None and self.bias_vl is not None and image_mask.numel() == 1:
                    bias_vec = torch.where(image_mask.view(1), self.bias_vl, self.e_score_bias)
                k_act = self.decode_top_k
                n_active = k_act + 1
                _triton_dsv41_tiered_router_kernel[(1,)](
                    raw_logits,
                    bias_vec,
                    self.tier_slot_map_gpu,
                    self._active_slots_buf,
                    self._active_gates_buf,
                    float(self.routed_scaling_factor),
                    SHARED_SLOT=self.shared_slot,
                    K_ACT=k_act,
                    num_warps=4,
                )
                if self.t3_on_cpu:
                    self._stage_t3_bounce_slots(k_act)
                _triton_dsv41_tiered_gu_swiglu_kernel[(2304 // 8, n_active)](
                    x,
                    self.w4_gu_u8,
                    self.s4_gu_bf16,
                    self.w2_gu_u8,
                    self.s2_gu_fp8,
                    self.w1_gu_u8,
                    self.s1_gu_fp8,
                    self._active_slots_buf,
                    self._active_gates_buf,
                    self._act_inter_buf,
                    float(self.swiglu_limit),
                    N_INTER=2304,
                    N_BLOCKS_K4=160,
                    N_GROUPS_K2=80,
                    BLOCK_N=8,
                    sub_blocks_k4=8,
                    sub_groups_k2=4,
                    HAS_TIER2=has_t2,
                    HAS_TIER3=has_t3,
                    num_warps=4,
                )
                _triton_dsv41_tiered_down_kernel[(5120 // 8, n_active)](
                    self._act_inter_buf,
                    self.w4_d_u8,
                    self.s4_d_bf16,
                    self.w2_d_u8,
                    self.s2_d_fp8,
                    self.w1_d_u8,
                    self.s1_d_fp8,
                    self._active_slots_buf,
                    self._y_partial_buf,
                    D_OUT=5120,
                    N_BLOCKS_K4=72,
                    N_GROUPS_K2=36,
                    BLOCK_D=8,
                    sub_blocks_k4=8,
                    sub_groups_k2=4,
                    HAS_TIER2=has_t2,
                    HAS_TIER3=has_t3,
                    num_warps=8,
                )
                _triton_dsv41_reduce_sum_bf16_kernel[(40,)](
                    self._y_partial_buf,
                    out_y[0],
                    N_ACT=n_active,
                    D_OUT=5120,
                    num_warps=2,
                )
                if self.r_basis is not None:
                    rb = self.r_basis.to(out_y.dtype)
                    out_y.addmm_(out_y @ rb.T, rb, beta=1.0, alpha=-1.0)
                out_y_v = out_y.view(orig_shape)
                return out_y_v if orig_dtype == torch.bfloat16 else out_y_v.to(orig_dtype)

            out_y = torch.empty_like(x)
            k_act = self.decode_top_k if n_toks <= 4 else self.top_k
            n_active = k_act + 1
            im_flat = image_mask.view(-1) if (image_mask is not None and image_mask.numel() == n_toks) else None
            for t_i in range(n_toks):
                bias_vec = self.e_score_bias
                if im_flat is not None and self.bias_vl is not None:
                    bias_vec = torch.where(im_flat[t_i : t_i + 1], self.bias_vl, self.e_score_bias)
                _triton_dsv41_tiered_router_kernel[(1,)](
                    raw_logits[t_i : t_i + 1],
                    bias_vec,
                    self.tier_slot_map_gpu,
                    self._active_slots_buf,
                    self._active_gates_buf,
                    float(self.routed_scaling_factor),
                    SHARED_SLOT=self.shared_slot,
                    K_ACT=k_act,
                    num_warps=4,
                )
                if self.t3_on_cpu:
                    self._stage_t3_bounce_slots(k_act)
                _triton_dsv41_tiered_gu_swiglu_kernel[(2304 // 8, n_active)](
                    x[t_i : t_i + 1],
                    self.w4_gu_u8,
                    self.s4_gu_bf16,
                    self.w2_gu_u8,
                    self.s2_gu_fp8,
                    self.w1_gu_u8,
                    self.s1_gu_fp8,
                    self._active_slots_buf,
                    self._active_gates_buf,
                    self._act_inter_buf,
                    float(self.swiglu_limit),
                    N_INTER=2304,
                    N_BLOCKS_K4=160,
                    N_GROUPS_K2=80,
                    BLOCK_N=8,
                    sub_blocks_k4=8,
                    sub_groups_k2=4,
                    HAS_TIER2=has_t2,
                    HAS_TIER3=has_t3,
                    num_warps=4,
                )
                _triton_dsv41_tiered_down_kernel[(5120 // 8, n_active)](
                    self._act_inter_buf,
                    self.w4_d_u8,
                    self.s4_d_bf16,
                    self.w2_d_u8,
                    self.s2_d_fp8,
                    self.w1_d_u8,
                    self.s1_d_fp8,
                    self._active_slots_buf,
                    self._y_partial_buf,
                    D_OUT=5120,
                    N_BLOCKS_K4=72,
                    N_GROUPS_K2=36,
                    BLOCK_D=8,
                    sub_blocks_k4=8,
                    sub_groups_k2=4,
                    HAS_TIER2=has_t2,
                    HAS_TIER3=has_t3,
                    num_warps=8,
                )
                _triton_dsv41_reduce_sum_bf16_kernel[(40,)](
                    self._y_partial_buf,
                    out_y[t_i],
                    N_ACT=n_active,
                    D_OUT=5120,
                    num_warps=2,
                )
            if self.r_basis is not None:
                rb = self.r_basis.to(out_y.dtype)
                out_y.addmm_(out_y @ rb.T, rb, beta=1.0, alpha=-1.0)
            out_y_v = out_y.view(orig_shape)
            return out_y_v if orig_dtype == torch.bfloat16 else out_y_v.to(orig_dtype)

        if image_mask is None and input_ids is not None:
            if input_ids.dtype == torch.bool:
                image_mask = input_ids
            elif self.bias_vl is not None:
                image_mask = (input_ids >= 129264) & (input_ids < 129269)

        # 1. sqrtsoftplus Top-6 noaux_tc Router Branch (BF16 GEMV + FP32 nonlinear scoring)
        raw_logits = F.linear(x, self.router_w).float()
        scores = F.softplus(raw_logits).sqrt()
        bias = self.e_score_bias.unsqueeze(0)
        if image_mask is not None and self.bias_vl is not None and image_mask.numel() == n_toks:
            bias = torch.where(image_mask.view(-1, 1), self.bias_vl.unsqueeze(0), bias)
        _, top_ids = torch.topk(scores + bias, k=self.top_k, dim=-1)
        top_weights = scores.gather(1, top_ids)
        top_gates_f32 = (top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-20)) * self.routed_scaling_factor

        if n_toks == 1:
            self._active_slots_buf[: self.top_k].copy_(self.expert_to_slot_gpu.index_select(0, top_ids[0]))
            self._active_gates_buf[: self.top_k].copy_(top_gates_f32[0])

            eids = [int(eid) for eid in top_ids[0].tolist()]
            if all(self.expert_to_slot[eid] >= 0 for eid in eids):
                self.cache_hits += self.top_k
                slot_ids = [self.expert_to_slot[eid] for eid in eids]
            else:
                protected_set = set(eids)
                group_bank_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
                slot_ids = []
                for eid_i in eids:
                    if self.expert_to_slot[eid_i] < 0 and self.hot_expert_to_idx[eid_i] < 0:
                        g_idx = eid_i // 8
                        if g_idx not in group_bank_cache:
                            group_bank_cache[g_idx] = self.groups[g_idx].get_group_banks()
                        s_gu_c, s_dn_c = group_bank_cache[g_idx]
                        slot = self._ensure_expert_slot(
                            eid_i, s_gu_cached=s_gu_c, s_dn_cached=s_dn_c, protected_eids=protected_set
                        )
                    else:
                        slot = self._ensure_expert_slot(eid_i, protected_eids=protected_set)
                    slot_ids.append(slot)
                self._active_slots_buf[: self.top_k].copy_(
                    torch.tensor(slot_ids, dtype=torch.int32, device=self.device)
                )

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

            top_gates = top_gates_f32.to(torch.bfloat16)
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
        top_gates = top_gates_f32.to(torch.bfloat16)
        if self.use_fp4_slot_cache:
            scr_gu, scr_d = self._get_slot_bf16(self.shared_slot)
            sh_gu_out = F.linear(x, scr_gu)
            y_shared = F.linear(dsv41_clamped_swiglu(sh_gu_out, 2304, self.swiglu_limit), scr_d)
        else:
            sh_gu_out = F.linear(x, self.sh_gu)
            y_shared = F.linear(dsv41_clamped_swiglu(sh_gu_out, 2304, self.swiglu_limit), self.sh_down)

        y_routed = torch.zeros_like(x)
        unique_eids = torch.unique(top_ids).tolist()
        warm_eids = [int(eid) for eid in unique_eids if self.expert_to_slot[int(eid)] >= 0]
        cold_eids = [int(eid) for eid in unique_eids if self.expert_to_slot[int(eid)] < 0]

        for eid in warm_eids:
            slot = self.expert_to_slot[eid]
            w_gu_s, w_d_s = self._get_slot_bf16(slot)
            mask = top_ids == eid
            tok_indices, slot_indices = torch.where(mask)
            x_sel = x[tok_indices]
            g_sel = top_gates[tok_indices, slot_indices].unsqueeze(-1)
            gu_out = F.linear(x_sel, w_gu_s)
            act = dsv41_clamped_swiglu(gu_out, 2304, self.swiglu_limit)
            y_exp = F.linear(act, w_d_s)
            y_routed.index_add_(0, tok_indices, g_sel * y_exp)

        if cold_eids:
            if self.use_fp4_slot_cache:
                cold_by_group: dict[int, list[int]] = {}
                for eid in cold_eids:
                    cold_by_group.setdefault(eid // 8, []).append(eid)
                for g_idx, g_eids in cold_by_group.items():
                    s_gu_c, s_dn_c = self.groups[g_idx].get_group_banks()
                    for eid in g_eids:
                        mask = top_ids == eid
                        tok_indices, slot_indices = torch.where(mask)
                        x_sel = x[tok_indices]
                        g_sel = top_gates[tok_indices, slot_indices].unsqueeze(-1)
                        y_exp = self.groups[g_idx].forward_cold_fast(
                            x_sel, eid % 8, s_gu_cached=s_gu_c, s_dn_cached=s_dn_c
                        )
                        y_routed.index_add_(0, tok_indices, g_sel * y_exp)
                    del s_gu_c, s_dn_c
            else:
                flat_ids = top_ids.view(-1)
                flat_gates = top_gates.view(-1).float()
                expert_mass = torch.zeros(384, dtype=torch.float32, device=self.device)
                expert_mass.scatter_add_(0, flat_ids, flat_gates)
                expert_mass.scatter_add_(0, top_ids[0], torch.full((self.top_k,), 100.0, dtype=torch.float32, device=self.device))
                cold_masses = [float(expert_mass[eid].item()) for eid in cold_eids]
                sorted_cold_asc = [eid for _, eid in sorted(zip(cold_masses, cold_eids), reverse=False)]
                dyn_span = max(1, self.hot_cache_size - self.pinned_slots)
                cache_cutoff = max(0, len(sorted_cold_asc) - dyn_span)
                group_bank_cache_pf: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
                for c_idx, eid in enumerate(sorted_cold_asc):
                    h_idx = self.hot_expert_to_idx[eid]
                    s_gu_c_pf: torch.Tensor | None = None
                    s_dn_c_pf: torch.Tensor | None = None
                    if h_idx < 0:
                        g_idx = eid // 8
                        if g_idx not in group_bank_cache_pf:
                            group_bank_cache_pf[g_idx] = self.groups[g_idx].get_group_banks()
                        s_gu_c_pf, s_dn_c_pf = group_bank_cache_pf[g_idx]

                    if c_idx >= cache_cutoff:
                        slot = self._ensure_expert_slot(eid, s_gu_cached=s_gu_c_pf, s_dn_cached=s_dn_c_pf)
                        w_gu_s, w_d_s = self._get_slot_bf16(slot)
                    else:
                        w_gu_s, w_d_s = _get_shared_bf16_scratch(self.device)
                        if h_idx >= 0:
                            self._reconstruct_hot_mxfp4_into(h_idx, w_gu_s, w_d_s)
                        else:
                            self.groups[eid // 8].reconstruct_expert_into_slot(
                                eid % 8, w_gu_s, w_d_s, s_gu_cached=s_gu_c_pf, s_dn_cached=s_dn_c_pf
                            )
                    mask = top_ids == eid
                    tok_indices, slot_indices = torch.where(mask)
                    x_sel = x[tok_indices]
                    g_sel = top_gates[tok_indices, slot_indices].unsqueeze(-1)
                    gu_out = F.linear(x_sel, w_gu_s)
                    act = dsv41_clamped_swiglu(gu_out, 2304, self.swiglu_limit)
                    y_exp = F.linear(act, w_d_s)
                    y_routed.index_add_(0, tok_indices, g_sel * y_exp)

        if self.use_fp4_slot_cache and self.r_basis is not None:
            rb = self.r_basis.to(y_routed.dtype)
            y_routed = y_routed - (y_routed @ rb.T) @ rb

        return (y_routed + y_shared).view(orig_shape).to(orig_dtype)
