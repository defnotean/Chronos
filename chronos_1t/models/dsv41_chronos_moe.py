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

except Exception:
    _HAS_TRITON = False

from chronos_1t.models.chronos_moe_block import (
    NVFP4_LUT,
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
    ) -> None:
        """Zero-temp-allocation in-place slot reconstruction (`v7.5`) with post-quantization refusal projection."""
        if s_gu_cached is None or s_dn_cached is None:
            s_gu, s_dn = self.get_group_banks()
        else:
            s_gu, s_dn = s_gu_cached, s_dn_cached

        u_gu_r, v_gu_r, u_dn_r, v_dn_r = self._fetch_healer_slices(local_m)
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
        slot_gu.addmm_(u_gu_r.to(torch.bfloat16), v_gu_r.to(torch.bfloat16))

        dr_s1, _, dr_inv, dr_s2, dc_s1, _, dc_inv, dc_s2 = generate_expert_1d_keys(
            5120, 2304, local_m, seed_base=303 + self.group_idx * 8, device=self.device
        )
        h_dn = unbind_2d_permuted_hadamard(s_dn, dr_s1, dr_inv, dr_s2, dc_s1, dc_inv, dc_s2, self.h128)
        if self.down_tail_bits == 2:
            slot_dn.copy_(unpack_2bit_lloyd_max(dn_p_m, dn_s_m, dtype=torch.bfloat16))
        else:
            slot_dn.copy_(unpack_1bit_rademacher(dn_p_m, dn_s_m, dtype=torch.bfloat16))
        slot_dn.add_(h_dn * self.beta_down[local_m])
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
        if f"{tp}.abliteration.r_basis" in all_keys:
            self.r_basis = self._mmap_handle.get_tensor(f"{tp}.abliteration.r_basis").to(device=device, dtype=torch.bfloat16).contiguous()
            self.r_basis_f32 = self.r_basis.float().contiguous()

        shared_h128 = build_hadamard_128(device, dtype=torch.bfloat16)
        self.groups = nn.ModuleList()

        if self.use_fp4_slot_cache:
            self._keys_384 = _get_shared_384_keys(device)
            self.group_gu_u8 = torch.empty((48, 4608, 2560), dtype=torch.uint8, device=device)
            self.group_gu_s_fp8 = torch.empty((48, 4608, 320), dtype=torch.float8_e4m3fn, device=device)
            self.group_d_u8 = torch.empty((48, 5120, 1152), dtype=torch.uint8, device=device)
            self.group_d_s_fp8 = torch.empty((48, 5120, 144), dtype=torch.float8_e4m3fn, device=device)
            self.healer_u_gu = torch.empty((384, 4608, 64), dtype=torch.float8_e4m3fn, device=device)
            self.healer_v_gu = torch.empty((384, 64, 5120), dtype=torch.float8_e4m3fn, device=device)
            self.healer_u_d = torch.empty((384, 5120, 64), dtype=torch.float8_e4m3fn, device=device)
            self.healer_v_d = torch.empty((384, 64, 2304), dtype=torch.float8_e4m3fn, device=device)
            self.cold_scale_gu_f32 = torch.empty((384,), dtype=torch.float32, device=device)
            self.cold_scale_d_f32 = torch.empty((384,), dtype=torch.float32, device=device)

            for g_idx in range(48):
                gp = f"{tp}.chronos_groups.{g_idx}."
                if f"{gp}s_gu.weight" not in all_keys:
                    gp = f"{tp}.chronos_groups.0."
                e_s, e_e = g_idx * 8, (g_idx + 1) * 8
                self.group_gu_u8[g_idx].copy_(self._mmap_handle.get_tensor(f"{gp}s_gu.weight"), non_blocking=True)
                self.group_gu_s_fp8[g_idx].copy_(self._mmap_handle.get_tensor(f"{gp}s_gu.weight_scale"), non_blocking=True)
                self.group_d_u8[g_idx].copy_(self._mmap_handle.get_tensor(f"{gp}s_down.weight"), non_blocking=True)
                self.group_d_s_fp8[g_idx].copy_(self._mmap_handle.get_tensor(f"{gp}s_down.weight_scale"), non_blocking=True)
                self.healer_u_gu[e_s:e_e].copy_(self._mmap_handle.get_tensor(f"{gp}healer.u_gu"), non_blocking=True)
                self.healer_v_gu[e_s:e_e].copy_(self._mmap_handle.get_tensor(f"{gp}healer.v_gu"), non_blocking=True)
                self.healer_u_d[e_s:e_e].copy_(self._mmap_handle.get_tensor(f"{gp}healer.u_down"), non_blocking=True)
                self.healer_v_d[e_s:e_e].copy_(self._mmap_handle.get_tensor(f"{gp}healer.v_down"), non_blocking=True)

                s_gu_s2 = self._mmap_handle.get_tensor(f"{gp}s_gu.weight_scale_2")
                b_gu = self._mmap_handle.get_tensor(f"{gp}s_gu.beta")
                s_dn_s2 = self._mmap_handle.get_tensor(f"{gp}s_down.weight_scale_2")
                b_dn = self._mmap_handle.get_tensor(f"{gp}s_down.beta")
                self.cold_scale_gu_f32[e_s:e_e].copy_(
                    (0.25 * s_gu_s2.float() * b_gu.float()).to(device, non_blocking=True)
                )
                self.cold_scale_d_f32[e_s:e_e].copy_(
                    (0.25 * s_dn_s2.float() * b_dn.float()).to(device, non_blocking=True)
                )

                sub_t = {
                    "s_gu.weight_scale_2": s_gu_s2,
                    "s_gu.beta": b_gu,
                    "s_down.weight_scale_2": s_dn_s2,
                    "s_down.beta": b_dn,
                    "tail.gu_bits": self._mmap_handle.get_tensor(f"{gp}tail.gu_bits") if f"{gp}tail.gu_bits" in all_keys else torch.tensor(1),
                    "tail.down_bits": self._mmap_handle.get_tensor(f"{gp}tail.down_bits") if f"{gp}tail.down_bits" in all_keys else torch.tensor(1),
                }
                grp = ChronosDsv41HolographicExpertGroup(
                    group_idx=g_idx,
                    tensors=sub_t,
                    device=device,
                    offload_tail_to_cpu=offload_tail_to_cpu,
                    mmap_zero_ram=True,
                    mmap_handle=self._mmap_handle,
                    mmap_prefix=gp,
                    h128=shared_h128,
                    r_basis=self.r_basis,
                    swiglu_limit=swiglu_limit,
                )
                grp.s_gu_packed = self.group_gu_u8[g_idx]
                grp.s_gu_scale = self.group_gu_s_fp8[g_idx]
                grp.s_down_packed = self.group_d_u8[g_idx]
                grp.s_down_scale = self.group_d_s_fp8[g_idx]
                grp.u_gu = self.healer_u_gu[e_s:e_e]
                grp.v_gu = self.healer_v_gu[e_s:e_e]
                grp.u_down = self.healer_u_d[e_s:e_e]
                grp.v_down = self.healer_v_d[e_s:e_e]
                grp.mmap_zero_ram = False
                grp.mmap_handle = None
                self.groups.append(grp)
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
        default_dec_k = "4" if layer_idx in (2, 3, 38, 39) else "3"
        self.decode_top_k = max(1, min(self.top_k, int(os.environ.get("CHRONOS_DECODE_TOP_K", default_dec_k))))
        if self.use_fp4_slot_cache:
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
            self.cache_gu_u8 = torch.empty((self.hot_cache_size + 1, 4608, 2560), dtype=torch.uint8, device=device)
            self.cache_gu_s_bf16 = torch.empty((self.hot_cache_size + 1, 4608, 160), dtype=torch.bfloat16, device=device)
            self.cache_d_u8 = torch.empty((self.hot_cache_size + 1, 5120, 1152), dtype=torch.uint8, device=device)
            self.cache_d_s_bf16 = torch.empty((self.hot_cache_size + 1, 5120, 72), dtype=torch.bfloat16, device=device)
            _triton_quantize_mse_fp4_32_kernel[(4608, 160)](
                sh_gu_tmp,
                self.cache_gu_u8[self.shared_slot],
                self.cache_gu_s_bf16[self.shared_slot],
                N_BLOCKS_C=160,
                num_warps=1,
            )
            _triton_quantize_mse_fp4_32_kernel[(5120, 72)](
                sh_down_tmp,
                self.cache_d_u8[self.shared_slot],
                self.cache_d_s_bf16[self.shared_slot],
                N_BLOCKS_C=72,
                num_warps=1,
            )
            del sh_gu_tmp, sh_down_tmp, sh_w1, sh_w3
        else:
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
        self.slot_to_expert: list[int] = [-1] * self.hot_cache_size
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
            self._x_hrr_buf = torch.empty((self.top_k, 5120), dtype=torch.bfloat16, device=device)
            self._h_gu_mid_buf = torch.empty((self.top_k, 64), dtype=torch.float32, device=device)
            self._z_hrr_buf = torch.empty((self.top_k, 4608), dtype=torch.bfloat16, device=device)
            self._z_gu_hlr_buf = torch.empty((self.top_k, 4608), dtype=torch.bfloat16, device=device)
            self._act_hrr_buf = torch.empty((self.top_k, 2304), dtype=torch.bfloat16, device=device)
            self._h_d_mid_buf = torch.empty((self.top_k, 64), dtype=torch.float32, device=device)
            self._y_hrr_buf = torch.empty((self.top_k, 5120), dtype=torch.bfloat16, device=device)
            self._y_routed_buf = torch.empty((1, 5120), dtype=torch.float32, device=device)
            self._y_final_buf = torch.empty((1, 5120), dtype=torch.bfloat16, device=device)

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

        self.pinned_slots = 0
        with torch.no_grad():
            if self.use_fp4_slot_cache and self.num_hot_mxfp4_experts > 0:
                n_pin = min(self.hot_cache_size, self.num_hot_mxfp4_experts)
                for slot_i in range(n_pin):
                    eid = int(self.hot_expert_ids_list[slot_i])
                    h_idx = self.hot_expert_to_idx[eid]
                    self._load_hot_mxfp4_into_fp4_slot(h_idx, slot_i)
                    self.expert_to_slot[eid] = slot_i
                    self.expert_to_slot_gpu[eid] = slot_i
                    self.slot_to_expert[slot_i] = eid
                self.pinned_slots = n_pin
                self.ring_cursor = 0
            else:
                rw_f = self.router_w.float()
                _, _, vh_rt = torch.linalg.svd(rw_f, full_matrices=False)
                u_ctx = vh_rt[:32, :].T
                prior_scores = torch.linalg.norm(rw_f @ u_ctx, dim=-1) + self.e_score_bias
                n_pin = self.hot_cache_size if self.use_fp4_slot_cache else min(8, max(0, self.hot_cache_size - self.top_k))
                _, top_hub_ids = torch.topk(prior_scores, k=n_pin)
                for slot_i, eid in enumerate(top_hub_ids.tolist()):
                    if self.use_fp4_slot_cache:
                        g_idx = int(eid) // 8
                        local_m = int(eid) % 8
                        self.groups[g_idx].mmap_zero_ram = True
                        self.groups[g_idx].mmap_handle = self._mmap_handle
                    self._ensure_expert_slot(int(eid))
                    if self.use_fp4_slot_cache:
                        self.groups[g_idx].mmap_zero_ram = False
                        self.groups[g_idx].mmap_handle = None
                self.pinned_slots = n_pin
                if self.ring_cursor < self.pinned_slots:
                    self.ring_cursor = self.pinned_slots % self.hot_cache_size

            if self.use_fp4_slot_cache:
                self._mmap_handle = None
                self._hot_handle = None
                if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
                    for fpath in (self.shard_path_str, str(self._hot_shard_cand)):
                        try:
                            if os.path.exists(fpath):
                                fd = os.open(fpath, os.O_RDONLY)
                                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                                os.close(fd)
                        except OSError:
                            pass

            if layer_idx == 2 and self.use_fp4_slot_cache and _HAS_TRITON and self.device.type == "cuda":
                _ = self.forward(torch.zeros((1, 5120), dtype=torch.bfloat16, device=self.device))
            self.cache_hits = 0
            self.cache_misses = 0

        if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
            try:
                fd = os.open(self.shard_path_str, os.O_RDONLY)
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
            elif hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
                try:
                    fd_h = os.open(str(self._hot_shard_cand), os.O_RDONLY)
                    os.posix_fadvise(fd_h, 0, 0, os.POSIX_FADV_DONTNEED)
                    os.close(fd_h)
                except OSError:
                    pass
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

        # Zero-sync 100% GPU-Resident Fused Triton Decode Path for n_toks == 1
        if n_toks == 1 and self.use_fp4_slot_cache and _HAS_TRITON and x.is_cuda:
            raw_logits = F.linear(x, self.router_w)
            bias_vec = self.e_score_bias
            if image_mask is not None and self.bias_vl is not None and image_mask.numel() == 1:
                bias_vec = torch.where(image_mask.view(1), self.bias_vl, self.e_score_bias)

            k_act = self.decode_top_k
            _triton_dsv41_router_top6_kernel[(1,)](
                raw_logits,
                bias_vec,
                self._top_ids_buf,
                self._active_gates_buf,
                float(self.routed_scaling_factor),
                K_ACT=k_act,
                num_warps=4,
            )

            keys = self._keys_384
            assert keys is not None
            n_active = k_act + 1

            _triton_dsv41_fused_pre_unbind_healer_v_kernel[(44, k_act)](
                x,
                self.healer_v_gu,
                self._top_ids_buf,
                self.expert_to_slot_gpu,
                keys["gu_c_perm"],
                keys["gu_c_s1_perm"],
                keys["gu_c_s2"],
                self._x_hrr_buf,
                self._h_gu_mid_buf,
                D_IN=5120,
                N_UNBIND_BLKS=40,
                X_STRIDE_E=0,
                BLOCK_K=256,
                num_warps=4,
            )
            _triton_dsv41_resident_gu_kernel[(2304 // 8, n_active)](
                x,
                self._x_hrr_buf,
                self.cache_gu_u8,
                self.cache_gu_s_bf16,
                self.group_gu_u8,
                self.group_gu_s_fp8,
                self.cold_scale_gu_f32,
                keys["gu_r_s2"],
                self.healer_u_gu,
                self._h_gu_mid_buf,
                self._top_ids_buf,
                self.expert_to_slot_gpu,
                self._active_gates_buf,
                self._z_hrr_buf,
                self._z_gu_hlr_buf,
                self._act_inter_buf,
                float(self.swiglu_limit),
                TOP_K=k_act,
                SHARED_SLOT=self.shared_slot,
                N_INTER=2304,
                N_BLOCKS_K=160,
                BLOCK_N=8,
                sub_blocks_k=32,
                num_warps=8,
            )
            _triton_dsv41_fwht_post_kernel[(36, k_act)](
                self._z_hrr_buf,
                self._top_ids_buf,
                self.expert_to_slot_gpu,
                D_OUT=4608,
                num_warps=2,
            )
            _triton_dsv41_cold_swiglu_kernel[(18, k_act)](
                self._z_hrr_buf,
                self._z_gu_hlr_buf,
                self._top_ids_buf,
                self.expert_to_slot_gpu,
                self._active_gates_buf,
                keys["gu_r_inv"],
                keys["gu_r_s1"],
                self._act_inter_buf,
                float(self.swiglu_limit),
                N_INTER=2304,
                num_warps=2,
            )
            _triton_dsv41_fused_pre_unbind_healer_v_kernel[(22, k_act)](
                self._act_inter_buf,
                self.healer_v_d,
                self._top_ids_buf,
                self.expert_to_slot_gpu,
                keys["dn_c_perm"],
                keys["dn_c_s1_perm"],
                keys["dn_c_s2"],
                self._act_hrr_buf,
                self._h_d_mid_buf,
                D_IN=2304,
                N_UNBIND_BLKS=18,
                X_STRIDE_E=2304,
                BLOCK_K=256,
                num_warps=4,
            )
            _triton_dsv41_resident_down_kernel[(5120 // 8, n_active)](
                self._act_inter_buf,
                self._act_hrr_buf,
                self.cache_d_u8,
                self.cache_d_s_bf16,
                self.group_d_u8,
                self.group_d_s_fp8,
                self.cold_scale_d_f32,
                keys["dn_r_s2"],
                self.healer_u_d,
                self._h_d_mid_buf,
                self._top_ids_buf,
                self.expert_to_slot_gpu,
                self._y_hrr_buf,
                self._y_partial_buf,
                TOP_K=k_act,
                SHARED_SLOT=self.shared_slot,
                D_OUT=5120,
                N_BLOCKS_K=72,
                BLOCK_D=8,
                sub_blocks_k=8,
                num_warps=8,
            )
            _triton_dsv41_fwht_post_kernel[(40, k_act)](
                self._y_hrr_buf,
                self._top_ids_buf,
                self.expert_to_slot_gpu,
                D_OUT=5120,
                num_warps=2,
            )
            _triton_dsv41_reduce_down_kernel[(40,)](
                self._y_hrr_buf,
                self._y_partial_buf,
                self._top_ids_buf,
                self.expert_to_slot_gpu,
                keys["dn_r_inv"],
                keys["dn_r_s1"],
                self._y_final_buf,
                TOP_K=k_act,
                D_OUT=5120,
                num_warps=2,
            )
            out_y = self._y_final_buf.view(orig_shape)
            return out_y if orig_dtype == torch.bfloat16 else out_y.to(orig_dtype)

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
