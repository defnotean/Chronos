#!/usr/bin/env python3
"""
Chronos-1T x PseudoBrain (pb27) — Packed Holographic Sparse MoE Block (`v6 Zero-Con`).

Implements real byte-packed `.safetensors` serialization and GB10 GPU execution for:
1. 8:1 NVFP4 Holographic Superposition Banks (`S_gu`, `S_down`) with deterministic
   1D Permuted-Hadamard Dual Unbinding keys generated on-the-fly from expert ID.
2. Coupled SwiGLU + Data-Free Router-Manifold Rank-32 FP8 SVD Healers (`float8_e4m3fn`),
   with Shared Input Basis `V_gu` across `gate_proj` and `up_proj`.
3. Real Bit-Packed Isotropic Gaussian Tail Lattice:
   - 2-Bit Lloyd-Max Gaussian Lattice (`4 weights per uint8 byte` + `float8_e4m3fn` scale per 64 elems)
   - 1-Bit Rademacher Phase Lattice (`8 weights per uint8 byte` + `float8_e4m3fn` scale per 64 elems)
4. Dynamic 32-Expert Sliding-Window EMA + Global Router Prior Hot Cache.
"""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F

NVFP4_LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
POS_T = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
LLOYD_MAX_2BIT_LUT = (-1.510417, -0.452780, 0.452780, 1.510417)


def build_hadamard_128(device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    h = torch.tensor([[1.0]], device=device, dtype=torch.float32)
    for _ in range(7):
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return (h / math.sqrt(128.0)).to(dtype)


def pack_nvfp4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Packs a 2D float32 weight tensor (rows, cols) into authentic NVFP4 tensors:
    - weight_packed: uint8 of shape (rows, cols // 2) (2 E2M1 nibbles per byte)
    - weight_scale: float8_e4m3fn of shape (rows, cols // 16)
    - weight_scale_2: float32 scalar
    """
    rows, cols = w.shape
    device = w.device
    w_amax = w.abs().max().clamp_min(1e-12)
    scale_2 = (w_amax / (6.0 * 448.0)).to(torch.float32)

    w_blk = w.view(rows, cols // 16, 16)
    blk_max = w_blk.abs().amax(dim=-1)
    raw_s = (blk_max / (6.0 * scale_2)).clamp(min=1e-6, max=448.0)
    scale_fp8 = raw_s.to(torch.float8_e4m3fn)
    eff_s = (scale_fp8.float() * scale_2).unsqueeze(-1).clamp_min(1e-12)

    w_norm = w_blk / eff_s
    signs = (w_norm < 0).to(torch.uint8) << 3
    abs_v = w_norm.abs()

    # Nearest E2M1 magnitude index in {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
    idx = torch.zeros_like(abs_v, dtype=torch.uint8)
    thresholds = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
    for ti, th in enumerate(thresholds, start=1):
        idx = torch.where(abs_v >= th, torch.tensor(ti, device=device, dtype=torch.uint8), idx)

    nibbles = (signs | idx).view(rows, cols)
    low = nibbles[:, 0::2]
    high = nibbles[:, 1::2]
    packed = (low | (high << 4)).to(torch.uint8)
    return packed, scale_fp8, scale_2


def unpack_nvfp4(
    packed: torch.Tensor,
    scale_fp8: torch.Tensor,
    scale_2: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    device = packed.device
    lut = torch.tensor(NVFP4_LUT, dtype=torch.float32, device=device)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    unpacked = torch.stack([low, high], dim=-1).view(packed.shape[0], -1)
    vals = lut[unpacked.long()].view(packed.shape[0], -1, 16)
    scales = scale_fp8.to(torch.float32).unsqueeze(-1) * scale_2.to(device=device, dtype=torch.float32)
    return (vals * scales).view(packed.shape[0], -1).to(dtype)


def pack_2bit_lloyd_max(res_tail: torch.Tensor, block_size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """Packs a residual tensor into real 2-bit uint8 (4 elements per byte) + float8_e4m3fn block scales."""
    rows, cols = res_tail.shape
    blocks = res_tail.reshape(-1, block_size)
    sigma = blocks.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-6, max=448.0)
    sigma_fp8 = sigma.to(torch.float8_e4m3fn)
    z = (blocks / sigma_fp8.float().clamp_min(1e-8)).reshape(rows, cols)

    # 2-bit codes in {0, 1, 2, 3}
    codes = torch.zeros_like(z, dtype=torch.uint8)
    codes = torch.where(z >= -0.981599, torch.ones_like(codes), codes)
    codes = torch.where(z >= 0.0, torch.full_like(codes, 2), codes)
    codes = torch.where(z >= 0.981599, torch.full_like(codes, 3), codes)

    # Pack 4 2-bit codes per uint8 byte along cols
    c0 = codes[:, 0::4]
    c1 = codes[:, 1::4] << 2
    c2 = codes[:, 2::4] << 4
    c3 = codes[:, 3::4] << 6
    packed_u8 = (c0 | c1 | c2 | c3).to(torch.uint8)
    return packed_u8, sigma_fp8.view(rows, cols // block_size)


def unpack_2bit_lloyd_max(
    packed_u8: torch.Tensor,
    sigma_fp8: torch.Tensor,
    block_size: int = 32,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    device = packed_u8.device
    rows = packed_u8.shape[0]
    cols = packed_u8.shape[1] * 4
    lut = torch.tensor(LLOYD_MAX_2BIT_LUT, dtype=torch.float32, device=device)

    c0 = packed_u8 & 0x03
    c1 = (packed_u8 >> 2) & 0x03
    c2 = (packed_u8 >> 4) & 0x03
    c3 = (packed_u8 >> 6) & 0x03
    codes = torch.stack([c0, c1, c2, c3], dim=-1).view(rows, cols)
    vals = lut[codes.long()].view(rows, cols // block_size, block_size)
    scales = sigma_fp8.float().unsqueeze(-1)
    return (vals * scales).view(rows, cols).to(dtype)


def pack_ternary_base3(res_tail: torch.Tensor, block_size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """Packs 5 optimal Gaussian ternary trits ({-1.224, 0, +1.224} * sigma) into 1 uint8 byte (3^5 = 243 <= 255)
    for 1.60 bits/weight (0.20 B/elem) + float8_e4m3fn scale per 32 elements.
    """
    rows, cols = res_tail.shape
    blocks = res_tail.reshape(-1, block_size)
    sigma = blocks.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-6, max=448.0)
    sigma_fp8 = sigma.to(torch.float8_e4m3fn)
    z = (blocks / sigma_fp8.float().clamp_min(1e-8)).reshape(rows, cols)

    # Trits in {0, 1, 2} corresponding to {-1.2240, 0.0, +1.2240}
    trits = torch.ones_like(z, dtype=torch.uint8)
    trits = torch.where(z <= -0.6120, torch.zeros_like(trits), trits)
    trits = torch.where(z >= 0.6120, torch.full_like(trits, 2), trits)

    t0 = trits[:, 0::5]
    t1 = trits[:, 1::5] * 3
    t2 = trits[:, 2::5] * 9
    t3 = trits[:, 3::5] * 27
    t4 = trits[:, 4::5] * 81
    packed_u8 = (t0 + t1 + t2 + t3 + t4).to(torch.uint8)
    return packed_u8, sigma_fp8.view(rows, cols // block_size)


def unpack_ternary_base3(
    packed_u8: torch.Tensor,
    sigma_fp8: torch.Tensor,
    block_size: int = 32,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    rows = packed_u8.shape[0]
    cols = packed_u8.shape[1] * 5
    rem = packed_u8.to(torch.int32)
    trits_list = []
    for _ in range(5):
        trits_list.append(rem % 3)
        rem = rem // 3
    trits = torch.stack(trits_list, dim=-1).view(rows, cols)
    vals = (trits.float() - 1.0) * 1.2240
    scales = sigma_fp8.float().unsqueeze(-1)
    return (vals.view(rows, cols // block_size, block_size) * scales).view(rows, cols).to(dtype)


def pack_1bit_rademacher(res_tail: torch.Tensor, block_size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """Packs a residual tensor into real 1-bit uint8 (8 elements per byte) + float8_e4m3fn block scales."""
    rows, cols = res_tail.shape
    blocks = res_tail.reshape(-1, block_size)
    sigma = (blocks.pow(2).mean(dim=-1, keepdim=True).sqrt() * 0.79788456).clamp(min=1e-6, max=448.0)
    sigma_fp8 = sigma.to(torch.float8_e4m3fn)
    bits = (res_tail >= 0.0).to(torch.uint8)

    packed_u8 = torch.zeros((rows, cols // 8), dtype=torch.uint8, device=res_tail.device)
    for b in range(8):
        packed_u8 |= bits[:, b::8] << b
    return packed_u8, sigma_fp8.view(rows, cols // block_size)


def unpack_1bit_rademacher(
    packed_u8: torch.Tensor,
    sigma_fp8: torch.Tensor,
    block_size: int = 64,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    rows = packed_u8.shape[0]
    cols = packed_u8.shape[1] * 8
    bits_list = [((packed_u8 >> b) & 0x01) for b in range(8)]
    bits = torch.stack(bits_list, dim=-1).view(rows, cols)
    signs = bits.float() * 2.0 - 1.0
    scales = sigma_fp8.float().unsqueeze(-1)
    return (signs.view(rows, cols // block_size, block_size) * scales).view(rows, cols).to(dtype)


def pack_sparse_2bit_lloyd_max(
    res_tail: torch.Tensor,
    col_importance: torch.Tensor,
    keep_ratio: float = 0.50,
    block_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Packs only the top `keep_ratio` functional-energy 64-element micro-blocks into 2-bit uint8.
    Returns:
      - mask_u8: uint8 bitmap of shape (num_total_blocks // 8,)
      - active_packed_u8: uint8 of shape (num_kept_blocks, block_size // 4)
      - active_sigma_fp8: float8_e4m3fn of shape (num_kept_blocks,)
    """
    rows, cols = res_tail.shape
    num_blocks = (rows * cols) // block_size
    n_keep = max(1, int(round(num_blocks * keep_ratio)))

    blocks = res_tail.view(rows, cols // block_size, block_size)
    c_imp = col_importance.view(1, cols // block_size, block_size)
    blk_energy = (blocks * c_imp).pow(2).mean(dim=-1).flatten()

    _, top_idx = torch.topk(blk_energy, k=n_keep, largest=True, sorted=True)
    top_idx_sorted, _ = torch.sort(top_idx)

    mask_bool = torch.zeros(num_blocks, dtype=torch.uint8, device=res_tail.device)
    mask_bool[top_idx_sorted] = 1
    mask_u8 = torch.zeros(num_blocks // 8, dtype=torch.uint8, device=res_tail.device)
    for b in range(8):
        mask_u8 |= mask_bool[b::8] << b

    flat_blocks = res_tail.reshape(num_blocks, block_size)
    active_blocks = flat_blocks[top_idx_sorted]  # (n_keep, 64)

    sigma = active_blocks.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-6, max=448.0)
    active_sigma_fp8 = sigma.squeeze(-1).to(torch.float8_e4m3fn)
    z = active_blocks / active_sigma_fp8.float().unsqueeze(-1).clamp_min(1e-8)

    codes = torch.zeros_like(z, dtype=torch.uint8)
    codes = torch.where(z >= -0.981599, torch.ones_like(codes), codes)
    codes = torch.where(z >= 0.0, torch.full_like(codes, 2), codes)
    codes = torch.where(z >= 0.981599, torch.full_like(codes, 3), codes)

    c0 = codes[:, 0::4]
    c1 = codes[:, 1::4] << 2
    c2 = codes[:, 2::4] << 4
    c3 = codes[:, 3::4] << 6
    active_packed_u8 = (c0 | c1 | c2 | c3).to(torch.uint8)
    return mask_u8, active_packed_u8, active_sigma_fp8


def unpack_sparse_2bit_lloyd_max(
    mask_u8: torch.Tensor,
    active_packed_u8: torch.Tensor,
    active_sigma_fp8: torch.Tensor,
    rows: int,
    cols: int,
    block_size: int = 64,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    device = active_packed_u8.device
    num_blocks = (rows * cols) // block_size
    bits_list = [((mask_u8 >> b) & 0x01) for b in range(8)]
    mask_bool = torch.stack(bits_list, dim=-1).view(-1).bool()

    lut = torch.tensor(LLOYD_MAX_2BIT_LUT, dtype=torch.float32, device=device)
    c0 = active_packed_u8 & 0x03
    c1 = (active_packed_u8 >> 2) & 0x03
    c2 = (active_packed_u8 >> 4) & 0x03
    c3 = (active_packed_u8 >> 6) & 0x03
    codes = torch.stack([c0, c1, c2, c3], dim=-1).view(-1, block_size)
    active_vals = lut[codes.long()] * active_sigma_fp8.float().unsqueeze(-1)

    out_blocks = torch.zeros((num_blocks, block_size), dtype=torch.float32, device=device)
    out_blocks[mask_bool] = active_vals
    return out_blocks.view(rows, cols).to(dtype)


def generate_expert_1d_keys(
    rows: int,
    cols: int,
    expert_id: int,
    seed_base: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed_base + (expert_id + 1) * 104729)
    r_s1 = torch.randint(0, 2, (rows,), generator=gen, device=device, dtype=torch.int8).float() * 2.0 - 1.0
    r_perm = torch.randperm(rows, generator=gen, device=device)
    r_inv = torch.argsort(r_perm)
    r_s2 = torch.randint(0, 2, (rows,), generator=gen, device=device, dtype=torch.int8).float() * 2.0 - 1.0
    c_s1 = torch.randint(0, 2, (cols,), generator=gen, device=device, dtype=torch.int8).float() * 2.0 - 1.0
    c_perm = torch.randperm(cols, generator=gen, device=device)
    c_inv = torch.argsort(c_perm)
    c_s2 = torch.randint(0, 2, (cols,), generator=gen, device=device, dtype=torch.int8).float() * 2.0 - 1.0
    return r_s1, r_perm, r_inv, r_s2, c_s1, c_perm, c_inv, c_s2


def apply_1d_permuted_hadamard_rows(
    mat: torch.Tensor,
    s1: torch.Tensor,
    perm: torch.Tensor,
    s2: torch.Tensor,
    h128: torch.Tensor,
) -> torch.Tensor:
    rows, cols = mat.shape
    x = (mat * s1.unsqueeze(0))[:, perm]
    x = (x.view(rows, -1, 128) @ h128).view(rows, cols)
    return x * s2.unsqueeze(0)


def unbind_1d_permuted_hadamard_rows(
    mat: torch.Tensor,
    s1: torch.Tensor,
    inv_perm: torch.Tensor,
    s2: torch.Tensor,
    h128: torch.Tensor,
) -> torch.Tensor:
    rows, cols = mat.shape
    x = mat * s2.unsqueeze(0)
    x = (x.view(rows, -1, 128) @ h128).view(rows, cols)
    return x[:, inv_perm] * s1.unsqueeze(0)


def fold_2d_permuted_hadamard(
    w: torch.Tensor,
    r_s1: torch.Tensor,
    r_perm: torch.Tensor,
    r_s2: torch.Tensor,
    c_s1: torch.Tensor,
    c_perm: torch.Tensor,
    c_s2: torch.Tensor,
    h128: torch.Tensor,
) -> torch.Tensor:
    tmp = apply_1d_permuted_hadamard_rows(w, c_s1, c_perm, c_s2, h128)
    return apply_1d_permuted_hadamard_rows(tmp.T.contiguous(), r_s1, r_perm, r_s2, h128).T.contiguous()


def unbind_2d_permuted_hadamard(
    s: torch.Tensor,
    r_s1: torch.Tensor,
    r_inv: torch.Tensor,
    r_s2: torch.Tensor,
    c_s1: torch.Tensor,
    c_inv: torch.Tensor,
    c_s2: torch.Tensor,
    h128: torch.Tensor,
) -> torch.Tensor:
    tmp = unbind_1d_permuted_hadamard_rows(s.T.contiguous(), r_s1, r_inv, r_s2, h128).T.contiguous()
    return unbind_1d_permuted_hadamard_rows(tmp, c_s1, c_inv, c_s2, h128)


class ChronosHolographicExpertGroup(nn.Module):
    """Stores a folded group of K=8 routed MoE experts in authentic byte-packed tensors
    and reconstructs or executes them on the fly in GB10 SRAM.
    """

    def __init__(self, group_idx: int, tensors: dict[str, torch.Tensor], device: torch.device) -> None:
        super().__init__()
        self.group_idx = group_idx
        self.device = device
        self.h128 = build_hadamard_128(device)

        # Packed 8:1 NVFP4 Holographic Banks
        self.s_gu_packed = tensors["s_gu.weight"].to(device)
        self.s_gu_scale = tensors["s_gu.weight_scale"].to(device)
        self.s_gu_scale2 = tensors["s_gu.weight_scale_2"].to(device)
        self.beta_gu = tensors["s_gu.beta"].to(device)

        self.s_down_packed = tensors["s_down.weight"].to(device)
        self.s_down_scale = tensors["s_down.weight_scale"].to(device)
        self.s_down_scale2 = tensors["s_down.weight_scale_2"].to(device)
        self.beta_down = tensors["s_down.beta"].to(device)

        # Coupled FP8 Rank-32 SVD Healers
        self.u_gu = tensors["healer.u_gu"].to(device)        # (K, 1280, r) float8_e4m3fn
        self.v_gu = tensors["healer.v_gu"].to(device)        # (K, r, 2560) float8_e4m3fn
        self.u_down = tensors["healer.u_down"].to(device)    # (K, 2560, r) float8_e4m3fn
        self.v_down = tensors["healer.v_down"].to(device)    # (K, r, 640)  float8_e4m3fn

        # Bit-packed Isotropic Gaussian Tail Lattices
        self.gu_tail_bits = int(tensors.get("tail.gu_bits", torch.tensor(1)).item())
        self.gu_tail_packed = tensors["tail.gu_packed"].to(device)
        self.gu_tail_scale = tensors["tail.gu_scale"].to(device)
        self.gu_tail_mask = tensors["tail.gu_mask"].to(device) if "tail.gu_mask" in tensors else None

        self.down_tail_bits = int(tensors.get("tail.down_bits", torch.tensor(2)).item())
        self.down_tail_packed = tensors["tail.down_packed"].to(device)
        self.down_tail_scale = tensors["tail.down_scale"].to(device)
        self.down_tail_mask = tensors["tail.down_mask"].to(device) if "tail.down_mask" in tensors else None
        self._s_gu_deq: torch.Tensor | None = None
        self._s_down_deq: torch.Tensor | None = None

    @torch.no_grad()
    def reconstruct_expert(self, local_m: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reconstructs (gate_w, up_w, down_w) for expert `local_m` in 0..K-1 from the packed tensors."""
        if self._s_gu_deq is None:
            self._s_gu_deq = unpack_nvfp4(self.s_gu_packed, self.s_gu_scale, self.s_gu_scale2)
            self._s_down_deq = unpack_nvfp4(self.s_down_packed, self.s_down_scale, self.s_down_scale2)
        s_gu = self._s_gu_deq
        s_down = self._s_down_deq

        r_s1, _, r_inv, r_s2, c_s1, _, c_inv, c_s2 = generate_expert_1d_keys(
            1280, 2560, local_m, seed_base=101 + self.group_idx * 8, device=self.device
        )
        h_gu = unbind_2d_permuted_hadamard(s_gu, r_s1, r_inv, r_s2, c_s1, c_inv, c_s2, self.h128)
        healer_gu = self.u_gu[local_m].float() @ self.v_gu[local_m].float()

        if self.gu_tail_bits == 20 and self.gu_tail_mask is not None:
            tail_gu = unpack_sparse_2bit_lloyd_max(
                self.gu_tail_mask[local_m], self.gu_tail_packed[local_m], self.gu_tail_scale[local_m], 1280, 2560
            )
        elif self.gu_tail_bits == 16:
            tail_gu = unpack_ternary_base3(self.gu_tail_packed[local_m], self.gu_tail_scale[local_m])
        elif self.gu_tail_bits == 2:
            tail_gu = unpack_2bit_lloyd_max(self.gu_tail_packed[local_m], self.gu_tail_scale[local_m])
        elif self.gu_tail_bits == 1:
            tail_gu = unpack_1bit_rademacher(self.gu_tail_packed[local_m], self.gu_tail_scale[local_m])
        else:
            tail_gu = torch.zeros_like(h_gu)

        gu_hat = float(self.beta_gu[local_m].item()) * h_gu + healer_gu + tail_gu
        gate_hat = gu_hat[:640, :]
        up_hat = gu_hat[640:, :]

        dr_s1, _, dr_inv, dr_s2, dc_s1, _, dc_inv, dc_s2 = generate_expert_1d_keys(
            2560, 640, local_m, seed_base=303 + self.group_idx * 8, device=self.device
        )
        h_down = unbind_2d_permuted_hadamard(s_down, dr_s1, dr_inv, dr_s2, dc_s1, dc_inv, dc_s2, self.h128)
        healer_down = self.u_down[local_m].float() @ self.v_down[local_m].float()

        if self.down_tail_bits == 20 and self.down_tail_mask is not None:
            tail_down = unpack_sparse_2bit_lloyd_max(
                self.down_tail_mask[local_m], self.down_tail_packed[local_m], self.down_tail_scale[local_m], 2560, 640
            )
        elif self.down_tail_bits == 16:
            tail_down = unpack_ternary_base3(self.down_tail_packed[local_m], self.down_tail_scale[local_m])
        elif self.down_tail_bits == 2:
            tail_down = unpack_2bit_lloyd_max(self.down_tail_packed[local_m], self.down_tail_scale[local_m])
        elif self.down_tail_bits == 1:
            tail_down = unpack_1bit_rademacher(self.down_tail_packed[local_m], self.down_tail_scale[local_m])
        else:
            tail_down = torch.zeros_like(h_down)

        down_hat = float(self.beta_down[local_m].item()) * h_down + healer_down + tail_down
        return gate_hat, up_hat, down_hat
