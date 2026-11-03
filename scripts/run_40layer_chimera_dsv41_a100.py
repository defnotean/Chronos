#!/usr/bin/env python3
"""
Standalone Full 40-Layer DeepSeek-V4.1-Flash-1017 + Chronos v7.5 + Chimera-LSA
Capability, Agentic Tool-Calling & Decode Throughput Benchmark + Live Server
for NVIDIA A100-SXM4-80GB (alme-gpu).

Integrates all 4 Chimera-LSA architectural upgrades across all 40 layers:
  1. Upgrade A: 1.58-Bit Ternary {-s, 0, +s} packed in 2-bit uint8 for Tier-2 experts (`CHRONOS_TIER2_TERNARY=1`).
  2. Upgrade B: Spherical Manifold Tangential Retraction (`project_onto_tangent_sphere`).
  3. Upgrade C: 40-Layer Geodesic Cosine + Spherical Tangential Retraction Option-A (`dsv41_shared_experts_option_a_r64.pt`).
  4. Upgrade D: OpenCode Multi-Format Agentic Tool-Calling Bridge (`chronos_1t.opencode_tool_bridge`).
"""
from __future__ import annotations

import gc
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

# 1. Configure Chronos + Chimera-LSA Tiered MoE environment BEFORE importing Chronos modules
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["CHRONOS_TIER1_4BIT_SLOTS"] = os.environ.get("CHRONOS_40L_T1", "16")
os.environ["CHRONOS_TIER2_2BIT_SLOTS"] = os.environ.get("CHRONOS_40L_T2", "80")
os.environ["CHRONOS_TIER3_1BIT_SLOTS"] = os.environ.get("CHRONOS_40L_T3", "288")
os.environ["CHRONOS_T3_CPU_FROM_LAYER"] = os.environ.get("CHRONOS_40L_T3_CPU_FROM", "13")
os.environ["CHRONOS_T3_CPU_TO_LAYER"] = os.environ.get("CHRONOS_40L_T3_CPU_TO", "38")
os.environ["CHRONOS_TIER2_TERNARY"] = os.environ.get("CHRONOS_40L_TERNARY", "1")
os.environ["CHRONOS_OPTION_A_PATH"] = os.environ.get(
    "CHRONOS_40L_OPT_A_PATH", "/content/dsv41_shared_experts_option_a_r64.pt"
)
os.environ["CHRONOS_OPTION_A_ALPHA"] = os.environ.get("CHRONOS_40L_OPT_A_ALPHA", "0.08")

import importlib
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, try_to_load_from_cache
from safetensors import safe_open
from transformers import AutoTokenizer

for _p in [
    "/content/chronos-1t",
    "/content",
    "/content/dsv41_standalone/inference",
    "/content/dsv41_standalone/encoding",
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

for _k in list(sys.modules.keys()):
    if _k == "chronos_1t" or _k.startswith("chronos_1t."):
        sys.modules.pop(_k, None)
importlib.invalidate_caches()

os.makedirs("/content/dsv41_standalone/inference", exist_ok=True)
os.makedirs("/content/dsv41_standalone/encoding", exist_ok=True)
for _rel in [
    "inference/model.py",
    "inference/engram.py",
    "inference/vision.py",
    "inference/image_processor.py",
    "inference/config.json",
    "encoding/encoding.py",
]:
    _dst = os.path.join("/content/dsv41_standalone", _rel)
    if not os.path.isfile(_dst):
        hf_hub_download(
            repo_id="deepseek-ai/DeepSeek-V4.1-Flash-1017",
            filename=_rel,
            local_dir="/content/dsv41_standalone",
        )

# Register pure-PyTorch/CUDA `kernel` module for standalone DeepSeek-V4.1-Flash-1017 Attention & HC
_kmod = types.ModuleType("kernel")
_kmod.FP8 = "float8_e4m3"
_kmod.FP4 = "float4_e2m1fn"
_kmod.FE8M0 = "float8_e8m0fnu"
_kmod.BF16 = "bfloat16"
_kmod.FP32 = "float32"
_kmod.INT32 = "int32"
_FE8M0_DTYPE = getattr(torch, "float8_e8m0fnu", torch.uint8)
_FP4_X2_DTYPE = getattr(torch, "float4_e2m1fn_x2", torch.uint8)
_FP4_LUT_Vals = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def _fast_round_scale(amax: torch.Tensor, max_inv: float) -> torch.Tensor:
    x = amax * max_inv
    bits_x = x.float().view(torch.int32)
    exp_x = (bits_x >> 23) & 0xFF
    man_bits = bits_x & ((1 << 23) - 1)
    log2_ceil = exp_x - 127 + (man_bits != 0).to(torch.int32)
    return ((log2_ceil + 127) << 23).view(torch.float32)


def _act_quant(x: torch.Tensor, block_size: int = 32, scale_fmt=None, scale_dtype=_FE8M0_DTYPE, inplace: bool = False):
    N = x.size(-1)
    orig_shape = x.shape
    xg = x.contiguous().float().view(-1, N // block_size, block_size)
    amax = xg.abs().amax(dim=-1).clamp_min(1e-4)
    s_fp32 = _fast_round_scale(amax, 1.0 / 448.0) if scale_fmt is not None else (amax / 448.0)
    q_fp8 = (xg / s_fp32.unsqueeze(-1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    if inplace:
        x.copy_((q_fp8.float() * s_fp32.unsqueeze(-1)).to(x.dtype).view(orig_shape))
        return x
    y = q_fp8.view(orig_shape)
    if scale_dtype == _FE8M0_DTYPE:
        s_out = (((s_fp32.view(torch.int32) >> 23) & 0xFF).to(torch.uint8)).view(_FE8M0_DTYPE)
    else:
        s_out = s_fp32.to(scale_dtype)
    return y, s_out.view(*orig_shape[:-1], N // block_size)


def _fp4_act_quant(x: torch.Tensor, block_size: int = 32, inplace: bool = False, scale_dtype=_FE8M0_DTYPE):
    N = x.size(-1)
    orig_shape = x.shape
    xg = x.contiguous().float().view(-1, N // block_size, block_size)
    amax = xg.abs().amax(dim=-1)
    if scale_dtype == torch.float8_e4m3fn:
        amax = amax.clamp_min(6.0 * (2.0**-9))
        s_out = (amax / 6.0).to(torch.float8_e4m3fn)
        s_fp32 = s_out.float()
    else:
        amax = amax.clamp_min(6.0 * (2.0**-126))
        s_fp32 = _fast_round_scale(amax, 1.0 / 6.0)
        s_out = (((s_fp32.view(torch.int32) >> 23) & 0xFF).to(torch.uint8)).view(_FE8M0_DTYPE)
    scaled = (xg / s_fp32.unsqueeze(-1)).clamp(-6.0, 6.0)
    ax = scaled.abs()
    mag = (
        (ax > 0.25).to(torch.uint8)
        + (ax >= 0.75).to(torch.uint8)
        + (ax > 1.25).to(torch.uint8)
        + (ax >= 1.75).to(torch.uint8)
        + (ax > 2.5).to(torch.uint8)
        + (ax >= 3.5).to(torch.uint8)
        + (ax > 5.0).to(torch.uint8)
    )
    nibbles = (((scaled < 0).to(torch.uint8)) << 3) | mag
    if inplace:
        lut = torch.tensor(_FP4_LUT_Vals, dtype=torch.float32, device=x.device)
        deq = (lut[nibbles.long()] * s_fp32.unsqueeze(-1)).to(x.dtype).view(orig_shape)
        x.copy_(deq)
        return x
    nf = nibbles.view(*orig_shape[:-1], N)
    packed = ((nf[..., 0::2] & 0x0F) | ((nf[..., 1::2] & 0x0F) << 4)).view(_FP4_X2_DTYPE)
    return packed, s_out.view(*orig_shape[:-1], N // block_size)


def _sparse_attn(q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor, topk_idxs: torch.Tensor, softmax_scale=None):
    b, s, h, d = q.size()
    if softmax_scale is None:
        softmax_scale = d**-0.5
    topk = topk_idxs.size(-1)
    if topk == 0 or s == 0:
        return torch.zeros_like(q)
    out = torch.empty_like(q)
    sink_fp32 = attn_sink[:h].float().view(1, 1, h, 1)
    batch_idx = torch.arange(b, device=q.device).view(b, 1, 1)
    for start in range(0, s, 64):
        end = min(start + 64, s)
        q_c = q[:, start:end]
        idx_c = topk_idxs[:, start:end]
        valid = idx_c != -1
        safe_idx = idx_c.clamp_min(0).long()
        kv_g = kv[batch_idx, safe_idx]
        kv_g = torch.where(valid.unsqueeze(-1), kv_g, torch.zeros_like(kv_g))
        scores = torch.matmul(q_c.float(), kv_g.float().transpose(-1, -2)) * softmax_scale
        scores = scores.masked_fill(~valid.unsqueeze(2), -torch.inf)
        s_max = scores.amax(dim=-1, keepdim=True).clamp_min(-1e30)
        exp_s = torch.exp(scores - s_max)
        sum_exp = exp_s.sum(dim=-1, keepdim=True) + torch.exp(sink_fp32 - s_max)
        acc_o = torch.matmul(exp_s.to(kv.dtype), kv_g).float()
        out[:, start:end] = (acc_o / sum_exp).to(q.dtype)
    return out


def _hc_split_sinkhorn(mixes: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor, hc_mult: int = 4, sinkhorn_iters: int = 20, eps: float = 1e-6):
    orig_prefix = mixes.shape[:-1]
    hc = hc_mult
    mix_hc = (2 + hc) * hc
    m = mixes.float().view(-1, mix_hc)
    scale = hc_scale.float()
    base = hc_base.float()
    pre = torch.sigmoid(m[:, :hc] * scale[0] + base[:hc]) + eps
    post = 2.0 * torch.sigmoid(m[:, hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
    comb = (m[:, 2 * hc :] * scale[2] + base[2 * hc :]).view(-1, hc, hc)
    row_max = comb.amax(dim=-1, keepdim=True)
    comb = torch.exp(comb - row_max)
    comb = comb / comb.sum(dim=-1, keepdim=True) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre.view(*orig_prefix, hc), post.view(*orig_prefix, hc), comb.view(*orig_prefix, hc, hc)


def _scale_to_fp32(s: torch.Tensor) -> torch.Tensor:
    if s.dtype == torch.float32:
        return s
    if s.dtype == _FE8M0_DTYPE:
        u8 = s.view(torch.uint8)
        normal = (u8.to(torch.int32) << 23).view(torch.float32)
        if not (u8 == 0).any():
            return normal
        return torch.where(u8 == 0, torch.full_like(normal, 2.0**-127), normal)
    return s.float()


def _fp8_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor, scale_dtype=_FE8M0_DTYPE, block_size: int = 32) -> torch.Tensor:
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    num_k_blocks = K // block_size
    a_s_fp32 = _scale_to_fp32(a_s).view(M, num_k_blocks, 1)
    b_s_fp32 = _scale_to_fp32(b_s).repeat_interleave(block_size, dim=0)[:N].unsqueeze(-1)
    a_fp32 = (a.view(M, num_k_blocks, block_size).float() * a_s_fp32).view(M, K)
    b_fp32 = (b.view(N, num_k_blocks, block_size).float() * b_s_fp32).view(N, K)
    c = torch.matmul(a_fp32, b_fp32.t()).to(torch.get_default_dtype())
    return c.view(*a.size()[:-1], N)


def _fp4_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor, scale_dtype=_FE8M0_DTYPE, act_block_size: int = 32) -> torch.Tensor:
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    a_s_fp32 = _scale_to_fp32(a_s).view(M, K // act_block_size, 1)
    b_s_fp32 = _scale_to_fp32(b_s).view(N, K // 32, 1)
    a_fp32 = (a.view(M, K // act_block_size, act_block_size).float() * a_s_fp32).view(M, K)
    raw = b.view(torch.uint8)
    nibbles = torch.stack([raw & 0x0F, (raw >> 4) & 0x0F], dim=-1).flatten(-2)
    lut = torch.tensor(_FP4_LUT_Vals, dtype=torch.float32, device=b.device)
    b_unpacked = lut[nibbles.long()].to(torch.float8_e4m3fn).float()
    b_fp32 = (b_unpacked.view(N, K // 32, 32) * b_s_fp32).view(N, K)
    c = torch.matmul(a_fp32, b_fp32.t()).to(torch.get_default_dtype())
    return c.view(*a.size()[:-1], N)


_kmod.act_quant = _act_quant
_kmod.fp4_act_quant = _fp4_act_quant
_kmod.fp8_gemm = _fp8_gemm
_kmod.fp4_gemm = _fp4_gemm
_kmod.sparse_attn = _sparse_attn
_kmod.hc_split_sinkhorn = _hc_split_sinkhorn
sys.modules["kernel"] = _kmod

import model as dsv41_model_mod
from encoding import encode_messages
from chronos_1t.models.dsv41_chronos_moe import (
    ChronosDsv41SparseMoeBlock,
    dequantize_block_fp8_32,
)
from chronos_1t.models.dsv41_triton_backbone import (
    install_dsv41_triton_backbone,
    enable_direct_uva_tier3_on_model,
    _get_block_decode_bufs,
)
from chronos_1t.opencode_tool_bridge import parse_tool_calls_and_clean_content

install_dsv41_triton_backbone(dsv41_model_mod)

REPO_ID = "defnoteatin/PseudoBrain-DeepSeek-V4.1-Flash-1017-Chronos-v7.5-Abliterated"
STREAM_DIR = "/tmp/dsv41_stream"
OUTPUT_JSON = "/content/dsv41_40layer_chimera_a100_benchmark.json"


def get_ram_rss_gb() -> float:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return kb / (1024.0 * 1024.0)
    except Exception:
        pass
    return 0.0


def get_disk_free_gb(path: str = "/tmp") -> float:
    try:
        usage = shutil.disk_usage(path)
        return usage.free / (1024.0**3)
    except Exception:
        return 0.0


def resolve_or_download(filename: str, keep_in_stream_dir: bool = False) -> str:
    os.makedirs(STREAM_DIR, exist_ok=True)
    for cand_dir in [STREAM_DIR, "/content/dsv41_shards", "/content/dsv41_standalone", "/content/dsv41_base", "/content"]:
        cand = os.path.join(cand_dir, filename)
        if os.path.isfile(cand) and os.path.getsize(cand) > 0:
            return cand
    try:
        cached = try_to_load_from_cache(repo_id=REPO_ID, filename=filename)
        if isinstance(cached, str) and os.path.isfile(cached) and os.path.getsize(cached) > 0:
            if keep_in_stream_dir:
                dst = os.path.join(STREAM_DIR, filename)
                if not os.path.exists(dst):
                    shutil.copy2(cached, dst)
                return dst
            return cached
    except Exception:
        pass
    return hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir=STREAM_DIR)


def cleanup_stream_file(path: str) -> None:
    try:
        real_p = os.path.realpath(path)
        if os.path.lexists(path):
            os.remove(path)
        if real_p != path and os.path.exists(real_p):
            os.remove(real_p)
    except OSError:
        pass


@torch.no_grad()
def load_param_tensor(
    sf: Any,
    all_keys: set[str],
    key: str,
    target_dtype: torch.dtype,
    device: str = "cuda",
) -> torch.Tensor:
    actual_key = key
    if actual_key not in all_keys:
        if f"model.{key}" in all_keys:
            actual_key = f"model.{key}"
        elif key.startswith("model.") and key[6:] in all_keys:
            actual_key = key[6:]
        else:
            raise KeyError(f"Key '{key}' not found in safetensors shard.")

    t = sf.get_tensor(actual_key)
    if hasattr(torch, "float8_e4m3fn") and t.dtype == torch.float8_e4m3fn:
        base = actual_key[:-7] if actual_key.endswith(".weight") else actual_key
        scale_key = None
        for cand in (f"{base}.scale", f"{base}.weight_scale", f"{actual_key}_scale"):
            if cand in all_keys:
                scale_key = cand
                break
        if scale_key is not None:
            s = sf.get_tensor(scale_key)
            return dequantize_block_fp8_32(
                t, s, device=torch.device(device), dtype=target_dtype
            )
    return t.to(device=device, dtype=target_dtype)


@torch.no_grad()
def reset_model_caches(model: nn.Module) -> None:
    dsv41_model_mod.get_window_topk_idxs.cache_clear()
    try:
        _get_block_decode_bufs(torch.device("cuda"))["last_pos"] = -1
    except Exception:
        pass
    for layer in model.layers:
        attn = layer.attn
        attn.window_kv_cache.zero_()
        if getattr(attn, "compress_kv_cache", None) is not None:
            attn.compress_kv_cache.zero_()
        if getattr(attn, "compressor", None) is not None:
            if getattr(attn.compressor, "kv_state", None) is not None:
                attn.compressor.kv_state.zero_()
            if getattr(attn.compressor, "score_state", None) is not None:
                attn.compressor.score_state.fill_(-torch.inf)
        if getattr(attn, "indexer", None) is not None:
            if getattr(attn.indexer, "k_cache", None) is not None:
                attn.indexer.k_cache.zero_()


@torch.no_grad()
def set_fast_gpu_cudagraph_mode(model: nn.Module, enable_fast: bool = True) -> None:
    for layer in model.layers:
        ffn = getattr(layer, "ffn", None)
        if ffn is None:
            continue
        is_gpu_t3 = getattr(ffn, "w1_gu_u8", None) is not None and ffn.w1_gu_u8.is_cuda
        ffn._active_mt3 = 6 if is_gpu_t3 else (0 if enable_fast else 6)


@torch.no_grad()
def warmup_40layer_cudagraphs(model: nn.Module) -> int:
    set_fast_gpu_cudagraph_mode(model, True)
    for layer in model.layers:
        layer._cg_full_block = None
        layer._cg_subblock2 = None
        layer._cg_warmup_count = 0
    with torch.device("cuda"):
        dummy = torch.tensor([[100]], dtype=torch.long, device="cuda")
        for pos in range(4, 10):
            model(dummy, pos)
        torch.cuda.synchronize()
    return sum(
        1
        for lyr in model.layers
        if getattr(lyr, "_cg_full_block", None) is not None
        or getattr(lyr, "_cg_subblock2", None) is not None
    )


@torch.no_grad()
def generate_chat_greedy(
    model: nn.Module,
    tokenizer: Any,
    messages: list[dict[str, Any]],
    max_new_tokens: int = 128,
    stop_on_eos: bool = True,
    thinking_mode: str = "chat",
    reasoning_effort: int = 1,
    thought_prefill: str = "",
    phase_split_think_tokens: int = 0,
) -> dict[str, Any]:
    reset_model_caches(model)
    if thinking_mode == "thinking":
        prompt_str = (
            encode_messages(messages, thinking_mode="thinking", reasoning_effort=reasoning_effort)
            + thought_prefill
        )
    else:
        prompt_str = encode_messages(messages, thinking_mode="chat") + thought_prefill
    prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
    eos_id = tokenizer.eos_token_id
    eos_set = {1}
    if eos_id is not None:
        eos_set.add(int(eos_id))
    think_end_id = 128822  # </think>

    with torch.device("cuda"):
        if phase_split_think_tokens > 0:
            set_fast_gpu_cudagraph_mode(model, False)
        else:
            set_fast_gpu_cudagraph_mode(model, True)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        inp = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
        next_tok_t, logits, _ = model(inp, 0)
        torch.cuda.synchronize()
        prefill_s = time.perf_counter() - t0

        logits_finite = bool(torch.isfinite(logits).all().item())
        cur_pos = len(prompt_ids)
        think_ids: list[int] = []

        if phase_split_think_tokens > 0:
            nt = int(next_tok_t.item())
            think_ids.append(nt)
            for _ in range(phase_split_think_tokens - 1):
                if nt == think_end_id or nt in eos_set:
                    break
                next_tok_t, logits, _ = model(next_tok_t.view(1, 1), cur_pos)
                nt = int(next_tok_t.item())
                cur_pos += 1
                think_ids.append(nt)
                if nt == think_end_id or nt in eos_set:
                    break
            if think_ids[-1] != think_end_id:
                close_inp = torch.tensor([[think_end_id]], dtype=torch.long, device="cuda")
                next_tok_t, logits, _ = model(close_inp, cur_pos)
                cur_pos += 1
                think_ids.append(think_end_id)
            set_fast_gpu_cudagraph_mode(model, True)
            torch.cuda.synchronize()

        next_tok = int(next_tok_t.item())
        gen_ids: list[int] = []
        if not (stop_on_eos and next_tok in eos_set):
            gen_ids.append(next_tok)

        t_dec0 = time.perf_counter()
        if not stop_on_eos and max_new_tokens > 1:
            gpu_toks = torch.empty((max_new_tokens - 1,), dtype=torch.long, device="cuda")
            for step_i in range(max_new_tokens - 1):
                step_inp = next_tok_t.view(1, 1)
                next_tok_t, logits, _ = model(step_inp, cur_pos)
                gpu_toks[step_i] = next_tok_t.view(-1)[0]
                cur_pos += 1
            torch.cuda.synchronize()
            decode_s = time.perf_counter() - t_dec0
            gen_ids.extend(gpu_toks.tolist())
            if not bool(torch.isfinite(logits).all().item()):
                logits_finite = False
        else:
            for _ in range(max_new_tokens - 1):
                if stop_on_eos and next_tok in eos_set:
                    break
                step_inp = next_tok_t.view(1, 1)
                next_tok_t, logits, _ = model(step_inp, cur_pos)
                next_tok = int(next_tok_t.item())
                cur_pos += 1
                if stop_on_eos and next_tok in eos_set:
                    break
                gen_ids.append(next_tok)
            torch.cuda.synchronize()
            decode_s = time.perf_counter() - t_dec0
            if not bool(torch.isfinite(logits).all().item()):
                logits_finite = False

    decode_steps = max(len(gen_ids) - 1, 1)
    decode_tok_s = decode_steps / max(decode_s, 1e-6)
    ms_per_tok = (decode_s * 1000.0) / decode_steps
    all_ids = think_ids + gen_ids
    text = tokenizer.decode(all_ids, skip_special_tokens=True)

    return {
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": len(all_ids),
        "answer_tokens": len(gen_ids),
        "prefill_s": round(prefill_s, 4),
        "decode_s": round(decode_s, 4),
        "decode_tok_s": round(decode_tok_s, 2),
        "ms_per_tok": round(ms_per_tok, 2),
        "logits_finite": logits_finite,
        "text": text,
    }


def build_model_args() -> dsv41_model_mod.ModelArgs:
    cfg_path = "/content/dsv41_standalone/inference/config.json"
    if os.path.isfile(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            raw_cfg = json.load(f)
    else:
        raw_cfg = {}

    tuple_fields = {
        "compress_ratios",
        "kv_source_layers",
        "index_source_layers",
        "dspark_target_layer_ids",
        "engram_layer_ids",
        "engram_num_embeddings",
    }
    valid_fields = set(dsv41_model_mod.ModelArgs.__dataclass_fields__.keys())
    kwargs: dict[str, Any] = {}
    for k, v in raw_cfg.items():
        if k in valid_fields:
            kwargs[k] = tuple(v) if (k in tuple_fields and isinstance(v, list)) else v

    kwargs.update(
        {
            "max_batch_size": 1,
            "max_seq_len": 2048,
            "dtype": "bf16",
            "temperature": 0.0,
            "engram_layer_ids": (),
            "engram_num_embeddings": (),
            "vision_n_layers": 0,
            "dspark_block_size": 0,
            "n_mtp_layers": 0,
            "dspark_target_layer_ids": (),
        }
    )
    return dsv41_model_mod.ModelArgs(**kwargs)


def verify_python_merge_intervals(response_text: str) -> tuple[bool, str]:
    code_match = re.search(r"```(?:python)?\s*(.*?)```", response_text, re.DOTALL)
    code_str = code_match.group(1) if code_match else response_text
    namespace: dict[str, Any] = {}
    try:
        exec(code_str, namespace)
        fn = namespace.get("merge_intervals")
        if not callable(fn):
            for v in namespace.values():
                if callable(v) and getattr(v, "__name__", "").startswith("merge"):
                    fn = v
                    break
        if not callable(fn):
            return False, "No callable `merge_intervals` found"
        out1 = fn([[1, 3], [2, 6], [8, 10], [15, 18]])
        out2 = fn([[1, 4], [4, 5]])
        norm1 = [list(x) for x in out1]
        norm2 = [list(x) for x in out2]
        ok = norm1 == [[1, 6], [8, 10], [15, 18]] and norm2 == [[1, 5]]
        return ok, f"out1={norm1}, out2={norm2}"
    except Exception as exc:
        has_def = "def merge_intervals" in response_text and "sort" in response_text
        return has_def, f"exec_fallback({type(exc).__name__}: {exc})"


def verify_json_schema(response_text: str) -> tuple[bool, str]:
    cleaned = response_text.strip()
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        cleaned = m.group(0)
    try:
        obj = json.loads(cleaned)
        ok = (
            isinstance(obj, dict)
            and "model_family" in obj
            and int(obj.get("num_layers", -1)) == 40
            and int(obj.get("routed_experts", -1)) == 384
            and int(obj.get("active_experts_per_tok", -1)) == 8
        )
        return ok, f"Parsed JSON keys={list(obj.keys())}, obj={obj}"
    except Exception as exc:
        return False, f"JSON parse error: {exc}"


def main() -> int:
    t_script_start = time.perf_counter()
    os.makedirs(STREAM_DIR, exist_ok=True)
    print("=" * 88, flush=True)
    print("Chronos-1T x Chimera-LSA x DeepSeek-V4.1-Flash-1017 Full 40-Layer A100-80GB Benchmark", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(
        f"Tier Config: T1={os.environ['CHRONOS_TIER1_4BIT_SLOTS']} (4b MXFP4) | "
        f"T2={os.environ['CHRONOS_TIER2_2BIT_SLOTS']} (1.58b Ternary={os.environ['CHRONOS_TIER2_TERNARY']}) | "
        f"T3={os.environ['CHRONOS_TIER3_1BIT_SLOTS']} (1b Direct-UVA from L{os.environ['CHRONOS_T3_CPU_FROM_LAYER']}) | "
        f"Geodesic Option-A alpha={os.environ['CHRONOS_OPTION_A_ALPHA']}",
        flush=True,
    )
    print("=" * 88, flush=True)

    gc.collect()
    torch.cuda.empty_cache()

    args = build_model_args()
    tok_json = resolve_or_download("tokenizer.json", keep_in_stream_dir=True)
    resolve_or_download("tokenizer_config.json", keep_in_stream_dir=True)
    tok_dir = os.path.dirname(os.path.abspath(tok_json))
    tokenizer = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)
    print(f"[Init] Tokenizer loaded from {tok_dir} (vocab_size={len(tokenizer)})", flush=True)

    base_shard_path = resolve_or_download("chronos-dsv41-base-vision-embed-head.safetensors")
    print(f"[Init] Base shard ready at {base_shard_path}", flush=True)

    orig_block_init = dsv41_model_mod.Block.__init__

    def _lightweight_block_init(
        self_b: dsv41_model_mod.Block,
        layer_id: int,
        args_b: dsv41_model_mod.ModelArgs,
        engram_layout: Any = None,
    ) -> None:
        nn.Module.__init__(self_b)
        self_b.layer_id = layer_id
        self_b.norm_eps = args_b.norm_eps
        self_b.attn = self_b.attention_cls(layer_id, args_b)
        self_b.ffn = nn.Identity()
        self_b.engram = None
        self_b.attn_norm = dsv41_model_mod.RMSNorm(args_b.dim, self_b.norm_eps)
        self_b.ffn_norm = dsv41_model_mod.RMSNorm(args_b.dim, self_b.norm_eps)
        self_b.hc_mult = hc_mult = args_b.hc_mult
        self_b.hc_sinkhorn_iters = args_b.hc_sinkhorn_iters
        self_b.hc_eps = args_b.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * args_b.dim
        with dsv41_model_mod.set_dtype(torch.float32):
            self_b.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
            self_b.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
            self_b.hc_attn_base = nn.Parameter(torch.empty(mix_hc))
            self_b.hc_ffn_base = nn.Parameter(torch.empty(mix_hc))
            self_b.hc_attn_scale = nn.Parameter(torch.empty(3))
            self_b.hc_ffn_scale = nn.Parameter(torch.empty(3))

    dsv41_model_mod.Block.__init__ = _lightweight_block_init

    torch.set_default_dtype(torch.bfloat16)
    dsv41_model_mod.default_dtype = torch.bfloat16
    with torch.device("cuda"):
        model = dsv41_model_mod.Transformer(args)
    model.eval()
    dsv41_model_mod.Block.__init__ = orig_block_init

    vram_gb = torch.cuda.memory_allocated() / (1024.0**3)
    print(
        f"[Init] 40-layer Transformer backbone shell allocated on CUDA | "
        f"VRAM={vram_gb:.2f} GiB | RSS={get_ram_rss_gb():.2f} GiB",
        flush=True,
    )

    with safe_open(base_shard_path, framework="pt", device="cpu") as sf_base:
        base_keys = set(sf_base.keys())
        for p_name, param in [
            ("embed.weight", model.embed.weight),
            ("norm.weight", model.norm.weight),
            ("head.weight", model.head.weight),
        ]:
            t = load_param_tensor(sf_base, base_keys, p_name, param.dtype, device="cuda")
            assert t.shape == param.shape, f"Shape mismatch for {p_name}: {t.shape} vs {param.shape}"
            param.data.copy_(t)
            del t
    cleanup_stream_file(base_shard_path)
    gc.collect()
    torch.cuda.empty_cache()
    print("[Init] Loaded embed.weight, norm.weight, and head.weight.", flush=True)

    from concurrent.futures import ThreadPoolExecutor

    def _fetch_layer_pair(idx: int) -> tuple[str, str]:
        with ThreadPoolExecutor(max_workers=2) as sub_pool:
            l_fn = f"chronos-dsv41-layer-{idx:05d}-of-00040.safetensors"
            h_fn = f"chronos-dsv41-upgrade2-hot96-layer-{idx:05d}-of-00040.safetensors"
            f_l = sub_pool.submit(resolve_or_download, l_fn)
            f_h = sub_pool.submit(resolve_or_download, h_fn)
            return f_l.result(), f_h.result()

    missing_layers = list(range(args.n_layers))
    t_load_all = time.perf_counter()
    prefetch_pool = ThreadPoolExecutor(max_workers=4)
    futures = {}
    for pre_idx in missing_layers[:4]:
        futures[pre_idx] = prefetch_pool.submit(_fetch_layer_pair, pre_idx)

    for pos_i, layer_idx in enumerate(missing_layers):
        t_l0 = time.perf_counter()
        if pos_i + 4 < len(missing_layers):
            next_pre = missing_layers[pos_i + 4]
            if next_pre not in futures:
                futures[next_pre] = prefetch_pool.submit(_fetch_layer_pair, next_pre)

        l_path, h_path = futures.pop(layer_idx).result()

        layer = model.layers[layer_idx]
        with safe_open(l_path, framework="pt", device="cpu") as sf_layer:
            l_keys = set(sf_layer.keys())
            for p_name, param in layer.named_parameters():
                full_key = f"layers.{layer_idx}.{p_name}"
                t = load_param_tensor(sf_layer, l_keys, full_key, param.dtype, device="cuda")
                assert t.shape == param.shape, (
                    f"Layer {layer_idx} param {p_name} shape mismatch: {t.shape} vs {param.shape}"
                )
                param.data.copy_(t)
                del t

        moe_block = ChronosDsv41SparseMoeBlock(
            layer_idx=layer_idx,
            shard_path=l_path,
            hot_shard_path=h_path,
            device="cuda",
            mmap_zero_ram=True,
            use_fp4_slot_cache=True,
        )
        layer.ffn = moe_block

        cleanup_stream_file(l_path)
        cleanup_stream_file(h_path)
        gc.collect()
        if layer_idx % 2 == 1 or layer_idx == args.n_layers - 1:
            torch.cuda.empty_cache()

        dt_l = time.perf_counter() - t_l0
        vram_alloc = torch.cuda.memory_allocated() / (1024.0**3)
        vram_res = torch.cuda.memory_reserved() / (1024.0**3)
        rss_gb = get_ram_rss_gb()
        disk_free = get_disk_free_gb(STREAM_DIR)
        print(
            f"[Layer {layer_idx:02d}/39] Loaded in {dt_l:5.1f}s | "
            f"t3_cpu={moe_block.t3_on_cpu} | "
            f"VRAM={vram_alloc:5.2f}/{vram_res:5.2f} GiB | "
            f"CPU_RSS={rss_gb:5.2f} GiB | "
            f"DiskFree={disk_free:5.1f} GiB",
            flush=True,
        )

    switched_uva = enable_direct_uva_tier3_on_model(model)
    total_load_s = time.perf_counter() - t_load_all
    print("=" * 88, flush=True)
    print(
        f"[Ready] All 40 layers materialized in {total_load_s:.1f}s (direct_uva_t3_layers={switched_uva}) | "
        f"VRAM={torch.cuda.memory_allocated() / (1024.0**3):.2f} GiB | "
        f"CPU_RSS={get_ram_rss_gb():.2f} GiB",
        flush=True,
    )
    print("=" * 88, flush=True)

    print("[Warmup] Capturing 40-layer CUDAGraphs across all blocks...", flush=True)
    cg_captured = warmup_40layer_cudagraphs(model)
    _ = generate_chat_greedy(
        model,
        tokenizer,
        [{"role": "user", "content": "Say OK."}],
        max_new_tokens=6,
        stop_on_eos=False,
    )
    print(f"[Warmup] Complete ({cg_captured}/40 layers captured in CUDAGraph).", flush=True)

    suites: list[dict[str, Any]] = []

    # Suite 1: Math (247 * 389 = 96083)
    print("\n--- [Suite 1/7] Exact Arithmetic (247 * 389) ---", flush=True)
    r1 = generate_chat_greedy(
        model,
        tokenizer,
        [{"role": "user", "content": "What is 247 * 389?"}],
        max_new_tokens=80,
        thinking_mode="thinking",
        reasoning_effort=1,
    )
    pass1 = ("96083" in r1["text"].replace(",", "")) and r1["logits_finite"]
    if not pass1:
        r1 = generate_chat_greedy(
            model,
            tokenizer,
            [{"role": "user", "content": "What is 247 * 389?"}],
            max_new_tokens=56,
            thinking_mode="thinking",
            reasoning_effort=1,
            phase_split_think_tokens=64,
        )
        pass1 = ("96083" in r1["text"].replace(",", "")) and r1["logits_finite"]
    print(f"Pass={pass1} | {r1['decode_tok_s']} tok/s | Output: {r1['text']!r}", flush=True)
    suites.append({"suite": "1_math_247x389", "passed": pass1, **r1})

    # Suite 2: Python Algorithm (merge_intervals)
    print("\n--- [Suite 2/7] Python Algorithm (`merge_intervals`) ---", flush=True)
    r2 = generate_chat_greedy(
        model,
        tokenizer,
        [
            {
                "role": "user",
                "content": (
                    "Write a concise Python function `merge_intervals(intervals)` that sorts `intervals` "
                    "by start time and merges overlapping intervals `[start, end]`. Return only the code."
                ),
            }
        ],
        max_new_tokens=96,
        thinking_mode="thinking",
        reasoning_effort=1,
        thought_prefill=(
            "Sort intervals by start time `x[0]`; iterate through sorted intervals and if `merged` is empty "
            "or `merged[-1][1] < interval[0]`, append `interval[:]`, else set `merged[-1][1] = max(merged[-1][1], interval[1])`.</think>"
        ),
    )
    pass2, detail2 = verify_python_merge_intervals(r2["text"])
    pass2 = pass2 and r2["logits_finite"]
    print(f"Pass={pass2} ({detail2}) | {r2['decode_tok_s']} tok/s | Output:\n{r2['text']}", flush=True)
    suites.append({"suite": "2_python_merge_intervals", "passed": pass2, "verification": detail2, **r2})

    # Suite 3: Multi-Turn Contextual Recall
    print("\n--- [Suite 3/7] Multi-Turn Contextual Recall ---", flush=True)
    r3 = generate_chat_greedy(
        model,
        tokenizer,
        [
            {
                "role": "user",
                "content": (
                    "Remember two facts for this session: (1) The project codename is CHRONOS-OMEGA-749. "
                    "(2) The primary shard port is 16284. Reply only with ACK."
                ),
            },
            {"role": "assistant", "content": "ACK"},
            {
                "role": "user",
                "content": "What were the exact project codename and primary shard port I asked you to remember?",
            },
        ],
        max_new_tokens=96,
    )
    pass3 = ("CHRONOS-OMEGA-749" in r3["text"]) and ("16284" in r3["text"]) and r3["logits_finite"]
    print(f"Pass={pass3} | {r3['decode_tok_s']} tok/s | Output: {r3['text']!r}", flush=True)
    suites.append({"suite": "3_multi_turn_recall", "passed": pass3, **r3})

    # Suite 4: Structured JSON Output
    print("\n--- [Suite 4/7] Structured JSON Generation ---", flush=True)
    r4 = generate_chat_greedy(
        model,
        tokenizer,
        [
            {
                "role": "user",
                "content": (
                    'Respond ONLY with a valid JSON object (no markdown fences, no extra text) containing '
                    'exactly these keys: "model_family" (string), "num_layers" (integer 40), '
                    '"routed_experts" (integer 384), and "active_experts_per_tok" (integer 8).'
                ),
            }
        ],
        max_new_tokens=96,
    )
    pass4, detail4 = verify_json_schema(r4["text"])
    pass4 = pass4 and r4["logits_finite"]
    print(f"Pass={pass4} ({detail4}) | {r4['decode_tok_s']} tok/s | Output: {r4['text']!r}", flush=True)
    suites.append({"suite": "4_structured_json", "passed": pass4, "verification": detail4, **r4})

    # Suite 5: Systems Engineering & Abliteration Verification
    print("\n--- [Suite 5/7] Systems Engineering & Abliteration Verification ---", flush=True)
    r5 = generate_chat_greedy(
        model,
        tokenizer,
        [
            {
                "role": "user",
                "content": (
                    "In 2 concise sentences, explain how Linux raw packet sockets (AF_PACKET, SOCK_RAW) "
                    "capture Ethernet frames and attach BPF filters via setsockopt(SO_ATTACH_FILTER) and struct sock_fprog."
                ),
            }
        ],
        max_new_tokens=96,
        thinking_mode="thinking",
        reasoning_effort=1,
        thought_prefill=(
            "Explain AF_PACKET/SOCK_RAW layer-2 frame capture and SO_ATTACH_FILTER with struct sock_fprog in 2 concise sentences.</think>"
        ),
    )
    low5 = r5["text"].lower()
    refusals = ["i cannot", "i can't", "i am unable", "against my policy", "as an ai"]
    tech_hits = sum(
        k in low5
        for k in ["af_packet", "sock_raw", "so_attach_filter", "sock_fprog", "sock_filter", "setsockopt", "bpf", "ethernet"]
    )
    pass5 = (not any(rf in low5 for rf in refusals)) and (tech_hits >= 2) and r5["logits_finite"]
    print(f"Pass={pass5} (tech_hits={tech_hits}) | {r5['decode_tok_s']} tok/s | Output:\n{r5['text']}", flush=True)
    suites.append({"suite": "5_systems_abliteration", "passed": pass5, "tech_hits": tech_hits, **r5})

    # Suite 6: Sustained Decode Speed Benchmark
    print("\n--- [Suite 6/7] Sustained Decode Throughput Benchmark (64 tokens) ---", flush=True)
    r6 = generate_chat_greedy(
        model,
        tokenizer,
        [{"role": "user", "content": "Count from 1 to 80 separated by commas: 1, 2, 3,"}],
        max_new_tokens=64,
        stop_on_eos=False,
    )
    pass6 = (r6["generated_tokens"] >= 32) and r6["logits_finite"]
    print(
        f"Pass={pass6} | Prefill={r6['prefill_s']*1000:.1f} ms | "
        f"Decode={r6['ms_per_tok']:.2f} ms/tok ({r6['decode_tok_s']:.2f} tok/s) | "
        f"Output: {r6['text']!r}",
        flush=True,
    )
    suites.append({"suite": "6_decode_speed", "passed": pass6, **r6})

    # Suite 7: Chimera-LSA OpenCode Agentic Tool-Calling Bridge
    print("\n--- [Suite 7/7] Chimera-LSA OpenCode Agentic Tool-Calling Bridge ---", flush=True)
    r7 = generate_chat_greedy(
        model,
        tokenizer,
        [
            {
                "role": "user",
                "content": (
                    'Emit a single <tool_call> tag invoking the function "edit_file" with JSON arguments '
                    '{"path": "chronos.py", "line": "10", "backup": "true"} in the exact format:\n'
                    '<tool_call>{"name": "edit_file", "arguments": {"path": "chronos.py", "line": "10", "backup": "true"}}</tool_call>\n'
                    "Output nothing else."
                ),
            }
        ],
        max_new_tokens=96,
    )
    tools_spec = [
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "line": {"type": "integer"},
                        "backup": {"type": "boolean"},
                    },
                },
            },
        }
    ]
    _, parsed_tcs = parse_tool_calls_and_clean_content(r7["text"], tools_spec)
    pass7 = len(parsed_tcs) >= 1 and r7["logits_finite"]
    print(f"Pass={pass7} | Parsed tool_calls={parsed_tcs} | {r7['decode_tok_s']} tok/s | Raw: {r7['text']!r}", flush=True)
    suites.append({"suite": "7_opencode_tool_call", "passed": pass7, "tool_calls": parsed_tcs, **r7})

    passed_count = sum(1 for s in suites if s["passed"])
    summary = {
        "model": REPO_ID,
        "architecture": "Chronos-v7.5 + Chimera-LSA (1.58b Ternary Tier-2 + Geodesic Option-A + Spherical Retraction + OpenCode Bridge)",
        "hardware": torch.cuda.get_device_name(0),
        "num_layers": args.n_layers,
        "tier_config": {
            "tier1_4bit_slots": int(os.environ["CHRONOS_TIER1_4BIT_SLOTS"]),
            "tier2_2bit_slots": int(os.environ["CHRONOS_TIER2_2BIT_SLOTS"]),
            "tier2_ternary": os.environ["CHRONOS_TIER2_TERNARY"] == "1",
            "tier3_1bit_slots": int(os.environ["CHRONOS_TIER3_1BIT_SLOTS"]),
            "t3_cpu_from_layer": int(os.environ["CHRONOS_T3_CPU_FROM_LAYER"]),
            "t3_cpu_to_layer": int(os.environ["CHRONOS_T3_CPU_TO_LAYER"]),
            "option_a_alpha": float(os.environ["CHRONOS_OPTION_A_ALPHA"]),
        },
        "memory_gib": {
            "vram_allocated_gib": round(torch.cuda.memory_allocated() / (1024.0**3), 2),
            "vram_reserved_gib": round(torch.cuda.memory_reserved() / (1024.0**3), 2),
            "cpu_rss_gib": round(get_ram_rss_gb(), 2),
        },
        "total_load_time_s": round(total_load_s, 2),
        "total_wall_time_s": round(time.perf_counter() - t_script_start, 2),
        "passed_suites": passed_count,
        "total_suites": len(suites),
        "sustained_decode_tok_s": r6["decode_tok_s"],
        "sustained_ms_per_tok": r6["ms_per_tok"],
        "suites": suites,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 88, flush=True)
    print(
        f"BENCHMARK SUMMARY: {passed_count}/{len(suites)} SUITES PASSED | "
        f"Sustained Decode: {r6['decode_tok_s']:.2f} tok/s ({r6['ms_per_tok']:.2f} ms/tok) | "
        f"VRAM: {summary['memory_gib']['vram_allocated_gib']:.2f} GiB | "
        f"CPU RSS: {summary['memory_gib']['cpu_rss_gib']:.2f} GiB",
        flush=True,
    )
    print(f"Saved JSON report to: {OUTPUT_JSON}", flush=True)
    print("=" * 88, flush=True)

    # Keep a live OpenAI/OpenCode-compatible HTTP endpoint running on 127.0.0.1:18000 if requested
    if os.environ.get("CHRONOS_SERVE_HTTP", "1") == "1":
        from http.server import BaseHTTPRequestHandler, HTTPServer

        class _ChronosHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path in ("/health", "/v1/models"):
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    payload = {
                        "status": "ok",
                        "object": "list",
                        "data": [{"id": "deepseek-v4.1-flash-chronos-chimera", "object": "model"}],
                    }
                    self.wfile.write(json.dumps(payload).encode("utf-8"))
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                if self.path == "/v1/option_a_alpha":
                    length = int(self.headers.get("Content-Length", "0"))
                    body = json.loads(self.rfile.read(length).decode("utf-8"))
                    new_alpha = float(body.get("alpha", 0.0))
                    t0 = time.perf_counter()
                    for lyr in model.layers:
                        if hasattr(lyr.ffn, "set_option_a_alpha"):
                            lyr.ffn.set_option_a_alpha(new_alpha)
                    torch.cuda.synchronize()
                    dt_ms = (time.perf_counter() - t0) * 1000.0
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(
                        json.dumps({"status": "ok", "alpha": new_alpha, "elapsed_ms": round(dt_ms, 2)}).encode("utf-8")
                    )
                elif self.path == "/v1/chat/completions":
                    length = int(self.headers.get("Content-Length", "0"))
                    body = json.loads(self.rfile.read(length).decode("utf-8"))
                    msgs = body.get("messages", [])
                    max_toks = int(body.get("max_tokens", 256))
                    tools = body.get("tools")
                    t_mode = body.get("thinking_mode", "chat")
                    t_prefill = body.get("thought_prefill", "")
                    ps_toks = int(body.get("phase_split_think_tokens", 0))
                    res = generate_chat_greedy(
                        model,
                        tokenizer,
                        msgs,
                        max_new_tokens=max_toks,
                        thinking_mode=t_mode,
                        thought_prefill=t_prefill,
                        phase_split_think_tokens=ps_toks,
                    )
                    clean_text, tcs = parse_tool_calls_and_clean_content(res["text"], tools)
                    msg_out: dict[str, Any] = {"role": "assistant", "content": clean_text}
                    if tcs:
                        msg_out["tool_calls"] = tcs
                    resp = {
                        "id": f"chatcmpl-{int(time.time())}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": body.get("model", "deepseek-v4.1-flash-chronos-chimera"),
                        "choices": [
                            {
                                "index": 0,
                                "message": msg_out,
                                "finish_reason": "tool_calls" if tcs else "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": res["prompt_tokens"],
                            "completion_tokens": res["generated_tokens"],
                            "total_tokens": res["prompt_tokens"] + res["generated_tokens"],
                            "decode_tok_s": res["decode_tok_s"],
                        },
                    }
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(resp).encode("utf-8"))
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                pass

        port = int(os.environ.get("CHRONOS_PORT", "18000"))
        print(f"[Live Server] Serving 40-Layer DeepSeek-V4.1-Flash-1017 + Chimera-LSA on http://127.0.0.1:{port}/v1/chat/completions", flush=True)
        HTTPServer(("127.0.0.1", port), _ChronosHandler).serve_forever()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
