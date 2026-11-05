#!/usr/bin/env python3
"""
Chronos-1T x DeepSeek-V4.1-Flash-1017 (`deepseek-ai/DeepSeek-V4.1-Flash-1017` / Abliterated)
Runtime Loader & vLLM Hook for `DeepseekV41ForCausalLM`.

Provides:
1. `enable_chronos_dsv41_vllm_hook()`:
   - Patches `DeepseekV4MoE` in `vllm.models.deepseek_v4.nvidia.model` and
     `vllm.models.deepseek_v41.nvidia.model` with a `ChronosDsv41SparseMoeBlock` class
     (preserving `isinstance(layer.ffn, DeepseekV4MoE)` checks in `set_moe_parameters()`)
     so layers `0..39` directly instantiate `ChronosDsv41SparseMoeBlock` without allocating
     stock FP8 expert tensors.
   - Hooks `DeepseekV4Model.load_weights` and `safetensors_weights_iterator` to skip
     `.ffn.` keys and `chronos-dsv41-upgrade2-hot96-*.safetensors` shards during vLLM
     backbone weight loading and evict page cache after each shard (`POSIX_FADV_DONTNEED`).
2. `install_chronos_moe_into_dsv41(...)`:
   - Swaps `layer.ffn` / `layer.mlp` on an already-constructed `DeepseekV41ForCausalLM` or
     reference `Transformer` instance with `ChronosDsv41SparseMoeBlock` for all converted
     MoE layers (`0..39`).
"""
from __future__ import annotations

import gc
import os
import re
import types
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
from safetensors import safe_open

from chronos_1t.models.dsv41_chronos_moe import (
    ChronosDsv41SparseMoeBlock,
    _HAS_TRITON,
    _triton_bmm_fp4_m1_kernel,
    _triton_bmm_int4_m1_kernel,
    _triton_bmm_mxfp8_m1_kernel,
    _triton_gemv_fp4_m1_kernel,
    _triton_gemv_int4_m1_kernel,
    _triton_gemv_mxfp8_m1_kernel,
    _triton_inv_rope_m1_kernel,
    _triton_quantize_int4_32_kernel,
    _triton_quantize_mse_fp4_32_kernel,
)


def _evict_page_cache(file_path: str) -> None:
    gc.collect()
    if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
        try:
            fd = os.open(os.path.realpath(file_path), os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except OSError:
            pass


@torch.no_grad()
def _dequant_mxfp8_to_bf16(w_fp8: torch.Tensor, s_u8: torch.Tensor) -> torch.Tensor:
    n_rows, k_cols = w_fp8.shape
    s_f32 = (s_u8.view(torch.uint8).to(torch.int32) << 23).view(torch.float32)
    if s_f32.shape[0] < n_rows:
        s_f32 = s_f32.repeat_interleave(n_rows // s_f32.shape[0], dim=0)[:n_rows]
    else:
        s_f32 = s_f32[:n_rows]
    s_f32 = s_f32[:, : k_cols // 32]
    w_f32 = w_fp8.view(torch.float8_e4m3fn).float().view(n_rows, k_cols // 32, 32)
    return (w_f32 * s_f32.unsqueeze(-1)).to(torch.bfloat16).view(n_rows, k_cols).contiguous()


def _chronos_fast_fused_wqa_wkv_gemm(
    self_a: Any, hidden_states: torch.Tensor, *args: Any, **kwargs: Any
) -> torch.Tensor:
    if hidden_states.shape[0] == 1 and hidden_states.is_cuda:
        out_buf = torch.empty((1, 1792), dtype=torch.bfloat16, device=hidden_states.device)
        _triton_gemv_mxfp8_m1_kernel[(1792 // 16,)](
            hidden_states,
            self_a._f_wqa_wkv_w_fp8,
            self_a._f_wqa_wkv_s_u8,
            out_buf[0],
            N_OUT=1792,
            N_BLOCKS_K=160,
            BLOCK_N=16,
            sub_blocks_k=32,
            SCALE_ROW_DIV=self_a._f_wqa_wkv_scale_row_div,
            num_warps=4,
        )
        return out_buf
    res = self_a.fused_wqa_wkv(hidden_states)
    return res[0] if isinstance(res, tuple) else res


def _chronos_fast_wq_b_proj(
    self_a: Any, qr: torch.Tensor, qr_scale: torch.Tensor | None = None
) -> torch.Tensor:
    m_tok = qr.shape[0]
    if m_tok == 0:
        return torch.empty((0, 32768), dtype=qr.dtype, device=qr.device)
    out_buf = torch.empty((m_tok, 32768), dtype=torch.bfloat16, device=qr.device)
    for t_i in range(m_tok):
        _triton_gemv_int4_m1_kernel[(32768 // 16,)](
            qr[t_i : t_i + 1],
            self_a._wq_b_w4_u8,
            self_a._wq_b_s4_bf16,
            out_buf[t_i],
            N_OUT=32768,
            N_BLOCKS_K=40,
            BLOCK_N=16,
            sub_blocks_k=8,
            num_warps=4,
        )
    return out_buf if qr.dtype == torch.bfloat16 else out_buf.to(qr.dtype)


def _chronos_fast_o_proj(
    self_a: Any, attn_out: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    o = attn_out[:, : self_a.n_local_heads, :]
    m_tok = o.shape[0]
    if m_tok == 0:
        return torch.empty((0, 5120), dtype=o.dtype, device=o.device)
    cos_sin_cache = self_a.rotary_emb.cos_sin_cache
    out_buf = torch.empty((m_tok, 5120), dtype=torch.bfloat16, device=o.device)
    x_inv_buf = self_a._o_proj_x_inv_buf
    z_inter_buf = self_a._o_proj_z_inter_buf
    has_fp8 = hasattr(self_a, "_wo_a_w_fp8")
    for t_i in range(m_tok):
        o_t = o[t_i]
        _triton_inv_rope_m1_kernel[(64,)](
            o_t,
            cos_sin_cache,
            positions[t_i : t_i + 1],
            x_inv_buf,
            o_t.stride(0),
            num_warps=2,
        )
        if has_fp8:
            _triton_bmm_mxfp8_m1_kernel[(1024 // 16, 8)](
                x_inv_buf.view(8, 4096),
                self_a._wo_a_w_fp8,
                self_a._wo_a_s_u8,
                z_inter_buf.view(8, 1024),
                N_OUT=1024,
                N_BLOCKS_K=128,
                BLOCK_N=16,
                sub_blocks_k=32,
                SCALE_ROW_DIV=self_a._wo_a_scale_row_div,
                num_warps=4,
            )
            _triton_gemv_mxfp8_m1_kernel[(5120 // 16,)](
                z_inter_buf,
                self_a._wo_b_w_fp8,
                self_a._wo_b_s_u8,
                out_buf[t_i],
                N_OUT=5120,
                N_BLOCKS_K=256,
                BLOCK_N=16,
                sub_blocks_k=32,
                SCALE_ROW_DIV=self_a._wo_b_scale_row_div,
                num_warps=4,
            )
        else:
            _triton_bmm_int4_m1_kernel[(1024 // 16, 8)](
                x_inv_buf.view(8, 4096),
                self_a._wo_a_w4_u8,
                self_a._wo_a_s4_bf16,
                z_inter_buf.view(8, 1024),
                N_OUT=1024,
                N_BLOCKS_K=128,
                BLOCK_N=16,
                sub_blocks_k=32,
                num_warps=4,
            )
            _triton_gemv_int4_m1_kernel[(5120 // 16,)](
                z_inter_buf,
                self_a._wo_b_w4_u8,
                self_a._wo_b_s4_bf16,
                out_buf[t_i],
                N_OUT=5120,
                N_BLOCKS_K=256,
                BLOCK_N=16,
                sub_blocks_k=32,
                num_warps=4,
            )
    return out_buf if o.dtype == torch.bfloat16 else out_buf.to(o.dtype)


def _chronos_fast_apply_head(
    self_lp: Any,
    lm_head: Any,
    hidden_states: torch.Tensor,
    embedding_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if hasattr(lm_head, "_chronos_w_fp8"):
        flat = hidden_states.reshape(-1, hidden_states.shape[-1]).to(torch.bfloat16)
        m_tok = flat.shape[0]
        n_vocab = lm_head._chronos_w_fp8.shape[0]
        out_buf = torch.empty((m_tok, n_vocab), dtype=torch.bfloat16, device=flat.device)
        for t_i in range(m_tok):
            _triton_gemv_mxfp8_m1_kernel[(n_vocab // 16,)](
                flat[t_i : t_i + 1],
                lm_head._chronos_w_fp8,
                lm_head._chronos_s_u8,
                out_buf[t_i],
                N_OUT=n_vocab,
                N_BLOCKS_K=160,
                BLOCK_N=16,
                sub_blocks_k=32,
                SCALE_ROW_DIV=1,
                num_warps=4,
            )
        if embedding_bias is not None:
            out_buf = out_buf + embedding_bias.to(out_buf.dtype)
        return out_buf.reshape(*hidden_states.shape[:-1], n_vocab)
    if hasattr(lm_head, "_chronos_w4_u8"):
        flat = hidden_states.reshape(-1, hidden_states.shape[-1]).to(torch.bfloat16)
        m_tok = flat.shape[0]
        n_vocab = lm_head._chronos_w4_u8.shape[0]
        out_buf = torch.empty((m_tok, n_vocab), dtype=torch.bfloat16, device=flat.device)
        for t_i in range(m_tok):
            _triton_gemv_int4_m1_kernel[(n_vocab // 16,)](
                flat[t_i : t_i + 1],
                lm_head._chronos_w4_u8,
                lm_head._chronos_s4_bf16,
                out_buf[t_i],
                N_OUT=n_vocab,
                N_BLOCKS_K=160,
                BLOCK_N=16,
                sub_blocks_k=32,
                num_warps=4,
            )
        if embedding_bias is not None:
            out_buf = out_buf + embedding_bias.to(out_buf.dtype)
        return out_buf.reshape(*hidden_states.shape[:-1], n_vocab)
    return self_lp._orig_apply_head_dsv41_fp4(lm_head, hidden_states, embedding_bias)


@torch.no_grad()
def _install_chronos_dsv41_decode_accelerators(model: nn.Module) -> None:
    if not _HAS_TRITON or not torch.cuda.is_available():
        return
    if getattr(model, "_chronos_decode_accels_installed", False):
        return
    model._chronos_decode_accels_installed = True

    lm = getattr(model, "language_model", model)
    inner_model = getattr(lm, "model", lm)
    layers = getattr(inner_model, "layers", [])
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    fast_wqb = os.environ.get("CHRONOS_FAST_WQB", "0") == "1"
    wo_int4 = os.environ.get("CHRONOS_WO_INT4", "0") == "1"
    lmhead_int4 = os.environ.get("CHRONOS_LMHEAD_INT4", "0") == "1"

    for lyr in layers:
        if hasattr(lyr, "mhc_stream"):
            lyr.mhc_stream = None
        attn = getattr(lyr, "attn", None)
        if attn is None:
            continue
        attn.aux_stream_list = None

        # 1. fused_wqa_wkv (1792 x 5120): keep 8-bit MXFP8 for M>1, add zero-pad Triton GEMV for M=1 (cos = 1.000000)
        f_wqa = getattr(attn, "fused_wqa_wkv", None)
        if f_wqa is not None and hasattr(f_wqa, "weight") and hasattr(f_wqa, "_chronos_raw_scale_u8"):
            attn._f_wqa_wkv_w_fp8 = f_wqa.weight.data.view(torch.float8_e4m3fn).contiguous()
            attn._f_wqa_wkv_s_u8 = f_wqa._chronos_raw_scale_u8
            delattr(f_wqa, "_chronos_raw_scale_u8")
            attn._f_wqa_wkv_scale_row_div = max(1, 1792 // int(attn._f_wqa_wkv_s_u8.shape[0]))
            attn._fused_wqa_wkv_gemm = types.MethodType(_chronos_fast_fused_wqa_wkv_gemm, attn)

        # 2. wq_b (32768 x 1280): default keep native SM120 8-bit MXFP8 + fused_q_kv_rmsnorm_quant (cos = 1.000000)
        wq_b = getattr(attn, "wq_b", None)
        if fast_wqb and wq_b is not None and hasattr(wq_b, "weight") and wq_b.weight.numel() == 32768 * 1280:
            raw_s_q = getattr(wq_b, "_chronos_raw_scale_u8", getattr(wq_b, "weight_scale", None))
            if raw_s_q is not None:
                attn.__dict__["_can_fuse_query_quant"] = False
                wq_bf16 = _dequant_mxfp8_to_bf16(wq_b.weight.data, raw_s_q)
                attn._wq_b_w4_u8 = torch.empty((32768, 640), dtype=torch.uint8, device=device)
                attn._wq_b_s4_bf16 = torch.empty((32768, 40), dtype=torch.bfloat16, device=device)
                _triton_quantize_int4_32_kernel[(32768, 40)](
                    wq_bf16, attn._wq_b_w4_u8, attn._wq_b_s4_bf16, N_BLOCKS_C=40, num_warps=1
                )
                del wq_bf16
                wq_b.weight.data = torch.empty((0,), dtype=wq_b.weight.dtype, device=device)
                if hasattr(wq_b, "weight_scale") and isinstance(wq_b.weight_scale, torch.Tensor):
                    wq_b.weight_scale.data = torch.empty((0,), dtype=wq_b.weight_scale.dtype, device=device)
                if hasattr(wq_b, "_chronos_raw_scale_u8"):
                    delattr(wq_b, "_chronos_raw_scale_u8")
                attn._wq_b_proj = types.MethodType(_chronos_fast_wq_b_proj, attn)
        elif wq_b is not None and hasattr(wq_b, "_chronos_raw_scale_u8"):
            delattr(wq_b, "_chronos_raw_scale_u8")

        # 3. wo_a (8 x 1024 x 4096) & wo_b (5120 x 8192): default keep Exact 8-bit MXFP8 (cos = 1.000000)
        wo_a = getattr(attn, "wo_a", None)
        wo_b = getattr(attn, "wo_b", None)
        if (
            wo_a is not None
            and wo_b is not None
            and hasattr(wo_a, "weight")
            and hasattr(wo_b, "weight")
            and wo_a.weight.numel() == 8192 * 4096
            and wo_b.weight.numel() == 5120 * 8192
        ):
            raw_s_oa = getattr(wo_a, "_chronos_raw_scale_u8", getattr(wo_a, "weight_scale", None))
            raw_s_ob = getattr(wo_b, "_chronos_raw_scale_u8", getattr(wo_b, "weight_scale", None))
            if raw_s_oa is not None and raw_s_ob is not None:
                if wo_int4:
                    oa_bf16 = _dequant_mxfp8_to_bf16(wo_a.weight.data.view(8192, 4096), raw_s_oa)
                    attn._wo_a_w4_u8 = torch.empty((8, 1024, 2048), dtype=torch.uint8, device=device)
                    attn._wo_a_s4_bf16 = torch.empty((8, 1024, 128), dtype=torch.bfloat16, device=device)
                    _triton_quantize_int4_32_kernel[(8192, 128)](
                        oa_bf16,
                        attn._wo_a_w4_u8.view(8192, 2048),
                        attn._wo_a_s4_bf16.view(8192, 128),
                        N_BLOCKS_C=128,
                        num_warps=1,
                    )
                    del oa_bf16
                    ob_bf16 = _dequant_mxfp8_to_bf16(wo_b.weight.data, raw_s_ob)
                    attn._wo_b_w4_u8 = torch.empty((5120, 4096), dtype=torch.uint8, device=device)
                    attn._wo_b_s4_bf16 = torch.empty((5120, 256), dtype=torch.bfloat16, device=device)
                    _triton_quantize_int4_32_kernel[(5120, 256)](
                        ob_bf16, attn._wo_b_w4_u8, attn._wo_b_s4_bf16, N_BLOCKS_C=256, num_warps=1
                    )
                    del ob_bf16
                else:
                    attn._wo_a_w_fp8 = wo_a.weight.data.view(torch.float8_e4m3fn).view(8192, 4096).contiguous()
                    s_oa_u8 = raw_s_oa.view(torch.uint8).reshape(-1, 128).contiguous()
                    attn._wo_a_s_u8 = s_oa_u8
                    attn._wo_a_scale_row_div = max(1, 8192 // int(s_oa_u8.shape[0]))

                    attn._wo_b_w_fp8 = wo_b.weight.data.view(torch.float8_e4m3fn).view(5120, 8192).contiguous()
                    s_ob_u8 = raw_s_ob.view(torch.uint8).reshape(-1, 256).contiguous()
                    attn._wo_b_s_u8 = s_ob_u8
                    attn._wo_b_scale_row_div = max(1, 5120 // int(s_ob_u8.shape[0]))

                wo_a.weight.data = torch.empty((0,), dtype=wo_a.weight.dtype, device=device)
                if hasattr(wo_a, "weight_scale") and isinstance(wo_a.weight_scale, torch.Tensor):
                    wo_a.weight_scale.data = torch.empty((0,), dtype=wo_a.weight_scale.dtype, device=device)
                if hasattr(wo_a, "_chronos_raw_scale_u8"):
                    delattr(wo_a, "_chronos_raw_scale_u8")
                wo_b.weight.data = torch.empty((0,), dtype=wo_b.weight.dtype, device=device)
                if hasattr(wo_b, "weight_scale") and isinstance(wo_b.weight_scale, torch.Tensor):
                    wo_b.weight_scale.data = torch.empty((0,), dtype=wo_b.weight_scale.dtype, device=device)
                if hasattr(wo_b, "_chronos_raw_scale_u8"):
                    delattr(wo_b, "_chronos_raw_scale_u8")

                attn._o_proj_x_inv_buf = torch.empty((64, 512), dtype=torch.bfloat16, device=device)
                attn._o_proj_z_inter_buf = torch.empty((1, 8192), dtype=torch.bfloat16, device=device)
                attn._o_proj = types.MethodType(_chronos_fast_o_proj, attn)

    for m in model.modules():
        if hasattr(m, "_chronos_raw_scale_u8"):
            delattr(m, "_chronos_raw_scale_u8")

    # 4. lm_head (129280 x 5120 BF16): default 8-bit MXFP8 (cos = 0.99995) or INT4-MSE if CHRONOS_LMHEAD_INT4=1
    lm_head = getattr(lm, "lm_head", getattr(model, "lm_head", None))
    logits_proc = getattr(lm, "logits_processor", getattr(model, "logits_processor", None))
    if (
        lm_head is not None
        and logits_proc is not None
        and hasattr(lm_head, "weight")
        and lm_head.weight.ndim == 2
        and lm_head.weight.shape[1] == 5120
    ):
        n_vocab = lm_head.weight.shape[0]
        lm_bf16 = lm_head.weight.data.to(device=device, dtype=torch.bfloat16).contiguous()
        if lmhead_int4:
            lm_head._chronos_w4_u8 = torch.empty((n_vocab, 2560), dtype=torch.uint8, device=device)
            lm_head._chronos_s4_bf16 = torch.empty((n_vocab, 160), dtype=torch.bfloat16, device=device)
            _triton_quantize_int4_32_kernel[(n_vocab, 160)](
                lm_bf16, lm_head._chronos_w4_u8, lm_head._chronos_s4_bf16, N_BLOCKS_C=160, num_warps=1
            )
        else:
            w_f32 = lm_bf16.float().view(n_vocab, 160, 32)
            amax = w_f32.abs().amax(dim=-1).clamp(min=1e-12)
            exp_u8 = (torch.ceil(torch.log2(amax / 256.0)).to(torch.int32) + 127).clamp(1, 254).to(torch.uint8)
            scale_f32 = (exp_u8.to(torch.int32) << 23).view(torch.float32)
            lm_head._chronos_w_fp8 = (
                (w_f32 / scale_f32.unsqueeze(-1))
                .clamp(-448.0, 448.0)
                .to(torch.float8_e4m3fn)
                .view(n_vocab, 5120)
                .contiguous()
            )
            lm_head._chronos_s_u8 = exp_u8.contiguous()
            del w_f32, amax, exp_u8, scale_f32
        del lm_bf16
        lm_head.weight.data = torch.empty((0,), dtype=lm_head.weight.dtype, device=device)
        if not hasattr(logits_proc, "_orig_apply_head_dsv41_fp4"):
            logits_proc._orig_apply_head_dsv41_fp4 = logits_proc._apply_head
            logits_proc._apply_head = types.MethodType(_chronos_fast_apply_head, logits_proc)

    gc.collect()
    torch.cuda.empty_cache()


def enable_chronos_dsv41_vllm_hook(
    default_ckpt_dir: str | Path = "/home/defnotean/pb27/hf/dsv41_chronos_v75_abliterated",
    hot_cache_size: int = 96,
    offload_tail_to_cpu: bool = True,
    mmap_zero_ram: bool = True,
) -> bool:
    """
    Patches `DeepseekV4MoE` and weight loaders in `vllm` so `DeepseekV41ForCausalLM` constructs
    `ChronosDsv41SparseMoeBlock` directly during model initialization with zero RAM spikes.
    """
    patched = False
    orig_moe_classes: dict[str, Any] = {}
    loaded_mods: list[tuple[str, Any, str]] = []
    for mod_name, cls_name in [
        ("vllm.models.deepseek_v4.nvidia.model", "DeepseekV4MoE"),
        ("vllm.models.deepseek_v41.nvidia.model", "DeepseekV4MoE"),
    ]:
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            orig_cls = getattr(mod, f"_Orig{cls_name}", getattr(mod, cls_name, None))
            if orig_cls is None:
                continue
            setattr(mod, f"_Orig{cls_name}", orig_cls)
            orig_moe_classes[mod_name] = orig_cls
            loaded_mods.append((mod_name, mod, cls_name))
        except ImportError:
            continue

    if loaded_mods:
        fallback_orig_cls = next(iter(orig_moe_classes.values()))

        class ChronosDeepseekV4MoE(ChronosDsv41SparseMoeBlock):
            _orig_moe_cls = fallback_orig_cls

            def __new__(cls, *args: Any, **kwargs: Any) -> Any:
                vllm_config = kwargs.get("vllm_config", args[0] if args else None)
                prefix = kwargs.get("prefix", args[1] if len(args) > 1 else "")
                m = re.search(r"layers\.(\d+)", str(prefix))
                layer_idx = int(m.group(1)) if m else 0
                model_dir = Path(
                    getattr(getattr(vllm_config, "model_config", None), "model", str(default_ckpt_dir))
                )
                shard_file = model_dir / f"chronos-dsv41-layer-{layer_idx:05d}-of-00040.safetensors"
                if not shard_file.is_file():
                    shard_file = Path(default_ckpt_dir) / f"chronos-dsv41-layer-{layer_idx:05d}-of-00040.safetensors"
                if (
                    shard_file.is_file()
                    and not str(prefix).startswith("mtp.")
                    and not str(prefix).startswith("dspark")
                ):
                    return super().__new__(cls)
                return cls._orig_moe_cls(*args, **kwargs)

            def __init__(self, vllm_config: Any = None, prefix: str = "", **kwargs: Any) -> None:
                m = re.search(r"layers\.(\d+)", str(prefix))
                layer_idx = int(m.group(1)) if m else 0
                model_dir = Path(
                    getattr(getattr(vllm_config, "model_config", None), "model", str(default_ckpt_dir))
                )
                shard_file = model_dir / f"chronos-dsv41-layer-{layer_idx:05d}-of-00040.safetensors"
                if not shard_file.is_file():
                    shard_file = Path(default_ckpt_dir) / f"chronos-dsv41-layer-{layer_idx:05d}-of-00040.safetensors"
                super().__init__(
                    layer_idx=layer_idx,
                    shard_path=shard_file,
                    hot_cache_size=hot_cache_size,
                    offload_tail_to_cpu=offload_tail_to_cpu,
                    mmap_zero_ram=mmap_zero_ram,
                    vllm_config=vllm_config,
                    prefix=prefix,
                )

        for mod_name, mod, cls_name in loaded_mods:
            setattr(mod, cls_name, ChronosDeepseekV4MoE)
            patched = True

            model_cls = getattr(mod, "DeepseekV4Model", None)
            if model_cls is not None and not hasattr(model_cls, "_orig_load_weights_chronos"):
                model_cls._orig_load_weights_chronos = model_cls.load_weights

                def _chronos_dsv4_load_weights(
                    self: Any, weights: Iterable[tuple[str, torch.Tensor]], _mcls: Any = model_cls
                ) -> set[str]:
                    has_chronos_ffn = any(
                        isinstance(getattr(lyr, "ffn", None), ChronosDsv41SparseMoeBlock)
                        for lyr in getattr(self, "layers", [])
                    )
                    if has_chronos_ffn:
                        filtered = (
                            (n, w)
                            for n, w in weights
                            if n.startswith("vision.") or (".ffn." not in n and ".mlp." not in n)
                        )
                        loaded = _mcls._orig_load_weights_chronos(self, filtered)
                        for lyr_idx, lyr in enumerate(getattr(self, "layers", [])):
                            ffn = getattr(lyr, "ffn", None)
                            if isinstance(ffn, ChronosDsv41SparseMoeBlock):
                                for p_name, _ in ffn.named_parameters(recurse=True):
                                    loaded.add(f"layers.{lyr_idx}.ffn.{p_name}")
                        return loaded
                    return _mcls._orig_load_weights_chronos(self, weights)

                model_cls.load_weights = _chronos_dsv4_load_weights

    # Hook safetensors_weights_iterator to skip .ffn. tensor reads on chronos-dsv41-* shards
    try:
        from vllm.model_executor.model_loader import default_loader as _dl
        from vllm.model_executor.model_loader import weight_utils as _wu

        if not getattr(_wu, "_chronos_dsv41_st_hooked", False):
            _prev_st_iter = _wu.safetensors_weights_iterator

            def _chronos_dsv41_safetensors_weights_iterator(
                hf_weights_files: list[str], *args: Any, **kwargs: Any
            ):
                is_dsv41 = any("chronos-dsv41-" in os.path.basename(f) for f in hf_weights_files)
                if not is_dsv41:
                    yield from _prev_st_iter(hf_weights_files, *args, **kwargs)
                    return

                sort_key = getattr(_wu, "_natural_sort_key", None)
                sorted_files = (
                    sorted(hf_weights_files, key=sort_key)
                    if sort_key is not None
                    else sorted(hf_weights_files)
                )
                for st_file in sorted_files:
                    bname = os.path.basename(st_file)
                    if "chronos-dsv41-upgrade2-hot96-" in bname:
                        continue
                    try:
                        with safe_open(st_file, framework="pt", device="cpu") as f:
                            for name in f.keys():
                                if not name.startswith("vision.") and (".ffn." in name or ".mlp." in name):
                                    continue
                                yield name, f.get_tensor(name)
                    finally:
                        _evict_page_cache(st_file)

            _wu.safetensors_weights_iterator = _chronos_dsv41_safetensors_weights_iterator
            _wu._chronos_dsv41_st_hooked = True
            if hasattr(_dl, "safetensors_weights_iterator"):
                _dl.safetensors_weights_iterator = _chronos_dsv41_safetensors_weights_iterator
            if hasattr(_dl, "DefaultModelLoader") and hasattr(_dl.DefaultModelLoader, "_prepare_weights"):
                _orig_prep_w = _dl.DefaultModelLoader._prepare_weights

                def _chronos_prepare_weights(self_dl: Any, *p_args: Any, **p_kwargs: Any):
                    files = _orig_prep_w(self_dl, *p_args, **p_kwargs)
                    if isinstance(files, list) and any("chronos-dsv41-" in os.path.basename(x) for x in files):
                        files = [
                            x
                            for x in files
                            if "chronos-dsv41-upgrade2-hot96-" not in os.path.basename(x)
                        ]
                    return files

                _dl.DefaultModelLoader._prepare_weights = _chronos_prepare_weights
    except Exception:
        pass

    # Patch DeepseekV4SWACache block_size=32 -> 64 on SM120 (Blackwell GB10),
    # set kv cache block_size=64 on DeepseekV41IndexerBackend & DeepseekV4FlashInferMLASparseBackend
    # (so compress_ratio=2 -> block_kv=32 and compress_ratio=1 -> block_kv=64, satisfying DeepGEMM's
    # block_kv in {32, 64} assertion), and chunk >64-token prefill calls to
    # flashinfer_trtllm_batch_decode_sparse_mla_dsv4.
    try:
        from vllm.v1.attention.backends.mla import sparse_swa as _swa_mod

        if not getattr(_swa_mod.DeepseekV4SWACache, "_chronos_sm120_pbs_patched", False):
            _orig_swa_init = _swa_mod.DeepseekV4SWACache.__init__

            def _chronos_swa_init(self: Any, *args: Any, **kwargs: Any) -> None:
                if kwargs.get("block_size") == 32:
                    kwargs["block_size"] = 64
                _orig_swa_init(self, *args, **kwargs)

            _swa_mod.DeepseekV4SWACache.__init__ = _chronos_swa_init
            _swa_mod.DeepseekV4SWACache._chronos_sm120_pbs_patched = True
    except Exception:
        pass

    try:
        from vllm.v1.attention.backends.mla import indexer as _idx_mod

        if hasattr(_idx_mod, "DeepseekV41IndexerBackend"):
            _idx_mod.DeepseekV41IndexerBackend.get_supported_kernel_block_sizes = staticmethod(
                lambda kv_cache_spec=None: [64]
            )
    except Exception:
        pass

    try:
        from vllm.model_executor.layers import sparse_attn_indexer as _sai_mod
        from vllm.utils import deep_gemm as _dg_mod

        if not getattr(_dg_mod, "_chronos_sm120_bkv32_patched", False):
            _orig_paged_mqa = _dg_mod.fp8_fp4_paged_mqa_logits

            def _chronos_paged_mqa_logits(
                q: Any,
                kv_cache: torch.Tensor,
                weights: torch.Tensor,
                context_lens: torch.Tensor,
                block_tables: torch.Tensor,
                schedule_metadata: torch.Tensor,
                max_model_len: int,
                clean_logits: bool = False,
                indices: torch.Tensor | None = None,
            ) -> torch.Tensor:
                eff_max_len = min(int(max_model_len), max(512, int(block_tables.shape[1]) * 64))
                if kv_cache.ndim == 4 and kv_cache.shape[1] == 32 and q[1] is None:
                    bsz = block_tables.shape[0]
                    max_blks = block_tables.shape[1]
                    n_blk_even = max(2, ((max_blks + 1) // 2) * 2)
                    if n_blk_even > max_blks:
                        pad_bt = torch.zeros(
                            (bsz, n_blk_even), dtype=block_tables.dtype, device=block_tables.device
                        )
                        pad_bt[:, :max_blks] = block_tables
                        bt_src = pad_bt.clamp(min=0, max=max(0, kv_cache.shape[0] - 1))
                    else:
                        bt_src = block_tables[:, :n_blk_even].clamp(
                            min=0, max=max(0, kv_cache.shape[0] - 1)
                        )
                    n_blk_64 = n_blk_even // 2
                    entry_size = kv_cache.shape[-1]
                    fp8_bytes_32 = 32 * (entry_size - 4)
                    scale_bytes_32 = 32 * 4
                    p32_pages = kv_cache[bt_src].reshape(bsz * n_blk_64, 2, 32 * entry_size)
                    p64_fp8 = p32_pages[:, :, :fp8_bytes_32].reshape(
                        bsz * n_blk_64, 2 * fp8_bytes_32
                    )
                    p64_scales = p32_pages[:, :, fp8_bytes_32:].reshape(
                        bsz * n_blk_64, 2 * scale_bytes_32
                    )
                    kvc_64 = (
                        torch.cat([p64_fp8, p64_scales], dim=-1)
                        .view(bsz * n_blk_64, 64, 1, entry_size)
                        .contiguous()
                    )
                    bt_64 = torch.arange(
                        bsz * n_blk_64, dtype=torch.int32, device=kv_cache.device
                    ).view(bsz, n_blk_64)
                    num_sms = schedule_metadata.shape[0] - 1
                    sched_64 = _dg_mod.get_paged_mqa_logits_metadata(
                        context_lens, 64, num_sms, indices=indices
                    )
                    return _orig_paged_mqa(
                        q,
                        kvc_64,
                        weights,
                        context_lens,
                        bt_64,
                        sched_64,
                        eff_max_len,
                        clean_logits=clean_logits,
                        indices=indices,
                    )
                return _orig_paged_mqa(
                    q,
                    kv_cache,
                    weights,
                    context_lens,
                    block_tables,
                    schedule_metadata,
                    eff_max_len,
                    clean_logits=clean_logits,
                    indices=indices,
                )

            _dg_mod.fp8_fp4_paged_mqa_logits = _chronos_paged_mqa_logits
            _sai_mod.fp8_fp4_paged_mqa_logits = _chronos_paged_mqa_logits
            _dg_mod._chronos_sm120_bkv32_patched = True
    except Exception:
        pass

    try:
        from vllm.models.deepseek_v41 import sparse_mla as _smla_mod
        from vllm.models.deepseek_v41.nvidia import flashinfer_sparse as _fis_mod

        if hasattr(_smla_mod, "DeepseekV4SparseMLABackend"):
            _smla_mod.DeepseekV4SparseMLABackend.get_supported_kernel_block_sizes = staticmethod(
                lambda kv_cache_spec=None: [64]
            )
        if hasattr(_fis_mod, "DeepseekV4FlashInferMLASparseBackend"):
            _fis_mod.DeepseekV4FlashInferMLASparseBackend.get_supported_kernel_block_sizes = staticmethod(
                lambda kv_cache_spec=None: [64]
            )

        if not getattr(_fis_mod, "_chronos_sm120_chunk_patched", False):
            _orig_fi_dsv4 = _fis_mod.flashinfer_trtllm_batch_decode_sparse_mla_dsv4

            def _chronos_fi_dsv4(
                *,
                query: torch.Tensor,
                swa_kv_cache: torch.Tensor,
                workspace_buffer: torch.Tensor,
                sparse_indices: torch.Tensor,
                compressed_kv_cache: torch.Tensor | None = None,
                out: torch.Tensor | None = None,
                bmm1_scale: float = 1.0,
                sinks: torch.Tensor | None = None,
                kv_layout: str = "NHD",
                swa_topk_lens: torch.Tensor | None = None,
                extra_sparse_indices: torch.Tensor | None = None,
                extra_sparse_topk_lens: torch.Tensor | None = None,
                **kwargs: Any,
            ) -> torch.Tensor:
                n_tok = query.shape[0]
                if n_tok <= 64:
                    return _orig_fi_dsv4(
                        query=query,
                        swa_kv_cache=swa_kv_cache,
                        workspace_buffer=workspace_buffer,
                        sparse_indices=sparse_indices,
                        compressed_kv_cache=compressed_kv_cache,
                        out=out,
                        bmm1_scale=bmm1_scale,
                        sinks=sinks,
                        kv_layout=kv_layout,
                        swa_topk_lens=swa_topk_lens,
                        extra_sparse_indices=extra_sparse_indices,
                        extra_sparse_topk_lens=extra_sparse_topk_lens,
                        **kwargs,
                    )
                if out is None:
                    out = torch.empty_like(query)
                for s in range(0, n_tok, 64):
                    e = min(s + 64, n_tok)
                    _orig_fi_dsv4(
                        query=query[s:e],
                        swa_kv_cache=swa_kv_cache,
                        workspace_buffer=workspace_buffer,
                        sparse_indices=sparse_indices[s:e],
                        compressed_kv_cache=compressed_kv_cache,
                        out=out[s:e],
                        bmm1_scale=bmm1_scale,
                        sinks=sinks,
                        kv_layout=kv_layout,
                        swa_topk_lens=swa_topk_lens[s:e] if swa_topk_lens is not None else None,
                        extra_sparse_indices=(
                            extra_sparse_indices[s:e] if extra_sparse_indices is not None else None
                        ),
                        extra_sparse_topk_lens=(
                            extra_sparse_topk_lens[s:e] if extra_sparse_topk_lens is not None else None
                        ),
                        **kwargs,
                    )
                return out

            _fis_mod.flashinfer_trtllm_batch_decode_sparse_mla_dsv4 = _chronos_fi_dsv4
            _fis_mod._chronos_sm120_chunk_patched = True
    except Exception:
        pass

    try:
        from flashinfer.mla import _sparse_mla_sm120 as _fi_sm120
        from flashinfer.mla import _sparse_mla_sm120_cpb as _fi_cpb
        from flashinfer.mla import _sparse_mla_sm120_plan as _fi_plan

        _fi_plan.prefill_mg_dual_eligible = lambda *args, **kwargs: False
        _fi_cpb.get_decode_max_tokens = lambda *args, **kwargs: 64
        _fi_cpb.is_calibration_failed = lambda *args, **kwargs: True
        _fi_cpb.is_crossover_failed = lambda *args, **kwargs: True
        _fi_plan._resolve_cpb = lambda *args, **kwargs: 1
        _fi_sm120._resolve_cpb = lambda *args, **kwargs: 1
        _fi_plan._plan_memo.clear()
        _fi_plan._cpb_hot_cache.clear()
    except Exception:
        pass

    # Prevent EmulationMxfp8LinearKernel from dequantizing wo_a to 2.68 GB of BF16 at load time,
    # stash raw unswizzled MXFP8 scales on linear layers before swizzling, and install fused
    # Triton FP4/MXFP8 decode accelerators right after process_weights_after_loading.
    os.environ.setdefault("VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD", "0")
    try:
        from vllm.model_executor.kernels.linear.mxfp8.emulation import EmulationMxfp8LinearKernel
        from vllm.model_executor.kernels.linear.mxfp8.flashinfer import FlashInferCutlassMxfp8LinearKernel

        if not getattr(EmulationMxfp8LinearKernel, "_chronos_pw_hooked", False):
            _orig_em_pw = EmulationMxfp8LinearKernel.process_weights_after_loading

            def _chronos_em_pw(self_k: Any, layer: nn.Module) -> None:
                self_k.dequant_at_load = False
                if hasattr(layer, "weight_scale") and hasattr(layer, "weight"):
                    w_flat = layer.weight.data.view(-1, layer.weight.shape[-1])
                    n_r, k_c = w_flat.shape
                    layer._chronos_raw_scale_u8 = layer.weight_scale.data[:n_r, : k_c // 32].clone().contiguous()
                _orig_em_pw(self_k, layer)

            EmulationMxfp8LinearKernel.process_weights_after_loading = _chronos_em_pw
            EmulationMxfp8LinearKernel._chronos_pw_hooked = True

        if not getattr(FlashInferCutlassMxfp8LinearKernel, "_chronos_pw_hooked", False):
            _orig_fi_pw = FlashInferCutlassMxfp8LinearKernel.process_weights_after_loading

            def _chronos_fi_pw(self_k: Any, layer: nn.Module) -> None:
                if hasattr(layer, "weight_scale") and hasattr(layer, "weight"):
                    n_r, k_c = layer.weight.shape
                    layer._chronos_raw_scale_u8 = layer.weight_scale.data[:n_r, : k_c // 32].clone().contiguous()
                _orig_fi_pw(self_k, layer)

            FlashInferCutlassMxfp8LinearKernel.process_weights_after_loading = _chronos_fi_pw
            FlashInferCutlassMxfp8LinearKernel._chronos_pw_hooked = True
    except Exception:
        pass

    try:
        from vllm.model_executor.model_loader import base_loader as _bl_mod

        if not getattr(_bl_mod.BaseModelLoader, "_chronos_load_model_hooked", False):
            _orig_load_model = _bl_mod.BaseModelLoader.load_model

            def _chronos_load_model(
                self_l: Any, vllm_config: Any, model_config: Any, prefix: str = ""
            ) -> nn.Module:
                model = _orig_load_model(self_l, vllm_config, model_config, prefix=prefix)
                if not str(prefix).startswith("mtp.") and not str(prefix).startswith("dspark"):
                    _install_chronos_dsv41_decode_accelerators(model)
                return model

            _bl_mod.BaseModelLoader.load_model = _chronos_load_model
            _bl_mod.BaseModelLoader._chronos_load_model_hooked = True
    except Exception:
        pass

    try:
        from vllm.model_executor.warmup import kernel_warmup as _kw_mod

        if hasattr(_kw_mod, "_flashinfer_autotune_token_counts"):
            _kw_mod._flashinfer_autotune_token_counts = lambda runner, include_bf16=False: (16,)
    except Exception:
        pass

    try:
        from vllm.tokenizers import deepseek_v41_encoding as _dsv41_enc

        _def_eff = int(os.environ.get("CHRONOS_DEFAULT_REASONING_EFFORT", "20"))
        if hasattr(_dsv41_enc, "REASONING_EFFORT_MAPPINGS"):
            _dsv41_enc.REASONING_EFFORT_MAPPINGS["low"] = _def_eff
            _dsv41_enc.REASONING_EFFORT_MAPPINGS["high"] = _def_eff
            _dsv41_enc.REASONING_EFFORT_MAPPINGS["xhigh"] = _def_eff
    except Exception:
        pass

    # Ensure DeepseekV4Attention uses the wide eager region (_prepare_and_attn_eager) under PIECEWISE CUDAGraphs
    # and populates topk_indices for short context (<= 2048 tokens)
    os.environ.setdefault("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    _fixed_sia_fn = None
    try:
        from vllm.compilation.breakable_cudagraph import eager_break_during_capture
        from vllm.forward_context import get_forward_context
        from vllm.triton_utils import triton
        from vllm.models.deepseek_v41.attention import (
            DeepseekV4Attention as _V41Attn,
            _fill_short_context_topk_indices as _triton_fill_short_topk,
        )

        if not getattr(_V41Attn, "_chronos_v41_short_ctx_patched", False):
            def _fixed_sparse_indexer_and_attn(
                self_a: Any,
                hidden_states: torch.Tensor,
                index_q: torch.Tensor | None,
                index_q_scale: torch.Tensor | None,
                index_weights: torch.Tensor | None,
                q: torch.Tensor,
                kv: torch.Tensor,
                positions: torch.Tensor,
                out: torch.Tensor,
            ) -> None:
                if self_a.indexer is not None:
                    attn_metadata = get_forward_context().attn_metadata
                    handled_short = False
                    if isinstance(attn_metadata, dict):
                        indexer_metadata = attn_metadata.get(self_a.indexer.k_cache.prefix)
                        if indexer_metadata is not None and (
                            indexer_metadata.max_seq_len // self_a.indexer.compress_ratio
                            <= self_a.indexer.topk_tokens
                        ):
                            num_tokens = (
                                indexer_metadata.num_decode_tokens + indexer_metadata.num_prefill_tokens
                            )
                            if num_tokens > 0 and self_a.indexer.topk_indices_buffer is not None:
                                _triton_fill_short_topk[(num_tokens,)](
                                    self_a.indexer.topk_indices_buffer,
                                    positions,
                                    TOP_K=self_a.indexer.topk_tokens,
                                    COMPRESS_RATIO=self_a.indexer.compress_ratio,
                                    PADDED_TOP_K=triton.next_power_of_2(self_a.indexer.topk_tokens),
                                    num_warps=8,
                                )
                            handled_short = True
                    if not handled_short and index_q is not None:
                        assert index_weights is not None
                        q_quant = (index_q, index_q_scale) if index_q_scale is not None else index_q
                        self_a.indexer.indexer_op(
                            hidden_states,
                            q_quant,
                            None,
                            index_weights,
                        )
                self_a.forward_mqa(q, kv, positions, out)

            _fixed_sia_fn = _fixed_sparse_indexer_and_attn
            _V41Attn._sparse_indexer_and_attn = eager_break_during_capture(_fixed_sparse_indexer_and_attn)
            _V41Attn._chronos_v41_short_ctx_patched = True
    except Exception:
        pass

    # Auto-enable PIECEWISE CUDAGraph capture & Triton anti-loop/think-budget breaker on GPUModelRunner
    try:
        from vllm.config import CUDAGraphMode
        from vllm.v1.worker.gpu import model_runner as _mr_mod
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner
        from vllm.models.deepseek_v41.nvidia.model import DeepseekV41LLMForCausalLM
        import chronos_1t.models.antiloop_think_breaker as _atb

        if not hasattr(GPUModelRunner, "_orig_exec_pw_dsv41"):
            GPUModelRunner._orig_exec_pw_dsv41 = GPUModelRunner.execute_model

            def _dsv41_exec_pw(self_mr: Any, *args: Any, **kwargs: Any) -> Any:
                _mr_mod._LIVE_RUNNER = self_mr
                if not getattr(self_mr, "_chronos_pw_filtered", False):
                    try:
                        cgm = getattr(self_mr, "cudagraph_manager", None)
                        if cgm is not None and getattr(cgm, "_candidates", None):
                            for k, v in list(cgm._candidates.items()):
                                pw = [d for d in v if d.cg_mode == CUDAGraphMode.PIECEWISE]
                                if pw:
                                    cgm._candidates[k] = pw
                            bcg = getattr(cgm, "breakable_cg_runner", None)
                            if bcg is not None and _fixed_sia_fn is not None:
                                for entry in getattr(bcg, "entries", {}).values():
                                    if getattr(entry, "capture", None) is not None:
                                        for seg in entry.capture.segments:
                                            for c in (getattr(seg, "__closure__", None) or []):
                                                val = c.cell_contents
                                                if callable(val) and getattr(val, "__name__", "") in (
                                                    "_sparse_indexer_and_attn",
                                                    "_fixed_sparse_indexer_and_attn",
                                                ):
                                                    c.cell_contents = _fixed_sia_fn
                            for _kvc in getattr(self_mr, "kv_caches", []):
                                if isinstance(_kvc, torch.Tensor) and _kvc.numel() > 0:
                                    _kvc.zero_()
                            self_mr._chronos_pw_filtered = True
                    except Exception:
                        pass
                return GPUModelRunner._orig_exec_pw_dsv41(self_mr, *args, **kwargs)

            GPUModelRunner.execute_model = _dsv41_exec_pw

        if not hasattr(GPUModelRunner, "_orig_sample_dsv41"):
            GPUModelRunner._orig_sample_dsv41 = GPUModelRunner.sample

            def _dsv41_sample(self_mr: Any, hidden_states: torch.Tensor, input_batch: Any, *args: Any, **kwargs: Any) -> Any:
                lm_mod = getattr(self_mr.model, "language_model", self_mr.model)
                try:
                    import importlib as _imp
                    _atb_p = "/pb27_vllm/chronos_1t/models/antiloop_think_breaker.py"
                    if os.path.exists(_atb_p):
                        _atb_mt = os.path.getmtime(_atb_p)
                        if getattr(self_mr, "_atb_mt", 0.0) != _atb_mt:
                            self_mr._atb_mt = _atb_mt
                            _imp.reload(_atb)
                    lm_mod._sample_input_ids = input_batch.input_ids[input_batch.logits_indices]
                    lm_mod._sample_batch = input_batch
                    lm_mod._sample_req_states = self_mr.req_states
                except Exception:
                    pass
                try:
                    return GPUModelRunner._orig_sample_dsv41(self_mr, hidden_states, input_batch, *args, **kwargs)
                finally:
                    lm_mod._sample_batch = None

            GPUModelRunner.sample = _dsv41_sample

        if not hasattr(DeepseekV41LLMForCausalLM, "_orig_compute_logits_dsv41"):
            DeepseekV41LLMForCausalLM._orig_compute_logits_dsv41 = DeepseekV41LLMForCausalLM.compute_logits

            def _dsv41_compute_logits(self_lm: Any, hidden_states: torch.Tensor) -> torch.Tensor:
                logits = self_lm.logits_processor(self_lm.lm_head, hidden_states)
                try:
                    s_batch = getattr(self_lm, "_sample_batch", None)
                    s_states = getattr(self_lm, "_sample_req_states", None)
                    s_ids = getattr(self_lm, "_sample_input_ids", None)
                    if logits is not None and s_batch is not None and s_states is not None and s_ids is not None:
                        budget = int(os.environ.get("CHRONOS_THINK_BUDGET", "192"))
                        _atb.apply_antiloop_and_think_breaker(
                            logits, s_batch, s_states, s_ids, think_budget=budget
                        )
                except Exception:
                    pass
                return logits

            DeepseekV41LLMForCausalLM.compute_logits = _dsv41_compute_logits
    except Exception:
        pass

    return patched


def install_chronos_moe_into_dsv41(
    model: nn.Module,
    ckpt_dir: str | Path = "/home/defnotean/pb27/hf/dsv41_chronos_v75_abliterated",
    layers: Iterable[int] | None = None,
    hot_cache_size: int = 96,
    offload_tail_to_cpu: bool = True,
    mmap_zero_ram: bool = False,
    device: torch.device | None = None,
) -> list[int]:
    """
    Swap `layer.ffn` (or `layer.mlp`) in `DeepseekV41ForCausalLM` / reference `Transformer` with
    `ChronosDsv41SparseMoeBlock` for every converted `chronos-dsv41-layer-XXXXX-of-00040.safetensors` shard (`0..39`).
    """
    ckpt_path = Path(ckpt_dir)
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    lm = getattr(model, "language_model", model)
    model_inner = getattr(lm, "model", lm)
    layer_list = getattr(model_inner, "layers", None)
    if layer_list is None:
        raise ValueError("Could not locate `.layers` on provided DeepSeek-V4.1-Flash-1017 model instance.")

    target_layers = list(range(len(layer_list))) if layers is None else list(layers)
    installed: list[int] = []

    for l_idx in target_layers:
        shard_file = ckpt_path / f"chronos-dsv41-layer-{l_idx:05d}-of-00040.safetensors"
        if not shard_file.is_file():
            continue
        chronos_block = ChronosDsv41SparseMoeBlock(
            layer_idx=l_idx,
            shard_path=shard_file,
            device=device,
            hot_cache_size=hot_cache_size,
            offload_tail_to_cpu=offload_tail_to_cpu,
            mmap_zero_ram=mmap_zero_ram,
        )
        if hasattr(layer_list[l_idx], "ffn"):
            layer_list[l_idx].ffn = chronos_block
        else:
            layer_list[l_idx].mlp = chronos_block
        installed.append(l_idx)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return installed
