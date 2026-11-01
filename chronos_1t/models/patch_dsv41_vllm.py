#!/usr/bin/env python3
"""
Chronos-1T x DeepSeek-V4.1-Flash-1017 (`deepseek-ai/DeepSeek-V4.1-Flash-1017` / Abliterated)
Runtime Loader & vLLM Hook for `DeepseekV41ForCausalLM`.

Provides:
1. `enable_chronos_dsv41_vllm_hook()`:
   Patches `vllm.models.deepseek_v4.nvidia.model.DeepseekV4MoE` before model construction so that layers `0..39`
   directly instantiate `ChronosDsv41SparseMoeBlock` whenever a converted
   `chronos-dsv41-layer-XXXXX-of-00040.safetensors` shard exists in `vllm_config.model_config.model`.
2. `install_chronos_moe_into_dsv41(...)`:
   Swaps `layer.ffn` / `layer.mlp` on an already-constructed `DeepseekV41ForCausalLM` or reference `Transformer`
   instance with `ChronosDsv41SparseMoeBlock` for all converted MoE layers (`0..39`).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from chronos_1t.models.dsv41_chronos_moe import ChronosDsv41SparseMoeBlock


def enable_chronos_dsv41_vllm_hook(
    default_ckpt_dir: str | Path = "/content/dsv41_chronos_v75",
    hot_cache_size: int = 16,
    offload_tail_to_cpu: bool = True,
) -> bool:
    """
    Patches `DeepseekV4MoE` in `vllm` (if present) so `DeepseekV4DecoderLayer` constructs
    `ChronosDsv41SparseMoeBlock` directly during model initialization.
    """
    patched = False
    for mod_name, cls_name in [
        ("vllm.models.deepseek_v4.nvidia.model", "DeepseekV4MoE"),
    ]:
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            orig_cls = getattr(mod, f"_Orig{cls_name}", getattr(mod, cls_name, None))
            if orig_cls is None:
                continue
            setattr(mod, f"_Orig{cls_name}", orig_cls)

            def _factory(vllm_config: Any = None, prefix: str = "", _orig: Any = orig_cls, **kwargs: Any) -> nn.Module:
                m = re.search(r"layers\.(\d+)", prefix)
                layer_idx = int(m.group(1)) if m else 0
                model_dir = Path(getattr(getattr(vllm_config, "model_config", None), "model", str(default_ckpt_dir)))
                shard_file = model_dir / f"chronos-dsv41-layer-{layer_idx:05d}-of-00040.safetensors"
                if not shard_file.is_file():
                    shard_file = Path(default_ckpt_dir) / f"chronos-dsv41-layer-{layer_idx:05d}-of-00040.safetensors"
                if shard_file.is_file() and not prefix.startswith("mtp.") and not prefix.startswith("dspark"):
                    return ChronosDsv41SparseMoeBlock(
                        layer_idx=layer_idx,
                        shard_path=shard_file,
                        hot_cache_size=hot_cache_size,
                        offload_tail_to_cpu=offload_tail_to_cpu,
                        vllm_config=vllm_config,
                        prefix=prefix,
                    )
                return _orig(vllm_config=vllm_config, prefix=prefix, **kwargs)

            setattr(mod, cls_name, _factory)
            patched = True
        except ImportError:
            continue
    return patched


def install_chronos_moe_into_dsv41(
    model: nn.Module,
    ckpt_dir: str | Path = "/content/dsv41_chronos_v75",
    layers: Iterable[int] | None = None,
    hot_cache_size: int = 16,
    offload_tail_to_cpu: bool = True,
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
        )
        if hasattr(layer_list[l_idx], "ffn"):
            layer_list[l_idx].ffn = chronos_block
        else:
            layer_list[l_idx].mlp = chronos_block
        installed.append(l_idx)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return installed
