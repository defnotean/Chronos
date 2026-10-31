#!/usr/bin/env python3
"""
Chronos-1T x PseudoBrain (pb27) — Runtime Loader & vLLM Hook for `PB27Qwen4ExpForConditionalGeneration`.

Provides:
1. `enable_chronos_vllm_hook()`:
   Patches `pb27_vllm.qwen4_exp_model.Qwen4ExpSparseMoeBlock` before model construction so that
   `PB27Qwen4ExpDecoderLayer` directly instantiates `ChronosQwen4ExpSparseMoeBlock` whenever a
   converted `chronos-layer-XXXXX-of-00048.safetensors` shard exists in `vllm_config.model_config.model`,
   without allocating the stock `Qwen4ExpSparseMoeBlock` first.
2. `install_chronos_moe_into_pb27(...)`:
   Swaps `layer.mlp` on an already-constructed `PB27Qwen4ExpForConditionalGeneration` (or `Qwen4ExpModel`)
   instance with `ChronosQwen4ExpSparseMoeBlock`.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from chronos_1t.models.vllm_chronos_moe import ChronosQwen4ExpSparseMoeBlock


def enable_chronos_vllm_hook(
    default_ckpt_dir: str | Path = "/home/defnotean/pb27/hf/flash_next_chronos_v6",
    hot_cache_size: int = 24,
    offload_tail_to_cpu: bool = True,
) -> bool:
    """
    Patches `pb27_vllm.qwen4_exp_model.Qwen4ExpSparseMoeBlock` so `PB27Qwen4ExpDecoderLayer`
    constructs `ChronosQwen4ExpSparseMoeBlock` directly during `PB27Qwen4ExpForConditionalGeneration.__init__`.
    """
    try:
        import pb27_vllm.qwen4_exp_model as qwen4_mod
    except ImportError:
        return False

    orig_cls = getattr(qwen4_mod, "_OrigQwen4ExpSparseMoeBlock", qwen4_mod.Qwen4ExpSparseMoeBlock)
    setattr(qwen4_mod, "_OrigQwen4ExpSparseMoeBlock", orig_cls)

    def _chronos_or_stock_moe_factory(vllm_config: Any, prefix: str = "", **kwargs: Any) -> nn.Module:
        m = re.search(r"layers\.(\d+)", prefix)
        layer_idx = int(m.group(1)) if m else 0
        model_dir = Path(getattr(vllm_config.model_config, "model", str(default_ckpt_dir)))
        shard_file = model_dir / f"chronos-layer-{layer_idx:05d}-of-00048.safetensors"
        if not shard_file.is_file():
            shard_file = Path(default_ckpt_dir) / f"chronos-layer-{layer_idx:05d}-of-00048.safetensors"
        if shard_file.is_file():
            return ChronosQwen4ExpSparseMoeBlock(
                layer_idx=layer_idx,
                shard_path=shard_file,
                hot_cache_size=hot_cache_size,
                offload_tail_to_cpu=offload_tail_to_cpu,
                vllm_config=vllm_config,
                prefix=prefix,
            )
        return orig_cls(vllm_config=vllm_config, prefix=prefix, **kwargs)

    qwen4_mod.Qwen4ExpSparseMoeBlock = _chronos_or_stock_moe_factory
    return True


def install_chronos_moe_into_pb27(
    model: nn.Module,
    ckpt_dir: str | Path = "/home/defnotean/pb27/hf/flash_next_chronos_v6",
    layers: Iterable[int] | None = None,
    hot_cache_size: int = 24,
    offload_tail_to_cpu: bool = True,
    device: torch.device | None = None,
) -> list[int]:
    """
    Swap `layer.mlp` in `PB27Qwen4ExpForConditionalGeneration` (or `Qwen4ExpModel`)
    with `ChronosQwen4ExpSparseMoeBlock` for every converted `chronos-layer-XXXXX-of-00048.safetensors` shard.
    """
    ckpt_path = Path(ckpt_dir)
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    lm = getattr(model, "language_model", model)
    model_inner = getattr(lm, "model", lm)
    layer_list = getattr(model_inner, "layers", None)
    if layer_list is None:
        raise ValueError("Could not locate `.layers` on provided model instance.")

    target_layers = list(range(len(layer_list))) if layers is None else list(layers)
    installed: list[int] = []

    for l_idx in target_layers:
        shard_file = ckpt_path / f"chronos-layer-{l_idx:05d}-of-00048.safetensors"
        if not shard_file.is_file():
            continue
        chronos_block = ChronosQwen4ExpSparseMoeBlock(
            layer_idx=l_idx,
            shard_path=shard_file,
            device=device,
            hot_cache_size=hot_cache_size,
            offload_tail_to_cpu=offload_tail_to_cpu,
        )
        layer_list[l_idx].mlp = chronos_block
        installed.append(l_idx)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return installed
