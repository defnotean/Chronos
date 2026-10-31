#!/usr/bin/env python3
"""
Chronos-1T x GLM-5.3-Flash (`zai-org/GLM-5.3-Flash`) — Runtime Loader & vLLM Hook for `Glm5NextForConditionalGeneration`.

Provides:
1. `enable_chronos_glm5_vllm_hook()`:
   Patches GLM-5 / GLM-4 MoE sparse block classes before model construction so that layers `3..45`
   directly instantiate `ChronosGlm5SparseMoeBlock` whenever a converted
   `chronos-glm5-layer-XXXXX-of-00046.safetensors` shard exists in `vllm_config.model_config.model`,
   without allocating the `7.27 GB/layer` stock `FP8` expert tensors first.
2. `install_chronos_moe_into_glm5(...)`:
   Swaps `layer.mlp` on an already-constructed `Glm5NextForConditionalGeneration` instance with
   `ChronosGlm5SparseMoeBlock` for all converted MoE layers (`3..45`).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from chronos_1t.models.glm5_chronos_moe import ChronosGlm5SparseMoeBlock


def enable_chronos_glm5_vllm_hook(
    default_ckpt_dir: str | Path = "/home/defnotean/pb27/hf/glm53_flash_chronos_v6",
    hot_cache_size: int = 16,
    offload_tail_to_cpu: bool = True,
) -> bool:
    """
    Patches GLM-5 MoE classes in `vllm` (if present) so `Glm5NextDecoderLayer` constructs
    `ChronosGlm5SparseMoeBlock` directly during model initialization.
    """
    patched = False
    for mod_name, cls_name in [
        ("vllm.model_executor.models.glm4_moe", "Glm4MoE"),
        ("vllm.model_executor.models.deepseek_v2", "DeepseekV2MoE"),
    ]:
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            orig_cls = getattr(mod, f"_Orig{cls_name}", getattr(mod, cls_name, None))
            if orig_cls is None:
                continue
            setattr(mod, f"_Orig{cls_name}", orig_cls)

            def _factory(vllm_config: Any = None, prefix: str = "", _orig: Any = orig_cls, **kwargs: Any) -> nn.Module:
                m = re.search(r"layers\.(\d+)", prefix)
                layer_idx = int(m.group(1)) if m else 3
                model_dir = Path(getattr(getattr(vllm_config, "model_config", None), "model", str(default_ckpt_dir)))
                shard_file = model_dir / f"chronos-glm5-layer-{layer_idx:05d}-of-00046.safetensors"
                if not shard_file.is_file():
                    shard_file = Path(default_ckpt_dir) / f"chronos-glm5-layer-{layer_idx:05d}-of-00046.safetensors"
                if shard_file.is_file():
                    return ChronosGlm5SparseMoeBlock(
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


def install_chronos_moe_into_glm5(
    model: nn.Module,
    ckpt_dir: str | Path = "/home/defnotean/pb27/hf/glm53_flash_chronos_v6",
    layers: Iterable[int] | None = None,
    hot_cache_size: int = 16,
    offload_tail_to_cpu: bool = True,
    device: torch.device | None = None,
) -> list[int]:
    """
    Swap `layer.mlp` in `Glm5NextForConditionalGeneration` with `ChronosGlm5SparseMoeBlock`
    for every converted `chronos-glm5-layer-XXXXX-of-00046.safetensors` shard (`layers 3..45`).
    """
    ckpt_path = Path(ckpt_dir)
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    lm = getattr(model, "language_model", model)
    model_inner = getattr(lm, "model", lm)
    layer_list = getattr(model_inner, "layers", None)
    if layer_list is None:
        raise ValueError("Could not locate `.layers` on provided GLM-5.3-Flash model instance.")

    target_layers = list(range(3, len(layer_list))) if layers is None else list(layers)
    installed: list[int] = []

    for l_idx in target_layers:
        shard_file = ckpt_path / f"chronos-glm5-layer-{l_idx:05d}-of-00046.safetensors"
        if not shard_file.is_file():
            continue
        chronos_block = ChronosGlm5SparseMoeBlock(
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
