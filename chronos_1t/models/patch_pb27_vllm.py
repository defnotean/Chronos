#!/usr/bin/env python3
"""
Chronos-1T x PseudoBrain (pb27) — Runtime Loader & Monkey-Patcher for `PB27Qwen4ExpForConditionalGeneration`.

Provides `install_chronos_moe_into_pb27(...)` to swap `Qwen4ExpSparseMoeBlock` layers in
`PB27Qwen4ExpForConditionalGeneration` with `ChronosQwen4ExpSparseMoeBlock` backed by
`/home/defnotean/pb27/hf/flash_next_chronos_v6/chronos-layer-XXXXX-of-00048.safetensors`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch import nn

from chronos_1t.models.vllm_chronos_moe import ChronosQwen4ExpSparseMoeBlock


def install_chronos_moe_into_pb27(
    model: nn.Module,
    ckpt_dir: str | Path = "/home/defnotean/pb27/hf/flash_next_chronos_v6",
    layers: Iterable[int] | None = None,
    hot_cache_size: int = 64,
    device: torch.device | None = None,
) -> list[int]:
    """
    Swap `layer.mlp` in `PB27Qwen4ExpForConditionalGeneration` (or `Qwen4ExpModel`)
    with `ChronosQwen4ExpSparseMoeBlock` for every converted `chronos-layer-XXXXX-of-00048.safetensors` shard.
    """
    ckpt_path = Path(ckpt_dir)
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Resolve language_model.layers
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
        )
        layer_list[l_idx].mlp = chronos_block
        installed.append(l_idx)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return installed
