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
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
from safetensors import safe_open

from chronos_1t.models.dsv41_chronos_moe import ChronosDsv41SparseMoeBlock


def _evict_page_cache(file_path: str) -> None:
    gc.collect()
    if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
        try:
            fd = os.open(file_path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except OSError:
            pass


def enable_chronos_dsv41_vllm_hook(
    default_ckpt_dir: str | Path = "/home/defnotean/pb27/hf/dsv41_chronos_v75_abliterated",
    hot_cache_size: int = 16,
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
                        extra_sparse_topk_lens=extra_sparse_lengths if False else extra_sparse_topk_lens,
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
        from flashinfer.mla import _sparse_mla_sm120_cpb as _fi_cpb
        from flashinfer.mla import _sparse_mla_sm120_plan as _fi_plan

        _fi_plan.prefill_mg_dual_eligible = lambda *args, **kwargs: False
        _fi_cpb.get_decode_max_tokens = lambda *args, **kwargs: 64
        _fi_cpb.is_calibration_failed = lambda *args, **kwargs: True
        _fi_cpb.is_crossover_failed = lambda *args, **kwargs: True
        _fi_plan._plan_memo.clear()
    except Exception:
        pass

    return patched


def install_chronos_moe_into_dsv41(
    model: nn.Module,
    ckpt_dir: str | Path = "/home/defnotean/pb27/hf/dsv41_chronos_v75_abliterated",
    layers: Iterable[int] | None = None,
    hot_cache_size: int = 16,
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
