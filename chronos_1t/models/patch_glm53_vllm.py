#!/usr/bin/env python3
"""
Chronos-1T x GLM-5.3 (`GlmMoeDsaForCausalLM`) — vLLM Hook for the DGX Spark (`pb27-vllm`, vLLM v0.31, sm_121).

vLLM v0.31 serves `glm_moe_dsa` through `vllm.models.deepseek_v32` (DSA sparse MLA). Its decoder layer constructs
`DeepseekV2MoE(config=..., parallel_config=..., quant_config=..., reduce_results=False, prefix=..., ...)` from the
module-global name in `vllm.models.deepseek_v32.nvidia.model` (also used by the MTP draft layer), so this hook:

  1. swaps that name (and the generic `deepseek_v2` one) for `ChronosGlm53VllmMoE`, which builds a
     `ChronosGlm53SparseMoeBlock` whenever `chronos-glm53-layer-XXXXX-of-00079.safetensors` exists in the model
     directory and otherwise transparently constructs the original class (stock GLM-5.x / DeepSeek-V3.2 untouched);
  2. routes the Chronos forward through an opaque `torch.library` custom op so `torch.compile` does not trace the
     tier dispatch, while the op itself stays CUDA-graph capturable for decode batches (no host syncs);
  3. wraps `safetensors_weights_iterator` so MoE-layer `.mlp.{experts,chronos_groups,shared_experts,gate}` tensors in
     `chronos-glm53-*` shards are never read by vLLM's loader (the block reads them itself, zero duplicate RAM) and
     evicts each shard from the page cache after use (unified-memory friendly, like the DSV4.1 hook).

All residency decisions come from `glm53_chronos_spec.plan_residency` (profile `spark_gb10` = 108 GB container cap);
override with `CHRONOS_GLM53_T{1,2,3}_SLOTS`, `CHRONOS_GLM53_CTX`, `CHRONOS_GLM53_PROFILE`.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any

import torch

from chronos_1t.models.glm53_chronos_spec import (
    Glm53Arch,
    is_glm53_chronos_file,
    is_hot_sidecar,
    layer_shard_name,
    plan_residency,
    detect_profile_name,
)

_SKIP_MLP = re.compile(r"\.mlp\.(chronos_groups|experts|shared_experts|hot_experts)\.|\.mlp\.gate\.(weight|e_score_correction_bias)$")
_ARCH_CACHE: dict[str, Glm53Arch] = {}
_OP_REGISTERED = False


def _current_vllm_config() -> Any:
    try:
        from vllm.config import get_current_vllm_config

        return get_current_vllm_config()
    except Exception:
        return None


def _model_dir(default: str | Path | None) -> Path:
    cfg = _current_vllm_config()
    model = getattr(getattr(cfg, "model_config", None), "model", None)
    if model and Path(model).is_dir():
        return Path(model)
    return Path(os.environ.get("CHRONOS_GLM53_MODEL_DIR", str(default or ".")))


def _arch(model_dir: Path) -> Glm53Arch | None:
    key = str(model_dir)
    if key not in _ARCH_CACHE:
        cfg = model_dir / "config.json"
        if not cfg.exists():
            return None
        try:
            _ARCH_CACHE[key] = Glm53Arch.from_json_file(cfg)
        except Exception:
            return None
    return _ARCH_CACHE[key]


def _layer_idx(prefix: str) -> int | None:
    m = re.search(r"layers\.(\d+)", str(prefix))
    return int(m.group(1)) if m else None


def _register_custom_op() -> None:
    global _OP_REGISTERED
    if _OP_REGISTERED:
        return
    from chronos_1t.models.glm53_chronos_moe import LIVE_BLOCKS

    @torch.library.custom_op("chronos_glm53::moe_forward", mutates_args=())
    def _moe_forward(hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        return LIVE_BLOCKS[layer_idx].forward_impl(hidden_states)

    @_moe_forward.register_fake
    def _(hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        return torch.empty_like(hidden_states)

    _OP_REGISTERED = True


def enable_chronos_glm53_vllm_hook(default_ckpt_dir: str | Path | None = None) -> bool:
    from chronos_1t.models.glm53_chronos_moe import ChronosGlm53SparseMoeBlock

    patched = False
    sites: list[tuple[Any, str]] = []
    for mod_name in (
        "vllm.models.deepseek_v32.nvidia.model",
        "vllm.models.deepseek_v32.nvidia.mtp",
        "vllm.models.deepseek_v32.amd.model",
        "vllm.model_executor.models.deepseek_v2",
    ):
        try:
            mod = __import__(mod_name, fromlist=["DeepseekV2MoE"])
        except Exception:
            continue
        if getattr(mod, "DeepseekV2MoE", None) is not None:
            sites.append((mod, mod_name))
    if not sites:
        return False

    # Resolve the genuine vLLM class. The older GLM-5.3-Flash hook (`patch_glm5_vllm`) replaces
    # `deepseek_v2.DeepseekV2MoE` with a plain factory function (stored original: `_OrigDeepseekV2MoE`); a function
    # would break `isinstance(layer.mlp, DeepseekV2MoE)` in vLLM's `set_moe_parameters`, so never fall back to it.
    orig_cls = None
    for mod, _ in sites:
        for cand in (getattr(mod, "_ChronosGlm53OrigDeepseekV2MoE", None), getattr(mod, "_OrigDeepseekV2MoE", None),
                     getattr(mod, "DeepseekV2MoE", None)):
            if isinstance(cand, type) and getattr(cand, "_chronos_glm53", False):
                return True
            if isinstance(cand, type):
                orig_cls = cand
                break
        if orig_cls is not None:
            break
    if orig_cls is None:
        return False
    _register_custom_op()

    class ChronosGlm53VllmMoE(ChronosGlm53SparseMoeBlock):
        _chronos_glm53 = True
        _orig_moe_cls = orig_cls

        def __new__(cls, *args: Any, **kwargs: Any) -> Any:
            prefix = kwargs.get("prefix", "")
            l_idx = _layer_idx(prefix)
            mdir = _model_dir(default_ckpt_dir)
            arch = _arch(mdir)
            if (
                l_idx is not None and arch is not None and l_idx in arch.moe_layer_ids
                and (mdir / layer_shard_name(l_idx, arch)).is_file()
            ):
                return super().__new__(cls)
            return cls._orig_moe_cls(*args, **kwargs)

        def __init__(self, config: Any = None, parallel_config: Any = None, quant_config: Any = None,
                     reduce_results: bool = True, prefix: str = "", apply_routed_scale_to_output: bool = False,
                     **kwargs: Any) -> None:
            mdir = _model_dir(default_ckpt_dir)
            arch = _arch(mdir)
            l_idx = _layer_idx(prefix)
            vcfg = _current_vllm_config()
            include_mtp = getattr(vcfg, "speculative_config", None) is not None
            ctx = int(os.environ.get("CHRONOS_GLM53_CTX", str(getattr(getattr(vcfg, "model_config", None), "max_model_len", 32768) or 32768)))
            plan = plan_residency(arch, detect_profile_name(), ctx_tokens=min(ctx, 262144), include_mtp=include_mtp)
            super().__init__(layer_idx=l_idx, shard_path=mdir / layer_shard_name(l_idx, arch), arch=arch, plan=plan)

        def forward(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            return torch.ops.chronos_glm53.moe_forward(hidden_states, self.layer_idx)

    for mod, _ in sites:
        setattr(mod, "_ChronosGlm53OrigDeepseekV2MoE", orig_cls)
        setattr(mod, "DeepseekV2MoE", ChronosGlm53VllmMoE)
        patched = True

    _hook_weight_iterators()
    return patched


def _evict(path: str) -> None:
    if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
        try:
            fd = os.open(os.path.realpath(path), os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        except OSError:
            pass


def _hook_weight_iterators() -> None:
    try:
        from safetensors import safe_open
        from vllm.model_executor.model_loader import default_loader as _dl
        from vllm.model_executor.model_loader import weight_utils as _wu
    except Exception:
        return
    if getattr(_wu, "_chronos_glm53_st_hooked", False):
        return
    prev = _wu.safetensors_weights_iterator

    def _iter(hf_weights_files: list[str], *args: Any, **kwargs: Any):
        if not any(is_glm53_chronos_file(f) for f in hf_weights_files):
            yield from prev(hf_weights_files, *args, **kwargs)
            return
        for st_file in sorted(hf_weights_files):
            if is_hot_sidecar(st_file) or "chronos-glm53-tiers-" in os.path.basename(st_file):
                continue
            try:
                with safe_open(st_file, framework="pt", device="cpu") as f:
                    for name in f.keys():
                        if _SKIP_MLP.search(name):
                            continue
                        yield name, f.get_tensor(name)
            finally:
                _evict(st_file)

    _wu.safetensors_weights_iterator = _iter
    if hasattr(_dl, "safetensors_weights_iterator"):
        _dl.safetensors_weights_iterator = _iter
    for alt in ("multi_thread_safetensors_weights_iterator",):
        prev_alt = getattr(_wu, alt, None)
        if prev_alt is None:
            continue

        def _alt(hf_weights_files: list[str], *args: Any, _p: Any = prev_alt, **kwargs: Any):
            if any(is_glm53_chronos_file(f) for f in hf_weights_files):
                yield from _iter(hf_weights_files)
            else:
                yield from _p(hf_weights_files, *args, **kwargs)

        setattr(_wu, alt, _alt)
        if hasattr(_dl, alt):
            setattr(_dl, alt, _alt)
    _wu._chronos_glm53_st_hooked = True


def install_chronos_moe_into_glm53(model: torch.nn.Module, ckpt_dir: str | Path, device: torch.device | None = None) -> list[int]:
    """Post-construction swap for models built without the hook (e.g. a meta-device transformers model)."""
    from chronos_1t.models.glm53_chronos_moe import ChronosGlm53SparseMoeBlock

    ckpt = Path(ckpt_dir)
    arch = Glm53Arch.from_json_file(ckpt / "config.json")
    inner = getattr(model, "model", model)
    installed = []
    for l_idx, layer in enumerate(getattr(inner, "layers", [])):
        shard = ckpt / layer_shard_name(l_idx, arch)
        if l_idx in arch.moe_layer_ids and shard.is_file():
            layer.mlp = ChronosGlm53SparseMoeBlock(l_idx, shard, arch, device)
            installed.append(l_idx)
    return installed
