"""
Chronos v7.5 vLLM Runtime Plugin & Option-A BF16 Shared-Expert Healer Hook for MiniMax-M3.1-Flash.

Registers PB27MiniMaxM3SparseForConditionalGeneration and PB27MiniMaxM3SparseForCausalLM
in vLLM's ModelRegistry and attaches the optional runtime BF16 Option-A Rank-64
residual healer on MiniMaxM3MoE.shared_experts.
"""

from __future__ import annotations

import logging
import os
from typing import Dict

import torch

logger = logging.getLogger(__name__)


def enable_chronos_minimax_m31_vllm_hook(
    default_ckpt_dir: str = "/home/defnotean/pb27/hf/minimax_m31_chronos_v75_abliterated",
    runtime_healer_scale: float = 0.0,
) -> None:
    """Register PB27MiniMaxM3Sparse* architectures and install optional BF16 healer hook."""
    try:
        from vllm import ModelRegistry
        from vllm.model_executor.models.config import (
            MODELS_CONFIG_MAP,
            Qwen3_5ForConditionalGenerationConfig,
        )

        supported = set(ModelRegistry.get_supported_archs())
        for arch, target in (
            (
                "PB27MiniMaxM3SparseForConditionalGeneration",
                "vllm.models.minimax_m3:MiniMaxM3SparseForConditionalGeneration",
            ),
            (
                "PB27MiniMaxM3SparseForCausalLM",
                "vllm.models.minimax_m3:MiniMaxM3SparseForCausalLM",
            ),
        ):
            if arch not in supported:
                ModelRegistry.register_model(arch, target)
            MODELS_CONFIG_MAP.setdefault(arch, Qwen3_5ForConditionalGenerationConfig)
    except Exception as exc:
        logger.debug("PB27MiniMaxM3 registration skipped: %s", exc)

    if runtime_healer_scale <= 0.0:
        # By default, the Chronos v7.5 Option-A Rank-64 healer is already folded
        # directly into the MXFP8 shared_experts.down_proj weights on disk for
        # zero-overhead execution in FlashInfer / CUTLASS. Set
        # CHRONOS_MINIMAX_RUNTIME_HEALER_SCALE > 0 to add an additional live BF16 delta.
        env_scale = float(os.environ.get("CHRONOS_MINIMAX_RUNTIME_HEALER_SCALE", "0.0") or 0.0)
        if env_scale <= 0.0:
            return
        runtime_healer_scale = env_scale

    try:
        from vllm.models.minimax_m3.nvidia.model import MiniMaxM3MLP

        if getattr(MiniMaxM3MLP, "_chronos_m31_hooked", False):
            return

        orig_forward = MiniMaxM3MLP.forward
        healer_cache: Dict[int, Dict[str, torch.Tensor]] = {}
        healer_path = os.environ.get(
            "CHRONOS_MINIMAX_HEALER_PT",
            os.path.join(default_ckpt_dir, "minimax_m31_shared_experts_option_a_r64.pt"),
        )
        if os.path.exists(healer_path):
            healer_cache = torch.load(healer_path, map_location="cpu", weights_only=True)

        def _chronos_mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
            gate_up, _ = self.gate_up_proj(x)
            act = self.act_fn(gate_up)
            out, _ = self.down_proj(act)
            vt = getattr(self, "_chronos_opt_a_VT", None)
            if vt is None and hasattr(self, "down_proj"):
                pref = getattr(self.down_proj, "prefix", "") or ""
                if ".shared_experts.down_proj" in pref and "layers." in pref:
                    try:
                        l_idx = int(pref.split("layers.")[1].split(".")[0])
                        if l_idx in healer_cache:
                            u = healer_cache[l_idx]["U"].to(device=x.device, dtype=torch.bfloat16)
                            v = healer_cache[l_idx]["V"].to(device=x.device, dtype=torch.bfloat16)
                            self._chronos_opt_a_VT = v.contiguous()
                            self._chronos_opt_a_UT = (u.T.contiguous() * runtime_healer_scale).to(torch.bfloat16)
                            vt = self._chronos_opt_a_VT
                        else:
                            self._chronos_opt_a_VT = False
                    except Exception:
                        self._chronos_opt_a_VT = False
                else:
                    self._chronos_opt_a_VT = False
            if isinstance(vt, torch.Tensor):
                out = out + (act.to(vt.dtype) @ vt) @ self._chronos_opt_a_UT
            return out

        MiniMaxM3MLP.forward = _chronos_mlp_forward
        MiniMaxM3MLP._chronos_m31_hooked = True
    except Exception as exc:
        logger.debug("MiniMaxM3MLP runtime healer hook skipped: %s", exc)
