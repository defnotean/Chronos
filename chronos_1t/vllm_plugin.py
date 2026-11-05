#!/usr/bin/env python3
"""
Chronos-1T Zero-Touch vLLM & HuggingFace Transformers Plugin (`vllm.general_plugins` Entrypoint).

When `chronos-1t` is installed (`pip install -e .`), vLLM automatically invokes `register()` in every
process (including spawned multiprocess GPU workers) prior to model construction, enabling stock
`vllm serve <checkpoint_dir>` to transparently load `ChronosQwen4ExpSparseMoeBlock`,
`ChronosGlm5SparseMoeBlock`, `ChronosDsv41SparseMoeBlock`, and `ChronosGlm53SparseMoeBlock` (GLM-5.3 full,
`chronos-glm53-layer-*` shards) whenever `"chronos_holographic_moe"` is present in `config.json`.
"""
from __future__ import annotations


def register() -> None:
    """Automatic vLLM plugin entrypoint (`vllm.general_plugins`)."""
    try:
        from chronos_1t.models.patch_pb27_vllm import enable_chronos_vllm_hook
        enable_chronos_vllm_hook()
    except Exception:
        pass

    try:
        from chronos_1t.models.patch_glm5_vllm import enable_chronos_glm5_vllm_hook
        enable_chronos_glm5_vllm_hook()
    except Exception:
        pass

    try:
        from chronos_1t.models.patch_dsv41_vllm import enable_chronos_dsv41_vllm_hook
        enable_chronos_dsv41_vllm_hook()
    except Exception:
        pass

    try:
        from chronos_1t.models.patch_glm53_vllm import enable_chronos_glm53_vllm_hook
        enable_chronos_glm53_vllm_hook()
    except Exception:
        pass
