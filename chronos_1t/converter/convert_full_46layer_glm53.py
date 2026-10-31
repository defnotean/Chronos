#!/usr/bin/env python3
"""
Full 46-Layer (`0..2` Dense + `3..45` Sparse MoE + MLA Attention + Embeddings + `lm_head` + Tokenizer)
Streaming Converter for `zai-org/GLM-5.3-Flash` (`Chronos-1T v7.5`).

Produces a complete, self-contained 46-layer runnable checkpoint in `--out-dir`:
  - `config.json`, `tokenizer.json`, `tokenizer_config.json`
  - `chronos-glm5-base-dense-00000-00002.safetensors` (`embed_tokens`, dense layers `0..2`, `norm`, `lm_head`)
  - `chronos-glm5-layer-00003-of-00046.safetensors` .. `chronos-glm5-layer-00045-of-00046.safetensors`
    (each containing that layer's `input_layernorm`, `post_attention_layernorm`, `self_attn` MLA weights,
    AND the Chronos `v7.5` compressed MoE weights for all 288 routed experts + shared expert).
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import time
import urllib.request

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file

import sys
_here = Path(__file__).resolve()
sys.path.insert(0, str(_here.parent))
if len(_here.parents) > 2:
    sys.path.insert(0, str(_here.parents[2]))

try:
    from chronos_1t.converter.convert_glm53_flash_colab import (
        build_hadamard_128,
        convert_glm5_expert_group,
        dequantize_block_fp8_128,
    )
except ImportError:
    from convert_glm53_flash_colab import (
        build_hadamard_128,
        convert_glm5_expert_group,
        dequantize_block_fp8_128,
    )


def drop_page_cache(path: Path) -> None:
    if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED") and path.exists():
        try:
            fd = os.open(str(path), os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except OSError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="zai-org/GLM-5.3-Flash")
    parser.add_argument("--out-dir", default="/home/defnotean/pb27/hf/glm53_flash_chronos_v6")
    parser.add_argument("--cache-dir", default="/home/defnotean/pb27/hf_cache_glm53")
    parser.add_argument("--layers", default="all", help="'all' (3..45) or comma-separated MoE layer list")
    parser.add_argument("--gu-bits", type=int, default=1, help="1 (Compact-Disk 73GB) or 2")
    parser.add_argument("--down-bits", type=int, default=1, help="1 (Compact-Disk 73GB) or 2")
    parser.add_argument("--svd-rank", type=int, default=64)
    parser.add_argument("--skip-base", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 105, flush=True)
    print(f"CHRONOS-1T v7.5 FULL 46-LAYER STREAMING CONVERTER FOR {args.repo_id} ON {torch.cuda.get_device_name(0)}", flush=True)
    print("=" * 105, flush=True)

    # 1. Download tokenizer and config files
    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"]:
        try:
            p = hf_hub_download(repo_id=args.repo_id, filename=fname, local_dir=str(out_dir))
            print(f"[Tokenizer/Config] Saved {fname} -> {p}", flush=True)
        except Exception as e:
            print(f"[Tokenizer/Config] Optional {fname} skipped ({e})", flush=True)

    idx_p = hf_hub_download(repo_id=args.repo_id, filename="model.safetensors.index.json", local_dir=str(out_dir))
    idx_data = json.loads(Path(idx_p).read_text())
    wmap: dict[str, str] = idx_data["weight_map"]

    downloaded_shards: dict[str, Path] = {}

    def ensure_shards(shard_names: set[str]) -> dict[str, safe_open]:
        handles = {}
        for sf in sorted(shard_names):
            if sf not in downloaded_shards or not downloaded_shards[sf].exists():
                local_p = hf_hub_download(repo_id=args.repo_id, filename=sf, local_dir=str(cache_dir))
                downloaded_shards[sf] = Path(local_p)
            handles[sf] = safe_open(str(downloaded_shards[sf]), framework="pt", device="cpu")
        return handles

    # 2. Save Base & Dense Layers (0, 1, 2) + embed_tokens + norm + lm_head
    base_shard = out_dir / "chronos-glm5-base-dense-00000-00002.safetensors"
    if not args.skip_base and not base_shard.exists():
        t_b0 = time.perf_counter()
        base_keys = [
            k for k in wmap.keys()
            if k.startswith("model.language_model.embed_tokens.")
            or k.startswith("model.language_model.layers.0.")
            or k.startswith("model.language_model.layers.1.")
            or k.startswith("model.language_model.layers.2.")
            or k.startswith("model.language_model.norm.")
            or k.startswith("lm_head.")
        ]
        needed_base_shards = {wmap[k] for k in base_keys}
        print(f"\n[Base + Dense Layers 0..2] Downloading/opening {len(needed_base_shards)} shards: {sorted(needed_base_shards)}", flush=True)
        handles = ensure_shards(needed_base_shards)
        base_tensors = {k: handles[wmap[k]].get_tensor(k).contiguous() for k in base_keys}
        save_file(base_tensors, str(base_shard))
        drop_page_cache(base_shard)
        del handles, base_tensors
        gc.collect()
        print(f"[Base + Dense Layers 0..2 DONE in {time.perf_counter()-t_b0:.1f}s] Saved {base_shard.name} ({base_shard.stat().st_size/1e9:.2f} GB)", flush=True)

    # 3. Stream and convert all 43 MoE Layers (3..45) + their Attention/MLA/Norm weights
    if args.layers == "all":
        target_layers = list(range(3, 46))
    else:
        target_layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]

    h128 = build_hadamard_128(device)
    progress_path = out_dir / "full_46layer_progress.json"
    completed_layers = []

    for idx_pos, l_idx in enumerate(target_layers):
        t0 = time.perf_counter()
        layer_prefix = f"model.language_model.layers.{l_idx}."
        mlp_prefix = f"model.language_model.layers.{l_idx}.mlp"
        out_shard = out_dir / f"chronos-glm5-layer-{l_idx:05d}-of-00046.safetensors"

        layer_keys = [k for k in wmap.keys() if k.startswith(layer_prefix)]
        non_moe_keys = [k for k in layer_keys if not k.startswith(mlp_prefix + ".")]

        # Check if out_shard already exists and already has both self_attn and chronos_groups
        if out_shard.exists():
            with safe_open(str(out_shard), framework="pt", device="cpu") as f_chk:
                existing_keys = set(f_chk.keys())
            has_moe = any(".chronos_groups.35." in k for k in existing_keys)
            has_attn = all(k in existing_keys for k in non_moe_keys)
            if has_moe and has_attn:
                print(f"[Layer {l_idx:02d}/45] Already complete ({out_shard.stat().st_size/1e9:.2f} GB), skipping.", flush=True)
                completed_layers.append({"layer": l_idx, "status": "cached", "size_gb": round(out_shard.stat().st_size / 1e9, 3)})
                continue
            elif has_moe and not has_attn:
                print(f"[Layer {l_idx:02d}/45] Adding non-MoE (MLA attention + norms) to existing MoE shard...", flush=True)
                needed_attn_shards = {wmap[k] for k in non_moe_keys}
                handles = ensure_shards(needed_attn_shards)
                out_tensors = load_file(str(out_shard), device="cpu")
                for k in non_moe_keys:
                    out_tensors[k] = handles[wmap[k]].get_tensor(k).contiguous()
                save_file(out_tensors, str(out_shard))
                drop_page_cache(out_shard)
                del handles, out_tensors
                gc.collect()
                completed_layers.append({"layer": l_idx, "status": "upgraded_with_attn", "size_gb": round(out_shard.stat().st_size / 1e9, 3)})
                continue

        needed_shards = {wmap[k] for k in layer_keys}
        print(f"\n[Layer {l_idx:02d}/45] Streaming {len(needed_shards)} shards: {sorted(needed_shards)}", flush=True)
        handles = ensure_shards(needed_shards)

        def get_t(key: str) -> torch.Tensor:
            return handles[wmap[key]].get_tensor(key)

        out_tensors: dict[str, torch.Tensor] = {}
        # Copy all non-MoE tensors for this layer (input_layernorm, post_attention_layernorm, self_attn MLA)
        for k in non_moe_keys:
            out_tensors[k] = get_t(k).contiguous()

        router_w = get_t(f"{mlp_prefix}.gate.weight").to(device=device, dtype=torch.bfloat16)
        e_bias = get_t(f"{mlp_prefix}.gate.e_score_correction_bias").to(device=device, dtype=torch.float32)

        _, s_rt, vh_rt = torch.linalg.svd(router_w.float(), full_matrices=False)
        v_rt_sub = vh_rt[:256, :].T.contiguous()
        s_ref = s_rt[min(2, s_rt.shape[0] - 1)].clamp_min(1e-6)
        s_rt_sub = 1.0 + 4.5 * ((s_rt[:256] / s_ref).clamp(max=1.5) ** 0.30)

        out_tensors[f"{mlp_prefix}.gate.weight"] = router_w.cpu().contiguous()
        out_tensors[f"{mlp_prefix}.gate.e_score_correction_bias"] = e_bias.cpu().contiguous()
        for sh_k in [
            "shared_experts.gate_proj.weight",
            "shared_experts.gate_proj.weight_scale_inv",
            "shared_experts.up_proj.weight",
            "shared_experts.up_proj.weight_scale_inv",
            "shared_experts.down_proj.weight",
            "shared_experts.down_proj.weight_scale_inv",
        ]:
            full_k = f"{mlp_prefix}.{sh_k}"
            out_tensors[full_k] = get_t(full_k).contiguous()

        for g_idx in range(36):
            g_list, u_list, d_list = [], [], []
            for m in range(8):
                eid = g_idx * 8 + m
                ep = f"{mlp_prefix}.experts.{eid}"
                gw = dequantize_block_fp8_128(get_t(f"{ep}.gate_proj.weight"), get_t(f"{ep}.gate_proj.weight_scale_inv"), device)
                uw = dequantize_block_fp8_128(get_t(f"{ep}.up_proj.weight"), get_t(f"{ep}.up_proj.weight_scale_inv"), device)
                dw = dequantize_block_fp8_128(get_t(f"{ep}.down_proj.weight"), get_t(f"{ep}.down_proj.weight_scale_inv"), device)
                g_list.append(gw)
                u_list.append(uw)
                d_list.append(dw)

            g_dict = convert_glm5_expert_group(
                torch.stack(g_list, dim=0),
                torch.stack(u_list, dim=0),
                torch.stack(d_list, dim=0),
                router_w,
                v_rt_sub,
                s_rt_sub,
                g_idx,
                h128,
                svd_rank=args.svd_rank,
                gu_tail_bits=args.gu_bits,
                down_tail_bits=args.down_bits,
            )
            for tk, tv in g_dict.items():
                out_tensors[f"{mlp_prefix}.chronos_groups.{g_idx}.{tk}"] = tv.contiguous()
            del g_list, u_list, d_list, g_dict

        save_file(out_tensors, str(out_shard))
        drop_page_cache(out_shard)
        shard_gb = out_shard.stat().st_size / 1e9
        elapsed = time.perf_counter() - t0
        print(f"[Layer {l_idx:02d}/45 DONE in {elapsed:.1f}s] Saved {out_shard.name} ({shard_gb:.3f} GB)", flush=True)

        completed_layers.append({
            "layer": l_idx,
            "status": "converted",
            "elapsed_s": round(elapsed, 2),
            "size_gb": round(shard_gb, 3),
        })
        progress_path.write_text(json.dumps({
            "completed_count": len(completed_layers),
            "total_target_layers": len(target_layers),
            "layers": completed_layers,
        }, indent=2))

        del handles, out_tensors, router_w, e_bias, v_rt_sub, s_rt_sub
        # Evict any raw HF shards not needed by remaining layers
        remaining_shards = set()
        for rem_l in target_layers[idx_pos + 1 :]:
            rem_p = f"model.language_model.layers.{rem_l}."
            for k, sf in wmap.items():
                if k.startswith(rem_p):
                    remaining_shards.add(sf)
        for sf, sp in list(downloaded_shards.items()):
            if sf not in remaining_shards:
                if sp.exists():
                    drop_page_cache(sp)
                    sp.unlink()
                del downloaded_shards[sf]
        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + "=" * 105, flush=True)
    print(f"ALL {len(target_layers)} TARGET LAYERS OF {args.repo_id} CONVERTED SUCCESSFULLY TO {out_dir}", flush=True)
    print("=" * 105, flush=True)


if __name__ == "__main__":
    main()
