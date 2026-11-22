#!/usr/bin/env python3
"""
Chronos-1T v8 Calibrated 3-Tier Offline Checkpoint Exporter (`DeepSeek-V4.1-Flash-1017`).

Streams unmodified official stock shards (`deepseek-ai/DeepSeek-V4.1-Flash-1017`, zero abliteration)
and saves compact per-layer Chronos v8 safetensors into `--save-layer-dir` using the
out-of-sample routing rankings from `--calib-path`.
"""
from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file

_here = Path(__file__).resolve()
_repo_root = str(_here.parents[2]) if len(_here.parents) > 2 else "/content"
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from chronos_1t.models.dsv41_chronos_moe import (
    _triton_quantize_2bit_g64_kernel,
    dequantize_mxfp4_32,
)
from scripts.load_dsv41_v8_calibrated_3tier import _download_stock_shard, _drop_page_cache


def export_layer_shard(
    layer_idx: int,
    raw_shard_path: Path,
    out_shard: Path,
    ranked_eids: list[int],
    n_t1: int,
    n_t2_fp4: int,
    n_t3_2bit: int,
    device: torch.device,
) -> float:
    """Convert one stock layer shard into a compact Chronos v8 3-tier safetensors file."""
    t0 = time.perf_counter()
    tp = f"layers.{layer_idx}.ffn"
    tier_fp4_eids = ranked_eids[: n_t1 + n_t2_fp4]
    tier3_eids = ranked_eids[n_t1 + n_t2_fp4 :]

    out_tensors: dict[str, torch.Tensor] = {
        f"{tp}.v8.ranked_expert_ids": torch.tensor(ranked_eids, dtype=torch.int16),
        f"{tp}.v8.tier_counts": torch.tensor([n_t1, n_t2_fp4, n_t3_2bit], dtype=torch.int32),
    }
    with safe_open(str(raw_shard_path), framework="pt", device="cpu") as sf:
        for k in sf.keys():
            if not k.startswith(f"{tp}.experts."):
                out_tensors[k] = sf.get_tensor(k).contiguous()

        n_fp4 = len(tier_fp4_eids)
        fp4_w_gu = torch.empty((n_fp4, 4608, 2560), dtype=torch.uint8)
        fp4_s_gu = torch.empty((n_fp4, 4608, 160), dtype=torch.uint8)
        fp4_w_d = torch.empty((n_fp4, 5120, 1152), dtype=torch.uint8)
        fp4_s_d = torch.empty((n_fp4, 5120, 72), dtype=torch.uint8)
        for slot_i, eid in enumerate(tier_fp4_eids):
            ep = f"{tp}.experts.{eid}"
            fp4_w_gu[slot_i, :2304].copy_(sf.get_tensor(f"{ep}.w1.weight").view(torch.uint8))
            fp4_w_gu[slot_i, 2304:].copy_(sf.get_tensor(f"{ep}.w3.weight").view(torch.uint8))
            fp4_w_d[slot_i].copy_(sf.get_tensor(f"{ep}.w2.weight").view(torch.uint8))
            fp4_s_gu[slot_i, :2304].copy_(sf.get_tensor(f"{ep}.w1.scale").view(torch.uint8))
            fp4_s_gu[slot_i, 2304:].copy_(sf.get_tensor(f"{ep}.w3.scale").view(torch.uint8))
            fp4_s_d[slot_i].copy_(sf.get_tensor(f"{ep}.w2.scale").view(torch.uint8))
        out_tensors[f"{tp}.v8_fp4.w_gu"] = fp4_w_gu
        out_tensors[f"{tp}.v8_fp4.s_gu"] = fp4_s_gu
        out_tensors[f"{tp}.v8_fp4.w_d"] = fp4_w_d
        out_tensors[f"{tp}.v8_fp4.s_d"] = fp4_s_d

        if tier3_eids:
            n_t3 = len(tier3_eids)
            t3_w_gu = torch.empty((n_t3, 4608, 1280), dtype=torch.uint8)
            t3_s_gu = torch.empty((n_t3, 4608, 80), dtype=torch.uint8)
            t3_w_d = torch.empty((n_t3, 5120, 576), dtype=torch.uint8)
            t3_s_d = torch.empty((n_t3, 5120, 36), dtype=torch.uint8)
            scr_gu = torch.empty((4608, 5120), dtype=torch.bfloat16, device=device)
            scr_d = torch.empty((5120, 2304), dtype=torch.bfloat16, device=device)
            g_w2_gu = torch.empty((4608, 1280), dtype=torch.uint8, device=device)
            g_s2_gu = torch.empty((4608, 80), dtype=torch.uint8, device=device)
            g_w2_d = torch.empty((5120, 576), dtype=torch.uint8, device=device)
            g_s2_d = torch.empty((5120, 36), dtype=torch.uint8, device=device)
            for s2, eid in enumerate(tier3_eids):
                ep = f"{tp}.experts.{eid}"
                w1_w = sf.get_tensor(f"{ep}.w1.weight").view(torch.uint8).to(device)
                w1_s = sf.get_tensor(f"{ep}.w1.scale").view(torch.uint8).to(device)
                w3_w = sf.get_tensor(f"{ep}.w3.weight").view(torch.uint8).to(device)
                w3_s = sf.get_tensor(f"{ep}.w3.scale").view(torch.uint8).to(device)
                w2_w = sf.get_tensor(f"{ep}.w2.weight").view(torch.uint8).to(device)
                w2_s = sf.get_tensor(f"{ep}.w2.scale").view(torch.uint8).to(device)
                scr_gu[:2304].copy_(dequantize_mxfp4_32(w1_w, w1_s, device, dtype=torch.bfloat16))
                scr_gu[2304:].copy_(dequantize_mxfp4_32(w3_w, w3_s, device, dtype=torch.bfloat16))
                scr_d.copy_(dequantize_mxfp4_32(w2_w, w2_s, device, dtype=torch.bfloat16))
                _triton_quantize_2bit_g64_kernel[(4608, 80)](scr_gu, g_w2_gu, g_s2_gu, N_GROUPS_C=80, num_warps=1)
                _triton_quantize_2bit_g64_kernel[(5120, 36)](scr_d, g_w2_d, g_s2_d, N_GROUPS_C=36, num_warps=1)
                t3_w_gu[s2].copy_(g_w2_gu, non_blocking=False)
                t3_s_gu[s2].copy_(g_s2_gu, non_blocking=False)
                t3_w_d[s2].copy_(g_w2_d, non_blocking=False)
                t3_s_d[s2].copy_(g_s2_d, non_blocking=False)
            out_tensors[f"{tp}.v8_2bit.w_gu"] = t3_w_gu
            out_tensors[f"{tp}.v8_2bit.s_gu"] = t3_s_gu
            out_tensors[f"{tp}.v8_2bit.w_d"] = t3_w_d
            out_tensors[f"{tp}.v8_2bit.s_d"] = t3_s_d
            del scr_gu, scr_d, g_w2_gu, g_s2_gu, g_w2_d, g_s2_d

    save_file(out_tensors, str(out_shard))
    _drop_page_cache(out_shard)
    del out_tensors
    gc.collect()
    torch.cuda.empty_cache()
    return time.perf_counter() - t0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="deepseek-ai/DeepSeek-V4.1-Flash-1017")
    parser.add_argument("--calib-path", default="results/chronos_dsv41_v8_routing_calibration.pt")
    parser.add_argument("--rank-key", default="train_ranked_ids")
    parser.add_argument("--save-layer-dir", required=True, help="Output directory for compact v8 shards")
    parser.add_argument("--stream-dir", default="/tmp/dsv41_v8_stream")
    parser.add_argument("--t1", type=int, default=96)
    parser.add_argument("--t2-fp4", type=int, default=128)
    parser.add_argument("--t3-2bit", type=int, default=160)
    parser.add_argument("--layers", default="all")
    args = parser.parse_args()

    if any(b in args.repo_id.lower() for b in ("obliterat", "dealign", "uncensored")):
        raise ValueError(f"Refusing non-stock source repo '{args.repo_id}'.")
    if args.t1 + args.t2_fp4 + args.t3_2bit != 384:
        raise ValueError("t1 + t2-fp4 + t3-2bit must equal 384.")

    out_dir = Path(args.save_layer_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stream_dir = Path(args.stream_dir)
    stream_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    calib = torch.load(args.calib_path, map_location="cpu", weights_only=True)
    ranked_40 = calib[args.rank_key]

    for fname in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        hf_hub_download(repo_id=args.repo_id, filename=fname, local_dir=str(out_dir))
    cfg = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    cfg["chronos_v8_calibrated_moe"] = {
        "version": "8.0",
        "source_repo": args.repo_id,
        "abliterated": False,
        "tier1_gpu_mxfp4": args.t1,
        "tier2_cpu_mxfp4": args.t2_fp4,
        "tier3_cpu_2bit_g64": args.t3_2bit,
        "rank_key": args.rank_key,
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    target_layers = list(range(40)) if args.layers == "all" else [int(x) for x in args.layers.split(",") if x.strip()]
    pool = ThreadPoolExecutor(max_workers=2)
    futs: dict[int, Future[Path]] = {}
    if target_layers:
        futs[target_layers[0]] = pool.submit(
            _download_stock_shard, args.repo_id, f"model-{target_layers[0] + 3:05d}-of-00048.safetensors", stream_dir
        )
    try:
        for i, l_idx in enumerate(target_layers):
            if i + 1 < len(target_layers):
                nxt = target_layers[i + 1]
                futs[nxt] = pool.submit(
                    _download_stock_shard, args.repo_id, f"model-{nxt + 3:05d}-of-00048.safetensors", stream_dir
                )
            out_s = out_dir / f"chronos-dsv41-v8-layer-{l_idx:05d}-of-00040.safetensors"
            raw_p = futs.pop(l_idx).result()
            if not out_s.exists():
                dt = export_layer_shard(
                    l_idx, raw_p, out_s, ranked_40[l_idx].tolist(), args.t1, args.t2_fp4, args.t3_2bit, device
                )
                print(
                    f"[v8 Export L{l_idx:02d}] {dt:.1f}s -> {out_s.name} ({out_s.stat().st_size / 1e9:.2f} GB) | "
                    f"free={shutil.disk_usage(out_dir).free / 1e9:.1f} GB",
                    flush=True,
                )
            _drop_page_cache(raw_p)
            raw_p.unlink(missing_ok=True)
    finally:
        pool.shutdown(wait=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
