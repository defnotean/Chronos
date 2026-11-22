#!/usr/bin/env python3
"""
Chronos-1T v8 Calibrated 3-Tier Streaming Loader (`DeepSeek-V4.1-Flash-1017`).

Streams unmodified official stock weights (`deepseek-ai/DeepSeek-V4.1-Flash-1017`, ZERO abliteration)
using the out-of-sample routing rankings from `chronos_dsv41_v8_routing_calibration.pt`:
  - Tier 1 (GPU VRAM, default 96/layer + 1 shared): unmodified stock 4-bit MXFP4
  - Tier 2 (CPU cudaHostRegister slab, default 128/layer): unmodified stock 4-bit MXFP4 (`uint8` E8M0)
  - Tier 3 (CPU cudaHostRegister slab, default 160/layer): direct 2-bit Lloyd-Max g64 (zero healers)
"""
from __future__ import annotations

import builtins
from concurrent.futures import Future, ThreadPoolExecutor
import gc
import importlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer


STOCK_REPO_DEFAULT = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
STREAM_DIR_DEFAULT = "/tmp/dsv41_v8_stream"


def _mem_telemetry() -> dict[str, float]:
    info: dict[str, float] = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for ln in fh:
                for key in ("MemTotal", "MemAvailable", "Shmem"):
                    if ln.startswith(f"{key}:"):
                        info[key] = int(ln.split()[1]) / (1024.0 * 1024.0)
        with open("/proc/self/status", "r", encoding="utf-8") as fh:
            for ln in fh:
                for key in ("VmRSS", "RssAnon", "RssShmem"):
                    if ln.startswith(f"{key}:"):
                        info[key] = int(ln.split()[1]) / (1024.0 * 1024.0)
    except Exception:
        pass
    return info


def _drop_page_cache(path: Path) -> None:
    if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED") and path.exists():
        try:
            fd = os.open(str(path), os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except OSError:
            pass


def _download_stock_shard(repo_id: str, filename: str, stream_dir: Path) -> Path:
    stream_dir.mkdir(parents=True, exist_ok=True)
    dest = stream_dir / filename
    if dest.exists() and dest.stat().st_size > 1024 * 1024:
        return dest
    tmp_part = stream_dir / f"{filename}.part"
    if tmp_part.exists():
        tmp_part.unlink(missing_ok=True)
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    cmd = [
        "curl",
        "-fSL",
        "--retry",
        "6",
        "--retry-delay",
        "3",
        "--connect-timeout",
        "20",
        "-o",
        str(tmp_part),
        url,
    ]
    res = subprocess.run(cmd, check=False)
    if res.returncode == 0 and tmp_part.exists() and tmp_part.stat().st_size > 1024 * 1024:
        tmp_part.replace(dest)
        return dest
    tmp_part.unlink(missing_ok=True)
    p = Path(hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(stream_dir)))
    return p


def _free_existing_resident_model() -> None:
    old = getattr(builtins, "GLOBAL_DSV41_MODEL", None)
    if old is not None:
        for lyr in getattr(old, "layers", []):
            ffn = getattr(lyr, "ffn", None)
            if ffn is not None and hasattr(ffn, "close"):
                try:
                    ffn.close()
                except Exception:
                    pass
    for attr in ("GLOBAL_DSV41_MODEL", "GLOBAL_DSV41_TOKENIZER", "GLOBAL_DSV41_GENERATE", "GLOBAL_DSV41_VERSION"):
        if hasattr(builtins, attr):
            delattr(builtins, attr)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> int:
    t0_all = time.perf_counter()
    repo_id = os.environ.get("CHRONOS_V8_STOCK_REPO", STOCK_REPO_DEFAULT).strip()
    lower_repo = repo_id.lower()
    if any(bad in lower_repo for bad in ("obliterat", "dealign", "uncensored")):
        raise ValueError(f"Refusing non-stock source repo '{repo_id}'. Must use unmodified official stock repo.")

    n_t1 = int(os.environ.get("CHRONOS_V8_T1", "96"))
    n_t2_fp4 = int(os.environ.get("CHRONOS_V8_T2_FP4", "128"))
    n_t3_2bit = int(os.environ.get("CHRONOS_V8_T3_2BIT", "160"))
    if n_t1 + n_t2_fp4 + n_t3_2bit != 384:
        raise ValueError(f"T1+T2+T3 must equal 384, got {n_t1}+{n_t2_fp4}+{n_t3_2bit}")

    max_seq_len = int(os.environ.get("CHRONOS_MAX_SEQ_LEN", "16384"))
    os.environ["CHRONOS_MAX_SEQ_LEN"] = str(max_seq_len)
    rank_key = os.environ.get("CHRONOS_V8_RANK_KEY", "train_ranked_ids").strip()
    stream_dir = Path(os.environ.get("CHRONOS_V8_STREAM_DIR", STREAM_DIR_DEFAULT))
    stream_dir.mkdir(parents=True, exist_ok=True)

    _repo_root = str(Path(__file__).resolve().parent.parent) if "__file__" in globals() else "/content"
    for p in (
        "/content",
        "/content/scripts",
        "/content/chronos-1t",
        _repo_root,
        "/content/dsv41_standalone/inference",
        "/content/dsv41_standalone/encoding",
    ):
        if p and os.path.exists(p) and p not in sys.path:
            sys.path.insert(0, p)

    existing = getattr(builtins, "GLOBAL_DSV41_MODEL", None)
    if existing is not None:
        if getattr(builtins, "GLOBAL_DSV41_VERSION", "") == "v8_calibrated_3tier" and os.environ.get("CHRONOS_V8_RELOAD", "0") != "1":
            print("[Init] Reusing already-loaded v8 GLOBAL_DSV41_MODEL.", flush=True)
            return 0
        if os.environ.get("CHRONOS_V8_REPLACE_RESIDENT", "0") == "1":
            print("[Init] Freeing previous resident GLOBAL_DSV41_MODEL...", flush=True)
            _free_existing_resident_model()
        else:
            raise RuntimeError(
                "A non-v8 GLOBAL_DSV41_MODEL is currently resident; set CHRONOS_V8_REPLACE_RESIDENT=1 or restart kernel."
            )

    for _k in list(sys.modules.keys()):
        if _k == "chronos_1t" or _k.startswith("chronos_1t."):
            sys.modules.pop(_k, None)
    importlib.invalidate_caches()

    import scripts.run_40layer_chimera_dsv41_a100 as runner_mod
    from chronos_1t.models.dsv41_chronos_moe import ChronosDsv41SparseMoeBlock

    calib_cands = [
        os.environ.get("CHRONOS_V8_CALIB_PATH", "").strip(),
        "/content/chronos_dsv41_v8_routing_calibration.pt",
        os.path.join(_repo_root, "results", "chronos_dsv41_v8_routing_calibration.pt"),
    ]
    calib_path = next((c for c in calib_cands if c and os.path.isfile(c)), "")
    if not calib_path:
        raise FileNotFoundError(f"Calibration file not found in candidates: {calib_cands}")
    calib = torch.load(calib_path, map_location="cpu", weights_only=True)
    if rank_key not in calib:
        raise KeyError(f"Rank key '{rank_key}' not found in {calib_path}. Keys={list(calib.keys())}")
    ranked_ids_40 = calib[rank_key]
    if tuple(ranked_ids_40.shape) != (40, 384):
        raise ValueError(f"Expected ranked_ids shape (40, 384), got {tuple(ranked_ids_40.shape)}")

    print("=" * 92, flush=True)
    print(f"CHRONOS v8 CALIBRATED 3-TIER LOADER | Source={repo_id} (UNMODIFIED STOCK, r_basis=None)", flush=True)
    print(
        f"Tier Split: T1={n_t1} (GPU MXFP4) / T2={n_t2_fp4} (CPU Pinned MXFP4) / T3={n_t3_2bit} (CPU Pinned 2-bit) | "
        f"max_seq_len={max_seq_len} | rank_key={rank_key}",
        flush=True,
    )
    print(f"Calibration={calib_path} | GPU={torch.cuda.get_device_name(0)}", flush=True)
    print("=" * 92, flush=True)

    tok_dir = Path("/content/dsv41_v8_tok")
    tok_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("tokenizer.json", "tokenizer_config.json", "config.json"):
        dst = tok_dir / fname
        if not dst.is_file() or dst.stat().st_size == 0:
            hf_hub_download(repo_id=repo_id, filename=fname, local_dir=str(tok_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(tok_dir), trust_remote_code=True)
    print(f"[Init] Tokenizer loaded from {tok_dir} (vocab={len(tokenizer)})", flush=True)

    args = runner_mod.build_model_args()
    args.max_seq_len = max_seq_len
    dsv41_mod = runner_mod.dsv41_model_mod
    orig_block_init = dsv41_mod.Block.__init__

    def _lightweight_block_init(
        self_b: Any,
        layer_id: int,
        args_b: Any,
        engram_layout: Any = None,
    ) -> None:
        nn.Module.__init__(self_b)
        self_b.layer_id = layer_id
        self_b.norm_eps = args_b.norm_eps
        self_b.attn = self_b.attention_cls(layer_id, args_b)
        self_b.ffn = nn.Identity()
        self_b.engram = None
        self_b.attn_norm = dsv41_mod.RMSNorm(args_b.dim, self_b.norm_eps)
        self_b.ffn_norm = dsv41_mod.RMSNorm(args_b.dim, self_b.norm_eps)
        self_b.hc_mult = hc_mult = args_b.hc_mult
        self_b.hc_sinkhorn_iters = args_b.hc_sinkhorn_iters
        self_b.hc_eps = args_b.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * args_b.dim
        with dsv41_mod.set_dtype(torch.float32):
            self_b.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
            self_b.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
            self_b.hc_attn_base = nn.Parameter(torch.empty(mix_hc))
            self_b.hc_ffn_base = nn.Parameter(torch.empty(mix_hc))
            self_b.hc_attn_scale = nn.Parameter(torch.empty(3))
            self_b.hc_ffn_scale = nn.Parameter(torch.empty(3))

    dsv41_mod.Block.__init__ = _lightweight_block_init
    try:
        torch.set_default_dtype(torch.bfloat16)
        dsv41_mod.default_dtype = torch.bfloat16
        with torch.device("cuda"):
            model = dsv41_mod.Transformer(args)
        model.eval()
    finally:
        dsv41_mod.Block.__init__ = orig_block_init

    mt = _mem_telemetry()
    print(
        f"[Init] 40-layer Transformer shell allocated | VRAM={torch.cuda.memory_allocated() / 1024**3:.2f} GiB | "
        f"MemAvail={mt.get('MemAvailable', 0):.2f} GiB",
        flush=True,
    )

    # Load embed.weight (shard 00002) and norm/head (shard 00043)
    for sh_name, p_list in [
        ("model-00002-of-00048.safetensors", [("embed.weight", model.embed.weight)]),
        ("model-00043-of-00048.safetensors", [("norm.weight", model.norm.weight), ("head.weight", model.head.weight)]),
    ]:
        sh_path = _download_stock_shard(repo_id, sh_name, stream_dir)
        with safe_open(str(sh_path), framework="pt", device="cpu") as sf_base:
            keys_b = set(sf_base.keys())
            for p_name, param in p_list:
                t = runner_mod.load_param_tensor(sf_base, keys_b, p_name, param.dtype, device="cuda")
                assert t.shape == param.shape, f"Shape mismatch {p_name}: {t.shape} vs {param.shape}"
                param.data.copy_(t)
                del t
        _drop_page_cache(sh_path)
        sh_path.unlink(missing_ok=True)
    gc.collect()
    torch.cuda.empty_cache()
    print("[Init] Loaded embed.weight, norm.weight, head.weight from official stock shards.", flush=True)

    # Stream layers 0..39 with 3-worker parallel curl prefetch
    n_layers = int(args.n_layers)
    prefetch_workers = int(os.environ.get("CHRONOS_V8_PREFETCH", "3"))
    prefetch_pool = ThreadPoolExecutor(max_workers=prefetch_workers)
    futures: dict[int, Future[Path]] = {}

    def _fetch_layer(idx: int) -> Path:
        return _download_stock_shard(repo_id, f"model-{idx + 3:05d}-of-00048.safetensors", stream_dir)

    for pre_i in range(min(prefetch_workers, n_layers)):
        futures[pre_i] = prefetch_pool.submit(_fetch_layer, pre_i)

    try:
        for l_idx in range(n_layers):
            t_l0 = time.perf_counter()
            for la in range(1, prefetch_workers + 1):
                nxt = l_idx + la
                if nxt < n_layers and nxt not in futures:
                    futures[nxt] = prefetch_pool.submit(_fetch_layer, nxt)

            l_path = futures.pop(l_idx).result()
            layer = model.layers[l_idx]
            with safe_open(str(l_path), framework="pt", device="cpu") as sf_layer:
                l_keys = set(sf_layer.keys())
                for p_name, param in layer.named_parameters():
                    full_k = f"layers.{l_idx}.{p_name}"
                    t = runner_mod.load_param_tensor(sf_layer, l_keys, full_k, param.dtype, device="cuda")
                    assert t.shape == param.shape, f"L{l_idx} {p_name} mismatch: {t.shape} vs {param.shape}"
                    param.data.copy_(t)
                    del t

                moe = ChronosDsv41SparseMoeBlock.from_v8_stock_shard(
                    layer_idx=l_idx,
                    stock_shard_path=l_path,
                    ranked_expert_ids=ranked_ids_40[l_idx],
                    n_t1=n_t1,
                    n_t2_fp4=n_t2_fp4,
                    n_t3_2bit=n_t3_2bit,
                    device="cuda",
                    sf_handle=sf_layer,
                )
                layer.ffn = moe

            _drop_page_cache(l_path)
            l_path.unlink(missing_ok=True)
            gc.collect()
            if l_idx % 4 == 3 or l_idx == n_layers - 1:
                torch.cuda.empty_cache()

            mt = _mem_telemetry()
            free_b, total_b = torch.cuda.mem_get_info()
            disk_free = shutil.disk_usage(str(stream_dir)).free / (1024.0**3)
            slab_pin = bool(getattr(moe._host_slab, "is_pinned", False))
            print(
                f"[L{l_idx:02d}/39 {time.perf_counter() - t_l0:4.1f}s] pin={slab_pin} | "
                f"VRAM_free={free_b / 1024**3:5.2f}/{total_b / 1024**3:5.2f} GiB | "
                f"MemAvail={mt.get('MemAvailable', 0):6.2f} GiB | "
                f"RSS={mt.get('VmRSS', 0):6.2f} (Shmem={mt.get('RssShmem', 0):5.2f}) | "
                f"DiskFree={disk_free:5.1f} GiB",
                flush=True,
            )
            if l_idx < n_layers - 1 and mt.get("MemAvailable", 999.0) < 8.0:
                raise RuntimeError(
                    f"Low host RAM safety stop at L{l_idx}: MemAvailable={mt.get('MemAvailable', 0):.2f} GiB < 8.0 GiB"
                )
    except Exception:
        for lyr in getattr(model, "layers", []):
            ffn = getattr(lyr, "ffn", None)
            if ffn is not None and hasattr(ffn, "close"):
                try:
                    ffn.close()
                except Exception:
                    pass
        try:
            model.layers.clear()
        except Exception:
            pass
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise
    finally:
        prefetch_pool.shutdown(wait=False)

    builtins.GLOBAL_DSV41_MODEL = model
    builtins.GLOBAL_DSV41_TOKENIZER = tokenizer
    builtins.GLOBAL_DSV41_GENERATE = runner_mod.generate_chat_greedy
    builtins.GLOBAL_DSV41_VERSION = "v8_calibrated_3tier"

    print("[Warmup] Capturing 40-layer CUDAGraphs for Chronos v8...", flush=True)
    cg_captured = runner_mod.warmup_40layer_cudagraphs(model)
    mt = _mem_telemetry()
    free_b, total_b = torch.cuda.mem_get_info()
    print(
        f"[Ready in {time.perf_counter() - t0_all:.1f}s] cg={cg_captured}/40 | "
        f"VRAM_free={free_b / 1024**3:.2f}/{total_b / 1024**3:.2f} GiB | "
        f"MemAvail={mt.get('MemAvailable', 0):.2f} GiB | RSS={mt.get('VmRSS', 0):.2f} GiB",
        flush=True,
    )

    if os.environ.get("CHRONOS_V8_SMOKE", "1") == "1":
        os.environ["CHRONOS_ABLATE_GUARDS"] = "1"
        r1 = runner_mod.generate_chat_greedy(
            model,
            tokenizer,
            [{"role": "user", "content": "What is 247 * 389? Reply with the exact integer."}],
            max_new_tokens=256,
            thinking_mode="thinking",
        )
        ok1 = ("96083" in r1["answer_text"].replace(",", "")) and r1["logits_finite"]
        print(
            f"[Smoke 1 Math] ok={ok1} finite={r1['logits_finite']} tok/s={r1['decode_tok_s']} "
            f"prefill={r1['prefill_s']}s ans={r1['answer_text']!r}",
            flush=True,
        )

        r2 = runner_mod.generate_chat_greedy(
            model,
            tokenizer,
            [
                {
                    "role": "user",
                    "content": (
                        "Write a concise Python function `merge_intervals(intervals)` that sorts `intervals` "
                        "by start time and merges overlapping intervals `[start, end]`. Return only the code."
                    ),
                }
            ],
            max_new_tokens=512,
            thinking_mode="thinking",
        )
        ok2, det2 = runner_mod.verify_python_merge_intervals(r2["answer_text"] or r2["text"])
        ok2 = ok2 and r2["logits_finite"]
        print(
            f"[Smoke 2 Code] ok={ok2} ({det2}) finite={r2['logits_finite']} tok/s={r2['decode_tok_s']} "
            f"prefill={r2['prefill_s']}s",
            flush=True,
        )
        runner_mod.reset_model_caches(model)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
