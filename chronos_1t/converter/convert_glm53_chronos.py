#!/usr/bin/env python3
"""
Chronos-1T v7.5 x GLM-5.3 (`zai-org/GLM-5.3`) — Full 79-Layer Streaming Converter (Colab A100 / H100 / DGX Spark).

Converts the 753B `GlmMoeDsaForCausalLM` block-FP8 checkpoint (`755.6 GB`, `141` shards) into the Chronos v7.5
holographic format used by the GLM-5.3-Flash and DeepSeek-V4.1 releases:

  chronos-glm53-base-embed-head-norm.safetensors        embed_tokens, lm_head, final norm (BF16, unchanged)
  chronos-glm53-layer-00000-of-00079.safetensors        dense layers 0..2: attention + MLP, stock FP8 (unchanged)
  chronos-glm53-layer-00003-of-00079.safetensors  ...   MoE layers 3..77 + MTP layer 78: attention, norms, indexer,
                                                        router, shared expert (stock FP8) + 32 `chronos_groups.*`
  model.safetensors.index.json / config.json            `chronos_holographic_moe` metadata (arch = glm53)

Disk / network behaviour (Colab friendly):
  * Source shards are downloaded on demand, prefetched one layer ahead, reference-counted across layers (shards
    straddle layer boundaries) and deleted as soon as no remaining layer needs them -> peak ~4 source shards on disk.
  * Output shards can be streamed to a Hugging Face repo in the background (`--upload-repo`) and deleted after a
    successful upload (`--delete-after-upload`) so a ~235 GB Colab disk can produce the ~300 GB standard checkpoint.
  * Fully resumable: finished layers are detected from the output dir (or the upload repo) and skipped.

Validation modes that do not need a GPU or a full download:
  * `--source http` streams individual tensors with HTTP Range requests (`--layers 3 --groups 0`), e.g. to measure
    reconstruction fidelity of one real GLM-5.3 group on a laptop CPU.
  * `--source local --source-dir DIR` converts a local checkpoint in GLM-5.3 layout (used by the synthetic tests).
"""
from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import gc
import json
import os
from pathlib import Path
import re
import shutil
import struct
import sys
import threading
import time
import urllib.request

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
if Path("/content/.hf_token").exists() and "HF_TOKEN" not in os.environ:
    os.environ["HF_TOKEN"] = Path("/content/.hf_token").read_text().strip()

_here = Path(__file__).resolve()
if len(_here.parents) > 2:
    sys.path.insert(0, str(_here.parents[2]))

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from chronos_1t.models.glm53_chronos_spec import (  # noqa: E402
    CHRONOS_VERSION,
    GLM53_CHRONOS_REPO_ID,
    GLM53_REPO_ID,
    GROUP_SIZE,
    Glm53Arch,
    base_shard_name,
    footprint_report,
    hot_shard_name,
    layer_shard_name,
    moe_prefix,
)
from chronos_1t.models.glm53_holo import (  # noqa: E402
    KEY_GENERATOR,
    convert_expert_group,
    make_router_probes,
    router_prior,
    router_subspace,
)
from chronos_1t.models.glm53_tiers import dequant_fp8_block, pack_t1_fp4  # noqa: E402

META_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
_DTYPES = {
    "F8_E4M3": torch.float8_e4m3fn, "BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
    "U8": torch.uint8, "I8": torch.int8, "I32": torch.int32, "I64": torch.int64,
}


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


def _auth_headers() -> dict[str, str]:
    h = {"User-Agent": "chronos-glm53-converter"}
    if os.environ.get("HF_TOKEN"):
        h["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    return h


def fetch_json(repo_id: str, filename: str) -> dict:
    req = urllib.request.Request(f"https://huggingface.co/{repo_id}/resolve/main/{filename}", headers=_auth_headers())
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


# =============================================================================================
# Tensor sources
# =============================================================================================
class _Source:
    wmap: dict[str, str]

    def layer_keys(self, layer_idx: int) -> list[str]:
        pref = f"model.layers.{layer_idx}."
        return sorted(k for k in self.wmap if k.startswith(pref))

    def get(self, key: str) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    def prefetch_layer(self, layer_idx: int) -> None:
        pass

    def release_layer(self, layer_idx: int) -> None:
        pass

    def close(self) -> None:
        pass


class LocalSource(_Source):
    def __init__(self, src_dir: Path) -> None:
        self.dir = Path(src_dir)
        idx = self.dir / "model.safetensors.index.json"
        if idx.exists():
            self.wmap = json.loads(idx.read_text())["weight_map"]
        else:
            self.wmap = {}
            for f in sorted(self.dir.glob("*.safetensors")):
                with safe_open(str(f), framework="pt", device="cpu") as fh:
                    for k in fh.keys():
                        self.wmap[k] = f.name
        self._handles: dict[str, object] = {}

    def get(self, key: str) -> torch.Tensor:
        sf = self.wmap[key]
        h = self._handles.get(sf)
        if h is None:
            h = safe_open(str(self.dir / sf), framework="pt", device="cpu")
            self._handles[sf] = h
        return h.get_tensor(key)

    def release_layer(self, layer_idx: int) -> None:
        self._handles.clear()


class HFStreamingSource(_Source):
    """Downloads whole source shards on demand with layer-aware reference counting and one-layer-ahead prefetch."""

    def __init__(self, repo_id: str, cache_dir: Path, wmap: dict[str, str], layer_order: list[int], keep: bool = False):
        self.repo_id = repo_id
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.wmap = wmap
        self.keep = keep
        self._lock = threading.Lock()
        self._futures: dict[str, Future] = {}
        self._handles: dict[str, object] = {}
        self._pool = ThreadPoolExecutor(max_workers=2)
        self._refs: dict[str, set] = {}
        for k, sf in wmap.items():
            m = re.match(r"model\.layers\.(\d+)\.", k)
            owner: object = int(m.group(1)) if m else "base"
            if m and int(m.group(1)) not in layer_order:
                continue
            self._refs.setdefault(sf, set()).add(owner)

    def _download(self, sf: str) -> Path:
        dest = self.cache_dir / sf
        if dest.exists() and dest.stat().st_size > 1024 * 1024:
            return dest
        t0 = time.perf_counter()
        try:
            from huggingface_hub import hf_hub_download

            p = Path(hf_hub_download(repo_id=self.repo_id, filename=sf, local_dir=str(self.cache_dir)))
        except Exception as e:  # curl fallback (what the DSV4.1 converter uses on Colab)
            print(f"  [download] hf_hub_download failed for {sf} ({e}); falling back to curl", flush=True)
            import subprocess

            tmp = dest.with_suffix(".part")
            cmd = ["curl", "-sL", "--retry", "5", "--retry-delay", "2", "-o", str(tmp),
                   f"https://huggingface.co/{self.repo_id}/resolve/main/{sf}"]
            if os.environ.get("HF_TOKEN"):
                cmd[1:1] = ["-H", f"Authorization: Bearer {os.environ['HF_TOKEN']}"]
            subprocess.run(cmd, check=True)
            tmp.replace(dest)
            p = dest
        gb = p.stat().st_size / 1e9
        print(f"  [download] {sf} ({gb:.2f} GB) in {time.perf_counter() - t0:.1f}s", flush=True)
        return p

    def _ensure(self, sf: str) -> Future:
        with self._lock:
            fut = self._futures.get(sf)
            if fut is None:
                fut = self._pool.submit(self._download, sf)
                self._futures[sf] = fut
            return fut

    def shards_for(self, owner: object) -> list[str]:
        return sorted(sf for sf, owners in self._refs.items() if owner in owners)

    def prefetch_layer(self, layer_idx: int) -> None:
        for sf in self.shards_for(layer_idx):
            self._ensure(sf)

    def prefetch_base(self) -> None:
        for sf in self.shards_for("base"):
            self._ensure(sf)

    def get(self, key: str) -> torch.Tensor:
        sf = self.wmap[key]
        h = self._handles.get(sf)
        if h is None:
            path = self._ensure(sf).result()
            h = safe_open(str(path), framework="pt", device="cpu")
            self._handles[sf] = h
        return h.get_tensor(key)

    def _release(self, owner: object) -> None:
        for sf in list(self._refs):
            owners = self._refs[sf]
            if owner in owners:
                owners.discard(owner)
                if not owners:
                    self._handles.pop(sf, None)
                    p = self.cache_dir / sf
                    if not self.keep and p.exists():
                        drop_page_cache(p)
                        p.unlink(missing_ok=True)
                    del self._refs[sf]

    def release_layer(self, layer_idx: int) -> None:
        self._handles.clear()
        gc.collect()
        self._release(layer_idx)

    def release_base(self) -> None:
        self._handles.clear()
        self._release("base")

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


class HttpRangeSource(_Source):
    """Fetches individual tensors with HTTP Range requests (no full-shard download)."""

    def __init__(self, repo_id: str, wmap: dict[str, str]) -> None:
        self.repo_id = repo_id
        self.wmap = wmap
        self._headers: dict[str, tuple[int, dict]] = {}

    def _header(self, sf: str) -> tuple[int, dict]:
        hit = self._headers.get(sf)
        if hit is None:
            url = f"https://huggingface.co/{self.repo_id}/resolve/main/{sf}"
            req = urllib.request.Request(url, headers={**_auth_headers(), "Range": "bytes=0-7"})
            with urllib.request.urlopen(req, timeout=60) as r:
                hlen = struct.unpack("<Q", r.read(8))[0]
            req = urllib.request.Request(url, headers={**_auth_headers(), "Range": f"bytes=8-{8 + hlen - 1}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                hit = (hlen, json.loads(r.read().decode("utf-8")))
            self._headers[sf] = hit
        return hit

    def get(self, key: str) -> torch.Tensor:
        sf = self.wmap[key]
        hlen, hdr = self._header(sf)
        info = hdr[key]
        s, e = info["data_offsets"]
        url = f"https://huggingface.co/{self.repo_id}/resolve/main/{sf}"
        for attempt in range(5):
            try:
                req = urllib.request.Request(url, headers={**_auth_headers(), "Range": f"bytes={8 + hlen + s}-{8 + hlen + e - 1}"})
                with urllib.request.urlopen(req, timeout=300) as r:
                    buf = bytearray(r.read())
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 * (attempt + 1))
        dt = _DTYPES[info["dtype"]]
        if dt == torch.float8_e4m3fn:
            return torch.frombuffer(buf, dtype=torch.uint8).view(torch.float8_e4m3fn).reshape(info["shape"]).clone()
        return torch.frombuffer(buf, dtype=dt).reshape(info["shape"]).clone()


# =============================================================================================
# Layer conversion
# =============================================================================================
def _expert_fp8(src: _Source, prefix: str, eid: int, proj: str, device: torch.device) -> torch.Tensor:
    base = f"{prefix}.experts.{eid}.{proj}"
    w = src.get(f"{base}.weight")
    s = src.get(f"{base}.weight_scale_inv")
    return dequant_fp8_block(w, s, dtype=torch.float32, device=device)


def _summ(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {}
    t = torch.tensor(vals, dtype=torch.float64)
    return {"mean": round(float(t.mean()), 5), "min": round(float(t.min()), 5)}


@torch.no_grad()
def convert_layer(
    src: _Source,
    arch: Glm53Arch,
    layer_idx: int,
    device: torch.device,
    svd_rank: int,
    gu_bits: int,
    down_bits: int,
    groups: list[int] | None = None,
    verify: bool = True,
    tier_metrics_groups: int = 1,
    n_metric_probes: int = 48,
    hot_experts: int = 0,
) -> tuple[dict[str, torch.Tensor], dict, dict[str, torch.Tensor]]:
    keys = src.layer_keys(layer_idx)
    prefix = moe_prefix(layer_idx)
    is_moe = layer_idx in arch.moe_layer_ids
    out: dict[str, torch.Tensor] = {}
    hot_out: dict[str, torch.Tensor] = {}
    for k in keys:
        if is_moe and k.startswith(f"{prefix}.experts."):
            continue
        out[k] = src.get(k).contiguous()
    report: dict = {"layer": layer_idx, "moe": is_moe, "passthrough_tensors": len(out)}
    if not is_moe:
        return out, report, hot_out

    router_w = out[f"{prefix}.gate.weight"].to(device=device, dtype=torch.float32)
    v_rt_sub, s_rt_sub = router_subspace(router_w)
    norm_w = out.get(f"model.layers.{layer_idx}.post_attention_layernorm.weight")
    probes = make_router_probes(router_w, v_rt_sub, n_metric_probes, norm_w, seed=2026 + layer_idx) if verify else None
    hot_set: list[int] = []
    hot_packed: dict[int, tuple[torch.Tensor, ...]] = {}
    if hot_experts > 0:
        prior = router_prior(router_w, out[f"{prefix}.gate.e_score_correction_bias"], arch.num_experts_per_tok, norm_w, layer_idx)
        hot_set = sorted(torch.argsort(prior, descending=True)[:hot_experts].tolist())
        report["hot_experts"] = hot_set

    agg: dict[str, list[float]] = {}
    target_groups = list(range(arch.n_groups)) if groups is None else groups
    t_groups = time.perf_counter()
    for gi, g_idx in enumerate(target_groups):
        g_list, u_list, d_list = [], [], []
        for m in range(GROUP_SIZE):
            eid = g_idx * GROUP_SIZE + m
            g_list.append(_expert_fp8(src, prefix, eid, "gate_proj", device))
            u_list.append(_expert_fp8(src, prefix, eid, "up_proj", device))
            d_list.append(_expert_fp8(src, prefix, eid, "down_proj", device))
            if eid in hot_set:
                gq, gs = pack_t1_fp4(torch.cat([g_list[-1], u_list[-1]], 0))
                dq, ds = pack_t1_fp4(d_list[-1])
                hot_packed[eid] = (gq.cpu(), gs.cpu(), dq.cpu(), ds.cpu())
        tensors, metrics = convert_expert_group(
            torch.stack(g_list), torch.stack(u_list), torch.stack(d_list),
            router_w, v_rt_sub, s_rt_sub, g_idx,
            svd_rank=svd_rank, gu_tail_bits=gu_bits, down_tail_bits=down_bits,
            metric_probes=probes, tier_metrics=verify and gi < tier_metrics_groups,
        )
        del g_list, u_list, d_list
        for tk, tv in tensors.items():
            out[f"{prefix}.chronos_groups.{g_idx}.{tk}"] = tv
        for mk, mv in metrics.items():
            agg.setdefault(mk, []).extend(mv)
        if device.type == "cuda" and gi % 4 == 3:
            torch.cuda.empty_cache()
        if gi == 0 or (gi + 1) % 8 == 0:
            el = time.perf_counter() - t_groups
            print(f"    [layer {layer_idx:02d}] group {g_idx + 1}/{arch.n_groups} ({el:.1f}s)", flush=True)
    report["groups"] = len(target_groups)
    report["metrics"] = {k: _summ(v) for k, v in agg.items()}
    if hot_packed:
        ids = sorted(hot_packed)
        hp = f"{prefix}.hot_experts."
        hot_out = {
            hp + "ids": torch.tensor(ids, dtype=torch.int32),
            hp + "gu_q": torch.stack([hot_packed[e][0] for e in ids]),
            hp + "gu_s": torch.stack([hot_packed[e][1] for e in ids]),
            hp + "dn_q": torch.stack([hot_packed[e][2] for e in ids]),
            hp + "dn_s": torch.stack([hot_packed[e][3] for e in ids]),
        }
    return out, report, hot_out


# =============================================================================================
# Main
# =============================================================================================
def _layer_done(out_dir: Path, layer_idx: int, arch: Glm53Arch, remote_files: set[str]) -> bool:
    name = layer_shard_name(layer_idx, arch)
    p = out_dir / name
    if p.exists():
        try:
            with safe_open(str(p), framework="pt", device="cpu") as fh:
                ks = set(fh.keys())
            if layer_idx not in arch.moe_layer_ids:
                return len(ks) > 0
            return any(f".chronos_groups.{arch.n_groups - 1}." in k for k in ks)
        except Exception:
            return False
    return name in remote_files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", default=GLM53_REPO_ID)
    ap.add_argument("--source", choices=["hf", "http", "local"], default="hf")
    ap.add_argument("--source-dir", default="", help="Local GLM-5.3-layout checkpoint (for --source local)")
    ap.add_argument("--out-dir", default="/content/glm53_chronos_v75")
    ap.add_argument("--cache-dir", default="/content/hf_cache_glm53")
    ap.add_argument("--layers", default="all", help="'all' (0..78), 'moe', or comma list / ranges like 3,10-12")
    ap.add_argument("--groups", default="", help="Only convert these group ids (validation); output not resumable")
    ap.add_argument("--gu-bits", type=int, default=2, choices=[1, 2], help="2 = standard, 1 = compact disk")
    ap.add_argument("--down-bits", type=int, default=2, choices=[1, 2])
    ap.add_argument("--compact", action="store_true", help="Shortcut for --gu-bits 1 --down-bits 1 (~199 GB total)")
    ap.add_argument("--svd-rank", type=int, default=64)
    ap.add_argument("--hot-experts", type=int, default=0,
                    help="Also write an Upgrade-2 sidecar with the N highest-prior experts/layer packed straight from "
                         "the stock FP8 into runtime T1 (21.2 MB/expert; N=32 adds ~51.6 GB)")
    ap.add_argument("--skip-base", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--keep-cache", action="store_true")
    ap.add_argument("--upload-repo", default="", help=f"e.g. {GLM53_CHRONOS_REPO_ID} (empty disables upload)")
    ap.add_argument("--delete-after-upload", action="store_true")
    ap.add_argument("--max-pending-uploads", type=int, default=6, help="Pause conversion while more uploads are queued")
    ap.add_argument("--dry-run", action="store_true", help="Print the plan/footprint and exit")
    ap.add_argument("--device", default="")
    args = ap.parse_args()
    if args.compact:
        args.gu_bits = args.down_bits = 1

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- config / index ----
    if args.source == "local":
        src_dir = Path(args.source_dir)
        cfg = json.loads((src_dir / "config.json").read_text())
        local_src = LocalSource(src_dir)
        wmap = local_src.wmap
    else:
        cfg = fetch_json(args.repo_id, "config.json")
        wmap = fetch_json(args.repo_id, "model.safetensors.index.json")["weight_map"]
    arch = Glm53Arch.from_config(cfg)

    all_layers = list(range(arch.num_total_layers))
    if args.layers == "all":
        target_layers = all_layers
    elif args.layers == "moe":
        target_layers = arch.moe_layer_ids
    else:
        target_layers = []
        for part in args.layers.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-")
                target_layers.extend(range(int(a), int(b) + 1))
            else:
                target_layers.append(int(part))
    groups = [int(g) for g in args.groups.split(",") if g.strip()] or None

    fp = footprint_report(arch, args.svd_rank)
    print("=" * 100, flush=True)
    dev_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"CHRONOS v{CHRONOS_VERSION} GLM-5.3 CONVERTER  source={args.source}:{args.repo_id}  device={dev_name}", flush=True)
    print(f"  layers={len(target_layers)}  experts/layer={arch.n_routed_experts}  groups/layer={arch.n_groups}  "
          f"tails gu={args.gu_bits}b down={args.down_bits}b  rank={args.svd_rank}", flush=True)
    print(f"  footprint: {json.dumps(fp)}", flush=True)
    print("=" * 100, flush=True)
    if args.dry_run:
        return

    # ---- upload plumbing ----
    hf_api = None
    remote_files: set[str] = set()
    upload_pool = ThreadPoolExecutor(max_workers=1)
    upload_futures: list[Future] = []
    if args.upload_repo:
        from huggingface_hub import HfApi

        hf_api = HfApi(token=os.environ.get("HF_TOKEN"))
        hf_api.create_repo(repo_id=args.upload_repo, repo_type="model", exist_ok=True)
        try:
            remote_files = set(hf_api.list_repo_files(args.upload_repo))
        except Exception:
            remote_files = set()
        print(f"[upload] streaming shards to https://huggingface.co/{args.upload_repo} ({len(remote_files)} files present)", flush=True)

    def _upload(p: Path, delete_after: bool) -> str:
        if hf_api is None or not p.exists():
            return "skipped"
        for attempt in range(4):
            try:
                t0 = time.perf_counter()
                hf_api.upload_file(path_or_fileobj=str(p), path_in_repo=p.name, repo_id=args.upload_repo, repo_type="model")
                print(f"[upload] {p.name} done in {time.perf_counter() - t0:.1f}s", flush=True)
                shutil.rmtree(Path.home() / ".cache" / "huggingface" / "xet", ignore_errors=True)
                if delete_after and p.suffix == ".safetensors":
                    p.unlink(missing_ok=True)
                return p.name
            except Exception as e:  # pragma: no cover - network dependent
                print(f"[upload] {p.name} attempt {attempt + 1} failed: {e}", flush=True)
                time.sleep(10 * (attempt + 1))
        return "error"

    def _queue_upload(p: Path) -> None:
        if hf_api is not None:
            upload_futures.append(upload_pool.submit(_upload, p, args.delete_after_upload))

    # ---- metadata files ----
    if args.source == "local":
        for f in META_FILES:
            if (Path(args.source_dir) / f).exists() and f != "config.json":
                shutil.copy2(Path(args.source_dir) / f, out_dir / f)
    else:
        from huggingface_hub import hf_hub_download

        for f in META_FILES:
            if f == "config.json":
                continue
            try:
                hf_hub_download(repo_id=args.repo_id, filename=f, local_dir=str(out_dir))
            except Exception as e:
                print(f"[meta] optional {f} skipped ({e})", flush=True)
    cfg_out = dict(cfg)
    cfg_out["chronos_holographic_moe"] = {
        "version": CHRONOS_VERSION,
        "arch": "glm53",
        "source_repo": args.repo_id,
        "group_size": GROUP_SIZE,
        "num_groups_per_layer": arch.n_groups,
        "svd_rank": args.svd_rank,
        "gu_tail_bits": args.gu_bits,
        "down_tail_bits": args.down_bits,
        "gu_tail_block": 32 if args.gu_bits == 2 else 64,
        "down_tail_block": 32 if args.down_bits == 2 else 64,
        "key_generator": KEY_GENERATOR,
        "hot_experts_per_layer": args.hot_experts,
        "runtime_tier_formats": {"t1": "e2m1_g32_fp16", "t2": "lloydmax2_g64_fp16", "t3": "sign1_g128_fp16"},
        "moe_layers": arch.moe_layer_ids,
        "shard_pattern": layer_shard_name(0, arch).replace("00000", "{layer:05d}"),
        "base_shard": base_shard_name(),
    }
    (out_dir / "config.json").write_text(json.dumps(cfg_out, indent=2))
    for f in META_FILES:
        if (out_dir / f).exists():
            _queue_upload(out_dir / f)

    # ---- source ----
    if args.source == "local":
        src: _Source = local_src
    elif args.source == "http":
        src = HttpRangeSource(args.repo_id, wmap)
    else:
        src = HFStreamingSource(args.repo_id, Path(args.cache_dir), wmap, target_layers, keep=args.keep_cache)

    progress_path = out_dir / "glm53_chronos_progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {"layers": {}}

    # ---- base shard (embed / lm_head / norm) ----
    base_path = out_dir / base_shard_name()
    if not args.skip_base and groups is None and not base_path.exists() and base_shard_name() not in remote_files:
        t0 = time.perf_counter()
        base_keys = [k for k in wmap if not k.startswith("model.layers.")]
        base = {k: src.get(k).contiguous() for k in base_keys}
        save_file(base, str(base_path), metadata={"format": "pt", "chronos": "glm53-base"})
        progress["base_keys"] = sorted(base.keys())
        progress_path.write_text(json.dumps(progress, indent=2))
        del base
        print(f"[base] {base_path.name} ({base_path.stat().st_size / 1e9:.2f} GB) in {time.perf_counter() - t0:.1f}s", flush=True)
        _queue_upload(base_path)
    if isinstance(src, HFStreamingSource):
        src.release_base()

    # ---- layers ----
    pending = [l for l in target_layers if groups is not None or not _layer_done(out_dir, l, arch, remote_files)]
    if isinstance(src, HFStreamingSource):
        # GLM-5.3 shards follow lexicographic layer order (0, 1, 10..19, 2, 20, ...); converting in shard order lets
        # every straddling shard be freed right after its two consumers -> ~4 source shards on disk at any time.
        def _first_shard(l: int) -> str:
            return min(src.shards_for(l) or ["~"])

        pending.sort(key=_first_shard)
    skipped = [l for l in target_layers if l not in pending]
    if skipped:
        print(f"[resume] {len(skipped)} layers already converted: {skipped}", flush=True)
        for l in skipped:
            p = out_dir / layer_shard_name(l, arch)
            if p.exists() and layer_shard_name(l, arch) not in remote_files:
                _queue_upload(p)
            if isinstance(src, HFStreamingSource):
                src.release_layer(l)
    if pending:
        src.prefetch_layer(pending[0])
    t_all = time.perf_counter()
    for i, l_idx in enumerate(pending):
        # Upload back-pressure: with --delete-after-upload the local disk only ever holds a few finished shards.
        while hf_api is not None and sum(1 for f in upload_futures if not f.done()) > args.max_pending_uploads:
            time.sleep(5)
        if i + 1 < len(pending):
            src.prefetch_layer(pending[i + 1])
        t0 = time.perf_counter()
        tensors, rep, hot_tensors = convert_layer(
            src, arch, l_idx, device, args.svd_rank, args.gu_bits, args.down_bits,
            groups=groups, verify=not args.no_verify, hot_experts=args.hot_experts,
        )
        if hot_tensors:
            hot_path = out_dir / hot_shard_name(l_idx, args.hot_experts, arch)
            save_file(hot_tensors, str(hot_path), metadata={"format": "pt", "chronos": f"glm53-hot-{l_idx}"})
            drop_page_cache(hot_path)
            if groups is None:
                _queue_upload(hot_path)
            del hot_tensors
        name = layer_shard_name(l_idx, arch)
        if groups is not None:
            name = name.replace(".safetensors", f".groups-{'-'.join(map(str, groups))}.partial.safetensors")
        out_path = out_dir / name
        save_file(tensors, str(out_path), metadata={"format": "pt", "chronos": f"glm53-layer-{l_idx}"})
        drop_page_cache(out_path)
        rep["size_gb"] = round(out_path.stat().st_size / 1e9, 4)
        rep["elapsed_s"] = round(time.perf_counter() - t0, 1)
        rep["free_disk_gb"] = round(shutil.disk_usage(out_dir).free / 1e9, 1)
        rep["keys"] = sorted(tensors.keys())
        progress["layers"][str(l_idx)] = rep
        progress_path.write_text(json.dumps(progress, indent=2))
        m = rep.get("metrics", {})
        fc = m.get("func_cos", {})
        print(
            f"[layer {l_idx:02d}] {rep['size_gb']:.3f} GB in {rep['elapsed_s']:.1f}s | func_cos={fc} "
            f"w_cos_gu={m.get('w_cos_gu', {})} w_cos_down={m.get('w_cos_down', {})} free={rep['free_disk_gb']} GB",
            flush=True,
        )
        del tensors
        src.release_layer(l_idx)
        if groups is None:
            _queue_upload(out_path)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        done = i + 1
        eta = (time.perf_counter() - t_all) / done * (len(pending) - done)
        print(f"[progress] {done}/{len(pending)} layers, ETA {eta / 60:.1f} min", flush=True)

    # ---- index ----
    if groups is None:
        weight_map: dict[str, str] = {}
        total = 0
        names = [base_shard_name()] + [layer_shard_name(l, arch) for l in all_layers]
        for nm in names:
            p = out_dir / nm
            if p.exists():
                with safe_open(str(p), framework="pt", device="cpu") as fh:
                    for k in fh.keys():
                        weight_map[k] = nm
                total += p.stat().st_size
            else:
                rec_layer = re.search(r"layer-(\d+)-of", nm)
                rec = progress["layers"].get(str(int(rec_layer.group(1)))) if rec_layer else None
                rec_keys = rec.get("keys") if rec else (progress.get("base_keys") if nm == base_shard_name() else None)
                for k in rec_keys or []:
                    weight_map[k] = nm
        complete = all((out_dir / n).exists() or n in remote_files for n in names)
        if weight_map and complete:
            idx_path = out_dir / "model.safetensors.index.json"
            idx_path.write_text(json.dumps({"metadata": {"total_size": total, "chronos": "glm53"}, "weight_map": weight_map}, indent=2))
            _queue_upload(idx_path)
            print(f"[index] wrote {idx_path.name} ({len(weight_map)} tensors)", flush=True)
        elif not complete:
            print("[index] skipped: not all layer shards are present locally (re-run with all shards to build it, "
                  "or run scripts/build_glm53_chronos_index.py against the upload repo)", flush=True)
    _queue_upload(progress_path)

    src.close()
    if upload_futures:
        print(f"[upload] waiting for {len(upload_futures)} uploads ...", flush=True)
        for f in upload_futures:
            f.result()
    upload_pool.shutdown(wait=True)
    print(f"DONE: {len(pending)} layers converted in {(time.perf_counter() - t_all) / 60:.1f} min -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
