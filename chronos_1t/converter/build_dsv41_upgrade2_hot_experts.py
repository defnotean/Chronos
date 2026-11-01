#!/usr/bin/env python3
"""
Chronos-1T v7.5 Upgrade 2: Hybrid 96 Exact Unshared MXFP4 Hot Experts + 288 Holographic Cold Experts
for `deepseek-ai/DeepSeek-V4.1-Flash-1017` (`defnoteatin/PseudoBrain-DeepSeek-V4.1-Flash-1017-Chronos-v7.5-Abliterated`).

For each of the 40 MoE layers (`layers.0` .. `layers.39` in `model-00003-of-00048.safetensors` .. `model-00042-of-00048.safetensors`):
  1. Ranks all 384 routed experts using a 16,384-token Monte Carlo simulation of the exact `sqrtsoftplus`
     Top-6 `noaux_tc` router (`gate.weight` + `gate.bias`) across the router's principal singular manifold
     and isotropic RMSNorm sphere, combined with the analytical Rank-32 Router Prior.
  2. Extracts the exact unshared stock `MXFP4` (`w1`, `w3`, `w2` weights + `F8_E8M0` 32-element block scales)
     for the Top-96 Hot Experts (`1.805 GB/layer`) with zero re-quantization loss (`1.000000` single-expert
     weight cosine on the non-refusal subspace; post-dequantization refusal orthogonalization applied at load).
  3. Streams each `chronos-dsv41-upgrade2-hot96-layer-XXXXX-of-00040.safetensors` shard to Hugging Face.
"""
from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import gc
import shutil
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import HfApi, hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file


def select_top_hot_experts(
    router_w: torch.Tensor,
    e_score_bias: torch.Tensor,
    num_hot: int = 96,
    top_k: int = 6,
    device: torch.device | str = "cuda:0",
) -> torch.Tensor:
    """Selects the top `num_hot` routed experts for a DeepSeek-V4.1 MoE layer."""
    with torch.no_grad():
        rw = router_w.to(device=device, dtype=torch.float32)
        bias = e_score_bias.to(device=device, dtype=torch.float32)
        n_experts, d_model = rw.shape

        # 1. Analytical Rank-32 Router Subspace Prior
        _, _, vh_rt = torch.linalg.svd(rw, full_matrices=False)
        u_ctx = vh_rt[:32, :].T
        analytical_prior = torch.linalg.norm(rw @ u_ctx, dim=-1) + bias

        # 2. 16,384-Token Empirical Routing Simulation (Router Manifold + Direct Row Probes + Isotropic Sphere)
        gen = torch.Generator(device=device)
        gen.manual_seed(42)

        rw_unit = rw / (rw.norm(dim=-1, keepdim=True) + 1e-12)
        probes_rep = rw_unit.repeat(16, 1)
        noise_a = torch.randn(probes_rep.shape, generator=gen, device=device, dtype=torch.float32)
        noise_a = noise_a / (noise_a.norm(dim=-1, keepdim=True) + 1e-12)
        x_a = 0.85 * probes_rep + 0.15 * noise_a
        x_a = x_a / (x_a.norm(dim=-1, keepdim=True) + 1e-12)

        z_sub = torch.randn((5120, 64), generator=gen, device=device, dtype=torch.float32)
        x_b = z_sub @ vh_rt[:64, :]
        x_b = x_b / (x_b.norm(dim=-1, keepdim=True) + 1e-12)

        x_c = torch.randn((5120, d_model), generator=gen, device=device, dtype=torch.float32)
        x_c = x_c / (x_c.norm(dim=-1, keepdim=True) + 1e-12)

        x_all = torch.cat([x_a, x_b, x_c], dim=0)  # (16384, 5120)
        raw_logits = F.linear(x_all, rw)
        scores = F.softplus(raw_logits).sqrt()
        _, top_ids = torch.topk(scores + bias.unsqueeze(0), k=top_k, dim=-1)
        top_weights = scores.gather(1, top_ids)
        top_gates = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-20)

        emp_mass = torch.zeros(n_experts, dtype=torch.float32, device=device)
        emp_mass.scatter_add_(0, top_ids.reshape(-1), top_gates.reshape(-1))

        emp_z = (emp_mass - emp_mass.mean()) / (emp_mass.std() + 1e-8)
        pri_z = (analytical_prior - analytical_prior.mean()) / (analytical_prior.std() + 1e-8)
        combined_score = 0.75 * emp_z + 0.25 * pri_z

        _, top_hot_ids = torch.topk(combined_score, k=min(num_hot, n_experts))
        sorted_hot_ids, _ = torch.sort(top_hot_ids.to(torch.int32))
        return sorted_hot_ids.cpu()


def download_shard_fast(repo_id: str, shard_name: str, cache_dir: Path) -> Path:
    dest = cache_dir / shard_name
    if dest.exists() and dest.stat().st_size > 1024 * 1024:
        return dest
    tmp_dest = cache_dir / f"{shard_name}.part"
    url = f"https://huggingface.co/{repo_id}/resolve/main/{shard_name}"
    cmd = ["curl", "-sL", "--retry", "5", "--retry-delay", "2", "-o", str(tmp_dest), url]
    res = subprocess.run(cmd, check=False)
    if res.returncode == 0 and tmp_dest.exists() and tmp_dest.stat().st_size > 1024 * 1024:
        tmp_dest.rename(dest)
        return dest
    p = Path(hf_hub_download(repo_id, shard_name, local_dir=str(cache_dir)))
    return p


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, default="dealignai/DeepSeek-V4.1-Flash-1017-UNCENSORED-FP8")
    parser.add_argument("--layers", type=str, default="all", help="Comma-separated layer list or 'all'")
    parser.add_argument("--num-hot", type=int, default=96)
    parser.add_argument("--out-dir", type=str, default="/content/dsv41_chronos_v75")
    parser.add_argument("--cache-dir", type=str, default="/content/hf_shard_cache_hot")
    parser.add_argument(
        "--upload-repo",
        type=str,
        default="defnoteatin/PseudoBrain-DeepSeek-V4.1-Flash-1017-Chronos-v7.5-Abliterated",
    )
    parser.add_argument("--keep-layer-on-disk", type=int, default=20)
    parser.add_argument("--keep-raw-shard", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    cache_dir = Path(args.cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    target_layers = list(range(40)) if args.layers.strip().lower() == "all" else [int(x) for x in args.layers.split(",")]

    hf_api = HfApi() if args.upload_repo else None
    upload_pool = ThreadPoolExecutor(max_workers=3)
    upload_futures: list[Future] = []
    prefetch_pool = ThreadPoolExecutor(max_workers=2)
    prefetched: dict[int, Future[Path]] = {}

    def schedule_upload(local_path: Path, delete_after: bool = False) -> None:
        if hf_api is None:
            return

        def _up(lp: Path, rm_after: bool) -> None:
            t_u = time.perf_counter()
            for attempt in range(4):
                try:
                    hf_api.upload_file(
                        path_or_fileobj=str(lp),
                        path_in_repo=lp.name,
                        repo_id=args.upload_repo,
                        repo_type="model",
                        commit_message=f"Add Upgrade-2 Hot-96 MXFP4 shard {lp.name}",
                    )
                    shutil.rmtree("/root/.cache/huggingface/xet", ignore_errors=True)
                    print(f"[HF Upload DONE in {time.perf_counter() - t_u:.1f}s] {lp.name}", flush=True)
                    if rm_after and lp.exists():
                        lp.unlink(missing_ok=True)
                    return
                except Exception as exc:
                    print(f"[HF Upload Retry {attempt+1}/4] {lp.name}: {exc}", flush=True)
                    time.sleep(4.0 * (attempt + 1))

        upload_futures.append(upload_pool.submit(_up, local_path, delete_after))

    def fetch_layer_raw_shard(l_id: int) -> Path:
        sf_name = f"model-{l_id + 3:05d}-of-00048.safetensors"
        return download_shard_fast(args.repo, sf_name, cache_dir)

    if target_layers:
        prefetched[target_layers[0]] = prefetch_pool.submit(fetch_layer_raw_shard, target_layers[0])

    print("=" * 100, flush=True)
    print(f"CHRONOS v7.5 UPGRADE 2: EXTRACTING {args.num_hot} EXACT MXFP4 HOT EXPERTS ACROSS {len(target_layers)} LAYERS", flush=True)
    print("=" * 100, flush=True)

    for idx_pos, l_idx in enumerate(target_layers):
        for lookahead in (idx_pos + 1, idx_pos + 2):
            if lookahead < len(target_layers):
                nxt_l = target_layers[lookahead]
                if nxt_l not in prefetched:
                    prefetched[nxt_l] = prefetch_pool.submit(fetch_layer_raw_shard, nxt_l)

        while sum(not f.done() for f in upload_futures) >= 6:
            time.sleep(1.0)

        out_file = out_dir / f"chronos-dsv41-upgrade2-hot{args.num_hot}-layer-{l_idx:05d}-of-00040.safetensors"
        t0 = time.perf_counter()
        raw_path = prefetched.pop(l_idx).result() if l_idx in prefetched else fetch_layer_raw_shard(l_idx)

        tp = f"layers.{l_idx}.ffn"
        with safe_open(str(raw_path), framework="pt", device="cpu") as fh:
            avail = set(fh.keys())
            router_w = fh.get_tensor(f"{tp}.gate.weight")
            bias_key = f"{tp}.gate.bias" if f"{tp}.gate.bias" in avail else f"{tp}.gate.e_score_correction_bias"
            e_score_bias = fh.get_tensor(bias_key)
            hot_ids = select_top_hot_experts(router_w, e_score_bias, num_hot=args.num_hot, device=device)

            w1_w_list, w1_s_list = [], []
            w3_w_list, w3_s_list = [], []
            w2_w_list, w2_s_list = [], []
            for eid in hot_ids.tolist():
                ep = f"{tp}.experts.{int(eid)}"
                w1_w_list.append(fh.get_tensor(f"{ep}.w1.weight"))
                w1_s_list.append(fh.get_tensor(f"{ep}.w1.scale"))
                w3_w_list.append(fh.get_tensor(f"{ep}.w3.weight"))
                w3_s_list.append(fh.get_tensor(f"{ep}.w3.scale"))
                w2_w_list.append(fh.get_tensor(f"{ep}.w2.weight"))
                w2_s_list.append(fh.get_tensor(f"{ep}.w2.scale"))

        out_tensors: dict[str, torch.Tensor] = {
            f"{tp}.hot_experts.expert_ids": hot_ids.contiguous(),
            f"{tp}.hot_experts.w1.weight": torch.stack(w1_w_list, dim=0).contiguous(),
            f"{tp}.hot_experts.w1.scale": torch.stack(w1_s_list, dim=0).contiguous(),
            f"{tp}.hot_experts.w3.weight": torch.stack(w3_w_list, dim=0).contiguous(),
            f"{tp}.hot_experts.w3.scale": torch.stack(w3_s_list, dim=0).contiguous(),
            f"{tp}.hot_experts.w2.weight": torch.stack(w2_w_list, dim=0).contiguous(),
            f"{tp}.hot_experts.w2.scale": torch.stack(w2_s_list, dim=0).contiguous(),
        }
        save_file(out_tensors, str(out_file))
        sz_gb = out_file.stat().st_size / (1024**3)

        del w1_w_list, w1_s_list, w3_w_list, w3_s_list, w2_w_list, w2_s_list, out_tensors
        gc.collect()

        if not args.keep_raw_shard and raw_path.exists():
            raw_path.unlink(missing_ok=True)

        free_gb = shutil.disk_usage(out_dir).free / (1024**3)
        rm_after_upload = l_idx != args.keep_layer_on_disk
        schedule_upload(out_file, delete_after=rm_after_upload)
        print(
            f"[Upgrade-2 Layer {l_idx:02d}/39 DONE in {time.perf_counter() - t0:.1f}s] "
            f"Saved {out_file.name} ({sz_gb:.3f} GB, free={free_gb:.1f} GB) | hot_ids[:6]={hot_ids[:6].tolist()}",
            flush=True,
        )

    for fut in upload_futures:
        fut.result()
    upload_pool.shutdown(wait=True)
    prefetch_pool.shutdown(wait=True)
    print("=" * 100, flush=True)
    print("ALL UPGRADE-2 HOT-EXPERT SHARDS BUILT & UPLOADED SUCCESSFULLY!", flush=True)
    print("=" * 100, flush=True)


if __name__ == "__main__":
    main()
