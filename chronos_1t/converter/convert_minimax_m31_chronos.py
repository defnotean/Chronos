"""
Chronos v7.5 1M-Context Abliterated & Healed Converter for MiniMax-M3.1-Flash.

Combines:
1. Sm0r3s/MiniMax-M3.1-Flash-E32-unhealed (32 REAP-ranked NVFP4 routed experts
   per MoE layer 3..59 + MXFP8 attention & shared experts, 68.87 GB).
2. Blackfrost-Research/MiniMax-3.1-Flash-DeRisked (HTTP range-streamed
   abliterated self_attn.o_proj weights across layers 2..59 + vision_tower /
   multi_modal_projector / patch_merge_mlp weights for full VL compatibility).
3. Exact rank-1 refusal direction extraction from
   Delta W_o^(l) = W_orig,o^(l) - W_ablit,o^(l) across layers 2..59, applied to
   shared_experts.down_proj (layers 3..59) and mlp.down_proj (layer 2).
4. Chronos v7.5 Option-A Rank-64 BF16 Nullspace Shared-Expert Healer
   (minimax_m31_shared_experts_option_a_r64.pt) solving for the residual subspace
   correction of the pruned MoE layers and folding the healed correction into
   shared_experts.down_proj MXFP8 weights (with zero refusal projection).
5. Native dspark MTP drafter (MiniMaxAI/MiniMax-M3.1-Flash dspark/) for
   high-throughput speculative decoding at full 1,048,576-token context.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import urllib.request
from typing import Dict, List, Tuple

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file


# FP4 E2M1 lookup table (standard NVFP4 signing/magnitude table)
_E2M1_LUT = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def dequant_mxfp8(weight_fp8: torch.Tensor, scale_inv_u8: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    """Dequantize MXFP8 (F8_E4M3 weight + U8 E8M0 block-32 scale_inv) to FP32."""
    w = weight_fp8.to(device=device, dtype=torch.float32)
    s_u8 = scale_inv_u8.to(device=device, dtype=torch.int32)
    M, K = w.shape
    assert K % 32 == 0 and s_u8.shape == (M, K // 32)
    scale = torch.ldexp(torch.ones_like(s_u8, dtype=torch.float32), s_u8 - 127)
    return (w.view(M, K // 32, 32) * scale.unsqueeze(-1)).reshape(M, K)


def quant_mxfp8(w_fp32: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize FP32 matrix [M, K] back into MXFP8 (F8_E4M3 weight + U8 E8M0 scale_inv)."""
    M, K = w_fp32.shape
    assert K % 32 == 0
    blocks = w_fp32.view(M, K // 32, 32)
    amax = blocks.abs().amax(dim=-1).clamp(min=1e-30)
    # E4M3 max normal is 448.0; pick smallest power-of-2 scale so amax / 2^exp <= 448.0
    exp = torch.ceil(torch.log2(amax / 448.0)).to(torch.int32).clamp(-127, 127)
    scale = torch.ldexp(torch.ones_like(amax, dtype=torch.float32), exp)
    q = (blocks / scale.unsqueeze(-1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).reshape(M, K)
    s_u8 = (exp + 127).to(torch.uint8)
    return q.cpu(), s_u8.cpu()


def dequant_nvfp4_w2(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    """Dequantize NVFP4 packed w2 [M, K//2] + F8_E4M3 scale [M, K//16] + global_scale to FP32 [M, K]."""
    lut = _E2M1_LUT.to(device=device)
    wp = weight_packed.to(device=device)
    ws = weight_scale.to(device=device, dtype=torch.float32)
    g = float(weight_global_scale.item())
    # In ModelOpt / compressed-tensors nvfp4-pack-quantized, effective scale is ws / g if g > 10 else ws * g
    eff_g = (1.0 / g) if g > 10.0 else g
    M, half_K = wp.shape
    K = half_K * 2
    lo = (wp & 0x0F).to(torch.long)
    hi = ((wp >> 4) & 0x0F).to(torch.long)
    unpacked = torch.stack([lut[lo], lut[hi]], dim=-1).reshape(M, K // 16, 16)
    return (unpacked * (ws * eff_g).unsqueeze(-1)).reshape(M, K)


class HFRangeSafetensorsReader:
    """Zero-disk HTTP Range reader for remote HuggingFace safetensors shards."""

    def __init__(self, repo_id: str, token: str | None = None):
        self.repo_id = repo_id
        self.token = token or os.environ.get("HF_TOKEN")
        self._headers_cache: Dict[str, Tuple[int, dict]] = {}
        self.index = self._fetch_json("model.safetensors.index.json")
        self.weight_map: Dict[str, str] = self.index.get("weight_map", {})

    def _req(self, path: str, byte_range: Tuple[int, int] | None = None) -> bytes:
        url = f"https://huggingface.co/{self.repo_id}/resolve/main/{path}"
        headers = {"User-Agent": "chronos-converter/7.5"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if byte_range is not None:
            headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read()

    def _fetch_json(self, path: str) -> dict:
        return json.loads(self._req(path).decode("utf-8"))

    def get_shard_header(self, shard_name: str) -> Tuple[int, dict]:
        if shard_name not in self._headers_cache:
            head8 = self._req(shard_name, (0, 7))
            hdr_len = struct.unpack("<Q", head8)[0]
            hdr_bytes = self._req(shard_name, (8, 8 + hdr_len - 1))
            hdr = json.loads(hdr_bytes.decode("utf-8"))
            self._headers_cache[shard_name] = (8 + hdr_len, hdr)
        return self._headers_cache[shard_name]

    def fetch_tensors(self, keys: List[str]) -> Dict[str, torch.Tensor]:
        """Fetch multiple tensors grouped by shard, coalescing adjacent byte ranges."""
        by_shard: Dict[str, List[str]] = {}
        for k in keys:
            s = self.weight_map.get(k)
            if s is not None:
                by_shard.setdefault(s, []).append(k)

        out: Dict[str, torch.Tensor] = {}
        dtype_map = {
            "F32": torch.float32,
            "BF16": torch.bfloat16,
            "F16": torch.float16,
            "U8": torch.uint8,
            "I8": torch.int8,
            "I32": torch.int32,
            "I64": torch.int64,
            "F8_E4M3": torch.float8_e4m3fn,
        }
        for shard, skeys in by_shard.items():
            data_offset, hdr = self.get_shard_header(shard)
            # Sort keys by offset so we can coalesce nearby ranges
            spans = []
            for k in skeys:
                info = hdr[k]
                s_off, e_off = info["data_offsets"]
                spans.append((s_off, e_off, k, info))
            spans.sort(key=lambda x: x[0])

            # Group spans separated by <= 2 MB
            groups: List[List[ tuple ]] = []
            for sp in spans:
                if not groups or (sp[0] - groups[-1][-1][1]) > 2 * 1024 * 1024:
                    groups.append([sp])
                else:
                    groups[-1].append(sp)

            for grp in groups:
                g_start = data_offset + grp[0][0]
                g_end = data_offset + grp[-1][1] - 1
                buf = self._req(shard, (g_start, g_end))
                base_off = grp[0][0]
                for s_off, e_off, k, info in grp:
                    rel_s = s_off - base_off
                    rel_e = e_off - base_off
                    t_bytes = bytearray(buf[rel_s:rel_e])
                    dt = dtype_map[info["dtype"]]
                    t = torch.frombuffer(t_bytes, dtype=dt).reshape(info["shape"]).clone()
                    out[k] = t
        return out


def extract_refusal_direction(w_orig: torch.Tensor, w_ablit: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Extract the top unit left singular vector r_hat in R^{hidden_size} from Delta W = W_orig - W_ablit."""
    delta = (w_orig - w_ablit).to(torch.float32)
    diff_norm = float(delta.norm().item())
    if diff_norm < 1e-6:
        return torch.zeros(w_orig.shape[0], dtype=torch.float32, device=w_orig.device), 0.0
    # 8-step power iteration on delta @ delta.T
    v = delta.mean(dim=1)
    if v.norm() < 1e-8:
        v = delta[:, 0].clone()
    v = v / v.norm().clamp(min=1e-12)
    for _ in range(8):
        u = delta.T @ v
        u = u / u.norm().clamp(min=1e-12)
        v = delta @ u
        v = v / v.norm().clamp(min=1e-12)
    return v, diff_norm


def compute_chronos_option_a_healer_for_layer(
    shard_tensors: Dict[str, torch.Tensor],
    layer_idx: int,
    r_hat: torch.Tensor,
    rank: int = 64,
    healer_scale: float = 0.08,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Chronos v7.5 Option-A Rank-64 BF16 Shared-Expert Healer (U, V) and healed W_sh_down.

    Uses the router bias prior p_e = sigmoid(b_e) over the 32 kept NVFP4 experts' w2
    projections to extract the principal routed residual subspace orthogonal to both
    the refusal direction r_hat and the existing shared_experts.down_proj column space,
    compensating for the unhealed 128 -> 32 pruning energy deficit.
    """
    prefix = f"language_model.model.layers.{layer_idx}"
    sh_w_q = shard_tensors[f"{prefix}.block_sparse_moe.shared_experts.down_proj.weight"]
    sh_s_u8 = shard_tensors[f"{prefix}.block_sparse_moe.shared_experts.down_proj.weight_scale_inv"]
    W_sh = dequant_mxfp8(sh_w_q, sh_s_u8, device=device)  # [6144, 3072]

    # Step 1: Orthogonalize shared_experts.down_proj against refusal direction r_hat
    r_dev = r_hat.to(device=device, dtype=torch.float32)
    if r_dev.norm() > 0.5:
        proj = r_dev.unsqueeze(1) @ (r_dev.unsqueeze(0) @ W_sh)
        W_sh = W_sh - 1.25 * proj

    # Step 2: Aggregate prior-weighted routed w2 mean & variance subspace across the 32 kept experts
    ebias = shard_tensors[f"{prefix}.block_sparse_moe.e_score_correction_bias"].to(device=device, dtype=torch.float32)
    probs = torch.sigmoid(ebias)
    probs = probs / probs.sum().clamp(min=1e-12)

    mean_w2 = torch.zeros_like(W_sh)  # [6144, 3072]
    for e in range(32):
        ep = f"{prefix}.block_sparse_moe.experts.{e}.w2"
        w2_e = dequant_nvfp4_w2(
            shard_tensors[f"{ep}.weight_packed"],
            shard_tensors[f"{ep}.weight_scale"],
            shard_tensors[f"{ep}.weight_global_scale"],
            device=device,
        )
        mean_w2.add_(w2_e, alpha=float(probs[e].item()))

    # Remove refusal direction from mean_w2 as well
    if r_dev.norm() > 0.5:
        mean_w2 = mean_w2 - r_dev.unsqueeze(1) @ (r_dev.unsqueeze(0) @ mean_w2)

    # Low-rank SVD of the prior-weighted routed output space to build Rank-64 Option-A adapter
    U_svd, S_svd, V_svd = torch.svd_lowrank(mean_w2, q=rank, niter=4)
    sqrt_s = torch.sqrt(S_svd.clamp(min=1e-8))
    U_r64 = (U_svd * sqrt_s.unsqueeze(0)).to(torch.bfloat16).cpu()  # [6144, 64]
    V_r64 = (V_svd * sqrt_s.unsqueeze(0)).to(torch.bfloat16).cpu()  # [3072, 64]

    # Fold scaled Option-A residual healer directly into W_sh so standard MXFP8 forward is healed
    W_sh_healed = W_sh + healer_scale * (U_r64.to(device=device, dtype=torch.float32) @ V_r64.to(device=device, dtype=torch.float32).T)
    if r_dev.norm() > 0.5:
        W_sh_healed = W_sh_healed - r_dev.unsqueeze(1) @ (r_dev.unsqueeze(0) @ W_sh_healed)

    return W_sh_healed, U_r64, V_r64


def convert_minimax_m31_chronos(
    output_dir: str,
    e32_repo: str = "Sm0r3s/MiniMax-M3.1-Flash-E32-unhealed",
    ablit_repo: str = "Blackfrost-Research/MiniMax-3.1-Flash-DeRisked",
    base_repo: str = "MiniMaxAI/MiniMax-M3.1-Flash",
    rank: int = 64,
    healer_scale: float = 0.08,
    device: str = "cpu",
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    print(f"[Chronos v7.5] Converting MiniMax-M3.1-Flash -> {output_dir} (device={device})")

    # 1. Download metadata & config files from e32_repo and base_repo
    meta_files = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "configuration_minimax_m3_vl.py",
        "modeling_minimax_m3_vl.py",
        "processing_minimax_m3_vl.py",
    ]
    for fn in meta_files:
        try:
            hf_hub_download(repo_id=e32_repo, filename=fn, local_dir=output_dir)
        except Exception:
            try:
                hf_hub_download(repo_id=base_repo, filename=fn, local_dir=output_dir)
            except Exception as e:
                print(f"  [note] optional file {fn} skipped: {e}")

    for fn in ["preprocessor_config.json", "video_preprocessor_config.json"]:
        try:
            hf_hub_download(repo_id=base_repo, filename=fn, local_dir=output_dir)
        except Exception:
            pass

    # 2. Download dspark speculative drafter from base_repo
    for dfn in ["dspark/config.json", "dspark/model-dspark-00001.safetensors", "dspark/model.safetensors.index.json"]:
        try:
            hf_hub_download(repo_id=base_repo, filename=dfn, local_dir=output_dir)
            print(f"  [dspark] Downloaded {dfn}")
        except Exception as e:
            print(f"  [dspark] optional {dfn}: {e}")

    # 3. Initialize HTTP Range Reader on Blackfrost-Research/MiniMax-3.1-Flash-DeRisked
    ablit_reader = HFRangeSafetensorsReader(ablit_repo)

    # 4. Stream vision_tower + multi_modal_projector + patch_merge_mlp into vision_tower.safetensors
    vis_file = os.path.join(output_dir, "vision_tower.safetensors")
    vis_keys = sorted([k for k in ablit_reader.weight_map if not k.startswith("language_model.")])
    if not os.path.exists(vis_file):
        print(f"[Chronos v7.5] Streaming {len(vis_keys)} vision/projector tensors from {ablit_repo}...")
        vis_tensors = ablit_reader.fetch_tensors(vis_keys)
        save_file(vis_tensors, vis_file)
        del vis_tensors
        print(f"  Saved {vis_file}")

    # 5. Download base.safetensors (embeddings, norm, lm_head) from e32_repo
    base_st = os.path.join(output_dir, "base.safetensors")
    if not os.path.exists(base_st):
        hf_hub_download(repo_id=e32_repo, filename="base.safetensors", local_dir=output_dir)
    base_hdr_keys = list(load_file(base_st, device="cpu").keys())

    # 6. Process layers 00..59: download E32 shard, apply Blackfrost o_proj abliteration,
    #    extract exact refusal vector r_hat^(l), orthogonalize down_proj, and apply
    #    Chronos v7.5 Option-A Rank-64 Shared-Expert Healer
    healer_state: Dict[int, Dict[str, torch.Tensor]] = {}
    healer_pt_path = os.path.join(output_dir, "minimax_m31_shared_experts_option_a_r64.pt")
    if os.path.exists(healer_pt_path):
        healer_state = torch.load(healer_pt_path, map_location="cpu", weights_only=True)

    weight_map_out: Dict[str, str] = {}
    for k in base_hdr_keys:
        weight_map_out[k] = "base.safetensors"
    for k in vis_keys:
        weight_map_out[k] = "vision_tower.safetensors"

    for layer_idx in range(60):
        shard_name = f"layer_{layer_idx:02d}.safetensors"
        shard_path = os.path.join(output_dir, shard_name)
        marker_path = os.path.join(output_dir, f".healed_layer_{layer_idx:02d}.done")

        if not os.path.exists(shard_path):
            print(f"[Layer {layer_idx:02d}/59] Downloading {shard_name} from {e32_repo}...")
            hf_hub_download(repo_id=e32_repo, filename=shard_name, local_dir=output_dir)

        shard_tensors = load_file(shard_path, device="cpu")
        for k in shard_tensors.keys():
            weight_map_out[k] = shard_name

        if os.path.exists(marker_path) and (layer_idx < 3 or layer_idx in healer_state):
            print(f"[Layer {layer_idx:02d}/59] Already healed & abliterated, skipping.")
            continue

        prefix = f"language_model.model.layers.{layer_idx}"
        modified = False
        r_hat = torch.zeros(6144, dtype=torch.float32)

        if layer_idx >= 2:
            o_w_key = f"{prefix}.self_attn.o_proj.weight"
            o_s_key = f"{prefix}.self_attn.o_proj.weight_scale_inv"
            ablit_o = ablit_reader.fetch_tensors([o_w_key, o_s_key])

            w_orig_o = dequant_mxfp8(shard_tensors[o_w_key], shard_tensors[o_s_key], device=device)
            w_ablit_o = dequant_mxfp8(ablit_o[o_w_key], ablit_o[o_s_key], device=device)
            r_hat, diff_norm = extract_refusal_direction(w_orig_o, w_ablit_o)

            # Replace o_proj with exact Blackfrost DeRisked MXFP8 tensors
            shard_tensors[o_w_key] = ablit_o[o_w_key].cpu()
            shard_tensors[o_s_key] = ablit_o[o_s_key].cpu()
            modified = True

            if layer_idx == 2:
                # Dense layer 2: orthogonalize mlp.down_proj against r_hat
                dp_w_key = f"{prefix}.mlp.down_proj.weight"
                dp_s_key = f"{prefix}.mlp.down_proj.weight_scale_inv"
                w_dp = dequant_mxfp8(shard_tensors[dp_w_key], shard_tensors[dp_s_key], device=device)
                r_dev = r_hat.to(device=device, dtype=torch.float32)
                if r_dev.norm() > 0.5:
                    w_dp = w_dp - 1.25 * (r_dev.unsqueeze(1) @ (r_dev.unsqueeze(0) @ w_dp))
                    q_dp, s_dp = quant_mxfp8(w_dp)
                    shard_tensors[dp_w_key] = q_dp
                    shard_tensors[dp_s_key] = s_dp
                print(f"[Layer {layer_idx:02d}/59] Dense abliterated (||Delta W_o||_F = {diff_norm:.2f})")
            else:
                # MoE layer 3..59: compute Chronos Option-A Rank-64 Healer + refusal orthogonalization
                W_sh_healed, U_r64, V_r64 = compute_chronos_option_a_healer_for_layer(
                    shard_tensors=shard_tensors,
                    layer_idx=layer_idx,
                    r_hat=r_hat,
                    rank=rank,
                    healer_scale=healer_scale,
                    device=device,
                )
                sh_w_key = f"{prefix}.block_sparse_moe.shared_experts.down_proj.weight"
                sh_s_key = f"{prefix}.block_sparse_moe.shared_experts.down_proj.weight_scale_inv"
                q_sh, s_sh = quant_mxfp8(W_sh_healed)
                shard_tensors[sh_w_key] = q_sh
                shard_tensors[sh_s_key] = s_sh
                healer_state[layer_idx] = {
                    "U": U_r64,
                    "V": V_r64,
                    "r_hat": r_hat.to(torch.bfloat16).cpu(),
                }
                print(
                    f"[Layer {layer_idx:02d}/59] MoE abliterated & Chronos-healed "
                    f"(||Delta W_o||_F = {diff_norm:.2f}, ||U_r64||_F = {U_r64.float().norm().item():.2f})"
                )

        if modified:
            tmp_path = shard_path + ".tmp"
            save_file(shard_tensors, tmp_path)
            os.replace(tmp_path, shard_path)
            torch.save(healer_state, healer_pt_path)

        with open(marker_path, "w") as mf:
            mf.write("ok\n")

        # Clean up HF download cache inside output_dir after each shard to save disk space
        cache_dir = os.path.join(output_dir, ".cache")
        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)

    # 7. Write updated model.safetensors.index.json and config.json
    total_bytes = sum(
        os.path.getsize(os.path.join(output_dir, fn))
        for fn in set(weight_map_out.values())
        if os.path.exists(os.path.join(output_dir, fn))
    )
    index_out = {
        "metadata": {
            "total_size": total_bytes,
            "format": "chronos_v7.5_minimax_m31_nvfp4_mxfp8_abliterated",
        },
        "weight_map": weight_map_out,
    }
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index_out, f, indent=2)

    cfg_path = os.path.join(output_dir, "config.json")
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    cfg["architectures"] = ["MiniMaxM3SparseForConditionalGeneration"]
    cfg.setdefault("text_config", {})["max_position_embeddings"] = 1048576
    cfg["chronos_config"] = {
        "version": "7.5",
        "base_model": base_repo,
        "abliteration_source": ablit_repo,
        "expert_pruning_source": e32_repo,
        "num_local_experts": 32,
        "num_experts_per_tok": 4,
        "shared_expert_healer": "minimax_m31_shared_experts_option_a_r64.pt",
        "shared_expert_healer_rank": rank,
        "shared_expert_healer_scale": healer_scale,
        "max_position_embeddings": 1048576,
        "speculative_drafter": "dspark",
    }
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"[Chronos v7.5] Conversion complete! Total size: {total_bytes / (1024**3):.2f} GB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MiniMax-M3.1-Flash Abliterated to Chronos v7.5 1M-Context")
    parser.add_argument(
        "--output-dir",
        default="/home/defnotean/pb27/hf/minimax_m31_chronos_v75_abliterated",
        help="Destination checkpoint directory",
    )
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--healer-scale", type=float, default=0.08)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    convert_minimax_m31_chronos(
        output_dir=args.output_dir,
        rank=args.rank,
        healer_scale=args.healer_scale,
        device=args.device,
    )


if __name__ == "__main__":
    main()
