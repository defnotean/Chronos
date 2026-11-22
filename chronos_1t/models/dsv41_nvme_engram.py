"""
NVMe-Backed 4-Bit & HTTP-Range Live Engram Modules for Chronos DeepSeek-V4.1-Flash-1017.

Attaches DeepSeek-V4.1-Flash's full 196B Engram n-gram tables (Layers 1 and 14 across
`model-00047-of-00048.safetensors` and `model-00048-of-00048.safetensors`) onto an
already-resident 40-layer `GLOBAL_DSV41_MODEL` without reloading backbone weights or
breaking pre-captured CUDAGraphs.
"""
from __future__ import annotations

import builtins
from concurrent.futures import ThreadPoolExecutor
import ctypes
import importlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any
import urllib.request

import requests
import torch
from torch import nn
import torch.nn.functional as F

from chronos_1t.models.dsv41_chronos_moe import dequantize_block_fp8_32, ue8m0_to_float32

STOCK_REPO_DEFAULT = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
ENGRAM_ROW_DIM = 256
ENGRAM_SCALE_COLS = 8  # 256 / 32
ENGRAM_NVME_ROW_BYTES = 144  # 128 B packed INT4 + 16 B FP16 scales (8 x fp16)
ENGRAM_LAYER_SHARDS = {
    1: "model-00047-of-00048.safetensors",
    14: "model-00048-of-00048.safetensors",
}

_PREAD_LIB: ctypes.CDLL | None = None
_PREAD_LOCK = threading.Lock()


def _mem_avail_gib() -> float:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for ln in fh:
                if ln.startswith("MemAvailable:"):
                    return int(ln.split()[1]) / (1024.0 * 1024.0)
    except Exception:
        pass
    return 999.0


def _drop_fd_cache(fd: int, offset: int = 0, length: int = 0) -> None:
    if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
        try:
            os.posix_fadvise(fd, int(offset), int(length), os.POSIX_FADV_DONTNEED)
        except OSError:
            pass


def resolve_cdn_url(repo_id: str, filename: str, timeout: float = 30.0) -> str:
    """Resolve HuggingFace file URL to its current redirect/CDN target."""
    src_url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    req = urllib.request.Request(src_url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return str(resp.geturl())


def fetch_engram_shard_meta(repo_id: str, filename: str, timeout: float = 30.0) -> dict[str, Any]:
    """Read safetensors header via HTTP Range without downloading the 95 GiB shard."""
    src_url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    req = urllib.request.Request(src_url, headers={"Range": "bytes=0-16383"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        cdn_url = str(resp.geturl())
        cr = str(resp.headers.get("Content-Range", ""))
        head = resp.read()
    if len(head) < 8:
        raise RuntimeError(f"Short safetensors header read ({len(head)} bytes) for {filename}")
    header_len = int(struct.unpack("<Q", head[:8])[0])
    data_base = 8 + header_len
    if data_base > len(head):
        req2 = urllib.request.Request(cdn_url, headers={"Range": f"bytes=0-{data_base - 1}"})
        with urllib.request.urlopen(req2, timeout=timeout) as resp2:
            head = resp2.read()
    hdr = json.loads(head[8:data_base].decode("utf-8"))
    total_bytes = int(cr.split("/")[-1]) if "/" in cr else -1
    return {
        "repo_id": repo_id,
        "filename": filename,
        "cdn_url": cdn_url,
        "header_len": header_len,
        "data_base": data_base,
        "total_bytes": total_bytes,
        "header": hdr,
    }


class _RangeClient:
    """Thread-safe HTTP Range fetcher with keep-alive sessions and CDN URL refresh."""

    def __init__(self, repo_id: str, filename: str, initial_cdn_url: str | None = None):
        self.repo_id = repo_id
        self.filename = filename
        self._url_lock = threading.Lock()
        self._cdn_url = initial_cdn_url or resolve_cdn_url(repo_id, filename)
        self._url_ts = time.monotonic()
        self._local = threading.local()

    def _session(self) -> requests.Session:
        sess = getattr(self._local, "session", None)
        if sess is None:
            sess = requests.Session()
            self._local.session = sess
        return sess

    def _refresh_url(self, failed_url: str) -> str:
        with self._url_lock:
            now = time.monotonic()
            if self._cdn_url == failed_url or (now - self._url_ts) > 5.0:
                self._cdn_url = resolve_cdn_url(self.repo_id, self.filename)
                self._url_ts = time.monotonic()
            return self._cdn_url

    def read_range(self, start: int, end_incl: int, max_retries: int = 8, timeout: float = 60.0) -> bytes:
        expected = int(end_incl) - int(start) + 1
        if expected <= 0:
            return b""
        last_err: Exception | None = None
        for attempt in range(max_retries):
            with self._url_lock:
                url = self._cdn_url
            try:
                resp = self._session().get(
                    url,
                    headers={"Range": f"bytes={start}-{end_incl}"},
                    timeout=timeout,
                )
                if resp.status_code in (401, 403, 410):
                    url = self._refresh_url(url)
                    continue
                resp.raise_for_status()
                content = resp.content
                if len(content) != expected:
                    raise IOError(f"Short range read {len(content)} != {expected} at {start}..{end_incl}")
                return content
            except Exception as exc:
                last_err = exc
                if attempt >= 2:
                    try:
                        self._refresh_url(url)
                    except Exception:
                        pass
                time.sleep(min(8.0, 0.4 * (2**attempt)))
        raise RuntimeError(f"Range read failed for {self.filename} [{start}..{end_incl}]: {last_err}")


def _tensor_file_span(meta: dict[str, Any], key: str) -> tuple[int, int, str, list[int]]:
    hdr = meta["header"]
    if key not in hdr:
        alt = f"model.{key}" if f"model.{key}" in hdr else key
        if alt not in hdr:
            raise KeyError(f"Tensor '{key}' not in {meta['filename']}")
        key = alt
    entry = hdr[key]
    off0, off1 = entry["data_offsets"]
    base = int(meta["data_base"])
    return (base + int(off0), base + int(off1) - 1, str(entry["dtype"]), list(entry["shape"]))


def download_engram_projections(
    repo_id: str,
    filename: str,
    layer_id: int,
    out_pt: str,
) -> str:
    """Download tiny (~150 MiB) `q_weight`, `k_weight`, `wkv` for one Engram layer and save BF16."""
    out_path = Path(out_pt)
    if out_path.is_file() and out_path.stat().st_size > 100 * 1024 * 1024:
        return str(out_path)

    meta = fetch_engram_shard_meta(repo_id, filename)
    client = _RangeClient(repo_id, filename, meta["cdn_url"])
    prefix = f"layers.{layer_id}.engram"

    def _fetch_tensor(suffix: str, torch_dtype: torch.dtype) -> torch.Tensor:
        s_off, e_off, _, shape = _tensor_file_span(meta, f"{prefix}.{suffix}")
        raw = client.read_range(s_off, e_off)
        return torch.frombuffer(bytearray(raw), dtype=torch_dtype).clone().view(*shape)

    q_weight = _fetch_tensor("q_weight", torch.bfloat16)
    k_weight = _fetch_tensor("k_weight", torch.bfloat16)
    wkv_w_u8 = _fetch_tensor("wkv.weight", torch.uint8)
    wkv_s_u8 = _fetch_tensor("wkv.scale", torch.uint8)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wkv_bf16 = dequantize_block_fp8_32(wkv_w_u8, wkv_s_u8, dev, dtype=torch.bfloat16).cpu().contiguous()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = str(out_path) + ".tmp"
    torch.save(
        {
            "layer_id": int(layer_id),
            "q_weight": q_weight,
            "k_weight": k_weight,
            "wkv_bf16": wkv_bf16,
            "data_base": int(meta["data_base"]),
            "header": meta["header"],
        },
        tmp_path,
    )
    os.replace(tmp_path, out_path)
    return str(out_path)


def build_engram_args(
    runner_mod: Any,
    config_json: str = "/content/dsv41_standalone/inference/config.json",
    max_seq_len: int | None = None,
) -> SimpleNamespace:
    """Construct minimal args matching official `config.json` for `EngramLayout` & `NgramHashState`."""
    base_args = runner_mod.build_model_args()
    cfg: dict[str, Any] = {}
    if os.path.isfile(config_json):
        with open(config_json, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
    eff_seq = int(max_seq_len if max_seq_len is not None else getattr(base_args, "max_seq_len", 131072))
    return SimpleNamespace(
        max_batch_size=1,
        max_seq_len=eff_seq,
        dim=int(cfg.get("dim", getattr(base_args, "dim", 5120))),
        hc_mult=int(cfg.get("hc_mult", getattr(base_args, "hc_mult", 4))),
        norm_eps=float(cfg.get("norm_eps", getattr(base_args, "norm_eps", 1e-6))),
        engram_layer_ids=tuple(cfg.get("engram_layer_ids", (1, 14))),
        engram_vocab_size=int(cfg.get("engram_vocab_size", 16000000)),
        engram_num_embeddings=tuple(cfg.get("engram_num_embeddings", (384006168, 384016682))),
        engram_max_ngram_size=int(cfg.get("engram_max_ngram_size", 4)),
        engram_pad_id=int(cfg.get("engram_pad_id", 2)),
        engram_compressed_vocab_size=int(cfg.get("engram_compressed_vocab_size", 99092)),
        engram_n_heads=int(cfg.get("engram_n_heads", 8)),
        engram_head_dim=int(cfg.get("engram_head_dim", 256)),
    )


def _apply_engram_residual(
    x: torch.Tensor,
    emb_flat: torch.Tensor,
    wkv_bf16: torch.Tensor,
    qk_weight: torch.Tensor,
    *,
    dim: int = 5120,
    hc_mult: int = 4,
    eps: float = 1e-6,
    clamp_value: float = 1e-6,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute exact `Engram.forward` gated residual from fetched embeddings `[B, L, 6144]`."""
    kv = F.linear(emb_flat.to(torch.bfloat16), wkv_bf16)
    key, value = kv.split([hc_mult * dim, dim], dim=-1)
    key = key.float().unflatten(-1, (hc_mult, dim))
    h = x.float()
    rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps)
    dot = (h * qk_weight * key).sum(-1) * rstd * (dim**-0.5)
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(clamp_value).sqrt(), dot))
    if token_mask is not None:
        gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
    return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)


class HttpRangeEngram(nn.Module):
    """Exact stock FP8 HTTP-Range Engram module for causal proof before full NVMe stream."""

    def __init__(
        self,
        *,
        repo_id: str,
        filename: str,
        layer_id: int,
        layer_hash_index: int,
        proj_pt: str,
        dim: int = 5120,
        hc_mult: int = 4,
        eps: float = 1e-6,
        device: str | torch.device = "cuda",
        workers: int = 16,
    ):
        super().__init__()
        self.layer_id = int(layer_id)
        self.layer_hash_index = int(layer_hash_index)
        self.dim = int(dim)
        self.hc_mult = int(hc_mult)
        self.eps = float(eps)
        self.clamp_value = 1e-6
        self.device = torch.device(device)

        proj = torch.load(proj_pt, map_location="cpu", weights_only=True)
        wkv_bf16 = proj["wkv_bf16"].to(device=self.device, dtype=torch.bfloat16).contiguous()
        qk_weight = (proj["q_weight"].float() * proj["k_weight"].float()).to(self.device).contiguous()
        self.register_buffer("wkv_bf16", wkv_bf16, persistent=False)
        self.register_buffer("qk_weight", qk_weight, persistent=False)

        meta = fetch_engram_shard_meta(repo_id, filename)
        self._client = _RangeClient(repo_id, filename, meta["cdn_url"])
        self._w_base, _, _, w_shape = _tensor_file_span(meta, f"layers.{self.layer_id}.engram.embed.weight")
        self._s_base, _, _, _ = _tensor_file_span(meta, f"layers.{self.layer_id}.engram.embed.scale")
        self.num_embeddings = int(w_shape[0])
        self._row_cache: dict[int, torch.Tensor] = {}
        self._pool = ThreadPoolExecutor(max_workers=int(workers))

    def close(self) -> None:
        self._pool.shutdown(wait=False)
        self._row_cache.clear()

    def _fetch_one_row(self, rid: int) -> tuple[int, torch.Tensor]:
        w_off = self._w_base + rid * ENGRAM_ROW_DIM
        s_off = self._s_base + rid * ENGRAM_SCALE_COLS
        w_raw = self._client.read_range(w_off, w_off + ENGRAM_ROW_DIM - 1)
        s_raw = self._client.read_range(s_off, s_off + ENGRAM_SCALE_COLS - 1)
        w_f32 = (
            torch.frombuffer(bytearray(w_raw), dtype=torch.uint8)
            .view(torch.float8_e4m3fn)
            .to(torch.float32)
            .view(8, 32)
        )
        s_u8 = torch.frombuffer(bytearray(s_raw), dtype=torch.uint8)
        s_f32 = ue8m0_to_float32(s_u8, torch.device("cpu")).view(8, 1)
        return rid, (w_f32 * s_f32).reshape(ENGRAM_ROW_DIM).to(torch.bfloat16)

    @torch.inference_mode()
    def forward(
        self,
        x: torch.Tensor,
        hash_ids: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        flat_ids = hash_ids.reshape(-1).to("cpu")
        uniq_ids, inv = torch.unique(flat_ids, return_inverse=True)
        uniq_list = uniq_ids.tolist()
        missing = [int(rid) for rid in uniq_list if int(rid) not in self._row_cache]
        if missing:
            for rid, row_bf16 in self._pool.map(self._fetch_one_row, missing):
                self._row_cache[rid] = row_bf16
        gathered_cpu = torch.stack([self._row_cache[int(rid)] for rid in uniq_list], dim=0)
        gathered_gpu = gathered_cpu.to(self.device, dtype=torch.bfloat16)
        emb = gathered_gpu[inv.to(self.device)].view(*hash_ids.shape, ENGRAM_ROW_DIM).flatten(-2)
        return _apply_engram_residual(
            x,
            emb,
            self.wkv_bf16,
            self.qk_weight,
            dim=self.dim,
            hc_mult=self.hc_mult,
            eps=self.eps,
            clamp_value=self.clamp_value,
            token_mask=token_mask,
        )


def _quantize_fp8_rows_to_nvme_int4_gpu(
    w_u8_cpu: torch.Tensor,
    s_u8_cpu: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Dequantize `[R, 256]` F8_E4M3 + `[R, 8]` F8_E8M0 on GPU and pack `[R, 144]` INT4+FP16."""
    n_rows = int(w_u8_cpu.size(0))
    w_gpu = w_u8_cpu.to(device=device, non_blocking=False)
    s_gpu = s_u8_cpu.to(device=device, non_blocking=False)
    x = dequantize_block_fp8_32(w_gpu, s_gpu, device, dtype=torch.float32)  # [R, 256]
    x_blk = x.view(n_rows, ENGRAM_SCALE_COLS, 32)
    max_abs = x_blk.abs().amax(dim=-1, keepdim=True)
    sc = (max_abs / 7.0).clamp_min(1e-8).to(torch.float16)
    sc = torch.where(max_abs == 0, torch.zeros_like(sc), sc)
    sc_f32 = sc.float().clamp_min(1e-12)
    q = torch.round(x_blk / sc_f32).clamp(-7, 7).to(torch.int32)
    q = torch.where(max_abs == 0, torch.zeros_like(q), q)
    q_u8 = (q + 8).to(torch.uint8).view(n_rows, 128, 2)
    packed = q_u8[..., 0] | (q_u8[..., 1] << 4)  # [R, 128]
    sc_bytes = sc.view(n_rows, ENGRAM_SCALE_COLS).contiguous().view(torch.uint8)  # [R, 16]
    out_gpu = torch.cat([packed, sc_bytes], dim=1).contiguous()
    return out_gpu.cpu(), out_gpu[: min(4096, n_rows)].clone()


def dequantize_nvme_int4_rows_gpu(raw_gpu: torch.Tensor) -> torch.Tensor:
    """Dequantize `[U, 144]` packed INT4+FP16 rows on GPU -> `[U, 256]` BF16."""
    u = int(raw_gpu.size(0))
    packed = raw_gpu[:, :128]
    lo = (packed & 0x0F).to(torch.int16) - 8
    hi = (packed >> 4).to(torch.int16) - 8
    q = torch.stack([lo, hi], dim=-1).view(u, ENGRAM_SCALE_COLS, 32).float()
    sc = raw_gpu[:, 128:].contiguous().view(torch.float16).view(u, ENGRAM_SCALE_COLS, 1).float()
    return (q * sc).view(u, ENGRAM_ROW_DIM).to(torch.bfloat16)


def stream_convert_engram_to_nvme_int4(
    repo_id: str,
    filename: str,
    layer_id: int,
    out_bin: str,
    *,
    proj_pt: str | None = None,
    chunk_rows: int = 1048576,
    workers: int = 8,
    min_cos_chunk0: float = 0.990,
) -> dict[str, Any]:
    """Stream `[N, 256]` FP8 Engram table over HTTP Range -> packed 144 B/row INT4 on NVMe SSD."""
    if proj_pt:
        download_engram_projections(repo_id, filename, layer_id, proj_pt)

    meta = fetch_engram_shard_meta(repo_id, filename)
    w_base, _, _, w_shape = _tensor_file_span(meta, f"layers.{layer_id}.engram.embed.weight")
    s_base, _, _, s_shape = _tensor_file_span(meta, f"layers.{layer_id}.engram.embed.scale")
    total_rows = int(w_shape[0])
    assert int(w_shape[1]) == ENGRAM_ROW_DIM and int(s_shape[0]) == total_rows and int(s_shape[1]) == ENGRAM_SCALE_COLS

    out_path = Path(out_bin)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(f"{out_bin}.meta.json")

    done_chunks: set[int] = set()
    cos0: float | None = None
    if ckpt_path.is_file() and out_path.is_file():
        prev = json.loads(ckpt_path.read_text(encoding="utf-8"))
        if int(prev.get("total_rows", -1)) == total_rows and int(prev.get("chunk_rows", -1)) == int(chunk_rows):
            done_chunks = {int(c) for c in prev.get("done_chunks", [])}
            cos0 = prev.get("cos_chunk0")

    n_chunks = (total_rows + chunk_rows - 1) // chunk_rows
    if len(done_chunks) == n_chunks and out_path.is_file() and out_path.stat().st_size == total_rows * ENGRAM_NVME_ROW_BYTES:
        return {
            "layer_id": int(layer_id),
            "out_bin": str(out_path),
            "total_rows": total_rows,
            "size_gib": round(out_path.stat().st_size / 1024**3, 3),
            "cos_chunk0": cos0,
            "resumed_complete": True,
        }

    client = _RangeClient(repo_id, filename, meta["cdn_url"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fd = os.open(str(out_path), os.O_CREAT | os.O_RDWR, 0o644)
    os.ftruncate(fd, total_rows * ENGRAM_NVME_ROW_BYTES)

    def _download_chunk(c_idx: int) -> tuple[int, int, torch.Tensor, torch.Tensor]:
        r0 = c_idx * chunk_rows
        r1 = min(total_rows, r0 + chunk_rows)
        n_r = r1 - r0
        w_buf = bytearray(n_r * ENGRAM_ROW_DIM)
        s_buf = bytearray(n_r * ENGRAM_SCALE_COLS)
        sub_step = max(65536, (n_r + workers - 1) // workers)
        spans = []
        for sr in range(0, n_r, sub_step):
            er = min(n_r, sr + sub_step)
            spans.append((sr, er))

        def _dl_sub(sp: tuple[int, int]) -> None:
            sr, er = sp
            gr0 = r0 + sr
            gn = er - sr
            wb = client.read_range(
                w_base + gr0 * ENGRAM_ROW_DIM,
                w_base + (gr0 + gn) * ENGRAM_ROW_DIM - 1,
            )
            sb = client.read_range(
                s_base + gr0 * ENGRAM_SCALE_COLS,
                s_base + (gr0 + gn) * ENGRAM_SCALE_COLS - 1,
            )
            w_buf[sr * ENGRAM_ROW_DIM : er * ENGRAM_ROW_DIM] = wb
            s_buf[sr * ENGRAM_SCALE_COLS : er * ENGRAM_SCALE_COLS] = sb

        list(dl_pool.map(_dl_sub, spans))
        w_t = torch.frombuffer(w_buf, dtype=torch.uint8).view(n_r, ENGRAM_ROW_DIM)
        s_t = torch.frombuffer(s_buf, dtype=torch.uint8).view(n_r, ENGRAM_SCALE_COLS)
        return (c_idx, r0, w_t, s_t)

    pending = [i for i in range(n_chunks) if i not in done_chunks]
    t_all = time.perf_counter()
    bytes_written_session = 0
    dl_pool = ThreadPoolExecutor(max_workers=int(workers))
    pf_pool = ThreadPoolExecutor(max_workers=1)
    try:
        fut_cur = pf_pool.submit(_download_chunk, pending[0]) if pending else None
        for idx_pos, c_idx in enumerate(pending):
            _, r0, w_t, s_t = fut_cur.result()  # type: ignore[union-attr]
            if idx_pos + 1 < len(pending):
                fut_cur = pf_pool.submit(_download_chunk, pending[idx_pos + 1])

            packed_cpu, sample_gpu = _quantize_fp8_rows_to_nvme_int4_gpu(w_t, s_t, dev)
            if c_idx == 0 and cos0 is None and sample_gpu is not None:
                n_chk = int(sample_gpu.size(0))
                ref = dequantize_block_fp8_32(
                    w_t[:n_chk].to(dev), s_t[:n_chk].to(dev), dev, dtype=torch.float32
                )
                rec = dequantize_nvme_int4_rows_gpu(sample_gpu).float()
                cos0 = float(F.cosine_similarity(ref.flatten(), rec.flatten(), dim=0).item())
                print(f"[Engram L{layer_id}] Chunk 0 INT4 vs FP8 cosine={cos0:.6f}", flush=True)
                if cos0 < min_cos_chunk0:
                    raise RuntimeError(f"Engram L{layer_id} chunk 0 cosine {cos0:.6f} < {min_cos_chunk0}")

            raw_bytes = memoryview(packed_cpu.numpy()).cast("B")
            off_b = r0 * ENGRAM_NVME_ROW_BYTES
            written = 0
            while written < len(raw_bytes):
                written += os.pwrite(fd, raw_bytes[written:], off_b + written)
            _drop_fd_cache(fd, off_b, len(raw_bytes))
            bytes_written_session += len(raw_bytes)
            done_chunks.add(c_idx)

            if len(done_chunks) % 16 == 0 or idx_pos == len(pending) - 1:
                tmp_c = str(ckpt_path) + ".tmp"
                Path(tmp_c).write_text(
                    json.dumps(
                        {
                            "layer_id": int(layer_id),
                            "total_rows": total_rows,
                            "chunk_rows": int(chunk_rows),
                            "n_chunks": n_chunks,
                            "done_chunks": sorted(done_chunks),
                            "cos_chunk0": cos0,
                        }
                    ),
                    encoding="utf-8",
                )
                os.replace(tmp_c, ckpt_path)
                elapsed = max(1e-3, time.perf_counter() - t_all)
                src_gib = len(done_chunks) * chunk_rows * (ENGRAM_ROW_DIM + ENGRAM_SCALE_COLS) / (1024**3)
                rate = bytes_written_session / (1024**2) / elapsed
                import shutil

                disk_free = shutil.disk_usage(str(out_path.parent)).free / (1024**3)
                print(
                    f"[Engram L{layer_id}] {len(done_chunks):3d}/{n_chunks} chunks "
                    f"(~{min(src_gib, 94.56):5.1f} GiB src) | write={rate:5.1f} MiB/s | "
                    f"MemAvail={_mem_avail_gib():5.2f} GiB | DiskFree={disk_free:5.1f} GiB",
                    flush=True,
                )
    finally:
        pf_pool.shutdown(wait=False)
        dl_pool.shutdown(wait=False)
        os.fsync(fd)
        _drop_fd_cache(fd, 0, 0)
        os.close(fd)

    return {
        "layer_id": int(layer_id),
        "out_bin": str(out_path),
        "total_rows": total_rows,
        "size_gib": round(out_path.stat().st_size / 1024**3, 3),
        "cos_chunk0": cos0,
        "elapsed_s": round(time.perf_counter() - t_all, 2),
    }


def _compile_pread_helper() -> ctypes.CDLL | None:
    global _PREAD_LIB
    with _PREAD_LOCK:
        if _PREAD_LIB is not None:
            return _PREAD_LIB
        so_p = Path("/tmp/chronos_engram_pread.so")
        c_p = Path("/tmp/chronos_engram_pread.c")
        if not so_p.is_file():
            c_src = r"""
#define _GNU_SOURCE
#include <stdint.h>
#include <unistd.h>
#include <pthread.h>

typedef struct {
    int fd;
    const int64_t *row_ids;
    int start_idx;
    int end_idx;
    int row_bytes;
    uint8_t *out_buf;
    int err;
} pread_task_t;

static void *worker_fn(void *arg) {
    pread_task_t *t = (pread_task_t *)arg;
    int rb = t->row_bytes;
    for (int i = t->start_idx; i < t->end_idx; i++) {
        off_t off = (off_t)t->row_ids[i] * (off_t)rb;
        uint8_t *dst = t->out_buf + (size_t)i * (size_t)rb;
        ssize_t got = 0;
        while (got < rb) {
            ssize_t r = pread(t->fd, dst + got, (size_t)(rb - got), off + got);
            if (r <= 0) {
                t->err = -1;
                return NULL;
            }
            got += r;
        }
    }
    return NULL;
}

int pread_rows_mt(int fd, const int64_t *row_ids, int n_rows, int row_bytes, uint8_t *out_buf, int n_threads) {
    if (n_rows <= 0) return 0;
    if (n_threads <= 1 || n_rows < 4) {
        pread_task_t t = {fd, row_ids, 0, n_rows, row_bytes, out_buf, 0};
        worker_fn(&t);
        return t.err;
    }
    if (n_threads > 32) n_threads = 32;
    if (n_threads > n_rows) n_threads = n_rows;
    pthread_t th[32];
    pread_task_t args[32];
    int step = (n_rows + n_threads - 1) / n_threads;
    int used = 0;
    for (int i = 0; i < n_threads; i++) {
        int s = i * step;
        if (s >= n_rows) break;
        int e = s + step;
        if (e > n_rows) e = n_rows;
        args[used] = (pread_task_t){fd, row_ids, s, e, row_bytes, out_buf, 0};
        if (pthread_create(&th[used], NULL, worker_fn, &args[used]) != 0) {
            worker_fn(&args[used]);
        } else {
            used++;
        }
    }
    int err = 0;
    for (int i = 0; i < used; i++) {
        pthread_join(th[i], NULL);
        if (args[i].err != 0) err = -1;
    }
    return err;
}
"""
            c_p.write_text(c_src, encoding="utf-8")
            cp = subprocess.run(
                ["gcc", "-O3", "-shared", "-fPIC", "-pthread", str(c_p), "-o", str(so_p)],
                capture_output=True,
                text=True,
                check=False,
            )
            if cp.returncode != 0:
                return None
        lib = ctypes.CDLL(str(so_p))
        lib.pread_rows_mt.argtypes = [
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        lib.pread_rows_mt.restype = ctypes.c_int
        _PREAD_LIB = lib
        return lib


class ChronosNVMeEngram(nn.Module):
    """Low-RAM NVMe-backed 4-bit Engram lookup module for Layers 1 and 14."""

    def __init__(
        self,
        *,
        layer_id: int,
        layer_hash_index: int,
        bin_path: str,
        proj_pt: str,
        dim: int = 5120,
        hc_mult: int = 4,
        eps: float = 1e-6,
        device: str | torch.device = "cuda",
        n_threads: int = 12,
    ):
        super().__init__()
        self.layer_id = int(layer_id)
        self.layer_hash_index = int(layer_hash_index)
        self.bin_path = str(bin_path)
        self.dim = int(dim)
        self.hc_mult = int(hc_mult)
        self.eps = float(eps)
        self.clamp_value = 1e-6
        self.device = torch.device(device)
        self.n_threads = int(n_threads)

        proj = torch.load(proj_pt, map_location="cpu", weights_only=True)
        self.register_buffer(
            "wkv_bf16",
            proj["wkv_bf16"].to(device=self.device, dtype=torch.bfloat16).contiguous(),
            persistent=False,
        )
        self.register_buffer(
            "qk_weight",
            (proj["q_weight"].float() * proj["k_weight"].float()).to(self.device).contiguous(),
            persistent=False,
        )

        file_size = os.path.getsize(self.bin_path)
        if file_size % ENGRAM_NVME_ROW_BYTES != 0:
            raise ValueError(f"{self.bin_path} size {file_size} not divisible by {ENGRAM_NVME_ROW_BYTES}")
        self.num_embeddings = file_size // ENGRAM_NVME_ROW_BYTES
        self.fd = os.open(self.bin_path, os.O_RDONLY)
        self._pread_lib = _compile_pread_helper()
        self._stage_rows = 4096
        self._stage_cpu = torch.empty(
            (self._stage_rows, ENGRAM_NVME_ROW_BYTES),
            dtype=torch.uint8,
            device="cpu",
            pin_memory=torch.cuda.is_available(),
        )
        self._call_count = 0

    def close(self) -> None:
        if getattr(self, "fd", -1) >= 0:
            try:
                _drop_fd_cache(self.fd, 0, 0)
                os.close(self.fd)
            except OSError:
                pass
            self.fd = -1

    def _read_rows_to_gpu(self, row_ids_cpu: torch.Tensor) -> torch.Tensor:
        n_u = int(row_ids_cpu.numel())
        if n_u == 0:
            return torch.empty((0, ENGRAM_NVME_ROW_BYTES), dtype=torch.uint8, device=self.device)
        ids_i64 = row_ids_cpu.to(dtype=torch.int64, device="cpu", copy=False).contiguous()
        min_id = int(ids_i64.min().item())
        max_id = int(ids_i64.max().item())
        assert 0 <= min_id and max_id < self.num_embeddings, (
            f"Engram row ID out of bounds: [{min_id}, {max_id}] vs num_embeddings={self.num_embeddings}"
        )
        if n_u <= self._stage_rows:
            stage = self._stage_cpu[:n_u]
        else:
            stage = torch.empty((n_u, ENGRAM_NVME_ROW_BYTES), dtype=torch.uint8, device="cpu")

        if self._pread_lib is not None:
            rc = self._pread_lib.pread_rows_mt(
                int(self.fd),
                ctypes.c_void_p(ids_i64.data_ptr()),
                int(n_u),
                ENGRAM_NVME_ROW_BYTES,
                ctypes.c_void_p(stage.data_ptr()),
                self.n_threads,
            )
            if rc != 0:
                raise IOError(f"pread_rows_mt failed on {self.bin_path} (n_rows={n_u})")
        else:
            mv = memoryview(stage.numpy()).cast("B")
            for i, rid in enumerate(ids_i64.tolist()):
                off = int(rid) * ENGRAM_NVME_ROW_BYTES
                data = os.pread(self.fd, ENGRAM_NVME_ROW_BYTES, off)
                mv[i * ENGRAM_NVME_ROW_BYTES : (i + 1) * ENGRAM_NVME_ROW_BYTES] = data
        return stage.to(self.device, non_blocking=False)

    @torch.inference_mode()
    def forward(
        self,
        x: torch.Tensor,
        hash_ids: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._call_count += 1
        if (self._call_count % 500) == 0 and _mem_avail_gib() < 10.0:
            _drop_fd_cache(self.fd, 0, 0)

        flat_ids = hash_ids.reshape(-1)
        if flat_ids.numel() <= 24:
            # Decode (L=1, 24 hash lookups): skip GPU unique/sort synchronization
            raw_gpu = self._read_rows_to_gpu(flat_ids.cpu())
            emb_rows = dequantize_nvme_int4_rows_gpu(raw_gpu)
            emb = emb_rows.view(*hash_ids.shape, ENGRAM_ROW_DIM).flatten(-2)
        else:
            uniq_ids, inv = torch.unique(flat_ids, sorted=True, return_inverse=True)
            raw_gpu = self._read_rows_to_gpu(uniq_ids.cpu())
            emb_rows = dequantize_nvme_int4_rows_gpu(raw_gpu)
            emb = emb_rows[inv].view(*hash_ids.shape, ENGRAM_ROW_DIM).flatten(-2)

        return _apply_engram_residual(
            x,
            emb,
            self.wkv_bf16,
            self.qk_weight,
            dim=self.dim,
            hc_mult=self.hc_mult,
            eps=self.eps,
            clamp_value=self.clamp_value,
            token_mask=token_mask,
        )


def attach_engram_live(
    model: Any,
    tokenizer: Any,
    dsv41_model_mod: Any,
    runner_mod: Any,
    layer1_mod: nn.Module,
    layer14_mod: nn.Module,
    *,
    config_json: str = "/content/dsv41_standalone/inference/config.json",
    reload_modules: bool = False,
) -> dict[str, Any]:
    """Attach Engram modules live to resident `GLOBAL_DSV41_MODEL` without reloading backbone weights."""
    for p in ("/content/dsv41_standalone/inference", "/content/dsv41_standalone/encoding"):
        if os.path.exists(p) and p not in sys.path:
            sys.path.insert(0, p)
    import engram as engram_mod
    import chronos_1t.models.dsv41_longctx_spec_engine as spec_eng
    import chronos_1t.models.dsv41_triton_backbone as triton_bb

    if reload_modules:
        importlib.reload(triton_bb)
        importlib.reload(runner_mod)
        importlib.reload(spec_eng)
    dsv41_model_mod.Block.forward = triton_bb.fast_block_forward

    max_seq_len = int(model.layers[0].attn.freqs_cis.size(0))
    e_args = build_engram_args(runner_mod, config_json=config_json, max_seq_len=max_seq_len)
    dev = next(model.parameters()).device

    layout = engram_mod.EngramLayout.from_args(e_args)
    hash_state = engram_mod.NgramHashState(e_args, layout, tokenizer).to(dev)
    with torch.inference_mode(False):
        model.engram_layout = layout
        model.engram_hash = hash_state
        for lid in (1, 14):
            old_eg = getattr(model.layers[lid], "engram", None)
            if old_eg is not None and hasattr(old_eg, "close"):
                try:
                    old_eg.close()
                except Exception:
                    pass
        model.layers[1].engram = layer1_mod
        model.layers[14].engram = layer14_mod

    runner_mod.reset_model_caches(model)
    cg_ready = sum(
        1
        for lyr in model.layers
        if getattr(lyr, "_cg_full_block", None) is not None or getattr(lyr, "_cg_subblock2", None) is not None
    )
    return {
        "attached": True,
        "max_seq_len": max_seq_len,
        "engram_layers": [1, 14],
        "cudagraphs_preserved": cg_ready,
    }
